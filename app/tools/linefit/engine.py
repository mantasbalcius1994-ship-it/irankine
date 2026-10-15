#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linefit.py — linijų (linestring) optimizavimas.

Planas:    tiesės + lankai, tangentiškai sujungti (G1), su nuokrypio tolerancija.
Profilis:  VPI + parabolinės vertikalios kreivės, su aukščio tolerancija.

Įvestis:   DXF (LWPOLYLINE / POLYLINE 2D/3D, lankai per bulge) arba CSV (x,y,z[,name]).
Išvestis:  <out>.xml  - LandXML 1.2 alignment'ai (planas + profilis), importuojami į TBC
           <out>.dxf  - {L}-AL (tiesės/lankai), {L}-LS (3D polilinija), {L}-ERR (taškai už tol.)
           <out>_report.txt
           <out>.png  - (su --plot) planas + nuokrypiai

Pvz.:
  python linefit.py bortai.dxf --layer BORTAS --tol-line 0.01 --tol-arc 0.015 --v-tol 0.005 --plot
"""
import argparse
import datetime as dt
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

try:                                    # kaip paketo modulis
    from ...geo import io_csv, io_dxf, io_landxml, polyline
    from ...geo.alignment import Alignment, HElem, Profile
except ImportError:                     # kaip skriptas: python engine.py ...
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.geo import io_csv, io_dxf, io_landxml, polyline
    from app.geo.alignment import Alignment, HElem, Profile

TWO_PI = 2.0 * math.pi
K_EPS = 1e-12


@dataclass
class Settings:
    # planas
    tol_line: float = 0.02      # m, max nuokrypis tiesėse
    tol_arc: float = 0.02       # m, max nuokrypis lankuose
    r_min: float = 5.0          # m
    r_max: float = 10000.0      # m, didesnis R -> tiesė
    min_len: float = 0.5        # m, trumpesni elementai bandomi pašalinti
    split_angle: float = 60.0   # °, aštresniuose plano kampuose linija skaidoma
    simplify: bool = True
    # profilis
    vertical: bool = True
    v_tol: float = 0.01         # m, max aukščio nuokrypis
    v_min_dg: float = 0.10      # %, min. nuolydžių skirtumas VPI
    v_max_len: float = 300.0    # m, max vertikalios kreivės ilgis
    v_sharp_dg: float = 0.0     # %, VPI su didesniu skirtumu - be kreivės (0 = išjungta)
    # bendra
    max_refine: int = 6
    max_nfev: int = 60          # optimizatoriaus iteracijų riba
    report_tol: float = 0.0     # m, 0 -> naudoti tol_*


# =====================================================================
#  Plano geometrija
# =====================================================================
class Elem:
    """Pastovaus kreivumo elementas: tiesė (k=0) arba lankas (k=±1/R, + = į kairę)."""
    __slots__ = ("p", "th", "k", "L", "s0")

    def __init__(self, p, th, k, L, s0):
        self.p = np.asarray(p, float)
        self.th, self.k, self.L, self.s0 = float(th), float(k), float(L), float(s0)

    @property
    def is_arc(self):
        return abs(self.k) > K_EPS

    @property
    def R(self):
        return 1.0 / abs(self.k) if self.is_arc else math.inf

    def xy(self, t):
        return elem_xy(self.p, self.th, self.k, t)

    def end(self):
        return self.xy([self.L])[0]

    def th_end(self):
        return self.th + self.k * self.L

    def center(self):
        return self.p + (1.0 / self.k) * np.array([-math.sin(self.th), math.cos(self.th)])


def elem_xy(p, th, k, t):
    t = np.atleast_1d(np.asarray(t, float))
    if abs(k) < K_EPS:
        return np.column_stack([p[0] + t * math.cos(th), p[1] + t * math.sin(th)])
    return np.column_stack([p[0] + (np.sin(th + k * t) - math.sin(th)) / k,
                            p[1] - (np.cos(th + k * t) - math.cos(th)) / k])


def pack(start, types, Ls, kfull):
    ks = [k for t, k in zip(types, kfull) if t == "A"]
    return np.concatenate([np.asarray(start, float), np.asarray(Ls, float), np.asarray(ks, float)])


def unpack(x, types):
    m = len(types)
    start = x[:3]
    Ls = list(x[3:3 + m])
    ks = list(x[3 + m:])
    kfull, ai = [], 0
    for t in types:
        if t == "A":
            kfull.append(ks[ai]); ai += 1
        else:
            kfull.append(0.0)
    return start, Ls, kfull


def build_chain(x, types):
    start, Ls, kfull = unpack(x, types)
    p = np.array(start[:2], float)
    th = float(start[2])
    out, s = [], 0.0
    for L, k in zip(Ls, kfull):
        e = Elem(p, th, k, L, s)
        out.append(e)
        p, th, s = e.end(), e.th_end(), s + L
    return out


def project(Q, elems):
    """Taškų projekcija į grandinę. Grąžina (offset [+ kairėje], stotis, elemento idx)."""
    m, n = len(elems), len(Q)
    D = np.empty((m, n)); O = np.empty((m, n)); S = np.empty((m, n))
    for ei, e in enumerate(elems):
        v = Q - e.p
        if not e.is_arc:
            c, s_ = math.cos(e.th), math.sin(e.th)
            t = v[:, 0] * c + v[:, 1] * s_
            off = c * v[:, 1] - s_ * v[:, 0]
        else:
            R, sg = e.R, math.copysign(1.0, e.k)
            cen = e.center()
            w = Q - cen
            r = np.hypot(w[:, 0], w[:, 1])
            a0 = math.atan2(e.p[1] - cen[1], e.p[0] - cen[0])
            a = np.arctan2(w[:, 1], w[:, 0])
            t = R * np.mod(sg * (a - a0), TWO_PI)
            circ = TWO_PI * R
            t = np.where(t > e.L + 0.5 * (circ - e.L), t - circ, t)   # taškai "prieš" pradžią
            off = sg * (R - r)
        inr = (t >= 0.0) & (t <= e.L)
        pe = e.end()
        d0 = np.hypot(v[:, 0], v[:, 1])
        d1 = np.hypot(Q[:, 0] - pe[0], Q[:, 1] - pe[1])
        dout = np.minimum(d0, d1)
        sgn = np.where(off >= 0, 1.0, -1.0)
        D[ei] = np.where(inr, np.abs(off), dout)
        O[ei] = np.where(inr, off, sgn * dout)
        S[ei] = e.s0 + np.where(inr, t, np.where(d0 <= d1, 0.0, e.L))
    idx = np.argmin(D, axis=0)
    ar = np.arange(n)
    return O[idx, ar], S[idx, ar], idx


# ---------- pirminis primityvų pritaikymas ----------
def fit_line(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    d = vt[0]
    if np.dot(P[-1] - P[0], d) < 0:
        d = -d
    nrm = np.array([-d[1], d[0]])
    return np.abs((P - c) @ nrm).max(), c, d


def fit_circle(P, iters=8):
    c0 = P.mean(0)
    Q = P - c0
    A = np.column_stack([2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))])
    b = (Q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    R2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(R2) or R2 <= 0:
        return None
    R = math.sqrt(R2)
    for _ in range(iters):          # Gauss-Newton geometrinis patikslinimas
        dx, dy = Q[:, 0] - cx, Q[:, 1] - cy
        r = np.hypot(dx, dy)
        if np.any(r < 1e-12):
            break
        J = np.column_stack([-dx / r, -dy / r, -np.ones_like(r)])
        delta, *_ = np.linalg.lstsq(J, -(r - R), rcond=None)
        cx, cy, R = cx + delta[0], cy + delta[1], R + delta[2]
        if np.abs(delta).max() < 1e-10:
            break
    R = abs(R)
    dev = np.abs(np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) - R).max()
    return dev, np.array([cx, cy]) + c0, R


def swept_angle(P, cen):
    a = P[:-1] - cen
    b = P[1:] - cen
    cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dt_ = (a * b).sum(1)
    return float(np.arctan2(cr, dt_).sum())


def gallop(jmin, ok, n):
    """Didžiausias j>=jmin (jmin jau tinka), kuriam ok(j)."""
    lo, step = jmin, 1
    while True:
        hi = min(lo + step, n - 1)
        if hi == lo:
            return lo
        if ok(hi):
            lo, step = hi, step * 2
        else:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def segment(P, tolL, tolA, st):
    n, i, segs = len(P), 0, []

    def arc_ok(Q):
        r = fit_circle(Q)
        if r is None:
            return False
        dev, cen, R = r
        if not (dev <= tolA and st.r_min <= R <= st.r_max):
            return False
        return abs(swept_angle(Q, cen)) < 1.9 * math.pi

    while i < n - 1:
        jl = gallop(i + 1, lambda j: fit_line(P[i:j + 1])[0] <= tolL, n)
        ja = None
        if i + 2 <= n - 1 and arc_ok(P[i:i + 3]):
            ja = gallop(i + 2, lambda j: arc_ok(P[i:j + 1]), n)
        if ja is not None and ja > jl:
            segs.append(("A", i, ja)); i = ja
        else:
            segs.append(("L", i, jl)); i = jl
    return segs


def init_params(P, segs):
    types, Ls, kfull, start = [], [], [], None
    for si, (t, i, j) in enumerate(segs):
        Q = P[i:j + 1]
        if t == "L":
            _, c, d = fit_line(Q)
            a = c + np.dot(Q[0] - c, d) * d
            b = c + np.dot(Q[-1] - c, d) * d
            L, th, k, sp = max(np.linalg.norm(b - a), 1e-3), math.atan2(d[1], d[0]), 0.0, a
        else:
            _, cen, R = fit_circle(Q)
            ang = swept_angle(Q, cen)
            sg = 1.0 if ang >= 0 else -1.0
            r = (Q[0] - cen) / np.linalg.norm(Q[0] - cen)
            sp = cen + R * r
            tang = sg * np.array([-r[1], r[0]])
            th, k, L = math.atan2(tang[1], tang[0]), sg / R, max(R * abs(ang), 1e-3)
        types.append(t); Ls.append(L); kfull.append(k)
        if si == 0:
            start = [sp[0], sp[1], th]
    return types, pack(start, types, Ls, kfull)


# ---------- globalus optimizavimas ----------
def h_bounds(types, x, st):
    m = len(types)
    lo = [-np.inf] * 3 + [1e-3] * m
    hi = [np.inf] * 3 + [np.inf] * m
    _, _, kfull = unpack(x, types)
    for t, k in zip(types, kfull):
        if t == "A":
            if k >= 0:
                lo.append(1.0 / st.r_max); hi.append(1.0 / st.r_min)
            else:
                lo.append(-1.0 / st.r_min); hi.append(-1.0 / st.r_max)
    lo, hi = np.array(lo), np.array(hi)
    span = np.where(np.isfinite(hi - lo), hi - lo, 1.0)
    x = np.clip(x, lo + 1e-9 * span, hi - 1e-9 * span)
    return lo, hi, x


def h_residual_tol(types, idx, st):
    isA = np.array([t == "A" for t in types])
    return np.where(isA[idx], st.tol_arc, st.tol_line)


def optimize_h(P, types, x, st):
    lo, hi, x = h_bounds(types, x, st)

    def fun(xx):
        el = build_chain(xx, types)
        off, _, idx = project(P, el)
        return off / h_residual_tol(types, idx, st)

    r = least_squares(fun, x, bounds=(lo, hi), x_scale="jac", method="trf",
                      max_nfev=st.max_nfev, ftol=1e-6, xtol=1e-8)
    return r.x


def h_check(P, types, x, st):
    el = build_chain(x, types)
    off, sta, idx = project(P, el)
    ratio = np.abs(off) / h_residual_tol(types, idx, st)
    return bool(ratio.max() <= 1.0 + 1e-6), float(ratio.max())


def merge_lines(types, Ls, kfull):
    T, L2, K2 = [], [], []
    for t, L, k in zip(types, Ls, kfull):
        if T and t == "L" and T[-1] == "L":
            L2[-1] += L
        else:
            T.append(t); L2.append(L); K2.append(k)
    return T, L2, K2


def op_remove(types, x, e):
    if len(types) < 2:
        return None
    el = build_chain(x, types)
    start, Ls, kfull = unpack(x, types)
    start = list(start)
    T = list(types)
    Le = Ls[e]
    if e == 0:
        start[2] = el[1].th
        Ls[1] += Le
    elif e == len(T) - 1:
        Ls[e - 1] += Le
    else:
        Ls[e - 1] += Le / 2; Ls[e + 1] += Le / 2
    del T[e], Ls[e], kfull[e]
    T, Ls, kfull = merge_lines(T, Ls, kfull)
    return T, pack(start, T, Ls, kfull)


def op_to_line(types, x, e):
    start, Ls, kfull = unpack(x, types)
    T = list(types)
    T[e], kfull[e] = "L", 0.0
    T, Ls, kfull = merge_lines(T, Ls, kfull)
    return T, pack(start, T, Ls, kfull)


def simplify_h(P, types, x, st, log):
    while len(types) > 1:
        el = build_chain(x, types)
        cands = []
        arcs = sorted([e for e, t in enumerate(types) if t == "A"], key=lambda e: abs(el[e].k))
        cands += [("lankas->tiesė", op_to_line, e) for e in arcs]
        cands += [("pašalinti", op_remove, e) for e in np.argsort([e.L for e in el])]
        done = False
        for name, op, e in cands:
            r = op(types, x, int(e))
            if r is None:
                continue
            T2, x2 = r
            x2 = optimize_h(P, T2, x2, st)
            ok, _ = h_check(P, T2, x2, st)
            if ok:
                log(f"    supaprastinta: {name} #{e} -> {len(T2)} el.")
                types, x, done = T2, x2, True
                break
        if not done:
            break
    return types, x


def fit_horizontal(P, st, log):
    f, best = 0.8, None
    for it in range(st.max_refine):
        segs = segment(P, st.tol_line * f, st.tol_arc * f, st)
        if best is not None and len(segs) > max(12, len(P) // 4):
            log("    planas: per daug elementų - tolerancija tikriausiai mažesnė už duomenų triukšmą")
            break
        types, x = init_params(P, segs)
        x = optimize_h(P, types, x, st)
        ok, ratio = h_check(P, types, x, st)
        log(f"    planas: iter {it + 1}, segm.tol x{f:.2f}, {len(types)} el., max nuokr./tol = {ratio:.2f}")
        if best is None or ratio < best[2]:
            best = (types, x, ratio)
        if ok:
            break
        f *= 0.6
    if best[2] > 1.0 + 1e-6:
        # tolerancija nepasiekta - bandomi paprastesni modeliai, imamas mažiausio nuokrypio
        for f in (1.5, 2.5):
            types, x = init_params(P, segment(P, st.tol_line * f, st.tol_arc * f, st))
            x = optimize_h(P, types, x, st)
            ok, ratio = h_check(P, types, x, st)
            log(f"    planas: segm.tol x{f:.2f}, {len(types)} el., max nuokr./tol = {ratio:.2f}")
            if ratio < best[2]:
                best = (types, x, ratio)
        log(f"    ĮSPĖJIMAS: tolerancija nepasiekta (max nuokr./tol = {best[2]:.2f}), žr. ataskaitą")
    types, x, ratio = best
    if st.simplify and ratio <= 1.0 + 1e-6:
        types, x = simplify_h(P, types, x, st, log)
    return types, x


def trim_chain(el, s_a, s_b):
    out = []
    for e in el:
        a, b = max(s_a, e.s0), min(s_b, e.s0 + e.L)
        if b - a <= 1e-9:
            continue
        t = a - e.s0
        out.append(Elem(e.xy([t])[0], e.th + e.k * t, e.k, b - a, a - s_a))
    return out


# =====================================================================
#  Profilis
# =====================================================================
def fit_vline(s, z):
    A = np.column_stack([np.ones_like(s), s])
    (a, b), *_ = np.linalg.lstsq(A, z, rcond=None)
    return np.abs(z - (a + b * s)).max(), a, b








def prof_eval(s, S, E, Lc):
    z = np.interp(s, S, E)
    g = np.diff(E) / np.diff(S)
    for i in range(1, len(S) - 1):
        L = Lc[i]
        if L < 1e-6:
            continue
        gi, go = g[i - 1], g[i]
        a = S[i] - L / 2
        msk = (s > a) & (s < S[i] + L / 2)
        xx = s[msk] - a
        z[msk] = E[i] - gi * L / 2 + gi * xx + (go - gi) / (2 * L) * xx * xx
    return z


class VModel:
    def __init__(self, S, E, st):
        self.S0, self.Send = S[0], S[-1]
        self.ni = len(S) - 2
        g = np.diff(E) / np.diff(S)
        dg = np.abs(np.diff(g)) * 100
        self.curve = (dg <= st.v_sharp_dg) if st.v_sharp_dg > 0 else np.ones(self.ni, bool)
        self.st = st
        gaps = np.diff(S)
        lo = [S[i] - 0.45 * gaps[i - 1] for i in range(1, self.ni + 1)]
        hi = [S[i] + 0.45 * gaps[i] for i in range(1, self.ni + 1)]
        nc = int(self.curve.sum())
        self.lo = np.array(lo + [-np.inf] * (self.ni + 2) + [0.0] * nc)
        self.hi = np.array(hi + [np.inf] * (self.ni + 2) + [1.0] * nc)
        self.x0 = np.concatenate([S[1:-1], E, np.full(nc, 0.5)])

    def unpack(self, x):
        ni = self.ni
        S = np.concatenate([[self.S0], x[:ni], [self.Send]])
        E = x[ni:2 * ni + 2]
        u = x[2 * ni + 2:]
        Lc = np.zeros(ni + 2)
        gaps = np.diff(S)
        j = 0
        for i in range(1, ni + 1):
            if self.curve[i - 1]:
                avail = max(min(gaps[i - 1], gaps[i], self.st.v_max_len), 0.0)
                Lc[i] = u[j] * avail; j += 1
        return S, E, Lc


def optimize_v(s, z, S, E, st):
    mdl = VModel(S, E, st)

    def fun(x):
        S_, E_, L_ = mdl.unpack(x)
        return (z - prof_eval(s, S_, E_, L_)) / st.v_tol

    x0 = np.clip(mdl.x0, mdl.lo + 1e-9, mdl.hi - 1e-9)
    if len(x0) == 0:
        return mdl.unpack(x0)
    r = least_squares(fun, x0, bounds=(mdl.lo, mdl.hi), x_scale="jac", method="trf",
                      max_nfev=st.max_nfev, ftol=1e-6, xtol=1e-8)
    return mdl.unpack(r.x)


def v_maxdev(s, z, S, E, Lc):
    return float(np.abs(z - prof_eval(s, S, E, Lc)).max())





def fit_vertical(s, z, st, log, max_vpi=200):
    """Iš viršaus į apačią: vienas nuolydis -> VPI įterpiamas ties didžiausiu nuokrypiu."""
    _, a, b = fit_vline(s, z)
    S, E, Lc = np.array([s[0], s[-1]]), np.array([a + b * s[0], a + b * s[-1]]), np.zeros(2)
    min_gap = max(1e-3, (s[-1] - s[0]) * 1e-4)
    for it in range(max_vpi):
        S, E, Lc = optimize_v(s, z, S, E, st)
        res = z - prof_eval(s, S, E, Lc)
        dev = float(np.abs(res).max())
        if dev <= st.v_tol * (1 + 1e-6):
            break
        # kandidatai pagal nuokrypį; imamas pirmas, nesutampantis su esamu VPI
        inserted = False
        for w in np.argsort(-np.abs(res)):
            sw = s[w]
            if np.min(np.abs(S - sw)) > min_gap:
                k = int(np.searchsorted(S, sw))
                S, E = np.insert(S, k, sw), np.insert(E, k, z[w])
                inserted = True
                break
        if not inserted:
            break
    log(f"    profilis: {len(S) - 2} VPI, max |dz| = {dev * 1000:.1f} mm")
    # VPI mažinimas (pirmiausia su mažiausiu Δi)
    while dev <= st.v_tol * (1 + 1e-6) and len(S) > 2:
        g = np.diff(E) / np.diff(S)
        order = np.argsort(np.abs(np.diff(g)))
        done = False
        for i in order:
            S2, E2 = np.delete(S, i + 1), np.delete(E, i + 1)
            S2, E2, L2 = optimize_v(s, z, S2, E2, st)
            d2 = v_maxdev(s, z, S2, E2, L2)
            if d2 <= st.v_tol * (1 + 1e-6):
                log(f"    supaprastinta: pašalintas VPI -> {len(S2) - 2} VPI")
                S, E, Lc, dev, done = S2, E2, L2, d2, True
                break
        if not done:
            break
    return S, E, Lc


# =====================================================================
#  Įvestis (bendras branduolys app/geo)
# =====================================================================
def read_dxf(path, layers=None):
    return [(p.layer, p.handle, p.pts) for p in io_dxf.read_polylines(path, layers)]


def read_csv(path):
    """CSV: taškai grupuojami pagal kodą (stulpelis po Z) - kiekvienas kodas = atskira linija."""
    ps = io_csv.read_points(path)
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(ps.codes):
        groups.setdefault(c or "LINE", []).append(i)
    return [(name, str(k), ps.xyz[idx]) for k, (name, idx) in enumerate(groups.items())]


def clean_points(P):
    return polyline.dedupe(P)


def split_sharp(P, angle_deg, min_chord):
    return polyline.split_at_corners(P, angle_deg, min_chord)


# =====================================================================
#  Rezultatas
# =====================================================================
class Result:
    def __init__(self, name, layer, P, origin, elems, vprof, st):
        self.name, self.layer, self.P, self.origin = name, layer, P, origin
        self.elems, self.st = elems, st
        Q = P[:, :2] - origin
        self.off, self.sta, self.idx = project(Q, elems)
        self.length = sum(e.L for e in elems)
        self.vprof = vprof
        if vprof is not None:
            S, E, Lc = vprof
            self.dz = P[:, 2] - prof_eval(self.sta, S, E, Lc)
        else:
            self.dz = None

    def z_at(self, s):
        if self.vprof is None:
            return np.interp(s, self.sta[np.argsort(self.sta)], self.P[np.argsort(self.sta), 2])
        return prof_eval(np.asarray(s, float), *self.vprof)

    def densify(self, step=1.0):
        pts, ss = [], []
        for e in self.elems:
            n = max(1, int(math.ceil(e.L / step))) if e.is_arc else 1
            t = np.linspace(0, e.L, n + 1)
            if pts:
                t = t[1:]
            pts.append(e.xy(t) + self.origin); ss.append(e.s0 + t)
        xy, s = np.vstack(pts), np.concatenate(ss)
        if self.vprof is not None:       # papildomi taškai profilio kreivėse
            S, E, Lc = self.vprof
            extra = [np.arange(S[i] - Lc[i] / 2, S[i] + Lc[i] / 2 + 1e-9, step)
                     for i in range(1, len(S) - 1) if Lc[i] > 0]
            extra.append(S)
            s_all = np.unique(np.round(np.concatenate([s] + extra), 6))
            s_all = s_all[(s_all >= 0) & (s_all <= self.length + 1e-9)]
            xy = self.xy_at(s_all); s = s_all
        return xy, s, self.z_at(s)

    def xy_at(self, s):
        out = np.empty((len(s), 2))
        for i, si in enumerate(s):
            e = next((e for e in self.elems if si <= e.s0 + e.L + 1e-9), self.elems[-1])
            out[i] = e.xy([min(max(si - e.s0, 0), e.L)])[0] + self.origin
        return out


def to_alignment(r) -> Alignment:
    """Rezultatas -> bendras Alignment (planas + profilis) LandXML rašymui."""
    el = [HElem("arc" if e.is_arc else "line", e.p + r.origin, e.th, e.L, e.k, e.k, e.s0) for e in r.elems]
    al = Alignment(r.name, el, 0.0)
    if r.vprof is not None:
        S, E, Lc = r.vprof
        al.profile = Profile(np.asarray(S), np.asarray(E), np.asarray(Lc), r.name)
    return al


def write_landxml(results, path):
    io_landxml.write_alignments(path, [to_alignment(r) for r in results], app="linefit")


def write_dxf(results, path, name_tpl):
    w = io_dxf.DxfWriter()
    for r in results:
        base = name_tpl.replace("{L}", r.layer)
        la, ll, le = w.layer(base + "-AL", 1), w.layer(base + "-LS", 5), w.layer(base + "-ERR", 6)
        for e in r.elems:
            a, b = e.p + r.origin, e.end() + r.origin
            if not e.is_arc:
                w.line(a, b, la)
            else:
                c = e.center() + r.origin
                a0 = math.degrees(math.atan2(a[1] - c[1], a[0] - c[0]))
                a1 = math.degrees(math.atan2(b[1] - c[1], b[0] - c[0]))
                if e.k < 0:
                    a0, a1 = a1, a0
                w.arc(c, e.R, a0, a1, la)
        xy, s_, z = r.densify(1.0)
        w.polyline3d(np.column_stack([xy, z]), ll)
        for i in r.bad_points():
            w.circle(r.P[i], 0.25, le)
    w.save(path)


def _bad(r):
    st = r.st
    htol = st.report_tol or None
    tolp = np.where([r.elems[i].is_arc for i in r.idx], st.tol_arc, st.tol_line) if htol is None else htol
    hb = np.abs(r.off) > tolp + 1e-9
    vb = np.abs(r.dz) > (st.report_tol or st.v_tol) + 1e-9 if r.dz is not None else np.zeros(len(r.P), bool)
    return hb, vb


Result.bad_points = lambda self: np.nonzero(np.logical_or(*_bad(self)))[0]


def report(results, path, st, src):
    o = []
    o.append("linefit ataskaita")
    o.append(f"Šaltinis: {src}   Data: {dt.datetime.now():%Y-%m-%d %H:%M}")
    o.append(f"Planas: tol tiesė {st.tol_line * 1000:.1f} mm, tol lankas {st.tol_arc * 1000:.1f} mm, "
             f"R {st.r_min:g}…{st.r_max:g} m, skaidymo kampas {st.split_angle:g}°")
    if st.vertical:
        o.append(f"Profilis: tol {st.v_tol * 1000:.1f} mm, min Δi {st.v_min_dg:g} %, max L {st.v_max_len:g} m")
    for r in results:
        o.append("")
        o.append("=" * 78)
        o.append(f"{r.name}   (sluoksnis {r.layer}, taškų {len(r.P)}, ilgis {r.length:.3f} m)")
        o.append("=" * 78)
        o.append("Planas:")
        o.append(f"{'Nr':>3} {'Tipas':<6} {'PK pradž.':>11} {'PK pab.':>11} {'Ilgis':>10} {'R':>11} {'Posūkis':>8}")
        for i, e in enumerate(r.elems):
            Rs = f"{e.R:11.3f}" if e.is_arc else f"{'-':>11}"
            turn = ("kairėn" if e.k > 0 else "dešinėn") if e.is_arc else ""
            o.append(f"{i + 1:>3} {'Lankas' if e.is_arc else 'Tiesė':<6} {e.s0:11.3f} {e.s0 + e.L:11.3f} {e.L:10.3f} {Rs} {turn:>8}")
        if r.vprof is not None:
            S, E, Lc = r.vprof
            g = np.diff(E) / np.diff(S) * 100
            o.append("Profilis:")
            o.append(f"{'Nr':>3} {'PK':>11} {'Aukštis':>10} {'i į %':>8} {'i iš %':>8} {'Δi %':>7} {'L':>9} {'Rv':>10}")
            for i in range(len(S)):
                gi = f"{g[i - 1]:8.3f}" if i > 0 else f"{'':>8}"
                go = f"{g[i]:8.3f}" if i < len(S) - 1 else f"{'':>8}"
                if 0 < i < len(S) - 1:
                    d = g[i] - g[i - 1]
                    Rv = Lc[i] / abs(d / 100) if Lc[i] > 0 and abs(d) > 1e-9 else 0
                    rvs = f"{Rv:10.0f}" if Lc[i] > 0 else f"{'-':>10}"
                    mark = " !" if abs(d) < st.v_min_dg else ""
                    o.append(f"{i:>3} {S[i]:11.3f} {E[i]:10.3f} {gi} {go} {d:7.3f} {Lc[i]:9.3f} {rvs}{mark}")
                else:
                    o.append(f"{i:>3} {S[i]:11.3f} {E[i]:10.3f} {gi} {go} {'':>7} {'':>9} {'':>10}")
        ah = np.abs(r.off)
        o.append("Statistika:")
        o.append(f"  planas:   max {ah.max() * 1000:7.1f} mm ties PK {r.sta[ah.argmax()]:.3f}, vid. {ah.mean() * 1000:6.1f} mm")
        if r.dz is not None:
            az = np.abs(r.dz)
            o.append(f"  profilis: max {az.max() * 1000:7.1f} mm ties PK {r.sta[az.argmax()]:.3f}, vid. {az.mean() * 1000:6.1f} mm")
        hb, vb = _bad(r)
        bad = np.nonzero(hb | vb)[0]
        o.append(f"  taškų už tolerancijos: {len(bad)}  (* planas, ^ profilis)")
        if len(bad):
            o.append(f"  {'Nr':>5} {'PK':>11} {'Δplan mm':>9} {'Δz mm':>8}  ")
            for i in bad[np.argsort(r.sta[bad])]:
                dzs = f"{r.dz[i] * 1000:8.1f}" if r.dz is not None else f"{'':>8}"
                o.append(f"  {i:>5} {r.sta[i]:11.3f} {r.off[i] * 1000:9.1f} {dzs}  {'*' if hb[i] else ' '}{'^' if vb[i] else ' '}")
        short = [i + 1 for i, e in enumerate(r.elems) if e.L < st.min_len]
        if short:
            o.append(f"  ĮSPĖJIMAS: elementai trumpesni nei {st.min_len} m: {short}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(o) + "\n")
    return "\n".join(o)


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axs = plt.subplots(n, 3, figsize=(18, 4.5 * n), squeeze=False)
    for row, r in zip(axs, results):
        xy, s, z = r.densify(0.5)
        ax = row[0]
        ax.plot(r.P[:, 0], r.P[:, 1], ".", ms=3, color="0.5", label="įvestis")
        for e in r.elems:
            t = np.linspace(0, e.L, 60)
            q = e.xy(t) + r.origin
            ax.plot(q[:, 0], q[:, 1], "-", color="tab:red" if e.is_arc else "tab:blue", lw=1.5)
            ax.plot(q[0, 0], q[0, 1], "k|", ms=8)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_title(f"{r.name}: planas (mėlyna tiesė, raudona lankas)")
        o = np.argsort(r.sta)
        row[1].plot(r.sta[o], r.off[o] * 1000, ".-", ms=3)
        row[1].axhline(0, color="k", lw=.5); row[1].set_title("plano nuokrypis, mm"); row[1].set_xlabel("PK, m")
        for e in r.elems:
            row[1].axvline(e.s0, color="0.85", lw=.8)
        if r.dz is not None:
            ax2 = row[2]
            ax2.plot(r.sta[o], r.dz[o] * 1000, ".-", ms=3, color="tab:green")
            ax2.axhline(0, color="k", lw=.5); ax2.set_title("profilio nuokrypis Δz, mm"); ax2.set_xlabel("PK, m")
            S, E, Lc = r.vprof
            for i in range(1, len(S) - 1):
                ax2.axvline(S[i], color="0.7", lw=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# =====================================================================
def process_line(name, layer, P, st, log):
    P = clean_points(P)
    if len(P) < 3:
        log(f"  {name}: per mažai taškų, praleista")
        return None
    origin = P[0, :2].copy()
    Q = P[:, :2] - origin
    types, x = fit_horizontal(Q, st, log)
    el = build_chain(x, types)
    _, sta, _ = project(Q, el)
    el = trim_chain(el, float(sta.min()), float(sta.max()))
    _, sta, _ = project(Q, el)
    vprof = None
    if st.vertical:
        o = np.argsort(sta, kind="stable")
        vprof = fit_vertical(sta[o], P[o, 2], st, log)
    return Result(name, layer, P, origin, el, vprof, st)


def run(inputs, st, out_prefix, name_tpl="{L}", layers=None, make_plot=True, log=print):
    """Bendras įėjimo taškas (CLI ir web). Grąžina (results, files)."""
    lines = []
    for path in inputs:
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".dxf":
            try:
                lines += read_dxf(str(path), layers)
            except (OSError, Exception) as e:
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"{os.path.basename(str(path))}: nepavyko perskaityti DXF ({e})")
        elif ext in (".csv", ".txt"):
            lines += read_csv(str(path))
        else:
            log(f"Praleista (nežinomas formatas): {os.path.basename(str(path))}")
    if not lines:
        raise ValueError("Nerasta linijų (DXF LWPOLYLINE/POLYLINE arba CSV x,y,z).")
    log(f"Rasta linijų: {len(lines)}")
    results = []
    for layer, h, P in lines:
        P = clean_points(P)
        parts = split_sharp(P, st.split_angle, 10 * st.tol_line)
        for pi, part in enumerate(parts):
            name = f"{layer}_{h}" + (f"_{pi + 1}" if len(parts) > 1 else "")
            log(f"{name}: {len(part)} tšk.")
            r = process_line(name, layer, part, st, log)
            if r:
                results.append(r)
    if not results:
        raise ValueError("Nė viena linija neturi pakankamai taškų.")
    files = {"xml": out_prefix + ".xml", "dxf": out_prefix + ".dxf", "report": out_prefix + "_report.txt"}
    write_landxml(results, files["xml"])
    write_dxf(results, files["dxf"], name_tpl)
    report(results, files["report"], st, ", ".join(os.path.basename(str(i)) for i in inputs))
    if make_plot:
        files["png"] = out_prefix + ".png"
        plot(results, files["png"])
    return results, files


def _decimate(a, nmax):
    if len(a) <= nmax:
        return a
    return a[np.linspace(0, len(a) - 1, nmax).astype(int)]


def preview(results):
    """Duomenys naršyklės peržiūrai: planas + nuokrypių grafikai."""
    plan, charts, summary = [], [], []
    for r in results:
        xy, s, z = r.densify(max(0.5, r.length / 2000))
        hb, vb = _bad(r)
        bad = np.nonzero(hb | vb)[0]
        joints = np.array([e.p + r.origin for e in r.elems] + [r.elems[-1].end() + r.origin])
        plan.append({
            "name": r.name,
            "points": np.round(_decimate(r.P[:, :2], 4000), 3).tolist(),
            "elements": [{"type": "arc" if e.is_arc else "line",
                          "xy": np.round(e.xy(np.linspace(0, e.L, max(2, int(e.L / max(0.5, r.length / 2000)) + 1)))
                                         + r.origin, 3).tolist(),
                          "s0": round(e.s0, 3), "L": round(e.L, 3),
                          "R": round(e.R, 3) if e.is_arc else None} for e in r.elems],
            "joints": np.round(joints, 3).tolist(),
            "bad": np.round(r.P[bad, :2], 3).tolist(),
        })
        o = np.argsort(r.sta)
        ch = {"title": r.name, "x": np.round(r.sta[o], 3).tolist(),
              "series": [{"name": "Δ planas, mm", "y": np.round(r.off[o] * 1000, 2).tolist(),
                           "tol": round((r.st.report_tol or max(r.st.tol_line, r.st.tol_arc)) * 1000, 2)}],
              "vlines": [round(e.s0, 3) for e in r.elems[1:]]}
        if r.dz is not None:
            ch["series"].append({"name": "Δz, mm", "y": np.round(r.dz[o] * 1000, 2).tolist(),
                                 "tol": round((r.st.report_tol or r.st.v_tol) * 1000, 2)})
        charts.append(ch)
        ah = np.abs(r.off)
        summary.append({"Linija": r.name, "Ilgis, m": round(r.length, 3),
                        "Elementų": len(r.elems),
                        "Lankų": sum(e.is_arc for e in r.elems),
                        "VPI": (len(r.vprof[0]) - 2) if r.vprof is not None else "-",
                        "max Δplan, mm": round(float(ah.max()) * 1000, 1),
                        "max Δz, mm": round(float(np.abs(r.dz).max()) * 1000, 1) if r.dz is not None else "-",
                        "Už tol.": int(len(bad))})
    return {"plan": plan, "charts": charts, "table": summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Linijų optimizavimas: tiesės/lankai + VPI/parabolės")
    ap.add_argument("input", nargs="+", help="DXF arba CSV (x,y,z[,pavadinimas])")
    ap.add_argument("--layer", action="append", help="DXF sluoksnis (galima kartoti)")
    ap.add_argument("--out", help="išvesties prefiksas (numatyta: <input>_fit)")
    ap.add_argument("--name", default="{L}", help="sluoksnių šablonas, {L} = šaltinio sluoksnis")
    ap.add_argument("--tol-line", type=float, default=0.02)
    ap.add_argument("--tol-arc", type=float, default=0.02)
    ap.add_argument("--r-min", type=float, default=5.0)
    ap.add_argument("--r-max", type=float, default=10000.0)
    ap.add_argument("--min-len", type=float, default=0.5)
    ap.add_argument("--split-angle", type=float, default=60.0)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--no-vertical", action="store_true")
    ap.add_argument("--v-tol", type=float, default=0.01)
    ap.add_argument("--v-min-dg", type=float, default=0.10, help="min nuolydžių skirtumas VPI, %%")
    ap.add_argument("--v-max-len", type=float, default=300.0)
    ap.add_argument("--v-sharp-dg", type=float, default=0.0, help="Δi %% virš kurio VPI be kreivės (0=išj.)")
    ap.add_argument("--report-tol", type=float, default=0.0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    st = Settings(tol_line=a.tol_line, tol_arc=a.tol_arc, r_min=a.r_min, r_max=a.r_max,
                  min_len=a.min_len, split_angle=a.split_angle, simplify=not a.no_simplify,
                  vertical=not a.no_vertical, v_tol=a.v_tol, v_min_dg=a.v_min_dg,
                  v_max_len=a.v_max_len, v_sharp_dg=a.v_sharp_dg, report_tol=a.report_tol)
    out = a.out or os.path.splitext(a.input[0])[0] + "_fit"
    try:
        _, files = run(a.input, st, out, a.name, a.layer, a.plot, lambda m: print(m, flush=True))
    except ValueError as e:
        sys.exit(str(e))
    print()
    print(open(files["report"], encoding="utf-8").read())


if __name__ == "__main__":
    main()

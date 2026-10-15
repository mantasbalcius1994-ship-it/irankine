"""Trasa: plane tiesės / lankai / klotoidės, profilyje VPI su parabolinėmis kreivėmis.

Susitarimai:
  - XY: X = rytai (E), Y = šiaurė (N), metrai.
  - Vidinė kryptis `th` - matematinis kampas nuo +X prieš laikrodžio rodyklę (rad).
  - Kreivumas k: + posūkis į kairę, - į dešinę. Klotoidėje k kinta tiesiškai k0 -> k1.
  - Atstumas nuo ašies (offset): + DEŠINĖJE (kaip TBC).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


@dataclass
class HElem:
    kind: str           # line | arc | spiral
    p: np.ndarray       # pradžios taškas (2,)
    th: float           # pradžios kryptis
    L: float
    k0: float = 0.0
    k1: float = 0.0
    s0: float = 0.0     # pradžios PK (nuo trasos pradžios, be staStart)

    def theta(self, t):
        t = np.asarray(t, float)
        return self.th + self.k0 * t + (self.k1 - self.k0) * t * t / (2 * self.L)

    def xy(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        if self.kind == "line" or (abs(self.k0) < 1e-15 and abs(self.k1) < 1e-15):
            return np.column_stack([self.p[0] + t * math.cos(self.th), self.p[1] + t * math.sin(self.th)])
        if self.kind == "arc" or abs(self.k1 - self.k0) < 1e-15:
            k = self.k0
            return np.column_stack([self.p[0] + (np.sin(self.th + k * t) - math.sin(self.th)) / k,
                                    self.p[1] - (np.cos(self.th + k * t) - math.cos(self.th)) / k])
        # klotoidė: Gauss-Legendre integralas [0, t]
        u = (t[:, None] / 2) * (_GL_X[None, :] + 1)
        a = self.theta(u)
        w = _GL_W[None, :] * (t[:, None] / 2)
        return np.column_stack([self.p[0] + (w * np.cos(a)).sum(1), self.p[1] + (w * np.sin(a)).sum(1)])

    def end(self):
        return self.xy([self.L])[0]

    def th_end(self):
        return float(self.theta(self.L))

    @property
    def radius(self):
        k = self.k0 if self.kind == "arc" else max(abs(self.k0), abs(self.k1))
        return 1 / abs(k) if abs(k) > 1e-15 else math.inf


@dataclass
class Profile:
    """VPI: stotys S, aukščiai E, kreivių ilgiai Lc (0 pradžioje/pabaigoje ir lūžiuose).
    LandXML CircCurve skaitomas kaip parabolė su tuo pačiu L (skirtumas keliams < 1 mm)."""
    S: np.ndarray
    E: np.ndarray
    Lc: np.ndarray
    name: str = ""

    def z_at(self, s):
        s = np.asarray(s, float)
        S, E, Lc = self.S, self.E, self.Lc
        z = np.interp(s, S, E, left=np.nan, right=np.nan)
        g = np.diff(E) / np.diff(S)
        for i in range(1, len(S) - 1):
            L = Lc[i]
            if L <= 1e-9:
                continue
            a = S[i] - L / 2
            m = (s > a) & (s < S[i] + L / 2)
            x = s[m] - a
            z[m] = E[i] - g[i - 1] * L / 2 + g[i - 1] * x + (g[i] - g[i - 1]) / (2 * L) * x * x
        return z

    def grade_at(self, s, ds=0.01):
        return (self.z_at(np.asarray(s) + ds) - self.z_at(np.asarray(s) - ds)) / (2 * ds)


@dataclass
class Alignment:
    name: str
    elems: list[HElem]
    sta_start: float = 0.0
    profile: Profile | None = None
    _dens: tuple | None = field(default=None, repr=False)

    # ---------------- bendra
    @property
    def length(self):
        return sum(e.L for e in self.elems)

    @property
    def sta_end(self):
        return self.sta_start + self.length

    @classmethod
    def from_elems(cls, name, specs, p0, th0, sta_start=0.0):
        """specs: [(kind, L, k0, k1), ...] - grandinė nuo p0 kryptimi th0 (G1)."""
        p, th, s, el = np.asarray(p0, float), float(th0), 0.0, []
        for kind, L, k0, k1 in specs:
            e = HElem(kind, p, th, float(L), float(k0), float(k1), s)
            el.append(e)
            p, th, s = e.end(), e.th_end(), s + L
        return cls(name, el, sta_start)

    def _elem_at(self, s):
        """s - vidinė stotis nuo 0. Grąžina elemento indeksus."""
        starts = np.array([e.s0 for e in self.elems])
        return np.clip(np.searchsorted(starts, s, side="right") - 1, 0, len(self.elems) - 1)

    def xy_at(self, sta, off=0.0):
        """PK (su sta_start) ir atstumas (+ dešinėje) -> XY."""
        s = np.atleast_1d(np.asarray(sta, float)) - self.sta_start
        off = np.broadcast_to(np.asarray(off, float), s.shape)
        out = np.empty((len(s), 2))
        idx = self._elem_at(s)
        for i in np.unique(idx):
            m = idx == i
            e = self.elems[i]
            t = np.clip(s[m] - e.s0, 0, e.L)
            xy = e.xy(t)
            th = e.theta(t)
            out[m] = xy + off[m, None] * np.column_stack([np.sin(th), -np.cos(th)])
        return out

    def theta_at(self, sta):
        s = np.atleast_1d(np.asarray(sta, float)) - self.sta_start
        idx = self._elem_at(s)
        return np.array([self.elems[i].theta(np.clip(si - self.elems[i].s0, 0, self.elems[i].L)) for i, si in zip(idx, s)])

    def curvature_at(self, sta):
        s = np.atleast_1d(np.asarray(sta, float)) - self.sta_start
        idx = self._elem_at(s)
        out = np.empty(len(s))
        for i in np.unique(idx):
            m = idx == i
            e = self.elems[i]
            t = np.clip(s[m] - e.s0, 0, e.L)
            out[m] = e.k0 + (e.k1 - e.k0) * t / e.L
        return out

    def bearing_at(self, sta):
        """Geodezinė kryptis (azimutas nuo šiaurės pagal laikrodžio rodyklę), laipsniais."""
        return np.degrees(np.pi / 2 - self.theta_at(sta)) % 360

    def z_at(self, sta):
        if self.profile is None:
            return np.full(np.shape(np.atleast_1d(sta)), np.nan)
        return self.profile.z_at(sta)

    # ---------------- projekcija
    def densify(self, max_sagitta=1e-4, max_step=5.0):
        """Stotys ir XY tankiai: lankuose žingsnis toks, kad chordos paklaida < max_sagitta."""
        ss = []
        for e in self.elems:
            R = e.radius
            step = max_step if not math.isfinite(R) else min(max_step, math.sqrt(8 * R * max_sagitta))
            n = max(1, int(math.ceil(e.L / step)))
            ss.append(e.s0 + np.linspace(0, e.L, n + 1)[:-1])
        ss.append([self.length])
        s = np.concatenate(ss)
        return s + self.sta_start, self.xy_at(s + self.sta_start)

    def station_offset(self, xy, refine=True):
        """XY -> (PK, atstumas [+ dešinėje], ar statmuo patenka į trasos ribas)."""
        from scipy.spatial import cKDTree
        xy = np.atleast_2d(np.asarray(xy, float))[:, :2]
        if self._dens is None:
            s, P = self.densify()
            mid = (P[1:] + P[:-1]) / 2
            self._dens = (s, P, cKDTree(mid))
        s, P, tree = self._dens
        k = min(16, len(P) - 1)
        _, cand = tree.query(xy, k=k)
        cand = np.atleast_2d(cand).reshape(len(xy), -1)
        A, B = P[cand], P[cand + 1]
        d = B - A
        l2 = np.maximum((d ** 2).sum(-1), 1e-18)
        t = np.clip(((xy[:, None, :] - A) * d).sum(-1) / l2, 0, 1)
        proj = A + t[..., None] * d
        dist = np.hypot(*(xy[:, None, :] - proj).transpose(2, 0, 1))
        j = dist.argmin(1)
        r = np.arange(len(xy))
        seg = cand[r, j]
        sta = s[seg] + t[r, j] * (s[seg + 1] - s[seg])
        if refine:                                    # Niutonas ant tikslios geometrijos (su kreivumu)
            for _ in range(4):
                th = self.theta_at(sta)
                v = xy - self.xy_at(sta)
                along = v[:, 0] * np.cos(th) + v[:, 1] * np.sin(th)
                left = -v[:, 0] * np.sin(th) + v[:, 1] * np.cos(th)
                den = np.maximum(1 - self.curvature_at(sta) * left, 0.2)
                sta = np.clip(sta + along / den, self.sta_start, self.sta_end)
        th = self.theta_at(sta)
        v = xy - self.xy_at(sta)
        off = v[:, 0] * np.sin(th) - v[:, 1] * np.cos(th)
        tan = np.column_stack([np.cos(th), np.sin(th)])
        along = (v * tan).sum(1)
        inside = (np.abs(along) < 1e-3) | ((sta > self.sta_start + 1e-9) & (sta < self.sta_end - 1e-9))
        return sta, off, inside


def format_station(sta, decimals=2, per=1000):
    """per=1000: 1234.5 -> '1+234.50' (km+m);  per=100: '12+34.50' (piketai)."""
    sta = float(sta)
    sign = "-" if sta < 0 else ""
    sta = round(abs(sta), decimals)
    big = int(sta // per)
    m = sta - big * per
    w = len(str(per - 1)) + 1 + decimals
    return f"{sign}{big}+{m:0{w}.{decimals}f}"

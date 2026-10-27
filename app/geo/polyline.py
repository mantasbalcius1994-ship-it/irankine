"""Linijų (taškų sekų) pagalbinės funkcijos. Taškai (n,2) arba (n,3)."""
from __future__ import annotations

import math

import numpy as np


def dedupe(P, tol=1e-6):
    """Pašalina iš eilės einančius sutampančius (plane) taškus."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i, :2] - P[keep[-1], :2])) > tol:
            keep.append(i)
    return P[keep]


def chainage(P):
    """Sukauptas 2D ilgis kiekviename taške."""
    P = np.asarray(P, float)
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]


def length(P):
    return float(chainage(P)[-1])


def resample(P, step, keep_vertices=False):
    """Taškai kas `step` m (2D ilgiu), z interpoliuojamas."""
    P = np.asarray(P, float)
    s = chainage(P)
    t = np.arange(0, s[-1], step)
    t = np.r_[t, s[-1]]
    if keep_vertices:
        t = np.unique(np.r_[t, s])
    return np.column_stack([np.interp(t, s, P[:, i]) for i in range(P.shape[1])])


def rdp(P, tol, use_z=False):
    """Ramer-Douglas-Peucker (iteracinis). use_z: atstumas 3D, kitaip 2D. Grąžina paliktų indeksų masyvą."""
    P = np.asarray(P, float)
    Q = P[:, :3] if use_z and P.shape[1] > 2 else P[:, :2]
    n = len(Q)
    if n < 3:
        return np.arange(n)
    keep = np.zeros(n, bool); keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = Q[b] - Q[a]
        L = np.linalg.norm(seg)
        V = Q[a + 1:b] - Q[a]
        if L < 1e-12:
            d = np.linalg.norm(V, axis=1)
        elif Q.shape[1] == 2:
            d = np.abs(seg[0] * V[:, 1] - seg[1] * V[:, 0]) / L
        else:
            d = np.linalg.norm(np.cross(V, seg / L), axis=1)
        i = int(d.argmax())
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def _span_dev(P, a, b):
    """Tarpinių viršūnių (a, b) nuokrypiai nuo atkarpos P[a]-P[b]: (plane, aukštyje).
    Plane - atstumas iki ATKARPOS (ne begalinės tiesės), aukštyje - z skirtumas toje pačioje vietoje."""
    A, B = P[a, :2], P[b, :2]
    V = P[a + 1:b, :2] - A
    d = B - A
    l2 = float(d @ d)
    if l2 < 1e-18:
        t = np.zeros(len(V))
    else:
        t = np.clip((V @ d) / l2, 0, 1)
    dh = np.hypot(*(V - t[:, None] * d).T)
    if P.shape[1] > 2:
        dv = np.abs(P[a + 1:b, 2] - (P[a, 2] + t * (P[b, 2] - P[a, 2])))
    else:
        dv = np.zeros(len(V))
    return dh, dv


def simplify(P, tol_h, tol_v=None, closed=False):
    """Viršūnių retinimas (RDP su dviem tolerancijomis). Paliekami ORIGINALŪS taškai (niekas nepastumiama).
    Garantija: kiekviena pašalinta viršūnė yra ne toliau kaip tol_h plane ir tol_v aukštyje nuo naujos linijos.
    tol_v=None - aukštis neribojamas (2D). Grąžina paliktų indeksų masyvą."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return np.arange(n)
    if closed:                                     # uždara: dalijama ties toliausiu nuo pradžios tašku
        far = int(np.hypot(*(P[:, :2] - P[0, :2]).T).argmax())
        if far in (0, n - 1):
            far = n // 2
        a = simplify(P[:far + 1], tol_h, tol_v)
        b = simplify(np.vstack([P[far:], P[:1]]), tol_h, tol_v) + far
        return np.unique(np.r_[a, b[b < n]])
    use_v = tol_v is not None and P.shape[1] > 2
    keep = np.zeros(n, bool)
    keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        dh, dv = _span_dev(P, a, b)
        score = dh / tol_h
        if use_v:
            score = np.maximum(score, dv / tol_v)
        i = int(score.argmax())
        if score[i] > 1.0:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def simplify_fixed(P, fixed, tol_h, tol_v=None, closed=False):
    """Kaip simplify, bet `fixed` viršūnės (pvz. lankų galai) visada paliekamos, o retinamos tik atkarpos tarp jų.
    Uždarai linijai paskutinė -> pirma atkarpa irgi retinama."""
    P = np.asarray(P, float)
    n = len(P)
    fixed = np.asarray(fixed, bool).copy()
    if closed:
        P = np.vstack([P, P[:1]])
        fixed = np.r_[fixed, fixed[:1]]
    fixed[[0, -1]] = True
    F = np.nonzero(fixed)[0]
    keep = [F]
    for a, b in zip(F[:-1], F[1:]):
        if b - a > 1:
            keep.append(simplify(P[a:b + 1], tol_h, tol_v) + a)
    k = np.unique(np.concatenate(keep))
    return k[k < n] if closed else k


def deviation(P, keep):
    """Pašalintų viršūnių nuokrypis nuo supaprastintos linijos: (max plane, max aukštyje)."""
    P = np.asarray(P, float)
    mh = mv = 0.0
    for a, b in zip(keep[:-1], keep[1:]):
        if b - a > 1:
            dh, dv = _span_dev(P, a, b)
            mh, mv = max(mh, float(dh.max())), max(mv, float(dv.max()))
    return mh, mv


def remove_spikes(P, max_len, angle_deg=150.0):
    """Šuoliai: viršūnė, kur linija staigiai grįžta atgal (posūkis > angle_deg),
    o bent vienas gretimas segmentas trumpesnis už max_len. Grąžina (naujas P, pašalinta)."""
    P = np.asarray(P, float)
    removed = 0
    changed = True
    while changed and len(P) > 2:
        changed = False
        d = np.abs(deflections(P))
        ch = np.hypot(*np.diff(P[:, :2], axis=0).T)
        bad = np.nonzero((d > angle_deg) & (np.minimum(ch[:-1], ch[1:]) <= max_len))[0] + 1
        if len(bad):
            keep = np.ones(len(P), bool)
            keep[bad[::2]] = False                 # gretimus šaliname po vieną, kitas ratas patikrins
            P = P[keep]
            removed += int((~keep).sum())
            changed = True
    return P, removed


def remove_backtracks(P, tol=0.05, min_run=3):
    """„Ten ir atgal“ ekskursijos (dažnos IFC/mesh lūžio linijose): linija nueina iki taško ir grįžta tuo pačiu
    keliu. Aptinkama palindromu: viršūnių poros (i, j), |i-j| > 3, atstumas < tol, grupuojamos pagal i+j;
    grupė su >= min_run porų = ekskursija. Pavienė pašalinama paliekant bazinį tašką (šaka grąžinama
    patikrai); dvi gretimos su bendru apex sujungiamos per apex.
    Grąžina (naujas P, šakų sąrašas [(n,3) masyvai])."""
    from collections import defaultdict
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    n = len(P)
    if n < 2 * min_run + 3:
        return P, []
    groups = defaultdict(list)
    for i, j in cKDTree(P[:, :2]).query_pairs(tol):
        if abs(i - j) > 3:
            groups[i + j].append((min(i, j), max(i, j)))
    sp = sorted(({"s": min(p[0] for p in g), "e": max(p[1] for p in g), "apex": k // 2}
                 for k, g in groups.items() if len(g) >= min_run), key=lambda d: d["s"])
    # persidengiančios grupės (tas pats artefaktas) - paliekama plačiausia
    spikes = []
    for d in sp:
        if spikes and d["s"] <= spikes[-1]["e"]:
            if d["e"] - d["s"] > spikes[-1]["e"] - spikes[-1]["s"]:
                spikes[-1] = d
            continue
        spikes.append(d)
    if not spikes:
        return P, []
    ranges, branches, used = [], [], [False] * len(spikes)
    for a, A in enumerate(spikes):
        if used[a]:
            continue
        B = next((b for b in range(a + 1, len(spikes)) if not used[b]
                  and spikes[b]["s"] - A["e"] <= 2
                  and np.hypot(*(P[A["apex"], :2] - P[spikes[b]["apex"], :2])) < tol), None)
        if B is not None:                           # dvi ekskursijos su bendru apex - sujungiama per jį
            used[B] = True
            Bd = spikes[B]
            seg = np.vstack([P[A["s"]:A["apex"] + 1], P[Bd["apex"] - 1:Bd["s"] - 1:-1]])
            ranges.append((A["s"], Bd["e"], seg))
        else:                                       # pavienė - lieka bazinis taškas, šaka atiduodama patikrai
            ranges.append((A["s"], A["e"], P[A["s"]:A["s"] + 1]))
            branches.append(P[A["s"]:A["apex"] + 1].copy())
    out, cur = [], 0
    for s, e, seg in sorted(ranges, key=lambda r: r[0]):
        if s < cur:
            continue
        out.append(P[cur:s]); out.append(seg); cur = e + 1
    out.append(P[cur:])
    return np.vstack([o for o in out if len(o)]), branches


def trim_hooks(P, max_len, min_angle=30.0):
    """Galų „kabliukai“: trumpesnis už max_len paskutinis segmentas, staigiai pasuktas (> min_angle).
    Grąžina (naujas P, nukirptų viršūnių skaičius)."""
    P = np.asarray(P, float)
    cut = 0
    for _ in range(2):
        while len(P) > 2:
            a, b, c = P[0, :2], P[1, :2], P[2, :2]
            v1, v2 = b - a, c - b
            if np.hypot(*v1) >= max_len:
                break
            ang = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], float(v1 @ v2))))
            if ang <= min_angle:
                break
            P = P[1:]
            cut += 1
        P = P[::-1]
    return P, cut


def deflections(P):
    """Plano posūkio kampai viršūnėse (laipsniais, su ženklu: + kairėn). Ilgis n-2."""
    P = np.asarray(P, float)
    a = np.diff(P[:-1, :2], axis=0)
    b = np.diff(P[1:, :2], axis=0)
    return np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))


def split_at_corners(P, angle_deg, min_chord=0.0):
    """Skaido liniją viršūnėse, kur |posūkis| > angle_deg (abiejų chordų ilgis >= min_chord)."""
    P = np.asarray(P, float)
    if angle_deg <= 0 or len(P) < 3:
        return [P]
    d = np.abs(deflections(P))
    ch = np.hypot(*np.diff(P[:, :2], axis=0).T)
    cut = [i + 1 for i in range(len(d)) if d[i] > angle_deg and ch[i] >= min_chord and ch[i + 1] >= min_chord]
    idx = [0] + cut + [len(P) - 1]
    return [P[idx[k]:idx[k + 1] + 1] for k in range(len(idx) - 1)]


def project_point(P, q):
    """Artimiausias taškas linijoje: (atstumas išilgai, atstumas statmenai su ženklu [+ dešinėje], segmento nr.)."""
    P = np.asarray(P, float)[:, :2]
    q = np.asarray(q, float)[:2]
    A, B = P[:-1], P[1:]
    d = B - A
    l2 = np.maximum((d ** 2).sum(1), 1e-18)
    t = np.clip(((q - A) * d).sum(1) / l2, 0, 1)
    pr = A + t[:, None] * d
    dist = np.hypot(*(q - pr).T)
    i = int(dist.argmin())
    s = chainage(P)
    side = d[i, 0] * (q - A[i])[1] - d[i, 1] * (q - A[i])[0]
    return s[i] + t[i] * math.sqrt(l2[i]), -math.copysign(dist[i], side) if side else 0.0, i

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

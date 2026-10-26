"""Operacijos su daug linijų: dublikatų paieška, galų sujungimas."""
from __future__ import annotations

import numpy as np


def find_duplicates(lines: list[np.ndarray], tol: float, share: float = 1.0, step: float = 0.25) -> dict[int, int]:
    """Linijos, kurios VISOS (share=1) yra ne toliau kaip tol nuo kitos (ilgesnės arba tokio pat ilgio, ankstesnės) -
    pašalinus tokią, geometrija neprarandama. Tikrinamos visos viršūnės ir taškai kas `step` m.
    Grąžina {dublikato indeksas: linijos, kurią jis kartoja, indeksas}."""
    import shapely
    geoms = [shapely.LineString(P[:, :2]) for P in lines]
    lens = np.array([g.length for g in geoms])
    tree = shapely.STRtree(geoms)
    out: dict[int, int] = {}
    order = np.argsort(lens)                       # trumpiausios tikrinamos pirmos
    for i in order:
        g = geoms[i]
        if g.length <= 0:
            continue
        n = max(2, int(g.length / step) + 1)
        pts = np.concatenate([shapely.line_interpolate_point(g, np.linspace(0, g.length, n)),
                              shapely.points(lines[i][:, :2])])
        for j in tree.query(g.buffer(tol)):
            j = int(j)
            if j == i or j in out:
                continue
            if lens[j] < lens[i] or (abs(lens[j] - lens[i]) < 1e-9 and j > i):
                continue                            # paliekama ilgesnė / ankstesnė
            if np.mean(shapely.distance(pts, geoms[j]) <= tol) >= share:
                out[int(i)] = j
                break
    return out


def join_ends(lines: list[np.ndarray], gap: float, groups: list | None = None) -> list[list[tuple[int, bool]]]:
    """Linijų grandinės, kurių galai ne toliau kaip `gap`. groups: linijos jungiamos tik tos pačios grupės
    (pvz. sluoksnio). Jungiama tik jei galas turi lygiai vieną kaimyną (išsišakojimai nejungiami).
    Grąžina grandines: [[(linijos nr., apversta?), ...], ...] (ir pavienes linijas)."""
    from scipy.spatial import cKDTree
    n = len(lines)
    if groups is None:
        groups = [0] * n
    E = np.array([P[k, :2] for P in lines for k in (0, -1)])      # 2*i = pradžia, 2*i+1 = pabaiga
    tree = cKDTree(E)
    nb: dict[int, list[int]] = {}
    for a, b in tree.query_pairs(gap):
        if a // 2 == b // 2 or groups[a // 2] != groups[b // 2]:
            continue
        nb.setdefault(a, []).append(b); nb.setdefault(b, []).append(a)
    link = {a: v[0] for a, v in nb.items() if len(v) == 1 and len(nb.get(v[0], [])) == 1}
    used = [False] * n
    chains = []

    def walk(start_line, rev):
        chain = [(start_line, rev)]
        used[start_line] = True
        cur, r = start_line, rev
        while True:
            tail = 2 * cur + (0 if r else 1)        # išeinantis galas
            nxt = link.get(tail)
            if nxt is None or used[nxt // 2]:
                return chain
            cur, r = nxt // 2, (nxt % 2 == 1)      # jei prisijungėm prie pabaigos - linija apverčiama
            chain.append((cur, r))
            used[cur] = True

    # pradedama nuo linijų, kurių bent vienas galas laisvas (kitaip - uždaras ratas)
    for i in range(n):
        if used[i]:
            continue
        if 2 * i not in link:
            chains.append(walk(i, False))
        elif 2 * i + 1 not in link:
            chains.append(walk(i, True))
    for i in range(n):                              # likę - uždari ratai
        if not used[i]:
            chains.append(walk(i, False))
    return chains


def merge_chain(lines: list[np.ndarray], chain: list[tuple[int, bool]]) -> np.ndarray:
    parts = []
    for k, (i, rev) in enumerate(chain):
        P = lines[i][::-1] if rev else lines[i]
        if k and np.hypot(*(P[0, :2] - parts[-1][-1, :2])) < 1e-9:
            P = P[1:]
        parts.append(P)
    return np.vstack(parts)

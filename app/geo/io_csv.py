"""Taškų CSV / TXT skaitymas ir rašymas.

Palaikomi formatai: P,X,Y,Z[,kodas] / X,Y,Z / N,E,Z ..., skyrikliai , ; tab tarpas,
dešimtainis kablelis. LKS-94 atveju (E ~ 150-700 tūkst., N ~ 5.9-6.3 mln.) ašys atpažįstamos automatiškai.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointSet:
    xyz: np.ndarray                         # (n,3): X=E, Y=N, Z
    names: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    order: str = "ENZ"                      # kaip buvo faile
    note: str = ""

    def __len__(self):
        return len(self.xyz)


def _num(s):
    s = s.strip().replace(" ", "").replace(" ", "")
    if s.count(",") == 1 and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def _sniff_delim(text):
    lines = [l for l in text.splitlines()[:50] if l.strip()]
    best, score = None, -1
    for d in [";", "\t", ","]:
        counts = [l.count(d) for l in lines]
        if counts and min(counts) >= 2 and len(set(counts)) == 1 and counts[0] > score:
            best, score = d, counts[0]
    if best:
        return best
    for d in [";", "\t", ","]:
        if sum(l.count(d) for l in lines) >= 2 * len(lines):
            return d
    return None                             # tarpai


def _is_lks_e(v):
    return 100_000 < v < 800_000


def _is_lks_n(v):
    return 5_900_000 < v < 6_400_000


def read_points(path_or_text, order: str = "auto") -> PointSet:
    """order: 'auto' | 'ENZ' | 'NEZ' | 'PENZ' | 'PNEZ' (P = pavadinimas, pirmas stulpelis)."""
    if isinstance(path_or_text, (str, bytes)) and "\n" in (path_or_text if isinstance(path_or_text, str) else ""):
        text = path_or_text
    else:
        raw = open(path_or_text, "rb").read()
        for enc in ("utf-8-sig", "cp1257", "latin-1"):
            try:
                text = raw.decode(enc); break
            except UnicodeDecodeError:
                continue
    delim = _sniff_delim(text)
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "//")):
            continue
        parts = next(csv.reader([line], delimiter=delim)) if delim else line.split()
        parts = [p.strip() for p in parts]
        rows.append(parts)
    # stulpelių tipai
    parsed = []
    for r in rows:
        vals = []
        for p in r:
            try:
                vals.append(_num(p))
            except ValueError:
                vals.append(None)
        parsed.append((r, vals))
    data = [(r, v) for r, v in parsed if sum(x is not None for x in v) >= 3]
    if not data:
        raise ValueError("CSV: nerasta eilučių su bent 3 skaičiais")
    ncol = max(len(v) for _, v in data)
    numeric = [sum(1 for _, v in data if i < len(v) and v[i] is not None) / len(data) > 0.9 for i in range(ncol)]
    num_idx = [i for i, ok in enumerate(numeric) if ok]
    if order.startswith("P"):
        name_col, coords = 0, [i for i in num_idx if i > 0][:3]
    elif order in ("ENZ", "NEZ"):
        name_col, coords = None, num_idx[:3]
    elif num_idx and num_idx[0] > 0:            # pirmas stulpelis tekstinis -> pavadinimas
        name_col, coords = 0, num_idx[:3]
    elif len(num_idx) >= 4 and num_idx[0] == 0 and all(
            v[0] is not None and float(v[0]).is_integer() for _, v in data[:200]):
        name_col, coords = 0, num_idx[1:4]      # P,X,Y,Z su skaitiniu taško nr.
    else:
        name_col, coords = None, num_idx[:3]
    if len(coords) < 3:
        raise ValueError("CSV: reikia bent trijų koordinačių stulpelių")
    a = np.array([[v[c] if c < len(v) and v[c] is not None else np.nan for c in coords] for _, v in data])
    ok = ~np.isnan(a).any(1)
    a = a[ok]
    datarows = [r for (r, _), k in zip(data, ok) if k]
    names = [r[name_col] if name_col is not None and name_col < len(r) else str(i + 1) for i, r in enumerate(datarows)]
    after = coords[-1] + 1
    codes = [r[after] if after < len(r) else "" for r in datarows]
    file_order = order.replace("P", "") if order != "auto" else None
    note = ""
    if file_order is None:
        c0, c1 = a[:, 0], a[:, 1]
        if np.mean([_is_lks_n(x) for x in c0]) > 0.9 and np.mean([_is_lks_e(x) for x in c1]) > 0.9:
            file_order, note = "NEZ", "Atpažinta LKS-94 tvarka N,E,Z - ašys sukeistos"
        else:
            file_order = "ENZ"
    if file_order == "NEZ":
        a = a[:, [1, 0, 2]]
    return PointSet(a, names, codes, file_order, note)


def write_points(path, xyz, names=None, codes=None, order="PENZ", delim=",", decimals=3, header=None):
    xyz = np.asarray(xyz, float)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delim)
        if header:
            w.writerow(header)
        for i, p in enumerate(xyz):
            e, n, z = (f"{v:.{decimals}f}" for v in p[:3])
            c = [e, n, z] if order.replace("P", "") == "ENZ" else [n, e, z]
            row = ([names[i] if names else str(i + 1)] if order.startswith("P") else []) + c
            if codes:
                row.append(codes[i])
            w.writerow(row)

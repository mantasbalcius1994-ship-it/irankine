"""TIN paviršius ir tinkleliai (DEM).

Viduje koordinatės laikomos atėmus `origin` (LKS-94 skaičiai dideli -> tikslumas trianguliuojant)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import limits


class Surface:
    def __init__(self, pts, tris=None, name="Paviršius", max_edge: float | None = None):
        pts = np.asarray(pts, float)
        if len(pts) < 3:
            raise ValueError(f"{name}: per mažai taškų paviršiui")
        if len(pts) > limits.MAX_TIN_POINTS:
            raise ValueError(f"{name}: {len(pts):,} taškų - daugiau nei serverio riba {limits.MAX_TIN_POINTS:,}. "
                             "Naudokite retinimą arba tinklelį (DEM).".replace(",", " "))
        self.name = name
        self.origin = pts[:, :2].min(0)
        self.pts = pts
        loc = pts[:, :2] - self.origin
        if tris is None or len(tris) == 0:
            from scipy.spatial import Delaunay
            loc_u, idx = np.unique(np.round(loc, 4), axis=0, return_index=True)
            self.pts = pts = pts[idx]
            loc = pts[:, :2] - self.origin
            tris = Delaunay(loc).simplices
            if max_edge:
                e = np.stack([np.linalg.norm(loc[tris[:, i]] - loc[tris[:, (i + 1) % 3]], axis=1) for i in range(3)], 1)
                tris = tris[e.max(1) <= max_edge]
        self._finder = None
        self.tris = np.asarray(tris, int)
        self._loc = loc

    # ------------------------------------------------------------ pagrindai
    @classmethod
    def from_dict(cls, d):
        return cls(d["pts"], d["tris"], d.get("name", "Paviršius"))

    @property
    def bbox(self):
        return (*self.pts[:, :2].min(0), *self.pts[:, :2].max(0))

    def _tri(self):
        if self._finder is None:
            self._finder = ("grid", None, _GridLocator(self._loc, self.tris))
        return self._finder

    def _locate(self, q):
        return self._tri()[2].locate(q, self._bary)

    def _bary(self, q, ti):
        a, b, c = (self._loc[self.tris[ti, i]] for i in range(3))
        v0, v1, v2 = b - a, c - a, q - a
        d = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        d = np.where(np.abs(d) < 1e-18, np.nan, d)
        wb = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / d
        wc = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / d
        return np.column_stack([1 - wb - wc, wb, wc])

    def z_at(self, xy, chunk=500_000):
        """Aukštis taškuose (n,2). Už paviršiaus ribų -> NaN."""
        xy = np.atleast_2d(np.asarray(xy, float))
        if len(xy) > 10_000:                 # erdvinis rūšiavimas: paieška daug greitesnė
            c = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9) / 512
            o = np.lexsort((np.floor(xy[:, 0] / c), np.floor(xy[:, 1] / c)))
            res = np.empty(len(xy))
            res[o] = self._z_at(xy[o], chunk)
            return res
        return self._z_at(xy, chunk)

    def _z_at(self, xy, chunk):
        out = np.full(len(xy), np.nan)
        for i in range(0, len(xy), chunk):
            q = xy[i:i + chunk, :2] - self.origin
            ti = self._locate(q)
            m = ti >= 0
            if m.any():
                w = self._bary(q[m], ti[m])
                z = self.pts[self.tris[ti[m]], 2]
                out[i:i + chunk][m] = (w * z).sum(1)
        return out

    # ------------------------------------------------------------ geometrija
    def boundary(self) -> list[np.ndarray]:
        """Išoriniai kraštai kaip uždari žiedai (XY)."""
        e = np.sort(np.concatenate([self.tris[:, [0, 1]], self.tris[:, [1, 2]], self.tris[:, [2, 0]]]), 1)
        u, c = np.unique(e, axis=0, return_counts=True)
        edges = u[c == 1]
        nxt: dict[int, list[int]] = {}
        for a, b in edges:
            nxt.setdefault(a, []).append(b); nxt.setdefault(b, []).append(a)
        rings, seen = [], set()
        for a, b in edges:
            if (a, b) in seen:
                continue
            ring, prev, cur = [a], a, b
            seen.add((a, b))
            while cur != a and len(ring) <= len(edges):
                ring.append(cur)
                cands = [n for n in nxt[cur] if n != prev and (min(cur, n), max(cur, n)) not in seen]
                if not cands:
                    break
                prev, cur = cur, cands[0]
                seen.add((min(prev, cur), max(prev, cur)))
            rings.append(self.pts[ring, :2])
        return rings

    def area_2d(self):
        a, b, c = (self._loc[self.tris[:, i]] for i in range(3))
        u, v = b - a, c - a
        return float(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum() / 2)

    def to_dict(self):
        return {"name": self.name, "pts": self.pts, "tris": self.tris}


# ------------------------------------------------------------------ tinkleliai
@dataclass
class Grid:
    x0: float           # pirmo langelio centro X
    y0: float
    step: float
    z: np.ndarray       # (ny, nx), NaN = nėra duomenų
    name: str = ""

    @property
    def shape(self):
        return self.z.shape

    def centers(self):
        ny, nx = self.z.shape
        X, Y = np.meshgrid(self.x0 + np.arange(nx) * self.step, self.y0 + np.arange(ny) * self.step)
        return X, Y

    @property
    def cell_area(self):
        return self.step * self.step


def grid_from_points(xyz, step, stat="mean", bounds=None, name="") -> Grid:
    """Taškų debesis -> tinklelis (langelio statistika: mean | min | max | median | count)."""
    xyz = np.asarray(xyz, float)
    if bounds is None:
        x0, y0 = np.floor(xyz[:, :2].min(0) / step) * step
        x1, y1 = np.ceil(xyz[:, :2].max(0) / step) * step
    else:
        x0, y0, x1, y1 = bounds
    nx, ny = max(1, int(round((x1 - x0) / step))), max(1, int(round((y1 - y0) / step)))
    ix = np.floor((xyz[:, 0] - x0) / step).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y0) / step).astype(np.int64)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, z = ix[m], iy[m], xyz[m, 2]
    lin = iy * nx + ix
    out = np.full(nx * ny, np.nan)
    if stat == "count":
        out = np.bincount(lin, minlength=nx * ny).astype(float)
    elif stat == "mean":
        s = np.bincount(lin, weights=z, minlength=nx * ny); c = np.bincount(lin, minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(c > 0, s / c, np.nan)
    elif stat in ("min", "max"):
        o = np.argsort(lin, kind="stable")
        lin_s, z_s = lin[o], z[o]
        starts = np.r_[0, np.nonzero(np.diff(lin_s))[0] + 1]
        red = np.minimum.reduceat(z_s, starts) if stat == "min" else np.maximum.reduceat(z_s, starts)
        out[lin_s[starts]] = red
    elif stat == "median":
        o = np.lexsort((z, lin))
        lin_s, z_s = lin[o], z[o]
        starts = np.r_[0, np.nonzero(np.diff(lin_s))[0] + 1]
        ends = np.r_[starts[1:], len(lin_s)]
        mid_lo, mid_hi = starts + (ends - starts - 1) // 2, starts + (ends - starts) // 2
        out[lin_s[starts]] = (z_s[mid_lo] + z_s[mid_hi]) / 2
    else:
        raise ValueError(f"Nežinoma statistika: {stat}")
    return Grid(x0 + step / 2, y0 + step / 2, step, out.reshape(ny, nx), name)


def grid_from_surface(surf: Surface, step, bounds=None, name="") -> Grid:
    if bounds is None:
        b = surf.bbox
        bounds = (math.floor(b[0] / step) * step, math.floor(b[1] / step) * step,
                  math.ceil(b[2] / step) * step, math.ceil(b[3] / step) * step)
    x0, y0, x1, y1 = bounds
    nx, ny = max(1, int(round((x1 - x0) / step))), max(1, int(round((y1 - y0) / step)))
    X, Y = np.meshgrid(x0 + step / 2 + np.arange(nx) * step, y0 + step / 2 + np.arange(ny) * step)
    z = surf.z_at(np.column_stack([X.ravel(), Y.ravel()])).reshape(ny, nx)
    return Grid(x0 + step / 2, y0 + step / 2, step, z, name or surf.name)


def polygon_mask(grid: Grid, rings) -> np.ndarray:
    """Langeliai, kurių centrai yra poligone (rings: [išorė, skylės...] XY masyvai)."""
    import shapely
    poly = shapely.Polygon(rings[0], rings[1:])
    X, Y = grid.centers()
    return shapely.contains_xy(poly, X, Y)


@dataclass
class VolumeResult:
    cut: float          # m³, kur `top` žemiau `base` (iškasa)
    fill: float         # m³, kur `top` aukščiau `base` (pylimas)
    area: float         # m², kur abu paviršiai turi duomenis
    dz: Grid            # top - base

    @property
    def net(self):
        return self.fill - self.cut


def volume_between(base: Grid, top: Grid, mask=None, dz_ignore=0.0) -> VolumeResult:
    """Tūris tarp dviejų vienodo tinklelio paviršių. dz_ignore - mažesni skirtumai neskaičiuojami (triukšmas)."""
    if base.shape != top.shape or abs(base.x0 - top.x0) > 1e-9 or abs(base.y0 - top.y0) > 1e-9:
        raise ValueError("Tinkleliai turi sutapti (ta pati kilmė, žingsnis, dydis)")
    dz = top.z - base.z
    ok = ~np.isnan(dz)
    if mask is not None:
        ok &= mask
    d = np.where(ok, dz, 0.0)
    d = np.where(np.abs(d) < dz_ignore, 0.0, d)
    A = base.cell_area
    return VolumeResult(cut=float(-d[d < 0].sum() * A), fill=float(d[d > 0].sum() * A),
                        area=float(ok.sum() * A), dz=Grid(base.x0, base.y0, base.step, np.where(ok, dz, np.nan), "dz"))


def fit_offset(ref: Surface, xyz, mode="dz", robust=True, mask=None):
    """Lokalus `ref` pritaikymas prie išmatuotų taškų (pvz. GPS aukščio poslinkis).
    mode: 'dz' - tik vertikalus poslinkis; 'plane' - dz + nuolydis (a + b*x + c*y).
    Grąžina (parametrai, funkcija f(xy)->dz pataisa, liekamieji)."""
    xyz = np.asarray(xyz, float)
    zr = ref.z_at(xyz[:, :2])
    ok = ~np.isnan(zr)
    if mask is not None:
        ok &= mask
    if ok.sum() < 3:
        raise ValueError("Per mažai taškų ant projektinio paviršiaus")
    r = xyz[ok, 2] - zr[ok]
    c = xyz[ok, :2].mean(0)
    A = np.ones((ok.sum(), 1)) if mode == "dz" else np.column_stack([np.ones(ok.sum()), xyz[ok, :2] - c])
    w = np.ones(len(r))
    for _ in range(10 if robust else 1):        # IRLS (Huber) - atmeta išplovas / sankaupas
        p, *_ = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)
        res = r - A @ p
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
        u = np.abs(res) / (1.5 * s)
        w = np.sqrt(np.where(u <= 1, 1.0, 1.0 / u))

    def corr(xy):
        xy = np.atleast_2d(xy)[:, :2]
        return np.full(len(xy), p[0]) if mode == "dz" else p[0] + (xy - c) @ p[1:]
    full = np.full(len(xyz), np.nan)
    full[ok] = xyz[ok, 2] - (zr[ok] + corr(xyz[ok]))
    return p, corr, full


class _GridLocator:
    """Trikampių indeksas tinklelyje: veikia ir su nedelaunay / persidengiančiais trikampiais."""

    def __init__(self, loc, tris, max_cells=4_000_000):
        P = loc[tris]                                   # (m,3,2)
        lo, hi = P.min(1), P.max(1)
        self.o = lo.min(0)
        ext = np.maximum(hi.max(0) - self.o, 1e-9)
        m = len(tris)
        c = math.sqrt(ext[0] * ext[1] / max(m, 1)) * 1.5
        c = max(c, math.sqrt(ext[0] * ext[1] / max_cells))
        self.c = c
        self.nx = int(ext[0] // c) + 1
        self.ny = int(ext[1] // c) + 1
        i0 = np.floor((lo - self.o) / c).astype(np.int64)
        i1 = np.floor((hi - self.o) / c).astype(np.int64)
        w = i1[:, 0] - i0[:, 0] + 1
        h = i1[:, 1] - i0[:, 1] + 1
        cnt = w * h
        tid = np.repeat(np.arange(m), cnt)
        k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        cx = i0[tid, 0] + k % w[tid]
        cy = i0[tid, 1] + k // w[tid]
        cell = cy * self.nx + cx
        o = np.argsort(cell, kind="stable")
        self.tid = tid[o]
        self.start = np.searchsorted(cell[o], np.arange(self.nx * self.ny + 1))

    def locate(self, q, bary):
        ij = np.floor((q - self.o) / self.c).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < self.ny)
        cell = np.where(ok, ij[:, 1] * self.nx + ij[:, 0], 0)
        a, b = self.start[cell], self.start[cell + 1]
        n = np.where(ok, b - a, 0)
        out = np.full(len(q), -1)
        j = 0
        todo = np.nonzero(n > 0)[0]
        while len(todo):
            ti = self.tid[a[todo] + j]
            w = bary(q[todo], ti)
            hit = (w >= -1e-9).all(1)
            out[todo[hit]] = ti[hit]
            j += 1
            todo = todo[~hit & (n[todo] > j)]
        return out

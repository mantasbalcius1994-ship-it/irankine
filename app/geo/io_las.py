"""LAS / LAZ taškų debesys. Skaitoma dalimis, kad dideli failai tilptų į serverio atmintį."""
from __future__ import annotations

import numpy as np


def info(path) -> dict:
    import laspy
    with laspy.open(str(path)) as f:
        h = f.header
        return {"points": int(h.point_count), "version": str(h.version), "format": int(h.point_format.id),
                "min": tuple(h.mins), "max": tuple(h.maxs), "scale": tuple(h.scales),
                "crs": (h.parse_crs().name if h.parse_crs() else None) if hasattr(h, "parse_crs") else None}


def read(path, bounds=None, polygon=None, classes=None, thin=None, thin_mode="min",
         chunk=2_000_000, max_points=None, progress=None) -> np.ndarray:
    """-> (n,3) XYZ.
    bounds: (x0,y0,x1,y1); polygon: shapely Polygon (tikslus apkirpimas);
    classes: pvz. [2] (žemė); thin: tinklelio žingsnis m - paliekamas vienas taškas langelyje
    (thin_mode 'min' = žemiausias, 'max', 'first')."""
    import laspy
    from . import limits
    max_points = max_points or limits.MAX_CLOUD_POINTS
    parts = []
    total = 0
    with laspy.open(str(path)) as f:
        n = int(f.header.point_count)
        done = 0
        for pts in f.chunk_iterator(chunk):
            x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
            m = np.ones(len(x), bool)
            if bounds is not None:
                m &= (x >= bounds[0]) & (y >= bounds[1]) & (x <= bounds[2]) & (y <= bounds[3])
            if classes is not None:
                m &= np.isin(np.asarray(pts.classification), classes)
            if polygon is not None and m.any():
                import shapely
                mm = np.zeros(len(x), bool)
                mm[m] = shapely.contains_xy(polygon, x[m], y[m])
                m = mm
            xyz = np.column_stack([x[m], y[m], z[m]])
            if thin:
                xyz = _thin(xyz, thin, thin_mode)
            parts.append(xyz)
            total += len(xyz)
            done += len(x)
            if progress:
                progress(done / max(n, 1))
            if total > max_points:
                raise ValueError(f"Per daug taškų (> {max_points:,}). Naudokite retinimą arba apkirpimą.".replace(",", " "))
    xyz = np.vstack(parts) if parts else np.zeros((0, 3))
    if thin and len(parts) > 1:
        xyz = _thin(xyz, thin, thin_mode)        # langeliai ant dalių ribų
    return xyz


def _thin(xyz, step, mode):
    if not len(xyz):
        return xyz
    k = np.floor(xyz[:, :2] / step).astype(np.int64)
    key = k[:, 0] * 10_000_019 + k[:, 1]
    if mode == "first":
        _, idx = np.unique(key, return_index=True)
        return xyz[idx]
    o = np.lexsort((xyz[:, 2] if mode == "min" else -xyz[:, 2], key))
    ks = key[o]
    first = np.r_[True, ks[1:] != ks[:-1]]
    return xyz[o[first]]


def write(path, xyz, scale=0.001, classification=None):
    import laspy
    xyz = np.asarray(xyz, float)
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = np.array([scale] * 3)
    h.offsets = np.floor(xyz.min(0)) if len(xyz) else np.zeros(3)
    las = laspy.LasData(h)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if classification is not None:
        las.classification = np.asarray(classification, np.uint8)
    las.write(str(path))

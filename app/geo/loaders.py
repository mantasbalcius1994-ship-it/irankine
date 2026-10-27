"""Universalus įkėlimas: paviršiai (TTM, LandXML, DXF 3DFACE) ir taškai (LAS/LAZ, CSV, DXF, paviršių viršūnės)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .surface import Surface

SURFACE_EXT = (".ttm", ".xml", ".dxf")
POINTS_EXT = (".las", ".laz", ".csv", ".txt", ".dxf", ".ttm", ".xml")


def load_surface(path, name=None) -> Surface:
    """Paviršius iš TTM / LandXML / DXF (3DFACE, POLYFACE). LandXML su keliais paviršiais - imamas `name` arba pirmas."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".ttm":
        from . import io_ttm
        d = io_ttm.read(p)
        return Surface(d["pts"], d["tris"], d["name"])
    if ext == ".xml":
        from . import io_landxml
        ss = io_landxml.read_surfaces(p, [name] if name else None)
        return Surface.from_dict(ss[0])
    if ext == ".dxf":
        from . import io_dxf
        pts, tris = io_dxf.read_faces(p)
        if not len(tris):
            raise ValueError(f"{p.name}: DXF faile nėra 3DFACE / POLYFACE paviršiaus. "
                             "Eksportuokite paviršių iš TBC kaip LandXML arba TTM.")
        return Surface(pts, tris, p.stem)
    raise ValueError(f"{p.name}: paviršiui tinka TTM, LandXML (.xml) arba DXF su 3DFACE")


def load_points(path, thin=None, bounds=None, log=None) -> tuple[np.ndarray | None, Surface | None]:
    """Faktiniai duomenys. Grąžina (taškai (n,3), None) arba (None, Surface), jei tai paviršius (TTM/LandXML)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".las", ".laz"):
        from . import io_las
        return io_las.read(p, bounds=bounds, thin=thin, thin_mode="first"), None
    if ext in (".csv", ".txt"):
        from . import io_csv
        ps = io_csv.read_points(p)
        if log and ps.note:
            log(f"  {p.name}: {ps.note}")
        return ps.xyz, None
    if ext == ".dxf":
        from . import io_dxf
        pts, tris = io_dxf.read_faces(p)
        if len(tris):
            return None, Surface(pts, tris, p.stem)
        xyz, _ = io_dxf.read_points(p)
        lines = [r.pts for r in io_dxf.read_polylines(p)]
        allp = np.vstack([xyz] + lines) if lines or len(xyz) else np.zeros((0, 3))
        if not len(allp):
            raise ValueError(f"{p.name}: DXF nėra taškų, linijų ar paviršiaus")
        return allp, None
    if ext in (".ttm", ".xml"):
        return None, load_surface(p)
    raise ValueError(f"{p.name}: nepalaikomas formatas (tinka LAS/LAZ, CSV, DXF, TTM, LandXML)")


def load_surfaces(path) -> list[Surface]:
    """Visi faile esantys paviršiai (LandXML gali turėti kelis)."""
    p = Path(path)
    if p.suffix.lower() == ".xml":
        from . import io_landxml
        return [Surface.from_dict(d) for d in io_landxml.read_surfaces(p) if len(d["tris"])]
    return [load_surface(p)]


def load_point_set(path, log=None):
    """Matavimo taškai su pavadinimais ir kodais: CSV/TXT (P,E,N,Z,kodas) arba DXF POINT (kodas = sluoksnis).
    -> (xyz (n,3), names, codes)"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".csv", ".txt"):
        from . import io_csv
        ps = io_csv.read_points(p)
        if log and ps.note:
            log(f"  {p.name}: {ps.note}")
        return ps.xyz, list(ps.names), list(ps.codes)
    if ext == ".dxf":
        from . import io_dxf
        xyz, lay = io_dxf.read_points(p, include_inserts=True)
        if not len(xyz):
            raise ValueError(f"{p.name}: DXF nėra taškų (POINT / blokų). Taškus eksportuokite iš TBC kaip CSV.")
        return xyz, [str(i + 1) for i in range(len(xyz))], list(lay)
    raise ValueError(f"{p.name}: taškams tinka CSV / TXT arba DXF")

"""DXF skaitymas ir rašymas (ezdxf). Koordinatės: X = rytai (E), Y = šiaurė (N)."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

LINE_TYPES = ("LWPOLYLINE", "POLYLINE", "LINE", "ARC", "SPLINE")


@dataclass
class Polyline:
    layer: str
    handle: str
    pts: np.ndarray                 # (n, 3)
    closed: bool = False
    kind: str = "POLYLINE"

    @property
    def name(self):
        return f"{self.layer}_{self.handle}"


class DxfError(ValueError):
    pass


def _open(path):
    import ezdxf
    from ezdxf import recover
    try:
        return ezdxf.readfile(str(path))
    except IOError as e:
        raise DxfError(f"{os.path.basename(str(path))}: ne DXF failas arba nepavyko atidaryti ({e})")
    except ezdxf.DXFStructureError:
        doc, auditor = recover.readfile(str(path))    # bandoma atkurti pažeistą failą
        return doc


def _layer_ok(e, layers):
    return not layers or e.dxf.layer in layers


def read_polylines(path, layers=None, types=("LWPOLYLINE", "POLYLINE"), flatten=0.002) -> list[Polyline]:
    """Linijos kaip taškų masyvai. Lankai (bulge, ARC, SPLINE) išskaidomi su `flatten` m chordos paklaida."""
    from ezdxf import path as ezpath
    doc = _open(path)
    out = []
    for e in doc.modelspace():
        t = e.dxftype()
        if t not in types or not _layer_ok(e, layers):
            continue
        if t == "POLYLINE" and (e.is_poly_face_mesh or e.is_polygon_mesh):
            continue
        try:
            p = ezpath.make_path(e)
            pts = np.array([[v.x, v.y, v.z] for v in p.flattening(flatten)])
        except Exception:
            continue
        if len(pts) < 2:
            continue
        closed = bool(getattr(e, "closed", False) or (t == "POLYLINE" and e.is_closed))
        if closed and np.linalg.norm(pts[0, :2] - pts[-1, :2]) < 1e-9:
            pts = pts[:-1]
        out.append(Polyline(e.dxf.layer, e.dxf.handle, pts, closed, t))
    return out


def read_points(path, layers=None, include_inserts=False) -> tuple[np.ndarray, list[str]]:
    """POINT (ir pasirinktinai INSERT) objektai -> (n,3) masyvas, sluoksniai."""
    doc = _open(path)
    xyz, lay = [], []
    for e in doc.modelspace():
        t = e.dxftype()
        if not _layer_ok(e, layers):
            continue
        if t == "POINT":
            v = e.dxf.location
        elif t == "INSERT" and include_inserts:
            v = e.dxf.insert
        else:
            continue
        xyz.append((v.x, v.y, v.z)); lay.append(e.dxf.layer)
    return np.array(xyz, float).reshape(-1, 3), lay


def read_faces(path, layers=None) -> tuple[np.ndarray, np.ndarray]:
    """3DFACE ir POLYFACE mesh -> (taškai (n,3), trikampiai (m,3)). Sutampančios viršūnės sujungiamos."""
    doc = _open(path)
    tris = []
    for e in doc.modelspace():
        if not _layer_ok(e, layers):
            continue
        t = e.dxftype()
        if t == "3DFACE":
            v = [e.dxf.vtx0, e.dxf.vtx1, e.dxf.vtx2, e.dxf.vtx3]
            tris.append([tuple(v[0]), tuple(v[1]), tuple(v[2])])
            if not v[3].isclose(v[2]):
                tris.append([tuple(v[0]), tuple(v[2]), tuple(v[3])])
        elif t == "POLYLINE" and e.is_poly_face_mesh:
            from ezdxf.render import MeshBuilder
            m = MeshBuilder.from_polyface(e)
            for f in m.faces:
                vv = [tuple(m.vertices[i]) for i in f]
                for k in range(1, len(vv) - 1):
                    tris.append([vv[0], vv[k], vv[k + 1]])
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    allv = np.array(tris, float).reshape(-1, 3)
    key = np.round(allv, 4)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    pts = np.zeros((len(uniq), 3))
    pts[inv] = allv
    faces = inv.reshape(-1, 3)
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return pts, faces[good]


def layers_in(path) -> dict[str, int]:
    """Sluoksniai ir objektų skaičius - naudinga klaidų pranešimams."""
    doc = _open(path)
    cnt: dict[str, int] = {}
    for e in doc.modelspace():
        cnt[e.dxf.layer] = cnt.get(e.dxf.layer, 0) + 1
    return cnt


class DxfWriter:
    """Paprastas DXF rašytojas su sluoksnių spalvomis (ACI)."""

    def __init__(self, version="R2013"):
        import ezdxf
        self.doc = ezdxf.new(version, setup=True)
        self.msp = self.doc.modelspace()

    def layer(self, name, color=7):
        if name not in self.doc.layers:
            self.doc.layers.add(name, color=color)
        return name

    def _a(self, layer, color):
        a = {"layer": self.layer(layer)}
        if color is not None:
            a["color"] = color
        return a

    def polyline3d(self, pts, layer, color=None, closed=False):
        e = self.msp.add_polyline3d([tuple(map(float, p[:3])) for p in pts], dxfattribs=self._a(layer, color))
        if closed:
            e.close(True)
        return e

    def lwpolyline(self, pts, layer, color=None, closed=False, elevation=0.0):
        e = self.msp.add_lwpolyline([(float(p[0]), float(p[1])) for p in pts], close=closed, dxfattribs=self._a(layer, color))
        e.dxf.elevation = float(elevation)
        return e

    def line(self, a, b, layer, color=None):
        return self.msp.add_line(tuple(map(float, a)), tuple(map(float, b)), dxfattribs=self._a(layer, color))

    def arc(self, center, radius, a0_deg, a1_deg, layer, color=None):
        return self.msp.add_arc(tuple(map(float, center)), float(radius), a0_deg, a1_deg, dxfattribs=self._a(layer, color))

    def circle(self, center, radius, layer, color=None):
        return self.msp.add_circle(tuple(map(float, center)), float(radius), dxfattribs=self._a(layer, color))

    def point(self, p, layer, color=None):
        return self.msp.add_point(tuple(map(float, p)), dxfattribs=self._a(layer, color))

    def text(self, s, p, height, layer, color=None, rotation=0.0):
        t = self.msp.add_text(str(s), height=height, rotation=rotation, dxfattribs=self._a(layer, color))
        t.set_placement(tuple(map(float, p[:2])))
        if len(p) > 2:
            t.dxf.insert = tuple(map(float, p[:3]))
        return t

    def faces(self, pts, tris, layer, color=None):
        a = self._a(layer, color)
        for t in tris:
            v = [tuple(map(float, pts[i])) for i in t]
            self.msp.add_3dface([v[0], v[1], v[2], v[2]], dxfattribs=a)

    def save(self, path):
        self.doc.saveas(str(path))

"""LandXML 1.0/1.1/1.2: paviršiai (TIN) ir trasos (planas + profilis).
LandXML koordinatės rašomos "N E [Z]"; skaitant LKS-94 atveju sukeista tvarka atpažįstama automatiškai."""
from __future__ import annotations

import datetime as dt
import math
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

from .alignment import Alignment, HElem, Profile


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _iter(el, name):
    return (c for c in el.iter() if _local(c.tag) == name)


def _ne(text, swap):
    v = [float(x) for x in text.split()]
    n, e = (v[1], v[0]) if swap else (v[0], v[1])
    return np.array([e, n] + v[2:3])        # -> X=E, Y=N, [Z]


def _detect_swap(root):
    """Ar faile vietoj 'N E' parašyta 'E N' (LKS-94: E < 1e6 < N)."""
    for el in root.iter():
        if _local(el.tag) in ("P", "Start", "End", "Center", "PI") and el.text and el.text.strip():
            v = [float(x) for x in el.text.split()[:2]]
            if len(v) == 2:
                return v[0] < 1_000_000 < v[1]
    return False


def _parse(path):
    try:
        return ET.parse(str(path)).getroot()
    except ET.ParseError as e:
        raise ValueError(f"LandXML: netinkamas XML ({e})")


# ------------------------------------------------------------------ paviršiai
def read_surfaces(path, names=None) -> list[dict]:
    """-> [{'name', 'pts' (n,3) X=E Y=N, 'tris' (m,3)}]. Paslėpti trikampiai (i="1") praleidžiami."""
    root = _parse(path)
    swap = _detect_swap(root)
    out = []
    for s in _iter(root, "Surface"):
        name = s.get("name", f"Surface{len(out) + 1}")
        if names and name not in names:
            continue
        d = _child(s, "Definition")
        if d is None:
            continue
        pn = _child(d, "Pnts")
        fc = _child(d, "Faces")
        if pn is None:
            continue
        ids, xyz = [], []
        for p in _children(pn, "P"):
            v = [float(x) for x in p.text.split()]
            n, e = (v[1], v[0]) if swap else (v[0], v[1])
            ids.append(p.get("id")); xyz.append((e, n, v[2] if len(v) > 2 else np.nan))
        index = {k: i for i, k in enumerate(ids)}
        tris = []
        if fc is not None:
            for f in _children(fc, "F"):
                if f.get("i") == "1":
                    continue
                try:
                    tris.append([index[k] for k in f.text.split()[:3]])
                except KeyError:
                    continue
        out.append({"name": name, "pts": np.array(xyz, float).reshape(-1, 3),
                    "tris": np.array(tris, int).reshape(-1, 3)})
    if not out:
        raise ValueError("LandXML: nerasta paviršių (Surface)")
    return out


def write_surfaces(path, surfaces, app="Irankine"):
    """surfaces: [{'name', 'pts', 'tris'}] (X=E, Y=N)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_header(app))
        fh.write('  <Surfaces>\n')
        for s in surfaces:
            fh.write(f'    <Surface name="{escape(s["name"])}">\n      <Definition surfType="TIN">\n        <Pnts>\n')
            for i, p in enumerate(s["pts"]):
                fh.write(f'          <P id="{i + 1}">{p[1]:.4f} {p[0]:.4f} {p[2]:.4f}</P>\n')
            fh.write('        </Pnts>\n        <Faces>\n')
            for t in s["tris"]:
                fh.write(f'          <F>{t[0] + 1} {t[1] + 1} {t[2] + 1}</F>\n')
            fh.write('        </Faces>\n      </Definition>\n    </Surface>\n')
        fh.write('  </Surfaces>\n</LandXML>\n')


# ------------------------------------------------------------------ trasos
def read_alignments(path, names=None) -> list[Alignment]:
    root = _parse(path)
    swap = _detect_swap(root)
    out = []
    for a in _iter(root, "Alignment"):
        name = a.get("name", f"Alignment{len(out) + 1}")
        if names and name not in names:
            continue
        cg = _child(a, "CoordGeom")
        if cg is None:
            continue
        elems, s, prev_th = [], 0.0, None
        for g in cg:
            kind = _local(g.tag)
            if kind not in ("Line", "Curve", "Spiral"):
                continue
            st = _ne(_child(g, "Start").text, swap)[:2]
            en = _ne(_child(g, "End").text, swap)[:2]
            if kind == "Line":
                L = float(g.get("length") or np.linalg.norm(en - st))
                th = math.atan2(en[1] - st[1], en[0] - st[0])
                e = HElem("line", st, th, L, 0, 0, s)
            elif kind == "Curve":
                c = _ne(_child(g, "Center").text, swap)[:2]
                R = float(g.get("radius") or np.linalg.norm(st - c))
                sg = 1.0 if g.get("rot", "ccw") == "ccw" else -1.0
                r = (st - c) / np.linalg.norm(st - c)
                tang = sg * np.array([-r[1], r[0]])
                th = math.atan2(tang[1], tang[0])
                if g.get("length"):
                    L = float(g.get("length"))
                else:
                    a0 = math.atan2(*(st - c)[::-1]); a1 = math.atan2(*(en - c)[::-1])
                    L = R * ((sg * (a1 - a0)) % (2 * math.pi))
                e = HElem("arc", st, th, L, sg / R, sg / R, s)
            else:
                L = float(g.get("length"))
                sg = 1.0 if g.get("rot", "ccw") == "ccw" else -1.0
                rs, re_ = (g.get("radiusStart") or "INF"), (g.get("radiusEnd") or "INF")
                k0 = 0.0 if rs.upper().startswith("INF") else sg / float(rs)
                k1 = 0.0 if re_.upper().startswith("INF") else sg / float(re_)
                pi = _child(g, "PI")
                if pi is not None:
                    pv = _ne(pi.text, swap)[:2]
                    th = math.atan2(pv[1] - st[1], pv[0] - st[0])
                else:
                    th = prev_th if prev_th is not None else 0.0
                e = HElem("spiral", st, th, L, k0, k1, s)
            elems.append(e); s += e.L; prev_th = e.th_end()
        if not elems:
            continue
        al = Alignment(name, elems, float(a.get("staStart", 0) or 0))
        pr = _child(a, "Profile")
        if pr is not None:
            pa = _child(pr, "ProfAlign")
            if pa is not None:
                S, E, Lc = [], [], []
                for v in pa:
                    t = _local(v.tag)
                    if t not in ("PVI", "ParaCurve", "CircCurve", "UnsymParaCurve"):
                        continue
                    x = [float(q) for q in v.text.split()]
                    S.append(x[0]); E.append(x[1])
                    if t == "UnsymParaCurve":
                        Lc.append(float(v.get("lengthIn", 0)) + float(v.get("lengthOut", 0)))
                    else:
                        Lc.append(float(v.get("length", 0)) if t != "PVI" else 0.0)
                if len(S) >= 2:
                    al.profile = Profile(np.array(S), np.array(E), np.array(Lc), pa.get("name", name))
        out.append(al)
    if not out:
        raise ValueError("LandXML: nerasta trasų (Alignment)")
    return out


def _fne(p):
    return f"{p[1]:.4f} {p[0]:.4f}"


def write_alignments(path, alignments: list[Alignment], app="Irankine"):
    L = [_header(app).rstrip("\n"), '  <Alignments name="Irankine">']
    for al in alignments:
        L.append(f'    <Alignment name="{escape(al.name)}" length="{al.length:.4f}" staStart="{al.sta_start:.4f}">')
        L.append('      <CoordGeom>')
        for e in al.elems:
            a, b = e.p, e.end()
            if e.kind == "line":
                L.append(f'        <Line length="{e.L:.4f}"><Start>{_fne(a)}</Start><End>{_fne(b)}</End></Line>')
            elif e.kind == "arc":
                k = e.k0
                c = a + (1 / k) * np.array([-math.sin(e.th), math.cos(e.th)])
                rot = "ccw" if k > 0 else "cw"
                delta = abs(k * e.L)
                pi = ""
                if delta < math.pi * 0.999:
                    T = abs(1 / k) * math.tan(delta / 2)
                    pi = f"<PI>{_fne(a + T * np.array([math.cos(e.th), math.sin(e.th)]))}</PI>"
                L.append(f'        <Curve rot="{rot}" crvType="arc" length="{e.L:.4f}" radius="{abs(1 / k):.4f}">'
                         f'<Start>{_fne(a)}</Start><Center>{_fne(c)}</Center><End>{_fne(b)}</End>{pi}</Curve>')
            else:
                rot = "ccw" if (e.k0 + e.k1) > 0 else "cw"
                rs = "INF" if abs(e.k0) < 1e-15 else f"{abs(1 / e.k0):.4f}"
                rend = "INF" if abs(e.k1) < 1e-15 else f"{abs(1 / e.k1):.4f}"
                # PI - pradžios ir pabaigos liestinių sankirta
                t0 = np.array([math.cos(e.th), math.sin(e.th)])
                t1 = np.array([math.cos(e.th_end()), math.sin(e.th_end())])
                den = t0[0] * t1[1] - t0[1] * t1[0]
                pi = ""
                if abs(den) > 1e-12:
                    u = ((b - a)[0] * t1[1] - (b - a)[1] * t1[0]) / den
                    pi = f"<PI>{_fne(a + u * t0)}</PI>"
                L.append(f'        <Spiral length="{e.L:.4f}" radiusStart="{rs}" radiusEnd="{rend}" rot="{rot}" '
                         f'spiType="clothoid"><Start>{_fne(a)}</Start>{pi}<End>{_fne(b)}</End></Spiral>')
        L.append('      </CoordGeom>')
        if al.profile is not None:
            pr = al.profile
            L.append(f'      <Profile name="{escape(al.name)}"><ProfAlign name="{escape(pr.name or al.name)}">')
            for i in range(len(pr.S)):
                if 0 < i < len(pr.S) - 1 and pr.Lc[i] > 1e-6:
                    L.append(f'        <ParaCurve length="{pr.Lc[i]:.4f}">{pr.S[i]:.4f} {pr.E[i]:.4f}</ParaCurve>')
                else:
                    L.append(f'        <PVI>{pr.S[i]:.4f} {pr.E[i]:.4f}</PVI>')
            L.append('      </ProfAlign></Profile>')
        L.append('    </Alignment>')
    L += ['  </Alignments>', '</LandXML>']
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def _header(app):
    now = dt.datetime.now()
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2" date="{now:%Y-%m-%d}" time="{now:%H:%M:%S}">\n'
            '  <Units><Metric linearUnit="meter" areaUnit="squareMeter" volumeUnit="cubicMeter" '
            'angularUnit="decimal degrees" directionUnit="decimal degrees"/></Units>\n'
            f'  <Application name="{escape(app)}" version="1.0"/>\n')

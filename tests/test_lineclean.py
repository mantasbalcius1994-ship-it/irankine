"""Linijų valymo ir retinimo testai.  python tests/test_lineclean.py"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.geo import io_dxf, linework, polyline  # noqa: E402
from app.geo.surface import Surface  # noqa: E402

rng = np.random.default_rng(11)
E0, N0 = 520000.0, 6070000.0
TMP = Path(tempfile.mkdtemp())


def make_surface():
    xy = rng.uniform([E0 - 20, N0 - 20], [E0 + 320, N0 + 220], (25000, 2))
    z = 100 + 1.5 * np.sin((xy[:, 0] - E0) / 25) + np.cos((xy[:, 1] - N0) / 40) + 0.01 * (xy[:, 0] - E0)
    return Surface(np.column_stack([xy, z]))


def drape(surf, P2):
    """Kaip TBC: viršūnė kiekvienoje TIN krašto sankirtoje + originalios viršūnės."""
    import shapely
    loc = surf.pts[:, :2]
    e = np.sort(np.concatenate([surf.tris[:, [0, 1]], surf.tris[:, [1, 2]], surf.tris[:, [2, 0]]]), 1)
    e = np.unique(e, axis=0)
    edges = shapely.MultiLineString([loc[a].tolist() for a in e])
    line = shapely.LineString(P2)
    inter = line.intersection(edges)
    pts = np.array([[g.x, g.y] for g in getattr(inter, "geoms", [inter]) if g.geom_type == "Point"])
    allp = np.vstack([P2, pts])
    s = shapely.line_locate_point(line, shapely.points(allp))
    allp = allp[np.argsort(s)]
    allp = polyline.dedupe(allp, 1e-7)
    return np.column_stack([allp, surf.z_at(allp)])


def road_line():
    t = np.linspace(0, 1, 40)
    a = np.column_stack([E0 + 10 + 120 * t, N0 + 30 + 0 * t])
    th = np.linspace(-np.pi / 2, 0, 60)
    b = np.column_stack([E0 + 130 + 60 * np.cos(th), N0 + 90 + 60 * np.sin(th)])
    c = np.column_stack([E0 + 190 + 0 * t, N0 + 90 + 100 * t])
    return polyline.dedupe(np.vstack([a, b, c]))


def test_draped_simplify_guarantee():
    surf = make_surface()
    D = drape(surf, road_line())
    assert len(D) > 400, len(D)                     # pritraukta linija tanki
    tol_h, tol_v = 0.010, 0.005
    keep = polyline.simplify(D, tol_h, tol_v)
    S = D[keep]
    dh, dv = polyline.deviation(D, keep)
    assert dh <= tol_h + 1e-12 and dv <= tol_v + 1e-12, (dh, dv)
    assert np.all(np.isin(S, D).all(1))               # tik originalūs taškai
    # nauja linija nuo PAVIRŠIAUS atitrūksta ne daugiau nei tol_v (tankiai tikrinant)
    dense = polyline.resample(S, 0.05)
    gap = np.nanmax(np.abs(dense[:, 2] - surf.z_at(dense[:, :2])))
    assert gap <= tol_v * 1.05 + 1e-4, gap
    print(f"   pritraukta: {len(D)} viršūnių -> {len(S)} ({100 * (1 - len(S) / len(D)):.0f} % mažiau), "
          f"max Δz iki paviršiaus {gap * 1000:.2f} mm")
    # 2D retinimas (be aukščio) pašalintų daugiau, bet atitrūktų nuo paviršiaus
    k2 = polyline.simplify(D, tol_h, None)
    assert len(k2) < len(keep)


def test_closed_simplify():
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    P = np.column_stack([50 * np.cos(th), 50 * np.sin(th), np.zeros_like(th)])
    keep = polyline.simplify(P, 0.01, 0.01, closed=True)
    dh, _ = polyline.deviation(np.vstack([P, P[:1]]), np.r_[keep, len(P)])
    assert dh <= 0.01 and 50 < len(keep) < 400


def test_spikes_hooks():
    P = np.column_stack([np.arange(0, 50, 1.0), np.zeros(50), np.zeros(50)])
    S = np.insert(P, 20, [19.0, 0.3, 0], axis=0)     # šuolys: į šoną ir atgal
    S = np.insert(S, 21, [19.05, 0.0, 0], axis=0)
    out, k = polyline.remove_spikes(S, 0.5)
    assert k >= 1 and np.abs(out[:, 1]).max() < 1e-9
    H = np.vstack([[0.0, 0.008, 0], P, [49.0, -0.01, 0]])  # kabliukai abiejuose galuose
    out, k = polyline.trim_hooks(H, 0.02)
    assert k == 2 and np.abs(out[:, 1]).max() < 1e-9
    # tikras kampas (ilgi segmentai) nekarpomas
    C = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0]], float)
    assert polyline.trim_hooks(C, 0.02)[1] == 0 and polyline.remove_spikes(C, 0.5)[1] == 0


def test_join_and_duplicates():
    P = road_line()
    P = np.column_stack([P, np.zeros(len(P))])
    a, b, c = P[:50], P[49:100], P[99:]
    b = b[::-1].copy()                                 # apversta
    c = c.copy(); c[0, :2] += [0.002, 0.001]           # 2 mm tarpas
    chains = linework.join_ends([c, a, b], 0.005)
    big = [ch for ch in chains if len(ch) == 3]
    assert len(big) == 1
    M = linework.merge_chain([c, a, b], big[0])
    if np.linalg.norm(M[0, :2] - P[0, :2]) > 1:
        M = M[::-1]
    assert abs(polyline.length(M) - polyline.length(P)) < 0.01
    # išsišakojimas nejungiamas
    t1 = np.array([[0, 0, 0], [10, 0, 0]], float); t2 = np.array([[10, 0, 0], [20, 0, 0]], float)
    t3 = np.array([[10, 0, 0], [10, 10, 0]], float)
    assert all(len(ch) == 1 for ch in linework.join_ends([t1, t2, t3], 0.005))
    # dublikatai: dalis linijos su 3 mm poslinkiu
    dup = P[20:80].copy(); dup[:, 1] += 0.003
    other = P.copy(); other[:, 1] += 5.0
    d = linework.find_duplicates([P, dup, other], 0.01)
    assert d == {1: 0}, d


def test_tool_end_to_end():
    """DXF su pritraukta linija, dublikatu, 2 dalimis su tarpu, tekstu (turi išlikti)."""
    surf = make_surface()
    D = drape(surf, road_line())
    w = io_dxf.DxfWriter()
    w.polyline3d(D, "BORTAS", 3)
    half = len(D) // 2
    w.polyline3d(D[:half + 1], "KRASTAS", 1)
    w.polyline3d(D[half:] + [0.002, 0, 0], "KRASTAS", 1)
    w.polyline3d(D[100:400] + [0, 0.003, 0], "BORTAS", 3)       # dublikatas
    w.text("PK 0+000", (E0, N0, 0), 1.0, "TEKSTAS")
    w.lwpolyline(D[:, :2], "PLANAS", 5, elevation=12.5)                     # 2D LWPOLYLINE
    # 100 viršūnių tiesėje, tada lankas (bulge 0.4), tada 100 viršūnių tiesėje
    xs = np.linspace(0, 50, 100)
    pts = [(E0 + x, N0 - 5, 0.0) for x in xs] + [(E0 + 60 + x, N0 - 5, 0.0) for x in xs]
    pts[99] = (pts[99][0], pts[99][1], 0.4)
    w.msp.add_lwpolyline(pts, format="xyb", dxfattribs={"layer": w.layer("LANKAS")})
    f = TMP / "drape.dxf"
    w.save(f)

    from app.core.jobs import JobManager
    from app.core.registry import discover
    tool = discover()["linefit"]
    jm = JobManager(TMP / "jobs", workers=1)
    params = tool.parse_params({"mode": "retinti", "tol_h": 10, "tol_v": 5, "dups": True, "join": True, "join_gap": 5})
    jid = jm.create(tool, params, {"files": [("drape.dxf", f.read_bytes())]})
    for _ in range(240):
        m = jm.get(jid)
        if m["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    jm.pool.shutdown()
    assert m["status"] == "done", (m.get("error"), "\n".join(m["log"]))
    out = TMP / "jobs" / jid / "out" / "drape_isvalyta.dxf"
    pl = io_dxf.read_polylines(out)
    lay = {}
    for p in pl:
        lay.setdefault(p.layer, []).append(p)
    assert len(lay["BORTAS"]) == 1 and len(lay["KRASTAS"]) == 1          # dublikatas pašalintas, dalys sujungtos
    P2 = lay["PLANAS"][0].pts
    assert len(P2) < len(D) * 0.5 and np.allclose(P2[:, 2], 12.5)            # LWPOLYLINE išlieka su elevation
    arc = io_dxf.open_doc(out).modelspace().query("LWPOLYLINE[layer=='LANKAS']")[0]
    xyb = np.array(arc.get_points("xyb"))
    assert len(xyb) == 4, xyb                       # tiesė (2) + lanko galai, lankas išliko
    assert np.isclose(xyb[1, 2], 0.4) and np.isclose(xyb[1, 0], E0 + 50) and np.isclose(xyb[2, 0], E0 + 60)
    B = lay["BORTAS"][0].pts
    assert len(B) < len(D) * 0.5
    assert np.isin(np.round(B, 6), np.round(D, 6)).all(1).mean() == 1.0     # likusios viršūnės - originalios
    assert abs(polyline.length(lay["KRASTAS"][0].pts) - polyline.length(D)) < 0.05
    doc = io_dxf.open_doc(out)
    assert len(doc.modelspace().query("TEXT")) == 1                          # kiti objektai nepaliesti
    prev = json.loads((TMP / "jobs" / jid / "preview.json").read_text())
    row = {r["Sluoksnis"]: r for r in prev["table"]}
    assert row["BORTAS"]["Dublikatų"] == 1 and row["KRASTAS"]["Sujungta"] == 1
    assert row["BORTAS"]["max Δz, mm"] <= 5.0 and row["BORTAS"]["max Δplan, mm"] <= 10.0



def test_plan_only_keeps_z():
    """Režimas „tik planas“: aukščiai tiksliai iš originalių taškų, planas - tiesės/lankai."""
    from app.geo import io_landxml
    from app.core.jobs import JobManager
    from app.core.registry import discover
    surf = make_surface()
    D = drape(surf, road_line())                 # tanki 3D linija (planas: tiesė-lankas-tiesė)
    D[:, :2] += rng.normal(0, 0.003, (len(D), 2))
    w = io_dxf.DxfWriter()
    w.polyline3d(D, "BORTAS", 3)
    f = TMP / "planas.dxf"
    w.save(f)
    tool = discover()["linefit"]
    jm = JobManager(TMP / "jobs_pl", workers=1)
    t0 = time.time()
    jid = jm.create(tool, tool.parse_params({"mode": "planas", "h_tol_line": 15, "h_tol_arc": 15}),
                    {"files": [("planas.dxf", f.read_bytes())]})
    for _ in range(600):
        m = jm.get(jid)
        if m["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    jm.pool.shutdown()
    assert m["status"] == "done", (m.get("error"), "\n".join(m["log"]))
    print(f"   tik planas: {len(D)} tšk., {time.time() - t0:.1f} s")
    al = io_landxml.read_alignments(TMP / "jobs_pl" / jid / "out" / "planas_fit.xml")[0]
    kinds = [e.kind for e in al.elems]
    assert kinds.count("arc") >= 1 and len(kinds) <= 5, kinds
    sta, off, _ = al.station_offset(D[:, :2])
    assert np.abs(off).max() <= 0.015 + 1e-6
    assert np.nanmax(np.abs(al.z_at(sta) - D[:, 2])) < 1e-3       # aukščiai išlaikyti (LandXML 4 skaitm.)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("OK ", fn.__name__)
    print(f"Visi {len(fns)} testai praėjo")

"""Branduolio (app/geo) testai su žinomais atsakymais.  python tests/test_geo.py"""
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.geo import io_csv, io_dxf, io_las, io_landxml, polyline, report  # noqa: E402
from app.geo.alignment import Alignment, Profile, format_station  # noqa: E402
from app.geo.surface import Surface, fit_offset, grid_from_points, grid_from_surface, polygon_mask, volume_between  # noqa: E402

TMP = Path(tempfile.mkdtemp())
E0, N0 = 512000.0, 6080000.0
rng = np.random.default_rng(3)


def road_alignment():
    A, R = 60.0, 300.0                      # klotoidės parametras A² = R*L
    Ls = A * A / R
    al = Alignment.from_elems("ASIS", [("line", 100, 0, 0), ("spiral", Ls, 0, 1 / R), ("arc", 150, 1 / R, 1 / R),
                                       ("spiral", Ls, 1 / R, 0), ("line", 80, 0, 0)],
                              (E0, N0), math.radians(20), sta_start=1000.0)
    al.profile = Profile(np.array([1000, 1150, 1300, al.sta_end]), np.array([100, 102.4, 101.0, 101.5]),
                         np.array([0, 100, 60, 0]), "PR")
    return al, Ls, R


def test_spiral_geometry():
    al, Ls, R = road_alignment()
    sp = al.elems[1]
    # klotoidės galo koordinatės lokaliai: Fresnel'io eilutė
    from scipy.special import fresnel
    a = math.sqrt(math.pi * R * Ls)
    S, C = fresnel(Ls / a)
    loc = np.array([a * C, a * S])
    rot = np.array([[math.cos(sp.th), -math.sin(sp.th)], [math.sin(sp.th), math.cos(sp.th)]])
    exp = sp.p + rot @ loc
    assert np.linalg.norm(sp.end() - exp) < 1e-6, (sp.end(), exp)
    assert abs(sp.th_end() - sp.th - Ls / (2 * R)) < 1e-12
    # G1 tolydumas
    for e1, e2 in zip(al.elems[:-1], al.elems[1:]):
        assert np.linalg.norm(e1.end() - e2.p) < 1e-9 and abs(e1.th_end() - e2.th) < 1e-12


def test_station_offset_roundtrip():
    al, *_ = road_alignment()
    sta = rng.uniform(al.sta_start, al.sta_end, 2000)
    off = rng.uniform(-25, 25, 2000)
    xy = al.xy_at(sta, off)
    s2, o2, inside = al.station_offset(xy)
    assert np.abs(s2 - sta).max() < 1e-6 and np.abs(o2 - off).max() < 1e-6
    assert inside.all()
    # + dešinėje: trasa eina ~ į šiaurės rytus, taškas į pietryčius turi būti dešinėje
    p = al.xy_at([1050], [5])[0]
    th = al.theta_at([1050])[0]
    assert np.dot(p - al.xy_at([1050])[0], [math.sin(th), -math.cos(th)]) > 0
    assert format_station(1234.5) == "1+234.50" and format_station(1234.5, per=100) == "12+34.50"


def test_profile():
    al, *_ = road_alignment()
    pr = al.profile
    assert abs(pr.z_at([1000])[0] - 100) < 1e-12
    assert abs(pr.z_at([1150 - 50 - 1])[0] - (100 + 0.016 * 99)) < 1e-9        # tiesė prieš kreivę
    # kreivės vidurys: E - (g2-g1)*L/8
    g1, g2 = 2.4 / 150, -1.4 / 150
    assert abs(pr.z_at([1150])[0] - (102.4 + (g2 - g1) * 100 / 8)) < 1e-9


def test_landxml_alignment_roundtrip():
    al, *_ = road_alignment()
    f = TMP / "al.xml"
    io_landxml.write_alignments(f, [al])
    al2 = io_landxml.read_alignments(f)[0]
    assert [e.kind for e in al2.elems] == [e.kind for e in al.elems]
    s = np.linspace(al.sta_start, al.sta_end, 500)
    assert np.abs(al2.xy_at(s, 3.0) - al.xy_at(s, 3.0)).max() < 2e-4      # 4 skaitmenys faile
    assert np.abs(al2.z_at(s) - al.z_at(s)).max() < 1e-4


def test_surface_and_volume():
    xy = rng.uniform([E0, N0], [E0 + 200, N0 + 150], (5000, 2))
    plane = lambda q: 5 + 0.01 * (q[:, 0] - E0) + 0.02 * (q[:, 1] - N0)
    base = Surface(np.column_stack([xy, plane(xy)]), name="base")
    q = rng.uniform([E0 + 20, N0 + 20], [E0 + 180, N0 + 130], (1000, 2))
    assert np.nanmax(np.abs(base.z_at(q) - plane(q))) < 1e-9
    assert np.isnan(base.z_at(np.array([[E0 - 10, N0]])))[0]
    top = Surface(np.column_stack([xy, plane(xy) + 0.5]), name="top")
    b = (E0 + 50, N0 + 50, E0 + 150, N0 + 100)
    g0, g1 = grid_from_surface(base, 0.5, b), grid_from_surface(top, 0.5, b)
    v = volume_between(g0, g1)
    assert abs(v.fill - 100 * 50 * 0.5) < 1e-6 and v.cut == 0
    ring = np.array([[E0 + 60, N0 + 60], [E0 + 100, N0 + 60], [E0 + 100, N0 + 90], [E0 + 60, N0 + 90]])
    v2 = volume_between(g0, g1, polygon_mask(g0, [ring]))
    assert abs(v2.fill - 40 * 30 * 0.5) < 1e-6
    # LandXML paviršiaus ratas
    f = TMP / "s.xml"
    io_landxml.write_surfaces(f, [base.to_dict()])
    s2 = Surface.from_dict(io_landxml.read_surfaces(f)[0])
    assert np.nanmax(np.abs(s2.z_at(q) - plane(q))) < 1e-3
    assert len(base.boundary()) == 1


def test_fit_offset_robust():
    xy = rng.uniform([0, 0], [100, 100], (3000, 2))
    ref = Surface(np.column_stack([xy, 0.02 * xy[:, 0]]))
    scan = np.column_stack([xy, 0.02 * xy[:, 0] + 0.037 + rng.normal(0, 0.005, len(xy))])
    hole = np.hypot(xy[:, 0] - 50, xy[:, 1] - 50) < 15           # išplova
    scan[hole, 2] -= 0.4
    p, corr, res = fit_offset(ref, scan, "dz")
    assert abs(p[0] - 0.037) < 0.002, p
    assert np.nanmean(res[hole]) < -0.35


def test_grid_from_points():
    pts = np.array([[0.1, 0.1, 1], [0.2, 0.3, 3], [1.5, 0.5, 7], [0.9, 0.9, 2]])
    g = grid_from_points(pts, 1.0, "min", bounds=(0, 0, 2, 1))
    assert g.z.tolist() == [[1.0, 7.0]]
    assert grid_from_points(pts, 1.0, "max", bounds=(0, 0, 2, 1)).z.tolist() == [[3.0, 7.0]]
    assert grid_from_points(pts, 1.0, "median", bounds=(0, 0, 2, 1)).z.tolist() == [[2.0, 7.0]]


def test_csv_detection():
    txt = "Pt;N;E;Z\n1;6080000,123;512000,456;100,1\n2;6080010,0;512005,0;100,2\n"
    ps = io_csv.read_points(txt)
    assert ps.order == "NEZ" and ps.names == ["1", "2"]
    assert np.allclose(ps.xyz[0], [512000.456, 6080000.123, 100.1])
    ps2 = io_csv.read_points("101,512000.0,6080000.0,99.5,BORT\n102,512001.0,6080001.0,99.6,BORT\n")
    assert ps2.names == ["101", "102"] and ps2.codes == ["BORT", "BORT"] and ps2.order == "ENZ"
    ps3 = io_csv.read_points("512000 6080000 10\n512001 6080001 11\n512002 6080001 12\n")
    assert len(ps3) == 3 and ps3.names == ["1", "2", "3"]
    f = TMP / "p.csv"
    io_csv.write_points(f, ps.xyz, ps.names)
    assert np.allclose(io_csv.read_points(f).xyz, ps.xyz, atol=1e-3)


def test_las_roundtrip():
    xyz = np.column_stack([rng.uniform(E0, E0 + 50, 20000), rng.uniform(N0, N0 + 50, 20000), rng.uniform(10, 11, 20000)])
    f = TMP / "c.laz"
    io_las.write(f, xyz)
    assert io_las.info(f)["points"] == 20000
    a = io_las.read(f, chunk=7000)
    assert len(a) == 20000 and np.abs(a.min(0) - xyz.min(0)).max() < 1e-3
    b = io_las.read(f, bounds=(E0, N0, E0 + 10, N0 + 10), thin=1.0, chunk=7000)
    assert 90 <= len(b) <= 121 and (b[:, 0] <= E0 + 10).all()


def test_dxf_roundtrip():
    w = io_dxf.DxfWriter()
    w.polyline3d([(E0, N0, 1), (E0 + 10, N0, 2), (E0 + 10, N0 + 5, 3)], "BORT", 1)
    w.lwpolyline([(E0, N0), (E0 + 1, N0 + 1)], "ASIS", elevation=5)
    w.faces(np.array([[0, 0, 1], [1, 0, 1], [0, 1, 2], [1, 1, 2]], float), [[0, 1, 2], [1, 3, 2]], "TIN")
    f = TMP / "a.dxf"
    w.save(f)
    pl = io_dxf.read_polylines(f)
    assert {p.layer for p in pl} == {"BORT", "ASIS"}
    lw = [p for p in pl if p.layer == "ASIS"][0]
    assert abs(lw.pts[0, 2] - 5) < 1e-9
    pts, tris = io_dxf.read_faces(f, ["TIN"])
    assert len(pts) == 4 and len(tris) == 2


def test_polyline_and_report():
    t = np.linspace(0, 1, 101)
    P = np.column_stack([t * 100, np.where(t < 0.5, 0, (t - 0.5) * 20), np.zeros_like(t)])
    keep = polyline.rdp(P, 0.01)
    assert len(keep) == 3
    assert polyline.split_at_corners(np.array([[0, 0], [10, 0], [10, 10]], float), 60)[1].shape == (2, 2)
    f = TMP / "r.xlsx"
    report.write_xlsx(f, [report.Sheet("Taškai", [report.Col("Nr", "n"), report.Col("dz, mm", "dz", "0.0")],
                                       [{"n": 1, "dz": 3.2}, {"n": 2, "dz": 25.0}], bad=lambda r: abs(r["dz"]) > 20,
                                       bad_cols=["dz, mm"], info=[("Tolerancija", "20 mm")], totals={"dz, mm": "MAX"})])
    from openpyxl import load_workbook
    ws = load_workbook(f)["Taškai"]
    vals = [c.value for c in ws["B"] if c.value is not None]
    assert 25.0 in vals and any(isinstance(v, str) and v.startswith("=MAX") for v in vals)


def test_linefit_output_readable():
    """linefit LandXML -> branduolio skaitytuvas: įvesties taškai turi būti arti trasos."""
    from app.tools.linefit import engine
    from app.geo.io_dxf import read_polylines
    d = Path(__file__).parent / "data"
    st = engine.Settings(tol_line=0.015, tol_arc=0.015, v_tol=0.008)
    res, files = engine.run([d / "test.dxf"], st, str(TMP / "lf"), make_plot=False, log=lambda m: None)
    al = [a for a in io_landxml.read_alignments(files["xml"]) if a.name.startswith("KELIO")][0]
    P = [p for p in read_polylines(d / "test.dxf") if p.layer == "KELIO_ASIS"][0].pts
    sta, off, _ = al.station_offset(P[:, :2])
    assert np.abs(off).max() < 0.015 + 1e-6
    assert np.nanmax(np.abs(al.z_at(sta) - P[:, 2])) < 0.008 + 1e-6
    assert [e.kind for e in al.elems].count("arc") == 3


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("OK ", fn.__name__)
    print(f"Visi {len(fns)} testai praėjo")

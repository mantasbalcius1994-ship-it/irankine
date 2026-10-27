"""Paviršių palyginimas ir tūriai: faktinis (skenavimas / taškai / paviršius) prieš projektinį.
Lokalus pritaikymas (GPS poslinkis), iškasos / pylimo tūriai, išplovų ir sankaupų zonos su įrodymais."""
from pathlib import Path

import numpy as np

from ...core.tool import FileInput, Param, Tool
from ...geo import limits, loaders
from ...geo.report import Col, Sheet, write_xlsx
from ...geo.surface import find_zones, fit_offset, grid_from_points, grid_from_surface, save_raster, volume_between


class Pavirsiai(Tool):
    id = "pavirsiai"
    title = "Paviršių palyginimas ir tūriai"
    group = "Paviršiai"
    icon = "⛰"
    description = ("Lygina faktinį paviršių (LAS/LAZ skenavimas, taškai arba TIN) su projektiniu: lokaliai "
                   "pritaiko aukštį (GPS poslinkis), skaičiuoja iškasos / pylimo tūrius ir randa išplovų "
                   "zonas su plotu, tūriu ir giliausia vieta.")
    help = ("<ul><li><b>Pritaikymas</b>: projektinis paviršius pastumiamas aukštyje (arba pakreipiamas plokštuma), "
            "kad sutaptų su skenavimu. Naudojamas robastinis metodas - išplovos ir sankaupos pritaikymo neiškreipia.</li>"
            "<li><b>dz = faktas − projektas</b>: neigiamas - žemiau projekto (išplova / iškasa), teigiamas - aukščiau.</li>"
            "<li>Tūriai skaičiuojami tinkleliu tik ten, kur yra abu paviršiai; langeliai be skenavimo taškų - "
            "nurodomi atskirai.</li>"
            "<li>Rezultatai: DXF (zonų kontūrai, užrašai, giliausios vietos), spalvotas dz vaizdas su .pgw "
            "(įkeliamas į TBC kaip georeferencuotas paveikslas), Excel ataskaita, pritaikytas projektinis paviršius "
            "(LandXML).</li></ul>")
    inputs = [FileInput("design", "Projektinis paviršius (TTM / LandXML / DXF 3DFACE)", accept=".ttm,.xml,.dxf",
                        multiple=False),
              FileInput("scan", "Faktas (LAS / LAZ / CSV / DXF / TTM / LandXML)",
                        accept=".las,.laz,.csv,.txt,.dxf,.ttm,.xml", multiple=True)]
    params = [
        Param("fit", "Projekto pritaikymas prie fakto", "choice", "dz", group="Pritaikymas", choices=[
            ["none", "Nepritaikyti"], ["dz", "Aukščio poslinkis"], ["plane", "Poslinkis + nuolydis (plokštuma)"]],
              help="Kompensuoja GPS / bazės aukščio paklaidą. Išplovos rezultato neiškreipia (robastinis metodas)."),
        Param("fit_slope", "Pritaikyti tik ten, kur nuolydis <", "float", 10.0, "%", "Pritaikymas", min=0,
              help="Išplovos būna šlaituose, o ant šlaito ir plano paklaida virsta aukščio skirtumu. Todėl aukštis "
                   "derinamas lygiose vietose (važiuojamoji dalis, kelkraštis). 0 = visur."),
        Param("max_fit", "Įspėti, kai poslinkis >", "float", 100.0, "mm", "Pritaikymas", min=0),
        Param("step", "Tinklelio žingsnis", "float", 0.25, "m", "Tinklelis", min=0.02,
              help="Mažesnis - tiksliau, bet reikia tankesnio skenavimo (≥ 2-3 taškai langelyje)."),
        Param("stat", "Langelio aukštis iš taškų", "choice", "median", group="Tinklelis", choices=[
            ["median", "Mediana"], ["mean", "Vidurkis"], ["min", "Žemiausias (augmenija)"]]),
        Param("ignore", "Neskaičiuoti skirtumų iki", "float", 20.0, "mm", "Tūriai", min=0,
              help="Skenavimo triukšmas - mažesni skirtumai į tūrius neįtraukiami."),
        Param("zone_thr", "Išplovos ribos (gylis)", "str", "50,100", "mm", "Zonos",
              help="Viena ar kelios ribos, kableliais. Kiekvienai - atskiri kontūrai (A - pirma riba, B - antra...)."),
        Param("zone_min", "Min zonos plotas", "float", 0.5, "m²", "Zonos", min=0),
        Param("fill_zones", "Žymėti ir sankaupas (aukščiau projekto)", "bool", False, group="Zonos"),
        Param("sec_len", "Ruožo ilgis (ilgiems objektams)", "float", 10.0, "m", "Zonos", min=0,
              help="Pylimas, griovys ir pan.: tūriai kas N m išilgai automatiškai rastos vidurio linijos. 0 = ne."),
        Param("crange", "Spalvų skalė ±", "float", 150.0, "mm", "Išvestis", min=5),
        Param("write_fit", "Išsaugoti pritaikytą projektinį paviršių", "bool", True, group="Išvestis"),
    ]

    def run(self, ctx):
        p = ctx.params
        log = ctx.log
        step = p["step"]
        des_f = ctx.inputs["design"][0]
        design = loaders.load_surface(des_f)
        b = design.bbox
        log(f"Projektas {des_f.name}: {len(design.pts)} tšk., {len(design.tris)} trikampių, "
            f"plotas {design.area_2d():.1f} m²")

        # --- faktas
        pts, surfs = [], []
        bounds = (b[0] - 1, b[1] - 1, b[2] + 1, b[3] + 1)
        for f in ctx.inputs["scan"]:
            thin = None
            if f.suffix.lower() in (".las", ".laz"):
                from ...geo import io_las
                n = io_las.info(f)["points"]
                if n > limits.MAX_CLOUD_POINTS:
                    thin = step / 4
                    log(f"  {f.name}: {n} taškų - retinama {thin * 1000:.0f} mm tinkleliu")
            xyz, s = loaders.load_points(f, thin=thin, bounds=bounds, log=log)
            if s is not None:
                surfs.append(s); log(f"Faktas {f.name}: paviršius, {len(s.pts)} tšk.")
            else:
                m = (xyz[:, 0] >= bounds[0]) & (xyz[:, 0] <= bounds[2]) & (xyz[:, 1] >= bounds[1]) & (xyz[:, 1] <= bounds[3])
                pts.append(xyz[m]); log(f"Faktas {f.name}: {m.sum()} tšk. projekto ribose (iš {len(xyz)})")
        if pts and surfs:
            raise ValueError("Faktą pateikite arba taškais (LAS/CSV), arba vienu paviršiumi - ne abiem kartu")
        if len(surfs) > 1:
            raise ValueError("Faktinis paviršius turi būti vienas")
        xyz = np.vstack(pts) if pts else surfs[0].pts
        if not len(xyz):
            raise ValueError("Fakto taškų projekto ribose nėra. Patikrinkite koordinačių sistemą (LKS-94, X = E).")

        # --- pritaikymas
        zr = design.z_at(xyz[:, :2])
        on = ~np.isnan(zr)
        if on.sum() < 3:
            raise ValueError("Fakto taškai nepatenka ant projektinio paviršiaus. Patikrinkite koordinates (E/N tvarka).")
        raw = xyz[on, 2] - zr[on]
        log(f"Prieš pritaikymą: dz mediana {np.median(raw) * 1000:+.1f} mm, 5-95 % "
            f"{np.percentile(raw, 5) * 1000:+.0f}…{np.percentile(raw, 95) * 1000:+.0f} mm ({on.sum()} tšk.)")
        corr, fit_txt = (lambda xy: np.zeros(len(np.atleast_2d(xy)))), "nepritaikyta"
        if p["fit"] != "none":
            fm = None
            if p["fit_slope"] > 0:
                fm = design.slope_at(xyz[:, :2]) < p["fit_slope"] / 100
                share = fm.sum() / max(on.sum(), 1)
                log(f"Pritaikymui naudojami taškai, kur projekto nuolydis < {p['fit_slope']:.0f} %: {fm.sum()} "
                    f"({share * 100:.0f} %)")
                if fm.sum() < 1000 and fm.sum() < 0.05 * on.sum():
                    log("ĮSPĖJIMAS: lygių vietų per mažai - pritaikoma pagal visus taškus (rezultatas gali "
                        "sumažinti išplovas). Geriau nurodykite didesnį nuolydį arba fit=none.")
                    fm = None
            par, corr, res = fit_offset(design, xyz, mode=p["fit"], mask=fm)
            res = xyz[:, 2] - (zr + corr(xyz))
            if p["fit"] == "dz":
                fit_txt = f"aukščio poslinkis {par[0] * 1000:+.1f} mm"
            else:
                g = np.hypot(par[1], par[2])
                fit_txt = (f"poslinkis centre {par[0] * 1000:+.1f} mm, nuolydis {g * 100:.3f} % "
                           f"(kryptis {np.degrees(np.arctan2(par[1], par[2])) % 360:.0f}°)")
            log(f"Pritaikymas: {fit_txt}")
            r = res[~np.isnan(res)]
            log(f"Po pritaikymo: dz mediana {np.median(r) * 1000:+.1f} mm, 5-95 % "
                f"{np.percentile(r, 5) * 1000:+.0f}…{np.percentile(r, 95) * 1000:+.0f} mm")
            if abs(par[0]) * 1000 > p["max_fit"]:
                log(f"ĮSPĖJIMAS: poslinkis {par[0] * 1000:+.0f} mm didesnis už {p['max_fit']:.0f} mm - "
                    "gal tai ne GPS paklaida, o tikras aukščių skirtumas (ar ne tas paviršius)?")

        # --- tinkleliai
        gx0, gy0 = np.floor(b[0] / step) * step, np.floor(b[1] / step) * step
        gx1, gy1 = np.ceil(b[2] / step) * step, np.ceil(b[3] / step) * step
        nx, ny = int(round((gx1 - gx0) / step)), int(round((gy1 - gy0) / step))
        if nx * ny > 40_000_000:
            raise ValueError(f"Per smulkus tinklelis ({nx}×{ny}). Padidinkite žingsnį.")
        gb = (gx0, gy0, gx1, gy1)
        gd = grid_from_surface(design, step, gb, "projektas")
        X, Y = gd.centers()
        gd.z = gd.z + corr(np.column_stack([X.ravel(), Y.ravel()])).reshape(gd.z.shape)
        if surfs:
            gs = grid_from_surface(surfs[0], step, gb, "faktas")
            cnt = (~np.isnan(gs.z)).astype(float)
        else:
            gs = grid_from_points(xyz, step, p["stat"], gb, "faktas")
            cnt = grid_from_points(xyz, step, "count", gb).z
        inside = ~np.isnan(gd.z)
        nodata = inside & np.isnan(gs.z)
        A = gd.cell_area
        vr = volume_between(gd, gs, dz_ignore=p["ignore"] / 1000)
        dz = vr.dz
        log(f"Tinklelis {step} m: {inside.sum()} langelių projekte, be fakto {nodata.sum()} "
            f"({nodata.sum() * A:.1f} m², {100 * nodata.sum() / max(inside.sum(), 1):.1f} %)")
        dense = cnt[inside & ~nodata]
        if len(dense) and np.median(dense) < 2 and not surfs:
            log(f"ĮSPĖJIMAS: langelyje vidutiniškai {np.median(dense):.1f} tšk. - padidinkite žingsnį.")
        log(f"Tūriai (|dz| ≥ {p['ignore']:.0f} mm): žemiau projekto {vr.cut:.3f} m³, aukščiau {vr.fill:.3f} m³, "
            f"plotas {vr.area:.1f} m²")

        # --- zonos (kelios gylio ribos)
        try:
            levels = sorted({float(v) for v in str(p["zone_thr"]).replace(";", ",").split(",") if v.strip()})
        except ValueError:
            raise ValueError("Išplovos ribos: skaičiai mm, atskirti kableliais (pvz. 50,100)")
        if not levels or levels[0] <= 0:
            raise ValueError("Išplovos riba turi būti > 0 mm")
        zones = []
        for li, lv in enumerate(levels):
            zs = find_zones(dz, lv / 1000, p["zone_min"], -1)
            for z in zs:
                z.name, z.level = f"{'ABCDEFGH'[li]}{z.id}", lv
            zones += zs
        if p["fill_zones"]:
            for z in find_zones(dz, levels[0] / 1000, p["zone_min"], +1):
                z.name, z.level = f"S{z.id}", levels[0]
                zones.append(z)
        for li, lv in enumerate(levels):
            zl = [z for z in zones if z.sign < 0 and z.level == lv]
            log(f"Išplovos giliau {lv:.0f} mm ({'ABCDEFGH'[li]}, ≥ {p['zone_min']} m²): {len(zl)} zonų, "
                f"plotas {sum(z.area for z in zl):.2f} m², tūris {sum(z.volume for z in zl):.3f} m³")
            for z in zl[:5]:
                log(f"  {z.name}: S {z.area:.2f} m², V {z.volume:.3f} m³, max {z.max_dz * 1000:.0f} mm "
                    f"ties E {z.xy_max[0]:.2f} N {z.xy_max[1]:.2f}")
        cutz = [z for z in zones if z.sign < 0 and z.level == levels[0]]

        # --- ilgas objektas: vidurio linija ir ruožai
        frame = self._frame(dz, inside)
        sections = []
        if frame is not None and p["sec_len"] > 0:
            sections = self._sections(dz, frame, p["sec_len"], p["ignore"] / 1000, levels[0] / 1000)
            log(f"Ilgas objektas ({frame['L']:.0f} m): ruožai kas {p['sec_len']:g} m išilgai vidurio linijos "
                f"(nuo {frame['start']} galo)")
            top = sorted(sections, key=lambda r: -r["cut"])[:3]
            log("  daugiausia išplauta: " + "; ".join(f"{r['from']:.0f}-{r['to']:.0f} m {r['cut']:.2f} m³" for r in top))

        # --- išvestis
        stem = Path(des_f).stem
        out = ctx.outdir
        self._dxf(out / f"{stem}_palyginimas.dxf", design, zones, gd, p, frame)
        ctx.add_output(out / f"{stem}_palyginimas.dxf", "DXF: zonų kontūrai, užrašai, giliausios vietos")
        cr = p["crange"] / 1000
        png, pgw = save_raster(dz, out / f"{stem}_dz.png", -cr, cr)
        ctx.add_output(png, "dz vaizdas (georeferencuotas, su .pgw)", "image")
        ctx.add_output(pgw, "Pasaulio failas (.pgw) vaizdui")
        if p["fit"] != "none" and p["write_fit"]:
            from ...geo import io_landxml
            P = design.pts.copy()
            P[:, 2] += corr(P[:, :2])
            fx = out / f"{stem}_pritaikytas.xml"
            io_landxml.write_surfaces(fx, [{"name": f"{design.name} pritaikytas", "pts": P, "tris": design.tris}])
            ctx.add_output(fx, "Pritaikytas projektinis paviršius (LandXML)")
        rows = [{"Zona": z.name, "Tipas": "išplova" if z.sign < 0 else "sankaupa", "Riba": z.level,
                 "Plotas, m²": round(z.area, 2), "Tūris, m³": round(z.volume, 3), "Max, mm": round(z.max_dz * 1000),
                 "Vid., mm": round(z.mean_dz * 1000), "E": round(z.xy_max[0], 3), "N": round(z.xy_max[1], 3),
                 "PK": round(frame["st"](np.array([z.xy_max]))[0], 1) if frame else None} for z in zones]
        summary = [("Projektas", des_f.name), ("Faktas", ", ".join(f.name for f in ctx.inputs["scan"])),
                   ("Pritaikymas", fit_txt), ("Tinklelis, m", step), ("Langelio aukštis", p["stat"]),
                   ("Neskaičiuota iki, mm", p["ignore"]),
                   ("Žemiau projekto (iškasa / išplovos), m³", round(vr.cut, 3)),
                   ("Aukščiau projekto (pylimas), m³", round(vr.fill, 3)),
                   ("Palygintas plotas, m²", round(vr.area, 2)),
                   ("Be fakto duomenų, m²", round(nodata.sum() * A, 2))]
        for li, lv in enumerate(levels):
            zl = [z for z in zones if z.sign < 0 and z.level == lv]
            summary += [(f"Išplovos giliau {lv:.0f} mm: zonų / plotas m² / tūris m³",
                         f"{len(zl)} / {sum(z.area for z in zl):.2f} / {sum(z.volume for z in zl):.3f}")]
        cols = [Col("Zona", "Zona", width=8), Col("Tipas", "Tipas", width=10), Col("Riba, mm", "Riba", "0"),
                Col("Plotas, m²", "Plotas, m²", "0.00"), Col("Tūris, m³", "Tūris, m³", "0.000"),
                Col("Max gylis/aukštis, mm", "Max, mm", "0"), Col("Vidutinis, mm", "Vid., mm", "0"),
                Col("Giliausia vieta E", "E", "0.000", 14), Col("Giliausia vieta N", "N", "0.000", 14)]
        if frame:
            cols.append(Col("Atstumas išilgai, m", "PK", "0.0", 12))
        sheets = [Sheet("Zonos", cols, rows, info=summary)]
        if sections:
            sheets.append(Sheet("Ruožai", [
                Col("Nuo, m", "from", "0.0"), Col("Iki, m", "to", "0.0"), Col("Plotas, m²", "area", "0.00"),
                Col("Žemiau projekto, m³", "cut", "0.000"), Col("Aukščiau projekto, m³", "fill", "0.000"),
                Col(f"Plotas giliau {levels[0]:.0f} mm, m²", "deep", "0.00"), Col("Vid. dz, mm", "mean", "0"),
                Col("Max gylis, mm", "max", "0"), Col("Giliausia vieta E", "E", "0.000", 14),
                Col("Giliausia vieta N", "N", "0.000", 14)], sections,
                info=[("Atstumas", f"išilgai vidurio linijos nuo {frame['start']} galo (DXF sluoksnis VIDURIO_LINIJA)")],
                totals={"Žemiau projekto, m³": "SUM", "Aukščiau projekto, m³": "SUM"}))
        write_xlsx(out / f"{stem}_ataskaita.xlsx", sheets, "Paviršių palyginimas")
        ctx.add_output(out / f"{stem}_ataskaita.xlsx", "Excel ataskaita")
        self._overview(out / f"{stem}_planas.png", dz, design, zones, cr, raw, fit_txt, vr, p, frame, sections,
                       levels)
        ctx.add_output(out / f"{stem}_planas.png", "Peržiūra: dz planas su zonomis ir histograma", "image")
        ctx.preview = {
            "image": f"{stem}_planas.png",
            "legend": [["removed", "Išplova (žemiau projekto)"], ["arc", "Sankaupa"], ["line", "Projekto riba"]],
            "plan": [{"elements": [{"type": "line", "xy": r[:, :2].round(3).tolist(), "label": "Projekto riba"}
                                   for r in design.boundary()] +
                      [{"type": "removed" if z.sign < 0 else "arc", "xy": z.rings[0].round(3).tolist(),
                        "label": f"{z.name}: S {z.area:.2f} m², V {z.volume:.3f} m³, max {z.max_dz * 1000:.0f} mm"}
                       for z in zones]}],
            "table": [{"Rodiklis": k, "Reikšmė": v} for k, v in summary[2:]],
        }
        if sections:
            ctx.preview["charts"] = [{"title": "Tūris žemiau projekto ruožuose, m³", "x": [r["from"] for r in sections],
                                      "series": [{"name": "m³", "y": [r["cut"] for r in sections]}]}]

    @staticmethod
    def _frame(dz, inside):
        """Ilgam siauram objektui - vidurio linija ir (atstumas, skersai) funkcijos. Kitaip None."""
        from ...geo.polyline import centerline, station_offset
        X, Y = dz.centers()
        P = np.column_stack([X[inside], Y[inside]])
        if len(P) < 50:
            return None
        c = P.mean(0)
        sv = np.linalg.svd(P - c, compute_uv=False)
        if sv[0] / max(sv[1], 1e-9) < 4:
            return None
        C = centerline(P, seg=max(5 * dz.step, min(10.0, np.sqrt(len(P)) * dz.step / 4)))
        s, o = station_offset(C, P)
        s0 = s.min()
        start = "pietinio" if C[-1][1] > C[0][1] else "šiaurinio"
        if abs(C[-1][0] - C[0][0]) > abs(C[-1][1] - C[0][1]):
            start = "vakarinio" if C[-1][0] > C[0][0] else "rytinio"
        return {"C": C, "s0": s0, "L": s.max() - s0, "W": np.ptp(o), "omin": o.min(), "omax": o.max(),
                "start": start,
                "st": lambda q: station_offset(C, q)[0] - s0,
                "so": lambda q: (lambda a: (a[0] - s0, a[1]))(station_offset(C, q))}

    @staticmethod
    def _sections(dz, frame, L, ign, thr):
        X, Y = dz.centers()
        ok = ~np.isnan(dz.z)
        P = np.column_stack([X[ok], Y[ok]])
        d = dz.z[ok]
        s = frame["st"](P)
        k = (s // L).astype(int)
        A = dz.cell_area
        dd = np.where(np.abs(d) < ign, 0, d)
        out = []
        for i in range(k.max() + 1):
            m = k == i
            if not m.any():
                continue
            j = np.argmin(np.where(m, d, np.inf))
            out.append({"from": i * L, "to": min((i + 1) * L, frame["L"]), "area": round(m.sum() * A, 2),
                        "cut": round(float(-dd[m & (dd < 0)].sum() * A), 3),
                        "fill": round(float(dd[m & (dd > 0)].sum() * A), 3),
                        "deep": round(float((m & (d < -thr)).sum() * A), 2), "mean": round(float(d[m].mean() * 1000)),
                        "max": round(float(max(0, -d[j]) * 1000)), "E": round(P[j, 0], 3), "N": round(P[j, 1], 3)})
        return out

    @staticmethod
    def _dxf(path, design, zones, gd, p, frame):
        from ...geo.io_dxf import DxfWriter
        w = DxfWriter()
        for r in design.boundary():
            w.polyline3d(r, "PROJEKTO_RIBA", 8, closed=True)
        h = max(0.15, p["step"] * 1.2)
        colors = [1, 6, 5, 3, 4]
        for z in zones:
            if z.sign < 0:
                k = sorted({q.level for q in zones if q.sign < 0}).index(z.level)
                lay, col = f"ISPLOVOS_{z.level:.0f}MM", colors[k % len(colors)]
            else:
                lay, col = "SANKAUPOS", 30
            for ring in z.rings:
                zz = gd.z[np.clip(((ring[:, 1] - gd.y0) / gd.step).round().astype(int), 0, gd.z.shape[0] - 1),
                          np.clip(((ring[:, 0] - gd.x0) / gd.step).round().astype(int), 0, gd.z.shape[1] - 1)]
                w.lwpolyline(ring, lay, col, closed=True,
                             elevation=float(np.nanmean(zz)) if np.isfinite(zz).any() else 0.0)
            w.circle(z.xy_max, h / 2, lay + "_MAX", col)
            w.text(f"{z.name} V={z.volume:.3f}m3 S={z.area:.2f}m2 max={z.max_dz * 1000:.0f}mm",
                   (z.xy_max[0] + h, z.xy_max[1]), h, lay + "_TXT", col)
        if frame:
            from ...geo.polyline import chainage, resample
            C = frame["C"]
            w.lwpolyline(C, "VIDURIO_LINIJA", 2)
            ch = chainage(C)
            L = p["sec_len"] or 10.0
            for t in np.arange(0, frame["L"] + 1e-6, L):
                u = t + frame["s0"]
                k = min(np.searchsorted(ch, u, side="right") - 1, len(C) - 2)
                d = (C[k + 1] - C[k]) / max(np.hypot(*(C[k + 1] - C[k])), 1e-9)
                q = C[k] + d * (u - ch[k])
                n = np.array([d[1], -d[0]])
                w.line(q - n * 0.5, q + n * 0.5, "VIDURIO_LINIJA", 2)
                w.text(f"{t:.0f}", q + n * 0.7, h, "VIDURIO_LINIJA", 2)
        w.save(path)

    @staticmethod
    def _overview(path, dz, design, zones, cr, raw, fit_txt, vr, p, frame, sections, levels):
        """Peržiūra. Ilgas siauras objektas - „išvyniojamas“ išilgai vidurio linijos į kelias juostas,
        kad išplovos būtų matomos; po juo - tūriai ruožuose ir skirtumų histograma."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("RdBu")
        vmax = cr * 1000
        ok = ~np.isnan(dz.z)
        lvl_col = ["#3b0016", "#c2006b", "#5a2d91", "#00703c"]
        title = (f"Pritaikymas: {fit_txt}. Žemiau projekto {vr.cut:.2f} m³, aukščiau {vr.fill:.2f} m³. "
                 f"Kontūrai: " + ", ".join(f"{'ABCDEFGH'[i]} - giliau {v:.0f} mm" for i, v in enumerate(levels)))

        def zc(z):
            return lvl_col[levels.index(z.level) % len(lvl_col)] if z.sign < 0 else "#b86a00"
        lab = sorted([z for z in zones if z.sign < 0 and z.level == levels[-1]], key=lambda z: -z.volume)[:30]
        if not lab:
            lab = sorted([z for z in zones if z.sign < 0], key=lambda z: -z.volume)[:30]
        if frame:
            X, Y = dz.centers()
            P = np.column_stack([X[ok], Y[ok]])
            st, of = frame["so"](P)
            s = dz.step
            ns, no = int(np.ceil(frame["L"] / s)) + 1, int(np.ceil((of.max() - of.min()) / s)) + 1
            i, j = np.clip((st / s).astype(int), 0, ns - 1), ((of - of.min()) / s).astype(int)
            lin = j * ns + i
            sm = np.bincount(lin, dz.z[ok], ns * no); cn = np.bincount(lin, minlength=ns * no)
            with np.errstate(invalid="ignore"):
                R = (sm / cn).reshape(no, ns) * 1000
            L, W = frame["L"], np.ptp(of)
            nrow = int(min(6, max(1, np.ceil(L / max(20 * W, 50)))))
            seg = L / nrow
            rh = float(np.clip(15 * W / seg * 2.5, 1.1, 3.0))
            fig = plt.figure(figsize=(16, nrow * rh + (2.4 if sections else 0) + 2.6), layout="constrained")
            hr = [rh] * nrow + ([2.4] if sections else []) + [2.6]
            gsp = fig.add_gridspec(len(hr), 1, height_ratios=hr)
            ext = [0, ns * s, of.min(), of.min() + no * s]
            rows = []
            for k in range(nrow):
                a = fig.add_subplot(gsp[k]); rows.append(a)
                im = a.imshow(R, origin="lower", extent=ext, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
                              interpolation="nearest")
                for z in zones:
                    u, v = frame["so"](z.rings[0])
                    a.plot(u, v, "-", color=zc(z), lw=0.8)
                for z in lab:
                    u, v = frame["so"](np.array([z.xy_max]))
                    a.annotate(z.name, (u[0], v[0]), xytext=(0, 5), textcoords="offset points", fontsize=7,
                               color=zc(z), weight="bold", ha="center", clip_on=True)
                a.set_xlim(k * seg, (k + 1) * seg)
                a.set_ylim(of.min(), of.max())
                a.tick_params(labelsize=7)
                a.set_ylabel("skersai, m", fontsize=7)
            rows[0].set_title("Išvyniota išilgai vidurio linijos (skersinis mastelis padidintas). " + title, fontsize=9)
            rows[-1].set_xlabel(f"atstumas išilgai, m (nuo {frame['start']} galo)", fontsize=8)
            cb = fig.colorbar(im, ax=rows, shrink=0.9, pad=0.005, aspect=30)
            cb.set_label("dz = faktas − projektas, mm", fontsize=8)
            if sections:
                a = fig.add_subplot(gsp[nrow])
                x = [(r["from"] + r["to"]) / 2 for r in sections]
                wd = [r["to"] - r["from"] for r in sections]
                a.bar(x, [r["cut"] for r in sections], width=wd, color="#c0392b", alpha=0.8, edgecolor="w",
                      label="žemiau projekto, m³")
                a.bar(x, [-r["fill"] for r in sections], width=wd, color="#2e6fb7", alpha=0.8, edgecolor="w",
                      label="aukščiau projekto, m³ (žemyn)")
                a2 = a.twinx()
                a2.plot(x, [r["max"] for r in sections], "k.-", lw=0.8, ms=3, label="max gylis, mm")
                a2.set_ylabel("max gylis, mm", fontsize=7); a2.tick_params(labelsize=7)
                a.axhline(0, color="k", lw=0.5)
                a.set_xlim(0, L); a.tick_params(labelsize=7)
                a.set_ylabel("m³", fontsize=7)
                a.legend(fontsize=7, loc="upper left"); a2.legend(fontsize=7, loc="upper right")
                a.set_title(f"Tūriai ruožuose kas {p['sec_len']:g} m", fontsize=9)
        else:
            fig = plt.figure(figsize=(14, 12), layout="constrained")
            gsp = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            a = fig.add_subplot(gsp[0])
            ext = [dz.x0 - dz.step / 2, dz.x0 + (dz.z.shape[1] - 0.5) * dz.step,
                   dz.y0 - dz.step / 2, dz.y0 + (dz.z.shape[0] - 0.5) * dz.step]
            im = a.imshow(dz.z * 1000, origin="lower", extent=ext, cmap=cmap, vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
            for r in design.boundary():
                a.plot(r[:, 0], r[:, 1], "-", color="0.35", lw=0.8)
            for z in zones:
                a.plot(z.rings[0][:, 0], z.rings[0][:, 1], "-", color=zc(z), lw=1)
            for z in lab:
                a.annotate(z.name, z.xy_max, xytext=(3, 3), textcoords="offset points", fontsize=7, color=zc(z),
                           weight="bold")
            a.set_aspect("equal"); a.ticklabel_format(useOffset=False, style="plain"); a.tick_params(labelsize=7)
            a.set_title(title, fontsize=9)
            cb = fig.colorbar(im, ax=a, shrink=0.8)
            cb.set_label("dz = faktas − projektas, mm", fontsize=8)
        a = fig.add_subplot(gsp[-1])
        d = dz.z[ok] * 1000
        lim = max(vmax * 1.5, 50)
        bins = np.linspace(-lim, lim, 121)
        a.hist(np.clip(raw * 1000, -lim, lim), bins=bins, color="0.75", density=True, label="taškai prieš pritaikymą")
        a.hist(np.clip(d, -lim, lim), bins=bins, color="#1f5aa6", alpha=0.65, density=True,
               label="tinklelis po pritaikymo")
        for i, v in enumerate(levels):
            a.axvline(-v, ls="--", lw=0.9, color=lvl_col[i % len(lvl_col)], label=f"{'ABCDEFGH'[i]}: -{v:.0f} mm")
        for t in (-p["ignore"], p["ignore"]):
            a.axvline(t, ls=":", lw=0.9, color="0.3")
        a.axvline(0, color="k", lw=0.6)
        a.set_xlabel("dz, mm", fontsize=8); a.legend(fontsize=7, loc="upper left"); a.set_yticks([])
        a.tick_params(labelsize=7)
        a.set_title("Skirtumų pasiskirstymas (taškinė linija - triukšmo riba, į tūrius neįtraukiama)", fontsize=9)
        fig.savefig(path, dpi=115)
        plt.close(fig)


TOOL = Pavirsiai()

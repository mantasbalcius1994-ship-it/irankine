"""Išpildomųjų taškų kontrolė: matuoti taškai prieš projektinį paviršių (dz), su PK / atstumu nuo ašies,
leistinais nuokrypiais pagal kodą ir atitikties procentu."""
from pathlib import Path

import numpy as np

from ...core.tool import FileInput, Param, Tool
from ...geo import loaders
from ...geo.report import Col, Sheet, write_xlsx


def _parse_map(text, what):
    """'AS=Danga, PAG=Pagrindas' arba 'AS:10/15' -> {raktas: reikšmė-tekstas}."""
    out = {}
    for part in str(text or "").replace(";", ",").split(","):
        if not part.strip():
            continue
        sep = "=" if "=" in part else ":"
        k, _, v = part.partition(sep)
        if not v.strip():
            raise ValueError(f"{what}: „{part.strip()}“ - reikia formato KODAS{sep}reikšmė")
        out[k.strip().upper()] = v.strip()
    return out


def _tol(v, what):
    """'10' -> (10,10); '10/15' -> (+10, -15) mm."""
    a, _, b = str(v).partition("/")
    try:
        up = float(a.replace(",", "."))
        dn = float(b.replace(",", ".")) if b else up
    except ValueError:
        raise ValueError(f"{what}: „{v}“ - reikia mm, pvz. 10 arba 10/15 (+ aukštyn / - žemyn)")
    return abs(up), abs(dn)


class Kontrole(Tool):
    id = "kontrole"
    title = "Išpildomųjų taškų kontrolė"
    group = "Paviršiai"
    icon = "✓"
    description = ("Matuoti taškai prieš projektinį paviršių: dz kiekviename taške, PK ir atstumas nuo ašies, "
                   "leistini nuokrypiai pagal taško kodą, atitikties procentas, blogi taškai DXF ir Excel.")
    help = ("<ul><li><b>dz = matuota − projektas</b> (+ aukščiau projekto).</li>"
            "<li>Leistini nuokrypiai: bendri arba pagal kodą, pvz. <code>AS:10, PAG:20/30</code> "
            "(+20 aukštyn, −30 žemyn), mm.</li>"
            "<li>Keli paviršiai (pvz. danga ir pagrindas viename LandXML): <code>AS=Danga, PAG=Pagrindas</code>.</li>"
            "<li>Trasa (LandXML) nebūtina - jei yra, ataskaitoje PK ir atstumas nuo ašies (+ dešinėje).</li>"
            "<li>Projektas prie taškų NEpritaikomas - kontrolei rodomas tikras skirtumas; vidurkis ir σ - "
            "suvestinėje (sisteminė paklaida matosi iš vidurkio).</li></ul>")
    inputs = [FileInput("points", "Matuoti taškai (CSV / TXT / DXF)", accept=".csv,.txt,.dxf", multiple=True),
              FileInput("design", "Projektinis paviršius (TTM / LandXML / DXF 3DFACE)", accept=".ttm,.xml,.dxf",
                        multiple=True),
              FileInput("alignment", "Trasa (LandXML, nebūtina)", accept=".xml", multiple=False, required=False)]
    params = [
        Param("tol", "Leistinas nuokrypis", "str", "20", "mm", "Leistini nuokrypiai",
              help="Vienas skaičius (±) arba +aukštyn/−žemyn, pvz. 15/20."),
        Param("tol_codes", "Pagal kodą", "str", "", group="Leistini nuokrypiai",
              help="Pvz. AS:10, PAG:20/30. Kodai be nurodymo - bendras nuokrypis."),
        Param("req_pct", "Reikalaujama atitiktis", "float", 95.0, "%", "Leistini nuokrypiai", min=0, max=100,
              help="Kiek taškų turi tilpti į leistinus nuokrypius (pagal kodą ir bendrai)."),
        Param("codes", "Tik šie kodai", "str", "", group="Taškai", help="Kableliais. Tuščia = visi."),
        Param("skip_codes", "Praleisti kodus", "str", "", group="Taškai",
              help="Pvz. BAZE, KONTR - stoties, kontroliniai ir kt. taškai."),
        Param("surf_map", "Paviršius pagal kodą", "str", "", group="Projektas",
              help="Kai paviršių keli: AS=Danga, PAG=Pagrindas (paviršiaus pavadinimas arba failo vardas). "
                   "Kodai be nurodymo - pirmas paviršius."),
        Param("label", "Užrašai DXF", "choice", "bad", group="Išvestis", choices=[
            ["bad", "Tik už nuokrypio"], ["all", "Visi taškai"], ["none", "Be užrašų"]]),
        Param("text_h", "Užrašo aukštis", "float", 0.3, "m", "Išvestis", min=0.01),
    ]

    def run(self, ctx):
        p, log = ctx.params, ctx.log
        # --- taškai
        xyz, names, codes, src = [], [], [], []
        for f in ctx.inputs["points"]:
            a, n, c = loaders.load_point_set(f, log)
            xyz.append(a); names += n; codes += [str(x).strip() for x in c]; src += [f.name] * len(a)
            log(f"Taškai {f.name}: {len(a)}")
        xyz = np.vstack(xyz)
        codes_u = np.array([c.upper() for c in codes])
        keep = np.ones(len(xyz), bool)
        inc = {c.strip().upper() for c in p["codes"].split(",") if c.strip()}
        exc = {c.strip().upper() for c in p["skip_codes"].split(",") if c.strip()}
        if inc:
            keep &= np.isin(codes_u, list(inc))
        if exc:
            keep &= ~np.isin(codes_u, list(exc))
        if not keep.any():
            raise ValueError("Po kodų filtro neliko taškų. Patikrinkite „Tik šie kodai“ / „Praleisti kodus“.")
        if (~keep).any():
            log(f"Praleista pagal kodą: {(~keep).sum()}")
        idx = np.nonzero(keep)[0]
        xyz, codes_u = xyz[idx], codes_u[idx]
        names = [names[i] for i in idx]; codes = [codes[i] for i in idx]; src = [src[i] for i in idx]

        # --- paviršiai
        surfs = []
        for f in ctx.inputs["design"]:
            for s in loaders.load_surfaces(f):
                s.file = Path(f).stem
                surfs.append(s)
                log(f"Paviršius {s.name} ({f.name}): {len(s.pts)} tšk., {len(s.tris)} trikampių")
        smap = _parse_map(p["surf_map"], "Paviršius pagal kodą")
        by_name = {}
        for s in surfs:
            by_name.setdefault(s.name.upper(), s); by_name.setdefault(s.file.upper(), s)
        for k, v in smap.items():
            if v.upper() not in by_name:
                raise ValueError(f"Paviršius „{v}“ (kodui {k}) nerastas. Yra: {', '.join(s.name for s in surfs)}")
        if len(surfs) > 1 and not smap:
            log(f"ĮSPĖJIMAS: paviršių {len(surfs)}, „Paviršius pagal kodą“ nenurodyta - visiems taškams "
                f"naudojamas {surfs[0].name}")
        sid = np.array([surfs.index(by_name[smap[c].upper()]) if c in smap else 0 for c in codes_u])
        zd = np.full(len(xyz), np.nan)
        for k, s in enumerate(surfs):
            m = sid == k
            if m.any():
                zd[m] = s.z_at(xyz[m, :2])
        dz = xyz[:, 2] - zd
        out = np.isnan(zd)
        if out.all():
            raise ValueError("Nė vienas taškas nepatenka ant projektinio paviršiaus. Patikrinkite koordinates "
                             "(LKS-94, X = E) ir ar parinktas tas paviršius.")
        if out.any():
            log(f"Už paviršiaus ribų: {out.sum()} tšk. (ataskaitoje pažymėti, į statistiką neįtraukti)")

        # --- leistini nuokrypiai
        t_up, t_dn = _tol(p["tol"], "Leistinas nuokrypis")
        tmap = {k: _tol(v, f"Kodas {k}") for k, v in _parse_map(p["tol_codes"], "Pagal kodą").items()}
        up = np.array([tmap.get(c, (t_up, t_dn))[0] for c in codes_u]) / 1000
        dn = np.array([tmap.get(c, (t_up, t_dn))[1] for c in codes_u]) / 1000
        status = np.where(out, "už ribų", np.where(dz > up, "per aukštai", np.where(dz < -dn, "per žemai", "OK")))
        ok = status == "OK"

        # --- trasa
        sta = off = None
        al = None
        if ctx.inputs.get("alignment"):
            from ...geo import io_landxml
            als = io_landxml.read_alignments(ctx.inputs["alignment"][0])
            if not als:
                raise ValueError("Trasos faile nerasta Alignment")
            if len(als) > 1:   # artimiausia trasa
                d = []
                for a in als:
                    s_, o_, ins = a.station_offset(xyz[:, :2])
                    d.append(np.median(np.where(ins, np.abs(o_), 1e9)))
                al = als[int(np.argmin(d))]
                log(f"Trasų {len(als)}, parinkta artimiausia: {al.name}")
            else:
                al = als[0]
            sta, off, ins = al.station_offset(xyz[:, :2])
            log(f"Trasa {al.name}: PK {sta.min():.1f}…{sta.max():.1f}, už trasos galų {(~ins).sum()} tšk.")

        # --- statistika
        valid = ~out
        req = p["req_pct"]

        def stats(m, code):
            d = dz[m & valid] * 1000
            n = len(d)
            if not n:
                return None
            k = ok[m].sum()
            u, w = tmap.get(code, (t_up, t_dn)) if code != "VISI" else (t_up, t_dn)
            pct = 100 * k / n
            return {"Kodas": code, "Taškų": n, "Leistina, mm": f"+{u:g} / −{w:g}" if code != "VISI" else "pagal kodą",
                    "Vidurkis, mm": round(float(d.mean()), 1), "σ, mm": round(float(d.std(ddof=1)) if n > 1 else 0, 1),
                    "Min, mm": round(float(d.min())), "Max, mm": round(float(d.max())),
                    "Per aukštai": int((status[m] == "per aukštai").sum()),
                    "Per žemai": int((status[m] == "per žemai").sum()),
                    "Atitiktis, %": round(pct, 1), "Įvertinimas": "atitinka" if pct >= req else "NEATITINKA"}
        rows_c = [r for c in sorted(set(codes_u)) if (r := stats(codes_u == c, c or "(be kodo)"))]
        tot = stats(np.ones(len(xyz), bool), "VISI")
        rows_c.append(tot)
        log(f"\nAtitiktis {tot['Atitiktis, %']} % (reikalaujama {req:g} %) - {tot['Įvertinimas']}; "
            f"vidurkis {tot['Vidurkis, mm']:+} mm, σ {tot['σ, mm']} mm, per aukštai {tot['Per aukštai']}, "
            f"per žemai {tot['Per žemai']}")
        for r in rows_c[:-1]:
            log(f"  {r['Kodas']}: {r['Taškų']} tšk., {r['Atitiktis, %']} %, vid. {r['Vidurkis, mm']:+} mm, "
                f"σ {r['σ, mm']} mm ({r['Įvertinimas']})")
        if abs(tot["Vidurkis, mm"]) > 0.5 * min(t_up, t_dn) and tot["Taškų"] > 10:
            log(f"PASTABA: vidurkis {tot['Vidurkis, mm']:+} mm - galima sisteminė paklaida (bazė, antenos aukštis, "
                "ne tas paviršius) arba visas sluoksnis išpildytas žemiau / aukščiau.")

        # --- išvestis
        stem = Path(ctx.inputs["points"][0]).stem
        outd = ctx.outdir
        rows = []
        for i in range(len(xyz)):
            r = {"Nr": names[i], "Kodas": codes[i], "E": round(xyz[i, 0], 3), "N": round(xyz[i, 1], 3),
                 "Z": round(xyz[i, 2], 3), "Z proj.": None if out[i] else round(zd[i], 3),
                 "dz, mm": None if out[i] else round(dz[i] * 1000, 1), "Leistina": f"+{up[i] * 1000:g}/−{dn[i] * 1000:g}",
                 "Būsena": status[i], "Paviršius": surfs[sid[i]].name, "Failas": src[i]}
            if sta is not None:
                r["PK"] = round(float(sta[i]), 2); r["Atstumas, m"] = round(float(off[i]), 3)
            rows.append(r)
        cols = [Col("Nr", "Nr", width=10), Col("Kodas", "Kodas", width=10)]
        if sta is not None:
            cols += [Col("PK", "PK", "0.00"), Col("Atstumas nuo ašies, m (+ dešinėje)", "Atstumas, m", "0.000", 14)]
        cols += [Col("E", "E", "0.000", 13), Col("N", "N", "0.000", 13), Col("Z matuota", "Z", "0.000"),
                 Col("Z projekto", "Z proj.", "0.000"), Col("dz, mm", "dz, mm", "0.0"), Col("Leistina, mm", "Leistina"),
                 Col("Būsena", "Būsena", width=12), Col("Paviršius", "Paviršius", width=16), Col("Failas", "Failas", width=16)]
        if sta is not None:
            o = np.lexsort((np.nan_to_num(off), np.nan_to_num(sta)))
            rows = [rows[i] for i in o]
        info = [("Taškai", ", ".join(f.name for f in ctx.inputs["points"])),
                ("Paviršiai", ", ".join(s.name for s in surfs)),
                ("Trasa", al.name if al else "-"),
                ("Leistina (bendra), mm", f"+{t_up:g} / −{t_dn:g}"),
                ("Pagal kodą", p["tol_codes"] or "-"), ("Reikalaujama atitiktis, %", req),
                ("Atitiktis, %", tot["Atitiktis, %"]), ("Įvertinimas", tot["Įvertinimas"])]
        sc = [Col(k, k, width=14 if k in ("Leistina, mm", "Įvertinimas") else None) for k in rows_c[0]]
        write_xlsx(outd / f"{stem}_kontrole.xlsx", [
            Sheet("Suvestinė", sc, rows_c, bad=lambda r: r["Įvertinimas"] != "atitinka", bad_cols=["Atitiktis, %"],
                  info=info),
            Sheet("Taškai", cols, rows, bad=lambda r: r["Būsena"] != "OK", bad_cols=["dz, mm", "Būsena"])],
            "Išpildomųjų taškų kontrolė")
        ctx.add_output(outd / f"{stem}_kontrole.xlsx", "Excel ataskaita (suvestinė pagal kodus, visi taškai)")

        from ...geo.io_csv import write_points
        write_points(outd / f"{stem}_dz.csv", xyz, names,
                     [f"{c} {'' if out[i] else f'{dz[i] * 1000:+.0f}'} {status[i]}".strip() for i, c in enumerate(codes)],
                     order="PNEZ")
        ctx.add_output(outd / f"{stem}_dz.csv", "Taškai su dz kode (P,N,E,Z,kodas) - importui į TBC")
        self._dxf(outd / f"{stem}_kontrole.dxf", xyz, names, dz, status, p, surfs)
        ctx.add_output(outd / f"{stem}_kontrole.dxf", "DXF: taškai pagal būseną su dz užrašais")
        self._plot(outd / f"{stem}_kontrole.png", xyz, dz, status, up, dn, sta, off, surfs, codes_u, tot, req)
        ctx.add_output(outd / f"{stem}_kontrole.png", "Peržiūra: planas, dz išilgai, pasiskirstymas", "image")
        ctx.preview = {"image": f"{stem}_kontrole.png",
                       "table": [{k: r[k] for k in ("Kodas", "Taškų", "Vidurkis, mm", "σ, mm", "Atitiktis, %",
                                                     "Įvertinimas")} for r in rows_c]}

    @staticmethod
    def _dxf(path, xyz, names, dz, status, p, surfs):
        from ...geo.io_dxf import DxfWriter
        w = DxfWriter()
        lay = {"OK": ("KONTR_OK", 3), "per aukštai": ("KONTR_AUKSCIAU", 5), "per žemai": ("KONTR_ZEMIAU", 1),
               "už ribų": ("KONTR_UZ_RIBU", 8)}
        h = p["text_h"]
        for s in surfs:
            for r in s.boundary():
                w.polyline3d(r, "PROJEKTO_RIBA", 8, closed=True)
        for i, q in enumerate(xyz):
            L, c = lay[status[i]]
            w.point(q, L, c)
            w.circle(q[:2], h * 0.4, L, c)
            if p["label"] == "all" or (p["label"] == "bad" and status[i] != "OK"):
                t = "?" if np.isnan(dz[i]) else f"{dz[i] * 1000:+.0f}"
                w.text(t, (q[0] + h * 0.6, q[1] - h * 0.5, q[2]), h, L + "_TXT", c)
        w.save(path)

    @staticmethod
    def _plot(path, xyz, dz, status, up, dn, sta, off, surfs, codes_u, tot, req):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"OK": "#2a9d4b", "per aukštai": "#1f5aa6", "per žemai": "#c0392b", "už ribų": "#999999"}
        has_sta = sta is not None
        fig = plt.figure(figsize=(15, 11), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[1.6, 1])
        a = fig.add_subplot(gs[0, :])
        if has_sta:
            for k, c in col.items():
                m = status == k
                if m.any():
                    a.scatter(sta[m], off[m], s=10 if k == "OK" else 22, c=c, label=f"{k} ({m.sum()})", zorder=3 if k != "OK" else 2)
            a.set_xlabel("PK, m"); a.set_ylabel("atstumas nuo ašies, m (+ dešinėje)")
            a.invert_yaxis()
            a.axhline(0, color="k", lw=0.6)
            a.set_title("Taškai trasos koordinatėse", fontsize=10)
        else:
            for s in surfs:
                for r in s.boundary():
                    a.plot(r[:, 0], r[:, 1], "-", color="0.6", lw=0.7)
            for k, c in col.items():
                m = status == k
                if m.any():
                    a.scatter(xyz[m, 0], xyz[m, 1], s=10 if k == "OK" else 22, c=c, label=f"{k} ({m.sum()})")
            a.set_aspect("equal", adjustable="datalim"); a.ticklabel_format(useOffset=False, style="plain")
            a.set_title("Taškų planas", fontsize=10)
        a.legend(fontsize=8, loc="upper right"); a.tick_params(labelsize=7)
        v = ~np.isnan(dz)
        x = sta if has_sta else np.arange(len(dz))
        cs = [c_ for c_ in sorted(set(codes_u[v]), key=lambda q: -(codes_u[v] == q).sum())]
        groups = [(c_, codes_u == c_) for c_ in cs[:3]]
        if len(cs) > 3:
            groups.append(("kiti", ~np.isin(codes_u, cs[:3])))
        sub = gs[1, 0].subgridspec(len(groups), 1, hspace=0.05)
        for gi, (name, gm) in enumerate(groups):
            b = fig.add_subplot(sub[gi])
            for k, c_ in col.items():
                m = (status == k) & v & gm
                if m.any():
                    b.scatter(x[m], dz[m] * 1000, s=8, c=c_)
            u_, d_ = np.unique(up[gm & v]), np.unique(dn[gm & v])
            if len(u_) == 1 and len(d_) == 1:
                b.axhspan(-d_[0] * 1000, u_[0] * 1000, color="#2a9d4b", alpha=0.12)
            b.axhline(0, color="k", lw=0.6)
            b.text(0.005, 0.95, f"{name} ({(gm & v).sum()})", transform=b.transAxes, fontsize=8, va="top",
                   weight="bold")
            b.tick_params(labelsize=7); b.set_ylabel("dz, mm", fontsize=7)
            if gi == 0:
                b.set_title("dz = matuota − projektas (žalia juosta - leistina)", fontsize=10)
            if gi < len(groups) - 1:
                b.tick_params(labelbottom=False)
            else:
                b.set_xlabel("PK, m" if has_sta else "taško eilės nr.")
        c = fig.add_subplot(gs[1, 1])
        d = dz[v] * 1000
        lim = max(np.percentile(np.abs(d), 99.5) * 1.1, up.max() * 1500, 10)
        c.hist([np.clip(dz[v & gm] * 1000, -lim, lim) for _, gm in groups], bins=60, stacked=True,
               label=[g[0] for g in groups])
        c.legend(fontsize=7)
        c.axvline(0, color="k", lw=0.6)
        c.set_xlabel("dz, mm"); c.set_yticks([]); c.tick_params(labelsize=7)
        c.set_title(f"Atitiktis {tot['Atitiktis, %']} % (reik. {req:g} %) - {tot['Įvertinimas']}\n"
                    f"vid. {tot['Vidurkis, mm']:+} mm, σ {tot['σ, mm']} mm", fontsize=10,
                    color="#2a9d4b" if tot["Įvertinimas"] == "atitinka" else "#c0392b")
        fig.savefig(path, dpi=110)
        plt.close(fig)


TOOL = Kontrole()

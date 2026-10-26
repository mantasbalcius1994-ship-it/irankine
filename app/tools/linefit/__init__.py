"""Linijų optimizavimas: trys režimai viename įrankyje.
  retinti  - perteklinės viršūnės šalinamos, likusios originalios (tas pats DXF)
  lankai   - tangentiškai sujungtos tiesės ir lankai + profilis (VPI, parabolės) -> LandXML, DXF
  planas   - tiesės ir lankai plane, aukščiai iš originalių taškų (pvz. bortai)
Valymas (šuoliai, kabliukai, dublikatai, sujungimas) taikomas visiems režimams."""
from pathlib import Path

import numpy as np

from ...core.tool import FileInput, Param, Tool
from . import clean, engine

FIT = "mode!=retinti"


class LineFit(Tool):
    id = "linefit"
    title = "Linijų optimizavimas"
    group = "Linijos"
    icon = "〰"
    description = ("Retina perteklines viršūnes (pvz. po pritraukimo prie paviršiaus) arba pakeičia linijas "
                   "tangentiškai sujungtomis tiesėmis ir lankais su profiliu. Prieš tai išvalo šuolius, galų "
                   "kabliukus, dublikatus ir sujungia nutrūkusias linijas.")
    help = ("<ul><li><b>Retinti</b>: paliekamos tik originalios viršūnės, jų padėtis ir aukščiai nekeičiami. "
            "Kiekviena pašalinta viršūnė yra ne toliau kaip tolerancija nuo naujos linijos - plane ir aukštyje. "
            "Lankai (bulge) išlaikomi. Rezultatas - tas pats DXF, kiti objektai nepaliesti.</li>"
            "<li><b>Tiesės ir lankai</b>: rezultatas LandXML alignment (TBC) su profiliu, DXF ir ataskaita.</li>"
            "<li><b>Tik planas</b>: plane tiesės ir lankai, aukščiai paimami iš originalių taškų - profilis "
            "nekeičiamas (pvz. bortai).</li>"
            "<li>Tolerancija turi būti didesnė už duomenų triukšmą, kitaip elementų daugės.</li></ul>")
    inputs = [FileInput("files", "Linijos (DXF / CSV)", accept=".dxf,.csv,.txt", multiple=True)]
    params = [
        Param("mode", "Ką daryti su linijomis", "choice", "retinti", group="Režimas", choices=[
            ["retinti", "Retinti viršūnes"],
            ["lankai", "Tiesės ir lankai + profilis"],
            ["planas", "Tiesės ir lankai, aukščiai nekeičiami"]]),
        Param("layers", "DXF sluoksniai", "str", "", group="Įvestis",
              help="Kableliais atskirti sluoksnių pavadinimai. Tuščia = visi."),
        # retinimas
        Param("tol_h", "Max nuokrypis plane", "float", 10.0, "mm", "Retinimas", min=0.1, show_if="mode=retinti"),
        Param("tol_v", "Max nuokrypis aukštyje", "float", 5.0, "mm", "Retinimas", min=0.0, show_if="mode=retinti",
              help="0 = aukščių profilis išlaikomas tiksliai."),
        # planas
        Param("h_tol_line", "Max nuokrypis (tiesė)", "float", 20.0, "mm", "Planas", min=0.1, step=1, show_if=FIT),
        Param("h_tol_arc", "Max nuokrypis (lankas)", "float", 20.0, "mm", "Planas", min=0.1, step=1, show_if=FIT),
        Param("r_min", "Min spindulys", "float", 5.0, "m", "Planas", min=0.1, show_if=FIT),
        Param("r_max", "Max spindulys", "float", 10000.0, "m", "Planas", min=1, show_if=FIT,
              help="Didesnio spindulio lankai laikomi tiesėmis."),
        Param("split_angle", "Skaidymo kampas", "float", 60.0, "°", "Planas", min=0, max=180, show_if=FIT,
              help="Plano lūžiai, aštresni už šį kampą, dalija liniją. 0 = neskaidyti."),
        Param("min_len", "Min elemento ilgis", "float", 0.5, "m", "Planas", min=0, show_if=FIT),
        # profilis
        Param("v_tol", "Max aukščio nuokrypis", "float", 10.0, "mm", "Profilis", min=0.1, step=1, show_if="mode=lankai"),
        Param("v_min_dg", "Min nuolydžių skirtumas VPI", "float", 0.10, "%", "Profilis", min=0, step=0.05,
              show_if="mode=lankai"),
        Param("v_max_len", "Max vertikalios kreivės ilgis", "float", 300.0, "m", "Profilis", min=0, show_if="mode=lankai"),
        Param("v_sharp_dg", "VPI be kreivės, kai Δi >", "float", 0.0, "%", "Profilis", min=0, show_if="mode=lankai",
              help="Aštrūs profilio lūžiai paliekami be parabolės. 0 = išjungta."),
        # valymas
        Param("spikes", "Šalinti šuolius", "bool", True, group="Valymas",
              help="Viršūnė, kur linija staigiai grįžta atgal (posūkis > 150°)."),
        Param("spike_len", "Šuolio max ilgis", "float", 0.5, "m", "Valymas", min=0.001, enabled_by="spikes"),
        Param("hooks", "Kirpti galų kabliukus", "bool", True, group="Valymas",
              help="Trumpas, staigiai pasuktas paskutinis segmentas (> 30°)."),
        Param("hook_len", "Kabliuko max ilgis", "float", 20.0, "mm", "Valymas", min=0.1, enabled_by="hooks"),
        Param("dups", "Šalinti dublikatus", "bool", False, group="Valymas",
              help="Linija, kuri VISA telpa į toleranciją nuo kitos (ilgesnės) to paties sluoksnio linijos."),
        Param("dup_tol", "Dublikato tolerancija", "float", 10.0, "mm", "Valymas", min=0.1, enabled_by="dups"),
        Param("join", "Sujungti nutrūkusias linijas", "bool", False, group="Valymas",
              help="To paties sluoksnio linijos, kurių galai arčiau nei nurodyta. Išsišakojimai nejungiami."),
        Param("join_gap", "Max tarpas tarp galų", "float", 5.0, "mm", "Valymas", min=0.1, enabled_by="join"),
        # išvestis
        Param("name_tpl", "Sluoksnių šablonas", "str", "{L}", group="Išvestis", show_if=FIT,
              help="{L} = šaltinio sluoksnis. Kuriami {L}-AL, {L}-LS, {L}-ERR."),
        Param("report_tol", "Ataskaitos tolerancija", "float", 0.0, "mm", "Išvestis", min=0, show_if=FIT,
              help="0 = naudoti optimizavimo tolerancijas."),
    ]

    def run(self, ctx):
        p = ctx.params
        layers = [s.strip() for s in p["layers"].split(",") if s.strip()] or None
        if p["mode"] == "retinti":
            return clean.run_simplify(ctx, p, layers)

        tl, ta = p["h_tol_line"] / 1000, p["h_tol_arc"] / 1000
        st = engine.Settings(
            tol_line=tl, tol_arc=ta, r_min=p["r_min"], r_max=p["r_max"], min_len=p["min_len"],
            split_angle=p["split_angle"], vertical=p["mode"] == "lankai", keep_z=p["mode"] == "planas",
            v_tol=p["v_tol"] / 1000, v_min_dg=p["v_min_dg"], v_max_len=p["v_max_len"], v_sharp_dg=p["v_sharp_dg"],
            report_tol=p["report_tol"] / 1000, prethin=min(tl, ta) / 10)
        if st.r_min >= st.r_max:
            raise ValueError("Min spindulys turi būti mažesnis už max spindulį")
        lines, names = [], []
        for src in ctx.inputs["files"]:
            names.append(src.name)
            if src.suffix.lower() == ".dxf":
                doc, refs = clean.open_refs(src, layers, ctx.log)
                _, S = clean.new_stats()
                work, _, _ = clean.prepare(doc, refs, p, S, ctx.log)
                for r, P in work:
                    if isinstance(P, str):
                        continue
                    lines.append((r.layer, r.handle, r.pts if P is None else P))
            else:
                lines += engine.read_csv(str(src))
        src0 = ctx.inputs["files"][0]
        prefix = str(ctx.outdir / (Path(src0).stem + "_fit"))
        results, files = engine.run_lines(lines, st, prefix, p["name_tpl"] or "{L}", True, ctx.log, ", ".join(names))
        ctx.add_output(files["xml"], "LandXML alignment (TBC)")
        ctx.add_output(files["dxf"], "DXF (AL / LS / ERR)")
        ctx.add_output(files["report"], "Ataskaita", "text")
        ctx.add_output(files["png"], "Grafikas", "image")
        ctx.preview = engine.preview(results)
        ctx.log(f"Baigta: {len(results)} alignment'ai")


TOOL = LineFit()

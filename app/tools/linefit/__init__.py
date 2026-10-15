"""Linijų optimizavimas: tiesės/lankai plane, VPI/parabolės profilyje."""
from pathlib import Path

from ...core.tool import FileInput, Param, Tool
from . import engine


class LineFit(Tool):
    id = "linefit"
    title = "Linijų optimizavimas"
    group = "Linijos"
    icon = "〰"
    description = ("Pakeičia linijas (bortai, ašys, kraštai) tangentiškai sujungtomis tiesėmis ir lankais, "
                   "o profilį — VPI su parabolėmis, laikantis nuokrypio tolerancijų. "
                   "Rezultatas: LandXML alignment (TBC), DXF ir ataskaita.")
    help = ("<ul><li>Įvestis: DXF su 3D polilinijomis / LWPOLYLINE (lankai per bulge) arba CSV <code>x,y,z[,pavadinimas]</code>.</li>"
            "<li>Tolerancija turi būti didesnė už duomenų triukšmą, kitaip elementų daugės.</li>"
            "<li>Aštresniuose nei <i>skaidymo kampas</i> plano lūžiuose linija dalijama į atskirus alignment'us.</li></ul>")
    inputs = [FileInput("files", "Linijos (DXF / CSV)", accept=".dxf,.csv,.txt", multiple=True)]
    params = [
        Param("layers", "DXF sluoksniai", "str", "", group="Įvestis",
              help="Kableliais atskirti sluoksnių pavadinimai. Tuščia = visi."),
        Param("h_tol_line", "Max nuokrypis (tiesė)", "float", 20.0, "mm", "Planas", min=0.1, step=1),
        Param("h_tol_arc", "Max nuokrypis (lankas)", "float", 20.0, "mm", "Planas", min=0.1, step=1),
        Param("r_min", "Min spindulys", "float", 5.0, "m", "Planas", min=0.1),
        Param("r_max", "Max spindulys", "float", 10000.0, "m", "Planas", min=1,
              help="Didesnio spindulio lankai laikomi tiesėmis."),
        Param("split_angle", "Skaidymo kampas", "float", 60.0, "°", "Planas", min=0, max=180,
              help="Plano lūžiai, aštresni už šį kampą, dalija liniją. 0 = neskaidyti."),
        Param("min_len", "Min elemento ilgis", "float", 0.5, "m", "Planas", min=0),
        Param("vertical", "Optimizuoti profilį", "bool", True, group="Profilis"),
        Param("v_tol", "Max aukščio nuokrypis", "float", 10.0, "mm", "Profilis", min=0.1, step=1, enabled_by="vertical"),
        Param("v_min_dg", "Min nuolydžių skirtumas VPI", "float", 0.10, "%", "Profilis", min=0, step=0.05, enabled_by="vertical"),
        Param("v_max_len", "Max vertikalios kreivės ilgis", "float", 300.0, "m", "Profilis", min=0, enabled_by="vertical"),
        Param("v_sharp_dg", "VPI be kreivės, kai Δi >", "float", 0.0, "%", "Profilis", min=0, enabled_by="vertical",
              help="Aštrūs profilio lūžiai paliekami be parabolės. 0 = išjungta."),
        Param("name_tpl", "Sluoksnių šablonas", "str", "{L}", group="Išvestis",
              help="{L} = šaltinio sluoksnis. Kuriami {L}-AL, {L}-LS, {L}-ERR."),
        Param("report_tol", "Ataskaitos tolerancija", "float", 0.0, "mm", "Išvestis", min=0,
              help="0 = naudoti optimizavimo tolerancijas."),
    ]

    def run(self, ctx):
        p = ctx.params
        st = engine.Settings(
            tol_line=p["h_tol_line"] / 1000, tol_arc=p["h_tol_arc"] / 1000,
            r_min=p["r_min"], r_max=p["r_max"], min_len=p["min_len"], split_angle=p["split_angle"],
            vertical=p["vertical"], v_tol=p["v_tol"] / 1000, v_min_dg=p["v_min_dg"],
            v_max_len=p["v_max_len"], v_sharp_dg=p["v_sharp_dg"], report_tol=p["report_tol"] / 1000)
        if st.r_min >= st.r_max:
            raise ValueError("Min spindulys turi būti mažesnis už max spindulį")
        layers = [s.strip() for s in p["layers"].split(",") if s.strip()] or None
        src = ctx.inputs["files"]
        prefix = str(ctx.outdir / (Path(src[0]).stem + "_fit"))
        results, files = engine.run(src, st, prefix, p["name_tpl"] or "{L}", layers, True, ctx.log)
        ctx.add_output(files["xml"], "LandXML alignment (TBC)")
        ctx.add_output(files["dxf"], "DXF (AL / LS / ERR)")
        ctx.add_output(files["report"], "Ataskaita", "text")
        ctx.add_output(files["png"], "Grafikas", "image")
        ctx.preview = engine.preview(results)
        ctx.log(f"Baigta: {len(results)} alignment'ai")


TOOL = LineFit()

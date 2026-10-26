"""Linijų valymas ir retinimas: perteklinės viršūnės (pvz. po pritraukimo prie paviršiaus TBC),
šuoliai, galų kabliukai, dublikatai, nesujungti galai. Rezultatas - tas pats DXF, kiti objektai nepaliesti."""
from pathlib import Path

import numpy as np

from ...core.tool import FileInput, Param, Tool
from ...geo import io_dxf, linework, polyline, report


class LineClean(Tool):
    id = "lineclean"
    title = "Linijų valymas ir retinimas"
    group = "Linijos"
    icon = "✂"
    description = ("Pašalina nereikalingas viršūnes (pvz. po pritraukimo prie paviršiaus), šuolius, galų kabliukus, "
                   "dublikatus ir sujungia nutrūkusias linijas. Paliekamos tik originalios viršūnės - jų padėtis "
                   "ir aukščiai nekeičiami. Kiti DXF objektai lieka nepaliesti.")
    help = ("<ul><li>Retinimas garantuoja: kiekviena pašalinta viršūnė yra ne toliau kaip nurodyta tolerancija "
            "nuo naujos linijos - ir plane, ir aukštyje.</li>"
            "<li>Linijoms, pritrauktoms prie paviršiaus, aukščio tolerancija svarbiausia: ji nusako, kiek linija "
            "tarp likusių viršūnių gali „atitrūkti“ nuo paviršiaus.</li>"
            "<li>Lankai (bulge) išlaikomi nepakeisti, retinamos tik tiesios atkarpos tarp jų.</li></ul>")
    inputs = [FileInput("files", "DXF failai", accept=".dxf", multiple=True)]
    params = [
        Param("layers", "Sluoksniai", "str", "", group="Įvestis", help="Kableliais atskirti. Tuščia = visi."),
        Param("simplify", "Retinti viršūnes", "bool", True, group="Retinimas"),
        Param("tol_h", "Max nuokrypis plane", "float", 10.0, "mm", "Retinimas", min=0.1, enabled_by="simplify"),
        Param("tol_v", "Max nuokrypis aukštyje", "float", 5.0, "mm", "Retinimas", min=0.0, enabled_by="simplify",
              help="0 = aukščių profilis išlaikomas tiksliai (šalinamos tik viršūnės tiesiose 3D atkarpose)."),
        Param("spikes", "Šalinti šuolius", "bool", True, group="Valymas",
              help="Viršūnė, kur linija staigiai grįžta atgal (posūkis > 150°)."),
        Param("spike_len", "Šuolio max ilgis", "float", 0.5, "m", "Valymas", min=0.001, enabled_by="spikes"),
        Param("hooks", "Kirpti galų kabliukus", "bool", True, group="Valymas",
              help="Trumpas, staigiai pasuktas paskutinis segmentas (> 30°)."),
        Param("hook_len", "Kabliuko max ilgis", "float", 20.0, "mm", "Valymas", min=0.1, enabled_by="hooks"),
        Param("dups", "Šalinti dublikatus", "bool", False, group="Valymas",
              help="Linija, kuri VISA telpa į toleranciją nuo kitos (ilgesnės) to paties sluoksnio linijos - geometrija neprarandama."),
        Param("dup_tol", "Dublikato tolerancija", "float", 10.0, "mm", "Valymas", min=0.1, enabled_by="dups"),
        Param("join", "Sujungti nutrūkusias linijas", "bool", False, group="Valymas",
              help="To paties sluoksnio linijos, kurių galai arčiau nei nurodyta. Išsišakojimai nejungiami."),
        Param("join_gap", "Max tarpas tarp galų", "float", 5.0, "mm", "Valymas", min=0.1, enabled_by="join"),
    ]

    def run(self, ctx):
        p = ctx.params
        layers = [s.strip() for s in p["layers"].split(",") if s.strip()] or None
        tol_h = p["tol_h"] / 1000
        tol_v = max(p["tol_v"] / 1000, 1e-9)
        total = {}
        plan_items, rows = [], []
        for src in ctx.inputs["files"]:
            ctx.log(f"{src.name}: skaitoma…")
            doc = io_dxf.open_doc(src)
            refs = io_dxf.polyline_refs(doc, layers)
            if not refs:
                have = ", ".join(sorted(io_dxf.layers_in(src))) or "nėra"
                raise ValueError(f"{src.name}: nerasta linijų (LWPOLYLINE/POLYLINE)"
                                 + (f" sluoksniuose {', '.join(layers)}. Faile esantys sluoksniai: {have}" if layers else ""))
            ctx.log(f"  linijų: {len(refs)}, viršūnių: {sum(len(r.pts) for r in refs)}")
            st = {}

            def S(layer):
                return st.setdefault(layer, dict(lines0=0, lines1=0, v0=0, v1=0, dups=0, joins=0, spikes=0,
                                                  hooks=0, arcs=0, dh=0.0, dv=0.0))
            work = []                                  # [ref, P]
            marks = []                                 # vietos peržiūrai (šuoliai, kabliukai, sujungimai)
            for r in refs:
                s = S(r.layer)
                s["lines0"] += 1; s["v0"] += len(r.verts) if r.has_arcs else len(r.pts)
                if r.has_arcs:
                    s["arcs"] += 1
                    work.append([r, None])
                    continue
                P = polyline.dedupe(r.pts, 1e-6)
                if p["spikes"]:
                    P0 = P
                    P, k = polyline.remove_spikes(P, p["spike_len"])
                    if k:
                        s["spikes"] += k
                        gone = np.setdiff1d(np.arange(len(P0)), _match(P0, P))
                        marks += P0[gone, :2].tolist()
                if p["hooks"] and not r.closed:
                    P, k = polyline.trim_hooks(P, p["hook_len"] / 1000)
                    if k:
                        s["hooks"] += k; marks += [P[0, :2].tolist(), P[-1, :2].tolist()]
                work.append([r, P])

            removed_xy = []
            if p["dups"]:
                idx = [i for i, (r, P) in enumerate(work) if P is not None]
                by_layer = {}
                for i in idx:
                    by_layer.setdefault(work[i][0].layer, []).append(i)
                for lay, ids in by_layer.items():
                    d = linework.find_duplicates([work[i][1] for i in ids], p["dup_tol"] / 1000)
                    for a in d:
                        i = ids[a]
                        removed_xy.append(work[i][1][:, :2])
                        io_dxf.delete_entity(doc, work[i][0])
                        work[i][1] = "deleted"
                        S(lay)["dups"] += 1
                ctx.log(f"  dublikatų pašalinta: {sum(v['dups'] for v in st.values())}")

            if p["join"]:
                idx = [i for i, (r, P) in enumerate(work) if isinstance(P, np.ndarray) and not r.closed]
                sub = [work[i][1] for i in idx]
                chains = linework.join_ends(sub, p["join_gap"] / 1000, [work[i][0].layer for i in idx])
                for ch in chains:
                    if len(ch) < 2:
                        continue
                    first = idx[ch[0][0]]
                    for i, rev in ch[1:]:
                        P = sub[i][::-1] if rev else sub[i]
                        marks.append(P[0, :2].tolist())          # sujungimo vieta
                        io_dxf.delete_entity(doc, work[idx[i]][0])
                        work[idx[i]][1] = "deleted"
                    work[first][1] = linework.merge_chain(sub, ch)
                    S(work[first][0].layer)["joins"] += len(ch) - 1
                ctx.log(f"  sujungta galų: {sum(v['joins'] for v in st.values())}")

            out_lines = []
            for r, P in work:
                s = S(r.layer)
                if isinstance(P, str):
                    continue
                if P is None:                           # su lankais: lankai nekeičiami, retinamos tiesios atkarpos
                    V, B = r.verts, r.bulges
                    layer, n0 = r.layer, len(V)
                    if p["simplify"]:
                        arc = np.abs(B) > 1e-12
                        fixed = arc | np.roll(arc, 1)          # lanko pradžia ir pabaiga
                        keep = polyline.simplify_fixed(V, fixed, tol_h, tol_v, closed=r.closed)
                        Vx = np.vstack([V, V[:1]]) if r.closed else V
                        dh, dv = polyline.deviation(Vx, np.r_[keep, len(V)] if r.closed else keep)
                        s["dh"], s["dv"] = max(s["dh"], dh), max(s["dv"], dv)
                        new = io_dxf.replace_polyline_bulged(doc, r, V[keep], B[keep])
                    else:
                        keep, new = np.arange(n0), r.entity
                    s["lines1"] += 1; s["v1"] += len(keep)
                    out_lines.append((layer, n0, r.pts, V[keep]))
                    rows.append({"file": src.name, "layer": layer, "handle": new.dxf.handle,
                                 "v0": n0, "v1": len(keep), "len": polyline.length(r.pts)})
                    continue
                if len(P) < 2:
                    io_dxf.delete_entity(doc, r)
                    continue
                if p["simplify"]:
                    keep = polyline.simplify(P, tol_h, tol_v, closed=r.closed)
                    dh, dv = polyline.deviation(np.vstack([P, P[:1]]) if r.closed else P,
                                                np.r_[keep, len(P)] if r.closed else keep)
                    s["dh"], s["dv"] = max(s["dh"], dh), max(s["dv"], dv)
                    newP = P[keep]
                else:
                    newP = P
                layer = r.layer
                new = io_dxf.replace_polyline(doc, r, newP)
                s["lines1"] += 1; s["v1"] += len(newP)
                out_lines.append((layer, len(r.pts), newP, newP))
                rows.append({"file": src.name, "layer": layer, "handle": new.dxf.handle,
                             "v0": len(r.pts), "v1": len(newP), "len": polyline.length(newP)})

            out = ctx.outdir / f"{src.stem}_isvalyta.dxf"
            doc.saveas(out)
            ctx.add_output(out, f"Išvalytas DXF ({src.name})")
            for lay, v in st.items():
                total[(src.name, lay)] = v
            v0 = sum(v["v0"] for v in st.values()); v1 = sum(v["v1"] for v in st.values())
            ctx.log(f"  viršūnių: {v0} -> {v1} ({100 * (1 - v1 / max(v0, 1)):.0f} % mažiau)")
            plan_items.append(_preview_item(src.name, refs, out_lines, removed_xy, marks))

        xlsx = ctx.outdir / "ataskaita.xlsx"
        _write_report(xlsx, total, rows, p)
        ctx.add_output(xlsx, "Ataskaita (Excel)")
        ctx.preview = {"plan": plan_items, "charts": [],
                       "legend": [["line", "išvalyta linija"], ["joint", "likusios viršūnės"],
                                  ["pt", "pradinės viršūnės"], ["removed", "pašalintas dublikatas"],
                                  ["bad", "šuolys / kabliukas / sujungimas"]],
                       "table": [
            {"Failas": f, "Sluoksnis": lay, "Linijų": f"{v['lines0']} → {v['lines1']}",
             "Viršūnių": f"{v['v0']} → {v['v1']}",
             "Sumažėjo, %": round(100 * (1 - v["v1"] / max(v["v0"], 1))),
             "Dublikatų": v["dups"], "Sujungta": v["joins"], "Šuolių": v["spikes"], "Kabliukų": v["hooks"],
             "max Δplan, mm": round(v["dh"] * 1000, 1), "max Δz, mm": round(v["dv"] * 1000, 1)}
            for (f, lay), v in total.items()]}
        ctx.log("Baigta")


def _match(P0, P):
    """Kurie P0 taškai liko P (indeksai)."""
    from scipy.spatial import cKDTree
    d, i = cKDTree(P0[:, :2]).query(P[:, :2])
    return np.unique(i[d < 1e-9])


def _decimate(a, nmax):
    return a if len(a) <= nmax else a[np.linspace(0, len(a) - 1, nmax).astype(int)]


def _preview_item(name, refs, out_lines, removed_xy, marks):
    orig = np.vstack([r.pts[:, :2] for r in refs])
    els = [{"type": "line", "xy": np.round(G[:, :2], 3).tolist(), "L": round(polyline.length(G), 3), "s0": 0,
            "R": None, "label": f"{lay}: {v0} → {len(V)} viršūnių"} for lay, v0, G, V in out_lines]
    els += [{"type": "removed", "xy": np.round(xy, 3).tolist(), "L": None, "s0": 0, "R": None,
             "label": "Dublikatas (pašalintas)"} for xy in removed_xy]
    joints = np.vstack([V[:, :2] for *_, V in out_lines]) if out_lines else np.zeros((0, 2))
    n0 = sum(len(r.verts) if r.has_arcs else len(r.pts) for r in refs)
    return {"name": f"{name}: {n0} → {len(joints)} viršūnių",
            "points": np.round(_decimate(orig, 6000), 3).tolist(),
            "elements": els,
            "joints": np.round(_decimate(joints, 20000), 3).tolist(),
            "bad": np.round(np.array(marks).reshape(-1, 2), 3).tolist()}


def _write_report(path, total, rows, p):
    info = [("Max nuokrypis plane", f"{p['tol_h']} mm" if p["simplify"] else "-"),
            ("Max nuokrypis aukštyje", f"{p['tol_v']} mm" if p["simplify"] else "-")]
    s1 = report.Sheet("Suvestinė", [
        report.Col("Failas", "f", width=28), report.Col("Sluoksnis", "lay", width=30),
        report.Col("Linijų prieš", "lines0"), report.Col("Linijų po", "lines1"),
        report.Col("Viršūnių prieš", "v0"), report.Col("Viršūnių po", "v1"),
        report.Col("Sumažėjo", lambda r: 1 - r["v1"] / max(r["v0"], 1), "0%"),
        report.Col("Dublikatų", "dups"), report.Col("Sujungta", "joins"), report.Col("Šuolių", "spikes"),
        report.Col("Kabliukų", "hooks"), report.Col("Su lankais", "arcs"),
        report.Col("max Δplan, mm", lambda r: r["dh"] * 1000, "0.0"),
        report.Col("max Δz, mm", lambda r: r["dv"] * 1000, "0.0")],
        [dict(v, f=f, lay=lay) for (f, lay), v in total.items()], info=info,
        totals={"Linijų prieš": "SUM", "Linijų po": "SUM", "Viršūnių prieš": "SUM", "Viršūnių po": "SUM"})
    s2 = report.Sheet("Linijos", [
        report.Col("Failas", "file", width=28), report.Col("Sluoksnis", "layer", width=30),
        report.Col("Handle", "handle"), report.Col("Viršūnių prieš", "v0"), report.Col("Viršūnių po", "v1"),
        report.Col("Ilgis, m", "len", "0.000")], rows)
    report.write_xlsx(path, [s1, s2], "Linijų valymas ir retinimas")


TOOL = LineClean()

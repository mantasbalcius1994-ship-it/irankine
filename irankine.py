#!/usr/bin/env python3
"""Įrankinė komandinėje eilutėje (naudoja Claude skill'as „irankine“).

  python irankine.py tools                                  įrankiai ir jų parametrai
  python irankine.py info failas.dxf [kitas.xml ...]         failo turinys (sluoksniai, linijos, viršūnės, ...)
  python irankine.py run linefit failas.dxf --set mode=retinti --set tol_h=10 --out rezultatai
  python irankine.py run pavirsiai design=projektas.ttm scan=skenas.laz --out rezultatai

Parametrų vienetai - kaip formoje (mm / m / ° / %), žr. `tools`. Rezultatai: <out>/ (failai, preview.png,
preview.json, log.txt). Išėjimo kodas 0 - pavyko, 2 - vartotojo klaida (pranešimas lietuviškai).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_tools(_):
    from app.core.registry import discover
    for t in discover().values():
        print(f"\n== {t.id}: {t.title}\n   {t.description}")
        if len(t.inputs) > 1:
            print("   įvestys: " + ", ".join(f"{i.key}=<{i.label}>" for i in t.inputs))
        for p in t.params:
            unit = f" {p.unit}" if p.unit else ""
            extra = []
            if p.choices:
                extra.append("reikšmės: " + ", ".join(f"{k} ({v})" for k, v in p.choices))
            if p.show_if:
                extra.append(f"tik kai {p.show_if}")
            if p.enabled_by:
                extra.append(f"kai {p.enabled_by}=true")
            print(f"   {p.key:<12} = {p.default!s:<10}{unit:<4} {p.label}" + (f"  [{'; '.join(extra)}]" if extra else ""))
            if p.help:
                print(f"   {'':<12}   {p.help}")


def _info_dxf(f):
    import numpy as np
    from app.geo import io_dxf, polyline
    cnt = io_dxf.layers_in(f)
    doc = io_dxf.open_doc(f)
    refs = io_dxf.polyline_refs(doc)
    types = {}
    for e in doc.modelspace():
        types[e.dxftype()] = types.get(e.dxftype(), 0) + 1
    print(f"  objektai: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items(), key=lambda kv: -kv[1])))
    by = {}
    for r in refs:
        d = by.setdefault(r.layer, dict(n=0, v=0, L=0.0, zmin=np.inf, zmax=-np.inf, closed=0, arcs=0, flat=0))
        d["n"] += 1
        d["v"] += len(r.verts) if r.has_arcs else len(r.pts)
        d["L"] += polyline.length(r.pts)
        d["zmin"], d["zmax"] = min(d["zmin"], r.pts[:, 2].min()), max(d["zmax"], r.pts[:, 2].max())
        d["closed"] += r.closed
        d["arcs"] += r.has_arcs
        d["flat"] += np.ptp(r.pts[:, 2]) < 1e-9
    if not by:
        print("  linijų (LWPOLYLINE/POLYLINE) nėra; sluoksniai: " + ", ".join(f"{k} ({v})" for k, v in cnt.items()))
        return
    print(f"  {'sluoksnis':<36} {'linijų':>6} {'viršūnių':>9} {'ilgis, m':>10} {'vrš./m':>7} {'z min':>9} {'z max':>9}"
          f" {'uždarų':>6} {'su lank.':>8} {'plokšč.':>7}")
    for lay, d in sorted(by.items()):
        print(f"  {lay[:36]:<36} {d['n']:>6} {d['v']:>9} {d['L']:>10.1f} {d['v'] / max(d['L'], 1e-9):>7.2f}"
              f" {d['zmin']:>9.3f} {d['zmax']:>9.3f} {d['closed']:>6} {d['arcs']:>8} {d['flat']:>7}")


def cmd_info(a):
    for f in map(Path, a.files):
        print(f"\n{f.name}")
        ext = f.suffix.lower()
        try:
            if ext == ".dxf":
                _info_dxf(f)
            elif ext in (".csv", ".txt"):
                from app.geo import io_csv
                ps = io_csv.read_points(f)
                print(f"  taškų {len(ps)}, tvarka {ps.order}. {ps.note}")
                print(f"  E {ps.xyz[:, 0].min():.3f}…{ps.xyz[:, 0].max():.3f}, N {ps.xyz[:, 1].min():.3f}…"
                      f"{ps.xyz[:, 1].max():.3f}, Z {ps.xyz[:, 2].min():.3f}…{ps.xyz[:, 2].max():.3f}")
            elif ext == ".xml":
                from app.geo import io_landxml
                for fn, kind in ((io_landxml.read_surfaces, "paviršius"), (io_landxml.read_alignments, "trasa")):
                    try:
                        for o in fn(f):
                            if kind == "paviršius":
                                print(f"  paviršius {o['name']}: {len(o['pts'])} tšk., {len(o['tris'])} trikampių")
                            else:
                                print(f"  trasa {o.name}: {o.length:.3f} m, {len(o.elems)} el., "
                                      f"profilis {'yra' if o.profile is not None else 'nėra'}")
                    except ValueError:
                        pass
            elif ext == ".ttm":
                from app.geo import io_ttm
                from app.geo.surface import Surface
                d = io_ttm.read(f)
                S = Surface(d["pts"], d["tris"], d["name"])
                P = d["pts"]
                print(f"  paviršius {d['name']}: {len(P)} tšk., {len(d['tris'])} trikampių, plotas {S.area_2d():.1f} m²")
                print(f"  E {P[:, 0].min():.3f}…{P[:, 0].max():.3f}, N {P[:, 1].min():.3f}…{P[:, 1].max():.3f}, "
                      f"Z {P[:, 2].min():.3f}…{P[:, 2].max():.3f}")
            elif ext in (".las", ".laz"):
                from app.geo import io_las
                for k, v in io_las.info(f).items():
                    print(f"  {k}: {v}")
            else:
                print("  nežinomas formatas")
        except ValueError as e:
            print(f"  KLAIDA: {e}")


def render_preview(prev, path):
    """Peržiūros PNG: planas (visi elementai) + grafikai."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    charts = (prev.get("charts") or [])[:3]
    plan = prev.get("plan") or []
    n = (1 if plan else 0) + len(charts)
    if not n:
        return None
    fig = plt.figure(figsize=(14, 9 if not charts else 9 + 3 * len(charts)))
    gs = fig.add_gridspec(n, 1, height_ratios=([4] if plan else []) + [1.3] * len(charts))
    k = 0
    if plan:
        ax = fig.add_subplot(gs[0]); k = 1
        col = {"line": "#1f5aa6", "arc": "#e08a00", "removed": "#d0006f"}
        for it in plan:
            pts = np.asarray(it.get("points") or [], float).reshape(-1, 2)
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], ".", ms=1.5, color="0.7", zorder=1)
            for e in it.get("elements", []):
                xy = np.asarray(e["xy"], float)
                ax.plot(xy[:, 0], xy[:, 1], "--" if e["type"] == "removed" else "-",
                        color=col.get(e["type"], "#1f5aa6"), lw=1.2, zorder=2)
            bad = np.asarray(it.get("bad") or [], float).reshape(-1, 2)
            if len(bad):
                ax.plot(bad[:, 0], bad[:, 1], "o", mfc="none", mec="#d0006f", ms=8, zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("Planas: mėlyna - tiesė/linija, oranžinė - lankas, raudona punktyrinė - pašalinta, "
                     "rožiniai apskritimai - šuoliai/kabliukai/sujungimai", fontsize=9)
        ax.ticklabel_format(useOffset=False, style="plain")
    for c in charts:
        ax = fig.add_subplot(gs[k]); k += 1
        x = np.asarray(c["x"], float)
        for sr in c["series"]:
            ax.plot(x, sr["y"], lw=1, label=sr["name"])
            if sr.get("tol"):
                for t in (sr["tol"], -sr["tol"]):
                    ax.axhline(t, ls="--", lw=0.8, color="0.5")
        for v in c.get("vlines") or []:
            ax.axvline(v, color="0.85", lw=0.6)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(c["title"], fontsize=9); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _assign_inputs(tool, args):
    """Failai -> įrankio įvestys. `raktas=failas` - aiškiai; kitaip pagal eilę (vienas failas vienai įvesčiai,
    paskutinė „multiple“ įvestis pasiima likusius)."""
    specs = tool.inputs or []
    if not specs:
        return {"files": [Path(f).resolve() for f in args]}
    keys = {s.key for s in specs}
    res, rest = {}, []
    for f in args:
        k, sep, v = f.partition("=")
        if sep and k in keys:
            res.setdefault(k, []).append(Path(v).resolve())
        else:
            rest.append(Path(f).resolve())
    free = [s for s in specs if s.key not in res]
    for i, s in enumerate(free):
        if not rest:
            break
        if s.multiple and i == len(free) - 1:
            res[s.key] = rest; rest = []
        else:
            res[s.key] = [rest.pop(0)]
    if rest:
        sys.exit(f"Per daug failų. Įvestys: " + ", ".join(f"{s.key} ({s.label})" for s in specs))
    for s in specs:
        if s.required and not res.get(s.key):
            sys.exit(f"Trūksta įvesties „{s.key}“ ({s.label}). Nurodykite: {s.key}=<failas>")
        for f in res.get(s.key, []):
            if not f.exists():
                sys.exit(f"Failas nerastas: {f}")
    return res


def cmd_run(a):
    from app.core.registry import discover
    from app.core.tool import JobContext
    tools = discover()
    if a.tool not in tools:
        sys.exit(f"Nėra įrankio „{a.tool}“. Yra: {', '.join(tools)}")
    tool = tools[a.tool]
    raw = {}
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        if k not in {p.key for p in tool.params}:
            sys.exit(f"Nežinomas parametras „{k}“. Žr.: python irankine.py tools")
        raw[k] = v
    try:
        params = tool.parse_params(raw)
    except ValueError as e:
        print(f"KLAIDA: {e}"); sys.exit(2)
    out = Path(a.out or (Path(a.files[0]).stem + "_rezultatai")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w", encoding="utf-8")

    def log(m):
        print(m, flush=True); logf.write(m + "\n")
    inputs = _assign_inputs(tool, a.files)
    ctx = JobContext("cli", out, inputs, params, log, lambda v: None, out_dir=out)
    t0 = time.time()
    try:
        tool.run(ctx)
    except ValueError as e:
        log(f"KLAIDA: {e}"); sys.exit(2)
    log(f"Laikas: {time.time() - t0:.1f} s")
    if ctx.preview:
        (out / "preview.json").write_text(json.dumps(ctx.preview, ensure_ascii=False), encoding="utf-8")
        png = (out / ctx.preview["image"]) if ctx.preview.get("image") else render_preview(ctx.preview, out / "preview.png")
        tab = ctx.preview.get("table") or []
        if tab:
            log("\nSUVESTINĖ")
            cols = list(tab[0])
            log(" | ".join(cols))
            for r in tab[:60]:
                log(" | ".join(str(r[c]) for c in cols))
            if len(tab) > 60:
                log(f"... dar {len(tab) - 60} eilučių (žr. ataskaitą)")
    log("\nREZULTATAI")
    for o in ctx.outputs:
        log(f"  {out / o.name}  - {o.label}")
    if ctx.preview and not ctx.preview.get("image"):
        log(f"  {out / 'preview.png'}  - peržiūra (planas, grafikai)")


def main():
    ap = argparse.ArgumentParser(description="Įrankinė (CLI)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools", help="įrankiai ir parametrai")
    pi = sub.add_parser("info", help="failo turinys")
    pi.add_argument("files", nargs="+")
    pr = sub.add_parser("run", help="paleisti įrankį")
    pr.add_argument("tool")
    pr.add_argument("files", nargs="+")
    pr.add_argument("--set", action="append", metavar="RAKTAS=REIKŠMĖ")
    pr.add_argument("--out")
    a = ap.parse_args()
    {"tools": cmd_tools, "info": cmd_info, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()

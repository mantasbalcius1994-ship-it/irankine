# Įrankinė

Inžinerinių įrankių paketas naršyklėje. Veikia tame pačiame Hetzner serveryje kaip roadgrader, bet kaip atskira paslauga, todėl roadgrader neliečiamas.

```
app/
  main.py              FastAPI: API + puslapis
  core/tool.py         Tool / Param / FileInput – įrankio aprašas
  core/jobs.py         užduotys: atskiras procesas, būsena diske, žurnalas, valymas po 72 h
  core/registry.py     automatiškai randa app/tools/*/
  tools/linefit/       1-as įrankis: linijų optimizavimas (engine.py veikia ir iš komandinės eilutės)
  static/              naršyklės dalis (be priklausomybių)
deploy/                systemd, nginx, install.sh, env pavyzdys
tests/                 python tests/test_smoke.py
```

## Branduolys `app/geo` (bendras visiems įrankiams)

| Modulis | Ką daro |
|---|---|
| `io_dxf` | LWPOLYLINE/POLYLINE/3DFACE/POINT skaitymas (lankai išskaidomi), `DxfWriter` |
| `io_csv` | Taškų CSV/TXT: skyrikliai `, ; tab tarpas`, dešimtainis kablelis, P,E,N,Z / N,E,Z – LKS-94 ašių tvarka atpažįstama automatiškai |
| `io_landxml` | Paviršiai (TIN) ir trasos: tiesės, lankai, klotoidės, profilis (PVI, ParaCurve, CircCurve) |
| `io_las` | LAS/LAZ skaitymas dalimis: apkirpimas pagal ribas ar poligoną, klasės, retinimas (žemiausias taškas langelyje) |
| `surface` | `Surface` (TIN): aukštis taške (1 mln. taškų ~2 s), kraštai, plotas; `Grid`: DEM iš taškų/paviršiaus; tūris tarp paviršių; `fit_offset` – projektinio paviršiaus pritaikymas prie skenavimo (atsparus išplovoms) |
| `alignment` | `Alignment`: PK+atstumas ↔ XY (atstumas + dešinėje), aukštis pagal profilį, kryptis; `format_station` (km+m arba piketai) |
| `polyline` | dublikatai, ilgis, perskirstymas, RDP, posūkio kampai, skaidymas kampuose |
| `report` | Excel ataskaitos: antraštė, formatai, už tolerancijos paryškinimas, sumos |

Testai: `python tests/test_geo.py` (žinomi atsakymai: klotoidė pagal Fresnel'io integralus, tūris tarp plokštumų ir t. t.).

## Diegimas serveryje

**Vieną kartą** – serveryje (root, per SSH) įklijuoti:

```
curl -fsSL https://raw.githubusercontent.com/mantasbalcius1994-ship-it/irankine/main/deploy/setup.sh | bash
```

Skriptas: įdiegia paketus, paima kodą iš GitHub į `/opt/toolbox-src`, įdiegia Įrankinę į `/opt/toolbox`,
nustato nginx (prievadas 80, senos svetainės išjungiamos, kopijos `/root/nginx-backup`), paklausia slaptažodžio
ir įjungia automatinį atnaujinimą.

**Toliau – nieko daryti nereikia:** `toolbox-update.timer` kas 2 min. tikrina GitHub; radęs naują versiją
(ir jei tuo metu nevyksta skaičiavimas) pats ją įdiegia.

Naudingos komandos serveryje:

| Ką | Komanda |
|---|---|
| Įrankinės žurnalas | `journalctl -u toolbox -f` |
| Atnaujinimų žurnalas | `journalctl -u toolbox-update -n 50` |
| Atnaujinti dabar | `systemctl start toolbox-update` |
| Slaptažodis / ribos | `nano /etc/toolbox.env` → `systemctl restart toolbox` |

Ribos (`/etc/toolbox.env`, serveriui 2 branduoliai / 4 GB): skaičiuojama po vieną užduotį; TIN iki 2,5 mln. taškų,
taškų debesis iki 40 mln. (po retinimo), įkėlimas iki 500 MB. Jei užduotis viršija 3,3 GB, systemd nužudo tik jos
procesą – užduotis pažymima klaida, eilėje laukusios paleidžiamos iš naujo.

Testai: `python tests/test_geo.py`, `python tests/test_smoke.py`, `python tests/test_jobs.py`, `python tests/test_lineclean.py`.

## Naujas įrankis

Sukurkite katalogą `app/tools/<id>/__init__.py`. Forma, failų įkėlimas, žurnalas, atsisiuntimai ir istorija atsiras automatiškai.

```python
from ...core.tool import Tool, Param, FileInput

class Pavyzdys(Tool):
    id = "pavyzdys"
    title = "Pavyzdinis įrankis"
    group = "Paviršiai"                       # meniu grupė
    description = "Ką daro, vienu sakiniu."
    inputs = [FileInput("files", "Taškai (CSV)", accept=".csv")]
    params = [
        Param("step", "Žingsnis", "float", 5.0, "m", "Parametrai", min=0.1),
        Param("lines", "Kurti linijas", "bool", True, group="Išvestis"),
    ]

    def run(self, ctx):
        src = ctx.inputs["files"][0]          # Path
        ctx.log(f"Skaitau {src.name}")
        out = ctx.outdir / "rezultatas.dxf"
        ...                                   # skaičiavimas
        ctx.add_output(out, "DXF")
        # ctx.preview = {"plan": [...], "charts": [...], "table": [...]}   # neprivaloma
        # raise ValueError("...") -> vartotojui rodomas tvarkingas klaidos pranešimas

TOOL = Pavyzdys()
```

Peržiūros (`ctx.preview`) formatas pateiktas `tools/linefit/engine.py` → `preview()`: plano linijos ir taškai, grafikai bei suvestinės lentelė.

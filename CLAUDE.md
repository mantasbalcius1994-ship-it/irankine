# Įrankinė – projekto atmintinė

Inžinerinių (kelių / geodezijos) įrankių platforma naršyklėje: FastAPI + HTML/JS be priklausomybių.
Savininkas: Mantas (kelių inžinerija, TBC). UI ir ataskaitos – lietuviškai.

## Struktūra
- `app/core/` – Tool/Param aprašas, užduotys (atskiras procesas, būsena diske), įrankių registras.
- `app/geo/` – bendras branduolys (DXF/CSV/LandXML/LAS IO, TIN, trasa su klotoidėmis, Excel). Nauji įrankiai
  naudoja TIK jį, nekopijuoja IO kodo.
- `app/tools/<id>/__init__.py` – įrankis (`TOOL = ...`), automatiškai atsiranda meniu.
- `app/static/` – naršyklės dalis (forma generuojama iš Param schemos).

## Susitarimai
- Koordinatės: X = E, Y = N (LKS-94). LandXML rašomas „N E“. Atstumas nuo ašies + dešinėje (kaip TBC).
- Vartotojui rodomos klaidos – `ValueError` lietuviškai, su patarimu, ką daryti.
- Serveris mažas (2 vCPU, 3.7 GB): ribos `app/geo/limits.py`; dideli duomenys – dalimis / tinkleliu.
- Kiekvienas įrankis tikrinamas su žinomais atsakymais (tests/) ir su vartotojo tikrais failais.

## Diegimas
`main` šaka = produkcija. Serveris (Hetzner 77.42.12.179) kas 2 min. pats pasiima `main` (deploy/update.sh).
Prieš push: `python tests/test_geo.py && python tests/test_smoke.py && python tests/test_jobs.py && python tests/test_lineclean.py`.
Vartotojo duomenų failų (DXF, LAZ, IFC) į repozitoriją NEDĖTI - ji vieša.

## Planas (eilė)
0. Branduolys + diegimas ✔  1. linefit „tik planas“ (laukiama kelias_1/3_ali DXF)  2. Linijų valymas ir retinimas ✔  3. Paviršių palyginimas, tūriai
4. Išpildomųjų taškų kontrolė  5. Ofsetų šablonai, šlaitai  6. IFC → DXF  7. PDF vektoriai / georeferencavimas

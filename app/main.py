"""Įrankinė — inžinerinių įrankių platforma (FastAPI).

Konfigūracija per aplinkos kintamuosius (žr. deploy/toolbox.env.example):
  TOOLBOX_TITLE, TOOLBOX_DATA, TOOLBOX_WORKERS, TOOLBOX_JOB_TTL_H, TOOLBOX_MAX_UPLOAD_MB,
  TOOLBOX_USER / TOOLBOX_PASS (neprivaloma basic auth), TOOLBOX_LINKS (JSON failas su nuorodomis)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .core.jobs import JobManager
from .core.registry import discover

BASE = Path(__file__).resolve().parent
TITLE = os.environ.get("TOOLBOX_TITLE", "Įrankinė")
DATA = Path(os.environ.get("TOOLBOX_DATA", BASE.parent / "data"))
WORKERS = int(os.environ.get("TOOLBOX_WORKERS", "1"))
TTL_H = float(os.environ.get("TOOLBOX_JOB_TTL_H", "72"))
MAX_UPLOAD = int(float(os.environ.get("TOOLBOX_MAX_UPLOAD_MB", "500")) * 1024 * 1024)
AUTH_USER = os.environ.get("TOOLBOX_USER")
AUTH_PASS = os.environ.get("TOOLBOX_PASS")
LINKS_FILE = Path(os.environ.get("TOOLBOX_LINKS", DATA / "links.json"))

TOOLS = discover()
jobs: JobManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global jobs
    jobs = JobManager(DATA / "jobs", WORKERS, TTL_H)

    async def janitor():
        while True:
            jobs.cleanup()
            await asyncio.sleep(3600)

    task = asyncio.create_task(janitor())
    yield
    task.cancel()
    jobs.pool.shutdown(cancel_futures=True)


app = FastAPI(title=TITLE, lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if AUTH_USER and AUTH_PASS:
        ok = False
        h = request.headers.get("authorization", "")
        if h.startswith("Basic "):
            try:
                u, _, p = base64.b64decode(h[6:]).decode().partition(":")
                ok = secrets.compare_digest(u, AUTH_USER) and secrets.compare_digest(p, AUTH_PASS)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Irankine", charset="UTF-8"'})
    return await call_next(request)


# ---------------------------------------------------------------- API
@app.get("/api/info")
def info():
    links = []
    if LINKS_FILE.exists():
        try:
            links = json.loads(LINKS_FILE.read_text(encoding="utf-8"))
        except Exception:
            links = []
    return {"title": TITLE, "links": links,
            "tools": [t.schema() for t in sorted(TOOLS.values(), key=lambda t: (t.group, t.title))]}


@app.post("/api/tools/{tool_id}/run")
async def run_tool(tool_id: str, request: Request):
    tool = TOOLS.get(tool_id)
    if not tool:
        raise HTTPException(404, "Nėra tokio įrankio")
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD:
        raise HTTPException(413, f"Per dideli failai (max {MAX_UPLOAD // 1024 // 1024} MB)")
    form = await request.form()
    try:
        params = tool.parse_params(json.loads(form.get("params") or "{}"))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))
    files: dict[str, list[tuple[str, bytes]]] = {}
    for spec in tool.inputs:
        items = [f for f in form.getlist(spec.key) if hasattr(f, "filename") and f.filename]
        if spec.required and not items:
            raise HTTPException(422, f"Neįkelta: {spec.label}")
        if not spec.multiple and len(items) > 1:
            items = items[:1]
        if spec.accept:
            allowed = {a.strip().lower() for a in spec.accept.split(",")}
            for f in items:
                if Path(f.filename).suffix.lower() not in allowed:
                    raise HTTPException(422, f"{f.filename}: leidžiami formatai {spec.accept}")
        files[spec.key] = [(f.filename, f.file) for f in items]     # failas jau diske (spooled)
    jid = jobs.create(tool, params, files)
    return {"job_id": jid}


@app.get("/api/jobs")
def list_jobs(limit: int = 30, tool: str | None = None):
    return jobs.list(limit, tool)


@app.get("/api/jobs/{jid}")
def get_job(jid: str, log_from: int = 0):
    m = jobs.get(jid, log_from)
    if not m:
        raise HTTPException(404, "Užduotis nerasta")
    return m


@app.get("/api/jobs/{jid}/preview")
def get_preview(jid: str):
    p = jobs.preview(jid)
    if p is None:
        raise HTTPException(404, "Peržiūros nėra")
    return JSONResponse(p)


@app.get("/api/jobs/{jid}/files/{name}")
def get_file(jid: str, name: str, inline: bool = False):
    d = jobs.dir(jid)
    f = d / "out" / Path(name).name if d else None
    if not f or not f.exists():
        raise HTTPException(404, "Failas nerastas")
    return FileResponse(f, filename=None if inline else f.name,
                        content_disposition_type="inline" if inline else "attachment")


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    if not jobs.delete(jid):
        raise HTTPException(404, "Užduotis nerasta")
    return {"ok": True}


# ---------------------------------------------------------------- frontend
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")

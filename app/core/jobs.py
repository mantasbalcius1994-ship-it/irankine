"""Užduotys. Būsena laikoma diske (jobs/<id>/job.json, log.txt), vykdoma atskirame procese,
todėl ilgas skaičiavimas neužblokuoja serverio, o užduotys išlieka po perkrovimo."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path


def _write_json(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execute(job_dir: str, tool_id: str):
    """Vykdoma vaikiniame procese."""
    from .registry import discover
    from .tool import JobContext

    d = Path(job_dir)
    meta = _read_json(d / "job.json")
    logf = open(d / "log.txt", "a", encoding="utf-8", buffering=1)

    def log(msg):
        logf.write(f"{msg}\n")

    def progress(v):
        meta["progress"] = max(0.0, min(1.0, float(v)))
        _write_json(d / "job.json", meta)

    meta.update(status="running", started=time.time(), pid=os.getpid())
    _write_json(d / "job.json", meta)
    try:
        tool = discover()[tool_id]
        inputs = {k: [d / "in" / n for n in names] for k, names in meta["inputs"].items()}
        ctx = JobContext(meta["id"], d, inputs, meta["params"], log, progress)
        tool.run(ctx)
        if ctx.preview is not None:
            _write_json(d / "preview.json", ctx.preview)
        meta.update(status="done", outputs=[asdict(o) for o in ctx.outputs], progress=1.0)
    except ValueError as e:                 # vartotojui suprantama klaida
        log(f"KLAIDA: {e}")
        meta.update(status="error", error=str(e))
    except Exception as e:
        log(traceback.format_exc())
        meta.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        meta["finished"] = time.time()
        _write_json(d / "job.json", meta)
        logf.close()


class JobManager:
    def __init__(self, root: Path, workers: int = 2, ttl_hours: float = 72, timeout_min: float = 30):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_hours * 3600
        self.timeout = timeout_min * 60
        threading.Thread(target=self._watchdog, daemon=True).start()
        self.workers = workers
        self._lock = threading.Lock()
        self._broken = False
        self.pool = self._new_pool()
        self._recover()

    def _watchdog(self, period=15):
        """Užduotis, vykdoma ilgiau nei limitas, nutraukiama (kad neblokuotų eilės ir atnaujinimų)."""
        import signal
        while True:
            time.sleep(period)
            try:
                for f in self.root.glob("*/job.json"):
                    m = _read_json(f)
                    if m.get("status") != "running" or not m.get("started"):
                        continue
                    if time.time() - m["started"] < self.timeout:
                        continue
                    m.update(status="error", finished=time.time(),
                             error=f"Viršytas laiko limitas ({self.timeout / 60:.0f} min.). Sumažinkite duomenų kiekį "
                                   "(mažiau linijų / retinimas) arba padidinkite tolerancijas.")
                    _write_json(f, m)
                    with open(f.parent / "log.txt", "a", encoding="utf-8") as lf:
                        lf.write("KLAIDA: viršytas laiko limitas, skaičiavimas nutrauktas\n")
                    if m.get("pid"):
                        try:
                            os.kill(int(m["pid"]), signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
            except Exception:
                pass

    def _new_pool(self):
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))

    def _submit(self, d: Path, tool_id: str):
        with self._lock:
            if self._broken:                      # darbinis procesas buvo nužudytas (pvz. atminties riba)
                try:
                    self.pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self.pool, self._broken = self._new_pool(), False
            fut = self.pool.submit(_execute, str(d), tool_id)
        fut.add_done_callback(lambda f: self._on_done(d, tool_id, f))

    def _on_done(self, d: Path, tool_id: str, fut):
        exc = fut.exception() if not fut.cancelled() else None
        if exc is None and not fut.cancelled():
            return
        self._broken = True
        f = d / "job.json"
        if not f.exists():
            return
        m = _read_json(f)
        if m.get("status") == "queued" and m.get("retries", 0) < 2:
            # nespėjo prasidėti - automatiškai paleidžiama naujame procese
            m["retries"] = m.get("retries", 0) + 1
            _write_json(f, m)
            threading.Timer(0.5, self._submit, args=(d, tool_id)).start()
            return
        if m.get("status") in ("queued", "running"):
            m.update(status="error", finished=time.time(),
                     error="Skaičiavimas nutrauktas: viršyta serverio atminties riba. "
                           "Sumažinkite duomenis (retinimas, apkirpimas, mažesnis plotas) ir bandykite dar kartą.")
            _write_json(f, m)
            with open(d / "log.txt", "a", encoding="utf-8") as lf:
                lf.write(f"KLAIDA: procesas nutrauktas ({type(exc).__name__})\n")

    def _recover(self):
        """Po perkrovimo nebaigtos užduotys pažymimos kaip nutrauktos."""
        for d in self.root.iterdir():
            f = d / "job.json"
            if f.exists():
                m = _read_json(f)
                if m.get("status") in ("queued", "running"):
                    m.update(status="error", error="Serveris buvo perkrautas")
                    _write_json(f, m)

    def create(self, tool, params: dict, files: dict[str, list[tuple]]) -> str:
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = self.root / jid
        (d / "in").mkdir(parents=True)
        names: dict[str, list[str]] = {}
        for key, items in files.items():
            for fname, data in items:
                safe = Path(fname).name.replace("/", "_") or "failas"
                base, n = safe, 1
                while (d / "in" / safe).exists():
                    stem, ext = os.path.splitext(base)
                    safe, n = f"{stem}_{n}{ext}", n + 1
                if isinstance(data, (bytes, bytearray)):
                    (d / "in" / safe).write_bytes(data)
                else:                                  # failo objektas - kopijuojama dalimis
                    data.seek(0)
                    with open(d / "in" / safe, "wb") as out:
                        shutil.copyfileobj(data, out, 4 * 1024 * 1024)
                names.setdefault(key, []).append(safe)
        meta = {"id": jid, "tool": tool.id, "title": tool.title, "status": "queued",
                "created": time.time(), "params": params, "inputs": names,
                "outputs": [], "progress": 0.0, "error": None}
        _write_json(d / "job.json", meta)
        (d / "log.txt").write_text("", encoding="utf-8")
        self._submit(d, tool.id)
        return jid

    def dir(self, jid: str) -> Path | None:
        d = self.root / Path(jid).name
        return d if (d / "job.json").exists() else None

    def get(self, jid: str, log_from: int = 0) -> dict | None:
        d = self.dir(jid)
        if not d:
            return None
        m = _read_json(d / "job.json")
        lines = (d / "log.txt").read_text(encoding="utf-8").splitlines()
        m["log"] = lines[log_from:]
        m["log_next"] = len(lines)
        m["has_preview"] = (d / "preview.json").exists()
        return m

    def preview(self, jid: str) -> dict | None:
        d = self.dir(jid)
        if d and (d / "preview.json").exists():
            return _read_json(d / "preview.json")
        return None

    def list(self, limit: int = 30, tool: str | None = None) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            f = d / "job.json"
            if not f.exists():
                continue
            m = _read_json(f)
            if tool and m["tool"] != tool:
                continue
            out.append({k: m.get(k) for k in ("id", "tool", "title", "status", "created", "finished", "inputs")})
            if len(out) >= limit:
                break
        return out

    def delete(self, jid: str) -> bool:
        d = self.dir(jid)
        if not d:
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    def cleanup(self):
        now = time.time()
        for d in self.root.iterdir():
            f = d / "job.json"
            try:
                m = _read_json(f) if f.exists() else {"created": d.stat().st_mtime, "status": "done"}
            except Exception:
                continue
            if m.get("status") not in ("queued", "running") and now - m.get("created", now) > self.ttl:
                shutil.rmtree(d, ignore_errors=True)

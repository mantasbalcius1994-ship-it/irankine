"""Užduočių atsparumas: nužudytas darbinis procesas (pvz. viršyta atminties riba) -> klaida,
eilėje buvusi užduotis paleidžiama iš naujo, serveris toliau veikia.  python tests/test_jobs.py"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.jobs import JobManager  # noqa: E402
from app.core.registry import discover  # noqa: E402


def wait(jm, *ids, t=180):
    st = []
    for _ in range(t):
        st = [jm.get(j)["status"] for j in ids]
        if all(s in ("done", "error") for s in st):
            break
        time.sleep(1)
    return st


def test_worker_killed():
    jm = JobManager(Path(tempfile.mkdtemp()), workers=1)
    tool = discover()["linefit"]
    dxf = (Path(__file__).parent / "data" / "test.dxf").read_bytes()
    p = tool.parse_params({"mode": "lankai", "h_tol_line": 15, "h_tol_arc": 15, "v_tol": 8})
    j1 = jm.create(tool, p, {"files": [("a.dxf", dxf)]})
    for _ in range(200):
        if jm.get(j1)["status"] == "running" and jm.pool._processes:
            break
        time.sleep(0.05)
    j2 = jm.create(tool, p, {"files": [("b.dxf", dxf)]})
    for pid in list(jm.pool._processes):
        os.kill(pid, 9)
    s1, s2 = wait(jm, j1, j2)
    assert s1 == "error" and "atminties" in jm.get(j1)["error"], jm.get(j1)
    assert s2 == "done", jm.get(j2)
    j3 = jm.create(tool, p, {"files": [("c.dxf", dxf)]})
    assert wait(jm, j3) == ["done"]
    jm.pool.shutdown()


if __name__ == "__main__":
    test_worker_killed()
    print("OK")

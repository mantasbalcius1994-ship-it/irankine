"""python -m pytest tests -q   (arba: python tests/test_smoke.py)"""
import json, os, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TOOLBOX_DATA", tempfile.mkdtemp())
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

DATA = Path(__file__).parent / "data"


def test_linefit_end_to_end():
    with TestClient(app) as c:
        assert any(t["id"] == "linefit" for t in c.get("/api/info").json()["tools"])
        r = c.post("/api/tools/linefit/run",
                   files=[("files", ("test.dxf", (DATA / "test.dxf").read_bytes()))],
                   data={"params": json.dumps({"h_tol_line": 15, "h_tol_arc": 15, "v_tol": 8})})
        assert r.status_code == 200, r.text
        jid = r.json()["job_id"]
        for _ in range(240):
            m = c.get(f"/api/jobs/{jid}").json()
            if m["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        assert m["status"] == "done", m.get("error")
        row = c.get(f"/api/jobs/{jid}/preview").json()["table"][0]
        assert row["Lankų"] == 3 and row["VPI"] == 2 and row["Už tol."] == 0
        for o in m["outputs"]:
            assert c.get(f"/api/jobs/{jid}/files/{o['name']}").status_code == 200


def test_rejects_wrong_format():
    with TestClient(app) as c:
        r = c.post("/api/tools/linefit/run", files=[("files", ("a.pdf", b"x"))], data={"params": "{}"})
        assert r.status_code == 422


if __name__ == "__main__":
    test_linefit_end_to_end(); test_rejects_wrong_format(); print("OK")

"""
FastAPI backend. Serves placement records and generated assets to the viewer,
and hosts the review queue (spec 9.3).

Async from commit one (spec 14). Generation takes 10-60 s, so the run endpoint
enqueues and returns an id immediately; the client polls. Retrofitting this at
hour 30 is miserable.

    uv run uvicorn server.app:app --reload
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"

app = FastAPI(title="Procedura placement API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}


class RunRequest(BaseModel):
    address: str
    prompt: str = "weathered concrete, ivy overgrowth, scorched upper floors"
    photos: list[str] = []
    dry_run: bool = True
    confidence: str = "threshold_table"


@app.get("/api/runs")
def list_runs() -> list[dict]:
    """Newest first. The viewer's dropdown."""
    out = []
    if not RUNS.exists():
        return out

    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        record = d / "record.json"
        if not record.exists():
            continue
        try:
            rec = json.loads(record.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        iou = (rec.get("fit") or {}).get("iou")
        out.append({
            "asset_id": rec["asset_id"],
            "label": (f"{rec.get('address_raw', '?')} · {rec.get('decision', '?')}"
                      + (f" · IoU {iou:.2f}" if isinstance(iou, (int, float)) else "")),
            "decision": rec.get("decision"),
        })
    return out


@app.get("/api/runs/{asset_id}")
def get_run(asset_id: str) -> dict:
    record = RUNS / asset_id / "record.json"
    if not record.exists():
        raise HTTPException(404, f"no record for {asset_id}")
    return json.loads(record.read_text(encoding="utf-8"))


@app.get("/api/review")
def review_queue() -> list[dict]:
    """Spec 9.3: everything not auto-accepted, with its overlay and candidates."""
    queue = []
    for row in list_runs():
        if row["decision"] == "auto_accept":
            continue
        rec = get_run(row["asset_id"])
        queue.append({
            "asset_id": rec["asset_id"],
            "address": rec.get("address_raw"),
            "decision": rec.get("decision"),
            "reasons": rec.get("review_reasons", []),
            "overlay": f"/assets-data/{rec['asset_id']}/overlay.png",
            "candidates": (rec.get("fit") or {}).get("candidate_ious", {}),
        })
    return queue


@app.get("/assets-data/{asset_id}/{filename}")
def asset(asset_id: str, filename: str):
    """Serve overlay.png, mesh.glb, edited_*.png from the run directory."""
    if "/" in filename or ".." in filename or "\\" in filename:
        raise HTTPException(400, "bad filename")
    path = RUNS / asset_id / filename
    if not path.exists():
        raise HTTPException(404, filename)
    return FileResponse(path)


@app.post("/api/run")
def start_run(req: RunRequest) -> dict:
    """Enqueue and return immediately. Generation blocks for 10-60 s."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "asset_id": None, "error": None}

    def work():
        _jobs[job_id]["status"] = "running"
        try:
            from pipeline import run

            rec = run(
                req.address,
                photos=[Path(p) for p in req.photos],
                prompt=req.prompt,
                dry_run=req.dry_run,
                confidence_method=req.confidence,
            )
            _jobs[job_id].update(status="done", asset_id=rec.asset_id)
        except Exception as exc:  # noqa: BLE001
            _jobs[job_id].update(status="failed", error=f"{type(exc).__name__}: {exc}")

    _pool.submit(work)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, job_id)
    return _jobs[job_id]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "runs": len(list_runs())}

"""Spec 9.3: a reviewer overrules the orientation, and nothing is re-generated."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from server import app as server


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "RUNS", tmp_path)
    return TestClient(server.app)


def _record(tmp_path, asset_id="a1", *, inputs=True):
    d = tmp_path / asset_id
    d.mkdir()
    rec = {
        "asset_id": asset_id,
        "address_raw": "Somewhere, Blacksburg, VA",
        "decision": "review",
        "orientation": {"chosen_k": 2, "auto_k": 2, "up_axis_idx": 4,
                        "auto_disambiguated_by": "road_normal",
                        "forced_by_reviewer": False,
                        "candidates": [{"k": 2, "iou": 0.61}, {"k": 0, "iou": 0.61}]},
    }
    if inputs:
        rec["inputs"] = {"photos": [], "perception": "stub", "dry_run": True,
                         "prompt": "", "seed": 42, "allow_anisotropy": False,
                         "mask_for_generation": True}
    (d / "record.json").write_text(json.dumps(rec), encoding="utf-8")
    return rec


def test_choose_enqueues_a_resolve(client, tmp_path, monkeypatch):
    _record(tmp_path)
    seen = {}

    def fake_run(address, **kw):
        seen.update(address=address, **kw)
        return None
    monkeypatch.setitem(__import__("sys").modules, "pipeline",
                        type("m", (), {"run": staticmethod(fake_run)}))

    r = client.post("/api/runs/a1/choose", json={"k": 3})
    assert r.status_code == 200
    job = r.json()["job_id"]
    for _ in range(100):
        status = client.get(f"/api/jobs/{job}").json()
        if status["status"] in ("done", "failed"):
            break
    assert status["status"] == "done", status
    # The reviewer changes the AZIMUTH and nothing else: same asset, same
    # inputs, so the cached edit and mesh are reused and nothing is billed.
    assert seen["force_candidate"] == 3
    assert seen["asset_id"] == "a1"
    assert seen["address"] == "Somewhere, Blacksburg, VA"
    assert seen["dry_run"] is True


def test_k_out_of_range_is_refused(client, tmp_path):
    _record(tmp_path)
    assert client.post("/api/runs/a1/choose", json={"k": 7}).status_code == 400


def test_record_without_inputs_is_refused(client, tmp_path):
    _record(tmp_path, inputs=False)
    r = client.post("/api/runs/a1/choose", json={"k": 1})
    assert r.status_code == 409
    assert "provenance" in r.json()["detail"]


def test_review_queue_lists_candidates(client, tmp_path):
    _record(tmp_path)
    row = client.get("/api/review").json()[0]
    assert row["candidates"] == [{"k": 2, "iou": 0.61}, {"k": 0, "iou": 0.61}]
    assert row["overlay"].endswith("/overlay.png")

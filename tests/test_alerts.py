import os
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app, security_incidents, incidents_lock
from sanitize_dataset import sanitize_bounding_boxes
from yolo_inference import YOLOInference

client = TestClient(app)


def test_sanitize_bounding_boxes_deduplication():
    # Duplicate lines and tiny noise boxes
    raw_lines = [
        "0 0.500000 0.500000 0.100000 0.200000",
        "0 0.500000 0.500000 0.100000 0.200000",  # Exact duplicate
        "0 0.500100 0.500200 0.100100 0.200100",  # Sub-millimeter near duplicate
        "0 0.100000 0.100000 0.005000 0.005000",  # Degenerate micro-box (< min_width/height)
        "1 0.300000 0.300000 0.100000 0.200000",  # Non-person class
    ]

    cleaned = sanitize_bounding_boxes(raw_lines)
    assert len(cleaned) == 1
    assert cleaned[0].startswith("0 0.500000 0.500000 0.100000 0.200000")


def test_settings_endpoints():
    # Test GET settings
    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()
    assert "threshold" in data
    assert "sustained_duration" in data
    assert "cooldown_duration" in data
    assert "inference_imgsz" in data

    # Test update threshold
    thresh_res = client.post("/api/settings/threshold", json={"threshold": 25})
    assert thresh_res.status_code == 200
    assert thresh_res.json()["threshold"] == 25

    # Verify updated in settings
    get_res = client.get("/api/settings")
    assert get_res.json()["threshold"] == 25

    # Test update webhook
    hook_res = client.post("/api/settings/webhook", json={"webhook_url": "https://example.com/webhook"})
    assert hook_res.status_code == 200
    assert hook_res.json()["webhook_url"] == "https://example.com/webhook"

    # Test update imgsz
    imgsz_res = client.post("/api/settings/imgsz", json={"imgsz": 1024})
    assert imgsz_res.status_code == 200
    assert imgsz_res.json()["inference_imgsz"] == 1024

    # Test invalid imgsz
    bad_imgsz = client.post("/api/settings/imgsz", json={"imgsz": 500})
    assert bad_imgsz.status_code == 400


def test_sustained_alert_logic():
    engine = YOLOInference()
    engine.set_alert_threshold(10)
    engine.sustained_duration = 0.2  # Fast for test
    engine.cooldown_duration = 0.5

    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

    # Below threshold: no alert
    assert engine.check_and_trigger_alert(dummy_frame, 5) is False
    assert engine.breach_start_time is None

    # First breach instant: starts breach timer, does not fire yet
    assert engine.check_and_trigger_alert(dummy_frame, 12) is False
    assert engine.breach_start_time is not None

    # Wait for sustained duration
    time.sleep(0.25)
    # Sustained breach: triggers alert
    triggered = engine.check_and_trigger_alert(dummy_frame, 12)
    assert triggered is True
    assert engine.active_alert_triggered is True

    # Immediate call during cooldown: should not re-trigger snapshot
    assert engine.check_and_trigger_alert(dummy_frame, 12) is False


def test_incidents_management():
    # Inject test incident
    test_incident = {
        "id": "inc_test_999",
        "timestamp": "2026-09-22 18:00:00",
        "people_count": 35,
        "threshold": 20,
        "snapshot_filename": "non_existent.jpg",
        "snapshot_url": "/api/snapshots/non_existent.jpg",
        "acknowledged": False,
    }
    with incidents_lock:
        security_incidents.insert(0, test_incident)

    # List incidents
    res = client.get("/api/incidents")
    assert res.status_code == 200
    incidents = res.json()["incidents"]
    assert any(i["id"] == "inc_test_999" for i in incidents)

    # Acknowledge incident
    ack_res = client.post("/api/incidents/inc_test_999/acknowledge")
    assert ack_res.status_code == 200
    assert ack_res.json()["acknowledged"] is True

    # Check non-existent incident
    bad_ack = client.post("/api/incidents/non_existent_id/acknowledge")
    assert bad_ack.status_code == 404

    # Snapshot 404
    snap_res = client.get("/api/snapshots/does_not_exist.jpg")
    assert snap_res.status_code == 404

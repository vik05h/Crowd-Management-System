"""Automated test suite for Phase 1.5 features.

Validates:
1. Real-time Face Privacy Anonymization Engine (DPDPA 2023 / GDPR).
2. Multi-Camera Feed Grid API and page routes (/cameras, /api/cameras).
3. CSV Data Export Engine (/api/export/csv for incidents, trends, zones).
4. Staged Safety Incident Drill Simulation (/api/incidents/simulate).
5. Incident log persistence and data enrichment.
"""

import csv
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from cms.privacy.face_blur import PrivacyManager
from yolo_inference import YOLOInference


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_privacy_manager_unit():
    """Verify PrivacyManager logic, toggling, and Gaussian blur masking."""
    pm = PrivacyManager(enabled=False, blur_kernel_size=25)
    assert not pm.enabled
    assert pm.get_status()["compliance"] == "DPDPA 2023 / GDPR Anonymization"

    # Frame without privacy enabled should return unmodified
    dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    out = pm.apply_face_blur(dummy_frame, boxes=[(10, 10, 50, 80)])
    assert np.array_equal(dummy_frame, out)

    # Enable privacy
    pm.set_enabled(True)
    assert pm.enabled

    # Box-based head blur
    test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    test_frame[10:90, 10:90] = 255
    blurred = pm.apply_face_blur(test_frame, boxes=[(10, 10, 90, 90)])
    assert not np.array_equal(test_frame, blurred)

    # Keypoint-based head blur
    keypoints = [[
        (50.0, 50.0, 0.9),  # nose
        (45.0, 45.0, 0.9),  # left eye
        (55.0, 45.0, 0.9),  # right eye
        (40.0, 45.0, 0.9),  # left ear
        (60.0, 45.0, 0.9),  # right ear
    ] + [(0.0, 0.0, 0.0)] * 12]
    kpt_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    kpt_frame[30:70, 30:70] = 200
    kpt_blurred = pm.apply_face_blur(kpt_frame, keypoints=keypoints)
    assert not np.array_equal(kpt_frame, kpt_blurred)


def test_privacy_api_endpoints(client):
    """Test GET and POST /api/settings/privacy endpoints."""
    # GET status
    res = client.get("/api/settings/privacy")
    assert res.status_code == 200
    data = res.json()
    assert "enabled" in data
    assert "compliance" in data

    # POST toggle enabled
    res = client.post("/api/settings/privacy", json={"enabled": True})
    assert res.status_code == 200
    assert res.json()["privacy_enabled"] is True

    # POST toggle disabled
    res = client.post("/api/settings/privacy", json={"enabled": False})
    assert res.status_code == 200
    assert res.json()["privacy_enabled"] is False


def test_cameras_grid_endpoints(client):
    """Test Multi-Camera page route and API."""
    # HTML Page
    page_res = client.get("/cameras")
    assert page_res.status_code == 200
    assert "CMS MATRIX" in page_res.text
    assert "CCTV SURVEILLANCE GRID" in page_res.text

    # JSON API
    api_res = client.get("/api/cameras")
    assert api_res.status_code == 200
    data = api_res.json()
    assert "cameras" in data
    assert len(data["cameras"]) == 4
    assert data["total_cameras"] == 4
    assert data["cameras"][0]["id"] == "cam-01"
    assert data["cameras"][0]["is_primary"] is True


def test_csv_export_endpoints(client):
    """Test /api/export/csv for incidents, trends, and zones."""
    # Export incidents
    res_inc = client.get("/api/export/csv?type=incidents")
    assert res_inc.status_code == 200
    assert "text/csv" in res_inc.headers.get("content-type", "")
    assert "attachment; filename=\"cms_incidents_" in res_inc.headers.get("content-disposition", "")
    reader = csv.reader(io.StringIO(res_inc.text))
    header = next(reader)
    assert "Incident ID" in header
    assert "Event Type" in header
    assert "Sector / Location" in header

    # Export trends
    res_trends = client.get("/api/export/csv?type=trends&hours=24")
    assert res_trends.status_code == 200
    assert "text/csv" in res_trends.headers.get("content-type", "")
    assert "attachment; filename=\"cms_crowd_trends_" in res_trends.headers.get("content-disposition", "")
    t_reader = csv.reader(io.StringIO(res_trends.text))
    t_header = next(t_reader)
    assert "Headcount" in t_header

    # Export zones
    res_zones = client.get("/api/export/csv?type=zones")
    assert res_zones.status_code == 200
    assert "text/csv" in res_zones.headers.get("content-type", "")
    assert "attachment; filename=\"cms_sectors_" in res_zones.headers.get("content-disposition", "")
    z_reader = csv.reader(io.StringIO(res_zones.text))
    z_header = next(z_reader)
    assert "Sector Name" in z_header


def test_staged_incident_simulation(client):
    """Test /api/incidents/simulate endpoint and audit log enrichment."""
    payload = {
        "event_type": "fall",
        "zone_name": "Stage Front West",
        "confidence": 0.98,
        "details": {"posture_angle": 72.1, "drill_code": "ALPHA_01"}
    }
    res = client.post("/api/incidents/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "simulated"
    incident = data["incident"]
    assert incident["event_type"] == "fall"
    assert incident["zone_name"] == "Stage Front West"
    assert incident["confidence"] == 0.98
    assert incident["snapshot_url"].startswith("/api/snapshots/")

    # Check that GET /api/incidents retrieves the simulated event
    list_res = client.get("/api/incidents")
    assert list_res.status_code == 200
    incidents = list_res.json().get("incidents", [])
    found = any(inc["id"] == incident["id"] for inc in incidents)
    assert found, "Simulated incident must be present in incidents list"

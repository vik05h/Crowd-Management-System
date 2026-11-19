"""Automated unit and integration test suite for CMS Phase 1 features.

Tests DatabaseManager, JWT authentication, ZoneTracker, PoseEngine,
and Analytics SOC API endpoints.
"""

import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from cms.analytics.zone_tracker import SecurityZone, ZoneTracker
from cms.auth.jwt_auth import create_access_token, decode_access_token
from cms.core.pose_engine import PoseEngine
from cms.data.database import db_manager, hash_password, verify_password

client = TestClient(app)


def test_password_hashing_and_verification():
    """Verify PBKDF2 password hashing and verification logic."""
    password = "SecretPassword2026!"
    stored_hash = hash_password(password)

    assert verify_password(password, stored_hash) is True
    assert verify_password("WrongPassword", stored_hash) is False
    assert verify_password("", stored_hash) is False


def test_jwt_token_generation_and_decoding():
    """Verify JWT access token creation and payload validation."""
    username = "test_operator"
    role = "operator"
    token = create_access_token(username=username, role=role, expires_in=3600)

    payload = decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == username
    assert payload["role"] == role
    assert payload["exp"] > time.time()

    # Invalid token check
    assert decode_access_token("invalid.token.signature") is None


def test_zone_tracker_containment_and_breach():
    """Verify point-in-polygon containment, dwell computation, and capacity breach."""
    tracker = ZoneTracker()
    test_zone_def = {
        "id": "test-lobby",
        "name": "Test Lobby Sector",
        "color": "#FFE500",
        "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "max_occupancy": 2,
    }
    tracker.set_zones([test_zone_def])
    summary = tracker.get_summary()
    assert len(summary) == 1
    assert summary[0]["name"] == "Test Lobby Sector"

    # Track 1 inside polygon at foot (50, 80)
    tracks_inside = [(1, 40, 20, 60, 80)]
    breaches = tracker.update(tracks_inside, frame_w=200, frame_h=200)
    assert len(breaches) == 0  # 1 person < capacity 2

    # Add Track 2 and Track 3 inside to trigger capacity breach
    tracks_over = [
        (1, 40, 20, 60, 80),
        (2, 20, 10, 40, 70),
        (3, 70, 20, 90, 85),
    ]
    # Simulate sustained breach over time
    now = time.time()
    tracker.update(tracks_over, frame_w=200, frame_h=200, now=now)
    breaches = tracker.update(tracks_over, frame_w=200, frame_h=200, now=now + 2.5)

    assert len(breaches) == 1
    assert breaches[0]["is_breached"] is True
    assert breaches[0]["current_occupancy"] == 3

    # Rendering test
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    rendered = tracker.draw_zones(dummy_frame)
    assert rendered is not None
    assert rendered.shape == (200, 200, 3)


def test_pose_engine_execution():
    """Verify PoseEngine frame processing and skeleton rendering."""
    engine = PoseEngine()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    skeletons, events = engine.process_frame(dummy_frame)
    assert isinstance(skeletons, list)
    assert isinstance(events, list)

    rendered = engine.draw_skeletons(dummy_frame, skeletons)
    assert rendered is not None
    assert rendered.shape == (480, 640, 3)


def test_auth_login_endpoint():
    """Test /api/auth/login endpoint with valid and invalid credentials."""
    # Valid login
    res = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "admin"
    assert data["username"] == "admin"

    token = data["access_token"]

    # Verify protected /api/auth/me route
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "admin"

    # Invalid login
    bad_res = client.post("/api/auth/login", json={"username": "admin", "password": "wrong_password"})
    assert bad_res.status_code == 401


def test_analytics_and_zones_endpoints():
    """Test Analytics SOC routes and security zones management."""
    # GET /analytics HTML dashboard
    page_res = client.get("/analytics")
    assert page_res.status_code == 200
    assert "Spatial Safety Analytics" in page_res.text

    # GET /api/analytics/summary
    summary_res = client.get("/api/analytics/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert "events_24h" in summary
    assert "zone_count" in summary
    assert "live_people_count" in summary

    # GET /api/analytics/trends
    trends_res = client.get("/api/analytics/trends?hours=24")
    assert trends_res.status_code == 200
    assert isinstance(trends_res.json(), list)

    # GET /api/analytics/zones
    live_zones_res = client.get("/api/analytics/zones")
    assert live_zones_res.status_code == 200
    assert isinstance(live_zones_res.json(), list)

    # POST /api/zones: Create new sector
    new_zone_id = f"test-zone-{int(time.time())}"
    create_res = client.post(
        "/api/zones",
        json={
            "id": new_zone_id,
            "name": "Automated Test Sector",
            "color": "#A6FAFF",
            "polygon": [[50, 50], [200, 50], [200, 200], [50, 200]],
            "max_occupancy": 8,
            "camera_id": "primary",
        },
    )
    assert create_res.status_code == 200
    assert create_res.json()["status"] == "success"

    # Verify present in GET /api/zones
    list_res = client.get("/api/zones")
    assert list_res.status_code == 200
    zones = list_res.json()
    assert any(z["id"] == new_zone_id for z in zones)

    # DELETE /api/zones/{id}
    del_res = client.delete(f"/api/zones/{new_zone_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Toggle endpoints
    pose_toggle_res = client.post("/api/pose/toggle", json={"enabled": False})
    assert pose_toggle_res.status_code == 200
    assert pose_toggle_res.json()["pose_enabled"] is False

    zones_toggle_res = client.post("/api/zones/toggle", json={"enabled": True})
    assert zones_toggle_res.status_code == 200
    assert zones_toggle_res.json()["zones_enabled"] is True


def test_hardware_acceleration_and_calibration():
    """Verify hardware profile switching, pose model selection, and live polygon calibration."""
    # 1. Hardware status telemetry
    hw_res = client.get("/api/settings/hardware")
    assert hw_res.status_code == 200
    hw_data = hw_res.json()
    assert "device" in hw_data
    assert "profile" in hw_data
    assert "pose_model" in hw_data
    assert "det_latency_ms" in hw_data
    assert "fps" in hw_data

    # 2. Hardware profile switching: Eco, Balanced, Ultra
    prof_res = client.post("/api/settings/hardware_profile", json={"profile": "eco"})
    assert prof_res.status_code == 200
    assert prof_res.json()["profile"] == "eco"
    assert prof_res.json()["pose_stride"] == 2

    bal_res = client.post("/api/settings/hardware_profile", json={"profile": "balanced"})
    assert bal_res.status_code == 200
    assert bal_res.json()["profile"] == "balanced"
    assert bal_res.json()["pose_stride"] == 1

    # 3. Pose model switching
    model_res = client.post("/api/settings/pose_model", json={"model_name": "yolo26n-pose.pt"})
    assert model_res.status_code == 200
    assert model_res.json()["pose_model"] == "yolo26n-pose.pt"

    # 4. Pose stride configuration
    stride_res = client.post("/api/settings/pose_stride", json={"stride": 3})
    assert stride_res.status_code == 200
    assert stride_res.json()["pose_stride"] == 3

    # 5. FP16 toggle
    fp16_res = client.post("/api/settings/fp16", json={"enabled": True})
    assert fp16_res.status_code == 200
    assert "fp16_enabled" in fp16_res.json()

    # 6. Live polygon calibration endpoint
    cal_zone_id = "cal-test-zone"
    client.post(
        "/api/zones",
        json={
            "id": cal_zone_id,
            "name": "Calibration Test",
            "color": "#FFE500",
            "polygon": [[10, 10], [100, 10], [100, 100], [10, 100]],
            "max_occupancy": 5,
        },
    )
    # Update polygon via PUT
    updated_polygon = [[20, 20], [150, 20], [150, 150], [20, 150]]
    put_res = client.put(f"/api/zones/{cal_zone_id}/polygon", json={"polygon": updated_polygon})
    assert put_res.status_code == 200
    assert put_res.json()["status"] == "success"
    assert put_res.json()["zone"]["polygon"] == updated_polygon

    # Cleanup
    client.delete(f"/api/zones/{cal_zone_id}")

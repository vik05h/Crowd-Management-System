import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "")
    content = response.text
    assert "Crowd Management System" in content
    assert "YOLO26m" in content
    assert "Inter" in content
    assert "consoleImage" in content
    assert "dropzoneBox" in content


def test_live_camera_page():
    response = client.get("/live_camera")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "")


def test_camera_stats_endpoint():
    response = client.get("/camera_stats")
    assert response.status_code == 200
    data = response.json()
    assert "people_count" in data
    assert "density_level" in data
    assert "alert_status" in data
    assert "fps" in data


def test_toggle_heatmap_endpoint():
    response = client.post("/toggle_camera_heatmap")
    assert response.status_code == 200
    data = response.json()
    assert "heatmap_enabled" in data


def test_set_zoom_endpoint():
    response = client.get("/set_zoom?row=2&col=3")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

    reset_resp = client.get("/set_zoom?row=-1&col=-1")
    assert reset_resp.status_code == 200
    assert reset_resp.json() == {"status": "OK"}


def test_camera_lifecycle():
    stop_resp = client.post("/stop_camera")
    assert stop_resp.status_code == 200
    assert stop_resp.json() == {"status": "Camera stopped"}


def test_live_preview_page():
    response = client.get("/live_preview")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "")


def test_toggle_heatmap_api():
    response = client.post("/toggle_heatmap")
    assert response.status_code == 200
    data = response.json()
    assert "heatmap_enabled" in data
    assert data["status"] == "success"


def test_video_studio_page():
    response = client.get("/video_studio")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "")


def test_processing_status_endpoint():
    response = client.get("/api/processing_status")
    assert response.status_code == 200
    data = response.json()
    assert "active" in data
    assert "percent" in data
    assert "total_frames" in data
    assert "is_complete" in data


def test_processed_videos_endpoint():
    response = client.get("/api/processed_videos")
    assert response.status_code == 200
    data = response.json()
    assert "videos" in data
    assert isinstance(data["videos"], list)


def test_upload_endpoint():
    # Test uploading a tiny valid byte stream as MP4
    fake_video = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42"
    response = client.post(
        "/upload",
        files={"video": ("test_sample.mp4", fake_video, "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "redirect_url" in data
    assert data["redirect_url"] == "/video_studio"


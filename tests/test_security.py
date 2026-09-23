import os
import pytest
from fastapi.testclient import TestClient

from app import app, yolo_infer

client = TestClient(app)


def test_ssrf_protection_blocked_ips():
    """Verify that private, loopback, and cloud metadata webhook targets are rejected with 400."""
    blocked_urls = [
        "http://localhost:8080/webhook",
        "http://127.0.0.1/admin",
        "http://127.0.0.1:8000/api",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5:9000/webhook",
        "http://192.168.1.1/router",
        "http://172.16.0.1/hook",
        "ftp://example.com/file",
        "gopher://example.com/",
    ]
    for url in blocked_urls:
        resp = client.post("/api/settings/webhook", json={"webhook_url": url})
        assert resp.status_code == 400, f"Expected 400 for dangerous URL: {url}"


def test_ssrf_protection_valid_url():
    """Verify that valid external HTTPS webhooks are accepted."""
    resp = client.post("/api/settings/webhook", json={"webhook_url": "https://example.com/security-webhook"})
    assert resp.status_code == 200
    assert resp.json()["webhook_url"] == "https://example.com/security-webhook"

    # Reset webhook to None
    reset_resp = client.post("/api/settings/webhook", json={"webhook_url": ""})
    assert reset_resp.status_code == 200
    assert reset_resp.json()["webhook_url"] is None


def test_upload_extension_whitelist():
    """Verify that executable, script, or non-video uploads are blocked with 400."""
    malicious_files = [
        ("malware.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream"),
        ("exploit.html", b"<script>alert(1)</script>", "text/html"),
        ("vector.svg", b"<svg onload=alert(1)>", "image/svg+xml"),
        ("script.sh", b"#!/bin/bash\necho pwned", "application/x-sh"),
        ("archive.zip", b"PK\x03\x04", "application/zip"),
    ]
    for filename, content, mime in malicious_files:
        resp = client.post(
            "/upload",
            files={"video": (filename, content, mime)},
        )
        assert resp.status_code == 400, f"Expected 400 for prohibited file: {filename}"
        assert "Unsupported file format" in resp.json()["detail"]


def test_path_traversal_snapshot_blocked():
    """Verify that path traversal attempts on snapshots are rejected."""
    traversal_payloads = [
        "../../../../windows/system.ini",
        "..\\..\\..\\..\\windows\\system.ini",
        "../app.py",
        "..%2F..%2Fapp.py",
    ]
    for payload in traversal_payloads:
        resp = client.get(f"/api/snapshots/{payload}")
        assert resp.status_code in (400, 404)


def test_path_traversal_processed_video_blocked():
    """Verify that path traversal attempts on processed videos are rejected."""
    traversal_payloads = [
        "../../../../windows/system.ini",
        "..\\..\\..\\..\\windows\\system.ini",
        "../app.py",
    ]
    for payload in traversal_payloads:
        resp = client.get(f"/processed/{payload}")
        assert resp.status_code in (400, 404)

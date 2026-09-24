"""Live verification script for CMS Phase 1 platform endpoints."""

import sys
import httpx

BASE_URL = "http://127.0.0.1:8000"


def main():
    print("Connecting to live CMS server at:", BASE_URL)
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 1. Main page
        r_home = client.get("/")
        print(f"1. [GET /] Status: {r_home.status_code}")
        assert r_home.status_code == 200

        # 2. Analytics SOC Page
        r_analytics = client.get("/analytics")
        print(f"2. [GET /analytics] Status: {r_analytics.status_code}, Bytes: {len(r_analytics.content)}")
        assert r_analytics.status_code == 200

        # 3. Analytics Summary API
        r_summary = client.get("/api/analytics/summary")
        print(f"3. [GET /api/analytics/summary] Status: {r_summary.status_code}, Data: {r_summary.json()}")
        assert r_summary.status_code == 200

        # 4. Analytics Trends API
        r_trends = client.get("/api/analytics/trends?hours=24")
        print(f"4. [GET /api/analytics/trends] Status: {r_trends.status_code}, Points: {len(r_trends.json())}")
        assert r_trends.status_code == 200

        # 5. Security Sectors API
        r_zones = client.get("/api/zones")
        print(f"5. [GET /api/zones] Status: {r_zones.status_code}, Total Sectors: {len(r_zones.json())}")
        assert r_zones.status_code == 200

        # 6. Auth Login API
        r_login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        data = r_login.json()
        print(f"6. [POST /api/auth/login] Status: {r_login.status_code}, Role: {data.get('role')}")
        assert r_login.status_code == 200
        token = data.get("access_token")

        # 7. Auth Protected /me API
        r_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        print(f"7. [GET /api/auth/me] Status: {r_me.status_code}, User: {r_me.json().get('username')}")
        assert r_me.status_code == 200

        # 8. Live Surveillance Preview Page
        r_live = client.get("/live_preview")
        print(f"8. [GET /live_preview] Status: {r_live.status_code}")
        assert r_live.status_code == 200

        # 9. Video Studio Batch Interface Page
        r_studio = client.get("/video_studio")
        print(f"9. [GET /video_studio] Status: {r_studio.status_code}")
        assert r_studio.status_code == 200

        # 10. Toggle Pose & Zones APIs
        r_pose = client.post("/api/pose/toggle", json={"enabled": True})
        print(f"10. [POST /api/pose/toggle] Status: {r_pose.status_code}, Pose Enabled: {r_pose.json().get('pose_enabled')}")
        assert r_pose.status_code == 200

        r_zone = client.post("/api/zones/toggle", json={"enabled": True})
        print(f"11. [POST /api/zones/toggle] Status: {r_zone.status_code}, Zones Enabled: {r_zone.json().get('zones_enabled')}")
        assert r_zone.status_code == 200

    print("\nSUCCESS: All Phase 1 live server endpoints verified and operating normally!")


if __name__ == "__main__":
    main()

"""Async SQLite database manager for CMS.

Handles persistence for zones, occupancy history, safety incidents,
crowd snapshot telemetry, and user authentication accounts.
"""

import contextlib
import hashlib
import json
import os
import secrets
import sqlite3
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import aiosqlite

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "cms.db")


def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash password using PBKDF2 with SHA-256."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    ).hex()
    return f"{salt}${hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored salt and hash."""
    try:
        salt, _ = stored_hash.split("$", 1)
        return hash_password(password, salt) == stored_hash
    except Exception:
        return False


def init_db_sync(db_path: str = DB_PATH) -> None:
    """Synchronous initialization of the SQLite schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'operator',
            created_at REAL NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zones (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            color TEXT NOT NULL,
            polygon_json TEXT NOT NULL,
            max_occupancy INTEGER NOT NULL DEFAULT 10,
            camera_id TEXT NOT NULL DEFAULT 'primary',
            created_at REAL NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zone_occupancy_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            zone_id TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            current_occupancy INTEGER NOT NULL,
            avg_dwell_seconds REAL NOT NULL DEFAULT 0.0
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS safety_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            camera_id TEXT NOT NULL DEFAULT 'primary',
            zone_name TEXT NOT NULL DEFAULT 'General Area',
            confidence REAL NOT NULL DEFAULT 1.0,
            details_json TEXT NOT NULL DEFAULT '{}',
            snapshot_path TEXT,
            acknowledged INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crowd_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            camera_id TEXT NOT NULL DEFAULT 'primary',
            total_count INTEGER NOT NULL,
            density_level TEXT NOT NULL,
            frame_path TEXT
        )
        """
    )

    # Seed default admin user if users table is empty
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        default_admin_hash = hash_password("admin123")
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ("admin", default_admin_hash, "admin", time.time()),
        )
        default_operator_hash = hash_password("operator123")
        cur.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ("operator", default_operator_hash, "operator", time.time()),
        )

    # Seed sample zones if zones table is empty
    cur.execute("SELECT COUNT(*) FROM zones")
    if cur.fetchone()[0] == 0:
        sample_zones = [
            (
                "zone-entry-a",
                "Main Entry A",
                "#FFE500",
                json.dumps([[40, 60], [280, 60], [280, 420], [40, 420]]),
                15,
                "primary",
                time.time(),
            ),
            (
                "zone-central-hub",
                "Central Concourse",
                "#A6FAFF",
                json.dumps([[300, 80], [700, 80], [700, 450], [300, 450]]),
                25,
                "primary",
                time.time(),
            ),
            (
                "zone-evac-corridor",
                "Emergency Exit",
                "#B8FF9F",
                json.dumps([[720, 100], [920, 100], [920, 460], [720, 460]]),
                10,
                "primary",
                time.time(),
            ),
        ]
        cur.executemany(
            """
            INSERT INTO zones (id, name, color, polygon_json, max_occupancy, camera_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            sample_zones,
        )

    # Seed sample safety incidents if safety_events table is empty
    cur.execute("SELECT COUNT(*) FROM safety_events")
    if cur.fetchone()[0] == 0:
        now = time.time()
        sample_events = [
            (
                "inc-fall-101",
                "fall",
                now - 3600,
                "primary",
                "Central Concourse",
                0.94,
                json.dumps({"posture_angle": 64.2, "duration": 2.1}),
                None,
                1,
            ),
            (
                "inc-breach-102",
                "zone_breach",
                now - 1800,
                "primary",
                "Main Entry A",
                0.98,
                json.dumps({"occupancy": 19, "capacity": 15}),
                None,
                0,
            ),
            (
                "inc-panic-103",
                "panic",
                now - 900,
                "primary",
                "Emergency Exit",
                0.89,
                json.dumps({"stride_velocity": 48.5}),
                None,
                0,
            ),
            (
                "inc-collapse-104",
                "collapse",
                now - 300,
                "primary",
                "Central Concourse",
                0.96,
                json.dumps({"stillness_seconds": 8.4}),
                None,
                0,
            ),
        ]
        cur.executemany(
            """
            INSERT INTO safety_events (id, event_type, timestamp, camera_id, zone_name, confidence, details_json, snapshot_path, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_events,
        )

    conn.commit()
    conn.close()


class DatabaseManager:
    """Asynchronous database manager interface."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        init_db_sync(self.db_path)

    @contextlib.asynccontextmanager
    async def get_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """Provide an async database connection context."""
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    # --- Safety Events Operations ---
    async def record_safety_event(
        self,
        event_id: str,
        event_type: str,
        timestamp: float,
        camera_id: str = "primary",
        zone_name: str = "General Area",
        confidence: float = 1.0,
        details: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a safety incident (fall, collapse, panic, overcrowding)."""
        details_str = json.dumps(details or {})
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO safety_events
                (id, event_type, timestamp, camera_id, zone_name, confidence, details_json, snapshot_path, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    event_id,
                    event_type,
                    timestamp,
                    camera_id,
                    zone_name,
                    confidence,
                    details_str,
                    snapshot_path,
                ),
            )
            await conn.commit()
        return {
            "id": event_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "camera_id": camera_id,
            "zone_name": zone_name,
            "confidence": confidence,
            "details": details or {},
            "snapshot_path": snapshot_path,
            "acknowledged": 0,
        }

    async def get_safety_events(
        self, limit: int = 50, unacknowledged_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve recent safety events."""
        query = "SELECT * FROM safety_events"
        params: List[Any] = []
        if unacknowledged_only:
            query += " WHERE acknowledged = 0"
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self.get_connection() as conn:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["details"] = json.loads(item["details_json"])
                    except Exception:
                        item["details"] = {}
                    results.append(item)
                return results

    async def acknowledge_safety_event(self, event_id: str) -> bool:
        """Mark a safety incident as acknowledged by security operator."""
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "UPDATE safety_events SET acknowledged = 1 WHERE id = ?", (event_id,)
            )
            await conn.commit()
            return cursor.rowcount > 0

    # --- Zone Management ---
    async def get_zones(self, camera_id: str = "primary") -> List[Dict[str, Any]]:
        """Fetch all configured polygonal security zones."""
        async with self.get_connection() as conn:
            async with conn.execute(
                "SELECT * FROM zones WHERE camera_id = ? ORDER BY created_at ASC", (camera_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                zones = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["polygon"] = json.loads(item["polygon_json"])
                    except Exception:
                        item["polygon"] = []
                    zones.append(item)
                return zones

    async def save_zone(
        self,
        zone_id: str,
        name: str,
        color: str,
        polygon: List[List[int]],
        max_occupancy: int = 10,
        camera_id: str = "primary",
    ) -> Dict[str, Any]:
        """Create or update a polygonal security zone."""
        polygon_json = json.dumps(polygon)
        now = time.time()
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO zones (id, name, color, polygon_json, max_occupancy, camera_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    color = excluded.color,
                    polygon_json = excluded.polygon_json,
                    max_occupancy = excluded.max_occupancy
                """,
                (zone_id, name, color, polygon_json, max_occupancy, camera_id, now),
            )
            await conn.commit()
        return {
            "id": zone_id,
            "name": name,
            "color": color,
            "polygon": polygon,
            "max_occupancy": max_occupancy,
            "camera_id": camera_id,
        }

    async def update_zone_polygon(
        self, zone_id: str, polygon: List[List[float]]
    ) -> Optional[Dict[str, Any]]:
        """Update polygon vertices for an existing security zone."""
        polygon_json = json.dumps(polygon)
        async with self.get_connection() as conn:
            cursor = await conn.execute(
                "UPDATE zones SET polygon_json = ? WHERE id = ?",
                (polygon_json, zone_id),
            )
            await conn.commit()
            if cursor.rowcount == 0:
                return None
            async with conn.execute("SELECT * FROM zones WHERE id = ?", (zone_id,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                item = dict(row)
                item["polygon"] = polygon
                return item

    async def delete_zone(self, zone_id: str) -> bool:
        """Remove a security zone configuration."""
        async with self.get_connection() as conn:
            cursor = await conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def log_zone_occupancy(
        self, zone_id: str, zone_name: str, occupancy: int, avg_dwell: float
    ) -> None:
        """Record real-time telemetry sample for a zone."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO zone_occupancy_log (timestamp, zone_id, zone_name, current_occupancy, avg_dwell_seconds)
                VALUES (?, ?, ?, ?, ?)
                """,
                (time.time(), zone_id, zone_name, occupancy, avg_dwell),
            )
            await conn.commit()

    # --- Crowd Telemetry Snapshots ---
    async def log_crowd_snapshot(
        self,
        total_count: int,
        density_level: str,
        camera_id: str = "primary",
        frame_path: Optional[str] = None,
    ) -> None:
        """Log a periodic crowd density sample."""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO crowd_snapshots (timestamp, camera_id, total_count, density_level, frame_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (time.time(), camera_id, total_count, density_level, frame_path),
            )
            await conn.commit()

    async def get_crowd_trends(self, hours: float = 24.0) -> List[Dict[str, Any]]:
        """Fetch historical crowd counts for chart rendering."""
        since = time.time() - (hours * 3600.0)
        async with self.get_connection() as conn:
            async with conn.execute(
                """
                SELECT timestamp, total_count, density_level
                FROM crowd_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
                """,
                (since,),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Aggregate high-level metrics for analytics command dashboard."""
        now = time.time()
        one_day_ago = now - 86400.0

        async with self.get_connection() as conn:
            # Total events past 24 hours
            async with conn.execute(
                "SELECT COUNT(*) FROM safety_events WHERE timestamp >= ?", (one_day_ago,)
            ) as cur:
                events_24h = (await cur.fetchone())[0]

            # Events by type
            async with conn.execute(
                """
                SELECT event_type, COUNT(*) as count
                FROM safety_events
                GROUP BY event_type
                """
            ) as cur:
                rows = await cur.fetchall()
                events_by_type = {row["event_type"]: row["count"] for row in rows}

            # Unacknowledged count
            async with conn.execute(
                "SELECT COUNT(*) FROM safety_events WHERE acknowledged = 0"
            ) as cur:
                unacknowledged_count = (await cur.fetchone())[0]

            # Peak crowd count past 24 hours
            async with conn.execute(
                "SELECT MAX(total_count) FROM crowd_snapshots WHERE timestamp >= ?",
                (one_day_ago,),
            ) as cur:
                row = await cur.fetchone()
                peak_crowd = row[0] if row and row[0] is not None else 0

            # Average crowd count past 24 hours
            async with conn.execute(
                "SELECT AVG(total_count) FROM crowd_snapshots WHERE timestamp >= ?",
                (one_day_ago,),
            ) as cur:
                row = await cur.fetchone()
                avg_crowd = round(row[0], 1) if row and row[0] is not None else 0.0

            # Zone count
            async with conn.execute("SELECT COUNT(*) FROM zones") as cur:
                zone_count = (await cur.fetchone())[0]

            return {
                "events_24h": events_24h,
                "events_by_type": events_by_type,
                "unacknowledged_count": unacknowledged_count,
                "peak_crowd": peak_crowd,
                "avg_crowd": avg_crowd,
                "zone_count": zone_count,
            }

    # --- User Authentication & RBAC ---
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user record by username."""
        async with self.get_connection() as conn:
            async with conn.execute(
                "SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?",
                (username,),
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None


# Shared singleton instance
db_manager = DatabaseManager()

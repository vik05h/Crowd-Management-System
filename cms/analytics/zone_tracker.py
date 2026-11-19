"""Interactive Polygonal Zone Tracking & Dwell-Time Analytics Engine.

Provides geometric point-in-polygon occupancy counting, track dwell-time
profiling, and per-zone capacity breach monitoring.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


class SecurityZone:
    """Represents a named polygonal security sector."""

    def __init__(
        self,
        zone_id: str,
        name: str,
        color: str,
        polygon: List[List[float]],
        max_occupancy: int = 10,
        camera_id: str = "primary",
    ):
        self.zone_id = zone_id
        self.name = name
        self.color = color  # Hex string, e.g. '#FFE500'
        self.polygon = polygon  # List of [x, y] coordinates
        self.max_occupancy = max_occupancy
        self.camera_id = camera_id

        # Runtime telemetry state
        self.current_occupancy: int = 0
        self.peak_occupancy: int = 0
        self.active_tracks: Set[int] = set()
        self.track_entry_times: Dict[int, float] = {}
        self.completed_dwell_times: List[float] = []
        self.breach_start_time: Optional[float] = None
        self.is_breached: bool = False

    def hex_to_bgr(self) -> Tuple[int, int, int]:
        """Convert hex color string to OpenCV BGR tuple."""
        clean = self.color.lstrip("#")
        if len(clean) == 6:
            r = int(clean[0:2], 16)
            g = int(clean[2:4], 16)
            b = int(clean[4:6], 16)
            return (b, g, r)
        return (0, 229, 255)  # Default yellow BGR

    def get_pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Resolve polygon vertices to absolute pixel coordinates."""
        if not self.polygon:
            return np.empty((0, 2), dtype=np.int32)

        pts = []
        for pt in self.polygon:
            px, py = pt[0], pt[1]
            # If coordinates are normalized in [0, 1] range, scale to frame dimensions
            if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and (frame_w > 1 and frame_h > 1):
                pts.append([int(px * frame_w), int(py * frame_h)])
            else:
                pts.append([int(px), int(py)])
        return np.array(pts, dtype=np.int32)

    def contains_point(self, x: float, y: float, frame_w: int, frame_h: int) -> bool:
        """Check if ground-plane foot coordinate lies within this polygon."""
        poly_pts = self.get_pixel_polygon(frame_w, frame_h)
        if len(poly_pts) < 3:
            return False
        # OpenCV pointPolygonTest returns >= 0 for inside or on boundary
        return cv2.pointPolygonTest(poly_pts, (float(x), float(y)), False) >= 0

    def update_tracks(self, present_track_ids: Set[int], now: float) -> None:
        """Update active tracks, compute dwell durations, and monitor breaches."""
        self.current_occupancy = len(present_track_ids)
        if self.current_occupancy > self.peak_occupancy:
            self.peak_occupancy = self.current_occupancy

        # Check new entrants
        for tid in present_track_ids:
            if tid not in self.track_entry_times:
                self.track_entry_times[tid] = now

        # Check departing tracks
        departed = self.active_tracks - present_track_ids
        for tid in departed:
            if tid in self.track_entry_times:
                dwell = now - self.track_entry_times.pop(tid)
                if dwell >= 0.5:
                    self.completed_dwell_times.append(dwell)
                    # Keep rolling window of last 200 dwell samples
                    if len(self.completed_dwell_times) > 200:
                        self.completed_dwell_times.pop(0)

        self.active_tracks = present_track_ids

        # Monitor occupancy breach
        if self.current_occupancy >= self.max_occupancy:
            if self.breach_start_time is None:
                self.breach_start_time = now
            self.is_breached = (now - self.breach_start_time) >= 2.0
        else:
            self.breach_start_time = None
            self.is_breached = False

    def get_avg_dwell_seconds(self, now: float) -> float:
        """Compute average dwell duration including active and completed tracks."""
        active_dwells = [now - t for t in self.track_entry_times.values()]
        all_dwells = self.completed_dwell_times + active_dwells
        if not all_dwells:
            return 0.0
        return round(float(np.mean(all_dwells)), 1)

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Serialize zone state for API and WebSocket broadcasts."""
        t_now = now or time.time()
        return {
            "id": self.zone_id,
            "name": self.name,
            "color": self.color,
            "polygon": self.polygon,
            "max_occupancy": self.max_occupancy,
            "camera_id": self.camera_id,
            "current_occupancy": self.current_occupancy,
            "peak_occupancy": self.peak_occupancy,
            "avg_dwell_seconds": self.get_avg_dwell_seconds(t_now),
            "is_breached": self.is_breached,
        }


class ZoneTracker:
    """Manages active zones and performs spatial matching on detections."""

    def __init__(self):
        self.zones: Dict[str, SecurityZone] = {}

    def set_zones(self, zone_defs: List[Dict[str, Any]]) -> None:
        """Synchronize zone definitions from database or user input."""
        current_ids = set(self.zones.keys())
        new_ids = set()

        for zdef in zone_defs:
            zid = str(zdef.get("id"))
            new_ids.add(zid)
            if zid in self.zones:
                zone = self.zones[zid]
                zone.name = zdef.get("name", zone.name)
                zone.color = zdef.get("color", zone.color)
                zone.polygon = zdef.get("polygon", zone.polygon)
                zone.max_occupancy = int(zdef.get("max_occupancy", zone.max_occupancy))
            else:
                self.zones[zid] = SecurityZone(
                    zone_id=zid,
                    name=zdef.get("name", "Zone"),
                    color=zdef.get("color", "#FFE500"),
                    polygon=zdef.get("polygon", []),
                    max_occupancy=int(zdef.get("max_occupancy", 10)),
                    camera_id=zdef.get("camera_id", "primary"),
                )

        # Remove deleted zones
        for removed_id in current_ids - new_ids:
            self.zones.pop(removed_id, None)

    def update_zone_polygon(self, zone_id: str, polygon: List[List[float]]) -> bool:
        """Update polygon vertices of an existing zone."""
        if zone_id in self.zones:
            self.zones[zone_id].polygon = polygon
            return True
        return False

    def update(
        self,
        tracks: List[Tuple[int, float, float, float, float]],
        frame_w: int,
        frame_h: int,
        now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Update all zones given active tracks [(track_id, x1, y1, x2, y2), ...].

        Returns list of newly breached zone dictionaries.
        """
        t_now = now or time.time()
        new_breaches: List[Dict[str, Any]] = []

        # Map person foot coordinates: (center_x, bottom_y)
        track_feet: Dict[int, Tuple[float, float]] = {}
        for tid, x1, y1, x2, y2 in tracks:
            cx = (x1 + x2) / 2.0
            feet_y = y2
            track_feet[tid] = (cx, feet_y)

        for zone in self.zones.values():
            was_breached = zone.is_breached
            present_tids: Set[int] = set()

            for tid, (cx, cy) in track_feet.items():
                if zone.contains_point(cx, cy, frame_w, frame_h):
                    present_tids.add(tid)

            zone.update_tracks(present_tids, t_now)

            if zone.is_breached and not was_breached:
                new_breaches.append(zone.to_dict(t_now))

        return new_breaches

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Render polygonal zones with semi-transparent tint and Neo-Brutalism badges."""
        if not self.zones or frame is None:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        for zone in self.zones.values():
            poly = zone.get_pixel_polygon(w, h)
            if len(poly) < 3:
                continue

            bgr = zone.hex_to_bgr()

            # Fill zone polygon with semi-transparent color tint
            if zone.is_breached:
                # Flashing crimson red when breached
                cv2.fillPoly(overlay, [poly], (0, 0, 220))
            else:
                cv2.fillPoly(overlay, [poly], bgr)

        # Alpha blend filled zones (alpha 0.22)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        # Draw solid boundaries and Neo-Brutalism metadata badges
        for zone in self.zones.values():
            poly = zone.get_pixel_polygon(w, h)
            if len(poly) < 3:
                continue

            bgr = zone.hex_to_bgr()
            border_color = (0, 0, 255) if zone.is_breached else bgr

            # Solid 2px outline
            cv2.polylines(frame, [poly], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            # Badge anchor at top-most polygon vertex
            min_idx = int(np.argmin(poly[:, 1]))
            anchor_x = int(poly[min_idx, 0])
            anchor_y = max(24, int(poly[min_idx, 1]) - 8)

            badge_text = f"{zone.name} | {zone.current_occupancy}/{zone.max_occupancy}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.44
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

            bx1 = max(4, anchor_x - 4)
            by1 = max(4, anchor_y - th - 6)
            bx2 = min(w - 4, bx1 + tw + 14)
            by2 = min(h - 4, by1 + th + 10)

            # Neo-Brutalism badge: solid black border with high contrast fill
            badge_bg = (0, 0, 220) if zone.is_breached else (24, 24, 27)
            text_color = (255, 255, 255)

            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)  # Drop shadow
            cv2.rectangle(frame, (bx1 - 2, by1 - 2), (bx2 - 2, by2 - 2), badge_bg, -1)
            cv2.rectangle(frame, (bx1 - 2, by1 - 2), (bx2 - 2, by2 - 2), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                badge_text,
                (bx1 + 4, by2 - baseline - 2),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return frame

    def get_summary(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return serialized state of all managed zones."""
        t_now = now or time.time()
        return [zone.to_dict(t_now) for zone in self.zones.values()]

    def get_all_zones(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Alias for get_summary to return all zone definitions with live occupancy."""
        return self.get_summary(now)

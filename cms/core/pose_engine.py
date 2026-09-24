"""YOLO-Pose Human Keypoint Intelligence & Safety Event Engine.

Detects falls, collapses, and panic/running behavior from 17 COCO keypoints.
Feeds sustained incidents directly into the security alerting pipeline.
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# 17 COCO Keypoints connection skeleton graph
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial landmarks
    (5, 6),                                    # Shoulders
    (5, 7), (7, 9),                            # Left arm
    (6, 8), (8, 10),                           # Right arm
    (5, 11), (6, 12), (11, 12),                # Torso box
    (11, 13), (13, 15),                        # Left leg
    (12, 14), (14, 16),                        # Right leg
]


class PoseEngine:
    """Ultralytics YOLO-Pose inference and heuristic behavior analyzer."""

    def __init__(self, model_path: str = "yolo26n-pose.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model: Optional[YOLO] = None
        self.use_fp16: bool = (self.device == "cuda")
        self.pose_stride: int = 2
        self.frame_counter: int = 0
        self.cached_skeletons: List[Dict[str, Any]] = []
        self.latest_latency_ms: float = 0.0

        self.load_model(self.model_path)

        # Temporal tracking state per spatial cluster / track ID
        self.person_states: Dict[str, Dict[str, Any]] = {}
        self.last_cleanup_time: float = time.time()

    def load_model(self, model_path: str) -> bool:
        """Dynamically load or hot-swap YOLO26 pose model weights."""
        try:
            self.model = YOLO(model_path).to(self.device)
            self.model_path = model_path
            print(f"[INFO] PoseEngine loaded {self.model_path} on {self.device}")
            return True
        except Exception as exc:
            print(f"[WARN] PoseEngine could not load {model_path}: {exc}. Pose detection will operate in passive mode.")
            return False

    def set_stride(self, stride: int) -> None:
        """Set temporal stride for frame skipping (1 = every frame, 2 = every 2nd frame)."""
        self.pose_stride = max(1, min(5, stride))

    def set_fp16(self, enabled: bool) -> None:
        """Enable or disable FP16 half precision execution on CUDA."""
        self.use_fp16 = enabled and (self.device == "cuda")

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: Optional[List[Tuple[int, float, float, float, float]]] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
        half: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run pose estimation on frame with optional temporal stride and FP16.

        Returns:
            skeletons: List of detected person skeleton dicts (boxes, keypoints, event_state)
            events: List of new confirmed safety incident dicts (fall, collapse, panic)
        """
        if self.model is None or frame is None:
            return [], []

        h, w = frame.shape[:2]
        now = time.time()
        self.frame_counter += 1

        # Check temporal stride: skip heavy neural inference on intermediate frames
        if self.pose_stride > 1 and (self.frame_counter % self.pose_stride != 0) and self.cached_skeletons:
            propagated_skeletons = []
            track_lookup = {}
            if tracks:
                for tid, tx1, ty1, tx2, ty2 in tracks:
                    track_lookup[tid] = (tx1, ty1, tx2, ty2)

            for skel in self.cached_skeletons:
                copied = dict(skel)
                skel_tid = copied.get("track_id")
                if skel_tid in track_lookup:
                    new_box = list(track_lookup[skel_tid])
                    old_box = copied.get("box", new_box)
                    old_cx = (old_box[0] + old_box[2]) / 2.0
                    old_cy = (old_box[1] + old_box[3]) / 2.0
                    new_cx = (new_box[0] + new_box[2]) / 2.0
                    new_cy = (new_box[1] + new_box[3]) / 2.0
                    shift_x = new_cx - old_cx
                    shift_y = new_cy - old_cy

                    copied["box"] = new_box
                    if "keypoints" in copied:
                        updated_kpts = []
                        for kx, ky in copied["keypoints"]:
                            updated_kpts.append([kx + shift_x, ky + shift_y])
                        copied["keypoints"] = updated_kpts
                propagated_skeletons.append(copied)
            return propagated_skeletons, []

        use_half = self.use_fp16 if half is None else (half and self.device == "cuda")

        # Run pose inference
        t_start = time.perf_counter()
        try:
            results = self.model(
                frame,
                verbose=False,
                conf=conf_threshold,
                device=self.device,
                imgsz=imgsz,
                half=use_half,
            )
            self.latest_latency_ms = (time.perf_counter() - t_start) * 1000.0
        except Exception as exc:
            print(f"[WARN] Pose inference error: {exc}")
            return [], []

        if not results or len(results) == 0:
            return [], []

        res = results[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return [], []

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts_xy = res.keypoints.xy.cpu().numpy()  # shape [N, 17, 2]
        kpts_conf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones((len(boxes_xyxy), 17))

        skeletons: List[Dict[str, Any]] = []
        new_incidents: List[Dict[str, Any]] = []

        active_keys = set()

        for idx, (box, box_conf, kpt, kconf) in enumerate(zip(boxes_xyxy, confs, kpts_xy, kpts_conf)):
            x1, y1, x2, y2 = box[:4]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Match with ByteTrack track ID if available, else spatial quantization key
            matched_tid = None
            if tracks:
                best_dist = float("inf")
                for tid, tx1, ty1, tx2, ty2 in tracks:
                    tcx = (tx1 + tx2) / 2.0
                    tcy = (ty1 + ty2) / 2.0
                    dist = math.hypot(cx - tcx, cy - tcy)
                    if dist < max(bw, bh) * 0.6 and dist < best_dist:
                        best_dist = dist
                        matched_tid = tid

            track_key = f"track_{matched_tid}" if matched_tid is not None else f"pos_{int(cx // 40)}_{int(cy // 40)}"
            active_keys.add(track_key)

            if track_key not in self.person_states:
                self.person_states[track_key] = {
                    "first_fall_time": None,
                    "fall_confirmed": False,
                    "collapse_confirmed": False,
                    "first_still_time": None,
                    "panic_start_time": None,
                    "panic_confirmed": False,
                    "last_pos": (cx, cy),
                    "last_seen": now,
                }

            state = self.person_states[track_key]
            state["last_seen"] = now

            # Keypoint indices
            # 5: L_shoulder, 6: R_shoulder, 11: L_hip, 12: R_hip, 15: L_ankle, 16: R_ankle
            l_sh, r_sh = kpt[5], kpt[6]
            l_hip, r_hip = kpt[11], kpt[12]
            l_ank, r_ank = kpt[15], kpt[16]

            has_torso = (kconf[5] > 0.3 or kconf[6] > 0.3) and (kconf[11] > 0.3 or kconf[12] > 0.3)

            # Evaluate posture
            is_fall_candidate = False
            aspect_ratio = bw / bh  # > 1.1 suggests horizontal orientation

            if has_torso:
                mid_sh_x = (l_sh[0] + r_sh[0]) / 2.0
                mid_sh_y = (l_sh[1] + r_sh[1]) / 2.0
                mid_hip_x = (l_hip[0] + r_hip[0]) / 2.0
                mid_hip_y = (l_hip[1] + r_hip[1]) / 2.0

                dx = abs(mid_sh_x - mid_hip_x)
                dy = abs(mid_sh_y - mid_hip_y)

                # Torso angle with vertical axis
                angle_deg = math.degrees(math.atan2(dx, max(dy, 1e-4)))
                if angle_deg > 52.0 or aspect_ratio > 1.25:
                    is_fall_candidate = True
            else:
                if aspect_ratio > 1.35 and bh < (h * 0.45):
                    is_fall_candidate = True

            # Evaluate movement velocity for panic detection
            last_cx, last_cy = state["last_pos"]
            movement_px = math.hypot(cx - last_cx, cy - last_cy)
            state["last_pos"] = (cx, cy)

            event_status = "normal"

            # 1. Fall & Collapse sustained state machine
            if is_fall_candidate:
                if state["first_fall_time"] is None:
                    state["first_fall_time"] = now
                fall_duration = now - state["first_fall_time"]

                if fall_duration >= 1.6 and not state["fall_confirmed"]:
                    state["fall_confirmed"] = True
                    event_status = "fall"
                    new_incidents.append({
                        "event_type": "fall",
                        "track_id": matched_tid,
                        "confidence": float(box_conf),
                        "box": [float(x1), float(y1), float(x2), float(y2)],
                        "timestamp": now,
                        "details": {"duration": round(fall_duration, 1), "aspect_ratio": round(aspect_ratio, 2)},
                    })
                elif state["fall_confirmed"]:
                    event_status = "fall"

                # Check for unconscious / collapsed person (still for >= 6.0 seconds)
                if movement_px < 6.0:
                    if state["first_still_time"] is None:
                        state["first_still_time"] = now
                    still_duration = now - state["first_still_time"]
                    if still_duration >= 6.0 and not state["collapse_confirmed"]:
                        state["collapse_confirmed"] = True
                        event_status = "collapse"
                        new_incidents.append({
                            "event_type": "collapse",
                            "track_id": matched_tid,
                            "confidence": float(box_conf),
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "timestamp": now,
                            "details": {"unresponsive_seconds": round(still_duration, 1)},
                        })
                    elif state["collapse_confirmed"]:
                        event_status = "collapse"
                else:
                    state["first_still_time"] = None
            else:
                state["first_fall_time"] = None
                state["fall_confirmed"] = False
                state["first_still_time"] = None
                state["collapse_confirmed"] = False

                # 2. Panic / Rapid running surge evaluation
                # If displacement per frame exceeds 18% of person height
                if movement_px > (bh * 0.28) and bh > 30:
                    if state["panic_start_time"] is None:
                        state["panic_start_time"] = now
                    panic_duration = now - state["panic_start_time"]
                    if panic_duration >= 1.2 and not state["panic_confirmed"]:
                        state["panic_confirmed"] = True
                        event_status = "panic"
                        new_incidents.append({
                            "event_type": "panic",
                            "track_id": matched_tid,
                            "confidence": float(box_conf),
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                            "timestamp": now,
                            "details": {"stride_velocity": round(movement_px, 1)},
                        })
                    elif state["panic_confirmed"]:
                        event_status = "panic"
                else:
                    state["panic_start_time"] = None
                    state["panic_confirmed"] = False

            skeletons.append({
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(box_conf),
                "keypoints": kpt.tolist(),
                "kconf": kconf.tolist(),
                "event_status": event_status,
                "track_id": matched_tid,
            })

        # Periodic cleanup of expired track keys (inactive for > 15s)
        if now - self.last_cleanup_time > 10.0:
            expired = [k for k, v in self.person_states.items() if now - v.get("last_seen", 0) > 15.0]
            for exp_k in expired:
                self.person_states.pop(exp_k, None)
            self.last_cleanup_time = now

        self.cached_skeletons = skeletons
        return skeletons, new_incidents

    def draw_skeletons(
        self,
        frame: np.ndarray,
        skeletons: List[Dict[str, Any]],
        show_bones: bool = True,
        show_badges: bool = True,
    ) -> np.ndarray:
        """Render 17 COCO skeletal bones and Neo-Brutalism status badges on frame."""
        if frame is None or not skeletons:
            return frame

        h, w = frame.shape[:2]

        for skel in skeletons:
            status = skel.get("event_status", "normal")
            kpts = skel.get("keypoints", [])
            kconfs = skel.get("kconf", [])
            box = skel.get("box", [0, 0, 0, 0])
            tid = skel.get("track_id")

            # Determine tactical color by event status
            if status == "fall":
                bone_color = (0, 0, 240)    # Crimson red
                dot_color = (255, 255, 255)
                badge_bg = (0, 0, 220)
                status_text = f"FALL DETECTED | ID {tid}" if tid else "FALL DETECTED"
            elif status == "collapse":
                bone_color = (0, 140, 255)  # Amber orange
                dot_color = (255, 255, 255)
                badge_bg = (0, 120, 230)
                status_text = f"UNRESPONSIVE | ID {tid}" if tid else "UNRESPONSIVE"
            elif status == "panic":
                bone_color = (246, 166, 255)  # Electric magenta
                dot_color = (0, 0, 0)
                badge_bg = (200, 50, 200)
                status_text = f"PANIC STRIDE | ID {tid}" if tid else "PANIC STRIDE"
            else:
                bone_color = (255, 230, 0)   # Luminous cyan / pale aqua BGR
                dot_color = (0, 255, 128)
                badge_bg = (24, 24, 27)
                status_text = f"TRACK {tid}" if tid else None

            # Draw skeletal bones
            if show_bones and len(kpts) == 17:
                for idx1, idx2 in SKELETON_CONNECTIONS:
                    if kconfs[idx1] > 0.35 and kconfs[idx2] > 0.35:
                        pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                        pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                        cv2.line(frame, pt1, pt2, (0, 0, 0), 3, cv2.LINE_AA)  # Black drop outline
                        cv2.line(frame, pt1, pt2, bone_color, 2, cv2.LINE_AA)

                # Draw joint circles
                for kx, ky in kpts:
                    px, py = int(kx), int(ky)
                    if 0 <= px < w and 0 <= py < h:
                        cv2.circle(frame, (px, py), 3, (0, 0, 0), -1)
                        cv2.circle(frame, (px, py), 2, dot_color, -1)

            # Draw Neo-Brutalism badge for safety events or track identification
            if show_badges and status_text:
                x1, y1, x2, y2 = box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.42
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)

                bx1 = max(4, int(x1))
                by1 = max(4, int(y1) - th - 8)
                bx2 = min(w - 4, bx1 + tw + 12)
                by2 = min(h - 4, by1 + th + 6)

                # Tactical shadow and hard border
                cv2.rectangle(frame, (bx1 + 2, by1 + 2), (bx2 + 2, by2 + 2), (0, 0, 0), -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), badge_bg, -1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    status_text,
                    (bx1 + 5, by2 - baseline - 1),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return frame

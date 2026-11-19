import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import imageio_ffmpeg
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from cms.analytics.zone_tracker import ZoneTracker
from cms.core.pose_engine import PoseEngine
from cms.privacy.face_blur import PrivacyManager

# Configuration defaults
CROWD_THRESHOLD = 30
ALERT_DELAY = 15
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

NUM_GRID_ROWS = 6
NUM_GRID_COLS = 6
GRID_CELL_THRESHOLD = 3


def resolve_model_path(preferred_path: Optional[str] = None) -> str:
    """Resolve model weights following the established fallback hierarchy."""
    candidates = []
    if preferred_path:
        candidates.append(preferred_path)

    env_path = os.getenv("CMS_MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    # Prioritize fine-tuned CrowdHuman model, then transfer model, then custom, then pretrained YOLO26m
    candidates.extend([
        "runs/detect/yolo26m_crowdhuman/weights/best.pt",
        "runs/detect/yolo26m_clean_transfer/weights/best.pt",
        "runs/detect/runs/detect/yolo26m_clean_transfer/weights/best.pt",
        "runs/detect/yolo26m_custom/weights/best.pt",
        "yolo26m.pt",
        "yolo11n.pt",
    ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    return "yolo26m.pt"


class YOLOInference:
    """Inference and crowd tracking engine using YOLO26 and ByteTrack."""

    def __init__(self, model_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resolved_model_path = resolve_model_path(model_path)

        try:
            self.model = YOLO(self.resolved_model_path).to(self.device)
            print(f"[INFO] Loaded YOLO model from {self.resolved_model_path} on {self.device}")
        except Exception as exc:
            print(f"[WARN] Failed to load {self.resolved_model_path}: {exc}. Falling back to yolo26m.pt")
            self.resolved_model_path = "yolo26m.pt"
            self.model = YOLO(self.resolved_model_path).to(self.device)

        # Legacy tracker reference for backward compatibility
        self.tracker = None

        # Thread management and single-worker guarantee
        self.video_thread_lock = threading.Lock()
        self.video_thread: Optional[threading.Thread] = None
        self.current_video_path: Optional[str] = None
        self.frame_id: int = 0

        # Processing state
        self.density_map: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.enable_heat_map: bool = False
        self.video_processing_active: bool = False
        self.current_people_count: int = 0
        self.frame_lock = threading.Lock()
        self.frame_condition = threading.Condition(self.frame_lock)
        self.latest_jpeg_bytes: Optional[bytes] = None
        self.inference_imgsz: int = 640

        # Overcrowding alert parameters
        self.alert_threshold: int = 15
        self.sustained_duration: float = 3.0
        self.cooldown_duration: float = 30.0
        self.breach_start_time: Optional[float] = None
        self.last_alert_time: float = 0.0
        self.active_alert_triggered: bool = False
        self.alert_callbacks: List[Any] = []

        # Pose Keypoint Intelligence & Behavioral Safety Engine (lazy load to conserve system RAM)
        self.pose_engine = PoseEngine("yolo26n-pose.pt", lazy_load=True)
        self.enable_pose: bool = True
        self.pose_callbacks: List[Any] = []

        # Polygonal Security Zones & Dwell Analytics Engine (default to False for clean preview)
        self.zone_tracker = ZoneTracker()
        self.enable_zones: bool = False
        self.zone_callbacks: List[Any] = []

        # Hardware Acceleration & Tiered Performance Profiles
        self.use_fp16: bool = (self.device == "cuda")
        self.hardware_profile: str = "eco"  # "eco", "balanced", "ultra", "custom"
        self.detection_latency_ms: float = 0.0
        self.fps_telemetry: float = 25.0

        # Privacy Anonymization Engine (DPDPA 2023 / GDPR)
        self.privacy_manager = PrivacyManager(enabled=False)

        # Grid-based zooming
        self.zoom_row: Optional[int] = None
        self.zoom_col: Optional[int] = None

        # Batch video processing state (Uploaded video analysis)
        self.batch_thread: Optional[threading.Thread] = None
        self.batch_thread_lock = threading.Lock()
        self.batch_frame_lock = threading.Lock()
        self.batch_latest_frame: Optional[np.ndarray] = None
        self.batch_frame_id: int = 0
        self.batch_job: Dict[str, Any] = {
            "active": False,
            "input_file": "",
            "output_file": "",
            "current_frame": 0,
            "total_frames": 0,
            "percent": 0.0,
            "fps": 0.0,
            "eta_seconds": 0,
            "is_complete": False,
            "error": None,
            "peak_count": 0,
            "avg_count": 0.0,
            "total_breaches": 0,
            "current_count": 0,
        }

    def ensure_video_processing(self, input_path: str, output_path: str, loop: bool = True) -> None:
        """Ensure a single video worker is active without restarting if already running."""
        with self.video_thread_lock:
            if (
                self.video_processing_active
                and self.current_video_path == input_path
                and self.video_thread is not None
                and self.video_thread.is_alive()
            ):
                return
            self._start_video_worker_locked(input_path, output_path, loop)

    def start_video_processing(self, input_path: str, output_path: str, loop: bool = True) -> None:
        """Force start or restart the single video worker thread for the given video."""
        with self.video_thread_lock:
            self._start_video_worker_locked(input_path, output_path, loop)

    def _start_video_worker_locked(self, input_path: str, output_path: str, loop: bool = True) -> None:
        """Internal worker launcher that safely terminates any prior running thread."""
        if self.video_processing_active:
            print("[INFO] Terminating previous video processing worker...")
            self.video_processing_active = False
            if self.video_thread is not None and self.video_thread.is_alive():
                self.video_thread.join(timeout=2.0)

        self.video_processing_active = True
        self.current_video_path = input_path
        self.reset_tracker()
        self.video_thread = threading.Thread(
            target=self.process_video,
            args=(input_path, output_path, loop),
            daemon=True,
        )
        self.video_thread.start()
        print(f"[INFO] Started single video worker thread for: {input_path}")

    def stop_video_processing(self) -> None:
        """Safely terminate the active video processing worker."""
        with self.video_thread_lock:
            self.video_processing_active = False
            with self.frame_condition:
                self.frame_condition.notify_all()
            if self.video_thread is not None and self.video_thread.is_alive():
                self.video_thread.join(timeout=2.0)
            self.video_thread = None
            self.current_video_path = None

    def _publish_frame(self, frame: np.ndarray) -> None:
        """Encode frame to JPEG in full native quality and push to all waiting stream consumers."""
        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = buf.tobytes() if ret else None

        with self.frame_condition:
            if self.latest_frame is None or self.latest_frame.shape != frame.shape:
                self.latest_frame = frame.copy()
            else:
                np.copyto(self.latest_frame, frame)
            self.latest_jpeg_bytes = jpeg_bytes
            self.frame_id += 1
            self.frame_condition.notify_all()

    def reset_tracker(self) -> None:
        """Reset internal tracker state on video rewind or restart."""
        try:
            if hasattr(self.model, "predictor") and self.model.predictor:
                if hasattr(self.model.predictor, "trackers"):
                    for tracker in self.model.predictor.trackers:
                        if hasattr(tracker, "reset"):
                            tracker.reset()
        except Exception:
            pass

    def set_heatmap_enabled(self, state: bool) -> None:
        """Toggle density heatmap overlay on or off."""
        self.enable_heat_map = state
        if not state and self.density_map is not None:
            self.density_map.fill(0)
        print(f"[INFO] Heatmap enabled: {self.enable_heat_map}")

    def set_alert_threshold(self, threshold: int) -> None:
        """Set the crowd headcount alert threshold."""
        self.alert_threshold = max(1, threshold)
        print(f"[INFO] Alert threshold set to {self.alert_threshold}")

    def set_inference_imgsz(self, imgsz: int) -> None:
        """Set inference resolution (e.g., 640 or 1024)."""
        self.inference_imgsz = imgsz
        print(f"[INFO] Inference image size set to {self.inference_imgsz}")

    def register_alert_callback(self, callback: Any) -> None:
        """Register callback invoked when sustained overcrowding occurs."""
        self.alert_callbacks.append(callback)

    def set_pose_enabled(self, state: bool) -> None:
        """Toggle pose estimation overlay on or off with automatic RAM/VRAM unloading."""
        self.enable_pose = state
        if state:
            self.pose_engine.ensure_model_loaded()
        else:
            self.pose_engine.unload_model()
        print(f"[INFO] Pose intelligence enabled: {self.enable_pose}")

    def set_zones_enabled(self, state: bool) -> None:
        """Toggle polygonal zone overlay on or off."""
        self.enable_zones = state
        print(f"[INFO] Zones overlay enabled: {self.enable_zones}")

    def update_zones(self, zone_defs: List[Dict[str, Any]]) -> None:
        """Update active security sectors."""
        self.zone_tracker.set_zones(zone_defs)

    def update_zone_polygon(self, zone_id: str, polygon: List[List[float]]) -> bool:
        """Update polygon vertices of an existing zone in memory."""
        return self.zone_tracker.update_zone_polygon(zone_id, polygon)

    def get_zones_summary(self) -> List[Dict[str, Any]]:
        """Return real-time state of all security zones."""
        return self.zone_tracker.get_summary()

    def set_hardware_profile(self, profile: str) -> Dict[str, Any]:
        """Apply tiered hardware performance profile."""
        p = profile.lower().strip()
        if p == "eco":
            self.inference_imgsz = 640
            self.use_fp16 = (self.device == "cuda")
            self.pose_engine.model_path = "yolo26n-pose.pt"
            if self.enable_pose:
                self.pose_engine.load_model("yolo26n-pose.pt")
            self.pose_engine.set_stride(2)
            self.pose_engine.set_fp16(self.use_fp16)
            self.hardware_profile = "eco"
        elif p == "balanced":
            self.inference_imgsz = 640
            self.use_fp16 = (self.device == "cuda")
            self.pose_engine.model_path = "yolo26s-pose.pt"
            if self.enable_pose:
                self.pose_engine.load_model("yolo26s-pose.pt")
            self.pose_engine.set_stride(1)
            self.pose_engine.set_fp16(self.use_fp16)
            self.hardware_profile = "balanced"
        elif p == "ultra":
            self.inference_imgsz = 1024
            self.use_fp16 = (self.device == "cuda")
            self.pose_engine.model_path = "yolo26m-pose.pt"
            if self.enable_pose:
                self.pose_engine.load_model("yolo26m-pose.pt")
            self.pose_engine.set_stride(1)
            self.pose_engine.set_fp16(self.use_fp16)
            self.hardware_profile = "ultra"
        else:
            self.hardware_profile = "custom"
        return self.get_hardware_status()

    def set_pose_model(self, model_name: str) -> bool:
        """Dynamically load or switch YOLO26 pose model."""
        ok = self.pose_engine.load_model(model_name)
        if ok:
            self.hardware_profile = "custom"
        return ok

    def set_pose_stride(self, stride: int) -> None:
        """Set pose estimation temporal frame stride."""
        self.pose_engine.set_stride(stride)
        self.hardware_profile = "custom"

    def set_fp16_enabled(self, enabled: bool) -> None:
        """Toggle FP16 half-precision execution on CUDA."""
        self.use_fp16 = enabled and (self.device == "cuda")
        self.pose_engine.set_fp16(self.use_fp16)
        self.hardware_profile = "custom"

    def get_hardware_status(self) -> Dict[str, Any]:
        """Fetch current hardware capability and active performance configuration."""
        gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU Host"
        gpu_vram_total_mb = 0
        gpu_vram_used_mb = 0
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            gpu_vram_total_mb = int(props.total_memory / (1024 * 1024))
            gpu_vram_used_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024))

        return {
            "device": self.device,
            "gpu_name": gpu_name,
            "vram_total_mb": gpu_vram_total_mb,
            "vram_used_mb": gpu_vram_used_mb,
            "profile": self.hardware_profile,
            "inference_imgsz": self.inference_imgsz,
            "fp16_enabled": self.use_fp16,
            "pose_model": self.pose_engine.model_path,
            "pose_stride": self.pose_engine.pose_stride,
            "pose_enabled": self.enable_pose,
            "zones_enabled": self.enable_zones,
            "privacy_enabled": self.privacy_manager.enabled,
            "det_latency_ms": round(self.detection_latency_ms, 1),
            "pose_latency_ms": round(self.pose_engine.latest_latency_ms, 1),
            "fps": round(self.fps_telemetry, 1),
        }

    def toggle_privacy(self, enabled: Optional[bool] = None) -> bool:
        """Toggle or explicitly set real-time face privacy anonymization."""
        if enabled is not None:
            self.privacy_manager.set_enabled(enabled)
        else:
            self.privacy_manager.toggle()
        return self.privacy_manager.enabled

    def get_privacy_status(self) -> Dict[str, Any]:
        """Return privacy anonymization state and compliance standard."""
        return self.privacy_manager.get_status()

    def register_pose_callback(self, callback: Any) -> None:
        """Register callback invoked when a pose event (fall, collapse, panic) occurs."""
        self.pose_callbacks.append(callback)

    def register_zone_callback(self, callback: Any) -> None:
        """Register callback invoked when a zone capacity breach occurs."""
        self.zone_callbacks.append(callback)

    def check_and_trigger_alert(self, frame: np.ndarray, count: int) -> bool:
        """Evaluate sustained overcrowding condition and trigger incident logging/alerting."""
        now = time.time()
        if count >= self.alert_threshold:
            if self.breach_start_time is None:
                self.breach_start_time = now
            elif (now - self.breach_start_time) >= self.sustained_duration:
                self.active_alert_triggered = True
                if (now - self.last_alert_time) >= self.cooldown_duration:
                    self.last_alert_time = now
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    snapshot_filename = f"incident_{timestamp_str}.jpg"
                    snapshot_path = os.path.join(SNAPSHOT_FOLDER, snapshot_filename)
                    cv2.imwrite(snapshot_path, frame)
                    print(
                        f"[ALERT] Sustained crowd threshold breached: {count} people "
                        f"(threshold={self.alert_threshold}). Snapshot: {snapshot_path}"
                    )

                    incident = {
                        "id": f"inc_{int(now)}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "people_count": count,
                        "threshold": self.alert_threshold,
                        "snapshot_filename": snapshot_filename,
                        "snapshot_path": snapshot_path,
                        "acknowledged": False,
                    }
                    for cb in self.alert_callbacks:
                        try:
                            cb(incident)
                        except Exception as exc:
                            print(f"[WARN] Error executing alert callback: {exc}")
                    return True
        else:
            self.breach_start_time = None
            if (now - self.last_alert_time) >= self.cooldown_duration:
                self.active_alert_triggered = False

        return False

    def set_zoom_cell(self, row: int, col: int) -> None:
        """Set grid cell to magnify. Negative values reset zoom."""
        if row < 0 or col < 0:
            self.zoom_row, self.zoom_col = None, None
            print("[INFO] Zoom reset.")
        else:
            self.zoom_row, self.zoom_col = row, col
            print(f"[INFO] Zoom cell set to row={row}, col={col}")

    def get_zoomed_subimage(self) -> Optional[np.ndarray]:
        """Return zoomed subimage for the selected grid cell."""
        with self.frame_lock:
            if self.latest_frame is None or self.zoom_row is None or self.zoom_col is None:
                return None
            frame = self.latest_frame.copy()

        height, width = frame.shape[:2]
        cell_height = max(1, height // NUM_GRID_ROWS)
        cell_width = max(1, width // NUM_GRID_COLS)

        start_y = self.zoom_row * cell_height
        end_y = min(height, (self.zoom_row + 1) * cell_height)
        start_x = self.zoom_col * cell_width
        end_x = min(width, (self.zoom_col + 1) * cell_width)

        subimg = frame[start_y:end_y, start_x:end_x]
        if subimg.size == 0:
            return None

        return cv2.resize(subimg, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)

    def _apply_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Apply a smoothed Gaussian thermal heatmap overlay onto the frame."""
        if self.density_map is None:
            return frame

        # Gaussian blur on quarter-resolution density map (16x faster compute, ~200KB transient)
        blurred = cv2.GaussianBlur(self.density_map, (17, 17), 0)
        max_val = float(blurred.max())
        if max_val > 1e-4:
            norm_map = np.uint8(255 * (blurred / max_val))
            heatmap_small = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            h, w = frame.shape[:2]
            heatmap = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_LINEAR)
            norm_full = cv2.resize(norm_map, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = norm_full > 15
            # In-place blending without allocating extra frame copies
            frame[mask] = cv2.addWeighted(frame[mask], 0.45, heatmap[mask], 0.55, 0)
            self.density_map *= 0.95

        return frame

    @staticmethod
    def _rects_overlap(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding rectangles (x1, y1, x2, y2) overlap."""
        return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

    @staticmethod
    def _nms(
        boxes: List[Tuple[int, int, int, int, float, Optional[int]]],
        iou_thresh: float = 0.40,
    ) -> List[Tuple[int, int, int, int, float, Optional[int]]]:
        """Perform Non-Maximum Suppression to eliminate double detections."""
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []
        while sorted_boxes:
            cur = sorted_boxes.pop(0)
            keep.append(cur)
            rem = []
            for b in sorted_boxes:
                ix1 = max(cur[0], b[0])
                iy1 = max(cur[1], b[1])
                ix2 = min(cur[2], b[2])
                iy2 = min(cur[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (cur[2] - cur[0]) * (cur[3] - cur[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                iou = inter / union if union > 0 else 0
                if iou < iou_thresh:
                    rem.append(b)
            sorted_boxes = rem
        return keep

    def _draw_annotations(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int, float, Optional[int]]],
    ) -> np.ndarray:
        """Draw sleek modern bounding boxes with collision-free, readable badges."""
        height, width = frame.shape[:2]
        occupied_badges: List[Tuple[int, int, int, int]] = []

        # Sort by confidence descending so higher confidence badges get priority placement
        sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)

        for x1, y1, x2, y2, score, track_id in sorted_boxes:
            # Primary box border (crisp tech emerald green)
            box_color = (0, 210, 85)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # High-tech corner accents (white brackets)
            c_len = max(6, min(14, (x2 - x1) // 4, (y2 - y1) // 4))
            corner_color = (255, 255, 255)
            cv2.line(frame, (x1, y1), (x1 + c_len, y1), corner_color, 2)
            cv2.line(frame, (x1, y1), (x1, y1 + c_len), corner_color, 2)
            cv2.line(frame, (x2, y1), (x2 - c_len, y1), corner_color, 2)
            cv2.line(frame, (x2, y1), (x2, y1 + c_len), corner_color, 2)
            cv2.line(frame, (x1, y2), (x1 + c_len, y2), corner_color, 2)
            cv2.line(frame, (x1, y2), (x1, y2 - c_len), corner_color, 2)
            cv2.line(frame, (x2, y2), (x2 - c_len, y2), corner_color, 2)
            cv2.line(frame, (x2, y2), (x2, y2 - c_len), corner_color, 2)

            # Build clean unified label
            if track_id is not None:
                label = f"#{track_id} Person {int(score * 100)}%"
            else:
                label = f"Person {int(score * 100)}%"

            font_scale = 0.42
            font_thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            pad_x, pad_y = 5, 4
            bw = tw + pad_x * 2
            bh = th + pad_y * 2

            # Candidate badge placements in priority order:
            candidates = [
                (x1, max(0, y1 - bh - 2), min(width, x1 + bw), y1 - 2),
                (x1, min(height, y1 + 2), min(width, x1 + bw), min(height, y1 + bh + 2)),
                (max(0, x2 - bw), max(0, y1 - bh - 2), x2, y1 - 2),
                (x1, min(height, y2 + 2), min(width, x1 + bw), min(height, y2 + bh + 2)),
            ]

            best_badge = None
            for cand in candidates:
                bx1, by1, bx2, by2 = cand
                if bx1 < 0 or by1 < 0 or bx2 > width or by2 > height:
                    continue
                if not any(self._rects_overlap(cand, prev) for prev in occupied_badges):
                    best_badge = cand
                    break

            if best_badge is None:
                by1 = max(0, y1 - bh - 2)
                best_badge = (x1, by1, min(width, x1 + bw), by1 + bh)

            occupied_badges.append(best_badge)
            bx1, by1, bx2, by2 = best_badge

            # Draw dark badge pill with subtle green border and white anti-aliased text
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 24, 28), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)
            cv2.putText(
                frame,
                label,
                (bx1 + pad_x, by2 - pad_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

        return frame

    def detect_frame(self, frame: np.ndarray, conf: float = 0.35) -> Tuple[np.ndarray, int]:
        """Perform object detection on a single video frame with annotations."""
        height, width = frame.shape[:2]
        dh, dw = max(1, height // 4), max(1, width // 4)
        if self.density_map is None or self.density_map.shape != (dh, dw):
            self.density_map = np.zeros((dh, dw), dtype=np.float32)

        use_half = (self.use_fp16 and self.device == "cuda")
        t_start = time.perf_counter()
        with torch.inference_mode():
            results = self.model(
                frame,
                conf=conf,
                imgsz=self.inference_imgsz,
                half=use_half,
                verbose=False,
            )
        self.detection_latency_ms = (time.perf_counter() - t_start) * 1000.0
        raw_boxes = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes_data = result.boxes.data.cpu().numpy()
            for box in boxes_data:
                x1, y1, x2, y2, score, class_id = box[:6]
                if int(class_id) == 0:
                    w, h = x2 - x1, y2 - y1
                    if w < 20 or h < 35:
                        continue
                    if (x1 < 4 or x2 > width - 4) and w < 28:
                        continue
                    raw_boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), None))
        del results

        clean_boxes = self._nms(raw_boxes, iou_thresh=0.40)
        self.current_people_count = len(clean_boxes)

        for x1, y1, x2, y2, _, _ in clean_boxes:
            cx, cy = int((x1 + x2) / 8), int((y1 + y2) / 8)
            cv2.circle(self.density_map, (cx, cy), 9, (1.0,), thickness=-1)

        frame = self._draw_annotations(frame, clean_boxes)

        # Spatial tracks extraction for zones and pose
        active_tracks = [
            (tid if tid is not None else (i + 1), x1, y1, x2, y2)
            for i, (x1, y1, x2, y2, score, tid) in enumerate(clean_boxes)
        ]

        if self.enable_zones:
            self.zone_tracker.update(active_tracks, width, height)
            frame = self.zone_tracker.draw_zones(frame)

        if self.enable_pose:
            skeletons, _ = self.pose_engine.process_frame(
                frame,
                tracks=active_tracks,
                imgsz=self.inference_imgsz,
                half=self.use_fp16,
            )
            frame = self.pose_engine.draw_skeletons(frame, skeletons)
        else:
            skeletons = []

        # Apply real-time face privacy anonymization
        if self.privacy_manager.enabled:
            kpts_list = [s.get("keypoints") for s in skeletons if isinstance(s, dict) and "keypoints" in s]
            frame = self.privacy_manager.apply_face_blur(
                frame,
                boxes=[(b[0], b[1], b[2], b[3]) for b in clean_boxes],
                keypoints=kpts_list,
                in_place=True,
            )

        if self.enable_heat_map:
            frame = self._apply_heatmap(frame)
        elif self.density_map is not None:
            self.density_map *= 0.90

        # Trigger sustained alert evaluation
        self.check_and_trigger_alert(frame, self.current_people_count)

        self._publish_frame(frame)
        return frame, self.current_people_count

    def process_video(self, input_path: str, output_path: str, loop: bool = True) -> None:
        """Process video file with YOLO26 detection, tracking, and optional heatmap."""
        print(f"[INFO] Starting video processing: {input_path}")
        self.video_processing_active = True
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {input_path}")
            self.video_processing_active = False
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        frame_duration = 1.0 / fps if fps > 0 else 0.04
        dh, dw = max(1, height // 4), max(1, width // 4)
        self.density_map = np.zeros((dh, dw), dtype=np.float32)
        save_to_disk = (not loop) and bool(output_path)
        out = None
        if save_to_disk:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while self.video_processing_active:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.reset_tracker()
                    continue
                else:
                    break

            # Run tracking with FP16 and latency monitoring
            raw_boxes = []
            use_half = (self.use_fp16 and self.device == "cuda")
            t_det = time.perf_counter()
            try:
                with torch.inference_mode():
                    track_results = self.model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=0.35,
                        imgsz=self.inference_imgsz,
                        half=use_half,
                        verbose=False,
                    )
                for res in track_results:
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    coords_np = res.boxes.xyxy.cpu().numpy()
                    confs_np = res.boxes.conf.cpu().numpy()
                    clses_np = res.boxes.cls.cpu().numpy().astype(int)
                    ids_np = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

                    for i in range(len(coords_np)):
                        if clses_np[i] == 0:
                            x1, y1, x2, y2 = coords_np[i]
                            w, h = x2 - x1, y2 - y1
                            if w < 20 or h < 35:
                                continue
                            if (x1 < 4 or x2 > width - 4) and w < 28:
                                continue
                            score = float(confs_np[i])
                            tid = int(ids_np[i]) if ids_np is not None else None
                            raw_boxes.append((int(x1), int(y1), int(x2), int(y2), score, tid))
                del track_results
            except Exception:
                # Fallback to standard inference if tracking encounters an error
                with torch.inference_mode():
                    res = self.model(
                        frame,
                        conf=0.35,
                        imgsz=self.inference_imgsz,
                        half=use_half,
                        verbose=False,
                    )
                for b in res[0].boxes.data.cpu().numpy():
                    if int(b[5]) == 0:
                        x1, y1, x2, y2, score, _ = b
                        w, h = x2 - x1, y2 - y1
                        if w < 20 or h < 35:
                            continue
                        if (x1 < 4 or x2 > width - 4) and w < 28:
                            continue
                        raw_boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), None))
                del res

            self.detection_latency_ms = (time.perf_counter() - t_det) * 1000.0
            clean_boxes = self._nms(raw_boxes, iou_thresh=0.40)
            self.current_people_count = len(clean_boxes)

            # Update density points on quarter-resolution grid
            for x1, y1, x2, y2, _, _ in clean_boxes:
                cx, cy = int((x1 + x2) / 8), int((y1 + y2) / 8)
                cv2.circle(self.density_map, (cx, cy), 9, (1.0,), thickness=-1)

            # Draw unified sleek annotations
            frame = self._draw_annotations(frame, clean_boxes)

            # Spatial tracks extraction for zones and pose
            active_tracks = [
                (tid if tid is not None else (i + 1), x1, y1, x2, y2)
                for i, (x1, y1, x2, y2, score, tid) in enumerate(clean_boxes)
            ]

            # Update polygonal security zones
            if self.enable_zones:
                breached_zones = self.zone_tracker.update(active_tracks, width, height)
                frame = self.zone_tracker.draw_zones(frame)
                for bz in breached_zones:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    snap_name = f"zone_{bz['id']}_{timestamp_str}.jpg"
                    snap_path = os.path.join(SNAPSHOT_FOLDER, snap_name)
                    cv2.imwrite(snap_path, frame)
                    incident = {
                        "id": f"zone_{int(time.time())}_{bz['id']}",
                        "event_type": "zone_breach",
                        "zone_name": bz["name"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "current_occupancy": bz["current_occupancy"],
                        "max_occupancy": bz["max_occupancy"],
                        "snapshot_filename": snap_name,
                        "snapshot_path": snap_path,
                        "acknowledged": False,
                    }
                    for zcb in self.zone_callbacks:
                        try:
                            zcb(incident)
                        except Exception as exc:
                            print(f"[WARN] Zone callback error: {exc}")

            # Run pose intelligence and draw skeletons
            if self.enable_pose:
                skeletons, new_pose_events = self.pose_engine.process_frame(
                    frame,
                    tracks=active_tracks,
                    imgsz=self.inference_imgsz,
                    half=self.use_fp16,
                )
                frame = self.pose_engine.draw_skeletons(frame, skeletons)
                for pe in new_pose_events:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    snap_name = f"pose_{pe['event_type']}_{timestamp_str}.jpg"
                    snap_path = os.path.join(SNAPSHOT_FOLDER, snap_name)
                    cv2.imwrite(snap_path, frame)
                    incident = {
                        "id": f"pose_{int(pe['timestamp'])}_{pe['event_type']}",
                        "event_type": pe["event_type"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "confidence": pe["confidence"],
                        "details": pe.get("details", {}),
                        "zone_name": "Active Sector",
                        "snapshot_filename": snap_name,
                        "snapshot_path": snap_path,
                        "acknowledged": False,
                    }
                    for pcb in self.pose_callbacks:
                        try:
                            pcb(incident)
                        except Exception as exc:
                            print(f"[WARN] Pose callback error: {exc}")

            # Apply heatmap overlay when toggled
            if self.enable_heat_map:
                frame = self._apply_heatmap(frame)
            elif self.density_map is not None:
                self.density_map *= 0.90

            # Trigger sustained alert evaluation
            self.check_and_trigger_alert(frame, self.current_people_count)

            # Apply real-time face privacy anonymization
            if self.privacy_manager.enabled:
                kpts_list = [s.get("keypoints") for s in skeletons if isinstance(s, dict) and "keypoints" in s] if self.enable_pose else []
                frame = self.privacy_manager.apply_face_blur(
                    frame,
                    boxes=[(b[0], b[1], b[2], b[3]) for b in clean_boxes],
                    keypoints=kpts_list,
                    in_place=True,
                )

            if out is not None:
                out.write(frame)

            self._publish_frame(frame)

            elapsed = time.time() - loop_start
            if elapsed > 0:
                self.fps_telemetry = (self.fps_telemetry * 0.85) + ((1.0 / elapsed) * 0.15)
            sleep_time = frame_duration - elapsed
            if sleep_time > 0.002:
                time.sleep(sleep_time)

            if self.frame_id % 300 == 0:
                import gc
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        cap.release()
        if out is not None:
            out.release()
        self.video_processing_active = False
        print(f"[INFO] Video processing finished: {output_path}")

    def start_batch_processing(self, input_path: str, output_path: str) -> None:
        """Start non-blocking batch video processing with real-time frame streaming and progress."""
        with self.batch_thread_lock:
            if self.batch_thread is not None and self.batch_thread.is_alive():
                self.batch_job["active"] = False
                self.batch_thread.join(timeout=2.0)

            self.batch_job = {
                "active": True,
                "input_file": os.path.basename(input_path),
                "output_file": os.path.basename(output_path),
                "current_frame": 0,
                "total_frames": 0,
                "percent": 0.0,
                "fps": 0.0,
                "eta_seconds": 0,
                "is_complete": False,
                "error": None,
                "peak_count": 0,
                "avg_count": 0.0,
                "total_breaches": 0,
                "current_count": 0,
            }
            self.batch_thread = threading.Thread(
                target=self._run_batch_processing,
                args=(input_path, output_path),
                daemon=True,
            )
            self.batch_thread.start()

    @property
    def batch_processing_active(self) -> bool:
        """Check whether batch video processing is actively running in background."""
        return bool(
            self.batch_thread is not None
            and self.batch_thread.is_alive()
            and self.batch_job.get("active", False)
        )

    def get_batch_status(self) -> Dict[str, Any]:
        """Return a snapshot of current batch processing progress and analytics."""
        with self.batch_thread_lock:
            return dict(self.batch_job)


    def _run_batch_processing(self, input_path: str, output_path: str) -> None:
        """Process video frame-by-frame, write annotated MP4, and remux to standard web H.264."""
        print(f"[INFO] Initiating batch processing on: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.batch_job["active"] = False
            self.batch_job["error"] = f"Could not open input video: {input_path}"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        self.batch_job["total_frames"] = total_frames
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_raw_path = output_path + ".raw.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))

        frame_counts = []
        breaches_count = 0
        sustained_counter = 0
        start_time = time.time()
        frame_idx = 0

        try:
            while self.batch_job["active"]:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                raw_boxes = []
                try:
                    with torch.inference_mode():
                        track_results = self.model.track(
                            frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=0.35,
                            imgsz=self.inference_imgsz,
                            verbose=False,
                        )
                    for res in track_results:
                        if res.boxes is None or len(res.boxes) == 0:
                            continue
                        coords_np = res.boxes.xyxy.cpu().numpy()
                        confs_np = res.boxes.conf.cpu().numpy()
                        clses_np = res.boxes.cls.cpu().numpy().astype(int)
                        ids_np = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else None

                        for i in range(len(coords_np)):
                            if clses_np[i] == 0:
                                x1, y1, x2, y2 = coords_np[i]
                                w, h = x2 - x1, y2 - y1
                                if w < 20 or h < 35:
                                    continue
                                if (x1 < 4 or x2 > width - 4) and w < 28:
                                    continue
                                score = float(confs_np[i])
                                tid = int(ids_np[i]) if ids_np is not None else None
                                raw_boxes.append((int(x1), int(y1), int(x2), int(y2), score, tid))
                    del track_results
                except Exception:
                    with torch.inference_mode():
                        res = self.model(frame, conf=0.35, imgsz=self.inference_imgsz, verbose=False)
                    for b in res[0].boxes.data.cpu().numpy():
                        if int(b[5]) == 0:
                            x1, y1, x2, y2, score, _ = b
                            w, h = x2 - x1, y2 - y1
                            if w < 20 or h < 35:
                                continue
                            if (x1 < 4 or x2 > width - 4) and w < 28:
                                continue
                            raw_boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), None))
                    del res

                clean_boxes = self._nms(raw_boxes, iou_thresh=0.40)
                count = len(clean_boxes)
                frame_counts.append(count)

                # Overcrowding tracking
                if count >= self.alert_threshold:
                    sustained_counter += 1
                    if sustained_counter == int(fps * self.sustained_duration):
                        breaches_count += 1
                else:
                    sustained_counter = 0

                # Draw annotations
                annotated_frame = self._draw_annotations(frame, clean_boxes)

                if self.privacy_manager.enabled:
                    annotated_frame = self.privacy_manager.apply_face_blur(
                        annotated_frame,
                        boxes=[(b[0], b[1], b[2], b[3]) for b in clean_boxes],
                        in_place=True,
                    )

                out.write(annotated_frame)

                # Update live streaming frame for frontend monitor with buffer reuse
                with self.batch_frame_lock:
                    if self.batch_latest_frame is None or self.batch_latest_frame.shape != annotated_frame.shape:
                        self.batch_latest_frame = annotated_frame.copy()
                    else:
                        np.copyto(self.batch_latest_frame, annotated_frame)
                    self.batch_frame_id += 1

                # Update progress
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed if elapsed > 0 else fps
                remaining_frames = max(0, total_frames - frame_idx)
                eta = int(remaining_frames / current_fps) if current_fps > 0 else 0
                pct = round(min(100.0, (frame_idx / max(1, total_frames)) * 100.0), 1)

                self.batch_job["current_frame"] = frame_idx
                self.batch_job["percent"] = pct
                self.batch_job["fps"] = round(current_fps, 1)
                self.batch_job["eta_seconds"] = eta
                self.batch_job["current_count"] = count
                self.batch_job["peak_count"] = max(self.batch_job["peak_count"], count)
                self.batch_job["total_breaches"] = breaches_count

            cap.release()
            out.release()

            if self.batch_job["active"]:
                # Convert raw opencv mp4v to web-standard H.264 MP4 with faststart
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"[INFO] Transcoding {temp_raw_path} to web H.264 {output_path} via FFmpeg...")
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i", temp_raw_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-movflags", "+faststart",
                    output_path,
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(temp_raw_path):
                    os.remove(temp_raw_path)

                avg = round(float(np.mean(frame_counts)), 1) if frame_counts else 0.0
                self.batch_job["percent"] = 100.0
                self.batch_job["is_complete"] = True
                self.batch_job["active"] = False
                self.batch_job["avg_count"] = avg
                print(f"[SUCCESS] Batch video processing complete: {output_path} (avg={avg}, peak={self.batch_job['peak_count']})")

        except Exception as exc:
            print(f"[ERROR] Batch video processing failed: {exc}")
            self.batch_job["active"] = False
            self.batch_job["error"] = str(exc)
            if cap.isOpened():
                cap.release()
            out.release()


if __name__ == "__main__":
    inference_engine = YOLOInference()
    print("[INFO] YOLOInference initialized successfully.")
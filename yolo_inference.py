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
        self.inference_imgsz: int = 640

        # Overcrowding alert parameters
        self.alert_threshold: int = 15
        self.sustained_duration: float = 3.0
        self.cooldown_duration: float = 30.0
        self.breach_start_time: Optional[float] = None
        self.last_alert_time: float = 0.0
        self.active_alert_triggered: bool = False
        self.alert_callbacks: List[Any] = []

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
            if self.video_thread is not None and self.video_thread.is_alive():
                self.video_thread.join(timeout=2.0)
            self.video_thread = None
            self.current_video_path = None

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

        blurred = cv2.GaussianBlur(self.density_map, (65, 65), 0)
        max_val = float(blurred.max())
        if max_val > 1e-4:
            norm_map = np.uint8(255 * (blurred / max_val))
            heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            mask = norm_map > 15
            annotated = frame.copy()
            annotated[mask] = cv2.addWeighted(frame[mask], 0.45, heatmap[mask], 0.55, 0)
            self.density_map *= 0.95
            return annotated

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
        if self.density_map is None or self.density_map.shape != (height, width):
            self.density_map = np.zeros((height, width), dtype=np.float32)

        with torch.inference_mode():
            results = self.model(frame, conf=conf, imgsz=self.inference_imgsz, verbose=False)
        raw_boxes = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, class_id = box
                if int(class_id) == 0:
                    w, h = x2 - x1, y2 - y1
                    if w < 20 or h < 35:
                        continue
                    if (x1 < 4 or x2 > width - 4) and w < 28:
                        continue
                    raw_boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), None))

        clean_boxes = self._nms(raw_boxes, iou_thresh=0.40)
        self.current_people_count = len(clean_boxes)

        for x1, y1, x2, y2, _, _ in clean_boxes:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(self.density_map, (cx, cy), 35, (1.0,), thickness=-1)

        frame = self._draw_annotations(frame, clean_boxes)

        if self.enable_heat_map:
            frame = self._apply_heatmap(frame)
        elif self.density_map is not None:
            self.density_map *= 0.90

        # Trigger sustained alert evaluation
        self.check_and_trigger_alert(frame, self.current_people_count)

        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.frame_id += 1

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
        self.density_map = np.zeros((height, width), dtype=np.float32)
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

            # Run tracking
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
                    if res.boxes is None:
                        continue
                    for i, b in enumerate(res.boxes):
                        cls_id = int(b.cls[0].cpu().numpy())
                        if cls_id == 0:
                            coords = b.xyxy[0].cpu().numpy()
                            score = float(b.conf[0].cpu().numpy())
                            tid = int(b.id[0].cpu().numpy()) if b.id is not None else None
                            x1, y1, x2, y2 = coords
                            w, h = x2 - x1, y2 - y1
                            if w < 20 or h < 35:
                                continue
                            if (x1 < 4 or x2 > width - 4) and w < 28:
                                continue
                            raw_boxes.append((int(x1), int(y1), int(x2), int(y2), score, tid))
            except Exception:
                # Fallback to standard inference if tracking encounters an error
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

            clean_boxes = self._nms(raw_boxes, iou_thresh=0.40)
            self.current_people_count = len(clean_boxes)

            # Update density points
            for x1, y1, x2, y2, _, _ in clean_boxes:
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(self.density_map, (cx, cy), 35, (1.0,), thickness=-1)

            # Draw unified sleek annotations
            frame = self._draw_annotations(frame, clean_boxes)

            # Apply heatmap overlay when toggled
            if self.enable_heat_map:
                frame = self._apply_heatmap(frame)
            elif self.density_map is not None:
                self.density_map *= 0.90

            # Trigger sustained alert evaluation
            self.check_and_trigger_alert(frame, self.current_people_count)

            out.write(frame)
            with self.frame_lock:
                self.latest_frame = frame.copy()
                self.frame_id += 1

            elapsed = time.time() - loop_start
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

        cap.release()
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
                        if res.boxes is None:
                            continue
                        for i, b in enumerate(res.boxes):
                            cls_id = int(b.cls[0].cpu().numpy())
                            if cls_id == 0:
                                coords = b.xyxy[0].cpu().numpy()
                                score = float(b.conf[0].cpu().numpy())
                                tid = int(b.id[0].cpu().numpy()) if b.id is not None else None
                                x1, y1, x2, y2 = coords
                                w, h = x2 - x1, y2 - y1
                                if w < 20 or h < 35:
                                    continue
                                if (x1 < 4 or x2 > width - 4) and w < 28:
                                    continue
                                raw_boxes.append((int(x1), int(y1), int(x2), int(y2), score, tid))
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
                out.write(annotated_frame)

                # Update live streaming frame for frontend monitor
                with self.batch_frame_lock:
                    self.batch_latest_frame = annotated_frame.copy()
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
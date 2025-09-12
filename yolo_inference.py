import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CROWD_THRESHOLD = 30  # For Discord alerts (overall count threshold)
ALERT_DELAY = 15  # Minimum seconds between alerts
MOTION_THRESHOLD = 5000  # For background subtraction
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1334259674580516979/pH92tTp_wnYG2a5j6KNgPHHHmbQRC7Hs8L01KANDKiQrw7iE4jPa6iuWqauLY1G6DqoD"
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

ROLLING_WINDOW = 30  # For smoothing crowd count
ENABLE_CROWD_PREDICTION = True
PREDICT_FRAMES_AHEAD = 30  # Not used in grid analysis

# --- Grid-based crowd analysis settings ---
ENABLE_GRID_ANALYSIS = True
NUM_GRID_ROWS = 6  # Grid rows
NUM_GRID_COLS = 6  # Grid cols
GRID_CELL_THRESHOLD = 3  # If cell count >= this, mark it

# DeepSort tracker
tracker = DeepSort(max_age=30, embedder="mobilenet")


class YOLOInference:
    def __init__(self, model_path="runs/detect/yolo11x_head12/weights/best.pt"):
        """Initialize YOLOv11 model and other settings."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = os.path.basename(model_path) if os.path.isabs(model_path) else model_path
        self.model = YOLO(model_path).to(self.device)
        print(f"[INFO] YOLOv11 model loaded on device: {self.device}")

        self.last_alert_time = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.density_map = None
        self.latest_frame = None

        self.frame_indices = []
        self.crowd_counts = []

        # Heatmap toggle
        self.enable_heat_map = False
        self.density_map = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Grid-based zooming
        self.zoom_row = None
        self.zoom_col = None

        # Store last processed overlay frame
        self.last_processed_overlay = None

        # Heatmap
        self.enable_heat_map = False
        self.density_map = None

        # Threading lock
        self.frame_lock = threading.Lock()

    def set_heatmap_enabled(self, state: bool):
        self.enable_heat_map = state
        if not state: # Reset density map when turned off
            if self.density_map is not None:
                self.density_map.fill(0)
        print(f"[INFO] Heatmap enabled: {self.enable_heat_map}")

    def set_zoom_cell(self, row: int, col: int):
        if row < 0 or col < 0:
            self.zoom_row, self.zoom_col = None, None
            print("[INFO] Zoom reset.")
        else:
            self.zoom_row, self.zoom_col = row, col
            print(f"[INFO] Zoom cell set to row={row}, col={col}")

    def get_zoomed_subimage(self):
        """Returns a magnified subimage of the latest frame."""
        with self.frame_lock:
            if self.latest_frame is None or self.zoom_row is None or self.zoom_col is None:
                return np.zeros((240, 320, 3), dtype=np.uint8) # Return a blank image
        
        height, width = self.latest_frame.shape[:2]
        cell_height = height // self.NUM_GRID_ROWS
        cell_width = width // self.NUM_GRID_COLS
        
        start_y, end_y = self.zoom_row * cell_height, (self.zoom_row + 1) * cell_height
        start_x, end_x = self.zoom_col * cell_width, (self.zoom_col + 1) * cell_width
        
        subimg = self.latest_frame[start_y:end_y, start_x:end_x]
        return cv2.resize(subimg, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)


    def _draw_overlays(self, frame, people_count):
        """Helper function to draw stats and optional heatmap/grid."""
        # Draw heatmap if enabled
        if self.enable_heat_map and self.density_map is not None:
            heatmap_viz = cv2.applyColorMap(
                cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            frame = cv2.addWeighted(frame, 0.6, heatmap_viz, 0.4, 0)
            # Decay the heatmap over time
            self.density_map *= 0.97
            
        # Draw stats
        cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame


    def set_zoom_cell(self, row: int, col: int):
        """Set the grid cell (row, col) to magnify; if negative, reset."""
        if row < 0 or col < 0:
            self.zoom_row = None
            self.zoom_col = None
            print("[INFO] Zoom reset (no cell selected).")
        else:
            self.zoom_row = row
            self.zoom_col = col
            print(f"[INFO] Zoom cell set to row={row}, col={col}")
    
    def get_zoomed_subimage(self):
        """Returns a zoomed subimage of the latest frame based on zoom_row and zoom_col."""
        if self.latest_frame is None or self.zoom_row is None or self.zoom_col is None:
            return None
            
        height, width = self.latest_frame.shape[:2]
        cell_height = height // NUM_GRID_ROWS
        cell_width = width // NUM_GRID_COLS
        
        # Calculate cell boundaries
        start_y = self.zoom_row * cell_height
        end_y = start_y + cell_height
        start_x = self.zoom_col * cell_width
        end_x = start_x + cell_width
        
        # Extract the subimage
        subimg = self.latest_frame[start_y:end_y, start_x:end_x].copy()
        
        # Resize to make it larger (optional)
        subimg = cv2.resize(subimg, (width // 2, height // 2))
        
        return subimg

    def process_video(self, input_path, output_path):
        """Processes the input video using YOLOv11 and DeepSort."""
        print(f"[INFO] Opening video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25
            print(f"[WARN] Invalid FPS detected. Defaulting to: {fps} fps")

        self.density_map = np.zeros((height, width), dtype=np.float32)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] No more frames. Processing complete.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[DEBUG] Processed {frame_count} frames...")

            # Run YOLOv11 detection
            results = self.model(frame, conf=0.3, device=self.device)
            detections = []

            for result in results:
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, class_id = box
                    if int(class_id) == 0:  # Person class
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, "person"))

            # Update DeepSort Tracker
            tracks = tracker.update_tracks(detections, frame=frame)

            # Draw bounding boxes & track IDs
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {track.track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Update density map
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(self.density_map, (cx, cy), 25, (1.0,), thickness=-1)

            # Overlay density heatmap
            if self.enable_heat_map:
                heatmap = cv2.applyColorMap(cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

            # Write frame
            out.write(frame)
            self.latest_frame = frame.copy()

        cap.release()
        out.release()
        print(f"[INFO] Finished processing. Output saved to: {output_path}")


if __name__ == "__main__":
    yolo_infer = YOLOInference("yolo11x.pt")
    yolo_infer.process_video("input_video.mp4", "output_video.mp4")
import os
import threading
import time
from typing import Annotated, Any, Dict, List, Optional

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Path, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from yolo_inference import SNAPSHOT_FOLDER, YOLOInference

app = FastAPI(title="Crowd Management System (CMS) API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "uploads")
PROCESSED_FOLDER = os.path.join(os.getcwd(), "static", "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates engine
templates = Jinja2Templates(directory="templates")

# Initialize shared YOLO26 inference engine with fallback resolution
yolo_infer = YOLOInference()

# Global state for live camera streaming
camera_active = False
camera_thread: threading.Thread | None = None
current_camera_frame: np.ndarray | None = None
frame_lock = threading.Lock()
camera_stats: Dict[str, Any] = {
    "people_count": 0,
    "density_level": "None",
    "alert_status": "Safe",
    "alert_threshold": yolo_infer.alert_threshold,
    "active_alert_triggered": False,
    "unacknowledged_incidents": 0,
    "inference_imgsz": yolo_infer.inference_imgsz,
    "fps": 0,
}

# Incident audit state and webhook configuration
incidents_lock = threading.Lock()
security_incidents: List[Dict[str, Any]] = []
external_webhook_url: Optional[str] = None


class ThresholdUpdateRequest(BaseModel):
    threshold: int = Field(ge=1, le=500, description="Minimum headcount to trigger an alert")


class WebhookUpdateRequest(BaseModel):
    webhook_url: str = Field(description="URL for external security webhook notifications")


class ImgszUpdateRequest(BaseModel):
    imgsz: int = Field(description="Inference image resolution: 640 or 1024")


def dispatch_webhook_notification(incident: Dict[str, Any], webhook_url: str) -> None:
    """Send non-blocking webhook notification to external security dispatch systems."""
    try:
        payload = {
            "event": "CROWD_OVERCROWDING_ALERT",
            "incident_id": incident["id"],
            "timestamp": incident["timestamp"],
            "people_count": incident["people_count"],
            "threshold": incident["threshold"],
            "snapshot_filename": incident["snapshot_filename"],
            "message": (
                f"Security Alert: Crowd headcount ({incident['people_count']}) has exceeded "
                f"the safety threshold ({incident['threshold']}) for sustained period."
            ),
        }
        with httpx.Client(timeout=4.0) as client:
            client.post(webhook_url, json=payload)
    except Exception as exc:
        print(f"[WARN] Failed to dispatch security webhook: {exc}")


def handle_incident_alert(incident: Dict[str, Any]) -> None:
    """Record incident in local audit log and trigger async webhook dispatch."""
    global external_webhook_url
    incident_record = dict(incident)
    incident_record["snapshot_url"] = f"/api/snapshots/{incident['snapshot_filename']}"

    with incidents_lock:
        security_incidents.insert(0, incident_record)
        if len(security_incidents) > 50:
            security_incidents.pop()

    if external_webhook_url:
        thread = threading.Thread(
            target=dispatch_webhook_notification,
            args=(incident_record, external_webhook_url),
            daemon=True,
        )
        thread.start()


# Register the callback with the inference engine
yolo_infer.register_alert_callback(handle_incident_alert)


def get_density_level(people_count: int, threshold: int = 15) -> str:
    """Classify crowd density level based on headcount and dynamic threshold."""
    if people_count == 0:
        return "None"
    elif people_count <= max(2, int(threshold * 0.25)):
        return "Low"
    elif people_count <= max(5, int(threshold * 0.60)):
        return "Medium"
    elif people_count < threshold:
        return "High"
    return "Critical"


def get_alert_status(people_count: int, threshold: int = 15, active_alert: bool = False) -> str:
    """Classify emergency alert status based on headcount, threshold, and active alert state."""
    if active_alert or people_count >= threshold:
        return "Alert"
    elif people_count >= max(3, int(threshold * 0.60)):
        return "Caution"
    return "Safe"


def run_camera_processing() -> None:
    """Background worker for live webcam capture and YOLO26 inference."""
    global current_camera_frame, camera_stats, camera_active, frame_lock

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera device 0.")
        camera_active = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[INFO] Live camera processing started with YOLO26.")
    fps_counter = 0
    fps_start_time = time.time()
    pulse_timer = 0

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        try:
            height, width = frame.shape[:2]
            annotated_frame, people_count = yolo_infer.detect_frame(frame, conf=0.35)

            # Live recording pulsing indicator
            pulse_timer += 1
            pulse_intensity = int(128 + 127 * np.sin(pulse_timer * 0.2))
            cv2.circle(annotated_frame, (20, height - 20), 10, (0, 0, pulse_intensity), -1)
            cv2.circle(annotated_frame, (20, height - 20), 10, (255, 255, 255), 2)
            cv2.putText(
                annotated_frame,
                "LIVE",
                (38, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Update stats
            camera_stats["people_count"] = people_count
            camera_stats["density_level"] = get_density_level(people_count)
            camera_stats["alert_status"] = get_alert_status(people_count)

            # Calculate FPS
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                camera_stats["fps"] = int(fps_counter / elapsed)
                fps_counter = 0
                fps_start_time = time.time()

            with frame_lock:
                current_camera_frame = annotated_frame.copy()

        except Exception as exc:
            print(f"[ERROR] Error during frame processing: {exc}")
            with frame_lock:
                current_camera_frame = frame

        time.sleep(0.033)

    cap.release()
    print("[INFO] Live camera processing stopped.")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.api_route("/toggle_heatmap", methods=["GET", "POST"])
def toggle_heatmap() -> Dict[str, Any]:
    new_state = not yolo_infer.enable_heat_map
    yolo_infer.set_heatmap_enabled(new_state)
    camera_stats["heatmap_enabled"] = new_state
    return {"heatmap_enabled": new_state, "status": "success"}


@app.get("/live_camera", response_class=HTMLResponse)
def live_camera(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "live_camera.html")


@app.post("/start_camera")
def start_camera() -> Dict[str, str]:
    global camera_active, camera_thread

    if camera_active:
        return {"status": "Camera already active"}

    camera_active = True
    camera_thread = threading.Thread(target=run_camera_processing, daemon=True)
    camera_thread.start()

    return {"status": "Camera started successfully"}


@app.post("/stop_camera")
def stop_camera() -> Dict[str, str]:
    global camera_active
    camera_active = False
    return {"status": "Camera stopped"}


@app.get("/camera_feed")
def camera_feed() -> StreamingResponse:
    """Stream multipart JPEG frames from live camera feed."""
    def generate():
        global current_camera_frame, frame_lock
        while camera_active:
            with frame_lock:
                if current_camera_frame is not None:
                    ret, buffer = cv2.imencode(
                        ".jpg", current_camera_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
                else:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "Initializing Camera...",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    ret, buffer = cv2.imencode(".jpg", placeholder)
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + buffer.tobytes()
                            + b"\r\n"
                        )
            time.sleep(0.033)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/camera_stats")
def get_camera_stats() -> Dict[str, Any]:
    count = yolo_infer.current_people_count
    threshold = yolo_infer.alert_threshold
    active_alert = yolo_infer.active_alert_triggered
    camera_stats["people_count"] = count
    camera_stats["alert_threshold"] = threshold
    camera_stats["density_level"] = get_density_level(count, threshold)
    camera_stats["alert_status"] = get_alert_status(count, threshold, active_alert)
    camera_stats["heatmap_enabled"] = yolo_infer.enable_heat_map
    camera_stats["active_alert_triggered"] = active_alert
    camera_stats["inference_imgsz"] = yolo_infer.inference_imgsz
    with incidents_lock:
        camera_stats["unacknowledged_incidents"] = sum(
            1 for inc in security_incidents if not inc.get("acknowledged", False)
        )
    return camera_stats


@app.post("/toggle_camera_heatmap")
def toggle_camera_heatmap() -> Dict[str, bool]:
    new_state = not yolo_infer.enable_heat_map
    yolo_infer.set_heatmap_enabled(new_state)
    camera_stats["heatmap_enabled"] = new_state
    return {"heatmap_enabled": new_state}


@app.post("/upload")
def upload(video: Annotated[UploadFile, File()]) -> Dict[str, Any]:
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    safe_name = os.path.basename(video.filename)
    video_path = os.path.join(UPLOAD_FOLDER, safe_name)
    with open(video_path, "wb") as buffer:
        content = video.file.read()
        buffer.write(content)

    processed_filename = f"processed_{safe_name}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

    # Start non-blocking batch video processing with real-time frame streaming and H.264 encoding
    yolo_infer.start_batch_processing(video_path, processed_path)

    return {
        "status": "success",
        "message": "Video uploaded and processing initiated",
        "filename": safe_name,
        "processed_file": processed_filename,
        "redirect_url": "/video_studio",
    }


@app.get("/live_preview", response_class=HTMLResponse)
def live_preview(request: Request) -> HTMLResponse:
    input_video_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    output_video_path = os.path.join(PROCESSED_FOLDER, "output.mp4")

    if os.path.isfile(input_video_path):
        yolo_infer.ensure_video_processing(input_video_path, output_video_path, loop=True)

    return templates.TemplateResponse(request, "live_preview.html")


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    def generate():
        last_sent_id = -1
        while True:
            with yolo_infer.frame_lock:
                current_id = yolo_infer.frame_id
                frame_to_send = (
                    yolo_infer.latest_frame.copy()
                    if yolo_infer.latest_frame is not None
                    else None
                )

            if frame_to_send is None or current_id == last_sent_id:
                time.sleep(0.01)
                continue

            last_sent_id = current_id
            ret, buffer = cv2.imencode(
                ".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/set_zoom")
def set_zoom(
    row: Annotated[int, Query()] = -1,
    col: Annotated[int, Query()] = -1,
) -> Dict[str, str]:
    yolo_infer.set_zoom_cell(row, col)
    return {"status": "OK"}


@app.get("/zoom_feed")
def zoom_feed() -> StreamingResponse:
    def generate():
        while True:
            subimg = yolo_infer.get_zoomed_subimage()
            if subimg is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode(".jpg", subimg)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(0.033)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/process_video")
def process_video_route() -> RedirectResponse:
    input_video_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    output_video_path = os.path.join(PROCESSED_FOLDER, "output.mp4")

    if os.path.isfile(input_video_path):
        yolo_infer.start_video_processing(input_video_path, output_video_path, loop=True)

    return RedirectResponse(url="/live_preview", status_code=302)


# Security and Alert Settings Endpoints
@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    """Retrieve current security and detection parameters."""
    return {
        "threshold": yolo_infer.alert_threshold,
        "sustained_duration": yolo_infer.sustained_duration,
        "cooldown_duration": yolo_infer.cooldown_duration,
        "webhook_url": external_webhook_url,
        "inference_imgsz": yolo_infer.inference_imgsz,
    }


@app.post("/api/settings/threshold")
def update_threshold(req: ThresholdUpdateRequest) -> Dict[str, Any]:
    """Update crowd headcount alert threshold."""
    yolo_infer.set_alert_threshold(req.threshold)
    camera_stats["alert_threshold"] = yolo_infer.alert_threshold
    return {"status": "success", "threshold": yolo_infer.alert_threshold}


@app.post("/api/settings/webhook")
def update_webhook(req: WebhookUpdateRequest) -> Dict[str, Any]:
    """Configure external security dispatch webhook URL."""
    global external_webhook_url
    url = req.webhook_url.strip()
    external_webhook_url = url if url else None
    return {"status": "success", "webhook_url": external_webhook_url}


@app.post("/api/settings/imgsz")
def update_imgsz(req: ImgszUpdateRequest) -> Dict[str, Any]:
    """Set detection resolution for small/distant human crowd detection."""
    if req.imgsz not in (640, 1024):
        raise HTTPException(status_code=400, detail="imgsz must be 640 or 1024")
    yolo_infer.set_inference_imgsz(req.imgsz)
    camera_stats["inference_imgsz"] = yolo_infer.inference_imgsz
    return {"status": "success", "inference_imgsz": yolo_infer.inference_imgsz}


@app.get("/api/incidents")
def list_incidents() -> Dict[str, Any]:
    """List recent security incidents with snapshot URLs."""
    with incidents_lock:
        return {"incidents": list(security_incidents)}


@app.post("/api/incidents/{incident_id}/acknowledge")
def acknowledge_incident(
    incident_id: Annotated[str, Path(description="The unique incident ID to acknowledge")]
) -> Dict[str, Any]:
    """Mark an incident as acknowledged by security personnel."""
    with incidents_lock:
        for inc in security_incidents:
            if inc["id"] == incident_id:
                inc["acknowledged"] = True
                return {"status": "success", "incident_id": incident_id, "acknowledged": True}
    raise HTTPException(status_code=404, detail="Incident not found")


@app.get("/api/snapshots/{filename}")
def get_snapshot(
    filename: Annotated[str, Path(description="Snapshot filename to retrieve")]
) -> FileResponse:
    """Serve captured incident snapshot photos securely."""
    # Prevent directory traversal attacks
    safe_filename = os.path.basename(filename)
    snapshot_path = os.path.join(SNAPSHOT_FOLDER, safe_filename)
    if not os.path.isfile(snapshot_path):
        raise HTTPException(status_code=404, detail="Snapshot file not found")
    return FileResponse(snapshot_path, media_type="image/jpeg")


@app.get("/video_studio", response_class=HTMLResponse)
def video_studio(request: Request) -> HTMLResponse:
    """Serve the dedicated Video Processing Studio interface."""
    return templates.TemplateResponse(request, "video_studio.html")


@app.get("/video_processing_feed")
def video_processing_feed() -> StreamingResponse:
    """Stream live annotated frames of the currently processing video."""
    def generate():
        last_sent_id = -1
        while True:
            with yolo_infer.batch_frame_lock:
                current_id = yolo_infer.batch_frame_id
                frame = (
                    yolo_infer.batch_latest_frame.copy()
                    if yolo_infer.batch_latest_frame is not None
                    else None
                )

            if frame is None or current_id == last_sent_id:
                time.sleep(0.02)
                continue

            last_sent_id = current_id
            ret, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/processing_status")
def get_processing_status() -> Dict[str, Any]:
    """Return live status of the batch video processing engine."""
    return yolo_infer.get_batch_status()


@app.get("/api/processed_videos")
def list_processed_videos() -> Dict[str, Any]:
    """List all stored processed videos with file metadata."""
    videos = []
    if os.path.exists(PROCESSED_FOLDER):
        for fname in sorted(os.listdir(PROCESSED_FOLDER), reverse=True):
            if fname.endswith(".mp4") and not fname.endswith(".raw.mp4") and not fname.endswith(".temp.mp4"):
                full_p = os.path.join(PROCESSED_FOLDER, fname)
                sz_mb = round(os.path.getsize(full_p) / (1024 * 1024), 2)
                mtime = os.path.getmtime(full_p)
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                videos.append({
                    "filename": fname,
                    "size_mb": sz_mb,
                    "created_at": timestamp_str,
                    "video_url": f"/processed/{fname}",
                    "download_url": f"/processed/{fname}",
                })
    return {"videos": videos}


@app.get("/processed/{filename}")
def serve_processed_video(
    filename: Annotated[str, Path(description="Processed video filename")]
) -> FileResponse:
    """Serve stored processed MP4 video files with range support for seeking."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(PROCESSED_FOLDER, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Processed video file not found")
    return FileResponse(file_path, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
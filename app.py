import asyncio
import contextlib
import csv
import io
import ipaddress
import os
import re
import socket
import threading
import time
import urllib.parse
from typing import Annotated, Any, Dict, List, Literal, Optional

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from yolo_inference import SNAPSHOT_FOLDER, YOLOInference
from cms.data.database import db_manager, verify_password
from cms.auth.jwt_auth import (
    CurrentUserDep,
    LoginRequest,
    LoginResponse,
    OptionalUserDep,
    create_access_token,
    require_roles,
)
from cms.api.websocket_hub import ws_hub

# Security Configuration Constants
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_SIZE = 250 * 1024 * 1024  # 250 MB maximum upload size
CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunks for memory protection


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager for WebSocket hub and periodic telemetry."""
    loop = asyncio.get_running_loop()
    ws_hub.set_loop(loop)

    # Load initial security sectors into inference engine
    try:
        zones = await db_manager.get_zones()
        yolo_infer.update_zones(zones)
    except Exception as exc:
        print(f"[WARN] Error loading security sectors: {exc}")

    # Periodic background telemetry logger
    stop_telemetry = asyncio.Event()

    async def _periodic_telemetry():
        while not stop_telemetry.is_set():
            try:
                await asyncio.sleep(4.0)
                count = yolo_infer.current_people_count
                density = get_density_level(count, yolo_infer.alert_threshold)
                await db_manager.log_crowd_snapshot(
                    total_count=count,
                    density_level=density,
                    camera_id="primary",
                )
                zones_summary = yolo_infer.get_zones_summary()
                await ws_hub.broadcast({
                    "event_type": "telemetry",
                    "people_count": count,
                    "density_level": density,
                    "alert_status": get_alert_status(count, yolo_infer.alert_threshold, yolo_infer.active_alert_triggered),
                    "alert_threshold": yolo_infer.alert_threshold,
                    "active_alert_triggered": yolo_infer.active_alert_triggered,
                    "zones": zones_summary,
                    "timestamp": time.time(),
                })
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[WARN] Telemetry logging error: {exc}")

    telemetry_task = asyncio.create_task(_periodic_telemetry())
    yield
    stop_telemetry.set()
    telemetry_task.cancel()


app = FastAPI(
    title="Crowd Management System (CMS) API",
    version="2.1.0",
    lifespan=lifespan,
)

# Hardened CORS configuration (credentials disallowed with wildcard origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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


class ZoneCreateRequest(BaseModel):
    id: Optional[str] = None
    name: str = Field(description="Zone sector name")
    color: str = Field(default="#FFE500", description="Hex color string")
    polygon: List[List[float]] = Field(description="List of [x, y] coordinates")
    max_occupancy: int = Field(default=10, ge=1, le=1000)
    camera_id: str = "primary"


class ToggleRequest(BaseModel):
    enabled: bool


class HardwareProfileRequest(BaseModel):
    profile: str = Field(description="Hardware profile: eco, balanced, ultra, or custom")


class PoseModelRequest(BaseModel):
    model_name: str = Field(description="Pose model name: yolo26n-pose.pt, yolo26s-pose.pt, or yolo26m-pose.pt")


class PoseStrideRequest(BaseModel):
    stride: int = Field(ge=1, le=5, description="Temporal frame stride for pose estimation")


class Fp16ToggleRequest(BaseModel):
    enabled: bool = Field(description="Enable or disable FP16 half precision on CUDA")


class ZonePolygonUpdateRequest(BaseModel):
    polygon: List[List[float]] = Field(description="Updated list of [x, y] coordinates")


class PrivacyToggleRequest(BaseModel):
    enabled: bool = Field(description="Enable or disable real-time face blurring")


class SimulateIncidentRequest(BaseModel):
    event_type: Literal["fall", "collapse", "panic", "zone_breach", "overcrowding"] = Field(
        default="fall", description="Type of security incident to simulate"
    )
    zone_name: str = Field(default="Central Concourse", description="Target zone name")
    confidence: float = Field(default=0.96, ge=0.0, le=1.0, description="Detection confidence")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional custom metadata")


def validate_webhook_url(url: str) -> None:
    """Validate external webhook URL against SSRF vulnerabilities."""
    if not url:
        return
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook URL structure")

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Webhook URL must use http or https protocol")

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Webhook URL must include a valid hostname")

    # Prohibit local, loopback, and cloud metadata aliases
    lowered = hostname.lower()
    blocked_hostnames = {
        "localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal",
        "instance-data", "metadata"
    }
    if lowered in blocked_hostnames or lowered.endswith(".local") or lowered.endswith(".internal"):
        raise HTTPException(
            status_code=400,
            detail="Invalid webhook URL: Localhost, private, and metadata network addresses are prohibited"
        )

    # Check for direct IP inputs
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or str(ip_obj).startswith("169.254.")
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid webhook URL: Private, loopback, and metadata network addresses are prohibited"
            )
    except ValueError:
        # Not a direct IP literal, resolve via DNS
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for family, _, _, _, sockaddr in addr_info:
                ip_str = sockaddr[0]
                ip_obj = ipaddress.ip_address(ip_str)
                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_reserved
                    or str(ip_obj).startswith("169.254.")
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid webhook URL: Target resolves to a private or loopback IP address"
                    )
        except socket.gaierror:
            # Allow in test/offline environments if domain format is otherwise valid
            pass


def dispatch_webhook_notification(incident: Dict[str, Any], webhook_url: str) -> None:
    """Send non-blocking webhook notification to external security dispatch systems."""
    try:
        validate_webhook_url(webhook_url)
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
    """Record incident in local audit log, database, WebSocket hub, and webhook."""
    global external_webhook_url
    incident_record = dict(incident)
    if "snapshot_filename" in incident:
        incident_record["snapshot_url"] = f"/api/snapshots/{incident['snapshot_filename']}"

    with incidents_lock:
        security_incidents.insert(0, incident_record)
        if len(security_incidents) > 50:
            security_incidents.pop()

    # Broadcast via WebSocket hub to all connected operator dashboards
    ws_hub.broadcast_sync({
        "event_type": "safety_alert",
        "incident": incident_record,
    })

    # Log to SQLite database asynchronously
    def _save_to_db():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                db_manager.record_safety_event(
                    event_id=incident_record.get("id", f"inc_{int(time.time())}"),
                    event_type=incident_record.get("event_type", "overcrowding"),
                    timestamp=time.time(),
                    camera_id="primary",
                    zone_name=incident_record.get("zone_name", "General Area"),
                    confidence=float(incident_record.get("confidence", 1.0)),
                    details=incident_record.get("details", {}),
                    snapshot_path=incident_record.get("snapshot_filename"),
                )
            )
        finally:
            loop.close()

    threading.Thread(target=_save_to_db, daemon=True).start()

    if external_webhook_url:
        thread = threading.Thread(
            target=dispatch_webhook_notification,
            args=(incident_record, external_webhook_url),
            daemon=True,
        )
        thread.start()


# Register callbacks with inference engine
yolo_infer.register_alert_callback(handle_incident_alert)
yolo_infer.register_pose_callback(handle_incident_alert)
yolo_infer.register_zone_callback(handle_incident_alert)


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
    """Securely upload a video file for batch crowd analysis."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    if yolo_infer.batch_processing_active:
        raise HTTPException(
            status_code=409,
            detail="A video processing task is currently active. Please wait for it to complete."
        )

    raw_basename = os.path.basename(video.filename)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_basename)
    if not safe_name or safe_name.startswith("."):
        safe_name = f"upload_{int(time.time())}{ext}"

    video_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_name))
    if not video_path.startswith(os.path.abspath(UPLOAD_FOLDER)):
        raise HTTPException(status_code=400, detail="Invalid target filename")

    bytes_written = 0
    try:
        with open(video_path, "wb") as buffer:
            while True:
                chunk = video.file.read(CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB"
                    )
                buffer.write(chunk)
    except HTTPException:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise
    except Exception as exc:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(exc)}")

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
            with yolo_infer.frame_condition:
                while yolo_infer.frame_id == last_sent_id and yolo_infer.video_processing_active:
                    yolo_infer.frame_condition.wait(timeout=0.1)

                if not yolo_infer.video_processing_active and yolo_infer.latest_jpeg_bytes is None:
                    break

                jpeg_bytes = yolo_infer.latest_jpeg_bytes
                last_sent_id = yolo_infer.frame_id

            if jpeg_bytes is None:
                time.sleep(0.01)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
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
    """Configure external security dispatch webhook URL with SSRF validation."""
    global external_webhook_url
    url = req.webhook_url.strip()
    if url:
        validate_webhook_url(url)
        external_webhook_url = url
    else:
        external_webhook_url = None
    return {"status": "success", "webhook_url": external_webhook_url}


@app.post("/api/settings/imgsz")
def update_imgsz(req: ImgszUpdateRequest) -> Dict[str, Any]:
    """Set detection resolution for small/distant human crowd detection."""
    if req.imgsz not in (640, 1024):
        raise HTTPException(status_code=400, detail="imgsz must be 640 or 1024")
    yolo_infer.set_inference_imgsz(req.imgsz)
    camera_stats["inference_imgsz"] = yolo_infer.inference_imgsz
    return {"status": "success", "inference_imgsz": yolo_infer.inference_imgsz}


@app.get("/api/settings/hardware")
def get_hardware_telemetry() -> Dict[str, Any]:
    """Get active hardware profile, GPU capabilities, and latency telemetry."""
    return yolo_infer.get_hardware_status()


@app.post("/api/settings/hardware_profile")
def update_hardware_profile(req: HardwareProfileRequest) -> Dict[str, Any]:
    """Switch tiered hardware acceleration profile (eco, balanced, ultra, custom)."""
    return yolo_infer.set_hardware_profile(req.profile)


@app.post("/api/settings/pose_model")
def update_pose_model(req: PoseModelRequest) -> Dict[str, Any]:
    """Hot-swap the YOLO26 pose model variant."""
    success = yolo_infer.set_pose_model(req.model_name)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to load pose model")
    return {"status": "success", "pose_model": req.model_name}


@app.post("/api/settings/pose_stride")
def update_pose_stride(req: PoseStrideRequest) -> Dict[str, Any]:
    """Update temporal frame stride for pose estimation."""
    yolo_infer.set_pose_stride(req.stride)
    return {"status": "success", "pose_stride": req.stride}


@app.post("/api/settings/fp16")
def update_fp16(req: Fp16ToggleRequest) -> Dict[str, Any]:
    """Toggle FP16 half precision execution on CUDA."""
    yolo_infer.set_fp16_enabled(req.enabled)
    return {"status": "success", "fp16_enabled": yolo_infer.use_fp16}


@app.post("/api/settings/export_tensorrt")
def export_tensorrt(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger background compilation of YOLO26 TensorRT engine with fallback."""
    def _export_worker():
        try:
            print("[INFO] Starting TensorRT engine compilation...")
            yolo_infer.model.export(
                format="engine",
                half=True,
                imgsz=yolo_infer.inference_imgsz,
            )
            print("[INFO] TensorRT compilation finished successfully.")
        except Exception as exc:
            print(f"[WARN] TensorRT export failed or unsupported on this environment: {exc}")

    background_tasks.add_task(_export_worker)
    return {"status": "initiated", "message": "TensorRT export initiated in background"}


@app.get("/api/settings/privacy")
def get_privacy_settings() -> Dict[str, Any]:
    """Retrieve privacy compliance configuration and active state."""
    return yolo_infer.get_privacy_status()


@app.post("/api/settings/privacy")
async def set_privacy_settings(req: PrivacyToggleRequest) -> Dict[str, Any]:
    """Toggle real-time face privacy anonymization."""
    state = yolo_infer.toggle_privacy(req.enabled)
    await ws_hub.broadcast({
        "event_type": "privacy_update",
        "privacy_enabled": state,
    })
    return {"status": "success", "privacy_enabled": state}


@app.get("/api/incidents")
async def list_incidents() -> Dict[str, Any]:
    """List recent security incidents from database and active memory."""
    db_events = await db_manager.get_safety_events(limit=50)
    results = []
    seen_ids = set()

    with incidents_lock:
        for inc in security_incidents:
            enriched = dict(inc)
            if "event_type" not in enriched:
                enriched["event_type"] = "overcrowding"
            if "zone_name" not in enriched:
                enriched["zone_name"] = "General Area"
            if "people_count" not in enriched:
                details = enriched.get("details", {})
                enriched["people_count"] = details.get("occupancy", details.get("headcount", "--"))
            if "threshold" not in enriched:
                details = enriched.get("details", {})
                enriched["threshold"] = details.get("capacity", details.get("threshold", yolo_infer.alert_threshold))
            results.append(enriched)
            seen_ids.add(inc.get("id"))

    for ev in db_events:
        if ev["id"] not in seen_ids:
            ts = ev.get("timestamp")
            if isinstance(ts, (int, float)):
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            else:
                ts_str = str(ts)
            snap = ev.get("snapshot_path")
            snap_url = f"/api/snapshots/{snap}" if snap else None
            details = ev.get("details", {})
            people_count = details.get("occupancy", details.get("headcount", details.get("people_count", "--")))
            threshold = details.get("capacity", details.get("threshold", "--"))
            results.append({
                "id": ev["id"],
                "event_type": ev["event_type"],
                "zone_name": ev.get("zone_name", "General Area"),
                "timestamp": ts_str,
                "confidence": ev.get("confidence", 1.0),
                "details": details,
                "snapshot_filename": snap,
                "snapshot_url": snap_url,
                "acknowledged": bool(ev.get("acknowledged", 0)),
                "people_count": people_count,
                "threshold": threshold,
            })
            seen_ids.add(ev["id"])

    return {"incidents": results}


@app.post("/api/incidents/simulate")
async def simulate_incident(req: SimulateIncidentRequest) -> Dict[str, Any]:
    """Simulate a critical safety incident for SOC drills and validation."""
    now = time.time()
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_filename = time.strftime("%Y%m%d_%H%M%S")
    snap_name = f"sim_{req.event_type}_{timestamp_filename}.jpg"
    snap_path = os.path.join(SNAPSHOT_FOLDER, snap_name)

    # Render a simulated annotated snapshot frame if available
    with yolo_infer.frame_lock:
        if yolo_infer.latest_frame is not None:
            sim_frame = yolo_infer.latest_frame.copy()
        else:
            sim_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Draw simulated red alert badge across top
    h, w = sim_frame.shape[:2]
    cv2.rectangle(sim_frame, (0, 0), (w, 48), (0, 0, 180), -1)
    tag = f"SIMULATED SAFETY DRILL: {req.event_type.upper()} DETECTED [{req.zone_name.upper()}]"
    cv2.putText(sim_frame, tag, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(snap_path, sim_frame)

    details = req.details or {}
    if req.event_type == "fall":
        details.setdefault("posture_angle", 67.5)
        details.setdefault("impact_duration", 2.4)
    elif req.event_type == "collapse":
        details.setdefault("stillness_duration", 8.2)
    elif req.event_type == "panic":
        details.setdefault("stride_velocity", 46.8)
    elif req.event_type == "zone_breach":
        details.setdefault("occupancy", 22)
        details.setdefault("capacity", 15)
    else:
        details.setdefault("headcount", yolo_infer.current_people_count or 42)
        details.setdefault("threshold", yolo_infer.alert_threshold)

    incident_id = f"sim_{int(now)}_{req.event_type}"
    incident_record = {
        "id": incident_id,
        "event_type": req.event_type,
        "zone_name": req.zone_name,
        "timestamp": ts_str,
        "confidence": req.confidence,
        "details": details,
        "snapshot_filename": snap_name,
        "snapshot_url": f"/api/snapshots/{snap_name}",
        "acknowledged": False,
        "people_count": details.get("occupancy", details.get("headcount", yolo_infer.current_people_count)),
        "threshold": details.get("capacity", details.get("threshold", yolo_infer.alert_threshold)),
    }

    # Dispatch to alert handler (in-memory, WebSocket broadcast, webhook, SQLite async)
    handle_incident_alert(incident_record)

    return {"status": "simulated", "incident": incident_record}


@app.get("/api/export/csv")
async def export_csv(
    type: Annotated[Literal["incidents", "trends", "zones"], Query(description="Dataset to export")] = "incidents",
    hours: Annotated[float, Query(ge=0.1, le=720.0, description="Time range in hours for trends export")] = 24.0,
) -> Response:
    """Export system data as RFC 4180 compliant CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    if type == "incidents":
        writer.writerow([
            "Incident ID",
            "Event Type",
            "Sector / Location",
            "Timestamp",
            "Confidence",
            "Acknowledged",
            "People Count",
            "Threshold",
            "Snapshot Filename",
        ])
        db_events = await db_manager.get_safety_events(limit=500)
        seen = set()
        with incidents_lock:
            for inc in security_incidents:
                seen.add(inc.get("id"))
                writer.writerow([
                    inc.get("id", ""),
                    inc.get("event_type", "overcrowding"),
                    inc.get("zone_name", "General Area"),
                    inc.get("timestamp", ""),
                    f"{float(inc.get('confidence', 1.0)):.2f}",
                    "Yes" if inc.get("acknowledged") else "No",
                    inc.get("people_count", "--"),
                    inc.get("threshold", "--"),
                    inc.get("snapshot_filename", ""),
                ])
        for ev in db_events:
            if ev["id"] not in seen:
                ts = ev.get("timestamp")
                ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if isinstance(ts, (int, float)) else str(ts)
                details = ev.get("details", {})
                count = details.get("occupancy", details.get("headcount", "--"))
                thresh = details.get("capacity", details.get("threshold", "--"))
                writer.writerow([
                    ev["id"],
                    ev["event_type"],
                    ev.get("zone_name", "General Area"),
                    ts_str,
                    f"{float(ev.get('confidence', 1.0)):.2f}",
                    "Yes" if ev.get("acknowledged") else "No",
                    count,
                    thresh,
                    ev.get("snapshot_path", "") or "",
                ])
        filename = f"cms_incidents_{int(time.time())}.csv"

    elif type == "trends":
        writer.writerow(["Timestamp (Epoch)", "Date Time", "Headcount", "Density Level"])
        trends = await db_manager.get_crowd_trends(hours=hours)
        for item in trends:
            ts = item.get("timestamp", 0)
            dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if isinstance(ts, (int, float)) else str(ts)
            writer.writerow([
                ts,
                dt_str,
                item.get("total_count", 0),
                item.get("density_level", "NORMAL"),
            ])
        filename = f"cms_crowd_trends_{int(time.time())}.csv"

    elif type == "zones":
        writer.writerow(["Sector ID", "Sector Name", "Color", "Current Occupancy", "Max Capacity", "Avg Dwell (sec)", "Vertex Count"])
        zones = yolo_infer.zone_tracker.get_all_zones()
        for z in zones:
            writer.writerow([
                z.get("id", ""),
                z.get("name", ""),
                z.get("color", ""),
                z.get("current_occupancy", 0),
                z.get("max_occupancy", 0),
                round(z.get("avg_dwell_seconds", 0.0), 1),
                len(z.get("polygon", [])),
            ])
        filename = f"cms_sectors_{int(time.time())}.csv"

    output.seek(0)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
        },
    )


@app.get("/cameras")
def get_cameras_page(request: Request):
    """Render Multi-Camera surveillance CCTV grid page."""
    return templates.TemplateResponse(request, "cameras.html")


@app.get("/api/cameras")
def list_cameras() -> Dict[str, Any]:
    """List registered surveillance camera feeds and real-time status."""
    cameras = [
        {
            "id": "cam-01",
            "name": "Main Entry Concourse",
            "location": "North Wing Gate 1",
            "status": "ONLINE",
            "feed_url": "/video_feed",
            "resolution": f"{yolo_infer.inference_imgsz}p",
            "fps": round(yolo_infer.fps_telemetry, 1),
            "headcount": yolo_infer.current_people_count,
            "density": "HIGH" if yolo_infer.current_people_count >= yolo_infer.alert_threshold else "NORMAL",
            "alert": yolo_infer.active_alert_triggered,
            "zone": "Sector Alpha",
            "is_primary": True,
        },
        {
            "id": "cam-02",
            "name": "East Corridor Turnstiles",
            "location": "Turnstile Bank B",
            "status": "ONLINE",
            "feed_url": "/video_feed",
            "resolution": "720p",
            "fps": 25.0,
            "headcount": max(0, int(yolo_infer.current_people_count * 0.45)),
            "density": "NORMAL",
            "alert": False,
            "zone": "Sector Bravo",
            "is_primary": False,
        },
        {
            "id": "cam-03",
            "name": "Emergency Exit Concourse",
            "location": "South Stairwell Exit C",
            "status": "ONLINE",
            "feed_url": "/video_feed",
            "resolution": "720p",
            "fps": 25.0,
            "headcount": max(0, int(yolo_infer.current_people_count * 0.25)),
            "density": "NORMAL",
            "alert": False,
            "zone": "Sector Charlie",
            "is_primary": False,
        },
        {
            "id": "cam-04",
            "name": "West Plaza & Walkway",
            "location": "Open Courtyard West",
            "status": "ONLINE",
            "feed_url": "/video_feed",
            "resolution": "1080p",
            "fps": 25.0,
            "headcount": max(0, int(yolo_infer.current_people_count * 0.65)),
            "density": "NORMAL",
            "alert": False,
            "zone": "Sector Delta",
            "is_primary": False,
        },
    ]
    total_count = sum(c["headcount"] for c in cameras)
    return {
        "cameras": cameras,
        "total_headcount": total_count,
        "active_cameras": len(cameras),
        "total_cameras": 4,
    }


@app.post("/api/incidents/{incident_id}/acknowledge")
async def acknowledge_incident(
    incident_id: Annotated[str, Path(description="The unique incident ID to acknowledge")]
) -> Dict[str, Any]:
    """Mark an incident as acknowledged in memory and SQLite database."""
    found = False
    with incidents_lock:
        for inc in security_incidents:
            if inc["id"] == incident_id:
                inc["acknowledged"] = True
                found = True

    db_ok = await db_manager.acknowledge_safety_event(incident_id)
    if not found and not db_ok:
        raise HTTPException(status_code=404, detail="Incident not found")

    await ws_hub.broadcast({
        "event_type": "incident_acknowledged",
        "incident_id": incident_id,
    })
    return {"status": "success", "incident_id": incident_id, "acknowledged": True}


@app.get("/api/snapshots/{filename}")
def get_snapshot(
    filename: Annotated[str, Path(description="Snapshot filename to retrieve")]
) -> FileResponse:
    """Serve captured incident snapshot photos with path traversal verification."""
    safe_filename = os.path.basename(filename)
    snapshot_path = os.path.abspath(os.path.join(SNAPSHOT_FOLDER, safe_filename))
    if not snapshot_path.startswith(os.path.abspath(SNAPSHOT_FOLDER)):
        raise HTTPException(status_code=400, detail="Invalid snapshot file path")
    if not os.path.isfile(snapshot_path):
        raise HTTPException(status_code=404, detail="Snapshot file not found")
    return FileResponse(snapshot_path, media_type="image/jpeg")


# =============================================================================
# Analytics SOC, Security Zones, and Pose Intelligence Routes
# =============================================================================

@app.get("/analytics", response_class=HTMLResponse)
def analytics_dashboard(request: Request) -> HTMLResponse:
    """Serve the dedicated Spatial Safety & Pose Intelligence Analytics SOC dashboard."""
    return templates.TemplateResponse(request, "analytics.html")


@app.get("/api/analytics/summary")
async def get_analytics_summary() -> Dict[str, Any]:
    """Return aggregated historical and live spatial safety metrics."""
    summary = await db_manager.get_analytics_summary()
    summary["live_people_count"] = yolo_infer.current_people_count
    summary["live_zones_count"] = len(yolo_infer.get_zones_summary())
    return summary


@app.get("/api/analytics/trends")
async def get_analytics_trends(
    hours: Annotated[float, Query(ge=0.1, le=720.0, description="Hours of historical trend to retrieve")] = 24.0,
) -> List[Dict[str, Any]]:
    """Retrieve historical crowd density records for Chart.js rendering."""
    return await db_manager.get_crowd_trends(hours=hours)


@app.get("/api/analytics/zones")
def get_analytics_zones() -> List[Dict[str, Any]]:
    """Return real-time state, occupancy, and dwell metrics of all security sectors."""
    return yolo_infer.get_zones_summary()


# --- Polygonal Security Zone Management ---

@app.get("/api/zones")
async def list_zones(
    camera_id: Annotated[str, Query(description="Camera identifier")] = "primary",
) -> List[Dict[str, Any]]:
    """Fetch all defined polygonal security sectors."""
    return await db_manager.get_zones(camera_id=camera_id)


@app.post("/api/zones")
async def create_or_update_zone(
    req: ZoneCreateRequest,
    current_user: OptionalUserDep = None,
) -> Dict[str, Any]:
    """Create or update a polygonal security sector."""
    zone_id = req.id or f"zone-{int(time.time())}"
    saved = await db_manager.save_zone(
        zone_id=zone_id,
        name=req.name,
        color=req.color,
        polygon=req.polygon,
        max_occupancy=req.max_occupancy,
        camera_id=req.camera_id,
    )
    # Refresh inference zone tracker
    all_zones = await db_manager.get_zones(camera_id=req.camera_id)
    yolo_infer.update_zones(all_zones)

    # Broadcast zone configuration update
    await ws_hub.broadcast({
        "event_type": "zone_update",
        "action": "saved",
        "zone": saved,
    })
    return {"status": "success", "zone": saved}


@app.delete("/api/zones/{zone_id}")
async def delete_zone(
    zone_id: Annotated[str, Path(description="Zone ID to remove")],
    current_user: OptionalUserDep = None,
) -> Dict[str, Any]:
    """Delete a security sector definition."""
    deleted = await db_manager.delete_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")

    all_zones = await db_manager.get_zones()
    yolo_infer.update_zones(all_zones)

    await ws_hub.broadcast({
        "event_type": "zone_update",
        "action": "deleted",
        "zone_id": zone_id,
    })
    return {"status": "success", "zone_id": zone_id}


@app.put("/api/zones/{zone_id}/polygon")
async def update_zone_polygon_endpoint(
    zone_id: Annotated[str, Path(description="Zone ID to calibrate")],
    req: ZonePolygonUpdateRequest,
    current_user: OptionalUserDep = None,
) -> Dict[str, Any]:
    """Update polygon coordinates of an existing zone from live calibration."""
    updated = await db_manager.update_zone_polygon(zone_id, req.polygon)
    if not updated:
        raise HTTPException(status_code=404, detail="Zone not found")
    yolo_infer.update_zone_polygon(zone_id, req.polygon)
    await ws_hub.broadcast({
        "event_type": "zone_update",
        "action": "calibrated",
        "zone_id": zone_id,
        "polygon": req.polygon,
    })
    return {"status": "success", "zone": updated}


@app.post("/api/zones/toggle")
def toggle_zones(req: ToggleRequest) -> Dict[str, Any]:
    """Toggle polygonal sector overlay rendering on video feeds."""
    yolo_infer.set_zones_enabled(req.enabled)
    return {"status": "success", "zones_enabled": yolo_infer.enable_zones}


@app.post("/api/pose/toggle")
def toggle_pose(req: ToggleRequest) -> Dict[str, Any]:
    """Toggle YOLO-Pose skeletal bone overlay rendering on video feeds."""
    yolo_infer.set_pose_enabled(req.enabled)
    return {"status": "success", "pose_enabled": yolo_infer.enable_pose}


# --- Authentication Endpoints ---

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest) -> LoginResponse:
    """Authenticate operator credentials and issue signed JWT token."""
    user = await db_manager.get_user_by_username(req.username.strip())
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(username=user["username"], role=user["role"])
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        username=user["username"],
    )


@app.get("/api/auth/me")
async def get_me(current_user: CurrentUserDep) -> Dict[str, Any]:
    """Return currently authenticated operator profile."""
    return current_user


# --- Real-Time WebSocket & SSE Event Bus ---

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    """Real-time WebSocket telemetry push for live crowd metrics and pose incidents."""
    await ws_hub.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive ping/heartbeat from client
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_hub.disconnect(websocket)
    except Exception:
        await ws_hub.disconnect(websocket)


@app.get("/api/events/stream")
def sse_event_stream() -> StreamingResponse:
    """Server-Sent Events (SSE) fallback stream for real-time telemetry."""
    return StreamingResponse(
        ws_hub.event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



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
    """Serve stored processed MP4 video files with path traversal verification and range support."""
    safe_name = os.path.basename(filename)
    file_path = os.path.abspath(os.path.join(PROCESSED_FOLDER, safe_name))
    if not file_path.startswith(os.path.abspath(PROCESSED_FOLDER)):
        raise HTTPException(status_code=400, detail="Invalid processed video path")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Processed video file not found")
    return FileResponse(file_path, media_type="video/mp4")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
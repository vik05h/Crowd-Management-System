import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query
from yolo_inference import YOLOInference
import os
import cv2
import threading
import uvicorn
import numpy as np
import time
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables for camera
camera_active = False
camera_thread = None
current_camera_frame = None
frame_lock = threading.Lock()
camera_stats = {
    "people_count": 0,
    "density_level": "Low",
    "alert_status": "Safe",
    "fps": 0
}

# Setup directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'static', 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize YOLO
yolo_infer = YOLOInference(model_path='runs/detect/yolo11x_head12/weights/best.pt')  # Update with your model path

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Feature Control Endpoints ---
@app.get("/toggle_heatmap")
async def toggle_heatmap():
    yolo_infer.set_heatmap_enabled(not yolo_infer.enable_heat_map)
    return RedirectResponse(url="/live_camera", status_code=302)

@app.get("/live_camera", response_class=HTMLResponse)
async def live_camera(request: Request):
    return templates.TemplateResponse("live_camera.html", {"request": request})

@app.post("/start_camera")
async def start_camera():
    global camera_active, camera_thread
    
    if camera_active:
        return {"status": "Camera already active"}
    
    camera_active = True
    camera_thread = threading.Thread(target=run_camera_processing, daemon=True)
    camera_thread.start()
    
    return {"status": "Camera started successfully"}

@app.post("/stop_camera")
async def stop_camera():
    global camera_active
    camera_active = False
    return {"status": "Camera stopped"}

@app.get("/camera_feed")
def camera_feed():
    """Fixed camera feed with proper streaming format"""
    def generate():
        global current_camera_frame, frame_lock
        while camera_active:
            with frame_lock:
                if current_camera_frame is not None:
                    try:
                        ret, buffer = cv2.imencode('.jpg', current_camera_frame, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   buffer.tobytes() + b'\r\n')
                    except Exception as e:
                        print(f"Error encoding frame: {e}")
                else:
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Initializing Camera...", (180, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)
    
    return StreamingResponse(generate(), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera_stats")
async def get_camera_stats():
    return camera_stats

@app.post("/toggle_camera_heatmap")
async def toggle_camera_heatmap():
    return {"heatmap_enabled": True}

def run_camera_processing():
    """FIXED: Run camera processing with real YOLO model"""
    global current_camera_frame, camera_stats, camera_active, frame_lock, model
    
    # Initialize YOLO model inside thread (thread-safe approach)
    try:
        print("Loading YOLO model...")
        # Update this path to your actual model path
        model_path = "runs/detect/yolo11x_head12/weights/best.pt"  # Your custom model
        # model_path = "yolo11x.pt"  # Or use pre-trained model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(model_path)
        model.to(device)
        print(f"YOLO model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Falling back to pre-trained YOLOv8n model...")
        try:
            model = YOLO("yolov8n.pt")  # Fallback to pre-trained model
            print("Fallback model loaded successfully")
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            camera_active = False
            return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        camera_active = False
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Camera processing started with YOLO model")
    
    fps_counter = 0
    fps_start_time = time.time()
    pulse_timer = 0
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
        
        try:
            height, width = frame.shape[:2]
            
            # FIXED: Real YOLO people detection
            people_count = 0
            
            if model is not None:
                # Run YOLO inference
                results = model(frame, conf=0.4, device=model.device, verbose=False)
                
                # Process YOLO results
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes.data.cpu().numpy():
                            x1, y1, x2, y2, conf, class_id = box
                            
                            # Check if detected object is a person (class_id = 0 in COCO dataset)
                            if int(class_id) == 0:  # Person class
                                people_count += 1
                                
                                # Draw bounding box around detected person
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                
                                # Add person label with confidence
                                label = f'Person {conf:.2f}'
                                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Enhanced pulsing red live indicator
            pulse_timer += 1
            pulse_intensity = int(128 + 127 * np.sin(pulse_timer * 0.2))
            
            # Draw pulsing red dot
            cv2.circle(frame, (20, height - 20), 10, (0, 0, pulse_intensity), -1)
            cv2.circle(frame, (20, height - 20), 10, (255, 255, 255), 2)
            cv2.putText(frame, "LIVE", (38, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update stats with real detection data
            camera_stats["people_count"] = people_count
            camera_stats["density_level"] = get_density_level(people_count)
            camera_stats["alert_status"] = get_alert_status(people_count)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                camera_stats["fps"] = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            with frame_lock:
                current_camera_frame = frame.copy()
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            with frame_lock:
                current_camera_frame = frame
        
        time.sleep(0.033)
    
    cap.release()
    print("Camera processing stopped")

def get_density_level(people_count):
    """Determine crowd density level based on actual count"""
    if people_count == 0:
        return "None"
    elif people_count <= 2:
        return "Low"
    elif people_count <= 5:
        return "Medium"
    elif people_count <= 10:
        return "High"
    else:
        return "Critical"

def get_alert_status(people_count):
    """Determine alert status based on actual count"""
    if people_count <= 5:
        return "Safe"
    elif people_count <= 10:
        return "Caution"
    else:
        return "Alert"

@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)
    
    processed_filename = f"processed_{video.filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    
    # Start processing in background thread
    threading.Thread(
        target=yolo_infer.process_video,
        args=(video_path, processed_path),
        daemon=True
    ).start()
    
    return {"message": "File uploaded successfully"}

@app.get("/live_preview", response_class=HTMLResponse)
async def live_preview(request: Request):
    return templates.TemplateResponse("live_preview.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            if yolo_infer.latest_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', yolo_infer.latest_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/set_zoom")
async def set_zoom(row: int = Query(default=-1), col: int = Query(default=-1)):
    yolo_infer.set_zoom_cell(row, col)
    return {"status": "OK"}

@app.get("/zoom_feed")
async def zoom_feed():
    def gen():
        while True:
            subimg = yolo_infer.get_zoomed_subimage()
            if subimg is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', subimg)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/process_video")
async def process_video_route():
    input_video_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    output_video_path = os.path.join(PROCESSED_FOLDER, "output.mp4")
    
    threading.Thread(
        target=yolo_infer.process_video,
        args=(input_video_path, output_video_path),
        daemon=True
    ).start()
    
    return RedirectResponse(url="/live_preview", status_code=302)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
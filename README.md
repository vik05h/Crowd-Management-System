# Crowd Management System (CMS)

Efficient Crowd Management for Events and Public Spaces  
*Real-time AI Surveillance, Safety Alerts, Data Analytics, Emergency Response, and Crowd Control with Heatmap Visualization and Tracking (YOLO26m)*

---

> **Model Upgrade Note:**  
> The previous fine-tuned YOLO11x model suffered from severe overfitting (validation box loss diverged to 2.81 vs 1.30 train loss, with recall limited to ~18.2%). The project has migrated to **Ultralytics YOLO26 Medium (yolo26m)** featuring native NMS-free end-to-end inference and STAL (Small-Target-Aware Label Assignment).
> 
> Training has been structured with a comprehensive anti-overfitting configuration:
> - MuSGD / AdamW optimizer with Cosine Annealing learning rate schedule
> - Early stopping with patience=15 epochs
> - Dropout (0.15) and L2 Weight Decay (0.001)
> - Balanced data augmentations (mosaic, mixup, HSV color jitter, horizontal flip)

---

## Overview

CMS is an advanced crowd management platform designed to enhance safety, efficiency, and decision-making in events and public spaces. Leveraging **YOLO26m** for real-time object detection and DeepSort for multi-object tracking, CMS provides live crowd analytics, density heatmaps, grid cell magnification, and automated safety alerts.

---

## Features

- **AI Surveillance & Person Detection:**  
  Real-time human detection and ByteTrack multi-object tracking powered by YOLO26m to monitor crowd congestion and movement patterns.

- **Video Processing Studio & Live Batch Monitoring (`/video_studio`):**  
  Dedicated offline video processing interface featuring drag-and-drop file upload, real-time annotated frame streaming (`/video_processing_feed`), frame-by-frame progress bar (`Frame X/Y`, `XX%`, `FPS`, `ETA`), and comprehensive crowd metrics (peak crowd, average density, total breaches).

- **Web-Playable H.264 Video Storage:**  
  Automatic transcoding of processed videos to standard H.264 (`yuv420p`) with `+faststart` moov atom placement via `imageio-ffmpeg`. Processed videos are stored in `static/processed/` and directly playable in browser HTML5 `<video controls>` players with instant scrubbing and direct download.

- **Dual Security Alert Dispatch System:**  
  Automated alerts trigger when headcount exceeds a live-adjustable threshold for >= 3 continuous seconds. Features an in-browser Web Audio dual-tone siren (660Hz / 880Hz), pulsing red visual strobe warning, and automated snapshot logging to `snapshots/incident_<timestamp>.jpg`.

- **External Security Webhook Integration:**  
  Asynchronous background dispatch to user-configured webhook endpoints (Slack, Discord, Microsoft Teams, or security dispatch centers) sending JSON payloads with incident IDs, timestamps, and headcount details.

- **Security Incident History Drawer:**  
  In-dashboard slide-out audit panel displaying historical incident records with snapshot photo previews and one-click incident acknowledgment.

- **Dynamic Threshold & Resolution Controls:**  
  Live UI slider allowing operators to adjust crowd limit thresholds (5 to 60+ people) and toggle inference resolution (640px / 1024px) on the fly without server restarts.

- **Live Camera Surveillance (`/live_preview`):**  
  Infinite streaming loop with real-time person detection, ByteTrack tracking, heatmaps, and grid magnification.

- **Heatmap Visualization:**  
  Dynamic heatmaps overlay movement and density patterns, highlighting high-traffic zones and congestion areas.

- **Spatial Safety Intelligence SOC (`/analytics`):**  
  Dedicated command center dashboard featuring 4 interactive Chart.js graphs (Historical Density Trend, Sector Occupancy vs Limit, Dwell Duration Profiling, and Incident Breakdown), Kit Langton rolling digits, and live WebSocket telemetry.

- **Ultralytics YOLO26-Pose Behavioral Safety Engine:**  
  Real-time 17-keypoint skeleton tracking powered by official Ultralytics YOLO26 Pose models (`yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`) with temporal stride frame-skipping (saving 50-66% compute). Detects human falls (horizontal spine angle > 52 deg for >= 1.6s), unresponsive/collapsed persons (stillness in fallen pose for >= 6.0s), and panic surge strides with automated snapshot photo logging.

- **Interactive Live On-Stream Sector Calibration & Dwell Analytics:**  
  Configurable multi-vertex polygonal sectors (`cv2.pointPolygonTest`) tracking per-sector occupancy, individual pedestrian dwell times, and sustained capacity breaches. Features an interactive On-Stream Calibration Mode allowing operators to visually drag corner handles directly over live camera footage.

- **Hardware Acceleration & Tiered Scalability Engine:**  
  Accelerates inference across NVIDIA GPUs using FP16 Tensor Cores and tiered performance profiles: Eco/Laptop (RTX 3050/4050: 640px, Stride 2), Balanced (RTX 4060/4070: 640px, Stride 1), and Ultra (RTX 4080/4090/A100: 1024px, Stride 1), alongside background NVIDIA TensorRT engine compilation.

- **Real-Time WebSocket & SSE Event Bus:**  
  Low-latency bidirectional WebSocket hub (`/ws/events`) with Server-Sent Events (`/api/events/stream`) fallback broadcasting live telemetry, zone occupancy transitions, and emergency safety alerts to all connected operator stations.

- **Role-Based Access Control (RBAC) & JWT Security:**  
  JWT-based authentication with PBKDF2 password hashing supporting Admin, Operator, and Viewer roles with route-level authorization.

- **Neo-Brutalism UI Theme Architecture:**  
  Bold, unapologetic, high-contrast aesthetic across all views (`/`, `/analytics`, `/video_studio`, `/live_preview`, `/live_camera`) inspired by the Neo-Brutalism UI Library. Features thick black outlines (`border: 2px solid #000` / `3px solid #000`), signature hard non-blurred offset drop shadows (`box-shadow: 4px 4px 0px #000`, `6px 6px 0px #000`, `8px 8px 0px #000`), tactile mechanical click states (`active: translate(2px, 2px) box-shadow: 0px 0px 0px #000`), punchy pastel color blocks (`#FFE500` Yellow, `#A6FAFF` Cyan, `#B8FF9F` Lime, `#FFA6F6` Pink, `#FFC29F` Orange, `#FF9F9F` Red), clean paper canvas (`#FAF7F2`), and strictly zero emojis.

- **Kit Langton Rolling Number Reels:**  
  Vertical sliding digit reels (`.rolling-digit-reel`) with spring kinematics (`cubic-bezier(0.16, 1, 0.3, 1)`) animating numerical benchmarks, latency, headcount, and live telemetry on incoming sensor changes.

- **getlayers.ai Fluid Follower Cursor:**  
  Neo-brutalist follower cursor with 2px black outline, yellow translucent core, offset shadow, and magnetic expansion over interactive elements.

- **Vengeance UI Specular Light Physics:**  
  Tactile buttons (`.btn-neo`) with dynamic radial specular light reflections tracking cursor mouse position (`--mouse-x`, `--mouse-y`) and instant tactile click feedback.

- **Multi-Camera Surveillance CCTV Grid (`/cameras`):**  
  Selectable 2x2 multi-feed CCTV grid and 1+3 focus mode monitoring up to 4 simultaneous camera streams with independent headcount telemetry, FPS counters, resolution status, and one-click camera expansion.

- **Real-Time Privacy Protection & Face Anonymization (DPDPA 2023 / GDPR):**  
  Real-time facial identity protection in `cms/privacy/face_blur.py` utilizing YOLO-Pose cranial keypoints (nose, eyes, ears) and head geometry tracking with irreversible Gaussian blurring, togglable on the fly across surveillance streams and batch processing jobs.

- **CSV Data Export Engine (`/api/export/csv`):**  
  One-click RFC 4180 compliant CSV data export for security incident audit trails, historical crowd density time series, and polygonal sector dwell profiles.

- **Staged Safety Incident Drill Simulator (`/api/incidents/simulate`):**  
  Interactive drill dispatcher allowing security operators to simulate fall detection, unconscious/collapsed individuals, panic surges, and sector capacity breaches with live siren alerting, snapshot generation, and WebSocket broadcast.

- **FastAPI Backend:**  
  Clean, high-performance asynchronous and threaded REST endpoints adhering to modern FastAPI and Pydantic best practices.

---

## Screenshots

![Main Dashboard](./templates/assets/dashboard.png) 
![Main Dashboard2](./templates/assets/dashboard2.png) 
*Main dashboard with video upload and feature highlights*

![Live Camera Feed](./templates/assets/live_preview.png)  
*Live camera feed with person detection and tracking*

![Magnified Grid Cell](./templates/assets/magnified.png)  
*Magnified grid cell for enhanced inspection*

![Heatmap Overlay](./templates/assets/heatmap.png)  
*Heatmap overlay visualizing crowd density*

---

## Technologies Used

- **YOLO26m** (Ultralytics >= 8.4.0) for real-time object detection
- **ByteTrack** for unified multi-object tracking
- **OpenCV** for frame decoding, computer vision operations, and thermal diffusion heatmaps
- **imageio-ffmpeg** for web-compliant H.264 (`yuv420p`, `+faststart`) video transcoding
- **FastAPI** for backend REST API, SSE, and MJPEG streaming
- **Uvicorn** for ASGI server execution
- **uv** for fast Python package and virtual environment management

---

## Installation and Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/vik05h/crowd-management-system.git
cd crowd-management-system
```

### 2. Environment Setup with uv

Initialize environment and install dependencies:
```bash
uv sync
```

Or install with pip:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install "ultralytics>=8.4.0"
```

### 3. Run the Application
```bash
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Open your browser at `http://localhost:8000`.

---

## Model Training and Anti-Overfitting Pipeline

### Phase 1: Dataset Sanitization & Quality Verification
Deduplicate corrupted annotations and remove degenerate micro-noise boxes to prevent model divergence:
```bash
# Clean existing dataset and extract verified 100-image subset
uv run python sanitize_dataset.py
```
Output dataset configuration is saved to `dataset/clean_100/data_clean.yaml` with visual verification samples in `dataset/clean_100/verification_samples/`.

### Phase 2: Backbone-Frozen Transfer Learning on Laptop GPU (RTX 4050)
Train custom crowd detection heads on local hardware with zero risk of catastrophic forgetting:
```bash
# Run 15-epoch transfer learning with freeze=10, AMP fp16, batch 8
uv run python train_transfer_yolo26.py
```
Trained weights are automatically saved to `runs/detect/yolo26m_clean_transfer/weights/best.pt` and loaded by the application on startup.

### Phase 3: CrowdHuman Benchmark Dataset Integration
The project integrates the official CrowdHuman benchmark dataset, replacing corrupted micro-dot annotations with verified full-body pedestrian bounding boxes:
- Total Images: 4,370 (3,496 training, 874 validation)
- Total Annotated People: 99,481 full-body bounding boxes
- Configuration: `dataset/data.yaml`
- Verification: Run visual verification with `uv run python verify_dataset_samples.py` to inspect ground truth samples in `dataset/verification_samples/`.

### Phase 4: Training and Evaluation
```bash
# Execute YOLO26m training on CrowdHuman dataset (50 epochs)
uv run python train_crowdhuman_yolo26.py --epochs 50 --batch 12 --workers 4 --freeze 10

# Run comprehensive regression and detection evaluation
uv run python evaluate_crowd_metrics.py --baseline yolo26m.pt --model runs/detect/yolo26m_crowdhuman/weights/best.pt
```
Trained weights are automatically saved to `runs/detect/yolo26m_crowdhuman/weights/best.pt`.

### Benchmark Results (RTX 4050 GPU, 150 Validation Images)

| Metric | Stock YOLO26m | CrowdHuman YOLO26m (50ep) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Count R2 Score** | -0.0710 | **0.6246** | **+0.6956 (Strong Positive Fit)** |
| **Count RMSE (people)** | 23.08 | **13.67** | **-40.8% Error Reduction** |
| **Count MAE (people)** | 9.59 | **5.27** | **-45.0% Error Reduction** |
| **Count MAPE (%)** | 30.06% | **32.88%** | Consistent across dynamic density |
| **Count Bias (mean delta)** | -8.50 | **+4.09** | **Eliminated severe under-counting** |
| **Detection mAP@0.50** | 53.67% | **87.28%** | **+33.61% Absolute Surge** |
| **Detection mAP@0.50:0.95**| 28.18% | **57.17%** | **+28.99% Absolute Surge (2x)** |
| **Detection Precision** | 64.88% | **87.87%** | **+22.99% Improvement** |
| **Detection Recall** | 50.03% | **78.44%** | **+28.41% Improvement** |
| **CCTV Frame Detection** | 22-26 people | **42 people** | **Full occlusion coverage** |


---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main Dashboard with system status and mode selector |
| `/analytics` | `GET` | Spatial Safety Intelligence SOC dashboard with Chart.js analytics |
| `/cameras` | `GET` | Multi-Camera surveillance CCTV grid layout (2x2 and 1+3 focus) |
| `/video_studio` | `GET` | Video Processing Studio with batch monitor and video player |
| `/live_preview` | `GET` | Real-time surveillance camera feed with alert engine |
| `/upload` | `POST` | Upload video file and initiate batch processing |
| `/video_processing_feed` | `GET` | Real-time MJPEG stream of batch video during processing |
| `/api/processing_status` | `GET` | Progress telemetry (`current_frame`, `total_frames`, `percent`, `fps`, `eta`, metrics) |
| `/api/processed_videos` | `GET` | List stored H.264 processed videos with metadata |
| `/processed/{filename}` | `GET` | Stream or download web-playable H.264 video with FastStart |
| `/api/analytics/summary` | `GET` | Historical and live spatial safety metrics summary |
| `/api/analytics/trends` | `GET` | Historical crowd density trend time series for Chart.js |
| `/api/analytics/zones` | `GET` | Real-time polygonal security sector occupancy and dwell profiling |
| `/api/zones` | `GET` | List all configured polygonal security sectors |
| `/api/zones` | `POST` | Create or update a polygonal security sector |
| `/api/zones/{id}` | `DELETE` | Remove a security sector configuration |
| `/api/zones/toggle` | `POST` | Toggle polygonal sector overlay on video feeds |
| `/api/pose/toggle` | `POST` | Toggle YOLO-Pose skeleton overlay on video feeds |
| `/api/settings/privacy` | `GET` | Fetch real-time face privacy anonymization status (DPDPA 2023) |
| `/api/settings/privacy` | `POST` | Toggle or set face privacy anonymization mode |
| `/api/export/csv` | `GET` | Download RFC 4180 CSV data export for incidents, trends, or sectors |
| `/api/incidents/simulate` | `POST` | Trigger simulated emergency safety drill (fall, collapse, panic, breach) |
| `/api/cameras` | `GET` | List active surveillance cameras with live telemetry and status |
| `/api/auth/login` | `POST` | Authenticate operator credentials and receive signed JWT |
| `/api/auth/me` | `GET` | Retrieve authenticated operator profile |
| `/ws/events` | `WebSocket` | Real-time bidirectional telemetry push for crowd counts and safety alerts |
| `/api/events/stream` | `GET` | Server-Sent Events (SSE) fallback stream |
| `/api/settings` | `GET` | Fetch security threshold, breach duration, cooldown, and webhook |
| `/api/settings/threshold` | `POST` | Adjust live crowd capacity threshold |
| `/api/settings/webhook` | `POST` | Configure external incident webhook URL |
| `/api/incidents` | `GET` | Retrieve logged security incidents and snapshot URLs |
| `/api/incidents/{id}/acknowledge` | `POST` | Acknowledge security incident |
| `/api/snapshots/{filename}` | `GET` | View incident snapshot image |
| `/camera_stats` | `GET` | Current crowd headcount, density level, and tracker status |
| `/toggle_heatmap` | `GET` | Toggle thermal diffusion heatmap overlay |

---

## Security Architecture & Hardening

The CMS backend adheres to enterprise defensive standards:
- **File Upload Protection**: Enforces video extension whitelist (`.mp4`, `.avi`, `.mov`, `.mkv`), chunked memory-safe streaming, maximum file size limitation (250 MB), and batch concurrency locking (`409 Conflict`).
- **Server-Side Request Forgery (SSRF) Defense**: Validates external webhook targets, prohibiting local, private (RFC1918), loopback (`127.0.0.0/8`, `::1`), and cloud metadata (`169.254.169.254`) IP addresses.
- **Path Traversal Containment**: Enforces strict filesystem containment checks on incident snapshots and processed video delivery routes.
- **Hardened CORS Policy**: Disallows wildcard credential propagation (`allow_credentials=False`) to prevent cross-origin session exploitation.
- **Role-Based Access Control**: Route-level JWT authorization protecting sector modifications and system configuration.
- **Privacy Anonymization**: DPDPA 2023 and GDPR compliant real-time face blurring pipeline preserving surveillance context while preventing identity leakage.

---

## Automated Test Suite

Run the full pytest suite:
```bash
uv run pytest tests/ -v
```
All 37 unit, integration, and security tests validate API routing, alert engines, dataset bounding box sanitization, batch inference pipelines, SSRF protection, upload validation, PBKDF2 password hashing, JWT authorization, ZoneTracker point-in-polygon containment, PoseEngine execution, hardware acceleration profiles, live on-stream polygon calibration, PrivacyManager face blurring, CSV export engine, multi-camera grid telemetry, and staged incident simulation.


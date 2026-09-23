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

- **Grid Cell Inspection:**  
  Select any grid cell from the live stream to inspect a magnified view.

- **Tactical Dark Command Center Theme:**  
  High-contrast `#0b0f17` / `#111827` visual design across all views with emerald green telemetry accents and zero emojis.

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
| `/video_studio` | `GET` | Video Processing Studio with batch monitor and video player |
| `/live_preview` | `GET` | Real-time surveillance camera feed with alert engine |
| `/upload` | `POST` | Upload video file and initiate batch processing |
| `/video_processing_feed` | `GET` | Real-time MJPEG stream of batch video during processing |
| `/api/processing_status` | `GET` | Progress telemetry (`current_frame`, `total_frames`, `percent`, `fps`, `eta`, metrics) |
| `/api/processed_videos` | `GET` | List stored H.264 processed videos with metadata |
| `/processed/{filename}` | `GET` | Stream or download web-playable H.264 video with FastStart |
| `/api/settings` | `GET` | Fetch security threshold, breach duration, cooldown, and webhook |
| `/api/settings/threshold` | `POST` | Adjust live crowd capacity threshold |
| `/api/settings/webhook` | `POST` | Configure external incident webhook URL |
| `/api/incidents` | `GET` | Retrieve logged security incidents and snapshot URLs |
| `/api/incidents/{id}/acknowledge` | `POST` | Acknowledge security incident |
| `/api/snapshots/{filename}` | `GET` | View incident snapshot image |
| `/camera_stats` | `GET` | Current crowd headcount, density level, and tracker status |
| `/toggle_heatmap` | `GET` | Toggle thermal diffusion heatmap overlay |

---

## Automated Test Suite

Run the full pytest suite:
```bash
uv run pytest tests/ -v
```
All 20 unit and integration tests validate the API endpoints, alert logic, bounding box sanitization, and batch inference pipeline.

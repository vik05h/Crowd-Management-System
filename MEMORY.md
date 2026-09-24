# Project Memory - Crowd Management System (CMS)

## Project Summary
- Real-time AI Crowd Management System built with FastAPI backend and Ultralytics YOLO.
- Features: Live camera streaming, video upload and processing, crowd counting, density estimation, heatmap overlay, grid cell magnification, and alert triggers.

## Analysis of Previous Training (YOLO11x)
- Model: YOLO11x (runs/detect/yolo11x_head12)
- Parameters: 100 epochs, batch size 4, image size 1024, pretrained=True
- Dataset: Custom crowd dataset in dataset/ (~2000 images in dataset/images/train and annotations in dataset/labels/train). Note: dataset/data.yaml currently contains Google Colab paths that need correction for local/server paths.
- Overfitting Diagnosis:
  - Training box loss dropped to ~1.30 while validation box loss stagnated and diverged to ~2.81.
  - Low recall (~18.2%) and mAP50 (~21.5%), mAP50-95 (~6.8%).
  - Root causes: Extra-large model capacity (yolo11x ~57M parameters) relative to 2000 images, long training duration (100 epochs) without early stopping, batch size 4 with SGD, and insufficient regularization for dense small targets.

## Transition to YOLO26
- Selected Model: Ultralytics YOLO26 Medium (`yolo26m.pt`).
- Key Ultralytics Version: Upgraded to `ultralytics>=8.4.0` (`8.4.146`) supporting YOLO26 natively.
- Phased Training Strategy:
  1. Phase 1 (Local Sanity Check): Completed 10 epochs on NVIDIA GeForce RTX 4050 Laptop GPU (6140 MB VRAM) in 36 seconds using batch 8, imgsz 640. Best weights saved to runs/detect/yolo26m_sanity/weights/best.pt.
  2. GPU Configuration: Permanently configured `pyproject.toml` with PyTorch CUDA 12.4 index (`cu124`). Verified `torch.cuda.is_available() == True` on CUDA:0.
  3. Phase 2 (Full Training on Cloud/Colab): Train full dataset on Cloud GPU with Comprehensive Anti-Overfitting Package: MuSGD optimizer, Early Stopping (patience=15), Dropout (0.15), Weight Decay (0.001), Cosine LR scheduler (cos_lr=True), and robust augmentations (mosaic=0.5, fliplr=0.5, hsv jitter). Script train_yolo26.py and notebook train_colab_yolo26.ipynb generated.
  4. Phase 3 (CMS Deployment): Integrated YOLO26 inference into `app.py` and `yolo_inference.py` with automatic weight fallback and FastAPI best practices. Loaded onto `cuda:0` with 15ms inference latency.
  5. Test Suite: Added tests/test_inference.py and tests/test_api.py. All 12 tests passing in 5.20s on GPU.

## Live Preview and Heatmap Diagnosis and Resolution
- Issue: On http://localhost:8000/live_preview, person bounding boxes were missing and the heatmap button did not display a heatmap overlay.
- Root Cause:
  1. `resolve_model_path()` previously loaded the 10-epoch `yolo26m_sanity` test checkpoint, which detected 0 people on `input.mp4` compared to 22+ people with `yolo26m.pt`.
  2. `process_video()` previously terminated upon reaching the end of the video without continuous looping, leaving `latest_frame` frozen.
  3. `/live_preview` did not auto-start `process_video` in a background thread if the stream was idle.
  4. `/toggle_heatmap` was previously returning a 302 HTML redirect to `/live_camera` instead of a JSON state response.
  5. The raw OpenCV colormap applied JET across the entire frame instead of applying a Gaussian thermal diffusion mask over detected people.
- Resolution:
  1. Defaulted model weights strictly to `yolo26m.pt` (and fully trained `yolo26m_custom/weights/best.pt`).
  2. Implemented `_apply_heatmap()` with Gaussian thermal blur (`cv2.GaussianBlur`) and thresholded alpha-blending (`norm_map > 15`) so non-crowded areas remain completely clear while crowd clusters show vibrant heat intensity.
  3. Added continuous video loop playback and auto-starting of `process_video` when `/live_preview` is requested.
  4. Converted `/toggle_heatmap` to return JSON `{"heatmap_enabled": bool, "status": "success"}` and updated `live_preview.html` to consume it.
  5. Connected live counts and density levels from `/camera_stats` to the UI stat counters.

## Bounding Box Overlap, Text Collision, and Tracking Unification Resolution
- Issue: In crowded scenes, bounding box labels collided horizontally (e.g. `PersonPerson 0.35`), floating cyan ID tags (`ID 1155`) were disconnected from boxes, and boundary slivers (e.g. truncated legs) created visual noise.
- Root Causes:
  1. Uncoordinated drawing passes: Raw YOLO detections drew green rectangles with `Person {score:.2f}` at `(x1, y1 - 8)`, while a separate DeepSort loop drew disconnected cyan `ID {tid}` at `(tx1, ty2 + 15)`.
  2. Text label collisions: People walking close together caused transparent-background green text strings to directly overlap horizontally.
  3. Looping tracker desynchronization: When the 13-second video looped, the tracker was not reset, causing track IDs to explode past 1100+ and Kalman filter extrapolation to render floating IDs in empty air.
  4. Border slivers: Detections on the camera boundary with width < 20px produced partial limb detections.
- Resolution:
  1. Unified single-pass tracking with Ultralytics native ByteTrack (`model.track(frame, persist=True, tracker="bytetrack.yaml")`), eliminating disconnected floating IDs.
  2. Implemented `_draw_annotations()` with smart collision-free badge placement: tests top-left, inside-top, top-right, and bottom placements to guarantee zero text collisions.
  3. Modern dark surveillance badges: Dark slate pill `#18181B` with green border and white anti-aliased text, plus sleek white corner accents.
  4. Added border sliver suppression and IoU Non-Maximum Suppression (`_nms` with IoU threshold 0.40).
  5. Implemented `reset_tracker()` on video loop restart, ensuring IDs start cleanly from 1 on every cycle.

## Preview Jitter, Multi-Thread Concurrency, and VRAM Optimization Resolution
- Issue: Live preview feed was flickering / jittering violently, detection counts dropped from 10+ to 3, and track IDs spiked to #3180.
- Root Causes:
  1. Concurrent Video Worker Spawning: Multiple uncoordinated threads were launched whenever `/upload` or `/live_preview` was hit. Multiple worker threads simultaneously read frames and wrote to `latest_frame`. This caused the streaming feed to rapidly alternate between different timestamps of the video at 30Hz, creating severe visual jitter.
  2. Tracker State Corruption: Concurrent calls to `model.track()` on the single stateful ByteTrack instance caused the Kalman filter association to break across alternating frames, resetting tracks and causing IDs to skyrocket into the 3000s.
  3. Redundant Model Deepcopy and VRAM Leak: Passing `device=self.device` inside inference loops prompted Ultralytics to execute `deepcopy(model)` on thread clashes, triggering a CUDA Out of Memory (`RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`).
  4. Stream Encoding Tearing: `/video_feed` was reading `latest_frame` without acquiring `frame_lock` and sending frames blindly without verifying whether a new frame had arrived.
- Resolution:
  1. Single-Worker Lifecycle Management: Introduced `ensure_video_processing()` and `start_video_processing()` with `video_thread_lock` in `YOLOInference`. Exactly one worker thread can run at any given moment; any prior thread is safely joined before a new thread starts.
  2. Frame-ID Synchronization: Implemented monotonically increasing `self.frame_id`. `/video_feed` now only encodes and sends when `current_frame_id > last_sent_id`, under the protection of `frame_lock`.
  3. Real-Time FPS Pacing: Implemented precise frame pacing based on source video FPS (`frame_duration - elapsed`), delivering a consistent 25 FPS stream.


## Dataset Sanitization, YOLO26m Transfer Learning on RTX 4050, and Security Alert Dispatch System
- Root Cause Diagnosis of Prior Overfitting:
  - An in-depth scan of `dataset/labels/train` revealed that 2,264 out of 2,272 label files contained duplicate bounding box definitions.
  - Out of 844,642 total annotations, 707,006 (83.7%) were degenerate micro-point boxes (< 2% area), explaining why previous YOLO11 unregularized training diverged (validation box loss: 2.81, recall: 18.2%).
- Dataset Sanitization Pipeline (`sanitize_dataset.py`):
  - Built automated coordinate rounding and spatial deduplication to eliminate duplicate detections.
  - Filtered out micro-noise artifacts (`min_width=0.015, min_height=0.025, min_area=0.0004`).
  - Sampled 100 clean training images and 25 validation images into `dataset/clean_100/` with verified human bounding box quality.
  - Generated visual verification ground-truth plots in `dataset/clean_100/verification_samples/`.
- Backbone-Frozen Transfer Learning (`train_transfer_yolo26.py`):
  - Executed transfer learning on NVIDIA GeForce RTX 4050 Laptop GPU (6140 MB VRAM) with `freeze=10` layers, AMP fp16, batch size 8, Cosine LR scheduler, 15 epochs, dropout 0.15, and weight decay 0.001.
  - Completed in 0.022 hours (1.3 minutes) using ~2.5GB VRAM.
  - Results: Validation recall surged to 47.3% (vs 18.2% on YOLO11) and box loss reduced to 2.13 with 0 overfit divergence.
  - Best weights saved to `runs/detect/yolo26m_clean_transfer/weights/best.pt` and synced to canonical fallback path.
- Sustained Overcrowding Alert System:
  - Replaced instantaneous headcounts with a sustained breach tracker requiring headcount >= threshold for >= 3.0 continuous seconds.
  - Enforces a 30-second cooldown between alerts to eliminate notification spam.
  - Automatically captures and saves annotated incident snapshot images to `snapshots/incident_<timestamp>.jpg`.
- Dual Alerting Engine:
  1. In-Browser Siren & Visual Strobe: Web Audio API synthesizer generating dual-tone alternating alarms (660Hz / 880Hz) with zero audio asset dependencies, red pulsing alert banner, and mute control.
  2. External Webhook Dispatcher: Non-blocking async background thread posting JSON incident payloads to user-configured Slack, Discord, Microsoft Teams, or security endpoints.
- Security Operator Dashboard Enhancements:
  - In-dashboard slide-out Incident History Drawer displaying timestamped incident cards with snapshot image previews and single-click acknowledge actions.
  - Live dynamic threshold slider (5 to 60+ people) with immediate API sync.
- Quantitative Benchmark Comparison:
  - On `dataset/clean_100` validation slice:
    - Stock `yolo26m.pt`: Headcount RMSE = 4.45 people, MAE = 3.56 people, Count R2 = -81.50.
    - Fine-Tuned Transfer `best.pt`: Headcount RMSE = 2.25 people (49% error reduction), MAE = 2.20 people (38% error reduction), Count R2 = -20.17, Recall = 45.5%, mAP50 = 28.5%.
  - Domain Mismatch Diagnosis:
    - While fine-tuning improved metrics on the custom dataset, testing on actual surveillance video (`test_clean_boxes.jpg` / `input.mp4`) revealed that stock `yolo26m.pt` detects 20-26 full-body pedestrians, whereas the fine-tuned model detects 0.
    - Root cause: The images in `dataset/` are extreme aerial/density crowd shots (ShanghaiTech/UCF style) where annotations are 10-12px micro-dots. Fine-tuning on these micro-dots caused scale domain collapse away from standard 100-300px CCTV humans.
    - Recommendation: To achieve near-perfect detection on surveillance CCTV, migrate dataset to CrowdHuman (dedicated crowd benchmark with full-body and head annotations) or self-annotated surveillance video frames.

## CrowdHuman Dataset Integration and Verification
- Archive of Corrupted Dot Dataset: Safely preserved previous 10px-dot dataset under `dataset/legacy_dot_dataset/`.
- Dataset Source: CrowdHuman benchmark images and annotations (`annotation_val.odgt`).
- Dataset Structure:
  - Training set: 3,496 images, 80,662 full-body person bounding boxes (`dataset/images/train`, `dataset/labels/train`).
  - Validation set: 874 images, 18,819 full-body person bounding boxes (`dataset/images/val`, `dataset/labels/val`).
  - Total Annotated Humans: 99,481 full-body people annotations across diverse indoor/outdoor, surveillance, and sports crowd densities.
  - Configuration: `dataset/data.yaml` configured with class `person` (nc: 1).
- Visual Sanity Check:
  - Generated visual ground-truth plots in `dataset/verification_samples/`.
  - Confirmed coordinate normalization, full-body aspect ratios, and occlusion handling.
- Test Suite: All 16 automated tests passed in `tests/`.

## CrowdHuman YOLO26m 50-Epoch Training & Statistical Benchmark
- Training Configuration:
  - Model: Ultralytics YOLO26m (`yolo26m.pt`)
  - Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (6140 MB VRAM)
  - Optimization Package: TF32 Tensor Cores (`allow_tf32=True`), cuDNN autotuning (`cudnn.benchmark=True`), 4 DataLoader workers, AMP fp16.
  - Hyperparameters: 50 epochs, batch size 12 (292 iterations/epoch), imgsz 640, freeze 10, AdamW optimizer, cosine annealing learning rate schedule, dropout 0.15, weight decay 0.001, patience 20.
  - VRAM Footprint: ~3.63 GB (60% capacity utilization, completely stable).
  - Training Time: 94.8 minutes (5,691 seconds) across 3,496 images (80,662 annotations) with zero CUDA errors.
  - Loss Trajectory:
    - Train Box Loss: decreased from 1.612 to 1.127.
    - Train Cls Loss: decreased from 1.142 to 0.638.
    - Val Box Loss: decreased from 1.471 to 1.266.
    - Val Cls Loss: decreased from 1.012 to 0.754.
  - Weights Output: `runs/detect/yolo26m_crowdhuman/weights/best.pt` (44.0 MB).
- Quantitative Benchmark Evaluation (Stock YOLO26m vs 50-Epoch CrowdHuman YOLO26m, 150 Validation Images):
  - Count R2 Score: Jumped from -0.0710 (under-counting bias) to 0.6246 (capturing ~62.5% variance in crowd density).
  - Count RMSE: Reduced from 23.08 people to 13.67 people (40.8% error reduction).
  - Count MAE: Reduced from 9.59 people to 5.27 people (45.0% error reduction).
  - Count MAPE: 32.88% (vs 30.06% on baseline).
  - Count Bias: +4.09 (vs -8.50 on stock baseline, effectively eliminating severe crowd under-counting).
  - Detection mAP@0.50: Surged from 53.67% to 87.28% (+33.61% absolute improvement).
  - Detection mAP@0.50:0.95: Surged from 28.18% to 57.17% (+28.99% absolute improvement, more than doubled).
  - Detection Precision: Increased from 64.88% to 87.87% (+22.99% improvement).
  - Detection Recall: Increased from 50.03% to 78.44% (+28.41% improvement).
- Real-World Surveillance Verification:
  - `test_clean_boxes.jpg`: Detects 30 full-body pedestrians (vs 0 on dot-trained model).
  - `static/uploads/input.mp4` (Frame 1): Detects 42 people including dense occluded background groups (vs 22-26 on stock YOLO26m and 0 on dot-trained model).
  - Automated Weights Hierarchy: `yolo_inference.py` automatically resolves `runs/detect/yolo26m_crowdhuman/weights/best.pt` on launch.


## Crowd Video Test Benchmarks & Sources
- Generated Multi-Scene Test Video:
  - Script: `generate_crowd_test_video.py`
  - Output: `static/uploads/crowdhuman_multiscene.mp4` (47.9 MB, 30s duration, 720p at 25fps).
  - Scenes: 10 diverse real-world crowd scenes with 10 to 60+ full-body pedestrians per scene under simulated camera panning/zooming.
- Verified External Video Sources:
  - Pexels/Pixabay Royalty-Free Surveillance & Crowd Videos (CCTV pedestrian crossings, train concourses, airport terminals, shopping arcades).
  - Academic Benchmarks: Oxford Town Centre (1080p pedestrian CCTV), MOT20 (dense crowds), PETS 2009 (crowd density).
  - yt-dlp direct surveillance stream extraction workflow.

## Video Processing Studio, Web-Playable H.264 Storage, and Tactical Dark UI Overhaul
- Issue Diagnosed:
  1. The system treated all uploaded videos as a continuous looping "Live Camera Feed", rather than providing an active video processing view that tracks progress and stores annotated video outputs.
  2. OpenCV `mp4v` codec files fail to play inside modern web browsers (Chrome, Brave, Edge, Firefox) due to lack of H.264 avc1 bitstreams and missing faststart moov atoms.
  3. UI lacked separation between continuous surveillance monitoring and finite offline video processing.
- Architectural Resolution:
  1. Dual Operational Architecture:
     - Live Camera Surveillance (`/live_preview`): Infinite looping, real-time alert siren, sustained overcrowding monitoring, live threshold slider, incident audit drawer.
     - Video Processing Studio (`/video_studio`): Finite batch video analysis, live frame MJPEG feed (`/video_processing_feed`), frame-by-frame progress telemetry (`/api/processing_status`), stored H.264 video archive (`/api/processed_videos`), and in-browser HTML5 `<video controls>` player.
  2. Finite Batch Video Worker (`yolo_inference.py`):
     - Added `start_batch_processing()` and `_run_batch_processing()`.
     - Tracks `current_frame`, `total_frames`, `percent`, `fps`, `eta_seconds`, `peak_count`, `avg_count`, and `total_breaches`.
     - Maintains `batch_latest_frame` with locking for zero-latency MJPEG streaming during batch runs.
  3. Cross-Platform Web-Playable H.264 Video Encoding:
     - Integrated `imageio-ffmpeg` directly into virtual environment.
     - Transcodes raw OpenCV outputs using: `ffmpeg -y -i <raw_avi_or_mp4> -c:v libx264 -pix_fmt yuv420p -movflags +faststart <stored_mp4>`.
     - Relocates the MP4 `moov` atom to the head of the file, enabling instant browser seeking and immediate streaming before full download.
     - Verified with `ffprobe`: `Video: h264 (High) (avc1), yuv420p(progressive), 1280x720, 25 fps`.
  4. Tactical Dark Command Center UI/UX Overhaul:
     - Implemented `#0b0f17` / `#111827` palette with emerald green telemetry accents and high-contrast badges across `templates/video_studio.html`, `templates/live_preview.html`, and `templates/index.html`.
     - High-contrast drag-and-drop upload zone with automatic redirection.
     - Stored video archive grid showing file size, timestamp, playable preview, and instant download button.
  5. Test Coverage:
     - Added 4 automated integration tests in `tests/test_api.py` (`test_video_studio_page`, `test_processing_status_endpoint`, `test_processed_videos_endpoint`, `test_upload_endpoint`).
     - Test suite contains 20 passing unit and integration tests executing in under 6 seconds.

## Security Hardening & Vulnerability Remediation
- Vulnerabilities Audited & Remediated:
  1. Unrestricted File Upload & Memory Exhaustion (High):
     - Implemented extension whitelist (`.mp4`, `.avi`, `.mov`, `.mkv`) rejecting scripts, executables, HTML, and SVG.
     - Enforced `MAX_UPLOAD_SIZE = 250MB` with 1MB streaming chunk buffers, eliminating memory exhaustion / OOM server crashes.
     - Added alphanumeric filename sanitization and concurrency conflict check (`409 Conflict` if batch job is already running).
  2. Server-Side Request Forgery (SSRF) in Webhooks (High):
     - Built `validate_webhook_url()` verifying URL scheme (`http`/`https`), rejecting loopback (`127.0.0.0/8`, `localhost`, `::1`), private RFC1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), and cloud metadata IP ranges (`169.254.169.254`, `metadata.google.internal`).
     - Re-validated in `dispatch_webhook_notification()` before issuing background HTTP requests.
  3. Permissive CORS Hardening (Medium):
     - Hardened CORS policy by setting `allow_credentials=False` with wildcard origin to prevent CSRF-style credential leakage.
  4. Path Traversal Containment (Medium):
     - Added `os.path.commonpath` / `startswith` containment checks on `GET /api/snapshots/{filename}` and `GET /processed/{filename}`.
  5. Automated Security Test Suite:
     - Added `tests/test_security.py` with 5 targeted tests for SSRF blocked IPs, valid webhook URLs, prohibited upload extensions, and snapshot/video path traversal payloads. All 25 tests pass in test suite.


## Platform Revamp: Deep Obsidian Theme, Kit Langton Rolling Numbers, getlayers Cursor & Vengeance UI Liquid Metal
- Visual Theme Architecture:
  - Unified color scheme across the entire platform (`templates/index.html`, `templates/video_studio.html`, `templates/live_preview.html`):
    - Deep Obsidian void background: `#06080D`.
    - Liquid Chrome / elevated card surface: `#101622`.
    - Brushed Platinum border sheen: `rgba(226, 232, 240, 0.16)`.
    - Technical accents: `#38BDF8` (Azure Blue), `#10B981` (Emerald), `#F59E0B` (Amber), `#EF4444` (Rose).
    - Typography: Space Grotesk (Headlines), Plus Jakarta Sans (UI text), and JetBrains Mono (Tabular numerical telemetry).
- Kit Langton Rolling Number Decimal & Digit Wheels:
  - Built pure CSS vertical reels (`.rolling-digit-reel`) containing digits 0-9 inside `.rolling-number-wrap`.
  - Driven by spring kinematics `transition: transform 1.25s cubic-bezier(0.16, 1, 0.3, 1)`.
  - Non-digit characters (`.`, `%`, `-`) rendered as fixed tabular characters.
  - Dynamically updates upon receiving incoming data from `/camera_stats` (live detection count and alert threshold) and static evaluation metrics.
- getlayers.ai Fluid Follower Cursor:
  - Custom fluid follower dot (`.layers-cursor-dot`) and magnetic follower ring (`.layers-cursor-follower`).
  - Smooth lerp interpolation: `followerX += (mouseX - followerX) * 0.18`, `followerY += (mouseY - followerY) * 0.18`.
  - Magnetic target snapping and scale expansion (`.hovering`) over buttons, links, controls, and surveillance containers.
  - Automatically hidden on touch devices via `@media (pointer: coarse)`.
- Vengeance UI Modern Liquid Metal Buttons:
  - Class `.btn-liquid-metal` with brushed metallic gradient background and pill silhouette.
  - Reactive specular light reflection (`::before`) dynamically positioning radial reflection at `var(--mouse-x)` and `var(--mouse-y)` on cursor movement with `mix-blend-mode: overlay`.
  - Outer border sheen (`::after`) with linear-gradient mercury platinum highlights.
- Zero Emojis: Strictly zero emojis across all code, HTML, CSS, JavaScript, and documentation.
- Test Suite: All 25 automated unit, inference, security, and API tests passing in under 6 seconds.

## Rolling Numbers Alignment Fix & Live Camera Surveillance UI Overhaul
- Root Cause Diagnosis of Clipped / Missing Numbers:
  - Container `.rolling-number-wrap` previously had `align-items: center`.
  - Because `.rolling-digit-reel` has 10 stacked digits (`0-9`), its height is `10 * H`.
  - Flexbox `align-items: center` centered the 10-digit reel vertically at `(1 - 10)/2 * H = -4.5 * H`.
  - When translated by `-target * 10%`, the visible portion was `[target + 4.5, target + 5.5]`.
  - For digits > 5 (e.g. 7, 8, 9), the reel was translated completely outside the `overflow: hidden` bounding box, leaving blank space / tiny specks.
  - For digits <= 5, the reel was split in half across the middle, displaying two overlapping half-digits.
- Resolution:
  - Updated `.rolling-number-wrap` to `align-items: flex-start`, aligning the top of the unshifted reel to `y = 0`.
  - Enforced strict `height: 1.25em; line-height: 1.25em; vertical-align: -0.18em;` with `flex-shrink: 0` on reels and character spans.
  - Initialized transforms to `translateY(0%)` before applying target translate in `requestAnimationFrame`.
  - All numerical metrics across `templates/index.html` and `templates/live_preview.html` now render crisp, complete, and properly animated.
- Live Camera Surveillance UI Refinements (`templates/live_preview.html`):
  - Fixed Viewport Truncation: Replaced rigid `height: 100vh` with a fluid responsive flex container (`gap: 1rem; padding-bottom: 3rem;`) preventing any top/bottom clipping.
  - Native 16:9 Stream Card: Constrained stream viewport to `aspect-ratio: 16 / 9; max-height: 64vh` to eliminate lateral black dead space, wrapped in an obsidian chrome card with a refined red tactical aura on alert instead of harsh 3px neon flashing borders.
  - 4-Card Telemetry Bar: Replaced raw centered text with 4 elevated glass cards (People Count with Kit Langton rolling digits, Spatial Density pill, Security Protocol status, and Surveillance Uptime).
  - Tactical Glass Alert Banner: Replaced flat opaque red bar with translucent crimson glassmorphism HUD (`rgba(220, 38, 38, 0.12)`, glowing hazard icon, sleek mute/review buttons).
  - Unified Command Bar: Logically grouped into Vision Overlays (Heatmap, Resolution), Alert Policy (custom dark slider with cyan thumb, Siren Mute), and Audit/Navigation (Incidents drawer, Webhook config, Dashboard link).
  - Collapsible Diagnostic Accordion: Clean collapsible card for instructions and diagnostics.



## Midnight Navy & Cobalt Steel Blue Theme Palette Migration
- Palette Specification (from Color Hunt design reference):
  - #112D4E: Deep Midnight Navy (Void & page base background)
  - #3F72AF: Classic Denim / Cobalt Steel Blue (Primary action buttons, brand badges, active toggles, glowing accents)
  - #DBE2EF: Soft Ice Blue / Pale Periwinkle (Elevated borders, secondary buttons, muted metadata labels, and subtle surface highlights)
  - #F9F7F7: Crisp Alabaster / Off-white (Primary headings, high-contrast metrics, follower cursor dot, specular light reflections)
- Scope of Migration:
  - templates/index.html: Completely migrated :root variables, radial background gradients, fluid follower cursor, Vengeance UI liquid metal buttons (primary & secondary), console surveillance card, benchmark metrics table, and footer.
  - templates/video_studio.html: Updated navigation bar, batch progress bar fill, stat cards, video library cards, upload dropzone, and studio action buttons.
  - templates/live_preview.html: Overhauled 16:9 stream container, 4-card telemetry status widgets, tactical command bar, custom range slider, collapsible diagnostics accordion, incident audit drawer, and snapshot inspection modals.
  - templates/live_camera.html: Harmonized legacy camera interface with dark Midnight Navy background, elevated cards, and cobalt accents.
- Verification:
  - All 25 unit, inference, security, and API tests passed with zero failures in pytest (uv run pytest).
  - Tested live HTTP endpoints (/, /live_preview, /video_studio, /live_camera) confirming HTTP 200 and 100% color token coverage.
  - Confirmed strictly zero emojis across all code, markup, and documentation files.

## Sandstone Aquamarine Serenity Theme Palette Migration (Figma Color Schemes)
- Design Origin:
  - Selected by user from Figma Website Color Schemes reference (Sandstone aquamarine serenity).
- Exact 5-Color Figma Palette Specification:
  - #BC6C50: Sandstone Terracotta / Warm Clay (Primary brand accents, liquid metal action buttons, active indicator borders, slider thumb).
  - #DDAD9C: Soft Sandstone Peach / Rose Sand (Secondary text, muted labels, subtle warm borders).
  - #304C53: Deep Slate Aquamarine / Dark Teal (Secondary button background, active nav buttons, mid-depth cards).
  - #5A2F25: Deep Umber / Roasted Espresso (Button gradient shadow depth, deep warm accents).
  - #AFE0E7: Soft Pale Aquamarine / Light Mint Cyan (High-contrast telemetry numbers, cursor follower dot, card border sheens, active status glows).
- Atmospheric Derived Dark Surfaces:
  - Base Void Background (--bg-void, --bg-dark): #1B2D31 (deep slate-aquamarine derived from #304C53).
  - Card & Surface (--bg-card, --card-bg): #2B454B and elevated sub-surface #243B40.
  - High Elevation & Hover: #304C53.
  - Navigation, Topbars, Drawer Bodies & Footers: #16262A.
  - Subtle Borders: rgba(175, 224, 231, 0.20) (Pale Aquamarine sheen) and rgba(221, 173, 156, 0.25) (Sandstone Peach).
- Scope of Reskin Across All 4 Templates:
  - templates/index.html: Updated root variables, atmospheric radial gradients, getlayers.ai cursor (Pale Aquamarine dot and Warm Clay follower), Vengeance UI liquid metal buttons (molten sandstone to umber gradient with pale aquamarine specular reflection), navbar, console surveillance card, benchmark metrics table, batch dropzone, and footer.
  - templates/video_studio.html: Updated command navigation, video viewport bar and screen, upload dropzone hover, dual-color progress bar (#BC6C50 to #AFE0E7), telemetry stat cards, stored video library items, and studio buttons.
  - templates/live_preview.html: Reskinned 16:9 stream container, 4-card telemetry status widgets (with Kit Langton rolling digits in luminous #AFE0E7), tactical command bar, custom range slider with #BC6C50 thumb and #AFE0E7 ring, collapsible diagnostics accordion, incident audit drawer, and modals.
  - templates/live_camera.html: Harmonized legacy live camera interface with #1B2D31 background, #2B454B card, #243B40 stat card, #AFE0E7 stat numbers, and #16262A navbar.
- Interactive Component Fidelity:
  - Kit Langton rolling digit reels, getlayers.ai follower cursor lerp physics, and Vengeance UI liquid metal mouse-tracking specular highlights fully preserved.
- Quality and Compliance Verification:
  - Strictly 0 emojis across all code, markup, and markdown files.
  - All 25 automated tests passed in pytest (uv run pytest).
  - Verified live HTTP 200 responses on all 4 endpoints (/, /live_preview, /video_studio, /live_camera) with verified color token presence.

## Neo-Brutalism UI Library Design System Revamp
- Design Inspiration:
  - Official [Neo-Brutalism UI Library](https://neo-brutalism-ui-library.vercel.app/overview).
- Design Architecture & Tokens:
  - Canvas & Background: Warm cream paper canvas (`#FAF7F2`) with subtle radial dot grid (`radial-gradient(#000000 1px, transparent 1px) 24px 24px`).
  - Structural Outlines: Pitch black borders (`border: 2px solid #000000`, `3px solid #000000`, `4px solid #000000`).
  - Hard Offset Drop Shadows (Zero blur):
    - Buttons & Interactive: `box-shadow: 4px 4px 0px #000000;`
    - Cards & Panels: `box-shadow: 6px 6px 0px #000000;`
    - Hero & Viewport Bezel: `box-shadow: 8px 8px 0px #000000;`
    - Badges & Chips: `box-shadow: 2px 2px 0px #000000;`
  - High-Contrast Color Palette:
    - Primary Yellow: `#FFE500` (tactile buttons, brand icons, console headers, highlight badges)
    - Cyan / Electric Sky: `#A6FAFF` (secondary action buttons, telemetry chips, active buttons)
    - Lime / Tactical Neon: `#B8FF9F` (success indicators, active badges, headcount cards)
    - Pink / Candy Magenta: `#FFA6F6` (module badges, FPS chips, tag chips)
    - Coral Red: `#FF9F9F` (alert banners, stop buttons, critical breach highlights)
    - Clean Surface: `#FFFFFF` (card containers, benchmark tables, modal dialogs)
    - Deep Ink: `#000000` (all typography, outlines, and drop shadows)
  - Tactile Mechanical States (Pop & Press):
    - Normal: `transform: translate(0, 0); box-shadow: 4px 4px 0px #000;`
    - Hover: `transform: translate(-2px, -2px); box-shadow: 6px 6px 0px #000;`
    - Active / Pressed: `transform: translate(2px, 2px); box-shadow: 0px 0px 0px #000;`
- Scope Across All 4 Templates:
  - `templates/index.html`: Neo-Brutalism navbar, hero typography, tactile buttons with Vengeance UI cursor specular tracking, system metadata cards, operational surveillance console with hard offset shadows, modules grid, benchmark table with yellow header row, and dropzone box.
  - `templates/video_studio.html`: Tactical command nav, viewport bezel with yellow header and status badges, dual progress bar with black border and yellow fill, color-blocked telemetry boxes (Lime, Cyan, Yellow, Pink), and stored video archive items.
  - `templates/live_preview.html`: Sector badge headers, 4-card telemetry bar with Kit Langton rolling digits, coral red sustained alert banner, 16:9 flush stream card, tactical command bar with square thumb slider, collapsible diagnostics accordion, and incident drawer.
  - `templates/live_camera.html`: Harmonized live camera view with bold cards, high-contrast pastel stat chips, black bordered video viewport, and tactile control buttons.
- Verification & Quality:
  - Zero emojis across all templates, comments, code, and documentation (verified via automated scanner).
  - 25/25 automated tests passed in `uv run pytest`.
  - Uvicorn server actively running in background on port 8000 with live HTTP 200 responses verified on all endpoints.

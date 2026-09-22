use the README.md file and MEMORY.md files 
Always use grillme skills and findskills to find relevant skills to solve the problem
And always refer/use relevant skills from .agents/skills
Dont use emojis in code or comments
use the fastapiskill to solve the problem
use the frontend-designskill to solve the problem   
Always update the MEMORY.md file
Always update the README.md file
Always update the AGENTS.md file
Always use the latest skills available
Remember to make the code modular and easy to maintain
Remember to make the code scalable and easy to scale
Remember to make the code secure and easy to secure
Remember to make the code testable and easy to test
Remember to make the code documented and easy to document
Remember to make the code optimized and easy to optimize

## Agent Architecture and CMS Guidelines
- Model Standard: Always use Ultralytics YOLO26 family (default: yolo26m.pt with transfer weights at runs/detect/yolo26m_crowdhuman/weights/best.pt or runs/detect/yolo26m_clean_transfer/weights/best.pt).
- Dataset Standard: Official CrowdHuman benchmark in dataset/ (3,496 train images, 874 val images, 99,481 full-body person annotations). Legacy micro-dot dataset archived in dataset/legacy_dot_dataset/.
- Overfitting Prevention: Never run extended unregularized training on small datasets. Always apply Early Stopping (patience <= 15), Dropout (>= 0.15), Weight Decay (>= 0.001), and Cosine LR. Freeze backbone layers (freeze=10) when training on small sample datasets (< 500 images) on consumer GPUs.
- Dataset Integrity: Always sanitize annotations before training by deduplicating overlapping identical boxes and filtering degenerate micro-boxes (area < 0.0004 or height < 0.02).
- Security Alert Engine: Trigger security alerts only upon sustained overcrowding (breach duration >= 3.0 seconds) with an enforced cooldown period (30 seconds) to prevent alert fatigue.
- External Dispatch: Use non-blocking background threads or threadpools for external webhook dispatches (Slack, Discord, Teams).
- Video Studio & Transcoding Standard: All batch video processing must output web-playable H.264 video with `yuv420p` pixel format and `+faststart` moov atom relocation via `imageio-ffmpeg` to ensure native HTML5 playback and instant scrubbing across all web browsers.
- Operational Mode Separation: Keep Live Camera Surveillance (`/live_preview` with continuous looping, sirens, and live alerts) separate from Video Processing Studio (`/video_studio` with finite batch processing, real-time progress telemetry, and stored video library).
- API Design: Adhere to FastAPI best practices (Annotated parameters, no ellipsis in defaults, asynchronous path routing with threadpools for blocking computer vision workloads).
- Code Cleanliness: Strictly zero emojis in code, comments, and HTML templates across all files.
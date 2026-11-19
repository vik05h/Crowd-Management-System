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
- Security Hardening Standards: Always validate file upload extensions (`.mp4`, `.avi`, `.mov`, `.mkv`), stream uploads in chunks with size limits (<= 250MB), validate external webhook URLs against SSRF (block private RFC1918, loopback, and metadata ranges), and enforce path traversal containment (`os.path.commonpath` / `startswith`).
- API Design: Adhere to FastAPI best practices (Annotated parameters, no ellipsis in defaults, asynchronous path routing with threadpools for blocking computer vision workloads).
- Code Cleanliness: Strictly zero emojis in code, comments, and HTML templates across all files.
- UI/UX & Component Standards: The platform maintains an authentic Neo-Brutalism UI design system (inspired by neo-brutalism-ui-library.vercel.app): bold black outlines (`border: 2px solid #000` / `border: 3px solid #000` / `border: 4px solid #000`), signature hard non-blurred offset drop shadows (`box-shadow: 4px 4px 0px #000`, `6px 6px 0px #000`, `8px 8px 0px #000`), tactile mechanical click states (`active: translate(2px, 2px) box-shadow: 0px 0px 0px #000`), punchy high-contrast pastel color blocks (`#FFE500` Yellow, `#A6FAFF` Cyan, `#B8FF9F` Lime, `#FFA6F6` Pink, `#FFC29F` Orange, `#FF9F9F` Red), clean paper canvas (`#FAF7F2`), technical typography (Space Grotesk, Inter, JetBrains Mono), Kit Langton rolling decimal/digit reel animations for numerical telemetry, getlayers.ai fluid follower cursor with lerp kinematics, and Vengeance UI specular light coordinate tracking on buttons.
- Spatial Safety & Pose Standards: Integrate Ultralytics YOLO26 Pose (default: yolo26n-pose.pt with hot-swapping to yolo26s-pose.pt and yolo26m-pose.pt) with temporal stride frame-skipping (stride=2 or 3) and 17 COCO landmarks. Apply sustained-event confirmation logic (horizontal spine angle > 52 deg for >= 1.6s for falls; motionless in fallen pose for >= 6.0s for collapse/unresponsive; stride velocity spikes for panic). Render skeletal bones with solid drop outlines.
- Polygonal Security Zones & Live Calibration: Define arbitrary multi-vertex sectors with `cv2.pointPolygonTest` on foot coordinates `(cx, y_feet)`. Compute per-zone occupancy, individual dwell times, and sustained capacity breaches. Support interactive on-stream canvas calibration mode on `/live_preview` allowing drag-and-drop repositioning of vertex handles directly over the live camera stream with live database sync (`PUT /api/zones/{id}/polygon`).
- Hardware Acceleration & Processing Profiles: Implement FP16 half-precision execution (`half=True`) on Tensor Cores across detection and pose pipelines. Provide streamlined processing presets (Eco: 640px, Stride 2; Balanced Quality: 640px, Stride 1) without misleading GPU-heavy terminology, and background TensorRT engine export with safe PyTorch FP16 fallbacks.
- Real-Time Telemetry Hub: Broadcast live telemetry, zone updates, and pose incidents via WebSocket (`/ws/events`) with SSE fallback (`/api/events/stream`).
- Authentication & DB Standards: Persist security events, zones, occupancy logs, and telemetry snapshots in SQLite via `aiosqlite`. Protect configuration routes with PBKDF2 password hashing and signed HS256 JWT tokens.
- Multi-Camera Surveillance CCTV Standards: Maintain dedicated multi-camera monitoring at `/cameras` (`GET /api/cameras`) supporting selectable 2x2 grid and 1+3 focus modes, simulated camera streams with live timecode/scanlines, and aggregate headcount calculation.
- Privacy & Identity Anonymization (DPDPA 2023 / GDPR): Maintain real-time face blurring in `cms/privacy/face_blur.py` utilizing YOLO-Pose keypoints and head geometry with dual-pass Gaussian blur and discrete on-stream privacy compliance watermark, togglable on the fly across surveillance streams and batch processing jobs.
- Data Export Standards: Maintain RFC 4180 CSV generation at `GET /api/export/csv` with explicit `Content-Disposition: attachment; filename=...` and no-cache headers for incidents, trends, and sectors.
- Staged Incident Simulation: Provide instant emergency drill simulation at `POST /api/incidents/simulate` triggering the full alert lifecycle (audio siren, visual banner, annotated snapshot, WebSocket broadcast, and SQLite persistence) for live demonstrations.
- Memory & Throughput Architecture: Always pre-allocate fixed numpy frame buffers (`np.copyto`), maintain density accumulation on quarter-resolution grids (`height//4, width//4`), batch-extract GPU bounding box tensors into numpy once per frame (`xyxy`, `conf`, `cls`, `id`), lazy-load and unload secondary neural networks (`PoseEngine.unload_model`) to conserve host RAM, eliminate unnecessary `.copy()` allocations in stream generation loops, and avoid artificial pacing sleeps when compute exceeds source frame interval.

## AI Agent Tooling (Claude Code, Codex, Antigravity)
All agent tooling is installed for this project only. The full install record, maintenance commands and known issues are in README.md under "AI Agent Tooling".
- Skills live in `.agents/skills/` (read natively by Codex and Antigravity; Claude Code reads the symlinks in `.claude/skills/`) and are tracked in `skills-lock.json`. Add or update them with `npx skills add <owner/repo> --skill <name> -a claude-code -a codex -a antigravity -y` and `npx skills update`. Never hand-edit vendored skill folders.
- Installed skill sets: Gravitas (`gravitas`, `gravitas-highstakes`, `gravitas-python`, `gravitas-typescript`), ai-design-skills (`landing-page-design`, `figma-designer`), `hallmark`, `lemo-opuscar`, `obscura`, ponytail (`ponytail`, `ponytail-review`, `ponytail-audit`, `ponytail-debt`, `ponytail-gain`, `ponytail-help`) and superpowers (15 skills, e.g. `brainstorming`, `writing-plans`, `systematic-debugging`, `test-driven-development`, `verification-before-completion`).
- Claude Code gets ponytail and superpowers as project-scoped plugins (`.claude/settings.json`), so they are namespaced there (`/ponytail:ponytail-review`, `superpowers:brainstorming`). Codex and Antigravity use the copies in `.agents/skills/`.
- Web access: the read-only `agent-reach` CLI (GitHub, YouTube, Bilibili, V2EX, web, X, Reddit, LinkedIn, Instagram, Facebook, Xiaohongshu, Xueqiu, Boss, Xiaoyuzhou; skill copy for Antigravity in `.agents/external-skills/`, declared by `.agents/skills.json`) and the `obscura` headless browser MCP server (`.mcp.json` for Claude Code, `.codex/config.toml` for Codex, `.agents/plugins/obscura-mcp/` for Antigravity).
- Design skills never override the Neo-Brutalism UI standard above. Treat that system as locked when `hallmark`, `landing-page-design` or `figma-designer` load, unless the user explicitly asks for a redesign.
- FreeLLMAPI (local free-tier LLM router) may only be wired in as an MCP tool. Never reroute an agent's main model through it.

## Ponytail Coding Mode (always-on)
Adapted from the `AGENTS.md` of DietrichGebert/ponytail. The project rules above take precedence where they conflict: documentation updates (README.md, MEMORY.md, AGENTS.md), security standards and the zero-emoji rule are always in scope.

You are a lazy senior developer. The best code is the code never written. You solve the whole problem with the least new code. End your reply with one or two lines: what you skipped or did not check, and any risk the user must know.

### Before you write
Read the task and the code it touches. List every place your change must reach: callers, tests, fixtures, config, exports. Check what your change could break for users: data it would destroy or expose, callers that stop working. That is scope. Extra features are not.

### The smallest complete change
Take the first option that fully works:

1. Does it need to exist? Skip features, options and flexibility nobody asked for, and name them in one line. A vague request ("build me X") gets the smallest version that does the core job.
2. Already in this codebase (a helper, component, service, pattern)? Use it the way the surrounding code does.
3. Standard library or a platform feature? Use it, unless the project has its own. A house component beats a native widget.
4. An installed dependency? Use it. Never add a dependency for a few lines.
5. Can it be one line a reader gets at a glance? One line.
6. Otherwise: the minimum code that works.

- Be lazy about the solution, never about the change itself: finish every part the task needs, including the callers, tests and fixtures your change breaks.
- No abstraction, wrapper, type conversion, option, config, boilerplate or "for later" code nobody asked for. Keep values in the form the platform already gives you. Deletion beats addition. Keep the structure the codebase already has: its layers, interfaces and conventions.
- The shortest working diff wins, once you know everything it must touch. A one-liner that needs decoding is not short.
- Comment only the why the code cannot show, in one line.
- Bug fix: before you edit, grep every caller of the function you touch, then fix the root cause once in the shared code.
- Code you move or merge keeps its error handling and validation.
- Between options of equal size, take the one that is correct on edge cases.
- Lazy code without its check is unfinished: new non-trivial logic (a branch, a loop, a parser, money or security, or a whole new script or app) leaves one small test or an assert-based self-check. Trivial changes need none.
- A shortcut with a known limit gets a code comment in this form: `shortcut: <the limit>, <when to upgrade>`.

Never cut: validation at trust boundaries, error handling that prevents data loss, security, accessibility, the calibration real hardware needs, anything the user asked for.


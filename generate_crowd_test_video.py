import glob
import os
import cv2
import numpy as np

def create_crowd_test_video(
    output_path: str = "static/uploads/crowdhuman_multiscene.mp4",
    num_scenes: int = 10,
    fps: int = 25,
    seconds_per_scene: int = 3,
    target_width: int = 1280,
    target_height: int = 720,
):
    val_images = sorted(glob.glob("dataset/images/val/*.jpg"))
    if not val_images:
        print("[ERROR] No validation images found in dataset/images/val")
        return

    # Select scenes with diverse crowd density (sort by annotation count)
    scenes = []
    for img_path in val_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join("dataset/labels/val", f"{basename}.txt")
        count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        scenes.append((count, img_path))

    scenes.sort(key=lambda x: x[0], reverse=True)
    # Pick a rich spectrum: very dense (50+), medium (25-40), low (10-20)
    selected_scenes = [s[1] for s in scenes[5:5 + num_scenes]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    frames_per_scene = fps * seconds_per_scene

    print(f"[INFO] Compiling {len(selected_scenes)} diverse crowd scenes into {output_path}...")
    for scene_idx, img_path in enumerate(selected_scenes):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Apply gentle pan/zoom to simulate real camera motion
        for f in range(frames_per_scene):
            t = f / float(frames_per_scene)
            # Subtle zoom from 1.0 to 1.05
            scale = 1.0 + 0.05 * t
            nh = int(target_height * scale)
            nw = int(target_width * scale)

            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            # Center crop to target resolution
            dy = (nh - target_height) // 2
            dx = (nw - target_width) // 2
            frame = resized[dy:dy + target_height, dx:dx + target_width]

            writer.write(frame)

    writer.release()
    print(f"[SUCCESS] Test crowd video generated at {output_path} ({num_scenes * seconds_per_scene}s, 720p, {fps}fps)")

if __name__ == "__main__":
    create_crowd_test_video()

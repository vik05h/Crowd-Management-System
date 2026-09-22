import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import cv2


def sanitize_bounding_boxes(
    lines: List[str],
    min_width: float = 0.015,
    min_height: float = 0.025,
    min_area: float = 0.0004,
) -> List[str]:
    """Deduplicate bounding boxes and remove degenerate micro-boxes."""
    clean_lines: List[str] = []
    seen_boxes = set()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        # Keep class 0 (person)
        if class_id != 0:
            continue

        # Check bounds and minimum dimensions
        if width <= 0 or height <= 0:
            continue
        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
            continue
        if width < min_width or height < min_height:
            continue
        if (width * height) < min_area:
            continue

        # Coordinate rounding for deduplication (approx 1/1000th pixel precision)
        box_key = (
            round(x_center, 3),
            round(y_center, 3),
            round(width, 3),
            round(height, 3),
        )

        if box_key in seen_boxes:
            continue

        seen_boxes.add(box_key)
        clean_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return clean_lines


def build_sanitized_subset(
    src_dataset_dir: str = "dataset",
    output_dir: str = "dataset/clean_100",
    target_train_count: int = 100,
    target_val_count: int = 25,
    max_verification_samples: int = 6,
) -> Dict[str, int]:
    """Filter, clean, and sample a verified high-quality subset for fine-tuning."""
    src_path = Path(src_dataset_dir).resolve()
    out_path = Path(output_dir).resolve()

    train_img_src = src_path / "images" / "train"
    train_lbl_src = src_path / "labels" / "train"
    val_img_src = src_path / "images" / "val"
    val_lbl_src = src_path / "labels" / "val"

    train_img_dst = out_path / "images" / "train"
    train_lbl_dst = out_path / "labels" / "train"
    val_img_dst = out_path / "images" / "val"
    val_lbl_dst = out_path / "labels" / "val"
    sample_vis_dst = out_path / "verification_samples"

    for directory in [train_img_dst, train_lbl_dst, val_img_dst, val_lbl_dst, sample_vis_dst]:
        directory.mkdir(parents=True, exist_ok=True)

    def process_split(
        img_src: Path,
        lbl_src: Path,
        img_dst: Path,
        lbl_dst: Path,
        target_count: int,
        is_train: bool = True,
    ) -> Tuple[int, int]:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_candidates = sorted([p for p in img_src.iterdir() if p.suffix.lower() in valid_exts])
        
        saved_images = 0
        total_retained_boxes = 0
        verification_saved = 0

        for img_file in image_candidates:
            lbl_file = lbl_src / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue

            with open(lbl_file, "r", encoding="utf-8") as f:
                raw_lines = f.readlines()

            sanitized_lines = sanitize_bounding_boxes(raw_lines)
            # Require at least 2 clean valid person annotations per image
            if len(sanitized_lines) < 2:
                continue

            # Copy image and write cleaned annotations
            shutil.copy2(img_file, img_dst / img_file.name)
            out_lbl_file = lbl_dst / f"{img_file.stem}.txt"
            with open(out_lbl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(sanitized_lines) + "\n")

            saved_images += 1
            total_retained_boxes += len(sanitized_lines)

            # Generate visual verification images for initial batch
            if is_train and verification_saved < max_verification_samples:
                img = cv2.imread(str(img_file))
                if img is not None:
                    h, w = img.shape[:2]
                    for s_line in sanitized_lines:
                        p = s_line.split()
                        xc, yc, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                        x1 = int((xc - bw / 2) * w)
                        y1 = int((yc - bh / 2) * h)
                        x2 = int((xc + bw / 2) * w)
                        y2 = int((yc + bh / 2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    vis_path = sample_vis_dst / f"verify_{img_file.stem}.jpg"
                    cv2.imwrite(str(vis_path), img)
                    verification_saved += 1

            if saved_images >= target_count:
                break

        return saved_images, total_retained_boxes

    print("[INFO] Processing and sanitizing training split...")
    train_saved, train_boxes = process_split(
        train_img_src, train_lbl_src, train_img_dst, train_lbl_dst, target_train_count, is_train=True
    )

    print("[INFO] Processing and sanitizing validation split...")
    val_saved, val_boxes = process_split(
        val_img_src, val_lbl_src, val_img_dst, val_lbl_dst, target_val_count, is_train=False
    )

    yaml_path = out_path / "data_clean.yaml"
    posix_path = out_path.as_posix()
    yaml_content = (
        f"path: {posix_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['person']\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[SUCCESS] Cleaned dataset generated at: {out_path}")
    print(f"  - Train: {train_saved} images ({train_boxes} clean person boxes)")
    print(f"  - Val:   {val_saved} images ({val_boxes} clean person boxes)")
    print(f"  - Config YAML: {yaml_path}")

    return {
        "train_images": train_saved,
        "train_boxes": train_boxes,
        "val_images": val_saved,
        "val_boxes": val_boxes,
    }


if __name__ == "__main__":
    build_sanitized_subset()

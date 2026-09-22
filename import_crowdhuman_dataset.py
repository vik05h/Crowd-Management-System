import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


def import_and_convert_crowdhuman(
    odgt_path: str = r"C:\Users\Vikash Kumar\Downloads\annotation_val.odgt",
    images_dir: str = r"C:\Users\Vikash Kumar\Downloads\CrowdHuman_va\Images",
    target_dataset_dir: str = "dataset",
    train_ratio: float = 0.80,
    seed: int = 42,
) -> Dict[str, int]:
    """Convert CrowdHuman annotations to standard YOLO format and place in dataset/."""
    odgt_file = Path(odgt_path)
    src_images = Path(images_dir)
    target_root = Path(target_dataset_dir).resolve()

    if not odgt_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {odgt_file}")
    if not src_images.exists():
        raise FileNotFoundError(f"Source images folder not found: {src_images}")

    # Archive old dataset images and labels if present
    legacy_dir = target_root / "legacy_dot_dataset"
    old_img_dir = target_root / "images"
    old_lbl_dir = target_root / "labels"

    if old_img_dir.exists() and not legacy_dir.exists():
        legacy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Archiving previous dataset to: {legacy_dir}")
        if old_img_dir.exists():
            shutil.move(str(old_img_dir), str(legacy_dir / "images"))
        if old_lbl_dir.exists():
            shutil.move(str(old_lbl_dir), str(legacy_dir / "labels"))
    elif old_img_dir.exists():
        # Clear out current images and labels to ensure clean replacement
        shutil.rmtree(str(old_img_dir), ignore_errors=True)
        shutil.rmtree(str(old_lbl_dir), ignore_errors=True)

    train_img_dst = target_root / "images" / "train"
    val_img_dst = target_root / "images" / "val"
    train_lbl_dst = target_root / "labels" / "train"
    val_lbl_dst = target_root / "labels" / "val"

    for d in [train_img_dst, val_img_dst, train_lbl_dst, val_lbl_dst]:
        d.mkdir(parents=True, exist_ok=True)

    # Read and parse ODGT records
    print("[INFO] Parsing CrowdHuman ODGT annotations...")
    records = []
    with open(odgt_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    print(f"[INFO] Loaded {len(records)} image records from {odgt_file.name}")

    # Set random seed and shuffle records
    random.seed(seed)
    random.shuffle(records)

    split_idx = int(len(records) * train_ratio)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    def process_records(
        recs: List[dict],
        img_out: Path,
        lbl_out: Path,
        split_name: str,
    ) -> Tuple[int, int]:
        saved_imgs = 0
        saved_boxes = 0

        for i, rec in enumerate(recs):
            img_id = rec.get("ID")
            if not img_id:
                continue

            src_img_file = src_images / f"{img_id}.jpg"
            if not src_img_file.exists():
                continue

            # Read image dimensions quickly without full raster decode
            try:
                with Image.open(src_img_file) as img:
                    img_w, img_h = img.size
            except Exception as e:
                print(f"[WARN] Error reading image header {src_img_file.name}: {e}")
                continue

            gtboxes = rec.get("gtboxes", [])
            yolo_lines = []

            for box in gtboxes:
                if box.get("tag") != "person":
                    continue

                extra = box.get("extra", {})
                if extra.get("ignore", 0) == 1:
                    continue

                fbox = box.get("fbox")
                if not fbox or len(fbox) < 4:
                    continue

                x, y, w, h = fbox
                x1 = max(0.0, min(float(img_w), float(x)))
                y1 = max(0.0, min(float(img_h), float(y)))
                x2 = max(0.0, min(float(img_w), float(x + w)))
                y2 = max(0.0, min(float(img_h), float(y + h)))

                bw = x2 - x1
                bh = y2 - y1

                # Filter out degenerate slivers
                if bw < 4 or bh < 8:
                    continue

                # Normalized YOLO coordinates: class 0 (person)
                xc = (x1 + x2) / 2.0 / img_w
                yc = (y1 + y2) / 2.0 / img_h
                nw = bw / img_w
                nh = bh / img_h

                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            # Write label file
            dst_lbl_file = lbl_out / f"{img_id}.txt"
            with open(dst_lbl_file, "w", encoding="utf-8") as lf:
                if yolo_lines:
                    lf.write("\n".join(yolo_lines) + "\n")

            # Copy image file
            dst_img_file = img_out / f"{img_id}.jpg"
            shutil.copy2(src_img_file, dst_img_file)

            saved_imgs += 1
            saved_boxes += len(yolo_lines)

            if (i + 1) % 500 == 0 or (i + 1) == len(recs):
                print(f"[{split_name.upper()}] Processed {i + 1}/{len(recs)} images ({saved_boxes} person boxes)...")

        return saved_imgs, saved_boxes

    print("\n[INFO] Converting and placing training set...")
    train_imgs, train_boxes = process_records(train_records, train_img_dst, train_lbl_dst, "train")

    print("\n[INFO] Converting and placing validation set...")
    val_imgs, val_boxes = process_records(val_records, val_img_dst, val_lbl_dst, "val")

    # Update dataset/data.yaml
    yaml_path = target_root / "data.yaml"
    posix_path = target_root.as_posix()
    yaml_content = (
        f"path: {posix_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['person']\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"\n[SUCCESS] New CrowdHuman dataset successfully installed in {target_root}!")
    print(f"  - Training Images:   {train_imgs} ({train_boxes} full-body person boxes)")
    print(f"  - Validation Images: {val_imgs} ({val_boxes} full-body person boxes)")
    print(f"  - Total People:      {train_boxes + val_boxes}")
    print(f"  - Config file:       {yaml_path}")

    return {
        "train_imgs": train_imgs,
        "train_boxes": train_boxes,
        "val_imgs": val_imgs,
        "val_boxes": val_boxes,
    }


if __name__ == "__main__":
    import_and_convert_crowdhuman()

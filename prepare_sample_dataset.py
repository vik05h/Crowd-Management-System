import os
import shutil
from pathlib import Path


def create_sample_dataset(
    base_dir: str = "dataset",
    sample_dir: str = "dataset/sample",
    num_train: int = 50,
    num_val: int = 15,
) -> str:
    """Create a sample dataset slice for local sanity-check training.

    Args:
        base_dir: Path to the root dataset folder containing images and labels.
        sample_dir: Path to the target sample directory.
        num_train: Number of training images to sample.
        num_val: Number of validation images to sample.

    Returns:
        The absolute path to the generated data_sample.yaml file.
    """
    base_path = Path(base_dir).resolve()
    sample_path = Path(sample_dir).resolve()

    train_img_src = base_path / "images" / "train"
    val_img_src = base_path / "images" / "val"
    train_lbl_src = base_path / "labels" / "train"
    val_lbl_src = base_path / "labels" / "val"

    train_img_dst = sample_path / "images" / "train"
    val_img_dst = sample_path / "images" / "val"
    train_lbl_dst = sample_path / "labels" / "train"
    val_lbl_dst = sample_path / "labels" / "val"

    for directory in [train_img_dst, val_img_dst, train_lbl_dst, val_lbl_dst]:
        directory.mkdir(parents=True, exist_ok=True)

    def copy_subset(img_src_dir: Path, lbl_src_dir: Path, img_dst_dir: Path, lbl_dst_dir: Path, count: int) -> int:
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        candidates = sorted([p for p in img_src_dir.iterdir() if p.suffix.lower() in valid_extensions])
        copied = 0

        for img_file in candidates:
            lbl_file = lbl_src_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(img_file, img_dst_dir / img_file.name)
                shutil.copy2(lbl_file, lbl_dst_dir / lbl_file.name)
                copied += 1
                if copied >= count:
                    break

        return copied

    train_copied = copy_subset(train_img_src, train_lbl_src, train_img_dst, train_lbl_dst, num_train)
    val_copied = copy_subset(val_img_src, val_lbl_src, val_img_dst, val_lbl_dst, num_val)

    print(f"[INFO] Sample dataset prepared: {train_copied} train images, {val_copied} val images.")

    yaml_path = sample_path / "data_sample.yaml"
    posix_sample_path = sample_path.as_posix()
    yaml_content = (
        f"path: {posix_sample_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['person']\n"
    )

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[INFO] Sample YAML created at: {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    create_sample_dataset()

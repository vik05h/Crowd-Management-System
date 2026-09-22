import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def run_sanity_check_training(
    data_yaml: str = "dataset/sample/data_sample.yaml",
    model_name: str = "yolo26m.pt",
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 8,
    project: str = "runs/detect",
    name: str = "yolo26m_sanity",
) -> None:
    """Run a small-scale sanity-check training run for YOLO26m.

    Args:
        data_yaml: Path to dataset YAML definition.
        model_name: Base model weight file or architecture identifier.
        epochs: Number of epochs to train.
        imgsz: Image resolution for training.
        batch: Batch size per iteration.
        project: Directory where training outputs are saved.
        name: Name of this specific training run.
    """
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {yaml_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing YOLO26 training on device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[INFO] GPU: {gpu_name} (Total VRAM: {vram_mb:.0f} MB)")

    model = YOLO(model_name)

    print(f"[INFO] Starting sanity training for {epochs} epochs on {yaml_path.as_posix()}...")
    results = model.train(
        data=yaml_path.as_posix(),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        optimizer="auto",
        amp=True,
        verbose=True,
    )

    print("[INFO] Sanity training completed.")
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        print(f"[SUCCESS] Trained weights saved at: {best_weights.resolve()}")
        # Also ensure runs/detect/yolo26m_sanity/weights/best.pt is populated
        canonical_target = Path("runs/detect/yolo26m_sanity/weights/best.pt")
        if canonical_target.resolve() != best_weights.resolve():
            canonical_target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(best_weights, canonical_target)
            print(f"[INFO] Synced weights to canonical path: {canonical_target.resolve()}")
    else:
        print(f"[WARN] Expected weights not found at: {best_weights.resolve()}")


if __name__ == "__main__":
    run_sanity_check_training()

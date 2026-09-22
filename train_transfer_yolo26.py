import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def run_transfer_training(
    data_yaml: str = "dataset/clean_100/data_clean.yaml",
    model_name: str = "yolo26m.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch: int = 8,
    freeze: int = 10,
    project: str = "runs/detect",
    name: str = "yolo26m_clean_transfer",
) -> Path:
    """Run lightweight backbone-frozen transfer learning on YOLO26m."""
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cleaned dataset YAML not found at: {yaml_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing YOLO26 transfer learning on device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[INFO] Detected GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    model = YOLO(model_name)

    print(
        f"[INFO] Starting transfer training for {epochs} epochs on {yaml_path.as_posix()} "
        f"(freeze={freeze}, batch={batch}, imgsz={imgsz})..."
    )

    results = model.train(
        data=yaml_path.as_posix(),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        device=device,
        workers=2,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        amp=True,
        cos_lr=True,
        patience=10,
        dropout=0.15,
        weight_decay=0.001,
        verbose=True,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if best_weights.exists():
        print(f"[SUCCESS] Best transfer weights saved at: {best_weights.resolve()}")
        # Ensure standard fallback path is also synced
        canonical_target = Path("runs/detect/yolo26m_custom/weights/best.pt")
        canonical_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, canonical_target)
        print(f"[INFO] Synced weights to canonical fallback path: {canonical_target.resolve()}")
        return best_weights
    else:
        print(f"[WARN] Expected best weights not found at: {best_weights.resolve()}")
        return Path(model_name)


if __name__ == "__main__":
    run_transfer_training()

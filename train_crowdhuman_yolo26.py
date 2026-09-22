import argparse
import os
import shutil
import torch
from ultralytics import YOLO


def train(
    data_yaml: str = "dataset/data.yaml",
    model_name: str = "yolo26m.pt",
    epochs: int = 30,
    batch_size: int = 8,
    imgsz: int = 640,
    freeze: int = 10,
    patience: int = 15,
    lr0: float = 0.001,
    weight_decay: float = 0.001,
    dropout: float = 0.15,
    project: str = None,
    name: str = "yolo26m_crowdhuman",
    run_eval: bool = True,
):
    if project is None:
        project = os.path.abspath("runs/detect")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing training on device={device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        freeze=freeze,
        patience=patience,
        optimizer="AdamW",
        lr0=lr0,
        lrf=0.01,
        cos_lr=True,
        weight_decay=weight_decay,
        dropout=dropout,
        amp=True,
        workers=2,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
    )

    print("[SUCCESS] Training finished successfully.")
    save_dir = getattr(results, "save_dir", os.path.join(project, name))
    best_weights = os.path.join(str(save_dir), "weights", "best.pt")
    target_canonical = os.path.join(project, "yolo26m_crowdhuman", "weights", "best.pt")

    if os.path.exists(best_weights):
        print(f"[INFO] Best model weights located at: {best_weights}")
        if os.path.abspath(best_weights) != os.path.abspath(target_canonical):
            os.makedirs(os.path.dirname(target_canonical), exist_ok=True)
            shutil.copy2(best_weights, target_canonical)
            print(f"[INFO] Synced best weights to canonical path: {target_canonical}")
    elif os.path.exists(target_canonical):
        best_weights = target_canonical
        print(f"[INFO] Using existing canonical weights: {target_canonical}")

    if run_eval and os.path.exists(best_weights):
        print("\n[INFO] Launching automated statistical benchmark evaluation...")
        from evaluate_crowd_metrics import evaluate_model_on_val, print_comparison_table
        b_det, b_reg = evaluate_model_on_val("yolo26m.pt", max_samples=150)
        m_det, m_reg = evaluate_model_on_val(best_weights, max_samples=150)
        print_comparison_table("Stock YOLO26m", b_det, b_reg, f"YOLO26m ({epochs}ep)", m_det, m_reg)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26m on CrowdHuman dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--model", type=str, default="yolo26m.pt", help="Base model weights")
    parser.add_argument("--no-eval", action="store_true", help="Skip post-training evaluation")
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze,
        patience=args.patience,
        run_eval=not args.no_eval,
    )


import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO26m model on crowd dataset with anti-overfitting regularizations."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to data.yaml dataset definition",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26m.pt",
        help="YOLO26 model architecture/checkpoint (e.g., yolo26m.pt, yolo26s.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without validation improvement)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (use 16 on Cloud GPUs like A100/T4, or 4-8 on local GPUs)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image size for training and inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on (e.g., '0', '0,1', 'cpu', or blank for auto)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
        help="Dropout regularization rate to prevent co-adaptation",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="L2 weight decay regularization",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save training results",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo26m_custom",
        help="Name of the training run experiment",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    device = args.device
    if not device:
        device = "0" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Initializing YOLO26 training with base model: {args.model}")
    print(f"[INFO] Selected device: {device}")
    if torch.cuda.is_available() and device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"[INFO] GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    model = YOLO(args.model)

    print("[INFO] Applying anti-overfitting training configuration:")
    print(f"       - Epochs (Max): {args.epochs}")
    print(f"       - Early Stopping Patience: {args.patience}")
    print(f"       - Dropout: {args.dropout}")
    print(f"       - Weight Decay: {args.weight_decay}")
    print(f"       - Cosine LR Schedule: Enabled")
    print(f"       - Batch Size: {args.batch}")
    print(f"       - Image Size: {args.imgsz}")

    train_kwargs = {
        "data": data_path.as_posix(),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "auto",
        "cos_lr": True,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "mosaic": 0.5,
        "mixup": 0.1,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "close_mosaic": 10,
        "amp": True,
        "plots": True,
        "save": True,
        "save_period": 10,
        "verbose": True,
    }

    try:
        results = model.train(**train_kwargs)
        print("[INFO] Training finished successfully.")
        output_dir = Path(args.project) / args.name
        best_pt = output_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"[SUCCESS] Best fine-tuned model saved at: {best_pt.resolve()}")
            print("[INFO] Copy this model to your CMS application path.")
    except Exception as exc:
        print(f"[ERROR] Training failed with exception: {exc}")
        raise


if __name__ == "__main__":
    main()

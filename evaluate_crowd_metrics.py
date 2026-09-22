import argparse
import glob
import math
import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def compute_regression_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute count regression statistics: R2, RMSE, MAE, MAPE, Bias."""
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)
    n = len(y_t)
    if n == 0:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0, "mape": 0.0, "bias": 0.0}

    diff = y_p - y_t
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(math.sqrt(mse))
    bias = float(np.mean(diff))

    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    nonzero = y_t > 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs(diff[nonzero]) / y_t[nonzero]) * 100.0)
    else:
        mape = 0.0

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
        "n_samples": n,
        "gt_mean": float(np.mean(y_t)),
        "pred_mean": float(np.mean(y_p)),
    }


def evaluate_model_on_val(
    model_path: str,
    val_images_dir: str = "dataset/images/val",
    val_labels_dir: str = "dataset/labels/val",
    conf_threshold: float = 0.25,
    max_samples: int = 150,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate both Ultralytics detection metrics and count regression statistics."""
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model {model_path} on device={device}...")
    model = YOLO(model_path)

    # 1. Run Ultralytics val
    print(f"[INFO] Running YOLO val() for {model_path}...")
    try:
        val_results = model.val(
            data="dataset/data.yaml",
            split="val",
            imgsz=640,
            batch=8,
            device=device,
            verbose=False,
            plots=False,
        )
        p = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
        r = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
        map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        det_metrics = {
            "precision": p,
            "recall": r,
            "map50": map50,
            "map50_95": map50_95,
        }
    except Exception as exc:
        print(f"[WARN] Validation call failed: {exc}")
        det_metrics = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}

    # 2. Run Headcount Evaluation
    img_files = sorted(glob.glob(os.path.join(val_images_dir, "*.jpg")))
    if max_samples > 0 and len(img_files) > max_samples:
        img_files = img_files[:max_samples]

    y_true: List[int] = []
    y_pred: List[int] = []

    print(f"[INFO] Evaluating headcount metrics on {len(img_files)} validation images...")
    for idx, img_path in enumerate(img_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(val_labels_dir, f"{basename}.txt")

        # Ground truth count
        gt_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                gt_count = sum(1 for line in f if line.strip())

        # Predicted count
        res = model.predict(
            source=img_path,
            conf=conf_threshold,
            classes=[0],
            device=device,
            verbose=False,
            imgsz=640,
        )
        pred_count = len(res[0].boxes) if len(res) > 0 and res[0].boxes is not None else 0

        y_true.append(gt_count)
        y_pred.append(pred_count)

    reg_metrics = compute_regression_metrics(y_true, y_pred)
    return det_metrics, reg_metrics


def print_comparison_table(
    baseline_name: str,
    baseline_det: Dict[str, float],
    baseline_reg: Dict[str, float],
    model_name: str,
    model_det: Dict[str, float],
    model_reg: Dict[str, float],
):
    print("\n" + "=" * 78)
    print(" CROWD DETECTION & COUNTING BENCHMARK REPORT ")
    print("=" * 78)
    print(f"{'Metric':<30} | {baseline_name:<20} | {model_name:<20}")
    print("-" * 78)
    print(f"{'Count R2 Score':<30} | {baseline_reg['r2']:<20.4f} | {model_reg['r2']:<20.4f}")
    print(f"{'Count RMSE (people)':<30} | {baseline_reg['rmse']:<20.2f} | {model_reg['rmse']:<20.2f}")
    print(f"{'Count MAE (people)':<30} | {baseline_reg['mae']:<20.2f} | {model_reg['mae']:<20.2f}")
    print(f"{'Count MAPE (%)':<30} | {baseline_reg['mape']:<20.2f} | {model_reg['mape']:<20.2f}")
    print(f"{'Count Bias (mean delta)':<30} | {baseline_reg['bias']:<20.2f} | {model_reg['bias']:<20.2f}")
    print("-" * 78)
    print(f"{'mAP@0.50 (%)':<30} | {baseline_det['map50']*100:<20.2f} | {model_det['map50']*100:<20.2f}")
    print(f"{'mAP@0.50:0.95 (%)':<30} | {baseline_det['map50_95']*100:<20.2f} | {model_det['map50_95']*100:<20.2f}")
    print(f"{'Detection Precision (%)':<30} | {baseline_det['precision']*100:<20.2f} | {model_det['precision']*100:<20.2f}")
    print(f"{'Detection Recall (%)':<30} | {baseline_det['recall']*100:<20.2f} | {model_det['recall']*100:<20.2f}")
    print("-" * 78)
    print(f"{'Evaluated Test Images':<30} | {baseline_reg['n_samples']:<20} | {model_reg['n_samples']:<20}")
    print(f"{'Ground Truth Mean Count':<30} | {baseline_reg['gt_mean']:<20.2f} | {model_reg['gt_mean']:<20.2f}")
    print(f"{'Predicted Mean Count':<30} | {baseline_reg['pred_mean']:<20.2f} | {model_reg['pred_mean']:<20.2f}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate crowd detection & regression statistics.")
    parser.add_argument("--baseline", type=str, default="yolo26m.pt", help="Baseline weights")
    parser.add_argument("--model", type=str, default="runs/detect/yolo26m_crowdhuman/weights/best.pt", help="Trained weights")
    parser.add_argument("--samples", type=int, default=150, help="Max val images for headcount")
    args = parser.parse_args()

    b_det, b_reg = evaluate_model_on_val(args.baseline, max_samples=args.samples)
    m_det, m_reg = evaluate_model_on_val(args.model, max_samples=args.samples)
    print_comparison_table("Stock YOLO26m", b_det, b_reg, "CrowdHuman YOLO26m", m_det, m_reg)

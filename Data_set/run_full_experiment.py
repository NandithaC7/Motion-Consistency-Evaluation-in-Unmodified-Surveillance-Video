r"""
run_full_experiment.py
-----------------------
Master execution script for Motion Consistency Evaluation in Unmodified Surveillance Video.

Executes all 6 stages of the methodology pipeline:
  - Stage 1 & Guard Layers (G1 Pixel Variance, G2 SSIM)
  - Stage 2: Clip Builder (16-frame sliding windows)
  - Stage 3: Dual PyTorch LSTM Autoencoder Training (Stream A Frame Diff & Stream B Optical Flow)
  - Stage 4: Divergence Anomaly Scoring & Grid Search (lambda in {0.1, 0.5, 1.0, 2.0})
  - Stage 5: Dual-Window Rolling Adaptive Thresholding (theta_s N_s=20, theta_l N_l=100)
  - Stage 6: Classification & Feed Reliability Score R in [0, 1]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path so 'src' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import project modules from src
from src.models.lstm_autoencoder import DualStreamLSTMAutoencoder
from src.evaluation.anomaly_scorer import (
    compute_divergence_anomaly_score,
    grid_search_lambda,
    DualWindowAdaptiveThreshold,
    classify_clips_and_compute_reliability,
)
from Data_set.preprocess_pipeline import GuardG1Variance, GuardG2SSIM, PipelineConfig


def load_stream_data(output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path_a = output_dir / "1_stream_A_frame_diff.npy"
    path_b = output_dir / "1_stream_B_optical_flow.npy"

    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(
            f"Pre-computed stream numpy files not found in {output_dir}. "
            f"Expected {path_a.name} and {path_b.name}."
        )

    stream_a = np.load(path_a).astype(np.float32)
    stream_b = np.load(path_b).astype(np.float32)
    print(f"[Data Loader] Loaded Stream A: shape={stream_a.shape}, dtype={stream_a.dtype}")
    print(f"[Data Loader] Loaded Stream B: shape={stream_b.shape}, dtype={stream_b.dtype}")
    return stream_a, stream_b


def run_guard_checks(stream_a: np.ndarray, config: PipelineConfig) -> List[str]:
    """Run Guard G1 (Variance) and G2 (SSIM) on spatial frame sequences."""
    g1 = GuardG1Variance(config)
    g2 = GuardG2SSIM(config)

    N = stream_a.shape[0]
    guard_statuses = []

    for i in range(N):
        window_frames = [stream_a[i, t] for t in range(stream_a.shape[1])]
        res_g1 = g1.check_window(window_frames)
        if res_g1["status"] == "FROZEN":
            guard_statuses.append("FROZEN")
            continue

        res_g2 = g2.check_window(window_frames)
        if res_g2["status"] == "FROZEN":
            guard_statuses.append("FROZEN")
            continue

        guard_statuses.append("PASS")

    print(f"[Guard Layers] Guard evaluations completed for {N} clips. "
          f"PASS: {guard_statuses.count('PASS')}, FROZEN: {guard_statuses.count('FROZEN')}")
    return guard_statuses


def plot_results(
    history_a: Any,
    history_b: Any,
    err_a: np.ndarray,
    err_b: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    classifications: List[str],
    reliability_R: float,
    output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history_a.history["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history_a.history and history_a.history["val_loss"]:
        ax1.plot(history_a.history["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--", linewidth=2)
    ax1.set_title("Stream A (Frame Diff) LSTM-AE Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history_b.history["loss"], label="Train Loss", color="#2ca02c", linewidth=2)
    if "val_loss" in history_b.history and history_b.history["val_loss"]:
        ax2.plot(history_b.history["val_loss"], label="Val Loss", color="#d62728", linestyle="--", linewidth=2)
    ax2.set_title("Stream B (Optical Flow) LSTM-AE Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path1 = output_dir / "training_loss_curves.png"
    plt.savefig(fig_path1, dpi=200)
    plt.close()

    # 2. Reconstruction Errors Distribution (Stream A vs Stream B)
    plt.figure(figsize=(9, 6))
    plt.scatter(err_a, err_b, c="#3498db", alpha=0.7, edgecolors="k", s=40, label="Clips")
    plt.axline((0, 0), slope=1, color="red", linestyle=":", label="1:1 Equality Line")
    plt.title("Reconstruction Error Comparison (Stream A vs Stream B)", fontsize=13, fontweight="bold")
    plt.xlabel("Stream A Reconstruction Error (e_A)", fontsize=11)
    plt.ylabel("Stream B Reconstruction Error (e_B)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path2 = output_dir / "reconstruction_errors_distribution.png"
    plt.savefig(fig_path2, dpi=200)
    plt.close()

    # 3. Stage 4 & 5: Divergence Score & Adaptive Threshold
    plt.figure(figsize=(12, 5))
    clip_indices = np.arange(len(scores))
    plt.plot(clip_indices, scores, label="Divergence Anomaly Score", color="#8e44ad", linewidth=1.8)
    plt.plot(clip_indices, thresholds, label="Dual-Window Adaptive Threshold (\\theta)", color="#e67e22", linestyle="--", linewidth=2.0)
    
    # Highlight irregular clips
    irregular_idx = [i for i, c in enumerate(classifications) if c == "IRREGULAR"]
    if irregular_idx:
        plt.scatter(irregular_idx, scores[irregular_idx], color="red", zorder=5, label="Irregular Anomaly")

    plt.title("Stage 4 & 5: Divergence Anomaly Score vs. Adaptive Threshold", fontsize=13, fontweight="bold")
    plt.xlabel("Clip Index (Time Window)", fontsize=11)
    plt.ylabel("Anomaly Score", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path3 = output_dir / "divergence_score_and_adaptive_threshold.png"
    plt.savefig(fig_path3, dpi=200)
    plt.close()

    # 4. Feed Reliability & Classification Summary
    plt.figure(figsize=(8, 5))
    counts = {
        "HEALTHY": classifications.count("HEALTHY"),
        "IRREGULAR": classifications.count("IRREGULAR"),
        "FROZEN": classifications.count("FROZEN")
    }
    colors = ["#2ecc71", "#e74c3c", "#34495e"]
    bars = plt.bar(list(counts.keys()), list(counts.values()), color=colors, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, str(yval), ha='center', va='bottom', fontweight='bold')

    plt.title(f"Stage 6: Clip Classifications (Feed Reliability Score R = {reliability_R:.4f})", fontsize=12, fontweight="bold")
    plt.xlabel("Classification Category", fontsize=11)
    plt.ylabel("Number of Clips", fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path4 = output_dir / "feed_reliability_summary.png"
    plt.savefig(fig_path4, dpi=200)
    plt.close()

    print(f"[Plotting] Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Motion Consistency Evaluation experiment")
    parser.add_argument("--output_dir", type=str, default="Data_set/processed_output", help="Directory containing stream numpy files")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs for LSTM-AE")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="LSTM hidden units")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    processed_dir = output_dir / "stage23"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" MOTION CONSISTENCY EVALUATION PIPELINE — FULL EXPERIMENT RUN")
    print("=" * 70)

    # 1. Load Stream Data
    stream_a, stream_b = load_stream_data(output_dir)
    N, T, H, W = stream_a.shape
    D = H * W  # 4096 spatial features per frame

    # Reshape (N, T, H, W) -> (N, T, D) for temporal LSTM modeling
    stream_a_flat = stream_a.reshape(N, T, D)
    stream_b_flat = stream_b.reshape(N, T, D)

    # 2. Run Guard Layers (G1 Variance, G2 SSIM)
    config = PipelineConfig()
    guard_statuses = run_guard_checks(stream_a, config)

    # 3. Train Stage 3 Dual PyTorch LSTM Autoencoders
    print(f"\n[Stage 3] Training Dual Stream PyTorch LSTM Autoencoders (Epochs={args.epochs}, Hidden={args.hidden_dim})...")
    model_a = DualStreamLSTMAutoencoder(seq_len=T, input_dim=D, hidden_dim=args.hidden_dim, num_layers=2, learning_rate=1e-3)
    model_b = DualStreamLSTMAutoencoder(seq_len=T, input_dim=D, hidden_dim=args.hidden_dim, num_layers=2, learning_rate=1e-3)

    history_a = model_a.fit(stream_a_flat, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)
    history_b = model_b.fit(stream_b_flat, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)

    # Reconstruction errors e_A and e_B
    err_a = model_a.reconstruction_error(stream_a_flat)
    err_b = model_b.reconstruction_error(stream_b_flat)

    print(f"  Stream A MSE Error: mean={err_a.mean():.6f}, std={err_a.std():.6f}")
    print(f"  Stream B MSE Error: mean={err_b.mean():.6f}, std={err_b.std():.6f}")

    # 4. Stage 4: Divergence Anomaly Score & Grid Search
    print("\n[Stage 4] Performing Grid Search on \\lambda \\in {0.1, 0.5, 1.0, 2.0}...")
    grid_res = grid_search_lambda(err_a, err_b, candidate_lambdas=[0.1, 0.5, 1.0, 2.0])
    best_lambda = grid_res["best_lambda"]
    print(f"  Optimal \\lambda selected: {best_lambda}")

    divergence_scores = compute_divergence_anomaly_score(err_a, err_b, lambda_param=best_lambda)

    # 5. Stage 5: Dual-Window Rolling Adaptive Thresholding
    print("\n[Stage 5] Computing Dual-Window Adaptive Thresholds (N_s=20, N_l=100)...")
    threshold_calc = DualWindowAdaptiveThreshold(short_window=20, long_window=100, k_std=2.0)
    adaptive_thresholds = threshold_calc.compute_thresholds(divergence_scores)

    # 6. Stage 6: Classification & Feed Reliability Score R
    print("\n[Stage 6] Classifying Clips & Calculating Feed Reliability Score R...")
    classifications, reliability_R, summary_dict = classify_clips_and_compute_reliability(
        guard_statuses, divergence_scores, adaptive_thresholds
    )

    print(f"  Feed Reliability Score R: {reliability_R:.4f}")
    print(f"  Clip Classifications: {summary_dict['class_counts']}")

    # Save trained models
    model_a.save(processed_dir / "stream_a_lstm_ae.pt")
    model_b.save(processed_dir / "stream_b_lstm_ae.pt")

    # Save Scores CSV
    csv_rows = []
    for i in range(N):
        csv_rows.append({
            "clip_id": i,
            "guard_status": guard_statuses[i],
            "stream_a_error": float(err_a[i]),
            "stream_b_error": float(err_b[i]),
            "divergence_score": float(divergence_scores[i]),
            "adaptive_threshold": float(adaptive_thresholds[i]),
            "classification": classifications[i]
        })

    csv_path = processed_dir / "train_reconstruction_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Save Grid Search Results CSV
    grid_rows = []
    for lam, meta in grid_res["grid_results"].items():
        grid_rows.append({
            "lambda": float(lam),
            "mean_score": meta["mean_score"],
            "std_score": meta["std_score"],
            "mean_divergence_term": meta["mean_divergence_term"],
            "variance_ratio": meta["variance_ratio"]
        })
    with open(processed_dir / "lambda_grid_search_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    # Save Run Summary JSON
    run_summary = {
        "dataset_clips": N,
        "sequence_len": T,
        "feature_dim": D,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "is_pytorch": model_a.is_pytorch,
        "stream_a_final_loss": float(history_a.history["loss"][-1]),
        "stream_b_final_loss": float(history_b.history["loss"][-1]),
        "optimal_lambda": float(best_lambda),
        "feed_reliability_score_R": round(reliability_R, 4),
        "class_counts": summary_dict["class_counts"],
        "models_saved": {
            "stream_a": str(processed_dir / "stream_a_lstm_ae.pt"),
            "stream_b": str(processed_dir / "stream_b_lstm_ae.pt"),
        },
        "output_dir": str(processed_dir)
    }

    with open(processed_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    # Plot and Save Visualizations
    plot_results(
        history_a, history_b, err_a, err_b,
        divergence_scores, adaptive_thresholds, classifications,
        reliability_R, processed_dir
    )

    print("\n" + "=" * 70)
    print(" SUCCESS: FULL EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f" Run summary written to: {processed_dir / 'run_summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

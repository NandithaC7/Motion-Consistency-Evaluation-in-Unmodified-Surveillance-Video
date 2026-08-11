r"""
anomaly_scorer.py
------------------
Stages 4, 5, & 6: Divergence Anomaly Scoring, Adaptive Thresholding, and Feed Reliability Score.
Project: Motion Consistency Evaluation in Unmodified Surveillance Video

Stage 4 - Divergence Anomaly Score (Novel Contribution):
    Score_i = e_A + e_B + lambda * |e_A - e_B|
    Measures reconstruction error between Stream A (Frame Diff) and Stream B (Optical Flow).
    The divergence term lambda * |e_A - e_B| catches single-stream failure modes.

Stage 5 - Dual-Window Adaptive Thresholding:
    Short window (N_s = 20):  theta_s = mu_s + 2*sigma_s (detects sudden faults)
    Long window (N_l = 100):  theta_l = mu_l + 2*sigma_l (detects gradual drift)
    Adaptive threshold:       theta = min(theta_s, theta_l)

Stage 6 - Classification Rule & Feed Reliability Score:
    Classification Rule:
        Class_i = FROZEN    if Guard G1 or G2 triggered
        Class_i = IRREGULAR if Score_i > theta
        Class_i = HEALTHY   otherwise
    Feed Reliability Score:
        R = 1 - (1 / N) * sum(Class_i != HEALTHY), R in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class AnomalyEvaluationResult:
    clip_id: int
    guard_status: str
    reconstruction_error_a: float
    reconstruction_error_b: float
    divergence_score: float
    adaptive_threshold: float
    classification: str


def compute_divergence_anomaly_score(
    error_a: Union[float, np.ndarray],
    error_b: Union[float, np.ndarray],
    lambda_param: float = 1.0
) -> Union[float, np.ndarray]:
    """
    Compute Stage 4 Divergence Anomaly Score:
    Score_i = e_A + e_B + lambda * |e_A - e_B|
    """
    error_a = np.asarray(error_a, dtype=np.float32)
    error_b = np.asarray(error_b, dtype=np.float32)
    divergence_term = lambda_param * np.abs(error_a - error_b)
    score = error_a + error_b + divergence_term
    return score


def grid_search_lambda(
    error_a: np.ndarray,
    error_b: np.ndarray,
    candidate_lambdas: List[float] = [0.1, 0.5, 1.0, 2.0]
) -> Dict[str, Any]:
    """
    Stage 4 Hyperparameter Grid Search:
    Evaluates candidate lambda parameters in {0.1, 0.5, 1.0, 2.0} for optimal stream divergence weighting.
    Returns dictionary with best lambda, per-lambda score metrics, and divergence statistics.
    """
    error_a = np.asarray(error_a, dtype=np.float32)
    error_b = np.asarray(error_b, dtype=np.float32)
    abs_diff = np.abs(error_a - error_b)
    sum_err = error_a + error_b

    results = {}
    best_lambda = candidate_lambdas[0]
    best_variance_ratio = -1.0

    for lam in candidate_lambdas:
        scores = sum_err + lam * abs_diff
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        mean_div = float(np.mean(lam * abs_diff))
        # Measure how strongly divergence separates variation
        var_ratio = float(std_score / (mean_score + 1e-8))

        results[float(lam)] = {
            "lambda": float(lam),
            "mean_score": mean_score,
            "std_score": std_score,
            "mean_divergence_term": mean_div,
            "variance_ratio": var_ratio,
            "scores": scores,
        }

        if var_ratio > best_variance_ratio:
            best_variance_ratio = var_ratio
            best_lambda = float(lam)

    return {
        "best_lambda": best_lambda,
        "grid_results": results,
        "candidate_lambdas": candidate_lambdas,
    }


class DualWindowAdaptiveThreshold:
    """
    Stage 5: Dual-Window Rolling Adaptive Threshold Calculator.
    Calculates dynamic threshold per camera feed without manual tuning.
    """
    def __init__(self, short_window: int = 20, long_window: int = 100, k_std: float = 2.0):
        self.short_window = short_window
        self.long_window = long_window
        self.k_std = k_std

    def compute_thresholds(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute rolling threshold theta_i for each score i in sequence.
        Returns array of shape (N,) matching input scores.
        """
        scores = np.asarray(scores, dtype=np.float32)
        N = len(scores)
        thresholds = np.zeros(N, dtype=np.float32)

        for i in range(N):
            # Short window up to current index i
            short_start = max(0, i - self.short_window + 1)
            sw_scores = scores[short_start : i + 1]
            mu_s = float(np.mean(sw_scores))
            sigma_s = float(np.std(sw_scores))
            theta_s = mu_s + self.k_std * sigma_s

            # Long window up to current index i
            long_start = max(0, i - self.long_window + 1)
            lw_scores = scores[long_start : i + 1]
            mu_l = float(np.mean(lw_scores))
            sigma_l = float(np.std(lw_scores))
            theta_l = mu_l + self.k_std * sigma_l

            # Adaptive threshold = min(theta_s, theta_l)
            thresholds[i] = min(theta_s, theta_l)

        return thresholds


def classify_clips_and_compute_reliability(
    guard_statuses: List[str],
    divergence_scores: np.ndarray,
    adaptive_thresholds: np.ndarray
) -> Tuple[List[str], float, Dict[str, Any]]:
    """
    Stage 6: Classify each clip as FROZEN, IRREGULAR, or HEALTHY,
    and compute the overall Feed Reliability Score R in [0, 1].

    Returns:
        (classifications, reliability_score_R, summary_dict)
    """
    N = len(guard_statuses)
    classifications: List[str] = []
    unhealthy_count = 0

    for i in range(N):
        g_status = guard_statuses[i]
        score = divergence_scores[i]
        thresh = adaptive_thresholds[i]

        if g_status == "FROZEN":
            cls_name = "FROZEN"
            unhealthy_count += 1
        elif score > thresh:
            cls_name = "IRREGULAR"
            unhealthy_count += 1
        else:
            cls_name = "HEALTHY"

        classifications.append(cls_name)

    reliability_score_R = float(1.0 - (unhealthy_count / N)) if N > 0 else 1.0

    counts = {
        "HEALTHY": classifications.count("HEALTHY"),
        "IRREGULAR": classifications.count("IRREGULAR"),
        "FROZEN": classifications.count("FROZEN"),
    }

    summary = {
        "total_clips": N,
        "class_counts": counts,
        "unhealthy_clips": unhealthy_count,
        "feed_reliability_score_R": round(reliability_score_R, 4),
    }

    return classifications, reliability_score_R, summary

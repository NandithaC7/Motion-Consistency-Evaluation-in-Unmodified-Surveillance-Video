"""
Stage 2 and Stage 3 pipeline for the motion-consistency project.

This module consumes the already preprocessed image dataset in Data_set/
and implements:

Stage 2 - Clip Builder
    - reads preprocessed grayscale frame images from folders
    - builds 16-frame clips with a sliding window
    - computes Stream A (frame-difference) and Stream B (optical-flow)

Stage 3 - Dual LSTM-AE
    - trains two independent LSTM autoencoders, one per stream
    - computes reconstruction errors per clip
    - saves trained models and score tables for downstream use

The code is written to work with the provided sample dataset layout first,
while also supporting a larger downloaded dataset later.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_STREAM_A = Path("Data_set/processed_output/1_stream_A_frame_diff.npy")
DEFAULT_STREAM_B = Path("Data_set/processed_output/1_stream_B_optical_flow.npy")


def _natural_key(path: Path) -> List[Any]:
    parts = re.split(r"(\d+)", path.stem)
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part.lower())
    return key


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenCV is required only when building clips from image folders. "
            "The provided saved stream arrays do not need it."
        ) from exc
    return cv2


def _load_grayscale_frame(frame_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")
    if frame.shape[:2] != (target_size[1], target_size[0]):
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    return frame


def _collect_image_sequences(root: Path) -> List[Tuple[str, List[Path]]]:
    """Collect image files grouped by directory.

    A folder containing images is treated as one sequence. If the root itself
    has images, that root-level image set is included too.
    """
    sequences: List[Tuple[str, List[Path]]] = []

    root_images = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=_natural_key,
    )
    if root_images:
        sequences.append((root.name or str(root), root_images))

    for directory in sorted([p for p in root.rglob("*") if p.is_dir()]):
        image_files = sorted(
            [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=_natural_key,
        )
        if image_files:
            sequences.append((directory.name, image_files))

    return sequences


def compute_frame_difference(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
    """Return a scalar motion value for Stream A."""
    cv2 = _cv2()
    diff = cv2.absdiff(current_frame, previous_frame)
    return float(np.mean(diff) / 255.0)


def compute_optical_flow_magnitude(previous_frame: np.ndarray, current_frame: np.ndarray) -> float:
    """Return a scalar motion value for Stream B."""
    cv2 = _cv2()
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame,
        current_frame,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


@dataclass
class ClipRecord:
    clip_id: int
    sequence_name: str
    frame_paths: List[str]
    status: str
    stream_a: np.ndarray
    stream_b: np.ndarray
    processed_frames: List[np.ndarray]
    padded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "sequence_name": self.sequence_name,
            "frame_paths": self.frame_paths,
            "status": self.status,
            "stream_a": self.stream_a,
            "stream_b": self.stream_b,
            "processed_frames": self.processed_frames,
            "padded": self.padded,
        }


class Stage2ClipBuilder:
    def __init__(
        self,
        window_size: int = 16,
        stride: int = 1,
        frame_size: Tuple[int, int] = (64, 64),
        demo_pad_short_sequences: bool = False,
    ):
        self.window_size = window_size
        self.stride = stride
        self.frame_size = frame_size
        self.demo_pad_short_sequences = demo_pad_short_sequences

    def load_sequence(self, frame_paths: Sequence[Path]) -> List[np.ndarray]:
        return [_load_grayscale_frame(path, self.frame_size) for path in frame_paths]

    def _pad_sequence(self, frames: List[np.ndarray], frame_paths: List[Path]) -> Tuple[List[np.ndarray], List[Path], bool]:
        if len(frames) >= self.window_size:
            return frames, frame_paths, False
        if not self.demo_pad_short_sequences or not frames:
            return frames, frame_paths, False

        padded_frames = list(frames)
        padded_paths = list(frame_paths)
        last_frame = frames[-1]
        last_path = frame_paths[-1]
        while len(padded_frames) < self.window_size:
            padded_frames.append(last_frame.copy())
            padded_paths.append(last_path)
        return padded_frames, padded_paths, True

    def build(self, data_root: str | Path) -> List[Dict[str, Any]]:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"Data root not found: {root}")

        sequences = _collect_image_sequences(root)
        clips: List[Dict[str, Any]] = []
        clip_id = 0

        for sequence_name, frame_paths in sequences:
            if not frame_paths:
                continue

            frames = self.load_sequence(frame_paths)
            frames, padded_paths, padded = self._pad_sequence(frames, frame_paths)
            if len(frames) < self.window_size:
                continue

            if padded:
                starts = [0]
            else:
                starts = list(range(0, len(frames) - self.window_size + 1, self.stride))

            for start in starts:
                window_frames = frames[start : start + self.window_size]
                window_paths = padded_paths[start : start + self.window_size]
                if len(window_frames) < self.window_size:
                    continue

                stream_a = self._build_stream_a(window_frames)
                stream_b = self._build_stream_b(window_frames)
                clips.append(
                    ClipRecord(
                        clip_id=clip_id,
                        sequence_name=sequence_name,
                        frame_paths=[str(p) for p in window_paths],
                        status="PASS",
                        stream_a=stream_a,
                        stream_b=stream_b,
                        processed_frames=[frame.astype(np.float32) / 255.0 for frame in window_frames],
                        padded=padded,
                    ).to_dict()
                )
                clip_id += 1

        return clips

    def _build_stream_a(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        values = [0.0]
        for idx in range(1, len(frames)):
            values.append(compute_frame_difference(frames[idx - 1], frames[idx]))
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)

    def _build_stream_b(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        values = [0.0]
        for idx in range(1, len(frames)):
            values.append(compute_optical_flow_magnitude(frames[idx - 1], frames[idx]))
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def build_clips_from_stream_arrays(stream_a: np.ndarray, stream_b: np.ndarray) -> List[Dict[str, Any]]:
    if stream_a.shape != stream_b.shape:
        raise ValueError("Stream A and Stream B arrays must have identical shapes.")
    if stream_a.ndim != 4:
        raise ValueError("Expected saved stream arrays with shape (N, 16, H, W).")

    clips: List[Dict[str, Any]] = []
    for clip_id in range(stream_a.shape[0]):
        clips.append(
            {
                "clip_id": clip_id,
                "sequence_name": f"sequence_{clip_id}",
                "frame_paths": [],
                "status": "PASS",
                "stream_a": np.asarray(stream_a[clip_id], dtype=np.float32),
                "stream_b": np.asarray(stream_b[clip_id], dtype=np.float32),
                "processed_frames": [],
                "padded": False,
            }
        )
    return clips


@dataclass
class TrainingHistory:
    history: Dict[str, List[float]]


def _flatten_sequences(sequences: np.ndarray) -> np.ndarray:
    if sequences.ndim < 2:
        raise ValueError("Expected sequences with at least 2 dimensions.")
    return sequences.reshape(sequences.shape[0], -1)


class DualLSTMAutoencoder:
    def __init__(self, input_shape: Tuple[int, int] = (16, 1), latent_units: int = 64):
        self.input_shape = input_shape
        self.latent_units = latent_units
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(
        self,
        train_sequences: np.ndarray,
        epochs: int = 30,
        batch_size: int = 16,
        validation_split: float = 0.2,
        patience: int = 5,
    ):
        if train_sequences.ndim < 2:
            raise ValueError("Expected train_sequences with shape (N, ...)")

        flat_sequences = _flatten_sequences(train_sequences).astype(np.float32)
        if len(flat_sequences) < 2:
            raise ValueError("Need at least 2 clips to train the autoencoder.")

        validation_size = max(1, int(round(len(flat_sequences) * validation_split)))
        if validation_size >= len(flat_sequences):
            validation_size = 1

        train_x = flat_sequences[:-validation_size]
        val_x = flat_sequences[-validation_size:]

        self.mean_ = train_x.mean(axis=0)
        centered = train_x - self.mean_
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        latent_dim = max(1, min(self.latent_units, vt.shape[0]))
        self.components_ = vt[:latent_dim]

        train_pred = self.reconstruct(train_x)
        val_pred = self.reconstruct(val_x)
        train_loss = float(np.mean(np.square(train_x - train_pred)))
        val_loss = float(np.mean(np.square(val_x - val_pred)))
        return TrainingHistory(history={"loss": [train_loss], "val_loss": [val_loss]})

    def reconstruct(self, sequences: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Model has not been trained yet.")
        flat_sequences = _flatten_sequences(sequences).astype(np.float32)
        centered = flat_sequences - self.mean_
        latent = centered @ self.components_.T
        reconstructed = latent @ self.components_ + self.mean_
        return reconstructed.reshape(sequences.shape)

    def reconstruction_error(self, sequences: np.ndarray) -> np.ndarray:
        reconstructed = self.reconstruct(sequences)
        squared_error = np.square(sequences - reconstructed)
        return np.mean(squared_error, axis=tuple(range(1, squared_error.ndim)))

    def save(self, path: str | Path):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Model has not been trained yet.")
        with open(path, "wb") as handle:
            pickle.dump({"mean_": self.mean_, "components_": self.components_}, handle)


def get_valid_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [clip for clip in clips if clip.get("status") == "PASS"]


def stream_array_from_clips(clips: List[Dict[str, Any]], key: str) -> np.ndarray:
    if not clips:
        return np.empty((0, 16, 1), dtype=np.float32)
    arrays = [np.asarray(clip[key], dtype=np.float32) for clip in clips]
    return np.stack(arrays, axis=0)


def _clip_motion_sequence(stream: np.ndarray) -> np.ndarray:
    array = np.asarray(stream, dtype=np.float32)
    if array.ndim >= 3:
        axes = tuple(range(1, array.ndim))
        return array.mean(axis=axes)
    if array.ndim == 2:
        return array.mean(axis=1)
    return array.reshape(-1)


def motion_sequences_from_clips(clips: List[Dict[str, Any]], key: str) -> np.ndarray:
    if not clips:
        return np.empty((0, 16), dtype=np.float32)
    sequences = [_clip_motion_sequence(clip[key]) for clip in clips]
    return np.stack(sequences, axis=0).astype(np.float32)


@dataclass
class SequenceStandardizer:
    mean_: np.ndarray
    std_: np.ndarray


def _fit_stream_scaler(streams: np.ndarray) -> SequenceStandardizer:
    flat = streams.reshape(streams.shape[0], -1)
    mean_ = flat.mean(axis=0)
    std_ = flat.std(axis=0)
    std_[std_ < 1e-8] = 1.0
    return SequenceStandardizer(mean_=mean_.astype(np.float32), std_=std_.astype(np.float32))


def _transform_streams(streams: np.ndarray, scaler: SequenceStandardizer) -> np.ndarray:
    flat = streams.reshape(streams.shape[0], -1).astype(np.float32)
    transformed = (flat - scaler.mean_) / scaler.std_
    return transformed.astype(np.float32)


def save_scores_csv(path: Path, rows: List[Dict[str, Any]]):
    _ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_stage23(
    data_root: str | Path,
    train_root: Optional[str | Path] = None,
    test_root: Optional[str | Path] = None,
    output_dir: str | Path = "Data_set/processed_output/stage23",
    window_size: int = 16,
    stride: int = 1,
    demo_pad_short_sequences: bool = False,
    epochs: int = 30,
    batch_size: int = 16,
    latent_units: int = 64,
) -> Dict[str, Any]:
    builder = Stage2ClipBuilder(
        window_size=window_size,
        stride=stride,
        demo_pad_short_sequences=demo_pad_short_sequences,
    )
    output_path = _ensure_dir(Path(output_dir))

    saved_streams = None
    if DEFAULT_STREAM_A.exists() and DEFAULT_STREAM_B.exists():
        saved_streams = (
            np.load(DEFAULT_STREAM_A, allow_pickle=True).astype(np.float32),
            np.load(DEFAULT_STREAM_B, allow_pickle=True).astype(np.float32),
        )

    if saved_streams is not None:
        train_clips = get_valid_clips(build_clips_from_stream_arrays(*saved_streams))
        test_clips: List[Dict[str, Any]] = []
    else:
        train_source = Path(train_root) if train_root is not None else Path(data_root)
        test_source = Path(test_root) if test_root is not None else None

        train_clips = get_valid_clips(builder.build(train_source))
        test_clips = get_valid_clips(builder.build(test_source)) if test_source is not None else []

    if not train_clips:
        raise RuntimeError(
            "No valid clips were built from the training data. "
            "If you are using the small sample dataset, enable demo_pad_short_sequences=True or download the full set."
        )

    train_a = motion_sequences_from_clips(train_clips, "stream_a")
    train_b = motion_sequences_from_clips(train_clips, "stream_b")

    scaler_a = _fit_stream_scaler(train_a)
    scaler_b = _fit_stream_scaler(train_b)
    train_a_scaled = _transform_streams(train_a, scaler_a)
    train_b_scaled = _transform_streams(train_b, scaler_b)

    model_a = DualLSTMAutoencoder(input_shape=train_a_scaled.shape[1:], latent_units=latent_units)
    model_b = DualLSTMAutoencoder(input_shape=train_b_scaled.shape[1:], latent_units=latent_units)

    history_a = model_a.fit(train_a_scaled, epochs=epochs, batch_size=batch_size)
    history_b = model_b.fit(train_b_scaled, epochs=epochs, batch_size=batch_size)

    train_err_a = model_a.reconstruction_error(train_a_scaled)
    train_err_b = model_b.reconstruction_error(train_b_scaled)

    # ── Stage 4: Divergence Anomaly Score ──────────────────────────────────────
    # Score_i = e_A + e_B + \lambda |e_A - e_B|
    lambda_param = 1.0
    divergence_scores = train_err_a + train_err_b + lambda_param * np.abs(train_err_a - train_err_b)

    # ── Stage 5: Dual-Window Adaptive Threshold ────────────────────────────────
    # \theta = min(\mu_s + 2\sigma_s, \mu_l + 2\sigma_l)
    N_clips = len(train_clips)
    adaptive_thresholds = np.zeros(N_clips, dtype=np.float32)
    for i in range(N_clips):
        sw = divergence_scores[max(0, i - 19) : i + 1]
        lw = divergence_scores[max(0, i - 99) : i + 1]
        theta_s = float(np.mean(sw) + 2.0 * np.std(sw))
        theta_l = float(np.mean(lw) + 2.0 * np.std(lw))
        adaptive_thresholds[i] = min(theta_s, theta_l)

    # ── Stage 6: Classification & Feed Reliability Score R ─────────────────────
    classifications = []
    unhealthy_count = 0
    for clip, score, thresh in zip(train_clips, divergence_scores, adaptive_thresholds):
        g_status = clip.get("status", "PASS")
        if g_status == "FROZEN":
            cls_name = "FROZEN"
            unhealthy_count += 1
        elif score > thresh:
            cls_name = "IRREGULAR"
            unhealthy_count += 1
        else:
            cls_name = "HEALTHY"
        classifications.append(cls_name)

    reliability_score_R = float(1.0 - (unhealthy_count / N_clips)) if N_clips > 0 else 1.0

    train_rows = []
    for clip, err_a, err_b, score, thresh, cls_name in zip(
        train_clips, train_err_a, train_err_b, divergence_scores, adaptive_thresholds, classifications
    ):
        train_rows.append(
            {
                "clip_id": clip["clip_id"],
                "sequence_name": clip["sequence_name"],
                "stream_a_error": float(err_a),
                "stream_b_error": float(err_b),
                "divergence_score": float(score),
                "adaptive_threshold": float(thresh),
                "classification": cls_name,
            }
        )

    test_rows = []
    if test_clips:
        test_a = _transform_streams(motion_sequences_from_clips(test_clips, "stream_a"), scaler_a)
        test_b = _transform_streams(motion_sequences_from_clips(test_clips, "stream_b"), scaler_b)
        test_err_a = model_a.reconstruction_error(test_a)
        test_err_b = model_b.reconstruction_error(test_b)
        test_div_scores = test_err_a + test_err_b + lambda_param * np.abs(test_err_a - test_err_b)
        for clip, err_a, err_b, score in zip(test_clips, test_err_a, test_err_b, test_div_scores):
            test_rows.append(
                {
                    "clip_id": clip["clip_id"],
                    "sequence_name": clip["sequence_name"],
                    "stream_a_error": float(err_a),
                    "stream_b_error": float(err_b),
                    "divergence_score": float(score),
                    "classification": "IRREGULAR" if score > float(np.mean(adaptive_thresholds)) else "HEALTHY",
                }
            )

    model_a_path = output_path / "stream_a_autoencoder.pkl"
    model_b_path = output_path / "stream_b_autoencoder.pkl"
    model_a.save(model_a_path)
    model_b.save(model_b_path)

    with open(output_path / "stream_a_scaler.pkl", "wb") as handle:
        pickle.dump(scaler_a, handle)
    with open(output_path / "stream_b_scaler.pkl", "wb") as handle:
        pickle.dump(scaler_b, handle)

    save_scores_csv(output_path / "train_reconstruction_scores.csv", train_rows)
    save_scores_csv(output_path / "test_reconstruction_scores.csv", test_rows)

    summary = {
        "train_clip_count": len(train_clips),
        "test_clip_count": len(test_clips),
        "stream_a_model": str(model_a_path),
        "stream_b_model": str(model_b_path),
        "train_loss_a_final": float(history_a.history["loss"][-1]),
        "train_loss_b_final": float(history_b.history["loss"][-1]),
        "feed_reliability_score_R": round(reliability_score_R, 4),
        "classification_counts": {
            "HEALTHY": classifications.count("HEALTHY"),
            "IRREGULAR": classifications.count("IRREGULAR"),
            "FROZEN": classifications.count("FROZEN"),
        },
        "output_dir": str(output_path),
    }

    with open(output_path / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "summary": summary,
        "train_scores": train_rows,
        "test_scores": test_rows,
    }

    with open(output_path / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "summary": summary,
        "train_scores": train_rows,
        "test_scores": test_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 and Stage 3 pipeline")
    parser.add_argument("--data_root", type=str, default="Data_set")
    parser.add_argument("--train_root", type=str, default="Data_set/training_samples")
    parser.add_argument("--test_root", type=str, default="Data_set/testing_samples")
    parser.add_argument("--output_dir", type=str, default="Data_set/processed_output/stage23")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--demo_pad_short_sequences", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--latent_units", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_stage23(
        data_root=args.data_root,
        train_root=args.train_root,
        test_root=args.test_root,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        demo_pad_short_sequences=args.demo_pad_short_sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_units=args.latent_units,
    )
    print(json.dumps(results["summary"], indent=2))
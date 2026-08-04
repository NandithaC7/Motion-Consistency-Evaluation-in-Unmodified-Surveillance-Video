# Preprocessing Pipeline
## Motion Consistency Evaluation in Unmodified Surveillance Video

---

### Scope

This module implements the **preprocessing pipeline** for the project.  
It covers everything up to and including the Guard Layers.

> **This module STOPS before Stream A (Motion Detection) and Stream B (Optical Flow).**  
> Those belong to a separate downstream module.

---

### What This Module Does

```
Raw Video (.mp4 / .avi / .mov)
    │
    ▼
[Module 1] VideoLoader
    │  • Validates format & file existence
    │  • Reads FPS, resolution, total frames, duration
    │  • Handles corrupted videos gracefully
    │  • tqdm progress bar while reading
    │  • Configurable frame skipping + max frame cap
    │
    ▼
[Module 2] FramePreprocessor
    │  • Resize to 64×64
    │  • Convert to grayscale
    │  • Normalize to [0, 1]  (float32)
    │  • Optional Gaussian Blur
    │
    ▼
[Module 3] GuardG1 — Pixel Variance
    │  • Sliding window of 16 frames
    │  • Variance < θ_v  → clip marked FROZEN
    │  • Returns: {"status","reason","variance"}
    │
    ▼  (only if G1 passes)
[Module 4] GuardG2 — SSIM
    │  • SSIM between every consecutive frame pair
    │  • ≥ 8 consecutive pairs with SSIM > 0.999 → FROZEN
    │  • Returns: {"status","reason","count","ssim_scores"}
    │
    ▼  (only if G2 passes)
[Module 5] SlidingWindowBuilder
    │  • Builds validated 16-frame clips
    │  • Returns clips as structured dicts
    │
    ▼
Output: list of clip dicts → ready for downstream developer
        (Stream A / Stream B)
```

---

### Setup

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Verify installation

```bash
python -c "import cv2, numpy, skimage, tqdm, matplotlib; print('OK')"
```

---

### Quick Start

```python
from preprocess_pipeline import PipelineConfig, PreprocessingPipeline, get_valid_clips

config = PipelineConfig()
config.max_frames    = 200      # limit for testing; None = full video
config.gaussian_blur = False
config.frame_skip    = 1        # 1 = no skipping, 2 = every other frame
config.debug_mode    = True     # saves diagnostic plots to debug_output_dir

pipeline = PreprocessingPipeline(config)
clips, metadata = pipeline.run("path/to/video.mp4")

# Get only valid clips for the next module
valid_clips = get_valid_clips(clips)
print(f"Valid clips ready: {len(valid_clips)}")
```

---

### Clip Output Format

Each element in the returned `clips` list is a dictionary:

```python
{
    "clip_id"          : int,            # window index (0-based)
    "frame_indices"    : list[int],      # original video frame numbers (16 values)
    "status"           : "PASS" | "FROZEN",
    "reason"           : str | None,     # e.g. "Low Pixel Variance", "High SSIM"
    "processed_frames" : list[np.ndarray],  # 16 × (64,64) float32 arrays
    "g1_result"        : dict,           # raw G1 output
    "g2_result"        : dict | None,    # raw G2 output (None if G1 froze it)
}
```

Only pass `status == "PASS"` clips to Stream A / Stream B.

---

### Run the Test Suite

```bash
python test_pipeline.py
```

Expected output:
```
══════════════════════════════════════════════════════════════════
  PREPROCESSING PIPELINE — TEST SUITE
══════════════════════════════════════════════════════════════════
  ✔  G1 detects frozen window
  ✔  G1 frozen reason text correct
  ...
  RESULTS: 25/25 passed
  ✔  All tests passed!
```

---

### Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_size` | `(64, 64)` | Target resize dimensions |
| `window_size` | `16` | Frames per clip |
| `variance_threshold` | `0.001` | G1 threshold θ_v |
| `ssim_threshold` | `0.999` | G2 per-pair SSIM cutoff |
| `ssim_consecutive` | `8` | G2 minimum run length |
| `gaussian_blur` | `False` | Apply 3×3 Gaussian blur |
| `frame_skip` | `1` | Keep every Nth frame |
| `max_frames` | `None` | Cap on frames read |
| `debug_mode` | `False` | Save diagnostic plots |
| `debug_output_dir` | `processed_output/` | Folder for debug images |

---

### File Structure

```
AICity21-Track4-Anomaly-Detection/
├── preprocess_pipeline.py   ← Main pipeline (this module)
├── test_pipeline.py         ← Test suite
├── requirements.txt         ← Dependencies
├── README.md                ← This file
└── processed_output/        ← Debug / visualization outputs
```

---

### What This Module Does NOT Do

- ❌ Frame difference (Stream A) — downstream module
- ❌ Optical flow (Stream B) — downstream module  
- ❌ Dual LSTM Autoencoder — downstream module
- ❌ Divergence scoring — downstream module
- ❌ Thresholding or Classification — downstream module

---

### Stage 2 and Stage 3 for the Current Dataset

The repository already includes the saved motion streams in:

- `Data_set/processed_output/1_stream_A_frame_diff.npy`
- `Data_set/processed_output/1_stream_B_optical_flow.npy`

Use these directly for Stage 2 clip building and Stage 3 model training.

#### Run

```bash
python Data_set/stage2_stage3_pipeline.py
```

#### What it does

- loads the provided Stream A and Stream B arrays if they exist
- treats each `(16, H, W)` slice as one clip
- trains two LSTM autoencoders, one per stream
- saves the trained models, scalers, and reconstruction scores under:

```text
Data_set/processed_output/stage23/
```

#### Output files

- `stream_a_lstm_autoencoder.keras`
- `stream_b_lstm_autoencoder.keras`
- `stream_a_autoencoder.pkl`
- `stream_b_autoencoder.pkl`
- `stream_a_scaler.pkl`
- `stream_b_scaler.pkl`
- `train_reconstruction_scores.csv`
- `test_reconstruction_scores.csv`
- `run_summary.json`

#### If you want to rebuild clips from images instead

Point the script at a folder containing preprocessed frame images and pass `--data_root`.
If the folder has fewer than 16 frames per sequence, use `--demo_pad_short_sequences` only for a demo run.

---

### What To Do After Stage 3

After training the two autoencoders, the next stage is to:

1. Compare Stream A and Stream B reconstruction errors.
2. Compute the divergence score from the two errors.
3. Apply the rolling threshold to label clips as `Healthy`, `Irregular`, or `Frozen`.
4. Export a results table and plots for your report.

---

### Tech Stack

- Python 3.11
- OpenCV — video I/O and image processing
- NumPy — array operations
- scikit-image — SSIM computation
- tqdm — progress bars
- pathlib — path handling
- matplotlib — debug visualizations only

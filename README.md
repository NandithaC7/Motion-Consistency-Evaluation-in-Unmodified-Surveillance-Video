# Motion Consistency Evaluation in Unmodified Surveillance Video

## Project Type

Final Year Project (Research-Based)
Course: 23CSE399 – Project Phase 1

---

## Problem Statement

Surveillance camera systems generate continuous video streams; however, real-world feeds often suffer from hidden issues such as frozen frames, prolonged low motion, and temporal inconsistencies caused by environmental or hardware conditions. These issues do not trigger traditional object-based alerts but significantly reduce monitoring reliability.

This project focuses on evaluating motion consistency in unmodified surveillance videos to assess camera feed health using a novel Dual-Stream LSTM Autoencoder approach.

---

## Proposed Methodology — 6-Stage Pipeline

### Guard Layer G1 – Pixel Variance
- Compute pixel variance across T=16 frames
- Var < θ_v ⟹ **FROZEN** (skip downstream)

### Guard Layer G2 – SSIM Check
- Compute SSIM(F_t, F_{t-1}) per frame pair
- SSIM > 0.999 for ≥ 8 consecutive frames ⟹ **FROZEN**

### Stage 1 – Preprocessing
- Resize to 64×64 · grayscale · normalise [0, 1]
- Frame diff: M_t^A = ‖F_t − F_{t-1}‖ (Stream A)
- Farneback optical flow magnitude: M_t^B (Stream B)

### Stage 2 – Clip Builder
- Sliding window T=16 frames, stride=1
- Builds Stream A: {M_1^A, ..., M_16^A}
- Builds Stream B: {M_1^B, ..., M_16^B}

### Stage 3 – Dual LSTM-AE (PyTorch)
- Two independent LSTM autoencoders (2-layer LSTM, hidden_dim=64)
- Encoder: 16→64 units; Decoder: 64→16 reconstruction
- Trained on normal footage only — no labels needed
- Outputs: reconstruction errors e_A, e_B

### Stage 4 – Divergence Anomaly Score
- Score_i = e_A + e_B + λ|e_A − e_B|
- λ selected via grid search: {0.1, 0.5, 1.0, 2.0}
- Novel contribution — divergence term catches single-stream faults

### Stage 5 – Dual-Window Rolling Threshold
- Short window (N_s = 20): detects sudden faults
- Long window (N_l = 100): detects gradual drift
- θ = min(μ_s + 2σ_s, μ_l + 2σ_l)
- Self-calibrates per camera — no manual tuning

### Stage 6 – Classification
- G1 or G2 triggered ⟹ **Frozen**
- Score > θ ⟹ **Irregular**
- Otherwise ⟹ **Healthy**
- Feed Reliability Score R ∈ [0, 1]

---

## Experimental Results (on AICity21 Track4 Dataset)

| Metric | Value |
|--------|-------|
| Total clips analyzed | 184 |
| Stream A final MSE loss | 0.000871 |
| Stream B final MSE loss | 0.031684 |
| Optimal λ (grid search) | 2.0 |
| Healthy clips | 177 (96.2%) |
| Irregular anomalies | 7 (3.8%) |
| Frozen clips | 0 |
| **Feed Reliability Score R** | **0.9620 (96.20%)** |

---

## Repository Structure

```
.
├── src/
│   ├── models/
│   │   └── lstm_autoencoder.py       # Dual Stream PyTorch LSTM Autoencoder
│   ├── evaluation/
│   │   └── anomaly_scorer.py         # Stages 4, 5, 6: divergence score, threshold, classification
│   └── preprocessing/
│       └── frame_extractor.py        # Video-to-frames extraction utility
│
├── Data_set/
│   ├── preprocess_pipeline.py        # Stage 1: Guard G1, G2, preprocessing, clip builder
│   ├── stage2_stage3_pipeline.py     # Stage 2 & 3: Clip builder + Dual LSTM-AE (from image folders)
│   ├── run_full_experiment.py        # Master end-to-end experiment runner (all 6 stages)
│   ├── test_pipeline.py              # Unit tests (44 tests)
│   ├── view_npy_output.py            # Visualise saved stream numpy arrays
│   ├── generate_frozen_frame_viz.py  # Visualise guard layer frozen detections
│   └── processed_output/
│       └── stage23/
│           ├── stream_a_lstm_ae.pt        # Trained LSTM-AE for Stream A
│           ├── stream_b_lstm_ae.pt        # Trained LSTM-AE for Stream B
│           ├── train_reconstruction_scores.csv
│           ├── lambda_grid_search_results.csv
│           ├── run_summary.json
│           ├── training_loss_curves.png
│           ├── reconstruction_errors_distribution.png
│           ├── divergence_score_and_adaptive_threshold.png
│           └── feed_reliability_summary.png
│
├── docs/
│   └── papers/
│       └── ieee_paper.tex            # IEEE format research paper
│
├── Documentation/                    # Project reports and presentations
├── Literature/                       # Reference papers
├── Results/                          # Historical results
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Setup & Running

### 1. Create Virtual Environment
```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv
```

### 2. Run Full Experiment (All 6 Stages)
```bash
.venv\Scripts\python.exe Data_set/run_full_experiment.py --epochs 30
```

### 3. Run Unit Tests
```bash
.venv\Scripts\python.exe -m pytest Data_set/test_pipeline.py -v
```

### Arguments for run_full_experiment.py
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 25 | LSTM-AE training epochs |
| `--hidden_dim` | 64 | LSTM hidden units |
| `--batch_size` | 16 | Training batch size |
| `--output_dir` | Data_set/processed_output | Directory with stream .npy files |

---

## Dataset

Primary: **AICity21 Track4 – Anomaly Detection** (static overhead surveillance cameras)

The pre-computed stream arrays `1_stream_A_frame_diff.npy` and `1_stream_B_optical_flow.npy` (shape: `N×16×64×64`) are expected in `Data_set/processed_output/`.

---

## Mathematical Formulation

**Stage 4 Divergence Anomaly Score:**
```
Score_i = e_A + e_B + λ|e_A − e_B|     λ ∈ {0.1, 0.5, 1.0, 2.0}
```

**Stage 5 Dual-Window Adaptive Threshold:**
```
θ_s = μ_s + 2σ_s   (N_s = 20 short window)
θ_l = μ_l + 2σ_l   (N_l = 100 long window)
θ   = min(θ_s, θ_l)
```

**Stage 6 Feed Reliability Score:**
```
R = 1 − (unhealthy_count / N)   ∈ [0, 1]
```

---

## Authors

Final Year Project Team
Department of Computer Science & Engineering
2025–2026

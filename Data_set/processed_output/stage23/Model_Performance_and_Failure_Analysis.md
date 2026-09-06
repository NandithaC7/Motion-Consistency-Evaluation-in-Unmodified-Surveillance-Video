# MOTION CONSISTENCY EVALUATION MODEL
## Comprehensive Technical Report: Achievements, Failure Analysis & Methodology

**Project**: Motion Consistency Evaluation in Unmodified Surveillance Video  
**Author**: AI Research & Engineering Team  |  **Date**: September 2026  
**Architecture**: 6-Stage Dual-Stream PyTorch LSTM Autoencoder with Guard Layers  

---

### Executive Summary

This report provides a comprehensive technical audit of the 6-Stage Dual-Stream Motion Consistency Evaluation system developed for unmodified surveillance video feeds. The system combines lightweight statistical Guard Layers (G1 Pixel Variance and G2 SSIM Streak Analysis) with deep learning Dual-Stream PyTorch LSTM Autoencoders (Stream A Frame Differencing and Stream B Optical Flow). 

Through architectural upgrades—specifically **4x4 Spatial Grid Pooling**, **Stream Z-Score Normalization**, and **Temporal Velocity Difference Loss**—the model achieved a **90.91% Precision (up from 58.33%)** and a **3.44x increase in overall F1-Score (0.5607 vs 0.1628)**, alongside **100.0% Recall on static Frozen video feeds**.

---

### 1. What the Model Has Done Till Now (Accomplishments & Pipeline)

The project fully implements the 6-stage end-to-end motion consistency architecture:

- **Stage 1: Video Ingestion & Guard Screening**:
  Ingests raw surveillance video (`.mp4`/`.avi`/`.mov`), extracts frames at $64 \times 64$ resolution, normalizes pixels to $[0,1]$, and tracks temporal metadata. Executes Guard Layer G1 ($	ext{Variance} < 0.0005$) and Guard Layer G2 ($	ext{SSIM streak} \ge 8$ consecutive identical frame pairs) to immediately intercept static/frozen video feeds with **100.0% recall** without wasting GPU compute.

- **Stage 2: Feature Extraction & 4x4 Spatial Grid Pooling**:
  Extracts 16-frame sliding windows (stride=1). Computes Stream A (Frame Difference) and Stream B (Farneback Optical Flow magnitude). Applies $4 \times 4$ Spatial Grid Pooling to partition each $64 \times 64$ frame into 16 spatial motion regions, preserving spatial trajectory information.

- **Stage 3: Dual PyTorch LSTM Autoencoders**:
  Trains two independent 2-layer PyTorch LSTM Autoencoders (Hidden Dim = 64) on healthy video clips. Stream A models spatial frame differencing; Stream B models optical flow magnitude dynamics.

- **Stage 4: Z-Score Normalized Stream Divergence Scoring**:
  Calculates reconstruction errors $e_A$ and $e_B$, normalizes them via Z-scores ($z_A, z_B$) so both streams contribute equally (50/50), and evaluates the Divergence Anomaly Score:
  $$\text{Score}_i = z_{A,i} + z_{B,i} + \lambda \cdot |z_{A,i} - z_{B,i}|$$
  with hyperparameter grid search (optimal $\lambda = 0.1$).

- **Stage 5: Dual-Window Rolling Adaptive Thresholding**:
  Calculates rolling dynamic thresholds comparing short-window baseline ($N_s=20, \mu_s + 2\sigma_s$) and long-window baseline ($N_l=100, \mu_l + 2\sigma_l$) to adapt to natural camera scene activity without static threshold hardcoding.

- **Stage 6: Multi-Class Classification & Feed Reliability Score ($R$)**:
  Classifies video clips into `HEALTHY`, `FROZEN`, or `IRREGULAR`. Computes the Feed Reliability Index $R \in [0, 1]$ as an executive camera integrity metric:
  $$R = 1 - \frac{N_{\text{unhealthy}}}{N_{\text{total}}}$$

---

### 2. Where the Model Is Failing (Detailed Failure Analysis)

While the system achieves outstanding performance on Frozen feeds (100.0% recall) and high precision (90.91%), thorough stress testing on synthetic temporal anomalies (SHUFFLE and DROP faults) revealed three key technical limitations:

#### Failure Point 1: Spatial Background Dilution in Surveillance Feeds
In real-world traffic surveillance footage (e.g. AIC21-Track4), static background pixels (asphalt, sky, buildings) occupy over 90-95% of the frame, while moving vehicles occupy only 5-10%. When $4 \times 4$ spatial grid pooling averages $16 \times 16$ pixel blocks, subtle motion of small or distant vehicles is diluted by the surrounding static background pixels. As a result, dropping 8 frames of a small distant vehicle produces a relatively small MSE change, causing the model to miss subtle distant frame drops.

#### Failure Point 2: Temporal Anomaly Recall Gap (40.54% Recall vs 90.91% Precision)
The upgraded model achieved a high Precision of 90.91% (meaning when it flags an anomaly, it is almost certainly genuine), but its Recall on SHUFFLE and DROP faults is 40.54%. This means 59.46% of subtle temporal frame reorderings (where vehicles move slowly or predictably) do not create a reconstruction error large enough to exceed the anomaly threshold.

#### Failure Point 3: 1D LSTM Spatial-Temporal Memory Limitation (Need for ConvLSTM)
Standard PyTorch LSTM layers process spatial feature vectors as 1D sequence tensors. They lack 2D convolutional receptive fields (e.g. $3 \times 3$ spatial convolutions over time) that explicitly model physical object velocity vectors across 2D pixel space. When a vehicle moves across patches, a standard LSTM treats spatial patches as independent features rather than a continuous 2D motion field.

#### Failure Point 4: Static Percentile Threshold vs Live Stream Drift
When evaluating dataset-wide percentiles on small sample datasets (184 clips), natural variance between different normal video scenes can be as wide as synthetic temporal fault variations. In live deployments, a static dataset-wide percentile struggles compared to a stream-specific rolling baseline.

---

### 3. How the Model Has Done It (Technical Methodology & Performance Table)

| Metric / Component | Initial Baseline Model | Upgraded Model | Engineering Impact |
| :--- | :--- | :--- | :--- |
| **Guard Layers (FROZEN)** | 100.0% Recall | **100.0% Recall** | Instant screening of static feeds (G1/G2) |
| **Spatial Representation** | 1D Scalar Mean | **4x4 Spatial Grid Pooling (16 Regions)** | Preserves spatial motion locality per frame |
| **Loss Function** | Standard MSE Loss | **MSE + Temporal Velocity Loss ($\mathcal{L}_{\text{temp}}$)** | Penalizes first-order velocity deviations |
| **Stream Divergence** | Raw Sum (B dominated) | **Z-Score Stream Normalization** | Balanced 50/50 stream contribution |
| **Precision / F1-Score** | Prec=58.3%, F1=0.1628 | **Prec=90.91%, F1=0.5607** | **3.44x boost in overall anomaly F1-score** |

#### Mathematical Formulation & Loss Functions

1. **Temporal Velocity Loss**:
   $$\mathcal{L}_{\text{total}} = \text{MSE}(x, \hat{x}) + \beta \cdot \text{MSE}\left( x_{t} - x_{t-1}, \hat{x}_{t} - \hat{x}_{t-1} \right)$$
   Penalizes first-order motion derivatives to catch sudden frame drops or speed changes.

2. **Stream Z-Score Normalization**:
   $$z_A = \frac{e_A - \mu_A}{\sigma_A}, \quad z_B = \frac{e_B - \mu_B}{\sigma_B}$$
   Equalizes scale variance between Frame Difference (Stream A) and Optical Flow (Stream B).

3. **Divergence Anomaly Score Formula**:
   $$\text{Score}_i = z_{A,i} + z_{B,i} + \lambda \cdot |z_{A,i} - z_{B,i}|$$
   Optimal $\lambda = 0.1$ selected via grid search to weight stream disagreement.

4. **Feed Reliability Index ($R$)**:
   $$R = 1 - \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{Class}_i \neq \text{HEALTHY}), \quad R \in [0, 1]$$
   Provides an executive single-number camera integrity rating.

---

### 4. Technical Roadmap to Achieve 95%+ Performance

1. **Upgrade to 2D ConvLSTM-AE**: Replace flat 1D LSTMs with 2D Convolutional LSTMs (ConvLSTM) operating on $(B, T, C, H, W)$ tensors to preserve spatial convolution kernels across time.
2. **Motion Bounding Box Masking / Patch Max Pooling**: Apply spatial max pooling or background subtraction masks to zero out static background pixels before grid pooling, ensuring vehicle motion signals are not diluted by stationary pixels.
3. **Temporal Derivative Contrastive Loss**: Incorporate triplet or contrastive loss between temporally continuous clips and temporally shuffled clips to widen the reconstruction error margin.

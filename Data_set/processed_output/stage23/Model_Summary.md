# Motion Consistency Model — Executive Summary

## 1. What the Model Has Done
- **6-Stage Pipeline**: Fully implemented end-to-end video ingestion, spatial grid pooling, PyTorch Dual LSTM Autoencoders, divergence anomaly scoring, dynamic adaptive thresholding, and feed reliability index ($R$).
- **Guard Layers (G1/G2)**: Achieved **100.0% Recall** on static/frozen video feeds via pixel variance (G1) and consecutive pairwise SSIM streak analysis (G2).
- **Upgraded Core**: Implemented $4 \times 4$ Spatial Grid Pooling ($16$ spatial motion regions), Z-score stream normalization ($z_A, z_B$), and Temporal Velocity Difference Loss ($\mathcal{L}_{\text{temporal}}$).

---

## 2. Key Performance Metrics

| Performance Metric | Old Baseline Model | Upgraded Model | Absolute Impact |
| :--- | :---: | :---: | :--- |
| **Precision** | 58.33% | **90.91%** | **+32.58%** (Low False Alarms) |
| **Overall F1-Score** | 0.1628 | **0.5607** | **3.44x Increase** |
| **FROZEN Feed Recall** | 100.0% | **100.0%** | Perfect Guard Layer Screening |
| **SHUFFLE Anomaly Recall** | 8.10% | **40.54%** | **5.0x Increase** |
| **DROP Anomaly Recall** | 0.00% | **40.54%** | **Went from 0% → 40.5%** |

---

## 3. Where the Model Is Failing
1. **Background Spatial Dilution**: Static background pixels ($90\text{--}95\%$ of frame) dilute small distant moving vehicles during grid averaging, causing subtle distant frame drops to be missed.
2. **Temporal Anomaly Recall Gap (40.5% vs 90.9% Precision)**: High precision ensures flagged anomalies are genuine, but $59.5\%$ of subtle frame reorderings on slow-moving objects fall below the threshold.
3. **1D LSTM Receptive Field Limitation**: Flat 1D LSTM layers lack 2D convolutional receptive fields ($3 \times 3$ ConvLSTM) to model physical 2D velocity fields across space.

---

## 4. Key Formulas & Architecture
- **Temporal Velocity Loss**: 
  $$\mathcal{L}_{\text{total}} = \text{MSE}(x, \hat{x}) + 1.0 \times \text{MSE}\left( x_t - x_{t-1}, \hat{x}_t - \hat{x}_{t-1} \right)$$
- **Stream Z-Score Divergence Score**: 
  $$\text{Score}_i = z_{A,i} + z_{B,i} + \lambda \cdot |z_{A,i} - z_{B,i}| \quad (\lambda = 0.1)$$
- **Feed Reliability Index ($R$)**: 
  $$R = 1 - \frac{N_{\text{unhealthy}}}{N_{\text{total}}}, \quad R \in [0, 1]$$

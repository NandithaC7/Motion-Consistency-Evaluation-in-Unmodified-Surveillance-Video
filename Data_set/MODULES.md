# Module Explanations & Execution Flow
## Motion Consistency Evaluation in Unmodified Surveillance Video
### Preprocessing Pipeline — Final Year Project

---

## Execution Flow

```
preprocess_pipeline.py
│
└─ PreprocessingPipeline.run(video_path)
       │
       ├─ VideoLoader.load()                      # Module 1
       │      └─ returns raw_frames, metadata
       │
       ├─ FramePreprocessor.process_all()         # Module 2
       │      └─ returns processed_frames
       │
       └─ SlidingWindowBuilder.build()            # Module 5 (orchestrates 3 & 4)
              │
              ├─ [for each window i]
              │      │
              │      ├─ GuardG1Variance.check_window()    # Module 3
              │      │      ├─ FROZEN → append clip, continue next window
              │      │      └─ PASS   → proceed to G2
              │      │
              │      ├─ GuardG2SSIM.check_window()        # Module 4
              │      │      ├─ FROZEN → append clip, continue next window
              │      │      └─ PASS   → append valid clip
              │      │
              │      └─ clip added to output list
              │
              └─ returns clips (all windows, status PASS or FROZEN)
```

---

## Module 1 — VideoLoader

**File**: `preprocess_pipeline.py` → class `VideoLoader`

### Purpose
Opens a surveillance video, validates it, reads all frames, and returns them alongside video-level metadata.

### Inputs
- `video_path` — absolute path to `.mp4`, `.avi`, or `.mov` file

### Outputs
- `raw_frames` — list of NumPy BGR arrays (H × W × 3, uint8)
- `metadata` — dict:
  ```python
  {
      "video_name"       : str,
      "fps"              : float,
      "resolution"       : (width, height),
      "total_frames"     : int,
      "duration_seconds" : float,
      "frames_extracted" : int,
      "timestamps"       : list[float],   # seconds per frame
      "frame_indices"    : list[int],     # original frame number in video
  }
  ```

### Key Design Decisions
| Decision | Reason |
|----------|--------|
| Raise `ValueError` for unsupported extensions | Fail early with a clear message rather than silently producing bad output |
| Raise `RuntimeError` if OpenCV can't open file | Catches corrupted containers (not just missing files) |
| `tqdm` progress bar | Processing 30-minute surveillance videos takes minutes; the bar shows progress and estimated time |
| `frame_skip` parameter | Allows the pipeline to process, e.g., every 2nd or 4th frame to reduce data volume on long videos |
| Timestamps preserved | Downstream modules may need to map clip output back to wall-clock time in the original footage |

---

## Module 2 — FramePreprocessor

**File**: `preprocess_pipeline.py` → class `FramePreprocessor`

### Purpose
Converts raw BGR video frames into normalised 64×64 grayscale arrays ready for guard layer evaluation and clip building.

### Inputs
- `raw_frames` — list of BGR uint8 frames from VideoLoader

### Outputs
- `processed_frames` — list of `(64, 64)` float32 arrays, values ∈ [0, 1]

### Processing Chain (per frame)
```
BGR frame (H×W×3, uint8)
    ↓  cv2.resize(..., INTER_AREA)
(64×64×3, uint8)
    ↓  cv2.cvtColor(..., COLOR_BGR2GRAY)
(64×64, uint8)
    ↓  / 255.0
(64×64, float32)  values ∈ [0, 1]
    ↓  [optional] cv2.GaussianBlur(..., (3,3), 0)
(64×64, float32)  smoothed
```

### Key Design Decisions
| Decision | Reason |
|----------|--------|
| 64×64 target | Small enough for laptop CPU/GPU; large enough to retain scene structure |
| Grayscale | Motion analysis doesn't require colour; discarding it halves memory and speeds up SSIM computation |
| float32 [0,1] | Neural networks converge faster with small floating-point inputs; SSIM requires float |
| `INTER_AREA` downsampling | Best quality for shrinking images (anti-aliasing); better than default `INTER_LINEAR` |
| Optional Gaussian blur | Can reduce sensor noise that would artificially inflate SSIM or variance readings; off by default |
| Separate class | Future changes (e.g. different resolution, RGB mode, augmentation) only touch this class |

---

## Module 3 — Guard Layer G1 (Pixel Variance)

**File**: `preprocess_pipeline.py` → class `GuardG1Variance`

### Purpose
First guard layer. Detects "frozen" clips — windows of 16 frames where the camera feed is static, stuck, or repeating.

### Algorithm

$$\text{Variance} = \frac{1}{HW} \sum_{h,w} \text{Var}_{t=0}^{T-1}\bigl(F_t[h,w]\bigr)$$

1. Stack 16 frames → tensor $F \in \mathbb{R}^{T \times 64 \times 64}$
2. Compute pixel-wise variance along time axis → $V \in \mathbb{R}^{64 \times 64}$
3. Take the mean across all pixels → scalar `mean_var`
4. If `mean_var < θ_v` → **FROZEN**

### Inputs
- `window` — list of 16 preprocessed `(64, 64)` float32 frames

### Output Format (exactly as specified)
```python
# If frozen:
{"status": "FROZEN", "reason": "Low Pixel Variance", "variance": float}

# If normal:
{"status": "PASS", "variance": float}
```

### Key Design Decisions
| Decision | Reason |
|----------|--------|
| Variance along time axis | Measures how much each pixel changes over time — zero variance = no change = frozen |
| Mean over all pixels | A single scalar is easier to threshold than a 64×64 map |
| θ_v configurable | Different cameras and scenes may need different sensitivity |
| Returns dict, not string | Structured output is easier to log, filter, and pass to downstream modules |

---

## Module 4 — Guard Layer G2 (SSIM)

**File**: `preprocess_pipeline.py` → class `GuardG2SSIM`

### Purpose
Second guard layer. Catches frozen or near-frozen clips that G1 might miss (e.g. very slowly changing scenes where variance is non-zero but frames are structurally identical).

### Algorithm

For 16 frames, compute 15 pairwise SSIM scores:
$$S_i = \text{SSIM}(F_i, F_{i+1}), \quad i = 0 \ldots 14$$

Find the longest consecutive run of scores exceeding the threshold:
$$\text{run} = \max_k \bigl|\{S_i, S_{i+1}, \ldots, S_{i+k}\} : \text{all} > \theta_s\bigr|$$

If $\text{run} \geq 8$ → **FROZEN**

### Inputs
- `window` — list of 16 preprocessed `(64, 64)` float32 frames

### Output Format (exactly as specified)
```python
# If frozen:
{"status": "FROZEN", "reason": "High SSIM", "count": int, "ssim_scores": list}

# If normal:
{"status": "PASS", "max_run": int, "ssim_scores": list}
```

### Key Design Decisions
| Decision | Reason |
|----------|--------|
| SSIM (not just MSE) | SSIM evaluates luminance, contrast, AND structure — far more perceptually accurate than pixel-wise difference |
| Consecutive run, not total count | A frozen camera produces a sustained plateau of high SSIM, not just a few scattered high scores |
| Run length threshold = 8 | At 16 fps, 8 consecutive identical pairs ≈ 0.5 s of frozen feed — a reliable signal |
| Only runs if G1 passes | Avoids wasted SSIM computation on windows G1 already caught |
| `data_range=1.0` | scikit-image's SSIM needs to know the input value range for correct normalisation |

---

## Module 5 — Sliding Window Clip Builder

**File**: `preprocess_pipeline.py` → class `SlidingWindowBuilder`

### Purpose
Builds validated 16-frame clips by sliding a window across all preprocessed frames, running G1 and G2 on each window, and returning structured clip dicts.

### Algorithm
```
for i in range(len(frames) - 16 + 1):
    window = frames[i : i+16]
    g1 = G1.check_window(window) → if FROZEN: record, skip
    g2 = G2.check_window(window) → if FROZEN: record, skip
    both PASS → emit valid clip
```

### Inputs
- `preprocessed_frames` — list of all `(64, 64)` float32 frames
- `frame_indices` — list of original video frame numbers (for provenance)

### Output Format (each clip, exactly as specified)
```python
{
    "clip_id"          : int,
    "frame_indices"    : list[int],       # 16 original frame numbers
    "status"           : "PASS" | "FROZEN",
    "reason"           : str | None,
    "processed_frames" : list[np.ndarray], # 16 × (64,64) float32
    "g1_result"        : dict,
    "g2_result"        : dict | None,
}
```

### Key Design Decisions
| Decision | Reason |
|----------|--------|
| Fully overlapping windows (stride=1) | Maximises training data; every 16-frame temporal context is represented |
| G1 gates G2 | Saves SSIM computation time on clearly frozen windows |
| Both guards gate clip builder | Downstream module receives only clean data |
| All clips returned (PASS + FROZEN) | Allows the caller to audit rejections, compute statistics, or visualise frozen detections |
| `frame_indices` in output | The next developer can map any clip back to the original video timestamp |
| Clips built on preprocessed frames | NOT on feature maps — features are the next module's job |

---

## Helper Functions

### `get_valid_clips(clips)`
Filters the clip list to return only `status == "PASS"` clips.  
**Use this before handing clips to the next developer.**

### `clips_to_numpy(clips)`
Stacks all valid clips into a single NumPy array of shape `(N, 16, 64, 64)`.  
Useful for saving to `.npy` or feeding into a DataLoader.

---

## What Comes Next (NOT this module)

The output of this module — a list of validated 16-frame clips — is the input to:

| Stream | What it does |
|--------|-------------|
| **Stream A** | Frame Difference — computes pixel-level motion maps |
| **Stream B** | Optical Flow — computes dense Farneback flow magnitudes |
| **Dual LSTM AE** | Encodes and reconstructs temporal motion patterns |
| **Divergence Score** | Measures reconstruction error between streams |
| **Threshold + Classification** | Labels clips as normal or anomalous |

---

## Summary of Conformance with Specification

| Spec Requirement | Status |
|------------------|--------|
| Accept mp4/avi/mov | ✅ |
| Read FPS, resolution, total frames, duration | ✅ |
| Handle corrupted videos | ✅ (RuntimeError) |
| tqdm progress bar | ✅ |
| Extract frames sequentially | ✅ |
| Preserve timestamps | ✅ |
| Configurable frame skipping | ✅ |
| Resize to 64×64 | ✅ |
| Grayscale | ✅ |
| Normalize to [0,1] | ✅ |
| Optional Gaussian Blur | ✅ |
| G1 — variance < θ_v → FROZEN | ✅ |
| G1 — sliding window T=16 | ✅ |
| G1 — return `{status,reason,variance}` | ✅ |
| G2 — SSIM between consecutive pairs | ✅ |
| G2 — ≥8 consecutive pairs > 0.999 → FROZEN | ✅ |
| G2 — return `{status,reason,count}` | ✅ |
| G2 only runs if G1 passes | ✅ |
| Clip builder only runs if both pass | ✅ |
| Clips of exactly 16 frames | ✅ |
| Output format `{clip_id, frame_indices, status, reason, processed_frames}` | ✅ |
| NO frame difference | ✅ removed |
| NO optical flow | ✅ removed |
| NO LSTM / divergence / classification | ✅ not implemented |

"""
Generates a professional frozen-frame detection visualization for the IEEE paper.
Shows: (a) normal frames, (b) simulated frozen frames, (c) SSIM scores over time,
(d) Guard Layer G1/G2 flags, demonstrating the guard layer detecting frozen feeds.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim

# ──────────────────────────────────────────────────────────────────────────────
VIDEO_PATH = r"D:\AICity21-Track4-Anomaly-Detection\AIC21-Track4-Anomaly-Detection\aic21-track4-train-data\1.mp4"
OUTPUT_PATH = r"D:\AICity21-Track4-Anomaly-Detection\processed_output\frozen_frame_detection.png"
FRAME_SIZE  = (64, 64)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(frame):
    resized = cv2.resize(frame, FRAME_SIZE)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0

def get_ssim(a, b):
    return ssim(a, b, data_range=1.0)

# ── Load frames ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
raw_frames = []
while len(raw_frames) < 80:
    ret, frame = cap.read()
    if not ret:
        break
    raw_frames.append(frame)
cap.release()

preprocessed = [preprocess(f) for f in raw_frames]

# ── Inject freeze: frames 40-59 are frozen (repeat frame 39) ─────────────────
FREEZE_START = 40
FREEZE_END   = 60
frozen_frames = preprocessed.copy()
for i in range(FREEZE_START, FREEZE_END):
    frozen_frames[i] = preprocessed[FREEZE_START - 1].copy()

# ── Compute SSIM between consecutive frames (faulty stream) ──────────────────
ssim_scores = []
for i in range(1, len(frozen_frames)):
    s = get_ssim(frozen_frames[i - 1], frozen_frames[i])
    ssim_scores.append(s)

# Guard flags
SSIM_THRESHOLD   = 0.999
VARIANCE_THRESH  = 0.001
WINDOW_SIZE      = 16

g1_flags = []
for i in range(len(frozen_frames) - WINDOW_SIZE + 1):
    window  = frozen_frames[i : i + WINDOW_SIZE]
    stacked = np.stack(window, axis=0)
    mean_var = np.mean(np.var(stacked, axis=0))
    g1_flags.append(1 if mean_var < VARIANCE_THRESH else 0)

g2_flags = []
for i in range(len(ssim_scores) - 10 + 1):
    window_scores = ssim_scores[i : i + 10]
    g2_flags.append(1 if all(s > SSIM_THRESHOLD for s in window_scores) else 0)

# ── Pick display frames ───────────────────────────────────────────────────────
normal_idx = [10, 20, 30]
frozen_idx = [40, 42, 44]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#0f0f1a')

gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.35,
                        top=0.92, bottom=0.08, left=0.05, right=0.97)

TITLE_COLOR  = '#e0e0ff'
LABEL_COLOR  = '#b0b0d0'
NORMAL_COLOR = '#4caf87'
FROZEN_COLOR = '#e05c5c'

fig.suptitle('Guard Layer: Frozen Frame Detection in Surveillance Feed',
             fontsize=14, color=TITLE_COLOR, fontweight='bold', y=0.97)

# ── Row 0: Normal frames ──────────────────────────────────────────────────────
for col, idx in enumerate(normal_idx):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(frozen_frames[idx], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Frame {idx}\n(Normal)', color=NORMAL_COLOR, fontsize=8.5, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(NORMAL_COLOR)
        spine.set_linewidth(2)
    ax.set_aspect('equal')

# ── Row 0: Frozen frames ─────────────────────────────────────────────────────
for col, idx in enumerate(frozen_idx):
    ax = fig.add_subplot(gs[0, col + 3])
    ax.imshow(frozen_frames[idx], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Frame {idx}\n(Frozen)', color=FROZEN_COLOR, fontsize=8.5, fontweight='bold')
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(FROZEN_COLOR)
        spine.set_linewidth(2)
    ax.set_aspect('equal')

# Row 0 section headers
ax_normal_label = fig.add_axes([0.05, 0.90, 0.43, 0.02])
ax_normal_label.set_facecolor('#1a2e1a')
ax_normal_label.text(0.5, 0.5, '  ✔  Normal Feed', ha='center', va='center',
                     color=NORMAL_COLOR, fontsize=9, fontweight='bold')
ax_normal_label.axis('off')

ax_frozen_label = fig.add_axes([0.51, 0.90, 0.45, 0.02])
ax_frozen_label.set_facecolor('#2e1a1a')
ax_frozen_label.text(0.5, 0.5, '  ✘  Frozen Feed (Injected Fault)', ha='center', va='center',
                     color=FROZEN_COLOR, fontsize=9, fontweight='bold')
ax_frozen_label.axis('off')

# ── Row 1: SSIM over time ─────────────────────────────────────────────────────
ax_ssim = fig.add_subplot(gs[1, :])
ax_ssim.set_facecolor('#12122a')
x = list(range(len(ssim_scores)))
ax_ssim.plot(x, ssim_scores, color='#7eb8f7', linewidth=1.2, label='SSIM (consecutive frames)')
ax_ssim.axhline(y=SSIM_THRESHOLD, color='#e05c5c', linewidth=1.4, linestyle='--',
                label=f'Threshold = {SSIM_THRESHOLD}')
ax_ssim.axvspan(FREEZE_START, FREEZE_END - 1, color='#e05c5c', alpha=0.18, label='Injected Freeze')
ax_ssim.text((FREEZE_START + FREEZE_END) / 2, 0.97, 'FREEZE\nZONE',
             ha='center', va='top', color='#e05c5c', fontsize=8, fontweight='bold')
ax_ssim.set_xlabel('Frame Index', color=LABEL_COLOR, fontsize=9)
ax_ssim.set_ylabel('SSIM Score', color=LABEL_COLOR, fontsize=9)
ax_ssim.set_title('G2 — SSIM Between Consecutive Frames (Spike → Freeze Detected)',
                  color=TITLE_COLOR, fontsize=10, fontweight='bold')
ax_ssim.tick_params(colors=LABEL_COLOR, labelsize=8)
for spine in ax_ssim.spines.values():
    spine.set_edgecolor('#333355')
ax_ssim.set_ylim(0.7, 1.05)
ax_ssim.legend(loc='lower left', fontsize=8, facecolor='#12122a', edgecolor='#333355',
               labelcolor=LABEL_COLOR)
ax_ssim.grid(True, alpha=0.2, color='#333355')

# ── Row 2: Guard flags bar ────────────────────────────────────────────────────
ax_g1 = fig.add_subplot(gs[2, :3])
ax_g1.set_facecolor('#12122a')
colors_g1 = [FROZEN_COLOR if v == 1 else NORMAL_COLOR for v in g1_flags]
ax_g1.bar(range(len(g1_flags)), g1_flags, color=colors_g1, width=1.0)
ax_g1.set_title('G1 — Pixel Variance Guard (Red = Frozen Window)',
                color=TITLE_COLOR, fontsize=9.5, fontweight='bold')
ax_g1.set_xlabel('Window Index', color=LABEL_COLOR, fontsize=8)
ax_g1.set_ylabel('Frozen Flag', color=LABEL_COLOR, fontsize=8)
ax_g1.tick_params(colors=LABEL_COLOR, labelsize=7)
ax_g1.set_ylim(0, 1.3)
for spine in ax_g1.spines.values():
    spine.set_edgecolor('#333355')
ax_g1.grid(True, alpha=0.15, color='#333355')
patch_n = mpatches.Patch(color=NORMAL_COLOR, label='Normal')
patch_f = mpatches.Patch(color=FROZEN_COLOR, label='Frozen')
ax_g1.legend(handles=[patch_n, patch_f], fontsize=8, facecolor='#12122a',
             edgecolor='#333355', labelcolor=LABEL_COLOR)

ax_g2 = fig.add_subplot(gs[2, 3:])
ax_g2.set_facecolor('#12122a')
colors_g2 = [FROZEN_COLOR if v == 1 else NORMAL_COLOR for v in g2_flags]
ax_g2.bar(range(len(g2_flags)), g2_flags, color=colors_g2, width=1.0)
ax_g2.set_title('G2 — SSIM Guard (Red = Frozen Region Detected)',
                color=TITLE_COLOR, fontsize=9.5, fontweight='bold')
ax_g2.set_xlabel('Frame Index', color=LABEL_COLOR, fontsize=8)
ax_g2.set_ylabel('Frozen Flag', color=LABEL_COLOR, fontsize=8)
ax_g2.tick_params(colors=LABEL_COLOR, labelsize=7)
ax_g2.set_ylim(0, 1.3)
for spine in ax_g2.spines.values():
    spine.set_edgecolor('#333355')
ax_g2.grid(True, alpha=0.15, color='#333355')
ax_g2.legend(handles=[patch_n, patch_f], fontsize=8, facecolor='#12122a',
             edgecolor='#333355', labelcolor=LABEL_COLOR)

plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {OUTPUT_PATH}")

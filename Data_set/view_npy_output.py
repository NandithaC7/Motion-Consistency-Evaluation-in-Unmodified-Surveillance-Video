"""
=============================================================================
NPY FILE VIEWER — View your saved pipeline output
Project: Motion Consistency Evaluation in Unmodified Surveillance Video
=============================================================================

HOW TO RUN:
    python view_npy_output.py

This script:
  1. Loads the saved .npy files (Stream A and Stream B)
  2. Prints stats (shape, min, max, mean)
  3. Saves visualizations of sample sequences to the output folder
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = r"D:\AICity21-Track4-Anomaly-Detection\processed_output"
PATH_A       = os.path.join(OUTPUT_DIR, "1_stream_A_frame_diff.npy")
PATH_B       = os.path.join(OUTPUT_DIR, "1_stream_B_optical_flow.npy")


# =============================================================================
# 1. LOAD THE FILES
# =============================================================================
print("=" * 60)
print("  NPY FILE VIEWER")
print("=" * 60)

print(f"\nLoading Stream A  ->  {PATH_A}")
seq_a = np.load(PATH_A)          # shape: (N, 16, 64, 64)

print(f"Loading Stream B  ->  {PATH_B}")
seq_b = np.load(PATH_B)          # shape: (N, 16, 64, 64)


# =============================================================================
# 2. PRINT STATISTICS  (what your professor wants to see as "proof it worked")
# =============================================================================
print("\n" + "-" * 60)
print("STREAM A  -  Frame Difference Sequences")
print("-" * 60)
print(f"  Array shape : {seq_a.shape}   (sequences, frames, height, width)")
print(f"  Data type   : {seq_a.dtype}")
print(f"  Min value   : {seq_a.min():.6f}")
print(f"  Max value   : {seq_a.max():.6f}")
print(f"  Mean value  : {seq_a.mean():.6f}")
print(f"  File size   : {os.path.getsize(PATH_A) / 1e6:.1f} MB")

print("\n" + "-" * 60)
print("STREAM B  -  Optical Flow Magnitude Sequences")
print("-" * 60)
print(f"  Array shape : {seq_b.shape}   (sequences, frames, height, width)")
print(f"  Data type   : {seq_b.dtype}")
print(f"  Min value   : {seq_b.min():.6f}")
print(f"  Max value   : {seq_b.max():.6f}")
print(f"  Mean value  : {seq_b.mean():.6f}")
print(f"  File size   : {os.path.getsize(PATH_B) / 1e6:.1f} MB")
print("-" * 60)


# =============================================================================
# 3. VISUALIZE SEQUENCE #0 — show all 16 frames of a single sequence
# =============================================================================
def plot_sequence(seq, title, colormap, save_name):
    """
    Plots all 16 frames inside one sequence clip as a 2-row grid.
    This shows what a single training sample looks like.
    """
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx in range(16):
        row = idx // 8
        col = idx % 8
        ax  = axes[row, col]
        ax.imshow(seq[idx], cmap=colormap, vmin=0, vmax=1)
        ax.set_title(f"Frame {idx}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=130)
    plt.close()
    print(f"  Saved -> {save_path}")


print("\n[VIZ 1] Plotting Stream A - Sequence #0  (Frame Differences)...")
plot_sequence(
    seq_a[0],                                   # first sequence, all 16 frames
    "Stream A - Frame Difference  |  Sequence #0  (16 consecutive frames)",
    colormap="hot",
    save_name="viz_streamA_sequence0.png"
)

print("[VIZ 2] Plotting Stream B - Sequence #0  (Optical Flow Magnitude)...")
plot_sequence(
    seq_b[0],
    "Stream B - Optical Flow Magnitude  |  Sequence #0  (16 consecutive frames)",
    colormap="jet",
    save_name="viz_streamB_sequence0.png"
)


# =============================================================================
# 4. COMPARE BOTH STREAMS SIDE BY SIDE (same frame from same sequence)
# =============================================================================
print("[VIZ 3] Plotting side-by-side comparison of both streams...")

fig, axes = plt.subplots(2, 8, figsize=(22, 6))
fig.suptitle(
    "Side-by-Side: Stream A (Frame Diff) vs Stream B (Optical Flow)\n"
    "Top Row = Stream A | Bottom Row = Stream B | Same Sequence, Same Frames",
    fontsize=11, fontweight="bold"
)

for i in range(8):   # show first 8 frames of sequence #5 (middle of video)
    seq_idx = min(5, len(seq_a) - 1)

    axes[0, i].imshow(seq_a[seq_idx][i], cmap="hot",  vmin=0, vmax=1)
    axes[0, i].set_title(f"A-f{i}", fontsize=8)
    axes[0, i].axis("off")

    axes[1, i].imshow(seq_b[seq_idx][i], cmap="jet",  vmin=0, vmax=1)
    axes[1, i].set_title(f"B-f{i}", fontsize=8)
    axes[1, i].axis("off")

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "viz_AB_comparison.png")
plt.savefig(save_path, dpi=130)
plt.close()
print(f"  Saved -> {save_path}")


# =============================================================================
# 5. MOTION INTENSITY OVER TIME (mean Frame Diff per sequence)
# =============================================================================
print("[VIZ 4] Plotting motion intensity over time...")

motion_intensity = seq_a.mean(axis=(1, 2, 3))   # mean pixel change per sequence

plt.figure(figsize=(14, 4))
plt.plot(motion_intensity, color="#e05c5c", linewidth=1.2)
plt.fill_between(range(len(motion_intensity)), motion_intensity,
                 alpha=0.25, color="#e05c5c")
plt.title("Motion Intensity Over Time  (Mean Frame Difference per Sequence)",
          fontsize=12, fontweight="bold")
plt.xlabel("Sequence Index (each = 16 frames, sliding by 1)")
plt.ylabel("Mean Pixel Change [0-1]")
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "viz_motion_intensity_over_time.png")
plt.savefig(save_path, dpi=130)
plt.close()
print(f"  Saved -> {save_path}")


# =============================================================================
# 6. SUMMARY REPORT
# =============================================================================
N = seq_a.shape[0]
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Total sequences generated    : {N}")
print(f"  Each sequence length         : 16 frames")
print(f"  Each frame size              : 64x64 pixels")
print(f"  Stream A data size           : {seq_a.nbytes / 1e6:.1f} MB (in memory)")
print(f"  Stream B data size           : {seq_b.nbytes / 1e6:.1f} MB (in memory)")
print(f"\n  Visualizations saved to      : {OUTPUT_DIR}")
print("    - viz_streamA_sequence0.png")
print("    - viz_streamB_sequence0.png")
print("    - viz_AB_comparison.png")
print("    - viz_motion_intensity_over_time.png")
print("=" * 60)
print("\n  DONE! Open the images above to show your professor the output.")
print("=" * 60)

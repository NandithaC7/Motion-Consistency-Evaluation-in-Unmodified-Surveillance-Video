"""
=============================================================================
TEST SCRIPT — Preprocessing Pipeline
Project : Motion Consistency Evaluation in Unmodified Surveillance Video
=============================================================================

Runs the preprocessing pipeline and verifies outputs at each stage.

HOW TO RUN:
    python test_pipeline.py

What this script checks:
    1. VideoLoader    — reads frames, metadata
    2. FramePreprocessor — correct shape, dtype, value range
    3. GuardG1Variance   — correct return format
    4. GuardG2SSIM       — correct return format
    5. SlidingWindowBuilder — correct clip output format
    6. Pipeline end-to-end run
=============================================================================
"""

import numpy as np
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Allow import from same directory ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess_pipeline import (
    PipelineConfig,
    VideoLoader,
    FramePreprocessor,
    GuardG1Variance,
    GuardG2SSIM,
    SlidingWindowBuilder,
    PreprocessingPipeline,
    get_valid_clips,
    clips_to_numpy,
)

# ── Video path — change this to any .mp4 / .avi / .mov on your machine ───────
VIDEO_PATH = r"D:\AICity21-Track4-Anomaly-Detection\AIC21-Track4-Anomaly-Detection\aic21-track4-train-data\1.mp4"


# =============================================================================
# Helper
# =============================================================================

PASS_COUNT = 0
FAIL_COUNT = 0

def check(label: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✔  {label}")
    else:
        FAIL_COUNT += 1
        print(f"  ✘  {label}  ← FAILED  {detail}")


# =============================================================================
# Unit test: Synthetic frozen frames (no video needed)
# =============================================================================

def test_guard_g1_with_synthetic_data():
    print("\n── Test 1: G1 on synthetic frozen frames ──────────────────────")
    config = PipelineConfig()
    g1 = GuardG1Variance(config)

    # Frozen: all 16 frames identical
    frozen_frame = np.full((64, 64), 0.5, dtype=np.float32)
    frozen_window = [frozen_frame.copy() for _ in range(16)]
    result = g1.check_window(frozen_window)
    check("G1 detects frozen window",            result["status"] == "FROZEN")
    check("G1 frozen reason key exists",          "reason"   in result)
    check("G1 frozen reason text correct",        result["reason"] == "Low Pixel Variance")
    check("G1 variance key exists",               "variance" in result)
    check("G1 variance is float",                 isinstance(result["variance"], float))

    # Normal: random frames
    normal_window = [np.random.rand(64, 64).astype(np.float32) for _ in range(16)]
    result2 = g1.check_window(normal_window)
    check("G1 passes normal window",              result2["status"] == "PASS")


def test_guard_g2_with_synthetic_data():
    print("\n── Test 2: G2 on synthetic frozen frames ──────────────────────")
    config = PipelineConfig()
    g2 = GuardG2SSIM(config)

    # Frozen: 16 identical frames → all SSIM = 1.0
    frozen_frame  = np.random.rand(64, 64).astype(np.float32)
    frozen_window = [frozen_frame.copy() for _ in range(16)]
    result = g2.check_window(frozen_window)
    check("G2 detects frozen window",             result["status"] == "FROZEN")
    check("G2 frozen reason key exists",           "reason" in result)
    check("G2 frozen reason text correct",         result["reason"] == "High SSIM")
    check("G2 count key exists",                   "count"  in result)
    check("G2 count is int",                       isinstance(result["count"], int))
    check("G2 consecutive count >= 8",             result["count"] >= 8)

    # Normal: random frames
    normal_window = [np.random.rand(64, 64).astype(np.float32) for _ in range(16)]
    result2 = g2.check_window(normal_window)
    check("G2 passes normal window",               result2["status"] == "PASS")
    check("G2 ssim_scores list in result",         "ssim_scores" in result2)
    check("G2 ssim_scores length = 15",            len(result2["ssim_scores"]) == 15)


def test_frame_preprocessor_shape():
    print("\n── Test 3: FramePreprocessor output shape / dtype ─────────────")
    import cv2
    config = PipelineConfig()
    fp = FramePreprocessor(config)

    # Create a dummy BGR frame (480×640×3)
    dummy_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = fp.process_frame(dummy_bgr)

    check("Output shape is (64, 64)",              out.shape == (64, 64))
    check("Output dtype is float32",               out.dtype == np.float32)
    check("Output min >= 0.0",                     float(out.min()) >= 0.0)
    check("Output max <= 1.0",                     float(out.max()) <= 1.0)


def test_clip_output_format_synthetic():
    print("\n── Test 4: SlidingWindowBuilder clip format (synthetic) ────────")
    config = PipelineConfig()
    config.window_size = 16
    g1 = GuardG1Variance(config)
    g2 = GuardG2SSIM(config)
    builder = SlidingWindowBuilder(config, g1, g2)

    # Create 30 random frames → should produce 30-16+1 = 15 windows
    frames  = [np.random.rand(64, 64).astype(np.float32) for _ in range(30)]
    indices = list(range(30))
    clips   = builder.build(frames, indices)

    check("Number of clips = 15",                  len(clips) == 15)

    clip = clips[0]
    check("clip has 'clip_id'",                    "clip_id"          in clip)
    check("clip has 'frame_indices'",              "frame_indices"    in clip)
    check("clip has 'status'",                     "status"           in clip)
    check("clip has 'reason'",                     "reason"           in clip)
    check("clip has 'processed_frames'",           "processed_frames" in clip)
    check("clip has 'g1_result'",                  "g1_result"        in clip)
    check("clip has 'g2_result'",                  "g2_result"        in clip)
    check("processed_frames length = 16",          len(clip["processed_frames"]) == 16)
    check("frame_indices length = 16",             len(clip["frame_indices"])    == 16)
    check("each frame shape = (64,64)",
          all(f.shape == (64, 64) for f in clip["processed_frames"]))


def test_get_valid_clips():
    print("\n── Test 5: get_valid_clips() helper ───────────────────────────")
    mock_clips = [
        {"status": "PASS",   "processed_frames": []},
        {"status": "FROZEN", "processed_frames": []},
        {"status": "PASS",   "processed_frames": []},
    ]
    valid = get_valid_clips(mock_clips)
    check("Returns only PASS clips",               len(valid) == 2)
    check("All returned clips have status PASS",
          all(c["status"] == "PASS" for c in valid))


def test_clips_to_numpy():
    print("\n── Test 6: clips_to_numpy() stacking ─────────────────────────")
    frames  = [np.random.rand(64, 64).astype(np.float32) for _ in range(16)]
    mock_clips = [
        {"status": "PASS", "processed_frames": frames},
        {"status": "PASS", "processed_frames": frames},
    ]
    arr = clips_to_numpy(mock_clips)
    check("clips_to_numpy shape = (2, 16, 64, 64)",  arr.shape == (2, 16, 64, 64))
    check("dtype is float32",                         arr.dtype == np.float32)


# =============================================================================
# Integration test: real video (skipped if file not found)
# =============================================================================

def test_full_pipeline_on_real_video():
    print("\n── Test 7: Full pipeline on real video ────────────────────────")

    if not os.path.exists(VIDEO_PATH):
        print(f"  [SKIP] Video not found: {VIDEO_PATH}")
        return

    config = PipelineConfig()
    config.max_frames    = 100   # keep test fast
    config.gaussian_blur = False
    config.debug_mode    = False

    pipeline = PreprocessingPipeline(config)
    clips, meta = pipeline.run(VIDEO_PATH)

    check("Pipeline returns list",                 isinstance(clips, list))
    check("Metadata has 'fps'",                    "fps" in meta)
    check("Metadata has 'resolution'",             "resolution" in meta)
    check("Metadata has 'total_frames'",           "total_frames" in meta)
    check("Metadata has 'duration_seconds'",       "duration_seconds" in meta)
    check("At least 1 clip produced",              len(clips) >= 1)

    valid = get_valid_clips(clips)
    print(f"  Valid clips : {len(valid)}  /  {len(clips)} total")

    if valid:
        arr = clips_to_numpy(valid)
        check("clips_to_numpy shape correct",
              arr.shape == (len(valid), 16, 64, 64))
        check("Values in [0,1]",
              float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0)

        # Verify no Stream A / B code is in scope
        from preprocess_pipeline import PreprocessingPipeline as PP
        check("compute_frame_differences NOT in module",
              not hasattr(PP, "compute_frame_differences"))
        check("compute_optical_flow NOT in module",
              not hasattr(PP, "compute_optical_flow"))


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PREPROCESSING PIPELINE — TEST SUITE")
    print("=" * 65)

    test_guard_g1_with_synthetic_data()
    test_guard_g2_with_synthetic_data()
    test_frame_preprocessor_shape()
    test_clip_output_format_synthetic()
    test_get_valid_clips()
    test_clips_to_numpy()
    test_full_pipeline_on_real_video()

    print("\n" + "=" * 65)
    total = PASS_COUNT + FAIL_COUNT
    print(f"  RESULTS: {PASS_COUNT}/{total} passed")
    if FAIL_COUNT == 0:
        print("  ✔  All tests passed!")
    else:
        print(f"  ✘  {FAIL_COUNT} test(s) failed — check output above.")
    print("=" * 65)

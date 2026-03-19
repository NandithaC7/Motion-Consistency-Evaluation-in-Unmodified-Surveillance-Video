"""
frame_extractor.py
------------------
Extracts frames from ShanghaiTech Campus Dataset video clips.

ShanghaiTech Dataset Structure (expected):
    data/raw/
        training/
            videos/
                01_001.avi
                01_002.avi
                ...
        testing/
            frames/
                01_0014/
                    000.jpg
                    001.jpg
                    ...

What this file does:
    - Reads .avi video files from the training split
    - Extracts individual frames at a configurable FPS
    - Converts to grayscale (as described in the PPT preprocessing step)
    - Resizes frames to a standard resolution
    - Saves extracted frames as .jpg files in data/processed/frames/
    - Logs extraction metadata (total frames, FPS, resolution) to a CSV

Why grayscale?
    Motion consistency analysis only needs brightness changes between frames,
    not color information. Grayscale reduces memory and computation by ~3x.

Usage:
    python frame_extractor.py --video_dir data/raw/training/videos
                              --output_dir data/processed/frames
                              --target_fps 10
                              --width 320 --height 240
"""

import os
import cv2
import csv
import argparse
import logging
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Core extractor ───────────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: int = 10,
    width: int = 320,
    height: int = 240,
    grayscale: bool = True
) -> dict:
    """
    Extract frames from a single video file.

    Args:
        video_path  : Path to the .avi video file
        output_dir  : Directory where extracted frames will be saved
        target_fps  : How many frames per second to extract.
                      ShanghaiTech is recorded at 24fps. If target_fps=10,
                      we sample every (24/10) ≈ 2-3 frames → keeps motion
                      signal without storing all 24 frames per second.
        width       : Resize frame to this width (pixels)
        height      : Resize frame to this height (pixels)
        grayscale   : Convert to single-channel grayscale

    Returns:
        dict with metadata: video_name, total_frames, extracted_frames,
                            original_fps, duration_sec, output_dir
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = video_path.stem                          # e.g. "01_001"
    frame_output_dir = Path(output_dir) / video_name
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps if original_fps > 0 else 0

    # How many original frames to skip between each extracted frame
    # Example: original 24fps, target 10fps → sample_interval ≈ 2
    sample_interval = max(1, round(original_fps / target_fps))

    frame_idx = 0           # counts every frame read from video
    saved_idx = 0           # counts frames we actually saved

    logger.info(f"Processing: {video_name} | "
                f"Original FPS: {original_fps:.1f} | "
                f"Total frames: {total_frames} | "
                f"Sample interval: {sample_interval}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save every Nth frame (temporal sampling from PPT slide)
        if frame_idx % sample_interval == 0:
            # Step 1: Resize (spatial normalization from PPT)
            frame_resized = cv2.resize(frame, (width, height))

            # Step 2: Grayscale conversion (from PPT preprocessing engine)
            if grayscale:
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Save as zero-padded filename: 000001.jpg
            filename = frame_output_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(filename), frame_resized)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    metadata = {
        "video_name": video_name,
        "total_frames_in_video": total_frames,
        "extracted_frames": saved_idx,
        "original_fps": round(original_fps, 2),
        "target_fps": target_fps,
        "duration_sec": round(duration_sec, 2),
        "width": width,
        "height": height,
        "grayscale": grayscale,
        "output_dir": str(frame_output_dir)
    }

    logger.info(f"Done: {video_name} → {saved_idx} frames saved to {frame_output_dir}")
    return metadata


def extract_all_videos(
    video_dir: str,
    output_dir: str,
    target_fps: int = 10,
    width: int = 320,
    height: int = 240,
    grayscale: bool = True,
    log_csv: str = "data/processed/extraction_log.csv"
) -> list:
    """
    Process all .avi files in a directory (the entire training split).

    Args:
        video_dir : Path to folder containing .avi files
        output_dir: Where to save extracted frames
        log_csv   : Path to save per-video metadata CSV

    Returns:
        List of metadata dicts (one per video)
    """
    video_dir = Path(video_dir)
    video_files = sorted(video_dir.glob("*.avi"))

    if not video_files:
        logger.warning(f"No .avi files found in {video_dir}")
        return []

    logger.info(f"Found {len(video_files)} videos in {video_dir}")

    all_metadata = []

    for video_path in video_files:
        try:
            meta = extract_frames_from_video(
                video_path=str(video_path),
                output_dir=output_dir,
                target_fps=target_fps,
                width=width,
                height=height,
                grayscale=grayscale
            )
            all_metadata.append(meta)

        except Exception as e:
            logger.error(f"Failed on {video_path.name}: {e}")
            continue

    # Save metadata log
    if all_metadata:
        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)
        logger.info(f"Extraction log saved to {log_path}")

    return all_metadata


def load_frames_from_dir(frame_dir: str) -> list:
    """
    Load saved frames back into memory as a list of numpy arrays.
    Used by motion_extractor.py to read frames for a specific video.

    Args:
        frame_dir : Path to directory containing .jpg frames for one video

    Returns:
        List of grayscale numpy arrays, sorted by filename
    """
    frame_dir = Path(frame_dir)
    frame_files = sorted(frame_dir.glob("*.jpg"))

    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    frames = []
    for fpath in frame_files:
        frame = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            frames.append(frame)

    logger.info(f"Loaded {len(frames)} frames from {frame_dir}")
    return frames


# ── CLI entry point ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from ShanghaiTech surveillance videos"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="data/raw/training/videos",
        help="Directory containing .avi video files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/frames",
        help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=10,
        help="Target frames per second to extract (original is 24fps)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Frame width after resize"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Frame height after resize"
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default="data/processed/extraction_log.csv",
        help="Path to save extraction metadata CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = extract_all_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        width=args.width,
        height=args.height,
        log_csv=args.log_csv
    )

    print(f"\n Extraction complete: {len(results)} videos processed.")
    for r in results:
        print(f"   {r['video_name']}: {r['extracted_frames']} frames "
              f"({r['duration_sec']}s @ {r['original_fps']}fps original)")
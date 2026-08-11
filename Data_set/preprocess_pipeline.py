"""
=============================================================================
PREPROCESSING PIPELINE
Project : Motion Consistency Evaluation in Unmodified Surveillance Video
Author  : Final Year Project — Preprocessing Module
=============================================================================

PIPELINE OVERVIEW (this module only):
  Raw Video
    └─► Video Loader          (metadata + frame extraction)
    └─► Frame Preprocessing   (resize → grayscale → normalize → optional blur)
    └─► Guard Layer G1        (pixel variance — detect frozen clips)
    └─► Guard Layer G2        (SSIM — detect near-identical frames)
    └─► Sliding Window Builder (produce 16-frame validated clips)
    └─► Output: list of clip dicts ready for downstream modules

STOP POINT:
  This module ends here. It does NOT implement:
    - Stream A  (Motion Detection / Frame Difference)
    - Stream B  (Optical Flow)
    - Dual LSTM Autoencoder
    - Divergence Scoring
    - Thresholding or Classification

DEPENDENCIES:
  pip install opencv-python numpy scikit-image tqdm matplotlib
=============================================================================
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — saves figures to file
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

class PipelineConfig:
    """
    Central configuration object.  Edit values here — every module reads
    from this class so there are no scattered magic numbers.

    Attributes
    ----------
    frame_size : tuple
        Target (width, height) after resizing.  Kept small (64×64) for
        speed and memory efficiency on a laptop GPU / CPU.
    window_size : int
        Number of consecutive frames that form one clip (16 per spec).
    variance_threshold : float
        G1 threshold θ_v.  A clip whose mean pixel variance falls below
        this is flagged as FROZEN.
    ssim_threshold : float
        G2 per-pair SSIM threshold.  Frame pairs above this are
        considered structurally identical.
    ssim_consecutive : int
        G2 run length.  If ≥ this many consecutive pairs all exceed
        ssim_threshold the clip is flagged as FROZEN.  Spec = 8.
    gaussian_blur : bool
        Apply a 3×3 Gaussian blur to each frame after normalisation.
    frame_skip : int
        Keep every Nth frame (1 = keep all, 2 = keep every other, …).
    max_frames : int or None
        Cap on how many frames to read.  None = entire video.
    debug_mode : bool
        When True, also save individual frames to disk and generate
        diagnostic plots.
    debug_output_dir : str
        Folder used in debug mode for saved images.
    """

    def __init__(self):
        self.frame_size          : Tuple[int, int] = (64, 64)
        self.window_size         : int   = 16
        self.variance_threshold  : float = 0.0001      # θ_v  (G1)
        #  NOTE: This dataset (AICity21 Track4) uses a static overhead
        #  camera.  Genuine motion produces variance ≈ 0.0003–0.0005.
        #  A truly frozen feed (identical frames) has variance < 0.0001.
        #  Raise this value (e.g. to 0.001) only if your target camera
        #  has higher inherent motion / noise.
        self.ssim_threshold      : float = 0.999       # G2 per-pair threshold
        self.ssim_consecutive    : int   = 8           # G2 run-length (spec = 8)
        self.gaussian_blur       : bool  = False       # optional pre-blur
        self.frame_skip          : int   = 1           # 1 = no skipping
        self.max_frames          : Optional[int] = None
        self.debug_mode          : bool  = False
        self.debug_output_dir    : str   = str(Path(__file__).parent / "processed_output")


# =============================================================================
# MODULE 1 — VIDEO LOADER
# =============================================================================

class VideoLoader:
    """
    Loads a surveillance video and extracts raw frames with metadata.

    Supported formats : .mp4, .avi, .mov
    Handles corrupted videos gracefully (logs a warning, returns empty list).
    Shows a tqdm progress bar while reading.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (frame_skip, max_frames).
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov"}

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    def load(self, video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Opens the video and reads frames into memory.

        Parameters
        ----------
        video_path : str
            Absolute path to the video file.

        Returns
        -------
        raw_frames : list of np.ndarray
            BGR frames (H × W × 3), unmodified.
        metadata : dict
            fps, resolution, total_frames, duration_seconds, video_name.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        FileNotFoundError
            If the path does not exist.
        RuntimeError
            If OpenCV cannot open the file (e.g. corrupted container).
        """
        path = Path(video_path)

        # ── Validate extension ────────────────────────────────────────
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'.  "
                f"Accepted: {self.SUPPORTED_EXTENSIONS}"
            )

        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # ── Open capture ─────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open the video (possibly corrupted): {video_path}"
            )

        # ── Read metadata ─────────────────────────────────────────────
        total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps                   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width                 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height                = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s            = total_frames_reported / fps if fps > 0 else 0.0

        metadata = {
            "video_name"       : path.stem,
            "fps"              : fps,
            "resolution"       : (width, height),
            "total_frames"     : total_frames_reported,
            "duration_seconds" : round(duration_s, 3),
        }

        print(f"\n[Module 1] Video Loader")
        print(f"  Path       : {video_path}")
        print(f"  FPS        : {fps:.2f}")
        print(f"  Resolution : {width}x{height}")
        print(f"  Frames     : {total_frames_reported}")
        print(f"  Duration   : {duration_s:.2f}s")

        # ── Determine how many frames to read ────────────────────────
        limit = self.config.max_frames or total_frames_reported

        # ── Read frames with progress bar ─────────────────────────────
        raw_frames : List[np.ndarray] = []
        timestamps : List[float]      = []    # seconds per frame
        frame_indices : List[int]     = []    # original frame number in video
        read_count = 0

        pbar = tqdm(total=min(limit, total_frames_reported),
                    desc="  Reading frames", unit="fr", ncols=70)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Timestamps and index (before skip logic)
            orig_idx  = read_count
            timestamp = orig_idx / fps

            read_count += 1

            # Apply frame skipping
            if (read_count - 1) % self.config.frame_skip != 0:
                continue

            raw_frames.append(frame)
            timestamps.append(round(timestamp, 6))
            frame_indices.append(orig_idx)

            pbar.update(1)

            if len(raw_frames) >= limit:
                break

        pbar.close()
        cap.release()

        # Attach per-frame info to metadata
        metadata["frames_extracted"] = len(raw_frames)
        metadata["timestamps"]       = timestamps      # seconds
        metadata["frame_indices"]    = frame_indices   # original video frame #

        print(f"  Extracted  : {len(raw_frames)} frames  "
              f"(skip={self.config.frame_skip})")

        return raw_frames, metadata


# =============================================================================
# MODULE 2 — FRAME PREPROCESSOR
# =============================================================================

class FramePreprocessor:
    """
    Converts raw BGR frames into normalised 64×64 grayscale arrays.

    Processing chain per frame:
        1. Resize to config.frame_size (default 64×64)
        2. Convert BGR → Grayscale (single channel)
        3. Normalise pixel values to [0, 1]  (float32)
        4. [Optional] Apply 3×3 Gaussian Blur

    Keeping preprocessing a separate class makes it trivial to swap
    resize resolution, add augmentation, or change colour space later.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the full preprocessing chain to a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from OpenCV (H × W × 3).

        Returns
        -------
        np.ndarray
            Processed frame: shape (64, 64), dtype float32, values in [0, 1].
        """
        # 1. Resize — OpenCV expects (width, height) order
        resized = cv2.resize(frame, self.config.frame_size,
                             interpolation=cv2.INTER_AREA)

        # 2. Grayscale — reduces 3 channels to 1, discards colour
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Normalise to [0, 1]
        normalised = gray.astype(np.float32) / 255.0

        # 4. Optional Gaussian blur — smooths noise, helps SSIM consistency
        if self.config.gaussian_blur:
            normalised = cv2.GaussianBlur(normalised, (3, 3), 0)

        return normalised

    # ------------------------------------------------------------------
    def process_all(self, raw_frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies process_frame() to every frame with a tqdm progress bar.

        Parameters
        ----------
        raw_frames : list of np.ndarray
            BGR frames from VideoLoader.

        Returns
        -------
        list of np.ndarray
            Preprocessed (64×64 float32) frames.
        """
        print(f"\n[Module 2] Frame Preprocessor")
        print(f"  Target size   : {self.config.frame_size[0]}×{self.config.frame_size[1]}")
        print(f"  Gaussian blur : {self.config.gaussian_blur}")

        processed = []
        for frame in tqdm(raw_frames, desc="  Preprocessing", unit="fr", ncols=70):
            processed.append(self.process_frame(frame))

        print(f"  Output shape  : {processed[0].shape}  dtype={processed[0].dtype}")
        return processed


# =============================================================================
# MODULE 3 — GUARD LAYER G1 : Pixel Variance
# =============================================================================

class GuardG1Variance:
    """
    Guard Layer G1 — Detects frozen clips via pixel-level variance.

    Algorithm
    ---------
    For each sliding window of T = 16 consecutive preprocessed frames:
        1. Stack frames → tensor of shape (T, 64, 64).
        2. Compute variance along the time axis → (64, 64) map.
        3. Take the mean across all pixels → scalar `mean_var`.
        4. If mean_var < θ_v → clip is FROZEN.

    A frozen clip has near-zero pixel variation, meaning the camera feed
    is stuck on a static or repeated frame.

    Parameters
    ----------
    config : PipelineConfig
        Reads window_size and variance_threshold (θ_v).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    def check_window(self, window: List[np.ndarray]) -> Dict[str, Any]:
        """
        Runs G1 on a single window of frames.

        Parameters
        ----------
        window : list of np.ndarray
            Exactly `window_size` preprocessed (64×64 float32) frames.

        Returns
        -------
        dict
            If frozen:  {"status": "FROZEN",  "reason": "Low Pixel Variance",
                         "variance": float}
            If normal:  {"status": "PASS",    "variance": float}
        """
        stacked  = np.stack(window, axis=0)          # (T, 64, 64)
        var_map  = np.var(stacked, axis=0)            # (64, 64)
        mean_var = float(np.mean(var_map))

        if mean_var < self.config.variance_threshold:
            return {
                "status"   : "FROZEN",
                "reason"   : "Low Pixel Variance",
                "variance" : mean_var,
            }
        return {
            "status"   : "PASS",
            "variance" : mean_var,
        }

    # ------------------------------------------------------------------
    def run(self, preprocessed_frames: List[np.ndarray],
            window_start: int) -> Dict[str, Any]:
        """
        Evaluates G1 for the window beginning at `window_start`.

        Parameters
        ----------
        preprocessed_frames : list of np.ndarray
        window_start : int
            Index of the first frame in the window.

        Returns
        -------
        dict — same as check_window().
        """
        ws     = self.config.window_size
        window = preprocessed_frames[window_start : window_start + ws]
        return self.check_window(window)


# =============================================================================
# MODULE 4 — GUARD LAYER G2 : SSIM
# =============================================================================

class GuardG2SSIM:
    """
    Guard Layer G2 — Detects frozen clips via Structural Similarity (SSIM).

    Algorithm
    ---------
    Given a window of T frames:
        1. Compute SSIM for every consecutive pair: (f0,f1), (f1,f2), …
           → gives T-1 scores.
        2. Count the longest run of consecutive scores that ALL exceed
           ssim_threshold (default 0.999).
        3. If the longest run ≥ ssim_consecutive (default 8) → FROZEN.

    Why G2 complements G1:
    - Variance catches global static scenes but can miss subtle frame-
      repeats where individual pixel values barely differ.
    - SSIM is sensitive to structural/luminance similarity; it reliably
      detects repeated or near-identical frames even with minor noise.

    Parameters
    ----------
    config : PipelineConfig
        Reads ssim_threshold and ssim_consecutive.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    def check_window(self, window: List[np.ndarray]) -> Dict[str, Any]:
        """
        Runs G2 on a single window of frames.

        Parameters
        ----------
        window : list of np.ndarray
            Exactly `window_size` preprocessed (64×64 float32) frames.

        Returns
        -------
        dict
            If frozen:  {"status": "FROZEN", "reason": "High SSIM",
                         "count": int, "ssim_scores": list[float]}
            If normal:  {"status": "PASS",   "max_run": int,
                         "ssim_scores": list[float]}
        """
        scores = []
        for i in range(len(window) - 1):
            score = ssim(window[i], window[i + 1], data_range=1.0)
            scores.append(float(score))

        # Count longest consecutive run above threshold
        max_run    = 0
        current    = 0
        for s in scores:
            if s > self.config.ssim_threshold:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0

        if max_run >= self.config.ssim_consecutive:
            return {
                "status"      : "FROZEN",
                "reason"      : "High SSIM",
                "count"       : max_run,
                "ssim_scores" : scores,
            }
        return {
            "status"      : "PASS",
            "max_run"     : max_run,
            "ssim_scores" : scores,
        }

    # ------------------------------------------------------------------
    def run(self, preprocessed_frames: List[np.ndarray],
            window_start: int) -> Dict[str, Any]:
        """
        Evaluates G2 for the window beginning at `window_start`.

        Parameters
        ----------
        preprocessed_frames : list of np.ndarray
        window_start : int

        Returns
        -------
        dict — same as check_window().
        """
        ws     = self.config.window_size
        window = preprocessed_frames[window_start : window_start + ws]
        return self.check_window(window)


# =============================================================================
# MODULE 5 — SLIDING WINDOW CLIP BUILDER
# =============================================================================

class SlidingWindowBuilder:
    """
    Builds validated 16-frame clips using a sliding window over the
    preprocessed frames.

    For each window position:
        1. Run Guard G1 — if FROZEN → record result, skip this clip.
        2. Run Guard G2 — if FROZEN → record result, skip this clip.
        3. Both guards PASS → include clip in output.

    The clip builder operates on preprocessed grayscale frames only.
    It does NOT compute frame differences, optical flow, or any other
    motion feature.  Those belong to downstream modules.

    Parameters
    ----------
    config : PipelineConfig
    g1     : GuardG1Variance
    g2     : GuardG2SSIM
    """

    def __init__(self, config: PipelineConfig,
                 g1: GuardG1Variance,
                 g2: GuardG2SSIM):
        self.config = config
        self.g1     = g1
        self.g2     = g2

    # ------------------------------------------------------------------
    def build(self,
              preprocessed_frames: List[np.ndarray],
              frame_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Slides a window across all preprocessed frames and returns
        a list of validated clip dictionaries.

        Parameters
        ----------
        preprocessed_frames : list of np.ndarray
            Output of FramePreprocessor — normalised 64×64 float32 arrays.
        frame_indices : list of int
            Original video frame numbers (for tracking provenance).
            Must be same length as preprocessed_frames.

        Returns
        -------
        clips : list of dict
            Each element:
            {
              "clip_id"         : int,
              "frame_indices"   : list[int],   # original video frame numbers
              "status"          : "PASS" | "FROZEN",
              "reason"          : str | None,
              "processed_frames": list[np.ndarray],  # 16 normalised 64×64 arrays
              "g1_result"       : dict,
              "g2_result"       : dict | None,        # None if G1 already froze it
            }

        Notes
        -----
        Only clips with status=="PASS" are intended for downstream use.
        Frozen clips are included in the return list for audit/logging.
        """
        ws          = self.config.window_size
        num_windows = len(preprocessed_frames) - ws + 1
        clips       : List[Dict[str, Any]] = []

        print(f"\n[Module 5] Sliding Window Clip Builder")
        print(f"  Total frames   : {len(preprocessed_frames)}")
        print(f"  Window size    : {ws}")
        print(f"  Total windows  : {num_windows}")

        frozen_g1 = 0
        frozen_g2 = 0
        passed    = 0

        for i in tqdm(range(num_windows), desc="  Building clips",
                      unit="clip", ncols=70):

            window_frames  = preprocessed_frames[i : i + ws]
            window_indices = frame_indices[i : i + ws]

            # ── Guard G1 ──────────────────────────────────────────────
            g1_result = self.g1.check_window(window_frames)

            if g1_result["status"] == "FROZEN":
                frozen_g1 += 1
                clips.append({
                    "clip_id"          : i,
                    "frame_indices"    : window_indices,
                    "status"           : "FROZEN",
                    "reason"           : g1_result["reason"],
                    "processed_frames" : window_frames,
                    "g1_result"        : g1_result,
                    "g2_result"        : None,
                })
                continue   # ← G1 failed: skip G2, skip this clip

            # ── Guard G2 ──────────────────────────────────────────────
            g2_result = self.g2.check_window(window_frames)

            if g2_result["status"] == "FROZEN":
                frozen_g2 += 1
                clips.append({
                    "clip_id"          : i,
                    "frame_indices"    : window_indices,
                    "status"           : "FROZEN",
                    "reason"           : g2_result["reason"],
                    "processed_frames" : window_frames,
                    "g1_result"        : g1_result,
                    "g2_result"        : g2_result,
                })
                continue   # ← G2 failed: skip this clip

            # ── Both guards passed ────────────────────────────────────
            passed += 1
            clips.append({
                "clip_id"          : i,
                "frame_indices"    : window_indices,
                "status"           : "PASS",
                "reason"           : None,
                "processed_frames" : window_frames,
                "g1_result"        : g1_result,
                "g2_result"        : g2_result,
            })

        print(f"\n  ── Guard Summary ──────────────────────────────")
        print(f"  Clips PASS   : {passed}")
        print(f"  Frozen (G1)  : {frozen_g1}")
        print(f"  Frozen (G2)  : {frozen_g2}")
        print(f"  Total clips  : {len(clips)}")

        return clips


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class PreprocessingPipeline:
    """
    Top-level orchestrator that wires all modules together.

    Usage
    -----
    >>> config   = PipelineConfig()
    >>> pipeline = PreprocessingPipeline(config)
    >>> clips    = pipeline.run("path/to/video.mp4")

    The returned `clips` list is the final output of this module.
    Pass ONLY clips where clip["status"] == "PASS" to the next developer.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.loader       = VideoLoader(self.config)
        self.preprocessor = FramePreprocessor(self.config)
        self.g1           = GuardG1Variance(self.config)
        self.g2           = GuardG2SSIM(self.config)
        self.builder      = SlidingWindowBuilder(self.config, self.g1, self.g2)

    # ------------------------------------------------------------------
    def run(self, video_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Runs the full preprocessing pipeline on one video.

        Parameters
        ----------
        video_path : str
            Path to .mp4 / .avi / .mov video file.

        Returns
        -------
        clips : list of dict
            All clips (PASS + FROZEN).  Filter by status=="PASS" for
            downstream use.
        metadata : dict
            Video-level metadata from the loader.
        """
        banner = "=" * 65
        print(f"\n{banner}")
        print("  PREPROCESSING PIPELINE")
        print("  Motion Consistency Evaluation — Guard Layers Module")
        print(f"{banner}")

        # ── Step 1 & 2 : Load + extract frames ────────────────────────
        raw_frames, metadata = self.loader.load(video_path)

        if not raw_frames:
            print("  [WARNING] No frames extracted.  Aborting pipeline.")
            return [], metadata

        # ── Step 3 : Preprocess frames ─────────────────────────────────
        processed_frames = self.preprocessor.process_all(raw_frames)

        # ── Step 4 & 5 & 6 : Guard layers + clip builder ──────────────
        clips = self.builder.build(processed_frames,
                                   metadata["frame_indices"])

        # ── Summary ───────────────────────────────────────────────────
        pass_clips   = [c for c in clips if c["status"] == "PASS"]
        frozen_clips = [c for c in clips if c["status"] == "FROZEN"]

        print(f"\n{banner}")
        print("  PIPELINE COMPLETE")
        print(f"  Video          : {metadata['video_name']}")
        print(f"  Total clips    : {len(clips)}")
        print(f"  Valid (PASS)   : {len(pass_clips)}")
        print(f"  Frozen         : {len(frozen_clips)}")
        print(f"  Ready for next module: {len(pass_clips)} clips")
        print(f"{banner}\n")

        # ── Optional: debug visualisation ─────────────────────────────
        if self.config.debug_mode:
            self._save_debug_viz(processed_frames, clips, metadata)

        return clips, metadata

    # ------------------------------------------------------------------
    def _save_debug_viz(self, processed_frames, clips, metadata):
        """
        Debug-only: saves a grid of sample preprocessed frames and a
        G1 variance plot to disk.  Only runs when config.debug_mode=True.
        """
        out_dir = Path(self.config.debug_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name    = metadata.get("video_name", "video")

        # ── Sample frame grid ─────────────────────────────────────────
        num_show = min(16, len(processed_frames))
        fig, axes = plt.subplots(2, 8, figsize=(20, 5))
        fig.suptitle(f"Preprocessed Frames — {name}", fontsize=12)
        for idx in range(num_show):
            r, c = divmod(idx, 8)
            axes[r, c].imshow(processed_frames[idx], cmap="gray",
                               vmin=0, vmax=1)
            axes[r, c].set_title(f"F{idx}", fontsize=7)
            axes[r, c].axis("off")
        for idx in range(num_show, 16):
            r, c = divmod(idx, 8)
            axes[r, c].axis("off")
        plt.tight_layout()
        grid_path = out_dir / f"{name}_debug_frames.png"
        plt.savefig(grid_path, dpi=120)
        plt.close()
        print(f"  [Debug] Frame grid saved → {grid_path}")

        # ── G1 variance per window ────────────────────────────────────
        variances = [c["g1_result"]["variance"] for c in clips]
        statuses  = [c["status"] for c in clips]
        colors    = ["#e05c5c" if s == "FROZEN" else "#4caf87"
                     for s in statuses]

        plt.figure(figsize=(14, 4))
        plt.bar(range(len(variances)), variances, color=colors, width=1.0)
        plt.axhline(self.config.variance_threshold, color="yellow",
                    linestyle="--", label=f"θ_v = {self.config.variance_threshold}")
        plt.title(f"G1 Pixel Variance per Window — {name}")
        plt.xlabel("Window index")
        plt.ylabel("Mean variance")
        plt.legend()
        plt.tight_layout()
        var_path = out_dir / f"{name}_debug_g1_variance.png"
        plt.savefig(var_path, dpi=120)
        plt.close()
        print(f"  [Debug] G1 variance plot saved → {var_path}")


# =============================================================================
# CONVENIENCE HELPERS
# =============================================================================

def get_valid_clips(clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter helper — returns only clips that passed both guard layers.

    Parameters
    ----------
    clips : list of dict
        Full output from PreprocessingPipeline.run().

    Returns
    -------
    list of dict
        Clips ready to be handed to the next developer (Stream A / B).
    """
    return [c for c in clips if c["status"] == "PASS"]


def clips_to_numpy(clips: List[Dict[str, Any]]) -> np.ndarray:
    """
    Stacks the processed_frames of all PASS clips into a single array.

    Parameters
    ----------
    clips : list of dict
        Already filtered to PASS clips (use get_valid_clips first).

    Returns
    -------
    np.ndarray
        Shape (N, 16, 64, 64), dtype float32.
        Each slice [i] is one 16-frame clip of normalised grayscale frames.
    """
    arrays = [np.stack(c["processed_frames"], axis=0) for c in clips]
    return np.array(arrays, dtype=np.float32)


# =============================================================================
# ENTRY POINT — quick demo run
# =============================================================================

if __name__ == "__main__":
    # ── Configure ─────────────────────────────────────────────────────
    config = PipelineConfig()
    config.max_frames   = 200          # limit for quick testing
    config.gaussian_blur = False
    config.frame_skip   = 1
    config.debug_mode   = True         # saves diagnostic images

    # ── Set video path ────────────────────────────────────────────────
    VIDEO = r"D:\AICity21-Track4-Anomaly-Detection\AIC21-Track4-Anomaly-Detection\aic21-track4-train-data\1.mp4"

    # ── Run pipeline ──────────────────────────────────────────────────
    pipeline = PreprocessingPipeline(config)
    clips, meta = pipeline.run(VIDEO)

    # ── Filter valid clips for next module ────────────────────────────
    valid = get_valid_clips(clips)
    print(f"Clips ready for downstream module : {len(valid)}")

    # ── Optional: convert to numpy array ─────────────────────────────
    if valid:
        arr = clips_to_numpy(valid)
        print(f"NumPy array shape : {arr.shape}")   # (N, 16, 64, 64)

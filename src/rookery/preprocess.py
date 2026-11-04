"""Pre-processing chain: to_mono → resample → remove_dc → bandpass → [noisereduce]
→ segment → trim_to_middle.

The `preprocess` CLI command and `pipeline.analyze` both call `preprocess()`,
so the two surfaces cannot diverge.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import PipelineConfig

# --------------------------------------------------------------------------
# Signal preparation
# --------------------------------------------------------------------------


def to_mono(y: np.ndarray) -> np.ndarray:
    """Collapse a multi-channel track to mono by averaging over channels."""
    if y.ndim == 1:
        return y
    return y.mean(axis=1)


def resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample to `target_sr` (no-op when already at the target rate)."""
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)


def remove_dc(y: np.ndarray) -> np.ndarray:
    """Subtract the mean (the only amplitude "normalization" in the pipeline)."""
    return y - y.mean()


def bandpass(y: np.ndarray, sr: int, cfg: PipelineConfig) -> np.ndarray:
    """Zero-phase 4th-order Butterworth bandpass [hp_hz, lp_hz]."""
    nyq = sr / 2.0
    if cfg.lp_hz >= nyq:
        raise ValueError(
            f"lp_hz={cfg.lp_hz} must be below Nyquist ({nyq} Hz at sr={sr})"
        )
    sos = butter(4, [cfg.hp_hz, cfg.lp_hz], btype="band", output="sos", fs=sr)
    return sosfiltfilt(sos, y)


def _noise_reduce(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply noisereduce (lazy import: optional `noise` extra)."""
    try:
        import noisereduce as nr
    except ImportError:
        raise SystemExit(
            "noisereduce is not installed — install the `noise` extra: "
            "pip install -e '.[noise]'"
        )
    return nr.reduce_noise(y=y, sr=sr)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


def segment(y: np.ndarray, sr: int, cfg: PipelineConfig) -> list[tuple[float, float]]:
    """Find voiced (energy) segments as (t0, t1) in seconds; [] if all silence."""
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    thr = cfg.rms_threshold_frac * rms.max()
    above = rms > thr
    if not above.any():
        return []

    # Runs of consecutive above-threshold frames, from mask transitions.
    change = np.flatnonzero(np.diff(above.astype(np.int8))) + 1
    starts = list(change[above[change]])  # 0→1 transitions
    if above[0]:
        starts = [0] + starts
    ends = list(change[~above[change]])  # 1→0 transitions
    if above[-1]:
        ends = ends + [len(above)]

    segs: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        # t0 is the first above-threshold frame's center; t1 extends one full
        # hop past the last frame's center so the run covers its trailing edge
        t0, t1 = float(times[s]), float(times[e - 1]) + 512 / sr
        # merge runs separated by less than min_gap_s
        if segs and t0 - segs[-1][1] < cfg.min_gap_s:
            segs[-1] = (segs[-1][0], t1)
        else:
            segs.append((t0, t1))
    # drop runs shorter than the voiced floor
    return [(t0, t1) for t0, t1 in segs if t1 - t0 >= cfg.min_segment_s]


def trim_to_middle(
    segments: list[tuple[float, float]], cfg: PipelineConfig
) -> list[tuple[float, float]]:
    """Drop first/last `trim_frac` of each segment (capped at `max_trim_s`)."""
    out: list[tuple[float, float]] = []
    for t0, t1 in segments:
        drop = min(cfg.max_trim_s, cfg.trim_frac * (t1 - t0))
        if t1 - t0 - 2 * drop < cfg.min_segment_s:
            # trimming would breach the voiced floor — keep the whole segment
            out.append((t0, t1))
        else:
            out.append((t0 + drop, t1 - drop))
    return out


def concat_segments(
    y: np.ndarray, sr: int, segments: list[tuple[float, float]]
) -> np.ndarray:
    """Concatenate the segments of `y` (empty input → empty float64 array)."""
    if not segments:
        return np.zeros(0, dtype=np.float64)
    parts = [y[int(t0 * sr) : int(t1 * sr)] for t0, t1 in segments]
    return np.concatenate(parts)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def preprocess(
    y: np.ndarray, sr: int, cfg: PipelineConfig
) -> tuple[np.ndarray, int, list[tuple[float, float]]]:
    """Run the full pre-processing chain; returns (audio, sr, trimmed segments)."""
    y = to_mono(y)
    y = resample(y, sr, cfg.target_sr)
    sr = cfg.target_sr
    y = remove_dc(y)
    y = bandpass(y, sr, cfg)
    if cfg.noise_reduce:
        y = _noise_reduce(y, sr)
    segments = trim_to_middle(segment(y, sr, cfg), cfg)
    return y, sr, segments

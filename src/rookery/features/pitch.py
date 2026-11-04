from __future__ import annotations

import librosa
import numpy as np

from ..models import PitchStats
from ..utils import semitone_change


def extract_pitch(
    y: np.ndarray,
    sr: int,
    segments: list[tuple[float, float]],
    thresholds_hz=(180.0, 200.0),
) -> PitchStats:
    """F0 stats over voiced frames of all speech segments (pYIN)."""
    f0_chunks: list[np.ndarray] = []
    total_frames = 0
    voiced_frames = 0

    for t0, t1 in segments:
        seg = y[int(t0 * sr) : int(t1 * sr)]
        if len(seg) < int(0.05 * sr):
            continue
        # Same pYIN algorithm as parselmouth's to_pitch, with matching defaults:
        # Praat default F0 range 75-600 Hz, ~46 ms window (2048 @ 44.1 kHz),
        # 10 ms frame step. Unvoiced frames come back as NaN.
        f0, _, _ = librosa.pyin(
            seg,
            fmin=75.0,
            fmax=600.0,
            sr=sr,
            frame_length=2048,
            hop_length=int(0.01 * sr),
        )
        voiced = ~np.isnan(f0)
        total_frames += len(f0)
        voiced_frames += int(voiced.sum())
        if voiced.any():
            f0_chunks.append(f0[voiced])

    stats = PitchStats()
    if not f0_chunks:
        return stats

    f0 = np.concatenate(f0_chunks)
    mean = float(np.mean(f0))
    stdev = float(np.std(f0, ddof=1)) if len(f0) > 1 else 0.0

    p10 = float(np.percentile(f0, 10))
    p25 = float(np.percentile(f0, 25))
    p75 = float(np.percentile(f0, 75))
    p90 = float(np.percentile(f0, 90))

    stats.median_hz = float(np.median(f0))
    stats.p10_hz = p10
    stats.p90_hz = p90
    stats.p25_hz = p25
    stats.p75_hz = p75
    stats.overall_range_semitones = semitone_change(p10, p90)
    stats.speaking_range_semitones = semitone_change(p25, p75)

    # internal — kept for calibration/debugging, not shown to end users
    stats.mean_hz = mean
    stats.stdev_hz = stdev
    stats.cv = stdev / mean if mean > 0 else None
    stats.voiced_ratio = voiced_frames / total_frames if total_frames else None

    for thr in thresholds_hz:
        stats.percent_above[str(int(thr))] = float(100.0 * np.mean(f0 > thr))
    return stats

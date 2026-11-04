from __future__ import annotations

import logging

import librosa
import numpy as np

from ..models import FormantStats

logger = logging.getLogger(__name__)

# --- LPC formant parameters -------------------------------------------------
# 40 ms analysis window (≈1800 samples @ 44.1 kHz) — long enough for stable
# autocorrelation, short enough to track vowel movement at a 10 ms frame step.
_WINDOW_S = 0.040
_LPC_ORDER = 24  # ~6 pole pairs → room for F1–F3 plus the low-frequency ones
# Median-filter radius in frames before stats: per-frame LPC poles can swap or
# drop out between adjacent windows (especially where resonances are close),
# so raw frame-to-frame jumps are tracking noise, not formant movement. A 5-
# frame median kills isolated outliers while preserving real vowel transitions,
# which unfold over hundreds of ms.
_MEDIAN_RADIUS = 2


def extract_formants(
    y: np.ndarray, sr: int, segments: list[tuple[float, float]]
) -> FormantStats:
    """F1–F3 medians/variability over voiced frames + vowel-space proxy."""
    hits = {k: 0 for k in range(1, 4)}
    total_frames = 0
    # Per-segment raw tracks; smoothed across segments before stats so that
    # formant continuity survives segment boundaries.
    raw_tracks: dict[int, list[list[float | None]]] = {k: [] for k in range(1, 4)}

    for t0, t1 in segments:
        seg = y[int(t0 * sr) : int(t1 * sr)]
        if len(seg) < int(0.05 * sr):
            continue
        # Voicing mask from the pyin F0 track (same pYIN params as pitch.py).
        f0, _, _ = librosa.pyin(
            seg,
            fmin=75.0,
            fmax=600.0,
            sr=sr,
            frame_length=2048,
            hop_length=int(0.01 * sr),
        )
        voiced = ~np.isnan(f0)
        if not voiced.any():
            continue
        n_frames = len(f0)
        hop = int(0.01 * sr)

        # Frame times: pyin centers frame i at (i + 0.5) * hop samples.
        frame_times = (np.arange(n_frames) + 0.5) * hop / sr
        half_win = int(_WINDOW_S * sr) // 2

        for k in range(1, 4):
            raw_tracks[k].append([None] * n_frames)
        for i in range(n_frames):
            if not voiced[i]:
                continue
            total_frames += 1
            t = frame_times[i]
            c = round(t * sr)
            x = seg[max(0, c - half_win) : min(len(seg), c + half_win)]
            # Skip windows too short for a stable autocorrelation (segment edges).
            if len(x) < 2 * _LPC_ORDER:
                continue
            fr = _lpc_formants(x, sr, order=_LPC_ORDER)
            for k in range(1, 4):
                fk = fr.get(k)
                if fk is not None:
                    hits[k] += 1
                    raw_tracks[k][-1][i] = fk

    # Smooth each track (median over ±_MEDIAN_RADIUS frames). Stats are taken
    # only at frames where the formant was raw-detected, so smoothing never
    # invents values for unvoiced/edge frames; coverage still counts raw hits.
    stats = FormantStats()
    flat_masks: dict[int, np.ndarray] = {}
    for k in range(1, 4):
        flat_raw: list[float | None] = [
            v for seg_track in raw_tracks[k] for v in seg_track
        ]
        mask = np.asarray([v is not None for v in flat_raw]).astype(bool)
        smooth = _median_smooth(flat_raw, radius=_MEDIAN_RADIUS)
        vals = smooth[mask]
        setattr(stats, f"f{k}_median_hz", float(np.median(vals)) if len(vals) else None)
        setattr(
            stats,
            f"f{k}_std_hz",
            float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        )
        setattr(
            stats, f"coverage_f{k}", hits[k] / total_frames if total_frames else None
        )
        flat_masks[k] = mask
    # Vowel-space pairs: frames where both F1 and F2 were raw-detected.
    m12 = flat_masks[1] & flat_masks[2]
    if m12.any():
        s1 = _median_smooth(
            [v for seg_track in raw_tracks[1] for v in seg_track], radius=_MEDIAN_RADIUS
        )
        s2 = _median_smooth(
            [v for seg_track in raw_tracks[2] for v in seg_track], radius=_MEDIAN_RADIUS
        )
        pts = np.column_stack((s1[m12], s2[m12]))
    else:
        pts = np.zeros((0, 2))
    stats.vowel_space_area_proxy_hz2 = _vowel_space_area(pts)
    return stats


# --------------------------------------------------------------------------
# LPC formants: Yule–Walker (Toeplitz solve) → pole-pair resonances.
# Replaces parselmouth's Burg method (Option B, PLAN.md decision 3).
# --------------------------------------------------------------------------


def _lpc_formants(x: np.ndarray, sr: int, order: int = 24) -> dict[int, float]:
    """Return {formant_index: freq_hz} for the up-to-`order//2` pole pairs.

    Solves Yule–Walker on the window's autocorrelation, then maps each
    conjugate pole pair to a resonance frequency. Formant index k is assigned
    by ascending frequency (index 1 = lowest), matching Praat's F1..F3 ordering.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2 * order:
        return {}

    # Taper the window before LPC: a Hanning taper suppresses spectral leakage
    # from the window edges so the pole pairs track true resonances.
    x = x * np.hanning(n)

    # Autocorrelation r[k] for k=0..order (linear, biased is fine here).
    r = _autocorr(x, order)
    if not np.isfinite(r[0]) or r[0] <= 0:
        return {}

    a = _lpc_coefficients(r, order)
    if a is None:
        return {}

    # Poles are roots of A(z) = 1 - a[1] z^-1 - ... - a[p] z^-p.
    # np.roots wants descending powers: multiply A by z^p → [1, -a[1], ..., -a[p]].
    coeffs = np.concatenate(([1.0], -a))
    roots = np.roots(coeffs)

    freqs: list[float] = []
    for z in roots:
        # Real-coefficient polynomial → non-real roots are conjugate pairs;
        # take one per pair (upper half-plane) and skip real poles.
        if abs(z.imag) < 1e-6 * max(1.0, abs(z)) or z.imag < 0:
            continue
        if abs(z) >= 1.0:  # outside the unit circle — not a stable resonance
            continue
        freq = abs(np.angle(z)) * sr / (2 * np.pi)
        if 50 < freq < sr / 2:
            freqs.append(freq)

    freqs.sort()
    return {k + 1: f for k, f in enumerate(freqs[:3])}


def _median_smooth(values: list[float | None], radius: int) -> np.ndarray:
    """Centered median over ±radius frames; NaN where the window is empty.

    Input is a flat track (None = unvoiced/undetected frame); output has one
    entry per input. The centered window means each smoothed value only uses
    neighbors from the same segment run, since unvoiced gaps are None.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        win = arr[lo:hi]
        if np.isfinite(win).any():
            out[i] = float(np.nanmedian(win))
    return out


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Linear autocorrelation r[k] = sum_n x[n+k]*x[n], k=0..max_lag."""
    n = len(x)
    # FFT-based for speed; the direct definition is what we want.
    f = np.fft.rfft(x, n=n + max_lag)
    acf = np.fft.irfft(f * np.conj(f), n=n + max_lag)[: max_lag + 1]
    return acf


def _lpc_coefficients(r: np.ndarray, order: int) -> np.ndarray | None:
    """Yule–Walker coefficients a[1..order] via direct Toeplitz solve.

    Solves T·a = r[1..order] with T[i,j] = r[|i-j|]; the solution is the
    predictor of A(z) = 1 - Σ a_i z^-i (x[n] ≈ Σ a_i·x[n-i]). A direct solve
    sidesteps Levinson–Durbin recursion edge cases and is trivially cheap at
    order 24. None if the system is singular or non-finite.
    """
    idx = np.arange(order)
    T = r[np.abs(idx[:, None] - idx[None, :])]
    try:
        a = np.linalg.solve(T, r[1 : order + 1])
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a


# --------------------------------------------------------------------------
# Vowel-space proxy: cluster (F1,F2) points into ~5 vowel-like groups and take
# the convex-hull area of the centroids. A true vowel space needs phoneme
# labels (faster-whisper roadmap item).
# --------------------------------------------------------------------------


def _vowel_space_area(points: np.ndarray) -> float | None:
    if len(points) < 10:
        return None
    cents = _kmeans_centroids(points, k=5)
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(cents).volume)  # in 2-D, .volume is area
    except Exception:
        logger.debug("vowel-space convex hull failed", exc_info=True)
        return None


def _kmeans_centroids(
    points: np.ndarray, k: int = 5, iters: int = 40, seed: int = 7
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(points)
    if n <= k:
        return points.astype(float).copy()
    C = points[rng.choice(n, size=k, replace=False)].astype(float).copy()
    for _ in range(iters):
        d2 = ((points[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab = d2.argmin(axis=1)
        newC = C.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                newC[j] = points[m].mean(axis=0)
        if np.allclose(newC, C, atol=1e-4):
            break
        C = newC
    return C

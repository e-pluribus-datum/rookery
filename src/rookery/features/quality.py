from __future__ import annotations

import logging
from itertools import pairwise

import librosa
import numpy as np

from ..models import VoiceQualityStats
from ..preprocess import concat_segments

logger = logging.getLogger(__name__)

# --- pYIN parameters --------------------------------------------------------
# Identical to pitch.py / formants.py so the voicing mask agrees with A and B:
# Praat default F0 range 75–600 Hz, ~46 ms window (2048 @ 44.1 kHz), 10 ms step.
_FMIN = 75.0
_FMAX = 600.0
_FRAME_LENGTH = 2048


# --- Voiced-run floors ------------------------------------------------------
# Jitter/shimmer/H1–Hk need ≥8 consecutive voiced frames (~10 ms step ≈ the
# "10–20 periods" floor). HNR is exempt: any voiced run with ≥4 harmonics of
# spectral support contributes.
_MIN_RUN_FRAMES = 8


# --- STFT parameters --------------------------------------------------------
# One power spectrum per 10 ms frame, aligned to the pyin frame grid (frame i is
# centered at (i + 0.5) * hop samples). A wide window keeps adjacent harmonics
# resolvable; the power is summed over each harmonic's bin neighborhood so a few
# Hz of F0 error doesn't drop the peak.
_NFFT = 2048


def _pyin_f0(seg: np.ndarray, sr: int) -> tuple[np.ndarray | None, int]:
    """Run pYIN on one segment; return (f0 track or None if empty, hop in samples)."""
    hop = int(0.01 * sr)
    f0, _, _ = librosa.pyin(
        seg,
        fmin=_FMIN,
        fmax=_FMAX,
        sr=sr,
        frame_length=_FRAME_LENGTH,
        hop_length=hop,
    )
    return f0, hop


def _voiced_runs(voiced: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive voiced frames as (start, end) half-open ranges."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(voiced)))
    return runs


def _stft_power(
    seg: np.ndarray, sr: int, hop: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """Per-frame STFT power spectrum in dB, shape (n_frames, n_freqs), + bin frequencies.

    Frame-major so rows index the pyin frame directly; librosa.stft returns
    (n_freqs, n_frames), hence the transpose.
    """
    n = len(seg)
    if n < 2 * _NFFT or hop <= 0:
        return None, np.array([])
    S = librosa.stft(
        seg,
        n_fft=_NFFT,
        hop_length=hop,
        win_length=_NFFT,
        window="hann",
        center=True,
    )
    power_db = (10.0 * np.log10(np.abs(S) ** 2 + 1e-9)).T
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_NFFT)
    return power_db, freqs


def _harmonic_peak(
    power_db: np.ndarray, freqs: np.ndarray, f_hz: float
) -> float | None:
    """Peak dB within a ±25% band around a harmonic frequency (None if no bins)."""
    lo = 0.75 * f_hz
    hi = 1.25 * f_hz
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return None
    return float(np.max(power_db[m]))


def _harmonic_support(f0: np.ndarray) -> bool:
    """Whether ≥4 harmonics of the run's median F0 sit below fmax (spectral support).

    The HNR harmonic comb only extends to fmax, so a run needs at least four
    resolvable harmonics in that band for the split to be meaningful.
    """
    f = float(np.median(f0))
    if f <= 0:
        return False
    k = int(_FMAX // f)  # number of harmonics with k*f < fmax
    return k >= 4


def _jitter_shimmer_run(
    seg: np.ndarray, sr: int, f0: np.ndarray, run: tuple[int, int]
) -> tuple[float | None, float | None, float | None]:
    """Jitter (rel % / abs ms) and shimmer (local dB) over one voiced run."""
    a, b = run
    T = 1.0 / f0[a:b]  # period per frame, in seconds
    n = len(T)

    # Jitter: cycle-to-cycle variation of the F0 period. Relative is speaker-
    # normalized (headline); absolute ms is the raw secondary.
    jit_rel = jit_abs = shim_db = None
    mean_T = float(np.mean(T)) if n else 0.0
    if n >= 2:
        dT = np.abs(np.diff(T))
        jit_rel = float(100.0 * np.median(dT) / (mean_T + 1e-9))
        jit_abs = float(np.median(dT) * 1000.0)

    # Shimmer: cycle-to-cycle amplitude variation, local dB. Cycle boundaries are
    # the cumulative period sums from the run start; each boundary pair is one
    # glottal cycle's RMS. Needs ≥2 cycles to differ two amplitudes.
    if n >= 2 and mean_T > 0:
        centers = (np.arange(a, b) + 0.5) * int(0.01 * sr) / sr  # frame times
        t_start = float(centers[0]) - T[0] / 2.0
        boundaries = t_start + np.concatenate([[0.0], np.cumsum(T)])
        rms: list[float] = []
        for s, e in pairwise(boundaries):
            i0, i1 = round(float(s) * sr), round(float(e) * sr)
            if i1 - i0 >= 2 and i0 >= 0 and i1 <= len(seg):
                rms.append(float(np.sqrt(np.mean(seg[i0:i1] ** 2))))
        if len(rms) >= 2:
            d = np.abs(np.diff(np.asarray(rms)))
            m = float(np.median(rms)) + 1e-9
            shim_db = float(20.0 * np.log10(np.median(d) / m))

    return jit_rel, jit_abs, shim_db


def _hnr_harmonic_run(
    f0: np.ndarray,
    run: tuple[int, int],
    power_db: np.ndarray | None,
    freqs: np.ndarray,
) -> tuple[float | None, list[float]]:
    """HNR (dB) and H1/Hk ratios for one voiced run from the STFT.

    The harmonic comb extends to fmax only — energy above it is treated as noise,
    matching Praat's 75–600 Hz F0 range. Per-frame masks use that frame's own F0
    so a drifting pitch still lands its harmonics in the comb; H1/Hk are computed
    from the run-median spectrum at the median-F0 comb.
    """
    if power_db is None or not len(freqs):
        return None, []
    a, b = run
    f = float(np.median(f0[a:b]))

    # HNR: harmonic bins vs everything else (noise), per frame. The caller has
    # already checked ≥4 harmonics of spectral support at the median F0.
    hnr_db = None
    harm_pows, noise_pows = [], []
    for i in range(a, b):
        fi = float(f0[i])
        mask = np.zeros(len(freqs), dtype=bool)
        for k in range(1, int(_FMAX // fi) + 1):
            lo, hi = 0.75 * k * fi, 1.25 * k * fi
            mask |= (freqs >= lo) & (freqs <= hi)
        if mask.sum() < 4 or (~mask).sum() < 4:  # need real harmonic + noise support
            continue
        harm_pows.append(power_db[i, mask].max())
        noise_pows.append(power_db[i, ~mask].max())
    if harm_pows:
        hnr_db = float(np.median(harm_pows) - np.median(noise_pows))

    # H1/Hk: peak dB at the run-median comb. Ratios need H1–H4 present.
    peaks: list[float] = []
    ok = True
    for k in range(1, 5):
        p = _harmonic_peak(power_db[a:b].mean(axis=0), freqs, k * f)
        if p is None:
            ok = False
            break
        peaks.append(p)

    h1_hk = []
    if ok and len(peaks) == 4:
        h1 = max(peaks[0], -300.0)  # floor so a quiet H1 doesn't blow the ratio
        for k in (1, 2, 3):
            h1_hk.append(h1 - peaks[k])

    return hnr_db, h1_hk


def extract_quality(
    y: np.ndarray, sr: int, segments: list[tuple[float, float]]
) -> VoiceQualityStats:
    """HNR / jitter / shimmer per speech segment (median across), plus loudness."""
    stats = VoiceQualityStats()
    hnr, jit_rel, jit_abs, shim = [], [], [], []
    h12, h13, h14 = [], [], []

    for t0, t1 in segments:
        seg = y[int(t0 * sr) : int(t1 * sr)]
        if len(seg) < int(0.3 * sr):  # too short for stable perturbation measures
            continue
        f0, hop = _pyin_f0(seg, sr)
        if f0 is None or len(f0) == 0:
            continue
        voiced = ~np.isnan(f0)

        power_db, freqs = _stft_power(seg, sr, hop)

        for run in _voiced_runs(voiced):
            a, b = run
            f0_run = f0[a:b]
            if not np.all(np.isfinite(f0_run)) or float(np.median(f0_run)) <= 0:
                continue

            # Jitter/shimmer/H1–Hk need ≥8 consecutive voiced frames; HNR runs on
            # any voiced run with ≥4 harmonics of spectral support.
            long_enough = b - a >= _MIN_RUN_FRAMES
            supported = _harmonic_support(f0_run)

            if not (long_enough or supported):
                continue
            try:
                if long_enough:
                    jr, ja, sh = _jitter_shimmer_run(seg, sr, f0, run)
                    if jr is not None:
                        jit_rel.append(jr)
                    if ja is not None:
                        jit_abs.append(ja)
                    if sh is not None:
                        shim.append(sh)

                hnr_db = ratios = None
                if supported:
                    hnr_db, ratios = _hnr_harmonic_run(f0, run, power_db, freqs)
                if hnr_db is not None:
                    hnr.append(hnr_db)
                if long_enough and ratios is not None and len(ratios) == 3:
                    h12.append(ratios[0])
                    h13.append(ratios[1])
                    h14.append(ratios[2])
            except Exception:
                logger.debug(
                    "quality metrics failed for segment %.2f-%.2f s run %d:%d",
                    t0,
                    t1,
                    a,
                    b,
                    exc_info=True,
                )

    if hnr:
        stats.hnr_db = float(np.median(hnr))
    if jit_rel:
        stats.jitter_relative_pct = float(np.median(jit_rel))
    if jit_abs:
        stats.jitter_absolute_ms = float(np.median(jit_abs))
    if shim:
        stats.shimmer_local_db = float(np.median(shim))
    if h12:
        stats.h1_h2_db, stats.h1_h3_db, stats.h1_h4_db = (
            float(np.median(h12)),
            float(np.median(h13)),
            float(np.median(h14)),
        )

    # Vocal-effort proxy.
    speech = concat_segments(y, sr, segments)
    if len(speech):
        rms = np.sqrt(np.mean(speech**2))
        stats.rms_db = float(20 * np.log10(rms + 1e-9))
    try:
        from pyloudnorm import Meter

        stats.lufs = float(Meter(sr).integrated_loudness(y))
    except Exception:
        logger.debug("LUFS measurement failed", exc_info=True)
    return stats

from __future__ import annotations

import librosa
import numpy as np

from ..models import SpectralStats
from ..preprocess import concat_segments


def extract_spectral(
    y: np.ndarray, sr: int, segments: list[tuple[float, float]]
) -> SpectralStats:
    """Spectral brightness/darkness over concatenated speech segments."""
    stats = SpectralStats()
    speech = concat_segments(y, sr, segments)
    if len(speech) < 2048:
        return stats

    n_fft, hop = 2048, 512
    S = np.abs(librosa.stft(speech, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    stats.centroid_hz = float(
        np.median(librosa.feature.spectral_centroid(y=speech, sr=sr, hop_length=hop)[0])
    )
    stats.rolloff_85_hz = float(
        np.median(
            librosa.feature.spectral_rolloff(
                y=speech, sr=sr, hop_length=hop, roll_percent=0.85
            )[0]
        )
    )
    stats.bandwidth_hz = float(
        np.median(
            librosa.feature.spectral_bandwidth(y=speech, sr=sr, hop_length=hop)[0]
        )
    )

    total_power = S.sum() + 1e-9
    hi = (freqs >= 2000) & (freqs <= 6000)
    stats.hf_energy_ratio = float(S[hi].sum() / total_power)

    # Spectral tilt: median slope of the power spectrum (dB) vs log2(frequency),
    # i.e. dB per octave, fitted over 100 Hz – 8 kHz. More negative = darker.
    valid = (freqs >= 100) & (freqs <= min(8000.0, sr / 2 - 50))
    if valid.sum() >= 4:
        x = np.log2(freqs[valid])
        mag_db = 10 * np.log10(S + 1e-9)[valid, :]
        slopes = np.linalg.lstsq(x[:, None], mag_db, rcond=None)[0][0]
        stats.tilt_db_per_octave = float(np.median(slopes))
    return stats

"""Synthetic test signal for offline testing (no microphone needed)."""

from __future__ import annotations

import numpy as np


def make_test_signal(
    duration_s: float = 3.0,
    sr: int = 44100,
    f0_hz: float | None = None,
) -> tuple[np.ndarray, int]:
    """Harmonic stack at `f0_hz` (default 215 Hz, A4-ish), ~0.5 peak, float32 mono."""
    t = np.arange(int(duration_s * sr)) / sr
    f0 = f0_hz if f0_hz is not None else 215.0
    y = (
        np.sin(2 * np.pi * f0 * t)
        + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    y = 0.5 * y / np.abs(y).max()
    return y.astype(np.float32), sr

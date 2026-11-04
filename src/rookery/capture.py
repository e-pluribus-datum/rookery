"""Audio I/O: file loading and microphone capture."""

from __future__ import annotations

import numpy as np
import sounddevice as sd
import soundfile as sf


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as float32 mono at its native sample rate."""
    data, sr = sf.read(path, always_2d=True)
    y = data.mean(axis=1).astype(np.float32)
    return y, int(sr)


def record(duration: float, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Record `duration` seconds of mono audio from the default input device."""
    y = sd.rec(duration, samplerate=sr, channels=1)
    return y[:, 0].astype(np.float32), sr

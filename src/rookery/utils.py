"""Shared helpers for feature extraction and scoring."""

from __future__ import annotations

import math


def semitone_change(low_hz: float, high_hz: float) -> float | None:
    """Return the semitone interval between two frequencies, or None if invalid."""
    if low_hz <= 0 or high_hz <= 0:
        return None
    return 12.0 * math.log2(high_hz / low_hz)

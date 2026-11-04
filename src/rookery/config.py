"""Pipeline configuration for the pre-processing chain."""

from __future__ import annotations

from pydantic import BaseModel

# Bumped when the JSON envelope shape changes (see __main__.py extract).
SCHEMA_VERSION = 1


class PipelineConfig(BaseModel):
    """All tunables for the pre-processing chain (defaults = PROJECT.md values)."""

    target_sr: int = 44100
    hp_hz: float = 60.0
    lp_hz: float = 8000.0
    noise_reduce: bool = False
    rms_threshold_frac: float = 0.10  # segment threshold = 10% of max RMS
    min_segment_s: float = 0.10  # voiced floor: jitter/shimmer need ~10–20 periods
    min_gap_s: float = 0.05  # merge runs separated by less than this
    trim_frac: float = 0.15  # trim-to-middle fraction per end (10–20% band)
    max_trim_s: float = 0.05  # cap each end at ~50 ms

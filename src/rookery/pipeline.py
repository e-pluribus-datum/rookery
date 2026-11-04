"""Offline feature pipeline: pre-process + A–D extraction, no I/O."""

from __future__ import annotations

import uuid

import numpy as np

from .config import PipelineConfig
from .features.formants import extract_formants
from .features.pitch import extract_pitch
from .features.quality import extract_quality
from .features.spectral import extract_spectral
from .models import SampleFeatures
from .preprocess import preprocess


def analyze(
    y: np.ndarray, sr: int, config: PipelineConfig | None = None
) -> SampleFeatures:
    """Run the full pipeline (pre-process + A–D extraction) on in-memory audio."""
    cfg = config or PipelineConfig()
    y, sr, segments = preprocess(y, sr, cfg)

    feats = SampleFeatures(
        sample_id=uuid.uuid4().hex[:8],
        duration_s=len(y) / sr,
        sr=sr,
        n_segments=len(segments),
    )
    # Extractors take the trimmed segments: features land on sustained vowels.
    feats.pitch = extract_pitch(y, sr, segments)
    feats.formants = extract_formants(y, sr, segments)
    feats.spectral = extract_spectral(y, sr, segments)
    feats.quality = extract_quality(y, sr, segments)
    return feats

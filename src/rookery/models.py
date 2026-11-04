from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class PitchStats(BaseModel):
    """A. Pitch (F0)."""

    # --- end-user metrics ---
    median_hz: float | None = None
    p10_hz: float | None = None
    p25_hz: float | None = None  # speaking-range lower endpoint
    p75_hz: float | None = None  # speaking-range upper endpoint
    p90_hz: float | None = None
    overall_range_semitones: float | None = None  # P10–P90; robust total range
    speaking_range_semitones: float | None = None  # P25–P75; "speaking pitch range"

    # --- internal (not shown to end users) ---
    mean_hz: float | None = None
    stdev_hz: float | None = None  # variability
    cv: float | None = None  # coefficient of variation (stdev/mean)
    voiced_ratio: float | None = None  # fraction of frames with a voiced F0
    percent_above: dict[str, float] = Field(default_factory=dict)
    # keys are thresholds in Hz as strings, e.g. {"180": 64.2, "200": 41.0}


class FormantStats(BaseModel):
    """B. Resonance / formants."""

    f1_median_hz: float | None = None
    f2_median_hz: float | None = None
    f3_median_hz: float | None = None
    f1_std_hz: float | None = None  # variability per formant
    f2_std_hz: float | None = None
    f3_std_hz: float | None = None
    coverage_f1: float | None = (
        None  # fraction of voiced frames where Fk was present (stability)
    )
    coverage_f2: float | None = None
    coverage_f3: float | None = None
    vowel_space_area_proxy_hz2: float | None = None
    # Proxy: convex-hull area of 5 k-means centroids in (F1, F2). A true vowel-space
    # area needs phoneme labels — see faster-whisper roadmap item.


class SpectralStats(BaseModel):
    """C. Brightness / darkness."""

    centroid_hz: float | None = None
    rolloff_85_hz: float | None = None  # 85th-percentile spectral rolloff
    bandwidth_hz: float | None = None
    hf_energy_ratio: float | None = None  # share of total STFT power in 2–6 kHz (0..1)
    tilt_db_per_octave: float | None = (
        None  # median slope of power spectrum vs log2(f); more negative = darker
    )


class VoiceQualityStats(BaseModel):
    """D. Voice quality / health."""

    hnr_db: float | None = None  # harmonics-to-noise ratio (Praat-style)
    jitter_relative_pct: float | None = (
        None  # headline; F0-normalized, speaker-comparable
    )
    jitter_absolute_ms: float | None = (
        None  # secondary; raw period-interval variability
    )
    shimmer_local_db: float | None = None
    h1_h2_db: float | None = None  # harmonic amplitude ratios; low H1/H2 ≈ breathy
    h1_h3_db: float | None = None
    h1_h4_db: float | None = None
    rms_db: float | None = None  # vocal-effort proxy (speech segments, dBFS)
    lufs: float | None = None  # whole-sample integrated loudness


class SampleFeatures(BaseModel):
    """Top-level result of the pipeline for one recorded sample."""

    sample_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_s: float
    sr: int
    n_segments: int
    pitch: PitchStats = Field(default_factory=PitchStats)
    formants: FormantStats = Field(default_factory=FormantStats)
    spectral: SpectralStats = Field(default_factory=SpectralStats)
    quality: VoiceQualityStats = Field(default_factory=VoiceQualityStats)


class SimilarityReport(BaseModel):
    """Similarity of a sample to one target voice profile (0–100 per dimension)."""

    target_name: str
    dimension_scores: dict[str, float | None] = Field(
        default_factory=dict
    )  # pitch/resonance/brightness/quality
    metric_scores: dict[str, float | None] = Field(
        default_factory=dict
    )  # dotted key -> score
    composite: float | None = None  # weighted overall

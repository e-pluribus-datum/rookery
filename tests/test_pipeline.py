import numpy as np

from rookery.config import PipelineConfig
from rookery.pipeline import analyze
from rookery.preprocess import segment
from rookery.testing import make_test_signal
from rookery.utils import semitone_change


def test_analyze_synth():
    # F0=125 Hz (not the default 215): HNR needs ≥4 harmonics below fmax=600
    # for spectral support, and ⌊600/125⌋ = 4 — at 215 Hz only 2 qualify,
    # so the gate never opens on the default tone.
    y, sr = make_test_signal(f0_hz=125.0)
    f = analyze(y, sr)

    assert abs(f.duration_s - 3.0) < 0.1
    assert f.sr == sr
    assert f.n_segments >= 1
    # pYIN should lock onto the fundamental
    assert f.pitch.median_hz is not None
    assert abs(f.pitch.median_hz - 125.0) < 30.0
    assert f.pitch.p10_hz is not None
    assert f.pitch.p90_hz is not None
    assert f.pitch.overall_range_semitones is not None
    assert f.pitch.speaking_range_semitones is not None
    # synthetic tone is ~constant 125 Hz → ranges should be near 0
    assert f.pitch.overall_range_semitones >= 0.0
    assert f.pitch.speaking_range_semitones >= 0.0
    assert f.quality.hnr_db is not None


def test_segmentation_finds_tone_between_silence():
    sr = 44100
    tone, _ = make_test_signal(duration_s=1.5)
    silence = np.zeros(sr, dtype=np.float32)  # 1 s of digital silence
    y = np.concatenate([silence, tone, silence])

    segs = segment(y.astype(np.float64), sr, PipelineConfig())
    assert len(segs) >= 1
    t0, t1 = segs[0]
    assert t0 > 0.5  # starts after the leading silence
    assert (t1 - t0) < 2.5  # doesn't cover the whole file


def test_semitone_change():
    # octave up = exactly 12 semitones
    assert semitone_change(220.0, 440.0) == 12.0
    # non-positive input is invalid
    assert semitone_change(0.0, 100.0) is None
    assert semitone_change(100.0, -5.0) is None

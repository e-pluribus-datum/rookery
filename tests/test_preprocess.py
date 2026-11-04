import numpy as np

from rookery.config import PipelineConfig
from rookery.preprocess import (
    bandpass,
    concat_segments,
    remove_dc,
    segment,
    trim_to_middle,
)
from rookery.testing import make_test_signal


def test_segmentation():
    sr = 44100
    tone, _ = make_test_signal(duration_s=1.5)
    silence = np.zeros(sr, dtype=np.float32)  # 1 s of digital silence
    y = np.concatenate([silence, tone, silence])

    segs = segment(y.astype(np.float64), sr, PipelineConfig())
    assert len(segs) == 1
    t0, t1 = segs[0]
    # tone occupies [1.0, 2.5] s; bounds follow segment()'s convention:
    # t0 = first above-threshold frame's center (abrupt onset pushes it up to
    # ~half a hop ≈ 5.8 ms early), t1 = last frame's trailing edge (+ full hop)
    assert 0.95 < t0 < 1.06  # measured ≈ 0.987 s
    assert 2.45 < t1 < 2.60  # measured ≈ 2.542 s

    # all-silence input → no segments
    assert segment(np.zeros(2 * sr, dtype=np.float64), sr, PipelineConfig()) == []


def test_trim_to_middle():
    cfg = PipelineConfig()

    # long segment: trimmed on both ends by min(max_trim_s, trim_frac * dur)
    segs = trim_to_middle([(0.0, 1.0)], cfg)
    assert len(segs) == 1
    t0, t1 = segs[0]
    assert abs(t0 - 0.05) < 1e-9
    assert abs(t1 - 0.95) < 1e-9

    # 0.12 s segment: trimming would breach the 0.1 s voiced floor → kept whole
    segs = trim_to_middle([(0.0, 0.12)], cfg)
    assert segs == [(0.0, 0.12)]


def test_bandpass():
    sr = 44100
    cfg = PipelineConfig()
    dur = 1.0
    t = np.arange(int(dur * sr)) / sr

    def rms_db(y):
        return 20.0 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12)

    for f in (30.0, 10_000.0):
        x = np.sin(2 * np.pi * f * t)
        y = bandpass(x, sr, cfg)
        # 4th-order edge rolls off −6 dB/oct: ~−19 dB at 10 kHz (only 1 oct
        # past the 8 kHz cutoff); 30 Hz sits deep in the stopband (~−37 dB)
        assert rms_db(x) - rms_db(y) > 15.0

    x = np.sin(2 * np.pi * 1000.0 * t)
    y = bandpass(x, sr, cfg)
    assert abs(rms_db(x) - rms_db(y)) < 1.0  # passband ≈ unity


def test_concat_segments():
    sr = 44100
    y = np.arange(sr, dtype=np.float64)  # 1 s
    segs = [(0.0, 0.25), (0.5, 0.75)]
    out = concat_segments(y, sr, segs)
    assert len(out) == int(0.25 * sr) + int(0.25 * sr)
    assert np.array_equal(
        out, np.concatenate([y[: int(0.25 * sr)], y[int(0.5 * sr) : int(0.75 * sr)]])
    )

    assert concat_segments(y, sr, []).size == 0


def test_remove_dc():
    sr = 44100
    t = np.arange(sr) / sr
    x = 0.25 + 0.5 * np.sin(2 * np.pi * 440.0 * t)  # 250 ms tone + DC offset
    y = remove_dc(x)
    assert abs(y.mean()) < 1e-9  # offset gone
    # AC content preserved: same shape, just re-centered
    assert np.allclose(y, x - x.mean())

from __future__ import annotations

import argparse
import contextlib
import json
import sys

import soundfile as sf

from .config import SCHEMA_VERSION, PipelineConfig
from .models import SampleFeatures
from .pipeline import analyze


def _f(v, fmt: str = "{:.1f}", unit: str = "") -> str:
    # fmt is a full template (e.g. "{:.0%}"), so format via str.format —
    # nesting it inside an f-string would treat the braces as literal.
    return "n/a" if v is None else fmt.format(v) + unit


def print_report(f: SampleFeatures, verbose: bool = False) -> None:
    """Print the human-readable A–D report (no scoring — Phase 3 decides)."""
    p, fr, sp, q = f.pitch, f.formants, f.spectral, f.quality
    pa = p.percent_above or {}

    print(
        f"sample={f.sample_id}  duration={f.duration_s:.2f}s  sr={f.sr}  segments={f.n_segments}"
    )
    print(
        "pitch      median="
        + _f(p.median_hz, unit=" Hz")
        + " overall range (P10–P90)="
        + _f(p.overall_range_semitones, unit=" st")
        + " speaking range (P25–P75)="
        + _f(p.speaking_range_semitones, unit=" st")
    )
    print(
        "           p10="
        + _f(p.p10_hz, unit=" Hz")
        + " p25="
        + _f(p.p25_hz, unit=" Hz")
        + " p75="
        + _f(p.p75_hz, unit=" Hz")
        + " p90="
        + _f(p.p90_hz, unit=" Hz")
        + " voiced="
        + _f(p.voiced_ratio, "{:.0%}")
        + " >180Hz="
        + _f(pa.get("180"), "{:.0f}", "%")
        + " >200Hz="
        + _f(pa.get("200"), "{:.0f}", "%")
    )
    if verbose:
        print(
            "           [internal] mean="
            + _f(p.mean_hz, unit=" Hz")
            + " stdev="
            + _f(p.stdev_hz, unit=" Hz")
            + " cv="
            + _f(p.cv, "{:.3f}")
        )
    print(
        "formants   F1="
        + _f(fr.f1_median_hz, unit=" Hz")
        + " F2="
        + _f(fr.f2_median_hz, unit=" Hz")
        + " F3="
        + _f(fr.f3_median_hz, unit=" Hz")
    )
    print(
        "           coverage(F1..F3)="
        + _f(fr.coverage_f1, "{:.0%}")
        + "/"
        + _f(fr.coverage_f2, "{:.0%}")
        + "/"
        + _f(fr.coverage_f3, "{:.0%}")
        + "  vowel-space proxy="
        + _f(fr.vowel_space_area_proxy_hz2, "{:.0f}", " Hz^2")
    )
    print(
        "spectral   centroid="
        + _f(sp.centroid_hz, unit=" Hz")
        + " rolloff85="
        + _f(sp.rolloff_85_hz, unit=" Hz")
        + " bandwidth="
        + _f(sp.bandwidth_hz, unit=" Hz")
    )
    print(
        "           HF ratio (2-6 kHz)="
        + _f(sp.hf_energy_ratio, "{:.3f}")
        + "  tilt="
        + _f(sp.tilt_db_per_octave, unit=" dB/oct")
    )
    print(
        "quality    HNR="
        + _f(q.hnr_db, unit=" dB")
        + " jitter="
        + _f(q.jitter_relative_pct, "{:.1f}", "%")
        + " (abs "
        + _f(q.jitter_absolute_ms, "{:.3f}", " ms)")
        + " shimmer="
        + _f(q.shimmer_local_db, unit=" dB")
    )
    print(
        "           H1/H2="
        + _f(q.h1_h2_db, unit=" dB")
        + " RMS="
        + _f(q.rms_db, unit=" dBFS")
        + " LUFS="
        + _f(q.lufs)
    )


def _envelope(
    path: str,
    src_dur_s: float,
    src_sr: int,
    segments: list[tuple[float, float]],
    noise_reduce: bool,
    features,
) -> dict:
    """Build the versioned JSON envelope (plain dict — no wrapper model)."""
    return {
        "rookery_version": "0.1.0",
        "schema_version": SCHEMA_VERSION,
        "source": {"path": path, "duration_s": src_dur_s, "sr": src_sr},
        "preprocess": {
            "segmentation": [list(s) for s in segments],
            "n_segments": len(segments),
            "noise_reduce": noise_reduce,
        },
        "features": features,
    }


def cmd_preprocess(args) -> int:
    from .capture import load_audio
    from .preprocess import preprocess

    y, sr = load_audio(args.path)
    cfg = PipelineConfig(noise_reduce=args.noise_reduce)
    y, sr, segments = preprocess(y, sr, cfg)
    # Write the concatenated trimmed segments — what extraction operates on.
    from .preprocess import concat_segments

    sf.write(args.out, concat_segments(y, sr, segments), sr)
    print(
        f"wrote {args.out} ({len(y) / sr:.2f}s @ {sr} Hz, {len(segments)} segment(s))"
    )
    return 0


def cmd_extract(args) -> int:
    from .capture import load_audio
    from .preprocess import preprocess

    y, sr = load_audio(args.path)
    cfg = PipelineConfig(noise_reduce=args.noise_reduce)
    y, sr, segments = preprocess(y, sr, cfg)

    feats = None  # aggregate SampleFeatures; only set in the non-per-segment path
    if args.per_segment:
        # One row per trimmed segment: re-run A–D on [(t0, t1)] each.
        from .features.formants import extract_formants
        from .features.pitch import extract_pitch
        from .features.quality import extract_quality
        from .features.spectral import extract_spectral

        rows = []
        for i, (t0, t1) in enumerate(segments):
            row = SampleFeatures(
                sample_id=f"{i:03d}",
                duration_s=t1 - t0,
                sr=sr,
                n_segments=1,
            )
            row.pitch = extract_pitch(y, sr, [(t0, t1)])
            row.formants = extract_formants(y, sr, [(t0, t1)])
            row.spectral = extract_spectral(y, sr, [(t0, t1)])
            row.quality = extract_quality(y, sr, [(t0, t1)])
            rows.append(row.model_dump(mode="json"))
        features: dict = {"segments": rows}
    else:
        feats = analyze(y, sr, config=cfg)
        features = feats.model_dump(mode="json")

    env = _envelope(args.path, len(y) / sr, sr, segments, cfg.noise_reduce, features)
    payload = json.dumps(env, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(payload + "\n")
        # Human report goes to stderr so stdout stays pipe-clean
        # (aggregate only; --per-segment has no single SampleFeatures to render).
        if feats is not None:
            with contextlib.redirect_stdout(sys.stderr):
                print_report(feats)
        print(f"wrote {args.out}")
    else:
        print(payload)
    return 0


def cmd_analyze(args) -> int:
    from .capture import load_audio

    y, sr = load_audio(args.path)
    cfg = PipelineConfig(noise_reduce=args.noise_reduce)
    feats = analyze(y, sr, config=cfg)
    print_report(feats, verbose=args.verbose)
    return 0


def cmd_record(args) -> int:
    from .capture import record

    y, sr = record(args.duration)
    sf.write(args.out, y, sr)
    print(f"wrote {args.out} ({len(y) / sr:.2f}s @ {sr} Hz)")
    return 0


def cmd_synth(args) -> int:
    from .testing import make_test_signal

    y, sr = make_test_signal(duration_s=args.duration)
    sf.write(args.out, y, sr)
    print(f"wrote {args.out} ({args.duration:.1f}s @ {sr} Hz)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="rookery", description="Local audio feature analysis for voice training"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "preprocess", help="pre-process an audio file and write the cleaned WAV"
    )
    p.add_argument("path")
    p.add_argument("out")
    p.add_argument("--noise-reduce", action="store_true")

    e = sub.add_parser("extract", help="pre-process + A–D extraction to JSON")
    e.add_argument("path")
    e.add_argument(
        "-o",
        dest="out",
        default=None,
        help="write JSON to FILE (human report goes to stderr); default: stdout",
    )
    e.add_argument(
        "--per-segment",
        action="store_true",
        help="one feature row per voiced segment instead of one aggregate",
    )
    e.add_argument("--noise-reduce", action="store_true")

    a = sub.add_parser(
        "analyze", help="run the feature pipeline and print the human report"
    )
    a.add_argument("path")
    a.add_argument("--noise-reduce", action="store_true")
    a.add_argument(
        "--verbose",
        action="store_true",
        help="also print internal pitch stats (mean/stdev/cv)",
    )

    r = sub.add_parser("record", help="record from the microphone to a WAV file")
    r.add_argument("out")
    r.add_argument("--duration", type=float, default=5.0)

    s = sub.add_parser("synth", help="write a synthetic test tone (offline testing)")
    s.add_argument("out")
    s.add_argument("--duration", type=float, default=3.0)

    args = ap.parse_args(argv)
    if args.cmd == "preprocess":
        return cmd_preprocess(args)
    if args.cmd == "extract":
        return cmd_extract(args)
    if args.cmd == "analyze":
        return cmd_analyze(args)
    if args.cmd == "record":
        return cmd_record(args)
    return cmd_synth(args)


if __name__ == "__main__":
    raise SystemExit(main())

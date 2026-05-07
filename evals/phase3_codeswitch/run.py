"""Phase 3 — code-switching STT eval.

Self-contained: TTS-generates the audio samples, runs three nb07 strategies
against them, computes CER + English-word recall, prints + saves a report.

Usage:
    python evals/phase3_codeswitch/run.py
    python evals/phase3_codeswitch/run.py --report-path reports/phase3.json

Pass criteria:
    avg English recall >= ground_truth.thresholds.min_avg_en_recall_for_pass
    avg CER            <= ground_truth.thresholds.max_avg_cer_for_pass
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.common import EvalReport, load_ground_truth, fail
from s3lab.text import normalize, english_recall


GROUND_TRUTH = Path(__file__).parent / "ground_truth.json"
DEFAULT_AUDIO_DIR = REPO_ROOT / "data" / "audio_samples" / "codeswitch"


def make_audio_samples(samples, audio_dir: Path):
    """TTS each sample to MP3, cache to disk. Idempotent."""
    from openai import OpenAI
    client = OpenAI()
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in samples:
        p = audio_dir / f"{s['id']}.mp3"
        if not p.exists():
            print(f"  generating {p.name}")
            client.audio.speech.create(model="tts-1", voice="alloy", input=s["text"]).stream_to_file(str(p))
        paths[s["id"]] = p
    return paths


def transcribe_with_strategy(asr, audio_path: Path, strategy: str, glossary: str) -> str:
    if strategy == "A_vanilla":
        seg, _ = asr.transcribe(str(audio_path))
    elif strategy == "B_biased":
        seg, _ = asr.transcribe(str(audio_path), language="zh", initial_prompt=glossary)
    elif strategy == "C_vad":
        seg, _ = asr.transcribe(
            str(audio_path),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
            condition_on_previous_text=False,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return " ".join(s.text.strip() for s in seg)


def cer(ref: str, hyp: str) -> float:
    """Wraps jiwer.cer with normalization. Imported lazily to keep the
    structure tests in CI dependency-free."""
    from jiwer import cer as jiwer_cer
    return jiwer_cer(normalize(ref), normalize(hyp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--report-path", type=Path, default=REPO_ROOT / "evals" / "reports" / "phase3.json")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        fail("OPENAI_API_KEY not set (for TTS sample generation)")

    gt = load_ground_truth(GROUND_TRUTH)
    samples = gt["samples"]
    strategies = gt["strategies_under_test"]
    glossary = (
        "standup, review, sync, backlog, OKR, Q1, Q2, Q3, Q4, function, complexity, "
        "optimize, schedule, deliverables, timeline, align, weekend, coffee, project, "
        "discuss, focus."
    )

    print(f"Phase 3 eval — {len(samples)} samples × {len(strategies)} strategies")

    print("ensuring audio samples...")
    paths = make_audio_samples(samples, args.audio_dir)

    print("loading faster-whisper large-v3 (int8)...")
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device="cpu", compute_type="int8")

    # One report per strategy, then a combined view.
    all_reports: dict[str, EvalReport] = {}
    for strat in strategies:
        rep = EvalReport(phase=f"phase3_codeswitch · {strat}", config={"strategy": strat})
        for s in samples:
            hyp = transcribe_with_strategy(asr, paths[s["id"]], strat, glossary)
            rep.add(
                sample_id=s["id"],
                metrics={
                    "cer": round(cer(s["text"], hyp), 3),
                    "en_recall": round(english_recall(hyp, s["expected_english"]), 3),
                },
                note=hyp[:80],
            )
        rep.aggregate_mean()
        rep.print()
        all_reports[strat] = rep

    # Pass / fail decision
    th = gt["thresholds"]
    print(f"\nThresholds for PASS: en_recall >= {th['min_avg_en_recall_for_pass']}, "
          f"cer <= {th['max_avg_cer_for_pass']}")
    print(f"{'strategy':<14} {'avg cer':>10} {'avg en_recall':>16}  {'verdict':>8}")
    any_passed = False
    for strat, rep in all_reports.items():
        passed = (
            rep.aggregate["en_recall"] >= th["min_avg_en_recall_for_pass"]
            and rep.aggregate["cer"]   <= th["max_avg_cer_for_pass"]
        )
        any_passed = any_passed or passed
        print(f"{strat:<14} {rep.aggregate['cer']:>10.3f} {rep.aggregate['en_recall']:>16.3f}  "
              f"{'PASS' if passed else 'fail':>8}")

    # Save combined JSON
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    from dataclasses import asdict
    with open(args.report_path, "w") as f:
        json.dump(
            {strat: asdict(rep) for strat, rep in all_reports.items()},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nreport saved to {args.report_path}")

    sys.exit(0 if any_passed else 1)


if __name__ == "__main__":
    main()

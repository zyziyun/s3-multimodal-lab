"""Run every available eval and aggregate the verdicts.

Skips a phase if its ground_truth.json doesn't exist (Phase 1 + Phase 4 are
templates that the user has to fill in; Phase 3 is fully self-contained).

Usage:
    python evals/run_all.py
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS = REPO_ROOT / "evals"


PHASES = [
    {
        "name": "phase1_visual_rag",
        "script": EVALS / "phase1_visual_rag" / "run.py",
        "needs_user_gt": True,
    },
    {
        "name": "phase3_codeswitch",
        "script": EVALS / "phase3_codeswitch" / "run.py",
        "needs_user_gt": False,
    },
    {
        "name": "phase4_video",
        "script": EVALS / "phase4_video" / "run.py",
        "needs_user_gt": True,
    },
]


def main():
    results = {}
    for p in PHASES:
        gt = p["script"].parent / "ground_truth.json"
        if p["needs_user_gt"] and not gt.exists():
            print(f"\n--- {p['name']}: SKIPPED (no ground_truth.json — see ground_truth.example.json)")
            results[p["name"]] = "skipped"
            continue
        print(f"\n--- {p['name']}: running")
        rc = subprocess.run([sys.executable, str(p["script"])]).returncode
        results[p["name"]] = "PASS" if rc == 0 else "fail"

    print("\n" + "=" * 50)
    print("SUMMARY")
    for name, verdict in results.items():
        print(f"  {name:<30} {verdict}")
    failed = [n for n, v in results.items() if v == "fail"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

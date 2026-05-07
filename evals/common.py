"""Shared formatting + I/O for the eval suite. Keep dependency-light: stdlib
only, so structure tests can validate eval files without installing the
heavy ML stack."""

from __future__ import annotations
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    """One row in an eval report."""
    sample_id: str
    metrics: dict[str, float]
    note: str = ""


@dataclass
class EvalReport:
    """Aggregate report for one eval run."""
    phase: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    config: dict[str, Any] = field(default_factory=dict)
    rows: list[EvalResult] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def add(self, sample_id: str, metrics: dict[str, float], note: str = "") -> None:
        self.rows.append(EvalResult(sample_id=sample_id, metrics=metrics, note=note))

    def aggregate_mean(self) -> None:
        """Populate `aggregate` with the per-metric mean across rows."""
        if not self.rows:
            return
        keys = self.rows[0].metrics.keys()
        for k in keys:
            self.aggregate[k] = round(sum(r.metrics[k] for r in self.rows) / len(self.rows), 4)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def print(self) -> None:
        print(f"\n=== {self.phase}  ({self.timestamp}) ===")
        if self.config:
            print("config:", json.dumps(self.config, ensure_ascii=False))
        if not self.rows:
            print("(no rows)")
            return
        cols = list(self.rows[0].metrics.keys())
        header = f"{'sample':<30} " + " ".join(f"{c:>12}" for c in cols)
        print(header)
        print("-" * len(header))
        for r in self.rows:
            vals = " ".join(f"{r.metrics[c]:>12.3f}" for c in cols)
            note = f"  | {r.note}" if r.note else ""
            print(f"{r.sample_id:<30} {vals}{note}")
        if self.aggregate:
            print("-" * len(header))
            agg = " ".join(f"{self.aggregate[c]:>12.3f}" for c in cols)
            print(f"{'AGGREGATE':<30} {agg}")


def load_ground_truth(path: Path) -> dict:
    """Load and minimally validate a ground-truth JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"ground truth file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be an object")
    return data


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

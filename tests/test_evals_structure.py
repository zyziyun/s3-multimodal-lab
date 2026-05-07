"""Validate the structure of every ground_truth*.json under evals/. Lightweight
— stdlib only, runs in CI without any heavy deps. Catches drift between the
runner scripts and the ground-truth schemas before students debug it the
hard way."""

import json
from pathlib import Path
import pytest


EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
GT_FILES = sorted(EVALS_DIR.rglob("ground_truth*.json"))

assert GT_FILES, f"no ground_truth*.json found under {EVALS_DIR}"


@pytest.mark.parametrize("path", GT_FILES, ids=[str(p.relative_to(EVALS_DIR)) for p in GT_FILES])
def test_ground_truth_is_valid_json(path):
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "phase" in data, f"{path}: missing 'phase'"


def _gt(name: str) -> dict:
    matches = [p for p in GT_FILES if p.parent.name == name and "example" not in p.name]
    if not matches:
        # Fall back to the example for templated phases
        matches = [p for p in GT_FILES if p.parent.name == name]
    assert matches, f"no ground truth for {name}"
    with open(matches[0]) as f:
        return json.load(f)


class TestPhase3Schema:
    def test_has_samples_with_required_fields(self):
        gt = _gt("phase3_codeswitch")
        assert "samples" in gt and isinstance(gt["samples"], list) and gt["samples"]
        for s in gt["samples"]:
            assert {"id", "text", "expected_english"} <= s.keys()
            assert isinstance(s["id"], str) and s["id"]
            assert isinstance(s["text"], str) and s["text"]
            assert isinstance(s["expected_english"], list)

    def test_unique_ids(self):
        gt = _gt("phase3_codeswitch")
        ids = [s["id"] for s in gt["samples"]]
        assert len(set(ids)) == len(ids), "duplicate sample ids"

    def test_strategies_under_test(self):
        gt = _gt("phase3_codeswitch")
        assert isinstance(gt["strategies_under_test"], list)
        assert all(isinstance(s, str) for s in gt["strategies_under_test"])

    def test_thresholds_in_unit_range(self):
        gt = _gt("phase3_codeswitch")
        th = gt["thresholds"]
        assert 0.0 <= th["min_avg_en_recall_for_pass"] <= 1.0
        assert 0.0 <= th["max_avg_cer_for_pass"] <= 1.0


class TestPhase1Schema:
    def test_template_has_required_keys(self):
        gt = _gt("phase1_visual_rag")
        assert "pdf" in gt
        assert "queries" in gt and isinstance(gt["queries"], list)
        for q in gt["queries"]:
            assert "q" in q and "expected_pages" in q
            assert isinstance(q["expected_pages"], list)
            assert all(isinstance(p, int) and p >= 1 for p in q["expected_pages"])


class TestPhase4Schema:
    def test_template_has_required_keys(self):
        gt = _gt("phase4_video")
        assert "video" in gt
        assert "queries" in gt and isinstance(gt["queries"], list)
        for q in gt["queries"]:
            assert "q" in q and "timestamp_s" in q and "tolerance_s" in q
            assert isinstance(q["timestamp_s"], (int, float))
            assert q["tolerance_s"] > 0

"""Sanity tests for MaxSim + RRF — the math that powers nb03 and nb09."""

import numpy as np
import pytest
from s3lab.retrieval import maxsim, rrf_fuse


class TestMaxSim:
    def test_identical_query_and_page_score_high(self):
        # Three identical normalized vectors on each side -> max=1, sum=3
        v = np.array([1.0, 0.0, 0.0])
        q = np.stack([v, v, v])
        p = np.stack([v, v, v])
        assert maxsim(q, p) == pytest.approx(3.0)

    def test_orthogonal_pages_score_zero(self):
        q = np.eye(3, dtype=np.float32)            # 3 token vectors
        p = -np.eye(3, dtype=np.float32) + 1e-9    # 3 patches anti-aligned
        # all dots are 0 or near-zero negative -> max picks ~0
        score = maxsim(q, p)
        assert score < 0.01

    def test_per_token_max_then_sum(self):
        # Hand-computed example. Each query token's best patch:
        #   q0 best dot = 0.9, q1 best dot = 0.5, q2 best dot = 0.7
        # Expected score = 2.1
        q = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7071, 0.7071],
        ], dtype=np.float32)
        p = np.array([
            [0.9, 0.1],
            [0.0, 0.5],
            [0.4, 0.4],
        ], dtype=np.float32)
        # Manually: per-query maxes
        manual = float((q @ p.T).max(axis=1).sum())
        assert maxsim(q, p) == pytest.approx(manual, abs=1e-6)

    def test_dim_mismatch_raises(self):
        with pytest.raises(ValueError):
            maxsim(np.zeros((2, 4)), np.zeros((3, 8)))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            maxsim(np.zeros(4), np.zeros((3, 4)))


class TestRRF:
    def test_fuses_two_lists(self):
        # Item A is rank 0 in run1, rank 2 in run2
        # Item B is rank 1 in run1, rank 0 in run2
        # Item C is rank 2 in run1, rank 1 in run2
        # With k=60, A = 1/61 + 1/63, B = 1/62 + 1/61, C = 1/63 + 1/62
        # B should win (best aggregate), then A, then C.
        result = rrf_fuse([["A", "B", "C"], ["B", "C", "A"]], k=60)
        ranked = [item for item, _ in result]
        assert ranked == ["B", "A", "C"]

    def test_single_run_preserves_order(self):
        result = rrf_fuse([[1, 2, 3, 4]])
        ranked = [item for item, _ in result]
        assert ranked == [1, 2, 3, 4]

    def test_unanimous_top_wins(self):
        # If item X is rank 0 in every run, it must end up first
        result = rrf_fuse([["X", "Y"], ["X", "Z"], ["X", "W"]])
        assert result[0][0] == "X"

    def test_score_formula_matches_canonical(self):
        # k=60, item at rank 0 in single run -> 1/61
        result = rrf_fuse([["only"]], k=60)
        assert result[0] == ("only", pytest.approx(1 / 61))

    def test_bucket_merges_close_timestamps(self):
        # Frame hit at 12s and audio hit at 14s should merge into the 10s
        # bucket if bucket=5 and tolerance allows.
        # rank_lists: run1 returns 12, run2 returns 14, both bucket to 15
        # (since round(12/5)*5 = 10, round(14/5)*5 = 15 — uneven). Let's
        # use a case where both round to the same bucket: 12 and 13 with bucket=5
        result = rrf_fuse([[12.0], [13.0]], bucket=5)
        # Both round(12/5)*5=10 and round(13/5)*5=15 — so we need closer ones
        # Use 12 and 11 -> both round to 10
        result = rrf_fuse([[12.0], [11.0]], bucket=5)
        # round(12/5)=2.4 -> 2 -> 10; round(11/5)=2.2 -> 2 -> 10
        assert len(result) == 1
        assert result[0][0] == 10.0
        assert result[0][1] == pytest.approx(2 / 61)

    def test_bucket_keeps_distant_separate(self):
        # 5s and 50s should NOT merge
        result = rrf_fuse([[5.0], [50.0]], bucket=5)
        items = {ts for ts, _ in result}
        assert items == {5.0, 50.0}

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            rrf_fuse([["A"]], k=0)
        with pytest.raises(ValueError):
            rrf_fuse([["A"]], k=-1)

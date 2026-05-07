"""Retrieval primitives: ColPali-style MaxSim + Reciprocal Rank Fusion."""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Sequence
import numpy as np


def maxsim(query_emb: np.ndarray, page_emb: np.ndarray) -> float:
    """ColPali late-interaction score (S3 §4.3).

    For each query token, take the max similarity against any patch on the
    page; sum the per-token maxes. Inputs are L2-normalized so dot product
    equals cosine.

    Args:
        query_emb: (T, D) — one row per query token.
        page_emb:  (P, D) — one row per page patch.
    Returns:
        scalar score; higher means better match.
    """
    q = np.asarray(query_emb, dtype=np.float32)
    p = np.asarray(page_emb, dtype=np.float32)
    if q.ndim != 2 or p.ndim != 2:
        raise ValueError(f"expected 2D arrays, got query {q.shape}, page {p.shape}")
    if q.shape[1] != p.shape[1]:
        raise ValueError(f"embedding dims differ: query {q.shape[1]} vs page {p.shape[1]}")
    sim = q @ p.T                      # (T, P)
    return float(sim.max(axis=1).sum())


def rrf_fuse(
    rank_lists: Iterable[Sequence],
    k: int = 60,
    bucket: float | None = None,
) -> list[tuple]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    rrf_score(item) = sum over runs of  1 / (k + rank_in_run + 1)

    Args:
        rank_lists: an iterable of ordered lists; each list is one retrieval
            run, items at index 0 are best. Items must be hashable (or, if
            `bucket` is set, must be numeric so we can bucket them).
        k: RRF dampening constant. 60 is the standard from Cormack 2009.
        bucket: if set, items are floored to nearest `bucket` units before
            fusion. Used for timestamp fusion in nb09 so frame and audio
            hits at the same moment merge.

    Returns:
        list of (item, fused_score) sorted by score descending.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")

    def maybe_bucket(item):
        if bucket is None:
            return item
        return round(float(item) / bucket) * bucket

    scores: dict = defaultdict(float)
    for run in rank_lists:
        for rank, item in enumerate(run):
            scores[maybe_bucket(item)] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])

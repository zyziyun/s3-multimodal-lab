"""Reusable, pure-Python implementations of the small utilities used across
the lab notebooks. Notebooks inline-define these for teaching clarity; this
module is the canonical version that tests run against, so CI catches drift.
"""

from .cost import gpt4o_image_tokens, gpt4o_cost_usd
from .text import normalize, english_recall, chunk_text
from .retrieval import maxsim, rrf_fuse

__all__ = [
    "gpt4o_image_tokens",
    "gpt4o_cost_usd",
    "normalize",
    "english_recall",
    "chunk_text",
    "maxsim",
    "rrf_fuse",
]

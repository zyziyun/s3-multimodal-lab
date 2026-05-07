"""GPT-4o image token + cost math (S3 §2.1). Mirrors the inline implementation
in nb01."""

from __future__ import annotations
import math


PRICING_PER_TOKEN = {
    # As of early 2026. Update when OpenAI ships changes.
    "gpt-4o-mini": {"in": 0.15 / 1_000_000, "out": 0.60 / 1_000_000},
    "gpt-4o":      {"in": 2.50 / 1_000_000, "out": 10.0 / 1_000_000},
}


def gpt4o_image_tokens(width: int, height: int, detail: str = "high") -> int:
    """Returns the number of input tokens GPT-4o will charge for an image.

    Implements the deterministic formula from OpenAI's vision pricing guide:
      detail=low  -> 85 tokens flat
      detail=high -> 85 + 170 * num_512px_tiles, after rescaling to fit a
                     2048x2048 box and shortest-side->768.
    """
    if detail not in {"low", "high"}:
        raise ValueError(f"detail must be 'low' or 'high', got {detail!r}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    if detail == "low":
        return 85

    # Step 1 — fit in 2048x2048
    if max(width, height) > 2048:
        scale = 2048 / max(width, height)
        width, height = int(width * scale), int(height * scale)

    # Step 2 — shortest side -> 768
    short = min(width, height)
    if short > 768:
        scale = 768 / short
        width, height = int(width * scale), int(height * scale)

    # Step 3 — count 512x512 tiles
    tiles_w = math.ceil(width / 512)
    tiles_h = math.ceil(height / 512)
    return 85 + 170 * tiles_w * tiles_h


def gpt4o_cost_usd(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """Returns dollar cost for a Chat Completions call given token counts."""
    if model not in PRICING_PER_TOKEN:
        raise ValueError(f"unknown model {model!r}; known: {list(PRICING_PER_TOKEN)}")
    p = PRICING_PER_TOKEN[model]
    return input_tokens * p["in"] + output_tokens * p["out"]

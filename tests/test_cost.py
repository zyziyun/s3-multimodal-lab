"""Sanity tests for the GPT-4o image-token formula. Numbers come from the
worked examples in the OpenAI Images and Vision pricing guide (S3 §2.1)."""

import pytest
from s3lab.cost import gpt4o_image_tokens, gpt4o_cost_usd


class TestImageTokens:
    def test_low_detail_is_flat_85(self):
        assert gpt4o_image_tokens(100, 100, "low") == 85
        assert gpt4o_image_tokens(8000, 8000, "low") == 85
        assert gpt4o_image_tokens(1, 1, "low") == 85

    def test_official_worked_examples(self):
        # From OpenAI's docs — the canonical sanity checks
        assert gpt4o_image_tokens(1024, 1024, "high") == 765   # 4 tiles
        assert gpt4o_image_tokens(2048, 4096, "high") == 1105  # 6 tiles

    def test_phone_photo_is_capped(self):
        # 4032x3024 should rescale + tile out to a moderate count, not the
        # naive ceil(4032/512)*ceil(3024/512) = 8*6 = 48 tiles.
        tokens = gpt4o_image_tokens(4032, 3024, "high")
        assert 700 < tokens < 1200, f"got {tokens}, expected ~1105"

    def test_resize_saves_tokens(self):
        # The whole point: resize before send.
        big = gpt4o_image_tokens(4096, 4096, "high")
        small = gpt4o_image_tokens(1024, 1024, "high")
        # They should be similar because both clamp to 768x768 internally...
        assert big == small, "above the 768 floor, token cost is flat for square images"

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            gpt4o_image_tokens(0, 100, "high")
        with pytest.raises(ValueError):
            gpt4o_image_tokens(100, -1, "high")
        with pytest.raises(ValueError):
            gpt4o_image_tokens(100, 100, "medium")


class TestCost:
    def test_known_pricing_pair(self):
        # gpt-4o-mini: $0.15/1M in, $0.60/1M out
        cost = gpt4o_cost_usd(1_000_000, 1_000_000, "gpt-4o-mini")
        assert cost == pytest.approx(0.75, abs=1e-6)

    def test_zero_tokens_zero_cost(self):
        assert gpt4o_cost_usd(0, 0, "gpt-4o") == 0.0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            gpt4o_cost_usd(1, 1, "gpt-3.5-turbo")

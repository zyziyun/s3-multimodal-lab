"""Sanity tests for normalization + code-switching metrics from nb05/nb07."""

import pytest
from s3lab.text import normalize, english_recall, chunk_text


class TestNormalize:
    def test_lowercases(self):
        assert normalize("Hello WORLD") == "hello world"

    def test_strips_ascii_punctuation(self):
        assert normalize("hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize("a   \t  b\n\n c") == "a b c"

    def test_keeps_cjk_characters(self):
        assert "你好" in normalize("你好，世界")

    def test_cjk_with_punctuation(self):
        # Chinese punctuation should be stripped, characters retained
        out = normalize("明天的 standup 我们 review。")
        assert "明天的" in out
        assert "standup" in out
        assert "review" in out
        assert "。" not in out

    def test_idempotent(self):
        s = "Q3 OKR review。 What about the 重点?"
        assert normalize(normalize(s)) == normalize(s)

    def test_handles_none(self):
        assert normalize(None) == ""

    def test_handles_empty(self):
        assert normalize("") == ""


class TestEnglishRecall:
    def test_all_present(self):
        score = english_recall(
            "明天 standup 我们 review Q3 OKR 然后 sync backlog",
            ["standup", "review", "Q3", "OKR", "sync", "backlog"],
        )
        assert score == 1.0

    def test_none_present(self):
        # Whisper translated everything to Chinese characters
        score = english_recall(
            "明天的站会我们评审第三季度目标然后同步待办",
            ["standup", "review", "Q3", "OKR", "sync", "backlog"],
        )
        assert score == 0.0

    def test_partial(self):
        score = english_recall(
            "明天 standup 我们评审 Q3 然后 sync 一下",
            ["standup", "review", "Q3", "OKR", "sync", "backlog"],
        )
        # standup + Q3 + sync = 3/6
        assert score == pytest.approx(3 / 6)

    def test_case_insensitive(self):
        assert english_recall("STANDUP", ["standup"]) == 1.0
        assert english_recall("standup", ["STANDUP"]) == 1.0

    def test_whole_word_only(self):
        # "stand" should NOT match "standup"
        assert english_recall("standing up", ["standup"]) == 0.0

    def test_empty_expected_returns_one(self):
        # Vacuously satisfied
        assert english_recall("anything", []) == 1.0


class TestChunkText:
    def test_drops_short_fragments(self):
        chunks = chunk_text("a. b. c.", max_chars=200, min_chars=30)
        assert chunks == []

    def test_respects_max_chars(self):
        long = "句子。" * 200   # 600 Chinese chars
        chunks = chunk_text(long, max_chars=100)
        assert all(len(c) <= 200 for c in chunks)
        assert len(chunks) >= 3

    def test_no_content_loss(self):
        # All non-trivial sentences should appear in some chunk
        text = "First sentence here is long enough to keep. Second sentence here is also long enough. Third sentence."
        chunks = chunk_text(text, max_chars=80, min_chars=10)
        joined = " ".join(chunks)
        assert "First" in joined
        assert "Second" in joined

    def test_handles_empty(self):
        assert chunk_text("") == []
        assert chunk_text("    ") == []

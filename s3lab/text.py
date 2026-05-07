"""Text normalization and code-switching metrics from nb05/nb07."""

from __future__ import annotations
import re


# Keep CJK Unified Ideographs (U+4E00–U+9FFF) plus alphanumerics and spaces.
_KEEP_RE = re.compile(r"[^\w\s一-鿿]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    """Lowercase, strip punctuation (preserving CJK), collapse whitespace.

    Apply identically to reference and hypothesis before any string-level
    metric so punctuation choices don't dominate the score.
    """
    if s is None:
        return ""
    s = s.lower()
    s = _KEEP_RE.sub(" ", s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def english_recall(transcript: str, expected_tokens: list[str]) -> float:
    """Fraction of `expected_tokens` (as whole words) present in `transcript`.

    Whole-word match — `\\bQ3\\b` finds `Q3` but not `Q3M`. Use this *in
    addition to* CER to catch the silent-translation failure mode where a
    Chinese-dominant model translates English jargon into Chinese characters,
    keeping CER low while erasing the English content.
    """
    if not expected_tokens:
        return 1.0
    t = normalize(transcript)
    found = sum(
        1 for w in expected_tokens
        if re.search(rf"\b{re.escape(w.lower())}\b", t)
    )
    return found / len(expected_tokens)


_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s+|\n\n+")


def chunk_text(text: str, max_chars: int = 600, min_chars: int = 30) -> list[str]:
    """Split text into chunks roughly bounded by `max_chars` on sentence
    boundaries, dropping fragments shorter than `min_chars`. Used by nb05
    for OCR-output chunking before embedding."""
    if not text:
        return []
    # Normalize: insert whitespace after Chinese sentence terminators that
    # don't already have any. The regex split below relies on whitespace
    # boundaries — Chinese text without spaces (`句子。句子。`) wouldn't split otherwise.
    text = re.sub(r"([。！？])(?!\s)", r"\1 ", text)
    sentences = _SENT_SPLIT.split(text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        s = s.strip() if s else ""
        if not s:
            continue
        if len(buf) + len(s) > max_chars and buf:
            chunks.append(buf.strip())
            buf = ""
        buf += s + " "
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) >= min_chars]

"""Unit tests for resume_assistant/rag_pipeline.py.

These tests only exercise pure, offline logic (chunking, keyword fallbacks,
the glued-word repair heuristic). They deliberately avoid calling OpenRouter,
loading the sentence-transformer model, or hitting FAISS, so they run fast
and without network access or an API key.
"""
from __future__ import annotations

import pytest

from resume_assistant.rag_pipeline import (
    _generate_interview_questions_fallback,
    _identify_strengths_and_gaps_fallback,
    _repair_glued_words,
    _summarize_candidate_profile_fallback,
    _suggest_resume_improvements_fallback,
    split_text_into_chunks,
)


class TestSplitTextIntoChunks:
    def test_empty_text_returns_no_chunks(self):
        assert split_text_into_chunks("") == []

    def test_short_text_returns_single_chunk(self):
        text = "Short resume text under the chunk size."
        assert split_text_into_chunks(text, chunk_size=500) == [text]

    def test_long_text_is_split_into_multiple_overlapping_chunks(self):
        words = [f"word{i}" for i in range(1200)]
        text = " ".join(words)
        chunks = split_text_into_chunks(text, chunk_size=500, overlap=80)

        assert len(chunks) > 1
        # Every chunk should contain at least chunk_size words (except possibly the last).
        for chunk in chunks[:-1]:
            assert len(chunk.split()) == 500
        # The final words of the text must appear in the last chunk.
        assert "word1199" in chunks[-1]

    def test_overlap_creates_shared_words_between_consecutive_chunks(self):
        words = [f"w{i}" for i in range(1000)]
        text = " ".join(words)
        chunks = split_text_into_chunks(text, chunk_size=300, overlap=50)

        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        assert first_words & second_words  # some overlap must exist


class TestRepairGluedWords:
    def test_splits_lowercase_to_uppercase_boundary(self):
        assert _repair_glued_words("JamesHospital") == "James Hospital"

    def test_splits_letter_digit_boundaries(self):
        assert _repair_glued_words("Ph2002gmail") == "Ph 2002 gmail"

    def test_leaves_normal_text_untouched(self):
        text = "This is normal text with spaces already"
        assert _repair_glued_words(text) == text

    def test_collapses_double_spaces(self):
        assert _repair_glued_words("a  b") == "a b"


class TestFallbackHeuristics:
    """These only run when no OpenRouter key is supplied, so they must never crash
    and must always return a sensible non-empty result."""

    def test_interview_questions_fallback_never_empty(self):
        assert len(_generate_interview_questions_fallback("")) >= 1
        assert len(_generate_interview_questions_fallback("Python FastAPI Docker team lead")) >= 1

    def test_summary_fallback_handles_unrelated_domain_gracefully(self):
        result = _summarize_candidate_profile_fallback("Anaesthesia technician, cardiac surgery, ventilator management")
        assert result  # should not crash, should return *something*

    def test_strengths_and_gaps_fallback_never_empty(self):
        result = _identify_strengths_and_gaps_fallback("Python Docker team lead")
        assert "Strengths" in result

    def test_resume_improvements_fallback_never_empty(self):
        assert len(_suggest_resume_improvements_fallback("")) >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
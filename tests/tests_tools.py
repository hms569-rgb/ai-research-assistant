"""
Tests for the 3 research tools.
Run with: pytest tests/ -v
"""

from tools.tools import extract_keywords, word_count


# ── Keyword Extractor Tests ────────────────────────────────────────────────────

def test_extract_keywords_returns_list():
    text = "machine learning models training neural networks deep learning research"
    result = extract_keywords(text)
    assert isinstance(result, list)

def test_extract_keywords_not_empty():
    text = "machine learning models training neural networks deep learning research"
    result = extract_keywords(text)
    assert len(result) > 0

def test_extract_keywords_empty_input():
    result = extract_keywords("")
    assert result == []

def test_extract_keywords_removes_stopwords():
    text = "the and for are but with this that they"
    result = extract_keywords(text)
    assert result == []

def test_extract_keywords_top_n():
    text = "research research research learning learning data data data data science"
    result = extract_keywords(text, top_n=2)
    assert len(result) <= 2


# ── Word Count Tests ───────────────────────────────────────────────────────────

def test_word_count_basic():
    result = word_count("Hello world. This is a test sentence.")
    assert result["words"] == 7

def test_word_count_sentences():
    result = word_count("First sentence. Second sentence. Third one.")
    assert result["sentences"] == 3

def test_word_count_empty():
    result = word_count("")
    assert result["words"] == 0
    assert result["sentences"] == 0
    assert result["characters"] == 0

def test_word_count_has_all_keys():
    result = word_count("Some sample text here.")
    expected_keys = [
        "words",
        "sentences",
        "characters",
        "paragraphs",
        "estimated_read_time",
        "estimated_study_time",
    ]
    for key in expected_keys:
        assert key in result

def test_word_count_characters():
    text = "Hello"
    result = word_count(text)
    assert result["characters"] == 5

def test_word_count_paragraphs():
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    result = word_count(text)
    assert result["paragraphs"] == 3
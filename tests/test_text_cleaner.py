"""Unit tests for Text Cleaner Module (Company Step 2)."""

import pytest
from src.preprocessing.text_cleaner import clean_text


def test_clean_text_empty_and_null():
    """Verify safe handling of null, missing, empty, or whitespace-only inputs."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""
    assert clean_text(12345) == "12345"


def test_clean_text_html_tags_and_entities():
    """Verify stripping of HTML tags and unescaping of HTML entities."""
    raw = "<p>Welcome to <b>Social Media</b> Intelligence &amp; NLP! &lt;tag&gt; &#39;quotes&#39; &quot;test&quot;.</p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "Social Media Intelligence & NLP! <tag> 'quotes' \"test\"." in cleaned


def test_clean_text_urls_removal():
    """Verify removal of raw URLs (http, https, ftp, www) while preserving adjacent punctuation."""
    raw = "Read the documentation at https://sparknlp.org/docs and ftp://data.repo/file or visit www.example.com. Great tool!"
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "ftp://" not in cleaned
    assert "www.example.com" not in cleaned
    assert "Read the documentation at and or visit . Great tool!" in cleaned or "Read the documentation at and or visit. Great tool!" in cleaned


def test_clean_text_markdown_links():
    """Verify markdown link syntax preserves display text and strips target URL."""
    raw = "Check out [Spark NLP Guide](https://sparknlp.org/install) for setup."
    cleaned = clean_text(raw)
    assert "Spark NLP Guide" in cleaned
    assert "https://sparknlp.org/install" not in cleaned


def test_clean_text_invisible_control_characters():
    """Verify removal of zero-width spaces, BOM, and non-printable control characters."""
    raw = "Clean\u200bText\ufeffWith\u200cInvisible\x00Noise"
    cleaned = clean_text(raw)
    assert "\u200b" not in cleaned
    assert "\ufeff" not in cleaned
    assert "\u200c" not in cleaned
    assert "\x00" not in cleaned
    assert "CleanTextWithInvisibleNoise" in cleaned


def test_clean_text_character_elongation_normalization():
    """Verify 3+ repeated characters are reduced to 2 without altering double letters."""
    raw = "I looooove this sooooo much!!!!!! Good coffee and free food."
    cleaned = clean_text(raw)
    assert "loove" in cleaned
    assert "soo" in cleaned
    assert "!!" in cleaned
    assert "!!!!!!" not in cleaned
    # Ensure legitimate double characters are NOT altered
    assert "Good" in cleaned
    assert "free" in cleaned


def test_clean_text_preserves_numbers_and_technical_terms():
    """Verify technical terms, versions, currency, and numbers are not altered or removed."""
    raw = "Evaluating GPT-4 and Python 3.12 on 10,000 records with 99.5% accuracy for $250.00."
    cleaned = clean_text(raw)
    assert "GPT-4" in cleaned
    assert "Python 3.12" in cleaned
    assert "10,000" in cleaned
    assert "99.5%" in cleaned
    assert "$250.00" in cleaned


def test_clean_text_preserves_hashtags_and_mentions():
    """Verify hashtags and user mentions are retained."""
    raw = "Great discussion with @john_doe and u/jane_doe on #SparkNLP and #MachineLearning!"
    cleaned = clean_text(raw)
    assert "@john_doe" in cleaned
    assert "u/jane_doe" in cleaned
    assert "#SparkNLP" in cleaned
    assert "#MachineLearning" in cleaned


def test_clean_text_preserves_case_and_sentence_boundaries():
    """Verify text is NOT blindly lowercased and sentence boundaries are preserved."""
    raw = "BERT and RoBERTa are powerful Transformer models. Do they scale well? Yes!"
    cleaned = clean_text(raw)
    assert "BERT" in cleaned
    assert "RoBERTa" in cleaned
    assert "Transformer" in cleaned
    assert "well? Yes!" in cleaned

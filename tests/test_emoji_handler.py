"""Unit tests for Emoji and Emoticon Handler Module (Company Step 2)."""

import pytest
from src.preprocessing.emoji_handler import handle_emojis


def test_handle_emojis_empty_and_null():
    """Verify safe handling of null, missing, or whitespace inputs."""
    assert handle_emojis(None) == ""
    assert handle_emojis("") == ""
    assert handle_emojis("   \t  ") == ""


def test_handle_emojis_single_emoji():
    """Verify conversion of single emoji into semantic descriptor."""
    raw = "Loving the results 🔥"
    processed = handle_emojis(raw)
    assert "fire" in processed
    assert "🔥" not in processed


def test_handle_emojis_multiple_emojis_in_sentence():
    """Verify conversion of multiple emojis matching the company example."""
    raw = "This product is amazing 😍🔥"
    processed = handle_emojis(raw)
    assert processed == "This product is amazing smiling_face_with_heart_eyes fire"


def test_handle_emojis_text_without_emojis():
    """Verify plain text remains unchanged without unwanted modifications."""
    raw = "This is a clean sentence without any special emoticons."
    assert handle_emojis(raw) == raw


def test_handle_emojis_common_emoticons():
    """Verify conversion of ASCII emoticons into sentiment tokens."""
    raw = "Happy to see you :) having fun :D with love <3"
    processed = handle_emojis(raw)
    assert "smiling_face" in processed
    assert "grinning_face" in processed
    assert "red_heart" in processed
    assert ":)" not in processed
    assert ":D" not in processed
    assert "<3" not in processed


def test_handle_emojis_preserves_colons_in_ordinary_text():
    """Verify colons in timestamps, URLs, or notes are not corrupted."""
    raw = "Note: Meeting starts at 10:30 AM tomorrow."
    processed = handle_emojis(raw)
    assert processed == "Note: Meeting starts at 10:30 AM tomorrow."


def test_handle_emojis_spacing_normalized():
    """Verify multiple spaces are not introduced around converted emojis."""
    raw = "Great work! 🚀🎉 Well done."
    processed = handle_emojis(raw)
    assert "  " not in processed
    assert "rocket party_popper" in processed


def test_handle_emojis_preserves_casing_and_punctuation():
    """Verify original text casing and punctuation are preserved."""
    raw = "AMAZING achievement, congratulations! 👏"
    processed = handle_emojis(raw)
    assert "AMAZING achievement, congratulations!" in processed
    assert "clapping_hands" in processed

"""Unit tests for SentimentTemporalTracker (Company Step 3).

Verifies timestamp parsing across UNIX epoch, ISO-8601 strings, and datetimes.
Verifies temporal aggregation, net sentiment computation, window grouping (daily/weekly),
and edge case handling for invalid/missing temporal values.
"""

from datetime import datetime, timezone
import pytest

from src.sentiment.temporal_tracker import SentimentTemporalTracker


# ==============================================================================
# 1. Timestamp Parsing Tests
# ==============================================================================

def test_parse_unix_integer_and_float() -> None:
    """Tests parsing of standard UNIX epoch seconds (int and float)."""
    # 1730101200 is 2024-10-28 07:40:00 UTC
    dt_int = SentimentTemporalTracker.parse_timestamp(1730101200)
    assert dt_int is not None
    assert dt_int.tzinfo == timezone.utc
    assert dt_int.year == 2024
    assert dt_int.month == 10
    assert dt_int.day == 28

    dt_float = SentimentTemporalTracker.parse_timestamp(1730101200.5)
    assert dt_float is not None
    assert dt_float.microsecond == 500000


def test_parse_numeric_strings() -> None:
    """Tests parsing of stringified timestamps."""
    dt = SentimentTemporalTracker.parse_timestamp("1730101200")
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 10
    assert dt.day == 28


def test_parse_millisecond_timestamp() -> None:
    """Tests that timestamps in milliseconds are automatically detected and converted."""
    dt = SentimentTemporalTracker.parse_timestamp(1730101200000)
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 10
    assert dt.day == 28


def test_parse_iso8601_strings() -> None:
    """Tests ISO-8601 string parsing with Z suffix and explicit offsets."""
    dt_z = SentimentTemporalTracker.parse_timestamp("2026-11-01T12:00:00Z")
    assert dt_z is not None
    assert dt_z.year == 2026
    assert dt_z.month == 11
    assert dt_z.day == 1
    assert dt_z.hour == 12
    assert dt_z.tzinfo == timezone.utc

    dt_offset = SentimentTemporalTracker.parse_timestamp("2026-11-05T16:54:13.868878+00:00")
    assert dt_offset is not None
    assert dt_offset.year == 2026
    assert dt_offset.month == 11
    assert dt_offset.day == 5


def test_parse_datetime_instances() -> None:
    """Tests passing existing datetime instances (naive and timezone-aware)."""
    naive = datetime(2025, 5, 10, 14, 30)
    parsed_naive = SentimentTemporalTracker.parse_timestamp(naive)
    assert parsed_naive is not None
    assert parsed_naive.tzinfo == timezone.utc

    aware = datetime(2025, 5, 10, 14, 30, tzinfo=timezone.utc)
    parsed_aware = SentimentTemporalTracker.parse_timestamp(aware)
    assert parsed_aware == aware


def test_parse_invalid_and_empty_timestamps() -> None:
    """Tests that unparseable or empty values safely return None."""
    assert SentimentTemporalTracker.parse_timestamp(None) is None
    assert SentimentTemporalTracker.parse_timestamp("") is None
    assert SentimentTemporalTracker.parse_timestamp("   ") is None
    assert SentimentTemporalTracker.parse_timestamp("not-a-timestamp") is None
    assert SentimentTemporalTracker.parse_timestamp(float("nan")) is None
    assert SentimentTemporalTracker.parse_timestamp(float("inf")) is None


# ==============================================================================
# 2. Window Formatting Tests
# ==============================================================================

def test_format_window_key() -> None:
    """Tests daily, weekly, and monthly bucket key formatting."""
    dt = datetime(2024, 10, 28, 15, 0, 0, tzinfo=timezone.utc)

    # Daily: YYYY-MM-DD
    assert SentimentTemporalTracker.format_window_key(dt, window="daily") == "2024-10-28"

    # Weekly: ISO year and ISO week number (2024-10-28 is Monday of Week 44)
    assert SentimentTemporalTracker.format_window_key(dt, window="weekly") == "2024-W44"

    # Monthly: YYYY-MM
    assert SentimentTemporalTracker.format_window_key(dt, window="monthly") == "2024-10"


# ==============================================================================
# 3. Sentiment Metric Calculations & Net Sentiment Tests
# ==============================================================================

def test_calculate_window_metrics_empty() -> None:
    """Tests metric calculation on an empty list."""
    metrics = SentimentTemporalTracker.calculate_window_metrics([])
    assert metrics["total"] == 0
    assert metrics["positive"] == 0
    assert metrics["neutral"] == 0
    assert metrics["negative"] == 0
    assert metrics["undetermined"] == 0
    assert metrics["positive_pct"] == 0.0
    assert metrics["net_sentiment"] == 0.0


def test_calculate_window_metrics_balanced() -> None:
    """Tests metrics with a balanced distribution."""
    records = [
        {"sentiment_label": "positive"},
        {"sentiment_label": "positive"},
        {"sentiment_label": "neutral"},
        {"sentiment_label": "negative"},
    ]
    metrics = SentimentTemporalTracker.calculate_window_metrics(records)
    assert metrics["total"] == 4
    assert metrics["positive"] == 2
    assert metrics["neutral"] == 1
    assert metrics["negative"] == 1
    assert metrics["undetermined"] == 0
    assert metrics["positive_pct"] == 50.0
    assert metrics["neutral_pct"] == 25.0
    assert metrics["negative_pct"] == 25.0
    # Net sentiment: (2 - 1) / 4 = +0.25
    assert metrics["net_sentiment"] == 0.25


def test_calculate_window_metrics_extremes() -> None:
    """Tests net sentiment boundaries (+1.0 and -1.0)."""
    all_pos = [{"sentiment_label": "positive"}] * 5
    pos_metrics = SentimentTemporalTracker.calculate_window_metrics(all_pos)
    assert pos_metrics["net_sentiment"] == 1.0
    assert pos_metrics["positive_pct"] == 100.0

    all_neg = [{"sentiment_label": "negative"}] * 3
    neg_metrics = SentimentTemporalTracker.calculate_window_metrics(all_neg)
    assert neg_metrics["net_sentiment"] == -1.0
    assert neg_metrics["negative_pct"] == 100.0


# ==============================================================================
# 4. Temporal Aggregation by Window Tests
# ==============================================================================

def test_aggregate_by_window_daily() -> None:
    """Tests multi-day daily aggregation with chronological sorting."""
    records = [
        {"created_utc": 1730101200, "sentiment_label": "positive"},  # 2024-10-28
        {"created_utc": 1730104800, "sentiment_label": "neutral"},   # 2024-10-28
        {"created_utc": 1730187600, "sentiment_label": "negative"},  # 2024-10-29
        {"created_utc": 1730191200, "sentiment_label": "negative"},  # 2024-10-29
    ]

    daily = SentimentTemporalTracker.aggregate_by_window(records, time_field="created_utc", window="daily")

    assert list(daily.keys()) == ["2024-10-28", "2024-10-29"]
    assert daily["2024-10-28"]["total"] == 2
    assert daily["2024-10-28"]["positive"] == 1
    assert daily["2024-10-28"]["neutral"] == 1
    assert daily["2024-10-28"]["net_sentiment"] == 0.5

    assert daily["2024-10-29"]["total"] == 2
    assert daily["2024-10-29"]["negative"] == 2
    assert daily["2024-10-29"]["net_sentiment"] == -1.0


def test_aggregate_by_window_weekly() -> None:
    """Tests weekly aggregation into ISO weeks."""
    records = [
        {"created_utc": 1730101200, "sentiment_label": "positive"},  # 2024-10-28 (W44)
        {"created_utc": 1730878800, "sentiment_label": "neutral"},   # 2024-11-06 (W45)
    ]

    weekly = SentimentTemporalTracker.aggregate_by_window(records, time_field="created_utc", window="weekly")
    assert "2024-W44" in weekly
    assert "2024-W45" in weekly
    assert weekly["2024-W44"]["positive"] == 1
    assert weekly["2024-W45"]["neutral"] == 1


def test_aggregate_by_window_with_unknown_and_fallback() -> None:
    """Tests handling of missing timestamps, fallback field, and unknown bucket sorting."""
    records = [
        {"created_utc": 1730101200, "sentiment_label": "positive"},
        # Missing created_utc, fallback to collected_at
        {"collected_at": "2024-10-28T12:00:00Z", "sentiment_label": "neutral"},
        # Completely missing timestamp
        {"title": "No timestamp post", "sentiment_label": "negative"},
        # Corrupt timestamp string
        {"created_utc": "corrupted_timestamp", "sentiment_label": "negative"},
    ]

    daily = SentimentTemporalTracker.aggregate_by_window(
        records,
        time_field="created_utc",
        fallback_time_field="collected_at",
        window="daily",
    )

    # Check bucket order: date buckets first, "unknown" last
    keys = list(daily.keys())
    assert keys[0] == "2024-10-28"
    assert keys[-1] == "unknown"

    assert daily["2024-10-28"]["total"] == 2
    assert daily["unknown"]["total"] == 2
    assert daily["unknown"]["negative"] == 2


def test_aggregate_by_window_type_error() -> None:
    """Tests that non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Expected records to be a list"):
        SentimentTemporalTracker.aggregate_by_window("invalid")  # type: ignore

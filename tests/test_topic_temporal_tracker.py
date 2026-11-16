"""Unit tests for TopicTemporalTracker (Company Step 4).

Verifies timestamp parsing, temporal grouping (daily, weekly, monthly), topic frequency shares,
advanced dynamics (velocity, acceleration, breakout status), and cross-platform divergence
(Reddit vs. YouTube) via Jensen-Shannon divergence.
"""

from datetime import datetime, timezone
import pytest

from src.topics.topic_temporal_tracker import (
    STATUS_BREAKOUT,
    STATUS_DECAY,
    STATUS_DORMANT,
    STATUS_EMERGING,
    STATUS_PLATEAU,
    STATUS_SUSTAINED_RISE,
    TRAJECTORY_EMERGING,
    TRAJECTORY_FALL,
    TRAJECTORY_RISE,
    TRAJECTORY_STABLE,
    TopicTemporalTracker,
)


# ==============================================================================
# 1. Timestamp Parsing Tests
# ==============================================================================

def test_parse_unix_timestamp() -> None:
    """Tests parsing of UNIX epoch timestamps (seconds and milliseconds)."""
    # 1730101200 is 2024-10-28 07:40:00 UTC
    dt = TopicTemporalTracker.parse_timestamp(1730101200)
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 10
    assert dt.day == 28

    dt_ms = TopicTemporalTracker.parse_timestamp(1730101200000)
    assert dt_ms is not None
    assert dt_ms.year == 2024
    assert dt_ms.day == 28


def test_parse_iso8601_and_strings() -> None:
    """Tests parsing of ISO-8601 strings and numeric strings."""
    dt_iso = TopicTemporalTracker.parse_timestamp("2026-11-01T12:00:00Z")
    assert dt_iso is not None
    assert dt_iso.year == 2026
    assert dt_iso.month == 11
    assert dt_iso.day == 1

    dt_str = TopicTemporalTracker.parse_timestamp("1730101200")
    assert dt_str is not None
    assert dt_str.year == 2024


def test_parse_invalid_timestamps() -> None:
    """Tests that unparseable or empty values safely return None."""
    assert TopicTemporalTracker.parse_timestamp(None) is None
    assert TopicTemporalTracker.parse_timestamp("") is None
    assert TopicTemporalTracker.parse_timestamp("invalid-date") is None
    assert TopicTemporalTracker.parse_timestamp(float("nan")) is None


# ==============================================================================
# 2. Window Formatting Tests
# ==============================================================================

def test_format_window_key() -> None:
    """Tests bucket key formatting across daily, weekly, and monthly windows."""
    dt = datetime(2024, 10, 28, 10, 0, tzinfo=timezone.utc)
    assert TopicTemporalTracker.format_window_key(dt, "daily") == "2024-10-28"
    assert TopicTemporalTracker.format_window_key(dt, "weekly") == "2024-W44"
    assert TopicTemporalTracker.format_window_key(dt, "monthly") == "2024-10"


# ==============================================================================
# 3. Topic Frequency Tracking Over Time
# ==============================================================================

def test_track_topic_frequency_empty_records() -> None:
    """Tests track_topic_frequency_over_time on empty list."""
    res = TopicTemporalTracker.track_topic_frequency_over_time([])
    assert res["windows"] == {}
    assert res["topic_summaries"] == {}
    assert res["all_topics"] == []


def test_track_topic_frequency_type_error() -> None:
    """Tests non-list input raises TypeError."""
    with pytest.raises(TypeError, match="Expected records to be a list"):
        TopicTemporalTracker.track_topic_frequency_over_time("not a list")  # type: ignore


def test_track_topic_frequency_multi_window() -> None:
    """Tests topic distribution across multiple time windows with share percentages."""
    records = [
        {"created_utc": 1730101200, "topic_id": 0},  # 2024-10-28
        {"created_utc": 1730104800, "topic_id": 0},  # 2024-10-28
        {"created_utc": 1730108400, "topic_id": 1},  # 2024-10-28
        {"created_utc": 1730187600, "topic_id": 0},  # 2024-10-29
        {"created_utc": 1730191200, "topic_id": 1},  # 2024-10-29
        {"created_utc": 1730194800, "topic_id": 1},  # 2024-10-29
    ]

    result = TopicTemporalTracker.track_topic_frequency_over_time(records, window="daily")

    windows = result["windows"]
    assert "2024-10-28" in windows
    assert "2024-10-29" in windows

    # Day 1: 2 of Topic 0 (66.67%), 1 of Topic 1 (33.33%)
    assert windows["2024-10-28"]["total_documents"] == 3
    assert windows["2024-10-28"]["topic_counts"]["0"] == 2
    assert windows["2024-10-28"]["topic_counts"]["1"] == 1
    assert windows["2024-10-28"]["topic_shares"]["0"] == pytest.approx(66.67, abs=0.01)

    # Day 2: 1 of Topic 0 (33.33%), 2 of Topic 1 (66.67%)
    assert windows["2024-10-29"]["total_documents"] == 3
    assert windows["2024-10-29"]["topic_counts"]["0"] == 1
    assert windows["2024-10-29"]["topic_counts"]["1"] == 2


def test_track_topic_frequency_with_unknown_timestamps() -> None:
    """Tests fallback timestamp field and routing to unknown bucket."""
    records = [
        {"created_utc": 1730101200, "topic_id": 0},
        {"collected_at": "2024-10-28T12:00:00Z", "topic_id": 1},  # fallback
        {"title": "Missing time", "topic_id": -1},                  # unknown
    ]

    result = TopicTemporalTracker.track_topic_frequency_over_time(records)
    windows = result["windows"]
    assert "2024-10-28" in windows
    assert "unknown" in windows
    assert windows["unknown"]["topic_counts"]["-1"] == 1


# ==============================================================================
# 4. Trajectory, Velocity & Acceleration Dynamics Tests
# ==============================================================================

def test_calculate_topic_dynamics_breakout_and_decay() -> None:
    """Tests breakout (accelerating growth) and decay dynamics over 3 time windows."""
    temporal_data = {
        "time_windows": ["2024-10-27", "2024-10-28", "2024-10-29"],
        "topic_summaries": {
            # Topic 0: 1 -> 3 -> 8.
            # v1 = 3 - 1 = 2; v2 = 8 - 3 = 5; a2 = 5 - 2 = 3 > 0 -> BREAKOUT
            "0": {"window_counts": {"2024-10-27": 1, "2024-10-28": 3, "2024-10-29": 8}},
            # Topic 1: 10 -> 6 -> 2.
            # v1 = 6 - 10 = -4; v2 = 2 - 6 = -4 -> DECAY
            "1": {"window_counts": {"2024-10-27": 10, "2024-10-28": 6, "2024-10-29": 2}},
            # Topic 2: 4 -> 4 -> 4.
            # v = 0 -> PLATEAU
            "2": {"window_counts": {"2024-10-27": 4, "2024-10-28": 4, "2024-10-29": 4}},
        },
    }

    dynamics = TopicTemporalTracker.calculate_topic_dynamics(temporal_data)

    # Topic 0: Breakout
    assert dynamics["0"]["status"] == STATUS_BREAKOUT
    assert dynamics["0"]["velocity"] == 5
    assert dynamics["0"]["acceleration"] == 3
    assert dynamics["0"]["moving_average"] == pytest.approx(4.0, abs=0.01)

    # Topic 1: Decay
    assert dynamics["1"]["status"] == STATUS_DECAY
    assert dynamics["1"]["velocity"] == -4

    # Topic 2: Plateau
    assert dynamics["2"]["status"] == STATUS_PLATEAU
    assert dynamics["2"]["velocity"] == 0


def test_calculate_topic_trajectories_single_window() -> None:
    """Tests trajectory behavior when only one time window exists (emerging)."""
    temporal_data = {
        "time_windows": ["2024-10-28"],
        "topic_summaries": {
            "0": {"window_counts": {"2024-10-28": 4}},
        },
    }

    trajectories = TopicTemporalTracker.calculate_topic_trajectories(temporal_data)
    assert trajectories["0"]["trajectory"] == TRAJECTORY_EMERGING
    assert trajectories["0"]["latest_count"] == 4
    assert trajectories["0"]["previous_window"] is None


# ==============================================================================
# 5. Cross-Platform Divergence Tests
# ==============================================================================

def test_calculate_cross_platform_divergence() -> None:
    """Tests cross-platform topic divergence calculation using Jensen-Shannon divergence."""
    records = [
        # Reddit focused on Topic 0
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730101200},
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730101200},
        {"source_platform": "reddit", "topic_id": 1, "created_utc": 1730101200},
        # YouTube focused on Topic 1
        {"source_platform": "youtube", "topic_id": 1, "created_utc": 1730101200},
        {"source_platform": "youtube", "topic_id": 1, "created_utc": 1730101200},
        {"source_platform": "youtube", "topic_id": 0, "created_utc": 1730101200},
    ]

    div_res = TopicTemporalTracker.calculate_cross_platform_divergence(records)

    assert "overall_divergence" in div_res
    assert "topic_adoption_gaps" in div_res
    assert "window_divergences" in div_res

    # Overall JS divergence must be > 0 and <= 1.0
    assert 0.0 < div_res["overall_divergence"] <= 1.0

    # Topic 0: 66.7% Reddit, 33.3% YouTube -> over-indexed on Reddit
    gaps = div_res["topic_adoption_gaps"]
    assert gaps["0"]["over_indexed_on"] == "reddit"
    assert gaps["0"]["gap_pct"] > 0

    # Topic 1: 33.3% Reddit, 66.7% YouTube -> over-indexed on YouTube
    assert gaps["1"]["over_indexed_on"] == "youtube"
    assert gaps["1"]["gap_pct"] < 0


def test_temporal_tracking_with_real_timestamps_and_multiday_fixtures() -> None:
    """Tests temporal tracking, trajectories, and divergence with realistic multi-day fixtures."""
    # 5-day realistic fixture with Reddit created_utc and YouTube published_at
    records = [
        # Day 1: 2024-10-27
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730016000},
        {"source_platform": "youtube", "topic_id": 0, "published_at": "2024-10-27T14:30:00Z"},
        # Day 2: 2024-10-28 (Topic 0 grows, Topic 1 appears)
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730102400},
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730106000},
        {"source_platform": "youtube", "topic_id": 0, "published_at": "2024-10-28T10:00:00Z"},
        {"source_platform": "youtube", "topic_id": 1, "published_at": "2024-10-28T16:00:00Z"},
        # Day 3: 2024-10-29 (Topic 0 accelerates: 2 -> 3 -> 6, breakout; Topic 1 stable)
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730188800},
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730192400},
        {"source_platform": "reddit", "topic_id": 0, "created_utc": 1730196000},
        {"source_platform": "youtube", "topic_id": 0, "published_at": "2024-10-29T08:00:00Z"},
        {"source_platform": "youtube", "topic_id": 0, "published_at": "2024-10-29T12:00:00Z"},
        {"source_platform": "youtube", "topic_id": 0, "published_at": "2024-10-29T18:00:00Z"},
        {"source_platform": "youtube", "topic_id": 1, "published_at": "2024-10-29T20:00:00Z"},
        # Day 4: 2024-10-30 (Topic -1 noise outliers)
        {"source_platform": "reddit", "topic_id": -1, "created_utc": 1730275200},
        {"source_platform": "youtube", "topic_id": -1, "published_at": "2024-10-30T11:00:00Z"},
    ]

    for rec in records:
        ts = rec.get("created_utc") or rec.get("published_at")
        rec["_temporal_timestamp"] = ts

    # 1. Daily tracking
    daily = TopicTemporalTracker.track_topic_frequency_over_time(
        records,
        topic_field="topic_id",
        time_field="_temporal_timestamp",
        window="daily",
    )
    assert "2024-10-27" in daily["windows"]
    assert "2024-10-28" in daily["windows"]
    assert "2024-10-29" in daily["windows"]
    assert "2024-10-30" in daily["windows"]

    # 2. Dynamics & Trajectories
    dynamics = TopicTemporalTracker.calculate_topic_dynamics(daily)
    assert "0" in dynamics
    # Topic 0 counts across 4 windows: Day 1=2, Day 2=3, Day 3=6, Day 4=0.
    # Latest transition (Day 3 -> Day 4): v = 0 - 6 = -6 -> DECAY
    assert dynamics["0"]["status"] == STATUS_DECAY
    assert dynamics["0"]["velocity"] == -6

    trajectories = TopicTemporalTracker.calculate_topic_trajectories(daily)
    assert "0" in trajectories
    assert trajectories["0"]["trajectory"] == TRAJECTORY_FALL

    # 3. Cross-platform divergence
    div = TopicTemporalTracker.calculate_cross_platform_divergence(
        records,
        platform_field="source_platform",
        topic_field="topic_id",
        time_field="_temporal_timestamp",
        window="daily",
    )
    assert 0.0 <= div["overall_divergence"] <= 1.0
    assert "topic_adoption_gaps" in div
    assert "window_divergences" in div


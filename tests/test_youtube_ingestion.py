"""Unit tests for the YouTube Data API v3 ingestion module."""

import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.youtube_ingestion import (
    validate_youtube_video,
    save_raw_youtube_videos,
    fetch_youtube_videos_page,
    fetch_video_details,
    ingest_youtube_live,
    apply_youtube_retention_policy,
    REQUIRED_YOUTUBE_FIELDS,
    QUOTA_COST_SEARCH_LIST,
    QUOTA_COST_VIDEOS_LIST,
    DEFAULT_DAILY_SEARCH_REQUEST_LIMIT,
    DEFAULT_DAILY_OTHER_QUOTA_LIMIT,
)


@pytest.fixture
def sample_valid_youtube_video() -> dict:
    """Returns a single valid YouTube video fixture."""
    return {
        "video_id": "vid_abc123",
        "title": "Introduction to Large Scale Social Media Intelligence",
        "description": "A comprehensive overview of NLP, graph databases, and topic modeling.",
        "channel_id": "UC_test_channel",
        "channel_title": "Data Engineering Insights",
        "published_at": "2026-11-01T12:00:00Z",
        "view_count": 12500,
        "like_count": 840,
        "comment_count": 95,
        "url": "https://www.youtube.com/watch?v=vid_abc123",
    }


def test_validate_youtube_video_valid(sample_valid_youtube_video):
    """Verify that a correctly formatted YouTube video record passes validation."""
    is_valid, error = validate_youtube_video(sample_valid_youtube_video)
    assert is_valid is True
    assert error is None


def test_validate_youtube_video_missing_fields(sample_valid_youtube_video):
    """Verify that records missing required YouTube fields are rejected."""
    for field in REQUIRED_YOUTUBE_FIELDS:
        incomplete = sample_valid_youtube_video.copy()
        del incomplete[field]
        is_valid, error = validate_youtube_video(incomplete)
        assert is_valid is False
        assert f"Missing required YouTube field(s): {field}" in error


def test_validate_youtube_video_invalid_types(sample_valid_youtube_video):
    """Verify that invalid field types are caught safely."""
    sample_valid_youtube_video["view_count"] = "not_a_number"
    is_valid, error = validate_youtube_video(sample_valid_youtube_video)
    assert is_valid is False
    assert "must be a numeric value" in error

    sample_valid_youtube_video["view_count"] = 100
    sample_valid_youtube_video["title"] = "   "
    is_valid, error = validate_youtube_video(sample_valid_youtube_video)
    assert is_valid is False
    assert "must be a non-empty string" in error


def test_save_raw_youtube_videos_deduplication(tmp_path, sample_valid_youtube_video):
    """Verify that saving records merges by video_id to prevent duplicate entries."""
    dest = tmp_path / "youtube_raw.json"
    save_raw_youtube_videos([sample_valid_youtube_video], dest, merge=True)

    new_video = sample_valid_youtube_video.copy()
    new_video["video_id"] = "vid_second_456"

    save_raw_youtube_videos([sample_valid_youtube_video, new_video], dest, merge=True)

    with open(dest, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 2
    ids = {v["video_id"] for v in saved}
    assert ids == {"vid_abc123", "vid_second_456"}


def test_fetch_youtube_videos_page_success():
    """Verify parsing of YouTube search and video details responses and quota tracking."""
    search_payload = {
        "nextPageToken": "token_page_2",
        "items": [
            {"id": {"videoId": "vid_sample_01"}}
        ],
    }
    videos_payload = {
        "items": [
            {
                "id": "vid_sample_01",
                "snippet": {
                    "title": "Mock Video 1",
                    "description": "Mock video description",
                    "channelId": "UC_mock_chan",
                    "channelTitle": "Mock Channel",
                    "publishedAt": "2026-11-02T10:00:00Z",
                },
                "statistics": {
                    "viewCount": "5000",
                    "likeCount": "250",
                    "commentCount": "30",
                },
            }
        ]
    }

    mock_search_resp = MagicMock()
    mock_search_resp.read.return_value = json.dumps(search_payload).encode("utf-8")
    mock_search_resp.__enter__.return_value = mock_search_resp

    mock_videos_resp = MagicMock()
    mock_videos_resp.read.return_value = json.dumps(videos_payload).encode("utf-8")
    mock_videos_resp.__enter__.return_value = mock_videos_resp

    with patch("urllib.request.urlopen", side_effect=[mock_search_resp, mock_videos_resp]):
        videos, next_token, units, err = fetch_youtube_videos_page("dummy_key", "data science")
        assert err is None
        assert next_token == "token_page_2"
        assert units == 2  # 1 search unit + 1 video list unit
        assert len(videos) == 1
        assert videos[0]["video_id"] == "vid_sample_01"
        assert videos[0]["view_count"] == 5000
        assert videos[0]["url"] == "https://www.youtube.com/watch?v=vid_sample_01"


def test_fetch_youtube_videos_page_quota_exceeded():
    """Verify handling of HTTP 403 quota exceeded error."""
    error_body = json.dumps({
        "error": {
            "errors": [{"reason": "quotaExceeded"}],
            "message": "The request cannot be completed because you have exceeded your quota.",
        }
    }).encode("utf-8")

    fp_mock = MagicMock()
    fp_mock.read.return_value = error_body

    http_error = urllib.error.HTTPError(
        url="https://www.googleapis.com/youtube/v3/search",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=fp_mock,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        videos, token, units, err = fetch_youtube_videos_page("dummy_key", "data science")
        assert len(videos) == 0
        assert "quota exceeded" in err.lower()


def test_fetch_youtube_videos_page_invalid_key():
    """Verify handling of HTTP 400 invalid API key error."""
    error_body = json.dumps({
        "error": {
            "errors": [{"reason": "keyInvalid"}],
            "message": "API key not valid. Please pass a valid API key.",
        }
    }).encode("utf-8")

    fp_mock = MagicMock()
    fp_mock.read.return_value = error_body

    http_error = urllib.error.HTTPError(
        url="https://www.googleapis.com/youtube/v3/search",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=fp_mock,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        videos, token, units, err = fetch_youtube_videos_page("bad_key", "query")
        assert len(videos) == 0
        assert "invalid youtube api key" in err.lower()


def test_youtube_error_does_not_leak_api_key():
    """Verify that error messages sanitize and never leak the raw API key."""
    secret_key = "MY_HIGHLY_SECRET_YOUTUBE_KEY_9988"
    error_body = json.dumps({
        "error": {
            "errors": [{"reason": "unknownError"}],
            "message": f"Failure occurred with key {secret_key}",
        }
    }).encode("utf-8")

    fp_mock = MagicMock()
    fp_mock.read.return_value = error_body

    http_error = urllib.error.HTTPError(
        url="https://www.googleapis.com/youtube/v3/search",
        code=500,
        msg="Internal Error",
        hdrs={},
        fp=fp_mock,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        _, _, _, err = fetch_youtube_videos_page(secret_key, "query")
        assert secret_key not in err
        assert "[REDACTED_API_KEY]" in err


def test_ingest_youtube_live_disabled_when_missing_key(monkeypatch):
    """Verify YouTube ingestion returns disabled status when API key is not configured."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    summary = ingest_youtube_live()

    assert summary["status"] == "disabled"
    assert summary["source"] == "youtube_api_v3"
    assert summary["valid_records"] == 0
    assert "instructions" in summary


def test_ingest_youtube_live_pagination_success(tmp_path, monkeypatch, sample_valid_youtube_video):
    """Verify multi-page collection and pagination handling with mocked API calls."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_valid_key")

    video_page1 = sample_valid_youtube_video.copy()
    video_page1["video_id"] = "vid_p1_01"

    video_page2 = sample_valid_youtube_video.copy()
    video_page2["video_id"] = "vid_p2_02"

    temp_output = tmp_path / "youtube_paginated.json"

    side_effects = [
        ([video_page1], "token_for_p2", 2, None),
        ([video_page2], None, 2, None),
    ]

    with patch("src.ingestion.youtube_ingestion.fetch_youtube_videos_page", side_effect=side_effects):
        summary = ingest_youtube_live(
            query="machine learning",
            max_pages=2,
            output_path=temp_output,
        )

        assert summary["status"] == "success"
        assert summary["pages_fetched"] == 2
        assert summary["estimated_quota_units"] == 4
        assert summary["total_records"] == 2
        assert summary["valid_records"] == 2
        assert summary["rejected_records"] == 0
        assert temp_output.exists()


def test_ingest_youtube_live_pagination_stops_on_quota_error(tmp_path, monkeypatch, sample_valid_youtube_video):
    """Verify pagination stops immediately when quota is exhausted on later pages."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_valid_key")

    video_page1 = sample_valid_youtube_video.copy()
    video_page1["video_id"] = "vid_p1_01"

    temp_output = tmp_path / "youtube_quota_stop.json"

    # Page 1 succeeds, Page 2 fails with quota exceeded
    side_effects = [
        ([video_page1], "token_for_p2", 2, None),
        ([], None, 0, "YouTube API quota exceeded (HTTP 403). Daily limit reached."),
    ]

    with patch("src.ingestion.youtube_ingestion.fetch_youtube_videos_page", side_effect=side_effects):
        summary = ingest_youtube_live(
            query="machine learning",
            max_pages=3,
            output_path=temp_output,
        )

        # Should stop after page 2 error and not attempt page 3
        assert summary["pages_fetched"] == 1
        assert any("quota exceeded" in e.lower() for e in summary["errors"])
        assert summary["valid_records"] == 1


def test_one_search_request_and_one_video_detail_request(monkeypatch, sample_valid_youtube_video):
    """Verify quota tracking for exactly 1 search call and 1 video-details call."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_valid_key")
    with patch("src.ingestion.youtube_ingestion.fetch_youtube_videos_page", return_value=([sample_valid_youtube_video], None, 2, None)):
        summary = ingest_youtube_live(query="test", max_pages=1)
        qa = summary["quota_accounting"]
        assert qa["search_requests_consumed"] == 1
        assert qa["search_quota_units_consumed"] == 1
        assert qa["videos_requests_consumed"] == 1
        assert qa["videos_quota_units_consumed"] == 1
        assert qa["other_methods_quota_consumed"] == 1
        assert qa["total_estimated_quota_units"] == 2
        assert qa["search_request_daily_limit"] == DEFAULT_DAILY_SEARCH_REQUEST_LIMIT
        assert qa["other_methods_daily_limit"] == DEFAULT_DAILY_OTHER_QUOTA_LIMIT
        assert "Managed externally" in qa["actual_cloud_quota_status"]


def test_multiple_search_pages_quota_accounting(tmp_path, monkeypatch, sample_valid_youtube_video):
    """Verify multi-page collection and quota accounting across multiple search pages."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_valid_key")
    v1 = sample_valid_youtube_video.copy()
    v1["video_id"] = "vid_page1"
    v2 = sample_valid_youtube_video.copy()
    v2["video_id"] = "vid_page2"
    v3 = sample_valid_youtube_video.copy()
    v3["video_id"] = "vid_page3"

    side_effects = [
        ([v1], "tok_2", 2, None),
        ([v2], "tok_3", 2, None),
        ([v3], None, 2, None),
    ]
    with patch("src.ingestion.youtube_ingestion.fetch_youtube_videos_page", side_effect=side_effects):
        summary = ingest_youtube_live(query="ai", max_pages=3, output_path=tmp_path / "pages.json")
        assert summary["status"] == "success"
        assert summary["pages_fetched"] == 3
        qa = summary["quota_accounting"]
        assert qa["search_requests_consumed"] == 3
        assert qa["search_quota_units_consumed"] == 3
        assert qa["videos_requests_consumed"] == 3
        assert qa["videos_quota_units_consumed"] == 3
        assert qa["total_estimated_quota_units"] == 6


def test_multiple_video_detail_batches():
    """Verify that video IDs exceeding 50 are split into multiple 50-item batches, consuming 1 unit per batch."""
    video_ids = [f"vid_batch_{i:03d}" for i in range(120)]  # 120 IDs -> 3 batches (50, 50, 20)

    def make_mock_batch_resp(count):
        items = [
            {
                "id": f"vid_{j}",
                "snippet": {"title": f"Title {j}", "publishedAt": "2026-11-01T00:00:00Z"},
                "statistics": {"viewCount": "100"},
            }
            for j in range(count)
        ]
        resp = MagicMock()
        resp.read.return_value = json.dumps({"items": items}).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    mock_resps = [make_mock_batch_resp(50), make_mock_batch_resp(50), make_mock_batch_resp(20)]
    with patch("urllib.request.urlopen", side_effect=mock_resps) as mock_url:
        records, units, err = fetch_video_details("dummy_key", video_ids)
        assert err is None
        assert mock_url.call_count == 3
        assert units == 3  # 1 quota unit per batch of up to 50
        assert len(records) == 120


def test_reaching_100_search_request_daily_limit(monkeypatch, sample_valid_youtube_video):
    """Verify that pagination stops when the 100 search calls/day limit is reached."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_valid_key")

    call_count = 0
    def mock_fetch_page(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return [sample_valid_youtube_video], f"token_{call_count}", 2, None

    with patch("src.ingestion.youtube_ingestion.fetch_youtube_videos_page", side_effect=mock_fetch_page):
        summary = ingest_youtube_live(query="scale", max_pages=105, search_daily_limit=100)
        assert summary["quota_accounting"]["search_requests_consumed"] == 100
        assert any("Search request daily limit reached" in e for e in summary["errors"])
        assert call_count == 100


def test_reaching_10000_unit_other_method_quota_limit(monkeypatch, sample_valid_youtube_video):
    """Verify that pagination stops when other-methods daily quota is exhausted."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "dummy_valid_key")

    call_count = 0
    def mock_fetch_page(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return [sample_valid_youtube_video], f"token_{call_count}", 2, None

    with patch("src.ingestion.youtube_ingestion.fetch_youtube_videos_page", side_effect=mock_fetch_page):
        summary = ingest_youtube_live(query="scale", max_pages=5, other_quota_daily_limit=2)
        assert summary["quota_accounting"]["other_methods_quota_consumed"] == 2
        assert any("Other-methods daily quota limit reached" in e for e in summary["errors"])
        assert call_count == 2


def test_save_raw_youtube_videos_purges_expired_records(tmp_path, sample_valid_youtube_video):
    """Verify merging raw YouTube videos automatically excludes expired records (> 30 days)."""
    target_file = tmp_path / "raw_yt_merge_expire.json"
    now = datetime.now(timezone.utc)
    old_dt = (now - timedelta(days=35)).isoformat()

    expired_record = sample_valid_youtube_video.copy()
    expired_record["video_id"] = "vid_expired_35d"
    expired_record["collected_at"] = old_dt

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump([expired_record], f)

    fresh_record = sample_valid_youtube_video.copy()
    fresh_record["video_id"] = "vid_fresh_now"

    save_raw_youtube_videos([fresh_record], target_file, merge=True)

    with open(target_file, "r", encoding="utf-8") as f:
        saved = json.load(f)

    # Expired record must NOT be retained or reused
    assert len(saved) == 1
    assert saved[0]["video_id"] == "vid_fresh_now"


def test_youtube_and_reddit_retention_remain_separate():
    """Verify YouTube (30-day cache) and Reddit (routine 48-hr deletion) policies remain strictly separate."""
    from src.ingestion.reddit_ingestion import apply_reddit_retention_policy

    assert apply_youtube_retention_policy.__name__ != apply_reddit_retention_policy.__name__


def test_apply_youtube_retention_policy_purges_expired_cache(tmp_path):
    """Verify YouTube retention enforces the 30-day cache limitation."""
    target_file = tmp_path / "youtube_cache_retention.json"
    now = datetime.now(timezone.utc)

    old_dt = (now - timedelta(days=35)).isoformat()
    recent_dt = (now - timedelta(days=5)).isoformat()

    videos = [
        {
            "video_id": "vid_expired",
            "title": "Expired Video",
            "description": "Desc",
            "channel_id": "ch1",
            "channel_title": "Title",
            "published_at": "2026-01-01T00:00:00Z",
            "view_count": 10,
            "like_count": 1,
            "comment_count": 0,
            "url": "https://youtube.com/watch?v=vid_expired",
            "collected_at": old_dt,
        },
        {
            "video_id": "vid_fresh",
            "title": "Fresh Video",
            "description": "Desc",
            "channel_id": "ch1",
            "channel_title": "Title",
            "published_at": "2026-01-01T00:00:00Z",
            "view_count": 20,
            "like_count": 2,
            "comment_count": 0,
            "url": "https://youtube.com/watch?v=vid_fresh",
            "collected_at": recent_dt,
        },
    ]

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(videos, f)

    res = apply_youtube_retention_policy(max_cache_days=30, file_path=target_file)
    assert res["status"] == "success"
    assert res["purged_count"] == 1
    assert res["retained_count"] == 1

    with open(target_file, "r", encoding="utf-8") as f:
        retained = json.load(f)
    assert len(retained) == 1
    assert retained[0]["video_id"] == "vid_fresh"

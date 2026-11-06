"""Unit tests for the Reddit data ingestion module (Mock & Live API)."""

import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.ingestion.reddit_ingestion import (
    validate_reddit_post,
    validate_reddit_user_agent,
    load_sample_posts,
    save_raw_posts,
    ingest_reddit_posts,
    ingest_reddit_live,
    fetch_reddit_oauth_token,
    fetch_subreddit_posts,
    anonymize_reddit_author,
    purge_reddit_posts,
    reconcile_reddit_deletions,
    apply_reddit_retention_policy,
    REQUIRED_FIELDS,
)
from config.settings import BASE_DIR


@pytest.fixture
def sample_valid_post() -> dict:
    """Returns a single valid Reddit post fixture."""
    return {
        "post_id": "t3_test01",
        "subreddit": "datascience",
        "title": "Sample Discussion on NLP",
        "text": "This is sample text for unit testing ingestion pipelines.",
        "author": "tester_dev",
        "created_utc": 1730000000,
        "score": 100,
        "num_comments": 15,
        "url": "https://reddit.com/r/datascience/comments/test01",
    }


def test_validate_reddit_post_valid(sample_valid_post):
    """Verify that a complete and correctly typed post passes validation."""
    is_valid, error = validate_reddit_post(sample_valid_post)
    assert is_valid is True
    assert error is None


def test_validate_reddit_post_missing_field(sample_valid_post):
    """Verify that records missing any required field are rejected."""
    for field in REQUIRED_FIELDS:
        incomplete_post = sample_valid_post.copy()
        del incomplete_post[field]
        is_valid, error = validate_reddit_post(incomplete_post)
        assert is_valid is False
        assert f"Missing required field(s): {field}" in error


def test_validate_reddit_post_empty_string_field(sample_valid_post):
    """Verify that empty or whitespace-only required string fields are rejected."""
    sample_valid_post["title"] = "   "
    is_valid, error = validate_reddit_post(sample_valid_post)
    assert is_valid is False
    assert "must be a non-empty string" in error


def test_validate_reddit_post_invalid_types(sample_valid_post):
    """Verify that invalid field types are rejected safely."""
    sample_valid_post["score"] = "not_a_number"
    is_valid, error = validate_reddit_post(sample_valid_post)
    assert is_valid is False
    assert "must be a numeric value" in error

    is_valid, error = validate_reddit_post(["not", "a", "dict"])
    assert is_valid is False
    assert "must be a dictionary object" in error


def test_load_sample_posts_success(tmp_path, sample_valid_post):
    """Verify loading posts from a valid JSON file."""
    test_file = tmp_path / "valid_posts.json"
    test_file.write_text(json.dumps([sample_valid_post]), encoding="utf-8")

    loaded = load_sample_posts(test_file)
    assert isinstance(loaded, list)
    assert len(loaded) == 1
    assert loaded[0]["post_id"] == "t3_test01"


def test_load_sample_posts_missing_file(tmp_path):
    """Verify FileNotFoundError is raised when file does not exist."""
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_sample_posts(missing_file)


def test_load_sample_posts_invalid_json(tmp_path):
    """Verify json.JSONDecodeError is raised for corrupted JSON."""
    bad_json_file = tmp_path / "malformed.json"
    bad_json_file.write_text("{this is not valid json:", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_sample_posts(bad_json_file)


def test_load_sample_posts_non_list_root(tmp_path):
    """Verify ValueError is raised if root JSON element is not a list."""
    obj_file = tmp_path / "object.json"
    obj_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a JSON list"):
        load_sample_posts(obj_file)


def test_save_raw_posts(tmp_path, sample_valid_post):
    """Verify saving records to a target destination in a temporary directory."""
    destination = tmp_path / "nested" / "raw_posts.json"
    saved_path = save_raw_posts([sample_valid_post], destination)

    assert saved_path.exists()
    assert destination.exists()

    with open(destination, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["post_id"] == sample_valid_post["post_id"]


def test_save_raw_posts_deduplication_merge(tmp_path, sample_valid_post):
    """Verify saving with merge=True deduplicates records by post_id."""
    destination = tmp_path / "posts.json"
    save_raw_posts([sample_valid_post], destination, merge=True)

    # Save again with same post + a new post
    new_post = sample_valid_post.copy()
    new_post["post_id"] = "t3_new02"

    save_raw_posts([sample_valid_post, new_post], destination, merge=True)

    with open(destination, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 2
    ids = {p["post_id"] for p in saved}
    assert ids == {"t3_test01", "t3_new02"}


def test_ingest_reddit_posts_summary_and_filtering(tmp_path, sample_valid_post):
    """Verify ingestion pipeline filters invalid records and outputs accurate summary."""
    invalid_post = sample_valid_post.copy()
    invalid_post["post_id"] = "t3_invalid"
    del invalid_post["author"]

    test_source = tmp_path / "test_source.json"
    test_source.write_text(json.dumps([sample_valid_post, invalid_post]), encoding="utf-8")

    output_dest = tmp_path / "output" / "ingested.json"
    summary = ingest_reddit_posts(source_path=test_source, output_path=output_dest)

    assert summary["status"] == "success"
    assert summary["total_records"] == 2
    assert summary["valid_records"] == 1
    assert summary["rejected_records"] == 1
    assert len(summary["errors"]) == 1
    assert "Missing required field(s): author" in summary["errors"][0]
    assert Path(summary["output_file"]).exists()


def test_ingest_reddit_posts_missing_source(tmp_path):
    """Verify error summary when input file is not found."""
    missing_source = tmp_path / "missing.json"
    summary = ingest_reddit_posts(source_path=missing_source, output_path=tmp_path / "out.json")

    assert summary["status"] == "error"
    assert summary["total_records"] == 0
    assert summary["valid_records"] == 0
    assert summary["output_file"] is None


def test_ingest_actual_sample_dataset_into_temp_dir(tmp_path):
    """Verify actual project sample dataset loads and validates cleanly."""
    actual_sample_file = BASE_DIR / "data" / "sample_reddit_posts.json"
    temp_output = tmp_path / "test_raw.json"

    summary = ingest_reddit_posts(source_path=actual_sample_file, output_path=temp_output)

    assert summary["status"] == "success"
    assert summary["total_records"] == 10
    assert summary["valid_records"] == 10
    assert summary["rejected_records"] == 0
    assert summary["errors"] == []


# -----------------------------------------------------------------------------
# Tests for Live Reddit API Client (Mocked)
# -----------------------------------------------------------------------------

def test_fetch_reddit_oauth_token_success():
    """Verify successful parsing of mock OAuth token response."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"access_token": "mock_token_xyz"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        token, err = fetch_reddit_oauth_token("mock_client", "mock_secret", "test_ua")
        assert token == "mock_token_xyz"
        assert err is None


def test_fetch_reddit_oauth_token_auth_failure():
    """Verify handling of HTTP 401 unauthorized error."""
    http_error = urllib.error.HTTPError(
        url="https://www.reddit.com/api/v1/access_token",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        token, err = fetch_reddit_oauth_token("bad_client", "bad_secret", "test_ua")
        assert token is None
        assert "Authentication failed" in err


def test_fetch_reddit_oauth_token_rate_limited():
    """Verify handling of HTTP 429 rate limit error."""
    http_error = urllib.error.HTTPError(
        url="https://www.reddit.com/api/v1/access_token",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        token, err = fetch_reddit_oauth_token("mock_client", "mock_secret", "test_ua")
        assert token is None
        assert "rate limit exceeded" in err


def test_fetch_subreddit_posts_success():
    """Verify parsing of mocked Reddit listing response and rate limit headers."""
    mock_payload = {
        "data": {
            "children": [
                {
                    "data": {
                        "name": "t3_post123",
                        "subreddit": "datascience",
                        "title": "Live API Test Post",
                        "selftext": "Content of the live test post.",
                        "author": "mock_author",
                        "created_utc": 1730005000,
                        "score": 42,
                        "num_comments": 8,
                        "permalink": "/r/datascience/comments/post123/live_test/",
                    }
                }
            ]
        }
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.headers = {
        "X-Ratelimit-Remaining": "590.0",
        "X-Ratelimit-Reset": "300",
        "X-Ratelimit-Used": "10",
    }
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        posts, limits, err = fetch_subreddit_posts("datascience", "mock_token", "test_ua")
        assert err is None
        assert len(posts) == 1
        assert posts[0]["post_id"] == "t3_post123"
        assert posts[0]["title"] == "Live API Test Post"
        assert limits["remaining"] == "590.0"


def test_fetch_subreddit_posts_rate_limited():
    """Verify handling of rate-limit 429 when requesting subreddit listing."""
    http_error = urllib.error.HTTPError(
        url="https://oauth.reddit.com/r/datascience/hot",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        posts, limits, err = fetch_subreddit_posts("datascience", "mock_token", "test_ua")
        assert len(posts) == 0
        assert "rate limit exceeded" in err


def test_ingest_reddit_live_disabled_when_missing_credentials(monkeypatch):
    """Verify live Reddit ingestion returns disabled status when credentials are not configured."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "")

    summary = ingest_reddit_live()
    assert summary["status"] == "disabled"
    assert summary["source"] == "reddit_live_api"
    assert "instructions" in summary
    assert summary["valid_records"] == 0


def test_ingest_reddit_live_success_with_mocked_api(tmp_path, monkeypatch):
    """Verify live Reddit ingestion pipeline succeeds when API calls are mocked."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "dummy_client_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "social-media-intelligence:v1.0 (by /u/tester)")

    mock_post = {
        "post_id": "t3_live01",
        "subreddit": "technology",
        "title": "Mock Live Post Title",
        "text": "Live post body text.",
        "author": "mock_live_author",
        "created_utc": 1730006000,
        "score": 55,
        "num_comments": 12,
        "url": "https://reddit.com/r/technology/comments/live01",
    }

    temp_output = tmp_path / "reddit_live.json"

    with patch("src.ingestion.reddit_ingestion.fetch_reddit_oauth_token", return_value=("mock_token", None)):
        with patch("src.ingestion.reddit_ingestion.fetch_subreddit_posts", return_value=([mock_post], {"remaining": "500"}, None)):
            summary = ingest_reddit_live(
                subreddits=["technology"],
                output_path=temp_output,
            )

            assert summary["status"] == "success"
            assert summary["total_records"] == 1
            assert summary["valid_records"] == 1
            assert summary["rejected_records"] == 0
            assert temp_output.exists()


def test_validate_reddit_user_agent():
    """Verify validation of User-Agent format according to Reddit standards."""
    valid_ua = "social-media-platform:v1.0 (by /u/researcher)"
    is_valid, err = validate_reddit_user_agent(valid_ua)
    assert is_valid is True
    assert err is None

    # Placeholder username
    is_valid, err = validate_reddit_user_agent("app:v1.0 (by /u/your_username)")
    assert is_valid is False
    assert "placeholder" in err.lower()

    # Missing author identity
    is_valid, err = validate_reddit_user_agent("generic_app_name")
    assert is_valid is False
    assert "format" in err.lower()

    # Empty
    is_valid, err = validate_reddit_user_agent("   ")
    assert is_valid is False


def test_ingest_reddit_live_halts_on_rate_limit(tmp_path, monkeypatch):
    """Verify ingestion stops iterating through subreddits if rate-limit error occurs."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "dummy_client_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "app:v1.0 (by /u/testuser)")

    temp_output = tmp_path / "reddit_halt.json"

    # First call returns rate limit error
    with patch("src.ingestion.reddit_ingestion.fetch_reddit_oauth_token", return_value=("mock_token", None)):
        with patch("src.ingestion.reddit_ingestion.fetch_subreddit_posts", return_value=([], {"remaining": "0"}, "Reddit API rate limit exceeded (HTTP 429).")) as mock_fetch:
            summary = ingest_reddit_live(
                subreddits=["datascience", "MachineLearning", "technology"],
                output_path=temp_output,
            )
            # Must stop after first subreddit and not call the remaining two
            assert mock_fetch.call_count == 1
            assert any("halting" in e.lower() for e in summary["errors"])


def test_ingest_reddit_live_halts_when_remaining_quota_zero(tmp_path, monkeypatch):
    """Verify ingestion stops when X-Ratelimit-Remaining is exhausted."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "dummy_client_secret")
    monkeypatch.setenv("REDDIT_USER_AGENT", "app:v1.0 (by /u/testuser)")

    mock_post = {
        "post_id": "t3_post_quota",
        "subreddit": "datascience",
        "title": "Post before quota exhaustion",
        "text": "Text",
        "author": "author",
        "created_utc": 1730007000,
        "score": 10,
        "num_comments": 2,
        "url": "https://reddit.com/r/datascience/post_quota",
    }
    temp_output = tmp_path / "reddit_quota_stop.json"

    # Returns post but indicates remaining quota is 0
    with patch("src.ingestion.reddit_ingestion.fetch_reddit_oauth_token", return_value=("mock_token", None)):
        with patch("src.ingestion.reddit_ingestion.fetch_subreddit_posts", return_value=([mock_post], {"remaining": "0.0", "reset_seconds": "300"}, None)) as mock_fetch:
            summary = ingest_reddit_live(
                subreddits=["datascience", "technology"],
                output_path=temp_output,
            )
            # Halts before calling technology
            assert mock_fetch.call_count == 1
            assert any("exhausted" in e.lower() for e in summary["errors"])
            assert summary["valid_records"] == 1


# -----------------------------------------------------------------------------
# Tests for Reddit Data Retention and Deletion Compliance
# -----------------------------------------------------------------------------

def test_save_raw_posts_stamps_collected_at(tmp_path, sample_valid_post):
    """Verify that save_raw_posts stamps 'collected_at' without altering post text."""
    target_file = tmp_path / "raw_reddit_timestamp.json"
    assert "collected_at" not in sample_valid_post
    original_text = sample_valid_post["text"]

    save_raw_posts([sample_valid_post], target_file)

    with open(target_file, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert len(saved) == 1
    assert "collected_at" in saved[0]
    assert saved[0]["text"] == original_text
    # Verify ISO timestamp format
    dt = datetime.fromisoformat(saved[0]["collected_at"])
    assert dt.tzinfo is not None


def test_anonymize_reddit_author(tmp_path, sample_valid_post):
    """Verify author-identifying info is replaced with '[deleted]' while keeping post text."""
    target_file = tmp_path / "raw_reddit_anon.json"

    post_a = sample_valid_post.copy()
    post_a["post_id"] = "t3_post_a"
    post_a["author"] = "UserTarget"

    post_b = sample_valid_post.copy()
    post_b["post_id"] = "t3_post_b"
    post_b["author"] = "OtherUser"

    save_raw_posts([post_a, post_b], target_file)

    res = anonymize_reddit_author("usertarget", file_path=target_file)
    assert res["status"] == "success"
    assert res["anonymized_count"] == 1

    with open(target_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert records[0]["author"] == "[deleted]"
    assert records[0]["text"] == post_a["text"]  # Unmodified text
    assert records[1]["author"] == "OtherUser"


def test_purge_reddit_posts(tmp_path, sample_valid_post):
    """Verify purging of specific Reddit posts by ID."""
    target_file = tmp_path / "raw_reddit_purge.json"

    post_1 = sample_valid_post.copy()
    post_1["post_id"] = "t3_keep_me"

    post_2 = sample_valid_post.copy()
    post_2["post_id"] = "t3_purge_me"

    save_raw_posts([post_1, post_2], target_file)

    # Purge using bare id without t3_ prefix to test prefix normalization
    res = purge_reddit_posts(["purge_me"], file_path=target_file)
    assert res["status"] == "success"
    assert res["purged_count"] == 1
    assert res["remaining_count"] == 1

    with open(target_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 1
    assert records[0]["post_id"] == "t3_keep_me"


def test_reconcile_reddit_deletions_detects_and_purges_deleted(tmp_path):
    """Verify reconciliation queries Reddit API and purges deleted/removed posts."""
    target_file = tmp_path / "raw_reddit_reconcile.json"

    posts = [
        {
            "post_id": "t3_active1",
            "subreddit": "datascience",
            "title": "Active Post",
            "text": "Still active.",
            "author": "active_user",
            "created_utc": 1730000000,
            "score": 10,
            "num_comments": 1,
            "url": "https://reddit.com/active1",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "post_id": "t3_author_deleted",
            "subreddit": "datascience",
            "title": "Post With Deleted Author",
            "text": "Content",
            "author": "old_user",
            "created_utc": 1730000000,
            "score": 5,
            "num_comments": 0,
            "url": "https://reddit.com/del1",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "post_id": "t3_fully_gone",
            "subreddit": "datascience",
            "title": "Gone Post",
            "text": "Gone",
            "author": "gone_user",
            "created_utc": 1730000000,
            "score": 2,
            "num_comments": 0,
            "url": "https://reddit.com/del2",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        },
    ]

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(posts, f)

    # Mock Reddit API /by_id/ response:
    # - t3_active1: active
    # - t3_author_deleted: returned but author="[deleted]"
    # - t3_fully_gone: omitted from response (HTTP 200 with only active children)
    mock_payload = {
        "data": {
            "children": [
                {
                    "data": {
                        "name": "t3_active1",
                        "author": "active_user",
                        "title": "Active Post",
                        "selftext": "Still active.",
                    }
                },
                {
                    "data": {
                        "name": "t3_author_deleted",
                        "author": "[deleted]",
                        "title": "Post With Deleted Author",
                        "selftext": "Content",
                    }
                },
            ]
        }
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = reconcile_reddit_deletions(
            access_token="mock_token",
            user_agent="app:v1.0 (by /u/test)",
            file_path=target_file,
        )

    assert res["status"] == "success"
    assert res["checked_count"] == 3
    assert res["deleted_detected_count"] == 2
    assert "t3_author_deleted" in res["deleted_post_ids"]
    assert "t3_fully_gone" in res["deleted_post_ids"]
    assert res["remaining_count"] == 1
    assert "cannot be detected automatically" in res["note"]

    with open(target_file, "r", encoding="utf-8") as f:
        retained = json.load(f)
    assert len(retained) == 1
    assert retained[0]["post_id"] == "t3_active1"


def test_apply_reddit_retention_policy_default_48_hours(tmp_path):
    """Verify default routine retention enforces 48-hour deletion, not 90 days."""
    target_file = tmp_path / "raw_reddit_retention_48h.json"
    now = datetime.now(timezone.utc)

    stale_50h = (now - timedelta(hours=50)).isoformat()
    fresh_10h = (now - timedelta(hours=10)).isoformat()

    records = [
        {
            "post_id": "t3_stale_50h",
            "subreddit": "datascience",
            "title": "Stale 50 Hours Ago Post",
            "text": "Stale body text.",
            "author": "author_stale",
            "created_utc": 1720000000,
            "score": 10,
            "num_comments": 1,
            "url": "https://reddit.com/stale",
            "collected_at": stale_50h,
        },
        {
            "post_id": "t3_fresh_10h",
            "subreddit": "datascience",
            "title": "Fresh 10 Hours Ago Post",
            "text": "Fresh body text.",
            "author": "author_fresh",
            "created_utc": 1730000000,
            "score": 20,
            "num_comments": 2,
            "url": "https://reddit.com/fresh",
            "collected_at": fresh_10h,
        },
    ]

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(records, f)

    # Call with default max_age_hours (48 hours)
    res = apply_reddit_retention_policy(file_path=target_file)
    assert res["status"] == "success"
    assert res["max_age_hours"] == 48.0
    assert res["purged_count"] == 1
    assert res["retained_count"] == 1
    assert "48 hours" in res["policy"]

    with open(target_file, "r", encoding="utf-8") as f:
        retained = json.load(f)
    assert len(retained) == 1
    assert retained[0]["post_id"] == "t3_fresh_10h"


def test_reddit_retention_does_not_assume_90_days(tmp_path):
    """Verify that a 3-day (72-hour) old post is purged under default policy, proving 90 days is not assumed."""
    target_file = tmp_path / "raw_reddit_no_90d.json"
    now = datetime.now(timezone.utc)
    dt_72h = (now - timedelta(hours=72)).isoformat()

    record_3days = {
        "post_id": "t3_3days_old",
        "subreddit": "datascience",
        "title": "3 Days Old Post",
        "text": "Body",
        "author": "user3d",
        "created_utc": 1730000000,
        "score": 5,
        "num_comments": 1,
        "url": "https://reddit.com/3d",
        "collected_at": dt_72h,
    }

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump([record_3days], f)

    res = apply_reddit_retention_policy(file_path=target_file)
    # Under 48h policy, 72h post must be purged (would have been retained under 90d assumption)
    assert res["purged_count"] == 1
    assert res["retained_count"] == 0

    with open(target_file, "r", encoding="utf-8") as f:
        retained = json.load(f)
    assert len(retained) == 0


def test_reconcile_reddit_deletions_ensures_deleted_records_not_retained(tmp_path):
    """Verify that deleted records identified via API check are permanently removed and not reusable."""
    target_file = tmp_path / "raw_reddit_not_retained.json"

    posts = [
        {
            "post_id": "t3_gone_forever",
            "subreddit": "datascience",
            "title": "Deleted Post",
            "text": "To be removed",
            "author": "author_del",
            "created_utc": 1730000000,
            "score": 1,
            "num_comments": 0,
            "url": "https://reddit.com/gone",
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
    ]

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(posts, f)

    # Empty children response signals post is deleted/removed from Reddit
    mock_payload = {"data": {"children": []}}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = reconcile_reddit_deletions(
            access_token="mock_token",
            user_agent="app:v1.0 (by /u/tester)",
            file_path=target_file,
        )

    assert res["status"] == "success"
    assert res["deleted_detected_count"] == 1
    assert res["purged_count"] == 1
    assert res["remaining_count"] == 0

    # Ensure disk storage contains 0 records (deleted records cannot be reused)
    with open(target_file, "r", encoding="utf-8") as f:
        stored = json.load(f)
    assert stored == []

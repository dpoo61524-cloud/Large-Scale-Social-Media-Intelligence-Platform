"""YouTube Data API v3 Ingestion Module (Company Step 1).

Provides modular, rate-limit and quota-aware data collection for YouTube videos
relevant to the social media intelligence platform.

Quota Accounting Rules (Official Google Developers Documentation):
- Ref: https://developers.google.com/youtube/data/determine_quota_cost
- search.list costs 1 quota unit per call.
- search.list has a separate default limit of 100 calls per day.
- Each additional search page consumes another search request from that separate limit.
- videos.list costs 1 quota unit per call from the other daily quota allocation (batches of up to 50 video IDs).
- The default allocation for other methods is 10,000 units per day.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from config.settings import load_config, _is_configured

# Official YouTube Data API v3 Quota Costs & Daily Allocations
# Ref: https://developers.google.com/youtube/data/determine_quota_cost
QUOTA_COST_SEARCH_LIST = 1              # 1 quota unit per search.list request
QUOTA_COST_VIDEOS_LIST = 1              # 1 quota unit per videos.list request
DEFAULT_DAILY_SEARCH_REQUEST_LIMIT = 100  # Separate default limit: 100 search calls per day
DEFAULT_DAILY_OTHER_QUOTA_LIMIT = 10000   # Default daily allocation for other methods: 10,000 units per day

# Required schema fields for a collected YouTube video
REQUIRED_YOUTUBE_FIELDS = (
    "video_id",
    "title",
    "description",
    "channel_id",
    "channel_title",
    "published_at",
    "view_count",
    "like_count",
    "comment_count",
    "url",
)


def validate_youtube_video(video: dict[str, Any]) -> tuple[bool, str | None]:
    """Validates that a video record contains required metadata fields and data types."""
    if not isinstance(video, dict):
        return False, "Record must be a dictionary object."

    missing = [f for f in REQUIRED_YOUTUBE_FIELDS if f not in video]
    if missing:
        return False, f"Missing required YouTube field(s): {', '.join(missing)}"

    for str_field in ("video_id", "title", "channel_id", "channel_title", "published_at", "url"):
        val = video[str_field]
        if not isinstance(val, str) or not val.strip():
            return False, f"Field '{str_field}' must be a non-empty string."

    if not isinstance(video["description"], str):
        return False, "Field 'description' must be a string."

    for num_field in ("view_count", "like_count", "comment_count"):
        val = video[num_field]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return False, f"Field '{num_field}' must be a numeric value."

    return True, None


def save_raw_youtube_videos(
    videos: list[dict[str, Any]],
    output_path: Path | str,
    merge: bool = True,
) -> Path:
    """Saves valid YouTube records to a raw JSON destination.

    Merges by video_id when merge=True to prevent accidental overwrites.
    Tracks collected_at timestamp on each record for retention compliance.
    Purges expired cache records (> 30 days) to prevent accidental reuse.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    current_timestamp = datetime.now(timezone.utc).isoformat()
    cutoff_30d = datetime.now(timezone.utc) - timedelta(days=30)

    for v in videos:
        if "collected_at" not in v:
            v["collected_at"] = current_timestamp

    existing_videos: list[dict[str, Any]] = []
    if merge and destination.exists():
        try:
            with open(destination, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing_videos = content
        except (json.JSONDecodeError, OSError):
            existing_videos = []

    combined_by_id = {}
    for v in existing_videos:
        if isinstance(v, dict) and "video_id" in v:
            # Purge expired cache records (> 30 days old) to prevent accidental reuse
            col_str = v.get("collected_at")
            if col_str:
                try:
                    if datetime.fromisoformat(col_str) < cutoff_30d:
                        continue
                except (ValueError, TypeError):
                    pass
            combined_by_id[v["video_id"]] = v

    for v in videos:
        if isinstance(v, dict) and "video_id" in v:
            combined_by_id[v["video_id"]] = v

    merged_list = list(combined_by_id.values()) if combined_by_id else videos

    with open(destination, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)

    return destination


def _sanitize_error(msg: str, secret_key: str | None) -> str:
    """Ensures API keys are never exposed in error messages or logs."""
    if secret_key and secret_key in msg:
        return msg.replace(secret_key, "[REDACTED_API_KEY]")
    return msg


def _safe_int(value: Any, default: int = 0) -> int:
    """Converts a value safely to integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fetch_video_details(
    api_key: str,
    video_ids: list[str],
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Fetches video snippet and statistics for video_ids in batches of up to 50.

    Each batch call to videos.list consumes 1 quota unit from the other-methods allocation (10,000 units/day).
    Returns:
        (records, other_methods_quota_units_consumed, error_message).
    """
    if not video_ids:
        return [], 0, None

    records: list[dict[str, Any]] = []
    quota_units = 0
    batch_size = 50
    current_timestamp = datetime.now(timezone.utc).isoformat()

    for i in range(0, len(video_ids), batch_size):
        chunk = video_ids[i : i + batch_size]
        videos_params = {
            "part": "snippet,statistics",
            "id": ",".join(chunk),
            "key": api_key,
        }
        videos_url = f"https://www.googleapis.com/youtube/v3/videos?{urllib.parse.urlencode(videos_params)}"
        req_videos = urllib.request.Request(videos_url, method="GET")

        try:
            with urllib.request.urlopen(req_videos, timeout=15) as resp:
                videos_data = json.loads(resp.read().decode("utf-8"))
                quota_units += QUOTA_COST_VIDEOS_LIST  # 1 unit per batch
        except urllib.error.HTTPError as err:
            return records, quota_units, _sanitize_error(f"YouTube API details error (HTTP {err.code}): {err.reason}", api_key)
        except Exception as err:
            return records, quota_units, _sanitize_error(f"Failed to retrieve video statistics: {err}", api_key)

        for item in videos_data.get("items", []):
            vid = item.get("id", "")
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            records.append({
                "video_id": vid,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "collected_at": current_timestamp,
                "view_count": _safe_int(stats.get("viewCount", 0)),
                "like_count": _safe_int(stats.get("likeCount", 0)),
                "comment_count": _safe_int(stats.get("commentCount", 0)),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })

    return records, quota_units, None


def fetch_youtube_videos_page(
    api_key: str,
    query: str,
    max_results: int = 25,
    page_token: str | None = None,
) -> tuple[list[dict[str, Any]], str | None, int, str | None]:
    """Fetches a single page of video search results and statistics from YouTube Data API v3.

    Cost accounting:
    - search.list costs 1 quota unit (tracked against separate limit of 100 search requests/day).
    - videos.list costs 1 quota unit per batch of up to 50 videos (from 10,000 other-methods units/day).

    Returns:
        (videos_list, next_page_token, quota_units_consumed, error_message).
    """
    quota_units = 0

    search_params = {
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": min(max(1, max_results), 50),
        "key": api_key,
    }
    if page_token:
        search_params["pageToken"] = page_token

    search_url = f"https://www.googleapis.com/youtube/v3/search?{urllib.parse.urlencode(search_params)}"
    req_search = urllib.request.Request(search_url, method="GET")

    try:
        with urllib.request.urlopen(req_search, timeout=15) as resp:
            search_data = json.loads(resp.read().decode("utf-8"))
            quota_units += QUOTA_COST_SEARCH_LIST  # 1 quota unit per search request
    except urllib.error.HTTPError as err:
        try:
            error_body = json.loads(err.read().decode("utf-8"))
            reasons = [e.get("reason", "") for e in error_body.get("error", {}).get("errors", [])]
            if "quotaExceeded" in reasons or "dailyLimitExceeded" in reasons:
                return [], None, quota_units, "YouTube API quota exceeded (HTTP 403). Daily limit reached."
            if "keyInvalid" in reasons or err.code == 400:
                return [], None, quota_units, "Invalid YouTube API key (HTTP 400/403)."
            raw_msg = error_body.get("error", {}).get("message", err.reason)
            return [], None, quota_units, _sanitize_error(f"YouTube API HTTP error {err.code}: {raw_msg}", api_key)
        except Exception:
            return [], None, quota_units, _sanitize_error(f"YouTube API HTTP error {err.code}: {err.reason}", api_key)
    except urllib.error.URLError as err:
        return [], None, quota_units, _sanitize_error(f"Network error connecting to YouTube API: {err.reason}", api_key)
    except Exception as err:
        return [], None, quota_units, _sanitize_error(f"Unexpected error querying YouTube API: {err}", api_key)

    next_token = search_data.get("nextPageToken")
    items = search_data.get("items", [])
    video_ids = [item["id"]["videoId"] for item in items if "id" in item and "videoId" in item["id"]]

    if not video_ids:
        return [], next_token, quota_units, None

    records, details_units, details_err = fetch_video_details(api_key, video_ids)
    quota_units += details_units

    return records, next_token, quota_units, details_err


def apply_youtube_retention_policy(
    max_cache_days: int = 30,
    output_path: Path | str | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Applies YouTube Data API compliance rule: cached metadata must not exceed 30 days.

    Ref: YouTube API Developer Terms of Service (Section III.E)
    Purges any video metadata older than max_cache_days (default 30 days) so that
    expired records cannot be retained or reused.
    """
    config = load_config()
    chosen_path = file_path or output_path
    target_path = Path(chosen_path) if chosen_path else config.raw_data_path / "youtube_videos_raw.json"

    if not target_path.exists():
        return {
            "status": "no_data",
            "source": "youtube_retention",
            "purged_count": 0,
            "retained_count": 0,
            "message": "No YouTube raw data file found.",
        }

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            videos = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "youtube_retention",
            "message": f"Failed reading file: {err}",
            "purged_count": 0,
            "retained_count": 0,
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_cache_days)
    retained: list[dict[str, Any]] = []
    purged_count = 0

    for video in videos:
        collected_str = video.get("collected_at")
        if collected_str:
            try:
                collected_dt = datetime.fromisoformat(collected_str)
                if collected_dt < cutoff:
                    purged_count += 1
                    continue
            except (ValueError, TypeError):
                pass
        retained.append(video)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(retained, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "source": "youtube_retention",
        "max_cache_days": max_cache_days,
        "purged_count": purged_count,
        "retained_count": len(retained),
        "file_path": str(target_path),
    }


def ingest_youtube_live(
    query: str = "social media intelligence machine learning",
    max_pages: int = 1,
    max_results_per_page: int = 25,
    output_path: Path | str | None = None,
    merge: bool = True,
    search_daily_limit: int = DEFAULT_DAILY_SEARCH_REQUEST_LIMIT,
    other_quota_daily_limit: int = DEFAULT_DAILY_OTHER_QUOTA_LIMIT,
) -> dict[str, Any]:
    """Orchestrates video collection from YouTube Data API v3 with quota tracking.

    Accounting Model:
    - search.list costs 1 quota unit per call, tracked against separate limit of 100 calls/day.
    - videos.list costs 1 quota unit per batch of up to 50 videos against 10,000 units/day allocation.
    - Halts pagination when the search request limit or other-methods quota limit is reached.
    """
    config = load_config()
    api_key = config.youtube_api_key

    if not _is_configured(api_key):
        return {
            "status": "disabled",
            "source": "youtube_api_v3",
            "message": "YouTube API key not configured in .env.",
            "instructions": "Set YOUTUBE_API_KEY in your .env file to enable live YouTube video ingestion.",
            "query": query,
            "pages_fetched": 0,
            "estimated_quota_units": 0,
            "quota_accounting": {
                "search_requests_consumed": 0,
                "search_request_daily_limit": search_daily_limit,
                "search_quota_units_consumed": 0,
                "videos_requests_consumed": 0,
                "videos_quota_units_consumed": 0,
                "other_methods_quota_consumed": 0,
                "other_methods_daily_limit": other_quota_daily_limit,
                "total_estimated_quota_units": 0,
                "actual_cloud_quota_status": "Managed externally in Google Cloud Console",
                "note": (
                    "Local tracking reflects estimated session consumption against standard defaults "
                    "(100 search calls/day, 10,000 other-methods units/day). Actual remaining project quota "
                    "is not provided by the YouTube API and is managed in Google Cloud Console."
                ),
            },
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "errors": ["Missing or placeholder YouTube API key in .env."],
        }

    dst = Path(output_path) if output_path else config.raw_data_path / "youtube_videos_raw.json"

    all_raw_videos: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    errors: list[str] = []
    current_page_token: str | None = None
    pages_count = 0
    total_quota_units = 0
    search_calls = 0
    videos_calls = 0
    other_methods_units = 0

    for page_idx in range(max_pages):
        # 1. Stop pagination if separate search request limit reached
        if search_calls >= search_daily_limit:
            errors.append(
                f"Search request daily limit reached ({search_calls}/{search_daily_limit} calls). Stopping pagination."
            )
            break

        # 2. Stop pagination if other-methods quota limit reached
        if other_methods_units >= other_quota_daily_limit:
            errors.append(
                f"Other-methods daily quota limit reached ({other_methods_units}/{other_quota_daily_limit} units). Stopping pagination."
            )
            break

        videos, next_token, units, err = fetch_youtube_videos_page(
            api_key=api_key,
            query=query,
            max_results=max_results_per_page,
            page_token=current_page_token,
        )
        search_calls += 1
        total_quota_units += units

        if videos:
            batches = (len(videos) + 49) // 50
            videos_calls += batches
            other_methods_units += batches * QUOTA_COST_VIDEOS_LIST
            for v in videos:
                vid = v.get("video_id")
                if vid and vid not in seen_video_ids:
                    seen_video_ids.add(vid)
                    all_raw_videos.append(v)

        if err:
            errors.append(f"Page {page_idx + 1}: {err}")
            break
        pages_count += 1
        current_page_token = next_token

        if not current_page_token:
            break

    valid_records: list[dict[str, Any]] = []
    for idx, video in enumerate(all_raw_videos):
        is_valid, validation_err = validate_youtube_video(video)
        if is_valid:
            valid_records.append(video)
        else:
            vid = video.get("video_id", f"record[{idx}]")
            errors.append(f"Rejected '{vid}': {validation_err}")

    rejected_count = len(all_raw_videos) - len(valid_records)
    saved_path: str | None = None
    if valid_records:
        saved_file = save_raw_youtube_videos(valid_records, dst, merge=merge)
        saved_path = str(saved_file)

    quota_accounting = {
        "search_requests_consumed": search_calls,
        "search_request_daily_limit": search_daily_limit,
        "search_quota_units_consumed": search_calls * QUOTA_COST_SEARCH_LIST,
        "videos_requests_consumed": videos_calls,
        "videos_quota_units_consumed": other_methods_units,
        "other_methods_quota_consumed": other_methods_units,
        "other_methods_daily_limit": other_quota_daily_limit,
        "total_estimated_quota_units": (search_calls * QUOTA_COST_SEARCH_LIST) + other_methods_units,
        "actual_cloud_quota_status": "Managed externally in Google Cloud Console",
        "note": (
            "Local tracking reflects estimated session consumption against standard defaults "
            "(100 search calls/day, 10,000 other-methods units/day). Actual remaining project quota "
            "is not provided by the YouTube API and is managed in Google Cloud Console."
        ),
    }

    return {
        "status": "success" if valid_records else ("error" if errors else "no_records"),
        "source": "youtube_api_v3",
        "query": query,
        "pages_fetched": pages_count,
        "estimated_quota_units": (search_calls * QUOTA_COST_SEARCH_LIST) + other_methods_units,
        "quota_accounting": quota_accounting,
        "total_records": len(all_raw_videos),
        "valid_records": len(valid_records),
        "rejected_records": rejected_count,
        "output_file": saved_path,
        "errors": errors,
    }


if __name__ == "__main__":
    print("Testing YouTube Ingestion Module...")
    res = ingest_youtube_live()
    print(f"Status  : {res['status']}")
    print(f"Message : {res.get('message', res.get('status'))}")

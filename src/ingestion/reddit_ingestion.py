"""Reddit Data Ingestion Module (Company Step 1).

Supports both:
1. Synthetic mock Reddit post ingestion for local offline testing.
2. Live Reddit API data collection via OAuth2 client-credentials grant,
   respecting Reddit rate limits, terms of use, and error conditions.
"""

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from config.settings import load_config, BASE_DIR, _is_configured

# Required fields for a valid Reddit-style post
REQUIRED_FIELDS = (
    "post_id",
    "subreddit",
    "title",
    "text",
    "author",
    "created_utc",
    "score",
    "num_comments",
    "url",
)


def validate_reddit_post(post: dict[str, Any]) -> tuple[bool, str | None]:
    """Validates that a post record contains all required fields and valid data types."""
    if not isinstance(post, dict):
        return False, "Record must be a dictionary object."

    missing_fields = [field for field in REQUIRED_FIELDS if field not in post]
    if missing_fields:
        return False, f"Missing required field(s): {', '.join(missing_fields)}"

    for string_field in ("post_id", "subreddit", "title", "author", "url"):
        val = post[string_field]
        if not isinstance(val, str) or not val.strip():
            return False, f"Field '{string_field}' must be a non-empty string."

    if not isinstance(post["text"], str):
        return False, "Field 'text' must be a string."

    for numeric_field in ("created_utc", "score", "num_comments"):
        val = post[numeric_field]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return False, f"Field '{numeric_field}' must be a numeric value."

    return True, None


def load_sample_posts(file_path: Path | str) -> list[dict[str, Any]]:
    """Loads Reddit posts from a local JSON file."""
    target_path = Path(file_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Source file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of posts, got {type(data).__name__}.")

    return data


def save_raw_posts(
    posts: list[dict[str, Any]],
    output_path: Path | str,
    merge: bool = True,
) -> Path:
    """Saves valid post records to raw JSON destination.

    Preserves existing collected data when merge=True to avoid data loss.
    Stamps 'collected_at' timestamp (ISO 8601 UTC) without modifying original post text.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()

    existing_posts: list[dict[str, Any]] = []
    if merge and destination.exists():
        try:
            with open(destination, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    existing_posts = content
        except (json.JSONDecodeError, OSError):
            existing_posts = []

    combined_by_id = {}
    for p in existing_posts:
        if isinstance(p, dict) and "post_id" in p:
            combined_by_id[p["post_id"]] = p

    for p in posts:
        if isinstance(p, dict) and "post_id" in p:
            if "collected_at" not in p:
                if p["post_id"] in combined_by_id and "collected_at" in combined_by_id[p["post_id"]]:
                    p["collected_at"] = combined_by_id[p["post_id"]]["collected_at"]
                else:
                    p["collected_at"] = now_iso
            combined_by_id[p["post_id"]] = p

    merged_list = list(combined_by_id.values()) if combined_by_id else posts

    with open(destination, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)

    return destination


# -----------------------------------------------------------------------------
# Live Reddit API Client Functions (OAuth2)
# -----------------------------------------------------------------------------

def validate_reddit_user_agent(user_agent: str | None) -> tuple[bool, str | None]:
    """Validates that User-Agent conforms to Reddit's required API standard.

    Reddit terms require: <platform>:<app ID>:<version> (by /u/<reddit username>)
    """
    if not user_agent or not isinstance(user_agent, str):
        return False, "User-Agent is missing. Reddit API requires a descriptive User-Agent."
    stripped = user_agent.strip()
    if not stripped:
        return False, "User-Agent cannot be empty or whitespace."
    if "your_username" in stripped.lower():
        return False, "User-Agent contains placeholder 'your_username'. Set your approved Reddit username."
    if "by /u/" not in stripped.lower() and "by u/" not in stripped.lower():
        return False, "User-Agent must follow Reddit format: 'platform:app_id:v1.0 (by /u/username)'."
    return True, None


def fetch_reddit_oauth_token(
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> tuple[str | None, str | None]:
    """Obtains a short-lived OAuth bearer token from Reddit via client_credentials.

    Returns:
        (access_token, error_message).
    """
    token_url = "https://www.reddit.com/api/v1/access_token"
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {auth_header}",
        "User-Agent": user_agent or "social-media-intelligence:v1.0 (default)",
    }
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
    req = urllib.request.Request(token_url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
            token = payload.get("access_token")
            if not token:
                return None, "Reddit OAuth response did not contain an access_token."
            return token, None
    except urllib.error.HTTPError as err:
        if err.code == 401:
            return None, "Authentication failed (HTTP 401): Invalid Reddit client ID or secret."
        if err.code == 403:
            return None, "Access forbidden (HTTP 403): Check Reddit app credentials and User-Agent."
        if err.code == 429:
            return None, "Reddit API rate limit exceeded (HTTP 429)."
        return None, f"Reddit OAuth HTTP error {err.code}: {err.reason}"
    except urllib.error.URLError as err:
        return None, f"Network error connecting to Reddit: {err.reason}"
    except Exception as err:
        return None, f"Unexpected error obtaining Reddit OAuth token: {err}"


def fetch_subreddit_posts(
    subreddit: str,
    access_token: str,
    user_agent: str,
    limit: int = 25,
    listing: str = "hot",
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Collects public posts from a target subreddit via Reddit OAuth API.

    Returns:
        (posts_list, rate_limit_info, error_message).
    """
    clean_sub = subreddit.strip().lstrip("r/").strip()
    api_url = f"https://oauth.reddit.com/r/{clean_sub}/{listing}?limit={min(max(1, limit), 100)}"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": user_agent or "social-media-intelligence:v1.0 (default)",
    }
    req = urllib.request.Request(api_url, headers=headers, method="GET")

    rate_limits: dict[str, Any] = {}

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            # Capture Reddit rate limit headers
            rate_limits["remaining"] = response.headers.get("X-Ratelimit-Remaining")
            rate_limits["reset_seconds"] = response.headers.get("X-Ratelimit-Reset")
            rate_limits["used"] = response.headers.get("X-Ratelimit-Used")

            payload = json.loads(response.read().decode("utf-8"))

            posts: list[dict[str, Any]] = []
            children = payload.get("data", {}).get("children", [])
            for child in children:
                data = child.get("data", {})
                permalink = data.get("permalink", "")
                url = f"https://reddit.com{permalink}" if permalink else data.get("url", "")
                post_id = data.get("name") or f"t3_{data.get('id', '')}"

                posts.append({
                    "post_id": post_id,
                    "subreddit": data.get("subreddit", clean_sub),
                    "title": data.get("title", ""),
                    "text": data.get("selftext", ""),
                    "author": data.get("author", "[deleted]"),
                    "created_utc": int(data.get("created_utc", 0)),
                    "score": int(data.get("score", 0)),
                    "num_comments": int(data.get("num_comments", 0)),
                    "url": url,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })

            return posts, rate_limits, None

    except urllib.error.HTTPError as err:
        if err.code == 429:
            return [], rate_limits, "Reddit API rate limit exceeded (HTTP 429)."
        if err.code in (401, 403):
            return [], rate_limits, f"Reddit authorization failed (HTTP {err.code})."
        return [], rate_limits, f"Reddit API HTTP error {err.code}: {err.reason}"
    except urllib.error.URLError as err:
        return [], rate_limits, f"Network error connecting to Reddit: {err.reason}"
    except Exception as err:
        return [], rate_limits, f"Unexpected error fetching subreddit posts: {err}"


def ingest_reddit_live(
    subreddits: list[str] | str | None = None,
    limit_per_subreddit: int = 25,
    output_path: Path | str | None = None,
    merge: bool = True,
) -> dict[str, Any]:
    """Orchestrates live Reddit API collection using OAuth2 credentials.

    Safely disables collection if credentials are missing or placeholders.
    """
    config = load_config()

    client_id = config.reddit_client_id
    client_secret = config.reddit_client_secret
    user_agent = config.reddit_user_agent

    # Validate credential readiness
    if not _is_configured(client_id) or not _is_configured(client_secret):
        return {
            "status": "disabled",
            "source": "reddit_live_api",
            "message": "Reddit API credentials not configured in .env.",
            "instructions": "Approved Reddit Developer API credentials required. Add valid REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env to enable.",
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "rate_limits": {},
            "errors": ["Missing or placeholder Reddit API credentials."],
        }

    # Validate User-Agent adheres to Reddit developer policy
    ua_valid, ua_err = validate_reddit_user_agent(user_agent)
    if not ua_valid:
        return {
            "status": "disabled",
            "source": "reddit_live_api",
            "message": f"Invalid User-Agent: {ua_err}",
            "instructions": "Set a valid REDDIT_USER_AGENT in .env adhering to: 'platform:app_id:v1.0 (by /u/username)'.",
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "rate_limits": {},
            "errors": [ua_err or "Invalid User-Agent"],
        }

    # Determine target subreddits
    if isinstance(subreddits, str):
        target_subs = [s.strip() for s in subreddits.split(",") if s.strip()]
    elif isinstance(subreddits, list):
        target_subs = subreddits
    else:
        target_subs = ["datascience", "technology", "MachineLearning"]

    dst = Path(output_path) if output_path else config.raw_data_path / "reddit_posts_raw.json"

    token, token_err = fetch_reddit_oauth_token(client_id, client_secret, user_agent)
    if token_err or not token:
        return {
            "status": "error",
            "source": "reddit_live_api",
            "message": "Failed to authenticate with Reddit OAuth.",
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "rate_limits": {},
            "errors": [token_err or "Unknown token error"],
        }

    all_raw_posts: list[dict[str, Any]] = []
    errors: list[str] = []
    latest_rate_limits: dict[str, Any] = {}

    for sub in target_subs:
        posts, limits, fetch_err = fetch_subreddit_posts(
            subreddit=sub,
            access_token=token,
            user_agent=user_agent,
            limit=limit_per_subreddit,
        )
        if limits:
            latest_rate_limits = limits

        if fetch_err:
            errors.append(f"Subreddit 'r/{sub}': {fetch_err}")
            # Ensure requests do not continue when rate limits or access restrictions prevent collection
            if any(term in fetch_err.lower() for term in ("rate limit", "authorization", "429", "401", "403")):
                errors.append("Halting further subreddit requests due to rate limiting or access restriction.")
                break
        else:
            all_raw_posts.extend(posts)

        # Inspect remaining rate limit quota: halt if quota exhausted
        rem = limits.get("remaining")
        if rem is not None:
            try:
                if float(rem) <= 0.0:
                    errors.append(f"Reddit rate limit exhausted (remaining={rem}, reset={limits.get('reset_seconds')}s). Halting collection.")
                    break
            except (ValueError, TypeError):
                pass

    valid_records: list[dict[str, Any]] = []
    for idx, post in enumerate(all_raw_posts):
        is_valid, validation_err = validate_reddit_post(post)
        if is_valid:
            valid_records.append(post)
        else:
            pid = post.get("post_id", f"record[{idx}]")
            errors.append(f"Rejected '{pid}': {validation_err}")

    rejected_count = len(all_raw_posts) - len(valid_records)
    saved_path: str | None = None
    if valid_records:
        saved_file = save_raw_posts(valid_records, dst, merge=merge)
        saved_path = str(saved_file)

    return {
        "status": "success" if valid_records else ("error" if errors else "no_records"),
        "source": "reddit_live_api",
        "total_records": len(all_raw_posts),
        "valid_records": len(valid_records),
        "rejected_records": rejected_count,
        "output_file": saved_path,
        "rate_limits": latest_rate_limits,
        "errors": errors,
    }


# -----------------------------------------------------------------------------
# Mock / Synthetic Ingestion (Preserved from Block 2)
# -----------------------------------------------------------------------------

def ingest_reddit_posts(
    source_path: Path | str | None = None,
    output_path: Path | str | None = None,
    merge: bool = True,
) -> dict[str, Any]:
    """Orchestrates loading, validating, and saving synthetic Reddit posts."""
    config = load_config()

    src = Path(source_path) if source_path else BASE_DIR / "data" / "sample_reddit_posts.json"
    dst = Path(output_path) if output_path else config.raw_data_path / "reddit_posts_raw.json"

    errors: list[str] = []

    try:
        raw_records = load_sample_posts(src)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as err:
        return {
            "status": "error",
            "source": "reddit_mock",
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "errors": [str(err)],
        }

    valid_records: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        is_valid, error_msg = validate_reddit_post(record)
        if is_valid:
            valid_records.append(record)
        else:
            record_id = record.get("post_id", f"record[{index}]") if isinstance(record, dict) else f"record[{index}]"
            errors.append(f"Rejected '{record_id}': {error_msg}")

    rejected_count = len(raw_records) - len(valid_records)

    saved_path: str | None = None
    if valid_records:
        saved_file = save_raw_posts(valid_records, dst, merge=merge)
        saved_path = str(saved_file)

    return {
        "status": "success" if valid_records else "no_valid_records",
        "source": "reddit_mock",
        "total_records": len(raw_records),
        "valid_records": len(valid_records),
        "rejected_records": rejected_count,
        "output_file": saved_path,
        "errors": errors,
    }


# -----------------------------------------------------------------------------
# Reddit Data Retention and Deletion Compliance
# (Reddit Data API Wiki & Developer Terms Compliance)
# -----------------------------------------------------------------------------

def anonymize_reddit_author(
    author_to_anonymize: str,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Anonymizes author-identifying information for a specific Reddit user.

    Replaces the author field with '[deleted]' across all records matching
    author_to_anonymize in stored raw JSON, preserving the original post text.
    """
    config = load_config()
    target_path = Path(file_path) if file_path else config.raw_data_path / "reddit_posts_raw.json"

    if not target_path.exists():
        return {
            "status": "not_found",
            "source": "reddit_author_anonymization",
            "message": f"Raw data file not found at {target_path}",
            "author_target": author_to_anonymize,
            "anonymized_count": 0,
            "total_records": 0,
            "file_path": str(target_path),
        }

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "reddit_author_anonymization",
            "message": f"Failed reading file: {err}",
            "author_target": author_to_anonymize,
            "anonymized_count": 0,
            "total_records": 0,
            "file_path": str(target_path),
        }

    target_clean = author_to_anonymize.strip().lower()
    anonymized_count = 0

    for post in posts:
        if isinstance(post, dict):
            curr_author = str(post.get("author", "")).strip().lower()
            if curr_author == target_clean:
                post["author"] = "[deleted]"
                anonymized_count += 1

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "source": "reddit_author_anonymization",
        "author_target": author_to_anonymize,
        "anonymized_count": anonymized_count,
        "total_records": len(posts),
        "file_path": str(target_path),
    }


def purge_reddit_posts(
    post_ids_to_purge: list[str],
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Purges specified Reddit posts by post_id from local storage.

    Used when source posts are known to be deleted or requested for removal.
    """
    config = load_config()
    target_path = Path(file_path) if file_path else config.raw_data_path / "reddit_posts_raw.json"

    if not target_path.exists():
        return {
            "status": "not_found",
            "source": "reddit_purge",
            "message": f"Raw data file not found at {target_path}",
            "requested_ids": len(post_ids_to_purge),
            "purged_count": 0,
            "remaining_count": 0,
            "file_path": str(target_path),
        }

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "reddit_purge",
            "message": f"Failed reading file: {err}",
            "requested_ids": len(post_ids_to_purge),
            "purged_count": 0,
            "remaining_count": 0,
            "file_path": str(target_path),
        }

    purge_set = set(post_ids_to_purge)
    for pid in list(purge_set):
        if pid.startswith("t3_"):
            purge_set.add(pid[3:])
        else:
            purge_set.add(f"t3_{pid}")

    retained: list[dict[str, Any]] = []
    purged_count = 0

    for post in posts:
        if isinstance(post, dict) and post.get("post_id") in purge_set:
            purged_count += 1
        else:
            retained.append(post)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(retained, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "source": "reddit_purge",
        "requested_ids": len(post_ids_to_purge),
        "purged_count": purged_count,
        "remaining_count": len(retained),
        "file_path": str(target_path),
    }


def reconcile_reddit_deletions(
    access_token: str,
    user_agent: str,
    post_ids_to_check: list[str] | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    """Reconciles stored Reddit posts against live Reddit API to identify deleted content.

    IMPORTANT COMPLIANCE NOTICE:
    Stored Reddit records cannot automatically know when posts or accounts are
    deleted without actively querying Reddit's authorized API or receiving deletion notices.
    This function queries Reddit via '/by_id/{names}' to verify whether stored records
    still exist or have been deleted/removed by users or moderators. Records confirmed
    as deleted are automatically purged from local storage.
    """
    config = load_config()
    target_path = Path(file_path) if file_path else config.raw_data_path / "reddit_posts_raw.json"

    if not target_path.exists():
        return {
            "status": "not_found",
            "source": "reddit_deletion_reconciliation",
            "message": f"Raw data file not found at {target_path}",
            "checked_count": 0,
            "deleted_detected_count": 0,
            "purged_count": 0,
            "remaining_count": 0,
            "file_path": str(target_path),
            "note": "Deleted content cannot be detected automatically without querying the Reddit API.",
        }

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "reddit_deletion_reconciliation",
            "message": f"Failed reading file: {err}",
            "checked_count": 0,
            "deleted_detected_count": 0,
            "purged_count": 0,
            "remaining_count": 0,
            "file_path": str(target_path),
            "note": "Deleted content cannot be detected automatically without querying the Reddit API.",
        }

    if post_ids_to_check is not None:
        target_ids = list(post_ids_to_check)
    else:
        target_ids = [p["post_id"] for p in posts if isinstance(p, dict) and "post_id" in p]

    if not target_ids:
        return {
            "status": "success",
            "source": "reddit_deletion_reconciliation",
            "checked_count": 0,
            "deleted_detected_count": 0,
            "purged_count": 0,
            "remaining_count": len(posts),
            "file_path": str(target_path),
            "note": "Deleted content cannot be detected automatically without querying the Reddit API.",
        }

    fullnames = [pid if pid.startswith("t3_") else f"t3_{pid}" for pid in target_ids]
    batch_size = 100
    deleted_ids: list[str] = []

    for i in range(0, len(fullnames), batch_size):
        batch = fullnames[i : i + batch_size]
        url = f"https://oauth.reddit.com/by_id/{','.join(batch)}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": user_agent or "social-media-intelligence:v1.0 (default)",
        }
        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                live_children = data.get("data", {}).get("children", [])
                live_posts_by_fullname = {}
                for child in live_children:
                    cdata = child.get("data", {})
                    fname = cdata.get("name")
                    if fname:
                        live_posts_by_fullname[fname] = cdata

                for fname in batch:
                    orig_id = fname if fname in target_ids else fname.removeprefix("t3_")
                    if orig_id not in target_ids and fname in target_ids:
                        orig_id = fname

                    if fname not in live_posts_by_fullname:
                        deleted_ids.append(orig_id)
                    else:
                        cdata = live_posts_by_fullname[fname]
                        is_deleted = (
                            cdata.get("author") == "[deleted]"
                            or cdata.get("selftext") == "[deleted]"
                            or cdata.get("title") == "[deleted]"
                            or bool(cdata.get("removed_by_category"))
                        )
                        if is_deleted:
                            deleted_ids.append(orig_id)
        except Exception as err:
            return {
                "status": "error",
                "source": "reddit_deletion_reconciliation",
                "message": f"API error during reconciliation: {err}",
                "checked_count": len(target_ids),
                "deleted_detected_count": len(deleted_ids),
                "purged_count": 0,
                "remaining_count": len(posts),
                "file_path": str(target_path),
                "note": "Deleted content cannot be detected automatically without querying the Reddit API.",
            }

    if deleted_ids:
        purge_res = purge_reddit_posts(deleted_ids, file_path=target_path)
        remaining = purge_res["remaining_count"]
    else:
        remaining = len(posts)

    return {
        "status": "success",
        "source": "reddit_deletion_reconciliation",
        "checked_count": len(target_ids),
        "deleted_detected_count": len(deleted_ids),
        "purged_count": len(deleted_ids),
        "remaining_count": remaining,
        "deleted_post_ids": deleted_ids,
        "file_path": str(target_path),
        "note": "Deleted content cannot be detected automatically without querying the Reddit API.",
    }


def apply_reddit_retention_policy(
    max_age_hours: int = 48,
    file_path: Path | str | None = None,
    max_age_days: int | float | None = None,
) -> dict[str, Any]:
    """Applies Reddit's data retention policy to stored records.

    Compliance Notice (Reddit Data API Wiki & Developer Terms):
    - Do NOT assume a 90-day retention period is compliant.
    - Reddit Data API policies recommend that stored data be routinely deleted within 48 hours
      unless active deletion reconciliation or authorized retention agreements apply.
    - Purges records where 'collected_at' exceeds the retention threshold (default: 48 hours).
    - Preserves unexpired records and completely removes deleted/expired records from disk
      to ensure they are not retained or accidentally reused.

    Args:
        max_age_hours: Maximum data retention window in hours (default: 48 hours).
        file_path: Optional custom path to raw Reddit JSON file.
        max_age_days: Optional backward-compatible argument specifying retention in days.
                      If passed, overrides max_age_hours with (max_age_days * 24).
    """
    config = load_config()
    target_path = Path(file_path) if file_path else config.raw_data_path / "reddit_posts_raw.json"

    if max_age_days is not None:
        effective_hours = float(max_age_days) * 24.0
    else:
        effective_hours = float(max_age_hours)

    if not target_path.exists():
        return {
            "status": "not_found",
            "source": "reddit_retention",
            "message": f"Raw data file not found at {target_path}",
            "max_age_hours": effective_hours,
            "purged_count": 0,
            "retained_count": 0,
            "file_path": str(target_path),
            "policy": "Reddit Data API: routine deletion within 48 hours recommended",
        }

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "reddit_retention",
            "message": f"Failed reading file: {err}",
            "max_age_hours": effective_hours,
            "purged_count": 0,
            "retained_count": 0,
            "file_path": str(target_path),
            "policy": "Reddit Data API: routine deletion within 48 hours recommended",
        }

    cutoff = datetime.now(timezone.utc) - timedelta(hours=effective_hours)
    retained: list[dict[str, Any]] = []
    purged_count = 0

    for post in posts:
        collected_str = post.get("collected_at")
        if collected_str:
            try:
                collected_dt = datetime.fromisoformat(collected_str)
                if collected_dt < cutoff:
                    purged_count += 1
                    continue
            except (ValueError, TypeError):
                pass
        retained.append(post)

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(retained, f, indent=2, ensure_ascii=False)

    return {
        "status": "success",
        "source": "reddit_retention",
        "max_age_hours": effective_hours,
        "purged_count": purged_count,
        "retained_count": len(retained),
        "file_path": str(target_path),
        "policy": "Reddit Data API: routine deletion within 48 hours recommended",
    }


if __name__ == "__main__":
    print("Running Reddit Ingestion...")
    mock_summary = ingest_reddit_posts()
    print(f"Mock Ingestion Status : {mock_summary['status']}")
    print(f"Records Ingested      : {mock_summary['valid_records']}")
    print(f"Output File           : {mock_summary['output_file']}")

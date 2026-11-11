"""Elasticsearch Indexing Adapter for Consolidated Social Intelligence Platform.

Validates consolidated social intelligence records against strict Elasticsearch mappings
and analytical consistency rules, normalizes legacy topic keywords non-destructively,
orchestrates resilient document indexing, and tracks complete status summaries.

Key Invariants:
- Refresh parameter validated before file access, client initialization, or driver calls.
- Strict document identity validation (platform, native_id without trimming, document_id).
- Envelope segregation: Envelope fields are never indexed into Elasticsearch documents.
- Mathematical count invariant: total_records == indexed_count + rejected_count + failed_count.
- Preflight failure accounting: Accounts all unattempted records as failed_count.
- Security and redaction: Diagnostic rejection reports contain safe metadata only.
- Client ownership: Automatically closes adapter-instantiated clients while leaving caller clients open.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.pipeline.consolidator import ConsolidatedArtifactValidator
from src.search.elasticsearch_client import (
    DEFAULT_BULK_BATCH_SIZE,
    DEFAULT_REQUEST_TIMEOUT,
    ElasticsearchClient,
    create_elasticsearch_client,
    validate_index_name,
)
from src.search.index_mapping import DEFAULT_INDEX_NAME

logger = logging.getLogger(__name__)

# Exact 13 root fields permitted in consolidated documents
CANONICAL_ROOT_FIELDS = {
    "document_id",
    "platform",
    "native_id",
    "url",
    "timestamps",
    "author",
    "content",
    "engagement",
    "metadata",
    "sentiment",
    "topics",
    "misinformation",
    "consolidation",
}

# Accepted graph status values produced by consolidator
VALID_GRAPH_STATUSES = {
    "ranked_influencer",
    "not_in_top_influencers",
    "anonymous",
    "unavailable",
    "corrupted",
}

# Recognized analytical status strings
RECOGNIZED_SENTIMENT_STATUSES = {
    "success",
    "fallback",
    "empty_text",
    "error",
    "unavailable",
    "corrupted",
    "missing_status",
}

RECOGNIZED_TOPIC_STATUSES = {
    "assigned",
    "outlier",
    "fallback",
    "empty_text",
    "error",
    "unavailable",
    "corrupted",
    "missing_status",
}

RECOGNIZED_MISINFO_STATUSES = {
    "success",
    "fallback",
    "empty_text",
    "error",
    "unavailable",
    "corrupted",
    "missing_status",
}

# Misinformation classifier verified outputs
VALID_VERACITY_LABELS = {"real", "fake"}
VALID_RISK_TIERS = {"low", "moderate", "high_risk"}

# Canonical stylometric signal keys
EXPECTED_STYLOMETRIC_SIGNALS = {
    "uppercase_char_ratio",
    "exclamation_intensity",
    "question_intensity",
    "word_count",
    "avg_word_length",
    "lexical_subjectivity_ratio",
    "digits_ratio",
    "all_caps_word_ratio",
}

# Content completeness core sources
REQUIRED_FOUR_CONTENT_SOURCES = {
    "preprocessing",
    "sentiment",
    "topics",
    "misinformation",
}


def _is_valid_refresh(refresh: Any) -> bool:
    """Validates refresh parameter with strict type/value rules.

    Accepts True, False, 'wait_for', None. Rejects integers 1 and 0, strings, lists.
    """
    if refresh is None:
        return True
    if refresh == "wait_for" and isinstance(refresh, str):
        return True
    if isinstance(refresh, bool):
        return True
    return False


def _make_rejection(
    doc_id: Optional[str],
    dimension: str,
    field: str,
    expected: str,
    actual_type: str,
    reason: str,
) -> Dict[str, Any]:
    """Builds a sanitized, non-reflective diagnostic rejection dictionary."""
    safe_doc_id = doc_id if isinstance(doc_id, str) and doc_id == doc_id.strip() else None
    return {
        "document_id": safe_doc_id,
        "dimension": dimension,
        "status": "rejected",
        "field": field,
        "expected": expected,
        "actual_type": actual_type,
        "reason": reason,
    }


def normalize_topic_keywords(
    raw_keywords: Any,
) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
    """Validates and normalizes keyword entries to canonical [{"word": str, "weight": float|None}].

    Follows the approved Step 7.1 specification:
    - Legacy string "machine learning" becomes {"word": "machine learning", "weight": None}.
    - Never fabricates a synthetic weight such as 1.0.
    - Canonical objects must use exact keys "word" and "weight", rejecting unapproved keys like "keyword".
    - Preserves valid canonical objects without modifying their analytical values.
    - Missing or null weight in canonical objects normalizes cleanly to None.
    - Strictly rejects malformed objects and invalid types (does not convert invalid input to empty data).
    - Preserves empty [] and null None keyword representations according to the schema and consolidator contract.
    - Pure function that does not mutate input sequences or nested dictionaries.

    Args:
        raw_keywords: Raw keyword sequence from document (list of str/dict, empty list, or None).

    Returns:
        (is_valid, normalized_list, error_message)
    """
    if raw_keywords is None:
        return True, None, None

    if not isinstance(raw_keywords, list):
        return False, None, f"Expected list or null for topics.keywords, got {type(raw_keywords).__name__}"

    if len(raw_keywords) == 0:
        return True, [], None

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_keywords):
        if isinstance(item, str):
            cleaned = item.strip()
            if not cleaned:
                return False, None, f"Empty keyword string at index {idx}"
            # Legacy string normalization: weight is explicitly None, never fabricated as 1.0
            normalized.append({"word": cleaned, "weight": None})
        elif isinstance(item, dict):
            # Check for unknown keys inside keyword object safely without sorting mixed types
            extra_keys = set(item.keys()) - {"word", "weight"}
            if extra_keys:
                return False, None, f"Unexpected key in keyword object at index {idx}"

            if "word" not in item:
                return False, None, f"Keyword object at index {idx} missing required key 'word'"

            word = item["word"]
            if not isinstance(word, str) or not word.strip():
                return False, None, f"Keyword object at index {idx} has invalid 'word' string"

            weight = item.get("weight")
            if weight is not None:
                # Sanitize weight checks: reject booleans and non-numeric types generically
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    return False, None, f"Keyword object at index {idx} has invalid weight"
                # Handle extremely large integer/float weights safely without unhandled OverflowError
                try:
                    f_weight = float(weight)
                    if not math.isfinite(f_weight):
                        return False, None, f"Keyword object at index {idx} has invalid weight"
                    weight = f_weight
                except (OverflowError, ValueError):
                    return False, None, f"Keyword object at index {idx} has invalid weight"

            # Preserve valid canonical object analytical values without modification
            normalized.append({"word": word, "weight": weight})
        else:
            return False, None, f"Invalid keyword element type at index {idx}: {type(item).__name__}"

    return True, normalized, None


def validate_and_normalize_record(
    record: Any,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Validates a social intelligence record against strict dynamic mapping and status consistency.

    Produces a clean normalized document copy without mutating the input record.

    Args:
        record: Candidate social intelligence record dictionary.

    Returns:
        (is_valid, normalized_document_copy, rejection_report_entry)
    """
    if not isinstance(record, dict):
        return False, None, _make_rejection(
            doc_id=None,
            dimension="schema",
            field="root",
            expected="dict",
            actual_type=type(record).__name__,
            reason="Expected dictionary for document record",
        )

    doc_id_raw = record.get("document_id")
    current_doc_id = str(doc_id_raw) if isinstance(doc_id_raw, str) else None

    # 1. Root fields check
    rec_keys = set(record.keys())
    missing_root = CANONICAL_ROOT_FIELDS - rec_keys
    if missing_root:
        missing_field = sorted(missing_root)[0]
        return False, None, _make_rejection(
            doc_id=current_doc_id,
            dimension="schema",
            field=missing_field,
            expected="present in root fields",
            actual_type="missing",
            reason=f"Missing required root field '{missing_field}'",
        )

    extra_root = rec_keys - CANONICAL_ROOT_FIELDS
    if extra_root:
        extra_field = sorted(extra_root)[0]
        return False, None, _make_rejection(
            doc_id=current_doc_id,
            dimension="schema",
            field=extra_field,
            expected="canonical root field",
            actual_type="unexpected",
            reason=f"Unknown root field '{extra_field}' not permitted by strict dynamic mapping",
        )

    # 2. Strict Document Identity Validation
    platform = record.get("platform")
    if platform not in {"reddit", "youtube"}:
        return False, None, _make_rejection(
            doc_id=current_doc_id,
            dimension="identity",
            field="platform",
            expected="'reddit' or 'youtube'",
            actual_type=type(platform).__name__,
            reason=f"Invalid platform '{platform}'; must be 'reddit' or 'youtube'",
        )

    native_id = record.get("native_id")
    if not isinstance(native_id, str) or not native_id:
        return False, None, _make_rejection(
            doc_id=current_doc_id,
            dimension="identity",
            field="native_id",
            expected="non-empty string",
            actual_type=type(native_id).__name__,
            reason="native_id must be a non-empty string",
        )

    if native_id != native_id.strip():
        return False, None, _make_rejection(
            doc_id=current_doc_id,
            dimension="identity",
            field="native_id",
            expected="string without leading or trailing whitespace",
            actual_type="str",
            reason="native_id contains leading or trailing whitespace",
        )

    expected_doc_id = f"{platform}_{native_id}"
    if doc_id_raw != expected_doc_id:
        return False, None, _make_rejection(
            doc_id=current_doc_id,
            dimension="identity",
            field="document_id",
            expected=f"exactly '{expected_doc_id}'",
            actual_type=type(doc_id_raw).__name__,
            reason=f"document_id does not match expected identity '{expected_doc_id}'",
        )

    # 3. URL Validation
    url_val = record.get("url")
    if url_val is not None:
        if not isinstance(url_val, str):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="url",
                field="url",
                expected="string or null",
                actual_type=type(url_val).__name__,
                reason="URL field must be a string or null",
            )
        if len(url_val) > 2048:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="url",
                field="url",
                expected="string length <= 2048 characters",
                actual_type="str",
                reason="URL exceeds maximum length of 2048 characters",
            )

    # 4. Timestamps Validation
    timestamps = record.get("timestamps")
    if not isinstance(timestamps, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="timestamps",
            field="timestamps",
            expected="dict",
            actual_type=type(timestamps).__name__,
            reason="timestamps field must be a dictionary",
        )

    expected_ts_keys = {"published_at", "collected_at", "processed_at"}
    ts_diff = (expected_ts_keys - set(timestamps.keys())) or (set(timestamps.keys()) - expected_ts_keys)
    if ts_diff:
        ts_field = sorted(ts_diff)[0]
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="timestamps",
            field=f"timestamps.{ts_field}",
            expected="exact timestamps structure (published_at, collected_at, processed_at)",
            actual_type="mismatch",
            reason=f"timestamps structure mismatch on field '{ts_field}'",
        )

    for ts_key in ("published_at", "collected_at", "processed_at"):
        ts_val = timestamps[ts_key]
        if ts_val is not None:
            if not isinstance(ts_val, str):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="timestamps",
                    field=f"timestamps.{ts_key}",
                    expected="ISO-8601 string or null",
                    actual_type=type(ts_val).__name__,
                    reason=f"Timestamp field '{ts_key}' must be an ISO-8601 string or null",
                )
            cleaned_ts = ts_val.strip()
            if cleaned_ts.endswith("Z"):
                cleaned_ts = cleaned_ts[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(cleaned_ts)
                if dt.tzinfo is None:
                    return False, None, _make_rejection(
                        doc_id=expected_doc_id,
                        dimension="timestamps",
                        field=f"timestamps.{ts_key}",
                        expected="ISO-8601 string with explicit timezone offset",
                        actual_type="str",
                        reason=f"Timestamp '{ts_key}' lacks explicit timezone offset",
                    )
            except Exception:
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="timestamps",
                    field=f"timestamps.{ts_key}",
                    expected="valid ISO-8601 date string",
                    actual_type="str",
                    reason=f"Invalid ISO-8601 date string for '{ts_key}'",
                )

    # 5. Author & Graph Analytics Validation
    author = record.get("author")
    if not isinstance(author, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="author",
            field="author",
            expected="dict",
            actual_type=type(author).__name__,
            reason="author field must be a dictionary",
        )

    expected_author_keys = {
        "username",
        "channel_id",
        "channel_title",
        "graph_status",
        "influencer_score",
        "pagerank",
        "community_id",
        "in_degree",
        "out_degree",
    }
    author_diff = (expected_author_keys - set(author.keys())) or (set(author.keys()) - expected_author_keys)
    if author_diff:
        auth_field = sorted(author_diff)[0]
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="author",
            field=f"author.{auth_field}",
            expected="canonical author fields",
            actual_type="mismatch",
            reason=f"author structure mismatch on field '{auth_field}'",
        )

    for str_key in ("username", "channel_id", "channel_title"):
        val = author[str_key]
        if val is not None and not isinstance(val, str):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="author",
                field=f"author.{str_key}",
                expected="string or null",
                actual_type=type(val).__name__,
                reason=f"author.{str_key} must be a string or null",
            )

    graph_status = author.get("graph_status")
    if graph_status not in VALID_GRAPH_STATUSES:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="graph",
            field="author.graph_status",
            expected="one of ranked_influencer, not_in_top_influencers, anonymous, unavailable, corrupted",
            actual_type=type(graph_status).__name__,
            reason=f"Invalid graph_status '{graph_status}'",
        )

    # Graph metrics consistency
    if graph_status == "ranked_influencer":
        # influencer_score: finite float [0.0, 1.0]
        inf_score = author.get("influencer_score")
        if (
            inf_score is None
            or isinstance(inf_score, bool)
            or not isinstance(inf_score, (int, float))
            or not math.isfinite(inf_score)
            or not (0.0 <= inf_score <= 1.0)
        ):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="graph",
                field="author.influencer_score",
                expected="finite float in [0.0, 1.0]",
                actual_type=type(inf_score).__name__,
                reason="influencer_score must be a finite float in [0.0, 1.0] for ranked_influencer",
            )

        # pagerank: finite float >= 0.0
        pr = author.get("pagerank")
        if (
            pr is None
            or isinstance(pr, bool)
            or not isinstance(pr, (int, float))
            or not math.isfinite(pr)
            or pr < 0.0
        ):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="graph",
                field="author.pagerank",
                expected="finite float >= 0.0",
                actual_type=type(pr).__name__,
                reason="pagerank must be a finite float >= 0.0 for ranked_influencer",
            )

        # community_id: signed 32-bit int >= 0
        cid = author.get("community_id")
        if cid is None or isinstance(cid, bool) or not isinstance(cid, int) or not (0 <= cid <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="graph",
                field="author.community_id",
                expected="32-bit signed integer >= 0",
                actual_type=type(cid).__name__,
                reason="community_id must be a non-negative 32-bit integer for ranked_influencer",
            )

        # in_degree: signed 32-bit int >= 0
        in_deg = author.get("in_degree")
        if in_deg is None or isinstance(in_deg, bool) or not isinstance(in_deg, int) or not (0 <= in_deg <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="graph",
                field="author.in_degree",
                expected="32-bit signed integer >= 0",
                actual_type=type(in_deg).__name__,
                reason="in_degree must be a non-negative 32-bit integer for ranked_influencer",
            )

        # out_degree: signed 32-bit int >= 0
        out_deg = author.get("out_degree")
        if out_deg is None or isinstance(out_deg, bool) or not isinstance(out_deg, int) or not (0 <= out_deg <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="graph",
                field="author.out_degree",
                expected="32-bit signed integer >= 0",
                actual_type=type(out_deg).__name__,
                reason="out_degree must be a non-negative 32-bit integer for ranked_influencer",
            )
    else:
        # Non-ranked graph status: all graph metrics must be null
        for metric_field in ("influencer_score", "pagerank", "community_id", "in_degree", "out_degree"):
            m_val = author.get(metric_field)
            if m_val is not None:
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="graph",
                    field=f"author.{metric_field}",
                    expected=f"null when graph_status is '{graph_status}'",
                    actual_type=type(m_val).__name__,
                    reason=f"author.{metric_field} must be null when graph_status is not 'ranked_influencer'",
                )

    # 6. Content Validation
    content = record.get("content")
    if not isinstance(content, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="content",
            field="content",
            expected="dict",
            actual_type=type(content).__name__,
            reason="content field must be a dictionary",
        )

    expected_content_keys = {
        "title_raw",
        "body_raw",
        "title_cleaned",
        "body_cleaned",
        "detected_language",
        "language_code",
        "preprocessing_status",
    }
    content_diff = (expected_content_keys - set(content.keys())) or (set(content.keys()) - expected_content_keys)
    if content_diff:
        c_field = sorted(content_diff)[0]
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="content",
            field=f"content.{c_field}",
            expected="canonical content fields",
            actual_type="mismatch",
            reason=f"content structure mismatch on field '{c_field}'",
        )

    for str_key in ("title_raw", "body_raw", "title_cleaned", "body_cleaned", "detected_language", "language_code"):
        c_val = content[str_key]
        if c_val is not None and not isinstance(c_val, str):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="content",
                field=f"content.{str_key}",
                expected="string or null",
                actual_type=type(c_val).__name__,
                reason=f"content.{str_key} must be a string or null",
            )

    prep_status = content.get("preprocessing_status")
    if not isinstance(prep_status, str) or not prep_status.strip():
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="content",
            field="content.preprocessing_status",
            expected="non-empty string",
            actual_type=type(prep_status).__name__,
            reason="content.preprocessing_status must be a non-empty string",
        )

    # 7. Engagement Validation
    engagement = record.get("engagement")
    if not isinstance(engagement, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="engagement",
            field="engagement",
            expected="dict",
            actual_type=type(engagement).__name__,
            reason="engagement field must be a dictionary",
        )

    expected_engagement_keys = {"score", "views_count", "likes_count", "comments_count"}
    eng_diff = (expected_engagement_keys - set(engagement.keys())) or (set(engagement.keys()) - expected_engagement_keys)
    if eng_diff:
        eng_field = sorted(eng_diff)[0]
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="engagement",
            field=f"engagement.{eng_field}",
            expected="canonical engagement fields",
            actual_type="mismatch",
            reason=f"engagement structure mismatch on field '{eng_field}'",
        )

    # score: 32-bit signed int
    eng_score = engagement.get("score")
    if eng_score is not None:
        if isinstance(eng_score, bool) or not isinstance(eng_score, int) or not (-2147483648 <= eng_score <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="engagement",
                field="engagement.score",
                expected="32-bit signed integer or null",
                actual_type=type(eng_score).__name__,
                reason="engagement.score must be a 32-bit signed integer (excluding bool)",
            )

    # views_count: 64-bit signed int >= 0
    views = engagement.get("views_count")
    if views is not None:
        if isinstance(views, bool) or not isinstance(views, int) or not (0 <= views <= 9223372036854775807):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="engagement",
                field="engagement.views_count",
                expected="64-bit signed integer >= 0 or null",
                actual_type=type(views).__name__,
                reason="engagement.views_count must be a non-negative 64-bit integer (excluding bool)",
            )

    # likes_count: 32-bit signed int >= 0
    likes = engagement.get("likes_count")
    if likes is not None:
        if isinstance(likes, bool) or not isinstance(likes, int) or not (0 <= likes <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="engagement",
                field="engagement.likes_count",
                expected="32-bit signed integer >= 0 or null",
                actual_type=type(likes).__name__,
                reason="engagement.likes_count must be a non-negative 32-bit integer (excluding bool)",
            )

    # comments_count: 32-bit signed int >= 0
    comments = engagement.get("comments_count")
    if comments is not None:
        if isinstance(comments, bool) or not isinstance(comments, int) or not (0 <= comments <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="engagement",
                field="engagement.comments_count",
                expected="32-bit signed integer >= 0 or null",
                actual_type=type(comments).__name__,
                reason="engagement.comments_count must be a non-negative 32-bit integer (excluding bool)",
            )

    # 8. Metadata Platform Separation Validation
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="metadata",
            field="metadata",
            expected="dict",
            actual_type=type(metadata).__name__,
            reason="metadata field must be a dictionary",
        )

    expected_metadata_keys = {"reddit", "youtube"}
    meta_diff = (expected_metadata_keys - set(metadata.keys())) or (set(metadata.keys()) - expected_metadata_keys)
    if meta_diff:
        m_field = sorted(meta_diff)[0]
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="metadata",
            field=f"metadata.{m_field}",
            expected="keys 'reddit' and 'youtube'",
            actual_type="mismatch",
            reason=f"metadata structure mismatch on field '{m_field}'",
        )

    if platform == "reddit":
        if metadata.get("youtube") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="metadata",
                field="metadata.youtube",
                expected="null for Reddit platform",
                actual_type=type(metadata.get("youtube")).__name__,
                reason="Reddit document contains non-null metadata.youtube",
            )
        reddit_meta = metadata.get("reddit")
        if reddit_meta is not None:
            if not isinstance(reddit_meta, dict):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="metadata",
                    field="metadata.reddit",
                    expected="dict or null",
                    actual_type=type(reddit_meta).__name__,
                    reason="metadata.reddit must be a dictionary or null",
                )
            expected_reddit_keys = {"subreddit", "score"}
            r_diff = (expected_reddit_keys - set(reddit_meta.keys())) or (set(reddit_meta.keys()) - expected_reddit_keys)
            if r_diff:
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="metadata",
                    field=f"metadata.reddit.{sorted(r_diff)[0]}",
                    expected="exact keys 'subreddit' and 'score'",
                    actual_type="mismatch",
                    reason=f"metadata.reddit structure mismatch on field '{sorted(r_diff)[0]}'",
                )
            sub = reddit_meta.get("subreddit")
            if sub is not None and not isinstance(sub, str):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="metadata",
                    field="metadata.reddit.subreddit",
                    expected="string or null",
                    actual_type=type(sub).__name__,
                    reason="metadata.reddit.subreddit must be a string or null",
                )
            r_score = reddit_meta.get("score")
            if r_score is not None:
                if isinstance(r_score, bool) or not isinstance(r_score, int) or not (-2147483648 <= r_score <= 2147483647):
                    return False, None, _make_rejection(
                        doc_id=expected_doc_id,
                        dimension="metadata",
                        field="metadata.reddit.score",
                        expected="32-bit signed integer or null",
                        actual_type=type(r_score).__name__,
                        reason="metadata.reddit.score must be a 32-bit signed integer (excluding bool)",
                    )
    else:
        # YouTube platform
        if metadata.get("reddit") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="metadata",
                field="metadata.reddit",
                expected="null for YouTube platform",
                actual_type=type(metadata.get("reddit")).__name__,
                reason="YouTube document contains non-null metadata.reddit",
            )
        youtube_meta = metadata.get("youtube")
        if youtube_meta is not None:
            if not isinstance(youtube_meta, dict):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="metadata",
                    field="metadata.youtube",
                    expected="dict or null",
                    actual_type=type(youtube_meta).__name__,
                    reason="metadata.youtube must be a dictionary or null",
                )
            expected_youtube_keys = {"channel_id", "channel_title"}
            y_diff = (expected_youtube_keys - set(youtube_meta.keys())) or (set(youtube_meta.keys()) - expected_youtube_keys)
            if y_diff:
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="metadata",
                    field=f"metadata.youtube.{sorted(y_diff)[0]}",
                    expected="exact keys 'channel_id' and 'channel_title'",
                    actual_type="mismatch",
                    reason=f"metadata.youtube structure mismatch on field '{sorted(y_diff)[0]}'",
                )
            for y_key in ("channel_id", "channel_title"):
                y_val = youtube_meta.get(y_key)
                if y_val is not None and not isinstance(y_val, str):
                    return False, None, _make_rejection(
                        doc_id=expected_doc_id,
                        dimension="metadata",
                        field=f"metadata.youtube.{y_key}",
                        expected="string or null",
                        actual_type=type(y_val).__name__,
                        reason=f"metadata.youtube.{y_key} must be a string or null",
                    )

    # 9. Sentiment Status & Analytical Consistency
    sentiment = record.get("sentiment")
    if not isinstance(sentiment, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="sentiment",
            field="sentiment",
            expected="dict",
            actual_type=type(sentiment).__name__,
            reason="sentiment field must be a dictionary",
        )

    expected_sentiment_keys = {"label", "score", "status"}
    s_diff = (expected_sentiment_keys - set(sentiment.keys())) or (set(sentiment.keys()) - expected_sentiment_keys)
    if s_diff:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="sentiment",
            field=f"sentiment.{sorted(s_diff)[0]}",
            expected="canonical sentiment fields",
            actual_type="mismatch",
            reason=f"sentiment structure mismatch on field '{sorted(s_diff)[0]}'",
        )

    sent_status = sentiment.get("status")
    if sent_status not in RECOGNIZED_SENTIMENT_STATUSES:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="sentiment",
            field="sentiment.status",
            expected="recognized sentiment status",
            actual_type=type(sent_status).__name__,
            reason=f"Unrecognized sentiment status '{sent_status}'",
        )

    if sent_status == "success":
        s_label = sentiment.get("label")
        if s_label not in {"positive", "neutral", "negative"}:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="sentiment",
                field="sentiment.label",
                expected="'positive', 'neutral', or 'negative'",
                actual_type=type(s_label).__name__,
                reason=f"Invalid sentiment label '{s_label}' for success status",
            )
        s_score = sentiment.get("score")
        if (
            s_score is None
            or isinstance(s_score, bool)
            or not isinstance(s_score, (int, float))
            or not math.isfinite(s_score)
            or not (-1.0 <= s_score <= 1.0)
        ):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="sentiment",
                field="sentiment.score",
                expected="finite float in [-1.0, 1.0]",
                actual_type=type(s_score).__name__,
                reason="sentiment.score must be a finite float in [-1.0, 1.0] for success status",
            )
    else:
        # Non-success sentiment
        if sentiment.get("label") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="sentiment",
                field="sentiment.label",
                expected="null for non-success sentiment status",
                actual_type=type(sentiment.get("label")).__name__,
                reason=f"sentiment.label must be null when status is '{sent_status}'",
            )
        if sentiment.get("score") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="sentiment",
                field="sentiment.score",
                expected="null for non-success sentiment status",
                actual_type=type(sentiment.get("score")).__name__,
                reason=f"sentiment.score must be null when status is '{sent_status}'",
            )

    # 10. Topics Status & Keyword Normalization
    topics = record.get("topics")
    if not isinstance(topics, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="topics",
            field="topics",
            expected="dict",
            actual_type=type(topics).__name__,
            reason="topics field must be a dictionary",
        )

    expected_topic_keys = {"topic_id", "probability", "keywords", "status"}
    top_diff = (expected_topic_keys - set(topics.keys())) or (set(topics.keys()) - expected_topic_keys)
    if top_diff:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="topics",
            field=f"topics.{sorted(top_diff)[0]}",
            expected="canonical topics fields",
            actual_type="mismatch",
            reason=f"topics structure mismatch on field '{sorted(top_diff)[0]}'",
        )

    top_status = topics.get("status")
    if top_status not in RECOGNIZED_TOPIC_STATUSES:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="topics",
            field="topics.status",
            expected="recognized topic status",
            actual_type=type(top_status).__name__,
            reason=f"Unrecognized topic status '{top_status}'",
        )

    raw_keywords = topics.get("keywords")
    is_valid_kw, normalized_keywords, kw_err = normalize_topic_keywords(raw_keywords)
    if not is_valid_kw:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="topics",
            field="topics.keywords",
            expected="valid list of keywords",
            actual_type=type(raw_keywords).__name__,
            reason=f"Invalid topic keywords: {kw_err}",
        )

    if top_status == "assigned":
        tid = topics.get("topic_id")
        if tid is None or isinstance(tid, bool) or not isinstance(tid, int) or not (0 <= tid <= 2147483647):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="topics",
                field="topics.topic_id",
                expected="integer >= 0 (excluding bool)",
                actual_type=type(tid).__name__,
                reason="topics.topic_id must be a non-negative integer for assigned status",
            )
        t_prob = topics.get("probability")
        if t_prob is not None:
            if isinstance(t_prob, bool) or not isinstance(t_prob, (int, float)) or not math.isfinite(t_prob) or not (0.0 <= t_prob <= 1.0):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="topics",
                    field="topics.probability",
                    expected="finite float in [0.0, 1.0] or null",
                    actual_type=type(t_prob).__name__,
                    reason="topics.probability must be a finite float in [0.0, 1.0] or null",
                )
    elif top_status == "outlier":
        tid = topics.get("topic_id")
        if tid != -1 or isinstance(tid, bool):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="topics",
                field="topics.topic_id",
                expected="exactly -1 (excluding bool)",
                actual_type=type(tid).__name__,
                reason="topics.topic_id must equal -1 for outlier status",
            )
        t_prob = topics.get("probability")
        if t_prob is not None:
            if isinstance(t_prob, bool) or not isinstance(t_prob, (int, float)) or not math.isfinite(t_prob) or not (0.0 <= t_prob <= 1.0):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="topics",
                    field="topics.probability",
                    expected="finite float in [0.0, 1.0] or null",
                    actual_type=type(t_prob).__name__,
                    reason="topics.probability must be a finite float in [0.0, 1.0] or null",
                )
    else:
        # Non-assigned statuses
        if topics.get("topic_id") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="topics",
                field="topics.topic_id",
                expected="null for non-assigned topic status",
                actual_type=type(topics.get("topic_id")).__name__,
                reason=f"topics.topic_id must be null when status is '{top_status}'",
            )
        if topics.get("probability") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="topics",
                field="topics.probability",
                expected="null for non-assigned topic status",
                actual_type=type(topics.get("probability")).__name__,
                reason=f"topics.probability must be null when status is '{top_status}'",
            )
        if normalized_keywords:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="topics",
                field="topics.keywords",
                expected="empty list or null for non-assigned topic status",
                actual_type="list",
                reason=f"topics.keywords must be empty when status is '{top_status}'",
            )

    # 11. Misinformation Validation
    misinfo = record.get("misinformation")
    if not isinstance(misinfo, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="misinformation",
            field="misinformation",
            expected="dict",
            actual_type=type(misinfo).__name__,
            reason="misinformation field must be a dictionary",
        )

    expected_misinfo_keys = {"veracity_label", "credibility_score", "risk_tier", "status", "stylometric_signals"}
    mis_diff = (expected_misinfo_keys - set(misinfo.keys())) or (set(misinfo.keys()) - expected_misinfo_keys)
    if mis_diff:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="misinformation",
            field=f"misinformation.{sorted(mis_diff)[0]}",
            expected="canonical misinformation fields",
            actual_type="mismatch",
            reason=f"misinformation structure mismatch on field '{sorted(mis_diff)[0]}'",
        )

    mis_status = misinfo.get("status")
    if mis_status not in RECOGNIZED_MISINFO_STATUSES:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="misinformation",
            field="misinformation.status",
            expected="recognized misinformation status",
            actual_type=type(mis_status).__name__,
            reason=f"Unrecognized misinformation status '{mis_status}'",
        )

    if mis_status == "success":
        veracity = misinfo.get("veracity_label")
        if veracity not in VALID_VERACITY_LABELS:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.veracity_label",
                expected="'real' or 'fake'",
                actual_type=type(veracity).__name__,
                reason=f"Invalid veracity_label '{veracity}' for success status",
            )

        risk = misinfo.get("risk_tier")
        if risk not in VALID_RISK_TIERS:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.risk_tier",
                expected="'low', 'moderate', or 'high_risk'",
                actual_type=type(risk).__name__,
                reason=f"Invalid risk_tier '{risk}' for success status",
            )

        cred = misinfo.get("credibility_score")
        if (
            cred is None
            or isinstance(cred, bool)
            or not isinstance(cred, (int, float))
            or not math.isfinite(cred)
            or not (0.0 <= cred <= 1.0)
        ):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.credibility_score",
                expected="finite float in [0.0, 1.0]",
                actual_type=type(cred).__name__,
                reason="credibility_score must be a finite float in [0.0, 1.0] for success status",
            )

        signals = misinfo.get("stylometric_signals")
        if not isinstance(signals, dict):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.stylometric_signals",
                expected="dictionary of 8 stylometric signals",
                actual_type=type(signals).__name__,
                reason="stylometric_signals must be a dictionary for success status",
            )

        sig_diff = (EXPECTED_STYLOMETRIC_SIGNALS - set(signals.keys())) or (set(signals.keys()) - EXPECTED_STYLOMETRIC_SIGNALS)
        if sig_diff:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field=f"misinformation.stylometric_signals.{sorted(sig_diff)[0]}",
                expected="canonical 8 stylometric signal keys",
                actual_type="mismatch",
                reason=f"stylometric_signals structure mismatch on key '{sorted(sig_diff)[0]}'",
            )

        for sig_name in EXPECTED_STYLOMETRIC_SIGNALS:
            sig_val = signals[sig_name]
            if sig_val is None or isinstance(sig_val, bool) or not isinstance(sig_val, (int, float)) or not math.isfinite(sig_val):
                return False, None, _make_rejection(
                    doc_id=expected_doc_id,
                    dimension="misinformation",
                    field=f"misinformation.stylometric_signals.{sig_name}",
                    expected="finite float",
                    actual_type=type(sig_val).__name__,
                    reason=f"stylometric signal '{sig_name}' must be a finite float",
                )
            if sig_name in {"uppercase_char_ratio", "lexical_subjectivity_ratio", "digits_ratio", "all_caps_word_ratio"}:
                if not (0.0 <= sig_val <= 1.0):
                    return False, None, _make_rejection(
                        doc_id=expected_doc_id,
                        dimension="misinformation",
                        field=f"misinformation.stylometric_signals.{sig_name}",
                        expected="ratio in [0.0, 1.0]",
                        actual_type="float",
                        reason=f"stylometric ratio '{sig_name}' must be in [0.0, 1.0]",
                    )
            else:
                if sig_val < 0.0:
                    return False, None, _make_rejection(
                        doc_id=expected_doc_id,
                        dimension="misinformation",
                        field=f"misinformation.stylometric_signals.{sig_name}",
                        expected="value >= 0.0",
                        actual_type="float",
                        reason=f"stylometric intensity/count '{sig_name}' must be >= 0.0",
                    )
    else:
        # Non-success misinformation
        if misinfo.get("veracity_label") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.veracity_label",
                expected="null for non-success misinformation",
                actual_type=type(misinfo.get("veracity_label")).__name__,
                reason=f"veracity_label must be null when status is '{mis_status}'",
            )
        if misinfo.get("risk_tier") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.risk_tier",
                expected="null for non-success misinformation",
                actual_type=type(misinfo.get("risk_tier")).__name__,
                reason=f"risk_tier must be null when status is '{mis_status}'",
            )
        if misinfo.get("credibility_score") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.credibility_score",
                expected="null for non-success misinformation",
                actual_type=type(misinfo.get("credibility_score")).__name__,
                reason=f"credibility_score must be null when status is '{mis_status}'",
            )
        if misinfo.get("stylometric_signals") is not None:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="misinformation",
                field="misinformation.stylometric_signals",
                expected="null for non-success misinformation",
                actual_type=type(misinfo.get("stylometric_signals")).__name__,
                reason=f"stylometric_signals must be null when status is '{mis_status}'",
            )

    # 12. Consolidation & Provenance Validation
    consolidation = record.get("consolidation")
    if not isinstance(consolidation, dict):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="consolidation",
            field="consolidation",
            expected="dict",
            actual_type=type(consolidation).__name__,
            reason="consolidation field must be a dictionary",
        )

    expected_con_keys = {"status", "is_complete", "sources_joined"}
    con_diff = (expected_con_keys - set(consolidation.keys())) or (set(consolidation.keys()) - expected_con_keys)
    if con_diff:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="consolidation",
            field=f"consolidation.{sorted(con_diff)[0]}",
            expected="canonical consolidation fields",
            actual_type="mismatch",
            reason=f"consolidation structure mismatch on field '{sorted(con_diff)[0]}'",
        )

    con_status = consolidation.get("status")
    if con_status not in {"complete", "degraded", "partial", "unprocessed"}:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="consolidation",
            field="consolidation.status",
            expected="one of complete, degraded, partial, unprocessed",
            actual_type=type(con_status).__name__,
            reason=f"Invalid consolidation status '{con_status}'",
        )

    is_complete = consolidation.get("is_complete")
    if not isinstance(is_complete, bool):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="consolidation",
            field="consolidation.is_complete",
            expected="boolean",
            actual_type=type(is_complete).__name__,
            reason="consolidation.is_complete must be a boolean",
        )

    sources_joined = consolidation.get("sources_joined")
    if not isinstance(sources_joined, list) or not all(isinstance(s, str) for s in sources_joined):
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="consolidation",
            field="consolidation.sources_joined",
            expected="list of strings",
            actual_type=type(sources_joined).__name__,
            reason="consolidation.sources_joined must be a list of strings",
        )

    # Core content completeness check
    is_content_complete = (
        prep_status == "success"
        and sent_status == "success"
        and top_status in {"assigned", "outlier"}
        and mis_status == "success"
    )

    if con_status == "complete":
        if not is_complete:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="consolidation",
                field="consolidation.is_complete",
                expected="True when status is 'complete'",
                actual_type="bool",
                reason="consolidation.is_complete must be True when status is 'complete'",
            )
        if not is_content_complete:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="consolidation",
                field="consolidation.status",
                expected="all four analytical content statuses must be successful",
                actual_type="mismatch",
                reason="consolidation.status cannot be 'complete' when analytical components failed",
            )
        if not REQUIRED_FOUR_CONTENT_SOURCES.issubset(set(sources_joined)):
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="consolidation",
                field="consolidation.sources_joined",
                expected="must include all four content sources: preprocessing, sentiment, topics, misinformation",
                actual_type="list",
                reason="complete record must include all four content sources in sources_joined",
            )
    else:
        if is_complete:
            return False, None, _make_rejection(
                doc_id=expected_doc_id,
                dimension="consolidation",
                field="consolidation.is_complete",
                expected="False when status is not 'complete'",
                actual_type="bool",
                reason="consolidation.is_complete must be False when status is not 'complete'",
            )

    # Graph provenance check
    if graph_status in {"unavailable", "corrupted"} and "graph" in sources_joined:
        return False, None, _make_rejection(
            doc_id=expected_doc_id,
            dimension="consolidation",
            field="consolidation.sources_joined",
            expected="must not contain 'graph' when graph is unavailable or corrupted",
            actual_type="list",
            reason="sources_joined cannot include 'graph' when author.graph_status is unavailable or corrupted",
        )

    # Construct safe, normalized record copy
    normalized_doc = {
        "document_id": expected_doc_id,
        "platform": platform,
        "native_id": native_id,
        "url": url_val,
        "timestamps": dict(timestamps),
        "author": dict(author),
        "content": dict(content),
        "engagement": dict(engagement),
        "metadata": {
            "reddit": dict(metadata["reddit"]) if isinstance(metadata.get("reddit"), dict) else None,
            "youtube": dict(metadata["youtube"]) if isinstance(metadata.get("youtube"), dict) else None,
        },
        "sentiment": dict(sentiment),
        "topics": {
            "topic_id": topics["topic_id"],
            "probability": topics["probability"],
            "keywords": normalized_keywords,
            "status": topics["status"],
        },
        "misinformation": {
            "veracity_label": misinfo["veracity_label"],
            "credibility_score": misinfo["credibility_score"],
            "risk_tier": misinfo["risk_tier"],
            "status": misinfo["status"],
            "stylometric_signals": dict(misinfo["stylometric_signals"]) if isinstance(misinfo.get("stylometric_signals"), dict) else None,
        },
        "consolidation": {
            "status": consolidation["status"],
            "is_complete": consolidation["is_complete"],
            "sources_joined": list(sources_joined),
        },
    }

    return True, normalized_doc, None


def index_consolidated_dataset(
    dataset_path: Union[str, Path] = "data/processed/consolidated_intelligence.json",
    client: Optional[ElasticsearchClient] = None,
    index_name: str = DEFAULT_INDEX_NAME,
    refresh: Optional[Union[str, bool]] = False,
    ensure_index: bool = True,
    batch_size: int = DEFAULT_BULK_BATCH_SIZE,
    validate_envelope: bool = True,
) -> Dict[str, Any]:
    """Indexes a consolidated intelligence dataset into Elasticsearch with strict pre-validation.

    Order of Execution & Accounting Contracts:
    1. Parameter validation (refresh, batch_size, validate_envelope executed first).
    2. Input file and envelope loading & pre-validation:
       - Schema-v2 validation is enabled by default (validate_envelope=True).
       - If envelope validation fails but a valid records list exists, the readable record count
         is preserved and all records are accounted as failed (total == failed, indexed == 0).
       - If the records collection is missing or malformed, total is unknown (0), and an explicit
         artifact-level failure is returned.
       - No Elasticsearch client initialization or driver operations occur if envelope validation fails.
       - Legacy envelopes require an explicit opt-out via validate_envelope=False.
    3. Elasticsearch client initialization and preflight (ping, index check):
       - Once envelope validation succeeds and establishes total_records = len(records), any
         preflight failure triggers the strict all-records-failed accounting contract:
         total_records == failed_count, indexed_count == 0, rejected_count == 0, status == 'failed'.
    4. Per-record validation, topic keyword normalization, and bounded bulk indexing.
    5. Clean summary aggregation and client lifecycle cleanup.

    Mathematical Count Invariant:
    Under all execution paths: total_records == indexed_count + rejected_count + failed_count.

    Args:
        dataset_path: Path to consolidated intelligence JSON file.
        client: Optional preconfigured ElasticsearchClient instance.
        index_name: Target index name (defaults to DEFAULT_INDEX_NAME).
        refresh: Refresh control parameter (True, False, 'wait_for', or None).
        ensure_index: Whether to verify or create index before document ingestion.
        batch_size: Bounded batch size for bulk indexing (positive integer > 0).
        validate_envelope: Whether to strictly validate Schema-v2 envelope metadata (default: True).
            To process unvalidated legacy envelopes, caller must explicitly pass validate_envelope=False.

    Returns:
        Structured Indexing Summary dictionary.

    Raises:
        ValueError: If refresh parameter, index_name, or batch_size is invalid.
        TypeError: If validate_envelope is not a boolean.
    """
    # 1. Parameter Validation (Executed First)
    if not _is_valid_refresh(refresh):
        raise ValueError("Invalid refresh parameter. Accepted values are True, False, 'wait_for', or None.")

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be an integer > 0, got {batch_size!r}.")

    if not isinstance(validate_envelope, bool):
        raise TypeError(f"validate_envelope must be a boolean, got {type(validate_envelope).__name__}.")

    started_at = datetime.now(timezone.utc).isoformat()
    target_index = validate_index_name(index_name or DEFAULT_INDEX_NAME)

    def _build_summary(
        status: str,
        total: Optional[int],
        indexed: int,
        rejected: int,
        failed: int,
        rejections: List[Dict[str, Any]],
        failures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        completed_at = datetime.now(timezone.utc).isoformat()
        try:
            d_start = datetime.fromisoformat(started_at)
            d_comp = datetime.fromisoformat(completed_at)
            duration = round((d_comp - d_start).total_seconds(), 4)
        except Exception:
            duration = 0.0

        if total is not None:
            assert total == indexed + rejected + failed, (
                f"Count invariant violated: total {total} != indexed {indexed} + rejected {rejected} + failed {failed}"
            )

        return {
            "status": status,
            "index_name": target_index,
            "total_records": total,
            "indexed_count": indexed,
            "rejected_count": rejected,
            "failed_count": failed,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration,
            "rejections": rejections,
            "indexing_failures": failures,
        }

    # 2. File & Envelope Loading
    path_obj = Path(dataset_path)
    if not path_obj.exists():
        logger.error("Dataset file not found: %s", path_obj)
        return _build_summary(
            status="failed",
            total=None,
            indexed=0,
            rejected=0,
            failed=0,
            rejections=[],
            failures=[{"status": "failed", "reason": f"Dataset file not found: {path_obj}"}],
        )

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                logger.error("Dataset file is empty: %s", path_obj)
                return _build_summary(
                    status="failed",
                    total=None,
                    indexed=0,
                    rejected=0,
                    failed=0,
                    rejections=[],
                    failures=[{"status": "failed", "reason": "Dataset file is empty"}],
                )
            data = json.loads(content)
    except Exception as exc:
        logger.error("Failed to parse JSON dataset %s (%s)", path_obj, type(exc).__name__)
        return _build_summary(
            status="failed",
            total=None,
            indexed=0,
            rejected=0,
            failed=0,
            rejections=[],
            failures=[{"status": "failed", "reason": f"Failed to parse JSON dataset ({type(exc).__name__})"}],
        )

    if not isinstance(data, dict):
        logger.error("Dataset root is not a dictionary")
        return _build_summary(
            status="failed",
            total=None,
            indexed=0,
            rejected=0,
            failed=0,
            rejections=[],
            failures=[{"status": "failed", "reason": "Dataset root is not a dictionary"}],
        )

    raw_records = data.get("records")
    has_readable_records = isinstance(raw_records, list)
    total_records: Optional[int] = len(raw_records) if has_readable_records else None

    # Schema-v2 Envelope Pre-Validation (Default Enabled)
    if validate_envelope:
        try:
            ConsolidatedArtifactValidator.validate_consolidated_artifact(data)
        except (ValueError, TypeError) as exc:
            logger.error("Schema-v2 envelope validation failed: %s", exc)
            if has_readable_records:
                # Envelope invalid, but readable records exist: account all readable records as failed
                return _build_summary(
                    status="failed",
                    total=total_records,
                    indexed=0,
                    rejected=0,
                    failed=total_records if total_records is not None else 0,
                    rejections=[],
                    failures=[{"status": "failed", "reason": f"Schema-v2 validation failed: {exc}"}],
                )
            else:
                # Missing or malformed records collection: explicit artifact-level failure with total_records=None
                return _build_summary(
                    status="failed",
                    total=None,
                    indexed=0,
                    rejected=0,
                    failed=0,
                    rejections=[],
                    failures=[{"status": "failed", "reason": f"Schema-v2 validation failed: {exc}"}],
                )

    # If validate_envelope is False (explicit opt-out), ensure readable records collection exists
    if not has_readable_records:
        logger.error("Dataset envelope missing 'records' list")
        return _build_summary(
            status="failed",
            total=None,
            indexed=0,
            rejected=0,
            failed=0,
            rejections=[],
            failures=[{"status": "failed", "reason": "Dataset envelope missing 'records' list"}],
        )

    if total_records == 0:
        return _build_summary(
            status="empty",
            total=0,
            indexed=0,
            rejected=0,
            failed=0,
            rejections=[],
            failures=[],
        )

    records = raw_records

    # 3. Client Acquisition & Preflight Verification
    owns_client = False
    es_client: Optional[ElasticsearchClient] = client

    if es_client is None:
        try:
            es_client = create_elasticsearch_client()
            owns_client = True
        except Exception as exc:
            logger.error("Client initialization failed (%s)", type(exc).__name__)
            return _build_summary(
                status="failed",
                total=total_records,
                indexed=0,
                rejected=0,
                failed=total_records,
                rejections=[],
                failures=[{"status": "failed", "reason": f"Elasticsearch client initialization failed ({type(exc).__name__})"}],
            )

    try:
        # Preflight ping
        try:
            is_pingable = es_client.ping()
            if not is_pingable:
                logger.error("Preflight ping returned False")
                return _build_summary(
                    status="failed",
                    total=total_records,
                    indexed=0,
                    rejected=0,
                    failed=total_records,
                    rejections=[],
                    failures=[{"status": "failed", "reason": "Cluster ping returned False"}],
                )
        except Exception as exc:
            logger.error("Preflight ping raised exception (%s)", type(exc).__name__)
            return _build_summary(
                status="failed",
                total=total_records,
                indexed=0,
                rejected=0,
                failed=total_records,
                rejections=[],
                failures=[{"status": "failed", "reason": f"Cluster ping failed ({type(exc).__name__})"}],
            )

        # Index verification / creation
        if ensure_index:
            try:
                es_client.create_index(index_name=target_index, force_recreate=False)
            except Exception as exc:
                logger.error("Preflight index creation failed (%s)", type(exc).__name__)
                return _build_summary(
                    status="failed",
                    total=total_records,
                    indexed=0,
                    rejected=0,
                    failed=total_records,
                    rejections=[],
                    failures=[{"status": "failed", "reason": f"Index creation failed ({type(exc).__name__})"}],
                )
        else:
            try:
                exists = es_client.index_exists(target_index)
                if not exists:
                    logger.error("Target index '%s' does not exist", target_index)
                    return _build_summary(
                        status="failed",
                        total=total_records,
                        indexed=0,
                        rejected=0,
                        failed=total_records,
                        rejections=[],
                        failures=[{"status": "failed", "reason": f"Target index '{target_index}' does not exist"}],
                    )
            except Exception as exc:
                logger.error("Preflight index check raised exception (%s)", type(exc).__name__)
                return _build_summary(
                    status="failed",
                    total=total_records,
                    indexed=0,
                    rejected=0,
                    failed=total_records,
                    rejections=[],
                    failures=[{"status": "failed", "reason": f"Index existence check failed ({type(exc).__name__})"}],
                )

        # 4. Record Validation, Topic Keyword Normalization, and Bounded Bulk Indexing
        indexed_count = 0
        rejected_count = 0
        failed_count = 0
        rejections: List[Dict[str, Any]] = []
        indexing_failures: List[Dict[str, Any]] = []

        # Phase A: Validate & Normalize records
        valid_docs: List[Dict[str, Any]] = []
        for rec in records:
            is_valid, normalized_doc, rejection_entry = validate_and_normalize_record(rec)
            if not is_valid or normalized_doc is None:
                rejected_count += 1
                if rejection_entry:
                    rejections.append(rejection_entry)
            else:
                valid_docs.append(normalized_doc)

        # Phase B: Bounded Bulk Ingestion
        for i in range(0, len(valid_docs), batch_size):
            batch = valid_docs[i : i + batch_size]
            batch_indexed = False
            if hasattr(es_client, "bulk_index_documents") and callable(es_client.bulk_index_documents):
                try:
                    res = es_client.bulk_index_documents(
                        documents=batch,
                        index_name=target_index,
                        refresh=refresh,
                        raise_on_error=False,
                    )
                    if isinstance(res, dict):
                        batch_indexed = True
                        indexed_count += int(res.get("indexed_count", 0))
                        failed_count += int(res.get("failed_count", 0))
                        for item_failure in res.get("failed_items", []):
                            err_msg = str(item_failure.get("error", "Unknown error"))
                            reason = err_msg if err_msg.startswith("Indexing operation failed") else f"Indexing operation failed: {err_msg}"
                            indexing_failures.append({
                                "document_id": item_failure.get("document_id", "unknown"),
                                "status": "failed",
                                "reason": reason,
                            })
                except Exception as exc:
                    logger.error("Batch indexing operation failed (%s)", type(exc).__name__)
                    failed_count += len(batch)
                    for doc in batch:
                        indexing_failures.append({
                            "document_id": doc.get("document_id", "unknown"),
                            "status": "failed",
                            "reason": f"Indexing operation failed ({type(exc).__name__})",
                        })
                    batch_indexed = True

            if not batch_indexed:
                # Fallback for bare mock clients that only configure index_document
                for doc in batch:
                    doc_id = doc.get("document_id", "unknown")
                    try:
                        es_client.index_document(
                            document=doc,
                            index_name=target_index,
                            refresh=refresh,
                        )
                        indexed_count += 1
                    except Exception as exc:
                        logger.error("Indexing failed for doc '%s' (%s)", doc_id, type(exc).__name__)
                        failed_count += 1
                        indexing_failures.append({
                            "document_id": doc_id,
                            "status": "failed",
                            "reason": f"Indexing operation failed ({type(exc).__name__})",
                        })

        # Calculate final overall status
        if indexed_count == total_records:
            overall_status = "success"
        elif indexed_count > 0:
            overall_status = "partial_success"
        else:
            overall_status = "failed"

        # Mathematical count invariant assertion
        assert total_records == indexed_count + rejected_count + failed_count, (
            f"Count invariant violated: total {total_records} != "
            f"indexed {indexed_count} + rejected {rejected_count} + failed {failed_count}"
        )

        return _build_summary(
            status=overall_status,
            total=total_records,
            indexed=indexed_count,
            rejected=rejected_count,
            failed=failed_count,
            rejections=rejections,
            failures=indexing_failures,
        )

    finally:
        if owns_client and es_client is not None:
            try:
                es_client.close()
            except Exception as exc:
                logger.warning("Error closing Elasticsearch client (%s)", type(exc).__name__)

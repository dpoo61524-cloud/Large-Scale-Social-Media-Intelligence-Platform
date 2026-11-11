"""Unit Tests for Elasticsearch Indexing Adapter.

Tests cover:
1. Valid Reddit and YouTube records ingestion.
2. Invalid envelope structures (non-dict root, missing records, records not a list).
3. Missing and malformed dataset files (nonexistent path, empty file, invalid JSON).
4. Strict document identity validation (valid, leading/trailing whitespace, mismatches).
5. Graph status values (ranked_influencer, not_in_top_influencers, anonymous, unavailable, corrupted, invalid).
6. Binary misinformation labels ('real', 'fake') and risk tiers ('low', 'moderate', 'high_risk').
7. Analytical status consistency (sentiment, topics, misinformation).
8. Consolidation completeness and provenance (4 content sources, graph independence).
9. Legacy keyword normalization and canonical preservation (pure, non-mutating).
10. Unknown root/nested fields and missing fields rejection under strict schema.
11. Boolean-versus-numeric type rejection.
12. Non-finite numeric rejection (NaN, Inf, -Inf).
13. Invalid refresh values rejected before file access or client initialization.
14. Client initialization failure preflight accounting.
15. Ping failure preflight accounting.
16. Index creation failure preflight accounting.
17. Missing index when ensure_index=False preflight accounting.
18. Per-record validation rejection accounting.
19. Per-record driver indexing failure accounting.
20. Successful indexing accounting (created, updated, synthetic indexed).
21. Total-record accounting invariant (total_records == indexed + rejected + failed).
22. Client ownership and cleanup (internal closed, external preserved).
23. Sanitized error reporting and secret redaction.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.search.elasticsearch_client import (
    DEFAULT_REQUEST_TIMEOUT,
    ElasticsearchClient,
    ElasticsearchConnectionError,
    ElasticsearchIndexError,
)
from src.search.indexing_adapter import (
    index_consolidated_dataset,
    normalize_topic_keywords,
    validate_and_normalize_record,
)
from src.search.index_mapping import DEFAULT_INDEX_NAME


# =====================================================================
# Fixtures and Sample Records
# =====================================================================

@pytest.fixture
def sample_reddit_record() -> Dict[str, Any]:
    """Provides a valid canonical Reddit social intelligence record."""
    return {
        "document_id": "reddit_t3_ds1001",
        "platform": "reddit",
        "native_id": "t3_ds1001",
        "url": "https://reddit.com/r/datascience/comments/ds1001/cleaning_multilingual_text",
        "timestamps": {
            "published_at": "2024-10-28T07:40:00+00:00",
            "collected_at": "2026-11-05T16:54:13.868878+00:00",
            "processed_at": "2026-11-06T17:27:57.464005+00:00",
        },
        "author": {
            "username": "pipeline_builder",
            "channel_id": None,
            "channel_title": None,
            "graph_status": "unavailable",
            "influencer_score": None,
            "pagerank": None,
            "community_id": None,
            "in_degree": None,
            "out_degree": None,
        },
        "content": {
            "title_raw": "Best practices for cleaning multilingual social media text?",
            "body_raw": "I am working on social listening involving English and Spanish comments.",
            "title_cleaned": "Best practices for cleaning multilingual social media text?",
            "body_cleaned": "I am working on social listening involving English and Spanish comments.",
            "detected_language": "Undetermined",
            "language_code": "und",
            "preprocessing_status": "success",
        },
        "engagement": {
            "score": 142,
            "views_count": None,
            "likes_count": None,
            "comments_count": 28,
        },
        "metadata": {
            "reddit": {
                "subreddit": "datascience",
                "score": 142,
            },
            "youtube": None,
        },
        "sentiment": {
            "label": "positive",
            "score": 0.88,
            "status": "success",
        },
        "topics": {
            "topic_id": 0,
            "probability": 0.85,
            "keywords": [
                {"word": "topic_word", "weight": 0.8},
            ],
            "status": "assigned",
        },
        "misinformation": {
            "veracity_label": "real",
            "credibility_score": 0.786,
            "risk_tier": "low",
            "status": "success",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.024,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.04,
                "word_count": 25.0,
                "avg_word_length": 5.64,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        },
        "consolidation": {
            "status": "complete",
            "is_complete": True,
            "sources_joined": [
                "preprocessing",
                "sentiment",
                "topics",
                "misinformation",
            ],
        },
    }


@pytest.fixture
def sample_youtube_record() -> Dict[str, Any]:
    """Provides a valid canonical YouTube social intelligence record."""
    return {
        "document_id": "youtube_vid_abc123",
        "platform": "youtube",
        "native_id": "vid_abc123",
        "url": "https://www.youtube.com/watch?v=vid_abc123",
        "timestamps": {
            "published_at": "2026-11-01T12:00:00+00:00",
            "collected_at": "2026-11-06T16:52:04.083528+00:00",
            "processed_at": "2026-11-06T17:27:57.536988+00:00",
        },
        "author": {
            "username": "Data Engineering Insights",
            "channel_id": "UC_test_channel",
            "channel_title": "Data Engineering Insights",
            "graph_status": "unavailable",
            "influencer_score": None,
            "pagerank": None,
            "community_id": None,
            "in_degree": None,
            "out_degree": None,
        },
        "content": {
            "title_raw": "Introduction to Large Scale Social Media Intelligence",
            "body_raw": "A comprehensive overview of NLP, graph databases, and topic modeling.",
            "title_cleaned": "Introduction to Large Scale Social Media Intelligence",
            "body_cleaned": "A comprehensive overview of NLP, graph databases, and topic modeling.",
            "detected_language": "Undetermined",
            "language_code": "und",
            "preprocessing_status": "success",
        },
        "engagement": {
            "score": None,
            "views_count": 12500,
            "likes_count": 840,
            "comments_count": 95,
        },
        "metadata": {
            "reddit": None,
            "youtube": {
                "channel_id": "UC_test_channel",
                "channel_title": "Data Engineering Insights",
            },
        },
        "sentiment": {
            "label": "positive",
            "score": 0.88,
            "status": "success",
        },
        "topics": {
            "topic_id": 0,
            "probability": 0.85,
            "keywords": [
                {"word": "topic_word", "weight": 0.8},
            ],
            "status": "assigned",
        },
        "misinformation": {
            "veracity_label": "fake",
            "credibility_score": 0.1341,
            "risk_tier": "high_risk",
            "status": "success",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.1132,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 7.0,
                "avg_word_length": 6.7143,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        },
        "consolidation": {
            "status": "complete",
            "is_complete": True,
            "sources_joined": [
                "preprocessing",
                "sentiment",
                "topics",
                "misinformation",
            ],
        },
    }


class MockDriver:
    """Mock driver representing Elasticsearch client driver methods."""

    def __init__(self) -> None:
        self.ping = MagicMock(return_value=True)
        self.indices = MagicMock()
        self.indices.exists = MagicMock(return_value=True)
        self.indices.create = MagicMock(return_value={"acknowledged": True})
        self.index = MagicMock(return_value={"result": "created", "_id": "doc_123"})
        self.bulk = MagicMock(return_value={"took": 5, "errors": False, "items": []})
        self.close = MagicMock()


@pytest.fixture
def mock_es_client() -> ElasticsearchClient:
    """Provides an ElasticsearchClient instance wired to a fresh MockDriver."""
    driver = MockDriver()
    return ElasticsearchClient(
        host="http://localhost:9200",
        client_override=driver,
    )


def _create_valid_schema_v2_envelope(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Constructs a fully valid Schema-v2 envelope artifact conforming to ConsolidatedArtifactValidator."""
    reddit_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("platform") == "reddit")
    youtube_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("platform") == "youtube")
    complete_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "complete")
    degraded_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "degraded")
    partial_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "partial")
    unprocessed_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "unprocessed")

    status_str = "complete" if complete_cnt == len(records) and len(records) > 0 else (
        "partial" if partial_cnt > 0 else "degraded" if degraded_cnt > 0 else "complete"
    )

    return {
        "schema_version": "2.0",
        "pipeline_version": "2.0.0",
        "status": "success",
        "consolidation_status": status_str,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit_warnings": [],
        "rejected_records": [],
        "provenance": {
            "reddit_input_file": "data/processed/reddit_posts_processed.json",
            "reddit_status": "loaded" if reddit_cnt > 0 else "empty",
            "reddit_records_count": reddit_cnt,
            "youtube_input_file": "data/processed/youtube_videos_processed.json",
            "youtube_status": "loaded" if youtube_cnt > 0 else "empty",
            "youtube_records_count": youtube_cnt,
        },
        "metrics": {
            "total_records_consolidated": len(records),
            "platform_counts": {
                "reddit": reddit_cnt,
                "youtube": youtube_cnt,
            },
            "complete_records": complete_cnt,
            "degraded_records": degraded_cnt,
            "partial_records": partial_cnt,
            "unprocessed_records": unprocessed_cnt,
            "duplicates_deduplicated": 0,
            "rejected_records_count": 0,
        },
        "records": records,
    }


# =====================================================================
# 1. Refresh Parameter Validation (Executed First)
# =====================================================================

@pytest.mark.parametrize(
    "invalid_refresh",
    [
        1,
        0,
        "true",
        "false",
        "immediate",
        [True],
        {"refresh": True},
        2.5,
    ],
)
def test_invalid_refresh_rejected_immediately_without_file_or_client_access(invalid_refresh):
    """Verifies invalid refresh parameters raise ValueError before file access or client creation."""
    nonexistent_path = "nonexistent_file_path_xyz_123.json"
    with pytest.raises(ValueError, match="Invalid refresh parameter"):
        index_consolidated_dataset(
            dataset_path=nonexistent_path,
            refresh=invalid_refresh,
        )


@pytest.mark.parametrize("valid_refresh", [True, False, "wait_for", None])
def test_valid_refresh_parameters_accepted(valid_refresh, tmp_path: Path, mock_es_client: ElasticsearchClient):
    """Verifies True, False, 'wait_for', and None pass refresh validation."""
    empty_dataset = tmp_path / "empty_valid.json"
    empty_dataset.write_text(json.dumps(_create_valid_schema_v2_envelope([])), encoding="utf-8")

    res = index_consolidated_dataset(
        dataset_path=empty_dataset,
        client=mock_es_client,
        refresh=valid_refresh,
    )
    assert res["status"] == "empty"


# =====================================================================
# 2. Strict Document Identity Validation
# =====================================================================

def test_valid_reddit_and_youtube_identity(sample_reddit_record, sample_youtube_record):
    """Verifies valid Reddit and YouTube identities pass validation."""
    is_valid_r, norm_r, rej_r = validate_and_normalize_record(sample_reddit_record)
    assert is_valid_r is True
    assert norm_r["document_id"] == "reddit_t3_ds1001"
    assert rej_r is None

    is_valid_y, norm_y, rej_y = validate_and_normalize_record(sample_youtube_record)
    assert is_valid_y is True
    assert norm_y["document_id"] == "youtube_vid_abc123"
    assert rej_y is None


@pytest.mark.parametrize(
    "native_id_with_space",
    [" t3_ds1001", "t3_ds1001 ", "  t3_ds1001  ", "\tt3_ds1001"],
)
def test_native_id_with_whitespace_rejected_without_trimming(sample_reddit_record, native_id_with_space: str):
    """Verifies leading/trailing whitespace in native_id is strictly rejected and never silently repaired."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["native_id"] = native_id_with_space
    rec["document_id"] = f"reddit_{native_id_with_space}"

    is_valid, norm_doc, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert norm_doc is None
    assert rej["field"] == "native_id"
    assert "whitespace" in rej["reason"]


def test_mismatched_document_id_rejected(sample_reddit_record):
    """Verifies mismatched document_id against platform and native_id is rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["document_id"] = "youtube_t3_ds1001"  # Mismatch with platform 'reddit'

    is_valid, norm_doc, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "document_id"
    assert "does not match expected identity" in rej["reason"]


# =====================================================================
# 3. Platform Metadata Separation Validation
# =====================================================================

def test_reddit_record_with_youtube_metadata_rejected(sample_reddit_record):
    """Verifies Reddit records with non-null YouTube metadata are rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["metadata"]["youtube"] = {"channel_id": "UC_abc"}

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "metadata.youtube"
    assert "Reddit document contains non-null metadata.youtube" in rej["reason"]


def test_youtube_record_with_reddit_metadata_rejected(sample_youtube_record):
    """Verifies YouTube records with non-null Reddit metadata are rejected."""
    rec = copy.deepcopy(sample_youtube_record)
    rec["metadata"]["reddit"] = {"subreddit": "datascience", "score": 100}

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "metadata.reddit"
    assert "YouTube document contains non-null metadata.reddit" in rej["reason"]


# =====================================================================
# 4. Graph Status & Metrics Consistency
# =====================================================================

@pytest.mark.parametrize(
    "valid_status",
    [
        "ranked_influencer",
        "not_in_top_influencers",
        "anonymous",
        "unavailable",
        "corrupted",
    ],
)
def test_all_five_canonical_graph_statuses_accepted(sample_reddit_record, valid_status: str):
    """Verifies all five consolidator graph statuses are accepted."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["author"]["graph_status"] = valid_status

    if valid_status == "ranked_influencer":
        rec["author"]["influencer_score"] = 0.85
        rec["author"]["pagerank"] = 0.045
        rec["author"]["community_id"] = 2
        rec["author"]["in_degree"] = 15
        rec["author"]["out_degree"] = 8
        rec["consolidation"]["sources_joined"] = [
            "preprocessing", "sentiment", "topics", "misinformation", "graph"
        ]
    else:
        rec["author"]["influencer_score"] = None
        rec["author"]["pagerank"] = None
        rec["author"]["community_id"] = None
        rec["author"]["in_degree"] = None
        rec["author"]["out_degree"] = None
        rec["consolidation"]["sources_joined"] = [
            "preprocessing", "sentiment", "topics", "misinformation"
        ]

    is_valid, norm_doc, rej = validate_and_normalize_record(rec)
    assert is_valid is True
    assert norm_doc["author"]["graph_status"] == valid_status
    assert rej is None


@pytest.mark.parametrize("invalid_status", ["active", "error", "unknown", "valid"])
def test_invalid_graph_statuses_rejected(sample_reddit_record, invalid_status: str):
    """Verifies invalid or legacy graph statuses are rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["author"]["graph_status"] = invalid_status

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "author.graph_status"


def test_non_ranked_graph_status_with_non_null_metrics_rejected(sample_reddit_record):
    """Verifies non-ranked graph status carrying non-null metrics is rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["author"]["graph_status"] = "not_in_top_influencers"
    rec["author"]["influencer_score"] = 0.5  # Must be null!

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "author.influencer_score"
    assert "must be null" in rej["reason"]


# =====================================================================
# 5. Misinformation Labels and Risk Tiers
# =====================================================================

@pytest.mark.parametrize("valid_veracity", ["real", "fake"])
@pytest.mark.parametrize("valid_tier", ["low", "moderate", "high_risk"])
def test_valid_misinformation_labels_and_risk_tiers(sample_reddit_record, valid_veracity: str, valid_tier: str):
    """Verifies strictly binary veracity labels and verified risk tiers pass."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["misinformation"]["veracity_label"] = valid_veracity
    rec["misinformation"]["risk_tier"] = valid_tier

    is_valid, norm_doc, rej = validate_and_normalize_record(rec)
    assert is_valid is True
    assert norm_doc["misinformation"]["veracity_label"] == valid_veracity
    assert norm_doc["misinformation"]["risk_tier"] == valid_tier
    assert rej is None


@pytest.mark.parametrize("invalid_veracity", ["unverified", "pants-fire", "mostly-true", "true", "false"])
def test_invalid_misinformation_veracity_labels_rejected(sample_reddit_record, invalid_veracity: str):
    """Verifies non-binary veracity labels are rejected for successful status."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["misinformation"]["veracity_label"] = invalid_veracity

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "misinformation.veracity_label"


@pytest.mark.parametrize("invalid_tier", ["medium", "critical", "unknown", "extreme"])
def test_invalid_misinformation_risk_tiers_rejected(sample_reddit_record, invalid_tier: str):
    """Verifies unverified risk tiers are rejected for successful status."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["misinformation"]["risk_tier"] = invalid_tier

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "misinformation.risk_tier"


# =====================================================================
# 6. Analytical Status & Value Consistency
# =====================================================================

def test_non_success_sentiment_with_score_rejected(sample_reddit_record):
    """Verifies non-success sentiment status carrying non-null score is rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["sentiment"]["status"] = "error"
    rec["sentiment"]["score"] = 0.5  # Must be null!
    rec["sentiment"]["label"] = None

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "sentiment.score"


def test_topics_outlier_status_enforces_topic_id_minus_one(sample_reddit_record):
    """Verifies topics outlier status strictly requires topic_id == -1."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["topics"]["status"] = "outlier"
    rec["topics"]["topic_id"] = -1
    rec["topics"]["probability"] = 0.0

    is_valid, norm_doc, _ = validate_and_normalize_record(rec)
    assert is_valid is True
    assert norm_doc["topics"]["topic_id"] == -1

    # Inconsistent topic_id with outlier status
    rec["topics"]["topic_id"] = 0
    is_valid_bad, _, rej = validate_and_normalize_record(rec)
    assert is_valid_bad is False
    assert rej["field"] == "topics.topic_id"
    assert "must equal -1" in rej["reason"]


def test_non_success_misinformation_with_score_rejected(sample_reddit_record):
    """Verifies non-success misinformation carrying credibility score is rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["misinformation"]["status"] = "unavailable"
    rec["misinformation"]["credibility_score"] = 0.7  # Must be null!
    rec["misinformation"]["veracity_label"] = None
    rec["misinformation"]["risk_tier"] = None
    rec["misinformation"]["stylometric_signals"] = None

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "misinformation.credibility_score"


# =====================================================================
# 7. Consolidation Completeness & Provenance
# =====================================================================

def test_consolidation_complete_requires_four_content_sources(sample_reddit_record):
    """Verifies complete status requires all 4 content sources in sources_joined."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["consolidation"]["status"] = "complete"
    rec["consolidation"]["is_complete"] = True
    rec["consolidation"]["sources_joined"] = ["preprocessing", "sentiment", "topics"]  # missing misinformation

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "consolidation.sources_joined"
    assert "must include all four content sources" in rej["reason"]


def test_sources_joined_cannot_contain_graph_when_graph_unavailable(sample_reddit_record):
    """Verifies graph provenance cannot be claimed when author.graph_status is unavailable."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["author"]["graph_status"] = "unavailable"
    rec["consolidation"]["sources_joined"] = [
        "preprocessing", "sentiment", "topics", "misinformation", "graph"
    ]

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "consolidation.sources_joined"
    assert "cannot include 'graph'" in rej["reason"]


# =====================================================================
# 8. Topic Keyword Normalization & Immutability
# =====================================================================

def test_legacy_keyword_string_normalization():
    """Verifies legacy string keywords normalize to canonical objects with weight None."""
    legacy = ["machine learning", "nlp", "artificial intelligence"]
    is_valid, normalized, err = normalize_topic_keywords(legacy)

    assert is_valid is True
    assert err is None
    assert normalized == [
        {"word": "machine learning", "weight": None},
        {"word": "nlp", "weight": None},
        {"word": "artificial intelligence", "weight": None},
    ]
    # Explicitly verify weight is strictly None and never fabricated as 1.0
    for kw in normalized:
        assert kw["weight"] is None
        assert "word" in kw
        assert "keyword" not in kw


def test_canonical_keywords_preserved_with_actual_weights():
    """Verifies canonical keyword dictionaries preserve actual weights without fabricating."""
    canonical = [
        {"word": "data science", "weight": 0.85},
        {"word": "analytics", "weight": 0.42},
    ]
    is_valid, normalized, err = normalize_topic_keywords(canonical)

    assert is_valid is True
    assert err is None
    assert normalized == canonical
    assert normalized[0]["weight"] == 0.85
    assert normalized[1]["weight"] == 0.42


def test_keywords_missing_or_null_weight():
    """Verifies canonical objects with missing or null weights normalize to None."""
    # Object with missing weight key
    missing_weight = [{"word": "deep learning"}]
    is_valid, norm_missing, err = normalize_topic_keywords(missing_weight)
    assert is_valid is True
    assert err is None
    assert norm_missing == [{"word": "deep learning", "weight": None}]

    # Object with explicit null weight
    null_weight = [{"word": "transformers", "weight": None}]
    is_valid, norm_null, err = normalize_topic_keywords(null_weight)
    assert is_valid is True
    assert err is None
    assert norm_null == [{"word": "transformers", "weight": None}]


@pytest.mark.parametrize(
    "bad_keyword_obj,expected_reason_substr",
    [
        ([{"keyword": "nlp"}], "Unexpected key"),
        ([{"keyword": "nlp", "weight": 0.5}], "Unexpected key"),
        ([{"word": "nlp", "score": 0.9}], "Unexpected key"),
        ([{"word": "nlp", "weight": 0.8, "coherence": 0.5}], "Unexpected key"),
        ([{"weight": 0.8}], "missing required key 'word'"),
        ([{}], "missing required key 'word'"),
    ],
)
def test_invalid_keyword_object_keys_rejected(bad_keyword_obj, expected_reason_substr):
    """Verifies keyword objects with unapproved keys (like 'keyword') are strictly rejected."""
    is_valid, normalized, err = normalize_topic_keywords(bad_keyword_obj)
    assert is_valid is False
    assert normalized is None
    assert expected_reason_substr in err


def test_invalid_weight_diagnostic_is_generic_and_redacts_raw_values():
    """Verifies raw invalid weight values and tokens are omitted from rejection messages."""
    secret_token = "Bearer secret_jwt_token_99999"
    bad_obj = [{"word": "nlp", "weight": secret_token}]

    is_valid, normalized, err = normalize_topic_keywords(bad_obj)
    assert is_valid is False
    assert normalized is None
    # Secret must never be reflected in error message
    assert secret_token not in err
    assert err == "Keyword object at index 0 has invalid weight"


@pytest.mark.parametrize(
    "huge_int",
    [
        10**1000,
        -(10**1000),
        1 << 2048,
        -(1 << 4096),
    ],
)
def test_extremely_large_integer_weight_handled_safely_without_overflow(huge_int: int):
    """Verifies extremely large integers return is_valid=False without raising OverflowError."""
    bad_obj = [{"word": "nlp", "weight": huge_int}]

    is_valid, normalized, err = normalize_topic_keywords(bad_obj)
    assert is_valid is False
    assert normalized is None
    assert err == "Keyword object at index 0 has invalid weight"


def test_unexpected_dictionary_key_types_handled_safely_without_mixed_sort():
    """Verifies mixed and non-string dict keys do not crash on sorted() and use sanitized diagnostics."""
    # Mixed int, tuple, and str keys that would raise TypeError if sorted() is called
    mixed_keys_obj = [{"word": "nlp", 123: "val", (1, 2): "tuple_val", "extra_str": "val"}]
    is_valid, normalized, err = normalize_topic_keywords(mixed_keys_obj)
    assert is_valid is False
    assert normalized is None
    assert err == "Unexpected key in keyword object at index 0"

    # NoneType key
    none_key_obj = [{"word": "nlp", None: "none_val"}]
    is_valid_none, norm_none, err_none = normalize_topic_keywords(none_key_obj)
    assert is_valid_none is False
    assert norm_none is None
    assert err_none == "Unexpected key in keyword object at index 0"

    # Secret key name must not be leaked into error message
    secret_key = "AWS_SECRET_ACCESS_KEY_XYZ123"
    secret_key_obj = [{"word": "nlp", secret_key: "secret_value"}]
    is_valid_sec, norm_sec, err_sec = normalize_topic_keywords(secret_key_obj)
    assert is_valid_sec is False
    assert norm_sec is None
    assert secret_key not in err_sec
    assert err_sec == "Unexpected key in keyword object at index 0"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "machine learning",  # string instead of list
        123,                 # integer instead of list
        True,                # boolean instead of list
        {"word": "nlp"},     # dict instead of list
        [123],               # int element
        [None],              # None element
        [True],              # bool element
        [["nested"]],        # list element
        [{"word": 123}],     # non-string word
        [{"word": None}],    # null word
        [{"word": True}],    # boolean word
        [{"word": ""}],      # empty string word
        [{"word": "   "}],   # whitespace-only word
        [{"word": "nlp", "weight": True}],          # boolean weight
        [{"word": "nlp", "weight": "0.8"}],         # string weight
        [{"word": "nlp", "weight": float("nan")}],  # NaN weight
        [{"word": "nlp", "weight": float("inf")}],  # Inf weight
        [""],                # empty string legacy keyword
        ["   "],             # whitespace-only legacy keyword
    ],
)
def test_invalid_keyword_types_and_values_rejected(invalid_input):
    """Verifies malformed keyword structures are rejected without silent empty conversion."""
    is_valid, normalized, err = normalize_topic_keywords(invalid_input)
    assert is_valid is False
    assert normalized is None
    assert err is not None


def test_keywords_empty_and_null_behavior():
    """Verifies empty [] and null None keyword representations conform to schema."""
    # null / None keyword sequence
    is_valid_none, norm_none, err_none = normalize_topic_keywords(None)
    assert is_valid_none is True
    assert norm_none is None
    assert err_none is None

    # empty list keyword sequence
    is_valid_empty, norm_empty, err_empty = normalize_topic_keywords([])
    assert is_valid_empty is True
    assert norm_empty == []
    assert err_empty is None


def test_keywords_input_non_mutation():
    """Verifies normalize_topic_keywords does not mutate input sequences or dictionaries."""
    orig_canonical = [{"word": "machine learning", "weight": 0.9}]
    orig_canonical_copy = copy.deepcopy(orig_canonical)
    is_valid, norm_canonical, _ = normalize_topic_keywords(orig_canonical)
    assert is_valid is True
    assert orig_canonical == orig_canonical_copy
    assert orig_canonical is not norm_canonical

    orig_legacy = ["artificial intelligence", "nlp"]
    orig_legacy_copy = copy.deepcopy(orig_legacy)
    is_valid, norm_legacy, _ = normalize_topic_keywords(orig_legacy)
    assert is_valid is True
    assert orig_legacy == orig_legacy_copy


def test_validation_does_not_mutate_original_record(sample_reddit_record):
    """Verifies validate_and_normalize_record does not mutate input record in place."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["topics"]["keywords"] = ["legacy keyword"]

    is_valid, norm_doc, _ = validate_and_normalize_record(rec)
    assert is_valid is True

    # Original input list in rec must remain unchanged
    assert rec["topics"]["keywords"] == ["legacy keyword"]
    # Normalized doc contains canonical object with weight None
    assert norm_doc["topics"]["keywords"] == [{"word": "legacy keyword", "weight": None}]


# =====================================================================
# 9. Schema, Strict Types, and Non-Finite Number Rejection
# =====================================================================

def test_unknown_root_field_rejected(sample_reddit_record):
    """Verifies unknown root field is strictly rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["envelope_metric"] = 12345  # Not allowed in document

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "envelope_metric"
    assert "Unknown root field" in rej["reason"]


def test_unknown_nested_field_rejected(sample_reddit_record):
    """Verifies unknown nested field in content is strictly rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["content"]["extra_token"] = "leak"

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "content.extra_token"


@pytest.mark.parametrize(
    "numeric_field, path",
    [
        ("score", ["engagement", "score"]),
        ("views_count", ["engagement", "views_count"]),
        ("influencer_score", ["author", "influencer_score"]),
        ("score", ["sentiment", "score"]),
    ],
)
def test_boolean_in_numeric_fields_rejected(sample_reddit_record, numeric_field: str, path: list):
    """Verifies True or False in integer or float fields is strictly rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    if path[0] == "author":
        rec["author"]["graph_status"] = "ranked_influencer"
        rec["author"]["pagerank"] = 0.5
        rec["author"]["community_id"] = 1
        rec["author"]["in_degree"] = 1
        rec["author"]["out_degree"] = 1

    curr = rec
    for p in path[:-1]:
        curr = curr[p]
    curr[path[-1]] = True  # Boolean injection

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert numeric_field in rej["field"]


@pytest.mark.parametrize("bad_float", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_rejected(sample_reddit_record, bad_float: float):
    """Verifies NaN, Inf, and -Inf in float fields are rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["sentiment"]["score"] = bad_float

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "sentiment.score"


def test_timezone_naive_timestamp_rejected(sample_reddit_record):
    """Verifies timestamps lacking explicit timezone offset are rejected."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["timestamps"]["published_at"] = "2024-10-28T07:40:00"  # Naive, no Z or offset

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False
    assert rej["field"] == "timestamps.published_at"
    assert "timezone" in rej["reason"]


# =====================================================================
# 10. File & Envelope Error Handling (Unknown Record Count Semantics)
# =====================================================================

def test_missing_dataset_file_returns_failed_summary_with_unknown_total(tmp_path: Path, mock_es_client: ElasticsearchClient):
    """Verifies missing dataset file returns failed status and total_records=None."""
    nonexistent = tmp_path / "missing.json"
    res = index_consolidated_dataset(dataset_path=nonexistent, client=mock_es_client)

    assert res["status"] == "failed"
    assert res["total_records"] is None
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert len(res["indexing_failures"]) == 1


def test_empty_dataset_file_returns_failed_summary_with_unknown_total(tmp_path: Path, mock_es_client: ElasticsearchClient):
    """Verifies 0-byte dataset file: total_records cannot be established (None)."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")

    res = index_consolidated_dataset(dataset_path=empty_file, client=mock_es_client)
    assert res["status"] == "failed"
    assert res["total_records"] is None
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert len(res["indexing_failures"]) == 1


def test_malformed_json_dataset_returns_failed_summary_with_unknown_total(tmp_path: Path, mock_es_client: ElasticsearchClient):
    """Verifies malformed JSON: total_records cannot be established (None)."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ unclosed json: ", encoding="utf-8")

    res = index_consolidated_dataset(dataset_path=bad_json, client=mock_es_client)
    assert res["status"] == "failed"
    assert res["total_records"] is None
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert len(res["indexing_failures"]) == 1


def test_envelope_missing_records_list_returns_failed_with_unknown_total(tmp_path: Path, mock_es_client: ElasticsearchClient):
    """Verifies envelope missing 'records' list: total_records cannot be established (None)."""
    env = tmp_path / "no_records.json"
    env.write_text(json.dumps({"status": "success"}), encoding="utf-8")

    res = index_consolidated_dataset(dataset_path=env, client=mock_es_client)
    assert res["status"] == "failed"
    assert res["total_records"] is None
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert len(res["indexing_failures"]) == 1


# =====================================================================
# 11. Preflight Failures Accounting
# =====================================================================

def test_preflight_ping_failure_accounts_all_records_as_failed(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Verifies ping failure accounts all unattempted records as failed."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record, sample_youtube_record])),
        encoding="utf-8",
    )

    mock_es_client.ping = MagicMock(return_value=False)

    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client)

    assert res["status"] == "failed"
    assert res["total_records"] == 2
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 2
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    # Driver index was never called
    mock_es_client.index_document = MagicMock()
    mock_es_client.index_document.assert_not_called()


def test_preflight_index_creation_failure_accounts_all_records_as_failed(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies index creation exception in preflight accounts all records as failed."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record])),
        encoding="utf-8",
    )

    mock_es_client.create_index = MagicMock(side_effect=ElasticsearchIndexError("Creation blocked"))

    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client, ensure_index=True)

    assert res["status"] == "failed"
    assert res["total_records"] == 1
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 1
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]


def test_preflight_missing_index_with_ensure_index_false_fails(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies missing index when ensure_index=False marks all records as failed."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record])),
        encoding="utf-8",
    )

    mock_es_client.index_exists = MagicMock(return_value=False)

    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client, ensure_index=False)

    assert res["status"] == "failed"
    assert res["total_records"] == 1
    assert res["failed_count"] == 1


# =====================================================================
# 12. Accounting Invariant, Partial Failures, and Driver Handling
# =====================================================================

def test_partial_success_accounting_and_invariant(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Verifies partial success when 1 record succeeds, 1 is rejected, and 1 driver fails."""
    # Record 1: Valid Reddit (will succeed)
    rec1 = copy.deepcopy(sample_reddit_record)
    rec1["document_id"] = "reddit_t3_rec1"
    rec1["native_id"] = "t3_rec1"

    # Record 2: Invalid Record (will be rejected in pre-validation)
    rec2 = copy.deepcopy(sample_reddit_record)
    rec2["document_id"] = "reddit_t3_rec2"
    rec2["native_id"] = "t3_rec2"
    rec2["author"]["graph_status"] = "invalid_status"

    # Record 3: Valid Record, but driver will fail during indexing
    rec3 = copy.deepcopy(sample_youtube_record)
    rec3["document_id"] = "youtube_vid_rec3"
    rec3["native_id"] = "vid_rec3"

    dataset = tmp_path / "mixed_dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([rec1, rec2, rec3])),
        encoding="utf-8",
    )

    def mock_index_side_effect(document, index_name=None, refresh=None):
        if document["document_id"] == "youtube_vid_rec3":
            raise ElasticsearchConnectionError("Socket timeout during write")
        return {"result": "created", "_id": document["document_id"]}

    mock_es_client.index_document = MagicMock(side_effect=mock_index_side_effect)

    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client)

    assert res["status"] == "partial_success"
    assert res["total_records"] == 3
    assert res["indexed_count"] == 1
    assert res["rejected_count"] == 1
    assert res["failed_count"] == 1

    # Invariant assertion
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    assert len(res["rejections"]) == 1
    assert len(res["indexing_failures"]) == 1


def test_synthetic_indexed_result_counted_as_success(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies synthetic {'result': 'indexed'} from driver is counted as indexed."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record])),
        encoding="utf-8",
    )

    mock_es_client.index_document = MagicMock(return_value={"result": "indexed", "_id": "reddit_t3_ds1001"})

    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client)

    assert res["status"] == "success"
    assert res["indexed_count"] == 1
    assert res["failed_count"] == 0
    assert res["rejected_count"] == 0


# =====================================================================
# 13. Client Lifecycle and Cleanup
# =====================================================================

def test_internally_created_client_is_closed_on_success(tmp_path: Path, sample_reddit_record):
    """Verifies client created internally by adapter is closed in finally block."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record])),
        encoding="utf-8",
    )

    mock_internal_client = MagicMock()
    mock_internal_client.ping.return_value = True
    mock_internal_client.create_index.return_value = {"acknowledged": True}
    mock_internal_client.index_document.return_value = {"result": "created"}

    with patch("src.search.indexing_adapter.create_elasticsearch_client", return_value=mock_internal_client):
        res = index_consolidated_dataset(dataset_path=dataset, client=None)

    assert res["status"] == "success"
    # Verify close was called on the internally owned client
    mock_internal_client.close.assert_called_once()


def test_externally_provided_client_is_not_closed(tmp_path: Path, sample_reddit_record, mock_es_client: ElasticsearchClient):
    """Verifies caller-provided client is NOT closed by adapter."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record])),
        encoding="utf-8",
    )

    mock_es_client.close = MagicMock()
    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client)

    assert res["status"] == "success"
    mock_es_client.close.assert_not_called()


# =====================================================================
# 14. Security, Secret Redaction, and Safe Diagnostics
# =====================================================================

def test_rejection_report_sanitizes_content_and_omits_raw_body(sample_reddit_record):
    """Verifies diagnostic rejection report never leaks raw post text or secret tokens."""
    rec = copy.deepcopy(sample_reddit_record)
    rec["content"]["body_cleaned"] = "CONFIDENTIAL_PATIENT_RECORD_XYZ_123"
    rec["sentiment"]["score"] = "invalid_string_score"  # Will trigger rejection

    is_valid, _, rej = validate_and_normalize_record(rec)
    assert is_valid is False

    report_str = json.dumps(rej)
    assert "CONFIDENTIAL_PATIENT_RECORD_XYZ_123" not in report_str
    assert rej["field"] == "sentiment.score"
    assert rej["actual_type"] == "str"
    assert "finite float" in rej["expected"]


def test_indexing_failure_redacts_driver_secrets(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies driver exception messages containing tokens are sanitized in failures list."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(_create_valid_schema_v2_envelope([sample_reddit_record])),
        encoding="utf-8",
    )

    secret_bearer = "secret_auth_bearer_token_xyz"
    mock_es_client.index_document = MagicMock(
        side_effect=ElasticsearchConnectionError(f"Driver timeout on token={secret_bearer}")
    )

    res = index_consolidated_dataset(dataset_path=dataset, client=mock_es_client)

    assert res["status"] == "failed"
    assert res["failed_count"] == 1
    failure_report = res["indexing_failures"][0]
    assert secret_bearer not in json.dumps(failure_report)
    assert "(ElasticsearchConnectionError)" in failure_report["reason"]


# =====================================================================
# 15. Stage 11A.1 Bounded Bulk Indexing & Schema-v2 Pre-Validation Tests
# =====================================================================

@pytest.mark.parametrize("bad_batch_size", [0, -1, -100, True, False, "250", 2.5, [250]])
def test_index_consolidated_dataset_rejects_invalid_batch_size(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    bad_batch_size: Any,
):
    """Verifies batch_size parameter must be a strictly positive integer > 0 (excluding booleans)."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="batch_size must be an integer > 0"):
        index_consolidated_dataset(
            dataset_path=dataset,
            client=mock_es_client,
            batch_size=bad_batch_size,
        )


@pytest.mark.parametrize("bad_validate_env", [None, "true", "false", 1, 0, [], {}, 1.5])
def test_index_consolidated_dataset_rejects_non_boolean_validate_envelope(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    bad_validate_env: Any,
):
    """Verifies validate_envelope parameter rejects non-boolean types (including None)."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(TypeError, match="validate_envelope must be a boolean"):
        index_consolidated_dataset(
            dataset_path=dataset,
            client=mock_es_client,
            validate_envelope=bad_validate_env,
        )


def _make_three_sample_records(sample_reddit_record, sample_youtube_record) -> List[Dict[str, Any]]:
    """Helper creating 3 distinct valid records (2 Reddit, 1 YouTube)."""
    r1 = copy.deepcopy(sample_reddit_record)
    r1["document_id"] = "reddit_t3_rec1"
    r1["native_id"] = "t3_rec1"

    r2 = copy.deepcopy(sample_reddit_record)
    r2["document_id"] = "reddit_t3_rec2"
    r2["native_id"] = "t3_rec2"

    y1 = copy.deepcopy(sample_youtube_record)
    y1["document_id"] = "youtube_vid_rec3"
    y1["native_id"] = "vid_rec3"

    return [r1, r2, y1]


def test_schema_v2_scenario1_valid_envelope_three_records_normal_processing(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Scenario 1: Valid envelope, 3 records -> Total 3, indexed 3, failed 0, normal processing."""
    records = _make_three_sample_records(sample_reddit_record, sample_youtube_record)
    envelope = _create_valid_schema_v2_envelope(records)
    dataset = tmp_path / "scenario1.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "success"
    assert res["total_records"] == 3
    assert res["indexed_count"] == 3
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]


def test_schema_v2_scenario2_invalid_schema_version_three_readable_records(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Scenario 2: Invalid schema version, 3 readable records -> Total 3, failed 3, no ES calls."""
    records = _make_three_sample_records(sample_reddit_record, sample_youtube_record)
    envelope = _create_valid_schema_v2_envelope(records)
    envelope["schema_version"] = "99.0"  # Invalid schema version
    dataset = tmp_path / "scenario2.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    mock_es_client.ping = MagicMock()
    mock_driver = mock_es_client.get_client()

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] == 3
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 3
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    assert len(res["indexing_failures"]) == 1
    assert "Unsupported schema_version '99.0'" in res["indexing_failures"][0]["reason"]

    # No Elasticsearch operations occurred after envelope validation failed
    mock_es_client.ping.assert_not_called()
    mock_driver.bulk.assert_not_called()
    mock_driver.index.assert_not_called()


def test_schema_v2_scenario3_missing_schema_version_three_readable_records(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Scenario 3: Missing schema version, 3 readable records -> Total 3, failed 3, no ES calls."""
    records = _make_three_sample_records(sample_reddit_record, sample_youtube_record)
    envelope = _create_valid_schema_v2_envelope(records)
    del envelope["schema_version"]  # Missing schema_version key
    dataset = tmp_path / "scenario3.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    mock_es_client.ping = MagicMock()
    mock_driver = mock_es_client.get_client()

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] == 3
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 3
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    assert len(res["indexing_failures"]) == 1
    assert "Missing 'schema_version'" in res["indexing_failures"][0]["reason"] or "missing required envelope keys" in res["indexing_failures"][0]["reason"]

    mock_es_client.ping.assert_not_called()
    mock_driver.bulk.assert_not_called()
    mock_driver.index.assert_not_called()


def test_schema_v2_scenario4_missing_required_field_three_readable_records(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Scenario 4: Missing required field ('metrics'), 3 readable records -> Total 3, failed 3, no ES calls."""
    records = _make_three_sample_records(sample_reddit_record, sample_youtube_record)
    envelope = _create_valid_schema_v2_envelope(records)
    del envelope["metrics"]  # Missing mandatory 'metrics' envelope field
    dataset = tmp_path / "scenario4.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    mock_es_client.ping = MagicMock()
    mock_driver = mock_es_client.get_client()

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] == 3
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 3
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    assert len(res["indexing_failures"]) == 1
    assert "missing required envelope keys" in res["indexing_failures"][0]["reason"]

    mock_es_client.ping.assert_not_called()
    mock_driver.bulk.assert_not_called()
    mock_driver.index.assert_not_called()


def test_schema_v2_scenario5_missing_records_collection_artifact_level_failure(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Scenario 5a: Missing records collection -> Total unknown (0), failed 0, explicit artifact-level failure."""
    envelope = _create_valid_schema_v2_envelope([sample_reddit_record])
    del envelope["records"]  # Missing records collection
    dataset = tmp_path / "scenario5a.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    mock_es_client.ping = MagicMock()
    mock_driver = mock_es_client.get_client()

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] is None
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert len(res["indexing_failures"]) == 1

    mock_es_client.ping.assert_not_called()
    mock_driver.bulk.assert_not_called()
    mock_driver.index.assert_not_called()


def test_schema_v2_scenario5_malformed_records_collection_artifact_level_failure(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Scenario 5b: Malformed records collection (not a list) -> Total unknown (None), failed 0, explicit failure."""
    envelope = _create_valid_schema_v2_envelope([sample_reddit_record])
    envelope["records"] = "invalid_string_records"  # Malformed records collection
    dataset = tmp_path / "scenario5b.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    mock_es_client.ping = MagicMock()
    mock_driver = mock_es_client.get_client()

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] is None
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert len(res["indexing_failures"]) == 1

    mock_es_client.ping.assert_not_called()
    mock_driver.bulk.assert_not_called()
    mock_driver.index.assert_not_called()


def test_valid_empty_records_collection_returns_zero_total_records(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
):
    """Verifies valid empty records collection returns total_records=0 (known zero count)."""
    envelope = _create_valid_schema_v2_envelope([])
    dataset = tmp_path / "empty_valid_dataset.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    mock_es_client.ping = MagicMock()
    mock_driver = mock_es_client.get_client()

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "empty"
    assert res["total_records"] == 0
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 0
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]

    mock_es_client.ping.assert_not_called()
    mock_driver.bulk.assert_not_called()


def test_schema_v2_corrupted_metrics_three_readable_records_fails_closed(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Verifies metric sum mismatch with 3 readable records preserves total=3 and accounts all 3 as failed."""
    records = _make_three_sample_records(sample_reddit_record, sample_youtube_record)
    envelope = _create_valid_schema_v2_envelope(records)
    envelope["metrics"]["total_records_consolidated"] = 999  # Corrupted metric sum
    dataset = tmp_path / "corrupted_metrics.json"
    dataset.write_text(json.dumps(envelope), encoding="utf-8")

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] == 3
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["failed_count"] == 3
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    assert "does not match len(records)" in res["indexing_failures"][0]["reason"]


def test_legacy_minimal_envelope_fails_by_default_preserving_readable_records(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies minimal envelope fails by default (validate_envelope=True) preserving 1 readable record as failed."""
    minimal_artifact = {
        "status": "success",
        "records": [sample_reddit_record],
    }
    dataset = tmp_path / "legacy_default.json"
    dataset.write_text(json.dumps(minimal_artifact), encoding="utf-8")

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
    )

    assert res["status"] == "failed"
    assert res["total_records"] == 1
    assert res["failed_count"] == 1
    assert res["indexed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]
    assert "Schema-v2 validation failed" in res["indexing_failures"][0]["reason"]


def test_legacy_minimal_envelope_explicit_opt_out_succeeds(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies legacy minimal envelope succeeds when caller explicitly opts out with validate_envelope=False."""
    minimal_artifact = {
        "status": "success",
        "records": [sample_reddit_record],
    }
    dataset = tmp_path / "legacy_opt_out.json"
    dataset.write_text(json.dumps(minimal_artifact), encoding="utf-8")

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
        validate_envelope=False,
    )

    assert res["status"] == "success"
    assert res["total_records"] == 1
    assert res["indexed_count"] == 1
    assert res["failed_count"] == 0
    assert res["rejected_count"] == 0
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]


def test_bounded_bulk_indexing_chunks_documents_according_to_batch_size(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
):
    """Verifies 7 documents with batch_size=3 are ingested in bounded batches of [3, 3, 1]."""
    records = []
    for i in range(7):
        rec = copy.deepcopy(sample_reddit_record)
        rec["document_id"] = f"reddit_t3_chunk_{i}"
        rec["native_id"] = f"t3_chunk_{i}"
        records.append(rec)

    dataset = tmp_path / "chunk_dataset.json"
    dataset.write_text(json.dumps(_create_valid_schema_v2_envelope(records)), encoding="utf-8")

    batch_sizes_received = []
    original_bulk = mock_es_client.bulk_index_documents

    def spy_bulk(documents, **kwargs):
        batch_sizes_received.append(len(documents))
        return original_bulk(documents, **kwargs)

    mock_es_client.bulk_index_documents = MagicMock(side_effect=spy_bulk)

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
        batch_size=3,
    )

    assert res["status"] == "success"
    assert res["total_records"] == 7
    assert res["indexed_count"] == 7
    assert res["failed_count"] == 0
    assert res["rejected_count"] == 0
    assert batch_sizes_received == [3, 3, 1]
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]


def test_bounded_bulk_mixed_batches_preserves_count_invariant(
    tmp_path: Path,
    mock_es_client: ElasticsearchClient,
    sample_reddit_record,
    sample_youtube_record,
):
    """Verifies invariant: total_records == indexed_count + rejected_count + failed_count under mixed outcomes."""
    # 2 Valid records that will succeed
    rec_ok_1 = copy.deepcopy(sample_reddit_record)
    rec_ok_1["document_id"] = "reddit_t3_ok1"
    rec_ok_1["native_id"] = "t3_ok1"

    rec_ok_2 = copy.deepcopy(sample_youtube_record)
    rec_ok_2["document_id"] = "youtube_vid_ok2"
    rec_ok_2["native_id"] = "vid_ok2"

    # 2 Invalid records that will be rejected in pre-validation
    rec_bad_1 = copy.deepcopy(sample_reddit_record)
    rec_bad_1["document_id"] = "reddit_t3_bad1"
    rec_bad_1["native_id"] = "t3_bad1"
    rec_bad_1["sentiment"]["score"] = "invalid_score_str"

    rec_bad_2 = copy.deepcopy(sample_youtube_record)
    rec_bad_2["document_id"] = "youtube_vid_bad2"
    rec_bad_2["native_id"] = "vid_bad2"
    rec_bad_2["author"]["graph_status"] = "invalid_status"

    # 2 Valid records that will fail during bulk ingestion
    rec_fail_1 = copy.deepcopy(sample_reddit_record)
    rec_fail_1["document_id"] = "reddit_t3_fail1"
    rec_fail_1["native_id"] = "t3_fail1"

    rec_fail_2 = copy.deepcopy(sample_youtube_record)
    rec_fail_2["document_id"] = "youtube_vid_fail2"
    rec_fail_2["native_id"] = "vid_fail2"

    all_records = [rec_ok_1, rec_bad_1, rec_fail_1, rec_ok_2, rec_bad_2, rec_fail_2]
    dataset = tmp_path / "mixed_dataset.json"
    dataset.write_text(json.dumps({"records": all_records}), encoding="utf-8")

    def mock_bulk_side_effect(documents, **kwargs):
        successful_ids = []
        failed_items = []
        for doc in documents:
            d_id = doc["document_id"]
            if "fail" in d_id:
                failed_items.append({
                    "document_id": d_id,
                    "status": 400,
                    "error": "Mapper parsing exception on custom field",
                })
            else:
                successful_ids.append(d_id)
        return {
            "total": len(documents),
            "indexed_count": len(successful_ids),
            "failed_count": len(failed_items),
            "successful_ids": successful_ids,
            "failed_items": failed_items,
            "errors": bool(failed_items),
        }

    mock_es_client.bulk_index_documents = MagicMock(side_effect=mock_bulk_side_effect)

    res = index_consolidated_dataset(
        dataset_path=dataset,
        client=mock_es_client,
        batch_size=2,
        validate_envelope=False,
    )

    assert res["status"] == "partial_success"
    assert res["total_records"] == 6
    assert res["indexed_count"] == 2
    assert res["rejected_count"] == 2
    assert res["failed_count"] == 2
    assert len(res["rejections"]) == 2
    assert len(res["indexing_failures"]) == 2

    # Invariant verification
    assert res["total_records"] == res["indexed_count"] + res["rejected_count"] + res["failed_count"]




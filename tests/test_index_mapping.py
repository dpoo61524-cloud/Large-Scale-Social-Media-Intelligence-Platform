"""Tests for Elasticsearch Index Mapping, Settings, and Strict Schema Enforcement.

Verifies that src/search/index_mapping.py adheres to all Step 7.1 and Step 7.2 requirements:
- Local dev settings: 1 shard, 0 replicas.
- Custom multilingual_text_analyzer (standard tokenizer, lowercase filter, no stopword drops).
- Strict dynamic mapping ("dynamic": "strict").
- Complete alignment with the exact record schema produced by src.pipeline.consolidator.
- Non-indexed raw content (index: False).
- Nested topic keywords with nullable weight.
- Absence of envelope-level fields and invented fields.
- Deep copy safety and configurable settings factory functions.
- Deterministic consolidator document validation with field type checking.
- Strict rejection of unexpected fields across all nested levels.
- Strict rejection of incompatible types (including booleans in numeric fields).
- Complete acceptance of valid nullable fields.
"""

from __future__ import annotations

import copy
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.pipeline.consolidator import consolidate_social_intelligence
from src.search.index_mapping import (
    DEFAULT_INDEX_NAME,
    INDEX_CONFIGURATION,
    INDEX_MAPPINGS,
    INDEX_SETTINGS,
    get_index_configuration,
    get_index_mapping,
    get_index_settings,
)


# =====================================================================
# Strict Mapping and Type Validation Simulator
# =====================================================================

def assert_doc_matches_mapping_strict(
    doc: Dict[str, Any],
    mapping_props: Dict[str, Any],
    path: str = "",
) -> None:
    """Validates that a document conforms to Elasticsearch strict dynamic mapping.

    Enforces:
    - Every key present in doc MUST be defined in mapping_props (strict dynamic mapping).
    - Field values must match expected Elasticsearch data types.
    - Boolean values are strictly rejected for numeric fields (integer, long, float).
    - Null values (None) are accepted for nullable fields.
    - Nested objects and arrays of nested objects are validated recursively.

    Raises:
        ValueError: If an unexpected field is encountered or values violate range/date bounds.
        TypeError: If a field value violates expected Elasticsearch type constraints.
    """
    for key, val in doc.items():
        current_path = f"{path}.{key}" if path else key
        if key not in mapping_props:
            raise ValueError(
                f"Strict mapping violation: unexpected field '{current_path}' not defined in mapping"
            )

        field_schema = mapping_props[key]
        field_type = field_schema.get("type")

        # Null values are permitted for any field in Elasticsearch mapping
        if val is None:
            continue

        # Nested Object Containers (e.g. metadata.reddit, author, content)
        if "properties" in field_schema and field_type != "nested":
            if not isinstance(val, dict):
                raise TypeError(
                    f"Type violation at '{current_path}': expected dict for object field, got {type(val).__name__}"
                )
            assert_doc_matches_mapping_strict(val, field_schema["properties"], current_path)
            continue

        # Nested Array of Objects (e.g. topics.keywords)
        if field_type == "nested":
            if not isinstance(val, list):
                raise TypeError(
                    f"Type violation at '{current_path}': expected list for nested field, got {type(val).__name__}"
                )
            nested_props = field_schema.get("properties", {})
            for idx, item in enumerate(val):
                if not isinstance(item, dict):
                    raise TypeError(
                        f"Type violation at '{current_path}[{idx}]': expected dict in nested array, got {type(item).__name__}"
                    )
                assert_doc_matches_mapping_strict(item, nested_props, f"{current_path}[{idx}]")
            continue

        # Keyword Fields (string or array of strings, e.g. sources_joined)
        if field_type == "keyword":
            if isinstance(val, list):
                for idx, item in enumerate(val):
                    if not isinstance(item, str):
                        raise TypeError(
                            f"Type violation at '{current_path}[{idx}]': expected string for keyword element, got {type(item).__name__}"
                        )
            elif not isinstance(val, str):
                raise TypeError(
                    f"Type violation at '{current_path}': expected string for keyword, got {type(val).__name__}"
                )

        # Text Fields (string or array of strings)
        elif field_type == "text":
            if isinstance(val, list):
                for idx, item in enumerate(val):
                    if not isinstance(item, str):
                        raise TypeError(
                            f"Type violation at '{current_path}[{idx}]': expected string for text element, got {type(item).__name__}"
                        )
            elif not isinstance(val, str):
                raise TypeError(
                    f"Type violation at '{current_path}': expected string for text, got {type(val).__name__}"
                )

        # Integer Fields (signed 32-bit: must exclude bool)
        elif field_type == "integer":
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    f"Type violation at '{current_path}': expected integer (excluding bool), got {type(val).__name__} ({val!r})"
                )
            if not (-2147483648 <= val <= 2147483647):
                raise ValueError(
                    f"Range violation at '{current_path}': integer {val} exceeds 32-bit signed integer range"
                )

        # Long Fields (signed 64-bit: must exclude bool)
        elif field_type == "long":
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    f"Type violation at '{current_path}': expected long integer (excluding bool), got {type(val).__name__} ({val!r})"
                )
            if not (-9223372036854775808 <= val <= 9223372036854775807):
                raise ValueError(
                    f"Range violation at '{current_path}': long {val} exceeds 64-bit signed integer range"
                )

        # Float Fields (finite numeric value: must exclude bool)
        elif field_type == "float":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"Type violation at '{current_path}': expected float/number (excluding bool), got {type(val).__name__} ({val!r})"
                )
            if not math.isfinite(val):
                raise ValueError(
                    f"Value violation at '{current_path}': float value must be finite, got {val}"
                )

        # Boolean Fields
        elif field_type == "boolean":
            if not isinstance(val, bool):
                raise TypeError(
                    f"Type violation at '{current_path}': expected boolean, got {type(val).__name__} ({val!r})"
                )

        # Date Fields (ISO-8601 string or numeric timestamp)
        elif field_type == "date":
            if isinstance(val, str):
                try:
                    cleaned = val.strip()
                    if cleaned.endswith("Z"):
                        cleaned = cleaned[:-1] + "+00:00"
                    datetime.fromisoformat(cleaned)
                except ValueError as err:
                    raise ValueError(
                        f"Format violation at '{current_path}': invalid ISO-8601 date string '{val}': {err}"
                    )
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                pass
            else:
                raise TypeError(
                    f"Type violation at '{current_path}': expected date string or numeric epoch, got {type(val).__name__}"
                )


# =====================================================================
# Deterministic Test Fixtures
# =====================================================================

@pytest.fixture
def deterministic_consolidated_records(tmp_path: Path) -> List[Dict[str, Any]]:
    """Generates complete, deterministic records directly via consolidate_social_intelligence."""
    reddit_data = [
        {
            "post_id": "r_sample_101",
            "subreddit": "datascience",
            "title": "Multilingual NLP Architectures",
            "text": "Detailed discussion of transformer embeddings and topic clusters.",
            "author": "analyst_user",
            "created_utc": 1730101200,
            "score": 250,
            "num_comments": 42,
            "url": "https://reddit.com/r/datascience/comments/r_sample_101",
            "collected_at": "2026-11-05T12:00:00Z",
            "title_cleaned": "Multilingual NLP Architectures",
            "text_cleaned": "Detailed discussion of transformer embeddings and topic clusters.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-05T12:05:00Z",
        }
    ]
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(reddit_data), encoding="utf-8")

    youtube_data = [
        {
            "video_id": "y_sample_202",
            "title": "Large Scale Graph Mining",
            "description": "Video explaining PageRank, community detection, and PySpark.",
            "channel_id": "UC_channel_abc",
            "channel_title": "TechEngineering",
            "published_at": "2026-11-01T10:00:00Z",
            "view_count": 55000,
            "like_count": 1800,
            "comment_count": 120,
            "url": "https://youtube.com/watch?v=y_sample_202",
            "collected_at": "2026-11-05T12:00:00Z",
            "title_cleaned": "Large Scale Graph Mining",
            "description_cleaned": "Video explaining PageRank, community detection, and PySpark.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-05T12:05:00Z",
        }
    ]
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(youtube_data), encoding="utf-8")

    sentiment_data = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_sample_101",
                "sentiment_label": "positive",
                "sentiment_score": 0.92,
                "sentiment_status": "success",
            }
        ],
        "youtube_records": [
            {
                "video_id": "y_sample_202",
                "sentiment_label": "neutral",
                "sentiment_score": 0.75,
                "sentiment_status": "success",
            }
        ],
    }
    s_file = tmp_path / "sentiment.json"
    s_file.write_text(json.dumps(sentiment_data), encoding="utf-8")

    topic_data = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_sample_101",
                "topic_id": 1,
                "topic_probability": 0.88,
                "topic_keywords": [{"word": "nlp", "weight": 0.95}, {"word": "transformers", "weight": 0.81}],
                "topic_status": "assigned",
            }
        ],
        "youtube_records": [
            {
                "video_id": "y_sample_202",
                "topic_id": -1,
                "topic_probability": 0.35,
                "topic_keywords": [{"word": "graph", "weight": 0.72}],
                "topic_status": "outlier",
            }
        ],
    }
    t_file = tmp_path / "topics.json"
    t_file.write_text(json.dumps(topic_data), encoding="utf-8")

    graph_data = {
        "status": "success",
        "top_influencers": [
            {
                "username": "analyst_user",
                "platform": "reddit",
                "influencer_score": 0.82,
                "pagerank": 0.041,
                "community_id": 3,
                "in_degree_raw": 12,
                "out_degree_raw": 4,
            }
        ],
    }
    g_file = tmp_path / "graph.json"
    g_file.write_text(json.dumps(graph_data), encoding="utf-8")

    misinfo_data = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_sample_101",
                "veracity_label": "real",
                "credibility_score": 0.91,
                "risk_tier": "low",
                "misinformation_status": "success",
                "stylometric_signals": {
                    "uppercase_char_ratio": 0.02,
                    "exclamation_intensity": 0.0,
                    "question_intensity": 0.0,
                    "word_count": 15.0,
                    "avg_word_length": 6.1,
                    "lexical_subjectivity_ratio": 0.0,
                    "digits_ratio": 0.0,
                    "all_caps_word_ratio": 0.0,
                },
            }
        ],
        "youtube_records": [
            {
                "video_id": "y_sample_202",
                "veracity_label": "fake",
                "credibility_score": 0.18,
                "risk_tier": "high_risk",
                "misinformation_status": "success",
                "stylometric_signals": {
                    "uppercase_char_ratio": 0.14,
                    "exclamation_intensity": 0.08,
                    "question_intensity": 0.04,
                    "word_count": 22.0,
                    "avg_word_length": 5.4,
                    "lexical_subjectivity_ratio": 0.05,
                    "digits_ratio": 0.0,
                    "all_caps_word_ratio": 0.0,
                },
            }
        ],
    }
    m_file = tmp_path / "misinfo.json"
    m_file.write_text(json.dumps(misinfo_data), encoding="utf-8")

    output_path = tmp_path / "consolidated_out.json"

    res = consolidate_social_intelligence(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        sentiment_path=s_file,
        topics_path=t_file,
        graph_path=g_file,
        misinformation_path=m_file,
        output_path=output_path,
        execution_time="2026-11-07T12:00:00Z",
    )

    return res["records"]


@pytest.fixture
def canonical_reddit_record() -> Dict[str, Any]:
    """Returns a canonical, complete Reddit document conforming to the consolidator schema."""
    return {
        "document_id": "reddit_t3_canonical1",
        "platform": "reddit",
        "native_id": "t3_canonical1",
        "url": "https://reddit.com/r/datascience/comments/t3_canonical1",
        "timestamps": {
            "published_at": "2026-11-01T12:00:00+00:00",
            "collected_at": "2026-11-05T12:00:00+00:00",
            "processed_at": "2026-11-06T12:00:00+00:00",
        },
        "author": {
            "username": "data_expert",
            "channel_id": None,
            "channel_title": None,
            "graph_status": "ranked_influencer",
            "influencer_score": 0.88,
            "pagerank": 0.045,
            "community_id": 2,
            "in_degree": 14,
            "out_degree": 6,
        },
        "content": {
            "title_raw": "Raw Reddit Post Title",
            "body_raw": "Raw Reddit Post Body Text",
            "title_cleaned": "Raw Reddit Post Title",
            "body_cleaned": "Raw Reddit Post Body Text",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
        },
        "engagement": {
            "score": 320,
            "views_count": None,
            "likes_count": None,
            "comments_count": 48,
        },
        "metadata": {
            "reddit": {
                "subreddit": "datascience",
                "score": 320,
            },
            "youtube": None,
        },
        "sentiment": {
            "label": "positive",
            "score": 0.85,
            "status": "success",
        },
        "topics": {
            "topic_id": 5,
            "probability": 0.92,
            "keywords": [
                {"word": "neural", "weight": 0.87},
                {"word": "embeddings", "weight": None},  # Nullable weight
            ],
            "status": "assigned",
        },
        "misinformation": {
            "veracity_label": "real",
            "credibility_score": 0.89,
            "risk_tier": "low",
            "status": "success",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.03,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 18.0,
                "avg_word_length": 5.8,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        },
        "consolidation": {
            "status": "complete",
            "is_complete": True,
            "sources_joined": ["preprocessing", "sentiment", "topics", "graph", "misinformation"],
        },
    }


@pytest.fixture
def canonical_youtube_record() -> Dict[str, Any]:
    """Returns a canonical, complete YouTube document with outlier topic and high view count."""
    return {
        "document_id": "youtube_vid_canonical2",
        "platform": "youtube",
        "native_id": "vid_canonical2",
        "url": "https://youtube.com/watch?v=vid_canonical2",
        "timestamps": {
            "published_at": "2026-11-02T15:30:00+00:00",
            "collected_at": "2026-11-05T12:00:00+00:00",
            "processed_at": "2026-11-06T12:00:00+00:00",
        },
        "author": {
            "username": "EngineeringInsights",
            "channel_id": "UC_canonical_yt",
            "channel_title": "EngineeringInsights",
            "graph_status": "not_in_top_influencers",
            "influencer_score": None,
            "pagerank": None,
            "community_id": None,
            "in_degree": None,
            "out_degree": None,
        },
        "content": {
            "title_raw": "Intro to Distributed Systems",
            "body_raw": "Full description of distributed pipelines and message queues.",
            "title_cleaned": "Intro to Distributed Systems",
            "body_cleaned": "Full description of distributed pipelines and message queues.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
        },
        "engagement": {
            "score": None,
            "views_count": 3500000000,  # > 2^31-1 to test 64-bit long integer
            "likes_count": 125000,
            "comments_count": 8900,
        },
        "metadata": {
            "reddit": None,
            "youtube": {
                "channel_id": "UC_canonical_yt",
                "channel_title": "EngineeringInsights",
            },
        },
        "sentiment": {
            "label": "neutral",
            "score": 0.65,
            "status": "success",
        },
        "topics": {
            "topic_id": -1,  # Outlier topic ID
            "probability": 0.42,
            "keywords": [
                {"word": "distributed", "weight": 0.79},
            ],
            "status": "outlier",
        },
        "misinformation": {
            "veracity_label": "real",
            "credibility_score": 0.81,
            "risk_tier": "low",
            "status": "success",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.05,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 12.0,
                "avg_word_length": 5.5,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        },
        "consolidation": {
            "status": "complete",
            "is_complete": True,
            "sources_joined": ["preprocessing", "sentiment", "topics", "misinformation"],
        },
    }


# =====================================================================
# Unit Tests for Mapping Structure & Metadata
# =====================================================================

def test_default_index_name():
    """Verifies standard default index naming."""
    assert DEFAULT_INDEX_NAME == "consolidated_intelligence"


def test_index_settings_single_node_configuration():
    """Verifies single-node settings: 1 primary shard, 0 replicas, and field limit."""
    settings = get_index_settings()
    idx_settings = settings["index"]

    assert idx_settings["number_of_shards"] == 1
    assert idx_settings["number_of_replicas"] == 0
    assert idx_settings["mapping"]["total_fields"]["limit"] == 1000


def test_multilingual_text_analyzer_definition():
    """Verifies multilingual text analyzer configuration.

    Must use standard tokenizer and lowercase filter without language-specific
    stopword filtering to avoid stripping tokens from multilingual posts.
    """
    settings = get_index_settings()
    analyzers = settings["analysis"]["analyzer"]

    assert "multilingual_text_analyzer" in analyzers
    ana = analyzers["multilingual_text_analyzer"]

    assert ana["type"] == "custom"
    assert ana["tokenizer"] == "standard"
    assert ana["filter"] == ["lowercase"]
    assert "stopwords" not in ana
    assert "_english_" not in str(ana)


def test_dynamic_strict_enforced():
    """Verifies strict dynamic mapping at the root level."""
    mapping = get_index_mapping()
    assert mapping["dynamic"] == "strict"


def test_root_properties_cover_exact_consolidator_schema():
    """Verifies all 13 top-level record fields produced by the consolidator exist."""
    mapping = get_index_mapping()
    properties = mapping["properties"]

    expected_root_fields = {
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

    assert set(properties.keys()) == expected_root_fields


def test_envelope_fields_not_present_in_document_mapping():
    """Verifies pipeline envelope metadata is strictly segregated from documents."""
    mapping = get_index_mapping()
    properties = mapping["properties"]

    envelope_fields = {
        "status",
        "consolidation_status",
        "generated_at",
        "pipeline_version",
        "metrics",
        "audit_warnings",
        "rejected_records",
        "records",
    }

    for env_field in envelope_fields:
        assert env_field not in properties, f"Envelope field '{env_field}' should not be in document properties"


def test_invented_fields_not_present_in_mapping():
    """Verifies previously rejected or invented fields are completely absent."""
    mapping = get_index_mapping()
    properties = mapping["properties"]

    # Top-level invented fields
    assert "id" not in properties
    assert "created_at" not in properties
    assert "char_count" not in properties
    assert "word_count" not in properties
    assert "network" not in properties
    assert "consolidated_at" not in properties
    assert "module_statuses" not in properties

    # Nested invented fields
    author_props = properties["author"]["properties"]
    assert "account_age_days" not in author_props
    assert "is_verified" not in author_props

    sentiment_props = properties["sentiment"]["properties"]
    assert "sentiment_probabilities" not in sentiment_props
    assert "sentiment_score" not in sentiment_props
    assert "sentiment_label" not in sentiment_props

    topic_props = properties["topics"]["properties"]
    assert "topic_label" not in topic_props
    assert "topic_confidence" not in topic_props


def test_primary_identifiers_mapping():
    """Verifies types for primary keys and URL."""
    properties = get_index_mapping()["properties"]

    assert properties["document_id"]["type"] == "keyword"
    assert properties["platform"]["type"] == "keyword"
    assert properties["native_id"]["type"] == "keyword"
    assert properties["url"]["type"] == "keyword"
    assert properties["url"]["ignore_above"] == 2048


def test_timestamps_mapping():
    """Verifies ISO date types for publication, collection, and preprocessing."""
    ts_props = get_index_mapping()["properties"]["timestamps"]["properties"]

    assert ts_props["published_at"]["type"] == "date"
    assert ts_props["collected_at"]["type"] == "date"
    assert ts_props["processed_at"]["type"] == "date"


def test_author_and_graph_analytics_mapping():
    """Verifies author metadata and graph metrics nested under author."""
    author_props = get_index_mapping()["properties"]["author"]["properties"]

    assert author_props["username"]["type"] == "keyword"
    assert author_props["channel_id"]["type"] == "keyword"
    assert author_props["channel_title"]["type"] == "keyword"
    assert author_props["graph_status"]["type"] == "keyword"
    assert author_props["influencer_score"]["type"] == "float"
    assert author_props["pagerank"]["type"] == "float"
    assert author_props["community_id"]["type"] == "integer"
    assert author_props["in_degree"]["type"] == "integer"
    assert author_props["out_degree"]["type"] == "integer"


def test_content_raw_vs_cleaned_indexing_rules():
    """Verifies that raw content is unindexed while cleaned content uses multilingual search."""
    content_props = get_index_mapping()["properties"]["content"]["properties"]

    assert content_props["title_raw"]["type"] == "text"
    assert content_props["title_raw"]["index"] is False
    assert content_props["body_raw"]["type"] == "text"
    assert content_props["body_raw"]["index"] is False

    assert content_props["title_cleaned"]["type"] == "text"
    assert content_props["title_cleaned"]["analyzer"] == "multilingual_text_analyzer"
    assert content_props["title_cleaned"]["fields"]["keyword"]["type"] == "keyword"
    assert content_props["title_cleaned"]["fields"]["keyword"]["ignore_above"] == 512

    assert content_props["body_cleaned"]["type"] == "text"
    assert content_props["body_cleaned"]["analyzer"] == "multilingual_text_analyzer"

    assert content_props["detected_language"]["type"] == "keyword"
    assert content_props["language_code"]["type"] == "keyword"
    assert content_props["preprocessing_status"]["type"] == "keyword"


def test_engagement_fields_mapping():
    """Verifies numeric types for engagement metrics across platforms."""
    eng_props = get_index_mapping()["properties"]["engagement"]["properties"]

    assert eng_props["score"]["type"] == "integer"
    assert eng_props["views_count"]["type"] == "long"
    assert eng_props["likes_count"]["type"] == "integer"
    assert eng_props["comments_count"]["type"] == "integer"


def test_platform_metadata_mapping():
    """Verifies platform-specific metadata containers and properties."""
    meta_props = get_index_mapping()["properties"]["metadata"]["properties"]

    reddit_props = meta_props["reddit"]["properties"]
    assert reddit_props["subreddit"]["type"] == "keyword"
    assert reddit_props["score"]["type"] == "integer"

    youtube_props = meta_props["youtube"]["properties"]
    assert youtube_props["channel_id"]["type"] == "keyword"
    assert youtube_props["channel_title"]["type"] == "keyword"


def test_sentiment_mapping():
    """Verifies sentiment dimension properties."""
    sent_props = get_index_mapping()["properties"]["sentiment"]["properties"]

    assert sent_props["label"]["type"] == "keyword"
    assert sent_props["score"]["type"] == "float"
    assert sent_props["status"]["type"] == "keyword"


def test_topics_nested_keywords_mapping():
    """Verifies topic modeling properties and nested keyword schema."""
    topic_props = get_index_mapping()["properties"]["topics"]["properties"]

    assert topic_props["topic_id"]["type"] == "integer"
    assert topic_props["probability"]["type"] == "float"
    assert topic_props["status"]["type"] == "keyword"

    kw_schema = topic_props["keywords"]
    assert kw_schema["type"] == "nested"
    kw_props = kw_schema["properties"]

    assert kw_props["word"]["type"] == "keyword"
    assert kw_props["word"]["fields"]["text"]["type"] == "text"
    assert kw_props["word"]["fields"]["text"]["analyzer"] == "multilingual_text_analyzer"
    assert kw_props["weight"]["type"] == "float"


def test_misinformation_and_stylometric_signals_mapping():
    """Verifies misinformation dimension and all 8 stylometric features."""
    mis_props = get_index_mapping()["properties"]["misinformation"]["properties"]

    assert mis_props["veracity_label"]["type"] == "keyword"
    assert mis_props["credibility_score"]["type"] == "float"
    assert mis_props["risk_tier"]["type"] == "keyword"
    assert mis_props["status"]["type"] == "keyword"

    sty_props = mis_props["stylometric_signals"]["properties"]
    expected_signals = {
        "uppercase_char_ratio",
        "exclamation_intensity",
        "question_intensity",
        "word_count",
        "avg_word_length",
        "lexical_subjectivity_ratio",
        "digits_ratio",
        "all_caps_word_ratio",
    }
    assert set(sty_props.keys()) == expected_signals
    for sig in expected_signals:
        assert sty_props[sig]["type"] == "float"


def test_consolidation_flags_mapping():
    """Verifies consolidation status, boolean completeness, and joined sources."""
    cons_props = get_index_mapping()["properties"]["consolidation"]["properties"]

    assert cons_props["status"]["type"] == "keyword"
    assert cons_props["is_complete"]["type"] == "boolean"
    assert cons_props["sources_joined"]["type"] == "keyword"


# =====================================================================
# Factory Functions & Deep Copy Independence Tests
# =====================================================================

def test_settings_factory_overrides_and_immutability():
    """Verifies settings factory function allows parameter overrides without mutating global state."""
    custom = get_index_settings(number_of_shards=3, number_of_replicas=2)
    assert custom["index"]["number_of_shards"] == 3
    assert custom["index"]["number_of_replicas"] == 2

    # Global constant remains unchanged
    assert INDEX_SETTINGS["index"]["number_of_shards"] == 1
    assert INDEX_SETTINGS["index"]["number_of_replicas"] == 0


def test_get_index_mapping_returns_independent_deep_copies():
    """Verifies that mutating the returned mapping dict does not affect subsequent calls or constants."""
    mapping1 = get_index_mapping()
    mapping1["properties"]["unexpected_mutated_field"] = {"type": "keyword"}
    mapping1["dynamic"] = "lenient"

    mapping2 = get_index_mapping()
    assert "unexpected_mutated_field" not in mapping2["properties"]
    assert mapping2["dynamic"] == "strict"
    assert "unexpected_mutated_field" not in INDEX_MAPPINGS["properties"]
    assert INDEX_MAPPINGS["dynamic"] == "strict"


def test_get_index_settings_returns_independent_deep_copies():
    """Verifies that mutating the returned settings dict does not affect subsequent calls or constants."""
    settings1 = get_index_settings()
    settings1["index"]["number_of_shards"] = 999
    settings1["index"]["number_of_replicas"] = 888

    settings2 = get_index_settings()
    assert settings2["index"]["number_of_shards"] == 1
    assert settings2["index"]["number_of_replicas"] == 0
    assert INDEX_SETTINGS["index"]["number_of_shards"] == 1
    assert INDEX_SETTINGS["index"]["number_of_replicas"] == 0


def test_get_index_configuration_returns_independent_deep_copies():
    """Verifies composite configuration generator bundles settings and mapping correctly and safely."""
    config1 = get_index_configuration(number_of_shards=2, number_of_replicas=1)
    config1["settings"]["index"]["number_of_shards"] = 999
    config1["mappings"]["properties"]["extra_field"] = {"type": "text"}

    config2 = get_index_configuration()
    assert config2["settings"]["index"]["number_of_shards"] == 1
    assert config2["settings"]["index"]["number_of_replicas"] == 0
    assert "extra_field" not in config2["mappings"]["properties"]
    assert config2 == INDEX_CONFIGURATION


# =====================================================================
# Document Validation Tests (Deterministic Fixtures & Strict Simulator)
# =====================================================================

def test_deterministic_consolidator_output_conforms_to_mapping(
    deterministic_consolidated_records: List[Dict[str, Any]],
):
    """Verifies real documents produced by consolidator pass strict mapping & type checks."""
    assert len(deterministic_consolidated_records) == 2
    mapping = get_index_mapping()
    root_props = mapping["properties"]

    for doc in deterministic_consolidated_records:
        assert_doc_matches_mapping_strict(doc, root_props)


def test_canonical_reddit_and_youtube_records_conform_to_mapping(
    canonical_reddit_record: Dict[str, Any],
    canonical_youtube_record: Dict[str, Any],
):
    """Verifies canonical Reddit and YouTube documents conform fully to strict schema."""
    mapping = get_index_mapping()
    root_props = mapping["properties"]

    assert_doc_matches_mapping_strict(canonical_reddit_record, root_props)
    assert_doc_matches_mapping_strict(canonical_youtube_record, root_props)


def test_all_nullable_fields_as_none_accepted(canonical_reddit_record: Dict[str, Any]):
    """Verifies that all nullable analytical and platform fields are safely accepted when None."""
    sparse_doc = copy.deepcopy(canonical_reddit_record)

    # Set nullable fields to None
    sparse_doc["url"] = ""
    sparse_doc["timestamps"]["published_at"] = None
    sparse_doc["timestamps"]["collected_at"] = None
    sparse_doc["timestamps"]["processed_at"] = None
    sparse_doc["author"]["channel_id"] = None
    sparse_doc["author"]["channel_title"] = None
    sparse_doc["author"]["influencer_score"] = None
    sparse_doc["author"]["pagerank"] = None
    sparse_doc["author"]["community_id"] = None
    sparse_doc["author"]["in_degree"] = None
    sparse_doc["author"]["out_degree"] = None
    sparse_doc["engagement"]["score"] = None
    sparse_doc["engagement"]["views_count"] = None
    sparse_doc["engagement"]["likes_count"] = None
    sparse_doc["engagement"]["comments_count"] = None
    sparse_doc["metadata"]["reddit"] = None
    sparse_doc["metadata"]["youtube"] = None
    sparse_doc["sentiment"]["label"] = None
    sparse_doc["sentiment"]["score"] = None
    sparse_doc["topics"]["topic_id"] = None
    sparse_doc["topics"]["probability"] = None
    sparse_doc["topics"]["keywords"] = []
    sparse_doc["misinformation"]["veracity_label"] = None
    sparse_doc["misinformation"]["credibility_score"] = None
    sparse_doc["misinformation"]["risk_tier"] = None
    sparse_doc["misinformation"]["stylometric_signals"] = None

    mapping = get_index_mapping()
    assert_doc_matches_mapping_strict(sparse_doc, mapping["properties"])


# =====================================================================
# Strict Dynamic Mapping Rejection Tests (Nested Level Injections)
# =====================================================================

@pytest.mark.parametrize(
    "tamper_fn, expected_error_substr",
    [
        (lambda d: d.update({"unmapped_root_field": "bad"}), "unmapped_root_field"),
        (lambda d: d["author"].update({"unknown_author_meta": 123}), "author.unknown_author_meta"),
        (lambda d: d["content"].update({"word_count_invented": 10}), "content.word_count_invented"),
        (lambda d: d["engagement"].update({"shares_count": 50}), "engagement.shares_count"),
        (lambda d: d["metadata"]["reddit"].update({"flair": "discussion"}), "metadata.reddit.flair"),
        (lambda d: d["sentiment"].update({"probabilities": {"pos": 0.8}}), "sentiment.probabilities"),
        (lambda d: d["topics"].update({"topic_label": "tech"}), "topics.topic_label"),
        (lambda d: d["topics"]["keywords"][0].update({"coherence": 0.95}), "topics.keywords[0].coherence"),
        (lambda d: d["misinformation"].update({"detector_version": "2.0"}), "misinformation.detector_version"),
        (lambda d: d["misinformation"]["stylometric_signals"].update({"synthetic_signal": 0.5}), "misinformation.stylometric_signals.synthetic_signal"),
        (lambda d: d["consolidation"].update({"audit_warnings_inlined": []}), "consolidation.audit_warnings_inlined"),
    ],
)
def test_strict_mapping_rejects_unexpected_fields_at_all_levels(
    canonical_reddit_record: Dict[str, Any],
    tamper_fn,
    expected_error_substr: str,
):
    """Verifies that injecting an undeclared field at any nested level triggers a strict validation error."""
    doc = copy.deepcopy(canonical_reddit_record)
    tamper_fn(doc)

    mapping = get_index_mapping()
    with pytest.raises(ValueError) as exc_info:
        assert_doc_matches_mapping_strict(doc, mapping["properties"])

    assert "Strict mapping violation" in str(exc_info.value)
    assert expected_error_substr in str(exc_info.value)


# =====================================================================
# Type Incompatibility Rejection Tests (Boolean and Incompatible Types)
# =====================================================================

@pytest.mark.parametrize(
    "tamper_fn, expected_field",
    [
        # Booleans where integers are expected (isinstance(True, int) is True in Python, must be rejected)
        (lambda d: d["engagement"].update({"score": True}), "engagement.score"),
        (lambda d: d["engagement"].update({"views_count": False}), "engagement.views_count"),
        (lambda d: d["author"].update({"community_id": True}), "author.community_id"),
        (lambda d: d["author"].update({"in_degree": False}), "author.in_degree"),
        (lambda d: d["author"].update({"out_degree": True}), "author.out_degree"),
        (lambda d: d["topics"].update({"topic_id": True}), "topics.topic_id"),
        (lambda d: d["metadata"]["reddit"].update({"score": False}), "metadata.reddit.score"),
        # Booleans where floats are expected
        (lambda d: d["author"].update({"influencer_score": True}), "author.influencer_score"),
        (lambda d: d["author"].update({"pagerank": False}), "author.pagerank"),
        (lambda d: d["sentiment"].update({"score": True}), "sentiment.score"),
        (lambda d: d["topics"].update({"probability": False}), "topics.probability"),
        (lambda d: d["topics"]["keywords"][0].update({"weight": True}), "topics.keywords[0].weight"),
        (lambda d: d["misinformation"].update({"credibility_score": False}), "misinformation.credibility_score"),
        (lambda d: d["misinformation"]["stylometric_signals"].update({"uppercase_char_ratio": True}), "misinformation.stylometric_signals.uppercase_char_ratio"),
        # Strings where numbers are expected
        (lambda d: d["engagement"].update({"score": "100"}), "engagement.score"),
        (lambda d: d["sentiment"].update({"score": "0.95"}), "sentiment.score"),
        # Non-booleans where booleans are expected
        (lambda d: d["consolidation"].update({"is_complete": 1}), "consolidation.is_complete"),
        (lambda d: d["consolidation"].update({"is_complete": "true"}), "consolidation.is_complete"),
        # Non-strings where strings/keywords are expected
        (lambda d: d.update({"platform": 123}), "platform"),
        (lambda d: d["content"].update({"title_cleaned": 999}), "content.title_cleaned"),
        (lambda d: d["consolidation"].update({"sources_joined": [1, 2, 3]}), "consolidation.sources_joined[0]"),
    ],
)
def test_validation_rejects_incompatible_types_and_booleans_in_numeric_fields(
    canonical_reddit_record: Dict[str, Any],
    tamper_fn,
    expected_field: str,
):
    """Verifies that incompatible data types (especially booleans for numeric fields) are rejected."""
    doc = copy.deepcopy(canonical_reddit_record)
    tamper_fn(doc)

    mapping = get_index_mapping()
    with pytest.raises(TypeError) as exc_info:
        assert_doc_matches_mapping_strict(doc, mapping["properties"])

    assert "Type violation" in str(exc_info.value)
    assert expected_field in str(exc_info.value)


@pytest.mark.parametrize("invalid_float", [float("nan"), float("inf"), float("-inf")])
def test_validation_rejects_non_finite_floats(
    canonical_reddit_record: Dict[str, Any],
    invalid_float: float,
):
    """Verifies that non-finite floats (NaN, Inf, -Inf) are rejected."""
    doc = copy.deepcopy(canonical_reddit_record)
    doc["sentiment"]["score"] = invalid_float

    mapping = get_index_mapping()
    with pytest.raises(ValueError) as exc_info:
        assert_doc_matches_mapping_strict(doc, mapping["properties"])

    assert "must be finite" in str(exc_info.value)

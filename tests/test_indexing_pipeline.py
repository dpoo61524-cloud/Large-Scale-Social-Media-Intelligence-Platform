"""Offline integration tests for Stage 11B: Indexing Pipeline Runner & CLI Dispatch.

Validates the end-to-end indexing pipeline execution, dependency injection,
error categorization, structured manifest generation, atomic persistence,
client lifecycle management, CLI argument parser mutual exclusion, and CLI dispatch.

Key Testing Invariants:
- Strictly offline: Uses mock Elasticsearch clients; no network calls or credentials.
- Schema-v2 compliance: Validates against strict Schema-v2 envelopes and records.
- Deterministic lifecycle: Verifies externally provided clients remain open while
  factory/pipeline-owned clients are closed guaranteed.
- Safeguards: Confirms --run-all remains disabled with EXIT_ORCHESTRATION_NOT_IMPLEMENTED.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Ensure emoji module is available for importing app.py in environments lacking emoji
if "emoji" not in sys.modules:
    try:
        import emoji  # noqa: F401
    except ImportError:
        sys.modules["emoji"] = MagicMock()

from app import (
    EXIT_ORCHESTRATION_NOT_IMPLEMENTED,
    EXIT_STAGE_FAILURE,
    EXIT_SUCCESS as APP_EXIT_SUCCESS,
    build_argument_parser,
    main,
    parse_cli_arguments,
    run_index_elasticsearch_cli,
    run_orchestrated_pipeline,
)
from src.search.elasticsearch_client import ElasticsearchClient
from src.search.index_mapping import DEFAULT_INDEX_NAME
from src.search.indexing_pipeline import (
    DEFAULT_DATASET_PATH,
    DEFAULT_INDEXING_MANIFEST_PATH,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    PERMITTED_ERROR_CATEGORIES,
    PERMITTED_MANIFEST_STATUSES,
    REQUIRED_MANIFEST_FIELDS,
    IndexingManifestValidator,
    run_indexing_pipeline,
    save_manifest_atomically,
    validate_indexing_manifest,
)


def _create_valid_manifest(
    status: str = "success",
    total_records: Optional[int] = 10,
    indexed_count: int = 10,
    rejected_count: int = 0,
    failed_count: int = 0,
    error_category: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Helper to assemble a valid, compliant Stage 11 execution manifest."""
    return {
        "stage": "elasticsearch_indexing",
        "status": status,
        "dataset_path": "data/processed/consolidated_intelligence.json",
        "target_index": "consolidated_intelligence",
        "total_records": total_records,
        "indexed_count": indexed_count,
        "rejected_count": rejected_count,
        "failed_count": failed_count,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": 1.2345,
        "rejections": [],
        "indexing_failures": [],
        "error_category": error_category,
        "error_message": error_message,
    }


# =====================================================================
# Fixtures and Sample Records
# =====================================================================

@pytest.fixture
def sample_reddit_record() -> Dict[str, Any]:
    """Valid Schema-v2 Reddit intelligence record."""
    return {
        "document_id": "reddit_t3_p1001",
        "platform": "reddit",
        "native_id": "t3_p1001",
        "url": "https://reddit.com/r/datascience/comments/p1001",
        "timestamps": {
            "published_at": "2024-10-28T07:40:00+00:00",
            "collected_at": "2026-11-05T16:54:13.868878+00:00",
            "processed_at": "2026-11-06T17:27:57.464005+00:00",
        },
        "author": {
            "username": "nlp_researcher",
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
            "title_raw": "Multilingual Social Media Intelligence",
            "body_raw": "Techniques for cross-platform data processing.",
            "title_cleaned": "Multilingual Social Media Intelligence",
            "body_cleaned": "Techniques for cross-platform data processing.",
            "detected_language": "Undetermined",
            "language_code": "und",
            "preprocessing_status": "success",
        },
        "engagement": {
            "score": 150,
            "views_count": None,
            "likes_count": None,
            "comments_count": 30,
        },
        "metadata": {
            "reddit": {
                "subreddit": "datascience",
                "score": 150,
            },
            "youtube": None,
        },
        "sentiment": {
            "label": "positive",
            "score": 0.91,
            "status": "success",
        },
        "topics": {
            "topic_id": 1,
            "probability": 0.88,
            "keywords": [
                {"word": "intelligence", "weight": 0.85},
            ],
            "status": "assigned",
        },
        "misinformation": {
            "veracity_label": "real",
            "credibility_score": 0.85,
            "risk_tier": "low",
            "status": "success",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.05,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 10.0,
                "avg_word_length": 5.5,
                "lexical_subjectivity_ratio": 0.1,
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


@pytest.fixture
def sample_youtube_record() -> Dict[str, Any]:
    """Valid Schema-v2 YouTube intelligence record."""
    return {
        "document_id": "youtube_yt_v2002",
        "platform": "youtube",
        "native_id": "yt_v2002",
        "url": "https://www.youtube.com/watch?v=yt_v2002",
        "timestamps": {
            "published_at": "2024-10-29T10:00:00+00:00",
            "collected_at": "2026-11-05T16:55:00.000000+00:00",
            "processed_at": "2026-11-06T17:30:00.000000+00:00",
        },
        "author": {
            "username": "AI Explorers",
            "channel_id": "UC_aiexplorers",
            "channel_title": "AI Explorers",
            "graph_status": "unavailable",
            "influencer_score": None,
            "pagerank": None,
            "community_id": None,
            "in_degree": None,
            "out_degree": None,
        },
        "content": {
            "title_raw": "Elasticsearch Indexing Strategies",
            "body_raw": "Bulk indexing and mapping optimizations.",
            "title_cleaned": "Elasticsearch Indexing Strategies",
            "body_cleaned": "Bulk indexing and mapping optimizations.",
            "detected_language": "Undetermined",
            "language_code": "und",
            "preprocessing_status": "success",
        },
        "engagement": {
            "score": None,
            "views_count": 5000,
            "likes_count": 320,
            "comments_count": 45,
        },
        "metadata": {
            "reddit": None,
            "youtube": {
                "channel_id": "UC_aiexplorers",
                "channel_title": "AI Explorers",
            },
        },
        "sentiment": {
            "label": "positive",
            "score": 0.82,
            "status": "success",
        },
        "topics": {
            "topic_id": 2,
            "probability": 0.79,
            "keywords": [
                {"word": "indexing", "weight": 0.90},
            ],
            "status": "assigned",
        },
        "misinformation": {
            "veracity_label": "real",
            "credibility_score": 0.90,
            "risk_tier": "low",
            "status": "success",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.04,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 8.0,
                "avg_word_length": 6.0,
                "lexical_subjectivity_ratio": 0.05,
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


def _create_envelope(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to assemble a valid Schema-v2 envelope."""
    reddit_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("platform") == "reddit")
    youtube_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("platform") == "youtube")
    complete_cnt = sum(1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "complete")

    return {
        "schema_version": "2.0",
        "pipeline_version": "2.0.0",
        "status": "success",
        "consolidation_status": "complete",
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
            "degraded_records": 0,
            "partial_records": 0,
            "unprocessed_records": 0,
            "duplicates_deduplicated": 0,
            "rejected_records_count": 0,
        },
        "records": records,
    }


class MockDriver:
    """Mock Elasticsearch driver for offline testing."""

    def __init__(self) -> None:
        self.ping = MagicMock(return_value=True)
        self.indices = MagicMock()
        self.indices.exists = MagicMock(return_value=True)
        self.indices.create = MagicMock(return_value={"acknowledged": True})
        self.index = MagicMock(return_value={"result": "created", "_id": "doc_test"})
        self.bulk = MagicMock(return_value={"took": 2, "errors": False, "items": []})
        self.close = MagicMock()


@pytest.fixture
def mock_driver() -> MockDriver:
    """Provides a fresh MockDriver instance."""
    return MockDriver()


@pytest.fixture
def mock_client(mock_driver: MockDriver) -> ElasticsearchClient:
    """Provides an ElasticsearchClient instance backed by mock_driver."""
    return ElasticsearchClient(host="http://localhost:9200", client_override=mock_driver)


# =====================================================================
# 1. Pipeline Execution Tests (Success, Empty, Failure)
# =====================================================================

def test_run_indexing_pipeline_success(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
    mock_driver: MockDriver,
    sample_reddit_record: Dict[str, Any],
    sample_youtube_record: Dict[str, Any],
) -> None:
    """Pipeline executes successfully with valid Schema-v2 dataset and records."""
    records = [sample_reddit_record, sample_youtube_record]
    dataset_file = tmp_path / "consolidated_intelligence.json"
    dataset_file.write_text(json.dumps(_create_envelope(records)), encoding="utf-8")

    # Mock bulk indexing response with 2 successful items
    mock_driver.bulk.return_value = {
        "took": 5,
        "errors": False,
        "items": [
            {"index": {"_id": records[0]["document_id"], "status": 201}},
            {"index": {"_id": records[1]["document_id"], "status": 201}},
        ],
    }

    manifest_dest = tmp_path / "execution_manifest.json"

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
        manifest_path=manifest_dest,
    )

    assert exit_code == EXIT_SUCCESS
    assert manifest["stage"] == "elasticsearch_indexing"
    assert manifest["status"] == "success"
    assert manifest["total_records"] == 2
    assert manifest["indexed_count"] == 2
    assert manifest["rejected_count"] == 0
    assert manifest["failed_count"] == 0
    assert manifest["error_category"] is None
    assert manifest["error_message"] is None
    assert manifest["target_index"] == DEFAULT_INDEX_NAME

    # Verify manifest was atomically persisted
    assert manifest_dest.exists()
    with open(manifest_dest, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["status"] == "success"
    assert persisted["indexed_count"] == 2


def test_run_indexing_pipeline_empty_dataset(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Pipeline treats a valid Schema-v2 empty dataset as a successful no-op."""
    dataset_file = tmp_path / "consolidated_empty.json"
    dataset_file.write_text(json.dumps(_create_envelope([])), encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
    )

    assert exit_code == EXIT_SUCCESS
    assert manifest["status"] == "empty"
    assert manifest["total_records"] == 0
    assert manifest["indexed_count"] == 0
    assert manifest["rejected_count"] == 0
    assert manifest["failed_count"] == 0
    assert manifest["error_category"] is None


def test_run_indexing_pipeline_dataset_not_found(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Missing dataset file is safely categorized as dataset_not_found with exit code 1."""
    missing_file = tmp_path / "non_existent.json"

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=missing_file,
        client=mock_client,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] is None
    assert manifest["error_category"] == "dataset_not_found"
    assert "not found" in manifest["error_message"].lower()


def test_run_indexing_pipeline_empty_file(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Empty dataset file is safely categorized as dataset_empty with exit code 1."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("   ", encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=empty_file,
        client=mock_client,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] is None
    assert manifest["error_category"] == "dataset_empty"


def test_run_indexing_pipeline_corrupted_json(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Unparseable JSON dataset is safely categorized as corrupted_dataset with exit code 1."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{\ninvalid_json", encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=corrupt_file,
        client=mock_client,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] is None
    assert manifest["error_category"] == "corrupted_dataset"


def test_run_indexing_pipeline_envelope_validation_failure_with_readable_records(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
    sample_reddit_record: Dict[str, Any],
) -> None:
    """Envelope failure with readable records accounts all readable records as failed."""
    invalid_envelope = {
        "schema_version": "1.0",  # Invalid version for Schema-v2
        "records": [sample_reddit_record],
    }
    dataset_file = tmp_path / "invalid_envelope.json"
    dataset_file.write_text(json.dumps(invalid_envelope), encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
        validate_envelope=True,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] == 1
    assert manifest["failed_count"] == 1
    assert manifest["indexed_count"] == 0
    assert manifest["rejected_count"] == 0
    assert manifest["error_category"] == "schema_validation_failed"


def test_run_indexing_pipeline_envelope_validation_failure_missing_records(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Envelope failure with unreadable/missing records has unknown (None) total_records."""
    invalid_envelope = {
        "schema_version": "2.0",
        "pipeline_version": "2.0.0",
        # missing 'records' key entirely
    }
    dataset_file = tmp_path / "no_records.json"
    dataset_file.write_text(json.dumps(invalid_envelope), encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
        validate_envelope=True,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] is None
    assert manifest["failed_count"] == 0
    assert manifest["error_category"] == "schema_validation_failed"


def test_run_indexing_pipeline_preflight_ping_failure(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
    mock_driver: MockDriver,
    sample_reddit_record: Dict[str, Any],
) -> None:
    """Preflight ping failure accounts all established records as failed."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([sample_reddit_record])), encoding="utf-8")

    # Mock ping returning False
    mock_driver.ping.return_value = False

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] == 1
    assert manifest["failed_count"] == 1
    assert manifest["indexed_count"] == 0
    assert manifest["error_category"] == "elasticsearch_unreachable"


def test_run_indexing_pipeline_index_creation_failure(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
    mock_driver: MockDriver,
    sample_reddit_record: Dict[str, Any],
) -> None:
    """Index mapping creation failure is categorized as index_initialization_failed."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([sample_reddit_record])), encoding="utf-8")

    mock_driver.indices.exists.return_value = False
    mock_driver.indices.create.side_effect = RuntimeError("ES mapping error")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
        ensure_index=True,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["error_category"] == "index_initialization_failed"
    assert manifest["total_records"] == 1
    assert manifest["failed_count"] == 1


def test_run_indexing_pipeline_partial_success(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
    mock_driver: MockDriver,
    sample_reddit_record: Dict[str, Any],
    sample_youtube_record: Dict[str, Any],
) -> None:
    """Partial success (some indexed, some failed) returns non-zero exit code 1."""
    records = [sample_reddit_record, sample_youtube_record]
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope(records)), encoding="utf-8")

    # Mock bulk indexing returning 1 success and 1 error item
    mock_driver.bulk.return_value = {
        "took": 4,
        "errors": True,
        "items": [
            {"index": {"_id": records[0]["document_id"], "status": 201}},
            {"index": {"_id": records[1]["document_id"], "status": 400, "error": {"reason": "mapping violation"}}},
        ],
    }

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "partial_success"
    assert manifest["total_records"] == 2
    assert manifest["indexed_count"] == 1
    assert manifest["failed_count"] == 1
    assert manifest["error_category"] == "partial_indexing_failure"


def test_run_indexing_pipeline_all_records_rejected(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Dataset with completely invalid record structures reports all_records_rejected."""
    invalid_record = {"invalid_key": "not a canonical record"}
    dataset_file = tmp_path / "rejected.json"
    # Create envelope but manually set records to invalid list
    env = _create_envelope([])
    env["records"] = [invalid_record]
    env["metrics"]["total_records_consolidated"] = 1
    dataset_file.write_text(json.dumps(env), encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
        validate_envelope=False,  # bypass envelope check to hit record validation
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["total_records"] == 1
    assert manifest["rejected_count"] == 1
    assert manifest["indexed_count"] == 0
    assert manifest["error_category"] == "all_records_rejected"


def test_run_indexing_pipeline_invalid_index_name(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """Invalid index name fails early with exit code 1."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([])), encoding="utf-8")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
        index_name="INVALID_UPPERCASE_NAME",
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"


# =====================================================================
# 2. Client Lifecycle & Ownership Tests
# =====================================================================

def test_client_lifecycle_external_client_not_closed(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
    mock_driver: MockDriver,
    sample_reddit_record: Dict[str, Any],
) -> None:
    """External client passed directly is NOT closed by the pipeline."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([sample_reddit_record])), encoding="utf-8")
    mock_driver.bulk.return_value = {
        "took": 2,
        "errors": False,
        "items": [{"index": {"_id": "reddit_t3_p1001", "status": 201}}],
    }

    exit_code, _ = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=mock_client,
    )

    assert exit_code == EXIT_SUCCESS
    # External client should retain ownership; close must NOT be called
    mock_driver.close.assert_not_called()


def test_client_lifecycle_factory_client_is_closed(
    tmp_path: Path,
    sample_reddit_record: Dict[str, Any],
) -> None:
    """Client acquired via client_factory IS guaranteed to be closed by pipeline."""
    driver = MockDriver()
    driver.bulk.return_value = {
        "took": 2,
        "errors": False,
        "items": [{"index": {"_id": "reddit_t3_p1001", "status": 201}}],
    }
    factory_client = ElasticsearchClient(host="http://localhost:9200", client_override=driver)

    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([sample_reddit_record])), encoding="utf-8")

    exit_code, _ = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=None,
        client_factory=lambda: factory_client,
    )

    assert exit_code == EXIT_SUCCESS
    # Factory-created client is pipeline-owned; must be closed
    driver.close.assert_called_once()


def test_client_lifecycle_factory_client_closed_on_failure(
    tmp_path: Path,
    sample_reddit_record: Dict[str, Any],
) -> None:
    """Factory client is closed in finally block even when indexing fails."""
    driver = MockDriver()
    driver.ping.return_value = False  # Trigger failure
    factory_client = ElasticsearchClient(host="http://localhost:9200", client_override=driver)

    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([sample_reddit_record])), encoding="utf-8")

    exit_code, _ = run_indexing_pipeline(
        dataset_path=dataset_file,
        client=None,
        client_factory=lambda: factory_client,
    )

    assert exit_code == EXIT_FAILURE
    driver.close.assert_called_once()


def test_client_lifecycle_factory_exception(
    tmp_path: Path,
) -> None:
    """Exception inside client_factory is caught safely and categorizes error."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([])), encoding="utf-8")

    def failing_factory():
        raise ConnectionRefusedError("Cannot connect to Elasticsearch cluster")

    exit_code, manifest = run_indexing_pipeline(
        dataset_path=dataset_file,
        client_factory=failing_factory,
    )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["error_category"] == "client_initialization_failed"
    assert "Cannot connect" in manifest["error_message"]


# =====================================================================
# 3. Atomic Manifest Persistence and Preservation Tests
# =====================================================================

def test_atomic_manifest_persistence_success(tmp_path: Path) -> None:
    """save_manifest_atomically serializes and validates manifest without leaving temp files."""
    dest = tmp_path / "manifests" / "test_manifest.json"
    manifest_data = _create_valid_manifest(
        status="success",
        total_records=10,
        indexed_count=10,
    )

    result_path = save_manifest_atomically(manifest_data, dest)

    assert result_path.exists()
    assert result_path == dest
    with open(dest, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["stage"] == "elasticsearch_indexing"
    assert loaded["indexed_count"] == 10

    # Ensure no leftover temporary files in the directory
    temp_files = list(dest.parent.glob(".tmp_*"))
    assert temp_files == []


def test_atomic_manifest_persistence_read_back_validation_failure(tmp_path: Path) -> None:
    """Manifest missing required keys fails read-back verification and cleans up."""
    dest = tmp_path / "bad_manifest.json"
    invalid_manifest = {"unrelated_field": 123}

    with pytest.raises(ValueError, match="missing required fields"):
        save_manifest_atomically(invalid_manifest, dest)

    assert not dest.exists()
    assert list(tmp_path.glob(".tmp_*")) == []


def test_atomic_manifest_preservation_on_replace_failure(tmp_path: Path) -> None:
    """Existing manifest remains byte-for-byte unchanged when os.replace raises OSError."""
    dest = tmp_path / "existing_manifest.json"
    original_data = _create_valid_manifest(
        status="success",
        total_records=5,
        indexed_count=5,
    )
    original_content = json.dumps(original_data, indent=2, ensure_ascii=False)
    dest.write_text(original_content, encoding="utf-8")
    original_bytes = dest.read_bytes()

    replacement_data = _create_valid_manifest(
        status="success",
        total_records=20,
        indexed_count=20,
    )

    with patch("os.replace", side_effect=OSError("Simulated filesystem error during atomic replace")):
        with pytest.raises(OSError, match="Simulated filesystem error"):
            save_manifest_atomically(replacement_data, dest)

    # Assert existing manifest remains byte-for-byte identical
    assert dest.read_bytes() == original_bytes
    assert dest.read_text(encoding="utf-8") == original_content

    # Assert temporary files are cleaned up
    assert list(dest.parent.glob(".tmp_*")) == []


def test_atomic_manifest_preservation_on_read_back_validation_failure(tmp_path: Path) -> None:
    """Existing manifest remains byte-for-byte unchanged when replacement fails read-back validation."""
    dest = tmp_path / "existing_manifest.json"
    original_data = _create_valid_manifest(
        status="success",
        total_records=5,
        indexed_count=5,
    )
    original_content = json.dumps(original_data, indent=2, ensure_ascii=False)
    dest.write_text(original_content, encoding="utf-8")
    original_bytes = dest.read_bytes()

    # Replacement has count mismatch
    invalid_replacement = _create_valid_manifest(
        status="success",
        total_records=10,
        indexed_count=5,  # 10 != 5 + 0 + 0
    )

    with pytest.raises(ValueError, match="Count invariant violated"):
        save_manifest_atomically(invalid_replacement, dest)

    # Assert existing manifest remains byte-for-byte identical
    assert dest.read_bytes() == original_bytes
    assert dest.read_text(encoding="utf-8") == original_content

    # Assert temporary files are cleaned up
    assert list(dest.parent.glob(".tmp_*")) == []


def test_pipeline_preserves_existing_manifest_on_atomic_promotion_failure(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """run_indexing_pipeline preserves existing manifest byte-for-byte when atomic replace fails."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([])), encoding="utf-8")

    manifest_dest = tmp_path / "existing_manifest.json"
    original_data = _create_valid_manifest(
        status="success",
        total_records=50,
        indexed_count=50,
    )
    original_content = json.dumps(original_data, indent=2, ensure_ascii=False)
    manifest_dest.write_text(original_content, encoding="utf-8")
    original_bytes = manifest_dest.read_bytes()

    with patch("os.replace", side_effect=OSError("Simulated replace failure")):
        exit_code, manifest = run_indexing_pipeline(
            dataset_path=dataset_file,
            client=mock_client,
            manifest_path=manifest_dest,
        )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["error_category"] == "manifest_persistence_failed"

    # Destination file remains byte-for-byte unchanged
    assert manifest_dest.read_bytes() == original_bytes
    assert manifest_dest.read_text(encoding="utf-8") == original_content
    assert list(manifest_dest.parent.glob(".tmp_*")) == []


def test_run_indexing_pipeline_manifest_save_failure_fails_closed(
    tmp_path: Path,
    mock_client: ElasticsearchClient,
) -> None:
    """If manifest persistence fails, pipeline fails closed and returns EXIT_FAILURE."""
    dataset_file = tmp_path / "valid.json"
    dataset_file.write_text(json.dumps(_create_envelope([])), encoding="utf-8")

    manifest_dest = tmp_path / "manifest.json"

    with patch("os.replace", side_effect=OSError("Disk full")):
        exit_code, manifest = run_indexing_pipeline(
            dataset_path=dataset_file,
            client=mock_client,
            manifest_path=manifest_dest,
        )

    assert exit_code == EXIT_FAILURE
    assert manifest["status"] == "failed"
    assert manifest["error_category"] == "manifest_persistence_failed"


# =====================================================================
# 3.5. Manifest Schema Validation Tests (IndexingManifestValidator)
# =====================================================================

def test_manifest_validator_root_not_dict() -> None:
    """Validator rejects non-dictionary roots with TypeError."""
    with pytest.raises(TypeError, match="Manifest root must be a dict"):
        validate_indexing_manifest(["not", "a", "dict"])

    with pytest.raises(TypeError, match="Manifest root must be a dict"):
        validate_indexing_manifest("not a dict")


def test_manifest_validator_missing_required_fields() -> None:
    """Validator rejects manifests with any missing required field."""
    valid_manifest = _create_valid_manifest()
    for field in REQUIRED_MANIFEST_FIELDS:
        corrupted = dict(valid_manifest)
        del corrupted[field]
        with pytest.raises(ValueError, match="Manifest missing required fields"):
            validate_indexing_manifest(corrupted)


def test_manifest_validator_invalid_stage() -> None:
    """Validator rejects manifests with unexpected stage identifiers."""
    manifest = _create_valid_manifest()
    manifest["stage"] = "kibana_validation"
    with pytest.raises(ValueError, match="Invalid manifest stage"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_unsupported_status() -> None:
    """Validator rejects manifests with unsupported status values."""
    manifest = _create_valid_manifest()
    manifest["status"] = "in_progress"
    with pytest.raises(ValueError, match="Invalid manifest status"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_target_index_invalid() -> None:
    """Validator rejects empty or non-string target_index."""
    manifest = _create_valid_manifest()
    manifest["target_index"] = "   "
    with pytest.raises(ValueError, match="target_index must be a non-empty string"):
        validate_indexing_manifest(manifest)

    manifest["target_index"] = 123
    with pytest.raises(ValueError, match="target_index must be a non-empty string"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_dataset_path_invalid() -> None:
    """Validator rejects empty or non-string dataset_path."""
    manifest = _create_valid_manifest()
    manifest["dataset_path"] = ""
    with pytest.raises(ValueError, match="dataset_path must be a non-empty string"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_negative_counts() -> None:
    """Validator rejects negative count metrics."""
    manifest = _create_valid_manifest()
    manifest["indexed_count"] = -1
    with pytest.raises(ValueError, match="'indexed_count' must be a non-negative integer"):
        validate_indexing_manifest(manifest)

    manifest = _create_valid_manifest()
    manifest["rejected_count"] = -5
    with pytest.raises(ValueError, match="'rejected_count' must be a non-negative integer"):
        validate_indexing_manifest(manifest)

    manifest = _create_valid_manifest()
    manifest["failed_count"] = -2
    with pytest.raises(ValueError, match="'failed_count' must be a non-negative integer"):
        validate_indexing_manifest(manifest)

    manifest = _create_valid_manifest()
    manifest["total_records"] = -10
    with pytest.raises(ValueError, match="'total_records' must be a non-negative integer"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_boolean_counts_rejected() -> None:
    """Validator strictly rejects boolean values supplied for count fields."""
    for field in ("indexed_count", "rejected_count", "failed_count", "total_records"):
        manifest = _create_valid_manifest()
        manifest[field] = True
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            validate_indexing_manifest(manifest)


def test_manifest_validator_inconsistent_known_total() -> None:
    """Validator enforces total == indexed + rejected + failed when total is known."""
    manifest = _create_valid_manifest()
    manifest["total_records"] = 10
    manifest["indexed_count"] = 5
    manifest["rejected_count"] = 1
    manifest["failed_count"] = 2  # sum is 8 != 10
    with pytest.raises(ValueError, match="Count invariant violated"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_unknown_total_accepted_when_valid() -> None:
    """Validator permits total_records=None for unreadable/missing record datasets."""
    manifest = _create_valid_manifest(
        status="failed",
        total_records=None,
        indexed_count=0,
        rejected_count=0,
        failed_count=0,
        error_category="dataset_not_found",
        error_message="Dataset file not found: non_existent.json",
    )
    # Should not raise exception
    validate_indexing_manifest(manifest)


def test_manifest_validator_unknown_total_semantics() -> None:
    """Explicitly verifies the 5 contracts for total_records=None vs total_records=0:

    1. total_records=None with zero counts is accepted for artifact-level failure.
    2. Arithmetic invariant is NOT applied when total_records=None (no synthetic zero applied).
    3. total_records=0 is accepted for valid empty dataset with status 'empty'.
    4. total_records=None is rejected for 'empty' and 'success' outcomes.
    5. Boolean values (True/False) for total_records are strictly rejected.
    """
    # 1. Accepted for artifact failure with total_records=None and zero counts
    m_unknown_failure = _create_valid_manifest(
        status="failed",
        total_records=None,
        indexed_count=0,
        rejected_count=0,
        failed_count=0,
        error_category="dataset_not_found",
        error_message="Dataset file not found: non_existent.json",
    )
    validate_indexing_manifest(m_unknown_failure)

    # 2. Arithmetic invariant (total == indexed + rejected + failed) is NOT applied when total_records=None
    # If the invariant or a synthetic zero were applied, total_records=None would conflict with uncounted metrics
    assert m_unknown_failure["total_records"] is None
    validate_indexing_manifest(m_unknown_failure)

    # 3. total_records=0 is accepted for valid empty dataset with status 'empty' and no error category
    m_empty_dataset = _create_valid_manifest(
        status="empty",
        total_records=0,
        indexed_count=0,
        rejected_count=0,
        failed_count=0,
        error_category=None,
        error_message=None,
    )
    validate_indexing_manifest(m_empty_dataset)

    # 4. total_records=None is rejected for 'empty' outcome
    m_empty_with_none = _create_valid_manifest(
        status="empty",
        total_records=None,
        indexed_count=0,
        rejected_count=0,
        failed_count=0,
    )
    with pytest.raises(ValueError, match="total_records must be 0 when status is 'empty'"):
        validate_indexing_manifest(m_empty_with_none)

    # 4b. total_records=None is rejected for 'success' outcome
    m_success_with_none = _create_valid_manifest(
        status="success",
        total_records=None,
        indexed_count=10,
        rejected_count=0,
        failed_count=0,
    )
    with pytest.raises(ValueError, match="total_records must be an integer > 0 when status is 'success'"):
        validate_indexing_manifest(m_success_with_none)

    # 5. Boolean values for total_records (True and False) are strictly rejected
    m_bool_true = _create_valid_manifest()
    m_bool_true["total_records"] = True
    with pytest.raises(ValueError, match="'total_records' must be a non-negative integer or None"):
        validate_indexing_manifest(m_bool_true)

    m_bool_false = _create_valid_manifest()
    m_bool_false["total_records"] = False
    with pytest.raises(ValueError, match="'total_records' must be a non-negative integer or None"):
        validate_indexing_manifest(m_bool_false)


def test_manifest_validator_invalid_timestamps() -> None:
    """Validator rejects non-ISO or empty timestamps."""
    manifest = _create_valid_manifest()
    manifest["started_at"] = "2026-99-99T99:99:99"
    with pytest.raises(ValueError, match="Invalid ISO-8601 timestamp"):
        validate_indexing_manifest(manifest)

    manifest = _create_valid_manifest()
    manifest["completed_at"] = "invalid_timestamp"
    with pytest.raises(ValueError, match="Invalid ISO-8601 timestamp"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_invalid_duration() -> None:
    """Validator rejects negative, boolean, non-finite, or non-numeric duration."""
    manifest = _create_valid_manifest()
    manifest["duration_seconds"] = -1.0
    with pytest.raises(ValueError, match="'duration_seconds' must be a finite non-negative number"):
        validate_indexing_manifest(manifest)

    manifest["duration_seconds"] = float("inf")
    with pytest.raises(ValueError, match="'duration_seconds' must be a finite non-negative number"):
        validate_indexing_manifest(manifest)

    manifest["duration_seconds"] = True
    with pytest.raises(ValueError, match="'duration_seconds' must be a finite non-negative number"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_invalid_error_category() -> None:
    """Validator rejects unrecognized error_category strings."""
    manifest = _create_valid_manifest(
        status="failed",
        total_records=None,
        indexed_count=0,
        error_category="unrecognized_arbitrary_error",
        error_message="Some error",
    )
    with pytest.raises(ValueError, match="Invalid error_category"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_status_error_category_inconsistency() -> None:
    """Validator enforces logical consistency between status and error category/message."""
    # success with error_category
    manifest = _create_valid_manifest(
        status="success",
        total_records=10,
        indexed_count=10,
        error_category="dataset_not_found",
    )
    with pytest.raises(ValueError, match="error_category must be None when status is 'success'"):
        validate_indexing_manifest(manifest)

    # empty with error_message
    manifest = _create_valid_manifest(
        status="empty",
        total_records=0,
        indexed_count=0,
        error_message="Error message present",
    )
    with pytest.raises(ValueError, match="error_message must be None when status is 'empty'"):
        validate_indexing_manifest(manifest)

    # failed without error_category
    manifest = _create_valid_manifest(
        status="failed",
        total_records=None,
        indexed_count=0,
        error_category=None,
        error_message="Some error",
    )
    with pytest.raises(ValueError, match="error_category must be populated when status is 'failed'"):
        validate_indexing_manifest(manifest)

    # partial_success without error_message
    manifest = _create_valid_manifest(
        status="partial_success",
        total_records=10,
        indexed_count=5,
        rejected_count=0,
        failed_count=5,
        error_category="partial_indexing_failure",
        error_message=None,
    )
    with pytest.raises(ValueError, match="error_message must be a non-empty string when status is 'partial_success'"):
        validate_indexing_manifest(manifest)

    # empty with non-zero total_records
    manifest = _create_valid_manifest(
        status="empty",
        total_records=1,
        indexed_count=1,
    )
    with pytest.raises(ValueError, match="total_records must be 0 when status is 'empty'"):
        validate_indexing_manifest(manifest)

    # success with zero indexed_count
    manifest = _create_valid_manifest(
        status="success",
        total_records=0,
        indexed_count=0,
    )
    with pytest.raises(ValueError, match="total_records must be an integer > 0 when status is 'success'"):
        validate_indexing_manifest(manifest)


def test_manifest_validator_valid_manifests_accepted() -> None:
    """Validator accepts all canonical valid manifest variations."""
    # 1. Valid success manifest
    m_success = _create_valid_manifest(
        status="success",
        total_records=100,
        indexed_count=100,
        rejected_count=0,
        failed_count=0,
    )
    validate_indexing_manifest(m_success)

    # 2. Valid empty manifest
    m_empty = _create_valid_manifest(
        status="empty",
        total_records=0,
        indexed_count=0,
        rejected_count=0,
        failed_count=0,
    )
    validate_indexing_manifest(m_empty)

    # 3. Valid partial_success manifest
    m_partial = _create_valid_manifest(
        status="partial_success",
        total_records=10,
        indexed_count=8,
        rejected_count=1,
        failed_count=1,
        error_category="partial_indexing_failure",
        error_message="Indexing partially succeeded: 8 indexed, 1 rejected, 1 failed.",
    )
    validate_indexing_manifest(m_partial)

    # 4. Valid failed manifest with unknown total
    m_failed_unknown = _create_valid_manifest(
        status="failed",
        total_records=None,
        indexed_count=0,
        rejected_count=0,
        failed_count=0,
        error_category="dataset_not_found",
        error_message="Dataset file not found: non_existent.json",
    )
    validate_indexing_manifest(m_failed_unknown)

    # 5. Valid failed manifest with known total (e.g. preflight ping failure)
    m_failed_known = _create_valid_manifest(
        status="failed",
        total_records=25,
        indexed_count=0,
        rejected_count=0,
        failed_count=25,
        error_category="elasticsearch_unreachable",
        error_message="Preflight ping returned False",
    )
    validate_indexing_manifest(m_failed_known)


# =====================================================================
# 4. CLI Argument Parser & Mutual Exclusion Tests
# =====================================================================

def test_cli_parser_index_elasticsearch_flag() -> None:
    """Parser accepts --index-elasticsearch and sets index_elasticsearch=True."""
    parsed = parse_cli_arguments(["--index-elasticsearch"])
    assert parsed.index_elasticsearch is True
    assert parsed.run_all is False
    assert parsed.ingest_reddit is False
    assert parsed.preprocess_reddit is False


def test_cli_mutual_exclusion_run_all_and_index_elasticsearch() -> None:
    """--index-elasticsearch combined with --run-all must be rejected by parse_cli_arguments."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--run-all", "--index-elasticsearch"])


def test_cli_mutual_exclusion_legacy_actions_and_index_elasticsearch() -> None:
    """--index-elasticsearch combined with any other execution flag must be rejected."""
    conflicting_flags = [
        "--ingest-reddit",
        "--ingest-reddit-live",
        "--ingest-youtube",
        "--preprocess-reddit",
        "--preprocess-youtube",
        "--validate-kibana",
    ]
    for flag in conflicting_flags:
        with pytest.raises(SystemExit):
            parse_cli_arguments([flag, "--index-elasticsearch"])


def test_cli_skip_flags_rejected_without_run_all() -> None:
    """Skip flags are invalid with --index-elasticsearch and must be rejected."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--index-elasticsearch", "--skip-reddit"])

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--index-elasticsearch", "--skip-youtube"])

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--index-elasticsearch", "--skip-neo4j"])

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--index-elasticsearch", "--skip-kibana"])


def test_cli_manifest_path_rejected_without_run_all() -> None:
    """--manifest-path can only be supplied with --run-all."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--index-elasticsearch", "--manifest-path=custom.json"])


# =====================================================================
# 5. CLI Dispatch and Main Entry Point Tests
# =====================================================================

def test_run_index_elasticsearch_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_index_elasticsearch_cli executes pipeline and returns 0 on success."""
    mock_run = MagicMock(return_value=(EXIT_SUCCESS, {
        "status": "success",
        "target_index": DEFAULT_INDEX_NAME,
        "total_records": 10,
        "indexed_count": 10,
        "rejected_count": 0,
        "failed_count": 0,
    }))
    monkeypatch.setattr("src.search.indexing_pipeline.run_indexing_pipeline", mock_run)

    exit_code = run_index_elasticsearch_cli()
    assert exit_code == EXIT_SUCCESS
    mock_run.assert_called_once_with(manifest_path=DEFAULT_INDEXING_MANIFEST_PATH)


def test_cli_main_dispatches_index_elasticsearch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() invokes run_index_elasticsearch_cli and exits cleanly on success."""
    mock_cli = MagicMock(return_value=APP_EXIT_SUCCESS)
    monkeypatch.setattr("app.run_index_elasticsearch_cli", mock_cli)

    main(["--index-elasticsearch"])
    mock_cli.assert_called_once()


def test_cli_main_dispatches_index_elasticsearch_failure_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() calls sys.exit when run_index_elasticsearch_cli returns non-zero."""
    mock_cli = MagicMock(return_value=EXIT_STAGE_FAILURE)
    monkeypatch.setattr("app.run_index_elasticsearch_cli", mock_cli)

    with pytest.raises(SystemExit) as exc_info:
        main(["--index-elasticsearch"])
    assert exc_info.value.code == EXIT_STAGE_FAILURE


# =====================================================================
# 6. Unified Orchestration Dispatch & Safeguard Tests
# =====================================================================

def test_run_all_safeguard_remains_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """--run-all dispatches to the unified orchestrator and propagates exit codes without running live services."""
    mock_manifest = {"run_id": "run_mock_20261111", "overall_outcome": "success"}
    mock_runner = MagicMock(return_value=(APP_EXIT_SUCCESS, mock_manifest))
    monkeypatch.setattr("app.run_orchestrated_pipeline", mock_runner)

    # 1. Verify --run-all dispatches to unified orchestrator and returns cleanly on success
    main(["--run-all"])
    mock_runner.assert_called_once_with(["--run-all"])

    # 2. Verify exit code propagation on failure
    mock_runner.reset_mock()
    mock_runner.return_value = (EXIT_STAGE_FAILURE, {"run_id": "run_mock_fail", "overall_outcome": "failed"})
    with pytest.raises(SystemExit) as exc_info_fail:
        main(["--run-all"])
    assert exc_info_fail.value.code == EXIT_STAGE_FAILURE
    mock_runner.assert_called_once_with(["--run-all"])

    # 3. Verify --index-elasticsearch remains independently usable
    mock_index_cli = MagicMock(return_value=APP_EXIT_SUCCESS)
    monkeypatch.setattr("app.run_index_elasticsearch_cli", mock_index_cli)
    main(["--index-elasticsearch"])
    mock_index_cli.assert_called_once()

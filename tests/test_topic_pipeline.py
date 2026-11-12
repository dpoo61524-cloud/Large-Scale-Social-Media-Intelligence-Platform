"""Integration and Orchestration Tests for Advanced Topic Pipeline (Company Step 4).

Verifies end-to-end execution of run_topic_pipeline:
- Ingestion and tagging of processed Reddit and YouTube records with source_platform.
- Non-destructive execution ensuring source raw and processed records are not corrupted.
- Topic assignment, keyword enrichment, similarity matrix, velocity/acceleration dynamics,
  and cross-platform divergence (Reddit vs. YouTube).
- Output JSON schema validation for data/processed/topic_modeling_results.json.
- Graceful handling of empty or missing input files.
"""

import json
from pathlib import Path
from typing import Any
import pytest

from src.topics.topic_modeler import SocialTopicModeler
from src.topics.topic_pipeline import (
    TOPIC_STATUS_ASSIGNED,
    TOPIC_STATUS_ERROR,
    TOPIC_STATUS_OUTLIER,
    derive_topic_status,
    normalize_topic_id,
    run_topic_pipeline,
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class MockTopicModeler:
    """Mock BERTopic modeler for fast, offline pipeline integration tests."""

    def __init__(self, topic_sequence: list[int] | None = None) -> None:
        self.topic_sequence = topic_sequence or [0, 1]
        self.embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self._fitted = True

    def fit_transform(self, documents: list[str]) -> tuple[list[int], list[float]]:
        if not documents:
            return [], []
        topics = [self.topic_sequence[i % len(self.topic_sequence)] for i in range(len(documents))]
        probs = [0.88] * len(documents)
        return topics, probs

    def get_topic_info(self) -> list[dict[str, Any]]:
        return [
            {
                "topic_id": 0,
                "name": "0_deep_learning_nlp",
                "count": 5,
                "keywords": [{"word": "nlp", "weight": 0.8}, {"word": "model", "weight": 0.7}],
            },
            {
                "topic_id": 1,
                "name": "1_data_streaming_spark",
                "count": 4,
                "keywords": [{"word": "spark", "weight": 0.85}, {"word": "stream", "weight": 0.65}],
            },
            {
                "topic_id": -1,
                "name": "-1_noise_outliers",
                "count": 1,
                "keywords": [{"word": "misc", "weight": 0.1}],
            },
        ]

    def get_topic_keywords(self, topic_id: int, top_n: int = 5) -> list[dict[str, Any]]:
        lookup = {
            0: [{"word": "nlp", "weight": 0.8}, {"word": "model", "weight": 0.7}],
            1: [{"word": "spark", "weight": 0.85}, {"word": "stream", "weight": 0.65}],
            -1: [{"word": "misc", "weight": 0.1}],
        }
        return lookup.get(topic_id, [])

    def get_dynamic_topic_keywords(self, topic_id: int, timestamps: Any = None) -> dict[str, Any]:
        return {
            "all_time": self.get_topic_keywords(topic_id),
        }

    def calculate_topic_similarity_matrix(self) -> dict[str, Any]:
        return {
            "topic_ids": [0, 1],
            "matrix": [[1.0, 0.35], [0.35, 1.0]],
            "pairwise_similarities": {"0_vs_1": 0.35},
        }


@pytest.fixture
def sample_reddit_records() -> list[dict[str, Any]]:
    """Sample preprocessed Reddit records matching Step 2 schema."""
    return [
        {
            "post_id": "t3_top1",
            "title_cleaned": "Advancements in Multilingual NLP Models",
            "text_cleaned": "Discussing transformer tokenizers and cross-lingual representation.",
            "created_utc": 1730101200,  # 2024-10-28
            "collected_at": "2026-11-05T16:54:13+00:00",
        },
        {
            "post_id": "t3_top2",
            "title_cleaned": "Scalable Streaming ETL with PySpark",
            "text_cleaned": "Best practices for micro-batching and windowed stateful aggregation.",
            "created_utc": 1730187600,  # 2024-10-29
            "collected_at": "2026-11-05T16:54:13+00:00",
        },
    ]


@pytest.fixture
def sample_youtube_records() -> list[dict[str, Any]]:
    """Sample preprocessed YouTube records matching Step 2 schema."""
    return [
        {
            "video_id": "vid_top1",
            "title_cleaned": "Data Engineering Pipelines Masterclass",
            "description_cleaned": "Comprehensive overview of Apache Spark, Kafka, and Delta Lake.",
            "published_at": "2024-10-29T12:00:00Z",  # 2024-10-29
            "collected_at": "2026-11-06T13:00:00+00:00",
        }
    ]


# ==============================================================================
# Pipeline Execution & Output Schema Validation
# ==============================================================================

def test_topic_pipeline_execution_and_schema(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies that run_topic_pipeline executes end-to-end and creates valid JSON schema."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_modeling_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    modeler = MockTopicModeler(topic_sequence=[0, 1, 1])

    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
        window="daily",
    )

    # 1. Output file exists
    assert output_path.exists()
    assert result["status"] == "success"
    assert result["total_documents_modeled"] == 3

    # 2. Check top-level payload structure
    assert "generated_at" in result
    assert "embedding_model" in result
    assert "outlier_count" in result
    assert "outlier_pct" in result
    assert "topic_info" in result
    assert "similarity_matrix" in result
    assert "trajectories" in result
    assert "cross_platform_divergence" in result
    assert "temporal_trends" in result

    # 3. Check similarity matrix
    sim_matrix = result["similarity_matrix"]
    assert "matrix" in sim_matrix
    assert len(sim_matrix["matrix"]) == 2

    # 4. Check cross-platform divergence
    cp_div = result["cross_platform_divergence"]
    assert "overall_divergence" in cp_div
    assert "topic_adoption_gaps" in cp_div

    # 5. Check topic assignments and keywords
    assert len(result["reddit_records"]) == 2
    assert len(result["youtube_records"]) == 1

    r0 = result["reddit_records"][0]
    assert r0["topic_id"] == 0
    assert r0["source_platform"] == "reddit"
    assert r0["topic_probability"] == 0.88
    assert r0["topic_status"] == TOPIC_STATUS_ASSIGNED
    assert len(r0["topic_keywords"]) > 0

    y0 = result["youtube_records"][0]
    assert y0["topic_id"] == 1
    assert y0["source_platform"] == "youtube"
    assert y0["topic_status"] == TOPIC_STATUS_ASSIGNED

    # 6. Check temporal trajectory metrics
    trajectories = result["trajectories"]
    assert "0" in trajectories
    assert "1" in trajectories
    assert "velocity" in trajectories["1"]
    assert "acceleration" in trajectories["1"]

    # 7. Check daily and weekly temporal trends
    trends = result["temporal_trends"]
    assert "daily" in trends
    assert "weekly" in trends
    assert "2024-10-28" in trends["daily"]["windows"]
    assert "2024-10-29" in trends["daily"]["windows"]


# ==============================================================================
# Non-Destructive Source Data Integrity
# ==============================================================================

def test_topic_pipeline_non_destructive_integrity(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies that source input files remain bitwise identical after pipeline run."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "output.json"

    reddit_bytes = json.dumps(sample_reddit_records, indent=2).encode("utf-8")
    youtube_bytes = json.dumps(sample_youtube_records, indent=2).encode("utf-8")

    reddit_path.write_bytes(reddit_bytes)
    youtube_path.write_bytes(youtube_bytes)

    modeler = MockTopicModeler()
    run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    # Input files must match byte-for-byte
    assert reddit_path.read_bytes() == reddit_bytes
    assert youtube_path.read_bytes() == youtube_bytes


# ==============================================================================
# Missing Inputs and Workspace File Runs
# ==============================================================================

def test_topic_pipeline_handles_missing_inputs(tmp_path: Path) -> None:
    """Verifies pipeline handles non-existent input files gracefully without crashing."""
    output_path = tmp_path / "empty_output.json"
    modeler = MockTopicModeler()

    result = run_topic_pipeline(
        reddit_input_path=tmp_path / "non_existent_reddit.json",
        youtube_input_path=tmp_path / "non_existent_youtube.json",
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert result["status"] == "success"
    assert result["total_documents_modeled"] == 0
    assert output_path.exists()
    assert result["similarity_matrix"]["topic_ids"] == []


def test_topic_pipeline_runs_on_actual_workspace_files(tmp_path: Path) -> None:
    """Tests executing pipeline on actual workspace processed data."""
    workspace_reddit = Path("data/processed/reddit_posts_processed.json")
    workspace_youtube = Path("data/processed/youtube_videos_processed.json")
    output_path = tmp_path / "topic_modeling_results.json"

    if not workspace_reddit.exists():
        pytest.skip("Workspace processed Reddit file not present.")

    modeler = MockTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=workspace_reddit,
        youtube_input_path=workspace_youtube,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert result["status"] == "success"
    assert result["total_documents_modeled"] > 0
    assert output_path.exists()
    assert "daily" in result["temporal_trends"]
    assert "weekly" in result["temporal_trends"]
    assert len(result["topic_info"]) > 0
    assert "overall_divergence" in result["cross_platform_divergence"]


# ==============================================================================
# Block 6B: Topic Output Contract Alignment & topic_status Regression Tests
# ==============================================================================

@pytest.mark.parametrize("valid_id", [0, 1, 2, 42, 100, 9999])
def test_derive_topic_status_valid_assigned(valid_id: int) -> None:
    """Verifies that non-negative integer topic IDs map deterministically to 'assigned'."""
    assert derive_topic_status(valid_id) == TOPIC_STATUS_ASSIGNED


def test_derive_topic_status_valid_outlier() -> None:
    """Verifies that outlier topic ID -1 maps deterministically to 'outlier'."""
    assert derive_topic_status(-1) == TOPIC_STATUS_OUTLIER


@pytest.mark.parametrize(
    "invalid_id",
    [
        -2,
        -99,
        True,
        False,
        0.0,
        1.0,
        -1.0,
        float("nan"),
        float("inf"),
        "0",
        "1",
        "-1",
        "assigned",
        "outlier",
        None,
        [],
        {},
    ],
)
def test_derive_topic_status_invalid_ids_mapped_to_error(invalid_id: Any) -> None:
    """Verifies that invalid, boolean, float, string, or negative < -1 IDs map to 'error'.

    Invalid IDs must NEVER be silently classified as 'assigned' or 'outlier'.
    """
    status = derive_topic_status(invalid_id)
    assert status == TOPIC_STATUS_ERROR
    assert status != TOPIC_STATUS_ASSIGNED
    assert status != TOPIC_STATUS_OUTLIER


def test_topic_pipeline_reddit_assigned_status(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
) -> None:
    """Verifies Reddit record assigned to a valid topic receives 'assigned' status."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "empty_youtube.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text("[]", encoding="utf-8")

    modeler = MockTopicModeler(topic_sequence=[0, 2])
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert len(result["reddit_records"]) == 2
    r0 = result["reddit_records"][0]
    r1 = result["reddit_records"][1]

    # Status, IDs, probabilities, and platform attribution
    assert r0["topic_status"] == TOPIC_STATUS_ASSIGNED
    assert r0["topic_id"] == 0
    assert r0["topic_probability"] == 0.88
    assert r0["source_platform"] == "reddit"

    assert r1["topic_status"] == TOPIC_STATUS_ASSIGNED
    assert r1["topic_id"] == 2
    assert r1["topic_probability"] == 0.88
    assert r1["source_platform"] == "reddit"


def test_topic_pipeline_youtube_assigned_status(
    tmp_path: Path,
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies YouTube record assigned to a valid topic receives 'assigned' status."""
    reddit_path = tmp_path / "empty_reddit.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text("[]", encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    modeler = MockTopicModeler(topic_sequence=[3])
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert len(result["youtube_records"]) == 1
    y0 = result["youtube_records"][0]

    assert y0["topic_status"] == TOPIC_STATUS_ASSIGNED
    assert y0["topic_id"] == 3
    assert y0["topic_probability"] == 0.88
    assert y0["source_platform"] == "youtube"


def test_topic_pipeline_reddit_outlier_status(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
) -> None:
    """Verifies Reddit outlier record receives 'outlier' status when topic_id == -1."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "empty_youtube.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text("[]", encoding="utf-8")

    modeler = MockTopicModeler(topic_sequence=[-1])
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    for rec in result["reddit_records"]:
        assert rec["topic_id"] == -1
        assert rec["topic_status"] == TOPIC_STATUS_OUTLIER
        assert rec["source_platform"] == "reddit"
        assert rec["topic_probability"] == 0.88


def test_topic_pipeline_youtube_outlier_status(
    tmp_path: Path,
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies YouTube outlier record receives 'outlier' status when topic_id == -1."""
    reddit_path = tmp_path / "empty_reddit.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text("[]", encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    modeler = MockTopicModeler(topic_sequence=[-1])
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    for rec in result["youtube_records"]:
        assert rec["topic_id"] == -1
        assert rec["topic_status"] == TOPIC_STATUS_OUTLIER
        assert rec["source_platform"] == "youtube"
        assert rec["topic_probability"] == 0.88


def test_topic_pipeline_every_enriched_record_has_topic_status(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies that every enriched record across both platforms contains valid topic_status."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    # Sequence with mixed assigned (0, 1) and outlier (-1)
    modeler = MockTopicModeler(topic_sequence=[0, -1, 1])
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    all_enriched = result["reddit_records"] + result["youtube_records"]
    assert len(all_enriched) == 3

    for rec in all_enriched:
        assert "topic_status" in rec
        assert isinstance(rec["topic_status"], str)
        assert rec["topic_status"] in {TOPIC_STATUS_ASSIGNED, TOPIC_STATUS_OUTLIER}
        if rec["topic_id"] >= 0:
            assert rec["topic_status"] == TOPIC_STATUS_ASSIGNED
        elif rec["topic_id"] == -1:
            assert rec["topic_status"] == TOPIC_STATUS_OUTLIER


def test_topic_pipeline_topic_ids_and_probabilities_remain_unchanged(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies that topic IDs, probabilities, and timestamps are not altered by status addition."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    modeler = MockTopicModeler(topic_sequence=[5, 10, -1])
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    r0 = result["reddit_records"][0]
    r1 = result["reddit_records"][1]
    y0 = result["youtube_records"][0]

    assert r0["topic_id"] == 5
    assert r0["topic_probability"] == 0.88
    assert r0["topic_status"] == TOPIC_STATUS_ASSIGNED

    assert r1["topic_id"] == 10
    assert r1["topic_probability"] == 0.88
    assert r1["topic_status"] == TOPIC_STATUS_ASSIGNED

    assert y0["topic_id"] == -1
    assert y0["topic_probability"] == 0.88
    assert y0["topic_status"] == TOPIC_STATUS_OUTLIER


def test_topic_pipeline_source_attribution_remains_unchanged(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies source_platform attribution remains strictly 'reddit' and 'youtube'."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    modeler = MockTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    for r in result["reddit_records"]:
        assert r["source_platform"] == "reddit"
        assert "post_id" in r
    for y in result["youtube_records"]:
        assert y["source_platform"] == "youtube"
        assert "video_id" in y


def test_topic_pipeline_input_records_not_destructively_modified_in_memory(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies that in-memory input record dictionaries are not mutated by the pipeline."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_copy = [dict(r) for r in sample_reddit_records]
    youtube_copy = [dict(y) for y in sample_youtube_records]

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    modeler = MockTopicModeler()
    run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    # In-memory original objects must not have any topic enrichment keys
    for r in sample_reddit_records:
        assert "topic_status" not in r
        assert "topic_id" not in r
        assert "source_platform" not in r
    for y in sample_youtube_records:
        assert "topic_status" not in y
        assert "topic_id" not in y
        assert "source_platform" not in y

    assert sample_reddit_records == reddit_copy
    assert sample_youtube_records == youtube_copy


def test_topic_pipeline_zero_record_schema_preserved(tmp_path: Path) -> None:
    """Verifies zero-record output schema is preserved with empty record lists and status success."""
    empty_r = tmp_path / "empty_reddit.json"
    empty_y = tmp_path / "empty_youtube.json"
    output_path = tmp_path / "topic_results.json"

    empty_r.write_text("[]", encoding="utf-8")
    empty_y.write_text("[]", encoding="utf-8")

    modeler = MockTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=empty_r,
        youtube_input_path=empty_y,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert result["status"] == "success"
    assert result["total_documents_modeled"] == 0
    assert result["reddit_records"] == []
    assert result["youtube_records"] == []
    assert result["topic_info"] == []
    assert result["similarity_matrix"] == {"topic_ids": [], "matrix": [], "pairwise_similarities": {}}
    assert result["trajectories"] == {}
    assert result["temporal_trends"] == {}

    # File on disk is valid JSON matching in-memory payload
    disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert disk_payload == result


def test_topic_pipeline_invalid_topic_ids_cannot_silently_receive_incorrect_status(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
) -> None:
    """Verifies that if a modeler returns an invalid topic ID, it maps to 'error'."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "empty_youtube.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text("[]", encoding="utf-8")

    class CorruptTopicModeler(MockTopicModeler):
        def fit_transform(self, documents: list[str]) -> tuple[list[int], list[float]]:
            # Return invalid IDs: -2 (invalid negative) and "invalid" (non-integer)
            return [-2, "invalid"], [0.5, 0.5]  # type: ignore

    modeler = CorruptTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert len(result["reddit_records"]) == 2
    for rec in result["reddit_records"]:
        assert rec["topic_status"] == TOPIC_STATUS_ERROR
        assert rec["topic_status"] != TOPIC_STATUS_ASSIGNED
        assert rec["topic_status"] != TOPIC_STATUS_OUTLIER


def test_topic_pipeline_downstream_consolidator_compatibility(tmp_path: Path) -> None:
    """Verifies that topic pipeline output is fully compatible with downstream consolidator."""
    from src.pipeline.consolidator import consolidate_social_intelligence

    # 1. Prepare Step 2 preprocessed fixtures matching native IDs
    reddit_data = [
        {
            "post_id": "r_1001",
            "subreddit": "MachineLearning",
            "title": "BERT vs Transformers for NLP",
            "text": "Comparing c-TF-IDF keyword extraction and clustering.",
            "title_cleaned": "BERT vs Transformers for NLP",
            "text_cleaned": "Comparing c-TF-IDF keyword extraction and clustering.",
            "author": "alice_dev",
            "created_utc": 1730101200,
            "score": 10,
            "num_comments": 2,
            "url": "https://reddit.com/r/ml/1",
            "collected_at": "2026-11-06T20:00:00Z",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-06T20:05:00Z",
        }
    ]
    youtube_data = [
        {
            "video_id": "y_2001",
            "channel_title": "DataEngineeringHub",
            "title": "Building Scalable Data Pipelines",
            "description": "PySpark streaming and Delta Lake architecture.",
            "title_cleaned": "Building Scalable Data Pipelines",
            "description_cleaned": "PySpark streaming and Delta Lake architecture.",
            "published_at": "2026-11-06T20:00:00Z",
            "view_count": 500,
            "like_count": 40,
            "comment_count": 5,
            "url": "https://youtube.com/watch?v=y_2001",
            "collected_at": "2026-11-06T20:00:00Z",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-06T20:05:00Z",
        }
    ]

    r_file = tmp_path / "reddit_processed.json"
    y_file = tmp_path / "youtube_processed.json"
    r_file.write_text(json.dumps(reddit_data, indent=2), encoding="utf-8")
    y_file.write_text(json.dumps(youtube_data, indent=2), encoding="utf-8")

    # 2. Run topic pipeline
    t_file = tmp_path / "topic_modeling_results.json"
    modeler = MockTopicModeler(topic_sequence=[0, 1])
    run_topic_pipeline(
        reddit_input_path=r_file,
        youtube_input_path=y_file,
        output_path=t_file,
        modeler=modeler,  # type: ignore
    )

    # 3. Create minimal valid Step 3, 5, 6 fixtures for complete consolidation
    sent_file = tmp_path / "sentiment.json"
    sent_file.write_text(
        json.dumps({
            "status": "success",
            "reddit_records": [{"post_id": "r_1001", "sentiment_label": "positive", "sentiment_score": 0.95, "sentiment_status": "success"}],
            "youtube_records": [{"video_id": "y_2001", "sentiment_label": "neutral", "sentiment_score": 0.85, "sentiment_status": "success"}],
        }),
        encoding="utf-8",
    )
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(
        json.dumps({
            "status": "success",
            "top_influencers": [{"username": "alice_dev", "platform": "reddit", "influencer_score": 0.9, "pagerank": 0.05, "community_id": 1, "in_degree_raw": 2, "out_degree_raw": 1}],
            "user_metrics": {"alice_dev": {"in_degree": 2, "out_degree": 1}},
        }),
        encoding="utf-8",
    )
    mis_file = tmp_path / "misinformation.json"
    mis_file.write_text(
        json.dumps({
            "status": "success",
            "reddit_records": [{"post_id": "r_1001", "veracity_label": "real", "credibility_score": 0.88, "risk_tier": "low", "misinformation_status": "success", "stylometric_signals": {}}],
            "youtube_records": [{"video_id": "y_2001", "veracity_label": "real", "credibility_score": 0.82, "risk_tier": "low", "misinformation_status": "success", "stylometric_signals": {}}],
        }),
        encoding="utf-8",
    )

    # 4. Run consolidation
    cons_output = tmp_path / "consolidated.json"
    res = consolidate_social_intelligence(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        sentiment_path=sent_file,
        topics_path=t_file,
        graph_path=graph_file,
        misinformation_path=mis_file,
        output_path=cons_output,
        execution_time="2026-11-09T12:00:00Z",
    )

    assert res["status"] == "success"
    assert res["consolidation_status"] == "complete"
    assert res["metrics"]["total_records_consolidated"] == 2
    assert res["metrics"]["complete_records"] == 2

    # Check topic fields inside consolidated records
    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    y_doc = next(d for d in res["records"] if d["platform"] == "youtube")

    assert r_doc["topics"]["topic_id"] == 0
    assert r_doc["topics"]["status"] == "assigned"
    assert r_doc["topics"]["probability"] == 0.88
    assert "topics" in r_doc["consolidation"]["sources_joined"]
    assert r_doc["consolidation"]["is_complete"] is True

    assert y_doc["topics"]["topic_id"] == 1
    assert y_doc["topics"]["status"] == "assigned"
    assert y_doc["topics"]["probability"] == 0.88
    assert "topics" in y_doc["consolidation"]["sources_joined"]
    assert y_doc["consolidation"]["is_complete"] is True


def test_topic_pipeline_downstream_consolidator_outlier_compatibility(tmp_path: Path) -> None:
    """Verifies that topic pipeline outlier output (-1) is recognized by downstream consolidator."""
    from src.pipeline.consolidator import consolidate_social_intelligence

    reddit_data = [
        {
            "post_id": "r_outlier",
            "subreddit": "datascience",
            "title": "Random outlier text",
            "text": "Unclusterable noisy text.",
            "title_cleaned": "Random outlier text",
            "text_cleaned": "Unclusterable noisy text.",
            "author": "alice_dev",
            "created_utc": 1730101200,
            "score": 5,
            "num_comments": 1,
            "url": "https://reddit.com/r/ds/outlier",
            "collected_at": "2026-11-06T20:00:00Z",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-06T20:05:00Z",
        }
    ]

    r_file = tmp_path / "reddit_processed.json"
    y_file = tmp_path / "empty_youtube.json"
    r_file.write_text(json.dumps(reddit_data, indent=2), encoding="utf-8")
    y_file.write_text("[]", encoding="utf-8")

    # Modeler returns outlier -1
    t_file = tmp_path / "topic_results.json"
    modeler = MockTopicModeler(topic_sequence=[-1])
    run_topic_pipeline(
        reddit_input_path=r_file,
        youtube_input_path=y_file,
        output_path=t_file,
        modeler=modeler,  # type: ignore
    )

    sent_file = tmp_path / "sentiment.json"
    sent_file.write_text(
        json.dumps({
            "status": "success",
            "reddit_records": [{"post_id": "r_outlier", "sentiment_label": "neutral", "sentiment_score": 0.5, "sentiment_status": "success"}],
            "youtube_records": [],
        }),
        encoding="utf-8",
    )
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(
        json.dumps({
            "status": "success",
            "top_influencers": [{"username": "alice_dev", "platform": "reddit", "influencer_score": 0.5, "pagerank": 0.01, "community_id": 0, "in_degree_raw": 1, "out_degree_raw": 1}],
            "user_metrics": {"alice_dev": {"in_degree": 1, "out_degree": 1}},
        }),
        encoding="utf-8",
    )
    mis_file = tmp_path / "misinfo.json"
    mis_file.write_text(
        json.dumps({
            "status": "success",
            "reddit_records": [{"post_id": "r_outlier", "veracity_label": "real", "credibility_score": 0.9, "risk_tier": "low", "misinformation_status": "success", "stylometric_signals": {}}],
            "youtube_records": [],
        }),
        encoding="utf-8",
    )

    cons_output = tmp_path / "consolidated.json"
    res = consolidate_social_intelligence(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        sentiment_path=sent_file,
        topics_path=t_file,
        graph_path=graph_file,
        misinformation_path=mis_file,
        output_path=cons_output,
        execution_time="2026-11-09T12:00:00Z",
    )

    assert res["status"] == "success"
    doc = res["records"][0]

    assert doc["topics"]["topic_id"] == -1
    assert doc["topics"]["status"] == "outlier"
    assert "topics" in doc["consolidation"]["sources_joined"]
    assert doc["consolidation"]["is_complete"] is True


# ==============================================================================
# Block 6B.1: NumPy Integer Compatibility & Hardening Regression Tests
# ==============================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
def test_derive_topic_status_numpy_integers() -> None:
    """Verifies that NumPy integer scalar types are supported by derive_topic_status."""
    assert derive_topic_status(np.int32(0)) == TOPIC_STATUS_ASSIGNED
    assert derive_topic_status(np.int64(5)) == TOPIC_STATUS_ASSIGNED
    assert derive_topic_status(np.int64(-1)) == TOPIC_STATUS_OUTLIER
    assert derive_topic_status(np.int32(-1)) == TOPIC_STATUS_OUTLIER
    assert derive_topic_status(np.int64(-2)) == TOPIC_STATUS_ERROR
    assert derive_topic_status(np.int32(-99)) == TOPIC_STATUS_ERROR


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
def test_derive_topic_status_numpy_booleans_rejected() -> None:
    """Verifies that NumPy boolean types (np.bool_) are explicitly rejected."""
    assert derive_topic_status(np.bool_(True)) == TOPIC_STATUS_ERROR
    assert derive_topic_status(np.bool_(False)) == TOPIC_STATUS_ERROR
    assert derive_topic_status(np.bool_(True)) != TOPIC_STATUS_ASSIGNED
    assert derive_topic_status(np.bool_(False)) != TOPIC_STATUS_ASSIGNED


def test_derive_topic_status_native_integers_preserved() -> None:
    """Verifies that native Python integer behavior remains completely unchanged."""
    assert derive_topic_status(0) == TOPIC_STATUS_ASSIGNED
    assert derive_topic_status(5) == TOPIC_STATUS_ASSIGNED
    assert derive_topic_status(100) == TOPIC_STATUS_ASSIGNED
    assert derive_topic_status(-1) == TOPIC_STATUS_OUTLIER
    assert derive_topic_status(-2) == TOPIC_STATUS_ERROR
    assert derive_topic_status(-100) == TOPIC_STATUS_ERROR


def test_derive_topic_status_python_booleans_rejected() -> None:
    """Verifies that Python booleans (True, False) are strictly rejected."""
    assert derive_topic_status(True) == TOPIC_STATUS_ERROR
    assert derive_topic_status(False) == TOPIC_STATUS_ERROR


def test_derive_topic_status_floats_and_strings_rejected() -> None:
    """Verifies that floats and strings are rejected and cannot be misclassified."""
    assert derive_topic_status(0.0) == TOPIC_STATUS_ERROR
    assert derive_topic_status(1.5) == TOPIC_STATUS_ERROR
    assert derive_topic_status(-1.0) == TOPIC_STATUS_ERROR
    assert derive_topic_status("0") == TOPIC_STATUS_ERROR
    assert derive_topic_status("5") == TOPIC_STATUS_ERROR
    assert derive_topic_status("-1") == TOPIC_STATUS_ERROR
    assert derive_topic_status("assigned") == TOPIC_STATUS_ERROR


def test_normalize_topic_id_behavior() -> None:
    """Verifies normalize_topic_id converts valid integrals and preserves invalid types."""
    # Native integers
    assert normalize_topic_id(0) == 0
    assert type(normalize_topic_id(0)) is int
    assert normalize_topic_id(-1) == -1
    assert type(normalize_topic_id(-1)) is int

    if HAS_NUMPY:
        # NumPy integers -> native int
        n_zero = normalize_topic_id(np.int32(0))
        assert n_zero == 0
        assert type(n_zero) is int

        n_five = normalize_topic_id(np.int64(5))
        assert n_five == 5
        assert type(n_five) is int

        n_out = normalize_topic_id(np.int64(-1))
        assert n_out == -1
        assert type(n_out) is int

        # NumPy bool_ preserved as bool, never converted to int
        nb_true = normalize_topic_id(np.bool_(True))
        assert nb_true is True
        assert type(nb_true) is bool

        nb_false = normalize_topic_id(np.bool_(False))
        assert nb_false is False
        assert type(nb_false) is bool

    # Python booleans preserved as bool
    assert normalize_topic_id(True) is True
    assert type(normalize_topic_id(True)) is bool

    # Floats and strings preserved as-is
    assert normalize_topic_id(1.5) == 1.5
    assert type(normalize_topic_id(1.5)) is float
    assert normalize_topic_id("0") == "0"
    assert type(normalize_topic_id("0")) is str


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
def test_topic_pipeline_numpy_integers_stored_as_native_ints_reddit_and_youtube(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies pipeline normalizes NumPy integers to native Python ints and json.dump succeeds."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    class NumpyMockTopicModeler(MockTopicModeler):
        def fit_transform(self, documents: list[str]) -> tuple[list[Any], list[float]]:
            # Return NumPy integers: np.int32(0), np.int64(5), np.int64(-1)
            np_topics = [np.int32(0), np.int64(5), np.int64(-1)]
            probs = [0.85] * len(documents)
            return np_topics, probs

    modeler = NumpyMockTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    # 1. Output file exists and was serialized without TypeError
    assert output_path.exists()
    disk_content = json.loads(output_path.read_text(encoding="utf-8"))
    assert disk_content["status"] == "success"

    # 2. Check Reddit records
    assert len(result["reddit_records"]) == 2
    r0 = result["reddit_records"][0]
    r1 = result["reddit_records"][1]

    assert r0["topic_id"] == 0
    assert type(r0["topic_id"]) is int
    assert r0["topic_status"] == TOPIC_STATUS_ASSIGNED
    assert r0["source_platform"] == "reddit"

    assert r1["topic_id"] == 5
    assert type(r1["topic_id"]) is int
    assert r1["topic_status"] == TOPIC_STATUS_ASSIGNED
    assert r1["source_platform"] == "reddit"

    # 3. Check YouTube records
    assert len(result["youtube_records"]) == 1
    y0 = result["youtube_records"][0]

    assert y0["topic_id"] == -1
    assert type(y0["topic_id"]) is int
    assert y0["topic_status"] == TOPIC_STATUS_OUTLIER
    assert y0["source_platform"] == "youtube"

    # 4. Check disk content matches in-memory record types
    assert disk_content["reddit_records"][0]["topic_id"] == 0
    assert disk_content["reddit_records"][0]["topic_status"] == "assigned"
    assert disk_content["youtube_records"][0]["topic_id"] == -1
    assert disk_content["youtube_records"][0]["topic_status"] == "outlier"


@pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
def test_topic_pipeline_numpy_booleans_rejected_reddit_and_youtube(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
    sample_youtube_records: list[dict[str, Any]],
) -> None:
    """Verifies that model output containing np.bool_ maps to error and does not crash serialization."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_records, indent=2), encoding="utf-8")

    class NumpyBoolMockTopicModeler(MockTopicModeler):
        def fit_transform(self, documents: list[str]) -> tuple[list[Any], list[float]]:
            return [np.bool_(True), np.bool_(False), np.bool_(True)], [0.5, 0.5, 0.5]

    modeler = NumpyBoolMockTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    assert output_path.exists()
    all_recs = result["reddit_records"] + result["youtube_records"]
    for rec in all_recs:
        assert rec["topic_status"] == TOPIC_STATUS_ERROR
        assert rec["topic_status"] != TOPIC_STATUS_ASSIGNED
        assert rec["topic_status"] != TOPIC_STATUS_OUTLIER
        assert isinstance(rec["topic_id"], bool)


def test_topic_pipeline_floats_and_strings_preserved_as_invalid(
    tmp_path: Path,
    sample_reddit_records: list[dict[str, Any]],
) -> None:
    """Verifies floats and strings are not silently converted to integers and receive error status."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "empty_youtube.json"
    output_path = tmp_path / "topic_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_records, indent=2), encoding="utf-8")
    youtube_path.write_text("[]", encoding="utf-8")

    class FloatStringMockTopicModeler(MockTopicModeler):
        def fit_transform(self, documents: list[str]) -> tuple[list[Any], list[float]]:
            return [1.5, "0"], [0.5, 0.5]

    modeler = FloatStringMockTopicModeler()
    result = run_topic_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        modeler=modeler,  # type: ignore
    )

    r0 = result["reddit_records"][0]
    r1 = result["reddit_records"][1]

    # Values preserved as-is, not converted to 1 or 0
    assert r0["topic_id"] == 1.5
    assert type(r0["topic_id"]) is float
    assert r0["topic_status"] == TOPIC_STATUS_ERROR

    assert r1["topic_id"] == "0"
    assert type(r1["topic_id"]) is str
    assert r1["topic_status"] == TOPIC_STATUS_ERROR

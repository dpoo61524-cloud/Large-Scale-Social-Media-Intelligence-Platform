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
from src.topics.topic_pipeline import run_topic_pipeline


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
    assert len(r0["topic_keywords"]) > 0

    y0 = result["youtube_records"][0]
    assert y0["topic_id"] == 1
    assert y0["source_platform"] == "youtube"

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

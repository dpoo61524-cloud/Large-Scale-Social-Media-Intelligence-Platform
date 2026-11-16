"""Integration and Orchestration Tests for Sentiment Pipeline (Company Step 3).

Verifies end-to-end execution of run_sentiment_pipeline:
- Ingestion of processed Reddit and YouTube records.
- Non-destructive execution ensuring raw and processed source datasets are not mutated.
- Extraction of temporal tracking metrics (daily and weekly aggregations, counts, percentages, net sentiment).
- Graceful handling of missing inputs and empty files.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import pytest

from src.sentiment.sentiment_analyzer import MultilingualSentimentAnalyzer
from src.sentiment.sentiment_pipeline import (
    extract_reddit_text,
    extract_youtube_text,
    run_sentiment_pipeline,
)


class MockSentimentPipeline:
    """Mock transformer pipeline for fast, isolated pipeline testing."""

    def __init__(self, default_label: str = "positive", default_score: float = 0.9) -> None:
        self.default_label = default_label
        self.default_score = default_score
        self.call_history: list[list[str]] = []

    def __call__(self, texts: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        self.call_history.append(texts)
        results: list[dict[str, Any]] = []
        for text in texts:
            # Deterministic label assignment based on simple keywords for testing
            lower = text.lower()
            if "great" in lower or "breakthrough" in lower or "impressive" in lower:
                results.append({"label": "positive", "score": 0.95})
            elif "terrible" in lower or "error" in lower or "downtime" in lower:
                results.append({"label": "negative", "score": 0.89})
            elif "neutral" in lower or "standard" in lower or "tutorial" in lower:
                results.append({"label": "neutral", "score": 0.78})
            else:
                results.append({"label": self.default_label, "score": self.default_score})
        return results


@pytest.fixture
def sample_reddit_processed() -> list[dict[str, Any]]:
    """Returns sample preprocessed Reddit records matching Step 2 schema."""
    return [
        {
            "post_id": "t3_sample1",
            "subreddit": "datascience",
            "title": "Exciting AI breakthrough announced today",
            "text": "Great work on the multilingual benchmark performance.",
            "title_cleaned": "Exciting AI breakthrough announced today",
            "text_cleaned": "Great work on the multilingual benchmark performance.",
            "created_utc": 1730101200,  # 2024-10-28
            "collected_at": "2026-11-05T16:54:13+00:00",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-06T13:02:35+00:00",
        },
        {
            "post_id": "t3_sample2",
            "subreddit": "MachineLearning",
            "title": "Frustrating server downtime during peak hours",
            "text": "Terrible connectivity error encountered repeatedly.",
            "title_cleaned": "Frustrating server downtime during peak hours",
            "text_cleaned": "Terrible connectivity error encountered repeatedly.",
            "created_utc": 1730187600,  # 2024-10-29
            "collected_at": "2026-11-05T16:54:13+00:00",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-06T13:02:35+00:00",
        },
    ]


@pytest.fixture
def sample_youtube_processed() -> list[dict[str, Any]]:
    """Returns sample preprocessed YouTube records matching Step 2 schema."""
    return [
        {
            "video_id": "vid_sample1",
            "title": "Standard Tutorial on Data Pipelines",
            "description": "Neutral commentary and technical architectural overview.",
            "channel_title": "Tech Academy",
            "published_at": "2026-11-01T12:00:00Z",  # 2026-11-01
            "collected_at": "2026-11-06T13:01:34+00:00",
            "title_cleaned": "Standard Tutorial on Data Pipelines",
            "description_cleaned": "Neutral commentary and technical architectural overview.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-06T13:02:35+00:00",
        }
    ]


def test_pipeline_execution_and_output_schema(
    tmp_path: Path,
    sample_reddit_processed: list[dict[str, Any]],
    sample_youtube_processed: list[dict[str, Any]],
) -> None:
    """Verifies pipeline runs end-to-end, writes valid JSON, and produces correct schema."""
    reddit_file = tmp_path / "reddit_processed.json"
    youtube_file = tmp_path / "youtube_processed.json"
    output_file = tmp_path / "sentiment_analysis_results.json"

    reddit_file.write_text(json.dumps(sample_reddit_processed, indent=2), encoding="utf-8")
    youtube_file.write_text(json.dumps(sample_youtube_processed, indent=2), encoding="utf-8")

    mock_pipe = MockSentimentPipeline()
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=mock_pipe)

    result = run_sentiment_pipeline(
        reddit_input_path=reddit_file,
        youtube_input_path=youtube_file,
        output_path=output_file,
        analyzer=analyzer,
    )

    # 1. Status and top-level structure
    assert result["status"] == "success"
    assert result["total_records_processed"] == 3
    assert output_file.exists()

    # 2. Check overall summary
    summary = result["summary"]
    assert summary["total"] == 3
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert summary["neutral"] == 1
    assert summary["undetermined"] == 0
    assert summary["positive_pct"] == pytest.approx(33.33, abs=0.01)
    assert summary["negative_pct"] == pytest.approx(33.33, abs=0.01)
    assert summary["neutral_pct"] == pytest.approx(33.33, abs=0.01)
    # Net sentiment = (1 - 1) / 3 = 0.0
    assert summary["net_sentiment"] == 0.0

    # 3. Check source-specific summaries
    assert result["reddit_summary"]["total"] == 2
    assert result["reddit_summary"]["positive"] == 1
    assert result["reddit_summary"]["negative"] == 1
    assert result["youtube_summary"]["total"] == 1
    assert result["youtube_summary"]["neutral"] == 1

    # 4. Check temporal aggregations
    temporal = result["temporal_aggregations"]
    assert "daily" in temporal
    assert "weekly" in temporal

    daily = temporal["daily"]
    assert "2024-10-28" in daily
    assert daily["2024-10-28"]["positive"] == 1
    assert daily["2024-10-28"]["net_sentiment"] == 1.0

    assert "2024-10-29" in daily
    assert daily["2024-10-29"]["negative"] == 1
    assert daily["2024-10-29"]["net_sentiment"] == -1.0

    assert "2026-11-01" in daily
    assert daily["2026-11-01"]["neutral"] == 1
    assert daily["2026-11-01"]["net_sentiment"] == 0.0

    # 5. Check enriched records
    assert len(result["reddit_records"]) == 2
    assert result["reddit_records"][0]["sentiment_label"] == "positive"
    assert result["reddit_records"][1]["sentiment_label"] == "negative"
    assert result["youtube_records"][0]["sentiment_label"] == "neutral"

    for rec in result["reddit_records"] + result["youtube_records"]:
        assert "sentiment_label" in rec
        assert "sentiment_score" in rec
        assert "sentiment_status" in rec
        assert "analyzed_at" in rec


def test_pipeline_non_destructive_integrity(
    tmp_path: Path,
    sample_reddit_processed: list[dict[str, Any]],
    sample_youtube_processed: list[dict[str, Any]],
) -> None:
    """Verifies that running the pipeline never mutates input processed or raw files."""
    reddit_file = tmp_path / "reddit_processed.json"
    youtube_file = tmp_path / "youtube_processed.json"
    output_file = tmp_path / "output.json"

    reddit_raw_bytes = json.dumps(sample_reddit_processed, indent=2).encode("utf-8")
    youtube_raw_bytes = json.dumps(sample_youtube_processed, indent=2).encode("utf-8")

    reddit_file.write_bytes(reddit_raw_bytes)
    youtube_file.write_bytes(youtube_raw_bytes)

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=MockSentimentPipeline())
    run_sentiment_pipeline(
        reddit_input_path=reddit_file,
        youtube_input_path=youtube_file,
        output_path=output_file,
        analyzer=analyzer,
    )

    # Input files must match byte-for-byte
    assert reddit_file.read_bytes() == reddit_raw_bytes
    assert youtube_file.read_bytes() == youtube_raw_bytes


def test_pipeline_handles_missing_inputs(tmp_path: Path) -> None:
    """Verifies pipeline does not crash when input files do not exist."""
    non_existent_reddit = tmp_path / "non_existent_reddit.json"
    non_existent_youtube = tmp_path / "non_existent_youtube.json"
    output_file = tmp_path / "empty_output.json"

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=MockSentimentPipeline())
    result = run_sentiment_pipeline(
        reddit_input_path=non_existent_reddit,
        youtube_input_path=non_existent_youtube,
        output_path=output_file,
        analyzer=analyzer,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] == 0
    assert result["summary"]["total"] == 0
    assert result["summary"]["net_sentiment"] == 0.0
    assert output_file.exists()


def test_pipeline_runs_on_actual_workspace_processed_files(tmp_path: Path) -> None:
    """Tests executing pipeline on actual workspace processed data if available."""
    workspace_reddit = Path("data/processed/reddit_posts_processed.json")
    workspace_youtube = Path("data/processed/youtube_videos_processed.json")
    test_output = tmp_path / "sentiment_analysis_results.json"

    if not workspace_reddit.exists():
        pytest.skip("Workspace processed reddit file not found")

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=MockSentimentPipeline())
    result = run_sentiment_pipeline(
        reddit_input_path=workspace_reddit,
        youtube_input_path=workspace_youtube,
        output_path=test_output,
        analyzer=analyzer,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] > 0
    assert test_output.exists()
    assert "daily" in result["temporal_aggregations"]
    assert "weekly" in result["temporal_aggregations"]


def test_sentiment_pipeline_all_inference_errors(
    tmp_path: Path,
    sample_reddit_processed: list[dict[str, Any]],
) -> None:
    """Verifies that when model inference fails, records get sentiment_status='error' and undetermined label."""
    reddit_file = tmp_path / "reddit_processed.json"
    output_file = tmp_path / "sentiment_error_results.json"
    reddit_file.write_text(json.dumps(sample_reddit_processed, indent=2), encoding="utf-8")

    class FailingPipeline:
        def __call__(self, texts: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
            raise RuntimeError("Model weights not available offline")

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=FailingPipeline())
    result = run_sentiment_pipeline(
        reddit_input_path=reddit_file,
        youtube_input_path=tmp_path / "empty_yt.json",
        output_path=output_file,
        analyzer=analyzer,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] == 2
    assert result["summary"]["undetermined"] == 2
    assert result["summary"]["positive"] == 0
    assert result["summary"]["negative"] == 0
    for rec in result["reddit_records"]:
        assert rec["sentiment_status"] == "error"
        assert rec["sentiment_label"] == "undetermined"
        assert rec["sentiment_score"] == 0.0


def test_sentiment_pipeline_all_empty_text_records(tmp_path: Path) -> None:
    """Verifies that records with empty/whitespace text receive sentiment_status='empty_text'."""
    reddit_file = tmp_path / "reddit_empty_text.json"
    output_file = tmp_path / "sentiment_empty_results.json"
    empty_records = [
        {"post_id": "p1", "title": "", "text": "   ", "created_utc": 1730101200},
        {"post_id": "p2", "title": "   ", "text": "", "created_utc": 1730101200},
    ]
    reddit_file.write_text(json.dumps(empty_records, indent=2), encoding="utf-8")

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=MockSentimentPipeline())
    result = run_sentiment_pipeline(
        reddit_input_path=reddit_file,
        youtube_input_path=tmp_path / "no_yt.json",
        output_path=output_file,
        analyzer=analyzer,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] == 2
    assert result["summary"]["undetermined"] == 2
    for rec in result["reddit_records"]:
        assert rec["sentiment_status"] == "empty_text"
        assert rec["sentiment_label"] == "undetermined"
        assert rec["sentiment_score"] == 0.0


def test_sentiment_pipeline_mixed_outcomes(tmp_path: Path) -> None:
    """Verifies proper classification when records contain a mixture of valid and empty texts."""
    reddit_file = tmp_path / "reddit_mixed.json"
    output_file = tmp_path / "sentiment_mixed_results.json"
    records = [
        {"post_id": "p1", "title": "Great breakthrough", "text": "Superb work", "created_utc": 1730101200},
        {"post_id": "p2", "title": "", "text": "   ", "created_utc": 1730101200},
    ]
    reddit_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=MockSentimentPipeline())
    result = run_sentiment_pipeline(
        reddit_input_path=reddit_file,
        youtube_input_path=tmp_path / "no_yt.json",
        output_path=output_file,
        analyzer=analyzer,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] == 2
    statuses = [r["sentiment_status"] for r in result["reddit_records"]]
    assert statuses == ["success", "empty_text"]
    assert result["reddit_records"][0]["sentiment_label"] == "positive"
    assert result["reddit_records"][1]["sentiment_label"] == "undetermined"


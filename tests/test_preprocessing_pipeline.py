"""Unit tests for Social Media Preprocessing Pipeline (Company Step 2)."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.preprocessing.preprocessing_pipeline import (
    preprocess_reddit_record,
    preprocess_youtube_record,
    preprocess_reddit_dataset,
    preprocess_youtube_dataset,
)


@pytest.fixture
def mock_spark_pipeline():
    """Returns a mock Spark NLP LightPipeline returning English predictions."""
    pipeline = MagicMock()
    pipeline.annotate.return_value = {"language": ["en"]}
    return pipeline


@pytest.fixture
def sample_raw_reddit_record():
    """Returns a sample raw Reddit record matching Step 1 schema."""
    return {
        "post_id": "t3_test100",
        "subreddit": "datascience",
        "title": "Excited for #AI and Spark NLP! 😍 Check out https://sparknlp.org",
        "text": "<p>We are building a large-scale data platform with Python 3.12 &amp; Spark. Visit www.spark.apache.org!</p>",
        "author": "pipeline_eng",
        "created_utc": 1730000000,
        "score": 150,
        "num_comments": 25,
        "url": "https://reddit.com/r/datascience/test100",
        "collected_at": "2026-11-05T12:00:00Z",
    }


@pytest.fixture
def sample_raw_youtube_record():
    """Returns a sample raw YouTube video record matching Step 1 schema."""
    return {
        "video_id": "vid_yt_100",
        "title": "Mastering Social Media Intelligence 🔥🚀",
        "description": "Learn NLP text cleaning, emoji demojization, and language detection at https://youtube.com/tutorial.",
        "channel_id": "UC_channel_100",
        "channel_title": "Data Science Hub",
        "published_at": "2026-11-01T15:00:00Z",
        "view_count": 50000,
        "like_count": 3200,
        "comment_count": 410,
        "url": "https://youtube.com/watch?v=vid_yt_100",
        "collected_at": "2026-11-05T12:00:00Z",
    }


def test_preprocess_reddit_record_enrichment(sample_raw_reddit_record, mock_spark_pipeline):
    """Verify single Reddit record is cleaned, emojis handled, and enriched with language fields."""
    processed = preprocess_reddit_record(sample_raw_reddit_record, pipeline_override=mock_spark_pipeline)

    # Verify original fields are preserved
    assert processed["post_id"] == "t3_test100"
    assert processed["author"] == "pipeline_eng"
    assert processed["score"] == 150
    assert processed["url"] == "https://reddit.com/r/datascience/test100"

    # Verify title processing: emoji converted, URL stripped, hashtag preserved
    assert "smiling_face_with_heart_eyes" in processed["title_cleaned"]
    assert "https://" not in processed["title_cleaned"]
    assert "#AI" in processed["title_cleaned"]

    # Verify text processing: HTML stripped, entity unescaped, URL stripped
    assert "<p>" not in processed["text_cleaned"]
    assert "&amp;" not in processed["text_cleaned"]
    assert "&" in processed["text_cleaned"]
    assert "www.spark.apache.org" not in processed["text_cleaned"]

    # Verify enrichment fields
    assert processed["detected_language"] == "English"
    assert processed["language_code"] == "en"
    assert processed["preprocessing_status"] == "success"
    assert "processed_at" in processed


def test_preprocess_youtube_record_enrichment(sample_raw_youtube_record, mock_spark_pipeline):
    """Verify single YouTube record is cleaned, emojis handled, and enriched with language fields."""
    processed = preprocess_youtube_record(sample_raw_youtube_record, pipeline_override=mock_spark_pipeline)

    # Verify original fields preserved
    assert processed["video_id"] == "vid_yt_100"
    assert processed["channel_id"] == "UC_channel_100"
    assert processed["view_count"] == 50000

    # Verify title processing: emojis converted
    assert "fire rocket" in processed["title_cleaned"]

    # Verify description processing: URL stripped
    assert "https://" not in processed["description_cleaned"]

    # Verify enrichment fields
    assert processed["detected_language"] == "English"
    assert processed["language_code"] == "en"
    assert processed["preprocessing_status"] == "success"
    assert "processed_at" in processed


def test_preprocess_reddit_dataset_batching(tmp_path, sample_raw_reddit_record, mock_spark_pipeline):
    """Verify batch processing of Reddit dataset in small configurable chunks."""
    records = []
    for i in range(5):
        rec = sample_raw_reddit_record.copy()
        rec["post_id"] = f"t3_rec_{i}"
        records.append(rec)

    raw_file = tmp_path / "reddit_raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(records, f)

    processed_file = tmp_path / "reddit_processed.json"

    # Process in batches of 2
    summary = preprocess_reddit_dataset(
        input_path=raw_file,
        output_path=processed_file,
        batch_size=2,
        pipeline_override=mock_spark_pipeline,
    )

    assert summary["status"] == "success"
    assert summary["total_records"] == 5
    assert summary["processed_records"] == 5
    assert summary["failed_records"] == 0
    assert summary["language_counts"]["English"] == 5

    # Verify processed JSON written to disk
    assert processed_file.exists()
    with open(processed_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 5

    # Verify raw file was NOT modified
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_check = json.load(f)
    assert len(raw_check) == 5
    assert "title_cleaned" not in raw_check[0]  # Raw untouched


def test_preprocess_youtube_dataset_batching(tmp_path, sample_raw_youtube_record, mock_spark_pipeline):
    """Verify batch processing of YouTube dataset."""
    records = []
    for i in range(4):
        rec = sample_raw_youtube_record.copy()
        rec["video_id"] = f"vid_{i}"
        records.append(rec)

    raw_file = tmp_path / "youtube_raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(records, f)

    processed_file = tmp_path / "youtube_processed.json"

    summary = preprocess_youtube_dataset(
        input_path=raw_file,
        output_path=processed_file,
        batch_size=2,
        pipeline_override=mock_spark_pipeline,
    )

    assert summary["status"] == "success"
    assert summary["total_records"] == 4
    assert summary["processed_records"] == 4
    assert summary["failed_records"] == 0

    assert processed_file.exists()
    with open(processed_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 4


def test_preprocessing_handles_invalid_records_without_stopping(tmp_path, sample_raw_reddit_record, mock_spark_pipeline):
    """Verify invalid records do not halt the entire dataset processing."""
    valid_record = sample_raw_reddit_record.copy()
    invalid_record = {"not_a_valid_post": True}

    raw_file = tmp_path / "mixed_raw.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump([valid_record, invalid_record], f)

    out_file = tmp_path / "mixed_out.json"

    summary = preprocess_reddit_dataset(
        input_path=raw_file,
        output_path=out_file,
        pipeline_override=mock_spark_pipeline,
    )

    assert summary["total_records"] == 2
    assert summary["processed_records"] == 1
    assert summary["failed_records"] == 1
    assert len(summary["errors"]) == 1

    with open(out_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) == 1
    assert saved[0]["post_id"] == valid_record["post_id"]


def test_preprocessing_missing_input_file(tmp_path):
    """Verify clear error message when raw input file does not exist."""
    missing_file = tmp_path / "non_existent.json"
    summary = preprocess_reddit_dataset(input_path=missing_file)

    assert summary["status"] == "error"
    assert summary["total_records"] == 0
    assert "Please run ingestion first" in summary["message"]

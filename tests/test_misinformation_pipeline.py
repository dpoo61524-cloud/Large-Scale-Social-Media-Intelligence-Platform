"""End-to-end integration tests for misinformation detection pipeline."""

import json
from pathlib import Path
import pytest

from src.misinformation.misinformation_pipeline import (
    evaluate_cross_validated,
    run_misinformation_pipeline,
)


def test_misinformation_pipeline_mock_inputs(tmp_path: Path) -> None:
    reddit_data = [
        {
            "post_id": "red_1",
            "title_cleaned": "NASA telescope captures remarkable deep space galaxies",
            "text_cleaned": "Astronomers published new peer-reviewed findings in astrophysics journal.",
            "author": "ScienceEnthusiast",
            "score": 150,
            "url": "https://reddit.com/r/science/1",
        },
        {
            "post_id": "red_2",
            "title_cleaned": "SHOCKING SECRET: Alien base discovered on dark side of moon!!!",
            "text_cleaned": "URGENT PROOF that world governments are hiding the extraterrestrial truth!",
            "author": "ConspiracyPoster",
            "score": 2,
            "url": "https://reddit.com/r/conspiracy/2",
        },
    ]
    youtube_data = [
        {
            "video_id": "yt_1",
            "title_cleaned": "Central Bank announces rate policy update",
            "description_cleaned": "Official press briefing detailing inflation data and interest rates.",
            "channel_title": "Financial News Channel",
            "view_count": 5000,
            "url": "https://youtube.com/watch?v=yt_1",
        }
    ]

    reddit_file = tmp_path / "mock_reddit.json"
    youtube_file = tmp_path / "mock_youtube.json"
    out_file = tmp_path / "output_results.json"

    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    youtube_file.write_text(json.dumps(youtube_data), encoding="utf-8")

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=youtube_file,
        output_path=out_file,
        binary=True,
    )

    assert out_file.exists()
    assert result["status"] == "success"
    assert result["total_records_processed"] == 3
    assert result["total_analyzed"] == 3
    assert "average_credibility_score" in result
    assert "flagged_high_risk_count" in result
    assert "risk_tier_distribution" in result
    assert "platform_breakdown" in result
    assert "predictions" in result
    assert "reddit_records" in result
    assert "youtube_records" in result

    # Check predictions structure
    predictions = result["predictions"]
    assert len(predictions) == 3
    for p in predictions:
        assert "id" in p
        assert "platform" in p
        assert "veracity_label" in p
        assert p["veracity_label"] in ("fake", "real")
        assert "credibility_score" in p
        assert 0.0 <= p["credibility_score"] <= 1.0
        assert p["risk_tier"] in ("low", "moderate", "high_risk")
        assert "stylometric_signals" in p
        assert p["status"] == "success"

    # Verify original fields preserved in enriched records
    assert len(result["reddit_records"]) == 2
    assert result["reddit_records"][0]["score"] == 150
    assert result["reddit_records"][0]["misinformation_status"] == "success"
    assert result["reddit_records"][0]["url"] == "https://reddit.com/r/science/1"

    assert len(result["youtube_records"]) == 1
    assert result["youtube_records"][0]["view_count"] == 5000
    assert result["youtube_records"][0]["misinformation_status"] == "success"

    # Verify non-destructive input preservation
    assert reddit_file.read_text(encoding="utf-8") == json.dumps(reddit_data)
    assert youtube_file.read_text(encoding="utf-8") == json.dumps(youtube_data)


def test_misinformation_pipeline_preserves_empty_and_whitespace_records(tmp_path: Path) -> None:
    """Verifies that empty or whitespace records are preserved with empty_text status."""
    reddit_data = [
        {
            "post_id": "valid_post",
            "title_cleaned": "Scientific report on climate modeling",
            "text_cleaned": "Comprehensive meteorological evaluation published.",
            "author": "DrClimate",
        },
        {
            "post_id": "empty_post_1",
            "title_cleaned": "",
            "text_cleaned": "   ",
            "author": "EmptyUser",
        },
        {
            "post_id": "empty_post_2",
            "title_cleaned": None,
            "text_cleaned": None,
            "author": "NullUser",
        },
    ]

    reddit_file = tmp_path / "mixed_reddit.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    out_file = tmp_path / "out_mixed.json"

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=tmp_path / "missing_yt.json",
        output_path=out_file,
        binary=True,
    )

    # 1:1 record alignment preserved
    assert result["total_records_processed"] == 3
    assert result["total_analyzed"] == 1  # only 1 was validly analyzable
    assert len(result["reddit_records"]) == 3
    assert len(result["predictions"]) == 3

    # Check empty record statuses
    empty_item_1 = result["predictions"][1]
    assert empty_item_1["id"] == "empty_post_1"
    assert empty_item_1["status"] == "empty_text"
    assert empty_item_1["veracity_label"] == "undetermined"
    assert empty_item_1["credibility_score"] == 0.0
    assert empty_item_1["risk_tier"] == "unknown"

    empty_item_2 = result["predictions"][2]
    assert empty_item_2["id"] == "empty_post_2"
    assert empty_item_2["status"] == "empty_text"


def test_misinformation_pipeline_batch_inference_error_containment(tmp_path: Path) -> None:
    """Verifies that unexpected batch inference exceptions are contained without crashing."""
    class FailingFeatureExtractor:
        def transform(self, texts):
            raise RuntimeError("Simulated tokenization matrix failure.")

    class DummyClassifier:
        pass

    reddit_data = [{"post_id": "fail_test", "title_cleaned": "Some title", "text_cleaned": "Some text"}]
    reddit_file = tmp_path / "fail_reddit.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    out_file = tmp_path / "out_fail.json"

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=tmp_path / "no_yt.json",
        output_path=out_file,
        classifier=DummyClassifier(),
        feature_extractor=FailingFeatureExtractor(),
    )

    assert result["status"] == "success"
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["status"] == "error"
    assert result["predictions"][0]["veracity_label"] == "undetermined"
    assert result["predictions"][0]["risk_tier"] == "unknown"


def test_misinformation_pipeline_six_class_mode(tmp_path: Path) -> None:
    """Verifies pipeline execution with 6-class fine-grained mode."""
    reddit_data = [
        {"post_id": "six_1", "title_cleaned": "The central bank raised benchmark interest rates.", "text_cleaned": ""},
        {"post_id": "six_2", "title_cleaned": "SHOCKING: Moon is made of cheese and hollow extraterrestrial base!", "text_cleaned": ""},
    ]
    reddit_file = tmp_path / "six_reddit.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    out_file = tmp_path / "out_six.json"

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=tmp_path / "no_yt.json",
        output_path=out_file,
        binary=False,
    )

    assert result["status"] == "success"
    preds = result["predictions"]
    valid_six_labels = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}
    for p in preds:
        assert p["veracity_label"] in valid_six_labels
        assert 0.0 <= p["credibility_score"] <= 1.0


def test_evaluate_cross_validated_no_leakage() -> None:
    """Verifies that evaluate_cross_validated computes leak-free out-of-fold metrics."""
    statements = [
        "Statement 1 with factual clean economic evidence.",
        "Statement 2 with verified scientific peer review discovery.",
        "Statement 3 with municipal infrastructure engineering records.",
        "URGENT HOAX 1: Bleach cures viral infection completely secret conspiracy!",
        "URGENT HOAX 2: Microchips tracking municipal tap water exposed shocking!",
        "URGENT HOAX 3: Alien bases on moon admitted by space organization!",
    ]
    labels = [1, 1, 1, 0, 0, 0]  # balanced binary labels

    res = evaluate_cross_validated(
        statements=statements,
        labels=labels,
        target_names=["fake", "real"],
        binary=True,
        n_splits=2,
        is_development_fixture=True,
    )

    assert res["evaluation_successful"] is True
    assert res["evaluation_method"] == "stratified_cross_validation"
    assert res["number_of_samples"] == 6
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["f1_macro"] <= 1.0
    assert "confusion_matrix" in res
    assert res["is_development_fixture"] is True


def test_misinformation_pipeline_empty_inputs(tmp_path: Path) -> None:
    empty_reddit = tmp_path / "empty_reddit.json"
    empty_youtube = tmp_path / "empty_youtube.json"
    out_file = tmp_path / "empty_results.json"

    empty_reddit.write_text(json.dumps([]), encoding="utf-8")
    empty_youtube.write_text(json.dumps([]), encoding="utf-8")

    result = run_misinformation_pipeline(
        reddit_path=empty_reddit,
        youtube_path=empty_youtube,
        output_path=out_file,
    )

    assert out_file.exists()
    assert result["status"] == "success"
    assert result["total_records_processed"] == 0
    assert result["total_analyzed"] == 0
    assert result["average_credibility_score"] == 0.0
    assert result["predictions"] == []
    assert result["reddit_records"] == []
    assert result["youtube_records"] == []


def test_misinformation_pipeline_on_workspace_processed_files(tmp_path: Path) -> None:
    out_file = tmp_path / "workspace_results.json"
    model_save = tmp_path / "saved_model.joblib"

    result = run_misinformation_pipeline(
        reddit_path="data/processed/reddit_posts_processed.json",
        youtube_path="data/processed/youtube_videos_processed.json",
        output_path=out_file,
        save_model_to=model_save,
        binary=True,
    )

    assert out_file.exists()
    assert model_save.exists()
    assert result["status"] == "success"
    assert result["total_records_processed"] == 11  # 10 reddit posts + 1 youtube video
    assert result["total_analyzed"] == 11
    assert result["platform_breakdown"]["reddit"]["total_analyzed"] == 10
    assert result["platform_breakdown"]["youtube"]["total_analyzed"] == 1
    assert len(result["reddit_records"]) == 10
    assert len(result["youtube_records"]) == 1
    assert result["benchmark_evaluation"] is not None
    assert result["benchmark_evaluation"]["evaluation_method"] == "stratified_cross_validation"
    assert result["benchmark_evaluation"]["is_development_fixture"] is True

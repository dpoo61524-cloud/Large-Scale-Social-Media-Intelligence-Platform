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
    assert result["status"] in ("success", "partial")
    assert result["total_records_processed"] == 3
    assert result["total_analyzed"] == 3
    assert "average_credibility_score" in result
    assert "flagged_high_risk_count" in result
    assert "risk_tier_distribution" in result
    assert "platform_breakdown" in result
    assert "predictions" in result
    assert "reddit_records" in result
    assert "youtube_records" in result
    assert "methodology_disclosure" in result
    assert "nature_of_predictions" in result["methodology_disclosure"]
    assert "credibility_score_definition" in result["methodology_disclosure"]

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
        assert p["evidence_basis"] in ("low_risk", "potential_risk", "pattern_flagged_risk", "insufficient_evidence")
        assert p["evidence_basis"] != "evidence_supported_risk"

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

    assert result["status"] in ("success", "partial")
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

    assert result["status"] in ("success", "partial")
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
    assert result["status"] in ("success", "empty")
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
    assert result["status"] in ("success", "partial")
    assert result["total_records_processed"] == 11  # 10 reddit posts + 1 youtube video
    assert result["total_analyzed"] == 11
    assert result["platform_breakdown"]["reddit"]["total_analyzed"] == 10
    assert result["platform_breakdown"]["youtube"]["total_analyzed"] == 1
    assert len(result["reddit_records"]) == 10
    assert len(result["youtube_records"]) == 1
    assert result["benchmark_evaluation"] is not None
    assert result["benchmark_evaluation"]["evaluation_method"] == "stratified_cross_validation"
    assert result["benchmark_evaluation"]["is_development_fixture"] is True


# ==============================================================================
# Additional Tests: Source Discrimination, Status Semantics, Failure Handling
# ==============================================================================

def test_misinformation_pipeline_partial_source_missing_youtube(tmp_path: Path) -> None:
    """Verifies that missing YouTube input produces 'partial' stage status while processing Reddit."""
    reddit_data = [{"post_id": "r_part_1", "title_cleaned": "Scientific analysis of climate trends", "text_cleaned": "Text"}]
    reddit_file = tmp_path / "reddit_part.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    out_file = tmp_path / "out_part.json"

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=tmp_path / "non_existent_yt.json",
        output_path=out_file,
    )

    assert out_file.exists()
    assert result["status"] == "partial"
    assert result["provenance"]["reddit_status"] == "loaded"
    assert result["provenance"]["youtube_status"] == "missing"
    assert result["total_records_processed"] == 1
    assert len(result["reddit_records"]) == 1
    assert len(result["youtube_records"]) == 0


def test_misinformation_pipeline_partial_source_corrupted_youtube(tmp_path: Path) -> None:
    """Verifies that corrupted YouTube input produces 'partial' stage status while processing Reddit."""
    reddit_data = [{"post_id": "r_part_2", "title_cleaned": "Astrophysics observations", "text_cleaned": "Text"}]
    reddit_file = tmp_path / "reddit_part2.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")

    corrupted_yt = tmp_path / "corrupted_yt.json"
    corrupted_yt.write_text("NOT VALID JSON {{{", encoding="utf-8")
    out_file = tmp_path / "out_part_corrupt.json"

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=corrupted_yt,
        output_path=out_file,
    )

    assert out_file.exists()
    assert result["status"] == "partial"
    assert result["provenance"]["reddit_status"] == "loaded"
    assert result["provenance"]["youtube_status"] == "corrupted"
    assert result["total_records_processed"] == 1


def test_misinformation_pipeline_fatal_error_neither_source_available(tmp_path: Path) -> None:
    """Verifies that when both sources are missing, pipeline produces diagnostic error and does NOT promote."""
    out_file = tmp_path / "out_fatal.json"

    result = run_misinformation_pipeline(
        reddit_path=tmp_path / "missing_r.json",
        youtube_path=tmp_path / "missing_y.json",
        output_path=out_file,
    )

    assert result["status"] == "error"
    assert result["total_records_processed"] == 0
    # Fail-closed: Diagnostic error artifact must NOT be promoted to disk!
    assert not out_file.exists()


def test_misinformation_pipeline_deduplicates_duplicate_records(tmp_path: Path) -> None:
    """Verifies that duplicate post_ids in upstream inputs are cleanly deduplicated."""
    reddit_data = [
        {"post_id": "dup_1", "title_cleaned": "First copy", "text_cleaned": "Text 1"},
        {"post_id": "dup_1", "title_cleaned": "Second copy", "text_cleaned": "Text 2"},
    ]
    reddit_file = tmp_path / "dup_reddit.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    empty_yt = tmp_path / "empty_yt.json"
    empty_yt.write_text(json.dumps([]), encoding="utf-8")
    out_file = tmp_path / "out_dup.json"

    result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=empty_yt,
        output_path=out_file,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] == 1
    assert len(result["reddit_records"]) == 1
    assert result["reddit_records"][0]["post_id"] == "dup_1"


def test_misinformation_pipeline_atomic_promotion_preserves_canonical_on_failure(tmp_path: Path) -> None:
    """Verifies that an existing canonical artifact is untouched if validation or promotion fails."""
    canonical_file = tmp_path / "canonical_results.json"
    canonical_file.write_text('{"existing_canonical": "data_untouched"}', encoding="utf-8")

    reddit_data = [{"post_id": "post_fail", "title_cleaned": "Some title", "text_cleaned": "Some text"}]
    reddit_file = tmp_path / "fail_reddit.json"
    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    empty_yt = tmp_path / "empty_yt.json"
    empty_yt.write_text(json.dumps([]), encoding="utf-8")

    from unittest.mock import patch
    from src.misinformation.misinformation_validator import MisinformationArtifactValidator

    # Simulate validation failure before promotion
    with patch.object(MisinformationArtifactValidator, "validate_misinformation_artifact", return_value=(False, ["Simulated error"])):
        with pytest.raises(ValueError, match="validation failed"):
            run_misinformation_pipeline(
                reddit_path=reddit_file,
                youtube_path=empty_yt,
                output_path=canonical_file,
            )

    # Existing canonical file MUST remain completely intact
    assert canonical_file.read_text(encoding="utf-8") == '{"existing_canonical": "data_untouched"}'

    # Temporary files must be cleaned up
    temp_files = list(tmp_path.glob(".tmp_*"))
    assert len(temp_files) == 0


def test_misinformation_pipeline_downstream_stage10_compatibility(tmp_path: Path) -> None:
    """End-to-end verification: Stage 10 data consolidator consumes Stage 9 output cleanly."""
    from src.pipeline.consolidator import consolidate_social_intelligence

    reddit_file = tmp_path / "r_base.json"
    youtube_file = tmp_path / "y_base.json"
    misinfo_file = tmp_path / "misinfo_out.json"
    consolidated_file = tmp_path / "consolidated_out.json"

    reddit_data = [
        {
            "post_id": "c_post_1",
            "title_cleaned": "Best practices for cleaning multilingual social media text",
            "text_cleaned": "I am working on social listening pipelines.",
            "author": "pipeline_builder",
            "score": 100,
            "url": "https://reddit.com/r/data/1",
            "preprocessing_status": "success",
            "created_utc": 1730101200,
            "collected_at": "2026-11-05T16:00:00Z",
            "processed_at": "2026-11-05T16:30:00Z",
        }
    ]
    youtube_data = [
        {
            "video_id": "c_vid_1",
            "title_cleaned": "Introduction to Large Scale Social Media Intelligence",
            "description_cleaned": "A comprehensive overview of NLP and graph databases.",
            "channel_title": "Data Insights",
            "channel_id": "UC_data_insights",
            "view_count": 5000,
            "url": "https://youtube.com/watch?v=1",
            "preprocessing_status": "success",
            "published_at": "2026-11-01T12:00:00Z",
            "collected_at": "2026-11-05T16:00:00Z",
            "processed_at": "2026-11-05T16:30:00Z",
        }
    ]

    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    youtube_file.write_text(json.dumps(youtube_data), encoding="utf-8")

    # Run Stage 9 pipeline
    mis_result = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=youtube_file,
        output_path=misinfo_file,
        binary=True,
    )
    assert mis_result["status"] == "success"
    assert misinfo_file.exists()

    # Run Stage 10 data consolidator consuming Stage 9 artifact
    cons_result = consolidate_social_intelligence(
        reddit_processed_path=reddit_file,
        youtube_processed_path=youtube_file,
        misinformation_path=misinfo_file,
        output_path=consolidated_file,
    )

    assert consolidated_file.exists()
    assert len(cons_result["records"]) == 2

    # Check that misinformation fields were cleanly populated for both platforms
    for doc in cons_result["records"]:
        assert doc["misinformation"]["status"] == "success"
        assert doc["misinformation"]["veracity_label"] in ("real", "fake")
        assert 0.0 <= doc["misinformation"]["credibility_score"] <= 1.0
        assert doc["misinformation"]["risk_tier"] in ("low", "moderate", "high_risk")
        assert doc["misinformation"]["stylometric_signals"] is not None
        assert "misinformation" in doc["consolidation"]["sources_joined"]


def test_methodology_disclosure_survives_promotion(tmp_path: Path) -> None:
    """Verifies methodology disclosure is persisted in promoted artifact file."""
    reddit_data = [{"post_id": "r1", "title_cleaned": "Hello world", "text_cleaned": "Sample text", "author": "u1"}]
    youtube_data = [{"video_id": "y1", "title_cleaned": "Video title", "description_cleaned": "Sample desc", "channel_title": "c1"}]

    reddit_file = tmp_path / "reddit.json"
    youtube_file = tmp_path / "youtube.json"
    out_file = tmp_path / "misinfo.json"

    reddit_file.write_text(json.dumps(reddit_data), encoding="utf-8")
    youtube_file.write_text(json.dumps(youtube_data), encoding="utf-8")

    res = run_misinformation_pipeline(
        reddit_path=reddit_file,
        youtube_path=youtube_file,
        output_path=out_file,
        binary=True,
    )
    assert out_file.exists()
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert "methodology_disclosure" in saved
    disc = saved["methodology_disclosure"]
    assert "nature_of_predictions" in disc
    assert "credibility_score_definition" in disc
    assert "verification_status" in disc
    assert "classifier_limitations" in disc
    assert "protected_categories" in disc



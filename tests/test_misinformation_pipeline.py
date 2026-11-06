"""End-to-end integration tests for misinformation detection pipeline."""

import json
from pathlib import Path
import pytest

from src.misinformation.misinformation_pipeline import run_misinformation_pipeline


def test_misinformation_pipeline_mock_inputs(tmp_path: Path) -> None:
    reddit_data = [
        {
            "post_id": "red_1",
            "title_cleaned": "NASA telescope captures remarkable deep space galaxies",
            "text_cleaned": "Astronomers published new peer-reviewed findings in astrophysics journal.",
            "author": "ScienceEnthusiast",
        },
        {
            "post_id": "red_2",
            "title_cleaned": "SHOCKING SECRET: Alien base discovered on dark side of moon!!!",
            "text_cleaned": "URGENT PROOF that world governments are hiding the extraterrestrial truth!",
            "author": "ConspiracyPoster",
        },
    ]
    youtube_data = [
        {
            "video_id": "yt_1",
            "title_cleaned": "Central Bank announces rate policy update",
            "description_cleaned": "Official press briefing detailing inflation data and interest rates.",
            "channel_title": "Financial News Channel",
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
    assert result["total_analyzed"] == 3
    assert "average_credibility_score" in result
    assert "flagged_high_risk_count" in result
    assert "risk_tier_distribution" in result
    assert "platform_breakdown" in result
    assert "predictions" in result

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

    # Verify non-destructive input preservation
    assert reddit_file.read_text(encoding="utf-8") == json.dumps(reddit_data)
    assert youtube_file.read_text(encoding="utf-8") == json.dumps(youtube_data)


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
    assert result["total_analyzed"] == 0
    assert result["average_credibility_score"] == 0.0
    assert result["predictions"] == []


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
    assert result["total_analyzed"] == 11  # 10 reddit posts + 1 youtube video
    assert result["platform_breakdown"]["reddit"]["total_analyzed"] == 10
    assert result["platform_breakdown"]["youtube"]["total_analyzed"] == 1
    assert result["benchmark_evaluation"] is not None

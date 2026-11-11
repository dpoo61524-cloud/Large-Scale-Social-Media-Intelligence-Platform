"""Unit and Regression Tests for Social Intelligence Data Consolidator."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from src.pipeline.consolidator import (
    ConsolidatedArtifactValidator,
    SUPPORTED_SCHEMA_VERSION,
    PIPELINE_VERSION,
    consolidate_social_intelligence,
    normalize_utc_timestamp,
    normalize_consolidator_topic_keywords,
    _promote_artifact_atomically,
    load_json_file,
)
from src.search.indexing_adapter import validate_and_normalize_record


@pytest.fixture
def sample_fixtures(tmp_path: Path):
    """Creates isolated mock datasets for all stages in tmp_path."""
    # 1. Base Reddit Processed
    reddit_processed = [
        {
            "post_id": "r_1001",
            "subreddit": "datascience",
            "title": "Discussion on LLMs",
            "text": "Detailed post body on machine learning architectures.",
            "author": "alice_dev",
            "created_utc": 1730101200,
            "score": 150,
            "num_comments": 25,
            "url": "https://reddit.com/r/datascience/comments/r_1001",
            "collected_at": "2026-11-05T12:00:00Z",
            "title_cleaned": "Discussion on LLMs",
            "text_cleaned": "Detailed post body on machine learning architectures.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-05T12:05:00Z",
        }
    ]
    r_file = tmp_path / "reddit_processed.json"
    r_file.write_text(json.dumps(reddit_processed), encoding="utf-8")

    # 2. Base YouTube Processed
    youtube_processed = [
        {
            "video_id": "y_2001",
            "title": "Intro to Data Engineering",
            "description": "Video description covering graph databases and NLP pipelines.",
            "channel_id": "UC_channel_123",
            "channel_title": "TechTalks",
            "published_at": "2026-11-01T10:00:00Z",
            "view_count": 10000,
            "like_count": 500,
            "comment_count": 40,
            "url": "https://youtube.com/watch?v=y_2001",
            "collected_at": "2026-11-05T12:00:00Z",
            "title_cleaned": "Intro to Data Engineering",
            "description_cleaned": "Video description covering graph databases and NLP pipelines.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-05T12:05:00Z",
        }
    ]
    y_file = tmp_path / "youtube_processed.json"
    y_file.write_text(json.dumps(youtube_processed), encoding="utf-8")

    # 3. Sentiment Results
    sentiment_data = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "sentiment_label": "positive",
                "sentiment_score": 0.95,
                "sentiment_status": "success",
            }
        ],
        "youtube_records": [
            {
                "video_id": "y_2001",
                "sentiment_label": "neutral",
                "sentiment_score": 0.80,
                "sentiment_status": "success",
            }
        ],
    }
    s_file = tmp_path / "sentiment.json"
    s_file.write_text(json.dumps(sentiment_data), encoding="utf-8")

    # 4. Topic Results (with explicit valid topic_status)
    topic_data = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "topic_id": 1,
                "topic_probability": 0.88,
                "topic_keywords": ["machine", "learning", "llms"],
                "topic_status": "assigned",
            }
        ],
        "youtube_records": [
            {
                "video_id": "y_2001",
                "topic_id": 2,
                "topic_probability": 0.75,
                "topic_keywords": ["databases", "pipelines", "graph"],
                "topic_status": "assigned",
            }
        ],
    }
    t_file = tmp_path / "topics.json"
    t_file.write_text(json.dumps(topic_data), encoding="utf-8")

    # 5. Graph Results
    graph_data = {
        "status": "success",
        "top_influencers": [
            {
                "username": "alice_dev",
                "platform": "reddit",
                "influencer_score": 0.85,
                "pagerank": 0.045,
                "community_id": 0,
                "in_degree_raw": 5,
                "out_degree_raw": 2,
            }
        ],
    }
    g_file = tmp_path / "graph.json"
    g_file.write_text(json.dumps(graph_data), encoding="utf-8")

    # 6. Misinformation Results
    canonical_signals_reddit = {
        "uppercase_char_ratio": 0.05,
        "exclamation_intensity": 0.0,
        "question_intensity": 0.0,
        "word_count": 8,
        "avg_word_length": 5.2,
        "lexical_subjectivity_ratio": 0.1,
        "digits_ratio": 0.0,
        "all_caps_word_ratio": 0.0,
    }
    canonical_signals_youtube = {
        "uppercase_char_ratio": 0.04,
        "exclamation_intensity": 0.0,
        "question_intensity": 0.0,
        "word_count": 10,
        "avg_word_length": 4.8,
        "lexical_subjectivity_ratio": 0.05,
        "digits_ratio": 0.0,
        "all_caps_word_ratio": 0.0,
    }
    misinfo_data = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "veracity_label": "real",
                "credibility_score": 0.90,
                "risk_tier": "low",
                "misinformation_status": "success",
                "stylometric_signals": canonical_signals_reddit,
            }
        ],
        "youtube_records": [
            {
                "video_id": "y_2001",
                "veracity_label": "real",
                "credibility_score": 0.85,
                "risk_tier": "low",
                "misinformation_status": "success",
                "stylometric_signals": canonical_signals_youtube,
            }
        ],
    }
    m_file = tmp_path / "misinfo.json"
    m_file.write_text(json.dumps(misinfo_data), encoding="utf-8")

    return {
        "reddit_path": r_file,
        "youtube_path": y_file,
        "sentiment_path": s_file,
        "topics_path": t_file,
        "graph_path": g_file,
        "misinformation_path": m_file,
        "output_path": tmp_path / "consolidated.json",
    }


def test_consolidation_complete_pipeline(sample_fixtures):
    """Test full consolidation when all Steps 2-6 are present and valid."""
    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
        execution_time="2026-11-06T22:00:00Z",
    )

    assert res["status"] == "success"
    assert res["consolidation_status"] == "complete"
    assert res["metrics"]["total_records_consolidated"] == 2
    assert res["metrics"]["complete_records"] == 2
    assert res["metrics"]["degraded_records"] == 0
    assert res["metrics"]["partial_records"] == 0

    records = res["records"]
    r_doc = next(d for d in records if d["platform"] == "reddit")
    y_doc = next(d for d in records if d["platform"] == "youtube")

    assert r_doc["document_id"] == "reddit_r_1001"
    assert r_doc["native_id"] == "r_1001"
    assert r_doc["sentiment"]["label"] == "positive"
    assert r_doc["sentiment"]["status"] == "success"
    assert r_doc["topics"]["topic_id"] == 1
    assert r_doc["topics"]["status"] == "assigned"
    assert r_doc["misinformation"]["veracity_label"] == "real"
    assert r_doc["misinformation"]["status"] == "success"
    assert r_doc["author"]["graph_status"] == "ranked_influencer"
    assert r_doc["consolidation"]["is_complete"] is True

    assert y_doc["document_id"] == "youtube_y_2001"
    assert y_doc["author"]["graph_status"] == "not_in_top_influencers"
    assert y_doc["author"]["influencer_score"] is None
    assert y_doc["consolidation"]["is_complete"] is True


def test_one_missing_base_file_preserves_other_platform(sample_fixtures, tmp_path):
    """Test 1: One missing base file preserves valid records from the other platform, status is partial."""
    missing_reddit = tmp_path / "nonexistent_reddit.json"

    res = consolidate_social_intelligence(
        reddit_processed_path=missing_reddit,
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    assert res["metrics"]["total_records_consolidated"] == 1
    assert res["records"][0]["platform"] == "youtube"
    assert res["consolidation_status"] == "partial"
    warnings = [w for w in res["audit_warnings"] if w.get("module") == "reddit_base" and w.get("status") == "missing"]
    assert len(warnings) == 1


def test_both_base_files_empty_results_in_unprocessed(tmp_path):
    """Test 2: Both base files containing empty datasets result in overall status unprocessed."""
    empty_r = tmp_path / "empty_reddit.json"
    empty_r.write_text("[]", encoding="utf-8")
    empty_y = tmp_path / "empty_youtube.json"
    empty_y.write_text("[]", encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=empty_r,
        youtube_processed_path=empty_y,
        output_path=tmp_path / "out_empty.json",
    )

    assert res["consolidation_status"] == "unprocessed"
    assert res["metrics"]["total_records_consolidated"] == 0
    assert res["metrics"]["unprocessed_records"] == 0


def test_empty_base_file_produces_audit_warning(sample_fixtures, tmp_path):
    """Test 3: Empty base file produces explicit audit warning and sets overall status to partial."""
    empty_r = tmp_path / "empty_reddit.json"
    empty_r.write_text("[]", encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=empty_r,
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    warnings = [w for w in res["audit_warnings"] if w.get("module") == "reddit_base" and w.get("status") == "empty"]
    assert len(warnings) == 1
    assert res["consolidation_status"] == "partial"
    assert res["metrics"]["total_records_consolidated"] == 1


def test_analytical_records_missing_explicit_status(sample_fixtures, tmp_path):
    """Test 4: Analytical records missing explicit status field produce missing_status and fail completeness."""
    s_bad = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "sentiment_label": "positive", "sentiment_score": 0.9}],
        "youtube_records": [],
    }
    s_file = tmp_path / "s_no_status.json"
    s_file.write_text(json.dumps(s_bad), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        sentiment_path=s_file,
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_no_status.json",
    )

    r_doc = res["records"][0]
    assert r_doc["sentiment"]["status"] == "missing_status"
    assert r_doc["consolidation"]["is_complete"] is False
    assert r_doc["consolidation"]["status"] == "partial"


def test_topic_explicit_status_validation_rules(sample_fixtures, tmp_path):
    """Test 5 & 6: Validate explicit topic status rules and pairings (assigned vs outlier vs error)."""
    # Case A: Missing/blank topic_status produces missing_status even if topic_id is present
    t_no_status = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": 1, "topic_probability": 0.9}],
    }
    f_a = tmp_path / "t_no_status.json"
    f_a.write_text(json.dumps(t_no_status), encoding="utf-8")

    res_a = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        topics_path=f_a,
        output_path=tmp_path / "out_ta.json",
    )
    doc_a = res_a["records"][0]
    assert doc_a["topics"]["status"] == "missing_status"
    assert doc_a["consolidation"]["is_complete"] is False

    # Case B: Outlier with topic_id == -1 is valid outlier
    t_outlier_ok = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": -1, "topic_status": "outlier"}],
    }
    f_b = tmp_path / "t_outlier_ok.json"
    f_b.write_text(json.dumps(t_outlier_ok), encoding="utf-8")

    res_b = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=f_b,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_tb.json",
    )
    doc_b = res_b["records"][0]
    assert doc_b["topics"]["status"] == "outlier"
    assert doc_b["topics"]["topic_id"] == -1
    assert doc_b["consolidation"]["is_complete"] is True  # Valid outlier is complete

    # Case C: Outlier paired with invalid ID (e.g. 0 or None) becomes error
    t_outlier_bad = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": 0, "topic_status": "outlier"}],
    }
    f_c = tmp_path / "t_outlier_bad.json"
    f_c.write_text(json.dumps(t_outlier_bad), encoding="utf-8")

    res_c = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        topics_path=f_c,
        output_path=tmp_path / "out_tc.json",
    )
    doc_c = res_c["records"][0]
    assert doc_c["topics"]["status"] == "error"
    assert doc_c["consolidation"]["is_complete"] is False

    # Case D: Assigned paired with topic_id == -1 becomes error
    t_assigned_bad = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": -1, "topic_status": "assigned"}],
    }
    f_d = tmp_path / "t_assigned_bad.json"
    f_d.write_text(json.dumps(t_assigned_bad), encoding="utf-8")

    res_d = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        topics_path=f_d,
        output_path=tmp_path / "out_td.json",
    )
    doc_d = res_d["records"][0]
    assert doc_d["topics"]["status"] == "error"
    assert doc_d["topics"]["topic_id"] is None
    assert doc_d["consolidation"]["is_complete"] is False

    # Case E: Assigned paired with boolean True/False or string or float becomes error
    t_assigned_bool = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": True, "topic_status": "assigned"}],
    }
    f_e = tmp_path / "t_assigned_bool.json"
    f_e.write_text(json.dumps(t_assigned_bool), encoding="utf-8")

    res_e = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        topics_path=f_e,
        output_path=tmp_path / "out_te.json",
    )
    doc_e = res_e["records"][0]
    assert doc_e["topics"]["status"] == "error"
    assert doc_e["topics"]["topic_id"] is None
    assert doc_e["consolidation"]["is_complete"] is False

    # Case F: Preserves fallback and empty_text
    t_fallback = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": -1, "topic_status": "fallback"}],
    }
    f_f = tmp_path / "t_fallback.json"
    f_f.write_text(json.dumps(t_fallback), encoding="utf-8")

    res_f = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=f_f,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_tf.json",
    )
    doc_f = res_f["records"][0]
    assert doc_f["topics"]["status"] == "fallback"
    assert doc_f["topics"]["topic_id"] is None
    assert doc_f["consolidation"]["status"] == "degraded"


def test_empty_graph_file_sets_status_unavailable(sample_fixtures, tmp_path):
    """Test 7: Empty graph file sets graph_status to unavailable with explicit audit warning."""
    empty_g = tmp_path / "empty_graph.json"
    empty_g.write_text("{}", encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=empty_g,
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    assert r_doc["author"]["graph_status"] == "unavailable"
    assert r_doc["author"]["influencer_score"] is None
    assert r_doc["consolidation"]["is_complete"] is True
    g_warnings = [w for w in res["audit_warnings"] if w.get("module") == "graph" and w.get("status") == "empty"]
    assert len(g_warnings) == 1


def test_structurally_invalid_graph_file_sets_corrupted(sample_fixtures, tmp_path):
    """Test 8: Structurally invalid graph file sets graph_status to corrupted."""
    bad_g = tmp_path / "invalid_graph.json"
    bad_g.write_text(json.dumps({"top_influencers": "not_a_list"}), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=bad_g,
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    assert r_doc["author"]["graph_status"] == "corrupted"
    assert r_doc["author"]["influencer_score"] is None
    g_warnings = [w for w in res["audit_warnings"] if w.get("module") == "graph" and w.get("status") == "corrupted"]
    assert len(g_warnings) == 1


def test_malformed_graph_entries_sets_corrupted(sample_fixtures, tmp_path):
    """Test 9: Graph list containing non-dict entries or missing username/platform sets status to corrupted."""
    bad_entries_g = tmp_path / "bad_entries_graph.json"
    bad_entries_g.write_text(json.dumps({
        "top_influencers": [
            "not_a_dict_entry",
            {"username": "valid_user"},  # Missing platform
            {"platform": "reddit"},  # Missing username
            {"username": "   ", "platform": "reddit"},  # Blank username
        ]
    }), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=bad_entries_g,
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    assert r_doc["author"]["graph_status"] == "corrupted"
    assert r_doc["author"]["influencer_score"] is None
    inv_warnings = [w for w in res["audit_warnings"] if w.get("status") == "invalid_graph_entry"]
    assert len(inv_warnings) == 4


def test_structurally_invalid_analytical_collections(sample_fixtures, tmp_path):
    """Test 10: Structurally invalid collections (null or non-dict items) log warnings and set dimension to corrupted."""
    s_invalid = tmp_path / "invalid_sentiment.json"
    s_invalid.write_text(json.dumps({
        "status": "success",
        "reddit_records": None,  # Expected list
        "youtube_records": ["not_a_dict"],  # List containing non-dict
    }), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=s_invalid,
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    assert res["status"] == "success"
    assert res["consolidation_status"] == "partial"
    # Reddit record had invalid_collection (None), YouTube record had non-dict item -> both mark sentiment corrupted
    assert res["records"][0]["sentiment"]["status"] == "corrupted"
    assert res["records"][1]["sentiment"]["status"] == "corrupted"


def test_preprocessing_status_validation_rules(sample_fixtures, tmp_path):
    """Test 11: Missing, blank, non-string, or failed preprocessing status is never marked complete."""
    cases = [
        ({"preprocessing_status": None, "text_cleaned": "Valid"}, "missing_status"),
        ({"preprocessing_status": "   ", "text_cleaned": "Valid"}, "missing_status"),
        ({"preprocessing_status": 12345, "text_cleaned": "Valid"}, "invalid_status"),
        ({"preprocessing_status": "failed", "text_cleaned": "Valid"}, "failed"),
        ({"preprocessing_status": "success", "title_cleaned": "", "text_cleaned": ""}, "success"),  # Empty text
    ]

    for idx, (overrides, expected_prep) in enumerate(cases):
        r_data = [
            {
                "post_id": f"r_case_{idx}",
                "author": "user",
                "title": "T",
                "text": "B",
                "title_cleaned": overrides.get("title_cleaned", "Clean Title"),
                "text_cleaned": overrides.get("text_cleaned", "Clean Body"),
                "preprocessing_status": overrides.get("preprocessing_status"),
            }
        ]
        rf = tmp_path / f"r_case_{idx}.json"
        rf.write_text(json.dumps(r_data), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=rf,
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=sample_fixtures["topics_path"],
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_case_{idx}.json",
        )

        doc = res["records"][0]
        assert doc["content"]["preprocessing_status"] == expected_prep
        assert doc["consolidation"]["status"] == "unprocessed"
        assert doc["consolidation"]["is_complete"] is False
        assert res["consolidation_status"] == "unprocessed"


def test_rejected_base_records_impact_on_dataset_status(sample_fixtures, tmp_path):
    """Test 12: Rejected records prevent false complete status and preserve valid records on other platforms."""
    # Reddit has ONLY invalid IDs
    r_bad = tmp_path / "r_all_invalid.json"
    r_bad.write_text(json.dumps([
        {"post_id": "", "text_cleaned": "A", "preprocessing_status": "success"},
        {"post_id": "   ", "text_cleaned": "B", "preprocessing_status": "success"},
        {"text_cleaned": "C", "preprocessing_status": "success"},
    ]), encoding="utf-8")

    # YouTube has 1 valid complete record
    res = consolidate_social_intelligence(
        reddit_processed_path=r_bad,
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_rejected_impact.json",
    )

    # Valid YouTube record is preserved and consolidated
    assert res["metrics"]["total_records_consolidated"] == 1
    assert res["records"][0]["platform"] == "youtube"
    # All 3 Reddit records rejected
    assert res["metrics"]["rejected_records_count"] == 3
    # Overall status CANNOT be complete because Reddit had total rejection
    assert res["consolidation_status"] == "partial"

    # Both platforms having only rejected records produces unprocessed
    y_bad = tmp_path / "y_all_invalid.json"
    y_bad.write_text(json.dumps([{"video_id": None}]), encoding="utf-8")

    res_both_bad = consolidate_social_intelligence(
        reddit_processed_path=r_bad,
        youtube_processed_path=y_bad,
        output_path=tmp_path / "out_both_bad.json",
    )
    assert res_both_bad["metrics"]["total_records_consolidated"] == 0
    assert res_both_bad["consolidation_status"] == "unprocessed"


def test_cross_platform_identity_and_deduplication(tmp_path):
    """Test 13: Deduplication priority and cross-platform identity isolation remain robust."""
    r_file = tmp_path / "r_test.json"
    r_file.write_text(json.dumps([
        {
            "post_id": "dup_1",
            "author": "same_user",
            "title": "Old",
            "text": "Short",
            "text_cleaned": "Short",
            "processed_at": "2026-11-05T10:00:00Z",
            "preprocessing_status": "success",
        },
        {
            "post_id": "dup_1",
            "author": "same_user",
            "title": "New",
            "text": "Longer text body.",
            "text_cleaned": "Longer text body.",
            "processed_at": "2026-11-05T12:00:00Z",
            "preprocessing_status": "success",
        },
    ]), encoding="utf-8")

    y_file = tmp_path / "y_test.json"
    y_file.write_text(json.dumps([
        {
            "video_id": "y_vid_1",
            "channel_title": "same_user",
            "title": "YT Video",
            "description": "Desc",
            "title_cleaned": "YT Video",
            "description_cleaned": "Desc",
            "preprocessing_status": "success",
        }
    ]), encoding="utf-8")

    g_file = tmp_path / "g_test.json"
    g_file.write_text(json.dumps({
        "status": "success",
        "top_influencers": [
            {
                "username": "same_user",
                "platform": "reddit",
                "influencer_score": 0.91,
            }
        ],
    }), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        graph_path=g_file,
        output_path=tmp_path / "out_test13.json",
    )

    assert res["metrics"]["total_records_consolidated"] == 2
    assert res["metrics"]["duplicates_deduplicated"] == 1

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    y_doc = next(d for d in res["records"] if d["platform"] == "youtube")

    assert r_doc["content"]["title_raw"] == "New"
    assert r_doc["author"]["graph_status"] == "ranked_influencer"
    assert r_doc["author"]["influencer_score"] == 0.91

    assert y_doc["author"]["graph_status"] == "not_in_top_influencers"
    assert y_doc["author"]["influencer_score"] is None


def test_timestamp_utc_normalization():
    """Verify UTC normalization across numeric epochs, naive ISO strings, and Z strings."""
    valid, iso_str, dt = normalize_utc_timestamp(1730101200)
    assert valid is True
    assert iso_str == "2024-10-28T07:40:00+00:00"
    assert dt.tzinfo == timezone.utc

    valid, iso_str, dt = normalize_utc_timestamp("2026-11-05T12:00:00")
    assert valid is True
    assert iso_str == "2026-11-05T12:00:00+00:00"
    assert dt.tzinfo == timezone.utc

    valid, iso_str, dt = normalize_utc_timestamp("2026-11-05T12:00:00Z")
    assert valid is True
    assert iso_str == "2026-11-05T12:00:00+00:00"
    assert dt.tzinfo == timezone.utc

    valid, iso_str, dt = normalize_utc_timestamp("invalid-date-string")
    assert valid is False
    assert iso_str is None
    assert dt == datetime.min.replace(tzinfo=timezone.utc)


def test_idempotent_repeated_execution(sample_fixtures, tmp_path):
    """Verify fixed execution timestamp produces 100% byte-for-byte identical output."""
    out_1 = tmp_path / "run_1.json"
    out_2 = tmp_path / "run_2.json"
    fixed_time = "2026-11-06T22:30:00+00:00"

    consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=out_1,
        execution_time=fixed_time,
    )

    consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=out_2,
        execution_time=fixed_time,
    )

    assert out_1.read_bytes() == out_2.read_bytes()


def test_youtube_channel_metadata_preservation(sample_fixtures, tmp_path):
    """Test preservation of channel_id and channel_title in metadata.youtube and author."""
    out_file = tmp_path / "yt_meta_out.json"
    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=out_file,
    )
    yt_doc = next(d for d in res["records"] if d["platform"] == "youtube")
    # Verify metadata.youtube retains both channel_id and channel_title
    assert yt_doc["metadata"]["youtube"]["channel_id"] == "UC_channel_123"
    assert yt_doc["metadata"]["youtube"]["channel_title"] == "TechTalks"
    # Verify author fields remain unchanged
    assert yt_doc["author"]["channel_id"] == "UC_channel_123"
    assert yt_doc["author"]["channel_title"] == "TechTalks"
    assert yt_doc["author"]["username"] == "TechTalks"


def test_non_string_and_unknown_sentiment_statuses(sample_fixtures, tmp_path):
    """Test non-string (int, bool, list, dict) and unknown string sentiment statuses map to error."""
    test_cases = [
        (12345, "error"),
        (True, "error"),
        (["success"], "error"),
        ({"status": "success"}, "error"),
        ("unknown_sentiment_state", "error"),
        ("   ", "missing_status"),
        (None, "missing_status"),
    ]

    for idx, (bad_status, expected_status) in enumerate(test_cases):
        s_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "sentiment_label": "positive",
                    "sentiment_score": 0.88,
                    "sentiment_status": bad_status,
                }
            ],
            "youtube_records": [],
        }
        s_path = tmp_path / f"s_bad_{idx}.json"
        s_path.write_text(json.dumps(s_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=s_path,
            topics_path=sample_fixtures["topics_path"],
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_s_{idx}.json",
        )
        doc = res["records"][0]
        assert doc["sentiment"]["status"] == expected_status
        assert doc["sentiment"]["label"] is None
        assert doc["sentiment"]["score"] is None
        assert doc["consolidation"]["status"] == "partial"
        assert doc["consolidation"]["is_complete"] is False


def test_non_string_and_unknown_misinformation_statuses(sample_fixtures, tmp_path):
    """Test non-string (int, bool, list, dict) and unknown string misinformation statuses map to error."""
    test_cases = [
        (999, "error"),
        (False, "error"),
        (["real"], "error"),
        ({"veracity": "real"}, "error"),
        ("unrecognized_misinfo_type", "error"),
        ("   ", "missing_status"),
        (None, "missing_status"),
    ]

    for idx, (bad_status, expected_status) in enumerate(test_cases):
        m_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "veracity_label": "real",
                    "credibility_score": 0.95,
                    "risk_tier": "low",
                    "misinformation_status": bad_status,
                }
            ],
            "youtube_records": [],
        }
        m_path = tmp_path / f"m_bad_{idx}.json"
        m_path.write_text(json.dumps(m_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=sample_fixtures["topics_path"],
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=m_path,
            output_path=tmp_path / f"out_m_{idx}.json",
        )
        doc = res["records"][0]
        assert doc["misinformation"]["status"] == expected_status
        assert doc["misinformation"]["veracity_label"] is None
        assert doc["misinformation"]["credibility_score"] is None
        assert doc["misinformation"]["risk_tier"] is None
        assert doc["misinformation"]["stylometric_signals"] is None
        assert doc["consolidation"]["status"] == "partial"
        assert doc["consolidation"]["is_complete"] is False


def test_non_string_and_unknown_topic_statuses(sample_fixtures, tmp_path):
    """Test non-string (int, bool, list, dict) and unknown topic statuses map to error with null topic_id."""
    test_cases = [
        (123, "error"),
        (True, "error"),
        (["assigned"], "error"),
        ({"topic": 1}, "error"),
        ("unexpected_topic_label", "error"),
    ]

    for idx, (bad_status, expected_status) in enumerate(test_cases):
        t_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "topic_id": 1,
                    "topic_status": bad_status,
                }
            ],
            "youtube_records": [],
        }
        t_path = tmp_path / f"t_bad_{idx}.json"
        t_path.write_text(json.dumps(t_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=t_path,
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_t_{idx}.json",
        )
        doc = res["records"][0]
        assert doc["topics"]["status"] == expected_status
        assert doc["topics"]["topic_id"] is None
        assert doc["topics"]["probability"] is None
        assert doc["topics"]["keywords"] == []
        assert doc["consolidation"]["status"] == "partial"
        assert doc["consolidation"]["is_complete"] is False


def test_analytical_entries_invalid_native_ids_and_platform_isolation(sample_fixtures, tmp_path):
    """Test analytical entries with missing, blank, or invalid native IDs mark only their own platform corrupted,
    while valid records on the other platform remain fully consolidated."""
    s_payload = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "sentiment_label": "positive",
                "sentiment_score": 0.95,
                "sentiment_status": "success",
            }
        ],
        "youtube_records": [
            {"sentiment_status": "success"},  # missing native ID
            {"video_id": None, "sentiment_status": "success"},  # None native ID
            {"video_id": "   ", "sentiment_status": "success"},  # blank native ID
            {"video_id": 12345, "sentiment_status": "success"},  # non-string native ID
        ],
    }
    s_path = tmp_path / "s_invalid_yt_entries.json"
    s_path.write_text(json.dumps(s_payload), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=s_path,
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_platform_iso.json",
    )

    inv_warnings = [
        w for w in res["audit_warnings"]
        if w.get("module") == "sentiment" and w.get("status") == "invalid_record_entry" and w.get("platform") == "youtube"
    ]
    assert len(inv_warnings) == 4

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    y_doc = next(d for d in res["records"] if d["platform"] == "youtube")

    # YouTube record has sentiment corrupted (NOT unavailable)
    assert y_doc["sentiment"]["status"] == "corrupted"
    assert y_doc["consolidation"]["status"] == "partial"
    assert y_doc["consolidation"]["is_complete"] is False

    # Reddit record was NOT corrupted by YouTube's invalid entries and remains complete
    assert r_doc["sentiment"]["status"] == "success"
    assert r_doc["sentiment"]["score"] == 0.95
    assert r_doc["consolidation"]["status"] == "complete"
    assert r_doc["consolidation"]["is_complete"] is True

    # Dataset status is partial because of YouTube's corrupted dimension and warnings
    assert res["consolidation_status"] == "partial"


def test_topics_and_misinformation_platform_isolation_with_invalid_native_ids(sample_fixtures, tmp_path):
    """Test platform isolation for topics and misinformation when one platform has invalid native IDs."""
    # Topics: Reddit has blank post_id, YouTube has valid video_id
    t_payload = {
        "status": "success",
        "reddit_records": [{"post_id": "   ", "topic_id": 1, "topic_status": "assigned"}],
        "youtube_records": [{"video_id": "y_2001", "topic_id": 2, "topic_status": "assigned"}],
    }
    t_path = tmp_path / "t_bad_reddit.json"
    t_path.write_text(json.dumps(t_payload), encoding="utf-8")

    # Misinformation: YouTube has integer video_id, Reddit has valid post_id
    m_payload = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "veracity_label": "real",
                "credibility_score": 0.9,
                "risk_tier": "low",
                "misinformation_status": "success",
            }
        ],
        "youtube_records": [
            {
                "video_id": 9999,
                "veracity_label": "fake",
                "credibility_score": 0.1,
                "risk_tier": "critical",
                "misinformation_status": "success",
            }
        ],
    }
    m_path = tmp_path / "m_bad_yt.json"
    m_path.write_text(json.dumps(m_payload), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=t_path,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=m_path,
        output_path=tmp_path / "out_tm_iso.json",
    )

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    y_doc = next(d for d in res["records"] if d["platform"] == "youtube")

    # For topics: Reddit is corrupted, YouTube is assigned
    assert r_doc["topics"]["status"] == "corrupted"
    assert y_doc["topics"]["status"] == "assigned"
    assert y_doc["topics"]["topic_id"] == 2

    # For misinformation: YouTube is corrupted, Reddit is success
    assert y_doc["misinformation"]["status"] == "corrupted"
    assert r_doc["misinformation"]["status"] == "success"
    assert r_doc["misinformation"]["credibility_score"] == 0.9


def test_missing_analytical_collection_keys_marks_only_affected_platform_corrupted(sample_fixtures, tmp_path):
    """Test that missing collection keys mark only the affected platform corrupted and log explicit audit warnings."""
    # Sentiment file is missing youtube_records, but contains valid reddit_records
    s_payload = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "sentiment_label": "positive",
                "sentiment_score": 0.95,
                "sentiment_status": "success",
            }
        ],
    }
    s_path = tmp_path / "s_missing_yt.json"
    s_path.write_text(json.dumps(s_payload), encoding="utf-8")

    # Topics file is missing reddit_records, but contains valid youtube_records
    t_payload = {
        "status": "success",
        "youtube_records": [
            {
                "video_id": "y_2001",
                "topic_id": 2,
                "topic_probability": 0.85,
                "topic_status": "assigned",
            }
        ],
    }
    t_path = tmp_path / "t_missing_reddit.json"
    t_path.write_text(json.dumps(t_payload), encoding="utf-8")

    # Misinformation file is missing youtube_records, but contains valid reddit_records
    m_payload = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "veracity_label": "real",
                "credibility_score": 0.92,
                "risk_tier": "low",
                "misinformation_status": "success",
            }
        ],
    }
    m_path = tmp_path / "m_missing_yt.json"
    m_path.write_text(json.dumps(m_payload), encoding="utf-8")

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=s_path,
        topics_path=t_path,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=m_path,
        output_path=tmp_path / "out_missing_keys.json",
    )

    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    y_doc = next(d for d in res["records"] if d["platform"] == "youtube")

    # Sentiment: Reddit is valid success, YouTube is corrupted
    assert r_doc["sentiment"]["status"] == "success"
    assert r_doc["sentiment"]["score"] == 0.95
    assert y_doc["sentiment"]["status"] == "corrupted"

    # Topics: YouTube is valid assigned, Reddit is corrupted
    assert y_doc["topics"]["status"] == "assigned"
    assert y_doc["topics"]["topic_id"] == 2
    assert r_doc["topics"]["status"] == "corrupted"

    # Misinformation: Reddit is valid success, YouTube is corrupted
    assert r_doc["misinformation"]["status"] == "success"
    assert r_doc["misinformation"]["credibility_score"] == 0.92
    assert y_doc["misinformation"]["status"] == "corrupted"

    # Verify audit warnings identify module, platform, collection key, and reason
    warnings = [w for w in res["audit_warnings"] if w.get("status") == "missing_collection_key"]
    assert len(warnings) == 3

    s_warn = next(w for w in warnings if w.get("module") == "sentiment")
    assert s_warn["platform"] == "youtube"
    assert s_warn["field"] == "youtube_records"
    assert s_warn["reason"] == "missing_collection_key"

    t_warn = next(w for w in warnings if w.get("module") == "topics")
    assert t_warn["platform"] == "reddit"
    assert t_warn["field"] == "reddit_records"
    assert t_warn["reason"] == "missing_collection_key"

    m_warn = next(w for w in warnings if w.get("module") == "misinformation")
    assert m_warn["platform"] == "youtube"
    assert m_warn["field"] == "youtube_records"
    assert m_warn["reason"] == "missing_collection_key"


def test_graph_platform_isolation_and_unidentified_entries(sample_fixtures, tmp_path):
    """Test platform isolation for graph entries with identifiable vs unidentifiable platform."""
    # Case 1: Graph has an invalid entry for reddit (missing username), but valid entry for youtube
    g_payload_1 = {
        "status": "success",
        "top_influencers": [
            {"username": "   ", "platform": "reddit"},  # invalid reddit entry
            {
                "username": "TechTalks",
                "platform": "youtube",
                "influencer_score": 0.88,
                "pagerank": 0.05,
                "community_id": 1,
                "in_degree_raw": 10,
                "out_degree_raw": 3,
            },
        ],
    }
    g_path_1 = tmp_path / "g_reddit_bad.json"
    g_path_1.write_text(json.dumps(g_payload_1), encoding="utf-8")

    res_1 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=g_path_1,
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_g1.json",
    )

    r_doc_1 = next(d for d in res_1["records"] if d["platform"] == "reddit")
    y_doc_1 = next(d for d in res_1["records"] if d["platform"] == "youtube")

    # Reddit graph status is corrupted
    assert r_doc_1["author"]["graph_status"] == "corrupted"
    assert r_doc_1["author"]["influencer_score"] is None
    # YouTube graph status is preserved as ranked_influencer with valid metrics
    assert y_doc_1["author"]["graph_status"] == "ranked_influencer"
    assert y_doc_1["author"]["influencer_score"] == 0.88
    assert y_doc_1["author"]["pagerank"] == 0.05

    # Case 2: Graph has an invalid entry with no identifiable platform (missing platform key)
    g_payload_2 = {
        "status": "success",
        "top_influencers": [
            {"username": "some_user"},  # no platform key
            {
                "username": "TechTalks",
                "platform": "youtube",
                "influencer_score": 0.88,
            },
        ],
    }
    g_path_2 = tmp_path / "g_unidentified_bad.json"
    g_path_2.write_text(json.dumps(g_payload_2), encoding="utf-8")

    res_2 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=g_path_2,
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_g2.json",
    )

    r_doc_2 = next(d for d in res_2["records"] if d["platform"] == "reddit")
    y_doc_2 = next(d for d in res_2["records"] if d["platform"] == "youtube")

    # Globally uncertain: both platforms are marked corrupted
    assert r_doc_2["author"]["graph_status"] == "corrupted"
    assert y_doc_2["author"]["graph_status"] == "corrupted"


def test_topic_fallback_and_empty_text_id_validation(sample_fixtures, tmp_path):
    """Test that fallback and empty_text reject IDs -2, True, 1.5, '1', keeping topic_id null while preserving status."""
    invalid_ids = [-2, True, 1.5, "1"]

    for status in ["fallback", "empty_text"]:
        for bad_id in invalid_ids:
            t_payload = {
                "status": "success",
                "reddit_records": [
                    {
                        "post_id": "r_1001",
                        "topic_id": bad_id,
                        "topic_status": status,
                    }
                ],
                "youtube_records": [],
            }
            t_path = tmp_path / f"t_{status}_{type(bad_id).__name__}_{bad_id}.json"
            t_path.write_text(json.dumps(t_payload), encoding="utf-8")

            res = consolidate_social_intelligence(
                reddit_processed_path=sample_fixtures["reddit_path"],
                youtube_processed_path=tmp_path / "empty_y.json",
                sentiment_path=sample_fixtures["sentiment_path"],
                topics_path=t_path,
                graph_path=sample_fixtures["graph_path"],
                misinformation_path=sample_fixtures["misinformation_path"],
                output_path=tmp_path / f"out_{status}_{type(bad_id).__name__}.json",
            )
            doc = res["records"][0]
            # Status is preserved
            assert doc["topics"]["status"] == status
            # Invalid ID rejected: topic_id is strictly null
            assert doc["topics"]["topic_id"] is None
            # Status is degraded (fallback/empty_text outcome)
            assert doc["consolidation"]["status"] == "degraded"

    # Verify that topic_id, probability, and keywords are strictly null/empty for fallback and empty_text
    # even when raw upstream record contains positive integers, probabilities, or keywords
    for raw_id in [0, 5]:
        t_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "topic_id": raw_id,
                    "topic_probability": 0.85,
                    "topic_keywords": ["topic", "sample"],
                    "topic_status": "fallback",
                }
            ],
            "youtube_records": [],
        }
        t_path = tmp_path / f"t_fallback_valid_{raw_id}.json"
        t_path.write_text(json.dumps(t_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=t_path,
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_fallback_valid_{raw_id}.json",
        )
        doc = res["records"][0]
        assert doc["topics"]["status"] == "fallback"
        assert doc["topics"]["topic_id"] is None
        assert doc["topics"]["probability"] is None
        assert doc["topics"]["keywords"] == []

    # Confirm that ID -1 is rejected (mapped to None) for both fallback and empty_text
    for status in ["fallback", "empty_text"]:
        t_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "topic_id": -1,
                    "topic_status": status,
                }
            ],
            "youtube_records": [],
        }
        t_path = tmp_path / f"t_{status}_neg_one.json"
        t_path.write_text(json.dumps(t_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=t_path,
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_{status}_neg_one.json",
        )
        doc = res["records"][0]
        assert doc["topics"]["status"] == status
        assert doc["topics"]["topic_id"] is None
        assert doc["consolidation"]["status"] == "degraded"


def test_youtube_author_fallback_and_graph_matching(sample_fixtures, tmp_path):
    """Test YouTube author fallback hierarchy, graph matching on fallback ID, and metadata preservation."""
    cases = [
        ("TechTalks", "UC123", "TechTalks", "case_normal"),
        ("   ", "UC123", "UC123", "case_whitespace"),
        (None, "UC123", "UC123", "case_none"),
        (123, "UC123", "UC123", "case_nonstring"),
        ("TechTalks", None, "TechTalks", "case_no_id"),
        (None, None, "", "case_both_none"),
    ]

    yt_records = []
    for title, ch_id, exp_user, cid in cases:
        yt_records.append({
            "video_id": f"y_{cid}",
            "title": f"Title {cid}",
            "description": "Video description for testing author fallback.",
            "channel_id": ch_id,
            "channel_title": title,
            "published_at": "2026-11-01T10:00:00Z",
            "view_count": 1000,
            "like_count": 100,
            "comment_count": 10,
            "url": f"https://youtube.com/watch?v=y_{cid}",
            "collected_at": "2026-11-05T12:00:00Z",
            "title_cleaned": f"Title {cid}",
            "description_cleaned": "Video description for testing author fallback.",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-05T12:05:00Z",
        })

    yt_file = tmp_path / "yt_author_cases.json"
    yt_file.write_text(json.dumps(yt_records), encoding="utf-8")

    # Graph results with an entry matching fallback channel ID 'UC123'
    graph_payload = {
        "status": "success",
        "top_influencers": [
            {
                "username": "UC123",
                "platform": "youtube",
                "influencer_score": 0.95,
                "pagerank": 0.08,
                "community_id": 2,
                "in_degree_raw": 15,
                "out_degree_raw": 5,
            },
            {
                "username": "TechTalks",
                "platform": "youtube",
                "influencer_score": 0.85,
                "pagerank": 0.04,
                "community_id": 1,
                "in_degree_raw": 8,
                "out_degree_raw": 2,
            },
        ],
    }
    g_file = tmp_path / "g_author_fallback.json"
    g_file.write_text(json.dumps(graph_payload), encoding="utf-8")

    out_file = tmp_path / "out_author_cases.json"
    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=yt_file,
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=g_file,
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=out_file,
    )

    doc_map = {d["native_id"]: d for d in res["records"] if d["platform"] == "youtube"}

    # Verify each case's author.username and metadata preservation
    for title, ch_id, exp_user, cid in cases:
        doc = doc_map[f"y_{cid}"]
        # 1. author.username matches expected fallback
        assert doc["author"]["username"] == exp_user
        # 2. Original channel metadata in metadata.youtube is preserved exactly as input
        assert doc["metadata"]["youtube"]["channel_title"] == title
        assert doc["metadata"]["youtube"]["channel_id"] == ch_id

    # Verify graph matching on fallback channel ID 'UC123'
    # Case whitespace ('   ', 'UC123') fell back to 'UC123' and matches graph influencer
    doc_ws = doc_map["y_case_whitespace"]
    assert doc_ws["author"]["username"] == "UC123"
    assert doc_ws["author"]["graph_status"] == "ranked_influencer"
    assert doc_ws["author"]["influencer_score"] == 0.95
    assert doc_ws["author"]["pagerank"] == 0.08

    # Case nonstring (123, 'UC123') fell back to 'UC123' and matches graph influencer
    doc_nonstr = doc_map["y_case_nonstring"]
    assert doc_nonstr["author"]["username"] == "UC123"
    assert doc_nonstr["author"]["graph_status"] == "ranked_influencer"
    assert doc_nonstr["author"]["influencer_score"] == 0.95

    # Case normal ('TechTalks', 'UC123') used 'TechTalks' and matches TechTalks influencer
    doc_norm = doc_map["y_case_normal"]
    assert doc_norm["author"]["username"] == "TechTalks"
    assert doc_norm["author"]["graph_status"] == "ranked_influencer"
    assert doc_norm["author"]["influencer_score"] == 0.85


# ==============================================================================
# Stage 10A Tests: Consolidated Artifact Validation, Provenance & Atomic Promotion
# ==============================================================================


def _make_valid_artifact_payload() -> dict:
    """Constructs a minimal, strictly valid Schema-v2.0 consolidated artifact payload."""
    return {
        "schema_version": "2.0",
        "pipeline_version": "1.0.0",
        "status": "success",
        "consolidation_status": "complete",
        "generated_at": "2026-11-06T22:00:00+00:00",
        "provenance": {
            "reddit_input_file": "data/processed/reddit_posts_processed.json",
            "reddit_status": "loaded",
            "reddit_records_count": 1,
            "youtube_input_file": "data/processed/youtube_videos_processed.json",
            "youtube_status": "loaded",
            "youtube_records_count": 1,
        },
        "metrics": {
            "total_records_consolidated": 2,
            "platform_counts": {"reddit": 1, "youtube": 1},
            "complete_records": 2,
            "degraded_records": 0,
            "partial_records": 0,
            "unprocessed_records": 0,
            "duplicates_deduplicated": 0,
            "rejected_records_count": 0,
        },
        "audit_warnings": [],
        "rejected_records": [],
        "records": [
            {
                "document_id": "reddit_r_1001",
                "platform": "reddit",
                "native_id": "r_1001",
                "url": "https://reddit.com/r/datascience/comments/r_1001",
                "timestamps": {
                    "published_at": "2024-10-28T07:40:00+00:00",
                    "collected_at": "2026-11-05T12:00:00+00:00",
                    "processed_at": "2026-11-05T12:05:00+00:00",
                },
                "author": {
                    "username": "alice_dev",
                    "channel_id": None,
                    "channel_title": None,
                    "graph_status": "ranked_influencer",
                    "influencer_score": 0.85,
                    "pagerank": 0.045,
                    "community_id": 0,
                    "in_degree": 5,
                    "out_degree": 2,
                },
                "content": {
                    "title_raw": "Discussion on LLMs",
                    "body_raw": "Detailed post body.",
                    "title_cleaned": "Discussion on LLMs",
                    "body_cleaned": "Detailed post body.",
                    "detected_language": "English",
                    "language_code": "en",
                    "preprocessing_status": "success",
                },
                "engagement": {
                    "score": 150,
                    "views_count": None,
                    "likes_count": None,
                    "comments_count": 25,
                },
                "metadata": {
                    "reddit": {"subreddit": "datascience", "score": 150},
                    "youtube": None,
                },
                "sentiment": {
                    "label": "positive",
                    "score": 0.95,
                    "status": "success",
                },
                "topics": {
                    "topic_id": 1,
                    "probability": 0.88,
                    "keywords": [{"word": "machine", "weight": None}],
                    "status": "assigned",
                },
                "misinformation": {
                    "veracity_label": "real",
                    "credibility_score": 0.90,
                    "risk_tier": "low",
                    "status": "success",
                    "stylometric_signals": {
                        "uppercase_char_ratio": 0.05,
                        "exclamation_intensity": 0.0,
                        "question_intensity": 0.0,
                        "word_count": 8,
                        "avg_word_length": 5.2,
                        "lexical_subjectivity_ratio": 0.1,
                        "digits_ratio": 0.0,
                        "all_caps_word_ratio": 0.0,
                    },
                },
                "consolidation": {
                    "status": "complete",
                    "is_complete": True,
                    "sources_joined": ["preprocessing", "sentiment", "topics", "graph", "misinformation"],
                },
            },
            {
                "document_id": "youtube_y_2001",
                "platform": "youtube",
                "native_id": "y_2001",
                "url": "https://youtube.com/watch?v=y_2001",
                "timestamps": {
                    "published_at": "2026-11-01T10:00:00+00:00",
                    "collected_at": "2026-11-05T12:00:00+00:00",
                    "processed_at": "2026-11-05T12:05:00+00:00",
                },
                "author": {
                    "username": "TechTalks",
                    "channel_id": "UC_channel_123",
                    "channel_title": "TechTalks",
                    "graph_status": "not_in_top_influencers",
                    "influencer_score": None,
                    "pagerank": None,
                    "community_id": None,
                    "in_degree": None,
                    "out_degree": None,
                },
                "content": {
                    "title_raw": "Intro to Data Engineering",
                    "body_raw": "Video description.",
                    "title_cleaned": "Intro to Data Engineering",
                    "body_cleaned": "Video description.",
                    "detected_language": "English",
                    "language_code": "en",
                    "preprocessing_status": "success",
                },
                "engagement": {
                    "score": None,
                    "views_count": 10000,
                    "likes_count": 500,
                    "comments_count": 40,
                },
                "metadata": {
                    "reddit": None,
                    "youtube": {"channel_id": "UC_channel_123", "channel_title": "TechTalks"},
                },
                "sentiment": {
                    "label": "neutral",
                    "score": 0.80,
                    "status": "success",
                },
                "topics": {
                    "topic_id": 2,
                    "probability": 0.75,
                    "keywords": [{"word": "databases", "weight": None}],
                    "status": "assigned",
                },
                "misinformation": {
                    "veracity_label": "real",
                    "credibility_score": 0.85,
                    "risk_tier": "low",
                    "status": "success",
                    "stylometric_signals": {
                        "uppercase_char_ratio": 0.04,
                        "exclamation_intensity": 0.0,
                        "question_intensity": 0.0,
                        "word_count": 10,
                        "avg_word_length": 4.8,
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
            },
        ],
    }


def test_schema_version_2_strict_enforcement():
    """Test schema_version enforcement: '2.0' passes; legacy, future, and malformed fail."""
    valid_payload = _make_valid_artifact_payload()
    ConsolidatedArtifactValidator.validate_consolidated_artifact(valid_payload)

    # 1. Missing schema_version
    bad = dict(valid_payload)
    del bad["schema_version"]
    with pytest.raises(ValueError, match="missing required envelope keys"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 2. Legacy '1.0'
    bad = dict(valid_payload, schema_version="1.0")
    with pytest.raises(ValueError, match="Unsupported schema_version '1.0'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 3. Future '3.0'
    bad = dict(valid_payload, schema_version="3.0")
    with pytest.raises(ValueError, match="Unsupported schema_version '3.0'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 4. Non-string schema_version
    for non_str in [2, 2.0, True, None]:
        bad = dict(valid_payload, schema_version=non_str)
        with pytest.raises(ValueError):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 5. Malformed string formats ('v2.0', '2.0.0', '')
    for malformed in ["v2.0", "2.0.0", ""]:
        bad = dict(valid_payload, schema_version=malformed)
        with pytest.raises(ValueError):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)


def test_pipeline_version_backward_compatibility():
    """Test pipeline_version is present and validated as non-empty string."""
    valid_payload = _make_valid_artifact_payload()
    assert valid_payload["pipeline_version"] == "1.0.0"
    ConsolidatedArtifactValidator.validate_consolidated_artifact(valid_payload)

    # Missing pipeline_version
    bad = dict(valid_payload)
    del bad["pipeline_version"]
    with pytest.raises(ValueError, match="missing required envelope keys"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # Empty or non-string pipeline_version
    for invalid in ["", "   ", 123, None]:
        bad = dict(valid_payload, pipeline_version=invalid)
        with pytest.raises(ValueError):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)


def test_envelope_structure_and_type_validation():
    """Test top-level envelope validation: non-dict payload, invalid status, timestamp, collections."""
    valid_payload = _make_valid_artifact_payload()

    # 1. Non-dict payload
    with pytest.raises(TypeError, match="Expected dictionary"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(["not_a_dict"])

    # 2. Status != 'success'
    bad = dict(valid_payload, status="failed")
    with pytest.raises(ValueError, match="Invalid artifact status"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 3. Invalid consolidation_status
    for bad_status in ["unknown", "in_progress", "", None]:
        bad = dict(valid_payload, consolidation_status=bad_status)
        with pytest.raises(ValueError, match="Invalid 'consolidation_status'"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 4. Invalid generated_at
    for bad_ts in ["not_a_timestamp", "2026-99-99", "", None, 12345]:
        bad = dict(valid_payload, generated_at=bad_ts)
        with pytest.raises(ValueError):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 5. Non-list audit_warnings, rejected_records, records
    for col_key in ["audit_warnings", "rejected_records", "records"]:
        bad = dict(valid_payload, **{col_key: "not_a_list"})
        with pytest.raises(ValueError, match=f"'{col_key}' must be a list"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)


def test_provenance_validation_and_invariants():
    """Test provenance validation: missing keys, invalid statuses, non-negative ints, zero invariants."""
    valid_payload = _make_valid_artifact_payload()

    # 1. Missing provenance block or non-dict
    bad = dict(valid_payload, provenance="not_a_dict")
    with pytest.raises(ValueError, match="'provenance' must be a dictionary"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 2. Missing provenance key
    bad_prov = dict(valid_payload["provenance"])
    del bad_prov["reddit_status"]
    bad = dict(valid_payload, provenance=bad_prov)
    with pytest.raises(ValueError, match="'provenance' missing required keys"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 3. Invalid status string
    bad_prov = dict(valid_payload["provenance"], reddit_status="invalid_status")
    bad = dict(valid_payload, provenance=bad_prov)
    with pytest.raises(ValueError, match="invalid; must be one of"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 4. Non-strict int count (boolean True/False)
    bad_prov = dict(valid_payload["provenance"], reddit_records_count=True)
    bad = dict(valid_payload, provenance=bad_prov)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 5. Negative count
    bad_prov = dict(valid_payload["provenance"], reddit_records_count=-1)
    bad = dict(valid_payload, provenance=bad_prov)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 6. Non-zero count when status is missing/empty/corrupted
    for st in ["missing", "empty", "corrupted"]:
        bad_prov = dict(valid_payload["provenance"], reddit_status=st, reddit_records_count=5)
        bad = dict(valid_payload, provenance=bad_prov)
        with pytest.raises(ValueError, match=f"must be 0 when status is '{st}'"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)


def test_metrics_types_and_sum_invariants():
    """Test metric validation: strict integer types, non-negativity, and mathematical sum invariants."""
    valid_payload = _make_valid_artifact_payload()

    # 1. Boolean metric values (rejecting bool as int)
    for field in ["complete_records", "total_records_consolidated", "duplicates_deduplicated"]:
        bad_metrics = dict(valid_payload["metrics"], **{field: True})
        bad = dict(valid_payload, metrics=bad_metrics)
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 2. Negative metric values
    bad_metrics = dict(valid_payload["metrics"], complete_records=-1)
    bad = dict(valid_payload, metrics=bad_metrics)
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 3. Total records mismatch vs len(records)
    bad_metrics = dict(valid_payload["metrics"], total_records_consolidated=99)
    bad = dict(valid_payload, metrics=bad_metrics)
    with pytest.raises(ValueError, match="does not match len"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 4. Rejected records mismatch vs len(rejected_records)
    bad_metrics = dict(valid_payload["metrics"], rejected_records_count=5)
    bad = dict(valid_payload, metrics=bad_metrics)
    with pytest.raises(ValueError, match="rejected_records_count"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 5. Platform counts sum mismatch vs total
    bad_metrics = dict(valid_payload["metrics"], platform_counts={"reddit": 1, "youtube": 0})
    bad = dict(valid_payload, metrics=bad_metrics)
    with pytest.raises(ValueError):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 6. Status counts sum mismatch vs total
    bad_metrics = dict(valid_payload["metrics"], complete_records=1)  # 1 + 0 + 0 + 0 = 1 != 2
    bad = dict(valid_payload, metrics=bad_metrics)
    with pytest.raises(ValueError, match="Sum of record status counts"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)


def test_records_canonical_fields_and_envelope_leakage():
    """Test record-level validation: exact 13 canonical fields, rejection of envelope field leakage."""
    valid_payload = _make_valid_artifact_payload()

    # 1. Missing canonical root field
    for field in ["consolidation", "misinformation", "author", "content", "sentiment"]:
        bad_rec = dict(valid_payload["records"][0])
        del bad_rec[field]
        bad = dict(valid_payload, records=[bad_rec, valid_payload["records"][1]])
        with pytest.raises(ValueError, match="missing canonical root fields"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 2. Envelope field leakage into record
    for leaked_field in ["schema_version", "pipeline_version", "status", "metrics", "provenance", "audit_warnings"]:
        bad_rec = dict(valid_payload["records"][0], **{leaked_field: "leaked_value"})
        bad = dict(valid_payload, records=[bad_rec, valid_payload["records"][1]])
        with pytest.raises(ValueError, match="contains forbidden extra fields"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 3. Duplicate document_id
    dup_rec = dict(valid_payload["records"][0])
    bad = dict(
        valid_payload,
        records=[dup_rec, dup_rec],
        metrics=dict(valid_payload["metrics"], platform_counts={"reddit": 2, "youtube": 0}),
    )
    with pytest.raises(ValueError, match="Duplicate document_id"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 4. Mismatched document_id vs platform_native_id
    bad_rec = dict(valid_payload["records"][0], document_id="reddit_wrong_id")
    bad = dict(valid_payload, records=[bad_rec, valid_payload["records"][1]])
    with pytest.raises(ValueError, match="does not match expected"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 5. Invalid platform
    bad_rec = dict(valid_payload["records"][0], platform="twitter", document_id="twitter_r_1001")
    bad = dict(valid_payload, records=[bad_rec, valid_payload["records"][1]])
    with pytest.raises(ValueError, match="invalid platform 'twitter'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 6. Non-dict sub-field
    for sub in ["timestamps", "author", "content", "engagement", "metadata", "sentiment", "topics", "misinformation", "consolidation"]:
        bad_rec = dict(valid_payload["records"][0], **{sub: "not_a_dict"})
        bad = dict(valid_payload, records=[bad_rec, valid_payload["records"][1]])
        with pytest.raises(ValueError, match=f"field '{sub}' must be a dictionary"):
            ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)


def test_topic_keyword_dict_with_additional_fields():
    """Test that keyword dictionaries containing additional fields preserve word and weight without silent loss."""
    raw = [
        {"word": "machine learning", "weight": 0.85, "source": "bertopic", "freq": 10},
        {"word": "neural nets", "weight": None, "cluster_id": 4, "is_stopword": False},
        {"word": "transformers", "weight": 0.95, "nested_meta": {"score": 1}},
    ]
    norm = normalize_consolidator_topic_keywords(raw)

    # Valid words and weights are strictly preserved
    assert norm == [
        {"word": "machine learning", "weight": 0.85},
        {"word": "neural nets", "weight": None},
        {"word": "transformers", "weight": 0.95},
    ]
    # Extra fields are stripped so the result satisfies the Elasticsearch strict dynamic mapping contract
    for entry in norm:
        assert set(entry.keys()) == {"word", "weight"}


def test_topic_keyword_valid_word_missing_weight():
    """Test that valid words with missing or None weight preserve the word and set weight strictly to None."""
    raw = [
        {"word": "deep learning"},
        "reinforcement learning",
        {"word": "nlp", "weight": None},
        {"word": "computer vision", "source": "model"},  # Missing weight with extra field
    ]
    norm = normalize_consolidator_topic_keywords(raw)

    assert norm == [
        {"word": "deep learning", "weight": None},
        {"word": "reinforcement learning", "weight": None},
        {"word": "nlp", "weight": None},
        {"word": "computer vision", "weight": None},
    ]


def test_topic_keyword_valid_word_invalid_or_non_finite_weight():
    """Test that valid words with invalid or non-finite weights preserve the word and normalize weight to None."""
    raw = [
        {"word": "ai", "weight": True},
        {"word": "dl", "weight": "heavy"},
        {"word": "rl", "weight": float("inf")},
        {"word": "cv", "weight": float("nan")},
        {"word": "nlp", "weight": ["list_weight"]},
        {"word": "llm", "weight": {"dict": 1}},
    ]
    norm = normalize_consolidator_topic_keywords(raw)

    assert norm == [
        {"word": "ai", "weight": None},
        {"word": "dl", "weight": None},
        {"word": "rl", "weight": None},
        {"word": "cv", "weight": None},
        {"word": "nlp", "weight": None},
        {"word": "llm", "weight": None},
    ]


def test_topic_keyword_empty_or_malformed_entries():
    """Test that uninterpretable entries are explicitly rejected and recorded without crashing or silent loss."""
    raw = [
        "",
        "   ",
        None,
        123,
        True,
        [],
        {"weight": 0.5},  # Missing 'word'
        {"word": ""},  # Empty string 'word'
        {"word": "   ", "weight": 0.9},  # Whitespace-only 'word'
        {"word": None, "weight": 0.5},  # Non-string 'word'
    ]
    rejected: list = []
    norm = normalize_consolidator_topic_keywords(raw, rejected_collector=rejected)

    # Output list contains no uninterpretable entries
    assert norm == []

    # All 10 entries are explicitly recorded with reasons to prevent silent data loss
    assert len(rejected) == 10
    reasons = [r["reason"] for r in rejected]
    assert "empty_or_whitespace_string" in reasons
    assert "missing_or_blank_word_field" in reasons
    assert any("unsupported_element_type" in r for r in reasons)


def test_topic_keyword_preservation_of_all_valid_words():
    """Test that all valid words in a mixed collection are preserved without data loss."""
    raw = [
        "transformers",
        {"word": "attention", "weight": 0.92, "source": "extra"},
        "",  # malformed
        {"word": "bert"},
        None,  # malformed
        {"word": "gpt", "weight": "top"},  # invalid weight
        {"weight": 0.5},  # malformed
        12345,  # malformed
        {"word": "claude", "weight": 0.98},
    ]
    rejected: list = []
    norm = normalize_consolidator_topic_keywords(raw, rejected_collector=rejected)

    # All 5 valid words are preserved
    preserved_words = [item["word"] for item in norm]
    assert preserved_words == ["transformers", "attention", "bert", "gpt", "claude"]

    # Valid weights preserved; invalid/missing weights normalized to None
    assert norm[0]["weight"] is None
    assert norm[1]["weight"] == 0.92
    assert norm[2]["weight"] is None
    assert norm[3]["weight"] is None
    assert norm[4]["weight"] == 0.98

    # The 4 malformed entries are documented in rejected collector
    assert len(rejected) == 4


def test_end_to_end_provenance_and_schema_version(sample_fixtures):
    """Test full pipeline execution produces Schema-v2.0, pipeline-1.0.0, and exact provenance."""
    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=sample_fixtures["output_path"],
    )

    # 1. Schema and pipeline versions
    assert res["schema_version"] == SUPPORTED_SCHEMA_VERSION
    assert res["schema_version"] == "2.0"
    assert res["pipeline_version"] == PIPELINE_VERSION
    assert res["pipeline_version"] == "1.0.0"

    # 2. Provenance block
    prov = res["provenance"]
    assert prov["reddit_input_file"] == str(sample_fixtures["reddit_path"])
    assert prov["reddit_status"] == "loaded"
    assert prov["reddit_records_count"] == 1
    assert prov["youtube_input_file"] == str(sample_fixtures["youtube_path"])
    assert prov["youtube_status"] == "loaded"
    assert prov["youtube_records_count"] == 1

    # 3. Topic keywords normalized to canonical nested objects
    r_doc = next(d for d in res["records"] if d["platform"] == "reddit")
    assert r_doc["topics"]["keywords"] == [
        {"word": "machine", "weight": None},
        {"word": "learning", "weight": None},
        {"word": "llms", "weight": None},
    ]

    # 4. Output file passes ConsolidatedArtifactValidator directly
    with open(sample_fixtures["output_path"], "r", encoding="utf-8") as f:
        read_file = json.load(f)
    ConsolidatedArtifactValidator.validate_consolidated_artifact(read_file)


def test_end_to_end_topic_keyword_additional_fields_and_malformed_warning(sample_fixtures, tmp_path):
    """Test end-to-end pipeline preserves keywords with additional fields and documents malformed entries in audit warnings."""
    # Topic results where Reddit has keywords with extra metadata and an empty keyword string
    t_payload = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "topic_id": 1,
                "topic_probability": 0.88,
                "topic_keywords": [
                    {"word": "machine learning", "weight": 0.9, "source": "tfidf", "rank": 1},
                    "",  # malformed empty keyword string
                    {"word": "nlp", "weight": None, "cluster": "A"},
                ],
                "topic_status": "assigned",
            }
        ],
        "youtube_records": [],
    }
    t_path = tmp_path / "t_extra_fields.json"
    t_path.write_text(json.dumps(t_payload), encoding="utf-8")
    out_file = tmp_path / "out_keywords_extra.json"

    res = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_yt.json",
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=t_path,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=out_file,
    )

    r_doc = res["records"][0]
    # Both valid keywords are preserved despite additional metadata fields
    assert r_doc["topics"]["keywords"] == [
        {"word": "machine learning", "weight": 0.9},
        {"word": "nlp", "weight": None},
    ]

    # Malformed empty keyword was logged as an uninterpretable_keyword_entry audit warning
    kw_warnings = [
        w for w in res["audit_warnings"]
        if w.get("status") == "uninterpretable_keyword_entry"
    ]
    assert len(kw_warnings) == 1
    assert kw_warnings[0]["reason"] == "empty_or_whitespace_string"
    assert kw_warnings[0]["native_id"] == "r_1001"

    # Artifact is valid against schema
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res)


def test_end_to_end_partial_and_empty_provenance(sample_fixtures, tmp_path):
    """Test provenance tracking when sources are missing or empty."""
    missing_reddit = tmp_path / "nonexistent_reddit.json"
    empty_youtube = tmp_path / "empty_youtube.json"
    empty_youtube.write_text("[]", encoding="utf-8")
    out_file = tmp_path / "out_prov_partial.json"

    res = consolidate_social_intelligence(
        reddit_processed_path=missing_reddit,
        youtube_processed_path=empty_youtube,
        output_path=out_file,
    )

    prov = res["provenance"]
    assert prov["reddit_status"] == "missing"
    assert prov["reddit_records_count"] == 0
    assert prov["youtube_status"] == "empty"
    assert prov["youtube_records_count"] == 0

    assert res["consolidation_status"] == "unprocessed"
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res)


# ==============================================================================
# Atomic Promotion Failure Tests (Failure paths 1 to 5 + Success Verification)
# ==============================================================================


def test_atomic_promotion_failure_write_or_flush(tmp_path):
    """Failure Path 1: Destination preserved and temp file cleaned up when writing or flushing fails."""
    dest = tmp_path / "protected_artifact.json"
    original_content = {"valid": "original_state_must_be_preserved"}
    dest.write_text(json.dumps(original_content), encoding="utf-8")

    payload = _make_valid_artifact_payload()

    # Simulate write/flush failure during temporary file writing
    with patch("src.pipeline.consolidator.json.dump", side_effect=OSError("Simulated disk write/flush failure")):
        with pytest.raises(OSError, match="Simulated disk write/flush failure"):
            _promote_artifact_atomically(payload, dest)

    # 1. Previous destination survives 100% untouched
    with open(dest, "r", encoding="utf-8") as f:
        preserved = json.load(f)
    assert preserved == original_content

    # 2. Temporary file is cleaned up
    assert not any(f.name.startswith(".tmp_") for f in tmp_path.iterdir())


def test_atomic_promotion_failure_fsync(tmp_path):
    """Failure Path 2: Destination preserved and temp file cleaned up when os.fsync fails."""
    dest = tmp_path / "protected_artifact.json"
    original_content = {"valid": "original_state_must_be_preserved"}
    dest.write_text(json.dumps(original_content), encoding="utf-8")

    payload = _make_valid_artifact_payload()

    with patch("os.fsync", side_effect=OSError("Simulated fsync sync failure")):
        with pytest.raises(OSError, match="Simulated fsync sync failure"):
            _promote_artifact_atomically(payload, dest)

    # 1. Previous destination survives 100% untouched
    with open(dest, "r", encoding="utf-8") as f:
        preserved = json.load(f)
    assert preserved == original_content

    # 2. Temporary file is cleaned up
    assert not any(f.name.startswith(".tmp_") for f in tmp_path.iterdir())


def test_atomic_promotion_failure_readback_json_parse(tmp_path):
    """Failure Path 3: Destination preserved and temp file cleaned up when read-back JSON parsing fails."""
    dest = tmp_path / "protected_artifact.json"
    original_content = {"valid": "original_state_must_be_preserved"}
    dest.write_text(json.dumps(original_content), encoding="utf-8")

    payload = _make_valid_artifact_payload()

    with patch("json.load", side_effect=json.JSONDecodeError("Simulated corrupted JSON readback", "doc", 0)):
        with pytest.raises(json.JSONDecodeError, match="Simulated corrupted JSON readback"):
            _promote_artifact_atomically(payload, dest)

    # 1. Previous destination survives 100% untouched
    with open(dest, "r", encoding="utf-8") as f:
        preserved = json.load(f)
    assert preserved == original_content

    # 2. Temporary file is cleaned up
    assert not any(f.name.startswith(".tmp_") for f in tmp_path.iterdir())


def test_atomic_promotion_failure_readback_validation(tmp_path):
    """Failure Path 4: Destination preserved and temp file cleaned up when read-back validation fails."""
    dest = tmp_path / "protected_artifact.json"
    original_content = {"valid": "original_state_must_be_preserved"}
    dest.write_text(json.dumps(original_content), encoding="utf-8")

    # Invalid payload (schema_version '3.0' is invalid)
    bad_payload = dict(_make_valid_artifact_payload(), schema_version="3.0")

    with pytest.raises(ValueError, match="Unsupported schema_version '3.0'"):
        _promote_artifact_atomically(bad_payload, dest)

    # 1. Previous destination survives 100% untouched
    with open(dest, "r", encoding="utf-8") as f:
        preserved = json.load(f)
    assert preserved == original_content

    # 2. Temporary file is cleaned up
    assert not any(f.name.startswith(".tmp_") for f in tmp_path.iterdir())


def test_atomic_promotion_failure_replace(tmp_path):
    """Failure Path 5: Destination preserved and temp file cleaned up when os.replace fails."""
    dest = tmp_path / "protected_artifact.json"
    original_content = {"valid": "original_state_must_be_preserved"}
    dest.write_text(json.dumps(original_content), encoding="utf-8")

    valid_payload = _make_valid_artifact_payload()

    with patch("os.replace", side_effect=OSError("Simulated os.replace atomic rename failure")):
        with pytest.raises(OSError, match="Simulated os.replace atomic rename failure"):
            _promote_artifact_atomically(valid_payload, dest)

    # 1. Previous destination survives 100% untouched
    with open(dest, "r", encoding="utf-8") as f:
        preserved = json.load(f)
    assert preserved == original_content

    # 2. Temporary file is cleaned up
    assert not any(f.name.startswith(".tmp_") for f in tmp_path.iterdir())


def test_atomic_promotion_success_cleans_temporary_files(tmp_path):
    """Success Verification: Promotion creates destination file and leaves zero temporary files."""
    dest = tmp_path / "nested" / "dir" / "final_artifact.json"
    valid_payload = _make_valid_artifact_payload()

    _promote_artifact_atomically(valid_payload, dest)

    # 1. Final artifact created and valid
    assert dest.exists()
    assert dest.is_file()
    with open(dest, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "2.0"
    ConsolidatedArtifactValidator.validate_consolidated_artifact(data)

    # 2. Zero temporary files left in destination directory
    parent_files = list(dest.parent.iterdir())
    assert len(parent_files) == 1
    assert parent_files[0] == dest
    assert not any(f.name.startswith(".tmp_") for f in dest.parent.iterdir())


# ==============================================================================
# Stage 10B.1 — Analytical Nullability and Elasticsearch Alignment Tests
# ==============================================================================


def test_sentiment_nullability_across_all_non_success_statuses(sample_fixtures, tmp_path):
    """Stage 10B.1: When sentiment.status is not 'success', label and score must be strictly None.
    
    Verifies that 'fallback', 'empty_text', 'error', 'missing_status', 'unavailable', and 'corrupted'
    always result in label=None and score=None, pass ConsolidatedArtifactValidator, and pass
    Elasticsearch indexing_adapter.validate_and_normalize_record without rejection.
    """
    statuses = ["fallback", "empty_text", "error", "missing_status"]

    for idx, status in enumerate(statuses):
        s_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "sentiment_label": "positive",
                    "sentiment_score": 0.88,
                    "sentiment_status": status,
                }
            ],
            "youtube_records": [],
        }
        s_path = tmp_path / f"s_null_{idx}.json"
        s_path.write_text(json.dumps(s_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=s_path,
            topics_path=sample_fixtures["topics_path"],
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_s_null_{idx}.json",
        )
        doc = res["records"][0]
        assert doc["sentiment"]["status"] == status
        assert doc["sentiment"]["label"] is None
        assert doc["sentiment"]["score"] is None

        # Verify adapter validation accepts this record
        is_valid, norm_doc, rej = validate_and_normalize_record(doc)
        assert is_valid is True, f"Adapter rejected document: {rej}"
        assert norm_doc is not None
        assert norm_doc["sentiment"]["label"] is None
        assert norm_doc["sentiment"]["score"] is None


def test_topics_nullability_across_all_statuses(sample_fixtures, tmp_path):
    """Stage 10B.1: Enforce strict topic nullability and outlier contract.
    
    - 'assigned': non-negative int topic_id, probability in [0, 1] or None, canonical keywords.
    - 'outlier': topic_id == -1, probability in [0, 1] or None, keywords == [].
    - statuses not in {'assigned', 'outlier'}: topic_id=None, probability=None, keywords=[].
    """
    # 1. Outlier with probability preserved
    t_outlier_prob = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "topic_id": -1,
                "topic_probability": 0.42,
                "topic_keywords": ["ignored", "keys"],
                "topic_status": "outlier",
            }
        ],
        "youtube_records": [],
    }
    t_path_1 = tmp_path / "t_outlier_prob.json"
    t_path_1.write_text(json.dumps(t_outlier_prob), encoding="utf-8")

    res_1 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=t_path_1,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_outlier_prob.json",
    )
    doc_1 = res_1["records"][0]
    assert doc_1["topics"]["status"] == "outlier"
    assert doc_1["topics"]["topic_id"] == -1
    assert doc_1["topics"]["probability"] == 0.42
    assert doc_1["topics"]["keywords"] == []

    is_valid_1, _, rej_1 = validate_and_normalize_record(doc_1)
    assert is_valid_1 is True, f"Adapter rejected outlier with prob: {rej_1}"

    # 2. Outlier with None probability
    t_outlier_none = {
        "status": "success",
        "reddit_records": [
            {
                "post_id": "r_1001",
                "topic_id": -1,
                "topic_probability": None,
                "topic_status": "outlier",
            }
        ],
        "youtube_records": [],
    }
    t_path_2 = tmp_path / "t_outlier_none.json"
    t_path_2.write_text(json.dumps(t_outlier_none), encoding="utf-8")

    res_2 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=t_path_2,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_outlier_none.json",
    )
    doc_2 = res_2["records"][0]
    assert doc_2["topics"]["status"] == "outlier"
    assert doc_2["topics"]["topic_id"] == -1
    assert doc_2["topics"]["probability"] is None
    assert doc_2["topics"]["keywords"] == []

    is_valid_2, _, rej_2 = validate_and_normalize_record(doc_2)
    assert is_valid_2 is True, f"Adapter rejected outlier with None prob: {rej_2}"

    # 3. Non-assigned, non-outlier statuses with dummy non-null values
    non_assigned_statuses = ["fallback", "empty_text", "error", "missing_status"]
    for idx, status in enumerate(non_assigned_statuses):
        t_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "topic_id": 10,
                    "topic_probability": 0.85,
                    "topic_keywords": ["ai", "tech"],
                    "topic_status": status,
                }
            ],
            "youtube_records": [],
        }
        t_path = tmp_path / f"t_non_assigned_{idx}.json"
        t_path.write_text(json.dumps(t_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=t_path,
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=sample_fixtures["misinformation_path"],
            output_path=tmp_path / f"out_t_non_assigned_{idx}.json",
        )
        doc = res["records"][0]
        assert doc["topics"]["status"] == status
        assert doc["topics"]["topic_id"] is None
        assert doc["topics"]["probability"] is None
        assert doc["topics"]["keywords"] == []

        is_valid, _, rej = validate_and_normalize_record(doc)
        assert is_valid is True, f"Adapter rejected non-assigned topic record: {rej}"


def test_misinformation_nullability_across_all_non_success_statuses(sample_fixtures, tmp_path):
    """Stage 10B.1: When misinformation.status is not 'success', veracity, credibility,
    risk_tier, and stylometric_signals must be strictly None.
    
    Verifies that 'fallback', 'empty_text', 'error', 'missing_status', 'unavailable', and 'corrupted'
    always zero out all analytical fields, pass ConsolidatedArtifactValidator, and pass
    Elasticsearch indexing_adapter.validate_and_normalize_record.
    """
    statuses = ["fallback", "empty_text", "error", "missing_status"]

    for idx, status in enumerate(statuses):
        m_payload = {
            "status": "success",
            "reddit_records": [
                {
                    "post_id": "r_1001",
                    "veracity_label": "fake",
                    "credibility_score": 0.22,
                    "risk_tier": "high_risk",
                    "misinformation_status": status,
                    "stylometric_signals": {
                        "uppercase_char_ratio": 0.25,
                        "exclamation_intensity": 1.0,
                        "question_intensity": 0.5,
                        "word_count": 15,
                        "avg_word_length": 6.1,
                        "lexical_subjectivity_ratio": 0.7,
                        "digits_ratio": 0.1,
                        "all_caps_word_ratio": 0.2,
                    },
                }
            ],
            "youtube_records": [],
        }
        m_path = tmp_path / f"m_null_{idx}.json"
        m_path.write_text(json.dumps(m_payload), encoding="utf-8")

        res = consolidate_social_intelligence(
            reddit_processed_path=sample_fixtures["reddit_path"],
            youtube_processed_path=tmp_path / "empty_y.json",
            sentiment_path=sample_fixtures["sentiment_path"],
            topics_path=sample_fixtures["topics_path"],
            graph_path=sample_fixtures["graph_path"],
            misinformation_path=m_path,
            output_path=tmp_path / f"out_m_null_{idx}.json",
        )
        doc = res["records"][0]
        assert doc["misinformation"]["status"] == status
        assert doc["misinformation"]["veracity_label"] is None
        assert doc["misinformation"]["credibility_score"] is None
        assert doc["misinformation"]["risk_tier"] is None
        assert doc["misinformation"]["stylometric_signals"] is None

        is_valid, norm_doc, rej = validate_and_normalize_record(doc)
        assert is_valid is True, f"Adapter rejected document for misinfo status {status}: {rej}"
        assert norm_doc is not None
        assert norm_doc["misinformation"]["veracity_label"] is None
        assert norm_doc["misinformation"]["credibility_score"] is None
        assert norm_doc["misinformation"]["risk_tier"] is None
        assert norm_doc["misinformation"]["stylometric_signals"] is None


def test_cross_stage_adapter_acceptance_representative_scenarios(sample_fixtures, tmp_path):
    """Stage 10B.1: End-to-end integration verifying 5 representative cross-stage scenarios
    pass both ConsolidatedArtifactValidator and Elasticsearch indexing_adapter.validate_and_normalize_record.
    
    Scenarios:
    1. Complete: All 3 analytical stages successful (Reddit + YouTube).
    2. Degraded: Sentiment fallback, topics assigned, misinformation success.
    3. Partial: Topic missing_status, sentiment success, misinformation success.
    4. Partial: Misinformation error, sentiment success, topics assigned.
    5. Degraded/Partial: Multiple non-success statuses combined (sentiment empty_text, topic fallback, misinformation error).
    """
    canonical_signals = {
        "uppercase_char_ratio": 0.05,
        "exclamation_intensity": 0.0,
        "question_intensity": 0.0,
        "word_count": 8,
        "avg_word_length": 5.2,
        "lexical_subjectivity_ratio": 0.1,
        "digits_ratio": 0.0,
        "all_caps_word_ratio": 0.0,
    }

    # Scenario 1: All 3 analytical stages successful
    res_1 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_scenario_1.json",
    )
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res_1)
    for rec in res_1["records"]:
        is_valid, norm, rej = validate_and_normalize_record(rec)
        assert is_valid is True, f"Scenario 1 record rejected: {rej}"
        assert norm["consolidation"]["status"] == "complete"

    # Scenario 2: Sentiment fallback (degraded)
    s_deg = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "sentiment_status": "fallback"}],
        "youtube_records": [{"video_id": "y_2001", "sentiment_label": "neutral", "sentiment_score": 0.5, "sentiment_status": "success"}],
    }
    s_deg_path = tmp_path / "s_deg.json"
    s_deg_path.write_text(json.dumps(s_deg), encoding="utf-8")

    res_2 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=s_deg_path,
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_scenario_2.json",
    )
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res_2)
    for rec in res_2["records"]:
        is_valid, norm, rej = validate_and_normalize_record(rec)
        assert is_valid is True, f"Scenario 2 record rejected: {rej}"
    r_rec_2 = next(r for r in res_2["records"] if r["platform"] == "reddit")
    assert r_rec_2["sentiment"]["label"] is None
    assert r_rec_2["sentiment"]["score"] is None
    assert r_rec_2["consolidation"]["status"] == "degraded"

    # Scenario 3: Topic missing_status (partial)
    t_partial = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_id": 1, "topic_status": "   "}],
        "youtube_records": [{"video_id": "y_2001", "topic_id": 2, "topic_status": "assigned"}],
    }
    t_partial_path = tmp_path / "t_partial.json"
    t_partial_path.write_text(json.dumps(t_partial), encoding="utf-8")

    res_3 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=t_partial_path,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=sample_fixtures["misinformation_path"],
        output_path=tmp_path / "out_scenario_3.json",
    )
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res_3)
    for rec in res_3["records"]:
        is_valid, norm, rej = validate_and_normalize_record(rec)
        assert is_valid is True, f"Scenario 3 record rejected: {rej}"
    r_rec_3 = next(r for r in res_3["records"] if r["platform"] == "reddit")
    assert r_rec_3["topics"]["status"] == "missing_status"
    assert r_rec_3["topics"]["topic_id"] is None
    assert r_rec_3["topics"]["probability"] is None
    assert r_rec_3["topics"]["keywords"] == []
    assert r_rec_3["consolidation"]["status"] == "partial"

    # Scenario 4: Misinformation error (partial)
    m_err = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "misinformation_status": "error"}],
        "youtube_records": [{
            "video_id": "y_2001",
            "veracity_label": "real",
            "credibility_score": 0.85,
            "risk_tier": "low",
            "misinformation_status": "success",
            "stylometric_signals": canonical_signals,
        }],
    }
    m_err_path = tmp_path / "m_err.json"
    m_err_path.write_text(json.dumps(m_err), encoding="utf-8")

    res_4 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=sample_fixtures["youtube_path"],
        sentiment_path=sample_fixtures["sentiment_path"],
        topics_path=sample_fixtures["topics_path"],
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=m_err_path,
        output_path=tmp_path / "out_scenario_4.json",
    )
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res_4)
    for rec in res_4["records"]:
        is_valid, norm, rej = validate_and_normalize_record(rec)
        assert is_valid is True, f"Scenario 4 record rejected: {rej}"
    r_rec_4 = next(r for r in res_4["records"] if r["platform"] == "reddit")
    assert r_rec_4["misinformation"]["status"] == "error"
    assert r_rec_4["misinformation"]["veracity_label"] is None
    assert r_rec_4["misinformation"]["credibility_score"] is None
    assert r_rec_4["misinformation"]["risk_tier"] is None
    assert r_rec_4["misinformation"]["stylometric_signals"] is None
    assert r_rec_4["consolidation"]["status"] == "partial"

    # Scenario 5: Multiple non-success statuses combined
    s_multi = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "sentiment_status": "empty_text"}],
        "youtube_records": [],
    }
    t_multi = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "topic_status": "fallback", "topic_id": 99}],
        "youtube_records": [],
    }
    m_multi = {
        "status": "success",
        "reddit_records": [{"post_id": "r_1001", "misinformation_status": "error", "veracity_label": "fake"}],
        "youtube_records": [],
    }
    s_m_p = tmp_path / "s_multi.json"
    t_m_p = tmp_path / "t_multi.json"
    m_m_p = tmp_path / "m_multi.json"
    s_m_p.write_text(json.dumps(s_multi), encoding="utf-8")
    t_m_p.write_text(json.dumps(t_multi), encoding="utf-8")
    m_m_p.write_text(json.dumps(m_multi), encoding="utf-8")

    res_5 = consolidate_social_intelligence(
        reddit_processed_path=sample_fixtures["reddit_path"],
        youtube_processed_path=tmp_path / "empty_y.json",
        sentiment_path=s_m_p,
        topics_path=t_m_p,
        graph_path=sample_fixtures["graph_path"],
        misinformation_path=m_m_p,
        output_path=tmp_path / "out_scenario_5.json",
    )
    ConsolidatedArtifactValidator.validate_consolidated_artifact(res_5)
    r_rec_5 = res_5["records"][0]
    assert r_rec_5["sentiment"]["status"] == "empty_text"
    assert r_rec_5["sentiment"]["label"] is None
    assert r_rec_5["sentiment"]["score"] is None

    assert r_rec_5["topics"]["status"] == "fallback"
    assert r_rec_5["topics"]["topic_id"] is None
    assert r_rec_5["topics"]["probability"] is None
    assert r_rec_5["topics"]["keywords"] == []

    assert r_rec_5["misinformation"]["status"] == "error"
    assert r_rec_5["misinformation"]["veracity_label"] is None
    assert r_rec_5["misinformation"]["credibility_score"] is None
    assert r_rec_5["misinformation"]["risk_tier"] is None
    assert r_rec_5["misinformation"]["stylometric_signals"] is None

    assert r_rec_5["consolidation"]["status"] == "partial"

    is_valid_5, norm_5, rej_5 = validate_and_normalize_record(r_rec_5)
    assert is_valid_5 is True, f"Scenario 5 multi-non-success rejected by adapter: {rej_5}"


def test_validator_rejects_inconsistent_records():
    """Stage 10B.1: ConsolidatedArtifactValidator strictly rejects any document that
    violates the analytical status nullability and value consistency rules.
    """
    valid_payload = _make_valid_artifact_payload()

    # 1. Sentiment: non-success status carrying label
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["sentiment"]["status"] = "fallback"
    bad["records"][0]["sentiment"]["label"] = "positive"
    bad["records"][0]["sentiment"]["score"] = None
    with pytest.raises(ValueError, match="sentiment.label must be None when status is 'fallback'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 2. Sentiment: non-success status carrying score
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["sentiment"]["status"] = "error"
    bad["records"][0]["sentiment"]["label"] = None
    bad["records"][0]["sentiment"]["score"] = 0.5
    with pytest.raises(ValueError, match="sentiment.score must be None when status is 'error'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 3. Sentiment: success carrying boolean score
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["sentiment"]["status"] = "success"
    bad["records"][0]["sentiment"]["label"] = "positive"
    bad["records"][0]["sentiment"]["score"] = True
    with pytest.raises(ValueError, match="sentiment.score must be finite float in \\[-1.0, 1.0\\]"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 4. Sentiment: success carrying invalid label
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["sentiment"]["label"] = "unknown_label"
    with pytest.raises(ValueError, match="sentiment.label 'unknown_label' invalid for success status"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 5. Topics: non-assigned status carrying topic_id
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["topics"]["status"] = "fallback"
    bad["records"][0]["topics"]["topic_id"] = 5
    bad["records"][0]["topics"]["probability"] = None
    bad["records"][0]["topics"]["keywords"] = []
    with pytest.raises(ValueError, match="topics.topic_id must be None when status is 'fallback'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 6. Topics: non-assigned status carrying probability
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["topics"]["status"] = "empty_text"
    bad["records"][0]["topics"]["topic_id"] = None
    bad["records"][0]["topics"]["probability"] = 0.75
    bad["records"][0]["topics"]["keywords"] = []
    with pytest.raises(ValueError, match="topics.probability must be None when status is 'empty_text'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 7. Topics: non-assigned status carrying non-empty keywords
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["topics"]["status"] = "error"
    bad["records"][0]["topics"]["topic_id"] = None
    bad["records"][0]["topics"]["probability"] = None
    bad["records"][0]["topics"]["keywords"] = [{"word": "forbidden", "weight": None}]
    with pytest.raises(ValueError, match="topics.keywords must be empty list when status is 'error'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 8. Topics: outlier status carrying topic_id != -1
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["topics"]["status"] = "outlier"
    bad["records"][0]["topics"]["topic_id"] = 0
    bad["records"][0]["topics"]["keywords"] = []
    with pytest.raises(ValueError, match="topics.topic_id must equal -1 for outlier status"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 9. Topics: outlier status carrying keywords
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["topics"]["status"] = "outlier"
    bad["records"][0]["topics"]["topic_id"] = -1
    bad["records"][0]["topics"]["keywords"] = [{"word": "outlier_kw", "weight": None}]
    with pytest.raises(ValueError, match="topics.keywords must be empty list for outlier status"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 10. Topics: assigned carrying boolean topic_id
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["topics"]["status"] = "assigned"
    bad["records"][0]["topics"]["topic_id"] = False
    with pytest.raises(ValueError, match="topics.topic_id must be non-negative integer"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 11. Misinformation: non-success carrying veracity_label
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["misinformation"]["status"] = "fallback"
    bad["records"][0]["misinformation"]["veracity_label"] = "real"
    bad["records"][0]["misinformation"]["credibility_score"] = None
    bad["records"][0]["misinformation"]["risk_tier"] = None
    bad["records"][0]["misinformation"]["stylometric_signals"] = None
    with pytest.raises(ValueError, match="misinformation.veracity_label must be None when status is 'fallback'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 12. Misinformation: non-success carrying credibility_score
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["misinformation"]["status"] = "error"
    bad["records"][0]["misinformation"]["veracity_label"] = None
    bad["records"][0]["misinformation"]["credibility_score"] = 0.5
    bad["records"][0]["misinformation"]["risk_tier"] = None
    bad["records"][0]["misinformation"]["stylometric_signals"] = None
    with pytest.raises(ValueError, match="misinformation.credibility_score must be None when status is 'error'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 13. Misinformation: non-success carrying risk_tier
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["misinformation"]["status"] = "missing_status"
    bad["records"][0]["misinformation"]["veracity_label"] = None
    bad["records"][0]["misinformation"]["credibility_score"] = None
    bad["records"][0]["misinformation"]["risk_tier"] = "low"
    bad["records"][0]["misinformation"]["stylometric_signals"] = None
    with pytest.raises(ValueError, match="misinformation.risk_tier must be None when status is 'missing_status'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 14. Misinformation: non-success carrying stylometric_signals
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["misinformation"]["status"] = "unavailable"
    bad["records"][0]["misinformation"]["veracity_label"] = None
    bad["records"][0]["misinformation"]["credibility_score"] = None
    bad["records"][0]["misinformation"]["risk_tier"] = None
    bad["records"][0]["misinformation"]["stylometric_signals"] = {"uppercase_char_ratio": 0.1}
    with pytest.raises(ValueError, match="misinformation.stylometric_signals must be None when status is 'unavailable'"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

    # 15. Misinformation: success carrying boolean credibility_score
    bad = json.loads(json.dumps(valid_payload))
    bad["records"][0]["misinformation"]["status"] = "success"
    bad["records"][0]["misinformation"]["credibility_score"] = True
    with pytest.raises(ValueError, match="misinformation.credibility_score must be finite float"):
        ConsolidatedArtifactValidator.validate_consolidated_artifact(bad)

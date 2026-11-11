"""Unit tests for MisinformationArtifactValidator (Stage 9).

Verifies strict Schema-v2 rules:
- Valid payload validation.
- Schema versioning (requiring 2.0, rejecting 1.0, 3.0, non-string, missing).
- Status validation (success, partial, empty, error).
- Timestamp and provenance validation.
- Record array validation (reddit_records, youtube_records, predictions).
- Deterministic identity uniqueness (post_id, video_id).
- Prediction count matching total_records_processed.
- Prediction identifier correspondence with record arrays.
- Evidence-basis enum validation (accepting valid set, rejecting retired/unknown values).
- Veracity distribution consistency (sum matching total records, no negatives/booleans).
- Methodology disclosure validation.
- Numeric integrity (finite floats, no NaN/Inf, count consistency).
- Distribution consistency (risk_tier sums).
- Stage 10 data consolidator compatibility.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict
import pytest

from src.misinformation.misinformation_validator import MisinformationArtifactValidator


def build_valid_misinformation_payload() -> Dict[str, Any]:
    """Helper to construct a fully valid Schema-v2 misinformation artifact."""
    reddit_records = [
        {
            "post_id": "t3_ds1001",
            "title_cleaned": "Best practices for cleaning multilingual text",
            "text_cleaned": "Looking for emoji handling pipelines.",
            "author": "pipeline_builder",
            "veracity_label": "real",
            "credibility_score": 0.85,
            "risk_tier": "low",
            "misinformation_status": "success",
            "status": "success",
            "analyzed_at": "2026-11-10T12:00:00Z",
            "evidence_basis": "low_risk",
            "signals_detected": ["opinion_or_inquiry_context"],
            "assessment_rationale": "Subjective inquiry without misinformation markers.",
            "limitations": "Statistical assessment.",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.02,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.05,
                "word_count": 20.0,
                "avg_word_length": 5.5,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        },
        {
            "post_id": "t3_ai1002",
            "title_cleaned": "Evaluating BERTopic on noisy social media",
            "text_cleaned": "Discussion on HDBSCAN micro-clustering.",
            "author": "latent_explorer",
            "veracity_label": "real",
            "credibility_score": 0.82,
            "risk_tier": "low",
            "misinformation_status": "success",
            "status": "success",
            "analyzed_at": "2026-11-10T12:00:00Z",
            "evidence_basis": "low_risk",
            "signals_detected": [],
            "assessment_rationale": "Standard discussion.",
            "limitations": "Statistical assessment.",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.05,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 15.0,
                "avg_word_length": 6.0,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        },
    ]

    youtube_records = [
        {
            "video_id": "vid_abc123",
            "title_cleaned": "Introduction to Large Scale Social Media Intelligence",
            "description_cleaned": "Overview of NLP and graph databases.",
            "channel_title": "Data Insights",
            "veracity_label": "real",
            "credibility_score": 0.88,
            "risk_tier": "low",
            "misinformation_status": "success",
            "status": "success",
            "analyzed_at": "2026-11-10T12:00:00Z",
            "evidence_basis": "low_risk",
            "signals_detected": [],
            "assessment_rationale": "Educational overview.",
            "limitations": "Statistical assessment.",
            "stylometric_signals": {
                "uppercase_char_ratio": 0.08,
                "exclamation_intensity": 0.0,
                "question_intensity": 0.0,
                "word_count": 12.0,
                "avg_word_length": 6.2,
                "lexical_subjectivity_ratio": 0.0,
                "digits_ratio": 0.0,
                "all_caps_word_ratio": 0.0,
            },
        }
    ]

    predictions = [
        {
            "id": "t3_ds1001",
            "platform": "reddit",
            "veracity_label": "real",
            "credibility_score": 0.85,
            "risk_tier": "low",
            "status": "success",
            "text_preview": "Best practices for cleaning multilingual text...",
            "evidence_basis": "low_risk",
            "signals_detected": [],
            "assessment_rationale": "Inquiry.",
            "limitations": "Statistical assessment.",
            "stylometric_signals": reddit_records[0]["stylometric_signals"],
        },
        {
            "id": "t3_ai1002",
            "platform": "reddit",
            "veracity_label": "real",
            "credibility_score": 0.82,
            "risk_tier": "low",
            "status": "success",
            "text_preview": "Evaluating BERTopic on noisy social media...",
            "evidence_basis": "low_risk",
            "signals_detected": [],
            "assessment_rationale": "Discussion.",
            "limitations": "Statistical assessment.",
            "stylometric_signals": reddit_records[1]["stylometric_signals"],
        },
        {
            "id": "vid_abc123",
            "platform": "youtube",
            "veracity_label": "real",
            "credibility_score": 0.88,
            "risk_tier": "low",
            "status": "success",
            "text_preview": "Introduction to Large Scale Social Media Intelligence...",
            "evidence_basis": "low_risk",
            "signals_detected": [],
            "assessment_rationale": "Educational.",
            "limitations": "Statistical assessment.",
            "stylometric_signals": youtube_records[0]["stylometric_signals"],
        },
    ]

    return {
        "schema_version": "2.0",
        "status": "success",
        "generated_at": "2026-11-10T12:00:00Z",
        "provenance": {
            "reddit_input_file": "data/processed/reddit_posts_processed.json",
            "reddit_status": "loaded",
            "reddit_records_count": 2,
            "youtube_input_file": "data/processed/youtube_videos_processed.json",
            "youtube_status": "loaded",
            "youtube_records_count": 1,
        },
        "methodology_disclosure": {
            "nature_of_predictions": "Heuristic-adjusted model predictions.",
            "credibility_score_definition": "Heuristic-adjusted index in [0, 1].",
            "verification_status": "No external ground-truth verification.",
            "classifier_limitations": "Fallback fixture is a development diagnostic.",
            "protected_categories": "Opinions and education protected from falsehood.",
        },
        "total_records_processed": 3,
        "total_analyzed": 3,
        "average_credibility_score": 0.85,
        "flagged_high_risk_count": 0,
        "risk_tier_distribution": {
            "low": 3,
            "moderate": 0,
            "high_risk": 0,
            "unknown": 0,
        },
        "veracity_distribution": {
            "real": 3,
            "fake": 0,
        },
        "platform_breakdown": {
            "reddit": {
                "total_analyzed": 2,
                "average_credibility_score": 0.835,
                "flagged_high_risk_count": 0,
            },
            "youtube": {
                "total_analyzed": 1,
                "average_credibility_score": 0.88,
                "flagged_high_risk_count": 0,
            },
        },
        "predictions": predictions,
        "reddit_records": reddit_records,
        "youtube_records": youtube_records,
        "benchmark_evaluation": {
            "evaluation_method": "stratified_cross_validation",
            "evaluation_successful": True,
            "accuracy": 0.80,
            "f1_macro": 0.78,
        },
    }


def test_valid_payload_passes_validation() -> None:
    payload = build_valid_misinformation_payload()
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is True
    assert errors == []


def test_missing_and_unsupported_schema_version() -> None:
    payload = build_valid_misinformation_payload()

    # Missing
    del payload["schema_version"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("missing_schema_version" in e for e in errors)

    # Legacy 1.0
    payload["schema_version"] = "1.0"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("legacy_schema_version" in e for e in errors)

    # Future 3.0
    payload["schema_version"] = "3.0"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("unsupported_schema_version" in e for e in errors)


def test_invalid_status_rejected() -> None:
    payload = build_valid_misinformation_payload()
    payload["status"] = "somewhat_valid"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("'status' must be one of" in e for e in errors)


def test_provenance_validation() -> None:
    payload = build_valid_misinformation_payload()

    # Missing key
    del payload["provenance"]["reddit_status"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("missing required key 'reddit_status'" in e for e in errors)

    # Inconsistent status vs count (loaded with 0 records)
    payload = build_valid_misinformation_payload()
    payload["provenance"]["reddit_records_count"] = 0
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("reddit_status='loaded' requires reddit_records_count >= 1" in e for e in errors)


def test_duplicate_identifiers_rejected() -> None:
    payload = build_valid_misinformation_payload()
    # Duplicate post_id
    payload["reddit_records"].append(copy.deepcopy(payload["reddit_records"][0]))
    payload["total_records_processed"] = 4
    payload["risk_tier_distribution"]["low"] = 4
    payload["veracity_distribution"]["real"] = 4
    # Also add matching prediction to test duplicate in reddit_records
    payload["predictions"].append(copy.deepcopy(payload["predictions"][0]))
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("duplicate post_id" in e for e in errors)


def test_prediction_count_mismatch_rejected() -> None:
    """Validator enforces len(predictions) == total_records_processed."""
    payload = build_valid_misinformation_payload()
    # Remove one prediction
    payload["predictions"] = payload["predictions"][:2]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("'predictions' count (2) does not match 'total_records_processed' (3)" in e for e in errors)


def test_prediction_duplicate_identifiers_rejected() -> None:
    """Validator rejects duplicate identifiers in predictions array."""
    payload = build_valid_misinformation_payload()
    payload["predictions"][1]["id"] = payload["predictions"][0]["id"]
    payload["predictions"][1]["platform"] = payload["predictions"][0]["platform"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("duplicate prediction identifier" in e for e in errors)


def test_prediction_mismatched_identifiers_rejected() -> None:
    """Validator rejects prediction entries whose IDs do not correspond to records."""
    payload = build_valid_misinformation_payload()
    payload["predictions"][0]["id"] = "unknown_phantom_id"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("predictions missing records" in e or "predictions contain unexpected records" in e for e in errors)


def test_evidence_basis_enum_validation() -> None:
    """Validator accepts supported evidence-basis values and strictly rejects retired/unknown values."""
    # 1. Supported values
    for valid_basis in ("low_risk", "potential_risk", "pattern_flagged_risk", "insufficient_evidence"):
        payload = build_valid_misinformation_payload()
        payload["reddit_records"][0]["evidence_basis"] = valid_basis
        payload["predictions"][0]["evidence_basis"] = valid_basis
        valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
        assert valid is True, f"Failed for valid basis '{valid_basis}': {errors}"

    # 2. Retired value: 'evidence_supported_risk'
    payload = build_valid_misinformation_payload()
    payload["reddit_records"][0]["evidence_basis"] = "evidence_supported_risk"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("'evidence_basis' must be one of" in e for e in errors)

    # 3. Unknown value
    payload = build_valid_misinformation_payload()
    payload["predictions"][0]["evidence_basis"] = "totally_unsupported_evidence"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("'evidence_basis' must be one of" in e for e in errors)

    # 4. Missing evidence_basis
    payload = build_valid_misinformation_payload()
    del payload["reddit_records"][0]["evidence_basis"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("missing required field 'evidence_basis'" in e for e in errors)


def test_veracity_distribution_sum_and_counts_validation() -> None:
    """Validator enforces that veracity_distribution counts are non-negative ints and sum to total_records."""
    # Sum mismatch
    payload = build_valid_misinformation_payload()
    payload["veracity_distribution"]["real"] = 10  # sum = 10, total_records = 3
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("veracity_distribution sum (10) does not equal total_records_processed (3)" in e for e in errors)

    # Negative count
    payload = build_valid_misinformation_payload()
    payload["veracity_distribution"]["real"] = -1
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("must be non-negative int" in e for e in errors)

    # Boolean count
    payload = build_valid_misinformation_payload()
    payload["veracity_distribution"]["real"] = True  # type: ignore[assignment]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("must be non-negative int" in e for e in errors)


def test_missing_methodology_disclosure_rejected() -> None:
    """Validator requires methodology_disclosure field."""
    payload = build_valid_misinformation_payload()
    del payload["methodology_disclosure"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("missing_methodology_disclosure" in e for e in errors)


def test_non_finite_numeric_values_rejected() -> None:
    payload = build_valid_misinformation_payload()
    payload["average_credibility_score"] = float("nan")
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("average_credibility_score" in e for e in errors)

    payload = build_valid_misinformation_payload()
    payload["reddit_records"][0]["credibility_score"] = float("inf")
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("credibility_score" in e for e in errors)


def test_risk_tier_sum_mismatch_rejected() -> None:
    payload = build_valid_misinformation_payload()
    payload["risk_tier_distribution"]["low"] = 10  # sum = 10, total_records = 3
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("risk_tier_distribution sum" in e for e in errors)


def test_stage10_consolidator_field_compatibility() -> None:
    """Verifies that all fields expected by Stage 10 data consolidator are validated."""
    payload = build_valid_misinformation_payload()

    # Consolidator requires post_id on reddit_records
    del payload["reddit_records"][0]["post_id"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("missing or invalid 'post_id'" in e for e in errors)

    # Consolidator requires video_id on youtube_records
    payload = build_valid_misinformation_payload()
    del payload["youtube_records"][0]["video_id"]
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("missing or invalid 'video_id'" in e for e in errors)

    # Consolidator requires valid misinformation_status
    payload = build_valid_misinformation_payload()
    payload["reddit_records"][0]["misinformation_status"] = "invalid_status_xyz"
    valid, errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    assert valid is False
    assert any("misinformation_status" in e for e in errors)

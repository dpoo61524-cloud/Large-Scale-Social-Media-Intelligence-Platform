"""Unit tests for transparent heuristic-based MisinformationDetector (Stage 9).

Verifies the critical classification rules (including Stage 9B.1 corrections):
- Contextual protections (debunking, education, investigative reporting, critical quotes)
  suppress unsupported heuristic overrides ('fake' or 'high_risk') from conspiracy patterns.
- Contextual protections do NOT automatically force a positive 'real' or 'low_risk' verdict.
- Underlying classifier predictions and credibility scores are preserved without artificial inflation.
- When contextual protection applies, evidence_basis is set to 'insufficient_evidence'.
- Emotional language, opinions, questions, or negative sentiment alone never produce high_risk/fake.
- Sensational wording or uppercase shouting alone cannot establish factual falsity.
- Genuine unevidenced conspiracy patterns produce pattern_flagged_risk and high_risk.
- The retired 'evidence_supported_risk' value is never emitted.
- All rationales clearly state that pattern matching does not establish factual falsity.
- Empty text produces undetermined status and unknown risk.
- Boundary threshold behavior (0.40, 0.70).
- Deterministic output.
"""

from __future__ import annotations

import pytest

from src.misinformation.detector import (
    METHODOLOGICAL_LIMITATIONS_DISCLOSURE,
    MisinformationDetector,
    determine_risk_tier,
)


@pytest.fixture
def detector() -> MisinformationDetector:
    return MisinformationDetector(high_risk_threshold=0.40, moderate_risk_threshold=0.70)


def test_determine_risk_tier_boundaries() -> None:
    """Verifies threshold boundary values for risk tier categorization."""
    assert determine_risk_tier(0.0) == "high_risk"
    assert determine_risk_tier(0.3999) == "high_risk"
    assert determine_risk_tier(0.40) == "moderate"
    assert determine_risk_tier(0.55) == "moderate"
    assert determine_risk_tier(0.6999) == "moderate"
    assert determine_risk_tier(0.70) == "low"
    assert determine_risk_tier(0.85) == "low"
    assert determine_risk_tier(1.0) == "low"


def test_empty_and_whitespace_text_handling(detector: MisinformationDetector) -> None:
    """Empty or whitespace text produces undetermined status with unknown risk."""
    for empty_input in (None, "", "   ", "\t\n"):
        res = detector.assess_record(
            text=empty_input,
            platform="reddit",
            native_id="empty_1",
            binary=True,
        )
        assert res["status"] == "empty_text"
        assert res["veracity_label"] == "undetermined"
        assert res["credibility_score"] == 0.0
        assert res["risk_tier"] == "unknown"
        assert res["evidence_basis"] == "insufficient_evidence"
        assert "empty or whitespace" in res["assessment_rationale"]
        assert res["limitations"] == METHODOLOGICAL_LIMITATIONS_DISCLOSURE


def test_conspiracy_pattern_flagged_high_risk(detector: MisinformationDetector) -> None:
    """Explicit pseudoscientific or conspiracy markers trigger pattern-flagged high risk."""
    conspiracy_statements = [
        "Dangerous microchips are secretly dissolved in all municipal tap water to track everyone.",
        "The earth has no gravitational pull and is entirely flat according to leaked files.",
        "The moon is completely hollow and secretly housing extraterrestrial military bases.",
        "Drinking industrial chlorine bleach guarantees 100% immunity against all viral infections.",
    ]

    for stmt in conspiracy_statements:
        res = detector.assess_record(
            text=stmt,
            platform="reddit",
            native_id="consp_1",
            binary=True,
        )
        assert res["status"] == "success"
        assert res["veracity_label"] == "fake"
        assert res["risk_tier"] == "high_risk"
        assert res["credibility_score"] < 0.40
        assert res["evidence_basis"] == "pattern_flagged_risk"
        assert "linguistic risk" in res["assessment_rationale"].lower()
        assert "does not establish factual falsity" in res["assessment_rationale"].lower()
        # Verify retired terminology is NOT present
        assert res["evidence_basis"] != "evidence_supported_risk"
        assert "evidence-supported" not in res["assessment_rationale"].lower()


def test_debunking_conspiracy_does_not_force_real_or_fake(detector: MisinformationDetector) -> None:
    """Debunking context must not automatically force 'real' NOR automatically force 'fake'.

    Preserves underlying classifier prediction and score.
    """
    stmt = "Claims that 5G radiation causes virus outbreaks are false."

    # 1. When classifier predicts 'fake' with low score (e.g. 0.35)
    # The debunking context must NOT force veracity to 'real'!
    res_fake = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="debunk_fake",
        classifier_score=0.35,
        classifier_prediction="fake",
        binary=True,
    )
    assert res_fake["veracity_label"] == "fake", "Debunking must not automatically force 'real'"
    assert res_fake["credibility_score"] == 0.35, "Score must not be artificially increased"
    assert res_fake["risk_tier"] == "high_risk"
    assert res_fake["evidence_basis"] == "insufficient_evidence"

    # 2. When classifier predicts 'real' with high score (e.g. 0.85)
    # The conspiracy match must NOT force veracity to 'fake'!
    res_real = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="debunk_real",
        classifier_score=0.85,
        classifier_prediction="real",
        binary=True,
    )
    assert res_real["veracity_label"] == "real", "Debunking must suppress automatic 'fake' override"
    assert res_real["credibility_score"] == 0.85
    assert res_real["risk_tier"] == "low"
    assert res_real["evidence_basis"] == "insufficient_evidence"


def test_educational_discussion_does_not_force_real_or_fake(detector: MisinformationDetector) -> None:
    """Educational discussion must not automatically force either factual verdict."""
    stmt = "Students learned why the hollow moon theory was disproven in science class."

    # When classifier predicts 'fake'
    res_fake = detector.assess_record(
        text=stmt,
        platform="youtube",
        native_id="edu_fake",
        classifier_score=0.38,
        classifier_prediction="fake",
        binary=True,
    )
    assert res_fake["veracity_label"] == "fake"
    assert res_fake["credibility_score"] == 0.38
    assert res_fake["evidence_basis"] == "insufficient_evidence"

    # When classifier predicts 'real'
    res_real = detector.assess_record(
        text=stmt,
        platform="youtube",
        native_id="edu_real",
        classifier_score=0.82,
        classifier_prediction="real",
        binary=True,
    )
    assert res_real["veracity_label"] == "real"
    assert res_real["credibility_score"] == 0.82
    assert res_real["evidence_basis"] == "insufficient_evidence"


def test_investigative_reporting_does_not_force_real_or_fake(detector: MisinformationDetector) -> None:
    """Investigative reporting must not automatically force either factual verdict."""
    stmt = "Investigative reporters uncover evidence of illegal dumping in a local river."

    # When classifier predicts 'fake'
    res_fake = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="inv_fake",
        classifier_score=0.30,
        classifier_prediction="fake",
        binary=True,
    )
    assert res_fake["veracity_label"] == "fake"
    assert res_fake["credibility_score"] == 0.30
    assert res_fake["evidence_basis"] == "insufficient_evidence"

    # When classifier predicts 'real'
    res_real = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="inv_real",
        classifier_score=0.78,
        classifier_prediction="real",
        binary=True,
    )
    assert res_real["veracity_label"] == "real"
    assert res_real["credibility_score"] == 0.78
    assert res_real["evidence_basis"] == "insufficient_evidence"


def test_critical_quotation_does_not_imply_endorsement(detector: MisinformationDetector) -> None:
    """Critical quotation of a conspiracy claim does not trigger forced 'fake' verdict."""
    stmt = "Theorists argue that earth has no gravitational pull, which lacks scientific backing."

    res = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="crit_1",
        classifier_score=0.76,
        classifier_prediction="real",
        binary=True,
    )
    # Conspiracy pattern was matched, but critical quote context suppressed automatic 'fake'
    assert res["veracity_label"] == "real"
    assert res["credibility_score"] == 0.76
    assert res["evidence_basis"] == "insufficient_evidence"


def test_contextual_protection_preserves_classifier_score_and_moderate_tier(detector: MisinformationDetector) -> None:
    """Contextual protection does not force 'low' risk when classifier score indicates 'moderate' risk."""
    stmt = "Health officials confirmed that drinking industrial chlorine bleach is dangerous and has been debunked."

    # Moderate classifier score (0.55)
    res = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="mod_1",
        classifier_score=0.55,
        classifier_prediction="real",
        binary=True,
    )
    # Score is NOT inflated to >= 0.70; tier is moderate
    assert res["credibility_score"] == 0.55
    assert res["risk_tier"] == "moderate"
    assert res["veracity_label"] == "real"
    assert res["evidence_basis"] == "insufficient_evidence"


def test_contextual_fallback_when_no_classifier_provided(detector: MisinformationDetector) -> None:
    """When no classifier prediction is supplied, uses neutral fallback without claiming factual certainty."""
    stmt = "Debunking myths: why claims that 5g radiation causes virus outbreaks are completely false."

    res = detector.assess_record(
        text=stmt,
        platform="reddit",
        native_id="debunk_no_model",
        binary=True,
    )
    assert res["status"] == "success"
    assert res["credibility_score"] == 0.50
    assert res["risk_tier"] == "moderate"
    assert res["evidence_basis"] == "insufficient_evidence"
    assert "insufficient to establish" in res["assessment_rationale"]


def test_sensational_wording_or_uppercase_alone_not_factual_falsity(detector: MisinformationDetector) -> None:
    """Stylistic sensationalism or shouting alone does not establish factual falsity."""
    sensational_statements = [
        "BREAKING NEWS: SENSATIONAL DISCOVERY IN DEEP SEA EXPLORATION!!!",
        "THIS IS A REMARKABLE AND SHOCKING TURN OF EVENTS IN THE CHAMPIONSHIP MATCH!",
        "INCREDIBLE BREAKTHROUGH IN RENEWABLE SOLAR ENERGY BATTERIES!",
    ]

    for stmt in sensational_statements:
        res = detector.assess_record(
            text=stmt,
            platform="youtube",
            native_id="sens_1",
            binary=True,
        )
        assert res["status"] == "success"
        assert res["veracity_label"] == "real"
        assert res["risk_tier"] in ("low", "moderate")
        assert res["evidence_basis"] == "potential_risk"
        assert "sensationalism or uppercase" in res["assessment_rationale"].lower() or "stylistic" in res["assessment_rationale"].lower()


def test_strong_opinions_protected_from_misinformation_label(detector: MisinformationDetector) -> None:
    """Critical Rule: Strong opinions must NOT be labeled as misinformation or fake."""
    opinion_statements = [
        "In my opinion, this new monetary policy is terrible and completely disastrous for local stores.",
        "I think the administrative government is making an awful mistake with these tax hikes.",
        "Personally, I believe that electric cars are overrated and charging infrastructure is horrible.",
        "From my perspective, this movie is the worst disaster in cinema history.",
    ]

    for stmt in opinion_statements:
        res = detector.assess_record(
            text=stmt,
            platform="reddit",
            native_id="op_1",
            binary=True,
        )
        assert res["status"] == "success"
        assert res["veracity_label"] == "real"
        assert res["risk_tier"] == "low"
        assert res["credibility_score"] >= 0.70
        assert res["evidence_basis"] == "low_risk"
        assert "opinion" in res["assessment_rationale"].lower()


def test_negative_sentiment_and_emotional_language_protected(detector: MisinformationDetector) -> None:
    """Critical Rule: Negative sentiment and emotional venting must NOT indicate misinformation."""
    emotional_statements = [
        "I am furious that our flight was delayed six hours! Disgusting airline service!",
        "This update is so frustrating and annoying, I hate the new user interface.",
        "Heartbreaking news about the flood damages in our regional municipal district.",
    ]

    for stmt in emotional_statements:
        res = detector.assess_record(
            text=stmt,
            platform="youtube",
            native_id="emo_1",
            binary=True,
        )
        assert res["status"] == "success"
        assert res["veracity_label"] == "real"
        assert res["risk_tier"] in ("low", "moderate")
        assert res["credibility_score"] >= 0.50
        assert res["evidence_basis"] in ("low_risk", "potential_risk")


def test_conversational_questions_and_inquiries_protected(detector: MisinformationDetector) -> None:
    """Questions and requests for advice must NOT be penalized as misinformation."""
    questions = [
        "What pipelines do you recommend for emoji handling and language identification?",
        "Has anyone evaluated BERTopic for dynamic topic discovery over short Twitter comments?",
        "How do you configure index mapping and Elasticsearch templates properly?",
    ]

    for q in questions:
        res = detector.assess_record(
            text=q,
            platform="reddit",
            native_id="quest_1",
            binary=True,
        )
        assert res["status"] == "success"
        assert res["veracity_label"] == "real"
        assert res["risk_tier"] == "low"
        assert res["credibility_score"] >= 0.70


def test_technical_and_scientific_content_low_risk(detector: MisinformationDetector) -> None:
    """Standard educational, technical, or scientific content receives low risk."""
    tech_text = (
        "Introduction to Large Scale Social Media Intelligence: A comprehensive overview "
        "of NLP pipelines, graph databases, and topic modeling."
    )
    res = detector.assess_record(
        text=tech_text,
        platform="youtube",
        native_id="tech_1",
        classifier_score=0.25,  # Simulate out-of-vocabulary low score from tiny classifier
        binary=True,
    )
    assert res["status"] == "success"
    assert res["veracity_label"] == "real"
    assert res["risk_tier"] == "low"
    assert res["credibility_score"] >= 0.70


def test_unsubstantiated_sensational_claim_pattern_flagged(detector: MisinformationDetector) -> None:
    """Sensational viral claim without proof triggers pattern-flagged risk."""
    viral_text = (
        "SHOCKING SECRET: Alien base discovered on dark side of moon!!! "
        "URGENT PROOF that world governments are hiding the extraterrestrial truth!"
    )
    res = detector.assess_record(
        text=viral_text,
        platform="reddit",
        native_id="viral_1",
        binary=True,
    )
    assert res["status"] == "success"
    assert res["veracity_label"] == "fake"
    assert res["risk_tier"] == "high_risk"
    assert res["evidence_basis"] == "pattern_flagged_risk"
    assert "does not establish factual falsity" in res["assessment_rationale"].lower()


def test_detector_fine_grained_six_class_mapping(detector: MisinformationDetector) -> None:
    """Verifies assessment output when binary=False (6 fine-grained classes)."""
    conspiracy = "Drinking industrial chlorine bleach guarantees 100% immunity against all viral infections."
    res = detector.assess_record(
        text=conspiracy,
        platform="reddit",
        native_id="six_consp",
        binary=False,
    )
    assert res["veracity_label"] in ("pants-fire", "false")
    assert res["risk_tier"] == "high_risk"
    assert res["evidence_basis"] == "pattern_flagged_risk"

    opinion = "In my opinion, Python is an awesome programming language for data pipelines."
    res_op = detector.assess_record(
        text=opinion,
        platform="reddit",
        native_id="six_op",
        binary=False,
    )
    assert res_op["veracity_label"] in ("mostly-true", "true")
    assert res_op["risk_tier"] == "low"


def test_detector_output_determinism(detector: MisinformationDetector) -> None:
    """Verifies that identical inputs produce identical assessment outputs deterministically."""
    sample = "Astronomers published new peer-reviewed findings in astrophysics journal."
    res1 = detector.assess_record(text=sample, platform="reddit", native_id="det_1", binary=True)
    res2 = detector.assess_record(text=sample, platform="reddit", native_id="det_1", binary=True)

    assert res1["veracity_label"] == res2["veracity_label"]
    assert res1["credibility_score"] == res2["credibility_score"]
    assert res1["risk_tier"] == res2["risk_tier"]
    assert res1["evidence_basis"] == res2["evidence_basis"]
    assert res1["signals_detected"] == res2["signals_detected"]

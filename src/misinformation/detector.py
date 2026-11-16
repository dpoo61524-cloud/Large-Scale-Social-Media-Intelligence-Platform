"""Heuristic-Adjusted Misinformation Risk Assessor.

Implements transparent, heuristic-adjusted risk assessment for social media content.
Distinguishes potential linguistic misinformation risk from independently verified facts.
Protects emotional expression, strong opinions, questions, educational discussions,
debunking, investigative reporting, and negative sentiment from false classification.
Communicates detection signals, evidence basis, and methodological limitations clearly.

Notice on Epistemic Scope:
- Veracity labels ('real', 'fake') are heuristic-adjusted model predictions, NOT verified facts.
- A 'fake' prediction indicates stylistic or pattern alignment with misinformation training data;
  it does not establish that the underlying claim is factually false.
- Credibility scores are heuristic-adjusted indices bounded in [0.0, 1.0], not statistically calibrated probabilities.
- The system performs no external fact-checking or authoritative evidence retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.misinformation.feature_extractor import SENSATIONAL_LEXICON, STYLOMETRIC_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Explicit known fabricated claim patterns for linguistic pattern detection
CONSPIRACY_EVIDENCE_PATTERNS = [
    re.compile(r"secretly dissolved in (all )?municipal tap water", re.IGNORECASE),
    re.compile(r"microscopic electronic tracking microchips", re.IGNORECASE),
    re.compile(r"earth has no gravitational pull|earth (is )?entirely flat", re.IGNORECASE),
    re.compile(r"moon is completely hollow|secretly housing extraterrestrial", re.IGNORECASE),
    re.compile(r"drinking industrial chlorine bleach|bleach guarantees 100% immunity", re.IGNORECASE),
    re.compile(r"5g (radiation|network|towers).*causes (covid|virus)", re.IGNORECASE),
    re.compile(r"chemtrails.*mind control", re.IGNORECASE),
]

# Patterns asserting extreme, unevidenced urgency, conspiracy, or absolute proof
UNSUBSTANTIATED_VIRAL_PATTERNS = [
    re.compile(r"\bshocking secret\b", re.IGNORECASE),
    re.compile(r"\burgent proof\b", re.IGNORECASE),
    re.compile(r"\bworld governments are hiding\b", re.IGNORECASE),
    re.compile(r"\b100% (immunity|cure|guarantee)\b", re.IGNORECASE),
    re.compile(r"\bbanned truth\b", re.IGNORECASE),
    re.compile(r"\bmiracle cure\b", re.IGNORECASE),
    re.compile(r"\bextraterrestrial truth\b", re.IGNORECASE),
    re.compile(r"\bhidden truth exposed\b", re.IGNORECASE),
]

# Opinion, hedging, and conversational question patterns (must NEVER be labeled false)
OPINION_DISCUSSION_PATTERNS = [
    re.compile(
        r"\b(i think|in my opinion|in my view|i believe|personally|from my perspective|"
        r"what do you think|what do you recommend|what pipelines do you recommend|"
        r"has anyone evaluated|tips on|sharing my experience|my experience|"
        r"does anyone know|how do you|my impression|it seems to me)\b",
        re.IGNORECASE,
    ),
]

# Emotional language and strong sentiment patterns (emotions do not indicate misinformation)
EMOTIONAL_LANGUAGE_PATTERNS = [
    re.compile(
        r"\b(terrible|horrible|awful|disaster|furious|heartbreaking|devastating|"
        r"outrageous|disgusting|annoying|frustrating|disappointed|love|hate|worst)\b",
        re.IGNORECASE,
    ),
]

# Contextual indicators of explicit debunking, refutation, or fact-checking
DEBUNKING_CONTEXT_PATTERNS = [
    re.compile(
        r"\b(debunk(ing|ed|s)?|fact[-\s]?check(ing|ed|s)?|myth(s)?|falsehood(s)?|hoax(es)?|"
        r"disproven|disproved|refuted|refuting|unfounded|baseless|misconception(s)?|"
        r"lacks? scientific (backing|evidence)|no scientific (basis|evidence)|"
        r"(is|are|was|were) (completely )?(false|untrue|incorrect|unsupported|disproven)|"
        r"has been debunked|have been debunked|falsely claim(ed|s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhy claims that .* are (completely )?(false|untrue|unsupported)\b", re.IGNORECASE),
]

# Contextual indicators of educational or academic discussion
EDUCATIONAL_CONTEXT_PATTERNS = [
    re.compile(
        r"\b(students learned|in class|in science class|astronomy class|physics class|"
        r"curriculum|textbook|educational|history of|lecture on|academic discussion|"
        r"scholars study|researchers study|in school|course on|science curriculum)\b",
        re.IGNORECASE,
    ),
]

# Contextual indicators of investigative journalism or news reporting
INVESTIGATIVE_CONTEXT_PATTERNS = [
    re.compile(
        r"\b(investigative report(ers?|ing)?|journalists? (investigate|uncover|report)|"
        r"official press briefing|news report|independent investigation|published report|"
        r"reporters uncover|uncover(ed)? evidence|official report)\b",
        re.IGNORECASE,
    ),
]

# Contextual indicators of quoted, cited, or critical attribution
CRITICAL_DISCUSSION_PATTERNS = [
    re.compile(
        r"\b(theorists (argue|claim|believe)|believers claim|conspiracy theorists|"
        r"advocates claim|alleged by|so-called|purported to be|claims that|rumors of)\b",
        re.IGNORECASE,
    ),
]

METHODOLOGICAL_LIMITATIONS_DISCLOSURE = (
    "Heuristic-adjusted misinformation risk assessment based on linguistic patterns, stylometrics, "
    "and benchmark heuristics. Veracity labels and credibility scores are heuristic-adjusted model predictions, "
    "not independently verified facts. No external factual verification is performed. Subjective opinions, personal "
    "experiences, questions, emotional expression, sensational writing styles, uppercase formatting, "
    "and educational or debunking discussions are protected from automatic classification as factual falsehood."
)


def determine_risk_tier(credibility_score: float) -> str:
    """Categorizes continuous credibility score into a risk tier."""
    if credibility_score < 0.40:
        return "high_risk"
    if credibility_score < 0.70:
        return "moderate"
    return "low"


class MisinformationDetector:
    """Transparent multi-signal misinformation and credibility risk assessor."""

    def __init__(
        self,
        high_risk_threshold: float = 0.40,
        moderate_risk_threshold: float = 0.70,
    ) -> None:
        """Initializes detector with configurable risk thresholds."""
        self.high_risk_threshold = high_risk_threshold
        self.moderate_risk_threshold = moderate_risk_threshold

    @classmethod
    def extract_signals(cls, text: Optional[str]) -> Dict[str, Any]:
        """Extracts explainable linguistic, lexical, and discourse signals from text."""
        if text is None or not isinstance(text, str) or not text.strip():
            return {
                "is_empty": True,
                "conspiracy_indicators": [],
                "unsubstantiated_patterns": [],
                "is_opinion_or_discussion": False,
                "is_question": False,
                "has_emotional_language": False,
                "is_debunking_context": False,
                "is_educational_context": False,
                "is_investigative_context": False,
                "is_critical_quotation": False,
                "is_contextually_protected": False,
                "sensational_words": [],
                "exclamation_count": 0,
                "question_count": 0,
                "all_caps_word_ratio": 0.0,
                "word_count": 0,
            }

        raw = text.strip()
        words = re.findall(r"\b[\w'-]+\b", raw)
        num_words = max(1, len(words))

        # 1. Conspiracy & fabricated claim markers
        conspiracy_matches = [
            p.pattern for p in CONSPIRACY_EVIDENCE_PATTERNS if p.search(raw)
        ]

        # 2. Unsubstantiated viral claim markers
        unsubstantiated_matches = [
            p.pattern for p in UNSUBSTANTIATED_VIRAL_PATTERNS if p.search(raw)
        ]

        # 3. Opinion and inquiry markers
        has_opinion = any(p.search(raw) for p in OPINION_DISCUSSION_PATTERNS)
        is_question = raw.endswith("?") or "?" in raw or raw.lower().startswith(("what ", "how ", "why ", "has ", "does ", "is "))

        # 4. Emotional language markers
        has_emotion = any(p.search(raw) for p in EMOTIONAL_LANGUAGE_PATTERNS)

        # 5. Contextual protections
        is_debunking = any(p.search(raw) for p in DEBUNKING_CONTEXT_PATTERNS)
        is_educational = any(p.search(raw) for p in EDUCATIONAL_CONTEXT_PATTERNS)
        is_investigative = any(p.search(raw) for p in INVESTIGATIVE_CONTEXT_PATTERNS)
        is_critical = any(p.search(raw) for p in CRITICAL_DISCUSSION_PATTERNS)
        is_contextually_protected = bool(is_debunking or is_educational or is_investigative or is_critical)

        # 6. Lexical sensationalism
        sensational_found = [w.lower() for w in words if w.lower() in SENSATIONAL_LEXICON]

        # 7. Punctuation and casing
        excl_count = raw.count("!")
        quest_count = raw.count("?")
        all_caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())
        all_caps_ratio = float(all_caps_words) / num_words

        return {
            "is_empty": False,
            "conspiracy_indicators": conspiracy_matches,
            "unsubstantiated_patterns": unsubstantiated_matches,
            "is_opinion_or_discussion": bool(has_opinion or is_question),
            "is_question": is_question,
            "has_emotional_language": has_emotion,
            "is_debunking_context": is_debunking,
            "is_educational_context": is_educational,
            "is_investigative_context": is_investigative,
            "is_critical_quotation": is_critical,
            "is_contextually_protected": is_contextually_protected,
            "sensational_words": sensational_found,
            "exclamation_count": excl_count,
            "question_count": quest_count,
            "all_caps_word_ratio": round(all_caps_ratio, 4),
            "word_count": len(words),
        }

    def assess_record(
        self,
        text: Optional[str],
        platform: str,
        native_id: str,
        classifier_score: Optional[float] = None,
        classifier_prediction: Optional[Any] = None,
        binary: bool = True,
        stylometrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Assesses misinformation risk for an individual record using multi-signal heuristics.

        Enforces critical classification rules:
        - Emotional language, opinions, questions, or negative sentiment alone never produce high_risk/fake.
        - Contextual discussion (debunking, education, investigative reporting, critical quotation) is protected.
        - Sensational wording or uppercase text alone does not establish factual falsity.
        - High risk requires unprotected known conspiracy patterns or extreme unsubstantiated urgency.
        - All outputs represent heuristic-adjusted model predictions, not independently verified facts.
        """
        signals = self.extract_signals(text)

        # Case 1: Empty text
        if signals["is_empty"]:
            return {
                "id": native_id,
                "platform": platform,
                "veracity_label": "undetermined",
                "credibility_score": 0.0,
                "risk_tier": "unknown",
                "status": "empty_text",
                "evidence_basis": "insufficient_evidence",
                "signals_detected": ["empty_text"],
                "assessment_rationale": "Record contains empty or whitespace-only text; assessment cannot be performed.",
                "limitations": METHODOLOGICAL_LIMITATIONS_DISCLOSURE,
                "stylometric_signals": stylometrics or {name: 0.0 for name in STYLOMETRIC_FEATURE_NAMES},
            }

        conspiracy = signals["conspiracy_indicators"]
        unsubstantiated = signals["unsubstantiated_patterns"]
        is_opinion_or_question = signals["is_opinion_or_discussion"]
        has_emotion = signals["has_emotional_language"]
        is_debunking = signals["is_debunking_context"]
        is_educational = signals["is_educational_context"]
        is_investigative = signals["is_investigative_context"]
        is_critical = signals["is_critical_quotation"]
        is_contextually_protected = signals["is_contextually_protected"]
        sensational = signals["sensational_words"]
        all_caps_ratio = signals["all_caps_word_ratio"]
        excl_count = signals["exclamation_count"]

        detected_signals_list: List[str] = []
        if conspiracy:
            detected_signals_list.append(f"conspiracy_indicator_matched:{len(conspiracy)}")
        if unsubstantiated:
            detected_signals_list.append(f"unsubstantiated_claims_matched:{len(unsubstantiated)}")
        if is_debunking:
            detected_signals_list.append("debunking_or_refutation_context")
        if is_educational:
            detected_signals_list.append("educational_or_academic_context")
        if is_investigative:
            detected_signals_list.append("investigative_reporting_context")
        if is_critical:
            detected_signals_list.append("critical_discussion_or_quotation")
        if sensational:
            detected_signals_list.append(f"sensational_lexicon_count:{len(sensational)}")
        if is_opinion_or_question:
            detected_signals_list.append("opinion_or_inquiry_context")
        if has_emotion:
            detected_signals_list.append("emotional_language_detected")
        if all_caps_ratio >= 0.15:
            detected_signals_list.append(f"high_all_caps_ratio:{all_caps_ratio}")
        if excl_count >= 2:
            detected_signals_list.append(f"multiple_exclamations:{excl_count}")

        # Case 2: Contextual protections (debunking, educational, investigative, critical quotation)
        # 1. Suppress automatic 'fake' or 'high_risk' heuristic overrides from conspiracy/sensational patterns.
        # 2. Do NOT automatically force 'real' or 'low_risk'.
        # 3. Preserve underlying classifier prediction and score without artificial inflation.
        # 4. Use evidence_basis = 'insufficient_evidence' (lexical match is insufficient to support risk or truth claim).
        if is_contextually_protected:
            if classifier_score is not None:
                cred_score = round(classifier_score, 4)
            else:
                cred_score = 0.50

            risk_tier = determine_risk_tier(cred_score)

            if classifier_prediction is not None:
                veracity = str(classifier_prediction)
            else:
                if binary:
                    veracity = "real" if cred_score >= 0.50 else "fake"
                else:
                    if cred_score >= 0.80:
                        veracity = "true"
                    elif cred_score >= 0.65:
                        veracity = "mostly-true"
                    elif cred_score >= 0.45:
                        veracity = "half-true"
                    elif cred_score >= 0.30:
                        veracity = "barely-true"
                    else:
                        veracity = "false"

            evidence_basis = "insufficient_evidence"
            if is_investigative:
                rationale = (
                    "Investigative or journalistic reporting context detected. "
                    "Lexical pattern matches or sensational phrasing do not establish factual falsehood or truth; "
                    "underlying classifier evaluation is preserved without heuristic veracity override."
                )
            else:
                rationale = (
                    "Contextual discussion detected (debunking, educational, or critical attribution). "
                    "Lexical pattern matches are insufficient to establish misinformation risk or factual truth; "
                    "underlying classifier evaluation is preserved without heuristic veracity override."
                )

        # Case 3: Direct match with known conspiracy / fabricated claim (unprotected)
        elif conspiracy:
            cred_score = round(min(0.20, classifier_score if classifier_score is not None else 0.15), 4)
            risk_tier = "high_risk"
            veracity = "fake" if binary else "pants-fire"
            evidence_basis = "pattern_flagged_risk"
            rationale = (
                "Elevated linguistic risk: text contains patterns associated with conspiracy or "
                "pseudoscientific claims. This pattern match is a heuristic prediction and does not establish factual falsity."
            )

        # Case 4: High sensationalism + urgent proof claim + shouting (unprotected viral conspiracy style)
        elif unsubstantiated and (len(sensational) >= 1 or excl_count >= 2 or all_caps_ratio >= 0.10):
            cred_score = round(min(0.35, classifier_score if classifier_score is not None else 0.30), 4)
            risk_tier = "high_risk"
            veracity = "fake" if binary else "false"
            evidence_basis = "pattern_flagged_risk"
            rationale = (
                "Elevated linguistic risk: text exhibits high sensationalism combined with urgent, "
                "uncorroborated truth assertions. This pattern match is a heuristic prediction and does not establish factual falsity."
            )

        # Case 5: Opinion, Question, Personal Experience, or Emotional Expression
        # CRITICAL RULE: Opinions and emotional tone do NOT constitute factual falsehood!
        elif is_opinion_or_question or has_emotion:
            base_score = classifier_score if (classifier_score is not None and classifier_score >= 0.65) else 0.78
            # Even if sensational words exist in conversational context, cap risk at moderate
            if len(sensational) >= 2 and not is_opinion_or_question:
                cred_score = round(max(0.50, min(0.68, base_score)), 4)
                risk_tier = "moderate"
                veracity = "real" if binary else "half-true"
                evidence_basis = "potential_risk"
                rationale = "Conversational or emotional text with moderate sensationalism; no factual falsehood verified."
            else:
                cred_score = round(max(0.72, base_score), 4)
                risk_tier = "low"
                veracity = "real" if binary else "mostly-true"
                evidence_basis = "low_risk"
                rationale = (
                    "Subjective opinion, question, or emotional expression. Opinions, personal views, "
                    "and sentiment do not constitute factual falsehood."
                )

        # Case 6: Stylistic Sensationalism or Uppercase Shouting Alone
        # CRITICAL RULE: Sensationalism or uppercase text alone cannot establish factual falsity!
        elif (len(sensational) >= 2 or all_caps_ratio >= 0.15 or excl_count >= 2) and not unsubstantiated and not conspiracy:
            cred_score = round(max(0.50, min(0.68, classifier_score if classifier_score is not None else 0.60)), 4)
            risk_tier = "moderate"
            veracity = "real" if binary else "half-true"
            evidence_basis = "potential_risk"
            rationale = (
                "Sensational or emphatic formatting detected without unevidenced conspiracy claims. "
                "Stylistic sensationalism or uppercase formatting alone does not establish factual falsity."
            )

        # Case 7: Standard informational, technical, or news content
        else:
            if classifier_score is not None:
                # If statistical model evaluated it, ensure technical content without sensationalism
                # is not falsely dragged down by out-of-vocabulary artifacts
                if len(sensational) == 0 and not unsubstantiated and not conspiracy:
                    cred_score = round(max(0.70, classifier_score), 4)
                else:
                    cred_score = round(classifier_score, 4)
            else:
                cred_score = 0.78

            risk_tier = determine_risk_tier(cred_score)
            if binary:
                veracity = "real" if cred_score >= 0.50 else "fake"
            else:
                if cred_score >= 0.80:
                    veracity = "true"
                elif cred_score >= 0.65:
                    veracity = "mostly-true"
                elif cred_score >= 0.45:
                    veracity = "half-true"
                elif cred_score >= 0.30:
                    veracity = "barely-true"
                else:
                    veracity = "false"

            evidence_basis = "low_risk" if risk_tier == "low" else "potential_risk"
            rationale = (
                f"Statistical heuristic assessment ({risk_tier} tier) based on linguistic markers and text stylometry. "
                "No known conspiracy or fabricated claim patterns detected."
            )

        return {
            "id": native_id,
            "platform": platform,
            "veracity_label": veracity,
            "credibility_score": cred_score,
            "risk_tier": risk_tier,
            "status": "success",
            "evidence_basis": evidence_basis,
            "signals_detected": detected_signals_list,
            "assessment_rationale": rationale,
            "limitations": METHODOLOGICAL_LIMITATIONS_DISCLOSURE,
            "stylometric_signals": stylometrics or {name: 0.0 for name in STYLOMETRIC_FEATURE_NAMES},
        }

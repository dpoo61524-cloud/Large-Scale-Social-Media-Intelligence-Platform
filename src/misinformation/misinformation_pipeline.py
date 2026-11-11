"""Misinformation Detection Pipeline Orchestrator (Stage 9).

Orchestrates transparent, heuristic-adjusted misinformation risk assessment across Reddit
and YouTube processed records. Integrates LIAR benchmark model estimation, stylometric
feature extraction, explicit provenance tracking, strict Schema-v2 validation,
and atomic temporary-file promotion.

Methodological Disclosures:
- Veracity labels ('real', 'fake') are heuristic-adjusted model predictions, NOT ground truth.
- Credibility scores are heuristic-adjusted indices bounded in [0.0, 1.0], NOT calibrated probabilities.
- When no pretrained model is supplied, the classifier falls back to an offline 24-sample development
  fixture, which is a development diagnostic and not representative of real-world scale or production accuracy.
- No external factual verification is performed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.misinformation.classifier import MisinformationClassifier
from src.misinformation.dataset_loader import (
    ID_TO_BINARY_LABEL,
    ID_TO_FINE_GRAINED_LABEL,
    LiarDatasetLoader,
)
from src.misinformation.detector import (
    METHODOLOGICAL_LIMITATIONS_DISCLOSURE,
    MisinformationDetector,
    determine_risk_tier,
)
from src.misinformation.feature_extractor import (
    STYLOMETRIC_FEATURE_NAMES,
    LinguisticFeatureExtractor,
)
from src.misinformation.misinformation_validator import MisinformationArtifactValidator

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_INPUT = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_INPUT = Path("data/processed/youtube_videos_processed.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/misinformation_analysis_results.json")
SUPPORTED_SCHEMA_VERSION = "2.0"

METHODOLOGY_DISCLOSURE: Dict[str, str] = {
    "nature_of_predictions": (
        "Heuristic-adjusted model predictions based on linguistic patterns, stylometrics, "
        "and supervised benchmark heuristics. Predictions do not constitute ground-truth factual verification."
    ),
    "credibility_score_definition": (
        "Heuristic-adjusted credibility index bounded in [0.0, 1.0], not a statistically calibrated probability."
    ),
    "verification_status": (
        "No external ground-truth fact-checking or authoritative evidence retrieval is performed. "
        "A 'fake' label indicates predicted stylistic or pattern alignment with misinformation training data, "
        "not proof of underlying factual falsehood."
    ),
    "classifier_limitations": (
        "When no pretrained model is supplied, the classifier falls back to an offline development fixture. "
        "The fallback fixture is a development diagnostic and is not representative of real-world scale or production accuracy."
    ),
    "protected_categories": (
        "Subjective opinions, questions, emotional expression, sensational writing styles, uppercase formatting, "
        "and educational or debunking discussions are protected from automatic classification as factual falsehood."
    ),
}


def _determine_risk_tier(credibility_score: float) -> str:
    """Categorize credibility score into a risk tier."""
    return determine_risk_tier(credibility_score)


def load_and_classify_source(
    file_path: Path,
    source_type: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], int]:
    """Loads and classifies an input social media processed file against schema contracts.

    Evaluates file existence, format validity, and data integrity to classify
    into exactly one of four mutual outcomes:
    - 'loaded': file exists, is valid JSON list, all items conform to source
      contract (Reddit requires 'post_id', YouTube requires 'video_id'), length >= 1.
    - 'empty': file exists, is valid JSON list, length == 0.
    - 'missing': file does not exist on disk.
    - 'corrupted': file exists but is unreadable, invalid JSON, not a list,
      or contains malformed non-dict items or records violating schema contracts.

    Args:
        file_path: Path to the input JSON file.
        source_type: Required source discriminator ('reddit' or 'youtube').

    Returns:
        Tuple of (status: str, records: list[dict[str, Any]], count: int).

    Raises:
        ValueError: If source_type is omitted or invalid.
    """
    if source_type not in ("reddit", "youtube"):
        raise ValueError(
            f"Invalid source_type: {source_type!r}. Must be explicitly 'reddit' or 'youtube'."
        )

    if not file_path.exists():
        logger.warning("Input source file does not exist: %s", file_path)
        return "missing", [], 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to read/parse input source %s: %s", file_path, exc)
        return "corrupted", [], 0

    if not isinstance(data, list):
        logger.error("Expected JSON list in %s, got %s", file_path, type(data).__name__)
        return "corrupted", [], 0

    if len(data) == 0:
        return "empty", [], 0

    valid_records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logger.error("Malformed record at index %d in %s: expected dict, got %s", idx, file_path, type(item).__name__)
            return "corrupted", [], 0

        if source_type == "reddit":
            pid = item.get("post_id")
            if not isinstance(pid, str) or not pid.strip():
                logger.error("Malformed Reddit record at index %d in %s: missing or invalid post_id", idx, file_path)
                return "corrupted", [], 0
            clean_id = pid.strip()
            if clean_id not in seen_ids:
                seen_ids.add(clean_id)
                valid_records.append(item)
        else:  # source_type == "youtube"
            vid = item.get("video_id")
            if not isinstance(vid, str) or not vid.strip():
                logger.error("Malformed YouTube record at index %d in %s: missing or invalid video_id", idx, file_path)
                return "corrupted", [], 0
            clean_id = vid.strip()
            if clean_id not in seen_ids:
                seen_ids.add(clean_id)
                valid_records.append(item)

    return "loaded", valid_records, len(valid_records)


def _extract_text_for_record(rec: Dict[str, Any], platform: str) -> Tuple[str, bool]:
    """Extracts cleaned text and evaluates if it is empty or whitespace-only."""
    if platform == "reddit":
        title_cleaned = rec.get("title_cleaned")
        text_cleaned = rec.get("text_cleaned")
        title_raw = rec.get("title")
        text_raw = rec.get("text")

        cleaned_text = (
            text_cleaned
            or title_cleaned
            or f"{title_raw or ''} {text_raw or ''}".strip()
        )
    else:
        title_cleaned = rec.get("title_cleaned")
        desc_cleaned = rec.get("description_cleaned")
        title_raw = rec.get("title")
        desc_raw = rec.get("description")

        cleaned_text = (
            title_cleaned
            or desc_cleaned
            or f"{title_raw or ''} {desc_raw or ''}".strip()
        )

    if cleaned_text is None or not str(cleaned_text).strip():
        return "", True
    return str(cleaned_text).strip(), False


def _promote_artifact_atomically(payload: Dict[str, Any], dest_path: Path) -> None:
    """Atomically serializes, validates, and promotes a misinformation artifact.

    Guarantees that readers of dest_path will never observe a partially written
    or corrupt file. If any error occurs prior to or during replacement, any
    existing canonical file is untouched and temporary files are cleaned up.

    Args:
        payload: Fully assembled and in-memory validated schema-v2 payload dictionary.
        dest_path: Canonical destination path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    unique_id = uuid.uuid4().hex
    temp_path = dest_path.parent / f".tmp_{dest_path.name}_{unique_id}"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except AttributeError:
                pass

        # Read-back verification before replacement
        with open(temp_path, "r", encoding="utf-8") as rf:
            read_back_payload = json.load(rf)

        rb_valid, rb_errors = MisinformationArtifactValidator.validate_misinformation_artifact(read_back_payload)
        if not rb_valid:
            raise ValueError(f"Read-back misinformation artifact validation failed: {rb_errors}")

        # Atomic replacement
        os.replace(temp_path, dest_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as exc:
                logger.debug("Failed to clean up temporary file %s: %s", temp_path, exc)


def evaluate_cross_validated(
    statements: List[str],
    labels: List[int],
    target_names: Optional[List[str]] = None,
    binary: bool = True,
    n_splits: int = 3,
    random_state: int = 42,
    is_development_fixture: bool = True,
) -> Dict[str, Any]:
    """Perform leak-free out-of-fold stratified cross-validation.

    Feature extraction and scaling are fitted strictly inside each training fold,
    preventing any test-set leakage.
    """
    y_all = np.asarray(labels)
    unique_classes, counts = np.unique(y_all, return_counts=True)
    min_class_count = int(np.min(counts)) if len(counts) > 0 else 0

    class_dist = {str(k): int(v) for k, v in zip(unique_classes, counts)}

    if len(unique_classes) < 2 or min_class_count < 2 or len(statements) < 6:
        return {
            "evaluation_method": "stratified_cross_validation",
            "evaluation_successful": False,
            "reason": (
                f"Insufficient samples per class (min_class_count={min_class_count} < 2) "
                f"or too few classes ({len(unique_classes)} < 2) for reliable cross-validation."
            ),
            "number_of_samples": len(statements),
            "class_distribution": class_dist,
            "is_development_fixture": is_development_fixture,
            "note": "Evaluation skipped due to insufficient sample support.",
        }

    actual_folds = max(2, min(n_splits, min_class_count))
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(y_all), dtype=int)
    oof_probs = np.zeros((len(y_all), len(unique_classes)), dtype=float)

    for train_idx, val_idx in skf.split(statements, y_all):
        train_texts = [statements[i] for i in train_idx]
        train_y = y_all[train_idx]
        val_texts = [statements[i] for i in val_idx]

        fold_ext = LinguisticFeatureExtractor(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            use_stylometrics=True,
        )
        X_fold_train = fold_ext.fit_transform(train_texts)
        X_fold_val = fold_ext.transform(val_texts)

        fold_clf = MisinformationClassifier(
            class_weight="balanced",
            calibrate=False,
            random_state=random_state,
        )
        fold_clf.fit(X_fold_train, train_y)

        oof_preds[val_idx] = fold_clf.predict(X_fold_val)
        oof_probs[val_idx] = fold_clf.predict_proba(X_fold_val)

    acc = float(accuracy_score(y_all, oof_preds))
    prec_macro = float(precision_score(y_all, oof_preds, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_all, oof_preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_all, oof_preds, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_all, oof_preds, average="weighted", zero_division=0))
    cm = confusion_matrix(y_all, oof_preds, labels=unique_classes).tolist()

    report_dict = classification_report(
        y_all,
        oof_preds,
        labels=unique_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    roc_auc: Optional[float] = None
    if len(unique_classes) == 2 and 1 in unique_classes:
        try:
            real_idx = int(np.where(unique_classes == 1)[0][0])
            roc_auc = float(roc_auc_score(y_all, oof_probs[:, real_idx]))
        except Exception as e:
            logger.warning(f"Could not compute out-of-fold binary ROC-AUC: {e}")
    elif len(unique_classes) > 2:
        try:
            roc_auc = float(roc_auc_score(y_all, oof_probs, multi_class="ovr", labels=unique_classes))
        except Exception as e:
            logger.warning(f"Could not compute out-of-fold multi-class ROC-AUC: {e}")

    return {
        "evaluation_method": "stratified_cross_validation",
        "evaluation_successful": True,
        "number_of_samples": len(statements),
        "number_of_folds": actual_folds,
        "class_distribution": class_dist,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classes": unique_classes.tolist(),
        "roc_auc_calculated": roc_auc is not None,
        "roc_auc": roc_auc,
        "is_development_fixture": is_development_fixture,
        "classification_report": report_dict,
        "note": (
            "Metrics derived from out-of-fold stratified cross-validation. "
            "Embedded fixture results are development diagnostics, not official LIAR benchmark evaluation."
        ),
    }


def run_misinformation_pipeline(
    reddit_path: Optional[str | Path] = None,
    youtube_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    dataset_path: Optional[str | Path] = None,
    classifier: Optional[MisinformationClassifier] = None,
    feature_extractor: Optional[LinguisticFeatureExtractor] = None,
    detector: Optional[MisinformationDetector] = None,
    save_model_to: Optional[str | Path] = None,
    binary: bool = True,
    uncertain_policy: str = "exclude",
) -> Dict[str, Any]:
    """Executes the end-to-end Stage 9 Misinformation Detection Pipeline.

    Enforces:
    - 4-way source classification: 'loaded', 'empty', 'missing', 'corrupted'.
    - Status semantics: 'success', 'partial', 'empty', 'error'.
    - Transparent multi-signal evidence-based risk assessment (never penalizing opinions/sentiment).
    - Preserving 100% of input records non-destructively.
    - Schema-v2 validation and atomic artifact promotion.

    Args:
        reddit_path: Path to processed Reddit JSON.
        youtube_path: Path to processed YouTube JSON.
        output_path: Canonical destination path for misinformation results JSON.
        dataset_path: Optional path to external LIAR benchmark dataset.
        classifier: Optional injected MisinformationClassifier.
        feature_extractor: Optional injected LinguisticFeatureExtractor.
        detector: Optional injected MisinformationDetector.
        save_model_to: Optional destination to persist fitted model via joblib.
        binary: If True, uses binary targets ('fake' vs 'real'); if False, uses 6 fine-grained classes.
        uncertain_policy: Policy for handling 'half-true' labels in binary classification.

    Returns:
        Structured Stage 9 execution report and analytical results payload.
    """
    in_reddit = Path(reddit_path) if reddit_path else DEFAULT_REDDIT_INPUT
    in_youtube = Path(youtube_path) if youtube_path else DEFAULT_YOUTUBE_INPUT
    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    now_iso = datetime.now(timezone.utc).isoformat()
    logger.info("Starting Stage 9 Misinformation Detection Pipeline...")

    # 1. Source loading and 4-way classification
    reddit_status, reddit_data, r_count = load_and_classify_source(in_reddit, "reddit")
    youtube_status, youtube_data, y_count = load_and_classify_source(in_youtube, "youtube")

    provenance = {
        "reddit_input_file": str(in_reddit),
        "reddit_status": reddit_status,
        "reddit_records_count": r_count,
        "youtube_input_file": str(in_youtube),
        "youtube_status": youtube_status,
        "youtube_records_count": y_count,
    }

    # 2. Determine Stage Status
    if reddit_status in {"missing", "corrupted"} and youtube_status in {"missing", "corrupted"}:
        stage_status = "error"
    elif reddit_status == "empty" and youtube_status == "empty":
        stage_status = "empty"
    elif (reddit_status in {"missing", "corrupted"} and youtube_status in {"loaded", "empty"}) or (
        youtube_status in {"missing", "corrupted"} and reddit_status in {"loaded", "empty"}
    ):
        stage_status = "partial"
    else:
        stage_status = "success"

    # Fail closed on fatal diagnostic error
    if stage_status == "error":
        logger.error("Both input sources are missing or corrupted. Stage 9 produced diagnostic error artifact.")
        return {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "status": "error",
            "generated_at": now_iso,
            "error": "Both input sources are missing or corrupted; cannot perform misinformation analysis.",
            "provenance": provenance,
            "methodology_disclosure": METHODOLOGY_DISCLOSURE,
            "total_records_processed": 0,
            "total_analyzed": 0,
            "average_credibility_score": 0.0,
            "flagged_high_risk_count": 0,
            "risk_tier_distribution": {"low": 0, "moderate": 0, "high_risk": 0, "unknown": 0},
            "veracity_distribution": {},
            "platform_breakdown": {
                "reddit": {"total_analyzed": 0, "average_credibility_score": 0.0, "flagged_high_risk_count": 0},
                "youtube": {"total_analyzed": 0, "average_credibility_score": 0.0, "flagged_high_risk_count": 0},
            },
            "predictions": [],
            "reddit_records": [],
            "youtube_records": [],
            "benchmark_evaluation": None,
        }

    # 3. Model Preparation (fit if not injected)
    model_eval_metrics: Optional[Dict[str, Any]] = None
    if classifier is None or feature_extractor is None:
        logger.info("Fitting misinformation classifier from benchmark dataset...")
        loader = LiarDatasetLoader(data_path=dataset_path)
        statements, labels, _ = loader.prepare_data(binary=binary, uncertain_policy=uncertain_policy)
        is_dev = dataset_path is None or not Path(dataset_path).exists()
        target_names = loader.get_binary_labels() if binary else loader.get_fine_grained_labels()

        model_eval_metrics = evaluate_cross_validated(
            statements=statements,
            labels=labels,
            target_names=target_names,
            binary=binary,
            n_splits=3,
            is_development_fixture=is_dev,
        )

        feature_extractor = LinguisticFeatureExtractor(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            use_stylometrics=True,
        )
        X_train = feature_extractor.fit_transform(statements)

        classifier = MisinformationClassifier(class_weight="balanced", calibrate=True, calibration_cv=3)
        classifier.fit(X_train, labels)

        if save_model_to:
            classifier.save_model(save_model_to)
            logger.info("Classifier saved to %s", save_model_to)

    assessor = detector or MisinformationDetector()

    # 4. Extract candidates preserving 100% of records
    candidates: List[Dict[str, Any]] = []

    for post in reddit_data:
        text, is_empty = _extract_text_for_record(post, "reddit")
        candidates.append({
            "id": str(post.get("post_id")),
            "platform": "reddit",
            "text": text,
            "original_record": post,
            "is_empty": is_empty,
        })

    for vid in youtube_data:
        text, is_empty = _extract_text_for_record(vid, "youtube")
        candidates.append({
            "id": str(vid.get("video_id")),
            "platform": "youtube",
            "text": text,
            "original_record": vid,
            "is_empty": is_empty,
        })

    # Batch feature extraction and model prediction
    valid_indices = [idx for idx, c in enumerate(candidates) if not c["is_empty"]]
    valid_texts = [candidates[idx]["text"] for idx in valid_indices]

    batch_preds: Dict[int, Any] = {}
    batch_scores: Dict[int, float] = {}
    batch_styles: Dict[int, Dict[str, float]] = {}
    batch_errors: Dict[int, str] = {}

    if valid_texts:
        try:
            X_cand = feature_extractor.transform(valid_texts)
            y_preds = classifier.predict(X_cand)
            cred_scores = classifier.get_credibility_score(X_cand)

            for local_idx, global_idx in enumerate(valid_indices):
                score = float(np.clip(cred_scores[local_idx], 0.0, 1.0))
                batch_preds[global_idx] = y_preds[local_idx]
                batch_scores[global_idx] = score

                style_arr = feature_extractor.extract_single_stylometrics(valid_texts[local_idx])
                batch_styles[global_idx] = {
                    name: round(float(val), 4)
                    for name, val in zip(STYLOMETRIC_FEATURE_NAMES, style_arr)
                }
        except Exception as exc:
            logger.error("Misinformation batch inference failed: %s", exc)
            for global_idx in valid_indices:
                batch_errors[global_idx] = str(exc)

    # 5. Perform Evidence-Based Assessments
    predictions: List[Dict[str, Any]] = []
    enriched_reddit: List[Dict[str, Any]] = []
    enriched_youtube: List[Dict[str, Any]] = []

    risk_tier_counts = {"low": 0, "moderate": 0, "high_risk": 0, "unknown": 0}
    veracity_counts: Dict[str, int] = {}
    platform_stats: Dict[str, Dict[str, Any]] = {
        "reddit": {"total": 0, "credibility_sum": 0.0, "high_risk_count": 0},
        "youtube": {"total": 0, "credibility_sum": 0.0, "high_risk_count": 0},
    }
    analyzed_credibility_scores: List[float] = []

    for idx, cand in enumerate(candidates):
        orig_rec = dict(cand["original_record"])
        plat = cand["platform"]
        native_id = cand["id"]

        if cand["is_empty"]:
            assessment = assessor.assess_record(
                text=None,
                platform=plat,
                native_id=native_id,
                binary=binary,
                stylometrics=None,
            )
        elif idx in batch_errors:
            assessment = {
                "id": native_id,
                "platform": plat,
                "veracity_label": "undetermined",
                "credibility_score": 0.0,
                "risk_tier": "unknown",
                "status": "error",
                "evidence_basis": "insufficient_evidence",
                "signals_detected": ["inference_execution_error"],
                "assessment_rationale": f"Batch inference failed: {batch_errors[idx]}",
                "limitations": METHODOLOGICAL_LIMITATIONS_DISCLOSURE,
                "stylometric_signals": {name: 0.0 for name in STYLOMETRIC_FEATURE_NAMES},
            }
        else:
            assessment = assessor.assess_record(
                text=cand["text"],
                platform=plat,
                native_id=native_id,
                classifier_score=batch_scores.get(idx),
                classifier_prediction=batch_preds.get(idx),
                binary=binary,
                stylometrics=batch_styles.get(idx),
            )

        status_val = assessment["status"]
        veracity_val = assessment["veracity_label"]
        cred_score = assessment["credibility_score"]
        tier_val = assessment["risk_tier"]
        style_dict = assessment["stylometric_signals"]

        risk_tier_counts[tier_val] = risk_tier_counts.get(tier_val, 0) + 1
        veracity_counts[veracity_val] = veracity_counts.get(veracity_val, 0) + 1

        if status_val == "success":
            analyzed_credibility_scores.append(cred_score)
            if plat in platform_stats:
                platform_stats[plat]["total"] += 1
                platform_stats[plat]["credibility_sum"] += cred_score
                if tier_val == "high_risk":
                    platform_stats[plat]["high_risk_count"] += 1

        # Summary prediction entry
        pred_entry = dict(assessment)
        pred_entry["text_preview"] = cand["text"][:120] + ("..." if len(cand["text"]) > 120 else "")
        predictions.append(pred_entry)

        # Enriched output preserving all original fields
        enriched_item = dict(orig_rec)
        enriched_item["veracity_label"] = veracity_val
        enriched_item["credibility_score"] = cred_score
        enriched_item["risk_tier"] = tier_val
        enriched_item["misinformation_status"] = status_val
        enriched_item["status"] = status_val
        enriched_item["analyzed_at"] = now_iso
        enriched_item["evidence_basis"] = assessment["evidence_basis"]
        enriched_item["signals_detected"] = assessment["signals_detected"]
        enriched_item["assessment_rationale"] = assessment["assessment_rationale"]
        enriched_item["limitations"] = assessment["limitations"]
        enriched_item["stylometric_signals"] = style_dict

        if plat == "reddit":
            enriched_reddit.append(enriched_item)
        elif plat == "youtube":
            enriched_youtube.append(enriched_item)

    total_records = len(candidates)
    total_analyzed = len(analyzed_credibility_scores)
    avg_credibility = (
        round(float(np.mean(analyzed_credibility_scores)), 4)
        if analyzed_credibility_scores
        else 0.0
    )

    platform_breakdown: Dict[str, Any] = {}
    for p_name, p_data in platform_stats.items():
        count = p_data["total"]
        platform_breakdown[p_name] = {
            "total_analyzed": count,
            "average_credibility_score": round(p_data["credibility_sum"] / max(1, count), 4),
            "flagged_high_risk_count": p_data["high_risk_count"],
        }

    # 7. Assemble Structured Artifact Payload
    payload: Dict[str, Any] = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "status": stage_status,
        "generated_at": now_iso,
        "provenance": provenance,
        "methodology_disclosure": METHODOLOGY_DISCLOSURE,
        "total_records_processed": total_records,
        "total_analyzed": total_analyzed,
        "average_credibility_score": avg_credibility,
        "flagged_high_risk_count": risk_tier_counts.get("high_risk", 0),
        "risk_tier_distribution": risk_tier_counts,
        "veracity_distribution": veracity_counts,
        "platform_breakdown": platform_breakdown,
        "predictions": predictions,
        "reddit_records": enriched_reddit,
        "youtube_records": enriched_youtube,
        "benchmark_evaluation": model_eval_metrics,
    }

    # 8. Pre-promotion Validation & Atomic Promotion
    is_valid, validation_errors = MisinformationArtifactValidator.validate_misinformation_artifact(payload)
    if not is_valid:
        logger.error("Pre-promotion artifact validation failed (%d errors): %s", len(validation_errors), validation_errors)
        raise ValueError(f"Misinformation artifact validation failed: {validation_errors}")

    _promote_artifact_atomically(payload, out_file)

    logger.info(
        "Stage 9 Misinformation Pipeline completed successfully (%s status). "
        "Processed %d records (%d analyzed). Output: %s",
        stage_status,
        total_records,
        total_analyzed,
        out_file,
    )

    return payload

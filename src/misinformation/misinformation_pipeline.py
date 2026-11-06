"""Misinformation Detection Pipeline Orchestrator.

Orchestrates out-of-sample evaluated benchmark training on LIAR datasets,
feature extraction, probability estimation, and non-destructive inference on social
media posts from Reddit and YouTube.

Statistical & Benchmark Scope:
- The embedded dataset fixture contains 24 statements designed for deterministic,
  offline unit testing. It is NOT the official 12,836-statement LIAR benchmark.
- Evaluation metrics in `benchmark_evaluation` are computed via out-of-fold stratified
  cross-validation (fitting feature extraction and models independently per fold)
  to strictly prevent data leakage and resubstitution bias.
- Credibility scores represent statistical veracity likelihoods based on text n-grams
  and stylometry; they DO NOT constitute independent ground-truth factual verification.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.misinformation.feature_extractor import (
    STYLOMETRIC_FEATURE_NAMES,
    LinguisticFeatureExtractor,
)

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_INPUT = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_INPUT = Path("data/processed/youtube_videos_processed.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/misinformation_analysis_results.json")


def _determine_risk_tier(credibility_score: float) -> str:
    """Categorize credibility score into a risk tier."""
    if credibility_score < 0.40:
        return "high_risk"
    if credibility_score < 0.70:
        return "moderate"
    return "low"


def _load_json_records(file_path: Path) -> List[Dict[str, Any]]:
    """Safely load JSON records from disk without modifying source files."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []
    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("posts", data.get("videos", data.get("records", [])))
            return []
    except Exception as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        return []


def _extract_social_records(
    reddit_data: List[Dict[str, Any]], youtube_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Normalize Reddit posts and YouTube videos into unified candidate records.

    Preserves 100% of input records, marking empty or whitespace text records
    with `is_empty = True` rather than discarding them.
    """
    normalized: List[Dict[str, Any]] = []

    # Process Reddit records
    for idx, post in enumerate(reddit_data):
        if not isinstance(post, dict):
            continue
        post_id = str(post.get("post_id", post.get("id", f"reddit_{idx}")))
        text = (
            post.get("text_cleaned")
            or post.get("title_cleaned")
            or f"{post.get('title', '')} {post.get('text', '')}".strip()
        )
        is_empty = text is None or not str(text).strip()
        normalized.append({
            "id": post_id,
            "platform": "reddit",
            "text": str(text).strip() if not is_empty else "",
            "author": post.get("author", "unknown"),
            "original_record": post,
            "is_empty": is_empty,
        })

    # Process YouTube records
    for idx, vid in enumerate(youtube_data):
        if not isinstance(vid, dict):
            continue
        video_id = str(vid.get("video_id", vid.get("id", f"youtube_{idx}")))
        text = (
            vid.get("title_cleaned")
            or vid.get("description_cleaned")
            or f"{vid.get('title', '')} {vid.get('description', '')}".strip()
        )
        is_empty = text is None or not str(text).strip()
        normalized.append({
            "id": video_id,
            "platform": "youtube",
            "text": str(text).strip() if not is_empty else "",
            "author": vid.get("channel_title", "unknown"),
            "original_record": vid,
            "is_empty": is_empty,
        })

    return normalized


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

        # Fold-isolated feature extractor
        fold_ext = LinguisticFeatureExtractor(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            use_stylometrics=True,
        )
        X_fold_train = fold_ext.fit_transform(train_texts)
        X_fold_val = fold_ext.transform(val_texts)

        # Fold-isolated classifier (fallback to base model if samples < 5)
        fold_clf = MisinformationClassifier(
            class_weight="balanced",
            calibrate=False,  # Uncalibrated for fold splits to prevent secondary CV instability
            random_state=random_state,
        )
        fold_clf.fit(X_fold_train, train_y)

        oof_preds[val_idx] = fold_clf.predict(X_fold_val)
        oof_probs[val_idx] = fold_clf.predict_proba(X_fold_val)

    # Compute metrics from pooled out-of-fold predictions
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
    save_model_to: Optional[str | Path] = None,
    binary: bool = True,
    uncertain_policy: str = "exclude",
) -> Dict[str, Any]:
    """Execute end-to-end misinformation detection pipeline.

    Preserves 100% of input records, enriches them non-destructively, and
    records diagnostic out-of-fold benchmark metrics.
    """
    in_reddit = Path(reddit_path) if reddit_path else DEFAULT_REDDIT_INPUT
    in_youtube = Path(youtube_path) if youtube_path else DEFAULT_YOUTUBE_INPUT
    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    logger.info("Starting misinformation pipeline...")
    reddit_posts = _load_json_records(in_reddit)
    youtube_videos = _load_json_records(in_youtube)
    candidates = _extract_social_records(reddit_posts, youtube_videos)

    now_iso = datetime.now(timezone.utc).isoformat()
    model_eval_metrics: Optional[Dict[str, Any]] = None

    # Train production model and compute leak-free cross-validated metrics if not injected
    if classifier is None or feature_extractor is None:
        logger.info("Preparing misinformation classifier from LIAR dataset...")
        loader = LiarDatasetLoader(data_path=dataset_path)
        statements, labels, _ = loader.prepare_data(
            binary=binary, uncertain_policy=uncertain_policy
        )

        is_dev = dataset_path is None or not Path(dataset_path).exists()
        target_names = loader.get_binary_labels() if binary else loader.get_fine_grained_labels()

        # 1. Honest out-of-sample benchmark evaluation
        model_eval_metrics = evaluate_cross_validated(
            statements=statements,
            labels=labels,
            target_names=target_names,
            binary=binary,
            n_splits=3,
            is_development_fixture=is_dev,
        )

        # 2. Fit final production model on complete training dataset
        feature_extractor = LinguisticFeatureExtractor(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True,
            use_stylometrics=True,
        )
        X_train = feature_extractor.fit_transform(statements)

        classifier = MisinformationClassifier(
            class_weight="balanced",
            calibrate=True,
            calibration_cv=3,
        )
        classifier.fit(X_train, labels)

        if save_model_to:
            classifier.save_model(save_model_to)
            logger.info(f"Classifier saved to {save_model_to}")

    predictions: List[Dict[str, Any]] = []
    enriched_reddit: List[Dict[str, Any]] = []
    enriched_youtube: List[Dict[str, Any]] = []

    risk_tier_counts = {"low": 0, "moderate": 0, "high_risk": 0, "unknown": 0}
    veracity_counts: Dict[str, int] = {}
    platform_stats: Dict[str, Dict[str, Any]] = {
        "reddit": {"total": 0, "credibility_sum": 0.0, "high_risk_count": 0},
        "youtube": {"total": 0, "credibility_sum": 0.0, "high_risk_count": 0},
    }

    label_map = ID_TO_BINARY_LABEL if binary else ID_TO_FINE_GRAINED_LABEL

    # Separate valid non-empty candidates for batch inference
    valid_indices: List[int] = []
    valid_texts: List[str] = []

    for idx, cand in enumerate(candidates):
        if not cand["is_empty"]:
            valid_indices.append(idx)
            valid_texts.append(cand["text"])

    batch_preds: Dict[int, int] = {}
    batch_scores: Dict[int, float] = {}
    batch_styles: Dict[int, Dict[str, float]] = {}
    batch_errors: Dict[int, str] = {}

    # Fault-contained batch inference
    if valid_texts:
        try:
            X_cand = feature_extractor.transform(valid_texts)
            y_preds = classifier.predict(X_cand)
            cred_scores = classifier.get_credibility_score(X_cand)

            for local_idx, global_idx in enumerate(valid_indices):
                score = float(np.clip(cred_scores[local_idx], 0.0, 1.0))
                batch_preds[global_idx] = int(y_preds[local_idx])
                batch_scores[global_idx] = score

                style_arr = feature_extractor.extract_single_stylometrics(valid_texts[local_idx])
                batch_styles[global_idx] = {
                    name: round(float(val), 4)
                    for name, val in zip(STYLOMETRIC_FEATURE_NAMES, style_arr)
                }
        except Exception as exc:
            logger.error(f"Misinformation batch inference failed: {exc}")
            for global_idx in valid_indices:
                batch_errors[global_idx] = "Inference execution error."

    # Assemble enriched outputs preserving 100% of input records
    analyzed_credibility_scores: List[float] = []

    for idx, cand in enumerate(candidates):
        orig_rec = dict(cand["original_record"])
        plat = cand["platform"]

        if cand["is_empty"]:
            veracity_label = "undetermined"
            score = 0.0
            risk_tier = "unknown"
            status = "empty_text"
            style_dict = {name: 0.0 for name in STYLOMETRIC_FEATURE_NAMES}
            risk_tier_counts["unknown"] += 1
        elif idx in batch_errors:
            veracity_label = "undetermined"
            score = 0.0
            risk_tier = "unknown"
            status = "error"
            style_dict = {name: 0.0 for name in STYLOMETRIC_FEATURE_NAMES}
            risk_tier_counts["unknown"] += 1
        else:
            raw_pred = batch_preds[idx]
            veracity_label = label_map.get(raw_pred, str(raw_pred))
            score = round(batch_scores[idx], 4)
            risk_tier = _determine_risk_tier(score)
            status = "success"
            style_dict = batch_styles[idx]

            risk_tier_counts[risk_tier] = risk_tier_counts.get(risk_tier, 0) + 1
            veracity_counts[veracity_label] = veracity_counts.get(veracity_label, 0) + 1
            analyzed_credibility_scores.append(score)

            if plat in platform_stats:
                platform_stats[plat]["total"] += 1
                platform_stats[plat]["credibility_sum"] += score
                if risk_tier == "high_risk":
                    platform_stats[plat]["high_risk_count"] += 1

        # Summary prediction entry (backward compatibility)
        pred_entry = {
            "id": cand["id"],
            "platform": plat,
            "text_preview": cand["text"][:120] + ("..." if len(cand["text"]) > 120 else ""),
            "veracity_label": veracity_label,
            "credibility_score": score,
            "risk_tier": risk_tier,
            "stylometric_signals": style_dict,
            "status": status,
        }
        predictions.append(pred_entry)

        # Enriched record preserving full original schema
        enriched_item = dict(orig_rec)
        enriched_item["veracity_label"] = veracity_label
        enriched_item["credibility_score"] = score
        enriched_item["risk_tier"] = risk_tier
        enriched_item["misinformation_status"] = status
        enriched_item["analyzed_at"] = now_iso
        enriched_item["stylometric_signals"] = style_dict

        if plat == "reddit":
            enriched_reddit.append(enriched_item)
        elif plat == "youtube":
            enriched_youtube.append(enriched_item)

    total_records = len(candidates)
    total_analyzed = len(analyzed_credibility_scores)
    avg_credibility = (
        float(np.mean(analyzed_credibility_scores))
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

    payload: Dict[str, Any] = {
        "status": "success",
        "generated_at": now_iso,
        "total_records_processed": total_records,
        "total_analyzed": total_analyzed,
        "average_credibility_score": round(avg_credibility, 4),
        "flagged_high_risk_count": risk_tier_counts.get("high_risk", 0),
        "risk_tier_distribution": risk_tier_counts,
        "veracity_distribution": veracity_counts,
        "platform_breakdown": platform_breakdown,
        "predictions": predictions,
        "reddit_records": enriched_reddit,
        "youtube_records": enriched_youtube,
        "benchmark_evaluation": model_eval_metrics,
    }

    # Safe, non-destructive file writing
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, mode="w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Misinformation pipeline completed. Processed {total_records} records "
        f"({total_analyzed} analyzed). Output: {out_file}"
    )

    return payload

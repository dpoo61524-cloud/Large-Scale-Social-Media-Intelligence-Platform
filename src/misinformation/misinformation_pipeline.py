"""Misinformation Detection Pipeline Orchestrator.

Orchestrates training on LIAR benchmark datasets, feature extraction,
calibrated probability estimation, and non-destructive inference on social
media posts from Reddit and YouTube.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    """Safely load JSON records from disk."""
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
    """Normalize Reddit posts and YouTube videos into unified candidate records."""
    normalized: List[Dict[str, Any]] = []

    # Process Reddit
    for idx, post in enumerate(reddit_data):
        if not isinstance(post, dict):
            continue
        post_id = str(post.get("post_id", post.get("id", f"reddit_{idx}")))
        text = (
            post.get("text_cleaned")
            or post.get("title_cleaned")
            or f"{post.get('title', '')} {post.get('text', '')}".strip()
        )
        if not text:
            continue
        normalized.append({
            "id": post_id,
            "platform": "reddit",
            "text": str(text),
            "author": post.get("author", "unknown"),
            "original_record": post,
        })

    # Process YouTube
    for idx, vid in enumerate(youtube_data):
        if not isinstance(vid, dict):
            continue
        video_id = str(vid.get("video_id", vid.get("id", f"youtube_{idx}")))
        text = (
            vid.get("title_cleaned")
            or vid.get("description_cleaned")
            or f"{vid.get('title', '')} {vid.get('description', '')}".strip()
        )
        if not text:
            continue
        normalized.append({
            "id": video_id,
            "platform": "youtube",
            "text": str(text),
            "author": vid.get("channel_title", "unknown"),
            "original_record": vid,
        })

    return normalized


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

    Parameters
    ----------
    reddit_path : Optional[str | Path]
        Path to processed Reddit posts JSON.
    youtube_path : Optional[str | Path]
        Path to processed YouTube videos JSON.
    output_path : Optional[str | Path]
        Destination for JSON analysis results.
    dataset_path : Optional[str | Path]
        Optional path to external LIAR dataset. If None, uses embedded benchmark sample.
    classifier : Optional[MisinformationClassifier]
        Optional pre-trained classifier instance (for testing/injection).
    feature_extractor : Optional[LinguisticFeatureExtractor]
        Optional pre-fitted feature extractor instance.
    save_model_to : Optional[str | Path]
        Optional path to persist trained classifier via joblib.
    binary : bool
        Whether to train and predict in binary mode ('fake' vs 'real').
    uncertain_policy : str
        Policy for 'half-true' in binary mode ('exclude', 'fake', 'real').

    Returns
    -------
    Dict[str, Any]
        Complete analysis payload.
    """
    in_reddit = Path(reddit_path) if reddit_path else DEFAULT_REDDIT_INPUT
    in_youtube = Path(youtube_path) if youtube_path else DEFAULT_YOUTUBE_INPUT
    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    logger.info("Starting misinformation pipeline...")
    reddit_posts = _load_json_records(in_reddit)
    youtube_videos = _load_json_records(in_youtube)
    candidates = _extract_social_records(reddit_posts, youtube_videos)

    # Prepare model and feature extractor if not injected
    model_eval_metrics: Optional[Dict[str, Any]] = None
    if classifier is None or feature_extractor is None:
        logger.info("Training misinformation classifier on LIAR benchmark dataset...")
        loader = LiarDatasetLoader(data_path=dataset_path)
        statements, labels, meta = loader.prepare_data(
            binary=binary, uncertain_policy=uncertain_policy
        )

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

        # Compute internal benchmark training metrics
        target_names = loader.get_binary_labels() if binary else loader.get_fine_grained_labels()
        model_eval_metrics = classifier.evaluate(X_train, labels, target_names=target_names)

        if save_model_to:
            classifier.save_model(save_model_to)
            logger.info(f"Classifier saved to {save_model_to}")

    predictions: List[Dict[str, Any]] = []
    risk_tier_counts = {"low": 0, "moderate": 0, "high_risk": 0}
    veracity_counts: Dict[str, int] = {}
    platform_stats: Dict[str, Dict[str, Any]] = {
        "reddit": {"total": 0, "credibility_sum": 0.0, "high_risk_count": 0},
        "youtube": {"total": 0, "credibility_sum": 0.0, "high_risk_count": 0},
    }

    if candidates:
        texts = [c["text"] for c in candidates]
        X_cand = feature_extractor.transform(texts)
        y_preds = classifier.predict(X_cand)
        cred_scores = classifier.get_credibility_score(X_cand)

        label_map = ID_TO_BINARY_LABEL if binary else ID_TO_FINE_GRAINED_LABEL

        for idx, cand in enumerate(candidates):
            raw_pred = int(y_preds[idx])
            veracity_label = label_map.get(raw_pred, str(raw_pred))
            score = float(np.clip(cred_scores[idx], 0.0, 1.0))
            risk_tier = _determine_risk_tier(score)

            # Update stats
            risk_tier_counts[risk_tier] = risk_tier_counts.get(risk_tier, 0) + 1
            veracity_counts[veracity_label] = veracity_counts.get(veracity_label, 0) + 1

            plat = cand["platform"]
            if plat in platform_stats:
                platform_stats[plat]["total"] += 1
                platform_stats[plat]["credibility_sum"] += score
                if risk_tier == "high_risk":
                    platform_stats[plat]["high_risk_count"] += 1

            # Extract sample stylometric preview
            single_style = feature_extractor.extract_single_stylometrics(cand["text"])
            style_dict = {
                name: round(float(val), 4)
                for name, val in zip(STYLOMETRIC_FEATURE_NAMES, single_style)
            }

            predictions.append({
                "id": cand["id"],
                "platform": cand["platform"],
                "text_preview": cand["text"][:120] + ("..." if len(cand["text"]) > 120 else ""),
                "veracity_label": veracity_label,
                "credibility_score": round(score, 4),
                "risk_tier": risk_tier,
                "stylometric_signals": style_dict,
            })

    total_analyzed = len(predictions)
    avg_credibility = (
        float(np.mean([p["credibility_score"] for p in predictions]))
        if predictions
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_analyzed": total_analyzed,
        "average_credibility_score": round(avg_credibility, 4),
        "flagged_high_risk_count": risk_tier_counts.get("high_risk", 0),
        "risk_tier_distribution": risk_tier_counts,
        "veracity_distribution": veracity_counts,
        "platform_breakdown": platform_breakdown,
        "predictions": predictions,
        "benchmark_evaluation": model_eval_metrics,
    }

    # Non-destructive file output writing
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, mode="w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Misinformation analysis complete. Analyzed {total_analyzed} items, "
        f"flagged {risk_tier_counts.get('high_risk', 0)} high-risk. Output: {out_file}"
    )

    return payload

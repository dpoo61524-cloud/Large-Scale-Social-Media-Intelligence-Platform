"""Advanced Topic Modeling Pipeline (Company Step 4).

Orchestrates dynamic BERTopic theme discovery, lexical shifts, topic similarity matrix,
temporal trajectory dynamics (velocity, acceleration, breakout status), and cross-platform
divergence (Reddit vs. YouTube) over processed records.
Persists structured results non-destructively to data/processed/topic_modeling_results.json.
"""

from datetime import datetime, timezone
import json
import logging
import numbers
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as _np
    _NUMPY_BOOL_TYPES: tuple[type, ...] = (_np.bool_,)
except ImportError:
    _NUMPY_BOOL_TYPES = ()

from src.topics.topic_modeler import (
    DEFAULT_EMBEDDING_MODEL,
    OUTLIER_TOPIC_ID,
    SocialTopicModeler,
)
from src.topics.topic_temporal_tracker import TopicTemporalTracker

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_INPUT = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_INPUT = Path("data/processed/youtube_videos_processed.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/topic_modeling_results.json")

TOPIC_STATUS_ASSIGNED = "assigned"
TOPIC_STATUS_OUTLIER = "outlier"
TOPIC_STATUS_ERROR = "error"


def _is_boolean(val: Any) -> bool:
    """Explicitly checks whether val is a Python bool or NumPy bool_."""
    if isinstance(val, bool):
        return True
    if _NUMPY_BOOL_TYPES and isinstance(val, _NUMPY_BOOL_TYPES):
        return True
    if type(val).__name__ == "bool_":
        return True
    return False


def normalize_topic_id(topic_id: Any) -> Any:
    """Normalizes valid integral topic IDs to native Python int.

    Preserves invalid values as invalid without converting floats, strings, or booleans to int,
    while ensuring non-serializable objects (like numpy.bool_) do not crash JSON persistence.

    Args:
        topic_id: Raw topic identifier from model.

    Returns:
        Native Python int for valid integral IDs, or preserved invalid value.
    """
    if _is_boolean(topic_id):
        return bool(topic_id)
    if isinstance(topic_id, (int, numbers.Integral)):
        try:
            return int(topic_id)
        except (TypeError, ValueError, OverflowError):
            return topic_id
    return topic_id


def derive_topic_status(topic_id: Any) -> str:
    """Derives deterministic topic status from topic identifier.

    Rules:
    - Non-negative integer (>= 0): 'assigned'
    - Outlier identifier (-1): 'outlier'
    - Any other value (negative < -1, non-int, bool, float, str, None): 'error'

    Supports both native Python integers and NumPy integer scalar types.
    Explicitly rejects Python booleans and NumPy bool_ values.

    Args:
        topic_id: Topic identifier to classify.

    Returns:
        One of 'assigned', 'outlier', or 'error'.
    """
    if _is_boolean(topic_id):
        return TOPIC_STATUS_ERROR
    if not isinstance(topic_id, (int, numbers.Integral)):
        return TOPIC_STATUS_ERROR

    try:
        val = int(topic_id)
    except (TypeError, ValueError, OverflowError):
        return TOPIC_STATUS_ERROR

    if val >= 0:
        return TOPIC_STATUS_ASSIGNED
    if val == OUTLIER_TOPIC_ID:
        return TOPIC_STATUS_OUTLIER
    return TOPIC_STATUS_ERROR


def extract_document_text(record: dict[str, Any], is_youtube: bool = False) -> str:
    """Extracts cleaned text from a processed record.

    Prioritizes title_cleaned + text_cleaned (Reddit) or description_cleaned (YouTube).
    """
    title = str(record.get("title_cleaned") or record.get("title") or "").strip()
    if is_youtube:
        body = str(record.get("description_cleaned") or record.get("description") or "").strip()
    else:
        body = str(record.get("text_cleaned") or record.get("text") or "").strip()
    return f"{title} {body}".strip()


def run_topic_pipeline(
    reddit_input_path: Optional[Path | str] = None,
    youtube_input_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    modeler: Optional[SocialTopicModeler] = None,
    window: str = "daily",
) -> dict[str, Any]:
    """Runs the advanced topic discovery and dynamic temporal tracking pipeline.

    Args:
        reddit_input_path: Path to Reddit processed JSON.
        youtube_input_path: Path to YouTube processed JSON.
        output_path: Destination path for topic modeling JSON results.
        modeler: Optional SocialTopicModeler instance (supports mock injection).
        window: Primary temporal aggregation window ('daily', 'weekly').

    Returns:
        Structured dictionary containing topic discovery results, temporal dynamics,
        cross-platform divergence, similarity matrix, and enriched records.
    """
    reddit_path = Path(reddit_input_path) if reddit_input_path else DEFAULT_REDDIT_INPUT
    youtube_path = Path(youtube_input_path) if youtube_input_path else DEFAULT_YOUTUBE_INPUT
    dest_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    if modeler is None:
        modeler = SocialTopicModeler()

    # Load Reddit processed records
    reddit_records: list[dict[str, Any]] = []
    if reddit_path.exists():
        try:
            with open(reddit_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    reddit_records = loaded
        except Exception as exc:
            logger.error("Failed to read Reddit input %s: %s", reddit_path, exc)

    # Load YouTube processed records
    youtube_records: list[dict[str, Any]] = []
    if youtube_path.exists():
        try:
            with open(youtube_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    youtube_records = loaded
        except Exception as exc:
            logger.error("Failed to read YouTube input %s: %s", youtube_path, exc)

    now_iso = datetime.now(timezone.utc).isoformat()
    total_records = len(reddit_records) + len(youtube_records)

    if total_records == 0:
        logger.warning("No records found to perform topic modeling.")
        empty_payload: dict[str, Any] = {
            "status": "success",
            "generated_at": now_iso,
            "embedding_model": getattr(modeler, "embedding_model_name", DEFAULT_EMBEDDING_MODEL),
            "total_documents_modeled": 0,
            "execution_mode": getattr(modeler, "execution_mode", "neural_bertopic"),
            "algorithm": getattr(modeler, "algorithm", "bertopic"),
            "fallback_reason": getattr(modeler, "fallback_reason", None),
            "corpus_size": 0,
            "valid_document_count": 0,
            "topic_count": 0,
            "model_name": getattr(modeler, "embedding_model_name", DEFAULT_EMBEDDING_MODEL),
            "model_version": getattr(modeler, "model_version", "0.17.4"),
            "execution_status": "success",
            "topic_info": [],
            "similarity_matrix": {"topic_ids": [], "matrix": [], "pairwise_similarities": {}},
            "trajectories": {},
            "cross_platform_divergence": {"overall_divergence": 0.0, "topic_adoption_gaps": {}, "window_divergences": {}},
            "temporal_trends": {},
            "reddit_records": [],
            "youtube_records": [],
        }
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(empty_payload, f, indent=2, ensure_ascii=False)
        return empty_payload

    # Prepare document text corpus
    reddit_texts = [extract_document_text(r, is_youtube=False) for r in reddit_records]
    youtube_texts = [extract_document_text(r, is_youtube=True) for r in youtube_records]
    corpus = reddit_texts + youtube_texts

    # Fit topic modeler and assign topic IDs
    topics, probabilities = modeler.fit_transform(corpus)
    topic_info = modeler.get_topic_info()

    # Create keyword lookup map by topic_id and dynamic keywords
    keyword_map: dict[int, list[dict[str, Any]]] = {}
    dynamic_keyword_map: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for info in topic_info:
        tid = info["topic_id"]
        keyword_map[tid] = info.get("keywords", [])
        dynamic_keyword_map[tid] = modeler.get_dynamic_topic_keywords(tid)

    # Enrich Reddit records
    enriched_reddit: list[dict[str, Any]] = []
    num_reddit = len(reddit_records)
    for i, rec in enumerate(reddit_records):
        tid = topics[i]
        prob = probabilities[i]
        norm_tid = normalize_topic_id(tid)
        enr = dict(rec)
        enr["source_platform"] = "reddit"
        enr["topic_id"] = norm_tid
        enr["topic_probability"] = prob
        enr["topic_keywords"] = keyword_map.get(norm_tid, modeler.get_topic_keywords(norm_tid, top_n=5))
        enr["topic_status"] = derive_topic_status(norm_tid)
        enr["topic_modeled_at"] = now_iso
        enriched_reddit.append(enr)

    # Enrich YouTube records
    enriched_youtube: list[dict[str, Any]] = []
    for j, rec in enumerate(youtube_records):
        idx = num_reddit + j
        tid = topics[idx]
        prob = probabilities[idx]
        norm_tid = normalize_topic_id(tid)
        enr = dict(rec)
        enr["source_platform"] = "youtube"
        enr["topic_id"] = norm_tid
        enr["topic_probability"] = prob
        enr["topic_keywords"] = keyword_map.get(norm_tid, modeler.get_topic_keywords(norm_tid, top_n=5))
        enr["topic_status"] = derive_topic_status(norm_tid)
        enr["topic_modeled_at"] = now_iso
        enriched_youtube.append(enr)

    all_enriched = enriched_reddit + enriched_youtube

    # Prepare harmonized records for temporal tracking
    for rec in all_enriched:
        ts = rec.get("created_utc") or rec.get("published_at") or rec.get("collected_at")
        rec["_temporal_timestamp"] = ts

    # Temporal aggregation across time windows
    daily_temporal = TopicTemporalTracker.track_topic_frequency_over_time(
        all_enriched,
        topic_field="topic_id",
        time_field="_temporal_timestamp",
        window="daily",
    )
    weekly_temporal = TopicTemporalTracker.track_topic_frequency_over_time(
        all_enriched,
        topic_field="topic_id",
        time_field="_temporal_timestamp",
        window="weekly",
    )

    # Calculate advanced topic dynamics and trajectories (velocity, acceleration, status)
    primary_temporal = daily_temporal if window == "daily" else weekly_temporal
    trajectories = TopicTemporalTracker.calculate_topic_trajectories(primary_temporal)

    # Cross-platform topic divergence (Reddit vs. YouTube)
    cross_platform = TopicTemporalTracker.calculate_cross_platform_divergence(
        all_enriched,
        platform_field="source_platform",
        topic_field="topic_id",
        time_field="_temporal_timestamp",
        window=window,
    )

    # Calculate topic similarity matrix
    similarity_matrix = modeler.calculate_topic_similarity_matrix()

    # Outlier statistics
    outliers = sum(1 for t in topics if t == OUTLIER_TOPIC_ID)
    outlier_pct = round((outliers / total_records) * 100.0, 2) if total_records > 0 else 0.0

    provenance = modeler.get_provenance() if hasattr(modeler, "get_provenance") else {}

    result_payload: dict[str, Any] = {
        "status": "success",
        "generated_at": now_iso,
        "embedding_model": getattr(modeler, "embedding_model_name", DEFAULT_EMBEDDING_MODEL),
        "total_documents_modeled": total_records,
        "execution_mode": provenance.get("execution_mode", getattr(modeler, "execution_mode", "neural_bertopic")),
        "algorithm": provenance.get("algorithm", getattr(modeler, "algorithm", "bertopic")),
        "fallback_reason": provenance.get("fallback_reason", getattr(modeler, "fallback_reason", None)),
        "corpus_size": provenance.get("corpus_size", total_records),
        "valid_document_count": provenance.get("valid_document_count", len([c for c in corpus if isinstance(c, str) and c.strip()])),
        "topic_count": provenance.get("topic_count", len([t for t in topic_info if t["topic_id"] != OUTLIER_TOPIC_ID])),
        "model_name": provenance.get("model_name", getattr(modeler, "embedding_model_name", DEFAULT_EMBEDDING_MODEL)),
        "model_version": provenance.get("model_version", getattr(modeler, "model_version", "0.17.4")),
        "execution_status": provenance.get("execution_status", "success"),
        "outlier_count": outliers,
        "outlier_pct": outlier_pct,
        "topic_info": topic_info,
        "similarity_matrix": similarity_matrix,
        "trajectories": trajectories,
        "cross_platform_divergence": cross_platform,
        "temporal_trends": {
            "daily": daily_temporal,
            "weekly": weekly_temporal,
        },
        "reddit_records": enriched_reddit,
        "youtube_records": enriched_youtube,
    }

    # Clean internal temporary keys before saving
    for rec in enriched_reddit + enriched_youtube:
        rec.pop("_temporal_timestamp", None)

    # Persist to output path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    logger.info(
        "Topic pipeline completed successfully. Output persisted to %s (%d documents).",
        dest_path,
        total_records,
    )

    return result_payload
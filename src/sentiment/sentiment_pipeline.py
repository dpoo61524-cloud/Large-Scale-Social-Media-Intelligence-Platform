"""Sentiment Analysis Pipeline (Company Step 3).

Orchestrates multilingual sentiment analysis and temporal aggregation across
Reddit and YouTube processed records. Reads inputs non-destructively from data/processed/,
applies MultilingualSentimentAnalyzer and SentimentTemporalTracker, and persists enriched
results to data/processed/sentiment_analysis_results.json.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.sentiment.sentiment_analyzer import (
    DEFAULT_MODEL_NAME,
    LABEL_UNDETERMINED,
    MultilingualSentimentAnalyzer,
)
from src.sentiment.temporal_tracker import SentimentTemporalTracker

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_INPUT = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_INPUT = Path("data/processed/youtube_videos_processed.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/sentiment_analysis_results.json")


def extract_reddit_text(record: dict[str, Any]) -> str:
    """Extracts cleaned text representation from a Reddit processed record.

    Prioritizes title_cleaned + text_cleaned, falling back to raw title + text.
    """
    title = str(record.get("title_cleaned") or record.get("title") or "").strip()
    body = str(record.get("text_cleaned") or record.get("text") or "").strip()
    return f"{title} {body}".strip()


def extract_youtube_text(record: dict[str, Any]) -> str:
    """Extracts cleaned text representation from a YouTube processed record.

    Prioritizes title_cleaned + description_cleaned, falling back to raw title + description.
    """
    title = str(record.get("title_cleaned") or record.get("title") or "").strip()
    desc = str(record.get("description_cleaned") or record.get("description") or "").strip()
    return f"{title} {desc}".strip()


def run_sentiment_pipeline(
    reddit_input_path: Optional[Path | str] = None,
    youtube_input_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    analyzer: Optional[MultilingualSentimentAnalyzer] = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Runs the multilingual sentiment analysis pipeline with temporal tracking.

    Args:
        reddit_input_path: Path to processed Reddit JSON. Defaults to data/processed/reddit_posts_processed.json.
        youtube_input_path: Path to processed YouTube JSON. Defaults to data/processed/youtube_videos_processed.json.
        output_path: Destination path for results JSON. Defaults to data/processed/sentiment_analysis_results.json.
        analyzer: Optional instance of MultilingualSentimentAnalyzer (e.g. mock or preconfigured).
        batch_size: Batch size for model inference.

    Returns:
        Structured dictionary containing pipeline summaries, temporal trends, and enriched records.
    """
    reddit_path = Path(reddit_input_path) if reddit_input_path else DEFAULT_REDDIT_INPUT
    youtube_path = Path(youtube_input_path) if youtube_input_path else DEFAULT_YOUTUBE_INPUT
    dest_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    # Initialize analyzer if not provided
    if analyzer is None:
        analyzer = MultilingualSentimentAnalyzer()

    # Load Reddit processed records
    reddit_records: list[dict[str, Any]] = []
    if reddit_path.exists():
        try:
            with open(reddit_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    reddit_records = loaded
                else:
                    logger.warning("Reddit input at %s is not a list. Skipping.", reddit_path)
        except Exception as exc:
            logger.error("Failed to read Reddit input %s: %s", reddit_path, exc)
    else:
        logger.info("Reddit processed file not found at %s. Proceeding with empty list.", reddit_path)

    # Load YouTube processed records
    youtube_records: list[dict[str, Any]] = []
    if youtube_path.exists():
        try:
            with open(youtube_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    youtube_records = loaded
                else:
                    logger.warning("YouTube input at %s is not a list. Skipping.", youtube_path)
        except Exception as exc:
            logger.error("Failed to read YouTube input %s: %s", youtube_path, exc)
    else:
        logger.info("YouTube processed file not found at %s. Proceeding with empty list.", youtube_path)

    now_iso = datetime.now(timezone.utc).isoformat()

    # Enrich Reddit records
    enriched_reddit: list[dict[str, Any]] = []
    if reddit_records:
        reddit_texts = [extract_reddit_text(r) for r in reddit_records]
        reddit_sentiments = analyzer.analyze_batch(reddit_texts, batch_size=batch_size)
        for rec, sent in zip(reddit_records, reddit_sentiments):
            enriched = dict(rec)
            enriched["sentiment_label"] = sent.get("label", LABEL_UNDETERMINED)
            enriched["sentiment_score"] = sent.get("score", 0.0)
            enriched["sentiment_status"] = sent.get("status", "unknown")
            enriched["analyzed_at"] = now_iso
            enriched_reddit.append(enriched)

    # Enrich YouTube records
    enriched_youtube: list[dict[str, Any]] = []
    if youtube_records:
        youtube_texts = [extract_youtube_text(r) for r in youtube_records]
        youtube_sentiments = analyzer.analyze_batch(youtube_texts, batch_size=batch_size)
        for rec, sent in zip(youtube_records, youtube_sentiments):
            enriched = dict(rec)
            enriched["sentiment_label"] = sent.get("label", LABEL_UNDETERMINED)
            enriched["sentiment_score"] = sent.get("score", 0.0)
            enriched["sentiment_status"] = sent.get("status", "unknown")
            enriched["analyzed_at"] = now_iso
            enriched_youtube.append(enriched)

    # Calculate summaries
    all_enriched = enriched_reddit + enriched_youtube
    overall_summary = SentimentTemporalTracker.calculate_window_metrics(all_enriched)
    reddit_summary = SentimentTemporalTracker.calculate_window_metrics(enriched_reddit)
    youtube_summary = SentimentTemporalTracker.calculate_window_metrics(enriched_youtube)

    # Temporal aggregations for all records
    # For combined, harmonize timestamp field: use created_utc if present, else published_at, else collected_at
    combined_for_temporal: list[dict[str, Any]] = []
    for rec in all_enriched:
        ts = rec.get("created_utc") or rec.get("published_at") or rec.get("collected_at")
        temp_rec = dict(rec)
        temp_rec["_temporal_timestamp"] = ts
        combined_for_temporal.append(temp_rec)

    daily_trends = SentimentTemporalTracker.aggregate_by_window(
        combined_for_temporal,
        time_field="_temporal_timestamp",
        window="daily",
    )
    weekly_trends = SentimentTemporalTracker.aggregate_by_window(
        combined_for_temporal,
        time_field="_temporal_timestamp",
        window="weekly",
    )

    result_payload: dict[str, Any] = {
        "status": "success",
        "generated_at": now_iso,
        "model_name": getattr(analyzer, "model_name", DEFAULT_MODEL_NAME),
        "total_records_processed": len(all_enriched),
        "summary": overall_summary,
        "reddit_summary": reddit_summary,
        "youtube_summary": youtube_summary,
        "temporal_aggregations": {
            "daily": daily_trends,
            "weekly": weekly_trends,
        },
        "reddit_records": enriched_reddit,
        "youtube_records": enriched_youtube,
    }

    # Persist to output file safely
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info(
        "Sentiment pipeline completed successfully. Output persisted to %s (%d records).",
        dest_path,
        len(all_enriched),
    )

    return result_payload

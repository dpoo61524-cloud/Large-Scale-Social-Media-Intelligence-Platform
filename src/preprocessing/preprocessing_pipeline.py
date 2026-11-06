"""Social Media Text Preprocessing Pipeline (Company Step 2).

Integrates text cleaning, emoji conversion, and Spark NLP language detection
into a high-throughput, batch-capable preprocessing pipeline for Reddit and YouTube.
Reads raw data from data/raw/, processes records without modifying original raw data,
and persists enriched records to data/processed/.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import load_config
from src.preprocessing.emoji_handler import handle_emojis
from src.preprocessing.language_detector import detect_language
from src.preprocessing.text_cleaner import clean_text


def preprocess_reddit_record(
    record: dict[str, Any],
    pipeline_override: Any = None,
) -> dict[str, Any]:
    """Preprocesses a single Reddit post record.

    Processes title and text separately:
    1. Text cleaning (HTML, URLs, whitespace, noise).
    2. Emoji handling (semantic token conversion).
    3. Language detection via Spark NLP.

    Preserves all original Step 1 metadata.
    """
    if not isinstance(record, dict) or "post_id" not in record:
        raise ValueError("Invalid Reddit record: must be a dict containing 'post_id'.")

    raw_title = record.get("title", "")
    raw_text = record.get("text", "")

    # Clean text and handle emojis for title and body separately
    title_cleaned = handle_emojis(clean_text(raw_title))
    text_cleaned = handle_emojis(clean_text(raw_text))

    # Detect language using combined content
    combined_content = f"{title_cleaned} {text_cleaned}".strip()
    lang_info = detect_language(combined_content, pipeline_override=pipeline_override)

    # Preserve all original fields and append processed enrichment fields
    processed = dict(record)
    processed["title_cleaned"] = title_cleaned
    processed["text_cleaned"] = text_cleaned
    processed["detected_language"] = lang_info.get("language_name", "Undetermined")
    processed["language_code"] = lang_info.get("language_code", "und")
    processed["preprocessing_status"] = "success"
    processed["processed_at"] = datetime.now(timezone.utc).isoformat()

    return processed


def preprocess_youtube_record(
    record: dict[str, Any],
    pipeline_override: Any = None,
) -> dict[str, Any]:
    """Preprocesses a single YouTube video record.

    Processes title and description separately:
    1. Text cleaning (HTML, URLs, whitespace, noise).
    2. Emoji handling (semantic token conversion).
    3. Language detection via Spark NLP.

    Preserves all original Step 1 metadata.
    """
    if not isinstance(record, dict) or "video_id" not in record:
        raise ValueError("Invalid YouTube record: must be a dict containing 'video_id'.")

    raw_title = record.get("title", "")
    raw_desc = record.get("description", "")

    # Clean text and handle emojis for title and description separately
    title_cleaned = handle_emojis(clean_text(raw_title))
    desc_cleaned = handle_emojis(clean_text(raw_desc))

    # Detect language using combined content
    combined_content = f"{title_cleaned} {desc_cleaned}".strip()
    lang_info = detect_language(combined_content, pipeline_override=pipeline_override)

    # Preserve all original fields and append processed enrichment fields
    processed = dict(record)
    processed["title_cleaned"] = title_cleaned
    processed["description_cleaned"] = desc_cleaned
    processed["detected_language"] = lang_info.get("language_name", "Undetermined")
    processed["language_code"] = lang_info.get("language_code", "und")
    processed["preprocessing_status"] = "success"
    processed["processed_at"] = datetime.now(timezone.utc).isoformat()

    return processed


def preprocess_reddit_dataset(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    batch_size: int = 100,
    merge: bool = True,
    pipeline_override: Any = None,
) -> dict[str, Any]:
    """Batched preprocessing pipeline for Reddit records.

    Reads raw JSON from data/raw/ without modifying it, processes records in
    configurable chunks, and writes enriched outputs to data/processed/.
    """
    config = load_config()
    src = Path(input_path) if input_path else config.raw_data_path / "reddit_posts_raw.json"
    dst = Path(output_path) if output_path else config.processed_data_path / "reddit_posts_processed.json"

    if not src.exists():
        return {
            "status": "error",
            "source": "reddit",
            "message": f"Raw data file not found at: {src}. Please run ingestion first.",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
            "output_file": None,
            "language_counts": {},
            "errors": [f"Missing input raw file: {src}"],
        }

    try:
        with open(src, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "reddit",
            "message": f"Failed reading raw JSON: {err}",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
            "output_file": None,
            "language_counts": {},
            "errors": [str(err)],
        }

    if not isinstance(raw_records, list):
        return {
            "status": "error",
            "source": "reddit",
            "message": f"Expected JSON list in {src}, got {type(raw_records).__name__}",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
            "output_file": None,
            "language_counts": {},
            "errors": ["Root JSON element must be an array."],
        }

    processed_by_id: dict[str, dict[str, Any]] = {}
    if merge and dst.exists():
        try:
            with open(dst, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    for item in existing:
                        if isinstance(item, dict) and "post_id" in item:
                            processed_by_id[item["post_id"]] = item
        except Exception:
            processed_by_id = {}

    success_count = 0
    failed_count = 0
    errors: list[str] = []
    language_counts: Counter[str] = Counter()

    # Process in configurable batches to avoid memory bloat
    total = len(raw_records)
    for i in range(0, total, max(1, batch_size)):
        batch = raw_records[i : i + batch_size]
        for record in batch:
            try:
                proc = preprocess_reddit_record(record, pipeline_override=pipeline_override)
                pid = proc["post_id"]
                processed_by_id[pid] = proc
                success_count += 1
                lang = proc.get("detected_language", "Undetermined")
                language_counts[lang] += 1
            except Exception as err:
                failed_count += 1
                rec_id = record.get("post_id", "unknown_id") if isinstance(record, dict) else "non_dict"
                errors.append(f"Record '{rec_id}': {err}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(list(processed_by_id.values()), f, indent=2, ensure_ascii=False)

    return {
        "status": "success" if success_count > 0 else ("failed" if errors else "no_records"),
        "source": "reddit",
        "total_records": total,
        "processed_records": success_count,
        "failed_records": failed_count,
        "language_counts": dict(language_counts),
        "output_file": str(dst),
        "errors": errors,
    }


def preprocess_youtube_dataset(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    batch_size: int = 100,
    merge: bool = True,
    pipeline_override: Any = None,
) -> dict[str, Any]:
    """Batched preprocessing pipeline for YouTube records.

    Reads raw JSON from data/raw/ without modifying it, processes records in
    configurable chunks, and writes enriched outputs to data/processed/.
    """
    config = load_config()
    src = Path(input_path) if input_path else config.raw_data_path / "youtube_videos_raw.json"
    dst = Path(output_path) if output_path else config.processed_data_path / "youtube_videos_processed.json"

    if not src.exists():
        return {
            "status": "error",
            "source": "youtube",
            "message": f"Raw data file not found at: {src}. Please run ingestion first.",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
            "output_file": None,
            "language_counts": {},
            "errors": [f"Missing input raw file: {src}"],
        }

    try:
        with open(src, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
    except Exception as err:
        return {
            "status": "error",
            "source": "youtube",
            "message": f"Failed reading raw JSON: {err}",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
            "output_file": None,
            "language_counts": {},
            "errors": [str(err)],
        }

    if not isinstance(raw_records, list):
        return {
            "status": "error",
            "source": "youtube",
            "message": f"Expected JSON list in {src}, got {type(raw_records).__name__}",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
            "output_file": None,
            "language_counts": {},
            "errors": ["Root JSON element must be an array."],
        }

    processed_by_id: dict[str, dict[str, Any]] = {}
    if merge and dst.exists():
        try:
            with open(dst, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    for item in existing:
                        if isinstance(item, dict) and "video_id" in item:
                            processed_by_id[item["video_id"]] = item
        except Exception:
            processed_by_id = {}

    success_count = 0
    failed_count = 0
    errors: list[str] = []
    language_counts: Counter[str] = Counter()

    total = len(raw_records)
    for i in range(0, total, max(1, batch_size)):
        batch = raw_records[i : i + batch_size]
        for record in batch:
            try:
                proc = preprocess_youtube_record(record, pipeline_override=pipeline_override)
                vid = proc["video_id"]
                processed_by_id[vid] = proc
                success_count += 1
                lang = proc.get("detected_language", "Undetermined")
                language_counts[lang] += 1
            except Exception as err:
                failed_count += 1
                rec_id = record.get("video_id", "unknown_id") if isinstance(record, dict) else "non_dict"
                errors.append(f"Record '{rec_id}': {err}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(list(processed_by_id.values()), f, indent=2, ensure_ascii=False)

    return {
        "status": "success" if success_count > 0 else ("failed" if errors else "no_records"),
        "source": "youtube",
        "total_records": total,
        "processed_records": success_count,
        "failed_records": failed_count,
        "language_counts": dict(language_counts),
        "output_file": str(dst),
        "errors": errors,
    }

"""Unified Social Intelligence Data Consolidator.

Consolidates preprocessed social media records (Reddit and YouTube) with downstream
analytical enrichments:
- Step 2: Cleaned text, emoji handling, and language detection
- Step 3: Multilingual sentiment polarity and confidence scoring
- Step 4: BERTopic dynamic topic assignment and keywords
- Step 5: Neo4j/NetworkX PageRank, centrality, and community partitions
- Step 6: LIAR misinformation veracity classification, credibility scores, and risk tiers

Prepares unified documents optimized for Elasticsearch indexing and Kibana visual search.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

logger = logging.getLogger(__name__)

# Schema and pipeline version declarations
SUPPORTED_SCHEMA_VERSION = "2.0"
PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)$")

# Allowed enumerated values for schema validation
ALLOWED_PLATFORMS = {"reddit", "youtube"}
ALLOWED_CONSOLIDATION_STATUSES = {"complete", "degraded", "partial", "unprocessed"}
ALLOWED_SOURCE_STATUSES = {"loaded", "empty", "missing", "corrupted"}

# Canonical record root fields (exact 13 required fields)
CANONICAL_RECORD_ROOT_FIELDS = {
    "document_id",
    "platform",
    "native_id",
    "url",
    "timestamps",
    "author",
    "content",
    "engagement",
    "metadata",
    "sentiment",
    "topics",
    "misinformation",
    "consolidation",
}

# Top-level envelope metadata keys
ENVELOPE_METADATA_FIELDS = {
    "schema_version",
    "pipeline_version",
    "status",
    "consolidation_status",
    "generated_at",
    "provenance",
    "metrics",
    "audit_warnings",
    "rejected_records",
    "records",
}

# Default paths for pipeline input and output files
DEFAULT_REDDIT_PROCESSED_PATH = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_PROCESSED_PATH = Path("data/processed/youtube_videos_processed.json")
DEFAULT_SENTIMENT_PATH = Path("data/processed/sentiment_analysis_results.json")
DEFAULT_TOPICS_PATH = Path("data/processed/topic_modeling_results.json")
DEFAULT_GRAPH_PATH = Path("data/processed/graph_analysis_results.json")
DEFAULT_MISINFORMATION_PATH = Path("data/processed/misinformation_analysis_results.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/consolidated_intelligence.json")

# Anonymous accounts to filter from active graph status
ANONYMOUS_USERS = {
    "[deleted]",
    "[removed]",
    "automoderator",
    "deleted",
    "none",
    "null",
    "",
}


def normalize_utc_timestamp(val: Any) -> Tuple[bool, Optional[str], datetime]:
    """Normalizes any timestamp format into a timezone-aware ISO 8601 UTC string.

    Handles:
    - Numeric Unix epoch timestamps (seconds as int/float, e.g. Reddit created_utc)
    - ISO 8601 strings with explicit timezone offsets (e.g. "+00:00" or "Z")
    - Timezone-naive ISO 8601 strings (explicitly treated as UTC, never local time)

    Args:
        val: Timestamp value in epoch numeric or string representation.

    Returns:
        Tuple of (is_valid, iso_utc_string, utc_datetime_object).
        If missing or invalid, returns (False, None, datetime.min in UTC).
    """
    fallback_dt = datetime.min.replace(tzinfo=timezone.utc)
    if val is None or val == "":
        return False, None, fallback_dt

    # 1. Numeric epoch (Reddit created_utc)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
            return True, dt.isoformat(), dt
        except (ValueError, OverflowError, OSError):
            return False, None, fallback_dt

    # 2. String timestamp (ISO 8601 / RFC 3339)
    if isinstance(val, str):
        cleaned = val.strip()
        if not cleaned:
            return False, None, fallback_dt
        try:
            if cleaned.endswith("Z"):
                cleaned = cleaned[:-1] + "+00:00"
            dt = datetime.fromisoformat(cleaned)

            # Explicitly interpret naive timestamps as UTC (never local host time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

            return True, dt.isoformat(), dt
        except ValueError:
            return False, None, fallback_dt

    return False, None, fallback_dt


def load_json_file(file_path: Path) -> Tuple[str, Any, Optional[str]]:
    """Loads a JSON file, distinguishing missing, empty, malformed, and valid states.

    Args:
        file_path: Path to target file.

    Returns:
        Tuple of (state, data, error_message)
        where state is one of: "valid", "missing", "empty", "malformed".
    """
    if not file_path.exists():
        return "missing", None, None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return "empty", None, None
            data = json.loads(content)
            if data == [] or data == {}:
                return "empty", data, None
            return "valid", data, None
    except Exception as exc:
        return "malformed", None, str(exc)


def _is_strict_bool(val: Any) -> bool:
    """Returns True if and only if val is an exact boolean."""
    return isinstance(val, bool)


def _is_strict_int(val: Any) -> bool:
    """Returns True if and only if val is an exact int and not a boolean."""
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_float(val: Any) -> bool:
    """Returns True if val is a numeric int/float (not bool) and finite."""
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def normalize_consolidator_topic_keywords(
    raw_keywords: Any,
    rejected_collector: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Normalizes topic keywords to canonical format: list of dicts with 'word' and 'weight'.

    Preserves every meaningful keyword with a non-empty string 'word', regardless of
    whether the input is a string or a dictionary containing additional metadata fields.
    Normalizes output strictly to the Elasticsearch canonical contract:
        {"word": str, "weight": float | None}

    Weight Handling:
    - Numeric int/float weights are converted to finite floats without inventing synthetic weights.
    - Missing, non-numeric, boolean, or non-finite weights normalize safely to None,
      consistent with the Elasticsearch nested mapping and schema contract.

    Rejection Policy for Uninterpretable Entries:
    - Entries that cannot be interpreted as meaningful keywords (e.g. non-string/non-dict types,
      empty/whitespace strings, dictionaries missing 'word', or dictionaries with blank 'word')
      are explicitly rejected and documented.
    - They are logged via logger.warning and, if rejected_collector is provided, recorded
      with their index, raw representation, and specific rejection reason to prevent silent data loss.

    Args:
        raw_keywords: Raw sequence of keyword strings or dictionaries.
        rejected_collector: Optional list to append diagnostic rejection records for uninterpretable entries.

    Returns:
        List of canonical dictionaries matching {"word": str, "weight": float | None}.
    """
    if not isinstance(raw_keywords, list) or len(raw_keywords) == 0:
        if raw_keywords is not None and not isinstance(raw_keywords, list):
            reason = f"expected_list_got_{type(raw_keywords).__name__}"
            logger.warning("Invalid topic_keywords container: %s", reason)
            if rejected_collector is not None:
                rejected_collector.append({"index": None, "entry": repr(raw_keywords), "reason": reason})
        return []

    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_keywords):
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                normalized.append({"word": cleaned, "weight": None})
            else:
                reason = "empty_or_whitespace_string"
                logger.warning("Uninterpretable topic keyword at index %d: %s", idx, reason)
                if rejected_collector is not None:
                    rejected_collector.append({"index": idx, "entry": repr(item), "reason": reason})
        elif isinstance(item, dict):
            raw_word = item.get("word")
            if not isinstance(raw_word, str) or not raw_word.strip():
                reason = "missing_or_blank_word_field"
                logger.warning("Uninterpretable topic keyword dict at index %d: %s (entry: %s)", idx, reason, repr(item))
                if rejected_collector is not None:
                    rejected_collector.append({"index": idx, "entry": repr(item), "reason": reason})
                continue

            word = raw_word.strip()
            raw_weight = item.get("weight")
            valid_weight: Optional[float] = None

            if raw_weight is not None:
                if _is_strict_int(raw_weight) or isinstance(raw_weight, float):
                    try:
                        f_weight = float(raw_weight)
                        if math.isfinite(f_weight):
                            valid_weight = f_weight
                        else:
                            logger.warning(
                                "Non-finite keyword weight at index %d for word '%s': %s; normalizing weight to None",
                                idx, word, raw_weight,
                            )
                    except (OverflowError, ValueError):
                        logger.warning(
                            "Overflow/ValueError keyword weight at index %d for word '%s': %s; normalizing weight to None",
                            idx, word, raw_weight,
                        )
                else:
                    logger.warning(
                        "Invalid keyword weight type at index %d for word '%s': %s; normalizing weight to None",
                        idx, word, type(raw_weight).__name__,
                    )

            # Preserve valid word and weight, ignoring extra fields without dropping the keyword
            normalized.append({"word": word, "weight": valid_weight})
        else:
            reason = f"unsupported_element_type_{type(item).__name__}"
            logger.warning("Uninterpretable topic keyword at index %d: %s (entry: %s)", idx, reason, repr(item))
            if rejected_collector is not None:
                rejected_collector.append({"index": idx, "entry": repr(item), "reason": reason})

    return normalized


class ConsolidatedArtifactValidator:
    """Strict validator for Consolidated Social Intelligence Stage 10 artifacts."""

    @classmethod
    def validate_consolidated_artifact(cls, payload: Any) -> None:
        """Validates a consolidated artifact payload against Schema-v2.0 contract.

        Raises:
            ValueError: If any schema, type, invariant, or consistency check fails.
            TypeError: If payload is not a dictionary.
        """
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dictionary for consolidated artifact, got {type(payload).__name__}")

        # 1. Top-Level Required Envelope Keys
        missing_keys = ENVELOPE_METADATA_FIELDS - set(payload.keys())
        if missing_keys:
            raise ValueError(f"Consolidated artifact missing required envelope keys: {sorted(missing_keys)}")

        # 2. Schema Version Validation (Strict '2.0' specification)
        schema_ver = payload.get("schema_version")
        if schema_ver is None:
            raise ValueError("Missing 'schema_version' in consolidated artifact")
        if not isinstance(schema_ver, str):
            raise ValueError(f"'schema_version' must be a string, got {type(schema_ver).__name__}")
        if schema_ver != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{schema_ver}'; expected '{SUPPORTED_SCHEMA_VERSION}'"
            )

        # 3. Pipeline Version Validation (Backward compatibility preservation)
        pipeline_ver = payload.get("pipeline_version")
        if not isinstance(pipeline_ver, str) or not pipeline_ver.strip():
            raise ValueError("Missing or invalid 'pipeline_version'; expected non-empty string")

        # 4. Status Validation
        status = payload.get("status")
        if status != "success":
            raise ValueError(f"Invalid artifact status '{status}'; expected 'success'")

        consolidation_status = payload.get("consolidation_status")
        if consolidation_status not in ALLOWED_CONSOLIDATION_STATUSES:
            raise ValueError(
                f"Invalid 'consolidation_status' '{consolidation_status}'; must be one of {sorted(ALLOWED_CONSOLIDATION_STATUSES)}"
            )

        # 5. Generated_at timestamp validation
        gen_at = payload.get("generated_at")
        if not isinstance(gen_at, str) or not gen_at.strip():
            raise ValueError("Missing or invalid 'generated_at'; expected ISO 8601 UTC timestamp string")
        is_valid_ts, _, _ = normalize_utc_timestamp(gen_at)
        if not is_valid_ts:
            raise ValueError(f"Unparseable ISO 8601 'generated_at' timestamp: '{gen_at}'")

        # 6. Audit Warnings & Rejected Records collections
        audit_warnings = payload.get("audit_warnings")
        if not isinstance(audit_warnings, list):
            raise ValueError(f"'audit_warnings' must be a list, got {type(audit_warnings).__name__}")

        rejected_records = payload.get("rejected_records")
        if not isinstance(rejected_records, list):
            raise ValueError(f"'rejected_records' must be a list, got {type(rejected_records).__name__}")

        # 7. Records collection
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError(f"'records' must be a list, got {type(records).__name__}")

        # 8. Provenance Validation
        provenance = payload.get("provenance")
        cls._validate_provenance(provenance)

        # 9. Record-Level Validation
        cls._validate_records(records)

        # 10. Metrics Validation
        metrics = payload.get("metrics")
        cls._validate_metrics(metrics, records=records, rejected_records=rejected_records)

    @classmethod
    def _validate_provenance(cls, provenance: Any) -> None:
        """Validates provenance dictionary structure, platform entries, and count invariants."""
        if not isinstance(provenance, dict):
            raise ValueError(f"'provenance' must be a dictionary, got {type(provenance).__name__}")

        required_prov_keys = {
            "reddit_input_file",
            "reddit_status",
            "reddit_records_count",
            "youtube_input_file",
            "youtube_status",
            "youtube_records_count",
        }
        missing = required_prov_keys - set(provenance.keys())
        if missing:
            raise ValueError(f"'provenance' missing required keys: {sorted(missing)}")

        for plat in ["reddit", "youtube"]:
            infile = provenance.get(f"{plat}_input_file")
            if not isinstance(infile, str) or not infile.strip():
                raise ValueError(f"provenance['{plat}_input_file'] must be a non-empty string")

            p_status = provenance.get(f"{plat}_status")
            if p_status not in ALLOWED_SOURCE_STATUSES:
                raise ValueError(
                    f"provenance['{plat}_status'] '{p_status}' invalid; must be one of {sorted(ALLOWED_SOURCE_STATUSES)}"
                )

            count = provenance.get(f"{plat}_records_count")
            if not _is_strict_int(count) or count < 0:
                raise ValueError(f"provenance['{plat}_records_count'] must be a non-negative integer, got {count}")

            if p_status in {"missing", "empty", "corrupted"} and count != 0:
                raise ValueError(
                    f"provenance['{plat}_records_count'] must be 0 when status is '{p_status}', got {count}"
                )

    @classmethod
    def _validate_metrics(
        cls,
        metrics: Any,
        records: List[Dict[str, Any]],
        rejected_records: List[Dict[str, Any]],
    ) -> None:
        """Validates metric integer types, non-negativity, and mathematical sum invariants."""
        if not isinstance(metrics, dict):
            raise ValueError(f"'metrics' must be a dictionary, got {type(metrics).__name__}")

        required_metric_keys = {
            "total_records_consolidated",
            "platform_counts",
            "complete_records",
            "degraded_records",
            "partial_records",
            "unprocessed_records",
            "duplicates_deduplicated",
            "rejected_records_count",
        }
        missing = required_metric_keys - set(metrics.keys())
        if missing:
            raise ValueError(f"'metrics' missing required keys: {sorted(missing)}")

        int_metric_fields = [
            "total_records_consolidated",
            "complete_records",
            "degraded_records",
            "partial_records",
            "unprocessed_records",
            "duplicates_deduplicated",
            "rejected_records_count",
        ]
        for field in int_metric_fields:
            val = metrics.get(field)
            if not _is_strict_int(val) or val < 0:
                raise ValueError(f"metrics['{field}'] must be a non-negative integer, got {val}")

        platform_counts = metrics.get("platform_counts")
        if not isinstance(platform_counts, dict):
            raise ValueError(f"metrics['platform_counts'] must be a dictionary, got {type(platform_counts).__name__}")

        for plat in ["reddit", "youtube"]:
            if plat not in platform_counts:
                raise ValueError(f"metrics['platform_counts'] missing key '{plat}'")
            cnt = platform_counts.get(plat)
            if not _is_strict_int(cnt) or cnt < 0:
                raise ValueError(f"metrics['platform_counts']['{plat}'] must be a non-negative integer, got {cnt}")

        total = metrics["total_records_consolidated"]
        if total != len(records):
            raise ValueError(
                f"metrics['total_records_consolidated'] ({total}) does not match len(records) ({len(records)})"
            )

        if metrics["rejected_records_count"] != len(rejected_records):
            raise ValueError(
                f"metrics['rejected_records_count'] ({metrics['rejected_records_count']}) does not match len(rejected_records) ({len(rejected_records)})"
            )

        reddit_actual = sum(1 for r in records if isinstance(r, dict) and r.get("platform") == "reddit")
        youtube_actual = sum(1 for r in records if isinstance(r, dict) and r.get("platform") == "youtube")

        if platform_counts["reddit"] != reddit_actual:
            raise ValueError(
                f"metrics['platform_counts']['reddit'] ({platform_counts['reddit']}) != actual ({reddit_actual})"
            )
        if platform_counts["youtube"] != youtube_actual:
            raise ValueError(
                f"metrics['platform_counts']['youtube'] ({platform_counts['youtube']}) != actual ({youtube_actual})"
            )

        if platform_counts["reddit"] + platform_counts["youtube"] != total:
            raise ValueError(
                f"Sum of platform counts ({platform_counts['reddit'] + platform_counts['youtube']}) != total ({total})"
            )

        status_sum = (
            metrics["complete_records"]
            + metrics["degraded_records"]
            + metrics["partial_records"]
            + metrics["unprocessed_records"]
        )
        if status_sum != total:
            raise ValueError(
                f"Sum of record status counts ({status_sum}) != total_records_consolidated ({total})"
            )

        complete_actual = sum(
            1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "complete"
        )
        degraded_actual = sum(
            1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "degraded"
        )
        partial_actual = sum(
            1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "partial"
        )
        unprocessed_actual = sum(
            1 for r in records if isinstance(r, dict) and r.get("consolidation", {}).get("status") == "unprocessed"
        )

        if metrics["complete_records"] != complete_actual:
            raise ValueError(
                f"metrics['complete_records'] ({metrics['complete_records']}) != actual ({complete_actual})"
            )
        if metrics["degraded_records"] != degraded_actual:
            raise ValueError(
                f"metrics['degraded_records'] ({metrics['degraded_records']}) != actual ({degraded_actual})"
            )
        if metrics["partial_records"] != partial_actual:
            raise ValueError(
                f"metrics['partial_records'] ({metrics['partial_records']}) != actual ({partial_actual})"
            )
        if metrics["unprocessed_records"] != unprocessed_actual:
            raise ValueError(
                f"metrics['unprocessed_records'] ({metrics['unprocessed_records']}) != actual ({unprocessed_actual})"
            )

    @classmethod
    def _validate_records(cls, records: List[Any]) -> None:
        """Validates canonical record fields, prevents envelope leakage, and verifies sub-dictionaries."""
        seen_doc_ids: Set[str] = set()

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ValueError(f"Record at index {idx} must be a dictionary, got {type(rec).__name__}")

            rec_keys = set(rec.keys())
            missing_fields = CANONICAL_RECORD_ROOT_FIELDS - rec_keys
            if missing_fields:
                raise ValueError(
                    f"Record at index {idx} missing canonical root fields: {sorted(missing_fields)}"
                )

            extra_fields = rec_keys - CANONICAL_RECORD_ROOT_FIELDS
            if extra_fields:
                raise ValueError(
                    f"Record at index {idx} contains forbidden extra fields (possible envelope leakage): {sorted(extra_fields)}"
                )

            plat = rec.get("platform")
            if plat not in ALLOWED_PLATFORMS:
                raise ValueError(
                    f"Record at index {idx} has invalid platform '{plat}'; must be in {sorted(ALLOWED_PLATFORMS)}"
                )

            nat_id = rec.get("native_id")
            if not isinstance(nat_id, str) or not nat_id.strip():
                raise ValueError(f"Record at index {idx} has missing or empty 'native_id'")

            doc_id = rec.get("document_id")
            expected_doc_id = f"{plat}_{nat_id}"
            if doc_id != expected_doc_id:
                raise ValueError(
                    f"Record at index {idx} document_id '{doc_id}' does not match expected '{expected_doc_id}'"
                )

            if doc_id in seen_doc_ids:
                raise ValueError(f"Duplicate document_id '{doc_id}' detected at index {idx}")
            seen_doc_ids.add(doc_id)

            # Sub-dictionary structural validation
            for subfield in [
                "timestamps",
                "author",
                "content",
                "engagement",
                "metadata",
                "sentiment",
                "topics",
                "misinformation",
                "consolidation",
            ]:
                sub_val = rec.get(subfield)
                if not isinstance(sub_val, dict):
                    raise ValueError(
                        f"Record '{doc_id}' field '{subfield}' must be a dictionary, got {type(sub_val).__name__}"
                    )

            # Topics keywords validation (nested list of canonical objects)
            top_obj = rec.get("topics", {})
            kw_list = top_obj.get("keywords")
            if not isinstance(kw_list, list):
                raise ValueError(f"Record '{doc_id}' topics.keywords must be a list, got {type(kw_list).__name__}")

            # 1. Sentiment validation & nullability enforcement
            sent_obj = rec.get("sentiment", {})
            sent_status = sent_obj.get("status")
            if sent_status not in {
                "success",
                "fallback",
                "empty_text",
                "error",
                "unavailable",
                "corrupted",
                "missing_status",
            }:
                raise ValueError(
                    f"Record '{doc_id}' sentiment.status '{sent_status}' invalid"
                )

            s_label = sent_obj.get("label")
            s_score = sent_obj.get("score")

            if sent_status == "success":
                if s_label not in {"positive", "neutral", "negative"}:
                    raise ValueError(
                        f"Record '{doc_id}' sentiment.label '{s_label}' invalid for success status"
                    )
                if not _is_finite_float(s_score) or not (-1.0 <= float(s_score) <= 1.0):
                    raise ValueError(
                        f"Record '{doc_id}' sentiment.score must be finite float in [-1.0, 1.0] for success status, got {s_score}"
                    )
            else:
                if s_label is not None:
                    raise ValueError(
                        f"Record '{doc_id}' sentiment.label must be None when status is '{sent_status}', got {s_label}"
                    )
                if s_score is not None:
                    raise ValueError(
                        f"Record '{doc_id}' sentiment.score must be None when status is '{sent_status}', got {s_score}"
                    )

            # 2. Topics validation & nullability enforcement
            top_status = top_obj.get("status")
            if top_status not in {
                "assigned",
                "outlier",
                "fallback",
                "empty_text",
                "error",
                "unavailable",
                "corrupted",
                "missing_status",
            }:
                raise ValueError(
                    f"Record '{doc_id}' topics.status '{top_status}' invalid"
                )

            t_id = top_obj.get("topic_id")
            t_prob = top_obj.get("probability")

            if top_status == "assigned":
                if not _is_strict_int(t_id) or t_id < 0:
                    raise ValueError(
                        f"Record '{doc_id}' topics.topic_id must be non-negative integer for assigned status, got {t_id}"
                    )
                if t_prob is not None:
                    if not _is_finite_float(t_prob) or not (0.0 <= float(t_prob) <= 1.0):
                        raise ValueError(
                            f"Record '{doc_id}' topics.probability must be finite float in [0.0, 1.0] or None for assigned status, got {t_prob}"
                        )
                for kidx, kw in enumerate(kw_list):
                    if not isinstance(kw, dict):
                        raise ValueError(
                            f"Record '{doc_id}' topics.keywords[{kidx}] must be a dict, got {type(kw).__name__}"
                        )
                    extra_kw_keys = set(kw.keys()) - {"word", "weight"}
                    if extra_kw_keys:
                        raise ValueError(
                            f"Record '{doc_id}' topics.keywords[{kidx}] contains unexpected keys: {sorted(extra_kw_keys)}"
                        )
                    if "word" not in kw or not isinstance(kw["word"], str) or not kw["word"].strip():
                        raise ValueError(
                            f"Record '{doc_id}' topics.keywords[{kidx}] missing valid 'word' string"
                        )
                    if "weight" in kw and kw["weight"] is not None:
                        if not _is_finite_float(kw["weight"]):
                            raise ValueError(
                                f"Record '{doc_id}' topics.keywords[{kidx}] 'weight' must be a finite float or None, got {kw['weight']}"
                            )
            elif top_status == "outlier":
                if not _is_strict_int(t_id) or t_id != -1:
                    raise ValueError(
                        f"Record '{doc_id}' topics.topic_id must equal -1 for outlier status, got {t_id}"
                    )
                if t_prob is not None:
                    if not _is_finite_float(t_prob) or not (0.0 <= float(t_prob) <= 1.0):
                        raise ValueError(
                            f"Record '{doc_id}' topics.probability must be finite float in [0.0, 1.0] or None for outlier status, got {t_prob}"
                        )
                if len(kw_list) != 0:
                    raise ValueError(
                        f"Record '{doc_id}' topics.keywords must be empty list for outlier status, got {len(kw_list)} items"
                    )
            else:
                if t_id is not None:
                    raise ValueError(
                        f"Record '{doc_id}' topics.topic_id must be None when status is '{top_status}', got {t_id}"
                    )
                if t_prob is not None:
                    raise ValueError(
                        f"Record '{doc_id}' topics.probability must be None when status is '{top_status}', got {t_prob}"
                    )
                if len(kw_list) != 0:
                    raise ValueError(
                        f"Record '{doc_id}' topics.keywords must be empty list when status is '{top_status}', got {len(kw_list)} items"
                    )

            # 3. Misinformation validation & nullability enforcement
            mis_obj = rec.get("misinformation", {})
            mis_status = mis_obj.get("status")
            if mis_status not in {
                "success",
                "fallback",
                "empty_text",
                "error",
                "unavailable",
                "corrupted",
                "missing_status",
            }:
                raise ValueError(
                    f"Record '{doc_id}' misinformation.status '{mis_status}' invalid"
                )

            v_label = mis_obj.get("veracity_label")
            c_score = mis_obj.get("credibility_score")
            r_tier = mis_obj.get("risk_tier")
            s_signals = mis_obj.get("stylometric_signals")

            if mis_status == "success":
                if v_label not in {"real", "fake"}:
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.veracity_label '{v_label}' invalid for success status"
                    )
                if r_tier not in {"low", "moderate", "high_risk"}:
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.risk_tier '{r_tier}' invalid for success status"
                    )
                if not _is_finite_float(c_score) or not (0.0 <= float(c_score) <= 1.0):
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.credibility_score must be finite float in [0.0, 1.0] for success status, got {c_score}"
                    )
                if s_signals is not None and not isinstance(s_signals, dict):
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.stylometric_signals must be a dictionary or None, got {type(s_signals).__name__}"
                    )
            else:
                if v_label is not None:
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.veracity_label must be None when status is '{mis_status}', got {v_label}"
                    )
                if r_tier is not None:
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.risk_tier must be None when status is '{mis_status}', got {r_tier}"
                    )
                if c_score is not None:
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.credibility_score must be None when status is '{mis_status}', got {c_score}"
                    )
                if s_signals is not None:
                    raise ValueError(
                        f"Record '{doc_id}' misinformation.stylometric_signals must be None when status is '{mis_status}', got {s_signals}"
                    )

            # Consolidation block validation
            cons_obj = rec.get("consolidation", {})
            c_status = cons_obj.get("status")
            if c_status not in ALLOWED_CONSOLIDATION_STATUSES:
                raise ValueError(
                    f"Record '{doc_id}' consolidation.status '{c_status}' invalid"
                )
            is_comp = cons_obj.get("is_complete")
            if not _is_strict_bool(is_comp):
                raise ValueError(
                    f"Record '{doc_id}' consolidation.is_complete must be a strict boolean, got {is_comp}"
                )
            src_joined = cons_obj.get("sources_joined")
            if not isinstance(src_joined, list):
                raise ValueError(
                    f"Record '{doc_id}' consolidation.sources_joined must be a list, got {type(src_joined).__name__}"
                )


def _promote_artifact_atomically(
    payload: Dict[str, Any],
    destination_path: Path | str,
) -> None:
    """Atomically persists and validates a consolidated intelligence artifact.

    Sequence:
    1. Ensures destination parent directory exists.
    2. Writes payload to a temporary file in the same parent directory.
    3. Flushes and calls os.fsync to ensure data is committed to disk.
    4. Closes temporary file.
    5. Reads back written JSON to verify serialization integrity.
    6. Validates read-back artifact with ConsolidatedArtifactValidator.
    7. Atomically renames temporary file to destination path via os.replace.
    8. In finally block, unlinks temporary file if it still exists (e.g. on error).

    Args:
        payload: Consolidated artifact payload dictionary.
        destination_path: Target destination path for final artifact.

    Raises:
        ValueError: If payload fails validation.
        OSError: If filesystem operations fail.
    """
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.parent / f".tmp_{dest.name}_{uuid.uuid4().hex}"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Verify integrity by reading back
        with open(temp_path, "r", encoding="utf-8") as f:
            read_back = json.load(f)

        # Validate read-back artifact against strict Schema-v2
        ConsolidatedArtifactValidator.validate_consolidated_artifact(read_back)

        # Atomic promotion
        os.replace(temp_path, dest)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _load_analytical_collection(
    module_name: str,
    file_path: Path,
    data: Dict[str, Any],
    collection_key: str,
    platform: str,
    id_field: str,
    target_map: Dict[Tuple[str, str], Dict[str, Any]],
    corrupted_platforms: Set[str],
    audit_warnings: List[Dict[str, Any]],
) -> None:
    """Validates an analytical collection and populates the lookup map.

    Enforces:
    - Collection must be a list
    - Each entry must be a dictionary
    - Each entry must contain a valid, non-empty string native ID
    - Invalid entries trigger audit warnings and mark platform corrupted
    """
    if collection_key not in data:
        corrupted_platforms.add(platform)
        audit_warnings.append({
            "module": module_name,
            "file": str(file_path),
            "field": collection_key,
            "platform": platform,
            "status": "missing_collection_key",
            "reason": "missing_collection_key",
        })
        return

    col = data.get(collection_key)
    if not isinstance(col, list):
        corrupted_platforms.add(platform)
        audit_warnings.append({
            "module": module_name,
            "file": str(file_path),
            "field": collection_key,
            "status": "invalid_collection",
            "got": type(col).__name__,
        })
        return

    for idx, rec in enumerate(col):
        if not isinstance(rec, dict):
            corrupted_platforms.add(platform)
            audit_warnings.append({
                "module": module_name,
                "file": str(file_path),
                "field": collection_key,
                "platform": platform,
                "status": "invalid_record_entry",
                "index": idx,
                "reason": "record_not_a_dictionary",
                "entry": str(rec),
            })
            continue

        raw_id = rec.get(id_field)
        if not isinstance(raw_id, str) or not raw_id.strip():
            corrupted_platforms.add(platform)
            audit_warnings.append({
                "module": module_name,
                "file": str(file_path),
                "field": collection_key,
                "platform": platform,
                "status": "invalid_record_entry",
                "index": idx,
                "reason": "missing_or_invalid_native_id",
                "id_field": id_field,
                "entry": str(rec),
            })
            continue

        native_id = raw_id.strip()
        target_map[(platform, native_id)] = rec


def _determine_record_consolidation_status(
    has_base: bool,
    sent_status: str,
    top_status: str,
    mis_status: str,
) -> str:
    """Computes consolidation status for a single record using precedence order.

    Precedence: unprocessed > partial > degraded > complete
    """
    if not has_base:
        return "unprocessed"

    statuses = {sent_status, top_status, mis_status}

    if any(s in {"unavailable", "corrupted", "error", "missing_status"} for s in statuses):
        return "partial"

    if any(s in {"fallback", "empty_text"} for s in statuses):
        return "degraded"

    if sent_status == "success" and top_status in {"assigned", "outlier"} and mis_status == "success":
        return "complete"

    return "partial"


def consolidate_social_intelligence(
    reddit_processed_path: Optional[Path | str] = None,
    youtube_processed_path: Optional[Path | str] = None,
    sentiment_path: Optional[Path | str] = None,
    topics_path: Optional[Path | str] = None,
    graph_path: Optional[Path | str] = None,
    misinformation_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    execution_time: Optional[str] = None,
) -> Dict[str, Any]:
    """Consolidates preprocessed social media records and downstream analytics.

    Joins Step 2 preprocessed posts with sentiment, topic, graph, and credibility
    analytics using platform-scoped composite keys and deterministic deduplication.
    Persists structured results non-destructively to data/processed/.

    Args:
        reddit_processed_path: Path to processed Reddit JSON.
        youtube_processed_path: Path to processed YouTube JSON.
        sentiment_path: Path to sentiment results JSON.
        topics_path: Path to topic modeling results JSON.
        graph_path: Path to graph analysis results JSON.
        misinformation_path: Path to misinformation results JSON.
        output_path: Destination path for consolidated JSON.
        execution_time: Optional fixed ISO UTC timestamp for deterministic execution.

    Returns:
        Structured dictionary containing consolidated records, metrics, and audit log.

    Raises:
        FileNotFoundError: If all base processed record sources are missing.
        ValueError: If base processed record files are malformed or structurally invalid.
    """
    r_path = Path(reddit_processed_path) if reddit_processed_path else DEFAULT_REDDIT_PROCESSED_PATH
    y_path = Path(youtube_processed_path) if youtube_processed_path else DEFAULT_YOUTUBE_PROCESSED_PATH
    s_path = Path(sentiment_path) if sentiment_path else DEFAULT_SENTIMENT_PATH
    t_path = Path(topics_path) if topics_path else DEFAULT_TOPICS_PATH
    g_path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
    m_path = Path(misinformation_path) if misinformation_path else DEFAULT_MISINFORMATION_PATH
    dest_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    audit_warnings: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    # 1. Load and Validate Base Record Files
    r_state, r_data, r_err = load_json_file(r_path)
    y_state, y_data, y_err = load_json_file(y_path)

    if r_state == "malformed":
        raise ValueError(f"Corrupted base Reddit file at {r_path}: {r_err}")
    if y_state == "malformed":
        raise ValueError(f"Corrupted base YouTube file at {y_path}: {y_err}")

    if r_state == "valid" and not isinstance(r_data, list):
        raise ValueError(f"Structurally invalid base Reddit file at {r_path}: expected JSON list, got {type(r_data).__name__}")
    if y_state == "valid" and not isinstance(y_data, list):
        raise ValueError(f"Structurally invalid base YouTube file at {y_path}: expected JSON list, got {type(y_data).__name__}")

    if r_state == "missing" and y_state == "missing":
        raise FileNotFoundError(
            f"No base processed records found. Looked in {r_path} and {y_path}."
        )

    # Record explicit audit warnings for missing or empty base files
    if r_state == "missing":
        audit_warnings.append({"module": "reddit_base", "file": str(r_path), "status": "missing"})
    elif r_state == "empty":
        audit_warnings.append({"module": "reddit_base", "file": str(r_path), "status": "empty"})

    if y_state == "missing":
        audit_warnings.append({"module": "youtube_base", "file": str(y_path), "status": "missing"})
    elif y_state == "empty":
        audit_warnings.append({"module": "youtube_base", "file": str(y_path), "status": "empty"})

    base_reddit: List[Dict[str, Any]] = r_data if (r_state == "valid" and isinstance(r_data, list)) else []
    base_youtube: List[Dict[str, Any]] = y_data if (y_state == "valid" and isinstance(y_data, list)) else []

    reddit_input_count = len(base_reddit)
    youtube_input_count = len(base_youtube)

    # 2. Load and Validate Analytical Output Files
    s_state, s_data, s_err = load_json_file(s_path)
    t_state, t_data, t_err = load_json_file(t_path)
    g_state, g_data, g_err = load_json_file(g_path)
    m_state, m_data, m_err = load_json_file(m_path)

    # Validate top-level structures of analytical files
    if s_state == "valid" and not isinstance(s_data, dict):
        s_state = "corrupted"
        s_err = f"Expected top-level object in sentiment results, got {type(s_data).__name__}"
    if t_state == "valid" and not isinstance(t_data, dict):
        t_state = "corrupted"
        t_err = f"Expected top-level object in topic results, got {type(t_data).__name__}"
    if m_state == "valid" and not isinstance(m_data, dict):
        m_state = "corrupted"
        m_err = f"Expected top-level object in misinformation results, got {type(m_data).__name__}"

    # Validate graph structure specifically
    if g_state == "valid":
        if not isinstance(g_data, dict):
            g_state = "corrupted"
            g_err = f"Expected top-level object in graph results, got {type(g_data).__name__}"
        elif not isinstance(g_data.get("top_influencers"), list):
            g_state = "corrupted"
            g_err = f"Expected 'top_influencers' list in graph results, got {type(g_data.get('top_influencers')).__name__}"

    for name, path, state, err in [
        ("sentiment", s_path, s_state, s_err),
        ("topics", t_path, t_state, t_err),
        ("graph", g_path, g_state, g_err),
        ("misinformation", m_path, m_state, m_err),
    ]:
        if state == "missing":
            audit_warnings.append({"module": name, "file": str(path), "status": "missing"})
        elif state == "empty":
            audit_warnings.append({"module": name, "file": str(path), "status": "empty"})
        elif state in {"malformed", "corrupted"}:
            audit_warnings.append({"module": name, "file": str(path), "status": "corrupted", "error": err})
            logger.error("Analytical file %s is corrupted: %s", path, err)

    # 3. Construct Analytical Lookup Tables (O(1) lookups) with Strict Collection Validation
    sentiment_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sentiment_platform_corrupted: Set[str] = set()
    if s_state == "valid" and isinstance(s_data, dict):
        _load_analytical_collection(
            "sentiment", s_path, s_data, "reddit_records", "reddit", "post_id",
            sentiment_map, sentiment_platform_corrupted, audit_warnings,
        )
        _load_analytical_collection(
            "sentiment", s_path, s_data, "youtube_records", "youtube", "video_id",
            sentiment_map, sentiment_platform_corrupted, audit_warnings,
        )

    topic_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    topic_platform_corrupted: Set[str] = set()
    if t_state == "valid" and isinstance(t_data, dict):
        _load_analytical_collection(
            "topics", t_path, t_data, "reddit_records", "reddit", "post_id",
            topic_map, topic_platform_corrupted, audit_warnings,
        )
        _load_analytical_collection(
            "topics", t_path, t_data, "youtube_records", "youtube", "video_id",
            topic_map, topic_platform_corrupted, audit_warnings,
        )

    misinfo_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    misinfo_platform_corrupted: Set[str] = set()
    if m_state == "valid" and isinstance(m_data, dict):
        _load_analytical_collection(
            "misinformation", m_path, m_data, "reddit_records", "reddit", "post_id",
            misinfo_map, misinfo_platform_corrupted, audit_warnings,
        )
        _load_analytical_collection(
            "misinformation", m_path, m_data, "youtube_records", "youtube", "video_id",
            misinfo_map, misinfo_platform_corrupted, audit_warnings,
        )

    # Graph strictly validates entries with platform isolation
    graph_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    graph_platform_corrupted: Set[str] = set()
    graph_has_unidentified_invalid_entries = False
    if g_state == "valid" and isinstance(g_data, dict):
        g_influencers = g_data.get("top_influencers")
        if not isinstance(g_influencers, list):
            g_state = "corrupted"
            audit_warnings.append({
                "module": "graph",
                "file": str(g_path),
                "field": "top_influencers",
                "status": "invalid_collection",
                "got": type(g_influencers).__name__,
            })
        else:
            for idx, inf in enumerate(g_influencers):
                if not isinstance(inf, dict):
                    graph_has_unidentified_invalid_entries = True
                    audit_warnings.append({
                        "module": "graph",
                        "file": str(g_path),
                        "status": "invalid_graph_entry",
                        "index": idx,
                        "reason": "entry_not_a_dictionary",
                        "entry": str(inf),
                    })
                    continue

                raw_plat = inf.get("platform")
                raw_user = inf.get("username")

                # Validate platform identity (do not infer from usernames or cross-platform matches)
                if not isinstance(raw_plat, str) or not raw_plat.strip() or raw_plat.strip().lower() not in {"reddit", "youtube"}:
                    graph_has_unidentified_invalid_entries = True
                    audit_warnings.append({
                        "module": "graph",
                        "file": str(g_path),
                        "status": "invalid_graph_entry",
                        "index": idx,
                        "reason": "missing_or_invalid_platform",
                        "entry": str(inf),
                    })
                    continue

                p_key = raw_plat.strip().lower()

                # Validate username for identifiable platform
                if not isinstance(raw_user, str) or not raw_user.strip():
                    graph_platform_corrupted.add(p_key)
                    audit_warnings.append({
                        "module": "graph",
                        "file": str(g_path),
                        "platform": p_key,
                        "status": "invalid_graph_entry",
                        "index": idx,
                        "reason": "missing_or_invalid_username",
                        "entry": str(inf),
                    })
                    continue

                u_key = raw_user.strip().lower()
                graph_map[(p_key, u_key)] = inf

    # 4. Deterministic Deduplication of Base Records
    dedup_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dedup_tiebreaker_map: Dict[Tuple[str, str], Tuple[datetime, datetime, datetime, int, int]] = {}
    duplicates_resolved = 0

    candidates = [("reddit", r) for r in base_reddit] + [("youtube", y) for y in base_youtube]

    for platform, record in candidates:
        if not isinstance(record, dict):
            rejected_records.append({"platform": platform, "reason": "record_not_a_dictionary", "record": record})
            continue

        raw_id = record.get("post_id") if platform == "reddit" else record.get("video_id")
        if raw_id is None or not isinstance(raw_id, str) or not raw_id.strip():
            rejected_records.append({
                "platform": platform,
                "reason": "missing_or_invalid_native_id",
                "record": dict(record),
            })
            continue

        native_id = raw_id.strip()
        record_key = (platform, native_id)

        # Parse timestamps for priority ranking
        _, _, dt_proc = normalize_utc_timestamp(record.get("processed_at"))
        _, _, dt_coll = normalize_utc_timestamp(record.get("collected_at"))
        raw_pub = record.get("created_utc") if platform == "reddit" else record.get("published_at")
        _, _, dt_pub = normalize_utc_timestamp(raw_pub)

        text_val = str(record.get("text_cleaned" if platform == "reddit" else "description_cleaned") or "")
        url_val = str(record.get("url") or "")
        ranking_tuple = (dt_proc, dt_coll, dt_pub, len(text_val), len(url_val))

        if record_key not in dedup_map:
            dedup_map[record_key] = record
            dedup_tiebreaker_map[record_key] = ranking_tuple
        else:
            duplicates_resolved += 1
            existing_tuple = dedup_tiebreaker_map[record_key]
            if ranking_tuple > existing_tuple:
                dedup_map[record_key] = record
                dedup_tiebreaker_map[record_key] = ranking_tuple

    # 5. Build Unified Documents
    consolidated_records: List[Dict[str, Any]] = []
    complete_count = 0
    degraded_count = 0
    partial_count = 0
    unprocessed_count = 0

    for (platform, native_id), record in dedup_map.items():
        doc_id = f"{platform}_{native_id}"
        url_val = str(record.get("url") or "")

        # Timestamps
        _, pub_iso, _ = normalize_utc_timestamp(
            record.get("created_utc") if platform == "reddit" else record.get("published_at")
        )
        _, coll_iso, _ = normalize_utc_timestamp(record.get("collected_at"))
        _, proc_iso, _ = normalize_utc_timestamp(record.get("processed_at"))

        # Author Resolution
        if platform == "reddit":
            raw_author = record.get("author")
            author_username = str(raw_author).strip() if (isinstance(raw_author, str) and raw_author.strip()) else ""
            channel_id = None
            channel_title = None
            graph_lookup_key = author_username.lower()
        else:
            ch_title = record.get("channel_title")
            ch_id = record.get("channel_id")

            # YouTube author resolution:
            # - Use channel_title only if it is a string containing non-whitespace characters
            # - Otherwise, fall back to channel_id if it is a non-empty string
            # - Strip whitespace from the selected value
            if isinstance(ch_title, str) and ch_title.strip():
                author_username = ch_title.strip()
            elif isinstance(ch_id, str) and ch_id.strip():
                author_username = ch_id.strip()
            else:
                author_username = ""

            channel_id = str(ch_id).strip() if (isinstance(ch_id, str) and ch_id.strip()) else None
            channel_title = str(ch_title).strip() if (isinstance(ch_title, str) and ch_title.strip()) else None
            graph_lookup_key = author_username.lower()

        # Graph Analytics Resolution
        influencer_score = None
        pagerank = None
        community_id = None
        in_degree = None
        out_degree = None

        if g_state in {"missing", "empty"}:
            graph_status = "unavailable"
        elif (
            g_state in {"malformed", "corrupted"}
            or graph_has_unidentified_invalid_entries
            or platform in graph_platform_corrupted
        ):
            graph_status = "corrupted"
        elif author_username.lower() in ANONYMOUS_USERS:
            graph_status = "anonymous"
        elif (platform, graph_lookup_key) in graph_map:
            graph_status = "ranked_influencer"
            inf = graph_map[(platform, graph_lookup_key)]
            influencer_score = inf.get("influencer_score")
            pagerank = inf.get("pagerank")
            community_id = inf.get("community_id")
            in_degree = inf.get("in_degree_raw")
            out_degree = inf.get("out_degree_raw")
        else:
            graph_status = "not_in_top_influencers"

        # Content Fields & Preprocessing Status (Strict: never default missing/invalid to success)
        title_raw = str(record.get("title") or "")
        body_raw = str(record.get("text" if platform == "reddit" else "description") or "")
        raw_title_clean = record.get("title_cleaned")
        raw_body_clean = record.get("text_cleaned" if platform == "reddit" else "description_cleaned")

        title_clean = str(raw_title_clean) if isinstance(raw_title_clean, str) else ""
        body_clean = str(raw_body_clean) if isinstance(raw_body_clean, str) else ""

        lang_name = str(record.get("detected_language") or "Undetermined")
        lang_code = str(record.get("language_code") or "und")

        raw_prep_status = record.get("preprocessing_status")
        if raw_prep_status is None or (isinstance(raw_prep_status, str) and not raw_prep_status.strip()):
            prep_status = "missing_status"
        elif not isinstance(raw_prep_status, str):
            prep_status = "invalid_status"
        else:
            prep_status = raw_prep_status.strip()

        # Engagement Fields
        if platform == "reddit":
            engagement = {
                "score": record.get("score"),
                "views_count": None,
                "likes_count": None,
                "comments_count": record.get("num_comments"),
            }
            platform_metadata = {
                "reddit": {"subreddit": record.get("subreddit"), "score": record.get("score")},
                "youtube": None,
            }
        else:
            engagement = {
                "score": None,
                "views_count": record.get("view_count"),
                "likes_count": record.get("like_count"),
                "comments_count": record.get("comment_count"),
            }
            platform_metadata = {
                "reddit": None,
                "youtube": {
                    "channel_id": record.get("channel_id"),
                    "channel_title": record.get("channel_title"),
                },
            }

        sent_rec = sentiment_map.get((platform, native_id))
        if s_state in {"missing", "empty"}:
            sent_status = "unavailable"
            sent_label = None
            sent_score = None
        elif s_state in {"malformed", "corrupted"} or platform in sentiment_platform_corrupted:
            sent_status = "corrupted"
            sent_label = None
            sent_score = None
        elif sent_rec is not None:
            raw_s_status = sent_rec.get("sentiment_status") if "sentiment_status" in sent_rec else sent_rec.get("status")
            if raw_s_status is None or (isinstance(raw_s_status, str) and not raw_s_status.strip()):
                sent_status = "missing_status"
                sent_label = None
                sent_score = None
            elif not isinstance(raw_s_status, str):
                sent_status = "error"
                sent_label = None
                sent_score = None
            else:
                s_clean = raw_s_status.strip()
                if s_clean == "success":
                    raw_s_label = sent_rec.get("sentiment_label")
                    raw_s_score = sent_rec.get("sentiment_score")
                    if (
                        raw_s_label in {"positive", "neutral", "negative"}
                        and _is_finite_float(raw_s_score)
                        and -1.0 <= float(raw_s_score) <= 1.0
                    ):
                        sent_status = "success"
                        sent_label = raw_s_label
                        sent_score = float(raw_s_score)
                    else:
                        sent_status = "error"
                        sent_label = None
                        sent_score = None
                elif s_clean in {"fallback", "empty_text", "error", "missing_status"}:
                    sent_status = s_clean
                    sent_label = None
                    sent_score = None
                else:
                    sent_status = "error"
                    sent_label = None
                    sent_score = None
        else:
            sent_status = "unavailable"
            sent_label = None
            sent_score = None

        top_rec = topic_map.get((platform, native_id))
        if t_state in {"missing", "empty"}:
            top_status = "unavailable"
            top_id = None
            top_prob = None
            top_keywords: List[Dict[str, Any]] = []
        elif t_state in {"malformed", "corrupted"} or platform in topic_platform_corrupted:
            top_status = "corrupted"
            top_id = None
            top_prob = None
            top_keywords = []
        elif top_rec is not None:
            raw_t_status = top_rec.get("topic_status") if "topic_status" in top_rec else top_rec.get("status")
            tid = top_rec.get("topic_id")
            raw_prob = top_rec.get("topic_probability")
            raw_kw = top_rec.get("topic_keywords", [])

            # 1. Missing or blank topic_status/status must produce missing_status
            if raw_t_status is None or (isinstance(raw_t_status, str) and not raw_t_status.strip()):
                top_status = "missing_status"
                top_id = None
                top_prob = None
                top_keywords = []
            elif not isinstance(raw_t_status, str):
                top_status = "error"
                top_id = None
                top_prob = None
                top_keywords = []
            else:
                status_clean = raw_t_status.strip()

                if status_clean == "assigned":
                    # An assigned topic MUST have a valid non-negative integer topic ID (rejecting bool, float, string, None, negative)
                    valid_id = _is_strict_int(tid) and tid >= 0
                    valid_prob = raw_prob is None or (_is_finite_float(raw_prob) and 0.0 <= float(raw_prob) <= 1.0)
                    if valid_id and valid_prob:
                        top_status = "assigned"
                        top_id = tid
                        top_prob = float(raw_prob) if raw_prob is not None else None
                        malformed_kw: List[Dict[str, Any]] = []
                        top_keywords = normalize_consolidator_topic_keywords(raw_kw, rejected_collector=malformed_kw)
                        if malformed_kw:
                            for rej in malformed_kw:
                                audit_warnings.append({
                                    "module": "topics",
                                    "platform": platform,
                                    "native_id": native_id,
                                    "field": "topic_keywords",
                                    "status": "uninterpretable_keyword_entry",
                                    "index": rej.get("index"),
                                    "reason": rej.get("reason"),
                                    "entry": rej.get("entry"),
                                })
                    else:
                        top_status = "error"
                        top_id = None
                        top_prob = None
                        top_keywords = []
                elif status_clean == "outlier":
                    # An outlier is valid ONLY when topic_status == "outlier" AND topic_id == -1
                    valid_outlier_id = _is_strict_int(tid) and tid == -1
                    valid_prob = raw_prob is None or (_is_finite_float(raw_prob) and 0.0 <= float(raw_prob) <= 1.0)
                    if valid_outlier_id and valid_prob:
                        top_status = "outlier"
                        top_id = -1
                        top_prob = float(raw_prob) if raw_prob is not None else None
                        top_keywords = []
                    else:
                        top_status = "error"
                        top_id = None
                        top_prob = None
                        top_keywords = []
                elif status_clean in {"fallback", "empty_text", "error", "missing_status"}:
                    top_status = status_clean
                    top_id = None
                    top_prob = None
                    top_keywords = []
                else:
                    # Unknown strings must not be accepted as valid statuses; map them to error
                    top_status = "error"
                    top_id = None
                    top_prob = None
                    top_keywords = []
        else:
            top_status = "unavailable"
            top_id = None
            top_prob = None
            top_keywords = []

        mis_rec = misinfo_map.get((platform, native_id))
        if m_state in {"missing", "empty"}:
            mis_status = "unavailable"
            veracity_label = None
            cred_score = None
            risk_tier = None
            stylometrics = None
        elif m_state in {"malformed", "corrupted"} or platform in misinfo_platform_corrupted:
            mis_status = "corrupted"
            veracity_label = None
            cred_score = None
            risk_tier = None
            stylometrics = None
        elif mis_rec is not None:
            raw_m_status = mis_rec.get("misinformation_status") if "misinformation_status" in mis_rec else mis_rec.get("status")
            if raw_m_status is None or (isinstance(raw_m_status, str) and not raw_m_status.strip()):
                mis_status = "missing_status"
                veracity_label = None
                cred_score = None
                risk_tier = None
                stylometrics = None
            elif not isinstance(raw_m_status, str):
                mis_status = "error"
                veracity_label = None
                cred_score = None
                risk_tier = None
                stylometrics = None
            else:
                m_clean = raw_m_status.strip()
                if m_clean == "success":
                    raw_veracity = mis_rec.get("veracity_label")
                    if str(raw_veracity).strip() in {"1", "real", "true"}:
                        raw_veracity = "real"
                    elif str(raw_veracity).strip() in {"0", "fake", "false"}:
                        raw_veracity = "fake"
                    raw_cred = mis_rec.get("credibility_score")
                    raw_risk = mis_rec.get("risk_tier")
                    raw_signals = mis_rec.get("stylometric_signals")

                    valid_veracity = raw_veracity in {"real", "fake"}
                    valid_risk = raw_risk in {"low", "moderate", "high_risk"}
                    valid_cred = _is_finite_float(raw_cred) and 0.0 <= float(raw_cred) <= 1.0
                    valid_signals = raw_signals is None or isinstance(raw_signals, dict)

                    if valid_veracity and valid_risk and valid_cred and valid_signals:
                        mis_status = "success"
                        veracity_label = raw_veracity
                        cred_score = float(raw_cred)
                        risk_tier = raw_risk
                        stylometrics = raw_signals
                    else:
                        mis_status = "error"
                        veracity_label = None
                        cred_score = None
                        risk_tier = None
                        stylometrics = None
                elif m_clean in {"fallback", "empty_text", "error", "missing_status"}:
                    mis_status = m_clean
                    veracity_label = None
                    cred_score = None
                    risk_tier = None
                    stylometrics = None
                else:
                    mis_status = "error"
                    veracity_label = None
                    cred_score = None
                    risk_tier = None
                    stylometrics = None
        else:
            mis_status = "unavailable"
            veracity_label = None
            cred_score = None
            risk_tier = None
            stylometrics = None

        # Determine Record-Level Completeness & Status
        has_usable_text = bool((isinstance(raw_title_clean, str) and raw_title_clean.strip()) or (isinstance(raw_body_clean, str) and raw_body_clean.strip()))
        has_base = bool(prep_status == "success" and has_usable_text)

        rec_status = _determine_record_consolidation_status(
            has_base=has_base,
            sent_status=sent_status,
            top_status=top_status,
            mis_status=mis_status,
        )

        # An unprocessed base record must never have is_complete: true
        is_complete = bool(has_base and rec_status == "complete")

        if rec_status == "complete":
            complete_count += 1
        elif rec_status == "degraded":
            degraded_count += 1
        elif rec_status == "partial":
            partial_count += 1
        else:
            unprocessed_count += 1

        sources_joined: List[str] = ["preprocessing"]
        if sent_status not in {"unavailable", "corrupted", "missing_status"}:
            sources_joined.append("sentiment")
        if top_status not in {"unavailable", "corrupted", "missing_status"}:
            sources_joined.append("topics")
        if graph_status not in {"unavailable", "corrupted"}:
            sources_joined.append("graph")
        if mis_status not in {"unavailable", "corrupted", "missing_status"}:
            sources_joined.append("misinformation")

        doc: Dict[str, Any] = {
            "document_id": doc_id,
            "platform": platform,
            "native_id": native_id,
            "url": url_val,
            "timestamps": {
                "published_at": pub_iso,
                "collected_at": coll_iso,
                "processed_at": proc_iso,
            },
            "author": {
                "username": author_username,
                "channel_id": channel_id,
                "channel_title": channel_title,
                "graph_status": graph_status,
                "influencer_score": influencer_score,
                "pagerank": pagerank,
                "community_id": community_id,
                "in_degree": in_degree,
                "out_degree": out_degree,
            },
            "content": {
                "title_raw": title_raw,
                "body_raw": body_raw,
                "title_cleaned": title_clean,
                "body_cleaned": body_clean,
                "detected_language": lang_name,
                "language_code": lang_code,
                "preprocessing_status": prep_status,
            },
            "engagement": engagement,
            "metadata": platform_metadata,
            "sentiment": {
                "label": sent_label,
                "score": sent_score,
                "status": sent_status,
            },
            "topics": {
                "topic_id": top_id,
                "probability": top_prob,
                "keywords": top_keywords,
                "status": top_status,
            },
            "misinformation": {
                "veracity_label": veracity_label,
                "credibility_score": cred_score,
                "risk_tier": risk_tier,
                "status": mis_status,
                "stylometric_signals": stylometrics,
            },
            "consolidation": {
                "status": rec_status,
                "is_complete": is_complete,
                "sources_joined": sources_joined,
            },
        }

        consolidated_records.append(doc)

    # 6. Determine Dataset-Wide Overall Status (Precedence: unprocessed > partial > degraded > complete)
    reddit_usable_count = sum(1 for d in consolidated_records if d["platform"] == "reddit")
    youtube_usable_count = sum(1 for d in consolidated_records if d["platform"] == "youtube")

    has_base_anomaly = bool(
        r_state in {"missing", "empty", "malformed"}
        or y_state in {"missing", "empty", "malformed"}
    )

    # Rejection impact: if a platform had inputs but all were rejected, or if any records were rejected
    has_platform_total_rejection = bool(
        (reddit_input_count > 0 and reddit_usable_count == 0)
        or (youtube_input_count > 0 and youtube_usable_count == 0)
    )

    if len(consolidated_records) == 0 or unprocessed_count == len(consolidated_records):
        overall_status = "unprocessed"
    elif unprocessed_count > 0:
        overall_status = "unprocessed"
    elif partial_count > 0 or has_base_anomaly or has_platform_total_rejection or len(rejected_records) > 0 or any(w.get("status") in {"missing", "empty", "malformed", "corrupted", "invalid_collection", "missing_collection_key", "invalid_record_entry", "invalid_graph_entry"} for w in audit_warnings):
        overall_status = "partial"
    elif degraded_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "complete"

    # Resolve execution timestamp (deterministic override supported)
    gen_at = execution_time if execution_time else datetime.now(timezone.utc).isoformat()

    # Map input source states to provenance statuses
    state_to_prov_status = {
        "valid": "loaded",
        "empty": "empty",
        "missing": "missing",
        "malformed": "corrupted",
    }
    reddit_prov_status = state_to_prov_status.get(r_state, "corrupted")
    youtube_prov_status = state_to_prov_status.get(y_state, "corrupted")

    provenance: Dict[str, Any] = {
        "reddit_input_file": str(r_path),
        "reddit_status": reddit_prov_status,
        "reddit_records_count": reddit_input_count if reddit_prov_status == "loaded" else 0,
        "youtube_input_file": str(y_path),
        "youtube_status": youtube_prov_status,
        "youtube_records_count": youtube_input_count if youtube_prov_status == "loaded" else 0,
    }

    output_payload: Dict[str, Any] = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "status": "success",
        "consolidation_status": overall_status,
        "generated_at": gen_at,
        "provenance": provenance,
        "metrics": {
            "total_records_consolidated": len(consolidated_records),
            "platform_counts": {"reddit": reddit_usable_count, "youtube": youtube_usable_count},
            "complete_records": complete_count,
            "degraded_records": degraded_count,
            "partial_records": partial_count,
            "unprocessed_records": unprocessed_count,
            "duplicates_deduplicated": duplicates_resolved,
            "rejected_records_count": len(rejected_records),
        },
        "audit_warnings": audit_warnings,
        "rejected_records": rejected_records,
        "records": consolidated_records,
    }

    # Atomic, validated persistence
    _promote_artifact_atomically(output_payload, dest_path)

    logger.info(
        "Consolidation complete. %d records persisted to %s with status '%s'.",
        len(consolidated_records),
        dest_path,
        overall_status,
    )

    return output_payload

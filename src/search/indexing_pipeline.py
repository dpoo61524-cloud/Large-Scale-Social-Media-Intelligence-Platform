"""Elasticsearch Indexing Pipeline Runner (Stage 11).

Orchestrates end-to-end Elasticsearch indexing for validated Schema-v2 consolidated
intelligence datasets. Provides parameter validation, pre-validation checks,
bounded bulk ingestion, safe error categorization, atomic execution manifest
persistence, and explicit Elasticsearch client lifecycle management.

Key Invariants:
- Standalone execution: run_indexing_pipeline() callable directly or via CLI.
- Dependency injection: dataset_path, client, index_name, refresh, ensure_index,
  batch_size, validate_envelope, manifest_path, and client_factory.
- Client ownership: Callers providing an external client retain ownership (client
  is left open). Pipeline-owned clients (e.g. from client_factory) are closed
  guaranteed in a finally block.
- Error categorization: Structured error codes and categories for operational clarity.
- Non-zero exit codes: Returns exit code 0 on complete success or empty dataset;
  returns exit code 1 on failures or partial successes.
- Atomic manifest persistence: Written to temporary file on same filesystem with
  read-back verification and atomic replacement via os.replace.
- Completely offline testable with zero live network dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

from src.search.elasticsearch_client import (
    DEFAULT_BULK_BATCH_SIZE,
    ElasticsearchClient,
    validate_index_name,
)
from src.search.index_mapping import DEFAULT_INDEX_NAME
from src.search.indexing_adapter import index_consolidated_dataset

logger = logging.getLogger(__name__)

# Canonical default paths
DEFAULT_DATASET_PATH = Path("data/processed/consolidated_intelligence.json")
DEFAULT_INDEXING_MANIFEST_PATH = Path("data/processed/elasticsearch_indexing_manifest.json")

# Exit codes
EXIT_SUCCESS: int = 0
EXIT_FAILURE: int = 1

# Supported manifest schemas and categories
PERMITTED_MANIFEST_STATUSES = frozenset({"success", "empty", "partial_success", "failed"})

PERMITTED_ERROR_CATEGORIES = frozenset({
    "dataset_not_found",
    "dataset_empty",
    "corrupted_dataset",
    "schema_validation_failed",
    "elasticsearch_unreachable",
    "index_initialization_failed",
    "client_initialization_failed",
    "partial_indexing_failure",
    "all_records_rejected",
    "manifest_persistence_failed",
    "pipeline_execution_error",
    "indexing_failed",
})

REQUIRED_MANIFEST_FIELDS = frozenset({
    "stage",
    "status",
    "dataset_path",
    "target_index",
    "total_records",
    "indexed_count",
    "rejected_count",
    "failed_count",
    "started_at",
    "completed_at",
    "duration_seconds",
    "rejections",
    "indexing_failures",
    "error_category",
    "error_message",
})


class IndexingManifestValidator:
    """Validates structural and numerical integrity of indexing execution manifests."""

    @classmethod
    def validate_manifest(cls, manifest: Any) -> None:
        """Validates an indexing manifest dictionary against strict schema contracts.

        Raises:
            TypeError: If manifest is not a dictionary.
            ValueError: If manifest fails schema or numerical invariants.
        """
        if not isinstance(manifest, dict):
            raise TypeError(f"Manifest root must be a dict, got {type(manifest).__name__}")

        # Check required fields
        missing_fields = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
        if missing_fields:
            raise ValueError(f"Manifest missing required fields: {sorted(missing_fields)}")

        # Validate stage
        stage = manifest.get("stage")
        if stage != "elasticsearch_indexing":
            raise ValueError(f"Invalid manifest stage {stage!r}; expected 'elasticsearch_indexing'")

        # Validate status
        status = manifest.get("status")
        if status not in PERMITTED_MANIFEST_STATUSES:
            raise ValueError(
                f"Invalid manifest status {status!r}; expected one of {sorted(PERMITTED_MANIFEST_STATUSES)}"
            )

        # Validate target_index
        target_index = manifest.get("target_index")
        if not isinstance(target_index, str) or not target_index.strip():
            raise ValueError(f"target_index must be a non-empty string, got {target_index!r}")

        # Validate dataset_path
        dataset_path = manifest.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError(f"dataset_path must be a non-empty string, got {dataset_path!r}")

        # Validate counts (reject booleans explicitly)
        for count_field in ("indexed_count", "rejected_count", "failed_count"):
            val = manifest.get(count_field)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"'{count_field}' must be a non-negative integer, got {val!r}")

        indexed_count = manifest["indexed_count"]
        rejected_count = manifest["rejected_count"]
        failed_count = manifest["failed_count"]

        # Validate total_records
        total_records = manifest.get("total_records")
        if total_records is not None:
            if isinstance(total_records, bool) or not isinstance(total_records, int) or total_records < 0:
                raise ValueError(f"'total_records' must be a non-negative integer or None, got {total_records!r}")

            # Numerical count invariant when total_records is known
            expected_sum = indexed_count + rejected_count + failed_count
            if total_records != expected_sum:
                raise ValueError(
                    f"Count invariant violated: total_records ({total_records}) != "
                    f"indexed_count ({indexed_count}) + rejected_count ({rejected_count}) + failed_count ({failed_count})"
                )

        # Validate timestamps
        for ts_field in ("started_at", "completed_at"):
            ts_val = manifest.get(ts_field)
            if not isinstance(ts_val, str) or not ts_val.strip():
                raise ValueError(f"'{ts_field}' must be a non-empty string, got {ts_val!r}")
            try:
                datetime.fromisoformat(ts_val)
            except Exception as exc:
                raise ValueError(f"Invalid ISO-8601 timestamp for '{ts_field}': {ts_val!r} ({exc})")

        # Validate duration_seconds
        duration = manifest.get("duration_seconds")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
            or not math.isfinite(duration)
        ):
            raise ValueError(f"'duration_seconds' must be a finite non-negative number, got {duration!r}")

        # Validate lists
        if not isinstance(manifest.get("rejections"), list):
            raise ValueError(f"'rejections' must be a list, got {type(manifest.get('rejections')).__name__}")
        if not isinstance(manifest.get("indexing_failures"), list):
            raise ValueError(f"'indexing_failures' must be a list, got {type(manifest.get('indexing_failures')).__name__}")

        # Validate error_category & error_message
        error_category = manifest.get("error_category")
        error_message = manifest.get("error_message")

        if error_category is not None:
            if error_category not in PERMITTED_ERROR_CATEGORIES:
                raise ValueError(
                    f"Invalid error_category {error_category!r}; expected one of {sorted(PERMITTED_ERROR_CATEGORIES)}"
                )

        # Logical consistency checks
        if status in ("success", "empty"):
            if error_category is not None:
                raise ValueError(f"error_category must be None when status is {status!r}, got {error_category!r}")
            if error_message is not None:
                raise ValueError(f"error_message must be None when status is {status!r}, got {error_message!r}")

            if status == "empty":
                if total_records != 0:
                    raise ValueError(f"total_records must be 0 when status is 'empty', got {total_records!r}")
                if indexed_count != 0 or rejected_count != 0 or failed_count != 0:
                    raise ValueError(
                        f"All counts must be 0 when status is 'empty', got "
                        f"indexed={indexed_count}, rejected={rejected_count}, failed={failed_count}"
                    )
            elif status == "success":
                if total_records is None or total_records <= 0:
                    raise ValueError(f"total_records must be an integer > 0 when status is 'success', got {total_records!r}")
                if indexed_count != total_records:
                    raise ValueError(
                        f"indexed_count ({indexed_count}) must equal total_records ({total_records}) when status is 'success'"
                    )
                if rejected_count != 0 or failed_count != 0:
                    raise ValueError(
                        f"rejected_count and failed_count must be 0 when status is 'success', got "
                        f"rejected={rejected_count}, failed={failed_count}"
                    )

        elif status == "partial_success":
            if error_category is None:
                raise ValueError("error_category must be populated when status is 'partial_success'")
            if not isinstance(error_message, str) or not error_message.strip():
                raise ValueError("error_message must be a non-empty string when status is 'partial_success'")
            if indexed_count <= 0:
                raise ValueError(f"indexed_count must be > 0 when status is 'partial_success', got {indexed_count}")
            if total_records is None:
                raise ValueError("total_records must be known when status is 'partial_success'")
            if rejected_count == 0 and failed_count == 0:
                raise ValueError("Either rejected_count or failed_count must be > 0 when status is 'partial_success'")

        elif status == "failed":
            if error_category is None:
                raise ValueError("error_category must be populated when status is 'failed'")
            if not isinstance(error_message, str) or not error_message.strip():
                raise ValueError("error_message must be a non-empty string when status is 'failed'")
            # If error is not a post-indexing persistence failure, indexed_count must be 0
            if error_category != "manifest_persistence_failed" and indexed_count != 0:
                raise ValueError(f"indexed_count must be 0 when status is 'failed', got {indexed_count}")


def validate_indexing_manifest(manifest: Any) -> None:
    """Validates an indexing execution manifest dictionary against strict schema contracts."""
    IndexingManifestValidator.validate_manifest(manifest)


def save_manifest_atomically(
    manifest_data: Dict[str, Any],
    dest_path: Union[str, Path],
) -> Path:
    """Atomically serializes, validates, and promotes an execution manifest.

    Guarantees that readers of dest_path will never observe a partially written
    or corrupt file. If any error occurs prior to or during replacement, any
    existing canonical file is untouched and temporary files are cleaned up.

    Args:
        manifest_data: Manifest dictionary to persist.
        dest_path: Canonical destination path.

    Returns:
        Path object of persisted destination.

    Raises:
        TypeError: If manifest_data is not a dictionary.
        ValueError: If read-back verification fails.
        OSError: If filesystem operations fail.
    """
    target_path = Path(dest_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    unique_id = uuid.uuid4().hex
    temp_path = target_path.parent / f".tmp_{target_path.name}_{unique_id}"

    try:
        serialized = json.dumps(manifest_data, indent=2, ensure_ascii=False)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (AttributeError, OSError):
                pass

        # Read-back verification before atomic replacement
        with open(temp_path, "r", encoding="utf-8") as rf:
            read_back_payload = json.load(rf)

        IndexingManifestValidator.validate_manifest(read_back_payload)

        os.replace(temp_path, target_path)
        logger.info("Indexing execution manifest successfully persisted to %s", target_path)
        return target_path
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as exc:
                logger.debug("Failed to clean up temporary manifest file %s: %s", temp_path, exc)


def _categorize_error(summary: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Determines error category and descriptive error message from adapter summary."""
    status = summary.get("status", "failed")
    failures = summary.get("indexing_failures", [])
    rejections = summary.get("rejections", [])
    indexed_count = summary.get("indexed_count", 0)
    rejected_count = summary.get("rejected_count", 0)
    failed_count = summary.get("failed_count", 0)

    if status in ("success", "empty"):
        return None, None

    if status == "partial_success":
        category = "partial_indexing_failure"
        message = (
            f"Indexing partially succeeded: {indexed_count} indexed, "
            f"{rejected_count} rejected, {failed_count} failed."
        )
        return category, message

    # Failure path: inspect failures list
    if failures:
        first_failure = failures[0]
        first_reason = str(first_failure.get("reason", "Unknown failure"))
        r_lower = first_reason.lower()

        if "file not found" in r_lower or "dataset file not found" in r_lower:
            category = "dataset_not_found"
        elif "dataset file is empty" in r_lower or "is empty" in r_lower:
            category = "dataset_empty"
        elif "failed to parse json" in r_lower or "json" in r_lower:
            category = "corrupted_dataset"
        elif "schema-v2" in r_lower or "schema" in r_lower or "envelope" in r_lower:
            category = "schema_validation_failed"
        elif "preflight ping" in r_lower or "ping" in r_lower:
            category = "elasticsearch_unreachable"
        elif "index creation" in r_lower or "index verification" in r_lower or "mapping" in r_lower or "target index" in r_lower:
            category = "index_initialization_failed"
        elif "client initialization" in r_lower or "client" in r_lower:
            category = "client_initialization_failed"
        elif "pipeline execution error" in r_lower:
            category = "pipeline_execution_error"
        else:
            category = "indexing_failed"

        return category, first_reason

    if rejected_count > 0 and indexed_count == 0:
        return "all_records_rejected", f"All {rejected_count} candidate records were rejected by adapter validation."

    return "indexing_failed", "Indexing pipeline failed with unrecorded failure cause."


def _build_manifest(
    summary: Dict[str, Any],
    dataset_path: Union[str, Path],
    target_index: str,
) -> Dict[str, Any]:
    """Assembles a structured Stage 11 execution manifest dictionary."""
    status = summary.get("status", "failed")
    error_category, error_message = _categorize_error(summary)

    started_at = summary.get("started_at") or datetime.now(timezone.utc).isoformat()
    completed_at = summary.get("completed_at") or datetime.now(timezone.utc).isoformat()
    raw_dur = summary.get("duration_seconds", 0.0)
    if isinstance(raw_dur, (int, float)) and not isinstance(raw_dur, bool) and math.isfinite(raw_dur) and raw_dur >= 0:
        duration_seconds = round(float(raw_dur), 4)
    else:
        duration_seconds = 0.0

    rejections = summary.get("rejections") if isinstance(summary.get("rejections"), list) else []
    indexing_failures = (
        summary.get("indexing_failures") if isinstance(summary.get("indexing_failures"), list) else []
    )

    return {
        "stage": "elasticsearch_indexing",
        "status": status,
        "dataset_path": str(dataset_path),
        "target_index": str(target_index),
        "total_records": summary.get("total_records"),
        "indexed_count": summary.get("indexed_count", 0),
        "rejected_count": summary.get("rejected_count", 0),
        "failed_count": summary.get("failed_count", 0),
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_seconds": duration_seconds,
        "rejections": rejections,
        "indexing_failures": indexing_failures,
        "error_category": error_category,
        "error_message": error_message,
    }


def run_indexing_pipeline(
    dataset_path: Union[str, Path] = DEFAULT_DATASET_PATH,
    client: Optional[ElasticsearchClient] = None,
    index_name: str = DEFAULT_INDEX_NAME,
    refresh: Optional[Union[str, bool]] = False,
    ensure_index: bool = True,
    batch_size: int = DEFAULT_BULK_BATCH_SIZE,
    validate_envelope: bool = True,
    manifest_path: Optional[Union[str, Path]] = None,
    client_factory: Optional[Callable[[], ElasticsearchClient]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Executes Stage 11 Elasticsearch Indexing Pipeline.

    Integrates input validation, client management, dataset ingestion,
    error categorization, structured manifest generation, and atomic persistence.

    Lifecycle Management:
    - If `client` is provided: Caller retains ownership; client is not closed.
    - If `client_factory` is provided: Pipeline acquires client and guarantees
      its closure in a `finally` block.
    - If neither is provided: Pipeline delegates client lifecycle to `index_consolidated_dataset`.

    Args:
        dataset_path: Path to consolidated intelligence JSON dataset.
        client: Optional preconfigured ElasticsearchClient instance.
        index_name: Target Elasticsearch index name.
        refresh: Refresh parameter ('wait_for', True, False, or None).
        ensure_index: Whether to verify or create index mappings before indexing.
        batch_size: Bounded batch size for bulk indexing.
        validate_envelope: Whether to strictly validate Schema-v2 envelope metadata (default: True).
        manifest_path: Optional destination path for persisting execution manifest.
        client_factory: Optional callable returning a new ElasticsearchClient instance.

    Returns:
        Tuple of (exit_code: int, manifest: Dict[str, Any]).
        exit_code is 0 for "success" or "empty", 1 for "failed" or "partial_success".
    """
    start_dt = datetime.now(timezone.utc)
    started_at = start_dt.isoformat()

    owned_client: Optional[ElasticsearchClient] = None
    effective_client: Optional[ElasticsearchClient] = client

    # Validate target index name format early
    try:
        validated_index_name = validate_index_name(index_name or DEFAULT_INDEX_NAME)
    except Exception as exc:
        logger.error("Invalid target index name %r: %s", index_name, exc)
        completed_at = datetime.now(timezone.utc).isoformat()
        duration = round((datetime.now(timezone.utc) - start_dt).total_seconds(), 4)
        failure_summary = {
            "status": "failed",
            "index_name": str(index_name),
            "total_records": None,
            "indexed_count": 0,
            "rejected_count": 0,
            "failed_count": 0,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration,
            "rejections": [],
            "indexing_failures": [{"status": "failed", "reason": f"Invalid target index name: {exc}"}],
        }
        manifest = _build_manifest(failure_summary, dataset_path, str(index_name))
        if manifest_path is not None:
            try:
                save_manifest_atomically(manifest, manifest_path)
            except Exception as p_exc:
                logger.error("Failed to persist failure manifest: %s", p_exc)
        return EXIT_FAILURE, manifest

    try:
        # Acquire client from factory if injected and no client provided
        if effective_client is None and client_factory is not None:
            try:
                effective_client = client_factory()
                owned_client = effective_client
            except Exception as exc:
                logger.error("Client factory failed to create client: %s", exc)
                completed_at = datetime.now(timezone.utc).isoformat()
                duration = round((datetime.now(timezone.utc) - start_dt).total_seconds(), 4)
                failure_summary = {
                    "status": "failed",
                    "index_name": validated_index_name,
                    "total_records": None,
                    "indexed_count": 0,
                    "rejected_count": 0,
                    "failed_count": 0,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": duration,
                    "rejections": [],
                    "indexing_failures": [
                        {"status": "failed", "reason": f"Client initialization failed: {type(exc).__name__}: {exc}"}
                    ],
                }
                manifest = _build_manifest(failure_summary, dataset_path, validated_index_name)
                if manifest_path is not None:
                    try:
                        save_manifest_atomically(manifest, manifest_path)
                    except Exception as p_exc:
                        logger.error("Failed to persist failure manifest: %s", p_exc)
                return EXIT_FAILURE, manifest

        # Execute indexing through adapter
        try:
            summary = index_consolidated_dataset(
                dataset_path=dataset_path,
                client=effective_client,
                index_name=validated_index_name,
                refresh=refresh,
                ensure_index=ensure_index,
                batch_size=batch_size,
                validate_envelope=validate_envelope,
            )
        except Exception as exc:
            logger.error("Indexing adapter raised unhandled exception (%s): %s", type(exc).__name__, exc)
            completed_at = datetime.now(timezone.utc).isoformat()
            duration = round((datetime.now(timezone.utc) - start_dt).total_seconds(), 4)
            summary = {
                "status": "failed",
                "index_name": validated_index_name,
                "total_records": None,
                "indexed_count": 0,
                "rejected_count": 0,
                "failed_count": 0,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration,
                "rejections": [],
                "indexing_failures": [
                    {"status": "failed", "reason": f"Pipeline execution error: {type(exc).__name__}: {exc}"}
                ],
            }

        # Build structured execution manifest
        manifest = _build_manifest(summary, dataset_path, validated_index_name)

        # Determine exit code: 0 for success/empty, 1 for failed/partial_success
        status = manifest.get("status")
        if status in ("success", "empty"):
            exit_code = EXIT_SUCCESS
        else:
            exit_code = EXIT_FAILURE

        # Persist manifest if destination provided
        if manifest_path is not None:
            try:
                save_manifest_atomically(manifest, manifest_path)
            except Exception as exc:
                logger.error("Failed to persist execution manifest to %s: %s", manifest_path, exc)
                manifest["status"] = "failed"
                manifest["error_category"] = "manifest_persistence_failed"
                manifest["error_message"] = f"Failed to persist manifest ({type(exc).__name__}): {exc}"
                exit_code = EXIT_FAILURE

        return exit_code, manifest

    finally:
        # Guarantee closure of pipeline-owned client
        if owned_client is not None:
            try:
                owned_client.close()
            except Exception as exc:
                logger.debug("Error closing internally owned Elasticsearch client: %s", exc)

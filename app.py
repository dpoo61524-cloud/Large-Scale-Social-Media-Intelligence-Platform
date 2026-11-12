"""Large Scale Social Media Intelligence Platform.

Main entry point for the intelligence platform.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import uuid

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# High-byte offset used for Windows OS advisory locking without interfering with content reads
LOCK_OFFSET: int = 1_000_000

from config.settings import AppConfig, load_config
from src.graph.graph_pipeline import (
    InteractionGraphBuilder,
    InfluencerRanker,
    run_graph_pipeline,
)
from src.graph.neo4j_connector import Neo4jGraphManager
from src.graph.neo4j_pipeline import run_neo4j_persistence
from src.ingestion.reddit_ingestion import ingest_reddit_posts, ingest_reddit_live
from src.ingestion.youtube_ingestion import ingest_youtube_live, validate_youtube_video
from src.misinformation.classifier import MisinformationClassifier
from src.misinformation.detector import MisinformationDetector
from src.misinformation.feature_extractor import LinguisticFeatureExtractor
from src.misinformation.misinformation_pipeline import run_misinformation_pipeline
from src.pipeline.consolidator import (
    ConsolidatedArtifactValidator,
    consolidate_social_intelligence,
)
from src.preprocessing.preprocessing_pipeline import (
    preprocess_reddit_dataset,
    preprocess_youtube_dataset,
)
from src.search.elasticsearch_client import ElasticsearchClient
from src.search.indexing_pipeline import (
    DEFAULT_INDEX_NAME,
    DEFAULT_INDEXING_MANIFEST_PATH,
    run_indexing_pipeline,
)
from src.search.kibana_dashboard_validator import (
    DEFAULT_BUNDLE_PATH,
    validate_kibana_dashboard_bundle,
)
from src.sentiment.sentiment_analyzer import MultilingualSentimentAnalyzer
from src.sentiment.sentiment_pipeline import run_sentiment_pipeline
from src.topics.topic_modeler import SocialTopicModeler
from src.topics.topic_pipeline import run_topic_pipeline

logger = logging.getLogger(__name__)

# =====================================================================
# Canonical 12-Stage Registry & Execution Manifest Foundation
# =====================================================================

CANONICAL_STAGES: tuple[str, ...] = (
    "reddit_ingestion",
    "youtube_ingestion",
    "reddit_preprocessing",
    "youtube_preprocessing",
    "sentiment",
    "topics",
    "graph",
    "neo4j_persistence",
    "misinformation",
    "consolidation",
    "elasticsearch_indexing",
    "kibana_validation",
)

PERMITTED_STAGE_STATUSES: frozenset[str] = frozenset(
    {"success", "partial", "empty", "failed", "skipped"}
)

DEFAULT_MANIFEST_PATH: Path = Path("data/processed/pipeline_execution_manifest.json")
DEFAULT_RUNS_DIR: Path = Path("data/runs")
DEFAULT_LOCK_PATH: Path = Path("data/.orchestration.lock")

EXIT_SUCCESS = 0
EXIT_STAGE_FAILURE = 1
EXIT_MANIFEST_PERSISTENCE_FAILURE = 2
EXIT_ORCHESTRATION_NOT_IMPLEMENTED = 3
EXIT_LOCK_ACQUISITION_FAILURE = 4

RUN_ID_REGEX: str = r"^run_[0-9]{8}_[0-9]{6}_[0-9a-f]{8}$"
RUN_ID_PATTERN: re.Pattern[str] = re.compile(RUN_ID_REGEX)


class ManifestPersistenceError(Exception):
    """Raised when the execution manifest cannot be atomically persisted."""


class OrchestrationLockError(RuntimeError):
    """Raised when the orchestration lock cannot be acquired or is held by an active process."""


class CanonicalPromotionError(RuntimeError):
    """Raised when canonical artifact promotion fails."""


def generate_run_id() -> str:
    """Generates a canonical validated run_id with UTC timestamp and random hex suffix."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{now_str}_{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> bool:
    """Validates that a run_id matches the canonical regex format."""
    return bool(isinstance(run_id, str) and RUN_ID_PATTERN.match(run_id))


def is_pid_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is currently active.

    Handles Windows and POSIX error conventions safely without false negatives.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as err:
        # Windows: error 87 is ERROR_INVALID_PARAMETER (PID does not exist)
        if getattr(err, "winerror", None) == 87:
            return False
        if err.errno == errno.ESRCH:
            return False
        # If EPERM (operation not permitted), process exists but owned by other user -> ALIVE
        if err.errno == errno.EPERM:
            return True
        return False


class OrchestrationLock:
    """Operating-system-backed exclusive advisory lock for pipeline execution.

    Uses OS kernel-enforced advisory locking (msvcrt.locking on Windows,
    fcntl.flock on POSIX) as the definitive authority for concurrency control,
    preventing TOCTOU races in stale-lock recovery while maintaining diagnostic metadata.
    """

    def __init__(self, lock_path: Optional[Path | str] = None) -> None:
        self.lock_path: Path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
        self.run_id: Optional[str] = None
        self._acquired: bool = False
        self._fd: Optional[int] = None

    @property
    def is_acquired(self) -> bool:
        return self._acquired

    def acquire(self, run_id: str) -> None:
        """Acquires the exclusive orchestration lock using OS advisory locking."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        pid = os.getpid()
        hostname = socket.gethostname()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Pre-acquisition safety inspection for uncertain/corrupt metadata
        # If the file already exists on disk and is non-empty, verify its metadata integrity first.
        # This guarantees that corrupt files require manual intervention and cannot be unsafely overwritten.
        existing_info: dict[str, Any] = {}
        if self.lock_path.exists() and self.lock_path.stat().st_size > 0:
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    existing_info = json.loads(content)
                    if not isinstance(existing_info, dict):
                        raise ValueError("Lock file root is not a dictionary.")
                    existing_pid = existing_info.get("pid")
                    if isinstance(existing_pid, bool) or not isinstance(existing_pid, int) or existing_pid <= 0:
                        raise ValueError(f"Invalid lock file PID: {existing_pid!r}")
            except Exception as verify_err:
                raise OrchestrationLockError(
                    f"Existing orchestration lock at {self.lock_path} could not be safely verified ({verify_err}). "
                    "Manual inspection or intervention required to prevent concurrent execution conflicts."
                ) from verify_err

        # Step 2: Open file descriptor
        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as open_err:
            raise OrchestrationLockError(
                f"Failed to open orchestration lock file at {self.lock_path}: {open_err}"
            ) from open_err

        # Step 3: Attempt non-blocking OS advisory lock
        locked = False
        try:
            if HAS_MSVCRT:
                os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBRLCK, 1)
                    locked = True
                except (OSError, IOError):
                    locked = False
            elif HAS_FCNTL:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except (OSError, IOError):
                    locked = False
            else:
                os.close(fd)
                raise OrchestrationLockError("No supported OS advisory locking mechanism found on this platform.")

            if not locked:
                os.close(fd)
                owner_pid = existing_info.get("pid", "active_process")
                owner_run = existing_info.get("run_id", "unknown")
                owner_ts = existing_info.get("timestamp_utc", "unknown")
                raise OrchestrationLockError(
                    f"Orchestration pipeline is currently locked by active PID {owner_pid} "
                    f"(run_id: '{owner_run}', acquired: {owner_ts})."
                )

            # Successfully acquired OS advisory lock!
            # If existing_info showed a dead PID, log that we safely reclaimed it without unlinking
            if existing_info:
                old_pid = existing_info.get("pid")
                if isinstance(old_pid, int) and not is_pid_alive(old_pid):
                    logger.warning(
                        "Stale orchestration lock detected from dead PID %d (run %s). Reclaiming via OS lock.",
                        old_pid,
                        existing_info.get("run_id"),
                    )

            # Step 4: Write lock metadata while holding the OS lock
            lock_data = {
                "pid": pid,
                "run_id": run_id,
                "timestamp_utc": now_iso,
                "hostname": hostname,
            }
            lock_json = json.dumps(lock_data, indent=2) + "\n"
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, lock_json.encode("utf-8"))

            self._fd = fd
            self._acquired = True
            logger.info("Acquired exclusive OS orchestration lock at %s for run %s", self.lock_path, run_id)

        except OrchestrationLockError:
            raise
        except Exception as exc:
            try:
                os.close(fd)
            except Exception:
                pass
            raise OrchestrationLockError(
                f"Unexpected error acquiring orchestration lock at {self.lock_path}: {exc}"
            ) from exc

    def release(self) -> None:
        """Releases the orchestration lock if owned by this process and run_id."""
        if not self._acquired or self._fd is None:
            return
        fd = self._fd
        self._fd = None
        self._acquired = False

        try:
            # Release OS advisory lock
            if HAS_MSVCRT:
                try:
                    os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except Exception as exc:
                    logger.debug("Error releasing msvcrt lock: %s", exc)
            elif HAS_FCNTL:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception as exc:
                    logger.debug("Error releasing fcntl lock: %s", exc)

            # Close file descriptor
            try:
                os.close(fd)
            except Exception as exc:
                logger.debug("Error closing lock fd: %s", exc)

            # Safely remove lock file if it still belongs to this run
            try:
                if self.lock_path.exists():
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("run_id") == self.run_id and data.get("pid") == os.getpid():
                        self.lock_path.unlink(missing_ok=True)
                        logger.info("Released orchestration lock at %s", self.lock_path)
            except Exception as exc:
                logger.debug("Could not verify or unlink lock file on release: %s", exc)
        except Exception as exc:
            logger.warning("Unexpected error during lock release: %s", exc)

    def __enter__(self) -> OrchestrationLock:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


def create_initial_stage_entry(
    status: str = "skipped",
    reason: Optional[str] = "stage_not_requested",
    record_count: int = 0,
    duration_seconds: float = 0.0,
    error_category: Optional[str] = None,
) -> dict[str, Any]:
    """Creates a validated stage entry dictionary.

    Args:
        status: One of 'success', 'partial', 'empty', 'failed', 'skipped'.
        reason: Optional skip or failure reason string.
        record_count: Non-negative integer of processed/validated records.
        duration_seconds: Non-negative float of execution duration.
        error_category: Optional safe error classification string.

    Returns:
        Structured stage entry dictionary.

    Raises:
        ValueError: If status, record_count, or duration_seconds is invalid.
    """
    if status not in PERMITTED_STAGE_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Permitted: {sorted(PERMITTED_STAGE_STATUSES)}."
        )
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise ValueError(f"record_count must be a non-negative integer, got {record_count!r}.")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or duration_seconds < 0
        or (isinstance(duration_seconds, float) and not math.isfinite(duration_seconds))
    ):
        raise ValueError(
            f"duration_seconds must be a finite, non-negative number, got {duration_seconds!r}."
        )

    if isinstance(duration_seconds, float):
        stored_duration: float | int = round(duration_seconds, 4)
    else:
        try:
            stored_duration = round(float(duration_seconds), 4)
        except OverflowError:
            stored_duration = duration_seconds

    return {
        "status": status,
        "record_count": record_count,
        "duration_seconds": stored_duration,
        "error_category": error_category,
        "reason": reason,
    }


def create_empty_stage_registry(
    default_status: str = "skipped",
    default_reason: Optional[str] = "stage_not_requested",
) -> dict[str, dict[str, Any]]:
    """Creates a fresh 12-stage registry with default status and reason."""
    return {
        stage_name: create_initial_stage_entry(
            status=default_status,
            reason=default_reason,
            record_count=0,
            duration_seconds=0.0,
            error_category=None,
        )
        for stage_name in CANONICAL_STAGES
    }


def calculate_manifest_summary(stages: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Calculates summary counts directly from the 12 stage entries.

    Args:
        stages: Dictionary mapping stage names to stage entry dictionaries.

    Returns:
        Dictionary containing total_stages_evaluated, total_stages_attempted,
        stages_succeeded, stages_failed, and stages_skipped.

    Raises:
        ValueError: If stages count != 12, unknown keys exist, or arithmetic identities fail.
        TypeError: If a stage value is not a dictionary.
    """
    if len(stages) != 12:
        raise ValueError(f"Expected exactly 12 stages, got {len(stages)}.")

    missing_stages = set(CANONICAL_STAGES) - set(stages)
    unknown_stages = set(stages) - set(CANONICAL_STAGES)
    if missing_stages or unknown_stages:
        raise ValueError(
            f"Stage registry mismatch: missing={sorted(missing_stages)}, unknown={sorted(unknown_stages)}"
        )

    for stage_name, stage_data in stages.items():
        if not isinstance(stage_data, dict):
            raise TypeError(f"Stage '{stage_name}' data must be a dict, got {type(stage_data).__name__}.")
        status = stage_data.get("status")
        if status not in PERMITTED_STAGE_STATUSES:
            raise ValueError(
                f"Stage '{stage_name}' has invalid status '{status}'. Permitted: {sorted(PERMITTED_STAGE_STATUSES)}."
            )

    stages_succeeded = sum(stage["status"] == "success" for stage in stages.values())
    stages_partial = sum(stage["status"] == "partial" for stage in stages.values())
    stages_empty = sum(stage["status"] == "empty" for stage in stages.values())
    stages_failed = sum(stage["status"] == "failed" for stage in stages.values())
    stages_skipped = sum(stage["status"] == "skipped" for stage in stages.values())

    total_stages_attempted = (
        stages_succeeded + stages_partial + stages_empty + stages_failed
    )
    total_stages_evaluated = total_stages_attempted + stages_skipped

    if total_stages_evaluated != 12:
        raise ValueError(f"Total evaluated stages must equal 12, got {total_stages_evaluated}.")
    if total_stages_attempted + stages_skipped != 12:
        raise ValueError(
            f"Attempted ({total_stages_attempted}) + skipped ({stages_skipped}) must equal 12."
        )

    return {
        "total_stages_evaluated": total_stages_evaluated,
        "total_stages_attempted": total_stages_attempted,
        "stages_succeeded": stages_succeeded,
        "stages_partial": stages_partial,
        "stages_empty": stages_empty,
        "stages_failed": stages_failed,
        "stages_skipped": stages_skipped,
    }


def build_execution_manifest(
    stages: dict[str, dict[str, Any]],
    pipeline_mode: str = "full",
    run_id: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    overall_outcome: Optional[str] = None,
    exit_code: Optional[int] = None,
) -> dict[str, Any]:
    """Constructs a validated canonical 12-stage execution manifest.

    Args:
        stages: Dictionary mapping each canonical stage name to its entry dict.
        pipeline_mode: Execution mode identifier ('full', 'dry-run', etc.).
        run_id: Unique execution identifier (defaults to uuid4 hex).
        timestamp_utc: ISO 8601 UTC timestamp string.
        overall_outcome: 'success', 'partial', or 'failed'. If omitted, inferred from stage failures/partials.
        exit_code: Process exit code. If omitted, inferred (0 if no failures/partials, 1 if failures/partials).

    Returns:
        Fully populated, ordered execution manifest dictionary.
    """
    summary = calculate_manifest_summary(stages)

    has_failures = summary["stages_failed"] > 0
    has_partial = summary.get("stages_partial", 0) > 0
    if has_failures:
        inferred_outcome = "failed"
        inferred_exit_code = EXIT_STAGE_FAILURE
    elif has_partial:
        inferred_outcome = "partial"
        inferred_exit_code = EXIT_STAGE_FAILURE
    else:
        inferred_outcome = "success"
        inferred_exit_code = EXIT_SUCCESS

    resolved_outcome = overall_outcome or inferred_outcome
    resolved_exit_code = inferred_exit_code if exit_code is None else exit_code

    rid = run_id or uuid.uuid4().hex
    ts = timestamp_utc or datetime.now(timezone.utc).isoformat()

    ordered_stages: dict[str, dict[str, Any]] = {}
    for stage_name in CANONICAL_STAGES:
        ordered_stages[stage_name] = stages[stage_name]

    return {
        "run_id": rid,
        "timestamp_utc": ts,
        "pipeline_mode": pipeline_mode,
        "overall_outcome": resolved_outcome,
        "exit_code": resolved_exit_code,
        "summary": summary,
        "stages": ordered_stages,
    }


def persist_manifest(
    manifest_data: dict[str, Any],
    destination_path: Optional[str | Path] = None,
    raise_on_error: bool = False,
) -> bool:
    """Atomically persists the execution manifest to disk.

    Enforces:
    1. Creates parent directory if it does not exist.
    2. Writes to a temporary file in the same directory as target.
    3. Serializes valid JSON.
    4. Flushes and syncs data before replacement.
    5. Replaces destination atomically using os.replace.
    6. Cleans up temporary file after failure.
    7. Never leaves partially written destination manifest.
    8. Does not expose absolute paths or raw exception messages in diagnostics.

    Args:
        manifest_data: Manifest dictionary conforming to canonical schema.
        destination_path: Target path (defaults to DEFAULT_MANIFEST_PATH).
        raise_on_error: If True, raises ManifestPersistenceError on failure.

    Returns:
        True if successfully persisted, False otherwise.

    Raises:
        ManifestPersistenceError: If persistence fails and raise_on_error is True.
    """
    target_path = Path(destination_path) if destination_path else DEFAULT_MANIFEST_PATH
    temp_path: Optional[Path] = None

    try:
        serialized = json.dumps(manifest_data, indent=2, ensure_ascii=False)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        temp_filename = f".tmp_{target_path.stem}_{uuid.uuid4().hex[:8]}.json"
        temp_path = target_path.parent / temp_filename

        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        os.replace(temp_path, target_path)
        logger.info("Execution manifest successfully persisted to %s", target_path.name)
        return True

    except Exception as exc:
        exc_type = type(exc).__name__
        logger.error("Failed to persist execution manifest (%s)", exc_type)
        if temp_path is not None:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

        if raise_on_error:
            raise ManifestPersistenceError(
                f"Failed to persist execution manifest ({exc_type})."
            ) from None

        return False


def print_header() -> None:
    """Displays project title and separator."""
    print("=" * 68)
    print("      LARGE SCALE SOCIAL MEDIA INTELLIGENCE PLATFORM")
    print("=" * 68)
    print("Status: Block 1 - Project Foundation Initialized | Step 1 - Ingestion | Step 2 - Preprocessing (Spark NLP)")
    print("=" * 68)


def print_project_overview() -> None:
    """Displays high-level platform objectives and pipeline stages."""
    print("\n[PROJECT OBJECTIVES]")
    print("1. Data Ingestion: Collect public posts & comments from Reddit & YouTube.")
    print("2. Text Processing: Clean noise, handle emojis, detect languages.")
    print("3. Sentiment Analysis: Multilingual sentiment tracking over time.")
    print("4. Topic Modeling: Dynamic topic trends using BERTopic.")
    print("5. Network Analysis: User-interaction graphs with Neo4j & NetworkX.")
    print("6. Misinformation Detection: Fake news classification using LIAR benchmark.")
    print("7. Search & Dashboards: Elasticsearch indexing with Kibana dashboards.")


def print_system_status() -> None:
    """Displays non-sensitive system status and folder readiness."""
    config = load_config()
    summary = config.get_safe_summary()

    print("\n[ENVIRONMENT & STORAGE STATUS]")
    print(f"Environment Mode : {summary['app_env']}")
    print(f"Log Level        : {summary['log_level']}")
    print(f"Raw Data Storage : {summary['raw_data_path']} (Ready)")
    print(f"Processed Storage: {summary['processed_data_path']} (Ready)")
    print(f"Models Storage   : {summary['models_path']} (Ready)")

    print("\n[DATA INGESTION STATUS]")
    print("Reddit Mock Data : Ready (Offline synthetic fixture)")
    reddit_live_status = "Configured (Ready for live fetch)" if summary["reddit_configured"] else "Disabled (Credentials required in .env)"
    youtube_live_status = "Configured (Ready for live fetch)" if summary["youtube_configured"] else "Disabled (YOUTUBE_API_KEY required in .env)"
    print(f"Reddit Live API  : {reddit_live_status}")
    print(f"YouTube Live API : {youtube_live_status}")
    print(f"Neo4j Database   : {'Ready' if summary['neo4j_configured'] else 'Pending setup (Block 5)'}")
    print(f"Elasticsearch    : Pending setup (Block 7)")

    print("\n[AVAILABLE CLI ACTIONS]")
    print("  python app.py                     : Show platform overview and status")
    print("  python app.py --run-all           : Execute unified 12-stage social media intelligence pipeline")
    print("  python app.py --ingest-reddit     : Ingest mock Reddit dataset into data/raw/")
    print("  python app.py --ingest-reddit-live: Fetch live Reddit posts via OAuth2 API")
    print("  python app.py --ingest-youtube    : Fetch live YouTube videos via Data API v3")
    print("  python app.py --preprocess-reddit : Clean, handle emojis & detect language for Reddit")
    print("  python app.py --preprocess-youtube: Clean, handle emojis & detect language for YouTube")
    print("  python app.py --index-elasticsearch: Index consolidated intelligence dataset into Elasticsearch")

    print("\n" + "-" * 68)
    print("Project foundation & preprocessing ready for incremental development.")
    print("-" * 68)


def run_mock_reddit_cli() -> None:
    """Executes the Reddit mock ingestion pipeline."""
    print("\n[STARTING REDDIT MOCK INGESTION]")
    summary = ingest_reddit_posts()
    _print_ingestion_summary("Reddit Mock", summary)


def run_live_reddit_cli() -> None:
    """Executes the live Reddit API ingestion pipeline."""
    print("\n[STARTING REDDIT LIVE API INGESTION]")
    summary = ingest_reddit_live()
    _print_ingestion_summary("Reddit Live API", summary)


def run_youtube_cli() -> None:
    """Executes the YouTube Data API v3 ingestion pipeline."""
    print("\n[STARTING YOUTUBE DATA API INGESTION]")
    summary = ingest_youtube_live()
    _print_ingestion_summary("YouTube Data API v3", summary)


def run_preprocess_reddit_cli() -> None:
    """Executes the batched Reddit preprocessing pipeline."""
    print("\n[STARTING REDDIT TEXT PREPROCESSING (STEP 2)]")
    summary = preprocess_reddit_dataset()
    _print_preprocessing_summary("Reddit Posts", summary)


def run_preprocess_youtube_cli() -> None:
    """Executes the batched YouTube preprocessing pipeline."""
    print("\n[STARTING YOUTUBE TEXT PREPROCESSING (STEP 2)]")
    summary = preprocess_youtube_dataset()
    _print_preprocessing_summary("YouTube Videos", summary)


def run_index_elasticsearch_cli() -> int:
    """Executes the Elasticsearch indexing pipeline (Stage 11)."""
    print("\n[STARTING ELASTICSEARCH INDEXING (STAGE 11)]")
    from src.search.indexing_pipeline import DEFAULT_INDEXING_MANIFEST_PATH, run_indexing_pipeline

    exit_code, summary = run_indexing_pipeline(manifest_path=DEFAULT_INDEXING_MANIFEST_PATH)
    _print_indexing_summary("Elasticsearch", summary)
    return exit_code


def run_validate_kibana_cli(bundle_path: Optional[Path | str] = None) -> int:
    """Executes offline Kibana dashboard bundle validation (Stage 12)."""
    print("\n[STARTING KIBANA DASHBOARD VALIDATION (STAGE 12)]")
    from src.search.kibana_dashboard_validator import (
        DEFAULT_BUNDLE_PATH,
        validate_kibana_dashboard_bundle,
    )

    target_path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH
    res = validate_kibana_dashboard_bundle(target_path)
    _print_kibana_validation_summary(target_path, res)
    return EXIT_SUCCESS if res.get("is_valid") else EXIT_STAGE_FAILURE


def _print_kibana_validation_summary(bundle_path: Path, summary: dict[str, Any]) -> None:
    """Formats and prints summary metrics for a Kibana dashboard bundle validation run."""
    print(f"Bundle Path          : {bundle_path}")
    print(f"Status               : {'PASSED' if summary.get('is_valid') else 'FAILED'}")
    print(f"Total Saved Objects  : {summary.get('total_objects', 0)}")
    counts = summary.get("counts_by_type", {})
    if counts:
        counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        print(f"Counts by Type       : {counts_str}")
    dashboards = summary.get("dashboards", {})
    if dashboards:
        dash_str = ", ".join(f"{k}: {v} panels" for k, v in sorted(dashboards.items()))
        print(f"Dashboards           : {dash_str}")
    errors = summary.get("errors", [])
    if errors:
        print(f"Validation Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    warnings = summary.get("warnings", [])
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for warn in warnings:
            print(f"  - {warn}")
    print("-" * 68)


def _print_indexing_summary(name: str, summary: dict[str, Any]) -> None:
    """Formats and prints summary metrics for an indexing run."""
    print(f"Target Index         : {summary.get('target_index', summary.get('index_name', 'Unknown'))}")
    print(f"Status               : {str(summary.get('status', 'unknown')).upper()}")
    print(f"Total Records        : {summary.get('total_records')}")
    print(f"Successfully Indexed : {summary.get('indexed_count', 0)}")
    print(f"Rejected Records     : {summary.get('rejected_count', 0)}")
    print(f"Failed Records       : {summary.get('failed_count', 0)}")
    if summary.get("error_category"):
        print(f"Error Category       : {summary.get('error_category')}")
    if summary.get("error_message"):
        print(f"Error Message        : {summary.get('error_message')}")
    print("-" * 68)


def _print_preprocessing_summary(name: str, summary: dict) -> None:
    """Formats and prints summary metrics for a preprocessing run."""
    print(f"Source               : {name}")
    print(f"Status               : {summary.get('status', 'unknown').upper()}")
    if "message" in summary:
        print(f"Message              : {summary['message']}")
    print(f"Total Records        : {summary.get('total_records', 0)}")
    print(f"Successfully Cleaned : {summary.get('processed_records', 0)}")
    print(f"Failed Records       : {summary.get('failed_records', 0)}")
    if summary.get("language_counts"):
        print("Detected Languages   :")
        for lang, count in sorted(summary["language_counts"].items(), key=lambda x: -x[1]):
            print(f"  - {lang}: {count}")
    print(f"Processed Output File: {summary.get('output_file', 'None')}")
    if summary.get("errors"):
        print("\nWarnings / Errors:")
        for err in summary["errors"]:
            print(f"  - {err}")
    print("-" * 68)


def _print_ingestion_summary(name: str, summary: dict) -> None:
    """Formats and prints summary metrics for an ingestion run."""
    print(f"Source           : {name}")
    print(f"Status           : {summary.get('status', 'unknown').upper()}")
    if "message" in summary:
        print(f"Message          : {summary['message']}")
    print(f"Total Records    : {summary.get('total_records', 0)}")
    print(f"Valid Records    : {summary.get('valid_records', 0)}")
    print(f"Rejected Records : {summary.get('rejected_records', 0)}")
    print(f"Raw Destination  : {summary.get('output_file', 'None')}")
    if summary.get("rate_limits"):
        limits = summary["rate_limits"]
        print(f"Rate Limits      : Remaining={limits.get('remaining')}, Reset={limits.get('reset_seconds')}s")
    if summary.get("errors"):
        print("\nWarnings / Errors:")
        for err in summary["errors"]:
            print(f"  - {err}")
    print("-" * 68)


# =====================================================================
# STEP 7.5C — Block 3A: Reddit Ingestion Stage Execution & Validation
# =====================================================================

def validate_reddit_ingestion_output(
    summary: Any,
    temp_file_path: Path,
) -> tuple[bool, int, Optional[str], Optional[str]]:
    """Validates the output of a Reddit ingestion run before promoting to canonical destination.

    Checks:
    1. Ingestion result is a dict and status is 'success'.
    2. Newly generated temporary output file exists and is non-empty.
    3. File contents are valid JSON.
    4. Root JSON structure is a list of dictionaries.
    5. Every record contains a valid, non-empty, trimmed 'post_id'.
    6. No duplicate 'post_id' exists in the serialized records.
    7. Total valid records > 0.
    8. Serialized record count strictly matches summary['valid_records'].

    Returns:
        tuple of (is_valid: bool, record_count: int, error_category: Optional[str], failure_reason: Optional[str])
    """
    if not isinstance(summary, dict):
        return False, 0, "ingestion_error", "Ingestion result must be a dictionary object."

    status = summary.get("status")
    if status != "success":
        error_cat = "credential_error" if status == "disabled" else "ingestion_error"
        return False, 0, error_cat, "Ingestion returned non-success status."

    if not temp_file_path.is_file():
        return False, 0, "missing_output", "Ingestion output file does not exist."

    try:
        if temp_file_path.stat().st_size == 0:
            return False, 0, "empty_output", "Ingestion output file is empty (0 bytes)."
    except OSError:
        return False, 0, "missing_output", "Failed to access ingestion output file."

    try:
        with open(temp_file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, 0, "corrupt_json", "Ingestion output contains malformed or unreadable JSON."

    if not isinstance(records, list):
        return (
            False,
            0,
            "schema_violation",
            "Ingestion output root must be a JSON list.",
        )

    if len(records) == 0:
        return False, 0, "empty_output", "Ingestion output list contains zero records."

    seen_ids: set[str] = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Record at index {idx} is not a dictionary.",
            )
        post_id = record.get("post_id")
        if not isinstance(post_id, str) or not post_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"Record at index {idx} has missing, empty, or whitespace post_id.",
            )
        trimmed_id = post_id.strip()
        if trimmed_id in seen_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate post_id detected in ingestion output.",
            )
        seen_ids.add(trimmed_id)

    if "valid_records" not in summary:
        return (
            False,
            0,
            "count_mismatch",
            "Summary is missing required valid_records field.",
        )

    reported_valid = summary["valid_records"]
    if isinstance(reported_valid, bool) or not isinstance(reported_valid, int) or reported_valid < 0:
        return (
            False,
            0,
            "count_mismatch",
            "Reported valid_records is not a valid non-negative integer.",
        )

    if len(records) != reported_valid:
        return (
            False,
            0,
            "count_mismatch",
            "Serialized record count does not match reported valid count.",
        )

    return True, len(records), None, None


def execute_reddit_ingestion_stage(
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    subreddits: Optional[list[str] | str] = None,
    limit_per_subreddit: int = 25,
    sample_source_path: Optional[Path | str] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes and validates the Reddit data ingestion stage.

    Supports:
    - Explicit skip handling (invoking neither ingestion function nor inspecting credentials).
    - Mock Reddit ingestion when valid Reddit credentials are not configured.
    - Live Reddit ingestion when valid credentials are configured.
    - Output validation and atomic promotion to protect against stale data.
    - Safe manifest stage entry creation.

    Args:
        skip: When True, skips ingestion and returns a 'skipped' stage entry.
        canonical_output_path: Destination path for valid raw Reddit data.
        config: Optional loaded AppConfig instance.
        subreddits: Optional subreddit list or comma-separated string for live ingestion.
        limit_per_subreddit: Maximum posts per subreddit for live ingestion.
        sample_source_path: Optional fixture path for mock ingestion.

    Returns:
        tuple of (stage_entry_dict, canonical_output_path_or_None)
    """
    if skip:
        logger.info("Reddit ingestion stage explicitly skipped by configuration or user flag.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    start_time = time.perf_counter()
    temp_output_path: Optional[Path] = None

    try:
        resolved_config = config or load_config()
        if canonical_output_path is not None:
            resolved_canonical = Path(canonical_output_path)
        else:
            resolved_canonical = resolved_config.raw_data_path / "reddit_posts_raw.json"

        # Ensure parent directory exists for both temporary and canonical files
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)

        # Unique temporary output file in the same directory to enable atomic replacement
        temp_output_path = resolved_canonical.parent / f".tmp_reddit_{uuid.uuid4().hex}.json"

        safe_summary = resolved_config.get_safe_summary()
        is_live = bool(safe_summary.get("reddit_configured"))

        if is_live:
            logger.info("Reddit credentials configured. Executing live Reddit API ingestion.")
            ingestion_summary = ingest_reddit_live(
                subreddits=subreddits,
                limit_per_subreddit=limit_per_subreddit,
                output_path=temp_output_path,
                merge=False,
            )
        else:
            logger.info("Reddit credentials not configured. Executing synthetic mock Reddit ingestion.")
            ingestion_summary = ingest_reddit_posts(
                source_path=sample_source_path,
                output_path=temp_output_path,
                merge=False,
            )

        # Validate newly generated output file and summary
        is_valid, record_count, error_cat, reason = validate_reddit_ingestion_output(
            summary=ingestion_summary,
            temp_file_path=temp_output_path,
        )

        elapsed = time.perf_counter() - start_time

        if not is_valid:
            logger.warning("Reddit ingestion output validation failed: %s (%s)", reason, error_cat)
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=error_cat,
                ),
                None,
            )

        # Atomic promotion: replace canonical destination with validated temporary output
        os.replace(temp_output_path, resolved_canonical)
        logger.info(
            "Reddit ingestion stage succeeded with %d records. Canonical output: %s",
            record_count,
            resolved_canonical.name,
        )

        return (
            create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during Reddit ingestion stage execution.")
        if temp_output_path and temp_output_path.exists():
            temp_output_path.unlink(missing_ok=True)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Ingestion execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# STEP 7.5C — Block 3B: YouTube Ingestion Stage Execution & Validation
# =====================================================================

def validate_youtube_ingestion_output(
    summary: Any,
    temp_file_path: Path,
) -> tuple[bool, int, Optional[str], Optional[str]]:
    """Validates the output of a YouTube ingestion run before promoting to canonical destination.

    Checks:
    1. Ingestion result is a dict and status is 'success'.
    2. Newly generated temporary output file exists and is non-empty.
    3. File contents are valid JSON.
    4. Root JSON structure is a list of dictionaries.
    5. Every record contains a valid, non-empty, trimmed 'video_id'.
    6. Every record satisfies the required YouTube schema.
    7. No duplicate 'video_id' exists in the serialized records.
    8. Total valid records > 0.
    9. summary contains 'valid_records' which must be an actual non-negative integer (not bool, float, str, None).
    10. Serialized record count strictly matches summary['valid_records'].

    Returns:
        tuple of (is_valid: bool, record_count: int, error_category: Optional[str], failure_reason: Optional[str])
    """
    if not isinstance(summary, dict):
        return False, 0, "ingestion_error", "Ingestion result must be a dictionary object."

    status = summary.get("status")
    if status != "success":
        error_cat = "credential_error" if status == "disabled" else "ingestion_error"
        return False, 0, error_cat, "Ingestion returned non-success status."

    if not temp_file_path.is_file():
        return False, 0, "missing_output", "Ingestion output file does not exist."

    try:
        if temp_file_path.stat().st_size == 0:
            return False, 0, "empty_output", "Ingestion output file is empty (0 bytes)."
    except OSError:
        return False, 0, "missing_output", "Failed to access ingestion output file."

    try:
        with open(temp_file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, 0, "corrupt_json", "Ingestion output contains malformed or unreadable JSON."

    if not isinstance(records, list):
        return (
            False,
            0,
            "schema_violation",
            "Ingestion output root must be a JSON list.",
        )

    if len(records) == 0:
        return False, 0, "empty_output", "Ingestion output list contains zero records."

    seen_ids: set[str] = set()
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Record at index {idx} is not a dictionary.",
            )
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"Record at index {idx} has missing, empty, or whitespace video_id.",
            )
        trimmed_id = video_id.strip()
        if trimmed_id in seen_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate video_id detected in ingestion output.",
            )
        seen_ids.add(trimmed_id)

        is_schema_valid, _ = validate_youtube_video(record)
        if not is_schema_valid:
            return (
                False,
                0,
                "schema_violation",
                f"Record at index {idx} failed YouTube schema validation.",
            )

    if "valid_records" not in summary:
        return (
            False,
            0,
            "count_mismatch",
            "Summary is missing required valid_records field.",
        )

    reported_valid = summary["valid_records"]
    if isinstance(reported_valid, bool) or not isinstance(reported_valid, int) or reported_valid < 0:
        return (
            False,
            0,
            "count_mismatch",
            "Reported valid_records is not a valid non-negative integer.",
        )

    if len(records) != reported_valid:
        return (
            False,
            0,
            "count_mismatch",
            "Serialized record count does not match reported valid count.",
        )

    return True, len(records), None, None


def execute_youtube_ingestion_stage(
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    query: Optional[str] = None,
    max_pages: int = 1,
    max_results_per_page: int = 25,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes and validates the YouTube data ingestion stage.

    Supports:
    - Explicit skip handling (invoking neither ingestion function nor inspecting credentials).
    - API-key-aware execution: skips stage cleanly if YouTube API key is unconfigured.
    - Live YouTube Data API v3 ingestion when API key is configured.
    - Output validation and atomic promotion to protect against stale data.
    - Safe manifest stage entry creation.

    Args:
        skip: When True, skips ingestion and returns a 'skipped' stage entry.
        canonical_output_path: Destination path for valid raw YouTube data.
        config: Optional loaded AppConfig instance.
        query: Search query for YouTube search.list.
        max_pages: Maximum search result pages to fetch.
        max_results_per_page: Number of videos per search page.

    Returns:
        tuple of (stage_entry_dict, canonical_output_path_or_None)
    """
    if skip:
        logger.info("YouTube ingestion stage explicitly skipped by configuration or user flag.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    start_time = time.perf_counter()
    temp_output_path: Optional[Path] = None

    try:
        resolved_config = config or load_config()
        safe_summary = resolved_config.get_safe_summary()
        is_configured = bool(safe_summary.get("youtube_configured"))

        if not is_configured:
            elapsed = time.perf_counter() - start_time
            logger.info("YouTube API key not configured. Skipping YouTube ingestion stage.")
            return (
                create_initial_stage_entry(
                    status="skipped",
                    reason="youtube_api_key_not_configured",
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=None,
                ),
                None,
            )

        if canonical_output_path is not None:
            resolved_canonical = Path(canonical_output_path)
        else:
            resolved_canonical = resolved_config.raw_data_path / "youtube_videos_raw.json"

        # Ensure parent directory exists for both temporary and canonical files
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)

        # Unique temporary output file in the same directory to enable atomic replacement
        temp_output_path = resolved_canonical.parent / f".tmp_youtube_{uuid.uuid4().hex}.json"

        logger.info("YouTube API key configured. Executing live YouTube Data API ingestion.")
        target_query = query if query is not None else "social media intelligence machine learning"
        ingestion_summary = ingest_youtube_live(
            query=target_query,
            max_pages=max_pages,
            max_results_per_page=max_results_per_page,
            output_path=temp_output_path,
            merge=False,
        )

        # Validate newly generated output file and summary
        is_valid, record_count, error_cat, reason = validate_youtube_ingestion_output(
            summary=ingestion_summary,
            temp_file_path=temp_output_path,
        )

        elapsed = time.perf_counter() - start_time

        if not is_valid:
            logger.warning("YouTube ingestion output validation failed: %s (%s)", reason, error_cat)
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=error_cat,
                ),
                None,
            )

        # Atomic promotion: replace canonical destination with validated temporary output
        os.replace(temp_output_path, resolved_canonical)
        logger.info(
            "YouTube ingestion stage succeeded with %d records. Canonical output: %s",
            record_count,
            resolved_canonical.name,
        )

        return (
            create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during YouTube ingestion stage execution.")
        if temp_output_path and temp_output_path.exists():
            temp_output_path.unlink(missing_ok=True)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Ingestion execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# STEP 7.5C — Block 4A: Reddit Preprocessing Execution & Validation
# =====================================================================

def validate_reddit_preprocessing_output(
    summary: Any,
    temp_file_path: Path,
) -> tuple[bool, int, Optional[str], Optional[str]]:
    """Validates the output of a Reddit preprocessing run before promoting to canonical destination.

    Checks:
    1. Preprocessing result is a dict and status is 'success'.
    2. Required count fields ('total_records', 'processed_records', 'failed_records') exist and are non-negative ints.
    3. Rejects bool, float, str, negative, or missing counts.
    4. total_records == processed_records + failed_records.
    5. Newly generated temporary output file exists and is non-empty.
    6. File contents are valid JSON.
    7. Root JSON structure is a list of dictionaries.
    8. Every record contains a valid, non-empty, trimmed 'post_id'.
    9. No duplicate 'post_id' exists in the serialized records.
    10. Required preprocessing enrichment fields exist on each record.
    11. Total processed records > 0.
    12. Serialized record count strictly matches summary['processed_records'].

    Returns:
        tuple of (is_valid: bool, record_count: int, error_category: Optional[str], failure_reason: Optional[str])
    """
    if not isinstance(summary, dict):
        return False, 0, "preprocessing_error", "Preprocessing result must be a dictionary object."

    status = summary.get("status")
    if status != "success":
        return False, 0, "preprocessing_error", "Preprocessing returned non-success status."

    # Validate count fields existence and types
    for count_key in ("processed_records", "total_records", "failed_records"):
        if count_key not in summary:
            return False, 0, "count_mismatch", f"Summary is missing required {count_key} field."
        val = summary[count_key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            return False, 0, "count_mismatch", f"Reported {count_key} is not a valid non-negative integer."

    total_records = summary["total_records"]
    processed_records = summary["processed_records"]
    failed_records = summary["failed_records"]

    if total_records != processed_records + failed_records:
        return False, 0, "count_mismatch", "Total records count does not equal sum of processed and failed records."

    if not temp_file_path.is_file():
        return False, 0, "missing_output", "Preprocessing output file does not exist."

    try:
        if temp_file_path.stat().st_size == 0:
            return False, 0, "empty_output", "Preprocessing output file is empty (0 bytes)."
    except OSError:
        return False, 0, "missing_output", "Failed to access preprocessing output file."

    try:
        with open(temp_file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, 0, "corrupt_json", "Preprocessing output contains malformed or unreadable JSON."

    if not isinstance(records, list):
        return (
            False,
            0,
            "schema_violation",
            "Preprocessing output root must be a JSON list.",
        )

    if len(records) == 0:
        return False, 0, "empty_output", "Preprocessing output list contains zero records."

    seen_ids: set[str] = set()
    required_fields = (
        "title_cleaned",
        "text_cleaned",
        "detected_language",
        "language_code",
        "preprocessing_status",
        "processed_at",
    )

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Record at index {idx} is not a dictionary.",
            )
        post_id = record.get("post_id")
        if not isinstance(post_id, str) or not post_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"Record at index {idx} has missing, empty, or whitespace post_id.",
            )
        trimmed_id = post_id.strip()
        if trimmed_id in seen_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate post_id detected in preprocessing output.",
            )
        seen_ids.add(trimmed_id)

        for req_field in required_fields:
            if req_field not in record:
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Record at index {idx} missing required preprocessing field.",
                )
            if not isinstance(record[req_field], str):
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Record at index {idx} field '{req_field}' must be a string.",
                )

    if len(records) != processed_records:
        return (
            False,
            0,
            "count_mismatch",
            "Serialized record count does not match reported processed count.",
        )

    return True, len(records), None, None


def execute_reddit_preprocessing_stage(
    raw_input_path: Optional[Path | str],
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    batch_size: int = 100,
    ingestion_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes and validates the Reddit text preprocessing stage.

    Supports:
    - Dependency checks: skips if ingestion was skipped or failed.
    - Path safety: requires an explicit valid raw_input_path; never falls back to defaults.
    - Temporary output isolation and atomic promotion to protect against stale data.
    - Schema, count, and integrity validation.
    - Safe manifest stage entry creation.

    Args:
        raw_input_path: Validated raw Reddit file path produced by current execution.
        skip: When True, skips preprocessing and returns a 'skipped' stage entry.
        canonical_output_path: Destination path for valid processed Reddit data.
        config: Optional loaded AppConfig instance.
        batch_size: Configurable chunk size for batch processing.
        ingestion_stage_entry: Optional stage manifest entry from the Reddit ingestion stage.

    Returns:
        tuple of (stage_entry_dict, canonical_output_path_or_None)
    """
    if skip:
        logger.info("Reddit preprocessing stage explicitly skipped by configuration or user flag.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if ingestion_stage_entry is not None:
        ingest_status = ingestion_stage_entry.get("status")
        if ingest_status == "skipped":
            logger.info("Reddit ingestion was skipped. Skipping Reddit preprocessing stage.")
            return (
                create_initial_stage_entry(
                    status="skipped",
                    reason="reddit_ingestion_skipped",
                    record_count=0,
                    duration_seconds=0.0,
                    error_category=None,
                ),
                None,
            )
        if ingest_status == "failed":
            logger.warning("Reddit ingestion failed. Skipping Reddit preprocessing stage.")
            return (
                create_initial_stage_entry(
                    status="skipped",
                    reason="reddit_ingestion_failed",
                    record_count=0,
                    duration_seconds=0.0,
                    error_category=None,
                ),
                None,
            )

    start_time = time.perf_counter()

    if raw_input_path is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Reddit preprocessing failed: raw input path is None.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Raw input path is missing or None",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_raw_input",
            ),
            None,
        )

    resolved_raw = Path(raw_input_path)
    if not resolved_raw.is_file():
        elapsed = time.perf_counter() - start_time
        logger.warning("Reddit preprocessing failed: raw input file does not exist.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Raw input file does not exist",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_raw_input",
            ),
            None,
        )

    temp_output_path: Optional[Path] = None

    try:
        resolved_config = config or load_config()
        if canonical_output_path is not None:
            resolved_canonical = Path(canonical_output_path)
        else:
            resolved_canonical = resolved_config.processed_data_path / "reddit_posts_processed.json"

        # Ensure parent directory exists for both temporary and canonical files
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)

        # Unique temporary output file in the same directory to enable atomic replacement
        temp_output_path = resolved_canonical.parent / f".tmp_reddit_processed_{uuid.uuid4().hex}.json"

        logger.info("Executing Reddit dataset preprocessing.")
        preprocessing_summary = preprocess_reddit_dataset(
            input_path=resolved_raw,
            output_path=temp_output_path,
            batch_size=batch_size,
            merge=False,
        )

        # Validate newly generated output file and summary
        is_valid, record_count, error_cat, reason = validate_reddit_preprocessing_output(
            summary=preprocessing_summary,
            temp_file_path=temp_output_path,
        )

        elapsed = time.perf_counter() - start_time

        if not is_valid:
            logger.warning("Reddit preprocessing output validation failed: %s (%s)", reason, error_cat)
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=error_cat,
                ),
                None,
            )

        # Atomic promotion: replace canonical destination with validated temporary output
        os.replace(temp_output_path, resolved_canonical)
        logger.info(
            "Reddit preprocessing stage succeeded with %d records. Canonical output: %s",
            record_count,
            resolved_canonical.name,
        )

        return (
            create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during Reddit preprocessing stage execution.")
        if temp_output_path and temp_output_path.exists():
            temp_output_path.unlink(missing_ok=True)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Preprocessing execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# STEP 7.5C — Block 4B: YouTube Preprocessing Execution & Validation
# =====================================================================

def validate_youtube_preprocessing_output(
    summary: Any,
    temp_file_path: Path,
) -> tuple[bool, int, Optional[str], Optional[str]]:
    """Validates the output of a YouTube preprocessing run before promoting to canonical destination.

    Checks:
    1. Preprocessing result is a dict and status is 'success'.
    2. Required count fields ('total_records', 'processed_records', 'failed_records') exist and are non-negative ints.
    3. Rejects bool, float, str, negative, or missing counts.
    4. total_records == processed_records + failed_records.
    5. Newly generated temporary output file exists and is non-empty.
    6. File contents are valid JSON.
    7. Root JSON structure is a list of dictionaries.
    8. Every record contains a valid, non-empty, trimmed 'video_id'.
    9. No duplicate 'video_id' exists in the serialized records.
    10. Required preprocessing enrichment fields exist on each record.
    11. Total processed records > 0.
    12. Serialized record count strictly matches summary['processed_records'].

    Returns:
        tuple of (is_valid: bool, record_count: int, error_category: Optional[str], failure_reason: Optional[str])
    """
    if not isinstance(summary, dict):
        return False, 0, "preprocessing_error", "Preprocessing result must be a dictionary object."

    status = summary.get("status")
    if status != "success":
        return False, 0, "preprocessing_error", "Preprocessing returned non-success status."

    # Validate count fields existence and types
    for count_key in ("processed_records", "total_records", "failed_records"):
        if count_key not in summary:
            return False, 0, "count_mismatch", f"Summary is missing required {count_key} field."
        val = summary[count_key]
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            return False, 0, "count_mismatch", f"Reported {count_key} is not a valid non-negative integer."

    total_records = summary["total_records"]
    processed_records = summary["processed_records"]
    failed_records = summary["failed_records"]

    if total_records != processed_records + failed_records:
        return False, 0, "count_mismatch", "Total records count does not equal sum of processed and failed records."

    if not temp_file_path.is_file():
        return False, 0, "missing_output", "Preprocessing output file does not exist."

    try:
        if temp_file_path.stat().st_size == 0:
            return False, 0, "empty_output", "Preprocessing output file is empty (0 bytes)."
    except OSError:
        return False, 0, "missing_output", "Failed to access preprocessing output file."

    try:
        with open(temp_file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, 0, "corrupt_json", "Preprocessing output contains malformed or unreadable JSON."

    if not isinstance(records, list):
        return (
            False,
            0,
            "schema_violation",
            "Preprocessing output root must be a JSON list.",
        )

    if len(records) == 0:
        return False, 0, "empty_output", "Preprocessing output list contains zero records."

    seen_ids: set[str] = set()
    required_fields = (
        "title_cleaned",
        "description_cleaned",
        "detected_language",
        "language_code",
        "preprocessing_status",
        "processed_at",
    )

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Record at index {idx} is not a dictionary.",
            )
        video_id = record.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"Record at index {idx} has missing, empty, or whitespace video_id.",
            )
        trimmed_id = video_id.strip()
        if trimmed_id in seen_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate video_id detected in preprocessing output.",
            )
        seen_ids.add(trimmed_id)

        for req_field in required_fields:
            if req_field not in record:
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Record at index {idx} missing required preprocessing field.",
                )
            if not isinstance(record[req_field], str):
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Record at index {idx} field '{req_field}' must be a string.",
                )

    if len(records) != processed_records:
        return (
            False,
            0,
            "count_mismatch",
            "Serialized record count does not match reported processed count.",
        )

    return True, len(records), None, None


def execute_youtube_preprocessing_stage(
    raw_input_path: Optional[Path | str],
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    batch_size: int = 100,
    ingestion_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes and validates the YouTube text preprocessing stage.

    Supports:
    - Dependency checks: skips if ingestion was skipped or failed.
    - Path safety: requires an explicit valid raw_input_path; never falls back to defaults.
    - Temporary output isolation and atomic promotion to protect against stale data.
    - Schema, count, and integrity validation.
    - Safe manifest stage entry creation.

    Args:
        raw_input_path: Validated raw YouTube file path produced by current execution.
        skip: When True, skips preprocessing and returns a 'skipped' stage entry.
        canonical_output_path: Destination path for valid processed YouTube data.
        config: Optional loaded AppConfig instance.
        batch_size: Configurable chunk size for batch processing.
        ingestion_stage_entry: Optional stage manifest entry from the YouTube ingestion stage.

    Returns:
        tuple of (stage_entry_dict, canonical_output_path_or_None)
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("YouTube preprocessing stage explicitly skipped by configuration or user flag.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if ingestion_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("YouTube preprocessing failed: missing upstream ingestion provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream ingestion provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(ingestion_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("YouTube preprocessing failed: invalid upstream ingestion stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream ingestion stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    ingest_status = ingestion_stage_entry.get("status")
    if ingest_status == "skipped":
        logger.info("YouTube ingestion was skipped. Skipping YouTube preprocessing stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="youtube_ingestion_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if ingest_status == "failed":
        logger.warning("YouTube ingestion failed. Skipping YouTube preprocessing stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="youtube_ingestion_failed",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if ingest_status != "success":
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "YouTube preprocessing failed: invalid or unsupported upstream ingestion status '%s'.",
            ingest_status,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid or unsupported upstream ingestion status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    if raw_input_path is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("YouTube preprocessing failed: raw input path is None.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Raw input path is missing or None",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_raw_input",
            ),
            None,
        )

    resolved_raw = Path(raw_input_path)
    if not resolved_raw.is_file():
        elapsed = time.perf_counter() - start_time
        logger.warning("YouTube preprocessing failed: raw input file does not exist.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Raw input file does not exist",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_raw_input",
            ),
            None,
        )

    temp_output_path: Optional[Path] = None

    try:
        resolved_config = config or load_config()
        if canonical_output_path is not None:
            resolved_canonical = Path(canonical_output_path)
        else:
            resolved_canonical = resolved_config.processed_data_path / "youtube_videos_processed.json"

        # Ensure parent directory exists for both temporary and canonical files
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)

        # Unique temporary output file in the same directory to enable atomic replacement
        temp_output_path = resolved_canonical.parent / f".tmp_youtube_processed_{uuid.uuid4().hex}.json"

        logger.info("Executing YouTube dataset preprocessing.")
        preprocessing_summary = preprocess_youtube_dataset(
            input_path=resolved_raw,
            output_path=temp_output_path,
            batch_size=batch_size,
            merge=False,
        )

        # Validate newly generated output file and summary
        is_valid, record_count, error_cat, reason = validate_youtube_preprocessing_output(
            summary=preprocessing_summary,
            temp_file_path=temp_output_path,
        )

        elapsed = time.perf_counter() - start_time

        if not is_valid:
            logger.warning("YouTube preprocessing output validation failed: %s (%s)", reason, error_cat)
            if temp_output_path.exists():
                temp_output_path.unlink(missing_ok=True)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=error_cat,
                ),
                None,
            )

        # Atomic promotion: replace canonical destination with validated temporary output
        os.replace(temp_output_path, resolved_canonical)
        logger.info(
            "YouTube preprocessing stage succeeded with %d records. Canonical output: %s",
            record_count,
            resolved_canonical.name,
        )

        return (
            create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during YouTube preprocessing stage execution.")
        if temp_output_path and temp_output_path.exists():
            temp_output_path.unlink(missing_ok=True)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Preprocessing execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# STEP 7.5C — Block 5B: Sentiment Output Validation
# =====================================================================

PERMITTED_SENTIMENT_LABELS: frozenset[str] = frozenset(
    {"positive", "neutral", "negative", "undetermined"}
)


def _is_valid_iso_timestamp(val: Any) -> bool:
    """Validates whether a value is a non-empty ISO-8601 timestamp string."""
    if not isinstance(val, str) or not val.strip():
        return False
    clean = val.strip()
    if clean.endswith("Z") or clean.endswith("z"):
        clean = clean[:-1] + "+00:00"
    try:
        datetime.fromisoformat(clean)
        return True
    except (ValueError, TypeError):
        return False


def _validate_sentiment_metrics_dict(
    metrics: Any,
    label: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Validates a summary metrics dictionary containing counts, percentages, and net score."""
    if not isinstance(metrics, dict):
        return False, "schema_violation", f"'{label}' metrics must be a dictionary."

    for count_key in ("total", "positive", "neutral", "negative", "undetermined"):
        if count_key not in metrics:
            return False, "count_mismatch", f"'{label}' is missing required count field '{count_key}'."
        c_val = metrics[count_key]
        if isinstance(c_val, bool) or not isinstance(c_val, int) or c_val < 0:
            return False, "count_mismatch", f"'{label}' count '{count_key}' is not a valid non-negative integer."

    total = metrics["total"]
    sum_parts = metrics["positive"] + metrics["neutral"] + metrics["negative"] + metrics["undetermined"]
    if total != sum_parts:
        return False, "count_mismatch", f"'{label}' total count does not equal sum of sentiment categories."

    for pct_key in ("positive_pct", "neutral_pct", "negative_pct", "undetermined_pct"):
        if pct_key not in metrics:
            return False, "schema_violation", f"'{label}' is missing required percentage field '{pct_key}'."
        pct_val = metrics[pct_key]
        if isinstance(pct_val, bool) or not isinstance(pct_val, (int, float)) or not math.isfinite(pct_val):
            return False, "schema_violation", f"'{label}' percentage '{pct_key}' must be a finite numeric value."
        if pct_val < 0.0 or pct_val > 100.0:
            return False, "schema_violation", f"'{label}' percentage '{pct_key}' is out of range [0.0, 100.0]."

    if "net_sentiment" not in metrics:
        return False, "schema_violation", f"'{label}' is missing required net_sentiment field."
    net_val = metrics["net_sentiment"]
    if isinstance(net_val, bool) or not isinstance(net_val, (int, float)) or not math.isfinite(net_val):
        return False, "schema_violation", f"'{label}' net_sentiment must be a finite numeric value."
    if net_val < -1.0 or net_val > 1.0:
        return False, "schema_violation", f"'{label}' net_sentiment is out of range [-1.0, 1.0]."

    # Mathematical consistency checks for percentage metrics and net sentiment
    if total == 0:
        for pct_key in ("positive_pct", "neutral_pct", "negative_pct", "undetermined_pct"):
            if not math.isclose(metrics[pct_key], 0.0, rel_tol=0.0, abs_tol=1e-9):
                return (
                    False,
                    "count_mismatch",
                    f"'{label}' percentage '{pct_key}' must equal 0.0 when total is 0.",
                )
        if not math.isclose(net_val, 0.0, rel_tol=0.0, abs_tol=1e-9):
            return (
                False,
                "count_mismatch",
                f"'{label}' net_sentiment must equal 0.0 when total is 0.",
            )
    else:
        for cat_key, pct_key in (
            ("positive", "positive_pct"),
            ("neutral", "neutral_pct"),
            ("negative", "negative_pct"),
            ("undetermined", "undetermined_pct"),
        ):
            expected_pct = round((metrics[cat_key] / total) * 100.0, 2)
            if not math.isclose(metrics[pct_key], expected_pct, rel_tol=0.0, abs_tol=1e-9):
                return (
                    False,
                    "count_mismatch",
                    f"'{label}' percentage '{pct_key}' does not match underlying count.",
                )

        expected_net = round((metrics["positive"] - metrics["negative"]) / total, 4)
        if not math.isclose(net_val, expected_net, rel_tol=0.0, abs_tol=1e-9):
            return (
                False,
                "count_mismatch",
                f"'{label}' net_sentiment does not match underlying counts.",
            )

    return True, None, None


def validate_sentiment_output(
    summary: Any,
    temp_file_path: Path,
    expected_reddit_count: Optional[int] = None,
    expected_youtube_count: Optional[int] = None,
) -> tuple[bool, int, Optional[str], Optional[str]]:
    """Validates the output of a sentiment analysis run before promoting to canonical destination.

    Checks:
    1. Pipeline summary is a dict and status is 'success'.
    2. Newly generated temporary output file exists, is a regular file, and is non-empty.
    3. File contents are valid JSON.
    4. Root JSON structure is a dictionary.
    5. Required top-level keys exist with valid types.
    6. Count fields are actual non-negative integers (rejecting bools, floats, strings, negative).
    7. Overall summary total matches reddit_summary total + youtube_summary total.
    8. Invariant: total == positive + neutral + negative + undetermined across all summaries.
    9. Temporal aggregations exist, contain 'daily' and 'weekly' dictionaries with valid metrics.
    10. Record counts strictly match their corresponding summaries and expected counts.
    11. Reddit records contain valid, non-empty, unique post_id values.
    12. YouTube records contain valid, non-empty, unique video_id values.
    13. Every record contains valid sentiment_label, finite sentiment_score in [0.0, 1.0],
        sentiment_status string, and valid ISO-8601 analyzed_at timestamp.

    Returns:
        tuple of (is_valid: bool, record_count: int, error_category: Optional[str], failure_reason: Optional[str])
    """
    if not isinstance(summary, dict):
        return False, 0, "sentiment_error", "Sentiment pipeline summary must be a dictionary object."

    if summary.get("status") != "success":
        return False, 0, "sentiment_error", "Sentiment pipeline returned non-success status."

    if (
        temp_file_path is None
        or isinstance(temp_file_path, bool)
        or not isinstance(temp_file_path, (str, Path, os.PathLike))
    ):
        return False, 0, "missing_output", "Sentiment output file path is invalid."

    try:
        resolved_temp_path = Path(temp_file_path)
    except (TypeError, ValueError):
        return False, 0, "missing_output", "Sentiment output file path is invalid."

    if not resolved_temp_path.is_file():
        return False, 0, "missing_output", "Sentiment output file does not exist."

    try:
        if resolved_temp_path.stat().st_size == 0:
            return False, 0, "empty_output", "Sentiment output file is empty (0 bytes)."
    except OSError:
        return False, 0, "missing_output", "Failed to access sentiment output file."

    try:
        with open(resolved_temp_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, 0, "corrupt_json", "Sentiment output contains malformed or unreadable JSON."

    if not isinstance(payload, dict):
        return (
            False,
            0,
            "schema_violation",
            "Sentiment output root must be a JSON dictionary object.",
        )

    required_top_keys = (
        "status",
        "generated_at",
        "model_name",
        "total_records_processed",
        "summary",
        "reddit_summary",
        "youtube_summary",
        "temporal_aggregations",
        "reddit_records",
        "youtube_records",
    )
    for req_key in required_top_keys:
        if req_key not in payload:
            return (
                False,
                0,
                "schema_violation",
                f"Sentiment output missing required top-level field '{req_key}'.",
            )

    if payload["status"] != "success":
        return False, 0, "schema_violation", "Sentiment output status must be 'success'."

    if not _is_valid_iso_timestamp(payload["generated_at"]):
        return (
            False,
            0,
            "schema_violation",
            "Field 'generated_at' must be a valid ISO-8601 timestamp string.",
        )

    if not isinstance(payload["model_name"], str) or not payload["model_name"].strip():
        return (
            False,
            0,
            "schema_violation",
            "Field 'model_name' must be a non-empty string.",
        )

    total_records = payload["total_records_processed"]
    if isinstance(total_records, bool) or not isinstance(total_records, int) or total_records < 0:
        return (
            False,
            0,
            "count_mismatch",
            "Field 'total_records_processed' must be a valid non-negative integer.",
        )

    if "total_records_processed" in summary:
        sum_total = summary["total_records_processed"]
        if isinstance(sum_total, bool) or not isinstance(sum_total, int) or sum_total != total_records:
            return (
                False,
                0,
                "count_mismatch",
                "Summary total_records_processed does not match payload total_records_processed.",
            )

    # Validate overall, reddit, and youtube summaries
    for summary_key in ("summary", "reddit_summary", "youtube_summary"):
        sub_valid, sub_cat, sub_reason = _validate_sentiment_metrics_dict(
            payload[summary_key], summary_key
        )
        if not sub_valid:
            return False, 0, sub_cat, sub_reason

    overall_summary = payload["summary"]
    reddit_summary = payload["reddit_summary"]
    youtube_summary = payload["youtube_summary"]

    if overall_summary["total"] != total_records:
        return (
            False,
            0,
            "count_mismatch",
            "Overall summary total does not match total_records_processed.",
        )

    if overall_summary["total"] != reddit_summary["total"] + youtube_summary["total"]:
        return (
            False,
            0,
            "count_mismatch",
            "Overall summary total does not equal sum of reddit and youtube summary totals.",
        )

    # Validate temporal aggregations
    temporal = payload["temporal_aggregations"]
    if not isinstance(temporal, dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'temporal_aggregations' must be a dictionary.",
        )

    for win_key in ("daily", "weekly"):
        if win_key not in temporal:
            return (
                False,
                0,
                "schema_violation",
                f"Field 'temporal_aggregations' missing required window '{win_key}'.",
            )
        win_dict = temporal[win_key]
        if not isinstance(win_dict, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Temporal window '{win_key}' must be a dictionary.",
            )
        for bucket_key, bucket_val in win_dict.items():
            if not isinstance(bucket_key, str) or not bucket_key.strip():
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Temporal window '{win_key}' contains invalid bucket key.",
                )
            b_valid, b_cat, b_reason = _validate_sentiment_metrics_dict(
                bucket_val, f"{win_key}[{bucket_key}]"
            )
            if not b_valid:
                return False, 0, b_cat, b_reason

    # Validate record collections
    reddit_records = payload["reddit_records"]
    youtube_records = payload["youtube_records"]

    if not isinstance(reddit_records, list) or not isinstance(youtube_records, list):
        return (
            False,
            0,
            "schema_violation",
            "'reddit_records' and 'youtube_records' must be JSON lists.",
        )

    if len(reddit_records) != reddit_summary["total"]:
        return (
            False,
            0,
            "count_mismatch",
            "Reddit records count does not match reddit_summary total.",
        )

    if len(youtube_records) != youtube_summary["total"]:
        return (
            False,
            0,
            "count_mismatch",
            "YouTube records count does not match youtube_summary total.",
        )

    if len(reddit_records) + len(youtube_records) != total_records:
        return (
            False,
            0,
            "count_mismatch",
            "Sum of record counts does not match total_records_processed.",
        )

    # Expected upstream count reconciliation
    if expected_reddit_count is not None:
        if (
            isinstance(expected_reddit_count, bool)
            or not isinstance(expected_reddit_count, int)
            or expected_reddit_count < 0
        ):
            return (
                False,
                0,
                "count_mismatch",
                "expected_reddit_count must be a valid non-negative integer.",
            )
        if len(reddit_records) != expected_reddit_count:
            return (
                False,
                0,
                "count_mismatch",
                f"Reddit record count ({len(reddit_records)}) does not match expected count ({expected_reddit_count}).",
            )

    if expected_youtube_count is not None:
        if (
            isinstance(expected_youtube_count, bool)
            or not isinstance(expected_youtube_count, int)
            or expected_youtube_count < 0
        ):
            return (
                False,
                0,
                "count_mismatch",
                "expected_youtube_count must be a valid non-negative integer.",
            )
        if len(youtube_records) != expected_youtube_count:
            return (
                False,
                0,
                "count_mismatch",
                f"YouTube record count ({len(youtube_records)}) does not match expected count ({expected_youtube_count}).",
            )

    # Validate individual Reddit records
    seen_reddit_ids: set[str] = set()
    for idx, rec in enumerate(reddit_records):
        if not isinstance(rec, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} is not a dictionary.",
            )
        post_id = rec.get("post_id")
        if not isinstance(post_id, str) or not post_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"Reddit record at index {idx} has missing, empty, or whitespace post_id.",
            )
        trimmed_post_id = post_id.strip()
        if trimmed_post_id in seen_reddit_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate post_id detected in reddit_records.",
            )
        seen_reddit_ids.add(trimmed_post_id)

        # Enrichment fields
        label = rec.get("sentiment_label")
        if label not in PERMITTED_SENTIMENT_LABELS:
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid sentiment_label.",
            )

        score = rec.get("sentiment_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0.0
            or score > 1.0
        ):
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid sentiment_score.",
            )

        status_val = rec.get("sentiment_status")
        if not isinstance(status_val, str) or not status_val.strip():
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has missing or invalid sentiment_status.",
            )

        analyzed_at = rec.get("analyzed_at")
        if not _is_valid_iso_timestamp(analyzed_at):
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid analyzed_at timestamp.",
            )

    # Validate individual YouTube records
    seen_youtube_ids: set[str] = set()
    for idx, rec in enumerate(youtube_records):
        if not isinstance(rec, dict):
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} is not a dictionary.",
            )
        video_id = rec.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"YouTube record at index {idx} has missing, empty, or whitespace video_id.",
            )
        trimmed_video_id = video_id.strip()
        if trimmed_video_id in seen_youtube_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate video_id detected in youtube_records.",
            )
        seen_youtube_ids.add(trimmed_video_id)

        # Enrichment fields
        label = rec.get("sentiment_label")
        if label not in PERMITTED_SENTIMENT_LABELS:
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid sentiment_label.",
            )

        score = rec.get("sentiment_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0.0
            or score > 1.0
        ):
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid sentiment_score.",
            )

        status_val = rec.get("sentiment_status")
        if not isinstance(status_val, str) or not status_val.strip():
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has missing or invalid sentiment_status.",
            )

        analyzed_at = rec.get("analyzed_at")
        if not _is_valid_iso_timestamp(analyzed_at):
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid analyzed_at timestamp.",
            )

    # Complete in-memory summary and JSON payload equality reconciliation
    if summary != payload:
        return (
            False,
            0,
            "schema_violation",
            "In-memory pipeline summary does not match temporary JSON payload.",
        )

    return True, total_records, None, None


# =====================================================================
# STEP 7.5C — Block 5C: Sentiment Stage Execution and Atomic Promotion
# =====================================================================


def _safe_unlink_temp_file(temp_path: Optional[Path]) -> None:
    """Safely unlinks a temporary file, suppressing and logging any OSError.

    Args:
        temp_path: Optional path to the temporary file to remove.
    """
    if temp_path is not None:
        try:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to clean up temporary file: %s",
                type(exc).__name__,
            )


def _validate_active_sentiment_source_file(
    source_name: str,
    file_path: Any,
    expected_count: int,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Validates an active processed source file before sentiment pipeline execution.

    Args:
        source_name: Human-readable source identifier ('Reddit' or 'YouTube').
        file_path: Path or path-like object to validate.
        expected_count: Non-negative expected record count from upstream stage entry.

    Returns:
        tuple of (is_valid: bool, error_category: Optional[str], failure_reason: Optional[str])
    """
    if (
        file_path is None
        or isinstance(file_path, bool)
        or not isinstance(file_path, (str, Path, os.PathLike))
    ):
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input path is missing or invalid",
        )

    try:
        resolved_path = Path(file_path)
    except (TypeError, ValueError):
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input path is invalid",
        )

    if not resolved_path.is_file():
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input file does not exist",
        )

    try:
        if resolved_path.stat().st_size == 0:
            return (
                False,
                "empty_input",
                f"Processed {source_name} input file is empty (0 bytes)",
            )
    except OSError:
        return (
            False,
            "missing_processed_input",
            f"Failed to access processed {source_name} input file",
        )

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return (
            False,
            "corrupt_json",
            f"Processed {source_name} input file contains malformed or unreadable JSON",
        )

    if not isinstance(records, list):
        return (
            False,
            "schema_violation",
            f"Processed {source_name} input file root must be a list",
        )

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            return (
                False,
                "schema_violation",
                f"Record at index {idx} in processed {source_name} input file is not a dictionary",
            )

    if len(records) != expected_count:
        return (
            False,
            "count_mismatch",
            f"Processed {source_name} file count ({len(records)}) does not match upstream manifest count ({expected_count})",
        )

    return True, None, None


def execute_sentiment_stage(
    reddit_processed_path: Optional[Path | str] = None,
    youtube_processed_path: Optional[Path | str] = None,
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    analyzer: Optional[MultilingualSentimentAnalyzer] = None,
    batch_size: int = 32,
    reddit_stage_entry: Optional[dict[str, Any]] = None,
    youtube_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 3 Sentiment Analysis orchestration stage.

    Consumes validated processed Reddit and YouTube records from the current execution.
    Fails closed if upstream provenance is missing, invalid, or malformed.
    Isolates inactive sources using non-existent sentinel paths to prevent loading stale canonical data.
    Enforces atomic output replacement and cleans up temporary files on any failure.

    Args:
        reddit_processed_path: Path to current execution's processed Reddit JSON.
        youtube_processed_path: Path to current execution's processed YouTube JSON.
        skip: When True, stage is skipped without execution.
        canonical_output_path: Destination canonical path for sentiment output.
        config: Optional application configuration.
        analyzer: Optional pre-configured sentiment analyzer instance (e.g. for testing).
        batch_size: Batch size for inference (default 32).
        reddit_stage_entry: Stage entry dict for reddit_preprocessing.
        youtube_stage_entry: Stage entry dict for youtube_preprocessing.

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Sentiment stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream provenance validation (fail closed)
    if reddit_stage_entry is None or youtube_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Sentiment stage failed: missing upstream preprocessing provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream preprocessing provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(reddit_stage_entry, dict) or not isinstance(youtube_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Sentiment stage failed: invalid upstream preprocessing stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_status = reddit_stage_entry.get("status")
    y_status = youtube_stage_entry.get("status")

    if r_status not in PERMITTED_STAGE_STATUSES or y_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Sentiment stage failed: invalid upstream preprocessing status (reddit='%s', youtube='%s').",
            r_status,
            y_status,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_count = reddit_stage_entry.get("record_count")
    y_count = youtube_stage_entry.get("record_count")

    if (
        isinstance(r_count, bool)
        or not isinstance(r_count, int)
        or r_count < 0
        or isinstance(y_count, bool)
        or not isinstance(y_count, int)
        or y_count < 0
    ):
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Sentiment stage failed: invalid upstream preprocessing record count (reddit=%r, youtube=%r).",
            r_count,
            y_count,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing record count",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    # 2. Multi-source dependency resolution
    use_reddit = (r_status == "success")
    use_youtube = (y_status == "success")

    if r_status == "skipped" and y_status == "skipped":
        logger.info("Both upstream preprocessing sources skipped. Skipping sentiment stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="all_sources_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit and not use_youtube:
        logger.warning("Neither upstream preprocessing source succeeded. Skipping sentiment stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit:
        logger.warning(
            "Reddit preprocessing was %s. Sentiment stage proceeding with YouTube only.",
            r_status,
        )
    if not use_youtube:
        logger.warning(
            "YouTube preprocessing was %s. Sentiment stage proceeding with Reddit only.",
            y_status,
        )

    # 3. Canonical path resolution
    resolved_config = config or load_config()
    if canonical_output_path is not None:
        resolved_canonical = Path(canonical_output_path)
    else:
        resolved_canonical = resolved_config.processed_data_path / "sentiment_analysis_results.json"

    # 4. Input file validation for active sources
    if use_reddit:
        valid_r, cat_r, reason_r = _validate_active_sentiment_source_file(
            "Reddit", reddit_processed_path, r_count
        )
        if not valid_r:
            elapsed = time.perf_counter() - start_time
            logger.warning("Sentiment stage input validation failed for Reddit: %s (%s)", reason_r, cat_r)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_r,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_r,
                ),
                None,
            )

    if use_youtube:
        valid_y, cat_y, reason_y = _validate_active_sentiment_source_file(
            "YouTube", youtube_processed_path, y_count
        )
        if not valid_y:
            elapsed = time.perf_counter() - start_time
            logger.warning("Sentiment stage input validation failed for YouTube: %s (%s)", reason_y, cat_y)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_y,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_y,
                ),
                None,
            )

    # 5. Sentinel paths for inactive sources to prevent fallback to canonical files
    if use_reddit:
        reddit_input: Path = Path(reddit_processed_path)
    else:
        reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"
        while reddit_input.exists():
            reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"

    if use_youtube:
        youtube_input: Path = Path(youtube_processed_path)
    else:
        youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"
        while youtube_input.exists():
            youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"

    # 6. Temporary file allocation, pipeline execution, and validation
    temp_output_path: Optional[Path] = None
    try:
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = resolved_canonical.parent / f".tmp_sentiment_{uuid.uuid4().hex}.json"

        logger.info("Executing multilingual sentiment analysis pipeline.")
        summary = run_sentiment_pipeline(
            reddit_input_path=reddit_input,
            youtube_input_path=youtube_input,
            output_path=temp_output_path,
            analyzer=analyzer,
            batch_size=batch_size,
        )

        expected_reddit = r_count if use_reddit else 0
        expected_youtube = y_count if use_youtube else 0

        is_valid, record_count, error_cat, reason = validate_sentiment_output(
            summary=summary,
            temp_file_path=temp_output_path,
            expected_reddit_count=expected_reddit,
            expected_youtube_count=expected_youtube,
        )

        elapsed = time.perf_counter() - start_time

        if not is_valid:
            logger.warning("Sentiment output validation failed: %s (%s)", reason, error_cat)
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=error_cat,
                ),
                None,
            )

        try:
            os.replace(temp_output_path, resolved_canonical)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Failed to atomically promote sentiment output to canonical destination.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason="Atomic output promotion failed",
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="promotion_error",
                ),
                None,
            )

        logger.info(
            "Sentiment stage succeeded with %d records. Canonical output: %s",
            record_count,
            resolved_canonical.name,
        )
        return (
            create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during sentiment stage execution.")
        _safe_unlink_temp_file(temp_output_path)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Sentiment execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# STEP 7.6C — Block 6C: Topic Output Validation
# =====================================================================


def _validate_topic_keywords(keywords: Any) -> tuple[bool, Optional[str]]:
    """Validates topic keyword list structure and contents.

    Args:
        keywords: Object expected to be a list of keyword dictionaries.

    Returns:
        tuple of (is_valid: bool, reason: Optional[str])
    """
    if not isinstance(keywords, list):
        return False, "topic_keywords must be a list."

    for idx, item in enumerate(keywords):
        if not isinstance(item, dict):
            return False, f"Keyword item at index {idx} must be a dictionary."
        word = item.get("word")
        if not isinstance(word, str) or not word.strip():
            return False, f"Keyword item at index {idx} missing valid non-empty 'word'."
        weight = item.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
        ):
            return False, f"Keyword item at index {idx} missing valid numeric 'weight'."

    return True, None


def validate_topic_output(
    summary: Any,
    temp_file_path: Path,
    expected_reddit_count: Optional[int] = None,
    expected_youtube_count: Optional[int] = None,
) -> tuple[bool, int, Optional[str], Optional[str]]:
    """Validates the output of a topic modeling run before promoting to canonical destination.

    Checks:
    1. Pipeline summary is a dict and status is 'success'.
    2. Newly generated temporary output file exists, is a regular file, and is non-empty.
    3. File contents are valid JSON.
    4. Root JSON structure is a dictionary.
    5. Required top-level keys exist with valid types.
    6. Count fields are actual non-negative integers (rejecting bools, floats, strings, negative).
    7. Overall total_documents_modeled matches sum of reddit_records and youtube_records.
    8. Outlier count and percentage reconcile with actual outlier records.
    9. Similarity matrix, trajectories, cross_platform_divergence, and temporal_trends
       conform to expected structural contracts.
    10. In zero-record runs, record collections, topic_info, temporal_trends, and trajectories are empty.
    11. Record counts strictly match expected counts if provided.
    12. Reddit records contain valid, non-empty, unique post_id values and source_platform 'reddit'.
    13. YouTube records contain valid, non-empty, unique video_id values and source_platform 'youtube'.
    14. Every record contains valid topic_id (int >= -1), deterministic topic_status ('assigned' or 'outlier'),
        finite topic_probability in [0.0, 1.0], valid topic_keywords, and valid ISO-8601 topic_modeled_at.
    15. In-memory summary is completely identical to temporary JSON payload.

    Returns:
        tuple of (is_valid: bool, record_count: int, error_category: Optional[str], failure_reason: Optional[str])
    """
    if not isinstance(summary, dict):
        return False, 0, "topic_error", "Topic pipeline summary must be a dictionary object."

    if summary.get("status") != "success":
        return False, 0, "topic_error", "Topic pipeline returned non-success status."

    if (
        temp_file_path is None
        or isinstance(temp_file_path, bool)
        or not isinstance(temp_file_path, (str, Path, os.PathLike))
    ):
        return False, 0, "missing_output", "Topic output file path is invalid."

    try:
        resolved_temp_path = Path(temp_file_path)
    except (TypeError, ValueError):
        return False, 0, "missing_output", "Topic output file path is invalid."

    if not resolved_temp_path.is_file():
        return False, 0, "missing_output", "Topic output file does not exist."

    try:
        if resolved_temp_path.stat().st_size == 0:
            return False, 0, "empty_output", "Topic output file is empty (0 bytes)."
    except OSError:
        return False, 0, "missing_output", "Failed to access topic output file."

    try:
        with open(resolved_temp_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False, 0, "corrupt_json", "Topic output contains malformed or unreadable JSON."

    if not isinstance(payload, dict):
        return (
            False,
            0,
            "schema_violation",
            "Topic output root must be a JSON dictionary object.",
        )

    required_top_keys = (
        "status",
        "generated_at",
        "embedding_model",
        "total_documents_modeled",
        "topic_info",
        "similarity_matrix",
        "trajectories",
        "cross_platform_divergence",
        "temporal_trends",
        "reddit_records",
        "youtube_records",
    )
    for req_key in required_top_keys:
        if req_key not in payload:
            return (
                False,
                0,
                "schema_violation",
                f"Topic output missing required top-level field '{req_key}'.",
            )

    if payload["status"] != "success":
        return False, 0, "schema_violation", "Topic output status must be 'success'."

    if not _is_valid_iso_timestamp(payload["generated_at"]):
        return (
            False,
            0,
            "schema_violation",
            "Field 'generated_at' must be a valid ISO-8601 timestamp string.",
        )

    if not isinstance(payload["embedding_model"], str) or not payload["embedding_model"].strip():
        return (
            False,
            0,
            "schema_violation",
            "Field 'embedding_model' must be a non-empty string.",
        )

    total_records = payload["total_documents_modeled"]
    if isinstance(total_records, bool) or not isinstance(total_records, int) or total_records < 0:
        return (
            False,
            0,
            "count_mismatch",
            "Field 'total_documents_modeled' must be a valid non-negative integer.",
        )

    if not isinstance(payload["topic_info"], list):
        return (
            False,
            0,
            "schema_violation",
            "Field 'topic_info' must be a list.",
        )

    if total_records == 0:
        if len(payload["topic_info"]) != 0:
            return (
                False,
                0,
                "schema_violation",
                "Zero-record topic output must contain empty topic_info list.",
            )
    else:
        for idx, info in enumerate(payload["topic_info"]):
            if not isinstance(info, dict):
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Item at index {idx} in 'topic_info' is not a dictionary.",
                )

            for req_info_key in ("topic_id", "name", "count", "keywords"):
                if req_info_key not in info:
                    return (
                        False,
                        0,
                        "schema_violation",
                        f"Item at index {idx} in 'topic_info' missing required field '{req_info_key}'.",
                    )

            tid = info["topic_id"]
            if isinstance(tid, bool) or not isinstance(tid, int) or tid < -1:
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Item at index {idx} in 'topic_info' has invalid topic_id.",
                )

            t_name = info["name"]
            if not isinstance(t_name, str) or not t_name.strip():
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Item at index {idx} in 'topic_info' has invalid or empty name.",
                )

            t_count = info["count"]
            if isinstance(t_count, bool) or not isinstance(t_count, int) or t_count < 0:
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Item at index {idx} in 'topic_info' has invalid count.",
                )

            t_kws = info["keywords"]
            kw_valid, kw_reason = _validate_topic_keywords(t_kws)
            if not kw_valid:
                return (
                    False,
                    0,
                    "schema_violation",
                    f"Item at index {idx} in 'topic_info' has invalid keywords: {kw_reason}",
                )

    if not isinstance(payload["similarity_matrix"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'similarity_matrix' must be a dictionary.",
        )
    for sim_key in ("topic_ids", "matrix", "pairwise_similarities"):
        if sim_key not in payload["similarity_matrix"]:
            return (
                False,
                0,
                "schema_violation",
                f"Field 'similarity_matrix' missing required key '{sim_key}'.",
            )
    if not isinstance(payload["similarity_matrix"]["topic_ids"], list):
        return (
            False,
            0,
            "schema_violation",
            "Field 'similarity_matrix.topic_ids' must be a list.",
        )
    if not isinstance(payload["similarity_matrix"]["matrix"], list):
        return (
            False,
            0,
            "schema_violation",
            "Field 'similarity_matrix.matrix' must be a list.",
        )
    if not isinstance(payload["similarity_matrix"]["pairwise_similarities"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'similarity_matrix.pairwise_similarities' must be a dictionary.",
        )

    if not isinstance(payload["trajectories"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'trajectories' must be a dictionary.",
        )

    if not isinstance(payload["cross_platform_divergence"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'cross_platform_divergence' must be a dictionary.",
        )
    for cpd_key in ("overall_divergence", "topic_adoption_gaps", "window_divergences"):
        if cpd_key not in payload["cross_platform_divergence"]:
            return (
                False,
                0,
                "schema_violation",
                f"Field 'cross_platform_divergence' missing required key '{cpd_key}'.",
            )
    overall_div = payload["cross_platform_divergence"]["overall_divergence"]
    if (
        isinstance(overall_div, bool)
        or not isinstance(overall_div, (int, float))
        or not math.isfinite(overall_div)
        or overall_div < 0.0
    ):
        return (
            False,
            0,
            "schema_violation",
            "Field 'cross_platform_divergence.overall_divergence' must be a non-negative finite numeric value.",
        )
    if not isinstance(payload["cross_platform_divergence"]["topic_adoption_gaps"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'cross_platform_divergence.topic_adoption_gaps' must be a dictionary.",
        )
    if not isinstance(payload["cross_platform_divergence"]["window_divergences"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'cross_platform_divergence.window_divergences' must be a dictionary.",
        )

    if not isinstance(payload["temporal_trends"], dict):
        return (
            False,
            0,
            "schema_violation",
            "Field 'temporal_trends' must be a dictionary.",
        )

    reddit_records = payload["reddit_records"]
    youtube_records = payload["youtube_records"]

    if not isinstance(reddit_records, list) or not isinstance(youtube_records, list):
        return (
            False,
            0,
            "schema_violation",
            "'reddit_records' and 'youtube_records' must be JSON lists.",
        )

    # Zero-record payload handling
    if total_records == 0:
        if len(reddit_records) != 0 or len(youtube_records) != 0:
            return (
                False,
                0,
                "count_mismatch",
                "Zero-record topic output must contain empty record lists.",
            )
        if len(payload["topic_info"]) != 0:
            return (
                False,
                0,
                "schema_violation",
                "Zero-record topic output must contain empty topic_info list.",
            )
        if payload["temporal_trends"] != {}:
            return (
                False,
                0,
                "schema_violation",
                "Zero-record topic output must contain empty temporal_trends dictionary.",
            )
        if payload["trajectories"] != {}:
            return (
                False,
                0,
                "schema_violation",
                "Zero-record topic output must contain empty trajectories dictionary.",
            )

        if expected_reddit_count is not None:
            if (
                isinstance(expected_reddit_count, bool)
                or not isinstance(expected_reddit_count, int)
                or expected_reddit_count < 0
            ):
                return (
                    False,
                    0,
                    "count_mismatch",
                    "expected_reddit_count must be a valid non-negative integer.",
                )
            if expected_reddit_count != 0:
                return (
                    False,
                    0,
                    "count_mismatch",
                    f"Reddit record count (0) does not match expected count ({expected_reddit_count}).",
                )

        if expected_youtube_count is not None:
            if (
                isinstance(expected_youtube_count, bool)
                or not isinstance(expected_youtube_count, int)
                or expected_youtube_count < 0
            ):
                return (
                    False,
                    0,
                    "count_mismatch",
                    "expected_youtube_count must be a valid non-negative integer.",
                )
            if expected_youtube_count != 0:
                return (
                    False,
                    0,
                    "count_mismatch",
                    f"YouTube record count (0) does not match expected count ({expected_youtube_count}).",
                )

        if summary != payload:
            return (
                False,
                0,
                "schema_violation",
                "In-memory topic pipeline summary does not match temporary JSON payload.",
            )

        return True, 0, None, None

    # Non-zero record payload validation
    if "outlier_count" not in payload:
        return (
            False,
            0,
            "schema_violation",
            "Topic output missing required field 'outlier_count'.",
        )
    outlier_count = payload["outlier_count"]
    if isinstance(outlier_count, bool) or not isinstance(outlier_count, int) or outlier_count < 0:
        return (
            False,
            0,
            "count_mismatch",
            "Field 'outlier_count' must be a valid non-negative integer.",
        )

    if "outlier_pct" not in payload:
        return (
            False,
            0,
            "schema_violation",
            "Topic output missing required field 'outlier_pct'.",
        )
    outlier_pct = payload["outlier_pct"]
    if (
        isinstance(outlier_pct, bool)
        or not isinstance(outlier_pct, (int, float))
        or not math.isfinite(outlier_pct)
        or outlier_pct < 0.0
        or outlier_pct > 100.0
    ):
        return (
            False,
            0,
            "count_mismatch",
            "Field 'outlier_pct' must be a valid percentage in range [0.0, 100.0].",
        )

    temporal = payload["temporal_trends"]
    for win_key in ("daily", "weekly"):
        if win_key not in temporal:
            return (
                False,
                0,
                "schema_violation",
                f"Field 'temporal_trends' missing required window '{win_key}'.",
            )
        if not isinstance(temporal[win_key], dict):
            return (
                False,
                0,
                "schema_violation",
                f"Temporal window '{win_key}' must be a dictionary.",
            )

    if len(reddit_records) + len(youtube_records) != total_records:
        return (
            False,
            0,
            "count_mismatch",
            "Sum of record counts does not match total_documents_modeled.",
        )

    # Expected upstream count reconciliation
    if expected_reddit_count is not None:
        if (
            isinstance(expected_reddit_count, bool)
            or not isinstance(expected_reddit_count, int)
            or expected_reddit_count < 0
        ):
            return (
                False,
                0,
                "count_mismatch",
                "expected_reddit_count must be a valid non-negative integer.",
            )
        if len(reddit_records) != expected_reddit_count:
            return (
                False,
                0,
                "count_mismatch",
                f"Reddit record count ({len(reddit_records)}) does not match expected count ({expected_reddit_count}).",
            )

    if expected_youtube_count is not None:
        if (
            isinstance(expected_youtube_count, bool)
            or not isinstance(expected_youtube_count, int)
            or expected_youtube_count < 0
        ):
            return (
                False,
                0,
                "count_mismatch",
                "expected_youtube_count must be a valid non-negative integer.",
            )
        if len(youtube_records) != expected_youtube_count:
            return (
                False,
                0,
                "count_mismatch",
                f"YouTube record count ({len(youtube_records)}) does not match expected count ({expected_youtube_count}).",
            )

    actual_outliers = 0

    # Validate individual Reddit records
    seen_reddit_ids: set[str] = set()
    for idx, rec in enumerate(reddit_records):
        if not isinstance(rec, dict):
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} is not a dictionary.",
            )
        post_id = rec.get("post_id")
        if not isinstance(post_id, str) or not post_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"Reddit record at index {idx} has missing, empty, or whitespace post_id.",
            )
        trimmed_post_id = post_id.strip()
        if trimmed_post_id in seen_reddit_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate post_id detected in reddit_records.",
            )
        seen_reddit_ids.add(trimmed_post_id)

        if rec.get("source_platform") != "reddit":
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid source_platform.",
            )

        topic_id = rec.get("topic_id")
        if isinstance(topic_id, bool) or not isinstance(topic_id, int) or topic_id < -1:
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid topic_id.",
            )

        topic_status = rec.get("topic_status")
        expected_status = "assigned" if topic_id >= 0 else "outlier"
        if topic_status != expected_status:
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has inconsistent topic_status '{topic_status}' for topic_id {topic_id}.",
            )

        if topic_id == -1:
            actual_outliers += 1

        prob = rec.get("topic_probability")
        if (
            isinstance(prob, bool)
            or not isinstance(prob, (int, float))
            or not math.isfinite(prob)
            or prob < 0.0
            or prob > 1.0
        ):
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid topic_probability.",
            )

        keywords = rec.get("topic_keywords")
        kw_valid, kw_reason = _validate_topic_keywords(keywords)
        if not kw_valid:
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid topic_keywords: {kw_reason}",
            )

        modeled_at = rec.get("topic_modeled_at")
        if not _is_valid_iso_timestamp(modeled_at):
            return (
                False,
                0,
                "schema_violation",
                f"Reddit record at index {idx} has invalid topic_modeled_at timestamp.",
            )

    # Validate individual YouTube records
    seen_youtube_ids: set[str] = set()
    for idx, rec in enumerate(youtube_records):
        if not isinstance(rec, dict):
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} is not a dictionary.",
            )
        video_id = rec.get("video_id")
        if not isinstance(video_id, str) or not video_id.strip():
            return (
                False,
                0,
                "invalid_record_id",
                f"YouTube record at index {idx} has missing, empty, or whitespace video_id.",
            )
        trimmed_video_id = video_id.strip()
        if trimmed_video_id in seen_youtube_ids:
            return (
                False,
                0,
                "duplicate_record_id",
                "Duplicate video_id detected in youtube_records.",
            )
        seen_youtube_ids.add(trimmed_video_id)

        if rec.get("source_platform") != "youtube":
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid source_platform.",
            )

        topic_id = rec.get("topic_id")
        if isinstance(topic_id, bool) or not isinstance(topic_id, int) or topic_id < -1:
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid topic_id.",
            )

        topic_status = rec.get("topic_status")
        expected_status = "assigned" if topic_id >= 0 else "outlier"
        if topic_status != expected_status:
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has inconsistent topic_status '{topic_status}' for topic_id {topic_id}.",
            )

        if topic_id == -1:
            actual_outliers += 1

        prob = rec.get("topic_probability")
        if (
            isinstance(prob, bool)
            or not isinstance(prob, (int, float))
            or not math.isfinite(prob)
            or prob < 0.0
            or prob > 1.0
        ):
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid topic_probability.",
            )

        keywords = rec.get("topic_keywords")
        kw_valid, kw_reason = _validate_topic_keywords(keywords)
        if not kw_valid:
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid topic_keywords: {kw_reason}",
            )

        modeled_at = rec.get("topic_modeled_at")
        if not _is_valid_iso_timestamp(modeled_at):
            return (
                False,
                0,
                "schema_violation",
                f"YouTube record at index {idx} has invalid topic_modeled_at timestamp.",
            )

    # Reconcile outlier statistics
    if outlier_count != actual_outliers:
        return (
            False,
            0,
            "count_mismatch",
            f"Payload outlier_count ({outlier_count}) does not match actual count of outlier records ({actual_outliers}).",
        )

    expected_outlier_pct = round((actual_outliers / total_records) * 100.0, 2)
    if abs(float(outlier_pct) - expected_outlier_pct) > 0.01:
        return (
            False,
            0,
            "count_mismatch",
            f"Payload outlier_pct ({outlier_pct}) does not match calculated percentage ({expected_outlier_pct}).",
        )

    # Complete in-memory summary and JSON payload equality reconciliation
    if summary != payload:
        return (
            False,
            0,
            "schema_violation",
            "In-memory topic pipeline summary does not match temporary JSON payload.",
        )

    return True, total_records, None, None


# =====================================================================
# STEP 7.6D — Block 6D: Topic Stage Execution and Atomic Promotion
# =====================================================================


def _validate_active_topic_source_file(
    source_name: str,
    file_path: Any,
    expected_count: int,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Validates an active processed source file before topic pipeline execution.

    Args:
        source_name: Human-readable source identifier ('Reddit' or 'YouTube').
        file_path: Path or path-like object to validate.
        expected_count: Non-negative expected record count from upstream stage entry.

    Returns:
        tuple of (is_valid: bool, error_category: Optional[str], failure_reason: Optional[str])
    """
    if (
        file_path is None
        or isinstance(file_path, bool)
        or not isinstance(file_path, (str, Path, os.PathLike))
    ):
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input path is missing or invalid",
        )

    try:
        resolved_path = Path(file_path)
    except (TypeError, ValueError):
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input path is invalid",
        )

    if not resolved_path.is_file():
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input file does not exist",
        )

    try:
        if resolved_path.stat().st_size == 0:
            return (
                False,
                "empty_input",
                f"Processed {source_name} input file is empty (0 bytes)",
            )
    except OSError:
        return (
            False,
            "missing_processed_input",
            f"Failed to access processed {source_name} input file",
        )

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return (
            False,
            "corrupt_json",
            f"Processed {source_name} input file contains malformed or unreadable JSON",
        )

    if not isinstance(records, list):
        return (
            False,
            "schema_violation",
            f"Processed {source_name} input file root must be a list",
        )

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            return (
                False,
                "schema_violation",
                f"Record at index {idx} in processed {source_name} input file is not a dictionary",
            )

    if len(records) != expected_count:
        return (
            False,
            "count_mismatch",
            f"Processed {source_name} file count ({len(records)}) does not match upstream manifest count ({expected_count})",
        )

    return True, None, None


def execute_topic_stage(
    reddit_processed_path: Optional[Path | str] = None,
    youtube_processed_path: Optional[Path | str] = None,
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    modeler: Optional[SocialTopicModeler] = None,
    window: str = "daily",
    reddit_stage_entry: Optional[dict[str, Any]] = None,
    youtube_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 4 Topic Modeling orchestration stage.

    Consumes validated processed Reddit and YouTube records from the current execution.
    Does NOT depend on Stage 5 sentiment analysis.
    Fails closed if upstream preprocessing provenance is missing, invalid, or malformed.
    Isolates inactive sources using non-existent sentinel paths to prevent loading stale canonical data.
    Enforces atomic output replacement and safely cleans up temporary files on any failure.

    Args:
        reddit_processed_path: Path to current execution's processed Reddit JSON.
        youtube_processed_path: Path to current execution's processed YouTube JSON.
        skip: When True, stage is skipped without execution.
        canonical_output_path: Destination canonical path for topic modeling output.
        config: Optional application configuration.
        modeler: Optional SocialTopicModeler instance (supports mock injection).
        window: Temporal aggregation window ('daily' or 'weekly', default 'daily').
        reddit_stage_entry: Stage entry dict for reddit_preprocessing.
        youtube_stage_entry: Stage entry dict for youtube_preprocessing.

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Topic modeling stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream provenance validation (fail closed)
    if reddit_stage_entry is None or youtube_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Topic stage failed: missing upstream preprocessing provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream preprocessing provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(reddit_stage_entry, dict) or not isinstance(youtube_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Topic stage failed: invalid upstream preprocessing stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_status = reddit_stage_entry.get("status")
    y_status = youtube_stage_entry.get("status")

    if r_status not in PERMITTED_STAGE_STATUSES or y_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Topic stage failed: invalid upstream preprocessing status (reddit='%s', youtube='%s').",
            r_status,
            y_status,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_count = reddit_stage_entry.get("record_count")
    y_count = youtube_stage_entry.get("record_count")

    if (
        isinstance(r_count, bool)
        or not isinstance(r_count, int)
        or r_count < 0
        or isinstance(y_count, bool)
        or not isinstance(y_count, int)
        or y_count < 0
    ):
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Topic stage failed: invalid upstream preprocessing record count (reddit=%r, youtube=%r).",
            r_count,
            y_count,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing record count",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    # 2. Multi-source dependency resolution
    use_reddit = (r_status == "success")
    use_youtube = (y_status == "success")

    if r_status == "skipped" and y_status == "skipped":
        logger.info("Both upstream preprocessing sources skipped. Skipping topic stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="all_sources_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit and not use_youtube:
        logger.warning("Neither upstream preprocessing source succeeded. Skipping topic stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit:
        logger.warning(
            "Reddit preprocessing was %s. Topic stage proceeding with YouTube only.",
            r_status,
        )
    if not use_youtube:
        logger.warning(
            "YouTube preprocessing was %s. Topic stage proceeding with Reddit only.",
            y_status,
        )

    # 3. Canonical path resolution
    resolved_config = config or load_config()
    if canonical_output_path is not None:
        resolved_canonical = Path(canonical_output_path)
    else:
        resolved_canonical = resolved_config.processed_data_path / "topic_modeling_results.json"

    # 4. Input file validation for active sources
    if use_reddit:
        valid_r, cat_r, reason_r = _validate_active_topic_source_file(
            "Reddit", reddit_processed_path, r_count
        )
        if not valid_r:
            elapsed = time.perf_counter() - start_time
            logger.warning("Topic stage input validation failed for Reddit: %s (%s)", reason_r, cat_r)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_r,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_r,
                ),
                None,
            )

    if use_youtube:
        valid_y, cat_y, reason_y = _validate_active_topic_source_file(
            "YouTube", youtube_processed_path, y_count
        )
        if not valid_y:
            elapsed = time.perf_counter() - start_time
            logger.warning("Topic stage input validation failed for YouTube: %s (%s)", reason_y, cat_y)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_y,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_y,
                ),
                None,
            )

    # 5. Sentinel paths for inactive sources to prevent fallback to canonical files
    if use_reddit:
        reddit_input: Path = Path(reddit_processed_path)
    else:
        reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"
        while reddit_input.exists():
            reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"

    if use_youtube:
        youtube_input: Path = Path(youtube_processed_path)
    else:
        youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"
        while youtube_input.exists():
            youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"

    # 6. Temporary file allocation, pipeline execution, and validation
    temp_output_path: Optional[Path] = None
    try:
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = resolved_canonical.parent / f".tmp_topics_{uuid.uuid4().hex}.json"
        while temp_output_path.exists():
            temp_output_path = resolved_canonical.parent / f".tmp_topics_{uuid.uuid4().hex}.json"

        logger.info("Executing advanced topic modeling pipeline.")
        summary = run_topic_pipeline(
            reddit_input_path=reddit_input,
            youtube_input_path=youtube_input,
            output_path=temp_output_path,
            modeler=modeler,
            window=window,
        )

        expected_reddit = r_count if use_reddit else 0
        expected_youtube = y_count if use_youtube else 0

        is_valid, record_count, error_cat, reason = validate_topic_output(
            summary=summary,
            temp_file_path=temp_output_path,
            expected_reddit_count=expected_reddit,
            expected_youtube_count=expected_youtube,
        )

        elapsed = time.perf_counter() - start_time

        if not is_valid:
            logger.warning("Topic output validation failed: %s (%s)", reason, error_cat)
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=error_cat,
                ),
                None,
            )

        try:
            os.replace(temp_output_path, resolved_canonical)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Failed to atomically promote topic output to canonical destination.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason="Atomic output promotion failed",
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="promotion_error",
                ),
                None,
            )

        logger.info(
            "Topic modeling stage succeeded with %d records. Canonical output: %s",
            record_count,
            resolved_canonical.name,
        )
        return (
            create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during topic stage execution.")
        _safe_unlink_temp_file(temp_output_path)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Topic execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# Company Step 5 Graph Analysis Stage (Stage 7)
# =====================================================================


def _validate_active_processed_source_file(
    source_name: str,
    file_path: Any,
    expected_count: int,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Validates an active processed source file before analytical stage execution.

    Args:
        source_name: Human-readable source identifier ('Reddit' or 'YouTube').
        file_path: Path or path-like object to validate.
        expected_count: Non-negative expected record count from upstream stage entry.

    Returns:
        tuple of (is_valid: bool, error_category: Optional[str], failure_reason: Optional[str])
    """
    if (
        file_path is None
        or isinstance(file_path, bool)
        or not isinstance(file_path, (str, Path, os.PathLike))
    ):
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input path is missing or invalid",
        )

    try:
        resolved_path = Path(file_path)
    except (TypeError, ValueError):
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input path is invalid",
        )

    if not resolved_path.is_file():
        return (
            False,
            "missing_processed_input",
            f"Processed {source_name} input file does not exist",
        )

    try:
        if resolved_path.stat().st_size == 0:
            return (
                False,
                "empty_input",
                f"Processed {source_name} input file is empty (0 bytes)",
            )
    except OSError:
        return (
            False,
            "missing_processed_input",
            f"Failed to access processed {source_name} input file",
        )

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return (
            False,
            "corrupt_json",
            f"Processed {source_name} input file contains malformed or unreadable JSON",
        )

    if not isinstance(records, list):
        return (
            False,
            "schema_violation",
            f"Processed {source_name} input file root must be a list",
        )

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            return (
                False,
                "schema_violation",
                f"Record at index {idx} in processed {source_name} input file is not a dictionary",
            )

    if len(records) != expected_count:
        return (
            False,
            "count_mismatch",
            f"Processed {source_name} file count ({len(records)}) does not match upstream manifest count ({expected_count})",
        )

    return True, None, None


def execute_graph_stage(
    reddit_processed_path: Optional[Path | str] = None,
    youtube_processed_path: Optional[Path | str] = None,
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    builder: Optional[InteractionGraphBuilder] = None,
    ranker: Optional[InfluencerRanker] = None,
    neo4j_manager: Optional[Neo4jGraphManager] = None,
    top_n_influencers: int = 50,
    reddit_stage_entry: Optional[dict[str, Any]] = None,
    youtube_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 5 Graph Analysis orchestration stage.

    Consumes validated processed Reddit and YouTube records from the current execution.
    Fails closed if upstream preprocessing provenance is missing, invalid, or malformed.
    Isolates inactive sources using non-existent sentinel paths to prevent loading stale canonical data.
    Enforces atomic output replacement and safely cleans up temporary files on any failure.

    Args:
        reddit_processed_path: Path to current execution's processed Reddit JSON.
        youtube_processed_path: Path to current execution's processed YouTube JSON.
        skip: When True, stage is skipped without execution.
        canonical_output_path: Destination canonical path for graph analysis output.
        config: Optional application configuration.
        builder: Optional InteractionGraphBuilder instance (supports mock injection).
        ranker: Optional InfluencerRanker instance (supports mock injection).
        neo4j_manager: Optional Neo4jGraphManager instance (supports mock injection).
        top_n_influencers: Maximum influencers to rank (default 50).
        reddit_stage_entry: Stage entry dict for reddit_preprocessing.
        youtube_stage_entry: Stage entry dict for youtube_preprocessing.

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Graph analysis stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream provenance validation (fail closed)
    if reddit_stage_entry is None or youtube_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Graph stage failed: missing upstream preprocessing provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream preprocessing provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(reddit_stage_entry, dict) or not isinstance(youtube_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Graph stage failed: invalid upstream preprocessing stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_status = reddit_stage_entry.get("status")
    y_status = youtube_stage_entry.get("status")

    if r_status not in PERMITTED_STAGE_STATUSES or y_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Graph stage failed: invalid upstream preprocessing status (reddit='%s', youtube='%s').",
            r_status,
            y_status,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_count = reddit_stage_entry.get("record_count")
    y_count = youtube_stage_entry.get("record_count")

    if (
        isinstance(r_count, bool)
        or not isinstance(r_count, int)
        or r_count < 0
        or isinstance(y_count, bool)
        or not isinstance(y_count, int)
        or y_count < 0
    ):
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Graph stage failed: invalid upstream preprocessing record count (reddit=%r, youtube=%r).",
            r_count,
            y_count,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing record count",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    # 2. Multi-source dependency resolution
    use_reddit = (r_status == "success")
    use_youtube = (y_status == "success")

    if r_status == "skipped" and y_status == "skipped":
        logger.info("Both upstream preprocessing sources skipped. Skipping graph stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="all_sources_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit and not use_youtube:
        logger.warning("Neither upstream preprocessing source succeeded. Skipping graph stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit:
        logger.warning("Reddit preprocessing was %s. Graph stage proceeding with YouTube only.", r_status)
    if not use_youtube:
        logger.warning("YouTube preprocessing was %s. Graph stage proceeding with Reddit only.", y_status)

    # 3. Canonical path resolution
    resolved_config = config or load_config()
    if canonical_output_path is not None:
        resolved_canonical = Path(canonical_output_path)
    else:
        resolved_canonical = resolved_config.processed_data_path / "graph_analysis_results.json"

    # 4. Input file validation for active sources
    if use_reddit:
        valid_r, cat_r, reason_r = _validate_active_processed_source_file("Reddit", reddit_processed_path, r_count)
        if not valid_r:
            elapsed = time.perf_counter() - start_time
            logger.warning("Graph stage input validation failed for Reddit: %s (%s)", reason_r, cat_r)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_r,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_r,
                ),
                None,
            )

    if use_youtube:
        valid_y, cat_y, reason_y = _validate_active_processed_source_file("YouTube", youtube_processed_path, y_count)
        if not valid_y:
            elapsed = time.perf_counter() - start_time
            logger.warning("Graph stage input validation failed for YouTube: %s (%s)", reason_y, cat_y)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_y,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_y,
                ),
                None,
            )

    # 5. Sentinel paths for inactive sources to prevent fallback to canonical files
    if use_reddit:
        reddit_input: Path = Path(reddit_processed_path)
    else:
        reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"
        while reddit_input.exists():
            reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"

    if use_youtube:
        youtube_input: Path = Path(youtube_processed_path)
    else:
        youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"
        while youtube_input.exists():
            youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"

    # 6. Temporary file allocation, pipeline execution, and validation
    temp_output_path: Optional[Path] = None
    try:
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = resolved_canonical.parent / f".tmp_graph_{uuid.uuid4().hex}.json"
        while temp_output_path.exists():
            temp_output_path = resolved_canonical.parent / f".tmp_graph_{uuid.uuid4().hex}.json"

        logger.info("Executing interaction graph analysis pipeline.")
        summary = run_graph_pipeline(
            reddit_input_path=reddit_input,
            youtube_input_path=youtube_input,
            output_path=temp_output_path,
            top_n_influencers=top_n_influencers,
            builder=builder,
            ranker=ranker,
            neo4j_manager=neo4j_manager,
        )

        status_val = summary.get("status")
        elapsed = time.perf_counter() - start_time

        if status_val == "error":
            logger.warning("Graph pipeline execution returned error status.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=summary.get("error", "Graph analysis failed"),
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="execution_error",
                ),
                None,
            )

        try:
            os.replace(temp_output_path, resolved_canonical)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Failed to atomically promote graph output to canonical destination.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason="Atomic output promotion failed",
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="promotion_error",
                ),
                None,
            )

        final_status = status_val if status_val in PERMITTED_STAGE_STATUSES else "success"
        record_count = summary.get("graph_metrics", {}).get("total_users", 0)

        logger.info(
            "Graph analysis stage succeeded with status '%s' and %d user nodes. Canonical output: %s",
            final_status,
            record_count,
            resolved_canonical.name,
        )
        return (
            create_initial_stage_entry(
                status=final_status,
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during graph stage execution.")
        _safe_unlink_temp_file(temp_output_path)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Graph execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# Company Step 6 Neo4j Persistence Stage (Stage 8)
# =====================================================================


def execute_neo4j_stage(
    graph_artifact_path: Optional[Path | str] = None,
    skip: bool = False,
    graph_stage_entry: Optional[dict[str, Any]] = None,
    manager: Optional[Neo4jGraphManager] = None,
    manager_factory: Optional[Callable[[], Neo4jGraphManager]] = None,
    batch_size: int = 500,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 6 Neo4j Persistence orchestration stage.

    Consumes validated Stage 7 graph analysis results JSON.
    Skips cleanly if Neo4j is skipped by user or if Stage 7 was skipped or failed.
    Handles empty valid graphs as successful no-ops without error.
    Records committed nodes and relationships counts.

    Args:
        graph_artifact_path: Optional path to Stage 7 graph results JSON.
        skip: When True, stage is skipped without execution.
        graph_stage_entry: Stage entry dict for graph (Stage 7).
        manager: Optional injected Neo4jGraphManager instance.
        manager_factory: Optional factory callable returning a Neo4jGraphManager.
        batch_size: Cypher UNWIND batch size (default 500).

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Neo4j persistence stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream graph provenance validation
    if graph_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Neo4j stage failed: missing upstream graph provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream graph provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(graph_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Neo4j stage failed: invalid upstream graph stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream graph stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    g_status = graph_stage_entry.get("status")
    if g_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning("Neo4j stage failed: invalid upstream graph status '%s'.", g_status)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream graph status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    if g_status == "skipped":
        logger.info("Stage 7 graph was skipped. Skipping Neo4j persistence stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if g_status == "failed":
        logger.warning("Stage 7 graph failed. Skipping Neo4j persistence stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 2. Target artifact path resolution
    if graph_artifact_path is not None:
        target_path = Path(graph_artifact_path)
    else:
        elapsed = time.perf_counter() - start_time
        logger.warning("Neo4j stage failed: graph artifact path is missing.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Graph artifact path is missing or None",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_artifact",
            ),
            None,
        )

    # If graph is empty and file doesn't exist, handle as valid empty no-op
    if g_status == "empty" and (not target_path.exists() or target_path.stat().st_size == 0):
        elapsed = time.perf_counter() - start_time
        logger.info("Stage 7 graph is empty and artifact is absent; Neo4j persistence successful no-op.")
        return (
            create_initial_stage_entry(
                status="empty",
                reason="empty_graph",
                record_count=0,
                duration_seconds=elapsed,
                error_category=None,
            ),
            None,
        )

    if not target_path.exists():
        elapsed = time.perf_counter() - start_time
        logger.warning("Neo4j stage failed: graph artifact not found at %s", target_path)
        return (
            create_initial_stage_entry(
                status="failed",
                reason=f"Graph artifact not found at {target_path}",
                record_count=0,
                duration_seconds=elapsed,
                error_category="artifact_not_found",
            ),
            None,
        )

    # 3. Execute Neo4j persistence
    try:
        logger.info("Executing Neo4j graph persistence pipeline.")
        report = run_neo4j_persistence(
            artifact_path=target_path,
            manager=manager,
            manager_factory=manager_factory,
            batch_size=batch_size,
        )
        elapsed = time.perf_counter() - start_time
        rep_status = report.get("status")

        if rep_status == "success":
            committed = report.get("committed_nodes", 0) + report.get("committed_relationships", 0)
            outcome = report.get("persistence_outcome")
            final_status = "empty" if (outcome == "skipped_empty" or g_status == "empty") else "success"
            logger.info(
                "Neo4j persistence completed with status '%s' (%d entities committed).",
                final_status,
                committed,
            )
            return (
                create_initial_stage_entry(
                    status=final_status,
                    reason=None,
                    record_count=committed,
                    duration_seconds=elapsed,
                    error_category=None,
                ),
                None,
            )
        else:
            err_cat = report.get("error_category", "persistence_error")
            reason = report.get("error_message", "Neo4j persistence failed")
            logger.warning("Neo4j persistence failed: %s (%s)", reason, err_cat)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=err_cat,
                ),
                None,
            )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during Neo4j stage execution.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Neo4j execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# Company Step 7 Misinformation Detection Stage (Stage 9)
# =====================================================================


def execute_misinformation_stage(
    reddit_processed_path: Optional[Path | str] = None,
    youtube_processed_path: Optional[Path | str] = None,
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    dataset_path: Optional[Path | str] = None,
    classifier: Optional[Any] = None,
    feature_extractor: Optional[Any] = None,
    detector: Optional[Any] = None,
    save_model_to: Optional[Path | str] = None,
    binary: bool = True,
    uncertain_policy: str = "filter",
    reddit_stage_entry: Optional[dict[str, Any]] = None,
    youtube_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 7 Misinformation Detection orchestration stage.

    Consumes validated processed Reddit and YouTube records from the current execution.
    Fails closed if upstream preprocessing provenance is missing, invalid, or malformed.
    Isolates inactive sources using non-existent sentinel paths to prevent loading stale canonical data.
    Enforces atomic output replacement and safely cleans up temporary files on any failure.

    Args:
        reddit_processed_path: Path to current execution's processed Reddit JSON.
        youtube_processed_path: Path to current execution's processed YouTube JSON.
        skip: When True, stage is skipped without execution.
        canonical_output_path: Destination canonical path for misinformation output.
        config: Optional application configuration.
        dataset_path: Optional path to external LIAR benchmark dataset.
        classifier: Optional injected MisinformationClassifier.
        feature_extractor: Optional injected LinguisticFeatureExtractor.
        detector: Optional injected MisinformationDetector.
        save_model_to: Optional destination to persist fitted model via joblib.
        binary: If True, uses binary targets ('fake' vs 'real').
        uncertain_policy: Policy for handling 'half-true' labels ('filter' or 'include_real').
        reddit_stage_entry: Stage entry dict for reddit_preprocessing.
        youtube_stage_entry: Stage entry dict for youtube_preprocessing.

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Misinformation stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream provenance validation (fail closed)
    if reddit_stage_entry is None or youtube_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Misinformation stage failed: missing upstream preprocessing provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream preprocessing provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(reddit_stage_entry, dict) or not isinstance(youtube_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Misinformation stage failed: invalid upstream preprocessing stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_status = reddit_stage_entry.get("status")
    y_status = youtube_stage_entry.get("status")

    if r_status not in PERMITTED_STAGE_STATUSES or y_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Misinformation stage failed: invalid upstream preprocessing status (reddit='%s', youtube='%s').",
            r_status,
            y_status,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_count = reddit_stage_entry.get("record_count")
    y_count = youtube_stage_entry.get("record_count")

    if (
        isinstance(r_count, bool)
        or not isinstance(r_count, int)
        or r_count < 0
        or isinstance(y_count, bool)
        or not isinstance(y_count, int)
        or y_count < 0
    ):
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Misinformation stage failed: invalid upstream preprocessing record count (reddit=%r, youtube=%r).",
            r_count,
            y_count,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing record count",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    # 2. Multi-source dependency resolution
    use_reddit = (r_status == "success")
    use_youtube = (y_status == "success")

    if r_status == "skipped" and y_status == "skipped":
        logger.info("Both upstream preprocessing sources skipped. Skipping misinformation stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="all_sources_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit and not use_youtube:
        logger.warning("Neither upstream preprocessing source succeeded. Skipping misinformation stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if not use_reddit:
        logger.warning("Reddit preprocessing was %s. Misinformation stage proceeding with YouTube only.", r_status)
    if not use_youtube:
        logger.warning("YouTube preprocessing was %s. Misinformation stage proceeding with Reddit only.", y_status)

    # 3. Canonical path resolution
    resolved_config = config or load_config()
    if canonical_output_path is not None:
        resolved_canonical = Path(canonical_output_path)
    else:
        resolved_canonical = resolved_config.processed_data_path / "misinformation_analysis_results.json"

    # 4. Input file validation for active sources
    if use_reddit:
        valid_r, cat_r, reason_r = _validate_active_processed_source_file("Reddit", reddit_processed_path, r_count)
        if not valid_r:
            elapsed = time.perf_counter() - start_time
            logger.warning("Misinformation stage input validation failed for Reddit: %s (%s)", reason_r, cat_r)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_r,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_r,
                ),
                None,
            )

    if use_youtube:
        valid_y, cat_y, reason_y = _validate_active_processed_source_file("YouTube", youtube_processed_path, y_count)
        if not valid_y:
            elapsed = time.perf_counter() - start_time
            logger.warning("Misinformation stage input validation failed for YouTube: %s (%s)", reason_y, cat_y)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason_y,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category=cat_y,
                ),
                None,
            )

    # 5. Sentinel paths for inactive sources to prevent fallback to canonical files
    if use_reddit:
        reddit_input: Path = Path(reddit_processed_path)
    else:
        reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"
        while reddit_input.exists():
            reddit_input = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"

    if use_youtube:
        youtube_input: Path = Path(youtube_processed_path)
    else:
        youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"
        while youtube_input.exists():
            youtube_input = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"

    # 6. Temporary file allocation, pipeline execution, and validation
    temp_output_path: Optional[Path] = None
    try:
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = resolved_canonical.parent / f".tmp_misinformation_{uuid.uuid4().hex}.json"
        while temp_output_path.exists():
            temp_output_path = resolved_canonical.parent / f".tmp_misinformation_{uuid.uuid4().hex}.json"

        logger.info("Executing misinformation detection pipeline.")
        summary = run_misinformation_pipeline(
            reddit_path=reddit_input,
            youtube_path=youtube_input,
            output_path=temp_output_path,
            dataset_path=dataset_path,
            classifier=classifier,
            feature_extractor=feature_extractor,
            detector=detector,
            save_model_to=save_model_to,
            binary=binary,
            uncertain_policy=uncertain_policy,
        )

        status_val = summary.get("status")
        elapsed = time.perf_counter() - start_time

        if status_val == "error":
            logger.warning("Misinformation pipeline execution returned error status.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=summary.get("error", "Misinformation analysis failed"),
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="execution_error",
                ),
                None,
            )

        try:
            os.replace(temp_output_path, resolved_canonical)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Failed to atomically promote misinformation output to canonical destination.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason="Atomic output promotion failed",
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="promotion_error",
                ),
                None,
            )

        final_status = status_val if status_val in PERMITTED_STAGE_STATUSES else "success"
        record_count = summary.get("total_records_processed", 0)

        logger.info(
            "Misinformation detection stage succeeded with status '%s' and %d records. Canonical output: %s",
            final_status,
            record_count,
            resolved_canonical.name,
        )
        return (
            create_initial_stage_entry(
                status=final_status,
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during misinformation stage execution.")
        _safe_unlink_temp_file(temp_output_path)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Misinformation execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# Company Step 8 Social Intelligence Consolidation Stage (Stage 10)
# =====================================================================


def execute_consolidation_stage(
    reddit_processed_path: Optional[Path | str] = None,
    youtube_processed_path: Optional[Path | str] = None,
    sentiment_path: Optional[Path | str] = None,
    topics_path: Optional[Path | str] = None,
    graph_path: Optional[Path | str] = None,
    misinformation_path: Optional[Path | str] = None,
    skip: bool = False,
    canonical_output_path: Optional[Path | str] = None,
    config: Optional[AppConfig] = None,
    execution_time: Optional[str] = None,
    reddit_stage_entry: Optional[dict[str, Any]] = None,
    youtube_stage_entry: Optional[dict[str, Any]] = None,
    sentiment_stage_entry: Optional[dict[str, Any]] = None,
    topic_stage_entry: Optional[dict[str, Any]] = None,
    graph_stage_entry: Optional[dict[str, Any]] = None,
    misinformation_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 8 Social Intelligence Consolidation orchestration stage.

    Consumes validated processed Reddit and YouTube base records, joining analytical
    results (sentiment, topics, graph, misinformation).
    Uses sentinel non-existent paths for inactive or failed sources to prevent stale file reading.
    Validates consolidated output envelope with ConsolidatedArtifactValidator.
    Enforces atomic output replacement and safely cleans up temporary files on any failure.

    Args:
        reddit_processed_path: Path to current execution's processed Reddit JSON.
        youtube_processed_path: Path to current execution's processed YouTube JSON.
        sentiment_path: Path to current execution's sentiment analysis results JSON.
        topics_path: Path to current execution's topic modeling results JSON.
        graph_path: Path to current execution's graph analysis results JSON.
        misinformation_path: Path to current execution's misinformation results JSON.
        skip: When True, stage is skipped without execution.
        canonical_output_path: Destination canonical path for consolidated output.
        config: Optional application configuration.
        execution_time: Optional fixed ISO UTC timestamp for deterministic execution.
        reddit_stage_entry: Stage entry dict for reddit_preprocessing.
        youtube_stage_entry: Stage entry dict for youtube_preprocessing.
        sentiment_stage_entry: Optional stage entry dict for sentiment.
        topic_stage_entry: Optional stage entry dict for topics.
        graph_stage_entry: Optional stage entry dict for graph.
        misinformation_stage_entry: Optional stage entry dict for misinformation.

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Consolidation stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream base source provenance validation (fail closed)
    if reddit_stage_entry is None or youtube_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Consolidation stage failed: missing upstream preprocessing provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream preprocessing provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(reddit_stage_entry, dict) or not isinstance(youtube_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Consolidation stage failed: invalid upstream preprocessing stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    r_status = reddit_stage_entry.get("status")
    y_status = youtube_stage_entry.get("status")

    if r_status not in PERMITTED_STAGE_STATUSES or y_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning(
            "Consolidation stage failed: invalid upstream preprocessing status (reddit='%s', youtube='%s').",
            r_status,
            y_status,
        )
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream preprocessing status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    if r_status == "skipped" and y_status == "skipped":
        logger.info("Both upstream preprocessing sources skipped. Skipping consolidation stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="all_sources_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if r_status != "success" and y_status != "success":
        logger.warning("Neither upstream preprocessing source succeeded. Skipping consolidation stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 2. Canonical path resolution
    resolved_config = config or load_config()
    if canonical_output_path is not None:
        resolved_canonical = Path(canonical_output_path)
    else:
        resolved_canonical = resolved_config.processed_data_path / "consolidated_intelligence.json"

    # 3. Sentinel paths for all 6 inputs to prevent reading stale canonical files
    # Base Reddit
    if r_status == "success" and reddit_processed_path and Path(reddit_processed_path).exists():
        r_in: Path = Path(reddit_processed_path)
    else:
        r_in = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"
        while r_in.exists():
            r_in = resolved_canonical.parent / f".sentinel_inactive_reddit_{uuid.uuid4().hex}.json"

    # Base YouTube
    if y_status == "success" and youtube_processed_path and Path(youtube_processed_path).exists():
        y_in: Path = Path(youtube_processed_path)
    else:
        y_in = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"
        while y_in.exists():
            y_in = resolved_canonical.parent / f".sentinel_inactive_youtube_{uuid.uuid4().hex}.json"

    # Analytical Sentiment
    s_valid = (
        sentiment_stage_entry is not None
        and isinstance(sentiment_stage_entry, dict)
        and sentiment_stage_entry.get("status") in {"success", "partial"}
        and sentiment_path is not None
        and Path(sentiment_path).exists()
    )
    if s_valid:
        s_in: Path = Path(sentiment_path)
    else:
        s_in = resolved_canonical.parent / f".sentinel_inactive_sentiment_{uuid.uuid4().hex}.json"
        while s_in.exists():
            s_in = resolved_canonical.parent / f".sentinel_inactive_sentiment_{uuid.uuid4().hex}.json"

    # Analytical Topics
    t_valid = (
        topic_stage_entry is not None
        and isinstance(topic_stage_entry, dict)
        and topic_stage_entry.get("status") in {"success", "partial"}
        and topics_path is not None
        and Path(topics_path).exists()
    )
    if t_valid:
        t_in: Path = Path(topics_path)
    else:
        t_in = resolved_canonical.parent / f".sentinel_inactive_topics_{uuid.uuid4().hex}.json"
        while t_in.exists():
            t_in = resolved_canonical.parent / f".sentinel_inactive_topics_{uuid.uuid4().hex}.json"

    # Analytical Graph
    g_valid = (
        graph_stage_entry is not None
        and isinstance(graph_stage_entry, dict)
        and graph_stage_entry.get("status") in {"success", "partial", "empty"}
        and graph_path is not None
        and Path(graph_path).exists()
    )
    if g_valid:
        g_in: Path = Path(graph_path)
    else:
        g_in = resolved_canonical.parent / f".sentinel_inactive_graph_{uuid.uuid4().hex}.json"
        while g_in.exists():
            g_in = resolved_canonical.parent / f".sentinel_inactive_graph_{uuid.uuid4().hex}.json"

    # Analytical Misinformation
    m_valid = (
        misinformation_stage_entry is not None
        and isinstance(misinformation_stage_entry, dict)
        and misinformation_stage_entry.get("status") in {"success", "partial", "empty"}
        and misinformation_path is not None
        and Path(misinformation_path).exists()
    )
    if m_valid:
        m_in: Path = Path(misinformation_path)
    else:
        m_in = resolved_canonical.parent / f".sentinel_inactive_misinformation_{uuid.uuid4().hex}.json"
        while m_in.exists():
            m_in = resolved_canonical.parent / f".sentinel_inactive_misinformation_{uuid.uuid4().hex}.json"

    # 4. Temporary file allocation, pipeline execution, and validation
    temp_output_path: Optional[Path] = None
    try:
        resolved_canonical.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = resolved_canonical.parent / f".tmp_consolidation_{uuid.uuid4().hex}.json"
        while temp_output_path.exists():
            temp_output_path = resolved_canonical.parent / f".tmp_consolidation_{uuid.uuid4().hex}.json"

        logger.info("Executing social intelligence consolidation pipeline.")
        payload = consolidate_social_intelligence(
            reddit_processed_path=r_in,
            youtube_processed_path=y_in,
            sentiment_path=s_in,
            topics_path=t_in,
            graph_path=g_in,
            misinformation_path=m_in,
            output_path=temp_output_path,
            execution_time=execution_time,
        )

        # Validate consolidated artifact with ConsolidatedArtifactValidator
        elapsed = time.perf_counter() - start_time
        try:
            ConsolidatedArtifactValidator.validate_consolidated_artifact(payload)
        except (ValueError, TypeError) as val_err:
            logger.warning("Consolidated artifact validation failed: %s", val_err)
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=str(val_err),
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="validation_failed",
                ),
                None,
            )

        try:
            os.replace(temp_output_path, resolved_canonical)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Failed to atomically promote consolidated output to canonical destination.")
            _safe_unlink_temp_file(temp_output_path)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason="Atomic output promotion failed",
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="promotion_error",
                ),
                None,
            )

        total_records = payload.get("total_records", 0)
        c_status = payload.get("consolidation_status")

        if total_records == 0:
            stage_status = "empty"
        elif c_status in {"complete", "success"}:
            stage_status = "success"
        else:
            stage_status = "partial"

        logger.info(
            "Consolidation stage succeeded with status '%s' and %d records. Canonical output: %s",
            stage_status,
            total_records,
            resolved_canonical.name,
        )
        return (
            create_initial_stage_entry(
                status=stage_status,
                reason=None,
                record_count=total_records,
                duration_seconds=elapsed,
                error_category=None,
            ),
            resolved_canonical,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during consolidation stage execution.")
        _safe_unlink_temp_file(temp_output_path)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Consolidation execution encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# Company Step 9 Elasticsearch Indexing Stage (Stage 11)
# =====================================================================


def execute_elasticsearch_indexing_stage(
    consolidated_path: Optional[Path | str] = None,
    skip: bool = False,
    consolidation_stage_entry: Optional[dict[str, Any]] = None,
    client: Optional[ElasticsearchClient] = None,
    index_name: Optional[str] = None,
    manifest_path: Optional[Path | str] = None,
    batch_size: int = 500,
    ensure_index: bool = True,
    refresh: Optional[Any] = False,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 9 Elasticsearch Indexing orchestration stage.

    Consumes validated Stage 10 consolidated intelligence JSON.
    Skips cleanly if indexing is skipped by user or if Stage 10 was skipped or failed.
    Handles empty valid datasets without error.
    Records indexed document counts, duration, and error classifications.

    Args:
        consolidated_path: Optional path to Stage 10 consolidated JSON.
        skip: When True, stage is skipped without execution.
        consolidation_stage_entry: Stage entry dict for consolidation (Stage 10).
        client: Optional injected ElasticsearchClient instance.
        index_name: Optional target Elasticsearch index name.
        manifest_path: Optional destination path for indexing manifest.
        batch_size: Bulk indexing batch size (default 500).
        ensure_index: Whether to verify or create index and mappings.
        refresh: Optional Elasticsearch refresh policy.

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Elasticsearch indexing stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 1. Upstream consolidation provenance validation
    if consolidation_stage_entry is None:
        elapsed = time.perf_counter() - start_time
        logger.warning("Elasticsearch indexing stage failed: missing upstream consolidation provenance.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Missing upstream consolidation provenance",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_upstream_provenance",
            ),
            None,
        )

    if not isinstance(consolidation_stage_entry, dict):
        elapsed = time.perf_counter() - start_time
        logger.warning("Elasticsearch indexing stage failed: invalid upstream consolidation stage entry format.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream consolidation stage entry format",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    c_status = consolidation_stage_entry.get("status")
    if c_status not in PERMITTED_STAGE_STATUSES:
        elapsed = time.perf_counter() - start_time
        logger.warning("Elasticsearch indexing stage failed: invalid upstream consolidation status '%s'.", c_status)
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Invalid upstream consolidation status",
                record_count=0,
                duration_seconds=elapsed,
                error_category="invalid_upstream_provenance",
            ),
            None,
        )

    if c_status == "skipped":
        logger.info("Stage 10 consolidation was skipped. Skipping Elasticsearch indexing stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_skipped",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    if c_status == "failed":
        logger.warning("Stage 10 consolidation failed. Skipping Elasticsearch indexing stage.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="prior_stage_failure",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    # 2. Target dataset path resolution
    if consolidated_path is not None:
        target_dataset = Path(consolidated_path)
    else:
        elapsed = time.perf_counter() - start_time
        logger.warning("Elasticsearch indexing stage failed: consolidated dataset path is missing.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Consolidated dataset path is missing or None",
                record_count=0,
                duration_seconds=elapsed,
                error_category="missing_artifact",
            ),
            None,
        )

    # If dataset is empty and file doesn't exist, handle as valid empty no-op
    if c_status == "empty" and (not target_dataset.exists() or target_dataset.stat().st_size == 0):
        elapsed = time.perf_counter() - start_time
        logger.info("Stage 10 dataset is empty and artifact is absent; indexing stage successful empty no-op.")
        return (
            create_initial_stage_entry(
                status="empty",
                reason="empty_consolidated_dataset",
                record_count=0,
                duration_seconds=elapsed,
                error_category=None,
            ),
            None,
        )

    if not target_dataset.exists():
        elapsed = time.perf_counter() - start_time
        logger.warning("Elasticsearch indexing stage failed: consolidated dataset not found at %s", target_dataset)
        return (
            create_initial_stage_entry(
                status="failed",
                reason=f"Consolidated dataset not found at {target_dataset}",
                record_count=0,
                duration_seconds=elapsed,
                error_category="artifact_not_found",
            ),
            None,
        )

    # 3. Execute Elasticsearch indexing pipeline
    try:
        logger.info("Executing Elasticsearch indexing pipeline.")
        exit_code, manifest = run_indexing_pipeline(
            dataset_path=target_dataset,
            client=client,
            index_name=index_name or DEFAULT_INDEX_NAME,
            refresh=refresh,
            ensure_index=ensure_index,
            batch_size=batch_size,
            manifest_path=manifest_path,
        )
        elapsed = time.perf_counter() - start_time
        m_status = manifest.get("status", "failed")

        if m_status == "success":
            stage_status = "success"
        elif m_status == "empty":
            stage_status = "empty"
        elif m_status == "partial_success":
            stage_status = "partial"
        else:
            stage_status = "failed"

        record_count = manifest.get("indexed_count", 0)
        duration = manifest.get("duration_seconds", round(elapsed, 4))
        err_cat = manifest.get("error_category") if stage_status == "failed" else None
        reason = manifest.get("error_message") if stage_status in {"failed", "partial"} else None

        logger.info(
            "Elasticsearch indexing stage completed with status '%s' (%d indexed).",
            stage_status,
            record_count,
        )
        return (
            create_initial_stage_entry(
                status=stage_status,
                reason=reason,
                record_count=record_count,
                duration_seconds=duration,
                error_category=err_cat,
            ),
            target_dataset,
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during Elasticsearch indexing stage execution.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Elasticsearch indexing encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


# =====================================================================
# Company Step 10 Kibana Dashboard Validation Stage (Stage 12)
# =====================================================================


def execute_kibana_validation_stage(
    bundle_path: Optional[Path | str] = None,
    skip: bool = False,
    indexing_stage_entry: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], Optional[Path]]:
    """Executes the Company Step 10 Kibana Dashboard Validation orchestration stage.

    Performs strictly offline validation of the Kibana saved-object bundle.
    Does NOT require a live Kibana or Elasticsearch instance.
    Validates Data View, Lens panels, Dashboard references, canonical mapping conformance,
    unindexed raw field exclusions, and methodology disclosures.

    Args:
        bundle_path: Optional path to the Kibana dashboards NDJSON bundle.
        skip: When True, stage is skipped without execution.
        indexing_stage_entry: Optional stage entry dict for indexing (Stage 11).

    Returns:
        tuple of (stage_entry: dict[str, Any], canonical_output_path: Optional[Path])
    """
    start_time = time.perf_counter()

    if skip:
        logger.info("Kibana validation stage explicitly skipped by user.")
        return (
            create_initial_stage_entry(
                status="skipped",
                reason="stage_skipped_by_user",
                record_count=0,
                duration_seconds=0.0,
                error_category=None,
            ),
            None,
        )

    target_bundle = Path(bundle_path) if bundle_path is not None else DEFAULT_BUNDLE_PATH

    try:
        logger.info("Executing offline Kibana dashboard bundle validation.")
        res = validate_kibana_dashboard_bundle(target_bundle)
        elapsed = time.perf_counter() - start_time

        if res.get("is_valid"):
            record_count = res.get("total_objects", 0)
            logger.info("Kibana dashboard bundle validation passed (%d objects verified).", record_count)
            entry = create_initial_stage_entry(
                status="success",
                reason=None,
                record_count=record_count,
                duration_seconds=elapsed,
                error_category=None,
            )
            entry["mode"] = "offline_bundle_validation"
            entry["scope"] = (
                "Offline saved-object bundle schema, reference, and mapping-conformance validation (41 objects)."
            )
            return (
                entry,
                target_bundle,
            )
        else:
            errors = res.get("errors", ["Bundle validation failed"])
            reason = "; ".join(str(e) for e in errors)
            logger.warning("Kibana dashboard bundle validation failed: %s", reason)
            return (
                create_initial_stage_entry(
                    status="failed",
                    reason=reason,
                    record_count=0,
                    duration_seconds=elapsed,
                    error_category="bundle_validation_error",
                ),
                None,
            )

    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.exception("Unexpected exception occurred during Kibana validation stage.")
        return (
            create_initial_stage_entry(
                status="failed",
                reason="Kibana validation encountered an unexpected error",
                record_count=0,
                duration_seconds=elapsed,
                error_category="execution_error",
            ),
            None,
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds and configures the CLI argument parser with mutually exclusive groups and options."""
    parser = argparse.ArgumentParser(
        prog="python app.py",
        description="Large Scale Social Media Intelligence Platform CLI.",
        add_help=True,
    )

    # Primary execution commands (mutually exclusive)
    action_group = parser.add_argument_group("Execution Commands")
    action_group.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Execute the unified 12-stage social media intelligence pipeline.",
    )
    action_group.add_argument(
        "--ingest-reddit",
        action="store_true",
        default=False,
        help="Ingest mock Reddit dataset into data/raw/.",
    )
    action_group.add_argument(
        "--ingest",
        dest="ingest_reddit",
        action="store_true",
        help="Alias for --ingest-reddit.",
    )
    action_group.add_argument(
        "--ingest-reddit-live",
        action="store_true",
        default=False,
        help="Fetch live Reddit posts via OAuth2 API.",
    )
    action_group.add_argument(
        "--ingest-youtube",
        action="store_true",
        default=False,
        help="Fetch live YouTube videos via Data API v3.",
    )
    action_group.add_argument(
        "--preprocess-reddit",
        action="store_true",
        default=False,
        help="Clean, handle emojis & detect language for Reddit.",
    )
    action_group.add_argument(
        "--preprocess-youtube",
        action="store_true",
        default=False,
        help="Clean, handle emojis & detect language for YouTube.",
    )
    action_group.add_argument(
        "--index-elasticsearch",
        action="store_true",
        default=False,
        help="Index consolidated intelligence dataset into Elasticsearch.",
    )
    action_group.add_argument(
        "--validate-kibana",
        action="store_true",
        default=False,
        help="Validate Kibana dashboard saved objects bundle offline.",
    )

    # Unified orchestration options (valid only with --run-all)
    orchestration_group = parser.add_argument_group("Unified Orchestration Options (--run-all)")
    orchestration_group.add_argument(
        "--skip-reddit",
        action="store_true",
        default=False,
        help="Skip Reddit ingestion and preprocessing during --run-all.",
    )
    orchestration_group.add_argument(
        "--skip-youtube",
        action="store_true",
        default=False,
        help="Skip YouTube ingestion and preprocessing during --run-all.",
    )
    orchestration_group.add_argument(
        "--skip-neo4j",
        action="store_true",
        default=False,
        help="Skip Neo4j Cypher batch persistence during --run-all.",
    )
    orchestration_group.add_argument(
        "--skip-kibana",
        action="store_true",
        default=False,
        help="Skip offline Kibana dashboard bundle validation during --run-all.",
    )
    orchestration_group.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Custom destination path for execution manifest (default: {DEFAULT_MANIFEST_PATH}).",
    )
    orchestration_group.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Custom destination directory for run artifacts (default: data/runs/<run_id>/).",
    )
    orchestration_group.add_argument(
        "--promote-canonical",
        action="store_true",
        default=False,
        help="Promote artifacts to canonical data/processed/ location on full pipeline success.",
    )

    return parser


def parse_cli_arguments(
    args: Optional[list[str]] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.Namespace:
    """Parses and validates command-line arguments.

    Enforces:
    1. --run-all cannot be combined with legacy primary execution commands.
    2. Multiple legacy primary execution commands cannot be combined.
    3. Skip flags (--skip-*) can only be used with --run-all.
    4. --manifest-path can only be used with --run-all.

    Args:
        args: Command-line arguments list (defaults to sys.argv[1:] if None).
        parser: Optional custom or mocked ArgumentParser instance.

    Returns:
        Validated argparse.Namespace object.

    Raises:
        SystemExit: If argument syntax is invalid or incompatible combinations are supplied.
    """
    p = parser or build_argument_parser()
    parsed = p.parse_args(args)

    # Detect all specified primary action flags
    legacy_actions: list[str] = []
    if parsed.ingest_reddit:
        legacy_actions.append("--ingest-reddit")
    if parsed.ingest_reddit_live:
        legacy_actions.append("--ingest-reddit-live")
    if parsed.ingest_youtube:
        legacy_actions.append("--ingest-youtube")
    if parsed.preprocess_reddit:
        legacy_actions.append("--preprocess-reddit")
    if parsed.preprocess_youtube:
        legacy_actions.append("--preprocess-youtube")
    if parsed.index_elasticsearch:
        legacy_actions.append("--index-elasticsearch")
    if parsed.validate_kibana:
        legacy_actions.append("--validate-kibana")

    # Rule 1: --run-all cannot be combined with legacy actions
    if parsed.run_all and legacy_actions:
        p.error(
            f"--run-all cannot be combined with legacy execution commands: {', '.join(legacy_actions)}"
        )

    # Rule 2: Multiple legacy actions cannot be combined
    if len(legacy_actions) > 1:
        p.error(
            f"Cannot combine multiple execution commands: {', '.join(legacy_actions)}"
        )

    # Rule 3: Skip flags require --run-all
    skip_flags: list[str] = []
    if parsed.skip_reddit:
        skip_flags.append("--skip-reddit")
    if parsed.skip_youtube:
        skip_flags.append("--skip-youtube")
    if parsed.skip_neo4j:
        skip_flags.append("--skip-neo4j")
    if parsed.skip_kibana:
        skip_flags.append("--skip-kibana")

    if skip_flags and not parsed.run_all:
        p.error(f"Skip flags ({', '.join(skip_flags)}) can only be used with --run-all.")

    # Rule 4: Custom manifest path requires --run-all
    raw_args = args if args is not None else sys.argv[1:]
    has_manifest_arg = any(
        a == "--manifest-path" or a.startswith("--manifest-path=") for a in raw_args
    )
    if has_manifest_arg and not parsed.run_all:
        p.error("--manifest-path can only be used with --run-all.")

    # Rule 5: --run-dir requires --run-all
    has_run_dir_arg = any(
        a == "--run-dir" or a.startswith("--run-dir=") for a in raw_args
    )
    if has_run_dir_arg and not parsed.run_all:
        p.error("--run-dir can only be used with --run-all.")

    # Rule 6: --promote-canonical requires --run-all
    if parsed.promote_canonical and not parsed.run_all:
        p.error("--promote-canonical can only be used with --run-all.")

    return parsed


def promote_run_to_canonical(
    run_dir: Path,
    processed_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Rollback-safe promotion of artifacts from a verified successful run directory to canonical locations.

    Enforces:
    1. NEVER promotes incomplete or failed runs (must be called only on overall success).
    2. Validates all source artifacts exist and are readable before beginning replacement.
    3. Preserves existing canonical files in a temporary backup staging directory.
    4. Stages destination files alongside target files prior to replacement.
    5. If any replacement fails, automatically rolls back all replaced canonical files
       from the backup staging directory, restoring the original canonical state.
    6. Leaves original run-directory artifacts untouched.
    """
    p_dir = Path(processed_dir) if processed_dir is not None else Path("data/processed")
    r_dir = Path(raw_dir) if raw_dir is not None else Path("data/raw")
    p_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "reddit_posts_raw.json": r_dir / "reddit_posts_raw.json",
        "youtube_videos_raw.json": r_dir / "youtube_videos_raw.json",
        "reddit_posts_processed.json": p_dir / "reddit_posts_processed.json",
        "youtube_videos_processed.json": p_dir / "youtube_videos_processed.json",
        "sentiment_analysis_results.json": p_dir / "sentiment_analysis_results.json",
        "topic_modeling_results.json": p_dir / "topic_modeling_results.json",
        "graph_analysis_results.json": p_dir / "graph_analysis_results.json",
        "misinformation_analysis_results.json": p_dir / "misinformation_analysis_results.json",
        "consolidated_intelligence.json": p_dir / "consolidated_intelligence.json",
        "elasticsearch_indexing_manifest.json": p_dir / "elasticsearch_indexing_manifest.json",
        "pipeline_execution_manifest.json": p_dir / "pipeline_execution_manifest.json",
    }

    # Step 1: Validate source files
    source_pairs: list[tuple[str, Path, Path]] = []
    for filename, target_path in mapping.items():
        src = run_dir / filename
        if src.exists():
            if not src.is_file() or src.stat().st_size == 0:
                raise CanonicalPromotionError(
                    f"Source artifact {src} is empty or not a valid regular file."
                )
            source_pairs.append((filename, src, target_path))

    if not source_pairs:
        logger.warning("No artifacts found in %s to promote to canonical storage.", run_dir)
        return {}

    # Step 2: Prepare backup directory and stage backups of existing canonical files
    backup_dir = p_dir / f".backup_promote_{uuid.uuid4().hex[:8]}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up_canonical: dict[Path, Path] = {}
    staged_temp_files: list[tuple[Path, Path, str]] = []
    replaced_targets: list[Path] = []

    try:
        # Back up existing canonical files
        for filename, _, target_path in source_pairs:
            if target_path.exists():
                backup_dest = backup_dir / f"backup_{filename}"
                shutil.copy2(target_path, backup_dest)
                backed_up_canonical[target_path] = backup_dest

        # Step 3: Stage all source files into temporary target files alongside destinations
        for filename, src, target_path in source_pairs:
            temp_target = target_path.parent / f".tmp_stage_{uuid.uuid4().hex[:8]}_{filename}"
            shutil.copy2(src, temp_target)
            staged_temp_files.append((temp_target, target_path, filename))

        # Step 4: Atomic replacement with rollback tracking
        promoted: dict[str, str] = {}
        for temp_target, target_path, filename in staged_temp_files:
            os.replace(temp_target, target_path)
            replaced_targets.append(target_path)
            promoted[filename] = str(target_path)

        # Step 5: Clean up backup directory on full success
        shutil.rmtree(backup_dir, ignore_errors=True)
        logger.info(
            "Successfully promoted %d artifacts from %s to canonical directories with rollback safety.",
            len(promoted),
            run_dir,
        )
        return promoted

    except Exception as exc:
        logger.error(
            "Canonical promotion encountered error (%s). Initiating rollback of %d files.",
            exc,
            len(replaced_targets),
        )
        # Roll back all replaced files
        for target_path in replaced_targets:
            try:
                if target_path in backed_up_canonical:
                    # Restore previous canonical version
                    shutil.copy2(backed_up_canonical[target_path], target_path)
                else:
                    # File was newly created during this run, remove it
                    target_path.unlink(missing_ok=True)
            except Exception as rb_exc:
                logger.critical("Failed to rollback file %s: %s", target_path, rb_exc)

        # Remove any lingering staged temporary files
        for temp_target, _, _ in staged_temp_files:
            try:
                temp_target.unlink(missing_ok=True)
            except Exception:
                pass

        # Clean up backup directory
        shutil.rmtree(backup_dir, ignore_errors=True)
        raise CanonicalPromotionError(
            f"Failed to promote artifacts to canonical storage (rolled back cleanly): {exc}"
        ) from exc


def run_orchestrated_pipeline(
    argv: Optional[list[str]] = None,
    config: Optional[AppConfig] = None,
    run_id: Optional[str] = None,
    base_runs_dir: Optional[Path | str] = None,
    lock_path: Optional[Path | str] = None,
    skip_lock: bool = False,
    promote_canonical: Optional[bool] = None,
) -> tuple[int, Optional[dict[str, Any]]]:
    """Unified 12-stage social media intelligence pipeline orchestrator.

    Executes all 12 pipeline stages in canonical order with full run isolation,
    cross-process locking, explicit artifact tracking, and manifest persistence.

    Lifecycle Management:
    - Cross-Process Lock: Exclusive file lock acquired before execution and reliably released in finally:.
    - Run Identity: Unique, validated run_id generated per execution.
    - Run Directory Isolation: All intermediate stage artifacts are written strictly into data/runs/<run_id>/.
    - Resource Ownership: Neo4j and Elasticsearch runners construct and close their own resources internally.
    - False-Success Prevention: Non-zero exit code (1) inferred if any stage failed or produced partial results.
    - Final Promotion: Intermediate files never overwrite canonical files; promotion occurs only on verified success.

    Args:
        argv: Command-line arguments list (defaults to sys.argv[1:] if None).
        config: Optional loaded AppConfig instance.
        run_id: Optional explicit run_id (must match RUN_ID_PATTERN if provided).
        base_runs_dir: Optional custom base runs directory (defaults to data/runs).
        lock_path: Optional custom lock file path (defaults to data/.orchestration.lock).
        skip_lock: When True, bypasses lock acquisition (for isolated testing).
        promote_canonical: Whether to promote artifacts to data/processed/ on overall success.

    Returns:
        tuple of (exit_code: int, manifest: Optional[dict[str, Any]])
    """
    args = parse_cli_arguments(argv)
    if not args.run_all:
        raise ValueError("run_orchestrated_pipeline requires --run-all.")

    # 1. Run ID generation and validation
    effective_run_id = run_id or generate_run_id()
    if not validate_run_id(effective_run_id):
        raise ValueError(
            f"Invalid run_id {effective_run_id!r}. Must match pattern {RUN_ID_REGEX}."
        )

    # 2. Base runs directory and isolated run directory resolution
    custom_run_dir = getattr(args, "run_dir", None)
    effective_base = Path(base_runs_dir) if base_runs_dir is not None else (custom_run_dir or DEFAULT_RUNS_DIR)
    run_dir = effective_base / effective_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    do_promote = promote_canonical if promote_canonical is not None else getattr(args, "promote_canonical", False)

    # 3. Lock acquisition & pipeline execution
    effective_lock_path = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    lock = OrchestrationLock(lock_path=effective_lock_path)
    stages: dict[str, dict[str, Any]] = {}
    resolved_config = config or load_config()

    try:
        if not skip_lock:
            lock.acquire(effective_run_id)

        logger.info("Starting unified 12-stage orchestration run %s in %s", effective_run_id, run_dir)

        # Stage 1: reddit_ingestion
        reddit_raw_target = run_dir / "reddit_posts_raw.json"
        stage1_entry, reddit_raw_path = execute_reddit_ingestion_stage(
            skip=args.skip_reddit,
            canonical_output_path=reddit_raw_target,
            config=resolved_config,
        )
        stages["reddit_ingestion"] = stage1_entry

        # Stage 2: youtube_ingestion
        youtube_raw_target = run_dir / "youtube_videos_raw.json"
        stage2_entry, youtube_raw_path = execute_youtube_ingestion_stage(
            skip=args.skip_youtube,
            canonical_output_path=youtube_raw_target,
            config=resolved_config,
        )
        stages["youtube_ingestion"] = stage2_entry

        # Stage 3: reddit_preprocessing
        reddit_proc_target = run_dir / "reddit_posts_processed.json"
        skip_r_proc = args.skip_reddit or (stage1_entry.get("status") != "success")
        stage3_entry, reddit_proc_path = execute_reddit_preprocessing_stage(
            raw_input_path=reddit_raw_path,
            skip=skip_r_proc,
            canonical_output_path=reddit_proc_target,
            config=resolved_config,
            ingestion_stage_entry=stage1_entry,
        )
        stages["reddit_preprocessing"] = stage3_entry

        # Stage 4: youtube_preprocessing
        youtube_proc_target = run_dir / "youtube_videos_processed.json"
        skip_y_proc = args.skip_youtube or (stage2_entry.get("status") != "success")
        stage4_entry, youtube_proc_path = execute_youtube_preprocessing_stage(
            raw_input_path=youtube_raw_path,
            skip=skip_y_proc,
            canonical_output_path=youtube_proc_target,
            config=resolved_config,
            ingestion_stage_entry=stage2_entry,
        )
        stages["youtube_preprocessing"] = stage4_entry

        # Stage 5: sentiment
        sentiment_target = run_dir / "sentiment_analysis_results.json"
        stage5_entry, sentiment_path = execute_sentiment_stage(
            reddit_processed_path=reddit_proc_path,
            youtube_processed_path=youtube_proc_path,
            skip=False,
            canonical_output_path=sentiment_target,
            config=resolved_config,
            reddit_stage_entry=stage3_entry,
            youtube_stage_entry=stage4_entry,
        )
        if stage5_entry.get("status") == "success" and (skip_r_proc or skip_y_proc):
            stage5_entry["status"] = "partial"
            stage5_entry["reason"] = "single_platform_execution"
        stages["sentiment"] = stage5_entry

        # Stage 6: topics
        topics_target = run_dir / "topic_modeling_results.json"
        stage6_entry, topics_path = execute_topic_stage(
            reddit_processed_path=reddit_proc_path,
            youtube_processed_path=youtube_proc_path,
            skip=False,
            canonical_output_path=topics_target,
            config=resolved_config,
            reddit_stage_entry=stage3_entry,
            youtube_stage_entry=stage4_entry,
        )
        if stage6_entry.get("status") == "success" and (skip_r_proc or skip_y_proc):
            stage6_entry["status"] = "partial"
            stage6_entry["reason"] = "single_platform_execution"
        stages["topics"] = stage6_entry

        # Stage 7: graph
        graph_target = run_dir / "graph_analysis_results.json"
        stage7_entry, graph_path = execute_graph_stage(
            reddit_processed_path=reddit_proc_path,
            youtube_processed_path=youtube_proc_path,
            skip=False,
            canonical_output_path=graph_target,
            config=resolved_config,
            reddit_stage_entry=stage3_entry,
            youtube_stage_entry=stage4_entry,
        )
        if stage7_entry.get("status") == "success" and (skip_r_proc or skip_y_proc):
            stage7_entry["status"] = "partial"
            stage7_entry["reason"] = "single_platform_execution"
        stages["graph"] = stage7_entry

        # Stage 8: neo4j_persistence
        stage8_entry, _ = execute_neo4j_stage(
            graph_artifact_path=graph_path,
            skip=args.skip_neo4j,
            graph_stage_entry=stage7_entry,
            manager=None,
            manager_factory=None,
        )
        stages["neo4j_persistence"] = stage8_entry

        # Stage 9: misinformation
        misinfo_target = run_dir / "misinformation_analysis_results.json"
        stage9_entry, misinfo_path = execute_misinformation_stage(
            reddit_processed_path=reddit_proc_path,
            youtube_processed_path=youtube_proc_path,
            skip=False,
            canonical_output_path=misinfo_target,
            config=resolved_config,
            reddit_stage_entry=stage3_entry,
            youtube_stage_entry=stage4_entry,
        )
        if stage9_entry.get("status") == "success" and (skip_r_proc or skip_y_proc):
            stage9_entry["status"] = "partial"
            stage9_entry["reason"] = "single_platform_execution"
        stages["misinformation"] = stage9_entry

        # Stage 10: consolidation
        consolidation_target = run_dir / "consolidated_intelligence.json"
        stage10_entry, consolidation_path = execute_consolidation_stage(
            reddit_processed_path=reddit_proc_path,
            youtube_processed_path=youtube_proc_path,
            sentiment_path=sentiment_path,
            topics_path=topics_path,
            graph_path=graph_path,
            misinformation_path=misinfo_path,
            skip=False,
            canonical_output_path=consolidation_target,
            config=resolved_config,
            reddit_stage_entry=stage3_entry,
            youtube_stage_entry=stage4_entry,
            sentiment_stage_entry=stage5_entry,
            topic_stage_entry=stage6_entry,
            graph_stage_entry=stage7_entry,
            misinformation_stage_entry=stage9_entry,
        )
        if stage10_entry.get("status") == "success" and (skip_r_proc or skip_y_proc):
            stage10_entry["status"] = "partial"
            stage10_entry["reason"] = "single_platform_execution"
        stages["consolidation"] = stage10_entry

        # Stage 11: elasticsearch_indexing
        es_manifest_target = run_dir / "elasticsearch_indexing_manifest.json"
        stage11_entry, _ = execute_elasticsearch_indexing_stage(
            consolidated_path=consolidation_path,
            skip=False,
            consolidation_stage_entry=stage10_entry,
            client=None,
            manifest_path=es_manifest_target,
        )
        stages["elasticsearch_indexing"] = stage11_entry

        # Stage 12: kibana_validation
        stage12_entry, _ = execute_kibana_validation_stage(
            bundle_path=None,
            skip=args.skip_kibana,
            indexing_stage_entry=stage11_entry,
        )
        stage12_entry["mode"] = "offline_bundle_validation"
        stage12_entry["scope"] = (
            "Offline saved-object bundle schema, reference, and mapping-conformance validation (41 objects)."
        )
        stages["kibana_validation"] = stage12_entry

        # 4. Manifest assembly and atomic persistence
        manifest = build_execution_manifest(
            stages=stages,
            pipeline_mode="orchestrated",
            run_id=effective_run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        manifest["run_directory"] = str(run_dir)
        manifest["artifacts"] = {
            "reddit_raw": str(reddit_raw_path) if reddit_raw_path else None,
            "youtube_raw": str(youtube_raw_path) if youtube_raw_path else None,
            "reddit_processed": str(reddit_proc_path) if reddit_proc_path else None,
            "youtube_processed": str(youtube_proc_path) if youtube_proc_path else None,
            "sentiment": str(sentiment_path) if sentiment_path else None,
            "topics": str(topics_path) if topics_path else None,
            "graph": str(graph_path) if graph_path else None,
            "misinformation": str(misinfo_path) if misinfo_path else None,
            "consolidation": str(consolidation_path) if consolidation_path else None,
            "elasticsearch_manifest": str(es_manifest_target) if es_manifest_target.exists() else None,
        }

        # Persist manifest to run directory
        run_manifest_file = run_dir / "pipeline_execution_manifest.json"
        persist_manifest(manifest, destination_path=run_manifest_file, raise_on_error=True)

        # Persist to requested target manifest path if different
        target_manifest_path = Path(args.manifest_path) if args.manifest_path else DEFAULT_MANIFEST_PATH
        if target_manifest_path.resolve() != run_manifest_file.resolve():
            persist_manifest(manifest, destination_path=target_manifest_path, raise_on_error=True)

        # 5. Canonical promotion on verified success
        if do_promote and manifest.get("overall_outcome") == "success":
            promote_run_to_canonical(run_dir)

        # 6. Console reporting
        print("\n[UNIFIED SOCIAL INTELLIGENCE PIPELINE ORCHESTRATION]")
        print(f"Run ID        : {effective_run_id}")
        print(f"Run Directory : {run_dir}")
        print(f"Manifest Path : {target_manifest_path}")
        print(f"Skip Reddit   : {args.skip_reddit}")
        print(f"Skip YouTube  : {args.skip_youtube}")
        print(f"Skip Neo4j    : {args.skip_neo4j}")
        print(f"Skip Kibana   : {args.skip_kibana}")
        print("-" * 68)
        for st_name in CANONICAL_STAGES:
            st_data = stages[st_name]
            st_status = st_data.get("status", "unknown").upper()
            st_cnt = st_data.get("record_count", 0)
            st_dur = st_data.get("duration_seconds", 0.0)
            print(f"  Stage: {st_name:<26} [{st_status:<7}] ({st_cnt:>4} records, {st_dur:>6.2f}s)")
        print("-" * 68)
        print(f"Overall Outcome : {manifest['overall_outcome'].upper()}")
        print(f"Exit Code       : {manifest['exit_code']}")
        print(f"Attempted/Total : {manifest['summary']['total_stages_attempted']} / {manifest['summary']['total_stages_evaluated']}")
        print("=" * 68)

        return manifest["exit_code"], manifest

    except CanonicalPromotionError as cp_err:
        logger.error("Canonical artifact promotion failed and was rolled back cleanly: %s", cp_err)
        return EXIT_STAGE_FAILURE, manifest
    except ManifestPersistenceError as p_err:
        logger.error("Failed to persist execution manifest: %s", p_err)
        return EXIT_MANIFEST_PERSISTENCE_FAILURE, None
    except OrchestrationLockError as l_err:
        logger.error("Orchestration lock conflict: %s", l_err)
        print(f"\n[ORCHESTRATION LOCK CONFLICT]\nError: {l_err}\n")
        return EXIT_LOCK_ACQUISITION_FAILURE, None
    except Exception as exc:
        logger.exception("Fatal unhandled exception during pipeline orchestration: %s", exc)
        return EXIT_STAGE_FAILURE, None
    finally:
        if not skip_lock:
            lock.release()


def main(argv: Optional[list[str]] = None) -> None:
    """Main execution entry point."""
    print_header()

    if argv is not None:
        args_list = argv
    elif len(sys.argv) > 1 and not sys.argv[0].lower().endswith("app.py"):
        # Test runner or external harness invoked main() without setting sys.argv to app.py
        args_list = []
    else:
        args_list = sys.argv[1:]

    parsed = parse_cli_arguments(args_list)

    if parsed.index_elasticsearch:
        exit_code = run_index_elasticsearch_cli()
        if exit_code != EXIT_SUCCESS:
            sys.exit(exit_code)
        return

    if parsed.validate_kibana:
        exit_code = run_validate_kibana_cli()
        if exit_code != EXIT_SUCCESS:
            sys.exit(exit_code)
        return

    if parsed.preprocess_reddit:
        run_preprocess_reddit_cli()
        return

    if parsed.preprocess_youtube:
        run_preprocess_youtube_cli()
        return

    if parsed.ingest_reddit_live:
        run_live_reddit_cli()
        return

    if parsed.ingest_youtube:
        run_youtube_cli()
        return

    if parsed.ingest_reddit:
        run_mock_reddit_cli()
        return

    if parsed.run_all:
        exit_code, _ = run_orchestrated_pipeline(args_list)
        if exit_code != EXIT_SUCCESS:
            sys.exit(exit_code)
        return

    print_project_overview()
    print_system_status()


if __name__ == "__main__":
    main()

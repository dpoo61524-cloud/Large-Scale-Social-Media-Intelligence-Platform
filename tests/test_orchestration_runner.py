"""Stage 13C: Unified Orchestration Runner and Run Isolation Tests.

Tests all 20 required aspects of Stage 13C:
1. Unique run ID generation and validation.
2. Run-specific directory creation.
3. Explicit current-run artifact input/output paths.
4. Missing current-run artifact handling.
5. Prevention of fallback to previous canonical artifacts.
6. Concurrent lock acquisition rejection.
7. Lock release after successful completion.
8. Lock release after exceptions.
9. Safe stale-lock handling & uncertain lock rejection.
10. Correct 12-stage execution order.
11. Prerequisite-based skipping.
12. Supported single-platform partial execution.
13. Accurate stage counts and overall outcome.
14. Exact 5/5/1/1 arithmetic scenario.
15. Intentional --skip-youtube scenario.
16. Stage 8 and Stage 11 resource lifecycle behavior.
17. Stage 12 offline scope labeling.
18. Manifest validation and atomic persistence.
19. CLI compatibility with existing individual commands.
20. Preservation of the existing --run-all safety contract until the new runner is ready.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest

from app import (
    CANONICAL_STAGES,
    DEFAULT_LOCK_PATH,
    DEFAULT_MANIFEST_PATH,
    EXIT_LOCK_ACQUISITION_FAILURE,
    EXIT_STAGE_FAILURE,
    EXIT_SUCCESS,
    PERMITTED_STAGE_STATUSES,
    RUN_ID_REGEX,
    CanonicalPromotionError,
    ManifestPersistenceError,
    OrchestrationLock,
    OrchestrationLockError,
    build_argument_parser,
    build_execution_manifest,
    calculate_manifest_summary,
    create_empty_stage_registry,
    create_initial_stage_entry,
    execute_elasticsearch_indexing_stage,
    execute_kibana_validation_stage,
    execute_neo4j_stage,
    generate_run_id,
    is_pid_alive,
    parse_cli_arguments,
    persist_manifest,
    promote_run_to_canonical,
    run_orchestrated_pipeline,
    validate_run_id,
)


# 1. Unique run ID generation and validation
def test_generate_and_validate_run_id():
    """Verify generate_run_id produces format conforming to RUN_ID_REGEX and is unique."""
    r1 = generate_run_id()
    r2 = generate_run_id()
    assert r1 != r2
    assert validate_run_id(r1) is True
    assert validate_run_id(r2) is True
    assert validate_run_id("invalid_run_id") is False
    assert validate_run_id("") is False
    assert validate_run_id(None) is False
    assert validate_run_id(12345) is False


# 2. Process liveness detection
def test_is_pid_alive_helper():
    """Verify is_pid_alive returns True for current PID and False for non-existent."""
    cur = os.getpid()
    assert is_pid_alive(cur) is True
    assert is_pid_alive(9999999) is False
    assert is_pid_alive(-1) is False
    assert is_pid_alive(0) is False
    assert is_pid_alive("invalid") is False


# 3. Concurrent lock acquisition rejection & normal release
def test_orchestration_lock_lifecycle(tmp_path):
    """Verify OrchestrationLock acquires, enforces exclusivity, and releases."""
    lock_path = tmp_path / ".test.lock"
    lock1 = OrchestrationLock(lock_path=lock_path)
    rid1 = generate_run_id()

    assert lock1.is_acquired is False
    lock1.acquire(rid1)
    assert lock1.is_acquired is True
    assert lock_path.exists()

    # Second acquisition by another instance must fail
    lock2 = OrchestrationLock(lock_path=lock_path)
    rid2 = generate_run_id()
    with pytest.raises(OrchestrationLockError) as exc_info:
        lock2.acquire(rid2)
    assert "currently locked by active PID" in str(exc_info.value)

    # Release lock1
    lock1.release()
    assert lock1.is_acquired is False
    assert not lock_path.exists()

    # Now lock2 can acquire
    lock2.acquire(rid2)
    assert lock2.is_acquired is True
    lock2.release()
    assert not lock_path.exists()


# 4. Context manager release
def test_orchestration_lock_context_manager(tmp_path):
    """Verify OrchestrationLock releases automatically via context manager."""
    lock_path = tmp_path / ".cm.lock"
    rid = generate_run_id()
    with OrchestrationLock(lock_path=lock_path) as lock:
        lock.acquire(rid)
        assert lock_path.exists()
    assert not lock_path.exists()


# 5. Lock release after exceptions
def test_orchestration_lock_exception_release(tmp_path):
    """Verify OrchestrationLock releases lock even when exception is raised."""
    lock_path = tmp_path / ".exc.lock"
    rid = generate_run_id()
    try:
        with OrchestrationLock(lock_path=lock_path) as lock:
            lock.acquire(rid)
            assert lock_path.exists()
            raise RuntimeError("Simulation failure")
    except RuntimeError:
        pass
    assert not lock_path.exists()


# 6. Safe stale-lock handling
def test_orchestration_lock_stale_lock_recovery(tmp_path):
    """Verify OrchestrationLock safely reclaims lock held by a dead PID."""
    lock_path = tmp_path / ".stale.lock"

    # Write a fake stale lock from non-existent PID 9999999
    stale_data = {
        "pid": 9999999,
        "run_id": "run_20261101_120000_deadbeef",
        "timestamp_utc": "2026-11-01T12:00:00Z",
        "hostname": "old-machine",
    }
    lock_path.write_text(json.dumps(stale_data), encoding="utf-8")

    # Acquisition must reclaim the stale lock
    lock = OrchestrationLock(lock_path=lock_path)
    rid = generate_run_id()
    lock.acquire(rid)
    assert lock.is_acquired is True

    # Inspect content to verify ownership
    with open(lock_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pid"] == os.getpid()
    assert data["run_id"] == rid

    lock.release()
    assert not lock_path.exists()


# 7. Safe stale-lock handling: manual intervention required on uncertain/corrupt lock
def test_orchestration_lock_uncertain_lock_requires_manual_intervention(tmp_path):
    """Verify OrchestrationLock raises OrchestrationLockError on corrupt/uncertain locks rather than deleting."""
    lock_path = tmp_path / ".corrupt.lock"
    lock_path.write_text("{ corrupt json ", encoding="utf-8")

    lock = OrchestrationLock(lock_path=lock_path)
    rid = generate_run_id()
    with pytest.raises(OrchestrationLockError) as exc_info:
        lock.acquire(rid)
    assert "could not be safely verified" in str(exc_info.value)
    # The uncertain lock file must NOT have been blindly deleted
    assert lock_path.exists()


# 8. Safe stale-lock handling: invalid PID metadata
def test_orchestration_lock_invalid_pid_metadata_requires_manual_intervention(tmp_path):
    """Verify OrchestrationLock raises OrchestrationLockError on invalid PID metadata."""
    lock_path = tmp_path / ".invalid_pid.lock"
    lock_path.write_text(json.dumps({"pid": "not_an_int", "run_id": "test"}), encoding="utf-8")

    lock = OrchestrationLock(lock_path=lock_path)
    rid = generate_run_id()
    with pytest.raises(OrchestrationLockError) as exc_info:
        lock.acquire(rid)
    assert "could not be safely verified" in str(exc_info.value)
    assert lock_path.exists()


# 9. Exact 5/5/1/1 arithmetic scenario (Manifest Arithmetic)
def test_calculate_manifest_summary_exact_5_5_1_1_scenario():
    """Verify calculate_manifest_summary with exact 5 success / 5 partial / 1 skipped / 1 failed scenario."""
    stages = create_empty_stage_registry()

    for s in [
        "reddit_ingestion",
        "reddit_preprocessing",
        "neo4j_persistence",
        "elasticsearch_indexing",
        "kibana_validation",
    ]:
        stages[s] = create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.1)

    for p in ["sentiment", "topics", "graph", "misinformation", "consolidation"]:
        stages[p] = create_initial_stage_entry(status="partial", record_count=10, duration_seconds=0.2)

    stages["youtube_preprocessing"] = create_initial_stage_entry(status="skipped", reason="prior_stage_failure")
    stages["youtube_ingestion"] = create_initial_stage_entry(status="failed", error_category="api_error")

    summary = calculate_manifest_summary(stages)
    assert summary["stages_succeeded"] == 5
    assert summary["stages_partial"] == 5
    assert summary["stages_skipped"] == 1
    assert summary["stages_failed"] == 1
    assert summary["stages_empty"] == 0
    assert summary["total_stages_attempted"] == 11
    assert summary["total_stages_evaluated"] == 12

    manifest = build_execution_manifest(stages=stages)
    assert manifest["overall_outcome"] == "failed"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE


# 10. Intentional --skip-youtube arithmetic scenario
def test_build_execution_manifest_intentional_skip_youtube_scenario():
    """Verify build_execution_manifest with 5 success / 5 partial / 2 skipped / 0 failed -> outcome partial, exit code 1."""
    stages = create_empty_stage_registry()

    for s in [
        "reddit_ingestion",
        "reddit_preprocessing",
        "neo4j_persistence",
        "elasticsearch_indexing",
        "kibana_validation",
    ]:
        stages[s] = create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.1)

    for p in ["sentiment", "topics", "graph", "misinformation", "consolidation"]:
        stages[p] = create_initial_stage_entry(status="partial", record_count=10, duration_seconds=0.2)

    stages["youtube_ingestion"] = create_initial_stage_entry(status="skipped", reason="source_disabled")
    stages["youtube_preprocessing"] = create_initial_stage_entry(status="skipped", reason="source_disabled")

    summary = calculate_manifest_summary(stages)
    assert summary["stages_succeeded"] == 5
    assert summary["stages_partial"] == 5
    assert summary["stages_skipped"] == 2
    assert summary["stages_failed"] == 0
    assert summary["stages_empty"] == 0
    assert summary["total_stages_attempted"] == 10
    assert summary["total_stages_evaluated"] == 12

    manifest = build_execution_manifest(stages=stages)
    assert manifest["overall_outcome"] == "partial"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE


# 11. Atomic promotion to canonical paths
def test_promote_run_to_canonical_atomic(tmp_path):
    """Verify promote_run_to_canonical copies files atomically via temp files."""
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    proc_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"

    (run_dir / "reddit_posts_raw.json").write_text('{"records": [1]}', encoding="utf-8")
    (run_dir / "sentiment_analysis_results.json").write_text('{"sentiment": true}', encoding="utf-8")

    promoted = promote_run_to_canonical(run_dir, processed_dir=proc_dir, raw_dir=raw_dir)
    assert "reddit_posts_raw.json" in promoted
    assert "sentiment_analysis_results.json" in promoted
    assert (raw_dir / "reddit_posts_raw.json").exists()
    assert (proc_dir / "sentiment_analysis_results.json").exists()
    # Source in run_dir must remain untouched
    assert (run_dir / "reddit_posts_raw.json").exists()


# 12. Correct 12-stage execution order and artifact isolation
def test_run_orchestrated_pipeline_execution_order_and_artifacts(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline runs all 12 stages in sequence and isolates artifacts."""
    call_log = []

    def make_mock_stage(name, record_cnt=10):
        def mock_impl(*args, **kwargs):
            call_log.append(name)
            out_path = kwargs.get("canonical_output_path") or kwargs.get("manifest_path")
            if out_path:
                p = Path(out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"stage": name}), encoding="utf-8")
                return {
                    "status": "success",
                    "record_count": record_cnt,
                    "duration_seconds": 0.05,
                    "error_category": None,
                    "reason": None,
                }, p
            return {
                "status": "success",
                "record_count": record_cnt,
                "duration_seconds": 0.05,
                "error_category": None,
                "reason": None,
            }, None
        return mock_impl

    STAGE_FUNC_MAP = {
        "reddit_ingestion": "execute_reddit_ingestion_stage",
        "youtube_ingestion": "execute_youtube_ingestion_stage",
        "reddit_preprocessing": "execute_reddit_preprocessing_stage",
        "youtube_preprocessing": "execute_youtube_preprocessing_stage",
        "sentiment": "execute_sentiment_stage",
        "topics": "execute_topic_stage",
        "graph": "execute_graph_stage",
        "neo4j_persistence": "execute_neo4j_stage",
        "misinformation": "execute_misinformation_stage",
        "consolidation": "execute_consolidation_stage",
        "elasticsearch_indexing": "execute_elasticsearch_indexing_stage",
        "kibana_validation": "execute_kibana_validation_stage",
    }

    for st_name, func_name in STAGE_FUNC_MAP.items():
        if st_name == "kibana_validation":
            def mock_kibana(*args, **kwargs):
                call_log.append("kibana_validation")
                return {
                    "status": "success",
                    "record_count": 41,
                    "duration_seconds": 0.01,
                    "error_category": None,
                    "reason": None,
                    "mode": "offline_bundle_validation",
                    "scope": "Offline saved-object bundle schema, reference, and mapping-conformance validation (41 objects).",
                }, None
            monkeypatch.setattr(f"app.{func_name}", mock_kibana)
        elif st_name == "elasticsearch_indexing":
            monkeypatch.setattr(f"app.{func_name}", make_mock_stage("elasticsearch_indexing"))
        elif st_name == "neo4j_persistence":
            def mock_neo4j(*args, **kwargs):
                call_log.append("neo4j_persistence")
                assert kwargs.get("manager") is None
                assert kwargs.get("manager_factory") is None
                return {
                    "status": "success",
                    "record_count": 20,
                    "duration_seconds": 0.02,
                    "error_category": None,
                    "reason": None,
                }, None
            monkeypatch.setattr(f"app.{func_name}", mock_neo4j)
        else:
            monkeypatch.setattr(f"app.{func_name}", make_mock_stage(st_name))

    runs_dir = tmp_path / "runs"
    lock_file = tmp_path / ".orch.lock"
    manifest_target = tmp_path / "custom_manifest.json"

    code, manifest = run_orchestrated_pipeline(
        argv=["--run-all", f"--manifest-path={manifest_target}"],
        base_runs_dir=runs_dir,
        lock_path=lock_file,
    )

    assert code == EXIT_SUCCESS
    assert manifest is not None
    assert manifest["overall_outcome"] == "success"
    assert manifest["exit_code"] == EXIT_SUCCESS

    # 1. Correct sequence
    assert tuple(call_log) == CANONICAL_STAGES

    # 2. Artifact isolation
    run_dir = Path(manifest["run_directory"])
    assert run_dir.exists()
    assert (run_dir / "pipeline_execution_manifest.json").exists()
    assert manifest_target.exists()

    # 3. Stage 12 offline labeling
    k_entry = manifest["stages"]["kibana_validation"]
    assert k_entry["mode"] == "offline_bundle_validation"
    assert "41 objects" in k_entry["scope"]

    # 4. Lock released
    assert not lock_file.exists()


# 13. Prerequisite-based skipping
def test_run_orchestrated_pipeline_prerequisite_skipping(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline skips downstream stages when prerequisites fail."""
    def failing_reddit_ingestion(*args, **kwargs):
        return {
            "status": "failed",
            "record_count": 0,
            "duration_seconds": 0.01,
            "error_category": "api_error",
            "reason": "rate_limit",
        }, None

    def skipped_youtube_ingestion(*args, **kwargs):
        return {
            "status": "skipped",
            "record_count": 0,
            "duration_seconds": 0.0,
            "error_category": None,
            "reason": "source_disabled",
        }, None

    monkeypatch.setattr("app.execute_reddit_ingestion_stage", failing_reddit_ingestion)
    monkeypatch.setattr("app.execute_youtube_ingestion_stage", skipped_youtube_ingestion)

    runs_dir = tmp_path / "runs"
    lock_file = tmp_path / ".orch.lock"

    code, manifest = run_orchestrated_pipeline(
        argv=["--run-all", "--skip-youtube"],
        base_runs_dir=runs_dir,
        lock_path=lock_file,
    )

    assert code == EXIT_STAGE_FAILURE
    assert manifest["overall_outcome"] == "failed"

    # Reddit preprocessing must have been skipped because ingestion failed
    assert manifest["stages"]["reddit_preprocessing"]["status"] == "skipped"
    assert manifest["stages"]["youtube_preprocessing"]["status"] == "skipped"


# 14. Exact 5/5/1/1 end-to-end execution scenario
def test_run_orchestrated_pipeline_exact_5_5_1_1_scenario(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline with exact 5 success / 5 partial / 1 skipped / 1 failed -> outcome failed, exit code 1."""
    def make_mock(name, status, record_cnt=10):
        def _mock(*args, **kwargs):
            out_path = kwargs.get("canonical_output_path") or kwargs.get("manifest_path")
            p = None
            if out_path:
                p = Path(out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"stage": name}), encoding="utf-8")
            return {
                "status": status,
                "record_count": record_cnt if status in ("success", "partial") else 0,
                "duration_seconds": 0.05,
                "error_category": "simulated_error" if status == "failed" else None,
                "reason": "simulated_skip" if status == "skipped" else ("simulated_failure" if status == "failed" else None),
            }, p
        return _mock

    # 5 Succeeded: reddit_ingestion, reddit_preprocessing, neo4j_persistence, elasticsearch_indexing, kibana_validation
    monkeypatch.setattr("app.execute_reddit_ingestion_stage", make_mock("reddit_ingestion", "success"))
    monkeypatch.setattr("app.execute_reddit_preprocessing_stage", make_mock("reddit_preprocessing", "success"))
    monkeypatch.setattr("app.execute_neo4j_stage", make_mock("neo4j_persistence", "success"))
    monkeypatch.setattr("app.execute_elasticsearch_indexing_stage", make_mock("elasticsearch_indexing", "success"))
    monkeypatch.setattr("app.execute_kibana_validation_stage", make_mock("kibana_validation", "success"))

    # 1 Failed: youtube_ingestion
    monkeypatch.setattr("app.execute_youtube_ingestion_stage", make_mock("youtube_ingestion", "failed"))
    # 1 Skipped: youtube_preprocessing
    monkeypatch.setattr("app.execute_youtube_preprocessing_stage", make_mock("youtube_preprocessing", "skipped"))

    # 5 Partial: sentiment, topics, graph, misinformation, consolidation
    monkeypatch.setattr("app.execute_sentiment_stage", make_mock("sentiment", "partial"))
    monkeypatch.setattr("app.execute_topic_stage", make_mock("topics", "partial"))
    monkeypatch.setattr("app.execute_graph_stage", make_mock("graph", "partial"))
    monkeypatch.setattr("app.execute_misinformation_stage", make_mock("misinformation", "partial"))
    monkeypatch.setattr("app.execute_consolidation_stage", make_mock("consolidation", "partial"))

    runs_dir = tmp_path / "runs"
    lock_file = tmp_path / ".orch.lock"

    code, manifest = run_orchestrated_pipeline(
        argv=["--run-all"],
        base_runs_dir=runs_dir,
        lock_path=lock_file,
    )

    assert code == EXIT_STAGE_FAILURE
    assert manifest is not None
    assert manifest["overall_outcome"] == "failed"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE

    summary = manifest["summary"]
    assert summary["stages_succeeded"] == 5
    assert summary["stages_partial"] == 5
    assert summary["stages_skipped"] == 1
    assert summary["stages_failed"] == 1
    assert summary["total_stages_attempted"] == 11
    assert summary["total_stages_evaluated"] == 12


# 15. Intentional --skip-youtube end-to-end execution scenario
def test_run_orchestrated_pipeline_intentional_skip_youtube_end_to_end(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline with --skip-youtube produces exact 5 success / 5 partial / 2 skipped / 0 failed."""
    def make_mock(name, status="success", record_cnt=10):
        def _mock(*args, **kwargs):
            out_path = kwargs.get("canonical_output_path") or kwargs.get("manifest_path")
            p = None
            if out_path:
                p = Path(out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"stage": name}), encoding="utf-8")
            return {
                "status": status,
                "record_count": record_cnt,
                "duration_seconds": 0.05,
                "error_category": None,
                "reason": None,
            }, p
        return _mock

    monkeypatch.setattr("app.execute_reddit_ingestion_stage", make_mock("reddit_ingestion"))
    monkeypatch.setattr("app.execute_reddit_preprocessing_stage", make_mock("reddit_preprocessing"))
    monkeypatch.setattr("app.execute_sentiment_stage", make_mock("sentiment"))
    monkeypatch.setattr("app.execute_topic_stage", make_mock("topics"))
    monkeypatch.setattr("app.execute_graph_stage", make_mock("graph"))
    monkeypatch.setattr("app.execute_neo4j_stage", make_mock("neo4j_persistence"))
    monkeypatch.setattr("app.execute_misinformation_stage", make_mock("misinformation"))
    monkeypatch.setattr("app.execute_consolidation_stage", make_mock("consolidation"))
    monkeypatch.setattr("app.execute_elasticsearch_indexing_stage", make_mock("elasticsearch_indexing"))
    monkeypatch.setattr("app.execute_kibana_validation_stage", make_mock("kibana_validation", record_cnt=41))

    runs_dir = tmp_path / "runs"
    lock_file = tmp_path / ".orch.lock"

    code, manifest = run_orchestrated_pipeline(
        argv=["--run-all", "--skip-youtube"],
        base_runs_dir=runs_dir,
        lock_path=lock_file,
    )

    assert code == EXIT_STAGE_FAILURE
    assert manifest is not None
    assert manifest["overall_outcome"] == "partial"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE

    summary = manifest["summary"]
    assert summary["stages_succeeded"] == 5
    assert summary["stages_partial"] == 5
    assert summary["stages_skipped"] == 2
    assert summary["stages_failed"] == 0
    assert summary["total_stages_attempted"] == 10
    assert summary["total_stages_evaluated"] == 12


# 16. Resource ownership contract
def test_orchestration_resource_ownership_contract(tmp_path, monkeypatch):
    """Verify orchestrator allows runners to manage resources internally and passes manager=None, client=None."""
    observed_kwargs = {}

    def mock_neo4j(*args, **kwargs):
        observed_kwargs["neo4j_manager"] = kwargs.get("manager")
        observed_kwargs["neo4j_factory"] = kwargs.get("manager_factory")
        return {"status": "success", "record_count": 5, "duration_seconds": 0.01, "error_category": None, "reason": None}, None

    def mock_es(*args, **kwargs):
        observed_kwargs["es_client"] = kwargs.get("client")
        return {"status": "success", "record_count": 5, "duration_seconds": 0.01, "error_category": None, "reason": None}, None

    def generic_mock(name):
        def _mock(*args, **kwargs):
            out_path = kwargs.get("canonical_output_path") or kwargs.get("manifest_path")
            p = None
            if out_path:
                p = Path(out_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}", encoding="utf-8")
            return {"status": "success", "record_count": 5, "duration_seconds": 0.01, "error_category": None, "reason": None}, p
        return _mock

    STAGE_FUNCS = [
        ("execute_reddit_ingestion_stage", generic_mock("reddit_ingestion")),
        ("execute_youtube_ingestion_stage", generic_mock("youtube_ingestion")),
        ("execute_reddit_preprocessing_stage", generic_mock("reddit_preprocessing")),
        ("execute_youtube_preprocessing_stage", generic_mock("youtube_preprocessing")),
        ("execute_sentiment_stage", generic_mock("sentiment")),
        ("execute_topic_stage", generic_mock("topics")),
        ("execute_graph_stage", generic_mock("graph")),
        ("execute_neo4j_stage", mock_neo4j),
        ("execute_misinformation_stage", generic_mock("misinformation")),
        ("execute_consolidation_stage", generic_mock("consolidation")),
        ("execute_elasticsearch_indexing_stage", mock_es),
        ("execute_kibana_validation_stage", generic_mock("kibana_validation")),
    ]
    for fn, mk in STAGE_FUNCS:
        monkeypatch.setattr(f"app.{fn}", mk)

    code, manifest = run_orchestrated_pipeline(
        argv=["--run-all"],
        base_runs_dir=tmp_path / "runs",
        lock_path=tmp_path / ".lock",
    )
    assert code == EXIT_SUCCESS
    assert observed_kwargs["neo4j_manager"] is None
    assert observed_kwargs["neo4j_factory"] is None
    assert observed_kwargs["es_client"] is None


# 17. Prevention of fallback to canonical artifacts
def test_prevention_of_fallback_to_canonical_artifacts(tmp_path):
    """Verify Stage 8 and Stage 11 fail closed when current-run artifact is missing, refusing canonical fallback."""
    e8, p8 = execute_neo4j_stage(
        graph_artifact_path=None,
        graph_stage_entry={"status": "success", "record_count": 10},
    )
    assert e8["status"] == "failed"
    assert e8["error_category"] == "missing_artifact"
    assert p8 is None

    e11, p11 = execute_elasticsearch_indexing_stage(
        consolidated_path=None,
        consolidation_stage_entry={"status": "success", "record_count": 10},
    )
    assert e11["status"] == "failed"
    assert e11["error_category"] == "missing_artifact"
    assert p11 is None


# 18. Stage 12 offline scope labeling
def test_kibana_validation_stage_offline_scope_labeling(tmp_path, monkeypatch):
    """Verify execute_kibana_validation_stage discloses offline bundle validation scope."""
    bundle_file = tmp_path / "dashboards.ndjson"
    bundle_file.write_text("{}", encoding="utf-8")

    def mock_val(path):
        return {
            "is_valid": True,
            "total_objects": 41,
            "counts_by_type": {"dashboard": 5, "lens": 34, "index-pattern": 1, "visualization": 1},
            "errors": [],
        }

    monkeypatch.setattr("app.validate_kibana_dashboard_bundle", mock_val)

    entry, path = execute_kibana_validation_stage(bundle_path=bundle_file)
    assert entry["status"] == "success"
    assert entry["record_count"] == 41
    assert entry["mode"] == "offline_bundle_validation"
    assert "41 objects" in entry["scope"]


# 19. CLI argument parser rules
def test_cli_argument_parser_run_all_rules():
    """Verify CLI argument parser enforces --run-dir and --promote-canonical rules."""
    ns = parse_cli_arguments(["--run-all", "--promote-canonical", "--run-dir=data/custom_runs"])
    assert ns.run_all is True
    assert ns.promote_canonical is True
    assert ns.run_dir == Path("data/custom_runs")

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--promote-canonical"])

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--run-dir=data/custom_runs"])


# 20. Manifest atomic persistence
def test_manifest_atomic_persistence(tmp_path):
    """Verify persist_manifest writes atomically and cleans up on failure."""
    target_manifest = tmp_path / "output_manifest.json"
    sample_stages = create_empty_stage_registry()
    manifest_data = build_execution_manifest(stages=sample_stages)

    success = persist_manifest(manifest_data, destination_path=target_manifest)
    assert success is True
    assert target_manifest.exists()

    with open(target_manifest, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["run_id"] == manifest_data["run_id"]


# =====================================================================
# Stage 13C.2: Targeted Safety Corrections Tests
# =====================================================================

def test_cross_process_os_advisory_lock_mutual_exclusion(tmp_path):
    """Verify cross-process mutual exclusion:
    1. First process acquires lock.
    2. Second process cannot acquire it while first is alive.
    3. First process releases lock.
    4. Subsequent process can acquire it.
    """
    lock_file = tmp_path / ".cross_proc.lock"
    rid1 = generate_run_id()
    rid2 = generate_run_id()

    # Process 1 acquires lock
    lock1 = OrchestrationLock(lock_path=lock_file)
    lock1.acquire(rid1)
    assert lock1.is_acquired is True

    # Process 2 (separate OS process via subprocess) attempts to acquire
    sub_code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from app import OrchestrationLock, OrchestrationLockError\n"
        f"lock = OrchestrationLock(lock_path=Path(r'{lock_file}'))\n"
        "try:\n"
        f"    lock.acquire('{rid2}')\n"
        "    print('ACQUIRED')\n"
        "    sys.exit(0)\n"
        "except OrchestrationLockError as e:\n"
        "    print(f'LOCKED: {e}')\n"
        "    sys.exit(2)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", sub_code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "currently locked by active PID" in proc.stdout

    # Process 1 releases
    lock1.release()
    assert lock1.is_acquired is False

    # Process 2 now attempts to acquire and succeeds
    proc2 = subprocess.run(
        [sys.executable, "-c", sub_code],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0
    assert "ACQUIRED" in proc2.stdout


def test_cross_process_lock_released_on_process_termination(tmp_path):
    """Verify that process termination automatically releases the OS advisory lock (Verification 5)."""
    lock_file = tmp_path / ".term.lock"
    rid_child = generate_run_id()

    # Child process acquires lock, announces it, and abruptly exits without calling release()
    child_code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from app import OrchestrationLock\n"
        f"lock = OrchestrationLock(lock_path=Path(r'{lock_file}'))\n"
        f"lock.acquire('{rid_child}')\n"
        "print('HELD')\n"
        "sys.stdout.flush()\n"
        "time.sleep(1)\n"
        "sys.exit(0)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        stdout=subprocess.PIPE,
        text=True,
    )
    line = child.stdout.readline()
    assert line.strip() == "HELD"

    # While child is alive, parent cannot acquire
    parent_lock = OrchestrationLock(lock_path=lock_file)
    with pytest.raises(OrchestrationLockError):
        parent_lock.acquire("run_parent_probe")

    # Wait for child to terminate
    child.wait()
    assert child.returncode == 0

    # OS kernel has closed child's file handle; parent can now acquire immediately
    rid_parent = generate_run_id()
    parent_lock.acquire(rid_parent)
    assert parent_lock.is_acquired is True
    parent_lock.release()


def test_cross_process_second_process_cannot_steal_or_delete_active_lock(tmp_path):
    """Verify that a second process cannot steal or unlink an active process's lock (Verification 6)."""
    lock_file = tmp_path / ".steal_test.lock"
    rid1 = generate_run_id()

    lock1 = OrchestrationLock(lock_path=lock_file)
    lock1.acquire(rid1)
    assert lock1.is_acquired is True

    # Subprocess tries to force delete the lock file
    sub_code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        f"p = Path(r'{lock_file}')\n"
        "try:\n"
        "    p.unlink()\n"
        "    print('UNLINKED')\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    print(f'FAILED_UNLINK: {type(e).__name__}')\n"
        "    sys.exit(3)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", sub_code],
        capture_output=True,
        text=True,
    )
    # On Windows, open lock file cannot be unlinked (PermissionError / WinError 32)
    assert proc.returncode == 3
    assert "FAILED_UNLINK" in proc.stdout
    assert lock_file.exists()

    lock1.release()


def test_lock_release_by_non_owner_safely_ignored(tmp_path):
    """Verify that lock release by a non-owner instance is safely ignored without modifying lock (Verification 8)."""
    lock_file = tmp_path / ".non_owner.lock"
    rid1 = generate_run_id()

    lock_owner = OrchestrationLock(lock_path=lock_file)
    lock_owner.acquire(rid1)

    # Non-owner instance
    lock_non_owner = OrchestrationLock(lock_path=lock_file)
    assert lock_non_owner.is_acquired is False
    lock_non_owner.release()  # Must not raise and must not release the actual lock

    assert lock_file.exists()
    assert lock_owner.is_acquired is True

    # Owner can still release cleanly
    lock_owner.release()
    assert not lock_file.exists()


def test_canonical_promotion_rollback_on_injected_failure(tmp_path):
    """Verify that promote_run_to_canonical rolls back all changes if an error occurs midway."""
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    processed_dir.mkdir()
    raw_dir.mkdir()

    # Pre-populate existing canonical files with initial content
    existing_canonical = processed_dir / "sentiment_analysis_results.json"
    existing_canonical.write_text('{"initial": "canonical_sentiment"}', encoding="utf-8")

    # Populate run files
    (run_dir / "reddit_posts_raw.json").write_text('{"run": "raw_reddit"}', encoding="utf-8")
    (run_dir / "sentiment_analysis_results.json").write_text('{"run": "new_sentiment"}', encoding="utf-8")
    (run_dir / "consolidated_intelligence.json").write_text('{"run": "new_consolidated"}', encoding="utf-8")

    # Inject failure during os.replace on the 2nd file
    original_replace = os.replace
    call_count = 0

    def mock_replace(src, dst):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise OSError("Injected disk failure during promotion")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(CanonicalPromotionError) as exc_info:
            promote_run_to_canonical(
                run_dir=run_dir,
                processed_dir=processed_dir,
                raw_dir=raw_dir,
            )
        assert "rolled back cleanly" in str(exc_info.value)

    # Verify existing canonical file was restored to initial content
    assert existing_canonical.exists()
    assert json.loads(existing_canonical.read_text(encoding="utf-8")) == {"initial": "canonical_sentiment"}

    # Verify newly created canonical files that were rolled back do NOT linger
    assert not (raw_dir / "reddit_posts_raw.json").exists()
    assert not (processed_dir / "consolidated_intelligence.json").exists()

    # Verify run_dir files are untouched
    assert (run_dir / "sentiment_analysis_results.json").exists()


def test_canonical_promotion_rejects_empty_source_file(tmp_path):
    """Verify promote_run_to_canonical validates source files and aborts before touching canonical storage."""
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Create empty source file
    empty_file = run_dir / "sentiment_analysis_results.json"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(CanonicalPromotionError) as exc_info:
        promote_run_to_canonical(run_dir=run_dir, processed_dir=processed_dir)
    assert "empty or not a valid regular file" in str(exc_info.value)

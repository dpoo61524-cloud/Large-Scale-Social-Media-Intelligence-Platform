"""Basic validation checks for main application entry point."""

import sys
from app import main


def test_app_main_runs_without_exceptions(capsys):
    """Verify app.py main() executes cleanly and outputs expected information."""
    main()
    captured = capsys.readouterr()

    # Verify standard project headers and sections exist
    assert "LARGE SCALE SOCIAL MEDIA INTELLIGENCE PLATFORM" in captured.out
    assert "Block 1 - Project Foundation Initialized" in captured.out
    assert "[PROJECT OBJECTIVES]" in captured.out
    assert "[ENVIRONMENT & STORAGE STATUS]" in captured.out


def test_app_main_with_ingest_flag(capsys, monkeypatch):
    """Verify app.py runs mock Reddit ingestion when --ingest-reddit is passed."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest-reddit"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT MOCK INGESTION" in captured.out
    assert "Total Records" in captured.out
    assert "Valid Records" in captured.out


def test_app_main_with_live_reddit_flag(capsys, monkeypatch):
    """Verify app.py handles live Reddit ingestion flag gracefully when unconfigured."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest-reddit-live"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT LIVE API INGESTION" in captured.out
    assert "Reddit Live API" in captured.out


def test_app_main_with_youtube_flag(capsys, monkeypatch):
    """Verify app.py handles live YouTube ingestion flag gracefully when unconfigured."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest-youtube"])
    main()
    captured = capsys.readouterr()

    assert "STARTING YOUTUBE DATA API INGESTION" in captured.out
    assert "YouTube Data API v3" in captured.out


def test_app_main_with_preprocess_reddit_flag(capsys, monkeypatch):
    """Verify app.py handles Reddit preprocessing flag."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--preprocess-reddit"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT TEXT PREPROCESSING (STEP 2)" in captured.out
    assert "Reddit Posts" in captured.out


def test_app_main_with_preprocess_youtube_flag(capsys, monkeypatch):
    """Verify app.py handles YouTube preprocessing flag."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--preprocess-youtube"])
    main()
    captured = capsys.readouterr()

    assert "STARTING YOUTUBE TEXT PREPROCESSING (STEP 2)" in captured.out
    assert "YouTube Videos" in captured.out


# =====================================================================
# STEP 7.5C — Block 1: Manifest Foundation & Stage Registry Tests
# =====================================================================

import json
from pathlib import Path
import pytest
from app import (
    CANONICAL_STAGES,
    DEFAULT_MANIFEST_PATH,
    EXIT_MANIFEST_PERSISTENCE_FAILURE,
    EXIT_ORCHESTRATION_NOT_IMPLEMENTED,
    EXIT_STAGE_FAILURE,
    EXIT_SUCCESS,
    PERMITTED_STAGE_STATUSES,
    ManifestPersistenceError,
    _validate_active_sentiment_source_file,
    _validate_active_topic_source_file,
    build_execution_manifest,
    calculate_manifest_summary,
    create_empty_stage_registry,
    create_initial_stage_entry,
    execute_consolidation_stage,
    execute_elasticsearch_indexing_stage,
    execute_graph_stage,
    execute_kibana_validation_stage,
    execute_misinformation_stage,
    execute_neo4j_stage,
    execute_reddit_ingestion_stage,
    execute_reddit_preprocessing_stage,
    execute_sentiment_stage,
    execute_topic_stage,
    execute_youtube_ingestion_stage,
    execute_youtube_preprocessing_stage,
    persist_manifest,
    validate_reddit_ingestion_output,
    validate_reddit_preprocessing_output,
    validate_sentiment_output,
    validate_topic_output,
    validate_youtube_ingestion_output,
    validate_youtube_preprocessing_output,
)
from src.sentiment.sentiment_analyzer import MultilingualSentimentAnalyzer
from tests.test_sentiment_pipeline import MockSentimentPipeline
from tests.test_topic_pipeline import MockTopicModeler


def test_canonical_stages_registry_properties():
    """Verify CANONICAL_STAGES has exactly 12 stages in deterministic order and valid status set."""
    expected_order = (
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
    assert CANONICAL_STAGES == expected_order
    assert len(CANONICAL_STAGES) == 12
    assert len(set(CANONICAL_STAGES)) == 12
    assert PERMITTED_STAGE_STATUSES == frozenset(
        {"success", "partial", "empty", "failed", "skipped"}
    )


def test_create_initial_stage_entry_valid():
    """Verify stage entry creation with valid attributes."""
    entry = create_initial_stage_entry(
        status="success",
        reason=None,
        record_count=150,
        duration_seconds=1.23456,
        error_category=None,
    )
    assert entry == {
        "status": "success",
        "record_count": 150,
        "duration_seconds": 1.2346,
        "error_category": None,
        "reason": None,
    }


@pytest.mark.parametrize("invalid_status", ["error", "pending", "unknown", "invalid", "", None])
def test_create_initial_stage_entry_rejects_invalid_status(invalid_status):
    """Verify stage entry creation rejects any status outside the permitted three."""
    with pytest.raises(ValueError, match="Invalid status"):
        create_initial_stage_entry(status=invalid_status)  # type: ignore


@pytest.mark.parametrize("invalid_count", [-1, -100, True, False, "10", 1.5, None])
def test_create_initial_stage_entry_rejects_invalid_record_count(invalid_count):
    """Verify stage entry creation strictly rejects negative, non-integer, or boolean counts."""
    with pytest.raises(ValueError, match="record_count must be a non-negative integer"):
        create_initial_stage_entry(status="success", record_count=invalid_count)  # type: ignore


@pytest.mark.parametrize(
    "invalid_duration",
    [
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.0001,
        -1.0,
        -50,
        -(10**1000),
        -(10**500),
        "1.0",
        "0",
        None,
        [],
        {},
    ],
)
def test_create_initial_stage_entry_rejects_invalid_duration(invalid_duration):
    """Verify stage entry creation strictly rejects boolean, non-numeric, negative, NaN, and infinite duration values."""
    with pytest.raises(ValueError, match="duration_seconds must be a finite, non-negative number"):
        create_initial_stage_entry(status="success", duration_seconds=invalid_duration)  # type: ignore


@pytest.mark.parametrize("valid_duration,expected_stored", [
    (0, 0.0),
    (0.0, 0.0),
    (1, 1.0),
    (1.23456, 1.2346),
    (99.99994, 99.9999),
    (100.0, 100.0),
    (10**500, 10**500),
    (10**1000, 10**1000),
])
def test_create_initial_stage_entry_accepts_valid_durations(valid_duration, expected_stored):
    """Verify stage entry accepts finite non-negative integer and float durations rounded to 4 decimals."""
    entry = create_initial_stage_entry(status="success", duration_seconds=valid_duration)
    assert entry["duration_seconds"] == expected_stored


def test_create_initial_stage_entry_extreme_integers():
    """Verify extremely large positive integers do not raise OverflowError and large negative integers raise ValueError."""
    extreme_pos_int = 10**1000
    extreme_neg_int = -(10**1000)

    # Extreme positive integer must be accepted without OverflowError
    entry = create_initial_stage_entry(status="success", duration_seconds=extreme_pos_int)
    assert entry["duration_seconds"] == extreme_pos_int

    # Extreme negative integer must raise ValueError (not OverflowError)
    with pytest.raises(ValueError, match="duration_seconds must be a finite, non-negative number"):
        create_initial_stage_entry(status="success", duration_seconds=extreme_neg_int)


def test_create_empty_stage_registry():
    """Verify empty stage registry initializes all 12 canonical stages with defaults."""
    registry = create_empty_stage_registry(default_status="skipped", default_reason="stage_not_requested")
    assert len(registry) == 12
    assert tuple(registry.keys()) == CANONICAL_STAGES
    for stage_name, entry in registry.items():
        assert entry["status"] == "skipped"
        assert entry["reason"] == "stage_not_requested"
        assert entry["record_count"] == 0
        assert entry["duration_seconds"] == 0.0
        assert entry["error_category"] is None


def test_calculate_manifest_summary_exact_8_1_3_example():
    """Verify summary arithmetic for the canonical 8 success / 1 failed / 3 skipped scenario."""
    stages = create_empty_stage_registry()

    # 8 Succeeded
    for s in [
        "reddit_ingestion",
        "reddit_preprocessing",
        "sentiment",
        "graph",
        "misinformation",
        "consolidation",
        "elasticsearch_indexing",
        "kibana_validation",
    ]:
        stages[s] = create_initial_stage_entry(status="success", record_count=50, duration_seconds=1.0)

    # 1 Failed
    stages["topics"] = create_initial_stage_entry(
        status="failed", error_category="topic_divergence_error", duration_seconds=0.5
    )

    # 3 Skipped
    stages["youtube_ingestion"]["reason"] = "source_disabled"
    stages["youtube_preprocessing"]["reason"] = "source_disabled"
    stages["neo4j_persistence"]["reason"] = "credentials_unconfigured"

    summary = calculate_manifest_summary(stages)

    assert summary["stages_succeeded"] == 8
    assert summary["stages_failed"] == 1
    assert summary["stages_skipped"] == 3
    assert summary["total_stages_attempted"] == 9
    assert summary["total_stages_evaluated"] == 12

    # Verify arithmetic equations strictly hold
    assert summary["stages_succeeded"] + summary["stages_failed"] == summary["total_stages_attempted"]
    assert summary["total_stages_attempted"] + summary["stages_skipped"] == 12
    assert summary["total_stages_evaluated"] == 12


def test_calculate_manifest_summary_all_success():
    """Verify summary arithmetic when all 12 stages succeed."""
    stages = {
        s: create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.2)
        for s in CANONICAL_STAGES
    }
    summary = calculate_manifest_summary(stages)

    assert summary["stages_succeeded"] == 12
    assert summary["stages_failed"] == 0
    assert summary["stages_skipped"] == 0
    assert summary["total_stages_attempted"] == 12
    assert summary["total_stages_evaluated"] == 12


def test_calculate_manifest_summary_all_skipped():
    """Verify summary arithmetic when all 12 stages are skipped."""
    stages = create_empty_stage_registry(default_status="skipped", default_reason="prior_stage_failure")
    summary = calculate_manifest_summary(stages)

    assert summary["stages_succeeded"] == 0
    assert summary["stages_failed"] == 0
    assert summary["stages_skipped"] == 12
    assert summary["total_stages_attempted"] == 0
    assert summary["total_stages_evaluated"] == 12


def test_calculate_manifest_summary_mixed_scenario():
    """Verify summary arithmetic in a mixed scenario with 6 succeeded, 2 failed, 4 skipped."""
    stages = create_empty_stage_registry()
    success_stages = list(CANONICAL_STAGES[:6])
    failed_stages = list(CANONICAL_STAGES[6:8])

    for s in success_stages:
        stages[s] = create_initial_stage_entry(status="success", record_count=20, duration_seconds=0.5)
    for s in failed_stages:
        stages[s] = create_initial_stage_entry(status="failed", error_category="exec_error", duration_seconds=0.3)

    summary = calculate_manifest_summary(stages)

    assert summary["stages_succeeded"] == 6
    assert summary["stages_failed"] == 2
    assert summary["stages_skipped"] == 4
    assert summary["total_stages_attempted"] == 8
    assert summary["total_stages_evaluated"] == 12


def test_calculate_manifest_summary_rejection_missing_stage():
    """Verify summary calculation rejects stages dict missing a canonical key."""
    stages = create_empty_stage_registry()
    del stages["kibana_validation"]

    with pytest.raises(ValueError, match="Expected exactly 12 stages, got 11"):
        calculate_manifest_summary(stages)


def test_calculate_manifest_summary_rejection_unknown_stage():
    """Verify summary calculation rejects stages dict with unknown keys."""
    stages = create_empty_stage_registry()
    del stages["kibana_validation"]
    stages["unknown_stage_name"] = create_initial_stage_entry(status="skipped")

    with pytest.raises(ValueError, match="Stage registry mismatch"):
        calculate_manifest_summary(stages)


def test_calculate_manifest_summary_rejection_invalid_stage_type():
    """Verify summary calculation rejects non-dict stage values."""
    stages = create_empty_stage_registry()
    stages["topics"] = "invalid_not_a_dict"  # type: ignore

    with pytest.raises(TypeError, match="Stage 'topics' data must be a dict"):
        calculate_manifest_summary(stages)


def test_calculate_manifest_summary_rejection_invalid_status_in_stage():
    """Verify summary calculation rejects invalid status value embedded in stage dict."""
    stages = create_empty_stage_registry()
    stages["sentiment"] = {"status": "unrecognized_status", "record_count": 0}

    with pytest.raises(ValueError, match="invalid status 'unrecognized_status'"):
        calculate_manifest_summary(stages)


def test_build_execution_manifest_structure_and_invariants():
    """Verify build_execution_manifest constructs a complete, correctly ordered manifest."""
    stages = create_empty_stage_registry()
    stages["reddit_ingestion"] = create_initial_stage_entry(status="success", record_count=100)

    manifest = build_execution_manifest(
        stages=stages,
        pipeline_mode="full",
        run_id="test_run_12345",
        timestamp_utc="2026-11-07T20:00:00Z",
    )

    assert manifest["run_id"] == "test_run_12345"
    assert manifest["timestamp_utc"] == "2026-11-07T20:00:00Z"
    assert manifest["pipeline_mode"] == "full"
    assert manifest["overall_outcome"] == "success"
    assert manifest["exit_code"] == EXIT_SUCCESS

    # Stages ordering must strictly match CANONICAL_STAGES
    assert tuple(manifest["stages"].keys()) == CANONICAL_STAGES

    summary = manifest["summary"]
    assert summary["stages_succeeded"] == 1
    assert summary["stages_failed"] == 0
    assert summary["stages_skipped"] == 11
    assert summary["total_stages_attempted"] == 1
    assert summary["total_stages_evaluated"] == 12


def test_build_execution_manifest_infers_failure_outcome_and_exit_code():
    """Verify manifest automatically infers outcome='failed' and exit_code=1 on stage failure."""
    stages = create_empty_stage_registry()
    stages["topics"] = create_initial_stage_entry(status="failed", error_category="model_error")

    manifest = build_execution_manifest(stages=stages)
    assert manifest["overall_outcome"] == "failed"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE
    assert manifest["summary"]["stages_failed"] == 1


def test_persist_manifest_creates_nested_directory_and_valid_json(tmp_path: Path):
    """Verify manifest persistence creates parent directories and writes valid JSON."""
    stages = create_empty_stage_registry()
    manifest = build_execution_manifest(stages=stages, run_id="nested_test")

    dest_file = tmp_path / "deep" / "nested" / "output_manifest.json"
    assert not dest_file.exists()

    result = persist_manifest(manifest, destination_path=dest_file)
    assert result is True
    assert dest_file.exists()

    with open(dest_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["run_id"] == "nested_test"
    assert loaded["summary"]["total_stages_evaluated"] == 12
    assert tuple(loaded["stages"].keys()) == CANONICAL_STAGES


def test_persist_manifest_atomic_replacement(tmp_path: Path):
    """Verify persist_manifest replaces an existing destination file cleanly."""
    stages = create_empty_stage_registry()
    manifest_v1 = build_execution_manifest(stages=stages, run_id="v1")
    manifest_v2 = build_execution_manifest(stages=stages, run_id="v2")

    target = tmp_path / "manifest.json"

    # Write v1
    assert persist_manifest(manifest_v1, destination_path=target) is True
    with open(target, "r", encoding="utf-8") as f:
        assert json.load(f)["run_id"] == "v1"

    # Atomically replace with v2
    assert persist_manifest(manifest_v2, destination_path=target) is True
    with open(target, "r", encoding="utf-8") as f:
        assert json.load(f)["run_id"] == "v2"


def test_persist_manifest_cleans_up_temporary_file_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify temporary staging file is unlinked and removed when atomic replacement fails."""
    stages = create_empty_stage_registry()
    manifest = build_execution_manifest(stages=stages)
    target = tmp_path / "manifest.json"

    # Simulate an error during os.replace
    def failing_replace(src, dst):
        raise OSError("Simulated filesystem I/O error during atomic replace")

    monkeypatch.setattr("os.replace", failing_replace)

    result = persist_manifest(manifest, destination_path=target)
    assert result is False
    assert not target.exists()

    # Verify no leftover temporary files in tmp_path
    leftover_tmps = list(tmp_path.glob(".tmp_*"))
    assert len(leftover_tmps) == 0


def test_persist_manifest_raises_controlled_exception_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify persist_manifest raises ManifestPersistenceError with sanitized message when requested."""
    stages = create_empty_stage_registry()
    manifest = build_execution_manifest(stages=stages)
    target = tmp_path / "manifest.json"

    def failing_replace(src, dst):
        raise PermissionError("Access denied")

    monkeypatch.setattr("os.replace", failing_replace)

    with pytest.raises(ManifestPersistenceError) as exc_info:
        persist_manifest(manifest, destination_path=target, raise_on_error=True)

    err_msg = str(exc_info.value)
    assert "Failed to persist execution manifest" in err_msg
    assert "PermissionError" in err_msg
    # Ensure sensitive paths or internal stack details are not embedded in message
    assert str(tmp_path) not in err_msg


def test_manifest_contains_no_sensitive_fields_or_raw_exceptions():
    """Verify manifest schema contains zero credentials, auth tokens, stack traces, or absolute paths."""
    stages = create_empty_stage_registry()
    stages["reddit_ingestion"] = create_initial_stage_entry(
        status="success", record_count=100, duration_seconds=1.2
    )
    stages["topics"] = create_initial_stage_entry(
        status="failed", error_category="topic_divergence_error", duration_seconds=0.4
    )

    manifest = build_execution_manifest(stages=stages)
    serialized = json.dumps(manifest)

    disallowed_substrings = [
        "password",
        "secret",
        "api_key",
        "bearer",
        "authorization",
        "Traceback",
        "File \"/",
        "File \"C:\\",
    ]
    for pattern in disallowed_substrings:
        assert pattern.lower() not in serialized.lower()


# =====================================================================
# STEP 7.5C — Block 2: CLI Argument Parsing and Dispatch Tests
# =====================================================================

import argparse
from app import (
    build_argument_parser,
    parse_cli_arguments,
    run_orchestrated_pipeline,
)


def test_build_argument_parser_structure():
    """Verify ArgumentParser is configured with expected program name and help options."""
    parser = build_argument_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "python app.py"

    actions = [a.dest for a in parser._actions]
    assert "run_all" in actions
    assert "ingest_reddit" in actions
    assert "ingest_reddit_live" in actions
    assert "ingest_youtube" in actions
    assert "preprocess_reddit" in actions
    assert "preprocess_youtube" in actions
    assert "skip_reddit" in actions
    assert "skip_youtube" in actions
    assert "skip_neo4j" in actions
    assert "skip_kibana" in actions
    assert "manifest_path" in actions


@pytest.mark.parametrize(
    "legacy_flag,expected_dest",
    [
        ("--ingest-reddit", "ingest_reddit"),
        ("--ingest", "ingest_reddit"),
        ("--ingest-reddit-live", "ingest_reddit_live"),
        ("--ingest-youtube", "ingest_youtube"),
        ("--preprocess-reddit", "preprocess_reddit"),
        ("--preprocess-youtube", "preprocess_youtube"),
    ],
)
def test_every_legacy_command_accepted(legacy_flag, expected_dest):
    """Verify every legacy CLI command is accepted and sets the appropriate attribute."""
    parsed = parse_cli_arguments([legacy_flag])
    assert getattr(parsed, expected_dest) is True
    assert parsed.run_all is False


def test_legacy_dispatch_behavior(monkeypatch: pytest.MonkeyPatch):
    """Verify main() dispatches to the correct legacy CLI functions."""
    dispatched = {}

    monkeypatch.setattr("app.run_mock_reddit_cli", lambda: dispatched.update({"mock_reddit": True}))
    monkeypatch.setattr("app.run_live_reddit_cli", lambda: dispatched.update({"live_reddit": True}))
    monkeypatch.setattr("app.run_youtube_cli", lambda: dispatched.update({"youtube": True}))
    monkeypatch.setattr("app.run_preprocess_reddit_cli", lambda: dispatched.update({"prep_reddit": True}))
    monkeypatch.setattr("app.run_preprocess_youtube_cli", lambda: dispatched.update({"prep_youtube": True}))

    main(["--ingest-reddit"])
    assert dispatched.get("mock_reddit") is True

    dispatched.clear()
    main(["--ingest-reddit-live"])
    assert dispatched.get("live_reddit") is True

    dispatched.clear()
    main(["--ingest-youtube"])
    assert dispatched.get("youtube") is True

    dispatched.clear()
    main(["--preprocess-reddit"])
    assert dispatched.get("prep_reddit") is True

    dispatched.clear()
    main(["--preprocess-youtube"])
    assert dispatched.get("prep_youtube") is True


def test_run_all_accepted():
    """Verify --run-all flag is accepted and defaults to all skip flags False and default manifest path."""
    parsed = parse_cli_arguments(["--run-all"])
    assert parsed.run_all is True
    assert parsed.skip_reddit is False
    assert parsed.skip_youtube is False
    assert parsed.skip_neo4j is False
    assert parsed.skip_elasticsearch is False
    assert parsed.skip_kibana is False
    assert parsed.manifest_path == DEFAULT_MANIFEST_PATH


@pytest.mark.parametrize(
    "skip_flag,expected_attr",
    [
        ("--skip-reddit", "skip_reddit"),
        ("--skip-youtube", "skip_youtube"),
        ("--skip-neo4j", "skip_neo4j"),
        ("--skip-elasticsearch", "skip_elasticsearch"),
        ("--skip-kibana", "skip_kibana"),
    ],
)
def test_each_skip_flag_accepted_with_run_all(skip_flag, expected_attr):
    """Verify each skip flag is accepted when --run-all is present."""
    parsed = parse_cli_arguments(["--run-all", skip_flag])
    assert parsed.run_all is True
    assert getattr(parsed, expected_attr) is True


def test_all_skip_flags_together_with_run_all():
    """Verify all five skip flags can be provided simultaneously with --run-all."""
    parsed = parse_cli_arguments([
        "--run-all",
        "--skip-reddit",
        "--skip-youtube",
        "--skip-neo4j",
        "--skip-elasticsearch",
        "--skip-kibana",
    ])
    assert parsed.run_all is True
    assert parsed.skip_reddit is True
    assert parsed.skip_youtube is True
    assert parsed.skip_neo4j is True
    assert parsed.skip_elasticsearch is True
    assert parsed.skip_kibana is True


@pytest.mark.parametrize(
    "skip_flag",
    [
        "--skip-reddit",
        "--skip-youtube",
        "--skip-neo4j",
        "--skip-elasticsearch",
        "--skip-kibana",
    ],
)
def test_skip_flags_without_run_all_rejected(skip_flag, capsys):
    """Verify skip flags without --run-all are rejected with controlled parser error."""
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_arguments([skip_flag])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert f"Skip flags ({skip_flag}) can only be used with --run-all." in captured.err


@pytest.mark.parametrize(
    "conflicting_pair",
    [
        ["--ingest-reddit", "--preprocess-reddit"],
        ["--ingest-reddit-live", "--ingest-youtube"],
        ["--preprocess-reddit", "--preprocess-youtube"],
        ["--ingest", "--ingest-reddit-live"],
        ["--ingest-youtube", "--preprocess-reddit"],
    ],
)
def test_conflicting_legacy_commands_rejected(conflicting_pair, capsys):
    """Verify providing multiple legacy primary execution commands raises parser error."""
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_arguments(conflicting_pair)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Cannot combine multiple execution commands" in captured.err


@pytest.mark.parametrize(
    "legacy_cmd",
    [
        "--ingest-reddit",
        "--ingest",
        "--ingest-reddit-live",
        "--ingest-youtube",
        "--preprocess-reddit",
        "--preprocess-youtube",
    ],
)
def test_run_all_cannot_be_combined_with_legacy_commands(legacy_cmd, capsys):
    """Verify --run-all cannot be combined with any legacy primary command."""
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_arguments(["--run-all", legacy_cmd])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--run-all cannot be combined with legacy execution commands" in captured.err


def test_custom_manifest_path_parsed_correctly():
    """Verify --manifest-path is parsed as a Path object when provided with --run-all."""
    custom_path_str = "data/custom_dir/custom_manifest.json"
    parsed = parse_cli_arguments(["--run-all", "--manifest-path", custom_path_str])
    assert parsed.run_all is True
    assert parsed.manifest_path == Path(custom_path_str)


def test_default_manifest_path_preserved():
    """Verify default manifest path is preserved when --manifest-path is not specified."""
    parsed = parse_cli_arguments(["--run-all"])
    assert parsed.manifest_path == DEFAULT_MANIFEST_PATH


def test_manifest_path_without_run_all_rejected(capsys):
    """Verify --manifest-path without --run-all raises a controlled parser error."""
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_arguments(["--manifest-path", "data/custom.json"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--manifest-path can only be used with --run-all." in captured.err


def test_run_orchestrated_pipeline_dispatch_contract(capsys, monkeypatch, tmp_path):
    """Verify run_orchestrated_pipeline executes the 12-stage unified orchestration flow in Stage 13C."""
    from app import CANONICAL_STAGES
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
            monkeypatch.setattr(
                f"app.{func_name}",
                lambda *args, **kwargs: (
                    {"status": "success", "record_count": 41, "duration_seconds": 0.01, "error_category": None, "reason": None},
                    None,
                ),
            )
        else:
            monkeypatch.setattr(
                f"app.{func_name}",
                lambda *args, **kwargs: (
                    {"status": "success", "record_count": 10, "duration_seconds": 0.01, "error_category": None, "reason": None},
                    kwargs.get("canonical_output_path"),
                ),
            )

    test_runs_dir = tmp_path / "runs"
    test_lock = tmp_path / ".orch.lock"

    exit_code, manifest = run_orchestrated_pipeline(
        ["--run-all"],
        base_runs_dir=test_runs_dir,
        lock_path=test_lock,
    )
    assert exit_code == EXIT_SUCCESS
    assert manifest is not None
    assert manifest["overall_outcome"] == "success"

    captured = capsys.readouterr()
    assert "[UNIFIED SOCIAL INTELLIGENCE PIPELINE ORCHESTRATION]" in captured.out
    assert "Skip YouTube  : False" in captured.out


def test_run_all_fails_gracefully_on_lock_conflict(tmp_path, capsys):
    """Verify --run-all returns EXIT_LOCK_ACQUISITION_FAILURE when lock is already held."""
    from app import OrchestrationLock, EXIT_LOCK_ACQUISITION_FAILURE, generate_run_id
    lock_file = tmp_path / ".active.lock"
    first_lock = OrchestrationLock(lock_path=lock_file)
    first_lock.acquire(generate_run_id())

    try:
        custom_manifest = tmp_path / "sub" / "custom_manifest.json"
        exit_code, manifest = run_orchestrated_pipeline(
            ["--run-all", "--manifest-path", str(custom_manifest)],
            lock_path=lock_file,
        )
        assert exit_code == EXIT_LOCK_ACQUISITION_FAILURE
        assert manifest is None
        assert not custom_manifest.exists()
    finally:
        first_lock.release()


def test_main_propagates_orchestrated_run_all_exit_code(monkeypatch):
    """Verify main() propagates the runner exit code when --run-all is requested in Stage 13C."""
    monkeypatch.setattr(
        "app.run_orchestrated_pipeline",
        lambda argv: (EXIT_STAGE_FAILURE, None),
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["--run-all"])
    assert exc_info.value.code == EXIT_STAGE_FAILURE


def test_legacy_ingest_alias_behavior(capsys, monkeypatch):
    """Verify legacy --ingest alias executes mock Reddit ingestion identically to --ingest-reddit."""
    monkeypatch.setattr(sys, "argv", ["app.py", "--ingest"])
    main()
    captured = capsys.readouterr()

    assert "STARTING REDDIT MOCK INGESTION" in captured.out
    assert "Total Records" in captured.out
    assert "Valid Records" in captured.out


def test_legacy_ingest_alias_via_argv(capsys):
    """Verify legacy --ingest alias executes mock Reddit ingestion when passed directly via argv."""
    main(["--ingest"])
    captured = capsys.readouterr()

    assert "STARTING REDDIT MOCK INGESTION" in captured.out
    assert "Total Records" in captured.out
    assert "Valid Records" in captured.out


def test_run_orchestrated_pipeline_requires_run_all():
    """Verify run_orchestrated_pipeline raises ValueError if --run-all is absent."""
    with pytest.raises(ValueError, match="run_orchestrated_pipeline requires --run-all"):
        run_orchestrated_pipeline(["--preprocess-reddit"])


def test_parser_only_tests_invoke_no_external_pipeline_functions(monkeypatch: pytest.MonkeyPatch):
    """Verify parser validation never invokes external pipeline stages or functions."""
    def fail_call(*args, **kwargs):
        pytest.fail("External pipeline function was unexpectedly called during argument parsing!")

    monkeypatch.setattr("src.ingestion.reddit_ingestion.ingest_reddit_posts", fail_call)
    monkeypatch.setattr("src.ingestion.reddit_ingestion.ingest_reddit_live", fail_call)
    monkeypatch.setattr("src.ingestion.youtube_ingestion.ingest_youtube_live", fail_call)
    monkeypatch.setattr("src.preprocessing.preprocessing_pipeline.preprocess_reddit_dataset", fail_call)
    monkeypatch.setattr("src.preprocessing.preprocessing_pipeline.preprocess_youtube_dataset", fail_call)

    # Valid parsing
    parse_cli_arguments(["--run-all", "--skip-reddit", "--skip-youtube"])
    parse_cli_arguments(["--ingest-reddit"])

    # Invalid parsing (parser exits cleanly with SystemExit)
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--skip-reddit"])
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--run-all", "--ingest-youtube"])


# =====================================================================
# STEP 7.5C — Block 3A: Reddit Ingestion Execution & Validation Tests
# =====================================================================

def _sample_reddit_records(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "post_id": f"t3_sample_{i}",
            "subreddit": "datascience",
            "title": f"Valid Reddit Post Title {i}",
            "text": f"Valid content text body {i}",
            "author": f"author_{i}",
            "created_utc": 1730000000 + i * 100,
            "score": 50 + i * 5,
            "num_comments": i * 2,
            "url": f"https://reddit.com/r/datascience/comments/sample_{i}",
            "collected_at": "2026-11-07T12:00:00Z",
        }
        for i in range(1, count + 1)
    ]


def test_reddit_ingestion_selects_mock_when_credentials_absent(tmp_path, monkeypatch):
    """Verify mock ingestion is selected when Reddit credentials are not configured."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "")
    canonical_file = tmp_path / "reddit_raw.json"

    records = _sample_reddit_records(3)

    def mock_ingest_posts(source_path=None, output_path=None, merge=True):
        assert merge is False
        assert output_path is not None
        Path(output_path).write_text(json.dumps(records), encoding="utf-8")
        return {
            "status": "success",
            "source": "reddit_mock",
            "total_records": 3,
            "valid_records": 3,
            "rejected_records": 0,
            "output_file": str(output_path),
            "errors": [],
        }

    def fail_live(*args, **kwargs):
        pytest.fail("Live ingestion called when credentials were absent!")

    monkeypatch.setattr("app.ingest_reddit_posts", mock_ingest_posts)
    monkeypatch.setattr("app.ingest_reddit_live", fail_live)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 3
    assert stage_entry["reason"] is None
    assert stage_entry["error_category"] is None
    assert path == canonical_file
    assert canonical_file.exists()


def test_reddit_ingestion_selects_live_when_credentials_configured(tmp_path, monkeypatch):
    """Verify live ingestion is selected when Reddit credentials are configured."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "valid_client_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "valid_client_secret")
    canonical_file = tmp_path / "reddit_raw.json"

    records = _sample_reddit_records(2)

    def mock_ingest_live(subreddits=None, limit_per_subreddit=25, output_path=None, merge=True):
        assert merge is False
        assert output_path is not None
        Path(output_path).write_text(json.dumps(records), encoding="utf-8")
        return {
            "status": "success",
            "source": "reddit_live_api",
            "total_records": 2,
            "valid_records": 2,
            "rejected_records": 0,
            "output_file": str(output_path),
            "rate_limits": {"remaining": "500"},
            "errors": [],
        }

    def fail_mock(*args, **kwargs):
        pytest.fail("Mock ingestion called when credentials were configured!")

    monkeypatch.setattr("app.ingest_reddit_live", mock_ingest_live)
    monkeypatch.setattr("app.ingest_reddit_posts", fail_mock)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()


def test_reddit_ingestion_executes_only_one_function(tmp_path, monkeypatch):
    """Verify only one of the ingestion functions is invoked per run."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "")
    canonical_file = tmp_path / "reddit_raw.json"

    posts_call_count = 0
    live_call_count = 0

    def mock_posts(*args, **kwargs):
        nonlocal posts_call_count
        posts_call_count += 1
        output_path = kwargs.get("output_path")
        Path(output_path).write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    def mock_live(*args, **kwargs):
        nonlocal live_call_count
        live_call_count += 1
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_reddit_posts", mock_posts)
    monkeypatch.setattr("app.ingest_reddit_live", mock_live)

    execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert posts_call_count == 1
    assert live_call_count == 0


def test_reddit_ingestion_live_failure_does_not_fall_back_to_mock(tmp_path, monkeypatch):
    """Verify configured live ingestion failure records failure and does not fall back to mock."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "configured_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "configured_secret")
    canonical_file = tmp_path / "reddit_raw.json"

    def mock_live_failure(*args, **kwargs):
        return {
            "status": "error",
            "source": "reddit_live_api",
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "errors": ["OAuth authentication failed (HTTP 401)"],
        }

    def fail_mock(*args, **kwargs):
        pytest.fail("Mock ingestion was called as fallback after live failure!")

    monkeypatch.setattr("app.ingest_reddit_live", mock_live_failure)
    monkeypatch.setattr("app.ingest_reddit_posts", fail_mock)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["record_count"] == 0
    assert stage_entry["error_category"] == "ingestion_error"
    assert path is None
    assert not canonical_file.exists()


def test_reddit_ingestion_skip_invokes_neither_function(monkeypatch):
    """Verify explicit skip invokes neither ingestion function nor accesses config."""
    def fail_call(*args, **kwargs):
        pytest.fail("Ingestion or config function unexpectedly called during skipped stage!")

    monkeypatch.setattr("app.ingest_reddit_posts", fail_call)
    monkeypatch.setattr("app.ingest_reddit_live", fail_call)
    monkeypatch.setattr("app.load_config", fail_call)

    stage_entry, path = execute_reddit_ingestion_stage(skip=True)
    assert stage_entry["status"] == "skipped"
    assert stage_entry["record_count"] == 0
    assert stage_entry["duration_seconds"] == 0.0
    assert stage_entry["reason"] == "stage_skipped_by_user"
    assert stage_entry["error_category"] is None
    assert path is None


def test_reddit_ingestion_missing_output_file(tmp_path, monkeypatch):
    """Verify validation fails if ingestion reports success but no file was written."""
    canonical_file = tmp_path / "reddit_raw.json"

    monkeypatch.setattr("app.ingest_reddit_posts", lambda *a, **kw: {
        "status": "success",
        "valid_records": 2,
    })

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_output"
    assert path is None


def test_reddit_ingestion_malformed_json(tmp_path, monkeypatch):
    """Verify validation fails if output file contains malformed JSON."""
    canonical_file = tmp_path / "reddit_raw.json"

    def write_corrupt(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("{not_valid_json", encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_reddit_posts", write_corrupt)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "corrupt_json"
    assert path is None


def test_reddit_ingestion_wrong_json_root_type(tmp_path, monkeypatch):
    """Verify validation fails if JSON root is a dictionary instead of a list."""
    canonical_file = tmp_path / "reddit_raw.json"

    def write_dict_root(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps({"posts": [{"post_id": "1"}]}), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_reddit_posts", write_dict_root)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_reddit_ingestion_non_dictionary_records(tmp_path, monkeypatch):
    """Verify validation fails if records inside the JSON list are not dictionaries."""
    canonical_file = tmp_path / "reddit_raw.json"

    def write_string_records(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(["string1", "string2"]), encoding="utf-8")
        return {"status": "success", "valid_records": 2}

    monkeypatch.setattr("app.ingest_reddit_posts", write_string_records)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


@pytest.mark.parametrize("invalid_post_id", ["", "   ", None, 12345])
def test_reddit_ingestion_missing_or_invalid_post_id(tmp_path, monkeypatch, invalid_post_id):
    """Verify validation fails if any record has missing, empty, or whitespace post_id."""
    canonical_file = tmp_path / "reddit_raw.json"

    bad_record = {"subreddit": "datascience", "title": "test"}
    if invalid_post_id is not None:
        bad_record["post_id"] = invalid_post_id

    def write_bad_id(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([bad_record]), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_reddit_posts", write_bad_id)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "invalid_record_id"
    assert path is None


def test_reddit_ingestion_duplicate_ids(tmp_path, monkeypatch):
    """Verify validation fails if duplicate post_ids are detected in output."""
    canonical_file = tmp_path / "reddit_raw.json"

    records = [
        {"post_id": "t3_dup_1", "title": "Post 1"},
        {"post_id": "t3_dup_1", "title": "Post 1 Duplicate"},
    ]

    def write_duplicates(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 2}

    monkeypatch.setattr("app.ingest_reddit_posts", write_duplicates)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "duplicate_record_id"
    assert path is None


@pytest.mark.parametrize("empty_content", ["", "[]", "   "])
def test_reddit_ingestion_empty_output(tmp_path, monkeypatch, empty_content):
    """Verify validation fails if output file is empty or contains zero records."""
    canonical_file = tmp_path / "reddit_raw.json"

    def write_empty(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(empty_content, encoding="utf-8")
        return {"status": "success", "valid_records": 0}

    monkeypatch.setattr("app.ingest_reddit_posts", write_empty)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] in ("empty_output", "corrupt_json")
    assert path is None


def test_reddit_ingestion_count_mismatch(tmp_path, monkeypatch):
    """Verify validation fails if serialized record count does not agree with returned valid_records."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(3)

    def write_mismatch(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        # Says 5 valid records, but wrote 3
        return {"status": "success", "valid_records": 5}

    monkeypatch.setattr("app.ingest_reddit_posts", write_mismatch)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


@pytest.mark.parametrize(
    "invalid_valid_records",
    [
        True,
        False,
        1.0,
        "1",
        -1,
        -10,
        float("nan"),
        float("inf"),
        None,
        [],
        {},
    ],
)
def test_reddit_ingestion_rejects_invalid_valid_records_types(tmp_path, monkeypatch, invalid_valid_records):
    """Verify validation strictly rejects boolean, float, string, negative, and non-int valid_records values."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(1)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": invalid_valid_records}

    monkeypatch.setattr("app.ingest_reddit_posts", write_output)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None
    # Verify the invalid value is never leaked into the diagnostic
    assert str(invalid_valid_records) not in (stage_entry.get("reason") or "")


def test_reddit_ingestion_missing_valid_records_fails(tmp_path, monkeypatch):
    """Verify validation fails if summary lacks the valid_records key."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(1)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success"}  # missing valid_records

    monkeypatch.setattr("app.ingest_reddit_posts", write_output)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


def test_reddit_ingestion_valid_records_true_with_one_record_fails(tmp_path, monkeypatch):
    """Verify True is strictly rejected even when serialized count is 1 (1 == True)."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(1)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": True}

    monkeypatch.setattr("app.ingest_reddit_posts", write_output)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None
    assert "True" not in (stage_entry.get("reason") or "")


def test_reddit_ingestion_matching_integer_valid_records_passes(tmp_path, monkeypatch):
    """Verify valid non-negative integer matching serialized count passes validation cleanly."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(3)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 3}

    monkeypatch.setattr("app.ingest_reddit_posts", write_output)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 3
    assert path == canonical_file
    assert canonical_file.exists()


def test_validate_reddit_ingestion_output_direct_defensive_checks(tmp_path):
    """Directly test validate_reddit_ingestion_output against invalid valid_records types."""
    test_file = tmp_path / "temp.json"
    test_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")

    # True with 1 record
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success", "valid_records": True}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"
    assert "True" not in (reason or "")

    # False
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success", "valid_records": False}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # 1.0 float
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success", "valid_records": 1.0}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # "1" string
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success", "valid_records": "1"}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # -1 negative int
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success", "valid_records": -1}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # missing valid_records
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success"}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # matching non-negative int passes
    is_valid, count, cat, reason = validate_reddit_ingestion_output(
        {"status": "success", "valid_records": 1}, test_file
    )
    assert is_valid is True
    assert count == 1
    assert cat is None
    assert reason is None


@pytest.mark.parametrize("bad_status", ["error", "disabled", "no_records", "no_valid_records", "unknown"])
def test_reddit_ingestion_invalid_status(tmp_path, monkeypatch, bad_status):
    """Verify non-success ingestion statuses produce a failed stage."""
    canonical_file = tmp_path / "reddit_raw.json"

    monkeypatch.setattr("app.ingest_reddit_posts", lambda *a, **kw: {
        "status": bad_status,
        "valid_records": 0,
        "errors": ["Something went wrong"],
    })

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_reddit_ingestion_failed_validation_preserves_canonical_output(tmp_path, monkeypatch):
    """Verify failed validation never overwrites or deletes existing canonical output."""
    canonical_file = tmp_path / "reddit_raw.json"
    original_data = [{"post_id": "t3_existing", "title": "Existing Previous Run Post"}]
    canonical_file.write_text(json.dumps(original_data), encoding="utf-8")

    def write_failing(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("invalid json corrupt", encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_reddit_posts", write_failing)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert path is None

    # Canonical file must remain completely intact with original content
    assert canonical_file.exists()
    assert json.loads(canonical_file.read_text(encoding="utf-8")) == original_data


def test_reddit_ingestion_successful_atomic_promotion(tmp_path, monkeypatch):
    """Verify newly validated output is atomically promoted to canonical destination."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(4)

    def write_valid(*args, **kwargs):
        out_p = Path(kwargs["output_path"])
        out_p.write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 4}

    monkeypatch.setattr("app.ingest_reddit_posts", write_valid)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 4
    assert path == canonical_file
    assert canonical_file.exists()

    loaded = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(loaded) == 4
    assert loaded[0]["post_id"] == "t3_sample_1"


def test_reddit_ingestion_temporary_file_cleanup_on_failure(tmp_path, monkeypatch):
    """Verify temporary files are cleaned up after validation or execution failure."""
    target_dir = tmp_path / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = target_dir / "reddit_raw.json"

    def write_corrupt_temp(*args, **kwargs):
        out_p = Path(kwargs["output_path"])
        out_p.write_text("bad data", encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_reddit_posts", write_corrupt_temp)

    stage_entry, path = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"

    # Verify no temporary files remain in the target directory
    temp_files = list(target_dir.glob(".tmp_reddit_*"))
    assert temp_files == []


def test_reddit_ingestion_diagnostics_contain_no_secrets_or_absolute_paths(tmp_path, monkeypatch):
    """Verify failure diagnostics never leak credentials, secret tokens, absolute paths, or raw exceptions."""
    canonical_file = tmp_path / "nested" / "dir" / "reddit_raw.json"

    def mock_secret_leak(*args, **kwargs):
        return {
            "status": "error",
            "errors": [
                "Secret token bearer_xyz123 failed for path C:\\Users\\secret\\path\\file.json"
            ],
        }

    monkeypatch.setattr("app.ingest_reddit_posts", mock_secret_leak)

    stage_entry, _ = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"

    diagnostic = stage_entry.get("reason") or ""
    category = stage_entry.get("error_category") or ""

    assert "bearer_xyz123" not in diagnostic
    assert "C:\\" not in diagnostic
    assert "C:/" not in diagnostic
    assert "Users" not in diagnostic
    assert "\\" not in diagnostic
    assert "/" not in diagnostic
    assert "secret" not in diagnostic.lower()
    assert category == "ingestion_error"


def test_reddit_ingestion_stage_entry_structure_and_types(tmp_path, monkeypatch):
    """Verify stage manifest entry strictly satisfies Block 1 schema invariants."""
    canonical_file = tmp_path / "reddit_raw.json"
    records = _sample_reddit_records(2)

    def write_valid(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 2}

    monkeypatch.setattr("app.ingest_reddit_posts", write_valid)

    stage_entry, _ = execute_reddit_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert isinstance(stage_entry["record_count"], int)
    assert stage_entry["record_count"] == 2
    assert isinstance(stage_entry["duration_seconds"], (int, float))
    assert stage_entry["duration_seconds"] >= 0.0
    assert stage_entry["reason"] is None
    assert stage_entry["error_category"] is None




# STEP 7.5C — Block 3B: YouTube Ingestion Execution & Validation Tests
# =====================================================================

def _sample_youtube_records(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "video_id": f"vid_test_{i}",
            "title": f"Test Video Title {i}",
            "description": f"Test Video Description {i}",
            "channel_id": f"UC_channel_{i}",
            "channel_title": f"Channel Title {i}",
            "published_at": "2026-11-01T12:00:00Z",
            "view_count": 1000 * i,
            "like_count": 50 * i,
            "comment_count": 10 * i,
            "url": f"https://www.youtube.com/watch?v=vid_test_{i}",
            "collected_at": "2026-11-07T12:00:00Z",
        }
        for i in range(1, count + 1)
    ]


def test_youtube_ingestion_selects_live_when_configured(tmp_path, monkeypatch):
    """Verify live ingestion is executed when YouTube API key is configured."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_api_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(3)

    def mock_ingest_live(*args, **kwargs):
        assert kwargs.get("merge") is False
        out_p = Path(kwargs["output_path"])
        out_p.write_text(json.dumps(records), encoding="utf-8")
        return {
            "status": "success",
            "source": "youtube_api_v3",
            "total_records": 3,
            "valid_records": 3,
            "rejected_records": 0,
            "output_file": str(out_p),
            "errors": [],
        }

    monkeypatch.setattr("app.ingest_youtube_live", mock_ingest_live)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 3
    assert stage_entry["reason"] is None
    assert stage_entry["error_category"] is None
    assert path == canonical_file
    assert canonical_file.exists()


def test_youtube_ingestion_skipped_when_api_key_absent(tmp_path, monkeypatch):
    """Verify YouTube ingestion is marked skipped when API key is missing, without calling ingestion."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    canonical_file = tmp_path / "youtube_raw.json"

    def fail_call(*args, **kwargs):
        pytest.fail("YouTube ingestion called when API key was absent!")

    monkeypatch.setattr("app.ingest_youtube_live", fail_call)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "skipped"
    assert stage_entry["record_count"] == 0
    assert stage_entry["reason"] == "youtube_api_key_not_configured"
    assert stage_entry["error_category"] is None
    assert path is None
    assert not canonical_file.exists()


def test_youtube_ingestion_skip_invokes_neither_config_nor_ingestion(monkeypatch):
    """Verify explicit skip invokes neither configuration loading nor live ingestion."""
    def fail_call(*args, **kwargs):
        pytest.fail("Ingestion or config function unexpectedly called during skipped stage!")

    monkeypatch.setattr("app.ingest_youtube_live", fail_call)
    monkeypatch.setattr("app.load_config", fail_call)

    stage_entry, path = execute_youtube_ingestion_stage(skip=True)
    assert stage_entry["status"] == "skipped"
    assert stage_entry["record_count"] == 0
    assert stage_entry["duration_seconds"] == 0.0
    assert stage_entry["reason"] == "stage_skipped_by_user"
    assert stage_entry["error_category"] is None
    assert path is None


def test_youtube_ingestion_executes_exactly_one_call(tmp_path, monkeypatch):
    """Verify live ingestion is called exactly once when configured."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    call_count = 0

    def mock_ingest(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        out_p = Path(kwargs["output_path"])
        out_p.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", mock_ingest)

    execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert call_count == 1


def test_youtube_ingestion_live_failure_does_not_fall_back_to_mock(tmp_path, monkeypatch):
    """Verify live YouTube failure records stage failure and never falls back to mock data."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "configured_key")
    canonical_file = tmp_path / "youtube_raw.json"

    def mock_live_failure(*args, **kwargs):
        return {
            "status": "error",
            "source": "youtube_api_v3",
            "total_records": 0,
            "valid_records": 0,
            "rejected_records": 0,
            "output_file": None,
            "errors": ["YouTube API quota exceeded (HTTP 403)."],
        }

    monkeypatch.setattr("app.ingest_youtube_live", mock_live_failure)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["record_count"] == 0
    assert stage_entry["error_category"] == "ingestion_error"
    assert path is None
    assert not canonical_file.exists()


def test_youtube_ingestion_missing_output_file(tmp_path, monkeypatch):
    """Verify validation fails if YouTube ingestion reports success but no file was written."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    monkeypatch.setattr("app.ingest_youtube_live", lambda *a, **kw: {
        "status": "success",
        "valid_records": 2,
    })

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_output"
    assert path is None


def test_youtube_ingestion_malformed_json(tmp_path, monkeypatch):
    """Verify validation fails if output file contains malformed JSON."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    def write_corrupt(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("{corrupt_json_not_valid", encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", write_corrupt)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "corrupt_json"
    assert path is None


def test_youtube_ingestion_wrong_json_root_type(tmp_path, monkeypatch):
    """Verify validation fails if JSON root is a dictionary instead of a list."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    def write_dict_root(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps({"videos": [{"video_id": "1"}]}), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", write_dict_root)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_youtube_ingestion_non_dictionary_records(tmp_path, monkeypatch):
    """Verify validation fails if records inside the JSON list are not dictionaries."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    def write_string_records(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(["string1", "string2"]), encoding="utf-8")
        return {"status": "success", "valid_records": 2}

    monkeypatch.setattr("app.ingest_youtube_live", write_string_records)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


@pytest.mark.parametrize("invalid_video_id", ["", "   ", None, 12345])
def test_youtube_ingestion_missing_or_invalid_video_id(tmp_path, monkeypatch, invalid_video_id):
    """Verify validation fails if any record has missing, empty, or whitespace video_id."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    bad_record = _sample_youtube_records(1)[0]
    if invalid_video_id is not None:
        bad_record["video_id"] = invalid_video_id
    else:
        del bad_record["video_id"]

    def write_bad_id(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([bad_record]), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", write_bad_id)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "invalid_record_id"
    assert path is None


def test_youtube_ingestion_missing_required_schema_fields(tmp_path, monkeypatch):
    """Verify validation fails if record is missing other required YouTube fields (e.g. channel_id)."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    bad_record = _sample_youtube_records(1)[0]
    del bad_record["channel_id"]  # Required schema field

    def write_missing_field(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([bad_record]), encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", write_missing_field)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_youtube_ingestion_duplicate_ids(tmp_path, monkeypatch):
    """Verify validation fails if duplicate video_ids are detected in output."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    records = _sample_youtube_records(2)
    records[1]["video_id"] = records[0]["video_id"]  # Duplicate ID

    def write_duplicates(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 2}

    monkeypatch.setattr("app.ingest_youtube_live", write_duplicates)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "duplicate_record_id"
    assert path is None


@pytest.mark.parametrize("empty_content", ["", "[]", "   "])
def test_youtube_ingestion_empty_output(tmp_path, monkeypatch, empty_content):
    """Verify validation fails if output file is empty or contains zero records."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    def write_empty(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(empty_content, encoding="utf-8")
        return {"status": "success", "valid_records": 0}

    monkeypatch.setattr("app.ingest_youtube_live", write_empty)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] in ("empty_output", "corrupt_json")
    assert path is None


def test_youtube_ingestion_count_mismatch(tmp_path, monkeypatch):
    """Verify validation fails if serialized record count does not agree with returned valid_records."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(3)

    def write_mismatch(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 5}

    monkeypatch.setattr("app.ingest_youtube_live", write_mismatch)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


@pytest.mark.parametrize(
    "invalid_valid_records",
    [
        True,
        False,
        1.0,
        "1",
        -1,
        -10,
        float("nan"),
        float("inf"),
        None,
        [],
        {},
    ],
)
def test_youtube_ingestion_rejects_invalid_valid_records_types(tmp_path, monkeypatch, invalid_valid_records):
    """Verify validation strictly rejects boolean, float, string, negative, and non-int valid_records values."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(1)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": invalid_valid_records}

    monkeypatch.setattr("app.ingest_youtube_live", write_output)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None
    # Verify the invalid value is never leaked into the diagnostic
    assert str(invalid_valid_records) not in (stage_entry.get("reason") or "")


def test_youtube_ingestion_missing_valid_records_fails(tmp_path, monkeypatch):
    """Verify validation fails if summary lacks the valid_records key."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(1)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.ingest_youtube_live", write_output)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


def test_youtube_ingestion_valid_records_true_with_one_record_fails(tmp_path, monkeypatch):
    """Verify True is strictly rejected even when serialized count is 1 (1 == True)."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(1)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": True}

    monkeypatch.setattr("app.ingest_youtube_live", write_output)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None
    assert "True" not in (stage_entry.get("reason") or "")


def test_youtube_ingestion_matching_integer_valid_records_passes(tmp_path, monkeypatch):
    """Verify valid non-negative integer matching serialized count passes validation cleanly."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(3)

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 3}

    monkeypatch.setattr("app.ingest_youtube_live", write_output)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 3
    assert path == canonical_file
    assert canonical_file.exists()


def test_validate_youtube_ingestion_output_direct_defensive_checks(tmp_path):
    """Directly test validate_youtube_ingestion_output against invalid valid_records types."""
    test_file = tmp_path / "temp_yt.json"
    test_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")

    # True with 1 record
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success", "valid_records": True}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"
    assert "True" not in (reason or "")

    # False
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success", "valid_records": False}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # 1.0 float
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success", "valid_records": 1.0}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # "1" string
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success", "valid_records": "1"}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # -1 negative int
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success", "valid_records": -1}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # missing valid_records
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success"}, test_file
    )
    assert is_valid is False
    assert cat == "count_mismatch"

    # matching non-negative int passes
    is_valid, count, cat, reason = validate_youtube_ingestion_output(
        {"status": "success", "valid_records": 1}, test_file
    )
    assert is_valid is True
    assert count == 1
    assert cat is None
    assert reason is None


@pytest.mark.parametrize("bad_status", ["error", "disabled", "no_records", "unknown"])
def test_youtube_ingestion_invalid_status(tmp_path, monkeypatch, bad_status):
    """Verify non-success YouTube ingestion statuses produce a failed stage."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"

    monkeypatch.setattr("app.ingest_youtube_live", lambda *a, **kw: {
        "status": bad_status,
        "valid_records": 0,
        "errors": ["Something went wrong"],
    })

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_youtube_ingestion_failed_validation_preserves_canonical_output(tmp_path, monkeypatch):
    """Verify failed validation never overwrites or deletes existing canonical YouTube output."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    original_data = [{"video_id": "vid_existing", "title": "Existing Previous Run Video"}]
    canonical_file.write_text(json.dumps(original_data), encoding="utf-8")

    def write_failing(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("invalid json corrupt", encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", write_failing)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"
    assert path is None

    # Canonical file must remain completely intact with original content
    assert canonical_file.exists()
    assert json.loads(canonical_file.read_text(encoding="utf-8")) == original_data


def test_youtube_ingestion_successful_atomic_promotion(tmp_path, monkeypatch):
    """Verify newly validated YouTube output is atomically promoted to canonical destination."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(4)

    def write_valid(*args, **kwargs):
        out_p = Path(kwargs["output_path"])
        out_p.write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 4}

    monkeypatch.setattr("app.ingest_youtube_live", write_valid)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 4
    assert path == canonical_file
    assert canonical_file.exists()

    loaded = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(loaded) == 4
    assert loaded[0]["video_id"] == "vid_test_1"


def test_youtube_ingestion_temporary_file_cleanup_on_failure(tmp_path, monkeypatch):
    """Verify temporary files are cleaned up after validation or execution failure."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    target_dir = tmp_path / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = target_dir / "youtube_raw.json"

    def write_corrupt_temp(*args, **kwargs):
        out_p = Path(kwargs["output_path"])
        out_p.write_text("bad data", encoding="utf-8")
        return {"status": "success", "valid_records": 1}

    monkeypatch.setattr("app.ingest_youtube_live", write_corrupt_temp)

    stage_entry, path = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"

    # Verify no temporary files remain in the target directory
    temp_files = list(target_dir.glob(".tmp_youtube_*"))
    assert temp_files == []


def test_youtube_ingestion_diagnostics_contain_no_secrets_or_absolute_paths(tmp_path, monkeypatch):
    """Verify failure diagnostics never leak API keys, secret tokens, absolute paths, or raw exceptions."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "secret_yt_key_12345")
    canonical_file = tmp_path / "nested" / "dir" / "youtube_raw.json"

    def mock_secret_leak(*args, **kwargs):
        return {
            "status": "error",
            "errors": [
                "Secret API key secret_yt_key_12345 failed for path C:\\Users\\secret\\path\\file.json"
            ],
        }

    monkeypatch.setattr("app.ingest_youtube_live", mock_secret_leak)

    stage_entry, _ = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "failed"

    diagnostic = stage_entry.get("reason") or ""
    category = stage_entry.get("error_category") or ""

    assert "secret_yt_key_12345" not in diagnostic
    assert "C:\\" not in diagnostic
    assert "C:/" not in diagnostic
    assert "Users" not in diagnostic
    assert "\\" not in diagnostic
    assert "/" not in diagnostic
    assert "secret" not in diagnostic.lower()
    assert category == "ingestion_error"


def test_youtube_ingestion_stage_entry_structure_and_types(tmp_path, monkeypatch):
    """Verify stage manifest entry strictly satisfies Block 1 schema invariants."""
    monkeypatch.setenv("YOUTUBE_API_KEY", "valid_key")
    canonical_file = tmp_path / "youtube_raw.json"
    records = _sample_youtube_records(2)

    def write_valid(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "valid_records": 2}

    monkeypatch.setattr("app.ingest_youtube_live", write_valid)

    stage_entry, _ = execute_youtube_ingestion_stage(canonical_output_path=canonical_file)
    assert stage_entry["status"] == "success"
    assert isinstance(stage_entry["record_count"], int)
    assert stage_entry["record_count"] == 2
    assert isinstance(stage_entry["duration_seconds"], (int, float))
    assert stage_entry["duration_seconds"] >= 0.0
    assert stage_entry["reason"] is None
    assert stage_entry["error_category"] is None




# STEP 7.5C — Block 4A: Reddit Preprocessing Execution & Validation Tests
# =====================================================================

def _sample_processed_reddit_records(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "post_id": f"t3_prep_{i}",
            "subreddit": "datascience",
            "title": f"Raw Title {i}",
            "text": f"Raw Text {i}",
            "title_cleaned": f"Cleaned Title {i}",
            "text_cleaned": f"Cleaned Text {i}",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-07T14:00:00Z",
            "author": f"author_{i}",
            "created_utc": 1730000000 + i * 100,
            "score": 100 + i,
            "num_comments": i,
            "url": f"https://reddit.com/r/datascience/prep_{i}",
            "collected_at": "2026-11-07T12:00:00Z",
        }
        for i in range(1, count + 1)
    ]


def test_reddit_preprocessing_skipped_when_ingestion_skipped():
    """Verify preprocessing is skipped when ingestion stage was skipped."""
    ingest_entry = {"status": "skipped", "reason": "stage_skipped_by_user", "record_count": 0}

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=Path("dummy_raw.json"),
        ingestion_stage_entry=ingest_entry,
    )
    assert stage_entry["status"] == "skipped"
    assert stage_entry["reason"] == "reddit_ingestion_skipped"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_reddit_preprocessing_skipped_when_ingestion_failed():
    """Verify preprocessing is skipped when ingestion stage failed."""
    ingest_entry = {"status": "failed", "reason": "ingestion_error", "record_count": 0}

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=Path("dummy_raw.json"),
        ingestion_stage_entry=ingest_entry,
    )
    assert stage_entry["status"] == "skipped"
    assert stage_entry["reason"] == "reddit_ingestion_failed"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_reddit_preprocessing_skipped_when_explicit_skip():
    """Verify preprocessing is skipped when skip=True."""
    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=Path("dummy_raw.json"),
        skip=True,
    )
    assert stage_entry["status"] == "skipped"
    assert stage_entry["reason"] == "stage_skipped_by_user"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_reddit_preprocessing_fails_when_raw_input_path_none():
    """Verify preprocessing fails safely without executing pipeline when raw_input_path is None."""
    stage_entry, path = execute_reddit_preprocessing_stage(raw_input_path=None)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_raw_input"
    assert path is None


def test_reddit_preprocessing_fails_when_raw_input_file_missing(tmp_path):
    """Verify preprocessing fails safely when raw input file does not exist."""
    missing_raw = tmp_path / "missing_reddit_raw.json"
    stage_entry, path = execute_reddit_preprocessing_stage(raw_input_path=missing_raw)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_raw_input"
    assert path is None


def test_reddit_preprocessing_never_silently_substitutes_default_raw(monkeypatch):
    """Verify pipeline function is not called when raw_input_path is None or missing."""
    def fail_call(*args, **kwargs):
        pytest.fail("preprocess_reddit_dataset was called despite missing raw_input_path!")

    monkeypatch.setattr("app.preprocess_reddit_dataset", fail_call)

    stage_entry, _ = execute_reddit_preprocessing_stage(raw_input_path=None)
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_raw_input"


def test_reddit_preprocessing_valid_invocation(tmp_path, monkeypatch):
    """Verify valid current-run raw input invokes preprocessing and promotes cleanly."""
    raw_file = tmp_path / "current_raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(3)), encoding="utf-8")

    canonical_file = tmp_path / "reddit_processed.json"
    processed_records = _sample_processed_reddit_records(3)

    called_kwargs = {}

    def mock_preprocess(*args, **kwargs):
        called_kwargs.update(kwargs)
        assert kwargs.get("merge") is False
        assert kwargs.get("input_path") == raw_file
        out_p = Path(kwargs["output_path"])
        out_p.write_text(json.dumps(processed_records), encoding="utf-8")
        return {
            "status": "success",
            "source": "reddit",
            "total_records": 3,
            "processed_records": 3,
            "failed_records": 0,
            "output_file": str(out_p),
            "errors": [],
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", mock_preprocess)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
    )
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 3
    assert path == canonical_file
    assert canonical_file.exists()
    assert len(json.loads(canonical_file.read_text(encoding="utf-8"))) == 3


def test_reddit_preprocessing_executes_exactly_once(tmp_path, monkeypatch):
    """Verify preprocess_reddit_dataset is called exactly once per stage execution."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    call_count = 0

    def mock_preprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 1,
            "processed_records": 1,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", mock_preprocess)

    execute_reddit_preprocessing_stage(raw_input_path=raw_file, canonical_output_path=canonical_file)
    assert call_count == 1


def test_reddit_preprocessing_pipeline_failure(tmp_path, monkeypatch):
    """Verify pipeline failure status produces a failed stage entry."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    monkeypatch.setattr(
        "app.preprocess_reddit_dataset",
        lambda *a, **kw: {
            "status": "failed",
            "total_records": 1,
            "processed_records": 0,
            "failed_records": 1,
            "errors": ["Spark NLP initialization error"],
        },
    )

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "preprocessing_error"
    assert path is None
    assert not canonical_file.exists()


def test_reddit_preprocessing_missing_output_file(tmp_path, monkeypatch):
    """Verify validation fails when preprocessing reports success but output file was not written."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    monkeypatch.setattr(
        "app.preprocess_reddit_dataset",
        lambda *a, **kw: {
            "status": "success",
            "total_records": 1,
            "processed_records": 1,
            "failed_records": 0,
        },
    )

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_output"
    assert path is None


def test_reddit_preprocessing_empty_output_file(tmp_path, monkeypatch):
    """Verify validation fails when output file is empty."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_empty(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return {"status": "success", "total_records": 0, "processed_records": 0, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_empty)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] in ("empty_output", "corrupt_json")
    assert path is None


def test_reddit_preprocessing_malformed_json(tmp_path, monkeypatch):
    """Verify validation fails when output file contains unparseable JSON."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_corrupt(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("{invalid_json_corrupted", encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_corrupt)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "corrupt_json"
    assert path is None


def test_reddit_preprocessing_wrong_root_type(tmp_path, monkeypatch):
    """Verify validation fails when JSON root is not a list."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_dict_root(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps({"posts": []}), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_dict_root)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_reddit_preprocessing_non_dictionary_record(tmp_path, monkeypatch):
    """Verify validation fails when records inside JSON list are not dictionaries."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_non_dict(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(["string_record"]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_non_dict)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


@pytest.mark.parametrize("bad_post_id", ["", "   ", None, 999])
def test_reddit_preprocessing_missing_or_invalid_post_id(tmp_path, monkeypatch, bad_post_id):
    """Verify validation fails when a record has missing, empty, or non-string post_id."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    bad_rec = _sample_processed_reddit_records(1)[0]
    if bad_post_id is not None:
        bad_rec["post_id"] = bad_post_id
    else:
        del bad_rec["post_id"]

    def write_bad_id(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([bad_rec]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_bad_id)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "invalid_record_id"
    assert path is None


def test_reddit_preprocessing_duplicate_post_ids(tmp_path, monkeypatch):
    """Verify validation fails when duplicate post_ids appear in output."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    records = _sample_processed_reddit_records(2)
    records[1]["post_id"] = records[0]["post_id"]  # Duplicate

    def write_duplicates(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {"status": "success", "total_records": 2, "processed_records": 2, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_duplicates)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "duplicate_record_id"
    assert path is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "title_cleaned",
        "text_cleaned",
        "detected_language",
        "language_code",
        "preprocessing_status",
        "processed_at",
    ],
)
def test_reddit_preprocessing_missing_enrichment_field(tmp_path, monkeypatch, missing_field):
    """Verify validation fails when any required preprocessing enrichment field is absent."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    rec = _sample_processed_reddit_records(1)[0]
    del rec[missing_field]

    def write_missing(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([rec]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_missing)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


@pytest.mark.parametrize(
    "invalid_count_val",
    [
        True,
        False,
        1.0,
        "1",
        -1,
        None,
    ],
)
def test_reddit_preprocessing_rejects_invalid_count_types(tmp_path, monkeypatch, invalid_count_val):
    """Verify validation strictly rejects boolean, float, string, and negative counts."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_output(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": invalid_count_val,
            "processed_records": 1,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_output)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None
    assert str(invalid_count_val) not in (stage_entry.get("reason") or "")


def test_reddit_preprocessing_count_sum_mismatch(tmp_path, monkeypatch):
    """Verify validation fails if total_records != processed_records + failed_records."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_mismatch(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8"
        )
        # 10 != 2 + 0
        return {
            "status": "success",
            "total_records": 10,
            "processed_records": 2,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_mismatch)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


def test_reddit_preprocessing_serialized_count_mismatch(tmp_path, monkeypatch):
    """Verify validation fails if serialized records count != processed_records."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(3)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_mismatch(*args, **kwargs):
        # Writes 2 records, but says processed_records=3
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 3,
            "processed_records": 3,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_mismatch)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


def test_reddit_preprocessing_partially_rejected_records_pass_when_counts_consistent(tmp_path, monkeypatch):
    """Verify pipeline contract allows partially rejected records when counts are consistent."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(3)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    # 3 total, 2 processed, 1 failed. Output contains 2 processed records.
    records = _sample_processed_reddit_records(2)

    def write_partial(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(records), encoding="utf-8")
        return {
            "status": "success",
            "total_records": 3,
            "processed_records": 2,
            "failed_records": 1,
            "errors": ["Rejected record t3_bad: Missing required field"],
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_partial)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()


def test_reddit_preprocessing_failed_validation_preserves_canonical_output(tmp_path, monkeypatch):
    """Verify failed validation never overwrites or deletes existing canonical processed output."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "reddit_processed.json"
    original_data = [{"post_id": "t3_existing", "title_cleaned": "Existing Cleaned"}]
    canonical_file.write_text(json.dumps(original_data), encoding="utf-8")

    def write_corrupt(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("corrupted json", encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_corrupt)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"
    assert path is None

    assert canonical_file.exists()
    assert json.loads(canonical_file.read_text(encoding="utf-8")) == original_data


def test_reddit_preprocessing_temporary_file_cleanup_on_failure(tmp_path, monkeypatch):
    """Verify temporary files are cleaned up after validation failure."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    target_dir = tmp_path / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = target_dir / "reddit_processed.json"

    def write_bad(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("bad data", encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_bad)

    stage_entry, path = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"

    temp_files = list(target_dir.glob(".tmp_reddit_processed_*"))
    assert temp_files == []


def test_reddit_preprocessing_diagnostics_contain_no_secrets_or_absolute_paths(tmp_path, monkeypatch):
    """Verify diagnostics never expose raw content, credentials, absolute paths, or stack traces."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "dir" / "processed.json"

    def mock_secret_error(*args, **kwargs):
        return {
            "status": "error",
            "errors": [
                "Secret token bearer_abc999 failed for path C:\\Users\\secret\\path\\file.json"
            ],
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", mock_secret_error)

    stage_entry, _ = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "failed"

    diagnostic = stage_entry.get("reason") or ""
    category = stage_entry.get("error_category") or ""

    assert "bearer_abc999" not in diagnostic
    assert "C:\\" not in diagnostic
    assert "C:/" not in diagnostic
    assert "Users" not in diagnostic
    assert "\\" not in diagnostic
    assert "/" not in diagnostic
    assert "secret" not in diagnostic.lower()
    assert category == "preprocessing_error"


def test_reddit_preprocessing_stage_entry_structure_and_types(tmp_path, monkeypatch):
    """Verify stage manifest entry strictly satisfies Block 1 schema invariants."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_reddit_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_valid(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 2,
            "processed_records": 2,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_reddit_dataset", write_valid)

    stage_entry, _ = execute_reddit_preprocessing_stage(
        raw_input_path=raw_file, canonical_output_path=canonical_file
    )
    assert stage_entry["status"] == "success"
    assert isinstance(stage_entry["record_count"], int)
    assert stage_entry["record_count"] == 2
    assert isinstance(stage_entry["duration_seconds"], (int, float))
    assert stage_entry["duration_seconds"] >= 0.0
    assert stage_entry["reason"] is None
    assert stage_entry["error_category"] is None




# STEP 7.5C — Block 4B: YouTube Preprocessing Execution & Validation Tests
# =====================================================================

def _sample_processed_youtube_records(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "video_id": f"vid_prep_{i}",
            "title": f"Raw Title {i}",
            "description": f"Raw Description {i}",
            "title_cleaned": f"Cleaned Title {i}",
            "description_cleaned": f"Cleaned Description {i}",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": "2026-11-07T14:00:00Z",
            "channel_id": f"UC_channel_{i}",
            "channel_title": f"Channel {i}",
            "published_at": "2026-11-07T10:00:00Z",
            "view_count": 1000 + i * 50,
            "like_count": 50 + i,
            "comment_count": 5 + i,
            "url": f"https://youtube.com/watch?v=vid_prep_{i}",
            "collected_at": "2026-11-07T12:00:00Z",
        }
        for i in range(1, count + 1)
    ]


def _valid_youtube_ingest_entry(record_count: int = 3) -> dict[str, Any]:
    return {
        "status": "success",
        "record_count": record_count,
        "duration_seconds": 1.23,
        "reason": None,
        "error_category": None,
    }


def test_youtube_preprocessing_skipped_when_ingestion_skipped():
    """Verify preprocessing is skipped when YouTube ingestion stage was skipped."""
    ingest_entry = {"status": "skipped", "reason": "missing_api_key", "record_count": 0}

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=Path("dummy_youtube_raw.json"),
        ingestion_stage_entry=ingest_entry,
    )
    assert stage_entry["status"] == "skipped"
    assert stage_entry["reason"] == "youtube_ingestion_skipped"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_youtube_preprocessing_skipped_when_ingestion_failed():
    """Verify preprocessing is skipped when YouTube ingestion stage failed."""
    ingest_entry = {"status": "failed", "reason": "api_error", "record_count": 0}

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=Path("dummy_youtube_raw.json"),
        ingestion_stage_entry=ingest_entry,
    )
    assert stage_entry["status"] == "skipped"
    assert stage_entry["reason"] == "youtube_ingestion_failed"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_youtube_preprocessing_skipped_when_explicit_skip():
    """Verify preprocessing is skipped when skip=True."""
    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=Path("dummy_youtube_raw.json"),
        skip=True,
    )
    assert stage_entry["status"] == "skipped"
    assert stage_entry["reason"] == "stage_skipped_by_user"
    assert stage_entry["record_count"] == 0
    assert path is None


def test_youtube_preprocessing_fails_when_ingestion_stage_entry_none(tmp_path, monkeypatch):
    """Verify preprocessing fails closed when ingestion_stage_entry is None, despite existing raw file."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def fail_call(*args, **kwargs):
        pytest.fail("preprocess_youtube_dataset should not be called when provenance is missing!")

    monkeypatch.setattr("app.preprocess_youtube_dataset", fail_call)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=None,
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_upstream_provenance"
    assert stage_entry["reason"] == "Missing upstream ingestion provenance"
    assert path is None
    assert not canonical_file.exists()


@pytest.mark.parametrize("invalid_entry", ["not_a_dict", 123, []])
def test_youtube_preprocessing_fails_when_ingestion_entry_not_dict(tmp_path, monkeypatch, invalid_entry):
    """Verify preprocessing fails closed when ingestion_stage_entry is not a dictionary."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def fail_call(*args, **kwargs):
        pytest.fail("preprocess_youtube_dataset should not be called for invalid ingestion entry!")

    monkeypatch.setattr("app.preprocess_youtube_dataset", fail_call)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=invalid_entry,
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


@pytest.mark.parametrize("bad_status_entry", [
    {},
    {"status": None},
    {"status": "unknown"},
    {"status": "pending"},
    {"status": "partial"},
    {"status": "error"},
])
def test_youtube_preprocessing_fails_when_ingestion_status_invalid_or_unsupported(tmp_path, monkeypatch, bad_status_entry):
    """Verify preprocessing fails closed when ingestion status is missing or unsupported."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def fail_call(*args, **kwargs):
        pytest.fail("preprocess_youtube_dataset should not be called for invalid ingestion status!")

    monkeypatch.setattr("app.preprocess_youtube_dataset", fail_call)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=bad_status_entry,
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_youtube_preprocessing_fails_when_raw_input_path_none():
    """Verify preprocessing fails safely without executing pipeline when raw_input_path is None."""
    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=None,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_raw_input"
    assert stage_entry["reason"] == "Raw input path is missing or None"
    assert path is None


def test_youtube_preprocessing_fails_when_raw_input_file_missing(tmp_path):
    """Verify preprocessing fails safely when raw input file does not exist."""
    missing_raw = tmp_path / "missing_youtube_raw.json"
    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=missing_raw,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_raw_input"
    assert stage_entry["reason"] == "Raw input file does not exist"
    assert path is None


def test_youtube_preprocessing_fails_when_raw_input_file_invalid_or_unreadable(tmp_path):
    """Verify preprocessing fails safely when raw input file contains corrupt non-JSON text."""
    corrupt_raw = tmp_path / "corrupt_raw.json"
    corrupt_raw.write_text("{corrupt non-json content", encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=corrupt_raw,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "preprocessing_error"
    assert path is None
    assert not canonical_file.exists()


def test_youtube_preprocessing_never_silently_substitutes_default_raw(monkeypatch):
    """Verify pipeline function is not called when raw_input_path is None or missing."""
    def fail_call(*args, **kwargs):
        pytest.fail("preprocess_youtube_dataset was called despite missing raw_input_path!")

    monkeypatch.setattr("app.preprocess_youtube_dataset", fail_call)

    stage_entry, _ = execute_youtube_preprocessing_stage(
        raw_input_path=None,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_raw_input"


def test_youtube_preprocessing_does_not_modify_raw_input_file(tmp_path, monkeypatch):
    """Verify the raw input file bytes remain completely unchanged before and after preprocessing."""
    raw_file = tmp_path / "raw.json"
    raw_content = json.dumps(_sample_youtube_records(3), indent=2)
    raw_file.write_text(raw_content, encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_valid(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_youtube_records(3)), encoding="utf-8"
        )
        return {"status": "success", "total_records": 3, "processed_records": 3, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_valid)

    stage_entry, _ = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(3),
    )
    assert stage_entry["status"] == "success"
    assert raw_file.read_text(encoding="utf-8") == raw_content


def test_youtube_preprocessing_valid_invocation(tmp_path, monkeypatch):
    """Verify valid current-run raw input invokes preprocessing and promotes cleanly."""
    raw_file = tmp_path / "current_raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(3)), encoding="utf-8")
    canonical_file = tmp_path / "canonical_processed.json"

    def mock_preprocess(*args, **kwargs):
        assert kwargs.get("input_path") == raw_file
        assert kwargs.get("merge") is False
        assert kwargs.get("batch_size") == 50
        out_p = Path(kwargs["output_path"])
        assert out_p.name.startswith(".tmp_youtube_processed_")
        assert out_p.parent == canonical_file.parent
        out_p.write_text(json.dumps(_sample_processed_youtube_records(3)), encoding="utf-8")
        return {
            "status": "success",
            "source": "youtube",
            "total_records": 3,
            "processed_records": 3,
            "failed_records": 0,
            "output_file": str(out_p),
            "errors": [],
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", mock_preprocess)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        batch_size=50,
        ingestion_stage_entry=_valid_youtube_ingest_entry(3),
    )
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 3
    assert path == canonical_file
    assert canonical_file.exists()
    assert len(json.loads(canonical_file.read_text(encoding="utf-8"))) == 3


def test_youtube_preprocessing_executes_exactly_once(tmp_path, monkeypatch):
    """Verify preprocess_youtube_dataset is called exactly once per stage execution."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    call_count = 0

    def mock_preprocess(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 1,
            "processed_records": 1,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", mock_preprocess)

    execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert call_count == 1


def test_youtube_preprocessing_pipeline_failure(tmp_path, monkeypatch):
    """Verify pipeline failure status is handled safely and produces failed stage entry."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def mock_failure(*args, **kwargs):
        return {
            "status": "failed",
            "source": "youtube",
            "total_records": 1,
            "processed_records": 0,
            "failed_records": 1,
            "errors": ["Some internal failure"],
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", mock_failure)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "preprocessing_error"
    assert path is None
    assert not canonical_file.exists()


def test_youtube_preprocessing_summary_not_dict(tmp_path):
    """Verify validation fails if preprocessing summary is not a dictionary."""
    is_valid, count, error_cat, reason = validate_youtube_preprocessing_output(
        summary="not_a_dict",
        temp_file_path=tmp_path / "dummy.json",
    )
    assert is_valid is False
    assert error_cat == "preprocessing_error"
    assert reason == "Preprocessing result must be a dictionary object."


@pytest.mark.parametrize("bad_status", ["failed", "error", "no_records", "unknown"])
def test_youtube_preprocessing_invalid_status(tmp_path, bad_status):
    """Verify validation fails when status is not 'success'."""
    summary = {
        "status": bad_status,
        "total_records": 1,
        "processed_records": 1,
        "failed_records": 0,
    }
    is_valid, count, error_cat, reason = validate_youtube_preprocessing_output(
        summary=summary,
        temp_file_path=tmp_path / "dummy.json",
    )
    assert is_valid is False
    assert error_cat == "preprocessing_error"
    assert reason == "Preprocessing returned non-success status."


@pytest.mark.parametrize("missing_field", ["total_records", "processed_records", "failed_records"])
def test_youtube_preprocessing_missing_count_field(tmp_path, missing_field):
    """Verify validation fails if any required count field is missing."""
    summary = {
        "status": "success",
        "total_records": 1,
        "processed_records": 1,
        "failed_records": 0,
    }
    del summary[missing_field]
    is_valid, count, error_cat, reason = validate_youtube_preprocessing_output(
        summary=summary,
        temp_file_path=tmp_path / "dummy.json",
    )
    assert is_valid is False
    assert error_cat == "count_mismatch"
    assert missing_field in reason


@pytest.mark.parametrize("bad_val", [True, False, 1.5, "1", -1, None])
def test_youtube_preprocessing_rejects_invalid_count_types(tmp_path, bad_val):
    """Verify count validation rejects booleans, floats, strings, negative ints, and None."""
    summary = {
        "status": "success",
        "total_records": bad_val,
        "processed_records": 1,
        "failed_records": 0,
    }
    is_valid, count, error_cat, reason = validate_youtube_preprocessing_output(
        summary=summary,
        temp_file_path=tmp_path / "dummy.json",
    )
    assert is_valid is False
    assert error_cat == "count_mismatch"


def test_youtube_preprocessing_count_sum_mismatch(tmp_path, monkeypatch):
    """Verify validation fails when total_records != processed_records + failed_records."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_mismatch(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 5,  # Mismatch: 5 != 2 + 0
            "processed_records": 2,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_mismatch)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(2),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


def test_youtube_preprocessing_missing_output_file(tmp_path, monkeypatch):
    """Verify validation fails when preprocessing reports success but output file was not written."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    monkeypatch.setattr(
        "app.preprocess_youtube_dataset",
        lambda *a, **kw: {
            "status": "success",
            "total_records": 1,
            "processed_records": 1,
            "failed_records": 0,
        },
    )

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "missing_output"
    assert path is None


def test_youtube_preprocessing_empty_output_file(tmp_path, monkeypatch):
    """Verify validation fails when output file is empty (0 bytes)."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_empty(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return {"status": "success", "total_records": 0, "processed_records": 0, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_empty)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] in ("empty_output", "corrupt_json")
    assert path is None


def test_youtube_preprocessing_malformed_json(tmp_path, monkeypatch):
    """Verify validation fails when output file contains unparseable JSON."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_corrupt(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("{invalid_json_corrupted", encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_corrupt)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "corrupt_json"
    assert path is None


def test_youtube_preprocessing_wrong_root_type(tmp_path, monkeypatch):
    """Verify validation fails when JSON root is not a list."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_dict_root(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps({"key": "not a list"}), encoding="utf-8"
        )
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_dict_root)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_youtube_preprocessing_non_dictionary_record(tmp_path, monkeypatch):
    """Verify validation fails when an item in the JSON list is not a dictionary."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_scalar(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(["string_item"]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_scalar)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


@pytest.mark.parametrize("invalid_id", ["", "   ", None, 999])
def test_youtube_preprocessing_missing_or_invalid_video_id(tmp_path, monkeypatch, invalid_id):
    """Verify validation fails when video_id is missing, whitespace, or non-string."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    rec = _sample_processed_youtube_records(1)[0]
    if invalid_id is None:
        del rec["video_id"]
    else:
        rec["video_id"] = invalid_id

    def write_bad_id(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([rec]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_bad_id)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "invalid_record_id"
    assert path is None


def test_youtube_preprocessing_duplicate_video_ids(tmp_path, monkeypatch):
    """Verify validation fails when duplicate video_ids are present in serialized records."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    recs = _sample_processed_youtube_records(2)
    recs[1]["video_id"] = recs[0]["video_id"]

    def write_duplicates(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(recs), encoding="utf-8")
        return {"status": "success", "total_records": 2, "processed_records": 2, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_duplicates)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(2),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "duplicate_record_id"
    assert path is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "title_cleaned",
        "description_cleaned",
        "detected_language",
        "language_code",
        "preprocessing_status",
        "processed_at",
    ],
)
def test_youtube_preprocessing_missing_enrichment_field(tmp_path, monkeypatch, missing_field):
    """Verify validation fails when any required YouTube enrichment field is absent."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    rec = _sample_processed_youtube_records(1)[0]
    del rec[missing_field]

    def write_missing(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([rec]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_missing)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_youtube_preprocessing_invalid_enrichment_field_type(tmp_path, monkeypatch):
    """Verify validation fails when a required enrichment field is not a string."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    rec = _sample_processed_youtube_records(1)[0]
    rec["description_cleaned"] = 12345  # Not a string

    def write_invalid_type(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps([rec]), encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_invalid_type)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "schema_violation"
    assert path is None


def test_youtube_preprocessing_serialized_count_mismatch(tmp_path, monkeypatch):
    """Verify validation fails when serialized count does not match summary['processed_records']."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_mismatch(*args, **kwargs):
        # Writes 2 records but reports 1 processed
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 2,
            "processed_records": 1,
            "failed_records": 1,
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_mismatch)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(2),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "count_mismatch"
    assert path is None


def test_youtube_preprocessing_zero_records_fails(tmp_path, monkeypatch):
    """Verify zero processed records produces a failed stage result."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_zero(*args, **kwargs):
        Path(kwargs["output_path"]).write_text("[]", encoding="utf-8")
        return {
            "status": "success",
            "total_records": 0,
            "processed_records": 0,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_zero)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "empty_output"
    assert path is None


def test_youtube_preprocessing_partially_rejected_records_pass_when_counts_consistent(tmp_path, monkeypatch):
    """Verify stage succeeds when some records fail but valid records match processed_records count."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(3)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    valid_recs = _sample_processed_youtube_records(2)

    def write_partial(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(valid_recs), encoding="utf-8")
        return {
            "status": "success",
            "source": "youtube",
            "total_records": 3,
            "processed_records": 2,
            "failed_records": 1,
            "output_file": kwargs["output_path"],
            "errors": ["Record 'vid_failed': Preprocessing failed"],
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_partial)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(3),
    )
    assert stage_entry["status"] == "success"
    assert stage_entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()


def test_youtube_preprocessing_pipeline_exception_handling(tmp_path, monkeypatch):
    """Verify unexpected pipeline exception is safely caught and produces failed stage entry."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def raise_err(*args, **kwargs):
        raise RuntimeError("Unexpected pipeline failure!")

    monkeypatch.setattr("app.preprocess_youtube_dataset", raise_err)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "execution_error"
    assert path is None


def test_youtube_preprocessing_failed_validation_preserves_canonical_output(tmp_path, monkeypatch):
    """Verify pre-existing canonical output is NOT overwritten when validation fails."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(2)), encoding="utf-8")

    canonical_file = tmp_path / "processed.json"
    original_data = [{"video_id": "original_1", "title_cleaned": "Orig"}]
    canonical_file.write_text(json.dumps(original_data), encoding="utf-8")

    def mock_invalid_output(*args, **kwargs):
        # Write corrupted output
        Path(kwargs["output_path"]).write_text("{bad json", encoding="utf-8")
        return {"status": "success", "total_records": 2, "processed_records": 2, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", mock_invalid_output)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(2),
    )
    assert stage_entry["status"] == "failed"
    assert path is None

    # Canonical file must still contain the original data
    assert json.loads(canonical_file.read_text(encoding="utf-8")) == original_data


def test_youtube_preprocessing_atomic_promotion_failure_cleans_up(tmp_path, monkeypatch):
    """Verify temp file cleanup if os.replace fails during promotion."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_valid(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8"
        )
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_valid)

    def fail_replace(src, dst):
        raise OSError("Permission denied on replace")

    monkeypatch.setattr("os.replace", fail_replace)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"
    assert stage_entry["error_category"] == "execution_error"
    assert path is None

    temp_files = list(canonical_file.parent.glob(".tmp_youtube_processed_*"))
    assert len(temp_files) == 0


def test_youtube_preprocessing_temporary_file_cleanup_on_failure(tmp_path, monkeypatch):
    """Verify temporary output file is cleaned up on validation failure."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_invalid(*args, **kwargs):
        out_p = Path(kwargs["output_path"])
        out_p.write_text("invalid json", encoding="utf-8")
        return {"status": "success", "total_records": 1, "processed_records": 1, "failed_records": 0}

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_invalid)

    stage_entry, path = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(1),
    )
    assert stage_entry["status"] == "failed"

    temp_files = list(canonical_file.parent.glob(".tmp_youtube_processed_*"))
    assert len(temp_files) == 0


def test_youtube_preprocessing_diagnostics_contain_no_secrets_or_absolute_paths(tmp_path):
    """Verify validation failure diagnostics do not expose file paths, secrets, or raw text."""
    bad_summary = {
        "status": "failed",
        "secret_token": "sensitive_yt_key_xyz",
        "errors": ["C:\\Users\\DEEKSHA\\secret_internal\\path.py failed"],
    }
    _, _, category, diagnostic = validate_youtube_preprocessing_output(
        bad_summary, tmp_path / "dummy.json"
    )
    assert "\\" not in diagnostic
    assert "/" not in diagnostic
    assert "secret" not in diagnostic.lower()
    assert category == "preprocessing_error"


def test_youtube_preprocessing_stage_entry_structure_and_types(tmp_path, monkeypatch):
    """Verify stage manifest entry strictly satisfies Block 1 schema invariants."""
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(_sample_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "processed.json"

    def write_valid(*args, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8"
        )
        return {
            "status": "success",
            "total_records": 2,
            "processed_records": 2,
            "failed_records": 0,
        }

    monkeypatch.setattr("app.preprocess_youtube_dataset", write_valid)

    stage_entry, _ = execute_youtube_preprocessing_stage(
        raw_input_path=raw_file,
        canonical_output_path=canonical_file,
        ingestion_stage_entry=_valid_youtube_ingest_entry(2),
    )
    assert stage_entry["status"] == "success"
    assert isinstance(stage_entry["record_count"], int)
    assert stage_entry["record_count"] == 2
    assert isinstance(stage_entry["duration_seconds"], (int, float))
    assert stage_entry["duration_seconds"] >= 0.0
    assert stage_entry["reason"] is None
    assert stage_entry["error_category"] is None



# =====================================================================
# STEP 7.5C — Block 5B: Sentiment Output Validation Tests
# =====================================================================

def _sample_sentiment_metrics(total: int = 3, pos: int = 1, neu: int = 1, neg: int = 1, und: int = 0) -> dict[str, Any]:
    pos_pct = round((pos / total) * 100.0, 2) if total > 0 else 0.0
    neu_pct = round((neu / total) * 100.0, 2) if total > 0 else 0.0
    neg_pct = round((neg / total) * 100.0, 2) if total > 0 else 0.0
    und_pct = round((und / total) * 100.0, 2) if total > 0 else 0.0
    net_sentiment = round((pos - neg) / total, 4) if total > 0 else 0.0
    return {
        "total": total,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "undetermined": und,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "undetermined_pct": und_pct,
        "net_sentiment": net_sentiment,
    }


def _sample_sentiment_payload(reddit_count: int = 2, youtube_count: int = 1) -> dict[str, Any]:
    total = reddit_count + youtube_count
    now_iso = "2026-11-07T14:00:00+00:00"

    reddit_records = [
        {
            "post_id": f"t3_sent_{i}",
            "title": f"Reddit Title {i}",
            "text": f"Reddit Body {i}",
            "title_cleaned": f"Reddit Cleaned Title {i}",
            "text_cleaned": f"Reddit Cleaned Body {i}",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": now_iso,
            "created_utc": 1730101200 + i * 3600,
            "sentiment_label": "positive" if i % 2 == 1 else "neutral",
            "sentiment_score": 0.85,
            "sentiment_status": "success",
            "analyzed_at": now_iso,
        }
        for i in range(1, reddit_count + 1)
    ]

    youtube_records = [
        {
            "video_id": f"vid_sent_{i}",
            "title": f"YouTube Title {i}",
            "description": f"YouTube Desc {i}",
            "title_cleaned": f"YouTube Cleaned Title {i}",
            "description_cleaned": f"YouTube Cleaned Desc {i}",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": now_iso,
            "published_at": now_iso,
            "sentiment_label": "negative",
            "sentiment_score": 0.92,
            "sentiment_status": "success",
            "analyzed_at": now_iso,
        }
        for i in range(1, youtube_count + 1)
    ]

    r_pos = sum(1 for r in reddit_records if r["sentiment_label"] == "positive")
    r_neu = sum(1 for r in reddit_records if r["sentiment_label"] == "neutral")
    r_neg = sum(1 for r in reddit_records if r["sentiment_label"] == "negative")
    r_und = sum(1 for r in reddit_records if r["sentiment_label"] == "undetermined")
    reddit_summary = _sample_sentiment_metrics(reddit_count, r_pos, r_neu, r_neg, r_und)

    y_pos = sum(1 for r in youtube_records if r["sentiment_label"] == "positive")
    y_neu = sum(1 for r in youtube_records if r["sentiment_label"] == "neutral")
    y_neg = sum(1 for r in youtube_records if r["sentiment_label"] == "negative")
    y_und = sum(1 for r in youtube_records if r["sentiment_label"] == "undetermined")
    youtube_summary = _sample_sentiment_metrics(youtube_count, y_pos, y_neu, y_neg, y_und)

    overall_summary = _sample_sentiment_metrics(
        total,
        r_pos + y_pos,
        r_neu + y_neu,
        r_neg + y_neg,
        r_und + y_und,
    )

    daily_bucket = _sample_sentiment_metrics(total, r_pos + y_pos, r_neu + y_neu, r_neg + y_neg, r_und + y_und)
    weekly_bucket = _sample_sentiment_metrics(total, r_pos + y_pos, r_neu + y_neu, r_neg + y_neg, r_und + y_und)

    return {
        "status": "success",
        "generated_at": now_iso,
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        "total_records_processed": total,
        "summary": overall_summary,
        "reddit_summary": reddit_summary,
        "youtube_summary": youtube_summary,
        "temporal_aggregations": {
            "daily": {"2026-11-07": daily_bucket} if total > 0 else {},
            "weekly": {"2026-W45": weekly_bucket} if total > 0 else {},
        },
        "reddit_records": reddit_records,
        "youtube_records": youtube_records,
    }


def _sample_sentiment_summary(total_records: int = 3) -> dict[str, Any]:
    if total_records == 0:
        return _sample_sentiment_payload(0, 0)
    elif total_records == 1:
        return _sample_sentiment_payload(1, 0)
    elif total_records == 2:
        return _sample_sentiment_payload(2, 0)
    elif total_records == 3:
        return _sample_sentiment_payload(2, 1)
    return _sample_sentiment_payload(total_records, 0)


def test_sentiment_validation_valid_both_sources(tmp_path):
    """Verify validation passes when both Reddit and YouTube output records exist."""
    payload = _sample_sentiment_payload(reddit_count=2, youtube_count=1)
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=2,
        expected_youtube_count=1,
    )
    assert is_valid is True
    assert count == 3
    assert cat is None
    assert reason is None


def test_sentiment_validation_valid_reddit_only(tmp_path):
    """Verify validation passes when only Reddit records exist."""
    payload = _sample_sentiment_payload(reddit_count=2, youtube_count=0)
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=2,
        expected_youtube_count=0,
    )
    assert is_valid is True
    assert count == 2
    assert cat is None
    assert reason is None


def test_sentiment_validation_valid_youtube_only(tmp_path):
    """Verify validation passes when only YouTube records exist."""
    payload = _sample_sentiment_payload(reddit_count=0, youtube_count=3)
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=0,
        expected_youtube_count=3,
    )
    assert is_valid is True
    assert count == 3
    assert cat is None
    assert reason is None


def test_sentiment_validation_valid_zero_records(tmp_path):
    """Verify validation passes for legitimate zero-record outputs."""
    payload = _sample_sentiment_payload(reddit_count=0, youtube_count=0)
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=0,
        expected_youtube_count=0,
    )
    assert is_valid is True
    assert count == 0
    assert cat is None
    assert reason is None


@pytest.mark.parametrize("bad_summary", [None, "string", 123, [], True])
def test_sentiment_validation_summary_not_dict(tmp_path, bad_summary):
    """Verify validation fails when summary is not a dictionary."""
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text("{}", encoding="utf-8")
    is_valid, count, cat, reason = validate_sentiment_output(bad_summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "sentiment_error"
    assert "dictionary" in reason.lower()


@pytest.mark.parametrize("bad_status", ["failed", "error", "disabled", None, 123])
def test_sentiment_validation_summary_status_not_success(tmp_path, bad_status):
    """Verify validation fails when summary status is not 'success'."""
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text("{}", encoding="utf-8")
    is_valid, count, cat, reason = validate_sentiment_output({"status": bad_status}, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "sentiment_error"


def test_sentiment_validation_summary_total_records_mismatch(tmp_path):
    """Verify validation fails when summary total_records_processed does not match payload."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "sentiment_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        {"status": "success", "total_records_processed": 10},
        file_path,
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_missing_output_file(tmp_path):
    """Verify validation fails when temporary output file does not exist."""
    missing_file = tmp_path / "does_not_exist.json"
    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), missing_file
    )
    assert is_valid is False
    assert count == 0
    assert cat == "missing_output"


def test_sentiment_validation_empty_output_file(tmp_path):
    """Verify validation fails when temporary output file is 0 bytes."""
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), empty_file
    )
    assert is_valid is False
    assert count == 0
    assert cat == "empty_output"


def test_sentiment_validation_corrupt_json(tmp_path):
    """Verify validation fails when output contains malformed JSON."""
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{broken json syntax", encoding="utf-8")
    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), bad_file
    )
    assert is_valid is False
    assert count == 0
    assert cat == "corrupt_json"


@pytest.mark.parametrize("bad_root", [[], [1, 2], "string", 123, True])
def test_sentiment_validation_root_not_dict(tmp_path, bad_root):
    """Verify validation fails when JSON root is not a dictionary."""
    file_path = tmp_path / "bad_root.json"
    file_path.write_text(json.dumps(bad_root), encoding="utf-8")
    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize(
    "missing_key",
    [
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
    ],
)
def test_sentiment_validation_missing_top_level_keys(tmp_path, missing_key):
    """Verify validation fails when any required top-level key is missing."""
    payload = _sample_sentiment_payload(1, 1)
    del payload[missing_key]
    file_path = tmp_path / "missing_key.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert missing_key in reason


def test_sentiment_validation_payload_status_not_success(tmp_path):
    """Verify validation fails when payload status is not 'success'."""
    payload = _sample_sentiment_payload(1, 1)
    payload["status"] = "failed"
    file_path = tmp_path / "bad_status.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_model", ["", "   ", None, 123, True])
def test_sentiment_validation_model_name_invalid(tmp_path, bad_model):
    """Verify validation fails when model_name is missing, empty, or non-string."""
    payload = _sample_sentiment_payload(1, 1)
    payload["model_name"] = bad_model
    file_path = tmp_path / "bad_model.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_ts", ["not-a-timestamp", "2026-99-99", "", 12345, True, None])
def test_sentiment_validation_generated_at_invalid(tmp_path, bad_ts):
    """Verify validation fails when generated_at is not a valid ISO timestamp."""
    payload = _sample_sentiment_payload(1, 1)
    payload["generated_at"] = bad_ts
    file_path = tmp_path / "bad_ts.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


def test_sentiment_validation_rejects_boolean_total_records(tmp_path):
    """Verify validation explicitly rejects boolean total_records_processed."""
    payload = _sample_sentiment_payload(1, 0)
    payload["total_records_processed"] = True
    file_path = tmp_path / "bool_count.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


@pytest.mark.parametrize("bad_val", [2.0, "2", None, -1])
def test_sentiment_validation_rejects_non_integer_or_negative_total_records(tmp_path, bad_val):
    """Verify validation rejects floats, strings, None, and negative total_records_processed."""
    payload = _sample_sentiment_payload(1, 1)
    payload["total_records_processed"] = bad_val
    file_path = tmp_path / "bad_count.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


@pytest.mark.parametrize("count_key", ["total", "positive", "neutral", "negative", "undetermined"])
def test_sentiment_validation_rejects_boolean_summary_counts(tmp_path, count_key):
    """Verify validation rejects boolean count fields in summary dictionaries."""
    payload = _sample_sentiment_payload(1, 1)
    payload["summary"][count_key] = True
    file_path = tmp_path / "bool_summary.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_rejects_negative_summary_counts(tmp_path):
    """Verify validation rejects negative count values in summary dictionaries."""
    payload = _sample_sentiment_payload(1, 1)
    payload["summary"]["negative"] = -1
    file_path = tmp_path / "neg_summary.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_summary_sum_mismatch(tmp_path):
    """Verify validation fails when category sum does not equal summary total."""
    payload = _sample_sentiment_payload(1, 1)
    payload["summary"]["positive"] += 5
    file_path = tmp_path / "sum_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_overall_vs_sources_sum_mismatch(tmp_path):
    """Verify validation fails when overall total != reddit total + youtube total."""
    payload = _sample_sentiment_payload(2, 1)
    # Set summary to be internally valid for total=5 (1 pos, 1 neu, 1 neg, 2 und)
    # while source totals remain 2 (reddit) + 1 (youtube) = 3 != 5
    payload["summary"] = _sample_sentiment_metrics(total=5, pos=1, neu=1, neg=1, und=2)
    payload["total_records_processed"] = 5
    file_path = tmp_path / "source_sum_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(5), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "sum of reddit and youtube summary totals" in reason


def test_sentiment_validation_overall_total_vs_total_records_mismatch(tmp_path):
    """Verify validation fails when overall summary total != total_records_processed."""
    payload = _sample_sentiment_payload(2, 1)
    # Keep overall summary valid with total=3, but set total_records_processed to 4
    payload["total_records_processed"] = 4
    file_path = tmp_path / "total_records_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        {"status": "success", "total_records_processed": 4}, file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "Overall summary total does not match total_records_processed." in reason


def test_sentiment_validation_reddit_record_count_vs_summary_mismatch(tmp_path):
    """Verify validation fails when reddit records count != reddit_summary.total."""
    payload = _sample_sentiment_payload(2, 1)
    payload["reddit_records"].pop()
    file_path = tmp_path / "reddit_count_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_youtube_record_count_vs_summary_mismatch(tmp_path):
    """Verify validation fails when youtube records count != youtube_summary.total."""
    payload = _sample_sentiment_payload(2, 1)
    payload["youtube_records"].pop()
    file_path = tmp_path / "youtube_count_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_expected_reddit_count_mismatch(tmp_path):
    """Verify validation fails when reddit records count != expected_reddit_count."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "expected_reddit_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path, expected_reddit_count=10
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_expected_youtube_count_mismatch(tmp_path):
    """Verify validation fails when youtube records count != expected_youtube_count."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "expected_youtube_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path, expected_youtube_count=10
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


@pytest.mark.parametrize("bad_expected", [True, 1.5, "2", -1])
def test_sentiment_validation_invalid_expected_count_types(tmp_path, bad_expected):
    """Verify validation rejects boolean, float, string, and negative expected count parameters."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "invalid_expected.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path, expected_reddit_count=bad_expected
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"


def test_sentiment_validation_duplicate_reddit_post_id(tmp_path):
    """Verify validation fails when duplicate post_id exists in reddit_records."""
    payload = _sample_sentiment_payload(2, 0)
    payload["reddit_records"][1]["post_id"] = payload["reddit_records"][0]["post_id"]
    file_path = tmp_path / "dup_reddit.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "duplicate_record_id"


def test_sentiment_validation_duplicate_youtube_video_id(tmp_path):
    """Verify validation fails when duplicate video_id exists in youtube_records."""
    payload = _sample_sentiment_payload(0, 2)
    payload["youtube_records"][1]["video_id"] = payload["youtube_records"][0]["video_id"]
    file_path = tmp_path / "dup_youtube.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "duplicate_record_id"


@pytest.mark.parametrize("bad_id", ["", "   ", None, 123, True])
def test_sentiment_validation_invalid_reddit_post_id(tmp_path, bad_id):
    """Verify validation fails when reddit post_id is empty, whitespace, or non-string."""
    payload = _sample_sentiment_payload(1, 0)
    payload["reddit_records"][0]["post_id"] = bad_id
    file_path = tmp_path / "bad_post_id.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "invalid_record_id"


@pytest.mark.parametrize("bad_id", ["", "   ", None, 123, True])
def test_sentiment_validation_invalid_youtube_video_id(tmp_path, bad_id):
    """Verify validation fails when youtube video_id is empty, whitespace, or non-string."""
    payload = _sample_sentiment_payload(0, 1)
    payload["youtube_records"][0]["video_id"] = bad_id
    file_path = tmp_path / "bad_video_id.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "invalid_record_id"


@pytest.mark.parametrize(
    "missing_field",
    ["sentiment_label", "sentiment_score", "sentiment_status", "analyzed_at"],
)
def test_sentiment_validation_missing_enrichment_field(tmp_path, missing_field):
    """Verify validation fails when any required sentiment enrichment field is missing."""
    payload = _sample_sentiment_payload(1, 1)
    del payload["reddit_records"][0][missing_field]
    file_path = tmp_path / "missing_enrichment.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_label", ["ecstatic", "POS", "good", "", 123, None, True])
def test_sentiment_validation_invalid_sentiment_label(tmp_path, bad_label):
    """Verify validation fails when sentiment_label is not in permitted set."""
    payload = _sample_sentiment_payload(1, 1)
    payload["reddit_records"][0]["sentiment_label"] = bad_label
    file_path = tmp_path / "bad_label.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


def test_sentiment_validation_rejects_boolean_sentiment_score(tmp_path):
    """Verify validation rejects boolean sentiment_score."""
    payload = _sample_sentiment_payload(1, 1)
    payload["reddit_records"][0]["sentiment_score"] = True
    file_path = tmp_path / "bool_score.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, 2.0, -1.0, "0.5", None])
def test_sentiment_validation_rejects_out_of_range_or_non_float_sentiment_score(tmp_path, bad_score):
    """Verify validation rejects out-of-range or non-numeric sentiment_score."""
    payload = _sample_sentiment_payload(1, 1)
    payload["reddit_records"][0]["sentiment_score"] = bad_score
    file_path = tmp_path / "bad_score.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sentiment_validation_rejects_nan_and_inf_sentiment_score(tmp_path, bad_val):
    """Verify validation rejects NaN and Infinity in record sentiment_score."""
    payload = _sample_sentiment_payload(1, 1)
    payload["reddit_records"][0]["sentiment_score"] = bad_val
    file_path = tmp_path / "nan_score.json"
    file_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sentiment_validation_rejects_nan_and_inf_in_summary_metrics(tmp_path, bad_val):
    """Verify validation rejects NaN and Infinity in summary percentages or net sentiment."""
    payload = _sample_sentiment_payload(1, 1)
    payload["summary"]["positive_pct"] = bad_val
    file_path = tmp_path / "nan_metric.json"
    file_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_ts", ["invalid-date", "2026/11/07", "", None, 12345, True])
def test_sentiment_validation_rejects_invalid_analyzed_at(tmp_path, bad_ts):
    """Verify validation rejects non-ISO or invalid analyzed_at timestamps."""
    payload = _sample_sentiment_payload(1, 1)
    payload["reddit_records"][0]["analyzed_at"] = bad_ts
    file_path = tmp_path / "bad_analyzed_at.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("bad_temporal", ["not_a_dict", [1, 2], 123, None, True])
def test_sentiment_validation_invalid_temporal_aggregations_type(tmp_path, bad_temporal):
    """Verify validation fails when temporal_aggregations is not a dictionary."""
    payload = _sample_sentiment_payload(1, 1)
    payload["temporal_aggregations"] = bad_temporal
    file_path = tmp_path / "bad_temporal.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


@pytest.mark.parametrize("missing_window", ["daily", "weekly"])
def test_sentiment_validation_missing_temporal_window(tmp_path, missing_window):
    """Verify validation fails when daily or weekly window is missing in temporal_aggregations."""
    payload = _sample_sentiment_payload(1, 1)
    del payload["temporal_aggregations"][missing_window]
    file_path = tmp_path / "missing_win.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


def test_sentiment_validation_temporal_bucket_invalid_metrics(tmp_path):
    """Verify validation fails when a temporal bucket contains invalid summary metrics."""
    payload = _sample_sentiment_payload(1, 1)
    payload["temporal_aggregations"]["daily"]["2026-11-07"]["positive_pct"] = 150.0
    file_path = tmp_path / "bad_bucket.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(2), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"


def test_sentiment_validation_diagnostics_contain_no_secrets_or_absolute_paths(tmp_path):
    """Verify failure diagnostics remain sanitized without leaking filesystem paths or credentials."""
    bad_file = tmp_path / "secret_customer_path" / "out.json"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("{broken", encoding="utf-8")

    is_valid, count, category, reason = validate_sentiment_output(
        {"status": "success"}, bad_file
    )
    assert is_valid is False
    assert count == 0
    assert category == "corrupt_json"
    assert "secret_customer_path" not in reason
    assert str(tmp_path) not in reason


def test_sentiment_validation_rejects_percentage_count_mismatch(tmp_path):
    """Verify validation fails when a percentage metric disagrees with underlying category counts."""
    payload = _sample_sentiment_payload(2, 1)
    payload["summary"]["positive_pct"] = 80.0
    file_path = tmp_path / "pct_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "positive_pct" in reason
    assert "does not match underlying count" in reason


def test_sentiment_validation_rejects_net_sentiment_count_mismatch(tmp_path):
    """Verify validation fails when net_sentiment disagrees with underlying category counts."""
    payload = _sample_sentiment_payload(2, 1)
    payload["summary"]["net_sentiment"] = 0.50
    file_path = tmp_path / "net_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "net_sentiment" in reason
    assert "does not match underlying counts" in reason


def test_sentiment_validation_rejects_zero_records_nonzero_percentage(tmp_path):
    """Verify validation fails when zero-record output contains non-zero percentage metrics."""
    payload = _sample_sentiment_payload(0, 0)
    payload["summary"]["positive_pct"] = 5.0
    file_path = tmp_path / "zero_nonzero_pct.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(0), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "must equal 0.0 when total is 0" in reason


def test_sentiment_validation_rejects_zero_records_nonzero_net_sentiment(tmp_path):
    """Verify validation fails when zero-record output contains non-zero net sentiment."""
    payload = _sample_sentiment_payload(0, 0)
    payload["summary"]["net_sentiment"] = 0.25
    file_path = tmp_path / "zero_nonzero_net.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(0), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "must equal 0.0 when total is 0" in reason


def test_sentiment_validation_rejects_non_dict_reddit_record(tmp_path):
    """Verify validation fails when an element in reddit_records is not a dictionary."""
    payload = _sample_sentiment_payload(2, 1)
    payload["reddit_records"][0] = 123
    file_path = tmp_path / "non_dict_reddit.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "Reddit record at index 0 is not a dictionary." in reason


def test_sentiment_validation_rejects_non_dict_youtube_record(tmp_path):
    """Verify validation fails when an element in youtube_records is not a dictionary."""
    payload = _sample_sentiment_payload(2, 1)
    payload["youtube_records"][0] = "invalid_string"
    file_path = tmp_path / "non_dict_youtube.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(3), file_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "YouTube record at index 0 is not a dictionary." in reason


@pytest.mark.parametrize("bad_path", [None, 123, True, False, [1, 2]])
def test_sentiment_validation_invalid_temp_file_path(bad_path):
    """Verify validate_sentiment_output handles invalid temp_file_path safely without uncaught exceptions."""
    is_valid, count, cat, reason = validate_sentiment_output(
        _sample_sentiment_summary(1), bad_path
    )
    assert is_valid is False
    assert count == 0
    assert cat == "missing_output"
    assert reason == "Sentiment output file path is invalid."


# =====================================================================
# STEP 7.5C — Block 5C: Sentiment Stage Execution Tests
# =====================================================================


def _create_mock_sentiment_analyzer() -> MultilingualSentimentAnalyzer:
    """Creates an isolated offline MultilingualSentimentAnalyzer backed by MockSentimentPipeline."""
    return MultilingualSentimentAnalyzer(pipeline_override=MockSentimentPipeline())


# --- Group 1: Provenance Validation (Fail Closed) ---


def test_sentiment_stage_missing_both_provenance_entries():
    """Verify execution fails closed with missing_upstream_provenance when both entries are None."""
    entry, path = execute_sentiment_stage(
        reddit_stage_entry=None,
        youtube_stage_entry=None,
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert entry["record_count"] == 0
    assert "Missing upstream preprocessing provenance" in entry["reason"]
    assert path is None


def test_sentiment_stage_missing_reddit_provenance():
    """Verify execution fails closed when reddit_stage_entry is None."""
    entry, path = execute_sentiment_stage(
        reddit_stage_entry=None,
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert entry["record_count"] == 0
    assert path is None


def test_sentiment_stage_missing_youtube_provenance():
    """Verify execution fails closed when youtube_stage_entry is None."""
    entry, path = execute_sentiment_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry=None,
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert entry["record_count"] == 0
    assert path is None


@pytest.mark.parametrize("bad_entry", ["not_a_dict", 123, [1, 2], True])
def test_sentiment_stage_malformed_provenance_not_dict(bad_entry):
    """Verify execution fails closed when provenance entry is not a dictionary."""
    entry1, _ = execute_sentiment_stage(
        reddit_stage_entry=bad_entry,
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry1["status"] == "failed"
    assert entry1["error_category"] == "invalid_upstream_provenance"

    entry2, _ = execute_sentiment_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry=bad_entry,
    )
    assert entry2["status"] == "failed"
    assert entry2["error_category"] == "invalid_upstream_provenance"


@pytest.mark.parametrize("bad_status", ["pending", "running", "unknown", "", None, 123])
def test_sentiment_stage_unsupported_status(bad_status):
    """Verify execution fails closed when provenance entry status is unsupported."""
    entry1, _ = execute_sentiment_stage(
        reddit_stage_entry={"status": bad_status, "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry1["status"] == "failed"
    assert entry1["error_category"] == "invalid_upstream_provenance"

    entry2, _ = execute_sentiment_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": bad_status, "record_count": 0},
    )
    assert entry2["status"] == "failed"
    assert entry2["error_category"] == "invalid_upstream_provenance"


@pytest.mark.parametrize("bad_count", [True, False, 1.5, -1, "10", None, []])
def test_sentiment_stage_invalid_record_count_types(bad_count):
    """Verify execution fails closed when provenance record_count is not a valid non-negative integer."""
    entry1, _ = execute_sentiment_stage(
        reddit_stage_entry={"status": "success", "record_count": bad_count},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry1["status"] == "failed"
    assert entry1["error_category"] == "invalid_upstream_provenance"

    entry2, _ = execute_sentiment_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "success", "record_count": bad_count},
    )
    assert entry2["status"] == "failed"
    assert entry2["error_category"] == "invalid_upstream_provenance"


# --- Group 2: Multi-Source Dependency and Skip Handling ---


def test_sentiment_stage_explicit_user_skip():
    """Verify explicit skip=True returns skipped entry immediately without inspecting provenance."""
    entry, path = execute_sentiment_stage(
        skip=True,
        reddit_stage_entry=None,
        youtube_stage_entry=None,
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert entry["record_count"] == 0
    assert entry["duration_seconds"] == 0.0
    assert entry["error_category"] is None
    assert path is None


def test_sentiment_stage_both_skipped():
    """Verify stage is skipped with all_sources_skipped when both sources skipped."""
    entry, path = execute_sentiment_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "all_sources_skipped"
    assert entry["record_count"] == 0
    assert path is None


def test_sentiment_stage_both_failed():
    """Verify stage is skipped with prior_stage_failure when both sources failed."""
    entry, path = execute_sentiment_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert entry["record_count"] == 0
    assert path is None


def test_sentiment_stage_failed_plus_skipped():
    """Verify stage is skipped with prior_stage_failure when one source failed and other skipped."""
    entry1, path1 = execute_sentiment_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry1["status"] == "skipped"
    assert entry1["reason"] == "prior_stage_failure"
    assert path1 is None

    entry2, path2 = execute_sentiment_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry2["status"] == "skipped"
    assert entry2["reason"] == "prior_stage_failure"
    assert path2 is None


def test_sentiment_stage_reddit_success_youtube_skipped(tmp_path):
    """Verify stage executes Reddit only when Reddit succeeded and YouTube skipped."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(payload["reddit_records"]) == 2
    assert len(payload["youtube_records"]) == 0


def test_sentiment_stage_youtube_success_reddit_skipped(tmp_path):
    """Verify stage executes YouTube only when YouTube succeeded and Reddit skipped."""
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=None,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(payload["reddit_records"]) == 0
    assert len(payload["youtube_records"]) == 2


def test_sentiment_stage_reddit_success_youtube_failed(tmp_path):
    """Verify stage executes Reddit only when Reddit succeeded and YouTube failed."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    assert path == canonical_file


def test_sentiment_stage_youtube_success_reddit_failed(tmp_path):
    """Verify stage executes YouTube only when YouTube succeeded and Reddit failed."""
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=None,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    assert path == canonical_file


def test_sentiment_stage_successful_zero_record_source(tmp_path):
    """Verify successful upstream source with record_count=0 is valid and processed."""
    r_file = tmp_path / "reddit_empty.json"
    r_file.write_text("[]", encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 0
    assert path == canonical_file
    assert canonical_file.exists()


# --- Group 3: Input Validation Helper Tests ---


@pytest.mark.parametrize("bad_path", [None, True, False, 123, [1, 2]])
def test_validate_active_source_missing_or_invalid_path(bad_path):
    """Verify _validate_active_sentiment_source_file rejects None, bools, and invalid types."""
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", bad_path, 0)
    assert valid is False
    assert cat == "missing_processed_input"
    assert "missing or invalid" in reason


def test_validate_active_source_file_not_found(tmp_path):
    """Verify validation fails when input file does not exist."""
    missing = tmp_path / "non_existent.json"
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", missing, 0)
    assert valid is False
    assert cat == "missing_processed_input"
    assert "does not exist" in reason


def test_validate_active_source_directory_instead_of_file(tmp_path):
    """Verify validation fails when directory is provided instead of a regular file."""
    valid, cat, reason = _validate_active_sentiment_source_file("YouTube", tmp_path, 0)
    assert valid is False
    assert cat == "missing_processed_input"
    assert "does not exist" in reason


def test_validate_active_source_zero_byte_input(tmp_path):
    """Verify validation fails with empty_input when file is 0 bytes."""
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", empty, 0)
    assert valid is False
    assert cat == "empty_input"
    assert "empty (0 bytes)" in reason


def test_validate_active_source_corrupt_json(tmp_path):
    """Verify validation fails with corrupt_json when file contains malformed JSON."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{broken", encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", bad_json, 0)
    assert valid is False
    assert cat == "corrupt_json"
    assert "malformed or unreadable JSON" in reason


def test_validate_active_source_root_not_list(tmp_path):
    """Verify validation fails with schema_violation when JSON root is not a list."""
    dict_root = tmp_path / "dict_root.json"
    dict_root.write_text(json.dumps({"data": []}), encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", dict_root, 0)
    assert valid is False
    assert cat == "schema_violation"
    assert "root must be a list" in reason


def test_validate_active_source_record_not_dict(tmp_path):
    """Verify validation fails with schema_violation when record element is not a dict."""
    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text(json.dumps([{"post_id": "1"}, 42]), encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", non_dict, 2)
    assert valid is False
    assert cat == "schema_violation"
    assert "is not a dictionary" in reason


def test_validate_active_source_count_mismatch(tmp_path):
    """Verify validation fails with count_mismatch when count does not match expected_count."""
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id": "1"}, {"id": "2"}]), encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", records, 3)
    assert valid is False
    assert cat == "count_mismatch"
    assert "does not match upstream manifest count" in reason


def test_validate_active_source_valid_empty(tmp_path):
    """Verify validation passes for valid empty list with expected count 0."""
    empty_list = tmp_path / "empty_list.json"
    empty_list.write_text("[]", encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("Reddit", empty_list, 0)
    assert valid is True
    assert cat is None
    assert reason is None


def test_validate_active_source_valid_records(tmp_path):
    """Verify validation passes for valid list of dicts with matching count."""
    records_file = tmp_path / "valid.json"
    records_file.write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
    valid, cat, reason = _validate_active_sentiment_source_file("YouTube", records_file, 2)
    assert valid is True
    assert cat is None
    assert reason is None


# --- Group 4: Stage Execution Input Validation Tests ---


def test_sentiment_stage_reddit_missing_active_input():
    """Verify stage fails when Reddit is active but reddit_processed_path is None."""
    entry, path = execute_sentiment_stage(
        reddit_processed_path=None,
        youtube_processed_path=None,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_sentiment_stage_reddit_file_not_found(tmp_path):
    """Verify stage fails when Reddit processed file does not exist."""
    missing = tmp_path / "does_not_exist.json"
    entry, path = execute_sentiment_stage(
        reddit_processed_path=missing,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_sentiment_stage_reddit_directory_supplied(tmp_path):
    """Verify stage fails when Reddit path is a directory instead of a file."""
    entry, path = execute_sentiment_stage(
        reddit_processed_path=tmp_path,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_sentiment_stage_reddit_zero_byte_file(tmp_path):
    """Verify stage fails with empty_input when Reddit processed file is 0 bytes."""
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    entry, path = execute_sentiment_stage(
        reddit_processed_path=empty,
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "empty_input"
    assert path is None


def test_sentiment_stage_reddit_corrupt_json(tmp_path):
    """Verify stage fails with corrupt_json when Reddit processed file has invalid JSON."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    entry, path = execute_sentiment_stage(
        reddit_processed_path=corrupt,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert path is None


def test_sentiment_stage_reddit_non_list_root(tmp_path):
    """Verify stage fails with schema_violation when Reddit processed file root is a dict."""
    not_list = tmp_path / "dict.json"
    not_list.write_text(json.dumps({"data": "bad"}), encoding="utf-8")
    entry, path = execute_sentiment_stage(
        reddit_processed_path=not_list,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_sentiment_stage_reddit_non_dict_record(tmp_path):
    """Verify stage fails with schema_violation when a Reddit record is not a dictionary."""
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps(["string_element"]), encoding="utf-8")
    entry, path = execute_sentiment_stage(
        reddit_processed_path=bad_rec,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_sentiment_stage_reddit_count_mismatch(tmp_path):
    """Verify stage fails with count_mismatch when Reddit record count does not match manifest."""
    rec_file = tmp_path / "records.json"
    rec_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    entry, path = execute_sentiment_stage(
        reddit_processed_path=rec_file,
        reddit_stage_entry={"status": "success", "record_count": 3},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "count_mismatch"
    assert path is None


def test_sentiment_stage_youtube_missing_active_input():
    """Verify stage fails when YouTube is active but youtube_processed_path is None."""
    entry, path = execute_sentiment_stage(
        reddit_processed_path=None,
        youtube_processed_path=None,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_sentiment_stage_youtube_file_not_found(tmp_path):
    """Verify stage fails when YouTube processed file does not exist."""
    missing = tmp_path / "does_not_exist.json"
    entry, path = execute_sentiment_stage(
        youtube_processed_path=missing,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_sentiment_stage_youtube_zero_byte_file(tmp_path):
    """Verify stage fails with empty_input when YouTube processed file is 0 bytes."""
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    entry, path = execute_sentiment_stage(
        youtube_processed_path=empty,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "empty_input"
    assert path is None


def test_sentiment_stage_youtube_corrupt_json(tmp_path):
    """Verify stage fails with corrupt_json when YouTube processed file has invalid JSON."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    entry, path = execute_sentiment_stage(
        youtube_processed_path=corrupt,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert path is None


def test_sentiment_stage_youtube_non_list_root(tmp_path):
    """Verify stage fails with schema_violation when YouTube processed file root is a dict."""
    not_list = tmp_path / "dict.json"
    not_list.write_text(json.dumps({"data": "bad"}), encoding="utf-8")
    entry, path = execute_sentiment_stage(
        youtube_processed_path=not_list,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_sentiment_stage_youtube_non_dict_record(tmp_path):
    """Verify stage fails with schema_violation when a YouTube record is not a dictionary."""
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps([12345]), encoding="utf-8")
    entry, path = execute_sentiment_stage(
        youtube_processed_path=bad_rec,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_sentiment_stage_youtube_count_mismatch(tmp_path):
    """Verify stage fails with count_mismatch when YouTube record count does not match manifest."""
    rec_file = tmp_path / "records.json"
    rec_file.write_text(json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8")
    entry, path = execute_sentiment_stage(
        youtube_processed_path=rec_file,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "count_mismatch"
    assert path is None


# --- Group 5: Inactive-Source Isolation / Sentinel Tests ---


def test_sentiment_stage_inactive_youtube_passes_sentinel(tmp_path, monkeypatch):
    """Verify Reddit-only execution passes non-existent sentinel path for YouTube and active path for Reddit."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    captured_args = {}

    def mock_run_pipe(**kwargs):
        captured_args.update(kwargs)
        # Create valid output file at output_path
        out_path = Path(kwargs["output_path"])
        payload = _sample_sentiment_payload(reddit_count=1, youtube_count=0)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_sentiment_pipeline", mock_run_pipe)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    # Verify Reddit input was passed unchanged
    assert captured_args["reddit_input_path"] == r_file
    # Verify YouTube input was passed a sentinel path that does NOT exist
    y_input = captured_args["youtube_input_path"]
    assert isinstance(y_input, Path)
    assert not y_input.exists()
    assert ".sentinel_inactive_youtube_" in y_input.name


def test_sentiment_stage_inactive_reddit_passes_sentinel(tmp_path, monkeypatch):
    """Verify YouTube-only execution passes non-existent sentinel path for Reddit and active path for YouTube."""
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    captured_args = {}

    def mock_run_pipe(**kwargs):
        captured_args.update(kwargs)
        out_path = Path(kwargs["output_path"])
        payload = _sample_sentiment_payload(reddit_count=0, youtube_count=1)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_sentiment_pipeline", mock_run_pipe)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=None,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert captured_args["youtube_input_path"] == y_file
    r_input = captured_args["reddit_input_path"]
    assert isinstance(r_input, Path)
    assert not r_input.exists()
    assert ".sentinel_inactive_reddit_" in r_input.name


def test_sentiment_stage_inactive_source_does_not_load_stale_canonical_files(tmp_path):
    """Verify inactive sources do not accidentally load stale canonical processed data."""
    # Place a stale processed file in the directory
    stale_youtube = tmp_path / "youtube_videos_processed.json"
    stale_youtube.write_text(
        json.dumps(_sample_processed_youtube_records(5)), encoding="utf-8"
    )

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "sentiment_analysis_results.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    # Check that canonical output contains ONLY Reddit records, not the 5 stale YouTube records
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert payload["total_records_processed"] == 1
    assert len(payload["youtube_records"]) == 0
    assert len(payload["reddit_records"]) == 1


# --- Group 6: Pipeline Execution, Output Validation, Atomic Promotion, Cleanup ---


def test_sentiment_stage_pipeline_exception_handling(tmp_path, monkeypatch):
    """Verify execution catches pipeline exceptions, cleans up temp file, and returns execution_error."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def blow_up(**kwargs):
        # Create temporary file to ensure it gets cleaned up
        Path(kwargs["output_path"]).write_text("partial", encoding="utf-8")
        raise RuntimeError("GPU Driver Crashed / Out of Memory")

    monkeypatch.setattr("app.run_sentiment_pipeline", blow_up)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert "Sentiment execution encountered an unexpected error" in entry["reason"]
    assert "GPU" not in entry["reason"]
    assert path is None
    # Verify no temporary files remain
    tmp_files = list(tmp_path.glob(".tmp_sentiment_*"))
    assert len(tmp_files) == 0


def test_sentiment_stage_missing_temporary_output(tmp_path, monkeypatch):
    """Verify stage fails with missing_output when pipeline returns success without creating file."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    monkeypatch.setattr(
        "app.run_sentiment_pipeline",
        lambda **kwargs: {"status": "success", "total_records_processed": 1},
    )

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_output"
    assert path is None


def test_sentiment_stage_corrupt_temporary_output(tmp_path, monkeypatch):
    """Verify stage fails with corrupt_json when pipeline writes invalid JSON."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_corrupt(**kwargs):
        Path(kwargs["output_path"]).write_text("{malformed", encoding="utf-8")
        return {"status": "success", "total_records_processed": 1}

    monkeypatch.setattr("app.run_sentiment_pipeline", write_corrupt)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_sentiment_*"))) == 0


def test_sentiment_stage_invalid_output_schema(tmp_path, monkeypatch):
    """Verify stage fails with schema_violation when pipeline writes payload missing fields."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_bad_schema(**kwargs):
        payload = _sample_sentiment_payload(1, 0)
        del payload["temporal_aggregations"]
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "success", "total_records_processed": 1}

    monkeypatch.setattr("app.run_sentiment_pipeline", write_bad_schema)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_sentiment_*"))) == 0


def test_sentiment_stage_output_count_mismatch(tmp_path, monkeypatch):
    """Verify stage fails with count_mismatch when output record count disagrees with upstream manifest."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_mismatch(**kwargs):
        # Writes 2 records, but upstream expected 1
        payload = _sample_sentiment_payload(2, 0)
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "success", "total_records_processed": 2}

    monkeypatch.setattr("app.run_sentiment_pipeline", write_mismatch)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "count_mismatch"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_sentiment_*"))) == 0


def test_sentiment_stage_validation_exception_handling(tmp_path, monkeypatch):
    """Verify unexpected validation exceptions are caught safely and return execution_error."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def mock_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("{}", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_sentiment_pipeline", mock_pipe)

    def blow_up_validator(*args, **kwargs):
        raise TypeError("Unexpected validator internal error")

    monkeypatch.setattr("app.validate_sentiment_output", blow_up_validator)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_sentiment_*"))) == 0


def test_sentiment_stage_successful_atomic_promotion(tmp_path):
    """Verify multi-source execution atomically creates canonical output and returns success."""
    r_file = tmp_path / "reddit_proc.json"
    y_file = tmp_path / "youtube_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 3
    assert entry["error_category"] is None
    assert entry["reason"] is None
    assert entry["duration_seconds"] >= 0.0
    assert path == canonical_file
    assert canonical_file.exists()
    assert len(list(tmp_path.glob(".tmp_sentiment_*"))) == 0


def test_sentiment_stage_atomic_replacement_failure(tmp_path, monkeypatch):
    """Verify atomic promotion failure cleans up temp file and returns promotion_error."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_replace(src, dst):
        raise OSError("Permission denied during atomic replace")

    monkeypatch.setattr("app.os.replace", fail_replace)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "promotion_error"
    assert "Atomic output promotion failed" in entry["reason"]
    assert path is None
    assert not canonical_file.exists()
    assert len(list(tmp_path.glob(".tmp_sentiment_*"))) == 0


def test_sentiment_stage_preserves_existing_canonical_on_pipeline_failure(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is completely preserved when pipeline execution fails."""
    canonical_file = tmp_path / "out.json"
    original_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(original_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    def fail_pipe(**kwargs):
        raise RuntimeError("Pipeline failed midway")

    monkeypatch.setattr("app.run_sentiment_pipeline", fail_pipe)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert path is None
    # Existing canonical file is preserved exactly
    assert canonical_file.exists()
    assert canonical_file.read_text(encoding="utf-8") == original_content


def test_sentiment_stage_preserves_existing_canonical_on_validation_failure(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is completely preserved when output validation fails."""
    canonical_file = tmp_path / "out.json"
    original_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(original_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    def write_corrupted(**kwargs):
        Path(kwargs["output_path"]).write_text("{bad", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_sentiment_pipeline", write_corrupted)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert path is None
    assert canonical_file.exists()
    assert canonical_file.read_text(encoding="utf-8") == original_content


def test_sentiment_stage_preserves_existing_canonical_on_promotion_failure(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is preserved when atomic promotion fails."""
    canonical_file = tmp_path / "out.json"
    original_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(original_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("Permission denied")

    monkeypatch.setattr("app.os.replace", fail_replace)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert path is None
    assert canonical_file.exists()
    assert canonical_file.read_text(encoding="utf-8") == original_content


def test_sentiment_stage_valid_zero_record_output_successfully_promoted(tmp_path):
    """Verify zero-record output from both active sources is valid and promoted."""
    r_file = tmp_path / "reddit_empty.json"
    y_file = tmp_path / "youtube_empty.json"
    r_file.write_text("[]", encoding="utf-8")
    y_file.write_text("[]", encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 0
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert payload["total_records_processed"] == 0


def test_sentiment_stage_default_canonical_path_resolution(tmp_path, monkeypatch):
    """Verify default canonical path resolves to processed_data_path / sentiment_analysis_results.json."""
    config_mock = type("ConfigMock", (), {"processed_data_path": tmp_path})()

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=None,
        config=config_mock,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    expected_path = tmp_path / "sentiment_analysis_results.json"
    assert path == expected_path
    assert expected_path.exists()


def test_sentiment_stage_diagnostics_sanitized(tmp_path):
    """Verify failure diagnostics remain sanitized without leaking filesystem paths or credentials."""
    secret_dir = tmp_path / "secret_customer_token"
    secret_dir.mkdir(parents=True, exist_ok=True)
    bad_file = secret_dir / "bad.json"
    bad_file.write_text("{broken", encoding="utf-8")

    entry, path = execute_sentiment_stage(
        reddit_processed_path=bad_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert "secret_customer_token" not in entry["reason"]
    assert str(tmp_path) not in entry["reason"]


# --- Complete In-Memory Summary vs Payload Equality Tests ---


def test_sentiment_validation_rejects_summary_metrics_mismatch(tmp_path):
    """Verify validation fails when in-memory summary metrics disagree with payload metrics."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "valid_payload.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    # Mismatched summary: same total (3) but altered positive metric
    summary = _sample_sentiment_payload(2, 1)
    summary["summary"]["positive"] = summary["summary"]["positive"] + 1
    summary["summary"]["positive_pct"] = round((summary["summary"]["positive"] / 3) * 100.0, 2)

    is_valid, count, cat, reason = validate_sentiment_output(summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "In-memory pipeline summary does not match temporary JSON payload." in reason


def test_sentiment_validation_rejects_summary_model_name_mismatch(tmp_path):
    """Verify validation fails when in-memory summary model_name disagrees with payload."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "valid_payload.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _sample_sentiment_payload(2, 1)
    summary["model_name"] = "different/transformer-model-v2"

    is_valid, count, cat, reason = validate_sentiment_output(summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "In-memory pipeline summary does not match temporary JSON payload." in reason


def test_sentiment_validation_rejects_summary_reddit_records_mismatch(tmp_path):
    """Verify validation fails when in-memory summary reddit_records disagree with payload."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "valid_payload.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _sample_sentiment_payload(2, 1)
    summary["reddit_records"][0]["sentiment_label"] = "neutral"

    is_valid, count, cat, reason = validate_sentiment_output(summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "In-memory pipeline summary does not match temporary JSON payload." in reason


def test_sentiment_validation_rejects_summary_temporal_aggregation_mismatch(tmp_path):
    """Verify validation fails when in-memory summary temporal aggregations disagree with payload."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "valid_payload.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _sample_sentiment_payload(2, 1)
    summary["temporal_aggregations"] = {"daily": {}, "weekly": {}}

    is_valid, count, cat, reason = validate_sentiment_output(summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "In-memory pipeline summary does not match temporary JSON payload." in reason


def test_sentiment_validation_rejects_summary_missing_required_fields(tmp_path):
    """Verify validation fails when in-memory summary is missing any required payload field."""
    payload = _sample_sentiment_payload(2, 1)
    file_path = tmp_path / "valid_payload.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _sample_sentiment_payload(2, 1)
    del summary["temporal_aggregations"]

    is_valid, count, cat, reason = validate_sentiment_output(summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "In-memory pipeline summary does not match temporary JSON payload." in reason


def test_sentiment_stage_in_memory_mismatch_prevents_promotion(tmp_path, monkeypatch):
    """Verify stage aborts and does not promote output if pipeline returns mismatched summary."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def mock_pipe_divergent(**kwargs):
        payload = _sample_sentiment_payload(reddit_count=1, youtube_count=0)
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        # Return summary with divergent model name
        mismatched_summary = dict(payload)
        mismatched_summary["model_name"] = "unauthorized_model_variant"
        return mismatched_summary

    monkeypatch.setattr("app.run_sentiment_pipeline", mock_pipe_divergent)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None
    assert not canonical_file.exists()


# --- Cleanup Failure Handling Tests ---


def test_sentiment_stage_validation_failure_survives_temp_unlink_oserror(tmp_path, monkeypatch):
    """Verify validation failure returns original error_category even if temp file unlink raises OSError."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_corrupt_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("{bad_json", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_sentiment_pipeline", write_corrupt_pipe)

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if ".tmp_sentiment_" in str(self):
            raise OSError("Permission denied: file locked by another process")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    # Must preserve the original corrupt_json category, NOT replace with execution_error
    assert entry["error_category"] == "corrupt_json"
    assert path is None
    assert not canonical_file.exists()


def test_sentiment_stage_promotion_failure_survives_temp_unlink_oserror(tmp_path, monkeypatch):
    """Verify atomic promotion failure returns promotion_error even if temp file unlink raises OSError."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_replace(src, dst):
        raise OSError("Permission denied during atomic os.replace")

    monkeypatch.setattr("app.os.replace", fail_replace)

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if ".tmp_sentiment_" in str(self):
            raise OSError("Permission denied during temp file cleanup")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        analyzer=_create_mock_sentiment_analyzer(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    # Must preserve promotion_error, NOT convert to execution_error
    assert entry["error_category"] == "promotion_error"
    assert path is None
    assert not canonical_file.exists()


def test_sentiment_stage_pipeline_exception_survives_temp_unlink_oserror(tmp_path, monkeypatch):
    """Verify pipeline execution exception is handled cleanly without escaping when temp unlink raises OSError."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("partial", encoding="utf-8")
        raise RuntimeError("Inference engine crash")

    monkeypatch.setattr("app.run_sentiment_pipeline", fail_pipe)

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if ".tmp_sentiment_" in str(self):
            raise OSError("Permission denied during temp cleanup")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert path is None
    assert not canonical_file.exists()


def test_sentiment_stage_sentinel_file_never_deleted(tmp_path, monkeypatch):
    """Verify stage never deletes any file located at a sentinel path."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    observed_sentinel: list[Path] = []

    def mock_pipe_with_sentinel_tracking(**kwargs):
        y_sentinel = Path(kwargs["youtube_input_path"])
        observed_sentinel.append(y_sentinel)
        # Simulate an unexpected file appearing at the sentinel path during execution
        y_sentinel.write_text("unexpected file at sentinel location", encoding="utf-8")

        out_path = Path(kwargs["output_path"])
        payload = _sample_sentiment_payload(reddit_count=1, youtube_count=0)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_sentiment_pipeline", mock_pipe_with_sentinel_tracking)

    entry, path = execute_sentiment_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert len(observed_sentinel) == 1
    # Verify the unexpected file at the sentinel location was NOT deleted by the stage!
    assert observed_sentinel[0].exists()
    assert observed_sentinel[0].read_text(encoding="utf-8") == "unexpected file at sentinel location"


# =====================================================================
# STEP 7.6C — Block 6C: Topic Output Validation Unit Tests
# =====================================================================


def _sample_topic_payload(
    reddit_count: int = 2,
    youtube_count: int = 1,
    outlier_count: int = 0,
) -> dict[str, Any]:
    total = reddit_count + youtube_count
    now_iso = "2026-11-07T14:00:00+00:00"

    if total == 0:
        return {
            "status": "success",
            "generated_at": now_iso,
            "embedding_model": "all-MiniLM-L6-v2",
            "total_documents_modeled": 0,
            "topic_info": [],
            "similarity_matrix": {"topic_ids": [], "matrix": [], "pairwise_similarities": {}},
            "trajectories": {},
            "cross_platform_divergence": {
                "overall_divergence": 0.0,
                "topic_adoption_gaps": {},
                "window_divergences": {},
            },
            "temporal_trends": {},
            "reddit_records": [],
            "youtube_records": [],
        }

    current_outliers = 0
    reddit_records: list[dict[str, Any]] = []
    for i in range(1, reddit_count + 1):
        if current_outliers < outlier_count:
            tid = -1
            status = "outlier"
            prob = 0.0
            kws: list[dict[str, Any]] = []
            current_outliers += 1
        else:
            tid = (i % 3)
            status = "assigned"
            prob = 0.85
            kws = [{"word": "ai", "weight": 0.8}, {"word": "model", "weight": 0.6}]

        reddit_records.append({
            "post_id": f"t3_topic_{i}",
            "title": f"Reddit Topic Title {i}",
            "text": f"Reddit Topic Body {i}",
            "title_cleaned": f"Reddit Cleaned Title {i}",
            "text_cleaned": f"Reddit Cleaned Body {i}",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": now_iso,
            "created_utc": 1730101200 + i * 3600,
            "source_platform": "reddit",
            "topic_id": tid,
            "topic_status": status,
            "topic_probability": prob,
            "topic_keywords": kws,
            "topic_modeled_at": now_iso,
        })

    youtube_records: list[dict[str, Any]] = []
    for i in range(1, youtube_count + 1):
        if current_outliers < outlier_count:
            tid = -1
            status = "outlier"
            prob = 0.0
            kws = []
            current_outliers += 1
        else:
            tid = (i % 3)
            status = "assigned"
            prob = 0.92
            kws = [{"word": "tech", "weight": 0.75}]

        youtube_records.append({
            "video_id": f"vid_topic_{i}",
            "title": f"YouTube Topic Title {i}",
            "description": f"YouTube Topic Desc {i}",
            "title_cleaned": f"YouTube Cleaned Title {i}",
            "description_cleaned": f"YouTube Cleaned Desc {i}",
            "detected_language": "English",
            "language_code": "en",
            "preprocessing_status": "success",
            "processed_at": now_iso,
            "published_at": now_iso,
            "source_platform": "youtube",
            "topic_id": tid,
            "topic_status": status,
            "topic_probability": prob,
            "topic_keywords": kws,
            "topic_modeled_at": now_iso,
        })

    actual_outliers = sum(1 for r in reddit_records + youtube_records if r["topic_id"] == -1)
    outlier_pct = round((actual_outliers / total) * 100.0, 2)

    topic_info = [
        {"topic_id": -1, "count": actual_outliers, "name": "-1_outlier", "keywords": []},
        {"topic_id": 0, "count": total - actual_outliers, "name": "0_ai_model", "keywords": [{"word": "ai", "weight": 0.8}]},
    ]

    return {
        "status": "success",
        "generated_at": now_iso,
        "embedding_model": "all-MiniLM-L6-v2",
        "total_documents_modeled": total,
        "outlier_count": actual_outliers,
        "outlier_pct": outlier_pct,
        "topic_info": topic_info,
        "similarity_matrix": {
            "topic_ids": [0],
            "matrix": [[1.0]],
            "pairwise_similarities": {},
        },
        "trajectories": {
            "0": {"velocity": 1.0, "acceleration": 0.0, "status": "emerging"}
        },
        "cross_platform_divergence": {
            "overall_divergence": 0.15,
            "topic_adoption_gaps": {"0": 0.1},
            "window_divergences": {"2026-11-07": 0.15},
        },
        "temporal_trends": {
            "daily": {
                "windows": {"2026-11-07": {"0": total - actual_outliers}},
                "topic_summaries": {},
                "all_topics": [0],
                "time_windows": ["2026-11-07"],
            },
            "weekly": {
                "windows": {"2026-W45": {"0": total - actual_outliers}},
                "topic_summaries": {},
                "all_topics": [0],
                "time_windows": ["2026-W45"],
            },
        },
        "reddit_records": reddit_records,
        "youtube_records": youtube_records,
    }


def test_topic_validation_valid_both_sources(tmp_path):
    """Verify topic validation passes when both Reddit and YouTube records exist."""
    payload = _sample_topic_payload(reddit_count=2, youtube_count=1, outlier_count=1)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=2,
        expected_youtube_count=1,
    )
    assert is_valid is True
    assert count == 3
    assert cat is None
    assert reason is None


def test_topic_validation_valid_reddit_only(tmp_path):
    """Verify topic validation passes when only Reddit records exist."""
    payload = _sample_topic_payload(reddit_count=3, youtube_count=0, outlier_count=0)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=3,
        expected_youtube_count=0,
    )
    assert is_valid is True
    assert count == 3
    assert cat is None
    assert reason is None


def test_topic_validation_valid_youtube_only(tmp_path):
    """Verify topic validation passes when only YouTube records exist."""
    payload = _sample_topic_payload(reddit_count=0, youtube_count=2, outlier_count=1)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=0,
        expected_youtube_count=2,
    )
    assert is_valid is True
    assert count == 2
    assert cat is None
    assert reason is None


def test_topic_validation_valid_zero_records(tmp_path):
    """Verify topic validation passes for zero-record schema."""
    payload = _sample_topic_payload(reddit_count=0, youtube_count=0)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=0,
        expected_youtube_count=0,
    )
    assert is_valid is True
    assert count == 0
    assert cat is None
    assert reason is None


def test_topic_validation_valid_all_outliers(tmp_path):
    """Verify topic validation passes when 100% of records are outliers."""
    payload = _sample_topic_payload(reddit_count=2, youtube_count=2, outlier_count=4)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=2,
        expected_youtube_count=2,
    )
    assert is_valid is True
    assert count == 4
    assert cat is None
    assert reason is None


@pytest.mark.parametrize("bad_summary", [None, "string", 123, [1, 2], True, False])
def test_topic_validation_invalid_summary_type(tmp_path, bad_summary):
    """Verify non-dict summary objects are rejected with topic_error."""
    payload = _sample_topic_payload(1, 0)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(bad_summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "topic_error"
    assert "must be a dictionary" in reason


@pytest.mark.parametrize("bad_status", ["failed", "error", "", None, 123])
def test_topic_validation_invalid_summary_status(tmp_path, bad_status):
    """Verify non-success summary status is rejected with topic_error."""
    payload = _sample_topic_payload(1, 0)
    file_path = tmp_path / "topics_out.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    bad_summary = dict(payload)
    bad_summary["status"] = bad_status

    is_valid, count, cat, reason = validate_topic_output(bad_summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "topic_error"
    assert "non-success status" in reason


@pytest.mark.parametrize("bad_path", [None, True, False, 123, []])
def test_topic_validation_invalid_path_type(bad_path):
    """Verify invalid path types are rejected with missing_output."""
    payload = _sample_topic_payload(1, 0)
    is_valid, count, cat, reason = validate_topic_output(payload, bad_path)
    assert is_valid is False
    assert count == 0
    assert cat == "missing_output"
    assert "file path is invalid" in reason


def test_topic_validation_temp_file_does_not_exist(tmp_path):
    """Verify missing temp file is rejected with missing_output."""
    payload = _sample_topic_payload(1, 0)
    non_existent = tmp_path / "does_not_exist.json"

    is_valid, count, cat, reason = validate_topic_output(payload, non_existent)
    assert is_valid is False
    assert count == 0
    assert cat == "missing_output"
    assert "does not exist" in reason


def test_topic_validation_temp_file_is_directory(tmp_path):
    """Verify directory path is rejected with missing_output."""
    payload = _sample_topic_payload(1, 0)
    is_valid, count, cat, reason = validate_topic_output(payload, tmp_path)
    assert is_valid is False
    assert count == 0
    assert cat == "missing_output"
    assert "does not exist" in reason


def test_topic_validation_empty_output_file(tmp_path):
    """Verify 0-byte file is rejected with empty_output."""
    payload = _sample_topic_payload(1, 0)
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, empty_file)
    assert is_valid is False
    assert count == 0
    assert cat == "empty_output"
    assert "0 bytes" in reason


def test_topic_validation_corrupt_json(tmp_path):
    """Verify malformed JSON file is rejected with corrupt_json."""
    payload = _sample_topic_payload(1, 0)
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ unclosed json content", encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, corrupt_file)
    assert is_valid is False
    assert count == 0
    assert cat == "corrupt_json"
    assert "malformed or unreadable JSON" in reason


@pytest.mark.parametrize("bad_root", [[1, 2, 3], "string", 42, True])
def test_topic_validation_root_not_dict(tmp_path, bad_root):
    """Verify JSON root that is not a dictionary is rejected with schema_violation."""
    file_path = tmp_path / "bad_root.json"
    file_path.write_text(json.dumps(bad_root), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary={"status": "success"},
        temp_file_path=file_path,
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "root must be a JSON dictionary" in reason


@pytest.mark.parametrize("missing_key", [
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
])
def test_topic_validation_missing_top_level_keys(tmp_path, missing_key):
    """Verify omission of any required top-level key is rejected with schema_violation."""
    payload = _sample_topic_payload(1, 0)
    summary = _sample_topic_payload(1, 0)
    del payload[missing_key]
    file_path = tmp_path / "missing_key.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(summary, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert f"missing required top-level field '{missing_key}'" in reason


def test_topic_validation_payload_status_not_success(tmp_path):
    """Verify payload status other than 'success' is rejected with schema_violation."""
    payload = _sample_topic_payload(1, 0)
    payload["status"] = "failed"
    file_path = tmp_path / "bad_status.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output({"status": "success"}, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "status must be 'success'" in reason


@pytest.mark.parametrize("bad_ts", ["not-a-date", "", 12345, None, "2026-99-99T99:99:99"])
def test_topic_validation_invalid_generated_at(tmp_path, bad_ts):
    """Verify invalid generated_at timestamp strings are rejected with schema_violation."""
    payload = _sample_topic_payload(1, 0)
    payload["generated_at"] = bad_ts
    file_path = tmp_path / "bad_ts.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "valid ISO-8601" in reason


@pytest.mark.parametrize("bad_model", ["", "   ", None, 123, []])
def test_topic_validation_invalid_embedding_model(tmp_path, bad_model):
    """Verify empty or non-string embedding_model is rejected with schema_violation."""
    payload = _sample_topic_payload(1, 0)
    payload["embedding_model"] = bad_model
    file_path = tmp_path / "bad_model.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "non-empty string" in reason


@pytest.mark.parametrize("bad_total", [True, False, 1.5, "1", -1, None])
def test_topic_validation_invalid_total_documents_modeled(tmp_path, bad_total):
    """Verify boolean, float, string, or negative total_documents_modeled is rejected with count_mismatch."""
    payload = _sample_topic_payload(1, 0)
    payload["total_documents_modeled"] = bad_total
    file_path = tmp_path / "bad_total.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "valid non-negative integer" in reason


def test_topic_validation_missing_outlier_fields_when_nonempty(tmp_path):
    """Verify non-empty payloads missing outlier_count or outlier_pct are rejected."""
    payload = _sample_topic_payload(1, 1)
    del payload["outlier_count"]
    file_path = tmp_path / "missing_outlier.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "missing required field 'outlier_count'" in reason

    payload = _sample_topic_payload(1, 1)
    del payload["outlier_pct"]
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "missing required field 'outlier_pct'" in reason


@pytest.mark.parametrize("bad_oc", [True, False, -1, 1.5, "0"])
def test_topic_validation_invalid_outlier_count_type(tmp_path, bad_oc):
    """Verify boolean, float, negative, or string outlier_count is rejected with count_mismatch."""
    payload = _sample_topic_payload(1, 0)
    payload["outlier_count"] = bad_oc
    file_path = tmp_path / "bad_oc.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "valid non-negative integer" in reason


@pytest.mark.parametrize("bad_op", [True, False, -0.1, 100.1, "50.0", float("nan"), float("inf")])
def test_topic_validation_invalid_outlier_pct(tmp_path, bad_op):
    """Verify invalid outlier_pct values are rejected with count_mismatch."""
    payload = _sample_topic_payload(1, 0)
    payload["outlier_pct"] = bad_op
    file_path = tmp_path / "bad_op.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "valid percentage in range" in reason


def test_topic_validation_outlier_count_mismatch(tmp_path):
    """Verify mismatch between outlier_count and actual outlier records is rejected."""
    payload = _sample_topic_payload(2, 0, outlier_count=1)
    # Tamper outlier_count to 0 while 1 record is an outlier
    payload["outlier_count"] = 0
    file_path = tmp_path / "outlier_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "does not match actual count of outlier records" in reason


def test_topic_validation_outlier_pct_mismatch(tmp_path):
    """Verify mismatch between outlier_pct and calculated percentage is rejected."""
    payload = _sample_topic_payload(2, 0, outlier_count=1)
    # Actual pct should be 50.0, tamper to 10.0
    payload["outlier_pct"] = 10.0
    file_path = tmp_path / "pct_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "does not match calculated percentage" in reason


@pytest.mark.parametrize("bad_temporal", ["string", [], None])
def test_topic_validation_temporal_trends_not_dict(tmp_path, bad_temporal):
    """Verify temporal_trends not being a dict is rejected with schema_violation."""
    payload = _sample_topic_payload(1, 0)
    payload["temporal_trends"] = bad_temporal
    file_path = tmp_path / "bad_temporal.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "temporal_trends' must be a dictionary" in reason


def test_topic_validation_temporal_trends_missing_daily_or_weekly(tmp_path):
    """Verify non-empty temporal_trends missing daily or weekly window is rejected."""
    payload = _sample_topic_payload(1, 0)
    del payload["temporal_trends"]["daily"]
    file_path = tmp_path / "missing_daily.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "missing required window 'daily'" in reason

    payload = _sample_topic_payload(1, 0)
    del payload["temporal_trends"]["weekly"]
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "missing required window 'weekly'" in reason


def test_topic_validation_similarity_matrix_invalid(tmp_path):
    """Verify structural defects in similarity_matrix are rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["similarity_matrix"] = "invalid"
    file_path = tmp_path / "bad_sim.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "similarity_matrix' must be a dictionary" in reason

    payload = _sample_topic_payload(1, 0)
    del payload["similarity_matrix"]["matrix"]
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "missing required key 'matrix'" in reason


def test_topic_validation_cross_platform_divergence_invalid(tmp_path):
    """Verify invalid cross_platform_divergence structure is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["cross_platform_divergence"] = []
    file_path = tmp_path / "bad_cpd.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "cross_platform_divergence' must be a dictionary" in reason

    payload = _sample_topic_payload(1, 0)
    payload["cross_platform_divergence"]["overall_divergence"] = -0.5
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "overall_divergence' must be a non-negative finite numeric value" in reason


def test_topic_validation_records_not_lists(tmp_path):
    """Verify non-list reddit_records or youtube_records are rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"] = {"post_id": "123"}
    file_path = tmp_path / "bad_records.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "must be JSON lists" in reason


def test_topic_validation_record_count_sum_mismatch(tmp_path):
    """Verify mismatch between total_documents_modeled and actual record list sizes."""
    payload = _sample_topic_payload(2, 1)
    payload["total_documents_modeled"] = 4  # 2 + 1 != 4
    file_path = tmp_path / "sum_mismatch.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "Sum of record counts does not match total_documents_modeled" in reason


def test_topic_validation_expected_reddit_count_mismatch(tmp_path):
    """Verify expected_reddit_count mismatch is rejected."""
    payload = _sample_topic_payload(2, 1)
    file_path = tmp_path / "topics.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        payload, file_path, expected_reddit_count=3, expected_youtube_count=1
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "Reddit record count (2) does not match expected count (3)" in reason


def test_topic_validation_expected_youtube_count_mismatch(tmp_path):
    """Verify expected_youtube_count mismatch is rejected."""
    payload = _sample_topic_payload(2, 1)
    file_path = tmp_path / "topics.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        payload, file_path, expected_reddit_count=2, expected_youtube_count=0
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "YouTube record count (1) does not match expected count (0)" in reason


@pytest.mark.parametrize("bad_expected", [True, False, -1, 1.5, "2"])
def test_topic_validation_invalid_expected_counts_type(tmp_path, bad_expected):
    """Verify non-integer or boolean expected counts are rejected with count_mismatch."""
    payload = _sample_topic_payload(1, 1)
    file_path = tmp_path / "topics.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        payload, file_path, expected_reddit_count=bad_expected
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "expected_reddit_count must be a valid non-negative integer" in reason

    is_valid, count, cat, reason = validate_topic_output(
        payload, file_path, expected_youtube_count=bad_expected
    )
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "expected_youtube_count must be a valid non-negative integer" in reason


def test_topic_validation_zero_records_nonempty_lists(tmp_path):
    """Verify zero-record payload containing records is rejected."""
    payload = _sample_topic_payload(0, 0)
    payload["reddit_records"] = [{"post_id": "r1"}]
    file_path = tmp_path / "zero_nonempty.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "count_mismatch"
    assert "Zero-record topic output must contain empty record lists" in reason


def test_topic_validation_zero_records_nonempty_topic_info(tmp_path):
    """Verify zero-record payload containing non-empty topic_info is rejected."""
    payload = _sample_topic_payload(0, 0)
    payload["topic_info"] = [{"topic_id": 0}]
    file_path = tmp_path / "zero_info.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "Zero-record topic output must contain empty topic_info list" in reason


def test_topic_validation_zero_records_nonempty_temporal_trends(tmp_path):
    """Verify zero-record payload containing non-empty temporal_trends is rejected."""
    payload = _sample_topic_payload(0, 0)
    payload["temporal_trends"] = {"daily": {}}
    file_path = tmp_path / "zero_temporal.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "Zero-record topic output must contain empty temporal_trends dictionary" in reason


def test_topic_validation_zero_records_nonempty_trajectories(tmp_path):
    """Verify zero-record payload containing non-empty trajectories is rejected."""
    payload = _sample_topic_payload(0, 0)
    payload["trajectories"] = {"0": {}}
    file_path = tmp_path / "zero_traj.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "Zero-record topic output must contain empty trajectories dictionary" in reason


def test_topic_validation_reddit_record_not_dict(tmp_path):
    """Verify non-dict Reddit records are rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0] = "not_a_dict"
    file_path = tmp_path / "bad_reddit_rec.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "Reddit record at index 0 is not a dictionary" in reason


@pytest.mark.parametrize("bad_id", ["", "   ", None, 123, []])
def test_topic_validation_reddit_record_missing_or_empty_post_id(tmp_path, bad_id):
    """Verify empty or non-string post_id is rejected with invalid_record_id."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["post_id"] = bad_id
    file_path = tmp_path / "bad_post_id.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "invalid_record_id"
    assert "missing, empty, or whitespace post_id" in reason


def test_topic_validation_reddit_record_duplicate_post_id(tmp_path):
    """Verify duplicate post_id in reddit_records is rejected with duplicate_record_id."""
    payload = _sample_topic_payload(2, 0)
    payload["reddit_records"][1]["post_id"] = payload["reddit_records"][0]["post_id"]
    file_path = tmp_path / "dup_reddit.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "duplicate_record_id"
    assert "Duplicate post_id detected in reddit_records" in reason


def test_topic_validation_reddit_record_invalid_source_platform(tmp_path):
    """Verify Reddit record with wrong source_platform is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["source_platform"] = "youtube"
    file_path = tmp_path / "bad_source.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "invalid source_platform" in reason


def test_topic_validation_youtube_record_not_dict(tmp_path):
    """Verify non-dict YouTube records are rejected."""
    payload = _sample_topic_payload(0, 1)
    payload["youtube_records"][0] = "not_a_dict"
    file_path = tmp_path / "bad_youtube_rec.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "YouTube record at index 0 is not a dictionary" in reason


@pytest.mark.parametrize("bad_id", ["", "   ", None, 123, []])
def test_topic_validation_youtube_record_missing_or_empty_video_id(tmp_path, bad_id):
    """Verify empty or non-string video_id is rejected with invalid_record_id."""
    payload = _sample_topic_payload(0, 1)
    payload["youtube_records"][0]["video_id"] = bad_id
    file_path = tmp_path / "bad_video_id.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "invalid_record_id"
    assert "missing, empty, or whitespace video_id" in reason


def test_topic_validation_youtube_record_duplicate_video_id(tmp_path):
    """Verify duplicate video_id in youtube_records is rejected with duplicate_record_id."""
    payload = _sample_topic_payload(0, 2)
    payload["youtube_records"][1]["video_id"] = payload["youtube_records"][0]["video_id"]
    file_path = tmp_path / "dup_youtube.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "duplicate_record_id"
    assert "Duplicate video_id detected in youtube_records" in reason


def test_topic_validation_youtube_record_invalid_source_platform(tmp_path):
    """Verify YouTube record with wrong source_platform is rejected."""
    payload = _sample_topic_payload(0, 1)
    payload["youtube_records"][0]["source_platform"] = "reddit"
    file_path = tmp_path / "bad_yt_source.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "invalid source_platform" in reason


@pytest.mark.parametrize("bad_tid", [True, False, 1.5, "0", None, []])
def test_topic_validation_record_invalid_topic_id_type(tmp_path, bad_tid):
    """Verify boolean, float, string, or non-int topic_id is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_id"] = bad_tid
    file_path = tmp_path / "bad_tid.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid topic_id" in reason


def test_topic_validation_record_invalid_negative_topic_id(tmp_path):
    """Verify negative topic_id less than -1 is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_id"] = -2
    file_path = tmp_path / "neg_tid.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid topic_id" in reason


def test_topic_validation_record_inconsistent_topic_status(tmp_path):
    """Verify topic_status inconsistency with topic_id is rejected."""
    # topic_id >= 0 but topic_status == 'outlier'
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_id"] = 0
    payload["reddit_records"][0]["topic_status"] = "outlier"
    file_path = tmp_path / "inconsistent1.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "inconsistent topic_status 'outlier' for topic_id 0" in reason

    # topic_id == -1 but topic_status == 'assigned'
    payload = _sample_topic_payload(1, 0, outlier_count=1)
    payload["reddit_records"][0]["topic_id"] = -1
    payload["reddit_records"][0]["topic_status"] = "assigned"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "inconsistent topic_status 'assigned' for topic_id -1" in reason

    # topic_status == 'error'
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_status"] = "error"
    file_path.write_text(json.dumps(payload), encoding="utf-8")
    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "inconsistent topic_status" in reason


@pytest.mark.parametrize("bad_prob", [True, False, -0.01, 1.01, float("nan"), float("inf"), "0.5", None])
def test_topic_validation_record_invalid_probability(tmp_path, bad_prob):
    """Verify invalid topic_probability values are rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_probability"] = bad_prob
    file_path = tmp_path / "bad_prob.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid topic_probability" in reason


@pytest.mark.parametrize("bad_kws", [
    "not_a_list",
    {"word": "tech"},
    ["string_item"],
    [{"weight": 0.5}],  # missing word
    [{"word": "", "weight": 0.5}],  # empty word
    [{"word": "   ", "weight": 0.5}],  # whitespace word
    [{"word": "ai", "weight": "heavy"}],  # non-numeric weight
    [{"word": "ai", "weight": True}],  # boolean weight
    [{"word": "ai", "weight": float("nan")}],  # NaN weight
])
def test_topic_validation_record_invalid_topic_keywords(tmp_path, bad_kws):
    """Verify malformed topic_keywords are rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_keywords"] = bad_kws
    file_path = tmp_path / "bad_kws.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "invalid topic_keywords" in reason


@pytest.mark.parametrize("bad_ts", ["not-a-date", "", 123, None, "2026-99-99T99:99:99"])
def test_topic_validation_record_invalid_topic_modeled_at(tmp_path, bad_ts):
    """Verify invalid topic_modeled_at timestamp is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["reddit_records"][0]["topic_modeled_at"] = bad_ts
    file_path = tmp_path / "bad_rec_ts.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid topic_modeled_at timestamp" in reason


def test_topic_validation_summary_payload_mismatch(tmp_path):
    """Verify discrepancy between in-memory summary and JSON payload is rejected."""
    payload = _sample_topic_payload(2, 1, outlier_count=1)
    file_path = tmp_path / "topics.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    # Mutate in-memory summary so it does not match persisted JSON
    tampered_summary = dict(payload)
    tampered_summary["outlier_count"] = 999

    is_valid, count, cat, reason = validate_topic_output(
        summary=tampered_summary,
        temp_file_path=file_path,
        expected_reddit_count=2,
        expected_youtube_count=1,
    )
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "does not match temporary JSON payload" in reason


def test_topic_validation_topic_info_valid_producer_entry(tmp_path):
    """Verify a valid producer-shaped topic entry in topic_info passes validation."""
    payload = _sample_topic_payload(2, 1, outlier_count=1)
    file_path = tmp_path / "valid_info.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is True
    assert count == 3
    assert cat is None
    assert reason is None


@pytest.mark.parametrize("missing_field", ["topic_id", "name", "count", "keywords"])
def test_topic_validation_topic_info_missing_required_fields(tmp_path, missing_field):
    """Verify topic_info entry missing any of the 4 required fields is rejected."""
    payload = _sample_topic_payload(1, 0)
    del payload["topic_info"][0][missing_field]
    file_path = tmp_path / "missing_info_field.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert f"missing required field '{missing_field}'" in reason


@pytest.mark.parametrize("bad_tid", [True, False, 1.5, "0", -2, None, []])
def test_topic_validation_topic_info_invalid_topic_id(tmp_path, bad_tid):
    """Verify topic_info entry with boolean, float, string, or < -1 topic_id is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["topic_info"][0]["topic_id"] = bad_tid
    file_path = tmp_path / "bad_info_tid.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid topic_id" in reason


@pytest.mark.parametrize("bad_name", ["", "   ", 123, None, []])
def test_topic_validation_topic_info_invalid_name(tmp_path, bad_name):
    """Verify topic_info entry with empty, whitespace-only, or non-string name is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["topic_info"][0]["name"] = bad_name
    file_path = tmp_path / "bad_info_name.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid or empty name" in reason


@pytest.mark.parametrize("bad_count", [-1, 1.5, True, False, "5", None])
def test_topic_validation_topic_info_invalid_count(tmp_path, bad_count):
    """Verify topic_info entry with negative, floating-point, boolean, or non-int count is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["topic_info"][0]["count"] = bad_count
    file_path = tmp_path / "bad_info_count.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid count" in reason


@pytest.mark.parametrize("bad_kws", [
    "not_a_list",
    {"word": "tech"},
    ["string_item"],
    [{"weight": 0.5}],
    [{"word": "", "weight": 0.5}],
    [{"word": "   ", "weight": 0.5}],
    [{"word": "ai", "weight": "heavy"}],
    [{"word": "ai", "weight": True}],
    [{"word": "ai", "weight": float("nan")}],
])
def test_topic_validation_topic_info_invalid_keywords(tmp_path, bad_kws):
    """Verify topic_info entry with invalid keywords structure is rejected."""
    payload = _sample_topic_payload(1, 0)
    payload["topic_info"][0]["keywords"] = bad_kws
    file_path = tmp_path / "bad_info_kws.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(payload, file_path)
    assert is_valid is False
    assert count == 0
    assert cat == "schema_violation"
    assert "has invalid keywords" in reason


def test_topic_validation_topic_info_zero_record_empty_remains_valid(tmp_path):
    """Verify an empty topic_info list remains valid for the zero-record payload."""
    payload = _sample_topic_payload(0, 0)
    assert payload["topic_info"] == []
    file_path = tmp_path / "zero_empty_info.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, cat, reason = validate_topic_output(
        summary=payload,
        temp_file_path=file_path,
        expected_reddit_count=0,
        expected_youtube_count=0,
    )
    assert is_valid is True
    assert count == 0
    assert cat is None
    assert reason is None




# =====================================================================
# STEP 7.6D — Block 6D: Topic Stage Execution Unit Tests
# =====================================================================


def _create_mock_topic_modeler(
    topic_sequence: Optional[list[int]] = None,
) -> MockTopicModeler:
    """Creates an isolated offline MockTopicModeler for testing."""
    return MockTopicModeler(topic_sequence=topic_sequence or [0, 1])


# --- Group 1: Provenance Validation (Fail Closed) ---


def test_topic_stage_missing_both_provenance_entries():
    """Verify execution fails closed with missing_upstream_provenance when both entries are None."""
    entry, path = execute_topic_stage(
        reddit_stage_entry=None,
        youtube_stage_entry=None,
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert entry["record_count"] == 0
    assert "Missing upstream preprocessing provenance" in entry["reason"]
    assert path is None


def test_topic_stage_missing_reddit_provenance():
    """Verify execution fails closed when reddit_stage_entry is None."""
    entry, path = execute_topic_stage(
        reddit_stage_entry=None,
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert entry["record_count"] == 0
    assert path is None


def test_topic_stage_missing_youtube_provenance():
    """Verify execution fails closed when youtube_stage_entry is None."""
    entry, path = execute_topic_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry=None,
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert entry["record_count"] == 0
    assert path is None


@pytest.mark.parametrize("bad_entry", ["not_a_dict", 123, [1, 2], True])
def test_topic_stage_malformed_provenance_not_dict(bad_entry):
    """Verify execution fails closed when provenance entry is not a dictionary."""
    entry1, _ = execute_topic_stage(
        reddit_stage_entry=bad_entry,
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry1["status"] == "failed"
    assert entry1["error_category"] == "invalid_upstream_provenance"

    entry2, _ = execute_topic_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry=bad_entry,
    )
    assert entry2["status"] == "failed"
    assert entry2["error_category"] == "invalid_upstream_provenance"


@pytest.mark.parametrize("bad_status", ["pending", "running", "unknown", "", None, 123])
def test_topic_stage_unsupported_status(bad_status):
    """Verify execution fails closed when provenance entry status is unsupported."""
    entry1, _ = execute_topic_stage(
        reddit_stage_entry={"status": bad_status, "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry1["status"] == "failed"
    assert entry1["error_category"] == "invalid_upstream_provenance"

    entry2, _ = execute_topic_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": bad_status, "record_count": 0},
    )
    assert entry2["status"] == "failed"
    assert entry2["error_category"] == "invalid_upstream_provenance"


@pytest.mark.parametrize("bad_count", [True, False, 1.5, -1, "10", None, []])
def test_topic_stage_invalid_record_count_types(bad_count):
    """Verify execution fails closed when provenance record_count is not a valid non-negative integer."""
    entry1, _ = execute_topic_stage(
        reddit_stage_entry={"status": "success", "record_count": bad_count},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry1["status"] == "failed"
    assert entry1["error_category"] == "invalid_upstream_provenance"

    entry2, _ = execute_topic_stage(
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "success", "record_count": bad_count},
    )
    assert entry2["status"] == "failed"
    assert entry2["error_category"] == "invalid_upstream_provenance"


# --- Group 2: Multi-Source Dependency and Skip Handling ---


def test_topic_stage_explicit_user_skip():
    """Verify explicit skip=True returns skipped entry immediately without inspecting provenance."""
    entry, path = execute_topic_stage(
        skip=True,
        reddit_stage_entry=None,
        youtube_stage_entry=None,
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert entry["record_count"] == 0
    assert entry["duration_seconds"] == 0.0
    assert entry["error_category"] is None
    assert path is None


def test_topic_stage_both_skipped():
    """Verify stage is skipped with all_sources_skipped when both sources skipped."""
    entry, path = execute_topic_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "all_sources_skipped"
    assert entry["record_count"] == 0
    assert path is None


def test_topic_stage_both_failed():
    """Verify stage is skipped with prior_stage_failure when both sources failed."""
    entry, path = execute_topic_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert entry["record_count"] == 0
    assert path is None


def test_topic_stage_failed_plus_skipped():
    """Verify stage is skipped with prior_stage_failure when one source failed and other skipped."""
    entry1, path1 = execute_topic_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry1["status"] == "skipped"
    assert entry1["reason"] == "prior_stage_failure"
    assert path1 is None

    entry2, path2 = execute_topic_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry2["status"] == "skipped"
    assert entry2["reason"] == "prior_stage_failure"
    assert path2 is None


def test_topic_stage_reddit_success_youtube_skipped(tmp_path):
    """Verify stage executes Reddit only when Reddit succeeded and YouTube skipped."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(payload["reddit_records"]) == 2
    assert len(payload["youtube_records"]) == 0


def test_topic_stage_youtube_success_reddit_skipped(tmp_path):
    """Verify stage executes YouTube only when YouTube succeeded and Reddit skipped."""
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=None,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(payload["reddit_records"]) == 0
    assert len(payload["youtube_records"]) == 2


def test_topic_stage_reddit_success_youtube_failed(tmp_path):
    """Verify stage executes Reddit only when Reddit succeeded and YouTube failed."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    assert path == canonical_file


def test_topic_stage_youtube_success_reddit_failed(tmp_path):
    """Verify stage executes YouTube only when YouTube succeeded and Reddit failed."""
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=None,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    assert path == canonical_file


def test_topic_stage_both_sources_successful(tmp_path):
    """Verify stage executes both Reddit and YouTube when both upstream stages succeeded."""
    r_file = tmp_path / "reddit_proc.json"
    y_file = tmp_path / "youtube_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 3
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert len(payload["reddit_records"]) == 2
    assert len(payload["youtube_records"]) == 1


def test_topic_stage_successful_zero_record_source(tmp_path):
    """Verify successful upstream source with record_count=0 is valid and processed."""
    r_file = tmp_path / "reddit_empty.json"
    r_file.write_text("[]", encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 0
    assert path == canonical_file
    assert canonical_file.exists()


# --- Group 3: Input Validation Helper Tests ---


@pytest.mark.parametrize("bad_path", [None, True, False, 123, [1, 2]])
def test_validate_active_topic_source_missing_or_invalid_path(bad_path):
    """Verify _validate_active_topic_source_file rejects None, bools, and invalid types."""
    valid, cat, reason = _validate_active_topic_source_file("Reddit", bad_path, 0)
    assert valid is False
    assert cat == "missing_processed_input"
    assert "missing or invalid" in reason


def test_validate_active_topic_source_file_not_found(tmp_path):
    """Verify validation fails when input file does not exist."""
    missing = tmp_path / "non_existent.json"
    valid, cat, reason = _validate_active_topic_source_file("Reddit", missing, 0)
    assert valid is False
    assert cat == "missing_processed_input"
    assert "does not exist" in reason


def test_validate_active_topic_source_directory_instead_of_file(tmp_path):
    """Verify validation fails when directory is provided instead of a regular file."""
    valid, cat, reason = _validate_active_topic_source_file("YouTube", tmp_path, 0)
    assert valid is False
    assert cat == "missing_processed_input"
    assert "does not exist" in reason


def test_validate_active_topic_source_zero_byte_input(tmp_path):
    """Verify validation fails with empty_input when file is 0 bytes."""
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("Reddit", empty, 0)
    assert valid is False
    assert cat == "empty_input"
    assert "empty (0 bytes)" in reason


def test_validate_active_topic_source_corrupt_json(tmp_path):
    """Verify validation fails with corrupt_json when file contains malformed JSON."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{broken", encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("Reddit", bad_json, 0)
    assert valid is False
    assert cat == "corrupt_json"
    assert "malformed or unreadable JSON" in reason


def test_validate_active_topic_source_root_not_list(tmp_path):
    """Verify validation fails with schema_violation when JSON root is not a list."""
    dict_root = tmp_path / "dict_root.json"
    dict_root.write_text(json.dumps({"data": []}), encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("Reddit", dict_root, 0)
    assert valid is False
    assert cat == "schema_violation"
    assert "root must be a list" in reason


def test_validate_active_topic_source_record_not_dict(tmp_path):
    """Verify validation fails with schema_violation when record element is not a dict."""
    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text(json.dumps([{"post_id": "1"}, 42]), encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("Reddit", non_dict, 2)
    assert valid is False
    assert cat == "schema_violation"
    assert "is not a dictionary" in reason


def test_validate_active_topic_source_count_mismatch(tmp_path):
    """Verify validation fails with count_mismatch when count does not match expected_count."""
    records = tmp_path / "records.json"
    records.write_text(json.dumps([{"id": "1"}, {"id": "2"}]), encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("Reddit", records, 3)
    assert valid is False
    assert cat == "count_mismatch"
    assert "does not match upstream manifest count" in reason


def test_validate_active_topic_source_valid_empty(tmp_path):
    """Verify validation passes for valid empty list with expected count 0."""
    empty_list = tmp_path / "empty_list.json"
    empty_list.write_text("[]", encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("Reddit", empty_list, 0)
    assert valid is True
    assert cat is None
    assert reason is None


def test_validate_active_topic_source_valid_records(tmp_path):
    """Verify validation passes for valid list of dicts with matching count."""
    records_file = tmp_path / "valid.json"
    records_file.write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
    valid, cat, reason = _validate_active_topic_source_file("YouTube", records_file, 2)
    assert valid is True
    assert cat is None
    assert reason is None


# --- Group 4: Stage Execution Input Validation Tests ---


def test_topic_stage_reddit_missing_active_input():
    """Verify stage fails when Reddit is active but reddit_processed_path is None."""
    entry, path = execute_topic_stage(
        reddit_processed_path=None,
        youtube_processed_path=None,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_topic_stage_reddit_file_not_found(tmp_path):
    """Verify stage fails when Reddit processed file does not exist."""
    missing = tmp_path / "does_not_exist.json"
    entry, path = execute_topic_stage(
        reddit_processed_path=missing,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_topic_stage_reddit_directory_supplied(tmp_path):
    """Verify stage fails when Reddit path is a directory instead of a file."""
    entry, path = execute_topic_stage(
        reddit_processed_path=tmp_path,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_topic_stage_reddit_zero_byte_file(tmp_path):
    """Verify stage fails with empty_input when Reddit processed file is 0 bytes."""
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    entry, path = execute_topic_stage(
        reddit_processed_path=empty,
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "empty_input"
    assert path is None


def test_topic_stage_reddit_corrupt_json(tmp_path):
    """Verify stage fails with corrupt_json when Reddit processed file has invalid JSON."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    entry, path = execute_topic_stage(
        reddit_processed_path=corrupt,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert path is None


def test_topic_stage_reddit_non_list_root(tmp_path):
    """Verify stage fails with schema_violation when Reddit processed file root is a dict."""
    not_list = tmp_path / "dict.json"
    not_list.write_text(json.dumps({"data": "bad"}), encoding="utf-8")
    entry, path = execute_topic_stage(
        reddit_processed_path=not_list,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_topic_stage_reddit_non_dict_record(tmp_path):
    """Verify stage fails with schema_violation when a Reddit record is not a dictionary."""
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps(["string_element"]), encoding="utf-8")
    entry, path = execute_topic_stage(
        reddit_processed_path=bad_rec,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_topic_stage_reddit_count_mismatch(tmp_path):
    """Verify stage fails with count_mismatch when Reddit record count does not match manifest."""
    rec_file = tmp_path / "records.json"
    rec_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    entry, path = execute_topic_stage(
        reddit_processed_path=rec_file,
        reddit_stage_entry={"status": "success", "record_count": 3},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "count_mismatch"
    assert path is None


def test_topic_stage_youtube_missing_active_input():
    """Verify stage fails when YouTube is active but youtube_processed_path is None."""
    entry, path = execute_topic_stage(
        reddit_processed_path=None,
        youtube_processed_path=None,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_topic_stage_youtube_file_not_found(tmp_path):
    """Verify stage fails when YouTube processed file does not exist."""
    missing = tmp_path / "does_not_exist.json"
    entry, path = execute_topic_stage(
        youtube_processed_path=missing,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_processed_input"
    assert path is None


def test_topic_stage_youtube_zero_byte_file(tmp_path):
    """Verify stage fails with empty_input when YouTube processed file is 0 bytes."""
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    entry, path = execute_topic_stage(
        youtube_processed_path=empty,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "empty_input"
    assert path is None


def test_topic_stage_youtube_corrupt_json(tmp_path):
    """Verify stage fails with corrupt_json when YouTube processed file has invalid JSON."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    entry, path = execute_topic_stage(
        youtube_processed_path=corrupt,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert path is None


def test_topic_stage_youtube_non_list_root(tmp_path):
    """Verify stage fails with schema_violation when YouTube processed file root is a dict."""
    not_list = tmp_path / "dict.json"
    not_list.write_text(json.dumps({"data": "bad"}), encoding="utf-8")
    entry, path = execute_topic_stage(
        youtube_processed_path=not_list,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_topic_stage_youtube_non_dict_record(tmp_path):
    """Verify stage fails with schema_violation when a YouTube record is not a dictionary."""
    bad_rec = tmp_path / "bad_rec.json"
    bad_rec.write_text(json.dumps([12345]), encoding="utf-8")
    entry, path = execute_topic_stage(
        youtube_processed_path=bad_rec,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None


def test_topic_stage_youtube_count_mismatch(tmp_path):
    """Verify stage fails with count_mismatch when YouTube record count does not match manifest."""
    rec_file = tmp_path / "records.json"
    rec_file.write_text(json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8")
    entry, path = execute_topic_stage(
        youtube_processed_path=rec_file,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "count_mismatch"
    assert path is None


# --- Group 5: Inactive-Source Isolation / Sentinel Tests ---


def test_topic_stage_inactive_youtube_passes_sentinel(tmp_path, monkeypatch):
    """Verify Reddit-only execution passes non-existent sentinel path for YouTube and active path for Reddit."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    captured_args = {}

    def mock_run_pipe(**kwargs):
        captured_args.update(kwargs)
        out_path = Path(kwargs["output_path"])
        payload = _sample_topic_payload(reddit_count=1, youtube_count=0)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_topic_pipeline", mock_run_pipe)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert captured_args["reddit_input_path"] == r_file
    y_input = captured_args["youtube_input_path"]
    assert isinstance(y_input, Path)
    assert not y_input.exists()
    assert ".sentinel_inactive_youtube_" in y_input.name


def test_topic_stage_inactive_reddit_passes_sentinel(tmp_path, monkeypatch):
    """Verify YouTube-only execution passes non-existent sentinel path for Reddit and active path for YouTube."""
    y_file = tmp_path / "youtube_proc.json"
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    captured_args = {}

    def mock_run_pipe(**kwargs):
        captured_args.update(kwargs)
        out_path = Path(kwargs["output_path"])
        payload = _sample_topic_payload(reddit_count=0, youtube_count=1)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_topic_pipeline", mock_run_pipe)

    entry, path = execute_topic_stage(
        reddit_processed_path=None,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert captured_args["youtube_input_path"] == y_file
    r_input = captured_args["reddit_input_path"]
    assert isinstance(r_input, Path)
    assert not r_input.exists()
    assert ".sentinel_inactive_reddit_" in r_input.name


def test_topic_stage_inactive_source_does_not_load_stale_canonical_files(tmp_path):
    """Verify inactive sources do not accidentally load stale canonical processed data."""
    # Place a stale processed YouTube file in the canonical directory
    stale_youtube = tmp_path / "youtube_videos_processed.json"
    stale_youtube.write_text(
        json.dumps(_sample_processed_youtube_records(5)), encoding="utf-8"
    )

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "topic_modeling_results.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    # Check that canonical output contains ONLY Reddit records, not the 5 stale YouTube records
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert payload["total_documents_modeled"] == 1
    assert len(payload["youtube_records"]) == 0
    assert len(payload["reddit_records"]) == 1


def test_topic_stage_sentinel_file_never_deleted(tmp_path, monkeypatch):
    """Verify stage never deletes any file located at a sentinel path."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    observed_sentinel: list[Path] = []

    def mock_pipe_with_sentinel_tracking(**kwargs):
        y_sentinel = Path(kwargs["youtube_input_path"])
        observed_sentinel.append(y_sentinel)
        # Simulate an unexpected file appearing at the sentinel path during execution
        y_sentinel.write_text("unexpected file at sentinel location", encoding="utf-8")

        out_path = Path(kwargs["output_path"])
        payload = _sample_topic_payload(reddit_count=1, youtube_count=0)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_topic_pipeline", mock_pipe_with_sentinel_tracking)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert len(observed_sentinel) == 1
    # Verify the unexpected file at the sentinel location was NOT deleted by the stage!
    assert observed_sentinel[0].exists()
    assert observed_sentinel[0].read_text(encoding="utf-8") == "unexpected file at sentinel location"


# --- Group 6: Pipeline Execution, Output Validation, Atomic Promotion, Cleanup ---


def test_topic_stage_pipeline_exception_handling(tmp_path, monkeypatch):
    """Verify execution catches pipeline exceptions, cleans up temp file, and returns execution_error."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def blow_up(**kwargs):
        Path(kwargs["output_path"]).write_text("partial", encoding="utf-8")
        raise RuntimeError("GPU Driver Crashed / Out of Memory")

    monkeypatch.setattr("app.run_topic_pipeline", blow_up)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert "Topic execution encountered an unexpected error" in entry["reason"]
    assert "GPU" not in entry["reason"]
    assert path is None
    # Verify no temporary files remain
    tmp_files = list(tmp_path.glob(".tmp_topics_*"))
    assert len(tmp_files) == 0


def test_topic_stage_missing_temporary_output(tmp_path, monkeypatch):
    """Verify stage fails with missing_output when pipeline returns success without creating file."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    monkeypatch.setattr(
        "app.run_topic_pipeline",
        lambda **kwargs: {"status": "success", "total_documents_modeled": 1},
    )

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_output"
    assert path is None


def test_topic_stage_corrupt_temporary_output(tmp_path, monkeypatch):
    """Verify stage fails with corrupt_json when pipeline writes invalid JSON."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_corrupt(**kwargs):
        Path(kwargs["output_path"]).write_text("{malformed", encoding="utf-8")
        return {"status": "success", "total_documents_modeled": 1}

    monkeypatch.setattr("app.run_topic_pipeline", write_corrupt)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_invalid_output_schema(tmp_path, monkeypatch):
    """Verify stage fails with schema_violation when pipeline writes payload missing fields."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_bad_schema(**kwargs):
        payload = _sample_topic_payload(1, 0)
        del payload["temporal_trends"]
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return {"status": "success", "total_documents_modeled": 1}

    monkeypatch.setattr("app.run_topic_pipeline", write_bad_schema)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_output_count_mismatch(tmp_path, monkeypatch):
    """Verify stage fails with count_mismatch when output record count disagrees with upstream manifest."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_mismatch(**kwargs):
        payload = _sample_topic_payload(2, 0)
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_topic_pipeline", write_mismatch)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "count_mismatch"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_validation_exception_handling(tmp_path, monkeypatch):
    """Verify unexpected validation exceptions are caught safely and return execution_error."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def mock_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("{}", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_topic_pipeline", mock_pipe)

    def blow_up_validator(*args, **kwargs):
        raise TypeError("Unexpected validator internal error")

    monkeypatch.setattr("app.validate_topic_output", blow_up_validator)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert path is None
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_successful_atomic_promotion(tmp_path):
    """Verify multi-source execution atomically creates canonical output and returns success."""
    r_file = tmp_path / "reddit_proc.json"
    y_file = tmp_path / "youtube_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "success", "record_count": 1},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 3
    assert entry["error_category"] is None
    assert entry["reason"] is None
    assert entry["duration_seconds"] >= 0.0
    assert path == canonical_file
    assert canonical_file.exists()
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_atomic_replacement_failure(tmp_path, monkeypatch):
    """Verify atomic promotion failure cleans up temp file and returns promotion_error."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_replace(src, dst):
        raise OSError("Permission denied during atomic replace")

    monkeypatch.setattr("app.os.replace", fail_replace)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "promotion_error"
    assert "Atomic output promotion failed" in entry["reason"]
    assert path is None
    assert not canonical_file.exists()
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_preserves_existing_canonical_on_pipeline_failure(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is completely preserved when pipeline execution fails."""
    canonical_file = tmp_path / "out.json"
    original_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(original_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    def fail_pipe(**kwargs):
        raise RuntimeError("Pipeline failed midway")

    monkeypatch.setattr("app.run_topic_pipeline", fail_pipe)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert path is None
    # Existing canonical file is preserved exactly
    assert canonical_file.exists()
    assert canonical_file.read_text(encoding="utf-8") == original_content


def test_topic_stage_preserves_existing_canonical_on_validation_failure(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is completely preserved when output validation fails."""
    canonical_file = tmp_path / "out.json"
    original_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(original_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    def write_corrupted(**kwargs):
        Path(kwargs["output_path"]).write_text("{bad", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_topic_pipeline", write_corrupted)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert path is None
    assert canonical_file.exists()
    assert canonical_file.read_text(encoding="utf-8") == original_content


def test_topic_stage_preserves_existing_canonical_on_promotion_failure(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is preserved when atomic promotion fails."""
    canonical_file = tmp_path / "out.json"
    original_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(original_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("Permission denied")

    monkeypatch.setattr("app.os.replace", fail_replace)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert path is None
    assert canonical_file.exists()
    assert canonical_file.read_text(encoding="utf-8") == original_content


def test_topic_stage_existing_canonical_replaced_only_after_successful_validation(tmp_path, monkeypatch):
    """Verify pre-existing canonical file is replaced only when validation succeeds."""
    canonical_file = tmp_path / "out.json"
    legacy_content = json.dumps({"legacy": "data"})
    canonical_file.write_text(legacy_content, encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    # Step 1: Validation failure keeps canonical file untouched
    def write_bad(**kwargs):
        Path(kwargs["output_path"]).write_text("{corrupt", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_topic_pipeline", write_bad)

    entry1, path1 = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry1["status"] == "failed"
    assert canonical_file.read_text(encoding="utf-8") == legacy_content

    # Step 2: Successful validation replaces canonical file
    monkeypatch.undo()

    entry2, path2 = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry2["status"] == "success"
    assert path2 == canonical_file
    new_payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert new_payload["total_documents_modeled"] == 1
    assert "legacy" not in new_payload


def test_topic_stage_valid_zero_record_output_successfully_promoted(tmp_path):
    """Verify zero-record output from both active sources is valid and promoted."""
    r_file = tmp_path / "reddit_empty.json"
    y_file = tmp_path / "youtube_empty.json"
    r_file.write_text("[]", encoding="utf-8")
    y_file.write_text("[]", encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 0
    assert path == canonical_file
    assert canonical_file.exists()
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert payload["total_documents_modeled"] == 0


def test_topic_stage_default_canonical_path_resolution(tmp_path):
    """Verify default canonical path resolves to processed_data_path / topic_modeling_results.json."""
    config_mock = type("ConfigMock", (), {"processed_data_path": tmp_path})()

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=None,
        config=config_mock,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    expected_path = tmp_path / "topic_modeling_results.json"
    assert path == expected_path
    assert expected_path.exists()


def test_topic_stage_diagnostics_sanitized(tmp_path):
    """Verify failure diagnostics remain sanitized without leaking filesystem paths or credentials."""
    secret_dir = tmp_path / "secret_customer_token"
    secret_dir.mkdir(parents=True, exist_ok=True)
    bad_file = secret_dir / "bad.json"
    bad_file.write_text("{broken", encoding="utf-8")

    entry, path = execute_topic_stage(
        reddit_processed_path=bad_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "corrupt_json"
    assert "secret_customer_token" not in entry["reason"]
    assert str(tmp_path) not in entry["reason"]


def test_topic_stage_in_memory_mismatch_prevents_promotion(tmp_path, monkeypatch):
    """Verify stage aborts and does not promote output if pipeline returns mismatched summary."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def mock_pipe_divergent(**kwargs):
        payload = _sample_topic_payload(reddit_count=1, youtube_count=0)
        Path(kwargs["output_path"]).write_text(json.dumps(payload), encoding="utf-8")
        # Return summary with divergent embedding model name
        mismatched_summary = dict(payload)
        mismatched_summary["embedding_model"] = "unauthorized_model_variant"
        return mismatched_summary

    monkeypatch.setattr("app.run_topic_pipeline", mock_pipe_divergent)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "schema_violation"
    assert path is None
    assert not canonical_file.exists()


def test_topic_stage_temp_file_cleanup_after_success(tmp_path):
    """Verify temporary files are completely cleaned up after successful execution."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_temp_file_cleanup_after_failure(tmp_path, monkeypatch):
    """Verify temporary files are cleaned up after execution failure."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("partial data", encoding="utf-8")
        raise RuntimeError("Failure during execution")

    monkeypatch.setattr("app.run_topic_pipeline", fail_pipe)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert len(list(tmp_path.glob(".tmp_topics_*"))) == 0


def test_topic_stage_validation_failure_survives_temp_unlink_oserror(tmp_path, monkeypatch):
    """Verify validation failure returns original error_category even if temp file unlink raises OSError."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def write_corrupt_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("{bad_json", encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr("app.run_topic_pipeline", write_corrupt_pipe)

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if ".tmp_topics_" in str(self):
            raise OSError("Permission denied: file locked by another process")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    # Must preserve the original corrupt_json category, NOT replace with execution_error
    assert entry["error_category"] == "corrupt_json"
    assert path is None
    assert not canonical_file.exists()


def test_topic_stage_promotion_failure_survives_temp_unlink_oserror(tmp_path, monkeypatch):
    """Verify atomic promotion failure returns promotion_error even if temp file unlink raises OSError."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_replace(src, dst):
        raise OSError("Permission denied during atomic os.replace")

    monkeypatch.setattr("app.os.replace", fail_replace)

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if ".tmp_topics_" in str(self):
            raise OSError("Permission denied during temp file cleanup")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    # Must preserve promotion_error, NOT convert to execution_error
    assert entry["error_category"] == "promotion_error"
    assert path is None
    assert not canonical_file.exists()


def test_topic_stage_pipeline_exception_survives_temp_unlink_oserror(tmp_path, monkeypatch):
    """Verify pipeline execution exception is handled cleanly without escaping when temp unlink raises OSError."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    def fail_pipe(**kwargs):
        Path(kwargs["output_path"]).write_text("partial", encoding="utf-8")
        raise RuntimeError("Inference engine crash")

    monkeypatch.setattr("app.run_topic_pipeline", fail_pipe)

    orig_unlink = Path.unlink

    def failing_unlink(self, *args, **kwargs):
        if ".tmp_topics_" in str(self):
            raise OSError("Permission denied during temp cleanup")
        return orig_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert path is None
    assert not canonical_file.exists()


# --- Group 7: Stage-Manifest and Block 6D Requirements 19-24 ---


def test_topic_stage_correct_source_counts(tmp_path):
    """Verify stage correctly reports and validates distinct Reddit and YouTube source counts."""
    r_file = tmp_path / "reddit_proc.json"
    y_file = tmp_path / "youtube_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(3)), encoding="utf-8")
    y_file.write_text(json.dumps(_sample_processed_youtube_records(2)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 3},
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 5
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert payload["total_documents_modeled"] == 5
    assert len(payload["reddit_records"]) == 3
    assert len(payload["youtube_records"]) == 2


def test_topic_stage_manifest_updates_on_success(tmp_path):
    """Verify stage manifest is correctly updated with topics stage success entry."""
    r_file = tmp_path / "reddit_proc.json"
    y_file = tmp_path / "youtube_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    y_file.write_text(json.dumps(_sample_processed_youtube_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "out.json"
    manifest_path = tmp_path / "manifest.json"

    stages = create_empty_stage_registry()
    stages["reddit_preprocessing"] = create_initial_stage_entry(
        status="success", record_count=2, duration_seconds=0.1
    )
    stages["youtube_preprocessing"] = create_initial_stage_entry(
        status="success", record_count=1, duration_seconds=0.1
    )

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=y_file,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry=stages["reddit_preprocessing"],
        youtube_stage_entry=stages["youtube_preprocessing"],
    )
    assert entry["status"] == "success"

    # Update manifest stage entry
    stages["topics"] = entry
    manifest = build_execution_manifest(stages=stages, run_id="test_topics_success_run")

    assert manifest["stages"]["topics"]["status"] == "success"
    assert manifest["stages"]["topics"]["record_count"] == 3
    assert manifest["stages"]["topics"]["error_category"] is None
    assert manifest["stages"]["topics"]["reason"] is None
    assert manifest["summary"]["stages_succeeded"] >= 3
    assert persist_manifest(manifest, manifest_path) is True
    assert manifest_path.exists()


def test_topic_stage_no_success_manifest_update_on_failure(tmp_path):
    """Verify stage failure does not mark topics as successful in manifest."""
    stages = create_empty_stage_registry()
    stages["reddit_preprocessing"] = create_initial_stage_entry(
        status="success", record_count=1, duration_seconds=0.1
    )
    stages["youtube_preprocessing"] = create_initial_stage_entry(
        status="skipped", record_count=0, duration_seconds=0.0
    )

    # Missing file causes failure
    entry, path = execute_topic_stage(
        reddit_processed_path=tmp_path / "missing_file.json",
        canonical_output_path=tmp_path / "out.json",
        reddit_stage_entry=stages["reddit_preprocessing"],
        youtube_stage_entry=stages["youtube_preprocessing"],
    )
    assert entry["status"] == "failed"

    # Update manifest with failed entry
    stages["topics"] = entry
    manifest = build_execution_manifest(stages=stages, run_id="test_topics_fail_run")

    assert manifest["stages"]["topics"]["status"] == "failed"
    assert manifest["stages"]["topics"]["record_count"] == 0
    assert manifest["stages"]["topics"]["error_category"] == "missing_processed_input"
    assert manifest["summary"]["stages_failed"] >= 1
    assert manifest["overall_outcome"] == "failed"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE


def test_topic_stage_stale_default_input_fallback_prevention(tmp_path):
    """Verify topic stage uses safe sentinel paths when a source is skipped to prevent loading stale default files."""
    stale_processed_dir = tmp_path / "data" / "processed"
    stale_processed_dir.mkdir(parents=True, exist_ok=True)
    stale_youtube = stale_processed_dir / "youtube_videos_processed.json"
    stale_youtube.write_text(json.dumps(_sample_processed_youtube_records(5)), encoding="utf-8")

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical_file = stale_processed_dir / "topic_modeling_results.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert payload["total_documents_modeled"] == 2
    assert len(payload["reddit_records"]) == 2
    # Crucial: stale YouTube data was NOT loaded
    assert len(payload["youtube_records"]) == 0


def test_topic_stage_stale_default_output_fallback_prevention(tmp_path, monkeypatch):
    """Verify topic pipeline is explicitly passed a temporary output path and cannot fall back to default output."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "topic_modeling_results.json"

    captured_output_paths: list[Any] = []

    def mock_pipe(**kwargs):
        captured_output_paths.append(kwargs.get("output_path"))
        out_path = Path(kwargs["output_path"])
        payload = _sample_topic_payload(reddit_count=1, youtube_count=0)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_topic_pipeline", mock_pipe)

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical_file,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert len(captured_output_paths) == 1
    passed_output = captured_output_paths[0]
    assert passed_output is not None
    assert isinstance(passed_output, Path)
    assert passed_output != canonical_file
    assert ".tmp_topics_" in passed_output.name
    assert passed_output.parent == canonical_file.parent


def test_topic_stage_no_dependency_on_stage5_sentiment(tmp_path):
    """Verify execute_topic_stage does not depend on sentiment stage or any Stage 5 output."""
    import inspect

    sig = inspect.signature(execute_topic_stage)
    for param_name in sig.parameters:
        assert "sentiment" not in param_name.lower(), (
            f"execute_topic_stage must not depend on Stage 5 sentiment; found parameter {param_name}"
        )

    # Execute topic stage with no sentiment data, and sentiment stage failed in the registry
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical_file = tmp_path / "topic_modeling_results.json"

    entry, path = execute_topic_stage(
        reddit_processed_path=r_file,
        youtube_processed_path=None,
        canonical_output_path=canonical_file,
        modeler=_create_mock_topic_modeler(),
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 1
    assert path == canonical_file
    assert canonical_file.exists()



# =====================================================================
# Stage 13B Test Helpers & Stage Adapter Test Suites (Stages 7 - 12)
# =====================================================================

from datetime import datetime, timezone


def _sample_graph_success_payload(num_nodes: int = 2) -> dict[str, Any]:
    nodes = [{"node_id": f"u{i}", "community_id": 0} for i in range(num_nodes)]
    edges = [{"source": "u0", "target": "u1", "weight": 1}] if num_nodes > 1 else []
    return {
        "schema_version": "2.0",
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "reddit_input_file": "reddit.json",
            "reddit_status": "loaded",
            "reddit_records_count": num_nodes,
            "youtube_input_file": "youtube.json",
            "youtube_status": "loaded",
            "youtube_records_count": 0,
        },
        "graph_metrics": {
            "total_users": num_nodes,
            "total_interactions": len(edges),
            "density": 0.5 if num_nodes > 1 else 0.0,
            "communities_count": 1 if num_nodes > 0 else 0,
            "is_directed": True,
        },
        "ranked_influencers": [{"node_id": "u0", "rank": 1, "score": 1.0}] if num_nodes > 0 else [],
        "communities": {f"u{i}": 0 for i in range(num_nodes)},
        "community_distributions": {"0": num_nodes} if num_nodes > 0 else {},
        "nodes": nodes,
        "edges": edges,
        "cypher_export": {
            "user_query": "UNWIND $batch AS user MERGE (u:User {node_id: user.node_id})",
            "relationship_query": "UNWIND $batch AS rel MATCH (s:User {node_id: rel.source}), (t:User {node_id: rel.target}) MERGE (s)-[:INTERACTED]->(t)",
            "total_user_batches": 1 if num_nodes > 0 else 0,
            "total_relationship_batches": 1 if len(edges) > 0 else 0,
            "total_users": num_nodes,
            "total_relationships": len(edges),
        },
    }


def _sample_misinformation_success_payload(count: int = 2) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "status": "success",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_disclosure": "Development fixture disclosure",
        "provenance": {
            "reddit_input_file": "reddit.json",
            "reddit_status": "loaded",
            "reddit_records_count": count,
            "youtube_input_file": "youtube.json",
            "youtube_status": "empty",
            "youtube_records_count": 0,
        },
        "total_records_processed": count,
        "total_analyzed": count,
        "average_credibility_score": 0.85,
        "flagged_high_risk_count": 0,
        "risk_tier_distribution": {"low_risk": count},
        "veracity_label_distribution": {"mostly_true": count},
        "predictions": [
            {
                "id": f"p_{i}",
                "platform": "reddit",
                "veracity_label": "mostly_true",
                "credibility_score": 0.85,
                "risk_tier": "low_risk",
                "status": "success",
                "evidence_basis": "stylometric",
                "signals_detected": [],
                "assessment_rationale": "Test rationale",
                "limitations": "Test limitations",
                "stylometric_signals": {},
                "text_preview": f"Preview text {i}",
            }
            for i in range(count)
        ],
        "enriched_reddit_records": [],
        "enriched_youtube_records": [],
    }


def _sample_consolidated_envelope(count: int = 2, consolidation_status: str = "complete") -> dict[str, Any]:
    from tests.test_data_consolidator import _make_valid_artifact_payload
    payload = _make_valid_artifact_payload()
    payload["consolidation_status"] = consolidation_status
    if count == 0:
        payload["records"] = []
        payload["total_records"] = 0
        payload["metrics"]["total_records_consolidated"] = 0
        payload["metrics"]["complete_records"] = 0
        payload["metrics"]["platform_counts"] = {"reddit": 0, "youtube": 0}
        payload["provenance"]["reddit_records_count"] = 0
        payload["provenance"]["youtube_records_count"] = 0
    else:
        payload["total_records"] = len(payload.get("records", []))
    return payload


# ---------------------------------------------------------------------
# Status Normalization & Summary Arithmetic Tests
# ---------------------------------------------------------------------


def test_calculate_manifest_summary_with_partial_and_empty_statuses():
    """Verify calculate_manifest_summary correctly counts partial and empty stages with valid invariants."""
    stages = create_empty_stage_registry()
    # 7 success, 2 partial, 1 empty, 1 failed, 1 skipped = 12
    successes = ["reddit_ingestion", "youtube_ingestion", "reddit_preprocessing", "youtube_preprocessing", "sentiment", "topics", "kibana_validation"]
    for s in successes:
        stages[s] = create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.1)

    stages["graph"] = create_initial_stage_entry(status="partial", record_count=5, duration_seconds=0.2)
    stages["consolidation"] = create_initial_stage_entry(status="partial", record_count=5, duration_seconds=0.3)
    stages["neo4j_persistence"] = create_initial_stage_entry(status="empty", record_count=0, duration_seconds=0.05)
    stages["elasticsearch_indexing"] = create_initial_stage_entry(status="failed", record_count=0, duration_seconds=0.1, error_category="indexing_failed")
    stages["misinformation"] = create_initial_stage_entry(status="skipped", reason="stage_skipped_by_user")

    summary = calculate_manifest_summary(stages)
    assert summary["stages_succeeded"] == 7
    assert summary["stages_partial"] == 2
    assert summary["stages_empty"] == 1
    assert summary["stages_failed"] == 1
    assert summary["stages_skipped"] == 1
    assert summary["total_stages_attempted"] == 11
    assert summary["total_stages_evaluated"] == 12


def test_build_execution_manifest_infers_partial_outcome_when_partial_present():
    """Verify build_execution_manifest infers outcome='partial' and exit_code=1 when partial stages exist without failure."""
    stages = create_empty_stage_registry()
    for s in CANONICAL_STAGES[:-1]:
        stages[s] = create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.1)
    stages["graph"] = create_initial_stage_entry(status="partial", record_count=5, duration_seconds=0.2)

    manifest = build_execution_manifest(stages=stages)
    assert manifest["overall_outcome"] == "partial"
    assert manifest["exit_code"] == EXIT_STAGE_FAILURE
    assert manifest["summary"]["stages_partial"] == 1
    assert manifest["summary"]["stages_failed"] == 0


# ---------------------------------------------------------------------
# Stage 7 (Graph Analysis) Adapter Tests
# ---------------------------------------------------------------------


def test_graph_stage_explicit_user_skip():
    """Verify execute_graph_stage skips cleanly when skip=True."""
    entry, path = execute_graph_stage(skip=True)
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert path is None


def test_graph_stage_missing_both_provenance_entries():
    """Verify execute_graph_stage fails closed when upstream provenance is missing."""
    entry, path = execute_graph_stage(reddit_stage_entry=None, youtube_stage_entry=None)
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert path is None


def test_graph_stage_malformed_provenance_not_dict():
    """Verify execute_graph_stage fails closed when upstream stage entries are not dictionaries."""
    entry, path = execute_graph_stage(
        reddit_stage_entry="invalid",  # type: ignore
        youtube_stage_entry={"status": "success", "record_count": 5},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_graph_stage_unsupported_status():
    """Verify execute_graph_stage fails closed when upstream status is unsupported."""
    entry, path = execute_graph_stage(
        reddit_stage_entry={"status": "corrupt_status", "record_count": 5},
        youtube_stage_entry={"status": "success", "record_count": 5},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_graph_stage_invalid_record_count():
    """Verify execute_graph_stage fails closed when record count is negative or boolean."""
    entry, path = execute_graph_stage(
        reddit_stage_entry={"status": "success", "record_count": -5},
        youtube_stage_entry={"status": "success", "record_count": 5},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_graph_stage_both_skipped():
    """Verify execute_graph_stage skips when both upstream sources were skipped."""
    entry, path = execute_graph_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "all_sources_skipped"
    assert path is None


def test_graph_stage_both_failed():
    """Verify execute_graph_stage skips with prior_stage_failure when neither source succeeded."""
    entry, path = execute_graph_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert path is None


def test_graph_stage_reddit_success_youtube_skipped(tmp_path, monkeypatch):
    """Verify execute_graph_stage runs with Reddit only when YouTube is skipped."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical = tmp_path / "graph_results.json"

    def mock_pipe(**kwargs):
        out_path = Path(kwargs["output_path"])
        payload = _sample_graph_success_payload(num_nodes=2)
        payload["status"] = "partial"
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_graph_pipeline", mock_pipe)

    entry, path = execute_graph_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "partial"
    assert entry["record_count"] == 2
    assert path == canonical
    assert canonical.exists()


def test_graph_stage_inactive_source_passes_sentinel(tmp_path, monkeypatch):
    """Verify inactive YouTube source receives a non-existent sentinel path."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical = tmp_path / "graph_results.json"

    captured_paths = {}

    def mock_pipe(**kwargs):
        captured_paths["reddit"] = kwargs.get("reddit_input_path")
        captured_paths["youtube"] = kwargs.get("youtube_input_path")
        out_path = Path(kwargs["output_path"])
        payload = _sample_graph_success_payload(num_nodes=1)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_graph_pipeline", mock_pipe)

    execute_graph_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    yt_input = captured_paths["youtube"]
    assert yt_input is not None
    assert isinstance(yt_input, Path)
    assert not yt_input.exists()
    assert ".sentinel_inactive_youtube_" in yt_input.name


def test_graph_stage_pipeline_error_fails_closed(tmp_path, monkeypatch):
    """Verify execute_graph_stage fails closed when graph pipeline returns status='error'."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical = tmp_path / "graph_results.json"

    def mock_pipe(**kwargs):
        return {"status": "error", "error": "Graph construction failed"}

    monkeypatch.setattr("app.run_graph_pipeline", mock_pipe)

    entry, path = execute_graph_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert path is None
    assert not canonical.exists()


def test_graph_stage_successful_atomic_promotion(tmp_path, monkeypatch):
    """Verify execute_graph_stage atomically promotes the output and preserves it."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical = tmp_path / "graph_results.json"

    def mock_pipe(**kwargs):
        out_path = Path(kwargs["output_path"])
        payload = _sample_graph_success_payload(num_nodes=2)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_graph_pipeline", mock_pipe)

    entry, path = execute_graph_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical
    assert canonical.exists()


# ---------------------------------------------------------------------
# Stage 8 (Neo4j Persistence) Adapter Tests
# ---------------------------------------------------------------------


def test_neo4j_stage_explicit_user_skip():
    """Verify execute_neo4j_stage skips cleanly when skip=True."""
    entry, path = execute_neo4j_stage(skip=True)
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert path is None


def test_neo4j_stage_missing_graph_provenance():
    """Verify execute_neo4j_stage fails closed when upstream graph provenance is missing."""
    entry, path = execute_neo4j_stage(graph_stage_entry=None)
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert path is None


def test_neo4j_stage_invalid_provenance_type():
    """Verify execute_neo4j_stage fails closed when graph_stage_entry is not a dict."""
    entry, path = execute_neo4j_stage(graph_stage_entry="invalid")  # type: ignore
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_neo4j_stage_graph_skipped():
    """Verify execute_neo4j_stage skips when Stage 7 graph was skipped."""
    entry, path = execute_neo4j_stage(graph_stage_entry={"status": "skipped", "record_count": 0})
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_skipped"
    assert path is None


def test_neo4j_stage_graph_failed():
    """Verify execute_neo4j_stage skips with prior_stage_failure when Stage 7 graph failed."""
    entry, path = execute_neo4j_stage(graph_stage_entry={"status": "failed", "record_count": 0})
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert path is None


def test_neo4j_stage_empty_graph_no_file_is_empty_noop(tmp_path):
    """Verify empty graph status with non-existent file is handled as a successful empty no-op."""
    missing_file = tmp_path / "absent_graph.json"
    entry, path = execute_neo4j_stage(
        graph_artifact_path=missing_file,
        graph_stage_entry={"status": "empty", "record_count": 0},
    )
    assert entry["status"] == "empty"
    assert entry["record_count"] == 0
    assert path is None


def test_neo4j_stage_missing_artifact_fails(tmp_path):
    """Verify execute_neo4j_stage fails closed if artifact is missing for active graph."""
    missing_file = tmp_path / "absent_graph.json"
    entry, path = execute_neo4j_stage(
        graph_artifact_path=missing_file,
        graph_stage_entry={"status": "success", "record_count": 5},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "artifact_not_found"
    assert path is None


def test_neo4j_stage_successful_persistence(tmp_path, monkeypatch):
    """Verify execute_neo4j_stage records committed counts and reports success."""
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(_sample_graph_success_payload(2)), encoding="utf-8")

    def mock_persist(**kwargs):
        return {
            "status": "success",
            "persistence_outcome": "success",
            "committed_nodes": 2,
            "committed_relationships": 1,
            "duration_seconds": 0.05,
        }

    monkeypatch.setattr("app.run_neo4j_persistence", mock_persist)

    entry, path = execute_neo4j_stage(
        graph_artifact_path=graph_file,
        graph_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 3  # 2 nodes + 1 relationship
    assert path is None


def test_neo4j_stage_persistence_failure(tmp_path, monkeypatch):
    """Verify execute_neo4j_stage captures failure report and sets status='failed'."""
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(_sample_graph_success_payload(2)), encoding="utf-8")

    def mock_persist(**kwargs):
        return {
            "status": "failed",
            "persistence_outcome": "failed",
            "error_category": "connection_error",
            "error_message": "Connection refused by Neo4j bolt server",
            "duration_seconds": 0.05,
        }

    monkeypatch.setattr("app.run_neo4j_persistence", mock_persist)

    entry, path = execute_neo4j_stage(
        graph_artifact_path=graph_file,
        graph_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "connection_error"
    assert "refused" in entry["reason"]
    assert path is None


# ---------------------------------------------------------------------
# Stage 9 (Misinformation Detection) Adapter Tests
# ---------------------------------------------------------------------


def test_misinformation_stage_explicit_user_skip():
    """Verify execute_misinformation_stage skips cleanly when skip=True."""
    entry, path = execute_misinformation_stage(skip=True)
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert path is None


def test_misinformation_stage_missing_both_provenance_entries():
    """Verify execute_misinformation_stage fails closed when upstream provenance is missing."""
    entry, path = execute_misinformation_stage(reddit_stage_entry=None, youtube_stage_entry=None)
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert path is None


def test_misinformation_stage_malformed_provenance_not_dict():
    """Verify execute_misinformation_stage fails closed when provenance entry is not a dict."""
    entry, path = execute_misinformation_stage(
        reddit_stage_entry="bad",  # type: ignore
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_misinformation_stage_unsupported_status():
    """Verify execute_misinformation_stage fails closed when upstream status is invalid."""
    entry, path = execute_misinformation_stage(
        reddit_stage_entry={"status": "bad_stat", "record_count": 2},
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "invalid_upstream_provenance"
    assert path is None


def test_misinformation_stage_both_skipped():
    """Verify execute_misinformation_stage skips when both upstream sources were skipped."""
    entry, path = execute_misinformation_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "all_sources_skipped"
    assert path is None


def test_misinformation_stage_both_failed():
    """Verify execute_misinformation_stage skips when neither source succeeded."""
    entry, path = execute_misinformation_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert path is None


def test_misinformation_stage_reddit_success_youtube_skipped(tmp_path, monkeypatch):
    """Verify execute_misinformation_stage runs with Reddit only when YouTube is skipped."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical = tmp_path / "misinfo_results.json"

    def mock_pipe(**kwargs):
        out_path = Path(kwargs["output_path"])
        payload = _sample_misinformation_success_payload(count=2)
        payload["status"] = "partial"
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_misinformation_pipeline", mock_pipe)

    entry, path = execute_misinformation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "partial"
    assert entry["record_count"] == 2
    assert path == canonical
    assert canonical.exists()


def test_misinformation_stage_inactive_source_passes_sentinel(tmp_path, monkeypatch):
    """Verify inactive YouTube source receives a non-existent sentinel path."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical = tmp_path / "misinfo_results.json"

    captured_paths = {}

    def mock_pipe(**kwargs):
        captured_paths["reddit"] = kwargs.get("reddit_path")
        captured_paths["youtube"] = kwargs.get("youtube_path")
        out_path = Path(kwargs["output_path"])
        payload = _sample_misinformation_success_payload(count=1)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_misinformation_pipeline", mock_pipe)

    execute_misinformation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    yt_input = captured_paths["youtube"]
    assert yt_input is not None
    assert isinstance(yt_input, Path)
    assert not yt_input.exists()
    assert ".sentinel_inactive_youtube_" in yt_input.name


def test_misinformation_stage_pipeline_error_fails_closed(tmp_path, monkeypatch):
    """Verify execute_misinformation_stage fails closed when pipeline returns status='error'."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical = tmp_path / "misinfo_results.json"

    def mock_pipe(**kwargs):
        return {"status": "error", "error": "Misinformation model failed"}

    monkeypatch.setattr("app.run_misinformation_pipeline", mock_pipe)

    entry, path = execute_misinformation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "execution_error"
    assert path is None
    assert not canonical.exists()


def test_misinformation_stage_successful_atomic_promotion(tmp_path, monkeypatch):
    """Verify execute_misinformation_stage atomically promotes output and succeeds."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical = tmp_path / "misinfo_results.json"

    def mock_pipe(**kwargs):
        out_path = Path(kwargs["output_path"])
        payload = _sample_misinformation_success_payload(count=2)
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.run_misinformation_pipeline", mock_pipe)

    entry, path = execute_misinformation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical
    assert canonical.exists()


# ---------------------------------------------------------------------
# Stage 10 (Consolidation) Adapter Tests
# ---------------------------------------------------------------------


def test_consolidation_stage_explicit_user_skip():
    """Verify execute_consolidation_stage skips cleanly when skip=True."""
    entry, path = execute_consolidation_stage(skip=True)
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert path is None


def test_consolidation_stage_missing_provenance():
    """Verify execute_consolidation_stage fails closed when upstream base provenance is missing."""
    entry, path = execute_consolidation_stage(reddit_stage_entry=None, youtube_stage_entry=None)
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert path is None


def test_consolidation_stage_both_base_skipped():
    """Verify execute_consolidation_stage skips when both base sources were skipped."""
    entry, path = execute_consolidation_stage(
        reddit_stage_entry={"status": "skipped", "record_count": 0},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "all_sources_skipped"
    assert path is None


def test_consolidation_stage_both_base_failed():
    """Verify execute_consolidation_stage skips with prior_stage_failure when neither base source succeeded."""
    entry, path = execute_consolidation_stage(
        reddit_stage_entry={"status": "failed", "record_count": 0},
        youtube_stage_entry={"status": "failed", "record_count": 0},
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert path is None


def test_consolidation_stage_inactive_sources_pass_sentinels(tmp_path, monkeypatch):
    """Verify all inactive or missing sources receive non-existent sentinel paths."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    captured_kwargs = {}

    def mock_consolidate(**kwargs):
        captured_kwargs.update(kwargs)
        out_path = Path(kwargs["output_path"])
        payload = _sample_consolidated_envelope(count=1, consolidation_status="partial")
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
        sentiment_stage_entry={"status": "skipped", "record_count": 0},
        topic_stage_entry={"status": "failed", "record_count": 0},
        graph_stage_entry=None,
        misinformation_stage_entry=None,
    )

    for src in ["youtube_processed_path", "sentiment_path", "topics_path", "graph_path", "misinformation_path"]:
        p = captured_kwargs.get(src)
        assert p is not None
        assert isinstance(p, Path)
        assert not p.exists(), f"Source path {src} must not exist (sentinel invariant violation)"
        assert ".sentinel_inactive_" in p.name


def test_consolidation_stage_validation_failure_fails_closed(tmp_path, monkeypatch):
    """Verify execute_consolidation_stage fails closed when artifact validation fails."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(1)), encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    def mock_consolidate(**kwargs):
        out_path = Path(kwargs["output_path"])
        payload = {"invalid": "payload_missing_keys"}
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    entry, path = execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 1},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "validation_failed"
    assert path is None
    assert not canonical.exists()


def test_consolidation_stage_successful_promotion(tmp_path, monkeypatch):
    """Verify execute_consolidation_stage successfully validates and atomically promotes."""
    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text(json.dumps(_sample_processed_reddit_records(2)), encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    def mock_consolidate(**kwargs):
        out_path = Path(kwargs["output_path"])
        payload = _sample_consolidated_envelope(count=2, consolidation_status="complete")
        out_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    entry, path = execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "skipped", "record_count": 0},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == canonical
    assert canonical.exists()


def test_consolidation_stage_schema_v2_10_records_reported_correctly(tmp_path, monkeypatch):
    """Verify execute_consolidation_stage correctly extracts record count (10) from Schema-v2 metrics."""
    import copy
    from tests.test_data_consolidator import _make_valid_artifact_payload

    p = _make_valid_artifact_payload()
    rec_reddit = p["records"][0]
    rec_youtube = p["records"][1]

    records = []
    for i in range(5):
        r = copy.deepcopy(rec_reddit)
        r["document_id"] = f"reddit_r_{1000 + i}"
        r["native_id"] = f"r_{1000 + i}"
        records.append(r)
        y = copy.deepcopy(rec_youtube)
        y["document_id"] = f"youtube_y_{2000 + i}"
        y["native_id"] = f"y_{2000 + i}"
        records.append(y)

    p["records"] = records
    p["metrics"]["total_records_consolidated"] = 10
    p["metrics"]["complete_records"] = 10
    p["metrics"]["platform_counts"] = {"reddit": 5, "youtube": 5}
    p["provenance"]["reddit_records_count"] = 5
    p["provenance"]["youtube_records_count"] = 5
    # Explicitly ensure NO root-level total_records exists in payload
    p.pop("total_records", None)

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text("[]", encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    def mock_consolidate(**kwargs):
        out_path = Path(kwargs["output_path"])
        out_path.write_text(json.dumps(p), encoding="utf-8")
        return p

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    entry, path = execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 5},
        youtube_stage_entry={"status": "success", "record_count": 5},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 10
    assert path == canonical
    assert canonical.exists()


def test_consolidation_stage_schema_v2_zero_records_distinguishable_from_failure(tmp_path, monkeypatch):
    """Verify execute_consolidation_stage with 0 valid records reports empty status, not failed."""
    from tests.test_data_consolidator import _make_valid_artifact_payload

    p = _make_valid_artifact_payload()
    p["records"] = []
    p["metrics"]["total_records_consolidated"] = 0
    p["metrics"]["complete_records"] = 0
    p["metrics"]["platform_counts"] = {"reddit": 0, "youtube": 0}
    p["provenance"]["reddit_records_count"] = 0
    p["provenance"]["youtube_records_count"] = 0
    p.pop("total_records", None)

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text("[]", encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    def mock_consolidate(**kwargs):
        out_path = Path(kwargs["output_path"])
        out_path.write_text(json.dumps(p), encoding="utf-8")
        return p

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    entry, path = execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 0},
        youtube_stage_entry={"status": "success", "record_count": 0},
    )
    assert entry["status"] == "empty"
    assert entry["record_count"] == 0
    assert entry["error_category"] is None
    assert path == canonical


def test_consolidation_stage_missing_metrics_fails(tmp_path, monkeypatch):
    """Verify execute_consolidation_stage fails closed when payload metrics are absent."""
    from tests.test_data_consolidator import _make_valid_artifact_payload

    p = _make_valid_artifact_payload()
    del p["metrics"]

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text("[]", encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    def mock_consolidate(**kwargs):
        out_path = Path(kwargs["output_path"])
        out_path.write_text(json.dumps(p), encoding="utf-8")
        return p

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    entry, path = execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "failed"
    assert entry["record_count"] == 0
    assert entry["error_category"] in {"validation_failed", "invalid_metrics"}
    assert path is None


def test_consolidation_stage_inconsistent_metrics_count_fails(tmp_path, monkeypatch):
    """Verify execute_consolidation_stage fails closed when metrics count does not match records length."""
    from tests.test_data_consolidator import _make_valid_artifact_payload

    p = _make_valid_artifact_payload()
    p["metrics"]["total_records_consolidated"] = 99  # Inconsistent with len(records)==2

    r_file = tmp_path / "reddit_proc.json"
    r_file.write_text("[]", encoding="utf-8")
    canonical = tmp_path / "consolidated.json"

    def mock_consolidate(**kwargs):
        out_path = Path(kwargs["output_path"])
        out_path.write_text(json.dumps(p), encoding="utf-8")
        return p

    monkeypatch.setattr("app.consolidate_social_intelligence", mock_consolidate)

    entry, path = execute_consolidation_stage(
        reddit_processed_path=r_file,
        canonical_output_path=canonical,
        reddit_stage_entry={"status": "success", "record_count": 2},
        youtube_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "failed"
    assert entry["record_count"] == 0
    assert entry["error_category"] in {"validation_failed", "inconsistent_metrics"}
    assert path is None


# ---------------------------------------------------------------------
# Stage 11 (Elasticsearch Indexing) Adapter Tests
# ---------------------------------------------------------------------


def test_elasticsearch_indexing_stage_explicit_user_skip():
    """Verify execute_elasticsearch_indexing_stage skips cleanly when skip=True."""
    entry, path = execute_elasticsearch_indexing_stage(skip=True)
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert path is None


def test_elasticsearch_indexing_stage_missing_provenance():
    """Verify execute_elasticsearch_indexing_stage fails closed when consolidation provenance is missing."""
    entry, path = execute_elasticsearch_indexing_stage(consolidation_stage_entry=None)
    assert entry["status"] == "failed"
    assert entry["error_category"] == "missing_upstream_provenance"
    assert path is None


def test_elasticsearch_indexing_stage_consolidation_skipped():
    """Verify execute_elasticsearch_indexing_stage skips when Stage 10 was skipped."""
    entry, path = execute_elasticsearch_indexing_stage(
        consolidation_stage_entry={"status": "skipped", "record_count": 0}
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_skipped"
    assert path is None


def test_elasticsearch_indexing_stage_consolidation_failed():
    """Verify execute_elasticsearch_indexing_stage skips when Stage 10 failed."""
    entry, path = execute_elasticsearch_indexing_stage(
        consolidation_stage_entry={"status": "failed", "record_count": 0}
    )
    assert entry["status"] == "skipped"
    assert entry["reason"] == "prior_stage_failure"
    assert path is None


def test_elasticsearch_indexing_stage_missing_dataset_fails(tmp_path):
    """Verify execute_elasticsearch_indexing_stage fails closed if dataset is missing for active consolidation."""
    missing_file = tmp_path / "absent_consolidated.json"
    entry, path = execute_elasticsearch_indexing_stage(
        consolidated_path=missing_file,
        consolidation_stage_entry={"status": "success", "record_count": 5},
    )
    assert entry["status"] == "failed"
    assert entry["error_category"] == "artifact_not_found"
    assert path is None


def test_elasticsearch_indexing_stage_empty_dataset_handling(tmp_path):
    """Verify empty consolidation status with absent file is handled as empty no-op."""
    missing_file = tmp_path / "absent_consolidated.json"
    entry, path = execute_elasticsearch_indexing_stage(
        consolidated_path=missing_file,
        consolidation_stage_entry={"status": "empty", "record_count": 0},
    )
    assert entry["status"] == "empty"
    assert entry["record_count"] == 0
    assert path is None


def test_elasticsearch_indexing_stage_successful_indexing(tmp_path, monkeypatch):
    """Verify execute_elasticsearch_indexing_stage maps successful indexing outcome."""
    ds_file = tmp_path / "consolidated.json"
    ds_file.write_text(json.dumps(_sample_consolidated_envelope(2)), encoding="utf-8")

    def mock_indexing(**kwargs):
        return 0, {
            "status": "success",
            "indexed_count": 2,
            "duration_seconds": 0.12,
            "error_category": None,
            "error_message": None,
        }

    monkeypatch.setattr("app.run_indexing_pipeline", mock_indexing)

    entry, path = execute_elasticsearch_indexing_stage(
        consolidated_path=ds_file,
        consolidation_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "success"
    assert entry["record_count"] == 2
    assert path == ds_file


def test_elasticsearch_indexing_stage_partial_indexing(tmp_path, monkeypatch):
    """Verify execute_elasticsearch_indexing_stage maps partial_success to partial."""
    ds_file = tmp_path / "consolidated.json"
    ds_file.write_text(json.dumps(_sample_consolidated_envelope(2)), encoding="utf-8")

    def mock_indexing(**kwargs):
        return 1, {
            "status": "partial_success",
            "indexed_count": 1,
            "failed_count": 1,
            "duration_seconds": 0.15,
            "error_category": "partial_bulk_failure",
            "error_message": "1 document failed to index",
        }

    monkeypatch.setattr("app.run_indexing_pipeline", mock_indexing)

    entry, path = execute_elasticsearch_indexing_stage(
        consolidated_path=ds_file,
        consolidation_stage_entry={"status": "success", "record_count": 2},
    )
    assert entry["status"] == "partial"
    assert entry["record_count"] == 1
    assert "1 document failed" in entry["reason"]
    assert path == ds_file


# ---------------------------------------------------------------------
# Stage 12 (Kibana Dashboard Validation) Adapter Tests
# ---------------------------------------------------------------------


def test_kibana_validation_stage_explicit_user_skip():
    """Verify execute_kibana_validation_stage skips cleanly when skip=True."""
    entry, path = execute_kibana_validation_stage(skip=True)
    assert entry["status"] == "skipped"
    assert entry["reason"] == "stage_skipped_by_user"
    assert path is None


def test_kibana_validation_stage_successful_validation(tmp_path, monkeypatch):
    """Verify execute_kibana_validation_stage maps valid bundle result to success with object count."""
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
    assert path == bundle_file


def test_kibana_validation_stage_invalid_bundle_fails(tmp_path, monkeypatch):
    """Verify execute_kibana_validation_stage maps invalid bundle result to failure."""
    bundle_file = tmp_path / "dashboards.ndjson"
    bundle_file.write_text("{}", encoding="utf-8")

    def mock_val(path):
        return {
            "is_valid": False,
            "total_objects": 0,
            "errors": ["Corrupted NDJSON: invalid line 1"],
        }

    monkeypatch.setattr("app.validate_kibana_dashboard_bundle", mock_val)

    entry, path = execute_kibana_validation_stage(bundle_path=bundle_file)
    assert entry["status"] == "failed"
    assert entry["error_category"] == "bundle_validation_error"
    assert "Corrupted NDJSON" in entry["reason"]
    assert path is None


# =====================================================================
# Stage 13C: Unified Orchestration Runner and Run Isolation Tests
# =====================================================================

def test_generate_and_validate_run_id():
    """Verify generate_run_id produces format conforming to RUN_ID_REGEX."""
    from app import generate_run_id, validate_run_id
    r1 = generate_run_id()
    r2 = generate_run_id()
    assert r1 != r2
    assert validate_run_id(r1) is True
    assert validate_run_id(r2) is True
    assert validate_run_id("invalid_run_id") is False
    assert validate_run_id("") is False
    assert validate_run_id(None) is False
    assert validate_run_id(12345) is False


def test_is_pid_alive_helper():
    """Verify is_pid_alive returns True for current PID and False for non-existent."""
    import os
    from app import is_pid_alive
    cur = os.getpid()
    assert is_pid_alive(cur) is True
    assert is_pid_alive(9999999) is False
    assert is_pid_alive(-1) is False
    assert is_pid_alive(0) is False
    assert is_pid_alive("invalid") is False


def test_orchestration_lock_lifecycle(tmp_path):
    """Verify OrchestrationLock acquires, enforces exclusivity, and releases."""
    from app import OrchestrationLock, OrchestrationLockError, generate_run_id
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


def test_orchestration_lock_context_manager(tmp_path):
    """Verify OrchestrationLock releases automatically via context manager."""
    from app import OrchestrationLock, generate_run_id
    lock_path = tmp_path / ".cm.lock"
    rid = generate_run_id()
    with OrchestrationLock(lock_path=lock_path) as lock:
        lock.acquire(rid)
        assert lock_path.exists()
    assert not lock_path.exists()


def test_orchestration_lock_exception_release(tmp_path):
    """Verify OrchestrationLock releases lock even when exception is raised."""
    from app import OrchestrationLock, generate_run_id
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


def test_orchestration_lock_stale_lock_recovery(tmp_path):
    """Verify OrchestrationLock safely reclaims lock held by a dead PID."""
    import os
    from app import OrchestrationLock, generate_run_id
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


def test_calculate_manifest_summary_exact_5_5_1_1_scenario():
    """Verify calculate_manifest_summary with exact 5 success / 5 partial / 1 skipped / 1 failed scenario."""
    from app import (
        calculate_manifest_summary,
        create_empty_stage_registry,
        create_initial_stage_entry,
        build_execution_manifest,
        EXIT_STAGE_FAILURE,
    )
    stages = create_empty_stage_registry()

    # 5 Succeeded
    for s in [
        "reddit_ingestion",
        "reddit_preprocessing",
        "neo4j_persistence",
        "elasticsearch_indexing",
        "kibana_validation",
    ]:
        stages[s] = create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.1)

    # 5 Partial
    for p in ["sentiment", "topics", "graph", "misinformation", "consolidation"]:
        stages[p] = create_initial_stage_entry(status="partial", record_count=10, duration_seconds=0.2)

    # 1 Skipped
    stages["youtube_preprocessing"] = create_initial_stage_entry(status="skipped", reason="prior_stage_failure")

    # 1 Failed
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


def test_build_execution_manifest_intentional_skip_youtube_scenario():
    """Verify build_execution_manifest with 5 success / 5 partial / 2 skipped / 0 failed -> outcome partial, exit code 1."""
    from app import (
        calculate_manifest_summary,
        create_empty_stage_registry,
        create_initial_stage_entry,
        build_execution_manifest,
        EXIT_STAGE_FAILURE,
    )
    stages = create_empty_stage_registry()

    # 5 Succeeded
    for s in [
        "reddit_ingestion",
        "reddit_preprocessing",
        "neo4j_persistence",
        "elasticsearch_indexing",
        "kibana_validation",
    ]:
        stages[s] = create_initial_stage_entry(status="success", record_count=10, duration_seconds=0.1)

    # 5 Partial
    for p in ["sentiment", "topics", "graph", "misinformation", "consolidation"]:
        stages[p] = create_initial_stage_entry(status="partial", record_count=10, duration_seconds=0.2)

    # 2 Skipped
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


def test_promote_run_to_canonical_atomic(tmp_path):
    """Verify promote_run_to_canonical copies files atomically via temp files."""
    from app import promote_run_to_canonical
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


def test_run_orchestrated_pipeline_execution_order_and_artifacts(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline runs all 12 stages in sequence and isolates artifacts."""
    from app import run_orchestrated_pipeline, EXIT_SUCCESS, CANONICAL_STAGES

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


def test_run_orchestrated_pipeline_prerequisite_skipping(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline skips downstream stages when prerequisites fail."""
    from app import run_orchestrated_pipeline, EXIT_STAGE_FAILURE

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


def test_cli_argument_parser_run_all_rules():
    """Verify CLI argument parser enforces --run-dir and --promote-canonical rules."""
    from app import parse_cli_arguments

    # Valid --run-all with new options
    ns = parse_cli_arguments(["--run-all", "--promote-canonical", "--run-dir=data/custom_runs"])
    assert ns.run_all is True
    assert ns.promote_canonical is True
    assert ns.run_dir == Path("data/custom_runs")

    # Rejected without --run-all
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--promote-canonical"])

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--run-dir=data/custom_runs"])


def test_orchestration_lock_uncertain_lock_requires_manual_intervention(tmp_path):
    """Verify OrchestrationLock raises OrchestrationLockError on corrupt/uncertain locks rather than deleting."""
    from app import OrchestrationLock, OrchestrationLockError, generate_run_id
    lock_path = tmp_path / ".corrupt.lock"
    # Corrupt content that cannot be safely parsed
    lock_path.write_text("{ corrupt json ", encoding="utf-8")

    lock = OrchestrationLock(lock_path=lock_path)
    rid = generate_run_id()
    with pytest.raises(OrchestrationLockError) as exc_info:
        lock.acquire(rid)
    assert "could not be safely verified" in str(exc_info.value)
    # Crucially, the uncertain lock file must NOT have been blindly deleted
    assert lock_path.exists()


def test_orchestration_lock_invalid_pid_metadata_requires_manual_intervention(tmp_path):
    """Verify OrchestrationLock raises OrchestrationLockError on invalid PID metadata."""
    from app import OrchestrationLock, OrchestrationLockError, generate_run_id
    lock_path = tmp_path / ".invalid_pid.lock"
    lock_path.write_text(json.dumps({"pid": "not_an_int", "run_id": "test"}), encoding="utf-8")

    lock = OrchestrationLock(lock_path=lock_path)
    rid = generate_run_id()
    with pytest.raises(OrchestrationLockError) as exc_info:
        lock.acquire(rid)
    assert "could not be safely verified" in str(exc_info.value)
    assert lock_path.exists()


def test_run_orchestrated_pipeline_exact_5_5_1_1_scenario(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline with exact 5 success / 5 partial / 1 skipped / 1 failed -> outcome failed, exit code 1."""
    from app import run_orchestrated_pipeline, EXIT_STAGE_FAILURE

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


def test_run_orchestrated_pipeline_intentional_skip_youtube_end_to_end(tmp_path, monkeypatch):
    """Verify run_orchestrated_pipeline with --skip-youtube produces exact 5 success / 5 partial / 2 skipped / 0 failed."""
    from app import run_orchestrated_pipeline, EXIT_STAGE_FAILURE

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


def test_orchestration_resource_ownership_contract(tmp_path, monkeypatch):
    """Verify orchestrator allows runners to manage resources internally and passes manager=None, client=None."""
    from app import run_orchestrated_pipeline, EXIT_SUCCESS

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
    # Assert orchestrator explicitly preserved internal runner resource lifecycle
    assert observed_kwargs["neo4j_manager"] is None
    assert observed_kwargs["neo4j_factory"] is None
    assert observed_kwargs["es_client"] is None


def test_prevention_of_fallback_to_canonical_artifacts(tmp_path):
    """Verify Stage 8 and Stage 11 fail closed when current-run artifact is missing, refusing canonical fallback."""
    from app import execute_neo4j_stage, execute_elasticsearch_indexing_stage

    # Stage 8: graph artifact path is None
    e8, p8 = execute_neo4j_stage(
        graph_artifact_path=None,
        graph_stage_entry={"status": "success", "record_count": 10},
    )
    assert e8["status"] == "failed"
    assert e8["error_category"] == "missing_artifact"
    assert p8 is None

    # Stage 11: consolidated path is None
    e11, p11 = execute_elasticsearch_indexing_stage(
        consolidated_path=None,
        consolidation_stage_entry={"status": "success", "record_count": 10},
    )
    assert e11["status"] == "failed"
    assert e11["error_category"] == "missing_artifact"
    assert p11 is None


# ---------------------------------------------------------------------
# Stage 5 / Stage 6 Memory Guards & Sentiment Status Validation
# ---------------------------------------------------------------------

def test_check_orchestrator_stage_memory_unit(monkeypatch):
    """Verify check_orchestrator_stage_memory correctly evaluates safe, low, and unmeasurable memory."""
    from app import check_orchestrator_stage_memory

    # Case 1: Sufficient memory
    monkeypatch.setattr("app.get_available_memory_mb", lambda: 2048.0)
    is_safe, avail, err_cat, reason = check_orchestrator_stage_memory("sentiment", min_mb=150.0)
    assert is_safe is True
    assert avail == 2048.0
    assert err_cat is None
    assert reason is None

    # Case 2: Insufficient memory
    monkeypatch.setattr("app.get_available_memory_mb", lambda: 75.0)
    is_safe, avail, err_cat, reason = check_orchestrator_stage_memory("topics", min_mb=150.0)
    assert is_safe is False
    assert avail == 75.0
    assert err_cat == "insufficient_memory"
    assert "below safe threshold" in reason

    # Case 3: Unmeasurable memory (fail-closed)
    monkeypatch.setattr("app.get_available_memory_mb", lambda: -1.0)
    is_safe, avail, err_cat, reason = check_orchestrator_stage_memory("sentiment", min_mb=150.0)
    assert is_safe is False
    assert avail == -1.0
    assert err_cat == "memory_availability_unmeasurable"
    assert "could not be measured" in reason


def test_validate_sentiment_output_rejects_unpermitted_record_status(tmp_path):
    """Verify validate_sentiment_output fails closed when any record contains an unpermitted sentiment_status."""
    from app import validate_sentiment_output

    payload = _sample_sentiment_payload(reddit_count=1, youtube_count=0)
    payload["reddit_records"][0]["sentiment_status"] = "unrecognized_status"  # Malformed status!

    test_file = tmp_path / "sentiment_malformed.json"
    test_file.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, err_cat, reason = validate_sentiment_output(
        summary=payload,
        temp_file_path=test_file,
    )
    assert is_valid is False
    assert err_cat == "schema_violation"
    assert "missing or invalid sentiment_status" in reason


def test_validate_sentiment_output_accepts_all_permitted_record_statuses(tmp_path):
    """Verify validate_sentiment_output accepts all valid statuses: success, error, fallback, empty_text."""
    from app import validate_sentiment_output

    payload = _sample_sentiment_payload(reddit_count=2, youtube_count=2)
    payload["reddit_records"][0]["sentiment_status"] = "success"
    payload["reddit_records"][1]["sentiment_status"] = "error"
    payload["youtube_records"][0]["sentiment_status"] = "fallback"
    payload["youtube_records"][1]["sentiment_status"] = "empty_text"

    test_file = tmp_path / "sentiment_permitted.json"
    test_file.write_text(json.dumps(payload), encoding="utf-8")

    is_valid, count, err_cat, reason = validate_sentiment_output(
        summary=payload,
        temp_file_path=test_file,
    )
    assert is_valid is True
    assert count == 4
    assert err_cat is None
    assert reason is None



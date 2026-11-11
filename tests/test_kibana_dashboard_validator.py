"""Comprehensive Unit Tests for Kibana Dashboard Validator and Importer.

Validates:
1. Canonical social_intelligence_dashboards.ndjson bundle passes all checks.
2. Exact 41 saved objects, 5 dashboards, 1 Data View, 34 visual panels, 1 Markdown banner.
3. Accurate Data View pattern ('consolidated_intelligence') and primary time field ('timestamps.published_at').
4. Panel counts per dashboard (8, 7, 6, 8, 6).
5. All references resolve without dangling or mismatched IDs.
6. Canonical field bindings and strict exclusion of unindexed raw fields.
7. Correct metric formulas (influencer max score ranking, community cardinality of username).
8. Presence of LIAR methodological disclaimer on Misinformation dashboard.
9. Defensive rejection of corrupted NDJSON, incorrect counts, duplicate IDs, and invalid fields.
10. Mocked HTTP testing of the optional importer without live network access.
"""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from src.search.index_mapping import DEFAULT_INDEX_NAME
from src.search.kibana_dashboard_importer import import_kibana_dashboard_bundle
from src.search.kibana_dashboard_validator import (
    DEFAULT_BUNDLE_PATH,
    EXPECTED_TOTAL_OBJECTS,
    validate_kibana_dashboard_bundle,
)


# =====================================================================
# 1. Canonical Bundle Verification Tests
# =====================================================================

def test_canonical_kibana_bundle_passes_all_validations():
    """Verifies the actual canonical NDJSON bundle passes offline validation."""
    assert DEFAULT_BUNDLE_PATH.exists(), f"Canonical bundle file {DEFAULT_BUNDLE_PATH} does not exist"
    res = validate_kibana_dashboard_bundle(DEFAULT_BUNDLE_PATH)

    assert res["is_valid"] is True, f"Validation failed with errors: {res['errors']}"
    assert res["total_objects"] == EXPECTED_TOTAL_OBJECTS
    assert len(res["errors"]) == 0


def test_canonical_bundle_has_exact_object_counts_by_type():
    """Verifies exactly 1 Data View, 34 visualizations, 1 Markdown banner, and 5 dashboards."""
    res = validate_kibana_dashboard_bundle(DEFAULT_BUNDLE_PATH)
    assert res["is_valid"] is True

    counts = res["counts_by_type"]
    assert counts.get("index-pattern") == 1
    assert counts.get("dashboard") == 5
    # 34 Lens visualizations + 1 Markdown banner = 35 visual objects
    assert counts.get("lens") == 34
    assert counts.get("visualization") == 1
    assert (counts.get("lens", 0) + counts.get("visualization", 0)) == 35


def test_canonical_bundle_dashboards_have_exact_panel_counts():
    """Verifies all five dashboards have their exact required panel allocations."""
    res = validate_kibana_dashboard_bundle(DEFAULT_BUNDLE_PATH)
    assert res["is_valid"] is True

    dashboards = res["dashboards"]
    assert dashboards["dashboard-executive-overview"] == 8
    assert dashboards["dashboard-sentiment-intelligence"] == 7
    assert dashboards["dashboard-topic-intelligence"] == 6
    assert dashboards["dashboard-misinformation-intelligence"] == 8  # 7 visuals + 1 markdown
    assert dashboards["dashboard-network-intelligence"] == 6


def test_canonical_data_view_attributes():
    """Verifies Data View title is 'consolidated_intelligence' and time field is 'timestamps.published_at'."""
    with open(DEFAULT_BUNDLE_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    data_view = next(obj for obj in lines if obj.get("type") == "index-pattern")
    attrs = data_view["attributes"]

    assert attrs["title"] == DEFAULT_INDEX_NAME
    assert attrs["title"] == "consolidated_intelligence"
    assert attrs["timeFieldName"] == "timestamps.published_at"


def test_canonical_bundle_disclaimer_present_on_misinformation_dashboard():
    """Verifies the LIAR methodological disclosure banner is referenced on Dashboard 4."""
    with open(DEFAULT_BUNDLE_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    misinfo_dash = next(obj for obj in lines if obj.get("id") == "dashboard-misinformation-intelligence")
    ref_ids = [r["id"] for r in misinfo_dash["references"]]
    assert "markdown-misinfo-disclaimer" in ref_ids

    banner_obj = next(obj for obj in lines if obj.get("id") == "markdown-misinfo-disclaimer")
    vis_state = banner_obj["attributes"]["visState"]
    assert "LIAR" in vis_state
    assert "Not Verified Factual Truth" in vis_state


def test_canonical_bundle_influencer_and_community_metric_definitions():
    """Verifies influencer leaderboard ranks by max influencer_score and community sizes use username cardinality."""
    with open(DEFAULT_BUNDLE_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    influencer_panel = next(obj for obj in lines if obj.get("id") == "vis-net-top-influencers")
    y_spec = influencer_panel["attributes"]["state"]["y"]
    assert y_spec["accessor"] == "author.influencer_score"
    assert y_spec["agg"] == "max"

    community_panel = next(obj for obj in lines if obj.get("id") == "vis-net-community-sizes")
    c_x = community_panel["attributes"]["state"]["x"]
    c_y = community_panel["attributes"]["state"]["y"]
    assert c_x["accessor"] == "author.community_id"
    assert c_y["accessor"] == "author.username"
    assert c_y["agg"] == "cardinality"


# =====================================================================
# 2. Defensive Rejection Tests
# =====================================================================

@pytest.fixture
def canonical_objects() -> List[Dict[str, Any]]:
    """Loads all 41 canonical saved objects as dictionaries."""
    with open(DEFAULT_BUNDLE_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_ndjson(tmp_path: Path, objects: List[Dict[str, Any]]) -> Path:
    out = tmp_path / "test_dashboards.ndjson"
    with open(out, "w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj) + "\n")
    return out


def test_validator_rejects_missing_file(tmp_path: Path):
    """Verifies validator reports missing bundle file."""
    nonexistent = tmp_path / "missing.ndjson"
    res = validate_kibana_dashboard_bundle(nonexistent)
    assert res["is_valid"] is False
    assert any("not found" in err for err in res["errors"])


def test_validator_rejects_corrupted_json_syntax(tmp_path: Path, canonical_objects):
    """Verifies validator rejects invalid JSON syntax on any line."""
    bad_file = tmp_path / "corrupted.ndjson"
    with open(bad_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(canonical_objects[0]) + "\n")
        f.write("{ invalid json line : \n")
        for obj in canonical_objects[2:]:
            f.write(json.dumps(obj) + "\n")

    res = validate_kibana_dashboard_bundle(bad_file)
    assert res["is_valid"] is False
    assert any("invalid JSON syntax" in err for err in res["errors"])


def test_validator_rejects_incorrect_object_count(tmp_path: Path, canonical_objects):
    """Verifies validator rejects bundles with missing or excess objects."""
    # Omit 1 object (40 objects instead of 41)
    short_file = _write_ndjson(tmp_path, canonical_objects[:-1])
    res = validate_kibana_dashboard_bundle(short_file)
    assert res["is_valid"] is False
    assert any("Expected exactly 41" in err for err in res["errors"])


def test_validator_rejects_duplicate_saved_object_ids(tmp_path: Path, canonical_objects):
    """Verifies duplicate saved-object IDs are caught and rejected."""
    cloned = copy.deepcopy(canonical_objects)
    # Duplicate the first object's ID into the second object
    cloned[1]["id"] = cloned[0]["id"]
    dup_file = _write_ndjson(tmp_path, cloned)

    res = validate_kibana_dashboard_bundle(dup_file)
    assert res["is_valid"] is False
    assert any("Duplicate saved object ID" in err for err in res["errors"])


def test_validator_rejects_incorrect_data_view_title(tmp_path: Path, canonical_objects):
    """Verifies Data View title not exactly matching 'consolidated_intelligence' is rejected."""
    cloned = copy.deepcopy(canonical_objects)
    for obj in cloned:
        if obj.get("type") == "index-pattern":
            obj["attributes"]["title"] = "social-intelligence*"  # Disallowed wildcard

    bad_view_file = _write_ndjson(tmp_path, cloned)
    res = validate_kibana_dashboard_bundle(bad_view_file)
    assert res["is_valid"] is False
    assert any("Data View title must exactly match default index" in err for err in res["errors"])


def test_validator_rejects_incorrect_data_view_time_field(tmp_path: Path, canonical_objects):
    """Verifies primary timeFieldName not matching 'timestamps.published_at' is rejected."""
    cloned = copy.deepcopy(canonical_objects)
    for obj in cloned:
        if obj.get("type") == "index-pattern":
            obj["attributes"]["timeFieldName"] = "timestamps.collected_at"  # Incorrect time field

    bad_time_file = _write_ndjson(tmp_path, cloned)
    res = validate_kibana_dashboard_bundle(bad_time_file)
    assert res["is_valid"] is False
    assert any("timeFieldName must be 'timestamps.published_at'" in err for err in res["errors"])


def test_validator_rejects_dangling_dashboard_reference(tmp_path: Path, canonical_objects):
    """Verifies dashboard referencing a non-existent saved object ID is caught."""
    cloned = copy.deepcopy(canonical_objects)
    dash = next(obj for obj in cloned if obj.get("id") == "dashboard-executive-overview")
    dash["references"][0]["id"] = "non-existent-vis-id"

    dangling_file = _write_ndjson(tmp_path, cloned)
    res = validate_kibana_dashboard_bundle(dangling_file)
    assert res["is_valid"] is False
    assert any("dangling reference" in err for err in res["errors"])


def test_validator_rejects_unindexed_raw_content_field(tmp_path: Path, canonical_objects):
    """Verifies referencing unindexed raw content (content.title_raw) is rejected."""
    cloned = copy.deepcopy(canonical_objects)
    vis = next(obj for obj in cloned if obj.get("id") == "vis-exec-total-indexed")
    vis["attributes"]["state"]["metric"]["accessor"] = "content.title_raw"  # Disallowed

    bad_raw_file = _write_ndjson(tmp_path, cloned)
    res = validate_kibana_dashboard_bundle(bad_raw_file)
    assert res["is_valid"] is False
    assert any("disallowed unindexed raw field" in err for err in res["errors"])


def test_validator_rejects_non_canonical_invented_field(tmp_path: Path, canonical_objects):
    """Verifies invented or non-canonical fields are rejected."""
    cloned = copy.deepcopy(canonical_objects)
    vis = next(obj for obj in cloned if obj.get("id") == "vis-exec-total-indexed")
    vis["attributes"]["state"]["metric"]["accessor"] = "engagement.invented_shares"

    invented_file = _write_ndjson(tmp_path, cloned)
    res = validate_kibana_dashboard_bundle(invented_file)
    assert res["is_valid"] is False
    assert any("non-canonical field" in err for err in res["errors"])


def test_validator_rejects_missing_disclaimer_on_misinformation_dashboard(tmp_path: Path, canonical_objects):
    """Verifies omitting the methodological disclaimer banner from Dashboard 4 fails validation."""
    cloned = copy.deepcopy(canonical_objects)
    misinfo_dash = next(obj for obj in cloned if obj.get("id") == "dashboard-misinformation-intelligence")
    # Remove the disclaimer reference
    misinfo_dash["references"] = [r for r in misinfo_dash["references"] if r["id"] != "markdown-misinfo-disclaimer"]

    no_disclaimer_file = _write_ndjson(tmp_path, cloned)
    res = validate_kibana_dashboard_bundle(no_disclaimer_file)
    assert res["is_valid"] is False
    assert any("must include 'markdown-misinfo-disclaimer'" in err for err in res["errors"])


# =====================================================================
# 3. Optional Importer Unit Tests (Mocked HTTP)
# =====================================================================

def test_importer_stops_if_bundle_validation_fails(tmp_path: Path):
    """Verifies importer aborts before attempting network requests if validation fails."""
    bad_file = tmp_path / "bad.ndjson"
    bad_file.write_text("{ unclosed", encoding="utf-8")

    res = import_kibana_dashboard_bundle(bundle_path=bad_file, kibana_host="http://localhost:5601")
    assert res["status"] == "failed"
    assert any("Pre-import validation error" in d for d in res["diagnostics"])


def test_importer_rejects_invalid_scheme(canonical_objects, tmp_path: Path):
    """Verifies importer rejects non-HTTP(S) hosts without network calls."""
    bundle_file = _write_ndjson(tmp_path, canonical_objects)
    res = import_kibana_dashboard_bundle(bundle_path=bundle_file, kibana_host="ftp://localhost:5601")

    assert res["status"] == "failed"
    assert any("scheme must be http or https" in d for d in res["diagnostics"])


@patch("src.search.kibana_dashboard_importer.urlopen")
def test_importer_successful_mocked_import(mock_urlopen, canonical_objects, tmp_path: Path):
    """Verifies successful import against mocked Kibana 200 response."""
    bundle_file = _write_ndjson(tmp_path, canonical_objects)

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps({"success": True, "successCount": 41, "errors": []}).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    res = import_kibana_dashboard_bundle(
        bundle_path=bundle_file,
        kibana_host="http://localhost:5601",
        username="elastic",
        password="secret_password_here",
    )

    assert res["status"] == "success"
    assert res["success_count"] == 41
    assert res["errors_count"] == 0

    # Verify HTTP request construction
    assert mock_urlopen.call_count == 1
    req = mock_urlopen.call_args[0][0]
    assert req.get_method() == "POST"
    assert req.has_header("Kbn-xsrf")
    assert req.has_header("Authorization")
    # Password must never appear in return diagnostics
    assert "secret_password_here" not in str(res)


@patch("src.search.kibana_dashboard_importer.urlopen")
def test_importer_handles_http_error_cleanly(mock_urlopen, canonical_objects, tmp_path: Path):
    """Verifies HTTPError (e.g. 401 Unauthorized) is caught safely and does not leak secrets."""
    bundle_file = _write_ndjson(tmp_path, canonical_objects)

    mock_urlopen.side_effect = HTTPError(
        url="http://localhost:5601/api/saved_objects/_import",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b"Unauthorized"),
    )

    res = import_kibana_dashboard_bundle(
        bundle_path=bundle_file,
        kibana_host="http://localhost:5601",
        password="super_sensitive_token",
    )

    assert res["status"] == "failed"
    assert any("HTTP 401" in d for d in res["diagnostics"])
    assert "super_sensitive_token" not in str(res)


@patch("src.search.kibana_dashboard_importer.urlopen")
def test_importer_handles_connection_refused_cleanly(mock_urlopen, canonical_objects, tmp_path: Path):
    """Verifies URLError (connection refused) is handled with safe diagnostic."""
    bundle_file = _write_ndjson(tmp_path, canonical_objects)
    mock_urlopen.side_effect = URLError("Connection refused")

    res = import_kibana_dashboard_bundle(bundle_path=bundle_file, kibana_host="http://localhost:5601")
    assert res["status"] == "failed"
    assert any("Connection to Kibana host failed" in d for d in res["diagnostics"])


# =====================================================================
# 4. CLI Argument Parser & Mutual Exclusion Tests
# =====================================================================

# Ensure emoji module is available for importing app.py in environments lacking emoji
import sys
if "emoji" not in sys.modules:
    try:
        import emoji  # noqa: F401
    except ImportError:
        sys.modules["emoji"] = MagicMock()

from app import (
    EXIT_ORCHESTRATION_NOT_IMPLEMENTED,
    EXIT_STAGE_FAILURE,
    EXIT_SUCCESS as APP_EXIT_SUCCESS,
    main,
    parse_cli_arguments,
    run_validate_kibana_cli,
    run_orchestrated_pipeline,
)


def test_cli_parser_validate_kibana_flag() -> None:
    """Parser accepts --validate-kibana and sets validate_kibana=True."""
    parsed = parse_cli_arguments(["--validate-kibana"])
    assert parsed.validate_kibana is True
    assert parsed.run_all is False
    assert parsed.index_elasticsearch is False
    assert parsed.ingest_reddit is False


def test_cli_mutual_exclusion_run_all_and_validate_kibana() -> None:
    """--validate-kibana combined with --run-all must be rejected."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--run-all", "--validate-kibana"])


def test_cli_mutual_exclusion_index_elasticsearch_and_validate_kibana() -> None:
    """--validate-kibana combined with --index-elasticsearch must be rejected."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--index-elasticsearch", "--validate-kibana"])


def test_cli_mutual_exclusion_legacy_actions_and_validate_kibana() -> None:
    """--validate-kibana combined with any other execution flag must be rejected."""
    conflicting_flags = [
        "--ingest-reddit",
        "--ingest-reddit-live",
        "--ingest-youtube",
        "--preprocess-reddit",
        "--preprocess-youtube",
    ]
    for flag in conflicting_flags:
        with pytest.raises(SystemExit):
            parse_cli_arguments([flag, "--validate-kibana"])


def test_cli_skip_flags_rejected_with_validate_kibana() -> None:
    """Skip flags are invalid with --validate-kibana and must be rejected."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--validate-kibana", "--skip-kibana"])

    with pytest.raises(SystemExit):
        parse_cli_arguments(["--validate-kibana", "--skip-reddit"])


def test_cli_manifest_path_rejected_with_validate_kibana() -> None:
    """--manifest-path can only be supplied with --run-all."""
    with pytest.raises(SystemExit):
        parse_cli_arguments(["--validate-kibana", "--manifest-path=custom.json"])


# =====================================================================
# 5. CLI Runner & Dispatch Tests
# =====================================================================

def test_run_validate_kibana_cli_success(canonical_objects, tmp_path: Path) -> None:
    """run_validate_kibana_cli returns 0 on valid bundle."""
    bundle_file = _write_ndjson(tmp_path, canonical_objects)
    exit_code = run_validate_kibana_cli(bundle_path=bundle_file)
    assert exit_code == APP_EXIT_SUCCESS


def test_run_validate_kibana_cli_failure(tmp_path: Path) -> None:
    """run_validate_kibana_cli returns 1 on invalid bundle."""
    bad_file = tmp_path / "bad.ndjson"
    bad_file.write_text("{ unclosed", encoding="utf-8")
    exit_code = run_validate_kibana_cli(bundle_path=bad_file)
    assert exit_code == EXIT_STAGE_FAILURE


def test_cli_main_dispatches_validate_kibana_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() invokes run_validate_kibana_cli and exits cleanly on success."""
    mock_cli = MagicMock(return_value=APP_EXIT_SUCCESS)
    monkeypatch.setattr("app.run_validate_kibana_cli", mock_cli)

    main(["--validate-kibana"])
    mock_cli.assert_called_once()


def test_cli_main_dispatches_validate_kibana_failure_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() calls sys.exit when run_validate_kibana_cli returns non-zero."""
    mock_cli = MagicMock(return_value=EXIT_STAGE_FAILURE)
    monkeypatch.setattr("app.run_validate_kibana_cli", mock_cli)

    with pytest.raises(SystemExit) as exc_info:
        main(["--validate-kibana"])
    assert exc_info.value.code == EXIT_STAGE_FAILURE


# =====================================================================
# 6. Offline Guarantees & Diagnostics Tests
# =====================================================================

def test_cli_runner_invokes_validator_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies run_validate_kibana_cli invokes the offline validator exactly once."""
    mock_val = MagicMock(return_value={
        "is_valid": True,
        "total_objects": 41,
        "counts_by_type": {"dashboard": 5},
        "dashboards": {"dashboard-executive-overview": 8},
        "errors": [],
        "warnings": [],
    })
    monkeypatch.setattr("src.search.kibana_dashboard_validator.validate_kibana_dashboard_bundle", mock_val)

    exit_code = run_validate_kibana_cli()
    assert exit_code == APP_EXIT_SUCCESS
    mock_val.assert_called_once_with(DEFAULT_BUNDLE_PATH)


def test_cli_runner_displays_diagnostics(capsys: pytest.CaptureFixture) -> None:
    """Verifies run_validate_kibana_cli prints useful diagnostic information."""
    exit_code = run_validate_kibana_cli()
    assert exit_code == APP_EXIT_SUCCESS
    captured = capsys.readouterr()
    assert "KIBANA DASHBOARD VALIDATION" in captured.out
    assert "Status               : PASSED" in captured.out
    assert "Total Saved Objects  : 41" in captured.out
    assert "dashboard-executive-overview" in captured.out


def test_cli_runner_offline_guarantees(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies no Elasticsearch client, Kibana importer, or HTTP calls occur during validation."""
    mock_urlopen = MagicMock()
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    mock_importer = MagicMock()
    monkeypatch.setattr("src.search.kibana_dashboard_importer.import_kibana_dashboard_bundle", mock_importer)

    # Read initial bytes of the canonical bundle
    initial_bytes = DEFAULT_BUNDLE_PATH.read_bytes()

    exit_code = run_validate_kibana_cli()
    assert exit_code == APP_EXIT_SUCCESS

    # Verify zero network/importer activity
    mock_urlopen.assert_not_called()
    mock_importer.assert_not_called()

    # Verify canonical bundle bytes are unchanged
    after_bytes = DEFAULT_BUNDLE_PATH.read_bytes()
    assert initial_bytes == after_bytes


def test_cli_run_all_safeguard_remains_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """--run-all dispatches to the unified orchestrator and propagates exit codes without running live services."""
    mock_manifest = {"run_id": "run_mock_20261111", "overall_outcome": "success"}
    mock_runner = MagicMock(return_value=(APP_EXIT_SUCCESS, mock_manifest))
    monkeypatch.setattr("app.run_orchestrated_pipeline", mock_runner)

    # 1. Verify --run-all dispatches to unified orchestrator and returns cleanly on success
    main(["--run-all"])
    mock_runner.assert_called_once_with(["--run-all"])

    # 2. Verify exit code propagation on failure
    mock_runner.reset_mock()
    mock_runner.return_value = (EXIT_STAGE_FAILURE, {"run_id": "run_mock_fail", "overall_outcome": "failed"})
    with pytest.raises(SystemExit) as exc_info_fail:
        main(["--run-all"])
    assert exc_info_fail.value.code == EXIT_STAGE_FAILURE
    mock_runner.assert_called_once_with(["--run-all"])

    # 3. Verify --validate-kibana remains independently usable
    mock_val_cli = MagicMock(return_value=APP_EXIT_SUCCESS)
    monkeypatch.setattr("app.run_validate_kibana_cli", mock_val_cli)
    main(["--validate-kibana"])
    mock_val_cli.assert_called_once()



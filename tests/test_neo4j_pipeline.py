"""Unit and Integration Tests for Stage 8 Neo4j Persistence Pipeline (Task 8B).

Verifies the independent Stage 8 runner (run_neo4j_persistence):
- Artifact loading, strict Schema-v2 validation, and status discrimination.
- Rejection of missing, corrupted, malformed, or diagnostic-error artifacts.
- Explicit non-negative integer community_id validation for every node (failing closed on booleans, strings, floats, missing).
- Empty graph successful no-op handling (zero writes, skipped_empty outcome).
- Strict execution ordering: ensure_constraints() called before Cypher batch transaction.
- Constraint failures fail closed and prevent persistence transactions.
- Connector result validation (requiring status == 'success', failing closed on error/unknown/missing status).
- Single-transaction atomic persistence with zero committed entity reporting on failure.
- Provenance and partial source completeness preservation.
- Resource lifecycle and ownership semantics (injected vs owned manager).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from src.graph.neo4j_pipeline import (
    DEFAULT_GRAPH_ARTIFACT_PATH,
    SUPPORTED_SCHEMA_VERSION,
    run_neo4j_persistence,
)


# ==============================================================================
# Mock Neo4j Test Doubles
# ==============================================================================

class MockNeo4jManager:
    """Mock Neo4jGraphManager for offline verification of Stage 8 persistence."""

    def __init__(
        self,
        ensure_constraints_fails: bool = False,
        constraint_exception: Exception | None = None,
        execute_cypher_batches_return: dict[str, Any] | Any = None,
        execute_cypher_batches_fails: bool = False,
        batch_exception: Exception | None = None,
    ) -> None:
        self.ensure_constraints_fails = ensure_constraints_fails
        self.constraint_exception = constraint_exception or RuntimeError("Constraint creation failed")
        self.execute_cypher_batches_return = execute_cypher_batches_return
        self.execute_cypher_batches_fails = execute_cypher_batches_fails
        self.batch_exception = batch_exception or RuntimeError("Cypher batch execution failed")
        self.call_log: list[str] = []
        self.generate_cypher_export_args: dict[str, Any] = {}
        self.closed: bool = False

    def ensure_constraints(self, session: Any = None) -> None:
        self.call_log.append("ensure_constraints")
        if self.ensure_constraints_fails:
            raise self.constraint_exception

    def generate_cypher_export(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        batch_size: int = 500,
        top_influencers: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.call_log.append("generate_cypher_export")
        self.generate_cypher_export_args = {
            "nodes": copy.deepcopy(nodes),
            "edges": copy.deepcopy(edges),
            "batch_size": batch_size,
            "top_influencers": copy.deepcopy(top_influencers),
        }
        return {
            "user_batches": [{"query": "UNWIND $batch AS row MERGE (u:User {node_id: row.node_id})", "batch": nodes}],
            "relationship_batches": [{"query": "UNWIND $batch AS row MATCH ...", "batch": edges}],
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

    def execute_cypher_batches(self, export_payload: dict[str, Any]) -> dict[str, Any]:
        self.call_log.append("execute_cypher_batches")
        if self.execute_cypher_batches_fails:
            raise self.batch_exception
        if self.execute_cypher_batches_return is not None:
            return self.execute_cypher_batches_return
        return {
            "status": "success",
            "total_nodes_ingested": export_payload.get("total_nodes", 0),
            "total_edges_ingested": export_payload.get("total_edges", 0),
            "executed_user_batches": len(export_payload.get("user_batches", [])),
            "executed_relationship_batches": len(export_payload.get("relationship_batches", [])),
        }

    def close(self) -> None:
        self.call_log.append("close")
        self.closed = True


# ==============================================================================
# Realistic Schema-v2 Fixture Builders
# ==============================================================================

def build_valid_stage7_success_payload() -> dict[str, Any]:
    """Constructs a valid Schema-v2 artifact with complete community IDs."""
    nodes = [
        {
            "node_id": "reddit:user:alice_data",
            "platform": "reddit",
            "identity_type": "account",
            "username": "Alice_Data",
            "channel_id": None,
            "channel_title": None,
            "is_author": True,
            "activity_count": 2,
            "community_id": 0,
        },
        {
            "node_id": "reddit:user:bob_ml",
            "platform": "reddit",
            "identity_type": "account",
            "username": "Bob_ML",
            "channel_id": None,
            "channel_title": None,
            "is_author": True,
            "activity_count": 1,
            "community_id": 0,
        },
        {
            "node_id": "youtube:channel:UC_TECHGURU",
            "platform": "youtube",
            "identity_type": "creator",
            "username": "TechGuru",
            "channel_id": "UC_TECHGURU",
            "channel_title": "TechGuru",
            "is_author": True,
            "activity_count": 3,
            "community_id": 1,
        },
        {
            "node_id": "youtube:mention:alice_data",
            "platform": "youtube",
            "identity_type": "mention",
            "username": "alice_data",
            "channel_id": None,
            "channel_title": None,
            "is_author": False,
            "activity_count": 0,
            "community_id": 0,
        },
    ]

    edges = [
        {
            "source": "reddit:user:alice_data",
            "target": "reddit:user:bob_ml",
            "weight": 2,
            "interaction_types": ["MENTIONED"],
            "platforms": ["reddit"],
        },
        {
            "source": "youtube:channel:UC_TECHGURU",
            "target": "youtube:mention:alice_data",
            "weight": 1,
            "interaction_types": ["MENTIONED"],
            "platforms": ["youtube"],
        },
    ]

    top_influencers = [
        {
            "node_id": "reddit:user:alice_data",
            "platform": "reddit",
            "identity_type": "account",
            "username": "Alice_Data",
            "channel_id": None,
            "channel_title": None,
            "is_author": True,
            "activity_count": 2,
            "influencer_score": 0.85,
            "pagerank": 0.35,
            "pagerank_norm": 1.0,
            "in_degree_centrality": 0.5,
            "in_degree_norm": 1.0,
            "betweenness_centrality": 0.2,
            "betweenness_norm": 0.5,
            "community_id": 0,
            "in_degree_raw": 1,
            "out_degree_raw": 1,
        },
        {
            "node_id": "youtube:channel:UC_TECHGURU",
            "platform": "youtube",
            "identity_type": "creator",
            "username": "TechGuru",
            "channel_id": "UC_TECHGURU",
            "channel_title": "TechGuru",
            "is_author": True,
            "activity_count": 3,
            "influencer_score": 0.65,
            "pagerank": 0.25,
            "pagerank_norm": 0.7,
            "in_degree_centrality": 0.0,
            "in_degree_norm": 0.0,
            "betweenness_centrality": 0.0,
            "betweenness_norm": 0.0,
            "community_id": 1,
            "in_degree_raw": 0,
            "out_degree_raw": 1,
        },
    ]

    return {
        "schema_version": "2.0",
        "status": "success",
        "generated_at": "2026-11-10T18:00:00Z",
        "provenance": {
            "reddit_input_file": "data/processed/reddit_posts_processed.json",
            "reddit_status": "loaded",
            "reddit_records_count": 2,
            "youtube_input_file": "data/processed/youtube_videos_processed.json",
            "youtube_status": "loaded",
            "youtube_records_count": 1,
        },
        "graph_metrics": {
            "total_users": 4,
            "total_interactions": 2,
            "density": 0.166667,
            "communities_count": 2,
            "is_directed": True,
        },
        "nodes": nodes,
        "edges": edges,
        "top_influencers": top_influencers,
        "community_distributions": {
            "0": 3,
            "1": 1,
        },
    }


def build_valid_stage7_partial_payload() -> dict[str, Any]:
    """Constructs a valid partial Schema-v2 artifact (Reddit loaded, YouTube missing)."""
    payload = build_valid_stage7_success_payload()
    payload["status"] = "partial"
    payload["provenance"]["youtube_status"] = "missing"
    payload["provenance"]["youtube_records_count"] = 0
    payload["nodes"] = [n for n in payload["nodes"] if n["platform"] == "reddit"]
    payload["edges"] = [e for e in payload["edges"] if e["platforms"] == ["reddit"]]
    payload["top_influencers"] = [i for i in payload["top_influencers"] if i["platform"] == "reddit"]
    payload["graph_metrics"]["total_users"] = 2
    payload["graph_metrics"]["total_interactions"] = 1
    payload["graph_metrics"]["communities_count"] = 1
    payload["graph_metrics"]["density"] = 0.5
    payload["community_distributions"] = {"0": 2}
    return payload


def build_valid_stage7_empty_payload() -> dict[str, Any]:
    """Constructs a valid empty Schema-v2 artifact."""
    return {
        "schema_version": "2.0",
        "status": "empty",
        "generated_at": "2026-11-10T18:00:00Z",
        "provenance": {
            "reddit_input_file": "data/processed/reddit_posts_processed.json",
            "reddit_status": "empty",
            "reddit_records_count": 0,
            "youtube_input_file": "data/processed/youtube_videos_processed.json",
            "youtube_status": "empty",
            "youtube_records_count": 0,
        },
        "graph_metrics": {
            "total_users": 0,
            "total_interactions": 0,
            "density": 0.0,
            "communities_count": 0,
            "is_directed": True,
        },
        "nodes": [],
        "edges": [],
        "top_influencers": [],
        "community_distributions": {},
    }


def write_artifact(path: Path, payload: Any) -> Path:
    """Helper to serialize payload to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f, indent=2)
    return path


# ==============================================================================
# Group 1: Artifact Validation Tests
# ==============================================================================

def test_valid_schema_v2_success_artifact_loads(tmp_path: Path) -> None:
    """1. Valid schema-v2 success artifact loads and persists completely."""
    artifact_path = write_artifact(tmp_path / "graph_analysis_results.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["input_status"] == "success"
    assert report["source_status"] == "success"
    assert report["source_completeness"] == "complete"
    assert report["persistence_outcome"] == "success"
    assert report["total_graph_nodes"] == 4
    assert report["total_graph_relationships"] == 2
    assert report["persisted_nodes"] == 4
    assert report["persisted_relationships"] == 2
    assert report["committed_nodes"] == 4
    assert report["committed_relationships"] == 2
    assert report["executed_user_batches"] == 1
    assert report["executed_relationship_batches"] == 1
    assert report["error_category"] is None
    assert report["error_message"] is None
    assert report["duration_seconds"] >= 0.0


def test_valid_partial_artifact_loads(tmp_path: Path) -> None:
    """2. Valid partial artifact loads and persists while preserving partial provenance."""
    artifact_path = write_artifact(tmp_path / "graph_analysis_results.json", build_valid_stage7_partial_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["input_status"] == "partial"
    assert report["source_status"] == "partial"
    assert report["source_completeness"] == "partial"
    assert report["persistence_outcome"] == "success"
    assert report["total_graph_nodes"] == 2
    assert report["total_graph_relationships"] == 1
    assert report["persisted_nodes"] == 2
    assert report["persisted_relationships"] == 1
    assert report["provenance"]["youtube_status"] == "missing"


def test_valid_empty_artifact_is_handled_as_successful_noop(tmp_path: Path) -> None:
    """3. Valid empty artifact is handled as a successful no-op without database writes."""
    artifact_path = write_artifact(tmp_path / "graph_analysis_results.json", build_valid_stage7_empty_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["input_status"] == "empty"
    assert report["source_status"] == "empty"
    assert report["source_completeness"] == "empty"
    assert report["persistence_outcome"] == "skipped_empty"
    assert report["total_graph_nodes"] == 0
    assert report["total_graph_relationships"] == 0
    assert report["persisted_nodes"] == 0
    assert report["persisted_relationships"] == 0
    assert report["executed_user_batches"] == 0
    assert report["executed_relationship_batches"] == 0

    # Ensure no database interactions occurred
    assert "ensure_constraints" not in mock_mgr.call_log
    assert "execute_cypher_batches" not in mock_mgr.call_log


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    """4. Missing artifact file path fails closed."""
    non_existent_path = tmp_path / "non_existent_artifact.json"
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=non_existent_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "artifact_not_found"
    assert "not found" in report["error_message"].lower()
    assert len(mock_mgr.call_log) == 0


def test_corrupted_json_is_rejected(tmp_path: Path) -> None:
    """5. Invalid / unparseable JSON file fails closed."""
    corrupted_path = write_artifact(tmp_path / "corrupted.json", '{"schema_version": "2.0", truncated...')
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=corrupted_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "corrupted_artifact"
    assert len(mock_mgr.call_log) == 0


def test_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    """6. Unsupported schema version (e.g., '1.0' or '2.1') fails closed."""
    payload = build_valid_stage7_success_payload()
    payload["schema_version"] = "1.0"
    artifact_path = write_artifact(tmp_path / "schema_v1.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "unsupported_schema_version"
    assert "1.0" in report["error_message"]
    assert len(mock_mgr.call_log) == 0


def test_invalid_graph_artifact_is_rejected(tmp_path: Path) -> None:
    """7. Artifact failing GraphArtifactValidator rules fails closed."""
    payload = build_valid_stage7_success_payload()
    # Break edge referential integrity: unknown target node
    payload["edges"].append({
        "source": "reddit:user:alice_data",
        "target": "ghost:node:unknown",
        "weight": 1,
        "interaction_types": ["MENTIONED"],
        "platforms": ["reddit"],
    })
    artifact_path = write_artifact(tmp_path / "invalid_graph.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "validation_failed"
    assert len(mock_mgr.call_log) == 0


def test_diagnostic_error_artifact_is_rejected(tmp_path: Path) -> None:
    """8. Diagnostic error artifact (status='error') is rejected upfront."""
    error_payload = {
        "schema_version": "2.0",
        "status": "error",
        "error": "Upstream source failure",
        "provenance": {"reddit_status": "corrupted"},
        "nodes": [],
        "edges": [],
        "top_influencers": [],
        "community_distributions": {},
        "graph_metrics": {"total_users": 0, "total_interactions": 0, "density": 0.0, "communities_count": 0, "is_directed": True},
    }
    artifact_path = write_artifact(tmp_path / "error_artifact.json", error_payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["source_status"] == "error"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "diagnostic_artifact_rejected"
    assert len(mock_mgr.call_log) == 0


def test_non_dict_root_artifact_is_rejected(tmp_path: Path) -> None:
    """Extra: Non-dictionary root JSON structure fails closed."""
    artifact_path = write_artifact(tmp_path / "list_artifact.json", [1, 2, 3])
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "malformed_artifact"
    assert len(mock_mgr.call_log) == 0


def test_invalid_artifact_status_is_rejected(tmp_path: Path) -> None:
    """Extra: Unknown artifact status string fails closed."""
    payload = build_valid_stage7_success_payload()
    payload["status"] = "quasi_valid"
    artifact_path = write_artifact(tmp_path / "invalid_status.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "invalid_artifact_status"
    assert len(mock_mgr.call_log) == 0


# ==============================================================================
# Group 2: Community Validation Tests
# ==============================================================================

def test_missing_community_id_is_rejected(tmp_path: Path) -> None:
    """9. Node missing 'community_id' key fails closed."""
    payload = build_valid_stage7_success_payload()
    del payload["nodes"][0]["community_id"]
    artifact_path = write_artifact(tmp_path / "missing_cid.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "missing_community_id"
    assert "missing 'community_id'" in report["error_message"]
    assert len(mock_mgr.call_log) == 0


def test_boolean_community_id_is_rejected(tmp_path: Path) -> None:
    """10. Boolean community_id (True/False) fails closed."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][0]["community_id"] = True
    artifact_path = write_artifact(tmp_path / "bool_cid.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "invalid_community_id"
    assert "must be a non-negative integer" in report["error_message"]
    assert len(mock_mgr.call_log) == 0


def test_negative_community_id_is_rejected(tmp_path: Path) -> None:
    """11. Negative community_id fails closed."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][1]["community_id"] = -1
    artifact_path = write_artifact(tmp_path / "neg_cid.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "invalid_community_id"
    assert len(mock_mgr.call_log) == 0


def test_float_community_id_is_rejected(tmp_path: Path) -> None:
    """12. Float community_id fails closed."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][2]["community_id"] = 2.0
    artifact_path = write_artifact(tmp_path / "float_cid.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "invalid_community_id"
    assert len(mock_mgr.call_log) == 0


def test_string_community_id_is_rejected(tmp_path: Path) -> None:
    """13. String community_id fails closed."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][0]["community_id"] = "0"
    artifact_path = write_artifact(tmp_path / "str_cid.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] == "invalid_community_id"
    assert len(mock_mgr.call_log) == 0


def test_valid_non_negative_integer_community_ids_accepted(tmp_path: Path) -> None:
    """14. Valid non-negative integer community IDs (0, 1, 100) are accepted."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][0]["community_id"] = 0
    payload["nodes"][1]["community_id"] = 1
    payload["nodes"][2]["community_id"] = 100
    payload["nodes"][3]["community_id"] = 0
    artifact_path = write_artifact(tmp_path / "valid_cids.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["persisted_nodes"] == 4


def test_node_missing_node_id_is_rejected(tmp_path: Path) -> None:
    """Extra: Node missing node_id fails closed (caught by validator or runner)."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][0]["node_id"] = ""
    artifact_path = write_artifact(tmp_path / "empty_nid.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["error_category"] in {"validation_failed", "missing_node_id"}
    assert len(mock_mgr.call_log) == 0


# ==============================================================================
# Group 3: Persistence Execution & Connector Integration Tests
# ==============================================================================

def test_complete_nodes_and_edges_passed_to_export_generator(tmp_path: Path) -> None:
    """15. Complete nodes and edges lists are passed to generate_cypher_export."""
    payload = build_valid_stage7_success_payload()
    artifact_path = write_artifact(tmp_path / "graph.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr, batch_size=250)

    assert report["status"] == "success"
    passed_args = mock_mgr.generate_cypher_export_args
    assert len(passed_args["nodes"]) == len(payload["nodes"])
    assert len(passed_args["edges"]) == len(payload["edges"])
    assert passed_args["batch_size"] == 250
    assert passed_args["top_influencers"] == payload["top_influencers"]


def test_constraint_called_before_batch_execution(tmp_path: Path) -> None:
    """16. Constraint is called before Cypher batch transaction execution."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert "ensure_constraints" in mock_mgr.call_log
    assert "execute_cypher_batches" in mock_mgr.call_log
    idx_constraint = mock_mgr.call_log.index("ensure_constraints")
    idx_batch = mock_mgr.call_log.index("execute_cypher_batches")
    assert idx_constraint < idx_batch


def test_constraint_failure_prevents_batch_execution(tmp_path: Path) -> None:
    """17. Constraint creation failure prevents batch execution and fails closed."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(ensure_constraints_fails=True)

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "constraint_creation_failed"
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0
    assert "execute_cypher_batches" not in mock_mgr.call_log


def test_successful_connector_result_produces_successful_stage8_outcome(tmp_path: Path) -> None:
    """18. Successful connector result produces successful Stage 8 outcome."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(
        execute_cypher_batches_return={
            "status": "success",
            "total_nodes_ingested": 4,
            "total_edges_ingested": 2,
            "executed_user_batches": 1,
            "executed_relationship_batches": 1,
        }
    )

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["persistence_outcome"] == "success"
    assert report["persisted_nodes"] == 4
    assert report["persisted_relationships"] == 2


def test_structured_connector_error_produces_failed_stage8_outcome(tmp_path: Path) -> None:
    """19. Structured connector error produces failed Stage 8 outcome with zero committed counts."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(
        execute_cypher_batches_return={
            "status": "error",
            "error": "Single-transaction rollback on relationship batch 1",
            "error_category": "RuntimeError",
            "total_nodes_ingested": 0,
            "total_edges_ingested": 0,
        }
    )

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "RuntimeError"
    assert "Single-transaction rollback" in report["error_message"]
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0
    assert report["persisted_nodes"] == 0
    assert report["persisted_relationships"] == 0


def test_missing_connector_status_produces_failure(tmp_path: Path) -> None:
    """20. Missing connector status key in response produces failure."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(execute_cypher_batches_return={"total_nodes_ingested": 4})

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0


def test_unknown_connector_status_produces_failure(tmp_path: Path) -> None:
    """21. Unknown connector status (e.g., 'in_progress' or 'partial') produces failure."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(execute_cypher_batches_return={"status": "in_progress"})

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0


def test_connector_exception_produces_failure(tmp_path: Path) -> None:
    """22. Connector exception during transaction execution produces failure."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(
        execute_cypher_batches_fails=True,
        batch_exception=ConnectionResetError("Neo4j socket disconnected"),
    )

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["persistence_outcome"] == "failed"
    assert report["error_category"] == "ConnectionResetError"
    assert "Neo4j socket disconnected" in report["error_message"]
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0


def test_failed_persistence_reports_zero_committed_counts(tmp_path: Path) -> None:
    """23. Failed persistence transaction reports zero committed counts even if payload was large."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(
        execute_cypher_batches_return={
            "status": "error",
            "error": "Deadlock detected",
            "error_category": "TransientError",
        }
    )

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "failed"
    assert report["total_graph_nodes"] == 4
    assert report["total_graph_relationships"] == 2
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0
    assert report["persisted_nodes"] == 0
    assert report["persisted_relationships"] == 0


def test_partial_source_status_remains_partial_after_successful_persistence(tmp_path: Path) -> None:
    """24. Partial source completeness is preserved independently of persistence success."""
    artifact_path = write_artifact(tmp_path / "partial.json", build_valid_stage7_partial_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["input_status"] == "partial"
    assert report["source_status"] == "partial"
    assert report["source_completeness"] == "partial"
    assert report["persistence_outcome"] == "success"
    assert report["persisted_nodes"] == 2


def test_empty_graph_performs_no_writes_and_reports_zero_counts(tmp_path: Path) -> None:
    """25. Empty graph performs zero writes and reports zero committed counts."""
    artifact_path = write_artifact(tmp_path / "empty.json", build_valid_stage7_empty_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    assert report["persistence_outcome"] == "skipped_empty"
    assert report["committed_nodes"] == 0
    assert report["committed_relationships"] == 0
    assert len(mock_mgr.call_log) == 0


def test_no_community_detection_or_recomputation_occurs_in_stage8(tmp_path: Path) -> None:
    """26. Verify community assignments from Stage 7 are preserved untouched without recomputation."""
    payload = build_valid_stage7_success_payload()
    payload["nodes"][0]["community_id"] = 42
    payload["nodes"][1]["community_id"] = 99
    payload["nodes"][2]["community_id"] = 7
    payload["nodes"][3]["community_id"] = 7
    artifact_path = write_artifact(tmp_path / "custom_cids.json", payload)
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert report["status"] == "success"
    passed_nodes = mock_mgr.generate_cypher_export_args["nodes"]
    cid_map = {n["node_id"]: n["community_id"] for n in passed_nodes}
    assert cid_map["reddit:user:alice_data"] == 42
    assert cid_map["reddit:user:bob_ml"] == 99
    assert cid_map["youtube:channel:UC_TECHGURU"] == 7
    assert cid_map["youtube:mention:alice_data"] == 7


# ==============================================================================
# Group 4: Resource Ownership & Lifecycle Tests
# ==============================================================================

def test_injected_manager_lifecycle_follows_ownership_contract(tmp_path: Path) -> None:
    """27. Injected manager instance is NOT closed by run_neo4j_persistence."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager()

    run_neo4j_persistence(artifact_path=artifact_path, manager=mock_mgr)

    assert mock_mgr.closed is False
    assert "close" not in mock_mgr.call_log


def test_internally_owned_manager_is_closed_in_finally(tmp_path: Path) -> None:
    """28. Internally owned manager (default or factory) is closed in finally."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager()

    with patch("src.graph.neo4j_pipeline.Neo4jGraphManager", return_value=mock_mgr):
        run_neo4j_persistence(artifact_path=artifact_path)

    assert mock_mgr.closed is True
    assert "close" in mock_mgr.call_log


def test_failure_paths_do_not_leak_resources(tmp_path: Path) -> None:
    """29. Internally owned manager is properly closed even when transaction fails."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager(execute_cypher_batches_fails=True)

    with patch("src.graph.neo4j_pipeline.Neo4jGraphManager", return_value=mock_mgr):
        report = run_neo4j_persistence(artifact_path=artifact_path)

    assert report["status"] == "failed"
    assert mock_mgr.closed is True
    assert "close" in mock_mgr.call_log


def test_manager_factory_injection(tmp_path: Path) -> None:
    """30. manager_factory callable is invoked and provides manager without closing it."""
    artifact_path = write_artifact(tmp_path / "graph.json", build_valid_stage7_success_payload())
    mock_mgr = MockNeo4jManager()

    report = run_neo4j_persistence(artifact_path=artifact_path, manager_factory=lambda: mock_mgr)

    assert report["status"] == "success"
    assert mock_mgr.closed is False

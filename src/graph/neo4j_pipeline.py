"""Stage 8 — Neo4j Persistence Pipeline (neo4j_persistence).

Consumes validated Stage 7 Schema-v2 graph analysis artifacts, verifies exact
node community coverage, creates uniqueness constraints, and executes atomic
multi-batch persistence into Neo4j using single-transaction guarantees.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Optional

from src.graph.graph_validator import GraphArtifactValidator
from src.graph.neo4j_connector import Neo4jGraphManager

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_ARTIFACT_PATH = Path("data/processed/graph_analysis_results.json")
SUPPORTED_SCHEMA_VERSION = "2.0"
PERMITTED_SOURCE_STATUSES = frozenset({"success", "partial", "empty"})


def _build_execution_report(
    status: str,
    source_status: Optional[str] = None,
    persistence_outcome: str = "failed",
    total_graph_nodes: int = 0,
    total_graph_relationships: int = 0,
    committed_nodes: int = 0,
    committed_relationships: int = 0,
    executed_user_batches: int = 0,
    executed_relationship_batches: int = 0,
    duration_seconds: float = 0.0,
    artifact_path: Optional[str] = None,
    error_category: Optional[str] = None,
    error_message: Optional[str] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Constructs a normalized, structured Stage 8 execution report dictionary."""
    if source_status == "success":
        source_completeness = "complete"
    elif source_status == "partial":
        source_completeness = "partial"
    elif source_status == "empty":
        source_completeness = "empty"
    else:
        source_completeness = "unknown"

    stored_duration = round(float(duration_seconds), 4) if math.isfinite(duration_seconds) else 0.0

    return {
        "stage": "neo4j_persistence",
        "status": status,
        "input_status": source_status,
        "source_status": source_status,
        "source_completeness": source_completeness,
        "persistence_outcome": persistence_outcome,
        "total_graph_nodes": total_graph_nodes,
        "total_graph_relationships": total_graph_relationships,
        "persisted_nodes": committed_nodes,
        "persisted_relationships": committed_relationships,
        "committed_nodes": committed_nodes,
        "committed_relationships": committed_relationships,
        "executed_user_batches": executed_user_batches,
        "executed_relationship_batches": executed_relationship_batches,
        "duration_seconds": stored_duration,
        "artifact_path": artifact_path,
        "error_category": error_category,
        "error_message": error_message,
        "provenance": provenance or {},
    }


def run_neo4j_persistence(
    artifact_path: Optional[Path | str] = None,
    manager: Optional[Neo4jGraphManager] = None,
    manager_factory: Optional[Callable[[], Neo4jGraphManager]] = None,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Executes Stage 8 Neo4j Persistence from a validated Stage 7 graph artifact.

    Args:
        artifact_path: Optional explicit path to Stage 7 graph analysis results JSON.
            Defaults to DEFAULT_GRAPH_ARTIFACT_PATH ("data/processed/graph_analysis_results.json").
        manager: Optional injected Neo4jGraphManager instance (for testing or caller-managed connections).
        manager_factory: Optional factory callable returning a Neo4jGraphManager instance.
        batch_size: Maximum entities per parameterized Cypher UNWIND batch.

    Returns:
        Structured Stage 8 execution report dictionary conforming to platform standards.
    """
    start_time = time.perf_counter()
    target_path = Path(artifact_path) if artifact_path is not None else DEFAULT_GRAPH_ARTIFACT_PATH
    path_str = str(target_path)

    # 1. Verify artifact exists
    if not target_path.exists():
        logger.error("Stage 7 graph artifact not found at %s", path_str)
        return _build_execution_report(
            status="failed",
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="artifact_not_found",
            error_message=f"Stage 7 graph artifact not found at {path_str}",
            duration_seconds=time.perf_counter() - start_time,
        )

    # 2. Safely read and parse JSON
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.error("Corrupted JSON in Stage 7 graph artifact %s: %s", path_str, exc)
        return _build_execution_report(
            status="failed",
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="corrupted_artifact",
            error_message=f"Corrupted JSON in graph artifact: {exc}",
            duration_seconds=time.perf_counter() - start_time,
        )
    except Exception as exc:
        logger.error("Failed to read Stage 7 graph artifact %s: %s", path_str, exc)
        return _build_execution_report(
            status="failed",
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="unreadable_artifact",
            error_message=f"Failed to read graph artifact: {exc}",
            duration_seconds=time.perf_counter() - start_time,
        )

    if not isinstance(payload, dict):
        logger.error("Stage 7 graph artifact %s root must be a dict, got %s", path_str, type(payload).__name__)
        return _build_execution_report(
            status="failed",
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="malformed_artifact",
            error_message=f"Top-level graph artifact must be a dict, got {type(payload).__name__}",
            duration_seconds=time.perf_counter() - start_time,
        )

    source_status = payload.get("status")
    provenance = payload.get("provenance", {})

    # 3. Reject diagnostic 'error' artifacts
    if source_status == "error":
        logger.error("Diagnostic error artifact rejected for persistence from %s", path_str)
        return _build_execution_report(
            status="failed",
            source_status="error",
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="diagnostic_artifact_rejected",
            error_message="Stage 7 artifact has status='error'; cannot persist diagnostic failure artifact",
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    # 4. Enforce strict Schema Version 2.0
    schema_ver = payload.get("schema_version")
    if schema_ver != SUPPORTED_SCHEMA_VERSION:
        logger.error("Unsupported graph schema version %r in %s, expected %r", schema_ver, path_str, SUPPORTED_SCHEMA_VERSION)
        return _build_execution_report(
            status="failed",
            source_status=source_status if isinstance(source_status, str) else None,
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="unsupported_schema_version",
            error_message=f"Unsupported schema_version {schema_ver!r}; expected {SUPPORTED_SCHEMA_VERSION!r}",
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    # 5. Verify established status
    if source_status not in PERMITTED_SOURCE_STATUSES:
        logger.error("Invalid graph artifact status %r in %s", source_status, path_str)
        return _build_execution_report(
            status="failed",
            source_status=str(source_status),
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="invalid_artifact_status",
            error_message=f"Invalid graph artifact status {source_status!r}; expected one of {sorted(PERMITTED_SOURCE_STATUSES)}",
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    # 6. Validate artifact with GraphArtifactValidator
    is_valid, validation_errors = GraphArtifactValidator.validate_graph_artifact(payload)
    if not is_valid:
        logger.error("GraphArtifactValidator failed (%d errors) on %s: %s", len(validation_errors), path_str, validation_errors)
        return _build_execution_report(
            status="failed",
            source_status=source_status,
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="validation_failed",
            error_message=f"Graph artifact validation failed: {validation_errors}",
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    # 7. Extract nodes and edges
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return _build_execution_report(
            status="failed",
            source_status=source_status,
            persistence_outcome="failed",
            artifact_path=path_str,
            error_category="malformed_graph_entities",
            error_message="'nodes' and 'edges' must be JSON lists",
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    total_nodes = len(nodes)
    total_edges = len(edges)

    # 8. Explicit Community ID Validation for every node
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                artifact_path=path_str,
                error_category="malformed_node_entry",
                error_message=f"nodes[{idx}] must be a dict, got {type(node).__name__}",
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

        nid = node.get("node_id")
        if not isinstance(nid, str) or not nid.strip():
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                artifact_path=path_str,
                error_category="missing_node_id",
                error_message=f"nodes[{idx}] is missing a valid 'node_id'",
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

        if "community_id" not in node:
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                artifact_path=path_str,
                error_category="missing_community_id",
                error_message=f"Node {nid!r} is missing 'community_id'",
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

        cid = node.get("community_id")
        if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0:
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                artifact_path=path_str,
                error_category="invalid_community_id",
                error_message=f"Invalid community_id {cid!r} for node {nid!r}: must be a non-negative integer",
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

    # 9. Handle Empty Graph as a successful no-op
    if source_status == "empty" or total_nodes == 0:
        logger.info("Stage 7 artifact represents an empty valid graph. Skipping Neo4j persistence as successful no-op.")
        return _build_execution_report(
            status="success",
            source_status="empty",
            persistence_outcome="skipped_empty",
            total_graph_nodes=0,
            total_graph_relationships=0,
            committed_nodes=0,
            committed_relationships=0,
            executed_user_batches=0,
            executed_relationship_batches=0,
            artifact_path=path_str,
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    # 10. Instantiate or obtain Neo4jGraphManager
    owned_manager = False
    mgr: Optional[Neo4jGraphManager] = None
    if manager is not None:
        mgr = manager
    elif manager_factory is not None:
        mgr = manager_factory()
    else:
        mgr = Neo4jGraphManager()
        owned_manager = True

    try:
        # 11. Ensure uniqueness constraint before data persistence
        try:
            mgr.ensure_constraints()
        except Exception as exc:
            logger.error("Failed to ensure uniqueness constraints before persistence: %s", exc)
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                committed_nodes=0,
                committed_relationships=0,
                artifact_path=path_str,
                error_category="constraint_creation_failed",
                error_message=f"Failed to ensure Neo4j uniqueness constraint: {exc}",
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

        # 12. Generate parameterized Cypher export from complete nodes and edges
        export_payload = mgr.generate_cypher_export(
            nodes=nodes,
            edges=edges,
            batch_size=batch_size,
            top_influencers=payload.get("top_influencers"),
        )

        # 13. Execute all batches inside single explicit transaction
        try:
            persistence_result = mgr.execute_cypher_batches(export_payload)
        except Exception as exc:
            logger.error("Exception during Cypher batch transaction: %s", exc)
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                committed_nodes=0,
                committed_relationships=0,
                artifact_path=path_str,
                error_category=type(exc).__name__,
                error_message=f"Exception during Cypher batch transaction: {exc}",
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

        # 14. Explicitly validate connector result
        if not isinstance(persistence_result, dict) or persistence_result.get("status") != "success":
            err_cat = (
                persistence_result.get("error_category", "database_persistence_error")
                if isinstance(persistence_result, dict)
                else "unknown_connector_error"
            )
            err_msg = (
                persistence_result.get("error", "Neo4j persistence transaction failed")
                if isinstance(persistence_result, dict)
                else f"Unexpected connector result type: {type(persistence_result).__name__}"
            )
            logger.error("Neo4j persistence failed: %s (%s)", err_msg, err_cat)
            return _build_execution_report(
                status="failed",
                source_status=source_status,
                persistence_outcome="failed",
                total_graph_nodes=total_nodes,
                total_graph_relationships=total_edges,
                committed_nodes=0,
                committed_relationships=0,
                executed_user_batches=0,
                executed_relationship_batches=0,
                artifact_path=path_str,
                error_category=err_cat,
                error_message=err_msg,
                provenance=provenance,
                duration_seconds=time.perf_counter() - start_time,
            )

        # 15. Successful persistence outcome
        committed_nodes = persistence_result.get("total_nodes_ingested", total_nodes)
        committed_edges = persistence_result.get("total_edges_ingested", total_edges)
        user_batches_count = persistence_result.get("executed_user_batches", len(export_payload.get("user_batches", [])))
        rel_batches_count = persistence_result.get("executed_relationship_batches", len(export_payload.get("relationship_batches", [])))

        logger.info(
            "Stage 8 Neo4j persistence completed successfully (%s source). %d nodes, %d edges persisted in %d user batches and %d relationship batches.",
            source_status,
            committed_nodes,
            committed_edges,
            user_batches_count,
            rel_batches_count,
        )

        return _build_execution_report(
            status="success",
            source_status=source_status,
            persistence_outcome="success",
            total_graph_nodes=total_nodes,
            total_graph_relationships=total_edges,
            committed_nodes=committed_nodes,
            committed_relationships=committed_edges,
            executed_user_batches=user_batches_count,
            executed_relationship_batches=rel_batches_count,
            artifact_path=path_str,
            provenance=provenance,
            duration_seconds=time.perf_counter() - start_time,
        )

    finally:
        if owned_manager and mgr is not None:
            try:
                mgr.close()
            except Exception as exc:
                logger.debug("Error closing internally owned Neo4j manager: %s", exc)

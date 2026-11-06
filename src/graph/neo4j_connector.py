"""Neo4j Graph Manager & Cypher Batch Ingestion (Company Step 5).

Manages Neo4j connections and builds batched, parameterized Cypher statements using UNWIND
for scalable graph persistence. Supports mock driver/session injection for offline execution.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = ""


class Neo4jGraphManager:
    """Manages Neo4j graph operations, parameterized Cypher queries, and batched UNWIND ingestion."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        driver_override: Optional[Any] = None,
    ) -> None:
        """Initializes Neo4jGraphManager.

        Args:
            uri: Neo4j Bolt connection URI.
            user: Neo4j username.
            password: Neo4j password.
            driver_override: Optional mock driver for offline testing.
        """
        self.uri = uri or os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI)
        self.user = user or os.getenv("NEO4J_USER", DEFAULT_NEO4J_USER)
        self.password = password or os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD)
        self.driver_override = driver_override
        self._driver: Optional[Any] = driver_override

    def get_driver(self) -> Any:
        """Lazily creates or returns the Neo4j driver instance.

        Raises:
            ValueError: If password is empty or not provided.
            RuntimeError: If neo4j package is missing.
        """
        if self._driver is not None:
            return self._driver

        if not self.password:
            raise ValueError("NEO4J_PASSWORD is required to connect to a live Neo4j database.")

        try:
            from neo4j import GraphDatabase  # type: ignore
        except ImportError as exc:
            logger.error("neo4j package is not installed: %s", exc)
            raise RuntimeError(
                "The 'neo4j' package is required for live Neo4j persistence. "
                "Install neo4j or provide driver_override for testing."
            ) from exc

        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        if self._driver is not None and hasattr(self._driver, "close"):
            self._driver.close()
            self._driver = None

    @staticmethod
    def build_user_unwind_query() -> str:
        """Builds parameterized Cypher query for batched User node creation."""
        return (
            "UNWIND $batch AS row "
            "MERGE (u:User {username: row.username}) "
            "ON CREATE SET u.platform = row.platform, "
            "u.activity_count = row.activity_count, "
            "u.community_id = row.community_id, "
            "u.influencer_score = row.influencer_score "
            "ON MATCH SET u.activity_count = row.activity_count, "
            "u.community_id = row.community_id, "
            "u.influencer_score = row.influencer_score"
        )

    @staticmethod
    def build_content_unwind_query() -> str:
        """Builds parameterized Cypher query for batched Content node creation."""
        return (
            "UNWIND $batch AS row "
            "MERGE (c:Content {id: row.id}) "
            "ON CREATE SET c.title = row.title, "
            "c.platform = row.platform"
        )

    @staticmethod
    def build_relationship_unwind_query() -> str:
        """Builds parameterized Cypher query for batched directed relationship creation."""
        return (
            "UNWIND $batch AS row "
            "MATCH (source:User {username: row.source}) "
            "MATCH (target:User {username: row.target}) "
            "MERGE (source)-[r:INTERACTED_WITH]->(target) "
            "ON CREATE SET r.weight = row.weight, "
            "r.platforms = row.platforms, "
            "r.interaction_types = row.interaction_types "
            "ON MATCH SET r.weight = row.weight, "
            "r.platforms = row.platforms, "
            "r.interaction_types = row.interaction_types"
        )

    def generate_cypher_export(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> dict[str, Any]:
        """Generates parameterized Cypher batches and execution plans for export or ingestion.

        Args:
            nodes: List of user node dictionaries (with username, influencer_score, community_id).
            edges: List of directed edge dictionaries (with source, target, weight, platforms).
            batch_size: Number of entities per parameterized batch.

        Returns:
            Dictionary containing Cypher queries, parameters, and batch counts.
        """
        # Prepare node batches
        node_batches: list[list[dict[str, Any]]] = []
        clean_nodes = [
            {
                "username": str(n.get("username") or n.get("id")),
                "platform": n.get("platform", "social"),
                "activity_count": n.get("activity_count", 1),
                "community_id": n.get("community_id", 0),
                "influencer_score": n.get("influencer_score", 0.0),
            }
            for n in nodes
        ]

        for i in range(0, len(clean_nodes), batch_size):
            node_batches.append(clean_nodes[i : i + batch_size])

        # Prepare edge batches
        edge_batches: list[list[dict[str, Any]]] = []
        clean_edges = [
            {
                "source": str(e.get("source")),
                "target": str(e.get("target")),
                "weight": int(e.get("weight", 1)),
                "platforms": e.get("platforms", []),
                "interaction_types": e.get("interaction_types", ["INTERACTION"]),
            }
            for e in edges
            if e.get("source") and e.get("target")
        ]

        for i in range(0, len(clean_edges), batch_size):
            edge_batches.append(clean_edges[i : i + batch_size])

        return {
            "user_query": self.build_user_unwind_query(),
            "user_batches": node_batches,
            "total_user_batches": len(node_batches),
            "total_users": len(clean_nodes),
            "relationship_query": self.build_relationship_unwind_query(),
            "relationship_batches": edge_batches,
            "total_relationship_batches": len(edge_batches),
            "total_relationships": len(clean_edges),
        }

    def execute_cypher_batches(
        self,
        export_payload: dict[str, Any],
        session: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Executes prepared Cypher batches against a live or mock Neo4j session.

        Args:
            export_payload: Payload returned by generate_cypher_export.
            session: Optional Neo4j session instance (e.g. from driver or mock).

        Returns:
            Dictionary with execution summary counts and status.
        """
        user_query = export_payload.get("user_query", "")
        rel_query = export_payload.get("relationship_query", "")
        user_batches = export_payload.get("user_batches", [])
        rel_batches = export_payload.get("relationship_batches", [])

        sess = session
        driver = None
        if sess is None:
            driver = self.get_driver()
            sess = driver.session()

        executed_user_batches = 0
        executed_rel_batches = 0

        try:
            for b in user_batches:
                sess.run(user_query, batch=b)
                executed_user_batches += 1

            for b in rel_batches:
                sess.run(rel_query, batch=b)
                executed_rel_batches += 1
        except Exception as exc:
            logger.error("Cypher batch execution failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "executed_user_batches": executed_user_batches,
                "executed_relationship_batches": executed_rel_batches,
            }
        finally:
            if session is None and sess is not None and hasattr(sess, "close"):
                sess.close()

        return {
            "status": "success",
            "executed_user_batches": executed_user_batches,
            "executed_relationship_batches": executed_rel_batches,
            "total_nodes_ingested": export_payload.get("total_users", 0),
            "total_edges_ingested": export_payload.get("total_relationships", 0),
        }

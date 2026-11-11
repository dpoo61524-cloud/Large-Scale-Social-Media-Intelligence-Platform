"""Neo4j Graph Manager & Cypher Batch Ingestion (Company Step 5).

Manages Neo4j connections, creates uniqueness constraints, builds parameterized Cypher
statements using UNWIND and canonical node_id matching, and executes ingestion inside
single explicit transactions with rollback guarantees. Supports mock driver/session
injection for offline execution.
"""

from __future__ import annotations

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
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver_override: Optional[Any] = None,
    ) -> None:
        """Initializes Neo4jGraphManager.

        Precedence for username resolution:
        1. Explicit username argument
        2. Explicit user argument (legacy)
        3. NEO4J_USERNAME environment variable
        4. NEO4J_USER environment variable
        5. DEFAULT_NEO4J_USER default ("neo4j")

        Args:
            uri: Neo4j Bolt connection URI.
            user: Neo4j username (legacy argument).
            username: Neo4j username (preferred argument).
            password: Neo4j password.
            driver_override: Optional mock driver for offline testing.
        """
        self.uri = uri or os.getenv("NEO4J_URI", DEFAULT_NEO4J_URI)

        explicit_user = username if username is not None else user
        if explicit_user is not None:
            self.user = explicit_user
        else:
            self.user = (
                os.getenv("NEO4J_USERNAME")
                or os.getenv("NEO4J_USER")
                or DEFAULT_NEO4J_USER
            )

        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD", DEFAULT_NEO4J_PASSWORD)
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

    def ensure_constraints(self, session: Optional[Any] = None) -> str:
        """Creates or ensures uniqueness constraint on User.node_id using modern Neo4j syntax.

        Args:
            session: Optional live or mock Neo4j session.

        Returns:
            The Cypher constraint statement executed.
        """
        query = (
            "CREATE CONSTRAINT user_node_id_unique IF NOT EXISTS "
            "FOR (u:User) REQUIRE u.node_id IS UNIQUE"
        )
        sess = session
        close_session = False
        if sess is None:
            driver = self.get_driver()
            sess = driver.session()
            close_session = True

        try:
            sess.run(query)
            return query
        except Exception as exc:
            logger.error("Failed to ensure Neo4j uniqueness constraint: %s", exc)
            raise
        finally:
            if close_session and sess is not None and hasattr(sess, "close"):
                sess.close()

    @staticmethod
    def build_user_unwind_query() -> str:
        """Builds parameterized Cypher query for batched User node creation using canonical node_id."""
        return (
            "UNWIND $batch AS row "
            "MERGE (u:User {node_id: row.node_id}) "
            "ON CREATE SET "
            "u.platform = row.platform, "
            "u.identity_type = row.identity_type, "
            "u.username = row.username, "
            "u.channel_id = row.channel_id, "
            "u.channel_title = row.channel_title, "
            "u.is_author = row.is_author, "
            "u.activity_count = row.activity_count, "
            "u.community_id = row.community_id, "
            "u.influencer_score = row.influencer_score, "
            "u.pagerank = row.pagerank, "
            "u.in_degree = row.in_degree, "
            "u.betweenness = row.betweenness "
            "ON MATCH SET "
            "u.platform = row.platform, "
            "u.identity_type = row.identity_type, "
            "u.username = row.username, "
            "u.channel_id = row.channel_id, "
            "u.channel_title = row.channel_title, "
            "u.is_author = row.is_author, "
            "u.activity_count = row.activity_count, "
            "u.community_id = row.community_id, "
            "u.influencer_score = row.influencer_score, "
            "u.pagerank = row.pagerank, "
            "u.in_degree = row.in_degree, "
            "u.betweenness = row.betweenness"
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
        """Builds parameterized Cypher query for batched directed relationship creation using canonical node_id."""
        return (
            "UNWIND $batch AS row "
            "MATCH (source:User {node_id: row.source}) "
            "MATCH (target:User {node_id: row.target}) "
            "MERGE (source)-[r:INTERACTED_WITH]->(target) "
            "ON CREATE SET "
            "r.weight = row.weight, "
            "r.platforms = row.platforms, "
            "r.interaction_types = row.interaction_types "
            "ON MATCH SET "
            "r.weight = row.weight, "
            "r.platforms = row.platforms, "
            "r.interaction_types = row.interaction_types"
        )

    def generate_cypher_export(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        batch_size: int = 500,
        top_influencers: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Generates parameterized Cypher batches and execution plans for export or ingestion.

        Args:
            nodes: List of complete user node dictionaries (with canonical node_id and community_id).
            edges: List of directed edge dictionaries (with source, target, weight, platforms).
            batch_size: Number of entities per parameterized batch.
            top_influencers: Optional list of ranked influencer dictionaries to join ranking metrics.

        Returns:
            Dictionary containing Cypher queries, parameters, and batch counts.
        """
        # Build lookup of top influencer metrics by canonical node_id
        inf_metrics_by_id: dict[str, dict[str, Any]] = {}
        if top_influencers:
            for inf in top_influencers:
                inf_id = inf.get("node_id")
                if inf_id:
                    inf_metrics_by_id[inf_id] = inf

        # Prepare clean node records
        clean_nodes: list[dict[str, Any]] = []
        for n in nodes:
            nid = str(n.get("node_id") or n.get("id") or n.get("username") or "")
            inf_data = inf_metrics_by_id.get(nid, {})

            inf_score = n.get("influencer_score", inf_data.get("influencer_score", 0.0))
            pr_score = n.get("pagerank", inf_data.get("pagerank", 0.0))
            in_deg = n.get("in_degree", inf_data.get("in_degree_centrality", 0.0))
            btw = n.get("betweenness", inf_data.get("betweenness_centrality", 0.0))

            clean_nodes.append({
                "node_id": nid,
                "platform": n.get("platform", "social"),
                "identity_type": n.get("identity_type", "account"),
                "username": str(n.get("username", "") or nid),
                "channel_id": n.get("channel_id"),
                "channel_title": n.get("channel_title"),
                "is_author": bool(n.get("is_author", False)),
                "activity_count": int(n.get("activity_count", 0)),
                "community_id": int(n.get("community_id", 0)),
                "influencer_score": float(inf_score) if inf_score is not None else 0.0,
                "pagerank": float(pr_score) if pr_score is not None else 0.0,
                "in_degree": float(in_deg) if in_deg is not None else 0.0,
                "betweenness": float(btw) if btw is not None else 0.0,
            })

        node_batches: list[list[dict[str, Any]]] = []
        for i in range(0, len(clean_nodes), batch_size):
            node_batches.append(clean_nodes[i : i + batch_size])

        # Prepare clean edge records
        clean_edges: list[dict[str, Any]] = []
        for e in edges:
            src = e.get("source")
            tgt = e.get("target")
            if src and tgt:
                clean_edges.append({
                    "source": str(src),
                    "target": str(tgt),
                    "weight": int(e.get("weight", 1)),
                    "platforms": list(e.get("platforms", [])),
                    "interaction_types": list(e.get("interaction_types", ["INTERACTION"])),
                })

        edge_batches: list[list[dict[str, Any]]] = []
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
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        """Executes prepared Cypher batches within a single explicit transaction.

        Enforces all-or-nothing atomicity:
        - Opens a session if not provided.
        - Begins an explicit transaction.
        - Executes all user batches followed by all relationship batches.
        - Commits only when all batches succeed.
        - Rolls back immediately on any error and reports 0 committed records.
        - Closes session reliably.

        Args:
            export_payload: Payload returned by generate_cypher_export.
            session: Optional live or mock Neo4j session.
            raise_on_error: If True, re-raises the caught exception after rollback.

        Returns:
            Dictionary with execution summary counts and status.
        """
        user_query = export_payload.get("user_query", "")
        rel_query = export_payload.get("relationship_query", "")
        user_batches = export_payload.get("user_batches", [])
        rel_batches = export_payload.get("relationship_batches", [])

        sess = session
        close_session = False
        if sess is None:
            driver = self.get_driver()
            sess = driver.session()
            close_session = True

        tx = None
        try:
            # Begin single explicit transaction
            if hasattr(sess, "begin_transaction"):
                tx = sess.begin_transaction()
            else:
                tx = sess

            for b in user_batches:
                tx.run(user_query, batch=b)

            for b in rel_batches:
                tx.run(rel_query, batch=b)

            if hasattr(tx, "commit"):
                tx.commit()

            return {
                "status": "success",
                "executed_user_batches": len(user_batches),
                "executed_relationship_batches": len(rel_batches),
                "total_nodes_ingested": export_payload.get("total_users", 0),
                "total_edges_ingested": export_payload.get("total_relationships", 0),
            }

        except Exception as exc:
            logger.error("Cypher batch transaction failed: %s", exc)
            if tx is not None and hasattr(tx, "rollback"):
                try:
                    tx.rollback()
                except Exception as rb_exc:
                    logger.error("Failed to rollback Neo4j transaction: %s", rb_exc)

            if raise_on_error:
                raise

            return {
                "status": "error",
                "error": str(exc),
                "error_category": type(exc).__name__,
                "executed_user_batches": 0,
                "executed_relationship_batches": 0,
                "total_nodes_ingested": 0,
                "total_edges_ingested": 0,
            }
        finally:
            if close_session and sess is not None and hasattr(sess, "close"):
                sess.close()

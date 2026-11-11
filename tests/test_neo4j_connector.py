"""Unit tests for Neo4jGraphManager & Cypher Batch Ingestion (Task 8A).

Verifies canonical node_id matching, parameterized Cypher generation, full-graph export,
identity metadata preservation, modern constraint syntax, configuration precedence,
and single-transaction execution with automatic rollback on error.
"""

from __future__ import annotations

from typing import Any
import pytest

from src.graph.neo4j_connector import Neo4jGraphManager


# ==============================================================================
# Mock Neo4j Test Doubles
# ==============================================================================

class MockNeo4jTransaction:
    """Mock Neo4j transaction for verifying single-transaction commit and rollback."""

    def __init__(
        self,
        session: MockNeo4jSession,
        fail_on_user_batch: bool = False,
        fail_on_rel_batch: bool = False,
    ) -> None:
        self.session = session
        self.fail_on_user_batch = fail_on_user_batch
        self.fail_on_rel_batch = fail_on_rel_batch
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def run(self, query: str, batch: list[dict[str, Any]] | None = None) -> None:
        if self.fail_on_user_batch and "MERGE (u:User" in query:
            raise RuntimeError("Injected user-batch execution failure")
        if self.fail_on_rel_batch and "INTERACTED_WITH" in query:
            raise RuntimeError("Injected relationship-batch execution failure")

        self.session.executed_queries.append(query)
        if batch:
            self.session.executed_batches.append(batch)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


class MockNeo4jSession:
    """Mock Neo4j session for offline execution verification."""

    def __init__(
        self,
        fail_on_user_batch: bool = False,
        fail_on_rel_batch: bool = False,
    ) -> None:
        self.executed_queries: list[str] = []
        self.executed_batches: list[list[dict[str, Any]]] = []
        self.closed = False
        self.fail_on_user_batch = fail_on_user_batch
        self.fail_on_rel_batch = fail_on_rel_batch
        self.current_transaction: MockNeo4jTransaction | None = None

    def begin_transaction(self) -> MockNeo4jTransaction:
        self.current_transaction = MockNeo4jTransaction(
            session=self,
            fail_on_user_batch=self.fail_on_user_batch,
            fail_on_rel_batch=self.fail_on_rel_batch,
        )
        return self.current_transaction

    def run(self, query: str, batch: list[dict[str, Any]] | None = None) -> None:
        self.executed_queries.append(query)
        if batch:
            self.executed_batches.append(batch)

    def close(self) -> None:
        self.closed = True


class MockNeo4jDriver:
    """Mock Neo4j driver for offline execution verification."""

    def __init__(self, session: MockNeo4jSession | None = None) -> None:
        self.session_instance = session or MockNeo4jSession()
        self.closed = False

    def session(self) -> MockNeo4jSession:
        return self.session_instance

    def close(self) -> None:
        self.closed = True


# ==============================================================================
# 1. Parameterized Cypher Syntax Validation (node_id based)
# ==============================================================================

def test_cypher_user_unwind_query_uses_node_id() -> None:
    """Tests syntax of the batched User UNWIND query uses canonical node_id key."""
    query = Neo4jGraphManager.build_user_unwind_query()
    assert "UNWIND $batch AS row" in query
    assert "MERGE (u:User {node_id: row.node_id})" in query
    assert "ON CREATE SET" in query
    assert "u.platform = row.platform" in query
    assert "u.identity_type = row.identity_type" in query
    assert "u.username = row.username" in query
    assert "u.community_id = row.community_id" in query
    assert "u.influencer_score = row.influencer_score" in query
    assert "ON MATCH SET" in query


def test_cypher_content_unwind_query_syntax() -> None:
    """Tests syntax of the batched Content UNWIND query."""
    query = Neo4jGraphManager.build_content_unwind_query()
    assert "UNWIND $batch AS row" in query
    assert "MERGE (c:Content {id: row.id})" in query


def test_cypher_relationship_unwind_query_uses_node_id() -> None:
    """Tests syntax of the batched directed relationship UNWIND query uses node_id."""
    query = Neo4jGraphManager.build_relationship_unwind_query()
    assert "UNWIND $batch AS row" in query
    assert "MATCH (source:User {node_id: row.source})" in query
    assert "MATCH (target:User {node_id: row.target})" in query
    assert "MERGE (source)-[r:INTERACTED_WITH]->(target)" in query
    assert "ON CREATE SET r.weight = row.weight" in query
    assert "ON MATCH SET r.weight = row.weight" in query


# ==============================================================================
# 2. Identity Disambiguation & Metadata Preservation
# ==============================================================================

def test_cross_platform_same_display_username_remains_distinct() -> None:
    """Tests that Reddit and YouTube users sharing a display username remain separate nodes."""
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
            "node_id": "youtube:channel:UC_ALICE_DATA",
            "platform": "youtube",
            "identity_type": "creator",
            "username": "Alice_Data",
            "channel_id": "UC_ALICE_DATA",
            "channel_title": "Alice_Data",
            "is_author": True,
            "activity_count": 1,
            "community_id": 1,
        },
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=nodes, edges=[], batch_size=10)
    batch = export["user_batches"][0]

    assert len(batch) == 2
    assert batch[0]["node_id"] == "reddit:user:alice_data"
    assert batch[0]["platform"] == "reddit"
    assert batch[1]["node_id"] == "youtube:channel:UC_ALICE_DATA"
    assert batch[1]["platform"] == "youtube"
    assert batch[0]["node_id"] != batch[1]["node_id"]


def test_youtube_creator_and_mention_identities_distinct() -> None:
    """Tests that a YouTube creator and a YouTube mention remain separate nodes."""
    nodes = [
        {
            "node_id": "youtube:channel:UC_TECHGURU",
            "platform": "youtube",
            "identity_type": "creator",
            "username": "TechGuru",
            "channel_id": "UC_TECHGURU",
            "channel_title": "TechGuru",
            "is_author": True,
            "activity_count": 5,
            "community_id": 0,
        },
        {
            "node_id": "youtube:mention:techguru",
            "platform": "youtube",
            "identity_type": "mention",
            "username": "techguru",
            "channel_id": None,
            "channel_title": None,
            "is_author": False,
            "activity_count": 0,
            "community_id": 1,
        },
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=nodes, edges=[], batch_size=10)
    batch = export["user_batches"][0]

    assert len(batch) == 2
    assert batch[0]["identity_type"] == "creator"
    assert batch[0]["channel_id"] == "UC_TECHGURU"
    assert batch[1]["identity_type"] == "mention"
    assert batch[1]["channel_id"] is None
    assert batch[0]["node_id"] != batch[1]["node_id"]


def test_export_preserves_node_metadata() -> None:
    """Tests that all required node metadata fields are preserved in export batches."""
    nodes = [
        {
            "node_id": "youtube:channel:UC_123",
            "platform": "youtube",
            "identity_type": "creator",
            "username": "Channel123",
            "channel_id": "UC_123",
            "channel_title": "Title 123",
            "is_author": True,
            "activity_count": 7,
            "community_id": 3,
        }
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=nodes, edges=[], batch_size=10)
    record = export["user_batches"][0][0]

    assert record["node_id"] == "youtube:channel:UC_123"
    assert record["platform"] == "youtube"
    assert record["identity_type"] == "creator"
    assert record["username"] == "Channel123"
    assert record["channel_id"] == "UC_123"
    assert record["channel_title"] == "Title 123"
    assert record["is_author"] is True
    assert record["activity_count"] == 7
    assert record["community_id"] == 3


def test_complete_graph_nodes_exported_not_only_top_influencers() -> None:
    """Tests that every graph node is included in the export, not just top influencers."""
    nodes = [
        {"node_id": f"reddit:user:user_{i}", "platform": "reddit", "community_id": i % 3}
        for i in range(25)
    ]
    top_inf = [
        {"node_id": "reddit:user:user_0", "influencer_score": 0.95, "pagerank": 0.4},
        {"node_id": "reddit:user:user_1", "influencer_score": 0.85, "pagerank": 0.3},
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(
        nodes=nodes,
        edges=[],
        batch_size=10,
        top_influencers=top_inf,
    )

    assert export["total_users"] == 25
    assert export["total_user_batches"] == 3
    exported_ids = [row["node_id"] for b in export["user_batches"] for row in b]
    assert len(exported_ids) == 25
    assert set(exported_ids) == {f"reddit:user:user_{i}" for i in range(25)}


def test_complete_graph_edges_in_relationship_batches() -> None:
    """Tests that all valid graph edges are represented in relationship batches."""
    edges = [
        {
            "source": f"reddit:user:user_{i}",
            "target": f"reddit:user:user_{i+1}",
            "weight": i + 1,
            "platforms": ["reddit"],
            "interaction_types": ["MENTIONED"],
        }
        for i in range(12)
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=[], edges=edges, batch_size=5)

    assert export["total_relationships"] == 12
    assert export["total_relationship_batches"] == 3
    assert len(export["relationship_batches"][0]) == 5
    assert len(export["relationship_batches"][1]) == 5
    assert len(export["relationship_batches"][2]) == 2


def test_influencer_metrics_join_by_node_id() -> None:
    """Tests that ranking metrics are joined to node records strictly by node_id."""
    nodes = [
        {"node_id": "reddit:user:alpha", "community_id": 0},
        {"node_id": "reddit:user:beta", "community_id": 0},
    ]
    top_inf = [
        {
            "node_id": "reddit:user:alpha",
            "influencer_score": 0.77,
            "pagerank": 0.33,
            "in_degree_centrality": 0.44,
            "betweenness_centrality": 0.11,
        }
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=nodes, edges=[], top_influencers=top_inf)
    batch = export["user_batches"][0]

    alpha_row = next(r for r in batch if r["node_id"] == "reddit:user:alpha")
    beta_row = next(r for r in batch if r["node_id"] == "reddit:user:beta")

    assert alpha_row["influencer_score"] == 0.77
    assert alpha_row["pagerank"] == 0.33
    assert alpha_row["in_degree"] == 0.44
    assert alpha_row["betweenness"] == 0.11

    # Non-top influencer receives neutral 0.0 defaults
    assert beta_row["influencer_score"] == 0.0
    assert beta_row["pagerank"] == 0.0
    assert beta_row["in_degree"] == 0.0
    assert beta_row["betweenness"] == 0.0


def test_missing_top_influencer_metrics_receive_neutral_defaults() -> None:
    """Tests that nodes absent from top_influencers retain identity and community with 0.0 metrics."""
    nodes = [
        {
            "node_id": "reddit:user:ordinary",
            "platform": "reddit",
            "identity_type": "account",
            "username": "OrdinaryUser",
            "community_id": 2,
        }
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=nodes, edges=[], top_influencers=[])
    row = export["user_batches"][0][0]

    assert row["node_id"] == "reddit:user:ordinary"
    assert row["community_id"] == 2
    assert row["influencer_score"] == 0.0
    assert row["pagerank"] == 0.0


def test_no_fabricated_channel_id_or_identity() -> None:
    """Tests that None channel_id is preserved and never fabricated into a string."""
    nodes = [
        {
            "node_id": "reddit:user:author_only",
            "platform": "reddit",
            "identity_type": "account",
            "username": "AuthorOnly",
            "channel_id": None,
            "channel_title": None,
        }
    ]

    manager = Neo4jGraphManager()
    export = manager.generate_cypher_export(nodes=nodes, edges=[])
    row = export["user_batches"][0][0]

    assert row["channel_id"] is None
    assert row["channel_title"] is None


# ==============================================================================
# 3. Modern Uniqueness Constraint
# ==============================================================================

def test_exact_modern_uniqueness_constraint_cypher() -> None:
    """Tests that ensure_constraints issues the exact modern Neo4j 5.x / 4.4 syntax."""
    mock_sess = MockNeo4jSession()
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    query = manager.ensure_constraints(session=mock_sess)

    expected = (
        "CREATE CONSTRAINT user_node_id_unique IF NOT EXISTS "
        "FOR (u:User) REQUIRE u.node_id IS UNIQUE"
    )
    assert query == expected
    assert expected in mock_sess.executed_queries


# ==============================================================================
# 4. Configuration Compatibility & Precedence
# ==============================================================================

def test_neo4j_username_precedence_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that NEO4J_USERNAME takes precedence over legacy NEO4J_USER."""
    monkeypatch.setenv("NEO4J_USERNAME", "preferred_user")
    monkeypatch.setenv("NEO4J_USER", "legacy_user")

    manager = Neo4jGraphManager()
    assert manager.user == "preferred_user"


def test_legacy_neo4j_user_works_when_preferred_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that legacy NEO4J_USER is used when NEO4J_USERNAME is not set."""
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.setenv("NEO4J_USER", "legacy_user")

    manager = Neo4jGraphManager()
    assert manager.user == "legacy_user"


def test_explicit_constructor_username_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that explicit constructor arguments take precedence over environment variables."""
    monkeypatch.setenv("NEO4J_USERNAME", "env_user")
    monkeypatch.setenv("NEO4J_USER", "env_legacy")

    # Preferred username argument
    mgr1 = Neo4jGraphManager(username="explicit_username")
    assert mgr1.user == "explicit_username"

    # Legacy user argument
    mgr2 = Neo4jGraphManager(user="explicit_user")
    assert mgr2.user == "explicit_user"


# ==============================================================================
# 5. Transaction-Safe Batch Execution
# ==============================================================================

def test_single_explicit_transaction_created_for_all_batches() -> None:
    """Tests that one explicit transaction is opened and committed for all batches."""
    mock_sess = MockNeo4jSession()
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    nodes = [
        {"node_id": "reddit:user:u1", "community_id": 0},
        {"node_id": "reddit:user:u2", "community_id": 0},
    ]
    edges = [{"source": "reddit:user:u1", "target": "reddit:user:u2", "weight": 1}]

    export = manager.generate_cypher_export(nodes=nodes, edges=edges, batch_size=1)
    result = manager.execute_cypher_batches(export, session=mock_sess)

    assert result["status"] == "success"
    assert result["executed_user_batches"] == 2
    assert result["executed_relationship_batches"] == 1
    assert result["total_nodes_ingested"] == 2
    assert result["total_edges_ingested"] == 1

    tx = mock_sess.current_transaction
    assert tx is not None
    assert tx.committed is True
    assert tx.rolled_back is False


def test_successful_batches_commit_exactly_once() -> None:
    """Tests that commit occurs exactly once on transaction completion."""
    mock_sess = MockNeo4jSession()
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    nodes = [{"node_id": f"u:{i}", "community_id": 0} for i in range(5)]
    edges = [{"source": "u:0", "target": "u:1", "weight": 1}]

    export = manager.generate_cypher_export(nodes=nodes, edges=edges, batch_size=2)
    result = manager.execute_cypher_batches(export, session=mock_sess)

    assert result["status"] == "success"
    tx = mock_sess.current_transaction
    assert tx is not None
    assert tx.committed is True
    assert tx.rolled_back is False


def test_user_batch_failure_triggers_rollback() -> None:
    """Tests that failure during a user batch triggers rollback and reports zero committed records."""
    mock_sess = MockNeo4jSession(fail_on_user_batch=True)
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    nodes = [{"node_id": "u:1", "community_id": 0}]
    edges = [{"source": "u:1", "target": "u:2", "weight": 1}]

    export = manager.generate_cypher_export(nodes=nodes, edges=edges)
    result = manager.execute_cypher_batches(export, session=mock_sess)

    assert result["status"] == "error"
    assert "Injected user-batch execution failure" in result["error"]
    assert result["total_nodes_ingested"] == 0
    assert result["total_edges_ingested"] == 0
    assert result["executed_user_batches"] == 0
    assert result["executed_relationship_batches"] == 0

    tx = mock_sess.current_transaction
    assert tx is not None
    assert tx.rolled_back is True
    assert tx.committed is False


def test_relationship_batch_failure_after_user_batches_rolls_back_whole_transaction() -> None:
    """Tests that relationship failure after user batches rolled back entire transaction."""
    mock_sess = MockNeo4jSession(fail_on_rel_batch=True)
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    nodes = [
        {"node_id": "u:1", "community_id": 0},
        {"node_id": "u:2", "community_id": 0},
    ]
    edges = [{"source": "u:1", "target": "u:2", "weight": 1}]

    export = manager.generate_cypher_export(nodes=nodes, edges=edges, batch_size=1)
    result = manager.execute_cypher_batches(export, session=mock_sess)

    assert result["status"] == "error"
    assert "Injected relationship-batch execution failure" in result["error"]
    assert result["total_nodes_ingested"] == 0
    assert result["total_edges_ingested"] == 0
    assert result["executed_user_batches"] == 0
    assert result["executed_relationship_batches"] == 0

    tx = mock_sess.current_transaction
    assert tx is not None
    assert tx.rolled_back is True
    assert tx.committed is False


def test_no_commit_occurs_after_failure() -> None:
    """Verifies that no commit is ever called when an error occurs."""
    mock_sess = MockNeo4jSession(fail_on_user_batch=True)
    manager = Neo4jGraphManager(driver_override=MockNeo4jDriver(mock_sess))

    export = manager.generate_cypher_export(
        nodes=[{"node_id": "u:fail", "community_id": 0}], edges=[]
    )
    manager.execute_cypher_batches(export, session=mock_sess)

    tx = mock_sess.current_transaction
    assert tx is not None
    assert tx.committed is False


def test_rollback_reports_zero_persisted_counts() -> None:
    """Verifies that rollback report contains strictly 0 for all record counts."""
    mock_sess = MockNeo4jSession(fail_on_rel_batch=True)
    manager = Neo4jGraphManager(driver_override=MockNeo4jDriver(mock_sess))

    export = manager.generate_cypher_export(
        nodes=[{"node_id": "u:1", "community_id": 0}],
        edges=[{"source": "u:1", "target": "u:2", "weight": 1}],
    )
    result = manager.execute_cypher_batches(export, session=mock_sess)

    assert result["executed_user_batches"] == 0
    assert result["executed_relationship_batches"] == 0
    assert result["total_nodes_ingested"] == 0
    assert result["total_edges_ingested"] == 0


def test_session_closes_on_success() -> None:
    """Verifies that driver session is closed when execute_cypher_batches manages session on success."""
    mock_sess = MockNeo4jSession()
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    export = manager.generate_cypher_export(
        nodes=[{"node_id": "u:1", "community_id": 0}], edges=[]
    )
    result = manager.execute_cypher_batches(export)  # session=None -> manages session

    assert result["status"] == "success"
    assert mock_sess.closed is True


def test_session_closes_on_failure() -> None:
    """Verifies that driver session is closed when execute_cypher_batches manages session on failure."""
    mock_sess = MockNeo4jSession(fail_on_user_batch=True)
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    export = manager.generate_cypher_export(
        nodes=[{"node_id": "u:1", "community_id": 0}], edges=[]
    )
    result = manager.execute_cypher_batches(export)

    assert result["status"] == "error"
    assert mock_sess.closed is True


def test_empty_graph_behavior_safe_and_deterministic() -> None:
    """Tests that exporting and executing an empty graph is safe and issues no invalid queries."""
    mock_sess = MockNeo4jSession()
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    export = manager.generate_cypher_export(nodes=[], edges=[])
    assert export["total_users"] == 0
    assert export["total_relationships"] == 0
    assert export["total_user_batches"] == 0
    assert export["total_relationship_batches"] == 0

    result = manager.execute_cypher_batches(export, session=mock_sess)
    assert result["status"] == "success"
    assert result["total_nodes_ingested"] == 0
    assert result["total_edges_ingested"] == 0

    tx = mock_sess.current_transaction
    assert tx is not None
    assert tx.committed is True


# ==============================================================================
# 6. Driver Lifecycle & Error Handling
# ==============================================================================

def test_missing_password_raises_value_error() -> None:
    """Tests that connecting without password when no override is given raises ValueError."""
    manager = Neo4jGraphManager(password="")
    with pytest.raises(ValueError, match="NEO4J_PASSWORD is required"):
        manager.get_driver()


def test_missing_neo4j_package_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that connecting without neo4j package raises RuntimeError."""
    import builtins
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "neo4j":
            raise ImportError("No module named 'neo4j'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    manager = Neo4jGraphManager(password="secret_pass")
    with pytest.raises(RuntimeError, match="The 'neo4j' package is required"):
        manager.get_driver()


def test_close_driver() -> None:
    """Tests driver close lifecycle method."""
    mock_driver = MockNeo4jDriver()
    manager = Neo4jGraphManager(driver_override=mock_driver)
    manager.close()
    assert mock_driver.closed is True
    assert manager._driver is None

"""Unit tests for Neo4jGraphManager (Company Step 5).

Verifies Cypher query generation (UNWIND $batch AS row), parameter batching,
mock session execution, error handling, and driver lifecycle management.
"""

from typing import Any
import pytest

from src.graph.neo4j_connector import Neo4jGraphManager


class MockNeo4jSession:
    """Mock Neo4j session for offline execution verification."""

    def __init__(self) -> None:
        self.executed_queries: list[str] = []
        self.executed_batches: list[list[dict[str, Any]]] = []
        self.closed = False

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
# 1. Parameterized Cypher Syntax Validation
# ==============================================================================

def test_cypher_user_unwind_query_syntax() -> None:
    """Tests syntax of the batched User UNWIND query."""
    query = Neo4jGraphManager.build_user_unwind_query()
    assert "UNWIND $batch AS row" in query
    assert "MERGE (u:User {username: row.username})" in query
    assert "ON CREATE SET" in query
    assert "ON MATCH SET" in query


def test_cypher_content_unwind_query_syntax() -> None:
    """Tests syntax of the batched Content UNWIND query."""
    query = Neo4jGraphManager.build_content_unwind_query()
    assert "UNWIND $batch AS row" in query
    assert "MERGE (c:Content {id: row.id})" in query


def test_cypher_relationship_unwind_query_syntax() -> None:
    """Tests syntax of the batched directed relationship UNWIND query."""
    query = Neo4jGraphManager.build_relationship_unwind_query()
    assert "UNWIND $batch AS row" in query
    assert "MATCH (source:User {username: row.source})" in query
    assert "MATCH (target:User {username: row.target})" in query
    assert "MERGE (source)-[r:INTERACTED_WITH]->(target)" in query


# ==============================================================================
# 2. Batch Chunking & Parameter Mapping
# ==============================================================================

def test_generate_cypher_export_batching() -> None:
    """Tests that nodes and edges are correctly partitioned into batch_size chunks."""
    nodes = [{"username": f"user_{i}", "platform": "reddit"} for i in range(25)]
    edges = [
        {"source": f"user_{i}", "target": f"user_{i+1}", "weight": 1}
        for i in range(24)
    ]

    manager = Neo4jGraphManager()
    # Batch size of 10 -> 3 node batches (10, 10, 5), 3 edge batches (10, 10, 4)
    export_data = manager.generate_cypher_export(nodes, edges, batch_size=10)

    assert export_data["total_users"] == 25
    assert export_data["total_user_batches"] == 3
    assert len(export_data["user_batches"][0]) == 10
    assert len(export_data["user_batches"][2]) == 5

    assert export_data["total_relationships"] == 24
    assert export_data["total_relationship_batches"] == 3
    assert len(export_data["relationship_batches"][2]) == 4


# ==============================================================================
# 3. Batch Execution via Mock Driver
# ==============================================================================

def test_execute_cypher_batches_with_mock_session() -> None:
    """Tests execution of batches using a mock Neo4j session."""
    mock_sess = MockNeo4jSession()
    mock_driver = MockNeo4jDriver(mock_sess)
    manager = Neo4jGraphManager(driver_override=mock_driver)

    nodes = [{"username": "Alice"}, {"username": "Bob"}]
    edges = [{"source": "Alice", "target": "Bob", "weight": 2}]

    export_payload = manager.generate_cypher_export(nodes, edges, batch_size=10)
    result = manager.execute_cypher_batches(export_payload, session=mock_sess)

    assert result["status"] == "success"
    assert result["executed_user_batches"] == 1
    assert result["executed_relationship_batches"] == 1
    assert len(mock_sess.executed_queries) == 2
    assert len(mock_sess.executed_batches) == 2


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

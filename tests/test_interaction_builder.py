"""Unit tests for InteractionGraphBuilder (Company Step 5).

Verifies user validation, mention extraction, graph construction, edge weight aggregation,
anonymous node filtering, and heterogeneous node/edge representations.
"""

import networkx as nx
import pytest

from src.graph.interaction_builder import InteractionGraphBuilder


# ==============================================================================
# 1. User Validation & Filtering Tests
# ==============================================================================

def test_is_valid_user() -> None:
    """Tests filtering of anonymous, deleted, bot, and empty usernames."""
    assert InteractionGraphBuilder.is_valid_user("data_engineer_99") is True
    assert InteractionGraphBuilder.is_valid_user("Alice") is True

    # Filtered accounts
    assert InteractionGraphBuilder.is_valid_user("[deleted]") is False
    assert InteractionGraphBuilder.is_valid_user("[removed]") is False
    assert InteractionGraphBuilder.is_valid_user("AutoModerator") is False
    assert InteractionGraphBuilder.is_valid_user("deleted") is False
    assert InteractionGraphBuilder.is_valid_user("") is False
    assert InteractionGraphBuilder.is_valid_user(None) is False
    assert InteractionGraphBuilder.is_valid_user("   ") is False


# ==============================================================================
# 2. Mention Extraction Tests
# ==============================================================================

def test_extract_mentions_reddit_and_youtube() -> None:
    """Tests extracting u/username and @username mentions from text."""
    text = "Thanks u/data_guru and @tech_channel for the great tutorial! Also cc u/pipeline_dev."
    mentions = InteractionGraphBuilder.extract_mentions(text)

    assert "data_guru" in mentions
    assert "tech_channel" in mentions
    assert "pipeline_dev" in mentions
    assert len(mentions) == 3


def test_extract_mentions_empty_and_invalid() -> None:
    """Tests mention extraction on empty, non-string, or text without mentions."""
    assert InteractionGraphBuilder.extract_mentions("") == []
    assert InteractionGraphBuilder.extract_mentions("No mentions here.") == []
    assert InteractionGraphBuilder.extract_mentions(None) == []  # type: ignore


# ==============================================================================
# 3. Graph Construction & Multi-Edge Weight Aggregation
# ==============================================================================

def test_build_interaction_graph_from_reddit_and_youtube() -> None:
    """Tests building directed interaction graph with aggregated edge weights."""
    reddit_records = [
        {
            "post_id": "r1",
            "author": "Alice",
            "title_cleaned": "Discussing neural nets with u/Bob",
            "text_cleaned": "Great points by u/Bob and u/Charlie.",
        },
        {
            "post_id": "r2",
            "author": "Alice",
            "title_cleaned": "Second post mentioning u/Bob again",
            "text_cleaned": "Continuing conversation with u/Bob.",
        },
    ]

    youtube_records = [
        {
            "video_id": "y1",
            "channel_title": "David_AI",
            "title_cleaned": "Video on Big Data",
            "description_cleaned": "Shoutout to @Alice for inspiring this project.",
        }
    ]

    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records, youtube_records)

    assert isinstance(graph, nx.DiGraph)
    assert graph.has_node("Alice")
    assert graph.has_node("Bob")
    assert graph.has_node("Charlie")
    assert graph.has_node("David_AI")

    # Alice mentioned Bob in two separate posts -> edge weight must be 2
    assert graph.has_edge("Alice", "Bob")
    assert graph["Alice"]["Bob"]["weight"] == 2
    assert "MENTIONED" in graph["Alice"]["Bob"]["interaction_types"]

    # Alice mentioned Charlie once -> weight 1
    assert graph.has_edge("Alice", "Charlie")
    assert graph["Alice"]["Charlie"]["weight"] == 1

    # David_AI mentioned Alice -> David_AI -> Alice
    assert graph.has_edge("David_AI", "Alice")
    assert graph["David_AI"]["Alice"]["weight"] == 1


def test_build_interaction_graph_filters_anonymous_authors() -> None:
    """Tests that [deleted] authors are not included in the graph."""
    reddit_records = [
        {
            "post_id": "r1",
            "author": "[deleted]",
            "title_cleaned": "Deleted user post mentioning u/Alice",
            "text_cleaned": "Content from deleted author.",
        }
    ]

    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records)

    assert not graph.has_node("[deleted]")
    assert not graph.has_node("deleted")


def test_get_nodes_and_edges() -> None:
    """Tests export methods get_nodes and get_edges."""
    builder = InteractionGraphBuilder()
    builder.user_graph.add_edge("UserA", "UserB", weight=3, platforms=["reddit"], interaction_types=["MENTIONED"])
    builder.user_graph.nodes["UserA"]["platform"] = "reddit"
    builder.user_graph.nodes["UserB"]["platform"] = "reddit"

    nodes = builder.get_nodes()
    edges = builder.get_edges()

    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0]["source"] == "UserA"
    assert edges[0]["target"] == "UserB"
    assert edges[0]["weight"] == 3

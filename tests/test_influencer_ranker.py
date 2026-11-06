"""Unit tests for InfluencerRanker (Company Step 5).

Verifies PageRank convergence, In-Degree Centrality, Betweenness Centrality,
community detection partitioning, and composite influencer scoring.
"""

import networkx as nx
import pytest

from src.graph.influencer_ranker import InfluencerRanker


# ==============================================================================
# 1. Normalization & Edge Cases
# ==============================================================================

def test_normalize_dict_values() -> None:
    """Tests min-max normalization into [0.0, 1.0]."""
    raw = {"A": 10.0, "B": 20.0, "C": 30.0}
    norm = InfluencerRanker._normalize_dict_values(raw)

    assert norm["A"] == 0.0
    assert norm["B"] == 0.5
    assert norm["C"] == 1.0


def test_normalize_identical_values() -> None:
    """Tests normalization when all values are identical."""
    raw = {"A": 5.0, "B": 5.0}
    norm = InfluencerRanker._normalize_dict_values(raw)
    assert norm["A"] == 1.0
    assert norm["B"] == 1.0

    raw_zero = {"A": 0.0, "B": 0.0}
    norm_zero = InfluencerRanker._normalize_dict_values(raw_zero)
    assert norm_zero["A"] == 0.0
    assert norm_zero["B"] == 0.0


def test_rank_influencers_empty_graph() -> None:
    """Tests ranking on empty graph returns empty list."""
    ranker = InfluencerRanker()
    empty_graph = nx.DiGraph()
    res = ranker.rank_influencers(empty_graph)
    assert res == []


# ==============================================================================
# 2. Centrality & Influencer Score Verification
# ==============================================================================

def test_rank_influencers_synthetic_star_graph() -> None:
    """Tests that the central hub node in an inbound star graph achieves highest rank."""
    # Hub node 'Guru' receives directed mentions from 4 peripheral nodes
    graph = nx.DiGraph()
    graph.add_edge("User1", "Guru", weight=2)
    graph.add_edge("User2", "Guru", weight=3)
    graph.add_edge("User3", "Guru", weight=1)
    graph.add_edge("User4", "Guru", weight=1)

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    assert len(ranked) == 5
    top = ranked[0]

    # 'Guru' must be ranked first with highest score
    assert top["username"] == "Guru"
    assert top["influencer_score"] > 0.5
    assert top["in_degree_raw"] == 4
    assert top["out_degree_raw"] == 0
    assert top["in_degree_norm"] == 1.0
    assert top["pagerank_norm"] == 1.0


def test_betweenness_centrality_bridge_node() -> None:
    """Tests that a bridge node between two clusters achieves highest betweenness."""
    # Cluster 1: A -> Bridge, B -> Bridge
    # Cluster 2: Bridge -> C, Bridge -> D
    graph = nx.DiGraph()
    graph.add_edge("A", "Bridge", weight=1)
    graph.add_edge("B", "Bridge", weight=1)
    graph.add_edge("Bridge", "C", weight=1)
    graph.add_edge("Bridge", "D", weight=1)

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    bridge_entry = next(r for r in ranked if r["username"] == "Bridge")
    assert bridge_entry["betweenness_norm"] == 1.0
    assert bridge_entry["betweenness_centrality"] > 0.0


def test_composite_score_formula_weights() -> None:
    """Tests exact composite score calculation: 0.50*PR + 0.30*InDeg + 0.20*Btw."""
    graph = nx.DiGraph()
    graph.add_edge("A", "B", weight=1)

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    for r in ranked:
        expected = round(
            0.50 * r["pagerank_norm"] + 0.30 * r["in_degree_norm"] + 0.20 * r["betweenness_norm"],
            4,
        )
        assert r["influencer_score"] == pytest.approx(expected, abs=0.0001)


# ==============================================================================
# 3. Community Detection Partitioning Tests
# ==============================================================================

def test_community_detection_partitions() -> None:
    """Tests that disconnected or modular cliques are partitioned into communities."""
    graph = nx.DiGraph()
    # Clique 1
    graph.add_edge("A1", "A2", weight=1)
    graph.add_edge("A2", "A1", weight=1)
    # Clique 2
    graph.add_edge("B1", "B2", weight=1)
    graph.add_edge("B2", "B1", weight=1)

    ranker = InfluencerRanker()
    communities = ranker.detect_communities(graph)

    assert len(communities) == 4
    # A1 and A2 should share community; B1 and B2 should share community
    assert communities["A1"] == communities["A2"]
    assert communities["B1"] == communities["B2"]
    assert communities["A1"] != communities["B1"]

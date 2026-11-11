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


# ==============================================================================
# 4. Identity-Aware Ranking & Downstream Compatibility Tests
# ==============================================================================

def test_reddit_ranking_display_username_and_canonical_node_id() -> None:
    """Verifies Reddit ranked entry uses display username, not namespaced node_id, and preserves node_id."""
    graph = nx.DiGraph()
    node_id = "reddit:user:alice_data"
    graph.add_node(
        node_id,
        node_id=node_id,
        platform="reddit",
        identity_type="account",
        username="Alice_Data",
        is_author=True,
        activity_count=5,
        channel_id=None,
        channel_title=None,
    )
    graph.add_edge("reddit:user:bob", node_id, weight=2)
    graph.nodes["reddit:user:bob"].update({
        "node_id": "reddit:user:bob",
        "platform": "reddit",
        "identity_type": "account",
        "username": "Bob",
        "is_author": False,
        "activity_count": 0,
        "channel_id": None,
        "channel_title": None,
    })

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    alice_entry = next(r for r in ranked if r["node_id"] == node_id)
    assert alice_entry["username"] == "Alice_Data"
    assert alice_entry["username"] != node_id
    assert alice_entry["node_id"] == "reddit:user:alice_data"
    assert alice_entry["platform"] == "reddit"
    assert alice_entry["identity_type"] == "account"
    assert alice_entry["is_author"] is True
    assert alice_entry["activity_count"] == 5
    assert alice_entry["channel_id"] is None
    assert alice_entry["channel_title"] is None


def test_youtube_creator_output_preserves_channel_id() -> None:
    """Verifies YouTube creator ranked entry preserves authoritative channel_id and channel_title."""
    graph = nx.DiGraph()
    creator_id = "youtube:channel:UC_ABC123"
    graph.add_node(
        creator_id,
        node_id=creator_id,
        platform="youtube",
        identity_type="creator",
        username="Data Science Channel",
        channel_id="UC_ABC123",
        channel_title="Data Science Channel",
        is_author=True,
        activity_count=3,
    )
    graph.add_edge("youtube:channel:UC_XYZ", creator_id, weight=1)
    graph.nodes["youtube:channel:UC_XYZ"].update({
        "node_id": "youtube:channel:UC_XYZ",
        "platform": "youtube",
        "identity_type": "creator",
        "username": "Other Creator",
        "channel_id": "UC_XYZ",
        "channel_title": "Other Creator",
        "is_author": True,
        "activity_count": 1,
    })

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    creator_entry = next(r for r in ranked if r["node_id"] == creator_id)
    assert creator_entry["channel_id"] == "UC_ABC123"
    assert creator_entry["channel_title"] == "Data Science Channel"
    assert creator_entry["username"] == "Data Science Channel"
    assert creator_entry["node_id"] == creator_id
    assert creator_entry["identity_type"] == "creator"
    assert creator_entry["is_author"] is True


def test_youtube_mention_only_nodes_retain_null_channel_metadata() -> None:
    """Verifies YouTube mention-only nodes retain null channel metadata, is_author=False, activity_count=0."""
    graph = nx.DiGraph()
    mention_id = "youtube:mention:techguru"
    creator_id = "youtube:channel:UC_456"

    graph.add_node(
        creator_id,
        node_id=creator_id,
        platform="youtube",
        identity_type="creator",
        username="Source Channel",
        channel_id="UC_456",
        channel_title="Source Channel",
        is_author=True,
        activity_count=1,
    )
    graph.add_node(
        mention_id,
        node_id=mention_id,
        platform="youtube",
        identity_type="mention",
        username="techguru",
        channel_id=None,
        channel_title=None,
        is_author=False,
        activity_count=0,
    )
    graph.add_edge(creator_id, mention_id, weight=2)

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    mention_entry = next(r for r in ranked if r["node_id"] == mention_id)
    assert mention_entry["identity_type"] == "mention"
    assert mention_entry["channel_id"] is None
    assert mention_entry["channel_title"] is None
    assert mention_entry["is_author"] is False
    assert mention_entry["activity_count"] == 0
    assert mention_entry["platform"] == "youtube"
    assert mention_entry["username"] == "techguru"


def test_cross_platform_same_username_distinct_node_ids() -> None:
    """Verifies that identical usernames on Reddit and YouTube remain separate ranked entities."""
    graph = nx.DiGraph()
    reddit_id = "reddit:user:techguru"
    yt_creator_id = "youtube:channel:UC_TG"

    graph.add_node(
        reddit_id,
        node_id=reddit_id,
        platform="reddit",
        identity_type="account",
        username="TechGuru",
        is_author=True,
        activity_count=1,
        channel_id=None,
        channel_title=None,
    )
    graph.add_node(
        yt_creator_id,
        node_id=yt_creator_id,
        platform="youtube",
        identity_type="creator",
        username="TechGuru",
        channel_id="UC_TG",
        channel_title="TechGuru",
        is_author=True,
        activity_count=1,
    )
    graph.add_edge("reddit:user:other", reddit_id, weight=1)
    graph.add_edge("youtube:channel:UC_OTHER", yt_creator_id, weight=1)

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    entries = [r for r in ranked if r["username"] == "TechGuru"]
    assert len(entries) == 2

    node_ids = {e["node_id"] for e in entries}
    assert node_ids == {reddit_id, yt_creator_id}

    platforms = {e["platform"] for e in entries}
    assert platforms == {"reddit", "youtube"}


def test_equal_score_ranking_tie_breaker_uses_canonical_node_id() -> None:
    """Verifies that nodes with identical scores and pagerank are ordered by canonical node_id ascending."""
    graph = nx.DiGraph()
    node_b = "reddit:user:beta"
    node_a = "reddit:user:alpha"

    graph.add_node(node_b, node_id=node_b, platform="reddit", identity_type="account", username="Beta")
    graph.add_node(node_a, node_id=node_a, platform="reddit", identity_type="account", username="Alpha")
    graph.add_edge(node_a, node_b, weight=1)
    graph.add_edge(node_b, node_a, weight=1)

    ranker = InfluencerRanker()
    ranked = ranker.rank_influencers(graph)

    assert len(ranked) == 2
    # Because scores and pageranks are identical, node_a must precede node_b ('reddit:user:alpha' < 'reddit:user:beta')
    assert ranked[0]["influencer_score"] == ranked[1]["influencer_score"]
    assert ranked[0]["pagerank"] == ranked[1]["pagerank"]
    assert ranked[0]["node_id"] == node_a
    assert ranked[1]["node_id"] == node_b


def test_graph_insertion_order_invariance_ranking() -> None:
    """Verifies that inserting nodes and edges in different order produces identical ranking order."""
    g1 = nx.DiGraph()
    g1.add_node("reddit:user:a", node_id="reddit:user:a", username="A", platform="reddit", identity_type="account")
    g1.add_node("reddit:user:b", node_id="reddit:user:b", username="B", platform="reddit", identity_type="account")
    g1.add_node("reddit:user:c", node_id="reddit:user:c", username="C", platform="reddit", identity_type="account")
    g1.add_edge("reddit:user:a", "reddit:user:b", weight=2)
    g1.add_edge("reddit:user:c", "reddit:user:b", weight=1)

    g2 = nx.DiGraph()
    g2.add_node("reddit:user:c", node_id="reddit:user:c", username="C", platform="reddit", identity_type="account")
    g2.add_node("reddit:user:b", node_id="reddit:user:b", username="B", platform="reddit", identity_type="account")
    g2.add_node("reddit:user:a", node_id="reddit:user:a", username="A", platform="reddit", identity_type="account")
    g2.add_edge("reddit:user:c", "reddit:user:b", weight=1)
    g2.add_edge("reddit:user:a", "reddit:user:b", weight=2)

    ranker = InfluencerRanker()
    ranked_1 = ranker.rank_influencers(g1)
    ranked_2 = ranker.rank_influencers(g2)

    order_1 = [r["node_id"] for r in ranked_1]
    order_2 = [r["node_id"] for r in ranked_2]
    assert order_1 == order_2


def test_canonical_community_labeling_stable_across_insertion_orders() -> None:
    """Verifies that community IDs are assigned canonically based on minimum node_id and are stable across insertion orders."""
    g1 = nx.DiGraph()
    g1.add_edge("X1", "X2", weight=1)
    g1.add_edge("X2", "X1", weight=1)
    g1.add_edge("Y1", "Y2", weight=1)
    g1.add_edge("Y2", "Y1", weight=1)

    g2 = nx.DiGraph()
    g2.add_edge("Y1", "Y2", weight=1)
    g2.add_edge("Y2", "Y1", weight=1)
    g2.add_edge("X1", "X2", weight=1)
    g2.add_edge("X2", "X1", weight=1)

    ranker = InfluencerRanker()
    c1 = ranker.detect_communities(g1)
    c2 = ranker.detect_communities(g2)

    assert c1 == c2
    # Because 'X1' < 'Y1', the X clique must be community 0, and Y clique community 1
    assert c1["X1"] == 0
    assert c1["X2"] == 0
    assert c1["Y1"] == 1
    assert c1["Y2"] == 1

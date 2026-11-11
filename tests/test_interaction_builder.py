"""Unit tests for InteractionGraphBuilder (Company Step 5).

Verifies user validation, mention extraction, graph construction, edge weight aggregation,
anonymous node filtering, platform-namespaced identities, input-order invariance, and
creator/mention isolation.
"""

from typing import Any
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
    """Tests building directed interaction graph with platform-namespaced identities."""
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
            "channel_id": "UC_david",
            "channel_title": "David_AI",
            "title_cleaned": "Video on Big Data",
            "description_cleaned": "Shoutout to @Alice for inspiring this project.",
        }
    ]

    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records, youtube_records)

    assert isinstance(graph, nx.DiGraph)
    alice_id = "reddit:user:alice"
    bob_id = "reddit:user:bob"
    charlie_id = "reddit:user:charlie"
    david_id = "youtube:channel:UC_david"
    yt_alice_mention_id = "youtube:mention:alice"

    assert graph.has_node(alice_id)
    assert graph.has_node(bob_id)
    assert graph.has_node(charlie_id)
    assert graph.has_node(david_id)
    assert graph.has_node(yt_alice_mention_id)

    # Alice mentioned Bob in two separate posts -> edge weight must be 2
    assert graph.has_edge(alice_id, bob_id)
    assert graph[alice_id][bob_id]["weight"] == 2
    assert "MENTIONED" in graph[alice_id][bob_id]["interaction_types"]
    assert "reddit" in graph[alice_id][bob_id]["platforms"]

    # Alice mentioned Charlie once -> weight 1
    assert graph.has_edge(alice_id, charlie_id)
    assert graph[alice_id][charlie_id]["weight"] == 1

    # David_AI mentioned @Alice in YouTube -> David_AI -> youtube:mention:alice
    # (Notice: does NOT point to reddit:user:alice, isolating platforms!)
    assert graph.has_edge(david_id, yt_alice_mention_id)
    assert graph[david_id][yt_alice_mention_id]["weight"] == 1
    assert not graph.has_edge(david_id, alice_id)


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

    assert not graph.has_node("reddit:user:[deleted]")
    assert not graph.has_node("reddit:user:deleted")
    assert not graph.has_node("[deleted]")


def test_get_nodes_and_edges() -> None:
    """Tests export methods get_nodes and get_edges return sorted structures."""
    builder = InteractionGraphBuilder()
    builder.user_graph.add_edge("UserB", "UserA", weight=3, platforms=["reddit"], interaction_types=["MENTIONED"])
    builder.user_graph.nodes["UserA"]["node_id"] = "UserA"
    builder.user_graph.nodes["UserA"]["platform"] = "reddit"
    builder.user_graph.nodes["UserB"]["node_id"] = "UserB"
    builder.user_graph.nodes["UserB"]["platform"] = "reddit"

    nodes = builder.get_nodes()
    edges = builder.get_edges()

    assert len(nodes) == 2
    assert len(edges) == 1
    # Sorted by node_id: UserA before UserB
    assert nodes[0]["node_id"] == "UserA"
    assert nodes[1]["node_id"] == "UserB"
    assert edges[0]["source"] == "UserB"
    assert edges[0]["target"] == "UserA"
    assert edges[0]["weight"] == 3


# ==============================================================================
# 4. Reddit Node Lifecycle & Input-Order Invariance Tests
# ==============================================================================

def test_reddit_node_lifecycle_input_order_invariance() -> None:
    """Verifies that Order A (mention first) and Order B (author first) produce identical nodes and edges."""
    post_mention = {
        "post_id": "p_alice",
        "author": "Alice",
        "title_cleaned": "Shoutout to u/Bob",
        "text_cleaned": "Great work u/Bob on graphs.",
    }
    post_author = {
        "post_id": "p_bob",
        "author": "Bob",
        "title_cleaned": "Bob's original post",
        "text_cleaned": "Here are my thoughts on graph modeling.",
    }

    # Order A: Alice mentions Bob, then Bob authors a post
    builder_a = InteractionGraphBuilder()
    graph_a = builder_a.build_interaction_graph(reddit_records=[post_mention, post_author])

    # Order B: Bob authors a post, then Alice mentions Bob
    builder_b = InteractionGraphBuilder()
    graph_b = builder_b.build_interaction_graph(reddit_records=[post_author, post_mention])

    bob_id = "reddit:user:bob"
    alice_id = "reddit:user:alice"

    for g in (graph_a, graph_b):
        assert g.has_node(bob_id)
        node_bob = g.nodes[bob_id]
        assert node_bob["node_id"] == bob_id
        assert node_bob["identity_type"] == "account"
        assert node_bob["is_author"] is True
        assert node_bob["activity_count"] == 1
        assert node_bob["platform"] == "reddit"
        assert node_bob["channel_id"] is None
        assert node_bob["channel_title"] is None

        # Verify edge Alice -> Bob
        assert g.has_edge(alice_id, bob_id)
        assert g[alice_id][bob_id]["weight"] == 1
        assert g[alice_id][bob_id]["interaction_types"] == ["MENTIONED"]

    # Compare canonicalized node attributes
    nodes_a = {n["node_id"]: n for n in builder_a.get_nodes()}
    nodes_b = {n["node_id"]: n for n in builder_b.get_nodes()}
    assert nodes_a == nodes_b

    # Compare canonicalized edges
    assert builder_a.get_edges() == builder_b.get_edges()


def test_reddit_display_username_casing_order_invariance() -> None:
    """Verifies that Order A (Alice mentions u/Bob, Bob authors as BOB) and Order B (Bob authors as BOB, Alice mentions u/Bob)
    produce completely identical canonical Reddit node attributes and edges regardless of input record ordering.
    """
    post_mention = {
        "post_id": "p_alice",
        "author": "Alice",
        "title_cleaned": "Discussing graph models with u/Bob",
        "text_cleaned": "Shoutout to u/Bob for great ideas.",
    }
    post_author = {
        "post_id": "p_bob",
        "author": "BOB",
        "title_cleaned": "Bob's authored post",
        "text_cleaned": "Here are graph findings.",
    }

    # Order A: Alice mentions u/Bob, then Bob authors as BOB
    builder_a = InteractionGraphBuilder()
    graph_a = builder_a.build_interaction_graph(reddit_records=[post_mention, post_author])

    # Order B: Bob authors as BOB, then Alice mentions u/Bob
    builder_b = InteractionGraphBuilder()
    graph_b = builder_b.build_interaction_graph(reddit_records=[post_author, post_mention])

    bob_id = "reddit:user:bob"
    alice_id = "reddit:user:alice"

    # Both graphs must contain the nodes
    for g in (graph_a, graph_b):
        assert g.has_node(bob_id)
        assert g.has_node(alice_id)

        # Assert canonicalized Reddit node attributes on Bob
        node_bob = g.nodes[bob_id]
        assert node_bob["node_id"] == "reddit:user:bob"
        assert node_bob["username"] == "BOB"
        assert node_bob["identity_type"] == "account"
        assert node_bob["is_author"] is True
        assert node_bob["activity_count"] == 1
        assert node_bob["platform"] == "reddit"
        assert node_bob["channel_id"] is None
        assert node_bob["channel_title"] is None

        # Assert canonicalized Reddit node attributes on Alice
        node_alice = g.nodes[alice_id]
        assert node_alice["node_id"] == "reddit:user:alice"
        assert node_alice["username"] == "Alice"
        assert node_alice["identity_type"] == "account"
        assert node_alice["is_author"] is True
        assert node_alice["activity_count"] == 1
        assert node_alice["platform"] == "reddit"

    # Full structural node equality
    nodes_a = {n["node_id"]: n for n in builder_a.get_nodes()}
    nodes_b = {n["node_id"]: n for n in builder_b.get_nodes()}
    assert nodes_a == nodes_b

    # Full structural edge equality
    assert builder_a.get_edges() == builder_b.get_edges()


def test_reddit_display_username_multiple_casing_variations_order_invariance() -> None:
    """Verifies that arbitrary permutations of records with differing casing ('bob', 'Bob', 'BOB')
    all produce the exact same canonical display name and attributes.
    """
    records_1 = [
        {"post_id": "p1", "author": "Alice", "title_cleaned": "", "text_cleaned": "Mention u/bob"},
        {"post_id": "p2", "author": "Bob", "title_cleaned": "Title", "text_cleaned": "Mention u/Charlie"},
        {"post_id": "p3", "author": "BOB", "title_cleaned": "Title", "text_cleaned": "Another post"},
    ]
    records_2 = list(reversed(records_1))

    builder_1 = InteractionGraphBuilder()
    builder_1.build_interaction_graph(reddit_records=records_1)

    builder_2 = InteractionGraphBuilder()
    builder_2.build_interaction_graph(reddit_records=records_2)

    assert builder_1.get_nodes() == builder_2.get_nodes()
    assert builder_1.get_edges() == builder_2.get_edges()


def test_reddit_mention_only_remains_non_author() -> None:
    """Verifies that a mentioned Reddit account that authors no posts has is_author=False, activity_count=0."""
    post = {
        "post_id": "p1",
        "author": "Alice",
        "title_cleaned": "Hello u/Charlie",
        "text_cleaned": "",
    }
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records=[post])

    charlie_id = "reddit:user:charlie"
    assert graph.has_node(charlie_id)
    charlie = graph.nodes[charlie_id]
    assert charlie["identity_type"] == "account"
    assert charlie["is_author"] is False
    assert charlie["activity_count"] == 0
    assert charlie["platform"] == "reddit"


def test_reddit_case_variations_resolve_to_same_identity() -> None:
    """Verifies that case variants resolve to the same canonical node and aggregate authored activity."""
    posts = [
        {"post_id": "p1", "author": "Alice", "title_cleaned": "First", "text_cleaned": ""},
        {"post_id": "p2", "author": "alice", "title_cleaned": "Second", "text_cleaned": ""},
        {"post_id": "p3", "author": "ALICE", "title_cleaned": "Third", "text_cleaned": ""},
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records=posts)

    assert graph.number_of_nodes() == 1
    alice_id = "reddit:user:alice"
    assert graph.has_node(alice_id)
    assert graph.nodes[alice_id]["activity_count"] == 3
    assert graph.nodes[alice_id]["is_author"] is True


def test_reddit_self_mention_prevention() -> None:
    """Verifies that an author mentioning their own username does not create self-loop edges."""
    post = {
        "post_id": "p1",
        "author": "Alice",
        "title_cleaned": "Me mentioning u/Alice and u/alice",
        "text_cleaned": "Also cc @ALICE in text.",
    }
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records=[post])

    alice_id = "reddit:user:alice"
    assert graph.has_node(alice_id)
    assert not graph.has_edge(alice_id, alice_id)
    assert graph.number_of_edges() == 0


# ==============================================================================
# 5. YouTube Creator and Mention Isolation Tests
# ==============================================================================

def test_youtube_creator_and_mention_nodes_never_collide() -> None:
    """Verifies that creator youtube:channel:UC_123 and mention youtube:mention:techguru never collide."""
    videos = [
        {
            "video_id": "v1",
            "channel_id": "UC_123",
            "channel_title": "TechGuru",
            "title_cleaned": "My Tech Video",
            "description_cleaned": "Check out @TechGuru on social media.",
        }
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(youtube_records=videos)

    creator_id = "youtube:channel:UC_123"
    mention_id = "youtube:mention:techguru"

    assert graph.has_node(creator_id)
    assert graph.has_node(mention_id)
    assert creator_id != mention_id
    assert graph.number_of_nodes() == 2

    # Creator node checks
    c_node = graph.nodes[creator_id]
    assert c_node["identity_type"] == "creator"
    assert c_node["is_author"] is True
    assert c_node["channel_id"] == "UC_123"
    assert c_node["channel_title"] == "TechGuru"
    assert c_node["activity_count"] == 1

    # Mention node checks
    m_node = graph.nodes[mention_id]
    assert m_node["identity_type"] == "mention"
    assert m_node["is_author"] is False
    assert m_node["channel_id"] is None
    assert m_node["channel_title"] is None
    assert m_node["activity_count"] == 0

    # Directed edge creator -> mention
    assert graph.has_edge(creator_id, mention_id)
    assert graph[creator_id][mention_id]["weight"] == 1


def test_youtube_duplicate_channel_titles_different_channel_ids_isolated() -> None:
    """Verifies that two channels sharing the same display title remain separate nodes."""
    videos = [
        {
            "video_id": "v1",
            "channel_id": "UC_CHANNEL_A",
            "channel_title": "Data Engineering",
            "title_cleaned": "Channel A Video",
            "description_cleaned": "",
        },
        {
            "video_id": "v2",
            "channel_id": "UC_CHANNEL_B",
            "channel_title": "Data Engineering",
            "title_cleaned": "Channel B Video",
            "description_cleaned": "",
        },
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(youtube_records=videos)

    id_a = "youtube:channel:UC_CHANNEL_A"
    id_b = "youtube:channel:UC_CHANNEL_B"

    assert graph.has_node(id_a)
    assert graph.has_node(id_b)
    assert graph.number_of_nodes() == 2

    assert graph.nodes[id_a]["channel_title"] == "Data Engineering"
    assert graph.nodes[id_b]["channel_title"] == "Data Engineering"
    assert graph.nodes[id_a]["activity_count"] == 1
    assert graph.nodes[id_b]["activity_count"] == 1


def test_repeated_youtube_records_reuse_creator_node() -> None:
    """Verifies repeated videos from the same channel increment activity_count on the same creator node."""
    videos = [
        {"video_id": "v1", "channel_id": "UC_100", "channel_title": "Creator", "title_cleaned": "Part 1", "description_cleaned": ""},
        {"video_id": "v2", "channel_id": "UC_100", "channel_title": "Creator", "title_cleaned": "Part 2", "description_cleaned": ""},
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(youtube_records=videos)

    creator_id = "youtube:channel:UC_100"
    assert graph.has_node(creator_id)
    assert graph.number_of_nodes() == 1
    assert graph.nodes[creator_id]["activity_count"] == 2


def test_repeated_youtube_mentions_reuse_mention_node() -> None:
    """Verifies repeated mentions of the same handle aggregate edge weight with activity_count remaining 0."""
    videos = [
        {"video_id": "v1", "channel_id": "UC_1", "channel_title": "C1", "title_cleaned": "Video 1", "description_cleaned": "Mention @dataeng"},
        {"video_id": "v2", "channel_id": "UC_2", "channel_title": "C2", "title_cleaned": "Video 2", "description_cleaned": "Also mention @dataeng"},
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(youtube_records=videos)

    mention_id = "youtube:mention:dataeng"
    assert graph.has_node(mention_id)
    assert graph.nodes[mention_id]["activity_count"] == 0
    assert graph.nodes[mention_id]["is_author"] is False

    # Two incoming edges to the same mention node
    assert graph.has_edge("youtube:channel:UC_1", mention_id)
    assert graph.has_edge("youtube:channel:UC_2", mention_id)
    assert graph.in_degree(mention_id) == 2


def test_youtube_and_reddit_same_username_isolated() -> None:
    """Verifies that identical usernames across Reddit and YouTube create distinct nodes."""
    reddit_records = [
        {"post_id": "r1", "author": "TechGuru", "title_cleaned": "Reddit Post", "text_cleaned": ""}
    ]
    youtube_records = [
        {"video_id": "y1", "channel_id": "UC_TG", "channel_title": "TechGuru", "title_cleaned": "YT Video", "description_cleaned": ""}
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(reddit_records, youtube_records)

    r_id = "reddit:user:techguru"
    y_id = "youtube:channel:UC_TG"

    assert graph.has_node(r_id)
    assert graph.has_node(y_id)
    assert graph.number_of_nodes() == 2
    assert graph.nodes[r_id]["platform"] == "reddit"
    assert graph.nodes[y_id]["platform"] == "youtube"


def test_youtube_title_only_record_cannot_create_creator_node() -> None:
    """Verifies that a title-only YouTube record cannot create a creator node or fabricate an ID."""
    videos = [
        {"video_id": "v1", "channel_title": "TechGuru", "title_cleaned": "V1", "description_cleaned": ""},
        {"video_id": "v2", "channel_id": None, "channel_title": "DataScience", "title_cleaned": "V2", "description_cleaned": ""},
        {"video_id": "v3", "channel_id": "", "channel_title": "DevTips", "title_cleaned": "V3", "description_cleaned": ""},
        {"video_id": "v4", "channel_id": "   ", "channel_title": "WhitespaceID", "title_cleaned": "V4", "description_cleaned": ""},
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(youtube_records=videos)

    # Must produce zero creator nodes
    assert graph.number_of_nodes() == 0
    # Must never fabricate node IDs from titles
    assert not graph.has_node("youtube:channel:TechGuru")
    assert not graph.has_node("youtube:channel:DataScience")
    assert not graph.has_node("youtube:channel:DevTips")
    assert not graph.has_node("youtube:channel:WhitespaceID")


def test_youtube_valid_channel_id_no_channel_title_creates_creator_node() -> None:
    """Verifies that a record with authoritative channel_id and no channel_title creates correct node."""
    videos = [
        {"video_id": "v101", "channel_id": "UC_AUTH_123", "title_cleaned": "V1", "description_cleaned": ""},
        {"video_id": "v102", "channel_id": "UC_AUTH_456", "channel_title": None, "title_cleaned": "V2", "description_cleaned": ""},
    ]
    builder = InteractionGraphBuilder()
    graph = builder.build_interaction_graph(youtube_records=videos)

    expected_id1 = "youtube:channel:UC_AUTH_123"
    expected_id2 = "youtube:channel:UC_AUTH_456"

    assert graph.has_node(expected_id1)
    assert graph.has_node(expected_id2)

    node1 = graph.nodes[expected_id1]
    assert node1["node_id"] == expected_id1
    assert node1["channel_id"] == "UC_AUTH_123"
    assert node1["channel_title"] is None
    assert node1["username"] == "UC_AUTH_123"  # fallback to channel_id when title is None
    assert node1["identity_type"] == "creator"
    assert node1["is_author"] is True
    assert node1["activity_count"] == 1

    node2 = graph.nodes[expected_id2]
    assert node2["node_id"] == expected_id2
    assert node2["channel_id"] == "UC_AUTH_456"
    assert node2["channel_title"] is None
    assert node2["username"] == "UC_AUTH_456"


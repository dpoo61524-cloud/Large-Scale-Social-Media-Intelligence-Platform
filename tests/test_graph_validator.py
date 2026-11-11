"""Unit tests for GraphArtifactValidator (Stage 7 - Graph Analysis).

Verifies strict Schema-v2 validation rules:
- Strict semantic versioning and major version classification
- Platform-specific identity rules (Reddit, YouTube creator, YouTube mention)
- Edge referential integrity, attributes, and deterministic sorting
- Top influencer exact identity metadata consistency with complete nodes
- Finite metric bounds and non-boolean integers
- Non-mutation guarantee and safe handling of unexpected malformed input structures
"""

import copy
import math
from typing import Any
import pytest

from src.graph.graph_validator import GraphArtifactValidator


# ==============================================================================
# Realistic Schema-v2 Fixture Builder Helper
# ==============================================================================

def build_valid_schema_v2_payload() -> dict[str, Any]:
    """Constructs a realistic, fully compliant Schema-v2 graph analysis payload."""
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
            "0": 2,
            "1": 2,
        },
    }


def build_valid_empty_payload() -> dict[str, Any]:
    """Constructs a realistic, fully compliant empty Schema-v2 graph analysis payload."""
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


def build_valid_partial_payload() -> dict[str, Any]:
    """Constructs a realistic, fully compliant partial Schema-v2 graph analysis payload."""
    payload = build_valid_schema_v2_payload()
    payload["status"] = "partial"
    payload["provenance"]["youtube_status"] = "missing"
    payload["provenance"]["youtube_records_count"] = 0
    # Keep only Reddit nodes and edges
    payload["nodes"] = [n for n in payload["nodes"] if n["platform"] == "reddit"]
    payload["edges"] = [e for e in payload["edges"] if e["platforms"] == ["reddit"]]
    payload["top_influencers"] = [i for i in payload["top_influencers"] if i["platform"] == "reddit"]
    payload["graph_metrics"]["total_users"] = 2
    payload["graph_metrics"]["total_interactions"] = 1
    payload["graph_metrics"]["communities_count"] = 1
    payload["graph_metrics"]["density"] = 0.5
    payload["community_distributions"] = {"0": 2}
    return payload


# ==============================================================================
# 1. Base Valid Payload
# ==============================================================================

def test_valid_schema_v2_payload_passes() -> None:
    """Test 1: Valid schema-v2 payload passes with zero errors."""
    payload = build_valid_schema_v2_payload()
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True
    assert errors == []


# ==============================================================================
# 2. Schema Version Classification Tests
# ==============================================================================

def test_missing_schema_version() -> None:
    """Test 2: Missing or None schema_version fails with missing_schema_version."""
    payload = build_valid_schema_v2_payload()
    del payload["schema_version"]
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("missing_schema_version" in e for e in errors)

    payload["schema_version"] = None
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid2 is False
    assert any("missing_schema_version" in e for e in errors2)


def test_malformed_version_strings() -> None:
    """Test 3: Malformed version strings fail with malformed_schema_version."""
    bad_versions = ["v2.0", "2.0.1", "2", "", "abc", "2.0-beta", "02.0", "2.00"]
    for bad in bad_versions:
        payload = build_valid_schema_v2_payload()
        payload["schema_version"] = bad
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False, f"Expected failure for version '{bad}'"
        assert any("malformed_schema_version" in e for e in errors)


def test_boolean_integer_float_versions() -> None:
    """Test 4: Non-string versions (boolean, integer, float) fail with malformed_schema_version."""
    for bad in [True, False, 2, 2.0, [2, 0]]:
        payload = build_valid_schema_v2_payload()
        payload["schema_version"] = bad
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("malformed_schema_version" in e for e in errors)


def test_legacy_major_version() -> None:
    """Test 5: Major version < 2 fails with legacy_graph_schema_lacks_channel_id."""
    for legacy in ["1.0", "1.9", "0.9", "0.1"]:
        payload = build_valid_schema_v2_payload()
        payload["schema_version"] = legacy
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("legacy_graph_schema_lacks_channel_id" in e for e in errors)


def test_unsupported_future_major_version() -> None:
    """Test 6: Major version >= 3 fails with unsupported_graph_schema_version."""
    for future in ["3.0", "3.1", "4.0", "10.0"]:
        payload = build_valid_schema_v2_payload()
        payload["schema_version"] = future
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("unsupported_graph_schema_version" in e for e in errors)


def test_supported_newer_minor_version() -> None:
    """Test 7: Supported minor versions on major version 2 (2.1, 2.12) pass."""
    for minor in ["2.0", "2.1", "2.12"]:
        payload = build_valid_schema_v2_payload()
        payload["schema_version"] = minor
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is True
        assert errors == []


# ==============================================================================
# 3. Top-Level Schema & Metadata Tests
# ==============================================================================

def test_missing_required_metadata() -> None:
    """Test 8: Missing top-level required metadata keys fail validation."""
    required_keys = [
        "status",
        "generated_at",
        "provenance",
        "nodes",
        "edges",
        "top_influencers",
        "graph_metrics",
        "community_distributions",
    ]
    for key in required_keys:
        payload = build_valid_schema_v2_payload()
        del payload[key]
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any(key in e for e in errors)


def test_invalid_status() -> None:
    """Test 9: Invalid status values fail validation."""
    for bad_status in ["unknown", "failed", "completed", 123, True, None, ""]:
        payload = build_valid_schema_v2_payload()
        payload["status"] = bad_status
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("'status'" in e for e in errors)

    # Valid statuses with appropriate payloads must pass
    valid_success, errs_success = GraphArtifactValidator.validate_graph_artifact(
        build_valid_schema_v2_payload()
    )
    assert valid_success is True
    assert errs_success == []

    valid_partial, errs_partial = GraphArtifactValidator.validate_graph_artifact(
        build_valid_partial_payload()
    )
    assert valid_partial is True
    assert errs_partial == []

    valid_empty, errs_empty = GraphArtifactValidator.validate_graph_artifact(
        build_valid_empty_payload()
    )
    assert valid_empty is True
    assert errs_empty == []


# ==============================================================================
# 4. Node Validation & Identity Rules Tests
# ==============================================================================

def test_valid_nodes_all_three_types() -> None:
    """Test 10: All three node types (Reddit account, YouTube creator, YouTube mention) pass."""
    payload = build_valid_schema_v2_payload()
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True


def test_missing_username_on_namespaced_node() -> None:
    """Test 11: Missing, empty, or raw namespaced username is rejected."""
    for bad_user in [None, "", "   ", "reddit:user:alice", "youtube:channel:UC_123"]:
        payload = build_valid_schema_v2_payload()
        payload["nodes"][0]["username"] = bad_user
        payload["top_influencers"][0]["username"] = bad_user
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("'username'" in e for e in errors)


def test_incorrect_platform_identity_combinations() -> None:
    """Test 12: Inconsistent platform and identity_type combinations are rejected."""
    # Reddit with creator
    payload1 = build_valid_schema_v2_payload()
    payload1["nodes"][0]["identity_type"] = "creator"
    payload1["top_influencers"][0]["identity_type"] = "creator"
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(payload1)
    assert valid1 is False
    assert any("reddit platform requires identity_type='account'" in e for e in errors1)

    # YouTube with account
    payload2 = build_valid_schema_v2_payload()
    payload2["nodes"][2]["identity_type"] = "account"
    payload2["top_influencers"][1]["identity_type"] = "account"
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(payload2)
    assert valid2 is False
    assert any("invalid youtube identity_type" in e for e in errors2)


def test_invalid_node_id_patterns() -> None:
    """Test 13: Node IDs violating platform naming patterns are rejected."""
    bad_node_ids = [
        (0, "reddit:user:"),  # Missing username
        (0, "reddit:user:Alice_Data"),  # Uppercase not allowed in canonical normalized key
        (2, "youtube:channel:"),  # Missing channel ID
        (2, "youtube:channel:UC 123"),  # Space not allowed
        (3, "youtube:mention:"),  # Missing handle
        (3, "bad_prefix:user:alice"),  # Invalid prefix
    ]
    for idx, bad_id in bad_node_ids:
        payload = build_valid_schema_v2_payload()
        payload["nodes"][idx]["node_id"] = bad_id
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("invalid" in e or "pattern" in e or "match" in e for e in errors)


def test_duplicate_node_ids() -> None:
    """Test 14: Duplicate node_id values in nodes array are rejected."""
    payload = build_valid_schema_v2_payload()
    dup_node = copy.deepcopy(payload["nodes"][0])
    payload["nodes"].append(dup_node)
    payload["graph_metrics"]["total_users"] = 5
    payload["community_distributions"]["0"] = 3
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("duplicate node_id" in e for e in errors)


def test_reddit_author_activity_count_mismatch() -> None:
    """Test 15: Reddit lifecycle consistency: is_author=True requires activity_count >= 1, is_author=False requires 0."""
    # Case 1: is_author=True but activity_count=0
    p1 = build_valid_schema_v2_payload()
    p1["nodes"][0]["is_author"] = True
    p1["nodes"][0]["activity_count"] = 0
    p1["top_influencers"][0]["is_author"] = True
    p1["top_influencers"][0]["activity_count"] = 0
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("is_author=True requires activity_count >= 1" in e for e in errors1)

    # Case 2: is_author=False but activity_count=1
    p2 = build_valid_schema_v2_payload()
    p2["nodes"][1]["is_author"] = False
    p2["nodes"][1]["activity_count"] = 1
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("is_author=False requires activity_count == 0" in e for e in errors2)


def test_youtube_creator_channel_id_mismatch() -> None:
    """Test 16: YouTube creator channel_id must be non-empty and exactly match node_id suffix."""
    # Mismatch with node_id
    payload1 = build_valid_schema_v2_payload()
    payload1["nodes"][2]["channel_id"] = "UC_DIFFERENT"
    payload1["top_influencers"][1]["channel_id"] = "UC_DIFFERENT"
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(payload1)
    assert valid1 is False
    assert any("does not match node_id suffix" in e for e in errors1)

    # Empty channel_id
    payload2 = build_valid_schema_v2_payload()
    payload2["nodes"][2]["channel_id"] = ""
    payload2["top_influencers"][1]["channel_id"] = ""
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(payload2)
    assert valid2 is False
    assert any("non-empty string 'channel_id'" in e for e in errors2)


def test_invalid_youtube_mention_metadata() -> None:
    """Test 17: YouTube mention-only nodes with non-null channel fields or author flags fail."""
    # channel_id not None
    p1 = build_valid_schema_v2_payload()
    p1["nodes"][3]["channel_id"] = "UC_FAKE"
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("channel_id=None" in e for e in errors1)

    # is_author True
    p2 = build_valid_schema_v2_payload()
    p2["nodes"][3]["is_author"] = True
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("is_author=False" in e for e in errors2)

    # activity_count != 0
    p3 = build_valid_schema_v2_payload()
    p3["nodes"][3]["activity_count"] = 1
    valid3, errors3 = GraphArtifactValidator.validate_graph_artifact(p3)
    assert valid3 is False
    assert any("activity_count=0" in e for e in errors3)


# ==============================================================================
# 5. Edge Validation Tests
# ==============================================================================

def test_missing_edge_endpoint() -> None:
    """Test 18: Edges referencing non-existent nodes fail validation."""
    payload = build_valid_schema_v2_payload()
    payload["edges"][0]["target"] = "reddit:user:ghost"
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("does not exist in nodes" in e for e in errors)


def test_invalid_edge_weight() -> None:
    """Test 19: Edge weights <= 0, booleans, or floats fail validation."""
    for bad_w in [0, -1, True, False, 1.5, "1", None]:
        payload = build_valid_schema_v2_payload()
        payload["edges"][0]["weight"] = bad_w
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("'weight'" in e for e in errors)


def test_invalid_interaction_types_or_platforms_lists() -> None:
    """Test 20: Empty, unsorted, duplicate, or invalid items in edge lists fail validation."""
    # Empty platforms list
    p1 = build_valid_schema_v2_payload()
    p1["edges"][0]["platforms"] = []
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False

    # Unsorted platforms list
    p2 = build_valid_schema_v2_payload()
    p2["edges"][0]["platforms"] = ["youtube", "reddit"]
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("platforms must be sorted without duplicates" in e for e in errors2)

    # Duplicate platforms
    p3 = build_valid_schema_v2_payload()
    p3["edges"][0]["platforms"] = ["reddit", "reddit"]
    valid3, errors3 = GraphArtifactValidator.validate_graph_artifact(p3)
    assert valid3 is False


def test_duplicate_edge_pair() -> None:
    """Test 21: Duplicate directed edge pairs (source, target) fail validation."""
    payload = build_valid_schema_v2_payload()
    dup_edge = copy.deepcopy(payload["edges"][0])
    payload["edges"].append(dup_edge)
    payload["graph_metrics"]["total_interactions"] = 3
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("duplicate directed edge pair" in e for e in errors)


def test_unsorted_nodes_and_edges() -> None:
    """Test 22: Unsorted nodes or edges arrays fail validation."""
    # Unsorted nodes
    p1 = build_valid_schema_v2_payload()
    p1["nodes"].reverse()
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("unsorted node list" in e for e in errors1)

    # Unsorted edges
    p2 = build_valid_schema_v2_payload()
    p2["edges"].reverse()
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("unsorted edge list" in e for e in errors2)


# ==============================================================================
# 6. Top Influencer Validation & Referential Consistency Tests
# ==============================================================================

def test_top_influencer_referencing_nonexistent_node() -> None:
    """Test 23: Top influencer referencing node_id missing from nodes array fails."""
    payload = build_valid_schema_v2_payload()
    payload["top_influencers"][0]["node_id"] = "reddit:user:ghost_node"
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("node_id does not exist in nodes array" in e for e in errors)


def test_top_influencer_identity_metadata_mismatch() -> None:
    """Test 24: Top influencer identity metadata differing from nodes entry fails."""
    mismatches = [
        ("username", "DifferentName"),
        ("activity_count", 99),
        ("is_author", False),
        ("identity_type", "mention"),
    ]
    for field, bad_val in mismatches:
        payload = build_valid_schema_v2_payload()
        payload["top_influencers"][0][field] = bad_val
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any(f"metadata mismatch for '{field}'" in e for e in errors)


def test_nan_and_infinite_ranking_metrics() -> None:
    """Test 25: NaN or infinite values in ranking metrics fail validation."""
    for bad_metric in [float("nan"), float("inf"), float("-inf")]:
        payload = build_valid_schema_v2_payload()
        payload["top_influencers"][0]["influencer_score"] = bad_metric
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("'influencer_score'" in e for e in errors)


def test_out_of_range_normalized_scores() -> None:
    """Test 26: Normalized scores outside [0.0, 1.0] fail validation."""
    for bad_score in [-0.01, 1.05, 2.0]:
        payload = build_valid_schema_v2_payload()
        payload["top_influencers"][0]["influencer_score"] = bad_score
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("'influencer_score'" in e for e in errors)


def test_negative_raw_pagerank() -> None:
    """Test 27: Negative raw PageRank values fail validation."""
    payload = build_valid_schema_v2_payload()
    payload["top_influencers"][0]["pagerank"] = -0.05
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("'pagerank'" in e for e in errors)


def test_invalid_raw_degree_and_community_id_types() -> None:
    """Test 28: Booleans, floats, or negative values for degrees or community IDs fail."""
    for bad_val in [True, False, 1.5, -1, "0"]:
        payload = build_valid_schema_v2_payload()
        payload["top_influencers"][0]["community_id"] = bad_val
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any("'community_id'" in e for e in errors)


# ==============================================================================
# 7. Metrics, Distributions, Non-Mutation, & Malformed Structure Tests
# ==============================================================================

def test_invalid_graph_metric_types_and_values() -> None:
    """Test 29: Count mismatches, negative numbers, or invalid booleans in graph_metrics fail."""
    # total_users mismatch with nodes length
    p1 = build_valid_schema_v2_payload()
    p1["graph_metrics"]["total_users"] = 99
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("mismatch with nodes length" in e or "count" in e for e in errors1)

    # is_directed False
    p2 = build_valid_schema_v2_payload()
    p2["graph_metrics"]["is_directed"] = False
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("is_directed" in e for e in errors2)


def test_invalid_community_distribution_counts() -> None:
    """Test 30: Community count mismatches, negative counts, or invalid keys fail."""
    # Sum of counts does not match total nodes
    p1 = build_valid_schema_v2_payload()
    p1["community_distributions"]["0"] = 99
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("sum of community counts" in e for e in errors1)

    # Key is not string integer
    p2 = build_valid_schema_v2_payload()
    p2["community_distributions"]["comm_A"] = 2
    del p2["community_distributions"]["0"]
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("string integer" in e for e in errors2)


def test_input_payload_remains_unchanged_after_validation() -> None:
    """Test 31: Validation does not mutate or alter the supplied input payload dictionary."""
    payload = build_valid_schema_v2_payload()
    payload_copy = copy.deepcopy(payload)

    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True
    assert payload == payload_copy

    # Even on invalid payloads, no mutation occurs
    payload_bad = build_valid_schema_v2_payload()
    payload_bad["nodes"][0]["username"] = ""
    bad_copy = copy.deepcopy(payload_bad)

    valid_bad, errors_bad = GraphArtifactValidator.validate_graph_artifact(payload_bad)
    assert valid_bad is False
    assert payload_bad == bad_copy


def test_unexpected_malformed_input_structures_do_not_crash() -> None:
    """Test 32: Unexpected top-level structures (None, list, string, empty dict) fail gracefully without raising."""
    for malformed in [None, [], "", 12345, True, False]:
        valid, errors = GraphArtifactValidator.validate_graph_artifact(malformed)
        assert valid is False
        assert len(errors) > 0

    valid_empty, errors_empty = GraphArtifactValidator.validate_graph_artifact({})
    assert valid_empty is False
    assert len(errors_empty) > 0


# ==============================================================================
# 8. Provenance Validation Tests
# ==============================================================================

def test_empty_provenance_rejected() -> None:
    """Test 33: Empty provenance dictionary is rejected."""
    payload = build_valid_schema_v2_payload()
    payload["provenance"] = {}
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("'provenance' must not be empty" in e for e in errors)


def test_missing_provenance_keys() -> None:
    """Test 34: Missing any of the 6 required provenance keys fails validation."""
    req_keys = [
        "reddit_input_file",
        "reddit_status",
        "reddit_records_count",
        "youtube_input_file",
        "youtube_status",
        "youtube_records_count",
    ]
    for key in req_keys:
        payload = build_valid_schema_v2_payload()
        del payload["provenance"][key]
        valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
        assert valid is False
        assert any(f"provenance: missing required key '{key}'" in e for e in errors)


def test_invalid_provenance_types() -> None:
    """Test 35: Non-string files, invalid status values, or non-integer counts in provenance fail."""
    # Empty / blank file
    p1 = build_valid_schema_v2_payload()
    p1["provenance"]["reddit_input_file"] = "   "
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("reddit_input_file: must be non-empty string" in e for e in errors1)

    # Invalid status
    p2 = build_valid_schema_v2_payload()
    p2["provenance"]["youtube_status"] = "finished"
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("youtube_status: must be one of" in e for e in errors2)

    # Boolean count (True is not a strict integer)
    p3 = build_valid_schema_v2_payload()
    p3["provenance"]["reddit_records_count"] = True
    valid3, errors3 = GraphArtifactValidator.validate_graph_artifact(p3)
    assert valid3 is False
    assert any("reddit_records_count: must be non-negative integer" in e for e in errors3)

    # Negative count
    p4 = build_valid_schema_v2_payload()
    p4["provenance"]["youtube_records_count"] = -1
    valid4, errors4 = GraphArtifactValidator.validate_graph_artifact(p4)
    assert valid4 is False
    assert any("youtube_records_count: must be non-negative integer" in e for e in errors4)


def test_contradictory_provenance_status_and_counts() -> None:
    """Test 36: Contradictory provenance status and record count combinations are rejected."""
    # 1. 'loaded' requires count >= 1, but count is 0
    p1 = build_valid_schema_v2_payload()
    p1["provenance"]["reddit_records_count"] = 0
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("reddit_status='loaded' requires reddit_records_count >= 1" in e for e in errors1)

    # 2. 'empty' requires count == 0, but count is 5
    p2 = build_valid_schema_v2_payload()
    p2["provenance"]["reddit_status"] = "empty"
    p2["provenance"]["reddit_records_count"] = 5
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("reddit_status='empty' requires reddit_records_count == 0" in e for e in errors2)

    # 3. 'missing' requires count == 0, but count is 2
    p3 = build_valid_schema_v2_payload()
    p3["provenance"]["youtube_status"] = "missing"
    p3["provenance"]["youtube_records_count"] = 2
    valid3, errors3 = GraphArtifactValidator.validate_graph_artifact(p3)
    assert valid3 is False
    assert any("youtube_status='missing' requires youtube_records_count == 0" in e for e in errors3)

    # 4. 'corrupted' requires count == 0, but count is 3
    p4 = build_valid_schema_v2_payload()
    p4["provenance"]["youtube_status"] = "corrupted"
    p4["provenance"]["youtube_records_count"] = 3
    valid4, errors4 = GraphArtifactValidator.validate_graph_artifact(p4)
    assert valid4 is False
    assert any("youtube_status='corrupted' requires youtube_records_count == 0" in e for e in errors4)


# ==============================================================================
# 9. Top Influencer Ranking Order & Community Reference Tests
# ==============================================================================

def test_unsorted_top_influencers_by_score() -> None:
    """Test 37: Unsorted influencer_score order is rejected."""
    payload = build_valid_schema_v2_payload()
    # Invert scores so lower score comes first
    payload["top_influencers"][0]["influencer_score"] = 0.40
    payload["top_influencers"][1]["influencer_score"] = 0.90
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("unsorted ranking list" in e for e in errors)


def test_unsorted_top_influencers_by_pagerank_tiebreak() -> None:
    """Test 38: Equal influencer_score but unsorted pagerank is rejected."""
    payload = build_valid_schema_v2_payload()
    payload["top_influencers"][0]["influencer_score"] = 0.80
    payload["top_influencers"][1]["influencer_score"] = 0.80
    # Invert PageRank: 0.10 before 0.50 is ascending (should be descending)
    payload["top_influencers"][0]["pagerank"] = 0.10
    payload["top_influencers"][1]["pagerank"] = 0.50
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("unsorted ranking list" in e for e in errors)


def test_unsorted_top_influencers_by_node_id_tiebreak() -> None:
    """Test 39: Equal influencer_score and PageRank but unsorted node_id is rejected."""
    payload = build_valid_schema_v2_payload()
    payload["top_influencers"][0]["influencer_score"] = 0.80
    payload["top_influencers"][1]["influencer_score"] = 0.80
    payload["top_influencers"][0]["pagerank"] = 0.30
    payload["top_influencers"][1]["pagerank"] = 0.30
    # "youtube:channel:UC_TECHGURU" > "reddit:user:alice_data" lexicographically
    # Swapping their node_ids and canonical properties to test tie-break order
    payload["top_influencers"].reverse()
    # Now index 0 has youtube and index 1 has reddit, which violates node_id ascending order
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("unsorted ranking list" in e for e in errors)


def test_top_influencer_community_not_in_distributions() -> None:
    """Test 40: Top influencer referencing community_id not in community_distributions fails."""
    payload = build_valid_schema_v2_payload()
    payload["top_influencers"][0]["community_id"] = 99
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("community_id 99 ('99') not found in community_distributions" in e for e in errors)


def test_top_influencer_community_zero_count() -> None:
    """Test 41: Top influencer community having count < 1 in distributions fails."""
    payload = build_valid_schema_v2_payload()
    payload["community_distributions"]["0"] = 0
    # Adjust community 1 to make total match total_nodes (4)
    payload["community_distributions"]["1"] = 4
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("count in community_distributions is 0, expected >= 1" in e for e in errors)


# ==============================================================================
# 10. Status Consistency Semantics Tests
# ==============================================================================

def test_status_empty_with_nodes_or_metrics_rejected() -> None:
    """Test 42: status='empty' with non-empty nodes, edges, or metrics fails."""
    payload = build_valid_schema_v2_payload()
    payload["status"] = "empty"
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("status 'empty' requires empty nodes list" in e for e in errors)
    assert any("status 'empty' requires empty edges list" in e for e in errors)
    assert any("status 'empty' requires empty top_influencers list" in e for e in errors)


def test_status_empty_both_sources_empty_accepted() -> None:
    """Test 43: status='empty' with both sources empty and zero graph entities is accepted."""
    payload = build_valid_empty_payload()
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True
    assert errors == []


def test_status_empty_both_sources_loaded_zero_entities_accepted() -> None:
    """Test 44: status='empty' with both sources loaded but zero resulting graph entities is accepted."""
    payload = build_valid_empty_payload()
    payload["provenance"]["reddit_status"] = "loaded"
    payload["provenance"]["reddit_records_count"] = 5
    payload["provenance"]["youtube_status"] = "loaded"
    payload["provenance"]["youtube_records_count"] = 3

    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True
    assert errors == []


def test_status_empty_one_missing_source_rejected() -> None:
    """Test 45: status='empty' with one missing source is rejected."""
    payload = build_valid_empty_payload()
    payload["provenance"]["reddit_status"] = "loaded"
    payload["provenance"]["reddit_records_count"] = 5
    payload["provenance"]["youtube_status"] = "missing"
    payload["provenance"]["youtube_records_count"] = 0

    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("status 'empty' requires both sources to succeed" in e for e in errors)


def test_status_empty_one_corrupted_source_rejected() -> None:
    """Test 46: status='empty' with one corrupted source is rejected."""
    payload = build_valid_empty_payload()
    payload["provenance"]["reddit_status"] = "corrupted"
    payload["provenance"]["reddit_records_count"] = 0
    payload["provenance"]["youtube_status"] = "empty"
    payload["provenance"]["youtube_records_count"] = 0

    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("status 'empty' requires both sources to succeed" in e for e in errors)


def test_status_empty_both_sources_unavailable_rejected() -> None:
    """Test 47: status='empty' with both sources unavailable (corrupted/missing) is rejected."""
    payload = build_valid_empty_payload()
    payload["provenance"]["reddit_status"] = "corrupted"
    payload["provenance"]["reddit_records_count"] = 0
    payload["provenance"]["youtube_status"] = "missing"
    payload["provenance"]["youtube_records_count"] = 0

    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("status 'empty' requires both sources to succeed" in e for e in errors)


def test_status_empty_metrics_and_distributions_nonzero_rejected() -> None:
    """Test 48: status='empty' with non-zero communities, density, or distributions is rejected."""
    # communities_count != 0
    p1 = build_valid_empty_payload()
    p1["graph_metrics"]["communities_count"] = 1
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("status 'empty' requires graph_metrics.communities_count == 0" in e for e in errors1)

    # density != 0.0
    p2 = build_valid_empty_payload()
    p2["graph_metrics"]["density"] = 0.5
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("status 'empty' requires graph_metrics.density == 0.0" in e for e in errors2)

    # community_distributions non-empty
    p3 = build_valid_empty_payload()
    p3["community_distributions"] = {"0": 0}
    valid3, errors3 = GraphArtifactValidator.validate_graph_artifact(p3)
    assert valid3 is False
    assert any("status 'empty' requires empty community_distributions" in e for e in errors3)


def test_status_partial_one_valid_and_one_unavailable_accepted() -> None:
    """Test 49: status='partial' with one valid source and one unavailable source is accepted."""
    payload = build_valid_partial_payload()
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True
    assert errors == []


def test_status_partial_when_all_sources_succeeded_rejected() -> None:
    """Test 50: status='partial' when all sources loaded without limitations is rejected."""
    payload = build_valid_schema_v2_payload()
    payload["status"] = "partial"
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("status is 'partial' but all sources succeeded without failure" in e for e in errors)


def test_status_partial_when_all_sources_failed_rejected() -> None:
    """Test 51: status='partial' when all sources failed is rejected (expected status 'error')."""
    payload = build_valid_empty_payload()
    payload["status"] = "partial"
    payload["provenance"]["reddit_status"] = "missing"
    payload["provenance"]["youtube_status"] = "corrupted"
    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is False
    assert any("status is 'partial' but all sources failed" in e for e in errors)


def test_status_success_when_source_failed_rejected() -> None:
    """Test 52: status='success' when any source had missing or corrupted data is rejected."""
    # missing source
    p1 = build_valid_schema_v2_payload()
    p1["provenance"]["youtube_status"] = "missing"
    p1["provenance"]["youtube_records_count"] = 0
    valid1, errors1 = GraphArtifactValidator.validate_graph_artifact(p1)
    assert valid1 is False
    assert any("status is 'success' but provenance indicates source failure" in e for e in errors1)

    # corrupted source
    p2 = build_valid_schema_v2_payload()
    p2["provenance"]["reddit_status"] = "corrupted"
    p2["provenance"]["reddit_records_count"] = 0
    valid2, errors2 = GraphArtifactValidator.validate_graph_artifact(p2)
    assert valid2 is False
    assert any("status is 'success' but provenance indicates source failure" in e for e in errors2)


def test_error_status_behavior_contract() -> None:
    """Test 53: status='error' is valid for diagnostic failure artifacts, but rejected when all sources succeeded."""
    # Valid error artifact: both sources failed
    err_payload = build_valid_empty_payload()
    err_payload["status"] = "error"
    err_payload["provenance"]["reddit_status"] = "missing"
    err_payload["provenance"]["youtube_status"] = "corrupted"
    valid_err, errs = GraphArtifactValidator.validate_graph_artifact(err_payload)
    assert valid_err is True
    assert errs == []

    # Invalid error artifact: all sources loaded successfully without failure
    bad_err_payload = build_valid_schema_v2_payload()
    bad_err_payload["status"] = "error"
    valid_bad, errs_bad = GraphArtifactValidator.validate_graph_artifact(bad_err_payload)
    assert valid_bad is False
    assert any("status is 'error' but all sources succeeded without failure" in e for e in errs_bad)


# ==============================================================================
# 11. Repeated Interactions & Multi-Weight Edge Tests
# ==============================================================================

def test_repeated_interactions_multi_weight_edges_preserve_total_interactions() -> None:
    """Test 54: Multi-weight edges from repeated interactions preserve total_interactions == len(edges)."""
    payload = build_valid_schema_v2_payload()
    # Two directed edges with high repeated interaction weights
    payload["edges"][0]["weight"] = 7
    payload["edges"][1]["weight"] = 13
    # Number of unique directed edges is 2; total_interactions must equal 2
    payload["graph_metrics"]["total_interactions"] = 2

    valid, errors = GraphArtifactValidator.validate_graph_artifact(payload)
    assert valid is True
    assert errors == []

    # If someone erroneously sets total_interactions to the sum of weights (20), it is rejected
    payload_bad = copy.deepcopy(payload)
    payload_bad["graph_metrics"]["total_interactions"] = 20
    valid_bad, errors_bad = GraphArtifactValidator.validate_graph_artifact(payload_bad)
    assert valid_bad is False
    assert any("total_interactions: count (20) does not match edges length (2)" in e for e in errors_bad)



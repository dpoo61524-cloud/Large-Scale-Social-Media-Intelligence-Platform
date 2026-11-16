"""Integration and Orchestration Tests for Graph Pipeline (Stage 7).

Verifies end-to-end execution of run_graph_pipeline:
- Ingestion of processed Reddit and YouTube records into directed interaction graph.
- Explicit source-loading discrimination (loaded, empty, missing, corrupted).
- Complete Schema-v2 emission (nodes, edges, top_influencers, community_distributions, provenance).
- Real GraphArtifactValidator pre-promotion validation.
- Safe atomic artifact promotion via temporary file synchronization and os.replace.
- Non-destructive execution ensuring input and canonical files are protected on failure.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from src.graph.graph_pipeline import load_and_classify_source, run_graph_pipeline
from src.graph.graph_validator import GraphArtifactValidator


@pytest.fixture
def sample_reddit_graph_data() -> list[dict[str, Any]]:
    """Sample preprocessed Reddit records with user interactions and mentions."""
    return [
        {
            "post_id": "r_101",
            "author": "Alice_Data",
            "title_cleaned": "Discussing graph theory with u/Bob_ML and u/Charlie_Eng",
            "text_cleaned": "Great insights from u/Bob_ML on PageRank algorithms.",
            "created_utc": 1730101200,
            "collected_at": "2026-11-05T16:54:13+00:00",
        },
        {
            "post_id": "r_102",
            "author": "Bob_ML",
            "title_cleaned": "Response to u/Alice_Data on Neo4j scaling",
            "text_cleaned": "Agree with u/Alice_Data and cc u/David_Dev.",
            "created_utc": 1730104800,
            "collected_at": "2026-11-05T16:54:13+00:00",
        },
    ]


@pytest.fixture
def sample_youtube_graph_data() -> list[dict[str, Any]]:
    """Sample preprocessed YouTube records with channel creators and mentions."""
    return [
        {
            "video_id": "y_201",
            "channel_id": "UC_TECHGURU",
            "channel_title": "TechGuru_YT",
            "title_cleaned": "Graph Analytics Masterclass with @Alice_Data",
            "description_cleaned": "Featuring ideas from @Alice_Data and @Bob_ML.",
            "published_at": "2026-11-01T12:00:00Z",
            "collected_at": "2026-11-06T13:00:00+00:00",
        }
    ]


# ==============================================================================
# A. Source Classification Tests
# ==============================================================================

def test_source_classification_valid_non_empty_loaded(tmp_path: Path) -> None:
    """Test A.1: Input file exists, parses, and has valid source records -> 'loaded'."""
    # Test valid Reddit source
    reddit_src = tmp_path / "reddit_valid.json"
    reddit_src.write_text(
        json.dumps([{"post_id": "r_101", "author": "Alice_Data", "title": "Hello"}]),
        encoding="utf-8",
    )
    status_r, records_r, count_r = load_and_classify_source(reddit_src, source_type="reddit")
    assert status_r == "loaded"
    assert count_r == 1
    assert len(records_r) == 1

    # Test valid YouTube source
    youtube_src = tmp_path / "youtube_valid.json"
    youtube_src.write_text(
        json.dumps([{"video_id": "y_201", "channel_id": "UC_TEST", "title": "Video"}]),
        encoding="utf-8",
    )
    status_y, records_y, count_y = load_and_classify_source(youtube_src, source_type="youtube")
    assert status_y == "loaded"
    assert count_y == 1
    assert len(records_y) == 1


def test_source_classification_io_read_error_corrupted(tmp_path: Path) -> None:
    """Test A.1b: Input file read I/O error (OSError) -> 'corrupted'."""
    src = tmp_path / "io_error.json"
    src.write_text("[]", encoding="utf-8")
    with patch("builtins.open", side_effect=OSError("Disk read failure")):
        status, records, count = load_and_classify_source(src, source_type="reddit")
    assert status == "corrupted"
    assert count == 0
    assert records == []


def test_source_classification_malformed_reddit_records_corrupted(tmp_path: Path) -> None:
    """Test A.1c: Reddit records missing or invalid post_id/author -> 'corrupted'."""
    # Missing post_id
    src1 = tmp_path / "reddit_no_post_id.json"
    src1.write_text(json.dumps([{"author": "alice"}]), encoding="utf-8")
    s1, r1, c1 = load_and_classify_source(src1, source_type="reddit")
    assert s1 == "corrupted"
    assert c1 == 0
    assert r1 == []

    # Empty post_id
    src2 = tmp_path / "reddit_empty_post_id.json"
    src2.write_text(json.dumps([{"post_id": "   ", "author": "alice"}]), encoding="utf-8")
    s2, r2, c2 = load_and_classify_source(src2, source_type="reddit")
    assert s2 == "corrupted"
    assert c2 == 0

    # Non-string post_id
    src3 = tmp_path / "reddit_int_post_id.json"
    src3.write_text(json.dumps([{"post_id": 12345, "author": "alice"}]), encoding="utf-8")
    s3, r3, c3 = load_and_classify_source(src3, source_type="reddit")
    assert s3 == "corrupted"
    assert c3 == 0

    # Missing author
    src4 = tmp_path / "reddit_no_author.json"
    src4.write_text(json.dumps([{"post_id": "r_1"}]), encoding="utf-8")
    s4, r4, c4 = load_and_classify_source(src4, source_type="reddit")
    assert s4 == "corrupted"
    assert c4 == 0

    # Empty author
    src5 = tmp_path / "reddit_empty_author.json"
    src5.write_text(json.dumps([{"post_id": "r_1", "author": ""}]), encoding="utf-8")
    s5, r5, c5 = load_and_classify_source(src5, source_type="reddit")
    assert s5 == "corrupted"
    assert c5 == 0


def test_source_classification_malformed_youtube_records_corrupted(tmp_path: Path) -> None:
    """Test A.1d: YouTube records missing/invalid video_id or channel_id -> 'corrupted'."""
    # Missing video_id
    src1 = tmp_path / "yt_no_video_id.json"
    src1.write_text(json.dumps([{"channel_id": "UC123"}]), encoding="utf-8")
    s1, r1, c1 = load_and_classify_source(src1, source_type="youtube")
    assert s1 == "corrupted"
    assert c1 == 0
    assert r1 == []

    # Empty video_id
    src2 = tmp_path / "yt_empty_video_id.json"
    src2.write_text(json.dumps([{"video_id": "  ", "channel_id": "UC123"}]), encoding="utf-8")
    s2, r2, c2 = load_and_classify_source(src2, source_type="youtube")
    assert s2 == "corrupted"
    assert c2 == 0

    # Non-string video_id
    src2b = tmp_path / "yt_int_video_id.json"
    src2b.write_text(json.dumps([{"video_id": 12345, "channel_id": "UC123"}]), encoding="utf-8")
    s2b, r2b, c2b = load_and_classify_source(src2b, source_type="youtube")
    assert s2b == "corrupted"
    assert c2b == 0

    # Missing channel_id (title-only)
    src3 = tmp_path / "yt_title_only.json"
    src3.write_text(json.dumps([{"video_id": "y_1", "channel_title": "Tech"}]), encoding="utf-8")
    s3, r3, c3 = load_and_classify_source(src3, source_type="youtube")
    assert s3 == "corrupted"
    assert c3 == 0

    # channel_id is None
    src4 = tmp_path / "yt_none_channel_id.json"
    src4.write_text(json.dumps([{"video_id": "y_1", "channel_id": None, "channel_title": "Tech"}]), encoding="utf-8")
    s4, r4, c4 = load_and_classify_source(src4, source_type="youtube")
    assert s4 == "corrupted"
    assert c4 == 0

    # channel_id is empty string
    src5 = tmp_path / "yt_empty_channel_id.json"
    src5.write_text(json.dumps([{"video_id": "y_1", "channel_id": ""}]), encoding="utf-8")
    s5, r5, c5 = load_and_classify_source(src5, source_type="youtube")
    assert s5 == "corrupted"
    assert c5 == 0

    # channel_id is whitespace-only
    src6 = tmp_path / "yt_whitespace_channel_id.json"
    src6.write_text(json.dumps([{"video_id": "y_1", "channel_id": "   "}]), encoding="utf-8")
    s6, r6, c6 = load_and_classify_source(src6, source_type="youtube")
    assert s6 == "corrupted"
    assert c6 == 0

    # channel_id contains spaces
    src7 = tmp_path / "yt_space_channel_id.json"
    src7.write_text(json.dumps([{"video_id": "y_1", "channel_id": "Cool Channel"}]), encoding="utf-8")
    s7, r7, c7 = load_and_classify_source(src7, source_type="youtube")
    assert s7 == "corrupted"
    assert c7 == 0

    # channel_id contains slash or invalid punctuation
    src8 = tmp_path / "yt_slash_channel_id.json"
    src8.write_text(json.dumps([{"video_id": "y_1", "channel_id": "UC/123"}]), encoding="utf-8")
    s8, r8, c8 = load_and_classify_source(src8, source_type="youtube")
    assert s8 == "corrupted"
    assert c8 == 0

    # Non-string channel_id
    src9 = tmp_path / "yt_int_channel.json"
    src9.write_text(json.dumps([{"video_id": "y_1", "channel_id": 999}]), encoding="utf-8")
    s9, r9, c9 = load_and_classify_source(src9, source_type="youtube")
    assert s9 == "corrupted"
    assert c9 == 0


def test_source_classification_empty_dict_records_corrupted(tmp_path: Path) -> None:
    """Test A.1e: [{}] rejected for Reddit, YouTube, and cannot masquerade as loaded or empty graph."""
    r_empty_dict = tmp_path / "reddit_empty_dict.json"
    r_empty_dict.write_text(json.dumps([{}]), encoding="utf-8")
    s_r, r_r, c_r = load_and_classify_source(r_empty_dict, source_type="reddit")
    assert s_r == "corrupted"
    assert c_r == 0
    assert r_r == []

    y_empty_dict = tmp_path / "youtube_empty_dict.json"
    y_empty_dict.write_text(json.dumps([{}]), encoding="utf-8")
    s_y, r_y, c_y = load_and_classify_source(y_empty_dict, source_type="youtube")
    assert s_y == "corrupted"
    assert c_y == 0
    assert r_y == []

    # End-to-end pipeline run: both sources with [{}] must derive 'error' and NOT promote
    out_path = tmp_path / "out.json"
    result = run_graph_pipeline(
        reddit_input_path=r_empty_dict,
        youtube_input_path=y_empty_dict,
        output_path=out_path,
    )
    assert result["status"] == "error"
    assert result["provenance"]["reddit_status"] == "corrupted"
    assert result["provenance"]["youtube_status"] == "corrupted"
    assert not out_path.exists()


def test_source_classification_legitimate_youtube_channel_variants(tmp_path: Path) -> None:
    """Test A.1f: Legitimate YouTube records with channel_id only, channel_title=None, or both."""
    # channel_id only
    src1 = tmp_path / "yt_id_only.json"
    src1.write_text(json.dumps([{"video_id": "v1", "channel_id": "UC_CHANNEL"}]), encoding="utf-8")
    s1, r1, c1 = load_and_classify_source(src1, source_type="youtube")
    assert s1 == "loaded"
    assert c1 == 1

    # channel_id with channel_title=None
    src2 = tmp_path / "yt_title_none.json"
    src2.write_text(
        json.dumps([{"video_id": "v2", "channel_id": "UC_CHANNEL", "channel_title": None}]),
        encoding="utf-8",
    )
    s2, r2, c2 = load_and_classify_source(src2, source_type="youtube")
    assert s2 == "loaded"
    assert c2 == 1

    # both channel_id and channel_title
    src3 = tmp_path / "yt_both.json"
    src3.write_text(
        json.dumps([{"video_id": "v3", "channel_id": "UC_CHANNEL", "channel_title": "Cool Channel"}]),
        encoding="utf-8",
    )
    s3, r3, c3 = load_and_classify_source(src3, source_type="youtube")
    assert s3 == "loaded"
    assert c3 == 1

    # channel_title only without channel_id is corrupted
    src4 = tmp_path / "yt_title_only_rejected.json"
    src4.write_text(
        json.dumps([{"video_id": "v4", "channel_title": "Cool Channel"}]),
        encoding="utf-8",
    )
    s4, r4, c4 = load_and_classify_source(src4, source_type="youtube")
    assert s4 == "corrupted"
    assert c4 == 0
    assert r4 == []


def test_source_classification_mixed_valid_and_malformed_rejects_all(tmp_path: Path) -> None:
    """Test A.1g: Mixed valid and malformed records fail closed as corrupted (no partial loading)."""
    # Reddit mix
    r_mix = tmp_path / "reddit_mixed.json"
    r_mix.write_text(
        json.dumps([
            {"post_id": "r_1", "author": "Alice"},
            {"post_id": "r_2"},  # missing author
        ]),
        encoding="utf-8",
    )
    s_r, r_r, c_r = load_and_classify_source(r_mix, source_type="reddit")
    assert s_r == "corrupted"
    assert c_r == 0
    assert r_r == []

    # YouTube mix: one valid, one title-only (missing channel_id)
    y_mix = tmp_path / "youtube_mixed.json"
    y_mix.write_text(
        json.dumps([
            {"video_id": "v_1", "channel_id": "UC1"},
            {"video_id": "v_2", "channel_title": "Tech"},  # missing channel_id
        ]),
        encoding="utf-8",
    )
    s_y, r_y, c_y = load_and_classify_source(y_mix, source_type="youtube")
    assert s_y == "corrupted"
    assert c_y == 0
    assert r_y == []


def test_source_discrimination_omitted_or_invalid_raises_value_error(tmp_path: Path) -> None:
    """Test A.1h: Omitted, None, misspelled, or unsupported source_type fails closed with ValueError."""
    src = tmp_path / "reddit_data.json"
    src.write_text(json.dumps([{"post_id": "r1", "author": "alice"}]), encoding="utf-8")

    # Omitted source_type
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src)

    # Explicit None
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src, source_type=None)

    # Unsupported platform
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src, source_type="twitter")

    # Misspelled platform
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src, source_type="reditt")

    # Uppercase
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src, source_type="REDDIT")

    # Whitespace padding
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src, source_type=" reddit ")

    # Non-string type
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(src, source_type=123)

    # Filename does not infer or bypass validation: non-existent file with bad source_type raises ValueError
    non_existent = tmp_path / "reddit_non_existent.json"
    with pytest.raises(ValueError, match="Invalid source_type"):
        load_and_classify_source(non_existent, source_type="unknown")


def test_source_classification_valid_empty(tmp_path: Path) -> None:
    """Test A.2: Input file exists and parses as empty list -> 'empty'."""
    src = tmp_path / "empty.json"
    src.write_text("[]", encoding="utf-8")
    status, records, count = load_and_classify_source(src, source_type="reddit")
    assert status == "empty"
    assert count == 0
    assert records == []


def test_source_classification_missing(tmp_path: Path) -> None:
    """Test A.3: Input file does not exist -> 'missing'."""
    src = tmp_path / "non_existent.json"
    status, records, count = load_and_classify_source(src, source_type="reddit")
    assert status == "missing"
    assert count == 0
    assert records == []


def test_source_classification_invalid_json_corrupted(tmp_path: Path) -> None:
    """Test A.4: Input file contains malformed JSON -> 'corrupted'."""
    src = tmp_path / "syntax_error.json"
    src.write_text("{ unquoted_bad_json: ", encoding="utf-8")
    status, records, count = load_and_classify_source(src, source_type="reddit")
    assert status == "corrupted"
    assert count == 0
    assert records == []


def test_source_classification_invalid_top_level_corrupted(tmp_path: Path) -> None:
    """Test A.5: Input file top-level is dict or string, not list -> 'corrupted'."""
    src1 = tmp_path / "top_level_dict.json"
    src1.write_text(json.dumps({"error": "not a list"}), encoding="utf-8")
    status1, records1, count1 = load_and_classify_source(src1, source_type="reddit")
    assert status1 == "corrupted"
    assert count1 == 0
    assert records1 == []

    src2 = tmp_path / "top_level_str.json"
    src2.write_text(json.dumps("just a string"), encoding="utf-8")
    status2, records2, count2 = load_and_classify_source(src2, source_type="youtube")
    assert status2 == "corrupted"
    assert count2 == 0
    assert records2 == []


def test_source_classification_non_dict_records_corrupted(tmp_path: Path) -> None:
    """Test A.6: Input list contains non-dict items -> 'corrupted'."""
    src = tmp_path / "mixed_items.json"
    src.write_text(json.dumps([{"author": "valid"}, "not_a_dict", 123]), encoding="utf-8")
    status, records, count = load_and_classify_source(src, source_type="reddit")
    assert status == "corrupted"
    assert count == 0
    assert records == []


def test_source_classification_one_loaded_one_missing(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test A.7: Pipeline provenance accurately tracks one loaded and one missing source."""
    r_path = tmp_path / "reddit.json"
    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path = tmp_path / "missing_youtube.json"
    out_path = tmp_path / "out.json"

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
    )
    prov = result["provenance"]
    assert prov["reddit_status"] == "loaded"
    assert prov["reddit_records_count"] == len(sample_reddit_graph_data)
    assert prov["youtube_status"] == "missing"
    assert prov["youtube_records_count"] == 0
    assert result["status"] == "partial"


def test_source_classification_one_empty_one_corrupted(tmp_path: Path) -> None:
    """Test A.8: Pipeline provenance accurately tracks one empty and one corrupted source."""
    r_path = tmp_path / "empty_reddit.json"
    r_path.write_text("[]", encoding="utf-8")
    y_path = tmp_path / "bad_youtube.json"
    y_path.write_text("{bad", encoding="utf-8")
    out_path = tmp_path / "out.json"

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
    )
    prov = result["provenance"]
    assert prov["reddit_status"] == "empty"
    assert prov["reddit_records_count"] == 0
    assert prov["youtube_status"] == "corrupted"
    assert prov["youtube_records_count"] == 0
    assert result["status"] == "partial"


# ==============================================================================
# B. Schema-v2 Emission Tests
# ==============================================================================

def test_pipeline_emits_valid_schema_v2_structure(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Test B: Verifies that run_graph_pipeline emits a complete, compliant Schema-v2 artifact."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "graph_analysis_results.json"

    reddit_path.write_text(json.dumps(sample_reddit_graph_data, indent=2), encoding="utf-8")
    youtube_path.write_text(json.dumps(sample_youtube_graph_data, indent=2), encoding="utf-8")

    result = run_graph_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        top_n_influencers=10,
    )

    # 1. Schema version
    assert result["schema_version"] == "2.0"
    assert result["status"] == "success"
    assert isinstance(result["generated_at"], str)

    # 2. Complete nodes validation
    nodes = result["nodes"]
    assert isinstance(nodes, list)
    assert len(nodes) >= 4  # Alice_Data, Bob_ML, Charlie_Eng, David_Dev, TechGuru_YT
    # Nodes are sorted by node_id ascending
    node_ids = [n["node_id"] for n in nodes]
    assert node_ids == sorted(node_ids)

    # 3. Complete edges validation
    edges = result["edges"]
    assert isinstance(edges, list)
    assert len(edges) >= 3
    # Edges are sorted by (source, target)
    edge_keys = [(e["source"], e["target"]) for e in edges]
    assert edge_keys == sorted(edge_keys)

    # Edge attributes
    for edge in edges:
        assert edge["weight"] >= 1
        assert isinstance(edge["interaction_types"], list)
        assert isinstance(edge["platforms"], list)

    # 4. Top influencers deterministic sorting
    top_inf = result["top_influencers"]
    assert len(top_inf) > 0
    for idx in range(len(top_inf) - 1):
        cur = top_inf[idx]
        nxt = top_inf[idx + 1]
        cur_key = (-cur["influencer_score"], -cur["pagerank"], cur["node_id"])
        nxt_key = (-nxt["influencer_score"], -nxt["pagerank"], nxt["node_id"])
        assert cur_key <= nxt_key

    # 5. Community distributions reconcile with graph nodes
    dist = result["community_distributions"]
    assert isinstance(dist, dict)
    total_in_dist = sum(dist.values())
    assert total_in_dist == len(nodes)

    # 6. Metrics definitions
    metrics = result["graph_metrics"]
    assert metrics["total_users"] == len(nodes)
    assert metrics["total_interactions"] == len(edges)
    assert metrics["is_directed"] is True
    assert metrics["communities_count"] == len(dist)

    # 7. Identity preservation & platform namespaces
    platforms = {n["platform"] for n in nodes}
    assert platforms == {"reddit", "youtube"}

    yt_creator_nodes = [n for n in nodes if n["identity_type"] == "creator"]
    assert len(yt_creator_nodes) == 1
    assert yt_creator_nodes[0]["channel_id"] == "UC_TECHGURU"
    assert yt_creator_nodes[0]["node_id"] == "youtube:channel:UC_TECHGURU"

    yt_mention_nodes = [n for n in nodes if n["platform"] == "youtube" and n["identity_type"] == "mention"]
    for mn in yt_mention_nodes:
        assert mn["channel_id"] is None
        assert mn["is_author"] is False
        assert mn["activity_count"] == 0

    # 8. Real validator accepts payload directly
    is_valid, errors = GraphArtifactValidator.validate_graph_artifact(result)
    assert is_valid is True
    assert errors == []


# ==============================================================================
# C. Status Tests
# ==============================================================================

def test_pipeline_status_success_both_sources_valid(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Test C.1: Both sources valid with graph entities -> 'success' and promoted."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text(json.dumps(sample_youtube_graph_data), encoding="utf-8")

    result = run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
    assert result["status"] == "success"
    assert out_path.exists()


def test_pipeline_status_empty_both_sources_valid_zero_entities(tmp_path: Path) -> None:
    """Test C.2: Both sources valid empty datasets -> 'empty' and promoted."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text("[]", encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    result = run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
    assert result["status"] == "empty"
    assert result["graph_metrics"]["total_users"] == 0
    assert result["graph_metrics"]["total_interactions"] == 0
    assert out_path.exists()

    # File on disk validates as empty schema-v2
    with open(out_path, "r", encoding="utf-8") as f:
        promoted_data = json.load(f)
    valid, errors = GraphArtifactValidator.validate_graph_artifact(promoted_data)
    assert valid is True
    assert errors == []


def test_pipeline_status_partial_one_valid_one_missing(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test C.3: Valid Reddit, missing YouTube -> 'partial' and promoted."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "non_existent_youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")

    result = run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
    assert result["status"] == "partial"
    assert out_path.exists()


def test_pipeline_status_partial_one_valid_one_corrupted(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test C.4: Valid Reddit, corrupted YouTube -> 'partial' and promoted."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "bad_youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("{bad json", encoding="utf-8")

    result = run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
    assert result["status"] == "partial"
    assert out_path.exists()


def test_pipeline_status_error_both_sources_unavailable(tmp_path: Path) -> None:
    """Test C.5: Both sources missing/corrupted -> 'error', NOT promoted to disk."""
    r_path = tmp_path / "missing_reddit.json"
    y_path = tmp_path / "corrupted_youtube.json"
    out_path = tmp_path / "out.json"

    y_path.write_text("not json content", encoding="utf-8")

    result = run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
    assert result["status"] == "error"
    assert not out_path.exists()


def test_pipeline_unexpected_execution_failure_becomes_error(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test C.6: Unexpected failure in builder/ranker becomes 'error' and is NOT promoted."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_builder = MagicMock()
    mock_builder.build_interaction_graph.side_effect = RuntimeError("Fatal graph memory error")

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
        builder=mock_builder,
    )
    assert result["status"] == "error"
    assert not out_path.exists()


# ==============================================================================
# D. Validator Integration Tests
# ==============================================================================

def test_validation_failure_prevents_persistence_and_promotion(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test D.2: If validator reports invalid payload, promotion is aborted and ValueError raised."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch.object(
        GraphArtifactValidator,
        "validate_graph_artifact",
        return_value=(False, ["Mock schema violation"]),
    ):
        with pytest.raises(ValueError, match="Graph artifact validation failed"):
            run_graph_pipeline(
                reddit_input_path=r_path,
                youtube_input_path=y_path,
                output_path=out_path,
            )

    assert not out_path.exists()


def test_invalid_payload_never_overwrites_existing_canonical(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test D.5: Existing canonical artifact survives validation failure without alteration."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    existing_content = json.dumps({"preserved": "data_before_failure"})
    out_path.write_text(existing_content, encoding="utf-8")

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch.object(
        GraphArtifactValidator,
        "validate_graph_artifact",
        return_value=(False, ["Strict rejection"]),
    ):
        with pytest.raises(ValueError):
            run_graph_pipeline(
                reddit_input_path=r_path,
                youtube_input_path=y_path,
                output_path=out_path,
            )

    # Existing file must be bitwise identical
    assert out_path.read_text(encoding="utf-8") == existing_content


def test_error_status_never_reaches_os_replace(tmp_path: Path) -> None:
    """Test D.6: An artifact with status='error' never triggers os.replace."""
    r_path = tmp_path / "missing_r.json"
    y_path = tmp_path / "missing_y.json"
    out_path = tmp_path / "out.json"

    with patch("os.replace") as mock_replace:
        result = run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
        )
        assert result["status"] == "error"
        mock_replace.assert_not_called()

    assert not out_path.exists()


# ==============================================================================
# E. Atomic Promotion Tests
# ==============================================================================

def test_atomic_promotion_creates_and_replaces(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.1-4: Promotes using os.replace; resulting file is parseable and valid."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "sub_dir" / "canonical.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch("os.replace", wraps=os.replace) as spy_replace:
        result = run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
        )
        assert spy_replace.called
        # Verify source temp file was in the same destination directory
        temp_src = spy_replace.call_args[0][0]
        dest_target = spy_replace.call_args[0][1]
        assert Path(temp_src).parent == Path(dest_target).parent

    assert out_path.exists()
    # Confirm valid JSON on disk
    with open(out_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["schema_version"] == "2.0"
    assert disk_data["status"] == "success"


def test_existing_canonical_survives_replace_failure(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.9: Existing canonical content survives os.replace failure, temp file cleaned up."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    existing_content = '{"prior": "valid_content"}'
    out_path.write_text(existing_content, encoding="utf-8")

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch("os.replace", side_effect=OSError("Simulated atomic replacement failure")):
        with pytest.raises(OSError, match="Simulated atomic replacement failure"):
            run_graph_pipeline(
                reddit_input_path=r_path,
                youtube_input_path=y_path,
                output_path=out_path,
            )

    # Existing content preserved
    assert out_path.read_text(encoding="utf-8") == existing_content
    # No orphaned temporary files left in destination directory
    temp_files = list(out_path.parent.glob(".tmp_*"))
    assert temp_files == []


def test_existing_canonical_survives_read_back_validation_failure(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.8: Read-back validation failure aborts promotion and preserves existing canonical file."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    existing_content = '{"canonical": "data"}'
    out_path.write_text(existing_content, encoding="utf-8")

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    # In-memory validation passes, but read-back validation fails
    call_count = 0
    original_validate = GraphArtifactValidator.validate_graph_artifact

    def mock_validate(payload: Any) -> tuple[bool, list[str]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return original_validate(payload)
        return False, ["Corrupted during serialization read-back"]

    with patch.object(GraphArtifactValidator, "validate_graph_artifact", side_effect=mock_validate):
        with pytest.raises(ValueError, match="Read-back graph artifact validation failed"):
            run_graph_pipeline(
                reddit_input_path=r_path,
                youtube_input_path=y_path,
                output_path=out_path,
            )

    assert out_path.read_text(encoding="utf-8") == existing_content
    temp_files = list(out_path.parent.glob(".tmp_*"))
    assert temp_files == []


def test_no_temporary_files_left_after_successful_promotion(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.11: No orphaned temporary files remain after successful promotion."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
    assert out_path.exists()
    temp_files = list(out_path.parent.glob(".tmp_*"))
    assert temp_files == []


def test_serialization_failure_preserves_canonical_artifact(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.12: json.dump failure during atomic promotion preserves canonical artifact and cleans up temp."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    existing_content = '{"preserved": "canonical_payload_before_serialization_error"}'
    out_path.write_text(existing_content, encoding="utf-8")

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch("json.dump", side_effect=TypeError("Non-serializable object in payload")):
        with pytest.raises(TypeError, match="Non-serializable object in payload"):
            run_graph_pipeline(
                reddit_input_path=r_path,
                youtube_input_path=y_path,
                output_path=out_path,
            )

    # Canonical artifact unchanged
    assert out_path.read_text(encoding="utf-8") == existing_content
    # Temporary files cleaned up
    temp_files = list(out_path.parent.glob(".tmp_*"))
    assert temp_files == []


def test_temporary_file_write_failure_preserves_canonical_artifact(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.13: File write/flush failure preserves canonical artifact and cleans up temp."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    existing_content = '{"preserved": "canonical_payload_before_write_error"}'
    out_path.write_text(existing_content, encoding="utf-8")

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch("json.dump", side_effect=OSError("No space left on device")):
        with pytest.raises(OSError, match="No space left on device"):
            run_graph_pipeline(
                reddit_input_path=r_path,
                youtube_input_path=y_path,
                output_path=out_path,
            )

    assert out_path.read_text(encoding="utf-8") == existing_content
    temp_files = list(out_path.parent.glob(".tmp_*"))
    assert temp_files == []


def test_fsync_failure_propagates_and_preserves_canonical_artifact(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.14: os.fsync failure propagates, prevents os.replace, preserves canonical artifact."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "canonical.json"

    existing_content = '{"preserved": "canonical_payload_before_fsync_error"}'
    out_path.write_text(existing_content, encoding="utf-8")

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    with patch("os.fsync", side_effect=OSError("Flush sync failed")):
        with patch("os.replace") as mock_replace:
            with pytest.raises(OSError, match="Flush sync failed"):
                run_graph_pipeline(
                    reddit_input_path=r_path,
                    youtube_input_path=y_path,
                    output_path=out_path,
                )
            mock_replace.assert_not_called()

    # Canonical artifact unchanged
    assert out_path.read_text(encoding="utf-8") == existing_content
    # Temporary files cleaned up
    temp_files = list(out_path.parent.glob(".tmp_*"))
    assert temp_files == []


def test_temporary_filename_uniqueness_and_directory(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Test E.15: Temporary files are placed in destination directory and have unique UUIDs."""
    import uuid

    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "sub_dir" / "canonical_results.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    temp_paths_used: list[Path] = []

    original_replace = os.replace

    def spy_replace(src: str | os.PathLike, dst: str | os.PathLike) -> None:
        temp_paths_used.append(Path(src))
        original_replace(src, dst)

    with patch("os.replace", side_effect=spy_replace):
        run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)
        run_graph_pipeline(reddit_input_path=r_path, youtube_input_path=y_path, output_path=out_path)

    assert len(temp_paths_used) == 2
    temp1, temp2 = temp_paths_used[0], temp_paths_used[1]

    # Verify directory matches destination directory
    assert temp1.parent == out_path.parent
    assert temp2.parent == out_path.parent

    # Verify filename pattern: .tmp_<name>_<uuid_hex>
    prefix = f".tmp_{out_path.name}_"
    assert temp1.name.startswith(prefix)
    assert temp2.name.startswith(prefix)

    # Verify unique UUID component
    uuid1_hex = temp1.name[len(prefix):]
    uuid2_hex = temp2.name[len(prefix):]
    assert len(uuid1_hex) == 32
    assert len(uuid2_hex) == 32
    assert uuid.UUID(hex=uuid1_hex)
    assert uuid.UUID(hex=uuid2_hex)
    assert temp1 != temp2


# ==============================================================================
# F. Non-Destructive Source Data Integrity & Workspace Runs
# ==============================================================================

def test_graph_pipeline_non_destructive_integrity(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Verifies that source input files remain bitwise identical after pipeline run."""
    reddit_path = tmp_path / "reddit_processed.json"
    youtube_path = tmp_path / "youtube_processed.json"
    output_path = tmp_path / "output.json"

    reddit_bytes = json.dumps(sample_reddit_graph_data, indent=2).encode("utf-8")
    youtube_bytes = json.dumps(sample_youtube_graph_data, indent=2).encode("utf-8")

    reddit_path.write_bytes(reddit_bytes)
    youtube_path.write_bytes(youtube_bytes)

    run_graph_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
    )

    # Input files must match byte-for-byte
    assert reddit_path.read_bytes() == reddit_bytes
    assert youtube_path.read_bytes() == youtube_bytes


def test_graph_pipeline_handles_missing_inputs(tmp_path: Path) -> None:
    """Verifies pipeline handles non-existent input files as diagnostic error without promotion."""
    output_path = tmp_path / "empty_graph_output.json"

    result = run_graph_pipeline(
        reddit_input_path=tmp_path / "non_existent_reddit.json",
        youtube_input_path=tmp_path / "non_existent_youtube.json",
        output_path=output_path,
    )

    assert result["status"] == "error"
    assert result["graph_metrics"]["total_users"] == 0
    assert result["graph_metrics"]["total_interactions"] == 0
    assert not output_path.exists()


def test_graph_pipeline_runs_on_actual_workspace_files(tmp_path: Path) -> None:
    """Tests executing pipeline on actual workspace processed data."""
    workspace_reddit = Path("data/processed/reddit_posts_processed.json")
    workspace_youtube = Path("data/processed/youtube_videos_processed.json")
    output_path = tmp_path / "graph_analysis_results.json"

    if not workspace_reddit.exists():
        pytest.skip("Workspace processed Reddit file not present.")

    result = run_graph_pipeline(
        reddit_input_path=workspace_reddit,
        youtube_input_path=workspace_youtube,
        output_path=output_path,
    )

    assert result["status"] in {"success", "partial"}
    assert result["graph_metrics"]["total_users"] > 0
    assert output_path.exists()
    assert "top_influencers" in result
    assert "nodes" in result
    assert "edges" in result
    assert "neo4j_export" in result

    # Validate output on disk with real validator
    with open(output_path, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    valid, errors = GraphArtifactValidator.validate_graph_artifact(disk_data)
    assert valid is True
    assert errors == []


# ==============================================================================
# H. Task 8A Focused Community Coverage & Full Export Tests
# ==============================================================================

def test_every_graph_node_receives_correct_community_id(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Tests that 100% of graph nodes in Stage 7 artifact have valid, non-negative community_id."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text(json.dumps(sample_youtube_graph_data), encoding="utf-8")

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
    )

    nodes = result["nodes"]
    assert len(nodes) > 0
    dist = result["community_distributions"]

    for node in nodes:
        assert "community_id" in node
        cid = node["community_id"]
        assert isinstance(cid, int)
        assert not isinstance(cid, bool)
        assert cid >= 0
        assert str(cid) in dist
        assert dist[str(cid)] >= 1


def test_exact_community_mapping_succeeds_when_all_nodes_covered(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that exact community mapping succeeds, passes validator, and promotes to disk."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "promoted_graph.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
    )

    assert result["status"] == "success"
    assert out_path.exists()
    assert all("community_id" in n for n in result["nodes"])

    # Verify promoted JSON on disk matches
    with open(out_path, "r", encoding="utf-8") as f:
        promoted = json.load(f)
    assert len(promoted["nodes"]) == len(result["nodes"])
    assert all("community_id" in n for n in promoted["nodes"])


def test_missing_community_assignment_raises_value_error(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that an incomplete community partition (missing node) raises ValueError and fails closed."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "should_not_exist.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_ranker = MagicMock()
    # Omit one node from communities
    mock_ranker.detect_communities.return_value = {
        "reddit:user:alice_data": 0,
        "reddit:user:bob_ml": 0,
        # missing Charlie_Eng, David_Dev
    }
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError, match="Community assignments do not match graph nodes: missing="):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
            ranker=mock_ranker,
        )

    assert not out_path.exists()


def test_unexpected_community_assignment_raises_value_error(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that unexpected node in community partition raises ValueError and fails closed."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "should_not_exist.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_ranker = MagicMock()
    # Include all 4 real nodes PLUS a ghost node
    mock_ranker.detect_communities.return_value = {
        "reddit:user:alice_data": 0,
        "reddit:user:bob_ml": 0,
        "reddit:user:charlie_eng": 0,
        "reddit:user:david_dev": 0,
        "reddit:user:ghost_node": 1,
    }
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError, match=r"Community assignments do not match graph nodes:.*unexpected="):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
            ranker=mock_ranker,
        )

    assert not out_path.exists()


def test_boolean_community_id_rejected(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that boolean community_id is explicitly rejected (not treated as integer)."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "should_not_exist.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_ranker = MagicMock()
    mock_ranker.detect_communities.return_value = {
        "reddit:user:alice_data": True,  # boolean!
        "reddit:user:bob_ml": 0,
        "reddit:user:charlie_eng": 0,
        "reddit:user:david_dev": 0,
    }
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError, match="Invalid community_id True for node 'reddit:user:alice_data': must be a non-negative integer"):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
            ranker=mock_ranker,
        )

    assert not out_path.exists()


def test_negative_community_id_rejected(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that negative integer community_id is rejected."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "should_not_exist.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_ranker = MagicMock()
    mock_ranker.detect_communities.return_value = {
        "reddit:user:alice_data": -1,
        "reddit:user:bob_ml": 0,
        "reddit:user:charlie_eng": 0,
        "reddit:user:david_dev": 0,
    }
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError, match="Invalid community_id -1 for node 'reddit:user:alice_data': must be a non-negative integer"):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
            ranker=mock_ranker,
        )

    assert not out_path.exists()


def test_float_community_id_rejected(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that float community_id is rejected without silent truncation."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "should_not_exist.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_ranker = MagicMock()
    mock_ranker.detect_communities.return_value = {
        "reddit:user:alice_data": 2.5,
        "reddit:user:bob_ml": 0,
        "reddit:user:charlie_eng": 0,
        "reddit:user:david_dev": 0,
    }
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError, match="Invalid community_id 2.5 for node 'reddit:user:alice_data': must be a non-negative integer"):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
            ranker=mock_ranker,
        )

    assert not out_path.exists()


def test_string_community_id_rejected(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that string community_id is rejected without string-to-int conversion."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "should_not_exist.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    mock_ranker = MagicMock()
    mock_ranker.detect_communities.return_value = {
        "reddit:user:alice_data": "0",
        "reddit:user:bob_ml": 0,
        "reddit:user:charlie_eng": 0,
        "reddit:user:david_dev": 0,
    }
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError, match="Invalid community_id '0' for node 'reddit:user:alice_data': must be a non-negative integer"):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=out_path,
            ranker=mock_ranker,
        )

    assert not out_path.exists()


def test_community_failure_does_not_replace_canonical_artifact_and_cleans_temp(
    tmp_path: Path, sample_reddit_graph_data: list[dict[str, Any]]
) -> None:
    """Tests that community failure preserves existing canonical artifact and leaves no temporary files."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    canonical_path = tmp_path / "canonical_results.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text("[]", encoding="utf-8")

    # Write preexisting canonical file
    sentinel_content = '{"sentinel": "untouched_canonical_artifact_data"}'
    canonical_path.write_text(sentinel_content, encoding="utf-8")

    mock_ranker = MagicMock()
    mock_ranker.detect_communities.return_value = {"missing_all_real_nodes": 0}
    mock_ranker.rank_influencers.return_value = []

    with pytest.raises(ValueError):
        run_graph_pipeline(
            reddit_input_path=r_path,
            youtube_input_path=y_path,
            output_path=canonical_path,
            ranker=mock_ranker,
        )

    # Canonical artifact must be completely untouched
    assert canonical_path.read_text(encoding="utf-8") == sentinel_content

    # No temporary files left behind
    tmp_files = list(tmp_path.glob(".tmp_*"))
    assert len(tmp_files) == 0


def test_neo4j_export_receives_complete_graph_nodes_not_only_top_influencers(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Tests that cypher_export contains batches for all graph nodes rather than only top influencers."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text(json.dumps(sample_youtube_graph_data), encoding="utf-8")

    # Run with top_n_influencers=1 to guarantee top_influencers count is smaller than total nodes
    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
        top_n_influencers=1,
    )

    all_nodes = result["nodes"]
    top_inf = result["top_influencers"]
    export = result["neo4j_export"]

    assert len(top_inf) == 1
    assert len(all_nodes) >= 4
    assert len(all_nodes) > len(top_inf)

    # Neo4j export must represent all nodes, not just the 1 top influencer
    assert export["total_users"] == len(all_nodes)
    exported_user_ids = [row["node_id"] for batch in export["user_batches"] for row in batch]
    assert len(exported_user_ids) == len(all_nodes)
    assert set(exported_user_ids) == {n["node_id"] for n in all_nodes}


def test_complete_graph_edges_remain_preserved_in_export(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Tests that all graph edges are fully preserved in graph artifact and Neo4j relationship batches."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text(json.dumps(sample_youtube_graph_data), encoding="utf-8")

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
    )

    edges = result["edges"]
    export = result["neo4j_export"]

    assert len(edges) > 0
    assert export["total_relationships"] == len(edges)

    exported_edges = [
        (row["source"], row["target"])
        for batch in export["relationship_batches"]
        for row in batch
    ]
    assert len(exported_edges) == len(edges)
    assert set(exported_edges) == {(e["source"], e["target"]) for e in edges}


def test_influencer_ranking_output_remains_separate_from_complete_nodes(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Tests that top_influencers ranking is retained separately from complete nodes array."""
    r_path = tmp_path / "reddit.json"
    y_path = tmp_path / "youtube.json"
    out_path = tmp_path / "out.json"

    r_path.write_text(json.dumps(sample_reddit_graph_data), encoding="utf-8")
    y_path.write_text(json.dumps(sample_youtube_graph_data), encoding="utf-8")

    result = run_graph_pipeline(
        reddit_input_path=r_path,
        youtube_input_path=y_path,
        output_path=out_path,
        top_n_influencers=2,
    )

    assert len(result["top_influencers"]) == 2
    assert len(result["nodes"]) >= 4

    # Every influencer is in nodes array and has matching community_id
    nodes_by_id = {n["node_id"]: n for n in result["nodes"]}
    for inf in result["top_influencers"]:
        assert inf["node_id"] in nodes_by_id
        assert inf["community_id"] == nodes_by_id[inf["node_id"]]["community_id"]


def test_representative_interaction_fixture_demonstrates_connected_graph(tmp_path: Path) -> None:
    """Verifies that the representative interaction fixture produces non-zero edges,
    meaningfully differentiated PageRank scores, Louvain community clusters, and valid Schema-v2 output.
    """
    fixture_path = Path("data/sample_reddit_interactions.json")
    assert fixture_path.exists(), "Interaction fixture data/sample_reddit_interactions.json must exist"

    # Also verify original sample fixture exists and remains untouched
    original_path = Path("data/sample_reddit_posts.json")
    assert original_path.exists(), "Original fixture data/sample_reddit_posts.json must remain intact"

    # Empty youtube input to isolate reddit interaction network
    empty_yt = tmp_path / "empty_yt.json"
    empty_yt.write_text("[]", encoding="utf-8")
    out_path = tmp_path / "graph_results.json"

    result = run_graph_pipeline(
        reddit_input_path=fixture_path,
        youtube_input_path=empty_yt,
        output_path=out_path,
    )

    # 1. Pipeline status and schema validity
    assert result["status"] == "success"
    is_valid, errors = GraphArtifactValidator.validate_graph_artifact(result)
    assert is_valid is True, f"Graph artifact validation failed: {errors}"

    # 2. Graph topology: nodes, non-zero edges, and density
    metrics = result["graph_metrics"]
    assert metrics["total_users"] == 10
    assert metrics["total_interactions"] == 19
    assert metrics["is_directed"] is True
    assert metrics["density"] > 0.0
    assert metrics["density"] == pytest.approx(0.2111, abs=1e-4)

    # 3. Directed edge checks and directionality
    edges = result["edges"]
    assert len(edges) == 19
    edge_pairs = {(e["source"], e["target"]) for e in edges}
    # Verify specific edge directions (source mentions target)
    assert ("reddit:user:clean_coder_dev", "reddit:user:pipeline_builder") in edge_pairs
    assert ("reddit:user:clean_coder_dev", "reddit:user:elastic_architect") in edge_pairs
    assert ("reddit:user:graph_analyst_pro", "reddit:user:pipeline_builder") in edge_pairs
    assert ("reddit:user:pipeline_builder", "reddit:user:clean_coder_dev") in edge_pairs

    # 4. PageRank differentiation
    influencers = result["top_influencers"]
    assert len(influencers) == 10
    pr_scores = [inf["pagerank"] for inf in influencers]
    assert len(set(pr_scores)) > 1, "PageRank scores must not be all identical"
    assert max(pr_scores) > min(pr_scores)
    # Top ranked influencers have highest PageRank and in-degree
    top_influencer = influencers[0]
    assert top_influencer["node_id"] in ("reddit:user:pipeline_builder", "reddit:user:graph_analyst_pro")
    assert top_influencer["in_degree_raw"] >= 5

    # 5. Community detection
    assert metrics["communities_count"] == 3
    community_ids = {inf["community_id"] for inf in influencers}
    assert len(community_ids) == 3

    # 6. Verify original sample fixture still produces zero edges
    out_zero = tmp_path / "zero_edges.json"
    result_zero = run_graph_pipeline(
        reddit_input_path=original_path,
        youtube_input_path=empty_yt,
        output_path=out_zero,
    )
    assert result_zero["graph_metrics"]["total_interactions"] == 0
    assert result_zero["graph_metrics"]["density"] == 0.0
    assert len(result_zero["edges"]) == 0



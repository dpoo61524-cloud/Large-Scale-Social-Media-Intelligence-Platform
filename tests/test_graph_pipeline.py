"""Integration and Orchestration Tests for Graph Pipeline (Company Step 5).

Verifies end-to-end execution of run_graph_pipeline:
- Ingestion of processed Reddit and YouTube records into directed interaction graph.
- NetworkX influencer ranking (PageRank, In-Degree, Betweenness, Communities).
- Parameterized Cypher batch generation for Neo4j.
- Non-destructive execution ensuring source raw and processed records are preserved.
- Output JSON schema validation for data/processed/graph_analysis_results.json.
"""

import json
from pathlib import Path
from typing import Any
import pytest

from src.graph.graph_pipeline import run_graph_pipeline


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
            "channel_title": "TechGuru_YT",
            "title_cleaned": "Graph Analytics Masterclass with @Alice_Data",
            "description_cleaned": "Featuring ideas from @Alice_Data and @Bob_ML.",
            "published_at": "2026-11-01T12:00:00Z",
            "collected_at": "2026-11-06T13:00:00+00:00",
        }
    ]


# ==============================================================================
# Pipeline Execution & Output Schema Validation
# ==============================================================================

def test_graph_pipeline_execution_and_schema(
    tmp_path: Path,
    sample_reddit_graph_data: list[dict[str, Any]],
    sample_youtube_graph_data: list[dict[str, Any]],
) -> None:
    """Verifies that run_graph_pipeline executes end-to-end and creates valid JSON schema."""
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

    # 1. Output exists and top-level status
    assert output_path.exists()
    assert result["status"] == "success"
    assert "generated_at" in result

    # 2. Graph metrics
    metrics = result["graph_metrics"]
    assert metrics["total_users"] >= 4  # Alice_Data, Bob_ML, Charlie_Eng, David_Dev, TechGuru_YT
    assert metrics["total_interactions"] >= 3
    assert metrics["is_directed"] is True
    assert "density" in metrics

    # 3. Top influencers structure
    influencers = result["top_influencers"]
    assert len(influencers) > 0
    top = influencers[0]
    assert "username" in top
    assert "influencer_score" in top
    assert "pagerank" in top
    assert "in_degree_centrality" in top
    assert "betweenness_centrality" in top
    assert "community_id" in top

    # Alice_Data is heavily mentioned by Bob_ML and TechGuru_YT -> should rank high
    usernames = [inf["username"] for inf in influencers]
    assert "Alice_Data" in usernames

    # 4. Neo4j export structure
    neo4j_export = result["neo4j_export"]
    assert "user_query" in neo4j_export
    assert "relationship_query" in neo4j_export
    assert neo4j_export["total_users"] > 0
    assert neo4j_export["total_relationships"] > 0


# ==============================================================================
# Non-Destructive Source Data Integrity
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


# ==============================================================================
# Missing Inputs and Workspace File Runs
# ==============================================================================

def test_graph_pipeline_handles_missing_inputs(tmp_path: Path) -> None:
    """Verifies pipeline handles non-existent input files gracefully."""
    output_path = tmp_path / "empty_graph_output.json"

    result = run_graph_pipeline(
        reddit_input_path=tmp_path / "non_existent_reddit.json",
        youtube_input_path=tmp_path / "non_existent_youtube.json",
        output_path=output_path,
    )

    assert result["status"] == "success"
    assert result["graph_metrics"]["total_users"] == 0
    assert result["graph_metrics"]["total_interactions"] == 0
    assert output_path.exists()


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

    assert result["status"] == "success"
    assert result["graph_metrics"]["total_users"] > 0
    assert output_path.exists()
    assert "top_influencers" in result
    assert "neo4j_export" in result

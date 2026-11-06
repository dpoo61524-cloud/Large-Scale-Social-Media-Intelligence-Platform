"""Graph Analysis Pipeline (Company Step 5).

Orchestrates user-interaction graph construction, NetworkX influencer ranking (PageRank,
Centrality, Community Detection), and Neo4j Cypher batch preparation.
Persists structured graph analytics non-destructively to data/processed/graph_analysis_results.json.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional
import networkx as nx

from src.graph.influencer_ranker import InfluencerRanker
from src.graph.interaction_builder import InteractionGraphBuilder
from src.graph.neo4j_connector import Neo4jGraphManager

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_INPUT = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_INPUT = Path("data/processed/youtube_videos_processed.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/graph_analysis_results.json")


def run_graph_pipeline(
    reddit_input_path: Optional[Path | str] = None,
    youtube_input_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    top_n_influencers: int = 50,
    builder: Optional[InteractionGraphBuilder] = None,
    ranker: Optional[InfluencerRanker] = None,
    neo4j_manager: Optional[Neo4jGraphManager] = None,
) -> dict[str, Any]:
    """Runs the interaction graph construction and influencer ranking pipeline.

    Args:
        reddit_input_path: Path to processed Reddit JSON.
        youtube_input_path: Path to processed YouTube JSON.
        output_path: Destination path for graph analysis JSON results.
        top_n_influencers: Number of top influencers to rank and output.
        builder: Optional custom or mocked InteractionGraphBuilder.
        ranker: Optional custom or mocked InfluencerRanker.
        neo4j_manager: Optional custom or mocked Neo4jGraphManager.

    Returns:
        Structured dictionary containing graph metrics, top influencers, communities,
        and Neo4j export batches.
    """
    reddit_path = Path(reddit_input_path) if reddit_input_path else DEFAULT_REDDIT_INPUT
    youtube_path = Path(youtube_input_path) if youtube_input_path else DEFAULT_YOUTUBE_INPUT
    dest_path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH

    if builder is None:
        builder = InteractionGraphBuilder()
    if ranker is None:
        ranker = InfluencerRanker()
    if neo4j_manager is None:
        neo4j_manager = Neo4jGraphManager()

    # Load Reddit processed records
    reddit_records: list[dict[str, Any]] = []
    if reddit_path.exists():
        try:
            with open(reddit_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    reddit_records = loaded
        except Exception as exc:
            logger.error("Failed to read Reddit input %s: %s", reddit_path, exc)

    # Load YouTube processed records
    youtube_records: list[dict[str, Any]] = []
    if youtube_path.exists():
        try:
            with open(youtube_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    youtube_records = loaded
        except Exception as exc:
            logger.error("Failed to read YouTube input %s: %s", youtube_path, exc)

    now_iso = datetime.now(timezone.utc).isoformat()

    # Build interaction graph
    user_graph = builder.build_interaction_graph(reddit_records, youtube_records)
    num_nodes = user_graph.number_of_nodes()
    num_edges = user_graph.number_of_edges()

    # Rank influencers and detect communities
    ranked_influencers = ranker.rank_influencers(user_graph, top_n=top_n_influencers)
    communities = ranker.detect_communities(user_graph)

    # Community sizes
    community_counts: dict[int, int] = {}
    for comm_id in communities.values():
        community_counts[comm_id] = community_counts.get(comm_id, 0) + 1

    # Graph density
    density = round(float(nx.density(user_graph)), 6) if num_nodes > 1 else 0.0

    # Extract nodes and edges for Neo4j export
    edges_list = builder.get_edges()
    cypher_export = neo4j_manager.generate_cypher_export(
        nodes=ranked_influencers if ranked_influencers else builder.get_nodes(),
        edges=edges_list,
        batch_size=500,
    )

    result_payload: dict[str, Any] = {
        "status": "success",
        "generated_at": now_iso,
        "graph_metrics": {
            "total_users": num_nodes,
            "total_interactions": num_edges,
            "density": density,
            "communities_count": len(community_counts),
            "is_directed": True,
        },
        "top_influencers": ranked_influencers,
        "community_distributions": {str(k): v for k, v in community_counts.items()},
        "neo4j_export": {
            "user_query": cypher_export["user_query"],
            "relationship_query": cypher_export["relationship_query"],
            "total_user_batches": cypher_export["total_user_batches"],
            "total_relationship_batches": cypher_export["total_relationship_batches"],
            "total_users": cypher_export["total_users"],
            "total_relationships": cypher_export["total_relationships"],
        },
        "edges_sample": edges_list[:20],
    }

    # Persist output
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    logger.info(
        "Graph analysis pipeline completed. %d users, %d interactions persisted to %s.",
        num_nodes,
        num_edges,
        dest_path,
    )

    return result_payload

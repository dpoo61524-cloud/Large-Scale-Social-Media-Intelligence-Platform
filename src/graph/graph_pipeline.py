"""Graph Analysis Pipeline (Stage 7).

Orchestrates user-interaction graph construction, NetworkX influencer ranking (PageRank,
Centrality, Community Detection), and Neo4j Cypher batch preparation.
Integrates strict Schema-v2 validation and safe atomic artifact promotion.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional
import uuid
import networkx as nx

from src.graph.graph_validator import GraphArtifactValidator
from src.graph.influencer_ranker import InfluencerRanker
from src.graph.interaction_builder import InteractionGraphBuilder
from src.graph.neo4j_connector import Neo4jGraphManager

logger = logging.getLogger(__name__)

DEFAULT_REDDIT_INPUT = Path("data/processed/reddit_posts_processed.json")
DEFAULT_YOUTUBE_INPUT = Path("data/processed/youtube_videos_processed.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/graph_analysis_results.json")

YOUTUBE_CHANNEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _is_valid_reddit_record(record: Any) -> bool:
    """Validates that a record matches the upstream Reddit preprocessing schema contract.

    Must be a dictionary with non-empty string 'post_id' and 'author'.
    """
    if not isinstance(record, dict):
        return False
    post_id = record.get("post_id")
    if not isinstance(post_id, str) or not post_id.strip():
        return False
    author = record.get("author")
    if not isinstance(author, str) or not author.strip():
        return False
    return True


def _is_valid_youtube_record(record: Any) -> bool:
    """Validates that a record matches the YouTube graph identity contract.

    Requires non-empty string 'video_id' and authoritative non-empty string 'channel_id'
    matching ^[A-Za-z0-9_-]+$. 'channel_title' is optional display metadata only.
    """
    if not isinstance(record, dict):
        return False
    video_id = record.get("video_id")
    if not isinstance(video_id, str) or not video_id.strip():
        return False
    channel_id = record.get("channel_id")
    if not isinstance(channel_id, str):
        return False
    ch_id_clean = channel_id.strip()
    if not ch_id_clean or not YOUTUBE_CHANNEL_ID_PATTERN.match(ch_id_clean):
        return False
    return True


def load_and_classify_source(
    file_path: Path,
    source_type: Optional[str] = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Loads and classifies an input social media processed file against schema contracts.

    Evaluates file existence, format validity, and data integrity to classify
    into exactly one of four mutual outcomes:
    - 'loaded': file exists, is valid JSON list, all items conform to the source
      schema contract (Reddit or YouTube), length >= 1.
    - 'empty': file exists, is valid JSON list, length == 0.
    - 'missing': file does not exist on disk.
    - 'corrupted': file exists but is unreadable, invalid JSON, not a list,
      or contains malformed non-dict items or records violating schema contracts.

    Args:
        file_path: Path to the input JSON file.
        source_type: Required source discriminator ('reddit' or 'youtube').
            Must be explicitly provided as exact lowercase string 'reddit' or 'youtube'.
            Omission (None) or unsupported types raise ValueError.

    Returns:
        tuple of (status: str, records: list[dict[str, Any]], count: int).

    Raises:
        ValueError: If source_type is omitted (None) or not in ('reddit', 'youtube').
    """
    if source_type not in ("reddit", "youtube"):
        raise ValueError(
            f"Invalid source_type: {source_type!r}. Must be explicitly 'reddit' or 'youtube'."
        )

    if not file_path.exists():
        logger.warning("Input source file does not exist: %s", file_path)
        return "missing", [], 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("Failed to read/parse input source %s: %s", file_path, exc)
        return "corrupted", [], 0

    if not isinstance(data, list):
        logger.error("Expected JSON list in %s, got %s", file_path, type(data).__name__)
        return "corrupted", [], 0

    if len(data) == 0:
        return "empty", [], 0

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logger.error(
                "Malformed record at index %d in %s: expected dict, got %s",
                idx,
                file_path,
                type(item).__name__,
            )
            return "corrupted", [], 0

        if source_type == "reddit":
            if not _is_valid_reddit_record(item):
                logger.error(
                    "Malformed Reddit record at index %d in %s: missing or invalid post_id/author",
                    idx,
                    file_path,
                )
                return "corrupted", [], 0
        else:  # source_type == "youtube"
            if not _is_valid_youtube_record(item):
                logger.error(
                    "Malformed YouTube record at index %d in %s: missing or invalid video_id/channel_id",
                    idx,
                    file_path,
                )
                return "corrupted", [], 0

    return "loaded", data, len(data)


def _promote_artifact_atomically(payload: dict[str, Any], dest_path: Path) -> None:
    """Atomically serializes, validates, and promotes a graph artifact.

    Guarantees that readers of dest_path will never observe a partially written
    or corrupt file. If any error occurs prior to or during replacement, any
    existing canonical file is untouched and temporary files are cleaned up.

    Args:
        payload: Fully assembled and in-memory validated schema-v2 payload dictionary.
        dest_path: Canonical destination path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    unique_id = uuid.uuid4().hex
    temp_path = dest_path.parent / f".tmp_{dest_path.name}_{unique_id}"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except AttributeError:
                # Handle test doubles or file-like objects that lack fileno()
                pass

        # Read-back verification before replacement
        with open(temp_path, "r", encoding="utf-8") as rf:
            read_back_payload = json.load(rf)

        rb_valid, rb_errors = GraphArtifactValidator.validate_graph_artifact(read_back_payload)
        if not rb_valid:
            raise ValueError(f"Read-back graph artifact validation failed: {rb_errors}")

        # Atomic replacement
        os.replace(temp_path, dest_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as exc:
                logger.debug("Failed to clean up temporary file %s: %s", temp_path, exc)


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
        Structured Schema-v2 dictionary containing graph metrics, complete nodes,
        complete edges, top influencers, communities, provenance, and Neo4j export batches.
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

    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Load and classify input sources
    reddit_status, reddit_records, reddit_count = load_and_classify_source(
        reddit_path, source_type="reddit"
    )
    youtube_status, youtube_records, youtube_count = load_and_classify_source(
        youtube_path, source_type="youtube"
    )

    provenance: dict[str, Any] = {
        "reddit_input_file": str(reddit_path),
        "reddit_status": reddit_status,
        "reddit_records_count": reddit_count,
        "youtube_input_file": str(youtube_path),
        "youtube_status": youtube_status,
        "youtube_records_count": youtube_count,
    }

    # 2. Source-level failure check
    valid_statuses = {"loaded", "empty"}
    reddit_ok = reddit_status in valid_statuses
    youtube_ok = youtube_status in valid_statuses

    user_graph = nx.DiGraph()
    ranked_influencers: list[dict[str, Any]] = []
    communities: dict[str, int] = {}
    community_counts: dict[int, int] = {}
    density: float = 0.0
    nodes_list: list[dict[str, Any]] = []
    edges_list: list[dict[str, Any]] = []

    # If both sources failed, skip graph construction and treat as error
    if not reddit_ok and not youtube_ok:
        status = "error"
    else:
        # Build interaction graph
        try:
            user_graph = builder.build_interaction_graph(reddit_records, youtube_records)
            num_nodes = user_graph.number_of_nodes()
            num_edges = user_graph.number_of_edges()

            ranked_influencers = ranker.rank_influencers(user_graph, top_n=top_n_influencers)
            communities = ranker.detect_communities(user_graph)

            for comm_id in communities.values():
                community_counts[comm_id] = community_counts.get(comm_id, 0) + 1

            density = round(float(nx.density(user_graph)), 6) if num_nodes > 1 else 0.0

            nodes_list = builder.get_nodes()
            edges_list = builder.get_edges()

            # Exact community mapping coverage check
            node_ids = {node["node_id"] for node in nodes_list}
            if set(communities.keys()) != node_ids:
                raise ValueError(
                    f"Community assignments do not match graph nodes: "
                    f"missing={sorted(list(node_ids - set(communities.keys())))}, "
                    f"unexpected={sorted(list(set(communities.keys()) - node_ids))}"
                )

            # Validate every community ID
            for node in nodes_list:
                cid = communities[node["node_id"]]
                if isinstance(cid, bool) or not isinstance(cid, int) or cid < 0:
                    raise ValueError(
                        f"Invalid community_id {cid!r} for node "
                        f"{node['node_id']!r}: must be a non-negative integer"
                    )
                node["community_id"] = cid

            # Derive artifact status
            if reddit_ok and youtube_ok:
                status = "empty" if num_nodes == 0 else "success"
            else:
                status = "partial"

        except ValueError:
            # Re-raise explicit validation errors to fail closed without artifact promotion
            raise
        except Exception as exc:
            logger.error("Unexpected execution failure during graph construction/ranking: %s", exc)
            status = "error"

    num_nodes = len(nodes_list)
    num_edges = len(edges_list)

    # Sort nodes and edges deterministically
    nodes_list.sort(key=lambda x: str(x.get("node_id") or ""))
    edges_list.sort(key=lambda x: (str(x.get("source") or ""), str(x.get("target") or "")))

    # Neo4j Cypher batch export with complete nodes list
    try:
        cypher_export = neo4j_manager.generate_cypher_export(
            nodes=nodes_list,
            edges=edges_list,
            batch_size=500,
            top_influencers=ranked_influencers,
        )
    except Exception as exc:
        logger.error("Failed to generate Cypher export: %s", exc)
        cypher_export = {
            "user_query": "",
            "relationship_query": "",
            "total_user_batches": 0,
            "total_relationship_batches": 0,
            "total_users": 0,
            "total_relationships": 0,
        }

    community_distributions = {str(k): v for k, v in community_counts.items()}

    result_payload: dict[str, Any] = {
        "schema_version": "2.0",
        "status": status,
        "generated_at": now_iso,
        "provenance": provenance,
        "graph_metrics": {
            "total_users": num_nodes,
            "total_interactions": num_edges,
            "density": density,
            "communities_count": len(community_counts),
            "is_directed": True,
        },
        "nodes": nodes_list,
        "edges": edges_list,
        "top_influencers": ranked_influencers,
        "community_distributions": community_distributions,
        "neo4j_export": cypher_export,
    }

    # 3. Fail-closed promotion gate
    # Never promote an artifact with status='error'
    if status == "error":
        logger.warning(
            "Graph analysis produced diagnostic error artifact (status='error'). "
            "Artifact will NOT be promoted to disk.",
        )
        return result_payload

    # Pre-promotion validation
    is_valid, validation_errors = GraphArtifactValidator.validate_graph_artifact(result_payload)
    if not is_valid:
        logger.error(
            "Pre-promotion validation failed (%d errors): %s",
            len(validation_errors),
            validation_errors,
        )
        raise ValueError(f"Graph artifact validation failed: {validation_errors}")

    # 4. Atomic artifact promotion
    _promote_artifact_atomically(result_payload, dest_path)

    logger.info(
        "Graph analysis pipeline completed successfully. %d users, %d interactions promoted to %s.",
        num_nodes,
        num_edges,
        dest_path,
    )

    return result_payload

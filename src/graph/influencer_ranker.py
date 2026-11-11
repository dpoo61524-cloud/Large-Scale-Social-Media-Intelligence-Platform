"""Influencer Ranker & Network Analysis (Company Step 5).

Ranks network influencers across social media interaction graphs using PageRank,
In-Degree Centrality, Betweenness Centrality, and Louvain/Modularity Community Detection.
Computes a normalized composite influencer score.
"""

import logging
from typing import Any
import networkx as nx

logger = logging.getLogger(__name__)


class InfluencerRanker:
    """Computes network centrality, community partitions, and composite influencer rankings."""

    def __init__(
        self,
        pagerank_alpha: float = 0.85,
        pagerank_max_iter: int = 200,
        pagerank_tol: float = 1e-6,
    ) -> None:
        """Initializes the InfluencerRanker.

        Args:
            pagerank_alpha: PageRank damping factor.
            pagerank_max_iter: PageRank maximum iterations for power-iteration convergence.
            pagerank_tol: PageRank convergence tolerance.
        """
        self.pagerank_alpha = pagerank_alpha
        self.pagerank_max_iter = pagerank_max_iter
        self.pagerank_tol = pagerank_tol

    @staticmethod
    def _normalize_dict_values(raw_scores: dict[str, float]) -> dict[str, float]:
        """Performs min-max normalization on dictionary float values into [0.0, 1.0]."""
        if not raw_scores:
            return {}

        vals = list(raw_scores.values())
        min_v = min(vals)
        max_v = max(vals)

        if max_v - min_v == 0.0:
            # If all values are identical and > 0, assign 1.0; if 0, assign 0.0
            return {k: 1.0 if max_v > 0 else 0.0 for k in raw_scores}

        return {
            k: round((v - min_v) / (max_v - min_v), 4)
            for k, v in raw_scores.items()
        }

    def detect_communities(self, graph: nx.DiGraph) -> dict[str, int]:
        """Detects network communities with canonical deterministic labeling.

        Uses Louvain community detection with fixed random seed 42 on the undirected projection.
        Communities are deterministically sorted by their minimum canonical node_id, and assigned
        canonical ascending integer IDs (0, 1, 2, ...).

        Args:
            graph: Directed NetworkX graph.

        Returns:
            Dictionary mapping node_id to community_id (integer).
        """
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected()
        try:
            # Attempt Louvain with fixed seed=42 if available, else greedy modularity
            if hasattr(nx.community, "louvain_communities"):
                communities = nx.community.louvain_communities(undirected, weight="weight", seed=42)
            else:
                communities = list(nx.community.greedy_modularity_communities(undirected))
        except Exception as exc:
            logger.debug("Advanced community detection failed, using connected components: %s", exc)
            communities = list(nx.connected_components(undirected))

        # 1. Sort node IDs within each detected community
        sorted_communities = [sorted(list(c), key=str) for c in communities if c]
        # 2. Sort communities by their smallest member node ID
        sorted_communities.sort(key=lambda members: str(members[0]))

        # 3. Assign canonical community IDs in that order
        node_community_map: dict[str, int] = {}
        for comm_id, members in enumerate(sorted_communities):
            for member in members:
                node_community_map[str(member)] = comm_id

        return node_community_map

    def rank_influencers(
        self,
        graph: nx.DiGraph,
        top_n: int = 50,
    ) -> list[dict[str, Any]]:
        """Computes PageRank, In-Degree, Betweenness, and composite influencer scores.

        Composite Score Formula:
            Score(u) = 0.50 * PageRank_norm(u) + 0.30 * InDegree_norm(u) + 0.20 * Betweenness_norm(u)

        Deterministic Ranking Order:
            1. Higher influencer_score first (-influencer_score)
            2. Higher PageRank first (-pagerank)
            3. Lexicographically smaller canonical node_id first (node_id)

        Args:
            graph: Directed NetworkX interaction graph.
            top_n: Number of top influencers to return.

        Returns:
            List of influencer dictionaries sorted deterministically.
        """
        if graph.number_of_nodes() == 0:
            return []

        # Sort node list deterministically for stable iteration
        nodes = sorted(list(graph.nodes()), key=str)

        # 1. PageRank
        try:
            raw_pagerank = nx.pagerank(
                graph,
                alpha=self.pagerank_alpha,
                max_iter=self.pagerank_max_iter,
                tol=self.pagerank_tol,
                weight="weight",
            )
        except Exception as exc:
            logger.warning("Weighted PageRank power iteration failed: %s. Using uniform fallback.", exc)
            raw_pagerank = {n: 1.0 / len(nodes) for n in nodes}

        # 2. In-Degree Centrality (Popularity / incoming citations)
        try:
            raw_in_degree = nx.in_degree_centrality(graph)
        except Exception:
            raw_in_degree = {n: 0.0 for n in nodes}

        # 3. Betweenness Centrality (Information brokerage)
        try:
            raw_betweenness = nx.betweenness_centrality(graph, weight="weight")
        except Exception:
            raw_betweenness = {n: 0.0 for n in nodes}

        # 4. Community partitions (canonical labeling)
        communities = self.detect_communities(graph)

        # 5. Normalize metrics
        norm_pagerank = self._normalize_dict_values(raw_pagerank)
        norm_in_degree = self._normalize_dict_values(raw_in_degree)
        norm_betweenness = self._normalize_dict_values(raw_betweenness)

        # 6. Compute composite score and package identity-aware metadata
        ranked: list[dict[str, Any]] = []
        for n in nodes:
            node_attrs = dict(graph.nodes[n]) if graph.has_node(n) else {}

            canonical_node_id = str(node_attrs.get("node_id") or n)
            platform = str(node_attrs.get("platform") or "unknown")

            # Determine identity_type
            if "identity_type" in node_attrs:
                identity_type = node_attrs["identity_type"]
            elif platform == "reddit" or canonical_node_id.startswith("reddit:"):
                identity_type = "account"
            elif canonical_node_id.startswith("youtube:mention:"):
                identity_type = "mention"
            elif platform == "youtube" or canonical_node_id.startswith("youtube:"):
                identity_type = "creator"
            else:
                identity_type = "account"

            # Determine display username from node metadata (never use raw namespaced node ID)
            display_username = node_attrs.get("username")
            if not display_username:
                if not (canonical_node_id.startswith("reddit:") or canonical_node_id.startswith("youtube:")):
                    display_username = str(n)
                else:
                    parts = canonical_node_id.split(":")
                    display_username = parts[-1]

            # Author flag and activity count
            is_author = bool(node_attrs.get("is_author", True if identity_type == "creator" else False))
            activity_count = int(node_attrs.get("activity_count", 1 if is_author else 0))

            # Channel metadata (preserve legitimate None values; do not fabricate)
            channel_id = node_attrs.get("channel_id")
            channel_title = node_attrs.get("channel_title")
            if identity_type == "mention":
                channel_id = None
                channel_title = None
                is_author = False
                activity_count = 0

            pr_val = norm_pagerank.get(n, 0.0)
            in_deg_val = norm_in_degree.get(n, 0.0)
            btw_val = norm_betweenness.get(n, 0.0)

            composite_score = round(
                0.50 * pr_val + 0.30 * in_deg_val + 0.20 * btw_val,
                4,
            )

            ranked.append({
                "node_id": canonical_node_id,
                "platform": platform,
                "username": display_username,
                "identity_type": identity_type,
                "is_author": is_author,
                "channel_id": channel_id,
                "channel_title": channel_title,
                "activity_count": activity_count,
                "influencer_score": composite_score,
                "pagerank": round(raw_pagerank.get(n, 0.0), 6),
                "pagerank_norm": pr_val,
                "in_degree_centrality": round(raw_in_degree.get(n, 0.0), 4),
                "in_degree_norm": in_deg_val,
                "betweenness_centrality": round(raw_betweenness.get(n, 0.0), 4),
                "betweenness_norm": btw_val,
                "community_id": communities.get(n, 0),
                "in_degree_raw": graph.in_degree(n),
                "out_degree_raw": graph.out_degree(n),
            })

        # Deterministic sorting:
        # 1. Higher influencer_score first (-x["influencer_score"])
        # 2. Higher PageRank first (-x["pagerank"])
        # 3. Lexicographically smaller canonical node_id first (x["node_id"])
        ranked.sort(key=lambda x: (-x["influencer_score"], -x["pagerank"], str(x["node_id"])))

        return ranked[:top_n]

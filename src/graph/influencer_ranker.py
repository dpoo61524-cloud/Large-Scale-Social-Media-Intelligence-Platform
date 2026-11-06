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
        """Detects network communities using greedy modularity on the undirected projection.

        Args:
            graph: Directed NetworkX graph.

        Returns:
            Dictionary mapping node_id to community_id (integer).
        """
        if graph.number_of_nodes() == 0:
            return {}

        undirected = graph.to_undirected()
        try:
            # Attempt Louvain if available, else greedy modularity
            if hasattr(nx.community, "louvain_communities"):
                communities = nx.community.louvain_communities(undirected, seed=42)
            else:
                communities = list(nx.community.greedy_modularity_communities(undirected))
        except Exception as exc:
            logger.debug("Advanced community detection failed, using connected components: %s", exc)
            communities = list(nx.connected_components(undirected))

        node_community_map: dict[str, int] = {}
        for comm_id, members in enumerate(communities):
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

        Args:
            graph: Directed NetworkX interaction graph.
            top_n: Number of top influencers to return.

        Returns:
            List of influencer dictionaries sorted by influencer_score descending.
        """
        if graph.number_of_nodes() == 0:
            return []

        nodes = list(graph.nodes())

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

        # 4. Community partitions
        communities = self.detect_communities(graph)

        # 5. Normalize metrics
        norm_pagerank = self._normalize_dict_values(raw_pagerank)
        norm_in_degree = self._normalize_dict_values(raw_in_degree)
        norm_betweenness = self._normalize_dict_values(raw_betweenness)

        # 6. Compute composite score
        ranked: list[dict[str, Any]] = []
        for n in nodes:
            pr_val = norm_pagerank.get(n, 0.0)
            in_deg_val = norm_in_degree.get(n, 0.0)
            btw_val = norm_betweenness.get(n, 0.0)

            composite_score = round(
                0.50 * pr_val + 0.30 * in_deg_val + 0.20 * btw_val,
                4,
            )

            ranked.append({
                "username": n,
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
                "platform": graph.nodes[n].get("platform", "unknown"),
            })

        # Sort by composite influencer score descending, then by raw PageRank
        ranked.sort(key=lambda x: (x["influencer_score"], x["pagerank"]), reverse=True)

        return ranked[:top_n]

"""Interaction Graph Builder (Company Step 5).

Extracts directed user-content and user-user interactions from Reddit and YouTube records.
Filters anonymous/deleted entities and aggregates interaction frequencies into a weighted
NetworkX directed graph (DiGraph).
"""

from collections import defaultdict
import logging
import re
from typing import Any, Optional
import networkx as nx

logger = logging.getLogger(__name__)

# Anonymous or automated accounts to exclude from influencer graph
ANONYMOUS_USERS = {
    "[deleted]",
    "[removed]",
    "automoderator",
    "deleted",
    "none",
    "null",
    "",
}

# Regex to extract user mentions (u/username on Reddit, @username on YouTube/socials)
REDDIT_USER_MENTION_PATTERN = re.compile(r"\bu/([A-Za-z0-9_-]+)\b", re.IGNORECASE)
GENERIC_USER_MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_.-]+)\b")


class InteractionGraphBuilder:
    """Constructs directed, weighted interaction graphs from social media records."""

    def __init__(self, exclude_anonymous: bool = True) -> None:
        """Initializes the InteractionGraphBuilder.

        Args:
            exclude_anonymous: Whether to filter out deleted/anonymous/bot usernames.
        """
        self.exclude_anonymous = exclude_anonymous
        self.user_graph: nx.DiGraph = nx.DiGraph()
        self.full_graph: nx.DiGraph = nx.DiGraph()

    @classmethod
    def is_valid_user(cls, username: Any) -> bool:
        """Checks if a username represents a valid, non-anonymous individual.

        Args:
            username: Username string.

        Returns:
            True if username is valid, False otherwise.
        """
        if not isinstance(username, str):
            return False
        clean = username.strip().lower()
        return bool(clean and clean not in ANONYMOUS_USERS)

    @classmethod
    def extract_mentions(cls, text: str) -> list[str]:
        """Extracts mentioned usernames from text content.

        Args:
            text: Input text string.

        Returns:
            List of mentioned usernames.
        """
        if not isinstance(text, str) or not text:
            return []

        mentions: list[str] = []
        for m in REDDIT_USER_MENTION_PATTERN.findall(text):
            if cls.is_valid_user(m):
                mentions.append(m)

        for m in GENERIC_USER_MENTION_PATTERN.findall(text):
            if cls.is_valid_user(m):
                mentions.append(m)

        return list(dict.fromkeys(mentions))  # Deduplicate preserving order

    def build_interaction_graph(
        self,
        reddit_records: Optional[list[dict[str, Any]]] = None,
        youtube_records: Optional[list[dict[str, Any]]] = None,
    ) -> nx.DiGraph:
        """Constructs a directed weighted user interaction graph from records.

        Edge (u1 -> u2) indicates u1 interacted with u2 (e.g. replied, mentioned, or co-discussed).
        Edge weight w represents total interaction volume.

        Args:
            reddit_records: Processed Reddit post/comment dictionaries.
            youtube_records: Processed YouTube video/comment dictionaries.

        Returns:
            NetworkX DiGraph of user interactions.
        """
        self.user_graph.clear()
        self.full_graph.clear()

        reddit_list = reddit_records or []
        youtube_list = youtube_records or []

        # Track user-to-user interactions: (source, target) -> {weight, interaction_types, platforms}
        user_edges: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"weight": 0, "types": set(), "platforms": set()}
        )

        # 1. Process Reddit records
        for post in reddit_list:
            if not isinstance(post, dict):
                continue

            author = post.get("author")
            post_id = post.get("post_id", "unknown_post")
            text_content = f"{post.get('title_cleaned', '')} {post.get('text_cleaned', '')} {post.get('title', '')} {post.get('text', '')}"

            if self.is_valid_user(author):
                author_clean = str(author).strip()
                self._add_or_update_node(self.user_graph, author_clean, node_type="user", platform="reddit")
                self._add_or_update_node(self.full_graph, author_clean, node_type="user", platform="reddit")
                self._add_or_update_node(self.full_graph, post_id, node_type="content", platform="reddit", title=post.get("title_cleaned") or post.get("title"))

                # Author -> Content edge
                self.full_graph.add_edge(author_clean, post_id, relationship="POSTED", weight=1, platform="reddit")

                # Extract mentions from post text: author -> mentioned_user
                mentions = self.extract_mentions(text_content)
                for mentioned in mentions:
                    if mentioned.lower() != author_clean.lower() and self.is_valid_user(mentioned):
                        self._add_or_update_node(self.user_graph, mentioned, node_type="user", platform="reddit")
                        user_edges[(author_clean, mentioned)]["weight"] += 1
                        user_edges[(author_clean, mentioned)]["types"].add("MENTIONED")
                        user_edges[(author_clean, mentioned)]["platforms"].add("reddit")

        # 2. Process YouTube records
        for video in youtube_list:
            if not isinstance(video, dict):
                continue

            channel_title = video.get("channel_title") or video.get("channel_id")
            video_id = video.get("video_id", "unknown_video")
            desc_content = f"{video.get('title_cleaned', '')} {video.get('description_cleaned', '')}"

            if self.is_valid_user(channel_title):
                creator_clean = str(channel_title).strip()
                self._add_or_update_node(self.user_graph, creator_clean, node_type="user", platform="youtube")
                self._add_or_update_node(self.full_graph, creator_clean, node_type="user", platform="youtube")
                self._add_or_update_node(self.full_graph, video_id, node_type="content", platform="youtube", title=video.get("title_cleaned") or video.get("title"))

                self.full_graph.add_edge(creator_clean, video_id, relationship="POSTED", weight=1, platform="youtube")

                # Mentions in video title / description
                mentions = self.extract_mentions(desc_content)
                for mentioned in mentions:
                    if mentioned.lower() != creator_clean.lower() and self.is_valid_user(mentioned):
                        self._add_or_update_node(self.user_graph, mentioned, node_type="user", platform="youtube")
                        user_edges[(creator_clean, mentioned)]["weight"] += 1
                        user_edges[(creator_clean, mentioned)]["types"].add("MENTIONED")
                        user_edges[(creator_clean, mentioned)]["platforms"].add("youtube")

        # Populate user graph edges
        for (src, tgt), data in user_edges.items():
            self.user_graph.add_edge(
                src,
                tgt,
                weight=data["weight"],
                interaction_types=list(data["types"]),
                platforms=list(data["platforms"]),
            )
            self.full_graph.add_edge(
                src,
                tgt,
                relationship="INTERACTED_WITH",
                weight=data["weight"],
                interaction_types=list(data["types"]),
                platforms=list(data["platforms"]),
            )

        logger.info(
            "Interaction graph constructed: %d users, %d directed edges.",
            self.user_graph.number_of_nodes(),
            self.user_graph.number_of_edges(),
        )

        return self.user_graph

    @staticmethod
    def _add_or_update_node(graph: nx.DiGraph, node_id: str, **attrs: Any) -> None:
        """Adds a node or updates its activity count if already present."""
        if graph.has_node(node_id):
            graph.nodes[node_id]["activity_count"] = graph.nodes[node_id].get("activity_count", 1) + 1
        else:
            graph.add_node(node_id, activity_count=1, **attrs)

    def get_nodes(self) -> list[dict[str, Any]]:
        """Returns structured list of user graph nodes."""
        nodes: list[dict[str, Any]] = []
        for node_id, attrs in self.user_graph.nodes(data=True):
            entry = dict(attrs)
            entry["id"] = node_id
            nodes.append(entry)
        return nodes

    def get_edges(self) -> list[dict[str, Any]]:
        """Returns structured list of user graph edges."""
        edges: list[dict[str, Any]] = []
        for src, tgt, attrs in self.user_graph.edges(data=True):
            entry = dict(attrs)
            entry["source"] = src
            entry["target"] = tgt
            edges.append(entry)
        return edges

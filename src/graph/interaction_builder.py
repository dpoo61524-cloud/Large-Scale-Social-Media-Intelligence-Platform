"""Interaction Graph Builder (Company Step 5).

Extracts directed user-content and user-user interactions from Reddit and YouTube records.
Filters anonymous/deleted entities and aggregates interaction frequencies into a weighted
NetworkX directed graph (DiGraph) using platform-namespaced identities.
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

    @staticmethod
    def make_reddit_user_id(username: str) -> str:
        """Constructs canonical Reddit account node identifier."""
        return f"reddit:user:{username.strip().lower()}"

    @staticmethod
    def make_youtube_creator_id(channel_id: str) -> str:
        """Constructs canonical YouTube creator node identifier."""
        return f"youtube:channel:{channel_id.strip()}"

    @staticmethod
    def make_youtube_mention_id(handle: str) -> str:
        """Constructs canonical YouTube mention-only node identifier."""
        return f"youtube:mention:{handle.strip().lower()}"

    @staticmethod
    def _resolve_deterministic_username(existing: Optional[str], incoming: Optional[str]) -> str:
        """Selects a deterministic display name independent of encounter order.

        Uses lexicographical minimum so that regardless of encounter order,
        the exact same canonical casing is selected across records.
        """
        if not existing:
            return incoming or ""
        if not incoming:
            return existing
        return min(existing, incoming)

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

        # Track user-to-user interactions: (source_id, target_id) -> {weight, types, platforms}
        user_edges: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"weight": 0, "types": set(), "platforms": set()}
        )

        # 1. Process Reddit records
        for post in reddit_list:
            if not isinstance(post, dict):
                continue

            author = post.get("author")
            if not self.is_valid_user(author):
                continue

            author_clean = str(author).strip()
            author_node_id = self.make_reddit_user_id(author_clean)
            post_id = str(post.get("post_id", "unknown_post"))
            text_content = f"{post.get('title_cleaned', '')} {post.get('text_cleaned', '')} {post.get('title', '')} {post.get('text', '')}"

            # Add or update Reddit author node
            if self.user_graph.has_node(author_node_id):
                self.user_graph.nodes[author_node_id]["is_author"] = True
                self.user_graph.nodes[author_node_id]["activity_count"] += 1
                existing_username = self.user_graph.nodes[author_node_id].get("username")
                self.user_graph.nodes[author_node_id]["username"] = self._resolve_deterministic_username(
                    existing_username, author_clean
                )
            else:
                self.user_graph.add_node(
                    author_node_id,
                    node_id=author_node_id,
                    platform="reddit",
                    identity_type="account",
                    is_author=True,
                    username=author_clean,
                    channel_id=None,
                    channel_title=None,
                    activity_count=1,
                )

            # Full graph content edge
            self.full_graph.add_node(
                author_node_id,
                node_id=author_node_id,
                node_type="user",
                platform="reddit",
                identity_type="account",
                username=self.user_graph.nodes[author_node_id]["username"],
            )
            self.full_graph.add_node(
                post_id,
                node_id=post_id,
                node_type="content",
                platform="reddit",
                title=post.get("title_cleaned") or post.get("title"),
            )
            self.full_graph.add_edge(
                author_node_id,
                post_id,
                relationship="POSTED",
                weight=1,
                platform="reddit",
            )

            # Extract mentions from post text
            mentions = self.extract_mentions(text_content)
            for mentioned in mentions:
                m_clean = str(mentioned).strip()
                if not self.is_valid_user(m_clean):
                    continue

                # Reddit self-mention check (case-insensitive)
                if m_clean.lower() == author_clean.lower():
                    continue

                target_node_id = self.make_reddit_user_id(m_clean)

                # Ensure mention target node exists without incrementing activity_count
                if not self.user_graph.has_node(target_node_id):
                    self.user_graph.add_node(
                        target_node_id,
                        node_id=target_node_id,
                        platform="reddit",
                        identity_type="account",
                        is_author=False,
                        username=m_clean,
                        channel_id=None,
                        channel_title=None,
                        activity_count=0,
                    )
                else:
                    existing_username = self.user_graph.nodes[target_node_id].get("username")
                    self.user_graph.nodes[target_node_id]["username"] = self._resolve_deterministic_username(
                        existing_username, m_clean
                    )

                user_edges[(author_node_id, target_node_id)]["weight"] += 1
                user_edges[(author_node_id, target_node_id)]["types"].add("MENTIONED")
                user_edges[(author_node_id, target_node_id)]["platforms"].add("reddit")

        # 2. Process YouTube records
        for video in youtube_list:
            if not isinstance(video, dict):
                continue

            raw_channel_id = video.get("channel_id")
            raw_channel_title = video.get("channel_title")

            # Determine authoritative channel_id (never fall back to channel_title)
            channel_id = str(raw_channel_id).strip() if (isinstance(raw_channel_id, str) and raw_channel_id.strip()) else ""
            if not channel_id:
                continue

            channel_title = str(raw_channel_title).strip() if (isinstance(raw_channel_title, str) and raw_channel_title.strip()) else None
            display_name = channel_title if channel_title else channel_id

            if not self.is_valid_user(display_name):
                continue

            creator_node_id = self.make_youtube_creator_id(channel_id)
            video_id = str(video.get("video_id", "unknown_video"))
            desc_content = f"{video.get('title_cleaned', '')} {video.get('description_cleaned', '')}"

            # Add or update YouTube creator node
            if self.user_graph.has_node(creator_node_id):
                self.user_graph.nodes[creator_node_id]["activity_count"] += 1
                existing_title = self.user_graph.nodes[creator_node_id].get("channel_title")
                if channel_title:
                    resolved_title = self._resolve_deterministic_username(existing_title, channel_title)
                    self.user_graph.nodes[creator_node_id]["channel_title"] = resolved_title
                    self.user_graph.nodes[creator_node_id]["username"] = resolved_title
            else:
                self.user_graph.add_node(
                    creator_node_id,
                    node_id=creator_node_id,
                    platform="youtube",
                    identity_type="creator",
                    is_author=True,
                    username=display_name,
                    channel_id=channel_id,
                    channel_title=channel_title,
                    activity_count=1,
                )

            # Full graph content edge
            self.full_graph.add_node(
                creator_node_id,
                node_id=creator_node_id,
                node_type="user",
                platform="youtube",
                identity_type="creator",
                username=self.user_graph.nodes[creator_node_id]["username"],
            )
            self.full_graph.add_node(
                video_id,
                node_id=video_id,
                node_type="content",
                platform="youtube",
                title=video.get("title_cleaned") or video.get("title"),
            )
            self.full_graph.add_edge(
                creator_node_id,
                video_id,
                relationship="POSTED",
                weight=1,
                platform="youtube",
            )

            # Mentions in video title / description
            mentions = self.extract_mentions(desc_content)
            for mentioned in mentions:
                m_clean = str(mentioned).strip()
                if not self.is_valid_user(m_clean):
                    continue

                mention_node_id = self.make_youtube_mention_id(m_clean)

                # Ensure mention target node exists without incrementing activity_count
                if not self.user_graph.has_node(mention_node_id):
                    self.user_graph.add_node(
                        mention_node_id,
                        node_id=mention_node_id,
                        platform="youtube",
                        identity_type="mention",
                        is_author=False,
                        username=m_clean,
                        channel_id=None,
                        channel_title=None,
                        activity_count=0,
                    )
                else:
                    existing_username = self.user_graph.nodes[mention_node_id].get("username")
                    self.user_graph.nodes[mention_node_id]["username"] = self._resolve_deterministic_username(
                        existing_username, m_clean
                    )

                user_edges[(creator_node_id, mention_node_id)]["weight"] += 1
                user_edges[(creator_node_id, mention_node_id)]["types"].add("MENTIONED")
                user_edges[(creator_node_id, mention_node_id)]["platforms"].add("youtube")

        # Populate user graph edges with sorted attributes
        for (src, tgt), data in user_edges.items():
            sorted_types = sorted(list(data["types"]))
            sorted_platforms = sorted(list(data["platforms"]))
            self.user_graph.add_edge(
                src,
                tgt,
                weight=data["weight"],
                interaction_types=sorted_types,
                platforms=sorted_platforms,
            )
            self.full_graph.add_edge(
                src,
                tgt,
                relationship="INTERACTED_WITH",
                weight=data["weight"],
                interaction_types=sorted_types,
                platforms=sorted_platforms,
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
            graph.nodes[node_id]["activity_count"] = graph.nodes[node_id].get("activity_count", 0) + 1
            for k, v in attrs.items():
                if k not in graph.nodes[node_id] or graph.nodes[node_id][k] is None:
                    graph.nodes[node_id][k] = v
        else:
            graph.add_node(node_id, activity_count=1, **attrs)

    def get_nodes(self) -> list[dict[str, Any]]:
        """Returns structured list of user graph nodes sorted deterministically."""
        nodes: list[dict[str, Any]] = []
        for node_id, attrs in self.user_graph.nodes(data=True):
            entry = dict(attrs)
            entry["id"] = node_id
            entry["node_id"] = node_id
            nodes.append(entry)
        nodes.sort(key=lambda x: str(x.get("node_id", "")))
        return nodes

    def get_edges(self) -> list[dict[str, Any]]:
        """Returns structured list of user graph edges sorted deterministically."""
        edges: list[dict[str, Any]] = []
        for src, tgt, attrs in self.user_graph.edges(data=True):
            entry = dict(attrs)
            entry["source"] = src
            entry["target"] = tgt
            if "platforms" in entry and isinstance(entry["platforms"], (list, set)):
                entry["platforms"] = sorted(list(set(entry["platforms"])))
            if "interaction_types" in entry and isinstance(entry["interaction_types"], (list, set)):
                entry["interaction_types"] = sorted(list(set(entry["interaction_types"])))
            edges.append(entry)
        edges.sort(key=lambda x: (str(x.get("source", "")), str(x.get("target", ""))))
        return edges

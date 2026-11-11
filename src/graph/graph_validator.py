"""Graph Artifact Validator (Stage 7 - Graph Analysis).

Validates Stage 7 graph analysis artifacts against Schema v2.
Enforces strict semantic versioning, namespaced identity rules, edge integrity,
deterministic ordering, and exact identity consistency between top_influencers and nodes.
"""

import math
import re
from typing import Any, Optional

# Strict semver regex: ^(0|[1-9]\d*)\.(0|[1-9]\d*)$
SCHEMA_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)$")

# Canonical node ID patterns
REDDIT_NODE_ID_PATTERN = re.compile(r"^reddit:user:[a-z0-9_-]+$")
YOUTUBE_CREATOR_NODE_ID_PATTERN = re.compile(r"^youtube:channel:[A-Za-z0-9_-]+$")
YOUTUBE_MENTION_NODE_ID_PATTERN = re.compile(r"^youtube:mention:[a-z0-9_.-]+$")

ALLOWED_STATUSES = {"success", "partial", "empty", "error"}
ALLOWED_SOURCE_STATUSES = {"loaded", "empty", "missing", "corrupted"}
ALLOWED_PLATFORMS = {"reddit", "youtube"}


def _is_strict_bool(val: Any) -> bool:
    """Returns True if val is strictly a Python boolean (not an int or other type)."""
    return type(val) is bool


def _is_strict_int(val: Any) -> bool:
    """Returns True if val is strictly a Python integer (excluding booleans)."""
    return type(val) is int and not isinstance(val, bool)


def _is_finite_float_in_range(val: Any, min_val: float, max_val: float) -> bool:
    """Returns True if val is a finite float or int within [min_val, max_val], excluding booleans."""
    if _is_strict_bool(val):
        return False
    if isinstance(val, (int, float)):
        return math.isfinite(val) and min_val <= float(val) <= max_val
    return False


class GraphArtifactValidator:
    """Strict schema-v2 validator for Stage 7 graph analysis artifacts.

    Validates schema versioning, entity namespaces, edge referential integrity,
    deterministic tie-breaking, provenance contracts, and cross-field status consistency.

    NOTE: Artifacts with status='error' are diagnostic only and must NEVER be promoted
    to production artifacts by the pipeline executor.
    """

    @classmethod
    def validate_graph_artifact(cls, payload: Any) -> tuple[bool, list[str]]:
        """Validates a graph analysis output payload against Schema v2.

        Args:
            payload: Parsed JSON payload to validate.

        Returns:
            Tuple of (is_valid: bool, errors: list[str]).
        """
        errors: list[str] = []
        if not isinstance(payload, dict):
            return False, [f"Top-level payload must be a dict, got {type(payload).__name__}"]

        # 1. Schema version validation
        version = payload.get("schema_version")
        ver_ok, ver_err = cls._validate_schema_version(version)
        if not ver_ok and ver_err:
            errors.append(ver_err)

        # 2. Status validation
        status = payload.get("status")
        if not isinstance(status, str) or status not in ALLOWED_STATUSES:
            errors.append(
                f"'status' must be one of {sorted(list(ALLOWED_STATUSES))}, got {status!r}"
            )

        # 3. Generated at
        gen_at = payload.get("generated_at")
        if not isinstance(gen_at, str) or not gen_at.strip():
            errors.append("'generated_at' must be a non-empty ISO-8601 timestamp string")

        # 4. Provenance
        prov = payload.get("provenance")
        prov_errs, prov_data = cls._validate_provenance(prov)
        errors.extend(prov_errs)

        # 5. Nodes validation
        nodes = payload.get("nodes")
        nodes_by_id: dict[str, dict[str, Any]] = {}
        if not isinstance(nodes, list):
            errors.append(f"'nodes' must be a list, got {type(nodes).__name__}")
        else:
            errors.extend(cls._validate_nodes(nodes, nodes_by_id))

        # 6. Edges validation
        edges = payload.get("edges")
        if not isinstance(edges, list):
            errors.append(f"'edges' must be a list, got {type(edges).__name__}")
        else:
            errors.extend(cls._validate_edges(edges, set(nodes_by_id.keys())))

        # 7. Graph metrics validation
        metrics = payload.get("graph_metrics")
        num_nodes = len(nodes) if isinstance(nodes, list) else 0
        num_edges = len(edges) if isinstance(edges, list) else 0
        if not isinstance(metrics, dict):
            errors.append(f"'graph_metrics' must be a dict, got {type(metrics).__name__}")
        else:
            errors.extend(cls._validate_graph_metrics(metrics, num_nodes, num_edges))

        # 8. Community distributions validation
        dist = payload.get("community_distributions")
        expected_comms: Optional[int] = None
        if isinstance(metrics, dict) and _is_strict_int(metrics.get("communities_count")):
            expected_comms = metrics.get("communities_count")

        if not isinstance(dist, dict):
            errors.append(f"'community_distributions' must be a dict, got {type(dist).__name__}")
        else:
            errors.extend(cls._validate_community_distributions(dist, expected_comms, num_nodes))

        # 9. Top influencers validation
        top_inf = payload.get("top_influencers")
        if not isinstance(top_inf, list):
            errors.append(f"'top_influencers' must be a list, got {type(top_inf).__name__}")
        else:
            errors.extend(
                cls._validate_top_influencers(
                    top_inf,
                    nodes_by_id,
                    dist if isinstance(dist, dict) else None,
                )
            )

        # 10. Status and provenance consistency validation
        if isinstance(status, str) and status in ALLOWED_STATUSES:
            status_errs = cls._validate_status_consistency(
                status=status,
                provenance_data=prov_data,
                num_nodes=num_nodes,
                num_edges=num_edges,
                top_inf_len=len(top_inf) if isinstance(top_inf, list) else 0,
                metrics=metrics if isinstance(metrics, dict) else {},
                dist=dist if isinstance(dist, dict) else None,
            )
            errors.extend(status_errs)

        return (len(errors) == 0, errors)

    @classmethod
    def _validate_schema_version(cls, version: Any) -> tuple[bool, Optional[str]]:
        """Validates and classifies schema version string.

        Classification:
        - None: 'missing_schema_version'
        - Non-string or invalid syntax: 'malformed_schema_version'
        - Major version < 2: 'legacy_graph_schema_lacks_channel_id'
        - Major version == 2: supported (2.0, 2.1, 2.12, ...)
        - Major version >= 3: 'unsupported_graph_schema_version'
        """
        if version is None:
            return False, "missing_schema_version: 'schema_version' is required"
        if not isinstance(version, str):
            return (
                False,
                f"malformed_schema_version: 'schema_version' must be a string, got {type(version).__name__}",
            )

        match = SCHEMA_VERSION_PATTERN.match(version)
        if not match:
            return (
                False,
                f"malformed_schema_version: '{version}' does not match pattern ^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)$",
            )

        major = int(match.group(1))

        if major < 2:
            return (
                False,
                f"legacy_graph_schema_lacks_channel_id: major version {major} < 2 is deprecated",
            )
        if major >= 3:
            return (
                False,
                f"unsupported_graph_schema_version: major version {major} >= 3 is not supported",
            )

        return True, None

    @classmethod
    def _validate_provenance(
        cls, prov: Any
    ) -> tuple[list[str], Optional[dict[str, Any]]]:
        """Validates provenance dictionary structure, types, and count consistency.

        Returns:
            Tuple of (errors: list[str], validated_dict_or_None).
        """
        errors: list[str] = []
        if not isinstance(prov, dict):
            return [f"'provenance' must be a dict, got {type(prov).__name__}"], None
        if len(prov) == 0:
            return ["'provenance' must not be empty: source provenance tracking is required"], None

        req_keys = [
            "reddit_input_file",
            "reddit_status",
            "reddit_records_count",
            "youtube_input_file",
            "youtube_status",
            "youtube_records_count",
        ]
        for k in req_keys:
            if k not in prov:
                errors.append(f"provenance: missing required key '{k}'")

        r_file = prov.get("reddit_input_file")
        if "reddit_input_file" in prov and (not isinstance(r_file, str) or not r_file.strip()):
            errors.append(f"provenance.reddit_input_file: must be non-empty string, got {r_file!r}")

        y_file = prov.get("youtube_input_file")
        if "youtube_input_file" in prov and (not isinstance(y_file, str) or not y_file.strip()):
            errors.append(f"provenance.youtube_input_file: must be non-empty string, got {y_file!r}")

        r_status = prov.get("reddit_status")
        if "reddit_status" in prov and (
            not isinstance(r_status, str) or r_status not in ALLOWED_SOURCE_STATUSES
        ):
            errors.append(
                f"provenance.reddit_status: must be one of {sorted(list(ALLOWED_SOURCE_STATUSES))}, got {r_status!r}"
            )

        y_status = prov.get("youtube_status")
        if "youtube_status" in prov and (
            not isinstance(y_status, str) or y_status not in ALLOWED_SOURCE_STATUSES
        ):
            errors.append(
                f"provenance.youtube_status: must be one of {sorted(list(ALLOWED_SOURCE_STATUSES))}, got {y_status!r}"
            )

        r_count = prov.get("reddit_records_count")
        if "reddit_records_count" in prov:
            if not _is_strict_int(r_count) or r_count < 0:
                errors.append(
                    f"provenance.reddit_records_count: must be non-negative integer, got {r_count!r}"
                )

        y_count = prov.get("youtube_records_count")
        if "youtube_records_count" in prov:
            if not _is_strict_int(y_count) or y_count < 0:
                errors.append(
                    f"provenance.youtube_records_count: must be non-negative integer, got {y_count!r}"
                )

        # Status and count consistency checks
        if (
            isinstance(r_status, str)
            and r_status in ALLOWED_SOURCE_STATUSES
            and _is_strict_int(r_count)
            and r_count >= 0
        ):
            if r_status == "loaded" and r_count < 1:
                errors.append(
                    f"provenance: reddit_status='loaded' requires reddit_records_count >= 1, got {r_count}"
                )
            elif r_status == "empty" and r_count != 0:
                errors.append(
                    f"provenance: reddit_status='empty' requires reddit_records_count == 0, got {r_count}"
                )
            elif r_status in {"missing", "corrupted"} and r_count != 0:
                errors.append(
                    f"provenance: reddit_status={r_status!r} requires reddit_records_count == 0, got {r_count}"
                )

        if (
            isinstance(y_status, str)
            and y_status in ALLOWED_SOURCE_STATUSES
            and _is_strict_int(y_count)
            and y_count >= 0
        ):
            if y_status == "loaded" and y_count < 1:
                errors.append(
                    f"provenance: youtube_status='loaded' requires youtube_records_count >= 1, got {y_count}"
                )
            elif y_status == "empty" and y_count != 0:
                errors.append(
                    f"provenance: youtube_status='empty' requires youtube_records_count == 0, got {y_count}"
                )
            elif y_status in {"missing", "corrupted"} and y_count != 0:
                errors.append(
                    f"provenance: youtube_status={y_status!r} requires youtube_records_count == 0, got {y_count}"
                )

        if len(errors) == 0:
            return errors, prov
        return errors, None

    @classmethod
    def _validate_node_identity(cls, node: dict[str, Any], context_label: str) -> list[str]:
        """Validates platform-specific identity rules for a node or top_influencer entry."""
        errors: list[str] = []

        node_id = node.get("node_id")
        platform = node.get("platform")
        identity_type = node.get("identity_type")
        username = node.get("username")
        channel_id = node.get("channel_id")
        channel_title = node.get("channel_title")
        is_author = node.get("is_author")
        activity_count = node.get("activity_count")

        if not isinstance(node_id, str) or not node_id.strip():
            errors.append(f"{context_label}: missing or invalid 'node_id'")
            return errors

        # Username validation: must be a non-empty string and never a raw namespaced ID
        if not isinstance(username, str) or not username.strip():
            errors.append(
                f"{context_label} ({node_id}): 'username' must be a non-empty string, got {username!r}"
            )
        elif username.startswith("reddit:") or username.startswith("youtube:"):
            errors.append(
                f"{context_label} ({node_id}): 'username' must be a display name, not raw namespaced ID '{username}'"
            )

        # Platform-specific validation
        if platform == "reddit":
            if identity_type != "account":
                errors.append(
                    f"{context_label} ({node_id}): reddit platform requires identity_type='account', got {identity_type!r}"
                )
            if not REDDIT_NODE_ID_PATTERN.match(node_id):
                errors.append(
                    f"{context_label} ({node_id}): invalid reddit node_id pattern, must match ^reddit:user:[a-z0-9_-]+$"
                )
            if channel_id is not None:
                errors.append(
                    f"{context_label} ({node_id}): reddit node requires channel_id=None, got {channel_id!r}"
                )
            if channel_title is not None:
                errors.append(
                    f"{context_label} ({node_id}): reddit node requires channel_title=None, got {channel_title!r}"
                )
            if not _is_strict_bool(is_author):
                errors.append(
                    f"{context_label} ({node_id}): 'is_author' must be a boolean, got {type(is_author).__name__}"
                )
            if not _is_strict_int(activity_count) or activity_count < 0:
                errors.append(
                    f"{context_label} ({node_id}): 'activity_count' must be non-negative integer, got {activity_count!r}"
                )
            elif _is_strict_bool(is_author):
                if is_author and activity_count < 1:
                    errors.append(
                        f"{context_label} ({node_id}): is_author=True requires activity_count >= 1, got {activity_count}"
                    )
                elif not is_author and activity_count != 0:
                    errors.append(
                        f"{context_label} ({node_id}): is_author=False requires activity_count == 0, got {activity_count}"
                    )

        elif platform == "youtube":
            if identity_type == "creator":
                if not YOUTUBE_CREATOR_NODE_ID_PATTERN.match(node_id):
                    errors.append(
                        f"{context_label} ({node_id}): invalid youtube creator node_id pattern, must match ^youtube:channel:[A-Za-z0-9_-]+$"
                    )
                if not isinstance(channel_id, str) or not channel_id.strip():
                    errors.append(
                        f"{context_label} ({node_id}): youtube creator requires non-empty string 'channel_id'"
                    )
                elif node_id != f"youtube:channel:{channel_id}":
                    errors.append(
                        f"{context_label} ({node_id}): 'channel_id' '{channel_id}' does not match node_id suffix in '{node_id}'"
                    )
                if channel_title is not None and (
                    not isinstance(channel_title, str) or not channel_title.strip()
                ):
                    errors.append(
                        f"{context_label} ({node_id}): 'channel_title' must be non-empty string or None, got {channel_title!r}"
                    )
                if is_author is not True or not _is_strict_bool(is_author):
                    errors.append(
                        f"{context_label} ({node_id}): youtube creator requires is_author=True, got {is_author!r}"
                    )
                if not _is_strict_int(activity_count) or activity_count < 1:
                    errors.append(
                        f"{context_label} ({node_id}): youtube creator requires activity_count >= 1, got {activity_count!r}"
                    )

            elif identity_type == "mention":
                if not YOUTUBE_MENTION_NODE_ID_PATTERN.match(node_id):
                    errors.append(
                        f"{context_label} ({node_id}): invalid youtube mention node_id pattern, must match ^youtube:mention:[a-z0-9_.-]+$"
                    )
                if channel_id is not None:
                    errors.append(
                        f"{context_label} ({node_id}): youtube mention requires channel_id=None, got {channel_id!r}"
                    )
                if channel_title is not None:
                    errors.append(
                        f"{context_label} ({node_id}): youtube mention requires channel_title=None, got {channel_title!r}"
                    )
                if is_author is not False or not _is_strict_bool(is_author):
                    errors.append(
                        f"{context_label} ({node_id}): youtube mention requires is_author=False, got {is_author!r}"
                    )
                if not _is_strict_int(activity_count) or activity_count != 0:
                    errors.append(
                        f"{context_label} ({node_id}): youtube mention requires activity_count=0, got {activity_count!r}"
                    )
            else:
                errors.append(
                    f"{context_label} ({node_id}): invalid youtube identity_type {identity_type!r}, must be 'creator' or 'mention'"
                )
        else:
            errors.append(
                f"{context_label} ({node_id}): unsupported platform {platform!r}, must be 'reddit' or 'youtube'"
            )

        return errors

    @classmethod
    def _validate_nodes(
        cls, nodes: list[Any], nodes_by_id: dict[str, dict[str, Any]]
    ) -> list[str]:
        """Validates complete nodes array for structure, unique IDs, and sorting."""
        errors: list[str] = []
        seen_ids: set[str] = set()

        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"nodes[{idx}]: expected dict, got {type(node).__name__}")
                continue

            node_errs = cls._validate_node_identity(node, f"nodes[{idx}]")
            errors.extend(node_errs)

            nid = node.get("node_id")
            if isinstance(nid, str) and nid.strip():
                if nid in seen_ids:
                    errors.append(f"nodes[{idx}]: duplicate node_id '{nid}'")
                seen_ids.add(nid)
                nodes_by_id[nid] = node

        # Deterministic node ordering: nodes must be sorted by canonical node_id ascending
        for idx in range(len(nodes) - 1):
            if isinstance(nodes[idx], dict) and isinstance(nodes[idx + 1], dict):
                cur_id = str(nodes[idx].get("node_id") or "")
                next_id = str(nodes[idx + 1].get("node_id") or "")
                if cur_id > next_id:
                    errors.append(
                        f"nodes: unsorted node list at index {idx} ('{cur_id}' > '{next_id}'). Must be sorted by canonical node_id."
                    )
                    break

        return errors

    @classmethod
    def _validate_edges(cls, edges: list[Any], node_ids_set: set[str]) -> list[str]:
        """Validates edges array for referential integrity, positive weights, and sorting."""
        errors: list[str] = []
        seen_pairs: set[tuple[str, str]] = set()

        for idx, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"edges[{idx}]: expected dict, got {type(edge).__name__}")
                continue

            src = edge.get("source")
            tgt = edge.get("target")
            weight = edge.get("weight")
            interaction_types = edge.get("interaction_types")
            platforms = edge.get("platforms")

            if not isinstance(src, str) or not src.strip():
                errors.append(f"edges[{idx}]: missing or invalid 'source'")
            elif src not in node_ids_set:
                errors.append(f"edges[{idx}]: source '{src}' does not exist in nodes")

            if not isinstance(tgt, str) or not tgt.strip():
                errors.append(f"edges[{idx}]: missing or invalid 'target'")
            elif tgt not in node_ids_set:
                errors.append(f"edges[{idx}]: target '{tgt}' does not exist in nodes")

            if not _is_strict_int(weight) or weight < 1:
                errors.append(f"edges[{idx}]: 'weight' must be integer >= 1, got {weight!r}")

            if not isinstance(interaction_types, list) or len(interaction_types) == 0:
                errors.append(f"edges[{idx}]: 'interaction_types' must be non-empty list")
            else:
                for t in interaction_types:
                    if not isinstance(t, str) or not t.strip():
                        errors.append(
                            f"edges[{idx}]: interaction_types contains non-string or blank entry: {t!r}"
                        )
                if interaction_types != sorted(list(set(interaction_types))):
                    errors.append(
                        f"edges[{idx}]: interaction_types must be sorted without duplicates, got {interaction_types}"
                    )

            if not isinstance(platforms, list) or len(platforms) == 0:
                errors.append(f"edges[{idx}]: 'platforms' must be non-empty list")
            else:
                for p in platforms:
                    if not isinstance(p, str) or p not in ALLOWED_PLATFORMS:
                        errors.append(
                            f"edges[{idx}]: platforms contains invalid platform {p!r}"
                        )
                if platforms != sorted(list(set(platforms))):
                    errors.append(
                        f"edges[{idx}]: platforms must be sorted without duplicates, got {platforms}"
                    )

            if isinstance(src, str) and isinstance(tgt, str):
                pair = (src, tgt)
                if pair in seen_pairs:
                    errors.append(f"edges[{idx}]: duplicate directed edge pair {pair}")
                seen_pairs.add(pair)

        # Deterministic edge ordering: edges must be sorted by (source, target)
        for idx in range(len(edges) - 1):
            if isinstance(edges[idx], dict) and isinstance(edges[idx + 1], dict):
                cur_key = (
                    str(edges[idx].get("source") or ""),
                    str(edges[idx].get("target") or ""),
                )
                next_key = (
                    str(edges[idx + 1].get("source") or ""),
                    str(edges[idx + 1].get("target") or ""),
                )
                if cur_key > next_key:
                    errors.append(
                        f"edges: unsorted edge list at index {idx} ({cur_key} > {next_key}). Must be sorted by (source, target)."
                    )
                    break

        return errors

    @classmethod
    def _validate_top_influencers(
        cls,
        top_influencers: list[Any],
        nodes_by_id: dict[str, dict[str, Any]],
        community_distributions: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """Validates top_influencers entries for node referential integrity, metrics, and deterministic sorting."""
        errors: list[str] = []
        seen_influencers: set[str] = set()

        for idx, inf in enumerate(top_influencers):
            if not isinstance(inf, dict):
                errors.append(f"top_influencers[{idx}]: expected dict, got {type(inf).__name__}")
                continue

            nid = inf.get("node_id")
            if not isinstance(nid, str) or not nid.strip():
                errors.append(f"top_influencers[{idx}]: missing or invalid 'node_id'")
                continue

            if nid in seen_influencers:
                errors.append(f"top_influencers[{idx}]: duplicate influencer entry for '{nid}'")
            seen_influencers.add(nid)

            # Node referential integrity: influencer must exist in nodes array
            if nid not in nodes_by_id:
                errors.append(
                    f"top_influencers[{idx}] ({nid}): node_id does not exist in nodes array"
                )
            else:
                # Identity metadata consistency between top_influencers and nodes
                canonical_node = nodes_by_id[nid]
                for field in [
                    "platform",
                    "identity_type",
                    "username",
                    "channel_id",
                    "channel_title",
                    "is_author",
                    "activity_count",
                ]:
                    inf_val = inf.get(field)
                    node_val = canonical_node.get(field)
                    if inf_val != node_val:
                        errors.append(
                            f"top_influencers[{idx}] ({nid}): metadata mismatch for '{field}': "
                            f"influencer has {inf_val!r}, node has {node_val!r}"
                        )

            # Validate identity rules
            node_errs = cls._validate_node_identity(inf, f"top_influencers[{idx}]")
            errors.extend(node_errs)

            # Validate ranking metrics
            inf_score = inf.get("influencer_score")
            if not _is_finite_float_in_range(inf_score, 0.0, 1.0):
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'influencer_score' must be finite float in [0.0, 1.0], got {inf_score!r}"
                )

            pr = inf.get("pagerank")
            if _is_strict_bool(pr) or not isinstance(pr, (int, float)) or not math.isfinite(pr) or pr < 0.0:
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'pagerank' must be non-negative finite float, got {pr!r}"
                )

            pr_norm = inf.get("pagerank_norm")
            if not _is_finite_float_in_range(pr_norm, 0.0, 1.0):
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'pagerank_norm' must be finite float in [0.0, 1.0], got {pr_norm!r}"
                )

            in_deg_cent = inf.get("in_degree_centrality")
            if not _is_finite_float_in_range(in_deg_cent, 0.0, 1.0):
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'in_degree_centrality' must be finite float in [0.0, 1.0], got {in_deg_cent!r}"
                )

            in_deg_norm = inf.get("in_degree_norm")
            if not _is_finite_float_in_range(in_deg_norm, 0.0, 1.0):
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'in_degree_norm' must be finite float in [0.0, 1.0], got {in_deg_norm!r}"
                )

            btw_cent = inf.get("betweenness_centrality")
            if not _is_finite_float_in_range(btw_cent, 0.0, 1.0):
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'betweenness_centrality' must be finite float in [0.0, 1.0], got {btw_cent!r}"
                )

            btw_norm = inf.get("betweenness_norm")
            if not _is_finite_float_in_range(btw_norm, 0.0, 1.0):
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'betweenness_norm' must be finite float in [0.0, 1.0], got {btw_norm!r}"
                )

            comm_id = inf.get("community_id")
            if not _is_strict_int(comm_id) or comm_id < 0:
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'community_id' must be non-negative integer, got {comm_id!r}"
                )
            elif community_distributions is not None:
                comm_key = str(comm_id)
                if comm_key not in community_distributions:
                    errors.append(
                        f"top_influencers[{idx}] ({nid}): community_id {comm_id} ('{comm_key}') not found in community_distributions"
                    )
                elif (
                    _is_strict_int(community_distributions.get(comm_key))
                    and community_distributions[comm_key] < 1
                ):
                    errors.append(
                        f"top_influencers[{idx}] ({nid}): community_id {comm_id} count in community_distributions is {community_distributions[comm_key]}, expected >= 1"
                    )

            in_raw = inf.get("in_degree_raw")
            if not _is_strict_int(in_raw) or in_raw < 0:
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'in_degree_raw' must be non-negative integer, got {in_raw!r}"
                )

            out_raw = inf.get("out_degree_raw")
            if not _is_strict_int(out_raw) or out_raw < 0:
                errors.append(
                    f"top_influencers[{idx}] ({nid}): 'out_degree_raw' must be non-negative integer, got {out_raw!r}"
                )

        # Deterministic ranking order check: top_influencers must be sorted by (-influencer_score, -pagerank, node_id)
        for idx in range(len(top_influencers) - 1):
            cur = top_influencers[idx]
            nxt = top_influencers[idx + 1]
            if isinstance(cur, dict) and isinstance(nxt, dict):
                cur_score = cur.get("influencer_score")
                nxt_score = nxt.get("influencer_score")
                cur_pr = cur.get("pagerank")
                nxt_pr = nxt.get("pagerank")
                cur_nid = str(cur.get("node_id") or "")
                nxt_nid = str(nxt.get("node_id") or "")

                if (
                    isinstance(cur_score, (int, float))
                    and not _is_strict_bool(cur_score)
                    and math.isfinite(cur_score)
                    and isinstance(nxt_score, (int, float))
                    and not _is_strict_bool(nxt_score)
                    and math.isfinite(nxt_score)
                    and isinstance(cur_pr, (int, float))
                    and not _is_strict_bool(cur_pr)
                    and math.isfinite(cur_pr)
                    and isinstance(nxt_pr, (int, float))
                    and not _is_strict_bool(nxt_pr)
                    and math.isfinite(nxt_pr)
                ):
                    cur_key = (-float(cur_score), -float(cur_pr), cur_nid)
                    nxt_key = (-float(nxt_score), -float(nxt_pr), nxt_nid)
                    if cur_key > nxt_key:
                        errors.append(
                            f"top_influencers: unsorted ranking list at index {idx} "
                            f"('{cur_nid}' > '{nxt_nid}'). Must be sorted deterministically by "
                            f"(-influencer_score, -pagerank, node_id)."
                        )
                        break

        return errors

    @classmethod
    def _validate_graph_metrics(
        cls, metrics: dict[str, Any], num_nodes: int, num_edges: int
    ) -> list[str]:
        """Validates graph_metrics fields and numerical constraints."""
        errors: list[str] = []

        required_keys = [
            "total_users",
            "total_interactions",
            "density",
            "communities_count",
            "is_directed",
        ]
        for k in required_keys:
            if k not in metrics:
                errors.append(f"graph_metrics: missing required key '{k}'")

        total_users = metrics.get("total_users")
        if not _is_strict_int(total_users) or total_users < 0:
            errors.append(
                f"graph_metrics.total_users: must be non-negative integer, got {total_users!r}"
            )
        elif total_users != num_nodes:
            errors.append(
                f"graph_metrics.total_users: count ({total_users}) does not match nodes length ({num_nodes})"
            )

        total_interactions = metrics.get("total_interactions")
        if not _is_strict_int(total_interactions) or total_interactions < 0:
            errors.append(
                f"graph_metrics.total_interactions: must be non-negative integer, got {total_interactions!r}"
            )
        elif total_interactions != num_edges:
            errors.append(
                f"graph_metrics.total_interactions: count ({total_interactions}) does not match edges length ({num_edges})"
            )

        density = metrics.get("density")
        if not _is_finite_float_in_range(density, 0.0, 1.0):
            errors.append(
                f"graph_metrics.density: must be finite float in [0.0, 1.0], got {density!r}"
            )

        comm_count = metrics.get("communities_count")
        if not _is_strict_int(comm_count) or comm_count < 0:
            errors.append(
                f"graph_metrics.communities_count: must be non-negative integer, got {comm_count!r}"
            )

        is_directed = metrics.get("is_directed")
        if is_directed is not True or not _is_strict_bool(is_directed):
            errors.append(f"graph_metrics.is_directed: must be True, got {is_directed!r}")

        return errors

    @classmethod
    def _validate_community_distributions(
        cls,
        dist: dict[str, Any],
        expected_comm_count: Optional[int],
        total_nodes: int,
    ) -> list[str]:
        """Validates community_distributions counts and consistency with graph_metrics."""
        errors: list[str] = []

        if expected_comm_count is not None and len(dist) != expected_comm_count:
            errors.append(
                f"community_distributions: count of communities ({len(dist)}) does not match graph_metrics.communities_count ({expected_comm_count})"
            )

        total_in_dist = 0
        for comm_key, count in dist.items():
            if not isinstance(comm_key, str) or not comm_key.isdigit():
                errors.append(
                    f"community_distributions: key '{comm_key}' must be a string integer (e.g. '0', '1')"
                )
            if not _is_strict_int(count) or count < 0:
                errors.append(
                    f"community_distributions[{comm_key}]: count must be non-negative integer, got {count!r}"
                )
            else:
                total_in_dist += count

        if total_nodes > 0 and total_in_dist != total_nodes:
            errors.append(
                f"community_distributions: sum of community counts ({total_in_dist}) does not match total nodes ({total_nodes})"
            )

        return errors

    @classmethod
    def _validate_status_consistency(
        cls,
        status: str,
        provenance_data: Optional[dict[str, Any]],
        num_nodes: int,
        num_edges: int,
        top_inf_len: int,
        metrics: dict[str, Any],
        dist: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        """Validates consistency between artifact status, provenance outcomes, and graph contents.

        Status Semantics:
        - 'empty': Valid ONLY when both source outcomes are successful ('loaded' or 'empty')
          and the resulting graph contains zero entities (0 nodes, 0 edges, 0 top_influencers,
          0 community_distributions, and zero-valued graph metrics). If either source is
          'missing' or 'corrupted', status must be 'partial' or 'error', never 'empty'.
        - 'partial': Requires at least one source failure ('missing' or 'corrupted') alongside
          at least one successful source ('loaded' or 'empty').
        - 'success': Requires both sources to succeed with status in {'loaded', 'empty'}.
        - 'error': Diagnostic status when unrecoverable source or execution failures occur.
          NOTE: An artifact with status='error' is diagnostic only and must NEVER be promoted
          to production artifacts by the pipeline executor.
        """
        errors: list[str] = []

        if status == "empty":
            if num_nodes != 0:
                errors.append(f"status 'empty' requires empty nodes list, got {num_nodes} nodes")
            if num_edges != 0:
                errors.append(f"status 'empty' requires empty edges list, got {num_edges} edges")
            if top_inf_len != 0:
                errors.append(
                    f"status 'empty' requires empty top_influencers list, got {top_inf_len} entries"
                )
            total_users = metrics.get("total_users")
            if _is_strict_int(total_users) and total_users != 0:
                errors.append(
                    f"status 'empty' requires graph_metrics.total_users == 0, got {total_users}"
                )
            total_interactions = metrics.get("total_interactions")
            if _is_strict_int(total_interactions) and total_interactions != 0:
                errors.append(
                    f"status 'empty' requires graph_metrics.total_interactions == 0, got {total_interactions}"
                )
            comm_count = metrics.get("communities_count")
            if _is_strict_int(comm_count) and comm_count != 0:
                errors.append(
                    f"status 'empty' requires graph_metrics.communities_count == 0, got {comm_count}"
                )
            density = metrics.get("density")
            if (
                isinstance(density, (int, float))
                and not _is_strict_bool(density)
                and density != 0.0
                and density != 0
            ):
                errors.append(
                    f"status 'empty' requires graph_metrics.density == 0.0, got {density}"
                )
            if dist is not None and len(dist) != 0:
                errors.append(
                    f"status 'empty' requires empty community_distributions, got {len(dist)} entries"
                )

        if provenance_data is not None:
            r_status = provenance_data.get("reddit_status")
            y_status = provenance_data.get("youtube_status")
            statuses = [r_status, y_status]

            has_failure = any(s in {"missing", "corrupted"} for s in statuses)
            all_succeeded = all(s in {"loaded", "empty"} for s in statuses)
            all_failed = all(s in {"missing", "corrupted"} for s in statuses)

            if status == "empty" and not all_succeeded:
                errors.append(
                    f"status 'empty' requires both sources to succeed with status in ['loaded', 'empty'], "
                    f"but got reddit_status={r_status!r}, youtube_status={y_status!r}. "
                    f"Failed or missing sources must yield 'partial' or 'error', never 'empty'."
                )
            elif status == "success" and not all_succeeded:
                errors.append(
                    f"status is 'success' but provenance indicates source failure/degradation: "
                    f"reddit_status={r_status!r}, youtube_status={y_status!r}"
                )
            elif status == "partial":
                if not has_failure:
                    errors.append(
                        f"status is 'partial' but all sources succeeded without failure: "
                        f"reddit_status={r_status!r}, youtube_status={y_status!r}"
                    )
                elif all_failed:
                    errors.append(
                        f"status is 'partial' but all sources failed: "
                        f"reddit_status={r_status!r}, youtube_status={y_status!r}. Expected status 'error'."
                    )
            elif status == "error":
                if all_succeeded:
                    errors.append(
                        f"status is 'error' but all sources succeeded without failure: "
                        f"reddit_status={r_status!r}, youtube_status={y_status!r}"
                    )

        return errors



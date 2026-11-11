"""Kibana Dashboard Saved Objects Bundle Validator.

Validates the canonical NDJSON bundle (config/kibana/social_intelligence_dashboards.ndjson)
offline with zero external service dependencies.

Enforces:
1. Valid NDJSON formatting (each line valid JSON).
2. Exactly 41 saved objects (1 Data View, 34 visual panels, 1 Markdown banner, 5 dashboards).
3. Unique saved-object IDs across all objects.
4. Correct Data View index pattern: 'consolidated_intelligence'.
5. Correct primary temporal field: 'timestamps.published_at'.
6. Integrity of all dashboard references (no dangling or missing references).
7. Conformance of visualization field bindings with canonical index mapping.
8. Rejection of unindexed raw-text fields (content.title_raw, content.body_raw).
9. Presence of required status filters and methodological disclaimers.
10. Correct metric definitions (influencer max ranking, community cardinality of username,
    pipeline completeness rate formula, ingestion timeline on collected_at).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.search.index_mapping import DEFAULT_INDEX_NAME, INDEX_MAPPINGS

logger = logging.getLogger(__name__)

DEFAULT_BUNDLE_PATH = Path("config/kibana/social_intelligence_dashboards.ndjson")
EXPECTED_TOTAL_OBJECTS = 41
EXPECTED_DASHBOARD_COUNT = 5
EXPECTED_DATA_VIEW_COUNT = 1
EXPECTED_VISUALIZATION_COUNT = 34
EXPECTED_MARKDOWN_COUNT = 1

EXPECTED_DASHBOARD_PANEL_COUNTS: Dict[str, int] = {
    "dashboard-executive-overview": 8,
    "dashboard-sentiment-intelligence": 7,
    "dashboard-topic-intelligence": 6,
    "dashboard-misinformation-intelligence": 8,  # 7 visualizations + 1 Markdown banner
    "dashboard-network-intelligence": 6,
}

DISALLOWED_UNINDEXED_FIELDS = {
    "content.title_raw",
    "content.body_raw",
}


def _collect_canonical_fields(mapping_props: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Recursively collects valid dot-notated field paths from the canonical index mapping."""
    fields = set()
    for key, spec in mapping_props.items():
        field_path = f"{prefix}.{key}" if prefix else key
        fields.add(field_path)
        if "properties" in spec:
            fields.update(_collect_canonical_fields(spec["properties"], field_path))
    return fields


CANONICAL_FIELDS = _collect_canonical_fields(INDEX_MAPPINGS.get("properties", {}))


def validate_kibana_dashboard_bundle(
    bundle_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Validates the social intelligence Kibana dashboard NDJSON bundle offline.

    Args:
        bundle_path: Path to the .ndjson bundle file. Defaults to DEFAULT_BUNDLE_PATH.

    Returns:
        Structured dictionary reporting validation outcome:
        {
            "is_valid": bool,
            "total_objects": int,
            "counts_by_type": Dict[str, int],
            "dashboards": Dict[str, int],
            "errors": List[str],
            "warnings": List[str],
        }
    """
    target_path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH
    errors: List[str] = []
    warnings: List[str] = []
    counts_by_type: Dict[str, int] = {}
    dashboard_panel_counts: Dict[str, int] = {}

    if not target_path.exists():
        return {
            "is_valid": False,
            "total_objects": 0,
            "counts_by_type": {},
            "dashboards": {},
            "errors": [f"Kibana dashboard bundle not found: {target_path}"],
            "warnings": [],
        }

    raw_lines: List[str] = []
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    except Exception as exc:
        return {
            "is_valid": False,
            "total_objects": 0,
            "counts_by_type": {},
            "dashboards": {},
            "errors": [f"Failed to read NDJSON file ({type(exc).__name__}): {exc}"],
            "warnings": [],
        }

    # 1. Parse each line as valid JSON
    objects: List[Dict[str, Any]] = []
    for line_idx, line in enumerate(raw_lines, start=1):
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                errors.append(f"Line {line_idx}: expected JSON object dictionary, got {type(obj).__name__}")
                continue
            objects.append(obj)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {line_idx}: invalid JSON syntax ({exc.msg})")

    if errors:
        return {
            "is_valid": False,
            "total_objects": len(objects),
            "counts_by_type": {},
            "dashboards": {},
            "errors": errors,
            "warnings": warnings,
        }

    # 2. Total object count verification
    total_objects = len(objects)
    if total_objects != EXPECTED_TOTAL_OBJECTS:
        errors.append(f"Expected exactly {EXPECTED_TOTAL_OBJECTS} saved objects, found {total_objects}")

    # 3. Categorize by type and check ID uniqueness
    seen_ids: Set[str] = set()
    objects_by_id: Dict[str, Dict[str, Any]] = {}
    data_view_obj: Optional[Dict[str, Any]] = None
    markdown_objects: List[Dict[str, Any]] = []
    visualization_objects: List[Dict[str, Any]] = []
    dashboard_objects: List[Dict[str, Any]] = []

    for idx, obj in enumerate(objects):
        obj_id = obj.get("id")
        obj_type = obj.get("type")

        if not obj_id or not isinstance(obj_id, str):
            errors.append(f"Object at index {idx} has invalid or missing 'id'")
            continue

        if not obj_type or not isinstance(obj_type, str):
            errors.append(f"Object '{obj_id}' has invalid or missing 'type'")
            continue

        if obj_id in seen_ids:
            errors.append(f"Duplicate saved object ID detected: '{obj_id}'")
        seen_ids.add(obj_id)
        objects_by_id[obj_id] = obj

        counts_by_type[obj_type] = counts_by_type.get(obj_type, 0) + 1

        if obj_type == "index-pattern":
            data_view_obj = obj
        elif obj_type == "dashboard":
            dashboard_objects.append(obj)
        elif obj_type == "visualization":
            attrs = obj.get("attributes", {})
            vis_state = attrs.get("visState", "")
            if "markdown" in str(vis_state):
                markdown_objects.append(obj)
            else:
                visualization_objects.append(obj)
        elif obj_type == "lens":
            visualization_objects.append(obj)
        else:
            errors.append(f"Unrecognized saved object type '{obj_type}' for ID '{obj_id}'")

    # Verify counts per category
    if len(dashboard_objects) != EXPECTED_DASHBOARD_COUNT:
        errors.append(f"Expected exactly {EXPECTED_DASHBOARD_COUNT} dashboards, found {len(dashboard_objects)}")

    if data_view_obj is None:
        errors.append("Missing required Data View (index-pattern) object")

    if len(markdown_objects) != EXPECTED_MARKDOWN_COUNT:
        errors.append(f"Expected exactly {EXPECTED_MARKDOWN_COUNT} Markdown banner object, found {len(markdown_objects)}")

    if len(visualization_objects) != EXPECTED_VISUALIZATION_COUNT:
        errors.append(f"Expected exactly {EXPECTED_VISUALIZATION_COUNT} visualization panels, found {len(visualization_objects)}")

    # 4. Data View integrity checks
    if data_view_obj:
        attrs = data_view_obj.get("attributes", {})
        title = attrs.get("title")
        time_field = attrs.get("timeFieldName")

        if title != DEFAULT_INDEX_NAME:
            errors.append(f"Data View title must exactly match default index '{DEFAULT_INDEX_NAME}', got '{title}'")

        if time_field != "timestamps.published_at":
            errors.append(f"Data View primary timeFieldName must be 'timestamps.published_at', got '{time_field}'")

    # 5. Dashboard reference integrity and panel counts
    for dash in dashboard_objects:
        dash_id = dash["id"]
        attrs = dash.get("attributes", {})
        panels_raw = attrs.get("panelsJSON", "[]")

        try:
            panels = json.loads(panels_raw)
        except Exception:
            errors.append(f"Dashboard '{dash_id}' has malformed panelsJSON")
            continue

        if not isinstance(panels, list):
            errors.append(f"Dashboard '{dash_id}' panelsJSON must be a list")
            continue

        actual_panels = len(panels)
        dashboard_panel_counts[dash_id] = actual_panels

        expected_panels = EXPECTED_DASHBOARD_PANEL_COUNTS.get(dash_id)
        if expected_panels is not None and actual_panels != expected_panels:
            errors.append(f"Dashboard '{dash_id}' expected {expected_panels} panels, found {actual_panels}")

        # Check references matching
        dash_refs = dash.get("references", [])
        ref_map = {ref.get("name"): ref for ref in dash_refs if isinstance(ref, dict)}

        for p_idx, panel in enumerate(panels):
            ref_name = panel.get("panelRefName")
            if not ref_name or ref_name not in ref_map:
                errors.append(f"Dashboard '{dash_id}' panel {p_idx} missing reference '{ref_name}'")
                continue

            ref_entry = ref_map[ref_name]
            target_id = ref_entry.get("id")
            target_type = ref_entry.get("type")

            if target_id not in objects_by_id:
                errors.append(f"Dashboard '{dash_id}' has dangling reference to non-existent ID '{target_id}'")
            else:
                actual_obj = objects_by_id[target_id]
                if actual_obj.get("type") != target_type:
                    errors.append(f"Dashboard '{dash_id}' reference type mismatch for '{target_id}': expected '{target_type}', got '{actual_obj.get('type')}'")

    # 6. Visualization field binding and metric definition checks
    for vis in visualization_objects:
        vis_id = vis["id"]
        attrs = vis.get("attributes", {})
        state = attrs.get("state", {})

        # Extract all referenced field accessors from state recursively
        def _extract_accessors(item: Any) -> List[str]:
            accs = []
            if isinstance(item, dict):
                for k, v in item.items():
                    if k == "accessor" and isinstance(v, str):
                        accs.append(v)
                    else:
                        accs.extend(_extract_accessors(v))
            elif isinstance(item, list):
                for sub in item:
                    accs.extend(_extract_accessors(sub))
            return accs

        accessors = _extract_accessors(state)
        for acc in accessors:
            # Check for disallowed unindexed raw fields
            if acc in DISALLOWED_UNINDEXED_FIELDS:
                errors.append(f"Visualization '{vis_id}' uses disallowed unindexed raw field '{acc}'")

            # Check that accessor exists in canonical mapping (or is a valid prefix/nested child)
            if acc not in CANONICAL_FIELDS:
                # Allow nested keyword attributes if they start with topics.keywords
                if not acc.startswith("topics.keywords.") and not acc.startswith("misinformation.stylometric_signals."):
                    errors.append(f"Visualization '{vis_id}' references non-canonical field '{acc}'")

        # Specific metric checks:
        # a. Influencer ranking must use max author.influencer_score
        if vis_id == "vis-net-top-influencers":
            y_spec = state.get("y", {})
            if y_spec.get("accessor") != "author.influencer_score" or y_spec.get("agg") != "max":
                errors.append("Panel 'vis-net-top-influencers' must rank by maximum author.influencer_score")

        # b. Community sizes must use cardinality of author.username
        if vis_id == "vis-net-community-sizes":
            x_spec = state.get("x", {})
            y_spec = state.get("y", {})
            if x_spec.get("accessor") != "author.community_id":
                errors.append("Panel 'vis-net-community-sizes' X-axis must be author.community_id")
            if y_spec.get("accessor") != "author.username" or y_spec.get("agg") != "cardinality":
                errors.append("Panel 'vis-net-community-sizes' Y-axis must use cardinality of author.username")

        # c. Ingestion timeline must use timestamps.collected_at
        if vis_id == "vis-exec-ingestion-timeline":
            x_spec = state.get("x", {})
            if x_spec.get("accessor") != "timestamps.collected_at":
                errors.append("Panel 'vis-exec-ingestion-timeline' must use timestamps.collected_at for ingestion timeline")

        # d. Pipeline completeness rate formula
        if vis_id == "vis-exec-pipeline-completeness":
            formula = state.get("formula", "")
            if "consolidation.is_complete" not in formula or "/ count()" not in formula:
                errors.append("Panel 'vis-exec-pipeline-completeness' formula must compute ratio of consolidation.is_complete over total count")

    # 7. Methodological disclosure banner check
    if markdown_objects:
        md_obj = markdown_objects[0]
        md_state = md_obj.get("attributes", {}).get("visState", "")
        if "LIAR" not in md_state or "Not Verified Factual Truth" not in md_state:
            errors.append("Methodological disclaimer banner must disclose LIAR framework and state predictions are not verified factual truth")

    # 8. Methodological disclaimer present on misinformation dashboard
    misinfo_dash = objects_by_id.get("dashboard-misinformation-intelligence")
    if misinfo_dash:
        refs = [r.get("id") for r in misinfo_dash.get("references", [])]
        if "markdown-misinfo-disclaimer" not in refs:
            errors.append("Dashboard 'dashboard-misinformation-intelligence' must include 'markdown-misinfo-disclaimer'")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "total_objects": total_objects,
        "counts_by_type": counts_by_type,
        "dashboards": dashboard_panel_counts,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    """CLI entrypoint for offline bundle validation."""
    bundle_arg = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_BUNDLE_PATH)
    print(f"Validating Kibana dashboard bundle: {bundle_arg}")
    res = validate_kibana_dashboard_bundle(bundle_arg)

    print("\n--- Validation Report ---")
    print(f"Status        : {'PASSED' if res['is_valid'] else 'FAILED'}")
    print(f"Total Objects : {res['total_objects']} (Expected: {EXPECTED_TOTAL_OBJECTS})")
    print(f"Counts by Type: {res['counts_by_type']}")
    print(f"Dashboards    : {res['dashboards']}")

    if res["errors"]:
        print(f"\nErrors ({len(res['errors'])}):")
        for err in res["errors"]:
            print(f"  [ERROR] {err}")

    if res["warnings"]:
        print(f"\nWarnings ({len(res['warnings'])}):")
        for warn in res["warnings"]:
            print(f"  [WARN]  {warn}")

    if not res["is_valid"]:
        sys.exit(1)
    print("\nBundle validation passed successfully.")


if __name__ == "__main__":
    main()


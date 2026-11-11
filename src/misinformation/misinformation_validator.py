"""Misinformation Artifact Schema Validator (Stage 9).

Validates Stage 9 misinformation and credibility analysis artifacts against Schema v2.
Enforces strict semantic versioning, source provenance contracts, finite numeric metrics,
stable identity integrity, prediction counts, evidence-basis enums, methodology disclosures,
and Stage 10 consolidation compatibility.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

SCHEMA_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)$")

ALLOWED_STAGE_STATUSES = {"success", "partial", "empty", "error"}
ALLOWED_SOURCE_STATUSES = {"loaded", "empty", "missing", "corrupted"}
ALLOWED_RECORD_STATUSES = {"success", "fallback", "empty_text", "error"}
ALLOWED_RISK_TIERS = {"low", "moderate", "high_risk", "unknown"}
ALLOWED_PLATFORMS = {"reddit", "youtube"}
ALLOWED_EVIDENCE_BASES = {"low_risk", "potential_risk", "pattern_flagged_risk", "insufficient_evidence"}


def _is_strict_bool(val: Any) -> bool:
    """Returns True if val is strictly a Python boolean."""
    return type(val) is bool


def _is_strict_int(val: Any) -> bool:
    """Returns True if val is strictly a Python integer (excluding booleans)."""
    return type(val) is int and not isinstance(val, bool)


def _is_finite_float_in_range(val: Any, min_val: float, max_val: float) -> bool:
    """Returns True if val is a finite number in [min_val, max_val], excluding booleans."""
    if _is_strict_bool(val):
        return False
    if isinstance(val, (int, float)):
        return math.isfinite(val) and min_val <= float(val) <= max_val
    return False


class MisinformationArtifactValidator:
    """Strict Schema-v2 validator for Stage 9 misinformation analysis artifacts."""

    @classmethod
    def validate_misinformation_artifact(cls, payload: Any) -> Tuple[bool, List[str]]:
        """Validates a misinformation analysis output payload against Schema v2.

        Args:
            payload: Parsed JSON artifact dictionary.

        Returns:
            Tuple of (is_valid: bool, errors: list[str]).
        """
        errors: List[str] = []
        if not isinstance(payload, dict):
            return False, [f"Top-level payload must be a dict, got {type(payload).__name__}"]

        # 1. Schema version validation
        version = payload.get("schema_version")
        if version is None:
            errors.append("missing_schema_version: 'schema_version' is required")
        elif not isinstance(version, str):
            errors.append(f"malformed_schema_version: 'schema_version' must be string, got {type(version).__name__}")
        else:
            match = SCHEMA_VERSION_PATTERN.match(version)
            if not match:
                errors.append(f"malformed_schema_version: '{version}' does not match pattern ^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)$")
            else:
                major = int(match.group(1))
                if major < 2:
                    errors.append(f"legacy_schema_version: major version {major} < 2 is deprecated")
                elif major >= 3:
                    errors.append(f"unsupported_schema_version: major version {major} >= 3 is not supported")

        # 2. Status validation
        status = payload.get("status")
        if not isinstance(status, str) or status not in ALLOWED_STAGE_STATUSES:
            errors.append(f"'status' must be one of {sorted(list(ALLOWED_STAGE_STATUSES))}, got {status!r}")

        # 3. Generated at timestamp
        gen_at = payload.get("generated_at")
        if not isinstance(gen_at, str) or not gen_at.strip():
            errors.append("'generated_at' must be a non-empty ISO-8601 timestamp string")

        # 4. Provenance validation
        prov = payload.get("provenance")
        prov_data: Optional[Dict[str, Any]] = None
        if not isinstance(prov, dict):
            errors.append(f"'provenance' must be a dict, got {type(prov).__name__}")
        else:
            prov_errs, prov_data = cls._validate_provenance(prov)
            errors.extend(prov_errs)

        # 5. Extract and validate record lists
        reddit_records = payload.get("reddit_records")
        youtube_records = payload.get("youtube_records")
        predictions = payload.get("predictions")

        if not isinstance(reddit_records, list):
            errors.append(f"'reddit_records' must be a list, got {type(reddit_records).__name__}")
        else:
            errors.extend(cls._validate_reddit_records(reddit_records))

        if not isinstance(youtube_records, list):
            errors.append(f"'youtube_records' must be a list, got {type(youtube_records).__name__}")
        else:
            errors.extend(cls._validate_youtube_records(youtube_records))

        if not isinstance(predictions, list):
            errors.append(f"'predictions' must be a list, got {type(predictions).__name__}")
        else:
            errors.extend(cls._validate_predictions(predictions))

        # 6. Numeric metrics validation
        total_proc = payload.get("total_records_processed")
        if not _is_strict_int(total_proc) or total_proc < 0:
            errors.append(f"'total_records_processed' must be a non-negative int, got {total_proc!r}")
        elif isinstance(reddit_records, list) and isinstance(youtube_records, list):
            expected_total = len(reddit_records) + len(youtube_records)
            if total_proc != expected_total:
                errors.append(
                    f"'total_records_processed' ({total_proc}) does not match sum of reddit "
                    f"and youtube records ({expected_total})"
                )

        # Invariant 4.1: Prediction count must match total_records_processed
        if isinstance(predictions, list) and _is_strict_int(total_proc):
            if len(predictions) != total_proc:
                errors.append(
                    f"'predictions' count ({len(predictions)}) does not match 'total_records_processed' ({total_proc})"
                )

        # Invariant 4.1: Prediction identifier correspondence & uniqueness
        if isinstance(reddit_records, list) and isinstance(youtube_records, list) and isinstance(predictions, list):
            expected_keys: Set[Tuple[str, str]] = set()
            for r in reddit_records:
                if isinstance(r, dict) and isinstance(r.get("post_id"), str) and r["post_id"].strip():
                    expected_keys.add(("reddit", r["post_id"].strip()))
            for y in youtube_records:
                if isinstance(y, dict) and isinstance(y.get("video_id"), str) and y["video_id"].strip():
                    expected_keys.add(("youtube", y["video_id"].strip()))

            seen_pred_keys: Set[Tuple[str, str]] = set()
            for idx, p in enumerate(predictions):
                if isinstance(p, dict):
                    plat = p.get("platform")
                    pid = p.get("id")
                    if isinstance(plat, str) and isinstance(pid, str) and plat.strip() and pid.strip():
                        key = (plat.strip(), pid.strip())
                        if key in seen_pred_keys:
                            errors.append(f"predictions[{idx}]: duplicate prediction identifier '{plat}:{pid}'")
                        seen_pred_keys.add(key)

            if _is_strict_int(total_proc) and len(expected_keys) == total_proc:
                missing_in_preds = expected_keys - seen_pred_keys
                extra_in_preds = seen_pred_keys - expected_keys
                if missing_in_preds:
                    errors.append(f"predictions missing records: {sorted(list(missing_in_preds))[:5]}")
                if extra_in_preds:
                    errors.append(f"predictions contain unexpected records: {sorted(list(extra_in_preds))[:5]}")

        total_ana = payload.get("total_analyzed")
        if not _is_strict_int(total_ana) or total_ana < 0:
            errors.append(f"'total_analyzed' must be a non-negative int, got {total_ana!r}")
        elif _is_strict_int(total_proc) and total_ana > total_proc:
            errors.append(f"'total_analyzed' ({total_ana}) cannot exceed 'total_records_processed' ({total_proc})")

        avg_cred = payload.get("average_credibility_score")
        if not _is_finite_float_in_range(avg_cred, 0.0, 1.0):
            errors.append(f"'average_credibility_score' must be finite float in [0.0, 1.0], got {avg_cred!r}")

        flagged_count = payload.get("flagged_high_risk_count")
        if not _is_strict_int(flagged_count) or flagged_count < 0:
            errors.append(f"'flagged_high_risk_count' must be a non-negative int, got {flagged_count!r}")

        # 7. Distributions validation
        risk_dist = payload.get("risk_tier_distribution")
        if not isinstance(risk_dist, dict):
            errors.append(f"'risk_tier_distribution' must be a dict, got {type(risk_dist).__name__}")
        else:
            for tier in ALLOWED_RISK_TIERS:
                if tier not in risk_dist:
                    errors.append(f"risk_tier_distribution: missing key '{tier}'")
                elif not _is_strict_int(risk_dist[tier]) or risk_dist[tier] < 0:
                    errors.append(f"risk_tier_distribution['{tier}'] must be non-negative int, got {risk_dist[tier]!r}")
            if _is_strict_int(total_proc):
                sum_tiers = sum(v for v in risk_dist.values() if _is_strict_int(v))
                if sum_tiers != total_proc:
                    errors.append(f"risk_tier_distribution sum ({sum_tiers}) does not equal total_records_processed ({total_proc})")

        # Invariant 4.3: Veracity distribution validation
        ver_dist = payload.get("veracity_distribution")
        if not isinstance(ver_dist, dict):
            errors.append(f"'veracity_distribution' must be a dict, got {type(ver_dist).__name__}")
        else:
            sum_veracity = 0
            has_invalid_ver_count = False
            for k, v in ver_dist.items():
                if not _is_strict_int(v) or v < 0:
                    errors.append(f"veracity_distribution['{k}'] must be non-negative int, got {v!r}")
                    has_invalid_ver_count = True
                else:
                    sum_veracity += v
            if not has_invalid_ver_count and _is_strict_int(total_proc):
                if sum_veracity != total_proc:
                    errors.append(
                        f"veracity_distribution sum ({sum_veracity}) does not equal total_records_processed ({total_proc})"
                    )

        plat_break = payload.get("platform_breakdown")
        if not isinstance(plat_break, dict):
            errors.append(f"'platform_breakdown' must be a dict, got {type(plat_break).__name__}")
        else:
            for p_name in ("reddit", "youtube"):
                if p_name not in plat_break:
                    errors.append(f"platform_breakdown: missing platform '{p_name}'")
                elif not isinstance(plat_break[p_name], dict):
                    errors.append(f"platform_breakdown['{p_name}'] must be a dict, got {type(plat_break[p_name]).__name__}")

        # 8. Methodology disclosure validation
        disc = payload.get("methodology_disclosure")
        if disc is None:
            errors.append("missing_methodology_disclosure: 'methodology_disclosure' is required")
        elif not isinstance(disc, (dict, str)):
            errors.append(f"'methodology_disclosure' must be a dict or string, got {type(disc).__name__}")

        # 9. Status and provenance consistency
        if isinstance(status, str) and prov_data is not None:
            r_st = prov_data.get("reddit_status")
            y_st = prov_data.get("youtube_status")
            r_cnt = prov_data.get("reddit_records_count", 0)
            y_cnt = prov_data.get("youtube_records_count", 0)

            if status == "success":
                if r_st not in {"loaded", "empty"} or y_st not in {"loaded", "empty"}:
                    errors.append(f"status='success' requires loaded/empty sources, got reddit={r_st!r}, youtube={y_st!r}")
            elif status == "partial":
                if not (r_st in {"missing", "corrupted"} or y_st in {"missing", "corrupted"}):
                    errors.append(f"status='partial' requires at least one missing/corrupted source, got reddit={r_st!r}, youtube={y_st!r}")
            elif status == "empty":
                if total_proc != 0:
                    errors.append(f"status='empty' requires total_records_processed=0, got {total_proc}")
                if r_cnt != 0 or y_cnt != 0:
                    errors.append(f"status='empty' requires 0 provenance records, got reddit={r_cnt}, youtube={y_cnt}")

        return (len(errors) == 0, errors)

    @classmethod
    def _validate_provenance(cls, prov: Dict[str, Any]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """Validates provenance dictionary structure, types, and count consistency."""
        errors: List[str] = []
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
            errors.append(f"provenance.reddit_input_file must be non-empty string, got {r_file!r}")

        y_file = prov.get("youtube_input_file")
        if "youtube_input_file" in prov and (not isinstance(y_file, str) or not y_file.strip()):
            errors.append(f"provenance.youtube_input_file must be non-empty string, got {y_file!r}")

        r_status = prov.get("reddit_status")
        if "reddit_status" in prov and (not isinstance(r_status, str) or r_status not in ALLOWED_SOURCE_STATUSES):
            errors.append(f"provenance.reddit_status must be one of {sorted(list(ALLOWED_SOURCE_STATUSES))}, got {r_status!r}")

        y_status = prov.get("youtube_status")
        if "youtube_status" in prov and (not isinstance(y_status, str) or y_status not in ALLOWED_SOURCE_STATUSES):
            errors.append(f"provenance.youtube_status must be one of {sorted(list(ALLOWED_SOURCE_STATUSES))}, got {y_status!r}")

        r_count = prov.get("reddit_records_count")
        if "reddit_records_count" in prov:
            if not _is_strict_int(r_count) or r_count < 0:
                errors.append(f"provenance.reddit_records_count must be non-negative integer, got {r_count!r}")

        y_count = prov.get("youtube_records_count")
        if "youtube_records_count" in prov:
            if not _is_strict_int(y_count) or y_count < 0:
                errors.append(f"provenance.youtube_records_count must be non-negative integer, got {y_count!r}")

        # Consistency checks
        if isinstance(r_status, str) and _is_strict_int(r_count):
            if r_status == "loaded" and r_count < 1:
                errors.append(f"provenance: reddit_status='loaded' requires reddit_records_count >= 1, got {r_count}")
            elif r_status in {"empty", "missing", "corrupted"} and r_count != 0:
                errors.append(f"provenance: reddit_status={r_status!r} requires reddit_records_count == 0, got {r_count}")

        if isinstance(y_status, str) and _is_strict_int(y_count):
            if y_status == "loaded" and y_count < 1:
                errors.append(f"provenance: youtube_status='loaded' requires youtube_records_count >= 1, got {y_count}")
            elif y_status in {"empty", "missing", "corrupted"} and y_count != 0:
                errors.append(f"provenance: youtube_status={y_status!r} requires youtube_records_count == 0, got {y_count}")

        return errors, prov if len(errors) == 0 else None

    @classmethod
    def _validate_reddit_records(cls, records: List[Any]) -> List[str]:
        """Validates reddit_records array for required keys, types, and uniqueness."""
        errors: List[str] = []
        seen_ids: Set[str] = set()

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                errors.append(f"reddit_records[{idx}]: expected dict, got {type(rec).__name__}")
                continue

            pid = rec.get("post_id")
            if not isinstance(pid, str) or not pid.strip():
                errors.append(f"reddit_records[{idx}]: missing or invalid 'post_id'")
            else:
                clean_pid = pid.strip()
                if clean_pid in seen_ids:
                    errors.append(f"reddit_records[{idx}]: duplicate post_id '{clean_pid}'")
                seen_ids.add(clean_pid)

            errs = cls._validate_common_record_fields(rec, f"reddit_records[{idx}]")
            errors.extend(errs)

        return errors

    @classmethod
    def _validate_youtube_records(cls, records: List[Any]) -> List[str]:
        """Validates youtube_records array for required keys, types, and uniqueness."""
        errors: List[str] = []
        seen_ids: Set[str] = set()

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                errors.append(f"youtube_records[{idx}]: expected dict, got {type(rec).__name__}")
                continue

            vid = rec.get("video_id")
            if not isinstance(vid, str) or not vid.strip():
                errors.append(f"youtube_records[{idx}]: missing or invalid 'video_id'")
            else:
                clean_vid = vid.strip()
                if clean_vid in seen_ids:
                    errors.append(f"youtube_records[{idx}]: duplicate video_id '{clean_vid}'")
                seen_ids.add(clean_vid)

            errs = cls._validate_common_record_fields(rec, f"youtube_records[{idx}]")
            errors.extend(errs)

        return errors

    @classmethod
    def _validate_common_record_fields(cls, rec: Dict[str, Any], prefix: str) -> List[str]:
        """Validates common analytical fields on an enriched record."""
        errors: List[str] = []

        veracity = rec.get("veracity_label")
        if not isinstance(veracity, str) or not veracity.strip():
            errors.append(f"{prefix}: 'veracity_label' must be a non-empty string, got {veracity!r}")

        cred = rec.get("credibility_score")
        if not _is_finite_float_in_range(cred, 0.0, 1.0):
            errors.append(f"{prefix}: 'credibility_score' must be finite float in [0.0, 1.0], got {cred!r}")

        tier = rec.get("risk_tier")
        if not isinstance(tier, str) or tier not in ALLOWED_RISK_TIERS:
            errors.append(f"{prefix}: 'risk_tier' must be one of {sorted(list(ALLOWED_RISK_TIERS))}, got {tier!r}")

        # Invariant 4.2: Evidence-basis validation
        eb = rec.get("evidence_basis")
        if eb is None:
            errors.append(f"{prefix}: missing required field 'evidence_basis'")
        elif not isinstance(eb, str):
            errors.append(f"{prefix}: 'evidence_basis' must be string, got {type(eb).__name__}")
        elif eb not in ALLOWED_EVIDENCE_BASES:
            errors.append(f"{prefix}: 'evidence_basis' must be one of {sorted(list(ALLOWED_EVIDENCE_BASES))}, got {eb!r}")

        # Check misinformation_status or status
        status = rec.get("misinformation_status", rec.get("status"))
        if not isinstance(status, str) or status not in ALLOWED_RECORD_STATUSES:
            errors.append(f"{prefix}: 'misinformation_status' must be one of {sorted(list(ALLOWED_RECORD_STATUSES))}, got {status!r}")

        # Stylometric signals
        styles = rec.get("stylometric_signals")
        if styles is not None and not isinstance(styles, dict):
            errors.append(f"{prefix}: 'stylometric_signals' must be a dict if present, got {type(styles).__name__}")
        elif isinstance(styles, dict):
            for k, v in styles.items():
                if _is_strict_bool(v) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    errors.append(f"{prefix}.stylometric_signals['{k}'] must be finite number, got {v!r}")

        return errors

    @classmethod
    def _validate_predictions(cls, predictions: List[Any]) -> List[str]:
        """Validates predictions summary list."""
        errors: List[str] = []
        for idx, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                errors.append(f"predictions[{idx}]: expected dict, got {type(pred).__name__}")
                continue

            pid = pred.get("id")
            if not isinstance(pid, str) or not pid.strip():
                errors.append(f"predictions[{idx}]: missing or invalid 'id'")

            plat = pred.get("platform")
            if not isinstance(plat, str) or plat not in ALLOWED_PLATFORMS:
                errors.append(f"predictions[{idx}]: 'platform' must be in {sorted(list(ALLOWED_PLATFORMS))}, got {plat!r}")

            errs = cls._validate_common_record_fields(pred, f"predictions[{idx}]")
            errors.extend(errs)

        return errors

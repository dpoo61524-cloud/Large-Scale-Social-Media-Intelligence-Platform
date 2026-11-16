"""Lightweight Local Read-Only Bridge Server for Social Media Intelligence Platform.

Phase 3.1 Implementation:
Provides an air-gapped, read-only HTTP bridge between local run artifacts and the
frontend development runtime. Implements two-tier path confinement, dynamic run
discovery, strict artifact allowlisting, loopback-only binding, origin validation,
and preflight OPTIONS handling while enforcing GET-only data access.

Standard library only: zero external dependencies.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse

logger = logging.getLogger("bridge_server")

# Approved Modern Run ID regex: run_YYYYMMDD_HHMMSS_<8hex>
MODERN_RUN_ID_PATTERN = re.compile(r"^run_\d{8}_\d{6}_[a-f0-9]{8}$")

# Server-assigned identifier for legacy historical data
LEGACY_DATASET_ID = "legacy_historical_processed"

# Explicitly authorized artifact filenames (exactly 5 analytical intelligence outputs)
ALLOWED_ARTIFACTS: Set[str] = {
    "consolidated_intelligence.json",
    "graph_analysis_results.json",
    "topic_modeling_results.json",
    "sentiment_analysis_results.json",
    "misinformation_analysis_results.json",
}

# Maximum JSON response payload size (50 MB)
DEFAULT_MAX_PAYLOAD_BYTES = 50 * 1024 * 1024


def is_symlink_or_reparse_point(path: Path | str) -> bool:
    """Checks whether a path is a symlink or Windows junction/reparse point.

    Inspection boundaries:
    - Path.is_symlink(): detects POSIX symlinks and Windows symlinks.
    - os.stat().st_file_attributes & 0x0400: inspects NTFS reparse point flag
      (FILE_ATTRIBUTE_REPARSE_POINT) to detect Windows directory junctions
      (IO_REPARSE_TAG_MOUNT_POINT), symlinks, and volume mounts.
    - os.path.islink(): fallback check for runtime-supported links.
    """
    p = Path(path)
    try:
        if p.is_symlink():
            return True
    except (OSError, ValueError):
        pass

    try:
        st = os.stat(str(p), follow_symlinks=False)
        if hasattr(st, "st_file_attributes") and bool(st.st_file_attributes & 0x0400):
            return True
    except (OSError, ValueError):
        pass

    try:
        if os.path.islink(str(p)):
            return True
    except (OSError, ValueError):
        pass

    return False


class BridgeSecurityError(Exception):
    """Raised when a security boundary or containment check is violated."""

    def __init__(self, code: str, message: str, status_code: int = HTTPStatus.FORBIDDEN) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class BridgeNotFoundError(Exception):
    """Raised when a requested resource (dataset or artifact) is not found."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = HTTPStatus.NOT_FOUND


class BridgeValidationError(Exception):
    """Raised when artifact contents or manifest fail structural validation."""

    def __init__(self, code: str, message: str, status_code: int = HTTPStatus.UNPROCESSABLE_ENTITY) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# Authoritative physical repository root derived from bridge_server.py location
AUTHORITATIVE_REPO_ROOT: Path = Path(__file__).resolve().parents[2].resolve()


class BridgeConfig:
    """Configuration container for the read-only bridge server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        runs_root: Optional[Path | str] = None,
        legacy_root: Optional[Path | str] = None,
        repo_root: Optional[Path | str] = None,
        allowed_origins: Optional[Set[str]] = None,
        allowed_hosts: Optional[Set[str]] = None,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        # Enforce loopback binding
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(f"Bridge must bind to loopback interface only; rejected: {host}")

        self.host = host
        self.port = port

        # Authoritative repository boundary enforcement (Correction B)
        authoritative = AUTHORITATIVE_REPO_ROOT.resolve()
        if repo_root is not None:
            resolved_repo = Path(repo_root).resolve()
            if resolved_repo != authoritative:
                raise BridgeSecurityError(
                    code="invalid_repo_root",
                    message=(
                        f"Configured repo_root '{resolved_repo}' does not match "
                        f"authoritative repository root '{authoritative}'."
                    ),
                    status_code=HTTPStatus.FORBIDDEN,
                )
            self.repo_root = resolved_repo
        else:
            self.repo_root = authoritative

        raw_runs = Path(runs_root or "data/runs")
        raw_legacy = Path(legacy_root or "data/processed")

        self._raw_runs_path = raw_runs if raw_runs.is_absolute() else (self.repo_root / raw_runs)
        self._raw_legacy_path = raw_legacy if raw_legacy.is_absolute() else (self.repo_root / raw_legacy)

        self.runs_root = self._raw_runs_path.resolve()
        self.legacy_root = self._raw_legacy_path.resolve()

        self.allowed_origins = allowed_origins or {
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        }
        # Strict Host port matching and validation (Correction A)
        self.allowed_hosts = self._validate_allowed_hosts(allowed_hosts)
        self.max_payload_bytes = max_payload_bytes

        # Validate approved data roots at initialization
        self._validate_data_roots()

    def _validate_allowed_hosts(self, allowed_hosts: Optional[Set[str]]) -> Set[str]:
        """Validates and returns the strict set of permitted Host header values.

        Enforcement rules:
        - Only loopback hosts bound to the exact configured port are permitted:
          '127.0.0.1:<port>' and 'localhost:<port>'.
        - Caller-supplied sets are strictly validated: bare hostnames, wrong ports,
          external domains, wildcards, empty entries, and malformed strings are rejected.
        - Invalid configurations raise BridgeSecurityError immediately (no silent filtering).
        """
        valid_loopback_hosts = {
            f"127.0.0.1:{self.port}",
            f"localhost:{self.port}",
        }

        if allowed_hosts is None:
            return valid_loopback_hosts

        if not isinstance(allowed_hosts, (set, list, tuple)):
            raise BridgeSecurityError(
                code="invalid_allowed_hosts",
                message=f"allowed_hosts must be a collection of strings; received {type(allowed_hosts).__name__}.",
                status_code=HTTPStatus.FORBIDDEN,
            )

        if not allowed_hosts:
            raise BridgeSecurityError(
                code="invalid_allowed_hosts",
                message="allowed_hosts cannot be empty; must include loopback host with configured port.",
                status_code=HTTPStatus.FORBIDDEN,
            )

        validated: Set[str] = set()
        for entry in allowed_hosts:
            if not isinstance(entry, str) or not entry.strip():
                raise BridgeSecurityError(
                    code="invalid_allowed_hosts",
                    message=f"Invalid or empty Host entry in allowed_hosts: {entry!r}.",
                    status_code=HTTPStatus.FORBIDDEN,
                )
            cleaned = entry.strip().lower()
            if cleaned not in valid_loopback_hosts:
                raise BridgeSecurityError(
                    code="invalid_allowed_hosts",
                    message=(
                        f"Disallowed host entry '{entry}'. Only loopback addresses with configured "
                        f"port {self.port} are permitted ({sorted(valid_loopback_hosts)})."
                    ),
                    status_code=HTTPStatus.FORBIDDEN,
                )
            validated.add(cleaned)

        return validated

    def _validate_data_roots(self) -> None:
        """Validates that approved data roots cannot escape repository through symlinks or junctions."""
        for name, raw_p, resolved_p in [
            ("runs_root", self._raw_runs_path, self.runs_root),
            ("legacy_root", self._raw_legacy_path, self.legacy_root),
        ]:
            if not raw_p.exists():
                continue

            # 1. Reject symlinks and Windows junctions / reparse points
            if is_symlink_or_reparse_point(raw_p):
                raise BridgeSecurityError(
                    code="unsafe_data_root",
                    message=f"Configured data root '{name}' is a symlink or reparse point, which is prohibited.",
                    status_code=HTTPStatus.FORBIDDEN,
                )

            # 2. Reject roots that resolve outside repository root
            try:
                repo_str = str(self.repo_root)
                if os.path.commonpath([repo_str, str(resolved_p)]) != repo_str:
                    raise BridgeSecurityError(
                        code="data_root_escapes_repo",
                        message=f"Configured data root '{name}' resolves outside the repository root.",
                        status_code=HTTPStatus.FORBIDDEN,
                    )
            except ValueError:
                raise BridgeSecurityError(
                    code="data_root_escapes_repo",
                    message=f"Configured data root '{name}' resides on a different drive than repository root.",
                    status_code=HTTPStatus.FORBIDDEN,
                )


class DatasetEntry:
    """Metadata container for a validated, discovered dataset directory."""

    def __init__(
        self,
        dataset_id: str,
        dataset_type: str,  # 'modern' | 'legacy'
        directory_path: Path,
        label: str,
        generated_at: Optional[str] = None,
        status: str = "unknown",
        stages: Optional[Dict[str, Any]] = None,
        available_artifacts: Optional[List[str]] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_type = dataset_type
        self.directory_path = directory_path
        self.label = label
        self.generated_at = generated_at
        self.status = status
        self.stages = stages or {}
        self.available_artifacts = available_artifacts or []

    def to_summary_dict(self) -> Dict[str, Any]:
        """Returns safe summary dictionary for API serialization (no physical paths)."""
        return {
            "id": self.dataset_id,
            "type": self.dataset_type,
            "label": self.label,
            "generated_at": self.generated_at,
            "status": self.status,
            "stages": self.stages,
            "available_artifacts": self.available_artifacts,
        }


class DatasetRegistry:
    """Manages dynamic discovery, validation, and two-tier confinement for datasets."""

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config

    def discover_datasets(self) -> Dict[str, DatasetEntry]:
        """Discovers, validates, and registers available modern and legacy datasets.

        Enforces Tier 1 Confinement:
        - Candidate run directories must resolve strictly within config.runs_root.
        - Directory names must match MODERN_RUN_ID_PATTERN.
        - Symlinks are rejected.
        - Required manifest (pipeline_execution_manifest.json) must exist and be valid JSON.
        - Legacy directory must resolve strictly within config.legacy_root.
        """
        registry: Dict[str, DatasetEntry] = {}

        # 1. Discover Modern Runs
        runs_root = self.config.runs_root
        if (
            runs_root.exists()
            and runs_root.is_dir()
            and not is_symlink_or_reparse_point(self.config._raw_runs_path)
            and self._is_contained(runs_root, self.config.repo_root)
        ):
            for entry in os.scandir(runs_root):
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue

                    # Check for symlink or Windows junction/reparse point
                    entry_path = Path(entry.path)
                    if is_symlink_or_reparse_point(entry_path):
                        logger.warning("Rejected symlink or reparse point in runs root: %s", entry.name)
                        continue

                    # Validate run ID pattern
                    if not MODERN_RUN_ID_PATTERN.match(entry.name):
                        logger.warning("Skipping directory with invalid run ID format: %s", entry.name)
                        continue

                    # Tier 1 Confinement Check: resolved path must reside strictly in runs_root
                    resolved_dir = entry_path.resolve()
                    if not self._is_contained(resolved_dir, runs_root):
                        logger.warning("Tier 1 confinement violation for run: %s", entry.name)
                        continue

                    # Validate required manifest
                    manifest_path = resolved_dir / "pipeline_execution_manifest.json"
                    if (
                        not manifest_path.exists()
                        or not manifest_path.is_file()
                        or is_symlink_or_reparse_point(manifest_path)
                    ):
                        logger.warning("Run %s missing valid manifest file", entry.name)
                        continue

                    try:
                        with open(manifest_path, "r", encoding="utf-8") as mf:
                            manifest_data = json.load(mf)
                    except Exception as exc:
                        logger.warning("Failed to parse manifest for %s: %s", entry.name, exc)
                        continue

                    if not isinstance(manifest_data, dict):
                        logger.warning("Malformed manifest (not a dict) in run %s", entry.name)
                        continue

                    # Extract manifest metadata
                    gen_at = manifest_data.get("timestamp_utc")
                    outcome = manifest_data.get("overall_outcome") or "completed"
                    raw_stages = manifest_data.get("stages") or {}
                    stages_summary: Dict[str, str] = {}
                    if isinstance(raw_stages, dict):
                        for s_name, s_info in raw_stages.items():
                            if isinstance(s_info, dict):
                                stages_summary[s_name] = s_info.get("status") or "unknown"

                    # Discover available allowlisted artifacts in this directory
                    available_artifacts: List[str] = []
                    for art_name in sorted(ALLOWED_ARTIFACTS):
                        art_file = resolved_dir / art_name
                        if (
                            art_file.exists()
                            and art_file.is_file()
                            and not is_symlink_or_reparse_point(art_file)
                        ):
                            available_artifacts.append(art_name)

                    entry_obj = DatasetEntry(
                        dataset_id=entry.name,
                        dataset_type="modern",
                        directory_path=resolved_dir,
                        label=f"Modern Run {entry.name}",
                        generated_at=gen_at,
                        status=outcome,
                        stages=stages_summary,
                        available_artifacts=available_artifacts,
                    )
                    registry[entry.name] = entry_obj

                except Exception as exc:
                    logger.error("Error inspecting run entry %s: %s", getattr(entry, "name", "unknown"), exc)

        # 2. Register Historical Legacy Baseline
        legacy_root = self.config.legacy_root
        if (
            legacy_root.exists()
            and legacy_root.is_dir()
            and not is_symlink_or_reparse_point(self.config._raw_legacy_path)
            and self._is_contained(legacy_root, self.config.repo_root)
        ):
            resolved_legacy = legacy_root.resolve()
            if self._is_contained(resolved_legacy, legacy_root):
                legacy_consolidated = resolved_legacy / "consolidated_intelligence.json"
                if (
                    legacy_consolidated.exists()
                    and legacy_consolidated.is_file()
                    and not is_symlink_or_reparse_point(legacy_consolidated)
                ):
                    available_legacy_artifacts: List[str] = []
                    for art_name in sorted(ALLOWED_ARTIFACTS):
                        art_file = resolved_legacy / art_name
                        if (
                            art_file.exists()
                            and art_file.is_file()
                            and not is_symlink_or_reparse_point(art_file)
                        ):
                            available_legacy_artifacts.append(art_name)

                    legacy_entry = DatasetEntry(
                        dataset_id=LEGACY_DATASET_ID,
                        dataset_type="legacy",
                        directory_path=resolved_legacy,
                        label="Historical Baseline (Pre-Schema-v2 / Nov 6 2024)",
                        generated_at=None,
                        status="legacy",
                        stages={"legacy_baseline": "available"},
                        available_artifacts=available_legacy_artifacts,
                    )
                    registry[LEGACY_DATASET_ID] = legacy_entry

        return registry

    def get_dataset(self, dataset_id: str) -> DatasetEntry:
        """Retrieves a single dataset entry by ID, refreshing registry if needed."""
        datasets = self.discover_datasets()
        if dataset_id not in datasets:
            raise BridgeNotFoundError(
                code="dataset_not_found",
                message=f"Dataset identifier '{dataset_id}' does not exist or is inaccessible.",
            )
        return datasets[dataset_id]

    def read_artifact_json(self, dataset_id: str, artifact_name: str) -> Tuple[Dict[str, Any], int]:
        """Resolves, validates, and reads an allowlisted artifact JSON file.

        Enforces Tier 2 Confinement:
        - artifact_name must be in ALLOWED_ARTIFACTS.
        - Physical path must resolve strictly within the validated dataset directory.
        - Symlinks are rejected.
        - File must be regular file and within max_payload_bytes.
        - Contents must be valid JSON.
        """
        dataset = self.get_dataset(dataset_id)

        # 1. Artifact allowlist enforcement
        if artifact_name not in ALLOWED_ARTIFACTS:
            raise BridgeSecurityError(
                code="artifact_forbidden",
                message=f"Requested artifact '{artifact_name}' is not on the authorized allowlist.",
                status_code=HTTPStatus.FORBIDDEN,
            )

        # 2. Tier 2 Confinement: check path traversal
        raw_artifact_path = dataset.directory_path / artifact_name

        # Prevent traversal in artifact_name string
        if ".." in artifact_name or "/" in artifact_name or "\\" in artifact_name:
            raise BridgeSecurityError(
                code="invalid_artifact_path",
                message="Path traversal in artifact filename is strictly prohibited.",
                status_code=HTTPStatus.FORBIDDEN,
            )

        resolved_artifact = raw_artifact_path.resolve()

        if not self._is_contained(resolved_artifact, dataset.directory_path):
            raise BridgeSecurityError(
                code="tier2_confinement_violation",
                message="Artifact path escapes dataset directory boundary.",
                status_code=HTTPStatus.FORBIDDEN,
            )

        # Check for symlink or Windows junction/reparse point
        if is_symlink_or_reparse_point(raw_artifact_path) or is_symlink_or_reparse_point(resolved_artifact):
            raise BridgeSecurityError(
                code="symlink_forbidden",
                message="Symlinks and reparse points are not permitted for analytical artifacts.",
                status_code=HTTPStatus.FORBIDDEN,
            )

        # Check existence
        if not resolved_artifact.exists() or not resolved_artifact.is_file():
            raise BridgeNotFoundError(
                code="artifact_unavailable",
                message=f"Artifact '{artifact_name}' was not produced or is unavailable in dataset '{dataset_id}'.",
            )

        # Check file size limit
        file_size = resolved_artifact.stat().st_size
        if file_size > self.config.max_payload_bytes:
            raise BridgeSecurityError(
                code="payload_too_large",
                message=f"Artifact size ({file_size} bytes) exceeds maximum permitted limit.",
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )

        # Read and validate JSON
        try:
            with open(resolved_artifact, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise BridgeValidationError(
                code="malformed_artifact_json",
                message=f"Artifact '{artifact_name}' contains malformed JSON data.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        # Special validation: Modern consolidated intelligence must pass Schema-v2 guard
        if dataset.dataset_type == "modern" and artifact_name == "consolidated_intelligence.json":
            self._validate_schema_v2_envelope(data, dataset_id)

        return data, file_size

    def _validate_schema_v2_envelope(self, data: Any, dataset_id: str) -> None:
        """Validates that a modern consolidated artifact satisfies Schema-v2 envelope requirements."""
        if not isinstance(data, dict):
            raise BridgeValidationError(
                code="schema_v2_violation",
                message="Consolidated artifact must be a JSON object.",
            )

        schema_version = data.get("schema_version")
        if schema_version != "2.0":
            raise BridgeValidationError(
                code="schema_version_mismatch",
                message=f"Expected Schema-v2 artifact (schema_version='2.0'), found '{schema_version}'.",
            )

        required_keys = {"schema_version", "status", "metrics", "records"}
        missing = required_keys - set(data.keys())
        if missing:
            raise BridgeValidationError(
                code="schema_keys_missing",
                message=f"Consolidated envelope missing required Schema-v2 keys: {sorted(missing)}",
            )

    @staticmethod
    def _is_contained(child: Path, parent: Path) -> bool:
        """Verifies that child path is strictly contained within parent path."""
        try:
            return os.path.commonpath([str(parent.resolve()), str(child.resolve())]) == str(parent.resolve())
        except ValueError:
            # Handles different drives on Windows (e.g. C: vs D:)
            return False


class BridgeRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler implementing security controls, CORS, and GET-only data access."""

    # Set by server instance
    config: BridgeConfig
    registry: DatasetRegistry

    # Suppress default stderr logging during tests unless desired
    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def do_OPTIONS(self) -> None:
        """Handles browser CORS preflight requests with safe HTTP 204 response."""
        # 1. Host header validation
        if not self._validate_host_header():
            return

        # 2. Origin validation
        origin = self.headers.get("Origin", "")
        if origin and origin not in self.config.allowed_origins:
            self._send_json_error(
                HTTPStatus.FORBIDDEN,
                "unauthorized_origin",
                "Origin is not authorized to access this resource.",
            )
            return

        # 3. Respond with 204 No Content and static CORS headers
        allow_origin = origin if origin in self.config.allowed_origins else list(self.config.allowed_origins)[0]
        cors_headers = {
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Accept, Content-Type",
            "Access-Control-Max-Age": "86400",
            "Content-Length": "0",
        }
        self.send_response(HTTPStatus.NO_CONTENT)
        for h, v in cors_headers.items():
            self.send_header(h, v)
        self.end_headers()

    def do_GET(self) -> None:
        """Handles read-only GET requests with strict routing and validation."""
        # 1. Host header validation
        if not self._validate_host_header():
            return

        # 2. Origin validation
        origin = self.headers.get("Origin", "")
        if origin and origin not in self.config.allowed_origins:
            self._send_json_error(
                HTTPStatus.FORBIDDEN,
                "unauthorized_origin",
                "Origin is not authorized to access this resource.",
            )
            return

        # 3. Route dispatch
        parsed = urllib.parse.urlparse(self.path)
        raw_path = urllib.parse.unquote(parsed.path)

        # Enforce no directory traversal in raw path
        if "/../" in raw_path or raw_path.endswith("/..") or "\\..\\" in raw_path or raw_path.endswith("\\.."):
            self._send_json_error(
                HTTPStatus.FORBIDDEN,
                "path_traversal_prohibited",
                "Path traversal sequences are strictly rejected.",
            )
            return

        path_parts = [p for p in raw_path.strip("/").split("/") if p]

        try:
            # Route: GET /
            if not path_parts:
                self._send_json_response(
                    HTTPStatus.OK,
                    {
                        "service": "Social Media Intelligence Platform Bridge API",
                        "status": "running",
                        "frontend_dashboard": "http://localhost:5173",
                        "message": "Bridge API is active. Visit http://localhost:5173 in your browser to view the interactive dashboard."
                    }
                )
                return

            # Route: GET /api/v1/health
            if path_parts == ["api", "v1", "health"]:
                self._handle_health()
                return

            # Route: GET /api/v1/datasets
            if path_parts == ["api", "v1", "datasets"]:
                self._handle_list_datasets()
                return

            # Route: GET /api/v1/datasets/<dataset_id>
            if len(path_parts) == 4 and path_parts[:3] == ["api", "v1", "datasets"]:
                dataset_id = path_parts[3]
                self._handle_get_dataset(dataset_id)
                return

            # Route: GET /api/v1/datasets/<dataset_id>/artifacts/<artifact_name>
            if len(path_parts) == 6 and path_parts[:3] == ["api", "v1", "datasets"] and path_parts[4] == "artifacts":
                dataset_id = path_parts[3]
                artifact_name = path_parts[5]
                self._handle_get_artifact(dataset_id, artifact_name)
                return

            # Unmatched route
            self._send_json_error(
                HTTPStatus.NOT_FOUND,
                "route_not_found",
                "The requested endpoint does not exist.",
            )

        except BridgeNotFoundError as exc:
            self._send_json_error(exc.status_code, exc.code, exc.message)
        except BridgeSecurityError as exc:
            self._send_json_error(exc.status_code, exc.code, exc.message)
        except BridgeValidationError as exc:
            self._send_json_error(exc.status_code, exc.code, exc.message)
        except Exception as exc:
            logger.exception("Unexpected server error: %s", exc)
            self._send_json_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "An unexpected server error occurred.",
            )

    # Reject all write and pipeline-execution HTTP methods
    def do_POST(self) -> None:
        self._reject_unsupported_method("POST")

    def do_PUT(self) -> None:
        self._reject_unsupported_method("PUT")

    def do_DELETE(self) -> None:
        self._reject_unsupported_method("DELETE")

    def do_PATCH(self) -> None:
        self._reject_unsupported_method("PATCH")

    def do_HEAD(self) -> None:
        self._reject_unsupported_method("HEAD")

    def _reject_unsupported_method(self, method: str) -> None:
        """Rejects non-GET/non-OPTIONS HTTP methods with 405 Method Not Allowed."""
        self._send_json_error(
            HTTPStatus.METHOD_NOT_ALLOWED,
            "method_not_allowed",
            f"HTTP method '{method}' is not permitted. Data bridge is strictly read-only (GET).",
            extra_headers={"Allow": "GET, OPTIONS"},
        )

    def _validate_host_header(self) -> bool:
        """Validates HTTP Host header against explicitly permitted loopback host values."""
        host_header = self.headers.get("Host", "").strip().lower()
        if not host_header:
            self._send_json_error(
                HTTPStatus.BAD_REQUEST,
                "missing_host_header",
                "HTTP Host header is required.",
            )
            return False

        if host_header not in {h.lower() for h in self.config.allowed_hosts}:
            logger.warning("Rejected disallowed Host header: '%s'", host_header)
            self._send_json_error(
                HTTPStatus.FORBIDDEN,
                "invalid_host_header",
                "Host header is not permitted. Must bind to loopback address with configured port.",
            )
            return False

        return True

    def _handle_health(self) -> None:
        """Responds to health check endpoint."""
        payload = {
            "status": "ok",
            "service": "social-intelligence-bridge",
            "mode": "read_only",
            "version": "1.0.0",
        }
        self._send_json_response(HTTPStatus.OK, payload)

    def _handle_list_datasets(self) -> None:
        """Discovers and returns list of accessible datasets."""
        datasets = self.registry.discover_datasets()
        summary_list = [entry.to_summary_dict() for entry in datasets.values()]
        # Sort so modern runs appear chronologically descending, legacy at the end
        summary_list.sort(key=lambda d: (d["type"] != "modern", d.get("generated_at") or ""), reverse=True)
        self._send_json_response(HTTPStatus.OK, {"datasets": summary_list, "total_datasets": len(summary_list)})

    def _handle_get_dataset(self, dataset_id: str) -> None:
        """Returns details for a specific dataset."""
        dataset = self.registry.get_dataset(dataset_id)
        self._send_json_response(HTTPStatus.OK, dataset.to_summary_dict())

    def _handle_get_artifact(self, dataset_id: str, artifact_name: str) -> None:
        """Resolves, reads, and returns JSON content for an allowlisted artifact."""
        data, file_size = self.registry.read_artifact_json(dataset_id, artifact_name)
        self._send_json_response(HTTPStatus.OK, data)

    def _send_json_response(
        self,
        status_code: int,
        payload: Any,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Serializes and sends JSON response with security and CORS headers."""
        try:
            body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        except Exception as exc:
            logger.error("JSON serialization error: %s", exc)
            self._send_json_error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "serialization_error",
                "Failed to serialize response payload.",
            )
            return

        origin = self.headers.get("Origin", "")
        allow_origin = origin if origin in self.config.allowed_origins else list(self.config.allowed_origins)[0]

        headers: Dict[str, str] = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Accept, Content-Type",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        }

        if extra_headers:
            headers.update(extra_headers)

        self.send_response(status_code)
        for h, v in headers.items():
            self.send_header(h, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(
        self,
        status_code: int,
        error_code: str,
        message: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Sends sanitized JSON error response without exposing host paths or stack traces."""
        error_payload = {
            "error": error_code,
            "message": message,
            "status": status_code,
        }
        try:
            body = json.dumps(error_payload, indent=2).encode("utf-8")
        except Exception:
            body = b'{"error":"unknown_error","message":"An error occurred."}'

        origin = self.headers.get("Origin", "")
        allow_origin = origin if origin in self.config.allowed_origins else list(self.config.allowed_origins)[0]

        headers: Dict[str, str] = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Accept, Content-Type",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        }

        if extra_headers:
            headers.update(extra_headers)

        self.send_response(status_code)
        for h, v in headers.items():
            self.send_header(h, v)
        self.end_headers()
        self.wfile.write(body)


def create_bridge_server(config: Optional[BridgeConfig] = None) -> HTTPServer:
    """Factory creating and configuring the bridge HTTPServer instance."""
    cfg = config or BridgeConfig()
    registry = DatasetRegistry(cfg)

    class CustomHandler(BridgeRequestHandler):
        pass

    CustomHandler.config = cfg
    CustomHandler.registry = registry

    server = HTTPServer((cfg.host, cfg.port), CustomHandler)
    return server


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)

    cfg = BridgeConfig(host="127.0.0.1", port=port)
    server = create_bridge_server(cfg)
    logger.info("Starting Social Intelligence Local Read-Only Bridge on http://%s:%d", cfg.host, cfg.port)
    logger.info("Allowed origins: %s", sorted(cfg.allowed_origins))
    logger.info("Runs root: %s", cfg.runs_root)
    logger.info("Legacy root: %s", cfg.legacy_root)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down bridge server...")
        server.server_close()


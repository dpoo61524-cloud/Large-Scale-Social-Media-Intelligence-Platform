"""Kibana Saved Objects Importer Module.

Provides an optional, self-contained HTTP importer for deploying the canonical
social intelligence dashboard bundle (config/kibana/social_intelligence_dashboards.ndjson)
directly into Kibana 8.x via the supported Saved Objects Import API:
    POST <kibana_host>/api/saved_objects/_import?overwrite=true

Security & Operational Invariants:
1. Requires explicit caller or CLI invocation (never connects on import).
2. Uses standard library urllib.request (zero additional external dependencies).
3. Strictly redacts passwords, tokens, and Authorization headers from all logs,
   exception messages, and return summaries.
4. Validates the bundle offline using kibana_dashboard_validator before attempting import.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config.settings import load_config
from src.search.kibana_dashboard_validator import (
    DEFAULT_BUNDLE_PATH,
    validate_kibana_dashboard_bundle,
)

logger = logging.getLogger(__name__)


def import_kibana_dashboard_bundle(
    bundle_path: Optional[str | Path] = None,
    kibana_host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    overwrite: bool = True,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Imports the dashboard saved-objects NDJSON bundle into Kibana.

    Args:
        bundle_path: Path to the .ndjson bundle. Defaults to DEFAULT_BUNDLE_PATH.
        kibana_host: Target Kibana base URL (e.g. 'http://localhost:5601').
                     Defaults to config.settings.kibana_host.
        username: Optional username for HTTP Basic authentication.
        password: Optional password for HTTP Basic authentication (redacted in output).
        overwrite: Whether to overwrite existing saved objects with matching IDs.
        timeout: Request timeout in seconds.

    Returns:
        Structured dictionary reporting the import outcome:
        {
            "status": "success" | "failed",
            "kibana_host": str,
            "bundle_path": str,
            "success_count": int,
            "errors_count": int,
            "diagnostics": List[str],
        }
    """
    target_bundle = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH
    diagnostics = []

    # 1. Pre-import offline validation
    val_res = validate_kibana_dashboard_bundle(target_bundle)
    if not val_res["is_valid"]:
        logger.error("Pre-import bundle validation failed with %d errors", len(val_res["errors"]))
        return {
            "status": "failed",
            "kibana_host": str(kibana_host or ""),
            "bundle_path": str(target_bundle),
            "success_count": 0,
            "errors_count": len(val_res["errors"]),
            "diagnostics": [f"Pre-import validation error: {e}" for e in val_res["errors"]],
        }

    # 2. Resolve Kibana host and credentials
    cfg = load_config()
    target_host = (kibana_host or cfg.kibana_host or "http://localhost:5601").rstrip("/")
    auth_user = username if username is not None else cfg.elasticsearch_username
    auth_pass = password if password is not None else cfg.elasticsearch_password

    # Sanitize host: ensure valid scheme
    if not target_host.startswith(("http://", "https://")):
        return {
            "status": "failed",
            "kibana_host": target_host,
            "bundle_path": str(target_bundle),
            "success_count": 0,
            "errors_count": 1,
            "diagnostics": ["Invalid Kibana host URL: scheme must be http or https"],
        }

    import_url = f"{target_host}/api/saved_objects/_import?overwrite={str(overwrite).lower()}"

    # 3. Read bundle bytes
    try:
        with open(target_bundle, "rb") as f:
            bundle_bytes = f.read()
    except Exception as exc:
        logger.error("Failed to read bundle file: %s", type(exc).__name__)
        return {
            "status": "failed",
            "kibana_host": target_host,
            "bundle_path": str(target_bundle),
            "success_count": 0,
            "errors_count": 1,
            "diagnostics": [f"File read error ({type(exc).__name__})"],
        }

    # 4. Construct multipart/form-data payload
    boundary = f"----KibanaBundleBoundary{uuid.uuid4().hex}"
    body_parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="file"; filename="dashboards.ndjson"\r\n',
        b"Content-Type: application/ndjson\r\n\r\n",
        bundle_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    payload = b"".join(body_parts)

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "kbn-xsrf": "true",
        "Content-Length": str(len(payload)),
    }

    if auth_user and auth_pass:
        cred = f"{auth_user}:{auth_pass}".encode("utf-8")
        headers["Authorization"] = f"Basic {base64.b64encode(cred).decode('ascii')}"

    # 5. Execute HTTP POST request
    req = Request(import_url, data=payload, headers=headers, method="POST")

    try:
        with urlopen(req, timeout=timeout) as resp:
            resp_bytes = resp.read()
            resp_code = resp.getcode()
            try:
                resp_json = json.loads(resp_bytes.decode("utf-8"))
            except Exception:
                resp_json = {}

            success_count = resp_json.get("successCount", 0)
            errors_count = resp_json.get("errors", [])
            err_count = len(errors_count) if isinstance(errors_count, list) else 0

            return {
                "status": "success" if resp_code == 200 and err_count == 0 else "partial_success",
                "kibana_host": target_host,
                "bundle_path": str(target_bundle),
                "success_count": success_count,
                "errors_count": err_count,
                "diagnostics": [f"Kibana HTTP {resp_code} response: {success_count} objects imported successfully"],
            }
    except HTTPError as exc:
        logger.error("Kibana Saved Objects API returned HTTP %d", exc.code)
        # Sanitize error detail to prevent secret reflection
        return {
            "status": "failed",
            "kibana_host": target_host,
            "bundle_path": str(target_bundle),
            "success_count": 0,
            "errors_count": 1,
            "diagnostics": [f"Kibana Saved Objects API returned HTTP {exc.code}"],
        }
    except URLError as exc:
        logger.error("Failed to connect to Kibana host at %s: %s", target_host, type(exc).__name__)
        return {
            "status": "failed",
            "kibana_host": target_host,
            "bundle_path": str(target_bundle),
            "success_count": 0,
            "errors_count": 1,
            "diagnostics": [f"Connection to Kibana host failed ({type(exc).__name__})"],
        }
    except Exception as exc:
        logger.error("Unexpected error during Kibana dashboard import: %s", type(exc).__name__)
        return {
            "status": "failed",
            "kibana_host": target_host,
            "bundle_path": str(target_bundle),
            "success_count": 0,
            "errors_count": 1,
            "diagnostics": [f"Import error ({type(exc).__name__})"],
        }


def main() -> None:
    """CLI entrypoint for explicit dashboard bundle import."""
    import argparse

    parser = argparse.ArgumentParser(description="Import Kibana social intelligence dashboards.")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE_PATH), help="Path to .ndjson bundle")
    parser.add_argument("--host", default=None, help="Kibana host (e.g. http://localhost:5601)")
    parser.add_argument("--username", default=None, help="Kibana username")
    parser.add_argument("--password", default=None, help="Kibana password")
    args = parser.parse_args()

    print(f"Starting explicit Kibana dashboard import from: {args.bundle}")
    res = import_kibana_dashboard_bundle(
        bundle_path=args.bundle,
        kibana_host=args.host,
        username=args.username,
        password=args.password,
    )

    print("\n--- Import Summary ---")
    print(f"Status        : {res['status'].upper()}")
    print(f"Kibana Host   : {res['kibana_host']}")
    print(f"Success Count : {res['success_count']}")
    print(f"Errors Count  : {res['errors_count']}")
    for diag in res["diagnostics"]:
        print(f"Detail        : {diag}")

    if res["status"] == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()


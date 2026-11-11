"""Elasticsearch Client and Connection Manager for Social Intelligence Platform.

Provides a secure, reusable client for Elasticsearch cluster operations,
health monitoring, index management, and document persistence using
configuration from config.settings and the canonical schema from src.search.index_mapping.

Security Invariants:
- Rejects host URLs with embedded credentials, query strings, fragments (even empty), or non-root paths.
- Rejects host URLs containing any leading, trailing, or internal whitespace.
- Prevents URL path and index name disclosure by using fixed, non-reflective validation error messages.
- Preserves square brackets on IPv6 host URLs and validates IPv4, IPv6, and DNS hostnames.
- Enforces valid port ranges (1–65535).
- Never exposes passwords, tokens, or raw exception strings in logs, exceptions, or __repr__.
- Sanitizes all driver-originated exceptions (including TypeError, ValueError, and construction failures) with suppressed chaining.
- Distinguishes caller input validation errors from driver execution failures.
- Enforces strict configuration validation (positive timeouts, non-negative integer retries, valid index names).
- Modern Elasticsearch 8.x API usage (settings/mappings for create, document for index).
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

from config.settings import AppConfig, load_config
from src.search.index_mapping import (
    DEFAULT_INDEX_NAME,
    get_index_configuration,
)

logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BULK_BATCH_SIZE = 250

# Characters disallowed in Elasticsearch index names
INVALID_INDEX_CHARS = set(r'\/*?"<>| ,#:')

# Regex for validating DNS hostname labels according to RFC 1123
DNS_LABEL_REGEX = re.compile(
    r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$"
)


class ElasticsearchClientError(Exception):
    """Base exception for search and client operations."""


class ElasticsearchConnectionError(ElasticsearchClientError):
    """Raised when cluster connection, ping, or health check fails."""


class ElasticsearchIndexError(ElasticsearchClientError):
    """Raised when index creation, deletion, or validation fails."""


def validate_host_url(raw_host: Any) -> str:
    """Validates, sanitizes, and normalizes an Elasticsearch host URL.

    Enforces:
    - Must be a non-empty string without leading, trailing, or internal whitespace.
    - Must use http or https scheme.
    - Must NOT contain query delimiters ('?') or fragment delimiters ('#'), even if empty.
    - Must NOT contain a non-root path (fixed generic error, no path disclosure).
    - Must NOT contain embedded credentials (username/password).
    - Preserves square brackets on IPv6 addresses (e.g. 'http://[::1]:9200').
    - Validates IPv4, IPv6, and DNS hostname syntax without reflecting input.
    - Validates port numbers (1–65535).

    Args:
        raw_host: Raw host string from configuration or constructor.

    Returns:
        Sanitized, normalized host URL string.

    Raises:
        TypeError: If raw_host is not a string.
        ValueError: If raw_host violates security or URL format rules.
    """
    if not isinstance(raw_host, str):
        raise TypeError(f"Elasticsearch host must be a string, got {type(raw_host).__name__}.")

    if not raw_host:
        raise ValueError("Elasticsearch host cannot be empty.")

    # Reject leading, trailing, or internal whitespace consistently
    if any(c.isspace() for c in raw_host):
        raise ValueError("Elasticsearch host URL cannot contain whitespace.")

    # Reject query and fragment delimiters even if their values are empty
    if "?" in raw_host:
        raise ValueError("Elasticsearch host URL must not contain query parameters ('?').")
    if "#" in raw_host:
        raise ValueError("Elasticsearch host URL must not contain URL fragments ('#').")

    # Validate bracket balance and placement for IPv6 hosts before urlsplit
    if "[" in raw_host or "]" in raw_host:
        if raw_host.count("[") != 1 or raw_host.count("]") != 1 or raw_host.index("[") > raw_host.index("]"):
            raise ValueError("Malformed IPv6 host: missing opening or closing bracket.")

    try:
        parsed = urlsplit(raw_host)
    except ValueError:
        raise ValueError("Malformed Elasticsearch host URL or IPv6 address.") from None

    # 1. Scheme check
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Invalid Elasticsearch host scheme. Host URL must start with 'http://' or 'https://'.")

    # 2. Path check (fixed generic error message to prevent URL path / token disclosure)
    if parsed.path and parsed.path not in {"", "/"}:
        raise ValueError("Elasticsearch host URL must not contain a path. Only root host URLs are supported.")

    # 3. Embedded credentials check (Security requirement)
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise ValueError(
            "Elasticsearch host URL must not contain embedded user credentials. "
            "Configure username and password separately via constructor or settings."
        )

    netloc = parsed.netloc
    if not netloc:
        raise ValueError("Elasticsearch host must include a valid hostname or IP address.")

    if netloc.endswith(":"):
        raise ValueError("Elasticsearch host URL contains an empty port delimiter ':'.")

    # 4. IPv6 handling with bracket preservation
    if "[" in netloc and "]" in netloc:
        closing_bracket_idx = netloc.index("]")
        ip_content = netloc[1:closing_bracket_idx]
        after_bracket = netloc[closing_bracket_idx + 1 :]

        try:
            ipv6_addr = ipaddress.IPv6Address(ip_content)
        except ValueError:
            raise ValueError("Malformed IPv6 address in host URL.") from None

        port = None
        if after_bracket:
            if not after_bracket.startswith(":"):
                raise ValueError("Malformed host: invalid characters following IPv6 closing bracket.")
            port_str = after_bracket[1:]
            if not port_str.isdigit():
                raise ValueError("Elasticsearch port must be an integer.")
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError("Elasticsearch port must be between 1 and 65535.")

        port_suffix = f":{port}" if port is not None else ""
        return f"{scheme}://[{ipv6_addr}]{port_suffix}"

    # 5. IPv4 or DNS Hostname handling
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Elasticsearch host URL contains an invalid port. Port must be an integer between 1 and 65535.")

    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Elasticsearch port must be between 1 and 65535.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Elasticsearch host must include a valid hostname or IP address.")

    # Check if valid IPv4 address
    try:
        ipaddress.IPv4Address(hostname)
        is_ipv4 = True
    except ValueError:
        is_ipv4 = False

    if not is_ipv4:
        # Validate DNS hostname syntax (RFC 1123)
        if len(hostname) > 253:
            raise ValueError("Hostname exceeds maximum length of 253 characters.")
        labels = hostname.split(".")
        for label in labels:
            if not DNS_LABEL_REGEX.match(label):
                raise ValueError("Invalid hostname syntax in Elasticsearch host URL.")

    port_suffix = f":{port}" if port is not None else ""
    return f"{scheme}://{hostname.lower()}{port_suffix}"


def validate_index_name(name: Any) -> str:
    """Validates that an index name conforms to Elasticsearch naming rules.

    Enforces:
    - Must be a non-empty string.
    - Cannot contain leading, trailing, or whitespace characters.
    - Maximum 255 bytes.
    - Lowercase characters only.
    - Cannot start with '-', '_', or '+'.
    - Cannot be '.' or '..'.
    - Cannot contain invalid characters (\\/*?"<>| ,#:)
    - Uses fixed generic error messages to prevent index name disclosure.

    Args:
        name: Index name candidate.

    Returns:
        Validated index name string.

    Raises:
        TypeError: If name is not a string.
        ValueError: If name violates Elasticsearch naming rules.
    """
    if not isinstance(name, str):
        raise TypeError(f"Index name must be a string, got {type(name).__name__}.")

    if not name or not name.strip():
        raise ValueError("Index name cannot be empty or blank.")

    if name != name.strip():
        raise ValueError("Index name cannot contain leading or trailing whitespace.")

    if len(name.encode("utf-8")) > 255:
        raise ValueError("Index name exceeds maximum length of 255 bytes.")

    if name != name.lower():
        raise ValueError("Index name must be lowercase.")

    if name[0] in {"-", "_", "+"}:
        raise ValueError("Index name cannot start with '-', '_', or '+'.")

    if name in {".", ".."}:
        raise ValueError("Index name cannot be '.' or '..'.")

    if any(ch in INVALID_INDEX_CHARS or ch.isspace() for ch in name):
        raise ValueError("Index name contains invalid characters.")

    return name


def _is_connection_error(exc: Exception) -> bool:
    """Determines whether an exception represents a network, connectivity, or timeout error."""
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    exc_name = type(exc).__name__.lower()
    return "connection" in exc_name or "timeout" in exc_name or "unreachable" in exc_name


def _sanitize_cluster_health(raw_health: Any) -> Dict[str, Any]:
    """Extracts and validates only explicitly approved fields from cluster health response.

    Approved fields:
    - status: str (e.g. 'green', 'yellow', 'red')
    - number_of_nodes: non-negative int (excluding bool)
    - active_shards: non-negative int (excluding bool)

    Args:
        raw_health: Raw response from client.cluster.health().

    Returns:
        Sanitized dictionary containing only validated approved fields.
    """
    sanitized: Dict[str, Any] = {}
    if not isinstance(raw_health, dict):
        return sanitized

    raw_status = raw_health.get("status")
    if isinstance(raw_status, str) and raw_status.strip():
        sanitized["status"] = raw_status.strip().lower()

    raw_nodes = raw_health.get("number_of_nodes")
    if isinstance(raw_nodes, int) and not isinstance(raw_nodes, bool) and raw_nodes >= 0:
        sanitized["number_of_nodes"] = raw_nodes

    raw_shards = raw_health.get("active_shards")
    if isinstance(raw_shards, int) and not isinstance(raw_shards, bool) and raw_shards >= 0:
        sanitized["active_shards"] = raw_shards

    return sanitized


class ElasticsearchClient:
    """Manages Elasticsearch connections, health checks, and index operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client_override: Optional[Any] = None,
    ) -> None:
        """Initializes the Elasticsearch client.

        Args:
            host: Elasticsearch host URL (e.g. 'http://localhost:9200').
            username: Optional username for Basic Authentication.
            password: Optional password for Basic Authentication.
            request_timeout: Connection and request timeout in seconds (finite float > 0).
            max_retries: Maximum retry attempts for transient request failures (int >= 0).
            client_override: Optional mock client instance for unit testing.
        """
        # Resolve configuration defaults independently for omitted parameters
        cfg: Optional[AppConfig] = None
        if host is None or username is None or password is None:
            cfg = load_config()

        if host is None:
            raw_host = "http://localhost:9200" if cfg.elasticsearch_host is None else cfg.elasticsearch_host
        else:
            raw_host = host

        if username is None:
            config_user = cfg.elasticsearch_username
            raw_user = "" if config_user is None else config_user
        else:
            raw_user = username

        if password is None:
            config_pw = cfg.elasticsearch_password
            raw_pw = "" if config_pw is None else config_pw
        else:
            raw_pw = password

        # Validate host and enforce credential protection
        self.sanitized_host = validate_host_url(raw_host)
        self.host = self.sanitized_host

        # Validate credentials types (reject False, 0, lists, and non-strings without coercion)
        if not isinstance(raw_user, str):
            raise TypeError(f"Username must be a string, got {type(raw_user).__name__}.")
        if not isinstance(raw_pw, str):
            raise TypeError(f"Password must be a string, got {type(raw_pw).__name__}.")

        self.username = raw_user.strip()
        self._password = raw_pw

        # Validate request timeout (non-reflective error messages, catch OverflowError)
        if isinstance(request_timeout, bool) or not isinstance(request_timeout, (int, float)):
            raise TypeError(f"request_timeout must be a number, got {type(request_timeout).__name__}.")
        try:
            timeout_float = float(request_timeout)
        except OverflowError:
            raise ValueError("request_timeout must be a finite number greater than 0.") from None
        if not math.isfinite(timeout_float) or timeout_float <= 0.0:
            raise ValueError("request_timeout must be a finite number greater than 0.")
        self.request_timeout = timeout_float

        # Validate max retries
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError(f"max_retries must be an integer, got {type(max_retries).__name__}.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be a non-negative integer, got {max_retries}.")
        self.max_retries = max_retries

        self.client_override = client_override
        self._client: Optional[Any] = client_override

    @property
    def is_auth_configured(self) -> bool:
        """Indicates whether authentication credentials are provided."""
        return bool(self.username and self._password)

    def __repr__(self) -> str:
        """Returns non-sensitive representation of the client (secrets masked)."""
        auth_status = "configured" if self.is_auth_configured else "none"
        user_display = repr(self.username) if self.username else "None"
        return (
            f"ElasticsearchClient("
            f"host='{self.sanitized_host}', "
            f"user={user_display}, "
            f"auth={auth_status}, "
            f"timeout={self.request_timeout}s)"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def get_client(self) -> Any:
        """Lazily creates or returns the underlying Elasticsearch driver.

        Returns:
            Configured Elasticsearch client instance.

        Raises:
            RuntimeError: If elasticsearch library is not installed and no override is set.
            ElasticsearchConnectionError: If connection fails during driver initialization.
            ElasticsearchClientError: If driver construction fails.
        """
        if self._client is not None:
            return self._client

        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            logger.error("elasticsearch package is not installed (ImportError)")
            raise RuntimeError(
                "The 'elasticsearch' package is required for live cluster connections. "
                "Install it via 'pip install elasticsearch>=8.0.0' or supply a client_override for testing."
            ) from None

        auth = (self.username, self._password) if self.is_auth_configured else None
        client_kwargs: Dict[str, Any] = {
            "hosts": [self.sanitized_host],
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
        }
        if auth:
            client_kwargs["basic_auth"] = auth

        logger.info("Initializing Elasticsearch connection to %s (auth=%s)", self.sanitized_host, "yes" if auth else "no")
        try:
            self._client = Elasticsearch(**client_kwargs)
            return self._client
        except Exception as exc:
            logger.error("Failed to construct Elasticsearch client for %s (%s)", self.sanitized_host, type(exc).__name__)
            if _is_connection_error(exc):
                raise ElasticsearchConnectionError(
                    f"Connection failed during driver initialization to {self.sanitized_host} ({type(exc).__name__})"
                ) from None
            raise ElasticsearchClientError(
                f"Failed to initialize Elasticsearch client ({type(exc).__name__})"
            ) from None

    def ping(self) -> bool:
        """Checks whether the Elasticsearch cluster is reachable.

        Returns:
            True if cluster responds to ping, False otherwise.

        Raises:
            ElasticsearchConnectionError: If connectivity failure occurs.
            ElasticsearchClientError: If an unexpected non-connectivity error occurs.
        """
        try:
            client = self.get_client()
            return bool(client.ping())
        except (ConnectionError, TimeoutError, OSError) as exc:
            logger.warning("Elasticsearch ping to %s failed (%s)", self.sanitized_host, type(exc).__name__)
            raise ElasticsearchConnectionError(f"Ping connection failed ({type(exc).__name__})") from None
        except ElasticsearchConnectionError:
            raise
        except ElasticsearchClientError:
            raise
        except Exception as exc:
            if _is_connection_error(exc):
                logger.warning("Elasticsearch ping to %s failed (%s)", self.sanitized_host, type(exc).__name__)
                raise ElasticsearchConnectionError(f"Ping connection failed ({type(exc).__name__})") from None
            logger.error("Unexpected error during Elasticsearch ping to %s (%s)", self.sanitized_host, type(exc).__name__)
            raise ElasticsearchClientError(f"Ping failed ({type(exc).__name__})") from None

    def health_check(self) -> Dict[str, Any]:
        """Performs a cluster health check and returns a non-sensitive diagnostic summary.

        Returns:
            Dictionary containing connectivity status and sanitized diagnostics.

        Raises:
            ElasticsearchClientError: If an unexpected programming error occurs.
        """
        try:
            client = self.get_client()
            is_pingable = bool(client.ping())
            if not is_pingable:
                return {
                    "status": "disconnected",
                    "host": self.sanitized_host,
                    "authenticated": self.is_auth_configured,
                    "error": "Cluster ping returned False",
                }

            # Retrieve and sanitize cluster health if available
            cluster_health: Dict[str, Any] = {}
            if hasattr(client, "cluster") and hasattr(client.cluster, "health"):
                cluster_health = _sanitize_cluster_health(client.cluster.health())
            elif hasattr(client, "info"):
                cluster_health = _sanitize_cluster_health(client.info())

            return {
                "status": "connected",
                "host": self.sanitized_host,
                "authenticated": self.is_auth_configured,
                "cluster_health": cluster_health,
            }
        except (ConnectionError, TimeoutError, OSError, ElasticsearchConnectionError) as exc:
            logger.warning("Elasticsearch health check to %s failed (%s)", self.sanitized_host, type(exc).__name__)
            return {
                "status": "disconnected",
                "host": self.sanitized_host,
                "authenticated": self.is_auth_configured,
                "error": f"Health check failed ({type(exc).__name__})",
            }
        except Exception as exc:
            if _is_connection_error(exc):
                logger.warning("Elasticsearch health check to %s failed (%s)", self.sanitized_host, type(exc).__name__)
                return {
                    "status": "disconnected",
                    "host": self.sanitized_host,
                    "authenticated": self.is_auth_configured,
                    "error": f"Health check failed ({type(exc).__name__})",
                }
            logger.error("Unexpected error during Elasticsearch health check to %s (%s)", self.sanitized_host, type(exc).__name__)
            raise ElasticsearchClientError(f"Health check failed ({type(exc).__name__})") from None

    def index_exists(self, index_name: Optional[str] = None) -> bool:
        """Checks if a specified index exists.

        Args:
            index_name: Name of target index (defaults to DEFAULT_INDEX_NAME).

        Returns:
            True if index exists, False otherwise.
        """
        target_index = validate_index_name(DEFAULT_INDEX_NAME if index_name is None else index_name)
        client = self.get_client()
        try:
            return bool(client.indices.exists(index=target_index))
        except Exception as exc:
            logger.error("Failed to check index existence for '%s' (%s)", target_index, type(exc).__name__)
            if _is_connection_error(exc):
                raise ElasticsearchConnectionError(
                    f"Connection failed while checking index '{target_index}' ({type(exc).__name__})"
                ) from None
            if isinstance(exc, TypeError):
                raise TypeError(f"Elasticsearch driver argument error ({type(exc).__name__})") from None
            if isinstance(exc, ValueError):
                raise ValueError(f"Elasticsearch driver value error ({type(exc).__name__})") from None
            raise ElasticsearchIndexError(
                f"Failed to check index existence for '{target_index}' ({type(exc).__name__})"
            ) from None

    def create_index(
        self,
        index_name: Optional[str] = None,
        number_of_shards: int = 1,
        number_of_replicas: int = 0,
        force_recreate: bool = False,
    ) -> Dict[str, Any]:
        """Creates an index using the modern Elasticsearch 8.x client API.

        Args:
            index_name: Target index name (defaults to DEFAULT_INDEX_NAME).
            number_of_shards: Primary shard count (default: 1).
            number_of_replicas: Replica shard count (default: 0).
            force_recreate: If True, deletes existing index before creation.

        Returns:
            Response dictionary from index creation.

        Raises:
            TypeError: If input types are invalid or driver raises argument error.
            ValueError: If input values are invalid.
            ElasticsearchConnectionError: If connection to cluster fails.
            ElasticsearchIndexError: If index creation fails.
        """
        target_index = validate_index_name(DEFAULT_INDEX_NAME if index_name is None else index_name)
        if isinstance(number_of_shards, bool) or not isinstance(number_of_shards, int) or number_of_shards <= 0:
            raise ValueError(f"number_of_shards must be an integer > 0, got {number_of_shards!r}.")
        if isinstance(number_of_replicas, bool) or not isinstance(number_of_replicas, int) or number_of_replicas < 0:
            raise ValueError(f"number_of_replicas must be an integer >= 0, got {number_of_replicas!r}.")

        client = self.get_client()

        if self.index_exists(target_index):
            if force_recreate:
                logger.info("Force recreating index '%s'", target_index)
                self.delete_index(target_index)
            else:
                logger.info("Index '%s' already exists; skipping creation", target_index)
                return {
                    "acknowledged": True,
                    "index": target_index,
                    "created": False,
                    "already_exists": True,
                }

        config = get_index_configuration(
            number_of_shards=number_of_shards,
            number_of_replicas=number_of_replicas,
        )

        logger.info("Creating index '%s' with strict schema mapping", target_index)
        try:
            # Modern Elasticsearch 8.x client API
            res = client.indices.create(
                index=target_index,
                settings=config["settings"],
                mappings=config["mappings"],
            )
            return res if isinstance(res, dict) else {"acknowledged": True, "index": target_index, "created": True}
        except Exception as exc:
            logger.error("Failed to create index '%s' (%s)", target_index, type(exc).__name__)
            if _is_connection_error(exc):
                raise ElasticsearchConnectionError(
                    f"Connection failed while creating index '{target_index}' ({type(exc).__name__})"
                ) from None
            if isinstance(exc, TypeError):
                raise TypeError(f"Elasticsearch driver argument error ({type(exc).__name__})") from None
            if isinstance(exc, ValueError):
                raise ValueError(f"Elasticsearch driver value error ({type(exc).__name__})") from None
            raise ElasticsearchIndexError(
                f"Failed to create index '{target_index}' ({type(exc).__name__})"
            ) from None

    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """Deletes an index if it exists.

        Args:
            index_name: Target index name (defaults to DEFAULT_INDEX_NAME).

        Returns:
            True if deleted, False if index did not exist.
        """
        target_index = validate_index_name(DEFAULT_INDEX_NAME if index_name is None else index_name)
        client = self.get_client()

        if not self.index_exists(target_index):
            logger.info("Index '%s' does not exist; skipping deletion", target_index)
            return False

        try:
            res = client.indices.delete(index=target_index)
            logger.info("Deleted index '%s'", target_index)
            return bool(res.get("acknowledged", True) if isinstance(res, dict) else True)
        except Exception as exc:
            logger.error("Failed to delete index '%s' (%s)", target_index, type(exc).__name__)
            if _is_connection_error(exc):
                raise ElasticsearchConnectionError(
                    f"Connection failed while deleting index '{target_index}' ({type(exc).__name__})"
                ) from None
            if isinstance(exc, TypeError):
                raise TypeError(f"Elasticsearch driver argument error ({type(exc).__name__})") from None
            if isinstance(exc, ValueError):
                raise ValueError(f"Elasticsearch driver value error ({type(exc).__name__})") from None
            raise ElasticsearchIndexError(
                f"Failed to delete index '{target_index}' ({type(exc).__name__})"
            ) from None

    def index_document(
        self,
        document: Dict[str, Any],
        index_name: Optional[str] = None,
        refresh: Optional[Union[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Indexes a single document using modern Elasticsearch 8.x API.

        Args:
            document: Document dictionary conforming to consolidated intelligence schema.
            index_name: Target index name (defaults to DEFAULT_INDEX_NAME).
            refresh: Optional refresh control ('wait_for', True, False).

        Returns:
            Elasticsearch index API response.

        Raises:
            TypeError: If document is not a dict or driver raises argument error.
            ValueError: If document lacks a valid document_id.
            ElasticsearchConnectionError: If network connection fails.
            ElasticsearchIndexError: If indexing operation fails.
        """
        # Caller input validation
        if not isinstance(document, dict):
            raise TypeError(f"Expected document dictionary, got {type(document).__name__}.")

        doc_id = document.get("document_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("Document missing valid 'document_id' string.")

        # Validate refresh parameter before obtaining client or making driver calls
        if refresh is not None:
            if not (isinstance(refresh, bool) or refresh == "wait_for"):
                raise ValueError("Invalid refresh parameter. Accepted values are True, False, 'wait_for', or None.")

        target_index = validate_index_name(DEFAULT_INDEX_NAME if index_name is None else index_name)
        client = self.get_client()

        index_kwargs: Dict[str, Any] = {
            "index": target_index,
            "id": doc_id.strip(),
            "document": document,
        }
        if refresh is not None:
            index_kwargs["refresh"] = refresh

        # Driver execution with sanitized exception handling
        try:
            res = client.index(**index_kwargs)
            return res if isinstance(res, dict) else {"result": "indexed", "_id": doc_id.strip()}
        except Exception as exc:
            logger.error("Failed to index document into '%s' (%s)", target_index, type(exc).__name__)
            if _is_connection_error(exc):
                raise ElasticsearchConnectionError(
                    f"Connection failed while indexing document into '{target_index}' ({type(exc).__name__})"
                ) from None
            if isinstance(exc, TypeError):
                raise TypeError(f"Elasticsearch driver argument error ({type(exc).__name__})") from None
            if isinstance(exc, ValueError):
                raise ValueError(f"Elasticsearch driver value error ({type(exc).__name__})") from None
            raise ElasticsearchIndexError(
                f"Failed to index document into '{target_index}' ({type(exc).__name__})"
            ) from None

    def bulk_index_documents(
        self,
        documents: List[Dict[str, Any]],
        index_name: Optional[str] = None,
        refresh: Optional[Union[str, bool]] = None,
        raise_on_error: bool = False,
    ) -> Dict[str, Any]:
        """Indexes a batch of documents into Elasticsearch with bounded retry and per-document isolation.

        Args:
            documents: List of document dictionaries, each containing a valid 'document_id'.
            index_name: Target index name (defaults to DEFAULT_INDEX_NAME).
            refresh: Optional refresh control ('wait_for', True, False, or None).
            raise_on_error: If True, raises ElasticsearchIndexError or ElasticsearchConnectionError
                when one or more documents fail. If False (default), returns structured per-document
                accounting without raising.

        Returns:
            Dictionary containing:
            - total: Total number of documents submitted in batch.
            - indexed_count: Count of successfully indexed documents.
            - failed_count: Count of failed documents.
            - successful_ids: List of document IDs successfully indexed.
            - failed_items: List of failure dicts with document_id, status code, and error details.
            - errors: Boolean indicating whether any document failed.

        Raises:
            TypeError: If documents is not a list, contains non-dicts, or driver raises argument error.
            ValueError: If any document lacks a valid document_id or refresh is invalid.
            ElasticsearchConnectionError: If network connection fails and retries are exhausted (when raise_on_error=True).
            ElasticsearchIndexError: If driver operations fail (when raise_on_error=True).
        """
        if isinstance(documents, bool) or not isinstance(documents, list):
            raise TypeError(f"Expected list of documents, got {type(documents).__name__}.")

        # Fast path for empty batch
        if not documents:
            return {
                "total": 0,
                "indexed_count": 0,
                "failed_count": 0,
                "successful_ids": [],
                "failed_items": [],
                "errors": False,
            }

        # Validate caller inputs before acquiring client or executing driver calls
        for idx, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(f"Document at index {idx} must be a dictionary, got {type(doc).__name__}.")
            doc_id = doc.get("document_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"Document at index {idx} missing valid 'document_id' string.")

        if refresh is not None:
            if not (isinstance(refresh, bool) or refresh == "wait_for"):
                raise ValueError("Invalid refresh parameter. Accepted values are True, False, 'wait_for', or None.")

        target_index = validate_index_name(DEFAULT_INDEX_NAME if index_name is None else index_name)
        client = self.get_client()

        # Check if driver lacks bulk or if index_document is customized/mocked on instance
        is_custom_index_doc = hasattr(self, "__dict__") and "index_document" in self.__dict__
        has_driver_bulk = hasattr(client, "bulk") and callable(getattr(client, "bulk"))

        if is_custom_index_doc or not has_driver_bulk:
            successful_ids: List[str] = []
            failed_items: List[Dict[str, Any]] = []
            for doc in documents:
                doc_id = doc["document_id"].strip()
                try:
                    self.index_document(document=doc, index_name=target_index, refresh=refresh)
                    successful_ids.append(doc_id)
                except Exception as exc:
                    logger.error("Sequential fallback indexing failed for '%s' (%s)", doc_id, type(exc).__name__)
                    failed_items.append({
                        "document_id": doc_id,
                        "status": getattr(exc, "status_code", 500) if hasattr(exc, "status_code") else 500,
                        "error": f"Indexing operation failed ({type(exc).__name__})",
                    })

            if raise_on_error and failed_items:
                raise ElasticsearchIndexError(
                    f"Bulk indexing failed for {len(failed_items)} document(s) in '{target_index}'."
                )
            return {
                "total": len(documents),
                "indexed_count": len(successful_ids),
                "failed_count": len(failed_items),
                "successful_ids": successful_ids,
                "failed_items": failed_items,
                "errors": bool(failed_items),
            }

        # Driver bulk execution with transient retries and per-document isolation
        successful_ids = []
        failed_items = []
        docs_to_send: List[Dict[str, Any]] = list(documents)
        attempts = 0
        max_attempts = max(1, self.max_retries + 1)
        transient_status_codes = {429, 502, 503, 504}

        while docs_to_send and attempts < max_attempts:
            attempts += 1
            operations: List[Dict[str, Any]] = []
            for doc in docs_to_send:
                doc_id = doc["document_id"].strip()
                operations.append({"index": {"_index": target_index, "_id": doc_id}})
                operations.append(doc)

            bulk_kwargs: Dict[str, Any] = {"operations": operations}
            if refresh is not None:
                bulk_kwargs["refresh"] = refresh

            try:
                try:
                    res = client.bulk(**bulk_kwargs)
                except TypeError as te:
                    # Support drivers/mocks expecting legacy 'body' parameter
                    if "operations" in str(te) or "unexpected keyword argument 'operations'" in str(te):
                        legacy_kwargs = dict(bulk_kwargs)
                        legacy_kwargs["body"] = legacy_kwargs.pop("operations")
                        res = client.bulk(**legacy_kwargs)
                    else:
                        raise
            except Exception as exc:
                if _is_connection_error(exc):
                    logger.warning(
                        "Transient connection failure during bulk index to %s (attempt %d/%d): %s",
                        self.sanitized_host, attempts, max_attempts, type(exc).__name__
                    )
                    if attempts < max_attempts:
                        continue
                    if raise_on_error:
                        raise ElasticsearchConnectionError(
                            f"Connection failed during bulk indexing to '{target_index}' ({type(exc).__name__})"
                        ) from None
                    for doc in docs_to_send:
                        failed_items.append({
                            "document_id": doc["document_id"].strip(),
                            "status": 503,
                            "error": f"Connection failed ({type(exc).__name__})",
                        })
                    docs_to_send = []
                    break

                # Permanent driver failure (TypeError, ValueError, unhandled client error)
                logger.error("Permanent error during bulk index to %s (%s)", self.sanitized_host, type(exc).__name__)
                if isinstance(exc, TypeError):
                    raise TypeError(f"Elasticsearch driver argument error ({type(exc).__name__})") from None
                if isinstance(exc, ValueError):
                    raise ValueError(f"Elasticsearch driver value error ({type(exc).__name__})") from None
                if raise_on_error:
                    raise ElasticsearchIndexError(
                        f"Bulk indexing operation failed for '{target_index}' ({type(exc).__name__})"
                    ) from None
                for doc in docs_to_send:
                    failed_items.append({
                        "document_id": doc["document_id"].strip(),
                        "status": 500,
                        "error": f"Bulk indexing driver error ({type(exc).__name__})",
                    })
                docs_to_send = []
                break

            # Parse items from response
            items = res.get("items", []) if isinstance(res, dict) else []
            transient_retry_docs: List[Dict[str, Any]] = []

            if not items:
                if isinstance(res, dict) and res.get("errors") is True:
                    for doc in docs_to_send:
                        failed_items.append({
                            "document_id": doc["document_id"].strip(),
                            "status": 500,
                            "error": "Bulk indexing reported unspecified error",
                        })
                else:
                    for doc in docs_to_send:
                        successful_ids.append(doc["document_id"].strip())
                docs_to_send = []
                break

            docs_by_id = {doc["document_id"].strip(): doc for doc in docs_to_send}

            for item_idx, item in enumerate(items):
                action_data = None
                for key in ("index", "create", "update", "delete"):
                    if isinstance(item, dict) and key in item:
                        action_data = item[key]
                        break
                if action_data is None:
                    action_data = item if isinstance(item, dict) else {}

                item_id = action_data.get("_id")
                if not item_id and item_idx < len(docs_to_send):
                    item_id = docs_to_send[item_idx]["document_id"].strip()

                status_code = action_data.get("status", 200)

                if 200 <= status_code < 300:
                    if item_id:
                        successful_ids.append(item_id)
                elif status_code in transient_status_codes and attempts < max_attempts:
                    doc_obj = docs_by_id.get(item_id) if item_id else None
                    if doc_obj is not None:
                        transient_retry_docs.append(doc_obj)
                    else:
                        err = action_data.get("error", {})
                        reason = err.get("reason", str(err)) if isinstance(err, dict) else str(err)
                        failed_items.append({
                            "document_id": item_id or "unknown",
                            "status": status_code,
                            "error": reason,
                        })
                else:
                    err = action_data.get("error", {})
                    reason = err.get("reason", str(err)) if isinstance(err, dict) else str(err)
                    failed_items.append({
                        "document_id": item_id or "unknown",
                        "status": status_code,
                        "error": reason,
                    })

            docs_to_send = transient_retry_docs

        # Any documents remaining after retry exhaustion
        for doc in docs_to_send:
            failed_items.append({
                "document_id": doc["document_id"].strip(),
                "status": 503,
                "error": "Transient retry attempts exhausted",
            })

        if raise_on_error and failed_items:
            raise ElasticsearchIndexError(
                f"Bulk indexing failed for {len(failed_items)} document(s) in '{target_index}'."
            )

        return {
            "total": len(documents),
            "indexed_count": len(successful_ids),
            "failed_count": len(failed_items),
            "successful_ids": successful_ids,
            "failed_items": failed_items,
            "errors": bool(failed_items),
        }

    def close(self) -> None:
        """Closes the underlying client connection if open."""
        if self._client is not None:
            if hasattr(self._client, "close") and callable(self._client.close):
                try:
                    self._client.close()
                except Exception as exc:
                    logger.warning("Error closing Elasticsearch client (%s)", type(exc).__name__)
            self._client = None


def create_elasticsearch_client(
    config: Optional[AppConfig] = None,
    client_override: Optional[Any] = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ElasticsearchClient:
    """Factory function for creating an ElasticsearchClient instance.

    Args:
        config: Optional AppConfig object. If omitted, loads via load_config().
        client_override: Optional mock driver for offline testing.
        request_timeout: Timeout in seconds.
        max_retries: Maximum retries for transient connection errors.

    Returns:
        Configured ElasticsearchClient instance.
    """
    cfg = config or load_config()
    return ElasticsearchClient(
        host=cfg.elasticsearch_host,
        username=cfg.elasticsearch_username,
        password=cfg.elasticsearch_password,
        request_timeout=request_timeout,
        max_retries=max_retries,
        client_override=client_override,
    )

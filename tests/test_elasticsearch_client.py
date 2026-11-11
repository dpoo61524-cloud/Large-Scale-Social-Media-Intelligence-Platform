"""Unit Tests for Elasticsearch Client and Connection Manager.

Verifies:
- Rejection of host URLs with embedded credentials, query strings, fragments (even empty), or non-root paths.
- Proper IPv6 URL reconstruction with preserved brackets and strict IPv6 format validation.
- Validation of IPv4, IPv6, and DNS hostnames and port bounds (1–65535).
- Non-exposure of credentials in __repr__, __str__, logs, health check, or exceptions.
- Sanitization of driver-originated exceptions (TypeError, ValueError, ConnectionError, construction errors) with suppressed chaining.
- Distinction between caller input validation and driver execution errors.
- Strict configuration validation (timeouts, retries, index names).
- Correct Modern Elasticsearch 8.x client API usage without legacy retries on TypeError.
- Connection health check (ping and diagnostic reporting).
- Safe error handling using custom exception hierarchy (ElasticsearchConnectionError vs ElasticsearchIndexError).
- Non-deletion of existing index when force_recreate=False.
- Driver lifecycle and offline testing using mocks without live cluster dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from config.settings import AppConfig
from src.search.elasticsearch_client import (
    DEFAULT_BULK_BATCH_SIZE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DNS_LABEL_REGEX,
    ElasticsearchClient,
    ElasticsearchClientError,
    ElasticsearchConnectionError,
    ElasticsearchIndexError,
    _sanitize_cluster_health,
    create_elasticsearch_client,
    validate_host_url,
    validate_index_name,
)
from src.search.index_mapping import (
    DEFAULT_INDEX_NAME,
    get_index_mapping,
    get_index_settings,
)


# =====================================================================
# Mock Elasticsearch Driver Factory
# =====================================================================

class MockIndicesClient:
    """Mock for client.indices namespace."""

    def __init__(self) -> None:
        self.exists = MagicMock(return_value=False)
        self.create = MagicMock(return_value={"acknowledged": True, "index": DEFAULT_INDEX_NAME})
        self.delete = MagicMock(return_value={"acknowledged": True})


class MockClusterClient:
    """Mock for client.cluster namespace."""

    def __init__(self) -> None:
        self.health = MagicMock(return_value={"status": "green", "number_of_nodes": 1, "active_shards": 1})


class MockElasticsearchDriver:
    """Mock driver representing the official Elasticsearch client."""

    def __init__(self) -> None:
        self.ping = MagicMock(return_value=True)
        self.info = MagicMock(return_value={"version": {"number": "8.17.0"}})
        self.index = MagicMock(return_value={"result": "created", "_id": "doc_123"})
        self.bulk = MagicMock(return_value={"took": 5, "errors": False, "items": []})
        self.close = MagicMock()
        self.indices = MockIndicesClient()
        self.cluster = MockClusterClient()


@pytest.fixture
def mock_driver() -> MockElasticsearchDriver:
    """Fixture providing a fresh MockElasticsearchDriver."""
    return MockElasticsearchDriver()


@pytest.fixture
def client_with_mock(mock_driver: MockElasticsearchDriver) -> ElasticsearchClient:
    """Fixture providing an ElasticsearchClient wired to the mock driver."""
    return ElasticsearchClient(
        host="http://localhost:9200",
        username="test_user",
        password="secret_password_123",
        client_override=mock_driver,
    )


# =====================================================================
# Host URL Validation & IPv6 / Port / Path Tests
# =====================================================================

@pytest.mark.parametrize(
    "ipv6_input, expected_output",
    [
        ("http://[::1]:9200", "http://[::1]:9200"),
        ("https://[2001:db8::1]:9243", "https://[2001:db8::1]:9243"),
        ("http://[::1]", "http://[::1]"),
        ("https://[::ffff:192.0.2.1]:9200", "https://[::ffff:192.0.2.1]:9200"),
        ("http://[fe80::1]:9200", "http://[fe80::1]:9200"),
    ],
)
def test_ipv6_url_reconstruction_preserves_brackets(ipv6_input: str, expected_output: str):
    """Verifies IPv6 host URLs preserve square brackets and correctly normalize."""
    assert validate_host_url(ipv6_input) == expected_output
    client = ElasticsearchClient(host=ipv6_input)
    assert client.host == expected_output
    assert client.sanitized_host == expected_output


@pytest.mark.parametrize(
    "malformed_ipv6",
    [
        "http://[zzz::1]:9200",
        "http://[::1:9200",
        "http://::1]:9200",
        "http://[2001:db8:::1]:9200",
        "http://[::1]trailing:9200",
        "http://[::1]:abc",
    ],
)
def test_malformed_ipv6_rejected(malformed_ipv6: str):
    """Verifies malformed IPv6 host strings are rejected with ValueError."""
    with pytest.raises(ValueError):
        validate_host_url(malformed_ipv6)


@pytest.mark.parametrize(
    "valid_url, expected_normalized",
    [
        ("http://localhost:9200", "http://localhost:9200"),
        ("http://localhost:9200/", "http://localhost:9200"),
        ("https://127.0.0.1:9200", "https://127.0.0.1:9200"),
        ("https://192.168.1.100:9243", "https://192.168.1.100:9243"),
        ("http://es-cluster.internal.net:9200", "http://es-cluster.internal.net:9200"),
        ("https://node1.elastic.co", "https://node1.elastic.co"),
    ],
)
def test_valid_dns_and_ipv4_urls_accepted(valid_url: str, expected_normalized: str):
    """Verifies compliant DNS and IPv4 URLs are properly validated and normalized."""
    assert validate_host_url(valid_url) == expected_normalized


@pytest.mark.parametrize(
    "invalid_port_url",
    [
        "http://localhost:0",
        "http://localhost:65536",
        "http://localhost:999999",
        "http://localhost:-1",
        "http://localhost:abc",
        "http://localhost:",
        "http://[::1]:0",
        "http://[::1]:65536",
    ],
)
def test_invalid_and_out_of_range_ports_rejected(invalid_port_url: str):
    """Verifies ports outside 1–65535 or non-integer port values are rejected."""
    with pytest.raises(ValueError, match="port|Port"):
        validate_host_url(invalid_port_url)


@pytest.mark.parametrize(
    "empty_delimiter_url",
    [
        "http://localhost:9200?",
        "http://localhost:9200#",
        "http://localhost:9200/?",
        "http://localhost:9200/#",
        "https://127.0.0.1:9200?",
        "https://127.0.0.1:9200#",
    ],
)
def test_query_and_fragment_delimiters_rejected_even_when_empty(empty_delimiter_url: str):
    """Verifies '?' and '#' delimiters are rejected even with empty parameter/fragment strings."""
    with pytest.raises(ValueError, match="query parameters|URL fragments"):
        validate_host_url(empty_delimiter_url)


@pytest.mark.parametrize(
    "non_root_path_url",
    [
        "http://localhost:9200/api",
        "http://localhost:9200/es/",
        "http://localhost:9200/sub/index",
        "https://cluster:9200/path",
    ],
)
def test_non_root_paths_rejected(non_root_path_url: str):
    """Verifies non-root paths are rejected to prevent path traversal or secret leaks."""
    with pytest.raises(ValueError, match="must not contain a path"):
        validate_host_url(non_root_path_url)


def test_secret_token_in_url_path_never_disclosed():
    """Verifies that distinctive secret tokens in non-root URL paths never appear in exception messages."""
    secret_path_token = "secret_api_token_path_leak_xyz_98765"
    url_with_secret_path = f"http://localhost:9200/{secret_path_token}"

    with pytest.raises(ValueError) as exc_info:
        validate_host_url(url_with_secret_path)

    err_msg = str(exc_info.value)
    # Token must NOT be reflected in the exception message
    assert secret_path_token not in err_msg
    # Fixed generic message must be used
    assert err_msg == "Elasticsearch host URL must not contain a path. Only root host URLs are supported."


@pytest.mark.parametrize(
    "whitespace_url, whitespace_type",
    [
        ("  http://localhost:9200", "leading spaces"),
        ("\thttp://localhost:9200", "leading tab"),
        ("http://localhost:9200  ", "trailing spaces"),
        ("http://localhost:9200\n", "trailing newline"),
        ("http://localhost:9200\r\n", "trailing crlf"),
        ("http://local host:9200", "internal whitespace in hostname"),
        ("http://localhost: 9200", "internal whitespace before port"),
        ("  http://localhost:9200  ", "leading and trailing whitespace"),
    ],
)
def test_host_url_whitespace_handling_rejected(whitespace_url: str, whitespace_type: str):
    """Verifies leading, trailing, and internal whitespace are all strictly rejected with consistent message."""
    with pytest.raises(ValueError) as exc_info:
        validate_host_url(whitespace_url)

    assert str(exc_info.value) == "Elasticsearch host URL cannot contain whitespace."


def test_host_url_normalization_preserves_valid_urls():
    """Verifies valid URL normalization is preserved without stripping or whitespace issues."""
    assert validate_host_url("http://localhost:9200") == "http://localhost:9200"
    assert validate_host_url("http://localhost:9200/") == "http://localhost:9200"
    assert validate_host_url("https://127.0.0.1:9200") == "https://127.0.0.1:9200"
    assert validate_host_url("https://127.0.0.1:9200/") == "https://127.0.0.1:9200"
    assert validate_host_url("http://[::1]:9200") == "http://[::1]:9200"
    assert validate_host_url("http://[::1]:9200/") == "http://[::1]:9200"


@pytest.mark.parametrize(
    "invalid_hostname_url",
    [
        "http://local host:9200",     # Whitespace
        "http://local_host:9200",     # Underscore in hostname
        "http://-invalid.com:9200",   # Leading hyphen
        "http://invalid-.com:9200",   # Trailing hyphen
        "http://host..com:9200",      # Empty label
    ],
)
def test_invalid_hostname_syntax_and_whitespace_rejected(invalid_hostname_url: str):
    """Verifies malformed hostname syntax or whitespace is rejected."""
    with pytest.raises(ValueError):
        validate_host_url(invalid_hostname_url)


@pytest.mark.parametrize(
    "valid_label",
    [
        "a",
        "0",
        "abc",
        "node1",
        "es-cluster",
        "a-b",
        "1-2",
        "a" * 63,
        "x" + "0" * 61 + "y",
    ],
)
def test_dns_label_regex_valid_rfc1123_labels(valid_label: str):
    """Verifies DNS_LABEL_REGEX matches RFC 1123 compliant labels."""
    assert DNS_LABEL_REGEX.match(valid_label) is not None


@pytest.mark.parametrize(
    "invalid_label",
    [
        "",
        "-",
        "-leading",
        "trailing-",
        "a_b",
        "a.b",
        "a@b",
        "a" * 64,
        "label with space",
    ],
)
def test_dns_label_regex_invalid_rfc1123_labels(invalid_label: str):
    """Verifies DNS_LABEL_REGEX rejects non-RFC 1123 labels."""
    assert DNS_LABEL_REGEX.match(invalid_label) is None


@pytest.mark.parametrize(
    "valid_dns_host, expected_normalized",
    [
        ("http://localhost:9200", "http://localhost:9200"),
        ("http://localhost", "http://localhost"),
        ("http://example.com:9200", "http://example.com:9200"),
        ("http://example.com", "http://example.com"),
        ("http://my-server.example.com:9200", "http://my-server.example.com:9200"),
        ("https://my-server.example.com", "https://my-server.example.com"),
        ("http://a.co:9200", "http://a.co:9200"),
        ("http://a.co", "http://a.co"),
        ("http://elasticsearch:9200", "http://elasticsearch:9200"),
        ("http://es-master-01:9200", "http://es-master-01:9200"),
        ("http://es-master-01.prod.internal:9200", "http://es-master-01.prod.internal:9200"),
        ("https://node1.elastic.co:9243", "https://node1.elastic.co:9243"),
        ("http://a.b.c.d:9200", "http://a.b.c.d:9200"),
        ("http://1-node.2-dc.com:9200", "http://1-node.2-dc.com:9200"),
        ("http://" + ("a" * 63) + ".com:9200", "http://" + ("a" * 63) + ".com:9200"),
        ("http://" + ("a" * 63) + ".example.com", "http://" + ("a" * 63) + ".example.com"),
    ],
)
def test_valid_rfc1123_dns_hostnames_accepted(valid_dns_host: str, expected_normalized: str):
    """Verifies compliant multi-label and single-label DNS hostnames are accepted."""
    assert validate_host_url(valid_dns_host) == expected_normalized


@pytest.mark.parametrize(
    "invalid_dns_host",
    [
        "http://-invalid.com:9200",
        "http://-invalid.com",
        "http://invalid-.com:9200",
        "http://invalid-.com",
        "http://invalid..com:9200",
        "http://invalid..com",
        "http://" + ("a" * 64) + ".com:9200",
        "http://" + ("a" * 64) + ".example.com",
        "http://invalid_name.com:9200",
        "http://my_server.example.com:9200",
        "http://invalid_underscore.com",
        "http://-node.cluster.net:9200",
        "http://cluster.-sub.net:9200",
        "http://node-.cluster.net:9200",
        "http://cluster.sub-.net:9200",
        "http://node..cluster.net:9200",
        "http://.cluster.net:9200",
        "http://cluster.net.:9200",
        "http://es_node.internal:9200",
        "http://" + ("a" * 60 + "." + "b" * 60 + "." + "c" * 60 + "." + "d" * 60 + "." + "e" * 20) + ":9200",
        "http://es$node.net:9200",
        "http://es!node.net:9200",
    ],
)
def test_invalid_rfc1123_dns_hostnames_rejected(invalid_dns_host: str):
    """Verifies non-compliant RFC 1123 DNS hostnames raise ValueError."""
    with pytest.raises(ValueError):
        validate_host_url(invalid_dns_host)


@pytest.mark.parametrize(
    "credential_url",
    [
        "http://user:password@localhost:9200",
        "https://admin:secret123@my-cluster.es.io:9243",
        "http://user@localhost:9200",
        "http://:password@localhost:9200",
        "https://elastic:super_secret_pw@127.0.0.1:9200",
    ],
)
def test_host_url_with_embedded_credentials_rejected(credential_url: str):
    """Verifies host URLs containing embedded credentials are strictly rejected."""
    with pytest.raises(ValueError, match="embedded user credentials"):
        validate_host_url(credential_url)


# =====================================================================
# Configuration Validation Tests (Timeouts, Retries, Index Names)
# =====================================================================

@pytest.mark.parametrize("invalid_timeout", [0, -1.0, -10, float("nan"), float("inf"), float("-inf")])
def test_invalid_request_timeout_value_rejected(invalid_timeout):
    """Verifies non-positive or non-finite timeout values raise ValueError."""
    with pytest.raises(ValueError, match="greater than 0"):
        ElasticsearchClient(host="http://localhost:9200", request_timeout=invalid_timeout)


@pytest.mark.parametrize("invalid_timeout_type", [True, False, "10", None, [10]])
def test_invalid_request_timeout_type_rejected(invalid_timeout_type):
    """Verifies non-numeric or boolean timeout types raise TypeError."""
    with pytest.raises(TypeError, match="must be a number"):
        ElasticsearchClient(host="http://localhost:9200", request_timeout=invalid_timeout_type)  # type: ignore


@pytest.mark.parametrize("invalid_retries", [-1, -5])
def test_invalid_max_retries_value_rejected(invalid_retries: int):
    """Verifies negative retry values raise ValueError."""
    with pytest.raises(ValueError, match="non-negative integer"):
        ElasticsearchClient(host="http://localhost:9200", max_retries=invalid_retries)


@pytest.mark.parametrize("invalid_retries_type", [True, False, 2.5, "3", None, [3]])
def test_invalid_max_retries_type_rejected(invalid_retries_type):
    """Verifies non-integer or boolean retry types raise TypeError."""
    with pytest.raises(TypeError, match="must be an integer"):
        ElasticsearchClient(host="http://localhost:9200", max_retries=invalid_retries_type)  # type: ignore


@pytest.mark.parametrize(
    "invalid_name, expected_err",
    [
        ("", "cannot be empty"),
        ("   ", "cannot be empty"),
        ("UPPERCASE_NAME", "must be lowercase"),
        ("MixedCaseName", "must be lowercase"),
        ("_starts_with_underscore", "cannot start with"),
        ("-starts_with_dash", "cannot start with"),
        ("+starts_with_plus", "cannot start with"),
        (".", "cannot be '.' or '..'"),
        ("..", "cannot be '.' or '..'"),
        ("contains*asterisk", "invalid characters"),
        ("contains?question", "invalid characters"),
        ("contains space", "invalid characters"),
        ("contains,comma", "invalid characters"),
        ("contains#hash", "invalid characters"),
        ("contains:colon", "invalid characters"),
        ("a" * 256, "exceeds maximum length"),
    ],
)
def test_invalid_index_names_rejected(invalid_name: str, expected_err: str):
    """Verifies non-compliant index names are rejected before API calls."""
    with pytest.raises(ValueError, match=expected_err):
        validate_index_name(invalid_name)


@pytest.mark.parametrize(
    "invalid_index_factory, expected_err_message",
    [
        (lambda secret: f"index_with_{secret.upper()}", "Index name must be lowercase."),
        (lambda secret: f"_{secret}", "Index name cannot start with '-', '_', or '+'."),
        (lambda secret: f"-{secret}", "Index name cannot start with '-', '_', or '+'."),
        (lambda secret: f"+{secret}", "Index name cannot start with '-', '_', or '+'."),
        (lambda secret: f"index_{secret}*token", "Index name contains invalid characters."),
        (lambda secret: f"index_{secret}?token", "Index name contains invalid characters."),
        (lambda secret: f"index_{secret}#token", "Index name contains invalid characters."),
        (lambda secret: f"index_{secret}:token", "Index name contains invalid characters."),
        (lambda secret: f"index_{secret}\\token", "Index name contains invalid characters."),
        (lambda secret: f"index_{secret} token", "Index name contains invalid characters."),
        (lambda secret: f"{secret}_{'a' * 260}", "Index name exceeds maximum length of 255 bytes."),
        (lambda secret: f"  {secret}  ", "Index name cannot contain leading or trailing whitespace."),
    ],
)
def test_secret_token_in_invalid_index_name_never_disclosed(invalid_index_factory, expected_err_message: str):
    """Verifies that distinctive secret strings contained in invalid index names never appear in exception messages."""
    distinctive_secret = "secret_classified_token_xyz999"
    candidate_name = invalid_index_factory(distinctive_secret)

    with pytest.raises(ValueError) as exc_info:
        validate_index_name(candidate_name)

    err_msg = str(exc_info.value)
    # Distinctive secret string must NEVER be reflected in the exception message
    assert distinctive_secret not in err_msg
    assert distinctive_secret.upper() not in err_msg
    # Generic description must match exactly
    assert err_msg == expected_err_message


# =====================================================================
# Driver Exception Leakage Prevention & Sanitization Tests
# =====================================================================

def test_driver_construction_error_sanitized_and_no_secrets_leaked(caplog: pytest.LogCaptureFixture):
    """Verifies exceptions during Elasticsearch(**kwargs) construction sanitize secrets."""
    caplog.set_level(logging.DEBUG)
    secret_token = "bearer_secret_jwt_token_construction"

    class FailingElasticsearchMock:
        def __init__(self, **kwargs):
            raise ConnectionError(f"Handshake failed with Authorization Bearer {secret_token}")

    with patch.dict("sys.modules", {"elasticsearch": MagicMock(Elasticsearch=FailingElasticsearchMock)}):
        client = ElasticsearchClient(host="http://localhost:9200", password="my_pw")
        with pytest.raises(ElasticsearchConnectionError) as exc_info:
            client.get_client()

        # Secret must not appear in raised exception message
        assert secret_token not in str(exc_info.value)
        assert "(ConnectionError)" in str(exc_info.value)

        # Chained cause must be suppressed (from None)
        assert exc_info.value.__cause__ is None

        # Secret must not appear in log output
        assert secret_token not in caplog.text


def test_driver_type_error_in_index_document_sanitized(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    caplog: pytest.LogCaptureFixture,
):
    """Verifies driver TypeError during index_document sanitizes secrets and preserves category."""
    caplog.set_level(logging.DEBUG)
    secret_bearer = "bearer_secret_token_type_error"
    mock_driver.index.side_effect = TypeError(f"Invalid kwargs with Authorization: {secret_bearer}")

    doc = {"document_id": "r_101"}
    with pytest.raises(TypeError) as exc_info:
        client_with_mock.index_document(doc)

    assert secret_bearer not in str(exc_info.value)
    assert "Elasticsearch driver argument error (TypeError)" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert secret_bearer not in caplog.text


def test_driver_value_error_in_index_document_sanitized(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    caplog: pytest.LogCaptureFixture,
):
    """Verifies driver ValueError during index_document sanitizes secrets and preserves category."""
    caplog.set_level(logging.DEBUG)
    secret_password = "password_secret_value_error"
    mock_driver.index.side_effect = ValueError(f"Bad document parameter: password={secret_password}")

    doc = {"document_id": "r_102"}
    with pytest.raises(ValueError) as exc_info:
        client_with_mock.index_document(doc)

    assert secret_password not in str(exc_info.value)
    assert "Elasticsearch driver value error (ValueError)" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert secret_password not in caplog.text


def test_driver_type_error_in_create_index_sanitized(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    caplog: pytest.LogCaptureFixture,
):
    """Verifies driver TypeError in create_index sanitizes secrets and does not retry with body."""
    caplog.set_level(logging.DEBUG)
    secret_auth = "Basic c2VjcmV0X3Rva2Vu"
    mock_driver.indices.exists.return_value = False
    mock_driver.indices.create.side_effect = TypeError(f"Settings error on Auth: {secret_auth}")

    with pytest.raises(TypeError) as exc_info:
        client_with_mock.create_index("new_idx")

    assert secret_auth not in str(exc_info.value)
    assert "Elasticsearch driver argument error (TypeError)" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert secret_auth not in caplog.text

    # Confirms no legacy retry was attempted
    assert mock_driver.indices.create.call_count == 1


def test_driver_value_error_in_delete_index_sanitized(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    caplog: pytest.LogCaptureFixture,
):
    """Verifies driver ValueError in delete_index sanitizes secrets."""
    caplog.set_level(logging.DEBUG)
    secret_key = "secret_api_key_delete"
    mock_driver.indices.exists.return_value = True
    mock_driver.indices.delete.side_effect = ValueError(f"Delete failed for token={secret_key}")

    with pytest.raises(ValueError) as exc_info:
        client_with_mock.delete_index("delete_idx")

    assert secret_key not in str(exc_info.value)
    assert "Elasticsearch driver value error (ValueError)" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert secret_key not in caplog.text


def test_caller_input_validation_distinct_from_driver_errors(client_with_mock: ElasticsearchClient):
    """Verifies caller input validation errors are distinct from driver errors."""
    with pytest.raises(TypeError, match="Expected document dictionary"):
        client_with_mock.index_document("not_a_dict")  # type: ignore

    with pytest.raises(ValueError, match="missing valid 'document_id'"):
        client_with_mock.index_document({"platform": "reddit"})

    with pytest.raises(ValueError, match="must be lowercase"):
        client_with_mock.create_index("UPPERCASE")


# =====================================================================
# Security & Secret Sanitization Tests
# =====================================================================

def test_no_credential_exposure_in_repr_and_str():
    """Verifies password is never exposed in client representations."""
    secret_pw = "super_classified_password_999"
    client = ElasticsearchClient(
        host="http://localhost:9200",
        username="agent_user",
        password=secret_pw,
    )
    repr_str = repr(client)
    str_str = str(client)

    assert secret_pw not in repr_str
    assert secret_pw not in str_str
    assert "auth=configured" in repr_str
    assert "user='agent_user'" in repr_str


def test_no_credential_exposure_in_logs(client_with_mock: ElasticsearchClient, caplog: pytest.LogCaptureFixture):
    """Verifies logs do not contain plaintext password during client operations."""
    caplog.set_level(logging.DEBUG)
    client_with_mock.ping()
    client_with_mock.health_check()
    client_with_mock.create_index("test_log_index")

    assert "secret_password_123" not in caplog.text


def test_raw_exception_sanitization_in_logs_and_health_check(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    caplog: pytest.LogCaptureFixture,
):
    """Verifies raw exception text with potential secrets is not exposed in logs or health check."""
    caplog.set_level(logging.DEBUG)
    sensitive_token = "bearer_sensitive_token_abc123"
    mock_driver.ping.side_effect = ConnectionError(f"HTTP 401 with Token {sensitive_token}")

    health = client_with_mock.health_check()

    assert sensitive_token not in str(health)
    assert sensitive_token not in caplog.text
    assert health["status"] == "disconnected"


# =====================================================================
# Consistency & Error Handling in Ping, Health Check, Close
# =====================================================================

def test_ping_unexpected_programming_error_raises_client_error(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies unexpected programming errors in ping raise ElasticsearchClientError."""
    mock_driver.ping.side_effect = AttributeError("Driver method corrupted")

    with pytest.raises(ElasticsearchClientError, match="Ping failed"):
        client_with_mock.ping()


def test_ping_connectivity_failure_raises_connection_error(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies network/connectivity failures during ping raise ElasticsearchConnectionError."""
    mock_driver.ping.side_effect = ConnectionError("Cluster connection refused")

    with pytest.raises(ElasticsearchConnectionError, match="Ping connection failed"):
        client_with_mock.ping()


def test_health_check_unexpected_error_raises_client_error(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies unexpected programming errors in health_check raise ElasticsearchClientError."""
    mock_driver.ping.return_value = True
    mock_driver.cluster.health.side_effect = AttributeError("Internal cluster API bug")

    with pytest.raises(ElasticsearchClientError, match="Health check failed"):
        client_with_mock.health_check()


def test_health_check_sanitizes_cluster_health_and_strips_arbitrary_fields(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies that health_check returns only status, number_of_nodes, and active_shards,
    stripping arbitrary fields and secrets from the driver response."""
    secret_cluster_token = "secret_health_bearer_token_xyz999"
    secret_auth_header = "Bearer secret_header_value"
    mock_driver.ping.return_value = True
    mock_driver.cluster.health.return_value = {
        "status": "green",
        "number_of_nodes": 4,
        "active_shards": 12,
        # Arbitrary and secret-bearing fields that must NEVER be returned:
        "cluster_name": "production-internal-cluster",
        "timed_out": False,
        "number_of_data_nodes": 3,
        "active_primary_shards": 6,
        "secret_token": secret_cluster_token,
        "admin_credentials": {"auth": secret_auth_header},
    }

    result = client_with_mock.health_check()

    assert result["status"] == "connected"
    assert result["host"] == "http://localhost:9200"
    assert result["authenticated"] is True

    # cluster_health must contain ONLY explicitly approved fields
    assert "cluster_health" in result
    assert set(result["cluster_health"].keys()) == {"status", "number_of_nodes", "active_shards"}
    assert result["cluster_health"]["status"] == "green"
    assert result["cluster_health"]["number_of_nodes"] == 4
    assert result["cluster_health"]["active_shards"] == 12

    # Secrets and arbitrary fields must NOT appear anywhere in the result
    assert secret_cluster_token not in str(result)
    assert secret_auth_header not in str(result)
    assert "cluster_name" not in result
    assert "cluster_name" not in result["cluster_health"]
    assert "secret_token" not in result
    assert "secret_token" not in result["cluster_health"]
    assert "timed_out" not in result["cluster_health"]
    assert "number_of_data_nodes" not in result["cluster_health"]
    assert "active_primary_shards" not in result["cluster_health"]


@pytest.mark.parametrize(
    "raw_cluster_health, expected_sanitized",
    [
        (
            {"status": "YELLOW", "number_of_nodes": 2, "active_shards": 4},
            {"status": "yellow", "number_of_nodes": 2, "active_shards": 4},
        ),
        (
            {"status": 123, "number_of_nodes": True, "active_shards": "10"},
            {},
        ),
        (
            {"status": "", "number_of_nodes": -1, "active_shards": False},
            {},
        ),
        (
            {"status": "red", "number_of_nodes": "not_an_int", "active_shards": 0},
            {"status": "red", "active_shards": 0},
        ),
        (
            {"status": ["green"], "number_of_nodes": 3.14, "active_shards": -5},
            {},
        ),
    ],
)
def test_health_check_validates_field_types_and_rejects_malformed_values(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    raw_cluster_health: dict,
    expected_sanitized: dict,
):
    """Verifies that non-string status, boolean or non-integer node/shard counts are rejected."""
    mock_driver.ping.return_value = True
    mock_driver.cluster.health.return_value = raw_cluster_health

    result = client_with_mock.health_check()
    assert result["status"] == "connected"
    assert result["cluster_health"] == expected_sanitized


def test_sanitize_cluster_health_handles_non_dict_inputs():
    """Verifies _sanitize_cluster_health safely handles non-dictionary inputs."""
    assert _sanitize_cluster_health(None) == {}
    assert _sanitize_cluster_health("corrupted") == {}
    assert _sanitize_cluster_health(123) == {}
    assert _sanitize_cluster_health([]) == {}


def test_close_best_effort_without_logging_raw_exception(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    caplog: pytest.LogCaptureFixture,
):
    """Verifies close handles driver errors best-effort without logging raw messages."""
    caplog.set_level(logging.DEBUG)
    mock_driver.close.side_effect = Exception("Socket error with token=secret_token")

    client_with_mock.get_client()
    client_with_mock.close()

    assert client_with_mock._client is None
    assert "secret_token" not in caplog.text
    assert "Error closing Elasticsearch client (Exception)" in caplog.text


# =====================================================================
# Factory & Lifecycle Tests
# =====================================================================

def test_client_initialization_with_defaults():
    """Verifies client initializes correctly from settings defaults."""
    client = ElasticsearchClient()
    assert client.host.startswith("http://")
    assert client.request_timeout == DEFAULT_REQUEST_TIMEOUT
    assert client.max_retries == 3
    assert client.client_override is None


def test_create_elasticsearch_client_factory():
    """Verifies factory function correctly instantiates client from AppConfig."""
    mock_app_config = AppConfig(
        app_env="testing",
        log_level="DEBUG",
        raw_data_path=None,  # type: ignore
        processed_data_path=None,  # type: ignore
        models_path=None,  # type: ignore
        reddit_client_id="",
        reddit_client_secret="",
        reddit_user_agent="",
        youtube_api_key="",
        neo4j_uri="",
        neo4j_username="",
        neo4j_password="",
        elasticsearch_host="http://es-cluster:9200",
        elasticsearch_username="cluster_admin",
        elasticsearch_password="cluster_password_xyz",
        kibana_host="http://kibana:5601",
    )
    client = create_elasticsearch_client(config=mock_app_config)
    assert client.host == "http://es-cluster:9200"
    assert client.username == "cluster_admin"
    assert client.is_auth_configured is True


def test_missing_elasticsearch_package_raises_runtime_error():
    """Verifies helpful error when elasticsearch package is missing and no override given."""
    client = ElasticsearchClient(host="http://localhost:9200")
    with patch.dict("sys.modules", {"elasticsearch": None}):
        with pytest.raises(RuntimeError) as exc_info:
            client.get_client()

    assert "The 'elasticsearch' package is required" in str(exc_info.value)
    assert "pip install elasticsearch" in str(exc_info.value)
    assert exc_info.value.__cause__ is None


# =====================================================================
# Index Management & Modern 8.x API Tests
# =====================================================================

def test_index_exists_true(client_with_mock: ElasticsearchClient, mock_driver: MockElasticsearchDriver):
    """Verifies index_exists returns True when index is found."""
    mock_driver.indices.exists.return_value = True
    assert client_with_mock.index_exists("my_index") is True
    mock_driver.indices.exists.assert_called_once_with(index="my_index")


def test_index_exists_false(client_with_mock: ElasticsearchClient, mock_driver: MockElasticsearchDriver):
    """Verifies index_exists returns False when index is not found."""
    mock_driver.indices.exists.return_value = False
    assert client_with_mock.index_exists("missing_index") is False


def test_create_index_uses_modern_api_and_reuses_canonical_mapping(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies create_index uses modern 8.x parameters (settings, mappings) without duplication."""
    mock_driver.indices.exists.return_value = False
    mock_driver.indices.create.return_value = {"acknowledged": True}

    res = client_with_mock.create_index(
        index_name="social_records",
        number_of_shards=1,
        number_of_replicas=0,
    )

    assert res["acknowledged"] is True
    mock_driver.indices.create.assert_called_once()

    call_kwargs = mock_driver.indices.create.call_args.kwargs
    assert call_kwargs["index"] == "social_records"
    assert call_kwargs["settings"] == get_index_settings(number_of_shards=1, number_of_replicas=0)
    assert call_kwargs["mappings"] == get_index_mapping()
    assert "body" not in call_kwargs


def test_create_index_already_exists_skips_creation_and_never_deletes(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies force_recreate=False never deletes an existing index and skips creation."""
    mock_driver.indices.exists.return_value = True

    res = client_with_mock.create_index("existing_index", force_recreate=False)

    assert res["already_exists"] is True
    assert res["created"] is False
    mock_driver.indices.delete.assert_not_called()
    mock_driver.indices.create.assert_not_called()


def test_create_index_force_recreate_deletes_then_creates(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies force_recreate=True deletes then creates index."""
    mock_driver.indices.exists.return_value = True
    mock_driver.indices.delete.return_value = {"acknowledged": True}
    mock_driver.indices.create.return_value = {"acknowledged": True}

    res = client_with_mock.create_index("recreate_index", force_recreate=True)

    assert res["acknowledged"] is True
    mock_driver.indices.delete.assert_called_once_with(index="recreate_index")
    mock_driver.indices.create.assert_called_once()


def test_delete_index_success(client_with_mock: ElasticsearchClient, mock_driver: MockElasticsearchDriver):
    """Verifies delete_index calls driver delete when index exists."""
    mock_driver.indices.exists.return_value = True
    mock_driver.indices.delete.return_value = {"acknowledged": True}

    assert client_with_mock.delete_index("delete_target") is True
    mock_driver.indices.delete.assert_called_once_with(index="delete_target")


def test_delete_index_not_exists_returns_false(client_with_mock: ElasticsearchClient, mock_driver: MockElasticsearchDriver):
    """Verifies delete_index returns False without calling delete when index does not exist."""
    mock_driver.indices.exists.return_value = False

    assert client_with_mock.delete_index("non_existent") is False
    mock_driver.indices.delete.assert_not_called()


# =====================================================================
# Document Indexing Tests (Modern API)
# =====================================================================

def test_index_document_success(client_with_mock: ElasticsearchClient, mock_driver: MockElasticsearchDriver):
    """Verifies index_document uses modern API (id and document) without legacy fallback."""
    mock_driver.index.return_value = {"result": "created", "_id": "reddit_t3_abc123"}
    doc = {
        "document_id": "reddit_t3_abc123",
        "platform": "reddit",
        "native_id": "t3_abc123",
    }

    res = client_with_mock.index_document(doc, index_name="custom_index", refresh="wait_for")

    assert res["result"] == "created"
    mock_driver.index.assert_called_once_with(
        index="custom_index",
        id="reddit_t3_abc123",
        document=doc,
        refresh="wait_for",
    )


# =====================================================================
# Regression Tests for Step 7.5 Final Defensive Hardening
# =====================================================================

@pytest.mark.parametrize(
    "valid_dns_host",
    [
        "http://localhost:9200",
        "http://elasticsearch:9200",
        "https://node-1.example.com:9243",
        "http://a.b.c:9200",
        f"http://{'a' * 63}.example.com:9200",
    ],
)
def test_dns_label_valid_hostnames_accepted(valid_dns_host: str):
    """Verifies RFC 1123 compliant DNS labels (localhost, elasticsearch, node-1, max 63-char) are accepted."""
    assert validate_host_url(valid_dns_host) == valid_dns_host


@pytest.mark.parametrize(
    "invalid_dns_host",
    [
        "http://-node1.example.com:9200",
        "http://node1-.example.com:9200",
        "http://node1..example.com:9200",
        f"http://{'a' * 64}.example.com:9200",
        "http://-localhost:9200",
        "http://localhost-:9200",
    ],
)
def test_dns_label_invalid_syntax_rejected(invalid_dns_host: str):
    """Verifies invalid leading/trailing hyphens, empty labels, and labels > 63 chars are rejected."""
    with pytest.raises(ValueError, match="Invalid hostname syntax"):
        validate_host_url(invalid_dns_host)


@pytest.mark.parametrize(
    "invalid_cred_val",
    [
        False,
        True,
        0,
        1,
        ["admin"],
        {"user": "admin"},
        1.5,
    ],
)
def test_credential_non_string_types_rejected_without_coercion(invalid_cred_val: Any):
    """Verifies boolean, integer, list, and other non-string credentials raise TypeError rather than coercing."""
    with pytest.raises(TypeError, match="Username must be a string"):
        ElasticsearchClient(host="http://localhost:9200", username=invalid_cred_val)

    with pytest.raises(TypeError, match="Password must be a string"):
        ElasticsearchClient(host="http://localhost:9200", password=invalid_cred_val)


@pytest.mark.parametrize(
    "invalid_cred_val",
    [
        False,
        0,
        [],
        {},
    ],
)
def test_config_derived_non_string_credentials_rejected(invalid_cred_val: Any):
    """Verifies configuration-derived non-string credentials raise TypeError independently without silent coercion."""
    # Test invalid username in config
    mock_app_config_user = AppConfig(
        app_env="testing",
        log_level="DEBUG",
        raw_data_path=None,  # type: ignore
        processed_data_path=None,  # type: ignore
        models_path=None,  # type: ignore
        reddit_client_id="",
        reddit_client_secret="",
        reddit_user_agent="",
        youtube_api_key="",
        neo4j_uri="",
        neo4j_username="",
        neo4j_password="",
        elasticsearch_host="http://localhost:9200",
        elasticsearch_username=invalid_cred_val,  # type: ignore
        elasticsearch_password="valid_password",
        kibana_host="http://kibana:5601",
    )
    with pytest.raises(TypeError, match="Username must be a string"):
        create_elasticsearch_client(config=mock_app_config_user)

    # Test invalid password in config
    mock_app_config_pw = AppConfig(
        app_env="testing",
        log_level="DEBUG",
        raw_data_path=None,  # type: ignore
        processed_data_path=None,  # type: ignore
        models_path=None,  # type: ignore
        reddit_client_id="",
        reddit_client_secret="",
        reddit_user_agent="",
        youtube_api_key="",
        neo4j_uri="",
        neo4j_username="",
        neo4j_password="",
        elasticsearch_host="http://localhost:9200",
        elasticsearch_username="valid_user",
        elasticsearch_password=invalid_cred_val,  # type: ignore
        kibana_host="http://kibana:5601",
    )
    with pytest.raises(TypeError, match="Password must be a string"):
        create_elasticsearch_client(config=mock_app_config_pw)

    # Test explicit constructor arguments without host parameter
    with pytest.raises(TypeError, match="Username must be a string"):
        ElasticsearchClient(username=invalid_cred_val)

    with pytest.raises(TypeError, match="Password must be a string"):
        ElasticsearchClient(password=invalid_cred_val)


def test_timeout_validation_non_reflective_and_overflow_safe():
    """Verifies extreme integers cannot cause uncaught OverflowError and values are never reflected."""
    extreme_large_int = 10**1000
    with pytest.raises(ValueError) as exc_info:
        ElasticsearchClient(host="http://localhost:9200", request_timeout=extreme_large_int)

    err_msg = str(exc_info.value)
    assert err_msg == "request_timeout must be a finite number greater than 0."
    assert str(extreme_large_int) not in err_msg


@pytest.mark.parametrize(
    "whitespace_index_name",
    [
        "index\tname",
        "index\nname",
        "index\rname",
        "my index",
        "index\vname",
        "index\fname",
    ],
)
def test_index_name_rejects_internal_whitespace(whitespace_index_name: str):
    """Verifies all internal whitespace characters (tabs, newlines, spaces) are rejected."""
    with pytest.raises(ValueError, match="Index name contains invalid characters"):
        validate_index_name(whitespace_index_name)


@pytest.mark.parametrize(
    "method_name",
    [
        "index_exists",
        "create_index",
        "delete_index",
    ],
)
def test_explicit_empty_index_name_rejected_consistently(
    client_with_mock: ElasticsearchClient,
    method_name: str,
):
    """Verifies explicitly passing empty or blank string is rejected with ValueError rather than defaulting."""
    method = getattr(client_with_mock, method_name)
    with pytest.raises(ValueError, match="Index name cannot be empty or blank"):
        method("")

    with pytest.raises(ValueError, match="Index name cannot be empty or blank"):
        method("   ")


def test_index_document_rejects_explicit_empty_index_name(client_with_mock: ElasticsearchClient):
    """Verifies index_document rejects explicitly passed empty index name."""
    doc = {"document_id": "test_doc_1"}
    with pytest.raises(ValueError, match="Index name cannot be empty or blank"):
        client_with_mock.index_document(doc, index_name="")

    with pytest.raises(ValueError, match="Index name cannot be empty or blank"):
        client_with_mock.index_document(doc, index_name="   ")


@pytest.mark.parametrize("valid_refresh", [True, False, "wait_for", None])
def test_index_document_accepts_valid_refresh_values(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    valid_refresh: Any,
):
    """Verifies index_document accepts exactly True, False, 'wait_for', and None."""
    mock_driver.index.return_value = {"result": "created", "_id": "doc_1"}
    doc = {"document_id": "doc_1"}

    res = client_with_mock.index_document(doc, refresh=valid_refresh)
    assert res["result"] == "created"


@pytest.mark.parametrize(
    "invalid_refresh",
    [
        1,
        0,
        "true",
        "false",
        "now",
        [],
        {},
        1.0,
    ],
)
def test_index_document_rejects_invalid_refresh_before_driver_call(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    invalid_refresh: Any,
):
    """Verifies index_document rejects non-boolean ints and arbitrary strings before driver call."""
    doc = {"document_id": "doc_1"}
    with pytest.raises(ValueError, match="Invalid refresh parameter"):
        client_with_mock.index_document(doc, refresh=invalid_refresh)

    mock_driver.index.assert_not_called()


def test_index_document_rejects_invalid_refresh_before_get_client(monkeypatch: pytest.MonkeyPatch):
    """Verifies refresh validation occurs strictly before get_client() is called and driver is never touched."""
    client = ElasticsearchClient(host="http://localhost:9200")
    get_client_called = False

    def fail_get_client():
        nonlocal get_client_called
        get_client_called = True
        raise AssertionError("get_client() must not be called when refresh is invalid")

    monkeypatch.setattr(client, "get_client", fail_get_client)
    with pytest.raises(ValueError, match="Invalid refresh parameter"):
        client.index_document({"document_id": "doc_1"}, refresh="invalid_refresh")

    assert get_client_called is False


# =====================================================================
# 15. Bounded Bulk Indexing (bulk_index_documents) Tests
# =====================================================================

def test_bulk_index_empty_documents_returns_zero_counts_without_driver_call(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies empty documents list returns zero counts immediately without touching driver."""
    res = client_with_mock.bulk_index_documents([])

    assert res["total"] == 0
    assert res["indexed_count"] == 0
    assert res["failed_count"] == 0
    assert res["successful_ids"] == []
    assert res["failed_items"] == []
    assert res["errors"] is False
    mock_driver.bulk.assert_not_called()


@pytest.mark.parametrize("invalid_docs", [None, True, False, "invalid_str", 123, {"document_id": "doc_1"}])
def test_bulk_index_rejects_non_list_documents(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    invalid_docs: Any,
):
    """Verifies non-list documents argument raises TypeError before driver calls."""
    with pytest.raises(TypeError, match="Expected list of documents"):
        client_with_mock.bulk_index_documents(invalid_docs)

    mock_driver.bulk.assert_not_called()


@pytest.mark.parametrize("invalid_item", ["string_doc", 42, None, True, [1, 2]])
def test_bulk_index_rejects_non_dict_element_in_documents(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    invalid_item: Any,
):
    """Verifies any non-dict item in documents list raises TypeError before driver call."""
    docs = [{"document_id": "doc_valid"}, invalid_item]
    with pytest.raises(TypeError, match="must be a dictionary"):
        client_with_mock.bulk_index_documents(docs)

    mock_driver.bulk.assert_not_called()


@pytest.mark.parametrize("bad_id", [None, "", "   ", 123, False, []])
def test_bulk_index_rejects_missing_or_blank_document_id(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    bad_id: Any,
):
    """Verifies any document lacking a non-empty document_id string raises ValueError."""
    docs = [{"document_id": "doc_1"}, {"document_id": bad_id}]
    with pytest.raises(ValueError, match="missing valid 'document_id'"):
        client_with_mock.bulk_index_documents(docs)

    mock_driver.bulk.assert_not_called()


@pytest.mark.parametrize("valid_refresh", [True, False, "wait_for", None])
def test_bulk_index_accepts_valid_refresh_values(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    valid_refresh: Any,
):
    """Verifies bulk_index accepts valid refresh options and passes them to driver."""
    docs = [{"document_id": "doc_1"}]
    mock_driver.bulk.return_value = {
        "took": 2,
        "errors": False,
        "items": [{"index": {"_id": "doc_1", "status": 201}}],
    }

    res = client_with_mock.bulk_index_documents(docs, refresh=valid_refresh)
    assert res["indexed_count"] == 1
    assert res["failed_count"] == 0

    call_kwargs = mock_driver.bulk.call_args[1]
    if valid_refresh is not None:
        assert call_kwargs.get("refresh") == valid_refresh
    else:
        assert "refresh" not in call_kwargs


@pytest.mark.parametrize("invalid_refresh", [1, 0, "true", "false", "immediate", [], {}])
def test_bulk_index_rejects_invalid_refresh_before_driver_call(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
    invalid_refresh: Any,
):
    """Verifies bulk_index rejects invalid refresh parameters before driver invocation."""
    docs = [{"document_id": "doc_1"}]
    with pytest.raises(ValueError, match="Invalid refresh parameter"):
        client_with_mock.bulk_index_documents(docs, refresh=invalid_refresh)

    mock_driver.bulk.assert_not_called()


def test_bulk_index_success_constructs_valid_8x_operations_payload(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies bulk_index serializes documents into modern Elasticsearch 8.x operations structure."""
    docs = [
        {"document_id": "doc_alpha", "content": "hello world"},
        {"document_id": "doc_beta", "content": "social media"},
    ]
    mock_driver.bulk.return_value = {
        "took": 15,
        "errors": False,
        "items": [
            {"index": {"_id": "doc_alpha", "status": 201}},
            {"index": {"_id": "doc_beta", "status": 201}},
        ],
    }

    res = client_with_mock.bulk_index_documents(docs, index_name="custom_index")

    assert res["total"] == 2
    assert res["indexed_count"] == 2
    assert res["failed_count"] == 0
    assert res["successful_ids"] == ["doc_alpha", "doc_beta"]
    assert res["failed_items"] == []
    assert res["errors"] is False

    mock_driver.bulk.assert_called_once()
    call_kwargs = mock_driver.bulk.call_args[1]
    ops = call_kwargs["operations"]
    assert len(ops) == 4
    assert ops[0] == {"index": {"_index": "custom_index", "_id": "doc_alpha"}}
    assert ops[1] == docs[0]
    assert ops[2] == {"index": {"_index": "custom_index", "_id": "doc_beta"}}
    assert ops[3] == docs[1]


def test_bulk_index_per_document_outcome_isolation_on_partial_failure(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies failure of one document does NOT mark successful documents in same batch as failed."""
    docs = [
        {"document_id": "doc_success_1"},
        {"document_id": "doc_mapper_error_2"},
        {"document_id": "doc_success_3"},
    ]
    # Document 2 fails with 400 permanent mapper error, while Docs 1 and 3 succeed (201)
    mock_driver.bulk.return_value = {
        "took": 10,
        "errors": True,
        "items": [
            {"index": {"_id": "doc_success_1", "status": 201}},
            {
                "index": {
                    "_id": "doc_mapper_error_2",
                    "status": 400,
                    "error": {
                        "type": "mapper_parsing_exception",
                        "reason": "failed to parse field [score] of type [float]",
                    },
                }
            },
            {"index": {"_id": "doc_success_3", "status": 201}},
        ],
    }

    res = client_with_mock.bulk_index_documents(docs)

    assert res["total"] == 3
    assert res["indexed_count"] == 2
    assert res["failed_count"] == 1
    assert "doc_success_1" in res["successful_ids"]
    assert "doc_success_3" in res["successful_ids"]
    assert len(res["failed_items"]) == 1
    assert res["failed_items"][0]["document_id"] == "doc_mapper_error_2"
    assert res["failed_items"][0]["status"] == 400
    assert "mapper_parsing_exception" in res["failed_items"][0]["error"] or "failed to parse field" in res["failed_items"][0]["error"]
    assert res["errors"] is True

    # Invariant assertion
    assert res["total"] == res["indexed_count"] + res["failed_count"]
    # Only 1 attempt made because 400 is permanent
    assert mock_driver.bulk.call_count == 1


def test_bulk_index_transient_retry_success(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies transient 429 item error is retried and only transient items are resent."""
    docs = [
        {"document_id": "doc_fast"},
        {"document_id": "doc_throttled"},
    ]

    attempt_1_resp = {
        "took": 12,
        "errors": True,
        "items": [
            {"index": {"_id": "doc_fast", "status": 201}},
            {"index": {"_id": "doc_throttled", "status": 429, "error": {"reason": "too many requests"}}},
        ],
    }
    attempt_2_resp = {
        "took": 5,
        "errors": False,
        "items": [
            {"index": {"_id": "doc_throttled", "status": 201}},
        ],
    }
    mock_driver.bulk.side_effect = [attempt_1_resp, attempt_2_resp]

    res = client_with_mock.bulk_index_documents(docs)

    assert res["total"] == 2
    assert res["indexed_count"] == 2
    assert res["failed_count"] == 0
    assert "doc_fast" in res["successful_ids"]
    assert "doc_throttled" in res["successful_ids"]
    assert res["errors"] is False
    assert mock_driver.bulk.call_count == 2

    # Verify attempt 2 only sent doc_throttled (doc_fast was already successful)
    attempt_2_call_ops = mock_driver.bulk.call_args_list[1][1]["operations"]
    assert len(attempt_2_call_ops) == 2
    assert attempt_2_call_ops[0]["index"]["_id"] == "doc_throttled"


def test_bulk_index_transient_retry_exhaustion(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies persistent 503 item errors are bounded by max_retries and recorded as failed."""
    docs = [{"document_id": "doc_service_unavailable"}]
    transient_503_resp = {
        "took": 5,
        "errors": True,
        "items": [
            {"index": {"_id": "doc_service_unavailable", "status": 503, "error": {"reason": "cluster overloaded"}}},
        ],
    }
    # Max attempts = max_retries + 1 = 4
    mock_driver.bulk.return_value = transient_503_resp

    res = client_with_mock.bulk_index_documents(docs)

    assert res["total"] == 1
    assert res["indexed_count"] == 0
    assert res["failed_count"] == 1
    assert res["failed_items"][0]["document_id"] == "doc_service_unavailable"
    assert res["failed_items"][0]["status"] == 503
    assert mock_driver.bulk.call_count == client_with_mock.max_retries + 1


def test_bulk_index_transport_connection_error_retry_and_recovery(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies transport-level ConnectionError on client.bulk() is retried and succeeds on recovery."""
    docs = [{"document_id": "doc_recover"}]
    success_resp = {
        "took": 8,
        "errors": False,
        "items": [{"index": {"_id": "doc_recover", "status": 201}}],
    }
    mock_driver.bulk.side_effect = [
        ConnectionError("Socket disconnected"),
        success_resp,
    ]

    res = client_with_mock.bulk_index_documents(docs)

    assert res["total"] == 1
    assert res["indexed_count"] == 1
    assert res["failed_count"] == 0
    assert res["successful_ids"] == ["doc_recover"]
    assert mock_driver.bulk.call_count == 2


def test_bulk_index_transport_connection_error_exhaustion_raise_on_error_false(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies transport error exhaustion marks all documents in batch as failed without raising."""
    docs = [{"document_id": "doc_1"}, {"document_id": "doc_2"}]
    mock_driver.bulk.side_effect = ConnectionError("Permanent network down")

    res = client_with_mock.bulk_index_documents(docs, raise_on_error=False)

    assert res["total"] == 2
    assert res["indexed_count"] == 0
    assert res["failed_count"] == 2
    assert res["failed_items"][0]["status"] == 503
    assert "(ConnectionError)" in res["failed_items"][0]["error"]
    assert mock_driver.bulk.call_count == client_with_mock.max_retries + 1


def test_bulk_index_transport_connection_error_exhaustion_raise_on_error_true(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies transport error exhaustion raises ElasticsearchConnectionError when raise_on_error=True."""
    docs = [{"document_id": "doc_1"}]
    mock_driver.bulk.side_effect = ConnectionError("Permanent network down")

    with pytest.raises(ElasticsearchConnectionError, match="Connection failed during bulk indexing"):
        client_with_mock.bulk_index_documents(docs, raise_on_error=True)


def test_bulk_index_permanent_driver_error_fails_immediately_without_retry(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies driver ValueError fails immediately on attempt 1 without retry."""
    docs = [{"document_id": "doc_1"}]
    mock_driver.bulk.side_effect = ValueError("Invalid serialization")

    with pytest.raises(ValueError, match="Elasticsearch driver value error"):
        client_with_mock.bulk_index_documents(docs)

    assert mock_driver.bulk.call_count == 1


def test_bulk_index_raise_on_error_true_raises_index_error_on_partial_failure(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies raise_on_error=True raises ElasticsearchIndexError when any item in batch fails."""
    docs = [{"document_id": "doc_1"}]
    mock_driver.bulk.return_value = {
        "errors": True,
        "items": [
            {"index": {"_id": "doc_1", "status": 400, "error": {"reason": "mapping error"}}},
        ],
    }

    with pytest.raises(ElasticsearchIndexError, match="Bulk indexing failed for 1 document"):
        client_with_mock.bulk_index_documents(docs, raise_on_error=True)


def test_bulk_index_sequential_fallback_when_driver_lacks_bulk(
    client_with_mock: ElasticsearchClient,
):
    """Verifies sequential fallback via index_document if underlying driver lacks bulk method."""
    mock_bare_driver = MagicMock(spec=["ping", "index"])
    mock_bare_driver.index.return_value = {"result": "created", "_id": "doc_fallback"}
    client_with_mock._client = mock_bare_driver

    docs = [{"document_id": "doc_fallback"}]
    res = client_with_mock.bulk_index_documents(docs)

    assert res["total"] == 1
    assert res["indexed_count"] == 1
    assert res["failed_count"] == 0
    assert res["successful_ids"] == ["doc_fallback"]
    mock_bare_driver.index.assert_called_once()


def test_bulk_index_legacy_body_fallback_when_operations_rejected(
    client_with_mock: ElasticsearchClient,
    mock_driver: MockElasticsearchDriver,
):
    """Verifies compatibility fallback to 'body' keyword if driver raises TypeError on 'operations'."""
    docs = [{"document_id": "doc_legacy"}]

    def mock_bulk_body_only(**kwargs):
        if "operations" in kwargs:
            raise TypeError("bulk() got an unexpected keyword argument 'operations'")
        if "body" in kwargs:
            return {"took": 3, "errors": False, "items": [{"index": {"_id": "doc_legacy", "status": 201}}]}
        raise ValueError("Missing payload")

    mock_driver.bulk.side_effect = mock_bulk_body_only

    res = client_with_mock.bulk_index_documents(docs)
    assert res["total"] == 1
    assert res["indexed_count"] == 1
    assert res["failed_count"] == 0
    assert res["successful_ids"] == ["doc_legacy"]



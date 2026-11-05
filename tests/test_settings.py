"""Unit tests for configuration and security settings management."""

from pathlib import Path
from config.settings import load_config, AppConfig


def test_default_config_loading():
    """Verify default configuration loads with expected fallback values."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.app_env in {"development", "testing", "production"}
    assert config.log_level in {"DEBUG", "INFO", "WARNING", "ERROR"}
    assert isinstance(config.raw_data_path, Path)
    assert isinstance(config.processed_data_path, Path)
    assert isinstance(config.models_path, Path)


def test_directory_creation():
    """Verify configured directories exist on disk."""
    config = load_config()
    assert config.raw_data_path.exists()
    assert config.processed_data_path.exists()
    assert config.models_path.exists()


def test_safe_summary_does_not_expose_secrets(monkeypatch):
    """Verify all 4 secrets are never exposed in get_safe_summary()."""
    dummy_reddit_secret = "dummy_reddit_secret_9988"
    dummy_youtube_key = "dummy_youtube_api_key_12345"
    dummy_neo4j_pass = "dummy_neo4j_password_abcde"
    dummy_elastic_pass = "dummy_elastic_password_7788"

    monkeypatch.setenv("REDDIT_CLIENT_ID", "dummy_client_id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", dummy_reddit_secret)
    monkeypatch.setenv("YOUTUBE_API_KEY", dummy_youtube_key)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_PASSWORD", dummy_neo4j_pass)
    monkeypatch.setenv("ELASTICSEARCH_HOST", "http://localhost:9200")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", dummy_elastic_pass)

    config = load_config()
    summary = config.get_safe_summary()

    secrets = [dummy_reddit_secret, dummy_youtube_key, dummy_neo4j_pass, dummy_elastic_pass]
    for key, value in summary.items():
        for secret in secrets:
            assert secret not in str(value), f"Secret exposed in summary field '{key}'"

    # With valid dummy credentials, all flags should evaluate to True
    assert summary["reddit_configured"] is True
    assert summary["youtube_configured"] is True
    assert summary["neo4j_configured"] is True
    assert summary["elasticsearch_configured"] is True


def test_repr_and_str_do_not_expose_secrets(monkeypatch):
    """Verify repr(config) and str(config) do not leak credentials."""
    dummy_reddit_secret = "dummy_reddit_secret_9988"
    dummy_youtube_key = "dummy_youtube_api_key_12345"
    dummy_neo4j_pass = "dummy_neo4j_password_abcde"
    dummy_elastic_pass = "dummy_elastic_password_7788"

    monkeypatch.setenv("REDDIT_CLIENT_SECRET", dummy_reddit_secret)
    monkeypatch.setenv("YOUTUBE_API_KEY", dummy_youtube_key)
    monkeypatch.setenv("NEO4J_PASSWORD", dummy_neo4j_pass)
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", dummy_elastic_pass)

    config = load_config()
    repr_output = repr(config)
    str_output = str(config)

    secrets = [dummy_reddit_secret, dummy_youtube_key, dummy_neo4j_pass, dummy_elastic_pass]
    for secret in secrets:
        assert secret not in repr_output, f"Secret leaked in repr(config): {secret}"
        assert secret not in str_output, f"Secret leaked in str(config): {secret}"


def test_placeholder_credentials_reported_as_unconfigured(monkeypatch):
    """Verify template placeholder values are not treated as configured."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "your_reddit_client_id_here")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "your_reddit_client_secret_here")
    monkeypatch.setenv("YOUTUBE_API_KEY", "your_youtube_api_key_here")
    monkeypatch.setenv("NEO4J_PASSWORD", "your_neo4j_password_here")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "your_elastic_password_here")

    config = load_config()
    summary = config.get_safe_summary()

    assert summary["reddit_configured"] is False
    assert summary["youtube_configured"] is False
    assert summary["neo4j_configured"] is False
    assert summary["elasticsearch_configured"] is False


def test_empty_and_whitespace_values_handled_safely(monkeypatch):
    """Verify empty and whitespace-only credentials evaluate to unconfigured."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "   ")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "\t\n ")
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    monkeypatch.setenv("NEO4J_PASSWORD", "  ")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "")

    config = load_config()
    summary = config.get_safe_summary()

    assert summary["reddit_configured"] is False
    assert summary["youtube_configured"] is False
    assert summary["neo4j_configured"] is False
    assert summary["elasticsearch_configured"] is False


def test_elasticsearch_not_falsely_operational_by_default(monkeypatch):
    """Verify Elasticsearch is not reported as configured merely from default host."""
    monkeypatch.delenv("ELASTICSEARCH_PASSWORD", raising=False)
    monkeypatch.setenv("ELASTICSEARCH_HOST", "http://localhost:9200")

    config = load_config()
    summary = config.get_safe_summary()

    assert summary["elasticsearch_configured"] is False

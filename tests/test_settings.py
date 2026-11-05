"""Unit tests for configuration and settings management."""

import os
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
    """Verify secrets are never exposed in get_safe_summary."""
    secret_key = "super_secret_youtube_api_key_12345"
    secret_pass = "top_secret_neo4j_password_abcde"
    monkeypatch.setenv("YOUTUBE_API_KEY", secret_key)
    monkeypatch.setenv("NEO4J_PASSWORD", secret_pass)

    config = load_config()
    summary = config.get_safe_summary()

    # Secret strings must not be present in the summary values
    for key, value in summary.items():
        assert secret_key not in str(value), f"Secret exposed in {key}"
        assert secret_pass not in str(value), f"Password exposed in {key}"

    # Status flags should indicate configured
    assert summary["youtube_configured"] is True
    assert summary["neo4j_configured"] is True

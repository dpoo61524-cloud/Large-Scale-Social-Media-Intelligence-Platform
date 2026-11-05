"""Configuration settings module.

Loads environment variables securely and provides application configuration
without exposing sensitive secrets in logs or application output.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    """Application configuration container."""

    # Environment & Logging
    app_env: str
    log_level: str

    # Paths
    raw_data_path: Path
    processed_data_path: Path
    models_path: Path

    # API Keys & Credentials (kept private, never exposed)
    reddit_client_id: str
    reddit_client_secret: str = field(repr=False)
    reddit_user_agent: str
    youtube_api_key: str = field(repr=False)

    # Database & Search engine connections
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str = field(repr=False)
    elasticsearch_host: str
    elasticsearch_username: str
    elasticsearch_password: str = field(repr=False)
    kibana_host: str

    def get_safe_summary(self) -> dict:
        """Returns non-sensitive status of configurations.

        Never exposes actual passwords, tokens, or API keys.
        """
        return {
            "app_env": self.app_env,
            "log_level": self.log_level,
            "raw_data_path": str(self.raw_data_path),
            "processed_data_path": str(self.processed_data_path),
            "models_path": str(self.models_path),
            "reddit_configured": _is_configured(self.reddit_client_id) and _is_configured(self.reddit_client_secret),
            "youtube_configured": _is_configured(self.youtube_api_key),
            "neo4j_configured": _is_configured(self.neo4j_uri) and _is_configured(self.neo4j_password),
            "elasticsearch_configured": _is_configured(self.elasticsearch_password) and _is_configured(self.elasticsearch_host),
        }


def _is_configured(value: str | None) -> bool:
    """Checks whether a setting contains a real, non-placeholder value."""
    if not value:
        return False
    stripped = value.strip()
    if not stripped or stripped.lower().startswith("your_"):
        return False
    return True


def _mask_presence(value: str | None) -> str:
    """Helper to check if a secret is provided without revealing it."""
    return "Set" if _is_configured(value) else "Not Set"


def load_config(env_path: Path | None = None) -> AppConfig:
    """Loads and validates configuration from environment variables."""
    # Load .env file if available
    target_env = env_path if env_path else BASE_DIR / ".env"
    if target_env.exists():
        load_dotenv(dotenv_path=target_env)
    else:
        # Load any existing environment variables without failing
        load_dotenv()

    # Read environment settings with defaults
    app_env = os.getenv("APP_ENV", "development").lower()
    valid_environments = {"development", "testing", "production"}
    if app_env not in valid_environments:
        app_env = "development"

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Paths
    raw_data_str = os.getenv("RAW_DATA_PATH", "data/raw")
    processed_data_str = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    models_str = os.getenv("MODELS_PATH", "models")

    raw_path = BASE_DIR / raw_data_str
    processed_path = BASE_DIR / processed_data_str
    models_path = BASE_DIR / models_str

    # Ensure local storage directories exist safely
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        app_env=app_env,
        log_level=log_level,
        raw_data_path=raw_path,
        processed_data_path=processed_path,
        models_path=models_path,
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", ""),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        elasticsearch_host=os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200"),
        elasticsearch_username=os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
        elasticsearch_password=os.getenv("ELASTICSEARCH_PASSWORD", ""),
        kibana_host=os.getenv("KIBANA_HOST", "http://localhost:5601"),
    )

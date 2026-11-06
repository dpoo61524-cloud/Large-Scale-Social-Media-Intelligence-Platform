"""Large Scale Social Media Intelligence Platform.

Main entry point for the intelligence platform.
"""

import sys
from config.settings import load_config
from src.ingestion.reddit_ingestion import ingest_reddit_posts, ingest_reddit_live
from src.ingestion.youtube_ingestion import ingest_youtube_live
from src.preprocessing.preprocessing_pipeline import (
    preprocess_reddit_dataset,
    preprocess_youtube_dataset,
)


def print_header() -> None:
    """Displays project title and separator."""
    print("=" * 68)
    print("      LARGE SCALE SOCIAL MEDIA INTELLIGENCE PLATFORM")
    print("=" * 68)
    print("Status: Block 1 - Project Foundation Initialized | Step 1 - Ingestion | Step 2 - Preprocessing (Spark NLP)")
    print("=" * 68)


def print_project_overview() -> None:
    """Displays high-level platform objectives and pipeline stages."""
    print("\n[PROJECT OBJECTIVES]")
    print("1. Data Ingestion: Collect public posts & comments from Reddit & YouTube.")
    print("2. Text Processing: Clean noise, handle emojis, detect languages.")
    print("3. Sentiment Analysis: Multilingual sentiment tracking over time.")
    print("4. Topic Modeling: Dynamic topic trends using BERTopic.")
    print("5. Network Analysis: User-interaction graphs with Neo4j & NetworkX.")
    print("6. Misinformation Detection: Fake news classification using LIAR benchmark.")
    print("7. Search & Dashboards: Elasticsearch indexing with Kibana dashboards.")


def print_system_status() -> None:
    """Displays non-sensitive system status and folder readiness."""
    config = load_config()
    summary = config.get_safe_summary()

    print("\n[ENVIRONMENT & STORAGE STATUS]")
    print(f"Environment Mode : {summary['app_env']}")
    print(f"Log Level        : {summary['log_level']}")
    print(f"Raw Data Storage : {summary['raw_data_path']} (Ready)")
    print(f"Processed Storage: {summary['processed_data_path']} (Ready)")
    print(f"Models Storage   : {summary['models_path']} (Ready)")

    print("\n[DATA INGESTION STATUS]")
    print("Reddit Mock Data : Ready (Offline synthetic fixture)")
    reddit_live_status = "Configured (Ready for live fetch)" if summary["reddit_configured"] else "Disabled (Credentials required in .env)"
    youtube_live_status = "Configured (Ready for live fetch)" if summary["youtube_configured"] else "Disabled (YOUTUBE_API_KEY required in .env)"
    print(f"Reddit Live API  : {reddit_live_status}")
    print(f"YouTube Live API : {youtube_live_status}")
    print(f"Neo4j Database   : {'Ready' if summary['neo4j_configured'] else 'Pending setup (Block 5)'}")
    print(f"Elasticsearch    : Pending setup (Block 7)")

    print("\n[AVAILABLE CLI ACTIONS]")
    print("  python app.py                     : Show platform overview and status")
    print("  python app.py --ingest-reddit     : Ingest mock Reddit dataset into data/raw/")
    print("  python app.py --ingest-reddit-live: Fetch live Reddit posts via OAuth2 API")
    print("  python app.py --ingest-youtube    : Fetch live YouTube videos via Data API v3")
    print("  python app.py --preprocess-reddit : Clean, handle emojis & detect language for Reddit")
    print("  python app.py --preprocess-youtube: Clean, handle emojis & detect language for YouTube")

    print("\n" + "-" * 68)
    print("Project foundation & preprocessing ready for incremental development.")
    print("-" * 68)


def run_mock_reddit_cli() -> None:
    """Executes the Reddit mock ingestion pipeline."""
    print("\n[STARTING REDDIT MOCK INGESTION]")
    summary = ingest_reddit_posts()
    _print_ingestion_summary("Reddit Mock", summary)


def run_live_reddit_cli() -> None:
    """Executes the live Reddit API ingestion pipeline."""
    print("\n[STARTING REDDIT LIVE API INGESTION]")
    summary = ingest_reddit_live()
    _print_ingestion_summary("Reddit Live API", summary)


def run_youtube_cli() -> None:
    """Executes the YouTube Data API v3 ingestion pipeline."""
    print("\n[STARTING YOUTUBE DATA API INGESTION]")
    summary = ingest_youtube_live()
    _print_ingestion_summary("YouTube Data API v3", summary)


def run_preprocess_reddit_cli() -> None:
    """Executes the batched Reddit preprocessing pipeline."""
    print("\n[STARTING REDDIT TEXT PREPROCESSING (STEP 2)]")
    summary = preprocess_reddit_dataset()
    _print_preprocessing_summary("Reddit Posts", summary)


def run_preprocess_youtube_cli() -> None:
    """Executes the batched YouTube preprocessing pipeline."""
    print("\n[STARTING YOUTUBE TEXT PREPROCESSING (STEP 2)]")
    summary = preprocess_youtube_dataset()
    _print_preprocessing_summary("YouTube Videos", summary)


def _print_preprocessing_summary(name: str, summary: dict) -> None:
    """Formats and prints summary metrics for a preprocessing run."""
    print(f"Source               : {name}")
    print(f"Status               : {summary.get('status', 'unknown').upper()}")
    if "message" in summary:
        print(f"Message              : {summary['message']}")
    print(f"Total Records        : {summary.get('total_records', 0)}")
    print(f"Successfully Cleaned : {summary.get('processed_records', 0)}")
    print(f"Failed Records       : {summary.get('failed_records', 0)}")
    if summary.get("language_counts"):
        print("Detected Languages   :")
        for lang, count in sorted(summary["language_counts"].items(), key=lambda x: -x[1]):
            print(f"  - {lang}: {count}")
    print(f"Processed Output File: {summary.get('output_file', 'None')}")
    if summary.get("errors"):
        print("\nWarnings / Errors:")
        for err in summary["errors"]:
            print(f"  - {err}")
    print("-" * 68)


def _print_ingestion_summary(name: str, summary: dict) -> None:
    """Formats and prints summary metrics for an ingestion run."""
    print(f"Source           : {name}")
    print(f"Status           : {summary.get('status', 'unknown').upper()}")
    if "message" in summary:
        print(f"Message          : {summary['message']}")
    print(f"Total Records    : {summary.get('total_records', 0)}")
    print(f"Valid Records    : {summary.get('valid_records', 0)}")
    print(f"Rejected Records : {summary.get('rejected_records', 0)}")
    print(f"Raw Destination  : {summary.get('output_file', 'None')}")
    if summary.get("rate_limits"):
        limits = summary["rate_limits"]
        print(f"Rate Limits      : Remaining={limits.get('remaining')}, Reset={limits.get('reset_seconds')}s")
    if summary.get("errors"):
        print("\nWarnings / Errors:")
        for err in summary["errors"]:
            print(f"  - {err}")
    print("-" * 68)


def main() -> None:
    """Main execution entry point."""
    print_header()

    if "--preprocess-reddit" in sys.argv:
        run_preprocess_reddit_cli()
        return

    if "--preprocess-youtube" in sys.argv:
        run_preprocess_youtube_cli()
        return

    if "--ingest-reddit-live" in sys.argv:
        run_live_reddit_cli()
        return

    if "--ingest-youtube" in sys.argv:
        run_youtube_cli()
        return

    if "--ingest-reddit" in sys.argv or "--ingest" in sys.argv:
        run_mock_reddit_cli()
        return

    print_project_overview()
    print_system_status()


if __name__ == "__main__":
    main()

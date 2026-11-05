"""Large Scale Social Media Intelligence Platform.

Main entry point for the platform foundation (Block 1).
"""

from config.settings import load_config


def print_header() -> None:
    """Displays project title and separator."""
    print("=" * 68)
    print("      LARGE SCALE SOCIAL MEDIA INTELLIGENCE PLATFORM")
    print("=" * 68)
    print("Status: Block 1 - Project Foundation Initialized")
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

    print("\n[INTEGRATION STATUS (Preview)]")
    print(f"Reddit API       : {'Ready' if summary['reddit_configured'] else 'Pending credentials in .env'}")
    print(f"YouTube API      : {'Ready' if summary['youtube_configured'] else 'Pending credentials in .env'}")
    print(f"Neo4j Database   : {'Ready' if summary['neo4j_configured'] else 'Pending setup (Block 5)'}")
    print(f"Elasticsearch    : {'Ready' if summary['elasticsearch_configured'] else 'Pending setup (Block 7)'}")

    print("\n" + "-" * 68)
    print("Project foundation verified. Ready for incremental development.")
    print("-" * 68)


def main() -> None:
    """Main execution entry point."""
    print_header()
    print_project_overview()
    print_system_status()


if __name__ == "__main__":
    main()

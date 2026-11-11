"""Search package for Elasticsearch indexing, mapping, and querying."""

from src.search.elasticsearch_client import (
    ElasticsearchClient,
    ElasticsearchClientError,
    ElasticsearchConnectionError,
    ElasticsearchIndexError,
    create_elasticsearch_client,
)
from src.search.index_mapping import (
    DEFAULT_INDEX_NAME,
    INDEX_CONFIGURATION,
    INDEX_MAPPINGS,
    INDEX_SETTINGS,
    get_index_configuration,
    get_index_mapping,
    get_index_settings,
)
from src.search.indexing_adapter import index_consolidated_dataset
from src.search.indexing_pipeline import (
    DEFAULT_DATASET_PATH,
    DEFAULT_INDEXING_MANIFEST_PATH,
    run_indexing_pipeline,
    save_manifest_atomically,
)
from src.search.kibana_dashboard_validator import (
    DEFAULT_BUNDLE_PATH,
    EXPECTED_TOTAL_OBJECTS,
    validate_kibana_dashboard_bundle,
)
from src.search.kibana_dashboard_importer import (
    import_kibana_dashboard_bundle,
)

__all__ = [
    "DEFAULT_INDEX_NAME",
    "INDEX_CONFIGURATION",
    "INDEX_MAPPINGS",
    "INDEX_SETTINGS",
    "get_index_configuration",
    "get_index_mapping",
    "get_index_settings",
    "ElasticsearchClient",
    "ElasticsearchClientError",
    "ElasticsearchConnectionError",
    "ElasticsearchIndexError",
    "create_elasticsearch_client",
    "index_consolidated_dataset",
    "run_indexing_pipeline",
    "save_manifest_atomically",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_INDEXING_MANIFEST_PATH",
    "DEFAULT_BUNDLE_PATH",
    "EXPECTED_TOTAL_OBJECTS",
    "validate_kibana_dashboard_bundle",
    "import_kibana_dashboard_bundle",
]

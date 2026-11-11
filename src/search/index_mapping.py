"""Elasticsearch Index Mapping and Settings for Consolidated Social Intelligence.

Defines the canonical Elasticsearch index settings and strict dynamic schema mapping
for social intelligence records produced by src.pipeline.consolidator.

Key design invariants:
- Single-node local development configuration: 1 primary shard, 0 replicas.
- Strict dynamic mapping ("dynamic": "strict") to prevent indexing unapproved fields.
- Unicode Standard Annex #29 multilingual text analyzer without English-only stopword drops.
- Non-indexed raw content preservation ("index": False).
- Multilingual full-text search over cleaned title and body.
- Nested object mapping for topic keywords with nullable weight.
- Strict segregation of envelope-level metadata from document-level index properties.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

# Default index name for social intelligence records
DEFAULT_INDEX_NAME = "consolidated_intelligence"

# Index settings optimized for local single-node Elasticsearch and multilingual search
INDEX_SETTINGS: Dict[str, Any] = {
    "index": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "mapping": {
            "total_fields": {
                "limit": 1000,
            },
        },
    },
    "analysis": {
        "analyzer": {
            "multilingual_text_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [
                    "lowercase",
                ],
            },
        },
    },
}

# Strict dynamic mapping matching exact consolidator record schema
INDEX_MAPPINGS: Dict[str, Any] = {
    "dynamic": "strict",
    "properties": {
        # Primary identifiers
        "document_id": {
            "type": "keyword",
        },
        "platform": {
            "type": "keyword",
        },
        "native_id": {
            "type": "keyword",
        },
        "url": {
            "type": "keyword",
            "ignore_above": 2048,
        },

        # Temporal timestamps
        "timestamps": {
            "properties": {
                "published_at": {
                    "type": "date",
                },
                "collected_at": {
                    "type": "date",
                },
                "processed_at": {
                    "type": "date",
                },
            },
        },

        # Author identity & graph analytics
        "author": {
            "properties": {
                "username": {
                    "type": "keyword",
                },
                "channel_id": {
                    "type": "keyword",
                },
                "channel_title": {
                    "type": "keyword",
                },
                "graph_status": {
                    "type": "keyword",
                },
                "influencer_score": {
                    "type": "float",
                },
                "pagerank": {
                    "type": "float",
                },
                "community_id": {
                    "type": "integer",
                },
                "in_degree": {
                    "type": "integer",
                },
                "out_degree": {
                    "type": "integer",
                },
            },
        },

        # Content: raw (stored, unindexed) and cleaned (analyzed for multilingual search)
        "content": {
            "properties": {
                "title_raw": {
                    "type": "text",
                    "index": False,
                },
                "body_raw": {
                    "type": "text",
                    "index": False,
                },
                "title_cleaned": {
                    "type": "text",
                    "analyzer": "multilingual_text_analyzer",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 512,
                        },
                    },
                },
                "body_cleaned": {
                    "type": "text",
                    "analyzer": "multilingual_text_analyzer",
                },
                "detected_language": {
                    "type": "keyword",
                },
                "language_code": {
                    "type": "keyword",
                },
                "preprocessing_status": {
                    "type": "keyword",
                },
            },
        },

        # Numeric engagement metrics across Reddit and YouTube
        "engagement": {
            "properties": {
                "score": {
                    "type": "integer",
                },
                "views_count": {
                    "type": "long",
                },
                "likes_count": {
                    "type": "integer",
                },
                "comments_count": {
                    "type": "integer",
                },
            },
        },

        # Platform-specific metadata (nullable objects)
        "metadata": {
            "properties": {
                "reddit": {
                    "properties": {
                        "subreddit": {
                            "type": "keyword",
                        },
                        "score": {
                            "type": "integer",
                        },
                    },
                },
                "youtube": {
                    "properties": {
                        "channel_id": {
                            "type": "keyword",
                        },
                        "channel_title": {
                            "type": "keyword",
                        },
                    },
                },
            },
        },

        # Sentiment enrichment
        "sentiment": {
            "properties": {
                "label": {
                    "type": "keyword",
                },
                "score": {
                    "type": "float",
                },
                "status": {
                    "type": "keyword",
                },
            },
        },

        # Topic modeling enrichment with nested keywords
        "topics": {
            "properties": {
                "topic_id": {
                    "type": "integer",
                },
                "probability": {
                    "type": "float",
                },
                "keywords": {
                    "type": "nested",
                    "properties": {
                        "word": {
                            "type": "keyword",
                            "fields": {
                                "text": {
                                    "type": "text",
                                    "analyzer": "multilingual_text_analyzer",
                                },
                            },
                        },
                        "weight": {
                            "type": "float",
                        },
                    },
                },
                "status": {
                    "type": "keyword",
                },
            },
        },

        # Misinformation & stylometrics enrichment
        "misinformation": {
            "properties": {
                "veracity_label": {
                    "type": "keyword",
                },
                "credibility_score": {
                    "type": "float",
                },
                "risk_tier": {
                    "type": "keyword",
                },
                "status": {
                    "type": "keyword",
                },
                "stylometric_signals": {
                    "properties": {
                        "uppercase_char_ratio": {"type": "float"},
                        "exclamation_intensity": {"type": "float"},
                        "question_intensity": {"type": "float"},
                        "word_count": {"type": "float"},
                        "avg_word_length": {"type": "float"},
                        "lexical_subjectivity_ratio": {"type": "float"},
                        "digits_ratio": {"type": "float"},
                        "all_caps_word_ratio": {"type": "float"},
                    },
                },
            },
        },

        # Consolidation join status & audit flags
        "consolidation": {
            "properties": {
                "status": {
                    "type": "keyword",
                },
                "is_complete": {
                    "type": "boolean",
                },
                "sources_joined": {
                    "type": "keyword",
                },
            },
        },
    },
}

# Complete index configuration package
INDEX_CONFIGURATION: Dict[str, Any] = {
    "settings": INDEX_SETTINGS,
    "mappings": INDEX_MAPPINGS,
}


def get_index_settings(
    number_of_shards: int = 1,
    number_of_replicas: int = 0,
) -> Dict[str, Any]:
    """Returns a deep copy of index settings with configurable shard/replica counts.

    Args:
        number_of_shards: Primary shard count (default: 1).
        number_of_replicas: Replica shard count (default: 0 for single-node local dev).

    Returns:
        Configured Elasticsearch settings dictionary.
    """
    settings = copy.deepcopy(INDEX_SETTINGS)
    settings["index"]["number_of_shards"] = number_of_shards
    settings["index"]["number_of_replicas"] = number_of_replicas
    return settings


def get_index_mapping() -> Dict[str, Any]:
    """Returns a deep copy of the canonical strict dynamic mapping.

    Returns:
        Elasticsearch mapping dictionary with dynamic: strict.
    """
    return copy.deepcopy(INDEX_MAPPINGS)


def get_index_configuration(
    number_of_shards: int = 1,
    number_of_replicas: int = 0,
) -> Dict[str, Any]:
    """Returns complete index creation payload containing settings and mappings.

    Args:
        number_of_shards: Primary shard count (default: 1).
        number_of_replicas: Replica shard count (default: 0).

    Returns:
        Full Elasticsearch index definition dictionary ready for client.indices.create().
    """
    return {
        "settings": get_index_settings(
            number_of_shards=number_of_shards,
            number_of_replicas=number_of_replicas,
        ),
        "mappings": get_index_mapping(),
    }


"""Text preprocessing package using Spark NLP for Social Media Intelligence Platform."""

from src.preprocessing.spark_session import (
    get_spark_session,
    stop_spark_session,
    is_spark_nlp_available,
    check_spark_nlp_prerequisites,
)
from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.emoji_handler import handle_emojis
from src.preprocessing.language_detector import detect_language
from src.preprocessing.preprocessing_pipeline import (
    preprocess_reddit_record,
    preprocess_youtube_record,
    preprocess_reddit_dataset,
    preprocess_youtube_dataset,
)

__all__ = [
    "get_spark_session",
    "stop_spark_session",
    "is_spark_nlp_available",
    "check_spark_nlp_prerequisites",
    "clean_text",
    "handle_emojis",
    "detect_language",
    "preprocess_reddit_record",
    "preprocess_youtube_record",
    "preprocess_reddit_dataset",
    "preprocess_youtube_dataset",
]

"""Integration tests for real Spark NLP runtime and LanguageDetectorDL inference.

These tests execute against the actual Apache Spark JVM runtime and pretrained
Spark NLP LanguageDetectorDL model ('ld_wiki_tatoeba_cnn_21', 'xx').
They are skipped automatically if the Python version or Java environment
is not compatible.
"""

import pytest

from src.preprocessing.language_detector import detect_language
from src.preprocessing.spark_session import (
    get_spark_session,
    is_spark_nlp_available,
    stop_spark_session,
)


@pytest.fixture(scope="module", autouse=True)
def manage_spark_lifecycle():
    """Ensure clean startup and shutdown of the real Spark session across tests."""
    yield
    stop_spark_session()


@pytest.mark.skipif(
    not is_spark_nlp_available(),
    reason="Spark NLP runtime prerequisites not satisfied (requires Python <= 3.12 and Java 17)",
)
class TestSparkNLPIntegration:
    """Integration test suite executing real Spark NLP models."""

    def test_spark_session_initialization_and_version(self):
        """Verify real SparkSession starts with conservative memory settings."""
        spark = get_spark_session(memory="1024M")
        assert spark is not None
        assert spark.version.startswith("3.")
        assert not spark._jsc.sc().isStopped()

    def test_real_language_detection_english(self):
        """Verify real LanguageDetectorDL model predicts English correctly."""
        text = "Machine learning helps organizations understand large amounts of data."
        res = detect_language(text)

        assert res["status"] == "detected"
        assert res["language_code"] == "en"
        assert res["language_name"] == "English"
        assert "confidence" in res
        assert res["confidence"] > 0.90

    def test_real_language_detection_spanish(self):
        """Verify real LanguageDetectorDL model predicts Spanish correctly."""
        text = "El aprendizaje automático ayuda a analizar grandes cantidades de datos."
        res = detect_language(text)

        assert res["status"] == "detected"
        assert res["language_code"] == "es"
        assert res["language_name"] == "Spanish"
        assert "confidence" in res
        assert res["confidence"] > 0.90

    def test_real_language_detection_french(self):
        """Verify real LanguageDetectorDL model predicts French correctly."""
        text = "L'apprentissage automatique permet d'analyser de grandes quantités de données."
        res = detect_language(text)

        assert res["status"] == "detected"
        assert res["language_code"] == "fr"
        assert res["language_name"] == "French"
        assert "confidence" in res
        assert res["confidence"] > 0.90

    def test_real_language_detection_short_and_empty_edge_cases(self):
        """Verify edge cases remain handled without invoking unnecessary model inference."""
        res_empty = detect_language("")
        assert res_empty["status"] == "empty_text"
        assert res_empty["language_code"] == "und"

        res_short = detect_language("Hello")
        assert res_short["status"] == "uncertain_short_text"
        assert res_short["language_code"] == "und"

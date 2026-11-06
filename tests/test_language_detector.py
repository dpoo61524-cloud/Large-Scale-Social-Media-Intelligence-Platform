"""Unit tests for Spark NLP Language Detector Module (Company Step 2)."""

from unittest.mock import MagicMock, patch
import pytest

from src.preprocessing.language_detector import (
    detect_language,
    get_language_detector_pipeline,
    LANGUAGE_NAMES,
)


def test_detect_language_empty_and_null():
    """Verify safe handling of null or empty text inputs."""
    res_none = detect_language(None)
    assert res_none["language_code"] == "und"
    assert res_none["language_name"] == "Undetermined"
    assert res_none["status"] == "empty_text"

    res_empty = detect_language("    \n\t")
    assert res_empty["language_code"] == "und"
    assert res_empty["language_name"] == "Undetermined"
    assert res_empty["status"] == "empty_text"


def test_detect_language_uncertain_short_text():
    """Verify short text (<15 chars) is marked uncertain without false claims."""
    res = detect_language("hello world")
    assert res["language_code"] == "und"
    assert res["language_name"] == "Undetermined"
    assert res["status"] == "uncertain_short_text"


def test_detect_language_reports_missing_prerequisites_honestly():
    """Verify that when Java or compatible Spark runtime is missing, it reports prerequisites honestly."""
    with patch("src.preprocessing.language_detector.is_spark_nlp_available", return_value=False):
        res = detect_language("This is a sufficiently long post discussing social media intelligence pipelines.")
        assert res["language_code"] == "und"
        assert res["language_name"] == "Undetermined"
        assert res["status"] == "prerequisite_missing"
        assert "message" in res


def test_detect_language_english_with_mocked_spark_nlp():
    """Verify English text detection when Spark NLP model is active."""
    mock_pipeline = MagicMock()
    mock_pipeline.annotate.return_value = {"language": ["en"]}

    text = "This is a detailed analysis of large scale data engineering workflows and transformer models."
    res = detect_language(text, pipeline_override=mock_pipeline)

    assert res["language_code"] == "en"
    assert res["language_name"] == "English"
    assert res["status"] == "detected"


def test_detect_language_multilingual_with_mocked_spark_nlp():
    """Verify non-English language detection (Spanish, French, German)."""
    mock_pipeline = MagicMock()

    # Spanish
    mock_pipeline.annotate.return_value = {"language": ["es"]}
    res_es = detect_language("Este es un proyecto completo de procesamiento de lenguaje natural y análisis de redes sociales.", pipeline_override=mock_pipeline)
    assert res_es["language_code"] == "es"
    assert res_es["language_name"] == "Spanish"
    assert res_es["status"] == "detected"

    # French
    mock_pipeline.annotate.return_value = {"language": ["fr"]}
    res_fr = detect_language("Ceci est un projet d'intelligence des medias sociaux utilisant Spark NLP et Python.", pipeline_override=mock_pipeline)
    assert res_fr["language_code"] == "fr"
    assert res_fr["language_name"] == "French"
    assert res_fr["status"] == "detected"


def test_detect_language_undetermined_prediction():
    """Verify handling when model returns unknown or empty prediction."""
    mock_pipeline = MagicMock()
    mock_pipeline.annotate.return_value = {"language": ["unk"]}

    res = detect_language("Random obscure characters 12345 67890 qwerty zxcvbnm.", pipeline_override=mock_pipeline)
    assert res["language_code"] == "und"
    assert res["language_name"] == "Undetermined"
    assert res["status"] == "undetermined"


def test_detect_language_handles_annotation_exception():
    """Verify exceptions during inference are caught safely without crashing."""
    mock_pipeline = MagicMock()
    mock_pipeline.annotate.side_effect = RuntimeError("Spark JVM connection interrupted")

    res = detect_language("A valid text string that encounters a model runtime exception.", pipeline_override=mock_pipeline)
    assert res["language_code"] == "und"
    assert res["status"] == "error"
    assert "Spark JVM" in res["message"]


def test_get_language_detector_pipeline_caching():
    """Verify pipeline singleton caching reuses existing instance."""
    with patch("src.preprocessing.language_detector._CACHED_LIGHT_PIPELINE", MagicMock()) as cached_instance:
        pipeline = get_language_detector_pipeline()
        assert pipeline == cached_instance

"""Language Detection Module using Spark NLP (Company Step 2).

Integrates Spark NLP's pretrained LanguageDetectorDL model for multilingual
social media language detection.
Handles edge cases: empty strings, short text (<140 char uncertainty), unknown languages,
and runtime prerequisite validation (Java 17, Python <= 3.12).
"""

from typing import Any
from src.preprocessing.spark_session import (
    check_spark_nlp_prerequisites,
    get_spark_session,
    is_spark_nlp_available,
)

# Standard ISO 639-1 language code to readable name mapping
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "und": "Undetermined",
}

# Module-level cache for pretrained Spark NLP LightPipeline
_CACHED_LIGHT_PIPELINE: Any = None


def get_language_detector_pipeline() -> Any:
    """Builds and caches a Spark NLP LanguageDetectorDL pipeline.

    Reuses the pretrained model across invocations to avoid downloading or
    rebuilding the pipeline for every record.
    """
    global _CACHED_LIGHT_PIPELINE

    if _CACHED_LIGHT_PIPELINE is not None:
        return _CACHED_LIGHT_PIPELINE

    # Ensure runtime prerequisites are met
    compatible, reason = check_spark_nlp_prerequisites()
    if not compatible:
        raise RuntimeError(f"Spark NLP language detector unavailable: {reason}")

    try:
        from pyspark.ml import Pipeline
        from sparknlp.annotator import LanguageDetectorDL
        from sparknlp.base import DocumentAssembler, LightPipeline

        spark = get_spark_session()

        document_assembler = (
            DocumentAssembler()
            .setInputCol("text")
            .setOutputCol("document")
        )

        language_detector = (
            LanguageDetectorDL.pretrained("ld_wiki_tatoeba_cnn_21", "xx")
            .setInputCols(["document"])
            .setOutputCol("language")
            .setThreshold(0.5)
        )

        pipeline = Pipeline(stages=[document_assembler, language_detector])
        empty_df = spark.createDataFrame([[""]]).toDF("text")
        pipeline_model = pipeline.fit(empty_df)

        _CACHED_LIGHT_PIPELINE = LightPipeline(pipeline_model)
        return _CACHED_LIGHT_PIPELINE
    except Exception as err:
        _CACHED_LIGHT_PIPELINE = None
        raise RuntimeError(f"Failed to initialize Spark NLP LanguageDetectorDL: {err}") from err


def detect_language(
    text: Any,
    pipeline_override: Any = None,
) -> dict[str, Any]:
    """Detects the natural language of input text using Spark NLP.

    Output Schema:
    {
        "language_code": "en",
        "language_name": "English",
        "status": "detected"
    }

    Edge Cases:
    1. Null/empty text -> language_code: "und", status: "empty_text".
    2. Short text (<15 characters) -> language_code: "und", status: "uncertain_short_text".
    3. Prerequisite failure -> status: "prerequisite_missing".
    4. Low confidence/unsupported -> language_code: "und", status: "undetermined".
    """
    if text is None:
        return {
            "language_code": "und",
            "language_name": "Undetermined",
            "status": "empty_text",
        }

    clean = str(text).strip()
    if not clean:
        return {
            "language_code": "und",
            "language_name": "Undetermined",
            "status": "empty_text",
        }

    # Spark NLP LanguageDetectorDL is most accurate for longer text (> 140 chars)
    # Very short text (< 15 chars like 'ok', 'yes', 'lol') cannot be reliably identified
    if len(clean) < 15:
        return {
            "language_code": "und",
            "language_name": "Undetermined",
            "status": "uncertain_short_text",
        }

    # Verify Spark NLP / Java readiness
    if pipeline_override is None and not is_spark_nlp_available():
        _, reason = check_spark_nlp_prerequisites()
        return {
            "language_code": "und",
            "language_name": "Undetermined",
            "status": "prerequisite_missing",
            "message": reason or "Spark NLP prerequisites not satisfied.",
        }

    try:
        pipeline = pipeline_override or get_language_detector_pipeline()
        annotation = pipeline.annotate(clean)
        lang_list = annotation.get("language", [])

        if not lang_list or not lang_list[0]:
            return {
                "language_code": "und",
                "language_name": "Undetermined",
                "status": "undetermined",
            }

        raw_code = str(lang_list[0]).lower().strip()
        if raw_code in ("unk", "und", "unknown"):
            return {
                "language_code": "und",
                "language_name": "Undetermined",
                "status": "undetermined",
            }

        name = LANGUAGE_NAMES.get(raw_code, raw_code.upper())
        result: dict[str, Any] = {
            "language_code": raw_code,
            "language_name": name,
            "status": "detected",
        }

        # Attempt to extract confidence score if full annotations are available
        if hasattr(pipeline, "fullAnnotate"):
            try:
                full_res = pipeline.fullAnnotate(clean)
                if full_res and full_res[0].get("language"):
                    ann_obj = full_res[0]["language"][0]
                    meta = getattr(ann_obj, "metadata", {}) or {}
                    raw_val = meta.get(raw_code)
                    if raw_val is not None:
                        result["confidence"] = round(float(raw_val), 4)
            except Exception:
                pass

        return result

    except Exception as err:
        return {
            "language_code": "und",
            "language_name": "Undetermined",
            "status": "error",
            "message": str(err),
        }

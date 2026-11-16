"""Multilingual Sentiment Analyzer (Company Step 3).

Provides transformer-based sentiment analysis for multilingual social media text
(Reddit posts, YouTube comments/descriptions).
Standardizes labels into 'positive', 'neutral', 'negative' with normalized confidence scores.
Supports lazy loading and dependency injection for offline testing.
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Standard sentiment labels
LABEL_POSITIVE = "positive"
LABEL_NEUTRAL = "neutral"
LABEL_NEGATIVE = "negative"
LABEL_UNDETERMINED = "undetermined"

# Mapping from common transformer output labels to standardized labels
LABEL_MAPPINGS: dict[str, str] = {
    # Standard lowercase
    "positive": LABEL_POSITIVE,
    "neutral": LABEL_NEUTRAL,
    "negative": LABEL_NEGATIVE,
    # Standard uppercase / abbreviated
    "pos": LABEL_POSITIVE,
    "neu": LABEL_NEUTRAL,
    "neg": LABEL_NEGATIVE,
    # CardiffNLP / HuggingFace RoBERTa indexed labels
    "label_0": LABEL_NEGATIVE,  # CardiffNLP: 0 -> Negative
    "label_1": LABEL_NEUTRAL,   # CardiffNLP: 1 -> Neutral
    "label_2": LABEL_POSITIVE,  # CardiffNLP: 2 -> Positive
    # Star ratings (e.g. multilingual review models)
    "1 star": LABEL_NEGATIVE,
    "2 stars": LABEL_NEGATIVE,
    "3 stars": LABEL_NEUTRAL,
    "4 stars": LABEL_POSITIVE,
    "5 stars": LABEL_POSITIVE,
}

DEFAULT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"


class MultilingualSentimentAnalyzer:
    """Multilingual Sentiment Analyzer supporting pretrained transformer models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pipeline_override: Optional[Callable[..., Any]] = None,
        device: Optional[Any] = None,
        max_length: int = 512,
    ) -> None:
        """Initializes the MultilingualSentimentAnalyzer.

        Args:
            model_name: HuggingFace model identifier.
            pipeline_override: Optional callable or mock pipeline for testing.
            device: Optional device identifier (e.g. 'cpu', 'cuda', -1).
            max_length: Maximum sequence length for truncation.
        """
        self.model_name = model_name
        self.pipeline_override = pipeline_override
        self.device = device
        self.max_length = max_length if max_length != 512 else 128
        self._pipeline: Optional[Any] = pipeline_override

    def _load_pipeline(self) -> Any:
        """Lazily loads the HuggingFace sentiment analysis pipeline.

        Returns:
            A callable pipeline for sentiment classification.

        Raises:
            RuntimeError: If transformers/torch cannot be imported and no override exists.
        """
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            logger.error("transformers package is not installed: %s", exc)
            raise RuntimeError(
                "HuggingFace 'transformers' and 'torch' are required for live model inference. "
                "Install them or provide a pipeline_override for testing."
            ) from exc

        kwargs: dict[str, Any] = {
            "task": "sentiment-analysis",
            "model": self.model_name,
            "tokenizer": self.model_name,
            "truncation": True,
            "max_length": self.max_length,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        logger.info("Loading sentiment analysis pipeline with model '%s'...", self.model_name)
        self._pipeline = pipeline(**kwargs)
        return self._pipeline

    @staticmethod
    def normalize_label(raw_label: str) -> str:
        """Normalizes raw model label strings into standard classes.

        Args:
            raw_label: Model prediction label (e.g. 'LABEL_0', 'positive', 'POS').

        Returns:
            Standardized label: 'positive', 'neutral', 'negative', or 'undetermined'.
        """
        if not isinstance(raw_label, str):
            return LABEL_UNDETERMINED

        cleaned = raw_label.strip().lower()
        return LABEL_MAPPINGS.get(cleaned, cleaned if cleaned in (LABEL_POSITIVE, LABEL_NEUTRAL, LABEL_NEGATIVE) else LABEL_UNDETERMINED)

    def analyze_text(self, text: Any) -> dict[str, Any]:
        """Analyzes sentiment for a single text string.

        Args:
            text: Input text string.

        Returns:
            Dict containing 'label', 'score', and 'status'.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return {
                "label": LABEL_UNDETERMINED,
                "score": 0.0,
                "status": "empty_text",
            }

        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(
        self,
        texts: list[Any],
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Analyzes sentiment for a batch of text inputs.

        Handles mixed inputs (empty strings, None, valid texts) safely without crashing.

        Args:
            texts: List of input texts.
            batch_size: Batch size for pipeline inference.

        Returns:
            List of dicts each containing 'label', 'score', and 'status'.
        """
        if not isinstance(texts, list):
            raise TypeError("Expected texts to be a list.")

        if not texts:
            return []

        results: list[dict[str, Any]] = [
            {"label": LABEL_UNDETERMINED, "score": 0.0, "status": "empty_text"}
            for _ in texts
        ]

        # Filter indices of valid non-empty strings
        valid_indices: list[int] = []
        valid_texts: list[str] = []

        for idx, item in enumerate(texts):
            if isinstance(item, str) and item.strip():
                valid_indices.append(idx)
                valid_texts.append(item.strip())

        if not valid_indices:
            return results

        try:
            pipe = self._load_pipeline()
            try:
                import torch
                with torch.inference_mode():
                    raw_predictions = pipe(valid_texts, batch_size=batch_size)
            except (ImportError, Exception):
                raw_predictions = pipe(valid_texts, batch_size=batch_size)

            # Ensure raw_predictions matches valid_indices
            for idx, raw_pred in zip(valid_indices, raw_predictions):
                # raw_pred might be a dict {'label': '...', 'score': 0.9}
                # or a list of dicts if top_k is used
                if isinstance(raw_pred, list) and raw_pred:
                    raw_pred = raw_pred[0]

                raw_label = raw_pred.get("label", "")
                raw_score = raw_pred.get("score", 0.0)

                norm_label = self.normalize_label(str(raw_label))
                clamped_score = max(0.0, min(1.0, float(raw_score)))

                results[idx] = {
                    "label": norm_label,
                    "score": round(clamped_score, 4),
                    "status": "success",
                }
        except Exception as exc:
            logger.error("Sentiment analysis batch inference failed: %s", exc)
            for idx in valid_indices:
                results[idx] = {
                    "label": LABEL_UNDETERMINED,
                    "score": 0.0,
                    "status": "error",
                    "error": str(exc),
                }

        return results

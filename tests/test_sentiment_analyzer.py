"""Unit tests for MultilingualSentimentAnalyzer and Sentiment Pipeline (Company Step 3).

Executes fast, deterministic unit tests completely offline with mock pipelines.
Verifies label normalization, input sanitization, batch handling, and pipeline orchestration.
"""

import json
from pathlib import Path
from typing import Any
import pytest

from src.sentiment.sentiment_analyzer import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    LABEL_UNDETERMINED,
    MultilingualSentimentAnalyzer,
)
from src.sentiment.sentiment_pipeline import (
    extract_reddit_text,
    extract_youtube_text,
    run_sentiment_pipeline,
)


class DummyMockPipeline:
    """Mock HuggingFace pipeline for deterministic testing."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.cursor = 0
        self.last_inputs: list[str] = []

    def __call__(self, texts: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        self.call_count += 1
        self.last_inputs = texts
        if self.responses:
            res: list[dict[str, Any]] = []
            for _ in texts:
                res.append(self.responses[self.cursor % len(self.responses)])
                self.cursor += 1
            return res
        # Default mock response: positive sentiment
        return [{"label": "positive", "score": 0.95} for _ in texts]


# ==============================================================================
# 1. Label Normalization Tests
# ==============================================================================

def test_normalize_standard_labels() -> None:
    """Tests standard lowercase label normalization."""
    assert MultilingualSentimentAnalyzer.normalize_label("positive") == LABEL_POSITIVE
    assert MultilingualSentimentAnalyzer.normalize_label("neutral") == LABEL_NEUTRAL
    assert MultilingualSentimentAnalyzer.normalize_label("negative") == LABEL_NEGATIVE


def test_normalize_case_and_whitespace() -> None:
    """Tests case-insensitive and whitespace handling in label normalization."""
    assert MultilingualSentimentAnalyzer.normalize_label("  POSITIVE  ") == LABEL_POSITIVE
    assert MultilingualSentimentAnalyzer.normalize_label("Neutral") == LABEL_NEUTRAL
    assert MultilingualSentimentAnalyzer.normalize_label("NEGATIVE\n") == LABEL_NEGATIVE


def test_normalize_abbreviations() -> None:
    """Tests abbreviated label formats (POS, NEU, NEG)."""
    assert MultilingualSentimentAnalyzer.normalize_label("pos") == LABEL_POSITIVE
    assert MultilingualSentimentAnalyzer.normalize_label("POS") == LABEL_POSITIVE
    assert MultilingualSentimentAnalyzer.normalize_label("neu") == LABEL_NEUTRAL
    assert MultilingualSentimentAnalyzer.normalize_label("neg") == LABEL_NEGATIVE


def test_normalize_cardiffnlp_indexed_labels() -> None:
    """Tests CardiffNLP RoBERTa indexed labels (LABEL_0 -> negative, LABEL_1 -> neutral, LABEL_2 -> positive)."""
    assert MultilingualSentimentAnalyzer.normalize_label("LABEL_0") == LABEL_NEGATIVE
    assert MultilingualSentimentAnalyzer.normalize_label("label_0") == LABEL_NEGATIVE
    assert MultilingualSentimentAnalyzer.normalize_label("LABEL_1") == LABEL_NEUTRAL
    assert MultilingualSentimentAnalyzer.normalize_label("LABEL_2") == LABEL_POSITIVE


def test_normalize_star_ratings() -> None:
    """Tests star rating labels conversion to sentiment classes."""
    assert MultilingualSentimentAnalyzer.normalize_label("1 star") == LABEL_NEGATIVE
    assert MultilingualSentimentAnalyzer.normalize_label("2 stars") == LABEL_NEGATIVE
    assert MultilingualSentimentAnalyzer.normalize_label("3 stars") == LABEL_NEUTRAL
    assert MultilingualSentimentAnalyzer.normalize_label("4 stars") == LABEL_POSITIVE
    assert MultilingualSentimentAnalyzer.normalize_label("5 stars") == LABEL_POSITIVE


def test_normalize_unknown_and_invalid_labels() -> None:
    """Tests that unrecognized or invalid types map to undetermined."""
    assert MultilingualSentimentAnalyzer.normalize_label("gibberish_label") == LABEL_UNDETERMINED
    assert MultilingualSentimentAnalyzer.normalize_label("") == LABEL_UNDETERMINED
    assert MultilingualSentimentAnalyzer.normalize_label(None) == LABEL_UNDETERMINED  # type: ignore
    assert MultilingualSentimentAnalyzer.normalize_label(123) == LABEL_UNDETERMINED  # type: ignore


# ==============================================================================
# 2. Text Analysis & Empty Input Handling Tests
# ==============================================================================

def test_analyze_empty_and_whitespace_inputs() -> None:
    """Tests that empty, whitespace, None, and non-string inputs are handled safely without calling model."""
    mock_pipe = DummyMockPipeline()
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=mock_pipe)

    for empty_val in ["", "   ", "\t\n", None, 12345]:
        res = analyzer.analyze_text(empty_val)
        assert res["label"] == LABEL_UNDETERMINED
        assert res["score"] == 0.0
        assert res["status"] == "empty_text"

    # Ensure pipeline was never invoked for empty texts
    assert mock_pipe.call_count == 0


def test_analyze_valid_text_with_mock() -> None:
    """Tests single text analysis with mocked positive prediction."""
    mock_pipe = DummyMockPipeline([{"label": "LABEL_2", "score": 0.8876}])
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=mock_pipe)

    result = analyzer.analyze_text("This is an exceptional feature release! Loving the performance.")
    assert result["label"] == LABEL_POSITIVE
    assert result["score"] == 0.8876
    assert result["status"] == "success"
    assert mock_pipe.call_count == 1


# ==============================================================================
# 3. Batch Analysis Tests
# ==============================================================================

def test_analyze_batch_empty_list() -> None:
    """Tests batch analysis on empty list returns empty list."""
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=DummyMockPipeline())
    assert analyzer.analyze_batch([]) == []


def test_analyze_batch_type_error() -> None:
    """Tests non-list input raises TypeError."""
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=DummyMockPipeline())
    with pytest.raises(TypeError, match="Expected texts to be a list"):
        analyzer.analyze_batch("not a list")  # type: ignore


def test_analyze_batch_mixed_inputs() -> None:
    """Tests batch analysis with a mix of valid, empty, None, and whitespace strings."""
    mock_pipe = DummyMockPipeline([
        {"label": "LABEL_0", "score": 0.9123},
        {"label": "LABEL_2", "score": 0.8456},
    ])
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=mock_pipe)

    texts = [
        "",                             # index 0: empty
        "Terrible experience, bugged.", # index 1: negative
        "   ",                          # index 2: whitespace
        "Fantastic work, loved it!",    # index 3: positive
        None,                           # index 4: None
    ]

    results = analyzer.analyze_batch(texts)
    assert len(results) == 5

    # Index 0: empty
    assert results[0]["label"] == LABEL_UNDETERMINED
    assert results[0]["status"] == "empty_text"

    # Index 1: negative
    assert results[1]["label"] == LABEL_NEGATIVE
    assert results[1]["score"] == 0.9123
    assert results[1]["status"] == "success"

    # Index 2: whitespace
    assert results[2]["label"] == LABEL_UNDETERMINED
    assert results[2]["status"] == "empty_text"

    # Index 3: positive
    assert results[3]["label"] == LABEL_POSITIVE
    assert results[3]["score"] == 0.8456
    assert results[3]["status"] == "success"

    # Index 4: None
    assert results[4]["label"] == LABEL_UNDETERMINED
    assert results[4]["status"] == "empty_text"

    # Pipeline was called once with only 2 valid texts
    assert mock_pipe.call_count == 1
    assert len(mock_pipe.last_inputs) == 2


def test_analyze_batch_score_clamping() -> None:
    """Tests that scores are clamped between 0.0 and 1.0."""
    mock_pipe = DummyMockPipeline([
        {"label": "positive", "score": 1.25},
        {"label": "negative", "score": -0.5},
    ])
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=mock_pipe)

    results = analyzer.analyze_batch(["Great!", "Awful!"])
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.0


def test_analyze_batch_pipeline_exception_handling() -> None:
    """Tests that exceptions inside the pipeline are caught and reported as error status."""
    def broken_pipe(texts: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        raise RuntimeError("Inference runtime OOM simulated error")

    analyzer = MultilingualSentimentAnalyzer(pipeline_override=broken_pipe)
    results = analyzer.analyze_batch(["Some valid text"])

    assert len(results) == 1
    assert results[0]["label"] == LABEL_UNDETERMINED
    assert results[0]["status"] == "error"
    assert "Inference runtime OOM" in results[0]["error"]


def test_lazy_loading_error_when_missing_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that RuntimeError is raised when transformers is missing and no override provided."""
    import builtins
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    analyzer = MultilingualSentimentAnalyzer()
    with pytest.raises(RuntimeError, match="HuggingFace 'transformers' and 'torch' are required"):
        analyzer._load_pipeline()


# ==============================================================================
# 4. Text Extraction & Pipeline Integration Tests
# ==============================================================================

def test_extract_reddit_text() -> None:
    """Tests text extraction from Reddit records with title and text fallback."""
    rec1 = {
        "title_cleaned": "Cleaned Title",
        "text_cleaned": "Cleaned Body Text",
    }
    assert extract_reddit_text(rec1) == "Cleaned Title Cleaned Body Text"

    rec2 = {
        "title": "Raw Title Only",
    }
    assert extract_reddit_text(rec2) == "Raw Title Only"


def test_extract_youtube_text() -> None:
    """Tests text extraction from YouTube records with description fallback."""
    rec = {
        "title_cleaned": "Video Title",
        "description_cleaned": "Video Description",
    }
    assert extract_youtube_text(rec) == "Video Title Video Description"


def test_run_sentiment_pipeline_end_to_end(tmp_path: Path) -> None:
    """Tests end-to-end sentiment pipeline non-destructive execution with mock analyzer."""
    reddit_path = tmp_path / "reddit_test.json"
    youtube_path = tmp_path / "youtube_test.json"
    output_path = tmp_path / "output_test.json"

    reddit_data = [
        {
            "post_id": "r1",
            "title_cleaned": "Exciting AI breakthrough announced today",
            "text_cleaned": "Truly impressive multilingual benchmark scores.",
            "created_utc": 1730101200,
            "collected_at": "2026-11-05T16:54:13+00:00",
        },
        {
            "post_id": "r2",
            "title_cleaned": "Frustrating server downtime during peak hours",
            "text_cleaned": "Cannot connect, keeps throwing error 500.",
            "created_utc": 1730187600,
            "collected_at": "2026-11-05T16:54:13+00:00",
        },
    ]

    youtube_data = [
        {
            "video_id": "y1",
            "title_cleaned": "Deep dive into vector databases",
            "description_cleaned": "Standard technical tutorial with neutral commentary.",
            "published_at": "2026-11-01T12:00:00Z",
            "collected_at": "2026-11-06T13:00:00+00:00",
        }
    ]

    with open(reddit_path, "w", encoding="utf-8") as f:
        json.dump(reddit_data, f)
    with open(youtube_path, "w", encoding="utf-8") as f:
        json.dump(youtube_data, f)

    # Mock predictions: 1st positive, 2nd negative, 3rd neutral
    mock_pipe = DummyMockPipeline([
        {"label": "positive", "score": 0.92},
        {"label": "negative", "score": 0.88},
        {"label": "neutral", "score": 0.75},
    ])
    analyzer = MultilingualSentimentAnalyzer(pipeline_override=mock_pipe)

    result = run_sentiment_pipeline(
        reddit_input_path=reddit_path,
        youtube_input_path=youtube_path,
        output_path=output_path,
        analyzer=analyzer,
    )

    assert result["status"] == "success"
    assert result["total_records_processed"] == 3
    assert output_path.exists()

    # Verify original files remain unmodified
    with open(reddit_path, "r", encoding="utf-8") as f:
        loaded_raw = json.load(f)
        assert "sentiment_label" not in loaded_raw[0]

    # Verify output contents
    with open(output_path, "r", encoding="utf-8") as f:
        saved_output = json.load(f)

    assert saved_output["summary"]["total"] == 3
    assert saved_output["summary"]["positive"] == 1
    assert saved_output["summary"]["negative"] == 1
    assert saved_output["summary"]["neutral"] == 1
    assert saved_output["reddit_summary"]["total"] == 2
    assert saved_output["youtube_summary"]["total"] == 1
    assert "daily" in saved_output["temporal_aggregations"]
    assert "weekly" in saved_output["temporal_aggregations"]
    assert len(saved_output["reddit_records"]) == 2
    assert len(saved_output["youtube_records"]) == 1
    assert saved_output["reddit_records"][0]["sentiment_label"] == "positive"
    assert saved_output["reddit_records"][1]["sentiment_label"] == "negative"
    assert saved_output["youtube_records"][0]["sentiment_label"] == "neutral"

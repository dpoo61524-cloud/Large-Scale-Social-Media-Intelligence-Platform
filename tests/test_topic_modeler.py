"""Unit tests for SocialTopicModeler (Company Step 4).

Executes fast, deterministic unit tests completely offline with mock BERTopic models.
Verifies keyword extraction, outlier handling (Topic -1), cosine similarity matrix,
dynamic c-TF-IDF keyword tracking, hierarchical topic reduction, and lazy loading.
"""

from typing import Any
import pytest

from src.topics.topic_modeler import (
    DEFAULT_EMBEDDING_MODEL,
    OUTLIER_TOPIC_ID,
    SocialTopicModeler,
)


class MockDataFrame:
    """Mock DataFrame for lightweight, dependency-free testing of topic info."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def iterrows(self):
        for idx, row in enumerate(self.rows):
            yield idx, row


class DummyMockBERTopic:
    """Mock BERTopic model for memory-safe and offline unit testing."""

    def __init__(self, topic_assignments: list[int] | None = None) -> None:
        self.topic_assignments = topic_assignments or [0, 1, 0]
        self.call_count = 0
        self.reduced_nr: int | None = None
        self.last_docs: list[str] = []

    def fit_transform(self, documents: list[str]) -> tuple[list[int], list[float]]:
        self.call_count += 1
        self.last_docs = documents
        topics = [self.topic_assignments[i % len(self.topic_assignments)] for i in range(len(documents))]
        probs = [0.85] * len(documents)
        return topics, probs

    def get_topic_info(self) -> MockDataFrame:
        if self.reduced_nr == 1:
            return MockDataFrame([
                {"Topic": 0, "Name": "0_merged_nlp_pipelines", "Count": 8},
                {"Topic": -1, "Name": "-1_noise_outliers", "Count": 2},
            ])
        return MockDataFrame([
            {"Topic": 0, "Name": "0_deep_learning_nlp", "Count": 5},
            {"Topic": 1, "Name": "1_data_pipelines_spark", "Count": 3},
            {"Topic": -1, "Name": "-1_noise_outliers", "Count": 2},
        ])

    def get_topic(self, topic_id: int) -> list[tuple[str, float]] | bool:
        topics_map = {
            0: [("deep", 0.8), ("learning", 0.75), ("nlp", 0.65)],
            1: [("pipelines", 0.85), ("spark", 0.7), ("streaming", 0.6)],
            -1: [("random", 0.1), ("noise", 0.05)],
        }
        return topics_map.get(topic_id, False)

    def reduce_topics(self, docs: list[str], nr_topics: int) -> None:
        self.reduced_nr = nr_topics

    def get_dynamic_topic_keywords(self, topic_id: int, timestamps: Any = None) -> dict[str, Any]:
        return {
            "2024-10-28": [{"word": "deep", "weight": 0.8}],
            "2024-10-29": [{"word": "transformer", "weight": 0.9}],
        }

    def transform(self, documents: list[str]) -> tuple[list[int], list[float]]:
        topics = [0] * len(documents)
        probs = [0.9] * len(documents)
        return topics, probs


# ==============================================================================
# 1. Initialization and Parameter Tests
# ==============================================================================

def test_topic_modeler_initialization_defaults() -> None:
    """Tests default configuration parameters."""
    modeler = SocialTopicModeler()
    assert modeler.embedding_model_name == DEFAULT_EMBEDDING_MODEL
    assert modeler.n_neighbors == 15
    assert modeler.min_dist == 0.1
    assert modeler.min_cluster_size == 2
    assert modeler.ngram_range == (1, 2)
    assert modeler.min_topic_size == 2
    assert modeler.model_override is None
    assert modeler._fitted is False


def test_topic_modeler_custom_parameters() -> None:
    """Tests custom model name and minimum topic size."""
    modeler = SocialTopicModeler(
        embedding_model_name="all-MiniLM-L6-v2",
        n_neighbors=10,
        min_dist=0.05,
        min_cluster_size=3,
        ngram_range=(1, 3),
        min_topic_size=5,
    )
    assert modeler.embedding_model_name == "all-MiniLM-L6-v2"
    assert modeler.n_neighbors == 10
    assert modeler.min_dist == 0.05
    assert modeler.ngram_range == (1, 3)
    assert modeler.min_topic_size == 5


# ==============================================================================
# 2. Fit Transform & Edge Cases
# ==============================================================================

def test_fit_transform_empty_list() -> None:
    """Tests fit_transform on empty document list returns empty lists."""
    modeler = SocialTopicModeler(model_override=DummyMockBERTopic())
    topics, probs = modeler.fit_transform([])
    assert topics == []
    assert probs == []


def test_fit_transform_invalid_type() -> None:
    """Tests fit_transform raises TypeError for non-list inputs."""
    modeler = SocialTopicModeler(model_override=DummyMockBERTopic())
    with pytest.raises(TypeError, match="Expected documents to be a list"):
        modeler.fit_transform("not a list")  # type: ignore


def test_fit_transform_all_whitespace_and_empty_strings() -> None:
    """Tests fit_transform on whitespace-only strings returns outlier topics."""
    modeler = SocialTopicModeler(model_override=DummyMockBERTopic())
    docs = ["", "   ", "\t\n"]
    topics, probs = modeler.fit_transform(docs)
    assert topics == [OUTLIER_TOPIC_ID, OUTLIER_TOPIC_ID, OUTLIER_TOPIC_ID]
    assert probs == [0.0, 0.0, 0.0]


def test_fit_transform_with_mock_model() -> None:
    """Tests successful fit_transform and assignment mapping."""
    mock = DummyMockBERTopic(topic_assignments=[0, 1])
    modeler = SocialTopicModeler(model_override=mock)

    docs = ["AI and transformers in production", "Building scalable Spark data pipelines"]
    topics, probs = modeler.fit_transform(docs)

    assert len(topics) == 2
    assert topics == [0, 1]
    assert len(probs) == 2
    assert probs[0] == 0.85
    assert mock.call_count == 1
    assert modeler._fitted is True


def test_fit_transform_mixed_valid_and_empty_docs() -> None:
    """Tests that empty documents among valid docs are assigned outlier topic -1."""
    mock = DummyMockBERTopic(topic_assignments=[0, 1])
    modeler = SocialTopicModeler(model_override=mock)

    docs = ["Valid document one", "", "Valid document two"]
    topics, probs = modeler.fit_transform(docs)

    assert len(topics) == 3
    assert topics[0] == 0
    assert topics[1] == OUTLIER_TOPIC_ID
    assert topics[2] == 1
    assert probs[1] == 0.0


# ==============================================================================
# 3. Topic Info & Keywords Extraction
# ==============================================================================

def test_get_topic_info() -> None:
    """Tests extraction of structured topic summary DataFrame rows."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    modeler._fitted = True

    info = modeler.get_topic_info()
    assert len(info) == 3
    assert info[0]["topic_id"] == 0
    assert info[0]["name"] == "0_deep_learning_nlp"
    assert info[0]["count"] == 5
    assert len(info[0]["keywords"]) > 0


def test_get_topic_keywords_valid_and_outlier() -> None:
    """Tests representative keyword extraction for valid and outlier topics."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    modeler._fitted = True

    kw_0 = modeler.get_topic_keywords(0, top_n=2)
    assert len(kw_0) == 2
    assert kw_0[0]["word"] == "deep"
    assert kw_0[0]["weight"] == 0.8

    kw_outlier = modeler.get_topic_keywords(-1)
    assert len(kw_outlier) > 0
    assert kw_outlier[0]["word"] == "random"

    kw_none = modeler.get_topic_keywords(999)
    assert kw_none == []


def test_get_topic_info_when_not_fitted() -> None:
    """Tests get_topic_info returns empty list when model is unfitted."""
    modeler = SocialTopicModeler()
    assert modeler.get_topic_info() == []
    assert modeler.get_topic_keywords(0) == []


# ==============================================================================
# 4. Hierarchical Reduction, Similarity Matrix & Dynamic Keywords
# ==============================================================================

def test_hierarchical_reduction() -> None:
    """Tests hierarchical topic reduction reduces topic counts on fitted model."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    modeler._fitted = True

    updated_info = modeler.hierarchical_reduction(nr_topics=1)
    assert mock.reduced_nr == 1
    assert len(updated_info) == 2  # Topic 0 (merged) and Topic -1


def test_hierarchical_reduction_unfitted_raises() -> None:
    """Tests that calling hierarchical reduction on unfitted model raises RuntimeError."""
    modeler = SocialTopicModeler()
    with pytest.raises(RuntimeError, match="Model must be fitted before hierarchical reduction"):
        modeler.hierarchical_reduction(nr_topics=5)


def test_calculate_topic_similarity_matrix() -> None:
    """Tests cosine similarity matrix computation across discovered topics."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    modeler._fitted = True

    matrix_res = modeler.calculate_topic_similarity_matrix()
    assert "topic_ids" in matrix_res
    assert "matrix" in matrix_res
    assert "pairwise_similarities" in matrix_res

    topic_ids = matrix_res["topic_ids"]
    assert 0 in topic_ids
    assert 1 in topic_ids
    assert OUTLIER_TOPIC_ID not in topic_ids  # Outliers excluded from centroid matrix

    matrix = matrix_res["matrix"]
    assert len(matrix) == len(topic_ids)
    # Diagonal self-similarity must be 1.0
    for i in range(len(topic_ids)):
        assert matrix[i][i] == 1.0


def test_get_dynamic_topic_keywords() -> None:
    """Tests dynamic c-TF-IDF keyword extraction across time intervals."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    modeler._fitted = True

    dyn_kws = modeler.get_dynamic_topic_keywords(0)
    assert isinstance(dyn_kws, dict)
    assert "2024-10-28" in dyn_kws
    assert "2024-10-29" in dyn_kws
    assert dyn_kws["2024-10-29"][0]["word"] == "transformer"


# ==============================================================================
# 5. Transform on New Documents & Dependency Checks
# ==============================================================================

def test_transform_unfitted_raises_runtime_error() -> None:
    """Tests that calling transform before fitting raises RuntimeError."""
    modeler = SocialTopicModeler()
    with pytest.raises(RuntimeError, match="Model must be fitted"):
        modeler.transform(["New document"])


def test_transform_success() -> None:
    """Tests transform on new documents using fitted model."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    modeler._fitted = True

    topics, probs = modeler.transform(["New AI research paper"])
    assert topics == [0]
    assert probs == [0.9]


def test_lazy_loading_error_when_missing_bertopic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that RuntimeError is raised when bertopic package is absent and no override is provided."""
    import builtins
    orig_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "bertopic":
            raise ImportError("No module named 'bertopic'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    modeler = SocialTopicModeler()
    with pytest.raises(RuntimeError, match="BERTopic is required for live topic modeling"):
        modeler._load_model()

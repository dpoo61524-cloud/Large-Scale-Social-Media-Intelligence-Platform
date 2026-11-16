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

    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: 2048.0)

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "bertopic":
            raise ImportError("No module named 'bertopic'")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    modeler = SocialTopicModeler()
    with pytest.raises(RuntimeError, match="BERTopic is required for live topic modeling"):
        modeler._load_model()


# ==============================================================================
# 6. Memory Guard, Statistical Fallback & Provenance Verification
# ==============================================================================

def test_native_memory_check_utility() -> None:
    """Verifies that get_available_memory_mb returns a non-negative float on Windows."""
    from src.topics.topic_modeler import get_available_memory_mb
    avail = get_available_memory_mb()
    assert isinstance(avail, float)
    # On Windows NT, should return a positive number; or -1.0 if not on Windows
    assert avail > 0.0 or avail == -1.0


def test_memory_preflight_guard_raises_memory_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that _load_model raises MemoryError when available RAM is below threshold."""
    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: 50.0)
    modeler = SocialTopicModeler()
    modeler.min_memory_mb = 150.0

    with pytest.raises(MemoryError, match="below safe threshold"):
        modeler._load_model()


def test_fit_transform_activates_fallback_on_memory_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that fit_transform cleanly routes to statistical fallback when memory is constrained."""
    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: 50.0)
    modeler = SocialTopicModeler()
    modeler.min_memory_mb = 150.0
    modeler.corpus_size_threshold = 2  # allow corpus to pass size check

    docs = [
        "Machine learning neural network deep transformer",
        "Natural language processing BERT tokenization embeddings",
        "Distributed computing Apache Spark streaming Kafka",
        "Database SQL query optimization indexing postgres",
        "Cloud kubernetes docker cluster orchestration",
    ]

    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == len(docs)
    assert len(probs) == len(docs)
    assert all(isinstance(t, int) for t in topics)
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probs)

    # Provenance verification: NEVER misrepresent fallback as neural bertopic
    assert modeler.execution_mode == "statistical_fallback"
    assert modeler.algorithm == "c-tf-idf_kmeans_fallback"
    assert modeler.fallback_reason is not None
    assert "below safe threshold" in modeler.fallback_reason

    provenance = modeler.get_provenance()
    assert provenance["execution_mode"] == "statistical_fallback"
    assert provenance["algorithm"] == "c-tf-idf_kmeans_fallback"
    assert provenance["execution_status"] == "fallback_success"
    assert provenance["model_name"] == "sklearn_tfidf_minibatch_kmeans"
    assert provenance["corpus_size"] == len(docs)
    assert provenance["valid_document_count"] == len(docs)


def test_fit_transform_fallback_on_small_corpus() -> None:
    """Verifies that small corpus (< 15 documents) routes deterministically to statistical fallback."""
    modeler = SocialTopicModeler()
    docs = [
        "Python data science pandas numpy scikit-learn",
        "Web development react typescript html css",
        "Machine learning pytorch tensor neural network",
        "Cloud computing aws azure gcp terraform",
    ]

    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == len(docs)
    assert modeler.execution_mode == "statistical_fallback"
    assert modeler.algorithm == "c-tf-idf_kmeans_fallback"
    assert "corpus_size_threshold" in str(modeler.fallback_reason)

    info = modeler.get_topic_info()
    assert len(info) >= 1
    for topic_item in info:
        assert "topic_id" in topic_item
        assert "name" in topic_item
        assert "count" in topic_item
        assert "keywords" in topic_item
        assert isinstance(topic_item["keywords"], list)

    # Similarity matrix computation over fallback topics
    sim = modeler.calculate_topic_similarity_matrix()
    assert "topic_ids" in sim
    assert "matrix" in sim
    assert "pairwise_similarities" in sim

    # Transform on new documents
    new_topics, new_probs = modeler.transform(["Deep learning pytorch models"])
    assert len(new_topics) == 1
    assert len(new_probs) == 1


def test_fallback_handles_duplicate_documents() -> None:
    """Verifies that 100% duplicate documents are handled safely without clustering failure."""
    modeler = SocialTopicModeler()
    docs = ["Natural language processing transformers BERT"] * 6

    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == 6
    assert len(probs) == 6
    assert all(isinstance(t, int) and t >= 0 for t in topics)
    assert modeler.execution_mode == "statistical_fallback"
    info = modeler.get_topic_info()
    assert len(info) >= 1


def test_fallback_handles_empty_and_whitespace_documents() -> None:
    """Verifies that empty and whitespace-only documents are routed to outlier topic -1."""
    modeler = SocialTopicModeler()
    docs = ["", "   ", "\n\t", "   \r\n "]

    topics, probs = modeler.fit_transform(docs)
    assert topics == [OUTLIER_TOPIC_ID] * 4
    assert probs == [0.0] * 4


def test_fallback_handles_stopwords_only() -> None:
    """Verifies that documents containing solely stop words are handled safely without crashing."""
    modeler = SocialTopicModeler()
    docs = ["the the the", "and and and", "of of of", "in in in"]

    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == 4
    assert len(probs) == 4
    assert all(isinstance(t, int) for t in topics)


def test_fallback_handles_single_document() -> None:
    """Verifies that a single document (below min_topic_size=2) is handled per spec."""
    modeler = SocialTopicModeler()
    docs = ["Single isolated post about artificial intelligence"]

    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == 1
    assert topics == [OUTLIER_TOPIC_ID]
    assert probs == [0.0]


def test_phase2_memory_guard_before_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the phase 2 memory check triggers fallback if memory drops prior to fitting."""
    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: 100.0)
    modeler = SocialTopicModeler()
    modeler.min_memory_mb = 1024.0
    modeler.corpus_size_threshold = 2

    # Mock the internal _load_model to return a dummy model object
    class DummyModel:
        def fit_transform(self, docs: list[str]) -> Any:
            raise AssertionError("fit_transform should not be called when phase 2 memory check fails")

    modeler._load_model = lambda sample_count=None: DummyModel()  # type: ignore

    docs = ["Doc 1 neural transformers", "Doc 2 spark kafka", "Doc 3 postgres sql"]
    topics, probs = modeler.fit_transform(docs)

    assert len(topics) == 3
    assert modeler.execution_mode == "statistical_fallback"
    assert "below safe threshold" in str(modeler.fallback_reason)


def test_mock_model_provenance_marked_mock_success() -> None:
    """Verifies that model_override results are explicitly labeled as mock_model and mock_success."""
    mock = DummyMockBERTopic()
    modeler = SocialTopicModeler(model_override=mock)
    docs = ["NLP deep learning", "Spark big data", "Misc chatter"]
    topics, probs = modeler.fit_transform(docs)

    provenance = modeler.get_provenance()
    assert provenance["execution_mode"] == "mock_model"
    assert provenance["algorithm"] == "mock"
    assert provenance["model_name"] == "mock_model"
    assert provenance["execution_status"] == "mock_success"
    # Ensure it is NEVER reported as neural_bertopic or fallback
    assert provenance["execution_mode"] != "neural_bertopic"
    assert provenance["execution_mode"] != "statistical_fallback"


def test_unknown_memory_fails_closed_in_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that _load_model fails closed with MemoryError when available RAM is unmeasurable."""
    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: -1.0)
    modeler = SocialTopicModeler()

    with pytest.raises(MemoryError, match="could not be measured"):
        modeler._load_model()


def test_unknown_memory_selects_fallback_in_fit_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that fit_transform cleanly routes to statistical fallback when memory is unmeasurable."""
    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: -1.0)
    modeler = SocialTopicModeler()
    modeler.corpus_size_threshold = 2

    docs = [
        "Machine learning neural network deep transformer",
        "Natural language processing BERT tokenization embeddings",
        "Distributed computing Apache Spark streaming Kafka",
        "Database SQL query optimization indexing postgres",
    ]
    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == 4
    assert modeler.execution_mode == "statistical_fallback"
    assert modeler.algorithm == "c-tf-idf_kmeans_fallback"
    assert "could not be measured" in str(modeler.fallback_reason)


def test_topic_modeler_neural_path_provenance_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies neural provenance contract (neural_bertopic, bertopic) when memory is sufficient."""
    monkeypatch.setattr("src.topics.topic_modeler.get_available_memory_mb", lambda: 2048.0)
    modeler = SocialTopicModeler()
    modeler.corpus_size_threshold = 2

    class MockBERTopicInstance:
        def fit_transform(self, docs: list[str]) -> tuple[list[int], list[float]]:
            return [0] * len(docs), [0.85] * len(docs)

    modeler._load_model = lambda sample_count=None: MockBERTopicInstance()  # type: ignore

    docs = [
        "Machine learning neural network deep transformer",
        "Natural language processing BERT tokenization embeddings",
        "Distributed computing Apache Spark streaming Kafka",
    ]
    topics, probs = modeler.fit_transform(docs)
    assert len(topics) == 3
    assert modeler.execution_mode == "neural_bertopic"
    assert modeler.algorithm == "bertopic"
    assert modeler.fallback_reason is None

    provenance = modeler.get_provenance()
    assert provenance["execution_mode"] == "neural_bertopic"
    assert provenance["algorithm"] == "bertopic"
    assert provenance["fallback_reason"] is None


def test_fresh_process_fallback_import_isolation() -> None:
    """Verifies in a fresh Python process that the statistical fallback path imports zero neural libraries."""
    import subprocess
    import sys

    script = """
import sys
from unittest.mock import patch

with patch('src.topics.topic_modeler.get_available_memory_mb', return_value=100.0):
    from src.topics.topic_modeler import SocialTopicModeler
    modeler = SocialTopicModeler()
    modeler.corpus_size_threshold = 2
    docs = [
        'Machine learning neural network deep transformer',
        'Natural language processing BERT tokenization embeddings',
        'Distributed computing Apache Spark streaming Kafka',
    ]
    topics, probs = modeler.fit_transform(docs)
    assert modeler.execution_mode == 'statistical_fallback'
    assert modeler.algorithm == 'c-tf-idf_kmeans_fallback'

forbidden = {'torch', 'bertopic', 'transformers', 'sentence_transformers'}
loaded = forbidden.intersection(set(sys.modules.keys()))
if loaded:
    print(f'FORBIDDEN_LOADED: {loaded}')
    sys.exit(1)
print('SUCCESS_ZERO_NEURAL_IMPORTS')
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert "SUCCESS_ZERO_NEURAL_IMPORTS" in result.stdout




"""Advanced Topic Modeler using BERTopic (Company Step 4).

Extracts latent discussion themes from social media text using modular, production-grade
c-TF-IDF representations, dynamic keyword tracking across temporal bins, hierarchical topic
reduction, and pairwise cosine topic similarity matrices.
"""

from collections import defaultdict
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OUTLIER_TOPIC_ID = -1


class SocialTopicModeler:
    """Advanced BERTopic-based topic modeler with dynamic c-TF-IDF and similarity matrices."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        min_cluster_size: int = 2,
        stop_words: str | list[str] = "english",
        ngram_range: tuple[int, int] = (1, 2),
        min_topic_size: int = 2,
        model_override: Optional[Any] = None,
    ) -> None:
        """Initializes the SocialTopicModeler with modular sub-component configurations.

        Args:
            embedding_model_name: SentenceTransformer model name for embeddings.
            n_neighbors: UMAP neighborhood size for dimensionality reduction.
            min_dist: UMAP minimum distance parameter.
            min_cluster_size: HDBSCAN minimum cluster size.
            stop_words: Vectorizer stop words.
            ngram_range: n-gram range tuple for c-TF-IDF keyword extraction.
            min_topic_size: Minimum documents to form a distinct topic.
            model_override: Optional mock or preconfigured model for testing.
        """
        self.embedding_model_name = embedding_model_name
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.min_topic_size = min_topic_size
        self.model_override = model_override
        self._model: Optional[Any] = model_override
        self._fitted: bool = model_override is not None
        self._last_documents: list[str] = []
        self._last_topics: list[int] = []

    def _load_model(self) -> Any:
        """Lazily loads the BERTopic model with configured sub-models.

        Returns:
            Configured BERTopic instance.

        Raises:
            RuntimeError: If bertopic is not installed and no override is provided.
        """
        if self._model is not None:
            return self._model

        try:
            from bertopic import BERTopic  # type: ignore
        except ImportError as exc:
            logger.error("bertopic package is not installed: %s", exc)
            raise RuntimeError(
                "BERTopic is required for live topic modeling. "
                "Install bertopic or provide a model_override for testing."
            ) from exc

        logger.info(
            "Initializing BERTopic with embedding model '%s' (min_topic_size=%d, n_neighbors=%d)...",
            self.embedding_model_name,
            self.min_topic_size,
            self.n_neighbors,
        )

        try:
            from hdbscan import HDBSCAN  # type: ignore
            from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
            from umap import UMAP  # type: ignore

            umap_model = UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric="cosine",
                random_state=42,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            vectorizer_model = CountVectorizer(
                stop_words=self.stop_words,
                ngram_range=self.ngram_range,
            )
            self._model = BERTopic(
                embedding_model=self.embedding_model_name,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                min_topic_size=self.min_topic_size,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("Sub-model initialization failed, using default BERTopic params: %s", exc)
            self._model = BERTopic(
                embedding_model=self.embedding_model_name,
                min_topic_size=self.min_topic_size,
                verbose=False,
            )

        return self._model

    def fit_transform(self, documents: list[str]) -> tuple[list[int], list[float]]:
        """Fits topic model on documents and predicts topic assignments.

        Args:
            documents: List of cleaned text documents.

        Returns:
            Tuple of (assigned topic IDs list, probabilities list).
        """
        if not isinstance(documents, list):
            raise TypeError("Expected documents to be a list of strings.")

        if not documents:
            logger.warning("Empty document list passed to fit_transform.")
            return [], []

        valid_indices: list[int] = []
        valid_docs: list[str] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, str) and doc.strip():
                valid_indices.append(idx)
                valid_docs.append(doc.strip())

        if not valid_docs:
            return [OUTLIER_TOPIC_ID] * len(documents), [0.0] * len(documents)

        if len(valid_docs) < self.min_topic_size and not self.model_override:
            logger.info("Corpus size (%d) < min_topic_size (%d); assigning all to outlier topic.", len(valid_docs), self.min_topic_size)
            return [OUTLIER_TOPIC_ID] * len(documents), [0.0] * len(documents)

        model = self._load_model()
        try:
            raw_topics, raw_probs = model.fit_transform(valid_docs)
            self._fitted = True
            self._last_documents = valid_docs
            self._last_topics = [int(t) for t in raw_topics]
        except Exception as exc:
            logger.error("BERTopic fit_transform failed: %s", exc)
            return [OUTLIER_TOPIC_ID] * len(documents), [0.0] * len(documents)

        all_topics = [OUTLIER_TOPIC_ID] * len(documents)
        all_probs = [0.0] * len(documents)

        for i, original_idx in enumerate(valid_indices):
            all_topics[original_idx] = int(raw_topics[i])
            prob_val = raw_probs[i] if raw_probs is not None and len(raw_probs) > i else 0.0
            if hasattr(prob_val, "__iter__"):
                prob_val = max(prob_val) if len(prob_val) > 0 else 0.0
            all_probs[original_idx] = round(float(prob_val), 4)

        return all_topics, all_probs

    def get_topic_info(self) -> list[dict[str, Any]]:
        """Returns structured information about discovered topics.

        Returns:
            List of dicts containing topic_id, name, count, and representative keywords.
        """
        if not self._fitted or self._model is None:
            return []

        try:
            info = self._model.get_topic_info()
            records: list[dict[str, Any]] = []

            if hasattr(info, "iterrows"):
                for _, row in info.iterrows():
                    tid = int(row.get("Topic", OUTLIER_TOPIC_ID))
                    name = str(row.get("Name", f"{tid}_topic"))
                    count = int(row.get("Count", 0))
                    keywords = self.get_topic_keywords(tid, top_n=5)
                    records.append({
                        "topic_id": tid,
                        "name": name,
                        "count": count,
                        "keywords": keywords,
                    })
            elif isinstance(info, list):
                for item in info:
                    if isinstance(item, dict):
                        tid = int(item.get("Topic", OUTLIER_TOPIC_ID))
                        name = str(item.get("Name", f"{tid}_topic"))
                        count = int(item.get("Count", 0))
                        keywords = self.get_topic_keywords(tid, top_n=5)
                        records.append({
                            "topic_id": tid,
                            "name": name,
                            "count": count,
                            "keywords": keywords,
                        })
            return records
        except Exception as exc:
            logger.error("Failed to extract topic info: %s", exc)
            return []

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> list[dict[str, Any]]:
        """Extracts top representative keywords and their c-TF-IDF weights for a topic.

        Args:
            topic_id: Topic identifier (-1 for outliers).
            top_n: Number of top keywords to return.

        Returns:
            List of dicts each containing 'word' and 'weight'.
        """
        if not self._fitted or self._model is None:
            return []

        try:
            words = self._model.get_topic(topic_id)
            if not words or words is False:
                return []

            return [
                {"word": str(w[0]), "weight": round(float(w[1]), 4)}
                for w in words[:top_n]
            ]
        except Exception as exc:
            logger.debug("Failed to extract keywords for topic %d: %s", topic_id, exc)
            return []

    def hierarchical_reduction(self, nr_topics: int, documents: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Reduces the number of topics hierarchically to nr_topics.

        Args:
            nr_topics: Target number of topics after agglomerative reduction.
            documents: Optional list of documents (falls back to last fitted documents).

        Returns:
            Updated list of topic info dictionaries after reduction.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before hierarchical reduction.")

        docs = documents if documents is not None else self._last_documents

        try:
            if hasattr(self._model, "reduce_topics"):
                self._model.reduce_topics(docs, nr_topics=nr_topics)
            elif hasattr(self._model, "hierarchical_reduction"):
                self._model.hierarchical_reduction(docs, nr_topics=nr_topics)
            logger.info("Hierarchical topic reduction to %d topics completed.", nr_topics)
        except Exception as exc:
            logger.warning("Hierarchical topic reduction failed on model: %s", exc)

        return self.get_topic_info()

    def get_dynamic_topic_keywords(
        self,
        topic_id: int,
        timestamps: Optional[list[Any]] = None,
        documents: Optional[list[str]] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Tracks dynamic c-TF-IDF lexical shifts in topic keywords over time.

        Args:
            topic_id: Topic identifier.
            timestamps: Optional list of timestamps matching documents.
            documents: Optional document texts.

        Returns:
            Dictionary mapping time bucket string to keyword weight dictionaries.
        """
        if not self._fitted or self._model is None:
            return {}

        # If model has dynamic topics_over_time support
        try:
            if hasattr(self._model, "get_dynamic_topic_keywords"):
                res = self._model.get_dynamic_topic_keywords(topic_id, timestamps=timestamps)
                if isinstance(res, dict):
                    return res
        except Exception as exc:
            logger.debug("Model-specific dynamic keywords failed: %s", exc)

        # Baseline representation shift fallback
        base_keywords = self.get_topic_keywords(topic_id, top_n=10)
        if not base_keywords:
            return {}

        return {
            "all_time": base_keywords,
        }

    def calculate_topic_similarity_matrix(self) -> dict[str, Any]:
        """Calculates pairwise cosine similarity matrix between all discovered topics.

        Uses the c-TF-IDF keyword weight vector representations of topics.

        Returns:
            Dictionary containing topic IDs list, 2D cosine similarity matrix, and pairwise lookups.
        """
        topic_info = self.get_topic_info()
        valid_topics = [t for t in topic_info if t["topic_id"] != OUTLIER_TOPIC_ID]

        if not valid_topics:
            return {
                "topic_ids": [],
                "matrix": [],
                "pairwise_similarities": {},
            }

        topic_ids = [t["topic_id"] for t in valid_topics]
        k = len(topic_ids)

        # Build sparse keyword vectors for each topic
        topic_kw_vectors: list[dict[str, float]] = []
        for tid in topic_ids:
            kws = self.get_topic_keywords(tid, top_n=20)
            vec = {kw["word"].lower(): float(kw["weight"]) for kw in kws}
            topic_kw_vectors.append(vec)

        # Compute cosine similarity
        matrix: list[list[float]] = [[0.0] * k for _ in range(k)]
        pairwise: dict[str, float] = {}

        for i in range(k):
            for j in range(i, k):
                if i == j:
                    sim = 1.0
                else:
                    vec_i = topic_kw_vectors[i]
                    vec_j = topic_kw_vectors[j]
                    common_words = set(vec_i.keys()).intersection(vec_j.keys())
                    if not common_words:
                        sim = 0.0
                    else:
                        dot = sum(vec_i[w] * vec_j[w] for w in common_words)
                        norm_i = math.sqrt(sum(v * v for v in vec_i.values()))
                        norm_j = math.sqrt(sum(v * v for v in vec_j.values()))
                        sim = (dot / (norm_i * norm_j)) if (norm_i > 0 and norm_j > 0) else 0.0

                sim_rounded = round(float(sim), 4)
                matrix[i][j] = sim_rounded
                matrix[j][i] = sim_rounded
                pair_key = f"{topic_ids[i]}_vs_{topic_ids[j]}"
                pairwise[pair_key] = sim_rounded

        return {
            "topic_ids": topic_ids,
            "matrix": matrix,
            "pairwise_similarities": pairwise,
        }

    def transform(self, documents: list[str]) -> tuple[list[int], list[float]]:
        """Predicts topics for new documents using the fitted model.

        Args:
            documents: List of cleaned text documents.

        Returns:
            Tuple of (assigned topic IDs, probabilities).
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before calling transform.")

        if not documents:
            return [], []

        try:
            topics, probs = self._model.transform(documents)
            clean_topics = [int(t) for t in topics]
            clean_probs: list[float] = []
            for p in (probs or []):
                val = max(p) if hasattr(p, "__iter__") else float(p)
                clean_probs.append(round(val, 4))
            return clean_topics, clean_probs
        except Exception as exc:
            logger.error("Topic model transform failed: %s", exc)
            return [OUTLIER_TOPIC_ID] * len(documents), [0.0] * len(documents)
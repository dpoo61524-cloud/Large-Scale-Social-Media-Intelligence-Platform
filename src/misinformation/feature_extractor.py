"""Linguistic and Stylometric Feature Extraction for Misinformation Detection.

Combines n-gram TF-IDF textual features with dense stylometric and credibility
signals (punctuation intensity, casing distributions, sensationalism metrics).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Sensationalist and clickbait indicator lexical terms
SENSATIONAL_LEXICON = {
    "shocking",
    "unbelievable",
    "secret",
    "exposed",
    "conspiracy",
    "urgent",
    "hoax",
    "truth",
    "warning",
    "breaking",
    "miracle",
    "hidden",
    "proven",
    "cure",
    "banned",
    "bombshell",
    "censored",
    "revealed",
    "leaked",
    "danger",
    "fraud",
    "coverup",
    "traitor",
    "disaster",
    "unmasked",
}

STYLOMETRIC_FEATURE_NAMES = [
    "uppercase_char_ratio",
    "exclamation_intensity",
    "question_intensity",
    "word_count",
    "avg_word_length",
    "lexical_subjectivity_ratio",
    "digits_ratio",
    "all_caps_word_ratio",
]


class LinguisticFeatureExtractor:
    """Extracts hybrid TF-IDF and stylometric feature representations from text."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 2500,
        sublinear_tf: bool = True,
        use_stylometrics: bool = True,
    ) -> None:
        """Initialize feature extractor with TF-IDF and stylometric parameters."""
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.use_stylometrics = use_stylometrics

        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            strip_accents="unicode",
        )
        self.stylometric_scaler = StandardScaler()
        self._is_fitted = False

    @staticmethod
    def extract_single_stylometrics(text: str) -> np.ndarray:
        """Extract dense stylometric feature vector for a single text string.

        Features:
        1. uppercase_char_ratio: uppercase chars / total chars
        2. exclamation_intensity: count('!') / total words
        3. question_intensity: count('?') / total words
        4. word_count: total word tokens
        5. avg_word_length: average character length per word
        6. lexical_subjectivity_ratio: sensationalism keywords / total words
        7. digits_ratio: numeric chars / total chars
        8. all_caps_word_ratio: words with len > 1 in ALL CAPS / total words
        """
        raw_text = text if isinstance(text, str) else str(text)
        total_chars = max(1, len(raw_text))
        words = re.findall(r"\b[\w'-]+\b", raw_text)
        word_count = max(1, len(words))

        uppercase_chars = sum(1 for c in raw_text if c.isupper())
        exclamations = raw_text.count("!")
        questions = raw_text.count("?")
        digits = sum(1 for c in raw_text if c.isdigit())

        # Lengths and casing of individual tokens
        word_lengths = [len(w) for w in words] if words else [0]
        avg_word_len = float(sum(word_lengths)) / word_count

        all_caps_words = sum(1 for w in words if len(w) > 1 and w.isupper())

        # Lexical indicators
        sensational_count = sum(
            1 for w in words if w.lower() in SENSATIONAL_LEXICON
        )

        return np.array(
            [
                float(uppercase_chars) / total_chars,
                float(exclamations) / word_count,
                float(questions) / word_count,
                float(len(words)),
                avg_word_len,
                float(sensational_count) / word_count,
                float(digits) / total_chars,
                float(all_caps_words) / word_count,
            ],
            dtype=np.float64,
        )

    def extract_stylometric_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """Compute 2D stylometric feature matrix of shape (n_samples, n_stylometrics)."""
        if not texts:
            return np.empty((0, len(STYLOMETRIC_FEATURE_NAMES)), dtype=np.float64)
        rows = [self.extract_single_stylometrics(t) for t in texts]
        return np.vstack(rows)

    def fit(self, texts: Sequence[str], y: Optional[Any] = None) -> LinguisticFeatureExtractor:
        """Fit TF-IDF vocabulary and stylometric scaler on training texts."""
        if not texts:
            raise ValueError("Cannot fit LinguisticFeatureExtractor on empty text sequence.")

        clean_texts = [str(t) if t else "" for t in texts]
        self.tfidf_vectorizer.fit(clean_texts)

        if self.use_stylometrics:
            dense_stylometrics = self.extract_stylometric_matrix(clean_texts)
            self.stylometric_scaler.fit(dense_stylometrics)

        self._is_fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> sparse.csr_matrix:
        """Transform texts into combined sparse feature matrix (TF-IDF + stylometrics)."""
        if not self._is_fitted:
            raise RuntimeError("LinguisticFeatureExtractor must be fitted before calling transform.")

        clean_texts = [str(t) if t else "" for t in texts]
        tfidf_features = self.tfidf_vectorizer.transform(clean_texts)

        if not self.use_stylometrics:
            return tfidf_features.tocsr()

        dense_stylometrics = self.extract_stylometric_matrix(clean_texts)
        scaled_stylometrics = self.stylometric_scaler.transform(dense_stylometrics)
        sparse_stylometrics = sparse.csr_matrix(scaled_stylometrics)

        combined = sparse.hstack([tfidf_features, sparse_stylometrics], format="csr")
        return combined

    def fit_transform(
        self, texts: Sequence[str], y: Optional[Any] = None
    ) -> sparse.csr_matrix:
        """Fit extractor and transform texts in a single call."""
        return self.fit(texts, y).transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names (TF-IDF tokens + stylometric names)."""
        if not self._is_fitted:
            raise RuntimeError("LinguisticFeatureExtractor must be fitted to get feature names.")

        tfidf_names = list(self.tfidf_vectorizer.get_feature_names_out())
        if self.use_stylometrics:
            return tfidf_names + [f"style__{name}" for name in STYLOMETRIC_FEATURE_NAMES]
        return tfidf_names

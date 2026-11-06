"""Unit tests for linguistic and stylometric feature extraction."""

import numpy as np
import pytest
from scipy import sparse

from src.misinformation.feature_extractor import (
    STYLOMETRIC_FEATURE_NAMES,
    LinguisticFeatureExtractor,
)


def test_stylometric_signals_calculation() -> None:
    extractor = LinguisticFeatureExtractor()

    # Normal English text
    text_normal = "The committee reviewed the fiscal budget today."
    style_normal = extractor.extract_single_stylometrics(text_normal)
    assert len(style_normal) == len(STYLOMETRIC_FEATURE_NAMES)
    assert 0.0 <= style_normal[0] <= 1.0  # uppercase char ratio
    assert style_normal[1] == 0.0  # exclamation count
    assert style_normal[2] == 0.0  # question count
    assert style_normal[3] == 7.0  # word count

    # All-caps text
    text_caps = "BREAKING SHOCKING BOMBSHELL EXPOSED NOW"
    style_caps = extractor.extract_single_stylometrics(text_caps)
    assert style_caps[0] > 0.7  # high uppercase char ratio
    assert style_caps[7] == 1.0  # all caps word ratio = 100%
    assert style_caps[5] > 0.5  # sensational words ratio

    # Heavy punctuation
    text_punct = "Is this real??? Yes it is!!!"
    style_punct = extractor.extract_single_stylometrics(text_punct)
    assert style_punct[1] > 0.0  # exclamation intensity
    assert style_punct[2] > 0.0  # question intensity


def test_stylometrics_sanitization_and_edge_cases() -> None:
    extractor = LinguisticFeatureExtractor()

    # None input must return an 8-element zero vector (must NOT become "None")
    style_none = extractor.extract_single_stylometrics(None)
    assert len(style_none) == len(STYLOMETRIC_FEATURE_NAMES)
    assert np.all(style_none == 0.0)

    # Empty string
    style_empty = extractor.extract_single_stylometrics("")
    assert np.all(style_empty == 0.0)

    # Whitespace-only string
    style_spaces = extractor.extract_single_stylometrics("   \t\n   ")
    assert np.all(style_spaces == 0.0)

    # Numeric-only text
    style_numeric = extractor.extract_single_stylometrics("12345 67890")
    assert not np.any(np.isnan(style_numeric))
    assert style_numeric[6] > 0.8  # digits ratio
    assert style_numeric[3] == 2.0  # 2 numeric word tokens

    # Punctuation-only text
    style_punct = extractor.extract_single_stylometrics("!!! ??? !!!")
    assert not np.any(np.isnan(style_punct))
    assert style_punct[3] == 0.0  # word count = 0
    assert style_punct[1] > 0.0  # exclamation count captured


def test_hybrid_feature_extraction_fit_transform() -> None:
    corpus = [
        "The Federal Reserve announced interest rates today.",
        "SHOCKING miracle cure exposed by anonymous sources!!!",
        "Scientific researchers discovered new astronomical phenomena in deep space.",
        "URGENT WARNING: Tap water contains dangerous government tracking chips???",
    ]

    extractor = LinguisticFeatureExtractor(ngram_range=(1, 2), max_features=100)
    features = extractor.fit_transform(corpus)

    assert sparse.issparse(features)
    assert features.shape[0] == 4

    # Combined features should include TF-IDF features plus stylometric features
    feature_names = extractor.get_feature_names()
    assert len(feature_names) == features.shape[1]
    assert any(name.startswith("style__") for name in feature_names)

    # Transform new text including empty string
    test_text = ["New fiscal policy proposed by senate.", ""]
    test_features = extractor.transform(test_text)
    assert test_features.shape[0] == 2
    assert test_features.shape[1] == features.shape[1]


def test_empty_vocabulary_handling() -> None:
    extractor = LinguisticFeatureExtractor()
    # All punctuation/whitespace texts contain no alphanumeric vocabulary
    with pytest.raises(ValueError, match="no valid vocabulary tokens"):
        extractor.fit(["   ", "!!! ???", "..."])


def test_extractor_without_stylometrics() -> None:
    corpus = ["A simple statement for testing vocabulary.", "Another basic test sentence."]
    extractor = LinguisticFeatureExtractor(use_stylometrics=False)
    features = extractor.fit_transform(corpus)

    assert sparse.issparse(features)
    names = extractor.get_feature_names()
    assert not any(name.startswith("style__") for name in names)


def test_transform_unfitted_raises_runtime_error() -> None:
    extractor = LinguisticFeatureExtractor()
    with pytest.raises(RuntimeError):
        extractor.transform(["Sample sentence."])
    with pytest.raises(RuntimeError):
        extractor.get_feature_names()

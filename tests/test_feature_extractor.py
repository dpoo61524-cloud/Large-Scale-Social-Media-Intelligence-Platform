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

    # Normal text
    text_normal = "The committee reviewed the fiscal budget today."
    style_normal = extractor.extract_single_stylometrics(text_normal)
    assert len(style_normal) == len(STYLOMETRIC_FEATURE_NAMES)
    # Check bounds
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


def test_stylometrics_edge_cases() -> None:
    extractor = LinguisticFeatureExtractor()

    # Empty string
    style_empty = extractor.extract_single_stylometrics("")
    assert len(style_empty) == len(STYLOMETRIC_FEATURE_NAMES)
    assert not np.any(np.isnan(style_empty))

    # Spaces and non-alphanumeric
    style_spaces = extractor.extract_single_stylometrics("   ... ???   ")
    assert not np.any(np.isnan(style_spaces))


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

    # Transform new text
    test_text = ["New fiscal policy proposed by senate."]
    test_features = extractor.transform(test_text)
    assert test_features.shape[0] == 1
    assert test_features.shape[1] == features.shape[1]


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

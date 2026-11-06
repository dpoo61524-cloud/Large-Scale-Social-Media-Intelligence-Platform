"""Unit tests for misinformation classifier and credibility scoring."""

from pathlib import Path
import numpy as np
import pytest

from src.misinformation.classifier import MisinformationClassifier
from src.misinformation.dataset_loader import (
    FINE_GRAINED_LABEL_TO_ID,
    LiarDatasetLoader,
)
from src.misinformation.feature_extractor import LinguisticFeatureExtractor


@pytest.fixture
def binary_sample_dataset():
    loader = LiarDatasetLoader()
    texts, labels, _ = loader.prepare_data(binary=True, uncertain_policy="exclude")
    extractor = LinguisticFeatureExtractor(ngram_range=(1, 2), max_features=500)
    X = extractor.fit_transform(texts)
    return X, labels, extractor


@pytest.fixture
def six_class_sample_dataset():
    loader = LiarDatasetLoader()
    texts, labels, _ = loader.prepare_data(binary=False)
    extractor = LinguisticFeatureExtractor(ngram_range=(1, 2), max_features=500)
    X = extractor.fit_transform(texts)
    return X, labels, extractor, loader.get_fine_grained_labels()


def test_classifier_fit_predict_and_proba_binary(binary_sample_dataset) -> None:
    X, y, _ = binary_sample_dataset
    clf = MisinformationClassifier(class_weight="balanced", calibrate=False)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert len(preds) == X.shape[0]
    assert set(preds).issubset({0, 1})

    probas = clf.predict_proba(X)
    assert probas.shape == (X.shape[0], 2)
    assert np.allclose(probas.sum(axis=1), 1.0)

    credibility = clf.get_credibility_score(X)
    assert len(credibility) == X.shape[0]
    assert np.all((credibility >= 0.0) & (credibility <= 1.0))


def test_classifier_six_class_training_and_metrics(six_class_sample_dataset) -> None:
    """Verifies complete 6-class fine-grained veracity classification."""
    X, y, _, target_names = six_class_sample_dataset
    assert len(np.unique(y)) == 6

    clf = MisinformationClassifier(class_weight="balanced", calibrate=False, random_state=42)
    clf.fit(X, y)

    # Class ordering and properties
    assert len(clf.classes_) == 6
    assert list(clf.classes_) == [0, 1, 2, 3, 4, 5]

    # Prediction and probabilities
    preds = clf.predict(X)
    assert len(preds) == X.shape[0]
    assert set(preds).issubset({0, 1, 2, 3, 4, 5})

    probas = clf.predict_proba(X)
    assert probas.shape == (X.shape[0], 6)
    assert np.allclose(probas.sum(axis=1), 1.0)

    # Multi-class ordinal credibility expectation in [0.0, 1.0]
    credibility = clf.get_credibility_score(X)
    assert len(credibility) == X.shape[0]
    assert np.all((credibility >= 0.0) & (credibility <= 1.0))

    # Comprehensive evaluation
    eval_res = clf.evaluate(X, y, target_names=target_names)
    assert "accuracy" in eval_res
    assert "f1_macro" in eval_res
    assert "confusion_matrix" in eval_res
    assert len(eval_res["confusion_matrix"]) == 6
    assert eval_res["is_calibrated"] is False


def test_calibration_fallback_on_insufficient_samples() -> None:
    """Verifies that small sample counts fall back to base LogisticRegression."""
    # Only 4 samples per class (below min_calibration_samples=5)
    X = np.eye(8)
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    clf = MisinformationClassifier(calibrate=True, calibration_cv=3, min_calibration_samples=5)
    clf.fit(X, y)

    assert clf.is_calibrated is False
    assert clf.model is clf.base_model


def test_calibration_succeeds_with_sufficient_samples() -> None:
    """Verifies that calibration executes when sample support is sufficient."""
    # 6 samples per class (above min_calibration_samples=5)
    X = np.eye(12)
    y = [0]*6 + [1]*6

    clf = MisinformationClassifier(calibrate=True, calibration_cv=3, min_calibration_samples=5)
    clf.fit(X, y)

    assert clf.is_calibrated is True


def test_classifier_out_of_sample_evaluation(binary_sample_dataset) -> None:
    """Tests evaluation on held-out test split to ensure no data leakage."""
    X, y, _ = binary_sample_dataset
    split_idx = int(X.shape[0] * 0.7)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = MisinformationClassifier(class_weight="balanced", calibrate=False)
    clf.fit(X_train, y_train)

    metrics = clf.evaluate(X_test, y_test, target_names=["fake", "real"])
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_model_serialization(tmp_path: Path, binary_sample_dataset) -> None:
    X, y, _ = binary_sample_dataset
    clf = MisinformationClassifier(calibrate=False)
    clf.fit(X, y)

    save_path = tmp_path / "model.joblib"
    clf.save_model(save_path)
    assert save_path.exists()

    loaded_clf = MisinformationClassifier.load_model(save_path)
    orig_preds = clf.predict(X)
    loaded_preds = loaded_clf.predict(X)
    assert np.array_equal(orig_preds, loaded_preds)
    assert loaded_clf.is_calibrated == clf.is_calibrated


def test_unfitted_classifier_raises_error() -> None:
    clf = MisinformationClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.zeros((2, 5)))
    with pytest.raises(RuntimeError):
        clf.predict_proba(np.zeros((2, 5)))
    with pytest.raises(RuntimeError):
        clf.save_model("dummy.joblib")


def test_single_class_raises_error() -> None:
    clf = MisinformationClassifier()
    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        clf.fit(np.zeros((4, 3)), [0, 0, 0, 0])

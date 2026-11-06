"""Unit tests for misinformation classifier and credibility scoring."""

from pathlib import Path
import numpy as np
import pytest

from src.misinformation.classifier import MisinformationClassifier
from src.misinformation.dataset_loader import LiarDatasetLoader
from src.misinformation.feature_extractor import LinguisticFeatureExtractor


@pytest.fixture
def sample_dataset():
    loader = LiarDatasetLoader()
    texts, labels, _ = loader.prepare_data(binary=True, uncertain_policy="exclude")
    extractor = LinguisticFeatureExtractor(ngram_range=(1, 2), max_features=500)
    X = extractor.fit_transform(texts)
    return X, labels, extractor


def test_classifier_fit_predict_and_proba(sample_dataset) -> None:
    X, y, _ = sample_dataset
    clf = MisinformationClassifier(class_weight="balanced", calibrate=True, calibration_cv=2)
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


def test_classifier_evaluation_metrics(sample_dataset) -> None:
    X, y, _ = sample_dataset
    clf = MisinformationClassifier(calibrate=False)
    clf.fit(X, y)

    metrics = clf.evaluate(X, y, target_names=["fake", "real"])
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert "classification_report" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert isinstance(metrics["confusion_matrix"], list)
    if metrics["roc_auc"] is not None:
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_model_serialization(tmp_path: Path, sample_dataset) -> None:
    X, y, _ = sample_dataset
    clf = MisinformationClassifier(calibrate=False)
    clf.fit(X, y)

    save_path = tmp_path / "model.joblib"
    clf.save_model(save_path)
    assert save_path.exists()

    loaded_clf = MisinformationClassifier.load_model(save_path)
    orig_preds = clf.predict(X)
    loaded_preds = loaded_clf.predict(X)
    assert np.array_equal(orig_preds, loaded_preds)

    orig_scores = clf.get_credibility_score(X)
    loaded_scores = loaded_clf.get_credibility_score(X)
    assert np.allclose(orig_scores, loaded_scores)


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

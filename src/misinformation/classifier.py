"""Misinformation and Fake-News Classifier.

Provides balanced, calibrated classification with probability estimation,
comprehensive performance evaluation (Precision, Recall, Macro F1, ROC-AUC),
and persistence via Joblib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MisinformationClassifier:
    """Calibrated veracity classifier for fake news and misinformation detection."""

    def __init__(
        self,
        class_weight: str = "balanced",
        calibrate: bool = True,
        calibration_cv: int = 3,
        c_penalty: float = 1.0,
        random_state: int = 42,
    ) -> None:
        """Initialize classifier with regularized logistic loss and calibration settings."""
        self.class_weight = class_weight
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.c_penalty = c_penalty
        self.random_state = random_state

        self.base_model = LogisticRegression(
            C=self.c_penalty,
            class_weight=self.class_weight,
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model: Any = None
        self.classes_: np.ndarray = np.array([])
        self._is_fitted: bool = False

    def fit(self, X: Any, y: Sequence[int] | np.ndarray) -> MisinformationClassifier:
        """Fit calibrated classifier on feature matrix X and target labels y.

        Parameters
        ----------
        X : sparse matrix or dense array of shape (n_samples, n_features)
            Extracted text and stylometric features.
        y : array-like of shape (n_samples,)
            Integer ground truth labels (binary 0/1 or multi-class 0..5).
        """
        y_arr = np.asarray(y)
        if len(y_arr) == 0:
            raise ValueError("Cannot train MisinformationClassifier on empty dataset.")

        unique_classes, counts = np.unique(y_arr, return_counts=True)
        self.classes_ = unique_classes

        if len(unique_classes) < 2:
            raise ValueError(
                f"Training requires at least 2 distinct classes, found {len(unique_classes)}."
            )

        min_class_count = int(np.min(counts))

        # Check if cross-validated calibration is feasible
        if self.calibrate and min_class_count >= self.calibration_cv and self.calibration_cv >= 2:
            try:
                calibrated = CalibratedClassifierCV(
                    estimator=self.base_model,
                    method="sigmoid",
                    cv=self.calibration_cv,
                )
                calibrated.fit(X, y_arr)
                self.model = calibrated
            except Exception as e:
                logger.warning(
                    f"CalibratedClassifierCV failed ({e}); falling back to standard balanced LogisticRegression."
                )
                self.base_model.fit(X, y_arr)
                self.model = self.base_model
        else:
            self.base_model.fit(X, y_arr)
            self.model = self.base_model

        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for samples in X."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("MisinformationClassifier must be fitted before predict.")
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict calibrated class probabilities for samples in X.

        Returns array of shape (n_samples, n_classes).
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("MisinformationClassifier must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def get_credibility_score(self, X: Any) -> np.ndarray:
        """Extract continuous credibility score in [0.0, 1.0] representing truth probability.

        For binary classification (0=fake, 1=real):
            credibility = P(real) = proba[:, 1]
        For multi-class (0=pants-fire to 5=true):
            credibility = normalized expected veracity value in [0.0, 1.0].
        """
        probas = self.predict_proba(X)
        if len(self.classes_) == 2 and 1 in self.classes_:
            real_idx = int(np.where(self.classes_ == 1)[0][0])
            return probas[:, real_idx]

        # Multi-class: weighted expectation divided by maximum class id
        max_class_id = float(np.max(self.classes_)) if len(self.classes_) > 1 else 1.0
        expected_val = np.sum(probas * self.classes_, axis=1) / max(1.0, max_class_id)
        return np.clip(expected_val, 0.0, 1.0)

    def evaluate(
        self,
        X_test: Any,
        y_test: Sequence[int] | np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation and return metrics dictionary.

        Returns
        -------
        Dict[str, Any]
            Precision, Recall, Macro F1, Weighted F1, Confusion Matrix, and ROC-AUC.
        """
        y_true = np.asarray(y_test)
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=self.classes_).tolist()
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=self.classes_,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        roc_auc: Optional[float] = None
        unique_test_classes = np.unique(y_true)
        if len(unique_test_classes) == 2 and len(self.classes_) == 2:
            try:
                # Binary ROC-AUC
                real_idx = int(np.where(self.classes_ == 1)[0][0])
                roc_auc = float(roc_auc_score(y_true, y_prob[:, real_idx]))
            except Exception as e:
                logger.warning(f"Could not compute binary ROC-AUC: {e}")
        elif len(self.classes_) > 2 and len(unique_test_classes) > 1:
            try:
                roc_auc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        labels=self.classes_,
                    )
                )
            except Exception as e:
                logger.warning(f"Could not compute multi-class ROC-AUC: {e}")

        return {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classes": self.classes_.tolist(),
            "roc_auc": roc_auc,
            "classification_report": report_dict,
        }

    def save_model(self, file_path: str | Path) -> None:
        """Serialize trained model to disk via joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted MisinformationClassifier.")
        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "base_model": self.base_model,
            "classes_": self.classes_,
            "class_weight": self.class_weight,
            "calibrate": self.calibrate,
            "random_state": self.random_state,
        }
        joblib.dump(payload, target_path)

    @classmethod
    def load_model(cls, file_path: str | Path) -> MisinformationClassifier:
        """Deserialize model from disk via joblib."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        classifier = cls(
            class_weight=payload.get("class_weight", "balanced"),
            calibrate=payload.get("calibrate", True),
            random_state=payload.get("random_state", 42),
        )
        classifier.model = payload["model"]
        classifier.base_model = payload.get("base_model", classifier.base_model)
        classifier.classes_ = np.asarray(payload["classes_"])
        classifier._is_fitted = True
        return classifier

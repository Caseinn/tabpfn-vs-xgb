import time
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.metrics import classification_report

from src.utils.config import RANDOM_STATE, TABPFN_N_ESTIMATORS


def build_vanilla() -> TabPFNClassifier:
    """Build TabPFN v2.5 with pure defaults - no class balancing.

    Returns:
        Configured TabPFNClassifier.
    """
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5,
        n_estimators=TABPFN_N_ESTIMATORS,
        device="cuda",
        random_state=RANDOM_STATE,
        fit_mode="fit_preprocessors",
        memory_saving_mode="balanced",
        n_preprocessing_jobs=-1,
    )


def build_balanced() -> TabPFNClassifier:
    """Build TabPFN v2.5 with balanced probabilities.

    Returns:
        Configured TabPFNClassifier.
    """
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5,
        n_estimators=TABPFN_N_ESTIMATORS,
        device="cuda",
        random_state=RANDOM_STATE,
        fit_mode="fit_preprocessors",
        balance_probabilities=True,
        memory_saving_mode="balanced",
        n_preprocessing_jobs=-1,
    )


def train_and_predict(model, X_train, y_train, X_test) -> tuple[np.ndarray, float]:
    """Train and predict, returning predictions and elapsed time.

    Args:
        model: TabPFNClassifier instance.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.

    Returns:
        Tuple of (predictions, elapsed_seconds).
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.perf_counter() - start
    return preds, elapsed


CLASS_NAMES = ["No Diabetes", "Diabetes"]


def report(y_true, y_pred) -> str:
    """Generate classification report string.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Formatted classification report.
    """
    return classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

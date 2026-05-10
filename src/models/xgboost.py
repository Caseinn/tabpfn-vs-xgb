import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from src.utils.config import RANDOM_STATE


def build_vanilla() -> XGBClassifier:
    """Build XGBoost with pure defaults - no class balancing.

    Returns:
        Configured XGBClassifier.
    """
    return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda")


def build_balanced(params: dict | None = None) -> XGBClassifier:
    """Build XGBoost classifier with class weight balancing.

    Args:
        params: Optional hyperparameter dict.

    Returns:
        Configured XGBClassifier.
    """
    defaults = dict(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", device="cuda")
    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


def build_tuned(params: dict | None = None) -> XGBClassifier:
    """Build XGBoost classifier with Optuna-tuned hyperparameters.

    Args:
        params: Hyperparameter dict from Optuna.

    Returns:
        Configured XGBClassifier.
    """
    return build_balanced(params)


def tune(trial, X_train, y_train, X_val, y_val, search_space) -> float:
    """Single Optuna trial evaluation.

    Args:
        trial: Optuna trial object.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        search_space: Dict of (low, high, step) ranges.

    Returns:
        Validation F1-score (macro).
    """
    from sklearn.metrics import f1_score

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    low_n, high_n, step_n = search_space["n_estimators"]
    low_d, high_d, step_d = search_space["max_depth"]
    low_lr, high_lr, step_lr = search_space["learning_rate"]
    low_ss, high_ss, step_ss = search_space["subsample"]
    low_cs, high_cs, step_cs = search_space["colsample_bytree"]
    low_mw, high_mw, step_mw = search_space["min_child_weight"]
    low_g, high_g, step_g = search_space["gamma"]

    params = {
        "n_estimators": trial.suggest_int("n_estimators", low_n, high_n, step=step_n),
        "max_depth": trial.suggest_int("max_depth", low_d, high_d, step=step_d),
        "learning_rate": trial.suggest_float("learning_rate", low_lr, high_lr, step=step_lr),
        "subsample": trial.suggest_float("subsample", low_ss, high_ss, step=step_ss),
        "colsample_bytree": trial.suggest_float("colsample_bytree", low_cs, high_cs, step=step_cs),
        "min_child_weight": trial.suggest_int("min_child_weight", low_mw, high_mw, step=step_mw),
        "gamma": trial.suggest_float("gamma", low_g, high_g, step=step_g),
        "scale_pos_weight": n_neg / n_pos,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": -1,
    }
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return f1_score(y_val, clf.predict(X_val), average="macro")


def train_and_predict(model, X_train, y_train, X_test) -> tuple[np.ndarray, float]:
    """Train and predict, returning predictions and elapsed time.

    Args:
        model: XGBClassifier instance.
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

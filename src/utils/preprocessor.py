import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import RANDOM_STATE, OPTUNA_VAL_SPLIT


def split_xy(
    df: pd.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Separate features and target from DataFrame.

    Args:
        df: DataFrame with features and target column.
        target: Name of the target column.

    Returns:
        Tuple of (X array, y array, feature names list).
    """
    X = df.drop(columns=[target]).values.astype(np.float32)
    y = df[target].values.astype(int)
    names = df.drop(columns=[target]).columns.tolist()
    return X, y, names


def tuning_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float | None = None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/validation split for tuning.

    Args:
        X: Feature matrix.
        y: Target array.
        test_size: Validation split ratio. Defaults to OPTUNA_VAL_SPLIT.
        seed: Random seed.

    Returns:
        Tuple of (X_train, X_val, y_train, y_val).
    """
    test_size = test_size or OPTUNA_VAL_SPLIT
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

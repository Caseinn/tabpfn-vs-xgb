from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
OUTPUT_DIR = BASE_DIR / "output"

TARGET_COLUMN = "Diabetes_binary"
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42
K_FOLDS = 5

OPTUNA_TRIALS = 100
OPTUNA_VAL_SPLIT = 0.2

XGB_SEARCH_SPACE = {
    "n_estimators": (100, 500, 10),
    "max_depth": (3, 12, 1),
    "learning_rate": (0.01, 0.3, 0.005),
    "subsample": (0.6, 1.0, 0.025),
    "colsample_bytree": (0.6, 1.0, 0.025),
    "min_child_weight": (1, 10, 1),
    "gamma": (0, 5, 0.1),
}

TABPFN_N_ESTIMATORS = 8

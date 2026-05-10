import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

CLASS_NAMES = ["No Diabetes", "Diabetes"]

from src.utils.config import OUTPUT_DIR


def _ensure_dir() -> None:
    """Ensure output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filepath: Path,
) -> None:
    """Plot and save confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Plot title.
        filepath: Output file path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filepath: Path,
) -> None:
    """Plot and save classification report as heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Plot title.
        filepath: Output file path.
    """
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    df = np.array([
        [report["No Diabetes"]["precision"], report["No Diabetes"]["recall"], report["No Diabetes"]["f1-score"], report["No Diabetes"]["support"]],
        [report["Diabetes"]["precision"], report["Diabetes"]["recall"], report["Diabetes"]["f1-score"], report["Diabetes"]["support"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        df, annot=True, fmt=".3f", cmap="Greens", ax=ax,
        xticklabels=["Precision", "Recall", "F1-Score", "Support"],
        yticklabels=CLASS_NAMES,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_fold_plots(store: dict, n_splits: int) -> None:
    """Save per-fold confusion matrix and report plots.

    Args:
        store: Results dictionary from run_cv.
        n_splits: Number of folds.
    """
    _ensure_dir()
    print(f"\n{'=' * 60}")
    print("SAVING PLOTS TO output/")
    print(f"{'=' * 60}")
    for i in range(n_splits):
        for label, key in [("xgb", "xgb"), ("tabpfn", "tabpfn")]:
            prefix = f"fold{i + 1}_{label}"
            plot_confusion_matrix(
                store[key]["trues"][i], store[key]["preds"][i],
                f"{label.upper()} - Fold {i + 1} Confusion Matrix",
                OUTPUT_DIR / f"{prefix}_cm.png",
            )
            plot_classification_report(
                store[key]["trues"][i], store[key]["preds"][i],
                f"{label.upper()} - Fold {i + 1} Classification Report",
                OUTPUT_DIR / f"{prefix}_report.png",
            )


def save_aggregate_plots(store: dict, n_splits: int) -> None:
    """Save aggregate plots across all folds.

    Args:
        store: Results dictionary from run_cv.
        n_splits: Number of folds.
    """
    _ensure_dir()
    all_trues_xgb = np.concatenate(store["xgb"]["trues"])
    all_preds_xgb = np.concatenate(store["xgb"]["preds"])
    all_trues_tabpfn = np.concatenate(store["tabpfn"]["trues"])
    all_preds_tabpfn = np.concatenate(store["tabpfn"]["preds"])

    for label, trues, preds in [
        ("xgb", all_trues_xgb, all_preds_xgb),
        ("tabpfn", all_trues_tabpfn, all_preds_tabpfn),
    ]:
        plot_confusion_matrix(
            trues, preds,
            f"{label.upper()} - All Folds Confusion Matrix",
            OUTPUT_DIR / f"{label}_aggregate_cm.png",
        )
        plot_classification_report(
            trues, preds,
            f"{label.upper()} - All Folds Classification Report",
            OUTPUT_DIR / f"{label}_aggregate_report.png",
        )

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score, accuracy_score

from src.utils.config import SAMPLE_SIZE, TARGET_COLUMN
from src.utils.data_loader import load_dataset, stratified_sample_df
from src.utils.preprocessor import split_xy, tuning_split
from src.utils.timer import timed
from src.models.xgboost import build_vanilla as build_xgb_vanilla, build_balanced as build_xgb_balanced, build_tuned as build_xgb_tuned
from src.models.tabpfn import build_vanilla as build_tabpfn_vanilla, build_balanced as build_tabpfn_balanced, build_tuned as build_tabpfn_tuned
from src.evaluation.tuner import run as run_tuning
from src.evaluation.kfold import run_cv, print_reports, print_summary
from src.visualization.plotter import save_fold_plots, save_aggregate_plots


def _run_single(X, y, xgb, tabpfn, label, n_folds):
    console = Console()
    console.print(f"\n{'=' * 60}")
    console.print(f"[bold yellow]{label}[/bold yellow]")
    console.print(f"{'=' * 60}")
    store = run_cv(X, y, xgb, tabpfn, n_splits=n_folds)
    return store


def run(mode: str = "vanilla", sample_size: int | None = None, n_trials: int | None = None, n_folds: int | None = None) -> None:
    """Execute comparison pipeline.

    Args:
        mode: vanilla, balanced, tuned, or all.
        sample_size: Number of rows to sample.
        n_trials: Optuna tuning trials.
        n_folds: Number of CV folds.
    """
    console = Console()
    n_folds = n_folds or 5

    df = stratified_sample_df(load_dataset(), n=sample_size or SAMPLE_SIZE)
    X, y, names = split_xy(df, target=TARGET_COLUMN)
    console.print(f"\n[bold]Features:[/bold] [cyan]{len(names)}[/cyan] | [bold]Samples:[/bold] [cyan]{len(y):,}[/cyan]")

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()

    if mode == "all":
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold magenta]XGBOOST vs TabPFN v2.5 - ALL MODES[/bold magenta]")
        console.print(f"{'=' * 60}")

        all_stores = {}

        xgb = build_xgb_vanilla()
        tabpfn = build_tabpfn_vanilla()
        all_stores["vanilla"] = _run_single(X, y, xgb, tabpfn, "MODE 1: VANILLA", n_folds)

        xgb = build_xgb_balanced({"scale_pos_weight": n_neg / n_pos})
        tabpfn = build_tabpfn_balanced()
        all_stores["balanced"] = _run_single(X, y, xgb, tabpfn, "MODE 2: BALANCED", n_folds)

        X_train, X_val, y_train, y_val = tuning_split(X, y)
        with timed("XGBoost Optuna Tuning"):
            best = run_tuning(X_train, y_train, X_val, y_val, n_trials=n_trials)
        best["scale_pos_weight"] = n_neg / n_pos
        xgb = build_xgb_tuned(best)
        tabpfn = build_tabpfn_tuned()
        all_stores["tuned"] = _run_single(X, y, xgb, tabpfn, "MODE 3: TUNED", n_folds)

        for mode_label, store in all_stores.items():
            print_reports(store, n_splits=n_folds)
            print_summary(store, n_splits=n_folds)

        console.print(f"\n{'=' * 60}")
        console.print(f"[bold magenta]FINAL SUMMARY[/bold magenta]")
        console.print(f"{'=' * 60}")

        table = Table(show_header=True, header_style="bold white")
        table.add_column("Mode")
        table.add_column("Model")
        table.add_column("F1 (No Diabetes)", justify="right")
        table.add_column("F1 (Diabetes)", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Total Time", justify="right")

        for mode_key, mode_name in [("vanilla", "1. Vanilla"), ("balanced", "2. Balanced"), ("tuned", "3. Tuned")]:
            store = all_stores[mode_key]
            for label, key, color in [("XGBoost", "xgb", "green"), ("TabPFN v2.5", "tabpfn", "cyan")]:
                all_trues = np.concatenate(store[key]["trues"])
                all_preds = np.concatenate(store[key]["preds"])
                total_time = sum(store[key]["times"])
                f1_minor = f1_score(all_trues, all_preds, labels=[1], average="macro", zero_division=0)
                f1_major = f1_score(all_trues, all_preds, labels=[0], average="macro", zero_division=0)
                acc = accuracy_score(all_trues, all_preds)
                table.add_row(mode_name, f"[{color}]{label}[/{color}]", f"{f1_major:.4f}", f"{f1_minor:.4f}", f"{acc:.4f}", f"{total_time:.2f}s")

        console.print(table)
        save_fold_plots(all_stores["tuned"], n_folds)
        save_aggregate_plots(all_stores["tuned"], n_folds)

    else:
        mode_labels = {"vanilla": "VANILLA", "balanced": "BALANCED", "tuned": "TUNED"}
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold magenta]XGBOOST vs TabPFN v2.5 - {mode_labels[mode]}[/bold magenta]")
        console.print(f"{'=' * 60}")

        if mode == "vanilla":
            xgb = build_xgb_vanilla()
            tabpfn = build_tabpfn_vanilla()
        elif mode == "balanced":
            xgb = build_xgb_balanced({"scale_pos_weight": n_neg / n_pos})
            tabpfn = build_tabpfn_balanced()
        else:
            X_train, X_val, y_train, y_val = tuning_split(X, y)
            with timed("XGBoost Optuna Tuning"):
                best = run_tuning(X_train, y_train, X_val, y_val, n_trials=n_trials)
            best["scale_pos_weight"] = n_neg / n_pos
            xgb = build_xgb_tuned(best)
            tabpfn = build_tabpfn_tuned()

        store = run_cv(X, y, xgb, tabpfn, n_splits=n_folds)
        print_reports(store, n_splits=n_folds)
        print_summary(store, n_splits=n_folds)
        save_fold_plots(store, n_folds)
        save_aggregate_plots(store, n_folds)

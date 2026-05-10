import sys
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score, accuracy_score

from src.utils.config import SAMPLE_SIZE, TARGET_COLUMN, get_mode_output_dir
from src.utils.data_loader import load_dataset, stratified_sample_df
from src.utils.preprocessor import split_xy, tuning_split
from src.utils.timer import timed
from src.utils.logger import OutputLogger
from src.models.xgboost import build_vanilla as build_xgb_vanilla, build_balanced as build_xgb_balanced, build_tuned as build_xgb_tuned
from src.models.tabpfn import build_vanilla as build_tabpfn_vanilla, build_balanced as build_tabpfn_balanced
from src.evaluation.tuner import run as run_tuning
from src.evaluation.kfold import run_cv, print_reports, print_summary
from src.visualization.plotter import save_fold_plots, save_aggregate_plots


def _run_single(X, y, xgb, tabpfn, label, n_folds, console):
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
    n_folds = n_folds or 5

    modes = ["vanilla", "balanced", "tuned"] if mode == "all" else [mode]

    for current_mode in modes:
        output_dir = get_mode_output_dir(current_mode)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_path = output_dir / "output.log"
        logger = OutputLogger(log_path)
        console = logger.get_console()

        sys.stdout = logger
        sys.stderr = logger

        try:
            _execute_mode(current_mode, sample_size, n_trials, n_folds, console, output_dir)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            logger.close()

        print(f"\n[bold green]Output saved to: {output_dir}/[/bold green]")


def _execute_mode(mode: str, sample_size: int | None, n_trials: int | None, n_folds: int, console: Console, output_dir: Path) -> None:
    """Execute a single mode of the pipeline."""
    df = stratified_sample_df(load_dataset(), n=sample_size or SAMPLE_SIZE)
    X, y, names = split_xy(df, target=TARGET_COLUMN)
    console.print(f"\n[bold]Features:[/bold] [cyan]{len(names)}[/cyan] | [bold]Samples:[/bold] [cyan]{len(y):,}[/cyan]")

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()

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
        tabpfn = build_tabpfn_balanced()

    store = run_cv(X, y, xgb, tabpfn, n_splits=n_folds)
    print_reports(store, n_splits=n_folds)
    print_summary(store, n_splits=n_folds)
    save_fold_plots(store, n_folds, output_dir)
    save_aggregate_plots(store, n_folds, output_dir)

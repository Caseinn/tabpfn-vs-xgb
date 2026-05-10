import numpy as np
import optuna
from optuna.samplers import TPESampler
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.utils.config import RANDOM_STATE, OPTUNA_TRIALS, XGB_SEARCH_SPACE
from src.models.xgboost import tune as xgb_tune


def run(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int | None = None) -> dict:
    """Run Optuna tuning and return best params.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of trials.

    Returns:
        Best hyperparameters dict.
    """
    n_trials = n_trials or OPTUNA_TRIALS
    console = Console()

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold magenta]OPTUNA HYPERPARAMETER TUNING[/bold magenta]")
    console.print(f"[dim]{n_trials} trials[/dim]")
    console.print(f"{'=' * 60}")

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Tuning...", total=n_trials)

        def callback(study, trial):
            progress.advance(task)
            progress.update(task, description=f"[cyan]Trial {trial.number + 1}/{n_trials} (best: {study.best_value:.4f})")

        study.optimize(
            lambda t: xgb_tune(t, X_train, y_train, X_val, y_val, XGB_SEARCH_SPACE),
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=False,
        )

        console.print(f"\n[bold green]Best F1-Score (macro):[/bold green] {study.best_value:.4f}")
    console.print(f"[bold green]Best Params:[/bold green]")
    for k, v in study.best_params.items():
        console.print(f"  [cyan]{k}[/cyan]: [yellow]{v}[/yellow]")
    return study.best_params

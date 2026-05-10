import numpy as np
from sklearn.model_selection import StratifiedKFold
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.utils.config import K_FOLDS, RANDOM_STATE
from src.models.xgboost import train_and_predict as xgb_run, report as xgb_report
from src.models.tabpfn import train_and_predict as tabpfn_run, report as tabpfn_report


def run_cv(X: np.ndarray, y: np.ndarray, model_xgb, model_tabpfn, n_splits: int | None = None) -> dict:
    """Run stratified k-fold CV with per-model timing.

    Args:
        X: Feature matrix.
        y: Target array.
        model_xgb: XGBClassifier instance.
        model_tabpfn: TabPFNClassifier instance.
        n_splits: Number of folds.

    Returns:
        Results dict with reports, preds, trues, and times per model.
    """
    n_splits = n_splits or K_FOLDS
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    console = Console()

    store = {
        "xgb": {"reports": [], "preds": [], "trues": [], "times": []},
        "tabpfn": {"reports": [], "preds": [], "trues": [], "times": []},
    }

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold magenta]{n_splits}-FOLD CROSS-VALIDATION[/bold magenta]")
    console.print(f"{'=' * 60}")

    print(f"\nWarming up GPU (XGBoost + TabPFN)...", end=" ", flush=True)
    import copy
    xgb_warm = copy.deepcopy(model_xgb)
    tabpfn_warm = copy.deepcopy(model_tabpfn)
    xgb_warm.fit(X[:100], y[:100])
    tabpfn_warm.fit(X[:100], y[:100])
    console.print("[green]done[/green]")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        console.print(f"\n[bold white on blue] Fold {fold}/{n_splits} [/bold white on blue] [dim]Train: {len(train_idx):,} | Test: {len(test_idx):,}[/dim]")

        print(f"  XGBoost...", end=" ", flush=True)
        preds_xgb, time_xgb = xgb_run(model_xgb, X_train, y_train, X_test)
        store["xgb"]["times"].append(time_xgb)
        store["xgb"]["reports"].append(xgb_report(y_test, preds_xgb))
        store["xgb"]["preds"].append(preds_xgb)
        store["xgb"]["trues"].append(y_test)
        console.print(f"[green]{time_xgb:.2f}s[/green]")

        print(f"  TabPFN v2.5...", end=" ", flush=True)
        preds_tabpfn, time_tabpfn = tabpfn_run(model_tabpfn, X_train, y_train, X_test)
        store["tabpfn"]["times"].append(time_tabpfn)
        store["tabpfn"]["reports"].append(tabpfn_report(y_test, preds_tabpfn))
        store["tabpfn"]["preds"].append(preds_tabpfn)
        store["tabpfn"]["trues"].append(y_test)
        console.print(f"[cyan]{time_tabpfn:.2f}s[/cyan]")

    return store


def print_reports(store: dict, n_splits: int | None = None) -> None:
    """Print classification reports per fold.

    Args:
        store: Results from run_cv.
        n_splits: Number of folds.
    """
    n_splits = n_splits or K_FOLDS
    console = Console()

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold magenta]CLASSIFICATION REPORTS PER FOLD[/bold magenta]")
    console.print(f"{'=' * 60}")

    for i in range(n_splits):
        console.print(f"\n{'─' * 40} [bold]FOLD {i + 1}[/bold] {'─' * 40}")
        console.print(Panel(store["xgb"]["reports"][i], title=f"[bold green]XGBoost[/bold green] ({store['xgb']['times'][i]:.2f}s)", border_style="green"))
        console.print(Panel(store["tabpfn"]["reports"][i], title=f"[bold cyan]TabPFN v2.5[/bold cyan] ({store['tabpfn']['times'][i]:.2f}s)", border_style="cyan"))


def print_summary(store: dict, n_splits: int | None = None) -> None:
    """Print aggregate reports and timing summary.

    Args:
        store: Results from run_cv.
        n_splits: Number of folds.
    """
    console = Console()
    n_splits = n_splits or K_FOLDS

    console.print(f"\n{'=' * 60}")
    console.print(f"[bold magenta]AGGREGATE CLASSIFICATION REPORT (ALL FOLDS COMBINED)[/bold magenta]")
    console.print(f"{'=' * 60}")

    for label, key, color in [("XGBoost", "xgb", "green"), ("TabPFN v2.5", "tabpfn", "cyan")]:
        all_trues = np.concatenate(store[key]["trues"])
        all_preds = np.concatenate(store[key]["preds"])
        total_time = sum(store[key]["times"])
        mean_time = np.mean(store[key]["times"])
        std_time = np.std(store[key]["times"])

        console.print(f"\n[bold {color}]{label}[/bold {color}] [dim]Total: {total_time:.2f}s | Mean: {mean_time:.2f}s +/- {std_time:.2f}s[/dim]")
        report_fn = xgb_report if key == "xgb" else tabpfn_report
        console.print(Panel(report_fn(all_trues, all_preds), border_style=color))

    xgb_total = sum(store["xgb"]["times"])
    tabpfn_total = sum(store["tabpfn"]["times"])

    table = Table(title="[bold magenta]TIMING COMPARISON[/bold magenta]", show_header=True, header_style="bold white")
    table.add_column("Model", style="bold")
    table.add_column("Total Time", justify="right")
    table.add_column("Avg per Fold", justify="right")
    table.add_row("[green]XGBoost[/green]", f"{xgb_total:.2f}s", f"{xgb_total / n_splits:.2f}s")
    table.add_row("[cyan]TabPFN v2.5[/cyan]", f"{tabpfn_total:.2f}s", f"{tabpfn_total / n_splits:.2f}s")

    faster = "XGBoost" if xgb_total < tabpfn_total else "TabPFN v2.5"
    diff = abs(xgb_total - tabpfn_total)
    table.caption = f"[bold green]Faster:[/bold green] {faster} by {diff:.2f}s"

    console.print(f"\n{'=' * 60}")
    console.print(table)
    console.print(f"{'=' * 60}")

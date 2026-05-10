import argparse
from rich.console import Console
from src.pipeline import run


def main() -> None:
    """Parse CLI arguments and run the comparison pipeline."""
    parser = argparse.ArgumentParser(description="XGBoost vs TabPFN v2.5")
    parser.add_argument("--mode", choices=["vanilla", "balanced", "tuned", "all"], default="vanilla", help="Pipeline mode")
    parser.add_argument("--samples", type=int, nargs="+", default=[10001], help="Sample size(s)")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials")
    parser.add_argument("--folds", type=int, default=5, help="K-fold splits")
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold yellow]Mode: {args.mode.upper()} | Samples: {args.samples}[/bold yellow]")

    for sample_size in args.samples:
        console.print(f"\n[bold magenta]{'=' * 60}[/bold magenta]")
        console.print(f"[bold magenta]SAMPLE SIZE: {sample_size:,}[/bold magenta]")
        console.print(f"[bold magenta]{'=' * 60}[/bold magenta]")
        run(mode=args.mode, sample_size=sample_size, n_trials=args.trials, n_folds=args.folds)


if __name__ == "__main__":
    main()

import argparse
from rich.console import Console
from src.pipeline import run


def main() -> None:
    """Parse CLI arguments and run the comparison pipeline."""
    parser = argparse.ArgumentParser(description="XGBoost vs TabPFN v2.5")
    parser.add_argument("--mode", choices=["vanilla", "balanced", "tuned", "all"], default="vanilla", help="Pipeline mode")
    parser.add_argument("--samples", type=int, default=10001, help="Sample size")
    parser.add_argument("--trials", type=int, default=100, help="Optuna trials")
    parser.add_argument("--folds", type=int, default=5, help="K-fold splits")
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold yellow]Mode: {args.mode.upper()}[/bold yellow]")

    run(mode=args.mode, sample_size=args.samples, n_trials=args.trials, n_folds=args.folds)


if __name__ == "__main__":
    main()

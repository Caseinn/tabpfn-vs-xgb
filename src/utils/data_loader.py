import pandas as pd
from pathlib import Path
from rich.console import Console

from src.utils.config import DATA_PATH, TARGET_COLUMN, SAMPLE_SIZE, RANDOM_STATE


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load dataset from CSV file.

    Args:
        path: Path to the CSV file. Defaults to DATA_PATH from config.

    Returns:
        DataFrame containing the full dataset.
    """
    console = Console()
    path = path or DATA_PATH
    df = pd.read_csv(path)
    console.print(f"[bold]Loaded[/bold] [cyan]{len(df):,}[/cyan] rows x [cyan]{len(df.columns)}[/cyan] columns")
    return df


def print_class_dist(df: pd.DataFrame, col: str = TARGET_COLUMN) -> None:
    """Print class distribution as counts and percentages.

    Args:
        df: DataFrame containing the target column.
        col: Name of the target column.
    """
    console = Console()
    dist = df[col].value_counts().sort_index()
    for cls, cnt in dist.items():
        label = "No Diabetes" if cls == 0 else "Diabetes"
        console.print(f"  [cyan]{label}[/cyan]: {cnt:,} ({cnt / len(df) * 100:.2f}%)")


def stratified_sample_df(
    df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
    target: str = TARGET_COLUMN,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Take a stratified sample preserving exact class distribution.

    Args:
        df: Source DataFrame.
        n: Number of rows to sample.
        target: Name of the target column for stratification.
        seed: Random seed for reproducibility.

    Returns:
        Stratified sample DataFrame with shuffled rows.
    """
    console = Console()
    console.print(f"\n[bold]Original distribution:[/bold]")
    print_class_dist(df)

    sample = pd.concat([
        g.sample(n=int(len(g) * n / len(df)), random_state=seed)
        for _, g in df.groupby(target)
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    console.print(f"\n[bold]Sampled[/bold] [cyan]{len(sample):,}[/cyan] [bold]rows:[/bold]")
    print_class_dist(sample)
    return sample

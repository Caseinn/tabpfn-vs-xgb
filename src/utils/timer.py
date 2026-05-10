import time
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")


@contextmanager
def timed(label: str):
    """Context manager that prints elapsed time.

    Args:
        label: Description of the timed operation.
    """
    from rich.console import Console
    console = Console()
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    console.print(f"  [bold cyan][{label}][/bold cyan] [green]{elapsed:.2f}s[/green]")


def elapsed(fn, *args, **kwargs) -> tuple:
    """Run function and return (result, seconds).

    Args:
        fn: Callable to execute.
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tuple of (return_value, elapsed_seconds).
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start

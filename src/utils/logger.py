import re
import sys
from pathlib import Path
from rich.console import Console
from typing import IO

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class OutputLogger:
    """Capture console output to both terminal and file."""

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] = open(self.filepath, "w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        sys.__stdout__.flush()
        self._file.write(ANSI_RE.sub("", text))
        self._file.flush()

    def flush(self) -> None:
        sys.__stdout__.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def isatty(self) -> bool:
        return True

    def get_console(self) -> Console:
        """Return a Rich Console that writes to both terminal and file."""
        return Console(file=self, force_terminal=True, width=120)

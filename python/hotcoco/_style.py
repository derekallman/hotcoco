"""Terminal styling for hotcoco CLI output.

Two independent gates control output behavior:
- _use_color: True unless NO_COLOR is set. ANSI codes work in Jupyter/Colab too.
- _use_spinner: Also requires isatty() — \\r cursor movement breaks in notebooks.

All styled status output goes to stderr, keeping stdout clean for --json and piping.
"""

import os
import sys
import threading
import time

_use_color: bool = "NO_COLOR" not in os.environ
_use_spinner: bool = sys.stderr.isatty() and _use_color

_RESET = "\033[0m"
_GREEN = "\033[32m"
_DIM = "\033[2m"
_BRIGHT_RED = "\033[91m"
_BRIGHT_YELLOW = "\033[93m"


def green(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}" if _use_color else text


def red(text: str) -> str:
    return f"{_BRIGHT_RED}{text}{_RESET}" if _use_color else text


def yellow(text: str) -> str:
    return f"{_BRIGHT_YELLOW}{text}{_RESET}" if _use_color else text


def dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}" if _use_color else text


def status(verb: str, message: str, *, elapsed: float | None = None) -> None:
    """Print a styled status line to stderr.

    Example:
        status("Loaded", "val2017.json (5,000 images)", elapsed=0.42)
        # Output: Loaded val2017.json (5,000 images) in 0.42s
    """
    parts = [f"{green(verb)} {message}"]
    if elapsed is not None:
        parts.append(dim(f"in {elapsed:.2f}s"))
    print(" ".join(parts), file=sys.stderr)


def error(message: str) -> None:
    """Print 'error: message' in red to stderr."""
    print(f"{red('error')}: {message}", file=sys.stderr)


def warning(message: str) -> None:
    """Print 'warning: message' in yellow to stderr."""
    print(f"{yellow('warning')}: {message}", file=sys.stderr)


def section(title: str, params: str = "") -> None:
    """Print a section header to stdout: green title, dim parameters."""
    if params:
        print(f"\n{green(title)}  {dim('(' + params + ')')}\n")
    else:
        print(f"\n{green(title)}\n")


class Timer:
    """Context manager that measures elapsed wall-clock time."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start


# Braille spinner frames (same sequence as uv/cargo)
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_DELAY = 0.1  # seconds before spinner appears
_SPINNER_INTERVAL = 0.08  # seconds between frame updates


class Spinner:
    """Delayed-start spinner on stderr. No-op in non-TTY environments.

    The spinner only appears if the operation takes longer than 100ms,
    avoiding visual jitter for fast operations.
    """

    def __init__(self, message: str) -> None:
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        if not _use_spinner:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        # Clear the spinner line
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()

    def _spin(self) -> None:
        # Delay before showing anything
        if self._stop.wait(_SPINNER_DELAY):
            return  # operation finished before delay elapsed

        idx = 0
        while not self._stop.wait(_SPINNER_INTERVAL):
            frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
            if _use_color:
                line = f"\r{_GREEN}{frame}{_RESET} {self._message}"
            else:
                line = f"\r{frame} {self._message}"
            sys.stderr.write(line)
            sys.stderr.flush()
            idx += 1

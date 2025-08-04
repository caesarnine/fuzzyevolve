from __future__ import annotations
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List

def setup_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    use_rich: bool = True,
    suppress_libs: tuple[str, ...] = ("litellm", "httpcore", "urllib3"),
    quiet: bool = False,
    log_file: Optional[Path] = None,
) -> None:
    """
    Configure root logger:

    • Console handler (Rich-coloured if available)
    • File handler  logs/YYYY-mm-dd_HHMMSS.log
    • Downgrade chatty 3rd-party libraries to WARNING
    """
    if log_file:
        file_path = log_file
        log_dir = log_file.parent
        log_dir.mkdir(exist_ok=True, parents=True)
    else:
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        file_path = log_dir / f"{ts}.log"

    handlers: list[logging.Handler] = []

    if not quiet:
        if use_rich:
            try:
                from rich.logging import RichHandler

                handlers.append(
                    RichHandler(
                        markup=True,
                        rich_tracebacks=True,
                        show_path=False,
                        level=level,
                        log_time_format="%H:%M:%S",
                    )
                )
            except ImportError:
                use_rich = False  # graceful fallback

        if not use_rich:
            stream = logging.StreamHandler(sys.stdout)
            stream.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
                )
            )
            handlers.append(stream)

    file_h = logging.FileHandler(file_path, encoding="utf-8")
    file_h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers.append(file_h)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.getLogger().info("Logging to %s", file_path)

    for lib in suppress_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)

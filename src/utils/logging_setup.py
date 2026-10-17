"""
src/utils/logging_setup.py
----------------------------
Configure root logger with structured, readable output.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Set up root logger with a clean format.

    Parameters
    ----------
    level : str — "DEBUG", "INFO", "WARNING", "ERROR"
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Avoid duplicate handlers on repeated calls
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers = [handler]

    # Suppress noisy library loggers at INFO level
    for noisy in ["urllib3", "filelock", "git", "PIL"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

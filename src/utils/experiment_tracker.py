"""
src/utils/experiment_tracker.py
---------------------------------
Lightweight CSV experiment tracker.

Every call to ExperimentTracker.log(row) appends one row to results.csv.
Columns are created automatically from the union of all keys ever logged.

Usage:
    tracker = ExperimentTracker("experiments/results.csv")
    tracker.log({"model": "qwen", "task": "qa", "f1": 72.3, ...})

    df = tracker.load()   # returns pandas DataFrame
    tracker.print_summary()
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical column order for the CSV (extra keys are appended at the end)
_PREFERRED_COLUMNS = [
    "experiment",
    "model",
    "dataset",
    "task",
    "method",
    "train_samples",
    "val_samples",
    "test_samples",
    "exact_match",
    "f1",
    "rouge1",
    "rouge2",
    "rougeL",
    "accuracy",
    "macro_f1",
    "precision",
    "recall",
    "train_time_s",
    "peak_gpu_mb",
    "avg_latency_ms",
    "throughput_samples_per_s",
    "checkpoint",
]


class ExperimentTracker:
    def __init__(self, results_csv: str = "experiments/results.csv") -> None:
        self.path = Path(results_csv)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def log(self, row: dict) -> None:
        """Append a result row to the CSV file."""
        existing_columns = self._get_existing_columns()

        # Build ordered column list
        new_keys = [k for k in row if k not in existing_columns]
        all_columns = existing_columns + new_keys
        # Re-sort to canonical order + extras
        all_columns = self._sort_columns(all_columns)

        # Rewrite header if new columns appeared
        file_exists = self.path.exists()
        if not file_exists or new_keys:
            self._rewrite_header(all_columns)

        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Experiment logged to %s", self.path)

    # ------------------------------------------------------------------
    def load(self):
        """Return a pandas DataFrame of all logged experiments."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to load results. Run: pip install pandas") from exc

        if not self.path.exists():
            import pandas as pd
            return pd.DataFrame()

        return pd.read_csv(self.path)

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print a formatted summary table to stdout."""
        df = self.load()
        if df.empty:
            print("No experiments logged yet.")
            return
        summary_cols = [c for c in _PREFERRED_COLUMNS if c in df.columns]
        print(df[summary_cols].to_string(index=False))

    # ------------------------------------------------------------------
    def _get_existing_columns(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                return next(reader)
            except StopIteration:
                return []

    def _sort_columns(self, columns: list[str]) -> list[str]:
        preferred = [c for c in _PREFERRED_COLUMNS if c in columns]
        extras = [c for c in columns if c not in _PREFERRED_COLUMNS]
        return preferred + extras

    def _rewrite_header(self, columns: list[str]) -> None:
        """Rewrite the CSV under new columns, keeping old rows aligned by column name."""
        if not self.path.exists():
            # Nothing to rewrite; header will be written on first append
            return
        with self.path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

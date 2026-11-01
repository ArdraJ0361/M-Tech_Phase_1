"""Fast, offline checks for the financial QA split and the results CSV. Run: python tests/test_fin10k.py"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets import Dataset

from src.data.dataset_loader import _split_fin10k
from src.utils.experiment_tracker import ExperimentTracker


def _rows():
    return Dataset.from_list([
        {"ticker": f"T{t}", "question": f"q{t}-{i}", "answer": f"a{t}x{i}", "context": f"c{t}-{i} a{t}x{i}"}
        for t in range(40) for i in range(10)
    ])


def test_split():
    first, second = _split_fin10k(_rows()), _split_fin10k(_rows())
    assert first["test"].to_list() == second["test"].to_list(), "test set must be reproducible"

    questions = {role: {e["question"] for e in first[role]} for role in first}
    companies = {role: {q.split("-")[0] for q in qs} for role, qs in questions.items()}
    assert not companies["train"] & companies["test"], "test companies leaked into train"
    assert not companies["train"] & companies["validation"]
    assert not companies["validation"] & companies["test"]

    for ds in first.values():
        unanswerable = [e for e in ds if not e["answers"]["text"]]
        assert len(unanswerable) == int((len(ds) - len(unanswerable)) * 0.2)
        for e in unanswerable:
            # Context ids look like "c7-3", questions like "q7-3": companies must differ.
            assert e["context"].split("-")[0][1:] != e["question"].split("-")[0][1:]


def test_tracker_keeps_rows_aligned():
    with tempfile.TemporaryDirectory() as d:
        tracker = ExperimentTracker(os.path.join(d, "results.csv"))
        tracker.log({"experiment": "base", "f1": 1.5, "limitation": "x"})
        # A new column that sorts before f1 used to shift the older row's values.
        tracker.log({"experiment": "lora", "train_samples": 10, "f1": 2.5})
        df = tracker.load()
        assert df.loc[0, "f1"] == 1.5 and df.loc[0, "limitation"] == "x"
        assert df.loc[1, "f1"] == 2.5 and df.loc[1, "train_samples"] == 10


if __name__ == "__main__":
    test_split()
    test_tracker_keeps_rows_aligned()
    print("[OK] all checks passed")

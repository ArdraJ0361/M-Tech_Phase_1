"""
src/evaluation/metrics.py
--------------------------
Task-specific evaluation metrics.

  - QA          : Exact Match + Token-F1  (SQuAD-style)
  - Summarization: ROUGE-1 / ROUGE-2 / ROUGE-L
  - Classification: Accuracy + Macro-F1
  - Extraction   : Token-level Precision / Recall / F1
"""

from __future__ import annotations

import re
import string
import logging
from collections import Counter
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QA Metrics
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = get_tokens(prediction)
    gt_tokens = get_tokens(ground_truth)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_qa_metrics(
    predictions: list[str],
    references: list[str],
    report_unanswerable_breakdown: bool = True,
) -> dict:
    """
    Compute Exact Match and F1 over a list of predictions and references.

    SQuAD v2 unanswerable breakdown (guide feedback round 2, point D):
    When report_unanswerable_breakdown=True (default), metrics are reported
    for three groups:
      - All examples combined
      - Answerable examples only (reference != "unanswerable")
      - Unanswerable examples only (reference == "unanswerable")

    This matters because a model can inflate its F1 score on unanswerable
    examples just by outputting the string "unanswerable" for everything.
    Reporting the breakdown separately makes this visible.
    """
    em_scores = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [compute_f1(p, r) for p, r in zip(predictions, references)]

    result = {
        "exact_match": round(np.mean(em_scores) * 100, 2),
        "f1": round(np.mean(f1_scores) * 100, 2),
        "num_examples": len(predictions),
    }

    if report_unanswerable_breakdown:
        answerable_pairs = [
            (p, r, em, f1)
            for p, r, em, f1 in zip(predictions, references, em_scores, f1_scores)
            if normalize_answer(r) != "unanswerable"
        ]
        unanswerable_pairs = [
            (p, r, em, f1)
            for p, r, em, f1 in zip(predictions, references, em_scores, f1_scores)
            if normalize_answer(r) == "unanswerable"
        ]

        if answerable_pairs:
            result["answerable_em"]  = round(np.mean([x[2] for x in answerable_pairs]) * 100, 2)
            result["answerable_f1"]  = round(np.mean([x[3] for x in answerable_pairs]) * 100, 2)
            result["answerable_n"]   = len(answerable_pairs)

        if unanswerable_pairs:
            result["unanswerable_em"] = round(np.mean([x[2] for x in unanswerable_pairs]) * 100, 2)
            result["unanswerable_f1"] = round(np.mean([x[3] for x in unanswerable_pairs]) * 100, 2)
            result["unanswerable_n"]  = len(unanswerable_pairs)

    return result


# ---------------------------------------------------------------------------
# ROUGE (Summarization)
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L using the rouge_score library."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("rouge-score is required. Run: pip install rouge-score") from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for k in scores:
            scores[k].append(result[k].fmeasure)
    return {
        "rouge1": round(np.mean(scores["rouge1"]) * 100, 2),
        "rouge2": round(np.mean(scores["rouge2"]) * 100, 2),
        "rougeL": round(np.mean(scores["rougeL"]) * 100, 2),
        "num_examples": len(predictions),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def compute_classification_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute accuracy and macro-F1 for document classification."""
    try:
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required. Run: pip install scikit-learn") from exc

    accuracy = accuracy_score(references, predictions)
    macro_f1 = f1_score(references, predictions, average="macro", zero_division=0)
    return {
        "accuracy": round(accuracy * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "num_examples": len(predictions),
    }


# ---------------------------------------------------------------------------
# Information Extraction (token-level P/R/F1)
# ---------------------------------------------------------------------------

def compute_extraction_metrics(predictions: list[str], references: list[str]) -> dict:
    """
    Token-level precision, recall, F1 for field extraction.
    Treats each space-separated token as a unit.
    """
    total_precision, total_recall, total_f1 = [], [], []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(get_tokens(pred))
        ref_tokens = set(get_tokens(ref))
        if not ref_tokens:
            continue
        tp = len(pred_tokens & ref_tokens)
        precision = tp / len(pred_tokens) if pred_tokens else 0.0
        recall = tp / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total_precision.append(precision)
        total_recall.append(recall)
        total_f1.append(f1)
    return {
        "precision": round(np.mean(total_precision) * 100, 2) if total_precision else 0.0,
        "recall": round(np.mean(total_recall) * 100, 2) if total_recall else 0.0,
        "f1": round(np.mean(total_f1) * 100, 2) if total_f1 else 0.0,
        "num_examples": len(predictions),
    }


# ---------------------------------------------------------------------------
# HuggingFace Trainer compute_metrics callback factory
# ---------------------------------------------------------------------------

def make_compute_metrics_fn(task: str, tokenizer) -> Optional[Callable]:
    """
    Return a compute_metrics function compatible with HuggingFace Trainer.

    The Trainer passes (predictions, label_ids) as EvalPrediction.
    We decode the token IDs back to text and compute task metrics.
    """

    def _decode(token_ids) -> list[str]:
        token_ids = np.where(token_ids == -100, tokenizer.pad_token_id, token_ids)
        return tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # For causal LM, predictions are logits — take argmax
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = _decode(predictions)
        decoded_labels = _decode(labels)

        # Clean up decoded outputs
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        if task == "qa":
            return compute_qa_metrics(decoded_preds, decoded_labels)
        elif task == "summarization":
            return compute_rouge(decoded_preds, decoded_labels)
        elif task == "classification":
            return compute_classification_metrics(decoded_preds, decoded_labels)
        elif task == "extraction":
            return compute_extraction_metrics(decoded_preds, decoded_labels)
        else:
            return {}

    return compute_metrics

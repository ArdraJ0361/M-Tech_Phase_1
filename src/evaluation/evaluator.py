"""
src/evaluation/evaluator.py
----------------------------
Standalone evaluation of a model on a dataset split.
Used for base-model baseline comparison and post-training evaluation.

The tokenized splits hold  prompt + completion + EOS  with labels = -100 on the
prompt. For generation we recover the PROMPT ONLY (everything before the first
trained label) so the model never sees the reference answer, and we decode the
reference from the labels.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .metrics import (
    compute_classification_metrics,
    compute_extraction_metrics,
    compute_qa_metrics,
    compute_rouge,
    normalize_answer,
)

logger = logging.getLogger(__name__)

_TASK_METRIC_FN = {
    "qa": compute_qa_metrics,
    "summarization": compute_rouge,
    "classification": compute_classification_metrics,
    "extraction": compute_extraction_metrics,
}


def _split_prompt_and_reference(example: dict) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, reference_ids) from one tokenized example."""
    labels = example["labels"]
    start = next(i for i, t in enumerate(labels) if t != -100)
    return list(example["input_ids"][:start]), [t for t in labels[start:] if t != -100]


def _postprocess(pred: str, task: str) -> str:
    """
    Cut the generation where the model starts a new prompt section.
    A base model does not emit EOS, so it keeps going ("\\n\\n### Question: ...").
    Applied identically to every model so the comparison stays fair.
    """
    pred = pred.split("\n###")[0].strip()
    if task == "qa":
        pred = pred.split("\n")[0].strip()
    return pred


def evaluate_model(model, tokenizer, dataset, cfg: dict, batch_size: int = 8) -> dict:
    """
    Generate predictions on `dataset` (tokenized split) and compute task metrics.

    Returns a dict of metric name -> value, including avg_latency_ms (per example)
    and throughput_samples_per_s.
    """
    task = cfg["task"]
    max_new_tokens = cfg["dataset"].get("max_target_length", 64)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    model.eval()
    pairs = [_split_prompt_and_reference(dataset[i]) for i in range(len(dataset))]

    all_preds: list[str] = []
    all_refs = [tokenizer.decode(ref, skip_special_tokens=True).strip() for _, ref in pairs]
    gen_seconds = 0.0

    with torch.no_grad():
        for b in range(0, len(pairs), batch_size):
            prompts = [p for p, _ in pairs[b:b + batch_size]]
            width = max(len(p) for p in prompts)
            # LEFT-pad so every prompt ends at the same position before generation
            input_ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in prompts], device=device)
            attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            gen_seconds += time.perf_counter() - t0

            decoded = tokenizer.batch_decode(output_ids[:, width:], skip_special_tokens=True)
            all_preds.extend(_postprocess(p, task) for p in decoded)

    metric_fn = _TASK_METRIC_FN.get(task)
    if metric_fn is None:
        logger.warning("No metric function for task: %s", task)
        return {}

    metrics = metric_fn(all_preds, all_refs)
    # Exact match is too strict for sentence answers, so such configs also request ROUGE.
    if task == "qa" and "rougeL" in cfg.get("evaluation", {}).get("metrics", []):
        answerable = [(p, r) for p, r in zip(all_preds, all_refs) if normalize_answer(r) != "unanswerable"]
        if answerable:
            rouge = compute_rouge([p for p, _ in answerable], [r for _, r in answerable])
            metrics.update({f"answerable_{k}": rouge[k] for k in ("rouge1", "rouge2", "rougeL")})
    metrics["avg_latency_ms"] = round(gen_seconds * 1000 / len(all_preds), 1)
    metrics["throughput_samples_per_s"] = round(len(all_preds) / gen_seconds, 2)
    metrics["_predictions"] = all_preds      # callers may save these for error analysis
    metrics["_references"] = all_refs
    metrics["_prompts"] = [tokenizer.decode(p, skip_special_tokens=True) for p, _ in pairs]
    return metrics


def save_predictions(metrics: dict, path: Path) -> dict:
    """Write per-example outputs to a CSV and return the metrics without them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "reference", "prediction"])
        writer.writerows(zip(metrics["_prompts"], metrics["_references"], metrics["_predictions"]))
    logger.info("Predictions saved to %s", path)
    return {k: v for k, v in metrics.items() if not k.startswith("_")}

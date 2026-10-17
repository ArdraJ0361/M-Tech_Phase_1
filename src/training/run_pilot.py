"""
src/training/run_pilot.py
--------------------------
Command-line entry point for fine-tuning experiments.

Usage:
    python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml

    # Evaluate only (skip training)
    python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml --eval_only
    
    # Base model inference (no fine-tuning — for baseline comparison)
    python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml --base_only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the Python path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config_loader import load_config
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEFT-SLM Document Analysis — Pilot Training")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; evaluate an existing checkpoint from config output_dir",
    )
    parser.add_argument(
        "--base_only",
        action="store_true",
        help="Evaluate the base (untuned) model only — for baseline comparison",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config)
    logger.info("Config loaded: %s", args.config)

    if args.base_only:
        _run_base_evaluation(cfg)
    elif args.eval_only:
        _run_checkpoint_evaluation(cfg)
    else:
        from src.training.trainer import run_training
        metrics = run_training(args.config)
        logger.info("Final test metrics: %s", metrics)


def _run_base_evaluation(cfg: dict) -> None:
    """
    Evaluate the untuned base model as a zero-shot baseline.

    IMPORTANT LIMITATION — base model zero-shot (guide feedback round 2, point B):

    A base (non-instruction-tuned) causal language model is not guaranteed to
    follow task instructions reliably in zero-shot mode. For example:
      - A base model asked "### Answer:\n" may continue generating more context
        rather than producing a concise answer.
      - Zero-shot QA and summarization performance of base models is often
        substantially lower than instruct-tuned models of the same size.

    This is a known limitation of the research design and MUST be reported
    when the results table is written. Do not compare base-model zero-shot
    results against instruction-tuned model results.

    The purpose of the base SLM row in the 3-way table is to show the
    IMPROVEMENT from PEFT fine-tuning, not to show the absolute capability
    of the model family.
    """
    import torch
    from src.training.model_loader import load_tokenizer, load_base_model
    from src.data.dataset_loader import load_dataset_for_task
    from src.evaluation.evaluator import evaluate_model
    from src.utils.experiment_tracker import ExperimentTracker

    model_variant = cfg.get("model_variant", "base")
    logger.info(
        "Running BASE MODEL (%s, zero-shot) evaluation for model: %s",
        model_variant, cfg["model_name_or_path"],
    )
    logger.warning(
        "Zero-shot base model evaluation — results may be low because base "
        "models do not reliably follow task instructions without fine-tuning. "
        "Report this limitation explicitly in your results section.",
    )

    tokenizer = load_tokenizer(cfg)
    model = load_base_model(cfg)
    model.eval()

    dataset = load_dataset_for_task(cfg, tokenizer)
    test_split = dataset.get("test") or dataset.get("validation")
    if test_split is None:
        raise RuntimeError("No test or validation split found. Check dataset config.")

    metrics = evaluate_model(model, tokenizer, test_split, cfg)
    logger.info("Base model metrics: %s", metrics)

    tracker = ExperimentTracker(cfg["results_csv"])
    tracker.log({
        "experiment": cfg.get("experiment_name", "") + "_base",
        "model": cfg["model_name_or_path"],
        "model_variant": model_variant,
        "dataset": cfg["dataset"].get("name", ""),
        "task": cfg["task"],
        "method": "base_zero_shot",
        "seed": cfg.get("training", {}).get("seed", 42),
        "limitation": "base_model_zero_shot_may_not_follow_instructions",
        **metrics,
    })


def _run_checkpoint_evaluation(cfg: dict) -> None:
    """Evaluate a saved LoRA checkpoint."""
    import torch
    from pathlib import Path
    from src.training.model_loader import load_tokenizer, load_peft_checkpoint
    from src.data.dataset_loader import load_dataset_for_task
    from src.evaluation.evaluator import evaluate_model
    from src.utils.experiment_tracker import ExperimentTracker

    checkpoint_path = str(Path(cfg["training"]["output_dir"]) / "adapter")
    logger.info("Evaluating checkpoint: %s", checkpoint_path)

    tokenizer = load_tokenizer(cfg)
    model = load_peft_checkpoint(cfg, checkpoint_path)

    dataset = load_dataset_for_task(cfg, tokenizer)
    test_split = dataset.get("test") or dataset.get("validation")

    metrics = evaluate_model(model, tokenizer, test_split, cfg)
    logger.info("Checkpoint metrics: %s", metrics)

    tracker = ExperimentTracker(cfg["results_csv"])
    tracker.log({
        "experiment": cfg.get("experiment_name", "") + "_eval",
        "model": cfg["model_name_or_path"],
        "dataset": cfg["dataset"].get("name", ""),
        "task": cfg["task"],
        "method": "LoRA_eval",
        "checkpoint": checkpoint_path,
        **metrics,
    })


if __name__ == "__main__":
    main()

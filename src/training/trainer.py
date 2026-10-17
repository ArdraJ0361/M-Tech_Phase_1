"""
src/training/trainer.py
------------------------
Fine-tuning loop using HuggingFace Trainer + PEFT LoRA.

Usage (via run_pilot.py):
    python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import time
from pathlib import Path

import torch
from transformers import TrainingArguments, Trainer

from .model_loader import load_tokenizer, load_base_model, attach_lora
from ..data.dataset_loader import load_dataset_for_task
from ..data.preprocessor import get_data_collator
from ..evaluation.evaluator import evaluate_model
from ..utils.experiment_tracker import ExperimentTracker
from ..utils.config_loader import load_config

logger = logging.getLogger(__name__)

# transformers>=5 dropped `warmup_ratio`; its `warmup_steps` accepts a float ratio instead.
_WARMUP_KEY = ("warmup_ratio" if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters
               else "warmup_steps")


# ---------------------------------------------------------------------------
# Main fine-tuning entry point
# ---------------------------------------------------------------------------

def run_training(config_path: str) -> dict:
    """
    Run the full fine-tuning pipeline from a YAML config file.

    Returns a dict of evaluation metrics from the best checkpoint.
    """
    cfg = load_config(config_path)
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.get("experiment_name", "unnamed"))
    logger.info("Task      : %s", cfg["task"])
    logger.info("Model     : %s", cfg["model_name_or_path"])
    logger.info("=" * 60)

    # 1. Tokenizer
    tokenizer = load_tokenizer(cfg)

    # 2. Dataset
    dataset = load_dataset_for_task(cfg, tokenizer)
    logger.info(
        "Dataset splits — train: %d  |  val: %d  |  test: %d",
        len(dataset.get("train", [])),
        len(dataset.get("validation", [])),
        len(dataset.get("test", [])),
    )

    # 3. Model + LoRA
    base_model = load_base_model(cfg)
    model = attach_lora(base_model, cfg)

    # 4. Efficiency tracking (GPU memory baseline)
    tracker = ExperimentTracker(cfg["results_csv"])
    peak_mem_before = _get_peak_gpu_mb()

    # 5. Training arguments
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        **{_WARMUP_KEY: float(train_cfg.get("warmup_ratio", 0.03))},
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        logging_steps=train_cfg.get("logging_steps", 50),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 200),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        seed=train_cfg.get("seed", 42),
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Validation is scored by loss only: keeping full-vocab logits for every
        # validation token (500 x seq x 152K) is what ran the T4 out of memory.
        prediction_loss_only=True,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get("train"),
        eval_dataset=dataset.get("validation"),
        data_collator=get_data_collator(tokenizer),
    )

    # 7. Train
    t_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - t_start
    peak_mem_mb = _get_peak_gpu_mb() - peak_mem_before

    logger.info("Training complete in %.1f s | peak GPU delta: %.0f MB", train_time, peak_mem_mb)

    # 8. Evaluate on test set by generation -- the same evaluate_model() used for the
    #    zero-shot baselines, so all three rows of the comparison are scored identically.
    test_metrics = {}
    if "test" in dataset:
        logger.info("Evaluating on test split...")
        metrics = evaluate_model(model, tokenizer, dataset["test"], cfg)
        test_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
        logger.info("Test metrics: %s", test_metrics)

    # 9. Save adapter + training log (loss per step)
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    (output_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))

    # 10. Log to results CSV — guide feedback: include reproducibility fields
    row = {
        "experiment": cfg.get("experiment_name", ""),
        "model": cfg["model_name_or_path"],
        # model_variant: "base" or "instruct" — must be consistent across all 3 comparison models
        "model_variant": cfg.get("model_variant", "base"),
        "dataset": cfg["dataset"].get("name", ""),
        "dataset_version": cfg["dataset"].get("version", "default"),
        "task": cfg["task"],
        "method": "LoRA" if cfg.get("load_in_bits") is None else "QLoRA",
        "seed": cfg.get("training", {}).get("seed", 42),
        "train_samples": len(dataset.get("train", [])),
        "val_samples": len(dataset.get("validation", [])),
        "test_samples": len(dataset.get("test", [])),
        # Actual trainable parameter counts from model_loader.py assertions
        "trainable_params": cfg.get("_trainable_params", "not_logged"),
        "total_params": cfg.get("_total_params", "not_logged"),
        "trainable_pct": cfg.get("_trainable_pct", "not_logged"),
        "train_time_s": round(train_time, 1),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        "checkpoint": str(output_dir / "adapter"),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()},
    }
    tracker.log(row)

    return test_metrics


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_peak_gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0

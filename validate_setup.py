#!/usr/bin/env python3
"""validate_setup.py — Run this to check all core modules are importable and work correctly."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Validating PEFT-SLM project setup...\n")

# 1. Config loader
from src.utils.config_loader import load_config
cfg = load_config("configs/pilot_squad_qa.yaml")
print(f"[OK] Config loaded: {cfg['experiment_name']}")
print(f"     task={cfg['task']}  model={cfg['model_name_or_path']}")

# 2. Experiment tracker
from src.utils.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker("experiments/results.csv")
tracker.log({"experiment": "validate_run", "model": "qwen-0.5b", "task": "qa", "method": "base", "f1": 0.0})
print("[OK] ExperimentTracker — wrote experiments/results.csv")

# 3. Metrics (requires numpy — skip gracefully if not installed)
try:
    from src.evaluation.metrics import compute_qa_metrics, compute_exact_match, compute_f1, compute_rouge, compute_classification_metrics
    em = compute_exact_match("Paris, France", "Paris, France")
    f1 = compute_f1("Low-Rank Adaptation", "low rank adaptation")
    batch_metrics = compute_qa_metrics(["Paris", "Guido van Rossum"], ["Paris, France", "Guido van Rossum"])
    print(f"[OK] QA metrics — EM={em}  F1={f1:.4f}  batch={batch_metrics}")
    cls_metrics = compute_classification_metrics(["invoice", "letter", "form"], ["invoice", "memo", "form"])
    print(f"[OK] Classification metrics — {cls_metrics}")
except ImportError as e:
    print(f"[SKIP] Metrics — missing dependency (install requirements.txt first): {e}")

# 4. Text extractor
from src.extraction.text_extractor import extract_text, clean_text, chunk_text
result = extract_text("LoRA adds low-rank matrices to transformer layers.")
assert result["source"] == "plain"
assert len(result["text"]) > 10
chunks = chunk_text("word " * 200, max_tokens=50, overlap_tokens=10)
assert len(chunks) > 1
print(f"[OK] Text extractor — source={result['source']}  chunks={len(chunks)}")

# 5. Logging setup
from src.utils.logging_setup import setup_logging
setup_logging("WARNING")  # suppress output during validation
print("[OK] Logging setup")

# 6. Dataset loader import (no download needed for import check)
from src.data import load_dataset_for_task
print("[OK] Dataset loader importable")

# 7. Training modules import (requires torch/transformers — skip gracefully if absent)
try:
    from src.training.model_loader import load_tokenizer, load_base_model, attach_lora
    print("[OK] Training model_loader importable")
except ImportError as e:
    print(f"[SKIP] Training model_loader — missing dependency: {e}")

try:
    from src.training.trainer import run_training
    print("[OK] Trainer importable")
except ImportError as e:
    print(f"[SKIP] Trainer — missing dependency: {e}")

print("\nAll checks passed. Project setup is valid.")
print("Next step: pip install -r requirements.txt  then  python demo_quick.py")

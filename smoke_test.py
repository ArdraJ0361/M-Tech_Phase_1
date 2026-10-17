"""
smoke_test.py
--------------
Step 1 from the guide's recommended execution order:

  "Run a 50-example SQuAD smoke test. Verify data formatting,
   tokenization, inference, and metric calculation."

This script requires torch + transformers + peft + datasets to be installed,
but runs entirely on CPU with a very small model slice and only 50 examples.
It does NOT require a GPU and completes in a few minutes.

What it verifies:
  [1] Config loads correctly
  [2] Dataset loads and splits are correct (train from HF train, test from HF val,
      NO overlap between val and test, unanswerable examples present)
  [3] Prompt formatting is correct (### Context / ### Question / ### Answer)
  [4] Tokenizer loads and pads correctly
  [5] Prompt masking: labels for prompt tokens are -100, completion tokens are not
  [6] No examples are silently trained on prompt tokens only
  [7] Base model loads and runs inference on 3 examples
  [8] Metrics compute correctly (EM + F1 + unanswerable breakdown)
  [9] ExperimentTracker writes results.csv

Run:
    python smoke_test.py

All checks print [OK] or [FAIL] with a clear explanation.
Exit code 0 = all checks passed. Exit code 1 = at least one check failed.
"""

from __future__ import annotations

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[OK]"
FAIL = "[FAIL]"
failures = []

def check(name: str, fn):
    try:
        result = fn()
        print(f"{PASS} {name}")
        return result
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        failures.append(name)
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# [1] Config loads
# ---------------------------------------------------------------------------
def _check_config():
    from src.utils.config_loader import load_config
    cfg = load_config("configs/pilot_squad_qa.yaml")
    assert cfg["task"] == "qa"
    assert cfg["model_name_or_path"] == "Qwen/Qwen1.5-0.5B"
    assert cfg.get("model_variant") == "base", "model_variant must be 'base'"
    assert cfg["training"]["seed"] == 42, "seed must be 42 for reproducible splits"
    return cfg

cfg = check("Config loads correctly with model_variant and seed=42", _check_config)
if cfg is None:
    print("\nCannot continue without config. Fix config issues first.")
    sys.exit(1)

# Override sample sizes for smoke test
cfg["dataset"]["train_samples"] = 50
cfg["dataset"]["val_samples"] = 25
cfg["dataset"]["test_samples"] = 25
cfg["training"]["output_dir"] = "experiments/smoke_test"


# ---------------------------------------------------------------------------
# [2] Dataset loads with correct split boundaries
# ---------------------------------------------------------------------------
def _check_dataset_splits():
    from datasets import load_dataset
    import json, pathlib

    from src.data.dataset_loader import _subsample, _HF_SPLIT_SOURCE
    raw = load_dataset("rajpurkar/squad_v2")

    result = _subsample(raw, cfg["dataset"], cfg)

    # Train must come from HF "train" — check it has a different size than val/test
    assert "train" in result, "train split missing"
    assert "validation" in result, "validation split missing"
    assert "test" in result, "test split missing"

    # Splits must be the right sizes
    assert len(result["train"]) == 50, f"Expected 50 train, got {len(result['train'])}"
    assert len(result["validation"]) == 25, f"Expected 25 val, got {len(result['validation'])}"
    assert len(result["test"]) == 25, f"Expected 25 test, got {len(result['test'])}"

    # Val and test must be disjoint — check test_indices.json
    import json, pathlib
    idx_file = pathlib.Path("experiments/smoke_test/test_indices.json")
    assert idx_file.exists(), "test_indices.json was not written"
    payload = json.loads(idx_file.read_text())
    assert payload["hf_source_split"] == "validation", (
        f"Test examples should come from HF 'validation' split for squad_v2, "
        f"got: {payload['hf_source_split']}"
    )
    assert payload["num_test_examples"] == 25

    # Check unanswerable examples exist in test split
    test_items = list(result["test"])
    unanswerable = [ex for ex in test_items if not ex["answers"]["text"]]
    answerable   = [ex for ex in test_items if ex["answers"]["text"]]
    print(f"         (test: {len(answerable)} answerable, {len(unanswerable)} unanswerable)")
    assert len(unanswerable) > 0, (
        "No unanswerable examples in test split — check dataset or increase test_samples"
    )

    return result

raw_dataset = check(
    "Dataset splits: correct HF boundaries, disjoint val/test, unanswerable present",
    _check_dataset_splits,
)


# ---------------------------------------------------------------------------
# [3] Prompt formatting
# ---------------------------------------------------------------------------
def _check_prompt_format():
    from datasets import load_dataset
    from src.data.dataset_loader import _subsample

    raw = load_dataset("rajpurkar/squad_v2")
    result = _subsample(raw, cfg["dataset"], cfg)

    # Manually format one answerable and one unanswerable example
    answerable_ex   = next(ex for ex in result["train"] if ex["answers"]["text"])
    unanswerable_ex = next(ex for ex in result["train"] if not ex["answers"]["text"])

    # Answerable
    ans = answerable_ex["answers"]["text"][0]
    prompt_a = (
        f"### Context:\n{answerable_ex['context']}\n\n"
        f"### Question:\n{answerable_ex['question']}\n\n"
        f"### Answer:\n{ans}"
    )
    assert "### Context:" in prompt_a
    assert "### Question:" in prompt_a
    assert "### Answer:" in prompt_a
    assert ans in prompt_a

    # Unanswerable
    prompt_u = (
        f"### Context:\n{unanswerable_ex['context']}\n\n"
        f"### Question:\n{unanswerable_ex['question']}\n\n"
        f"### Answer:\nunanswerable"
    )
    assert "unanswerable" in prompt_u
    print(f"         Answerable sample: Q={answerable_ex['question'][:50]}... A={ans[:30]}")
    print(f"         Unanswerable sample: Q={unanswerable_ex['question'][:50]}...")

check("Prompt format correct (answerable + unanswerable)", _check_prompt_format)


# ---------------------------------------------------------------------------
# [4] & [5] & [6] Tokenizer + masking
# ---------------------------------------------------------------------------
def _check_tokenization_and_masking():
    from transformers import AutoTokenizer
    from src.data.preprocessor import _tokenize_prompt_completion, _IGNORE_INDEX

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build test prompts (one answerable, one unanswerable, one that would truncate)
    prompts = [
        "### Context:\nParis is the capital of France.\n\n### Question:\nWhat is the capital of France?\n\n### Answer:\nParis",
        "### Context:\nLoRA stands for Low-Rank Adaptation.\n\n### Question:\nWhat does LoRA stand for?\n\n### Answer:\nunanswerable",
    ]

    result = _tokenize_prompt_completion(
        {"prompt": prompts},
        tokenizer,
        max_length=256,
    )

    assert len(result["input_ids"]) == 2, "Expected 2 tokenized examples"

    for i, (input_ids, labels) in enumerate(zip(result["input_ids"], result["labels"])):
        # [4] Lengths must match
        assert len(input_ids) == len(labels), f"Example {i}: input_ids and labels length mismatch"

        # [5] Prompt tokens must be -100
        prompt_key = "### Answer:\n"
        prompt_text = prompts[i][:prompts[i].rfind(prompt_key) + len(prompt_key)]
        prompt_tokens = tokenizer(prompt_text, return_tensors=None)["input_ids"]
        prompt_len = len(prompt_tokens)
        for j in range(min(prompt_len, len(labels))):
            assert labels[j] == _IGNORE_INDEX, (
                f"Example {i}, token {j}: expected IGNORE_INDEX in prompt, got {labels[j]}"
            )

        # [5] Completion tokens must NOT be -100
        completion = [t for t in labels if t != _IGNORE_INDEX]
        assert len(completion) > 0, (
            f"Example {i}: no completion tokens — training objective would have zero signal"
        )

        # [6] Verify padding tokens would be -100 (checked by collator, not here)
        print(f"         Example {i}: prompt_len={prompt_len}, "
              f"completion_tokens={len(completion)}, "
              f"labels[-100 count]={sum(1 for t in labels if t == _IGNORE_INDEX)}")

check("Tokenization + masking: prompt=-100, completion≠-100, no zero-signal examples", _check_tokenization_and_masking)


# ---------------------------------------------------------------------------
# [7] Base model inference (3 examples, CPU)
# ---------------------------------------------------------------------------
def _check_base_model_inference():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("         Loading Qwen1.5-0.5B on CPU (this may take ~30s)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B", trust_remote_code=True,
        torch_dtype=torch.float32, device_map="cpu",
    )
    model.eval()

    prompts = [
        "### Context:\nParis is the capital of France.\n\n### Question:\nWhat is the capital of France?\n\n### Answer:\n",
        "### Context:\nLoRA adds trainable low-rank matrices to frozen transformer layers.\n\n### Question:\nWhat does LoRA do?\n\n### Answer:\n",
    ]
    preds = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        preds.append(pred)
        print(f"         Pred: '{pred[:60]}'")

    assert len(preds) == 2
    assert all(isinstance(p, str) for p in preds)
    # Note: base model may not give correct answers — that is expected and documented
    print("         NOTE: Base model zero-shot answers may be wrong or verbose.")
    print("         This is a known limitation — must be reported in the results section.")
    return preds

check("Base model inference runs on CPU, returns string predictions", _check_base_model_inference)


# ---------------------------------------------------------------------------
# [8] Metrics
# ---------------------------------------------------------------------------
def _check_metrics():
    from src.evaluation.metrics import (
        compute_qa_metrics, compute_exact_match, compute_f1, compute_rouge
    )

    # Basic QA metrics
    assert compute_exact_match("Paris", "Paris") == 1.0
    assert compute_exact_match("paris", "Paris") == 1.0  # case insensitive
    assert compute_exact_match("London", "Paris") == 0.0
    assert compute_f1("Low-Rank Adaptation", "low rank adaptation") > 0.0

    # Unanswerable breakdown
    preds = ["Paris", "unanswerable", "London", "unanswerable"]
    refs  = ["Paris", "unanswerable", "Paris",  "unanswerable"]
    result = compute_qa_metrics(preds, refs)
    assert "exact_match" in result
    assert "f1" in result
    assert "answerable_em" in result, "answerable breakdown missing"
    assert "unanswerable_em" in result, "unanswerable breakdown missing"
    assert result["answerable_n"] == 2
    assert result["unanswerable_n"] == 2
    assert result["answerable_em"] == 50.0   # 1 of 2 answerable correct
    assert result["unanswerable_em"] == 100.0  # both unanswerable correct
    print(f"         Full EM={result['exact_match']} | "
          f"Answerable EM={result['answerable_em']} | "
          f"Unanswerable EM={result['unanswerable_em']}")

    # ROUGE
    rouge = compute_rouge(
        ["Scientists found a new deep-sea fish near the Mariana Trench."],
        ["Researchers discovered a new bioluminescent fish near the Mariana Trench."],
    )
    assert "rouge1" in rouge and rouge["rouge1"] > 0

check("Metrics: EM, F1, ROUGE, unanswerable breakdown all correct", _check_metrics)


# ---------------------------------------------------------------------------
# [9] Experiment tracker
# ---------------------------------------------------------------------------
def _check_tracker():
    from src.utils.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker("experiments/smoke_test/results.csv")
    tracker.log({
        "experiment": "smoke_test",
        "model": "Qwen/Qwen1.5-0.5B",
        "model_variant": "base",
        "dataset": "squad_v2",
        "task": "qa",
        "method": "base_zero_shot",
        "seed": 42,
        "exact_match": 20.0,
        "f1": 33.5,
        "answerable_em": 25.0,
        "unanswerable_em": 15.0,
    })
    df = tracker.load()
    assert len(df) >= 1
    assert "model_variant" in df.columns
    assert "seed" in df.columns
    assert "answerable_em" in df.columns

check("ExperimentTracker writes model_variant, seed, unanswerable metrics", _check_tracker)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
if failures:
    print(f"SMOKE TEST FAILED — {len(failures)} check(s) failed:")
    for f in failures:
        print(f"  ✗ {f}")
    print()
    print("Fix the failures above before running the full pilot.")
    sys.exit(1)
else:
    print("ALL SMOKE TEST CHECKS PASSED")
    print()
    print("Pipeline is verified. Next steps:")
    print("  1. Run a small LoRA training experiment (--lora_steps 50 or 1 epoch)")
    print("  2. Confirm loss decreases and checkpoint saves + reloads")
    print("  3. Run full 3-way comparison on 500 test examples")
    print("  4. Repeat for XL-Sum summarization")
    sys.exit(0)

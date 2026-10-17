"""
demo_quick.py
--------------
End-to-end demo of the 3-way comparison at the heart of this M.Tech project:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  PEFT-adapted SLM  vs  Base SLM (before tuning)  vs  LLM baseline  │
  └─────────────────────────────────────────────────────────────────────┘

  Step 1 — Load SLM  (Qwen1.5-0.5B, 500 M params)
  Step 2 — Run zero-shot inference  → base SLM metrics
  Step 3 — Attach LoRA adapter (PEFT) and fine-tune for N steps
  Step 4 — Run inference again      → PEFT-SLM metrics   ← your contribution
  Step 5 — Load LLM  (Qwen1.5-7B, 7 B params, zero-shot)
  Step 6 — Run LLM inference        → LLM baseline metrics
  Step 7 — Print 3-way comparison table + per-example predictions
  Step 8 — Log all 3 runs to experiments/results.csv

Why Qwen1.5-7B as the larger-model baseline?
  Same model family as the SLM → fairest possible comparison.
  Tokenizer is identical. Only the scale differs (0.5B vs 7B).
  This isolates the effect of model size, making it a clean research result.

Identical test examples — what this means precisely (guide feedback B):
  For each task separately, all three models are evaluated on the same held-out
  test examples, using the same input format and evaluation procedure.
  SQuAD v2 QA examples and XL-Sum summarization examples are DIFFERENT datasets —
  the same 500 examples cannot be shared across tasks. "Identical" means:
    - same 500 SQuAD v2 examples for all three models on the QA task, and
    - same 500 XL-Sum examples for all three models on the summarization task.
  This is enforced by seed=42 in all three YAML configs.

Usage:
    python demo_quick.py                      # QA task, default models
    python demo_quick.py --task summarization
    python demo_quick.py --lora_steps 20      # more tuning steps
    python demo_quick.py --skip_llm           # skip LLM step (CPU / low VRAM)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logging_setup import setup_logging
from src.utils.experiment_tracker import ExperimentTracker
from src.evaluation.metrics import compute_qa_metrics, compute_rouge

setup_logging("INFO")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SLM and LLM model choices
# ---------------------------------------------------------------------------
SLM_MODEL   = "Qwen/Qwen1.5-0.5B"   # ~500 M params  — the model we PEFT-tune
LLM_MODEL   = "Qwen/Qwen1.5-7B"     # ~7 B params    — the LARGER-MODEL BASELINE
# Both are from the same Qwen1.5 family → fair, controlled comparison.
# GUIDE FEEDBACK B: The 7B model is a "larger-model baseline" — NOT an "upper bound".
# A larger base model evaluated zero-shot is not guaranteed to outperform a fine-tuned
# smaller model. We call it a baseline because that is what it is in the experiment.

# ---------------------------------------------------------------------------
# Built-in demo examples (no dataset download required)
# ---------------------------------------------------------------------------
DEMO_QA_EXAMPLES = [
    {
        "context": "The Eiffel Tower is located in Paris, France. It was built in 1889 as the entrance arch for the 1889 World's Fair.",
        "question": "Where is the Eiffel Tower located?",
        "answer": "Paris, France",
    },
    {
        "context": "Python is a high-level programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
        "question": "Who created Python?",
        "answer": "Guido van Rossum",
    },
    {
        "context": "The Amazon River is the largest river by discharge volume in the world. It flows through Brazil, Peru, and Colombia.",
        "question": "Which countries does the Amazon River flow through?",
        "answer": "Brazil, Peru, and Colombia",
    },
    {
        "context": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that adds trainable low-rank matrices to transformer layers.",
        "question": "What does LoRA stand for?",
        "answer": "Low-Rank Adaptation",
    },
    {
        "context": "The 2024 Paris Olympics were held in France during the summer. Over 10,000 athletes from around 200 countries participated.",
        "question": "How many athletes participated in the 2024 Paris Olympics?",
        "answer": "Over 10,000 athletes",
    },
]

DEMO_SUMMARIZATION_EXAMPLES = [
    {
        "text": (
            "Scientists have discovered a new species of deep-sea fish near the Mariana Trench. "
            "The fish, named Abyssia luminosa, produces bioluminescent light from organs along its body. "
            "Researchers believe this adaptation helps it attract prey in the darkness of the deep ocean. "
            "The discovery was published in the journal Nature on Tuesday."
        ),
        "summary": "Scientists discovered a new bioluminescent deep-sea fish near the Mariana Trench.",
    },
    {
        "text": (
            "Global electric vehicle sales reached a record 14 million units in 2023, according to the "
            "International Energy Agency. China accounted for 60% of all EV sales. Europe and the United "
            "States saw significant growth as well. Battery costs have fallen by 90% over the last decade, "
            "making EVs more affordable."
        ),
        "summary": "Global EV sales hit a record 14 million in 2023, led by China, as falling battery costs boost affordability.",
    },
    {
        "text": (
            "Researchers at MIT have developed a new battery technology that can charge a smartphone in "
            "under five minutes. The battery uses a solid-state electrolyte instead of a liquid one, "
            "which makes it safer and more energy-dense. The team expects commercial products within three years."
        ),
        "summary": "MIT researchers developed a fast-charging solid-state battery that can charge a phone in under five minutes.",
    },
]

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_qa_prompt(context: str, question: str) -> str:
    return (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"### Answer:\n"
    )

def build_summarization_prompt(text: str) -> str:
    return (
        f"### Article:\n{text}\n\n"
        f"### Summary:\n"
    )

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, prompts: list[str], max_new_tokens: int = 64,
                  device: str = "cpu") -> tuple[list[str], float]:
    """
    Run greedy inference on a list of prompts.
    Returns (list_of_predictions, total_latency_ms).
    """
    import torch
    model.eval()
    all_preds = []
    t_start = time.perf_counter()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        all_preds.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    total_ms = (time.perf_counter() - t_start) * 1000
    return all_preds, total_ms

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device,
    )
    model.config.use_cache = False
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded %s — %.0f M parameters", model_name, params_m)
    return model, tokenizer, params_m

# ---------------------------------------------------------------------------
# LoRA fine-tuning helper
# ---------------------------------------------------------------------------

def attach_and_finetune_lora(base_model, tokenizer, full_texts: list[str],
                              lora_steps: int, device: str):
    """
    Attach a LoRA adapter to base_model and run lora_steps training steps.
    Returns the PEFT-wrapped model and training time in ms.
    """
    import torch
    import torch.optim as optim
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA attached — trainable params: %.1f K / %.0f M  (%.3f%%)",
        trainable / 1e3, total / 1e6, 100 * trainable / total,
    )

    optimizer = optim.AdamW(peft_model.parameters(), lr=2e-4)
    peft_model.train()
    t_start = time.perf_counter()

    for step in range(lora_steps):
        step_loss = 0.0
        for full_text in full_texts:
            inputs = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=256
            ).to(device)
            labels = inputs["input_ids"].clone()
            outputs = peft_model(**inputs, labels=labels)
            outputs.loss.backward()
            step_loss += outputs.loss.item()
        optimizer.step()
        optimizer.zero_grad()
        logger.info(
            "LoRA step %d/%d | avg loss: %.4f", step + 1, lora_steps,
            step_loss / len(full_texts),
        )

    train_ms = (time.perf_counter() - t_start) * 1000
    logger.info("LoRA training complete in %.0f ms", train_ms)
    return peft_model, train_ms

# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------

def _print_table(task, slm_name, llm_name,
                 base_m, peft_m, llm_m,
                 base_lat, peft_lat, llm_lat):
    w = 76
    print("\n" + "=" * w)
    print(f"  3-WAY COMPARISON  |  Task: {task.upper()}")
    print(f"  SLM: {slm_name}   Larger-model baseline: {llm_name}")
    print("=" * w)

    if task == "qa":
        print(f"{'Model / Method':<36} {'EM':>8} {'F1':>8} {'Latency(ms)':>13} {'Params':>9}")
        print("-" * w)
        print(f"{'SLM — base (zero-shot)':<36} {base_m['exact_match']:>8.1f} {base_m['f1']:>8.1f} {base_lat:>13.0f} {'0.5B':>9}")
        print(f"{'SLM + LoRA (PEFT-tuned)':<36} {peft_m['exact_match']:>8.1f} {peft_m['f1']:>8.1f} {peft_lat:>13.0f} {'0.5B':>9}")
        print(f"{'LLM baseline — Qwen1.5-7B (zero-shot)':<36} {llm_m['exact_match']:>8.1f} {llm_m['f1']:>8.1f} {llm_lat:>13.0f} {'7B':>9}")
    else:
        print(f"{'Model / Method':<36} {'R-1':>7} {'R-2':>7} {'R-L':>7} {'Latency(ms)':>13} {'Params':>9}")
        print("-" * w)
        print(f"{'SLM — base (zero-shot)':<36} {base_m['rouge1']:>7.1f} {base_m['rouge2']:>7.1f} {base_m['rougeL']:>7.1f} {base_lat:>13.0f} {'0.5B':>9}")
        print(f"{'SLM + LoRA (PEFT-tuned)':<36} {peft_m['rouge1']:>7.1f} {peft_m['rouge2']:>7.1f} {peft_m['rougeL']:>7.1f} {peft_lat:>13.0f} {'0.5B':>9}")
        print(f"{'LLM baseline — Qwen1.5-7B (zero-shot)':<36} {llm_m['rouge1']:>7.1f} {llm_m['rouge2']:>7.1f} {llm_m['rougeL']:>7.1f} {llm_lat:>13.0f} {'7B':>9}")

    print("=" * w)
    print("  ↑ Key result: PEFT-SLM vs SLM-base (gain from LoRA) and vs LLM baseline (efficiency gap).")
    print("  NOTE: LLM is a larger-model baseline, not a guaranteed upper bound.")
    print("=" * w)


def _print_predictions(examples, base_preds, peft_preds, llm_preds):
    print("\nPer-example predictions:")
    for i, ex in enumerate(examples):
        ref = ex.get("answer") or ex.get("summary")
        print(f"\n  [{i+1}] Reference     : {ref}")
        print(f"       Base SLM     : {base_preds[i][:110]}")
        print(f"       PEFT-SLM     : {peft_preds[i][:110]}")
        print(f"       LLM (7B)     : {llm_preds[i][:110]}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PEFT-SLM 3-way comparison demo")
    parser.add_argument("--task", default="qa", choices=["qa", "summarization"])
    parser.add_argument("--lora_steps", type=int, default=5,
                        help="LoRA fine-tuning steps (use 50+ for real training)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM step — use this on CPU or low-VRAM machines")
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s  |  Task: %s  |  LoRA steps: %d", device, args.task, args.lora_steps)

    # ------------------------------------------------------------------
    # Prepare examples and prompts
    # ------------------------------------------------------------------
    if args.task == "qa":
        examples = DEMO_QA_EXAMPLES[:args.num_examples]
        prompts  = [build_qa_prompt(e["context"], e["question"]) for e in examples]
        answers  = [e["answer"] for e in examples]
    else:
        examples = DEMO_SUMMARIZATION_EXAMPLES[:args.num_examples]
        prompts  = [build_summarization_prompt(e["text"]) for e in examples]
        answers  = [e["summary"] for e in examples]

    def compute_metrics(preds):
        if args.task == "qa":
            return compute_qa_metrics(preds, answers)
        return compute_rouge(preds, answers)

    # ------------------------------------------------------------------
    # STEP 1 — Load SLM (Qwen1.5-0.5B)
    # ------------------------------------------------------------------
    slm_model, slm_tokenizer, slm_params = load_model_and_tokenizer(SLM_MODEL, device)

    # ------------------------------------------------------------------
    # STEP 2 — Base SLM zero-shot inference
    # ------------------------------------------------------------------
    logger.info("=== STEP 2: Base SLM (zero-shot) ===")
    base_preds, base_lat = run_inference(slm_model, slm_tokenizer, prompts, device=device)
    base_metrics = compute_metrics(base_preds)
    logger.info("Base SLM metrics: %s", base_metrics)

    # ------------------------------------------------------------------
    # STEP 3 — Attach LoRA + fine-tune  (PEFT)
    # ------------------------------------------------------------------
    logger.info("=== STEP 3: PEFT fine-tuning (LoRA) ===")
    full_texts = [p + a + slm_tokenizer.eos_token for p, a in zip(prompts, answers)]
    peft_model, train_ms = attach_and_finetune_lora(
        slm_model, slm_tokenizer, full_texts, args.lora_steps, device
    )

    # ------------------------------------------------------------------
    # STEP 4 — PEFT-SLM inference  ← your research contribution
    # ------------------------------------------------------------------
    logger.info("=== STEP 4: PEFT-SLM inference ===")
    peft_preds, peft_lat = run_inference(peft_model, slm_tokenizer, prompts, device=device)
    peft_metrics = compute_metrics(peft_preds)
    logger.info("PEFT-SLM metrics: %s", peft_metrics)

    # ------------------------------------------------------------------
    # STEP 5 & 6 — LLM baseline (Qwen1.5-7B, zero-shot)
    # ------------------------------------------------------------------
    llm_preds = ["[skipped — run without --skip_llm]"] * len(examples)
    llm_lat   = 0.0
    llm_metrics = {k: 0.0 for k in peft_metrics}

    if not args.skip_llm:
        logger.info("=== STEP 5: Loading LLM baseline (%s) ===", LLM_MODEL)
        try:
            llm_model, llm_tokenizer, _ = load_model_and_tokenizer(LLM_MODEL, device)
            logger.info("=== STEP 6: LLM inference (zero-shot) ===")
            llm_preds, llm_lat = run_inference(llm_model, llm_tokenizer, prompts, device=device)
            llm_metrics = compute_metrics(llm_preds)
            logger.info("LLM metrics: %s", llm_metrics)
            # Free GPU memory before printing
            del llm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Could not load LLM (%s): %s", LLM_MODEL, exc)
            logger.warning("Re-run with --skip_llm to suppress this, or ensure enough VRAM.")
    else:
        logger.info("LLM step skipped (--skip_llm).")

    # ------------------------------------------------------------------
    # STEP 7 — 3-way comparison table
    # ------------------------------------------------------------------
    _print_table(
        args.task, SLM_MODEL, LLM_MODEL,
        base_metrics, peft_metrics, llm_metrics,
        base_lat, peft_lat, llm_lat,
    )
    _print_predictions(examples, base_preds, peft_preds, llm_preds)

    # ------------------------------------------------------------------
    # STEP 8 — Log all 3 runs to experiment tracker
    # ------------------------------------------------------------------
    tracker = ExperimentTracker("experiments/results.csv")
    dataset_name = f"demo_{args.task}"

    def _base_row(method, model, metrics, latency):
        row = {
            "model": model,
            "dataset": dataset_name,
            "task": args.task,
            "method": method,
            "test_samples": len(examples),
            "avg_latency_ms": round(latency / len(examples), 1),
        }
        row.update({k: v for k, v in metrics.items() if k != "num_examples"})
        return row

    tracker.log({"experiment": "demo_base_slm",
                 **_base_row("base_SLM", SLM_MODEL, base_metrics, base_lat)})
    tracker.log({"experiment": "demo_peft_slm",
                 "train_time_s": round(train_ms / 1000, 1),
                 **_base_row("PEFT_LoRA", SLM_MODEL, peft_metrics, peft_lat)})
    if not args.skip_llm:
        tracker.log({"experiment": "demo_llm_baseline",
                     **_base_row("LLM_baseline", LLM_MODEL, llm_metrics, llm_lat)})

    logger.info("All 3 runs logged to experiments/results.csv")
    print("\n  Full experiment: python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml")
    print("  Summarization : python demo_quick.py --task summarization")
    print("  CPU/low VRAM  : python demo_quick.py --skip_llm\n")


if __name__ == "__main__":
    main()

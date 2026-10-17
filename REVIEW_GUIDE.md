# Review Session Guide
## M.Tech Project — PEFT-SLM Document Analysis

**Title:** A Study on Parameter-Efficient Fine-Tuning of Small Language Models for Document Analysis

---

## What this project is about (1-minute pitch)

Large Language Models can handle document tasks but are expensive to run. This
project asks: *can a small model — adapted using Parameter-Efficient Fine-Tuning
(LoRA) — match a large model on document tasks at a fraction of the cost?*

We compare **three setups on the same test examples**:

| Model | Method | Parameters | Role |
|---|---|---|---|
| Qwen1.5-0.5B | Zero-shot (base) | 500 M | Baseline — before any tuning |
| Qwen1.5-0.5B + LoRA | PEFT fine-tuned | 500 M + 0.169% adapter | **Your contribution** |
| Qwen1.5-7B | Zero-shot (base) | 7 B | Upper-bound reference |

The key question for each result: *Does the LoRA-adapted small model get closer
to the 7B model than the untuned small model did?*

---

## Quick-start (before the review)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify environment (no GPU needed — pure Python + torch)
python validate_setup.py

# 3. Run smoke test (CPU, ~2-3 min — verifies the full pipeline)
python smoke_test.py

# 4. Run the 3-way review demo
python demo_quick.py --skip_llm        # CPU / low-VRAM machines
python demo_quick.py                   # GPU machine (loads Qwen1.5-7B too)

# 5. Open the interactive notebook
jupyter notebook notebooks/01_review_demo.ipynb
```

---

## What to show the guide — step by step

### Step 1 — Pipeline diagram (Architecture slide)

Open `docs/architecture.md` or the diagram in `notebooks/01_review_demo.ipynb`
(Cell 2). Point out:

- Native PDFs → direct text extraction (pdfplumber / python-docx)
- Scanned PDFs → per-page OCR (Tesseract) with quality flag
- Both paths produce clean text tagged with source type
- That text feeds into the PEFT fine-tuning loop

### Step 2 — Dataset inspection

In the notebook (Cells 3-4) or from Python:

```python
from datasets import load_dataset

# SQuAD v2 — QA task
squad = load_dataset("squad_v2", split="validation[:3]")
# XL-Sum — summarization
xlsum = load_dataset("csebuetnlp/xlsum", "english", split="validation[:3]")
```

Point out:
- SQuAD v2 has **answerable** and **unanswerable** questions (~50/50)
- XL-Sum provides official train/validation/test splits
- Both are text-native (no OCR needed for Phase 1)

### Step 3 — Code walkthrough (5 minutes)

Key files to show:

| File | What it does |
|---|---|
| [`src/data/dataset_loader.py`](src/data/dataset_loader.py) | Loads datasets, enforces split boundaries, saves test indices |
| [`src/data/preprocessor.py`](src/data/preprocessor.py) | Masking logic — prompt tokens are -100, completion tokens train |
| [`src/training/model_loader.py`](src/training/model_loader.py) | Attaches LoRA, validates target modules, asserts base is frozen |
| [`src/evaluation/metrics.py`](src/evaluation/metrics.py) | EM, F1 with answerable/unanswerable breakdown; ROUGE |

### Step 4 — Run the demo

```bash
python demo_quick.py --task qa --lora_steps 5 --skip_llm
```

This shows:
1. Qwen1.5-0.5B loaded (500 M params)
2. Zero-shot inference on 5 QA examples
3. LoRA attached (trainable params: 0.169% of total)
4. 5-step fine-tuning (loss decreases)
5. Post-tuning inference on same examples
6. Side-by-side comparison table

Expected output (approximate):
```
======================================================================
  3-WAY COMPARISON  |  Task: QA
======================================================================
Model / Method                          EM       F1  Latency(ms)  Params
----------------------------------------------------------------------
SLM — base (zero-shot)                 0.0     xx.x          xxx    0.5B
SLM + LoRA (PEFT-tuned)               xx.x     xx.x          xxx    0.5B
LLM — Qwen1.5-7B (zero-shot)           ---      ---            0      7B
======================================================================
```

Note: with only 5 LoRA steps on 5 demo examples, the improvement will be
visible but not dramatic. For full evaluation, the pilot trains 3 epochs on
5,000 SQuAD examples.

### Step 5 — Show the experiment tracker

```python
from src.utils.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker("experiments/demo_quick/results.csv")  # demo runs
tracker.print_summary()                                           # real runs: experiments/results.csv
```

Every run is logged here automatically with model, method, metrics, latency,
GPU memory, and seed — so all results are reproducible and auditable.

### Step 6 — Research contribution (talk track)

Emphasise:

1. **Fair comparison — per task**: for each task separately, all three models
   are evaluated on the **same held-out test examples**, using the same input
   format and evaluation procedure. SQuAD v2 (QA) and XL-Sum (summarization)
   are different datasets — the same 500 examples are not shared across tasks.
   "Identical" means: same 500 SQuAD examples for all three models on QA, and
   same 500 XL-Sum examples for all three models on summarization.
   Enforced by `seed: 42` in every YAML config.

2. **Same model variant**: all three use `model_variant: "base"` — not mixing
   base and instruct variants, which would make the zero-shot numbers unfair.

3. **Answerable / unanswerable breakdown**: we report metrics separately to
   prevent inflation from models that output "unanswerable" for everything.

4. **Efficiency tracking**: we record training time, peak GPU memory, and
   inference latency — the efficiency gain is a research result, not an
   assumption.

5. **Documented limitations**: FUNSD extraction is deferred to Phase 2 (pipeline
   available but experiment not yet run); base models zero-shot may not follow
   task instructions well — both are explicitly stated in the code and results.

---

## Running the full pilot (GPU required)

```bash
# Step 1 — Base SLM baseline (zero-shot)
python src/training/run_pilot.py \
    --config configs/pilot_squad_qa.yaml \
    --base_only

# Step 2 — PEFT fine-tuning (3 epochs, ~5000 training examples)
python src/training/run_pilot.py \
    --config configs/pilot_squad_qa.yaml

# Step 3 — LLM zero-shot baseline (needs ~14 GB VRAM or use load_in_bits: 4)
python src/training/run_pilot.py \
    --config configs/llm_baseline_qwen7b.yaml \
    --base_only

# Step 4 — Compare results
python -c "
from src.utils.experiment_tracker import ExperimentTracker
ExperimentTracker('experiments/results.csv').print_summary()
"

# Step 5 — Summarization pilot
python src/training/run_pilot.py \
    --config configs/pilot_xlsum_summarization.yaml
```

---

## Hardware requirements

| Machine | What you can run |
|---|---|
| CPU only (no GPU) | `validate_setup.py`, `smoke_test.py`, `demo_quick.py --skip_llm` |
| 6–8 GB VRAM | Full SLM fine-tuning (set `load_in_bits: 4` in config for QLoRA) |
| 16 GB VRAM | Full SLM + LLM baseline without quantization |
| 24+ GB VRAM | Everything including LLM at bfloat16 precision |

If running on a CPU-only machine for the review demo, use:
```bash
python demo_quick.py --skip_llm --lora_steps 3 --num_examples 3
```

---

## Frequently asked questions

**Q: Why Qwen1.5-0.5B?**  
A: Smallest model in the Qwen1.5 family — fast iteration, runs on CPU for demos.
   The pipeline is model-agnostic; swap `model_name_or_path` in the config to
   use Phi-2, Gemma-2B, or any other HuggingFace causal LM.

**Q: Why not use Qwen1.5-0.5B-Chat (instruct) instead?**  
A: Mixing base and instruct variants in a 3-way comparison is unfair. An instruct
   model has already been instruction-tuned, making its zero-shot numbers higher
   for reasons unrelated to our PEFT contribution. Using the base variant for all
   three makes the improvement from LoRA directly attributable to our training.

**Q: What does "identical test examples" mean — does it span tasks?**
A: No. For each task separately, all three models are evaluated on the same
   held-out examples. SQuAD v2 QA examples and XL-Sum summarization examples
   are different datasets — the same 500 examples cannot be shared across tasks.
   "Identical" means same 500 SQuAD examples for all three models on QA, and
   same 500 XL-Sum examples for all three models on summarization. Per-task
   consistency is enforced by `seed: 42` in every YAML config.

**Q: SQuAD v2 has no test split — how do you avoid data leakage?**
A: Both our validation and test subsets come from the HF `validation` split, but
   they are drawn together without replacement using `numpy.random.default_rng(42)`,
   so they are guaranteed disjoint. A runtime assertion verifies zero overlap.
   The validation subset is used only for checkpoint selection (not as test).
   The test subset is held out until the final evaluation.
   The test indices are saved to `experiments/*/test_indices.json` for audit.

**Q: Why do the unanswerable metrics matter?**
A: A naive model could output "unanswerable" for every question and get 100%
   unanswerable EM while getting 0% on answerable questions. Reporting both
   breakdown groups separately makes this visible and prevents misleading averages.

**Q: What is Phase 2?**
A: Phase 2 (Semester 4) extends to:
   - FUNSD information extraction (pipeline available; experiment deferred to Phase 2)
   - RVL-CDIP classification (requires OCR pre-extraction)
   - DocVQA document QA (image-based, requires OCR)
   - Error analysis and thesis writing
   Note: XL-Sum summarization is Phase 1 (second pilot, after SQuAD v2).

---

## Repository map (quick reference)

```
Mtech/
├── REVIEW_GUIDE.md              ← this file
├── README.md                    ← project overview
├── requirements.txt             ← all dependencies
├── validate_setup.py            ← run first — pure Python checks
├── smoke_test.py                ← run second — 9-check pipeline verifier (CPU)
├── demo_quick.py                ← run for review — 3-way comparison
│
├── configs/
│   ├── pilot_squad_qa.yaml      ← SLM + LoRA, SQuAD QA, seed=42
│   ├── pilot_xlsum_summarization.yaml
│   └── llm_baseline_qwen7b.yaml ← 7B zero-shot, seed=42 (identical test set)
│
├── src/
│   ├── extraction/text_extractor.py     ← PDF/OCR pipeline
│   ├── data/dataset_loader.py           ← HF split boundaries + disjoint splits
│   ├── data/preprocessor.py             ← prompt masking (prompt=-100)
│   ├── training/model_loader.py         ← LoRA attach + validation
│   ├── training/trainer.py              ← HF Trainer loop
│   ├── training/run_pilot.py            ← CLI entry point
│   ├── evaluation/metrics.py            ← EM/F1/ROUGE + unanswerable breakdown
│   ├── evaluation/evaluator.py          ← standalone inference loop
│   └── utils/experiment_tracker.py     ← CSV logger
│
├── notebooks/01_review_demo.ipynb       ← interactive 3-way demo
├── notebooks/simple_pipeline.ipynb      ← full 3-way experiment on Colab T4
├── docs/
│   ├── architecture.md                  ← pipeline architecture reference
│   ├── literature_notes.md              ← 6 papers annotated
│   └── dataset_notes.md                 ← 5 datasets documented
└── experiments/
    └── results.csv                      ← all run logs (auto-populated)
```

# Project Progress Tracker
## M.Tech: Parameter-Efficient Fine-Tuning of Small Language Models for Document Analysis

> **How to use this file:**
> Work through the steps in order. Each step has a clear goal, the exact commands to run,
> and a "Done when" checklist so you know when to move to the next step.
> Mark steps as you complete them: change `[ ]` to `[x]`.

---

## Legend
- `[x]` Done
- `[ ]` Not started
- `[-]` In progress
- `[~]` Blocked (needs something external)

---

## PHASE 0 — Setup (No GPU needed)

These steps run entirely on CPU. Do these before getting GPU access.

---

### Step 0.1 — Verify your Python environment

**Goal:** Confirm Python and the project structure are working correctly.

```bash
python validate_setup.py
```

**Done when:**
- [x] All lines print `[OK]` or `[SKIP]`
- [x] Config loads correctly (`squad_qa_lora_pilot`)
- [x] ExperimentTracker writes `experiments/results.csv`
- [x] Text extractor returns `source=plain`

**Notes:** `[SKIP]` lines are fine — they need `pip install -r requirements.txt` first.

---

### Step 0.2 — Install dependencies

**Goal:** Install all Python packages needed for training.

```bash
pip install -r requirements.txt
```

**Done when:**
- [ ] `import torch` works
- [ ] `import transformers` works
- [ ] `import peft` works
- [ ] `import datasets` works
- [ ] `python validate_setup.py` shows no `[SKIP]` lines

**Notes:** Takes ~5 minutes. On Google Colab, run this as the first cell.

---

### Step 0.3 — Run the CPU demo (understand the project)

**Goal:** See the full pipeline working without any model download.

```bash
python run_demo_cpu.py --skip_model
```

**Done when:**
- [ ] Text extraction demo prints `[OK]`
- [ ] Prompt formatting demo prints `[OK]` with `### Context / ### Question / ### Answer`
- [ ] Experiment tracker writes `experiments/cpu_demo_results.csv`

**Notes:** This runs in ~5 seconds. It shows the pipeline logic without loading any model.

---

### Step 0.4 — Download datasets and build task prompts (CPU)

**Goal:** Pull SQuAD v2 and XL-Sum (English) from the Hugging Face Hub and apply the prompt transformations.

```bash
python scripts/prepare_data.py --config configs/pilot_squad_qa.yaml
python scripts/prepare_data.py --config configs/pilot_xlsum_summarization.yaml
```

**Done when:**
- [x] SQuAD v2 (`rajpurkar/squad_v2`): train 4997 / val 500 / test 500; val and test are disjoint, with 232 answerable + 268 unanswerable in val and 270 + 230 in test
- [x] XL-Sum English (`csebuetnlp/xlsum`, via the Hub parquet branch): train 5000 / val 500 / test 500 from the official splits
- [x] Output written to `data/processed/{squad_v2,xlsum_en}/{train,validation,test}.jsonl` (gitignored)
- [x] `smoke_test.py` passes all checks

**Transformations (in `src/data/prompts.py`):**
- Whitespace cleaning
- Context or article trimmed by tokens, so the answer header, answer and EOS are never truncated away
- EOS appended to every training target
- Training rows whose answer was cut out of the context are dropped (3 of 5000)
- Empty SQuAD answers become `unanswerable`

---

## PHASE 1A — GPU Smoke Test (First GPU use)

Do these the first time you have GPU access. They verify the machine is set up correctly.

---

### Step 1.1 — Confirm GPU is available

**Goal:** Know which GPU you have and how much VRAM it has.

```python
import torch
print(torch.cuda.get_device_name(0))
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram_gb:.1f} GB")
```

**Done when:**
- [ ] GPU name is printed (e.g. "Tesla T4", "NVIDIA A100")
- [ ] VRAM is printed (need at least 6 GB for SLM training)

**Notes:**
- T4 (15 GB) — fine for everything
- 6–8 GB — use `load_in_bits: 4` in config for QLoRA
- CPU only — can only run `--skip_llm` and `--skip_model` flags

---

### Step 1.2 — Run the full smoke test

**Goal:** Verify that every component of the pipeline works correctly on this GPU.

```bash
python smoke_test.py
```

**Done when:**
- [x] All 9 checks print `[OK]`
- [x] Check [2]: dataset splits are disjoint, unanswerable examples present
- [x] Check [4,5,6]: prompt masking is correct (prompt=-100, completion=trained)
- [x] Check [7]: base model inference runs and returns string predictions
- [x] Check [8]: unanswerable breakdown metrics are correct
- [x] Final line: `ALL SMOKE TEST CHECKS PASSED`

**Notes:** Ran on CPU (no GPU required). Results logged to `experiments/smoke_test/results.csv`:
Qwen1.5-0.5B base zero-shot on SQuAD v2 — EM=20.0, F1=33.5, Answerable EM=25.0, Unanswerable EM=15.0.

---

### Step 1.3 — Run the 3-way demo (see the full comparison)

**Goal:** Watch all three setups run and produce a comparison table.

```bash
# With GPU (recommended):
python demo_quick.py --lora_steps 10

# Without LLM (low VRAM or CPU):
python demo_quick.py --skip_llm --lora_steps 10
```

**Done when:**
- [ ] Base SLM inference completes (prints predictions)
- [ ] LoRA training runs for 10 steps (loss should decrease each step)
- [ ] PEFT-SLM inference completes
- [ ] Comparison table is printed
- [ ] Results written to `experiments/results.csv`

**Notes:** `--lora_steps 10` is just a demo. Real training is 3 full epochs (~hundreds of steps).

---

## PHASE 1B — SQuAD v2 QA Experiment (Core Phase 1 results)

This is the main experiment. It produces the 3-way comparison table for the Phase 1 report.

---

### Step 2.1 — Measure the actual LoRA parameter percentage

**Goal:** Replace the "~0.4%" estimate in all documents with the real measured value.

This happens automatically when training starts. Watch for this output:
```
┌─────────────────────────────────────────────────────────────┐
│  LoRA Parameter Report                                      │
│  Total model parameters    :    xxx,xxx,xxx  (0.xxx B)     │
│  Trainable (LoRA only)     :      x,xxx,xxx  (x.xxx%)      │
│  Frozen (base model)       :    xxx,xxx,xxx  (xx.xxx%)     │
│  LoRA rank                 : 8                             │
│  Target modules            : q_proj, v_proj               │
└─────────────────────────────────────────────────────────────┘
```

**Done when:**
- [x] The measured trainable % is recorded here: **0.169 %** (786,432 / 464,774,144 total params)
- [ ] This number is used in the report (replace "~0.4%")

---

### Step 2.2 — Run the base SLM baseline (zero-shot)

**Goal:** Record what the untuned 0.5B model scores on SQuAD v2 — the starting point.

```bash
python src/training/run_pilot.py \
    --config configs/pilot_squad_qa.yaml \
    --base_only
```

**Time estimate:** ~15–20 minutes on a T4 (inference only, no training)

**Done when:**
- [x] Run completes without error
- [x] Results logged to `experiments/results.csv` with `method=base_zero_shot`
- [x] Record these numbers:

| Metric | Value |
|---|---|
| Exact Match (overall) | **confirmed via Cell 6A on Colab T4** |
| F1 (overall) | **confirmed via Cell 6A on Colab T4** |
| Answerable EM | — |
| Unanswerable EM | — |
| Inference latency (ms/example) | avg_latency_ms logged |

**Notes:**
- Cell 6A completed successfully on Google Colab T4 (~5–6 min)
- Base model zero-shot scores are low — expected and documented
- This is Row 1 of your results table

---

### Step 2.3 — Run the full LoRA fine-tuning

**Goal:** Train the PEFT-adapted SLM — this is your research contribution.

```bash
python src/training/run_pilot.py \
    --config configs/pilot_squad_qa.yaml
```

**Time estimate:** ~1–2 hours on a T4 (3 epochs × 5000 examples)

**Watch for during training:**
```
Step 50  | loss: x.xxxx   ← should decrease over time
Step 100 | loss: x.xxxx
Step 200 | eval_f1: xx.xx ← should be higher than baseline
```

**Done when:**
- [-] Training completes 3 epochs without OOM — **IN PROGRESS: OOM being fixed**
- [ ] Loss decreased from start to end
- [ ] Best checkpoint saved to `experiments/squad_qa/adapter/`
- [ ] Test metrics logged to `experiments/results.csv`
- [ ] Record these numbers:

| Metric | Value |
|---|---|
| Exact Match (overall) | ___._% |
| F1 (overall) | ___._% |
| Answerable EM | ___._% |
| Unanswerable EM | ___._% |
| Training time | ___ minutes |
| Peak GPU memory | ___ MB |
| Trainable params | 786 K (0.169%) |

**OOM fixes applied (iterative, Colab T4 15 GB VRAM):**
- Attempt 1: `batch_size 4→2`, `grad_accum 4→8`, `fp16=true`, `gradient_checkpointing=True` — still OOM
- Attempt 2: `max_source_length 512→256`, `eval_batch 4→2`, `eval_steps 200→500`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — in progress

---

### Step 2.4 — Run the larger-model baseline

**Goal:** Record what the 7B model scores zero-shot on the same 500 test examples.

```bash
python src/training/run_pilot.py \
    --config configs/llm_baseline_qwen7b.yaml \
    --base_only
```

**Time estimate:** ~20–30 minutes on a T4 (inference only)

**If you get OOM error:** Edit `configs/llm_baseline_qwen7b.yaml` and set `load_in_bits: 4`

**Done when:**
- [ ] Run completes without error
- [ ] Results logged to `experiments/results.csv` with `method=base_zero_shot`
- [ ] Model is `Qwen/Qwen1.5-7B`
- [ ] Record these numbers:

| Metric | Value |
|---|---|
| Exact Match (overall) | ___._% |
| F1 (overall) | ___._% |
| Answerable EM | ___._% |
| Unanswerable EM | ___._% |
| Inference latency (ms/example) | ___._ ms |

---

### Step 2.5 — Compile the 3-way QA comparison table

**Goal:** Produce the central result table for your Phase 1 report.

```bash
python -c "
from src.utils.experiment_tracker import ExperimentTracker
ExperimentTracker('experiments/results.csv').print_summary()
"
```

**Done when:**
- [ ] All three rows are in `results.csv`
- [ ] Fill in this table from the logged results:

```
3-WAY COMPARISON — SQuAD v2 QA — 500 test examples — seed=42
================================================================
Model / Method               EM      F1    Lat(ms)  Params
----------------------------------------------------------------
SLM base (zero-shot)        ??.?   ??.?     ???      0.5B
SLM + LoRA (PEFT) ← OURS   ??.?   ??.?     ???      0.5B
LLM baseline (zero-shot)    ??.?   ??.?     ???       7B
================================================================
```

- [ ] LoRA model improves over base SLM (EM and F1 both higher)
- [ ] Efficiency gap recorded (latency and GPU memory differences noted)

---

## PHASE 1C — XL-Sum Summarization Experiment (Second pilot)

Run this after the SQuAD experiment is complete and the pipeline is confirmed working.

---

### Step 3.1 — Run the base SLM summarization baseline

```bash
python src/training/run_pilot.py \
    --config configs/pilot_xlsum_summarization.yaml \
    --base_only
```

**Done when:**
- [ ] Completes without error
- [ ] Results logged with `dataset=csebuetnlp/xlsum`, `method=base_zero_shot`
- [ ] ROUGE scores recorded: R1=___._ R2=___._ RL=___._

---

### Step 3.2 — Run the XL-Sum LoRA fine-tuning

```bash
python src/training/run_pilot.py \
    --config configs/pilot_xlsum_summarization.yaml
```

**Time estimate:** ~1.5–2.5 hours on a T4

**Done when:**
- [ ] Training completes 3 epochs
- [ ] Checkpoint saved to `experiments/xlsum_summarization/adapter/`
- [ ] ROUGE scores recorded: R1=___._ R2=___._ RL=___._

---

### Step 3.3 — Run the LLM summarization baseline

```bash
# Re-use the same LLM config but override the task
# Or create a new config — the pipeline supports both
python src/training/run_pilot.py \
    --config configs/llm_baseline_qwen7b.yaml \
    --base_only
```

**Note:** You may need to create `configs/llm_baseline_xlsum.yaml` with `task: summarization`.

**Done when:**
- [ ] LLM summarization baseline results are logged
- [ ] 3-way ROUGE comparison table is compiled

---

## PHASE 1D — Analysis and Report

---

### Step 4.1 — Analyse the QA results

**Goal:** Draw conclusions from the 3-way table. Answer these questions in writing:

- [ ] By how much did LoRA improve Exact Match over the base SLM?
- [ ] What is the gap between PEFT-SLM and the LLM baseline?
- [ ] Is the LoRA model better on answerable questions, unanswerable, or both?
- [ ] How much faster is the 0.5B model vs the 7B model at inference?
- [ ] How much less GPU memory did LoRA training use vs estimated full fine-tuning?

---

### Step 4.2 — Error analysis (manual)

**Goal:** Look at examples where the model was wrong and understand why.

```bash
# After running evaluation, inspect the predictions directly
# The evaluator stores decoded predictions — add a --save_predictions flag
# or inspect the output from demo_quick.py for individual examples
```

Look for patterns:
- [ ] Are there common types of questions the model gets wrong?
- [ ] Does the model sometimes generate too long or too short answers?
- [ ] For unanswerable questions, does the model still try to answer?

---

### Step 4.3 — Write Phase 1 report section

Using the completed tables from Steps 2.5 and 3.3, write:

- [ ] Experimental setup (models, datasets, training config, seed)
- [ ] Results table — 3-way QA comparison
- [ ] Results table — 3-way summarization comparison
- [ ] Efficiency table — training time, GPU memory, latency, trainable %
- [ ] Discussion — what the numbers mean
- [ ] Limitations — base model zero-shot weakness, text-only pipeline, SQuAD vs real enterprise docs

---

## PHASE 2 — Extended Experiments (Semester 4)

Start these only after Phase 1 results are complete and reported.

---

### Step 5.1 — FUNSD information extraction
- [ ] Set up FUNSD dataset loading (pipeline already built)
- [ ] Define extraction output format (field: value pairs)
- [ ] Run base SLM extraction baseline
- [ ] Fine-tune with LoRA
- [ ] Manual spot-check of 10 extracted forms
- [ ] Record precision, recall, F1
- [ ] Note: spatial layout is lost in text-only mode — document this limitation

---

### Step 5.2 — RVL-CDIP document classification
- [ ] Pre-extract OCR text from document images (Tesseract pipeline exists)
- [ ] Prepare classification dataset with 16 class labels
- [ ] Run base SLM classification baseline
- [ ] Fine-tune with LoRA
- [ ] Record accuracy and Macro-F1

---

### Step 5.3 — DocVQA (if time permits)
- [ ] Use dataset's provided OCR text (or run Tesseract on images)
- [ ] Adapt QA pipeline for document image context
- [ ] Run baseline and fine-tuned comparison

---

### Step 5.4 — Comprehensive analysis
- [ ] Cross-task comparison: which tasks benefit most from PEFT?
- [ ] Efficiency analysis: performance-per-compute-unit across all tasks
- [ ] OCR quality impact: compare native text vs OCR-derived text results
- [ ] Hallucination spot-check: are model answers grounded in the source document?

---

### Step 5.5 — Thesis and paper
- [ ] Write full thesis chapters
- [ ] Prepare tables and figures from all experiment results
- [ ] Draft research paper (if applicable)
- [ ] Guide review and revision

---

## Quick Reference — Key Commands

```bash
# Environment check (no GPU)
python validate_setup.py

# Pipeline smoke test (CPU, ~5 min with model download)
python smoke_test.py

# CPU demo (instant, no model download)
python run_demo_cpu.py --skip_model

# Full 3-way demo (GPU recommended)
python demo_quick.py

# SQuAD QA — base SLM baseline
python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml --base_only

# SQuAD QA — LoRA fine-tuning (your contribution)
python src/training/run_pilot.py --config configs/pilot_squad_qa.yaml

# LLM baseline (needs ~14 GB VRAM or set load_in_bits: 4)
python src/training/run_pilot.py --config configs/llm_baseline_qwen7b.yaml --base_only

# XL-Sum summarization
python src/training/run_pilot.py --config configs/pilot_xlsum_summarization.yaml

# View all results
python -c "from src.utils.experiment_tracker import ExperimentTracker; ExperimentTracker('experiments/results.csv').print_summary()"
```

---

## Results Log (fill in as you go)

### SQuAD v2 — Question Answering

| Model | Method | EM | F1 | Ans EM | Unans EM | Lat ms | GPU MB | Train min |
|---|---|---|---|---|---|---|---|---|
| Qwen1.5-0.5B | base_zero_shot | ✅ logged | ✅ logged | — | — | logged | — | — |
| Qwen1.5-0.5B | LoRA (ours) | pending | pending | pending | pending | pending | pending | pending |
| Qwen1.5-7B | base_zero_shot | pending | pending | — | — | pending | — | — |

### XL-Sum — Summarization

| Model | Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Lat ms | GPU MB | Train min |
|---|---|---|---|---|---|---|---|---|
| Qwen1.5-0.5B | base_zero_shot | | | | | | — |
| Qwen1.5-0.5B | LoRA (ours) | | | | | | |
| Qwen1.5-7B | base_zero_shot | | | | | | — |

### LoRA Adapter Details

| Config | Total params | Trainable params | Trainable % |
|---|---|---|---|
| r=8, q_proj+v_proj | 464,774,144 | 786,432 | **0.169%** |

---

## Current Status

```
Phase 0 — Setup            [x] Complete (incl. 0.4 data preparation)
Phase 1A — Smoke Test      [x] Run on CPU — results logged (Qwen1.5-0.5B base zero-shot on SQuAD v2: EM=20.0, F1=33.5, Ans EM=25.0, Unans EM=15.0)
Phase 1B — SQuAD QA        [-] IN PROGRESS on Google Colab T4
                                [x] Cell 6A: Base SLM zero-shot baseline — COMPLETE
                                [-] Cell 6B: LoRA fine-tuning — OOM fixes being applied (attempt 2)
                                [ ] Cell 6C: LLM 7B baseline
                                [ ] Cell 6D: Results table
Phase 1C — XL-Sum Summary  [ ] Needs GPU (after 1B complete)
Phase 1D — Report          [ ] Needs results from 1B + 1C
Phase 2 — Extended         [ ] Semester 4
```

**Colab notebook history:**
- `05_colab_ready.ipynb` — initial build
- `06_colab_fixed.ipynb` — fix: removed `logging_dir`, `report_to='none'`
- `07_colab_fixed2.ipynb` — fix: `Trainer(tokenizer=)` → runtime-detected `processing_class`
- `08_colab_oom_fix.ipynb` — fix: batch 4→2, fp16, gradient_checkpointing
- `09_colab_oom_fix2.ipynb` — fix: seq_len 384→256, eval_batch 4→2, expandable_segments

**Immediate next action:**
> Upload `notebooks/09_colab_oom_fix2.ipynb` to Colab and run Cell 6B (LoRA fine-tuning).
> If OOM persists, run Cell 6B-OOM first (switches to 4-bit QLoRA) then re-run Cell 6B.

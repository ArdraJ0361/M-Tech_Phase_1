# M.Tech Project — PEFT-SLM Document Analysis

**Title:** A Study on Parameter-Efficient Fine-Tuning of Small Language Models for Document Analysis  
**Area:** Natural Language Processing · Small Language Models · Document Intelligence  
**Phase 1 scope:** Semester 3 — SQuAD QA + XL-Sum summarization (text-native tasks)

---

## Research Question

> Can PEFT-adapted Small Language Models achieve competitive performance on
> selected document-analysis tasks while offering measurable computational-efficiency
> advantages over larger models?

---

## The Core Experiment

Three models are compared on **identical test examples** (same seed, same task,
same prompt format):

| Model | Method | Parameters | Role |
|---|---|---|---|
| `Qwen/Qwen1.5-0.5B` | Base, zero-shot | 500 M | Untuned SLM baseline |
| `Qwen/Qwen1.5-0.5B` + LoRA | PEFT fine-tuned | 500 M + ~0.4% adapter | **Research contribution** |
| `Qwen/Qwen1.5-7B` | Base, zero-shot | 7 B | LLM upper-bound reference |

All three use `model_variant: "base"` and `seed: 42`.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate environment (no GPU needed)
python validate_setup.py

# 3. Run smoke test — 9-check pipeline verifier (CPU, ~2-3 min)
python smoke_test.py

# 4. Run the 3-way review demo
python demo_quick.py --skip_llm       # CPU / low-VRAM
python demo_quick.py                  # GPU (also loads 7B LLM)

# 5. Open the Jupyter notebook
jupyter notebook notebooks/01_review_demo.ipynb
```

---

## Full Pilot (GPU required)

```bash
# Base SLM zero-shot baseline
python src/training/run_pilot.py \
    --config configs/pilot_squad_qa.yaml --base_only

# PEFT fine-tuning (3 epochs, 5000 train examples)
python src/training/run_pilot.py \
    --config configs/pilot_squad_qa.yaml

# LLM zero-shot baseline (needs ~14 GB VRAM or set load_in_bits: 4)
python src/training/run_pilot.py \
    --config configs/llm_baseline_qwen7b.yaml --base_only

# Summarization pilot
python src/training/run_pilot.py \
    --config configs/pilot_xlsum_summarization.yaml

# View all results
python -c "
from src.utils.experiment_tracker import ExperimentTracker
ExperimentTracker('experiments/results.csv').print_summary()
"
```

---

## Repository Structure

```
Mtech/
├── REVIEW_GUIDE.md                 ← demo guide for the review session
├── README.md                       ← this file
├── requirements.txt
├── validate_setup.py               ← pure-Python import check (run first)
├── smoke_test.py                   ← 9-check pipeline verifier (CPU)
├── demo_quick.py                   ← 3-way comparison demo
│
├── configs/
│   ├── pilot_squad_qa.yaml         ← SLM + LoRA, SQuAD v2, seed=42
│   ├── pilot_xlsum_summarization.yaml
│   └── llm_baseline_qwen7b.yaml    ← 7B baseline, seed=42
│
├── src/
│   ├── extraction/
│   │   └── text_extractor.py       ← per-page PDF/OCR, quality flag, chunker
│   ├── data/
│   │   ├── dataset_loader.py       ← HF split boundaries, disjoint test/val
│   │   └── preprocessor.py         ← prompt masking, truncation safety
│   ├── training/
│   │   ├── model_loader.py         ← LoRA attach + validation
│   │   ├── trainer.py              ← HF Trainer loop
│   │   └── run_pilot.py            ← CLI entry point
│   ├── evaluation/
│   │   ├── metrics.py              ← EM/F1 + unanswerable breakdown, ROUGE
│   │   └── evaluator.py            ← standalone inference + latency
│   └── utils/
│       ├── config_loader.py
│       ├── experiment_tracker.py   ← CSV logger
│       └── logging_setup.py
│
├── notebooks/
│   └── 01_review_demo.ipynb        ← interactive 3-way demo with charts
│
├── docs/
│   ├── architecture.md             ← full system architecture reference
│   ├── literature_notes.md         ← 6 survey papers annotated
│   └── dataset_notes.md            ← 5 candidate datasets documented
│
└── experiments/
    └── results.csv                 ← all run logs (auto-populated)
```

---

## Datasets

| Dataset | Task | Input | Phase | Status |
|---|---|---|---|---|
| SQuAD v2 | Question Answering | Text passages | Phase 1 | ✅ Validated |
| XL-Sum (English) | Summarization | News articles | Phase 1 | ✅ Validated |
| FUNSD | Information Extraction | Scanned forms | Phase 1 partial | ⚠️ Layout lost |
| RVL-CDIP | Classification | Document images | Phase 2 | Requires OCR |
| DocVQA | Visual QA | Document images | Phase 2 | Requires OCR |

**SQuAD v2 note:** There is no public labelled test split. Both validation and
test subsets are drawn from the HF `validation` split using disjoint index
ranges (zero overlap verified at runtime). Test indices are saved to
`experiments/*/test_indices.json` for full audit.

---

## Models

| Model | Parameters | Default config |
|---|---|---|
| `Qwen/Qwen1.5-0.5B` | 500 M | `pilot_squad_qa.yaml` (SLM to tune) |
| `Qwen/Qwen1.5-7B` | 7 B | `llm_baseline_qwen7b.yaml` (LLM baseline) |
| `microsoft/phi-2` | 2.7 B | Change `model_name_or_path` in config |
| `google/gemma-2b` | 2 B | Requires HuggingFace access token |

---

## Hardware Requirements

| Setup | Minimum | Recommended |
|---|---|---|
| CPU demo only | 8 GB RAM | 16 GB RAM |
| SLM (0.5B) LoRA training | 6 GB VRAM | 8 GB VRAM |
| SLM (0.5B) QLoRA (4-bit) | 4 GB VRAM | 6 GB VRAM |
| LLM (7B) inference | 14 GB VRAM | 16 GB VRAM |
| LLM (7B) 4-bit inference | 6 GB VRAM | 8 GB VRAM |

To enable 4-bit quantization, set `load_in_bits: 4` in the YAML config.

---

## Evaluation Metrics

| Task | Primary metrics |
|---|---|
| QA | Exact Match, Token F1, answerable/unanswerable breakdown |
| Summarization | ROUGE-1, ROUGE-2, ROUGE-L |
| Classification | Accuracy, Macro-F1 |
| Extraction | Token Precision, Recall, F1 |
| All | Training time, peak GPU memory, inference latency, throughput |

---

## Phase 1 Milestones (Semester 3)

- [x] Literature survey — 6 papers annotated
- [x] Dataset investigation — 5 datasets documented
- [x] Repository scaffold and full pipeline code
- [x] Text extraction (PDF native + per-page OCR fallback)
- [x] Dataset loader with split integrity guarantees
- [x] PEFT trainer with LoRA validation and reproducibility logging
- [x] Evaluation metrics with unanswerable breakdown
- [x] 3-way demo script (`demo_quick.py`)
- [x] Smoke test — 9 end-to-end checks (`smoke_test.py`)
- [x] Jupyter review notebook (`01_review_demo.ipynb`)
- [ ] GPU baseline run — base SLM on 500 SQuAD examples
- [ ] Full LoRA fine-tuning — 3 epochs, 5000 examples
- [ ] LLM baseline run — Qwen1.5-7B on same 500 examples
- [ ] Phase 1 report and 3-way comparison table

## Phase 2 Milestones (Semester 4)

- [ ] XL-Sum summarization pilot
- [ ] FUNSD extraction evaluation + manual spot-check
- [ ] RVL-CDIP classification (OCR pre-extraction required)
- [ ] DocVQA with OCR pipeline
- [ ] Error analysis per task
- [ ] Efficiency analysis (performance per compute unit)
- [ ] Thesis writing and research paper

---

## Key Design Decisions

1. **Base not instruct**: All three comparison models use `model_variant: "base"`.
   Mixing base and instruct variants is not a fair comparison.

2. **Seed=42 everywhere**: Ensures all three models evaluate on the exact same
   500 test examples. Verified in `smoke_test.py`.

3. **Prompt masking**: Training loss is computed only on completion tokens
   (labels[prompt_len:] = actual ids). Prompt tokens are masked with -100.

4. **Skip truncated examples**: If truncation removes all answer tokens, the
   example is skipped entirely rather than training on prompt-only tokens.

5. **Unanswerable breakdown**: EM and F1 are reported separately for answerable
   and unanswerable questions to prevent score inflation.

---

## References

See [`docs/literature_notes.md`](docs/literature_notes.md) for annotated notes
on the six survey papers in the initial literature review.

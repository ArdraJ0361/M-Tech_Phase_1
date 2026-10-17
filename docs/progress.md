# Project Progress
## M.Tech: Parameter-Efficient Fine-Tuning of Small Language Models for Document Analysis

Legend: `[x]` done · `[-]` in progress · `[ ]` not started

---

## Status

```
Phase 0  — Setup & pipeline code        [x] Complete
Phase 1A — Pipeline verification (CPU)  [x] tests/smoke_test.py passes all 9 checks
Phase 1B — SQuAD v2 QA (3-way)          [x] Complete — Colab T4, 2026-10-17
Phase 1C — XL-Sum summarization         [ ] Next
Phase 1D — Analysis & Phase 1 report    [-] Analysis answered below; error analysis + write-up pending
Phase 2  — Document tasks (Semester 4)  [ ] Not started
```

**Next action:** error analysis on `experiments/simple/predictions.csv` (start with the 44 answerable
questions the LoRA model answered "unanswerable"), then run XL-Sum.

---

## Phase 0 — Setup

- [x] Literature survey (6 papers, `docs/literature_notes.md`) and dataset study (5 datasets, `docs/dataset_notes.md`)
- [x] Pipeline code in `src/`: extraction, dataset loading with disjoint splits, prompt masking, LoRA trainer,
      generation-based evaluator, metrics with answerable/unanswerable breakdown, CSV experiment tracker
- [x] Datasets validated: SQuAD v2 (`rajpurkar/squad_v2`) and XL-Sum English (`csebuetnlp/xlsum`)

## Phase 1A — Pipeline verification

- [x] `python tests/smoke_test.py` — all 9 checks pass on CPU: config, disjoint splits with unanswerable
      examples present, prompt format, prompt masking (prompt = -100, answer trained), base-model inference,
      unanswerable-breakdown metrics, experiment tracker.

## Phase 1B — SQuAD v2 QA (complete)

Run with `notebooks/simple_pipeline.ipynb` on a Colab T4. Raw files: `experiments/simple/`.

**Setup:** 5,000 train (4,926 kept after length filter) / 500 validation / 500 test
(250 answerable, 250 unanswerable), seed 42; LoRA r=8, α=16, q_proj+v_proj; 3 epochs, lr 2e-4, effective batch 16.

| Model | Method | EM | F1 | Ans EM | Unans EM | Latency ms | Peak GPU MB | Train min |
|---|---|---|---|---|---|---|---|---|
| Qwen1.5-0.5B | zero-shot | 5.6 | 16.39 | 11.2 | 0.0 | 190.6 | — | — |
| Qwen1.5-0.5B | **LoRA (ours)** | **68.8** | **72.44** | **66.4** | **71.2** | **150.2** | 4,978 | 21.2 |
| Qwen1.5-7B (4-bit) | zero-shot | 27.8 | 36.55 | 55.6 | 0.0 | 1,139.3 | — | — |

**LoRA adapter:** 786,432 trainable / 464,774,144 total = **0.169 %**.
**Training curve:** train loss 1.47 → 0.24; validation loss per epoch 0.417 → 0.399 → 0.393.

### Engineering notes
- **Out-of-memory on the T4 (root cause, 2026-10-17):** the Trainer's `compute_metrics` callback made it keep
  full-vocabulary logits for every validation token (~500 × seq × 152K values, tens of GB). Smaller batches and
  shorter sequences (earlier attempts) could not fix it. Fix: `prediction_loss_only=True` (validation scored by
  loss) and test EM/F1 computed by generation — the same scorer as the zero-shot baselines.
- **Colab quirks:** recent `peft` needs `torchao>=0.16`; the Hub's Xet downloader stalled and filled RAM on the
  15 GB 7B download, so the notebook sets `HF_HUB_DISABLE_XET=1`.
- The base-model score (EM 5.6, F1 16.39) was reproduced exactly on a local CPU run.

## Phase 1C — XL-Sum summarization (next)

```bash
python src/training/run_pilot.py --config configs/pilot_xlsum_summarization.yaml --base_only   # base SLM
python src/training/run_pilot.py --config configs/pilot_xlsum_summarization.yaml               # LoRA
```
- [ ] Base SLM ROUGE-1/2/L
- [ ] LoRA SLM ROUGE-1/2/L, training time, peak GPU memory
- [ ] 7B baseline — needs a summarization variant of `configs/llm_baseline_qwen7b.yaml` (`task: summarization`)
- [ ] 3-way ROUGE table

## Phase 1D — Analysis & report

QA analysis questions:
- [x] LoRA gain over base SLM: **+63.2 EM** (5.6 → 68.8)
- [x] Gap to the 7B baseline: tuned 0.5B is **41.0 EM ahead** (68.8 vs 27.8)
- [x] Answerable vs unanswerable: better on **both** — 66.4 vs 55.6 answerable EM; 71.2 vs 0.0 unanswerable EM
- [x] Inference speed: ~**7.6×** faster in this setup (150 vs 1,139 ms; 7B in 4-bit at batch 4)
- [ ] LoRA memory vs full fine-tuning — LoRA peaked at 4,978 MB; full fine-tuning not measured

Error analysis (`experiments/simple/predictions.csv`):
- [ ] Over-abstention: 44 answerable questions answered "unanswerable"
- [ ] Spans given on unanswerable questions
- [ ] Question types the model gets wrong

Report:
- [ ] Experimental setup · QA table · summarization table · efficiency table · discussion
- [ ] Limitations: one seed; zero-shot 7B without an "unanswerable" instruction; latency across different
      precisions/batch sizes; clean Wikipedia text rather than real business documents

## Phase 2 — Document tasks (Semester 4)

- [ ] FUNSD information extraction (loader exists; text-only mode loses layout — document it)
- [ ] RVL-CDIP classification (OCR pre-extraction with the Tesseract pipeline in `src/extraction/`)
- [ ] DocVQA (if time permits)
- [ ] Cross-task and efficiency analysis; OCR-quality impact; grounding spot-check
- [ ] Ablations: LoRA rank and target modules; 2–3 seeds; few-shot 7B baseline
- [ ] Thesis chapters and paper

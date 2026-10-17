# Problem Statement
## M.Tech Research: Parameter-Efficient Fine-Tuning of Small Language Models for Document Analysis

---

## 1. Background

Modern organisations process enormous volumes of documents daily — contracts, invoices,
research papers, policy documents, medical records, and financial reports. Extracting
meaningful information from these documents — answering questions, generating summaries,
classifying document types, extracting structured fields — is a core workflow in many
industries.

**Large Language Models (LLMs)** such as GPT-4 and Llama-3-70B have demonstrated strong
capability on these tasks. However, deploying LLMs in production is expensive:

- A 70B model requires 4–8 high-end GPUs just to run inference
- Latency is high (seconds per document), making real-time pipelines impractical
- API costs accumulate rapidly at scale
- Organisations with data privacy constraints cannot use cloud-hosted LLMs

**Small Language Models (SLMs)** — models in the 0.5B–3B parameter range — are much
cheaper to deploy and run on commodity hardware. However, their zero-shot performance
on domain-specific document tasks is significantly weaker than large models.

---

## 2. The Core Research Problem

The central question this project investigates is:

> **Can a small language model (0.5B parameters), fine-tuned using parameter-efficient
> techniques, achieve performance competitive with a large language model (7B parameters)
> on document analysis tasks — while requiring a fraction of the compute?**

This is not a hypothetical question. It directly determines whether organisations can
replace expensive LLM API calls with small, locally-hosted fine-tuned models without
sacrificing meaningful accuracy.

---

## 3. Why Standard Fine-Tuning Is Not the Answer

The obvious solution — fully fine-tuning a small model on task data — has its own
problems:

| Problem | Detail |
|---|---|
| **Memory cost** | Full fine-tuning a 0.5B model requires storing gradients for all 500M parameters — needs ~8–12 GB VRAM |
| **Storage cost** | Each task requires saving a separate full copy of the model (~1 GB per task) |
| **Training time** | Updating 500M parameters takes significantly longer than a targeted adapter |
| **Catastrophic forgetting** | Updating all weights risks degrading the model's general language understanding |

These costs multiply when you have multiple tasks (QA, summarization, classification,
extraction) — you would need a separate full model for each.

---

## 4. The Proposed Solution: PEFT with LoRA

**Parameter-Efficient Fine-Tuning (PEFT)** addresses all of the above by training only a
tiny fraction of the model's parameters while keeping the base model completely frozen.

**LoRA (Low-Rank Adaptation)** is the specific PEFT technique used in this project.
Instead of updating the full weight matrices in the transformer, LoRA injects small
trainable matrices (called adapters) alongside each frozen layer:

```
Standard fine-tuning:  update all 500,000,000 parameters
LoRA fine-tuning:      update only ~786,000 adapter parameters  (0.169% of total)
```

This delivers:
- Training on a single 15 GB GPU (NVIDIA T4 — free tier on Google Colab)
- Adapter storage of a few MB per task (vs ~1 GB for full model)
- The base model remains unchanged and reusable across all tasks
- Training completes in hours instead of days

---

## 5. Research Objectives

This project pursues four specific objectives:

**Objective 1 — Quantify the LoRA improvement over zero-shot**
Measure how much LoRA fine-tuning improves a 0.5B model compared to its own
untuned (zero-shot) baseline on extractive QA and abstractive summarization.

**Objective 2 — Measure the gap to a larger model**
Compare the fine-tuned 0.5B model against a zero-shot 7B model (same model family,
same test set) to quantify how much of the large-model advantage can be recovered
through targeted fine-tuning.

**Objective 3 — Measure training efficiency**
Record trainable parameter count, peak GPU memory, and training time for LoRA
fine-tuning — to demonstrate that the efficiency gains are real and reproducible.

**Objective 4 — Measure inference efficiency**
Record per-example latency and throughput for the 0.5B vs 7B models — to demonstrate
the practical deployment advantage of the smaller model.

---

## 6. Tasks and Datasets

Two standard NLP benchmark tasks are used, chosen to represent the most common
document analysis workflows:

### Task 1 — Extractive Question Answering
**Dataset:** SQuAD v2 (Stanford Question Answering Dataset v2)
- Reading comprehension: given a context passage and a question, extract the answer span
- Includes **unanswerable questions** (model must output "unanswerable" if no answer exists)
- Metric: Exact Match (EM) and Token F1
- Split used: 5,000 train / 500 validation / 500 test (disjoint, seed=42)

### Task 2 — Abstractive Summarization
**Dataset:** XL-Sum (BBC multilingual summarization, English subset)
- Given a news article, generate a one-paragraph summary
- Tests the model's ability to compress and paraphrase — not just extract
- Metric: ROUGE-1, ROUGE-2, ROUGE-L
- Split used: 5,000 train / 500 validation / 500 test (official splits)

---

## 7. Models

| Role | Model | Parameters | Training |
|---|---|---|---|
| SLM baseline | Qwen1.5-0.5B (base) | 0.5B | Zero-shot, no training |
| SLM + PEFT (ours) | Qwen1.5-0.5B (base) + LoRA | 0.5B + 786K adapter | LoRA fine-tuned, 3 epochs |
| LLM reference | Qwen1.5-7B (base) | 7B | Zero-shot, no training |

All three models are from the **same model family** (Qwen1.5) and use the **base** (not
instruct) variant. This ensures the comparison is controlled: any performance difference
is attributable to model size or LoRA fine-tuning — not to instruction tuning or
different pre-training data.

---

## 8. Experimental Design

The experiment is a controlled **3-way comparison** on identical test sets:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Setup A         │     │  Setup B         │     │  Setup C         │
│  Qwen1.5-0.5B    │     │  Qwen1.5-0.5B    │     │  Qwen1.5-7B      │
│  zero-shot       │     │  + LoRA adapter  │     │  zero-shot       │
│                  │     │  (PEFT-tuned)    │     │                  │
│  Lower bound     │     │  ← Contribution  │     │  Reference       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         |                        |                        |
         └────────────────────────┴────────────────────────┘
                          Same 500 test examples
                          Same random seed (42)
                          Same prompt format
```

**What each comparison answers:**
- **A vs B**: Does LoRA fine-tuning improve the small model?
- **B vs C**: How close can a fine-tuned small model get to a large model?
- **A vs C**: How much does scale alone help (without fine-tuning)?

---

## 9. Success Criteria

The project is considered successful if:

1. **LoRA improves the base SLM** — Setup B scores higher than Setup A on both EM/F1
   (QA) and ROUGE (summarization)
2. **The efficiency gap is real** — LoRA training uses < 15 GB VRAM and completes
   within a single Colab session (~2 hours)
3. **The 0.169% trainable claim is verified** — the actual measured trainable
   parameter percentage is confirmed at 0.169% (786,432 / 464,774,144)
4. **Latency advantage is documented** — the 0.5B model is demonstrably faster
   per example than the 7B model

---

## 10. Scope and Limitations

**In scope (Phase 1 — Semester 3):**
- Two tasks: extractive QA and abstractive summarization
- One SLM: Qwen1.5-0.5B
- One PEFT method: LoRA (r=8, target modules: q_proj, v_proj)
- Text-only pipeline (no image understanding)

**Out of scope for Phase 1:**
- Document images / scanned PDFs (text extraction pipeline built but not used in training)
- Other PEFT methods (prefix tuning, adapter layers, IA³)
- Multi-task training (each task trained separately)
- Domain-specific documents (experiments use public benchmarks)

**Extended scope (Phase 2 — Semester 4):**
- Information extraction on FUNSD dataset
- Document classification on RVL-CDIP
- DocVQA (if time permits)
- Cross-task analysis and full thesis write-up

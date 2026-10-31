# Financial Document Training Plan

**New direction:** pivot Phase 1 / Phase 2 toward financial document analysis —
fine-tune the SLM on SEC filings / financial QA instead of generic SQuAD v2 /
XL-Sum. Everything else in the pipeline (LoRA, evaluator, prompt format, serving)
stays the same.

---

## What the Sources Say

### Paper 1 — DocFinQA (arxiv 2401.06915)
- Extends FinQA with **full SEC filing context** (avg 123k words per document)
- 7,437 QA pairs grounded in real SEC 10-K reports
- Key finding: even state-of-the-art LLMs struggle with long financial documents
- **Dataset:** `kensho/DocFinQA` on HuggingFace
- Task type: extractive + numerical QA (same structure as SQuAD v2 — fits your pipeline)

### Paper 2 — FinRAG-QA (arxiv 2609.03654)
- 999 practitioner-curated questions on 10 financial indicators
- 209 annual reports from 24 major banks (European + US), avg 198k words
- Shows RAG pipeline with chunk enrichment lifts NDCG@10 from 0.322 → 0.710
- **Takeaway:** chunking + retrieval matters a lot for long financial docs

### Article — IntuitionLabs LLM Financial Analysis
- FinanceBench accuracy: 79% when filing is in context, 19% closed-book
- GPT-5.2 reaches 88% on financial reasoning — but hallucinations still a problem
- Domain-specific fine-tuning (BloombergGPT, FinBERT) outperforms general models
- **Takeaway:** a fine-tuned small model beats a large zero-shot model — exactly your research question

### Dataset 1 — financial-reports-sec (HuggingFace)
- `JanosAudran/financial-reports-sec` (the correct HF ID)
- SEC 10-K filings 1993–2020, sentence-level, 20 sections per filing
- Sentiment labels (positive/negative) per filing
- **Tasks it supports:** sentiment classification, masked LM, sentence ordering
- **Fit:** good for classification task (Phase 2 equivalent)

### Dataset 2 — DataNeed/company-reportsfinalc
- Page returns 404 — dataset does not exist or has been removed
- **Do not use**

---

## How This Fits Your Existing Pipeline

Your pipeline already handles these task types:

| Task type | Your pipeline supports | Financial dataset |
|---|---|---|
| QA (extractive) | ✅ `task: qa` | DocFinQA, FinQA |
| Summarization | ✅ `task: summarization` | Financial reports → summaries |
| Classification | ✅ `task: classification` | financial-reports-sec (sentiment) |
| Extraction | ✅ `task: extraction` | SEC filing key metrics |

**No new pipeline code needed.** New configs + new datasets.

---

## Proposed Training Plan

### Phase 1B (replace SQuAD) — Financial QA

| Item | Value |
|---|---|
| Dataset | `kensho/DocFinQA` |
| Task | QA — question + SEC document chunk → answer |
| Why | Same extractive QA format as SQuAD v2, directly comparable |
| Train samples | 5,000 (same as SQuAD pilot) |
| Test samples | 500 |
| Metrics | Exact Match, Token F1 (same as SQuAD) |
| Config to create | `configs/pilot_docfinqa.yaml` |

**Research value:** shows LoRA-tuned SLM on domain-specific financial QA vs
zero-shot 7B — directly answers the research question in a real-world domain.

### Phase 1C — Financial Sentiment Classification

| Item | Value |
|---|---|
| Dataset | `JanosAudran/financial-reports-sec` (small_full config) |
| Task | Sentence-level classification → positive / negative |
| Train samples | 5,000 sentences |
| Test samples | 500 |
| Metrics | Accuracy, Macro-F1 |
| Config to create | `configs/pilot_fin_sentiment.yaml` |

**Research value:** classification is a different task type — shows PEFT generalises
across task types in the financial domain.

### Phase 2 (Semester 4) — Long-doc RAG + Extraction

| Item | Value |
|---|---|
| Approach | Chunk SEC filings → retrieve relevant chunks → LoRA-tuned SLM answers |
| Dataset | DocFinQA full documents (avg 123k words) |
| Key challenge | Context is too long for the SLM — needs chunking + retrieval first |
| Benchmark to beat | FinanceBench 19% closed-book → target >50% with retrieval |

---

## Key Challenges (be aware before starting)

| Challenge | Detail | Mitigation |
|---|---|---|
| Long context | DocFinQA docs avg 123k words — SLM max is 512 tokens | Chunk + retrieve relevant passages first |
| Numerical reasoning | Financial QA often needs arithmetic (e.g. % change) | Report separately; LoRA helps on pattern but not calculation |
| Table understanding | SEC filings have dense tables | Phase 2 — text-only first, tables later |
| Dataset licence | DocFinQA is research-only | Fine for M.Tech thesis |
| DataNeed dataset 404 | Dataset does not exist | Use DocFinQA + financial-reports-sec instead |

---

## Immediate Next Steps

| Step | What | Config file |
|---|---|---|
| 1 | Load DocFinQA, inspect format, map to your `qa` prompt format | — |
| 2 | Create `configs/pilot_docfinqa.yaml` (copy pilot_squad_qa.yaml, change dataset) | `configs/pilot_docfinqa.yaml` |
| 3 | Run smoke test with DocFinQA on CPU (`DRY_RUN = True`, 16 examples) | — |
| 4 | Full training run on Kaggle/Vast.ai (~21 min, same as SQuAD) | — |
| 5 | Create `configs/pilot_fin_sentiment.yaml` for classification task | `configs/pilot_fin_sentiment.yaml` |

---

## References

- DocFinQA paper: https://arxiv.org/abs/2401.06915
- FinRAG-QA paper: https://arxiv.org/abs/2609.03654
- IntuitionLabs article: https://intuitionlabs.ai/articles/llm-financial-document-analysis
- DocFinQA dataset: https://huggingface.co/datasets/kensho/DocFinQA
- financial-reports-sec dataset: https://huggingface.co/datasets/JanosAudran/financial-reports-sec

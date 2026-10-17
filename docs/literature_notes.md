# Literature Survey Notes
## M.Tech Project — PEFT-SLM Document Analysis

---

## Paper 1: Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey
**Link:** https://arxiv.org/pdf/2403.14608  
**Main focus:** Reviews PEFT techniques — additive, selective, reparameterized, hybrid.  
**Relevance:** Foundation for choosing LoRA/QLoRA.  
**Key limitation:** Broad survey; does not establish per-task document-analysis performance.  
**How we use it:** Justify the fine-tuning approach.

---

## Paper 2: A Survey of Small Language Models
**Link:** https://arxiv.org/pdf/2410.20011  
**Main focus:** SLM architectures, training, compression, evaluation.  
**Relevance:** Explains why SLMs matter; guides model selection.  
**Key limitation:** General survey; no controlled doc-analysis comparison.  
**How we use it:** SLM selection and evaluation framing.

---

## Paper 3: What is the Role of Small Models in the LLM Era: A Survey
**Link:** https://arxiv.org/pdf/2409.06857  
**Main focus:** When small models complement or compete with LLMs.  
**Relevance:** Directly motivates the SLM vs LLM comparison.  
**Key limitation:** No task-specific experiments using our datasets.  
**How we use it:** Motivate empirical comparison.

---

## Paper 4: A Survey on MLLM-based Visually Rich Document Understanding
**Link:** https://arxiv.org/pdf/2507.09861  
**Main focus:** Multimodal document understanding — OCR-based and OCR-free.  
**Relevance:** Landscape context for text-only vs vision-language approaches.  
**Key limitation:** Multimodal focus; findings don't transfer directly to text-only SLMs.  
**How we use it:** Define scope of our text-based approach.

---

## Paper 5: Parameter-Efficient Fine-Tuning in Large Language Models: A Survey of Methodologies
**Link:** https://link.springer.com/article/10.1007/s10462-025-11236-4  
**Main focus:** PEFT methods and practical considerations.  
**Relevance:** Additional coverage of PEFT configurations.  
**Key limitation:** Broad review; does not resolve optimal PEFT config for our tasks.  
**How we use it:** Support PEFT configuration selection.

---

## Paper 6: A Comprehensive Survey of Small Language Models in the Era of LLMs
**Link:** https://dl.acm.org/doi/full/10.1145/3768165  
**Main focus:** SLM techniques, applications, trustworthiness, LLM collaboration.  
**Relevance:** Broad research directions and SLM limitations.  
**Key limitation:** Does not establish per-benchmark PEFT-SLM performance.  
**How we use it:** Contextualize evaluation; identify research gaps.

---

## Preliminary Research Gap

A controlled experimental evaluation of PEFT-adapted SLMs across selected
document-analysis tasks (QA, summarization, classification, extraction) with:
- Comparison against untuned SLM baselines
- Comparison against larger-model baselines  
- Systematic efficiency measurements (time, memory, latency)

**Note:** Validate this gap against closely related empirical studies before
committing to the final claim. Check experimental sections of cited papers.

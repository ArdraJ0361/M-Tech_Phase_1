# Dataset Notes
## M.Tech Project — PEFT-SLM Document Analysis

---

## Dataset Summary

| Dataset | Task | Input Type | Text-only? | HF Hub Name |
|---|---|---|---|---|
| SQuAD v2 | QA | Text passages | ✅ Native | `squad_v2` |
| XL-Sum (English) | Summarization | News articles | ✅ Native | `csebuetnlp/xlsum` |
| DocVQA | QA | Document images | ❌ OCR needed | `HuggingFaceM4/DocumentVQA` |
| FUNSD | Extraction | Scanned forms | ❌ OCR/layout | `nielsr/funsd` |
| RVL-CDIP | Classification | Document images | ❌ OCR needed | `aharley/rvl_cdip` |

---

## SQuAD v2
- **Train:** 130,319 examples  
- **Validation:** 11,873 examples  
- **Format:** `context`, `question`, `answers` (text + start positions)  
- **Note:** 50% of examples are unanswerable — model must also output "unanswerable"  
- **For pilot:** Use 5,000 train / 500 val / 500 test  

## XL-Sum (English)
- **Train:** 306,522 examples  
- **Validation:** 11,535 examples  
- **Test:** 11,535 examples  
- **Format:** `title`, `text` (article), `summary`  
- **For pilot:** Use 5,000 train / 500 val / 500 test  

## DocVQA
- **Status:** Requires image processing or pre-extracted OCR text  
- **Access:** HuggingFace Hub (check license)  
- **Note:** Defer to Phase 2 or use available OCR annotations  

## FUNSD
- **Train:** 149 documents, 9,709 words  
- **Test:** 50 documents, 3,410 words  
- **Format:** Word-level annotations with B-QUESTION, B-ANSWER, B-HEADER  
- **Note:** Small dataset; needs conversion to generative extraction format  

## RVL-CDIP
- **Total:** 400,000 images, 16 classes  
- **Note:** Very large; use a stratified 1,000-sample subset for text-only experiments  
- **OCR:** Pre-extracted OCR text available in some HF variants  

---

## Phase 1 Priority
1. SQuAD v2 — QA pilot (text-native, easiest baseline)
2. XL-Sum English — Summarization pilot
3. FUNSD — Extraction (small enough to process quickly)

Phase 2: DocVQA, RVL-CDIP (requires OCR pipeline)

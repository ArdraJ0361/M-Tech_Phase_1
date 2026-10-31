# Final Master Plan — PEFT-SLM Financial Document Analysis

---

## The Big Picture

```
ALL ON RENTED GPU (Vast.ai / Kaggle)
─────────────────────────────────────────────────────────────────
Step 1 — TRAIN    Fine-tune SLM on financial documents with LoRA
Step 2 — SERVE    Start FastAPI endpoint (same session, no download needed)
Step 3 — PANEL    Run notebook panel → type question → get answer live
Step 4 — DOWNLOAD Download adapter/ (~3-5 MB) before destroying instance
Step 5 — DESTROY  Kill the GPU instance → billing stops

LATER on your laptop (no GPU needed, ever again)
─────────────────────────────────────────────────────────────────
Load adapter + base model → serve endpoint → answer questions locally
```

**Research question:** can a LoRA-tuned 0.5B model match a 7B model on
financial document tasks, at 7× lower latency and a fraction of the cost?

---

## Phase 1 — What You Train On

### Run 1 — Financial QA  (~25-30 min, Kaggle free)

| Item | Value |
|---|---|
| Dataset | `kensho/DocFinQA` — 7,437 QA pairs from real SEC 10-K filings |
| Task | Question + document passage → answer |
| Example | "What was revenue growth in Q4 2012?" → "12%" |
| Train / Val / Test | 5,000 / 500 / 500 |
| Metrics | Exact Match, Token F1 (answerable + unanswerable) |
| Config | `configs/pilot_docfinqa.yaml` (to create) |
| GPU cost | ₹0 on Kaggle |

### Run 2 — Financial Sentiment  (~15-20 min, Kaggle free)

| Item | Value |
|---|---|
| Dataset | `JanosAudran/financial-reports-sec` — SEC 10-K sentences, 1993-2020 |
| Task | Sentence → positive / negative (based on market reaction) |
| Example | "Revenue declined 8% year-on-year..." → negative |
| Train / Val / Test | 5,000 / 500 / 500 |
| Metrics | Accuracy, Macro-F1 |
| Config | `configs/pilot_fin_sentiment.yaml` (to create) |
| GPU cost | ₹0 on Kaggle |

---

## Phase 2 — Long-doc RAG (Semester 4)

| Item | Value |
|---|---|
| Dataset | DocFinQA full documents (avg 123k words per SEC filing) |
| Approach | Chunk → retrieve relevant passages → LoRA SLM answers |
| Challenge | SLM max context is 512 tokens; full doc is 123k words |
| Benchmark to beat | FinanceBench baseline: 19% closed-book → target >50% with RAG |

---

## The Three Models Compared (same as Phase 1B)

| Model | Method | Role |
|---|---|---|
| Qwen1.5-0.5B | Zero-shot base | Starting point — how bad without training? |
| **Qwen1.5-0.5B + LoRA** | Fine-tuned | **Your contribution** |
| Qwen1.5-7B | Zero-shot base | Reference — how close can small tuned get? |

All three: same seed (42), same test examples, same prompt format.

---

## After Training — Serve + Panel

```
Stage 1 — TRAIN  (on GPU, Kaggle)
  └─ LoRA fine-tune on DocFinQA
  └─ Save adapter/ → download to laptop

Stage 2 — SERVE  (on laptop, no GPU)
  └─ Load base model + adapter
  └─ FastAPI server on localhost:8000
  └─ POST /predict  { context, question } → { answer, latency_ms }
  └─ GET  /health   → { status, model }

Stage 3 — PANEL  (notebook, interactive)
  └─ ipywidgets: context box + question box + Ask button
  └─ Click → hits endpoint → answer appears
  └─ Works in Colab or locally
```

---

## Where to Train (GPU) — You Are Renting

| Option | Cost | Recommended? |
|---|---|---|
| **Vast.ai RTX 3090** | ~₹7/run (~$0.166/hr) | ✅ Cheapest, already looked at listings |
| Jarvislabs.ai RTX 5000 | ~₹12/run | Indian platform, UPI accepted |
| Kaggle Notebooks | ₹0 (30 h/week) | Free fallback if Vast.ai is slow |

**You need ≥ 6 GB VRAM. Your peak usage is 5 GB (from Phase 1B).**

### Vast.ai Instance to Pick
```
GPU:        1x RTX 3090  (24 GB VRAM — well above your 5 GB need)
Template:   PyTorch       (pre-installed, no setup)
Disk:       16 GB
Price:      ~$0.166/hr   (~₹14/hr)
Your cost:  2 runs × 30 min = ~$0.17 total  (~₹14)
```

---

## Timeline — Everything on Rented GPU

| Step | What | Where | Time | Cost |
|---|---|---|---|---|
| Before renting | Create configs on laptop, smoke test CPU | Laptop | 30 min | ₹0 |
| GPU session 1 | DocFinQA training + serve + panel demo | Vast.ai | ~45 min | ~₹10 |
| GPU session 2 | Sentiment training + serve + panel demo | Vast.ai | ~35 min | ~₹8 |
| After GPU | Download adapters, destroy instance | — | 5 min | ₹0 |
| Local | Serve endpoint from laptop with downloaded adapter | Laptop | ongoing | ₹0 |

**Total GPU cost: ~₹18 for both training runs + live demo.**

---

## Step-by-Step: Full Rented GPU Workflow

### Before You Rent (laptop, free)
```
1. Create configs/pilot_docfinqa.yaml
   (copy pilot_squad_qa.yaml, change dataset name to kensho/DocFinQA)

2. Smoke test on CPU — proves pipeline works before paying for GPU
   python src/training/run_pilot.py --config configs/pilot_docfinqa.yaml --base_only
   (downloads 0.5B model, runs 16 examples, ~5 min)
```

### On the Rented GPU (Vast.ai session)
```
3. Rent instance → Template: PyTorch → Disk: 16 GB → RENT

4. In terminal:
   pip install -q peft bitsandbytes transformers datasets accelerate fastapi uvicorn
   export HF_HUB_DISABLE_XET=1
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO
   cd my-project

5. Run all 3 models (base SLM + LoRA train + 7B baseline):
   python src/training/run_pilot.py --config configs/pilot_docfinqa.yaml
   (~25-30 min)

6. Start the serving endpoint (still on GPU — fastest answers):
   python -m src.serving.server --config configs/pilot_docfinqa.yaml &

7. Open the notebook panel in Jupyter on the instance:
   jupyter notebook --ip=0.0.0.0 --port=8888
   → open inference_demo.ipynb → type a financial question → get answer

8. Download the adapter before destroying:
   zip -r adapter_docfinqa.zip experiments/docfinqa/adapter/
   (download via Vast.ai file browser)

9. DESTROY the instance → billing stops
```

### After (laptop, no GPU, free forever)
```
10. pip install -r requirements.txt
    python -m src.serving.server --config configs/pilot_docfinqa.yaml
    → open localhost:8000/docs → ask financial questions at ~1-2 sec/answer
```

---

## Files You'll Have at the End

```
configs/
├── pilot_docfinqa.yaml          ← Financial QA config (to create)
└── pilot_fin_sentiment.yaml     ← Sentiment config (to create)

experiments/
├── docfinqa/
│   ├── adapter/                 ← LoRA weights for financial QA (~3-5 MB)
│   ├── results.csv              ← EM/F1: base SLM vs LoRA vs 7B
│   └── predictions.csv
└── fin_sentiment/
    ├── adapter/                 ← LoRA weights for sentiment
    └── results.csv              ← Accuracy/F1: base SLM vs LoRA vs 7B

notebooks/
└── inference_demo.ipynb         ← Train → Serve → Panel in one notebook

docs/
├── financial_training_plan.md   ← Dataset + task details
├── gpu_rental_plan.md           ← Where and how to rent GPU
├── serving_plan.md              ← Endpoint architecture
└── inference_notebook_plan.md   ← Notebook cell-by-cell plan
```

---

## Key Numbers to Know

```
Model:            Qwen1.5-0.5B  (500M params)
LoRA adapter:     0.169% of params trained  (786,432 weights)
Adapter size:     ~3-5 MB saved to disk
Training time:    ~25 min per run on T4/P100
Peak GPU memory:  ~5 GB  (need ≥ 6 GB VRAM)
Inference speed:  ~150 ms/answer on GPU,  ~1-2 sec on CPU
Total GPU cost:   ₹0 on Kaggle free tier
```

---

## What Makes This a Strong Research Contribution

1. **Domain-specific** — financial documents, not generic Wikipedia text
2. **Practical** — SEC filings are real documents analysts use every day
3. **Efficient** — 0.169% parameters trained, beats a 14× larger model
4. **Reproducible** — same seed, same splits, same prompt format across all 3 models
5. **Deployable** — trained model runs as a live endpoint, not just a benchmark number

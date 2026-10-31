# Inference Notebook Plan — Train → Serve → Interactive Panel

**Goal:** a single Colab notebook that (1) fine-tunes the SLM with LoRA,
(2) starts a local HTTP endpoint, and (3) gives you a live panel where you
type a question + context and get an answer back.

---

## Three-Stage Flow

```
Stage 1 — TRAIN
  └─ LoRA fine-tune Qwen1.5-0.5B on SQuAD v2 (same as Phase 1B)
  └─ Save adapter to experiments/inference_demo/adapter/

Stage 2 — SERVE
  └─ Load the saved adapter
  └─ Start a FastAPI + uvicorn server in a background thread
  └─ Server listens on localhost:8000
  └─ POST /predict  { context, question } → { answer, latency_ms }
  └─ GET  /health   → { status, model }

Stage 3 — PANEL
  └─ ipywidgets form in the notebook (text boxes + button)
  └─ Button click → POST to localhost:8000/predict
  └─ Answer + latency printed below the form
  └─ Works entirely in Colab — no external URL needed
```

---

## Notebook Cells (one per row)

| Cell | Type | What it does |
|---|---|---|
| 0 | code | Install deps: `peft`, `bitsandbytes`, `fastapi`, `uvicorn`, `ipywidgets`; set `HF_HUB_DISABLE_XET=1` |
| 1 | code | Config: `DRY_RUN`, model name, LoRA params, `OUT` dir, `SKIP_TRAIN` flag |
| 2 | code | Load SQuAD v2, build disjoint train/val/test splits (identical to Phase 1B) |
| 3 | code | Tokenize + prompt-mask (same prompt format the model was trained on) |
| 4 | code | LoRA fine-tune with HF Trainer; save adapter; print loss + peak GPU memory |
| 5 | markdown | **Stage 2 — Serve** explanation |
| 6 | code | Define FastAPI app inline (15 lines); load adapter + tokenizer on startup |
| 7 | code | Start uvicorn in a `threading.Thread`; health-check with `requests.get` |
| 8 | markdown | **Stage 3 — Panel** explanation |
| 9 | code | `ipywidgets` form: context textarea, question input, Ask button, output area |
| 10 | code | Sample questions pre-filled — run to test without typing |

---

## Key Design Decisions

**`SKIP_TRAIN = True` flag** — if the adapter already exists from a previous
run (or from Phase 1B), set this to skip straight to Stage 2. No re-training.

**Server in a background thread** — `threading.Thread(target=uvicorn.run, daemon=True)`.
Colab cells are blocking by default; a daemon thread lets the notebook stay
interactive while the server runs.

**Same prompt format** — the server uses the exact same
`### Context / ### Question / ### Answer:` template the model was trained on.
Using a different format at inference time would silently hurt quality.

**`ipywidgets` panel** — works natively in Colab without any extra tunnelling.
The button's `on_click` handler calls `requests.post("http://localhost:8000/predict", ...)`.

**`DRY_RUN = True` for quick check** — uses 64 train / 16 val / 16 test examples
and 1 epoch (~5 min). The panel still works after a dry-run; answers will be
weaker because the model is barely trained.

---

## How to Run

```
1. Upload notebooks/inference_demo.ipynb to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom
4. Stage 1 (~21 min real run, ~5 min dry run): model trains and saves
5. Stage 2 (~10 sec): server starts, health check prints "status: ok"
6. Stage 3: type your own context + question, click Ask, see the answer
```

---

## Files Produced

```
notebooks/
└── build_inference_notebook.py   ← builder script (edit this, not the .ipynb)
└── inference_demo.ipynb          ← generated Colab notebook

experiments/inference_demo/
├── adapter/                      ← saved LoRA weights
├── results.csv
├── train_log.json
└── predictions.csv
```

---

## Out of Scope (for this notebook)

- Summarization task (same pattern, separate notebook)
- Public URL sharing (add `pyngrok` later — one extra cell)
- Authentication on the endpoint
- Phase 2 document-image tasks

# Serving Plan — Shareable Inference Endpoint for the PEFT-SLM

**Goal:** expose the fine-tuned LoRA checkpoint as an HTTP API so a panel
demo, a colleague, or a UI can send a question + context and get an answer
back — without touching training code.

---

## What We Have vs. What We Need

| Layer | Status | Note |
|---|---|---|
| LoRA training pipeline | ✅ Done | `src/training/` |
| Saved LoRA adapter checkpoint | ✅ Done | `experiments/simple/` (Phase 1B) |
| Batch evaluator (dataset → metrics) | ✅ Done | `src/evaluation/evaluator.py` |
| Single-example inference function | ❌ Missing | Needed by the server |
| HTTP server / endpoint | ❌ Missing | The shareable piece |
| `fastapi` + `uvicorn` in deps | ❌ Missing | Two lines in `requirements.txt` |

**Minimal scope:** one new module (`src/serving/server.py`) + two lines in
`requirements.txt`. No changes to existing training or evaluation code.

---

## Approach

### Why FastAPI?

- Minimal boilerplate: ~60 lines to a working endpoint.
- Auto-generates interactive docs at `/docs` — the "shareable panel" is just
  a URL anyone can open in a browser and type into.
- Async-friendly; uvicorn serves it with a single command.
- Fits the existing Python stack (no new languages or containers required).

### Alternatives considered

| Option | Verdict |
|---|---|
| Gradio / Streamlit | Good for quick demos but adds a heavy UI dependency; overkill if you just need an API |
| Flask | Works but no async, no automatic OpenAPI schema |
| TorchServe / BentoML | Production-grade but far too heavy for a research demo |
| **FastAPI + uvicorn** | ✅ Right size for a research project |

---

## What the Server Does

```
Startup
  └─ load_tokenizer()        (from src/training/model_loader)
  └─ load_peft_checkpoint()  (base model + LoRA adapter weights)
  └─ model pinned to CUDA / CPU

POST /predict
  Body: { "context": "...", "question": "..." }
  ├─ build_qa_prompt(context, question)   (from src/data/prompts)
  ├─ tokenize prompt
  ├─ model.generate(...)
  ├─ decode + postprocess
  └─ Response: { "answer": "...", "latency_ms": 142 }

GET /health
  └─ { "status": "ok", "model": "...", "task": "qa" }
```

---

## File Plan

```
src/serving/
├── __init__.py          (empty)
└── server.py            (FastAPI app — ~100 lines)
```

### `server.py` — key design decisions

1. **Model loaded once at startup** via FastAPI's `lifespan` context manager —
   not on every request. A cold request should not wait for a 500 MB download.

2. **Config-driven** — the server takes a `--config` flag pointing at the same
   YAML used for training (e.g. `configs/pilot_squad_qa.yaml`). No hardcoded
   model paths.

3. **Checkpoint path** defaults to `{training.output_dir}/adapter` (same
   convention as `_run_checkpoint_evaluation` in `run_pilot.py`), but can be
   overridden with `--checkpoint`.

4. **Reuses existing code** — `load_tokenizer`, `load_peft_checkpoint`,
   `build_qa_prompt`, `_postprocess`. No copy-paste.

5. **CPU fallback** — if no GPU is available the model still loads (useful for
   a laptop demo); latency will be higher but it works.

6. **No auth** — this is a research demo, not production. Add a static API key
   header later if needed.

---

## How to Run It

```bash
# Install new deps (first time only)
pip install fastapi uvicorn

# Start the server (GPU machine / Colab)
python -m src.serving.server \
    --config configs/pilot_squad_qa.yaml \
    --checkpoint experiments/simple/adapter   # or wherever your adapter is saved

# The interactive panel is at:
#   http://localhost:8000/docs
```

### Share with a colleague (same network)

```bash
# Bind to all interfaces so others on the same network can reach it
python -m src.serving.server \
    --config configs/pilot_squad_qa.yaml \
    --host 0.0.0.0 --port 8000
# They open: http://<your-IP>:8000/docs
```

### Share publicly (e.g. from Colab)

```python
# In a Colab cell — ngrok gives a public HTTPS URL
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print("Public URL:", public_url)
# Then in another cell:
# !python -m src.serving.server --config configs/pilot_squad_qa.yaml --host 0.0.0.0
```

---

## API Contract

### `POST /predict`

**Request body (JSON):**
```json
{
  "context": "The Amazon rainforest covers over 5.5 million square kilometres.",
  "question": "How large is the Amazon rainforest?"
}
```

**Response:**
```json
{
  "answer": "over 5.5 million square kilometres",
  "latency_ms": 148
}
```

### `GET /health`

```json
{
  "status": "ok",
  "model": "Qwen/Qwen1.5-0.5B",
  "task": "qa",
  "checkpoint": "experiments/simple/adapter"
}
```

---

## What to Build — Step by Step

| Step | File | What |
|---|---|---|
| 1 | `requirements.txt` | Add `fastapi>=0.111.0` and `uvicorn>=0.29.0` |
| 2 | `src/serving/__init__.py` | Empty file (makes it a package) |
| 3 | `src/serving/server.py` | FastAPI app with `/predict` and `/health` |
| 4 | Smoke test | `curl` the `/health` endpoint to confirm it starts |

Estimated code: ~100 lines. No changes to any existing file except
`requirements.txt`.

---

## Scope Boundaries

**In scope:**
- QA task (`task: qa`) served from a saved LoRA checkpoint
- `/predict` and `/health` endpoints
- Config + checkpoint path via CLI flags
- Works on CPU and GPU

**Out of scope (future):**
- Summarization endpoint (same pattern, add later)
- Authentication / rate limiting
- Containerisation (Docker)
- Phase 2 document-upload endpoint (requires `src/extraction/`)
- Streaming token output (SSE / WebSockets)

---

## Invariants (must not break)

- Server reuses `build_qa_prompt` from `src/data/prompts.py` — same format the
  model was trained on. Do not invent a new prompt format in the server.
- `load_peft_checkpoint` already validates the adapter loaded correctly (asserts
  `adapter_params > 0`). The server does not need to duplicate this check.
- `seed: 42` and dataset split logic are irrelevant to serving — the server
  does live inference on user-supplied text, not dataset rows.

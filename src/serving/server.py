"""
src/serving/server.py
----------------------
HTTP endpoint for the fine-tuned model: send a passage and a question, get an answer.

Run from the repo root:
    python -m src.serving.server --config configs/pilot_fin10k_qa.yaml

Interactive docs: http://127.0.0.1:8000/docs

To listen beyond this machine (--host 0.0.0.0), set SERVING_API_KEY first. Callers
then send it in the X-API-Key header.
"""

from __future__ import annotations

import argparse
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, StringConstraints

from src.data.prompts import build_qa_prompt, clean_text, truncate_to_tokens
from src.evaluation.evaluator import _postprocess
from src.training.model_loader import load_peft_checkpoint, load_tokenizer
from src.utils.config_loader import load_config

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

# Limits keep a single request from tying up the model; the context is trimmed further below.
Context = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200_000)]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)]


class PredictRequest(BaseModel):
    context: Context
    question: Question


class PredictResponse(BaseModel):
    answer: str
    latency_ms: int
    context_truncated: bool   # true = only the start of the context was read


def answer_question(model, tokenizer, cfg: dict, context: str, question: str) -> tuple[str, bool]:
    """Answer one question using the same prompt, trimming and cleanup as evaluation."""
    max_src = cfg["dataset"]["max_source_length"]
    overhead = len(tokenizer(build_qa_prompt("", ""), add_special_tokens=False)["input_ids"])
    q_len = len(tokenizer(question, add_special_tokens=False)["input_ids"])
    budget = max(max_src - q_len - overhead, 32)

    cleaned = clean_text(context)
    trimmed = truncate_to_tokens(cleaned, tokenizer, budget)

    ids = tokenizer(build_qa_prompt(trimmed, question), return_tensors="pt")["input_ids"]
    ids = ids.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=cfg["dataset"]["max_target_length"],
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    return _postprocess(text, "qa"), trimmed != cleaned


def create_app(cfg: dict, checkpoint: str, api_key: str | None = None) -> FastAPI:
    state: dict = {}
    generate_lock = threading.Lock()   # one generation at a time on the shared model
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["tokenizer"] = load_tokenizer(cfg)
        state["model"] = load_peft_checkpoint(cfg, checkpoint)
        yield
        state.clear()

    def require_key(provided: str | None = Security(key_header)) -> None:
        if api_key and not (provided and secrets.compare_digest(provided, api_key)):
            raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")

    app = FastAPI(title="PEFT-SLM Financial QA", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": cfg["model_name_or_path"], "task": cfg["task"]}

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
    def predict(req: PredictRequest) -> PredictResponse:
        with generate_lock:
            start = time.perf_counter()
            answer, truncated = answer_question(
                state["model"], state["tokenizer"], cfg, req.context, req.question
            )
            latency_ms = round((time.perf_counter() - start) * 1000)
        return PredictResponse(answer=answer, latency_ms=latency_ms, context_truncated=truncated)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned LoRA checkpoint over HTTP")
    parser.add_argument("--config", required=True, help="Path to the training YAML config")
    parser.add_argument("--checkpoint", help="Adapter folder (default: <output_dir>/adapter)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    api_key = os.environ.get("SERVING_API_KEY")
    if args.host not in _LOOPBACK_HOSTS and not api_key:
        parser.error(f"--host {args.host} exposes the endpoint to other machines: set SERVING_API_KEY first.")

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg["training"]["output_dir"]) / "adapter")
    if not Path(checkpoint).is_dir():
        parser.error(f"Adapter folder not found: {checkpoint}")

    uvicorn.run(create_app(cfg, checkpoint, api_key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

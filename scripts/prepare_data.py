"""
scripts/prepare_data.py
------------------------
Download the Phase-1 datasets from the Hugging Face Hub, apply the task-specific
transformations, and write inspectable JSONL files to data/processed/<task>/.

    python scripts/prepare_data.py --config configs/pilot_squad_qa.yaml
    python scripts/prepare_data.py --config configs/pilot_xlsum_summarization.yaml
    python scripts/prepare_data.py --config <cfg> --small     # 50/25/25 quick check

Each JSONL record:
    prompt      inference prompt - ends with the answer/summary header (NO answer)
    reference   gold completion (str)
    references  all gold completions (QA dev sets have several)
    text        prompt + reference + EOS  (what the model is trained on)
    + id / is_answerable (QA)

Split policy is identical to the training pipeline (it calls the same `_subsample`),
so the files match the exact train/val/test examples used for fine-tuning.
Needs: datasets + transformers (tokenizer only - no model weights, no GPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import (  # noqa: E402
    _PROMPT_OVERHEAD_TOKENS,
    _load_raw,
    _subsample,
)
from src.data.prompts import (  # noqa: E402
    UNANSWERABLE,
    build_qa_prompt,
    build_summarization_prompt,
    build_training_text,
    clean_text,
    qa_completion,
    qa_references,
    truncate_to_tokens,
)
from src.utils.config_loader import load_config  # noqa: E402

logger = logging.getLogger("prepare_data")


def _qa_record(ex: dict, tok, ds_cfg: dict) -> dict:
    q_len = len(tok(ex["question"], add_special_tokens=False)["input_ids"])
    budget = max(ds_cfg["max_source_length"] - q_len - _PROMPT_OVERHEAD_TOKENS, 32)
    context = truncate_to_tokens(clean_text(ex["context"]), tok, budget)
    completion = qa_completion(ex["answers"])
    prompt = build_qa_prompt(context, ex["question"])
    return {
        "id": ex["id"],
        "prompt": prompt,
        "reference": completion,
        "references": qa_references(ex["answers"]),
        "is_answerable": completion != UNANSWERABLE,
        # False => answer was trimmed out of the context; such rows are dropped from train
        "answer_in_context": completion == UNANSWERABLE or completion.lower() in context.lower(),
        "text": build_training_text(prompt, completion, tok.eos_token),
    }


def _sum_record(ex: dict, tok, ds_cfg: dict) -> dict:
    article = truncate_to_tokens(
        clean_text(ex[ds_cfg.get("text_column", "text")]),
        tok, ds_cfg["max_source_length"] - _PROMPT_OVERHEAD_TOKENS,
    )
    summary = truncate_to_tokens(
        clean_text(ex[ds_cfg.get("summary_column", "summary")]),
        tok, ds_cfg["max_target_length"] - 1,
    )
    prompt = build_summarization_prompt(article)
    return {
        "id": str(ex.get("id", "")),
        "prompt": prompt,
        "reference": summary,
        "references": [summary],
        "text": build_training_text(prompt, summary, tok.eos_token),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--small", action="store_true", help="50 train / 25 val / 25 test")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    ds_cfg, task = cfg["dataset"], cfg["task"]
    if task not in ("qa", "summarization"):
        sys.exit(f"Phase 1 prepares 'qa' and 'summarization' only (got '{task}').")
    if args.small:
        ds_cfg.update(train_samples=50, val_samples=25, test_samples=25)
        cfg["training"]["output_dir"] = "experiments/prepare_data_small"

    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer_name_or_path") or cfg["model_name_or_path"], trust_remote_code=True
    )
    raw = _subsample(_load_raw(ds_cfg["name"], ds_cfg.get("subset")), ds_cfg, cfg)

    make = _qa_record if task == "qa" else _sum_record
    out_root = Path(args.out_dir) / ("squad_v2" if task == "qa" else "xlsum_en")
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        if split not in raw:
            continue
        rows = [make(ex, tok, ds_cfg) for ex in raw[split]]
        if split == "train" and task == "qa":
            kept = [r for r in rows if r["answer_in_context"]]
            logger.info("train: dropped %d answer-truncated rows", len(rows) - len(kept))
            rows = kept
        path = out_root / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        lens = [len(tok(r["text"])["input_ids"]) for r in rows]
        extra = ""
        if task == "qa":
            c = Counter(r["is_answerable"] for r in rows)
            extra = f" answerable={c[True]} unanswerable={c[False]}"
        logger.info(
            "%-10s n=%d tokens(mean/max)=%.0f/%d limit=%d%s -> %s",
            split, len(rows), sum(lens) / len(lens), max(lens),
            ds_cfg["max_source_length"] + ds_cfg["max_target_length"], extra, path,
        )


if __name__ == "__main__":
    main()

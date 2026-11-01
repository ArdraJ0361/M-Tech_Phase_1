"""
src/data/dataset_loader.py
---------------------------
Unified dataset loader for all four tasks.

Supported task keys: "qa", "summarization", "classification", "extraction"

Each loader returns a HuggingFace DatasetDict with splits:
    train / validation / test

SPLIT PROCEDURE — OFFICIAL BOUNDARIES (guide feedback round 2, point A):

SQuAD v2:
  - Train examples come from the official HuggingFace "train" split only.
  - Validation examples come from the official "validation" split only.
  - Test examples come from the official "validation" split only
    (SQuAD v2 has no public "test" split with answer labels).
  - train and test subsets are drawn from DIFFERENT source splits,
    so there is no leakage between them.
  - The split_source field in test_indices.json records which HF split
    each subset was drawn from.

XL-Sum English:
  - Provides official train / validation / test splits — used directly.
  - No custom re-splitting required.

virattt/financial-qa-10K:
  - Ships a single split, so we split it BY COMPANY (ticker): test companies are
    never seen in training. Uses a fixed seed, so every run gets the same test set.
  - Adds "unanswerable" examples by pairing questions with another company's passage.

FUNSD:
  - Provides official train / test splits — used directly.
  - The test split is small (50 documents); no further subsampling needed.

GENERAL RULES:
  - No example ever appears in more than one of our train/val/test subsets.
  - Subsampling is applied WITHIN each official split, never ACROSS splits.
  - Subsampling uses numpy.random.default_rng(seed) for reproducibility.
  - Test indices are saved to <output_dir>/test_indices.json for audit.

UNANSWERABLE QUESTIONS (SQuAD v2):
  - ~50% of examples have no answer (answers["text"] is empty).
  - These are NOT skipped — the expected output is the string "unanswerable".
  - Metrics are reported for the full mix AND broken down into
    answerable / unanswerable sub-groups.

PHASE 1 SCOPE (guide feedback corrections A, D, E):

  PHASE 1 OBJECTIVE: Evaluate whether PEFT fine-tuning improves a small language
  model on selected TEXT-BASED tasks (QA + summarization), and compare task quality
  and computational efficiency with a larger model. SQuAD v2 uses Wikipedia passages;
  XL-Sum uses news articles. These are text-native pilots — NOT full visual document
  understanding. Scanned-document extraction and classification are Phase 2.

  - qa           : SQuAD v2         (text-native; data pipeline VALIDATED;
                                      training PENDING; evaluation PENDING;
                                      FIRST training pilot)
  - summarization: XL-Sum (English) (text-native; data pipeline VALIDATED;
                                      training PENDING; evaluation PENDING;
                                      SECOND pilot after SQuAD v2 is confirmed)
  - extraction   : FUNSD            (Phase 2 — PLANNED: data pipeline implemented,
                                      extraction experiment and evaluation deferred;
                                      spatial layout LOST in text-only mode)
  - classification: RVL-CDIP       (Phase 2 ONLY — requires OCR pre-extraction)
  - DocVQA                          (Phase 2 ONLY — image-based, requires OCR)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .prompts import (
    UNANSWERABLE,
    build_qa_prompt,
    build_summarization_prompt,
    build_training_text,
    clean_text,
    qa_completion,
    truncate_to_tokens,
)

logger = logging.getLogger(__name__)

_PHASE1_TASKS   = {"qa", "summarization"}
_PHASE2_PLANNED = {"extraction"}   # pipeline built; experiment and evaluation deferred to Phase 2
_PHASE2_TASKS   = {"classification"}

# Legacy bare dataset names are rejected by current huggingface_hub
# ("Repository id must be 'namespace/name'"), so map them to canonical Hub ids.
_DATASET_ALIASES = {"squad_v2": "rajpurkar/squad_v2"}

# The Hub's csebuetnlp/xlsum repo still ships a loading script (xlsum.py), which
# `datasets` >= 4 refuses to run. Hub's auto-converted parquet branch contains
# the same data with the same official train/validation/test splits.
_XLSUM_REPO = "csebuetnlp/xlsum"
_XLSUM_PARQUET = "hf://datasets/csebuetnlp/xlsum@refs%2Fconvert%2Fparquet"

# Financial sentence-answer QA from 10-K filings. Revision pinned so the split is reproducible.
_FIN10K_REPO = "virattt/financial-qa-10K"
_FIN10K_REVISION = "30e48b49ef40389cce6488b0bddef7d6f00f4c7a"
# Fixed, NOT training.seed: extra-seed runs must still score the same test companies.
_FIN10K_SPLIT_SEED = 42
_FIN10K_HELDOUT_TICKERS = 7        # for each of validation and test (~10% of 69 companies)
_FIN10K_UNANSWERABLE_RATIO = 0.2   # added per split, relative to its answerable rows

# Token budget reserved for the "### ... :" headers and newlines of a prompt,
# so the (possibly truncated) body + headers never exceed max_source_length.
_PROMPT_OVERHEAD_TOKENS = 24

# Documents which official HF split each of our roles is drawn from.
# This is logged to test_indices.json for full audit trail.
_SQUAD_SPLIT_SOURCE = {
    "train": "train",        # HF train split → our training set
    "validation": "validation",  # HF validation split → our validation set
    # IMPORTANT: SQuAD v2 has no public test split with labels.
    # We draw our "test" subset from the official HF validation split.
    # This means validation and test subsets come from the same HF split
    # but are DISJOINT subsets (different randomly-selected indices).
    "test": "validation",
}
_HF_SPLIT_SOURCE = {
    "squad_v2": _SQUAD_SPLIT_SOURCE,
    "rajpurkar/squad_v2": _SQUAD_SPLIT_SOURCE,
    # For datasets that provide all three official splits, use them directly.
    "_default": {
        "train": "train",
        "validation": "validation",
        "test": "test",
    },
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset_for_task(cfg: dict, tokenizer=None) -> "DatasetDict":  # type: ignore[name-defined]
    """
    Load and preprocess a dataset according to the task specified in cfg.

    Parameters
    ----------
    cfg       : dict  — experiment config (from YAML)
    tokenizer : PreTrainedTokenizer — required for tokenization step

    Returns
    -------
    datasets.DatasetDict  with keys "train", "validation", "test"
    """
    task = cfg["task"]
    ds_cfg = cfg["dataset"]

    if task in _PHASE2_TASKS:
        logger.warning(
            "Task '%s' is Phase 2 scope (requires OCR pre-extraction + image datasets). "
            "Phase 1 tasks (data pipeline VALIDATED): %s",
            task, sorted(_PHASE1_TASKS),
        )
    elif task in _PHASE2_PLANNED:
        logger.warning(
            "Task '%s' is Phase 2 — PLANNED: the data pipeline is implemented but "
            "the extraction experiment and evaluation are deferred to Phase 2. "
            "Text-only processing does not preserve the original spatial layout of forms. "
            "Do not claim full document-image understanding from text-only results.",
            task,
        )

    if task == "qa":
        return _load_qa(ds_cfg, tokenizer, cfg)
    elif task == "summarization":
        return _load_summarization(ds_cfg, tokenizer, cfg)
    elif task == "classification":
        return _load_classification(ds_cfg, tokenizer, cfg)
    elif task == "extraction":
        return _load_extraction(ds_cfg, tokenizer, cfg)
    else:
        raise ValueError(
            f"Unknown task: '{task}'. Choose from: qa, summarization, classification, extraction"
        )


# ---------------------------------------------------------------------------
# Raw download from the Hugging Face Hub
# ---------------------------------------------------------------------------

def _load_raw(name: str, subset: str | None = None):
    """Download a dataset from the Hub (cached locally by `datasets`)."""
    from datasets import load_dataset

    name = _DATASET_ALIASES.get(name, name)
    if name == _FIN10K_REPO:
        return _load_fin10k()
    if name == _XLSUM_REPO:
        lang = subset or "english"
        base = f"{_XLSUM_PARQUET}/{lang}"
        return load_dataset(
            "parquet",
            data_files={s: f"{base}/{s}/*.parquet" for s in ("train", "validation", "test")},
        )
    return load_dataset(name, subset) if subset else load_dataset(name)


def _load_fin10k():
    """Load financial-qa-10K in the SQuAD schema, so `_load_qa` handles it unchanged."""
    from datasets import load_dataset

    raw = load_dataset(_FIN10K_REPO, revision=_FIN10K_REVISION, split="train")
    raw = raw.filter(lambda e: all((e[k] or "").strip() for k in ("question", "answer", "context")))
    return _split_fin10k(raw)


def _split_fin10k(raw):
    """Split rows by ticker into train/validation/test, then add unanswerable examples."""
    import numpy as np
    from datasets import Dataset, DatasetDict

    rng = np.random.default_rng(_FIN10K_SPLIT_SEED)
    tickers = rng.permutation(sorted(set(raw["ticker"]))).tolist()
    n = _FIN10K_HELDOUT_TICKERS
    groups = {
        "validation": set(tickers[:n]),
        "test": set(tickers[n:2 * n]),
        "train": set(tickers[2 * n:]),
    }
    logger.info("financial-qa-10K test companies: %s", sorted(groups["test"]))

    result = {}
    for role, group in groups.items():
        rows = [r for r in raw if r["ticker"] in group]
        records = [
            {"context": r["context"], "question": r["question"], "answers": {"text": [r["answer"]]}}
            for r in rows
        ]
        n_unanswerable = int(len(rows) * _FIN10K_UNANSWERABLE_RATIO)
        for i in rng.choice(len(rows), size=n_unanswerable, replace=False).tolist():
            records.append({
                "context": _pick_unrelated_context(rows, rows[i], rng),
                "question": rows[i]["question"],
                "answers": {"text": []},
            })
        result[role] = Dataset.from_list(records)
        logger.info("%s: %d answerable + %d unanswerable", role, len(rows), n_unanswerable)
    return DatasetDict(result)


def _pick_unrelated_context(rows: list, row: dict, rng) -> str:
    """A passage from a different company in the same split that lacks this row's answer."""
    answer = row["answer"].lower()
    candidates = [
        r["context"] for r in rows
        if r["ticker"] != row["ticker"] and answer not in r["context"].lower()
    ]
    return candidates[int(rng.integers(len(candidates)))]


# ---------------------------------------------------------------------------
# Task-specific loaders
# ---------------------------------------------------------------------------

def _load_qa(ds_cfg: dict, tokenizer, cfg: dict):
    """
    Load SQuAD v2 for question answering.

    SPLIT BOUNDARY POLICY (guide feedback round 2, point A):
      - Training examples: drawn from HF "train" split only.
      - Validation examples: drawn from HF "validation" split only.
      - Test examples: drawn from HF "validation" split only
        (SQuAD v2 has no public labelled "test" split).
      - Validation and test subsets are DISJOINT — different randomly selected
        indices from the same HF validation split, with no overlap guaranteed
        by using sorted, non-overlapping index ranges.

    UNANSWERABLE QUESTIONS:
      - ~50% of SQuAD v2 validation examples are unanswerable.
      - These are included in ALL splits (train, val, test).
      - Expected output: the literal string "unanswerable".
      - Metrics report full mix + answerable/unanswerable breakdown.
    """
    from .preprocessor import tokenize_qa

    name = ds_cfg.get("name", "rajpurkar/squad_v2")
    subset = ds_cfg.get("subset")
    logger.info("Loading QA dataset: %s (subset=%s)", name, subset)

    raw = _load_raw(name, subset)
    raw = _subsample(raw, ds_cfg, cfg)

    if tokenizer is None:
        return raw

    max_src = ds_cfg["max_source_length"]
    # Datasets with reworded (not copied) answers set this to false.
    require_answer_in_context = ds_cfg.get("require_answer_in_context", True)
    prompt_overhead = len(tokenizer(build_qa_prompt("", ""), add_special_tokens=False)["input_ids"])

    def _format(example):
        # Trim the CONTEXT (not the whole sequence) so the question, answer
        # header, answer and EOS always survive tokenization.
        q_len = len(tokenizer(example["question"], add_special_tokens=False)["input_ids"])
        budget = max(max_src - q_len - prompt_overhead, 32)
        context = truncate_to_tokens(clean_text(example["context"]), tokenizer, budget)

        answer_text = qa_completion(example["answers"])
        # Training guard: an answerable example whose answer was trimmed out of
        # the context would teach the model to hallucinate - flag it for removal.
        keep = (
            not require_answer_in_context
            or answer_text == UNANSWERABLE
            or answer_text.lower() in context.lower()
        )

        prompt = build_training_text(
            build_qa_prompt(context, example["question"]), answer_text, tokenizer.eos_token
        )
        return {"prompt": prompt, "answer": answer_text, "keep": keep}

    formatted = raw.map(_format, remove_columns=raw["train"].column_names)
    n_before = len(formatted["train"])
    formatted["train"] = formatted["train"].filter(lambda e: e["keep"])
    if len(formatted["train"]) < n_before:
        logger.info(
            "Dropped %d/%d train examples whose answer was cut off by context truncation.",
            n_before - len(formatted["train"]), n_before,
        )
    formatted = formatted.remove_columns("keep")
    tokenized = formatted.map(
        lambda ex: tokenize_qa(ex, tokenizer, cfg),
        batched=True,
        remove_columns=["prompt", "answer"],
    )
    return tokenized


def _load_summarization(ds_cfg: dict, tokenizer, cfg: dict):
    """
    Load XL-Sum (English) for summarization.

    XL-Sum provides official train/validation/test splits — used directly.
    No custom re-splitting is performed.

    Checkpoint selection uses validation ROUGE-L (set in config as
    metric_for_best_model: "eval_rougeL").
    """
    from .preprocessor import tokenize_seq2seq

    name = ds_cfg.get("name", "csebuetnlp/xlsum")
    subset = ds_cfg.get("subset", "english")
    logger.info("Loading summarization dataset: %s (subset=%s)", name, subset)

    raw = _load_raw(name, subset)
    raw = _subsample(raw, ds_cfg, cfg)

    if tokenizer is None:
        return raw

    max_src = ds_cfg["max_source_length"]
    max_tgt = ds_cfg["max_target_length"]

    def _format(example):
        # News articles are long: trim the ARTICLE so the "### Summary:" header,
        # the summary and EOS are never truncated away.
        article = truncate_to_tokens(
            clean_text(example[ds_cfg.get("text_column", "text")]),
            tokenizer, max_src - _PROMPT_OVERHEAD_TOKENS,
        )
        # Reserve one token for EOS inside the target budget.
        summary = truncate_to_tokens(
            clean_text(example[ds_cfg.get("summary_column", "summary")]),
            tokenizer, max_tgt - 1,
        )
        prompt = build_training_text(
            build_summarization_prompt(article), summary, tokenizer.eos_token
        )
        return {"prompt": prompt, "summary": summary}

    formatted = raw.map(_format, remove_columns=raw["train"].column_names)
    tokenized = formatted.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer, cfg),
        batched=True,
        remove_columns=["prompt", "summary"],
    )
    return tokenized


def _load_classification(ds_cfg: dict, tokenizer, cfg: dict):
    """
    Load RVL-CDIP for document classification.

    PHASE 2 TASK — requires pre-extracted OCR text in a "text" column.
    If absent, a warning is raised and results will not be meaningful.
    """
    from datasets import load_dataset
    from .preprocessor import tokenize_classification

    name = ds_cfg.get("name", "aharley/rvl_cdip")
    subset = ds_cfg.get("subset")
    logger.info("Loading classification dataset: %s (subset=%s)", name, subset)

    raw = load_dataset(name, subset) if subset else load_dataset(name)
    raw = _subsample(raw, ds_cfg, cfg)

    if tokenizer is None:
        return raw

    label_names = raw["train"].features["label"].names

    def _format(example):
        label_str = label_names[example["label"]]
        text = example.get("text", "")
        if not text:
            logger.warning(
                "Classification example has no 'text' column. "
                "Pre-extract OCR text from document images before training. "
                "Results will NOT be meaningful with placeholder text."
            )
            text = "[OCR text not available — pre-extract before training]"
        prompt = (
            f"### Document:\n{text}\n\n"
            f"### Category:\n{label_str}"
        )
        return {"prompt": prompt, "label": label_str}

    formatted = raw.map(_format, remove_columns=raw["train"].column_names)
    tokenized = formatted.map(
        lambda ex: tokenize_classification(ex, tokenizer, cfg),
        batched=True,
        remove_columns=["prompt", "label"],
    )
    return tokenized


def _load_extraction(ds_cfg: dict, tokenizer, cfg: dict):
    """
    Load FUNSD for information extraction.

    PHASE 2 — PLANNED (guide feedback correction A):
    The data pipeline is implemented and the dataset can be loaded, but the
    extraction experiment and evaluation are deferred to Phase 2. This function
    is available for pipeline testing but should NOT be used to report Phase 1
    results.

    Key limitations (unchanged):

    1. SPATIAL INFORMATION IS LOST: FUNSD annotates words with their
       bounding boxes and spatial relationships on the form. Converting
       the form to a flat text string discards this layout information.
       The model receives only word-order text, not a 2D form layout.

    2. This is a TEXT-ONLY approximation of the extraction task.
       Do NOT claim this evaluates full document-image understanding
       or that results are comparable with vision-language models that
       see the full form image.

    3. The FUNSD test split is small (50 documents). Results on this
       split should be interpreted cautiously and validated manually
       on a sample of examples before reporting.

    FUNSD NER tag mapping:
      O=0, B-HEADER=1, I-HEADER=2, B-QUESTION=3, I-QUESTION=4,
      B-ANSWER=5, I-ANSWER=6
    """
    from datasets import load_dataset
    from .preprocessor import tokenize_seq2seq

    name = ds_cfg.get("name", "nielsr/funsd")
    subset = ds_cfg.get("subset")
    logger.info("Loading extraction dataset: %s (subset=%s)", name, subset)

    raw = load_dataset(name, subset) if subset else load_dataset(name)
    raw = _subsample(raw, ds_cfg, cfg)

    if tokenizer is None:
        return raw

    def _format(example):
        words = example.get("words", [])
        ner_tags = example.get("ner_tags", [])
        text = " ".join(words)
        field_pairs = _extract_qa_pairs(words, ner_tags)
        output = "\n".join(f"{q}: {a}" for q, a in field_pairs) if field_pairs else "[no fields]"
        prompt = (
            f"### Form text:\n{text}\n\n"
            f"### Extracted fields (key: value):\n{output}"
        )
        return {"prompt": prompt, "output": output}

    formatted = raw.map(_format, remove_columns=raw["train"].column_names)
    tokenized = formatted.map(
        lambda ex: tokenize_seq2seq(ex, tokenizer, cfg),
        batched=True,
        remove_columns=["prompt", "output"],
    )
    return tokenized


def _extract_qa_pairs(words: list, ner_tags: list) -> list[tuple[str, str]]:
    """Extract (question, answer) pairs from FUNSD NER tags."""
    pairs = []
    i = 0
    while i < len(words):
        if ner_tags[i] == 3:  # B-QUESTION
            q_tokens = [words[i]]
            i += 1
            while i < len(words) and ner_tags[i] == 4:
                q_tokens.append(words[i])
                i += 1
            a_tokens = []
            while i < len(words) and ner_tags[i] in (5, 6):
                a_tokens.append(words[i])
                i += 1
            if a_tokens:
                pairs.append((" ".join(q_tokens), " ".join(a_tokens)))
        else:
            i += 1
    return pairs


# ---------------------------------------------------------------------------
# Deterministic subsampling — respects official HF split boundaries
# ---------------------------------------------------------------------------

def _subsample(raw_dataset, ds_cfg: dict, cfg: dict):
    """
    Subsample each split using a fixed seed.

    SPLIT BOUNDARY RULES (guide feedback round 2, point A):

    For SQuAD v2 specifically:
      - "train" subset  ← drawn from HF "train"      split
      - "validation"    ← drawn from HF "validation" split
      - "test"          ← drawn from HF "validation" split
        (no public SQuAD v2 test labels exist)

    To ensure val and test are DISJOINT when both come from the same HF split:
      - We draw (val_samples + test_samples) indices at once without replacement.
      - The first val_samples become validation; the remaining become test.
      - This guarantees zero overlap.

    For datasets with all three official splits (XL-Sum, FUNSD):
      - Each role is drawn from its own dedicated HF split.
      - No overlap is possible by construction.

    Indices are saved to test_indices.json for full audit.
    """
    import numpy as np
    from datasets import DatasetDict

    seed = cfg.get("training", {}).get("seed", 42)
    rng = np.random.default_rng(seed)

    dataset_name = ds_cfg.get("name", "")
    split_sources = _HF_SPLIT_SOURCE.get(dataset_name, _HF_SPLIT_SOURCE["_default"])

    n_train = ds_cfg.get("train_samples")
    n_val   = ds_cfg.get("val_samples")
    n_test  = ds_cfg.get("test_samples")

    result = {}

    # -----------------------------------------------------------------------
    # Case 1: validation and test both drawn from the same HF split (SQuAD v2)
    # -----------------------------------------------------------------------
    val_src  = split_sources["validation"]
    test_src = split_sources["test"]

    if val_src == test_src and val_src in raw_dataset:
        # Draw val + test indices together without replacement to guarantee no overlap
        shared_split = raw_dataset[val_src]
        total_needed = (n_val or 0) + (n_test or 0)

        if total_needed > 0 and total_needed < len(shared_split):
            all_indices = rng.choice(len(shared_split), size=total_needed, replace=False).tolist()
            all_indices.sort()
            val_indices  = all_indices[:n_val]  if n_val  else all_indices
            test_indices = all_indices[n_val:]  if n_val and n_test else []
        else:
            # Use entire shared split, split it deterministically
            all_indices  = list(range(len(shared_split)))
            val_indices  = all_indices[:n_val]  if n_val  else all_indices
            test_indices = all_indices[n_val:n_val + (n_test or 0)] if n_test else []

        if val_indices:
            result["validation"] = shared_split.select(val_indices)
            logger.info(
                "Validation: %d examples from HF '%s' split (seed=%d)",
                len(val_indices), val_src, seed,
            )
        if test_indices:
            result["test"] = shared_split.select(test_indices)
            logger.info(
                "Test: %d examples from HF '%s' split — DISJOINT from validation (seed=%d)",
                len(test_indices), test_src, seed,
            )
            _save_test_indices(test_indices, cfg, seed, hf_split=test_src)

        # Confirm zero overlap
        if val_indices and test_indices:
            overlap = set(val_indices) & set(test_indices)
            assert len(overlap) == 0, (
                f"BUG: {len(overlap)} examples appear in both validation and test sets. "
                "This indicates a split implementation error."
            )
            logger.info("Split integrity check: validation and test sets have ZERO overlap.")
    else:
        # Case 2: separate HF splits for val and test (XL-Sum, FUNSD, etc.)
        for role, hf_split_name in [("validation", val_src), ("test", test_src)]:
            if hf_split_name not in raw_dataset:
                logger.warning("HF split '%s' not found for role '%s'.", hf_split_name, role)
                continue
            ds = raw_dataset[hf_split_name]
            n  = n_val if role == "validation" else n_test
            if n is not None and n < len(ds):
                indices = rng.choice(len(ds), size=n, replace=False).tolist()
                indices.sort()
                ds = ds.select(indices)
                if role == "test":
                    _save_test_indices(indices, cfg, seed, hf_split=hf_split_name)
            result[role] = ds

    # -----------------------------------------------------------------------
    # Training split — always from HF "train"
    # -----------------------------------------------------------------------
    train_src = split_sources.get("train", "train")
    if train_src in raw_dataset:
        ds = raw_dataset[train_src]
        if n_train is not None and n_train < len(ds):
            indices = rng.choice(len(ds), size=n_train, replace=False).tolist()
            indices.sort()
            ds = ds.select(indices)
            logger.info(
                "Train: %d examples from HF '%s' split (seed=%d)",
                n_train, train_src, seed,
            )
        result["train"] = ds
    else:
        logger.warning("No training split found for dataset.")

    return DatasetDict(result)


def _save_test_indices(indices: list, cfg: dict, seed: int, hf_split: str = "unknown") -> None:
    """Save test split indices to disk for reproducibility auditing."""
    output_dir = Path(cfg.get("training", {}).get("output_dir", "experiments/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "test_indices.json"
    payload = {
        "seed": seed,
        "dataset": cfg["dataset"].get("name", "unknown"),
        "dataset_version": cfg["dataset"].get("version", "default"),
        "task": cfg.get("task", "unknown"),
        "hf_source_split": hf_split,
        "num_test_examples": len(indices),
        "indices": indices,
        "note": (
            "For SQuAD v2: test indices are from the HF 'validation' split. "
            "Validation indices use a disjoint subset of the same split. "
            "Zero overlap verified at runtime by assertion."
        ),
    }
    with index_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Test indices saved to %s (hf_split='%s', n=%d)",
        index_file, hf_split, len(indices),
    )

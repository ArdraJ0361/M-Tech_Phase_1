"""
src/data/preprocessor.py
--------------------------
Tokenization helpers for causal language model fine-tuning.

Each function takes a batch of examples and returns a dict with:
    input_ids, attention_mask, labels

CRITICAL — PROMPT MASKING (guide feedback on training correctness):

For causal LM fine-tuning, the loss must ONLY be computed on the completion
tokens (answer / summary / label), NOT on the prompt tokens.

We enforce this by:
1. Tokenizing the full sequence (prompt + completion).
2. Separately tokenizing only the prompt portion.
3. Setting labels[0 : prompt_len] = IGNORE_INDEX (-100).
4. Adding an assertion that at least one completion token remains after masking.
   If all tokens are masked (e.g. the completion was truncated away), the
   example would contribute zero signal and the assertion catches this.

IMPORTANT: The separator strings used in _extract_prompt_part() must exactly
match the strings used in dataset_loader.py prompt templates. If you change
a prompt template there, update the separators list here too.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_IGNORE_INDEX = -100

# These must match the last header line in each prompt template in dataset_loader.py
_COMPLETION_SEPARATORS = [
    "**Answer:**\n",       # financial QA prompt (instruction-style)
    "### Answer:\n",       # legacy SQuAD prompt — kept for backwards compat
    "### Summary:\n",
    "### Category:\n",
    "### Extracted fields (key: value):\n",
]


def tokenize_qa(examples: dict, tokenizer, cfg: dict) -> dict:
    """Tokenize QA prompt+answer for causal LM."""
    return _tokenize_prompt_completion(
        examples,
        tokenizer,
        max_length=cfg["dataset"]["max_source_length"] + cfg["dataset"]["max_target_length"],
    )


def tokenize_seq2seq(examples: dict, tokenizer, cfg: dict) -> dict:
    """Tokenize source+target for summarization or extraction via causal LM."""
    return _tokenize_prompt_completion(
        examples,
        tokenizer,
        max_length=cfg["dataset"]["max_source_length"] + cfg["dataset"]["max_target_length"],
    )


def tokenize_classification(examples: dict, tokenizer, cfg: dict) -> dict:
    """Tokenize document+label for classification via causal LM."""
    return _tokenize_prompt_completion(
        examples,
        tokenizer,
        max_length=cfg["dataset"]["max_source_length"] + cfg["dataset"].get("max_target_length", 16),
    )


# ---------------------------------------------------------------------------
# Core tokenization logic
# ---------------------------------------------------------------------------

def _tokenize_prompt_completion(
    examples: dict,
    tokenizer,
    max_length: int,
) -> dict:
    """
    Tokenize prompt+completion pairs for causal LM fine-tuning.

    Four masking conditions verified (guide feedback round 2, point C):

    1. PROMPT TOKENS → IGNORED: labels[0 : prompt_len] = IGNORE_INDEX (-100).
       These tokens do NOT contribute to the loss.

    2. COMPLETION TOKENS → TRAINED: labels[prompt_len :] = actual token ids.
       Only these tokens contribute to the loss.

    3. PADDING TOKENS → IGNORED: DataCollatorForSeq2Seq sets padding
       positions to IGNORE_INDEX. Padding never contributes to loss.

    4. TRUNCATION SAFETY: if truncation removes all completion tokens
       (prompt alone fills max_length), the example is SKIPPED entirely.
       It is NOT trained on — not even on the prompt tokens.
       This is the safe behaviour: it is better to skip an example than
       to silently train on the wrong tokens.

    The skipped-example count is accumulated and logged at the end
    so the caller can see if max_length needs to be increased.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    skipped = 0

    prompts = (
        examples["prompt"]
        if isinstance(examples["prompt"], list)
        else [examples["prompt"]]
    )

    for prompt in prompts:
        # Step 1: tokenize full sequence (prompt + completion)
        full = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        # Step 2: tokenize only the prompt portion to find the boundary
        prompt_part = _extract_prompt_part(prompt)
        prompt_enc = tokenizer(
            prompt_part,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        prompt_len = len(prompt_enc["input_ids"])

        # Step 3: mask prompt tokens — they must NOT contribute to loss
        labels = list(input_ids)
        mask_up_to = min(prompt_len, len(labels))
        for i in range(mask_up_to):
            labels[i] = _IGNORE_INDEX

        # Step 4: safety check — are any completion tokens present?
        # This catches cases where truncation removed the answer entirely.
        completion_tokens = [t for t in labels if t != _IGNORE_INDEX]
        if not completion_tokens:
            # SKIP this example entirely — do NOT fall back to training on
            # the prompt tokens. Training on prompt tokens would:
            #   (a) violate the supervised fine-tuning objective, and
            #   (b) teach the model to generate prompts, not answers.
            logger.warning(
                "SKIPPING example — no completion tokens after prompt masking. "
                "Truncation removed the answer before it could be tokenized. "
                "Fix: increase max_source_length (prompt=%d tokens) or "
                "max_target_length. Total sequence=%d, max_length=%d.",
                prompt_len, len(input_ids), max_length,
            )
            skipped += 1
            continue  # do not append — example is dropped

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    if skipped > 0:
        logger.warning(
            "%d / %d examples skipped due to truncation removing completion tokens. "
            "Consider increasing max_source_length or max_target_length in your config.",
            skipped, len(prompts),
        )

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def _extract_prompt_part(full_prompt: str) -> str:
    """
    Extract the prompt portion up to and including the last completion header.

    This is the text whose token length determines where masking ends.
    The separators here MUST match the prompt templates in dataset_loader.py.

    Example:
        full_prompt = "### Context:\\nParis...\\n\\n### Question:\\nWhere?\\n\\n### Answer:\\nParis"
        returns    → "### Context:\\nParis...\\n\\n### Question:\\nWhere?\\n\\n### Answer:\\n"
                      ↑ everything up to and including the header line
                      The completion "Paris" is NOT included → its tokens are not masked.
    """
    for sep in _COMPLETION_SEPARATORS:
        if sep in full_prompt:
            idx = full_prompt.rfind(sep) + len(sep)
            return full_prompt[:idx]
    # Fallback: no separator found — treat whole string as prompt.
    # This means the entire sequence will be masked and will log a warning above.
    logger.warning(
        "No completion separator found in prompt. "
        "Expected one of: %s. "
        "All tokens will be masked — no training signal for this example.",
        _COMPLETION_SEPARATORS,
    )
    return full_prompt


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

def get_data_collator(tokenizer):
    """
    Return a data collator that pads sequences to the batch maximum length
    and replaces padding positions in labels with IGNORE_INDEX.

    Using DataCollatorForSeq2Seq rather than the default collator ensures
    the label padding token is set to -100 (not the pad_token_id), so padded
    positions never contribute to the training loss.
    """
    from transformers import DataCollatorForSeq2Seq
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=_IGNORE_INDEX,
        pad_to_multiple_of=8,   # speeds up tensor core utilisation on modern GPUs
    )

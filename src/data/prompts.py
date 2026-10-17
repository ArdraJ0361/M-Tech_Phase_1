"""
src/data/prompts.py
--------------------
Task-specific prompt builders. Dependency-free (no torch / datasets) so they can
be imported by the loader, the data-prep script and the evaluator alike.

Contract
--------
A training text is always:   <prompt> + <completion> + <eos>
An inference text is:        <prompt>                       (model generates the rest)

The LAST header of every prompt must appear in
`preprocessor._COMPLETION_SEPARATORS` - that is how prompt masking finds the
boundary between "masked" and "trained" tokens.
"""

from __future__ import annotations

import re

UNANSWERABLE = "unanswerable"

QA_SEPARATOR = "### Answer:\n"
SUMMARY_SEPARATOR = "### Summary:\n"

_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise whitespace: collapse runs of spaces/tabs, cap blank lines, strip."""
    text = (text or "").replace("\u00a0", " ").replace("\u200b", "")
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# --------------------------------------------------------------------------- QA
def build_qa_prompt(context: str, question: str) -> str:
    """Prompt for SQuAD v2 style extractive QA (ends with the answer header)."""
    return (
        f"### Context:\n{clean_text(context)}\n\n"
        f"### Question:\n{clean_text(question)}\n\n"
        f"{QA_SEPARATOR}"
    )


def qa_completion(answers: dict) -> str:
    """First gold answer, or the literal 'unanswerable' when SQuAD v2 has none."""
    texts = answers.get("text") or []
    return clean_text(texts[0]) if texts else UNANSWERABLE


def qa_references(answers: dict) -> list[str]:
    """All gold answers (SQuAD v2 dev has several) - or ['unanswerable']."""
    texts = [clean_text(t) for t in (answers.get("text") or [])]
    return texts or [UNANSWERABLE]


# ----------------------------------------------------------------- Summarisation
def build_summarization_prompt(article: str) -> str:
    """Prompt for XL-Sum style summarisation (ends with the summary header)."""
    return f"### Article:\n{clean_text(article)}\n\n{SUMMARY_SEPARATOR}"


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """
    Trim `text` to at most `max_tokens` tokens (decoded back to text).

    Needed for long news articles: truncating the *whole* sequence would cut
    off the summary header + summary and the example would be skipped.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).rstrip()


def build_training_text(prompt: str, completion: str, eos_token: str | None) -> str:
    """prompt + completion + eos. The eos teaches the model WHEN TO STOP."""
    return f"{prompt}{completion}{eos_token or ''}"

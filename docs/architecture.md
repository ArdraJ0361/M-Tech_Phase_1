# Architecture — Understanding the Project from the Ground Up

> **This branch holds fine-tuning and inference only.** SQuAD, XL-Sum, the Colab notebook and PDF
> extraction are on `feat/joel-changes`, so links to them below may not resolve here.

## M.Tech: Parameter-Efficient Fine-Tuning of Small Language Models for Document Analysis

---

## Part 1 — What Problem Are We Solving?

Organizations deal with documents every day — invoices, reports, forms, research papers,
policy documents. Someone needs to answer questions from these documents, summarize them,
classify them by type, or extract specific fields.

**Large Language Models (LLMs)** like GPT-4 or Llama-70B can do these tasks well — but they
are expensive. They need large GPUs, take longer to run, and cost more to deploy.

**Small Language Models (SLMs)** are cheaper and faster — but they are less capable out of
the box. A 0.5B model asked a question zero-shot might not give a useful answer.

**The question this project asks:**

> If we take a small model and *fine-tune* it specifically for document tasks using a
> technique called **PEFT (Parameter-Efficient Fine-Tuning)**, can it get close to a
> large model's performance — at a fraction of the cost?

That is the entire research question. Everything else in the project is set up to answer it.

---

## Part 2 — The Three-Way Comparison

To answer that question, we run a controlled experiment with **three setups**:

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│  Setup 1             │   │  Setup 2             │   │  Setup 3             │
│                      │   │                      │   │                      │
│  Small model         │   │  Small model         │   │  Large model         │
│  (Qwen1.5-0.5B)      │   │  (Qwen1.5-0.5B)      │   │  (Qwen1.5-7B)        │
│                      │   │  + LoRA adapter      │   │                      │
│  No training         │   │  Fine-tuned on       │   │  No training         │
│  Zero-shot only      │   │  task examples       │   │  Zero-shot only      │
│                      │   │                      │   │                      │
│  "How bad is a       │   │  ← YOUR              │   │  "How good is a      │
│  small model before  │   │    CONTRIBUTION      │   │  large model         │
│  any tuning?"        │   │                      │   │  without training?"  │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
        ↑                           ↑                           ↑
   Baseline                  What improves                 Reference
   (starting point)          with fine-tuning              (comparison target)
```

**Why this specific design?**

- Comparing Setup 1 vs Setup 2 tells us: *Does LoRA fine-tuning actually help?*
- Comparing Setup 2 vs Setup 3 tells us: *How close can a small tuned model get to a big model?*
- All three use the **same model family** (Qwen1.5), the **same base variant**, and are tested
  on the **same 500 examples per task** — so the comparison is fair.

---

## Part 3 — What is PEFT and Why LoRA?

### The problem with normal fine-tuning

A pretrained model like Qwen1.5-0.5B has ~500 million numerical values called **parameters**.
These were learned by training on huge amounts of text.

When you **fine-tune** a model on a new task, you update these parameters to make the model
better at that task. But updating 500 million parameters requires a lot of GPU memory — and
you need to store a separate 500M-parameter model for every task.

### What PEFT does differently

PEFT says: **don't update all 500M parameters**. Instead, keep them all frozen and add a
small extra component that learns the task-specific behaviour.

**LoRA (Low-Rank Adaptation)** is the PEFT technique we use. Here is how it works:

```
Normal weight matrix W in the transformer:
┌──────────────────────────────────────────────┐
│         W  (500M params total, FROZEN)        │
│         never updated during training         │
└──────────────────────────────────────────────┘

LoRA adds two small matrices alongside W:
┌──────────────────────────────────────────────┐
│    A  (small matrix, e.g. 576 × 8)           │  ← trainable
│    B  (small matrix, e.g. 8 × 576)           │  ← trainable
└──────────────────────────────────────────────┘

Output = W·x  +  B·A·x · (alpha / rank)
           ↑           ↑
      original    task-specific
      frozen       learned part
```

The "rank" (r=8 in our config) controls how small the adapter is. With r=8:
- Total parameters: **464,774,144** (frozen) — confirmed on Qwen1.5-0.5B
- LoRA adapter parameters: **786,432** (trainable) — confirmed on Colab T4
- Trainable fraction: **0.169%** of the total (measured, not estimated)
- You only need to store and update the small adapter, not the full model

**Why this matters for the project:**
- Much less GPU memory needed during training
- Training is faster
- The base model stays unchanged — you can attach different adapters for different tasks
- This efficiency saving is *measured* in the experiments (training time, GPU memory)

---

## Part 4 — The Full System Pipeline

Here is how data flows through the whole system, from raw document to final result:

```
                    ┌─────────────────────────────────────────────┐
                    │             DOCUMENT INPUT                   │
                    └────────────────┬──────────────┬─────────────┘
                                     │              │
                    ┌────────────────▼──┐    ┌──────▼────────────────┐
                    │  Native document  │    │  Scanned document      │
                    │  PDF with text    │    │  PDF image / JPEG      │
                    │  layer or DOCX    │    │  no text layer         │
                    └────────────────┬──┘    └──────┬────────────────┘
                                     │              │
                    ┌────────────────▼──┐    ┌──────▼────────────────┐
                    │  Direct extract   │    │  OCR engine            │
                    │  pdfplumber       │    │  Tesseract             │
                    │  python-docx      │    │  per-page fallback     │
                    └────────────────┬──┘    └──────┬────────────────┘
                                     │              │
                                     └──────┬───────┘
                                            │
                    ┌───────────────────────▼─────────────────────────┐
                    │               UNIFIED TEXT PIPELINE              │
                    │  - Cleans whitespace, form-feed characters       │
                    │  - Tags source: "native_pdf" / "ocr" / "mixed"  │
                    │  - Returns quality flag: good / low / empty      │
                    │  - Splits long docs into overlapping chunks      │
                    └───────────────────────┬─────────────────────────┘
                                            │
                    ┌───────────────────────▼─────────────────────────┐
                    │           TASK-SPECIFIC PROMPT BUILDING          │
                    │                                                  │
                    │  QA:           "### Context:\n...\n              │
                    │                 ### Question:\n...\n             │
                    │                 ### Answer:\n..."                │
                    │                                                  │
                    │  Summarization: "### Article:\n...\n             │
                    │                  ### Summary:\n..."              │
                    └───────────────────────┬─────────────────────────┘
                                            │
                    ┌───────────────────────▼─────────────────────────┐
                    │               TOKENIZATION + MASKING             │
                    │  Full prompt → tokens                            │
                    │  Prompt portion → labels = -100 (masked)         │
                    │  Answer portion → labels = actual token IDs      │
                    │  Truncated examples → SKIPPED (not trained on)   │
                    └───────────────────────┬─────────────────────────┘
                                            │
                         ┌──────────────────┴──────────────────┐
                         │                                      │
              ┌──────────▼──────────┐             ┌────────────▼───────────┐
              │   PEFT TRAINING     │             │    LARGER-MODEL        │
              │   Qwen1.5-0.5B      │             │    BASELINE            │
              │   + LoRA adapter    │             │    Qwen1.5-7B          │
              │   3 epochs          │             │    zero-shot only      │
              │   5000 examples     │             │    no training         │
              └──────────┬──────────┘             └────────────┬───────────┘
                         │                                      │
                         └──────────────────┬──────────────────┘
                                            │
                    ┌───────────────────────▼─────────────────────────┐
                    │                  EVALUATION                      │
                    │  Same 500 held-out test examples for all models  │
                    │  QA: Exact Match + Token F1                      │
                    │       (split: answerable / unanswerable)         │
                    │  Summarization: ROUGE-1, ROUGE-2, ROUGE-L        │
                    │  Efficiency: latency, GPU memory, train time     │
                    └─────────────────────────────────────────────────┘
```

---

## Part 5 — How Training Actually Works

### Step 1: Prepare an example

Take a SQuAD v2 question-answering example:

```
Context : "The Eiffel Tower is located in Paris, France. It was built in 1889."
Question: "Where is the Eiffel Tower located?"
Answer  : "Paris, France"
```

Format it into a prompt:

```
### Context:
The Eiffel Tower is located in Paris, France. It was built in 1889.

### Question:
Where is the Eiffel Tower located?

### Answer:
Paris, France
```

### Step 2: Tokenize it

The tokenizer converts this text into a sequence of numbers (token IDs):

```
[1234, 567, 89, 12, ...]    ← token IDs for the whole prompt+answer
```

### Step 3: Mask the prompt

We only want the model to *learn* to produce the answer — not to reproduce the question or context.
So we **mask** all the prompt tokens by setting their labels to -100:

```
Prompt tokens: labels = [-100, -100, -100, ...]   ← masked, no gradient flows back
Answer tokens: labels = [3521, 892, ...]           ← trained, loss computed here
```

The model generates a prediction, the loss compares it to the reference answer tokens,
and the gradient updates **only the LoRA adapter matrices** — not the frozen base model.

### Step 4: The LoRA update

After each batch, only A and B matrices are updated via backpropagation:

```
Before step 1:  A and B are random/zero initialized
After 100 steps: A and B have learned "for this task, given this prompt format,
                 produce concise answers"
After 3 epochs: The model reliably extracts the right span from the context
```

### Step 5: Save the adapter

Only the adapter (~few MB) is saved — not the full 500M model. To run inference,
you load the frozen base model + attach the saved adapter.

---

## Part 6 — The Dataset Split Design

### Why this matters

The whole point of the evaluation is to test on examples the model **has never seen**.
If any test example leaked into training, the results would be meaningless.

### SQuAD v2 — special case (no public test split)

SQuAD v2 on HuggingFace has only `train` and `validation` splits — there is no labelled
`test` split publicly available. So we build our own:

```
HuggingFace "train" split (~130,000 examples)
    │
    └── We sample 5,000 → our TRAINING set
          (model learns from these)

HuggingFace "validation" split (~11,000 examples)
    │
    ├── We sample 500 → our VALIDATION set
    │     (used during training to pick the best checkpoint)
    │
    └── We sample another 500 → our TEST set
          (held out completely, only used for final evaluation)
          ↑ drawn together with validation in one random draw
            so they are GUARANTEED to have zero overlap
```

The disjoint guarantee works like this in code:

```python
# Draw 1000 indices at once (500 val + 500 test)
all_indices = rng.choice(11000, size=1000, replace=False)
val_indices  = all_indices[:500]   # first 500 → validation
test_indices = all_indices[500:]   # last  500 → test
# By construction: no index appears in both
assert len(set(val_indices) & set(test_indices)) == 0
```

### XL-Sum — simple case

XL-Sum provides official `train`, `validation`, and `test` splits. We use them directly
with no custom splitting needed.

### Why seed=42 matters

The same random seed is used in every config file. This means:
- All three models (base SLM, PEFT-SLM, LLM baseline) are evaluated on the **exact same
  500 test examples**
- If you use a different seed for the LLM baseline, it would be tested on different
  questions — and the comparison would be meaningless

---

## Part 7 — The Evaluation Design

### Why three groups for QA?

SQuAD v2 is unusual — about **half the questions have no answer** in the context.
The correct output for those is the string `"unanswerable"`.

If we only report the overall EM score, a lazy model could cheat:
- Predict "unanswerable" for every single question
- Get 50% EM (correct on all the unanswerable ones, wrong on all the answerable ones)
- This looks decent on paper but is useless in practice

So we report **three breakdowns**:

```
Overall EM    = score on all 500 examples
Answerable EM = score on the ~250 examples that have an answer
                (did the model find the right span?)
Unanswerable EM = score on the ~250 examples with no answer
                  (did the model correctly say "unanswerable"?)
```

This makes it immediately visible if a model is gaming the metric.

### ROUGE for summarization

ROUGE measures how much the generated summary **overlaps** with the reference summary:

- **ROUGE-1**: single-word overlap (did the summary use the same words?)
- **ROUGE-2**: two-word phrase overlap (did it use the same phrases?)
- **ROUGE-L**: longest matching sequence (does the summary flow similarly?)

Higher is better, but ROUGE is imperfect — a summary can score low ROUGE but still be
good, or score high by copying the original text. Qualitative analysis alongside ROUGE
is always useful.

### Efficiency metrics

These are just as important as the task metrics for this project:

| What we measure | Why it matters |
|---|---|
| Training time (seconds) | LoRA should train much faster than full fine-tuning |
| Peak GPU memory (MB) | LoRA should need much less VRAM |
| Inference latency (ms/example) | How long does each prediction take? |
| Throughput (examples/second) | How many docs can it process per second? |
| Trainable parameter count | How small is the adapter actually? |

---

## Part 8 — Why These Specific Models?

### Qwen1.5-0.5B as the SLM

- 500 million parameters — small enough to train on a free Colab T4 GPU
- From the Qwen1.5 family (Alibaba) — good base quality for its size
- No special access token required (unlike Llama or Gemma)
- Runs on CPU for demos — important for showing the project without a GPU

### Qwen1.5-7B as the larger-model baseline

- **Same model family** as the 0.5B — this is the key design choice
- Same tokenizer, same training data, same architecture pattern
- Only difference is scale (7B vs 0.5B)
- This means any difference in results is due to **model size**, not architecture differences
- It is called a "baseline" not an "upper bound" — a larger zero-shot model is not
  guaranteed to outperform a smaller fine-tuned model

### Why base variants (not instruct)?

The "instruct" version of a model has been further trained to follow instructions.
If we compared:
- 0.5B base (fine-tuned with LoRA) vs 7B instruct (zero-shot)

...the 7B model would have an unfair advantage because it was *already instruction-tuned*
before we even started. The improvement from LoRA would be hidden.

By using `base` for all three, **any improvement from PEFT is directly attributable
to our LoRA training** — nothing else changed.

---

## Part 9 — Source Code Map

This is where each concept from above lives in the codebase:

```
│
├── configs/
│   ├── pilot_squad_qa.yaml              ← all hyperparams for the QA experiment
│   ├── pilot_xlsum_summarization.yaml   ← all hyperparams for summarization
│   └── llm_baseline_qwen7b.yaml         ← config for the 7B zero-shot baseline
│
├── src/
│   │
│   ├── extraction/
│   │   └── text_extractor.py
│   │       • extract_text(path)         ← Part 4: document input → clean text
│   │       • _extract_pdf()             ← per-page native + OCR fallback
│   │       • _assess_quality()          ← quality flag: good / low / empty
│   │       • chunk_text()               ← split long docs into windows
│   │
│   ├── data/
│   │   ├── dataset_loader.py
│   │   │   • load_dataset_for_task()    ← Part 6: loads + splits the dataset
│   │   │   • _subsample()              ← draws val+test together, disjoint guarantee
│   │   │   • _HF_SPLIT_SOURCE          ← documents which HF split each role uses
│   │   │   • _save_test_indices()      ← saves indices to test_indices.json
│   │   │   • _load_qa()                ← formats SQuAD v2 into QA prompts
│   │   │   • _load_summarization()     ← formats XL-Sum into summary prompts
│   │   │
│   │   └── preprocessor.py
│   │       • _tokenize_prompt_completion()  ← Part 5: masking logic
│   │       • _extract_prompt_part()         ← finds the ### Answer:\n boundary
│   │       • get_data_collator()            ← padding collator for batches
│   │
│   ├── training/
│   │   ├── model_loader.py
│   │   │   • load_tokenizer()          ← loads tokenizer, sets pad token
│   │   │   • load_base_model()         ← loads model with right dtype/quantization
│   │   │   • attach_lora()             ← Part 3: attaches LoRA adapter
│   │   │   • _validate_target_modules()← checks q_proj/v_proj exist in model
│   │   │   • _assert_base_frozen()     ← verifies base weights are not trainable
│   │   │   • print_lora_param_report() ← prints exact trainable % from loaded model
│   │   │
│   │   ├── trainer.py
│   │   │   • run_training()            ← Part 5: full HF Trainer training loop
│   │   │                                 loads data, attaches LoRA, trains,
│   │   │                                 evaluates on test set, logs to CSV
│   │   │
│   │   └── run_pilot.py
│   │       • main()                    ← CLI entry point
│   │       • _run_base_evaluation()    ← --base_only: zero-shot inference
│   │       • _run_checkpoint_evaluation() ← --eval_only: load saved adapter
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   │   • compute_qa_metrics()      ← Part 7: EM + F1 + breakdown
│   │   │   • compute_rouge()           ← ROUGE-1/2/L for summarization
│   │   │   • compute_exact_match()     ← normalized string comparison
│   │   │   • compute_f1()             ← token overlap F1
│   │   │
│   │   └── evaluator.py
│   │       • evaluate_model()          ← standalone inference loop:
│   │                                     generates predictions, records latency,
│   │                                     computes task metrics + throughput
│   │
│   └── utils/
│       ├── config_loader.py
│       │   • load_config()             ← loads YAML, validates required keys
│       │
│       ├── experiment_tracker.py
│       │   • ExperimentTracker.log()   ← appends one result row to CSV
│       │   • ExperimentTracker.load()  ← returns pandas DataFrame of all runs
│       │
│       └── logging_setup.py
│           • setup_logging()           ← configures log level and format
│
├── notebooks/
│   ├── simple_pipeline.ipynb  ← self-contained Colab run of the whole QA experiment
│   └── build_simple_notebook.py ← generates it (edit this, not the .ipynb)
│
└── tests/
    └── smoke_test.py ← 9 automated checks that verify the whole pipeline (CPU)
```

---

## Part 10 — Data Flow Example (End to End)

Let's trace one SQuAD v2 example through the entire system:

### Input (raw HuggingFace example)

```python
{
  "context": "The Amazon River flows through Brazil, Peru, and Colombia.",
  "question": "Which countries does the Amazon River flow through?",
  "answers": {"text": ["Brazil, Peru, and Colombia"], "answer_start": [38]}
}
```

### Stage 1: Dataset loader formats it

```
### Context:
The Amazon River flows through Brazil, Peru, and Colombia.

### Question:
Which countries does the Amazon River flow through?

### Answer:
Brazil, Peru, and Colombia
```

### Stage 2: Tokenizer converts to numbers

```
[13890, 17, 2602, 3551, ...]     ← token IDs, one per subword
```

### Stage 3: Masking applied

```
Prompt tokens → labels = -100 (everything up to "### Answer:\n")
Answer tokens → labels = [15483, 11, 9945, 11, 323, 36006]
                                  ↑     ↑     ↑    ↑
                               Brazil  ,   Peru  and  Colombia
```

### Stage 4: Model forward pass

```
Input:  [13890, 17, 2602, ...]
Output: [softmax distribution over vocabulary at each position]
Loss:   cross-entropy between output[prompt_len:] and labels[prompt_len:]
        → only the answer tokens contribute to the loss
Gradient → flows back only to LoRA matrices A and B
```

### Stage 5: At inference time

```
Input:  "### Context:\nThe Amazon River...\n### Question:\n...\n### Answer:\n"
             ↑ no answer appended — model must generate it
Output: model generates token by token → "Brazil, Peru, and Colombia"
Metric: EM = 1.0 (exact match after normalization)
         F1 = 1.0 (all tokens match)
```

---

## Part 11 — What the Results Show

Phase 1 QA results (Google Colab T4, 2026-10-17, `notebooks/simple_pipeline.ipynb`,
raw files in `experiments/simple/`):

```
3-WAY COMPARISON  |  Task: QA  |  Dataset: SQuAD v2  |  Test examples: 500 (250 answerable / 250 unanswerable)
=====================================================================================================
Model / Method                    EM      F1    Ans EM  Unans EM  Latency(ms)  Params
-----------------------------------------------------------------------------------------------------
SLM — base (zero-shot)            5.6   16.4    11.2      0.0         191      0.5B
SLM + LoRA (PEFT-tuned)  ← OURS  68.8   72.4    66.4     71.2         150      0.5B (+0.169% adapter)
LLM baseline (zero-shot, 4-bit)  27.8   36.6    55.6      0.0        1139        7B
=====================================================================================================
LoRA training: 3 epochs, 21 min, 4,978 MB peak GPU memory, 786,432 trainable parameters.
```

**What this shows for QA:** SLM+LoRA > SLM base by 63 EM points, and SLM+LoRA > LLM baseline —
overall and on answerable questions alone (66.4 vs 55.6 EM). Neither zero-shot model ever
answers "unanswerable"; the tuned model learned to abstain (71.2 % correct on unanswerable
questions) but also over-abstains on 44 answerable ones. Latency is measured in this setup
(7B in 4-bit at batch 4, 0.5B in fp32 at batch 8), so the speed ratio is indicative.
Summarization (XL-Sum) is still to be run.

**General reading guide:**

**How to read the results:**

- If **SLM+LoRA > SLM base**: LoRA fine-tuning helped. The gap shows how much.
- If **SLM+LoRA ≈ LLM baseline**: a small fine-tuned model can match a large one.
- If **SLM+LoRA < LLM baseline**: the larger model still has an edge — but check
  the latency and GPU memory rows to see whether the efficiency gain justifies the
  performance gap.

The research conclusion comes from this table across both tasks (QA and summarization)
and the efficiency numbers recorded alongside it.

---

## Summary: The Five Key Ideas

| Idea | Why it matters |
|---|---|
| **PEFT / LoRA** | Adapts a frozen small model using only **0.169%** trainable params (786K / 465M) — cheap and fast |
| **Three-way comparison** | Isolates the effect of fine-tuning from the effect of model size |
| **Base variant for all three** | Any performance gain is from LoRA, not from instruction tuning |
| **Same seed / same test set** | Makes the comparison fair — all models see the same questions |
| **Unanswerable breakdown** | Prevents score inflation; gives an honest view of what the model can and cannot do |

# CLAUDE.md — Project Index

M.Tech research project, financial QA branch: PEFT (LoRA) fine-tuning of Qwen1.5-0.5B, compared
against the untuned model and Qwen1.5-7B on identical test examples, then served over HTTP.
This branch holds fine-tuning and inference only. SQuAD, XL-Sum, the Colab notebook and PDF
extraction are on `feat/joel-changes`. Results log: [docs/progress.md](docs/progress.md).

## Commands

```bash
bash scripts/start_server.sh             # install on first run, host the model on localhost
python -m src.serving.server --config configs/pilot_fin10k_qa.yaml   # HTTP endpoint, docs at /docs
python tests/test_fin10k.py              # fast offline checks: financial split, results CSV
python tests/test_server.py              # fast offline checks for the endpoint
python scripts/test_markdown_docs.py     # model on Markdown documents, short and long
python src/training/run_pilot.py --config configs/dry_run_fin10k.yaml   # full pipeline on 16 examples
python src/training/run_pilot.py --config configs/pilot_fin10k_qa.yaml [--base_only|--eval_only]
bash scripts/run_vast.sh                 # full comparison on a rented GPU (use tmux)
```

Run the model from a fresh clone: [docs/run_locally.md](docs/run_locally.md).

## Code map

Pipeline: config → dataset → tokenize → model + LoRA → train → evaluate → log CSV → serve.

| Stage | File | Key functions |
|---|---|---|
| Entry point | [src/training/run_pilot.py](src/training/run_pilot.py) | `main`, `_run_base_evaluation`, `_run_checkpoint_evaluation` |
| Config | [src/utils/config_loader.py](src/utils/config_loader.py) | `load_config` (validates YAML) |
| Data | [src/data/dataset_loader.py](src/data/dataset_loader.py) | `load_dataset_for_task`, `_load_fin10k`, `_split_fin10k`, `_save_test_indices` |
| Prompts | [src/data/prompts.py](src/data/prompts.py) | `build_qa_prompt`, `qa_completion` — single source of prompt format |
| Tokenize | [src/data/preprocessor.py](src/data/preprocessor.py) | `tokenize_qa`, `_tokenize_prompt_completion` (prompt masked with -100) |
| Model | [src/training/model_loader.py](src/training/model_loader.py) | `load_tokenizer`, `load_base_model` (4/8-bit via `load_in_bits`), `attach_lora`, `load_peft_checkpoint` |
| Train | [src/training/trainer.py](src/training/trainer.py) | `run_training` (HF Trainer) |
| Evaluate | [src/evaluation/evaluator.py](src/evaluation/evaluator.py), [metrics.py](src/evaluation/metrics.py) | `evaluate_model` (+latency), `save_predictions`; `compute_qa_metrics` (EM/F1, answerable vs unanswerable), `compute_rouge` |
| Results | [src/utils/experiment_tracker.py](src/utils/experiment_tracker.py) | `ExperimentTracker.log` → `experiments/results.csv` |
| Serving | [src/serving/server.py](src/serving/server.py) | `create_app`, `answer_question` |

Configs: [configs/](configs/) — `pilot_fin10k_qa.yaml`, `llm_baseline_fin10k_qwen7b.yaml`, `dry_run_fin10k.yaml`.
The loaders still contain SQuAD and XL-Sum branches, which have no config on this branch.

## Results (financial-qa-10K, 838 test questions, unseen companies, 2026-11-01)

Answerable F1: base 0.5B 54.35 · **0.5B + LoRA 68.90** · 7B 4-bit 61.74.
Unanswerable EM: 0.72 · **97.12** · 28.06. One seed; only LoRA was tuned.

## Invariants (don't break)

- All three compared runs use `seed: 42` and `model_variant: "base"` — same test examples, fair comparison.
- financial-qa-10K has one split: split by ticker with a fixed seed (not `training.seed`), so test companies are unseen and every run shares one test set.
- Loss only on completion tokens; examples whose answer is truncated away are skipped.
- Report EM/F1 split into answerable / unanswerable.
- Do not change `build_qa_prompt` without retesting: the adapter was trained on that wording.

## Results files

- `experiments/results.csv` is reserved for real runs (`run_pilot.py`); it is tracked in git.
- In `experiments/fin10k/adapter/`, only `adapter_config.json` and `adapter_model.safetensors` are tracked, so a fresh clone can serve the model. Other experiment outputs are gitignored.
- Validation during training is scored by loss only (`prediction_loss_only=True`). Test EM/F1 always come from generation via `evaluate_model()`.

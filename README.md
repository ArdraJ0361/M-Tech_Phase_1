# Financial QA with a Fine-Tuned Small Language Model

Fine-tune Qwen1.5-0.5B with LoRA to answer questions about financial text, then serve it over HTTP.
This branch holds fine-tuning and inference only. SQuAD, XL-Sum, the Colab notebook and PDF
extraction are on `feat/joel-changes`.

## Run the model

```bash
bash scripts/start_server.sh
```

This installs everything on the first run and starts the endpoint at
http://127.0.0.1:8765/docs. Step-by-step guide: [docs/run_locally.md](docs/run_locally.md).

## Results

Dataset: `virattt/financial-qa-10K`. 838 test questions from companies not used in training, with
the same prompt and scorer for every model, on one seed.

| Model | Answerable F1 | ROUGE-L | Unanswerable EM | Overall F1 | Latency / answer |
|---|---|---|---|---|---|
| Qwen1.5-0.5B | 54.35 | 51.73 | 0.72 | 45.46 | 283 ms |
| **Qwen1.5-0.5B + LoRA** | **68.90** | **66.68** | **97.12** | **73.58** | **204 ms** |
| Qwen1.5-7B (4-bit) | 61.74 | 58.57 | 28.06 | 56.19 | 864 ms |

Only the LoRA model was fine-tuned. Details and limits: [docs/progress.md](docs/progress.md).

## Fine-tune it again (GPU)

```bash
bash scripts/run_vast.sh    # full comparison on a rented GPU, use tmux
```

Or run one step at a time:

```bash
python src/training/run_pilot.py --config configs/dry_run_fin10k.yaml                  # quick check on 16 examples
python src/training/run_pilot.py --config configs/pilot_fin10k_qa.yaml --base_only     # untuned 0.5B
python src/training/run_pilot.py --config configs/pilot_fin10k_qa.yaml                 # LoRA
python src/training/run_pilot.py --config configs/llm_baseline_fin10k_qwen7b.yaml --base_only   # 7B
```

## Tests

```bash
python tests/test_fin10k.py              # data split and results CSV
python tests/test_server.py              # endpoint
python scripts/test_markdown_docs.py     # model on Markdown documents, short and long
```

## Layout

```
configs/        one YAML per run
src/data/       dataset loading, prompts, tokenizing
src/training/   model + LoRA loading, training, CLI
src/evaluation/ generation-based scoring
src/serving/    HTTP endpoint
src/utils/      config, results CSV, logging
scripts/        start_server.sh, run_vast.sh, test_markdown_docs.py
experiments/fin10k/adapter/   the trained adapter
```

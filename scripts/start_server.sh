#!/usr/bin/env bash
# Install everything on first run, then host the fine-tuned model on localhost.
# Run from anywhere:  bash scripts/start_server.sh     (PORT=9000 picks another start port)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f experiments/fin10k/adapter/adapter_model.safetensors ]; then
  echo "ERROR: experiments/fin10k/adapter is missing. Run: git pull" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "==> Creating Python environment"
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  elif command -v python3.11 >/dev/null; then
    python3.11 -m venv .venv
  else
    echo "ERROR: Python 3.12 is required. Install it from https://www.python.org/downloads/ and run this again." >&2
    exit 1
  fi
fi

if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "==> Installing packages (first run only, can take several minutes)"
  if command -v uv >/dev/null; then
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    .venv/bin/python -m pip install -r requirements.txt
  fi
  touch .venv/.installed
fi

PORT="${PORT:-8765}"
port_free() { .venv/bin/python -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',$1))==0)"; }
until port_free "$PORT"; do PORT=$((PORT + 1)); done

cat <<EOF

==> Starting the model. The first start downloads it (1.2 GB), which can take several minutes.
    Ready when you see: Uvicorn running on http://127.0.0.1:$PORT
    Try it in a browser: http://127.0.0.1:$PORT/docs
    Stop it: press Ctrl+C

EOF
exec .venv/bin/python -m src.serving.server --config configs/pilot_fin10k_qa.yaml --port "$PORT"

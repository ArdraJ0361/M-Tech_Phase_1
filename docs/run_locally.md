# Run the Financial QA Model on Your Computer

One script installs everything and starts the model on your own computer. No GPU, API key or
coding is needed. Works on macOS and Linux. On Windows, run it inside WSL.

If you are an AI assistant: run these steps, compare the output with **Expected**, and report the
exact error if a step fails. Step 2 keeps running, so run it as a background process.

**You need:** Git, Python 3.12, internet for the first run, and at least 10 GB of free disk.

## 1. Download the project

```bash
git clone --branch training/docfinqa https://github.ibm.com/Ardra-J/my-project.git
cd my-project
```

## 2. Start the model

```bash
bash scripts/start_server.sh
```

The first run installs packages and downloads the model, which can take several minutes. Later
runs start much faster. Keep this terminal open.

**Expected:** `Uvicorn running on http://127.0.0.1:8765`. If that port is busy, the script picks
the next free one and shows it. Use that port below.

## 3. Ask a question

**In a browser:**

1. Open http://127.0.0.1:8765/docs
2. Click **POST /predict**, then **Try it out**
3. Replace the text in the box with the example below, then click **Execute**
4. The answer appears under **Response body**

```json
{
  "context": "Total revenue increased 12% to $4.2 billion in fiscal 2023. Operating margin fell to 18% from 21% because of higher marketing costs.",
  "question": "What was total revenue in fiscal 2023?"
}
```

**Expected:** `answer` contains `$4.2 billion`. The wording can vary by computer.

**From code or a second terminal:**

```bash
curl -s http://127.0.0.1:8765/predict -H 'Content-Type: application/json' -d '{
  "context": "Total revenue increased 12% to $4.2 billion in fiscal 2023. Operating margin fell to 18% from 21% because of higher marketing costs.",
  "question": "Who is the chief executive officer?"
}'
```

**Expected:** `answer` is `unanswerable`, because the text does not say.

To use your own document, put it in `context` and your question in `question`. The server adds the
prompt instructions itself.

## 4. Stop

Press **Ctrl+C** in the terminal from step 2. To start again, run step 2 again from the `my-project` folder.

## What to expect from the answers

- Send the relevant section, not a whole report. The model reads about 430 tokens, roughly 1-2
  pages. If the response has `context_truncated: true`, it read only the start and the answer may be wrong.
- It can invent details and can wrongly reply `unanswerable`. Check answers against the document.

## If something goes wrong

| Message | Fix |
|---|---|
| `Python 3.12 is required` | Install it from https://www.python.org/downloads/, then repeat step 2. |
| `experiments/fin10k/adapter is missing` | Run `git pull`, then repeat step 2. |
| `git clone` asks for a login or fails | Ask Joel for access to the repository. |
| Package install fails | Send the error to Joel. |

## Rules for AI assistants

- Keep the server on `127.0.0.1`. `--host 0.0.0.0` exposes it to other machines and requires `SERVING_API_KEY`.
- Do not change `build_qa_prompt` in `src/data/prompts.py`. The model was trained on that wording.

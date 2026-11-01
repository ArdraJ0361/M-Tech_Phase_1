"""Fast, offline checks for the HTTP endpoint (no model download). Run: python tests/test_server.py"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from fastapi.testclient import TestClient

from src.serving import server

CFG = {
    "task": "qa",
    "model_name_or_path": "fake",
    "dataset": {"max_source_length": 120, "max_target_length": 4},
}


class FakeTokenizer:
    """One token per word, enough to exercise trimming and decoding."""
    pad_token_id = eos_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [1] * len(text.split())
        return {"input_ids": torch.tensor([ids]) if return_tensors else ids}

    def decode(self, ids, skip_special_tokens=False):
        return "$4.2 billion\nextra line" if torch.is_tensor(ids) else " ".join(["word"] * len(ids))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **_):
        return torch.cat([input_ids, torch.tensor([[2, 2]])], dim=1)


def _client(api_key=None):
    server.load_tokenizer = lambda _cfg: FakeTokenizer()
    server.load_peft_checkpoint = lambda _cfg, _path: FakeModel()
    return TestClient(server.create_app(CFG, "unused", api_key))


def main():
    body = {"context": "Revenue was 4.2 billion.", "question": "What was revenue?"}
    with _client() as c:
        assert c.get("/health").json()["status"] == "ok"

        r = c.post("/predict", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["answer"] == "$4.2 billion", "only the first line is the answer"
        assert r.json()["context_truncated"] is False

        long_context = " ".join(["word"] * 500)
        assert c.post("/predict", json={**body, "context": long_context}).json()["context_truncated"] is True

        for bad in ({"question": "q"}, {**body, "context": "   "}, {**body, "question": ""},
                    {**body, "context": "x" * 200_001}):
            assert c.post("/predict", json=bad).status_code == 422, bad

    with _client(api_key="secret") as c:
        assert c.post("/predict", json=body).status_code == 401
        assert c.post("/predict", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/predict", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    print("[OK] all checks passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exercise the live TP2 API and preserve responses/timings for review."""
import json
import time
import urllib.request
from pathlib import Path
BASE = "http://127.0.0.1:8080"
out = Path(__file__).parent / "backups/tp2-20260908"
out.mkdir(parents=True, exist_ok=True)
results = []
for prompt in ["只回答數字：17 加 25 等於多少？", "請用繁體中文，分三點說明張量並行如何讓兩台 GPU 共同執行一個大型語言模型。"]:
    payload = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 192, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=240) as response:
        data = json.load(response)
    elapsed = time.monotonic() - start
    content = data["choices"][0]["message"].get("content", "")
    if not content:
        raise RuntimeError("Empty model response: " + json.dumps(data))
    if len(results) == 0 and "42" not in content:
        raise RuntimeError("Arithmetic sanity check failed: " + content)
    results.append({"prompt": prompt, "elapsed_seconds": elapsed, "response": data})
    print(json.dumps({"elapsed_seconds": round(elapsed, 2), "content": content, "usage": data.get("usage")}, ensure_ascii=False), flush=True)
(out / "smoke-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

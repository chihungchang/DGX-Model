#!/usr/bin/env python3
"""Exercise many-image Responses requests with cold/warm preprocessing cache."""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.images <= 64:
        parser.error("--images must be between 2 and 64")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 180)
    images = []
    for i in range(args.images):
        image = Image.new("RGB", (1024, 768), (230, 200 + i % 40, 180 + i % 60))
        ImageDraw.Draw(image).text((260, 260), str(100 + i), fill="black", font=font)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        images.append({
            "type": "input_image",
            "image_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        })

    results = []
    for case, position, expected in (
        ("cold_first", "FIRST", "100"),
        ("warm_last", "LAST", str(99 + args.images)),
    ):
        payload = {
            "model": "qwen3.8-27b",
            "input": [{"role": "user", "content": images + [{
                "type": "input_text",
                "text": f"Read the large number in the {position} image. Reply with only that number.",
            }]}],
            "temperature": 0,
            "max_output_tokens": 32,
            "stream": True,
            "store": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        print(f"Starting {case}: {args.images} images, 1024x768 each", flush=True)
        start = time.monotonic()
        content, completed, first_token, usage = "", False, None, None
        with requests.post(args.base_url + "/v1/responses", json=payload,
                           stream=True, timeout=(10, args.timeout)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                kind = event.get("type")
                if event.get("error") or kind in ("error", "response.failed", "response.incomplete"):
                    raise RuntimeError(event)
                if kind == "response.output_text.delta":
                    if first_token is None:
                        first_token = time.monotonic() - start
                    content += event.get("delta", "")
                if kind == "response.completed":
                    completed = True
                    usage = event.get("response", {}).get("usage")
                    break
        result = {
            "case": case, "images": args.images, "content": content,
            "expected": expected, "completed": completed, "usage": usage,
            "first_token_seconds": first_token,
            "total_seconds": time.monotonic() - start,
        }
        results.append(result)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        assert completed and content.strip() == expected, result

    response = requests.post(args.base_url + "/v1/chat/completions", json={
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": "What is 17 + 25? Reply with only the number."}],
        "temperature": 0, "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=(10, args.timeout))
    response.raise_for_status()
    body = response.json()
    (args.output_dir / "text-after.json").write_text(json.dumps(body, indent=2) + "\n")
    assert body["choices"][0]["message"]["content"].strip() == "42", body
    print("PASS: cold/warm multi-image Responses streams and subsequent text completion", flush=True)


if __name__ == "__main__":
    main()

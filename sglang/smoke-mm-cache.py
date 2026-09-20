#!/usr/bin/env python3
"""Check image correctness and repeated-image latency with synthetic fixtures."""

import argparse
import base64
import concurrent.futures
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--images", type=int, default=17)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--hello-delay", type=float, default=5.0)
    parser.add_argument("--concurrent-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def generate(name, messages):
        payload = {
            "model": "qwen3.8-27b", "messages": messages,
            "temperature": 0, "max_tokens": 32, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.monotonic()
        content = ""
        first_token = None
        usage = None
        done = False
        with requests.post(args.base_url + "/v1/chat/completions", json=payload,
                           stream=True, timeout=(10, args.timeout)) as response:
            response.raise_for_status()
            headers_time = time.monotonic() - start
            for line in response.iter_lines(chunk_size=1):
                if line == b"data: [DONE]":
                    done = True
                    break
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(event["error"])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    if text and first_token is None:
                        first_token = time.monotonic() - start
                    content += text
        if not done:
            raise RuntimeError("Stream ended without [DONE]")
        return {"case": name, "headers_seconds": headers_time,
                "first_content_seconds": first_token,
                "total_seconds": time.monotonic() - start,
                "content": content, "usage": usage}

    def save(result, expected=None):
        results.append(result)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        if expected is not None and result["content"].strip() != expected:
            raise AssertionError(f"{result['case']}: expected {expected}, got {result['content']!r}")

    if not args.concurrent_only:
        save(generate("text_baseline", [{"role": "user", "content":
             "What is 17 + 25? Reply with only the number."}]), "42")
    images = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 180)
    for index in range(args.images):
        image = Image.new("RGB", (1024, 768), (240, 235 - index, 210 + index))
        ImageDraw.Draw(image).text((260, 260), str(100 + index), fill="black", font=font)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        images.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}})
    prompt = [{"role": "user", "content": images + [{"type": "text", "text":
        "Read the large number in the FIRST image. Reply with only that number."}]}]
    cold = {"content": "100"}
    if not args.concurrent_only:
        cold = generate("images_cold", prompt)
        save(cold, "100")
    warm = prompt + [{"role": "assistant", "content": cold["content"]},
                     {"role": "user", "content": "Confirm the number in the first image. Only the number."}]
    if not args.concurrent_only:
        save(generate("images_warm", warm), "100")
    mixed = warm + [{"role": "assistant", "content": "100"},
                    {"role": "user", "content": "Again, first image number only."}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        image_future = pool.submit(generate, "images_warm_concurrent", mixed)
        time.sleep(args.hello_delay)
        hello_future = pool.submit(generate, "hello_concurrent", [{"role": "user", "content":
            "hello. Reply with only HELLO."}])
        save(hello_future.result(), "HELLO")
        save(image_future.result(), "100")


if __name__ == "__main__":
    main()

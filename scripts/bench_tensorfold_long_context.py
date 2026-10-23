#!/usr/bin/env python3
"""Measure streamed TTFT and decode rate on the existing TensorFold server.

Uses its /tokenize endpoint to size every rendered chat prompt exactly.
Source corpus, request bodies, SSE events, health samples and results are
saved in the output directory. Does not change the serving configuration.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import mimetypes
from pathlib import Path
import shlex
import statistics
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlsplit
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.base = args.base.rstrip("/")
        self.phase = "prepare"
        self.stop_monitor = threading.Event()
        self.lock = threading.Lock()
        self.image_parts = []
        images = []
        for index, filename in enumerate(args.image):
            path = Path(filename)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ("image/png", "image/jpeg", "image/webp"):
                raise ValueError("Benchmark images must be PNG, JPEG or WebP")
            data = path.read_bytes()
            self.image_parts.append({"type": "image_url", "image_url": {
                "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
            (self.output / f"image-{index + 1}{path.suffix}").write_bytes(data)
            images.append({"filename": path.name, "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()})
        self.report = {"started_utc": now(), "api": self.base, "model": args.model,
                       "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                       "temperature": 0, "thinking": False, "ignore_eos": True,
                       "single_repeats": args.single_repeats, "images_per_request": images, "results": [],
                       "method": {"ttft": "POST start to first nonempty content delta",
                                  "decode_tok_s": "(completion_tokens - 1) / (last content time - first content time)",
                                  "end_to_end_tok_s": "all completion tokens / batch wall time, including prefill and queue",
                                  "cache": ("reuse prior request bodies; report actual cached tokens" if args.allow_cache else
                                            "unique nonce before the document; require cached_tokens == 0")}}

    def event(self, **fields):
        item = {"time_utc": now(), "phase": self.phase, **fields}
        with self.lock:
            with (self.output / "events.jsonl").open("a") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(json.dumps(item, ensure_ascii=False), flush=True)

    def save(self):
        temporary = self.output / "results.tmp"
        temporary.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(self.output / "results.json")

    def request(self, path, body=None, timeout=60):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(self.base + path, data, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    def count(self, messages):
        return int(self.request("/tokenize", {"model": self.args.model, "messages": messages,
                                             "chat_template_kwargs": {"enable_thinking": False}})["count"])

    def prepare(self, label, corpus):
        nonce = uuid.uuid4().hex

        def messages(chars, padding=0):
            text = (
                "請完整閱讀以下本文。\n<本文>\n" + corpus[:chars] + "\n</本文>\n"
                + ("填充字元：" + " x" * padding + "\n" if padding else "")
                + ("請先用一行描述附圖中的數字、圖形和顏色，再分析本文。" if self.image_parts else "")
                + "請撰寫至少兩千字的連貫分析，討論本文中的領導、信任、結盟與決策，"
                "結合人物和事件舉例，說明不同人物的選擇及其後果。不要逐字引用長段原文，"
                "不要介紹識別碼，不要以條列或表格作答，直接開始正文。")
            content = ([*self.image_parts, {"type": "text", "text": text}]
                       if self.image_parts else text)
            return [{"role": "system", "content": (
                f"本次請求識別碼：{nonce}。忽略識別碼與填充字元。"
                "你是一位閱讀繁體中文長篇小說的研究者，請依據本文回答，使用繁體中文。")},
                {"role": "user", "content": content}]

        # First find a natural-text prefix slightly below the target. The last
        # few tokens are benign filler, checked with the server's exact renderer.
        goal = self.args.input_tokens - 24
        low, high = 0, min(len(corpus), self.args.input_tokens * 4)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            count = self.count(messages(middle))
            if count <= goal:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        for padding in range(1, 80):
            msg = messages(best, padding)
            count = self.count(msg)
            if count == self.args.input_tokens:
                break
        else:
            raise RuntimeError("Could not construct the exact token count")
        body = {"model": self.args.model, "messages": msg, "stream": True,
                "stream_options": {"include_usage": True}, "max_tokens": self.args.output_tokens,
                "temperature": 0, "top_p": 1, "ignore_eos": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        raw = json.dumps(body, ensure_ascii=False).encode()
        (self.output / (label + ".request.json")).write_bytes(raw)
        self.event(event="prepared", label=label, prompt_tokens=count,
                   document_characters=best, filler_words=padding, request_bytes=len(raw))
        return body, {"document_characters": best, "filler_words": padding,
                      "request_sha256": hashlib.sha256(raw).hexdigest()}

    def monitor(self):
        with (self.output / "health.jsonl").open("a", buffering=1) as f:
            while not self.stop_monitor.is_set():
                sample = {"time_utc": now(), "monotonic": time.perf_counter(), "phase": self.phase}
                try:
                    sample["health"] = self.request("/health", timeout=10)
                    with urllib.request.urlopen(self.base + "/metrics", timeout=10) as res:
                        lines = res.read().decode().splitlines()
                    sample["queue"] = {line.split()[0]: float(line.split()[1]) for line in lines
                                       if line.startswith(("tensorfold:requests_running ",
                                                           "tensorfold:requests_waiting "))}
                    mem = {line.split(":")[0]: int(line.split()[1]) for line in
                           Path("/proc/meminfo").read_text().splitlines()}
                    sample["head_mem_available_gib"] = mem["MemAvailable"] / 1024**2
                    if self.args.worker:
                        code = ("import pathlib,json; m={l.split(':')[0]:int(l.split()[1]) for l in "
                                "pathlib.Path('/proc/meminfo').read_text().splitlines()}; "
                                "print(json.dumps(m['MemAvailable']/1024**2))")
                        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                                                 self.args.worker, shlex.join(["python3", "-c", code])],
                                                capture_output=True, text=True, timeout=8, check=True)
                        sample["worker_mem_available_gib"] = json.loads(result.stdout)
                except Exception as exc:
                    sample["error"] = str(exc)
                f.write(json.dumps(sample) + "\n")
                self.stop_monitor.wait(5)

    def stream(self, label, body, meta, barrier):
        parsed = urlsplit(self.base)
        raw = json.dumps(body, ensure_ascii=False).encode()
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=1800)
        result = {"label": label, **meta}
        usage, stats, text = {}, {}, []
        first = last = None
        done, finish = False, None
        barrier.wait()
        start = time.perf_counter()
        result.update(started_utc=now(), started_monotonic=start)
        self.event(event="sent", label=label)
        try:
            conn.request("POST", parsed.path.rstrip("/") + "/v1/chat/completions", raw,
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            result["headers_s"] = time.perf_counter() - start
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {response.read()[:1000]!r}")
            with (self.output / (label + ".sse.jsonl")).open("w", buffering=1) as events:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    arrived = time.perf_counter()
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    item = json.loads(payload)
                    events.write(json.dumps({"elapsed_s": arrived - start, "data": item},
                                            ensure_ascii=False) + "\n")
                    if item.get("error"):
                        raise RuntimeError(item["error"])
                    if item.get("usage"):
                        usage = item["usage"]
                    if item.get("tensorfold"):
                        stats = item["tensorfold"]
                    for choice in item.get("choices") or []:
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content:
                            text.append(content)
                            last = arrived
                            if first is None:
                                first = arrived
                                self.event(event="first_token", label=label, ttft_s=first - start)
            result.update(total_s=time.perf_counter() - start, ttft_s=first - start if first else None,
                          first_token_monotonic=first, last_token_monotonic=last,
                          usage=usage, tensorfold=stats, finish_reason=finish, done=done,
                          text="".join(text))
            n = int(usage.get("completion_tokens", 0))
            result["decode_s"] = last - first if first and last else 0
            result["decode_tok_s"] = (n - 1) / result["decode_s"] if result["decode_s"] > 0 else 0
            result["prefill_tok_s"] = usage.get("prompt_tokens", 0) / stats["prefill_s"] if stats.get("prefill_s") else None
            if not done or first is None:
                raise RuntimeError("Stream ended without content or [DONE]")
            if usage.get("prompt_tokens") != self.args.input_tokens:
                raise RuntimeError("Actual prompt token count differs from the target")
            if not self.args.allow_cache and (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) != 0:
                raise RuntimeError("Prompt cache hit invalidates the cold-prompt measurement")
            if n != self.args.output_tokens:
                raise RuntimeError(f"Expected {self.args.output_tokens} completion tokens, got {n}")
            result["status"] = "ok"
            self.event(event="finished", label=label, ttft_s=result["ttft_s"],
                       decode_tok_s=result["decode_tok_s"], total_s=result["total_s"],
                       prompt_tokens=usage.get("prompt_tokens"), completion_tokens=n,
                       cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        except Exception as exc:
            result.update(status="error", error=str(exc), total_s=time.perf_counter() - start,
                          usage=usage, tensorfold=stats, text="".join(text))
            self.event(event="failed", label=label, error=str(exc))
        finally:
            conn.close()
        (self.output / (label + ".result.json")).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return result

    def batch(self, phase, prepared):
        self.phase = phase
        initial = self.request("/health")
        self.event(event="batch_start", count=len(prepared), health=initial)
        barrier = threading.Barrier(len(prepared))
        start = time.perf_counter()
        rows = []
        with ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = [pool.submit(self.stream, label, body, meta, barrier)
                       for label, body, meta in prepared]
            for future in as_completed(futures):
                rows.append(future.result())
                self.report["results"].append(rows[-1])
                self.save()
        end = self.request("/health")
        summary = {"phase": phase, "wall_s": time.perf_counter() - start,
                   "health_before": initial, "health_after": end,
                   "unexpected_finished_requests": end.get("requests_total", 0) - initial.get("requests_total", 0) - len(prepared)}
        self.report.setdefault("batches", []).append(summary)
        self.save()
        if any(row["status"] != "ok" for row in rows):
            raise RuntimeError("Benchmark request failed; inspect its result and server logs")

    def run(self):
        if self.args.reuse_four:
            if self.args.single_repeats or not self.args.allow_cache:
                raise ValueError("--reuse-four requires --single-repeats 0 and --allow-cache")
            if self.args.image:
                raise ValueError("--reuse-four already includes the saved images; omit --image")
            previous = json.loads((Path(self.args.reuse_four) / "results.json").read_text())
            self.report["images_per_request"] = previous.get("images_per_request", [])
            prepared = []
            for index in range(4):
                label = f"four-{index + 1}"
                raw = (Path(self.args.reuse_four) / (label + ".request.json")).read_bytes()
                body = json.loads(raw)
                assert self.count(body["messages"]) == self.args.input_tokens
                (self.output / (label + ".request.json")).write_bytes(raw)
                prepared.append((label, body, {"request_sha256": hashlib.sha256(raw).hexdigest()}))
        else:
            source = (self.output / "pg23950.txt").read_text(encoding="utf-8-sig")
            source = source.replace("\r\n", "\n")
            start = source.find("*** START OF THE PROJECT GUTENBERG EBOOK")
            if start >= 0:
                source = source[source.find("\n", start) + 1:]
            end = source.find("*** END OF THE PROJECT GUTENBERG EBOOK")
            if end >= 0:
                source = source[:end]
            labels = [f"single-{i + 1}" for i in range(self.args.single_repeats)] + [f"four-{i + 1}" for i in range(4)]
            prepared = [(label, *self.prepare(label, source)) for label in labels]
        self.report["health_before"] = self.request("/health")
        self.save()
        monitor = threading.Thread(target=self.monitor, daemon=True)
        monitor.start()
        try:
            for item in prepared[:self.args.single_repeats]:
                self.batch(item[0], [item])
            self.batch("four-concurrent", prepared[self.args.single_repeats:])
            self.report.update(status="complete", ended_utc=now())
        except BaseException as exc:
            self.report.update(status="failed", error=str(exc), ended_utc=now())
            raise
        finally:
            self.stop_monitor.set()
            monitor.join(timeout=15)
            self.save()
        rows = self.report["results"]
        summary = {}
        for group, prefix in [("single", "single-"), ("four_concurrent", "four-")]:
            selected = [r for r in rows if r["label"].startswith(prefix)]
            if not selected:
                continue
            summary[group] = {"requests": len(selected),
                              "ttft_median_s": statistics.median(r["ttft_s"] for r in selected),
                              "ttft_mean_s": statistics.mean(r["ttft_s"] for r in selected),
                              "decode_mean_tok_s": statistics.mean(r["decode_tok_s"] for r in selected),
                              "decode_median_tok_s": statistics.median(r["decode_tok_s"] for r in selected)}
        burst = [r for r in rows if r["label"].startswith("four-")]
        wall = max(r["started_monotonic"] + r["total_s"] for r in burst) - min(r["started_monotonic"] for r in burst)
        summary["four_concurrent"].update(wall_s=wall, end_to_end_tok_s=sum(r["usage"]["completion_tokens"] for r in burst) / wall)
        self.report["summary"] = summary
        self.save()
        self.event(event="complete", summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument("--input-tokens", type=int, default=261632,
                        help="Rendered input tokens, including images (default leaves 512 reply tokens in 256K)")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--single-repeats", type=int, default=3)
    parser.add_argument("--worker", help="SSH host for recording the peer's memory")
    parser.add_argument("--allow-cache", action="store_true")
    parser.add_argument("--reuse-four", help="Reuse the four request bodies from a previous cold run")
    parser.add_argument("--image", action="append", default=[],
                        help="Attach this image to each new request; repeat for multiple images. Input tokens include images.")
    Benchmark(parser.parse_args()).run()


if __name__ == "__main__":
    main()

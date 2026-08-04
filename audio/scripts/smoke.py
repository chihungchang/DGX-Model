#!/usr/bin/env python3
"""Concurrent LLM + Qwen3-ASR + Qwen3-TTS deployment smoke test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from profile import Profile, load_profile  # noqa: E402


class SmokeError(RuntimeError):
    pass


ASR_SAMPLE_RATE_HZ = 16_000
ASR_SAMPLE_BYTES = 4
ASR_TRANSPORT_CHUNK_MS = 100
ASR_FINISH_AUDIO_MS = 1_200


def request_json(url: str, *, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if response.status < 200 or response.status >= 300:
                raise SmokeError(f"{url} returned HTTP {response.status}")
    except (urllib.error.URLError, TimeoutError) as error:
        raise SmokeError(f"{url} failed: {error}") from error
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SmokeError(f"{url} did not return JSON") from error
    if not isinstance(value, dict):
        raise SmokeError(f"{url} returned a non-object JSON payload")
    return value


def request_no_content(url: str, *, method: str = "POST", timeout: float = 30.0) -> None:
    request = urllib.request.Request(url, data=b"", method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            if response.status != 204:
                raise SmokeError(f"{url} returned HTTP {response.status}; expected 204")
    except (urllib.error.URLError, TimeoutError) as error:
        raise SmokeError(f"{url} failed: {error}") from error
    if payload:
        raise SmokeError(f"{url} returned a body for a 204 response")


def silent_pcm_chunks(*, total_ms: int, chunk_ms: int) -> list[bytes]:
    if total_ms <= 0 or chunk_ms <= 0 or total_ms % chunk_ms:
        raise SmokeError("ASR smoke duration must be a positive multiple of its transport chunk")
    samples_per_chunk = ASR_SAMPLE_RATE_HZ * chunk_ms
    if samples_per_chunk % 1000:
        raise SmokeError("ASR transport chunk must contain a whole number of 16 kHz samples")
    chunk = b"\0" * (samples_per_chunk // 1000 * ASR_SAMPLE_BYTES)
    return [chunk for _ in range(total_ms // chunk_ms)]


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 3)


def asr_smoke(profile: Profile) -> dict[str, object]:
    port = profile.option("asr", "--port")
    base = f"http://127.0.0.1:{port}"
    ready = request_json(f"{base}/readyz", timeout=5)
    if ready.get("ready") is not True or ready.get("model") != "Qwen/Qwen3-ASR-1.7B":
        raise SmokeError("ASR readiness does not identify the fixed Qwen3-ASR model")
    configured_window = float(profile.option("asr", "--chunk-size-sec") or "0")
    if ready.get("decodeWindowSeconds") != configured_window:
        raise SmokeError(
            "ASR readiness decode window does not match the selected profile: "
            f"runtime={ready.get('decodeWindowSeconds')!r}, profile={configured_window}"
        )

    transport_chunks = silent_pcm_chunks(
        total_ms=ASR_FINISH_AUDIO_MS,
        chunk_ms=ASR_TRANSPORT_CHUNK_MS,
    )
    started = request_json(f"{base}/api/start", method="POST", body=b"", timeout=10)
    session_id = started.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise SmokeError("ASR start did not return session_id")
    query = urllib.parse.urlencode({"session_id": session_id})
    chunk_latencies_ms: list[float] = []
    for pcm in transport_chunks:
        chunk_started_at = time.perf_counter()
        partial = request_json(
            f"{base}/api/chunk?{query}",
            method="POST",
            body=pcm,
            headers={"Content-Type": "application/octet-stream"},
            timeout=60,
        )
        chunk_latencies_ms.append(elapsed_ms(chunk_started_at))
        if not isinstance(partial.get("text", ""), str):
            raise SmokeError("ASR partial response has an invalid text field")
    finish_started_at = time.perf_counter()
    final = request_json(f"{base}/api/finish?{query}", method="POST", body=b"", timeout=60)
    finish_latency_ms = elapsed_ms(finish_started_at)
    if not isinstance(final.get("text", ""), str):
        raise SmokeError("ASR final response has an invalid text field")

    cancel_started = request_json(f"{base}/api/start", method="POST", body=b"", timeout=10)
    cancel_session_id = cancel_started.get("session_id")
    if not isinstance(cancel_session_id, str) or not cancel_session_id:
        raise SmokeError("ASR cancel probe did not return session_id")
    cancel_query = urllib.parse.urlencode({"session_id": cancel_session_id})
    request_json(
        f"{base}/api/chunk?{cancel_query}",
        method="POST",
        body=transport_chunks[0],
        headers={"Content-Type": "application/octet-stream"},
        timeout=30,
    )
    cancel_started_at = time.perf_counter()
    request_no_content(f"{base}/api/cancel?{cancel_query}", timeout=30)
    cancel_latency_ms = elapsed_ms(cancel_started_at)

    return {
        "model": ready["model"],
        "engine": ready.get("engine"),
        "contracts": ["start/chunk/finish", "start/chunk/cancel"],
        "decodeWindowSeconds": configured_window,
        "transportChunkMilliseconds": ASR_TRANSPORT_CHUNK_MS,
        "transportChunkBytes": len(transport_chunks[0]),
        "transportChunkCount": len(transport_chunks),
        "audioMilliseconds": ASR_FINISH_AUDIO_MS,
        "maxChunkRoundTripMilliseconds": max(chunk_latencies_ms),
        "finishRoundTripMilliseconds": finish_latency_ms,
        "cancelRoundTripMilliseconds": cancel_latency_ms,
    }


def tts_smoke(profile: Profile) -> dict[str, object]:
    port = profile.option("tts", "--port")
    base = f"http://127.0.0.1:{port}"
    with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
        if response.status != 200:
            raise SmokeError(f"TTS health returned HTTP {response.status}")
    payload = json.dumps(
        {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "input": "這是 Mazi Live Mode 的語音部署測試。",
            "voice": "vivian",
            "language": "Chinese",
            "task_type": "CustomVoice",
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/v1/audio/speech",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            pcm = response.read()
            content_type = response.headers.get("content-type", "")
    except (urllib.error.URLError, TimeoutError) as error:
        raise SmokeError(f"TTS streaming request failed: {error}") from error
    if len(pcm) < 4096 or len(pcm) % 2:
        raise SmokeError(f"TTS returned invalid PCM byte count: {len(pcm)}")
    return {"model": profile.option("tts", "--model"), "bytes": len(pcm), "contentType": content_type}


def llm_smoke(profile: Profile) -> dict[str, object]:
    base = profile.meta["llm_api_base"].rstrip("/")
    models = request_json(f"{base}/models", timeout=10)
    data = models.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict) or not isinstance(data[0].get("id"), str):
        raise SmokeError("LLM /models did not return a served model id")
    model = data[0]["id"]
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply only OK."}],
            "temperature": 0,
            "max_tokens": 4,
        }
    ).encode("utf-8")
    completion = request_json(
        f"{base}/chat/completions",
        method="POST",
        body=body,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SmokeError("LLM completion returned no choices")
    return {"model": model, "contract": "chat/completions"}


def run_smoke(profile: Profile, *, audio_only: bool) -> dict[str, object]:
    checks = {"asr": asr_smoke, "tts": tts_smoke}
    if not audio_only:
        checks["llm"] = llm_smoke
    results: dict[str, object] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(checks)) as executor:
        futures = {name: executor.submit(check, profile) for name, check in checks.items()}
        for name, future in futures.items():
            results[name] = future.result()
    return {"status": "ok", "concurrent": True, "checks": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--audio-only", action="store_true", help="diagnostic only; never establishes deployment readiness")
    args = parser.parse_args()
    try:
        profile = load_profile(args.profile)
        result = run_smoke(profile, audio_only=args.audio_only)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (SmokeError, OSError, ValueError, concurrent.futures.TimeoutError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Qwen3-ASR vLLM streaming runtime with model-specific session state only."""

from __future__ import annotations

import argparse
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from qwen_asr import Qwen3ASRModel
from starlette.concurrency import run_in_threadpool


MODEL_NAME = "Qwen/Qwen3-ASR-1.7B"
ENGINE_NAME = "qwen-asr-vllm"
LOGGER = logging.getLogger("mazi.qwen3_asr")


class RequestError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Session:
    state: Any
    created_at: float
    last_seen: float
    # Includes work waiting for either the per-session lock or global model lock.
    inflight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class AsrRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: Qwen3ASRModel | None = None
        self.started_at = time.monotonic()
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.RLock()
        self.model_lock = threading.RLock()
        self.shutting_down = False

    def load(self) -> None:
        if self.args.asr_model_path != MODEL_NAME:
            raise RuntimeError(f"model is fixed to {MODEL_NAME}")
        LOGGER.info("loading model=%s engine=%s", MODEL_NAME, ENGINE_NAME)
        model = Qwen3ASRModel.LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_model_len,
            max_inference_batch_size=self.args.max_inference_batch_size,
            max_new_tokens=self.args.max_new_tokens,
            enforce_eager=self.args.enforce_eager,
        )
        # Health must mean the first live microphone chunk can be served within
        # the Cloud request budget. Qwen/vLLM still JITs audio kernels on the
        # first generate call even with enforce_eager, so perform that work
        # before publishing the runtime as ready.
        warmup_started_at = time.monotonic()
        warmup_state = model.init_streaming_state(
            unfixed_chunk_num=self.args.unfixed_chunk_num,
            unfixed_token_num=self.args.unfixed_token_num,
            chunk_size_sec=self.args.chunk_size_sec,
        )
        model.streaming_transcribe(
            np.zeros(warmup_state.chunk_size_samples, dtype=np.float32),
            warmup_state,
        )
        LOGGER.info("model warmup complete elapsed_seconds=%.3f", time.monotonic() - warmup_started_at)
        self.model = model
        LOGGER.info("model ready model=%s engine=%s", MODEL_NAME, ENGINE_NAME)

    @property
    def ready(self) -> bool:
        return self.model is not None and not self.shutting_down

    def _require_model(self) -> Qwen3ASRModel:
        if not self.ready or self.model is None:
            raise RequestError("Qwen3-ASR runtime is not ready", 503)
        return self.model

    def _expire_sessions(self) -> None:
        cutoff = time.monotonic() - self.args.session_ttl_sec
        expired_count = 0
        with self.sessions_lock:
            for session_id, session in list(self.sessions.items()):
                if session.inflight == 0 and session.last_seen < cutoff:
                    self.sessions.pop(session_id)
                    expired_count += 1
        if expired_count:
            LOGGER.info("expired_sessions=%d", expired_count)

    def start_session(self) -> str:
        model = self._require_model()
        self._expire_sessions()
        with self.sessions_lock:
            if len(self.sessions) >= self.args.max_sessions:
                raise RequestError("ASR session capacity reached", 429)
            with self.model_lock:
                state = model.init_streaming_state(
                    unfixed_chunk_num=self.args.unfixed_chunk_num,
                    unfixed_token_num=self.args.unfixed_token_num,
                    chunk_size_sec=self.args.chunk_size_sec,
                )
            session_id = uuid.uuid4().hex
            now = time.monotonic()
            self.sessions[session_id] = Session(state=state, created_at=now, last_seen=now)
        return session_id

    def _acquire_session(self, session_id: str) -> Session:
        self._expire_sessions()
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise RequestError("invalid session_id")
            session.inflight += 1
            session.last_seen = time.monotonic()
            return session

    def _release_session(
        self,
        session_id: str,
        session: Session,
        *,
        remove: bool = False,
    ) -> None:
        with self.sessions_lock:
            session.inflight = max(0, session.inflight - 1)
            session.last_seen = time.monotonic()
            if remove and self.sessions.get(session_id) is session:
                self.sessions.pop(session_id)

    def push(self, session_id: str, raw: bytes) -> dict[str, str]:
        model = self._require_model()
        if not raw or len(raw) > self.args.max_chunk_bytes:
            raise RequestError(f"float32 PCM chunk must contain 1-{self.args.max_chunk_bytes} bytes")
        if len(raw) % 4:
            raise RequestError("float32 PCM byte length must be divisible by four")
        samples = np.frombuffer(raw, dtype="<f4").reshape(-1)
        if not np.isfinite(samples).all():
            raise RequestError("float32 PCM contains non-finite samples")
        session = self._acquire_session(session_id)
        try:
            with session.lock, self.model_lock:
                model.streaming_transcribe(samples, session.state)
                return {
                    "language": getattr(session.state, "language", "") or "",
                    "text": getattr(session.state, "text", "") or "",
                }
        finally:
            self._release_session(session_id, session)

    def finish(self, session_id: str) -> dict[str, str]:
        model = self._require_model()
        session = self._acquire_session(session_id)
        completed = False
        try:
            with session.lock, self.model_lock:
                model.finish_streaming_transcribe(session.state)
                result = {
                    "language": getattr(session.state, "language", "") or "",
                    "text": getattr(session.state, "text", "") or "",
                }
            completed = True
            return result
        finally:
            self._release_session(session_id, session, remove=completed)

    def cancel(self, session_id: str) -> None:
        with self.sessions_lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            raise RequestError("invalid session_id")
        # Qwen3-ASR streaming state is in-process PCM/text state, not a remote
        # engine allocation. Cancellation deliberately drops it without a
        # final decode so abandoned microphone audio is never transcribed.

    def snapshot(self) -> dict[str, object]:
        with self.sessions_lock:
            active_sessions = len(self.sessions)
        return {
            "status": "ready" if self.ready else "not_ready",
            "ready": self.ready,
            "model": MODEL_NAME,
            "engine": ENGINE_NAME,
            "sampleRate": 16000,
            "format": "pcm_f32le",
            # Qwen buffers arbitrary transport frames until this internal
            # decode window is full. Keep the two concepts observable so the
            # 0.5s/1.0s deployment A/B cannot be confused with packet sizing.
            "decodeWindowSeconds": self.args.chunk_size_sec,
            "decodeWindowSamples": int(round(self.args.chunk_size_sec * 16000)),
            "activeSessions": active_sessions,
            "maxSessions": self.args.max_sessions,
            "maxModelLen": self.args.max_model_len,
            "uptimeSeconds": round(time.monotonic() - self.started_at, 3),
        }

    def shutdown(self) -> None:
        self.shutting_down = True
        with self.sessions_lock:
            self.sessions.clear()


def create_app(runtime: AsrRuntime) -> FastAPI:
    app = FastAPI(title="Mazi Qwen3-ASR Runtime", docs_url=None, redoc_url=None)

    @app.exception_handler(RequestError)
    async def request_error_handler(_request: Request, error: RequestError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content={"error": str(error)})

    @app.get("/")
    @app.get("/readyz")
    def readiness() -> JSONResponse:
        snapshot = runtime.snapshot()
        return JSONResponse(status_code=200 if runtime.ready else 503, content=snapshot)

    @app.get("/livez")
    def liveness() -> dict[str, object]:
        return {"status": "alive", "model": MODEL_NAME, "engine": ENGINE_NAME}

    @app.post("/api/start")
    def start() -> dict[str, str]:
        return {"session_id": runtime.start_session()}

    @app.post("/api/chunk")
    async def chunk(request: Request, session_id: str = Query(min_length=1, max_length=128)) -> dict[str, str]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/octet-stream":
            raise RequestError("expected application/octet-stream")
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > runtime.args.max_chunk_bytes:
                    raise RequestError("PCM chunk exceeds the configured byte limit", 413)
            except ValueError as error:
                raise RequestError("invalid Content-Length header") from error
        raw = await request.body()
        return await run_in_threadpool(runtime.push, session_id, raw)

    @app.post("/api/finish")
    def finish(session_id: str = Query(min_length=1, max_length=128)) -> dict[str, str]:
        return runtime.finish(session_id)

    @app.post("/api/cancel", status_code=204)
    def cancel(session_id: str = Query(min_length=1, max_length=128)) -> None:
        runtime.cancel(session_id)

    @app.on_event("shutdown")
    def shutdown() -> None:
        runtime.shutdown()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-model-path", default=MODEL_NAME)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--unfixed-chunk-num", type=int, default=4)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--max-sessions", type=int, default=4)
    parser.add_argument("--session-ttl-sec", type=float, default=120.0)
    parser.add_argument("--max-chunk-bytes", type=int, default=131072)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runtime = AsrRuntime(args)
    runtime.load()
    uvicorn.run(
        create_app(runtime),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()

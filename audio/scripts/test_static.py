#!/usr/bin/env python3
"""Offline contract checks for DGX audio profiles and smoke orchestration."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "scripts"))

import smoke  # noqa: E402
from profile import ProfileError, load_profile  # noqa: E402


BASELINE_PROFILE = ROOT / "profiles" / "qwen-live.args"
LOW_LATENCY_PROFILE = ROOT / "profiles" / "qwen-live-500ms.args"


def load_asr_server_module():
    class FakeArray:
        def reshape(self, *_shape):
            return self

    numpy = types.ModuleType("numpy")
    numpy.frombuffer = lambda *_args, **_kwargs: FakeArray()
    numpy.isfinite = lambda _samples: types.SimpleNamespace(all=lambda: True)
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = type("FastAPI", (), {})
    fastapi.Query = lambda *args, **kwargs: None
    fastapi.Request = type("Request", (), {})
    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.JSONResponse = type("JSONResponse", (), {})
    qwen_asr = types.ModuleType("qwen_asr")
    qwen_asr.Qwen3ASRModel = type("Qwen3ASRModel", (), {})
    starlette = types.ModuleType("starlette")
    starlette_concurrency = types.ModuleType("starlette.concurrency")
    starlette_concurrency.run_in_threadpool = None
    uvicorn = types.ModuleType("uvicorn")
    module_name = "mazi_test_qwen3_asr_server"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "asr" / "server.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load ASR server module")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            module_name: module,
            "fastapi": fastapi,
            "fastapi.responses": fastapi_responses,
            "numpy": numpy,
            "qwen_asr": qwen_asr,
            "starlette": starlette,
            "starlette.concurrency": starlette_concurrency,
            "uvicorn": uvicorn,
        },
    ):
        spec.loader.exec_module(module)
    return module


ASR_SERVER = load_asr_server_module()


class ProfileAbTest(unittest.TestCase):
    def test_ab_profiles_only_change_decode_window_and_description(self) -> None:
        baseline = load_profile(BASELINE_PROFILE)
        low_latency = load_profile(LOW_LATENCY_PROFILE)

        baseline_asr = dict(baseline.options["asr"])
        low_latency_asr = dict(low_latency.options["asr"])
        self.assertEqual(baseline_asr.pop("--chunk-size-sec"), "1.0")
        self.assertEqual(low_latency_asr.pop("--chunk-size-sec"), "0.5")
        self.assertEqual(baseline_asr, low_latency_asr)
        self.assertEqual(baseline.options["tts"], low_latency.options["tts"])
        self.assertEqual(baseline.meta["api_name"], low_latency.meta["api_name"])
        self.assertEqual(baseline.meta["llm_api_base"], low_latency.meta["llm_api_base"])

    def test_unapproved_decode_window_is_rejected(self) -> None:
        profile_text = BASELINE_PROFILE.read_text(encoding="utf-8").replace(
            "asr --chunk-size-sec 1.0",
            "asr --chunk-size-sec 0.75",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unapproved.args"
            path.write_text(profile_text, encoding="utf-8")
            with self.assertRaisesRegex(ProfileError, "approved A/B values"):
                load_profile(path)


class SmokeContractTest(unittest.TestCase):
    def test_small_transport_chunks_exercise_finish_and_cancel(self) -> None:
        profile = load_profile(BASELINE_PROFILE)
        next_session = 0
        chunk_sizes: list[int] = []
        cancelled_sessions: list[str] = []

        def fake_request_json(
            url: str,
            *,
            method: str = "GET",
            body: bytes | None = None,
            headers: dict[str, str] | None = None,
            timeout: float = 30.0,
        ) -> dict:
            del method, headers, timeout
            nonlocal next_session
            parsed = urllib.parse.urlparse(url)
            if parsed.path == "/readyz":
                return {
                    "ready": True,
                    "model": "Qwen/Qwen3-ASR-1.7B",
                    "engine": "qwen-asr-vllm",
                    "decodeWindowSeconds": 1.0,
                }
            if parsed.path == "/api/start":
                next_session += 1
                return {"session_id": f"session-{next_session}"}
            if parsed.path == "/api/chunk":
                self.assertIsNotNone(body)
                chunk_sizes.append(len(body or b""))
                return {"language": "", "text": ""}
            if parsed.path == "/api/finish":
                return {"language": "", "text": ""}
            self.fail(f"unexpected JSON request: {url}")

        def fake_request_no_content(
            url: str,
            *,
            method: str = "POST",
            timeout: float = 30.0,
        ) -> None:
            del method, timeout
            parsed = urllib.parse.urlparse(url)
            self.assertEqual(parsed.path, "/api/cancel")
            cancelled_sessions.extend(urllib.parse.parse_qs(parsed.query)["session_id"])

        with (
            mock.patch.object(smoke, "request_json", side_effect=fake_request_json),
            mock.patch.object(
                smoke,
                "request_no_content",
                side_effect=fake_request_no_content,
            ),
        ):
            result = smoke.asr_smoke(profile)

        self.assertEqual(result["transportChunkMilliseconds"], 100)
        self.assertEqual(result["transportChunkBytes"], 6_400)
        self.assertEqual(result["transportChunkCount"], 12)
        self.assertEqual(chunk_sizes, [6_400] * 13)
        self.assertEqual(cancelled_sessions, ["session-2"])
        self.assertEqual(
            result["contracts"],
            ["start/chunk/finish", "start/chunk/cancel"],
        )

    def test_full_smoke_starts_asr_tts_and_llm_concurrently(self) -> None:
        profile = load_profile(BASELINE_PROFILE)
        barrier = threading.Barrier(3, timeout=2)

        def concurrent_check(name: str):
            def check(_profile):
                barrier.wait()
                return {"name": name}

            return check

        with (
            mock.patch.object(smoke, "asr_smoke", new=concurrent_check("asr")),
            mock.patch.object(smoke, "tts_smoke", new=concurrent_check("tts")),
            mock.patch.object(smoke, "llm_smoke", new=concurrent_check("llm")),
        ):
            result = smoke.run_smoke(profile, audio_only=False)

        self.assertTrue(result["concurrent"])
        self.assertEqual(set(result["checks"]), {"asr", "tts", "llm"})


class AsrSessionExpiryTest(unittest.TestCase):
    def test_ttl_does_not_collect_push_or_finish_waiting_for_the_model(self) -> None:
        for operation in ("push", "finish"):
            with self.subTest(operation=operation):
                entered_model = threading.Event()
                release_model = threading.Event()

                class BlockingModel:
                    def streaming_transcribe(self, _samples, state) -> None:
                        entered_model.set()
                        self._wait_for_release()
                        state.text = "partial"

                    def finish_streaming_transcribe(self, state) -> None:
                        entered_model.set()
                        self._wait_for_release()
                        state.text = "final"

                    def _wait_for_release(self) -> None:
                        if not release_model.wait(timeout=1):
                            raise RuntimeError("test model release timed out")

                args = types.SimpleNamespace(
                    session_ttl_sec=0.02,
                    max_chunk_bytes=16,
                )
                runtime = ASR_SERVER.AsrRuntime(args)
                runtime.model = BlockingModel()
                session_id = f"{operation}-session"
                state = types.SimpleNamespace(language="", text="")
                session = ASR_SERVER.Session(
                    state=state,
                    created_at=time.monotonic(),
                    last_seen=time.monotonic(),
                )
                runtime.sessions[session_id] = session
                errors: list[BaseException] = []

                def invoke() -> None:
                    try:
                        if operation == "push":
                            runtime.push(session_id, b"\0\0\0\0")
                        else:
                            runtime.finish(session_id)
                    except BaseException as error:  # pragma: no cover - assertion below reports it
                        errors.append(error)

                with runtime.model_lock:
                    worker = threading.Thread(target=invoke, daemon=True)
                    worker.start()
                    deadline = time.monotonic() + 1
                    while session.inflight != 1 and time.monotonic() < deadline:
                        time.sleep(0.001)
                    self.assertEqual(session.inflight, 1)
                    time.sleep(0.03)
                    runtime._expire_sessions()
                    self.assertIs(runtime.sessions.get(session_id), session)

                self.assertTrue(entered_model.wait(timeout=1))
                time.sleep(0.03)
                runtime._expire_sessions()
                self.assertIs(runtime.sessions.get(session_id), session)
                release_model.set()
                worker.join(timeout=1)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(session.inflight, 0)

                if operation == "push":
                    self.assertIs(runtime.sessions.get(session_id), session)
                    time.sleep(0.03)
                    runtime._expire_sessions()
                    self.assertNotIn(session_id, runtime.sessions)
                else:
                    self.assertNotIn(session_id, runtime.sessions)


if __name__ == "__main__":
    unittest.main(verbosity=2)

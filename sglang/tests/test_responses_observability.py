"""CPU-only regression checks for logging using the deployed RequestLogger."""

import ast
import asyncio
import io
import json
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils.log_utils import log_json
from sglang.srt.utils.request_logger import RequestLogger


SOURCE = Path("/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py")
tree = ast.parse(SOURCE.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingResponses")
methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and n.name in {"_log_responses_request", "_log_responses_terminal"}]
assert len(methods) == 2
scope = {"log_json": log_json}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), scope)


class RawRequest:
    headers = {"authorization": "SECRET-HEADER-MUST-NOT-BE-LOGGED", "x-smg-routing-key": "probe"}

    def __init__(self, payload):
        self.payload = payload
        self.reads = 0

    async def json(self):
        self.reads += 1
        return self.payload


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.sink = logging.Logger("responses-observability-test")
        self.sink.addHandler(logging.StreamHandler(self.stream))
        self.log = RequestLogger(True, 3, "json", None)
        self.log.targets = [self.sink]
        self.owner = SimpleNamespace(tokenizer_manager=SimpleNamespace(request_logger=self.log))
        self.request = SimpleNamespace(request_id="resp_test", max_output_tokens=32768, stream=True)

    def raw(self, body):
        raw = RawRequest(body)
        asyncio.run(scope["_log_responses_request"](self.owner, self.request, raw))
        return raw

    def terminal(self, status, reason):
        response = {"id": "resp_test", "status": status,
                    "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
                    "error": {"code": "server_error"} if status == "failed" else None,
                    "usage": {"output_tokens": 999}}
        scope["_log_responses_terminal"](self.owner, self.request, response, reason, {"max_new_tokens": 32766})
        return json.loads(self.stream.getvalue().splitlines()[-1])

    def test_original_body_and_generated_id_are_kept_separate_without_truncation(self):
        payload = {"input": "原始提示" * 3000, "max_output_tokens": 32768, "stream": True}
        self.raw(payload)
        record = json.loads(self.stream.getvalue())
        self.assertEqual(record["obj"], {"request_id": "resp_test", "request": payload})
        self.assertNotIn("SECRET-HEADER", self.stream.getvalue())
        self.assertEqual(record["headers"], {"x-smg-routing-key": "probe"})

    def test_disabled_logging_does_not_read_or_emit_body(self):
        self.log.log_requests = False
        raw = self.raw({"input": "private"})
        self.assertEqual(raw.reads, 0)
        self.assertEqual(self.stream.getvalue(), "")

    def test_metadata_level_does_not_read_or_emit_body(self):
        self.log.log_requests_level = 1
        raw = self.raw({"input": "private"})
        self.assertEqual(raw.reads, 0)
        self.assertEqual(self.stream.getvalue(), "")

    def test_terminal_preserves_actual_engine_reason_without_inferring_from_usage(self):
        for status, reason in [("incomplete", {"type": "length", "length": 32766}),
                               ("completed", {"type": "stop", "matched": 123}),
                               ("failed", {"type": "abort", "message": "test abort"})]:
            with self.subTest(status=status):
                record = self.terminal(status, reason)
                self.assertEqual(record["engine_finish_reason"], reason)
                self.assertEqual(record["status"], status)
                self.assertEqual(record["max_output_tokens"], 32768)
                self.assertEqual(record["adapted_max_new_tokens"], 32766)
                self.assertEqual(record["rid"], record["response_id"])

    def test_source_hooks_cover_request_full_response_and_stream_terminal(self):
        expected = {"create_responses": "_log_responses_request",
                    "responses_full_generator": "_log_responses_terminal",
                    "_responses_stream_generator_non_harmony": "_log_responses_terminal"}
        for method, hook in expected.items():
            node = next(n for n in cls.body if getattr(n, "name", "") == method)
            calls = [n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
            self.assertIn(hook, calls)


if __name__ == "__main__":
    unittest.main()

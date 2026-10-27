"""Add correlated Responses request/terminal logs to the pinned DGX runtime.

This does not change request adaptation, sampling, termination or response data.
"""

import ast
import hashlib
import sys
from pathlib import Path


TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_responses.py"
)
# Audited official 2026-09-14 and 2026-10-15/16 nightly sources. Fail closed when
# upstream changes the request/terminal paths so this patch is reviewed again.
EXPECTED = {
    "a641d482c3760090dddddab823a43664974cd33e34a3683fbf5667ebed212aed",
    "641ac81e77d055e453d06825cb0a1588295f12b3f14647f0f0e8fb997851da3d",
}
text = TARGET.read_text()
if "async def _log_responses_request(" in text:
    raise SystemExit("Responses observability patch is already present")
assert hashlib.sha256(TARGET.read_bytes()).hexdigest() in EXPECTED, "Unexpected source revision"


def replace_once(old, new):
    global text
    assert text.count(old) == 1, f"Expected one patch anchor: {old[:100]!r}"
    text = text.replace(old, new, 1)


replace_once(
    "from sglang.srt.utils import random_uuid\n",
    "from sglang.srt.utils import random_uuid\n"
    "from sglang.srt.utils.log_utils import log_json\n",
)
replace_once(
    "        # Handle the previous response ID\n",
    "        await self._log_responses_request(request, raw_request)\n\n"
    "        # Handle the previous response ID\n",
)
replace_once(
    "        response.error = self._error_from_finish_reason(finish_reason)\n",
    "        response.error = self._error_from_finish_reason(finish_reason)\n"
    "        self._log_responses_terminal(\n"
    "            request, response.model_dump(), finish_reason, sampling_params\n"
    "        )\n",
)
replace_once(
    '            failed["error"] = {"code": "server_error", "message": str(e)}\n',
    '            failed["error"] = {"code": "server_error", "message": str(e)}\n'
    "            self._log_responses_terminal(\n"
    "                request, failed, finish_reason, sampling_params\n"
    "            )\n",
)
replace_once(
    "        yield _send_event(self._terminal_stream_event(response_dict))\n",
    "        self._log_responses_terminal(\n"
    "            request, response_dict, finish_reason, sampling_params\n"
    "        )\n"
    "        yield _send_event(self._terminal_stream_event(response_dict))\n",
)
replace_once(
    "    @staticmethod\n    def _status_from_finish_reason(finish_reason: Any) -> str:\n",
    '''    async def _log_responses_request(self, request, raw_request):
        # Reuse RequestLogger's enable switch, content level and header allowlist.
        request_logger = self.tokenizer_manager.request_logger
        if not request_logger.log_requests or request_logger.log_requests_level < 2:
            return
        payload = (
            await raw_request.json()
            if raw_request is not None
            else request.model_dump(exclude_none=True)
        )
        request_logger.log_openai_received_request(
            {"request_id": request.request_id, "request": payload},
            request=raw_request,
        )

    def _log_responses_terminal(self, request, response, finish_reason, sampling_params):
        request_logger = self.tokenizer_manager.request_logger
        if not request_logger.log_requests:
            return
        effective_limit = (
            sampling_params.get("max_new_tokens")
            if isinstance(sampling_params, dict)
            else getattr(sampling_params, "max_new_tokens", None)
        )
        log_json(
            request_logger.targets,
            "response.terminal",
            {
                "rid": request.request_id,
                "response_id": response.get("id"),
                "status": response.get("status"),
                "stream": request.stream,
                "max_output_tokens": request.max_output_tokens,
                "adapted_max_new_tokens": effective_limit,
                "engine_finish_reason": finish_reason,
                "incomplete_details": response.get("incomplete_details"),
                "error": response.get("error"),
                "usage": response.get("usage"),
            },
        )

    @staticmethod
    def _status_from_finish_reason(finish_reason: Any) -> str:
''',
)
ast.parse(text)
TARGET.write_text(text)
print("Patched Responses request and terminal logging:", hashlib.sha256(TARGET.read_bytes()).hexdigest())

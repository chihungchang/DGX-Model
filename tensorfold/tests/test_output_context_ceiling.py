"""GPU-free checks against the pinned, real request and HTTP owners."""

import json

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_cuda_request_policy import app_for
from tensorfold.server.errors import ContextLengthError


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_incident_prompt_preserved_and_output_uses_remaining_capacity(tmp_path, field):
    app = app_for(tmp_path)
    app.context_window = 262_144
    prompt = [ord("x")] * 207_521
    body = {"prompt": prompt, field: 65_535}
    assert app.check(body) is None
    prepared = app.prepare(body, False)
    assert prepared.prompt == prompt
    assert prepared.max_tokens == 54_623
    assert body[field] == 65_535


@pytest.mark.parametrize("asked,expected", [(8, 8), (32, 32), (33, 32), (65_535, 32)])
def test_ceiling_never_expands_and_does_not_remove_input(tmp_path, asked, expected):
    app = app_for(tmp_path)
    app.context_window = 64
    body = {"prompt": [ord("x")] * 32, "max_tokens": asked}
    prepared = app.prepare(body, False)
    assert prepared.prompt == body["prompt"]
    assert prepared.max_tokens == expected


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("chat", [False, True])
def test_http_accepts_fitting_input_and_reports_actual_usage(tmp_path, field, stream, chat):
    app = app_for(tmp_path)
    app.context_window = 64
    body = ({"messages": [{"role": "user", "content": "Hello"}]} if chat
            else {"prompt": "Hello"})
    body.update({field: 65_535, "stream": stream,
                 "stream_options": {"include_usage": True}})
    with http_server(app) as port:
        status, payload = post(port, body, chat)
    assert status == 200
    assert len(app.engine.calls) == 1
    prompt, output_limit, _ = app.engine.calls[0]
    assert output_limit == 64 - len(prompt)
    if stream:
        events = [json.loads(line[5:]) for line in payload.splitlines()
                  if line.startswith("data:") and line != "data: [DONE]"]
        usage = next(event["usage"] for event in events if event.get("usage"))
    else:
        usage = json.loads(payload)["usage"]
    assert usage["prompt_tokens"] == len(prompt)
    assert usage["completion_tokens"] == len("Hello")
    assert usage["total_tokens"] == len(prompt) + len("Hello")


@pytest.mark.parametrize("length", [64, 65])
def test_full_input_still_refuses_before_generation(tmp_path, length):
    app = app_for(tmp_path)
    app.context_window = 64
    body = {"prompt": [ord("x")] * length, "max_tokens": 1}
    with pytest.raises(ContextLengthError):
        app.prepare(body, False)
    with http_server(app) as port:
        status, payload = post(port, {**body, "stream": True}, False)
    assert status == 400
    assert json.loads(payload)["error"]["code"] == "context_length_exceeded"
    assert not app.engine.calls


def test_one_remaining_token_is_usable(tmp_path):
    app = app_for(tmp_path)
    app.context_window = 64
    prepared = app.prepare({"prompt": [ord("x")] * 63, "max_tokens": 65_535}, False)
    assert len(prepared.prompt) == 63
    assert prepared.max_tokens == 1


def test_engine_capacity_remains_an_upper_bound(tmp_path):
    app = app_for(tmp_path)
    app.context_window = 262_144
    app.engine.context_window = 64
    prepared = app.prepare({"prompt": [ord("x")] * 32, "max_tokens": 65_535}, False)
    assert prepared.max_tokens == 32

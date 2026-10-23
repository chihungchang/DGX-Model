"""Actual DeepSeek image decoding and App admission, without GPU/model weights."""
import base64
import copy
import io
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from tensorfold.cuda.server import App, PreparedRequest
from tensorfold.families.deepseek_v41.cuda.app import DsApp
from tensorfold.families.deepseek_v41.cuda import vision as v
from tensorfold.server.errors import ContextLengthError, RequestError


def frontend():
    cfg = v.VisionConfig(1, 8, 1, 8, 14, 10000, 3, 4096, 0, None, 99, 8)
    return v.DsVision(cfg)


def history(count, role="user"):
    result = []
    for i in range(count):
        buf = io.BytesIO()
        Image.new("RGB", (42, 42), (i % 255, 90, 200)).save(buf, format="PNG")
        image = {"type": "image_url", "image_url": {"url":
            "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}
        result.append({"role": role, "content": [{"type": "text", "text": str(i)}, image]})
    return result


@pytest.mark.parametrize("count", [8, 9, 16, 64])
@pytest.mark.parametrize("role", ["user", "tool"])
def test_history_is_preserved_beyond_eight(count, role):
    messages = history(count, role)
    original = copy.deepcopy(messages)
    rendered, pictures = frontend().split(messages)
    assert len(pictures) == count
    assert [m["content"] for m in rendered] == [str(i) + "\n\n" + v.PLACEHOLDER for i in range(count)]
    assert messages == original
    assert all(p.tokens == 4 for p in pictures)
    assert not pictures[0].patches.equal(pictures[-1].patches)


def app(monkeypatch, count_limit=None, context=1000):
    def base_init(self, *_args, **_kwargs):
        self.template = NS(render=lambda *_a, **_k: "")
        self.engine = NS(vision=frontend())
        self.context_window = context
        self.native_context_window = context
        self.sampling_for = lambda *_a: None
    monkeypatch.setattr(App, "__init__", base_init)
    def base_prepare(self, body, chat):
        tokens = [1]
        for m in body["messages"]:
            tokens.extend([99] * m["content"].count(v.PLACEHOLDER))
        return PreparedRequest(tokens, 2, [], False, None)
    monkeypatch.setattr(App, "_prepare", base_prepare)
    return DsApp(vision_max_images=count_limit)


def test_app_honors_explicit_count_and_recovers(monkeypatch):
    bounded = app(monkeypatch, count_limit=9)
    prepared = bounded._prepare({"messages": history(9)}, True)
    assert len(prepared.vision.spans) == 9
    assert len(prepared.prompt) == 37
    with pytest.raises(RequestError, match="at most 9"):
        bounded._prepare({"messages": history(10)}, True)
    assert len(bounded._prepare({"messages": history(9)}, True).vision.spans) == 9
    normal = app(monkeypatch)
    assert len(normal._prepare({"messages": history(16)}, True).vision.spans) == 16


def test_context_admission_stops_decoding_and_is_typed(monkeypatch):
    target = app(monkeypatch, context=36)
    original = v.decode
    decoded = []
    def decode(*args, **kwargs):
        decoded.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(v, "decode", decode)
    with pytest.raises(ContextLengthError, match="maximum context length"):
        target._prepare({"messages": history(16)}, True)
    assert len(decoded) == 9
    assert len(target._prepare({"messages": history(8)}, True).vision.spans) == 8


def test_complete_expanded_prompt_still_uses_existing_context_check(monkeypatch):
    target = app(monkeypatch, context=37)
    monkeypatch.setattr(App, "_check_fields", lambda *_a: None)
    with pytest.raises(ContextLengthError):
        target.prepare({"messages": history(9), "max_tokens": 2}, True)


def test_image_bytes_and_role_guards_remain():
    target = frontend()
    target.max_bytes = 10
    with pytest.raises(ValueError, match="larger than"):
        target.split(history(9))
    with pytest.raises(ValueError, match="only in user"):
        frontend().split(history(1, "assistant"))


def test_configured_constructor_limit_still_applies():
    target = frontend()
    target.max_images = 2
    with pytest.raises(ValueError, match="at most 2"):
        target.split(history(3))

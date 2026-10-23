"""Adapt pinned upstream admission expectations to the local output ceiling.

Run only inside an isolated test container. Input overflow still refuses; an
explicit output ceiling may consume the remaining window without reserving it.
The unrelated GLM-specific admission policy is deliberately retained.
"""

from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Pinned admission test changed: " + old[:80])
    return source.replace(old, new)


def patch(path):
    source = path.read_text()
    source = replace_once(source,
        "def test_native_boundary_refuses_explicit_completion_alias(model_dir, chat):",
        "def test_native_boundary_caps_explicit_completion_alias(model_dir, chat):")
    source = replace_once(source,
        '''        status, payload = post(port, request(11, chat, max_completion_tokens=9), chat)
    assert status == 400''',
        '''        status, payload = post(port, request(11, chat, max_completion_tokens=9), chat)
        assert status == 200
        assert len(engine.calls) == 1
        assert len(engine.calls[0][0]) == 11 and engine.calls[0][1] == 1
        status, payload = post(port, request(12, chat, max_completion_tokens=9), chat)
    assert status == 400''')
    source = replace_once(source,
        '''    assert "9 reply tokens" in error["message"] and "12-token context window" in error["message"]''',
        '''    assert "no room for a reply" in error["message"] and "12-token context window" in error["message"]''')
    source = replace_once(source,
        '''    assert re.search(r"maximum context length is \\d+ tokens", error["message"])
    assert not engine.calls''',
        '''    assert re.search(r"maximum context length is \\d+ tokens", error["message"])
    assert len(engine.calls) == 1''')
    source = replace_once(source,
        "def test_explicit_reply_overflow_refuses_before_generation_and_recovers(model_dir, chat, stream):",
        "def test_explicit_reply_ceiling_uses_remaining_capacity(model_dir, chat, stream):")
    source = replace_once(source,
        '''        assert status == 400
        error = json.loads(payload)["error"]
        assert error["type"] == "invalid_request_error"
        assert "9 reply tokens" in error["message"] and "13-token safe cache capacity" in error["message"]
        assert "reduce" in error["message"]
        assert not engine.calls
        status, _ = post(port, request(12, chat, max_tokens=1, stream=stream), chat)
    assert status == 200 and engine.calls[0][1] == 1''',
        '''        assert status == 200
        assert len(engine.calls) == 1
        assert len(engine.calls[0][0]) == 12 and engine.calls[0][1] == 1
        assert "answer" in payload
        status, _ = post(port, request(12, chat, max_tokens=1, stream=stream), chat)
    assert status == 200 and len(engine.calls) == 2
    assert engine.calls[1] == engine.calls[0]''')
    source = replace_once(source,
        '''        problem = app.check(request(8, False, max_tokens=5))''',
        '''        problem = app.check(request(12, False, max_tokens=1))''')
    compile(source, str(path), "exec")
    path.write_text(source)


if __name__ == "__main__":
    patch(Path("/opt/tensorfold/tests/test_cuda_admission.py"))

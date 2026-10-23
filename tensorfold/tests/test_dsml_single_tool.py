"""GPU-free regressions against the real pinned CUDA request/stream owner."""

import json

import pytest

from tests.test_cuda_admission import http_server, post
from tests.test_cuda_request_policy import app_for
from tests.test_request_policy import TOOLS, stream_calls


def dsml(value):
    return ('<｜DSML｜ calls><｜DSML｜ invoke name="measure">'
            '<｜DSML｜ parameter name="value" string="false">'
            + json.dumps(value) + '</｜DSML｜ parameter>'
            '</｜DSML｜ invoke></｜DSML｜ calls>')


class ChunkedEngine:
    eos = (0,)

    def __init__(self, text, width):
        self.text, self.width = text, width

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        tokens = [ord(c) for c in self.text] + [0]
        for start in range(0, len(tokens), self.width):
            if on_tokens(tokens[start:start + self.width]):
                break
        return {"generated": len(tokens)}


def setup(tmp_path, text, width):
    app = app_for(tmp_path)
    app.engine = ChunkedEngine(text, width)
    app.tok.token_to_id = lambda _: None
    return app


@pytest.mark.parametrize("width", [1, 2, 7, 10000])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_dsml_deltas_hide_markup_and_keep_arguments_and_call_limit(tmp_path, width, thinking, parallel):
    answer = 'Before.' + dsml(17) + dsml(25) + '之後。'
    raw = ('Check the city.</think>\n\n' if thinking else '') + answer
    app = setup(tmp_path, raw, width)
    deltas = []
    result = app.run({"messages": [{"role": "user", "content": "Measure"}],
                      "tools": TOOLS, "parallel_tool_calls": parallel,
                      "chat_template_kwargs": {"enable_thinking": thinking}},
                     True, lambda delta: deltas.append(delta) or True)
    content = ''.join(d.get('content', '') for d in deltas) + result['final'].get('content', '')
    assert content == result['content'] == 'Before.之後。'
    assert result['finish'] == 'tool_calls'
    assert [json.loads(c['function']['arguments']) for c in result['calls']] == (
        [{'value': 17}, {'value': 25}] if parallel else [{'value': 17}])
    assert all(c['function']['name'] == 'measure' for c in result['calls'])
    assert ''.join(d.get('reasoning_content', '') for d in deltas) == (
        'Check the city.' if thinking else '')
    assert result['completion_tokens'] == len(raw) + 1


@pytest.mark.parametrize("stream", [False, True])
def test_single_dsml_call_http_json_and_sse_agree(tmp_path, stream):
    app = setup(tmp_path, 'Ready.' + dsml(42), 1)
    with http_server(app) as port:
        status, body = post(port, {"messages": [{"role": "user", "content": "Measure"}],
                                  "tools": TOOLS, "parallel_tool_calls": False,
                                  "stream": stream, "stream_options": {"include_usage": True}}, True)
    assert status == 200
    if stream:
        calls = stream_calls(body)
        events = [json.loads(line[5:]) for line in body.splitlines()
                  if line.startswith('data:') and line != 'data: [DONE]']
        content = ''.join(choice['delta'].get('content', '')
                          for event in events for choice in event.get('choices', []))
        assert len(calls) == 1 and calls[0]['name'] == 'measure'
        assert json.loads(calls[0]['arguments']) == {'value': 42}
        assert any(choice.get('finish_reason') == 'tool_calls'
                   for event in events for choice in event.get('choices', []))
    else:
        choice = json.loads(body)['choices'][0]
        content = choice['message']['content']
        assert choice['finish_reason'] == 'tool_calls'
        calls = choice['message']['tool_calls']
        assert len(calls) == 1
        assert json.loads(calls[0]['function']['arguments']) == {'value': 42}
    assert content == 'Ready.'


@pytest.mark.parametrize("tools", [[], TOOLS])
def test_normal_prose_is_preserved(tmp_path, tools):
    text = '正常中文 <math>x</math> & ordinary text.'
    app = setup(tmp_path, text, 1)
    deltas = []
    result = app.run({"messages": [{"role": "user", "content": "Hi"}],
                      "tools": tools, "parallel_tool_calls": False},
                     True, lambda delta: deltas.append(delta) or True)
    assert ''.join(d.get('content', '') for d in deltas) + result['final'].get('content', '') == text
    assert result['content'] == text and result['calls'] is None


def test_incomplete_dsml_never_leaks_or_executes(tmp_path):
    app = setup(tmp_path, 'Before.' + dsml(42).split('</｜DSML｜ parameter>')[0], 1)
    deltas = []
    result = app.run({"messages": [{"role": "user", "content": "Measure"}],
                      "tools": TOOLS, "parallel_tool_calls": False},
                     True, lambda delta: deltas.append(delta) or True)
    assert ''.join(d.get('content', '') for d in deltas) + result['final'].get('content', '') == 'Before.'
    assert result['content'] == 'Before.'
    assert result['calls'] is None

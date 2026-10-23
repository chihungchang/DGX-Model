"""Treat an explicit output limit as a ceiling, not reserved prompt capacity.

The pinned CUDA server already limits generation to the remaining context in
App.prepare. Its earlier explicit-limit check prevents that code from running.
Keep exact rendered-input admission and the engine's capacity protections.
"""

from pathlib import Path


def patch(path):
    source = path.read_text()
    old = '''        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        if limit is not None and asked and len(prepared.prompt) + prepared.max_tokens > limit:
            kind = "safe cache capacity" if limit == self._engine_capacity() else "context window"
            return (f"{CONTEXT_LIMIT} {limit} tokens: the rendered prompt has {len(prepared.prompt)} tokens and "
                    f"requests {prepared.max_tokens} reply tokens, which exceeds the context window (the server's "
                    f"{limit}-token {kind}); reduce the prompt or reply length"
                    f"{self._restart(len(prepared.prompt) + prepared.max_tokens)}")
'''
    new = '''        # max_tokens is a generation ceiling, not a reservation against input.
        # prepare() caps it to the remaining exact rendered context before the
        # engine receives the request. A full prompt still fails above.
'''
    if source.count(old) != 1:
        raise RuntimeError("Pinned CUDA output-admission source changed")
    if source.count("prepared.max_tokens = min(prepared.max_tokens, limit - len(prepared.prompt))") != 1:
        raise RuntimeError("Pinned CUDA remaining-context generation guard changed")
    updated = source.replace(old, new)
    compile(updated, str(path), "exec")
    path.write_text(updated)


if __name__ == "__main__":
    patch(Path("/opt/tensorfold/src/tensorfold/cuda/server.py"))

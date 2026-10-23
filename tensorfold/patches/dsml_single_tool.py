"""Keep CUDA's family-specific tool text filter active for single-call requests.

Pinned TensorFold 07aa025 bypasses hide_tool_calls when parallel_tool_calls is
false. Its generic single-call policy recognizes tool_call envelopes, but not
DeepSeek V4.1's spaced DSML tags. The final parser still emits the formal call,
so streaming leaks the raw envelope as additional assistant content.
"""

from pathlib import Path


def patch(path):
    source = path.read_text()
    old = """                answer = (policy.content(answer, finished=finished) if policy.single
                          else hide_tool_calls(answer, finished=finished))"""
    new = """                # Family markup must be hidden in both single and parallel modes.
                # The generic single-call policy alone does not recognize DSML.
                answer = hide_tool_calls(answer, finished=finished)
                if policy.single:
                    answer = policy.content(answer, finished=finished)"""
    if source.count(old) != 1:
        raise RuntimeError("Pinned TensorFold CUDA tool visibility source changed")
    source = source.replace(old, new)
    old_final = "        content = policy.content(content) if tools else content"
    new_final = ("        # A truncated DSML block can remain after the completed-call parser.\n"
                 "        content = (policy.content(hide_tool_calls(content, finished=True))\n"
                 "                   if tools and policy.single else content)")
    if source.count(old_final) != 1:
        raise RuntimeError("Pinned TensorFold final tool visibility source changed")
    source = source.replace(old_final, new_final)
    compile(source, str(path), "exec")
    path.write_text(source)


if __name__ == "__main__":
    patch(Path("/opt/tensorfold/src/tensorfold/cuda/server.py"))

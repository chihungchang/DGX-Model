#!/usr/bin/env python3
"""Use SGLang's native mRoPE path for multidimensional image positions."""

from pathlib import Path


TARGETS = (
    Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/models/qwen3_5.py"),
    Path("/opt/sglang/sglang-src/python/sglang/srt/models/qwen3_5.py"),
    Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py"),
)
OLD = "        if _is_cuda and self.attn_output_gate:\n"
NEW = "        if _is_cuda and self.attn_output_gate and positions.ndim == 1:\n"


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"mRoPE guard already present: {path}")
        return
    occurrences = text.count(OLD)
    if occurrences != 1:
        raise RuntimeError(
            f"Expected exactly one Qwen3.5 CUDA fused dispatch in {path}, "
            f"found {occurrences}"
        )
    path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"Patched multidimensional mRoPE dispatch: {path}")


existing = [path for path in TARGETS if path.is_file()]
if not existing:
    raise RuntimeError("SGLang Qwen3.5 implementation was not found in the base image")
for target in existing:
    patch(target)

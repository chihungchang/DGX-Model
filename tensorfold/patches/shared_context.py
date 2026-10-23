"""Keep the model's per-request context limit separate from shared KV storage.

The pinned DeepSeek engine otherwise uses --context for both. Four full
256K inputs plus replies cannot fit in its 1M native request limit. This
changes only shared storage; the CLI and API retain their native limit.
"""

from pathlib import Path


def patch(path):
    source = path.read_text()
    replacements = [
        (
            "        self.limit = int(context or 65536)",
            "        self.limit = int(context or 65536)\n"
            '        self.pool_context = int(os.environ.get("TF_DS_POOL_CONTEXT") or self.limit)\n'
            "        if self.pool_context < self.limit:\n"
            '            raise ValueError("TF_DS_POOL_CONTEXT must be at least --context")',
        ),
        (
            "cap=self.limit + self.max_rows + 8)",
            "cap=self.pool_context + self.max_rows + 8)",
        ),
        (
            "int(markov.ON), int(markov.SPLIT)],",
            "int(markov.ON), int(markov.SPLIT), self.pool_context],",
        ),
        (
            "(world * 4,), dtype=torch.int64",
            "(world * 5,), dtype=torch.int64",
        ),
        (
            "every.view(world, 4).tolist()",
            "every.view(world, 5).tolist()",
        ),
        (
            '"--vision, --parallel, TF_DS_MARKOV and TF_DS_MARKOV_SPLIT must be the same on every "',
            '"--vision, --parallel, TF_DS_MARKOV, TF_DS_MARKOV_SPLIT and TF_DS_POOL_CONTEXT "\n'
            '                                 "must be the same on every "',
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Pinned TensorFold context source changed: " + old)
        source = source.replace(old, new)
    path.write_text(source)


def patch_allocator(path):
    source = path.read_text()
    old = """        while True:
            base = self.extents.take(need)
"""
    new = """        # A large fresh batch can use every slot, but an unrelated kept prompt
        # can leave a hole one alignment unit short of a new stream's extent.
        # Reclaim inactive entries before placing the first such stream so the
        # remaining batch sees one contiguous pool. Matched prefixes retain
        # their usual reuse path; persisted entries remain on disk.
        if (src is None and not self.streams and not self.filling
                and need * len(self.slots) <= self.extents.total < need * (len(self.slots) + 1)):
            for kept in list(self.kept.values()):
                self._forget(kept)
        while True:
            base = self.extents.take(need)
"""
    if source.count(old) != 1:
        raise RuntimeError("Pinned TensorFold extent placement changed")
    source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)


if __name__ == "__main__":
    patch(Path("/opt/tensorfold/src/tensorfold/families/deepseek_v41/cuda/engine.py"))
    patch_allocator(Path("/opt/tensorfold/src/tensorfold/families/deepseek_v41/cuda/multi.py"))

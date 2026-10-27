#!/usr/bin/env python3
"""Bounded, CPU-only regression for cached multi-image pickle amplification.

Run in the pinned SGLang image. Exercise the real artifact splitter with a fake
HF preprocessor, then both IPC pickle round trips; no model is loaded.
"""

import gc
import hashlib
import json
import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.cache.preprocess_cache import MultimodalPreprocessCache
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor


def check(count: int) -> dict:
    rows, width = 32, 512
    features = torch.arange(count * rows * width, dtype=torch.float32).reshape(
        count * rows, width
    )
    grids = torch.tensor([[1, 1, rows]] * count, dtype=torch.int64)
    processor = SimpleNamespace(
        image_processor=lambda **kwargs: {
            "pixel_values": features,
            "image_grid_thw": grids,
        }
    )
    splitter = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    splitter._resolve_processor = lambda _: (processor, None)
    splitter.image_config = {}
    splitter._temporary_fast_processor_cuda_pool = lambda _: nullcontext()
    splitter.mm_preprocess_cache = SimpleNamespace(enabled=True)
    splitter.precompute_hash_before_cpu_transfer = False
    keys = ["sha256:" + hashlib.sha256(str(i).encode()).hexdigest() for i in range(count)]
    entries = [
        SimpleNamespace(media=None, content_digest=key, artifact_key=key)
        for key in keys
    ]
    artifacts = splitter.prepare_artifact_batch(entries)
    logical_bytes = features.numel() * features.element_size()
    cache = MultimodalPreprocessCache(logical_bytes + count * 4096)
    for i, artifact in enumerate(artifacts):
        torch.testing.assert_close(artifact.feature, features[i * rows : (i + 1) * rows])
        assert artifact.feature.untyped_storage().nbytes() == rows * width * 4
        grid = artifact.model_specific_data["image_grid_thw"]
        assert grid.untyped_storage().nbytes() == 3 * 8
        assert cache.put(keys[i], artifact.cache_value())
    assert len(cache) == count, "Cache accounting must reflect individual images"

    measurements = {}
    for label, selected in (
        ("cold", artifacts),
        ("warm", [cache.get(key) for key in keys]),
    ):
        assert all(item is not None for item in selected)
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=artifact.feature,
                offsets=[(i, i)],
                model_specific_data=artifact.model_specific_data,
            )
            for i, artifact in enumerate(selected)
        ]
        # Tokenizer -> TP0 (highest protocol), then TP0 -> TP1 (default).
        for protocol in (pickle.HIGHEST_PROTOCOL, pickle.DEFAULT_PROTOCOL):
            payload = pickle.dumps(items, protocol=protocol)
            assert len(payload) < logical_bytes * 1.1 + 16384, len(payload)
            items = pickle.loads(payload)
            for i, item in enumerate(items):
                torch.testing.assert_close(item.feature, features[i * rows : (i + 1) * rows])
                torch.testing.assert_close(item.model_specific_data["image_grid_thw"], grids[i : i + 1])
                assert item.feature.untyped_storage().nbytes() == rows * width * 4
            measurements[f"{label}_protocol_{protocol}_MiB"] = round(len(payload) / 2**20, 4)
    assert cache.hits == count
    if count > 1:
        # Mutating the temporary batch cannot corrupt cached artifacts.
        features.fill_(-1)
        grids.fill_(-1)
        assert artifacts[0].feature[0, 0].item() == 0
        assert artifacts[0].model_specific_data["image_grid_thw"][0, 0].item() == 1
    return {"images": count, "logical_MiB": logical_bytes / 2**20, **measurements}


if __name__ == "__main__":
    torch.set_num_threads(1)
    for count in (1, 8, 32):
        print(json.dumps(check(count)), flush=True)
        gc.collect()
    print("PASS: compact CPU image storage, cold/warm cache, two pickle hops")

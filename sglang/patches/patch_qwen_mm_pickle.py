#!/usr/bin/env python3
"""Detach cached Qwen CPU image slices before pickle IPC/broadcast.

The pinned Qwen artifact processor uses contiguous views of the batch storage.
Python pickle serializes that entire storage for each image, causing quadratic
memory growth. Own only each image's data, including its grid metadata, while
preserving the existing device-resident feature path.
"""

from pathlib import Path


TARGET = Path(
    "/sgl-workspace/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py"
)
OLD = """            feature = features[
                feature_offset : feature_offset + feature_length
            ].contiguous()
            feature_offset += feature_length
"""
NEW = """            feature = features[
                feature_offset : feature_offset + feature_length
            ]
            # CPU pickle serializes the complete backing storage of each view.
            # contiguous() alone keeps the batch storage for these row slices.
            # Give cached CPU artifacts independent storage before transport.
            feature = (
                feature.clone()
                if feature.device.type == "cpu"
                else feature.contiguous()
            )
            image_grid = grid.unsqueeze(0)
            if image_grid.device.type == "cpu":
                image_grid = image_grid.clone()
            feature_offset += feature_length
"""
OLD_GRID = '                    model_specific_data={"image_grid_thw": grid.unsqueeze(0)},'
NEW_GRID = '                    model_specific_data={"image_grid_thw": image_grid},'


def patch(path: Path = TARGET) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(NEW) == 1 and source.count(NEW_GRID) == 1:
        print(f"Qwen CPU image storage fix already present: {path}")
        return
    if source.count(OLD) != 1 or source.count(OLD_GRID) != 1:
        raise RuntimeError(f"Unexpected Qwen artifact implementation in {path}")
    updated = source.replace(OLD, NEW, 1).replace(OLD_GRID, NEW_GRID, 1)
    compile(updated, str(path), "exec")
    path.write_text(updated, encoding="utf-8")
    print(f"Patched Qwen CPU image storage: {path}")


if __name__ == "__main__":
    patch()

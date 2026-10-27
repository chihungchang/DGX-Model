#!/usr/bin/env python3
"""Replace the old LibertAI pack with its pinned BLACKFOLD derivative on one Spark.

Requires an explicit --replace-libertai, stopped serving ranks and a downloaded,
authorized HF overlay directory. Common Engram data and SGLang are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "BlackwellBoy/DeepSeek-V4.1-Flash-1017-BLACKFOLD-EXL3-2.9bpw-TP2"
REVISION = "d795e62c91fab64531cfe6b2b5d0c0d8a95a9295"
OVERLAY_SHA = "f42912922985684e7a0cd8f3f6d73efafec620399e8100f09cfdb0ab67b0e649"
IMAGE = "dgx-tensorfold:deepseek-v41-07aa02585f64-pool-v2-dsml-v1-images-v1-output-ceiling-v1"
OLD_VIEW = Path.home() / "models/tensorfold/deepseek-v41-flash-1017-exl3-2.9bpw"
NEW_VIEW = Path.home() / "models/tensorfold/deepseek-v41-flash-1017-blackfold-exl3-2.9bpw"
OLD_CACHE = Path.home() / ".cache/tensorfold/deepseek-v41-flash-1017"
NEW_CACHE = Path.home() / ".cache/tensorfold/deepseek-v41-flash-1017-blackfold"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sums(path):
    return {Path(line.split()[1]).name: line.split()[0]
            for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, choices=[0, 1], required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--replace-libertai", action="store_true", required=True)
    ap.add_argument("--cache-builder", type=Path, required=True)
    args = ap.parse_args()
    assets = args.assets.resolve()
    meta = json.loads((assets / "GRAFT_META.json").read_text())
    if meta.get("stock_revision") != "afc7aa0aab98ee735c0f54e0a122019af1ac08f1":
        raise RuntimeError("Unexpected BLACKFOLD base revision")
    overlay = assets / meta["overlay"]
    if sha(overlay) != OVERLAY_SHA or meta["overlay_sha256"] != OVERLAY_SHA:
        raise RuntimeError("BLACKFOLD overlay checksum mismatch")
    status = subprocess.run(["docker", "inspect", f"dgx-tensorfold-tp2-r{args.rank}"],
                            capture_output=True, text=True)
    if not status.returncode and json.loads(status.stdout)[0]["State"]["Status"] in ("running", "paused", "restarting"):
        raise RuntimeError("Stop TensorFold before replacing model files")
    NEW_CACHE.mkdir(parents=True, exist_ok=True)
    (NEW_CACHE / "tf-ds-rank").mkdir(exist_ok=True)
    # Only derived rank files of the replaced model are removed; extension caches are left alone.
    code = ("from pathlib import Path\n"
            f"files=list(Path('/old').glob('rank{args.rank}of2-*.bin'))\n"
            "for f in files:\n"
            " with f.open('rb') as stream:stream.seek(-8,2);magic=stream.read(8)\n"
            " if magic!=b'TFDSRK01':raise RuntimeError('unexpected cache format')\n"
            " print('Removing obsolete rank cache',f.name,f.stat().st_size,flush=True);f.unlink()\n")
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "1g",
                    "-v", f"{OLD_CACHE}/tf-ds-rank:/old", "--entrypoint", "python3", IMAGE, "-c", code], check=True)
    shutil.copy2(OLD_CACHE / "dsv41-token-map.json", NEW_CACHE / "dsv41-token-map.json")
    if OLD_VIEW.exists():
        if NEW_VIEW.exists():
            raise RuntimeError("Both old and new model views exist; inspect before replacing either")
        OLD_VIEW.rename(NEW_VIEW)
    model = NEW_VIEW / "model"
    stock, expected = sums(assets / "stock_shards.sha256"), sums(assets / "SHA256SUMS")
    source = OLD_VIEW.with_name(OLD_VIEW.name + ".src") / "model"
    removed = []
    for i, name in enumerate(sorted(stock), 1):
        path = model / name
        digest = sha(path)
        if digest not in (stock[name], expected[name]):
            raise RuntimeError("Unknown source shard: " + name)
        # The original download directory was hardlinked to the view. Drop only that identical
        # old pathname, so atomic shard replacement also releases the original inode's space.
        alias = source / name
        if alias.is_file() and path.samefile(alias):
            alias.unlink()
            removed.append(str(alias))
        if i % 4 == 0 or i == len(stock):
            print(f"rank {args.rank}: verified source {i}/{len(stock)}; free {shutil.disk_usage(model).free / 1024**3:.1f} GiB", flush=True)
    (assets / f"rank{args.rank}-removed-source-links.json").write_text(json.dumps(removed, indent=2))
    if shutil.disk_usage(model).free < 20 * 1024**3:
        raise RuntimeError("Need 20 GiB scratch before the in-place conversion")
    import os
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "20g",
                    "--memory-swap", "20g", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", f"{model}:/model", "-v", f"{assets}:/overlay:ro",
                    "--entrypoint", "python3", IMAGE, "/overlay/apply_overlay.py", "--base", "/model",
                    "--overlay", "/overlay/" + overlay.name, "--inplace", "--sha256sums", "/overlay/SHA256SUMS"], check=True)
    for name in ["GRAFT_META.json", "SHA256SUMS", "stock_shards.sha256"]:
        shutil.copy2(assets / name, model / ("BLACKFOLD_" + name))
    cache = NEW_CACHE / "tf-ds-rank"
    existing = list(cache.glob(f"rank{args.rank}of2-*.bin"))
    if not existing:
        if shutil.disk_usage(cache).free < 115 * 1024**3:
            raise RuntimeError("Need 115 GiB for the replacement rank cache and reserve")
        subprocess.run([args.cache_builder, model, cache, str(args.rank), "2"], check=True)
    files = list(cache.glob(f"rank{args.rank}of2-*.bin"))
    if len(files) != 1:
        raise RuntimeError("Expected exactly one completed native rank cache")
    f = files[0]
    with f.open("rb") as stream:
        stream.seek(-16, 2); footer = stream.read(16)
    if footer[8:] != b"TFDSRK01":
        raise RuntimeError("Incomplete rank cache")
    receipt = {"repo_id": MODEL_ID, "revision": REVISION, "overlay_sha256": OVERLAY_SHA,
               "rank": args.rank, "cache_file": f.name, "cache_size": f.stat().st_size,
               "cache_index_offset": struct.unpack("<Q", footer[:8])[0],
               "stock_source_links_removed": len(removed), "time": time.time()}
    (NEW_CACHE / "blackfold-model.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (assets / f"rank{args.rank}-prepared.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"rank {args.rank}: BLACKFOLD weights and native cache ready; free {shutil.disk_usage(cache).free / 1024**3:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()

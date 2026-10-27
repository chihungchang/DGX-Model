#!/usr/bin/env python3
"""Build the pinned native TP2 engine without a GPU; preserve Python weights/caches."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
ENGINE = "a57deb9d470b27e1e3fab2a9494915a0049415aa"
RECIPE = "30a5bf9cad93b9355574ae65e2b89f7159350f76"
IMAGE = "dgx-tensorfold:deepseek-v41-07aa02585f64-pool-v2-dsml-v1-images-v1-output-ceiling-v1"
MODEL = Path.home() / "models/tensorfold/deepseek-v41-flash-1017-exl3-2.9bpw/model"
RUNTIME = ROOT / "tensorfold/native/runtime"
SOURCES = [
    ("engine-v072", f"https://codeload.github.com/bertholomus/TensorFold/tar.gz/{ENGINE}",
     "5e57eabb88fd4c853355023c295fb13d3db90f0badcf0a523ec55fd287baf52f"),
    ("kit-recipe-v072", f"https://codeload.github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/tar.gz/{RECIPE}",
     "f286f5e71f2912f9f12836b686e1b23362bc6913abde448a98b0ad902d7b76f8"),
    ("zig", "https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz",
     "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8"),
]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(work, name, url, digest):
    dest = work / name
    if dest.is_dir():
        receipt = json.loads((dest / ".source.json").read_text())
        if receipt != {"url": url, "sha256": digest}:
            raise RuntimeError(f"Unrecognized source checkout: {dest}")
        return dest
    print("Downloading " + name, flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(200 * 1024**2)
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Archive checksum mismatch: " + url)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        archive.extractall(work, filter="data")
    (work / prefix).rename(dest)
    (dest / ".source.json").write_text(json.dumps({"url": url, "sha256": digest}, indent=2))
    return dest


def install(engine, recipe, model=MODEL):
    tests = (engine / "dgx-tests.log").read_text()
    build = (engine / "dgx-build.log").read_text()
    if "Build Summary:" not in tests or "steps succeeded;" not in tests or "failed);" in tests:
        raise RuntimeError("Native host tests have not passed")
    if "Build Summary:" not in build or "failed" in build:
        raise RuntimeError("Native build has not passed")
    binaries = {
        "bin/tensorfold-native": engine / "zig-out/native/bin/tensorfold-native",
        "bin/tf-dsv41-lanes": engine / "zig-out/bin/tf-dsv41-lanes",
        "rdma_gather.fatbin": engine / "zig-out/fatbin/dsv41_rdma.fatbin",
    }
    for relative, src in binaries.items():
        dest = RUNTIME / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    template = (model / "chat_template.jinja").read_text()
    template = template.replace("dict(schema.d, **{k: v})", "dict((schema.d.items() | list) + [[k, v]])")
    (RUNTIME / "chat_template.native.jinja").write_text(template)
    subprocess.run(["bash", RUNTIME / "kit/verify_kit.sh", model], check=True)
    files = [*binaries, "chat_template.native.jinja"]
    for line in (RUNTIME / "kit/MANIFEST").read_text().splitlines():
        if line and not line.startswith("#"):
            files.append("kit/" + line.split()[4])
    receipt = {"engine_commit": ENGINE, "recipe_commit": RECIPE, "zig": "0.17.0",
               "image": IMAGE, "files": [{"path": name, "size": (RUNTIME / name).stat().st_size,
                                            "sha256": sha(RUNTIME / name)} for name in files],
               "patches": {p.name: sha(p) for p in sorted((ROOT / "tensorfold/native/patches").glob("*.patch"))}}
    (RUNTIME / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Native runtime ready: {len(files)} verified files, engine {ENGINE}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tensorfold/.state/native-build-a57deb9")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--install-only", action="store_true", help="Install an already tested build from work-dir")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    engine, recipe, zig = [source(work, *item) for item in SOURCES]
    if args.install_only:
        install(engine, recipe, args.model)
        return
    live = subprocess.run(["docker", "ps", "-q", "--filter", "name=dgx-tensorfold-tp2"],
                          text=True, capture_output=True, check=True)
    if live.stdout.strip():
        raise RuntimeError("Stop both TensorFold ranks before compiling the native runtime")
    for patch in sorted((ROOT / "tensorfold/native/patches").glob("*.patch")):
        reverse = subprocess.run(["patch", "--dry-run", "--batch", "-R", "-p1", "-i", str(patch)],
                                 cwd=engine, capture_output=True)
        if reverse.returncode:
            subprocess.run(["patch", "--batch", "--forward", "-p1", "-i", patch], cwd=engine, check=True)
    shutil.copytree(recipe / "kit", RUNTIME / "kit", dirs_exist_ok=True)
    subprocess.run(["python3", RUNTIME / "kit/tools/make_bias_vl.py", args.model,
                    RUNTIME / "kit/vision/gate_bias_vl.safetensors"], check=True)
    script = """set -eu
export PATH=/zig:$PATH
export OMP_NUM_THREADS=4
if [ ! -s /kit/rope-plain-cos.f32 ]; then python3 /kit/tools/make_rope.py /model /kit; fi
if ! zig build test -j6 --summary all > /work/dgx-tests.log 2>&1; then head -80 /work/dgx-tests.log; exit 1; fi
grep 'Build Summary' /work/dgx-tests.log
if ! zig build install native fatbins -j6 -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --summary all > /work/dgx-build.log 2>&1; then head -80 /work/dgx-build.log; exit 1; fi
grep 'Build Summary' /work/dgx-build.log
"""
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--memory", "20g",
                    "--memory-swap", "20g", "--cpuset-cpus", "5-9,15-19",
                    "--user", f"{os.getuid()}:{os.getgid()}",
                    "-e", "ZIG_GLOBAL_CACHE_DIR=/work/.zig-global", "-v", f"{engine}:/work",
                    "-v", f"{zig}:/zig:ro", "-v", f"{RUNTIME}/kit:/kit",
                    "-v", f"{args.model}:/model:ro", "-w", "/work", "--entrypoint", "/bin/bash",
                    IMAGE, "-c", script], check=True)
    install(engine, recipe, args.model)


if __name__ == "__main__":
    main()

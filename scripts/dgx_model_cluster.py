#!/usr/bin/env python3
"""Manage the TensorFold TP2 profile and the existing SGLang TP2 rollback."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
WORKER = "spark2"
TF_PROFILE = "deepseek-v41-flash-1017-exl3-tp2"
SG_PROFILE = "qwen-dflash2-tp2"
TF_IMAGE = "dgx-tensorfold:deepseek-v41-07aa02585f64-pool-v2-dsml-v1-images-v1-output-ceiling-v1"
TF_COMMIT = "07aa02585f64cfb04c60a96de6554c0d4708ed46"
TF_BASE_CONFIG = "sha256:15b8de8baabe68e46afc51e88f3ccad5042013bafe8f244ec082ce0bb06d47e3"
TF_LOCAL_BASE = "nvcr.io/nvidia/pytorch:26.07-py3-verified-arm64"
MODEL_VIEW = Path.home() / "models/tensorfold/deepseek-v41-flash-1017-exl3-2.9bpw"
CACHE = Path.home() / ".cache/tensorfold/deepseek-v41-flash-1017"
STATE = ROOT / "tensorfold/.state"
RECIPE = ROOT / "tensorfold/vendor/deepseek-v41"
TEMPLATE_SHA = "defbdfc2a28ea921d9ad4c6be7244fa82c53f6d552507527a944ad4cbdd24a6d"
API = "http://127.0.0.1:8080"


def run(args, rank=0, *, capture=True, check=True, timeout=3600, data=None):
    args = [str(a) for a in args]
    if rank:
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                WORKER, shlex.join(args)]
    return subprocess.run(args, input=data, text=True, capture_output=capture,
                          check=check, timeout=timeout)


def probe(code, rank=0):
    return json.loads(run(["python3", "-c", code], rank).stdout)


def inspect(name, rank=0):
    result = run(["docker", "inspect", name], rank, check=False, timeout=30)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def names(engine):
    return (["dgx-tensorfold-tp2-r0", "dgx-tensorfold-tp2-r1"]
            if engine == "tensorfold" else ["dgx-sglang-dflash2-tp2"] * 2)


def active(engine):
    item = inspect(names(engine)[0])
    return bool(item and item["State"]["Status"] in ("running", "paused", "restarting", "created"))


def arguments(path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def value(args, flag, default=""):
    return args[args.index(flag) + 1] if flag in args else default


def profile_path(engine, profile):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", profile):
        raise ValueError("Invalid profile name")
    if engine == "sglang" and profile == SG_PROFILE:
        return ROOT / "sglang/profiles/qwen-dflash2-tp2-node0.args"
    path = ROOT / "tensorfold/profiles" / (profile + ".args")
    if engine != "tensorfold" or profile != TF_PROFILE or not path.is_file():
        raise ValueError("Unknown cluster profile: " + engine + "/" + profile)
    return path


def prepare_env(profile=TF_PROFILE):
    for rank in (0, 1):
        path = ROOT / "tensorfold" / f"node{rank}.env"
        path.write_text(
            f"TENSORFOLD_RANK={rank}\nTENSORFOLD_PROFILE={profile}\n"
            f"TENSORFOLD_MODEL_VIEW={MODEL_VIEW}\nTENSORFOLD_CACHE_DIR={CACHE}\n"
        )
        path.chmod(0o600)


def compose(engine, rank, profile=None):
    if engine == "tensorfold":
        directory = ROOT / "tensorfold"
        env = directory / f"node{rank}.env"
        filename = directory / "compose.yaml"
    else:
        directory = ROOT / "sglang"
        env = directory / f"tp2-node{rank}.env"
        filename = directory / "compose.dflash2-tp2.yaml"
    return ["docker", "compose", "--project-directory", directory, "--env-file", env,
            "-f", filename]


def sync_config(profile=TF_PROFILE):
    prepare_env(profile)
    run(["mkdir", "-p", ROOT / "tensorfold", ROOT / "scripts", STATE], 1)
    run(["rsync", "-a", "--exclude=.state", "--exclude=__pycache__",
         str(ROOT / "tensorfold") + "/", f"{WORKER}:{ROOT}/tensorfold/"], capture=False)
    run(["rsync", "-a", ROOT / "dgx-model", f"{WORKER}:{ROOT}/dgx-model"], capture=False)
    run(["rsync", "-a", Path(__file__), f"{WORKER}:{ROOT}/scripts/"], capture=False)
    run(["rsync", "-a", ROOT / "scripts/probe_tensorfold_rdma.py",
         f"{WORKER}:{ROOT}/scripts/"], capture=False)
    run(["rsync", "-a", ROOT / "sglang/profiles/qwen-dflash2-tp2.args",
         f"{WORKER}:{ROOT}/sglang/profiles/"], capture=False)


def image_info(rank):
    result = run(["docker", "image", "inspect", TF_IMAGE], rank, check=False)
    if result.returncode:
        return None
    item = json.loads(result.stdout)[0]
    revision = (item["Config"].get("Labels") or {}).get("org.opencontainers.image.revision")
    if revision != TF_COMMIT:
        raise ValueError(f"Rank {rank}: TensorFold image revision is {revision}, expected {TF_COMMIT}")
    patch = (item["Config"].get("Labels") or {}).get("ai.dgx.tensorfold.patch")
    if patch != "rdma-active-mtu-v1":
        raise ValueError(f"Rank {rank}: TensorFold image lacks the active RDMA MTU fix")
    pool = (item["Config"].get("Labels") or {}).get("ai.dgx.tensorfold.pool-context")
    if pool != "independent-v2":
        raise ValueError(f"Rank {rank}: TensorFold image lacks independent shared context storage")
    dsml = (item["Config"].get("Labels") or {}).get("ai.dgx.tensorfold.dsml-single-tool")
    if dsml != "v1":
        raise ValueError(f"Rank {rank}: TensorFold image lacks the single-call DSML filter")
    image_context = (item["Config"].get("Labels") or {}).get("ai.dgx.tensorfold.image-context")
    if image_context != "v1":
        raise ValueError(f"Rank {rank}: TensorFold image lacks context-based image history admission")
    output_ceiling = (item["Config"].get("Labels") or {}).get("ai.dgx.tensorfold.output-ceiling")
    if output_ceiling != "v1":
        raise ValueError(f"Rank {rank}: TensorFold image lacks remaining-context output capping")
    return item["Id"]


def model_errors(rank):
    manifest = json.loads((ROOT / "tensorfold/model-files.json").read_text())
    expected = {x["path"]: x["size"] for x in manifest["files"]
                if x["path"].endswith(".safetensors")}
    code = (
        "import pathlib,json,hashlib\n"
        f"root=pathlib.Path({str(MODEL_VIEW)!r})\nexpected={expected!r}\n"
        "errors=[]\n"
        "for name,size in expected.items():\n"
        " p=root/name\n"
        " if not p.is_file() or p.stat().st_size!=size:errors.append(name)\n"
        "for name in ['model/config.json','model/tokenizer.json','model/tokenizer_config.json',"
        "'model/model.safetensors.index.json']:\n"
        " if not (root/name).is_file():errors.append(name)\n"
        "p=root/'model/chat_template.jinja'\n"
        f"if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!={TEMPLATE_SHA!r}:"
        "errors.append('model/chat_template.jinja (missing or wrong template)')\n"
        "print(json.dumps(errors))\n"
    )
    return probe(code, rank)


def extension_errors(rank, image_id):
    return probe(
        "import pathlib,json\n"
        f"root=pathlib.Path({str(CACHE)!r})\n"
        "p=root/'extension-build.json'\n"
        "v=json.loads(p.read_text()) if p.exists() else {}\n"
        f"errors=[] if v.get('image_id')=={image_id!r} else ['CUDA extension image mismatch']\n"
        "files=v.get('files',[])\n"
        "if len(files)!=4:errors.append('CUDA extensions have not been compiled')\n"
        "errors.extend(str(root/f) for f in files if not (root/f).is_file() or (root/f).stat().st_size==0)\n"
        "print(json.dumps(errors))", rank)


def check_cache_space(rank):
    info = probe(
        "import pathlib,json,shutil\n"
        f"root=pathlib.Path({str(CACHE)!r})\n"
        "p=root/'tf-ds-rank'\n"
        "used=sum(f.stat().st_blocks*512 for f in p.rglob('*') if f.is_file()) if p.exists() else 0\n"
        "print(json.dumps({'free':shutil.disk_usage(root).free,'rank_cache':used}))", rank)
    need = max(0, 110 * 1024**3 - info["rank_cache"]) + 18 * 1024**3
    if info["free"] < need:
        raise ValueError(f"Rank {rank}: need {need / 1024**3:.1f} GiB free for rank and prompt caches; "
                         f"only {info['free'] / 1024**3:.1f} GiB remains")


def validate(engine, profile, *, require_data=True):
    profile_path(engine, profile)
    if engine == "tensorfold":
        prepare_env(profile)
    image_ids = []
    for rank in (0, 1):
        run([*compose(engine, rank), "config", "--quiet"], rank)
        if not require_data:
            continue
        if engine == "tensorfold":
            errors = model_errors(rank)
            if errors:
                raise ValueError(f"Rank {rank}: missing or invalid model files: {errors[:8]}")
            image_id = image_info(rank)
            if not image_id:
                raise ValueError(f"Rank {rank}: build the TensorFold image first")
            errors = extension_errors(rank, image_id)
            if errors:
                raise ValueError(f"Rank {rank}: {errors}; run the extensions command before switching")
            check_cache_space(rank)
            image_ids.append(image_id)
        else:
            code = (
                "import pathlib,json\n"
                f"p=pathlib.Path({str(ROOT / 'sglang/profiles')!r})/"
                f"'qwen-dflash2-tp2-node{rank}.args'\n"
                "a=[x.strip() for x in p.read_text().splitlines() if x.strip() and not x.startswith('#')]\n"
                "errors=[]\n"
                "for flag in ['--model-path','--speculative-draft-model-path']:\n"
                f" d=pathlib.Path({str(Path.home() / 'models/huggingface')!r})/"
                "a[a.index(flag)+1].removeprefix('/models/')\n"
                " if not (d/'config.json').is_file():errors.append(str(d));continue\n"
                " idx=d/'model.safetensors.index.json'\n"
                " shards=set(json.loads(idx.read_text())['weight_map'].values()) if idx.exists() else {'model.safetensors'}\n"
                " errors.extend(str(d/s) for s in shards if not (d/s).is_file())\n"
                "print(json.dumps(errors))"
            )
            errors = probe(code, rank)
            if errors:
                raise ValueError(f"Rank {rank}: missing SGLang artifacts: {errors[:8]}")
            image = run([*compose(engine, rank), "config", "--images"], rank).stdout.splitlines()[0]
            info = run(["docker", "image", "inspect", image], rank)
            item = json.loads(info.stdout)[0]
            image_ids.append((item["Config"].get("Labels") or {}).get(
                "org.opencontainers.image.revision") or item["Id"])
    if require_data and (not all(image_ids) or image_ids[0] != image_ids[1]):
        raise ValueError("Both nodes must have the same engine image before switching")
    print(f"Validated {engine}/{profile} on both nodes.", flush=True)


def list_row(engine, profile):
    args = arguments(profile_path(engine, profile))
    model = value(args, "--name") if engine == "tensorfold" else value(args, "--served-model-name")
    context = value(args, "--context") if engine == "tensorfold" else value(args, "--context-length")
    running = active(engine)
    if engine == "tensorfold":
        manifest = json.loads((ROOT / "tensorfold/model-files.json").read_text())
        ready = all((MODEL_VIEW / f["path"]).is_file() and
                    (MODEL_VIEW / f["path"]).stat().st_size == f["size"]
                    for f in manifest["files"] if f["path"].endswith(".safetensors"))
        source = "LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
    else:
        source = value(args, "--model-path").removeprefix("/models/")
        ready = (Path.home() / "models/huggingface" / source / "config.json").exists()
    print(f"{engine:<10} {('*' if running else '') + profile:<40} {model:<27} "
          f"{context:<9} {'ready' if ready else 'missing':<14} "
          f"{'running (TP2)' if running else '-':<20} {source}")


def show(engine, profile):
    args = arguments(profile_path(engine, profile))
    print(json.dumps({"engine": engine, "profile": profile, "nodes": ["spark1", WORKER],
                      "image": TF_IMAGE if engine == "tensorfold" else
                      run([*compose(engine, 0), "config", "--images"]).stdout.strip(),
                      "model_view": str(MODEL_VIEW) if engine == "tensorfold" else None,
                      "arguments": args, "api": API + "/v1"}, indent=2))


def stop_named(name, rank, force=False):
    item = inspect(name, rank)
    if not item:
        return
    run(["docker", "update", "--restart=no", name], rank, capture=True)
    if item["State"]["Status"] == "paused":
        run(["docker", "unpause", name], rank)
    if item["State"]["Status"] in ("running", "paused", "restarting", "created"):
        run(["docker", "stop", "--time", "0" if force else "120", name], rank,
            capture=False, timeout=180)


def stop(engine, force=False):
    # Stop the API rank first, then its peer; preserve containers and cache volumes.
    for rank, name in enumerate(names(engine)):
        stop_named(name, rank, force)


def stop_other_models():
    for rank in (0, 1):
        for name in ["dgx-vllm", "dgx-sglang", "dgx-sglang-tp2",
                     "dgx-sglang-dflash2-tp2", "dgx-llamacpp", "dgx-llamacpp-rpc"]:
            stop_named(name, rank)


def memory(rank):
    return probe("import pathlib,json\n"
                 "m={l.split(':')[0]:int(l.split()[1]) for l in "
                 "pathlib.Path('/proc/meminfo').read_text().splitlines()}\n"
                 "print(json.dumps(m['MemAvailable']/1024**2))", rank)


def wait_ready(engine, model, timeout=3600):
    deadline = time.monotonic() + timeout
    last_notice = 0
    while time.monotonic() < deadline:
        for rank, name in enumerate(names(engine)):
            item = inspect(name, rank)
            if not item or item["State"]["Status"] not in ("running", "created"):
                raise RuntimeError(f"{name} on rank {rank} exited during startup")
        try:
            with urllib.request.urlopen(API + "/v1/models", timeout=5) as response:
                ids = [m["id"] for m in json.load(response).get("data", [])]
            if model in ids:
                print("Serving " + model + " at " + API + "/v1", flush=True)
                return
        except (OSError, ValueError):
            pass
        if time.monotonic() - last_notice > 30:
            print(f"Waiting for {engine} model loading and warm-up...", flush=True)
            last_notice = time.monotonic()
        time.sleep(5)
    raise TimeoutError("Engine did not become ready within the startup timeout")


def restore_comfyui():
    path = STATE / "restore.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("comfyui_stopped"):
            run(["docker", "start", "mazi-comfyui"], capture=False)
        path.unlink()


def start_sglang():
    for rank in (1, 0):
        run([*compose("sglang", rank), "up", "-d", "--no-build"], rank, capture=False)
    wait_ready("sglang", value(arguments(profile_path("sglang", SG_PROFILE)), "--served-model-name"))
    for rank, name in enumerate(names("sglang")):
        run(["docker", "update", "--restart=unless-stopped", name], rank)
    restore_comfyui()


def use(engine, profile, recreate=False):
    # All validation is before the first stop; an invalid profile must leave the live server alone.
    validate(engine, profile)
    if active(engine) and not recreate:
        args = arguments(profile_path(engine, profile))
        model = value(args, "--name") if engine == "tensorfold" else value(args, "--served-model-name")
        wait_ready(engine, model, timeout=60)
        return
    if engine == "sglang":
        stop("tensorfold")
        stop_other_models()
        start_sglang()
        return
    sync_config(profile)
    validate(engine, profile)
    # Keep the existing SGLang rollback usable before releasing its memory.
    validate("sglang", SG_PROFILE)
    old_sglang = active("sglang")
    STATE.mkdir(parents=True, exist_ok=True)
    saved = STATE / "restore.json"
    restore = json.loads(saved.read_text()) if saved.exists() else {}
    restore.update(sglang_was_active=old_sglang)
    restore.setdefault("comfyui_stopped", False)
    try:
        stop("tensorfold")
        stop_other_models()
        comfy = inspect("mazi-comfyui")
        if comfy and comfy["State"]["Status"] == "running":
            print("Stopping ComfyUI to provide memory for the DeepSeek ranks.", flush=True)
            run(["docker", "stop", "--time", "120", "mazi-comfyui"], capture=False)
            restore["comfyui_stopped"] = True
        (STATE / "restore.json").write_text(json.dumps(restore, indent=2))
        for rank in (0, 1):
            free = memory(rank)
            if free < 100:
                raise RuntimeError(f"Rank {rank}: only {free:.1f} GiB available; need at least 100 GiB")
            run(["mkdir", "-p", CACHE], rank)
        # Rank 0 creates the rendezvous store. Rank 1 then joins it.
        run([*compose("tensorfold", 0), "up", "-d", "--no-build", "--force-recreate"],
            capture=False)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if re.search(r":29551\s", run(["ss", "-lntH"]).stdout):
                break
            item = inspect(names("tensorfold")[0])
            if not item or item["State"]["Status"] == "exited":
                raise RuntimeError("TensorFold rank 0 failed before rendezvous")
            time.sleep(2)
        else:
            raise RuntimeError("TensorFold rendezvous did not open")
        run([*compose("tensorfold", 1), "up", "-d", "--no-build", "--force-recreate"],
            1, capture=False)
        wait_ready("tensorfold", value(arguments(profile_path(engine, profile)), "--name"))
    except Exception:
        print("TensorFold startup failed; stopping both ranks.", flush=True)
        stop("tensorfold", True)
        if old_sglang:
            print("Restoring the previous SGLang service.", flush=True)
            start_sglang()
        else:
            restore_comfyui()
        raise


def build_image(profile):
    profile_path("tensorfold", profile)
    prepare_env(profile)
    build_args = []
    local_base = run(["docker", "image", "inspect", "--format", "{{.Id}}", TF_LOCAL_BASE],
                     check=False)
    if local_base.returncode == 0 and local_base.stdout.strip() == TF_BASE_CONFIG:
        build_args = ["--build-arg", "BASE_IMAGE=" + TF_LOCAL_BASE]
    run(["docker", "build", "--progress=plain", "-t", TF_IMAGE, *build_args, ROOT / "tensorfold"],
        capture=False)
    sync_config(profile)
    print("Copying the TensorFold image to spark2...", flush=True)
    sender = subprocess.Popen(["docker", "save", TF_IMAGE], stdout=subprocess.PIPE)
    try:
        receiver = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "Compression=no",
                                   "-c", "aes128-gcm@openssh.com", WORKER, "docker load"],
                                  stdin=sender.stdout, check=True)
    finally:
        sender.stdout.close()
    if sender.wait() or receiver.returncode:
        raise RuntimeError("Image transfer failed")
    if image_info(0) != image_info(1):
        raise RuntimeError("Transferred engine image differs")
    print("Both nodes have the same pinned TensorFold image.", flush=True)
    if not active("tensorfold"):
        build_extensions()
    else:
        print("Compile extensions after stopping TensorFold before using this new image.", flush=True)


def build_extensions():
    for rank, name in enumerate(names("tensorfold")):
        item = inspect(name, rank)
        if item and item["State"]["Status"] in ("running", "paused", "restarting", "created"):
            raise RuntimeError("Stop TensorFold on both nodes before compiling its extensions")
    code = ("import pathlib,json,fcntl\n"
            "guard=open('/cache/extension-build.lock','a')\n"
            "fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "for p in pathlib.Path('/cache/torch_extensions').rglob('lock'):p.unlink(missing_ok=True)\n"
            "from tensorfold.cuda import rdma\n"
            "from tensorfold.cuda.exl3 import experts, linear, prefetch\n"
            "files=[]\n"
            "for module in (rdma, linear, experts, prefetch):\n"
            " print('Building', module.__name__, flush=True)\n"
            " files.append(str(pathlib.Path(module._ext().__file__).relative_to('/cache')))\n")
    def build_rank(rank):
        image_id = image_info(rank)
        if not image_id:
            raise RuntimeError(f"Rank {rank}: TensorFold image is missing")
        run(["mkdir", "-p", CACHE], rank)
        # No model weights are loaded in this temporary build container.
        run(["docker", "run", "--rm", "--gpus", "all", "--network=none",
             "--memory=20g", "--memory-swap=20g", "--device=/dev/infiniband",
             "-v", f"{CACHE}:/cache", "-e", "TORCH_EXTENSIONS_DIR=/cache/torch_extensions",
             "-e", "CUDA_CACHE_PATH=/cache/nv", "-e", "MAX_JOBS=4",
             "--entrypoint", "python3", TF_IMAGE, "-c", code +
             "pathlib.Path('/cache/extension-build.json').write_text(json.dumps("
             f"{{'image_id':{image_id!r},'files':files}}))\n"], rank, capture=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(build_rank, rank) for rank in (0, 1)]:
            future.result()


def probe_rdma():
    for rank, name in enumerate(names("tensorfold")):
        item = inspect(name, rank)
        if item and item["State"]["Status"] in ("running", "paused", "restarting", "created"):
            raise RuntimeError("Stop TensorFold on both nodes before probing RDMA")
    sync_config()

    def probe_rank(rank):
        run(["docker", "run", "--rm", "--gpus", "all", "--network=host", "--ipc=host",
             "--memory=8g", "--memory-swap=8g", "--cpuset-cpus=5-9,15-19",
             "--device=/dev/infiniband", "--cap-add=IPC_LOCK", "--ulimit=memlock=-1:-1",
             "-v", f"{CACHE}:/cache",
             "-v", f"{ROOT}/scripts/probe_tensorfold_rdma.py:/probe.py:ro",
             "-e", "TORCH_EXTENSIONS_DIR=/cache/torch_extensions",
             "-e", "CUDA_CACHE_PATH=/cache/nv", "-e", "MAX_JOBS=4",
             "-e", "TF_RDMA_DEVICES=rocep1s0f0,roceP2p1s0f0",
             "-e", "NCCL_IB_GID_INDEX=3", "--entrypoint=python3",
             TF_IMAGE, "/probe.py", str(rank)], rank, capture=False, timeout=180)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(probe_rank, rank) for rank in (0, 1)]:
            future.result()


def pull_models():
    sync_config()
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloads = [pool.submit(run, ["bash", RECIPE / "scripts/download.sh", phase],
                                 capture=False, timeout=86400)
                     for phase in ("download", "engram")]
        for download in downloads:
            download.result()
    run(["env", f"PATH={ROOT / 'tensorfold/transport'}:{os.environ['PATH']}",
         "bash", RECIPE / "scripts/download.sh",
         "verify", "view", "worker"], capture=False, timeout=86400)
    run(["python3", RECIPE / "scripts/patch_template.py", "--worker"], capture=False)
    for rank in (0, 1):
        errors = model_errors(rank)
        if errors:
            raise RuntimeError(f"Rank {rank}: downloaded files are invalid: {errors[:8]}")


def status(engine):
    for rank, name in enumerate(names(engine)):
        item = inspect(name, rank)
        state = item["State"]["Status"] if item else "absent"
        profile = (TF_PROFILE if engine == "tensorfold" else SG_PROFILE) if item else "-"
        print(f"engine={engine} rank={rank} container={name} profile={profile} state={state}")


def smoke(engine):
    model = "deepseek-v4.1-flash-1017" if engine == "tensorfold" else "qwen3.8-27b"
    body = {"model": model, "messages": [{"role": "user", "content": "Reply exactly with: DGX MODEL OK"}],
            "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(API + "/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    content = result["choices"][0]["message"].get("content") or ""
    if "DGX MODEL OK" not in content:
        raise RuntimeError("Unexpected smoke reply: " + content)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("engine", choices=["tensorfold", "sglang"])
    parser.add_argument("profile", nargs="?")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    profile = args.profile or (TF_PROFILE if args.engine == "tensorfold" else SG_PROFILE)
    if args.command == "supports":
        try:
            profile_path(args.engine, profile)
        except ValueError:
            return 1
    elif args.command == "active":
        return 0 if active(args.engine) else 1
    elif args.command == "list":
        list_row(args.engine, profile)
    elif args.command == "show":
        show(args.engine, profile)
    elif args.command == "validate":
        validate(args.engine, profile)
    elif args.command == "validate-config":
        validate(args.engine, profile, require_data=False)
    elif args.command == "compose-config":
        if args.engine == "tensorfold":
            prepare_env(profile)
        run([*compose(args.engine, 0), "config"], capture=False)
    elif args.command == "use":
        use(args.engine, profile, args.recreate)
    elif args.command == "stop":
        stop(args.engine, args.force)
    elif args.command == "status":
        status(args.engine)
    elif args.command == "smoke":
        smoke(args.engine)
    elif args.command == "image-build":
        build_image(profile)
    elif args.command == "extensions":
        build_extensions()
    elif args.command == "rdma-probe":
        probe_rdma()
    elif args.command == "model-pull":
        pull_models()
    else:
        parser.error("Unknown command: " + args.command)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print("Cluster operation failed: " + str(exc), file=sys.stderr)
        sys.exit(1)

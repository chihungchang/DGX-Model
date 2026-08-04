#!/usr/bin/env python3
"""Build deterministic dgx-model profiles from Hugging Face metadata/cards."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import urllib.parse
from pathlib import Path
from typing import Any


MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+@=-]*$")
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)

VLLM_VALUE_FLAGS = {
    "--attention-backend",
    "--config-format",
    "--dtype",
    "--gpu-memory-utilization",
    "--hf-overrides",
    "--kv-cache-dtype",
    "--limit-mm-per-prompt",
    "--load-format",
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
    "--mm-processor-kwargs",
    "--moe-backend",
    "--quantization",
    "--reasoning-parser",
    "--rope-scaling",
    "--speculative-config",
    "--tokenizer-mode",
    "--tool-call-parser",
}
VLLM_BOOL_FLAGS = {
    "--async-scheduling",
    "--enable-auto-tool-choice",
    "--enable-chunked-prefill",
    "--enable-prefix-caching",
    "--enforce-eager",
    "--language-model-only",
    "--trust-remote-code",
}
VLLM_JSON_FLAGS = {
    "--hf-overrides",
    "--limit-mm-per-prompt",
    "--mm-processor-kwargs",
    "--rope-scaling",
    "--speculative-config",
}
VLLM_INTEGER_FLAGS = {
    "--max-model-len",
    "--max-num-batched-tokens",
    "--max-num-seqs",
}

LLAMA_VALUE_FLAGS = {
    "--batch-size",
    "--cache-type-k",
    "--cache-type-v",
    "--chat-template",
    "--ctx-size",
    "--fit",
    "--flash-attn",
    "--n-gpu-layers",
    "--parallel",
    "--reasoning-format",
    "--spec-draft-n-max",
    "--spec-type",
    "--ubatch-size",
}
LLAMA_BOOL_FLAGS = {"--cont-batching", "--jinja", "--metrics", "--no-mmap"}
LLAMA_SHORT_FLAGS = {
    "-b": "--batch-size",
    "-c": "--ctx-size",
    "-fa": "--flash-attn",
    "-ngl": "--n-gpu-layers",
    "-np": "--parallel",
    "-ub": "--ubatch-size",
}


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def load_json(path: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Unable to read Hugging Face metadata: {exc}")
    if not isinstance(value, dict):
        fail("Hugging Face model metadata is not a JSON object.")
    return value


def safe_profile_name(model_id: str) -> str:
    name = re.sub(r"[^a-z0-9_-]+", "-", model_id.rsplit("/", 1)[1].lower())
    name = re.sub(r"[-_]{2,}", "-", name).strip("-_")[:80].rstrip("-_")
    if not PROFILE_RE.fullmatch(name):
        fail(f"Unable to derive a safe profile name from {model_id}")
    return name


def safe_alias(model_id: str) -> str:
    return safe_profile_name(model_id)


def fenced_blocks(card: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    for match in re.finditer(r"```[^\n]*\n(.*?)```", card, re.DOTALL):
        block = re.sub(r"\\\s*\n", " ", match.group(1))
        blocks.append((match.start(), block))
    return blocks


def nearest_heading(card: str, position: int) -> str:
    headings = list(re.finditer(r"^#{1,6}\s+(.+?)\s*$", card[:position], re.MULTILINE))
    return headings[-1].group(1).lower() if headings else ""


def candidate_commands(card: str, marker: re.Pattern[str]) -> list[tuple[int, str, list[str]]]:
    candidates: list[tuple[int, str, list[str]]] = []
    for position, block in fenced_blocks(card):
        for line in block.splitlines():
            line = re.sub(r"^\s*(?:\$|>)\s*", "", line).strip()
            if not marker.search(line):
                continue
            try:
                tokens = shlex.split(line, comments=True, posix=True)
            except ValueError:
                continue
            if tokens:
                heading = nearest_heading(card, position)
                score = 0
                if re.search(r"tool|function", heading):
                    score += 10
                if re.search(r"text.only", heading):
                    score += 20
                if re.search(r"mtp|speculat|draft", heading):
                    score += 30
                if re.search(r"standard|default|recommended", heading):
                    score -= 10
                candidates.append((score, line, tokens))
    return sorted(candidates, key=lambda item: item[0])


def normalize_json_flag(flag: str, value: str) -> str | None:
    if flag not in VLLM_JSON_FLAGS:
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if flag == "--speculative-config" and isinstance(parsed, dict) and parsed.get("model"):
        # An external draft repository would need its own immutable revision and
        # download plan. Do not leave a generated profile with an implicit,
        # unpinned network dependency.
        return None
    return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)


def extract_vllm_card_args(card: str) -> tuple[list[str], str]:
    marker = re.compile(
        r"(?:^|\s)(?:vllm\s+serve|(?:python|python3)\s+-m\s+vllm[.]entrypoints[.]openai[.]api_server)(?:\s|$)"
    )
    candidates = candidate_commands(card, marker)
    if not candidates:
        return [], ""
    _, command, tokens = candidates[0]
    try:
        serve_index = next(i for i, token in enumerate(tokens[:-1]) if token == "vllm" and tokens[i + 1] == "serve")
    except StopIteration:
        try:
            module_index = tokens.index("vllm.entrypoints.openai.api_server")
        except ValueError:
            return [], ""
        i = module_index + 1
    else:
        i = serve_index + 3  # Skip "vllm serve MODEL".

    result: list[str] = []
    seen: set[str] = set()
    while i < len(tokens):
        token = tokens[i]
        if "=" in token and token.startswith("--"):
            flag, value = token.split("=", 1)
            step = 1
        else:
            flag = token
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
            step = 2

        if flag in VLLM_BOOL_FLAGS:
            if flag not in seen:
                result.append(flag)
                seen.add(flag)
            i += 1
            continue
        if flag not in VLLM_VALUE_FLAGS or not value or value.startswith("--"):
            i += 1
            continue
        if flag in VLLM_INTEGER_FLAGS and not re.fullmatch(r"[1-9][0-9]*", value):
            i += step
            continue
        if flag == "--gpu-memory-utilization":
            try:
                utilization = float(value)
            except ValueError:
                i += step
                continue
            if not 0 < utilization < 1:
                i += step
                continue
        normalized = normalize_json_flag(flag, value)
        if normalized is not None and flag not in seen:
            result.extend((flag, normalized))
            seen.add(flag)
        i += step
    return result, command


def context_from_card(card: str) -> int | None:
    patterns = (
        r"(?:default|native|maximum|max(?:imum)?)[^\n]{0,50}context(?:\s+window|\s+length)?[^\d\n]{0,20}([0-9][0-9,]{3,})",
        r"context(?:\s+window|\s+length)[^\d\n]{0,20}([0-9][0-9,]{3,})",
    )
    for pattern in patterns:
        match = re.search(pattern, card, re.IGNORECASE)
        if match:
            value = int(match.group(1).replace(",", ""))
            if 512 <= value <= 4_194_304:
                return value
    return None


def vllm_plan(
    model_id: str,
    revision: str,
    card: str,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tags = {tag.lower() for tag in ((info or {}).get("tags") or []) if isinstance(tag, str)}
    unsupported_formats = tags.intersection({"gguf", "mlx", "onnx", "coreml", "tflite"})
    if unsupported_formats:
        fail(
            f"{model_id} is tagged as {', '.join(sorted(unsupported_formats))}, "
            "not as a vLLM checkpoint. Use the matching engine or repository."
        )
    pipeline_tag = (info or {}).get("pipeline_tag")
    supported_tasks = {"text-generation", "image-text-to-text", "text2text-generation"}
    if isinstance(pipeline_tag, str) and pipeline_tag not in supported_tasks:
        fail(
            f"{model_id} uses the {pipeline_tag} task, which this vLLM deployment "
            "manager does not support."
        )
    card_args, card_command = extract_vllm_card_args(card)
    flags = set(card_args[::2]) | {item for item in card_args if item.startswith("--")}
    if "--max-model-len" not in flags:
        context = context_from_card(card)
        if context:
            card_args.extend(("--max-model-len", str(context)))
    if "--gpu-memory-utilization" not in flags:
        card_args.extend(("--gpu-memory-utilization", "0.8"))

    args = [
        f"/models/{model_id}",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
        "--served-model-name",
        safe_alias(model_id),
        "--tensor-parallel-size",
        "1",
        *card_args,
    ]
    return {
        "engine": "vllm",
        "model_id": model_id,
        "revision": revision,
        "profile": safe_profile_name(model_id),
        "card_command": card_command,
        "artifacts": [
            {
                "kind": "hf",
                "filename": "",
                "container_path": f"/models/{model_id}",
            }
        ],
        "args": args,
    }


def sibling_filenames(info: dict[str, Any]) -> list[str]:
    siblings = info.get("siblings") or []
    result: list[str] = []
    for sibling in siblings:
        filename = sibling.get("rfilename") if isinstance(sibling, dict) else None
        if (
            isinstance(filename, str)
            and SAFE_FILE_RE.fullmatch(filename)
            and all(part not in {"", ".", ".."} for part in filename.split("/"))
        ):
            result.append(filename)
    return result


def quantization_hint(card: str, model_id: str) -> str:
    patterns = (
        rf"(?:-hf|--hf-repo)\s+{re.escape(model_id)}(?::|\s+)([A-Za-z0-9_.-]+)",
        r"recommend(?:ed)?[^\n]{0,80}\b((?:IQ|Q)[2-8]_[A-Za-z0-9_]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, card, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def gguf_score(filename: str, hint: str) -> tuple[int, int, str]:
    upper = filename.upper()
    basename = filename.rsplit("/", 1)[-1].lower()
    if re.match(r"^(?:MMPROJ|DRAFT|MTP)[-_]", basename):
        return (10_000, len(filename), filename)
    preferences = [hint, "Q4_K_M", "Q5_K_M", "Q4_K_S", "IQ4_XS", "Q6_K", "Q8_0"]
    for index, quant in enumerate(item for item in preferences if item):
        if re.search(rf"(?:^|[-_.]){re.escape(quant)}(?:[-_.]|$)", upper):
            return (index, len(filename), filename)
    return (100, len(filename), filename)


def shard_group(filename: str, files: list[str]) -> list[str]:
    match = SHARD_RE.match(filename)
    if not match:
        return [filename]
    prefix, _, count_text, suffix = match.groups()
    count = int(count_text)
    members = [
        item
        for item in files
        if (other := SHARD_RE.match(item))
        and other.group(1) == prefix
        and other.group(3) == count_text
        and other.group(4).lower() == suffix.lower()
    ]
    members.sort(key=lambda item: int(SHARD_RE.match(item).group(2)))  # type: ignore[union-attr]
    if len(members) != count:
        fail(f"GGUF shard set is incomplete for {filename}: found {len(members)} of {count}")
    return members


def extract_llama_card_args(card: str) -> tuple[list[str], str]:
    marker = re.compile(r"(?:^|\s)(?:llama-server|llama-cli)(?:\s|$)")
    candidates = candidate_commands(card, marker)
    if not candidates:
        return [], ""
    _, command, tokens = candidates[0]
    try:
        start = next(i for i, token in enumerate(tokens) if token in {"llama-server", "llama-cli"}) + 1
    except StopIteration:
        return [], ""
    result: list[str] = []
    seen: set[str] = set()
    i = start
    while i < len(tokens):
        token = tokens[i]
        if "=" in token and token.startswith("--"):
            flag, value = token.split("=", 1)
            step = 1
        else:
            flag = LLAMA_SHORT_FLAGS.get(token, token)
            value = tokens[i + 1] if i + 1 < len(tokens) else ""
            step = 2
        if flag in LLAMA_BOOL_FLAGS:
            if flag not in seen:
                result.append(flag)
                seen.add(flag)
            i += 1
            continue
        if flag not in LLAMA_VALUE_FLAGS or not value or value.startswith("-"):
            i += 1
            continue
        if flag in {"--ctx-size", "--parallel", "--batch-size", "--ubatch-size", "--spec-draft-n-max"}:
            if not re.fullmatch(r"[1-9][0-9]*", value):
                i += step
                continue
        if flag not in seen:
            result.extend((flag, value))
            seen.add(flag)
        i += step
    return result, command


def choose_mmproj(files: list[str]) -> str | None:
    candidates = [item for item in files if item.lower().endswith(".gguf") and "mmproj" in item.lower()]
    if not candidates:
        return None
    def score(item: str) -> tuple[int, int, str]:
        upper = item.upper()
        if "BF16" in upper or "F16" in upper:
            return (0, len(item), item)
        if "F32" in upper:
            return (2, len(item), item)
        return (1, len(item), item)
    return min(candidates, key=score)


def llama_plan(model_id: str, revision: str, card: str, info: dict[str, Any], requested: str) -> dict[str, Any]:
    files = sibling_filenames(info)
    ggufs = [item for item in files if item.lower().endswith(".gguf")]
    main_candidates = [
        item for item in ggufs
        if "mmproj" not in item.lower()
        and not re.match(r"^(?:draft|mtp)[-_]", item.rsplit("/", 1)[-1], re.IGNORECASE)
    ]
    if requested:
        if requested not in ggufs:
            fail(f"Requested GGUF is not present in {model_id}: {requested}")
        if "mmproj" in requested.lower():
            fail(f"Requested GGUF is a multimodal projector, not a main model: {requested}")
        selected = requested
    else:
        if not main_candidates:
            fail(f"No deployable GGUF file was found in {model_id}")
        hint = quantization_hint(card, model_id)
        selected = min(main_candidates, key=lambda item: gguf_score(item, hint))

    model_files = shard_group(selected, ggufs)
    mmproj = choose_mmproj(files)
    mmproj_files = shard_group(mmproj, ggufs) if mmproj else []
    artifacts: list[dict[str, str]] = []
    for index, filename in enumerate(model_files):
        artifacts.append(
            {
                "kind": "model" if index == 0 else "shard",
                "filename": filename,
                "container_path": f"/models/{model_id}/{filename}",
            }
        )
    for index, filename in enumerate(mmproj_files):
        artifacts.append(
            {
                "kind": "mmproj" if index == 0 else "mmproj-shard",
                "filename": filename,
                "container_path": f"/models/{model_id}/{filename}",
            }
        )

    card_args, card_command = extract_llama_card_args(card)
    card_flags = {item for item in card_args if item.startswith("--")}
    if "--ctx-size" not in card_flags:
        context = context_from_card(card)
        card_args.extend(("--ctx-size", str(context or 32768)))
    defaults: list[str] = []
    default_pairs = (
        ("--parallel", "1"),
        ("--cache-type-k", "f16"),
        ("--cache-type-v", "f16"),
        ("--n-gpu-layers", "all"),
        ("--fit", "on"),
        ("--flash-attn", "on"),
        ("--batch-size", "2048"),
        ("--ubatch-size", "512"),
    )
    for flag, value in default_pairs:
        if flag not in card_flags:
            defaults.extend((flag, value))
    for flag in ("--no-mmap", "--cont-batching", "--jinja", "--metrics"):
        if flag not in card_flags:
            defaults.append(flag)

    args = [
        "--model",
        artifacts[0]["container_path"],
    ]
    mmproj_artifact = next((item for item in artifacts if item["kind"] == "mmproj"), None)
    if mmproj_artifact:
        args.extend(("--mmproj", mmproj_artifact["container_path"]))
    args.extend(("--alias", safe_alias(model_id), "--host", "0.0.0.0", "--port", "8080"))
    args.extend(card_args)
    args.extend(defaults)
    return {
        "engine": "llamacpp",
        "model_id": model_id,
        "revision": revision,
        "profile": safe_profile_name(model_id),
        "card_command": card_command,
        "artifacts": artifacts,
        "args": args,
    }


def make_plan(args: argparse.Namespace) -> None:
    if not MODEL_ID_RE.fullmatch(args.model_id):
        fail(f"Unsafe Hugging Face model ID: {args.model_id}")
    info = load_json(args.info)
    if info.get("id") != args.model_id:
        fail(f"Hugging Face returned metadata for {info.get('id')!r}, expected {args.model_id!r}")
    revision = info.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        fail(f"Hugging Face did not return an immutable revision for {args.model_id}")
    try:
        card = Path(args.card).read_text(encoding="utf-8")
    except OSError:
        card = ""
    if args.engine == "vllm":
        plan = vllm_plan(args.model_id, revision, card, info)
    else:
        plan = llama_plan(args.model_id, revision, card, info, args.gguf)
    json.dump(plan, sys.stdout, ensure_ascii=False, indent=2)
    print()


def resolve_url(model_id: str, revision: str, filename: str) -> str:
    encoded = urllib.parse.quote(filename, safe="/")
    return f"https://huggingface.co/{model_id}/resolve/{revision}/{encoded}?download=true"


def render_profile(args: argparse.Namespace) -> None:
    plan = load_json(args.plan)
    try:
        checksums = json.loads(args.checksums_json)
    except json.JSONDecodeError as exc:
        fail(f"Invalid checksum JSON: {exc}")
    if not isinstance(checksums, dict):
        fail("Checksum data must be a JSON object.")
    model_id = plan.get("model_id")
    revision = plan.get("revision")
    engine = plan.get("engine")
    artifacts = plan.get("artifacts")
    profile_args = plan.get("args")
    if not isinstance(model_id, str) or not MODEL_ID_RE.fullmatch(model_id):
        fail("Plan contains an unsafe model ID.")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        fail("Plan contains an invalid revision.")
    if engine not in {"vllm", "llamacpp"} or not isinstance(artifacts, list) or not isinstance(profile_args, list):
        fail("Plan has an invalid structure.")

    lines = [
        "# Generated by dgx-model from Hugging Face repository metadata and model card.",
        f"# hf-model: {model_id}",
        f"# hf-revision-id: {revision}",
        f"# Pinned revision: {revision}",
    ]
    card_command = plan.get("card_command")
    if isinstance(card_command, str) and card_command:
        lines.append(f"# Model card command: {card_command}")
    else:
        lines.append("# Model card command: none found; conservative DGX Spark defaults were used.")

    if engine == "vllm":
        lines.append(f"# hf-revision: /models/{model_id} {revision}")
    else:
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                fail("Plan contains an invalid artifact.")
            kind = artifact.get("kind")
            filename = artifact.get("filename")
            container_path = artifact.get("container_path")
            checksum = checksums.get(filename) if isinstance(filename, str) else None
            if (
                not isinstance(kind, str)
                or not isinstance(filename, str)
                or not isinstance(container_path, str)
                or not isinstance(checksum, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum)
            ):
                fail(f"Missing or invalid checksum for {filename!r}")
            url = resolve_url(model_id, revision, filename)
            if kind == "model":
                lines.extend((f"# model-url: {url}", f"# model-sha256: {checksum.lower()}"))
            elif kind == "mmproj":
                lines.extend((f"# mmproj-url: {url}", f"# mmproj-sha256: {checksum.lower()}"))
            else:
                lines.append(f"# artifact: {kind} {container_path} {url} {checksum.lower()}")

    for item in profile_args:
        if not isinstance(item, str) or "\n" in item or "\r" in item or not item:
            fail("Plan contains an invalid profile argument.")
        lines.append(item)
    print("\n".join(lines))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--engine", choices=("vllm", "llamacpp"), required=True)
    plan.add_argument("--model-id", required=True)
    plan.add_argument("--info", required=True)
    plan.add_argument("--card", required=True)
    plan.add_argument("--gguf", default="")
    plan.set_defaults(func=make_plan)
    render = subparsers.add_parser("render")
    render.add_argument("--plan", required=True)
    render.add_argument("--checksums-json", default="{}")
    render.set_defaults(func=render_profile)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

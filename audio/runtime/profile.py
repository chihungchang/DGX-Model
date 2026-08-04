#!/usr/bin/env python3
"""Strict parser and launcher for DGX audio args-only profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final


ASR_MODEL: Final = "Qwen/Qwen3-ASR-1.7B"
TTS_MODEL: Final = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BOOLEAN_FLAGS: Final = {
    "asr": {"--enforce-eager"},
    "tts": {
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--omni",
    },
}
ALLOWED_FLAGS: Final = {
    "asr": {
        "--asr-model-path",
        "--host",
        "--port",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-inference-batch-size",
        "--max-new-tokens",
        "--enforce-eager",
        "--unfixed-chunk-num",
        "--unfixed-token-num",
        "--chunk-size-sec",
        "--max-sessions",
        "--session-ttl-sec",
        "--max-chunk-bytes",
        "--log-level",
    },
    "tts": {
        "--model",
        "--host",
        "--port",
        "--task-type",
        "--deploy-config",
        "--stage-overrides",
        "--trust-remote-code",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
        "--omni",
    },
}
REQUIRED_FLAGS: Final = {
    "asr": ALLOWED_FLAGS["asr"],
    "tts": ALLOWED_FLAGS["tts"],
}
ALLOWED_META: Final = {"api_name", "description", "llm_api_base"}


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    path: Path
    meta: dict[str, str]
    options: dict[str, list[tuple[str, str | None]]]

    def option(self, service: str, flag: str) -> str | None:
        for current_flag, value in self.options[service]:
            if current_flag == flag:
                return value
        raise ProfileError(f"missing {service} option {flag}")

    def flattened_args(self, service: str) -> list[str]:
        result: list[str] = []
        for flag, value in self.options[service]:
            result.append(flag)
            if value is not None:
                result.append(value)
        return result


def _parse(path: Path) -> Profile:
    if not path.is_file():
        raise ProfileError(f"profile does not exist: {path}")

    meta: dict[str, str] = {}
    options: dict[str, list[tuple[str, str | None]]] = {"asr": [], "tts": []}
    seen_flags: dict[str, set[str]] = {"asr": set(), "tts": set()}

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        scope = parts[0]
        if scope == "meta":
            if len(parts) != 3:
                raise ProfileError(f"{path}:{line_number}: meta requires a key and value")
            key, value = parts[1], parts[2]
            if key not in ALLOWED_META:
                raise ProfileError(f"{path}:{line_number}: unsupported meta key {key!r}")
            if key in meta:
                raise ProfileError(f"{path}:{line_number}: duplicate meta key {key!r}")
            meta[key] = value
            continue

        if scope not in options:
            raise ProfileError(f"{path}:{line_number}: unsupported scope {scope!r}")
        if len(parts) < 2:
            raise ProfileError(f"{path}:{line_number}: {scope} requires a flag")
        flag = parts[1]
        value = parts[2] if len(parts) == 3 else None
        if flag not in ALLOWED_FLAGS[scope]:
            raise ProfileError(f"{path}:{line_number}: unsupported {scope} flag {flag!r}")
        if flag in seen_flags[scope]:
            raise ProfileError(f"{path}:{line_number}: duplicate {scope} flag {flag!r}")
        if flag in BOOLEAN_FLAGS[scope] and value is not None:
            raise ProfileError(f"{path}:{line_number}: boolean flag {flag!r} cannot have a value")
        if flag not in BOOLEAN_FLAGS[scope] and value is None:
            raise ProfileError(f"{path}:{line_number}: flag {flag!r} requires one literal value")
        if value and ("${" in value or "$(" in value or "`" in value):
            raise ProfileError(f"{path}:{line_number}: shell expansion is not allowed")
        seen_flags[scope].add(flag)
        options[scope].append((flag, value))

    profile = Profile(path=path, meta=meta, options=options)
    _validate(profile)
    return profile


def _required_float(profile: Profile, service: str, flag: str) -> float:
    value = profile.option(service, flag)
    assert value is not None
    try:
        return float(value)
    except ValueError as error:
        raise ProfileError(f"{service} {flag} must be numeric") from error


def _required_int(profile: Profile, service: str, flag: str) -> int:
    value = profile.option(service, flag)
    assert value is not None
    try:
        return int(value)
    except ValueError as error:
        raise ProfileError(f"{service} {flag} must be an integer") from error


def _validate(profile: Profile) -> None:
    missing_meta = {"api_name", "llm_api_base"} - profile.meta.keys()
    if missing_meta:
        raise ProfileError(f"missing meta keys: {', '.join(sorted(missing_meta))}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", profile.meta["api_name"]):
        raise ProfileError("meta api_name must be a lowercase service identifier")

    if profile.meta["llm_api_base"] != "http://127.0.0.1:8080/v1":
        raise ProfileError("meta llm_api_base must target the co-resident DGX LLM at http://127.0.0.1:8080/v1")

    for service in ("asr", "tts"):
        present = {flag for flag, _ in profile.options[service]}
        missing = REQUIRED_FLAGS[service] - present
        if missing:
            raise ProfileError(f"missing {service} flags: {', '.join(sorted(missing))}")

    if profile.option("asr", "--asr-model-path") != ASR_MODEL:
        raise ProfileError(f"ASR model is fixed to {ASR_MODEL}; fallback models are not allowed")
    if profile.option("tts", "--model") != TTS_MODEL:
        raise ProfileError(f"TTS model is fixed to {TTS_MODEL}; fallback models are not allowed")
    if profile.option("tts", "--task-type") != "CustomVoice":
        raise ProfileError("TTS task type must remain CustomVoice")
    if profile.option("tts", "--deploy-config") != "/app/vllm-omni/vllm_omni/deploy/qwen3_tts.yaml":
        raise ProfileError("TTS must use the pinned qwen3_tts deploy config")

    ports = []
    for service in ("asr", "tts"):
        if profile.option(service, "--host") != "0.0.0.0":
            raise ProfileError(f"{service} must bind 0.0.0.0 for DGX LAN serving")
        port = _required_int(profile, service, "--port")
        if not 1024 <= port <= 65535:
            raise ProfileError(f"{service} port is outside 1024-65535")
        ports.append(port)
    if ports[0] == ports[1]:
        raise ProfileError("ASR and TTS ports must be different")

    asr_gpu = _required_float(profile, "asr", "--gpu-memory-utilization")
    if not 0.01 <= asr_gpu <= 0.15:
        raise ProfileError("ASR gpu_memory_utilization must stay between 0.01 and 0.15")
    if _required_int(profile, "asr", "--max-model-len") != 8192:
        raise ProfileError("ASR max model length is fixed to 8192 for the shared DGX deployment")
    if not 1 <= _required_int(profile, "asr", "--max-inference-batch-size") <= 8:
        raise ProfileError("ASR max inference batch size must stay between 1 and 8")
    if not 1 <= _required_int(profile, "asr", "--max-new-tokens") <= 64:
        raise ProfileError("ASR streaming max_new_tokens must stay between 1 and 64")
    if not 1 <= _required_int(profile, "asr", "--max-sessions") <= 8:
        raise ProfileError("ASR max sessions must stay between 1 and 8")
    if not 1 <= _required_float(profile, "asr", "--session-ttl-sec") <= 600:
        raise ProfileError("ASR session TTL must stay between 1 and 600 seconds")
    if not 0 <= _required_int(profile, "asr", "--unfixed-chunk-num") <= 10:
        raise ProfileError("ASR unfixed chunk count must stay between 0 and 10")
    if not 0 <= _required_int(profile, "asr", "--unfixed-token-num") <= 32:
        raise ProfileError("ASR unfixed token count must stay between 0 and 32")
    decode_window_seconds = _required_float(profile, "asr", "--chunk-size-sec")
    if decode_window_seconds not in {0.5, 1.0}:
        raise ProfileError(
            "ASR decode window must be one of the approved A/B values: 0.5 or 1.0 seconds"
        )
    if profile.option("asr", "--log-level") not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ProfileError("ASR log level is invalid")
    max_chunk_bytes = _required_int(profile, "asr", "--max-chunk-bytes")
    if max_chunk_bytes < 4 or max_chunk_bytes > 262144 or max_chunk_bytes % 4:
        raise ProfileError("ASR max chunk bytes must be a multiple of four and at most 262144")

    stage_value = profile.option("tts", "--stage-overrides")
    assert stage_value is not None
    try:
        stages = json.loads(stage_value)
    except json.JSONDecodeError as error:
        raise ProfileError("TTS stage overrides must be valid compact JSON") from error
    if not isinstance(stages, dict) or set(stages) != {"0", "1"}:
        raise ProfileError("TTS stage overrides must configure exactly stages 0 and 1")
    tts_gpu = 0.0
    allowed_stage_keys = {
        "0": {"gpu_memory_utilization", "max_num_seqs", "enable_prefix_caching"},
        "1": {"gpu_memory_utilization", "max_num_seqs"},
    }
    for stage_id in ("0", "1"):
        stage = stages[stage_id]
        if not isinstance(stage, dict) or "gpu_memory_utilization" not in stage:
            raise ProfileError(f"TTS stage {stage_id} must set gpu_memory_utilization")
        unsupported_keys = set(stage) - allowed_stage_keys[stage_id]
        if unsupported_keys:
            raise ProfileError(f"TTS stage {stage_id} has unsupported overrides: {', '.join(sorted(unsupported_keys))}")
        try:
            utilization = float(stage["gpu_memory_utilization"])
            max_num_seqs = int(stage.get("max_num_seqs", 99))
        except (TypeError, ValueError) as error:
            raise ProfileError(f"TTS stage {stage_id} memory and sequence limits must be numeric") from error
        if not 0.01 <= utilization <= 0.15:
            raise ProfileError(f"TTS stage {stage_id} gpu_memory_utilization must stay between 0.01 and 0.15")
        if not 1 <= max_num_seqs <= 4:
            raise ProfileError(f"TTS stage {stage_id} max_num_seqs cannot exceed 4")
        tts_gpu += utilization
    if stages["0"].get("enable_prefix_caching") is not False:
        raise ProfileError("TTS stage 0 prefix caching must remain disabled in the shared-memory profile")
    if asr_gpu + tts_gpu > 0.30 + 1e-9:
        raise ProfileError("combined audio gpu_memory_utilization cannot exceed 0.30")


def load_profile(path: str | Path) -> Profile:
    return _parse(Path(path).expanduser().resolve())


def _summary(profile: Profile) -> dict[str, object]:
    stage_value = profile.option("tts", "--stage-overrides")
    assert stage_value is not None
    stages = json.loads(stage_value)
    return {
        "apiName": profile.meta["api_name"],
        "llmApiBase": profile.meta["llm_api_base"],
        "asr": {
            "model": profile.option("asr", "--asr-model-path"),
            "port": int(profile.option("asr", "--port") or "0"),
            "gpuMemoryUtilization": float(profile.option("asr", "--gpu-memory-utilization") or "0"),
            "maxModelLen": int(profile.option("asr", "--max-model-len") or "0"),
            "decodeWindowSeconds": float(profile.option("asr", "--chunk-size-sec") or "0"),
        },
        "tts": {
            "model": profile.option("tts", "--model"),
            "port": int(profile.option("tts", "--port") or "0"),
            "stageOverrides": stages,
        },
        "fallback": False,
    }


def _deployment_fingerprint(profile: Profile) -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    artifacts = (
        profile.path,
        root / "audio-model",
        root / "compose.yaml",
        root / "runtime" / "profile.py",
        root / "asr" / "Dockerfile",
        root / "asr" / "server.py",
        root / "tts" / "Dockerfile",
        root / "scripts" / "smoke.py",
    )
    for artifact in artifacts:
        resolved = artifact.resolve()
        if not resolved.is_file():
            raise ProfileError(f"deployment artifact does not exist: {resolved}")
        digest.update(str(resolved.relative_to(root) if resolved.is_relative_to(root) else resolved).encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")

    for container in ("mazi-qwen3-asr", "mazi-qwen3-tts"):
        inspected = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Id}}|{{.Image}}|{{.State.StartedAt}}",
                container,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if inspected.returncode != 0 or not inspected.stdout.strip():
            raise ProfileError(f"deployment container is unavailable: {container}")
        digest.update(container.encode("utf-8"))
        digest.update(b"\0")
        digest.update(inspected.stdout.strip().encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _launch(profile: Profile, service: str) -> None:
    args = profile.flattened_args(service)
    if service == "asr":
        command = [sys.executable, "/opt/mazi/asr_server.py", *args]
    else:
        model_index = args.index("--model")
        model = args[model_index + 1]
        del args[model_index : model_index + 2]
        command = ["vllm", "serve", model, *args]
    os.execvp(command[0], command)


def _health(profile: Profile, service: str) -> None:
    port = profile.option(service, "--port")
    endpoint = "/readyz" if service == "asr" else "/health"
    url = f"http://127.0.0.1:{port}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=2.5) as response:
            if response.status != 200:
                raise ProfileError(f"{service} health returned HTTP {response.status}")
    except (urllib.error.URLError, TimeoutError) as error:
        raise ProfileError(f"{service} health failed: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "summary", "fingerprint", "deployment-fingerprint"):
        child = subparsers.add_parser(command)
        child.add_argument("profile")
    value_parser = subparsers.add_parser("value")
    value_parser.add_argument("profile")
    value_parser.add_argument("scope", choices=("meta", "asr", "tts"))
    value_parser.add_argument("key")
    for command in ("launch", "health"):
        child = subparsers.add_parser(command)
        child.add_argument("service", choices=("asr", "tts"))
        child.add_argument("profile")
    args = parser.parse_args()

    try:
        profile = load_profile(args.profile)
        if args.command in {"validate", "summary"}:
            print(json.dumps(_summary(profile), ensure_ascii=False, indent=2, sort_keys=True))
        elif args.command == "fingerprint":
            print(hashlib.sha256(profile.path.read_bytes()).hexdigest())
        elif args.command == "deployment-fingerprint":
            print(_deployment_fingerprint(profile))
        elif args.command == "value":
            if args.scope == "meta":
                if args.key not in profile.meta:
                    raise ProfileError(f"missing meta value {args.key}")
                print(profile.meta[args.key])
            else:
                value = profile.option(args.scope, args.key)
                if value is not None:
                    print(value)
        elif args.command == "launch":
            _launch(profile, args.service)
        elif args.command == "health":
            _health(profile, args.service)
        return 0
    except (ProfileError, OSError, ValueError) as error:
        print(f"profile error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

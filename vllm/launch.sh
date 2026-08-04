#!/usr/bin/env bash
set -Eeuo pipefail

profile="${1:-}"
if [[ ! "$profile" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  printf 'Invalid vLLM profile: %s\n' "$profile" >&2
  exit 64
fi

profile_file="/config/profiles/${profile}.args"
if [[ ! -r "$profile_file" ]]; then
  printf 'Profile args file is not readable: %s\n' "$profile_file" >&2
  exit 66
fi

profile_args=()
while IFS= read -r arg || [[ -n "$arg" ]]; do
  arg="${arg%$'\r'}"
  [[ -z "$arg" || "$arg" == \#* ]] && continue
  profile_args+=("$arg")
done < "$profile_file"

model_path="${profile_args[0]:-}"
if [[ "$model_path" != /models/* || ! -d "$model_path" || ! -r "$model_path/config.json" ]]; then
  printf 'First profile argument must be a readable model directory: %s\n' "$model_path" >&2
  exit 66
fi

arg_value() {
  local wanted="$1"
  local i
  for ((i = 0; i < ${#profile_args[@]}; i++)); do
    if [[ "${profile_args[i]}" == "$wanted" ]]; then
      printf '%s\n' "${profile_args[i + 1]:-}"
      return 0
    fi
    if [[ "${profile_args[i]}" == "$wanted="* ]]; then
      printf '%s\n' "${profile_args[i]#*=}"
      return 0
    fi
  done
  return 1
}

validate_model_directory() {
  local kind="$1"
  local path="$2"
  if [[ "$path" != /models/* || ! -d "$path" || ! -r "$path/config.json" ]]; then
    printf 'Profile %s must be a readable model directory: %s\n' "$kind" "$path" >&2
    exit 66
  fi
}

for flag in --model --speculative-model --draft-model; do
  artifact_path="$(arg_value "$flag" || true)"
  [[ -n "$artifact_path" ]] || continue
  validate_model_directory "$flag" "$artifact_path"
done

speculative_config="$(arg_value --speculative-config || true)"
if [[ -n "$speculative_config" ]]; then
  if ! speculative_model="$(python3 -c '
import json
import sys

value = json.loads(sys.argv[1])
if not isinstance(value, dict):
    raise SystemExit("--speculative-config must be a JSON object")
model = value.get("model", "")
if model and not isinstance(model, str):
    raise SystemExit("--speculative-config model must be a string")
print(model)
' "$speculative_config")"; then
    printf 'Invalid --speculative-config in profile: %s\n' "$profile" >&2
    exit 64
  fi
  if [[ -n "$speculative_model" ]]; then
    validate_model_directory --speculative-config.model "$speculative_model"
  fi
fi

printf 'Starting vLLM: profile=%s model=%s\n' "$profile" "$model_path"

exec vllm serve "${profile_args[@]}"

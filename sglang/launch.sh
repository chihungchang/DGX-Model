#!/usr/bin/env bash
set -Eeuo pipefail

profile="${1:-}"
if [[ ! "$profile" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  printf 'Invalid SGLang profile: %s\n' "$profile" >&2
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

model_path="$(arg_value --model-path || true)"
[[ -n "$model_path" ]] || model_path="$(arg_value --model || true)"
validate_model_directory --model-path "$model_path"

draft_path="$(arg_value --speculative-draft-model-path || true)"
[[ -n "$draft_path" ]] || draft_path="$(arg_value --speculative-draft-model || true)"
if [[ -n "$draft_path" ]]; then
  validate_model_directory --speculative-draft-model-path "$draft_path"
fi

printf 'Starting SGLang: profile=%s model=%s\n' "$profile" "$model_path"

exec python3 -m sglang.launch_server "${profile_args[@]}"

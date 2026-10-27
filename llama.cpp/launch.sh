#!/usr/bin/env bash
set -Eeuo pipefail

profile="${1:-}"
if [[ ! "$profile" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  printf 'Invalid llama.cpp profile: %s\n' "$profile" >&2
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

validate_model_file() {
  local kind="$1"
  local path="$2"
  if [[ "$path" != /models/* || ! -f "$path" || ! -r "$path" ]]; then
    printf 'Profile %s must be a readable model file: %s\n' "$kind" "$path" >&2
    exit 66
  fi
}

model_path="$(arg_value --model || true)"
validate_model_file --model "$model_path"

for flag in --spec-draft-model --model-draft --draft-model --mmproj; do
  artifact_path="$(arg_value "$flag" || true)"
  [[ -n "$artifact_path" ]] || continue
  validate_model_file "$flag" "$artifact_path"
done

# Split GGUF siblings are metadata artifacts, not llama-server arguments.
while read -r marker label kind artifact_path source checksum extra; do
  [[ "$marker" == '#' && "$label" == artifact: ]] || continue
  [[ -n "$kind" && -n "$source" && -n "$checksum" && -z "$extra" ]] || {
    printf 'Malformed artifact metadata in %s\n' "$profile_file" >&2
    exit 64
  }
  validate_model_file "$kind" "$artifact_path"
done < "$profile_file"

printf 'Starting llama.cpp: profile=%s model=%s\n' "$profile" "$model_path"

exec /app/llama-server "${profile_args[@]}"

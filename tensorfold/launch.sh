#!/usr/bin/env bash
set -Eeuo pipefail
profile="$1"
[[ "$profile" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || exit 64
profile_file="/config/profiles/$profile.args"
[[ -r "$profile_file" ]] || { echo "Missing profile: $profile_file" >&2; exit 66; }
[[ -r /model/config.json && -r /model/chat_template.jinja ]] || exit 66
[[ -r /engram/model-00047-of-00048.safetensors && -r /engram/model-00048-of-00048.safetensors ]] || exit 66
args=()
while IFS= read -r arg || [[ -n "$arg" ]]; do
  [[ -z "$arg" || "$arg" == \#* ]] && continue
  args+=("$arg")
done < "$profile_file"
exec python3 -c '
import ctypes, os, sys
# Inherited across exec: keep THP disabled for this engine without changing host settings.
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(41, 1, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
os.execvp("tensorfold", ["tensorfold", *sys.argv[1:]])
' "${args[@]}" --rank "$TENSORFOLD_RANK"

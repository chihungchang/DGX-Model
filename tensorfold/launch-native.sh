#!/usr/bin/env bash
set -Eeuo pipefail

profile="$1"
[[ "$profile" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || exit 64
profile_file="/config/profiles/$profile.args"
[[ -s "$profile_file" && -s /native/bin/tensorfold-native ]] || exit 66
[[ -s /cache/dsv41-token-map.json && -s /kit/aot/aot.json ]] || exit 66
[[ -s /engram/model-00047-of-00048.safetensors && -s /engram/model-00048-of-00048.safetensors ]] || exit 66

if [[ "$TENSORFOLD_RANK" == 1 ]]; then
  exec /native/bin/tf-dsv41-lanes /model /cache/tf-ds-rank 1 2 \
    "$TF_TP_MASTER" "$TF_TP_PORT" /kit \
    --engram /engram --token-map /cache/dsv41-token-map.json \
    --pool "$TF_DS_WINDOW" --context 262144 \
    --rdma "$TF_RDMA_DEVICES" --rdma-kernels /native/rdma_gather.fatbin \
    --graphs 1 --side 1 --prefetch 1 --aio 1 --streams 4 --round-rows 16
fi
[[ "$TENSORFOLD_RANK" == 0 ]] || exit 64
args=()
while IFS= read -r arg || [[ -n "$arg" ]]; do
  [[ -z "$arg" || "$arg" == \#* ]] && continue
  args+=("$arg")
done < "$profile_file"
exec /native/bin/tensorfold-native "${args[@]}"

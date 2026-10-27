#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
local_compose=(docker compose --env-file tp2-node0.env -f compose.dflash2-tp2.yaml)
remote_compose='cd /home/chihung/ai/sglang && docker compose --env-file tp2-node1.env -f compose.dflash2-tp2.yaml'
case "${1:-status}" in
  start)
    "${local_compose[@]}" config --quiet
    ssh -o BatchMode=yes spark2 "${remote_compose} config --quiet && docker image inspect dgx-sglang:qwen38-dflash2-nightly-20261016-localfix-v1 >/dev/null && test -r /home/chihung/models/huggingface/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal/model.safetensors"
    docker image inspect dgx-sglang:qwen38-dflash2-nightly-20261016-localfix-v1 >/dev/null
    test -r /home/chihung/models/huggingface/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal/model.safetensors
    if docker inspect dgx-sglang-tp2 >/dev/null 2>&1; then
      docker update --restart=no dgx-sglang-tp2 >/dev/null
    fi
    ssh spark2 'if docker inspect dgx-sglang-tp2 >/dev/null 2>&1; then docker update --restart=no dgx-sglang-tp2 >/dev/null; fi'
    ./tp2.sh stop
    for container in dgx-sglang dgx-sglang-tp2; do
      if docker inspect "$container" >/dev/null 2>&1; then
        docker update --restart=no "$container" >/dev/null
        docker stop -t 120 "$container" >/dev/null
      fi
    done
    ssh spark2 'docker update --restart=no dgx-sglang-tp2 >/dev/null'
    ssh spark2 "${remote_compose} up -d"
    "${local_compose[@]}" up -d
    docker update --restart=unless-stopped dgx-sglang-dflash2-tp2 >/dev/null
    ssh spark2 'docker update --restart=unless-stopped dgx-sglang-dflash2-tp2 >/dev/null'
    ;;
  stop)
    "${local_compose[@]}" stop
    ssh spark2 "${remote_compose} stop"
    ;;
  status)
    "${local_compose[@]}" ps
    ssh spark2 "${remote_compose} ps"
    ;;
  rollback|dspark)
    ./tp2.sh start
    ;;
  *) echo 'Usage: ./dflash2-tp2.sh {start|stop|status|dspark}' >&2; exit 64 ;;
esac

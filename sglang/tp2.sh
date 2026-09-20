#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
local_compose=(docker compose --env-file tp2-node0.env -f compose.tp2.yaml)
remote_compose='cd /home/chihung/ai/sglang && docker compose --env-file tp2-node1.env -f compose.tp2.yaml'
stop_dflash2() {
  if docker inspect dgx-sglang-dflash2-tp2 >/dev/null 2>&1; then
    docker update --restart=no dgx-sglang-dflash2-tp2 >/dev/null
    docker stop -t 120 dgx-sglang-dflash2-tp2 >/dev/null
  fi
  ssh spark2 'if docker inspect dgx-sglang-dflash2-tp2 >/dev/null 2>&1; then docker update --restart=no dgx-sglang-dflash2-tp2 >/dev/null && docker stop -t 120 dgx-sglang-dflash2-tp2 >/dev/null; fi'
}
case "${1:-status}" in
  start)
    ssh -o BatchMode=yes spark2 "${remote_compose} config --quiet"
    "${local_compose[@]}" config --quiet
    stop_dflash2
    if docker inspect dgx-sglang >/dev/null 2>&1; then
      docker update --restart=no dgx-sglang >/dev/null
      docker stop -t 120 dgx-sglang
    fi
    ssh spark2 "${remote_compose} up -d"
    "${local_compose[@]}" up -d
    docker update --restart=unless-stopped dgx-sglang-tp2 >/dev/null
    ssh spark2 'docker update --restart=unless-stopped dgx-sglang-tp2 >/dev/null'
    ;;
  stop)
    "${local_compose[@]}" stop
    ssh spark2 "${remote_compose} stop"
    ;;
  status)
    "${local_compose[@]}" ps
    ssh spark2 "${remote_compose} ps"
    ;;
  rollback)
    stop_dflash2
    "${local_compose[@]}" down
    ssh spark2 "${remote_compose} down"
    docker update --restart=unless-stopped dgx-sglang >/dev/null
    docker start dgx-sglang
    ;;
  *) echo 'Usage: ./tp2.sh {start|stop|status|rollback}' >&2; exit 64 ;;
esac

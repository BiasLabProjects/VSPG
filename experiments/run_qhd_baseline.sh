#!/usr/bin/env bash
# Figure 2 — QHD baseline (Ni et al., GLSVLSI'23), across the same 6 envs, 5
# seeds each, at the best-tuned config baked into configs/qhd/<env>.yaml (see
# ../HYPERPARAMETER_TUNING.md).
#
# Usage:
#   ./experiments/run_qhd_baseline.sh
#   ./experiments/run_qhd_baseline.sh --envs CartPole-v1 Empty-5x5
#   DRY_RUN=1 ./experiments/run_qhd_baseline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/_common.sh

declare -A DOMAIN=(
  [CartPole-v1]=classic_control [LunarLander-v2]=classic_control [Acrobot-v1]=classic_control
  [Empty-5x5]=minigrid [DoorKey-5x5]=minigrid [DoorKey-8x8]=minigrid
)
ALL_ENVS="CartPole-v1 LunarLander-v2 Acrobot-v1 Empty-5x5 DoorKey-5x5 DoorKey-8x8"
ALL_SEEDS="0 1 2 3 4"

ENVS="$ALL_ENVS"
SEEDS="$ALL_SEEDS"

while [ $# -gt 0 ]; do
  case "$1" in
    --envs) shift; ENVS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ENVS="$ENVS $1"; shift; done ;;
    --seeds) shift; SEEDS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS="$SEEDS $1"; shift; done ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

for env in $ENVS; do
  domain="${DOMAIN[$env]}"
  cfg="configs/qhd/${env}.yaml"
  [ -f "$cfg" ] || { log "[skip] no QHD config for $env: $cfg"; continue; }
  for seed in $SEEDS; do
    leaf="$RESULTS_ROOT/$domain/$env/qhd/seed$seed"
    if skip_if_done "$leaf"; then log "[skip] $leaf"; continue; fi
    maybe_mkdir "$leaf"
    log "[run]  $leaf"
    run_py -m qhd.train --config "$cfg" --seed "$seed" \
      --run-name "seed$seed" --log-dir "$leaf" --checkpoint-dir "$leaf"
  done
done

log "run_qhd_baseline.sh: done."

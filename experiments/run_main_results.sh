#!/usr/bin/env bash
# Figure 2 — main comparison: VSPG (Bipolar/FHRR/RFF) vs DNN vs Raw-Linear,
# across all 6 single-agent envs, 5 seeds each. Every run trains at the
# best-tuned config baked into configs/{classic_control,minigrid}/<env>_<method>.yaml
# (see ../HYPERPARAMETER_TUNING.md for how those configs were found).
#
# Usage:
#   ./experiments/run_main_results.sh                # everything
#   ./experiments/run_main_results.sh --envs CartPole-v1 Empty-5x5
#   ./experiments/run_main_results.sh --methods bipolar_vspg dnn
#   ./experiments/run_main_results.sh --seeds 0
#   DRY_RUN=1 ./experiments/run_main_results.sh       # preview commands only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/_common.sh

# env -> domain (config subdir + results subdir)
declare -A DOMAIN=(
  [CartPole-v1]=classic_control [LunarLander-v2]=classic_control [Acrobot-v1]=classic_control
  [Empty-5x5]=minigrid [DoorKey-5x5]=minigrid [DoorKey-8x8]=minigrid
)
ALL_ENVS="CartPole-v1 LunarLander-v2 Acrobot-v1 Empty-5x5 DoorKey-5x5 DoorKey-8x8"
ALL_METHODS="bipolar_vspg fhrr_vspg rff_vspg dnn raw_linear"
ALL_SEEDS="0 1 2 3 4"

ENVS="$ALL_ENVS"
METHODS="$ALL_METHODS"
SEEDS="$ALL_SEEDS"

while [ $# -gt 0 ]; do
  case "$1" in
    --envs) shift; ENVS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ENVS="$ENVS $1"; shift; done ;;
    --methods) shift; METHODS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do METHODS="$METHODS $1"; shift; done ;;
    --seeds) shift; SEEDS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS="$SEEDS $1"; shift; done ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

for env in $ENVS; do
  domain="${DOMAIN[$env]}"
  for method in $METHODS; do
    cfg="configs/$domain/${env}_${method}.yaml"
    [ -f "$cfg" ] || { log "[skip] no config for ($env, $method): $cfg"; continue; }
    for seed in $SEEDS; do
      leaf="$RESULTS_ROOT/$domain/$env/$method/seed$seed"
      if skip_if_done "$leaf"; then log "[skip] $leaf"; continue; fi
      maybe_mkdir "$leaf"
      log "[run]  $leaf"
      run_py -m vspg.train --config "$cfg" --seed "$seed" \
        --run-name "seed$seed" --log-dir "$leaf" --checkpoint-dir "$leaf"
    done
  done
done

log "run_main_results.sh: done."

#!/usr/bin/env bash
# Table 2 — SustainGym BuildingEnv, MAPPO-actor-swap comparison: DNN,
# Raw-Linear, and VSPG (Bipolar/FHRR/RFF), both climates, 5 seeds each, at the
# best-tuned config baked into configs/sustaingym/<climate>_<method>.yaml (see
# ../HYPERPARAMETER_TUNING.md). Requires the SustainGym env (see README.md).
#
# Usage:
#   ./experiments/run_sustaingym.sh
#   ./experiments/run_sustaingym.sh --climates hot_dry
#   ./experiments/run_sustaingym.sh --methods bipolar_vspg fhrr_vspg
#   DRY_RUN=1 ./experiments/run_sustaingym.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/_common.sh

CLIMATES="hot_dry warm_humid"
METHODS="dnn raw_linear bipolar_vspg fhrr_vspg rff_vspg"
SEEDS="0 1 2 3 4"

while [ $# -gt 0 ]; do
  case "$1" in
    --climates) shift; CLIMATES=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do CLIMATES="$CLIMATES $1"; shift; done ;;
    --methods) shift; METHODS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do METHODS="$METHODS $1"; shift; done ;;
    --seeds) shift; SEEDS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS="$SEEDS $1"; shift; done ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

for climate in $CLIMATES; do
  for method in $METHODS; do
    cfg="configs/sustaingym/${climate}_${method}.yaml"
    [ -f "$cfg" ] || { log "[skip] no config for ($climate, $method): $cfg"; continue; }
    parent="$RESULTS_ROOT/sustaingym/$climate/$method"
    for seed in $SEEDS; do
      # train_sustaingym.py writes to {result_dir}/{run_name}/summary.json, so
      # --log-dir/--checkpoint-dir/--result-dir take the parent and --run-name
      # supplies the per-seed leaf -- keeps the on-disk layout identical to
      # vspg.train's {leaf}/summary.json convention that skip_if_done expects.
      leaf="$parent/seed$seed"
      if skip_if_done "$leaf"; then log "[skip] $leaf"; continue; fi
      maybe_mkdir "$parent"
      log "[run]  $leaf"
      run_py train_sustaingym.py --config "$cfg" --seed "$seed" \
        --run-name "seed$seed" --log-dir "$parent" --checkpoint-dir "$parent" --result-dir "$parent"
    done
  done
done

log "run_sustaingym.sh: done."

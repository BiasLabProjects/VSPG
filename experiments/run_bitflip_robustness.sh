#!/usr/bin/env bash
# Figure 5 — bit-flip / quantization robustness of the stored SustainGym
# actor weights (Proposition 3). Wraps scripts/eval_bitflip_robustness.py,
# which post-training-quantizes each checkpoint to {1,2,4,8} bits, flips bits
# at swept probabilities, and evaluates on CLEAN observations. Writes a JSON
# summary only (no plots) to results/figures/<name>.json.
#
# Requires checkpoints from run_sustaingym.sh to already exist.
#
# Usage:
#   ./experiments/run_bitflip_robustness.sh smoke-test   # 1 checkpoint/method, hot_dry seed0 only
#   ./experiments/run_bitflip_robustness.sh full          # all 50 checkpoints, full sweep
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/_common.sh

METHODS="dnn raw_linear bipolar_vspg fhrr_vspg rff_vspg"

step_smoke_test() {
  local ckpts=""
  for m in $METHODS; do
    local d="$RESULTS_ROOT/sustaingym/hot_dry/$m/seed0"
    [ -d "$d" ] && ckpts="$ckpts $d"
  done
  if [ -z "$ckpts" ]; then
    log "[warn] no hot_dry/seed0 checkpoints found -- run experiments/run_sustaingym.sh first"
    return
  fi
  log "[smoke-test] checkpoints:$ckpts"
  run_py scripts/eval_bitflip_robustness.py --checkpoints $ckpts \
    --bits 4 8 --flip-probs 0 0.2 0.8 --trials 1 --episodes 5 \
    --out "$RESULTS_ROOT/figures/bitflip_smoketest.json"
}

step_full() {
  local ckpt_dirs=()
  for climate in hot_dry warm_humid; do
    for m in $METHODS; do
      for d in "$RESULTS_ROOT"/sustaingym/"$climate"/"$m"/seed*; do
        [ -d "$d" ] && ckpt_dirs+=("$d")
      done
    done
  done
  if [ "${#ckpt_dirs[@]}" -eq 0 ]; then
    log "[warn] no checkpoint dirs found -- run experiments/run_sustaingym.sh first"
    return
  fi
  log "[full] ${#ckpt_dirs[@]} checkpoint dirs, bit-flip robustness sweep"
  run_py scripts/eval_bitflip_robustness.py --checkpoints "${ckpt_dirs[@]}" \
    --bits 1 2 4 8 --flip-probs 0 0.05 0.1 0.2 0.4 0.6 0.8 --trials 1 --episodes 10 \
    --out "$RESULTS_ROOT/figures/bitflip_robustness_two_climates.json"
}

case "${1:-}" in
  smoke-test) step_smoke_test ;;
  full) step_full ;;
  *) echo "Usage: $0 {smoke-test|full}" >&2; exit 1 ;;
esac

log "run_bitflip_robustness.sh: done."

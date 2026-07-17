#!/usr/bin/env bash
# Proposition 1 verification: the manual closed-form VSPG update (RealHDPolicy)
# vs. three PyTorch-autograd equivalents (AutogradHDPolicy). First proves the
# algebraic identity on one real batch (sgd_ablation/tests/test_equivalence.py),
# then trains all variants across every tuned (env, encoder) pair, 5 seeds each.
#
# Usage:
#   ./experiments/run_sgd_ablation.sh                         # full ablation
#   ./experiments/run_sgd_ablation.sh --envs Empty-5x5 --encoders bipolar
#   ./experiments/run_sgd_ablation.sh --variants manual_vspg sgd_normalized
#   ./experiments/run_sgd_ablation.sh --episodes 200 --seeds 0   # fast smoke test
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/_common.sh

ENVS="CartPole-v1 LunarLander-v2 Acrobot-v1 Empty-5x5 DoorKey-5x5 DoorKey-8x8"
ENCODERS="bipolar fhrr rff"
VARIANTS="manual_vspg sgd_normalized sgd_unnormalized adam_normalized"
SEEDS="0 1 2 3 4"
EPISODES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --envs) shift; ENVS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ENVS="$ENVS $1"; shift; done ;;
    --encoders) shift; ENCODERS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ENCODERS="$ENCODERS $1"; shift; done ;;
    --variants) shift; VARIANTS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do VARIANTS="$VARIANTS $1"; shift; done ;;
    --seeds) shift; SEEDS=""; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SEEDS="$SEEDS $1"; shift; done ;;
    --episodes) shift; EPISODES="$1"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

log "[1/2] Numerical equivalence test (manual update == autograd REINFORCE gradient)"
run_py -m sgd_ablation.tests.test_equivalence

log "[2/2] Training: envs=[$ENVS] encoders=[$ENCODERS] variants=[$VARIANTS] seeds=[$SEEDS]"
EP_ARGS=()
[ -n "$EPISODES" ] && EP_ARGS=(--episodes "$EPISODES")
if [ "$DRY_RUN" = "1" ]; then
  log "[dry-run] python3 -m sgd_ablation.run_ablation --envs $ENVS --encoders $ENCODERS --variants $VARIANTS --seeds $SEEDS ${EP_ARGS[*]}"
else
  python3 -m sgd_ablation.run_ablation --envs $ENVS --encoders $ENCODERS --variants $VARIANTS --seeds $SEEDS "${EP_ARGS[@]}"
fi

log "run_sgd_ablation.sh: done."

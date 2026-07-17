#!/usr/bin/env bash
# Quick end-to-end sanity check (~1-2 min): confirms the manual closed-form
# VSPG update (RealHDPolicy) and the optimizer-based update
# (AutogradHDPolicy(mode="sgd_normalized")) both run correctly and agree,
# on Empty-5x5 at its best-tuned config (bipolar, tau=10.0, eta=0.005,
# dim=10000) at a reduced 200-episode budget. See ../HYPERPARAMETER_TUNING.md
# for how that config was found.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/2] Numerical equivalence test ==="
python3 -m sgd_ablation.tests.test_equivalence

echo
echo "=== [2/2] Manual vs. optimizer-based VSPG on Empty-5x5 (200 episodes, seed 0) ==="
python3 -m sgd_ablation.run_ablation \
  --envs Empty-5x5 --encoders bipolar \
  --variants manual_vspg sgd_normalized \
  --seeds 0 --episodes 200 \
  --results-root results/smoke_test

echo
echo "=== Summary ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

for variant in ("manual_vspg", "sgd_normalized"):
    path = Path(f"results/smoke_test/{variant}/Empty-5x5/bipolar/seed0/summary.json")
    d = json.load(open(path))
    print(f"{variant:16s}  final_return_mean_last100={d['final_return_mean_last100']:.4f}"
          f"  success_rate_last100={d['success_rate_last100']:.3f}")
PYEOF

echo
echo "smoke_test.sh: done."

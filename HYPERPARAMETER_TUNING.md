# Hyperparameter tuning budget

This repo trains directly at the best-tuned configs baked into `configs/` — it
does **not** re-run any grid search by default. This file documents the
search that produced those winning configs: the grid axes, the number of
points swept, the seeds used for tuning, and the selection metric, so a
reader can judge how much tuning budget backs each number and, if they want,
reproduce or extend the search themselves. Re-running a search is opt-in
future work, not part of the default reproduction path.

| Result family | Grid axes | Grid points | Seeds/point | Selection metric |
|---|---|---|---|---|
| VSPG MiniGrid Bipolar/FHRR (Empty-5x5, DoorKey-5x5, DoorKey-8x8) | τ ∈ {7, 10}, η ∈ {0.001, 0.005, 0.01, 0.05} | 8 | 1 (seed 42) | `final_return_mean_last100` |
| VSPG MiniGrid RFF (Empty-5x5, DoorKey-5x5) | τ ∈ {5, 10, 20}, η ∈ {0.005, 0.01, 0.05}, σ ∈ {0.5, 1.0} | 18 | 1 (seed 42) | `final_return_mean_last100` |
| VSPG MiniGrid RFF (DoorKey-8x8) | τ ∈ {5, 10}, η ∈ {0.01, 0.05}, σ ∈ {0.5, 1.0} (τ, σ transferred from DoorKey-5x5's winner; η re-swept — DoorKey-8x8 only converges at η=0.001, unlike DoorKey-5x5's η=0.005 winner) | 8 | 1 (seed 42) | `final_return_mean_last100` |
| VSPG classic control (CartPole-v1, LunarLander-v2, Acrobot-v1, all 3 encoders) | τ, η — organic/iterative search accumulated across several exploratory rounds (REINFORCE, REINFORCE+baseline, GAE+PPO-clip), not a single scripted grid | ~29 (Acrobot), ~12 (LunarLander), several dozen (CartPole) distinct (τ, η) cells on record | 1 (seed 42) per cell | `final_return_mean_last100` / solve threshold |
| DNN / Raw-Linear (classic control + MiniGrid) | DNN: lr ∈ {1e-4, 3e-4, 1e-3, 3e-3} × hidden ∈ {[64,32], [128,64], [256,128], [256,256]}; Linear: lr ∈ {1e-4, 3e-4, 1e-3, 3e-3} × τ ∈ {0.5, 1.0, 2.0, 5.0} | 16 each | 1 (seed 42) | `final_return_mean_last100` |
| QHD (all 6 envs) | β ∈ {0.001, 0.005, 0.01, 0.05} × batch ∈ {2, 4, 10, 32} × target-update ∈ {50, 200} × buffer ∈ {2000, 50000} | 64 | 1 (seed 0) | `final_return_mean_last100` |
| SustainGym (5 methods × 2 climates) | DNN: lr ∈ {1e-4, 3e-4, 1e-3, 3e-3} × hidden(4 options); Linear: lr(4) × τ ∈ {0.5, 1, 2, 5}; VSPG × 3 encoders: τ ∈ {0.5, 1, 2, 5} × η ∈ {0.001, 0.005, 0.01, 0.05} | 16 per method | 1 (seed 42) | `final_avg100_avg_reward_per_step` |

The classic-control VSPG row is reported honestly as an organic/manual search
rather than a tidy grid — that is genuinely how those numbers were found, and
inventing a cleaner-sounding budget would misrepresent the provenance.

**Total tuning cost** is dominated by SustainGym (16 points × 5 methods × 2
climates = 160 single-seed tuning runs) and QHD (64 points × 5 envs = 320
single-seed tuning runs); every other family is well under 100 tuning runs
per environment.

## Where the winning values live

- `configs/classic_control/<env>_<method>.yaml` and
  `configs/minigrid/<env>_<method>.yaml` — VSPG (bipolar/fhrr/rff), DNN, and
  Raw-Linear, one file per (env, method), with τ/η (or lr/hidden) already
  baked in.
- `configs/qhd/<env>.yaml` — QHD's winning (β, batch size, buffer size,
  target-update interval) per env.
- `configs/sustaingym/<climate>_<method>.yaml` — all 5 SustainGym methods ×
  2 climates, with τ/η (or lr/hidden) already baked in.
- `sgd_ablation/env_registry.py` — the same VSPG (env, encoder) → (τ, η, dim,
  σ/fhrr_w) table in code form, used by `experiments/run_sgd_ablation.sh` /
  `sgd_ablation/run_ablation.py` to hold every hyperparameter fixed while
  varying only the actor-update mechanism (manual vs. autograd).

# VSPG — Vector-Symbolic Policy Gradient

Official implementation of **Vector-Symbolic Policy Gradient (VSPG)**, a
categorical actor for discrete-action reinforcement learning that stores one
unit-norm hypervector per action and scores actions by similarity to an
encoded state. Under the standard softmax policy-gradient surrogate, the
exact actor update is advantage-weighted hypervector bundling followed by
row-wise normalization — a closed-form Hebbian-style update with no
backpropagation and no optimizer state. Trained action hypervectors form
compressed kernel memories over experience, and bipolar action memories
tolerate random bit flips with failure probability decaying exponentially in
the hypervector dimension.

This repository intentionally omits result-analysis and plotting code. Every
run writes `metrics.csv` (per-episode) and `summary.json` (final statistics +
hyperparameters) to its leaf directory; downstream analysis is left to the
reader.

## Code map

| Path | What it is |
|---|---|
| `vspg/policies.py:RealHDPolicy` | The manual, closed-form VSPG update (Algorithm 1): `C <- row_normalize(C + eta * Lambda^T @ S)`, no autograd. |
| `vspg/policies.py` (`MLPReinforcePolicy`, `LinearReinforceActor`) | DNN and Raw-Linear baselines. |
| `vspg/encoders.py` | The HDC encoders (Bipolar, FHRR, RFF) and the Gaussian/flat encoders used by baselines. |
| `vspg/train.py` | Single-agent training loop (classic control + MiniGrid). |
| `sgd_ablation/autograd_policy.py:AutogradHDPolicy` | The optimizer-based counterpart to `RealHDPolicy` — identical policy and initialization, but the update is computed via PyTorch autograd + `torch.optim` (SGD or Adam) instead of the closed-form rule. Verifies Proposition 1. |
| `qhd/` | QHD (Ni et al., GLSVLSI'23) value-based HDC baseline. |
| `policies/ma_vspg_policy.py`, `critics/centralized_critic.py`, `train_sustaingym.py` | Multi-agent VSPG (MAPPO-style, centralized critic, decentralized actors) for SustainGym building control. |
| `scripts/eval_bitflip_robustness.py` | Post-training quantization + random bit-flip corruption of stored actor weights (Proposition 3). |

## Install

**Local (everything except SustainGym):**

```bash
pip install -r requirements.txt
```

**SustainGym** needs its own environment (its `env_building.yml` pulls in
conda-only physics/build dependencies) — follow the upstream
[SustainGym building-control install instructions](https://github.com/chrisyeh96/sustaingym)
(`pip install "sustaingym[building]"` inside that environment), then make
sure this repo's root is on `PYTHONPATH`.

**Docker** (two separate images on purpose — see `Dockerfile`'s header):

```bash
docker compose build && docker compose up -d          # core: classic control, MiniGrid, QHD, sgd_ablation
docker compose -f docker/docker-compose.sustaingym.yml build   # SustainGym only
```

## Quickstart

```bash
./experiments/smoke_test.sh
```

Runs in about 1–2 minutes on CPU. It first proves the algebraic identity
behind Proposition 1 on one real batch of MiniGrid transitions
(`sgd_ablation/tests/test_equivalence.py`), then trains Empty-5x5 at its
best-tuned config (Bipolar, τ=10.0, η=0.005, dim=10000) for 200 episodes
through both `RealHDPolicy` (manual) and
`AutogradHDPolicy(mode="sgd_normalized")` (optimizer-based), and prints both
runs' final return/success rate side by side.

## Reproducing the paper

Every script below skips a run if its `summary.json` already exists, so they
are safe to interrupt and resume. See `HYPERPARAMETER_TUNING.md` for how the
configs each script trains at were found.

| Paper result | Command | Envs × methods × seeds |
|---|---|---|
| Fig. 2 — main comparison (VSPG, DNN, Raw-Linear) | `./experiments/run_main_results.sh` | 6 envs × 5 methods × 5 seeds |
| Fig. 2 — QHD baseline | `./experiments/run_qhd_baseline.sh` | 6 envs × 5 seeds |
| Proposition 1 — manual vs. optimizer-based update | `./experiments/run_sgd_ablation.sh` | 6 envs × up to 3 encoders × up to 4 update variants × 5 seeds |
| Table 2 — SustainGym building control | `./experiments/run_sustaingym.sh` | 2 climates × 5 methods × 5 seeds |
| Fig. 5 — bit-flip robustness | `./experiments/run_bitflip_robustness.sh {smoke-test\|full}` | reads checkpoints from `run_sustaingym.sh` |

Each script accepts `--envs`/`--methods`/`--seeds`/`--climates` (as
applicable) to run a subset, and respects `DRY_RUN=1` to print the planned
commands without executing them:

```bash
./experiments/run_main_results.sh --envs CartPole-v1 Empty-5x5 --seeds 0 1
DRY_RUN=1 ./experiments/run_sustaingym.sh
```

`run_main_results.sh` and `run_qhd_baseline.sh` are CPU/GPU-light per run but
cover 150 and 30 runs respectively at full episode budgets (up to 10,000
episodes for classic control) — expect a multi-hour full sweep; a `--seeds 0`
subset is much faster for a quick look. `run_sgd_ablation.sh` defaults to all
4 update variants; pass `--variants manual_vspg sgd_normalized` to halve the
cost. `run_sustaingym.sh` needs the SustainGym environment (see Install) and
is the most compute-heavy script (50 runs × 500 episodes of building
simulation) — normally run under `docker/docker-compose.sustaingym.yml` on a
GPU.

## Results layout

```
results/{classic_control,minigrid}/{env}/{method}/seed{S}/
    metrics.csv       per-episode return, length, entropy, ...
    summary.json      final return/success stats + hyperparameters
    config.yaml        resolved config for this run
    seed{S}_ep{N}.npz  periodic actor checkpoints (RealHDPolicy) or .pt (DNN/Linear)

results/sustaingym/{climate}/{method}/seed{S}/
    metrics.csv, summary.json, config.yaml, actor_final_agent{i}.{npz,pt}

results/sgd_ablation/{variant}/{env}/{encoder}/seed{S}/
    metrics.csv, summary.json, config.yaml
```

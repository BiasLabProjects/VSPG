"""Central registry of VSPG's per-(environment, encoder) best-tuned hyperparameters.

Every value below is read from a completed, existing results/ leaf in this
repository (config.yaml + summary.json of the winning tau/eta/sigma group for
that (env, encoder) pair) -- none of it is re-derived or re-tuned here. See
sgd_ablation/README.md's "Hyperparameter provenance" section for the exact
source directory of every row.

Two domains:
  - minigrid:     encoder_type in {bipolar, fhrr, rff}, uses obs_view_size (HDCEncoder/
                   FHRREncoder/RFFEncoder all consume the raw MiniGrid obs dict).
  - classic_gym:  encoder_type in {bipolar, fhrr_flat, rff}, uses obs_flat_dim.
                   NOTE: encoder_type="bipolar" here does NOT invoke HDCEncoder --
                   build_encoder() falls through to GaussianEncoder whenever
                   obs_flat_dim is set (vspg/encoders.py:614-619).
                   This is an existing property of the production pipeline, not
                   something introduced here; the method_name "bipolar_vspg" is
                   kept because that's what the production result trees call it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    domain: str                          # "minigrid" | "classic_gym"
    episodes: int
    gamma: float
    obs_view_size: int | None = None      # minigrid only
    obs_flat_dim: int | None = None       # classic_gym only
    score_range: tuple[float, float] = (0.0, 1.0)   # (ymin, ymax) for normalized AUC, paper Eq. (22)
    parallel_ok: bool = True              # False only for DoorKey-8x8, per explicit user instruction


ENVS: dict[str, EnvSpec] = {
    "Empty-5x5": EnvSpec(
        env_id="MiniGrid-Empty-5x5-v0", domain="minigrid", episodes=2000, gamma=0.99,
        obs_view_size=7, score_range=(0.0, 1.0),
    ),
    "DoorKey-5x5": EnvSpec(
        env_id="MiniGrid-DoorKey-5x5-v0", domain="minigrid", episodes=5000, gamma=0.99,
        obs_view_size=7, score_range=(0.0, 1.0),
    ),
    "DoorKey-8x8": EnvSpec(
        env_id="MiniGrid-DoorKey-8x8-v0", domain="minigrid", episodes=5000, gamma=0.99,
        obs_view_size=7, score_range=(0.0, 1.0), parallel_ok=False,
    ),
    "CartPole-v1": EnvSpec(
        env_id="CartPole-v1", domain="classic_gym", episodes=10000, gamma=0.99,
        obs_flat_dim=4, score_range=(0.0, 500.0),
    ),
    "LunarLander-v2": EnvSpec(
        env_id="LunarLander-v2", domain="classic_gym", episodes=10000, gamma=0.99,
        obs_flat_dim=8, score_range=(-200.0, 300.0),
    ),
    "Acrobot-v1": EnvSpec(
        env_id="Acrobot-v1", domain="classic_gym", episodes=10000, gamma=0.99,
        obs_flat_dim=6, score_range=(-500.0, 0.0),
    ),
}


@dataclass(frozen=True)
class EncoderSpec:
    encoder_type: str             # passed straight through to build_encoder() via cfg.encoder_type
    method_name: str              # matches production results-dir naming: bipolar_vspg | fhrr_vspg | rff_vspg
    tau: float
    eta: float
    dim: int = 10000
    rff_sigma: float | None = None
    fhrr_w: float | None = None
    advantage_type: str = "reinforce"
    gae_lambda: float | None = None
    ppo_clip_eps: float = 0.0
    value_fn_hidden_sizes: tuple[int, ...] = (128, 128)
    value_fn_lr: float = 3e-4


# (env_short, encoder_key) -> EncoderSpec. encoder_key is always one of
# "bipolar" | "fhrr" | "rff" (the short, domain-agnostic name used throughout
# sgd_ablation/ for CLI args and result paths); EncoderSpec.encoder_type holds
# the actual cfg.encoder_type string build_encoder() expects, which differs
# for "fhrr" between domains (minigrid: "fhrr", classic_gym: "fhrr_flat").
HPARAMS: dict[tuple[str, str], EncoderSpec] = {
    # Empty-5x5: results/minigrid/{bipolar,fhrr,rff}_vspg/Empty-5x5/tau*_eta*[_sig*]_seed{0-4}/
    ("Empty-5x5", "bipolar"): EncoderSpec("bipolar", "bipolar_vspg", tau=10.0, eta=0.005),
    ("Empty-5x5", "fhrr"):    EncoderSpec("fhrr",    "fhrr_vspg",    tau=10.0, eta=0.01, fhrr_w=1.0),
    ("Empty-5x5", "rff"):     EncoderSpec("rff",     "rff_vspg",     tau=20.0, eta=0.01, rff_sigma=1.0),

    # DoorKey-5x5: bipolar/fhrr from results_vspg_baselines/{method}/DoorKey-5x5_t100_e001_seed{0-4}_*/
    #              rff from results/minigrid/rff_vspg/DoorKey-5x5/tau10.0_eta0.005_sig1.0_seed{0-4}/
    ("DoorKey-5x5", "bipolar"): EncoderSpec("bipolar", "bipolar_vspg", tau=10.0, eta=0.001),
    ("DoorKey-5x5", "fhrr"):    EncoderSpec("fhrr",    "fhrr_vspg",    tau=10.0, eta=0.001, fhrr_w=1.0),
    ("DoorKey-5x5", "rff"):     EncoderSpec("rff",     "rff_vspg",     tau=10.0, eta=0.005, rff_sigma=1.0),

    # DoorKey-8x8: bipolar/fhrr from results_vspg_baselines/{method}/DoorKey-8x8_t100_e001_seed{0-4}_*/
    #              rff from results/minigrid/rff_vspg/DoorKey-8x8/tau5.0_eta0.05_sig0.5_seed{0-4}/
    # (rff's tuned config essentially fails on DoorKey-8x8, ~0.03 success rate --
    #  kept as-is since it IS the actual best-found rff config, not a mistake.)
    ("DoorKey-8x8", "bipolar"): EncoderSpec("bipolar", "bipolar_vspg", tau=10.0, eta=0.001),
    ("DoorKey-8x8", "fhrr"):    EncoderSpec("fhrr",    "fhrr_vspg",    tau=10.0, eta=0.001, fhrr_w=1.0),
    ("DoorKey-8x8", "rff"):     EncoderSpec("rff",     "rff_vspg",     tau=5.0,  eta=0.05, rff_sigma=0.5),

    # CartPole-v1: results/classic_gym/{method}/CartPole-v1/tau40.0_eta1e-5_seed{0-4}/
    ("CartPole-v1", "bipolar"): EncoderSpec("bipolar",   "bipolar_vspg", tau=40.0, eta=1e-5),
    ("CartPole-v1", "fhrr"):    EncoderSpec("fhrr_flat",  "fhrr_vspg",    tau=40.0, eta=1e-5, fhrr_w=1.0),
    ("CartPole-v1", "rff"):     EncoderSpec("rff",        "rff_vspg",     tau=40.0, eta=1e-5, rff_sigma=1.0),

    # LunarLander-v2: results/classic_gym/{method}/LunarLander-v2/tau40.0_eta1e-5_seed{0-4}/
    ("LunarLander-v2", "bipolar"): EncoderSpec("bipolar",   "bipolar_vspg", tau=40.0, eta=1e-5),
    ("LunarLander-v2", "fhrr"):    EncoderSpec("fhrr_flat",  "fhrr_vspg",    tau=40.0, eta=1e-5, fhrr_w=1.0),
    ("LunarLander-v2", "rff"):     EncoderSpec("rff",        "rff_vspg",     tau=40.0, eta=1e-5, rff_sigma=1.0),

    # Acrobot-v1: results/classic_gym/{method}/Acrobot-v1/tau10.0_eta1e-3_seed{0-4}/
    # Only bipolar was tuned+found under GAE+PPO-clip (the config that actually
    # solves Acrobot, mean return ~-86 across 5/5 seeds); fhrr/rff were tuned
    # under plain REINFORCE and both fail to solve (fhrr: stuck at -500 on all
    # 5 seeds; rff: -421.6 mean) -- kept as-is since that IS the real tuned
    # result for those two encoders on this env, not a bug in this registry.
    ("Acrobot-v1", "bipolar"): EncoderSpec(
        "bipolar", "bipolar_vspg", tau=10.0, eta=1e-3,
        advantage_type="gae", gae_lambda=0.95, ppo_clip_eps=0.2,
        value_fn_hidden_sizes=(128, 128), value_fn_lr=3e-4,
    ),
    ("Acrobot-v1", "fhrr"): EncoderSpec("fhrr_flat", "fhrr_vspg", tau=10.0, eta=1e-3, fhrr_w=1.0),
    ("Acrobot-v1", "rff"):  EncoderSpec("rff",       "rff_vspg",  tau=10.0, eta=1e-3, rff_sigma=1.0),
}

ALL_PAIRS: list[tuple[str, str]] = list(HPARAMS.keys())


def encoders_for(env_short: str) -> list[str]:
    return [enc for (env, enc) in ALL_PAIRS if env == env_short]

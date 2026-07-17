"""Isolated SGD-ablation runner for VSPG, generalized across all of VSPG's
single-agent benchmark environments and encoders (see sgd_ablation/env_registry.py
for the per-(env, encoder) hyperparameter table, all values reused unmodified
from this repo's existing best-tuned production results).

Trains 4 actor-update variants under otherwise identical conditions (same
initialization, encoder, temperature, advantage estimator, episode budget,
and per-seed random seeds):

    manual_vspg        RealHDPolicy                          -- existing, unmodified
    sgd_normalized      AutogradHDPolicy(mode="sgd_normalized")   -- autograd SGD + row L2-normalize
    sgd_unnormalized    AutogradHDPolicy(mode="sgd_unnormalized") -- autograd SGD, no normalize
    adam_normalized      AutogradHDPolicy(mode="adam_normalized")  -- autograd Adam + row L2-normalize

Reuses (imports only -- none of these are reimplemented):
    build_encoder                             vspg.encoders
    RealHDPolicy                              vspg.policies
    load_config, collect_episode,
    compute_reinforce_advantages, compute_gae  vspg.train
    MetricsLogger, compute_run_summary,
    normalize_advantages, compute_returns      vspg.utils
    SimpleValueFunction                        vspg.value_function

Deliberately does NOT call vspg.train.train() -- that function
is monolithic (wandb, plotting, checkpoint cruft) and its policy factory
build_policy() has no concept of AutogradHDPolicy's three modes. The thin
per-episode loop below is the intentional, isolated reimplementation the
task calls for; episode collection, advantage computation, and GAE are not
reimplemented.

Two advantage paths, matching each (env, encoder)'s actual best-tuned config
from sgd_ablation/env_registry.py:
  - reinforce (default): compute_reinforce_advantages, no value function.
  - gae (Acrobot-v1's bipolar config only): SimpleValueFunction critic +
    compute_gae + normalize_advantages, with PPO-clip on the actor update
    (log_probs_old + ppo_clip_eps) -- mirrors vspg/train.py's
    own "gae" branch exactly.

Backward compatibility: `--envs`/`--encoders` default to ["DoorKey-8x8"]/
["bipolar"] -- the exact original single-env ablation -- and that specific
(env, encoder) pair keeps writing to the original flat
results/sgd_ablation/<variant>/DoorKey-8x8/seed<seed>/ path (no encoder
subdirectory), so existing/in-progress runs of experiments/sgd_sanity.sh are
unaffected. Every other (env, encoder) pair writes to the new nested
results/sgd_ablation/<variant>/<env>/<encoder>/seed<seed>/ path.

Usage:
    python3 sgd_ablation/run_ablation.py --seeds 0 1 2 3 4
    python3 sgd_ablation/run_ablation.py --seeds 0 --episodes 200   # smoke test
    python3 sgd_ablation/run_ablation.py --envs CartPole-v1 --encoders rff --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
np.float_ = np.float64  # gymnasium/minigrid + modern numpy compat shim, same as vspg/train.py

import yaml
from tqdm import tqdm

from vspg.encoders import build_encoder
from vspg.policies import RealHDPolicy
from vspg.train import load_config, collect_episode, compute_reinforce_advantages
from vspg.utils import (
    MetricsLogger, compute_run_summary, normalize_advantages, compute_returns, compute_gae,
)
from vspg.value_function import SimpleValueFunction
from sgd_ablation.autograd_policy import AutogradHDPolicy
from sgd_ablation.env_registry import ENVS, HPARAMS

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "ablation_base.yaml")
_DEFAULT_RESULTS_ROOT = "results/sgd_ablation"

VARIANTS = {
    "manual_vspg":      dict(kind="manual"),
    "sgd_normalized":   dict(kind="autograd", mode="sgd_normalized"),
    "sgd_unnormalized": dict(kind="autograd", mode="sgd_unnormalized"),
    "adam_normalized":  dict(kind="autograd", mode="adam_normalized"),
}


def _select_device(cfg) -> str:
    import torch as _torch
    requested = getattr(cfg, "device", "cuda")
    if requested != "cpu" and not _torch.cuda.is_available():
        print("  [warn] CUDA not available, falling back to CPU")
        return "cpu"
    return requested


def _build_policy(variant_name: str, cfg, n_actions: int, device: str, zero_init: bool = False):
    spec = VARIANTS[variant_name]
    if spec["kind"] == "manual":
        return RealHDPolicy(
            dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=cfg.seed,
            device=device, row_normalize=bool(getattr(cfg, "row_normalize", True)),
            zero_init=zero_init,
        )
    return AutogradHDPolicy(
        dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=cfg.seed, device=device,
        mode=spec["mode"], eta=cfg.eta, adam_lr=getattr(cfg, "adam_lr", cfg.eta),
        zero_init=zero_init,
    )


def _cfg_overrides(env_short: str, encoder_key: str, seed: int, episodes_override: int | None,
                    device_override: str | None = None, tau_override: float | None = None,
                    eta_override: float | None = None) -> dict:
    env_spec = ENVS[env_short]
    enc_spec = HPARAMS[(env_short, encoder_key)]

    tau = tau_override if tau_override is not None else enc_spec.tau
    eta = eta_override if eta_override is not None else enc_spec.eta

    overrides = {
        "env_id": env_spec.env_id,
        "variant": "real",
        "encoder_type": enc_spec.encoder_type,
        "dim": enc_spec.dim,
        "tau": tau,
        "eta": eta,
        "gamma": env_spec.gamma,
        "episodes": episodes_override if episodes_override is not None else env_spec.episodes,
        "advantage_type": enc_spec.advantage_type,
        "seed": seed,
        "row_normalize": True,
        "adam_lr": eta,  # matches eta by design -- see README "Design decisions" -- tracks the (possibly overridden) eta
    }
    if device_override is not None:
        overrides["device"] = device_override
    if env_spec.obs_view_size is not None:
        overrides["obs_view_size"] = env_spec.obs_view_size
    if env_spec.obs_flat_dim is not None:
        overrides["obs_flat_dim"] = env_spec.obs_flat_dim
    if enc_spec.rff_sigma is not None:
        overrides["rff_sigma"] = enc_spec.rff_sigma
    if enc_spec.fhrr_w is not None:
        overrides["fhrr_w"] = enc_spec.fhrr_w
    if enc_spec.advantage_type == "gae":
        overrides["gae_lambda"] = enc_spec.gae_lambda
        overrides["ppo_clip_eps"] = enc_spec.ppo_clip_eps
        overrides["value_fn_hidden_sizes"] = list(enc_spec.value_fn_hidden_sizes)
        overrides["value_fn_lr"] = enc_spec.value_fn_lr
    return overrides


def _leaf_dir(results_root: str, variant_name: str, env_short: str, encoder_key: str, seed: int,
              zero_init: bool = False) -> Path:
    variant_folder = f"{variant_name}_zeroinit" if zero_init else variant_name
    if env_short == "DoorKey-8x8" and encoder_key == "bipolar" and not zero_init:
        # Backward-compat with the original (pre-multi-env) ablation path --
        # see module docstring. Only applies to standard-init runs: zero-init
        # runs always use the nested path (via the _zeroinit-suffixed
        # variant_folder above), so they can never collide with that
        # already-populated flat path, or with each other.
        return Path(results_root) / variant_folder / env_short / f"seed{seed}"
    return Path(results_root) / variant_folder / env_short / encoder_key / f"seed{seed}"


def run_one(env_short: str, encoder_key: str, variant_name: str, seed: int, cfg_path: str,
            episodes_override: int | None, results_root: str, device_override: str | None = None,
            zero_init: bool = False, tau_override: float | None = None,
            eta_override: float | None = None) -> dict:
    import gymnasium as gym
    import minigrid  # noqa: F401  registers MiniGrid envs

    overrides = _cfg_overrides(env_short, encoder_key, seed, episodes_override, device_override,
                                tau_override, eta_override)
    cfg = load_config(cfg_path, overrides=overrides)
    use_gae = (getattr(cfg, "advantage_type", "reinforce") == "gae")

    # Replicate vspg/train.py::train()'s seeding order exactly:
    # global numpy seed set before env/encoder/policy construction.
    np.random.seed(cfg.seed)

    env = gym.make(cfg.env_id)
    device = _select_device(cfg)
    encoder = build_encoder(cfg)
    n_actions = env.action_space.n

    policy = _build_policy(variant_name, cfg, n_actions, device, zero_init=zero_init)
    logger = MetricsLogger()

    value_fn = None
    ppo_clip_eps = 0.0
    if use_gae:
        value_fn = SimpleValueFunction(
            obs_dim=cfg.dim,
            hidden_sizes=tuple(getattr(cfg, "value_fn_hidden_sizes", (128, 128))),
            lr=float(getattr(cfg, "value_fn_lr", 3e-4)),
            device=device,
        )
        ppo_clip_eps = float(getattr(cfg, "ppo_clip_eps", 0.0))

    zero_init_tag = "_zeroinit" if zero_init else ""
    tqdm.write(f"[sgd_ablation] env={env_short} encoder={encoder_key} variant={variant_name}{zero_init_tag} "
               f"seed={seed} episodes={cfg.episodes} dim={cfg.dim} tau={cfg.tau} eta={cfg.eta} "
               f"adv={cfg.advantage_type} device={device}")

    t0 = time.time()
    pbar = tqdm(range(cfg.episodes), desc=f"{env_short}/{encoder_key}/{variant_name}{zero_init_tag}/seed{seed}",
                unit="ep", dynamic_ncols=True)
    for ep in pbar:
        # env.reset() inside collect_episode is intentionally left unseeded,
        # exactly matching vspg/train.py's collect_episode --
        # see sgd_ablation/README.md ("Env seeding" section) for why.
        traj, ep_return, ep_len = collect_episode(env, policy, encoder, cfg, value_fn=value_fn)
        s_hvs = [step[0] for step in traj]
        actions = [step[1] for step in traj]

        if use_gae:
            rewards = [step[3] for step in traj]
            values = [step[6] for step in traj]
            advantages = normalize_advantages(
                compute_gae(rewards, values, cfg.gamma, getattr(cfg, "gae_lambda", 0.95)))
            log_probs_old = np.array([step[2] for step in traj]) if ppo_clip_eps > 0.0 else None
            update_metrics = policy.update(s_hvs, actions, advantages, cfg.eta,
                                            log_probs_old=log_probs_old, ppo_clip_eps=ppo_clip_eps)
            vf_loss = value_fn.update(s_hvs, compute_returns(rewards, cfg.gamma))
            update_metrics["value_fn_loss"] = vf_loss
        else:
            advantages, _adv_info = compute_reinforce_advantages([(traj, ep_return, ep_len)], cfg.gamma)
            update_metrics = policy.update(s_hvs, actions, advantages, cfg.eta)

        entropies = [step[4] for step in traj]
        logger.log(ep, {
            "return": ep_return,
            "length": ep_len,
            "mean_entropy": float(np.mean(entropies)),
            "mean_abs_adv": float(np.mean(np.abs(advantages))),
            **update_metrics,
        })

        avg100 = logger.moving_avg("return")
        recent_returns = np.array(logger.series("return")[-100:])
        success100 = float((recent_returns > 0).mean())
        pbar.set_postfix(avg100_return=f"{avg100:.3f}", success100=f"{success100:.2f}", refresh=False)

    elapsed = time.time() - t0

    leaf = _leaf_dir(results_root, variant_name, env_short, encoder_key, seed, zero_init=zero_init)
    leaf.mkdir(parents=True, exist_ok=True)
    logger.save_csv(leaf / "metrics.csv")

    summary = compute_run_summary(logger)
    summary.update({
        "variant": variant_name,
        "zero_init": zero_init,
        "env": env_short,
        "encoder": encoder_key,
        "seed": seed,
        "env_id": cfg.env_id,
        "dim": cfg.dim,
        "tau": cfg.tau,
        "eta": cfg.eta,
        "advantage_type": cfg.advantage_type,
        "episodes": cfg.episodes,
        "elapsed_sec": elapsed,
    })
    with open(leaf / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(leaf / "config.yaml", "w") as f:
        yaml.safe_dump(vars(cfg), f)

    print(f"[sgd_ablation] done  env={env_short} encoder={encoder_key} variant={variant_name}{zero_init_tag} "
          f"seed={seed}  success_rate_last100={summary['success_rate_last100']:.3f}  "
          f"final_return_mean_last100={summary['final_return_mean_last100']:.3f}  "
          f"elapsed={elapsed:.1f}s  -> {leaf}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated SGD ablation of VSPG, generalized across all envs/encoders")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--envs", nargs="+", default=["DoorKey-8x8"], choices=list(ENVS.keys()))
    parser.add_argument("--encoders", nargs="+", default=["bipolar"], choices=["bipolar", "fhrr", "rff"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                         choices=list(VARIANTS.keys()))
    parser.add_argument("--episodes", type=int, default=None,
                         help="Override cfg.episodes (e.g. for a fast smoke test)")
    parser.add_argument("--device", default=None,
                         help="Override cfg.device (e.g. 'cpu' to force CPU even when CUDA is available -- "
                              "useful when running many jobs concurrently, since they'd otherwise contend "
                              "for one GPU). Default: whatever the config says (cuda, falling back to cpu "
                              "if unavailable).")
    parser.add_argument("--results-root", default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument("--zero-init", action="store_true",
                         help="Initialize C to exact zero instead of the usual random-then-row-normalized "
                              "draw (see RealHDPolicy(zero_init=...)). Results land in a "
                              "'<variant>_zeroinit' leaf so they never collide with standard-init runs. "
                              "See sgd_ablation/run_zero_init_ablation.py for a dedicated entrypoint that "
                              "always sets this.")
    args = parser.parse_args()

    for env_short in args.envs:
        for encoder_key in args.encoders:
            if (env_short, encoder_key) not in HPARAMS:
                print(f"[sgd_ablation] [skip] no tuned hyperparameters for ({env_short}, {encoder_key})")
                continue
            for variant_name in args.variants:
                for seed in args.seeds:
                    run_one(env_short, encoder_key, variant_name, seed, args.config,
                            args.episodes, args.results_root, args.device, args.zero_init)


if __name__ == "__main__":
    main()

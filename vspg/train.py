"""Training loop for Advantage-Weighted OnlineHD Policy Learning on MiniGrid.

Usage:
    python vspg/train.py --config vspg/configs/empty5x5_real.yaml
    python vspg/train.py --config ... --dim 5000 --episodes 5000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
np.float_ = np.float64

import yaml
from tqdm import tqdm

import wandb

from .encoders import build_encoder
from .policies import build_policy
from .utils import MetricsLogger, compute_returns, compute_gae, compute_run_summary, normalize_advantages
from .value_function import SimpleValueFunction


def _make_run_name(cfg) -> str:
    """Build a fully-descriptive W&B run name from all experiment conditions."""
    variant = getattr(cfg, "variant", "real")

    if variant == "mlp_reinforce":
        return (f"mlp_reinforce"
                f"_lr{getattr(cfg, 'mlp_lr', 3e-4)}"
                f"_s{getattr(cfg, 'seed', 42)}")

    if variant == "linear_reinforce":
        return (f"linear_reinforce"
                f"_lr{getattr(cfg, 'linear_lr', 3e-4)}"
                f"_tau{getattr(cfg, 'tau', 1.0)}"
                f"_s{getattr(cfg, 'seed', 42)}")

    parts = []

    # encoder + encoder-specific params
    enc = getattr(cfg, "encoder_type", "bipolar")
    if enc == "fhrr":
        parts.append(f"fhrr_w{getattr(cfg, 'fhrr_w', 1.0)}")
    elif enc == "rff":
        parts.append(f"rff_sig{getattr(cfg, 'rff_sigma', 1.0)}")
    else:
        parts.append(enc)

    # advantage method + group params
    adv = getattr(cfg, "advantage_type", "reinforce")
    adv_str = adv
    if adv in ("grpo", "gigpo"):
        adv_str += f"_g{getattr(cfg, 'group_size', 8)}"
    if adv == "gigpo":
        omega = getattr(cfg, "gigpo_omega", 1.0)
        adv_str += f"_om{omega}"
    parts.append(adv_str)

    # core hyperparameters
    parts.append(f"tau{getattr(cfg, 'tau', 1.0)}")
    parts.append(f"eta{getattr(cfg, 'eta', 0.05)}")
    parts.append(f"D{getattr(cfg, 'dim', 1000)}")

    # optional flags (only shown when non-default)
    if getattr(cfg, "fhrr_init", False):
        parts.append("fhrr_init")
    tw = getattr(cfg, "tau_warmup", 0)
    if tw > 0:
        parts.append(f"tw{tw}")

    # seed always last
    parts.append(f"s{getattr(cfg, 'seed', 42)}")

    return "_".join(parts)


def _init_wandb(cfg):
    wandb_dir = getattr(cfg, "wandb_dir", "wandb_runs")
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        entity=getattr(cfg, "wandb_entity", None),
        project=getattr(cfg, "wandb_project", "memory_comprehensive"),
        name=_make_run_name(cfg),
        config=vars(cfg),
        dir=wandb_dir,
        reinit="finish_previous",
    )


# ── config ────────────────────────────────────────────────────────────────────


def load_config(yaml_path: str, overrides: dict) -> SimpleNamespace:
    with open(yaml_path) as f:
        cfg_dict = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg_dict[k] = v
    # Derive a unique run name from the config filename stem, e.g.
    # "empty5x5_fhrr_w10_grpo.yaml" → "fhrr_w10_grpo".  The first
    # underscore-delimited component is always the env shorthand.
    stem = Path(yaml_path).stem
    parts = stem.split("_", 1)
    run_name = parts[1] if len(parts) > 1 else stem

    # Normalise tau/eta tags to 3-digit zero-padded form so configs from
    # different directories with the same parameters produce the same run name.
    # e.g. "fhrr_grpo_t50_e010" → "fhrr_grpo_t050_e010"
    import re as _re
    if "tau" in cfg_dict:
        canonical_t = f"t{int(round(cfg_dict['tau'] * 10)):03d}"
        run_name = _re.sub(r'\bt\d+\b', canonical_t, run_name)
    if "eta" in cfg_dict:
        canonical_e = f"e{int(round(cfg_dict['eta'] * 1000)):03d}"
        run_name = _re.sub(r'\be\d+\b', canonical_e, run_name)

    cfg_dict.setdefault("run_name", run_name)
    return SimpleNamespace(**cfg_dict)


# ── episode collection ────────────────────────────────────────────────────────


def _state_key(obs) -> bytes:
    """Compact hashable key for a MiniGrid state (image + direction) or flat array."""
    if isinstance(obs, dict):
        return obs["image"].tobytes() + bytes([int(obs["direction"])])
    return obs.tobytes()


def collect_episode(env, policy, encoder, cfg, value_fn=None) -> tuple[list, float, int]:
    """Run one episode; return (trajectory, total_return, episode_length).

    trajectory: list of (s_hv, action, logp, reward, entropy, state_key[, v_t])
    v_t (index 6) is included only when value_fn is not None, for GAE estimation.
    state_key is a bytes hash used for GiGPO anchor state grouping.
    """
    use_bipolar = getattr(cfg, "variant", "real") == "onebit"
    obs, _ = env.reset()
    if hasattr(encoder, 'reset'):
        encoder.reset()

    trajectory = []
    ep_return = 0.0
    done = False
    prev_action = None

    while not done:
        sk = _state_key(obs)
        if hasattr(encoder, 'step'):
            s_hv = encoder.step(obs, prev_action=prev_action)
        elif use_bipolar:
            s_hv = encoder.encode_bipolar(obs)
        else:
            s_hv = encoder.encode(obs)

        action, logp, entropy = policy.act(s_hv)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        v_t = value_fn.predict(s_hv) if value_fn is not None else 0.0
        trajectory.append((s_hv, action, logp, float(reward), entropy, sk, v_t))
        ep_return += float(reward)
        obs = next_obs
        prev_action = action

    return trajectory, ep_return, len(trajectory)


# ── group-based advantage computation ─────────────────────────────────────────


def _ep_returns(group_trajs):
    return np.array([ep_r for _, ep_r, _ in group_trajs], dtype=np.float32)


def compute_reinforce_advantages(group_trajs, gamma):
    """Per-episode normalized discounted returns (classic REINFORCE).
    Works for group_size=1 or >1 (each episode normalized independently).
    """
    all_adv = []
    for traj, _, _ in group_trajs:
        rewards = [s[3] for s in traj]
        all_adv.extend(normalize_advantages(compute_returns(rewards, gamma)))
    return np.array(all_adv, dtype=np.float32), {}


def compute_reinforce_baseline_advantages(group_trajs, gamma, baseline):
    """REINFORCE with moving-average baseline: A_t = G_t - baseline.

    Unlike classic REINFORCE, returns are not normalized — the baseline
    is subtracted but the scale is preserved, which matters when eta is very
    small (continuous-obs environments like CartPole).
    """
    all_adv = []
    for traj, _, _ in group_trajs:
        rewards = [s[3] for s in traj]
        all_adv.extend([G - baseline for G in compute_returns(rewards, gamma)])
    return np.array(all_adv, dtype=np.float32), {"baseline": baseline}


def compute_grpo_advantages(group_trajs, gamma):
    """GRPO: episode returns normalized across the group, broadcast to all steps.

    A^E(τ_i) = (R(τ_i) − mean_j R(τ_j)) / (std_j R(τ_j) + ε)
    Every step in episode i gets advantage A^E(τ_i).
    """
    ep_ret = _ep_returns(group_trajs)
    std = ep_ret.std()
    A_E = (ep_ret - ep_ret.mean()) / (std + 1e-8) if std >= 1e-8 else np.zeros(len(group_trajs))

    all_adv = []
    for i, (traj, _, _) in enumerate(group_trajs):
        all_adv.extend([float(A_E[i])] * len(traj))
    return np.array(all_adv, dtype=np.float32), {}


def compute_gigpo_advantages(group_trajs, gamma, omega):
    """GiGPO: episode-level GRPO + step-level anchor state relative advantage.

    A(a_t^i) = A^E(τ_i)  +  ω · A^S(a_t^i)

    A^E: GRPO over episode returns (macro, trajectory-wide signal).
    A^S: for each unique state that recurs across trajectories, normalize the
         discounted returns of all (action, state) pairs in that anchor group
         → fine-grained per-step credit, no value network required.

    Anchor states are identified by exact byte-match of (image, direction).
    Singleton groups (state appears only once) contribute A^S = 0.
    """
    from collections import defaultdict

    # ── episode-level advantage ───────────────────────────────────────────────
    ep_ret = _ep_returns(group_trajs)
    std_e = ep_ret.std()
    A_E = (ep_ret - ep_ret.mean()) / (std_e + 1e-8) if std_e >= 1e-8 else np.zeros(len(group_trajs))

    # ── step-level discounted returns ─────────────────────────────────────────
    step_rtgs = [
        compute_returns([s[3] for s in traj], gamma)
        for traj, _, _ in group_trajs
    ]

    # ── anchor state grouping ─────────────────────────────────────────────────
    state_groups: dict = defaultdict(list)   # key → [(traj_i, step_t, rtg)]
    for i, (traj, _, _) in enumerate(group_trajs):
        for t, step in enumerate(traj):
            state_groups[step[5]].append((i, t, step_rtgs[i][t]))

    # ── step-level relative advantage ────────────────────────────────────────
    A_S = [[0.0] * len(traj) for traj, _, _ in group_trajs]
    n_anchor = 0
    total_sz = 0
    for _, grp in state_groups.items():
        if len(grp) < 2:
            continue
        n_anchor += 1
        total_sz += len(grp)
        rtgs_g = np.array([g[2] for g in grp], dtype=np.float32)
        std_g = rtgs_g.std()
        if std_g < 1e-8:
            continue
        adv_g = (rtgs_g - rtgs_g.mean()) / (std_g + 1e-8)
        for (i, t, _), a in zip(grp, adv_g):
            A_S[i][t] = float(a)

    # ── combined ──────────────────────────────────────────────────────────────
    all_adv = []
    for i, (traj, _, _) in enumerate(group_trajs):
        for t in range(len(traj)):
            all_adv.append(float(A_E[i]) + omega * A_S[i][t])

    info = {
        "n_anchor_groups": n_anchor,
        "mean_anchor_size": float(total_sz / max(n_anchor, 1)),
        "anchor_coverage": float(n_anchor / max(len(state_groups), 1)),
    }
    return np.array(all_adv, dtype=np.float32), info


# ── training loop ──────────────────────────────────────────────────────────────


def train(cfg) -> None:
    import gymnasium as gym
    import minigrid  # noqa: F401  registers MiniGrid envs
    from . import envs as _custom_envs  # noqa: F401  registers custom envs

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    np.random.seed(getattr(cfg, "seed", 42))

    env = gym.make(cfg.env_id)

    # ── device selection ──────────────────────────────────────────────────────
    import torch as _torch
    requested = getattr(cfg, "device", "cuda")
    if requested != "cpu" and not _torch.cuda.is_available():
        tqdm.write("  [warn] CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested

    encoder = build_encoder(cfg)
    memory_cfg = getattr(cfg, 'memory', None) or {}
    if isinstance(memory_cfg, dict) and memory_cfg.get('enabled', False):
        from .encoders import VSAMemoryWrapper
        encoder = VSAMemoryWrapper(
            obs_encoder=encoder,
            dim=int(cfg.dim),
            combine=memory_cfg.get('combine', 'none'),
            decay=float(memory_cfg.get('decay', 1.0)),
            include_prev_action=bool(memory_cfg.get('include_prev_action', False)),
            num_actions=env.action_space.n,
            seed=int(getattr(cfg, 'seed', 0)),
        )
    policy = build_policy(cfg, n_actions=env.action_space.n, device=device)
    _adv_type_init = getattr(cfg, "advantage_type", "reinforce")
    value_fn = None
    if _adv_type_init == "gae":
        value_fn = SimpleValueFunction(
            obs_dim=int(getattr(cfg, "dim", 1000)),
            hidden_sizes=tuple(getattr(cfg, "value_fn_hidden_sizes", [128, 128])),
            lr=float(getattr(cfg, "value_fn_lr", 3e-4)),
            device=device,
        )
    logger = MetricsLogger()

    eta = getattr(cfg, "eta", 0.05)
    variant = getattr(cfg, "variant", "real")

    use_wandb = getattr(cfg, "use_wandb", False)
    if use_wandb:
        _init_wandb(cfg)

    dim_str = str(getattr(cfg, "dim", "N/A"))
    print(
        f"\n[OnlineHD] env={cfg.env_id}  variant={variant}  "
        f"dim={dim_str}  eta={eta}  tau={getattr(cfg, 'tau', 1.0)}  "
        f"encoder={getattr(cfg, 'encoder_type', 'bipolar')}  device={device}"
        + ("  wandb=on" if use_wandb else "") + "\n"
    )

    log_interval  = getattr(cfg, "log_interval",  50)
    save_interval = getattr(cfg, "save_interval", 500)

    # ── advantage method config ───────────────────────────────────────────────
    adv_type   = getattr(cfg, "advantage_type", "reinforce")
    group_size = max(1, getattr(cfg, "group_size", 1))   # N episodes per update
    omega      = float(getattr(cfg, "gigpo_omega", 1.0))  # step-level weight

    # Force group_size≥8 for GRPO/GiGPO if not set explicitly
    if adv_type in ("grpo", "gigpo") and group_size < 2:
        group_size = 8

    tqdm.write(f"  advantage_type={adv_type}  group_size={group_size}"
               + (f"  omega={omega}" if adv_type == "gigpo" else ""))

    tau_warmup = max(0, getattr(cfg, "tau_warmup", 0))
    if tau_warmup > 0:
        tqdm.write(f"  tau_warmup={tau_warmup} eps  ({1.0:.2f} → {getattr(cfg, 'tau', 1.0):.2f})")

    t0 = time.time()

    group_buffer: list = []    # accumulates (traj, ep_return, ep_len) tuples
    update_metrics: dict = {}
    last_advantages = np.zeros(1)

    pbar = tqdm(range(cfg.episodes), desc=f"{variant}/{adv_type}", unit="ep", dynamic_ncols=True)
    for ep in pbar:
        if tau_warmup > 0 and ep < tau_warmup:
            policy.tau = 1.0 + (getattr(cfg, "tau", 1.0) - 1.0) * (ep / tau_warmup)

        traj, ep_return, ep_len = collect_episode(env, policy, encoder, cfg, value_fn=value_fn)
        group_buffer.append((traj, ep_return, ep_len))

        # ── update when group is complete ────────────────────────────────────
        if len(group_buffer) >= group_size:
            log_probs_old = None
            ppo_eps = 0.0

            if adv_type == "gae":
                advantages = []
                for traj_g, _, _ in group_buffer:
                    rewards_g = [step[3] for step in traj_g]
                    values_g  = [step[6] for step in traj_g]
                    adv_g = compute_gae(rewards_g, values_g, cfg.gamma,
                                        getattr(cfg, "gae_lambda", 0.95))
                    advantages.extend(adv_g.tolist())
                advantages = normalize_advantages(np.array(advantages))
                adv_info = {}
                ppo_eps = float(getattr(cfg, "ppo_clip_eps", 0.0))
                if ppo_eps > 0.0:
                    log_probs_old = np.array([step[2] for traj_g, _, _ in group_buffer
                                              for step in traj_g])
                all_s_hvs_vf = np.array([step[0] for traj_g, _, _ in group_buffer
                                         for step in traj_g])
                all_returns_vf = np.concatenate([
                    compute_returns([step[3] for step in traj_g], cfg.gamma)
                    for traj_g, _, _ in group_buffer
                ])
                vf_loss = value_fn.update(all_s_hvs_vf, all_returns_vf)
                adv_info["value_fn_loss"] = vf_loss
            elif adv_type == "grpo":
                advantages, adv_info = compute_grpo_advantages(group_buffer, cfg.gamma)
            elif adv_type == "gigpo":
                advantages, adv_info = compute_gigpo_advantages(group_buffer, cfg.gamma, omega)
            elif adv_type == "reinforce_baseline":
                b = logger.moving_avg("return")
                baseline = 0.0 if np.isnan(b) else b
                advantages, adv_info = compute_reinforce_baseline_advantages(group_buffer, cfg.gamma, baseline)
            else:  # reinforce (per-episode normalization)
                advantages, adv_info = compute_reinforce_advantages(group_buffer, cfg.gamma)

            all_s_hvs   = [step[0] for traj, _, _ in group_buffer for step in traj]
            all_actions = [step[1] for traj, _, _ in group_buffer for step in traj]
            update_metrics = policy.update(all_s_hvs, all_actions, advantages, eta,
                                           log_probs_old=log_probs_old, ppo_clip_eps=ppo_eps)
            update_metrics.update(adv_info)
            last_advantages = advantages
            group_buffer.clear()

        # ── per-episode logging ───────────────────────────────────────────────
        entropies   = [step[4] for step in traj]
        counts      = policy.action_counts()
        action_hist = counts / max(counts.sum(), 1)

        metrics = {
            "return": ep_return,
            "length": ep_len,
            "mean_entropy": float(np.mean(entropies)),
            "mean_abs_adv": float(np.mean(np.abs(last_advantages))),
            **update_metrics,
        }
        avg100 = logger.moving_avg("return")
        logger.log(ep, metrics)
        std100 = logger.moving_std("return")

        pbar.set_postfix(
            ret=f"{ep_return:+.3f}",
            avg=f"{avg100:+.3f}",
            std=f"{std100:.3f}" if not np.isnan(std100) else "nan",
            len=ep_len,
        )

        if use_wandb:
            wandb.log(
                {
                    "train/return": ep_return,
                    "train/return_avg100": avg100,
                    "train/return_std100": std100,
                    "train/length": ep_len,
                    "train/mean_entropy": metrics["mean_entropy"],
                    "train/mean_abs_adv": metrics["mean_abs_adv"],
                    **{f"train/{k}": v for k, v in update_metrics.items()},
                    **{f"actions/a{i}": p for i, p in enumerate(action_hist)},
                },
                step=ep,
            )

        if (ep + 1) % log_interval == 0:
            tqdm.write(logger.summary_str(ep, metrics, moving_avg=avg100))
            if getattr(cfg, "log_action_hist", True):
                tqdm.write(
                    "  actions: " + " ".join(f"a{i}={p:.2f}" for i, p in enumerate(action_hist))
                )

        if (ep + 1) % save_interval == 0:
            ckpt = f"{cfg.checkpoint_dir}/{getattr(cfg, 'run_name', variant)}_ep{ep + 1}"
            policy.save(ckpt)
            tqdm.write(f"  [saved {ckpt}.npz]")

    elapsed = time.time() - t0
    run_name = getattr(cfg, "run_name", variant)
    metrics_filename = getattr(cfg, "metrics_filename", None)
    if metrics_filename:
        csv_path = f"{cfg.log_dir}/{metrics_filename}"
    else:
        csv_path = f"{cfg.log_dir}/{run_name}_{cfg.env_id.replace('/', '_')}_metrics.csv"
    logger.save_csv(csv_path)

    if getattr(cfg, "save_summary_json", False):
        cfg_path = f"{cfg.log_dir}/config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(vars(cfg), f, default_flow_style=False, sort_keys=False)

        summary = compute_run_summary(logger)
        summary.update({
            "method": getattr(cfg, "method_name", variant),
            "env_id": cfg.env_id,
            "encoder": getattr(cfg, "encoder_type", None),
            "seed": getattr(cfg, "seed", None),
            "dim": getattr(cfg, "dim", None),
            "tau": getattr(cfg, "tau", None),
            "eta": getattr(cfg, "eta", None),
            "sigma": getattr(cfg, "rff_sigma", None),
            "advantage": getattr(cfg, "advantage_type", None),
            "n_episodes": getattr(cfg, "episodes", None),
            "climate": None,
        })
        _mem = getattr(cfg, 'memory', None) or {}
        if isinstance(_mem, dict):
            summary.update({
                "memory_enabled": _mem.get('enabled', False),
                "memory_combine": _mem.get('combine', 'none'),
                "memory_decay": _mem.get('decay', 1.0),
                "memory_include_prev_action": _mem.get('include_prev_action', False),
            })
        summary_path = f"{cfg.log_dir}/summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        tqdm.write(f"Summary saved to {summary_path}")

    if use_wandb:
        wandb.finish()

    final_avg = logger.moving_avg("return")
    tqdm.write(f"\nDone. {cfg.episodes} episodes in {elapsed:.1f}s.")
    tqdm.write(f"Final moving avg return (100 ep): {final_avg:+.4f}")
    tqdm.write(f"Metrics saved to {csv_path}\n")

    env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> tuple[str, dict]:
    parser = argparse.ArgumentParser(description="OnlineHD policy training on MiniGrid")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-name", dest="run_name", default=None,
                        help="Override the auto-derived run name (used in CSV/W&B/checkpoint filenames)")
    parser.add_argument("--env-id", dest="env_id", default=None)
    parser.add_argument("--variant", default=None, choices=["real", "mlp_reinforce", "linear_reinforce", "random"])
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="Training device: cpu | cuda | cuda:0 (default: cuda)")
    parser.add_argument("--encoder-type", dest="encoder_type", default=None, choices=["bipolar", "fhrr", "fhrr_flat", "rff", "flat"])
    parser.add_argument("--m-max", dest="m_max", type=float, default=None)
    parser.add_argument("--p-max", dest="p_max", type=float, default=None)
    parser.add_argument("--advantage-type", dest="advantage_type", default=None,
                        choices=["reinforce", "reinforce_baseline", "grpo", "gigpo", "gae"],
                        help="Advantage estimation method (default: reinforce)")
    parser.add_argument("--group-size", dest="group_size", type=int, default=None,
                        help="Episodes per policy update for GRPO/GiGPO (default: 8 for group methods)")
    parser.add_argument("--gigpo-omega", dest="gigpo_omega", type=float, default=None,
                        help="GiGPO step-level advantage weight ω (default: 1.0)")
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None)
    parser.add_argument("--save-interval", dest="save_interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", default=None,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", dest="wandb_project", default=None)
    parser.add_argument("--save-summary-json", dest="save_summary_json", action="store_true", default=None,
                        help="Write summary.json (final return stats + hyperparams) to log_dir")
    parser.add_argument("--method-name", dest="method_name", default=None,
                        help="Override method label recorded in summary.json (e.g. fhrr_vspg, dnn)")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return args.config, overrides


def main() -> None:
    config_path, overrides = _parse_args()
    cfg = load_config(config_path, overrides)
    train(cfg)


if __name__ == "__main__":
    main()

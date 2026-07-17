"""Off-policy training loop for QHD (Hyperdimensional Q-learning).

Usage:
    python3 train_qhd.py --config qhd/configs/cartpole_qhd.yaml
    python3 train_qhd.py --config qhd/configs/cartpole_qhd.yaml --beta 0.01 --batch-size 4 --seed 0

Unlike vspg/train.py's on-policy, per-episode-batched REINFORCE
loop, QHD updates from a replay buffer after every environment step (paper
Sec 3.4). Encoders and metrics logging are reused unchanged from vspg so
CSV/summary.json have the exact same shape as every VSPG/DNN/Raw-Linear run.
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

from vspg.encoders import build_encoder
from vspg.utils import MetricsLogger, compute_run_summary

from .policy import QHDPolicy
from .replay_buffer import ReplayBuffer


def _make_run_name(cfg) -> str:
    return (
        f"beta{getattr(cfg, 'beta', 0.01)}"
        f"_bs{getattr(cfg, 'batch_size', 4)}"
        f"_buf{getattr(cfg, 'buffer_size', 50000)}"
        f"_tu{getattr(cfg, 'target_update_interval', 100)}"
        f"_s{getattr(cfg, 'seed', 42)}"
    )


def load_config(yaml_path: str, overrides: dict) -> SimpleNamespace:
    with open(yaml_path) as f:
        cfg_dict = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg_dict[k] = v
    cfg_dict.setdefault("run_name", None)
    ns = SimpleNamespace(**cfg_dict)
    if ns.run_name is None:
        ns.run_name = _make_run_name(ns)
    return ns


# ── step collection (off-policy: every step is a training sample) ────────────


def collect_step(env, policy, encoder, obs, epsilon):
    s_hv = encoder.encode(obs)
    action = policy.act(s_hv, epsilon=epsilon)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    s2_hv = encoder.encode(next_obs)
    return s_hv, action, float(reward), s2_hv, done, next_obs


# ── training loop ─────────────────────────────────────────────────────────────


def train(cfg) -> None:
    import gymnasium as gym
    import minigrid  # noqa: F401  registers MiniGrid envs

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    np.random.seed(getattr(cfg, "seed", 42))

    env = gym.make(cfg.env_id)
    n_actions = env.action_space.n

    import torch as _torch
    requested = getattr(cfg, "device", "cpu")
    if requested != "cpu" and not _torch.cuda.is_available():
        tqdm.write("  [warn] CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested

    encoder = build_encoder(cfg)
    dim = int(getattr(cfg, "dim", 10000))

    eps_decay_episodes = int(getattr(cfg, "eps_decay_episodes", max(1, int(0.5 * cfg.episodes))))
    policy = QHDPolicy(
        dim=dim,
        n_actions=n_actions,
        beta=float(getattr(cfg, "beta", 0.01)),
        gamma=float(getattr(cfg, "gamma", 0.99)),
        eps_start=float(getattr(cfg, "eps_start", 1.0)),
        eps_end=float(getattr(cfg, "eps_end", 0.05)),
        eps_decay_episodes=eps_decay_episodes,
        target_update_interval=int(getattr(cfg, "target_update_interval", 100)),
        seed=int(getattr(cfg, "seed", 42)),
        device=device,
    )

    buffer_size = int(getattr(cfg, "buffer_size", 50000))
    batch_size = int(getattr(cfg, "batch_size", 4))
    buffer = ReplayBuffer(capacity=buffer_size, dim=dim, seed=int(getattr(cfg, "seed", 42)))

    logger = MetricsLogger()
    run_name = cfg.run_name
    os.makedirs(f"{cfg.log_dir}/{run_name}", exist_ok=True)

    log_interval = int(getattr(cfg, "log_interval", 50))
    save_interval = int(getattr(cfg, "save_interval", 500))

    print(
        f"\n[QHD] env={cfg.env_id}  dim={dim}  beta={policy.beta}  gamma={policy.gamma}  "
        f"batch_size={batch_size}  buffer_size={buffer_size}  "
        f"target_update_interval={policy.target_update_interval}  "
        f"eps {policy.eps_start}->{policy.eps_end} over {eps_decay_episodes} eps  "
        f"encoder={getattr(cfg, 'encoder_type', 'rff')}  device={device}\n"
    )

    t0 = time.time()
    update_metrics: dict = {}
    ep, ep_return, ep_len, avg100, std100 = 0, 0.0, 0, float("nan"), float("nan")

    pbar = tqdm(range(cfg.episodes), desc="qhd", unit="ep", dynamic_ncols=True)
    for ep in pbar:
        obs, _ = env.reset()
        epsilon = policy.epsilon(ep)
        ep_return = 0.0
        ep_len = 0
        done = False

        while not done:
            s_hv, action, reward, s2_hv, done, next_obs = collect_step(env, policy, encoder, obs, epsilon)
            buffer.push(s_hv, action, reward, s2_hv, done)
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                update_metrics = policy.update_from_batch(*batch)
            ep_return += reward
            ep_len += 1
            obs = next_obs

        metrics = {"return": ep_return, "length": ep_len, "epsilon": epsilon, **update_metrics}
        avg100 = logger.moving_avg("return")
        logger.log(ep, metrics)
        std100 = logger.moving_std("return")

        pbar.set_postfix(
            ret=f"{ep_return:+.3f}",
            avg=f"{avg100:+.3f}",
            eps=f"{epsilon:.3f}",
            Mrow=f"{update_metrics.get('qhd_mean_M_rownorm', 0.0):.2f}",
        )

        if (ep + 1) % log_interval == 0:
            tqdm.write(
                f"ep {ep + 1:>5}  ret={ep_return:+.3f}  avg100={avg100:+.3f}  eps={epsilon:.3f}  "
                f"td_err={update_metrics.get('qhd_td_error_mean', float('nan')):.3f}  "
                f"M_rownorm={update_metrics.get('qhd_mean_M_rownorm', float('nan')):.3f}"
            )

        if (ep + 1) % save_interval == 0:
            ckpt = f"{cfg.checkpoint_dir}/{run_name}_ep{ep + 1}"
            policy.save(ckpt)
            tqdm.write(f"  [saved {ckpt}.npz]")

    elapsed = time.time() - t0
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
            "method": getattr(cfg, "method_name", "qhd"),
            "env_id": cfg.env_id,
            "encoder": getattr(cfg, "encoder_type", None),
            "seed": getattr(cfg, "seed", None),
            "dim": dim,
            "beta": getattr(cfg, "beta", None),
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "target_update_interval": getattr(cfg, "target_update_interval", None),
            "gamma": getattr(cfg, "gamma", None),
            "eps_start": getattr(cfg, "eps_start", None),
            "eps_end": getattr(cfg, "eps_end", None),
            "eps_decay_episodes": eps_decay_episodes,
            "n_episodes": getattr(cfg, "episodes", None),
            "final_mean_M_rownorm": update_metrics.get("qhd_mean_M_rownorm"),
            "final_max_M_rownorm": update_metrics.get("qhd_max_M_rownorm"),
            "climate": None,
        })
        summary_path = f"{cfg.log_dir}/summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        tqdm.write(f"Summary saved to {summary_path}")

    final_avg = logger.moving_avg("return")
    tqdm.write(f"\nDone. {cfg.episodes} episodes in {elapsed:.1f}s.")
    tqdm.write(f"Final moving avg return (100 ep): {final_avg:+.4f}")
    tqdm.write(f"Metrics saved to {csv_path}\n")

    env.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> tuple[str, dict]:
    parser = argparse.ArgumentParser(description="QHD (off-policy hyperdimensional Q-learning) training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--env-id", dest="env_id", default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None, help="TD regression step size (paper's beta)")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=None)
    parser.add_argument("--target-update-interval", dest="target_update_interval", type=int, default=None,
                        help="Steps between Double-Q target syncs")
    parser.add_argument("--eps-start", dest="eps_start", type=float, default=None)
    parser.add_argument("--eps-end", dest="eps_end", type=float, default=None)
    parser.add_argument("--eps-decay-episodes", dest="eps_decay_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 (default: cpu)")
    parser.add_argument("--encoder-type", dest="encoder_type", default=None,
                        choices=["bipolar", "fhrr", "fhrr_flat", "rff", "flat", "gaussian", "flat_norm"])
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None)
    parser.add_argument("--save-interval", dest="save_interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--save-summary-json", dest="save_summary_json", action="store_true", default=None,
                        help="Write summary.json (final return stats + hyperparams) to log_dir")
    parser.add_argument("--method-name", dest="method_name", default=None,
                        help="Override method label recorded in summary.json (default: qhd)")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return args.config, overrides


def main() -> None:
    config_path, overrides = _parse_args()
    cfg = load_config(config_path, overrides)
    train(cfg)


if __name__ == "__main__":
    main()

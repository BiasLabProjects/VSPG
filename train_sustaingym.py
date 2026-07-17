"""MA-VSPG training on SustainGym BuildingEnv.

Each agent has a decentralized VSPG/HDC actor (explicit C-matrix update).
A centralized neural critic estimates V(global_state) for advantage computation.
The critic is used only during training; inference uses actors only.

Usage (inside Docker):
    python train_sustaingym_vspg.py --config configs/sustaingym_building_ma_vspg.yaml
    python train_sustaingym_vspg.py --config configs/sustaingym_building_ma_vspg.yaml \
        --discrete-bins 10 --episodes 500 --no-wandb
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from tqdm import tqdm

from vspg.train import load_config
from vspg.utils import compute_returns, normalize_advantages, MetricsLogger
from envs.sustaingym_wrapper import MultiAgentBuildingWrapper
from policies.ma_vspg_policy import MultiAgentVSPGPolicy
from critics.centralized_critic import CentralizedCritic


# ── GAE ───────────────────────────────────────────────────────────────────────


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Generalized Advantage Estimation (Schulman et al., 2015).

    Returns advantages of shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0  # episode ends → V(s_T+1) = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages


# ── Episode collection ────────────────────────────────────────────────────────


def collect_episode(
    wrapper: MultiAgentBuildingWrapper,
    policy: MultiAgentVSPGPolicy,
    critic: CentralizedCritic | None,
    cfg: SimpleNamespace,
    seed: int | None = None,
) -> tuple[list[list[tuple]], list[float], list[np.ndarray], float, int]:
    """Run one episode.

    Returns:
        traj_per_agent: list[list[(z_i, a_i, advantage_t, log_prob_old_i)]]  — filled after GAE
        rewards: raw reward per step
        global_states: global state per step
        ep_return: sum of rewards
        ep_len: episode length
    """
    gamma = float(getattr(cfg, "gamma", 0.99))
    lam = float(getattr(cfg, "gae_lambda", 0.95))
    use_gae = bool(getattr(cfg, "use_gae", True))

    obs_list, _ = wrapper.reset(seed=seed)
    done = False

    # Per-step buffers
    step_obs: list[list] = []                  # obs_list at each step
    step_z: list[list[np.ndarray]] = []        # encoded hvs at each step
    step_actions: list[list[int]] = []
    step_log_probs: list[list[float]] = []     # log π(a_i|z_i) at act time
    step_rewards: list[float] = []
    step_global: list[np.ndarray] = []
    step_values: list[float] = []

    while not done:
        global_s = wrapper.get_global_state()
        step_global.append(global_s)

        # Critic value estimate
        v = critic.predict(global_s) if critic is not None else 0.0
        step_values.append(v)

        # Encode observations
        z_list = policy.encode_all(obs_list)
        step_z.append(z_list)
        step_obs.append(obs_list)

        # Act — capture log_probs for PPO-clip ratio
        actions, log_probs, _ = policy.act(z_list, encoded=True)
        step_actions.append(actions)
        step_log_probs.append(log_probs)

        obs_list, reward, done, _ = wrapper.step(actions)
        step_rewards.append(reward)

    ep_return = float(np.sum(step_rewards))
    ep_len = len(step_rewards)

    # Compute advantages
    if use_gae and critic is not None:
        advantages = compute_gae(step_rewards, step_values, gamma, lam)
    else:
        returns = compute_returns(step_rewards, gamma)
        advantages = normalize_advantages(returns)

    if bool(getattr(cfg, "advantage_normalize", True)):
        std = advantages.std()
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    # Build per-agent trajectories: (z_i, a_i, advantage_t, log_prob_old_i)
    n_agents = wrapper.num_agents
    traj_per_agent: list[list[tuple]] = [[] for _ in range(n_agents)]
    for t in range(ep_len):
        for i in range(n_agents):
            traj_per_agent[i].append((
                step_z[t][i],
                step_actions[t][i],
                float(advantages[t]),
                float(step_log_probs[t][i]),
            ))

    return traj_per_agent, step_rewards, step_global, ep_return, ep_len


# ── Training loop ─────────────────────────────────────────────────────────────


def train(cfg: SimpleNamespace) -> None:
    import torch as _torch

    # ── Device ───────────────────────────────────────────────────────────
    requested = getattr(cfg, "device", "cpu")
    if requested != "cpu" and not _torch.cuda.is_available():
        tqdm.write("  [warn] CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = requested

    # ── Output directories ────────────────────────────────────────────────
    ckpt_dir = getattr(cfg, "checkpoint_dir", "checkpoints_sustaingym")
    log_dir = getattr(cfg, "log_dir", "logs_sustaingym")
    res_dir = getattr(cfg, "result_dir", "results_sustaingym")
    run_name = getattr(cfg, "run_name", "ma_vspg")
    os.makedirs(f"{ckpt_dir}/{run_name}", exist_ok=True)
    os.makedirs(f"{log_dir}/{run_name}", exist_ok=True)
    os.makedirs(f"{res_dir}/{run_name}", exist_ok=True)

    # ── Environment ───────────────────────────────────────────────────────
    bins_cfg = getattr(cfg, "discrete_action_bins", None)
    disc_bins = getattr(cfg, "discrete_bins", None)

    wrapper = MultiAgentBuildingWrapper(
        building=getattr(cfg, "sustaingym_building_type", "OfficeSmall"),
        weather=getattr(cfg, "sustaingym_weather", "Hot_Dry"),
        location=getattr(cfg, "sustaingym_location", "Tucson"),
        bins=bins_cfg if isinstance(bins_cfg, list) else None,
        discrete_bins=int(disc_bins) if disc_bins is not None else None,
        episode_len=int(getattr(cfg, "episode_len", 288)),
        obs_noise_std=float(getattr(cfg, "obs_noise_std", 0.0)),
    )

    num_agents = wrapper.num_agents
    action_dim = wrapper.action_dim
    global_state_dim = wrapper.get_global_state().shape[0]

    tqdm.write(
        f"\n[MA-VSPG] building={getattr(cfg, 'sustaingym_building_type', 'OfficeSmall')}"
        f"  num_agents={num_agents}  action_dim={action_dim}"
        f"  global_state_dim={global_state_dim}  device={device}\n"
    )

    # Save config
    cfg_path = f"{log_dir}/{run_name}/config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(vars(cfg), f, default_flow_style=False, sort_keys=False)

    # ── Policy & Critic ───────────────────────────────────────────────────
    share_actor = bool(getattr(cfg, "share_actor", False))
    policy = MultiAgentVSPGPolicy(
        cfg=cfg,
        num_agents=num_agents,
        action_dim=action_dim,
        device=device,
        share_actor=share_actor,
    )

    use_critic = bool(getattr(cfg, "use_centralized_critic", True))
    critic: CentralizedCritic | None = None
    if use_critic:
        critic = CentralizedCritic(
            state_dim=global_state_dim,
            hidden_sizes=tuple(getattr(cfg, "critic_hidden_sizes", [128, 128])),
            lr=float(getattr(cfg, "critic_lr", 3e-4)),
            device=device,
            return_normalize=bool(getattr(cfg, "return_normalize", True)),
            grad_clip=float(getattr(cfg, "critic_grad_clip", 10.0)),
        )

    logger = MetricsLogger()
    eta = float(getattr(cfg, "eta", 0.05))
    episodes = int(getattr(cfg, "episodes", 500))
    log_interval = int(getattr(cfg, "log_interval", 50))
    save_interval = int(getattr(cfg, "save_interval", 100))

    # ── W&B ────────────────────────────────────────────────────────────────
    use_wandb = bool(getattr(cfg, "use_wandb", False))
    if use_wandb:
        import wandb
        wandb_dir = getattr(cfg, "wandb_dir", "wandb_runs")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project=getattr(cfg, "wandb_project", "sustaingym-ma-vspg"),
            name=run_name,
            config=vars(cfg),
            dir=wandb_dir,
            reinit="finish_previous",
        )

    t0 = time.time()
    pbar = tqdm(range(episodes), desc="MA-VSPG", unit="ep", dynamic_ncols=True)

    for ep in pbar:
        seed = int(getattr(cfg, "seed", 42)) + ep

        traj_per_agent, rewards, global_states, ep_return, ep_len = collect_episode(
            wrapper, policy, critic, cfg, seed=seed
        )

        # ── Actor update ──────────────────────────────────────────────────
        actor_metrics = policy.update_with_advantages(traj_per_agent, eta)

        # ── Critic update ─────────────────────────────────────────────────
        critic_loss = 0.0
        if critic is not None and global_states:
            returns_arr = compute_returns(rewards, float(getattr(cfg, "gamma", 0.99)))
            states_arr = np.stack(global_states)
            critic_loss = critic.update(states_arr, returns_arr)

        # ── Logging ───────────────────────────────────────────────────────
        avg_r = ep_return / max(ep_len, 1)
        metrics = {
            "return": ep_return,
            "avg_reward_per_step": avg_r,
            "length": ep_len,
            "critic_loss": critic_loss,
            **actor_metrics,
        }
        logger.log(ep, metrics)
        avg100 = logger.moving_avg("avg_reward_per_step")

        pbar.set_postfix(
            avg_r=f"{avg_r:+.4f}",
            avg100=f"{avg100:+.4f}",
            v_loss=f"{critic_loss:.4f}",
        )

        if use_wandb:
            import wandb
            wandb.log(
                {
                    "train/return": ep_return,
                    "train/avg_reward_per_step": avg_r,
                    "train/avg_reward_per_step_avg100": avg100,
                    "train/length": ep_len,
                    "train/critic_loss": critic_loss,
                    **{f"train/{k}": v for k, v in actor_metrics.items()},
                },
                step=ep,
            )

        if (ep + 1) % log_interval == 0:
            tqdm.write(logger.summary_str(ep, metrics, moving_avg=avg100))

        if (ep + 1) % save_interval == 0:
            actor_path = f"{ckpt_dir}/{run_name}/actor_ep{ep + 1}"
            policy.save_actor(actor_path)
            tqdm.write(f"  [saved actor] {actor_path}")
            if critic is not None:
                critic_path = f"{ckpt_dir}/{run_name}/critic_ep{ep + 1}.pt"
                critic.save(critic_path)
                tqdm.write(f"  [saved critic] {critic_path}")

    # ── Final save ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    actor_path = f"{ckpt_dir}/{run_name}/actor_final"
    policy.save_actor(actor_path)
    if critic is not None:
        critic.save(f"{ckpt_dir}/{run_name}/critic_final.pt")

    csv_path = f"{log_dir}/{run_name}/metrics.csv"
    logger.save_csv(csv_path)

    summary = {
        "run_name": run_name,
        "episodes": episodes,
        "elapsed_s": round(elapsed, 1),
        "final_avg100_return": float(logger.moving_avg("return")),
        "final_avg100_avg_reward_per_step": float(logger.moving_avg("avg_reward_per_step")),
        "num_agents": num_agents,
        "action_dim": action_dim,
        "bins": wrapper.bins.tolist(),
    }
    with open(f"{res_dir}/{run_name}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    tqdm.write(f"\nDone. {episodes} episodes in {elapsed:.1f}s.")
    tqdm.write(f"Metrics: {csv_path}")
    tqdm.write(f"Actor:   {actor_path}")

    wrapper.close()
    if use_wandb:
        import wandb
        wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> tuple[str, dict]:
    parser = argparse.ArgumentParser(description="MA-VSPG training on SustainGym BuildingEnv")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--run-name", dest="run_name", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--encoder-type", dest="encoder_type", default=None)
    parser.add_argument("--rff-sigma", dest="rff_sigma", type=float, default=None,
                        help="RFF bandwidth σ (only used when encoder_type=rff)")
    parser.add_argument("--discrete-bins", dest="discrete_bins", type=int, default=None,
                        help="Number of uniform bins over [-1, 1] per agent (overrides discrete_action_bins list)")
    parser.add_argument("--ppo-clip-eps", dest="ppo_clip_eps", type=float, default=None,
                        help="PPO clip epsilon for VSPG actor (0.0 = disabled)")
    parser.add_argument("--critic-grad-clip", dest="critic_grad_clip", type=float, default=None,
                        help="Max gradient norm for centralized critic (0.0 = disabled)")
    parser.add_argument("--no-return-norm", dest="return_normalize", action="store_false", default=None,
                        help="Disable Welford return normalisation in critic")
    parser.add_argument("--obs-noise-std", dest="obs_noise_std", type=float, default=None,
                        help="Gaussian noise std added to observations (default: 0.0)")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", default=None)
    parser.add_argument("--wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None)
    parser.add_argument("--save-interval", dest="save_interval", type=int, default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--log-dir", dest="log_dir", default=None)
    parser.add_argument("--result-dir", dest="result_dir", default=None)
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    return args.config, overrides


def main() -> None:
    config_path, overrides = _parse_args()
    cfg = load_config(config_path, overrides)
    train(cfg)


if __name__ == "__main__":
    main()

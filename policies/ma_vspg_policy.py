"""Multi-agent VSPG policy.

Each agent has its own RealHDPolicy (C_i matrix) and its own encoder.
The actor update is the same explicit vector update as the single-agent case
— no backprop, no PyTorch optimizer on the actor.

Centralized advantage (from the critic) is passed in at update time and shared
across all agents as the team advantage signal.

Set cfg.policy_type = "linear_reinforce" to swap in LinearReinforceActor
(gradient-based Adam policy) without changing the training loop.

Usage::

    policy = MultiAgentVSPGPolicy(cfg, num_agents=4, action_dim=5, device="cpu")
    actions = policy.act(obs_list)                         # list[int]
    policy.update_with_advantages(obs_list, actions, advs) # ΔC_i per agent
    policy.save_actor("checkpoints_sustaingym/run/actor_ep100")
    policy.load_actor("checkpoints_sustaingym/run/actor_ep100")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Reuse existing single-agent building blocks
from vspg.policies import RealHDPolicy, LinearReinforceActor, MLPReinforcePolicy
from vspg.encoders import build_encoder


class MultiAgentVSPGPolicy:
    """VSPG actor for N discrete-action agents.

    Args:
        cfg: config namespace (same keys as single-agent: dim, tau, seed,
             encoder_type, fhrr_w, rff_sigma, row_normalize, …)
        num_agents: number of agents
        action_dim: number of discrete actions per agent (same for all)
        device: "cpu" | "cuda" | …
        share_actor: if True, all agents share one C matrix and one encoder
    """

    def __init__(
        self,
        cfg: Any,
        num_agents: int,
        action_dim: int,
        device: str = "cpu",
        share_actor: bool = False,
    ) -> None:
        self.cfg = cfg
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.device = device
        self.share_actor = share_actor

        dim           = int(getattr(cfg, "dim", 1000))
        tau           = float(getattr(cfg, "tau", 1.0))
        seed          = int(getattr(cfg, "seed", 42))
        fhrr_init     = bool(getattr(cfg, "fhrr_init", False))
        row_normalize = bool(getattr(cfg, "row_normalize", True))
        self.policy_type = getattr(cfg, "policy_type", "vspg")
        self._use_bipolar = getattr(cfg, "variant", "real") == "onebit"

        def _make_actor(agent_seed: int):
            if self.policy_type == "linear_reinforce":
                obs_dim   = int(getattr(cfg, "obs_flat_dim", dim))
                linear_lr = float(getattr(cfg, "linear_lr", 1e-3))
                return LinearReinforceActor(
                    obs_dim=obs_dim,
                    n_actions=action_dim,
                    tau=tau,
                    lr=linear_lr,
                    seed=agent_seed,
                    device=device,
                )
            if self.policy_type == "mlp_reinforce":
                obs_dim   = int(getattr(cfg, "obs_flat_dim", dim))
                mlp_hidden = tuple(getattr(cfg, "mlp_hidden", [128, 64]))
                mlp_lr     = float(getattr(cfg, "mlp_lr", 3e-4))
                return MLPReinforcePolicy(
                    obs_dim=obs_dim,
                    n_actions=action_dim,
                    hidden=mlp_hidden,
                    lr=mlp_lr,
                    seed=agent_seed,
                    device=device,
                )
            return RealHDPolicy(
                dim=dim,
                n_actions=action_dim,
                tau=tau,
                seed=agent_seed,
                device=device,
                fhrr_init=fhrr_init,
                row_normalize=row_normalize,
            )

        if share_actor:
            self._encoders = [build_encoder(cfg)]
            self._actors   = [_make_actor(seed)]
        else:
            self._encoders = [build_encoder(cfg) for _ in range(num_agents)]
            self._actors   = [_make_actor(seed + i) for i in range(num_agents)]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _actor(self, i: int) -> RealHDPolicy:
        return self._actors[0] if self.share_actor else self._actors[i]

    def _encoder(self, i: int):
        return self._encoders[0] if self.share_actor else self._encoders[i]

    def _encode(self, i: int, obs) -> np.ndarray:
        enc = self._encoder(i)
        if self._use_bipolar and hasattr(enc, "encode_bipolar"):
            return enc.encode_bipolar(obs)
        if hasattr(enc, "encode"):
            return enc.encode(obs)
        return obs  # fallback (should not happen with standard encoders)

    # ── Public interface ───────────────────────────────────────────────────

    def encode_all(self, obs_list: list) -> list[np.ndarray]:
        """Encode observations for all agents → list of hypervectors."""
        return [self._encode(i, obs) for i, obs in enumerate(obs_list)]

    def act(
        self,
        obs_list: list,
        deterministic: bool = False,
        encoded: bool = False,
    ) -> tuple[list[int], list[float], list[float]]:
        """Select actions for all agents.

        Args:
            obs_list: raw observations (one per agent)
            deterministic: greedy argmax if True, sample otherwise
            encoded: if True, obs_list already contains hypervectors (skip encode)

        Returns:
            (actions, log_probs, entropies)  — each a list of length num_agents
        """
        actions, log_probs, entropies = [], [], []
        for i, obs in enumerate(obs_list):
            z = obs if encoded else self._encode(i, obs)
            a, lp, ent = self._actor(i).act(z, greedy=deterministic)
            actions.append(a)
            log_probs.append(lp)
            entropies.append(ent)
        return actions, log_probs, entropies

    def get_log_probs(
        self, obs_list: list, actions: list[int], encoded: bool = False
    ) -> list[float]:
        """Compute log π_i(a_i | o_i) for each agent."""
        log_probs = []
        for i, (obs, a) in enumerate(zip(obs_list, actions)):
            z = obs if encoded else self._encode(i, obs)
            log_probs.append(self._actor(i).get_log_prob(z, a))
        return log_probs

    def update_with_advantages(
        self,
        traj_per_agent: list[list[tuple]],
        eta: float,
    ) -> dict:
        """Update each agent's C_i matrix with shared team advantage.

        Args:
            traj_per_agent: list of agent trajectories.
                traj_per_agent[i] = [(z_i_t, a_i_t, advantage_t[, log_prob_old_t]), ...]
                The 4th element (log_prob_old) is optional; when present and
                cfg.ppo_clip_eps > 0, PPO-style importance-ratio clipping is applied.
            eta: OnlineHD learning rate

        Returns:
            dict of scalar metrics per agent (mean_rho, mean_abs_adv)
        """
        ppo_clip_eps = float(getattr(self.cfg, "ppo_clip_eps", 0.0))
        all_metrics: dict = {}
        for i, traj in enumerate(traj_per_agent):
            if not traj:
                continue
            z_hvs = [t[0] for t in traj]
            acts  = [t[1] for t in traj]
            advs  = np.array([t[2] for t in traj], dtype=np.float32)
            log_probs_old = (
                np.array([t[3] for t in traj], dtype=np.float32)
                if len(traj[0]) > 3 else None
            )
            m = self._actor(i).update(
                z_hvs, acts, advs, eta,
                log_probs_old=log_probs_old,
                ppo_clip_eps=ppo_clip_eps,
            )
            all_metrics[f"agent{i}/mean_rho"] = m.get("mean_rho", 0.0)
            all_metrics[f"agent{i}/mean_abs_adv"] = m.get("mean_abs_adv", 0.0)
        return all_metrics

    # ── Save / load ────────────────────────────────────────────────────────

    def save_actor(self, path: str) -> None:
        """Save all actor C matrices.  Critic is NOT saved here."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self.share_actor:
            self._actors[0].save(str(p) + "_shared")
        else:
            for i, actor in enumerate(self._actors):
                actor.save(str(p) + f"_agent{i}")

    def load_actor(self, path: str) -> None:
        """Load actor weights (critic NOT required)."""
        p = Path(path)
        if self.share_actor:
            self._actors[0].load(str(p) + "_shared")
        else:
            for i, actor in enumerate(self._actors):
                actor.load(str(p) + f"_agent{i}")

    def action_counts(self) -> list[np.ndarray]:
        return [self._actor(i).action_counts() for i in range(self.num_agents)]

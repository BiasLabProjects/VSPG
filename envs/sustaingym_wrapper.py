"""Multi-agent building environment wrapper for MA-VSPG.

Wraps MultiAgentBuildingEnv (PettingZoo ParallelEnv) and exposes a uniform
interface shared with MinigridWrapper so the training loop stays env-agnostic.

All SustainGym imports are lazy (inside __init__) so MiniGrid-only experiments
never require SustainGym to be installed.

Discrete action resolution (priority order):
  1. bins argument (list of floats) — use exactly those values
  2. discrete_bins argument (int N) — np.linspace(-1, 1, N)
  3. Default: 5-bin list [-1.0, -0.5, 0.0, 0.5, 1.0]
"""
from __future__ import annotations

from typing import Any

import numpy as np


_IMPORT_ERROR_MSG = """
SustainGym is not installed in the current environment.

SustainGym is required only for SustainGym experiments and must be run inside
the dedicated Docker container:

  bash scripts/build_sustaingym_docker.sh   # build image (once)
  bash scripts/run_sustaingym_docker.sh     # open shell in container

See docs/setup_sustaingym_docker.md for full setup instructions.
"""

_DEFAULT_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]


class MultiAgentBuildingWrapper:
    """Thin wrapper around MultiAgentBuildingEnv for MA-VSPG training.

    Args:
        building: building name (e.g. "OfficeSmall") — see
            sustaingym.envs.building.utils.BUILDINGS for valid names
        weather: weather scenario name (e.g. "Hot_Dry") — see WEATHER dict
        location: location name for ground temps (e.g. "Tucson") — see GROUND_TEMP
        bins: explicit list of continuous values to map discrete action indices to.
            Overrides discrete_bins if provided.
        discrete_bins: number of uniform bins over [-1, 1].  Used when bins is None.
        episode_len: steps per episode (default: 288 = 1 day at 5-min resolution)
        **param_kwargs: forwarded to ParameterGenerator (e.g. reward_beta, target)
    """

    def __init__(
        self,
        building: str = "OfficeSmall",
        weather: str = "Hot_Dry",
        location: str = "Tucson",
        bins: list[float] | None = None,
        discrete_bins: int | None = None,
        episode_len: int = 288,
        obs_noise_std: float = 0.0,
        noise_seed: int = 0,
        **param_kwargs: Any,
    ) -> None:
        try:
            from sustaingym.envs.building.env import BuildingEnv  # noqa: F401
            from sustaingym.envs.building.multiagent_env import MultiAgentBuildingEnv
            from sustaingym.envs.building.utils import ParameterGenerator
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MSG) from exc

        # Resolve discrete bins
        if bins is not None:
            self.bins = np.array(bins, dtype=np.float32)
        elif discrete_bins is not None:
            self.bins = np.linspace(-1.0, 1.0, int(discrete_bins), dtype=np.float32)
        else:
            self.bins = np.array(_DEFAULT_BINS, dtype=np.float32)

        params = ParameterGenerator(
            building=building,
            weather=weather,
            location=location,
            is_continuous_action=True,  # wrapper handles discretization
            episode_len=episode_len,
            **param_kwargs,
        )

        self._env = MultiAgentBuildingEnv(params)
        self._agents = self._env.possible_agents  # list[int] — zone indices with AC

        # Cache shapes for callers
        self._num_agents: int = len(self._agents)
        self._obs_dim: int = int(self._env.single_env.observation_space.shape[0])
        self._episode_len: int = episode_len

        # Observation noise
        self._obs_noise_std: float = obs_noise_std
        self._noise_rng = np.random.default_rng(noise_seed)

        # Internal state (set on reset)
        self._last_obs: dict[int, np.ndarray] = {}

    # ── Uniform interface ──────────────────────────────────────────────────

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def action_dim(self) -> int:
        return len(self.bins)

    @property
    def episode_len(self) -> int:
        return self._episode_len

    def reset(self, seed: int | None = None) -> tuple[list[np.ndarray], dict]:
        obs_dict, info_dict = self._env.reset(seed=seed)
        self._last_obs = obs_dict
        # Restore agents list (PettingZoo clears it on episode end)
        self._env.agents = self._env.possible_agents[:]
        obs = self._obs_list(obs_dict)
        if self._obs_noise_std > 0.0:
            obs = [o + self._noise_rng.normal(0, self._obs_noise_std, o.shape).astype(np.float32)
                   for o in obs]
        return obs, info_dict

    def step(
        self, actions: list[int]
    ) -> tuple[list[np.ndarray], float, bool, dict]:
        """Step environment.

        Args:
            actions: list of discrete action indices (one per agent)

        Returns:
            obs_list: per-agent observations
            reward: shared team scalar reward
            done: episode ended
            info: info dict from first agent
        """
        assert len(actions) == self._num_agents, (
            f"Expected {self._num_agents} actions, got {len(actions)}"
        )

        # Map discrete index → continuous value for each agent
        action_dict: dict[int, np.ndarray] = {}
        for i, agent_id in enumerate(self._agents):
            cont_val = float(self.bins[int(actions[i])])
            action_dict[i] = np.array([cont_val], dtype=np.float32)

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self._env.step(action_dict)
        self._last_obs = obs_dict if obs_dict else self._last_obs

        # Shared reward: average over agents (all receive same value from env)
        reward = float(np.mean(list(rew_dict.values()))) if rew_dict else 0.0
        done = any(term_dict.values()) or any(trunc_dict.values())
        info = next(iter(info_dict.values())) if info_dict else {}

        obs = self._obs_list(obs_dict if obs_dict else self._last_obs)
        if self._obs_noise_std > 0.0:
            obs = [o + self._noise_rng.normal(0, self._obs_noise_std, o.shape).astype(np.float32)
                   for o in obs]
        return obs, reward, done, info

    def get_global_state(self) -> np.ndarray:
        """Return global state (full observation vector from underlying env)."""
        if hasattr(self._env, "state"):
            try:
                return np.asarray(self._env.state(), dtype=np.float32)
            except AttributeError:
                pass  # env not yet reset; _state not populated
        # Fallback: use last known obs of agent 0 (same obs broadcast to all)
        if self._last_obs:
            return np.asarray(next(iter(self._last_obs.values())), dtype=np.float32)
        return np.zeros(self._obs_dim, dtype=np.float32)

    def get_agent_obs(self) -> list[np.ndarray]:
        """Return per-agent observation list (same as last step/reset output)."""
        return self._obs_list(self._last_obs)

    def close(self) -> None:
        self._env.close()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _obs_list(self, obs_dict: dict[int, np.ndarray]) -> list[np.ndarray]:
        """Convert PettingZoo obs dict → ordered list aligned with self._agents."""
        out = []
        for agent_id in self._agents:
            if agent_id in obs_dict:
                out.append(np.asarray(obs_dict[agent_id], dtype=np.float32))
            elif self._last_obs and agent_id in self._last_obs:
                out.append(np.asarray(self._last_obs[agent_id], dtype=np.float32))
            else:
                out.append(np.zeros(self._obs_dim, dtype=np.float32))
        return out

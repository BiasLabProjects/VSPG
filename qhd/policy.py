"""QHD: off-policy, value-based Hyperdimensional Q-learning.

Reimplementation of Ni et al., "Efficient Off-Policy Reinforcement Learning
via Brain-Inspired Computing" (GLSVLSI'23). Adapted to this codebase's
real-valued, L2-normalised HD vector convention (the same convention
RealHDPolicy already uses — our encoders store FHRR's re/im parts
interleaved in one real vector, so a plain real dot product already plays
the role of the paper's real(M . S-conjugate / D)).

Q(s, a)   = dot(M[a], s_hv)                              (paper Eq. 2)
q_true    = r + gamma * (1-done) * max_a' Q'(s', a')      (paper Eq. 3, Double Q-learning)
M[a]     += beta * (q_true - q_pred) * s_hv               (paper Eq. 4, vectorised over a batch)

M is initialised to all-zero (paper Sec 3.3) and never normalised — unlike
RealHDPolicy.C, which is row-normalised after every update. That is a
deliberate, faithful-to-the-paper choice: M is a linear function
approximator over a fixed feature map, trained against a bootstrapped,
off-policy target (max_a' Q'(s',a')) — the classic "deadly triad"
(Tsitsiklis & Van Roy, 1997) that can drift unboundedly, with no built-in
bound the way VSPG's cosine-similarity logits have. update_from_batch()
reports mean ||M_a|| every step specifically so that drift is visible as a
diagnostic curve rather than a silent failure mode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class QHDPolicy:
    def __init__(
        self,
        dim: int,
        n_actions: int,
        beta: float = 0.01,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 1000,
        target_update_interval: int = 100,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.dim = dim
        self.n_actions = n_actions
        self.beta = beta
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = max(1, eps_decay_episodes)
        self.target_update_interval = target_update_interval
        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_actions, dtype=np.int64)
        self._step_count = 0

        # M initialised to all-zero, exactly as in the paper (Sec 3.3) — unlike
        # RealHDPolicy.C's random unit-norm init.
        self.M = torch.zeros((n_actions, dim), dtype=torch.float32, device=self.device)
        self.M_target = self.M.clone()

    # ── action selection ──────────────────────────────────────────────────

    def epsilon(self, episode: int) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_episodes."""
        frac = min(1.0, episode / self.eps_decay_episodes)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _s(self, s_hv: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(s_hv, dtype=torch.float32, device=self.device)

    def q_values(self, s_hv: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return (self.M @ self._s(s_hv)).cpu().numpy()

    def act(self, s_hv: np.ndarray, epsilon: float = 0.0, greedy: bool = False) -> int:
        """Eps-greedy action selection (paper Eq. 1). greedy=True forces eps=0 (eval)."""
        if not greedy and self._rng.random() < epsilon:
            action = int(self._rng.integers(self.n_actions))
        else:
            action = int(np.argmax(self.q_values(s_hv)))
        self._counts[action] += 1
        return action

    # ── target network (Double Q-learning, paper Sec 3.4) ────────────────

    def sync_target(self) -> None:
        self.M_target = self.M.clone()

    def maybe_sync_target(self) -> bool:
        self._step_count += 1
        if self._step_count % self.target_update_interval == 0:
            self.sync_target()
            return True
        return False

    # ── TD regression update (paper Eq. 3-4, vectorised over a minibatch) ─

    def update_from_batch(
        self, s: np.ndarray, a: np.ndarray, r: np.ndarray, s2: np.ndarray, done: np.ndarray
    ) -> dict:
        with torch.no_grad():
            S = torch.as_tensor(s, dtype=torch.float32, device=self.device)        # (B, D)
            A = torch.as_tensor(a, dtype=torch.long, device=self.device)           # (B,)
            R = torch.as_tensor(r, dtype=torch.float32, device=self.device)        # (B,)
            S2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device)      # (B, D)
            Done = torch.as_tensor(done, dtype=torch.float32, device=self.device)  # (B,)

            # q_true = r + gamma * (1-done) * max_a' Q'(s', a')   (Double Q-learning target)
            q2_target = S2 @ self.M_target.T                       # (B, n_actions)
            max_q2 = q2_target.max(dim=1).values                   # (B,)
            q_true = R + self.gamma * (1.0 - Done) * max_q2        # (B,)

            # q_pred = dot(M[a_i], s_i)
            q_pred = (S * self.M[A]).sum(dim=1)                    # (B,)
            error = q_true - q_pred                                 # (B,)

            # M[a] += beta * sum_{i: a_i=a} error_i * s_i   — scatter-add, no normalisation
            grad = torch.zeros_like(self.M)
            grad.index_add_(0, A, error[:, None] * S)
            self.M.add_(self.beta * grad)

            target_synced = self.maybe_sync_target()

            return {
                "qhd_td_error_mean": float(error.abs().mean().item()),
                "qhd_q_pred_mean": float(q_pred.mean().item()),
                "qhd_mean_M_rownorm": float(self.M.norm(dim=1).mean().item()),
                "qhd_max_M_rownorm": float(self.M.norm(dim=1).max().item()),
                "qhd_target_synced": float(target_synced),
            }

    def action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            M=self.M.cpu().numpy(),
            M_target=self.M_target.cpu().numpy(),
            counts=self._counts,
        )

    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self.M = torch.from_numpy(data["M"]).to(self.device)
        self.M_target = torch.from_numpy(data["M_target"]).to(self.device)
        self._counts = data["counts"]

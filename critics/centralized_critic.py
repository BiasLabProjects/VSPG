"""Centralized neural critic for MA-VSPG.

Input:  global state (concatenated observations or environment state vector)
Output: scalar V(s) estimate

The critic is used ONLY during training for advantage estimation.
It is NOT required at inference/evaluation time.
Actor checkpoints are saved separately and can be loaded without the critic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class RunningNormalizer:
    """Welford online mean/variance for normalizing return targets."""

    def __init__(self) -> None:
        self.count: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0

    def update(self, x: np.ndarray) -> None:
        for val in x.flat:
            self.count += 1
            delta = float(val) - self._mean
            self._mean += delta / self.count
            self._M2 += delta * (float(val) - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(max(self._M2 / (self.count - 1), 0.0)))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (self.std + 1e-8)).astype(np.float32)

    def denormalize(self, x: float) -> float:
        return float(x * (self.std + 1e-8) + self.mean)

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self._mean, "M2": self._M2}

    def load_state_dict(self, d: dict) -> None:
        self.count = int(d["count"])
        self._mean = float(d["mean"])
        self._M2 = float(d["M2"])


class CentralizedCritic:
    """MLP value function V(global_state) → scalar.

    Args:
        state_dim: dimensionality of the global state vector
        hidden_sizes: hidden layer widths (default: [128, 128])
        lr: Adam learning rate
        device: "cpu" | "cuda" | …
        return_normalize: if True, normalise return targets via Welford stats
        grad_clip: max gradient norm (nn.utils.clip_grad_norm_); 0 = disabled
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 3e-4,
        device: str = "cpu",
        return_normalize: bool = True,
        grad_clip: float = 10.0,
    ) -> None:
        self.device = torch.device(device)
        self.return_normalize = return_normalize
        self.grad_clip = grad_clip

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()
        self._normalizer = RunningNormalizer()

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, state: np.ndarray) -> float:
        """Predict V(state) for a single state vector (denormalised)."""
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw = float(self.net(s).squeeze().item())
        if self.return_normalize:
            return self._normalizer.denormalize(raw)
        return raw

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """Predict V for a batch of states, shape (T, state_dim) → (T,)."""
        with torch.no_grad():
            s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            raw = self.net(s).squeeze(1).cpu().numpy()
        if self.return_normalize:
            return np.array([self._normalizer.denormalize(float(v)) for v in raw], dtype=np.float32)
        return raw

    # ── Training ───────────────────────────────────────────────────────────

    def update(self, states: np.ndarray, returns: np.ndarray) -> float:
        """One gradient step: MSE(V(s), normalised_G_t).

        Args:
            states: shape (T, state_dim)
            returns: shape (T,) — discounted returns or GAE targets

        Returns:
            scalar value loss (in normalised space when return_normalize=True)
        """
        if self.return_normalize:
            self._normalizer.update(returns)
            targets = self._normalizer.normalize(returns)
        else:
            targets = returns.astype(np.float32)

        S = torch.as_tensor(states,  dtype=torch.float32, device=self.device)
        G = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        V_pred = self.net(S).squeeze(1)
        loss = self._loss_fn(V_pred, G)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    # ── Save / load ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save critic weights.  Actor does NOT need this at inference."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "normalizer": self._normalizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        if "normalizer" in data:
            self._normalizer.load_state_dict(data["normalizer"])

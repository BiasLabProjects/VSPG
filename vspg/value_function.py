"""Lightweight MLP value function V(s_hv) → scalar for single-agent VSPG + GAE."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class SimpleValueFunction:
    """MLP critic V(s_hv) for single-agent advantage estimation.

    Takes the HD-encoded state vector (shape = cfg.dim) as input and outputs
    a scalar V(s) estimate. Used exclusively for GAE advantage computation
    during training; not needed at inference time.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers).to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

    def predict(self, s_hv: np.ndarray) -> float:
        """Predict V(s) for a single HD state vector."""
        with torch.no_grad():
            s = torch.as_tensor(s_hv, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.net(s).squeeze().item())

    def predict_batch(self, s_hvs) -> np.ndarray:
        """Predict V for a batch; s_hvs is list or array of shape (T, dim) → (T,)."""
        with torch.no_grad():
            S = torch.as_tensor(np.array(s_hvs), dtype=torch.float32, device=self.device)
            return self.net(S).squeeze(1).cpu().numpy()

    def update(self, s_hvs, targets: np.ndarray) -> float:
        """One gradient step minimising MSE(V(s), G_t). Returns scalar loss."""
        S = torch.as_tensor(np.array(s_hvs), dtype=torch.float32, device=self.device)
        G = torch.as_tensor(targets.astype(np.float32), dtype=torch.float32, device=self.device)

        V_pred = self.net(S).squeeze(1)
        loss = self._loss_fn(V_pred, G)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

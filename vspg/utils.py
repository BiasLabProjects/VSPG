"""Utility functions: return computation, advantage normalization, metrics logging."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.text import Text

_console = Console()


def compute_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Discounted return G_t = r_t + gamma * G_{t+1}, shape (T,)."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def normalize_advantages(adv: np.ndarray) -> np.ndarray:
    """Normalize to zero mean, unit std. Returns zeros if variance is too small."""
    std = adv.std()
    if std < 1e-8:
        return np.zeros_like(adv)
    return (adv - adv.mean()) / (std + 1e-8)


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Generalized Advantage Estimation (Schulman et al., 2015).

    δ_t = r_t + γ·V(s_{t+1}) − V(s_t),  A_t = δ_t + (γλ)·δ_{t+1} + …
    Returns advantages of shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0  # episode ends → V(s_{T+1}) = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compute_run_summary(logger: "MetricsLogger", smooth_window: int = 50, final_window: int = 100) -> dict:
    """End-of-run summary statistics for summary.json.

    Mirrors the SMOOTH=50 / FINAL_WINDOW=100, rolling(center=True) / tail()
    conventions already used by analyze.py and make_report_figures.py.
    """
    episodes = logger.episodes()
    returns = pd.Series(logger.series("return"), dtype="float64")
    lengths = pd.Series(logger.series("length"), dtype="float64")

    if returns.empty:
        return {
            "final_return_mean_last100": float("nan"),
            "final_return_std_last100": float("nan"),
            "success_rate_last100": float("nan"),
            "episode_length_mean_last100": float("nan"),
            "best_rolling_return": float("nan"),
            "best_episode": None,
        }

    tail = returns.tail(final_window)
    smoothed = returns.rolling(smooth_window, min_periods=1, center=True).mean()
    best_idx = int(smoothed.idxmax())

    return {
        "final_return_mean_last100": float(tail.mean()),
        "final_return_std_last100": float(tail.std()),
        "success_rate_last100": float((tail > 0).mean()),
        "episode_length_mean_last100": float(lengths.tail(final_window).mean()) if not lengths.empty else float("nan"),
        "best_rolling_return": float(smoothed.max()),
        "best_episode": int(episodes[best_idx]),
    }


class MetricsLogger:
    def __init__(self) -> None:
        self._data: dict[str, list] = defaultdict(list)
        self._episodes: list[int] = []

    def log(self, episode: int, metrics: dict) -> None:
        self._episodes.append(episode)
        for k, v in metrics.items():
            self._data[k].append(float(v))

    def moving_avg(self, key: str, window: int = 100) -> float:
        vals = self._data.get(key, [])
        if not vals:
            return float("nan")
        return float(np.mean(vals[-window:]))

    def moving_std(self, key: str, window: int = 100) -> float:
        vals = self._data.get(key, [])
        if len(vals) < 2:
            return float("nan")
        return float(np.std(vals[-window:]))

    def last(self, key: str, default: float = float("nan")) -> float:
        vals = self._data.get(key, [])
        return float(vals[-1]) if vals else default

    def series(self, key: str) -> list[float]:
        """Read-only access to a logged metric's full history."""
        return list(self._data.get(key, []))

    def episodes(self) -> list[int]:
        """Read-only access to the list of logged episode indices."""
        return list(self._episodes)

    def summary_str(self, episode: int, metrics: dict, moving_avg: float) -> str:
        ep_ret = metrics.get("return", float("nan"))
        ep_len = metrics.get("length", float("nan"))
        entropy = metrics.get("mean_entropy", float("nan"))
        rho = metrics.get("mean_rho", float("nan"))
        flip = metrics.get("flip_pct", None)
        parts = [
            f"ep {episode + 1:>5}",
            f"ret={ep_ret:+.3f}",
            f"avg100={moving_avg:+.3f}",
            f"len={ep_len:.0f}",
            f"H={entropy:.3f}",
        ]
        if not np.isnan(rho):
            parts.append(f"rho={rho:.3f}")
        if flip is not None:
            parts.append(f"flip%={100*flip:.2f}")
        return "  ".join(parts)

    def print_summary(self, episode: int, metrics: dict, moving_avg: float) -> None:
        _console.print(self.summary_str(episode, metrics, moving_avg))

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._data.keys())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode"] + keys)
            for i, ep in enumerate(self._episodes):
                row = [ep] + [self._data[k][i] if i < len(self._data[k]) else "" for k in keys]
                writer.writerow(row)

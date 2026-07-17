"""HDC and DNN policy variants for Advantage-Weighted Policy Learning on MiniGrid.

All policies expose:
    act(s_hv, greedy=False)                -> (action, logp, entropy)
    update(s_hvs, actions, advantages, eta) -> metrics dict
    save(path) / load(path)
    action_counts()                         -> np.ndarray  (n_actions,)

RealHDPolicy      : action HVs c_a ∈ R^D, L2-normalised.
                    Update = full softmax PG gradient, batched over the episode.
MLPReinforcePolicy: small MLP + REINFORCE + Adam (DNN baseline).
                    Batched forward pass in update(); no log-prob buffer.
RandomPolicy      : uniform random, no learning (sanity baseline).

All tensor-based policies accept a `device` argument ("cpu", "cuda", …).
Default is "cpu"; cfg.device / build_policy() selects the device at runtime.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam as _Adam
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from utils import softmax
except ImportError:
    from .utils import softmax

# ─────────────────────────────────────────────────────────────────────────────


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 42) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_actions, dtype=np.int64)

    def act(self, s_hv: np.ndarray, greedy: bool = False) -> tuple[int, float, float]:
        action = int(self._rng.integers(self.n_actions))
        self._counts[action] += 1
        return action, float(np.log(1.0 / self.n_actions)), float(np.log(self.n_actions))

    def update(
        self,
        s_hvs,
        actions,
        advantages,
        eta: float,
        log_probs_old: np.ndarray | None = None,  # ignored — no learning
        ppo_clip_eps: float = 0.0,                # kept for call-site compatibility
    ) -> dict:
        return {}

    def action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def save(self, path: str) -> None:
        np.savez_compressed(path, counts=self._counts)

    def load(self, path: str) -> None:
        self._counts = np.load(path)["counts"]


# ─────────────────────────────────────────────────────────────────────────────


class RealHDPolicy:
    """Real-valued, L2-normalised action hypervectors.

    Policy:   logit_a = tau * dot(s, c_a)   (= cosine sim since both unit-norm)
              pi(a|s) = softmax(logits)

    Update (full softmax policy gradient, batched over the episode):
        S      = stack(s_hvs)                           (T, D)
        logits = tau * S @ C.T                          (T, n_actions)
        probs  = softmax(logits, dim=1)                 (T, n_actions)
        scales = A[:,None] * tau * (one_hot(acts) - probs)  (T, n_actions)
        grad   = scales.T @ S                           (n_actions, D)
        C     += eta * grad
        C      = C / (||C||_row + eps)

    All T steps contribute in one vectorised pass — no per-step loop.
    """

    def __init__(
        self,
        dim: int,
        n_actions: int,
        tau: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        fhrr_init: bool = False,
        row_normalize: bool = True,
        zero_init: bool = False,
    ) -> None:
        self.dim = dim
        self.n_actions = n_actions
        self.tau = tau
        self.device = torch.device(device)
        self.row_normalize = row_normalize
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_actions, dtype=np.int64)

        if zero_init:
            # C starts at exact zero instead of a random unit-norm draw --
            # act() is exactly uniform until the first update, and every
            # trained hypervector becomes a pure kernel-memory expansion
            # over experience (no random-direction contribution). Note
            # self._rng consumes no draws here, unlike the branches below,
            # so the RNG stream act() later samples from starts at a
            # different point than a non-zero-init run with the same seed.
            raw = np.zeros((n_actions, dim), dtype=np.float32)
        elif fhrr_init and dim % 2 == 0:
            # FHRR-proper: each (Re_k, Im_k) pair on the unit circle
            # → dot products with FHRR state encodings are well-concentrated
            cdim = dim // 2
            theta = self._rng.uniform(-np.pi, np.pi, (n_actions, cdim)).astype(np.float32)
            raw = np.empty((n_actions, dim), dtype=np.float32)
            raw[:, 0::2] = np.cos(theta)
            raw[:, 1::2] = np.sin(theta)
        else:
            raw = self._rng.standard_normal((n_actions, dim)).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            raw = raw / (norms + 1e-8)

        self.C = torch.from_numpy(raw).to(self.device)

    def _s(self, s_hv: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(s_hv, dtype=torch.float32, device=self.device)

    def act(self, s_hv: np.ndarray, greedy: bool = False) -> tuple[int, float, float]:
        with torch.no_grad():
            # logit_a = tau * cos_sim(s, c_a)  ∈ [-tau, tau]
            logits = (self.tau * (self.C @ self._s(s_hv))).cpu().numpy()
        probs = softmax(logits)
        action = int(np.argmax(probs)) if greedy else int(self._rng.choice(self.n_actions, p=probs))
        self._counts[action] += 1
        return action, float(np.log(probs[action] + 1e-8)), float(-np.sum(probs * np.log(probs + 1e-8)))

    def get_log_prob(self, s_hv: np.ndarray, action: int) -> float:
        """Compute log π(action | s_hv) under the current policy."""
        with torch.no_grad():
            logits = (self.tau * (self.C @ self._s(s_hv))).cpu().numpy()
        probs = softmax(logits)
        return float(np.log(probs[action] + 1e-8))

    def update(
        self,
        s_hvs,
        actions,
        advantages,
        eta: float,
        log_probs_old: np.ndarray | None = None,
        ppo_clip_eps: float = 0.0,
    ) -> dict:
        with torch.no_grad():
            T = len(s_hvs)
            # ── batch all episode data into tensors ──────────────────────────
            S    = torch.stack([self._s(s) for s in s_hvs])              # (T, D)
            A    = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)  # (T,)
            acts = torch.as_tensor(actions,   dtype=torch.long,    device=self.device)   # (T,)

            # ── softmax policy probs for every step ──────────────────────────
            logits = self.tau * (S @ self.C.T)                            # (T, n_actions)
            probs  = torch.softmax(logits, dim=1)                         # (T, n_actions)

            # ── PPO-clip on the advantage signal ─────────────────────────────
            if ppo_clip_eps > 0.0 and log_probs_old is not None:
                lp_new = torch.log_softmax(logits, dim=1).gather(
                    1, acts[:, None]
                ).squeeze(1)                                               # (T,)
                lp_old = torch.as_tensor(
                    log_probs_old, dtype=torch.float32, device=self.device
                )                                                          # (T,)
                ratio   = torch.exp(lp_new - lp_old)                      # (T,)
                clipped = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
                # standard PPO: min(r*A, clip(r)*A) — conservative update
                effective_A = torch.min(ratio * A, clipped * A)           # (T,)
            else:
                effective_A = A

            # ── full softmax PG gradient ─────────────────────────────────────
            # scales[t, a] = eff_A_t · τ · (𝟙[a=a_t] − π(a|s_t))
            one_hot = torch.zeros(T, self.n_actions, device=self.device)
            one_hot.scatter_(1, acts[:, None], 1.0)
            scales = effective_A[:, None] * self.tau * (one_hot - probs)  # (T, n_actions)

            # grad[a, d] = sum_t scales[t,a] * s_t[d]  — one matmul
            grad = scales.T @ S                                            # (n_actions, D)

            self.C.add_(eta * grad)
            self._last_pre_norm_rownorms = self.C.norm(dim=1).clone()
            if self.row_normalize:
                self.C.div_(self.C.norm(dim=1, keepdim=True) + 1e-8)

            # ── logging metrics ──────────────────────────────────────────────
            mean_rho = float(torch.sum(S * self.C[acts], dim=1).mean().item())

        return {"mean_rho": mean_rho, "mean_abs_adv": float(A.abs().mean().item())}

    def action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if hasattr(self, "_last_pre_norm_rownorms"):
            extra["pre_norm_rownorms"] = self._last_pre_norm_rownorms.cpu().numpy()
        np.savez_compressed(path, C=self.C.cpu().numpy(), counts=self._counts, **extra)

    def load(self, path: str) -> None:
        data = np.load(path + ".npz")
        self.C = torch.from_numpy(data["C"]).to(self.device)
        self._counts = data["counts"]


# ─────────────────────────────────────────────────────────────────────────────


class MLPReinforcePolicy:
    """DNN baseline: MLP trained with REINFORCE + Adam.

    update() recomputes log-probs from s_hvs in a single batched forward pass
    — no per-step log-prob buffer needed.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (128, 64),
        lr: float = 3e-4,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for MLPReinforcePolicy.")
        torch.manual_seed(seed)
        self.device = torch.device(device)

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers).to(self.device)
        self.optimizer = _Adam(self.net.parameters(), lr=lr)

        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)
        self._counts = np.zeros(n_actions, dtype=np.int64)

    def _x(self, obs_flat: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(obs_flat, dtype=torch.float32, device=self.device)

    def act(self, obs_flat: np.ndarray, greedy: bool = False) -> tuple[int, float, float]:
        with torch.no_grad():
            logits = self.net(self._x(obs_flat).unsqueeze(0)).squeeze(0)
            probs_t = torch.softmax(logits, dim=0)
        probs = probs_t.cpu().numpy()
        action = int(np.argmax(probs)) if greedy else int(self._rng.choice(self.n_actions, p=probs))
        self._counts[action] += 1
        logp    = float(np.log(probs[action] + 1e-8))
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))
        return action, logp, entropy

    def update(
        self,
        s_hvs,
        actions,
        advantages,
        eta: float,
        log_probs_old: np.ndarray | None = None,
        ppo_clip_eps: float = 0.0,
    ) -> dict:
        if not s_hvs:
            return {}

        # ── single batched forward pass ──────────────────────────────────────
        S    = torch.stack([self._x(s) for s in s_hvs])                   # (T, obs_dim)
        acts = torch.as_tensor(actions,   dtype=torch.long,    device=self.device)
        A    = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        logits    = self.net(S)                                            # (T, n_actions)
        log_probs = torch.log_softmax(logits, dim=1)                      # (T, n_actions)
        sel_lp    = log_probs.gather(1, acts[:, None]).squeeze(1)         # (T,)

        if ppo_clip_eps > 0.0 and log_probs_old is not None:
            lp_old  = torch.as_tensor(log_probs_old, dtype=torch.float32, device=self.device)
            ratio   = torch.exp(sel_lp - lp_old)
            clipped = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
            obj     = torch.min(ratio * A, clipped * A)
        else:
            obj = sel_lp * A

        loss = -obj.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"policy_loss": float(loss.item())}

    def action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "counts": self._counts}, path + ".pt")

    def load(self, path: str) -> None:
        # map_location=self.device: a checkpoint saved on a machine with more
        # GPUs (e.g. cuda:1) would otherwise fail with "Attempting to
        # deserialize object on CUDA device 1 but torch.cuda.device_count()
        # is 1" on a single-GPU machine -- remap to wherever this policy
        # instance actually lives instead of trusting the saved location.
        data = torch.load(path + ".pt", weights_only=False, map_location=self.device)
        self.net.load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self._counts = data["counts"]


# ─────────────────────────────────────────────────────────────────────────────


class LinearReinforceActor:
    """Linear policy (W @ obs + b → logits) trained with REINFORCE + Adam.

    Same external interface as RealHDPolicy so MultiAgentVSPGPolicy can
    swap it in transparently. The `eta` argument to update() is ignored
    (Adam lr is fixed at construction); kept for call-site compatibility.

    Supports optional PPO-clip on the REINFORCE update.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        tau: float = 1.0,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("LinearReinforceActor requires PyTorch.")
        torch.manual_seed(seed)
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.tau       = tau
        self.device    = torch.device(device)
        self._counts   = np.zeros(n_actions, dtype=np.int64)

        self.net = nn.Linear(obs_dim, n_actions).to(self.device)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)
        self.optimizer = _Adam(self.net.parameters(), lr=lr)

    def _logits(self, obs_hv: np.ndarray) -> "torch.Tensor":
        x = torch.as_tensor(obs_hv, dtype=torch.float32, device=self.device)
        return self.tau * self.net(x)

    def act(
        self, obs_hv: np.ndarray, greedy: bool = False
    ) -> tuple[int, float, float]:
        with torch.no_grad():
            logits = self._logits(obs_hv)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs     = log_probs.exp()
            entropy   = float(-(probs * log_probs).sum())
        if greedy:
            action = int(logits.argmax())
        else:
            action = int(torch.multinomial(probs, 1))
        self._counts[action] += 1
        return action, float(log_probs[action]), entropy

    def get_log_prob(self, obs_hv: np.ndarray, action: int) -> float:
        with torch.no_grad():
            logits = self._logits(obs_hv)
            return float(torch.log_softmax(logits, dim=-1)[action])

    def update(
        self,
        obs_list: list,
        actions: list[int],
        advantages: np.ndarray,
        eta: float,                        # ignored — Adam lr used instead
        log_probs_old: np.ndarray | None = None,
        ppo_clip_eps: float = 0.0,
    ) -> dict:
        if len(obs_list) == 0:
            return {}

        S   = torch.tensor(np.array(obs_list), dtype=torch.float32, device=self.device)
        acts = torch.tensor(actions, dtype=torch.long, device=self.device)
        A    = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        logits    = self.tau * self.net(S)                              # (T, n_actions)
        log_probs = torch.log_softmax(logits, dim=-1)
        lp_new    = log_probs.gather(1, acts[:, None]).squeeze(1)       # (T,)

        if ppo_clip_eps > 0.0 and log_probs_old is not None:
            lp_old  = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)
            ratio   = torch.exp(lp_new - lp_old)
            clipped = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
            obj     = torch.min(ratio * A, clipped * A)
        else:
            obj = lp_new * A

        loss = -obj.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            probs   = torch.softmax(logits, dim=-1)
            entropy = float(-(probs * log_probs).sum(dim=-1).mean())

        return {"loss": float(loss), "mean_entropy": entropy}

    def action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def save(self, path: str) -> None:
        torch.save({
            "net":       self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "counts":    self._counts,
        }, path + ".pt")

    def load(self, path: str) -> None:
        # map_location=self.device: a checkpoint saved on a machine with more
        # GPUs (e.g. cuda:1) would otherwise fail with "Attempting to
        # deserialize object on CUDA device 1 but torch.cuda.device_count()
        # is 1" on a single-GPU machine -- remap to wherever this policy
        # instance actually lives instead of trusting the saved location.
        data = torch.load(path + ".pt", weights_only=False, map_location=self.device)
        self.net.load_state_dict(data["net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self._counts = data["counts"]


# ─────────────────────────────────────────────────────────────────────────────


def build_policy(cfg, n_actions: int, device: str = "cpu"):
    """Factory: select policy class from cfg.variant."""
    variant = getattr(cfg, "variant", "real")
    dim     = getattr(cfg, "dim",  1000)
    tau     = getattr(cfg, "tau",  1.0)
    seed    = getattr(cfg, "seed", 42)

    if variant == "random":
        return RandomPolicy(n_actions=n_actions, seed=seed)

    if variant == "real":
        fhrr_init = getattr(cfg, "fhrr_init", False)
        row_normalize = bool(getattr(cfg, "row_normalize", True))
        return RealHDPolicy(dim=dim, n_actions=n_actions, tau=tau, seed=seed,
                            device=device, fhrr_init=fhrr_init, row_normalize=row_normalize)

    if variant == "mlp_reinforce":
        view    = getattr(cfg, "obs_view_size", 7)
        obs_dim = getattr(cfg, "obs_dim", view * view * 3 + 1)
        hidden  = tuple(getattr(cfg, "mlp_hidden", [128, 64]))
        lr      = float(getattr(cfg, "mlp_lr", 3e-4))
        return MLPReinforcePolicy(obs_dim=obs_dim, n_actions=n_actions,
                                  hidden=hidden, lr=lr, seed=seed, device=device)

    if variant == "linear_reinforce":
        view    = getattr(cfg, "obs_view_size", 7)
        obs_dim = getattr(cfg, "obs_dim", view * view * 3 + 1)
        lr      = float(getattr(cfg, "linear_lr", 3e-4))
        return LinearReinforceActor(obs_dim=obs_dim, n_actions=n_actions,
                                    tau=tau, lr=lr, seed=seed, device=device)

    raise ValueError(
        f"Unknown variant: {variant!r}. Choose: real | mlp_reinforce | linear_reinforce | random"
    )

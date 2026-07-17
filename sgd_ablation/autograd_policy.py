"""Autograd-based categorical actor with exactly the same policy as RealHDPolicy.

    logits = tau * phi(x) @ C.T
    pi(a | x) = softmax(logits)

RealHDPolicy (vspg/policies.py) learns C via a hand-derived,
closed-form update computed under torch.no_grad() -- no autograd, no
optimizer, no momentum:

    Lambda[t, a] = A_t * tau * (1[a == a_t] - pi(a | x_t))
    C <- row_normalize(C + eta * Lambda.T @ S)

AutogradHDPolicy replaces only that update rule with PyTorch autograd +
torch.optim, while keeping the policy, initialization, and action-sampling
logic byte-identical (via subclassing, not reimplementation). No gradient
ever reaches the encoder: encoded states arrive as plain numpy arrays
(HDCEncoder.encode() returns np.ndarray, not a tensor with a graph), so the
autograd graph for update() is rooted only at C.

Three modes:
    sgd_normalized   -- SGD(lr=eta),  row-L2-renormalize C after the step
                         (matches RealHDPolicy.update's default, row_normalize=True)
    sgd_unnormalized -- SGD(lr=eta),  no renormalization
                         (matches RealHDPolicy.update with row_normalize=False)
    adam_normalized  -- Adam(lr=adam_lr or eta), row-L2-renormalize C after the step
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vspg.policies import RealHDPolicy  # import only -- never modified


class AutogradHDPolicy(RealHDPolicy):
    VALID_MODES = ("sgd_normalized", "sgd_unnormalized", "adam_normalized")

    def __init__(
        self,
        dim: int,
        n_actions: int,
        tau: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        fhrr_init: bool = False,
        zero_init: bool = False,
        mode: str = "sgd_normalized",
        eta: float = 1e-3,
        adam_lr: float | None = None,
    ) -> None:
        # Reused verbatim: draws C's initial values from the exact same
        # np.random.default_rng(seed) call sequence as RealHDPolicy, so
        # initialization is byte-identical to the manual-update variant
        # (zero_init=True: both start at exact zero instead).
        super().__init__(
            dim=dim, n_actions=n_actions, tau=tau, seed=seed,
            device=device, fhrr_init=fhrr_init, row_normalize=True,
            zero_init=zero_init,
        )
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        self.mode = mode
        self._normalize_after_step = (mode != "sgd_unnormalized")

        # Promote the plain Tensor RealHDPolicy.__init__ built into a leaf
        # nn.Parameter with identical values. No enclosing nn.Module is
        # needed -- torch.optim.SGD/Adam accept a bare parameter list.
        self.C = nn.Parameter(self.C.clone().detach())

        lr = float(adam_lr) if (mode == "adam_normalized" and adam_lr is not None) else float(eta)
        opt_cls = torch.optim.Adam if mode == "adam_normalized" else torch.optim.SGD
        self.optimizer = opt_cls([self.C], lr=lr)

    def update(
        self,
        s_hvs,
        actions,
        advantages,
        eta: float,
        log_probs_old: np.ndarray | None = None,
        ppo_clip_eps: float = 0.0,
    ) -> dict:
        # `eta` is accepted for call-site compatibility with RealHDPolicy /
        # the production train.py loop, but ignored here: the optimizer's lr
        # is fixed at construction (same "eta ignored" convention already
        # used by LinearReinforceActor.update in vspg/policies.py).
        S = torch.stack([self._s(s) for s in s_hvs])                              # (T, D) -- no grad, encoder output is numpy
        A = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)  # (T,)
        acts = torch.as_tensor(actions, dtype=torch.long, device=self.device)     # (T,)

        logits = self.tau * (S @ self.C.T)                                        # graph rooted only at self.C

        # PPO-clip branch, mirroring RealHDPolicy.update's exact (non-textbook)
        # approximation: `effective_A` is a plain scalar-weight substitute for
        # `A` computed from a DETACHED ratio (no gradient flows through the
        # ratio/ clip itself) -- it is not the literal derivative of
        # min(ratio*A, clip(ratio)*A). Replicating this exactly (rather than a
        # "textbook" PPO loss) is required for AutogradHDPolicy's gradient to
        # stay provably equivalent to RealHDPolicy's update under this mode
        # too -- see tests/test_equivalence.py's PPO-clip test.
        if ppo_clip_eps > 0.0 and log_probs_old is not None:
            with torch.no_grad():
                lp_new_detached = torch.log_softmax(logits.detach(), dim=1).gather(1, acts[:, None]).squeeze(1)
                lp_old = torch.as_tensor(log_probs_old, dtype=torch.float32, device=self.device)
                ratio = torch.exp(lp_new_detached - lp_old)
                clipped = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps)
                effective_A = torch.min(ratio * A, clipped * A)
        else:
            effective_A = A

        log_probs = torch.log_softmax(logits, dim=1)
        sel_logp = log_probs.gather(1, acts[:, None]).squeeze(1)
        loss = -(effective_A * sel_logp).sum()                                    # .sum(), not .mean() -- matches Lambda.T @ S scale exactly

        # C is a single persistent nn.Parameter reused across every episode
        # of a run; skipping zero_grad() would silently accumulate stale
        # gradients across episodes instead of crashing.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self._normalize_after_step:
            # In-place op on a leaf Parameter with requires_grad=True must be
            # wrapped in no_grad() -- unlike RealHDPolicy.update, where C is a
            # plain Tensor (requires_grad=False by default) and this isn't an issue.
            with torch.no_grad():
                self.C.div_(self.C.norm(dim=1, keepdim=True) + 1e-8)

        with torch.no_grad():
            mean_rho = float(torch.sum(S * self.C[acts], dim=1).mean().item())

        return {
            "mean_rho": mean_rho,
            "mean_abs_adv": float(A.abs().mean().item()),
            "policy_loss": float(loss.item()),
        }

    def save(self, path: str) -> None:
        # RealHDPolicy.save does self.C.cpu().numpy() with no .detach(),
        # which raises on a Parameter with requires_grad=True. Detach first.
        # Note: this is for post-hoc inspection only -- the optimizer's
        # moment buffers (Adam's exp_avg/exp_avg_sq) are not serialized,
        # since this ablation never resumes a run mid-training.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, C=self.C.detach().cpu().numpy(), counts=self._counts)

    def load(self, path: str) -> None:
        # RealHDPolicy.load does self.C = torch.from_numpy(...), which would
        # replace the Parameter object the optimizer holds a reference to,
        # silently decoupling it from further optimizer.step() calls.
        # copy_() in place preserves parameter identity instead.
        data = np.load(path + ".npz")
        with torch.no_grad():
            self.C.copy_(torch.from_numpy(data["C"]).to(self.device))
        self._counts = data["counts"]

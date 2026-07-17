"""Numerical equivalence test: manual VSPG update vs. PyTorch autograd.

Proves, on one fixed batch of *real* MiniGrid-DoorKey-8x8 transitions
(encoded through the actual HDCEncoder, actions/advantages from the actual
collect_episode + compute_reinforce_advantages pipeline -- all imported,
none reimplemented):

  1. The gradient implied by RealHDPolicy.update (Lambda.T @ S, read off as
     (C_after - C_before) / eta with row_normalize=False) equals the
     negative of PyTorch autograd's gradient of

         loss = -(A_t * log pi_C(a_t | x_t)).sum()

     w.r.t. C, within float32 tolerance.

  2. One AutogradHDPolicy(mode="sgd_unnormalized") update step reproduces
     RealHDPolicy.update's resulting C exactly (same init, same batch, same
     eta -- SGD(lr=eta) on `loss` is by definition the same step).

  3. Same as (2), but with row-normalization enabled on both sides
     (mode="sgd_normalized" / row_normalize=True), proving the
     post-step renormalization is applied identically, not just the raw
     gradient step.

Runs as a plain script (pytest is not installed in the training container),
but follows pytest's `test_*` naming so `pytest sgd_ablation/` discovers it
too -- same convention as the existing (untouched) reference test at
tests/test_linear_policy_equivalence.py.

Note on "one fixed batch": collect_episode()'s env.reset() is intentionally
left unseeded (see sgd_ablation/README.md -- inherited unchanged from
production vspg/train.py, per explicit user instruction not to
touch env seeding, since doing so would invalidate the existing experiment
corpus). So the batch's *contents* differ from run to run of this script.
That does not weaken the test: the algebraic identity under test holds for
any batch, and "fixed" means fixed *within* one invocation -- the same S/A/
acts tensors are reused, unmodified, on both sides of every comparison below.
"""
from __future__ import annotations

import os
import sys

import numpy as np
np.float_ = np.float64  # gymnasium/minigrid + modern numpy compat shim, same as vspg/train.py

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from vspg.policies import RealHDPolicy
from vspg.encoders import HDCEncoder
from vspg.train import load_config, collect_episode, compute_reinforce_advantages
from sgd_ablation.autograd_policy import AutogradHDPolicy

DEVICE = "cpu"
_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "ablation_base.yaml")
_N_EPISODES_FOR_BATCH = 3


def _make_batch():
    """Collect one fixed batch of real DoorKey-8x8 transitions."""
    import gymnasium as gym
    import minigrid  # noqa: F401  registers MiniGrid envs

    cfg = load_config(_CFG_PATH, overrides={"device": DEVICE})
    env = gym.make(cfg.env_id)
    encoder = HDCEncoder(dim=cfg.dim, obs_view_size=cfg.obs_view_size, seed=cfg.seed)

    # A throwaway policy just to drive action sampling while collecting the
    # batch -- its own weights are irrelevant, only the resulting (s_hv,
    # action, reward) trajectories are used below.
    driver = RealHDPolicy(dim=cfg.dim, n_actions=env.action_space.n, tau=cfg.tau,
                           seed=cfg.seed, device=DEVICE)

    group = []
    for _ in range(_N_EPISODES_FOR_BATCH):
        traj, ep_return, ep_len = collect_episode(env, driver, encoder, cfg)
        group.append((traj, ep_return, ep_len))

    advantages, _ = compute_reinforce_advantages(group, cfg.gamma)
    s_hvs = [step[0] for traj, _, _ in group for step in traj]
    actions = [step[1] for traj, _, _ in group for step in traj]

    return s_hvs, actions, advantages, cfg


# ── 1. THE critical equivalence test ────────────────────────────────────────

def test_autograd_gradient_matches_manual_vspg_gradient():
    s_hvs, actions, advantages, cfg = _make_batch()
    n_actions = int(max(actions)) + 1

    manual = RealHDPolicy(dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=11,
                           device=DEVICE, row_normalize=False)
    C0 = manual.C.clone()
    manual.update(s_hvs, actions, advantages, eta=cfg.eta)
    manual_grad = (manual.C - C0) / cfg.eta  # == Lambda.T @ S exactly, row_normalize=False

    S = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in s_hvs])
    A = torch.as_tensor(advantages, dtype=torch.float32)
    acts = torch.as_tensor(actions, dtype=torch.long)

    C_leaf = C0.clone().detach().requires_grad_(True)
    logits = cfg.tau * (S @ C_leaf.T)
    log_probs = torch.log_softmax(logits, dim=1)
    sel_logp = log_probs.gather(1, acts[:, None]).squeeze(1)
    loss = -(A * sel_logp).sum()
    loss.backward()
    autograd_grad = C_leaf.grad.detach()

    max_abs_diff = (manual_grad - (-autograd_grad)).abs().max().item()
    max_rel_diff = ((manual_grad - (-autograd_grad)).abs()
                     / (-autograd_grad).abs().clamp_min(1e-8)).max().item()
    print(f"    max_abs_diff={max_abs_diff:.3e}  max_rel_diff={max_rel_diff:.3e}")

    assert torch.allclose(manual_grad, -autograd_grad, atol=1e-5, rtol=1e-4), (
        "Manual VSPG update's implied gradient does not match the autograd "
        "REINFORCE gradient of sum_t A_t log pi(a_t|x_t) -- this would mean "
        "VSPG's update is not equivalent to SGD on the standard policy-"
        "gradient surrogate, contradicting Proposition 1."
    )


# ── 2. one SGD(lr=eta) step, no normalization, reproduces C exactly ────────

def test_sgd_unnormalized_matches_manual_update_exactly():
    s_hvs, actions, advantages, cfg = _make_batch()
    n_actions = int(max(actions)) + 1
    seed = 23

    manual = RealHDPolicy(dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=seed,
                           device=DEVICE, row_normalize=False)
    C0 = manual.C.clone()
    manual.update(s_hvs, actions, advantages, eta=cfg.eta)

    ag = AutogradHDPolicy(dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=seed,
                           device=DEVICE, mode="sgd_unnormalized", eta=cfg.eta)
    assert torch.equal(ag.C.detach(), C0), "AutogradHDPolicy init does not match RealHDPolicy init for the same seed"

    ag.update(s_hvs, actions, advantages, eta=cfg.eta)

    max_abs_diff = (ag.C.detach() - manual.C).abs().max().item()
    print(f"    max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(ag.C.detach(), manual.C, atol=1e-5, rtol=1e-4), (
        "AutogradHDPolicy(mode='sgd_unnormalized').update does not reproduce "
        "RealHDPolicy.update's resulting C exactly."
    )


# ── 3. same, but with row-normalization enabled on both sides ──────────────

def test_sgd_normalized_matches_manual_row_normalize():
    s_hvs, actions, advantages, cfg = _make_batch()
    n_actions = int(max(actions)) + 1
    seed = 37

    manual = RealHDPolicy(dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=seed,
                           device=DEVICE, row_normalize=True)
    C0 = manual.C.clone()
    manual.update(s_hvs, actions, advantages, eta=cfg.eta)

    ag = AutogradHDPolicy(dim=cfg.dim, n_actions=n_actions, tau=cfg.tau, seed=seed,
                           device=DEVICE, mode="sgd_normalized", eta=cfg.eta)
    assert torch.equal(ag.C.detach(), C0), "AutogradHDPolicy init does not match RealHDPolicy init for the same seed"

    ag.update(s_hvs, actions, advantages, eta=cfg.eta)

    row_norms = ag.C.detach().norm(dim=1)
    assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5), \
        "AutogradHDPolicy(mode='sgd_normalized') did not renormalize C rows to unit length"

    max_abs_diff = (ag.C.detach() - manual.C).abs().max().item()
    print(f"    max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(ag.C.detach(), manual.C, atol=1e-5, rtol=1e-4), (
        "AutogradHDPolicy(mode='sgd_normalized').update does not reproduce "
        "RealHDPolicy.update's row-normalized C exactly."
    )


# ── 4. PPO-clip path (needed for Acrobot-v1's bipolar/GAE config) ──────────
#
# Synthetic data here, not the real DoorKey-8x8 batch: this test is purely
# about the algebraic identity of the PPO-clip branch (env/encoder-agnostic),
# and needs a fabricated log_probs_old buffer that plain REINFORCE batches
# don't carry. Mirrors PGHD/tests/test_linear_policy_equivalence.py's own
# synthetic-problem convention (that file is read-only reference, not modified).

def _make_synthetic_ppo_batch(seed=0, T=20, D=24, n_actions=4):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((T, D)).astype(np.float32)
    S_np = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)
    s_hvs = [S_np[t] for t in range(T)]
    actions = rng.integers(0, n_actions, size=T).tolist()
    advantages = rng.standard_normal(T).astype(np.float32)
    log_probs_old = rng.uniform(-2.0, -0.1, size=T).astype(np.float32)
    return s_hvs, actions, advantages, log_probs_old, D, n_actions


def test_ppo_clip_sgd_unnormalized_matches_manual_update_exactly():
    s_hvs, actions, advantages, log_probs_old, D, n_actions = _make_synthetic_ppo_batch()
    tau, eta, ppo_clip_eps = 3.0, 0.02, 0.2
    seed = 43

    manual = RealHDPolicy(dim=D, n_actions=n_actions, tau=tau, seed=seed,
                           device=DEVICE, row_normalize=False)
    C0 = manual.C.clone()
    manual.update(s_hvs, actions, advantages, eta=eta,
                  log_probs_old=log_probs_old, ppo_clip_eps=ppo_clip_eps)

    ag = AutogradHDPolicy(dim=D, n_actions=n_actions, tau=tau, seed=seed,
                           device=DEVICE, mode="sgd_unnormalized", eta=eta)
    assert torch.equal(ag.C.detach(), C0), "AutogradHDPolicy init does not match RealHDPolicy init for the same seed"

    ag.update(s_hvs, actions, advantages, eta=eta,
              log_probs_old=log_probs_old, ppo_clip_eps=ppo_clip_eps)

    max_abs_diff = (ag.C.detach() - manual.C).abs().max().item()
    print(f"    max_abs_diff={max_abs_diff:.3e}")
    assert torch.allclose(ag.C.detach(), manual.C, atol=1e-5, rtol=1e-4), (
        "AutogradHDPolicy's PPO-clip branch does not reproduce RealHDPolicy."
        "update's PPO-clip resulting C exactly -- required for a fair "
        "Acrobot-v1 (bipolar/GAE) ablation comparison."
    )


_TESTS = [
    test_autograd_gradient_matches_manual_vspg_gradient,
    test_sgd_unnormalized_matches_manual_update_exactly,
    test_sgd_normalized_matches_manual_row_normalize,
    test_ppo_clip_sgd_unnormalized_matches_manual_update_exactly,
]


if __name__ == "__main__":
    failures = []
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {exc}")
    if failures:
        raise SystemExit(f"\n{len(failures)} test(s) failed: {failures}")
    print(f"\nAll {len(_TESTS)} tests passed.")

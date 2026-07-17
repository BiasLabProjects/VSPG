"""Evaluate pretrained SustainGym checkpoints under quantization + random bit-flip
corruption of the STORED WEIGHTS -- the LogHD-style robustness axis, distinct
from scripts/eval_noise_robustness.py's INPUT-observation-noise axis.

Protocol (mirrors the LogHD paper, Section IV-A / Fig. 3-4):
    "Training is performed in 32-bit floating point; for each target precision
    (1, 2, 4, 8 bits) we apply post-training quantization to the learned model
    parameters and then evaluate on the test set... Random bit flips are
    injected into the stored model state prior to each test evaluation...
    Test inputs are not corrupted."

Concretely, for each checkpoint x each (bits, flip_prob) cell:
  1. Load the float32 checkpoint (unchanged from training).
  2. Post-training quantize to `bits`-bit signed integers (per-tensor
     min-max affine quantization -- the standard scheme, analogous to
     LogHD's post-training quantization pass).
  3. Flip each bit of the quantized integer representation independently
     with probability `flip_prob`.
  4. Dequantize back to float32 and load into the policy in memory (no
     checkpoint files are ever overwritten on disk).
  5. Roll out `--episodes` CLEAN (no input noise) episodes and record
     avg_reward_per_step.
  6. Repeat for `--trials` independent random bit-flip masks per cell and
     average, since a single random mask draw is a noisy estimate of the
     true corruption sensitivity.

What gets corrupted, per baseline (see docs/sustaingym_noise_clean.md):
  - MAPPO(Bipolar-VSPG) / MAPPO(FHRR) / MAPPO(RFF): the action-hypervector
    matrix `C` (RealHDPolicy.save() writes this as the "C" array in the
    checkpoint's .npz) -- the direct analogue of LogHD's bundle hypervectors.
  - MAPPO(DNN) / MAPPO(Raw-Linear): every weight/bias tensor in the actor's
    `net` state dict (both LinearReinforceActor and MLPReinforcePolicy save
    `{"net": net.state_dict(), ...}` to a .pt file).

Usage:
    python scripts/eval_bitflip_robustness.py \\
        --checkpoints results/sustaingym/hot_dry/*/*/ \\
        --bits 1 2 4 8 --flip-probs 0 0.05 0.1 0.2 0.4 0.6 0.8 \\
        --episodes 20 --trials 3 \\
        --out results/figures/bitflip_robustness_two_climates.png

    # Stochastic action selection instead of greedy (see eval_noise_robustness.py
    # --stochastic and docs/sustaingym_noise_clean.md "greedy vs stochastic eval"
    # for why -- every baseline is trained stochastically, so this is the more
    # directly training-comparable condition; use uniformly across all 5
    # baselines, not just dnn, to keep this a controlled comparison):
    python scripts/eval_bitflip_robustness.py --stochastic ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    def _pbar(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)
except ImportError:
    def _pbar(iterable, desc="", total=None, **kwargs):
        if desc:
            print(f"{desc} ...")
        return iterable


# ── Config loader (same convention as eval_noise_robustness.py) ──────────────

def _load_cfg(ckpt_dir: str):
    from types import SimpleNamespace
    run_name = os.path.basename(ckpt_dir.rstrip("/"))
    candidates = [
        os.path.join(ckpt_dir, "config.yaml"),
        os.path.join("logs_sustaingym", run_name, "config.yaml"),
        os.path.join(os.path.dirname(ckpt_dir.rstrip("/")), "..", "logs_sustaingym", run_name, "config.yaml"),
    ]
    import yaml
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            with open(p) as f:
                d = yaml.safe_load(f)
            return SimpleNamespace(**d)
    raise FileNotFoundError(f"config.yaml not found for {run_name}. Tried: {candidates}")


def _label(cfg, method: str) -> str:
    """SustainGym-specific MAPPO(actor) labels (docs/sustaingym_noise_clean.md).
    nonsb3_mappo_* entries are the isolated retrain from experiments/
    mappo_sb3_comparison.sh's train-nonsb3 step (results/mappo_sb3_comparison/),
    not the main results/sustaingym tree -- given a distinct label so a
    combined plot across both trees doesn't read as duplicate/ambiguous
    "MAPPO(DNN)" lines."""
    names = {
        "dnn": "MAPPO(DNN)", "raw_linear": "MAPPO(Raw-Linear)",
        "bipolar_vspg": "VSPG (Bipolar)", "fhrr_vspg": "VSPG (FHRR)",
        "rff_vspg": "VSPG (RFF)",
        "nonsb3_mappo_dnn": "Non-SB3 MAPPO(DNN)",
        "nonsb3_mappo_raw_linear": "Non-SB3 MAPPO(Raw-Linear)",
    }
    return names.get(method, method)


# ── Quantize + bit-flip + dequantize ──────────────────────────────────────────

def _quantize_flip_dequantize(x: np.ndarray, bits: int, flip_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Per-tensor affine quantization to `bits`-bit signed ints, independent
    random bit flips at probability `flip_prob` per bit, then dequantize.
    bits=0 or flip_prob=0 short-circuits appropriately (still applies
    quantization noise at bits>0 even if flip_prob==0, matching the paper's
    "for each target precision... quantization... [then evaluate]" -- clean
    quantization-only points are meaningful baselines against later-added
    bit flips)."""
    if x.size == 0:
        return x
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return x.astype(np.float32)

    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    scale = (hi - lo) / (qmax - qmin)
    if scale <= 0:
        return x.astype(np.float32)

    q = np.round((x - lo) / scale + qmin).astype(np.int64)
    q = np.clip(q, qmin, qmax)
    # shift to unsigned range [0, 2**bits - 1] for clean bitwise flipping
    uq = (q - qmin).astype(np.uint32)

    if flip_prob > 0:
        flip_bits = rng.random((uq.size, bits)) < flip_prob
        bit_values = (1 << np.arange(bits, dtype=np.uint32))
        flip_mask = (flip_bits.astype(np.uint32) * bit_values).sum(axis=1).reshape(uq.shape)
        uq = np.bitwise_xor(uq, flip_mask.astype(np.uint32))
        uq = np.clip(uq, 0, qmax - qmin)  # guard against out-of-range post-flip values

    q_corrupted = uq.astype(np.int64) + qmin
    x_corrupted = (q_corrupted - qmin) * scale + lo
    return x_corrupted.astype(np.float32)


def _corrupt_actor(actor, bits: int, flip_prob: float, rng: np.random.Generator) -> None:
    """Corrupts the actor's stored weights IN PLACE (in memory only -- never
    touches the checkpoint file on disk). Dispatches on the actor's own
    attributes rather than its class name, since RealHDPolicy (VSPG family)
    stores its action-hypervector matrix as `.C`, while LinearReinforceActor
    and MLPReinforcePolicy both store a torch `nn.Module` as `.net`."""
    import torch

    if hasattr(actor, "C"):
        arr = actor.C.detach().cpu().numpy()
        corrupted = _quantize_flip_dequantize(arr, bits, flip_prob, rng)
        actor.C = torch.from_numpy(corrupted).to(actor.C.device)
        return

    if hasattr(actor, "net"):
        with torch.no_grad():
            for p in actor.net.parameters():
                arr = p.detach().cpu().numpy()
                corrupted = _quantize_flip_dequantize(arr, bits, flip_prob, rng)
                p.copy_(torch.from_numpy(corrupted).to(p.device))
        return

    raise TypeError(f"Don't know how to corrupt actor of type {type(actor)} (no .C or .net attribute)")


# ── Eval loop ─────────────────────────────────────────────────────────────────

def _eval_run(ckpt_dir: str, bits_list: list[int], flip_probs: list[float],
               episodes: int, trials: int, stochastic: bool = False,
               base_seed: int = 42, device: str = "cpu") -> dict[tuple[int, float], list[float]]:
    """Returns {(bits, flip_prob): [avg_reward_per_step per episode per trial]}."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from envs.sustaingym_wrapper import MultiAgentBuildingWrapper
    from policies.ma_vspg_policy import MultiAgentVSPGPolicy

    cfg = _load_cfg(ckpt_dir)
    bins_cfg = getattr(cfg, "discrete_action_bins", None)
    disc_bins = getattr(cfg, "discrete_bins", None)
    episode_len = int(getattr(cfg, "episode_len", 288))

    # Clean wrapper -- NO input observation noise. This script's whole point is
    # isolating weight/parameter corruption from the separate input-noise axis
    # already covered by eval_noise_robustness.py.
    wrapper = MultiAgentBuildingWrapper(
        building=getattr(cfg, "sustaingym_building_type", "OfficeSmall"),
        weather=getattr(cfg, "sustaingym_weather", "Hot_Dry"),
        location=getattr(cfg, "sustaingym_location", "Tucson"),
        bins=bins_cfg if isinstance(bins_cfg, list) else None,
        discrete_bins=int(disc_bins) if disc_bins is not None else None,
        episode_len=episode_len,
        obs_noise_std=0.0,
        noise_seed=base_seed,
    )

    results: dict[tuple[int, float], list[float]] = {}

    for bits in _pbar(bits_list, desc="bits", leave=False):
        for flip_prob in _pbar(flip_probs, desc=f"  bits={bits}", leave=False):
            trial_vals: list[float] = []
            for trial in range(trials):
                rng = np.random.default_rng(base_seed + 1000 * bits + trial)

                policy = MultiAgentVSPGPolicy(
                    cfg, num_agents=wrapper.num_agents,
                    action_dim=wrapper.action_dim,
                    device=device,
                    share_actor=bool(getattr(cfg, "share_actor", False)),
                )
                actor_path = os.path.join(ckpt_dir, "actor_final")
                try:
                    policy.load_actor(actor_path)
                except FileNotFoundError:
                    print(f"  [warn] No checkpoint at {actor_path}* -- using random weights")

                for i in range(wrapper.num_agents):
                    _corrupt_actor(policy._actor(i), bits, flip_prob, rng)

                for ep in range(episodes):
                    obs_list, _ = wrapper.reset(seed=base_seed + trial * episodes + ep)
                    done = False
                    ep_return = 0.0
                    ep_len = 0
                    while not done:
                        z_list = policy.encode_all(obs_list)
                        actions, _, _ = policy.act(z_list, deterministic=not stochastic, encoded=True)
                        obs_list, reward, done, _ = wrapper.step(actions)
                        ep_return += reward
                        ep_len += 1
                    trial_vals.append(ep_return / max(ep_len, 1))

            results[(bits, flip_prob)] = trial_vals
            mean_r = float(np.mean(trial_vals))
            print(f"  bits={bits}  p={flip_prob:.3f}  mean={mean_r:+.4f}  std={float(np.std(trial_vals)):.4f}")

    wrapper.close()
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Bit-flip/quantization robustness eval for SustainGym baselines")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Checkpoint dirs (each must have actor_final_agent*.{npz,pt} + config.yaml)")
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 4, 8],
                        help="Target quantization precisions to sweep (LogHD paper default: 1,2,4,8)")
    parser.add_argument("--flip-probs", nargs="+", type=float,
                        default=[0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2],
                        help="Fine-grained within [0, 0.2] by default -- at higher "
                             "flip rates every method collapses (unrealistic regime, "
                             "not informative). Override with --flip-probs if needed.")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Clean-observation episodes per (bits, flip_prob, trial)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Independent random bit-flip mask draws per (bits, flip_prob) cell to average over")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of greedy argmax -- see "
                             "eval_noise_robustness.py --stochastic and "
                             "docs/sustaingym_noise_clean.md for why this matters; "
                             "apply uniformly across all 5 baselines, not just dnn.")
    parser.add_argument("--device", default="cpu",
                        help="Overrides the 'device' field baked into each checkpoint's "
                             "saved config.yaml -- checkpoints trained on a different "
                             "machine (e.g. multi-GPU) can carry a device ordinal (like "
                             "cuda:1) that doesn't exist here, causing 'invalid device "
                             "ordinal'. Bitflip eval is cheap (tiny MLPs/HDC ops), so cpu "
                             "is the safe default; matches eval_bitflip_robustness_sb3.py "
                             "and _single_agent.py, which already hardcode cpu.")
    parser.add_argument("--out", default="results/figures/bitflip_robustness_two_climates.json",
                        help="JSON path for the per-checkpoint (bits, flip_prob) -> "
                             "mean/std summary (this script writes no plots)")
    args = parser.parse_args()

    ckpt_dirs: list[str] = []
    for d in args.checkpoints:
        g = sorted(glob.glob(d))
        ckpt_dirs.extend(g if g else [d])

    if not ckpt_dirs:
        print("No checkpoint dirs found. Use --checkpoints.")
        return

    bits_list = sorted(args.bits)
    flip_probs = sorted(args.flip_probs)
    all_results: list[tuple[str, str, dict[tuple[int, float], list[float]]]] = []

    json_path = args.out
    summary: dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            summary = json.load(f)
        print(f"[resume] Loaded {len(summary)} existing results from {json_path}")

    for ckpt_dir in _pbar(ckpt_dirs, desc="checkpoints"):
        # climate/method/leaf, same disambiguation convention as eval_noise_robustness.py
        parts = Path(ckpt_dir.rstrip("/")).parts[-3:]
        run_name = "/".join(parts)
        climate = parts[0]
        print(f"\n[{run_name}]")
        try:
            cfg = _load_cfg(ckpt_dir)
        except FileNotFoundError as e:
            print(f"  [skip] {e}")
            continue

        method = Path(ckpt_dir.rstrip("/")).parts[-2]
        label = _label(cfg, method)

        cell_keys = [f"{b}_{p}" for b in bits_list for p in flip_probs]
        if run_name in summary and all(k in summary[run_name]["mean"] for k in cell_keys):
            print("  [skip] already evaluated")
            cached = summary[run_name]
            results = {
                (b, p): [float(cached["mean"][f"{b}_{p}"])]
                for b in bits_list for p in flip_probs
            }
            all_results.append((cached.get("climate", climate), cached["label"], results))
            continue

        results = _eval_run(ckpt_dir, bits_list, flip_probs, args.episodes, args.trials,
                             stochastic=args.stochastic, device=args.device)
        all_results.append((climate, label, results))
        summary[run_name] = {
            "label": label,
            "climate": climate,
            "bits": bits_list,
            "flip_probs": flip_probs,
            "mean": {f"{b}_{p}": float(np.mean(v)) for (b, p), v in results.items()},
            "std":  {f"{b}_{p}": float(np.std(v))  for (b, p), v in results.items()},
        }

    if not all_results:
        print("No runs could be evaluated.")
        return

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

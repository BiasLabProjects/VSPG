"""HDC observation encoders for MiniGrid.

HDCEncoder  — bipolar {-1,+1} random HVs, binding = element-wise multiply.
FHRREncoder — Fourier Holographic Reduced Representations.
              Positions use fractional-power phasors with Gaussian base phases
              (NOT uniform) so that nearby cells share tunable cosine similarity:
                  sim(pos(x), pos(x+Δ)) ≈ exp(−Δ²/(2w²))
              Kernel width w controls the learning/cognition tradeoff:
                  small w  → near-orthogonal positions  (exclusive / cognitive)
                  large w  → correlated positions       (inclusive / learning)
              Default w=1.0 gives sim≈0.61 between adjacent cells.
RFFEncoder  — Random Fourier Features on the flattened obs vector.
              S = cos(W @ flat_obs + b) * sqrt(2/D),  W ~ N(0, 1/σ²)
              Approximates RBF kernel: κ(x,x') ≈ exp(−‖x−x'‖²/(2σ²)).
              σ controls bandwidth (same semantics as FHRR w but for flat obs).
FlatEncoder — raw obs flattening for DNN baselines.

All encoders return:
  encode(obs)         -> np.ndarray[float32, (dim,)]   L2-normalised
  encode_bipolar(obs) -> np.ndarray[int8,    (dim,)]   sign of encode()

Reference:
  Poduval et al. (2026) "Optimal hyperdimensional representation for learning
  and cognitive computation." Frontiers in Artificial Intelligence.
"""

from __future__ import annotations

import math

import numpy as np

_N_OBJ   = 11
_N_COLOR = 6
_N_STATE = 3
_N_DIR   = 4
_UNSEEN  = 0


# ─────────────────────────────────────────────────────────────────────────────
# HDCEncoder  (bipolar, vectorised)
# ─────────────────────────────────────────────────────────────────────────────

class HDCEncoder:
    """Bipolar {-1,+1} HDC encoder — fully vectorised encode().

    Binding  : element-wise multiply.
    Bundling : sum over visible cells, then L2-normalise.
    """

    def __init__(self, dim: int = 1000, obs_view_size: int = 7, seed: int = 42) -> None:
        self.dim = dim
        self.obs_view_size = obs_view_size
        rng = np.random.default_rng(seed)

        def _bip(shape):
            v = np.sign(rng.standard_normal(shape)).astype(np.float32)
            return np.where(v == 0.0, 1.0, v)

        self._Px    = _bip((obs_view_size, dim))
        self._Py    = _bip((obs_view_size, dim))
        self._Obj   = _bip((_N_OBJ,   dim))
        self._Color = _bip((_N_COLOR, dim))
        self._State = _bip((_N_STATE, dim))
        self._Dir   = _bip((_N_DIR,   dim))

        ys, xs = np.mgrid[0:obs_view_size, 0:obs_view_size]
        self._ys = ys.ravel()
        self._xs = xs.ravel()

    def encode(self, obs: dict) -> np.ndarray:
        img = obs["image"]
        flat = img.reshape(-1, 3)
        mask = flat[:, 0] != _UNSEEN

        if mask.any():
            xs = self._xs[mask]
            ys = self._ys[mask]
            cell_hvs = (
                self._Px[xs]
                * self._Py[ys]
                * self._Obj[flat[mask, 0]]
                * self._Color[flat[mask, 1]]
                * self._State[flat[mask, 2]]
            )
            s = cell_hvs.sum(axis=0)
        else:
            s = np.zeros(self.dim, dtype=np.float32)

        s += self._Dir[int(obs["direction"])]
        norm = np.linalg.norm(s)
        return (s / (norm + 1e-8)).astype(np.float32)

    def encode_bipolar(self, obs: dict) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# ─────────────────────────────────────────────────────────────────────────────
# FHRREncoder  (Gaussian phases, tunable kernel width)
# ─────────────────────────────────────────────────────────────────────────────

class FHRREncoder:
    """FHRR encoder with Gaussian base phases and tunable kernel width w.

    Position phasors use fractional-power encoding (Plate 1995, Frady 2022):
        phi_x ~ N(0, 1) / w
        pos(x, y)[k] = exp(i * (x * phi_x[k] + y * phi_y[k]))
    giving cosine similarity:
        sim(pos(x), pos(x+Δ)) ≈ exp(−Δ²/(2w²))   [Gaussian kernel]

    Kernel width w (default 1.0):
        w ≈ 0.1  → near-orthogonal positions (exclusive, good for retrieval)
        w ≈ 1.0  → sim≈0.61 at Δ=1  (intermediate)
        w ≈ 2.0  → sim≈0.88 at Δ=1  (correlated, good for learning)

    All arithmetic stays in float32. dim must be even (cdim = dim//2 complex dims).
    """

    def __init__(
        self,
        dim: int = 1000,
        obs_view_size: int = 7,
        seed: int = 42,
        fhrr_w: float = 1.0,
    ) -> None:
        if dim % 2 != 0:
            raise ValueError(f"FHRREncoder requires even dim, got {dim}")
        self.dim  = dim
        self.cdim = dim // 2
        self.obs_view_size = obs_view_size
        self.fhrr_w = fhrr_w
        rng = np.random.default_rng(seed)

        # Gaussian base phases scaled by 1/w  (see paper Eq. 1)
        def _gauss(shape):
            return (rng.standard_normal(shape) / fhrr_w).astype(np.float32)

        # Uniform phases for discrete features (full ±π range → near-orthogonal
        # across different object/color/state/dir IDs — good for retrieval of symbols)
        def _unif(shape):
            return rng.uniform(-np.pi, np.pi, shape).astype(np.float32)

        self._phi_x = _gauss(self.cdim)                # (cdim,)
        self._phi_y = _gauss(self.cdim)                # (cdim,)
        self._phi_obj   = _unif((_N_OBJ,   self.cdim))
        self._phi_color = _unif((_N_COLOR, self.cdim))
        self._phi_state = _unif((_N_STATE, self.cdim))

        dir_phases = _unif((_N_DIR, self.cdim))
        self._dir_cos = np.cos(dir_phases)             # (4, cdim) float32
        self._dir_sin = np.sin(dir_phases)             # (4, cdim) float32

        ys, xs = np.mgrid[0:obs_view_size, 0:obs_view_size]
        self._ys = ys.ravel().astype(np.float32)
        self._xs = xs.ravel().astype(np.float32)

    def encode(self, obs: dict) -> np.ndarray:
        img  = obs["image"]
        flat = img.reshape(-1, 3)
        mask = flat[:, 0] != _UNSEEN

        s_re = np.zeros(self.cdim, dtype=np.float32)
        s_im = np.zeros(self.cdim, dtype=np.float32)

        if mask.any():
            xs = self._xs[mask]
            ys = self._ys[mask]
            phase = (
                xs[:, None] * self._phi_x[None, :]
                + ys[:, None] * self._phi_y[None, :]
                + self._phi_obj[flat[mask, 0].astype(np.intp)]
                + self._phi_color[flat[mask, 1].astype(np.intp)]
                + self._phi_state[flat[mask, 2].astype(np.intp)]
            )
            s_re = np.cos(phase).sum(axis=0)
            s_im = np.sin(phase).sum(axis=0)

        d = int(obs["direction"])
        s_re += self._dir_cos[d]
        s_im += self._dir_sin[d]

        mag = np.sqrt(s_re * s_re + s_im * s_im)
        mag = np.where(mag < 1e-8, 1.0, mag)
        s_re /= mag
        s_im /= mag

        out = np.empty(self.dim, dtype=np.float32)
        out[0::2] = s_re
        out[1::2] = s_im
        norm = np.linalg.norm(out)
        return out / (norm + 1e-8)

    def encode_bipolar(self, obs: dict) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# ─────────────────────────────────────────────────────────────────────────────
# RFFEncoder  (Random Fourier Features on flat obs)
# ─────────────────────────────────────────────────────────────────────────────

class RFFEncoder:
    """Random Fourier Features encoder — approximates RBF kernel on flat obs.

    S = cos(W @ flat_obs + b) * sqrt(2/D)
    W ~ N(0, 1/σ²),  b ~ U[0, 2π]

    κ(x, x') ≈ exp(−‖x−x'‖²/(2σ²))   [RBF / Gaussian kernel]

    σ controls bandwidth:
        small σ → fast decay → near-orthogonal representations (exclusive)
        large σ → slow decay → correlated representations (inclusive, learning)

    Works on the normalised flat MiniGrid obs (7×7×3 + 1 = 148-dim).
    For continuous flat obs (e.g. SustainGym), pass obs_low/obs_high to normalise
    each feature to [-1, 1] before projection so no feature dominates by scale.
    """

    def __init__(
        self,
        dim: int = 1000,
        obs_view_size: int = 7,
        seed: int = 42,
        rff_sigma: float = 1.0,
        obs_flat_dim: int | None = None,
        obs_low: list[float] | None = None,
        obs_high: list[float] | None = None,
    ) -> None:
        self.dim = dim
        self.obs_view_size = obs_view_size
        self.flat_dim = obs_flat_dim if obs_flat_dim is not None else obs_view_size * obs_view_size * 3 + 1
        rng = np.random.default_rng(seed)

        self._W = (rng.standard_normal((dim, self.flat_dim)) / rff_sigma).astype(np.float32)
        self._b = rng.uniform(0, 2 * math.pi, dim).astype(np.float32)
        self._scale = math.sqrt(2.0 / dim)

        if obs_low is not None and obs_high is not None:
            lo = np.asarray(obs_low, dtype=np.float32)
            hi = np.asarray(obs_high, dtype=np.float32)
            self._norm_center = (lo + hi) / 2.0
            self._norm_scale  = (hi - lo) / 2.0 + 1e-8
        else:
            self._norm_center = None
            self._norm_scale  = None

    def _flat(self, obs) -> np.ndarray:
        if not isinstance(obs, dict):
            x = np.asarray(obs, dtype=np.float32).flatten()
            if self._norm_center is not None:
                x = (x - self._norm_center) / self._norm_scale
            return x
        img = obs["image"].astype(np.float32).flatten()
        img[0::3] /= 10.0
        img[1::3] /= 5.0
        img[2::3] /= 2.0
        return np.concatenate([img, [obs["direction"] / 3.0]]).astype(np.float32)

    def encode(self, obs: dict) -> np.ndarray:
        x = self._flat(obs)
        s = np.cos(self._W @ x + self._b) * self._scale  # (dim,)
        norm = np.linalg.norm(s)
        return (s / (norm + 1e-8)).astype(np.float32)

    def encode_bipolar(self, obs: dict) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# ─────────────────────────────────────────────────────────────────────────────
# FHRRFlatEncoder  (FHRR fractional-power encoding on flat continuous obs)
# ─────────────────────────────────────────────────────────────────────────────

class FHRRFlatEncoder:
    """FHRR encoder generalised to flat continuous observations (classic-gym).

    Generalises FHRREncoder's 2-D position binding to k continuous obs dims
    via fractional-power encoding (Plate 1995, Frady 2022):
        phi_i ~ N(0, 1) / w   (one (cdim,) phase vector per obs dimension i)
        phase = sum_i v_i * phi_i
        s_re, s_im = cos(phase), sin(phase)

    No multi-cell bundling (unlike MiniGrid's per-grid-cell sum) — one obs
    vector in, one binding op, then L2-normalise. dim must be even.
    """

    def __init__(
        self,
        dim: int = 1000,
        seed: int = 42,
        fhrr_w: float = 1.0,
        obs_flat_dim: int = 4,
        obs_low: list[float] | None = None,
        obs_high: list[float] | None = None,
    ) -> None:
        if dim % 2 != 0:
            raise ValueError(f"FHRRFlatEncoder requires even dim, got {dim}")
        self.dim = dim
        self.cdim = dim // 2
        self.flat_dim = obs_flat_dim
        rng = np.random.default_rng(seed)

        self._phi = (rng.standard_normal((obs_flat_dim, self.cdim)) / fhrr_w).astype(np.float32)

        if obs_low is not None and obs_high is not None:
            lo = np.asarray(obs_low, dtype=np.float32)
            hi = np.asarray(obs_high, dtype=np.float32)
            self._norm_center = (lo + hi) / 2.0
            self._norm_scale  = (hi - lo) / 2.0 + 1e-8
        else:
            self._norm_center = None
            self._norm_scale  = None

    def encode(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).flatten()
        if self._norm_center is not None:
            x = (x - self._norm_center) / self._norm_scale

        phase = (x[:, None] * self._phi).sum(axis=0)  # (cdim,)
        s_re = np.cos(phase)
        s_im = np.sin(phase)

        out = np.empty(self.dim, dtype=np.float32)
        out[0::2] = s_re
        out[1::2] = s_im
        norm = np.linalg.norm(out)
        return (out / (norm + 1e-8)).astype(np.float32)

    def encode_bipolar(self, obs) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# ─────────────────────────────────────────────────────────────────────────────
# FlatEncoder  (raw flattening, for MLP baseline)
# ─────────────────────────────────────────────────────────────────────────────

class FlatEncoder:
    """Raw obs flattening for DNN baselines. dim = view² × 3 + 1.

    feature_normalize=False (default) reproduces the exact MLP baseline input
    (channel-rescaled, NOT L2-normalised). Set True to additionally L2-normalise
    the flattened vector — used by the raw-linear-REINFORCE baseline so that
    `tau * dot(s, c_a)` stays a bounded cosine similarity, matching how the
    other fixed-feature encoders feed RealHDPolicy.
    """

    def __init__(self, obs_view_size: int = 7, feature_normalize: bool = False) -> None:
        self.obs_view_size = obs_view_size
        self.feature_normalize = feature_normalize
        self.dim = obs_view_size * obs_view_size * 3 + 1

    def encode(self, obs) -> np.ndarray:
        if not isinstance(obs, dict):
            s = np.asarray(obs, dtype=np.float32).flatten()
            if self.feature_normalize:
                s = s / (np.linalg.norm(s) + 1e-8)
            return s.astype(np.float32)
        img = obs["image"].astype(np.float32).flatten()
        img[0::3] /= 10.0
        img[1::3] /= 5.0
        img[2::3] /= 2.0
        s = np.concatenate([img, [obs["direction"] / 3.0]], dtype=np.float32)
        if self.feature_normalize:
            s = s / (np.linalg.norm(s) + 1e-8)
        return s.astype(np.float32)

    def encode_bipolar(self, obs: dict) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# ─────────────────────────────────────────────────────────────────────────────
# FlatNormEncoder  (range-norm + L2-norm, no projection — linear-VSPG baseline)
# ─────────────────────────────────────────────────────────────────────────────

class FlatNormEncoder:
    """Identity encoder for continuous flat obs: range-normalize then L2-normalize.

    Output dim equals input dim (obs_flat_dim). Used as the encoder for the
    linear-VSPG baseline so the Hebbian C-matrix update runs on raw features
    rather than a high-dimensional projection. Requires obs_low/obs_high.
    """

    def __init__(
        self,
        obs_flat_dim: int,
        obs_low: list[float] | None = None,
        obs_high: list[float] | None = None,
        **kwargs,
    ) -> None:
        self.dim = obs_flat_dim

        if obs_low is not None and obs_high is not None:
            lo = np.asarray(obs_low, dtype=np.float32)
            hi = np.asarray(obs_high, dtype=np.float32)
            self._norm_center = (lo + hi) / 2.0
            self._norm_scale  = (hi - lo) / 2.0 + 1e-8
        else:
            self._norm_center = None
            self._norm_scale  = None

    def encode(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).flatten()
        if self._norm_center is not None:
            x = (x - self._norm_center) / self._norm_scale
        norm = np.linalg.norm(x)
        return (x / (norm + 1e-8)).astype(np.float32)

    def encode_bipolar(self, obs) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# ─────────────────────────────────────────────────────────────────────────────
# GaussianEncoder  (fixed Gaussian projection B ∈ R^{D×d})
# ─────────────────────────────────────────────────────────────────────────────

class GaussianEncoder:
    """Standard fixed Gaussian random projection for flat continuous observations.

    z = B @ x,  B ~ N(0, 1),  then L2-normalised to the unit sphere.

    Use when obs is a 1-D float array (e.g. SustainGym building state).
    Pass obs_low/obs_high to map each feature to [-1, 1] before projection
    so no feature dominates by scale.
    """

    def __init__(
        self,
        dim: int = 1000,
        obs_flat_dim: int = 10,
        seed: int = 42,
        obs_low: list[float] | None = None,
        obs_high: list[float] | None = None,
    ) -> None:
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._W = rng.standard_normal((dim, obs_flat_dim)).astype(np.float32)

        if obs_low is not None and obs_high is not None:
            lo = np.asarray(obs_low, dtype=np.float32)
            hi = np.asarray(obs_high, dtype=np.float32)
            self._norm_center = (lo + hi) / 2.0
            self._norm_scale  = (hi - lo) / 2.0 + 1e-8
        else:
            self._norm_center = None
            self._norm_scale  = None

    def encode(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).flatten()
        if self._norm_center is not None:
            x = (x - self._norm_center) / self._norm_scale
        z = self._W @ x
        norm = np.linalg.norm(z)
        return (z / (norm + 1e-8)).astype(np.float32)

    def encode_bipolar(self, obs) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b


# backward-compat alias
FlatBipolarEncoder = GaussianEncoder


# ─────────────────────────────────────────────────────────────────────────────
# VSAMemoryWrapper  (optional recurrent VSA feature)
# ─────────────────────────────────────────────────────────────────────────────

class VSAMemoryWrapper:
    """Wraps any obs encoder with an optional VSA recurrent memory state.

    At each step the memory is updated:
        m_t = normalize(decay * m_{t-1}[perm] + memory_input_t)

    where perm is a fixed random permutation (shift-register delay operator) and
    memory_input_t = phi_t, or bind(phi_t, action_vec[prev_action]) when
    include_prev_action=True.

    The policy feature z_t is then constructed according to `combine`:
        "none"  → z_t = phi_t                    (old behavior, no change)
        "add"   → z_t = normalize(phi_t + m_t)
        "bind"  → z_t = normalize(phi_t * m_t)   (elementwise product)

    reset() must be called at every episode boundary.
    Default combine="none" reproduces the plain VSPG behavior exactly.
    """

    def __init__(
        self,
        obs_encoder,
        dim: int,
        combine: str = "none",
        decay: float = 1.0,
        include_prev_action: bool = False,
        num_actions: int | None = None,
        seed: int = 0,
    ) -> None:
        self.obs_encoder = obs_encoder
        self.dim = dim
        self.combine = combine
        self.decay = decay
        self.include_prev_action = include_prev_action

        rng = np.random.default_rng(seed)
        self._perm = rng.permutation(dim)

        self._action_vecs: np.ndarray | None = None
        if include_prev_action and num_actions is not None:
            av = rng.standard_normal((num_actions, dim)).astype(np.float32)
            self._action_vecs = av / (np.linalg.norm(av, axis=1, keepdims=True) + 1e-8)

        self._memory = np.zeros(dim, dtype=np.float32)

    def reset(self) -> None:
        self._memory[:] = 0.0

    def step(self, obs, prev_action: int | None = None) -> np.ndarray:
        """Encode obs, update memory, return z_t (the policy feature)."""
        phi = self.obs_encoder.encode(obs)

        mem_input = phi.copy()
        if self.include_prev_action and prev_action is not None and self._action_vecs is not None:
            mem_input = phi * self._action_vecs[prev_action]
            mem_input /= np.linalg.norm(mem_input) + 1e-8

        raw = self.decay * self._memory[self._perm] + mem_input
        self._memory = raw / (np.linalg.norm(raw) + 1e-8)

        if self.combine == "none":
            return phi
        if self.combine == "add":
            z = phi + self._memory
        elif self.combine == "bind":
            z = phi * self._memory
        else:
            raise ValueError(f"Unknown combine mode: {self.combine!r}. Choose: none | add | bind")
        return (z / (np.linalg.norm(z) + 1e-8)).astype(np.float32)

    def encode(self, obs) -> np.ndarray:
        """Encoder-interface alias for step() with no prev_action."""
        return self.step(obs, prev_action=None)

    def encode_bipolar(self, obs) -> np.ndarray:
        v = self.encode(obs)
        b = np.sign(v).astype(np.int8)
        b[b == 0] = 1
        return b

    def encode_episode(self, observations, actions=None) -> np.ndarray:
        """Sequentially encode a full episode; returns Z of shape (T, dim)."""
        self.reset()
        Z = []
        for t, obs in enumerate(observations):
            prev = actions[t - 1] if (actions is not None and t > 0) else None
            Z.append(self.step(obs, prev_action=prev))
        return np.array(Z, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_encoder(cfg) -> "HDCEncoder | FHRREncoder | FHRRFlatEncoder | RFFEncoder | FlatEncoder | GaussianEncoder | FlatNormEncoder":
    enc_type = getattr(cfg, "encoder_type", "bipolar")
    seed     = getattr(cfg, "seed",     42)
    dim      = getattr(cfg, "dim",    1000)
    view     = getattr(cfg, "obs_view_size", 7)
    flat_dim = getattr(cfg, "obs_flat_dim", None)
    obs_low  = getattr(cfg, "obs_low",  None)
    obs_high = getattr(cfg, "obs_high", None)

    if enc_type == "flat_norm":
        if flat_dim is None:
            raise ValueError("encoder_type='flat_norm' requires obs_flat_dim in config")
        return FlatNormEncoder(obs_flat_dim=int(flat_dim), obs_low=obs_low, obs_high=obs_high)
    if enc_type == "gaussian":
        if flat_dim is None:
            raise ValueError("encoder_type='gaussian' requires obs_flat_dim in config")
        return GaussianEncoder(dim=dim, obs_flat_dim=int(flat_dim), seed=seed,
                               obs_low=obs_low, obs_high=obs_high)
    if enc_type == "rff":
        sigma = getattr(cfg, "rff_sigma", 1.0)
        return RFFEncoder(dim=dim, obs_view_size=view, seed=seed, rff_sigma=sigma,
                          obs_flat_dim=flat_dim, obs_low=obs_low, obs_high=obs_high)
    if enc_type == "fhrr":
        w = getattr(cfg, "fhrr_w", 1.0)
        return FHRREncoder(dim=dim, obs_view_size=view, seed=seed, fhrr_w=w)
    if enc_type == "fhrr_flat":
        if flat_dim is None:
            raise ValueError("encoder_type='fhrr_flat' requires obs_flat_dim in config")
        w = getattr(cfg, "fhrr_w", 1.0)
        return FHRRFlatEncoder(dim=dim, seed=seed, fhrr_w=w, obs_flat_dim=int(flat_dim),
                               obs_low=obs_low, obs_high=obs_high)
    if enc_type == "flat":
        feature_normalize = bool(getattr(cfg, "feature_normalize", False))
        return FlatEncoder(obs_view_size=view, feature_normalize=feature_normalize)
    # bipolar: grid encoder for MiniGrid (flat_dim not needed)
    if flat_dim is not None:
        # old configs that set encoder_type=bipolar with flat obs → treat as gaussian
        return GaussianEncoder(dim=dim, obs_flat_dim=int(flat_dim), seed=seed,
                               obs_low=obs_low, obs_high=obs_high)
    return HDCEncoder(dim=dim, obs_view_size=view, seed=seed)

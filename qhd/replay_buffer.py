"""Ring-buffer experience replay for QHD.

Stores (s_hv, action, reward, next_s_hv, done) transitions and supports
uniform random minibatch sampling. Deliberately bare-bones — QHD's own
selling point (Ni et al., GLSVLSI'23, Sec 4.3-4.4) is that it still learns
well with a tiny batch_size/capacity, so the buffer itself has no priority
weighting or n-step logic to keep that comparison clean.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, dim: int, seed: int = 42) -> None:
        self.capacity = capacity
        self._s = np.zeros((capacity, dim), dtype=np.float32)
        self._a = np.zeros(capacity, dtype=np.int64)
        self._r = np.zeros(capacity, dtype=np.float32)
        self._s2 = np.zeros((capacity, dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._ptr = 0
        self._rng = np.random.default_rng(seed)

    def push(self, s_hv: np.ndarray, action: int, reward: float, next_s_hv: np.ndarray, done: bool) -> None:
        i = self._ptr
        self._s[i] = s_hv
        self._a[i] = action
        self._r[i] = reward
        self._s2[i] = next_s_hv
        self._done[i] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return self._s[idx], self._a[idx], self._r[idx], self._s2[idx], self._done[idx]

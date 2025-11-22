from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RNGPool:
    base_seed: int
    counter: int = 0

    def next(self) -> np.random.Generator:
        seed = (self.base_seed + self.counter) % (2**32)
        self.counter += 1
        return np.random.default_rng(seed)

    def fork(self, n: int) -> list[np.random.Generator]:
        return [self.next() for _ in range(n)]


def make_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def reseed_all(generators: Iterable[np.random.Generator], seed: int) -> None:
    for idx, gen in enumerate(generators):
        gen.bit_generator = np.random.PCG64(seed + idx)

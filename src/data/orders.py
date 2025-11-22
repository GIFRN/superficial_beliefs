from __future__ import annotations

from typing import Sequence

import numpy as np


def balanced_order_indices(n: int, orders: Sequence[Sequence[str]], rng: np.random.Generator) -> list[int]:
    if not orders:
        raise ValueError("orders must be non-empty")
    order_ids = list(range(len(orders)))
    reps = (n + len(order_ids) - 1) // len(order_ids)
    pool = (order_ids * reps)[:n]
    rng.shuffle(pool)
    return pool


def assign_orders(n: int, orders: Sequence[Sequence[str]], rng: np.random.Generator) -> tuple[list[int], list[int]]:
    ids_a = balanced_order_indices(n, orders, rng)
    ids_b = balanced_order_indices(n, orders, rng)
    return ids_a, ids_b


def positions_for_order(order: Sequence[str]) -> dict[str, int]:
    return {attr: idx + 1 for idx, attr in enumerate(order)}

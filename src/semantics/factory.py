from __future__ import annotations

from .df_quad_semantics import DFQuadSemantics
from .quadratic_energy_semantics import QuadraticEnergySemantics


def parse_semantics(
    semantics: str,
    conservativeness: float = 1.0,
    *,
    max_iters: int = 3,
    epsilon: float = 0.0,
):
    name = semantics.strip().lower()
    if name == "dfq":
        return DFQuadSemantics(
            max_iters=max_iters,
            epsilon=epsilon,
            conservativeness=conservativeness,
        )
    if name == "qe":
        return QuadraticEnergySemantics(
            max_iters=max_iters,
            epsilon=epsilon,
            conservativeness=conservativeness,
        )
    raise ValueError(f"Unsupported semantics: {semantics}")

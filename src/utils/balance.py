from __future__ import annotations

from collections import Counter
from typing import Iterable


def is_balanced(sequence: Iterable[int], tolerance: float = 0.05) -> bool:
    counts = Counter(sequence)
    if not counts:
        return True
    avg = sum(counts.values()) / len(counts)
    return all(abs(count - avg) <= tolerance * avg for count in counts.values())

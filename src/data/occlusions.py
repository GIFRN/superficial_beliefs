from __future__ import annotations

from typing import Sequence

from .schema import Attribute, Profile


def apply_structural_occlusion(
    profile_a: Profile,
    profile_b: Profile,
    order_a: Sequence[Attribute],
    order_b: Sequence[Attribute],
    manipulation: str,
    attribute_target: Attribute | None,
) -> tuple[Profile, Profile, tuple[Attribute, ...], tuple[Attribute, ...]]:
    """Return effective profiles/orders after applying structural occlusion."""
    if not attribute_target:
        return profile_a, profile_b, tuple(order_a), tuple(order_b)

    if manipulation == "occlude_equalize":
        levels_a = dict(profile_a.levels)
        levels_b = dict(profile_b.levels)
        levels_a[attribute_target] = "Medium"
        levels_b[attribute_target] = "Medium"
        return Profile(levels_a), Profile(levels_b), tuple(order_a), tuple(order_b)

    if manipulation == "occlude_swap":
        levels_a = dict(profile_a.levels)
        levels_b = dict(profile_b.levels)
        if attribute_target in levels_a and attribute_target in levels_b:
            levels_a[attribute_target], levels_b[attribute_target] = (
                levels_b[attribute_target],
                levels_a[attribute_target],
            )
        return Profile(levels_a), Profile(levels_b), tuple(order_a), tuple(order_b)

    if manipulation == "occlude_drop":
        effective_a = tuple(attr for attr in order_a if attr != attribute_target)
        effective_b = tuple(attr for attr in order_b if attr != attribute_target)
        return profile_a, profile_b, effective_a, effective_b

    return profile_a, profile_b, tuple(order_a), tuple(order_b)


def apply_occlusion_to_deltas(
    deltas: dict[Attribute, int],
    manipulation: str,
    attribute_target: Attribute | None,
) -> dict[Attribute, int]:
    """Adjust delta values to reflect prompt-visible occlusions."""
    visible = dict(deltas)
    if not attribute_target:
        return visible

    if manipulation in {"occlude_equalize", "occlude_drop"}:
        visible[attribute_target] = 0
    elif manipulation == "occlude_swap":
        visible[attribute_target] = -visible.get(attribute_target, 0)
    return visible

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _name(name: str, value: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _coordinates(value: Sequence[int], /) -> tuple[int, ...]:
    result = tuple(int(item) for item in value)
    if not result or any(item < 0 for item in result):
        raise ValueError(
            "Span coordinates must be a nonempty sequence of nonnegative integers."
        )
    return result


class BaseSpanId(StrictModule, NonTrainableState):
    """Structural identity of one positive knot span in one patch."""

    patch_id: str = eqx.field(static=True)
    coordinates: tuple[int, ...] = eqx.field(static=True)
    value: str = eqx.field(static=True)

    def __init__(self, patch_id: str, coordinates: Sequence[int], /):
        patch = _name("patch_id", patch_id)
        coordinate_values = _coordinates(coordinates)
        self.patch_id = patch
        self.coordinates = coordinate_values
        self.value = canonical_fingerprint(
            {
                "kind": "iga-base-span",
                "patch": patch,
                "coordinates": list(coordinate_values),
            }
        )

    @property
    def span_id(self) -> str:
        return self.value


class OverlayCellId(StrictModule, NonTrainableState):
    """Structural identity of one integration-overlay cell."""

    overlay_id: str = eqx.field(static=True)
    coordinates: tuple[int, ...] = eqx.field(static=True)
    value: str = eqx.field(static=True)

    def __init__(self, overlay_id: str, coordinates: Sequence[int], /):
        overlay = _name("overlay_id", overlay_id)
        coordinate_values = _coordinates(coordinates)
        self.overlay_id = overlay
        self.coordinates = coordinate_values
        self.value = canonical_fingerprint(
            {
                "kind": "iga-overlay-cell",
                "overlay": overlay,
                "coordinates": list(coordinate_values),
            }
        )

    @property
    def cell_id(self) -> str:
        return self.value


class InterfaceId(StrictModule, NonTrainableState):
    """Order-sensitive structural identity of an oriented patch interface."""

    left_patch_id: str = eqx.field(static=True)
    right_patch_id: str = eqx.field(static=True)
    left_route: tuple[int, int] = eqx.field(static=True)
    right_route: tuple[int, int] = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    value: str = eqx.field(static=True)

    def __init__(
        self,
        left_patch_id: str,
        right_patch_id: str,
        left_route: Sequence[int],
        right_route: Sequence[int],
        /,
        *,
        periodic: bool = False,
    ):
        left = _name("left_patch_id", left_patch_id)
        right = _name("right_patch_id", right_patch_id)
        left_values = tuple(int(item) for item in left_route)
        right_values = tuple(int(item) for item in right_route)
        if len(left_values) != 2 or len(right_values) != 2:
            raise ValueError("Interface routes must each contain an axis and side.")
        if (
            left_values[0] < 0
            or right_values[0] < 0
            or left_values[1] not in (-1, 1)
            or right_values[1] not in (-1, 1)
        ):
            raise ValueError("Interface routes require a nonnegative axis and side +/-1.")
        periodic_ = bool(periodic)
        if left == right and not periodic_:
            raise ValueError("A self-interface must be explicitly periodic.")
        self.left_patch_id = left
        self.right_patch_id = right
        self.left_route = left_values
        self.right_route = right_values
        self.periodic = periodic_
        self.value = canonical_fingerprint(
            {
                "kind": "iga-interface",
                "left_patch": left,
                "right_patch": right,
                "left_route": list(left_values),
                "right_route": list(right_values),
                "periodic": periodic_,
            }
        )

    @property
    def interface_id(self) -> str:
        return self.value


__all__ = ["BaseSpanId", "InterfaceId", "OverlayCellId"]

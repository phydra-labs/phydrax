#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx

from .._strict import StrictModule


CoordinateKind = Literal["scalar", "array", "pytree", "graph"]


class CoordinateSpec(StrictModule):
    """Static schema for one labeled domain coordinate.

    ``event_shape`` describes only unnamed value dimensions. Named sampling axes
    are supplied by a sampling batch and are deliberately not part of this
    schema. Structured PyTree and graph coordinates use ``event_shape=None`` and
    carry their detailed schemas in their owning domain factors.
    """

    event_shape: tuple[int, ...] | None = eqx.field(static=True)
    kind: CoordinateKind = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    dtype: str | None = eqx.field(static=True)

    def __init__(
        self,
        event_shape: tuple[int, ...] | None,
        /,
        *,
        kind: CoordinateKind,
        differentiable: bool,
        dtype: str | None = "float",
    ):
        if event_shape is not None:
            shape = tuple(int(size) for size in event_shape)
            if any(size <= 0 for size in shape):
                raise ValueError("Coordinate event dimensions must be positive.")
        else:
            shape = None
        if kind not in ("scalar", "array", "pytree", "graph"):
            raise ValueError(f"Unknown coordinate kind {kind!r}.")
        if kind == "scalar" and shape != ():
            raise ValueError("Scalar coordinates must have event_shape=().")
        if kind in ("pytree", "graph") and shape is not None:
            raise ValueError(f"{kind} coordinates must have event_shape=None.")
        self.event_shape = shape
        self.kind = kind
        self.differentiable = bool(differentiable)
        self.dtype = None if dtype is None else str(dtype)

    @property
    def event_size(self) -> int:
        """Number of scalar values in one dense coordinate event."""
        if self.event_shape is None:
            raise TypeError(
                f"{self.kind} coordinates do not have a dense scalar event size."
            )
        return math.prod(self.event_shape)

    def compatible(self, other: object, /) -> bool:
        """Return whether two coordinates have the same evaluation schema."""
        return (
            isinstance(other, CoordinateSpec)
            and self.event_shape == other.event_shape
            and self.kind == other.kind
            and self.differentiable == other.differentiable
            and self.dtype == other.dtype
        )


__all__ = ["CoordinateKind", "CoordinateSpec"]

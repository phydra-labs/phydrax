#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .typing import parse


type CoordinateMeasure = Literal[
    "cartesian", "cylindrical-radial", "axisymmetric-rz", "spherical-radial"
]
type CoordinateFrame = Literal["reference", "current"]


@final
class CoordinateMeasurePlan(StrictModule, NonTrainableState):
    """Rotational reduction of an explicitly bound Cartesian coordinate measure.

    ``radial_axis`` identifies the coordinate component, never an inferred shape.
    ``frame`` declares which configuration the supplied coordinates and base
    measure inhabit. ``weight`` multiplies only the rotational factor: callers
    supply quadrature weights with their Cartesian metric determinant already
    included. A full 3D ``DeformedMeasurePlan`` instead starts from these complete
    reference weights and applies its Jacobian/Nanson ratio once; do not weight
    the result again by the current radius.

    Radius zero is a regular axis, with zero boundary area, not invalid geometry.
    Volume capacity rules sampling the axis must be refused by their owner; a
    consistent interior quadrature followed by P1 row-sum lumping retains capacity.
    Invalid dynamic geometry is returned as NaN, without a host callback or sync.
    """

    kind: CoordinateMeasure = eqx.field(static=True)
    radial_axis: int | None = eqx.field(static=True)
    frame: CoordinateFrame = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CoordinateMeasure,
        /,
        *,
        radial_axis: int | None = None,
        frame: CoordinateFrame = "reference",
    ) -> None:
        selected = parse(kind, CoordinateMeasure, "kind")
        binding = parse(frame, CoordinateFrame, "frame")
        match selected:
            case "cartesian":
                if radial_axis is not None:
                    raise ValueError("Cartesian measures have no radial axis.")
            case "cylindrical-radial" | "axisymmetric-rz" | "spherical-radial":
                if not isinstance(radial_axis, int) or isinstance(radial_axis, bool):
                    raise TypeError("Rotational measures require an integer radial_axis.")
                if radial_axis < 0:
                    raise ValueError("radial_axis must be nonnegative.")
        self.kind = selected
        self.radial_axis = radial_axis
        self.frame = binding
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coordinate-measure",
                "measure": selected,
                "radial_axis": radial_axis,
                "frame": binding,
            }
        )

    def validate_coordinates(self, coordinates: ArrayLike, /) -> None:
        """Host-only geometry admission; dynamic evaluation uses ``factor``."""
        points = np.asarray(coordinates)
        self._validate_layout(points.shape)
        if not np.all(np.isfinite(points)):
            raise ValueError("Coordinate measure requires finite coordinates.")
        if self.radial_axis is not None and np.any(points[..., self.radial_axis] < 0.0):
            raise ValueError("Coordinate measure requires nonnegative radius.")

    def _validate_layout(self, shape: tuple[int, ...], /) -> None:
        if not shape:
            raise ValueError("Coordinate points require a trailing coordinate axis.")
        if self.radial_axis is not None and self.radial_axis >= shape[-1]:
            raise ValueError("radial_axis is outside the coordinate components.")

    def factor(self, coordinates: ArrayLike, /, *, frame: CoordinateFrame) -> Array:
        """Return 1, 2πr, or 4πr², preserving coordinate differentiation."""
        binding = parse(frame, CoordinateFrame, "frame")
        if binding != self.frame:
            raise ValueError("Coordinate measure and coordinate frame do not match.")
        points = jnp.asarray(coordinates)
        self._validate_layout(points.shape)
        match self.kind:
            case "cartesian":
                return jnp.ones(points.shape[:-1], dtype=points.dtype)
            case "cylindrical-radial" | "axisymmetric-rz" | "spherical-radial":
                if self.radial_axis is None:
                    raise RuntimeError("Rotational measure lost its radial identity.")
                radius = points[..., self.radial_axis]
                valid = jnp.all(jnp.isfinite(points), axis=-1) & (radius >= 0.0)
                factor = (
                    4.0 * jnp.pi * radius * radius
                    if self.kind == "spherical-radial"
                    else 2.0 * jnp.pi * radius
                )
                return jnp.where(valid, factor, jnp.nan)

    def weight(
        self,
        coordinates: ArrayLike,
        base_weights: ArrayLike,
        /,
        *,
        frame: CoordinateFrame,
    ) -> Array:
        """Apply the rotational factor once to metric-weighted quadrature."""
        weights = jnp.asarray(base_weights)
        if self.kind == "cartesian":
            if parse(frame, CoordinateFrame, "frame") != self.frame:
                raise ValueError("Coordinate measure and coordinate frame do not match.")
            return weights
        return weights * self.factor(coordinates, frame=frame)


__all__ = ["CoordinateFrame", "CoordinateMeasure", "CoordinateMeasurePlan"]

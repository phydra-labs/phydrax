#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PreparedTensorGrid
from ...geometry import RigidFrame


PlaneTopology = Literal["finite-window", "periodic-cell"]


def _complex_field_values(
    name: str,
    values: ArrayLike,
    shape: tuple[int, ...],
    /,
) -> Array:
    array = jnp.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype, jnp.bool_
    ):
        raise TypeError(f"{name} must be numeric.")
    return array.astype(jnp.result_type(array.dtype, jnp.complex64))


def _angular_frequency(value: ArrayLike, /) -> Array:
    supplied = jnp.asarray(value)
    if supplied.shape != ():
        raise ValueError("angular_frequency must be a scalar.")
    if (
        jnp.iscomplexobj(supplied)
        or not jnp.issubdtype(supplied.dtype, jnp.number)
        or jnp.issubdtype(supplied.dtype, jnp.bool_)
    ):
        raise TypeError("angular_frequency must be real numeric data.")
    frequency = supplied.astype(jnp.result_type(supplied.dtype, jnp.float32))
    return eqx.error_if(
        frequency,
        (~jnp.isfinite(frequency)) | (frequency <= 0.0),
        "angular_frequency must be finite and strictly positive.",
    )


def _longitudinal_coordinate(value: ArrayLike, /) -> Array:
    supplied = jnp.asarray(value)
    if supplied.shape != ():
        raise ValueError("longitudinal_coordinate must be a scalar.")
    if (
        jnp.iscomplexobj(supplied)
        or not jnp.issubdtype(supplied.dtype, jnp.number)
        or jnp.issubdtype(supplied.dtype, jnp.bool_)
    ):
        raise TypeError("longitudinal_coordinate must be real numeric data.")
    coordinate = supplied.astype(jnp.result_type(supplied.dtype, jnp.float32))
    return eqx.error_if(
        coordinate,
        ~jnp.isfinite(coordinate),
        "longitudinal_coordinate must be finite.",
    )


class PlaneFieldSpace(StrictModule, NonTrainableState):
    """One physical transverse plane over an existing prepared tensor grid.

    The tensor grid remains the sole owner of transverse coordinates and physical
    quadrature weights. The rigid frame embeds local ``(axis0, axis1, 0)`` points
    into three-dimensional space; its first two columns are the tangential basis
    and its third column is the oriented longitudinal normal.
    """

    grid: PreparedTensorGrid
    frame: RigidFrame
    topology: PlaneTopology = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        frame: RigidFrame,
        topology: PlaneTopology,
        /,
        *,
        space_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        if len(grid.shape) != 2:
            raise ValueError(
                "A plane field space requires an exactly two-dimensional grid."
            )
        if not isinstance(frame, RigidFrame) or frame.dimension != 3:
            raise ValueError("frame must be a three-dimensional RigidFrame.")
        if topology not in ("finite-window", "periodic-cell"):
            raise ValueError("topology must be 'finite-window' or 'periodic-cell'.")
        periodic = tuple(bool(axis.periodic) for axis in grid.axes)
        required_periodicity = (topology == "periodic-cell",) * 2
        if periodic != required_periodicity:
            raise ValueError(
                f"{topology!r} requires grid-axis periodicity {required_periodicity}; "
                f"got {periodic}."
            )
        generated = canonical_fingerprint(
            {
                "kind": "plane-field-space",
                "grid": grid.prepared_id,
                "frame": array_tree_fingerprint((frame.rotation, frame.translation)),
                "topology": topology,
            }
        )
        identifier = generated if space_id is None else str(space_id)
        if not identifier:
            raise ValueError("space_id must be non-empty.")
        self.grid = grid
        self.frame = frame
        self.topology = topology
        self.space_id = identifier

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape  # type: ignore[return-value]

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def coordinate_axes(self) -> tuple[Array, Array]:
        axes = self.grid.primary_entity_layout.coordinates_by_axis
        return axes  # type: ignore[return-value]

    @property
    def transverse_coordinates(self) -> Array:
        """Return local transverse coordinates with shape ``space.shape + (2,)``."""
        return self.grid.points.reshape(self.shape + (2,))

    @property
    def local_points(self) -> Array:
        transverse = self.transverse_coordinates
        return jnp.concatenate(
            (transverse, jnp.zeros(self.shape + (1,), dtype=transverse.dtype)),
            axis=-1,
        )

    @property
    def world_points(self) -> Array:
        return self.frame.apply(self.local_points)

    @property
    def transverse_basis(self) -> Array:
        return self.frame.rotation[:, :2]

    @property
    def normal(self) -> Array:
        return self.frame.rotation[:, 2]

    @property
    def area_weights(self) -> Array:
        return self.grid.quadrature_weights


class ScalarPlaneField(StrictModule):
    """A monochromatic complex scalar field sampled on one plane."""

    space: PlaneFieldSpace
    values: Array
    angular_frequency: Array
    longitudinal_coordinate: Array

    def __init__(
        self,
        space: PlaneFieldSpace,
        values: ArrayLike,
        angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        self.space = space
        self.values = _complex_field_values("values", values, space.shape)
        self.angular_frequency = _angular_frequency(angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)


class TangentialPlaneField(StrictModule):
    """A monochromatic Jones field in the plane frame's tangential basis."""

    space: PlaneFieldSpace
    values: Array
    angular_frequency: Array
    longitudinal_coordinate: Array

    def __init__(
        self,
        space: PlaneFieldSpace,
        values: ArrayLike,
        angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        self.space = space
        self.values = _complex_field_values("values", values, space.shape + (2,))
        self.angular_frequency = _angular_frequency(angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)


class IntensityPlane(StrictModule):
    """A real, nonnegative intensity density sampled on one physical plane."""

    space: PlaneFieldSpace
    values: Array
    angular_frequency: Array
    longitudinal_coordinate: Array

    def __init__(
        self,
        space: PlaneFieldSpace,
        values: ArrayLike,
        angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        array = jnp.asarray(values)
        if array.shape != space.shape:
            raise ValueError(f"values must have shape {space.shape}; got {array.shape}.")
        if (
            jnp.iscomplexobj(array)
            or not jnp.issubdtype(array.dtype, jnp.number)
            or jnp.issubdtype(array.dtype, jnp.bool_)
        ):
            raise TypeError("Intensity values must be real numeric values.")
        intensity = array.astype(jnp.result_type(array.dtype, jnp.float32))
        intensity = eqx.error_if(
            intensity,
            jnp.any(~jnp.isfinite(intensity)) | jnp.any(intensity < 0.0),
            "Intensity values must be finite and nonnegative.",
        )
        self.space = space
        self.values = intensity
        self.angular_frequency = _angular_frequency(angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)


__all__ = [
    "IntensityPlane",
    "PlaneFieldSpace",
    "PlaneTopology",
    "ScalarPlaneField",
    "TangentialPlaneField",
]

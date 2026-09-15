#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._topology import TensorTopology


def _axis(value: int, dimension: int, /) -> int:
    axis = int(value)
    if axis < 0 or axis >= dimension:
        raise ValueError(f"axis must lie in [0, {dimension}).")
    return axis


class LatticeBoundaryPhasePlan(StrictModule, NonTrainableState):
    """Static twisted-periodic phases and open-boundary masks on a tensor lattice.

    ``shift(values, axis, displacement)[x]`` reads ``values[x + displacement]``.
    A positive (negative) periodic crossing contributes the declared phase (its
    inverse).  Crossings of a nonperiodic boundary contribute zero.  Lattice axes
    are the leading axes of ``values`` and flat site ordering is C order.
    """

    topology: TensorTopology
    phases: Array
    inverse_phases: Array
    site_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    maximum_displacement: int = eqx.field(static=True)
    complex_phases: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: TensorTopology,
        phases: ArrayLike,
        /,
        *,
        maximum_displacement: int = 1,
    ):
        if not isinstance(topology, TensorTopology):
            raise TypeError("topology must be TensorTopology.")
        values = np.asarray(phases)
        dimension = len(topology.axis_sizes)
        if values.shape != (dimension,):
            raise ValueError(f"phases must have shape {(dimension,)}.")
        values = values.astype(np.result_type(values.dtype, np.complex64), copy=False)
        if np.any(~np.isfinite(values)):
            raise ValueError("Boundary phases must be finite.")
        periodic = np.asarray(topology.periodic, dtype=bool)
        if np.any(np.abs(np.abs(values[periodic]) - 1.0) > 1.0e-7):
            raise ValueError("Periodic boundary phases must have unit modulus.")
        if np.any(np.abs(values[~periodic] - 1.0) > 1.0e-7):
            raise ValueError("Nonperiodic axes must declare the neutral phase one.")
        maximum = int(maximum_displacement)
        if maximum < 1 or maximum > 64:
            raise ValueError("maximum_displacement must lie in [1, 64].")
        self.topology = topology
        self.phases = jnp.asarray(values)
        self.inverse_phases = jnp.conj(self.phases)
        self.complex_phases = bool(np.any(np.abs(np.imag(values)) > 1.0e-14))
        self.site_count = prod(topology.axis_sizes)
        self.dimension = dimension
        self.maximum_displacement = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boundary-phase-plan",
                "topology": topology.topology_id,
                "phases": array_tree_fingerprint(values),
                "maximum_displacement": maximum,
                "shift_convention": "output-x-reads-input-x-plus-displacement",
                "site_order": "c-order",
            }
        )

    def phase_factors(
        self,
        axis: int,
        displacement: int,
        /,
        *,
        dtype: Any | None = None,
    ) -> Array:
        """Return the lattice-shaped phase/mask for one statically bounded shift."""
        axis_ = _axis(axis, self.dimension)
        offset = int(displacement)
        if abs(offset) > self.maximum_displacement:
            raise ValueError("displacement exceeds the prepared fixed-shape bound.")
        size = self.topology.axis_sizes[axis_]
        coordinates = jnp.arange(size, dtype=jnp.int32)
        destination = coordinates + offset
        if self.topology.periodic[axis_]:
            crossings = jnp.floor_divide(destination, size)
            factors = self.phases[axis_] ** crossings
        else:
            factors = ((destination >= 0) & (destination < size)).astype(
                self.phases.dtype
            )
        shape = [1] * self.dimension
        shape[axis_] = size
        result = jnp.broadcast_to(factors.reshape(tuple(shape)), self.topology.axis_sizes)
        return result if dtype is None else result.astype(dtype)

    def shift(self, values: ArrayLike, axis: int, displacement: int, /) -> Array:
        """Apply one fixed-shape boundary-aware lattice shift."""
        field = jnp.asarray(values)
        if field.shape[: self.dimension] != self.topology.axis_sizes:
            raise ValueError(
                "values must begin with the topology axis sizes "
                f"{self.topology.axis_sizes}; got {field.shape}."
            )
        axis_ = _axis(axis, self.dimension)
        offset = int(displacement)
        raw_factors = self.phase_factors(axis_, offset)
        if jnp.issubdtype(field.dtype, jnp.complexfloating):
            factors = raw_factors.astype(field.dtype)
        elif self.complex_phases:
            factors = raw_factors
        else:
            factors = jnp.real(raw_factors).astype(field.dtype)
        payload_axes = (1,) * (field.ndim - self.dimension)
        shifted = jnp.roll(field, -offset, axis=axis_)
        return shifted * factors.reshape(factors.shape + payload_axes)

    def forward(self, values: ArrayLike, axis: int, /) -> Array:
        return self.shift(values, axis, 1)

    def backward(self, values: ArrayLike, axis: int, /) -> Array:
        return self.shift(values, axis, -1)


class CheckerboardEntityLayout(StrictModule, NonTrainableState):
    """Certified nearest-neighbor bipartition in C-order tensor-lattice sites."""

    topology: TensorTopology
    site_parity: Array
    even_indices: Array
    odd_indices: Array
    origin_parity: int = eqx.field(static=True)
    even_count: int = eqx.field(static=True)
    odd_count: int = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, topology: TensorTopology, /, *, origin_parity: int = 0):
        if not isinstance(topology, TensorTopology):
            raise TypeError("topology must be TensorTopology.")
        origin = int(origin_parity)
        if origin not in (0, 1):
            raise ValueError("origin_parity must be zero or one.")
        active = np.asarray(topology.active_mask, dtype=bool)
        if not np.all(active):
            raise ValueError(
                "Checkerboard certification requires the complete tensor lattice."
            )
        for size, periodic in zip(topology.axis_sizes, topology.periodic, strict=True):
            if periodic and size % 2:
                raise ValueError("A periodic checkerboard axis must have even extent.")
        coordinates = np.indices(topology.axis_sizes, dtype=np.int64)
        parity = (np.sum(coordinates, axis=0) + origin) % 2
        flat = parity.reshape((-1,)).astype(np.int32)
        even = np.flatnonzero(flat == 0).astype(np.int32)
        odd = np.flatnonzero(flat == 1).astype(np.int32)
        self.topology = topology
        self.site_parity = jnp.asarray(flat)
        self.even_indices = jnp.asarray(even)
        self.odd_indices = jnp.asarray(odd)
        self.origin_parity = origin
        self.even_count = int(even.size)
        self.odd_count = int(odd.size)
        self.site_count = int(flat.size)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "checkerboard-entity-layout",
                "topology": topology.topology_id,
                "origin_parity": origin,
                "site_parity": array_tree_fingerprint(flat),
                "site_order": "c-order",
                "certification": "nearest-neighbor-bipartite",
            }
        )

    def parity_at(self, coordinates: ArrayLike, /) -> Array:
        points = jnp.asarray(coordinates)
        if points.shape[-1:] != (len(self.topology.axis_sizes),):
            raise ValueError(
                "coordinates must have one trailing component per lattice axis."
            )
        return (jnp.sum(points.astype(jnp.int32), axis=-1) + self.origin_parity) & 1

    def _flat_field(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        dimension = len(self.topology.axis_sizes)
        if field.shape[:dimension] == self.topology.axis_sizes:
            return field.reshape((self.site_count,) + field.shape[dimension:])
        if field.ndim >= 1 and field.shape[0] == self.site_count:
            return field
        raise ValueError(
            "values must be lattice-shaped or begin with the flat site count."
        )

    def gather(self, values: ArrayLike, parity: int, /) -> Array:
        parity_ = int(parity)
        if parity_ not in (0, 1):
            raise ValueError("parity must be zero or one.")
        indices = self.even_indices if parity_ == 0 else self.odd_indices
        return self._flat_field(values)[indices]

    def gather_even(self, values: ArrayLike, /) -> Array:
        return self.gather(values, 0)

    def gather_odd(self, values: ArrayLike, /) -> Array:
        return self.gather(values, 1)

    def split(self, values: ArrayLike, /) -> tuple[Array, Array]:
        field = self._flat_field(values)
        return field[self.even_indices], field[self.odd_indices]

    def merge(self, even_values: ArrayLike, odd_values: ArrayLike, /) -> Array:
        even = jnp.asarray(even_values)
        odd = jnp.asarray(odd_values)
        if even.ndim < 1 or even.shape[0] != self.even_count:
            raise ValueError("even_values must begin with even_count.")
        if odd.ndim < 1 or odd.shape[0] != self.odd_count:
            raise ValueError("odd_values must begin with odd_count.")
        if even.shape[1:] != odd.shape[1:]:
            raise ValueError("Even and odd checkerboard payload shapes must agree.")
        flat = jnp.zeros(
            (self.site_count,) + even.shape[1:], dtype=jnp.result_type(even, odd)
        )
        flat = flat.at[self.even_indices].set(even)
        flat = flat.at[self.odd_indices].set(odd)
        return flat.reshape(self.topology.axis_sizes + even.shape[1:])


__all__ = ["CheckerboardEntityLayout", "LatticeBoundaryPhasePlan"]

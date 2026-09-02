#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState


_UINT64_MAX = np.iinfo(np.uint64).max
_MAX_CODE_BITS = 63


class MortonEncoding(NonTrainableState, StrictModule):
    """Morton codes and explicit domain evidence for a point batch."""

    codes: jax.Array
    integer_coordinates: jax.Array
    coordinates: jax.Array
    finite: jax.Array
    in_domain: jax.Array
    successful: jax.Array


class MortonCellGeometry(NonTrainableState, StrictModule):
    """Physical geometry of Morton prefix cells."""

    lower: jax.Array
    upper: jax.Array
    center: jax.Array
    half_width: jax.Array


class MortonAddressPlan(StrictModule):
    """Canonical dyadic addressing over a finite Cartesian box."""

    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        maximum_depth: int,
        *,
        periodic_axes: Sequence[bool] | None = None,
    ) -> None:
        lower_tuple = tuple(float(value) for value in lower)
        upper_tuple = tuple(float(value) for value in upper)
        if len(lower_tuple) not in (1, 2, 3):
            raise ValueError("Morton addressing supports dimensions 1, 2, and 3.")
        if len(upper_tuple) != len(lower_tuple):
            raise ValueError("Morton lower and upper bounds must have equal length.")
        lower_array = np.asarray(lower_tuple, dtype=float)
        upper_array = np.asarray(upper_tuple, dtype=float)
        if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
            raise ValueError("Morton bounds must be finite.")
        if np.any(upper_array <= lower_array):
            raise ValueError("Every Morton upper bound must exceed its lower bound.")
        dimension = len(lower_tuple)
        depth = int(maximum_depth)
        maximum_supported = _MAX_CODE_BITS // dimension
        if depth < 1 or depth > maximum_supported:
            raise ValueError(
                f"maximum_depth must lie in [1, {maximum_supported}] for "
                f"dimension {dimension}."
            )
        if periodic_axes is None:
            periodic_tuple = (False,) * dimension
        else:
            periodic_tuple = tuple(bool(value) for value in periodic_axes)
            if len(periodic_tuple) != dimension:
                raise ValueError("periodic_axes must match the Morton dimension.")
        object.__setattr__(self, "lower", lower_tuple)
        object.__setattr__(self, "upper", upper_tuple)
        object.__setattr__(self, "periodic_axes", periodic_tuple)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "maximum_depth", depth)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "morton-address-plan",
                    "lower": list(lower_tuple),
                    "upper": list(upper_tuple),
                    "periodic_axes": list(periodic_tuple),
                    "maximum_depth": depth,
                }
            ),
        )

    @property
    def resolution(self) -> int:
        return 1 << self.maximum_depth

    def encode(self, points: jax.Array) -> MortonEncoding:
        values = jnp.asarray(points)
        if values.ndim < 1 or values.shape[-1] != self.dimension:
            raise ValueError(
                f"points must have trailing dimension {self.dimension}; got "
                f"shape {values.shape}."
            )
        lower = jnp.asarray(self.lower, dtype=values.dtype)
        upper = jnp.asarray(self.upper, dtype=values.dtype)
        extent = upper - lower
        periodic = jnp.asarray(self.periodic_axes, dtype=bool)
        finite_components = jnp.isfinite(values)
        finite = jnp.all(finite_components, axis=-1)
        safe = jnp.where(finite_components, values, lower)
        wrapped = jnp.where(periodic, lower + jnp.mod(safe - lower, extent), safe)
        component_in_domain = periodic | ((wrapped >= lower) & (wrapped < upper))
        in_domain = finite & jnp.all(component_in_domain, axis=-1)
        normalized = (wrapped - lower) / extent
        resolution = jnp.asarray(self.resolution, dtype=normalized.dtype)
        integer = jnp.floor(normalized * resolution).astype(jnp.int64)
        integer = jnp.clip(integer, 0, self.resolution - 1)
        integer = jnp.where(in_domain[..., None], integer, 0)
        codes = morton_encode_integer(integer, self.maximum_depth)
        codes = jnp.where(in_domain, codes, jnp.asarray(0, dtype=jnp.uint64))
        codes = jax.lax.stop_gradient(codes)
        integer = jax.lax.stop_gradient(integer)
        return MortonEncoding(
            codes=codes,
            integer_coordinates=integer,
            coordinates=wrapped,
            finite=finite,
            in_domain=in_domain,
            successful=jnp.all(in_domain),
        )

    def decode(self, codes: jax.Array) -> jax.Array:
        return morton_decode_integer(codes, self.dimension, self.maximum_depth)

    def prefix(self, codes: jax.Array, level: int | jax.Array) -> jax.Array:
        levels = jnp.asarray(level, dtype=jnp.int32)
        shift = self.dimension * (self.maximum_depth - levels)
        return jnp.asarray(codes, dtype=jnp.uint64) >> shift.astype(jnp.uint64)

    def parent(
        self, prefixes: jax.Array, levels: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        level_values = jnp.asarray(levels, dtype=jnp.int32)
        valid = level_values > 0
        parent_prefix = jnp.asarray(prefixes, dtype=jnp.uint64) >> self.dimension
        return jnp.where(valid, parent_prefix, 0), jnp.maximum(level_values - 1, 0)

    def children(self, prefixes: jax.Array) -> jax.Array:
        prefix_values = jnp.asarray(prefixes, dtype=jnp.uint64)
        digits = jnp.arange(1 << self.dimension, dtype=jnp.uint64)
        return (prefix_values[..., None] << self.dimension) | digits

    def descendant_interval(
        self, prefixes: jax.Array, levels: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        level_values = jnp.asarray(levels, dtype=jnp.int32)
        shift = self.dimension * (self.maximum_depth - level_values)
        start = jnp.asarray(prefixes, dtype=jnp.uint64) << shift.astype(jnp.uint64)
        end = (jnp.asarray(prefixes, dtype=jnp.uint64) + 1) << shift.astype(jnp.uint64)
        return start, end

    def cell_geometry(self, prefixes: jax.Array, levels: jax.Array) -> MortonCellGeometry:
        level_values = jnp.asarray(levels, dtype=jnp.int32)
        start, _ = self.descendant_interval(prefixes, level_values)
        integer_lower = self.decode(start)
        lower = jnp.asarray(self.lower)
        upper = jnp.asarray(self.upper)
        extent = upper - lower
        resolution = jnp.asarray(self.resolution, dtype=extent.dtype)
        physical_lower = lower + extent * integer_lower.astype(extent.dtype) / resolution
        widths = extent * jnp.exp2(-level_values.astype(extent.dtype))[..., None]
        physical_upper = physical_lower + widths
        return MortonCellGeometry(
            lower=physical_lower,
            upper=physical_upper,
            center=0.5 * (physical_lower + physical_upper),
            half_width=0.5 * widths,
        )


def morton_encode_integer(integer_coordinates: jax.Array, depth: int) -> jax.Array:
    """Interleave low-to-high coordinate bits into canonical Morton codes."""
    coordinates = jnp.asarray(integer_coordinates, dtype=jnp.uint64)
    if coordinates.ndim < 1 or coordinates.shape[-1] not in (1, 2, 3):
        raise ValueError("integer_coordinates must have trailing dimension 1, 2, or 3.")
    dimension = coordinates.shape[-1]
    depth_value = int(depth)
    if depth_value < 1 or dimension * depth_value > _MAX_CODE_BITS:
        raise ValueError("The requested Morton depth exceeds the uint64 code budget.")
    code = jnp.zeros(coordinates.shape[:-1], dtype=jnp.uint64)
    for bit in range(depth_value):
        for axis in range(dimension):
            value = (coordinates[..., axis] >> bit) & jnp.uint64(1)
            code = code | (value << (dimension * bit + axis))
    return code


def morton_decode_integer(codes: jax.Array, dimension: int, depth: int) -> jax.Array:
    """Deinterleave canonical Morton codes into integer coordinates."""
    dimension_value = int(dimension)
    depth_value = int(depth)
    if dimension_value not in (1, 2, 3):
        raise ValueError("Morton decoding supports dimensions 1, 2, and 3.")
    if depth_value < 1 or dimension_value * depth_value > _MAX_CODE_BITS:
        raise ValueError("The requested Morton depth exceeds the uint64 code budget.")
    code_values = jnp.asarray(codes, dtype=jnp.uint64)
    coordinates = []
    for axis in range(dimension_value):
        coordinate = jnp.zeros(code_values.shape, dtype=jnp.uint64)
        for bit in range(depth_value):
            value = (code_values >> (dimension_value * bit + axis)) & jnp.uint64(1)
            coordinate = coordinate | (value << bit)
        coordinates.append(coordinate.astype(jnp.int64))
    return jnp.stack(coordinates, axis=-1)


def _morton_encode_host(coordinates: tuple[int, ...], dimension: int, depth: int) -> int:
    code = 0
    for bit in range(depth):
        for axis in range(dimension):
            code |= ((int(coordinates[axis]) >> bit) & 1) << (dimension * bit + axis)
    return code


def _morton_decode_host(code: int, dimension: int, depth: int) -> tuple[int, ...]:
    coordinates = []
    for axis in range(dimension):
        coordinate = 0
        for bit in range(depth):
            coordinate |= ((int(code) >> (dimension * bit + axis)) & 1) << bit
        coordinates.append(coordinate)
    return tuple(coordinates)


def canonical_morton_order(
    codes: jax.Array, stable_ids: jax.Array, valid: jax.Array
) -> jax.Array:
    """Return a deterministic valid-first Morton ordering."""
    code_values = jnp.asarray(codes, dtype=jnp.uint64)
    id_values = jnp.asarray(stable_ids)
    valid_values = jnp.asarray(valid, dtype=bool)
    if code_values.ndim != 1 or id_values.shape != code_values.shape:
        raise ValueError("codes and stable_ids must be rank-one arrays with equal shape.")
    if valid_values.shape != code_values.shape:
        raise ValueError("valid must match codes.")
    return jnp.lexsort(
        (
            id_values,
            code_values,
            (~valid_values).astype(jnp.int32),
        )
    ).astype(jnp.int32)


__all__ = [
    "MortonAddressPlan",
    "MortonCellGeometry",
    "MortonEncoding",
    "canonical_morton_order",
    "morton_decode_integer",
    "morton_encode_integer",
]

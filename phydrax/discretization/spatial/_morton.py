#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState


if TYPE_CHECKING:
    from phydrax.domain import Domain, PeriodicIdentification


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


class _MortonPointOrder(NonTrainableState, StrictModule):
    """Canonical fixed-capacity ordering shared by Morton realizations."""

    encoding: MortonEncoding
    active: jax.Array
    stable_ids: jax.Array
    sorted_codes: jax.Array
    sorted_stable_ids: jax.Array
    sorted_active: jax.Array
    storage_to_logical: jax.Array
    logical_to_storage: jax.Array
    active_count: jax.Array
    invalid_points: jax.Array
    stable_ids_unique: jax.Array


class MortonCellGeometry(NonTrainableState, StrictModule):
    """Physical geometry of Morton prefix cells."""

    lower: jax.Array
    upper: jax.Array
    center: jax.Array
    half_width: jax.Array


class MortonAddressPlan(StrictModule):
    """Canonical dyadic addressing over a finite Cartesian box.

    Periodic axes are addressed half-open, ``[lower, upper)``: encoding wraps a
    coordinate into the cell, so a cloud on a periodic address carries one
    representative per identified seam orbit and never the duplicate upper-face
    node of the closed fundamental box. Minimum-image, wrapping, and chart
    arithmetic consume only this numerical descriptor.

    A raw address (``coordinates`` and ``identifications`` ``None``) is a
    domain-less numerical box. A domain-bound address comes from
    `from_periodic_identifications`, which records the canonical coordinate of
    every point axis and, on periodic axes, the revision of its
    `PeriodicIdentification`; both enter ``plan_id``, so equal numerical boxes
    over different canonical seams are distinct addresses.
    """

    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    coordinates: tuple[tuple[str, int | None], ...] | None = eqx.field(static=True)
    identifications: tuple[str | None, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        maximum_depth: int,
        *,
        periodic_axes: Sequence[bool] | None = None,
        coordinates: Sequence[tuple[str, int | None]] | None = None,
        identifications: Sequence[str | None] | None = None,
    ) -> None:
        lower_tuple = tuple(float(value) for value in lower)
        upper_tuple = tuple(float(value) for value in upper)
        if len(lower_tuple) not in (1, 2, 3):
            raise ValueError("Morton addressing supports dimensions 1, 2, and 3.")
        if len(upper_tuple) != len(lower_tuple):
            raise ValueError("Morton lower and upper bounds must have equal length.")
        lower_array = np.asarray(lower_tuple, dtype=np.float64)
        upper_array = np.asarray(upper_tuple, dtype=np.float64)
        if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
            raise ValueError("Morton bounds must be finite.")
        if np.any(upper_array <= lower_array):
            raise ValueError("Every Morton upper bound must exceed its lower bound.")
        dimension = len(lower_tuple)
        depth = int(maximum_depth)
        maximum_supported = _MAX_CODE_BITS // dimension
        if depth < 1 or depth > maximum_supported:
            raise ValueError(
                f"maximum_depth must lie in [1, {maximum_supported}] for dimension {dimension}."
            )
        if periodic_axes is None:
            periodic_tuple = (False,) * dimension
        else:
            periodic_tuple = tuple(bool(value) for value in periodic_axes)
            if len(periodic_tuple) != dimension:
                raise ValueError("periodic_axes must match the Morton dimension.")
        binding = _address_binding(coordinates, identifications, periodic_tuple)
        content: dict[str, object] = {
            "kind": "morton-address-plan",
            "lower": list(lower_tuple),
            "upper": list(upper_tuple),
            "periodic_axes": list(periodic_tuple),
            "maximum_depth": depth,
        }
        if binding is not None:
            content["coordinates"] = [list(coordinate) for coordinate in binding[0]]
            content["identifications"] = list(binding[1])
        self.lower = lower_tuple
        self.upper = upper_tuple
        self.periodic_axes = periodic_tuple
        self.dimension = dimension
        self.maximum_depth = depth
        self.plan_id = canonical_fingerprint(content)
        self.coordinates = None if binding is None else binding[0]
        self.identifications = None if binding is None else binding[1]

    @classmethod
    def from_periodic_identifications(
        cls,
        identifications: Sequence[PeriodicIdentification],
        /,
        *,
        maximum_depth: int,
        coordinates: Sequence[tuple[str, int | None]] | None = None,
    ) -> MortonAddressPlan:
        """Derive the numerical address of a fundamental box and its seams.

        Every identification must belong to one fundamental domain. Point axis
        ``i`` carries the canonical coordinate ``coordinates[i]`` (a
        ``(label, component)`` pair; ``component`` is ``None`` for a
        `ScalarInterval` factor); a single-label domain binds its coordinate
        components in order by default. Bounds come from the Cartesian factor
        faces, and exactly the identified axes are periodic with period
        ``identification.period``. A coordinate identified twice, an
        identification whose coordinate is not a point axis, or a factor without
        Cartesian faces is refused.
        """
        from phydrax.domain import PeriodicIdentification

        seams = tuple(identifications)
        if not seams:
            raise ValueError(
                "A derived address needs at least one PeriodicIdentification."
            )
        if any(not isinstance(seam, PeriodicIdentification) for seam in seams):
            raise TypeError("identifications must be PeriodicIdentification values.")
        domain = seams[0].domain
        if any(not seam.domain.same_support(domain) for seam in seams[1:]):
            raise ValueError(
                "Periodic identifications must share one fundamental domain."
            )
        axes, lower, upper = _domain_point_coordinates(domain, coordinates)
        revisions: list[str | None] = [None] * len(axes)
        for seam in seams:
            coordinate = (seam.label, seam.component)
            if coordinate not in axes:
                raise ValueError(
                    f"Identified coordinate {coordinate!r} is not a point axis of {axes!r}."
                )
            axis = axes.index(coordinate)
            if revisions[axis] is not None:
                raise ValueError(
                    f"Coordinate {coordinate!r} is identified more than once."
                )
            if (seam.lower, seam.upper) != (lower[axis], upper[axis]):
                raise ValueError(
                    f"Identification of {coordinate!r} disagrees with its factor bounds."
                )
            revisions[axis] = seam.revision
        return cls(
            lower,
            upper,
            maximum_depth,
            periodic_axes=tuple(revision is not None for revision in revisions),
            coordinates=axes,
            identifications=tuple(revisions),
        )

    @property
    def resolution(self) -> int:
        return 1 << self.maximum_depth

    def encode(self, points: jax.Array) -> MortonEncoding:
        values = jnp.asarray(points)
        if values.ndim < 1 or values.shape[-1] != self.dimension:
            raise ValueError(
                f"points must have trailing dimension {self.dimension}; got shape {values.shape}."
            )
        lower = jnp.asarray(self.lower, dtype=values.dtype)
        upper = jnp.asarray(self.upper, dtype=values.dtype)
        extent = upper - lower
        periodic = jnp.asarray(self.periodic_axes, dtype=jnp.bool_)
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


def morton_encode_integer_host(integer_coordinates: np.ndarray, depth: int) -> np.ndarray:
    """Host NumPy form of `morton_encode_integer` for immutable preparation."""
    coordinates = np.asarray(integer_coordinates).astype(np.uint64)
    if coordinates.ndim < 1 or coordinates.shape[-1] not in (1, 2, 3):
        raise ValueError("integer_coordinates must have trailing dimension 1, 2, or 3.")
    dimension = coordinates.shape[-1]
    depth_value = operator.index(depth)
    if depth_value < 1 or dimension * depth_value > _MAX_CODE_BITS:
        raise ValueError("The requested Morton depth exceeds the uint64 code budget.")
    code = np.zeros(coordinates.shape[:-1], dtype=np.uint64)
    one = np.uint64(1)
    for bit in range(depth_value):
        for axis in range(dimension):
            value = (coordinates[..., axis] >> np.uint64(bit)) & one
            code |= value << np.uint64(dimension * bit + axis)
    return code


def morton_decode_integer_host(
    codes: np.ndarray, dimension: int, depth: int
) -> np.ndarray:
    """Host NumPy form of `morton_decode_integer` for immutable preparation."""
    dimension_value = operator.index(dimension)
    depth_value = operator.index(depth)
    if dimension_value not in (1, 2, 3):
        raise ValueError("Morton decoding supports dimensions 1, 2, and 3.")
    if depth_value < 1 or dimension_value * depth_value > _MAX_CODE_BITS:
        raise ValueError("The requested Morton depth exceeds the uint64 code budget.")
    code_values = np.asarray(codes).astype(np.uint64)
    one = np.uint64(1)
    coordinates = []
    for axis in range(dimension_value):
        coordinate = np.zeros(code_values.shape, dtype=np.uint64)
        for bit in range(depth_value):
            value = (code_values >> np.uint64(dimension_value * bit + axis)) & one
            coordinate |= value << np.uint64(bit)
        coordinates.append(coordinate.astype(np.int64))
    return np.stack(coordinates, axis=-1)


def hilbert_encode_integer(integer_coordinates: jax.Array, depth: int) -> jax.Array:
    """Encode integer coordinates as canonical Hilbert-curve indices.

    Uses Skilling's transpose construction (AIP Conf. Proc. 707, 2004): an
    inverse-undo pass and Gray encoding produce the transposed index, whose bits
    are interleaved most-significant first with axis 0 leading. Consecutive
    indices are face-adjacent grid cells, which Morton order does not guarantee.
    """
    coordinates = jnp.asarray(integer_coordinates, dtype=jnp.uint64)
    if coordinates.ndim < 1 or coordinates.shape[-1] not in (1, 2, 3):
        raise ValueError("integer_coordinates must have trailing dimension 1, 2, or 3.")
    dimension = coordinates.shape[-1]
    depth_value = operator.index(depth)
    if depth_value < 1 or dimension * depth_value > _MAX_CODE_BITS:
        raise ValueError("The requested Hilbert depth exceeds the uint64 code budget.")
    axes = [coordinates[..., axis] for axis in range(dimension)]
    for bit in range(depth_value - 1, 0, -1):
        high = jnp.uint64(1 << bit)
        low = jnp.uint64((1 << bit) - 1)
        for axis in range(dimension):
            exchange = (axes[0] ^ axes[axis]) & low
            inverted = (axes[axis] & high) != 0
            first = jnp.where(inverted, axes[0] ^ low, axes[0] ^ exchange)
            if axis:
                axes[axis] = jnp.where(inverted, axes[axis], axes[axis] ^ exchange)
            axes[0] = first
    for axis in range(1, dimension):
        axes[axis] = axes[axis] ^ axes[axis - 1]
    flips = jnp.zeros(coordinates.shape[:-1], dtype=jnp.uint64)
    for bit in range(depth_value - 1, 0, -1):
        high = jnp.uint64(1 << bit)
        flips = jnp.where(
            (axes[-1] & high) != 0, flips ^ jnp.uint64((1 << bit) - 1), flips
        )
    axes = [value ^ flips for value in axes]
    code = jnp.zeros(coordinates.shape[:-1], dtype=jnp.uint64)
    for bit in range(depth_value - 1, -1, -1):
        for axis in range(dimension):
            code = (code << jnp.uint64(1)) | ((axes[axis] >> jnp.uint64(bit)) & 1)
    return code


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
    valid_values = jnp.asarray(valid, dtype=jnp.bool_)
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


def _canonical_morton_point_order(
    address_plan: MortonAddressPlan,
    points: jax.Array,
    *,
    point_capacity: int,
    active_mask: jax.Array | None = None,
    stable_ids: jax.Array | None = None,
) -> _MortonPointOrder:
    """Validate, encode, and deterministically order one fixed-capacity point set."""
    positions = jnp.asarray(points)
    capacity = int(point_capacity)
    expected_shape = (capacity, address_plan.dimension)
    if positions.shape != expected_shape:
        raise ValueError(
            f"points must have shape {expected_shape}; got {positions.shape}."
        )
    if active_mask is None:
        active = jnp.ones((capacity,), dtype=jnp.bool_)
    else:
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        if active.shape != (capacity,):
            raise ValueError("active_mask must match point_capacity.")
    if stable_ids is None:
        identifiers = jnp.arange(capacity, dtype=jnp.int64)
    else:
        identifiers = jnp.asarray(stable_ids)
        if identifiers.shape != (capacity,):
            raise ValueError("stable_ids must match point_capacity.")
        if not jnp.issubdtype(identifiers.dtype, jnp.integer):
            raise TypeError("stable_ids must have integer dtype.")

    encoding = address_plan.encode(positions)
    valid = active & encoding.in_domain
    order = canonical_morton_order(encoding.codes, identifiers, valid)
    inverse = jnp.zeros_like(order).at[order].set(jnp.arange(capacity, dtype=jnp.int32))
    identifier_order = jnp.lexsort((identifiers, (~active).astype(jnp.int32)))
    identifiers_by_id = identifiers[identifier_order]
    active_by_id = active[identifier_order]
    duplicate_id = (
        active_by_id[1:]
        & active_by_id[:-1]
        & (identifiers_by_id[1:] == identifiers_by_id[:-1])
    )
    return _MortonPointOrder(
        encoding=encoding,
        active=active,
        stable_ids=identifiers,
        sorted_codes=encoding.codes[order],
        sorted_stable_ids=identifiers[order],
        sorted_active=valid[order],
        storage_to_logical=order,
        logical_to_storage=inverse,
        active_count=jnp.sum(valid, dtype=jnp.int32),
        invalid_points=jnp.sum(active & ~encoding.in_domain, dtype=jnp.int32),
        stable_ids_unique=~jnp.any(duplicate_id),
    )


def _address_binding(
    coordinates: Sequence[tuple[str, int | None]] | None,
    identifications: Sequence[str | None] | None,
    periodic: tuple[bool, ...],
    /,
) -> tuple[tuple[tuple[str, int | None], ...], tuple[str | None, ...]] | None:
    """Validate the canonical coordinate and seam revision of every address axis."""
    if coordinates is None and identifications is None:
        return None
    if coordinates is None or identifications is None:
        raise ValueError("Address coordinates and identifications are declared together.")
    axes: list[tuple[str, int | None]] = []
    for coordinate in coordinates:
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            raise ValueError("Address coordinates must be (label, component) pairs.")
        label, component = coordinate
        if not isinstance(label, str) or not label:
            raise ValueError("Address coordinate labels must be non-empty strings.")
        if component is not None and (
            isinstance(component, bool) or not isinstance(component, int) or component < 0
        ):
            raise ValueError("Address coordinate components must be nonnegative or None.")
        axes.append((label, component))
    revisions = tuple(identifications)
    if len(axes) != len(periodic) or len(revisions) != len(periodic):
        raise ValueError("Address coordinates must bind every Morton axis once.")
    if len(set(axes)) != len(axes):
        raise ValueError("Address coordinates must be distinct.")
    for revision, periodic_axis in zip(revisions, periodic, strict=True):
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ValueError("Identification revisions must be non-empty strings.")
        if (revision is not None) != periodic_axis:
            raise ValueError(
                "Exactly the periodic axes carry an identification revision."
            )
    return tuple(axes), revisions


def _domain_point_coordinates(
    domain: Domain, coordinates: Sequence[tuple[str, int | None]] | None, /
) -> tuple[tuple[tuple[str, int | None], ...], tuple[float, ...], tuple[float, ...]]:
    """Bind point-array axes to Cartesian coordinates of ``domain`` and their bounds.

    ``None`` binds the components of a single-label domain in order. Geometry
    coordinates canonicalize a one-dimensional component to ``0`` as
    `PeriodicIdentification` does; factors without Cartesian faces are refused.
    """
    from phydrax.domain import AbstractGeometry, ScalarInterval

    if coordinates is None:
        if len(domain.labels) != 1:
            raise ValueError(
                f"Domain labels {domain.labels!r} need explicit point coordinates."
            )
        (label,) = domain.labels
        factor = domain.factor(label)
        requested: tuple[tuple[str, int | None], ...] = (
            tuple((label, axis) for axis in range(factor.spatial_dim))
            if isinstance(factor, AbstractGeometry)
            else ((label, None),)
        )
    else:
        requested = tuple(coordinates)
    axes: list[tuple[str, int | None]] = []
    lower: list[float] = []
    upper: list[float] = []
    for coordinate in requested:
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            raise ValueError("Point coordinates must be (label, component) pairs.")
        label, component = coordinate
        if not isinstance(label, str) or label not in domain.labels:
            raise ValueError(f"{label!r} is not a coordinate of {domain.labels!r}.")
        factor = domain.factor(label)
        if isinstance(factor, ScalarInterval):
            if component is not None:
                raise ValueError(f"Scalar coordinate {label!r} takes component None.")
            bounds = (float(factor.fixed("start")), float(factor.fixed("end")))
        elif isinstance(factor, AbstractGeometry):
            dimension = factor.spatial_dim
            if component is None and dimension == 1:
                component = 0
            if (
                component is None
                or isinstance(component, bool)
                or not isinstance(component, int)
                or not 0 <= component < dimension
            ):
                raise ValueError(
                    f"Coordinate {label!r} needs a component in [0, {dimension})."
                )
            face_lower, face_upper = factor.coordinate_face_bounds()
            bounds = (
                float(np.asarray(face_lower)[component]),
                float(np.asarray(face_upper)[component]),
            )
        else:
            raise TypeError(
                f"Point coordinate {label!r} names a {type(factor).__name__}, not a "
                "ScalarInterval or Cartesian geometry factor."
            )
        axes.append((label, component))
        lower.append(bounds[0])
        upper.append(bounds[1])
    if len(set(axes)) != len(axes):
        raise ValueError("Point coordinates must be distinct.")
    return tuple(axes), tuple(lower), tuple(upper)


__all__ = [
    "MortonAddressPlan",
    "MortonCellGeometry",
    "MortonEncoding",
    "canonical_morton_order",
    "hilbert_encode_integer",
    "morton_decode_integer",
    "morton_decode_integer_host",
    "morton_encode_integer",
    "morton_encode_integer_host",
]

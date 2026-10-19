#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import operator
from collections.abc import Sequence
from types import ModuleType
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.typing import ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.typing import (
    as_host_array,
    checked,
    Dim,
    HostBool,
    HostFloat,
    HostInt64,
    HostInteger,
    Scalar,
    Scope,
    VariadicDim,
)


if TYPE_CHECKING:
    from phydrax import domain as imported_phydrax_domain
    from phydrax.domain import Domain


def _domain_owner() -> ModuleType:
    # Domain → discretization topology → spatial Morton → Domain.
    from phydrax import domain

    return domain


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


class MortonPointOrder(NonTrainableState, StrictModule):
    """Deterministic fixed-capacity ordering shared by Morton realizations."""

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


class MortonPointDims(VariadicDim):
    """Leading dimensions of one batch of Morton-addressed points."""


class MortonCoordinateDim(Dim):
    """Coordinate components declared by a Morton address plan."""


@final
class MortonHostEncoding(NonTrainableState, StrictModule):
    """NumPy Morton codes, wrapped coordinates, and domain evidence.

    Codes use uint64 and integer coordinates use int64. Coordinates retain the
    input floating dtype; nonfinite or out-of-domain points carry zero addresses.
    """

    __strict_contract__ = True

    codes: HostInteger[MortonPointDims]
    integer_coordinates: HostInt64[MortonPointDims, MortonCoordinateDim]
    coordinates: HostFloat[MortonPointDims, MortonCoordinateDim]
    finite: HostBool[MortonPointDims]
    in_domain: HostBool[MortonPointDims]
    successful: HostBool[Scalar]


class _MortonAddressValues(StrictModule):
    codes: jax.Array | np.ndarray
    integer_coordinates: jax.Array | np.ndarray
    coordinates: jax.Array | np.ndarray
    finite: jax.Array | np.ndarray
    in_domain: jax.Array | np.ndarray
    successful: jax.Array | np.ndarray


class MortonAddressPlan(StrictModule):
    """Deterministic dyadic addressing over a finite Cartesian box.

    Periodic axes are addressed half-open, ``[lower, upper)``: encoding wraps a
    coordinate into the cell, so a cloud on a periodic address carries one
    representative per identified seam orbit and never the duplicate upper-face
    node of the closed fundamental box. Minimum-image, wrapping, and chart
    arithmetic consume only this numerical descriptor.

    A raw address (``coordinates`` and ``identifications`` ``None``) is a
    domain-less numerical box. A domain-bound address comes from
    `from_periodic_identifications`, which records the deterministic coordinate of
    every point axis and, on periodic axes, the revision of its
    `PeriodicIdentification`; both enter ``plan_id``, so equal numerical boxes
    over different deterministic seams are distinct addresses.
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
            raise ValueError(
                f"MortonAddressPlan.lower must support dimensions 1, 2, and 3; got {type(lower).__name__}"
            )
        if len(upper_tuple) != len(lower_tuple):
            raise ValueError(
                f"MortonAddressPlan.lower must have equal lower and upper extents; got {type(lower).__name__}"
            )
        lower_array = np.asarray(lower_tuple, dtype=np.float64)
        upper_array = np.asarray(upper_tuple, dtype=np.float64)
        if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
            raise ValueError(
                f"MortonAddressPlan.lower must be finite; got {type(lower).__name__}"
            )
        if np.any(upper_array <= lower_array):
            raise ValueError(
                f"MortonAddressPlan.lower must exceed the corresponding lower bound on every axis; got {type(lower).__name__}"
            )
        dimension = len(lower_tuple)
        depth = int(maximum_depth)
        maximum_supported = _MAX_CODE_BITS // dimension
        if depth < 1 or depth > maximum_supported:
            raise ValueError(
                f"MortonAddressPlan.maximum_depth must lie in [1, {maximum_supported}] for dimension {dimension}; got {maximum_depth!r}"
            )
        if periodic_axes is None:
            periodic_tuple = (False,) * dimension
        else:
            periodic_tuple = tuple(bool(value) for value in periodic_axes)
            if len(periodic_tuple) != dimension:
                raise ValueError(
                    f"MortonAddressPlan.periodic_axes must match the Morton dimension; got {type(periodic_axes).__name__}"
                )
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
        identifications: Sequence[imported_phydrax_domain.PeriodicIdentification],
        /,
        *,
        maximum_depth: int,
        coordinates: Sequence[tuple[str, int | None]] | None = None,
    ) -> MortonAddressPlan:
        """Derive the numerical address of a fundamental box and its seams.

        Every identification must belong to one fundamental domain. Point axis
        ``i`` carries the deterministic coordinate ``coordinates[i]`` (a
        ``(label, component)`` pair; ``component`` is ``None`` for a
        `ScalarInterval` factor); a single-label domain binds its coordinate
        components in order by default. Bounds come from the Cartesian factor
        faces, and exactly the identified axes are periodic with period
        ``identification.period``. A coordinate identified twice, an
        identification whose coordinate is not a point axis, or a factor without
        Cartesian faces is refused.
        """

        seams = tuple(identifications)
        if not seams:
            raise ValueError(
                f"MortonAddressPlan.from_periodic_identifications.identifications must provide at least one PeriodicIdentification; got {type(identifications).__name__}"
            )
        if any(
            not isinstance(seam, _domain_owner().PeriodicIdentification) for seam in seams
        ):
            raise TypeError(
                f"MortonAddressPlan.from_periodic_identifications.identifications must be PeriodicIdentification values; got {type(identifications).__name__}"
            )
        domain = seams[0].domain
        if any(not seam.domain.same_support(domain) for seam in seams[1:]):
            raise ValueError(
                f"MortonAddressPlan.from_periodic_identifications.identifications must share one fundamental domain; got {type(identifications).__name__}"
            )
        axes, lower, upper = domain_point_coordinates(domain, coordinates)
        revisions: list[str | None] = [None] * len(axes)
        for seam in seams:
            coordinate = (seam.label, seam.component)
            if coordinate not in axes:
                raise ValueError(
                    f"MortonAddressPlan.from_periodic_identifications.identifications must bind an axis in the declared point coordinates; got {type(identifications).__name__}; context {coordinate!r}, {axes!r}"
                )
            axis = axes.index(coordinate)
            if revisions[axis] is not None:
                raise ValueError(
                    f"MortonAddressPlan.from_periodic_identifications.identifications must have one identification per coordinate; got {type(identifications).__name__}; context {coordinate!r}"
                )
            if (seam.lower, seam.upper) != (lower[axis], upper[axis]):
                raise ValueError(
                    f"MortonAddressPlan.from_periodic_identifications.identifications must match the declared factor bounds; got {type(identifications).__name__}; context {coordinate!r}"
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

    def encode(self, points: ArrayLike) -> MortonEncoding:
        values = _morton_address_values(self, points, jnp)
        return MortonEncoding(
            codes=jax.lax.stop_gradient(jnp.asarray(values.codes)),
            integer_coordinates=jax.lax.stop_gradient(
                jnp.asarray(values.integer_coordinates)
            ),
            coordinates=jnp.asarray(values.coordinates),
            finite=jnp.asarray(values.finite),
            in_domain=jnp.asarray(values.in_domain),
            successful=jnp.asarray(values.successful),
        )

    @checked
    def encode_host(
        self, points: HostFloat[MortonPointDims, MortonCoordinateDim]
    ) -> MortonHostEncoding:
        """Encode NumPy floating points without dispatching or publishing device data."""
        # NumPy warnings from masked/overflowing intermediates are not additional
        # refusals: the shared finite/domain evidence and JAX integer conversion
        # semantics determine every published field.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            values = _morton_address_values(self, points, np)
        scope = Scope()
        return MortonHostEncoding(
            codes=as_host_array(
                values.codes,
                HostInteger[MortonPointDims],
                "MortonHostEncoding.codes",
                scope=scope,
            ),
            integer_coordinates=as_host_array(
                values.integer_coordinates,
                HostInt64[MortonPointDims, MortonCoordinateDim],
                "MortonHostEncoding.integer_coordinates",
                scope=scope,
            ),
            coordinates=as_host_array(
                values.coordinates,
                HostFloat[MortonPointDims, MortonCoordinateDim],
                "MortonHostEncoding.coordinates",
                scope=scope,
            ),
            finite=as_host_array(
                values.finite,
                HostBool[MortonPointDims],
                "MortonHostEncoding.finite",
                scope=scope,
            ),
            in_domain=as_host_array(
                values.in_domain,
                HostBool[MortonPointDims],
                "MortonHostEncoding.in_domain",
                scope=scope,
            ),
            successful=as_host_array(
                values.successful,
                HostBool[Scalar],
                "MortonHostEncoding.successful",
                scope=scope,
            ),
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


def _morton_host_indices(scaled: np.ndarray) -> np.ndarray:
    """Convert floored coordinates with JAX's saturating int64 semantics."""
    upper = scaled >= np.float64(2**63)
    lower = scaled <= np.float64(-(2**63))
    representable = np.isfinite(scaled) & ~upper & ~lower
    integer = np.where(representable, scaled, np.asarray(0, dtype=scaled.dtype)).astype(
        np.int64
    )
    np.copyto(integer, np.int64(np.iinfo(np.int64).max), where=upper)
    np.copyto(integer, np.int64(np.iinfo(np.int64).min), where=lower)
    return integer


def _morton_address_values(
    plan: MortonAddressPlan, points: ArrayLike, xp: ModuleType
) -> _MortonAddressValues:
    values = xp.asarray(points)
    if values.ndim < 1 or values.shape[-1] != plan.dimension:
        raise ValueError(
            f"MortonAddressPlan.points must have trailing dimension {plan.dimension}; got shape {values.shape}"
        )
    lower = xp.broadcast_to(xp.asarray(plan.lower, dtype=values.dtype), values.shape)
    upper = xp.broadcast_to(xp.asarray(plan.upper, dtype=values.dtype), values.shape)
    extent = upper - lower
    periodic = xp.broadcast_to(
        xp.asarray(plan.periodic_axes, dtype=xp.bool_), values.shape
    )
    finite_components = xp.isfinite(values)
    finite = xp.all(finite_components, axis=-1)
    safe = xp.where(finite_components, values, lower)
    wrapped = xp.where(periodic, lower + xp.mod(safe - lower, extent), safe)
    component_in_domain = periodic | ((wrapped >= lower) & (wrapped < upper))
    in_domain = finite & xp.all(component_in_domain, axis=-1)
    normalized = (wrapped - lower) / extent
    resolution = xp.asarray(plan.resolution, dtype=normalized.dtype)
    scaled = xp.floor(normalized * resolution)
    integer = _morton_host_indices(scaled) if xp is np else scaled.astype(jnp.int64)
    integer = xp.clip(integer, 0, plan.resolution - 1)
    integer = xp.where(in_domain[..., None], integer, 0)
    codes = (
        _morton_encode_kernel(integer, plan.maximum_depth, np)
        if xp is np
        else morton_encode_integer(integer, plan.maximum_depth)
    )
    codes = xp.where(in_domain, codes, xp.asarray(0, dtype=xp.uint64))
    return _MortonAddressValues(
        codes=codes,
        integer_coordinates=integer,
        coordinates=wrapped,
        finite=xp.asarray(finite),
        in_domain=xp.asarray(in_domain),
        successful=xp.asarray(xp.all(in_domain)),
    )


def _morton_encode_kernel(
    integer_coordinates: ArrayLike, depth: int, xp: ModuleType
) -> jax.Array | np.ndarray:
    coordinates = xp.asarray(integer_coordinates, dtype=xp.uint64)
    if coordinates.ndim < 1 or coordinates.shape[-1] not in (1, 2, 3):
        raise ValueError(
            f"morton.morton_encode_kernel.integer_coordinates must have trailing dimension 1, 2, or 3; got {coordinates.shape!r}"
        )
    dimension = coordinates.shape[-1]
    if depth < 1 or dimension * depth > _MAX_CODE_BITS:
        raise ValueError(
            f"morton.morton_encode_kernel.depth must fit the uint64 code budget for the declared dimension; got {depth!r}"
        )
    code = xp.zeros(coordinates.shape[:-1], dtype=xp.uint64)
    for bit in range(depth):
        for axis in range(dimension):
            value = (coordinates[..., axis] >> xp.uint64(bit)) & xp.uint64(1)
            code = code | (value << xp.uint64(dimension * bit + axis))
    return code


_morton_encode_device = jax.jit(_morton_encode_kernel, static_argnames=("depth", "xp"))


def morton_encode_integer(integer_coordinates: ArrayLike, depth: int) -> jax.Array:
    """Interleave coordinate bits, preparing host inputs before device publication."""
    match integer_coordinates:
        case jax.Array() | Tracer():
            return jnp.asarray(
                _morton_encode_device(integer_coordinates, int(depth), jnp)
            )
        case _:
            return jax.device_put(
                morton_encode_integer_host(np.asarray(integer_coordinates), int(depth))
            )


def morton_decode_integer(codes: ArrayLike, dimension: int, depth: int) -> jax.Array:
    """Deinterleave deterministic Morton codes into integer coordinates."""
    dimension_value = int(dimension)
    depth_value = int(depth)
    if dimension_value not in (1, 2, 3):
        raise ValueError(
            f"morton.morton_decode_integer.codes must support dimensions 1, 2, and 3; got {type(codes).__name__}"
        )
    if depth_value < 1 or dimension_value * depth_value > _MAX_CODE_BITS:
        raise ValueError(
            f"morton.morton_decode_integer.depth must fit the uint64 code budget for the declared dimension; got {depth!r}"
        )
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
    return np.asarray(
        _morton_encode_kernel(integer_coordinates, operator.index(depth), np)
    )


def morton_decode_integer_host(
    codes: np.ndarray, dimension: int, depth: int
) -> np.ndarray:
    """Host NumPy form of `morton_decode_integer` for immutable preparation."""
    dimension_value = operator.index(dimension)
    depth_value = operator.index(depth)
    if dimension_value not in (1, 2, 3):
        raise ValueError(
            f"morton.morton_decode_integer_host.codes must support dimensions 1, 2, and 3; got {type(codes).__name__}"
        )
    if depth_value < 1 or dimension_value * depth_value > _MAX_CODE_BITS:
        raise ValueError(
            f"morton.morton_decode_integer_host.depth must fit the uint64 code budget for the declared dimension; got {depth!r}"
        )
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


def hilbert_encode_integer(integer_coordinates: ArrayLike, depth: int) -> jax.Array:
    """Encode integer coordinates as deterministic Hilbert-curve indices.

    Uses Skilling's transpose construction (AIP Conf. Proc. 707, 2004): an
    inverse-undo pass and Gray encoding produce the transposed index, whose bits
    are interleaved most-significant first with axis 0 leading. Consecutive
    indices are face-adjacent grid cells, which Morton order does not guarantee.
    """
    coordinates = jnp.asarray(integer_coordinates, dtype=jnp.uint64)
    if coordinates.ndim < 1 or coordinates.shape[-1] not in (1, 2, 3):
        raise ValueError(
            f"morton.hilbert_encode_integer.integer_coordinates must have trailing dimension 1, 2, or 3; got {coordinates.shape!r}"
        )
    dimension = coordinates.shape[-1]
    depth_value = operator.index(depth)
    if depth_value < 1 or dimension * depth_value > _MAX_CODE_BITS:
        raise ValueError(
            f"morton.hilbert_encode_integer.depth must fit the uint64 code budget for the declared dimension; got {depth!r}"
        )
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


def morton_encode_host(coordinates: tuple[int, ...], dimension: int, depth: int) -> int:
    code = 0
    for bit in range(depth):
        for axis in range(dimension):
            code |= ((int(coordinates[axis]) >> bit) & 1) << (dimension * bit + axis)
    return code


def morton_decode_host(code: int, dimension: int, depth: int) -> tuple[int, ...]:
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
        raise ValueError(
            f"morton.canonical_morton_order.codes must be rank-one arrays with equal shape; got {id_values.shape!r}"
        )
    if valid_values.shape != code_values.shape:
        raise ValueError(
            f"morton.canonical_morton_order.codes must match codes; got {valid_values.shape!r}"
        )
    return jnp.lexsort(
        (
            id_values,
            code_values,
            (~valid_values).astype(jnp.int32),
        )
    ).astype(jnp.int32)


@checked
def canonical_morton_point_order(
    address_plan: MortonAddressPlan,
    points: ArrayLike,
    *,
    point_capacity: int,
    active_mask: ArrayLike | None = None,
    stable_ids: ArrayLike | None = None,
) -> MortonPointOrder:
    """Validate, encode, and deterministically order one fixed-capacity point set."""
    positions = jnp.asarray(points)
    capacity = int(point_capacity)
    expected_shape = (capacity, address_plan.dimension)
    if positions.shape != expected_shape:
        raise ValueError(
            f"morton.canonical_morton_point_order.points must have shape {expected_shape}; got {positions.shape}"
        )
    if active_mask is None:
        active = jnp.ones((capacity,), dtype=jnp.bool_)
    else:
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        if active.shape != (capacity,):
            raise ValueError(
                f"morton.canonical_morton_point_order.point_capacity must match point_capacity; got {active.shape!r}"
            )
    if stable_ids is None:
        identifiers = jnp.arange(capacity, dtype=jnp.int64)
    else:
        identifiers = jnp.asarray(stable_ids)
        if identifiers.shape != (capacity,):
            raise ValueError(
                f"morton.canonical_morton_point_order.point_capacity must match point_capacity; got {identifiers.shape!r}"
            )
        if not jnp.issubdtype(identifiers.dtype, jnp.integer):
            raise TypeError(
                f"morton.canonical_morton_point_order.stable_ids must have integer dtype; got {type(stable_ids).__name__}"
            )

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
    return MortonPointOrder(
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
    """Validate the deterministic coordinate and seam revision of every address axis."""
    if coordinates is None and identifications is None:
        return None
    if coordinates is None or identifications is None:
        raise ValueError(
            f"morton.address_binding.coordinates must declare coordinates and identifications together; got {coordinates!r}"
        )
    axes: list[tuple[str, int | None]] = []
    for coordinate in coordinates:
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            raise ValueError(
                f"morton.address_binding.coordinates must be (label, component) pairs; got {type(coordinates).__name__}"
            )
        label, component = coordinate
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"morton.address_binding.coordinates must be non-empty strings; got {type(coordinates).__name__}"
            )
        if component is not None and (
            isinstance(component, bool) or not isinstance(component, int) or component < 0
        ):
            raise ValueError(
                f"morton.address_binding.coordinates must be nonnegative or None; got {type(coordinates).__name__}"
            )
        axes.append((label, component))
    revisions = tuple(identifications)
    if len(axes) != len(periodic) or len(revisions) != len(periodic):
        raise ValueError(
            f"morton.address_binding.coordinates must bind every Morton axis once; got {type(coordinates).__name__}"
        )
    if len(set(axes)) != len(axes):
        raise ValueError(
            f"morton.address_binding.coordinates must be distinct; got {type(coordinates).__name__}"
        )
    for revision, periodic_axis in zip(revisions, periodic, strict=True):
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise ValueError(
                f"morton.address_binding.coordinates must be non-empty strings; got {type(coordinates).__name__}"
            )
        if (revision is not None) != periodic_axis:
            raise ValueError(
                f"morton.address_binding.periodic must carry an identification revision on every periodic axis and no nonperiodic axis; got {type(periodic).__name__}"
            )
    return tuple(axes), revisions


def domain_point_coordinates(
    domain: Domain, coordinates: Sequence[tuple[str, int | None]] | None, /
) -> tuple[tuple[tuple[str, int | None], ...], tuple[float, ...], tuple[float, ...]]:
    """Bind point-array axes to Cartesian coordinates of ``domain`` and their bounds.

    ``None`` binds the components of a single-label domain in order. Geometry
    coordinates canonicalize a one-dimensional component to ``0`` as
    `PeriodicIdentification` does; factors without Cartesian faces are refused.
    """

    if coordinates is None:
        if len(domain.labels) != 1:
            raise ValueError(
                f"morton.domain_point_coordinates.coordinates must provide explicit point coordinates; got {type(coordinates).__name__}; context {domain.labels!r}"
            )
        (label,) = domain.labels
        factor = domain.factor(label)
        requested: tuple[tuple[str, int | None], ...] = (
            tuple((label, axis) for axis in range(factor.spatial_dim))
            if isinstance(factor, _domain_owner().AbstractGeometry)
            else ((label, None),)
        )
    else:
        requested = tuple(coordinates)
    axes: list[tuple[str, int | None]] = []
    lower: list[float] = []
    upper: list[float] = []
    for coordinate in requested:
        if not isinstance(coordinate, tuple) or len(coordinate) != 2:
            raise ValueError(
                f"morton.domain_point_coordinates.coordinates must be (label, component) pairs; got {type(coordinates).__name__}"
            )
        label, component = coordinate
        if not isinstance(label, str) or label not in domain.labels:
            raise ValueError(
                f"morton.domain_point_coordinates.domain must be a coordinate of {domain.labels!r}; got {type(domain).__name__}; context {label!r}"
            )
        factor = domain.factor(label)
        domain_owner = _domain_owner()
        match factor:
            case domain_owner.ScalarInterval():
                if component is not None:
                    raise ValueError(
                        f"morton.domain_point_coordinates.domain must use component None for scalar coordinates; got {type(domain).__name__}"
                    )
                bounds = (float(factor.fixed("start")), float(factor.fixed("end")))
            case domain_owner.AbstractGeometry():
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
                        f"morton.domain_point_coordinates.domain must provide a component in [0, {dimension}); got {component!r}; context {label!r}"
                    )
                face_lower, face_upper = factor.coordinate_face_bounds()
                bounds = (
                    float(np.asarray(face_lower)[component]),
                    float(np.asarray(face_upper)[component]),
                )
            case _:
                raise TypeError(
                    f"morton.domain_point_coordinates.domain must name a ScalarInterval or Cartesian geometry factor; got {type(domain).__name__}"
                )
        axes.append((label, component))
        lower.append(bounds[0])
        upper.append(bounds[1])
    if len(set(axes)) != len(axes):
        raise ValueError(
            f"morton.domain_point_coordinates.coordinates must be distinct; got {type(coordinates).__name__}"
        )
    return tuple(axes), tuple(lower), tuple(upper)


__all__ = [
    "MortonAddressPlan",
    "MortonCellGeometry",
    "MortonEncoding",
    "MortonHostEncoding",
    "canonical_morton_order",
    "hilbert_encode_integer",
    "morton_decode_integer",
    "morton_decode_integer_host",
    "morton_encode_integer",
    "morton_encode_integer_host",
]

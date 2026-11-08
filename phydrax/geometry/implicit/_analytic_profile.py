#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from fractions import Fraction
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...typing import checked
from .._certificate import SignReliability, ZeroSetAccuracy
from .._contracts import CompiledGeometry
from .._interval_enclosure import (
    interval_add,
    interval_center,
    interval_cos,
    interval_divide,
    interval_multiply,
    interval_sin,
    interval_sqrt,
    interval_square,
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import binary64_norm_upper
from .._source_boundary_records import (
    SourceBoundaryDistance,
    SourceBoundarySamples,
    SourceProjectionStatus,
)
from .._source_identity import source_entity_id
from ..analytic._extended import _TorusKernel
from ..analytic._primitives import _BallKernel


AnalyticImplicitFamily: TypeAlias = Literal["sphere", "ring-torus"]


def _lower_fraction(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        return rounded
    return float(np.nextafter(rounded, -np.inf)) if Fraction(rounded) > value else rounded


def _upper_fraction(value: Fraction) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        return rounded
    return float(np.nextafter(rounded, np.inf)) if Fraction(rounded) < value else rounded


_INTERVAL_BATCH_SIZE = 64


def _evaluate_interval_points(
    program: PreparedIntervalFunction, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for start in range(0, points.shape[0], _INTERVAL_BATCH_SIZE):
        stop = min(start + _INTERVAL_BATCH_SIZE, points.shape[0])
        lower, upper = program.evaluate(points[start:stop], points[start:stop])
        lower_parts.append(np.asarray(lower, dtype=np.float64))
        upper_parts.append(np.asarray(upper, dtype=np.float64))
    if not lower_parts:
        empty = np.empty((0,), dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(lower_parts, axis=0), np.concatenate(upper_parts, axis=0)


def _evaluate_interval_boxes(
    program: PreparedIntervalFunction,
    lower: np.ndarray,
    upper: np.ndarray,
    output_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for start in range(0, lower.shape[0], _INTERVAL_BATCH_SIZE):
        stop = min(start + _INTERVAL_BATCH_SIZE, lower.shape[0])
        bound_lower, bound_upper = program.evaluate(lower[start:stop], upper[start:stop])
        lower_parts.append(np.asarray(bound_lower, dtype=np.float64))
        upper_parts.append(np.asarray(bound_upper, dtype=np.float64))
    if not lower_parts:
        empty = np.empty((0, *output_shape), dtype=np.float64)
        return empty, empty.copy()
    return np.concatenate(lower_parts, axis=0), np.concatenate(upper_parts, axis=0)


def _decided_trigonometric(angles: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Cosine and sine of binary64 angles at their rigorous enclosure centers.

    The centers lie within their enclosures' few-ulp width of the exact values
    and are the same on every platform, unlike a platform libm evaluation.
    """
    values = np.asarray(angles, dtype=np.float64)
    return (
        interval_center(interval_cos((values, values))),
        interval_center(interval_sin((values, values))),
    )



def _sphere_cube_cover(
    center: np.ndarray,
    radius: float,
    subdivisions: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enclose the radial images of every cube-face parameter cell."""
    edges = np.linspace(-1.0, 1.0, subdivisions + 1, dtype=np.float64)
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("The requested cube-sphere cover reaches binary64 resolution.")
    first_lower, second_lower = np.meshgrid(edges[:-1], edges[:-1], indexing="ij")
    first_upper, second_upper = np.meshgrid(edges[1:], edges[1:], indexing="ij")
    first_lower = first_lower.reshape((-1,))
    first_upper = first_upper.reshape((-1,))
    second_lower = second_lower.reshape((-1,))
    second_upper = second_upper.reshape((-1,))
    first_interval = (first_lower, first_upper)
    second_interval = (second_lower, second_upper)
    one = (np.ones_like(first_lower), np.ones_like(first_upper))
    norm_squared = interval_add(
        one,
        interval_add(interval_square(first_interval), interval_square(second_interval)),
    )
    inverse_norm = interval_divide(one, interval_sqrt(norm_squared))
    first_direction = interval_multiply(first_interval, inverse_norm)
    second_direction = interval_multiply(second_interval, inverse_norm)
    first_center = first_lower + 0.5 * (first_upper - first_lower)
    second_center = second_lower + 0.5 * (second_upper - second_lower)
    # Correctly rounded IEEE operations only: the same sample on every platform.
    center_norm = np.sqrt(
        1.0 + first_center * first_center + second_center * second_center
    )
    normalized_first = first_center / center_norm
    normalized_second = second_center / center_norm
    center_ = np.asarray(center, dtype=np.float64)
    radius_value = np.asarray(radius, dtype=np.float64)
    point_parts: list[np.ndarray] = []
    lower_parts: list[np.ndarray] = []
    upper_parts: list[np.ndarray] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            fixed = interval_multiply(
                (np.full_like(first_lower, sign), np.full_like(first_upper, sign)),
                inverse_norm,
            )
            fixed_point = sign / center_norm
            if axis == 0:
                direction_interval = (
                    np.stack(
                        (fixed[0], first_direction[0], second_direction[0]),
                        axis=-1,
                    ),
                    np.stack(
                        (fixed[1], first_direction[1], second_direction[1]),
                        axis=-1,
                    ),
                )
                direction = np.stack(
                    (fixed_point, normalized_first, normalized_second), axis=-1
                )
            elif axis == 1:
                direction_interval = (
                    np.stack(
                        (first_direction[0], fixed[0], second_direction[0]),
                        axis=-1,
                    ),
                    np.stack(
                        (first_direction[1], fixed[1], second_direction[1]),
                        axis=-1,
                    ),
                )
                direction = np.stack(
                    (normalized_first, fixed_point, normalized_second), axis=-1
                )
            else:
                direction_interval = (
                    np.stack(
                        (first_direction[0], second_direction[0], fixed[0]),
                        axis=-1,
                    ),
                    np.stack(
                        (first_direction[1], second_direction[1], fixed[1]),
                        axis=-1,
                    ),
                )
                direction = np.stack(
                    (normalized_first, normalized_second, fixed_point), axis=-1
                )
            point_parts.append(center_[None, :] + radius * direction)
            relative = interval_multiply((radius_value, radius_value), direction_interval)
            image = interval_add((center_[None, :], center_[None, :]), relative)
            lower_parts.append(np.asarray(image[0], dtype=np.float64))
            upper_parts.append(np.asarray(image[1], dtype=np.float64))
    return (
        np.concatenate(point_parts, axis=0),
        np.concatenate(lower_parts, axis=0),
        np.concatenate(upper_parts, axis=0),
    )


def _sqrt_fraction_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Exact dyadic enclosure of a nonnegative rational square root."""
    if value <= 0:
        return Fraction(0), Fraction(0)
    scale = 1 << 128
    floor = math.isqrt((value.numerator << 256) // value.denominator)
    lower = Fraction(floor, scale)
    return lower, lower if lower * lower == value else Fraction(floor + 1, scale)


def _absolute_interval(lower: Fraction, upper: Fraction) -> tuple[Fraction, Fraction]:
    if lower <= 0 <= upper:
        return Fraction(0), max(-lower, upper)
    if upper < 0:
        return -upper, -lower
    return lower, upper


def _state_identity(geometry: CompiledGeometry) -> str:
    return canonical_fingerprint(
        {
            "kind": "analytic-implicit-state",
            "kernel": type(geometry.kernel).__module__
            + "."
            + type(geometry.kernel).__qualname__,
            "state": array_tree_fingerprint(geometry.state),
            "schema": geometry.schema.schema_id,
        }
    )


@dataclass(frozen=True, slots=True)
class _FieldBoxBounds:
    value_lower: np.ndarray
    value_upper: np.ndarray
    gradient_lower: np.ndarray
    gradient_upper: np.ndarray


@dataclass(frozen=True, slots=True)
class _AnalyticFieldBounds:
    """Outward interval field and gradient queries for a nominal analytic source."""

    geometry: CompiledGeometry
    enclosure: str = "interval"
    value_rigorous: bool = True
    gradient_rigorous: bool = dataclass_field(init=False, default=False)
    continuity_rigorous: bool = True
    _value_program: PreparedIntervalFunction = dataclass_field(
        init=False, repr=False, compare=False
    )
    _gradient_program: PreparedIntervalFunction | None = dataclass_field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        kernel, state = self.geometry.kernel, self.geometry.state

        def values(points: jax.Array) -> jax.Array:
            return kernel.boundary_field(state, points)

        def gradients(point: jax.Array) -> jax.Array:
            def scalar(coordinates: jax.Array) -> jax.Array:
                return kernel.boundary_field(state, coordinates[None, :])[0]

            return jax.grad(scalar)(point)

        object.__setattr__(self, "_value_program", prepare_interval_function(values, 3))
        gradient_program: PreparedIntervalFunction | None = None
        try:
            gradient_program = prepare_interval_function(gradients, 3)
        except IntervalExtensionError:
            gradient_program = None
        object.__setattr__(self, "_gradient_program", gradient_program)
        object.__setattr__(self, "gradient_rigorous", gradient_program is not None)

    def point_values(self, points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with shape (N, 3).")
        lower, upper = _evaluate_interval_points(self._value_program, values)
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def field_values(self, points: ArrayLike) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        result = self.geometry.kernel.boundary_field(
            self.geometry.state, jnp.asarray(values)
        )
        return np.asarray(result, dtype=np.float64)

    def boxes(self, lower: ArrayLike, upper: ArrayLike) -> _FieldBoxBounds:
        low = np.asarray(lower, dtype=np.float64)
        high = np.asarray(upper, dtype=np.float64)
        if (
            low.ndim != 2
            or high.shape != low.shape
            or low.shape[1] != 3
            or not np.all(np.isfinite(low))
            or not np.all(np.isfinite(high))
            or np.any(low > high)
        ):
            raise ValueError(
                "box bounds must be finite arrays with shape (N, 3) and lower <= upper."
            )
        if low.shape[0] == 0:
            empty = np.empty((0,), dtype=np.float64)
            empty_gradient = np.empty((0, 3), dtype=np.float64)
            return _FieldBoxBounds(
                empty, empty.copy(), empty_gradient, empty_gradient.copy()
            )
        value_lower, value_upper = _evaluate_interval_boxes(
            self._value_program, low, high, ()
        )
        if self._gradient_program is None:
            gradient_lower = np.full((low.shape[0], 3), -np.inf, dtype=np.float64)
            gradient_upper = np.full((low.shape[0], 3), np.inf, dtype=np.float64)
        else:
            try:
                gradient_lower, gradient_upper = _evaluate_interval_boxes(
                    self._gradient_program, low, high, (3,)
                )
            except IntervalExtensionError:
                gradient_lower = np.full((low.shape[0], 3), -np.inf, dtype=np.float64)
                gradient_upper = np.full((low.shape[0], 3), np.inf, dtype=np.float64)
        return _FieldBoxBounds(
            np.asarray(value_lower, dtype=np.float64).reshape((-1,)),
            np.asarray(value_upper, dtype=np.float64).reshape((-1,)),
            np.asarray(gradient_lower, dtype=np.float64).reshape((-1, 3)),
            np.asarray(gradient_upper, dtype=np.float64).reshape((-1, 3)),
        )

    def validate_source_integrity(self) -> None:
        if not bool(np.asarray(self.geometry.validity().accepted)):
            raise ValueError("The retained analytic field has invalid geometry state.")


def _implicit_scope_banks(
    source_id: str,
    source_revision: str,
    /,
) -> tuple[tuple[np.ndarray, tuple[tuple[str, tuple[str, ...]], ...], str], ...]:
    """Reuse the native implicit source's canonical strata and numeric IDs."""
    from ...meshing._plc_source import _numeric_entity_ids
    from ...meshing.providers._native_sources import _prepare_source_scope_bank

    entities = (
        (),
        (),
        (("implicit-surface", "zero-set", "native-implicit-surface"),),
        (("implicit-region", "interior", "native-implicit-region"),),
    )
    banks = []
    for dimension, rows in enumerate(entities):
        records = []
        for kind, entity_id, numeric_kind in rows:
            key = source_entity_id(source_id, source_revision, kind, entity_id)
            identifier = int(_numeric_entity_ids(numeric_kind, (key,), ((),))[0])
            records.append((kind, entity_id, (), identifier))
        banks.append(
            _prepare_source_scope_bank(
                source_id, source_revision, dimension, tuple(records)
            )
        )
    return tuple(banks)


@final
class AnalyticImplicitProfile(StrictModule, NonTrainableState):
    """Established topology, reach and certified distance queries for spheres and ring tori.

    Only the nominal three-dimensional ball kernel and a full nonsingular ring
    torus are admitted. Boolean expressions, derived kernels, sectors and
    horn/spindle tori do not inherit these global geometric premises.
    """

    __strict_contract__ = True

    geometry: CompiledGeometry
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)
    tube_radius: float = eqx.field(static=True)
    cover_radius: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    family: AnalyticImplicitFamily = eqx.field(static=True)
    center: np.ndarray
    boundary_bounds: np.ndarray
    interior_witness: np.ndarray
    radius: float = eqx.field(static=True)
    major_radius: float = eqx.field(static=True)
    reach_lower: float = eqx.field(static=True)
    hessian_upper: float = eqx.field(static=True)
    normal_lipschitz_upper: float = eqx.field(static=True)
    gradient_norm_bounds: tuple[float, float] = eqx.field(static=True)
    interior_margin_lower: float = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    genus: int = eqx.field(static=True)
    boundary_euler_characteristic: int = eqx.field(static=True)
    represented_geometry_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    field_bounds: _AnalyticFieldBounds
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...] = (
        eqx.field(static=True)
    )
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)
    _map_program: PreparedIntervalFunction

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        coordinate_contract: SpatialCoordinateContract,
        tube_radius: float | None = None,
        cover_radius: float | None = None,
        source_id: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        if not bool(np.asarray(geometry.validity().accepted)):
            raise ValueError("geometry must have accepted validity evidence.")
        kernel = geometry.kernel
        if type(kernel) is _BallKernel and kernel.dimension == 3:
            family: AnalyticImplicitFamily = "sphere"
            center = np.asarray(kernel.center.read(geometry.state), dtype=np.float64)
            radius = float(np.asarray(kernel.radius.read(geometry.state)))
            major = 0.0
            reach = Fraction(radius)
            outer = reach
            genus = 0
            witness = center.copy()
            represented = kernel.source_id
            binding_ids = (kernel.center, kernel.radius)
        elif type(kernel) is _TorusKernel and kernel.full:
            family = "ring-torus"
            center = np.asarray(kernel.center.read(geometry.state), dtype=np.float64)
            radius = float(np.asarray(kernel.minor.read(geometry.state)))
            major = float(np.asarray(kernel.major.read(geometry.state)))
            angle = float(np.asarray(kernel.angle.read(geometry.state)))
            if angle != 2.0 * math.pi or not major > radius > 0.0:
                raise ValueError(
                    "The torus must be full and satisfy major_radius > minor_radius > 0."
                )
            reach = min(Fraction(radius), Fraction(major) - Fraction(radius))
            outer = Fraction(major) + Fraction(radius)
            genus = 1
            witness = center.copy()
            witness[0] += major
            represented = kernel.source_id
            binding_ids = (kernel.center, kernel.major, kernel.minor, kernel.angle)
        else:
            raise ValueError(
                "Only nominal three-dimensional spheres and full ring tori are supported."
            )
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise ValueError(
                "The analytic source must have a finite center and positive finite radius."
            )
        if geometry.field_certificate.zero_set_accuracy is not ZeroSetAccuracy.EXACT:
            raise ValueError("The analytic profile requires an exact zero set.")
        if geometry.field_certificate.sign_reliability is not SignReliability.RELIABLE:
            raise ValueError(
                "The analytic profile requires reliable inside/outside semantics."
            )
        reach_lower = _lower_fraction(reach)
        if reach_lower <= 0.0 or not math.isfinite(reach_lower):
            raise ValueError(
                "The source reach must be positive and representable in binary64."
            )
        tube = (
            _lower_fraction(Fraction(reach_lower) / 4)
            if tube_radius is None
            else float(tube_radius)
        )
        if not math.isfinite(tube) or not 0.0 < tube < reach_lower:
            raise ValueError(
                "tube_radius must be finite, positive, and strictly below the established reach."
            )
        cover = tube / 4.0 if cover_radius is None else float(cover_radius)
        if not math.isfinite(cover) or cover <= 0.0:
            raise ValueError("cover_radius must be finite and positive.")
        hessian = _upper_fraction(1 / (reach - Fraction(tube)))
        if not math.isfinite(hessian):
            raise ValueError(
                "The source Hessian bound must be representable in binary64."
            )
        extents = (
            (outer, outer, Fraction(radius)) if family == "ring-torus" else (outer,) * 3
        )
        bounds = np.asarray(
            [
                [
                    _lower_fraction(Fraction(c) - extent)
                    for c, extent in zip(center.tolist(), extents, strict=True)
                ],
                [
                    _upper_fraction(Fraction(c) + extent)
                    for c, extent in zip(center.tolist(), extents, strict=True)
                ],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(bounds)):
            raise ValueError(
                "The analytic boundary bounds must be finite and representable in binary64."
            )
        source = (
            represented
            if source_id is None
            else canonical_identifier(source_id, "source_id")
        )
        state_id = _state_identity(geometry)
        schema_id = geometry.schema.schema_id
        kernel_id = canonical_fingerprint(
            {
                "kind": "analytic-implicit-kernel",
                "kernel": type(kernel).__module__ + "." + type(kernel).__qualname__,
                "represented_geometry": represented,
                "bindings": tuple(
                    (
                        binding.parameter_id.feature_id,
                        binding.parameter_id.name,
                        binding.index,
                    )
                    for binding in binding_ids
                ),
            }
        )
        revision = canonical_fingerprint(
            {
                "kind": "analytic-implicit-revision",
                "source": source,
                "state": state_id,
                "schema": schema_id,
                "kernel": kernel_id,
                "coordinate_contract": coordinate_contract.spatial_id,
            }
        )
        if source_revision is not None:
            revision = canonical_identifier(source_revision, "source_revision")
        margin = radius
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                "The nominal interior witness must have positive finite separation from the boundary."
            )
        center = np.array(center, dtype=np.float64, copy=True)
        witness = np.array(witness, dtype=np.float64, copy=True)
        for value in (center, witness, bounds):
            value.setflags(write=False)
        profile_id = canonical_fingerprint(
            {
                "kind": "analytic-implicit-profile",
                "source": source,
                "revision": revision,
                "state": state_id,
                "schema": schema_id,
                "kernel": kernel_id,
                "coordinates": coordinate_contract.spatial_id,
                "family": family,
                "center": array_tree_fingerprint(center),
                "bounds": array_tree_fingerprint(bounds),
                "radius": radius,
                "major_radius": major,
                "reach": reach_lower,
                "tube": tube,
                "cover_radius": cover,
                "hessian": hessian,
            }
        )
        self.geometry = geometry
        self.coordinate_contract = coordinate_contract
        self.source_id = source
        self.source_revision = revision
        self.family = family
        self.center = center
        self.boundary_bounds = bounds
        self.interior_witness = witness
        self.radius = radius
        self.major_radius = major
        self.reach_lower = reach_lower
        self.tube_radius = tube
        self.cover_radius = cover
        self.hessian_upper = hessian
        self.normal_lipschitz_upper = hessian
        self.gradient_norm_bounds = (1.0, 1.0)
        self.interior_margin_lower = margin
        self.component_count = 1
        self.genus = genus
        self.boundary_euler_characteristic = 2 - 2 * genus
        self.represented_geometry_id = represented
        self.state_id = state_id
        self.schema_id = schema_id
        self.kernel_id = kernel_id
        self.profile_id = profile_id
        self.field_bounds = _AnalyticFieldBounds(geometry)
        scope_banks = _implicit_scope_banks(source, revision)
        self.scope_entity_id_banks = tuple(bank[0] for bank in scope_banks)
        self.scope_entity_key_banks = tuple(bank[1] for bank in scope_banks)
        self.scope_entity_set_ids = tuple(bank[2] for bank in scope_banks)

        center_array = jnp.asarray(center, dtype=jnp.float64)

        def parameter_map(parameters: jax.Array) -> jax.Array:
            first, second = parameters[0], parameters[1]
            if family == "sphere":
                radial = radius * jnp.sin(first)
                return jnp.stack(
                    (
                        center_array[0] + radial * jnp.cos(second),
                        center_array[1] + radial * jnp.sin(second),
                        center_array[2] + radius * jnp.cos(first),
                    ),
                    axis=-1,
                )
            radial = major + radius * jnp.cos(second)
            return jnp.stack(
                (
                    center_array[0] + radial * jnp.cos(first),
                    center_array[1] + radial * jnp.sin(first),
                    center_array[2] + radius * jnp.sin(second),
                ),
                axis=-1,
            )

        self._map_program = prepare_interval_function(parameter_map, 2)

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def region_ids(self) -> tuple[str, ...]:
        return ("interior",)

    @property
    def region_occurrence_paths(self) -> tuple[tuple[str, ...], ...]:
        return ((),)

    @property
    def frame_id(self) -> str:
        return self.coordinate_contract.spatial_id

    @property
    def domain_id(self) -> str:
        self.validate_source_integrity()
        return canonical_fingerprint(
            {
                "kind": "analytic-implicit-region-domain",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "represented_geometry_id": self.represented_geometry_id,
                "state_id": self.state_id,
                "schema_id": self.schema_id,
                "kernel_id": self.kernel_id,
                "coordinate_contract": self.coordinate_contract.spatial_id,
                "region_ids": self.region_ids,
                "region_occurrence_paths": self.region_occurrence_paths,
            }
        )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Implicit profile dimension must be in 0..3.")
        return self.scope_entity_id_banks[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Implicit profile dimension must be in 0..3.")
        return self.scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if isinstance(dimension, bool) or dimension not in (0, 1, 2, 3):
            raise ValueError("Implicit profile dimension must be in 0..3.")
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            raise TypeError("entity_id must be one stable integer scope ID.")
        identifiers = self.scope_entity_id_banks[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("entity_id is absent from this implicit profile dimension.")
        return self.scope_entity_key_banks[dimension][position]

    def validate_source_integrity(self) -> None:
        if (
            _state_identity(self.geometry) != self.state_id
            or self.geometry.schema.schema_id != self.schema_id
        ):
            raise ValueError(
                "The retained analytic source state no longer matches the established profile."
            )
        self.field_bounds.validate_source_integrity()

    def require_bound(
        self,
        geometry: CompiledGeometry,
        coordinate_contract: SpatialCoordinateContract,
        /,
    ) -> None:
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if (
            not isinstance(geometry, CompiledGeometry)
            or _state_identity(geometry) != self.state_id
            or geometry.schema.schema_id != self.schema_id
            or coordinate_contract.spatial_id != self.coordinate_contract.spatial_id
            or type(geometry.kernel) is not type(self.geometry.kernel)
        ):
            raise ValueError(
                "The profile must remain bound to the same source state and coordinate contract."
            )
        self.validate_source_integrity()

    def _field_singular_set_intersects_box(self, box: np.ndarray, /) -> bool:
        """Test exact contact with the profile's signed-distance singular set."""
        bounds = np.asarray(box, dtype=np.float64)
        if (
            bounds.shape != (2, 3)
            or np.any(~np.isfinite(bounds))
            or np.any(bounds[0] > bounds[1])
        ):
            raise ValueError("A singularity query requires one finite ordered 3D box.")
        lower = tuple(Fraction.from_float(float(value)) for value in bounds[0])
        upper = tuple(Fraction.from_float(float(value)) for value in bounds[1])
        center = tuple(Fraction.from_float(float(value)) for value in self.center)
        match self.family:
            case "sphere":
                return all(
                    lower[axis] <= center[axis] <= upper[axis] for axis in range(3)
                )
            case "ring-torus":
                if (
                    lower[0] <= center[0] <= upper[0]
                    and lower[1] <= center[1] <= upper[1]
                ):
                    return True
                if not lower[2] <= center[2] <= upper[2]:
                    return False
                minimum_distance = tuple(
                    Fraction(0)
                    if lower[axis] <= center[axis] <= upper[axis]
                    else min(
                        abs(lower[axis] - center[axis]),
                        abs(upper[axis] - center[axis]),
                    )
                    for axis in range(2)
                )
                maximum_distance = tuple(
                    max(
                        abs(lower[axis] - center[axis]),
                        abs(upper[axis] - center[axis]),
                    )
                    for axis in range(2)
                )
                minimum_radius_squared = sum(
                    (value * value for value in minimum_distance), Fraction(0)
                )
                maximum_radius_squared = sum(
                    (value * value for value in maximum_distance), Fraction(0)
                )
                major_radius_squared = Fraction.from_float(self.major_radius) ** 2
                return (
                    minimum_radius_squared
                    <= major_radius_squared
                    <= maximum_radius_squared
                )
            case _ as unreachable:
                assert_never(unreachable)

    def boundary_distance(self, points: ArrayLike, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with shape (N, 3).")
        self.validate_source_integrity()
        lower, upper = _exact_boundary_distance(
            values, self.family, self.center, self.radius, self.major_radius
        )
        return SourceBoundaryDistance(lower, upper, "certified")

    def project_boundary(
        self,
        points: ArrayLike,
        /,
        *,
        entity_ids: tuple[str, ...] | None = None,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> _ImplicitSourceProjectionResult:
        """Project to the exact analytic zero-set with singular loci retained."""
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with shape (N, 3).")
        self.validate_source_integrity()
        entity_id = source_entity_id(
            self.source_id,
            self.source_revision,
            "implicit-surface",
            "zero-set",
        )
        if entity_ids is not None and (
            not isinstance(entity_ids, tuple)
            or any(not isinstance(value, str) or not value for value in entity_ids)
        ):
            raise TypeError("entity_ids must be a tuple of canonical source entity IDs.")
        if entity_ids is not None and any(value != entity_id for value in entity_ids):
            raise ValueError("entity_ids must belong to this implicit source revision.")
        if occurrence_paths is not None and (
            not isinstance(occurrence_paths, tuple)
            or len(occurrence_paths) != values.shape[0]
            or any(path != () for path in occurrence_paths)
        ):
            raise ValueError(
                "Implicit source projections use the definition-only empty occurrence path."
            )
        selected = entity_ids is None or entity_id in entity_ids
        if not selected:
            count = values.shape[0]
            return _ImplicitSourceProjectionResult(
                self.source_id,
                self.source_revision,
                self.frame_id,
                values,
                np.full((count, 3), np.nan),
                np.full((count, 2), np.nan),
                np.full((count,), np.inf),
                np.zeros((count,)),
                np.full((count,), SourceProjectionStatus.FAILED, dtype=np.int8),
                np.full((count, 3), np.nan),
                np.full((count, 2, 3), np.nan),
                tuple(() for _ in range(count)),
                tuple("" for _ in range(count)),
            )
        lower, upper = _exact_boundary_distance(
            values, self.family, self.center, self.radius, self.major_radius
        )
        projected = np.full((values.shape[0], 3), np.nan, dtype=np.float64)
        parameters = np.full((values.shape[0], 2), np.nan, dtype=np.float64)
        normals = np.full((values.shape[0], 3), np.nan, dtype=np.float64)
        tangents = np.full((values.shape[0], 2, 3), np.nan, dtype=np.float64)
        status = np.full(
            (values.shape[0],), SourceProjectionStatus.UNRESOLVED, dtype=np.int8
        )
        center = np.asarray(self.center, dtype=np.float64)
        for row, query in enumerate(values):
            relative = query - center
            if self.family == "sphere":
                length = math.hypot(*(float(value) for value in relative))
                if length == 0.0:
                    status[row] = SourceProjectionStatus.NON_ISOLATED
                    continue
                normal = relative / length
                longitude = math.atan2(float(relative[1]), float(relative[0]))
                if longitude < 0.0:
                    longitude += math.tau
                latitude = math.atan2(
                    float(relative[2]), math.hypot(float(relative[0]), float(relative[1]))
                )
                polar = math.pi / 2.0 - latitude
                point = center + self.radius * normal
                longitude_tangent = np.asarray(
                    (-math.sin(longitude), math.cos(longitude), 0.0),
                    dtype=np.float64,
                )
                latitude_tangent = np.asarray(
                    (
                        -math.sin(latitude) * math.cos(longitude),
                        -math.sin(latitude) * math.sin(longitude),
                        math.cos(latitude),
                    ),
                    dtype=np.float64,
                )
                first_tangent, second_tangent = -latitude_tangent, longitude_tangent
            else:
                x, y, z = (float(value) for value in relative)
                radial = math.hypot(x, y)
                tube_x, tube_y = radial - self.major_radius, z
                tube_length = math.hypot(tube_x, tube_y)
                if radial == 0.0 or tube_length == 0.0:
                    status[row] = SourceProjectionStatus.NON_ISOLATED
                    continue
                longitude = math.atan2(y, x)
                latitude = math.atan2(tube_y, tube_x)
                if longitude < 0.0:
                    longitude += math.tau
                if latitude < 0.0:
                    latitude += math.tau
                ring_radius = self.major_radius + self.radius * math.cos(latitude)
                height = self.radius * math.sin(latitude)
                radial_direction = np.asarray(
                    (math.cos(longitude), math.sin(longitude), 0.0),
                    dtype=np.float64,
                )
                normal = math.cos(latitude) * radial_direction + math.sin(
                    latitude
                ) * np.asarray((0.0, 0.0, 1.0))
                point = (
                    center
                    + ring_radius * radial_direction
                    + np.asarray((0.0, 0.0, height))
                )
                first_tangent = np.asarray(
                    (-math.sin(longitude), math.cos(longitude), 0.0),
                    dtype=np.float64,
                )
                second_tangent = np.asarray(
                    (
                        -math.sin(latitude) * math.cos(longitude),
                        -math.sin(latitude) * math.sin(longitude),
                        math.cos(latitude),
                    ),
                    dtype=np.float64,
                )
            projected[row] = point
            parameters[row] = (
                (polar, longitude) if self.family == "sphere" else (longitude, latitude)
            )
            normals[row] = normal
            tangents[row] = (first_tangent, second_tangent)
            status[row] = SourceProjectionStatus.UNIQUE
        return _ImplicitSourceProjectionResult(
            self.source_id,
            self.source_revision,
            self.frame_id,
            values,
            projected,
            parameters,
            upper,
            lower,
            status,
            normals,
            tangents,
            tuple(() for _ in range(values.shape[0])),
            tuple(entity_id for _ in range(values.shape[0])),
        )

    def normal_variation_upper(self, distance: float, /) -> float:
        value = float(distance)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("distance must be finite and nonnegative.")
        return _upper_fraction(
            min(Fraction(2), Fraction(self.normal_lipschitz_upper) * Fraction(value))
        )

    def _cover_counts(self, maximum_radius: float, /) -> tuple[int, int]:
        precision = Fraction(maximum_radius)

        def ceiling(value: Fraction) -> int:
            return max(1, (value.numerator + value.denominator - 1) // value.denominator)

        index_maximum = int(np.iinfo(np.intp).max)
        if self.family == "sphere":
            count = ceiling(Fraction(3, 2) * Fraction(self.radius) / precision)
            if count >= index_maximum or count * count > index_maximum // 18:
                raise OverflowError(
                    "The requested analytic boundary cover exceeds array index capacity."
                )
            return count, count

        pi_upper = Fraction(float(np.nextafter(math.pi, np.inf)))
        first_length = (
            2 * pi_upper * (Fraction(self.major_radius) + Fraction(self.radius))
        )
        second_length = 2 * pi_upper * Fraction(self.radius)
        first = ceiling(first_length / precision)
        second = ceiling(second_length / precision)
        if (
            first >= index_maximum
            or second >= index_maximum
            or first * second > index_maximum // 3
        ):
            raise OverflowError(
                "The requested analytic boundary cover exceeds array index capacity."
            )
        return first, second

    def _sample_grid(self, first_count: int, second_count: int) -> SourceBoundarySamples:
        if self.family == "sphere":
            points, image_lower, image_upper = _sphere_cube_cover(
                self.center, self.radius, first_count
            )
        else:
            first_edges = np.linspace(
                0.0, 2.0 * math.pi, first_count + 1, dtype=np.float64
            )
            second_edges = np.linspace(
                0.0, 2.0 * math.pi, second_count + 1, dtype=np.float64
            )
            if np.any(np.diff(first_edges) <= 0.0) or np.any(
                np.diff(second_edges) <= 0.0
            ):
                raise ValueError(
                    "The requested analytic parameter cover reaches binary64 resolution."
                )
            first = first_edges[:-1] + 0.5 * np.diff(first_edges)
            second = second_edges[:-1] + 0.5 * np.diff(second_edges)
            first_grid, second_grid = np.meshgrid(first, second, indexing="ij")
            first_cosine, first_sine = _decided_trigonometric(first_grid)
            second_cosine, second_sine = _decided_trigonometric(second_grid)
            radial = self.major_radius + self.radius * second_cosine
            points = np.stack(
                (
                    self.center[0] + radial * first_cosine,
                    self.center[1] + radial * first_sine,
                    self.center[2] + self.radius * second_sine,
                ),
                axis=-1,
            ).reshape((-1, 3))
            first_lower, second_lower = np.meshgrid(
                first_edges[:-1], second_edges[:-1], indexing="ij"
            )
            first_upper, second_upper = np.meshgrid(
                first_edges[1:], second_edges[1:], indexing="ij"
            )
            parameter_lower = np.stack((first_lower, second_lower), axis=-1).reshape(
                (-1, 2)
            )
            parameter_upper = np.stack((first_upper, second_upper), axis=-1).reshape(
                (-1, 2)
            )
        if self.family == "ring-torus":
            image_lower, image_upper = _evaluate_interval_boxes(
                self._map_program, parameter_lower, parameter_upper, (3,)
            )
        image_lower = np.asarray(image_lower, dtype=np.float64)
        image_upper = np.asarray(image_upper, dtype=np.float64)
        displacement = np.maximum(
            np.nextafter(np.abs(points - image_lower), np.inf),
            np.nextafter(np.abs(image_upper - points), np.inf),
        )
        if self.family == "sphere":
            period_slack = 0.0
        else:
            pi_lower = Fraction(float(np.nextafter(math.pi, -np.inf)))
            pi_upper = Fraction(float(np.nextafter(math.pi, np.inf)))
            scale = Fraction(self.major_radius) + 2 * Fraction(self.radius)
            period_slack = _upper_fraction(4 * (pi_upper - pi_lower) * scale)
        radii = binary64_norm_upper(displacement)
        radii = np.nextafter(radii + period_slack, np.inf)
        errors = self.boundary_distance(points).upper
        return SourceBoundarySamples(
            points,
            radii,
            errors,
            "certified",
            complete=True,
        )

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        if isinstance(maximum_radius, (bool, np.bool_)):
            raise TypeError(
                "maximum_radius must be a physical real distance, not a boolean."
            )
        requested = float(maximum_radius)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("maximum_radius must be finite and positive.")
        first, second = self._cover_counts(requested)
        if self.family == "ring-torus":
            pi_upper = Fraction(float(np.nextafter(math.pi, np.inf)))
            first_length = (
                2 * pi_upper * (Fraction(self.major_radius) + Fraction(self.radius))
            )
            second_length = 2 * pi_upper * Fraction(self.radius)
        index_maximum = int(np.iinfo(np.intp).max)
        while True:
            samples = self._sample_grid(first, second)
            if (
                not samples.covering_radius.size
                or np.max(samples.covering_radius) <= requested
            ):
                return samples
            if self.family == "sphere":
                first += 1
                second = first
            elif first_length / first >= second_length / second:
                first += 1
            else:
                second += 1
            sample_capacity = 18 if self.family == "sphere" else 3
            if (
                first >= index_maximum
                or second >= index_maximum
                or first * second > index_maximum // sample_capacity
            ):
                raise OverflowError(
                    "The requested analytic boundary cover exceeds array index capacity."
                )


def _exact_boundary_distance(
    points: np.ndarray,
    family: AnalyticImplicitFamily,
    center: np.ndarray,
    radius: float,
    major_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty((points.shape[0],), dtype=np.float64)
    upper = np.empty_like(lower)
    center_q = tuple(Fraction(float(value)) for value in center)
    minor_q = Fraction(float(radius))
    major_q = Fraction(float(major_radius))
    for index, point in enumerate(points):
        delta = tuple(
            Fraction(float(value)) - origin
            for value, origin in zip(point, center_q, strict=True)
        )
        if family == "sphere":
            square = sum((component * component for component in delta), Fraction(0))
            radial_lower, radial_upper = _sqrt_fraction_bounds(square)
        else:
            rho_lower, rho_upper = _sqrt_fraction_bounds(
                delta[0] * delta[0] + delta[1] * delta[1]
            )
            radial_lower, radial_upper = rho_lower - major_q, rho_upper - major_q
            if radial_lower <= 0 <= radial_upper:
                radial_square_lower = Fraction(0)
            else:
                radial_square_lower = min(
                    radial_lower * radial_lower, radial_upper * radial_upper
                )
            radial_square_upper = max(
                radial_lower * radial_lower, radial_upper * radial_upper
            )
            square_lower = radial_square_lower + delta[2] * delta[2]
            square_upper = radial_square_upper + delta[2] * delta[2]
            radial_lower, _ = _sqrt_fraction_bounds(square_lower)
            _, radial_upper = _sqrt_fraction_bounds(square_upper)
        signed_lower, signed_upper = radial_lower - minor_q, radial_upper - minor_q
        absolute_lower, absolute_upper = _absolute_interval(signed_lower, signed_upper)
        lower[index] = max(0.0, _lower_fraction(absolute_lower))
        upper[index] = max(lower[index], _upper_fraction(absolute_upper))
    return lower, upper


@final
class _ImplicitSourceProjectionResult(StrictModule, NonTrainableState):
    """Certified row-aligned projection evidence for one analytic source revision."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    queries: jax.Array
    points: jax.Array
    parameters: jax.Array
    residuals: jax.Array
    distance_lower_bounds: jax.Array
    status: jax.Array
    normals: jax.Array
    tangents: jax.Array
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    _entity_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_id: str,
        source_revision: str,
        frame_id: str,
        queries: ArrayLike,
        points: ArrayLike,
        parameters: ArrayLike,
        residuals: ArrayLike,
        distance_lower_bounds: ArrayLike,
        status: ArrayLike,
        normals: ArrayLike,
        tangents: ArrayLike,
        occurrence_paths: tuple[tuple[str, ...], ...],
        entity_ids: tuple[str, ...],
        /,
    ) -> None:
        identities = (source_id, source_revision, frame_id)
        if any(
            not isinstance(value, str) or not value or value.strip() != value
            for value in identities
        ):
            raise ValueError(
                "Projection source, revision, and frame IDs must be nonempty."
            )
        query = np.asarray(queries, dtype=np.float64)
        point = np.asarray(points, dtype=np.float64)
        parameter = np.asarray(parameters, dtype=np.float64)
        upper = np.asarray(residuals, dtype=np.float64)
        lower = np.asarray(distance_lower_bounds, dtype=np.float64)
        raw_status = np.asarray(status)
        normal = np.asarray(normals, dtype=np.float64)
        tangent = np.asarray(tangents, dtype=np.float64)
        count = query.shape[0] if query.ndim == 2 else -1
        if not np.issubdtype(raw_status.dtype, np.integer):
            raise TypeError("Projection status must be an integer array.")
        allowed = np.asarray([int(value) for value in SourceProjectionStatus])
        if np.any(~np.isin(raw_status, allowed)):
            raise ValueError("Projection status values are invalid.")
        states = raw_status.astype(np.int8)
        if (
            query.shape != (count, 3)
            or point.shape != (count, 3)
            or parameter.shape != (count, 2)
            or upper.shape != (count,)
            or lower.shape != (count,)
            or states.shape != (count,)
            or normal.shape != (count, 3)
            or tangent.shape != (count, 2, 3)
            or len(occurrence_paths) != count
            or len(entity_ids) != count
        ):
            raise ValueError(
                "Projection evidence arrays and identities must be row-aligned."
            )
        if not np.all(np.isfinite(query)):
            raise ValueError("Projection queries must be finite.")
        paths = occurrence_paths
        if not isinstance(paths, tuple) or any(path != () for path in paths):
            raise ValueError(
                "Analytic implicit projections require definition-only occurrence paths."
            )
        identifiers = entity_ids
        if not isinstance(identifiers, tuple) or any(
            not isinstance(value, str) for value in identifiers
        ):
            raise TypeError(
                "entity_ids must contain one source identity per projection row."
            )
        failed = states == SourceProjectionStatus.FAILED
        if any(
            bool(failed[row]) != (identifier == "")
            for row, identifier in enumerate(identifiers)
        ):
            raise ValueError(
                "Only failed projection rows may omit their source entity ID."
            )
        missing_entity = np.asarray(
            [identifier == "" for identifier in identifiers], dtype=np.bool_
        )
        if (
            np.any(np.isnan(upper) | (upper < 0.0))
            or np.any(~np.isfinite(lower) | (lower < 0.0) | (lower > upper))
            or np.any(np.isinf(point))
            or np.any(np.isinf(parameter))
            or np.any(np.isinf(normal))
            or np.any(np.isinf(tangent))
        ):
            raise ValueError(
                "Projection data must contain ordered bounds and no infinities."
            )
        resolved = np.isin(
            states,
            (SourceProjectionStatus.UNIQUE, SourceProjectionStatus.SEAM),
        )
        if np.any(
            resolved
            & (
                ~np.all(np.isfinite(point), axis=1)
                | ~np.all(np.isfinite(parameter), axis=1)
                | ~np.all(np.isfinite(normal), axis=1)
                | ~np.all(np.isfinite(tangent), axis=(1, 2))
                | missing_entity
            )
        ):
            raise ValueError(
                "Resolved projection rows require finite source geometry evidence."
            )
        unselected = np.isin(
            states,
            (
                SourceProjectionStatus.UNRESOLVED,
                SourceProjectionStatus.NON_ISOLATED,
                SourceProjectionStatus.FAILED,
            ),
        )
        if np.any(
            unselected
            & (~np.all(np.isnan(point), axis=1) | ~np.all(np.isnan(parameter), axis=1))
        ):
            raise ValueError(
                "Unresolved and non-isolated projections cannot select a point."
            )
        self.source_id = source_id
        self.source_revision = source_revision
        self.frame_id = frame_id
        self.queries = jnp.asarray(query)
        self.points = jnp.asarray(point)
        self.parameters = jnp.asarray(parameter)
        self.residuals = jnp.asarray(upper)
        self.distance_lower_bounds = jnp.asarray(lower)
        self.status = jnp.asarray(states)
        self.normals = jnp.asarray(normal)
        self.tangents = jnp.asarray(tangent)
        self.occurrence_paths = paths
        self._entity_ids = identifiers
        self.result_id = canonical_fingerprint(
            {
                "kind": "analytic-implicit-source-projection",
                "source": (source_id, source_revision),
                "frame_id": frame_id,
                "queries": array_tree_fingerprint(query),
                "points": array_tree_fingerprint(point),
                "parameters": array_tree_fingerprint(parameter),
                "distance_lower_bounds": array_tree_fingerprint(lower),
                "residuals": array_tree_fingerprint(upper),
                "status": array_tree_fingerprint(states),
                "normals": array_tree_fingerprint(normal),
                "tangents": array_tree_fingerprint(tangent),
                "occurrence_paths": paths,
                "entity_ids": identifiers,
            }
        )

    @property
    def resolved(self) -> jax.Array:
        return (self.status == SourceProjectionStatus.UNIQUE) | (
            self.status == SourceProjectionStatus.SEAM
        )

    def entity_ids(self, /) -> tuple[str, ...]:
        return self._entity_ids


__all__ = [
    "AnalyticImplicitFamily",
    "AnalyticImplicitProfile",
]

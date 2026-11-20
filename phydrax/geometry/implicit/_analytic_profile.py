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

from ..._exact_kernels import float64_interval_elementary
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
    interval_multiply,
    interval_sin,
    interval_sqrt,
    IntervalExtensionError,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._rational import binary64_difference_magnitude, binary64_norm_upper
from .._source_boundary_records import (
    SourceBoundaryDistance,
    SourceProjectionStatus,
)
from .._source_identity import source_entity_id
from ..analytic._extended import _TorusKernel
from ..analytic._primitives import _BallKernel


AnalyticImplicitFamily: TypeAlias = Literal["sphere", "ring-torus"]
ZeroSetReverseProof: TypeAlias = Literal["tube", "cover"]


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


def _evaluate_interval_points(
    program: PreparedIntervalFunction, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _evaluate_interval_boxes(program, points, points, ())


def _evaluate_interval_boxes(
    program: PreparedIntervalFunction,
    lower: np.ndarray,
    upper: np.ndarray,
    output_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    # One call: the host executor blocks the boxes itself, so an outer
    # partition would only add dispatches.
    if lower.shape[0] == 0:
        empty = np.empty((0, *output_shape), dtype=np.float64)
        return empty, empty.copy()
    bound_lower, bound_upper = program.evaluate(lower, upper)
    return (
        np.asarray(bound_lower, dtype=np.float64),
        np.asarray(bound_upper, dtype=np.float64),
    )


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


def _decided_norm(values: tuple[float, ...], /) -> float:
    """Choose a reproducible norm without overflow in the sum of squares."""
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    scaled = tuple(value / scale for value in values)
    squared = np.float64(sum(value * value for value in scaled))
    return scale * float(interval_center(interval_sqrt((squared, squared))))


def _decided_atan2(y: float, x: float, /) -> float:
    """Choose a reproducible angle while keeping the atan argument bounded."""
    if x == 0.0:
        if y == 0.0:
            return math.copysign(math.pi if math.copysign(1.0, x) < 0.0 else 0.0, y)
        return math.copysign(math.pi / 2.0, y)
    ratio = np.float64(min(abs(y), abs(x)) / max(abs(y), abs(x)))
    angle = float(interval_center(float64_interval_elementary("atan", ratio, ratio)))
    if abs(y) > abs(x):
        angle = math.pi / 2.0 - angle
    if x < 0.0:
        angle = math.pi - angle
    return math.copysign(angle, y)


# Repairs refuse the certificate, so the split angle of a repair never shapes
# an accepted surface; it only has to be admissible.
_ZERO_SET_REPAIR_ANGLE_DEGREES = 30.0


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


def _triangle_diameter_upper(triangles: np.ndarray, /) -> np.ndarray:
    """Directed upper bound of the longest edge of each ``(3, 3)`` triangle."""
    edges = binary64_difference_magnitude(triangles, np.roll(triangles, -1, axis=1))[1]
    return np.max(binary64_norm_upper(edges), axis=1)


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

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        coordinate_contract: SpatialCoordinateContract,
        tube_radius: float | None = None,
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

    def _field_singular_set_intersects_boxes(self, boxes: np.ndarray, /) -> bool:
        """Decide whether any box meets the exact signed-distance singular set."""
        bounds = np.asarray(boxes, dtype=np.float64)
        if (
            bounds.ndim != 3
            or bounds.shape[1:] != (2, 3)
            or np.any(~np.isfinite(bounds))
            or np.any(bounds[:, 0] > bounds[:, 1])
        ):
            raise ValueError("A singularity query requires one finite ordered 3D box.")
        center = np.asarray(self.center, dtype=np.float64)
        contains = (bounds[:, 0] <= center) & (center <= bounds[:, 1])
        match self.family:
            case "sphere":
                return bool(np.any(np.all(contains, axis=1)))
            case "ring-torus":
                if np.any(contains[:, 0] & contains[:, 1]):
                    return True
                selected = np.flatnonzero(contains[:, 2])
                if not selected.size:
                    return False
                from .._interval_enclosure import interval_subtract

                lower = bounds[selected, 0, :2]
                upper = bounds[selected, 1, :2]
                center_xy = np.broadcast_to(center[:2], lower.shape)
                closest = np.where(
                    contains[selected, :2],
                    center_xy,
                    np.where(lower > center_xy, lower, upper),
                )
                nearest_difference = interval_subtract(
                    (closest, closest), (center_xy, center_xy)
                )
                lower_difference = interval_subtract(
                    (lower, lower), (center_xy, center_xy)
                )
                upper_difference = interval_subtract(
                    (upper, upper), (center_xy, center_xy)
                )
                nearest_squared = interval_multiply(
                    nearest_difference, nearest_difference
                )
                lower_squared = interval_multiply(lower_difference, lower_difference)
                upper_squared = interval_multiply(upper_difference, upper_difference)
                nearest_low, nearest_high = (
                    np.asarray(value, dtype=np.float64) for value in nearest_squared
                )
                far_low = np.maximum(lower_squared[0], upper_squared[0])
                far_high = np.maximum(lower_squared[1], upper_squared[1])
                minimum = interval_add(
                    (nearest_low[:, 0], nearest_high[:, 0]),
                    (nearest_low[:, 1], nearest_high[:, 1]),
                )
                maximum = interval_add(
                    (far_low[:, 0], far_high[:, 0]),
                    (far_low[:, 1], far_high[:, 1]),
                )
                radius = np.asarray([self.major_radius], dtype=np.float64)
                radius_squared = interval_multiply((radius, radius), (radius, radius))
                minimum_low, minimum_high = (np.asarray(value) for value in minimum)
                maximum_low, maximum_high = (np.asarray(value) for value in maximum)
                radius_low, radius_high = (np.asarray(value) for value in radius_squared)
                outside = (radius_high < minimum_low) | (radius_low > maximum_high)
                inside = (minimum_high < radius_low) & (radius_high < maximum_low)
                if np.any(inside):
                    return True
                exact_center = tuple(
                    Fraction.from_float(float(value)) for value in center[:2]
                )
                exact_radius = Fraction.from_float(self.major_radius)
                exact_radius_squared = exact_radius * exact_radius
                for row in np.flatnonzero(~outside):
                    exact_lower = tuple(
                        Fraction.from_float(float(value)) for value in lower[row]
                    )
                    exact_upper = tuple(
                        Fraction.from_float(float(value)) for value in upper[row]
                    )
                    minimum_distance = tuple(
                        Fraction(0)
                        if exact_lower[axis] <= exact_center[axis] <= exact_upper[axis]
                        else min(
                            abs(exact_lower[axis] - exact_center[axis]),
                            abs(exact_upper[axis] - exact_center[axis]),
                        )
                        for axis in range(2)
                    )
                    maximum_distance = tuple(
                        max(
                            abs(exact_lower[axis] - exact_center[axis]),
                            abs(exact_upper[axis] - exact_center[axis]),
                        )
                        for axis in range(2)
                    )
                    minimum_radius_squared = sum(
                        (value * value for value in minimum_distance), Fraction(0)
                    )
                    maximum_radius_squared = sum(
                        (value * value for value in maximum_distance), Fraction(0)
                    )
                    if (
                        minimum_radius_squared
                        <= exact_radius_squared
                        <= maximum_radius_squared
                    ):
                        return True
                return False
            case _ as unreachable:
                assert_never(unreachable)

    def boundary_distance(self, points: ArrayLike, /) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("points must be finite with shape (N, 3).")
        self.validate_source_integrity()
        # The field is the exact signed distance; its outward interval at each
        # point encloses the distance after taking the absolute value.
        value_lower, value_upper = self.field_bounds.point_values(values)
        value_lower = value_lower.reshape((-1,))
        value_upper = value_upper.reshape((-1,))
        straddles = (value_lower <= 0.0) & (value_upper >= 0.0)
        lower = np.where(
            straddles, 0.0, np.minimum(np.abs(value_lower), np.abs(value_upper))
        )
        upper = np.maximum(np.abs(value_lower), np.abs(value_upper))
        return SourceBoundaryDistance(lower, upper, "certified")

    def triangle_distance_upper(self, triangles: ArrayLike, /) -> np.ndarray:
        """Certified upper bound of the distance to the source over each triangle.

        ``triangles`` has shape ``(N, 3, 3)``. The profile field is a signed
        distance of reach ``R``: at distance ``t < R`` from the zero set the
        principal curvatures of its level sets, hence its Hessian norm, are at
        most ``1 / (R - t)``. A triangle lies within ``t`` = min(vertex
        distance + diameter, interval bound over its box) of the zero set, and
        linear interpolation of the field over it errs by at most the Hessian
        bound times ``diameter^2 / 6``. Every operation rounds outward.
        """
        values = np.asarray(triangles, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[1:] != (3, 3)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("triangles must be finite with shape (N, 3, 3).")
        if values.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)
        # The field is the signed distance: its interval encloses the distance.
        value_lower, value_upper = self.field_bounds.point_values(values.reshape((-1, 3)))
        vertex_distance = np.maximum(np.abs(value_lower), np.abs(value_upper))
        error = np.max(vertex_distance.reshape((-1, 3)), axis=1)
        diameter = _triangle_diameter_upper(values)
        boxes = self.field_bounds.boxes(np.min(values, axis=1), np.max(values, axis=1))
        box_bound = np.maximum(np.abs(boxes.value_lower), np.abs(boxes.value_upper))
        distance = np.minimum(np.nextafter(error + diameter, np.inf), box_bound)
        in_tube = distance < self.reach_lower
        margin = np.nextafter(
            np.where(in_tube, self.reach_lower - distance, 1.0), -np.inf
        )
        hessian = np.nextafter(1.0 / margin, np.inf)
        curvature = np.nextafter(
            np.nextafter(hessian * np.nextafter(diameter * diameter, np.inf), np.inf)
            / 6.0,
            np.inf,
        )
        interpolation = np.nextafter(error + curvature, np.inf)
        return np.where(in_tube, np.minimum(box_bound, interpolation), box_bound)

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
        distance = self.boundary_distance(values)
        lower, upper = distance.lower, distance.upper
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
                length = _decided_norm(tuple(float(value) for value in relative))
                if length == 0.0:
                    status[row] = SourceProjectionStatus.NON_ISOLATED
                    continue
                normal = relative / length
                longitude = _decided_atan2(float(relative[1]), float(relative[0]))
                if longitude < 0.0:
                    longitude += math.tau
                latitude = _decided_atan2(
                    float(relative[2]),
                    _decided_norm((float(relative[0]), float(relative[1]))),
                )
                polar = math.pi / 2.0 - latitude
                point = center + self.radius * normal
                cosines, sines = _decided_trigonometric(
                    np.asarray((longitude, latitude), dtype=np.float64)
                )
                cos_longitude, cos_latitude = cosines
                sin_longitude, sin_latitude = sines
                longitude_tangent = np.asarray(
                    (-sin_longitude, cos_longitude, 0.0),
                    dtype=np.float64,
                )
                latitude_tangent = np.asarray(
                    (
                        -sin_latitude * cos_longitude,
                        -sin_latitude * sin_longitude,
                        cos_latitude,
                    ),
                    dtype=np.float64,
                )
                first_tangent, second_tangent = -latitude_tangent, longitude_tangent
            else:
                x, y, z = (float(value) for value in relative)
                radial = _decided_norm((x, y))
                tube_x, tube_y = radial - self.major_radius, z
                tube_length = _decided_norm((tube_x, tube_y))
                if radial == 0.0 or tube_length == 0.0:
                    status[row] = SourceProjectionStatus.NON_ISOLATED
                    continue
                longitude = _decided_atan2(y, x)
                latitude = _decided_atan2(tube_y, tube_x)
                if longitude < 0.0:
                    longitude += math.tau
                if latitude < 0.0:
                    latitude += math.tau
                cosines, sines = _decided_trigonometric(
                    np.asarray((longitude, latitude), dtype=np.float64)
                )
                cos_longitude, cos_latitude = cosines
                sin_longitude, sin_latitude = sines
                ring_radius = self.major_radius + self.radius * cos_latitude
                height = self.radius * sin_latitude
                radial_direction = np.asarray(
                    (cos_longitude, sin_longitude, 0.0), dtype=np.float64
                )
                normal = cos_latitude * radial_direction + sin_latitude * np.asarray(
                    (0.0, 0.0, 1.0)
                )
                point = (
                    center
                    + ring_radius * radial_direction
                    + np.asarray((0.0, 0.0, height))
                )
                first_tangent = np.asarray(
                    (-sin_longitude, cos_longitude, 0.0),
                    dtype=np.float64,
                )
                second_tangent = np.asarray(
                    (
                        -sin_latitude * cos_longitude,
                        -sin_latitude * sin_longitude,
                        cos_latitude,
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

    def certify_zero_set_cover(
        self, vertices: ArrayLike, triangles: ArrayLike, tolerance: float, /
    ) -> tuple[float, ZeroSetReverseProof]:
        """Bound the distance from every boundary point to a closed surface.

        The closed surface (``vertices`` rows indexed by ``triangles``, oriented
        outward) is certified by the native zero-set certificate against the
        boundary field ``f``, which is negative inside, so outward is the
        orientation of ``grad f``: every triangle within ``tolerance`` of
        ``{f = 0}`` and transversal, then every zero of ``f`` in the boundary
        box padded by twice the tolerance (it holds the whole zero set and
        every surface vertex) within the returned distance of the surface,
        proved by the normal tube or by the possible-zero cover alone (the
        returned proof). Every cover leaf must lie near the surface, which is
        what proves every boundary component is reached. Requires the native
        meshcore library; a surface the certificate would repair, and every
        refusal, raise ``ValueError``.
        """
        from ..._meshcore import MeshcoreError, MeshcoreUnavailableError, zero_set_certify
        from ._zero_set_programs import implicit_zero_set_programs

        bound = float(tolerance)
        if not math.isfinite(bound) or not 0.0 < bound < self.reach_lower:
            raise ValueError(
                "tolerance must be finite, positive, and below the established reach."
            )
        points = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(triangles, dtype=np.int64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or faces.ndim != 2
            or faces.shape[1] != 3
            or faces.shape[0] == 0
        ):
            raise ValueError("The surface must be (N, 3) vertices and (M, 3) triangles.")
        edges = points[faces] - points[np.roll(faces, -1, axis=1)]
        edge_length = float(np.mean(np.sqrt(np.sum(edges * edges, axis=-1))))
        pad = np.nextafter(2.0 * bound, np.inf)
        bounds = np.asarray(self.boundary_bounds, dtype=np.float64)
        domain = np.stack(
            (
                np.nextafter(bounds[0] - pad, -np.inf),
                np.nextafter(bounds[1] + pad, np.inf),
            )
        )
        try:
            value, gradient = implicit_zero_set_programs(self.geometry)
            certificate = zero_set_certify(
                points,
                faces,
                value,
                gradient,
                bound,
                1,
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int64),
                np.empty((0, 2, 3), dtype=np.float64),
                domain,
                edge_length,
                _ZERO_SET_REPAIR_ANGLE_DEGREES,
            )
        except (
            IntervalExtensionError,
            MeshcoreError,
            MeshcoreUnavailableError,
            RuntimeError,
        ) as error:
            raise ValueError(
                f"The zero-set certificate does not hold for the surface: {error}"
            ) from error
        if int(certificate.counters[4]) != 0 or not (
            np.array_equal(certificate.vertices, points)
            and np.array_equal(certificate.triangles, faces)
        ):
            raise ValueError(
                "The zero-set certificate holds only for a repaired surface, not this one."
            )
        proof: ZeroSetReverseProof = (
            "tube" if int(certificate.counters[8]) == 1 else "cover"
        )
        return certificate.reverse_distance, proof


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

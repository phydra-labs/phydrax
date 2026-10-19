#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Established global facts for nominal sphere and full ring-torus sources.

The source kernels, design state, physical frame and units are bound explicitly.
For a sphere the signed-distance medial locus is its center and the reach is r.
For a ring torus the two relevant singular loci are its core circle and axis;
the reach is min(r, R-r). In a signed-distance tube of radius d below that
reach, the Hessian eigenvalues are bounded by 1/(reach-d): the meridional
curvature is bounded by 1/(r-d), and the azimuthal curvature by
1/(R-r-d). These facts derive from the nominal kernel formulas, not a
user declaration, a sampled topology or an unchecked declaration.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.geometry._source_boundary_records import (
    SourceBoundaryDistance,
    SourceBoundarySamples,
)

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, HostFloat64, parse
from .._atlas import BoundaryAtlas
from .._certified_implicit import implicit_state_id
from .._contracts import CompiledGeometry
from ..analytic._extended import TorusKernel
from ..analytic._primitives import BallKernel
from ._enclosure import (
    BOX_CHUNK,
    consume_queries,
    derived_interval_program,
    ENCLOSURE_ROUNDING_MODEL,
    IntervalFieldBounds,
    IntervalProgram,
    query_budget,
)


AnalyticImplicitFamily: TypeAlias = Literal["sphere", "ring-torus"]


def _lower_fraction(value: Fraction, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            "AnalyticImplicitProfile.bound must be representable in binary64; got a nonfinite rounded bound"
        )
    return float(np.nextafter(result, -np.inf)) if Fraction(result) > value else result


def _upper_fraction(value: Fraction, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            "AnalyticImplicitProfile.bound must be representable in binary64; got a nonfinite rounded bound"
        )
    return float(np.nextafter(result, np.inf)) if Fraction(result) < value else result


def _norm_upper(values: np.ndarray, /) -> np.ndarray:
    # Correctly rounded elementary operations are widened individually.
    scale = np.max(values, axis=-1)
    divisor = np.where(scale > 0.0, scale, 1.0)
    normalized = np.nextafter(values / divisor[..., None], np.inf)
    square = np.nextafter(normalized * normalized, np.inf)
    total = np.zeros(values.shape[:-1], dtype=np.float64)
    for axis in range(values.shape[-1]):
        total = np.nextafter(total + square[..., axis], np.inf)
    return np.nextafter(np.nextafter(np.sqrt(total), np.inf) * scale, np.inf)


def _schema_id(geometry: CompiledGeometry, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "analytic-source-design-schema",
            "parameters": tuple(
                (
                    spec.parameter_id.feature_id,
                    spec.parameter_id.name,
                    spec.shape,
                    spec.dtype,
                    spec.role,
                    spec.unit.unit_id,
                    spec.physical_scale,
                    spec.bounds,
                    spec.trainable,
                )
                for spec in geometry.schema.specs
            ),
        }
    )


def _kernel_id(geometry: CompiledGeometry, /) -> str:
    kernel = geometry.kernel
    match kernel:
        case BallKernel():
            if type(kernel) is not BallKernel:
                raise ValueError(
                    "AnalyticImplicitProfile.kernel must be the nominal BallKernel; got a derived ball kernel"
                )
            bindings = (kernel.center, kernel.radius)
            names = ("center", "radius")
            structural = ("sphere", kernel.dimension)
        case TorusKernel():
            if type(kernel) is not TorusKernel:
                raise ValueError(
                    "AnalyticImplicitProfile.kernel must be the nominal TorusKernel; got a derived torus kernel"
                )
            bindings = (kernel.center, kernel.major, kernel.minor, kernel.angle)
            names = ("center", "major_radius", "minor_radius", "angle")
            structural = ("ring-torus", kernel.full)
        case _:
            raise ValueError(
                f"AnalyticImplicitProfile.kernel must be a supported nominal ball or ring torus; got {type(kernel).__name__}"
            )
    if any(
        binding.parameter_id.feature_id != kernel.source_id
        or binding.parameter_id.name != name
        or geometry.schema.index(binding.parameter_id) != binding.index
        for binding, name in zip(bindings, names, strict=True)
    ):
        raise ValueError(
            "AnalyticImplicitProfile.bindings must match source IDs, parameter names and schema indices; got inconsistent bindings"
        )
    return canonical_fingerprint(
        {
            "kind": "analytic-source-kernel",
            "represented_geometry": kernel.source_id,
            "structure": structural,
            "bindings": tuple(
                (
                    binding.parameter_id.feature_id,
                    binding.parameter_id.name,
                    binding.index,
                )
                for binding in bindings
            ),
        }
    )


class AnalyticBoundaryCoverCapacityError(ValueError):
    """An exhaustive cover exists but cannot meet the requested radius in budget."""

    requested_radius: float
    achieved_radius: float
    maximum_samples: int

    def __init__(
        self, requested_radius: float, achieved_radius: float, maximum_samples: int, /
    ) -> None:
        self.requested_radius = requested_radius
        self.achieved_radius = achieved_radius
        self.maximum_samples = maximum_samples
        super().__init__(
            f"Analytic source cover radius {achieved_radius:g} exceeds "
            f"{requested_radius:g} within {maximum_samples} samples."
        )


@final
class _BoundaryMapBounds(StrictModule, NonTrainableState):
    """Reuse the outward-rounded JAX-expression interpreter for charts."""

    atlas: BoundaryAtlas
    point_dtype: str = eqx.field(static=True)
    value_dtype: str = eqx.field(static=True)
    rounding_model: str = eqx.field(static=True)

    @checked
    def __init__(self, atlas: BoundaryAtlas, /) -> None:
        if atlas.reference_dimension != 2 or atlas.ambient_dimension != 3:
            raise TypeError(
                "analytic_bounds.atlas must map two reference dimensions to three spatial dimensions; got incompatible dimensions"
            )

        def point_map(reference: Array) -> Array:
            return atlas.map(jnp.zeros((1,), dtype=jnp.int32), reference[None, :])[0]

        point_dtype = np.dtype(np.float64).name
        program = derived_interval_program(
            atlas,
            point_map,
            (id(type(atlas).map), id(type(atlas.mapping).map)),
            2,
            (3,),
            np.zeros((0, 2), dtype=np.float64),
            point_dtype=point_dtype,
        )
        self.atlas = atlas
        self.point_dtype = point_dtype
        self.value_dtype = program.value_dtype
        self.rounding_model = ENCLOSURE_ROUNDING_MODEL

    def _validated_program(self) -> IntervalProgram:
        if (
            not isinstance(self.atlas, BoundaryAtlas)
            or self.atlas.reference_dimension != 2
            or self.atlas.ambient_dimension != 3
            or self.point_dtype != np.dtype(np.float64).name
            or self.rounding_model != ENCLOSURE_ROUNDING_MODEL
        ):
            raise ValueError(
                "AnalyticBoundaryMap.state must retain a two-to-three dimensional atlas and binary64 rounding model; got invalid restored dimensions or precision"
            )

        def point_map(reference: Array) -> Array:
            # The sphere square continuously extends the same two triangular
            # charts through their shared atlas map.
            return self.atlas.map(jnp.zeros((1,), dtype=jnp.int32), reference[None, :])[0]

        return derived_interval_program(
            self.atlas,
            point_map,
            (id(type(self.atlas).map), id(type(self.atlas.mapping).map)),
            2,
            (3,),
            np.zeros((0, 2), dtype=np.float64),
            point_dtype=self.point_dtype,
            value_dtype=self.value_dtype,
        )

    def validate_source_integrity(self) -> None:
        self._validated_program()

    def boxes(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, np.ndarray]:
        consume_queries(budget, 0, work_units=1)
        program = self._validated_program()
        if lower.ndim != 2 or lower.shape[1] != 2 or upper.shape != lower.shape:
            raise ValueError(
                f"AnalyticBoundaryMap.boxes must have matching shape (N, 2); got {lower.shape} and {upper.shape}"
            )
        consume_queries(budget, lower.shape[0])
        mapped_lower = np.empty((lower.shape[0], 3), dtype=np.float64)
        mapped_upper = np.empty_like(mapped_lower)
        for start in range(0, lower.shape[0], BOX_CHUNK):
            stop = min(start + BOX_CHUNK, lower.shape[0])
            low, high, _, _ = program.evaluate(
                jnp.asarray(lower[start:stop]), jnp.asarray(upper[start:stop])
            )
            mapped_lower[start:stop] = np.asarray(low, dtype=np.float64)
            mapped_upper[start:stop] = np.asarray(high, dtype=np.float64)
            consume_queries(budget, 0, work_units=1)
        return mapped_lower, mapped_upper


@final
class AnalyticImplicitProfile(StrictModule, NonTrainableState):
    """Establish topology, reach and rigorous distance queries.

    Three-dimensional ball kernels and full ring-torus kernels
    qualify. Boolean expressions, transformed wrappers, arbitrary callables,
    sectors and singular horn/spindle tori cannot inherit these facts. Radius
    values and the full sweep are read from the current DesignState; the static
    kernel family alone is not validity evidence.
    """

    __strict_contract__ = True

    geometry: CompiledGeometry
    coordinate_contract: SpatialCoordinateContract
    field_bounds: IntervalFieldBounds
    map_bounds: _BoundaryMapBounds
    center: HostFloat64[Literal[3]]
    boundary_bounds: HostFloat64[Literal[2], Literal[3]]
    interior_witness: HostFloat64[Literal[3]]
    family: AnalyticImplicitFamily = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    major_radius: float = eqx.field(static=True)
    reach_lower: float = eqx.field(static=True)
    tube_radius: float = eqx.field(static=True)
    cover_radius: float = eqx.field(static=True)
    hessian_upper: float = eqx.field(static=True)
    normal_lipschitz_upper: float = eqx.field(static=True)
    gradient_norm_bounds: tuple[float, float] = eqx.field(static=True)
    interior_margin_lower: float = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    genus: int = eqx.field(static=True)
    boundary_euler_characteristic: int = eqx.field(static=True)
    represented_geometry_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    rounding_model: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        tube_radius: float | None = None,
        cover_radius: float | None = None,
        source_id: str | None = None,
        source_revision: str | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        validity = geometry.validity()
        if not bool(np.asarray(validity.accepted)):
            raise ValueError(
                "AnalyticImplicitProfile.geometry must have accepted DesignState validity; got a rejected state"
            )
        kernel = geometry.kernel
        kernel_id = _kernel_id(geometry)
        match kernel:
            case BallKernel():
                if kernel.dimension != 3:
                    raise ValueError(
                        f"AnalyticImplicitProfile.kernel.dimension must be three for a sphere volume; got {kernel.dimension}"
                    )
                center = np.asarray(kernel.center.read(geometry.state), dtype=np.float64)
                radius = float(np.asarray(kernel.radius.read(geometry.state)))
                major = 0.0
                family: AnalyticImplicitFamily = "sphere"
                reach = Fraction(radius)
                outer = reach
                genus = 0
                witness = center.copy()
            case TorusKernel():
                center = np.asarray(kernel.center.read(geometry.state), dtype=np.float64)
                radius = float(np.asarray(kernel.minor.read(geometry.state)))
                major = float(np.asarray(kernel.major.read(geometry.state)))
                angle = float(np.asarray(kernel.angle.read(geometry.state)))
                if not kernel.full or angle != 2.0 * math.pi or not major > radius > 0.0:
                    raise ValueError(
                        f"AnalyticImplicitProfile.kernel must be a full nonsingular ring torus; got full={kernel.full}, angle={angle}, major={major}, minor={radius}"
                    )
                family = "ring-torus"
                reach = min(Fraction(radius), Fraction(major) - Fraction(radius))
                outer = Fraction(major) + Fraction(radius)
                genus = 1
                witness = center.copy()
                witness[0] += major
            case _:
                raise ValueError(
                    f"AnalyticImplicitProfile.kernel must have a supported analytic global profile; got {type(kernel).__name__}"
                )
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or not math.isfinite(radius)
            or radius <= 0.0
        ):
            raise ValueError(
                f"AnalyticImplicitProfile.parameters must have a finite three-dimensional center and positive finite radius; got center shape {center.shape} and radius {radius}"
            )
        reach_low = _lower_fraction(reach)
        if reach_low <= 0.0:
            raise ValueError(
                f"AnalyticImplicitProfile.reach_lower must be positive in binary64; got {reach_low}"
            )
        tube = (
            _lower_fraction(Fraction(reach_low) / 4)
            if tube_radius is None
            else float(tube_radius)
        )
        if not math.isfinite(tube) or not 0.0 < tube < reach_low:
            raise ValueError(
                f"AnalyticImplicitProfile.tube_radius must be finite, positive and below reach {reach_low}; got {tube}"
            )
        cover = tube / 4 if cover_radius is None else float(cover_radius)
        if not math.isfinite(cover) or cover <= 0.0:
            raise ValueError(
                f"AnalyticImplicitProfile.cover_radius must be finite and positive; got {cover}"
            )
        hessian = _upper_fraction(1 / (reach - Fraction(tube)))
        extents = (
            (outer, outer, Fraction(radius)) if family == "ring-torus" else (outer,) * 3
        )
        bounds = np.asarray(
            [
                [
                    _lower_fraction(Fraction(value) - extent)
                    for value, extent in zip(center.tolist(), extents, strict=True)
                ],
                [
                    _upper_fraction(Fraction(value) + extent)
                    for value, extent in zip(center.tolist(), extents, strict=True)
                ],
            ],
            dtype=np.float64,
        )
        field_bounds = IntervalFieldBounds(geometry)
        _, upper = field_bounds.point_values(witness[None, :], budget=budget)
        margin = float(-upper[0])
        if not math.isfinite(margin) or margin <= 0.0:
            raise ValueError(
                f"AnalyticImplicitProfile.interior_margin must establish positive finite interior separation; got {margin}"
            )
        state_id = implicit_state_id(geometry)
        schema_id = _schema_id(geometry)
        revision = canonical_fingerprint(
            {
                "kind": "analytic-source-state",
                "state": state_id,
                "schema": schema_id,
                "kernel": kernel_id,
                "coordinates": coordinate_contract.spatial_id,
                "represented_geometry": kernel.source_id,
            }
        )
        source = (
            kernel.source_id
            if source_id is None
            else canonical_identifier(source_id, "source_id")
        )
        external_revision = (
            revision
            if source_revision is None
            else canonical_identifier(source_revision, "source_revision")
        )
        self.geometry = geometry
        self.coordinate_contract = coordinate_contract
        self.field_bounds = field_bounds
        self.map_bounds = _BoundaryMapBounds(geometry.boundary_atlas)
        self.center = parse(center, HostFloat64[Literal[3]], "center")
        self.boundary_bounds = parse(
            bounds, HostFloat64[Literal[2], Literal[3]], "boundary_bounds"
        )
        self.interior_witness = parse(
            witness, HostFloat64[Literal[3]], "interior_witness"
        )
        self.family = parse(family, AnalyticImplicitFamily, "family")
        self.radius = radius
        self.major_radius = major
        self.reach_lower = reach_low
        self.tube_radius = tube
        self.cover_radius = cover
        self.hessian_upper = hessian
        self.normal_lipschitz_upper = hessian
        self.gradient_norm_bounds = (1.0, 1.0)
        self.interior_margin_lower = margin
        self.component_count = 1
        self.genus = genus
        self.boundary_euler_characteristic = 2 - 2 * genus
        self.represented_geometry_id = kernel.source_id
        self.source_id = source
        self.source_revision = external_revision
        self.state_id = state_id
        self.schema_id = schema_id
        self.kernel_id = kernel_id
        self.rounding_model = ENCLOSURE_ROUNDING_MODEL
        self.profile_id = canonical_fingerprint(
            {
                "kind": "analytic-implicit-profile",
                "source": source,
                "revision": external_revision,
                "state": state_id,
                "schema": schema_id,
                "kernel": kernel_id,
                "represented_geometry": kernel.source_id,
                "coordinates": coordinate_contract.spatial_id,
                "family": family,
                "reach": reach_low,
                "tube": tube,
                "cover_radius": cover,
                "hessian": hessian,
                "bounds": array_tree_fingerprint(bounds),
                "witness": array_tree_fingerprint(witness),
                "interior_margin": margin,
                "rounding": self.rounding_model,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    def validate_source_integrity(
        self, *, budget: CoordinateEnclosureBudget | None = None
    ) -> None:
        """Recompute restored source facts and compare their retained values."""
        self.field_bounds.validate_source_integrity()
        self.map_bounds.validate_source_integrity()
        expected = AnalyticImplicitProfile(
            self.geometry,
            self.coordinate_contract,
            tube_radius=self.tube_radius,
            cover_radius=self.cover_radius,
            source_id=self.source_id,
            source_revision=self.source_revision,
            budget=budget,
        )
        if not bool(np.asarray(eqx.tree_equal(self, expected, typematch=True))):
            raise ValueError(
                "AnalyticImplicitProfile.state must match recomputed scientific facts; got inconsistent scientific facts"
            )

    def require_bound(
        self,
        geometry: CompiledGeometry,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if (
            implicit_state_id(geometry) != self.state_id
            or _schema_id(geometry) != self.schema_id
            or _kernel_id(geometry) != self.kernel_id
            or coordinate_contract.spatial_id != self.coordinate_contract.spatial_id
            or type(geometry.kernel) is not type(self.geometry.kernel)
        ):
            raise ValueError(
                "AnalyticImplicitProfile.binding must match source, DesignState and physical contract; got a stale source, DesignState or physical contract"
            )
        self.validate_source_integrity(budget=budget)

    def boundary_distance(
        self, points: ArrayLike, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> SourceBoundaryDistance:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"AnalyticImplicitProfile.points must be finite with shape (N, 3); got shape {values.shape} or nonfinite coordinates"
            )
        with query_budget(
            budget, work_upper=0, queries_upper=values.shape[0] + 1
        ) as query:
            self.require_bound(self.geometry, self.coordinate_contract, budget=query)
            lower, upper = self.field_bounds.point_values(values, budget=query)
            crosses = (lower <= 0.0) & (upper >= 0.0)
            distance_lower = np.where(
                crosses, 0.0, np.minimum(np.abs(lower), np.abs(upper))
            )
            return SourceBoundaryDistance(
                distance_lower, np.maximum(np.abs(lower), np.abs(upper)), "certified"
            )

    def normal_variation_upper(self, distance: float, /) -> float:
        value = float(distance)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"AnalyticImplicitProfile.separation must be finite and nonnegative; got {value}"
            )
        return _upper_fraction(
            min(Fraction(2), Fraction(self.normal_lipschitz_upper) * Fraction(value))
        )

    def _sample_grid(
        self,
        first_count: int,
        second_count: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> SourceBoundarySamples:
        """Enclose every rectangle in one exhaustive periodic/polar parameter grid."""
        match self.family:
            case "sphere":
                polar = np.linspace(0.0, math.pi, second_count + 1, dtype=np.float64)
                second = 0.5 * (1.0 - np.cos(polar))
                second[0], second[-1] = 0.0, 1.0
            case "ring-torus":
                second = np.linspace(0.0, 1.0, second_count + 1, dtype=np.float64)
            case _:
                raise ValueError(
                    f"AnalyticImplicitProfile.family must select a supported profile; got {self.family!r}"
                )
        first = np.linspace(0.0, 1.0, first_count + 1, dtype=np.float64)
        if np.any(np.diff(first) <= 0.0) or np.any(np.diff(second) <= 0.0):
            raise ValueError(
                "AnalyticImplicitProfile.sample_grid must have strictly increasing representable coordinates; got a collapsed interval"
            )
        count = first_count * second_count
        consume_queries(budget, count)
        # Interval-map outputs, points, radii and errors remain live together.
        storage = 88 * count
        if storage > budget.ticket.bytes_upper:
            budget.ticket.refuse(0, bytes_upper=storage)
        budget.ticket.consume(managed_bytes=storage)
        lower = np.stack(
            np.meshgrid(first[:-1], second[:-1], indexing="ij"), axis=-1
        ).reshape((-1, 2))
        upper = np.stack(
            np.meshgrid(first[1:], second[1:], indexing="ij"), axis=-1
        ).reshape((-1, 2))
        reference = 0.5 * lower + 0.5 * upper
        mapped = self.map_bounds.atlas.map(
            jnp.zeros((reference.shape[0],), dtype=jnp.int32),
            jnp.asarray(reference),
        )
        points = np.asarray(mapped, dtype=np.float64)
        image_lower, image_upper = self.map_bounds.boxes(lower, upper, budget=budget)
        displacement = np.maximum(
            np.abs(points - image_lower), np.abs(image_upper - points)
        )
        radius = _norm_upper(np.nextafter(displacement, np.inf))
        pi_low = Fraction(float(np.nextafter(math.pi, -np.inf)))
        pi_high = Fraction(float(np.nextafter(math.pi, np.inf)))
        period_error = 4 * (pi_high - pi_low)
        scale = (
            Fraction(self.radius)
            if self.family == "sphere"
            else Fraction(self.major_radius) + 2 * Fraction(self.radius)
        )
        gap = _upper_fraction(period_error * scale)
        radius = np.nextafter(radius + gap, np.inf)
        errors = self.boundary_distance(points, budget=budget).upper
        return SourceBoundarySamples(points, radius, errors, "certified", complete=True)

    def _precision_grid(self, maximum_radius: float, /) -> tuple[int, int]:
        # Interval images establish the achieved cover radius; this estimate
        # bounds the work admitted before checking precision.
        pi_upper = Fraction(float(np.nextafter(math.pi, np.inf)))
        precision = Fraction(maximum_radius)
        match self.family:
            case "sphere":
                first = 2 * pi_upper * Fraction(self.radius) / precision
                second = pi_upper * Fraction(self.radius) / precision
            case "ring-torus":
                first = (
                    2
                    * pi_upper
                    * (Fraction(self.major_radius) + Fraction(self.radius))
                    / precision
                )
                second = 2 * pi_upper * Fraction(self.radius) / precision
            case _:
                raise ValueError(
                    f"AnalyticImplicitProfile.family must select a supported profile; got {self.family!r}"
                )
        return max(1, math.ceil(first)), max(1, math.ceil(second))

    def _budget_grid(self, maximum_samples: int, /) -> tuple[int, int]:
        match self.family:
            case "sphere":
                second = max(1, math.isqrt(max(1, maximum_samples // 2)))
                first = max(1, maximum_samples // second)
            case "ring-torus":
                ratio = (
                    Fraction(maximum_samples)
                    * (Fraction(self.major_radius) + Fraction(self.radius))
                    / Fraction(self.radius)
                )
                first = min(
                    maximum_samples,
                    max(1, math.isqrt(ratio.numerator // ratio.denominator)),
                )
                second = max(1, maximum_samples // first)
            case _:
                raise ValueError(
                    f"AnalyticImplicitProfile.family must select a supported profile; got {self.family!r}"
                )
        return first, second

    def boundary_cover(
        self,
        maximum_radius: float,
        maximum_samples: int,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SourceBoundarySamples:
        """Establish a requested physical-radius cover or report counted capacity."""
        requested = float(maximum_radius)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError(
                f"AnalyticImplicitProfile.maximum_radius must be finite and positive; got {requested}"
            )
        capacity = positive_integer(maximum_samples, "maximum_samples")
        # Trial grids grow by four: their total plus the fallback is < 7N/3.
        with query_budget(
            budget,
            work_upper=2 * capacity + 2 * capacity.bit_length(),
            queries_upper=7 * capacity + 2 * capacity.bit_length(),
            bytes_upper=88 * capacity,
        ) as query:
            self.require_bound(self.geometry, self.coordinate_contract, budget=query)
            first, second = self._precision_grid(requested)
            while first * second <= capacity:
                samples = self._sample_grid(first, second, budget=query)
                achieved = float(np.max(samples.covering_radius))
                if achieved <= requested:
                    return samples
                first *= 2
                second *= 2
            first, second = self._budget_grid(capacity)
            samples = self._sample_grid(first, second, budget=query)
            achieved = float(np.max(samples.covering_radius))
            if achieved <= requested:
                return samples
            raise AnalyticBoundaryCoverCapacityError(requested, achieved, capacity)

    def boundary_samples(
        self, maximum_samples: int, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> SourceBoundarySamples:
        """Protocol query: exhaustive covers retain their achieved coarse bounds."""
        capacity = positive_integer(maximum_samples, "maximum_samples")
        with query_budget(
            budget,
            work_upper=2 * capacity + 2 * capacity.bit_length(),
            queries_upper=7 * capacity + 2 * capacity.bit_length(),
            bytes_upper=88 * capacity,
        ) as query:
            self.require_bound(self.geometry, self.coordinate_contract, budget=query)
            first, second = self._precision_grid(self.cover_radius)
            while first * second <= capacity:
                samples = self._sample_grid(first, second, budget=query)
                if float(np.max(samples.covering_radius)) <= self.cover_radius:
                    return samples
                first *= 2
                second *= 2
            return self._sample_grid(*self._budget_grid(capacity), budget=query)


__all__ = [
    "AnalyticBoundaryCoverCapacityError",
    "AnalyticImplicitFamily",
    "AnalyticImplicitProfile",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared native queries over the exact geometry of a native B-Rep.

Closest points use conservative source-carrier interval covers for discovery,
native Newton proposals, and bounded interval subdivision for host certification.
Traced execution retains the cover's lower bound and reports unresolved work
instead of treating a finite seed set as a completeness certificate.

Measures use Green-reduced Gauss--Legendre quadrature over the exact trim
loops: ``int_D g du dv = oint_{dD} G dv`` with ``G(u, v) = int_{u_0}^u g(s, v)
ds``, integrated by nested rules along every p-curve. Inner rules respect the
original surface's native knot walls. Point containment in a closed solid uses
a complete exact affine-source, sphere or closed-meridian revolution theorem,
or bounded native ray parity. Diagnostic winding never certifies membership.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from math import pi
from threading import Lock
from typing import final, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from ..._bvh import bvh_nearest_items, PackedBVH, prepare_bvh
from ..._fingerprint import canonical_fingerprint
from ..._numerics._quadrature_rules import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer
from ...linalg import inverse_small_linear, SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import VectorLocalRootPlan
from .._atlas import CurveTrimLoop, TrimDomain
from .._interval_enclosure import (
    interval_add,
    interval_multiply,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from .._planar_embedding import PlanarEmbedding
from ._affine_query import AffineFace, QueryBroadphase
from ._constructors import brep_trim_domain
from ._intersection import NativePeriodEndpoint, RootEndpoint
from ._intersection_curve import (
    _period_offset,
    AffinePCurve,
    coefficient_enclosures,
    IntersectionCurve,
    IntersectionPCurve,
    pcurve_periodic_source,
    PeriodicPCurve,
    surface_pieces,
    SurfaceEvaluator,
    SurfaceRegion,
)
from ._model import (
    BRepCurve,
    BRepEntityId,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    BRepPCurve,
)
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ._placed import PlacedCurve, PlacedSurface
from ._projection_contracts import (
    _closure_codes,
    AbstractBRepProjection,
    BRepEntityDimension,
    BRepProjectionPolicy,
    BRepProjectionResult,
    BRepProjectionStatus,
)
from ._revolution_membership import (
    _RevolutionSource,
    classify_full_revolution,
    prepare_full_revolution,
)
from ._sphere_membership import _SphereSource, prepare_full_sphere


_UNIQUE = int(BRepProjectionStatus.UNIQUE)
_SEAM = int(BRepProjectionStatus.SEAM)
_AMBIGUOUS = int(BRepProjectionStatus.AMBIGUOUS)
_FAILED = int(BRepProjectionStatus.FAILED)


class BRepQueryResourceError(RuntimeError):
    """A query refuses owner work before consuming an insufficient allowance."""

    def __init__(self, resource: str, requested: int, remaining: int, /) -> None:
        self.resource, self.requested, self.remaining = resource, requested, remaining
        super().__init__(
            f"B-Rep query {resource} requires {requested}; remaining allowance is {remaining}."
        )


@dataclass(slots=True)
class BRepQueryBudget:
    """Shared host allowance for point, carrier, trim, BVH and interval queries.

    Counts actual owner operations, not Newton steps or native arithmetic
    primitives. Admission reserves the complete worst-case cover before work;
    only executed operations are consumed. Source preparation is not a point
    query and does not consume candidate-pair allowance.
    """

    maximum_operations: int
    maximum_points: int | None = None
    maximum_scratch_bytes: int | None = None
    operations: int = field(init=False, default=0)
    points: int = field(init=False, default=0)
    subdivisions: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.maximum_operations = nonnegative_integer(
            self.maximum_operations, "maximum_operations"
        )
        if self.maximum_points is not None:
            self.maximum_points = nonnegative_integer(
                self.maximum_points, "maximum_points"
            )
        if self.maximum_scratch_bytes is not None:
            self.maximum_scratch_bytes = nonnegative_integer(
                self.maximum_scratch_bytes, "maximum_scratch_bytes"
            )

    def admit(self, operations: int, points: int, scratch_bytes: int, /) -> None:
        operations = nonnegative_integer(operations, "operations")
        points = nonnegative_integer(points, "points")
        scratch_bytes = nonnegative_integer(scratch_bytes, "scratch_bytes")
        for resource, requested, remaining in (
            ("operations", operations, self.maximum_operations - self.operations),
            (
                "points",
                points,
                None
                if self.maximum_points is None
                else self.maximum_points - self.points,
            ),
            ("scratch_bytes", scratch_bytes, self.maximum_scratch_bytes),
        ):
            if remaining is not None and requested > remaining:
                raise BRepQueryResourceError(resource, requested, remaining)

    def consume(
        self, operations: int, /, *, points: int = 0, subdivisions: int = 0
    ) -> None:
        operations = nonnegative_integer(operations, "operations")
        points = nonnegative_integer(points, "points")
        subdivisions = nonnegative_integer(subdivisions, "subdivisions")
        self.admit(operations, points, 0)
        self.operations += operations
        self.points += points
        self.subdivisions += subdivisions


@final
class BRepQueryPolicy(StrictModule, NonTrainableState):
    """Bounded discovery, refinement and quadrature resolution of native queries.

    ``seed_resolution`` controls the initial interval cover; ``seeds`` bounds
    Newton proposals. ``maximum_subdivisions`` and ``maximum_depth`` bound
    host interval certification; unfinished work remains explicit.
    Measures use ``quadrature_order``-point rules on ``quadrature_subdivisions``
    pieces of every coedge, with each inner Green leg partitioned at the owning
    surface's native knot walls. The reported quadrature error is the difference
    to the rule on half as many coedge pieces. ``chunk_size`` bounds the query working
    set of every vectorized batch.
    """

    seed_resolution: int = eqx.field(static=True)
    seeds: int = eqx.field(static=True)
    newton_steps: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    quadrature_subdivisions: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    maximum_subdivisions: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    maximum_operations: int = eqx.field(static=True)
    maximum_points: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        seed_resolution: int = 16,
        seeds: int = 4,
        newton_steps: int = 40,
        quadrature_order: int = 12,
        quadrature_subdivisions: int = 8,
        chunk_size: int = 256,
        maximum_subdivisions: int = 8192,
        maximum_depth: int = 48,
        maximum_operations: int = 50_000_000,
        maximum_points: int = 1_000_000,
        maximum_scratch_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        values = {
            "seed_resolution": seed_resolution,
            "seeds": seeds,
            "newton_steps": newton_steps,
            "quadrature_order": quadrature_order,
            "quadrature_subdivisions": quadrature_subdivisions,
            "chunk_size": chunk_size,
            "maximum_subdivisions": maximum_subdivisions,
            "maximum_depth": maximum_depth,
            "maximum_operations": maximum_operations,
            "maximum_points": maximum_points,
            "maximum_scratch_bytes": maximum_scratch_bytes,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if seed_resolution < 4 or seeds < 1 or newton_steps < 1:
            raise ValueError("Query seeding and refinement budgets are too small.")
        if quadrature_order < 2 or quadrature_subdivisions < 2 or chunk_size < 1:
            raise ValueError("Quadrature and chunk budgets are too small.")
        if maximum_subdivisions < 0 or maximum_depth < 0:
            raise ValueError("Subdivision budgets must be nonnegative.")
        if maximum_operations < 1 or maximum_points < 1 or maximum_scratch_bytes < 1:
            raise ValueError("Query work, point and scratch budgets must be positive.")
        self.seed_resolution = seed_resolution
        self.seeds = seeds
        self.newton_steps = newton_steps
        self.quadrature_order = quadrature_order
        self.quadrature_subdivisions = quadrature_subdivisions
        self.chunk_size = chunk_size
        self.maximum_subdivisions = maximum_subdivisions
        self.maximum_depth = maximum_depth
        self.maximum_operations = maximum_operations
        self.maximum_points = maximum_points
        self.maximum_scratch_bytes = maximum_scratch_bytes
        self.policy_id = canonical_fingerprint({"kind": "native-brep-query", **values})


@final
class _Tolerances(StrictModule, NonTrainableState):
    """Policy metadata and dynamic, translation-independent source length."""

    distance: float = eqx.field(static=True)
    parametric: float = eqx.field(static=True)
    classifier: float = eqx.field(static=True)
    scale: Array
    newton_steps: int = eqx.field(static=True)
    seeds: int = eqx.field(static=True)


# ------------------------------------------------------------------ carriers


class _EdgeData(StrictModule):
    curve: BRepCurve | None
    start_point: Array
    end_point: Array
    seed_parameters: Array
    seed_points: Array
    interval_boxes: Array
    discovery_bvh: PackedBVH
    first: float = eqx.field(static=True)
    last: float = eqx.field(static=True)
    closed: bool = eqx.field(static=True)


class _FaceData(StrictModule):
    patch: AbstractSurfacePatch
    seed_uv: Array
    seed_points: Array
    trim_domain: TrimDomain
    interval_boxes: Array
    discovery_bvh: PackedBVH
    parameter_boxes: Array
    pole_points: Array
    pole_parameters: Array
    pcurves: tuple[BRepPCurve, ...]
    carrier_center: Array
    carrier_projection: Array
    carrier_radii: Array
    lower: tuple[float, float] = eqx.field(static=True)
    upper: tuple[float, float] = eqx.field(static=True)
    periods: tuple[float, float] = eqx.field(static=True)
    orientation: float = eqx.field(static=True)
    coedge_edges: tuple[int, ...] = eqx.field(static=True)
    seam_coedges: tuple[bool, ...] = eqx.field(static=True)
    full_rectangle: bool = eqx.field(static=True)


class _Quadrature(StrictModule):
    """Green-reduced trimmed-face nodes: points, oriented vector areas, areas."""

    points: Array
    vector_areas: Array
    areas: Array
    parameters: Array
    weights: Array
    faces: Array


# ------------------------------------------------------------------ results


@final
class BRepSurfaceQueryResult(StrictModule):
    """Closest boundary points with pointwise projection evidence.

    ``status`` holds :class:`BRepProjectionStatus` values; ``stationarity`` is
    the normalized first-order optimality residual at the returned point.
    ``normals`` are unit outward (oriented) face normals, NaN where undefined.
    """

    points: Array
    parameters: Array
    distances: Array
    faces: Array
    status: Array
    normals: Array
    first_derivatives: Array
    second_derivatives: Array
    stationarity: Array
    distance_lower_bounds: Array
    unresolved: Array
    refinements: Array
    parameter_bounds: Array
    occurrences: Array
    proposal_status: Array
    edge_indices: Array
    edge_parameters: Array
    point_error_bounds: Array
    query_operations: Array
    source_revision: str = eqx.field(static=True)


@final
class BRepContainmentResult(StrictModule):
    """Closed-solid membership with its decision route.

    ``status`` is UNIQUE for a certified inside/outside decision and AMBIGUOUS
    for points on the boundary (within the classifier tolerance) or with an
    unresolved source proof. ``winding`` is NaN when no diagnostic is requested.
    ``distances`` and ``distance_upper_bounds`` bound boundary distance from
    above; the lower bound may be a conservative source-carrier separation
    when membership does not require a closest-point solve.
    """

    inside: Array
    status: Array
    winding: Array
    distances: Array
    unresolved: Array
    distance_lower_bounds: Array
    distance_upper_bounds: Array
    query_operations: Array
    resource_exhausted: Array


@final
class BRepMeasureResult(StrictModule):
    """Trimmed face areas and solid volumes with quadrature error estimates."""

    face_areas: Array
    face_area_errors: Array
    solid_volumes: Array
    solid_volume_errors: Array


# ------------------------------------------------------------ JAX kernels


def _clamp_open(uv: Array, face: _FaceData, /) -> Array:
    lower = jnp.asarray(face.lower)
    upper = jnp.asarray(face.upper)
    periodic = jnp.asarray(face.periods) > 0.0
    return jnp.where(periodic, uv, jnp.clip(uv, lower, upper))


def _wrap(uv: Array, face: _FaceData, /) -> Array:
    lower = jnp.asarray(face.lower)
    periods = jnp.asarray(face.periods)
    wrapped = lower + jnp.mod(uv - lower, jnp.where(periods > 0.0, periods, 1.0))
    return jnp.where(periods > 0.0, wrapped, uv)


type _JInterval = tuple[Array, Array]


def _jround(lower: Array, upper: Array, /) -> _JInterval:
    return (
        jnp.nextafter(jax.lax.stop_gradient(lower), -jnp.inf),
        jnp.nextafter(jax.lax.stop_gradient(upper), jnp.inf),
    )


def _jadd(first: _JInterval, second: _JInterval, /) -> _JInterval:
    return _jround(first[0] + second[0], first[1] + second[1])


def _jsub(first: _JInterval, second: _JInterval, /) -> _JInterval:
    return _jround(first[0] - second[1], first[1] - second[0])


def _jmul(first: _JInterval, second: _JInterval, /) -> _JInterval:
    a, b = first[0] * second[0], first[0] * second[1]
    c, d = first[1] * second[0], first[1] * second[1]
    return _jround(
        jnp.minimum(jnp.minimum(a, b), jnp.minimum(c, d)),
        jnp.maximum(jnp.maximum(a, b), jnp.maximum(c, d)),
    )


def _jdiv(first: _JInterval, second: _JInterval, /) -> _JInterval:
    invalid = (second[0] <= 0.0) & (second[1] >= 0.0)
    reciprocal = _jround(1.0 / second[1], 1.0 / second[0])
    result = _jmul(first, reciprocal)
    return jnp.where(invalid, -jnp.inf, result[0]), jnp.where(invalid, jnp.inf, result[1])


def _jmat(first: _JInterval, second: _JInterval, /) -> _JInterval:
    products = _jmul(
        (first[0][:, :, None], first[1][:, :, None]), (second[0][None], second[1][None])
    )
    magnitude = jnp.sum(jnp.maximum(jnp.abs(products[0]), jnp.abs(products[1])), axis=1)
    count = first[0].shape[1]
    unit = jnp.finfo(jnp.float64).eps
    gamma = count * unit / (1.0 - count * unit)
    return _jround(
        jnp.sum(products[0], axis=1) - gamma * magnitude,
        jnp.sum(products[1], axis=1) + gamma * magnitude,
    )


def _jvector(first: _JInterval, second: _JInterval, /) -> _JInterval:
    result = _jmat(first, (second[0][:, None], second[1][:, None]))
    return result[0][:, 0], result[1][:, 0]


@final
class _QuadricSystem(StrictModule):
    """Source polynomial normal equations, including imperfect stored frames."""

    center: Array
    basis: Array
    axis: Array
    basis_lower: Array
    basis_upper: Array
    center_lower: Array
    center_upper: Array
    axis_lower: Array
    axis_upper: Array
    normalization: Array
    axis_normalization: Array
    normalized: bool = eqx.field(static=True)
    axial: bool = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.basis.shape[1] + self.normalized + self.axial

    def point(self, values: Array, /) -> Array:
        point = self.center + self.basis @ values[: self.basis.shape[1]]
        return point + values[self.basis.shape[1]] * self.axis if self.axial else point

    def equations(self, values: Array, point: Array, /) -> Array:
        count = self.basis.shape[1]
        delta = point - self.point(values)
        equation = self.basis.T @ delta / self.normalization
        if self.normalized:
            equation = equation - values[-1] * values[:count]
        terms = [equation]
        if self.axial:
            terms.append((self.axis @ delta / self.axis_normalization)[None])
        if self.normalized:
            terms.append((jnp.sum(values[:count] ** 2) - 1.0)[None])
        return jnp.concatenate(terms)

    def point_interval(self, values: _JInterval, /) -> _JInterval:
        count = self.basis.shape[1]
        result = _jadd(
            (self.center_lower, self.center_upper),
            _jvector(
                (self.basis_lower, self.basis_upper),
                (values[0][:count], values[1][:count]),
            ),
        )
        if self.axial:
            result = _jadd(
                result,
                _jmul(
                    (self.axis_lower, self.axis_upper),
                    (values[0][count], values[1][count]),
                ),
            )
        return result

    def equation_interval(self, values: _JInterval, point: Array, /) -> _JInterval:
        count = self.basis.shape[1]
        delta = _jsub((point, point), self.point_interval(values))
        result = _jdiv(
            _jvector((self.basis_lower.T, self.basis_upper.T), delta),
            (self.normalization, self.normalization),
        )
        if self.normalized:
            result = _jsub(
                result,
                _jmul(
                    (values[0][-1], values[1][-1]), (values[0][:count], values[1][:count])
                ),
            )
        lower, upper = [result[0]], [result[1]]
        if self.axial:
            axial = _jdiv(
                _jvector((self.axis_lower[None], self.axis_upper[None]), delta),
                (self.axis_normalization, self.axis_normalization),
            )
            lower.append(axial[0])
            upper.append(axial[1])
        if self.normalized:
            norm = _jmat(
                (values[0][:count][None], values[1][:count][None]),
                (values[0][:count, None], values[1][:count, None]),
            )
            lower.append(norm[0][0] - 1.0)
            upper.append(norm[1][0] - 1.0)
        return _jround(jnp.concatenate(lower), jnp.concatenate(upper))

    def jacobian_interval(self, values: _JInterval, /) -> _JInterval:
        count, dimension = self.basis.shape[1], self.dimension
        gram = _jmat(
            (self.basis_lower.T, self.basis_upper.T), (self.basis_lower, self.basis_upper)
        )
        block = _jdiv((-gram[1], -gram[0]), (self.normalization, self.normalization))
        if self.normalized:
            identity = jnp.eye(count, dtype=jnp.float64)
            block = _jsub(
                block, _jmul((values[0][-1], values[1][-1]), (identity, identity))
            )
        lower = (
            jnp.zeros((dimension, dimension), dtype=jnp.float64)
            .at[:count, :count]
            .set(block[0])
        )
        upper = (
            jnp.zeros((dimension, dimension), dtype=jnp.float64)
            .at[:count, :count]
            .set(block[1])
        )
        if self.axial:
            cross = _jvector(
                (self.basis_lower.T, self.basis_upper.T),
                (self.axis_lower, self.axis_upper),
            )
            height = _jdiv(
                (-cross[1], -cross[0]), (self.normalization, self.normalization)
            )
            radial = _jdiv(
                (-cross[1], -cross[0]), (self.axis_normalization, self.axis_normalization)
            )
            axial_norm = _jmat(
                (self.axis_lower[None], self.axis_upper[None]),
                (self.axis_lower[:, None], self.axis_upper[:, None]),
            )
            axial_diagonal = _jdiv(
                (-axial_norm[1][0, 0], -axial_norm[0][0, 0]),
                (self.axis_normalization, self.axis_normalization),
            )
            lower = (
                lower.at[:count, count]
                .set(height[0])
                .at[count, :count]
                .set(radial[0])
                .at[count, count]
                .set(axial_diagonal[0])
            )
            upper = (
                upper.at[:count, count]
                .set(height[1])
                .at[count, :count]
                .set(radial[1])
                .at[count, count]
                .set(axial_diagonal[1])
            )
        if self.normalized:
            lower = (
                lower.at[:count, -1]
                .set(-values[1][:count])
                .at[-1, :count]
                .set(2.0 * values[0][:count])
            )
            upper = (
                upper.at[:count, -1]
                .set(-values[0][:count])
                .at[-1, :count]
                .set(2.0 * values[1][:count])
            )
        return _jround(lower, upper)

    def squared_singular_bounds(self) -> _JInterval:
        basis = (self.basis_lower, self.basis_upper)
        if self.axial:
            denominator = _jmat(
                (self.axis_lower[None], self.axis_upper[None]),
                (self.axis_lower[:, None], self.axis_upper[:, None]),
            )
            product = _jmat(
                (self.axis_lower[:, None], self.axis_upper[:, None]),
                (self.axis_lower[None], self.axis_upper[None]),
            )
            projection = _jsub(
                (jnp.eye(3, dtype=jnp.float64), jnp.eye(3, dtype=jnp.float64)),
                _jdiv(product, (denominator[0][0, 0], denominator[1][0, 0])),
            )
            basis = _jmat(projection, basis)
        gram = _jmat((basis[0].T, basis[1].T), basis)
        absolute = jnp.maximum(jnp.abs(gram[0]), jnp.abs(gram[1]))
        off_diagonal = absolute.at[jnp.diag_indices(absolute.shape[0])].set(0.0)
        radius = jnp.sum(off_diagonal, axis=1)
        rounding = 8.0 * jnp.finfo(jnp.float64).eps * jnp.sum(absolute, axis=1)
        return jnp.min(jnp.diag(gram[0]) - radius - rounding), jnp.max(
            jnp.diag(gram[1]) + radius + rounding
        )


def _quadric_system(
    center: Array,
    basis: Array,
    basis_interval: _JInterval,
    normalized: bool,
    axis: Array | None = None,
    center_interval: _JInterval | None = None,
    axis_interval: _JInterval | None = None,
    /,
) -> _QuadricSystem:
    axial = axis is not None
    direction = jnp.zeros((3,), dtype=jnp.float64) if axis is None else axis
    normalization = jnp.maximum(
        jnp.max(jnp.sum(basis**2, axis=0)), jnp.finfo(jnp.float64).tiny
    )
    axis_normalization = jnp.maximum(jnp.sum(direction**2), jnp.finfo(jnp.float64).tiny)
    center_bounds = (center, center) if center_interval is None else center_interval
    axis_bounds = (direction, direction) if axis_interval is None else axis_interval
    return _QuadricSystem(
        center,
        basis,
        direction,
        basis_interval[0],
        basis_interval[1],
        center_bounds[0],
        center_bounds[1],
        axis_bounds[0],
        axis_bounds[1],
        normalization,
        axis_normalization,
        normalized,
        axial,
    )


@eqx.filter_jit
def _quadric_root(
    system: _QuadricSystem, points: Array, seeds: Array, tol: _Tolerances
) -> tuple[Array, Array, Array, Array]:
    plan = VectorLocalRootPlan(
        system.dimension,
        maximum_steps=tol.newton_steps,
        tolerance=1.0e-13,
        plan_id="native-brep-quadric-projection",
    )

    def solve(point: Array, seed: Array) -> tuple[Array, Array, Array, Array]:
        def residual(values: Array) -> Array:
            return system.equations(values, point)

        root, diagnostics = plan.solve_with_diagnostics(
            residual, jax.lax.stop_gradient(seed)
        )
        proof = jax.tree_util.tree_map(jax.lax.stop_gradient, system)
        center = jax.lax.stop_gradient(root)
        point_ = jax.lax.stop_gradient(point)
        inverse = inverse_small_linear(
            SmallLinearSolvePlan(system.dimension),
            jax.jacfwd(proof.equations)(center, point_),
        )
        preconditioner = jax.lax.stop_gradient(inverse.value)
        width = jnp.maximum(
            4096.0 * jnp.finfo(jnp.float64).eps,
            jnp.minimum(tol.parametric, tol.distance / tol.scale) / 64.0,
        ) * (1.0 + jnp.abs(center))
        lower, upper = _jround(center - width, center + width)
        value = proof.equation_interval((center, center), point_)
        derivative = proof.jacobian_interval((lower, upper))
        product = _jmat((preconditioner, preconditioner), derivative)
        identity = jnp.eye(system.dimension, dtype=jnp.float64)
        remainder = _jsub((identity, identity), product)
        offset = _jsub((lower, upper), (center, center))
        image = _jadd(
            _jsub((center, center), _jvector((preconditioner, preconditioner), value)),
            _jvector(remainder, offset),
        )
        absolute = jnp.maximum(jnp.abs(remainder[0]), jnp.abs(remainder[1]))
        norm = jnp.max(
            jnp.sum(absolute, axis=1)
            + 8.0 * jnp.finfo(jnp.float64).eps * jnp.sum(absolute, axis=1)
        )
        unique_root = (
            inverse.successful
            & diagnostics.finite
            & jnp.all(image[0] > lower)
            & jnp.all(image[1] < upper)
            & (norm < 1.0)
        )
        root_lower, root_upper = (
            jnp.maximum(lower, image[0]),
            jnp.minimum(upper, image[1]),
        )
        global_unique = unique_root
        if system.normalized:
            eigen_lower, eigen_upper = proof.squared_singular_bounds()
            reach_multiplier = eigen_lower * jnp.sqrt(
                jnp.maximum(eigen_lower / eigen_upper, 0.0)
            )
            multiplier = (
                jnp.maximum(jnp.abs(root_lower[-1]), jnp.abs(root_upper[-1]))
                * proof.normalization
            )
            global_unique &= (eigen_lower > 0.0) & (
                (root_lower[-1] > 0.0) | (multiplier < reach_multiplier)
            )
        return root, root_lower, root_upper, global_unique

    return jax.vmap(solve)(points, seeds)


def _trim_box_clear(domain: TrimDomain, parameters: Array, errors: Array, /) -> Array:
    clear = domain.contains(parameters)
    for loop in domain.loops:
        if not isinstance(loop, CurveTrimLoop):
            raise ValueError("Native trim certificates require source curve loops.")
        lower = jnp.concatenate(
            (jnp.asarray(loop.arc_lower), jnp.asarray(loop.junction_lower))
        )
        upper = jnp.concatenate(
            (jnp.asarray(loop.arc_upper), jnp.asarray(loop.junction_upper))
        )
        clear &= ~jnp.any(
            jnp.all(
                (parameters[:, None] + errors[:, None] >= lower)
                & (parameters[:, None] - errors[:, None] <= upper),
                axis=-1,
            ),
            axis=-1,
        )
    return clear


def _place_quadric_system(
    system: _QuadricSystem,
    placements: tuple[PlacedSurface, ...],
    /,
) -> _QuadricSystem:
    for placement in reversed(placements):
        rotation, shift = placement.rotation, placement.translation
        center = rotation @ system.center + shift
        basis = rotation @ system.basis
        center_interval = _jadd(
            _jvector((rotation, rotation), (system.center_lower, system.center_upper)),
            (shift, shift),
        )
        basis_interval = _jmat(
            (rotation, rotation), (system.basis_lower, system.basis_upper)
        )
        axis = rotation @ system.axis if system.axial else None
        axis_interval = _jvector(
            (rotation, rotation), (system.axis_lower, system.axis_upper)
        )
        system = _quadric_system(
            center,
            basis,
            basis_interval,
            system.normalized,
            axis,
            center_interval,
            axis_interval,
        )
    return system


def _surface_quadric(
    face: _FaceData, points: Array, parameters: Array, tol: _Tolerances, /
) -> dict[str, Array] | None:
    patch = face.patch
    placements = []
    while isinstance(patch, PlacedSurface):
        placements.append(patch)
        patch = patch.definition
    if isinstance(patch, PlanePatch):
        basis = jnp.stack((patch.first_axis, patch.second_axis), axis=1)
        system = _quadric_system(patch.origin, basis, (basis, basis), False)
        system = _place_quadric_system(system, tuple(placements))
        seeds = parameters
    elif isinstance(patch, (SpherePatch, CylinderPatch)):
        if isinstance(patch, SpherePatch):
            axes = jnp.stack((patch.first_axis, patch.second_axis, patch.axis), axis=1)
            center = patch.center
            angular = jnp.stack(
                (
                    jnp.cos(parameters[:, 1]) * jnp.cos(parameters[:, 0]),
                    jnp.cos(parameters[:, 1]) * jnp.sin(parameters[:, 0]),
                    jnp.sin(parameters[:, 1]),
                ),
                axis=1,
            )
            axis = None
        else:
            axes = jnp.stack((patch.first_axis, patch.second_axis), axis=1)
            center = patch.origin
            angular = jnp.stack(
                (jnp.cos(parameters[:, 0]), jnp.sin(parameters[:, 0])), axis=1
            )
            axis = patch.axis
        basis = patch.radius * axes
        system = _quadric_system(
            center, basis, _jmul((patch.radius, patch.radius), (axes, axes)), True, axis
        )
        system = _place_quadric_system(system, tuple(placements))
        partial = (
            jnp.concatenate((angular, parameters[:, 1:2]), axis=1)
            if system.axial
            else angular
        )

        def initial(value: Array, point: Array) -> Array:
            multiplier = (
                jnp.sum(
                    (system.basis.T @ (point - system.point(value)))
                    * value[: system.basis.shape[1]]
                )
                / system.normalization
            )
            return jnp.concatenate((value, multiplier[None]))

        seeds = jax.vmap(initial)(partial, points)
    else:
        return None
    roots, lower, upper, unique = _quadric_root(system, points, seeds, tol)
    projected = jax.vmap(system.point)(roots)

    def enclosure(low: Array, high: Array) -> tuple[Array, Array]:
        return system.point_interval((low, high))

    point_lower, point_upper = jax.vmap(enclosure)(lower, upper)
    point_error = jnp.linalg.norm(
        jnp.maximum(jnp.abs(point_lower - projected), jnp.abs(point_upper - projected)),
        axis=-1,
    )
    if isinstance(patch, PlanePatch):
        uv = roots[:, :2]
        error = jnp.maximum(jnp.abs(lower[:, :2] - uv), jnp.abs(upper[:, :2] - uv))
        pole = jnp.zeros((points.shape[0],), dtype=jnp.bool_)
    else:
        count = system.basis.shape[1]
        angular = roots[:, :count]
        angular_error = jnp.linalg.norm(
            jnp.maximum(
                jnp.abs(lower[:, :count] - angular), jnp.abs(upper[:, :count] - angular)
            ),
            axis=-1,
        )
        radial = jnp.linalg.norm(angular[:, :2], axis=-1)
        u = face.lower[0] + jnp.mod(
            jnp.arctan2(angular[:, 1], angular[:, 0]) - face.lower[0], 2.0 * pi
        )
        u_error = jnp.minimum(
            2.0 * pi,
            2.0
            * angular_error
            / jnp.maximum(radial - angular_error, jnp.finfo(jnp.float64).tiny),
        )
        if isinstance(patch, SpherePatch):
            v = jnp.arctan2(angular[:, 2], radial)
            v_error = 2.0 * angular_error
            pole = radial <= 16.0 * angular_error
        else:
            v = roots[:, count]
            v_error = jnp.maximum(
                jnp.abs(lower[:, count] - v), jnp.abs(upper[:, count] - v)
            )
            pole = jnp.zeros((points.shape[0],), dtype=jnp.bool_)
        uv, error = jnp.stack((u, v), axis=1), jnp.stack((u_error, v_error), axis=1)
    rounding = 128.0 * jnp.finfo(jnp.float64).eps * (1.0 + jnp.abs(uv))
    error += rounding
    admissible = _trim_box_clear(face.trim_domain, uv, error)
    box_valid = jnp.all(
        (uv - error >= jnp.asarray(face.lower)) & (uv + error <= jnp.asarray(face.upper)),
        axis=-1,
    )
    if face.periods[0] > 0.0:
        box_valid = (uv[:, 1] - error[:, 1] >= face.lower[1]) & (
            uv[:, 1] + error[:, 1] <= face.upper[1]
        )
    if face.full_rectangle:
        admissible = box_valid
        if (
            isinstance(patch, SpherePatch)
            and face.lower[1] == -0.5 * pi
            and face.upper[1] == 0.5 * pi
            and face.periods[0] > 0.0
        ):
            admissible = jnp.ones(points.shape[0], dtype=jnp.bool_)
            box_valid = admissible
    unique &= admissible & box_valid & (point_error <= tol.distance)
    if isinstance(patch, PlanePatch):
        _, _, normal = _frame(face, uv)
    else:
        matrix = system.basis.T
        right = roots[:, : system.basis.shape[1]]
        if system.axial:
            matrix = jnp.concatenate((matrix, system.axis[None]), axis=0)
            right = jnp.concatenate(
                (right, jnp.zeros((points.shape[0], 1), dtype=jnp.float64)), axis=1
            )
        solved = solve_small_linear(
            SmallLinearSolvePlan(3),
            jnp.broadcast_to(matrix, (points.shape[0], 3, 3)),
            right[..., None],
        )
        normal = solved.value[..., 0]
        normal = (
            face.orientation
            * jnp.sign(solved.determinant)[:, None]
            * normal
            / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        )
        unique &= solved.successful
    return {
        "point": projected,
        "parameter": uv,
        "parameter_bound": jnp.max(error, axis=-1),
        "point_lower": point_lower,
        "point_upper": point_upper,
        "point_error": point_error,
        "unique": unique,
        "pole": pole,
        "normal": normal,
    }


def _curve_value(curve: BRepCurve, parameter: Array, /) -> Array:
    if isinstance(curve, IntersectionCurve):
        return curve.evaluate(parameter).point
    return curve.evaluate(parameter)


def _box_distance(points: Array, boxes: Array, /) -> Array:
    delta = jnp.maximum(
        jnp.maximum(
            boxes[None, :, 0] - points[:, None], points[:, None] - boxes[None, :, 1]
        ),
        0.0,
    )
    return jnp.linalg.norm(delta, axis=-1)


def _discovery_seeds(
    bvh: PackedBVH, seed_points: Array, points: Array, count: int, scale: Array, /
) -> tuple[Array, Array]:
    """BVH discovery with a numerical tie-breaker that cannot become completeness evidence."""
    epsilon = 16.0 * jnp.finfo(jnp.float64).eps

    def priority(point: Array, items: Array) -> Array:
        lower, upper = bvh.item_bbox_min[items], bvh.item_bbox_max[items]
        delta = jnp.maximum(jnp.maximum(lower - point, point - upper), 0.0)
        box_distance = jnp.sum(delta * delta, axis=-1)
        sample_distance = jnp.sum((seed_points[items] - point) ** 2, axis=-1)
        return box_distance + epsilon * sample_distance

    discovered = bvh_nearest_items(
        bvh, points, k=min(count, seed_points.shape[0]), item_distance_squared=priority
    )
    maximum_sample_norm = jnp.sqrt(jnp.max(jnp.sum(seed_points**2, axis=-1)))
    point_norm = jnp.linalg.norm(points, axis=-1)
    maximum_bonus = epsilon * (point_norm + maximum_sample_norm) ** 2
    rounding = (
        256.0
        * jnp.finfo(jnp.float64).eps
        * (scale**2 + point_norm**2 + maximum_sample_norm**2)
    )
    lower_squared = discovered.distance_squared[:, 0] - maximum_bonus - rounding
    lower = jnp.sqrt(
        jnp.maximum(jnp.where(jnp.isfinite(lower_squared), lower_squared, 0.0), 0.0)
    )
    return discovered.items, lower


def _edge_carrier_lower(edge: _EdgeData, points: Array, scale: Array, /) -> Array:
    curve = edge.curve
    lower = jnp.zeros((points.shape[0],), dtype=jnp.float64)
    if isinstance(curve, LineCurve):
        relative = points - curve.origin
        parameter = jnp.clip(
            jnp.sum(relative * curve.direction, axis=-1) / jnp.sum(curve.direction**2),
            edge.first,
            edge.last,
        )
        lower = jnp.linalg.norm(points - curve.evaluate(parameter), axis=-1)
    elif isinstance(curve, CircleCurve):
        relative = points - curve.center
        normal = jnp.cross(curve.first_axis, curve.second_axis)
        normal = normal / jnp.linalg.norm(normal)
        height = relative @ normal
        radial = jnp.linalg.norm(relative - height[:, None] * normal, axis=-1)
        frame = jnp.stack((curve.first_axis, curve.second_axis), axis=1)
        defect = jnp.linalg.norm(frame.T @ frame - jnp.eye(2, dtype=jnp.float64))
        radius_error = curve.radius * defect
        gap = jnp.maximum(jnp.abs(radial - curve.radius) - radius_error, 0.0)
        lower = jnp.sqrt(gap**2 + height**2)
    margin = (
        256.0
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(scale, jnp.linalg.norm(points, axis=-1))
    )
    return jnp.maximum(lower - margin, 0.0)


def _edge_query(edge: _EdgeData, points: Array, tol: _Tolerances, /) -> dict[str, Array]:
    """Closest point on one edge: seeded Newton roots plus both end points."""
    count = points.shape[0]
    if edge.curve is None:
        # A degenerate edge is geometrically its single vertex.
        point = jnp.broadcast_to(edge.start_point, (count, 3))
        return {
            "point": point,
            "parameter": jnp.full((count,), edge.first),
            "distance": jnp.linalg.norm(points - point, axis=-1),
            "status": jnp.full((count,), _UNIQUE, dtype=jnp.int8),
            "tangent": jnp.full((count, 3), jnp.nan),
            "proposal_status": jnp.full((count,), _UNIQUE, dtype=jnp.int8),
            "lower_bound": jnp.linalg.norm(points - point, axis=-1),
            "unresolved": jnp.zeros((count,), dtype=jnp.bool_),
            "parameter_bound": jnp.zeros((count,), dtype=jnp.float64),
        }
    curve = edge.curve
    first, last = edge.first, edge.last
    seed_indices, lower_bound = _discovery_seeds(
        edge.discovery_bvh, edge.seed_points, points, tol.seeds, tol.scale
    )
    seeds = edge.seed_parameters[seed_indices]
    plan = VectorLocalRootPlan(
        1,
        maximum_steps=tol.newton_steps,
        tolerance=1.0e-13,
        plan_id="native-brep-edge-projection",
    )

    def evaluate(parameter: Array) -> Array:
        return _curve_value(curve, jnp.clip(parameter, first, last))

    def refine(point: Array, seed: Array) -> tuple[Array, Array]:
        def residual(value: Array) -> Array:
            parameter = value[0]
            difference = evaluate(parameter) - point
            return (difference @ jax.jacfwd(evaluate)(parameter))[None] / tol.scale**2

        root, diagnostics = plan.solve_with_diagnostics(residual, seed[None])
        # A singular refinement (a continuum of minima) still yields a root.
        return root[0], diagnostics.finite & (diagnostics.residual_norm <= 1.0e-12)

    roots, converged = jax.vmap(jax.vmap(refine, (None, 0)), (0, 0))(points, seeds)
    slack = tol.parametric * (last - first)
    valid = converged & (roots >= first - slack) & (roots <= last + slack)
    candidates = jnp.concatenate(
        (jnp.clip(roots, first, last), jnp.full((count, 2), jnp.asarray((first, last)))),
        axis=1,
    )
    valid = jnp.concatenate((valid, jnp.ones((count, 2), dtype=jnp.bool_)), axis=1)
    candidate_points = jax.vmap(jax.vmap(evaluate))(candidates)
    distance = jnp.where(
        valid, jnp.linalg.norm(candidate_points - points[:, None, :], axis=-1), jnp.inf
    )
    best = jnp.argmin(distance, axis=1)
    rows = jnp.arange(count)
    best_point = candidate_points[rows, best]
    best_distance = distance[rows, best]
    same = valid & (
        jnp.linalg.norm(candidate_points - best_point[:, None, :], axis=-1)
        <= tol.distance
    )
    tie = jnp.any(valid & ~same & (distance <= best_distance[:, None] + tol.distance), 1)
    parameter = jnp.min(jnp.where(same, candidates, jnp.inf), axis=1)
    point = jax.vmap(evaluate)(parameter)
    velocity = jax.vmap(jax.jacfwd(evaluate))(parameter)
    acceleration = jax.vmap(jax.jacfwd(jax.jacfwd(evaluate)))(parameter)
    difference = point - points
    speed_squared = jnp.sum(velocity**2, -1)
    # A stationary point whose second variation vanishes is not isolated.
    stationary = jnp.abs(jnp.sum(difference * velocity, -1)) <= 1.0e-8 * jnp.sqrt(
        speed_squared
    ) * jnp.maximum(best_distance, tol.distance)
    second_variation = speed_squared + jnp.sum(difference * acceleration, -1)
    continuum = stationary & (jnp.abs(second_variation) <= 1.0e-8 * speed_squared)
    at_end = (parameter - first <= slack) | (last - parameter <= slack)
    lower_bound = jnp.maximum(lower_bound, _edge_carrier_lower(edge, points, tol.scale))
    parameter_bound = jnp.zeros((count,), dtype=jnp.float64)
    coupled_resolved = jnp.ones((count,), dtype=jnp.bool_)
    if isinstance(curve, IntersectionCurve):
        coupled = curve.evaluate(parameter)
        parameter_bound = coupled.parameter_bound
        coupled_resolved = (
            jnp.asarray(curve.fully_certified)
            & jnp.isfinite(parameter_bound)
            & (parameter_bound <= tol.parametric)
            & (coupled.gap <= tol.distance)
        )
    seam = jnp.asarray(edge.closed) & at_end
    status = jnp.where(
        tie | continuum,
        _AMBIGUOUS,
        jnp.where(seam, _SEAM, _UNIQUE),
    ).astype(jnp.int8)
    unresolved = (best_distance - lower_bound > tol.distance) | ~coupled_resolved
    status = jnp.where(unresolved, _FAILED, status).astype(jnp.int8)
    return {
        "point": point,
        "parameter": parameter,
        "distance": best_distance,
        "status": status,
        "tangent": velocity,
        "lower_bound": lower_bound,
        "unresolved": unresolved,
        "parameter_bound": parameter_bound,
        "proposal_status": jnp.where(
            tie | continuum, _AMBIGUOUS, jnp.where(seam, _SEAM, _UNIQUE)
        ).astype(jnp.int8),
    }


def _frame(face: _FaceData, uv: Array, /) -> tuple[Array, Array, Array]:
    """``(S_u, S_v, oriented unit normal)`` with the chart-limit normal at poles."""
    patch = face.patch
    first = jax.vmap(jax.jacfwd(patch.evaluate))(uv)
    second = jax.vmap(jax.jacfwd(jax.jacfwd(patch.evaluate)))(uv)
    du, dv = first[:, :, 0], first[:, :, 1]
    mixed = second[:, :, 0, 1]
    normal = jnp.cross(du, dv)
    norm_u = jnp.linalg.norm(du, axis=-1)
    norm_v = jnp.linalg.norm(dv, axis=-1)
    regular = jnp.linalg.norm(normal, axis=-1) > 1.0e-9 * jnp.maximum(norm_u, norm_v) ** 2
    middle_u = 0.5 * (face.lower[0] + face.upper[0])
    middle_v = 0.5 * (face.lower[1] + face.upper[1])
    # Near a collapsed isoline S_u ~ (v - v_pole) S_uv, so the interior side of
    # the chart fixes the sign of the limit normal (and symmetrically for S_v).
    collapsed_u = jnp.sign(middle_v - uv[:, 1])[:, None] * jnp.cross(mixed, dv)
    collapsed_v = jnp.sign(middle_u - uv[:, 0])[:, None] * jnp.cross(du, mixed)
    limit = jnp.where((norm_u <= norm_v)[:, None], collapsed_u, collapsed_v)
    chosen = jnp.where(regular[:, None], normal, limit)
    length = jnp.linalg.norm(chosen, axis=-1, keepdims=True)
    unit = jnp.where(length > 0.0, chosen / jnp.where(length > 0.0, length, 1.0), jnp.nan)
    return du, dv, face.orientation * unit


def _carrier_lower(face: _FaceData, points: Array, scale: Array, /) -> Array:
    radial = jnp.linalg.norm(
        (points - face.carrier_center) @ face.carrier_projection.T, axis=-1
    )
    gap = jnp.maximum(
        jnp.maximum(face.carrier_radii[0] - radial, radial - face.carrier_radii[1]), 0.0
    )
    margin = (
        256.0
        * jnp.finfo(jnp.float64).eps
        * jnp.maximum(scale, jnp.linalg.norm(points, axis=-1))
    )
    return jnp.maximum(gap - margin, 0.0)


def _torus_axis_distance_bounds(
    face: _FaceData, points: Array, tol: _Tolerances, /
) -> tuple[Array, Array]:
    """Bound the full toroidal angular family from its authored frame."""
    patch = face.patch
    if (
        not isinstance(patch, TorusPatch)
        or not face.full_rectangle
        or any(period == 0.0 for period in face.periods)
    ):
        return jnp.zeros((points.shape[0],), dtype=jnp.bool_), jnp.zeros(
            (points.shape[0],)
        )
    frame = jnp.stack((patch.first_axis, patch.second_axis, patch.axis), axis=1)
    defect = jnp.linalg.norm(frame.T @ frame - jnp.eye(3), ord="fro")
    axis = patch.axis / jnp.linalg.norm(patch.axis)
    relative = points - patch.center
    height = relative @ axis
    radial = jnp.linalg.norm(relative - height[:, None] * axis, axis=-1)
    # The polar carrier is only a proof enclosure: source coefficients stay
    # authored. Include both source displacement and carrier-axis displacement.
    error = (
        radial
        + (
            jnp.abs(patch.major_radius)
            + jnp.abs(patch.minor_radius)
            + 2.0 * jnp.linalg.norm(relative, axis=-1)
        )
        * defect
        + 1024.0 * jnp.finfo(jnp.float64).eps * tol.scale
    )
    distance = jnp.abs(
        jnp.hypot(patch.major_radius, height) - jnp.abs(patch.minor_radius)
    )
    uniform = (
        (2.0 * error <= tol.distance)
        & (patch.major_radius > patch.minor_radius)
        & (defect < 0.5)
    )
    return uniform, jnp.maximum(distance - error, 0.0)


def _uniform_quadric_ambiguity(
    face: _FaceData, points: Array, tol: _Tolerances, /
) -> Array:
    """Bound the complete angular family, rather than inferring a continuum from probes."""
    if not face.full_rectangle or face.periods[0] == 0.0:
        return jnp.zeros((points.shape[0],), dtype=jnp.bool_)
    patch = face.patch
    if (
        isinstance(patch, SpherePatch)
        and face.lower[1] == -0.5 * pi
        and face.upper[1] == 0.5 * pi
    ):
        distance = jnp.linalg.norm(points - patch.center, axis=-1)
        spread = face.carrier_radii[1] - face.carrier_radii[0] + 2.0 * distance
        return (
            spread + 1024.0 * jnp.finfo(jnp.float64).eps * tol.scale <= tol.distance
        ) & (face.carrier_radii[0] > 4.0 * tol.distance)
    if isinstance(patch, CylinderPatch):
        axis_squared = jnp.sum(patch.axis**2)
        relative = points - patch.origin
        height = (relative @ patch.axis) / axis_squared
        radial = jnp.linalg.norm(relative - height[:, None] * patch.axis, axis=-1)
        axial_radius = (
            jnp.abs(patch.radius)
            * (
                jnp.abs(patch.first_axis @ patch.axis)
                + jnp.abs(patch.second_axis @ patch.axis)
            )
            / jnp.sqrt(axis_squared)
        )
        upper = jnp.sqrt((face.carrier_radii[1] + radial) ** 2 + axial_radius**2)
        lower = jnp.maximum(face.carrier_radii[0] - radial, 0.0)
        return (
            (height > face.lower[1] + tol.parametric)
            & (height < face.upper[1] - tol.parametric)
            & (
                upper - lower + 1024.0 * jnp.finfo(jnp.float64).eps * tol.scale
                <= tol.distance
            )
            & (face.carrier_radii[0] > 4.0 * tol.distance)
        )
    return jnp.zeros((points.shape[0],), dtype=jnp.bool_)


def _face_query(
    face: _FaceData,
    edge_hits: tuple[dict[str, Array], ...],
    points: Array,
    tol: _Tolerances,
    /,
) -> dict[str, Array]:
    """Closest point of every query on one trimmed face."""
    count = points.shape[0]
    patch = face.patch
    periodic = (face.periods[0] > 0.0, face.periods[1] > 0.0)
    seed_indices, lower_bound = _discovery_seeds(
        face.discovery_bvh, face.seed_points, points, tol.seeds, tol.scale
    )
    seeds = face.seed_uv[seed_indices]
    plan = VectorLocalRootPlan(
        2,
        maximum_steps=tol.newton_steps,
        tolerance=1.0e-13,
        plan_id="native-brep-face-projection",
    )

    def evaluate(uv: Array) -> Array:
        return patch.evaluate(_clamp_open(uv, face))

    def stationarity(uv: Array, point: Array) -> Array:
        difference = evaluate(uv) - point
        return jax.jacfwd(evaluate)(uv).T @ difference / tol.scale**2

    def refine(point: Array, seed: Array) -> tuple[Array, Array]:
        root, diagnostics = plan.solve_with_diagnostics(
            lambda value: stationarity(value, point), seed
        )
        # A singular refinement (a continuum of minima) still yields a root.
        return root, diagnostics.finite & (diagnostics.residual_norm <= 1.0e-12)

    roots, converged = jax.vmap(jax.vmap(refine, (None, 0)), (0, 0))(points, seeds)
    wrapped = _wrap(roots, face)
    lower, upper = jnp.asarray(face.lower), jnp.asarray(face.upper)
    slack = tol.parametric * jnp.maximum(upper - lower, 1.0)
    in_box = jnp.all(
        jnp.asarray(periodic) | ((wrapped >= lower - slack) & (wrapped <= upper + slack)),
        axis=-1,
    )
    interior_uv = jnp.clip(wrapped, lower, upper)
    trim_unresolved = face.trim_domain.in_band(interior_uv)
    inside = face.trim_domain.contains(interior_uv) & ~trim_unresolved
    interior_valid = converged & in_box & inside
    interior_points = jax.vmap(jax.vmap(patch.evaluate))(interior_uv)
    boundary_points = jnp.stack([hit["point"] for hit in edge_hits], axis=1)
    boundary_uv = jnp.stack(
        [
            pcurve.evaluate(hit["parameter"])
            for pcurve, hit in zip(face.pcurves, edge_hits, strict=True)
        ],
        axis=1,
    )
    boundary_status = jnp.stack([hit["proposal_status"] for hit in edge_hits], axis=1)
    candidate_points = jnp.concatenate((interior_points, boundary_points), axis=1)
    candidate_uv = jnp.concatenate((interior_uv, boundary_uv), axis=1)
    valid = jnp.concatenate(
        (interior_valid, jnp.ones(boundary_status.shape, dtype=jnp.bool_)), axis=1
    )
    distance = jnp.where(
        valid, jnp.linalg.norm(candidate_points - points[:, None, :], axis=-1), jnp.inf
    )
    rows = jnp.arange(count)
    best = jnp.argmin(distance, axis=1)
    best_point = candidate_points[rows, best]
    best_distance = distance[rows, best]
    boundary_parameters = jnp.stack([hit["parameter"] for hit in edge_hits], axis=1)
    candidate_edge_parameters = jnp.concatenate(
        (jnp.full(interior_valid.shape, jnp.nan, dtype=jnp.float64), boundary_parameters),
        axis=1,
    )
    candidate_edge_indices = jnp.concatenate(
        (
            jnp.full(interior_valid.shape, -1, dtype=jnp.int32),
            jnp.broadcast_to(
                jnp.asarray(face.coedge_edges, dtype=jnp.int32), boundary_status.shape
            ),
        ),
        axis=1,
    )
    same = valid & (
        jnp.linalg.norm(candidate_points - best_point[:, None, :], axis=-1)
        <= tol.distance
    )
    tie = jnp.any(valid & ~same & (distance <= best_distance[:, None] + tol.distance), 1)
    # Lexicographically lowest parameter representative of the closest point.
    u_low = jnp.min(jnp.where(same, candidate_uv[..., 0], jnp.inf), axis=1)
    same_u = same & (candidate_uv[..., 0] == u_low[:, None])
    v_low = jnp.min(jnp.where(same_u, candidate_uv[..., 1], jnp.inf), axis=1)
    uv = jnp.stack((u_low, v_low), axis=-1)
    uv = jnp.where(jnp.isfinite(uv), uv, jnp.asarray(face.lower))
    pole_distance = (
        jnp.min(
            jnp.linalg.norm(best_point[:, None, :] - face.pole_points[None], axis=-1),
            axis=1,
        )
        if face.pole_points.shape[0]
        else jnp.full((count,), jnp.inf)
    )
    at_pole = pole_distance <= tol.distance
    # Shape-operator test at the chosen point: at a stationary point
    # det(G^-1 H) = prod(1 - d kappa_i), which vanishes exactly where the
    # minimum is not isolated (a focal point such as a sphere center or a
    # cylinder axis). Boundary minima with a nonzero gradient are isolated in
    # the face interior directions and keep their edge status.
    jacobian = jax.vmap(jax.jacfwd(evaluate))(uv)
    difference = jax.vmap(evaluate)(uv) - points
    gradient = jnp.linalg.norm(
        (jnp.swapaxes(jacobian, -1, -2) @ difference[..., None])[..., 0], axis=-1
    )
    stationary_residual = gradient / (
        jnp.linalg.norm(jacobian, axis=(-2, -1))
        * jnp.maximum(jnp.linalg.norm(difference, axis=-1), tol.distance)
    )
    hessian = jax.vmap(jax.jacfwd(stationarity))(uv, points) * tol.scale**2
    metric = jnp.swapaxes(jacobian, -1, -2) @ jacobian
    plan_2 = SmallLinearSolvePlan(2)
    metric_solve = solve_small_linear(plan_2, metric, hessian)
    shape_operator = solve_small_linear(
        plan_2,
        metric_solve.value,
        jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), hessian.shape),
    )
    continuum = (
        (stationary_residual <= 1.0e-8)
        & metric_solve.successful
        & (jnp.abs(shape_operator.determinant) <= 1.0e-8)
        & ~at_pole
    )
    edge_ambiguous = jnp.any(
        same[:, roots.shape[1] :] & (boundary_status == _AMBIGUOUS), axis=1
    )
    seam_edge = jnp.any(same[:, roots.shape[1] :] & jnp.asarray(face.seam_coedges), 1)
    periods = jnp.asarray(face.periods)
    near_bound = (
        (uv - lower <= tol.parametric * periods)
        | (upper - uv <= tol.parametric * periods)
    ) & (periods > 0.0)
    seam = seam_edge | jnp.any(near_bound, axis=1) | at_pole
    edge_lower = jnp.stack([hit["lower_bound"] for hit in edge_hits], axis=1)
    lower_bound = jnp.minimum(lower_bound, jnp.min(edge_lower, axis=1))
    lower_bound = jnp.maximum(lower_bound, _carrier_lower(face, points, tol.scale))
    coupled_bound = jnp.stack([hit["parameter_bound"] for hit in edge_hits], axis=1)
    parameter_bound = jnp.max(
        jnp.where(same[:, roots.shape[1] :], coupled_bound, 0.0), axis=1
    )
    unresolved = (
        (best_distance - lower_bound > tol.distance)
        | (parameter_bound > tol.parametric)
        | ~jnp.isfinite(parameter_bound)
    )
    status = jnp.where(
        ~jnp.isfinite(best_distance),
        _FAILED,
        jnp.where(
            tie | continuum | edge_ambiguous, _AMBIGUOUS, jnp.where(seam, _SEAM, _UNIQUE)
        ),
    ).astype(jnp.int8)
    multiplicity = jnp.zeros((count,), dtype=jnp.bool_)
    point_error = jnp.full((count,), jnp.inf, dtype=jnp.float64)
    quadric = _surface_quadric(face, points, uv, tol)
    if quadric is not None:
        proved = quadric["unique"]
        best_point = jnp.where(proved[:, None], quadric["point"], best_point)
        uv = jnp.where(proved[:, None], quadric["parameter"], uv)
        candidate_distance = (
            jnp.linalg.norm(quadric["point"] - points, axis=-1) + quadric["point_error"]
        )
        delta = jnp.maximum(
            jnp.maximum(quadric["point_lower"] - points, points - quadric["point_upper"]),
            0.0,
        )
        candidate_lower = jnp.maximum(
            jnp.linalg.norm(delta, axis=-1)
            - 64.0 * jnp.finfo(jnp.float64).eps * tol.scale,
            0.0,
        )
        best_distance = jnp.where(proved, candidate_distance, best_distance)
        lower_bound = jnp.where(proved, candidate_lower, lower_bound)
        parameter_bound = jnp.where(proved, quadric["parameter_bound"], parameter_bound)
        point_error = jnp.where(proved, quadric["point_error"], point_error)
        multiplicity = proved
        status = jnp.where(
            proved,
            jnp.where(quadric["pole"] | jnp.any(near_bound, axis=1), _SEAM, _UNIQUE),
            status,
        ).astype(jnp.int8)
        unresolved = jnp.where(
            proved, best_distance - lower_bound > tol.distance, unresolved
        )
    uniform = _uniform_quadric_ambiguity(face, points, tol)
    torus_uniform, torus_lower = _torus_axis_distance_bounds(face, points, tol)
    uniform |= torus_uniform
    lower_bound = jnp.where(
        torus_uniform, jnp.maximum(lower_bound, torus_lower), lower_bound
    )
    multiplicity |= uniform
    point_error = jnp.where(
        uniform, 1024.0 * jnp.finfo(jnp.float64).eps * tol.scale, point_error
    )
    status = jnp.where(uniform, _AMBIGUOUS, status).astype(jnp.int8)
    unresolved = jnp.where(
        uniform, best_distance - lower_bound > tol.distance, unresolved
    )
    unresolved |= ~multiplicity
    du, dv, normal = _frame(face, uv)
    if quadric is not None:
        normal = jnp.where(quadric["unique"][:, None], quadric["normal"], normal)
    return {
        "point": best_point,
        "parameter": uv,
        "distance": best_distance,
        "status": status,
        "normal": normal,
        "du": du,
        "dv": dv,
        "stationarity": stationary_residual,
        "lower_bound": lower_bound,
        "unresolved": unresolved,
        "parameter_bound": parameter_bound,
        "edge_index": candidate_edge_indices[rows, best],
        "edge_parameter": candidate_edge_parameters[rows, best],
        "multiplicity": multiplicity,
        "point_error": point_error,
    }


@eqx.filter_jit
def _project_edge(edge: _EdgeData, points: Array, tol: _Tolerances) -> dict[str, Array]:
    return _edge_query(edge, points, tol)


@eqx.filter_jit
def _polish_edge(edge: _EdgeData, point: Array, seed: Array, tol: _Tolerances) -> Array:
    curve = edge.curve
    if curve is None:
        return jnp.asarray(edge.first, dtype=jnp.float64)

    def evaluate(parameter: Array) -> Array:
        return _curve_value(curve, jnp.clip(parameter, edge.first, edge.last))

    def residual(value: Array) -> Array:
        parameter = value[0]
        return ((evaluate(parameter) - point) @ jax.jacfwd(evaluate)(parameter))[
            None
        ] / tol.scale**2

    plan = VectorLocalRootPlan(
        1,
        maximum_steps=tol.newton_steps,
        tolerance=1.0e-13,
        plan_id="native-brep-interval-edge-projection",
    )
    root, diagnostics = plan.solve_with_diagnostics(residual, seed[None])
    valid = diagnostics.finite & (diagnostics.residual_norm <= 1.0e-12)
    return jnp.where(valid, jnp.clip(root[0], edge.first, edge.last), seed)


def _certified_edge(
    edge: _EdgeData,
    points: Array,
    tol: _Tolerances,
    policy: BRepQueryPolicy,
    /,
    *,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
) -> dict[str, Array]:
    hit = _project_edge(edge, points, tol)
    curve = edge.curve
    if curve is None:
        return hit
    host_points = np.asarray(points)
    parameters = np.array(hit["parameter"], dtype=np.float64)
    distances = np.array(hit["distance"], dtype=np.float64)
    lower_bounds = np.array(hit["lower_bound"], dtype=np.float64)
    parameter_bounds = np.array(hit["parameter_bound"], dtype=np.float64)
    for row, point in enumerate(host_points):
        analytic = float(
            _edge_carrier_lower(edge, jnp.asarray(point)[None], tol.scale)[0]
        )
        queue: list[tuple[float, int, float, float, int]] = []
        box = (
            curve.bounding_box(edge.first, edge.last, endpoint_roots=endpoint_roots)
            if isinstance(curve, IntersectionCurve)
            else curve.bounding_box(edge.first, edge.last)
        )
        lower = max(
            analytic,
            float(_box_distance(jnp.asarray(point)[None], jnp.asarray(box)[None])[0, 0]),
        )
        heapq.heappush(queue, (lower, 0, edge.first, edge.last, 0))
        closed_lower, serial, work = np.inf, 1, 0
        while queue and work < policy.maximum_subdivisions:
            lower, _, first, last, depth = heapq.heappop(queue)
            if lower >= distances[row] - tol.distance:
                closed_lower = min(closed_lower, lower)
                continue
            middle = 0.5 * (first + last)
            parameter = _polish_edge(
                edge, jnp.asarray(point), jnp.asarray(middle, dtype=jnp.float64), tol
            )
            projected = _curve_value(curve, parameter)
            distance = float(jnp.linalg.norm(projected - jnp.asarray(point)))
            if distance < distances[row]:
                distances[row], parameters[row] = distance, float(parameter)
            if depth >= policy.maximum_depth or not first < middle < last:
                closed_lower = min(closed_lower, lower)
                continue
            for start, stop in ((first, middle), (middle, last)):
                child_box = (
                    curve.bounding_box(start, stop, endpoint_roots=endpoint_roots)
                    if isinstance(curve, IntersectionCurve)
                    else curve.bounding_box(start, stop)
                )
                child_lower = max(
                    analytic,
                    float(
                        _box_distance(
                            jnp.asarray(point)[None], jnp.asarray(child_box)[None]
                        )[0, 0]
                    ),
                )
                heapq.heappush(queue, (child_lower, serial, start, stop, depth + 1))
                serial += 1
            work += 1
        lower_bounds[row] = min(
            distances[row], closed_lower, queue[0][0] if queue else np.inf
        )
        if isinstance(curve, IntersectionCurve):
            parameter_bounds[row] = float(
                curve.evaluate(
                    jnp.asarray(parameters[row], dtype=jnp.float64)
                ).parameter_bound
            )
    values = jnp.asarray(parameters, dtype=jnp.float64)
    projected = _curve_value(curve, values)
    tangent = jax.vmap(jax.jacfwd(lambda parameter: _curve_value(curve, parameter)))(
        values
    )
    unresolved = (
        (distances - lower_bounds > tol.distance)
        | (parameter_bounds > tol.parametric)
        | ~np.isfinite(parameter_bounds)
    )
    return {
        **hit,
        "point": projected,
        "parameter": values,
        "distance": jnp.asarray(distances),
        "tangent": tangent,
        "lower_bound": jnp.asarray(lower_bounds),
        "parameter_bound": jnp.asarray(parameter_bounds),
        "unresolved": jnp.asarray(unresolved),
        "status": jnp.where(
            jnp.asarray(unresolved), _FAILED, hit["proposal_status"]
        ).astype(jnp.int8),
    }


_Result = TypeVar("_Result")


def _chunked(
    function: Callable[[Array], _Result], points: Array, chunk: int, /
) -> _Result:
    """Bound the working set without evaluating fictitious padded query points."""
    count = points.shape[0]
    blocks, remainder = divmod(count, chunk)
    if blocks == 0:
        return function(points)
    result = jax.lax.map(function, points[: blocks * chunk].reshape((blocks, chunk, 3)))
    complete = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:])), result)
    if remainder == 0:
        return complete
    partial = function(points[blocks * chunk :])
    return jax.tree.map(
        lambda first, last: jnp.concatenate((first, last)), complete, partial
    )


# ------------------------------------------------------------------ prepared


def _gauss(order: int, /) -> tuple[np.ndarray, np.ndarray]:
    rule = gauss_legendre_data(order)
    return np.asarray(rule.nodes, dtype=np.float64), np.asarray(
        rule.weights, dtype=np.float64
    )


def _coedge_quadrature_breaks(
    pcurve: BRepPCurve,
    first: float,
    last: float,
    subdivisions: int,
    budget: BRepQueryBudget,
    /,
) -> np.ndarray:
    """Every source chart/knot interval is integrated separately before rule refinement."""
    source = pcurve.source_curve if isinstance(pcurve, AffinePCurve) else pcurve
    visits = (
        source.curve.num_charts + 1
        if isinstance(source, IntersectionPCurve)
        else source.knots.size
        if isinstance(source, BSplineCurve)
        else 0
    )
    scratch = 160 * (visits + 2) + 96 * budget.points
    budget.admit(visits, 0, scratch)
    budget.consume(visits)
    boundaries = [first, last]
    if isinstance(source, IntersectionPCurve):
        for chart in range(source.curve.num_charts + 1):
            parameter = (
                source.first + source.last - chart if source.reversed else float(chart)
            )
            if first < parameter < last:
                boundaries.append(parameter)
    elif isinstance(source, BSplineCurve):
        boundaries.extend(
            float(value) for value in np.asarray(source.knots) if first < value < last
        )
    ordered = sorted(set(boundaries))
    count = (len(ordered) - 1) * subdivisions + 1
    budget.admit(count, 0, scratch + 24 * count)
    budget.consume(count)
    return np.concatenate(
        [
            np.linspace(start, end, subdivisions + 1, dtype=np.float64)[:-1]
            for start, end in zip(ordered[:-1], ordered[1:], strict=True)
        ]
        + [np.asarray([last], dtype=np.float64)]
    )


def _surface_quadrature_u_breaks(
    patch: AbstractSurfacePatch, box: np.ndarray, budget: BRepQueryBudget, /
) -> np.ndarray:
    """Native source chart walls, without materializing tensor-product pieces."""
    budget.admit(0, 0, 128)
    cuts = {float(box[0, 0]), float(box[1, 0])}
    scratch = 128

    def knots(values: Array, first: float, last: float) -> None:
        nonlocal scratch
        count = values.size
        # Source visits and simultaneous host leaf/set/publication storage,
        # admitted before transferring or growing any source-derived payload.
        scratch += 160 * count
        budget.admit(count, 0, scratch)
        budget.consume(count)
        for value in np.asarray(values):
            if first < value < last:
                cuts.add(float(value))

    def curve_walls(
        curve: AbstractCurve | IntersectionCurve, first: float, last: float
    ) -> None:
        nonlocal scratch
        budget.admit(1, 0, scratch)
        budget.consume(1)
        if isinstance(curve, PlacedCurve):
            curve_walls(curve.definition, first, last)
        elif isinstance(curve, OffsetCurve):
            curve_walls(curve.base, first, last)
        elif isinstance(curve, SurfaceIsoparametricCurve):
            surface_walls(
                curve.surface, curve.parameter_box(first, last), 1 - curve.fixed_axis
            )
        elif isinstance(curve, BSplineCurve):
            knots(curve.knots, first, last)
        elif isinstance(curve, IntersectionCurve):
            count = curve.num_charts + 1
            scratch += 160 * count
            budget.admit(count, 0, scratch)
            budget.consume(count)
            for chart in range(count):
                if first < chart < last:
                    cuts.add(float(chart))

    def surface_walls(
        source: AbstractSurfacePatch, bounds: np.ndarray, axis: int
    ) -> None:
        budget.admit(1, 0, scratch)
        budget.consume(1)
        if isinstance(source, PlacedSurface):
            surface_walls(source.definition, bounds, axis)
        elif isinstance(source, OffsetSurface):
            surface_walls(source.base, bounds, axis)
        elif isinstance(source, ExtrusionSurface) and axis == 0:
            curve_walls(source.curve, float(bounds[0, 0]), float(bounds[1, 0]))
        elif isinstance(source, RevolutionSurface) and axis == 1:
            curve_walls(source.curve, float(bounds[0, 1]), float(bounds[1, 1]))
        elif isinstance(source, RuledSurface) and axis == 0:
            first, last = float(bounds[0, 0]), float(bounds[1, 0])
            curve_walls(source.first, first, last)
            curve_walls(source.second, first, last)
        elif isinstance(source, BSplineSurfacePatch):
            knots(
                source.u_knots if axis == 0 else source.v_knots,
                float(bounds[0, axis]),
                float(bounds[1, axis]),
            )

    surface_walls(patch, box, 0)
    return np.asarray(sorted(cuts), dtype=np.float64)


def _green_nodes(
    geometry: BRepGeometry,
    face: int,
    box: np.ndarray,
    order: int,
    subdivisions: int,
    u_breaks: np.ndarray,
    budget: BRepQueryBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter nodes and weights with ``sum w g(u, v) = int_D g du dv``."""
    nodes, weights = _gauss(order)
    ranges = np.asarray(geometry.edge_ranges)
    lower_u = float(box[0, 0])
    parameters, node_weights = [], []
    for loop in geometry.face_loops[face]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            first, last = ranges[edge]
            pcurve = geometry.pcurves[coedge]
            breaks = _coedge_quadrature_breaks(
                pcurve, float(first), float(last), subdivisions, budget
            )
            outer_count = (breaks.size - 1) * nodes.size
            budget.admit(
                outer_count,
                outer_count,
                96 * budget.points + 128 * outer_count,
            )
            budget.consume(outer_count, points=outer_count)
            half = 0.5 * (breaks[1:] - breaks[:-1])
            t = (0.5 * (breaks[1:] + breaks[:-1]))[:, None] + half[:, None] * nodes
            outer_weights = (half[:, None] * weights).reshape(-1)
            t_flat = jnp.asarray(t.reshape(-1))
            uv = np.asarray(pcurve.evaluate(t_flat))
            velocity = np.asarray(jax.vmap(jax.jacfwd(pcurve.evaluate))(t_flat))
            outer = geometry.coedge_senses[coedge] * outer_weights * velocity[:, 1]
            keep = outer != 0.0
            boundary_u, boundary_v, outer = uv[keep, 0], uv[keep, 1], outer[keep]
            for first_u, last_u in zip(u_breaks[:-1], u_breaks[1:], strict=True):
                lower = np.maximum(lower_u, first_u)
                upper = np.minimum(boundary_u, last_u)
                selected = upper > lower
                if not np.any(selected):
                    continue
                count = int(np.count_nonzero(selected)) * nodes.size
                scratch = 96 * (budget.points + count) + 24 * boundary_u.size
                budget.admit(count, count, scratch)
                budget.consume(count, points=count)
                span = 0.5 * (upper[selected] - lower)
                inner_u = (lower + span)[:, None] + span[:, None] * nodes
                parameters.append(
                    np.stack(
                        (
                            inner_u,
                            np.broadcast_to(boundary_v[selected, None], inner_u.shape),
                        ),
                        axis=-1,
                    ).reshape(-1, 2)
                )
                node_weights.append(
                    (outer[selected, None] * span[:, None] * weights).reshape(-1)
                )
    return np.concatenate(parameters), np.concatenate(node_weights)


def _face_quadrature(
    patch: AbstractSurfacePatch,
    orientation: float,
    parameters: np.ndarray,
    weights: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv = jnp.asarray(parameters)
    points = np.asarray(patch.evaluate(uv))
    differential = np.asarray(jax.vmap(jax.jacfwd(patch.evaluate))(uv))
    normal = np.cross(differential[:, :, 0], differential[:, :, 1])
    return (
        points,
        orientation * weights[:, None] * normal,
        weights * np.linalg.norm(normal, axis=1),
    )


class _SquaredDistance(StrictModule):
    evaluator: SurfaceEvaluator

    def evaluate(self, values: Array, /) -> Array:
        difference = self.evaluator.evaluate(values[:2]) - values[2:]
        return jnp.sum(difference * difference)

    def gradient(self, values: Array, /) -> Array:
        return jax.jacfwd(self.evaluate)(values)[:2]

    def hessian(self, values: Array, /) -> Array:
        return jax.jacfwd(self.gradient)(values)[:, :2]


@dataclass(frozen=True, slots=True, eq=False)
class _FaceCertificate:
    face: int
    piece_index: int
    lower: tuple[float, float]
    upper: tuple[float, float]
    value: PreparedIntervalFunction
    gradient: PreparedIntervalFunction
    hessian: PreparedIntervalFunction

    def bound(self, box: np.ndarray, point: np.ndarray, /) -> float:
        middle = 0.5 * (box[0] + box[1])
        center = np.concatenate((middle, point))[None]
        lower_value, _ = self.value.evaluate(center, center)
        lower = np.concatenate((box[0], point))[None]
        upper = np.concatenate((box[1], point))[None]
        gradient = self.gradient.evaluate(lower, upper)
        offset = (
            np.nextafter(box[0] - middle, -np.inf),
            np.nextafter(box[1] - middle, np.inf),
        )
        terms = interval_multiply((gradient[0][0], gradient[1][0]), offset)
        total = (lower_value[0], lower_value[0])
        for axis in range(2):
            total = interval_add(total, (terms[0][axis], terms[1][axis]))
        return float(np.nextafter(np.sqrt(max(0.0, float(total[0]))), -np.inf))


def _face_certificates(faces: tuple[_FaceData, ...], /) -> tuple[_FaceCertificate, ...]:
    certificates = []
    for face, data in enumerate(faces):
        for piece in surface_pieces(
            SurfaceRegion(
                data.patch, np.asarray((data.lower, data.upper), dtype=np.float64)
            )
        ):
            objective = _SquaredDistance(piece.evaluator)
            coefficients = coefficient_enclosures(objective)
            certificates.append(
                _FaceCertificate(
                    face,
                    piece.index,
                    (float(piece.lower[0]), float(piece.lower[1])),
                    (float(piece.upper[0]), float(piece.upper[1])),
                    prepare_interval_function(
                        objective.evaluate,
                        5,
                        batch_capacity=1,
                        constant_bounds=coefficients,
                    ),
                    prepare_interval_function(
                        objective.gradient,
                        5,
                        batch_capacity=1,
                        constant_bounds=coefficients,
                    ),
                    prepare_interval_function(
                        objective.hessian,
                        5,
                        batch_capacity=1,
                        constant_bounds=coefficients,
                    ),
                )
            )
    return tuple(certificates)


_INTERVAL_CERTIFICATES: OrderedDict[str, tuple[_FaceCertificate, ...]] = OrderedDict()
_INTERVAL_CERTIFICATES_LOCK = Lock()


def _interval_certificates(query: PreparedBRepQuery, /) -> tuple[_FaceCertificate, ...]:
    """Bounded runtime compilation cache; no live program is authored query state."""
    with _INTERVAL_CERTIFICATES_LOCK:
        cached = _INTERVAL_CERTIFICATES.get(query.query_id)
        if cached is not None:
            _INTERVAL_CERTIFICATES.move_to_end(query.query_id)
            return cached
    prepared = _face_certificates(query.faces)
    # Compilation does not hold the cache lock. Concurrent publication and
    # eviction are atomic, and every caller retains its immutable certificate.
    with _INTERVAL_CERTIFICATES_LOCK:
        cached = _INTERVAL_CERTIFICATES.setdefault(query.query_id, prepared)
        _INTERVAL_CERTIFICATES.move_to_end(query.query_id)
        if len(_INTERVAL_CERTIFICATES) > 8:
            _INTERVAL_CERTIFICATES.popitem(last=False)
        return cached


def _prepare_edge(geometry: BRepGeometry, edge: int, resolution: int, /) -> _EdgeData:
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    start, end = geometry.edge_vertices[edge]
    points = np.asarray(geometry.vertex_points)
    curve_index = geometry.edge_curves[edge]
    curve = None if curve_index == -1 else geometry.curves[curve_index]
    if isinstance(curve, IntersectionCurve):
        parameter_box = geometry.edge_parameter_enclosure(edge)
        box_first, box_last = float(parameter_box[0, 0]), float(parameter_box[1, 1])
    else:
        box_first, box_last = first, last
    endpoints = np.linspace(box_first, box_last, resolution + 1, dtype=np.float64)
    parameters = 0.5 * (endpoints[:-1] + endpoints[1:])
    seed_points = (
        np.broadcast_to(points[start], (resolution, 3))
        if curve is None
        else np.asarray(_curve_value(curve, jnp.asarray(parameters)))
    )
    interval_boxes = (
        np.broadcast_to(np.stack((points[start], points[start])), (resolution, 2, 3))
        if curve is None
        else np.stack(
            [
                curve.bounding_box(
                    float(a), float(b), endpoint_roots=geometry.edge_endpoint_roots[edge]
                )
                if isinstance(curve, IntersectionCurve)
                else curve.bounding_box(float(a), float(b))
                for a, b in zip(endpoints[:-1], endpoints[1:], strict=True)
            ]
        )
    )
    return _EdgeData(
        curve=curve,
        start_point=jnp.asarray(points[start]),
        end_point=jnp.asarray(points[end]),
        seed_parameters=jnp.asarray(parameters),
        seed_points=jnp.asarray(seed_points),
        interval_boxes=jnp.asarray(interval_boxes, dtype=jnp.float64),
        discovery_bvh=prepare_bvh(
            interval_boxes[:, 0], interval_boxes[:, 1], dtype=jnp.float64
        ),
        first=first,
        last=last,
        closed=curve is not None and start == end,
    )


def _carrier_envelope(patch: AbstractSurfacePatch, /) -> tuple[Array, Array, Array]:
    center = np.zeros((3,), dtype=np.float64)
    projection = np.zeros((3, 3), dtype=np.float64)
    radii = np.zeros((2,), dtype=np.float64)
    if isinstance(patch, PlanePatch):
        center = np.asarray(patch.origin)
        normal = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            raise ValueError("A native query requires a regular plane carrier.")
        unit = normal / norm
        projection = np.outer(unit, unit)
    elif isinstance(patch, SpherePatch):
        center = np.asarray(patch.center)
        projection = np.eye(3, dtype=np.float64)
        frame = np.stack(
            (
                np.asarray(patch.first_axis),
                np.asarray(patch.second_axis),
                np.asarray(patch.axis),
            ),
            axis=1,
        )
        singular = np.linalg.svd(frame, compute_uv=False)
        radius = abs(float(patch.radius))
        margin = 128.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(frame)))
        radii = radius * np.asarray(
            (max(0.0, singular[-1] - margin), singular[0] + margin)
        )
    elif isinstance(patch, CylinderPatch):
        center = np.asarray(patch.origin)
        axis = np.asarray(patch.axis)
        length = np.linalg.norm(axis)
        if length == 0.0:
            raise ValueError("A native query requires a nonzero cylinder axis.")
        unit = axis / length
        projection = np.eye(3, dtype=np.float64) - np.outer(unit, unit)
        frame = projection @ np.stack(
            (np.asarray(patch.first_axis), np.asarray(patch.second_axis)), axis=1
        )
        singular = np.linalg.svd(frame, compute_uv=False)
        radius = abs(float(patch.radius))
        margin = 128.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(frame)))
        radii = radius * np.asarray(
            (max(0.0, singular[-1] - margin), singular[0] + margin)
        )
    return (
        jnp.asarray(center, dtype=jnp.float64),
        jnp.asarray(projection, dtype=jnp.float64),
        jnp.asarray(radii, dtype=jnp.float64),
    )


def _periodic_rectangle(
    geometry: BRepGeometry, face: int, patch: AbstractSurfacePatch, box: np.ndarray, /
) -> bool:
    """Prove four chart walls from native integer gauges and authored edge phases."""
    from ._intersection_curve import _affine_curve_coefficients

    loops = geometry.face_loops[face]
    if len(loops) != 1 or len(loops[0]) != 4:
        return False
    if not any(
        isinstance(geometry.pcurves[coedge], PeriodicPCurve) for coedge in loops[0]
    ):
        return False
    bounds = tuple(tuple(Fraction(float(value)) for value in row) for row in box)
    corners: list[tuple[tuple[int, int], tuple[int, int]]] = []
    walls: set[tuple[int, int]] = set()
    for coedge in loops[0]:
        roots = geometry.coedge_endpoint_roots[coedge]
        if any(
            root is not None and not isinstance(root, NativePeriodEndpoint)
            for root in roots
        ):
            return False
        extracted = pcurve_periodic_source(geometry.pcurves[coedge], patch)
        if extracted is None:
            return False
        base, shifts = extracted
        if not isinstance(base, AbstractCurve):
            return False
        coefficients = _affine_curve_coefficients(base, np.zeros((2,), dtype=np.float64))
        if coefficients is None:
            return False
        origin, direction = coefficients
        fixed = [axis for axis in range(2) if direction[axis] == 0]
        if len(fixed) != 1:
            return False
        axis = fixed[0]
        if any(root is not None for root in roots) and patch.periods[1 - axis] is None:
            return False
        varying = 1 - axis
        if patch.periods[axis] is not None:
            if origin[axis] != bounds[0][axis] or shifts[axis] not in (0, 1):
                return False
            side = shifts[axis]
        else:
            if shifts[axis] != 0 or origin[axis] not in (
                bounds[0][axis],
                bounds[1][axis],
            ):
                return False
            side = 0 if origin[axis] == bounds[0][axis] else 1
        edge = geometry.coedge_edges[coedge]
        first, last = (float(value) for value in geometry.edge_ranges[edge])
        if patch.periods[varying] is None:
            values = tuple(
                origin[varying] + direction[varying] * Fraction(value)
                for value in (first, last)
            )
            if (
                set(values) != {bounds[0][varying], bounds[1][varying]}
                or shifts[varying] != 0
            ):
                return False
            start = 0 if values[0] == bounds[0][varying] else 1
        else:
            period = patch.periods[varying]
            if period is None:
                raise RuntimeError("A periodic chart wall lost its owning native period.")
            if direction[varying] != 1 or shifts[varying] != 0:
                return False
            if (
                origin[varying] + Fraction(first) != bounds[0][varying]
                or last != first + period
            ):
                return False
            # Authored native-period endpoints are exactly the start and one
            # full turn later on this wall, never floating representatives.
            expected = (
                _period_offset(Fraction(first), Fraction(0)),
                _period_offset(Fraction(first), Fraction(1)),
            )
            if any(
                root is not None
                and (
                    not isinstance(root, NativePeriodEndpoint)
                    or root.exact_parameter != value
                )
                for root, value in zip(roots, expected, strict=True)
            ):
                return False
            carrier_index = geometry.edge_curves[edge]
            if carrier_index >= 0:
                carrier = geometry.curves[carrier_index]
                if not isinstance(
                    carrier, (CircleCurve, EllipseCurve, SurfaceIsoparametricCurve)
                ):
                    return False
                if carrier.period != period:
                    return False
            else:
                if (axis, float(origin[axis])) not in patch.degenerate_isolines(box):
                    return False
            start = 0
        end = 1 - start
        first_corner = (side, start) if axis == 0 else (start, side)
        last_corner = (side, end) if axis == 0 else (end, side)
        corners.append(
            (first_corner, last_corner)
            if geometry.coedge_senses[coedge] > 0
            else (last_corner, first_corner)
        )
        walls.add((axis, side))
    return walls == {(0, 0), (0, 1), (1, 0), (1, 1)} and all(
        last == corners[(index + 1) % len(corners)][0]
        for index, (_, last) in enumerate(corners)
    )


def _full_rectangle(
    geometry: BRepGeometry, face: int, patch: AbstractSurfacePatch, box: np.ndarray, /
) -> bool:
    if _periodic_rectangle(geometry, face, patch, box):
        return True
    loops = geometry.face_loops[face]
    if len(loops) != 1:
        return False
    bounds = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in box
    )
    ranges = np.asarray(geometry.edge_ranges)
    segments: list[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]] = []
    for coedge in loops[0]:
        curve = geometry.pcurves[coedge]
        edge = geometry.coedge_edges[coedge]
        if not isinstance(curve, LineCurve) or any(
            root is not None for root in geometry.coedge_endpoint_roots[coedge]
        ):
            return False
        origin, direction = np.asarray(curve.origin), np.asarray(curve.direction)
        ends = tuple(
            tuple(
                Fraction.from_float(float(origin[axis]))
                + Fraction.from_float(float(parameter))
                * Fraction.from_float(float(direction[axis]))
                for axis in range(2)
            )
            for parameter in ranges[edge]
        )
        first = (ends[0][0], ends[0][1])
        last = (ends[1][0], ends[1][1])
        if not all(
            bounds[0][axis] <= point[axis] <= bounds[1][axis]
            for point in (first, last)
            for axis in range(2)
        ):
            return False
        if not any(
            first[axis] == last[axis]
            and first[axis] in (bounds[0][axis], bounds[1][axis])
            for axis in range(2)
        ):
            return False
        segments.append(
            (first, last) if geometry.coedge_senses[coedge] > 0 else (last, first)
        )
    if any(
        last != segments[(index + 1) % len(segments)][0]
        for index, (_, last) in enumerate(segments)
    ):
        return False
    area = sum(
        (first[0] * last[1] - first[1] * last[0] for first, last in segments), Fraction(0)
    )
    return abs(area) == 2 * (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])


def _whole_turn_chart(model: BRepModel, face_index: int, face: _FaceData, /) -> bool:
    """Prove the trim is the whole chart and spans one complete native turn."""
    if face.full_rectangle:
        return True
    geometry = model.geometry
    if geometry is None:
        raise RuntimeError("Prepared whole-turn containment lost its source geometry.")
    # A root-enclosure box may exceed one turn. Prove its actual native-gauge
    # walls rather than treating that box as a trim.
    native_box = np.asarray(
        ((0.0, face.lower[1]), (face.periods[0], face.upper[1])), dtype=np.float64
    )
    return (
        face.lower[0] <= 0.0
        and face.upper[0] >= face.periods[0]
        and _periodic_rectangle(geometry, face_index, face.patch, native_box)
    )


def _solid_sphere_source(
    model: BRepModel,
    faces: tuple[int, ...],
    face: _FaceData,
    /,
) -> _SphereSource | None:
    """Prove whole-source angular coverage before using exact radial membership."""
    if not (
        len(faces) == 1
        and face.periods[0] > 0.0
        and face.lower[1] == -0.5 * pi
        and face.upper[1] == 0.5 * pi
    ):
        return None
    source = prepare_full_sphere(face.patch)
    if source is None or _whole_turn_chart(model, faces[0], face):
        return source
    return None


def _solid_revolution_source(
    model: BRepModel,
    faces: tuple[int, ...],
    face: _FaceData,
    /,
) -> _RevolutionSource | None:
    """Prove one complete turn of an exactly closed meridian before half-plane parity."""
    if len(faces) != 1 or not face.periods[0] > 0.0:
        return None
    source = prepare_full_revolution(face.patch, face.lower[1], face.upper[1])
    if source is None or not _whole_turn_chart(model, faces[0], face):
        return None
    return source


def _prepare_face(
    model: BRepModel, geometry: BRepGeometry, face: int, resolution: int, /
) -> _FaceData:
    patch = model.patches[face]
    box = np.asarray(model.parameter_bounds)[face]
    periods = tuple(
        float(period)
        if period is not None and box[1, axis] - box[0, axis] >= period * (1.0 - 1.0e-12)
        else 0.0
        for axis, period in enumerate(patch.periods)
    )
    cells = [box]
    while len(cells) < resolution * resolution:
        following = []
        for cell in cells:
            axis = int(np.argmax((cell[1] - cell[0]) / (box[1] - box[0])))
            middle = 0.5 * (cell[0, axis] + cell[1, axis])
            first, second = cell.copy(), cell.copy()
            first[1, axis], second[0, axis] = middle, middle
            following.extend((first, second))
        cells = following
    seed_uv = np.stack([0.5 * (cell[0] + cell[1]) for cell in cells])
    interval_boxes = np.stack([patch.bounding_box(cell) for cell in cells])
    trim_domain = brep_trim_domain(
        geometry, face, patch, tolerance=float(np.max(box[1] - box[0])) / resolution
    )
    coedges = tuple(c for loop in geometry.face_loops[face] for c in loop)
    edges = tuple(geometry.coedge_edges[c] for c in coedges)
    poles = np.asarray(
        [
            np.asarray(geometry.vertex_points)[geometry.edge_vertices[edge][0]]
            for edge in edges
            if geometry.edge_curves[edge] == -1
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    pole_parameters = np.asarray(
        [
            np.asarray(
                geometry.pcurves[coedge].evaluate(
                    jnp.asarray(
                        np.asarray(geometry.edge_ranges)[edge, 0], dtype=jnp.float64
                    )
                )
            )
            for coedge, edge in zip(coedges, edges, strict=True)
            if geometry.edge_curves[edge] == -1
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    center, projection, radii = _carrier_envelope(patch)
    return _FaceData(
        patch=patch,
        seed_uv=jnp.asarray(seed_uv),
        seed_points=jnp.asarray(np.asarray(patch.evaluate(jnp.asarray(seed_uv)))),
        trim_domain=trim_domain,
        interval_boxes=jnp.asarray(interval_boxes, dtype=jnp.float64),
        discovery_bvh=prepare_bvh(
            interval_boxes[:, 0], interval_boxes[:, 1], dtype=jnp.float64
        ),
        parameter_boxes=jnp.asarray(np.stack(cells), dtype=jnp.float64),
        pole_points=jnp.asarray(poles),
        pole_parameters=jnp.asarray(pole_parameters, dtype=jnp.float64),
        pcurves=tuple(geometry.pcurves[c] for c in coedges),
        carrier_center=center,
        carrier_projection=projection,
        carrier_radii=radii,
        lower=(float(box[0, 0]), float(box[0, 1])),
        upper=(float(box[1, 0]), float(box[1, 1])),
        periods=(periods[0], periods[1]),
        orientation=float(np.asarray(model.orientation)[face]),
        coedge_edges=edges,
        seam_coedges=tuple(edges.count(edge) > 1 for edge in edges),
        full_rectangle=_full_rectangle(geometry, face, patch, box),
    )


@final
class PreparedBRepQuery(StrictModule, NonTrainableState):
    """Prepared native closest-point, containment, bound and measure queries.

    Construct with :func:`prepare_brep_query`. Traced point queries are pure JAX
    with conservative prepared-cover evidence. Untraced queries additionally
    run bounded host interval certification against the immutable source.
    """

    model: BRepModel
    faces: tuple[_FaceData, ...]
    edges: tuple[_EdgeData, ...]
    quadrature: _Quadrature
    vertex_points: Array
    face_boxes: Array
    edge_boxes: Array
    face_bvh: PackedBVH
    measures: BRepMeasureResult
    face_solids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    solid_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    occurrences: tuple[BRepOccurrence, ...] = eqx.field(static=True)
    occurrence_rotations: Array
    occurrence_translations: Array
    solid_face_signs: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    boundary_instances: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(static=True)
    container_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    tolerances: _Tolerances
    policy: BRepQueryPolicy = eqx.field(static=True)
    projection_policy: BRepProjectionPolicy = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)
    certify_on_host: bool = eqx.field(static=True)
    world_query: PreparedBRepQuery | None
    world_face_sources: tuple[int, ...] = eqx.field(static=True)
    world_edge_sources: tuple[int, ...] = eqx.field(static=True)
    world_vertex_sources: tuple[int, ...] = eqx.field(static=True)
    world_solids: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        policy: BRepQueryPolicy,
        projection_policy: BRepProjectionPolicy,
        /,
    ) -> None:
        geometry = model.geometry
        if geometry is None:
            raise ValueError(
                "Native B-Rep queries require the model's exact geometry "
                "(curves, p-curves and loops)."
            )
        face_count = len(model.patches)
        if face_count == 0:
            raise ValueError("Closest-point preparation requires a nonempty boundary.")
        points = np.asarray(geometry.vertex_points)
        boxes = np.stack(
            [
                model.patches[face].bounding_box(np.asarray(model.parameter_bounds)[face])
                for face in range(face_count)
            ]
        )
        scale = max(
            1.0, float(np.max(np.max(boxes[:, 1], axis=0) - np.min(boxes[:, 0], axis=0)))
        )
        self.model = model
        self.faces = tuple(
            _prepare_face(model, geometry, face, policy.seed_resolution)
            for face in range(face_count)
        )
        self.edges = tuple(
            _prepare_edge(geometry, edge, 4 * policy.seed_resolution + 1)
            for edge in range(len(geometry.edge_curves))
        )
        self.quadrature, self.measures = _prepare_quadrature(model, geometry, policy)
        self.vertex_points = jnp.asarray(points)
        self.face_boxes = jnp.asarray(boxes)
        self.face_bvh = prepare_bvh(boxes[:, 0], boxes[:, 1], dtype=jnp.float64)
        self.edge_boxes = jnp.asarray(
            np.asarray(
                [
                    np.stack(
                        (
                            np.min(np.asarray(edge.interval_boxes)[:, 0], axis=0),
                            np.max(np.asarray(edge.interval_boxes)[:, 1], axis=0),
                        )
                    )
                    for edge in self.edges
                ],
                dtype=np.float64,
            ).reshape(-1, 2, 3)
        )
        self.face_solids = model.topology.face_solids
        self.solid_faces = model.topology.solid_faces
        self.occurrences = geometry.occurrences
        self.occurrence_rotations = jnp.asarray(
            [value.rotation for value in self.occurrences], dtype=jnp.float64
        ).reshape(-1, 3, 3)
        self.occurrence_translations = jnp.asarray(
            [value.translation for value in self.occurrences], dtype=jnp.float64
        ).reshape(-1, 3)
        membership = {
            path: container.path
            for container in geometry.assembly_containers
            for path in container.member_paths
        }
        self.container_paths = tuple(
            membership.get(occurrence.path, ()) for occurrence in geometry.occurrences
        )
        self.solid_face_signs = model.topology.solid_face_orientations
        self.boundary_instances = self._boundary_instances()
        self.tolerances = _Tolerances(
            distance=projection_policy.ambiguity_tolerance,
            parametric=projection_policy.parametric_tolerance,
            classifier=projection_policy.classifier_tolerance,
            scale=jnp.asarray(scale, dtype=jnp.float64),
            newton_steps=policy.newton_steps,
            seeds=policy.seeds,
        )
        self.policy = policy
        self.projection_policy = projection_policy
        self.certify_on_host = True
        self.source_revision = model.source_revision
        self.model_id = model.model_id
        self.query_id = canonical_fingerprint(
            {
                "kind": "prepared-native-brep-query",
                "model_id": model.model_id,
                "policy": policy.policy_id,
                "projection_policy": projection_policy.policy_id,
            }
        )
        self.world_query = None
        self.world_face_sources = ()
        self.world_edge_sources = ()
        self.world_solids = ()
        self.world_vertex_sources = ()
        placed = any(
            not np.array_equal(occurrence.rotation, np.eye(3))
            or not np.array_equal(occurrence.translation, np.zeros(3))
            for occurrence in self.occurrences
        )
        if placed:
            from ._constructors import BRepTessellationPolicy
            from ._placement import materialize_brep_occurrences

            materialized = materialize_brep_occurrences(
                model,
                tessellation=BRepTessellationPolicy(realize=False),
            )
            world = PreparedBRepQuery(materialized.model, policy, projection_policy)
            face_sources = [-1] * len(world.faces)
            edge_sources = [-1] * len(world.edges)
            vertex_sources = [-1] * len(world.vertex_points)
            solids = [-1] * len(self.occurrences)
            paths = {
                occurrence.path: index
                for index, occurrence in enumerate(self.occurrences)
            }
            for source, target in zip(
                materialized.source_entities, materialized.target_entities, strict=True
            ):
                if target.kind == "face":
                    face_sources[target.index] = source.index
                elif target.kind == "edge":
                    edge_sources[target.index] = source.index
                elif target.kind == "vertex":
                    vertex_sources[target.index] = source.index
                elif target.kind == "solid":
                    solids[paths[source.occurrence_path]] = target.index
            if (
                min((*face_sources, *edge_sources, *vertex_sources, *solids), default=0)
                < 0
            ):
                raise RuntimeError(
                    "World query preparation lost exhaustive source incidence."
                )
            self.world_query = world
            self.world_face_sources = tuple(face_sources)
            self.world_edge_sources = tuple(edge_sources)
            self.world_vertex_sources = tuple(vertex_sources)
            self.world_solids = tuple(solids)

    def _world_closest(
        self,
        points: Array,
        faces: tuple[int, ...],
        occurrence: int,
        /,
        *,
        budget: BRepQueryBudget | None = None,
    ) -> BRepSurfaceQueryResult:
        world = self.world_query
        if world is None:
            raise RuntimeError(
                "A world-source query requires prepared occurrence geometry."
            )
        allowed = (
            world.solid_faces[self.world_solids[occurrence]]
            if occurrence >= 0
            else tuple(
                face for face, solids in enumerate(world.face_solids) if not solids
            )
        )
        selected = tuple(
            face for face in allowed if self.world_face_sources[face] in faces
        )
        result = world._closest_definition(points, selected, budget=budget)
        source_faces = jnp.asarray(self.world_face_sources, dtype=jnp.int32)
        source_edges = jnp.asarray(self.world_edge_sources, dtype=jnp.int32)
        return eqx.tree_at(
            lambda value: (value.faces, value.edge_indices, value.occurrences),
            result,
            (
                source_faces[result.faces],
                jnp.where(
                    result.edge_indices >= 0,
                    source_edges[jnp.maximum(result.edge_indices, 0)],
                    -1,
                ),
                jnp.full(result.distances.shape, occurrence, dtype=jnp.int32),
            ),
        )

    @property
    def bounds(self) -> Array:
        """Outward-rounded axis-aligned box of the authored boundary occurrences."""
        boxes = []
        for occurrence, faces in self.boundary_instances:
            box = jnp.stack(
                (
                    jnp.min(self.face_boxes[jnp.asarray(faces), 0], axis=0),
                    jnp.max(self.face_boxes[jnp.asarray(faces), 1], axis=0),
                )
            )
            rotation, translation = self._placement(occurrence)
            mapped = _jvector((rotation, rotation), (box[0], box[1]))
            lower, upper = _jadd(mapped, (translation, translation))
            boxes.append(jnp.stack((lower, upper)))
        stacked = jnp.stack(boxes)
        return jnp.stack((jnp.min(stacked[:, 0], axis=0), jnp.max(stacked[:, 1], axis=0)))

    def _boundary_instances(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        if not self.occurrences:
            return ((-1, tuple(range(len(self.faces)))),)
        instances = []
        for index, occurrence in enumerate(self.occurrences):
            faces = []
            for face in self.solid_faces[occurrence.solid]:
                shared = any(
                    other.solid != occurrence.solid
                    and bool(self.container_paths[index])
                    and self.container_paths[other_index] == self.container_paths[index]
                    and face in self.solid_faces[other.solid]
                    for other_index, other in enumerate(self.occurrences)
                )
                if not shared:
                    faces.append(face)
            if faces:
                instances.append((index, tuple(faces)))
        unshelled = tuple(
            face for face, solids in enumerate(self.face_solids) if not solids
        )
        if unshelled:
            instances.append((-1, unshelled))
        return tuple(instances)

    def _placement(self, occurrence: int, /) -> tuple[Array, Array]:
        if occurrence == -1:
            return jnp.eye(3, dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64)
        return self.occurrence_rotations[occurrence], self.occurrence_translations[
            occurrence
        ]

    def _prepare_placed_query(
        self,
        occurrence: int,
        faces: tuple[int, ...],
        /,
    ) -> tuple[tuple[AffineFace, ...], QueryBroadphase, np.ndarray] | None:
        from ._affine_query import prepare_affine_faces
        from ._placed import source_transform_bounds

        rotation, translation = self._placement(occurrence)
        matrix, shift = np.asarray(rotation), np.asarray(translation)
        prepared = prepare_affine_faces(self.model, faces, pose=(matrix, shift))
        if prepared is None:
            return None
        boxes = []
        for box in np.asarray(self.face_boxes):
            transformed = source_transform_bounds(matrix, box[0], box[1])
            boxes.append(np.stack(interval_add(transformed, (shift, shift))))
        bounds = np.asarray(boxes, dtype=np.float64)
        broadphase = prepare_bvh(bounds[:, 0], bounds[:, 1], dtype=jnp.float64)
        return prepared, QueryBroadphase.prepare(broadphase), bounds

    def _path_index(self, path: tuple[str, ...], /) -> int:
        for index, occurrence in enumerate(self.occurrences):
            if occurrence.path == path:
                return index
        raise ValueError("path must name an existing solid occurrence.")

    def _edge_hits(self, points: Array, /) -> tuple[dict[str, Array], ...]:
        return tuple(_edge_query(edge, points, self.tolerances) for edge in self.edges)

    def _face_hits(self, points: Array, faces: tuple[int, ...], /) -> dict[str, Array]:
        selected_edges = tuple(
            sorted({edge for face in faces for edge in self.faces[face].coedge_edges})
        )
        edge_hits = {
            edge: _edge_query(self.edges[edge], points, self.tolerances)
            for edge in selected_edges
        }
        hits = [
            _face_query(
                self.faces[face],
                tuple(edge_hits[edge] for edge in self.faces[face].coedge_edges),
                points,
                self.tolerances,
            )
            for face in faces
        ]
        return {key: jnp.stack([hit[key] for hit in hits]) for key in hits[0]}

    def _closest_chunk(
        self, points: Array, faces: tuple[int, ...], /
    ) -> BRepSurfaceQueryResult:
        hits = self._face_hits(points, faces)
        count = points.shape[0]
        rows = jnp.arange(count)
        best = jnp.argmin(hits["distance"], axis=0)
        best_point = hits["point"][best, rows]
        best_distance = hits["distance"][best, rows]
        same = (
            jnp.linalg.norm(hits["point"] - best_point[None], axis=-1)
            <= self.tolerances.distance
        )
        tie = jnp.any(
            ~same & (hits["distance"] <= best_distance[None] + self.tolerances.distance),
            axis=0,
        )
        lower_bound = jnp.min(hits["lower_bound"], axis=0)
        competing = hits["lower_bound"] <= best_distance[None] + self.tolerances.distance
        unresolved = (best_distance - lower_bound > self.tolerances.distance) | jnp.any(
            competing & hits["unresolved"], axis=0
        )
        status = jnp.where(
            unresolved, _FAILED, jnp.where(tie, _AMBIGUOUS, hits["status"][best, rows])
        ).astype(jnp.int8)
        return BRepSurfaceQueryResult(
            points=best_point,
            parameters=hits["parameter"][best, rows],
            distances=best_distance,
            faces=jnp.asarray(faces, dtype=jnp.int32)[best],
            status=status,
            normals=hits["normal"][best, rows],
            first_derivatives=hits["du"][best, rows],
            second_derivatives=hits["dv"][best, rows],
            stationarity=hits["stationarity"][best, rows],
            distance_lower_bounds=lower_bound,
            unresolved=unresolved,
            refinements=jnp.zeros((count,), dtype=jnp.int32),
            parameter_bounds=hits["parameter_bound"][best, rows],
            occurrences=jnp.full((count,), -1, dtype=jnp.int32),
            source_revision=self.source_revision,
            proposal_status=jnp.where(tie, _AMBIGUOUS, hits["status"][best, rows]).astype(
                jnp.int8
            ),
            edge_indices=hits["edge_index"][best, rows],
            edge_parameters=hits["edge_parameter"][best, rows],
            point_error_bounds=hits["point_error"][best, rows],
            query_operations=jnp.full(
                (count,),
                1
                + len(faces)
                + len({edge for face in faces for edge in self.faces[face].coedge_edges}),
                dtype=jnp.int64,
            ),
        )

    def _affine_closest(
        self,
        points: np.ndarray,
        faces: tuple[int, ...],
        /,
        *,
        budget: BRepQueryBudget | None = None,
        prepared: tuple[AffineFace, ...] | None = None,
        broadphase: QueryBroadphase | None = None,
        source_boxes: np.ndarray | None = None,
    ) -> BRepSurfaceQueryResult | None:
        from ._affine_query import closest_affine, prepare_affine_faces
        from ._correspondence import _fraction_norm_upper

        if prepared is None:
            prepared = prepare_affine_faces(self.model, faces)
            if prepared is None:
                return None
        if broadphase is None:
            broadphase = QueryBroadphase.prepare(self.face_bvh)
        source_bytes = 4096 * (
            self.face_bvh.node_count
            + len(faces)
            + sum(len(face.segments) for face in prepared)
        )
        chunk_size = self._query_chunk_size(budget, source_bytes)
        if len(points) > chunk_size:
            chunks = [
                self._affine_closest(
                    points[start : start + chunk_size],
                    faces,
                    budget=budget,
                    prepared=prepared,
                    broadphase=broadphase,
                    source_boxes=source_boxes,
                )
                for start in range(0, len(points), chunk_size)
            ]
            if any(chunk is None for chunk in chunks):
                raise RuntimeError(
                    "An affine source lost its query theorem during bounded batching."
                )
            return jax.tree.map(lambda *values: jnp.concatenate(values), *chunks)
        boxes = (
            np.asarray(self.face_boxes)[list(faces)]
            if source_boxes is None
            else source_boxes[list(faces)]
        )
        results = [
            closest_affine(
                point, prepared, boxes, self.tolerances.distance, broadphase, faces
            )
            for point in points
        ]
        if budget is not None:
            budget.consume(sum(result.operations for result in results))
        projected = np.asarray(
            [[float(value) for value in result.point] for result in results],
            dtype=np.float64,
        )
        uv = np.asarray(
            [[float(value) for value in result.parameter] for result in results],
            dtype=np.float64,
        )
        parameter_error = np.asarray(
            [
                _fraction_norm_upper(
                    tuple(
                        Fraction(float(value)) - exact
                        for value, exact in zip(
                            represented, result.parameter, strict=True
                        )
                    )
                )
                for represented, result in zip(uv, results, strict=True)
            ],
            dtype=np.float64,
        )
        point_error = np.asarray(
            [
                _fraction_norm_upper(
                    tuple(
                        Fraction(float(value)) - exact
                        for value, exact in zip(represented, result.point, strict=True)
                    )
                )
                for represented, result in zip(projected, results, strict=True)
            ],
            dtype=np.float64,
        )
        indices = np.asarray([faces[result.face] for result in results], dtype=np.int32)
        first = np.asarray(
            [
                [float(value) for value in prepared[result.face].first_axis]
                for result in results
            ],
            dtype=np.float64,
        )
        second = np.asarray(
            [
                [float(value) for value in prepared[result.face].second_axis]
                for result in results
            ],
            dtype=np.float64,
        )
        normals = np.cross(
            first / np.max(np.abs(first), axis=1)[:, None],
            second / np.max(np.abs(second), axis=1)[:, None],
        )
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals *= np.asarray(self.model.orientation)[indices, None]
        lower, upper = np.asarray(
            [result.distance_bounds for result in results], dtype=np.float64
        ).T
        unresolved = (
            (upper - lower > self.tolerances.distance)
            | (parameter_error > self.tolerances.parametric)
            | (point_error > self.tolerances.distance)
            | ~np.all(np.isfinite(normals), axis=1)
        )
        proposal = np.asarray(
            [
                _AMBIGUOUS
                if result.ambiguous
                else (_SEAM if result.edge >= 0 else _UNIQUE)
                for result in results
            ],
            dtype=np.int8,
        )
        count = len(results)
        delta = projected - points
        gradient = np.stack(
            (np.sum(delta * first, axis=1), np.sum(delta * second, axis=1)), axis=1
        )
        denominator = np.maximum(
            np.linalg.norm(np.stack((first, second), axis=1), axis=(1, 2))
            * np.maximum(upper, self.tolerances.distance),
            np.finfo(np.float64).tiny,
        )
        return BRepSurfaceQueryResult(
            points=jnp.asarray(projected),
            parameters=jnp.asarray(uv),
            distances=jnp.asarray(upper),
            faces=jnp.asarray(indices),
            status=jnp.asarray(np.where(unresolved, _FAILED, proposal), dtype=jnp.int8),
            normals=jnp.asarray(normals),
            first_derivatives=jnp.asarray(first),
            second_derivatives=jnp.asarray(second),
            stationarity=jnp.asarray(np.linalg.norm(gradient, axis=1) / denominator),
            distance_lower_bounds=jnp.asarray(lower),
            unresolved=jnp.asarray(unresolved),
            refinements=jnp.zeros((count,), dtype=jnp.int32),
            parameter_bounds=jnp.asarray(parameter_error),
            occurrences=jnp.full((count,), -1, dtype=jnp.int32),
            proposal_status=jnp.asarray(proposal),
            edge_indices=jnp.asarray(
                [result.edge for result in results], dtype=jnp.int32
            ),
            edge_parameters=jnp.asarray(
                [
                    np.nan
                    if result.edge_parameter is None
                    else float(result.edge_parameter)
                    for result in results
                ],
                dtype=jnp.float64,
            ),
            point_error_bounds=jnp.asarray(point_error),
            source_revision=self.source_revision,
            query_operations=jnp.asarray(
                [1 + result.operations for result in results], dtype=jnp.int64
            ),
        )

    def _closest_definition(
        self,
        points: Array,
        faces: tuple[int, ...],
        /,
        *,
        maximum_boxes: int | None = None,
        budget: BRepQueryBudget | None = None,
        affine_preparation: tuple[tuple[AffineFace, ...], QueryBroadphase, np.ndarray]
        | None = None,
    ) -> BRepSurfaceQueryResult:
        if maximum_boxes is not None and (
            type(maximum_boxes) is not int
            or not 1 <= maximum_boxes <= self.policy.maximum_subdivisions
        ):
            raise ValueError(
                "A scoped closest-point budget must tighten the prepared query budget."
            )
        host = self.certify_on_host and not isinstance(points, Tracer)
        prepared = None
        if host:
            from ._affine_query import prepare_affine_faces

            if budget is None:
                budget = BRepQueryBudget(
                    self.policy.maximum_operations,
                    self.policy.maximum_points,
                    self.policy.maximum_scratch_bytes,
                )
            prepared = (
                prepare_affine_faces(self.model, faces)
                if affine_preparation is None
                else affine_preparation[0]
            )
            edges = len(
                {edge for face in faces for edge in self.faces[face].coedge_edges}
            )
            source_bytes = 4096 * (
                self.face_bvh.node_count
                + len(faces)
                + sum(len(self.faces[face].coedge_edges) for face in faces)
            )
            chunk = self._query_chunk_size(budget, source_bytes)
            boxes = (
                self.policy.maximum_subdivisions
                if maximum_boxes is None
                else maximum_boxes
            )
            closure_count = sum(len(self.faces[face].coedge_edges) for face in faces)
            geometry = self.model.geometry
            if geometry is None:
                raise RuntimeError("A prepared query lost its authoritative geometry.")
            vertices = len(
                {
                    vertex
                    for face in faces
                    for edge in self.faces[face].coedge_edges
                    for vertex in geometry.edge_vertices[edge]
                }
            )
            piece_count = (
                0
                if prepared is not None
                else sum(
                    len(
                        surface_pieces(
                            SurfaceRegion(
                                self.faces[face].patch,
                                np.asarray(
                                    (self.faces[face].lower, self.faces[face].upper),
                                    dtype=np.float64,
                                ),
                            )
                        )
                    )
                    for face in faces
                )
            )
            reservation = points.shape[0] * (
                1
                + 2 * self.face_bvh.node_count
                + 2 * len(faces)
                + 2 * sum(len(face.segments) for face in prepared)
                if prepared is not None
                else 1
                + len(faces)
                + edges
                + boxes
                + closure_count
                + vertices
                + piece_count
            )
            scratch = source_bytes + 4096 * min(points.shape[0], chunk)
            BRepQueryBudget(
                self.policy.maximum_operations,
                self.policy.maximum_points,
                self.policy.maximum_scratch_bytes,
            ).admit(reservation, points.shape[0], scratch)
            budget.admit(reservation, points.shape[0], scratch)
            budget.consume(points.shape[0], points=points.shape[0])
            if prepared is not None:
                affine = self._affine_closest(
                    np.asarray(points),
                    faces,
                    budget=budget,
                    prepared=prepared,
                    broadphase=None
                    if affine_preparation is None
                    else affine_preparation[1],
                    source_boxes=None
                    if affine_preparation is None
                    else affine_preparation[2],
                )
                if affine is None:
                    raise RuntimeError(
                        "An admitted affine source lost its closest-point theorem."
                    )
                return affine
        elif budget is not None:
            raise ValueError(
                "A mutable host query budget cannot be consumed by traced execution."
            )
        result = _compiled_closest_chunks(
            self, points, faces, min(self.policy.chunk_size, points.shape[0])
        )
        if not host:
            return result
        result = self._certify_definition(
            np.asarray(points), faces, result, maximum_boxes=maximum_boxes
        )
        if budget is not None:
            refinements = int(np.sum(np.asarray(result.refinements)))
            budget.consume(
                int(np.sum(np.asarray(result.query_operations))) - points.shape[0],
                subdivisions=refinements,
            )
        return result

    def _certify_definition(
        self,
        points: np.ndarray,
        faces: tuple[int, ...],
        result: BRepSurfaceQueryResult,
        /,
        *,
        maximum_boxes: int | None = None,
    ) -> BRepSurfaceQueryResult:
        from ._stationary_isolation import (
            isolate_closest_stationary,
            source_closure_upper,
            stationary_root_fidelity,
            stationary_root_representative,
        )

        if (
            self.policy.maximum_depth == 0
            or (
                self.policy.maximum_subdivisions
                if maximum_boxes is None
                else maximum_boxes
            )
            == 0
        ):
            return result
        pending = np.asarray(result.unresolved, dtype=np.bool_)
        if not np.any(pending):
            return result
        parameters = np.array(result.parameters, dtype=np.float64)
        indices = np.array(result.faces, dtype=np.int32)
        distances = np.array(result.distances, dtype=np.float64)
        lower = np.array(result.distance_lower_bounds, dtype=np.float64)
        projected = np.array(result.points, dtype=np.float64)
        normals = np.array(result.normals, dtype=np.float64)
        first = np.array(result.first_derivatives, dtype=np.float64)
        second = np.array(result.second_derivatives, dtype=np.float64)
        stationarity = np.array(result.stationarity, dtype=np.float64)
        refinements = np.array(result.refinements, dtype=np.int32)
        parameter_bound = np.array(result.parameter_bounds, dtype=np.float64)
        point_error = np.array(result.point_error_bounds, dtype=np.float64)
        edge_indices = np.array(result.edge_indices, dtype=np.int32)
        edge_parameters = np.array(result.edge_parameters, dtype=np.float64)
        unresolved = np.array(pending, dtype=np.bool_)
        status = np.array(result.status, dtype=np.int8)
        certificates = _interval_certificates(self)
        domains = tuple(face.trim_domain for face in self.faces)
        geometry = self.model.geometry
        if geometry is None:
            raise RuntimeError(
                "Prepared native closest-point certification lost its exact source geometry."
            )
        source_coedges = tuple(
            coedge
            for face in faces
            for loop in geometry.face_loops[face]
            for coedge in loop
        )
        correspondence = float(
            np.max(
                np.asarray(self.model.coedge_deviation_bounds)[list(source_coedges)],
                initial=0.0,
            )
        )
        cover_operations = np.zeros((len(points),), dtype=np.int64)
        vertices = {
            vertex
            for coedge in source_coedges
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        }
        initial_cover = len(vertices) + sum(
            certificate.face in faces for certificate in certificates
        )
        for row, point in enumerate(points):
            if not pending[row]:
                continue
            # A numerical proposal is not a root-error certificate.
            parameter_bound[row] = point_error[row] = np.inf
            upper = source_closure_upper(self.model, point, faces)
            cover_operations[row] += len(source_coedges)
            if not np.isfinite(upper) or not np.isfinite(correspondence):
                status[row] = _FAILED
                continue
            isolation = isolate_closest_stationary(
                self.model,
                point,
                certificates=certificates,
                trim_domains=domains,
                distance_upper=upper + correspondence,
                tolerance=self.tolerances.distance,
                parameter_tolerance=self.tolerances.parametric,
                max_boxes=self.policy.maximum_subdivisions
                if maximum_boxes is None
                else maximum_boxes,
                max_depth=self.policy.maximum_depth,
                faces=faces,
            )
            cover_operations[row] += initial_cover
            refinements[row] = isolation.boxes_processed
            if not isolation.complete or not isolation.roots:
                if not isolation.complete:
                    # Every minimizer lies in a retained root or an undecided
                    # stratum; boxes certified outside their trim never count.
                    # That trim-aware floor is a sound lower bound even here.
                    floor = min(
                        (
                            root.distance_lower
                            - stationary_root_fidelity(self.model, root)
                            for root in isolation.roots
                        ),
                        default=np.inf,
                    )
                    floor = min(
                        floor, isolation.unresolved_distance_lower, upper + correspondence
                    )
                    lower[row] = max(
                        lower[row], max(0.0, float(np.nextafter(floor, -np.inf)))
                    )
                status[row] = _FAILED
                continue
            errors = tuple(
                stationary_root_fidelity(self.model, root) for root in isolation.roots
            )
            best_index = min(
                range(len(isolation.roots)),
                key=lambda index: isolation.roots[index].distance_upper + errors[index],
            )
            best = isolation.roots[best_index]
            best_upper = best.distance_upper + errors[best_index]
            competitors = tuple(
                root
                for root, error in zip(isolation.roots, errors, strict=True)
                if root.distance_lower - error <= best_upper + self.tolerances.distance
            )
            (
                projected[row],
                parameters[row],
                parameter_bound[row],
                point_error[row],
                edge_indices[row],
                edge_parameters[row],
                regular,
            ) = stationary_root_representative(self.model, best)
            point_error[row] = max(point_error[row], errors[best_index])
            indices[row] = best.face
            distances[row] = float(np.nextafter(best_upper, np.inf))
            lower[row] = max(
                0.0,
                float(
                    np.nextafter(
                        min(
                            root.distance_lower - error
                            for root, error in zip(isolation.roots, errors, strict=True)
                        ),
                        -np.inf,
                    )
                ),
            )
            du, dv, normal = _frame(
                self.faces[best.face], jnp.asarray(parameters[row])[None]
            )
            first[row], second[row], normals[row] = (
                np.asarray(du[0]),
                np.asarray(dv[0]),
                np.asarray(normal[0]),
            )
            delta = projected[row] - point
            gradient = np.asarray((delta @ first[row], delta @ second[row]))
            denominator = max(
                np.linalg.norm(np.stack((first[row], second[row])))
                * max(distances[row], self.tolerances.distance),
                np.finfo(np.float64).tiny,
            )
            stationarity[row] = np.linalg.norm(gradient) / denominator
            unresolved[row] = (
                distances[row] - lower[row] > self.tolerances.distance
                or parameter_bound[row] > self.tolerances.parametric
                or point_error[row] > self.tolerances.distance
                or not np.isfinite(distances[row])
                or not np.isfinite(parameter_bound[row])
                or not np.isfinite(point_error[row])
                or not np.all(np.isfinite(projected[row]))
                or not np.all(np.isfinite(normals[row]))
                or not regular
            )
            status[row] = (
                _FAILED
                if unresolved[row]
                else (
                    _AMBIGUOUS
                    if len(competitors) > 1 or best.continuum
                    else (_SEAM if best.dimension < 2 else _UNIQUE)
                )
            )
        return eqx.tree_at(
            lambda value: (
                value.points,
                value.parameters,
                value.faces,
                value.distances,
                value.distance_lower_bounds,
                value.unresolved,
                value.refinements,
                value.parameter_bounds,
                value.status,
                value.normals,
                value.first_derivatives,
                value.second_derivatives,
                value.stationarity,
                value.edge_indices,
                value.edge_parameters,
                value.point_error_bounds,
                value.query_operations,
            ),
            result,
            tuple(
                jnp.asarray(value)
                for value in (
                    projected,
                    parameters,
                    indices,
                    distances,
                    lower,
                    unresolved,
                    refinements,
                    parameter_bound,
                    status,
                    normals,
                    first,
                    second,
                    stationarity,
                    edge_indices,
                    edge_parameters,
                    point_error,
                    np.asarray(result.query_operations) + refinements + cover_operations,
                )
            ),
        )

    def closest_point(
        self,
        points: ArrayLike,
        /,
        *,
        faces: tuple[int, ...] | None = None,
        path: tuple[str, ...] | None = None,
        budget: BRepQueryBudget | None = None,
    ) -> BRepSurfaceQueryResult:
        """World-space boundary projection retaining definition face and occurrence identity."""
        points_ = _points(points)
        faces_ = tuple(range(len(self.faces))) if faces is None else tuple(faces)
        if not faces_ or any(face < 0 or face >= len(self.faces) for face in faces_):
            raise ValueError("faces must list existing face indices.")
        if path is None:
            instances = (
                tuple(
                    (index, self.solid_faces[occurrence.solid])
                    for index, occurrence in enumerate(self.occurrences)
                )
                if faces is not None and self.occurrences
                else self.boundary_instances
            )
        else:
            index = self._path_index(path)
            instances = ((index, self.solid_faces[self.occurrences[index].solid]),)
        if budget is not None and isinstance(points_, Tracer):
            raise ValueError(
                "A mutable host query budget cannot be consumed by traced execution."
            )
        if not isinstance(points_, Tracer) and budget is None:
            budget = BRepQueryBudget(
                self.policy.maximum_operations,
                self.policy.maximum_points,
                self.policy.maximum_scratch_bytes,
            )
        hits = []
        for occurrence, allowed in instances:
            selected = tuple(face for face in allowed if face in faces_)
            if not selected:
                continue
            if self.world_query is not None:
                hit = self._world_closest(points_, selected, occurrence, budget=budget)
                signs = jnp.ones((len(self.faces),), dtype=jnp.float64)
                if occurrence != -1:
                    solid = self.occurrences[occurrence].solid
                    signs = signs.at[jnp.asarray(self.solid_faces[solid])].set(
                        jnp.asarray(self.solid_face_signs[solid], dtype=jnp.float64)
                    )
                hit = eqx.tree_at(
                    lambda value: value.normals, hit, hit.normals * signs[hit.faces, None]
                )
                hits.append(hit)
                continue
            rotation, translation = self._placement(occurrence)
            preparation = (
                self._prepare_placed_query(occurrence, selected)
                if self.certify_on_host and not isinstance(points_, Tracer)
                else None
            )
            local = (points_ - translation) @ rotation if preparation is None else points_
            hit = self._closest_definition(
                local, selected, budget=budget, affine_preparation=preparation
            )
            if preparation is not None:
                rotation, translation = (
                    jnp.eye(3, dtype=jnp.float64),
                    jnp.zeros((3,), dtype=jnp.float64),
                )
            signs = jnp.ones((len(self.faces),), dtype=jnp.float64)
            if occurrence != -1:
                solid = self.occurrences[occurrence].solid
                signs = signs.at[jnp.asarray(self.solid_faces[solid])].set(
                    jnp.asarray(self.solid_face_signs[solid], dtype=jnp.float64)
                )
            hit = eqx.tree_at(
                lambda value: (
                    value.points,
                    value.normals,
                    value.first_derivatives,
                    value.second_derivatives,
                    value.occurrences,
                ),
                hit,
                (
                    hit.points @ rotation.T + translation,
                    (hit.normals * signs[hit.faces, None]) @ rotation.T,
                    hit.first_derivatives @ rotation.T,
                    hit.second_derivatives @ rotation.T,
                    jnp.full(hit.distances.shape, occurrence, dtype=jnp.int32),
                ),
            )
            hits.append(hit)
        if not hits:
            raise ValueError("The selected occurrence has no requested boundary faces.")
        distances = jnp.stack([hit.distances for hit in hits])
        best = jnp.argmin(distances, axis=0)
        rows = jnp.arange(points_.shape[0])
        result = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values)[best, rows], *hits
        )
        lower = jnp.min(jnp.stack([hit.distance_lower_bounds for hit in hits]), axis=0)
        unresolved = result.unresolved | (
            result.distances - lower > self.tolerances.distance
        )
        different = (
            jnp.linalg.norm(
                jnp.stack([hit.points for hit in hits]) - result.points[None], axis=-1
            )
            > self.tolerances.distance
        )
        tied = jnp.any(
            different & (distances <= result.distances[None] + self.tolerances.distance),
            axis=0,
        )
        return eqx.tree_at(
            lambda value: (
                value.distance_lower_bounds,
                value.unresolved,
                value.status,
                value.query_operations,
                value.refinements,
            ),
            result,
            (
                lower,
                unresolved,
                jnp.where(
                    unresolved, _FAILED, jnp.where(tied, _AMBIGUOUS, result.status)
                ).astype(jnp.int8),
                jnp.sum(jnp.stack([hit.query_operations for hit in hits]), axis=0),
                jnp.sum(jnp.stack([hit.refinements for hit in hits]), axis=0),
            ),
        )

    def _definition_winding(self, points: ArrayLike, /) -> Array:
        """Definition-space diagnostic winding; never a containment certificate."""
        points_ = _points(points)
        nodes, solids = self.quadrature.points.shape[0], len(self.solid_faces)
        source_bytes = (nodes + self.policy.chunk_size) * (48 + 8 * solids) + 8 * len(
            self.faces
        ) * solids
        available = self.policy.maximum_scratch_bytes - source_bytes
        if available < 64:
            raise BRepQueryResourceError(
                "scratch_bytes", source_bytes + 64, self.policy.maximum_scratch_bytes
            )
        width = min(self.policy.chunk_size, nodes, available // 64)
        chunk = min(self.policy.chunk_size, points_.shape[0], available // (64 * width))
        BRepQueryBudget(
            self.policy.maximum_operations,
            self.policy.maximum_points,
            self.policy.maximum_scratch_bytes,
        ).admit(
            points_.shape[0] * nodes, points_.shape[0], source_bytes + 64 * width * chunk
        )
        return _compiled_definition_winding(self, points_, chunk, width)

    def _winding_chunks(self, points_: Array, chunk_size: int, width: int, /) -> Array:
        quadrature = self.quadrature
        membership = jnp.asarray(
            [
                [
                    signs[faces.index(face)] if face in faces else 0
                    for faces, signs in zip(
                        self.solid_faces, self.solid_face_signs, strict=True
                    )
                ]
                for face in range(len(self.faces))
            ],
            dtype=jnp.float64,
        ).reshape(len(self.faces), -1)

        width = min(width, quadrature.points.shape[0])
        blocks = -(-quadrature.points.shape[0] // width)
        padding = blocks * width - quadrature.points.shape[0]
        nodes = jnp.pad(quadrature.points, ((0, padding), (0, 0))).reshape(
            blocks, width, 3
        )
        vectors = jnp.pad(quadrature.vector_areas, ((0, padding), (0, 0))).reshape(
            blocks, width, 3
        )
        weights = jnp.pad(membership[quadrature.faces], ((0, padding), (0, 0))).reshape(
            blocks, width, -1
        )

        def chunk(values: Array) -> Array:
            def accumulate(
                total: Array, data: tuple[Array, Array, Array]
            ) -> tuple[Array, None]:
                positions, areas, signs = data
                difference = positions[None] - values[:, None, :]
                distance = jnp.linalg.norm(difference, axis=-1)
                denominator = jnp.where(
                    jnp.any(signs != 0, axis=-1)[None], distance**3, 1.0
                )
                flux = jnp.sum(difference * areas[None], -1) / denominator
                return total + flux @ signs / (4.0 * pi), None

            return jax.lax.scan(
                accumulate,
                jnp.zeros((values.shape[0], len(self.solid_faces)), dtype=jnp.float64),
                (nodes, vectors, weights),
            )[0]

        return _chunked(chunk, points_, chunk_size)

    def winding_numbers(self, points: ArrayLike, /) -> Array:
        """Diagnostic world-space winding numbers ``(points, occurrences)``."""
        points_ = _points(points)
        values = []
        for index, occurrence in enumerate(self.occurrences):
            rotation, translation = self._placement(index)
            values.append(
                self._definition_winding((points_ - translation) @ rotation)[
                    :, occurrence.solid
                ]
            )
        return jnp.stack(values, axis=1)

    def _contains_definition(
        self,
        points: Array,
        solid: int,
        /,
        *,
        budget: BRepQueryBudget | None = None,
    ) -> BRepContainmentResult:
        if self.certify_on_host and not isinstance(points, Tracer):
            if budget is None:
                budget = BRepQueryBudget(
                    self.policy.maximum_operations,
                    self.policy.maximum_points,
                    self.policy.maximum_scratch_bytes,
                )
                self._admit_containment(points.shape[0], (solid,), budget)
            return self._contains_host(np.asarray(points), solid, budget=budget)
        closest = self._closest_definition(points, self.solid_faces[solid])
        signs = jnp.asarray(self.solid_face_signs[solid], dtype=jnp.float64)
        face_signs = (
            jnp.ones((len(self.faces),), dtype=jnp.float64)
            .at[jnp.asarray(self.solid_faces[solid])]
            .set(signs)
        )
        normal = closest.normals * face_signs[closest.faces, None]
        delta = points - closest.points
        length = jnp.linalg.norm(delta, axis=-1)
        aligned = (
            jnp.all(jnp.isfinite(normal), axis=-1)
            & ~closest.unresolved
            & (
                jnp.linalg.norm(jnp.cross(delta, jnp.nan_to_num(normal)), axis=-1)
                <= 1.0e-6 * length
            )
        )
        boundary = closest.distances <= self.tolerances.classifier
        boxes = self.face_boxes[jnp.asarray(self.solid_faces[solid], dtype=jnp.int32)]
        outside = jnp.any(
            (points < jnp.min(boxes[:, 0], axis=0))
            | (points > jnp.max(boxes[:, 1], axis=0)),
            axis=-1,
        )
        unresolved = ~(aligned | outside) | boundary
        inside = boundary | (
            ~outside & aligned & (jnp.sum(delta * jnp.nan_to_num(normal), axis=-1) < 0.0)
        )
        return BRepContainmentResult(
            inside=inside,
            status=jnp.where(unresolved, _AMBIGUOUS, _UNIQUE).astype(jnp.int8),
            winding=jnp.full(inside.shape, jnp.nan, dtype=jnp.float64),
            distances=closest.distances,
            unresolved=unresolved,
            distance_lower_bounds=closest.distance_lower_bounds,
            distance_upper_bounds=closest.distances,
            query_operations=closest.query_operations,
            resource_exhausted=jnp.zeros(inside.shape, dtype=jnp.bool_),
        )

    def _contains_host(
        self,
        points: np.ndarray,
        solid: int,
        /,
        *,
        budget: BRepQueryBudget | None,
        affine_preparation: tuple[tuple[AffineFace, ...], QueryBroadphase, np.ndarray]
        | None = None,
    ) -> BRepContainmentResult:
        from ._affine_query import contains_affine, prepare_affine_faces
        from ._containment import certify_solid_containment
        from ._sphere_membership import _square_root_interval, classify_full_sphere

        faces = self.solid_faces[solid]
        prepared = (
            prepare_affine_faces(self.model, faces)
            if affine_preparation is None
            else affine_preparation[0]
        )
        broadphase = (
            QueryBroadphase.prepare(self.face_bvh)
            if affine_preparation is None
            else affine_preparation[1]
        )
        if budget is not None:
            budget.consume(0, points=len(points))
        inside = np.zeros((len(points),), dtype=np.bool_)
        unresolved = np.ones((len(points),), dtype=np.bool_)
        lower = np.zeros((len(points),), dtype=np.float64)
        upper = np.full((len(points),), np.inf, dtype=np.float64)
        query_operations = np.zeros((len(points),), dtype=np.int64)
        exhausted = np.zeros((len(points),), dtype=np.bool_)
        if prepared is not None:
            closest = self._affine_closest(
                points,
                faces,
                budget=budget,
                prepared=prepared,
                broadphase=broadphase,
                source_boxes=None
                if affine_preparation is None
                else affine_preparation[2],
            )
            if closest is None:
                raise RuntimeError(
                    "An admitted affine source lost its constrained query theorem."
                )
            lower, upper = (
                np.asarray(closest.distance_lower_bounds),
                np.asarray(closest.distances),
            )
            query_operations[:] = np.asarray(closest.query_operations)
            for row, point in enumerate(points):
                if budget is not None:
                    budget.consume(1)
                if upper[row] <= self.tolerances.classifier:
                    inside[row] = True
                    continue
                decision, complete, operations = contains_affine(
                    point, prepared, broadphase, faces
                )
                if budget is not None:
                    budget.consume(operations)
                query_operations[row] += operations
                inside[row], unresolved[row] = (
                    decision,
                    not complete or lower[row] <= self.tolerances.classifier,
                )
        else:
            face = self.faces[faces[0]]
            sphere_source = _solid_sphere_source(self.model, faces, face)
            revolution_source = (
                _solid_revolution_source(self.model, faces, face)
                if sphere_source is None
                else None
            )
            boxes = np.asarray(self.face_boxes)[list(faces)]
            for row, point in enumerate(points):
                query_operations[row] = 1 + len(faces)
                if budget is not None:
                    budget.consume(1 + len(faces))
                if sphere_source is not None:
                    decision = classify_full_sphere(sphere_source, point)
                    lower[row], upper[row] = (
                        decision.distance_lower,
                        decision.distance_upper,
                    )
                    inside[row] = (
                        decision.inside or upper[row] <= self.tolerances.classifier
                    )
                    unresolved[row] = lower[row] <= self.tolerances.classifier
                    continue
                if revolution_source is not None:
                    remaining = (
                        self.policy.maximum_subdivisions
                        if budget is None
                        else min(
                            self.policy.maximum_subdivisions,
                            budget.maximum_operations - budget.operations,
                        )
                    )
                    if remaining < 1:
                        exhausted[row] = True
                        continue
                    membership = classify_full_revolution(
                        revolution_source,
                        point,
                        tolerance=self.tolerances.classifier,
                        maximum_rows=remaining,
                    )
                    if budget is not None:
                        budget.consume(
                            membership.operations, subdivisions=membership.operations
                        )
                    query_operations[row] += membership.operations
                    lower[row], upper[row] = (
                        membership.distance_lower,
                        membership.distance_upper,
                    )
                    inside[row] = (
                        membership.inside or upper[row] <= self.tolerances.classifier
                    )
                    unresolved[row] = (
                        not membership.complete
                        or lower[row] <= self.tolerances.classifier
                    )
                    exhausted[row] = membership.exhausted
                    continue
                bounds = []
                for box in boxes:
                    delta = tuple(Fraction(float(value)) for value in point)
                    first = tuple(Fraction(float(value)) for value in box[0])
                    last = tuple(Fraction(float(value)) for value in box[1])
                    gap = tuple(
                        max(Fraction(0), a - q, q - b)
                        for a, b, q in zip(first, last, delta, strict=True)
                    )
                    far = tuple(
                        max(abs(a - q), abs(b - q))
                        for a, b, q in zip(first, last, delta, strict=True)
                    )
                    bounds.append(
                        (
                            _square_root_interval(
                                sum((value * value for value in gap), Fraction(0))
                            )[0],
                            _square_root_interval(
                                sum((value * value for value in far), Fraction(0))
                            )[1],
                        )
                    )
                box_lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
                lower[row], upper[row] = (
                    min(bound[0] for bound in bounds),
                    min(bound[1] for bound in bounds),
                )
                near = tuple(
                    faces[index]
                    for index in np.flatnonzero(box_lower <= self.tolerances.classifier)
                )
                if near:
                    remaining = (
                        self.policy.maximum_subdivisions
                        if budget is None
                        else min(
                            self.policy.maximum_subdivisions,
                            budget.maximum_operations - budget.operations,
                        )
                    )
                    if remaining < 1:
                        exhausted[row] = True
                        continue
                    closest = self._closest_definition(
                        jnp.asarray(point[None]), near, maximum_boxes=remaining
                    )
                    if budget is not None:
                        refinements = int(np.sum(np.asarray(closest.refinements)))
                        budget.consume(
                            int(closest.query_operations[0]), subdivisions=refinements
                        )
                    query_operations[row] += int(closest.query_operations[0])
                    far_lower = min(
                        (
                            value
                            for index, value in enumerate(box_lower)
                            if faces[index] not in near
                        ),
                        default=np.inf,
                    )
                    lower[row], upper[row] = (
                        min(float(closest.distance_lower_bounds[0]), far_lower),
                        float(closest.distances[0]),
                    )
                    if upper[row] <= self.tolerances.classifier:
                        inside[row] = True
                        continue
                    if lower[row] <= self.tolerances.classifier:
                        exhausted[row] = (
                            self.policy.maximum_depth == 0
                            or self.policy.maximum_subdivisions == 0
                            or int(closest.refinements[0]) >= remaining
                        )
                        continue
                remaining = (
                    self.policy.maximum_subdivisions
                    if budget is None
                    else min(
                        self.policy.maximum_subdivisions,
                        budget.maximum_operations - budget.operations,
                    )
                )
                if remaining < 1:
                    exhausted[row] = True
                    continue
                certificate = certify_solid_containment(
                    self.model,
                    point,
                    solid,
                    maximum_boxes=remaining,
                    maximum_depth=self.policy.maximum_depth,
                    maximum_operations=None
                    if budget is None
                    else budget.maximum_operations - budget.operations,
                    source_boxes=boxes,
                    trim_domains=tuple(self.faces[face].trim_domain for face in faces),
                    broadphase=broadphase,
                )
                if budget is not None:
                    budget.consume(
                        certificate.owner_operations,
                        subdivisions=certificate.boxes_processed,
                    )
                query_operations[row] += certificate.owner_operations
                exhausted[row] = certificate.resource_exhausted
                inside[row], unresolved[row] = (
                    certificate.inside,
                    not certificate.complete,
                )
        return BRepContainmentResult(
            inside=jnp.asarray(inside),
            status=jnp.asarray(
                np.where(exhausted, _FAILED, np.where(unresolved, _AMBIGUOUS, _UNIQUE)),
                dtype=jnp.int8,
            ),
            winding=jnp.full(inside.shape, jnp.nan, dtype=jnp.float64),
            distances=jnp.asarray(upper),
            unresolved=jnp.asarray(unresolved),
            distance_lower_bounds=jnp.asarray(lower),
            distance_upper_bounds=jnp.asarray(upper),
            query_operations=jnp.asarray(query_operations),
            resource_exhausted=jnp.asarray(exhausted),
        )

    def _query_chunk_size(
        self,
        budget: BRepQueryBudget | None,
        source_bytes: int,
        /,
        *,
        per_point_bytes: int = 4096,
    ) -> int:
        available = self.policy.maximum_scratch_bytes
        if budget is not None and budget.maximum_scratch_bytes is not None:
            available = min(available, budget.maximum_scratch_bytes)
        if available < source_bytes + per_point_bytes:
            raise BRepQueryResourceError(
                "scratch_bytes", source_bytes + per_point_bytes, available
            )
        return min(self.policy.chunk_size, (available - source_bytes) // per_point_bytes)

    def _admit_containment(
        self, count: int, solids: tuple[int, ...], budget: BRepQueryBudget, /
    ) -> None:
        from ._affine_query import prepare_affine_faces

        reservation, scratch = 0, 0
        for solid in solids:
            faces = self.solid_faces[solid]
            edges = sum(len(self.faces[face].coedge_edges) for face in faces)
            affine = prepare_affine_faces(self.model, faces)
            source_bytes = 4096 * (self.face_bvh.node_count + len(faces) + edges)
            chunk = self._query_chunk_size(budget, source_bytes)
            reservation += count * (
                1 + 8 * self.face_bvh.node_count + 8 * len(faces) + 5 * edges
                if affine is not None
                else 1 + 2 * self.policy.maximum_subdivisions + 2 * len(faces) + edges
            )
            scratch = max(scratch, source_bytes + 4096 * min(count, chunk))
        BRepQueryBudget(
            self.policy.maximum_operations,
            self.policy.maximum_points,
            self.policy.maximum_scratch_bytes,
        ).admit(reservation, count * len(solids), scratch)
        budget.admit(reservation, count * len(solids), scratch)

    def contains(
        self,
        points: ArrayLike,
        /,
        *,
        solid: int = 0,
        path: tuple[str, ...] | None = None,
        budget: BRepQueryBudget | None = None,
    ) -> BRepContainmentResult:
        """Membership of an occurrence path, or the union of occurrences of ``solid``."""
        if path is None:
            if not 0 <= solid < len(self.solid_faces):
                raise ValueError("solid must index an existing solid.")
            selected = tuple(
                index
                for index, occurrence in enumerate(self.occurrences)
                if occurrence.solid == solid
            )
        else:
            selected = (self._path_index(path),)
        if not selected:
            raise ValueError("The requested solid has no assembly occurrences.")
        points_ = _points(points)
        if budget is not None and isinstance(points_, Tracer):
            raise ValueError(
                "A mutable host query budget cannot be consumed by traced execution."
            )
        if not isinstance(points_, Tracer):
            if budget is None:
                budget = BRepQueryBudget(
                    self.policy.maximum_operations,
                    self.policy.maximum_points,
                    self.policy.maximum_scratch_bytes,
                )
            owner = self if self.world_query is None else self.world_query
            solids = (
                tuple(self.occurrences[index].solid for index in selected)
                if self.world_query is None
                else tuple(self.world_solids[index] for index in selected)
            )
            owner._admit_containment(points_.shape[0], solids, budget)
        results = []
        for index in selected:
            if self.world_query is not None:
                results.append(
                    self.world_query._contains_definition(
                        points_,
                        self.world_solids[index],
                        budget=budget,
                    )
                )
                continue
            occurrence = self.occurrences[index]
            rotation, translation = self._placement(index)
            preparation = (
                self._prepare_placed_query(index, self.solid_faces[occurrence.solid])
                if self.certify_on_host and not isinstance(points_, Tracer)
                else None
            )
            if preparation is not None:
                results.append(
                    self._contains_host(
                        np.asarray(points_),
                        occurrence.solid,
                        budget=budget,
                        affine_preparation=preparation,
                    )
                )
            else:
                local = (points_ - translation) @ rotation
                results.append(
                    self._contains_definition(local, occurrence.solid, budget=budget)
                )
        inside = jnp.stack([result.inside for result in results])
        unresolved = jnp.stack([result.unresolved for result in results])
        certified_inside = jnp.any(inside & ~unresolved, axis=0)
        pending = jnp.any(unresolved, axis=0) & ~certified_inside
        exhausted = (
            jnp.any(jnp.stack([result.resource_exhausted for result in results]), axis=0)
            & pending
        )
        return BRepContainmentResult(
            inside=jnp.any(inside, axis=0),
            status=jnp.where(
                exhausted, _FAILED, jnp.where(pending, _AMBIGUOUS, _UNIQUE)
            ).astype(jnp.int8),
            winding=jnp.sum(jnp.stack([result.winding for result in results]), axis=0),
            distances=jnp.min(
                jnp.stack([result.distances for result in results]), axis=0
            ),
            unresolved=pending,
            distance_lower_bounds=jnp.min(
                jnp.stack([result.distance_lower_bounds for result in results]), axis=0
            ),
            distance_upper_bounds=jnp.min(
                jnp.stack([result.distance_upper_bounds for result in results]), axis=0
            ),
            query_operations=jnp.sum(
                jnp.stack([result.query_operations for result in results]), axis=0
            ),
            resource_exhausted=exhausted,
        )


@eqx.filter_jit
def _compiled_closest_chunks(
    query: PreparedBRepQuery,
    points: Array,
    faces: tuple[int, ...],
    chunk_size: int,
) -> BRepSurfaceQueryResult:
    return _chunked(
        lambda values: query._closest_chunk(values, faces), points, chunk_size
    )


@eqx.filter_jit
def _compiled_definition_winding(
    query: PreparedBRepQuery,
    points: Array,
    chunk_size: int,
    width: int,
    /,
) -> Array:
    return query._winding_chunks(points, chunk_size, width)


def _points(points: ArrayLike, /) -> Array:
    values = jnp.asarray(points, dtype=jnp.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("Query points must have shape (n > 0, 3).")
    if not isinstance(values, Tracer) and not np.all(np.isfinite(np.asarray(values))):
        raise ValueError("Query points must be finite.")
    return values


def _trimmed_face_rules(
    model: BRepModel, policy: BRepQueryPolicy, /
) -> tuple[tuple[Array, Array], ...]:
    """Per-face parameter nodes and weights integrating over the exact trims."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Trimmed face rules require the model's exact geometry.")
    bounds = np.asarray(model.parameter_bounds)
    rules = []
    budget = BRepQueryBudget(
        policy.maximum_operations, policy.maximum_points, policy.maximum_scratch_bytes
    )
    for face in range(len(model.patches)):
        parameters, weights = _green_nodes(
            geometry,
            face,
            bounds[face],
            policy.quadrature_order,
            policy.quadrature_subdivisions,
            _surface_quadrature_u_breaks(model.patches[face], bounds[face], budget),
            budget,
        )
        rules.append((jnp.asarray(parameters), jnp.asarray(weights)))
    return tuple(rules)


def _prepare_quadrature(
    model: BRepModel, geometry: BRepGeometry, policy: BRepQueryPolicy, /
) -> tuple[_Quadrature, BRepMeasureResult]:
    orientation = np.asarray(model.orientation)
    bounds = np.asarray(model.parameter_bounds)
    fine, coarse = [], []
    budget = BRepQueryBudget(
        policy.maximum_operations, policy.maximum_points, policy.maximum_scratch_bytes
    )
    u_breaks = tuple(
        _surface_quadrature_u_breaks(patch, bounds[face], budget)
        for face, patch in enumerate(model.patches)
    )
    for subdivisions, store in (
        (policy.quadrature_subdivisions, fine),
        (policy.quadrature_subdivisions // 2, coarse),
    ):
        for face, patch in enumerate(model.patches):
            parameters, weights = _green_nodes(
                geometry,
                face,
                bounds[face],
                policy.quadrature_order,
                subdivisions,
                u_breaks[face],
                budget,
            )
            points, vector_areas, areas = _face_quadrature(
                patch, float(orientation[face]), parameters, weights
            )
            store.append((parameters, weights, points, vector_areas, areas))

    def measure(store: list, /) -> tuple[np.ndarray, np.ndarray]:
        face_areas = np.asarray([np.sum(entry[4]) for entry in store])
        face_volumes = np.asarray([np.sum(entry[2] * entry[3]) / 3.0 for entry in store])
        solid_volumes = np.asarray(
            [
                sum(
                    sign * face_volumes[face]
                    for face, sign in zip(faces, signs, strict=True)
                )
                for faces, signs in zip(
                    model.topology.solid_faces,
                    model.topology.solid_face_orientations,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        return face_areas, solid_volumes

    fine_areas, fine_volumes = measure(fine)
    coarse_areas, coarse_volumes = measure(coarse)
    quadrature = _Quadrature(
        points=jnp.asarray(np.concatenate([entry[2] for entry in fine])),
        vector_areas=jnp.asarray(np.concatenate([entry[3] for entry in fine])),
        areas=jnp.asarray(np.concatenate([entry[4] for entry in fine])),
        parameters=jnp.asarray(np.concatenate([entry[0] for entry in fine])),
        weights=jnp.asarray(np.concatenate([entry[1] for entry in fine])),
        faces=jnp.asarray(
            np.concatenate(
                [np.full(entry[0].shape[0], face) for face, entry in enumerate(fine)]
            ).astype(np.int32)
        ),
    )
    return quadrature, BRepMeasureResult(
        face_areas=jnp.asarray(fine_areas),
        face_area_errors=jnp.asarray(np.abs(fine_areas - coarse_areas)),
        solid_volumes=jnp.asarray(fine_volumes),
        solid_volume_errors=jnp.asarray(np.abs(fine_volumes - coarse_volumes)),
    )


def prepare_brep_query(
    model: BRepModel,
    /,
    *,
    policy: BRepQueryPolicy | None = None,
    projection_policy: BRepProjectionPolicy | None = None,
) -> PreparedBRepQuery:
    """Prepare native queries over the exact geometry of ``model``."""
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    policy_ = BRepQueryPolicy() if policy is None else policy
    projection_ = (
        BRepProjectionPolicy() if projection_policy is None else projection_policy
    )
    if not isinstance(policy_, BRepQueryPolicy):
        raise TypeError("policy must be a BRepQueryPolicy or None.")
    if not isinstance(projection_, BRepProjectionPolicy):
        raise TypeError("projection_policy must be a BRepProjectionPolicy or None.")
    return PreparedBRepQuery(model, policy_, projection_)


# ------------------------------------------------------------ projection


def _group_size(count: int, /) -> int:
    """Power-of-two padding bounds recompilation across batch sizes."""
    return 1 << max(0, (count - 1).bit_length())


@dataclass(frozen=True, slots=True)
class _QualifiedProjection:
    paths: tuple[tuple[str, ...], ...]
    entity_paths: tuple[tuple[int, ...], ...]
    placements: tuple[tuple[int, ...], ...]
    codes: np.ndarray
    boxes: np.ndarray
    closure: np.ndarray
    world_boxes: np.ndarray


def _qualified_projection(
    geometry: BRepGeometry, query: PreparedBRepQuery, definition_boxes: np.ndarray, /
) -> _QualifiedProjection:
    records = geometry.qualified_entity_incidence(query.source_revision)
    identities = {record.container for record in records} | {
        record.member for record in records
    }
    paths = tuple(
        sorted(
            {
                identity.occurrence_path
                for identity in identities
                if identity.occurrence_path
            }
        )
    )
    path_indices = {path: index for index, path in enumerate(paths)}
    counts = (
        geometry.vertex_points.shape[0],
        len(geometry.edge_curves),
        len(geometry.face_loops),
        len(geometry.solid_shells),
    )
    offsets = np.cumsum((0,) + counts)
    total = sum(counts)
    qualified_total = total * (len(paths) + 1)

    def code(identity: BRepEntityId) -> int:
        dimension = BRepEntityDimension[identity.kind.upper()].value
        namespace = (
            path_indices[identity.occurrence_path] + 1 if identity.occurrence_path else 0
        )
        return int(offsets[dimension]) + identity.index + namespace * total

    adjacency: dict[int, set[int]] = {}
    for record in records:
        adjacency.setdefault(code(record.container), set()).add(code(record.member))
    # Definition closure and occurrence-qualified closure are distinct
    # namespaces. The authored occurrence graph need not repeat unplaced
    # definitions, but public definition queries still own their exact topology.
    definition_closure = _closure_codes(
        geometry.topology(),
        np.asarray(geometry.edge_vertices, dtype=np.int64).reshape(-1, 2),
        counts,
    )
    codes = sorted({code(identity) for identity in identities})
    reach = {divmod(int(relation), total) for relation in definition_closure}
    for container in codes:
        pending, seen = [container], set()
        while pending:
            member = pending.pop()
            if member in seen:
                continue
            seen.add(member)
            reach.add((container, member))
            pending.extend(adjacency.get(member, ()))
    solid_codes = tuple(
        code(
            BRepEntityId(
                query.source_revision, "solid", occurrence.solid, occurrence.path
            )
        )
        for occurrence in geometry.occurrences
    )
    placements, boxes = [], []
    entity_paths: list[list[int]] = [[] for _ in range(total)]
    world_boxes = np.full(definition_boxes.shape, np.inf, dtype=np.float64)
    world_boxes[:, 1] = -np.inf
    for qualified in codes:
        base, namespace = qualified % total, qualified // total
        indices = (
            (-1,)
            if namespace == 0
            else tuple(
                index
                for index, solid in enumerate(solid_codes)
                if (solid, qualified) in reach
            )
        )
        if not indices:
            raise ValueError(
                "The authored qualified entity graph has no physical placement."
            )
        placed = []
        for index in indices:
            rotation, translation = query._placement(index)
            products = np.asarray(rotation) * definition_boxes[base, :, None, :]
            box = np.stack(
                (
                    np.sum(np.min(products, axis=0), axis=-1),
                    np.sum(np.max(products, axis=0), axis=-1),
                )
            ) + np.asarray(translation)
            margin = (
                256.0
                * np.finfo(np.float64).eps
                * max(
                    1.0,
                    float(np.max(np.abs(definition_boxes[base]))),
                    float(np.max(np.abs(box))),
                )
            )
            placed.append(np.stack((box[0] - margin, box[1] + margin)))
        placed_boxes = np.stack(placed)
        box = np.stack(
            (np.min(placed_boxes[:, 0], axis=0), np.max(placed_boxes[:, 1], axis=0))
        )
        boxes.append(box)
        placements.append(indices)
        entity_paths[base].append(namespace - 1)
        world_boxes[base, 0] = np.minimum(world_boxes[base, 0], box[0])
        world_boxes[base, 1] = np.maximum(world_boxes[base, 1], box[1])
    return _QualifiedProjection(
        paths,
        tuple(tuple(indices) for indices in entity_paths),
        tuple(placements),
        np.asarray(codes, dtype=np.int64),
        np.stack(boxes),
        np.asarray(
            sorted(container * qualified_total + member for container, member in reach),
            dtype=np.int64,
        ),
        world_boxes,
    )


@final
class NativeBRepProjection(AbstractBRepProjection):
    """Native closest-point, classification and solid location of one revision.

    Construct with :func:`prepare_brep_projection` from a model that carries
    exact native geometry. Entity indices are the model's ``BRepEntityId``
    indices; a planar ``embedding`` binds face-only planar revisions to
    two-dimensional coordinates.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    policy: BRepProjectionPolicy
    embedding: PlanarEmbedding | None = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    entity_counts: tuple[int, int, int, int] = eqx.field(static=True)
    closure_codes: Array
    entity_boxes: Array
    edge_degenerate: Array
    edge_vertices: Array
    edge_closed: Array
    vertex_points: Array
    query: PreparedBRepQuery
    projection_id: str = eqx.field(static=True)
    entity_occurrences: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    qualified_entity_codes: Array
    qualified_entity_boxes: Array
    qualified_closure_codes: Array
    qualified_placements: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        policy: BRepProjectionPolicy,
        embedding: PlanarEmbedding | None,
        query_policy: BRepQueryPolicy,
        /,
    ) -> None:
        geometry = model.geometry
        if geometry is None:
            raise ValueError("A native projection requires the model's exact geometry.")
        query = PreparedBRepQuery(model, query_policy, policy)
        topology = model.topology
        points = np.asarray(geometry.vertex_points)
        edge_vertices = np.asarray(geometry.edge_vertices, dtype=np.int64).reshape(-1, 2)
        counts = (
            topology.num_vertices,
            topology.num_edges,
            topology.num_faces,
            topology.num_solids,
        )
        face_boxes = np.asarray(query.face_boxes)
        solid_boxes = np.asarray(
            [
                np.stack(
                    (
                        np.min(face_boxes[list(faces), 0], axis=0),
                        np.max(face_boxes[list(faces), 1], axis=0),
                    )
                )
                for faces in topology.solid_faces
            ],
            dtype=np.float64,
        ).reshape(-1, 2, 3)
        ambient = 3
        if embedding is not None:
            if not isinstance(embedding, PlanarEmbedding):
                raise TypeError("embedding must be PlanarEmbedding or None.")
            if topology.num_solids:
                raise ValueError("A planar embedding binds face-only B-Rep revisions.")
            if (
                np.max(np.abs(embedding.plane_residual(points)))
                > policy.classifier_tolerance
            ):
                raise ValueError(
                    "The B-Rep revision does not lie on its planar embedding."
                )
            ambient = 2
        self.source_id = model.source_id
        self.source_revision = model.source_revision
        self.model_id = model.model_id
        self.policy = policy
        self.embedding = embedding
        self.ambient_dimension = ambient
        self.entity_counts = counts
        self.closure_codes = jnp.asarray(_closure_codes(topology, edge_vertices, counts))
        definition_boxes = np.concatenate(
            (
                np.stack((points, points), axis=1),
                np.asarray(query.edge_boxes),
                face_boxes,
                solid_boxes,
            )
        )
        qualified = _qualified_projection(geometry, query, definition_boxes)
        self.entity_boxes = jnp.asarray(qualified.world_boxes, dtype=jnp.float64)
        self.entity_occurrences = qualified.entity_paths
        self.occurrence_paths = qualified.paths
        self.qualified_entity_codes = jnp.asarray(qualified.codes, dtype=jnp.int64)
        self.qualified_entity_boxes = jnp.asarray(qualified.boxes, dtype=jnp.float64)
        self.qualified_closure_codes = jnp.asarray(qualified.closure, dtype=jnp.int64)
        self.qualified_placements = qualified.placements
        self.edge_degenerate = jnp.asarray(
            np.asarray(geometry.degenerate_edges, dtype=np.bool_)
        )
        self.edge_vertices = jnp.asarray(edge_vertices.astype(np.int32))
        self.edge_closed = jnp.asarray(
            np.asarray([edge.closed for edge in query.edges], dtype=np.bool_)
        )
        self.vertex_points = jnp.asarray(points)
        self.query = query
        self.projection_id = canonical_fingerprint(
            {
                "kind": "native-brep-projection",
                "source_revision": model.source_revision,
                "model_id": model.model_id,
                "policy": policy.policy_id,
                "query": query.query_id,
                "embedding": None if embedding is None else embedding.embedding_id,
            }
        )

    def _entity_targets(
        self, dimension: int, entity: int, namespace: int | None, /
    ) -> tuple[tuple[int, int], ...]:
        base = sum(self.entity_counts[:dimension]) + entity
        paths = self.entity_occurrences[base] if namespace is None else (namespace,)
        published = np.asarray(self.qualified_entity_codes)
        targets = []
        for path in paths:
            code = base + (path + 1) * sum(self.entity_counts)
            position = int(np.searchsorted(published, code))
            if position >= published.size or published[position] != code:
                raise ValueError(
                    "The entity has no authored identity at the requested occurrence path."
                )
            targets.extend(
                (path, placement) for placement in self.qualified_placements[position]
            )
        return tuple(targets)

    def _placed_lower_entity(
        self, points: Array, dimension: int, entity: int, /, occurrence: int | None = None
    ) -> dict[str, Array]:
        occurrences = self._entity_targets(dimension, entity, occurrence)
        if not occurrences:
            raise ValueError("The requested entity has no placed occurrences.")
        hits = []
        for namespace, index in occurrences:
            world = self.query.world_query
            if world is not None:
                from ._stationary_isolation import source_distance_bounds

                geometry = world.model.geometry
                if geometry is None:
                    raise RuntimeError(
                        "A world stratum query lost its authoritative source."
                    )
                path = () if index < 0 else self.query.occurrences[index].path
                kind = "vertex" if dimension == BRepEntityDimension.VERTEX else "edge"
                sources = (
                    self.query.world_vertex_sources
                    if kind == "vertex"
                    else self.query.world_edge_sources
                )
                candidates = {
                    record.member.index
                    for record in geometry.qualified_entity_incidence(
                        world.source_revision
                    )
                    if record.member.kind == kind
                    and record.member.occurrence_path == path
                    and sources[record.member.index] == entity
                }
                if len(candidates) != 1:
                    raise RuntimeError(
                        "A world stratum query lost unique qualified source incidence."
                    )
                target = candidates.pop()
                if kind == "edge":
                    hit = _certified_edge(
                        world.edges[target],
                        points,
                        world.tolerances,
                        world.policy,
                        endpoint_roots=geometry.edge_endpoint_roots[target],
                    )
                else:
                    projected = jnp.broadcast_to(
                        world.vertex_points[target], points.shape
                    )
                    root = geometry.vertex_roots[target]
                    enclosure = (
                        np.stack((np.asarray(world.vertex_points[target]),) * 2)
                        if root is None
                        else root.point_enclosure()
                    )
                    lower = np.asarray(
                        [
                            source_distance_bounds(enclosure, point)[0]
                            for point in np.asarray(points)
                        ]
                    )
                    distance = jnp.linalg.norm(points - projected, axis=-1)
                    hit = {
                        "point": projected,
                        "distance": distance,
                        "parameter": jnp.full(
                            (points.shape[0],), jnp.nan, dtype=jnp.float64
                        ),
                        "tangent": jnp.full(points.shape, jnp.nan, dtype=jnp.float64),
                        "status": jnp.where(
                            distance - jnp.asarray(lower)
                            <= self.policy.ambiguity_tolerance,
                            _UNIQUE,
                            _FAILED,
                        ).astype(jnp.int8),
                        "lower_bound": jnp.asarray(lower),
                    }
                hits.append(
                    {
                        **hit,
                        "occurrence": jnp.full(
                            (points.shape[0],), namespace, dtype=jnp.int32
                        ),
                    }
                )
                continue
            rotation, translation = self.query._placement(index)
            local = (points - translation) @ rotation
            if dimension == BRepEntityDimension.VERTEX:
                projected = jnp.broadcast_to(self.vertex_points[entity], points.shape)
                hit = {
                    "point": projected,
                    "distance": jnp.linalg.norm(local - projected, axis=-1),
                    "parameter": jnp.full((points.shape[0],), jnp.nan, dtype=jnp.float64),
                    "tangent": jnp.full(points.shape, jnp.nan, dtype=jnp.float64),
                    "status": jnp.full((points.shape[0],), _UNIQUE, dtype=jnp.int8),
                    "lower_bound": jnp.linalg.norm(local - projected, axis=-1),
                }
            else:
                geometry = self.query.model.geometry
                if geometry is None:
                    raise RuntimeError(
                        "An exact edge query lost its authoritative source geometry."
                    )
                hit = _certified_edge(
                    self.query.edges[entity],
                    local,
                    self.query.tolerances,
                    self.query.policy,
                    endpoint_roots=geometry.edge_endpoint_roots[entity],
                )
            hits.append(
                {
                    **hit,
                    "point": hit["point"] @ rotation.T + translation,
                    "tangent": hit["tangent"] @ rotation.T,
                    "occurrence": jnp.full(
                        (points.shape[0],), namespace, dtype=jnp.int32
                    ),
                }
            )
        distance = jnp.stack([hit["distance"] for hit in hits])
        best = jnp.argmin(distance, axis=0)
        rows = jnp.arange(points.shape[0])
        result = {
            key: jnp.stack([hit[key] for hit in hits])[best, rows] for key in hits[0]
        }
        lower = jnp.min(jnp.stack([hit["lower_bound"] for hit in hits]), axis=0)
        different = (
            jnp.linalg.norm(
                jnp.stack([hit["point"] for hit in hits]) - result["point"][None], axis=-1
            )
            > self.policy.ambiguity_tolerance
        )
        distinct_occurrence = (
            jnp.stack([hit["occurrence"] for hit in hits]) != result["occurrence"][None]
        )
        tied = jnp.any(
            (different | distinct_occurrence)
            & (distance <= result["distance"][None] + self.policy.ambiguity_tolerance),
            axis=0,
        )
        unresolved = result["distance"] - lower > self.policy.ambiguity_tolerance
        return {
            **result,
            "status": jnp.where(
                unresolved, _FAILED, jnp.where(tied, _AMBIGUOUS, result["status"])
            ).astype(jnp.int8),
        }

    def _placed_face(
        self, points: Array, entity: int, occurrence: int | None, /
    ) -> tuple[BRepSurfaceQueryResult, Array]:
        hits, namespaces = [], []
        for namespace, placement in self._entity_targets(2, entity, occurrence):
            if self.query.world_query is not None:
                result = self.query._world_closest(points, (entity,), placement)
            else:
                rotation, translation = self.query._placement(placement)
                preparation = self.query._prepare_placed_query(placement, (entity,))
                local = (
                    points
                    if preparation is not None
                    else (points - translation) @ rotation
                )
                result = self.query._closest_definition(
                    local, (entity,), affine_preparation=preparation
                )
                if preparation is not None:
                    rotation, translation = (
                        jnp.eye(3, dtype=jnp.float64),
                        jnp.zeros((3,), dtype=jnp.float64),
                    )
                result = eqx.tree_at(
                    lambda value: (
                        value.points,
                        value.normals,
                        value.first_derivatives,
                        value.second_derivatives,
                        value.occurrences,
                    ),
                    result,
                    (
                        result.points @ rotation.T + translation,
                        result.normals @ rotation.T,
                        result.first_derivatives @ rotation.T,
                        result.second_derivatives @ rotation.T,
                        jnp.full(result.distances.shape, placement, dtype=jnp.int32),
                    ),
                )
            hits.append(result)
            namespaces.append(
                jnp.full(result.distances.shape, namespace, dtype=jnp.int32)
            )
        distance = jnp.stack([hit.distances for hit in hits])
        best = jnp.argmin(distance, axis=0)
        rows = jnp.arange(points.shape[0])
        result = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values)[best, rows], *hits
        )
        selected_path = jnp.stack(namespaces)[best, rows]
        lower = jnp.min(jnp.stack([hit.distance_lower_bounds for hit in hits]), axis=0)
        unresolved = result.unresolved | (
            result.distances - lower > self.policy.ambiguity_tolerance
        )
        tied_path = jnp.any(
            (jnp.stack(namespaces) != selected_path[None])
            & (distance <= result.distances[None] + self.policy.ambiguity_tolerance),
            axis=0,
        )
        status = jnp.where(
            unresolved, _FAILED, jnp.where(tied_path, _AMBIGUOUS, result.status)
        ).astype(jnp.int8)
        return eqx.tree_at(
            lambda value: (value.status, value.unresolved, value.distance_lower_bounds),
            result,
            (status, unresolved, lower),
        ), selected_path

    def project(
        self,
        points: ArrayLike,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Closest point of each query on its own vertex, edge, or face."""
        queries = self._queries(points, "Projection queries")
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError("Projection entity dimensions and indices must be integers.")
        if dims.shape != (queries.shape[0],) or rows.shape != dims.shape:
            raise ValueError("One entity dimension and index is required per query.")
        if np.any((dims < 0) | (dims > 2)):
            raise ValueError("Projection targets vertices, edges, or faces only.")
        limits = np.asarray(self.entity_counts[:3])[dims]
        if np.any((rows < 0) | (rows >= limits)):
            raise ValueError("Projection entity indices are out of range.")
        world = self._world(queries)
        count = queries.shape[0]
        closest = np.full((count, 3), np.nan)
        parameters = np.full((count, 2), np.nan)
        residuals = np.full((count,), np.inf)
        status = np.full((count,), _FAILED, dtype=np.int8)
        normals = np.full((count, 3), np.nan)
        first = np.full((count, 3), np.nan)
        second = np.full((count, 3), np.nan)
        occurrences = np.full((count,), -1, dtype=np.int32)
        codes = self.entity_codes(dims, rows, occurrence_paths=occurrence_paths)
        order = np.argsort(codes, kind="stable")
        groups, starts = np.unique(codes[order], return_index=True)
        boundaries = np.append(starts, order.size)
        for code, start, stop in zip(
            groups.tolist(), boundaries[:-1], boundaries[1:], strict=True
        ):
            dimensions_, indices_ = self.code_entities(
                np.asarray((code,), dtype=np.int64)
            )
            dimension, entity = int(dimensions_[0]), int(indices_[0])
            occurrence = (
                None if occurrence_paths is None else code // sum(self.entity_counts) - 1
            )
            selected = order[start:stop]
            size = _group_size(selected.size)
            batch = jnp.asarray(
                np.concatenate(
                    (
                        world[selected],
                        np.broadcast_to(world[selected[:1]], (size - selected.size, 3)),
                    )
                )
            )
            match dimension:
                case BRepEntityDimension.VERTEX:
                    hit = self._placed_lower_entity(batch, dimension, entity, occurrence)
                case BRepEntityDimension.EDGE:
                    hit = self._placed_lower_entity(batch, dimension, entity, occurrence)
                    parameters[selected, 0] = np.asarray(hit["parameter"])[
                        : selected.size
                    ]
                    first[selected] = np.asarray(hit["tangent"])[: selected.size]
                case BRepEntityDimension.FACE:
                    result, selected_paths = self._placed_face(batch, entity, occurrence)
                    hit = {
                        "point": result.points,
                        "distance": result.distances,
                        "status": result.status,
                        "occurrence": selected_paths,
                    }
                    parameters[selected] = np.asarray(result.parameters)[: selected.size]
                    normals[selected] = np.asarray(result.normals)[: selected.size]
                    first[selected] = np.asarray(result.first_derivatives)[
                        : selected.size
                    ]
                    second[selected] = np.asarray(result.second_derivatives)[
                        : selected.size
                    ]
                case _:
                    raise ValueError("Unsupported projection entity dimension.")
            closest[selected] = np.asarray(hit["point"])[: selected.size]
            residuals[selected] = np.asarray(hit["distance"])[: selected.size]
            status[selected] = np.asarray(hit["status"])[: selected.size]
            occurrences[selected] = np.asarray(hit["occurrence"])[: selected.size]
        return self._result(
            dims,
            rows,
            queries,
            closest,
            parameters,
            residuals,
            status,
            normals,
            first,
            second,
            occurrences,
        )

    def _result(
        self,
        dimensions: np.ndarray,
        indices: np.ndarray,
        queries: np.ndarray,
        closest: np.ndarray,
        parameters: np.ndarray,
        residuals: np.ndarray,
        status: np.ndarray,
        normals: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        occurrences: np.ndarray,
        /,
    ) -> BRepProjectionResult:
        count = dimensions.size
        tangents = np.full((count, 2, 3), np.nan)
        face = (dimensions == BRepEntityDimension.FACE) & np.all(np.isfinite(normals), 1)
        first_norm = np.linalg.norm(first, axis=1)
        usable = np.isfinite(first_norm) & (first_norm > 0.0)
        tangents[usable, 0] = first[usable] / first_norm[usable, None]
        # At a pole the first derivative vanishes: seed the frame with the other.
        second_norm = np.linalg.norm(second, axis=1)
        swap = face & ~usable & np.isfinite(second_norm) & (second_norm > 0.0)
        tangents[swap, 0] = second[swap] / second_norm[swap, None]
        frame = face & np.all(np.isfinite(tangents[:, 0]), axis=1)
        tangents[frame, 0] -= (
            np.sum(tangents[frame, 0] * normals[frame], axis=1)[:, None] * normals[frame]
        )
        tangents[frame, 0] /= np.linalg.norm(tangents[frame, 0], axis=1)[:, None]
        tangents[frame, 1] = np.cross(normals[frame], tangents[frame, 0])
        found = np.all(np.isfinite(closest), axis=1)
        ambient_points = self._ambient(np.where(found[:, None], closest, 0.0))
        ambient_points[~found] = np.nan
        ambient_normals = (
            normals if self.embedding is None else np.full((count, 2), np.nan)
        )
        return BRepProjectionResult(
            self.source_revision,
            dimensions,
            indices,
            queries,
            ambient_points,
            parameters,
            residuals,
            status,
            ambient_normals,
            self._directions(tangents),
            occurrence_indices=occurrences,
            occurrence_paths=self.occurrence_paths,
        )

    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Locate definition solids while retaining the selected occurrence path."""
        queries = self._queries(points, "Solid location queries")
        if self.entity_counts[3] == 0:
            raise ValueError("The B-Rep revision has no solids.")
        solid_paths = tuple(occurrence.path for occurrence in self.query.occurrences)
        if occurrence_paths is not None:
            if len(occurrence_paths) != queries.shape[0] or any(
                path not in solid_paths for path in occurrence_paths
            ):
                raise ValueError(
                    "Solid location requires row-aligned authored solid occurrence paths."
                )
        targets = tuple(
            (index, occurrence.solid)
            for index, occurrence in enumerate(self.query.occurrences)
        )
        if not targets:
            raise ValueError("The B-Rep revision has no placed solid occurrences.")
        count = queries.shape[0]
        inside = np.zeros((count, len(targets)), dtype=np.bool_)
        unresolved = np.zeros((count, len(targets)), dtype=np.bool_)
        exhausted = np.zeros((count, len(targets)), dtype=np.bool_)
        world = self._world(queries)
        budget = BRepQueryBudget(
            self.query.policy.maximum_operations,
            self.query.policy.maximum_points,
            self.query.policy.maximum_scratch_bytes,
        )
        for column, (index, _solid) in enumerate(targets):
            path = self.query.occurrences[index].path
            active = (
                np.ones((count,), dtype=np.bool_)
                if occurrence_paths is None
                else np.asarray(
                    [value == path for value in occurrence_paths], dtype=np.bool_
                )
            )
            if not np.any(active):
                continue
            # Use the public occurrence route: its world-space affine preparation
            # preserves reflected coorientation and avoids lossy inverse placement
            # at large translations. All occurrences share one actual query budget.
            result = self.query.contains(world[active], path=path, budget=budget)
            inside[active, column] = np.asarray(result.inside)
            unresolved[active, column] = np.asarray(result.unresolved)
            exhausted[active, column] = np.asarray(result.resource_exhausted)
        hits = np.count_nonzero(inside, axis=1)
        best = np.argmax(inside, axis=1)
        solids = np.asarray([solid for _, solid in targets], dtype=np.int32)
        occurrences = np.asarray(
            [
                self.occurrence_paths.index(self.query.occurrences[index].path)
                for index, _ in targets
            ],
            dtype=np.int32,
        )
        selected = np.where(hits > 0, occurrences[best], -1)
        if occurrence_paths is not None:
            selected = np.where(
                hits > 0,
                selected,
                np.asarray(
                    [
                        self.occurrence_paths.index(path) if path else -1
                        for path in occurrence_paths
                    ],
                    dtype=np.int32,
                ),
            )
        state = np.where(
            np.any(exhausted, axis=1),
            _FAILED,
            np.where(
                np.any(unresolved, axis=1) | (hits > 1),
                _AMBIGUOUS,
                np.where(hits == 0, _FAILED, _UNIQUE),
            ),
        ).astype(np.int8)
        return BRepProjectionResult(
            self.source_revision,
            np.where(hits > 0, 3, -1).astype(np.int8),
            np.where(hits > 0, solids[best], -1),
            queries,
            np.where(hits[:, None] > 0, queries, np.nan),
            np.full((count, 2), np.nan, dtype=np.float64),
            np.where(hits > 0, 0.0, np.inf),
            state,
            np.full((count, self.ambient_dimension), np.nan, dtype=np.float64),
            np.full((count, 2, self.ambient_dimension), np.nan, dtype=np.float64),
            occurrence_indices=selected,
            occurrence_paths=self.occurrence_paths,
        )


__all__ = [
    "BRepContainmentResult",
    "BRepMeasureResult",
    "BRepQueryPolicy",
    "BRepQueryBudget",
    "BRepQueryResourceError",
    "BRepSurfaceQueryResult",
    "NativeBRepProjection",
    "PreparedBRepQuery",
    "prepare_brep_query",
]

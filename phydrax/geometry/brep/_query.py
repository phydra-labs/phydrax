#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare closest-point and diagnostic winding queries over native B-Rep sources.

Closest points use source-carrier interval covers for discovery, native root
proposals, and bounded interval subdivision for host certification. Traced
execution retains the cover's lower bound and reports unresolved work.

The prepared query composes Green-reduced trim quadrature from ``_measure``
and closed-solid membership strategies from ``_query_membership``.
Diagnostic winding does not certify membership.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from math import pi
from typing import final, Protocol, TYPE_CHECKING, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

import phydrax.geometry.brep._analytic_carrier_closest as analytic_carrier_closest_owner
import phydrax.geometry.brep._analytic_cylinder_closest as analytic_cylinder_closest_owner
import phydrax.geometry.brep._analytic_disk_closest as analytic_disk_closest_owner
import phydrax.geometry.brep._analytic_edge_closest as analytic_edge_closest_owner
import phydrax.geometry.brep._analytic_source_regions as analytic_source_regions_owner
import phydrax.geometry.brep._placement as placement_owner
import phydrax.geometry.brep._stationary_isolation as stationary_isolation_owner

from ..._bvh import bvh_nearest_items, PackedBVH, prepare_bvh
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg import inverse_small_linear, SmallLinearSolvePlan, solve_small_linear
from ...nonlinear import VectorLocalRootPlan
from ...typing import checked
from .._atlas import CurveTrimLoop, DynamicCurveTrimLoop, TrimDomain
from .._interval_enclosure import (
    interval_add,
    prepare_interval_function,
)
from ._affine_query import (
    AffineFace,
    closest_affine,
    prepare_affine_faces,
    QueryBroadphase,
)
from ._constructors import brep_trim_domain, BRepTessellationPolicy
from ._coordinate_request import coordinate_source_request
from ._intersection import RootEndpoint
from ._intersection_curve import (
    IntersectionCurve,
    surface_pieces,
    SurfaceRegion,
)
from ._measure import BRepMeasureResult, BRepQuadrature, prepare_quadrature
from ._model import BRepCurve, BRepGeometry, BRepModel, BRepOccurrence
from ._patches import (
    AbstractSurfacePatch,
    CircleCurve,
    coefficient_enclosures,
    curve_value_and_derivative_host,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    prepare_curve_host,
    SpherePatch,
    TorusPatch,
)
from ._placed import PlacedSurface, source_transform_bounds
from ._projection_contracts import BRepProjectionPolicy, BRepProjectionStatus
from ._query_membership import (
    BRepContainmentResult,
    classify_host_containment,
    full_rectangle,
)
from ._query_types import (
    BRepQueryBudget,
    BRepQueryResourceError,
    EdgeData,
    EdgeHit,
    FaceCertificate,
    FaceData,
    FaceHit,
    QuadricHit,
    Tolerances,
)
from ._request_preparations import TrimRequestPreparations
from ._source_correspondence import fraction_norm_upper


if TYPE_CHECKING:
    from ._analytic_source_regions import AnalyticClosestBatch


_UNIQUE = int(BRepProjectionStatus.UNIQUE)
_SEAM = int(BRepProjectionStatus.SEAM)
_AMBIGUOUS = int(BRepProjectionStatus.AMBIGUOUS)
_FAILED = int(BRepProjectionStatus.FAILED)


class _AnalyticClosestQuery(Protocol):
    def __call__(
        self,
        model: BRepModel,
        face: int,
        points: np.ndarray,
        tolerances: Tolerances,
        budget: BRepQueryBudget,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> AnalyticClosestBatch | None: ...


@final
class BRepQueryPolicy(StrictModule, NonTrainableState):
    """Bounded discovery, refinement and quadrature resolution of native queries.

    ``seed_resolution`` controls the initial interval cover; ``seeds`` bounds
    Newton proposals. ``maximum_subdivisions`` and ``maximum_depth`` bound
    host interval certification; unfinished work remains explicit.
    Measures use ``quadrature_order``-point rules on ``quadrature_subdivisions``
    pieces of every coedge, with each inner Green leg partitioned at the source
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
            values[name] = nonnegative_integer(value, name)
        if seed_resolution < 4 or seeds < 1 or newton_steps < 1:
            raise ValueError(
                f"BRepQueryPolicy.seeding must have seed_resolution >= 4, seeds >= 1 and newton_steps >= 1; got {(seed_resolution, seeds, newton_steps)}"
            )
        if quadrature_order < 2 or quadrature_subdivisions < 2 or chunk_size < 1:
            raise ValueError(
                f"BRepQueryPolicy.quadrature must have order >= 2, subdivisions >= 2 and chunk_size >= 1; got {(quadrature_order, quadrature_subdivisions, chunk_size)}"
            )
        if maximum_operations < 1 or maximum_points < 1 or maximum_scratch_bytes < 1:
            raise ValueError(
                f"BRepQueryPolicy.work must have positive operation, point and scratch budgets; got {(maximum_operations, maximum_points, maximum_scratch_bytes)}"
            )
        self.seed_resolution = values["seed_resolution"]
        self.seeds = values["seeds"]
        self.newton_steps = values["newton_steps"]
        self.quadrature_order = values["quadrature_order"]
        self.quadrature_subdivisions = values["quadrature_subdivisions"]
        self.chunk_size = values["chunk_size"]
        self.maximum_subdivisions = values["maximum_subdivisions"]
        self.maximum_depth = values["maximum_depth"]
        self.maximum_operations = values["maximum_operations"]
        self.maximum_points = values["maximum_points"]
        self.maximum_scratch_bytes = values["maximum_scratch_bytes"]
        self.policy_id = canonical_fingerprint({"kind": "native-brep-query", **values})


# ------------------------------------------------------------------ carriers


# ------------------------------------------------------------------ results


@final
class BRepSurfaceQueryResult(StrictModule):
    """Closest boundary points with pointwise projection evidence.

    ``status`` holds :class:`BRepProjectionStatus` values; ``stationarity`` is
    the normalized first-order optimality residual at the returned point.
    ``normals`` are unit outward (oriented) face normals, NaN where undefined.
    ``distances`` are certified upper bounds, including stationary-root and
    source-correspondence uncertainty; they need not equal the numerical norm
    between the returned representative and the input. ``distance_lower_bounds``
    supplies the corresponding lower evidence.
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
class _CertifiedSurfaceBatch(StrictModule):
    """Host buffers for stationary-root certification and result publication."""

    projected: np.ndarray
    parameters: np.ndarray
    indices: np.ndarray
    distances: np.ndarray
    lower: np.ndarray
    unresolved: np.ndarray
    refinements: np.ndarray
    parameter_bound: np.ndarray
    status: np.ndarray
    normals: np.ndarray
    first: np.ndarray
    second: np.ndarray
    stationarity: np.ndarray
    edge_indices: np.ndarray
    edge_parameters: np.ndarray
    point_error: np.ndarray

    @classmethod
    def from_result(cls, result: BRepSurfaceQueryResult, /) -> _CertifiedSurfaceBatch:
        return cls(
            projected=np.array(result.points, dtype=np.float64),
            parameters=np.array(result.parameters, dtype=np.float64),
            indices=np.array(result.faces, dtype=np.int32),
            distances=np.array(result.distances, dtype=np.float64),
            lower=np.array(result.distance_lower_bounds, dtype=np.float64),
            unresolved=np.array(result.unresolved, dtype=np.bool_),
            refinements=np.array(result.refinements, dtype=np.int32),
            parameter_bound=np.array(result.parameter_bounds, dtype=np.float64),
            status=np.array(result.status, dtype=np.int8),
            normals=np.array(result.normals, dtype=np.float64),
            first=np.array(result.first_derivatives, dtype=np.float64),
            second=np.array(result.second_derivatives, dtype=np.float64),
            stationarity=np.array(result.stationarity, dtype=np.float64),
            edge_indices=np.array(result.edge_indices, dtype=np.int32),
            edge_parameters=np.array(result.edge_parameters, dtype=np.float64),
            point_error=np.array(result.point_error_bounds, dtype=np.float64),
        )

    def result(
        self, proposal: BRepSurfaceQueryResult, cover_operations: np.ndarray, /
    ) -> BRepSurfaceQueryResult:
        return BRepSurfaceQueryResult(
            points=jnp.asarray(self.projected),
            parameters=jnp.asarray(self.parameters),
            faces=jnp.asarray(self.indices),
            distances=jnp.asarray(self.distances),
            distance_lower_bounds=jnp.asarray(self.lower),
            unresolved=jnp.asarray(self.unresolved),
            refinements=jnp.asarray(self.refinements),
            parameter_bounds=jnp.asarray(self.parameter_bound),
            status=jnp.asarray(self.status),
            normals=jnp.asarray(self.normals),
            first_derivatives=jnp.asarray(self.first),
            second_derivatives=jnp.asarray(self.second),
            stationarity=jnp.asarray(self.stationarity),
            edge_indices=jnp.asarray(self.edge_indices),
            edge_parameters=jnp.asarray(self.edge_parameters),
            point_error_bounds=jnp.asarray(self.point_error),
            occurrences=proposal.occurrences,
            proposal_status=proposal.proposal_status,
            query_operations=jnp.asarray(
                np.asarray(proposal.query_operations)
                + self.refinements
                + cover_operations
            ),
            source_revision=proposal.source_revision,
        )


# ------------------------------------------------------------ JAX kernels


def _clamp_open(uv: Array, face: FaceData, /) -> Array:
    lower = jnp.asarray(face.lower)
    upper = jnp.asarray(face.upper)
    periodic = jnp.asarray(face.periods) > 0.0
    return jnp.where(periodic, uv, jnp.clip(uv, lower, upper))


def _wrap(uv: Array, face: FaceData, /) -> Array:
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


def jadd(first: _JInterval, second: _JInterval, /) -> _JInterval:
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


def jvector(first: _JInterval, second: _JInterval, /) -> _JInterval:
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
        result = jadd(
            (self.center_lower, self.center_upper),
            jvector(
                (self.basis_lower, self.basis_upper),
                (values[0][:count], values[1][:count]),
            ),
        )
        if self.axial:
            result = jadd(
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
            jvector((self.basis_lower.T, self.basis_upper.T), delta),
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
                jvector((self.axis_lower[None], self.axis_upper[None]), delta),
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
            cross = jvector(
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
    system: _QuadricSystem, points: Array, seeds: Array, tol: Tolerances
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
        image = jadd(
            _jsub((center, center), jvector((preconditioner, preconditioner), value)),
            jvector(remainder, offset),
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


def _trim_box_clear(
    domain: TrimDomain,
    parameters: Array,
    errors: Array,
    /,
) -> Array:
    clear = domain.contains(parameters)
    for loop in domain.loops:
        if not isinstance(loop, (CurveTrimLoop, DynamicCurveTrimLoop)):
            raise ValueError(
                f"TrimDomain.loops must contain native source-arc covers; got {type(loop).__name__}"
            )
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
        center_interval = jadd(
            jvector((rotation, rotation), (system.center_lower, system.center_upper)),
            (shift, shift),
        )
        basis_interval = _jmat(
            (rotation, rotation), (system.basis_lower, system.basis_upper)
        )
        axis = rotation @ system.axis if system.axial else None
        axis_interval = jvector(
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
    face: FaceData,
    points: Array,
    parameters: Array,
    tol: Tolerances,
    /,
) -> QuadricHit | None:
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
    # Enclosure radii are numerical evidence, not geometric motion. Their
    # endpoints already have frozen tangents; freeze the nominal centre too,
    # or its motion against those endpoints contaminates the distance tangent.
    proof_projected = jax.lax.stop_gradient(projected)
    point_error = jnp.linalg.norm(
        jnp.maximum(
            jnp.abs(point_lower - proof_projected),
            jnp.abs(point_upper - proof_projected),
        ),
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
        if isinstance(patch, SpherePatch) and face.periods[0] > 0.0:
            whole_sphere = (face.lower[1] == -0.5 * pi) & (face.upper[1] == 0.5 * pi)
            admissible = jnp.where(whole_sphere, True, admissible)
            box_valid = jnp.where(whole_sphere, True, box_valid)
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
    return QuadricHit(
        point=projected,
        parameter=uv,
        parameter_bound=jnp.max(error, axis=-1),
        point_lower=point_lower,
        point_upper=point_upper,
        point_error=point_error,
        unique=unique,
        pole=pole,
        normal=normal,
    )


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


def _edge_carrier_lower(edge: EdgeData, points: Array, scale: Array, /) -> Array:
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


def _edge_query(edge: EdgeData, points: Array, tol: Tolerances, /) -> EdgeHit:
    """Closest point on one edge: seeded Newton roots plus both end points."""
    count = points.shape[0]
    if edge.curve is None:
        # A degenerate edge is geometrically its single vertex.
        point = jnp.broadcast_to(edge.start_point, (count, 3))
        return EdgeHit(
            point=point,
            parameter=jnp.full((count,), edge.first),
            distance=jnp.linalg.norm(points - point, axis=-1),
            status=jnp.full((count,), _UNIQUE, dtype=jnp.int8),
            tangent=jnp.full((count, 3), jnp.nan),
            lower_bound=jnp.linalg.norm(points - point, axis=-1),
            unresolved=jnp.zeros((count,), dtype=jnp.bool_),
            parameter_bound=jnp.zeros((count,), dtype=jnp.float64),
            proposal_status=jnp.full((count,), _UNIQUE, dtype=jnp.int8),
        )
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
    return EdgeHit(
        point=point,
        parameter=parameter,
        distance=best_distance,
        status=status,
        tangent=velocity,
        lower_bound=lower_bound,
        unresolved=unresolved,
        parameter_bound=parameter_bound,
        proposal_status=jnp.where(
            tie | continuum, _AMBIGUOUS, jnp.where(seam, _SEAM, _UNIQUE)
        ).astype(jnp.int8),
    )


def _frame(face: FaceData, uv: Array, /) -> tuple[Array, Array, Array]:
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


def _host_frame(
    face: FaceData, uv: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the chart-limit frame through the same authored NumPy plan."""

    def frame(parameter: Array) -> Array:
        du, dv, normal = _frame(face, parameter[None])
        return jnp.stack((du[0], dv[0], normal[0]))

    prepared = prepare_interval_function(frame, 2, batch_capacity=1)
    result = prepared.evaluate_host_points(uv.reshape(1, 2))[0]
    return result[0], result[1], result[2]


def _carrier_lower(face: FaceData, points: Array, scale: Array, /) -> Array:
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
    face: FaceData, points: Array, tol: Tolerances, /
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
    face: FaceData, points: Array, tol: Tolerances, /
) -> Array:
    """Bound the angular family across its chart, including non-isolated minima."""
    if not face.full_rectangle or face.periods[0] == 0.0:
        return jnp.zeros((points.shape[0],), dtype=jnp.bool_)
    patch = face.patch
    if isinstance(patch, SpherePatch):
        distance = jnp.linalg.norm(points - patch.center, axis=-1)
        spread = face.carrier_radii[1] - face.carrier_radii[0] + 2.0 * distance
        return (
            (face.lower[1] == -0.5 * pi)
            & (face.upper[1] == 0.5 * pi)
            & (spread + 1024.0 * jnp.finfo(jnp.float64).eps * tol.scale <= tol.distance)
            & (face.carrier_radii[0] > 4.0 * tol.distance)
        )
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


@final
class _InteriorFaceCandidates(StrictModule):
    """Refined interior proposals admitted by chart bounds and trim membership."""

    parameters: Array
    points: Array
    valid: Array
    lower_bound: Array


def _face_interior_candidates(
    face: FaceData, points: Array, tol: Tolerances, /
) -> _InteriorFaceCandidates:
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
    return _InteriorFaceCandidates(
        parameters=interior_uv,
        points=interior_points,
        valid=interior_valid,
        lower_bound=lower_bound,
    )


@final
class _FaceProjectionProposal(StrictModule):
    """Selected numerical face proposal before source-distance certification."""

    point: Array
    parameter: Array
    distance: Array
    status: Array
    stationarity: Array
    lower_bound: Array
    unresolved: Array
    parameter_bound: Array
    edge_index: Array
    edge_parameter: Array
    near_bound: Array


def _certify_face_proposal(
    face: FaceData, points: Array, tol: Tolerances, proposal: _FaceProjectionProposal, /
) -> FaceHit:
    count = points.shape[0]
    best_point = proposal.point
    uv = proposal.parameter
    best_distance = proposal.distance
    status = proposal.status
    stationary_residual = proposal.stationarity
    lower_bound = proposal.lower_bound
    unresolved = proposal.unresolved
    parameter_bound = proposal.parameter_bound
    near_bound = proposal.near_bound
    multiplicity = jnp.zeros((count,), dtype=jnp.bool_)
    point_error = jnp.full((count,), jnp.inf, dtype=jnp.float64)
    quadric = _surface_quadric(face, points, uv, tol)
    if quadric is not None:
        proved = quadric.unique
        best_point = jnp.where(proved[:, None], quadric.point, best_point)
        uv = jnp.where(proved[:, None], quadric.parameter, uv)
        candidate_distance = (
            jnp.linalg.norm(quadric.point - points, axis=-1) + quadric.point_error
        )
        delta = jnp.maximum(
            jnp.maximum(quadric.point_lower - points, points - quadric.point_upper),
            0.0,
        )
        candidate_lower = jnp.maximum(
            jnp.linalg.norm(delta, axis=-1)
            - 64.0 * jnp.finfo(jnp.float64).eps * tol.scale,
            0.0,
        )
        best_distance = jnp.where(proved, candidate_distance, best_distance)
        lower_bound = jnp.where(proved, candidate_lower, lower_bound)
        parameter_bound = jnp.where(proved, quadric.parameter_bound, parameter_bound)
        point_error = jnp.where(proved, quadric.point_error, point_error)
        multiplicity = proved
        status = jnp.where(
            proved,
            jnp.where(quadric.pole | jnp.any(near_bound, axis=1), _SEAM, _UNIQUE),
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
        normal = jnp.where(quadric.unique[:, None], quadric.normal, normal)
    return FaceHit(
        point=best_point,
        parameter=uv,
        distance=best_distance,
        status=status,
        normal=normal,
        du=du,
        dv=dv,
        stationarity=stationary_residual,
        lower_bound=lower_bound,
        unresolved=unresolved,
        parameter_bound=parameter_bound,
        edge_index=proposal.edge_index,
        edge_parameter=proposal.edge_parameter,
        multiplicity=multiplicity,
        point_error=point_error,
    )


def _face_query(
    face: FaceData,
    edge_hits: tuple[EdgeHit, ...],
    points: Array,
    tol: Tolerances,
    /,
) -> FaceHit:
    """Closest point of every query on one trimmed face."""
    count = points.shape[0]
    patch = face.patch
    interior = _face_interior_candidates(face, points, tol)
    interior_uv, interior_points, interior_valid = (
        interior.parameters,
        interior.points,
        interior.valid,
    )
    lower_bound = interior.lower_bound
    lower, upper = jnp.asarray(face.lower), jnp.asarray(face.upper)

    def evaluate(uv: Array) -> Array:
        return patch.evaluate(_clamp_open(uv, face))

    def stationarity(uv: Array, point: Array) -> Array:
        difference = evaluate(uv) - point
        return jax.jacfwd(evaluate)(uv).T @ difference / tol.scale**2

    boundary_points = jnp.stack([hit.point for hit in edge_hits], axis=1)
    boundary_uv = jnp.stack(
        [
            pcurve.evaluate(hit.parameter)
            for pcurve, hit in zip(face.pcurves, edge_hits, strict=True)
        ],
        axis=1,
    )
    boundary_status = jnp.stack([hit.proposal_status for hit in edge_hits], axis=1)
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
    boundary_parameters = jnp.stack([hit.parameter for hit in edge_hits], axis=1)
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
        same[:, interior.valid.shape[1] :] & (boundary_status == _AMBIGUOUS), axis=1
    )
    seam_edge = jnp.any(
        same[:, interior.valid.shape[1] :] & jnp.asarray(face.seam_coedges), 1
    )
    periods = jnp.asarray(face.periods)
    near_bound = (
        (uv - lower <= tol.parametric * periods)
        | (upper - uv <= tol.parametric * periods)
    ) & (periods > 0.0)
    seam = seam_edge | jnp.any(near_bound, axis=1) | at_pole
    edge_lower = jnp.stack([hit.lower_bound for hit in edge_hits], axis=1)
    lower_bound = jnp.minimum(lower_bound, jnp.min(edge_lower, axis=1))
    lower_bound = jnp.maximum(lower_bound, _carrier_lower(face, points, tol.scale))
    coupled_bound = jnp.stack([hit.parameter_bound for hit in edge_hits], axis=1)
    parameter_bound = jnp.max(
        jnp.where(same[:, interior.valid.shape[1] :], coupled_bound, 0.0), axis=1
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
    return _certify_face_proposal(
        face,
        points,
        tol,
        _FaceProjectionProposal(
            point=best_point,
            parameter=uv,
            distance=best_distance,
            status=status,
            stationarity=stationary_residual,
            lower_bound=lower_bound,
            unresolved=unresolved,
            parameter_bound=parameter_bound,
            edge_index=candidate_edge_indices[rows, best],
            edge_parameter=candidate_edge_parameters[rows, best],
            near_bound=near_bound,
        ),
    )


@eqx.filter_jit
def _project_edge(edge: EdgeData, points: Array, tol: Tolerances) -> EdgeHit:
    return _edge_query(edge, points, tol)


@eqx.filter_jit
def _polish_edge(edge: EdgeData, point: Array, seed: Array, tol: Tolerances) -> Array:
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


@dataclass(frozen=True, slots=True, order=True)
class _EdgeIntervalEntry:
    """Distance-ordered source interval with deterministic insertion tie-breaking."""

    lower: float
    serial: int
    first: float
    last: float
    depth: int


def _host_box_distance(point: np.ndarray, box: np.ndarray, /) -> float:
    delta = np.maximum(np.maximum(box[0] - point, point - box[1]), 0.0)
    return float(np.linalg.norm(delta))


def _host_edge_carrier_lower(
    edge: EdgeData, points: np.ndarray, scale: float, /
) -> np.ndarray:
    """Host form of the prepared edge carrier bound used by interval refinement."""
    curve = edge.curve
    lower = np.zeros((points.shape[0],), dtype=np.float64)
    match curve:
        case LineCurve():
            origin, direction = np.asarray(curve.origin), np.asarray(curve.direction)
            relative = points - origin
            parameter = np.clip(
                np.sum(relative * direction, axis=-1) / np.sum(direction**2),
                edge.first,
                edge.last,
            )
            lower = np.linalg.norm(
                points - (origin + parameter[:, None] * direction), axis=-1
            )
        case CircleCurve():
            center = np.asarray(curve.center)
            first_axis, second_axis = (
                np.asarray(curve.first_axis),
                np.asarray(curve.second_axis),
            )
            radius = float(curve.radius)
            relative = points - center
            normal = np.cross(first_axis, second_axis)
            normal = normal / np.linalg.norm(normal)
            height = relative @ normal
            radial = np.linalg.norm(relative - height[:, None] * normal, axis=-1)
            frame = np.stack((first_axis, second_axis), axis=1)
            defect = np.linalg.norm(frame.T @ frame - np.eye(2, dtype=np.float64))
            radius_error = radius * defect
            gap = np.maximum(np.abs(radial - radius) - radius_error, 0.0)
            lower = np.sqrt(gap**2 + height**2)
        case _:
            pass
    margin = (
        256.0
        * np.finfo(np.float64).eps
        * np.maximum(scale, np.linalg.norm(points, axis=-1))
    )
    return np.maximum(lower - margin, 0.0)


@final
class _CurvePointEvaluator(StrictModule):
    """Expose a source curve's point map to the shared host evaluator."""

    curve: BRepCurve

    def evaluate(self, parameter: Array, /) -> Array:
        return _curve_value(self.curve, parameter)


@checked
def certified_edge_query(
    edge: EdgeData,
    points: Array,
    tol: Tolerances,
    policy: BRepQueryPolicy,
    /,
    *,
    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] | None = None,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> EdgeHit:
    if (
        isinstance(points, Tracer)
        or isinstance(edge.first, Tracer)
        or isinstance(edge.last, Tracer)
    ):
        raise ValueError(
            "PreparedBRepQuery.certified_edge_query must receive concrete host inputs; got traced points or edge limits."
        )
    # This certified host boundary normalizes dynamic range leaves once.
    first_host, last_host = float(edge.first), float(edge.last)
    curve = edge.curve
    if curve is None:
        return _project_edge(edge, points, tol)
    hit = _project_edge(edge, points, tol)
    # Padded or repeated queries are certified once and scattered back.
    _, representatives, inverse = np.unique(
        np.asarray(points, dtype=np.float64),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    inverse = inverse.reshape(-1)
    host_points = np.asarray(points, dtype=np.float64)[representatives]
    parameters = np.array(hit.parameter, dtype=np.float64)[representatives]
    distances = np.array(hit.distance, dtype=np.float64)[representatives]
    lower_bounds = np.array(hit.lower_bound, dtype=np.float64)[representatives]
    parameter_bounds = np.array(hit.parameter_bound, dtype=np.float64)[representatives]
    evaluator = _CurvePointEvaluator(curve=curve)
    pending = np.ones((host_points.shape[0],), dtype=np.bool_)
    carriers = analytic_edge_closest_owner.edge_carriers(curve)
    if carriers is not None and isinstance(curve, IntersectionCurve):
        # Any arc end is a candidate; a whole closed branch has none.
        whole = (
            edge.closed
            and curve.closed
            and last_host - first_host == float(curve.num_charts)
        )
        end_boxes = (
            np.zeros((0, 2, 3), dtype=np.float64)
            if whole
            else np.stack(
                [
                    np.asarray(
                        curve.bounding_box(
                            parameter,
                            parameter,
                            endpoint_roots=endpoint_roots,
                            preparations=preparations,
                            budget=budget,
                        ),
                        dtype=np.float64,
                    )
                    for parameter in (first_host, last_host)
                ]
            )
        )
        certified = analytic_edge_closest_owner.edge_distance_lower_bounds(
            carriers,
            host_points,
            np.asarray(hit.point, dtype=np.float64)[representatives],
            distances,
            float(tol.distance),
            end_boxes=end_boxes,
            maximum_boxes=policy.maximum_subdivisions,
            maximum_depth=policy.maximum_depth,
            budget=budget,
        )
        lower_bounds = np.where(certified.decided, certified.lower, lower_bounds)
        pending = ~certified.decided
        decided = np.flatnonzero(certified.decided)
        if decided.size:
            # The traced proposal bounds coupled parameters by whole chart
            # boxes; decided rows take the concrete host chart enclosure.
            parameter_bounds[decided] = np.asarray(
                curve.evaluate(
                    jnp.asarray(parameters[decided], dtype=jnp.float64),
                    preparations=preparations,
                    budget=budget,
                ).parameter_bound,
                dtype=np.float64,
            )
    fallback = np.flatnonzero(pending)
    if fallback.size:
        point_plan = prepare_curve_host(evaluator, batch_capacity=1, budget=budget)
        analytic_bounds = _host_edge_carrier_lower(edge, host_points, float(tol.scale))
        box = (
            curve.bounding_box(
                first_host,
                last_host,
                endpoint_roots=endpoint_roots,
                preparations=preparations,
                budget=budget,
            )
            if isinstance(curve, IntersectionCurve)
            else curve.bounding_box(
                first_host,
                last_host,
                preparations=preparations,
                budget=budget,
            )
        )
        source_box = np.asarray(box)
    else:
        point_plan, analytic_bounds, source_box = None, lower_bounds, lower_bounds
    for row in fallback:
        assert point_plan is not None
        point = host_points[row]
        analytic = float(analytic_bounds[row])
        queue: list[_EdgeIntervalEntry] = []
        lower = max(
            analytic,
            _host_box_distance(point, source_box),
        )
        heapq.heappush(
            queue,
            _EdgeIntervalEntry(
                lower=lower, serial=0, first=first_host, last=last_host, depth=0
            ),
        )
        closed_lower, serial, work = np.inf, 1, 0
        while queue and work < policy.maximum_subdivisions:
            entry = heapq.heappop(queue)
            lower, first, last, depth = entry.lower, entry.first, entry.last, entry.depth
            if lower >= distances[row] - tol.distance:
                closed_lower = min(closed_lower, lower)
                continue
            middle = 0.5 * (first + last)
            parameter = float(
                jax.device_get(
                    _polish_edge(
                        edge,
                        jnp.asarray(point),
                        jnp.asarray(middle, dtype=jnp.float64),
                        tol,
                    )
                )
            )
            projected = point_plan.evaluate_host_points(
                np.asarray(((parameter,),), dtype=np.float64)
            )[0]
            distance = float(np.linalg.norm(projected - point))
            if distance < distances[row]:
                distances[row], parameters[row] = distance, float(parameter)
            if depth >= policy.maximum_depth or not first < middle < last:
                closed_lower = min(closed_lower, lower)
                continue
            for start, stop in ((first, middle), (middle, last)):
                child_box = (
                    curve.bounding_box(
                        start,
                        stop,
                        endpoint_roots=endpoint_roots,
                        preparations=preparations,
                        budget=budget,
                    )
                    if isinstance(curve, IntersectionCurve)
                    else curve.bounding_box(
                        start, stop, preparations=preparations, budget=budget
                    )
                )
                child_lower = max(
                    analytic,
                    _host_box_distance(point, np.asarray(child_box)),
                )
                heapq.heappush(
                    queue,
                    _EdgeIntervalEntry(
                        lower=child_lower,
                        serial=serial,
                        first=start,
                        last=stop,
                        depth=depth + 1,
                    ),
                )
                serial += 1
            work += 1
        lower_bounds[row] = min(
            distances[row], closed_lower, queue[0].lower if queue else np.inf
        )
        if isinstance(curve, IntersectionCurve):
            parameter_bounds[row] = float(
                curve.evaluate(
                    jnp.asarray(parameters[row], dtype=jnp.float64)
                ).parameter_bound
            )
    host_projected, host_tangent = curve_value_and_derivative_host(evaluator, parameters)
    unresolved = (
        (distances - lower_bounds > tol.distance)
        | (parameter_bounds > tol.parametric)
        | ~np.isfinite(parameter_bounds)
    )[inverse]
    return EdgeHit(
        point=jnp.asarray(host_projected[inverse]),
        parameter=jnp.asarray(parameters[inverse], dtype=jnp.float64),
        distance=jnp.asarray(distances[inverse]),
        status=jnp.where(jnp.asarray(unresolved), _FAILED, hit.proposal_status).astype(
            jnp.int8
        ),
        tangent=jnp.asarray(host_tangent[inverse]),
        lower_bound=jnp.asarray(lower_bounds[inverse]),
        unresolved=jnp.asarray(unresolved),
        parameter_bound=jnp.asarray(parameter_bounds[inverse]),
        proposal_status=hit.proposal_status,
    )


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


class _SquaredDistance(StrictModule):
    evaluator: AbstractSurfacePatch

    def evaluate(self, values: Array, /) -> Array:
        difference = self.evaluator.evaluate(values[:2]) - values[2:]
        return jnp.sum(difference * difference)

    def gradient(self, values: Array, /) -> Array:
        return jax.jacfwd(self.evaluate)(values)[:2]

    def hessian(self, values: Array, /) -> Array:
        return jax.jacfwd(self.gradient)(values)[:, :2]


def _face_certificates(
    faces: tuple[FaceData, ...], /, *, budget: CoordinateEnclosureBudget
) -> tuple[FaceCertificate, ...]:
    certificates = []
    for face, data in enumerate(faces):
        for piece in surface_pieces(
            SurfaceRegion(
                data.patch,
                np.asarray((data.lower, data.upper), dtype=np.float64),
                budget=budget,
            ),
            budget=budget,
        ):
            objective = _SquaredDistance(piece.evaluator)
            coefficients = coefficient_enclosures(objective)
            certificates.append(
                FaceCertificate(
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


def _interval_certificates(
    query: PreparedBRepQuery,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
) -> tuple[FaceCertificate, ...]:
    """Retain at most eight interval programs in the explicit query request."""
    certificates = preparations.interval_certificates
    cached = certificates.get(query.query_id)
    if cached is not None:
        certificates.move_to_end(query.query_id)
        return cached
    prepared = _face_certificates(query.faces, budget=budget)
    certificates[query.query_id] = prepared
    certificates.move_to_end(query.query_id)
    if len(certificates) > 8:
        certificates.popitem(last=False)
    return prepared


def _prepare_edge(
    geometry: BRepGeometry,
    edge: int,
    resolution: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> EdgeData:
    first, last = (float(value) for value in np.asarray(geometry.edge_ranges)[edge])
    start, end = geometry.edge_vertices[edge]
    points = np.asarray(geometry.vertex_points)
    curve_index = geometry.edge_curves[edge]
    curve = None if curve_index == -1 else geometry.curves[curve_index]
    if isinstance(curve, IntersectionCurve):
        parameter_box = geometry.edge_parameter_enclosure(
            edge, preparations=preparations, budget=budget
        )
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
                    float(a),
                    float(b),
                    endpoint_roots=geometry.edge_endpoint_roots[edge],
                    preparations=preparations,
                    budget=budget,
                )
                if isinstance(curve, IntersectionCurve)
                else curve.bounding_box(
                    float(a), float(b), preparations=preparations, budget=budget
                )
                for a, b in zip(endpoints[:-1], endpoints[1:], strict=True)
            ]
        )
    )
    return EdgeData(
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
            raise ValueError(
                f"PlanePatch.axes must span a plane; got normal length {norm}"
            )
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
            raise ValueError(f"CylinderPatch.axis must have nonzero length; got {length}")
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


def _prepare_face(
    model: BRepModel,
    geometry: BRepGeometry,
    face: int,
    resolution: int,
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
) -> FaceData:
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
    interval_boxes = np.stack(
        [
            patch.bounding_box(cell, preparations=preparations, budget=budget)
            for cell in cells
        ]
    )
    trim_domain = brep_trim_domain(
        geometry,
        face,
        patch,
        tolerance=float(np.max(box[1] - box[0])) / resolution,
        budget=budget,
        preparations=preparations,
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
    return FaceData(
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
        full_rectangle=full_rectangle(geometry, face, patch, box),
    )


@final
class PreparedBRepQuery(StrictModule, NonTrainableState):
    """Prepared native closest-point, containment, bound and measure queries.

    Construct with :func:`prepare_brep_query`. Traced point queries are pure JAX
    with conservative prepared-cover evidence. Untraced queries additionally
    run bounded host interval certification against the immutable source.
    """

    model: BRepModel
    faces: tuple[FaceData, ...]
    edges: tuple[EdgeData, ...]
    quadrature: BRepQuadrature
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
    tolerances: Tolerances
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PreparedBRepQuery],
        value: PreparedBRepQuery,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedBRepQuery:
        """Rebuild prepared query data through its source and policy preparation."""
        rebuilt = cls(
            value.model,
            value.policy,
            value.projection_policy,
            certify_on_host=value.certify_on_host,
            budget=budget,
        )
        if (
            rebuilt.query_id != value.query_id
            or rebuilt.model_id != value.model_id
            or rebuilt.source_revision != value.source_revision
        ):
            raise ValueError(
                "PreparedBRepQuery.stored_fields must preserve query and source identity; "
                f"got query={value.query_id!r}, model={value.model_id!r}, revision={value.source_revision!r}."
            )
        return rebuilt

    @checked
    def __init__(
        self,
        model: BRepModel,
        policy: BRepQueryPolicy,
        projection_policy: BRepProjectionPolicy,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        certify_on_host: bool = True,
    ) -> None:
        if not isinstance(certify_on_host, bool):
            raise TypeError(
                f"PreparedBRepQuery.certify_on_host must be a bool; got {certify_on_host!r}"
            )
        if preparations is None:
            preparations = TrimRequestPreparations()
        with coordinate_source_request(
            (
                model,
                policy,
                projection_policy,
            ),
            budget=budget,
        ) as budget:
            geometry = model.require_geometry()
            face_count = len(model.patches)
            if face_count == 0:
                raise ValueError(
                    f"PreparedBRepQuery.model.patches must contain a nonempty boundary; got {face_count} faces"
                )
            points = np.asarray(geometry.vertex_points)
            boxes = np.stack(
                [
                    model.patches[face].bounding_box(
                        np.asarray(model.parameter_bounds)[face],
                        budget=budget,
                        preparations=preparations,
                    )
                    for face in range(face_count)
                ]
            )
            scale = max(
                1.0,
                float(np.max(np.max(boxes[:, 1], axis=0) - np.min(boxes[:, 0], axis=0))),
            )
            self.model = model
            self.faces = tuple(
                _prepare_face(
                    model,
                    geometry,
                    face,
                    policy.seed_resolution,
                    budget=budget,
                    preparations=preparations,
                )
                for face in range(face_count)
            )
            self.edges = tuple(
                _prepare_edge(
                    geometry,
                    edge,
                    4 * policy.seed_resolution + 1,
                    budget=budget,
                    preparations=preparations,
                )
                for edge in range(len(geometry.edge_curves))
            )
            self.quadrature, self.measures = prepare_quadrature(
                model, geometry, policy, coordinate_budget=budget
            )
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
            self.tolerances = Tolerances(
                distance=projection_policy.ambiguity_tolerance,
                parametric=projection_policy.parametric_tolerance,
                classifier=projection_policy.classifier_tolerance,
                scale=jnp.asarray(scale, dtype=jnp.float64),
                newton_steps=policy.newton_steps,
                seeds=policy.seeds,
            )
            self.policy = policy
            self.projection_policy = projection_policy
            self.certify_on_host = certify_on_host
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
                materialized = placement_owner.materialize_brep_occurrences(
                    model,
                    tessellation=BRepTessellationPolicy(realize=False),
                    budget=budget,
                    preparations=preparations,
                )
                world = PreparedBRepQuery(
                    materialized.model,
                    policy,
                    projection_policy,
                    budget=budget,
                    preparations=preparations,
                    certify_on_host=certify_on_host,
                )
                face_sources = [-1] * len(world.faces)
                edge_sources = [-1] * len(world.edges)
                vertex_sources = [-1] * len(world.vertex_points)
                solids = [-1] * len(self.occurrences)
                paths = {
                    occurrence.path: index
                    for index, occurrence in enumerate(self.occurrences)
                }
                for source, target in zip(
                    materialized.source_entities,
                    materialized.target_entities,
                    strict=True,
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
                    min(
                        (*face_sources, *edge_sources, *vertex_sources, *solids),
                        default=0,
                    )
                    < 0
                ):
                    raise RuntimeError(
                        "PreparedBRepQuery.world_source must map every source entity; got incomplete incidence"
                    )
                self.world_query = world
                self.world_face_sources = tuple(face_sources)
                self.world_edge_sources = tuple(edge_sources)
                self.world_vertex_sources = tuple(vertex_sources)
                self.world_solids = tuple(solids)

    @checked
    def world_closest(
        self,
        points: Array,
        faces: tuple[int, ...],
        occurrence: int,
        /,
        *,
        budget: BRepQueryBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> BRepSurfaceQueryResult:
        world = self.world_query
        if world is None:
            raise RuntimeError(
                "PreparedBRepQuery.world_query must contain prepared occurrence geometry; got None"
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
        result = world.closest_definition(
            points,
            selected,
            budget=budget,
            preparations=preparations,
            coordinate_budget=coordinate_budget,
        )
        if result.source_revision != world.source_revision:
            raise ValueError(
                "PreparedBRepQuery.world_closest must retain its prepared world source revision."
            )
        source_faces = jnp.asarray(self.world_face_sources, dtype=jnp.int32)
        source_edges = jnp.asarray(self.world_edge_sources, dtype=jnp.int32)
        # Face and edge rows return to the authored definition namespace; the
        # source revision must follow them, not name the expanded world model.
        return replace(
            result,
            faces=source_faces[result.faces],
            edge_indices=jnp.where(
                result.edge_indices >= 0,
                source_edges[jnp.maximum(result.edge_indices, 0)],
                -1,
            ),
            occurrences=jnp.full(result.distances.shape, occurrence, dtype=jnp.int32),
            source_revision=self.source_revision,
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
            rotation, translation = self.placement(occurrence)
            mapped = jvector((rotation, rotation), (box[0], box[1]))
            lower, upper = jadd(mapped, (translation, translation))
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

    def placement(self, occurrence: int, /) -> tuple[Array, Array]:
        if occurrence == -1:
            return jnp.eye(3, dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64)
        return self.occurrence_rotations[occurrence], self.occurrence_translations[
            occurrence
        ]

    @checked
    def prepare_placed_query(
        self,
        occurrence: int,
        faces: tuple[int, ...],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[AffineFace, ...], QueryBroadphase, np.ndarray] | None:

        if occurrence == -1:
            matrix, shift = np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64)
        else:
            rotations, translations = jax.device_get(
                (self.occurrence_rotations, self.occurrence_translations)
            )
            matrix, shift = rotations[occurrence], translations[occurrence]
        prepared = prepare_affine_faces(
            self.model, faces, pose=(matrix, shift), budget=budget
        )
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
        raise ValueError(
            f"PreparedBRepQuery.path must name an existing solid occurrence; got {path!r}"
        )

    def _edge_hits(self, points: Array, /) -> tuple[EdgeHit, ...]:
        return tuple(_edge_query(edge, points, self.tolerances) for edge in self.edges)

    def _face_hits(
        self,
        points: Array,
        faces: tuple[int, ...],
        /,
    ) -> FaceHit:
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
        return jax.tree.map(lambda *values: jnp.stack(values), *hits)

    def _closest_chunk(
        self,
        points: Array,
        faces: tuple[int, ...],
        /,
    ) -> BRepSurfaceQueryResult:
        hits = self._face_hits(points, faces)
        count = points.shape[0]
        rows = jnp.arange(count)
        best = jnp.argmin(hits.distance, axis=0)
        best_point = hits.point[best, rows]
        best_distance = hits.distance[best, rows]
        same = (
            jnp.linalg.norm(hits.point - best_point[None], axis=-1)
            <= self.tolerances.distance
        )
        tie = jnp.any(
            ~same & (hits.distance <= best_distance[None] + self.tolerances.distance),
            axis=0,
        )
        lower_bound = jnp.min(hits.lower_bound, axis=0)
        competing = hits.lower_bound <= best_distance[None] + self.tolerances.distance
        unresolved = (best_distance - lower_bound > self.tolerances.distance) | jnp.any(
            competing & hits.unresolved, axis=0
        )
        status = jnp.where(
            unresolved, _FAILED, jnp.where(tie, _AMBIGUOUS, hits.status[best, rows])
        ).astype(jnp.int8)
        return BRepSurfaceQueryResult(
            points=best_point,
            parameters=hits.parameter[best, rows],
            distances=best_distance,
            faces=jnp.asarray(faces, dtype=jnp.int32)[best],
            status=status,
            normals=hits.normal[best, rows],
            first_derivatives=hits.du[best, rows],
            second_derivatives=hits.dv[best, rows],
            stationarity=hits.stationarity[best, rows],
            distance_lower_bounds=lower_bound,
            unresolved=unresolved,
            refinements=jnp.zeros((count,), dtype=jnp.int32),
            parameter_bounds=hits.parameter_bound[best, rows],
            occurrences=jnp.full((count,), -1, dtype=jnp.int32),
            source_revision=self.source_revision,
            proposal_status=jnp.where(tie, _AMBIGUOUS, hits.status[best, rows]).astype(
                jnp.int8
            ),
            edge_indices=hits.edge_index[best, rows],
            edge_parameters=hits.edge_parameter[best, rows],
            point_error_bounds=hits.point_error[best, rows],
            query_operations=jnp.full(
                (count,),
                1
                + len(faces)
                + len({edge for face in faces for edge in self.faces[face].coedge_edges}),
                dtype=jnp.int64,
            ),
        )

    @checked
    def affine_closest(
        self,
        points: np.ndarray,
        faces: tuple[int, ...],
        /,
        *,
        budget: BRepQueryBudget | None = None,
        prepared: tuple[AffineFace, ...] | None = None,
        broadphase: QueryBroadphase | None = None,
        source_boxes: np.ndarray | None = None,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> BRepSurfaceQueryResult | None:

        if prepared is None:
            prepared = prepare_affine_faces(self.model, faces, budget=coordinate_budget)
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
                self.affine_closest(
                    points[start : start + chunk_size],
                    faces,
                    budget=budget,
                    prepared=prepared,
                    broadphase=broadphase,
                    source_boxes=source_boxes,
                    coordinate_budget=coordinate_budget,
                )
                for start in range(0, len(points), chunk_size)
            ]
            if any(chunk is None for chunk in chunks):
                raise RuntimeError(
                    "PreparedBRepQuery.affine_source must retain its admitted bounded query capability; got unsupported chunk"
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
                fraction_norm_upper(
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
                fraction_norm_upper(
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

    def _analytic_closest(
        self,
        points: np.ndarray,
        faces: tuple[int, ...],
        budget: BRepQueryBudget,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> BRepSurfaceQueryResult | None:
        """Certify closed source cylinders and disks without stationary-root isolation."""

        queries: list[
            tuple[
                int,
                _AnalyticClosestQuery,
            ]
        ] = []
        for face in faces:
            match self.model.patches[face]:
                case CylinderPatch():
                    region = analytic_source_regions_owner.complete_cylinder_region(
                        self.model, budget=coordinate_budget
                    )
                    if region is None or region.lateral_face != face:
                        return None
                    queries.append(
                        (face, analytic_cylinder_closest_owner.closest_complete_cylinder)
                    )
                case PlanePatch():
                    if (
                        analytic_source_regions_owner.complete_disk_region(
                            self.model, face, budget=coordinate_budget
                        )
                        is None
                    ):
                        return None
                    queries.append(
                        (face, analytic_disk_closest_owner.closest_complete_disk)
                    )
                case _:
                    return None
        if not queries:
            return None
        source_bytes = 4096 * (len(faces) + self.face_bvh.node_count)
        chunk_size = self._query_chunk_size(budget, source_bytes)
        # Admit all points and source visits before consuming any source work.
        # The extra bank covers retained face results and merge/control arrays.
        scratch = (
            source_bytes
            + analytic_cylinder_closest_owner.closest_workspace_bytes(
                min(len(points), chunk_size)
            )
            + 1024 * len(queries) * min(len(points), chunk_size)
        )
        budget.admit(len(points) * len(queries), len(points), scratch)
        if len(points) > chunk_size:
            chunks = [
                self._analytic_closest(
                    points[start : start + chunk_size],
                    faces,
                    budget,
                    coordinate_budget=coordinate_budget,
                )
                for start in range(0, len(points), chunk_size)
            ]
            if any(chunk is None for chunk in chunks):
                raise RuntimeError(
                    "PreparedBRepQuery.analytic_source must retain its admitted source capabilities; "
                    "got an unsupported admitted chunk."
                )
            return jax.tree.map(lambda *values: jnp.concatenate(values), *chunks)
        budget.consume(0, points=len(points))
        hits = []
        for face, query in queries:
            hit = query(
                self.model,
                face,
                points,
                self.tolerances,
                budget,
                coordinate_budget=coordinate_budget,
            )
            if hit is None:
                raise RuntimeError(
                    "PreparedBRepQuery.analytic_source must retain its admitted source face; "
                    f"got unsupported face {face}."
                )
            hits.append(hit)
        return self._merge_analytic_closest(hits)

    def _carrier_closest(
        self,
        points: np.ndarray,
        faces: tuple[int, ...],
        budget: BRepQueryBudget,
        /,
        *,
        preparations: TrimRequestPreparations | None,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> tuple[BRepSurfaceQueryResult, np.ndarray] | None:
        """Certify interior minimizers of analytic faces from their exact carriers.

        Returns the merged result and the rows that still need stationary-root
        isolation, or ``None`` when a selected face has no closed-form carrier.
        Every row is charged as one dispatch plus its executed face and trim work;
        only decided rows are charged as completed points here.
        """
        if not all(
            analytic_carrier_closest_owner.carrier_supported(self.model.patches[face])
            for face in faces
        ):
            return None
        if preparations is None:
            preparations = TrimRequestPreparations()
        source_bytes = 4096 * (len(faces) + self.face_bvh.node_count)
        per_point = 1024 * (1 + len(faces))
        chunk_size = self._query_chunk_size(
            budget, source_bytes + 16384, per_point_bytes=per_point
        )
        merged, pending = [], []
        for start in range(0, len(points), chunk_size):
            chunk = points[start : start + chunk_size]
            budget.admit(
                len(chunk) * (1 + len(faces)),
                len(chunk),
                source_bytes
                + len(faces)
                * analytic_carrier_closest_owner.carrier_workspace_bytes(len(chunk)),
            )
            hits = []
            for face in faces:
                hit = analytic_carrier_closest_owner.closest_carrier_face(
                    self.model,
                    face,
                    self.faces[face],
                    chunk,
                    self.tolerances,
                    maximum_depth=self.policy.maximum_depth,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
                if hit is None:
                    raise RuntimeError(
                        "PreparedBRepQuery.carrier_source must retain its admitted analytic carrier; "
                        f"got unsupported face {face}."
                    )
                hits.append(hit)
            result = self._merge_analytic_closest(hits)
            unresolved = np.asarray(result.unresolved, dtype=np.bool_)
            budget.consume(
                len(chunk) + int(np.sum(np.asarray(result.query_operations))),
                points=int(np.count_nonzero(~unresolved)),
            )
            merged.append(result)
            pending.append(unresolved)
        result = (
            merged[0]
            if len(merged) == 1
            else jax.tree.map(lambda *values: jnp.concatenate(values), *merged)
        )
        return result, np.concatenate(pending)

    def _merge_analytic_closest(
        self, hits: list[AnalyticClosestBatch], /
    ) -> BRepSurfaceQueryResult:
        """Merge directed distance bounds and retain non-isolated fiber evidence."""
        points = hits[0].points
        distance = np.stack([hit.distances for hit in hits])
        lower = np.stack([hit.distance_lower_bounds for hit in hits])
        status = np.stack([hit.status for hit in hits])
        closest = np.stack([hit.points for hit in hits])
        best = np.argmin(distance, axis=0)
        rows = np.arange(len(points), dtype=np.int64)
        upper = distance[best, rows]
        selected = closest[best, rows]
        competing = lower <= upper[None] + self.tolerances.distance
        separated = np.linalg.norm(closest - selected[None], axis=-1)
        fiber = status == int(BRepProjectionStatus.NON_ISOLATED)
        tied = np.any(
            competing & ((separated > self.tolerances.distance) | fiber), axis=0
        )
        selected_status = status[best, rows]
        only_selected_fiber = (
            selected_status == int(BRepProjectionStatus.NON_ISOLATED)
        ) & (np.sum(competing, axis=0) == 1)
        unresolved = np.any(np.stack([hit.unresolved for hit in hits]), axis=0) | (
            upper - np.min(lower, axis=0) > self.tolerances.distance
        )
        merged_status = np.where(
            unresolved,
            _FAILED,
            np.where(
                only_selected_fiber,
                int(BRepProjectionStatus.NON_ISOLATED),
                np.where(tied, _AMBIGUOUS, selected_status),
            ),
        ).astype(np.int8)
        return BRepSurfaceQueryResult(
            points=jnp.asarray(selected, dtype=jnp.float64),
            parameters=jnp.asarray(
                np.stack([hit.parameters for hit in hits])[best, rows], dtype=jnp.float64
            ),
            distances=jnp.asarray(upper, dtype=jnp.float64),
            faces=jnp.asarray(
                np.stack([hit.faces for hit in hits])[best, rows], dtype=jnp.int32
            ),
            status=jnp.asarray(merged_status, dtype=jnp.int8),
            normals=jnp.asarray(
                np.stack([hit.normals for hit in hits])[best, rows], dtype=jnp.float64
            ),
            first_derivatives=jnp.asarray(
                np.stack([hit.first_derivatives for hit in hits])[best, rows],
                dtype=jnp.float64,
            ),
            second_derivatives=jnp.asarray(
                np.stack([hit.second_derivatives for hit in hits])[best, rows],
                dtype=jnp.float64,
            ),
            stationarity=jnp.asarray(
                np.stack([hit.stationarity for hit in hits])[best, rows],
                dtype=jnp.float64,
            ),
            distance_lower_bounds=jnp.asarray(np.min(lower, axis=0), dtype=jnp.float64),
            unresolved=jnp.asarray(unresolved, dtype=jnp.bool_),
            refinements=jnp.asarray(
                np.sum(np.stack([hit.refinements for hit in hits]), axis=0),
                dtype=jnp.int32,
            ),
            parameter_bounds=jnp.asarray(
                np.stack([hit.parameter_bounds for hit in hits])[best, rows],
                dtype=jnp.float64,
            ),
            occurrences=jnp.full((len(points),), -1, dtype=jnp.int32),
            proposal_status=jnp.asarray(merged_status, dtype=jnp.int8),
            edge_indices=jnp.asarray(
                np.stack([hit.edge_indices for hit in hits])[best, rows], dtype=jnp.int32
            ),
            edge_parameters=jnp.asarray(
                np.stack([hit.edge_parameters for hit in hits])[best, rows],
                dtype=jnp.float64,
            ),
            point_error_bounds=jnp.asarray(
                np.stack([hit.point_error_bounds for hit in hits])[best, rows],
                dtype=jnp.float64,
            ),
            query_operations=jnp.asarray(
                np.sum(np.stack([hit.query_operations for hit in hits]), axis=0),
                dtype=jnp.int64,
            ),
            source_revision=self.model.source_revision,
        )

    @checked
    def closest_definition(
        self,
        points: Array,
        faces: tuple[int, ...],
        /,
        *,
        maximum_boxes: int | None = None,
        budget: BRepQueryBudget | None = None,
        affine_preparation: tuple[tuple[AffineFace, ...], QueryBroadphase, np.ndarray]
        | None = None,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> BRepSurfaceQueryResult:
        if maximum_boxes is not None and (
            type(maximum_boxes) is not int
            or not 1 <= maximum_boxes <= self.policy.maximum_subdivisions
        ):
            raise ValueError(
                "PreparedBRepQuery.maximum_boxes must tighten the prepared subdivision budget; got invalid scoped allowance"
            )
        host = self.certify_on_host and not isinstance(points, Tracer)
        prepared = None
        carrier: tuple[BRepSurfaceQueryResult, np.ndarray] | None = None
        if host:
            if budget is None:
                budget = BRepQueryBudget(
                    self.policy.maximum_operations,
                    self.policy.maximum_points,
                    self.policy.maximum_scratch_bytes,
                )
            analytic = self._analytic_closest(
                np.asarray(points), faces, budget, coordinate_budget=coordinate_budget
            )
            if analytic is not None:
                return analytic
            prepared = (
                prepare_affine_faces(self.model, faces, budget=coordinate_budget)
                if affine_preparation is None
                else affine_preparation[0]
            )
            if prepared is None:
                carrier = self._carrier_closest(
                    np.asarray(points),
                    faces,
                    budget,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
                if carrier is not None:
                    if not np.any(carrier[1]):
                        return carrier[0]
                    # Only rows without a certified carrier decision continue
                    # to stationary-root isolation.
                    points = jnp.asarray(np.asarray(points)[carrier[1]])
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
            geometry = self.model.require_geometry()
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
                                budget=coordinate_budget,
                                preparations=preparations,
                            ),
                            budget=coordinate_budget,
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
                affine = self.affine_closest(
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
                    coordinate_budget=coordinate_budget,
                )
                if affine is None:
                    raise RuntimeError(
                        "PreparedBRepQuery.affine_source must retain its admitted closest-point capability; got unsupported source"
                    )
                return affine
        elif budget is not None:
            raise ValueError(
                "PreparedBRepQuery.budget must be None during traced execution; got mutable BRepQueryBudget"
            )
        # Closest-point refinement is traced; interval certification and its host
        # coordinate accounting happen afterwards in `_certify_definition`.
        count = points.shape[0]
        if carrier is None:
            result = _compiled_closest_chunks(
                self, points, faces, min(self.policy.chunk_size, count)
            )
        else:
            # Power-of-two padding bounds recompilation across residual subsets.
            padded = 1 << max(0, (count - 1).bit_length())
            result = jax.tree.map(
                lambda value: value[:count],
                _compiled_closest_chunks(
                    self,
                    jnp.concatenate(
                        (points, jnp.broadcast_to(points[:1], (padded - count, 3)))
                    ),
                    faces,
                    min(self.policy.chunk_size, padded),
                ),
            )
        if not host:
            return result
        result = self._certify_definition(
            np.asarray(points),
            faces,
            result,
            maximum_boxes=maximum_boxes,
            preparations=preparations,
            budget=coordinate_budget,
        )
        if budget is not None:
            refinements = int(np.sum(np.asarray(result.refinements)))
            budget.consume(
                int(np.sum(np.asarray(result.query_operations))) - points.shape[0],
                subdivisions=refinements,
            )
        if carrier is not None:
            rows = jnp.asarray(np.flatnonzero(carrier[1]))
            result = jax.tree.map(
                lambda whole, part: whole.at[rows].set(part), carrier[0], result
            )
        return result

    def _publish_stationary_root(
        self,
        state: _CertifiedSurfaceBatch,
        row: int,
        point: np.ndarray,
        isolation: stationary_isolation_owner.ClosestStationaryIsolation,
        /,
        *,
        preparations: TrimRequestPreparations,
        budget: CoordinateEnclosureBudget,
    ) -> _CertifiedSurfaceBatch:
        """Publish a retained minimizer and its fidelity-adjusted competing-root evidence."""
        errors = tuple(
            stationary_isolation_owner.stationary_root_fidelity(
                self.model, root, preparations=preparations, budget=budget
            )
            for root in isolation.roots
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
        representative = stationary_isolation_owner.stationary_root_representative(
            self.model, best, preparations=preparations, budget=budget
        )
        state.projected[row] = representative.point
        state.parameters[row] = representative.uv
        state.parameter_bound[row] = representative.uv_error
        state.point_error[row] = representative.point_error
        state.edge_indices[row] = representative.edge
        state.edge_parameters[row] = representative.parameter
        regular = representative.regular
        state.point_error[row] = max(state.point_error[row], errors[best_index])
        state.indices[row] = best.face
        state.distances[row] = float(np.nextafter(best_upper, np.inf))
        state.lower[row] = max(
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
        du, dv, normal = _host_frame(self.faces[best.face], state.parameters[row])
        state.first[row], state.second[row], state.normals[row] = du, dv, normal
        delta = state.projected[row] - point
        gradient = np.asarray((delta @ state.first[row], delta @ state.second[row]))
        denominator = max(
            np.linalg.norm(np.stack((state.first[row], state.second[row])))
            * max(state.distances[row], self.tolerances.distance),
            np.finfo(np.float64).tiny,
        )
        state.stationarity[row] = np.linalg.norm(gradient) / denominator
        state.unresolved[row] = (
            state.distances[row] - state.lower[row] > self.tolerances.distance
            or state.parameter_bound[row] > self.tolerances.parametric
            or state.point_error[row] > self.tolerances.distance
            or not np.isfinite(state.distances[row])
            or not np.isfinite(state.parameter_bound[row])
            or not np.isfinite(state.point_error[row])
            or not np.all(np.isfinite(state.projected[row]))
            or not np.all(np.isfinite(state.normals[row]))
            or not regular
        )
        state.status[row] = (
            _FAILED
            if state.unresolved[row]
            else (
                _AMBIGUOUS
                if len(competitors) > 1 or best.continuum
                else (_SEAM if best.dimension < 2 else _UNIQUE)
            )
        )
        return state

    def _certify_definition(
        self,
        points: np.ndarray,
        faces: tuple[int, ...],
        result: BRepSurfaceQueryResult,
        /,
        *,
        maximum_boxes: int | None = None,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> BRepSurfaceQueryResult:
        if preparations is None:
            preparations = TrimRequestPreparations()

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
        state = _CertifiedSurfaceBatch.from_result(result)
        certificates = _interval_certificates(
            self, budget=budget, preparations=preparations
        )
        domains = tuple(face.trim_domain for face in self.faces)
        geometry = self.model.require_geometry()
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
            state.parameter_bound[row] = state.point_error[row] = np.inf
            upper = stationary_isolation_owner.source_closure_upper(
                self.model, point, faces, preparations=preparations, budget=budget
            )
            cover_operations[row] += len(source_coedges)
            if not np.isfinite(upper) or not np.isfinite(correspondence):
                state.status[row] = _FAILED
                continue
            isolation = stationary_isolation_owner.isolate_closest_stationary(
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
                preparations=preparations,
                budget=budget,
            )
            cover_operations[row] += initial_cover
            state.refinements[row] = isolation.boxes_processed
            if not isolation.complete or not isolation.roots:
                if not isolation.complete:
                    # Every minimizer lies in a retained root or an undecided
                    # stratum; boxes certified outside their trim never count.
                    # That trim-aware floor is a sound state.lower bound even here.
                    floor = min(
                        (
                            root.distance_lower
                            - stationary_isolation_owner.stationary_root_fidelity(
                                self.model, root, preparations=preparations, budget=budget
                            )
                            for root in isolation.roots
                        ),
                        default=np.inf,
                    )
                    floor = min(
                        floor, isolation.unresolved_distance_lower, upper + correspondence
                    )
                    state.lower[row] = max(
                        state.lower[row], max(0.0, float(np.nextafter(floor, -np.inf)))
                    )
                state.status[row] = _FAILED
                continue
            state = self._publish_stationary_root(
                state, row, point, isolation, preparations=preparations, budget=budget
            )
        return state.result(result, cover_operations)

    @checked
    def closest_point(
        self,
        points: ArrayLike,
        /,
        *,
        faces: tuple[int, ...] | None = None,
        path: tuple[str, ...] | None = None,
        budget: BRepQueryBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> BRepSurfaceQueryResult:
        """World-space boundary projection retaining definition face and occurrence identity."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if coordinate_budget is None:
            with coordinate_source_request(
                (
                    self,
                    points,
                )
            ) as coordinate_budget:
                return self.closest_point(
                    points,
                    faces=faces,
                    path=path,
                    budget=budget,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
        points_ = _points(points)
        faces_ = tuple(range(len(self.faces))) if faces is None else tuple(faces)
        if not faces_ or any(face < 0 or face >= len(self.faces) for face in faces_):
            raise ValueError(
                f"PreparedBRepQuery.faces must list nonempty existing face indices; got {faces_!r}"
            )
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
                "PreparedBRepQuery.budget must be None during traced execution; got mutable BRepQueryBudget"
            )
        if self.certify_on_host and not isinstance(points_, Tracer) and budget is None:
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
                hit = self.world_closest(
                    points_,
                    selected,
                    occurrence,
                    budget=budget,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
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
            rotation, translation = self.placement(occurrence)
            preparation = (
                self.prepare_placed_query(occurrence, selected, budget=coordinate_budget)
                if self.certify_on_host and not isinstance(points_, Tracer)
                else None
            )
            local = (points_ - translation) @ rotation if preparation is None else points_
            hit = self.closest_definition(
                local,
                selected,
                budget=budget,
                affine_preparation=preparation,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
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
            raise ValueError(
                "PreparedBRepQuery.faces must intersect the selected occurrence boundary; got no matching faces"
            )
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
            rotation, translation = self.placement(index)
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
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> BRepContainmentResult:
        if self.certify_on_host and not isinstance(points, Tracer):
            if budget is None:
                budget = BRepQueryBudget(
                    self.policy.maximum_operations,
                    self.policy.maximum_points,
                    self.policy.maximum_scratch_bytes,
                )
                self._admit_containment(
                    points.shape[0], (solid,), budget, coordinate_budget=coordinate_budget
                )
            return classify_host_containment(
                self,
                np.asarray(points),
                solid,
                budget=budget,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
        closest = self.closest_definition(
            points,
            self.solid_faces[solid],
            preparations=preparations,
            coordinate_budget=coordinate_budget,
        )
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
        self,
        count: int,
        solids: tuple[int, ...],
        budget: BRepQueryBudget,
        /,
        *,
        coordinate_budget: CoordinateEnclosureBudget,
    ) -> None:

        reservation, scratch = 0, 0
        for solid in solids:
            faces = self.solid_faces[solid]
            edges = sum(len(self.faces[face].coedge_edges) for face in faces)
            affine = prepare_affine_faces(self.model, faces, budget=coordinate_budget)
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

    @checked
    def contains(
        self,
        points: ArrayLike,
        /,
        *,
        solid: int = 0,
        path: tuple[str, ...] | None = None,
        budget: BRepQueryBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> BRepContainmentResult:
        """Membership of an occurrence path, or the union of occurrences of ``solid``."""
        if preparations is None:
            preparations = TrimRequestPreparations()
        if coordinate_budget is None:
            with coordinate_source_request(
                (
                    self,
                    points,
                )
            ) as coordinate_budget:
                return self.contains(
                    points,
                    solid=solid,
                    path=path,
                    budget=budget,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
        if path is None:
            if not 0 <= solid < len(self.solid_faces):
                raise ValueError(
                    f"PreparedBRepQuery.solid must index an existing solid; got {solid}"
                )
            selected = tuple(
                index
                for index, occurrence in enumerate(self.occurrences)
                if occurrence.solid == solid
            )
        else:
            selected = (self._path_index(path),)
        if not selected:
            raise ValueError(
                f"PreparedBRepQuery.solid must have assembly occurrences; got solid {solid} with none"
            )
        points_ = _points(points)
        if budget is not None and isinstance(points_, Tracer):
            raise ValueError(
                "PreparedBRepQuery.budget must be None during traced execution; got mutable BRepQueryBudget"
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
            owner._admit_containment(
                points_.shape[0], solids, budget, coordinate_budget=coordinate_budget
            )
        results = []
        for index in selected:
            if self.world_query is not None:
                results.append(
                    self.world_query._contains_definition(
                        points_,
                        self.world_solids[index],
                        budget=budget,
                        preparations=preparations,
                        coordinate_budget=coordinate_budget,
                    )
                )
                continue
            occurrence = self.occurrences[index]
            rotation, translation = self.placement(index)
            preparation = (
                self.prepare_placed_query(
                    index, self.solid_faces[occurrence.solid], budget=coordinate_budget
                )
                if self.certify_on_host and not isinstance(points_, Tracer)
                else None
            )
            if preparation is not None:
                results.append(
                    classify_host_containment(
                        self,
                        np.asarray(points_),
                        occurrence.solid,
                        budget=budget,
                        affine_preparation=preparation,
                        preparations=preparations,
                        coordinate_budget=coordinate_budget,
                    )
                )
            else:
                local = (points_ - translation) @ rotation
                results.append(
                    self._contains_definition(
                        local,
                        occurrence.solid,
                        budget=budget,
                        preparations=preparations,
                        coordinate_budget=coordinate_budget,
                    )
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
        raise ValueError(
            f"PreparedBRepQuery.points must have shape (n > 0, 3); got {values.shape}"
        )
    if not isinstance(values, Tracer) and not np.all(np.isfinite(np.asarray(values))):
        raise ValueError(
            "PreparedBRepQuery.points must be finite; got nonfinite coordinates"
        )
    return values


@checked
def prepare_brep_query(
    model: BRepModel,
    /,
    *,
    policy: BRepQueryPolicy | None = None,
    projection_policy: BRepProjectionPolicy | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    preparations: TrimRequestPreparations | None = None,
    certify_on_host: bool = True,
) -> PreparedBRepQuery:
    """Prepare native queries, reusing supplied host request evidence and tickets."""
    policy_ = BRepQueryPolicy() if policy is None else policy
    projection_ = (
        BRepProjectionPolicy() if projection_policy is None else projection_policy
    )
    return PreparedBRepQuery(
        model,
        policy_,
        projection_,
        budget=budget,
        preparations=preparations,
        certify_on_host=certify_on_host,
    )


# ------------------------------------------------------------ projection


__all__ = [
    "BRepQueryPolicy",
    "BRepSurfaceQueryResult",
    "PreparedBRepQuery",
    "prepare_brep_query",
]

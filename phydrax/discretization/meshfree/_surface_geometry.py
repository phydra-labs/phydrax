# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Declared embedded curve/sheet geometry; sampled diagnostics stay estimates."""

from __future__ import annotations

from enum import IntEnum
from itertools import product
from math import comb
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...ein import contract
from ...geometry._capabilities import GeometryCapability
from ...geometry._certificate import FieldRegularity
from ...geometry._contracts import CompiledGeometry
from ...integration._rules import GaussLegendreRule
from ...linalg import (
    DenseLinearOperator,
    FailurePolicy,
    orthonormal_frame,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from ...metrix._ambient import RegularLevelSetManifold
from ...metrix._embedded import EmbeddedChart
from ...sparse import RowRelation
from ...typing import Bool, ConvertibleToArray, Dim, Float, Int32, parse
from ._stencils import weighted_svd_factors


SurfaceGeometrySource: TypeAlias = Literal["implicit", "chart", "sample-estimate"]
SurfaceBoundarySource: TypeAlias = Literal["declared", "chart"]

# Declared (intrinsic, ambient) pairs: curves in R2/R3 and sheets in R3.
_ADMITTED_DIMENSIONS = ((1, 2), (1, 3), (2, 3))


class SurfaceNodeDim(Dim):
    pass


class SurfaceAmbientDim(Dim):
    pass


class SurfaceIntrinsicDim(Dim):
    pass


class SurfaceCodimensionDim(Dim):
    pass


class SurfaceBoundaryNodeDim(Dim):
    pass


class SurfaceGeometryStatus(IntEnum):
    VALID = 0
    DEGENERATE = 1
    PROJECTION_FAILURE = 2
    ORIENTATION_FAILURE = 3
    TUBE_FAILURE = 4
    SUPPORT_INVALID = 5


def admitted_dimensions(intrinsic_dimension: int, ambient_dimension: int) -> None:
    """Refuse undeclared embeddings; equal array sizes never imply a dimension."""
    if (intrinsic_dimension, ambient_dimension) not in _ADMITTED_DIMENSIONS:
        raise ValueError(
            "Declared embedding must be a curve in R2/R3 or a sheet in R3; "
            f"got intrinsic={intrinsic_dimension}, ambient={ambient_dimension}."
        )


@final
class SurfaceBoundary(StrictModule):
    """Declared open-boundary quadrature: node measures and weighted conormals.

    ``weighted_conormals[b]`` is the sum of measure-weighted outward unit
    conormals of every boundary piece owned by node ``b``; a corner therefore
    retains both sides instead of an averaged direction. For curves the boundary
    is a point set and ``measures`` are counting weights.
    """

    __strict_contract__ = True
    nodes: Int32[SurfaceBoundaryNodeDim]
    measures: Float[SurfaceBoundaryNodeDim]
    weighted_conormals: Float[SurfaceBoundaryNodeDim, SurfaceAmbientDim]
    source: SurfaceBoundarySource = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        measures: ArrayLike,
        weighted_conormals: ArrayLike,
        *,
        source: SurfaceBoundarySource = "declared",
        boundary_id: str = "surface-boundary",
    ) -> None:
        source_ = parse(source, SurfaceBoundarySource, "source")
        indices = np.asarray(nodes)
        weights = np.asarray(measures)
        conormals = np.asarray(weighted_conormals)
        if (
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("Boundary nodes must be unique nonnegative integers.")
        if (
            weights.shape != indices.shape
            or not np.issubdtype(weights.dtype, np.floating)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("Boundary measures must be finite, positive and per node.")
        if (
            conormals.ndim != 2
            or conormals.shape[0] != indices.size
            or conormals.shape[1] not in (2, 3)
            or not np.all(np.isfinite(conormals))
            or np.any(np.linalg.norm(conormals, axis=-1) == 0)
        ):
            raise ValueError("Boundary conormals must be finite nonzero (B,A) vectors.")
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("boundary_id must be a nonempty identity.")
        self.nodes = jnp.asarray(indices, dtype=jnp.int32)
        self.measures = jnp.asarray(weights)
        self.weighted_conormals = jnp.asarray(conormals, dtype=self.measures.dtype)
        self.source, self.boundary_id = source_, boundary_id

    @property
    def conormals(self) -> Array:
        """Unit outward conormal directions of each boundary node."""
        return self.weighted_conormals / jnp.linalg.norm(
            self.weighted_conormals, axis=-1, keepdims=True
        )

    def mask(self, count: int) -> Array:
        """Boundary indicator; capacity is validated once at plan admission."""
        return jnp.zeros((count,), dtype=jnp.bool_).at[self.nodes].set(True)

    def dense_conormals(self, count: int) -> Array:
        """Per-node weighted conormals; interior rows are zero."""
        return (
            jnp.zeros((count, self.weighted_conormals.shape[1]), self.measures.dtype)
            .at[self.nodes]
            .set(self.weighted_conormals)
        )


@final
class SurfaceGeometryEvidence(StrictModule):
    __strict_contract__ = True
    valid: Bool[SurfaceNodeDim]
    status: Int32[SurfaceNodeDim]
    projection_valid: Bool[SurfaceNodeDim]
    tube_valid: Bool[SurfaceNodeDim]
    trust_margin: Float[SurfaceNodeDim]
    projector_comparison_estimate: Float[SurfaceNodeDim]
    sheet_separation_estimate: Float[SurfaceNodeDim]
    fit_residual: Float[SurfaceNodeDim]
    normal_error_estimate: Float[SurfaceNodeDim]
    curvature_error_estimate: Float[SurfaceNodeDim]
    regularity_margin: Float[SurfaceNodeDim]
    gauge_margin: Float[SurfaceNodeDim]
    boundary: Bool[SurfaceNodeDim]
    boundary_conormal_residual: Float[SurfaceNodeDim]
    orientation_consistent: Bool[SurfaceNodeDim]
    tube_certified: bool = eqx.field(static=True)
    projection_to_source: bool = eqx.field(static=True)
    estimate_only: bool = eqx.field(static=True)
    source: SurfaceGeometrySource = eqx.field(static=True)


@final
class SurfaceGeometryEvaluation(StrictModule):
    """Gauge frames plus gauge-invariant projector and second fundamental form.

    ``second_fundamental_form[n, a, b, c]`` is the ambient normal-valued form
    ``II(e_b, e_c)_a`` restricted to the tangent space; it does not depend on
    the discrete tangent/normal gauge. ``chart_*`` describe the local graph
    chart ``x(u) = p + chart_frames u + chart_normal_frames h(u)``.
    """

    __strict_contract__ = True
    points: Float[SurfaceNodeDim, SurfaceAmbientDim]
    tangent_frames: Float[SurfaceNodeDim, SurfaceAmbientDim, SurfaceIntrinsicDim]
    normal_frames: Float[SurfaceNodeDim, SurfaceAmbientDim, SurfaceCodimensionDim]
    chart_frames: Float[SurfaceNodeDim, SurfaceAmbientDim, SurfaceIntrinsicDim]
    chart_normal_frames: Float[SurfaceNodeDim, SurfaceAmbientDim, SurfaceCodimensionDim]
    projectors: Float[SurfaceNodeDim, SurfaceAmbientDim, SurfaceAmbientDim]
    second_fundamental_form: Float[
        SurfaceNodeDim, SurfaceAmbientDim, SurfaceAmbientDim, SurfaceAmbientDim
    ]
    chart_slope: Float[SurfaceNodeDim, SurfaceCodimensionDim, SurfaceIntrinsicDim]
    chart_hessian: Float[
        SurfaceNodeDim, SurfaceCodimensionDim, SurfaceIntrinsicDim, SurfaceIntrinsicDim
    ]
    evidence: SurfaceGeometryEvidence

    @property
    def intrinsic_dimension(self) -> int:
        return self.tangent_frames.shape[-1]

    @property
    def ambient_dimension(self) -> int:
        return self.points.shape[-1]

    @property
    def codimension(self) -> int:
        return self.normal_frames.shape[-1]

    def _require_hypersurface(self, name: str) -> None:
        if self.codimension != 1:
            raise ValueError(f"{name} is defined for codimension-one geometry only.")

    @property
    def normals(self) -> Array:
        """Oriented unit normals of a codimension-one curve or sheet."""
        self._require_hypersurface("normals")
        return self.normal_frames[..., 0]

    @property
    def curvature_tensor(self) -> Array:
        """Ambient shape operator (tangential gradient of the oriented normal)."""
        self._require_hypersurface("curvature_tensor")
        return -contract("na,nabc->nbc", self.normals, self.second_fundamental_form)

    @property
    def mean_curvature_vector(self) -> Array:
        """Trace of the second fundamental form, equal to Laplace-Beltrami of x."""
        return jnp.trace(self.second_fundamental_form, axis1=-2, axis2=-1)

    @property
    def mean_curvature(self) -> Array:
        """Trace of the shape operator; positive for outward-oriented spheres."""
        return jnp.trace(self.curvature_tensor, axis1=-2, axis2=-1)


def _oriented_frames(
    projector: Array, intrinsic: int, orientation: Array
) -> tuple[Array, Array, Array, Array]:
    """Discrete tangent/normal gauge from a projector with explicit switch margin.

    Columns of the projector with the largest tangential weight seed the native
    orthonormal frame; ``gauge_margin`` is the gap to the next candidate axis,
    so a frame switch is visible rather than silently differentiated through.
    ``orientation`` orients the normal of a codimension-one geometry, else the
    tangent of a curve; the frame keeps positive ambient orientation.
    """
    ambient = projector.shape[-1]
    weights = jnp.diagonal(projector, axis1=-2, axis2=-1)
    order = jnp.argsort(-weights, axis=-1, stable=True)
    sorted_weights = jnp.take_along_axis(weights, order, axis=-1)
    seeds = jnp.take_along_axis(projector, order[:, None, :intrinsic], axis=2)
    frame = orthonormal_frame(seeds)
    tangents, normals = frame.tangents, frame.normal_basis
    margin = sorted_weights[:, intrinsic - 1] - sorted_weights[:, intrinsic]
    if ambient - intrinsic == 1:
        sign = jnp.where(jnp.sum(normals[..., 0] * orientation, axis=-1) < 0, -1.0, 1.0)
        normals = normals * sign[:, None, None]
        tangents = tangents.at[..., -1].multiply(sign[:, None])
    else:
        sign = jnp.where(jnp.sum(tangents[..., 0] * orientation, axis=-1) < 0, -1.0, 1.0)
        tangents = tangents * sign[:, None, None]
        normals = normals.at[..., -1].multiply(sign[:, None])
    return tangents, normals, margin, frame.successful


def _orientation_field(tangents: Array, normals: Array) -> Array:
    """Scientific orientation vector: normal of hypersurfaces, else curve tangent."""
    return normals[..., 0] if normals.shape[-1] == 1 else tangents[..., 0]


def _ambient_second_form(
    normal_hessian: Array, dual: Array, normal_projector: Array
) -> Array:
    """II[a,b,c] = sum_ij (P_N d_ij x)_a dual_bi dual_cj."""
    projected = contract("nab,nbij->naij", normal_projector, normal_hessian)
    return contract("naij,nbi,ncj->nabc", projected, dual, dual)


def _chart_components(second_form: Array, tangents: Array, normals: Array) -> Array:
    return contract("nac,nabd,nbi,ndj->ncij", normals, second_form, tangents, tangents)


def _surrounding_support(
    points: Array, relation: RowRelation, frames: Array, boundary: Array
) -> Array:
    """Interior chart surround; declared boundary rows admit one-sided support."""
    offsets = points[relation.source_indices] - points[:, None, :]
    xy = contract("nka,nai->nki", offsets, frames)
    nonzero = relation.valid & (
        jnp.linalg.norm(xy, axis=-1) > jnp.finfo(points.dtype).eps
    )
    count = jnp.sum(nonzero, axis=-1)
    if frames.shape[-1] == 1:
        coordinate = xy[..., 0]
        surround = jnp.any(nonzero & (coordinate > 0), axis=-1) & jnp.any(
            nonzero & (coordinate < 0), axis=-1
        )
    else:
        angles = jnp.sort(
            jnp.where(nonzero, jnp.arctan2(xy[..., 1], xy[..., 0]), jnp.inf), axis=-1
        )
        last = jnp.take_along_axis(angles, jnp.maximum(count - 1, 0)[:, None], axis=-1)[
            :, 0
        ]
        gaps = jnp.diff(angles, axis=-1)
        gaps = jnp.where(
            jnp.arange(gaps.shape[-1])[None, :] < count[:, None] - 1, gaps, 0
        )
        maximum = jnp.maximum(jnp.max(gaps, axis=-1), angles[:, 0] + 2 * jnp.pi - last)
        surround = (count >= 3) & jnp.isfinite(maximum) & (maximum < jnp.pi - 1e-6)
    return surround | (boundary & (count >= frames.shape[-1] + 1))


def _boundary_fields(
    boundary: SurfaceBoundary | None, count: int, ambient: int, dtype: jnp.dtype
) -> tuple[Array, Array]:
    if boundary is None:
        return jnp.zeros((count,), dtype=jnp.bool_), jnp.zeros((count, ambient), dtype)
    if boundary.weighted_conormals.shape[1] != ambient:
        raise ValueError("Boundary conormals must live in the declared ambient space.")
    return boundary.mask(count), boundary.dense_conormals(count).astype(dtype)


def _conormal_residual(
    conormals: Array, boundary_mask: Array, projectors: Array
) -> Array:
    norm = jnp.linalg.norm(conormals, axis=-1)
    tangential = contract("nab,nb->na", projectors, conormals)
    residual = jnp.linalg.norm(conormals - tangential, axis=-1) / jnp.where(
        norm > 0, norm, 1
    )
    return jnp.where(boundary_mask, residual, 0)


def _status(
    valid: Array,
    projection_valid: Array,
    orientation: Array,
    tube_failure: Array,
) -> Array:
    return jnp.where(
        valid,
        int(SurfaceGeometryStatus.VALID),
        jnp.where(
            ~projection_valid,
            int(SurfaceGeometryStatus.PROJECTION_FAILURE),
            jnp.where(
                ~orientation,
                int(SurfaceGeometryStatus.ORIENTATION_FAILURE),
                jnp.where(
                    tube_failure,
                    int(SurfaceGeometryStatus.TUBE_FAILURE),
                    int(SurfaceGeometryStatus.DEGENERATE),
                ),
            ),
        ),
    ).astype(jnp.int32)


def _validate_identity(closed: bool, geometry_id: str) -> None:
    if (
        not isinstance(closed, bool)
        or not isinstance(geometry_id, str)
        or not geometry_id
    ):
        raise ValueError(
            "closed must be bool and geometry_id must be a nonempty source identity."
        )


@final
class ImplicitSurfaceGeometry(StrictModule):
    """Native source projection and level-set manifold tangent geometry.

    The intrinsic dimension is the source's declared ambient dimension minus its
    declared codimension (curves in R2/R3, sheets in R3). ``closed`` is a source
    declaration; an open piece needs a declared ``SurfaceBoundary`` at the plan.
    A tube radius is admitted only as an explicit source-domain guarantee.
    """

    source: CompiledGeometry | RegularLevelSetManifold
    closed: bool = eqx.field(static=True)
    certified_tube_radius: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        source: CompiledGeometry | RegularLevelSetManifold,
        *,
        closed: bool = True,
        certified_tube_radius: float | None = None,
        tolerance: float = 1e-7,
        geometry_id: str = "implicit-surface",
    ) -> None:
        if isinstance(source, CompiledGeometry):
            if (
                source.ambient_dimension not in (2, 3)
                or source.intrinsic_dimension
                not in (source.ambient_dimension - 1, source.ambient_dimension)
                or source.field_certificate.regularity is FieldRegularity.NONSMOOTH
            ):
                raise ValueError("Surface requires a smooth boundary source in R2/R3.")
            source.require(GeometryCapability.CLOSEST_POINT)
            # A radial signed-distance field is nonsmooth at its medial set,
            # although its boundary is smooth. Native curvature admission
            # distinguishes that case from sharp sources without a new calculus.
            if source.field_certificate.regularity is not FieldRegularity.SMOOTH:
                source.require(GeometryCapability.CONTACT_CURVATURE)
            ambient, codimension = source.ambient_dimension, 1
        elif isinstance(source, RegularLevelSetManifold) and len(source.point_shape) == 1:
            ambient, codimension = source.point_shape[0], source.codimension
        else:
            raise TypeError("source must be a native Euclidean curve or surface source.")
        admitted_dimensions(ambient - codimension, ambient)
        if tolerance <= 0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive.")
        if certified_tube_radius is not None and (
            not np.isfinite(certified_tube_radius) or certified_tube_radius <= 0
        ):
            raise ValueError("certified_tube_radius must be positive.")
        _validate_identity(closed, geometry_id)
        self.source, self.closed = source, closed
        self.certified_tube_radius, self.tolerance = certified_tube_radius, tolerance
        self.geometry_id = geometry_id
        self.intrinsic_dimension = ambient - codimension
        self.ambient_dimension = ambient

    def _project(
        self, points: Array
    ) -> tuple[Array, Array, Array, RegularLevelSetManifold]:
        """Projected points, validity, native constraint orientation and manifold."""
        if isinstance(self.source, CompiledGeometry):
            projected = self.source.closest_point(points)
            coordinates = projected.closest_point.astype(points.dtype)
            valid = (
                projected.unique & projected.regular & projected.normal_coordinate_valid
            )
            source = self.source

            def constraint(point: Array) -> Array:
                return source.boundary_field(point[None, :])[0][None]

            manifold = RegularLevelSetManifold(
                constraint,
                ambient_dimension=self.ambient_dimension,
                codimension=1,
                tolerance=self.tolerance,
            )
            if self.source.field_certificate.regularity is not FieldRegularity.SMOOTH:
                valid &= self.source.contact_curvature(coordinates).valid
            return (
                coordinates,
                valid,
                projected.oriented_normal.astype(points.dtype),
                manifold,
            )
        manifold = self.source
        projected = manifold.project_normal(points)
        jacobian = projected.geometry.constraint_jacobian
        if manifold.codimension == 1:
            orientation = jacobian[:, 0, :] * manifold.orientation_sign
        else:
            # A space curve is the transversal intersection of two level sets;
            # the ordered constraint pair orients its tangent.
            orientation = (
                jnp.cross(jacobian[:, 0, :], jacobian[:, 1, :])
                * manifold.orientation_sign
            )
        valid = projected.valid & (projected.residual <= self.tolerance)
        return projected.points, valid, orientation, manifold

    def evaluate(
        self,
        points: Array,
        relation: RowRelation,
        *,
        reference: SurfaceGeometryEvaluation | None = None,
        require_tube: bool = False,
        boundary: SurfaceBoundary | None = None,
    ) -> SurfaceGeometryEvaluation:
        if points.ndim != 2 or points.shape[1] != self.ambient_dimension:
            raise ValueError("Points must match the source's declared ambient dimension.")
        count, ambient = points.shape
        intrinsic = self.intrinsic_dimension
        codimension = ambient - intrinsic
        coordinates, projection_valid, native_orientation, manifold = self._project(
            points
        )
        local = manifold.local_geometry(coordinates)
        identity = jnp.eye(ambient, dtype=points.dtype)
        metric_ok = (
            jnp.max(jnp.abs(local.metric - identity), axis=(-2, -1)) <= self.tolerance
        )
        projector = local.tangent_projector.astype(points.dtype)
        tangents, normals, gauge, frame_ok = _oriented_frames(
            projector, intrinsic, native_orientation
        )
        constraint_jacobian = local.constraint_jacobian.astype(points.dtype)
        constraint_hessian = jax.vmap(jax.hessian(manifold.constraint))(coordinates)
        # Normal-space pseudo-inverse of the constraint Jacobian, J^+ = N (J N)^-1.
        transverse = contract("nka,nam->nkm", constraint_jacobian, normals)
        pseudo = solve_small_linear(
            SmallLinearSolvePlan(codimension),
            jnp.swapaxes(transverse, -1, -2),
            jnp.swapaxes(normals, -1, -2),
        )
        tangential_hessian = contract(
            "nab,nkbc,ncd->nkad", projector, constraint_hessian, projector
        )
        second_form = -contract("nka,nkbc->nabc", pseudo.value, tangential_hessian)
        hessian = _chart_components(second_form, tangents, normals)
        boundary_mask, conormals = _boundary_fields(
            boundary, count, ambient, points.dtype
        )
        conormal_residual = _conormal_residual(conormals, boundary_mask, projector)
        distance = jnp.linalg.norm(coordinates - points, axis=-1)
        radius = self.certified_tube_radius
        tube_valid = (
            jnp.zeros((count,), dtype=jnp.bool_) if radius is None else distance < radius
        )
        margin = (
            jnp.zeros((count,), dtype=points.dtype)
            if radius is None
            else radius - distance
        )
        orientation_field = _orientation_field(tangents, normals)
        consistent = (
            jnp.ones((count,), dtype=jnp.bool_)
            if reference is None
            else jnp.sum(
                orientation_field
                * _orientation_field(reference.tangent_frames, reference.normal_frames),
                axis=-1,
            )
            > 0
        )
        valid = (
            projection_valid
            & local.valid
            & metric_ok
            & frame_ok
            & pseudo.successful
            & consistent
            & (conormal_residual <= jnp.sqrt(self.tolerance))
            & _surrounding_support(coordinates, relation, tangents, boundary_mask)
            & jnp.all(jnp.isfinite(coordinates), axis=-1)
            & jnp.all(jnp.isfinite(second_form), axis=(-3, -2, -1))
        )
        if require_tube:
            valid &= tube_valid
        residual = jnp.max(jnp.abs(jax.vmap(manifold.constraint)(coordinates)), axis=-1)
        nan = jnp.full((count,), jnp.nan, dtype=points.dtype)
        evidence = SurfaceGeometryEvidence(
            valid=valid,
            status=_status(
                valid, projection_valid, consistent, require_tube & ~tube_valid
            ),
            projection_valid=projection_valid,
            tube_valid=tube_valid,
            trust_margin=margin,
            projector_comparison_estimate=jnp.max(
                jnp.abs(projector - contract("nai,nbi->nab", tangents, tangents)),
                axis=(-2, -1),
            ),
            sheet_separation_estimate=nan,
            fit_residual=residual,
            normal_error_estimate=jnp.zeros((count,), dtype=points.dtype),
            curvature_error_estimate=jnp.zeros((count,), dtype=points.dtype),
            regularity_margin=local.rank_margin.astype(points.dtype),
            gauge_margin=gauge,
            boundary=boundary_mask,
            boundary_conormal_residual=conormal_residual,
            orientation_consistent=consistent,
            tube_certified=radius is not None,
            projection_to_source=True,
            estimate_only=False,
            source="implicit",
        )
        return SurfaceGeometryEvaluation(
            points=coordinates,
            tangent_frames=tangents,
            normal_frames=normals,
            chart_frames=tangents,
            chart_normal_frames=normals,
            projectors=projector,
            second_fundamental_form=second_form,
            chart_slope=jnp.zeros((count, codimension, intrinsic), dtype=points.dtype),
            chart_hessian=hessian,
            evidence=evidence,
        )


@final
class ChartSurfaceGeometry(StrictModule):
    """Authoritative embedded-chart source: each sample carries its preimage.

    Points, tangent bases, metric and second fundamental form come from the
    declared ``metrix.EmbeddedChart`` jets, never from sample fitting. Rank and
    conditioning of each chart Jacobian come from the native orthonormal frame.
    ``orientation`` multiplies the chart-induced normal (hypersurfaces) or
    tangent (curves).
    """

    __strict_contract__ = True
    charts: tuple[EmbeddedChart, ...]
    chart_indices: Int32[SurfaceNodeDim]
    chart_coordinates: Float[SurfaceNodeDim, SurfaceIntrinsicDim]
    closed: bool = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        charts: tuple[EmbeddedChart, ...] | EmbeddedChart,
        chart_coordinates: ArrayLike,
        *,
        chart_indices: ArrayLike | None = None,
        closed: bool,
        orientation: int = 1,
        tolerance: float = 1e-9,
        condition_limit: float = 1e8,
        geometry_id: str = "chart-surface",
    ) -> None:
        charts_ = (charts,) if isinstance(charts, EmbeddedChart) else tuple(charts)
        if not charts_ or any(not isinstance(chart, EmbeddedChart) for chart in charts_):
            raise TypeError("charts must contain metrix EmbeddedChart values.")
        intrinsic = charts_[0].chart.dimension
        ambient = charts_[0].ambient_dimension
        if any(
            chart.chart.dimension != intrinsic or chart.ambient_dimension != ambient
            for chart in charts_
        ):
            raise ValueError("Every chart must declare the same embedding dimensions.")
        admitted_dimensions(intrinsic, ambient)
        coordinates = np.asarray(chart_coordinates)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != intrinsic
            or not np.issubdtype(coordinates.dtype, np.floating)
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("chart_coordinates must be finite floating (N, intrinsic).")
        indices = (
            np.zeros((coordinates.shape[0],), dtype=np.int32)
            if chart_indices is None
            else np.asarray(chart_indices)
        )
        if (
            indices.shape != coordinates.shape[:1]
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(charts_))
        ):
            raise ValueError("chart_indices must select one declared chart per sample.")
        if orientation not in (-1, 1):
            raise ValueError("orientation must be +1 or -1.")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        if not np.isfinite(condition_limit) or condition_limit <= 1:
            raise ValueError("condition_limit must be finite and exceed one.")
        _validate_identity(closed, geometry_id)
        self.charts = charts_
        self.chart_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.chart_coordinates = jnp.asarray(coordinates)
        self.closed, self.orientation = closed, orientation
        self.tolerance, self.condition_limit = tolerance, condition_limit
        self.geometry_id = geometry_id
        self.intrinsic_dimension, self.ambient_dimension = intrinsic, ambient

    def jets(
        self, chart_indices: Array, coordinates: Array
    ) -> tuple[Array, Array, Array]:
        """Embedding value, tangent basis and Hessian for chart-located samples."""
        branches = tuple(
            (
                lambda u, chart=chart: (
                    chart(u),
                    chart.tangent_basis(u),
                    chart.embedding_hessian(u),
                )
            )
            for chart in self.charts
        )
        return jax.vmap(lambda index, u: jax.lax.switch(index, branches, u))(
            chart_indices, coordinates
        )

    def embed(self, chart_indices: Array, coordinates: Array) -> Array:
        return self.jets(chart_indices, coordinates)[0]

    def evaluate(
        self,
        points: Array,
        relation: RowRelation,
        *,
        reference: SurfaceGeometryEvaluation | None = None,
        require_tube: bool = False,
        boundary: SurfaceBoundary | None = None,
    ) -> SurfaceGeometryEvaluation:
        count, ambient = points.shape
        if ambient != self.ambient_dimension or count != self.chart_coordinates.shape[0]:
            raise ValueError("Points must match the declared chart samples.")
        intrinsic = self.intrinsic_dimension
        coordinates = self.chart_coordinates.astype(points.dtype)
        embedded, basis, embedding_hessian = self.jets(self.chart_indices, coordinates)
        frame = orthonormal_frame(basis)
        projector = contract("nai,nbi->nab", frame.tangents, frame.tangents)
        induced = (
            frame.normal_basis[..., 0] if ambient - intrinsic == 1 else basis[..., 0]
        )
        tangents, normals, gauge, frame_ok = _oriented_frames(
            projector, intrinsic, self.orientation * induced
        )
        metric = contract("nai,naj->nij", basis, basis)
        dual = jnp.swapaxes(
            solve_small_linear(
                SmallLinearSolvePlan(intrinsic), metric, jnp.swapaxes(basis, -1, -2)
            ).value,
            -1,
            -2,
        )
        identity = jnp.eye(ambient, dtype=points.dtype)
        second_form = _ambient_second_form(embedding_hessian, dual, identity - projector)
        boundary_mask, conormals = _boundary_fields(
            boundary, count, ambient, points.dtype
        )
        conormal_residual = _conormal_residual(conormals, boundary_mask, projector)
        projection_residual = jnp.linalg.norm(embedded - points, axis=-1)
        projection_valid = projection_residual <= self.tolerance * jnp.maximum(
            1, jnp.linalg.norm(points, axis=-1)
        )
        orientation_field = _orientation_field(tangents, normals)
        consistent = (
            jnp.ones((count,), dtype=jnp.bool_)
            if reference is None
            else jnp.sum(
                orientation_field
                * _orientation_field(reference.tangent_frames, reference.normal_frames),
                axis=-1,
            )
            > 0
        )
        regular = (
            frame.successful
            & frame_ok
            & (frame.condition_estimate <= self.condition_limit)
        )
        valid = (
            projection_valid
            & regular
            & consistent
            & ~jnp.asarray(require_tube)
            & (conormal_residual <= jnp.sqrt(self.tolerance))
            & _surrounding_support(embedded, relation, tangents, boundary_mask)
            & jnp.all(jnp.isfinite(second_form), axis=(-3, -2, -1))
        )
        zeros = jnp.zeros((count,), dtype=points.dtype)
        evidence = SurfaceGeometryEvidence(
            valid=valid,
            status=_status(
                valid,
                projection_valid,
                consistent,
                jnp.full((count,), require_tube, dtype=jnp.bool_),
            ),
            projection_valid=projection_valid,
            tube_valid=jnp.zeros((count,), dtype=jnp.bool_),
            trust_margin=zeros,
            projector_comparison_estimate=zeros,
            sheet_separation_estimate=jnp.full((count,), jnp.nan, dtype=points.dtype),
            fit_residual=projection_residual,
            normal_error_estimate=zeros,
            curvature_error_estimate=zeros,
            regularity_margin=frame.regularity_margin.astype(points.dtype),
            gauge_margin=gauge,
            boundary=boundary_mask,
            boundary_conormal_residual=conormal_residual,
            orientation_consistent=consistent,
            tube_certified=False,
            projection_to_source=True,
            estimate_only=False,
            source="chart",
        )
        return SurfaceGeometryEvaluation(
            points=embedded,
            tangent_frames=tangents,
            normal_frames=normals,
            chart_frames=tangents,
            chart_normal_frames=normals,
            projectors=projector,
            second_fundamental_form=second_form,
            chart_slope=jnp.zeros(
                (count, ambient - intrinsic, intrinsic), dtype=points.dtype
            ),
            chart_hessian=_chart_components(second_form, tangents, normals),
            evidence=evidence,
        )

    def box_boundary(
        self,
        lower: ConvertibleToArray,
        upper: ConvertibleToArray,
        *,
        chart: int = 0,
        quadrature_order: int = 8,
        tolerance: float = 1e-12,
        boundary_id: str = "chart-box-boundary",
    ) -> SurfaceBoundary:
        """Authoritative boundary quadrature of a box patch in one declared chart.

        Nodes on a face ``u_k = lower_k|upper_k`` own the arc of that face up to
        the midpoints with their face neighbors; the arc length is integrated by
        Gauss-Legendre on the chart Jacobian. Conormals are the unit tangent
        directions orthogonal to the face, pointing toward increasing ``±u_k``.
        """
        low = np.asarray(lower, dtype=np.float64).reshape(-1)
        high = np.asarray(upper, dtype=np.float64).reshape(-1)
        dimension = self.intrinsic_dimension
        if low.shape != (dimension,) or high.shape != (dimension,) or np.any(high <= low):
            raise ValueError("Box bounds must be ordered and match the chart dimension.")
        if not 0 <= chart < len(self.charts):
            raise ValueError("chart must select a declared chart.")
        coordinates = np.asarray(self.chart_coordinates, dtype=np.float64)
        in_chart = np.asarray(self.chart_indices) == chart
        scale = np.maximum(high - low, 1.0) * tolerance
        outside = (coordinates < low - scale) | (coordinates > high + scale)
        if np.any(in_chart & np.any(outside, axis=1)):
            raise ValueError("Chart samples lie outside the declared box patch.")
        rule = GaussLegendreRule(quadrature_order).data()
        nodes_ref, weights_ref = np.asarray(rule.nodes), np.asarray(rule.weights)
        measures: dict[int, float] = {}
        conormals: dict[int, np.ndarray] = {}

        def face_conormal(u: np.ndarray, axis: int) -> np.ndarray:
            _, basis, _ = self.jets(
                jnp.full((u.shape[0],), chart, dtype=jnp.int32), jnp.asarray(u)
            )
            basis_ = np.asarray(basis)
            direction = basis_[:, :, axis]
            if dimension == 2:
                along = basis_[:, :, 1 - axis]
                along = along / np.linalg.norm(along, axis=-1, keepdims=True)
                direction = (
                    direction - np.sum(direction * along, axis=-1)[:, None] * along
                )
            return direction / np.linalg.norm(direction, axis=-1, keepdims=True)

        for axis in range(dimension):
            for sign, value in ((-1.0, low[axis]), (1.0, high[axis])):
                face = np.flatnonzero(
                    in_chart & (np.abs(coordinates[:, axis] - value) <= scale[axis])
                )
                if face.size == 0:
                    continue
                direction = sign * face_conormal(coordinates[face], axis)
                if dimension == 1:
                    lengths = np.ones(face.size)
                else:
                    free = 1 - axis
                    order = np.argsort(coordinates[face, free], kind="stable")
                    face, direction = face[order], direction[order]
                    position = coordinates[face, free]
                    middle = (position[1:] + position[:-1]) / 2
                    left = np.concatenate(([low[free]], middle))
                    right = np.concatenate((middle, [high[free]]))
                    centers = (left + right) / 2
                    halves = (right - left) / 2
                    samples = centers[:, None] + halves[:, None] * nodes_ref[None, :]
                    u = np.empty((samples.size, 2))
                    u[:, axis] = value
                    u[:, free] = samples.reshape(-1)
                    _, basis, _ = self.jets(
                        jnp.full((u.shape[0],), chart, dtype=jnp.int32), jnp.asarray(u)
                    )
                    speed = np.linalg.norm(np.asarray(basis)[:, :, free], axis=-1)
                    lengths = (speed.reshape(samples.shape) * weights_ref[None, :]).sum(
                        axis=1
                    ) * halves
                for node, length, normal in zip(face, lengths, direction, strict=True):
                    measures[int(node)] = measures.get(int(node), 0.0) + float(length)
                    conormals[int(node)] = (
                        conormals.get(int(node), np.zeros(self.ambient_dimension))
                        + length * normal
                    )
        if not measures:
            raise ValueError("No chart samples lie on the declared box boundary.")
        ordered = np.asarray(sorted(measures), dtype=np.int32)
        return SurfaceBoundary(
            ordered,
            np.asarray([measures[int(node)] for node in ordered]),
            np.stack([conormals[int(node)] for node in ordered]),
            source="chart",
            boundary_id=boundary_id,
        )


def _height_basis(
    intrinsic: int, degree: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monomials 1<=|a|<=degree, and the rows of linear and quadratic terms.

    Pure host enumeration (static metadata), safe under any JAX trace.
    """
    rows = sorted(
        (
            index
            for index in product(range(degree + 1), repeat=intrinsic)
            if 1 <= sum(index) <= degree
        ),
        key=lambda index: (sum(index), tuple(-value for value in index)),
    )
    position = {index: row for row, index in enumerate(rows)}
    unit = [tuple(int(i == k) for k in range(intrinsic)) for i in range(intrinsic)]
    linear = np.asarray([position[unit[i]] for i in range(intrinsic)], dtype=np.int64)
    quadratic = np.asarray(
        [
            [
                position[tuple(a + b for a, b in zip(unit[i], unit[j], strict=True))]
                for j in range(intrinsic)
            ]
            for i in range(intrinsic)
        ],
        dtype=np.int64,
    )
    return np.asarray(rows, dtype=np.int32), linear, quadratic


def _fit_height(
    points: Array,
    relation: RowRelation,
    frames: Array,
    normals: Array,
    degree: int,
    rank_cutoff: float,
    minimum_neighbors: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Degree-p graph fit of the normal heights over the tangent chart.

    Returns slope (C,D), Hessian (C,D,D), residual, slope and Hessian standard
    errors propagated from the fit residual through the native pseudo-inverse,
    and fit admission. Errors are NaN when the fit has no redundant samples.
    """
    intrinsic = frames.shape[-1]
    exponents, linear, quadratic = _height_basis(intrinsic, degree)
    offsets = points[relation.source_indices] - points[:, None, :]
    xy = contract("nka,nab->nkb", offsets, frames)
    lengths = jnp.linalg.norm(xy, axis=-1)
    active = relation.valid & (lengths > jnp.finfo(points.dtype).eps)
    scale = jnp.max(jnp.where(active, lengths, 0), axis=-1)
    safe_scale = jnp.maximum(scale, jnp.finfo(points.dtype).tiny)
    uv = xy / safe_scale[:, None, None]
    design = jnp.prod(uv[:, :, None, :] ** jnp.asarray(exponents)[None, None], axis=-1)
    design = jnp.where(active[..., None], design, 0)
    heights = jnp.where(active[..., None], contract("nka,nac->nkc", offsets, normals), 0)
    factors, rank, condition, _ = weighted_svd_factors(
        design, jnp.ones(active.shape, dtype=points.dtype), active
    )
    coefficients = contract("nfk,nkc->ncf", factors, heights)
    residuals = jnp.where(
        active[..., None], contract("nkf,ncf->nkc", design, coefficients) - heights, 0
    )
    count = jnp.sum(active, axis=-1)
    features = exponents.shape[0]
    redundancy = count - features
    variance = jnp.sum(residuals * residuals, axis=(1, 2)) / jnp.where(
        redundancy > 0, redundancy * normals.shape[-1], 1
    )
    row_norms = jnp.linalg.norm(factors, axis=-1)
    slope_rows = row_norms[:, linear]
    quadratic_rows = row_norms[:, quadratic.reshape(-1)].reshape(
        (-1, intrinsic, intrinsic)
    )
    multiplicity = 1.0 + jnp.eye(intrinsic, dtype=points.dtype)
    slope_error = jnp.sqrt(variance * jnp.sum(slope_rows**2, axis=-1)) / safe_scale
    curvature_error = (
        jnp.sqrt(variance * jnp.sum((multiplicity * quadratic_rows) ** 2, axis=(-2, -1)))
        / safe_scale**2
    )
    unavailable = redundancy <= 0
    slope = coefficients[:, :, linear] / safe_scale[:, None, None]
    # d_ij h = (1 + delta_ij) c_{e_i + e_j} for monomials without factorial scaling.
    hessian = (
        coefficients[:, :, quadratic.reshape(-1)].reshape(
            (-1, normals.shape[-1], intrinsic, intrinsic)
        )
        * multiplicity
        / safe_scale[:, None, None, None] ** 2
    )
    residual = jnp.max(jnp.abs(residuals), axis=(1, 2))
    admitted = (
        (rank == features)
        & jnp.isfinite(condition)
        & (condition * rank_cutoff < 1)
        & (scale > 0)
        & (count >= minimum_neighbors)
    )
    return (
        slope,
        hessian,
        residual,
        jnp.where(unavailable, jnp.nan, slope_error),
        jnp.where(unavailable, jnp.nan, curvature_error),
        admitted,
    )


def _graph_geometry(
    frames: Array, normals: Array, slope: Array, hessian: Array
) -> tuple[Array, Array, Array, Array]:
    """Projector, ambient II, regularity margin and admission of the graph chart."""
    intrinsic = frames.shape[-1]
    basis = frames + contract("nac,ncd->nad", normals, slope)
    frame = orthonormal_frame(basis)
    projector = contract("nai,nbi->nab", frame.tangents, frame.tangents)
    metric = contract("nai,naj->nij", basis, basis)
    dual = jnp.swapaxes(
        solve_small_linear(
            SmallLinearSolvePlan(intrinsic), metric, jnp.swapaxes(basis, -1, -2)
        ).value,
        -1,
        -2,
    )
    identity = jnp.eye(frames.shape[1], dtype=frames.dtype)
    second = contract("nac,ncij->naij", normals, hessian)
    return (
        projector,
        _ambient_second_form(second, dual, identity - projector),
        frame.regularity_margin,
        frame.successful,
    )


@final
class SampledSurfaceGeometry(StrictModule):
    """Degree-p tangent-height reconstruction on a declared smooth curve or sheet.

    ``fit_degree`` (2..4) sets the local graph polynomial; ``oversampling``
    requires ``ceil(oversampling * features)`` distinct neighbors. Fit, normal
    and curvature errors are independent local estimates. Orientation is the
    normal of hypersurfaces or the tangent of space curves. Without a reference,
    orientability is checked by host graph propagation. No finite sample set
    certifies a tube, a projection to the true surface or global topology.
    """

    __strict_contract__ = True
    reference_normals: Float[SurfaceNodeDim, SurfaceAmbientDim] | None
    reference_tangents: Float[SurfaceNodeDim, SurfaceAmbientDim] | None
    declared_tube_radius: float | None = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    rank_cutoff: float = eqx.field(static=True)
    maximum_slope: float = eqx.field(static=True)
    fit_tolerance: float = eqx.field(static=True)
    fit_degree: int = eqx.field(static=True)
    oversampling: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        reference_normals: Array | None = None,
        *,
        reference_tangents: Array | None = None,
        intrinsic_dimension: int = 2,
        ambient_dimension: int = 3,
        declared_tube_radius: float | None = None,
        closed: bool = True,
        regularity: Literal["smooth"] = "smooth",
        rank_cutoff: float = 1e-9,
        maximum_slope: float = 0.5,
        fit_tolerance: float = 0.15,
        fit_degree: int = 2,
        oversampling: float = 1.0,
        geometry_id: str = "sampled-surface",
    ) -> None:
        parse(regularity, Literal["smooth"], "regularity")
        admitted_dimensions(intrinsic_dimension, ambient_dimension)
        codimension = ambient_dimension - intrinsic_dimension
        if declared_tube_radius is not None and (
            not np.isfinite(declared_tube_radius) or declared_tube_radius <= 0
        ):
            raise ValueError(
                "declared_tube_radius must be positive; it remains a declaration, not a certificate."
            )
        if min(rank_cutoff, maximum_slope, fit_tolerance) <= 0 or not np.all(
            np.isfinite((rank_cutoff, maximum_slope, fit_tolerance))
        ):
            raise ValueError("Sampled fit tolerances must be finite and positive.")
        if (
            isinstance(fit_degree, bool)
            or not isinstance(fit_degree, int)
            or not 2 <= fit_degree <= 4
        ):
            raise ValueError("fit_degree must be an integer in [2, 4].")
        if not np.isfinite(oversampling) or oversampling < 1:
            raise ValueError("oversampling must be finite and at least one.")
        _validate_identity(closed, geometry_id)
        for name, value, allowed in (
            ("reference_normals", reference_normals, codimension == 1),
            ("reference_tangents", reference_tangents, codimension > 1),
        ):
            if value is None:
                continue
            if not allowed:
                raise ValueError(
                    f"{name} does not orient this embedding: hypersurfaces use normals, space curves use tangents."
                )
            host = np.asarray(value)
            if (
                host.ndim != 2
                or host.shape[1] != ambient_dimension
                or not np.issubdtype(host.dtype, np.floating)
                or np.any(~np.isfinite(host))
                or np.any(np.linalg.norm(host, axis=-1) == 0)
            ):
                raise ValueError(
                    f"{name} requires finite nonzero floating (N,{ambient_dimension}) vectors."
                )
        self.reference_normals = (
            None if reference_normals is None else jnp.asarray(reference_normals)
        )
        self.reference_tangents = (
            None if reference_tangents is None else jnp.asarray(reference_tangents)
        )
        self.declared_tube_radius, self.closed = declared_tube_radius, closed
        self.rank_cutoff, self.maximum_slope, self.fit_tolerance = (
            rank_cutoff,
            maximum_slope,
            fit_tolerance,
        )
        self.fit_degree, self.oversampling = fit_degree, float(oversampling)
        self.geometry_id = geometry_id
        self.intrinsic_dimension = intrinsic_dimension
        self.ambient_dimension = ambient_dimension

    @property
    def minimum_neighbors(self) -> int:
        # Height monomials of degree 1..p (the constant is the node itself).
        features = comb(self.intrinsic_dimension + self.fit_degree, self.fit_degree) - 1
        return int(np.ceil(self.oversampling * features - 1e-12))

    @property
    def _declared_orientation(self) -> Array | None:
        return (
            self.reference_normals
            if self.ambient_dimension - self.intrinsic_dimension == 1
            else self.reference_tangents
        )

    def prepare_reference(self, points: Array, relation: RowRelation) -> Array:
        """Host orientation field: hypersurface normals or space-curve tangents."""
        offsets = jnp.where(
            relation.valid[..., None],
            points[relation.source_indices] - points[:, None, :],
            0,
        )
        hypersurface = self.ambient_dimension - self.intrinsic_dimension == 1
        ambient = self.ambient_dimension

        def direction(matrix: Array) -> Array:
            result = svd(
                SVDProblem(
                    DenseLinearOperator(matrix, operator_id="surface-pca"),
                    problem_id="surface-pca",
                ),
                policy=SVDSolvePolicy(count=ambient, failure=FailurePolicy("status")),
            )
            return result.right_vectors[:, -1 if hypersurface else 0]

        vectors = np.asarray(jax.vmap(direction)(offsets)).copy()
        indices, active = np.asarray(relation.source_indices), np.asarray(relation.valid)
        declared = self._declared_orientation
        if declared is not None:
            reference = np.asarray(declared)
            if reference.shape != vectors.shape:
                raise ValueError("Reference orientation must match points.")
            vectors *= np.where(np.sum(vectors * reference, axis=-1) < 0, -1, 1)[:, None]
        else:
            visited = np.zeros(points.shape[0], dtype=np.bool_)
            for root in range(points.shape[0]):
                if visited[root]:
                    continue
                visited[root] = True
                queue = [root]
                for node in queue:
                    for neighbor in indices[node, active[node]]:
                        dot = float(vectors[node] @ vectors[neighbor])
                        if abs(dot) < 0.25:
                            raise ValueError(
                                "Sampled tangent planes are inconsistent or undersampled."
                            )
                        if not visited[neighbor]:
                            vectors[neighbor] *= 1 if dot > 0 else -1
                            visited[neighbor] = True
                            queue.append(int(neighbor))
                        elif dot < 0:
                            raise ValueError(
                                "Sampled normal orientation is inconsistent."
                            )
        return jnp.asarray(vectors, dtype=points.dtype)

    def evaluate(
        self,
        points: Array,
        relation: RowRelation,
        *,
        reference: SurfaceGeometryEvaluation | None = None,
        reference_orientation: Array | None = None,
        require_tube: bool = False,
        boundary: SurfaceBoundary | None = None,
    ) -> SurfaceGeometryEvaluation:
        count, ambient = points.shape
        if ambient != self.ambient_dimension:
            raise ValueError("Points must match the declared ambient dimension.")
        intrinsic = self.intrinsic_dimension
        identity = jnp.eye(ambient, dtype=points.dtype)
        if reference is not None:
            orientation0 = _orientation_field(
                reference.tangent_frames, reference.normal_frames
            )
            projector0 = reference.projectors
        elif reference_orientation is not None:
            orientation0 = reference_orientation / jnp.linalg.norm(
                reference_orientation, axis=-1, keepdims=True
            )
            outer = orientation0[..., :, None] * orientation0[..., None, :]
            projector0 = identity - outer if ambient - intrinsic == 1 else outer
        else:
            raise ValueError("Sampled evaluation requires an admitted fixed orientation.")
        frames0, normals0, _, frame0_ok = _oriented_frames(
            projector0, intrinsic, orientation0
        )
        minimum = self.minimum_neighbors
        slope0, hessian0, _, _, _, ok0 = _fit_height(
            points,
            relation,
            frames0,
            normals0,
            self.fit_degree,
            self.rank_cutoff,
            minimum,
        )
        projector1, _, _, graph0_ok = _graph_geometry(frames0, normals0, slope0, hessian0)
        frames, chart_normals, _, frame1_ok = _oriented_frames(
            projector1, intrinsic, orientation0
        )
        slope, hessian, residual, normal_error, curvature_error, fit_ok = _fit_height(
            points,
            relation,
            frames,
            chart_normals,
            self.fit_degree,
            self.rank_cutoff,
            minimum,
        )
        projector, second_form, regularity, graph_ok = _graph_geometry(
            frames, chart_normals, slope, hessian
        )
        tangents, normals, gauge, frame_ok = _oriented_frames(
            projector, intrinsic, orientation0
        )
        orientation_field = _orientation_field(tangents, normals)
        offsets = points[relation.source_indices] - points[:, None, :]
        lengths = jnp.linalg.norm(offsets, axis=-1)
        scale = jnp.max(jnp.where(relation.valid, lengths, 0), axis=-1)
        orientation = jnp.sum(orientation_field * orientation0, axis=-1) > 0
        neighbor_alignment = jnp.sum(
            orientation_field[:, None, :] * orientation_field[relation.source_indices],
            axis=-1,
        )
        orientation &= jnp.all(~relation.valid | (neighbor_alignment > 0), axis=-1)
        # Estimate near-normal neighbors: useful diagnostic, not a reach bound.
        normal_offset = jnp.linalg.norm(
            contract("nab,nkb->nka", identity - projector, offsets), axis=-1
        )
        sheet = jnp.min(
            jnp.where(
                relation.valid & (normal_offset > 0.75 * lengths) & (lengths > 0),
                lengths,
                jnp.inf,
            ),
            axis=-1,
        )
        boundary_mask, conormals = _boundary_fields(
            boundary, count, ambient, points.dtype
        )
        conormal_residual = _conormal_residual(conormals, boundary_mask, projector)
        valid = (
            fit_ok
            & ok0
            & frame0_ok
            & graph0_ok
            & graph_ok
            & frame1_ok
            & frame_ok
            & orientation
            & (sheet > scale)
            & _surrounding_support(points, relation, frames, boundary_mask)
            & (jnp.linalg.norm(slope, axis=(-2, -1)) < self.maximum_slope)
            & (residual < self.fit_tolerance * scale)
            & (conormal_residual <= self.fit_tolerance)
            & jnp.all(jnp.isfinite(second_form), axis=(-3, -2, -1))
        )
        displacement = (
            jnp.zeros((count,), dtype=points.dtype)
            if reference is None
            else jnp.linalg.norm(points - reference.points, axis=-1)
        )
        radius = self.declared_tube_radius
        margin = (
            jnp.zeros((count,), dtype=points.dtype)
            if radius is None
            else radius - displacement
        )
        # A declared sample tube is deliberately not admitted as a certified tube.
        tube_valid = jnp.zeros((count,), dtype=jnp.bool_)
        if require_tube:
            valid &= tube_valid
        evidence = SurfaceGeometryEvidence(
            valid=valid,
            status=_status(
                valid,
                jnp.ones((count,), dtype=jnp.bool_),
                orientation,
                jnp.full((count,), require_tube, dtype=jnp.bool_),
            ),
            projection_valid=jnp.zeros((count,), dtype=jnp.bool_),
            tube_valid=tube_valid,
            trust_margin=margin,
            projector_comparison_estimate=jnp.linalg.norm(
                projector - projector0, axis=(-2, -1)
            ),
            sheet_separation_estimate=sheet,
            fit_residual=residual,
            normal_error_estimate=normal_error,
            curvature_error_estimate=curvature_error,
            regularity_margin=regularity,
            gauge_margin=gauge,
            boundary=boundary_mask,
            boundary_conormal_residual=conormal_residual,
            orientation_consistent=orientation,
            tube_certified=False,
            projection_to_source=False,
            estimate_only=True,
            source="sample-estimate",
        )
        return SurfaceGeometryEvaluation(
            points=points,
            tangent_frames=tangents,
            normal_frames=normals,
            chart_frames=frames,
            chart_normal_frames=chart_normals,
            projectors=projector,
            second_fundamental_form=second_form,
            chart_slope=slope,
            chart_hessian=hessian,
            evidence=evidence,
        )


SurfaceGeometry: TypeAlias = (
    ImplicitSurfaceGeometry | ChartSurfaceGeometry | SampledSurfaceGeometry
)


__all__ = [
    "ChartSurfaceGeometry",
    "ImplicitSurfaceGeometry",
    "SampledSurfaceGeometry",
    "SurfaceAmbientDim",
    "SurfaceBoundary",
    "SurfaceBoundaryNodeDim",
    "SurfaceBoundarySource",
    "SurfaceCodimensionDim",
    "SurfaceGeometry",
    "SurfaceGeometryEvaluation",
    "SurfaceGeometryEvidence",
    "SurfaceGeometrySource",
    "SurfaceGeometryStatus",
    "SurfaceIntrinsicDim",
    "SurfaceNodeDim",
]

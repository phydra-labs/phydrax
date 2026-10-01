# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Smooth surface geometry; sampled diagnostics are estimates, never reach proofs."""

from __future__ import annotations

from enum import IntEnum
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...ein import contract
from ...geometry._capabilities import GeometryCapability
from ...geometry._certificate import FieldRegularity
from ...geometry._contracts import CompiledGeometry
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    FailurePolicy,
    LeastSquaresProblem,
    LinearSolvePolicy,
    RankPolicy,
    solve,
)
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from ...metrix._ambient import RegularLevelSetManifold
from ...sparse import RowRelation
from ...typing import Bool, Dim, Float, Int32, parse


class SurfaceNodeDim(Dim):
    pass


class SurfaceGeometryStatus(IntEnum):
    VALID = 0
    DEGENERATE = 1
    PROJECTION_FAILURE = 2
    ORIENTATION_FAILURE = 3
    TUBE_FAILURE = 4
    SUPPORT_INVALID = 5


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
    orientation_consistent: Bool[SurfaceNodeDim]
    tube_certified: bool = eqx.field(static=True)
    projection_to_source: bool = eqx.field(static=True)
    estimate_only: bool = eqx.field(static=True)


@final
class SurfaceGeometryEvaluation(StrictModule):
    __strict_contract__ = True
    points: Float[SurfaceNodeDim, Literal[3]]
    normals: Float[SurfaceNodeDim, Literal[3]]
    tangent_frames: Float[SurfaceNodeDim, Literal[3], Literal[2]]
    chart_frames: Float[SurfaceNodeDim, Literal[3], Literal[2]]
    chart_normal: Float[SurfaceNodeDim, Literal[3]]
    projectors: Float[SurfaceNodeDim, Literal[3], Literal[3]]
    curvature_tensor: Float[SurfaceNodeDim, Literal[3], Literal[3]]
    chart_slope: Float[SurfaceNodeDim, Literal[2]]
    chart_hessian: Float[SurfaceNodeDim, Literal[2], Literal[2]]
    evidence: SurfaceGeometryEvidence


def tangent_frames(normals: Array) -> Array:
    """Deterministic local gauge; its discrete axis selection is not differentiated."""
    axis = jnp.argmin(jnp.abs(normals), axis=-1)
    seed = jax.nn.one_hot(axis, 3, dtype=normals.dtype)
    first = jnp.cross(normals, seed)
    first /= jnp.maximum(
        jnp.linalg.norm(first, axis=-1, keepdims=True), jnp.finfo(normals.dtype).tiny
    )
    return jnp.stack((first, jnp.cross(normals, first)), axis=-1)


def _surrounding_support(points: Array, relation: RowRelation, frames: Array) -> Array:
    """Local chart surround evidence; refuses one-sided/open sample rows."""
    offsets = points[relation.source_indices] - points[:, None, :]
    xy = contract("nka,nai->nki", offsets, frames)
    nonzero = relation.valid & (
        jnp.linalg.norm(xy, axis=-1) > jnp.finfo(points.dtype).eps
    )
    count = jnp.sum(nonzero, axis=-1)
    angles = jnp.sort(
        jnp.where(nonzero, jnp.arctan2(xy[..., 1], xy[..., 0]), jnp.inf), axis=-1
    )
    last = jnp.take_along_axis(angles, jnp.maximum(count - 1, 0)[:, None], axis=-1)[:, 0]
    gaps = jnp.diff(angles, axis=-1)
    gaps = jnp.where(jnp.arange(gaps.shape[-1])[None, :] < count[:, None] - 1, gaps, 0)
    maximum = jnp.maximum(jnp.max(gaps, axis=-1), angles[:, 0] + 2 * jnp.pi - last)
    return (count >= 3) & jnp.isfinite(maximum) & (maximum < jnp.pi - 1e-6)


def _fit_height(
    points: Array, relation: RowRelation, frames: Array, normals: Array, cutoff: float
) -> tuple[Array, Array, Array, Array]:
    offsets = points[relation.source_indices] - points[:, None, :]
    xy = contract("nka,nab->nkb", offsets, frames)
    scale = jnp.max(jnp.where(relation.valid, jnp.linalg.norm(xy, axis=-1), 0), axis=-1)
    safe_scale = jnp.maximum(scale, jnp.finfo(points.dtype).tiny)
    uv = xy / safe_scale[:, None, None]
    u, v = uv[..., 0], uv[..., 1]
    design = jnp.stack((u, v, u * u / 2, u * v, v * v / 2), axis=-1)
    design = jnp.where(relation.valid[..., None], design, 0)
    height = jnp.where(relation.valid, contract("nka,na->nk", offsets, normals), 0)

    def fit(matrix: Array, rhs: Array) -> tuple[Array, Array]:
        result = solve(
            LeastSquaresProblem(
                DenseLinearOperator(matrix, operator_id="surface-height-fit"),
                problem_id="surface-height-fit",
            ),
            rhs,
            policy=LinearSolvePolicy(
                DenseSVD(),
                rank=RankPolicy(relative_cutoff=cutoff, require_full_rank=True),
                failure=FailurePolicy("status"),
            ),
        )
        return result.value, jnp.all(result.diagnostics.converged)

    coefficients, converged = jax.vmap(fit)(design, height)
    slope = coefficients[:, :2] / safe_scale[:, None]
    hessian = (
        jnp.stack(
            (
                coefficients[:, 2],
                coefficients[:, 3],
                coefficients[:, 3],
                coefficients[:, 4],
            ),
            axis=-1,
        ).reshape((-1, 2, 2))
        / safe_scale[:, None, None] ** 2
    )
    residual = jnp.max(
        jnp.abs(contract("nkp,np->nk", design, coefficients) - height), axis=-1
    )
    return (
        slope,
        hessian,
        residual,
        converged & (scale > 0) & (jnp.sum(relation.valid, axis=-1) >= 6),
    )


@final
class ImplicitSurfaceGeometry(StrictModule):
    """Native source projection and level-set manifold tangent geometry.

    ``closed`` is a source declaration, not inferred from finite samples. A tube
    radius is admitted only as an explicit source-domain guarantee; curvature is
    never used to manufacture a global reach bound.
    """

    source: CompiledGeometry | RegularLevelSetManifold
    closed: bool = eqx.field(static=True)
    certified_tube_radius: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: CompiledGeometry | RegularLevelSetManifold,
        *,
        closed: bool = True,
        certified_tube_radius: float | None = None,
        tolerance: float = 1e-7,
        geometry_id: str = "implicit-surface",
    ) -> None:
        if not closed:
            raise ValueError("Open surfaces are unsupported.")
        if isinstance(source, CompiledGeometry):
            if (
                source.ambient_dimension != 3
                or source.intrinsic_dimension not in (2, 3)
                or source.field_certificate.regularity is FieldRegularity.NONSMOOTH
            ):
                raise ValueError("Surface requires a smooth boundary source in R3.")
            source.require(GeometryCapability.CLOSEST_POINT)
            # A radial signed-distance field is nonsmooth at its medial set,
            # although its boundary is smooth. Native curvature admission
            # distinguishes that case from sharp sources without a new calculus.
            if source.field_certificate.regularity is not FieldRegularity.SMOOTH:
                source.require(GeometryCapability.CONTACT_CURVATURE)
        elif (
            not isinstance(source, RegularLevelSetManifold)
            or source.point_shape != (3,)
            or source.codimension != 1
        ):
            raise TypeError("source must be a native codimension-one Euclidean surface.")
        if tolerance <= 0 or not np.isfinite(tolerance):
            raise ValueError("tolerance must be finite and positive.")
        if certified_tube_radius is not None and (
            not np.isfinite(certified_tube_radius) or certified_tube_radius <= 0
        ):
            raise ValueError("certified_tube_radius must be positive.")
        if (
            not isinstance(closed, bool)
            or not isinstance(geometry_id, str)
            or not geometry_id
        ):
            raise ValueError(
                "closed must be bool and geometry_id must be a nonempty source identity."
            )
        self.source, self.closed = source, closed
        self.certified_tube_radius, self.tolerance = certified_tube_radius, tolerance
        self.geometry_id = geometry_id

    def evaluate(
        self,
        points: Array,
        relation: RowRelation,
        *,
        reference: SurfaceGeometryEvaluation | None = None,
        require_tube: bool = False,
    ) -> SurfaceGeometryEvaluation:
        if isinstance(self.source, CompiledGeometry):
            projected = self.source.closest_point(points)
            coordinates = projected.closest_point.astype(points.dtype)
            projection_valid = (
                projected.unique & projected.regular & projected.normal_coordinate_valid
            )
            native_normals = projected.oriented_normal.astype(points.dtype)
            source = self.source

            def field(point: Array) -> Array:
                return source.boundary_field(point[None, :])[0]

            def constraint(point: Array) -> Array:
                return field(point)[None]

            manifold = RegularLevelSetManifold(
                constraint, ambient_dimension=3, codimension=1, tolerance=self.tolerance
            )
            if self.source.field_certificate.regularity is not FieldRegularity.SMOOTH:
                projection_valid &= self.source.contact_curvature(coordinates).valid
        else:
            manifold = self.source

            def field(point: Array) -> Array:
                return manifold.constraint(point)[0]

            projected = manifold.project_normal(points)
            coordinates = projected.points
            native_normals = (
                projected.geometry.constraint_jacobian[:, 0, :]
                * manifold.orientation_sign
            )
            projection_valid = projected.valid & (projected.residual <= self.tolerance)
        local = manifold.local_geometry(coordinates)
        metric_ok = (
            jnp.max(jnp.abs(local.metric - jnp.eye(3, dtype=points.dtype)), axis=(-2, -1))
            <= self.tolerance
        )
        gradients = jax.vmap(jax.grad(field))(coordinates)
        norm = jnp.linalg.norm(native_normals, axis=-1)
        normals = native_normals / jnp.maximum(
            norm[:, None], jnp.finfo(points.dtype).tiny
        )
        frames = tangent_frames(normals)
        hessians = jax.vmap(jax.hessian(field))(coordinates)
        shape = contract("nai,nab,nbj->nij", frames, hessians, frames) / jnp.maximum(
            jnp.linalg.norm(gradients, axis=-1)[:, None, None],
            jnp.finfo(points.dtype).tiny,
        )
        orientation = jnp.sign(jnp.sum(normals * gradients, axis=-1))
        shape *= orientation[:, None, None]
        projector = (
            jnp.eye(3, dtype=points.dtype) - normals[..., :, None] * normals[..., None, :]
        )
        distance = jnp.linalg.norm(coordinates - points, axis=-1)
        radius = self.certified_tube_radius
        tube_valid = (
            jnp.zeros(points.shape[0], dtype=jnp.bool_)
            if radius is None
            else distance < radius
        )
        margin = (
            jnp.full((points.shape[0],), 0, dtype=points.dtype)
            if radius is None
            else radius - distance
        )
        consistent = (
            jnp.ones(points.shape[0], dtype=jnp.bool_)
            if reference is None
            else jnp.sum(normals * reference.normals, axis=-1) > 0
        )
        valid = (
            projection_valid
            & local.valid
            & metric_ok
            & (norm > self.tolerance)
            & consistent
            & _surrounding_support(coordinates, relation, frames)
            & jnp.all(jnp.isfinite(coordinates), axis=-1)
        )
        if require_tube:
            valid &= tube_valid
        status = jnp.where(
            valid,
            0,
            jnp.where(
                ~projection_valid,
                2,
                jnp.where(~consistent, 3, jnp.where(require_tube & ~tube_valid, 4, 1)),
            ),
        ).astype(jnp.int32)
        evidence = SurfaceGeometryEvidence(
            valid=valid,
            status=status,
            projection_valid=projection_valid,
            tube_valid=tube_valid,
            trust_margin=margin,
            projector_comparison_estimate=jnp.max(
                jnp.abs(projector - local.tangent_projector), axis=(-2, -1)
            ),
            sheet_separation_estimate=jnp.full(
                (points.shape[0],), jnp.nan, dtype=points.dtype
            ),
            fit_residual=jnp.abs(jax.vmap(field)(coordinates)),
            orientation_consistent=consistent,
            tube_certified=radius is not None,
            projection_to_source=True,
            estimate_only=False,
        )
        return SurfaceGeometryEvaluation(
            points=coordinates,
            normals=normals,
            tangent_frames=frames,
            chart_frames=frames,
            chart_normal=normals,
            projectors=projector,
            curvature_tensor=contract("nai,nij,nbj->nab", frames, shape, frames),
            chart_slope=jnp.zeros((points.shape[0], 2), dtype=points.dtype),
            chart_hessian=-shape,
            evidence=evidence,
        )


@final
class SampledSurfaceGeometry(StrictModule):
    """Quadratic tangent-height reconstruction on a declared closed smooth sheet.

    Without a reference, orientability is checked by host graph propagation.
    Sheet separation and projector comparisons remain explicitly local estimates.
    No finite sample set certifies a tube or projection to the true surface.
    """

    __strict_contract__ = True
    reference_normals: Float[SurfaceNodeDim, Literal[3]] | None
    declared_tube_radius: float | None = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    rank_cutoff: float = eqx.field(static=True)
    maximum_slope: float = eqx.field(static=True)
    fit_tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_normals: Array | None = None,
        *,
        declared_tube_radius: float | None = None,
        closed: bool = True,
        regularity: Literal["smooth"] = "smooth",
        rank_cutoff: float = 1e-9,
        maximum_slope: float = 0.5,
        fit_tolerance: float = 0.15,
        geometry_id: str = "sampled-surface",
    ) -> None:
        parse(regularity, Literal["smooth"], "regularity")
        if not closed:
            raise ValueError("Open surfaces are unsupported.")
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
            not isinstance(closed, bool)
            or not isinstance(geometry_id, str)
            or not geometry_id
        ):
            raise ValueError(
                "closed must be bool and geometry_id must be a nonempty source identity."
            )
        if reference_normals is not None:
            host = np.asarray(reference_normals)
            if (
                host.ndim != 2
                or host.shape[1] != 3
                or not np.issubdtype(host.dtype, np.floating)
                or np.any(~np.isfinite(host))
                or np.any(np.linalg.norm(host, axis=-1) == 0)
            ):
                raise ValueError(
                    "reference_normals requires finite nonzero floating (N,3) vectors."
                )
        self.reference_normals = (
            None if reference_normals is None else jnp.asarray(reference_normals)
        )
        self.declared_tube_radius, self.closed = declared_tube_radius, closed
        self.rank_cutoff, self.maximum_slope, self.fit_tolerance = (
            rank_cutoff,
            maximum_slope,
            fit_tolerance,
        )
        self.geometry_id = geometry_id

    def prepare_reference(self, points: Array, relation: RowRelation) -> Array:
        offsets = jnp.where(
            relation.valid[..., None],
            points[relation.source_indices] - points[:, None, :],
            0,
        )

        def normal(matrix: Array) -> Array:
            result = svd(
                SVDProblem(
                    DenseLinearOperator(matrix, operator_id="surface-pca"),
                    problem_id="surface-pca",
                ),
                policy=SVDSolvePolicy(count=3, failure=FailurePolicy("status")),
            )
            return result.right_vectors[:, -1]

        normals = np.asarray(jax.vmap(normal)(offsets)).copy()
        indices, active = np.asarray(relation.source_indices), np.asarray(relation.valid)
        if self.reference_normals is not None:
            reference = np.asarray(self.reference_normals)
            if (
                reference.shape != normals.shape
                or not np.all(np.isfinite(reference))
                or np.any(np.linalg.norm(reference, axis=-1) == 0)
            ):
                raise ValueError(
                    "reference_normals must be finite nonzero vectors matching points."
                )
            normals *= np.where(np.sum(normals * reference, axis=-1) < 0, -1, 1)[:, None]
        else:
            visited = np.zeros(points.shape[0], dtype=np.bool_)
            for root in range(points.shape[0]):
                if visited[root]:
                    continue
                visited[root] = True
                queue = [root]
                for node in queue:
                    for neighbor in indices[node, active[node]]:
                        dot = float(normals[node] @ normals[neighbor])
                        if abs(dot) < 0.25:
                            raise ValueError(
                                "Sampled tangent planes are inconsistent or undersampled."
                            )
                        if not visited[neighbor]:
                            normals[neighbor] *= 1 if dot > 0 else -1
                            visited[neighbor] = True
                            queue.append(int(neighbor))
                        elif dot < 0:
                            raise ValueError(
                                "Sampled normal orientation is inconsistent."
                            )
        return jnp.asarray(normals, dtype=points.dtype)

    def evaluate(
        self,
        points: Array,
        relation: RowRelation,
        *,
        reference: SurfaceGeometryEvaluation | None = None,
        reference_normals: Array | None = None,
        require_tube: bool = False,
    ) -> SurfaceGeometryEvaluation:
        if reference is not None:
            normals0, frames0 = reference.normals, reference.tangent_frames
        elif reference_normals is not None:
            normals0, frames0 = reference_normals, tangent_frames(reference_normals)
        else:
            raise ValueError("Sampled evaluation requires an admitted fixed orientation.")
        slope0, _, _, ok0 = _fit_height(
            points, relation, frames0, normals0, self.rank_cutoff
        )
        normals = normals0 - contract("nai,ni->na", frames0, slope0)
        normals /= jnp.linalg.norm(normals, axis=-1, keepdims=True)
        frames = tangent_frames(normals)
        slope, hessian, residual, fit_ok = _fit_height(
            points, relation, frames, normals, self.rank_cutoff
        )
        chart_normal = normals
        normals = chart_normal - contract("nai,ni->na", frames, slope)
        normals /= jnp.linalg.norm(normals, axis=-1, keepdims=True)
        offsets = points[relation.source_indices] - points[:, None, :]
        lengths = jnp.linalg.norm(offsets, axis=-1)
        scale = jnp.max(jnp.where(relation.valid, lengths, 0), axis=-1)
        orientation = jnp.sum(normals * normals0, axis=-1) > 0
        neighbor_alignment = jnp.sum(
            normals[:, None, :] * normals[relation.source_indices], axis=-1
        )
        orientation &= jnp.all(~relation.valid | (neighbor_alignment > 0), axis=-1)
        # Estimate near-normal neighbors: useful diagnostic, not a reach bound.
        normal_offset = jnp.abs(contract("nka,na->nk", offsets, normals))
        sheet = jnp.min(
            jnp.where(
                relation.valid & (normal_offset > 0.75 * lengths) & (lengths > 0),
                lengths,
                jnp.inf,
            ),
            axis=-1,
        )
        valid = (
            fit_ok
            & ok0
            & orientation
            & (sheet > scale)
            & _surrounding_support(points, relation, frames)
            & (jnp.linalg.norm(slope, axis=-1) < self.maximum_slope)
            & (residual < self.fit_tolerance * scale)
            & jnp.all(jnp.isfinite(normals), axis=-1)
        )
        displacement = (
            jnp.zeros(points.shape[0], dtype=points.dtype)
            if reference is None
            else jnp.linalg.norm(points - reference.points, axis=-1)
        )
        radius = self.declared_tube_radius
        margin = (
            jnp.zeros(points.shape[0], dtype=points.dtype)
            if radius is None
            else radius - displacement
        )
        # A declared sample tube is deliberately not admitted as a certified tube.
        tube_valid = jnp.zeros(points.shape[0], dtype=jnp.bool_)
        if require_tube:
            valid &= tube_valid
        projector = (
            jnp.eye(3, dtype=points.dtype) - normals[..., :, None] * normals[..., None, :]
        )
        comparison = jnp.linalg.norm(
            projector
            - (
                jnp.eye(3, dtype=points.dtype)
                - normals0[..., :, None] * normals0[..., None, :]
            ),
            axis=(-2, -1),
        )
        status = jnp.where(
            valid, 0, jnp.where(require_tube, 4, jnp.where(~orientation, 3, 1))
        ).astype(jnp.int32)
        evidence = SurfaceGeometryEvidence(
            valid=valid,
            status=status,
            projection_valid=jnp.zeros(points.shape[0], dtype=jnp.bool_),
            tube_valid=tube_valid,
            trust_margin=margin,
            projector_comparison_estimate=comparison,
            sheet_separation_estimate=sheet,
            fit_residual=residual,
            orientation_consistent=orientation,
            tube_certified=False,
            projection_to_source=False,
            estimate_only=True,
        )
        # The graph chart carries its first and second fundamental forms.
        denominator = 1 + jnp.sum(slope * slope, axis=-1)
        inverse = (
            jnp.eye(2, dtype=points.dtype)
            - slope[..., :, None] * slope[..., None, :] / denominator[:, None, None]
        )
        chart = frames + chart_normal[..., :, None] * slope[:, None, :]
        dual = contract("nai,nij->naj", chart, inverse)
        curvature = (
            -contract("nai,nij,nbj->nab", dual, hessian, dual)
            / jnp.sqrt(denominator)[:, None, None]
        )
        return SurfaceGeometryEvaluation(
            points=points,
            normals=normals,
            tangent_frames=tangent_frames(normals),
            chart_frames=frames,
            chart_normal=chart_normal,
            projectors=projector,
            curvature_tensor=curvature,
            chart_slope=slope,
            chart_hessian=hessian,
            evidence=evidence,
        )


__all__ = [
    "ImplicitSurfaceGeometry",
    "SampledSurfaceGeometry",
    "SurfaceGeometryStatus",
    "SurfaceGeometryEvidence",
    "SurfaceGeometryEvaluation",
    "SurfaceNodeDim",
]

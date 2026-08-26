#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.spatial import cKDTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._polynomial._total_degree import TotalDegreePolynomialFeatures
from .._strict import StrictModule
from ..linalg import ArraySpace, DiagonalPairing
from ..sparse import RowRelation
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from ._lifecycle import validate_prepared_metadata
from ._local_polynomial import (
    prepare_weighted_least_squares,
    PreparedWeightedLeastSquares,
)
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace, TensorDofLayout
from ._support import DiscreteSupport
from ._tensor import AbstractStrongFormDiscretization
from ._topology import EntitySet, PointTopology


class PointStencilReport(StrictModule):
    maximum_condition_number: float = eqx.field(static=True)
    minimum_singular_value: float = eqx.field(static=True)
    maximum_moment_residual: float = eqx.field(static=True)
    maximum_amplification: float = eqx.field(static=True)
    minimum_trust_radius: float = eqx.field(static=True)
    worst_point: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class PointCloudPlan(eqx.Module):
    points: Array
    quadrature_weights: Array
    boundary_mask: Array
    boundary_normals: Array
    boundary_quadrature_weights: Array | None
    degree: int = eqx.field(static=True)
    neighbor_count: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        boundary_mask: ArrayLike | None = None,
        boundary_normals: ArrayLike | None = None,
        boundary_quadrature_weights: ArrayLike | None = None,
        degree: int = 2,
        neighbor_count: int | None = None,
        condition_limit: float = 1e8,
    ):
        points_ = np.asarray(points, dtype=float)
        weights = np.asarray(quadrature_weights, dtype=float)
        if points_.ndim != 2 or points_.shape[0] == 0 or points_.shape[1] == 0:
            raise ValueError("Point cloud must have shape (points, dimension).")
        if np.any(~np.isfinite(points_)):
            raise ValueError("Point cloud coordinates must be finite.")
        if (
            weights.shape != points_.shape[:1]
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("Point quadrature weights must be finite and positive.")
        if np.unique(points_, axis=0).shape[0] != points_.shape[0]:
            raise ValueError("Point cloud must not contain duplicate coordinates.")
        degree_ = int(degree)
        if degree_ < 2:
            raise ValueError("Point polynomial degree must be at least two.")
        feature_count = math.comb(points_.shape[1] + degree_, degree_)
        neighbors = (
            max(feature_count + points_.shape[1], 2 * feature_count)
            if neighbor_count is None
            else int(neighbor_count)
        )
        if neighbors < feature_count or neighbors > points_.shape[0]:
            raise ValueError("neighbor_count must cover the polynomial basis and cloud.")
        condition = float(condition_limit)
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("condition_limit must exceed one.")
        boundary = (
            np.zeros(points_.shape[0], dtype=bool)
            if boundary_mask is None
            else np.asarray(boundary_mask)
        )
        if boundary.dtype != np.dtype(bool) or boundary.shape != points_.shape[:1]:
            raise ValueError("boundary_mask must be Boolean with shape (points,).")
        boundary = np.asarray(boundary, dtype=bool)
        normals = (
            np.zeros_like(points_)
            if boundary_normals is None
            else np.asarray(boundary_normals, dtype=float).copy()
        )
        if normals.shape != points_.shape or np.any(~np.isfinite(normals)):
            raise ValueError("boundary_normals must be finite with point-cloud shape.")
        lengths = np.linalg.norm(normals[boundary], axis=1)
        if lengths.size and np.any(lengths <= 0.0):
            raise ValueError("Boundary point normals must be nonzero.")
        if lengths.size:
            normals[boundary] /= lengths[:, None]
        boundary_weights = (
            None
            if boundary_quadrature_weights is None
            else np.asarray(boundary_quadrature_weights, dtype=float)
        )
        if boundary_weights is not None:
            if (
                boundary_weights.shape != points_.shape[:1]
                or np.any(~np.isfinite(boundary_weights))
                or np.any(boundary_weights[boundary] <= 0.0)
                or np.any(boundary_weights[~boundary] != 0.0)
            ):
                raise ValueError(
                    "boundary_quadrature_weights must be positive on boundary "
                    "points and zero elsewhere."
                )
        self.points = jnp.asarray(points_)
        self.quadrature_weights = jnp.asarray(weights)
        self.boundary_mask = jnp.asarray(boundary)
        self.boundary_normals = jnp.asarray(normals)
        self.boundary_quadrature_weights = (
            None if boundary_weights is None else jnp.asarray(boundary_weights)
        )
        self.degree = degree_
        self.neighbor_count = neighbors
        self.condition_limit = condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "point-cloud-plan",
                "points": array_tree_fingerprint(points_),
                "weights": array_tree_fingerprint(weights),
                "boundary": array_tree_fingerprint(boundary),
                "boundary_normals": array_tree_fingerprint(normals),
                "boundary_weights": (
                    None
                    if boundary_weights is None
                    else array_tree_fingerprint(boundary_weights)
                ),
                "degree": degree_,
                "neighbor_count": neighbors,
                "condition_limit": condition,
            }
        )

    def prepare(self, /) -> PreparedPointCloudDiscretization:
        return PreparedPointCloudDiscretization(self)


class PreparedPointCloudDiscretization(AbstractStrongFormDiscretization):
    plan: PointCloudPlan
    relation: RowRelation
    fit: PreparedWeightedLeastSquares
    derivative_weights: tuple[tuple[Array, Array], ...]
    report: PointStencilReport
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    trust_radius: Array

    def __init__(self, plan: PointCloudPlan, /):
        points = np.asarray(plan.points)
        count, dimension = points.shape
        tree = cKDTree(points)
        query_count = min(plan.neighbor_count + 1, count)
        distances, indices = tree.query(points, k=query_count)
        if query_count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        selected_distances = distances[:, : plan.neighbor_count]
        selected_indices = indices[:, : plan.neighbor_count].astype(np.int32)
        valid = np.isfinite(selected_distances)
        if np.any(np.sum(valid, axis=1) < plan.neighbor_count):
            raise ValueError("Point cloud has insufficient finite neighbors.")
        offsets = points[selected_indices] - points[:, None, :]
        characteristic = np.max(selected_distances, axis=1)
        if np.any(characteristic <= 0.0):
            raise ValueError("Point stencil characteristic lengths must be positive.")
        standardized = offsets / characteristic[:, None, None]
        features = TotalDegreePolynomialFeatures(dimension, plan.degree)
        exponents = np.asarray(features.exponents, dtype=np.int32)
        design = np.ones((count, plan.neighbor_count, exponents.shape[0] + 1))
        if exponents.shape[0]:
            design[:, :, 1:] = np.prod(
                standardized[:, :, None, :] ** exponents[None, None, :, :],
                axis=-1,
            )
        radial_weights = (
            1.0
            / np.maximum(
                selected_distances / characteristic[:, None],
                0.25,
            )
            ** 2
        )
        fit = prepare_weighted_least_squares(
            design,
            radial_weights,
            valid,
            condition_limit=plan.condition_limit,
        )
        factors = np.asarray(fit.factors)
        derivative_weights = []
        residuals = []
        amplifications = []
        for axis in range(dimension):
            axis_weights = []
            for order in (1, 2):
                target = np.zeros((count, exponents.shape[0] + 1))
                exponent = np.zeros(dimension, dtype=np.int32)
                exponent[axis] = order
                matches = np.nonzero(np.all(exponents == exponent[None, :], axis=1))[0]
                if matches.size != 1:
                    raise ValueError(
                        "Polynomial basis does not contain requested derivative."
                    )
                target[:, 1 + matches[0]] = math.factorial(order) / characteristic**order
                weights = np.einsum("rf,rfk->rk", target, factors)
                moments = np.einsum("rk,rkf->rf", weights, design)
                residuals.append(np.max(np.abs(moments - target)))
                amplifications.append(np.max(np.sum(np.abs(weights), axis=1)))
                axis_weights.append(jnp.asarray(weights))
            derivative_weights.append(tuple(axis_weights))
        trust = (
            np.full(count, np.inf)
            if query_count == plan.neighbor_count
            else 0.5
            * np.maximum(
                distances[:, plan.neighbor_count] - distances[:, plan.neighbor_count - 1],
                0.0,
            )
        )
        relation = RowRelation(
            selected_indices,
            source_size=count,
            valid=valid,
        )
        entities = EntitySet("point_cloud_points", 0, np.arange(count))
        topology = PointTopology(
            entities,
            neighborhoods=relation,
            refreshable_neighborhoods=True,
        )
        support = DiscreteSupport(topology, dimension, plan.plan_id)
        key = DiscretizationKey(
            "point_cloud",
            DiscretizationRole.PHYSICAL,
            domain_labels=("point",),
        )
        layout = TensorDofLayout(("point",), (count,))
        pairing = DiagonalPairing(plan.quadrature_weights)
        field_space = DiscreteFieldSpace(
            "point_state",
            support.support_id,
            layout,
            ArraySpace(
                (count,),
                pairing=pairing,
                space_id=canonical_fingerprint(
                    {"kind": "point-cloud-array-space", "plan": plan.plan_id}
                ),
            ),
            representation="point_value",
            reconstruction_id=fit.prepared_id,
        )
        measure = DiscreteMeasure(
            "point_cloud",
            support.support_id,
            entities.entity_set_id,
            plan.quadrature_weights,
            normalization="physical",
        )
        capabilities = (
            DiscretizationCapability.STRONG_DERIVATIVE,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.SPARSE_ASSEMBLY,
        )
        preparation = PreparationReport(
            capabilities=capabilities,
            resource_counts={
                "points": count,
                "dimension": dimension,
                "neighbor_capacity": plan.neighbor_count,
                "polynomial_features": design.shape[2],
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=key,
            support=support,
            field_spaces=(field_space,),
            measures=(measure,),
            capabilities=capabilities,
            preparation=preparation,
        )
        worst = fit.report.worst_row
        report_id = canonical_fingerprint(
            {
                "kind": "point-stencil-report",
                "fit": fit.prepared_id,
                "residuals": residuals,
                "amplifications": amplifications,
                "trust": array_tree_fingerprint(trust),
            }
        )
        self.plan = plan
        self.relation = relation
        self.fit = fit
        self.derivative_weights = tuple(derivative_weights)
        self.report = PointStencilReport(
            maximum_condition_number=fit.report.maximum_condition_number,
            minimum_singular_value=fit.report.minimum_singular_value,
            maximum_moment_residual=float(max(residuals)),
            maximum_amplification=float(max(amplifications)),
            minimum_trust_radius=float(np.min(trust)),
            worst_point=worst,
            report_id=report_id,
        )
        self.key = key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-point-cloud", "plan": plan.plan_id, "fit": fit.prepared_id}
        )
        self.numeric_version = "1"
        self.preparation = preparation
        self.trust_radius = jnp.asarray(trust)

    @property
    def spatial_dimension(self) -> int:
        return int(self.plan.points.shape[1])

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (int(self.plan.points.shape[0]),)

    @property
    def quadrature_weights(self) -> Array:
        return self.plan.quadrature_weights

    @property
    def discretization_id(self) -> str:
        return self.prepared_id

    @property
    def points(self) -> Array:
        return self.plan.points

    def _validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[:1] != self.state_shape:
            raise ValueError("Point-cloud state must begin with point count.")
        return value

    def _selected_axes(
        self,
        axes: int | Sequence[int] | None,
        /,
    ) -> tuple[int, ...]:
        selected = (
            tuple(range(self.spatial_dimension))
            if axes is None
            else (int(axes),)
            if isinstance(axes, int)
            else tuple(int(axis) for axis in axes)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(axis < 0 or axis >= self.spatial_dimension for axis in selected)
        ):
            raise ValueError("Point-cloud axes must be unique valid spatial axes.")
        return selected

    def _apply_weights(self, state: Array, weights: Array, /) -> Array:
        patches = state[self.relation.source_indices]
        payload = patches.shape[2:]
        masked = jnp.where(
            self.relation.valid.reshape(self.relation.valid.shape + (1,) * len(payload)),
            patches,
            0,
        )
        return jnp.sum(
            weights.reshape(weights.shape + (1,) * len(payload)) * masked,
            axis=1,
        )

    def partial_derivative(
        self,
        state: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        value = self._validate_state(state)
        axis_ = int(axis)
        order_ = int(order)
        if axis_ < 0 or axis_ >= self.spatial_dimension or order_ not in (1, 2):
            raise ValueError(
                "Point-cloud derivatives support valid axes and orders one/two."
            )
        return self._apply_weights(value, self.derivative_weights[axis_][order_ - 1])

    def transpose_partial_derivative(
        self,
        cotangent: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        value = self._validate_state(cotangent)
        axis_ = int(axis)
        order_ = int(order)
        if axis_ < 0 or axis_ >= self.spatial_dimension or order_ not in (1, 2):
            raise ValueError(
                "Point-cloud transpose derivatives support valid axes and orders one/two."
            )
        weights = self.derivative_weights[axis_][order_ - 1]
        payload = value.shape[1:]
        messages = weights.reshape(weights.shape + (1,) * len(payload)) * value[:, None]
        messages = jnp.where(
            self.relation.valid.reshape(self.relation.valid.shape + (1,) * len(payload)),
            messages,
            0,
        )
        output = jnp.zeros(self.state_shape + payload, dtype=value.dtype)
        return output.at[self.relation.source_indices].add(messages)

    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = self._selected_axes(axes)
        return jnp.stack(
            tuple(self.partial_derivative(state, axis=axis) for axis in selected), axis=-1
        )

    def divergence(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        del dual
        value = jnp.asarray(state)
        selected = self._selected_axes(axes)
        if value.shape[-1] != len(selected):
            raise ValueError(
                "Point-cloud divergence components must match selected axes."
            )
        result = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for component, axis in enumerate(selected):
            result = result + self.partial_derivative(value[..., component], axis=axis)
        return result

    def laplacian(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = self._selected_axes(axes)
        result = jnp.zeros_like(self._validate_state(state))
        for axis in selected:
            result = result + self.partial_derivative(state, axis=axis, order=2)
        return result

    def integral(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        del axes
        value = self._validate_state(state)
        return jnp.sum(
            self.quadrature_weights.reshape(self.state_shape + (1,) * (value.ndim - 1))
            * value,
            axis=0,
        )

    def flatten(self, state: ArrayLike, /) -> Array:
        return self._validate_state(state).reshape((-1,))

    def unflatten(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.size != self.state_shape[0]:
            raise ValueError("Flattened point-cloud state has wrong size.")
        return value.reshape(self.state_shape)

    def laplacian_matrix(self) -> Array:
        if self.state_shape[0] > 4096:
            raise ValueError(
                "Point-cloud Laplacian matrix exceeds dense analysis budget."
            )
        identity = jnp.eye(self.state_shape[0])
        return jax.vmap(self.laplacian, in_axes=1, out_axes=1)(identity)

    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        del rank
        raise ValueError(
            "Raw point-cloud Laplacians are not certified self-adjoint; use a "
            "certified dissipative point operator for spectral analysis."
        )


__all__ = [
    "PointCloudPlan",
    "PointStencilReport",
    "PreparedPointCloudDiscretization",
]

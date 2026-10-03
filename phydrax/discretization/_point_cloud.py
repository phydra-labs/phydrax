#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    ArraySpace,
    DiagonalPairing,
    prepare_linearization,
    PreparedLinearization,
)
from ..sparse import RowRelation
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from ._lifecycle import validate_prepared_metadata
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace, TensorDofLayout
from ._support import DiscreteSupport
from ._tensor import AbstractStrongFormDiscretization
from ._topology import EntitySet, PointTopology
from ._views import FieldQueryEvidence
from .meshfree._neighbors import _integer, _points, MeshfreeNeighborhoodPlan
from .meshfree._stencils import (
    LocalStencilEvidence,
    LocalStencilPolicy,
    LocalStencilReport,
    MeshfreeFunctional,
    prepare_local_stencils,
    PreparedLocalStencils,
    refresh_local_stencils,
)
from .spatial import MortonAddressPlan


@final
class PointCloudPlan(StrictModule):
    points: Array
    quadrature_weights: Array
    boundary_mask: Array
    boundary_normals: Array
    boundary_quadrature_weights: Array | None
    stencil: LocalStencilPolicy
    point_ids: Array
    address: MortonAddressPlan
    neighbors: int = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
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
        stencil: LocalStencilPolicy | None = None,
        neighbors: int | None = None,
        point_ids: ArrayLike | None = None,
        address: MortonAddressPlan | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        points_ = _points(points, "points", unique=True)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        if (
            weights.shape != points_.shape[:1]
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("Point quadrature weights must be finite and positive.")
        policy = LocalStencilPolicy() if stencil is None else stencil
        if not isinstance(policy, LocalStencilPolicy):
            raise TypeError("stencil must be a LocalStencilPolicy.")
        if policy.polynomial_degree < 2:
            raise ValueError(
                "Point-cloud strong derivatives require polynomial degree at least two."
            )
        feature_count = math.comb(
            points_.shape[1] + policy.polynomial_degree, policy.polynomial_degree
        )
        if policy.support is not None and neighbors is None:
            raise ValueError(
                "A smooth fixed-radius support needs an explicit candidate capacity "
                "(neighbors)."
            )
        count = (
            min(points_.shape[0], 2 * feature_count) if neighbors is None else neighbors
        )
        neighborhood = MeshfreeNeighborhoodPlan(
            points_,
            count,
            source_ids=point_ids,
            address=address,
            maximum_candidates=maximum_candidates,
            target_chunk_size=target_chunk_size,
            envelope=policy.support,
        )
        if neighborhood.neighbors < feature_count:
            raise ValueError("neighbors must cover the polynomial basis.")
        boundary = (
            np.zeros(points_.shape[0], dtype=np.bool_)
            if boundary_mask is None
            else np.asarray(boundary_mask)
        )
        if boundary.dtype != np.dtype(np.bool_) or boundary.shape != points_.shape[:1]:
            raise ValueError("boundary_mask must be Boolean with shape (points,).")
        boundary = np.asarray(boundary, dtype=np.bool_)
        normals = (
            np.zeros_like(points_)
            if boundary_normals is None
            else np.asarray(boundary_normals, dtype=np.float64).copy()
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
            else np.asarray(boundary_quadrature_weights, dtype=np.float64)
        )
        if boundary_weights is not None:
            if (
                boundary_weights.shape != points_.shape[:1]
                or np.any(~np.isfinite(boundary_weights))
                or np.any(boundary_weights[boundary] <= 0.0)
                or np.any(boundary_weights[~boundary] != 0.0)
            ):
                raise ValueError(
                    "boundary_quadrature_weights must be positive on boundary points and zero elsewhere."
                )
        self.points = jnp.asarray(points_)
        self.quadrature_weights = jnp.asarray(weights)
        self.boundary_mask = jnp.asarray(boundary)
        self.boundary_normals = jnp.asarray(normals)
        self.boundary_quadrature_weights = (
            None if boundary_weights is None else jnp.asarray(boundary_weights)
        )
        self.stencil = policy
        self.point_ids = neighborhood.source_ids
        self.address = neighborhood.address
        self.neighbors = neighborhood.neighbors
        self.maximum_candidates = neighborhood.maximum_candidates
        self.target_chunk_size = neighborhood.target_chunk_size
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
                "neighborhood": neighborhood.plan_id,
                "approximation": policy.approximation,
                "degree": policy.polynomial_degree,
                "phs_power": policy.phs_power,
                "weight_kernel": policy.weight_kernel,
                "coordinate_order": policy.coordinate_order,
                "condition_limit": policy.condition_limit,
                "amplification_limit": policy.amplification_limit,
                "acceptance": policy.acceptance,
            }
        )

    def prepare(self, /) -> PreparedPointCloudDiscretization:
        return PreparedPointCloudDiscretization(self)


@final
class PreparedPointCloudDiscretization(AbstractStrongFormDiscretization):
    """Prepared strong-form point cloud of one support epoch.

    ``plan`` holds the anchored reference geometry, stable point identities,
    quadrature measures and boundary data; ``coordinates`` and the stencil
    weights are the dynamic numerical state replaced by fixed-support
    ``refresh``. New support discovery requires a new plan (epoch boundary).
    """

    plan: PointCloudPlan
    coordinates: Array
    relation: RowRelation
    stencils: PreparedLocalStencils
    derivative_weights: tuple[tuple[Array, Array], ...]
    mixed_weights: tuple[tuple[tuple[int, ...], Array], ...]
    report: LocalStencilReport
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

    def __init__(self, plan: PointCloudPlan, /) -> None:
        points = np.asarray(plan.points)
        count, dimension = points.shape
        neighborhood = MeshfreeNeighborhoodPlan(
            plan.points,
            plan.neighbors,
            source_ids=np.asarray(plan.point_ids),
            address=plan.address,
            maximum_candidates=plan.maximum_candidates,
            target_chunk_size=plan.target_chunk_size,
            envelope=plan.stencil.support,
        ).prepare()
        functionals: list[MeshfreeFunctional] = []
        indices: list[tuple[int, ...]] = []
        for axis in range(dimension):
            for order in (1, 2):
                index = tuple(order if d == axis else 0 for d in range(dimension))
                indices.append(index)
                functionals.append(
                    MeshfreeFunctional((index,), (1.0,), name=f"d{axis}:{order}")
                )
        for first in range(dimension):
            for second in range(first + 1, dimension):
                index = tuple(int(d == first or d == second) for d in range(dimension))
                indices.append(index)
                functionals.append(
                    MeshfreeFunctional((index,), (1.0,), name=f"d{first}d{second}")
                )
        stencils = prepare_local_stencils(
            neighborhood, plan.points, plan.points, tuple(functionals), plan.stencil
        )
        derivative_weights = tuple(
            (stencils.weights[2 * axis], stencils.weights[2 * axis + 1])
            for axis in range(dimension)
        )
        trust = np.asarray(neighborhood.trust_margin)
        relation = neighborhood.relation
        entities = EntitySet("point_cloud_points", 0, np.asarray(plan.point_ids))
        # Fixed-support refresh is implemented by ``refresh`` below.
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
            reconstruction_id=stencils.prepared_id,
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
            diagnostics=(
                f"local-stencil-report:{stencils.report.report_id}",
                "neighborhood-motion-bound:strict-displacement-less-than-gap/4"
                if plan.stencil.support is None
                else "smooth-support-motion-bound:strict-displacement-less-than-envelope",
            ),
            resource_counts={
                "points": count,
                "dimension": dimension,
                "neighbor_capacity": plan.neighbors,
                "polynomial_features": math.comb(
                    dimension + plan.stencil.polynomial_degree,
                    plan.stencil.polynomial_degree,
                ),
                "refused_stencil_rows": stencils.report.refused_rows,
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
        self.plan = plan
        self.coordinates = plan.points
        self.relation = relation
        self.stencils = stencils
        self.derivative_weights = derivative_weights
        self.mixed_weights = tuple(zip(indices, stencils.weights, strict=True))
        self.report = stencils.report
        self.key = key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-point-cloud",
                "plan": plan.plan_id,
                "stencils": stencils.prepared_id,
            }
        )
        self.numeric_version = "1"
        self.preparation = preparation
        self.trust_radius = jnp.asarray(trust)

    @property
    def spatial_dimension(self) -> int:
        return self.plan.points.shape[1]

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.plan.points.shape[0],)

    @property
    def quadrature_weights(self) -> Array:
        return self.plan.quadrature_weights

    @property
    def discretization_id(self) -> str:
        return self.prepared_id

    @property
    def points(self) -> Array:
        return self.coordinates

    @property
    def stable_ids(self) -> Array:
        return self.plan.point_ids

    def refresh(self, points: ArrayLike, /) -> PointCloudRefresh:
        """Refit derivative stencils at moved points on the frozen support.

        Traceable and differentiable in ``points``. Displacement is measured
        from the anchored plan geometry; measures, boundary data, stable
        identities and the relation stay anchored to the epoch. The candidate
        is returned with status; consumers must inspect ``accepted``. A
        refused candidate carries NaN derivative weights.
        """
        coordinates = jnp.asarray(points, dtype=self.coordinates.dtype)
        if coordinates.shape != self.coordinates.shape:
            raise ValueError("Point-cloud refresh preserves point count and dimension.")
        refreshed = refresh_local_stencils(self.stencils, coordinates, coordinates)
        stencils = refreshed.stencils
        derivative_weights = tuple(
            (stencils.weights[2 * axis], stencils.weights[2 * axis + 1])
            for axis in range(self.spatial_dimension)
        )
        mixed_weights = tuple(
            (index, weights)
            for (index, _), weights in zip(
                self.mixed_weights, stencils.weights, strict=True
            )
        )
        candidate = eqx.tree_at(
            lambda item: (
                item.coordinates,
                item.stencils,
                item.derivative_weights,
                item.mixed_weights,
            ),
            self,
            (coordinates, stencils, derivative_weights, mixed_weights),
        )
        return PointCloudRefresh(
            discretization=candidate,
            status=refreshed.status,
            accepted=refreshed.accepted,
            displacement=refreshed.displacement,
            support_margin=refreshed.support_margin,
            evidence=stencils.evidence,
        )

    def coordinate_sensitivity(
        self, points: ArrayLike, /
    ) -> PointCloudCoordinateSensitivity:
        """Fixed-support coordinate linearization of every derivative stencil.

        The published map sends point coordinates to the refitted weights of
        every prepared derivative functional (``mixed_weights`` order) on the
        frozen relation. Its JVP/VJP are genuine derivatives of that map with
        per-row rank and conditioning evidence. A nearest-neighbor support is
        differentiable only strictly inside its selection gap; a smooth
        fixed-radius support across neighbors entering or leaving the radius
        while the motion stays strictly inside its envelope. A refused
        refresh (support exit, rank loss, nonfinite coordinates) publishes
        NaN weights, tangents and cotangents.
        """
        coordinates = jnp.asarray(points, dtype=self.coordinates.dtype)
        if coordinates.shape != self.coordinates.shape:
            raise ValueError(
                "Point-cloud sensitivity preserves point count and dimension."
            )

        def weights(
            moved: Array,
        ) -> tuple[tuple[Array, ...], tuple[Array, Array, LocalStencilEvidence]]:
            refreshed = self.refresh(moved)
            return (
                tuple(item for _, item in refreshed.discretization.mixed_weights),
                (refreshed.status, refreshed.accepted, refreshed.evidence),
            )

        linearization = prepare_linearization(weights, coordinates, has_aux=True)
        status, accepted, evidence = linearization.auxiliary
        return PointCloudCoordinateSensitivity(
            linearization=linearization,
            status=status,
            accepted=accepted,
            evidence=evidence,
        )

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
            else tuple(axes)
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

    def mixed_partial_derivative(
        self, state: ArrayLike, /, *, multi_index: tuple[int, ...]
    ) -> Array:
        index = tuple(_integer(order, "derivative order", 0) for order in multi_index)
        for prepared_index, weights in self.mixed_weights:
            if index == prepared_index:
                return self._apply_weights(self._validate_state(state), weights)
        raise ValueError(
            "Point-cloud mixed derivatives require a spatial multi-index of total order one/two."
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
        if dual:
            raise ValueError("Point-cloud strong divergence has no certified dual route.")
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


@final
class PointCloudRefresh(StrictModule):
    """Status-returning fixed-support point-cloud refresh candidate.

    ``status`` uses ``LocalStencilRefreshStatus``. ``discretization`` retains the
    anchored plan (reference geometry, stable ids, measures, boundary data),
    the frozen relation and the refitted derivative weights.
    """

    discretization: PreparedPointCloudDiscretization
    status: Array
    accepted: Array
    displacement: Array
    support_margin: Array
    evidence: LocalStencilEvidence


@final
class PointCloudCoordinateSensitivity(StrictModule):
    """Fixed-support coordinate linearization with its admission evidence.

    ``linearization`` retains the published map's primal value and its
    JVP/VJP at one coordinate state. ``status`` uses
    ``LocalStencilRefreshStatus`` and ``accepted`` admits the whole map; a
    refused map is NaN with NaN tangents and cotangents. ``evidence`` is the
    native per-row rank/conditioning evidence: ``LocalStencilEvidence`` for
    discretization stencil rows, ``FieldQueryEvidence`` for reconstruction
    queries.
    """

    linearization: PreparedLinearization
    status: Array
    accepted: Array
    evidence: LocalStencilEvidence | FieldQueryEvidence


__all__ = [
    "PointCloudCoordinateSensitivity",
    "PointCloudPlan",
    "PointCloudRefresh",
    "PreparedPointCloudDiscretization",
]

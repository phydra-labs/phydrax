#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._distance import edge_edge_distance
from ...discretization.contact._implicit_geometry import PlaneContactGeometry
from ...discretization.contact._participant import (
    AbstractContactParticipant,
    ContactParticipantScene,
    FunctionContactParticipant,
)
from ...discretization.contact._search import (
    contact_epoch_identity,
    ContactCandidateEpoch,
    ContactRouteLayouts,
    ContactSearchStatus,
    scene_route_geometry,
)
from ...discretization.contact._stencils import (
    ContactPrimitiveProfile,
    ContactStencilBatch,
    ContactStencilKind,
    ContactStencilLayout,
    empty_contact_stencil_batch,
    prepare_contact_stencil_batch,
)
from ...discretization.contact._surface import CollisionSurfacePlan
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    dual_transpose,
    DualSpace,
    FunctionLinearOperator,
    MaterializationPolicy,
    materialize,
)
from ...typing import checked
from ._cone import (
    contact_cone_space,
    ContactBlockLayout,
    ContactCertificatePolicy,
    ContactConeBranch,
    ContactConeEvidence,
    ContactConeProgram,
    ContactConeResult,
    ContactConeSolverPlan,
    dense_contact_response,
    solve_contact_cone,
)
from ._manifold import (
    ContactFrameTransport,
    ContactManifoldObservation,
    ContactManifoldPolicy,
    ContactManifoldState,
    ContactManifoldTransition,
    update_contact_manifold,
)
from ._rod_capsule import (
    PreparedRodCapsuleGeometry,
    ReducedRodCapsuleContactParticipant,
)


_ROD_VERTEX_LINEAR_PATH = "rod-vertex-linear-path"
"""Swept rod epochs bind coverage to linearly interpolated vertex paths."""


class RodContactSearchRoute(IntEnum):
    DENSE = 0
    LBVH = 1


class RodContactSearchFailure(IntEnum):
    NONE = 0
    CAPACITY_OVERFLOW = 1
    TREE_DEPTH_OVERFLOW = 2
    TRAVERSAL_OVERFLOW = 3
    WITNESS_FAILURE = 4
    INVALID_ROUTE_GEOMETRY = 5


class RodContactSearchEvidence(StrictModule):
    candidate_count: Array
    required_capacity: Array
    overflow_count: Array
    adjacency_filtered_count: Array
    policy_filtered_count: Array
    aabb_test_count: Array
    narrow_phase_count: Array
    traversal_visits: Array
    tree_depth: Array
    finite: Array
    complete: Array
    successful: Array
    failure: Array
    route: RodContactSearchRoute = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class RodContactWitnessBatch(StrictModule, NonTrainableState):
    """Exact circular-capsule witnesses in canonical candidate-slot order."""

    vertex_indices: Array
    left_feature_ids: Array
    right_feature_ids: Array
    route_keys: Array
    stencil_kinds: Array
    left_segment_indices: Array
    right_segment_indices: Array
    left_parameters: Array
    right_parameters: Array
    coefficients: Array
    left_centerline_witness: Array
    right_centerline_witness: Array
    left_surface_witness: Array
    right_surface_witness: Array
    left_axis: Array
    right_axis: Array
    normal: Array
    tangent_basis: Array
    centerline_distance: Array
    physical_gap: Array
    activation_gap: Array
    left_radius: Array
    right_radius: Array
    valid: Array
    finite: Array
    capacity: int = eqx.field(static=True)
    batch_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertex_indices: ArrayLike,
        left_feature_ids: ArrayLike,
        right_feature_ids: ArrayLike,
        route_keys: ArrayLike,
        stencil_kinds: ArrayLike,
        left_segment_indices: ArrayLike,
        right_segment_indices: ArrayLike,
        left_parameters: ArrayLike,
        right_parameters: ArrayLike,
        coefficients: ArrayLike,
        left_centerline_witness: ArrayLike,
        right_centerline_witness: ArrayLike,
        left_surface_witness: ArrayLike,
        right_surface_witness: ArrayLike,
        left_axis: ArrayLike,
        right_axis: ArrayLike,
        normal: ArrayLike,
        tangent_basis: ArrayLike,
        centerline_distance: ArrayLike,
        physical_gap: ArrayLike,
        activation_gap: ArrayLike,
        left_radius: ArrayLike,
        right_radius: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        capacity: int,
        finite: ArrayLike | None = None,
        batch_id: str | None = None,
    ) -> None:
        count = int(capacity)
        if count <= 0:
            raise ValueError("Rod contact witness capacity must be positive.")
        indices = np.asarray(vertex_indices)
        left = np.asarray(left_feature_ids)
        right = np.asarray(right_feature_ids)
        keys = np.asarray(route_keys)
        kinds = np.asarray(stencil_kinds)
        left_segments = np.asarray(left_segment_indices)
        right_segments = np.asarray(right_segment_indices)
        left_parameter = np.asarray(left_parameters)
        right_parameter = np.asarray(right_parameters)
        coefficient = np.asarray(coefficients)
        left_center = np.asarray(left_centerline_witness)
        right_center = np.asarray(right_centerline_witness)
        left_surface = np.asarray(left_surface_witness)
        right_surface = np.asarray(right_surface_witness)
        left_axis_ = np.asarray(left_axis)
        right_axis_ = np.asarray(right_axis)
        normal_ = np.asarray(normal)
        basis = np.asarray(tangent_basis)
        distance = np.asarray(centerline_distance)
        gap = np.asarray(physical_gap)
        activation = np.asarray(activation_gap)
        left_radius_ = np.asarray(left_radius)
        right_radius_ = np.asarray(right_radius)
        active = np.asarray(valid, dtype=np.bool_)
        if indices.shape != (count, 4) or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("vertex_indices must be an integer (capacity, 4) array.")
        vector_fields = (left, right, keys, kinds, left_segments, right_segments)
        if any(value.shape != (count,) for value in vector_fields) or not all(
            np.issubdtype(value.dtype, np.integer) for value in vector_fields
        ):
            raise TypeError(
                "Rod contact identifiers, kinds, and segments must be integer vectors."
            )
        scalar_fields = (
            left_parameter,
            right_parameter,
            distance,
            gap,
            activation,
            left_radius_,
            right_radius_,
        )
        if any(value.shape != (count,) for value in scalar_fields):
            raise ValueError(
                "Rod contact scalar witness fields must have capacity shape."
            )
        if coefficient.shape != (count, 4):
            raise ValueError("Rod contact coefficients must have shape (capacity, 4).")
        for value in (
            left_center,
            right_center,
            left_surface,
            right_surface,
            left_axis_,
            right_axis_,
            normal_,
        ):
            if value.shape != (count, 3):
                raise ValueError(
                    "Rod contact vector witness fields must have shape (capacity, 3)."
                )
        if basis.shape != (count, 3, 2):
            raise ValueError(
                "Rod contact tangent bases must have shape (capacity, 3, 2)."
            )
        if active.shape != (count,):
            raise ValueError("Rod contact validity must have capacity shape.")
        finite_ = (
            np.ones((count,), dtype=np.bool_)
            if finite is None
            else np.asarray(finite, dtype=np.bool_)
        )
        if finite_.shape != (count,):
            raise ValueError("Rod contact finite evidence must have capacity shape.")
        if np.any(active & ((left < 0) | (right < 0) | (keys < 0))):
            raise ValueError("Active rod contact identifiers must be nonnegative.")
        if np.unique(keys[active]).size != int(np.count_nonzero(active)):
            raise ValueError("Active rod contact route keys must be unique.")
        if np.any(left_radius_ < 0.0) or np.any(right_radius_ < 0.0):
            raise ValueError("Capsule radii must be nonnegative.")
        numeric = (
            left_parameter,
            right_parameter,
            coefficient,
            left_center,
            right_center,
            left_surface,
            right_surface,
            left_axis_,
            right_axis_,
            normal_,
            basis,
            distance,
            gap,
            activation,
            left_radius_,
            right_radius_,
        )
        if any(
            np.any(active & ~np.all(np.isfinite(value), axis=tuple(range(1, value.ndim))))
            if value.ndim > 1
            else np.any(active & ~np.isfinite(value))
            for value in numeric
        ):
            raise ValueError("Active rod contact witnesses must be finite.")
        generated = canonical_fingerprint(
            {
                "kind": "rod-contact-witness-batch",
                "indices": array_tree_fingerprint(indices),
                "keys": array_tree_fingerprint(keys),
                "valid": array_tree_fingerprint(active),
            }
        )
        identifier = generated if batch_id is None else str(batch_id)
        if not identifier:
            raise ValueError("batch_id must be nonempty or None.")
        dtype = jnp.result_type(*scalar_fields, coefficient)
        self.vertex_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.left_feature_ids = jnp.asarray(left, dtype=jnp.int64)
        self.right_feature_ids = jnp.asarray(right, dtype=jnp.int64)
        self.stencil_kinds = jnp.asarray(kinds, dtype=jnp.int32)
        self.left_segment_indices = jnp.asarray(left_segments, dtype=jnp.int32)
        self.right_segment_indices = jnp.asarray(right_segments, dtype=jnp.int32)
        self.route_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.left_parameters = jnp.asarray(left_parameter, dtype=dtype)
        self.right_parameters = jnp.asarray(right_parameter, dtype=dtype)
        self.coefficients = jnp.asarray(coefficient, dtype=dtype)
        self.left_centerline_witness = jnp.asarray(left_center, dtype=dtype)
        self.right_centerline_witness = jnp.asarray(right_center, dtype=dtype)
        self.left_surface_witness = jnp.asarray(left_surface, dtype=dtype)
        self.right_surface_witness = jnp.asarray(right_surface, dtype=dtype)
        self.left_axis = jnp.asarray(left_axis_, dtype=dtype)
        self.right_axis = jnp.asarray(right_axis_, dtype=dtype)
        self.normal = jnp.asarray(normal_, dtype=dtype)
        self.tangent_basis = jnp.asarray(basis, dtype=dtype)
        self.centerline_distance = jnp.asarray(distance, dtype=dtype)
        self.physical_gap = jnp.asarray(gap, dtype=dtype)
        self.activation_gap = jnp.asarray(activation, dtype=dtype)
        self.left_radius = jnp.asarray(left_radius_, dtype=dtype)
        self.right_radius = jnp.asarray(right_radius_, dtype=dtype)
        self.valid = jnp.asarray(active)
        self.finite = jnp.asarray(finite_)
        self.capacity = count
        self.batch_id = identifier

    @classmethod
    def empty(
        cls, capacity: int, /, *, dtype: Any = np.float64
    ) -> RodContactWitnessBatch:
        count = int(capacity)
        zeros = np.zeros((count,), dtype=dtype)
        vectors = np.zeros((count, 3), dtype=dtype)
        return cls(
            np.full((count, 4), -1, dtype=np.int32),
            np.zeros((count,), dtype=np.int64),
            np.zeros((count,), dtype=np.int64),
            np.zeros((count,), dtype=np.int64),
            np.zeros((count,), dtype=np.int32),
            np.full((count,), -1, dtype=np.int32),
            np.full((count,), -1, dtype=np.int32),
            zeros,
            zeros,
            np.zeros((count, 4), dtype=dtype),
            vectors,
            vectors,
            vectors,
            vectors,
            vectors,
            vectors,
            vectors,
            np.zeros((count, 3, 2), dtype=dtype),
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            np.zeros((count,), dtype=np.bool_),
            capacity=count,
        )


class RodContactSearchResult(StrictModule, NonTrainableState):
    epoch: ContactCandidateEpoch
    witnesses: RodContactWitnessBatch
    evidence: RodContactSearchEvidence
    pair_witnesses: RodContactWitnessBatch
    plane_witnesses: RodContactWitnessBatch | None

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def pair_witness(self) -> RodContactWitnessBatch:
        return self.pair_witnesses

    @property
    def plane_witness(self) -> RodContactWitnessBatch | None:
        return self.plane_witnesses


class RodContactSearchPlan(StrictModule, NonTrainableState):
    """Fixed-budget exact circular-capsule search policy.

    Dense search is the correctness authority.  The LBVH route changes only the
    broad phase; both routes canonicalize exact narrow-phase records by stable
    feature-pair keys before packing the candidate epoch.
    """

    capacity: int = eqx.field(static=True)
    plane_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    route: RodContactSearchRoute = eqx.field(static=True)
    maximum_tree_depth: int = eqx.field(static=True)
    maximum_traversal_visits: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        capacity: int,
        activation_distance: float,
        plane_capacity: int = 0,
        route: str | RodContactSearchRoute = "dense",
        maximum_tree_depth: int = 64,
        maximum_traversal_visits: int = 1_000_000,
    ) -> None:
        capacity_ = int(capacity)
        plane_capacity_ = int(plane_capacity)
        activation = float(activation_distance)
        depth = int(maximum_tree_depth)
        visits = int(maximum_traversal_visits)
        if capacity_ <= 0 or plane_capacity_ < 0:
            raise ValueError("capacity must be positive and plane_capacity nonnegative.")
        if not isfinite(activation) or activation < 0.0:
            raise ValueError("activation_distance must be finite and nonnegative.")
        if depth <= 0 or visits <= 0:
            raise ValueError("LBVH depth and traversal budgets must be positive.")
        if isinstance(route, RodContactSearchRoute):
            route_ = route
        else:
            route_name = str(route).lower()
            if route_name not in ("dense", "lbvh"):
                raise ValueError("route must be 'dense' or 'lbvh'.")
            route_ = (
                RodContactSearchRoute.DENSE
                if route_name == "dense"
                else RodContactSearchRoute.LBVH
            )
        self.capacity = capacity_
        self.plane_capacity = plane_capacity_
        self.activation_distance = activation
        self.route = route_
        self.maximum_tree_depth = depth
        self.maximum_traversal_visits = visits
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rod-contact-search-plan",
                "capacity": capacity_,
                "plane_capacity": plane_capacity_,
                "activation_distance": activation.hex(),
                "route": int(route_),
                "maximum_tree_depth": depth,
                "maximum_traversal_visits": visits,
            }
        )

    @property
    def total_capacity(self) -> int:
        return self.capacity + self.plane_capacity

    def prepare(
        self,
        geometry: PreparedRodCapsuleGeometry | CollisionSurfacePlan,
        /,
        *,
        planes: Sequence[PlaneContactGeometry] = (),
    ) -> PreparedRodContactSearch:
        return PreparedRodContactSearch(self, geometry, planes)


class PreparedRodContactSearch(StrictModule, NonTrainableState):
    plan: RodContactSearchPlan
    surface_plan: CollisionSurfacePlan
    geometry: PreparedRodCapsuleGeometry | None
    planes: tuple[PlaneContactGeometry, ...]
    surface_segment_indices: Array
    edges: Array
    edge_feature_ids: Array
    edge_participant_ids: Array
    edge_body_ids: Array
    edge_static_mask: Array
    edge_physical_radius: Array
    edge_contact_extent: Array
    excluded_vertex_pairs: Array
    allowed_participant_pairs: Array
    scene: ContactParticipantScene
    layouts: ContactRouteLayouts
    edge_scene_features: Array
    edge_ranks: Array
    plane_scene_features: Array
    plane_ranks: Array
    pair_policy_unrestricted: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: RodContactSearchPlan,
        geometry: PreparedRodCapsuleGeometry | CollisionSurfacePlan,
        planes: Sequence[PlaneContactGeometry],
        /,
    ) -> None:
        if isinstance(geometry, PreparedRodCapsuleGeometry):
            capsule_geometry: PreparedRodCapsuleGeometry | None = geometry
            surface_plan = geometry.surface_plan
            segment_indices = np.asarray(geometry.surface_edge_order, dtype=np.int32)
        elif isinstance(geometry, CollisionSurfacePlan):
            capsule_geometry = None
            surface_plan = geometry
            segment_indices = np.arange(surface_plan.edge_count, dtype=np.int32)
        else:
            raise TypeError(
                "geometry must be PreparedRodCapsuleGeometry or CollisionSurfacePlan."
            )
        planes_ = tuple(planes)
        if not all(isinstance(value, PlaneContactGeometry) for value in planes_):
            raise TypeError("planes must contain PlaneContactGeometry values.")
        if planes_ and plan.plane_capacity <= 0:
            raise ValueError("plane_capacity must be positive when planes are prepared.")
        if not planes_ and plan.plane_capacity:
            raise ValueError("A positive plane_capacity requires at least one plane.")
        if any(value.ambient_dimension != 3 for value in planes_):
            raise ValueError("Rod contact planes must be three-dimensional.")
        if surface_plan.ambient_dimension != 3 or surface_plan.face_count != 0:
            raise ValueError(
                "Rod capsule contact requires a three-dimensional edge surface."
            )
        if surface_plan.edge_count <= 0:
            raise ValueError("Rod capsule contact requires at least one segment.")
        features = surface_plan.feature_policy
        edge_slice = features.edge_slice
        edge_ids = np.asarray(features.feature_ids[edge_slice], dtype=np.int64)
        if np.unique(edge_ids).size != edge_ids.size:
            raise ValueError("Rod capsule edge feature IDs must be unique.")
        radius = np.asarray(features.physical_radius[edge_slice], dtype=np.float64)
        extent = np.asarray(features.contact_extent[edge_slice], dtype=np.float64)
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError("Circular rod capsules require positive finite edge radii.")
        if np.any(~np.isfinite(extent)) or np.any(extent < radius):
            raise ValueError("Rod capsule contact extents are invalid.")
        self.plan = plan
        plane_feature_ids = [
            int(
                np.asarray(
                    plane.feature_policy.feature_ids[plane.feature_policy.analytic_slice]
                )[0]
            )
            for plane in planes_
        ]
        if (
            len(set(plane_feature_ids)) != len(plane_feature_ids)
            or np.intersect1d(
                edge_ids, np.asarray(plane_feature_ids, dtype=np.int64)
            ).size
        ):
            raise ValueError(
                "Rod and plane analytic feature IDs must be globally distinct."
            )
        self.surface_plan = surface_plan
        self.geometry = capsule_geometry
        self.planes = planes_
        self.surface_segment_indices = jnp.asarray(segment_indices, dtype=jnp.int32)
        self.edges = jnp.asarray(surface_plan.edges, dtype=jnp.int32)
        self.edge_feature_ids = jnp.asarray(edge_ids, dtype=jnp.int64)
        self.edge_participant_ids = jnp.asarray(
            features.participant_ids[edge_slice], dtype=jnp.int64
        )
        self.edge_body_ids = jnp.asarray(features.body_ids[edge_slice], dtype=jnp.int64)
        self.edge_static_mask = jnp.asarray(
            features.static_mask[edge_slice], dtype=jnp.bool_
        )
        self.edge_physical_radius = jnp.asarray(radius)
        self.edge_contact_extent = jnp.asarray(extent)
        self.excluded_vertex_pairs = jnp.asarray(
            surface_plan.pair_policy.excluded_vertex_pairs, dtype=jnp.int64
        )
        self.allowed_participant_pairs = jnp.asarray(
            surface_plan.pair_policy.allowed_participant_pairs, dtype=jnp.int64
        )
        self.pair_policy_unrestricted = surface_plan.pair_policy.unrestricted
        vertex_space = ArraySpace((surface_plan.vertex_count, 3), dtype=np.float64)

        def vertex_positions(configuration: Array) -> Array:
            return configuration

        scene = ContactParticipantScene(
            (
                FunctionContactParticipant(
                    surface_plan,
                    vertex_space,
                    vertex_positions,
                    tangent_space=vertex_space,
                    participant_id=canonical_fingerprint(
                        {
                            "kind": "rod-contact-vertex-participant",
                            "surface": surface_plan.topology_id,
                            "features": features.policy_id,
                        }
                    ),
                ),
            ),
            planes=planes_,
        )
        layouts = ContactRouteLayouts(
            scene,
            capacities=(0, 0, plan.capacity, 0, 0, plan.plane_capacity),
            profile=ContactPrimitiveProfile.MESH,
        )
        ranks = np.asarray(scene.feature_catalogue.canonical_ranks, dtype=np.int64)
        edge_features = surface_plan.vertex_count + np.arange(
            surface_plan.edge_count, dtype=np.int64
        )
        plane_features = (
            surface_plan.vertex_count
            + surface_plan.edge_count
            + np.arange(len(planes_), dtype=np.int64)
        )
        self.scene = scene
        self.layouts = layouts
        self.edge_scene_features = jnp.asarray(edge_features)
        self.edge_ranks = jnp.asarray(ranks[edge_features])
        self.plane_scene_features = jnp.asarray(plane_features)
        self.plane_ranks = jnp.asarray(ranks[plane_features])
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rod-contact-search",
                "plan": plan.plan_id,
                "surface": surface_plan.topology_id,
                "features": features.policy_id,
                "planes": [value.geometry_id for value in planes_],
                "segment_indices": array_tree_fingerprint(segment_indices),
                "scene": scene.scene_id,
                "layouts": layouts.layouts_id,
            }
        )

    @property
    def catalogue_id(self) -> str:
        return self.scene.feature_catalogue.catalogue_id

    def search(
        self,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
    ) -> RodContactSearchResult:
        return _search_rod_capsules(self, positions, end_positions)


class _BVHNode:
    __slots__ = ("lower", "upper", "left", "right", "edge", "count", "depth")

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        left: _BVHNode | None = None,
        right: _BVHNode | None = None,
        edge: int = -1,
        count: int = 1,
        depth: int = 1,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.left = left
        self.right = right
        self.edge = edge
        self.count = count
        self.depth = depth

    @property
    def leaf(self) -> bool:
        return self.edge >= 0


def _morton_codes(centers: np.ndarray, /) -> np.ndarray:
    lower = centers.min(axis=0)
    extent = centers.max(axis=0) - lower
    normalized = (centers - lower) / np.where(extent > 0.0, extent, 1.0)
    integer = np.clip(np.floor(normalized * 1023.0), 0, 1023).astype(np.uint32)
    codes = np.zeros((centers.shape[0],), dtype=np.uint64)
    for bit in range(10):
        codes |= ((integer[:, 0] >> bit) & 1).astype(np.uint64) << (3 * bit)
        codes |= ((integer[:, 1] >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        codes |= ((integer[:, 2] >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    return codes


def _build_lbvh(
    order: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> _BVHNode:
    if order.size == 1:
        edge = int(order[0])
        return _BVHNode(lower[edge], upper[edge], edge=edge)
    middle = order.size // 2
    left = _build_lbvh(order[:middle], lower, upper)
    right = _build_lbvh(order[middle:], lower, upper)
    return _BVHNode(
        np.minimum(left.lower, right.lower),
        np.maximum(left.upper, right.upper),
        left=left,
        right=right,
        count=left.count + right.count,
        depth=1 + max(left.depth, right.depth),
    )


def _bounds_overlap(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
    /,
) -> bool:
    return bool(np.all(left_lower <= right_upper) and np.all(right_lower <= left_upper))


def _lbvh_candidate_pairs(
    lower: np.ndarray,
    upper: np.ndarray,
    feature_ids: np.ndarray,
    maximum_depth: int,
    maximum_visits: int,
    /,
) -> tuple[list[tuple[int, int]], int, int, RodContactSearchFailure]:
    centers = 0.5 * (lower + upper)
    codes = _morton_codes(centers)
    order = np.lexsort((feature_ids, codes))
    root = _build_lbvh(order, lower, upper)
    if root.depth > maximum_depth:
        return [], 0, root.depth, RodContactSearchFailure.TREE_DEPTH_OVERFLOW
    pairs: list[tuple[int, int]] = []
    stack: list[tuple[_BVHNode, _BVHNode]] = [(root, root)]
    visits = 0
    while stack:
        if visits >= maximum_visits:
            return pairs, visits, root.depth, RodContactSearchFailure.TRAVERSAL_OVERFLOW
        first, second = stack.pop()
        visits += 1
        if not _bounds_overlap(first.lower, first.upper, second.lower, second.upper):
            continue
        if first is second:
            if first.leaf:
                continue
            if first.left is None or first.right is None:
                raise RuntimeError("Malformed rod contact LBVH node.")
            stack.append((first.right, first.right))
            stack.append((first.left, first.right))
            stack.append((first.left, first.left))
            continue
        if first.leaf and second.leaf:
            pairs.append((min(first.edge, second.edge), max(first.edge, second.edge)))
            continue
        split_first = (not first.leaf) and (second.leaf or first.count >= second.count)
        if split_first:
            if first.left is None or first.right is None:
                raise RuntimeError("Malformed rod contact LBVH branch.")
            stack.append((first.right, second))
            stack.append((first.left, second))
        else:
            if second.left is None or second.right is None:
                raise RuntimeError("Malformed rod contact LBVH branch.")
            stack.append((first, second.right))
            stack.append((first, second.left))
    return list(dict.fromkeys(pairs)), visits, root.depth, RodContactSearchFailure.NONE


def _pair_explicitly_allowed(
    first_participant: int,
    second_participant: int,
    allowed: np.ndarray,
    unrestricted: bool,
    /,
) -> bool:
    pair = np.sort(np.asarray((first_participant, second_participant), dtype=np.int64))
    explicit = bool(allowed.size and np.any(np.all(allowed == pair, axis=1)))
    return unrestricted or explicit


def _edge_pair_allowed(
    first: int,
    second: int,
    edges: np.ndarray,
    participants: np.ndarray,
    static: np.ndarray,
    excluded: set[tuple[int, int]],
    allowed_participants: np.ndarray,
    unrestricted: bool,
    /,
) -> tuple[bool, bool]:
    first_vertices = tuple(edges[first])
    second_vertices = tuple(edges[second])
    if set(first_vertices) & set(second_vertices):
        return False, True
    if any(
        (min(left, right), max(left, right)) in excluded
        for left in first_vertices
        for right in second_vertices
    ):
        return False, True
    if bool(static[first]) and bool(static[second]):
        return False, False
    permitted = _pair_explicitly_allowed(
        int(participants[first]),
        int(participants[second]),
        allowed_participants,
        unrestricted,
    )
    return permitted, False


def _objective_tangent_basis(
    normal: np.ndarray,
    left_axis: np.ndarray,
    right_axis: np.ndarray,
    /,
) -> np.ndarray:
    first = left_axis - np.dot(left_axis, normal) * normal
    norm = float(np.linalg.norm(first))
    if norm <= 64.0 * np.finfo(normal.dtype).eps:
        first = right_axis - np.dot(right_axis, normal) * normal
        norm = float(np.linalg.norm(first))
    if norm <= 64.0 * np.finfo(normal.dtype).eps:
        axis = np.zeros((3,), dtype=normal.dtype)
        axis[int(np.argmin(np.abs(normal)))] = 1.0
        first = np.cross(normal, axis)
        norm = float(np.linalg.norm(first))
    first = first / norm
    second = np.cross(normal, first)
    second = second / np.linalg.norm(second)
    return np.stack((first, second), axis=-1)


def _ordered_rows(
    keys: np.ndarray, capacity: int, /
) -> np.ndarray:
    return np.argsort(keys, kind="stable")[:capacity]


def _stencil_batch(
    prepared: PreparedRodContactSearch,
    layout: ContactStencilLayout,
    left: np.ndarray,
    right: np.ndarray,
    slots: np.ndarray,
    separation: np.ndarray,
    active: np.ndarray,
    actual: int,
    overflow: int,
    dtype: DTypeLike,
    /,
) -> ContactStencilBatch:
    return prepare_contact_stencil_batch(
        prepared.scene,
        layout,
        left,
        right,
        witness_slots=slots,
        minimum_separation=separation,
        valid=active,
        actual_count=actual,
        overflow_count=overflow,
        dtype=dtype,
    )


def _pack_witnesses(
    prepared: PreparedRodContactSearch,
    positions: np.ndarray,
    records: list[tuple[int, int]],
    failure: RodContactSearchFailure,
    /,
) -> tuple[ContactStencilBatch, RodContactWitnessBatch, RodContactSearchFailure]:
    """Pack edge-edge capsule witnesses in canonical key order.

    Rows are oriented with the lower canonical feature rank on the left and
    use the single unique-closest-feature witness slot of the rod profile.
    """

    capacity = prepared.plan.capacity
    layout = prepared.layouts.edge_edge
    actual = len(records)
    overflow = max(actual - capacity, 0)
    edges = np.asarray(prepared.edges, dtype=np.int32)
    feature_ids = np.asarray(prepared.edge_feature_ids, dtype=np.int64)
    ranks = np.asarray(prepared.edge_ranks, dtype=np.int64)
    scene_features = np.asarray(prepared.edge_scene_features, dtype=np.int64)
    radii = np.asarray(prepared.edge_physical_radius, dtype=positions.dtype)
    extents = np.asarray(prepared.edge_contact_extent, dtype=positions.dtype)
    ordered_records: list[tuple[int, int, int]] = []
    if records:
        oriented = [
            (first, second) if ranks[first] < ranks[second] else (second, first)
            for first, second in records
        ]
        keys = np.asarray(
            layout.schema.encode(
                int(ContactStencilKind.EDGE_EDGE),
                np.asarray([ranks[first] for first, _ in oriented]),
                np.asarray([ranks[second] for _, second in oriented]),
                np.zeros((len(oriented),), dtype=np.int64),
            )
        ).astype(np.int64)
        ordered_records = [
            (oriented[index][0], oriented[index][1], int(keys[index]))
            for index in _ordered_rows(keys, capacity)
        ]
    indices = np.full((capacity, 4), -1, dtype=np.int32)
    left_features = np.full((capacity,), -1, dtype=np.int64)
    right_features = np.full((capacity,), -1, dtype=np.int64)
    left_ids = np.zeros((capacity,), dtype=np.int64)
    right_ids = np.zeros((capacity,), dtype=np.int64)
    route_keys = np.zeros((capacity,), dtype=np.int64)
    stencil_kinds = np.full(
        (capacity,), int(ContactStencilKind.EDGE_EDGE), dtype=np.int32
    )
    left_segments = np.full((capacity,), -1, dtype=np.int32)
    right_segments = np.full((capacity,), -1, dtype=np.int32)
    separation = np.zeros((capacity,), dtype=positions.dtype)
    active = np.zeros((capacity,), dtype=np.bool_)
    parameters = np.zeros((capacity, 2), dtype=positions.dtype)
    coefficients = np.zeros((capacity, 4), dtype=positions.dtype)
    left_center = np.zeros((capacity, 3), dtype=positions.dtype)
    right_center = np.zeros((capacity, 3), dtype=positions.dtype)
    left_surface = np.zeros((capacity, 3), dtype=positions.dtype)
    right_surface = np.zeros((capacity, 3), dtype=positions.dtype)
    left_axis = np.zeros((capacity, 3), dtype=positions.dtype)
    right_axis = np.zeros((capacity, 3), dtype=positions.dtype)
    normal = np.zeros((capacity, 3), dtype=positions.dtype)
    tangent = np.zeros((capacity, 3, 2), dtype=positions.dtype)
    centerline_distance = np.zeros((capacity,), dtype=positions.dtype)
    physical_gap = np.zeros((capacity,), dtype=positions.dtype)
    activation_gap = np.zeros((capacity,), dtype=positions.dtype)
    left_radius = np.zeros((capacity,), dtype=positions.dtype)
    right_radius = np.zeros((capacity,), dtype=positions.dtype)
    finite = np.zeros((capacity,), dtype=np.bool_)
    witness_failure = False
    segment_indices = np.asarray(prepared.surface_segment_indices)
    for slot, (first, second, key) in enumerate(ordered_records):
        first_edge = edges[first]
        second_edge = edges[second]
        evaluation = edge_edge_distance(
            positions[first_edge[0]],
            positions[first_edge[1]],
            positions[second_edge[0]],
            positions[second_edge[1]],
        )
        coefficient = np.asarray(evaluation.coefficients, dtype=positions.dtype)
        normal_ = np.asarray(evaluation.normal, dtype=positions.dtype)
        finite_ = bool(np.asarray(evaluation.finite)) and bool(
            np.asarray(evaluation.nondegenerate)
        )
        indices[slot] = np.concatenate((first_edge, second_edge))
        left_features[slot] = scene_features[first]
        right_features[slot] = scene_features[second]
        left_ids[slot] = feature_ids[first]
        right_ids[slot] = feature_ids[second]
        route_keys[slot] = key
        left_segments[slot] = int(segment_indices[first])
        right_segments[slot] = int(segment_indices[second])
        separation[slot] = extents[first] + extents[second]
        parameters[slot] = (coefficient[1], -coefficient[3])
        coefficients[slot] = coefficient
        left_center[slot] = np.asarray(evaluation.left_witness)
        right_center[slot] = np.asarray(evaluation.right_witness)
        left_radius[slot] = radii[first]
        right_radius[slot] = radii[second]
        distance = float(
            np.sqrt(max(float(np.asarray(evaluation.squared_distance)), 0.0))
        )
        centerline_distance[slot] = distance
        physical_gap[slot] = distance - radii[first] - radii[second]
        activation_gap[slot] = distance - extents[first] - extents[second]
        left_axis_ = positions[first_edge[1]] - positions[first_edge[0]]
        right_axis_ = positions[second_edge[1]] - positions[second_edge[0]]
        left_axis_norm = np.linalg.norm(left_axis_)
        right_axis_norm = np.linalg.norm(right_axis_)
        finite_ = finite_ and left_axis_norm > 0.0 and right_axis_norm > 0.0
        if finite_:
            left_axis[slot] = left_axis_ / left_axis_norm
            right_axis[slot] = right_axis_ / right_axis_norm
            normal[slot] = normal_
            tangent[slot] = _objective_tangent_basis(
                normal_, left_axis[slot], right_axis[slot]
            )
            left_surface[slot] = left_center[slot] - radii[first] * normal_
            right_surface[slot] = right_center[slot] + radii[second] * normal_
        else:
            witness_failure = True
        finite[slot] = finite_
    pack_success = failure == RodContactSearchFailure.NONE and overflow == 0
    if pack_success:
        active[: len(ordered_records)] = finite[: len(ordered_records)]
    if witness_failure and failure == RodContactSearchFailure.NONE:
        failure = RodContactSearchFailure.WITNESS_FAILURE
        active[:] = False
    batch = _stencil_batch(
        prepared,
        layout,
        left_features,
        right_features,
        np.zeros((capacity,), dtype=np.int64),
        separation,
        active,
        actual,
        overflow,
        positions.dtype,
    )
    witnesses = RodContactWitnessBatch(
        indices,
        left_ids,
        right_ids,
        route_keys,
        stencil_kinds,
        left_segments,
        right_segments,
        parameters[:, 0],
        parameters[:, 1],
        coefficients,
        left_center,
        right_center,
        left_surface,
        right_surface,
        left_axis,
        right_axis,
        normal,
        tangent,
        centerline_distance,
        physical_gap,
        activation_gap,
        left_radius,
        right_radius,
        active,
        capacity=capacity,
        finite=finite,
    )
    return batch, witnesses, failure


def _pack_plane_witnesses(
    prepared: PreparedRodContactSearch,
    positions: np.ndarray,
    records: list[tuple[int, int]],
    failure: RodContactSearchFailure,
    /,
) -> tuple[
    ContactStencilBatch,
    RodContactWitnessBatch | None,
    RodContactSearchFailure,
]:
    """Pack capsule-plane witnesses at the lower edge endpoint.

    The rod profile keeps one witness per capsule-plane pair; its witness slot
    is the selected endpoint, so an endpoint switch is a new route rather than
    an invented continuation.
    """

    capacity = prepared.plan.plane_capacity
    layout = prepared.layouts.edge_plane
    if capacity == 0:
        return empty_contact_stencil_batch(layout, dtype=positions.dtype), None, failure
    actual = len(records)
    overflow = max(actual - capacity, 0)
    edges = np.asarray(prepared.edges, dtype=np.int32)
    feature_ids = np.asarray(prepared.edge_feature_ids, dtype=np.int64)
    edge_ranks = np.asarray(prepared.edge_ranks, dtype=np.int64)
    edge_features = np.asarray(prepared.edge_scene_features, dtype=np.int64)
    plane_ranks = np.asarray(prepared.plane_ranks, dtype=np.int64)
    plane_features = np.asarray(prepared.plane_scene_features, dtype=np.int64)
    segment_indices = np.asarray(prepared.surface_segment_indices, dtype=np.int32)
    radii = np.asarray(prepared.edge_physical_radius, dtype=positions.dtype)
    extents = np.asarray(prepared.edge_contact_extent, dtype=positions.dtype)
    plane_ids = np.asarray(
        [
            int(
                np.asarray(
                    plane.feature_policy.feature_ids[plane.feature_policy.analytic_slice]
                )[0]
            )
            for plane in prepared.planes
        ],
        dtype=np.int64,
    )

    def endpoint_distance(edge_index: int, plane_index: int) -> np.ndarray:
        plane = prepared.planes[plane_index]
        plane_normal = np.asarray(plane.unit_normal, dtype=positions.dtype)
        return positions[edges[edge_index]] @ plane_normal - float(plane.offset)

    if records:
        axial_slots = np.asarray(
            [
                int(endpoint_distance(edge, plane)[1] < endpoint_distance(edge, plane)[0])
                for edge, plane in records
            ],
            dtype=np.int64,
        )
        keys = np.asarray(
            layout.schema.encode(
                int(ContactStencilKind.EDGE_PLANE),
                np.asarray([edge_ranks[edge] for edge, _ in records]),
                np.asarray([plane_ranks[plane] for _, plane in records]),
                axial_slots,
            )
        ).astype(np.int64)
        ordered = [
            (records[index][0], records[index][1], int(axial_slots[index]), int(keys[index]))
            for index in _ordered_rows(keys, capacity)
        ]
    else:
        ordered = []
    indices = np.full((capacity, 4), -1, dtype=np.int32)
    left_features = np.full((capacity,), -1, dtype=np.int64)
    right_features = np.full((capacity,), -1, dtype=np.int64)
    slots = np.zeros((capacity,), dtype=np.int64)
    left_ids = np.zeros((capacity,), dtype=np.int64)
    right_ids = np.zeros((capacity,), dtype=np.int64)
    route_keys = np.zeros((capacity,), dtype=np.int64)
    kinds = np.full((capacity,), int(ContactStencilKind.EDGE_PLANE), dtype=np.int32)
    left_segments = np.full((capacity,), -1, dtype=np.int32)
    right_segments = np.full((capacity,), -1, dtype=np.int32)
    parameters = np.zeros((capacity, 2), dtype=positions.dtype)
    coefficients = np.zeros((capacity, 4), dtype=positions.dtype)
    left_center = np.zeros((capacity, 3), dtype=positions.dtype)
    right_center = np.zeros((capacity, 3), dtype=positions.dtype)
    left_surface = np.zeros((capacity, 3), dtype=positions.dtype)
    right_surface = np.zeros((capacity, 3), dtype=positions.dtype)
    left_axis = np.zeros((capacity, 3), dtype=positions.dtype)
    right_axis = np.zeros((capacity, 3), dtype=positions.dtype)
    normal = np.zeros((capacity, 3), dtype=positions.dtype)
    tangent = np.zeros((capacity, 3, 2), dtype=positions.dtype)
    centerline_distance = np.zeros((capacity,), dtype=positions.dtype)
    physical_gap = np.zeros((capacity,), dtype=positions.dtype)
    activation_gap = np.zeros((capacity,), dtype=positions.dtype)
    left_radius = np.zeros((capacity,), dtype=positions.dtype)
    right_radius = np.zeros((capacity,), dtype=positions.dtype)
    separation = np.zeros((capacity,), dtype=positions.dtype)
    finite = np.zeros((capacity,), dtype=np.bool_)
    active = np.zeros((capacity,), dtype=np.bool_)
    witness_failure = False
    for slot, (edge_index, plane_index, axial, key) in enumerate(ordered):
        edge = edges[edge_index]
        plane = prepared.planes[plane_index]
        plane_normal = np.asarray(plane.unit_normal, dtype=positions.dtype)
        distances = endpoint_distance(edge_index, plane_index)
        center = positions[edge[axial]]
        signed = float(distances[axial])
        radius = float(radii[edge_index])
        axis = positions[edge[1]] - positions[edge[0]]
        axis_norm = float(np.linalg.norm(axis))
        finite_ = bool(
            np.all(np.isfinite(distances)) and np.all(np.isfinite(axis)) and axis_norm > 0.0
        )
        indices[slot] = (edge[0], edge[1], -1, -1)
        left_features[slot] = edge_features[edge_index]
        right_features[slot] = plane_features[plane_index]
        slots[slot] = axial
        left_ids[slot] = feature_ids[edge_index]
        right_ids[slot] = plane_ids[plane_index]
        route_keys[slot] = key
        left_segments[slot] = segment_indices[edge_index]
        parameters[slot, 0] = float(axial)
        coefficients[slot, :2] = (1.0 - float(axial), float(axial))
        left_center[slot] = center
        right_center[slot] = center - signed * plane_normal
        left_surface[slot] = center - radius * plane_normal
        right_surface[slot] = right_center[slot]
        left_radius[slot] = radius
        centerline_distance[slot] = signed
        physical_gap[slot] = signed - radius
        plane_extent = float(
            np.asarray(plane.feature_policy.contact_extent)[
                plane.feature_policy.analytic_slice
            ][0]
        )
        separation[slot] = extents[edge_index] + plane_extent
        activation_gap[slot] = signed - separation[slot]
        if finite_:
            left_axis[slot] = axis / axis_norm
            normal[slot] = plane_normal
            tangent[slot] = _objective_tangent_basis(
                plane_normal, left_axis[slot], right_axis[slot]
            )
        else:
            witness_failure = True
        finite[slot] = finite_
    pack_success = failure == RodContactSearchFailure.NONE and overflow == 0
    if pack_success:
        active[: len(ordered)] = finite[: len(ordered)]
    if witness_failure and failure == RodContactSearchFailure.NONE:
        failure = RodContactSearchFailure.WITNESS_FAILURE
        active[:] = False
    batch = _stencil_batch(
        prepared,
        layout,
        left_features,
        right_features,
        slots,
        separation,
        active,
        actual,
        overflow,
        positions.dtype,
    )
    witnesses = RodContactWitnessBatch(
        indices,
        left_ids,
        right_ids,
        route_keys,
        kinds,
        left_segments,
        right_segments,
        parameters[:, 0],
        parameters[:, 1],
        coefficients,
        left_center,
        right_center,
        left_surface,
        right_surface,
        left_axis,
        right_axis,
        normal,
        tangent,
        centerline_distance,
        physical_gap,
        activation_gap,
        left_radius,
        right_radius,
        active,
        capacity=capacity,
        finite=finite,
    )
    return batch, witnesses, failure


def _invalidate_batch(batch: ContactStencilBatch, /) -> ContactStencilBatch:
    """Keep discovered diagnostic rows but mark every row invalid."""

    return eqx.tree_at(lambda value: value.valid, batch, jnp.zeros_like(batch.valid))


def _merge_witnesses(
    pair: RodContactWitnessBatch,
    plane: RodContactWitnessBatch | None,
    /,
    *,
    complete: bool,
) -> RodContactWitnessBatch:
    if plane is None:
        if complete:
            return pair
        return RodContactWitnessBatch(
            pair.vertex_indices,
            pair.left_feature_ids,
            pair.right_feature_ids,
            pair.route_keys,
            pair.stencil_kinds,
            pair.left_segment_indices,
            pair.right_segment_indices,
            pair.left_parameters,
            pair.right_parameters,
            pair.coefficients,
            pair.left_centerline_witness,
            pair.right_centerline_witness,
            pair.left_surface_witness,
            pair.right_surface_witness,
            pair.left_axis,
            pair.right_axis,
            pair.normal,
            pair.tangent_basis,
            pair.centerline_distance,
            pair.physical_gap,
            pair.activation_gap,
            pair.left_radius,
            pair.right_radius,
            np.zeros((pair.capacity,), dtype=np.bool_),
            capacity=pair.capacity,
            finite=pair.finite,
            batch_id=pair.batch_id,
        )
    capacity = pair.capacity + plane.capacity
    present_pair = np.asarray(pair.finite)
    present_plane = np.asarray(plane.finite)
    keys = np.concatenate(
        (
            np.asarray(pair.route_keys)[present_pair],
            np.asarray(plane.route_keys)[present_plane],
        )
    )
    order = np.argsort(keys, kind="stable")

    def combined(first: Array, second: Array, fill_shape: tuple[int, ...]) -> np.ndarray:
        first_values = np.asarray(first)[present_pair]
        second_values = np.asarray(second)[present_plane]
        values = np.concatenate((first_values, second_values), axis=0)[order]
        output = np.zeros((capacity,) + fill_shape, dtype=values.dtype)
        output[: values.shape[0]] = values
        return output

    indices = combined(pair.vertex_indices, plane.vertex_indices, (4,))
    left_ids = combined(pair.left_feature_ids, plane.left_feature_ids, ())
    right_ids = combined(pair.right_feature_ids, plane.right_feature_ids, ())
    route_keys = combined(pair.route_keys, plane.route_keys, ())
    kinds = combined(pair.stencil_kinds, plane.stencil_kinds, ())
    left_segments = combined(pair.left_segment_indices, plane.left_segment_indices, ())
    right_segments = combined(pair.right_segment_indices, plane.right_segment_indices, ())
    left_parameters = combined(pair.left_parameters, plane.left_parameters, ())
    right_parameters = combined(pair.right_parameters, plane.right_parameters, ())
    coefficients = combined(pair.coefficients, plane.coefficients, (4,))
    left_center = combined(
        pair.left_centerline_witness, plane.left_centerline_witness, (3,)
    )
    right_center = combined(
        pair.right_centerline_witness, plane.right_centerline_witness, (3,)
    )
    left_surface = combined(pair.left_surface_witness, plane.left_surface_witness, (3,))
    right_surface = combined(
        pair.right_surface_witness, plane.right_surface_witness, (3,)
    )
    left_axis = combined(pair.left_axis, plane.left_axis, (3,))
    right_axis = combined(pair.right_axis, plane.right_axis, (3,))
    normal = combined(pair.normal, plane.normal, (3,))
    tangent = combined(pair.tangent_basis, plane.tangent_basis, (3, 2))
    distance = combined(pair.centerline_distance, plane.centerline_distance, ())
    physical_gap = combined(pair.physical_gap, plane.physical_gap, ())
    activation_gap = combined(pair.activation_gap, plane.activation_gap, ())
    left_radius = combined(pair.left_radius, plane.left_radius, ())
    right_radius = combined(pair.right_radius, plane.right_radius, ())
    finite = np.zeros((capacity,), dtype=np.bool_)
    finite[: keys.size] = True
    valid = finite & complete
    return RodContactWitnessBatch(
        indices,
        left_ids,
        right_ids,
        route_keys,
        kinds,
        left_segments,
        right_segments,
        left_parameters,
        right_parameters,
        coefficients,
        left_center,
        right_center,
        left_surface,
        right_surface,
        left_axis,
        right_axis,
        normal,
        tangent,
        distance,
        physical_gap,
        activation_gap,
        left_radius,
        right_radius,
        valid,
        capacity=capacity,
        finite=finite,
    )


def _search_rod_capsules(
    prepared: PreparedRodContactSearch,
    positions: ArrayLike,
    end_positions: ArrayLike | None,
    /,
) -> RodContactSearchResult:
    start = np.asarray(positions)
    if not np.issubdtype(start.dtype, np.floating):
        start = start.astype(np.float64)
    end = start if end_positions is None else np.asarray(end_positions, dtype=start.dtype)
    swept = end_positions is not None
    expected = (prepared.surface_plan.vertex_count, 3)
    if start.shape != expected or end.shape != expected:
        raise ValueError(f"Rod contact positions must have shape {expected}.")
    if np.any(~np.isfinite(start)) or np.any(~np.isfinite(end)):
        raise ValueError("Rod contact positions must be finite.")
    edges = np.asarray(prepared.edges, dtype=np.int32)
    feature_ids = np.asarray(prepared.edge_feature_ids, dtype=np.int64)
    participants = np.asarray(prepared.edge_participant_ids, dtype=np.int64)
    static = np.asarray(prepared.edge_static_mask, dtype=np.bool_)
    extents = np.asarray(prepared.edge_contact_extent, dtype=start.dtype)
    start_segments = start[edges]
    end_segments = end[edges]
    expansion = extents + 0.5 * prepared.plan.activation_distance
    lower = (
        np.minimum(start_segments.min(axis=1), end_segments.min(axis=1))
        - expansion[:, None]
    )
    upper = (
        np.maximum(start_segments.max(axis=1), end_segments.max(axis=1))
        + expansion[:, None]
    )
    failure = RodContactSearchFailure.NONE
    if prepared.plan.route == RodContactSearchRoute.DENSE:
        broad_pairs = [
            (first, second)
            for first in range(edges.shape[0])
            for second in range(first + 1, edges.shape[0])
        ]
        traversal_visits = 0
        tree_depth = 0
    else:
        broad_pairs, traversal_visits, tree_depth, failure = _lbvh_candidate_pairs(
            lower,
            upper,
            feature_ids,
            prepared.plan.maximum_tree_depth,
            prepared.plan.maximum_traversal_visits,
        )
    excluded = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in np.asarray(prepared.excluded_vertex_pairs).tolist()
    }
    allowed_pairs = np.asarray(prepared.allowed_participant_pairs, dtype=np.int64)
    pair_records: list[tuple[int, int]] = []
    adjacency_filtered = 0
    policy_filtered = 0
    aabb_tests = 0
    narrow_tests = 0
    if failure == RodContactSearchFailure.NONE:
        for first, second in broad_pairs:
            permitted, adjacency = _edge_pair_allowed(
                first,
                second,
                edges,
                participants,
                static,
                excluded,
                allowed_pairs,
                prepared.pair_policy_unrestricted,
            )
            if not permitted:
                adjacency_filtered += int(adjacency)
                policy_filtered += int(not adjacency)
                continue
            aabb_tests += 1
            if not _bounds_overlap(
                lower[first], upper[first], lower[second], upper[second]
            ):
                continue
            narrow_tests += 1
            if swept:
                pair_records.append((first, second))
                continue
            first_edge = edges[first]
            second_edge = edges[second]
            evaluation = edge_edge_distance(
                start[first_edge[0]],
                start[first_edge[1]],
                start[second_edge[0]],
                start[second_edge[1]],
            )
            distance = float(
                np.sqrt(max(float(np.asarray(evaluation.squared_distance)), 0.0))
            )
            threshold = (
                float(extents[first])
                + float(extents[second])
                + prepared.plan.activation_distance
            )
            if distance <= threshold:
                pair_records.append((first, second))
    plane_records: list[tuple[int, int]] = []
    for plane_index, plane in enumerate(prepared.planes):
        normal = np.asarray(plane.unit_normal, dtype=start.dtype)
        plane_feature = plane.feature_policy.analytic_slice
        plane_participant = int(
            np.asarray(plane.feature_policy.participant_ids)[plane_feature][0]
        )
        plane_static = bool(
            np.asarray(plane.feature_policy.static_mask)[plane_feature][0]
        )
        plane_extent = float(
            np.asarray(plane.feature_policy.contact_extent)[plane_feature][0]
        )
        for edge_index, edge in enumerate(edges):
            permitted = _pair_explicitly_allowed(
                int(participants[edge_index]),
                plane_participant,
                allowed_pairs,
                prepared.pair_policy_unrestricted,
            )
            if not permitted or (bool(static[edge_index]) and plane_static):
                policy_filtered += 1
                continue
            box_lower = np.minimum(start[edge], end[edge])
            box_upper = np.maximum(start[edge], end[edge])
            signed = float(
                np.min(
                    np.sum(np.minimum(normal * box_lower, normal * box_upper), axis=-1)
                )
                - plane.offset
            )
            threshold = (
                float(extents[edge_index])
                + plane_extent
                + prepared.plan.activation_distance
            )
            narrow_tests += 1
            if signed <= threshold:
                plane_records.append((edge_index, plane_index))
    edge_edge, pair_witnesses, pair_failure = _pack_witnesses(
        prepared, start, pair_records, failure
    )
    edge_plane, plane_witnesses, plane_failure = _pack_plane_witnesses(
        prepared, start, plane_records, failure
    )
    failure = max(pair_failure, plane_failure, key=int)
    pair_overflow = max(len(pair_records) - prepared.plan.capacity, 0)
    plane_overflow = max(len(plane_records) - prepared.plan.plane_capacity, 0)
    overflow = pair_overflow + plane_overflow
    if overflow and failure == RodContactSearchFailure.NONE:
        failure = RodContactSearchFailure.CAPACITY_OVERFLOW
    contact_extents, plane_normals, plane_offsets, geometry_valid = (
        scene_route_geometry(prepared.scene, jnp.dtype(start.dtype))
    )
    if not bool(geometry_valid) and failure == RodContactSearchFailure.NONE:
        failure = RodContactSearchFailure.INVALID_ROUTE_GEOMETRY
    complete = failure == RodContactSearchFailure.NONE
    if not complete:
        edge_edge = _invalidate_batch(edge_edge)
        edge_plane = _invalidate_batch(edge_plane)
    witnesses = _merge_witnesses(pair_witnesses, plane_witnesses, complete=complete)
    if complete:
        status = ContactSearchStatus.SUCCESS
    elif failure == RodContactSearchFailure.CAPACITY_OVERFLOW:
        status = ContactSearchStatus.CANDIDATE_OVERFLOW
    elif failure in (
        RodContactSearchFailure.TREE_DEPTH_OVERFLOW,
        RodContactSearchFailure.TRAVERSAL_OVERFLOW,
    ):
        status = ContactSearchStatus.TRAVERSAL_EXHAUSTED
    else:
        status = ContactSearchStatus.INVALID_INPUT
    candidate_count = len(pair_records) + len(plane_records)
    layouts = prepared.layouts
    trajectory_id = _ROD_VERTEX_LINEAR_PATH if swept else ""
    dtype = start.dtype

    def empty(layout: ContactStencilLayout) -> ContactStencilBatch:
        return empty_contact_stencil_batch(layout, dtype=dtype)

    epoch = ContactCandidateEpoch(
        vertex_vertex=empty(layouts.vertex_vertex),
        edge_vertex=empty(layouts.edge_vertex),
        edge_edge=edge_edge,
        face_vertex=empty(layouts.face_vertex),
        vertex_plane=empty(layouts.vertex_plane),
        edge_plane=edge_plane,
        reference_positions=jnp.asarray(start),
        coverage_lower=jnp.asarray(np.minimum(start, end)),
        coverage_upper=jnp.asarray(np.maximum(start, end)),
        envelope_radius=jnp.asarray(
            0.5 * prepared.plan.activation_distance, dtype=dtype
        ),
        lower_time=jnp.asarray(0.0, dtype=dtype),
        upper_time=jnp.asarray(1.0, dtype=dtype),
        revision=jnp.asarray(0, dtype=jnp.int64),
        contact_extents=contact_extents,
        plane_normals=plane_normals,
        plane_offsets=plane_offsets,
        candidate_count=jnp.asarray(candidate_count, dtype=jnp.int32),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        complete=jnp.asarray(complete),
        search_id=prepared.plan.plan_id,
        catalogue_id=prepared.catalogue_id,
        topology_id=prepared.scene.scene_id,
        routing_id=prepared.scene.routing_id,
        trajectory_id=trajectory_id,
        epoch_id=contact_epoch_identity(prepared.plan.plan_id, prepared.scene, layouts),
    )
    finite = jnp.all(jnp.isfinite(jnp.asarray(start))) & jnp.all(
        jnp.isfinite(jnp.asarray(end))
    )
    evidence = RodContactSearchEvidence(
        jnp.asarray(candidate_count, dtype=jnp.int32),
        jnp.asarray(candidate_count, dtype=jnp.int32),
        jnp.asarray(overflow, dtype=jnp.int32),
        jnp.asarray(adjacency_filtered, dtype=jnp.int32),
        jnp.asarray(policy_filtered, dtype=jnp.int32),
        jnp.asarray(aabb_tests, dtype=jnp.int32),
        jnp.asarray(narrow_tests, dtype=jnp.int32),
        jnp.asarray(traversal_visits, dtype=jnp.int32),
        jnp.asarray(tree_depth, dtype=jnp.int32),
        finite,
        jnp.asarray(complete),
        finite & jnp.asarray(complete),
        jnp.asarray(int(failure), dtype=jnp.int32),
        prepared.plan.route,
        prepared.prepared_id,
    )
    return RodContactSearchResult(
        epoch, witnesses, evidence, pair_witnesses, plane_witnesses
    )


class RodContactManifoldTransition(StrictModule):
    """Generic manifold transition plus the rod witnesses that produced it."""

    manifold: ContactManifoldTransition
    witnesses: RodContactWitnessBatch


def empty_rod_contact_manifold(
    capacity: int,
    /,
    *,
    catalogue_id: str,
    dtype: DTypeLike = jnp.float64,
) -> ContactManifoldState:
    """Empty mechanical history with left/right capsule surface witnesses."""

    return ContactManifoldState.empty(
        capacity,
        catalogue_id=catalogue_id,
        ambient_dimension=3,
        witness_count=2,
        dtype=dtype,
    )


def rod_contact_manifold_observation(
    witnesses: RodContactWitnessBatch,
    /,
    *,
    material_revision: ArrayLike,
    geometry_revision: ArrayLike,
    catalogue_id: str,
) -> ContactManifoldObservation:
    """Observe rod witnesses: keys, material frames and surface witness points."""

    if not isinstance(witnesses, RodContactWitnessBatch):
        raise TypeError("witnesses must be RodContactWitnessBatch.")
    return ContactManifoldObservation(
        witnesses.route_keys,
        witnesses.valid,
        witnesses.normal,
        witnesses.tangent_basis,
        jnp.stack(
            (witnesses.left_surface_witness, witnesses.right_surface_witness), axis=1
        ),
        material_revision,
        geometry_revision,
        catalogue_id=catalogue_id,
    )


def update_rod_contact_manifold(
    state: ContactManifoldState,
    witnesses: RodContactWitnessBatch,
    /,
    *,
    material_revision: ArrayLike,
    retention_steps: int,
) -> RodContactManifoldTransition:
    """Join rod witnesses into generic history with material-frame transport.

    Rod contact frames are built from body-fixed capsule axes, so retained local
    impulses and slips follow the material carrier.  Rod geometry has no
    separate geometry revision; the state's revision is carried unchanged.
    """

    if not isinstance(state, ContactManifoldState):
        raise TypeError("state must be ContactManifoldState.")
    policy = ContactManifoldPolicy(
        retention_steps=retention_steps,
        frame_transport=ContactFrameTransport.MATERIAL_FRAME,
    )
    observation = rod_contact_manifold_observation(
        witnesses,
        material_revision=material_revision,
        geometry_revision=state.geometry_revision,
        catalogue_id=state.catalogue_id,
    )
    return RodContactManifoldTransition(
        update_contact_manifold(state, observation, policy=policy), witnesses
    )


class RodContactCCDStatus(IntEnum):
    FULL_STEP_SAFE = 0
    IMPACT = 1
    CERTIFIED_SAFE_PREFIX = 2
    SEARCH_FAILED = 3


class RodContactCCDEvidence(StrictModule):
    search_successful: Array
    full_step_safe: Array
    impact_detected: Array
    certified_safe_prefix: Array
    conservative_advancement_iterations: Array
    distance_evaluations: Array
    minimum_gap: Array
    maximum_speed_bound: Array
    finite: Array
    complete: Array
    successful: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class RodContactCCDResult(StrictModule, NonTrainableState):
    search: RodContactSearchResult
    safe_step_fraction: Array
    impact_fraction: Array
    impact_route_key: Array
    evidence: RodContactCCDEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class RodContactCCDPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    distance_tolerance: float = eqx.field(static=True)
    safety_fraction: float = eqx.field(static=True)
    minimum_progress: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 64,
        distance_tolerance: float = 1.0e-9,
        safety_fraction: float = 0.9,
        minimum_progress: float = 1.0e-12,
    ) -> None:
        iterations = int(maximum_iterations)
        tolerance = float(distance_tolerance)
        safety = float(safety_fraction)
        progress = float(minimum_progress)
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("distance_tolerance must be finite and positive.")
        if not isfinite(safety) or not 0.0 < safety < 1.0:
            raise ValueError("safety_fraction must lie in (0, 1).")
        if not isfinite(progress) or progress <= 0.0:
            raise ValueError("minimum_progress must be finite and positive.")
        self.maximum_iterations = iterations
        self.distance_tolerance = tolerance
        self.safety_fraction = safety
        self.minimum_progress = progress
        self.plan_id = canonical_fingerprint(
            {
                "kind": "rod-contact-ccd-plan",
                "maximum_iterations": iterations,
                "distance_tolerance": tolerance.hex(),
                "safety_fraction": safety.hex(),
                "minimum_progress": progress.hex(),
            }
        )

    def evaluate(
        self,
        search: PreparedRodContactSearch,
        start_positions: ArrayLike,
        end_positions: ArrayLike,
        /,
        *,
        supported_initial_plane_route_keys: ArrayLike | None = None,
    ) -> RodContactCCDResult:
        return _rod_contact_ccd(
            self,
            search,
            start_positions,
            end_positions,
            supported_initial_plane_route_keys=supported_initial_plane_route_keys,
        )


def _pair_gap_at_fraction(
    start: np.ndarray,
    displacement: np.ndarray,
    endpoints: np.ndarray,
    left_radius: float,
    right_radius: float,
    fraction: float,
    /,
) -> float:
    positions = start + fraction * displacement
    evaluation = edge_edge_distance(
        positions[endpoints[0]],
        positions[endpoints[1]],
        positions[endpoints[2]],
        positions[endpoints[3]],
    )
    return float(np.sqrt(max(float(np.asarray(evaluation.squared_distance)), 0.0))) - (
        left_radius + right_radius
    )


def _plane_gap_at_fraction(
    start: np.ndarray,
    displacement: np.ndarray,
    endpoints: np.ndarray,
    radius: float,
    plane: PlaneContactGeometry,
    fraction: float,
    /,
) -> float:
    positions = start + fraction * displacement
    normal = np.asarray(plane.unit_normal, dtype=start.dtype)
    signed = positions[endpoints[:2]] @ normal - float(plane.offset)
    return float(np.min(signed)) - radius


def _rod_contact_ccd(
    plan: RodContactCCDPlan,
    search: PreparedRodContactSearch,
    start_positions: ArrayLike,
    end_positions: ArrayLike,
    /,
    *,
    supported_initial_plane_route_keys: ArrayLike | None = None,
) -> RodContactCCDResult:
    if not isinstance(search, PreparedRodContactSearch):
        raise TypeError("search must be PreparedRodContactSearch.")
    start = np.asarray(start_positions)
    if not np.issubdtype(start.dtype, np.floating):
        start = start.astype(np.float64)
    end = np.asarray(end_positions, dtype=start.dtype)
    expected = (search.surface_plan.vertex_count, 3)
    if start.shape != expected or end.shape != expected:
        raise ValueError(f"Rod CCD positions must have shape {expected}.")
    if np.any(~np.isfinite(start)) or np.any(~np.isfinite(end)):
        raise ValueError("Rod CCD trajectories must be finite.")
    broad = search.search(start, end_positions=end)
    if not bool(np.asarray(broad.successful)):
        evidence = RodContactCCDEvidence(
            broad.successful,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=jnp.asarray(start).dtype),
            jnp.asarray(0.0, dtype=jnp.asarray(start).dtype),
            jnp.asarray(True),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(int(RodContactCCDStatus.SEARCH_FAILED), dtype=jnp.int32),
            plan.plan_id,
        )
        return RodContactCCDResult(
            broad,
            jnp.asarray(0.0, dtype=jnp.asarray(start).dtype),
            jnp.asarray(jnp.nan, dtype=jnp.asarray(start).dtype),
            jnp.asarray(-1, dtype=jnp.int64),
            evidence,
        )
    displacement = end - start
    active = np.asarray(broad.witnesses.valid)
    indices = np.asarray(broad.witnesses.vertex_indices)
    route_keys = np.asarray(broad.witnesses.route_keys)
    left_radii = np.asarray(broad.witnesses.left_radius)
    right_radii = np.asarray(broad.witnesses.right_radius)
    stencil_kinds = np.asarray(broad.witnesses.stencil_kinds)
    right_feature_ids = np.asarray(broad.witnesses.right_feature_ids)
    if supported_initial_plane_route_keys is None:
        supported_plane_route_keys: set[int] = set()
    else:
        supported_values = np.asarray(supported_initial_plane_route_keys)
        if supported_values.ndim != 1 or not np.issubdtype(
            supported_values.dtype, np.integer
        ):
            raise TypeError(
                "supported_initial_plane_route_keys must be a rank-one integer array."
            )
        supported_plane_route_keys = set(
            int(value) for value in supported_values.tolist()
        )
    planes_by_feature = {
        int(
            np.asarray(
                plane.feature_policy.feature_ids[plane.feature_policy.analytic_slice]
            )[0]
        ): plane
        for plane in search.planes
    }
    global_safe = 1.0
    earliest_impact = np.inf
    impact_key = -1
    any_prefix = False
    total_iterations = 0
    evaluations = 0
    minimum_gap = np.inf
    maximum_speed = 0.0
    for slot in np.flatnonzero(active).tolist():
        endpoints = indices[slot]
        if stencil_kinds[slot] == int(ContactStencilKind.EDGE_PLANE):
            plane = planes_by_feature[int(right_feature_ids[slot])]
            speed = max(
                float(np.linalg.norm(displacement[endpoints[0]])),
                float(np.linalg.norm(displacement[endpoints[1]])),
            )
            gap_action = lambda fraction: _plane_gap_at_fraction(
                start,
                displacement,
                endpoints,
                float(left_radii[slot]),
                plane,
                fraction,
            )
        else:
            speed = max(
                float(np.linalg.norm(displacement[endpoints[0]])),
                float(np.linalg.norm(displacement[endpoints[1]])),
            ) + max(
                float(np.linalg.norm(displacement[endpoints[2]])),
                float(np.linalg.norm(displacement[endpoints[3]])),
            )
            gap_action = lambda fraction: _pair_gap_at_fraction(
                start,
                displacement,
                endpoints,
                float(left_radii[slot]),
                float(right_radii[slot]),
                fraction,
            )
        maximum_speed = max(maximum_speed, speed)
        time = 0.0
        safe_time = 0.0
        gap = gap_action(time)
        evaluations += 1
        minimum_gap = min(minimum_gap, gap)
        pair_impact = False
        supported_initial_plane = (
            int(route_keys[slot]) in supported_plane_route_keys
            and stencil_kinds[slot] == int(ContactStencilKind.EDGE_PLANE)
            and gap <= plan.distance_tolerance
        )
        if supported_initial_plane:
            end_gap = gap_action(1.0)
            evaluations += 1
            minimum_gap = min(minimum_gap, end_gap)
            if end_gap >= -plan.distance_tolerance:
                safe_time = 1.0
            else:
                pair_impact = True
        elif gap <= plan.distance_tolerance:
            pair_impact = True
        elif speed <= np.finfo(start.dtype).eps:
            safe_time = 1.0
        else:
            for _ in range(plan.maximum_iterations):
                total_iterations += 1
                certified_increment = max((gap - plan.distance_tolerance) / speed, 0.0)
                if certified_increment >= 1.0 - time:
                    safe_time = 1.0
                    break
                increment = plan.safety_fraction * certified_increment
                next_time = time + increment
                if next_time <= time:
                    any_prefix = True
                    break
                safe_time = time
                time += increment
                gap = gap_action(time)
                evaluations += 1
                minimum_gap = min(minimum_gap, gap)
                if gap <= plan.distance_tolerance:
                    pair_impact = True
                    break
            else:
                any_prefix = True
        global_safe = min(global_safe, safe_time)
        if pair_impact and time < earliest_impact:
            earliest_impact = time
            impact_key = int(route_keys[slot])
    if not np.any(active):
        minimum_gap = np.inf
    impact_detected = np.isfinite(earliest_impact)
    full_safe = global_safe >= 1.0 and not impact_detected and not any_prefix
    certified_prefix = not full_safe and not impact_detected
    if full_safe:
        status = RodContactCCDStatus.FULL_STEP_SAFE
        safe_fraction = 1.0
    elif impact_detected:
        status = RodContactCCDStatus.IMPACT
        safe_fraction = min(global_safe, earliest_impact)
    else:
        status = RodContactCCDStatus.CERTIFIED_SAFE_PREFIX
        safe_fraction = global_safe
    finite = bool(
        np.isfinite(safe_fraction)
        and np.isfinite(maximum_speed)
        and (np.isfinite(minimum_gap) or not np.any(active))
    )
    complete = full_safe or impact_detected or certified_prefix
    evidence = RodContactCCDEvidence(
        broad.successful,
        jnp.asarray(full_safe),
        jnp.asarray(impact_detected),
        jnp.asarray(certified_prefix),
        jnp.asarray(total_iterations, dtype=jnp.int32),
        jnp.asarray(evaluations, dtype=jnp.int32),
        jnp.asarray(minimum_gap, dtype=jnp.asarray(start).dtype),
        jnp.asarray(maximum_speed, dtype=jnp.asarray(start).dtype),
        jnp.asarray(finite),
        jnp.asarray(complete),
        jnp.asarray(finite and complete),
        jnp.asarray(int(status), dtype=jnp.int32),
        plan.plan_id,
    )
    return RodContactCCDResult(
        broad,
        jnp.asarray(safe_fraction, dtype=jnp.asarray(start).dtype),
        jnp.asarray(
            earliest_impact if impact_detected else np.nan,
            dtype=jnp.asarray(start).dtype,
        ),
        jnp.asarray(impact_key, dtype=jnp.int64),
        evidence,
    )


class CompositeContactParticipantBlock(StrictModule, NonTrainableState):
    velocity_operator: AbstractLinearOperator
    inverse_mass_operator: AbstractLinearOperator
    free_velocity: PyTree[Array]
    block_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        velocity_operator: AbstractLinearOperator,
        inverse_mass_operator: AbstractLinearOperator,
        free_velocity: PyTree[Any],
        /,
        *,
        block_id: str | None = None,
    ) -> None:
        if velocity_operator.batch_shape or inverse_mass_operator.batch_shape:
            raise ValueError("Composite contact operators cannot be operator-batched.")
        tangent = velocity_operator.source
        effort = DualSpace(tangent)
        if not inverse_mass_operator.source.compatible(effort) or not (
            inverse_mass_operator.target.compatible(tangent)
        ):
            raise ValueError(
                "inverse_mass_operator must map the participant tangent dual to tangent."
            )
        free = tangent.validate(free_velocity)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "composite-contact-participant-block",
                    "velocity": velocity_operator.operator_id,
                    "inverse_mass": inverse_mass_operator.operator_id,
                }
            )
            if block_id is None
            else str(block_id)
        )
        if not identifier:
            raise ValueError("block_id must be nonempty or None.")
        self.velocity_operator = velocity_operator
        self.inverse_mass_operator = inverse_mass_operator
        self.free_velocity = free
        self.block_id = identifier


def build_rod_contact_velocity_operator(
    participant: AbstractContactParticipant,
    configuration: PyTree[Any],
    witnesses: RodContactWitnessBatch,
    /,
    *,
    vertex_offset: int = 0,
) -> AbstractLinearOperator:
    """Build exact local ``G`` with algebraic-transpose effort pullback."""

    if not isinstance(participant, AbstractContactParticipant):
        raise TypeError("participant must be AbstractContactParticipant.")
    if not isinstance(witnesses, RodContactWitnessBatch):
        raise TypeError("witnesses must be RodContactWitnessBatch.")
    offset = int(vertex_offset)
    if offset < 0:
        raise ValueError("vertex_offset must be nonnegative.")
    configuration_ = participant.source_space.validate(configuration)
    tangent = participant.tangent_space
    dtype = participant.positions(configuration_).dtype
    contact_space = ArraySpace((witnesses.capacity, 3), dtype=dtype)
    local_indices = witnesses.vertex_indices - offset
    inside = (local_indices >= 0) & (
        local_indices < participant.surface_plan.vertex_count
    )
    safe_indices = jnp.clip(local_indices, 0, participant.surface_plan.vertex_count - 1)

    def action(rate: PyTree[Array]) -> Array:
        surface_velocity = participant.velocities(configuration_, rate)
        gathered = jnp.where(inside[..., None], surface_velocity[safe_indices], 0.0)
        relative = jnp.sum(witnesses.coefficients[..., None] * gathered, axis=1)
        normal_velocity = jnp.sum(relative * witnesses.normal, axis=-1, keepdims=True)
        tangent_velocity = jnp.sum(
            witnesses.tangent_basis * relative[..., :, None], axis=-2
        )
        local = jnp.concatenate((normal_velocity, tangent_velocity), axis=-1)
        return jnp.where(witnesses.valid[:, None], local, 0.0)

    def transpose_action(local_impulse: Array) -> PyTree[Array]:
        local = jnp.where(witnesses.valid[:, None], local_impulse, 0.0)
        world = witnesses.normal * local[:, :1] + jnp.sum(
            witnesses.tangent_basis * local[:, None, 1:], axis=-1
        )
        endpoint_effort = witnesses.coefficients[..., None] * world[:, None, :]
        surface_effort = jnp.zeros(
            (
                participant.surface_plan.vertex_count,
                participant.surface_plan.ambient_dimension,
            ),
            dtype=world.dtype,
        )
        surface_effort = surface_effort.at[safe_indices].add(
            jnp.where(inside[..., None], endpoint_effort, 0.0)
        )
        return participant.effort_pullback(configuration_, surface_effort)

    return FunctionLinearOperator(
        action,
        source=tangent,
        target=contact_space,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "rod-contact-velocity-operator",
                "participant": participant.participant_id,
                "witnesses": witnesses.batch_id,
                "vertex_offset": offset,
            }
        ),
    )


def build_capsule_contact_velocity_operator(
    participant: ReducedRodCapsuleContactParticipant,
    configuration: PyTree[Any],
    witnesses: RodContactWitnessBatch,
    /,
) -> AbstractLinearOperator:
    """Build exact capsule-surface ``G`` including material-spin offsets."""

    if not isinstance(participant, ReducedRodCapsuleContactParticipant):
        raise TypeError("participant must be ReducedRodCapsuleContactParticipant.")
    if not isinstance(witnesses, RodContactWitnessBatch):
        raise TypeError("witnesses must be RodContactWitnessBatch.")
    configuration_ = participant.source_space.validate(configuration)
    tangent = participant.tangent_space
    dtype = participant.positions(configuration_).dtype
    contact_space = ArraySpace((witnesses.capacity, 3), dtype=dtype)
    maximum_segment = participant.geometry.rod.plan.segment_count - 1
    left_segment = jnp.clip(witnesses.left_segment_indices, 0, maximum_segment)
    right_segment = jnp.clip(witnesses.right_segment_indices, 0, maximum_segment)
    has_right = witnesses.right_segment_indices >= 0
    left_offset = witnesses.left_surface_witness - witnesses.left_centerline_witness
    right_offset = witnesses.right_surface_witness - witnesses.right_centerline_witness
    left_radius = participant.geometry.segment_radii[left_segment].astype(dtype)
    right_radius = participant.geometry.segment_radii[right_segment].astype(dtype)
    centerline = participant.positions(configuration_)
    segment_nodes = participant.geometry.rod.plan.segment_node_ids

    def radial_fallback(indices: Array, radii: Array) -> Array:
        nodes = segment_nodes[indices]
        axes = centerline[nodes[:, 1]] - centerline[nodes[:, 0]]
        axes = axes / jnp.linalg.norm(axes, axis=-1, keepdims=True)
        first_reference = jnp.broadcast_to(
            jnp.asarray((1.0, 0.0, 0.0), dtype=dtype), axes.shape
        )
        second_reference = jnp.broadcast_to(
            jnp.asarray((0.0, 1.0, 0.0), dtype=dtype), axes.shape
        )
        references = jnp.where(
            (jnp.abs(axes[:, :1]) < 0.9),
            first_reference,
            second_reference,
        )
        radial = jnp.cross(axes, references)
        radial = radial / jnp.linalg.norm(radial, axis=-1, keepdims=True)
        return radii[:, None] * radial

    left_fallback = radial_fallback(left_segment, left_radius)
    right_fallback = radial_fallback(right_segment, right_radius)
    left_offset = jnp.where(
        witnesses.valid[:, None],
        left_offset,
        left_fallback,
    )
    right_offset = jnp.where(
        (witnesses.valid & has_right)[:, None],
        right_offset,
        right_fallback,
    )

    def action(rate: PyTree[Array]) -> Array:
        left_velocity = participant.surface_velocity(
            configuration_,
            rate,
            left_segment,
            witnesses.left_parameters,
            left_offset,
        )
        right_velocity = participant.surface_velocity(
            configuration_,
            rate,
            right_segment,
            witnesses.right_parameters,
            right_offset,
        )
        relative = left_velocity - jnp.where(has_right[:, None], right_velocity, 0.0)
        normal_velocity = jnp.sum(relative * witnesses.normal, axis=-1, keepdims=True)
        tangent_velocity = jnp.sum(
            witnesses.tangent_basis * relative[..., :, None], axis=-2
        )
        local = jnp.concatenate((normal_velocity, tangent_velocity), axis=-1)
        return jnp.where(witnesses.valid[:, None], local, 0.0)

    def transpose_action(local_impulse: Array) -> PyTree[Array]:
        local = jnp.where(witnesses.valid[:, None], local_impulse, 0.0)
        world = witnesses.normal * local[:, :1] + jnp.sum(
            witnesses.tangent_basis * local[:, None, 1:], axis=-1
        )
        segment_indices = jnp.concatenate((left_segment, right_segment))
        axial_coordinates = jnp.concatenate(
            (witnesses.left_parameters, witnesses.right_parameters)
        )
        surface_offsets = jnp.concatenate((left_offset, right_offset), axis=0)
        surface_efforts = jnp.concatenate(
            (world, jnp.where(has_right[:, None], -world, 0.0)), axis=0
        )
        return participant.surface_effort_pullback(
            configuration_,
            segment_indices,
            axial_coordinates,
            surface_offsets,
            surface_efforts,
        )

    return FunctionLinearOperator(
        action,
        source=tangent,
        target=contact_space,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "capsule-contact-velocity-operator",
                "participant": participant.participant_id,
                "witnesses": witnesses.batch_id,
            }
        ),
    )


def prepare_composite_contact_block(
    participant: AbstractContactParticipant | ReducedRodCapsuleContactParticipant,
    configuration: PyTree[Any],
    free_velocity: PyTree[Any],
    inverse_mass_operator: AbstractLinearOperator,
    witnesses: RodContactWitnessBatch,
    /,
    *,
    vertex_offset: int = 0,
) -> CompositeContactParticipantBlock:
    velocity = (
        build_capsule_contact_velocity_operator(participant, configuration, witnesses)
        if isinstance(participant, ReducedRodCapsuleContactParticipant)
        else build_rod_contact_velocity_operator(
            participant, configuration, witnesses, vertex_offset=vertex_offset
        )
    )
    return CompositeContactParticipantBlock(
        velocity,
        inverse_mass_operator,
        free_velocity,
        block_id=canonical_fingerprint(
            {
                "kind": "prepared-rod-composite-contact-block",
                "participant": participant.participant_id,
                "witnesses": witnesses.batch_id,
                "inverse_mass": inverse_mass_operator.operator_id,
            }
        ),
    )


class CompositeContactResponseEvidence(StrictModule):
    """Cone evidence plus per-block true-dual application evidence."""

    cone: ContactConeEvidence
    duality_residual: Array
    duality_scale: Array
    duality_valid: Array
    finite: Array
    applied: Array
    fail_closed: Array
    successful: Array
    backend: str = eqx.field(static=True)
    response_id: str = eqx.field(static=True)


class CompositeContactResult(StrictModule):
    """Applied composite impulse; ``post_contact_velocity`` is physical."""

    impulse: Array
    candidate_impulse: Array
    generalized_impulses: tuple[PyTree[Array], ...]
    velocity_updates: tuple[PyTree[Array], ...]
    post_velocities: tuple[PyTree[Array], ...]
    post_contact_velocity: Array
    branch: Array
    slip_velocity: Array
    route_keys: Array
    cone: ContactConeResult
    evidence: CompositeContactResponseEvidence
    response_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class CompositeContactResponse(StrictModule, NonTrainableState):
    """Composite true-dual response ``G M^-1 G*`` lowered to the common cone.

    The rod law is the common fixed-route Signorini--Coulomb program with the
    witness gap rate as normal law bias; certificates are the program's
    dimensional physical certificates.
    """

    blocks: tuple[CompositeContactParticipantBlock, ...]
    witnesses: RodContactWitnessBatch
    program: ContactConeProgram
    solver: ContactConeSolverPlan
    response_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        blocks: Sequence[CompositeContactParticipantBlock],
        witnesses: RodContactWitnessBatch,
        /,
        *,
        certificate: ContactCertificatePolicy,
        dynamic_friction: ArrayLike,
        static_friction: ArrayLike | None = None,
        compliance: ArrayLike = 0.0,
        normal_bias: ArrayLike = 0.0,
        solver: ContactConeSolverPlan | None = None,
    ) -> None:
        values = tuple(blocks)
        if not values or not all(
            isinstance(value, CompositeContactParticipantBlock) for value in values
        ):
            raise TypeError(
                "blocks must contain CompositeContactParticipantBlock values."
            )
        contact_space = contact_cone_space(witnesses.capacity, 3, witnesses.normal.dtype)
        if any(
            not value.velocity_operator.target.compatible(contact_space)
            for value in values
        ):
            raise ValueError(
                "Composite contact blocks must share the witness contact space."
            )
        solver_ = ContactConeSolverPlan() if solver is None else solver
        if not isinstance(solver_, ContactConeSolverPlan):
            raise TypeError("solver must be ContactConeSolverPlan or None.")

        def delassus_action(local_impulse: Array) -> Array:
            total = contact_space.zeros()
            for block in values:
                effort = dual_transpose(block.velocity_operator).mv(local_impulse)
                velocity_update = block.inverse_mass_operator.mv(effort)
                total = total + block.velocity_operator.mv(velocity_update)
            return total

        delassus = FunctionLinearOperator(
            delassus_action,
            source=DualSpace(contact_space),
            target=contact_space,
            transpose_action=delassus_action,
            operator_id=canonical_fingerprint(
                {
                    "kind": "composite-contact-delassus",
                    "blocks": [value.block_id for value in values],
                    "witnesses": witnesses.batch_id,
                }
            ),
        )
        free = contact_space.zeros()
        for block in values:
            free = free + block.velocity_operator.mv(block.free_velocity)
        free = jnp.where(witnesses.valid[:, None], free, 0.0)
        self.blocks = values
        self.witnesses = witnesses
        self.program = _composite_program(
            delassus,
            witnesses,
            free,
            certificate,
            dynamic_friction,
            static_friction,
            compliance,
            normal_bias,
            "matrix-free",
        )
        self.solver = solver_
        self.response_id = witnesses.batch_id

    @property
    def delassus_operator(self) -> AbstractLinearOperator:
        return self.program.response

    def solve(
        self, /, *, initial_impulse: ArrayLike | None = None
    ) -> CompositeContactResult:
        """Solve with matrix-free Delassus actions; failures apply zero impulse."""

        cone = solve_contact_cone(
            self.program, solver=self.solver, initial_impulse=initial_impulse
        )
        return _apply_composite_response(self, self.program, cone, "matrix-free")

    def solve_dense_authority(
        self,
        /,
        *,
        initial_impulse: ArrayLike | None = None,
        materialization_policy: MaterializationPolicy | None = None,
    ) -> CompositeContactResult:
        """Solve the same law over the explicitly materialized bounded authority."""

        policy = (
            MaterializationPolicy()
            if materialization_policy is None
            else materialization_policy
        )
        if not isinstance(policy, MaterializationPolicy):
            raise TypeError(
                "materialization_policy must be MaterializationPolicy or None."
            )
        program = self.program
        dense = dense_contact_response(
            materialize(program.response, policy),
            program.contact_space,
            operator_id=f"{program.response.operator_id}/dense-authority",
        )
        authority = _composite_program(
            dense,
            self.witnesses,
            program.free_velocity,
            program.certificate,
            program.dynamic_friction,
            program.static_friction,
            program.compliance,
            program.normal_bias,
            "dense-authority",
        )
        cone = solve_contact_cone(
            authority, solver=self.solver, initial_impulse=initial_impulse
        )
        return _apply_composite_response(self, authority, cone, "dense-authority")


def _composite_program(
    response: AbstractLinearOperator,
    witnesses: RodContactWitnessBatch,
    free: Array,
    certificate: ContactCertificatePolicy,
    dynamic_friction: ArrayLike,
    static_friction: ArrayLike | None,
    compliance: ArrayLike,
    normal_bias: ArrayLike,
    backend: str,
    /,
) -> ContactConeProgram:
    return ContactConeProgram(
        response,
        free,
        compliance,
        dynamic_friction,
        witnesses.route_keys,
        witnesses.valid,
        ContactBlockLayout.coulomb(
            witnesses.capacity, 3, family_count=certificate.family_count
        ),
        certificate,
        canonical_fingerprint(
            {
                "kind": "rod-composite-contact-program",
                "witnesses": witnesses.batch_id,
                "response": response.operator_id,
                "backend": backend,
            }
        ),
        static_friction=static_friction,
        normal_bias=normal_bias,
    )


def _apply_composite_response(
    response: CompositeContactResponse,
    program: ContactConeProgram,
    cone: ContactConeResult,
    backend: str,
    /,
) -> CompositeContactResult:
    accepted = cone.impulse
    dtype = accepted.dtype
    contact_dual = DualSpace(program.contact_space)
    generalized: list[PyTree[Array]] = []
    updates: list[PyTree[Array]] = []
    residuals: list[Array] = []
    scales: list[Array] = []
    finite = cone.evidence.finite
    for block in response.blocks:
        effort = dual_transpose(block.velocity_operator).mv(accepted)
        update = block.inverse_mass_operator.mv(effort)
        local_update = block.velocity_operator.mv(update)
        contact_power = contact_dual.pair(accepted, local_update)
        generalized_power = DualSpace(block.velocity_operator.source).pair(
            effort, update
        )
        residuals.append(jnp.abs(contact_power - generalized_power))
        scales.append(
            jnp.maximum(
                1.0, jnp.maximum(jnp.abs(contact_power), jnp.abs(generalized_power))
            )
        )
        finite = finite & _tree_finite(effort) & _tree_finite(update)
        generalized.append(effort)
        updates.append(update)
    duality_residual = jnp.max(jnp.stack(tuple(residuals)), initial=0.0)
    duality_scale = jnp.max(jnp.stack(tuple(scales)), initial=1.0)
    duality_tolerance = jnp.finfo(dtype).eps * max(
        64,
        8 * program.contact_space.size,
        *(8 * block.velocity_operator.source.size for block in response.blocks),
    )
    duality_valid = duality_residual <= duality_tolerance * duality_scale
    successful = cone.evidence.successful & duality_valid & finite
    impulse = jnp.where(successful, accepted, 0.0)
    generalized_ = tuple(
        jax.tree.map(lambda value: jnp.where(successful, value, 0.0), effort)
        for effort in generalized
    )
    updates_ = tuple(
        jax.tree.map(lambda value: jnp.where(successful, value, 0.0), update)
        for update in updates
    )
    post_velocities = tuple(
        jax.tree.map(lambda free, delta: free + delta, block.free_velocity, update)
        for block, update in zip(response.blocks, updates_, strict=True)
    )
    post_contact = jnp.where(
        successful, cone.post_relative_velocity, program.free_velocity
    )
    zero_applied = jnp.max(jnp.abs(impulse), initial=0.0) == 0.0
    evidence = CompositeContactResponseEvidence(
        cone.evidence,
        duality_residual,
        duality_scale,
        duality_valid,
        finite,
        successful,
        successful | zero_applied,
        successful,
        backend,
        response.response_id,
    )
    return CompositeContactResult(
        impulse,
        cone.candidate_impulse,
        generalized_,
        updates_,
        post_velocities,
        post_contact,
        jnp.where(successful, cone.branch, int(ContactConeBranch.UNCLASSIFIED)).astype(
            jnp.int8
        ),
        post_contact[:, 1:],
        response.witnesses.route_keys,
        cone,
        evidence,
        response.response_id,
    )


def _tree_finite(value: PyTree[Array], /) -> Array:
    leaves = jax.tree.leaves(value)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


__all__ = [
    "build_capsule_contact_velocity_operator",
    "build_rod_contact_velocity_operator",
    "CompositeContactParticipantBlock",
    "CompositeContactResponse",
    "CompositeContactResponseEvidence",
    "CompositeContactResult",
    "empty_rod_contact_manifold",
    "prepare_composite_contact_block",
    "PreparedRodContactSearch",
    "rod_contact_manifold_observation",
    "RodContactCCDEvidence",
    "RodContactCCDPlan",
    "RodContactCCDResult",
    "RodContactCCDStatus",
    "RodContactManifoldTransition",
    "RodContactSearchEvidence",
    "RodContactSearchFailure",
    "RodContactSearchPlan",
    "RodContactSearchResult",
    "RodContactSearchRoute",
    "RodContactWitnessBatch",
    "update_rod_contact_manifold",
]

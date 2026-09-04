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
from jaxtyping import Array, ArrayLike, PyTree
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._distance import edge_edge_distance
from ...discretization.contact._implicit_geometry import PlaneContactGeometry
from ...discretization.contact._participant import AbstractContactParticipant
from ...discretization.contact._search import (
    ContactCandidateEpoch,
    ContactSearchStatus,
)
from ...discretization.contact._stencils import (
    canonical_contact_route_keys,
    ContactStencilBatch,
    ContactStencilKind,
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
from ._cone import ContactConeSolverPlan, project_signorini_coulomb_product
from ._rod_capsule import (
    PreparedRodCapsuleGeometry,
    ReducedRodCapsuleContactParticipant,
)


class RodContactSearchRoute(IntEnum):
    DENSE = 0
    LBVH = 1


class RodContactSearchFailure(IntEnum):
    NONE = 0
    CAPACITY_OVERFLOW = 1
    TREE_DEPTH_OVERFLOW = 2
    TRAVERSAL_OVERFLOW = 3
    WITNESS_FAILURE = 4


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
    ):
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
        active = np.asarray(valid, dtype=bool)
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
            np.ones((count,), dtype=bool)
            if finite is None
            else np.asarray(finite, dtype=bool)
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
            np.zeros((count,), dtype=bool),
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
    ):
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
    pair_policy_unrestricted: bool = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RodContactSearchPlan,
        geometry: PreparedRodCapsuleGeometry | CollisionSurfacePlan,
        planes: Sequence[PlaneContactGeometry],
        /,
    ):
        if not isinstance(plan, RodContactSearchPlan):
            raise TypeError("plan must be RodContactSearchPlan.")
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
        radius = np.asarray(features.physical_radius[edge_slice], dtype=float)
        extent = np.asarray(features.contact_extent[edge_slice], dtype=float)
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
        self.edge_static_mask = jnp.asarray(features.static_mask[edge_slice], dtype=bool)
        self.edge_physical_radius = jnp.asarray(radius)
        self.edge_contact_extent = jnp.asarray(extent)
        self.excluded_vertex_pairs = jnp.asarray(
            surface_plan.pair_policy.excluded_vertex_pairs, dtype=jnp.int64
        )
        self.allowed_participant_pairs = jnp.asarray(
            surface_plan.pair_policy.allowed_participant_pairs, dtype=jnp.int64
        )
        self.pair_policy_unrestricted = surface_plan.pair_policy.unrestricted
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-rod-contact-search",
                "plan": plan.plan_id,
                "surface": surface_plan.topology_id,
                "features": features.policy_id,
                "planes": [value.geometry_id for value in planes_],
                "segment_indices": array_tree_fingerprint(segment_indices),
            }
        )

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
    ):
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
    first_vertices = tuple(int(value) for value in edges[first])
    second_vertices = tuple(int(value) for value in edges[second])
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


def _pack_witnesses(
    prepared: PreparedRodContactSearch,
    positions: np.ndarray,
    records: list[tuple[int, int]],
    failure: RodContactSearchFailure,
    /,
) -> tuple[ContactStencilBatch, RodContactWitnessBatch, RodContactSearchFailure]:
    capacity = prepared.plan.capacity
    actual = len(records)
    overflow = max(actual - capacity, 0)
    edges = np.asarray(prepared.edges, dtype=np.int32)
    feature_ids = np.asarray(prepared.edge_feature_ids, dtype=np.int64)
    radii = np.asarray(prepared.edge_physical_radius, dtype=positions.dtype)
    extents = np.asarray(prepared.edge_contact_extent, dtype=positions.dtype)
    ordered_records: list[tuple[int, int, int]] = []
    if records:
        first_features = np.asarray([feature_ids[first] for first, _ in records])
        second_features = np.asarray([feature_ids[second] for _, second in records])
        keys = canonical_contact_route_keys(
            ContactStencilKind.EDGE_EDGE, first_features, second_features
        )
        order = np.argsort(keys, kind="stable")
        ordered_records = [
            (records[index][0], records[index][1], int(keys[index]))
            for index in order[:capacity]
        ]
    indices = np.full((capacity, 4), -1, dtype=np.int32)
    left_ids = np.zeros((capacity,), dtype=np.int64)
    right_ids = np.zeros((capacity,), dtype=np.int64)
    route_keys = np.zeros((capacity,), dtype=np.int64)
    stencil_kinds = np.full(
        (capacity,), int(ContactStencilKind.EDGE_EDGE), dtype=np.int32
    )
    left_segments = np.full((capacity,), -1, dtype=np.int32)
    right_segments = np.full((capacity,), -1, dtype=np.int32)
    separation = np.zeros((capacity,), dtype=positions.dtype)
    active = np.zeros((capacity,), dtype=bool)
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
    finite = np.zeros((capacity,), dtype=bool)
    witness_failure = False
    for slot, (first, second, key) in enumerate(ordered_records):
        if feature_ids[first] > feature_ids[second]:
            first, second = second, first
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
        left_ids[slot] = feature_ids[first]
        right_ids[slot] = feature_ids[second]
        route_keys[slot] = key
        left_segments[slot] = int(np.asarray(prepared.surface_segment_indices)[first])
        right_segments[slot] = int(np.asarray(prepared.surface_segment_indices)[second])
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
    batch = ContactStencilBatch(
        ContactStencilKind.EDGE_EDGE,
        indices,
        left_ids,
        right_ids,
        capacity=capacity,
        minimum_separation=separation,
        valid=active,
        actual_count=actual,
        overflow_count=overflow,
        route_keys=route_keys,
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
    capacity = prepared.plan.plane_capacity
    if capacity == 0:
        return (
            ContactStencilBatch.empty(
                ContactStencilKind.EDGE_VERTEX, 0, dtype=positions.dtype
            ),
            None,
            failure,
        )
    actual = len(records)
    overflow = max(actual - capacity, 0)
    edges = np.asarray(prepared.edges, dtype=np.int32)
    feature_ids = np.asarray(prepared.edge_feature_ids, dtype=np.int64)
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
    if records:
        keys = canonical_contact_route_keys(
            ContactStencilKind.EDGE_VERTEX,
            np.asarray([feature_ids[edge] for edge, _ in records]),
            np.asarray([plane_ids[plane] for _, plane in records]),
        )
        order = np.argsort(keys, kind="stable")
        ordered = [
            (records[index][0], records[index][1], int(keys[index]))
            for index in order[:capacity]
        ]
    else:
        ordered = []
    indices = np.full((capacity, 4), -1, dtype=np.int32)
    left_ids = np.zeros((capacity,), dtype=np.int64)
    right_ids = np.zeros((capacity,), dtype=np.int64)
    route_keys = np.zeros((capacity,), dtype=np.int64)
    kinds = np.full((capacity,), int(ContactStencilKind.EDGE_VERTEX), dtype=np.int32)
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
    finite = np.zeros((capacity,), dtype=bool)
    active = np.zeros((capacity,), dtype=bool)
    witness_failure = False
    for slot, (edge_index, plane_index, key) in enumerate(ordered):
        edge = edges[edge_index]
        plane = prepared.planes[plane_index]
        plane_normal = np.asarray(plane.unit_normal, dtype=positions.dtype)
        endpoint_distance = positions[edge] @ plane_normal - float(plane.offset)
        axial = int(endpoint_distance[1] < endpoint_distance[0])
        center = positions[edge[axial]]
        signed = float(endpoint_distance[axial])
        radius = float(radii[edge_index])
        axis = positions[edge[1]] - positions[edge[0]]
        axis_norm = float(np.linalg.norm(axis))
        finite_ = bool(
            np.all(np.isfinite(endpoint_distance))
            and np.all(np.isfinite(axis))
            and axis_norm > 0.0
        )
        indices[slot] = (edge[0], edge[1], edge[0], -1)
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
    batch = ContactStencilBatch(
        ContactStencilKind.EDGE_VERTEX,
        indices,
        left_ids,
        right_ids,
        capacity=capacity,
        minimum_separation=separation,
        valid=active,
        actual_count=actual,
        overflow_count=overflow,
        route_keys=route_keys,
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
    return ContactStencilBatch(
        batch.kind,
        batch.vertex_indices,
        batch.left_feature_ids,
        batch.right_feature_ids,
        capacity=batch.capacity,
        weights=batch.weights,
        minimum_separation=batch.minimum_separation,
        valid=np.zeros((batch.capacity,), dtype=bool),
        actual_count=batch.actual_count,
        overflow_count=batch.overflow_count,
        route_keys=batch.route_keys,
        batch_id=batch.batch_id,
    )


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
            np.zeros((pair.capacity,), dtype=bool),
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
    finite = np.zeros((capacity,), dtype=bool)
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
    static = np.asarray(prepared.edge_static_mask, dtype=bool)
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
            signed = min(
                float(np.min(start[edge] @ normal - plane.offset)),
                float(np.min(end[edge] @ normal - plane.offset)),
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
    edge_vertex, plane_witnesses, plane_failure = _pack_plane_witnesses(
        prepared, start, plane_records, failure
    )
    failure = max(pair_failure, plane_failure, key=int)
    pair_overflow = max(len(pair_records) - prepared.plan.capacity, 0)
    plane_overflow = max(len(plane_records) - prepared.plan.plane_capacity, 0)
    overflow = pair_overflow + plane_overflow
    if overflow and failure == RodContactSearchFailure.NONE:
        failure = RodContactSearchFailure.CAPACITY_OVERFLOW
    complete = failure == RodContactSearchFailure.NONE
    if not complete:
        edge_edge = _invalidate_batch(edge_edge)
        edge_vertex = _invalidate_batch(edge_vertex)
    witnesses = _merge_witnesses(pair_witnesses, plane_witnesses, complete=complete)
    status = (
        ContactSearchStatus.SUCCESS
        if complete
        else (
            ContactSearchStatus.CANDIDATE_OVERFLOW
            if failure == RodContactSearchFailure.CAPACITY_OVERFLOW
            else ContactSearchStatus.MEMORY_LIMIT
        )
    )
    face_vertex = ContactStencilBatch.empty(
        ContactStencilKind.FACE_VERTEX, 0, dtype=start.dtype
    )
    candidate_count = len(pair_records) + len(plane_records)
    epoch_id = canonical_fingerprint(
        {
            "kind": "rod-contact-candidate-epoch",
            "prepared": prepared.prepared_id,
            "positions": array_tree_fingerprint(start),
            "edge_vertex": edge_vertex.batch_id,
            "edge_edge": edge_edge.batch_id,
            "complete": complete,
        }
    )
    epoch = ContactCandidateEpoch(
        edge_vertex,
        edge_edge,
        face_vertex,
        jnp.asarray(start),
        jnp.asarray(0.0, dtype=start.dtype),
        jnp.asarray(candidate_count, dtype=jnp.int32),
        jnp.asarray(candidate_count * 192, dtype=jnp.int64),
        jnp.asarray(0.0, dtype=start.dtype),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(complete),
        prepared.plan.plan_id,
        epoch_id,
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


class RodContactManifoldState(StrictModule, NonTrainableState):
    """Persistent key-addressed rod contact history with retained inactive routes."""

    route_keys: Array
    occupied: Array
    active: Array
    left_witness: Array
    right_witness: Array
    normal: Array
    tangent_basis: Array
    impulse: Array
    sticking: Array
    slip: Array
    age: Array
    retention: Array
    material_revision: Array
    capacity: int = eqx.field(static=True)
    tangent_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        route_keys: ArrayLike,
        occupied: ArrayLike,
        active: ArrayLike,
        left_witness: ArrayLike,
        right_witness: ArrayLike,
        normal: ArrayLike,
        tangent_basis: ArrayLike,
        impulse: ArrayLike,
        sticking: ArrayLike,
        slip: ArrayLike,
        age: ArrayLike,
        retention: ArrayLike,
        material_revision: ArrayLike,
        /,
        *,
        capacity: int,
        tangent_dimension: int = 2,
    ):
        count = int(capacity)
        tangent = int(tangent_dimension)
        if count <= 0 or tangent != 2:
            raise ValueError("Rod manifold requires positive capacity and two tangents.")
        keys = np.asarray(route_keys)
        occupied_ = np.asarray(occupied, dtype=bool)
        active_ = np.asarray(active, dtype=bool)
        left = np.asarray(left_witness)
        right = np.asarray(right_witness)
        normal_ = np.asarray(normal)
        basis = np.asarray(tangent_basis)
        impulse_ = np.asarray(impulse)
        sticking_ = np.asarray(sticking, dtype=bool)
        slip_ = np.asarray(slip)
        age_ = np.asarray(age)
        retention_ = np.asarray(retention)
        revision = np.asarray(material_revision)
        if keys.shape != (count,) or not np.issubdtype(keys.dtype, np.integer):
            raise TypeError("Manifold route keys must be one integer capacity vector.")
        if occupied_.shape != (count,) or active_.shape != (count,):
            raise ValueError("Manifold masks must have capacity shape.")
        if np.any(active_ & ~occupied_):
            raise ValueError("Active manifold routes must be occupied.")
        if np.unique(keys[occupied_]).size != int(np.count_nonzero(occupied_)):
            raise ValueError("Occupied manifold route keys must be unique.")
        for value in (left, right, normal_):
            if value.shape != (count, 3):
                raise ValueError(
                    "Manifold witness vectors must have shape (capacity, 3)."
                )
        if basis.shape != (count, 3, tangent):
            raise ValueError("Manifold tangent bases have invalid shape.")
        if impulse_.shape != (count, 1 + tangent) or slip_.shape != (count, tangent):
            raise ValueError("Manifold impulse/slip arrays have invalid shape.")
        if sticking_.shape != (count,):
            raise ValueError("Manifold sticking flags must have capacity shape.")
        for value in (age_, retention_, revision):
            if value.shape != (count,) or not np.issubdtype(value.dtype, np.integer):
                raise TypeError(
                    "Manifold age, retention, and revision must be integer vectors."
                )
        if np.any(age_ < 0) or np.any(retention_ < 0):
            raise ValueError("Manifold age and retention must be nonnegative.")
        self.route_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.occupied = jnp.asarray(occupied_)
        self.active = jnp.asarray(active_)
        dtype = jnp.result_type(left, right, normal_, basis, impulse_, slip_)
        self.left_witness = jnp.asarray(left, dtype=dtype)
        self.right_witness = jnp.asarray(right, dtype=dtype)
        self.normal = jnp.asarray(normal_, dtype=dtype)
        self.tangent_basis = jnp.asarray(basis, dtype=dtype)
        self.impulse = jnp.asarray(impulse_, dtype=dtype)
        self.sticking = jnp.asarray(sticking_)
        self.slip = jnp.asarray(slip_, dtype=dtype)
        self.age = jnp.asarray(age_, dtype=jnp.int32)
        self.retention = jnp.asarray(retention_, dtype=jnp.int32)
        self.material_revision = jnp.asarray(revision, dtype=jnp.int64)
        self.capacity = count
        self.tangent_dimension = tangent

    @classmethod
    def empty(
        cls,
        capacity: int,
        /,
        *,
        tangent_dimension: int = 2,
        dtype: Any = np.float64,
    ) -> RodContactManifoldState:
        count = int(capacity)
        tangent = int(tangent_dimension)
        vectors = np.zeros((count, 3), dtype=dtype)
        return cls(
            np.zeros((count,), dtype=np.int64),
            np.zeros((count,), dtype=bool),
            np.zeros((count,), dtype=bool),
            vectors,
            vectors,
            vectors,
            np.zeros((count, 3, tangent), dtype=dtype),
            np.zeros((count, 1 + tangent), dtype=dtype),
            np.zeros((count,), dtype=bool),
            np.zeros((count, tangent), dtype=dtype),
            np.zeros((count,), dtype=np.int32),
            np.zeros((count,), dtype=np.int32),
            np.zeros((count,), dtype=np.int64),
            capacity=count,
            tangent_dimension=tangent,
        )

    def update(
        self,
        witnesses: RodContactWitnessBatch,
        /,
        *,
        material_revision: ArrayLike,
        retention_steps: int = 2,
    ) -> RodContactManifoldTransition:
        return _update_manifold(
            self,
            witnesses,
            material_revision=material_revision,
            retention_steps=retention_steps,
        )

    def record_response(
        self,
        route_keys: ArrayLike,
        impulses: ArrayLike,
        sticking: ArrayLike,
        slip_velocity: ArrayLike,
        /,
        *,
        step_size: ArrayLike,
    ) -> RodContactManifoldState:
        keys = np.asarray(route_keys)
        impulse_ = np.asarray(impulses)
        sticking_ = np.asarray(sticking, dtype=bool)
        slip_rate = np.asarray(slip_velocity)
        step = float(np.asarray(step_size))
        if not isfinite(step) or step < 0.0:
            raise ValueError("step_size must be finite and nonnegative.")
        count = keys.size
        if keys.shape != (count,) or impulse_.shape != (count, 3):
            raise ValueError("Response keys/impulses have inconsistent shapes.")
        if sticking_.shape != (count,) or slip_rate.shape != (count, 2):
            raise ValueError("Response stick/slip arrays have inconsistent shapes.")
        stored_keys = np.asarray(self.route_keys)
        occupied = np.asarray(self.occupied)
        impulse = np.asarray(self.impulse).copy()
        stick = np.asarray(self.sticking).copy()
        slip = np.asarray(self.slip).copy()
        lookup = {
            int(key): index
            for index, key in enumerate(stored_keys.tolist())
            if occupied[index]
        }
        for response_index, key in enumerate(keys.tolist()):
            state_index = lookup.get(int(key))
            if state_index is None:
                continue
            impulse[state_index] = impulse_[response_index]
            stick[state_index] = sticking_[response_index]
            slip[state_index] += step * slip_rate[response_index]
        return RodContactManifoldState(
            self.route_keys,
            self.occupied,
            self.active,
            self.left_witness,
            self.right_witness,
            self.normal,
            self.tangent_basis,
            impulse,
            stick,
            slip,
            self.age,
            self.retention,
            self.material_revision,
            capacity=self.capacity,
            tangent_dimension=self.tangent_dimension,
        )

    def commit(
        self,
        witnesses: RodContactWitnessBatch,
        response: CompositeContactResult,
        /,
        *,
        step_size: ArrayLike,
    ) -> RodContactManifoldState:
        if not isinstance(response, CompositeContactResult):
            raise TypeError("response must be CompositeContactResult.")
        if response.response_id != witnesses.batch_id:
            raise ValueError("Response and witness batches do not match.")
        return self.record_response(
            witnesses.route_keys,
            response.impulse,
            response.sticking,
            response.slip_velocity,
            step_size=step_size,
        )


class RodContactManifoldTransition(StrictModule, NonTrainableState):
    state: RodContactManifoldState
    witnesses: RodContactWitnessBatch
    warm_start: Array
    born_keys: Array
    died_keys: Array
    resurrected_keys: Array
    born_count: Array
    died_count: Array
    resurrected_count: Array
    material_changed_count: Array
    frame_transport_residual: Array
    finite: Array
    successful: Array


class _HistoryRecord:
    __slots__ = (
        "key",
        "active",
        "left",
        "right",
        "normal",
        "basis",
        "impulse",
        "sticking",
        "slip",
        "age",
        "retention",
        "revision",
    )

    def __init__(
        self,
        key: int,
        active: bool,
        left: np.ndarray,
        right: np.ndarray,
        normal: np.ndarray,
        basis: np.ndarray,
        impulse: np.ndarray,
        sticking: bool,
        slip: np.ndarray,
        age: int,
        retention: int,
        revision: int,
    ):
        self.key = key
        self.active = active
        self.left = left
        self.right = right
        self.normal = normal
        self.basis = basis
        self.impulse = impulse
        self.sticking = sticking
        self.slip = slip
        self.age = age
        self.retention = retention
        self.revision = revision


def _revision_vector(
    material_revision: ArrayLike,
    witnesses: RodContactWitnessBatch,
    /,
) -> np.ndarray:
    revision = np.asarray(material_revision)
    if revision.shape == ():
        return np.full((witnesses.capacity,), int(revision), dtype=np.int64)
    if revision.shape != (witnesses.capacity,) or not np.issubdtype(
        revision.dtype, np.integer
    ):
        raise TypeError("material_revision must be an integer scalar or capacity vector.")
    return revision.astype(np.int64, copy=False)


def _witnesses_with_frames(
    witnesses: RodContactWitnessBatch, frames: np.ndarray, /
) -> RodContactWitnessBatch:
    return RodContactWitnessBatch(
        witnesses.vertex_indices,
        witnesses.left_feature_ids,
        witnesses.right_feature_ids,
        witnesses.route_keys,
        witnesses.stencil_kinds,
        witnesses.left_segment_indices,
        witnesses.right_segment_indices,
        witnesses.left_parameters,
        witnesses.right_parameters,
        witnesses.coefficients,
        witnesses.left_centerline_witness,
        witnesses.right_centerline_witness,
        witnesses.left_surface_witness,
        witnesses.right_surface_witness,
        witnesses.left_axis,
        witnesses.right_axis,
        witnesses.normal,
        frames,
        witnesses.centerline_distance,
        witnesses.physical_gap,
        witnesses.activation_gap,
        witnesses.left_radius,
        witnesses.right_radius,
        witnesses.valid,
        capacity=witnesses.capacity,
        finite=witnesses.finite,
        batch_id=witnesses.batch_id,
    )


def _update_manifold(
    state: RodContactManifoldState,
    witnesses: RodContactWitnessBatch,
    /,
    *,
    material_revision: ArrayLike,
    retention_steps: int,
) -> RodContactManifoldTransition:
    if not isinstance(state, RodContactManifoldState):
        raise TypeError("state must be RodContactManifoldState.")
    if not isinstance(witnesses, RodContactWitnessBatch):
        raise TypeError("witnesses must be RodContactWitnessBatch.")
    if witnesses.capacity > state.capacity:
        raise ValueError("Witness capacity exceeds manifold history capacity.")
    retention_limit = int(retention_steps)
    if retention_limit < 0:
        raise ValueError("retention_steps must be nonnegative.")
    revisions = _revision_vector(material_revision, witnesses)
    old_occupied = np.asarray(state.occupied)
    old_active = np.asarray(state.active)
    old_keys = np.asarray(state.route_keys)
    old_lookup = {
        int(key): index
        for index, key in enumerate(old_keys.tolist())
        if old_occupied[index]
    }
    witness_active = np.asarray(witnesses.valid)
    witness_keys = np.asarray(witnesses.route_keys)
    frames = np.asarray(witnesses.tangent_basis).copy()
    warm = np.zeros((witnesses.capacity, 3), dtype=np.asarray(state.impulse).dtype)
    records: list[_HistoryRecord] = []
    born: list[int] = []
    resurrected: list[int] = []
    material_changed = 0
    active_key_set: set[int] = set()
    for slot in np.flatnonzero(witness_active).tolist():
        key = int(witness_keys[slot])
        active_key_set.add(key)
        old_slot = old_lookup.get(key)
        revision = int(revisions[slot])
        if old_slot is None:
            born.append(key)
            impulse = np.zeros((3,), dtype=warm.dtype)
            slip = np.zeros((2,), dtype=warm.dtype)
            sticking = True
            age = 0
        else:
            if not old_active[old_slot]:
                resurrected.append(key)
            changed = int(np.asarray(state.material_revision)[old_slot]) != revision
            material_changed += int(changed)
            if changed:
                impulse = np.zeros((3,), dtype=warm.dtype)
                slip = np.zeros((2,), dtype=warm.dtype)
                sticking = True
            else:
                impulse = np.asarray(state.impulse)[old_slot].copy()
                slip = np.asarray(state.slip)[old_slot].copy()
                sticking = bool(np.asarray(state.sticking)[old_slot])
            age = int(np.asarray(state.age)[old_slot]) + 1
            warm[slot] = impulse
        records.append(
            _HistoryRecord(
                key,
                True,
                np.asarray(witnesses.left_surface_witness)[slot].copy(),
                np.asarray(witnesses.right_surface_witness)[slot].copy(),
                np.asarray(witnesses.normal)[slot].copy(),
                frames[slot].copy(),
                impulse,
                sticking,
                slip,
                age,
                retention_limit,
                revision,
            )
        )
    for old_slot in np.flatnonzero(old_occupied).tolist():
        key = int(old_keys[old_slot])
        if key in active_key_set:
            continue
        remaining = int(np.asarray(state.retention)[old_slot]) - 1
        if remaining <= 0:
            continue
        records.append(
            _HistoryRecord(
                key,
                False,
                np.asarray(state.left_witness)[old_slot].copy(),
                np.asarray(state.right_witness)[old_slot].copy(),
                np.asarray(state.normal)[old_slot].copy(),
                np.asarray(state.tangent_basis)[old_slot].copy(),
                np.asarray(state.impulse)[old_slot].copy(),
                bool(np.asarray(state.sticking)[old_slot]),
                np.asarray(state.slip)[old_slot].copy(),
                int(np.asarray(state.age)[old_slot]) + 1,
                remaining,
                int(np.asarray(state.material_revision)[old_slot]),
            )
        )
    active_records = [record for record in records if record.active]
    retained_records = [record for record in records if not record.active]
    retained_records.sort(key=lambda record: (-record.retention, -record.age, record.key))
    kept = (
        active_records + retained_records[: max(state.capacity - len(active_records), 0)]
    )
    kept.sort(key=lambda record: record.key)
    kept_keys = {record.key for record in kept}
    died = sorted(
        int(key)
        for index, key in enumerate(old_keys.tolist())
        if old_occupied[index] and int(key) not in kept_keys
    )
    capacity = state.capacity
    keys = np.zeros((capacity,), dtype=np.int64)
    occupied = np.zeros((capacity,), dtype=bool)
    active = np.zeros((capacity,), dtype=bool)
    dtype = np.asarray(state.impulse).dtype
    left = np.zeros((capacity, 3), dtype=dtype)
    right = np.zeros((capacity, 3), dtype=dtype)
    normal = np.zeros((capacity, 3), dtype=dtype)
    basis = np.zeros((capacity, 3, 2), dtype=dtype)
    impulse = np.zeros((capacity, 3), dtype=dtype)
    sticking = np.zeros((capacity,), dtype=bool)
    slip = np.zeros((capacity, 2), dtype=dtype)
    age = np.zeros((capacity,), dtype=np.int32)
    retention = np.zeros((capacity,), dtype=np.int32)
    revision = np.zeros((capacity,), dtype=np.int64)
    for slot, record in enumerate(kept):
        keys[slot] = record.key
        occupied[slot] = True
        active[slot] = record.active
        left[slot] = record.left
        right[slot] = record.right
        normal[slot] = record.normal
        basis[slot] = record.basis
        impulse[slot] = record.impulse
        sticking[slot] = record.sticking
        slip[slot] = record.slip
        age[slot] = record.age
        retention[slot] = record.retention
        revision[slot] = record.revision
    new_state = RodContactManifoldState(
        keys,
        occupied,
        active,
        left,
        right,
        normal,
        basis,
        impulse,
        sticking,
        slip,
        age,
        retention,
        revision,
        capacity=capacity,
        tangent_dimension=2,
    )
    born_keys = np.full((capacity,), -1, dtype=np.int64)
    died_keys = np.full((capacity,), -1, dtype=np.int64)
    resurrected_keys = np.full((capacity,), -1, dtype=np.int64)
    born_keys[: len(born)] = sorted(born)
    died_keys[: len(died)] = died
    resurrected_keys[: len(resurrected)] = sorted(resurrected)
    active_frames = frames[witness_active]
    active_normals = np.asarray(witnesses.normal)[witness_active]
    if active_frames.size:
        normal_residual = np.max(
            np.abs(np.einsum("nij,ni->nj", active_frames, active_normals))
        )
        gram = np.einsum("nij,nik->njk", active_frames, active_frames)
        orthogonal_residual = np.max(np.abs(gram - np.eye(2)))
        frame_residual = max(float(normal_residual), float(orthogonal_residual))
    else:
        frame_residual = 0.0
    finite = bool(
        np.all(np.isfinite(left[occupied]))
        and np.all(np.isfinite(right[occupied]))
        and np.all(np.isfinite(basis[occupied]))
        and np.all(np.isfinite(impulse[occupied]))
        and np.all(np.isfinite(slip[occupied]))
    )
    return RodContactManifoldTransition(
        new_state,
        _witnesses_with_frames(witnesses, frames),
        jnp.asarray(warm),
        jnp.asarray(born_keys),
        jnp.asarray(died_keys),
        jnp.asarray(resurrected_keys),
        jnp.asarray(len(born), dtype=jnp.int32),
        jnp.asarray(len(died), dtype=jnp.int32),
        jnp.asarray(len(resurrected), dtype=jnp.int32),
        jnp.asarray(material_changed, dtype=jnp.int32),
        jnp.asarray(frame_residual, dtype=new_state.impulse.dtype),
        jnp.asarray(finite),
        jnp.asarray(finite),
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
    ):
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
        if stencil_kinds[slot] == int(ContactStencilKind.EDGE_VERTEX):
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
        pair_full_safe = False
        supported_initial_plane = (
            int(route_keys[slot]) in supported_plane_route_keys
            and stencil_kinds[slot] == int(ContactStencilKind.EDGE_VERTEX)
            and gap <= plan.distance_tolerance
        )
        if supported_initial_plane:
            end_gap = gap_action(1.0)
            evaluations += 1
            minimum_gap = min(minimum_gap, end_gap)
            if end_gap >= -plan.distance_tolerance:
                pair_full_safe = True
                safe_time = 1.0
            else:
                pair_impact = True
        elif gap <= plan.distance_tolerance:
            pair_impact = True
        elif speed <= np.finfo(start.dtype).eps:
            pair_full_safe = True
            safe_time = 1.0
        else:
            for _ in range(plan.maximum_iterations):
                total_iterations += 1
                certified_increment = max((gap - plan.distance_tolerance) / speed, 0.0)
                if certified_increment >= 1.0 - time:
                    safe_time = 1.0
                    pair_full_safe = True
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

    def __init__(
        self,
        velocity_operator: AbstractLinearOperator,
        inverse_mass_operator: AbstractLinearOperator,
        free_velocity: PyTree[Any],
        /,
        *,
        block_id: str | None = None,
    ):
        if not isinstance(velocity_operator, AbstractLinearOperator):
            raise TypeError("velocity_operator must be AbstractLinearOperator.")
        if not isinstance(inverse_mass_operator, AbstractLinearOperator):
            raise TypeError("inverse_mass_operator must be AbstractLinearOperator.")
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

    def action(rate):
        surface_velocity = participant.velocities(configuration_, rate)
        gathered = jnp.where(inside[..., None], surface_velocity[safe_indices], 0.0)
        relative = jnp.sum(witnesses.coefficients[..., None] * gathered, axis=1)
        normal_velocity = jnp.sum(relative * witnesses.normal, axis=-1, keepdims=True)
        tangent_velocity = jnp.sum(
            witnesses.tangent_basis * relative[..., :, None], axis=-2
        )
        local = jnp.concatenate((normal_velocity, tangent_velocity), axis=-1)
        return jnp.where(witnesses.valid[:, None], local, 0.0)

    def transpose_action(local_impulse):
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

    def radial_fallback(indices, radii):
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

    def action(rate):
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

    def transpose_action(local_impulse):
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
    converged: Array
    iterations: Array
    projected_residual: Array
    complementarity_defect: Array
    cone_defect: Array
    minimum_normal_impulse: Array
    minimum_normal_velocity: Array
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
    impulse: Array
    candidate_impulse: Array
    generalized_impulses: tuple[PyTree[Array], ...]
    velocity_updates: tuple[PyTree[Array], ...]
    post_velocities: tuple[PyTree[Array], ...]
    post_contact_velocity: Array
    sticking: Array
    slip_velocity: Array
    route_keys: Array
    evidence: CompositeContactResponseEvidence
    response_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class CompositeContactResponse(StrictModule, NonTrainableState):
    """Block response through the true-dual composite ``G M^-1 G*``."""

    blocks: tuple[CompositeContactParticipantBlock, ...]
    witnesses: RodContactWitnessBatch
    delassus_operator: AbstractLinearOperator
    free_contact_velocity: Array
    dynamic_friction: Array
    static_friction: Array
    compliance: Array
    solver: ContactConeSolverPlan
    response_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[CompositeContactParticipantBlock],
        witnesses: RodContactWitnessBatch,
        /,
        *,
        dynamic_friction: ArrayLike,
        static_friction: ArrayLike | None = None,
        compliance: ArrayLike = 0.0,
        normal_bias: ArrayLike = 0.0,
        solver: ContactConeSolverPlan | None = None,
    ):
        values = tuple(blocks)
        if not values or not all(
            isinstance(value, CompositeContactParticipantBlock) for value in values
        ):
            raise TypeError(
                "blocks must contain CompositeContactParticipantBlock values."
            )
        if not isinstance(witnesses, RodContactWitnessBatch):
            raise TypeError("witnesses must be RodContactWitnessBatch.")
        contact_space = values[0].velocity_operator.target
        expected = ArraySpace(
            (witnesses.capacity, 3),
            dtype=witnesses.normal.dtype,
        )
        if not contact_space.compatible(expected) or any(
            not value.velocity_operator.target.compatible(contact_space)
            for value in values[1:]
        ):
            raise ValueError(
                "Composite contact blocks must share the witness contact space."
            )
        dynamic = _route_values(
            dynamic_friction,
            witnesses.capacity,
            witnesses.normal.dtype,
            "dynamic_friction",
        )
        static = (
            dynamic
            if static_friction is None
            else _route_values(
                static_friction,
                witnesses.capacity,
                witnesses.normal.dtype,
                "static_friction",
            )
        )
        if np.any(np.asarray(dynamic) < 0.0) or np.any(
            np.asarray(static) < np.asarray(dynamic)
        ):
            raise ValueError("Friction coefficients require static >= dynamic >= 0.")
        compliance_ = _local_values(
            compliance,
            witnesses.capacity,
            witnesses.normal.dtype,
            "compliance",
        )
        if np.any(np.asarray(compliance_) < 0.0):
            raise ValueError("Contact compliance must be nonnegative.")
        bias = _route_values(
            normal_bias,
            witnesses.capacity,
            witnesses.normal.dtype,
            "normal_bias",
        )
        solver_ = ContactConeSolverPlan() if solver is None else solver
        if not isinstance(solver_, ContactConeSolverPlan):
            raise TypeError("solver must be ContactConeSolverPlan or None.")

        def delassus_action(local_impulse):
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
        free = free.at[:, 0].add(bias)
        free = jnp.where(witnesses.valid[:, None], free, 0.0)
        identifier = witnesses.batch_id
        self.blocks = values
        self.witnesses = witnesses
        self.delassus_operator = delassus
        self.free_contact_velocity = free
        self.dynamic_friction = dynamic
        self.static_friction = static
        self.compliance = compliance_
        self.solver = solver_
        self.response_id = identifier

    def solve(
        self, /, *, initial_impulse: ArrayLike | None = None
    ) -> CompositeContactResult:
        """Solve with matrix-free Delassus actions; failures apply zero impulse."""

        return _solve_composite_response(self, initial_impulse, backend="matrix-free")

    def solve_dense_authority(
        self,
        /,
        *,
        initial_impulse: ArrayLike | None = None,
        materialization_policy: MaterializationPolicy | None = None,
    ) -> CompositeContactResult:
        """Solve the bounded dense correctness authority for parity evidence."""

        policy = (
            MaterializationPolicy()
            if materialization_policy is None
            else materialization_policy
        )
        if not isinstance(policy, MaterializationPolicy):
            raise TypeError(
                "materialization_policy must be MaterializationPolicy or None."
            )
        dense = materialize(self.delassus_operator, policy).reshape(
            (
                self.witnesses.capacity,
                3,
                self.witnesses.capacity,
                3,
            )
        )
        return _solve_composite_response(
            self, initial_impulse, backend="dense-authority", dense=dense
        )


def _route_values(
    value: ArrayLike,
    capacity: int,
    dtype: Any,
    name: str,
    /,
) -> Array:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.shape == ():
        array = np.full((capacity,), float(array), dtype=np.dtype(dtype))
    if array.shape != (capacity,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or capacity vector.")
    return jnp.asarray(array)


def _local_values(
    value: ArrayLike,
    capacity: int,
    dtype: Any,
    name: str,
    /,
) -> Array:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.shape == ():
        array = np.full((capacity, 3), float(array), dtype=np.dtype(dtype))
    elif array.shape == (capacity,):
        array = np.repeat(array[:, None], 3, axis=1)
    if array.shape != (capacity, 3) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar, capacity, or (capacity, 3).")
    return jnp.asarray(array)


def _matrix_free_contact_solve(
    action,
    free: Array,
    compliance: Array,
    friction: Array,
    valid: Array,
    initial: Array,
    solver: ContactConeSolverPlan,
    /,
) -> tuple[Array, Array, Array]:
    mask = valid[:, None]

    def law_action(value):
        active = jnp.where(mask, value, 0.0)
        return jnp.where(mask, action(active) + compliance * active, 0.0)

    vector = jnp.where(mask, jnp.ones_like(initial), 0.0)
    vector_norm = jnp.sqrt(jnp.sum(vector * vector))
    vector = jnp.where(vector_norm > 0.0, vector / vector_norm, vector)
    spectral = jnp.asarray(0.0, dtype=free.dtype)
    for _ in range(16):
        image = law_action(vector)
        image_norm = jnp.sqrt(jnp.sum(image * image))
        vector = jnp.where(image_norm > 0.0, image / image_norm, vector)
        spectral = jnp.maximum(spectral, image_norm)
    step = solver.relaxation / jnp.maximum(
        1.05 * spectral, jnp.asarray(1.0, dtype=free.dtype)
    )
    tolerance = solver.absolute_tolerance + solver.relative_tolerance * jnp.maximum(
        1.0, jnp.sqrt(jnp.sum(free * free))
    )
    value = jnp.where(
        mask,
        project_signorini_coulomb_product(initial, friction),
        0.0,
    )
    converged = jnp.asarray(False)
    first_converged = jnp.asarray(solver.maximum_iterations, dtype=jnp.int32)
    residual_norm = jnp.asarray(jnp.inf, dtype=free.dtype)

    def iteration_body(iteration, state):
        value_, converged_, first_converged_, residual_norm_ = state
        gradient = law_action(value_) + free
        projected = project_signorini_coulomb_product(
            value_ - step * gradient,
            friction,
        )
        projected = jnp.where(mask, projected, 0.0)
        residual = value_ - projected
        norm = jnp.sqrt(jnp.sum(residual * residual))
        now = norm <= tolerance
        first_converged_ = jnp.where(
            (~converged_) & now,
            jnp.asarray(iteration + 1, dtype=jnp.int32),
            first_converged_,
        )
        value_ = jnp.where(converged_, value_, projected)
        converged_ = converged_ | now
        residual_norm_ = jnp.where(
            converged_,
            jnp.minimum(residual_norm_, norm),
            norm,
        )
        return value_, converged_, first_converged_, residual_norm_

    value, _, first_converged, residual_norm = jax.lax.fori_loop(
        0,
        solver.maximum_iterations,
        iteration_body,
        (value, converged, first_converged, residual_norm),
    )
    return value, first_converged, residual_norm


def _solve_composite_response(
    response: CompositeContactResponse,
    initial_impulse: ArrayLike | None,
    /,
    *,
    backend: str,
    dense: Array | None = None,
) -> CompositeContactResult:
    capacity = response.witnesses.capacity
    dtype = response.free_contact_velocity.dtype
    initial = (
        jnp.zeros((capacity, 3), dtype=dtype)
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=dtype)
    )
    if initial.shape != (capacity, 3):
        raise ValueError("initial_impulse must have witness contact shape.")
    valid = response.witnesses.valid
    if dense is None:
        action = response.delassus_operator.mv
    else:
        action = lambda value: contract("aibj,bj->ai", dense, value)
    static_candidate, _, _ = _matrix_free_contact_solve(
        action,
        response.free_contact_velocity,
        response.compliance,
        response.static_friction,
        valid,
        initial,
        response.solver,
    )
    static_velocity = (
        action(static_candidate)
        + response.compliance * static_candidate
        + response.free_contact_velocity
    )
    static_tangent_norm = jnp.sqrt(
        jnp.sum(static_velocity[:, 1:] * static_velocity[:, 1:], axis=-1)
    )
    static_impulse_norm = jnp.sqrt(
        jnp.sum(static_candidate[:, 1:] * static_candidate[:, 1:], axis=-1)
    )
    tolerance = response.solver.absolute_tolerance + response.solver.relative_tolerance
    static_sticking = (
        valid
        & (static_tangent_norm <= tolerance)
        & (
            static_impulse_norm
            <= response.static_friction * jnp.maximum(static_candidate[:, 0], 0.0)
            + tolerance
        )
    )
    selected_friction = jnp.where(
        static_sticking, response.static_friction, response.dynamic_friction
    )
    candidate, iterations, residual = _matrix_free_contact_solve(
        action,
        response.free_contact_velocity,
        response.compliance,
        selected_friction,
        valid,
        static_candidate,
        response.solver,
    )
    law_velocity = (
        action(candidate)
        + response.compliance * candidate
        + response.free_contact_velocity
    )
    tangent_norm = jnp.sqrt(jnp.sum(candidate[:, 1:] ** 2, axis=-1))
    cone_defect = jnp.max(
        jnp.where(
            valid,
            jnp.maximum(
                tangent_norm - selected_friction * jnp.maximum(candidate[:, 0], 0.0),
                0.0,
            ),
            0.0,
        ),
        initial=0.0,
    )
    complementarity = jnp.max(
        jnp.where(
            valid,
            jnp.abs(jnp.minimum(candidate[:, 0], law_velocity[:, 0])),
            0.0,
        ),
        initial=0.0,
    )
    active = jnp.any(valid)
    minimum_impulse = jnp.where(
        active,
        jnp.min(jnp.where(valid, candidate[:, 0], jnp.inf)),
        jnp.asarray(0.0, dtype=dtype),
    )
    minimum_velocity = jnp.where(
        active,
        jnp.min(jnp.where(valid, law_velocity[:, 0], jnp.inf)),
        jnp.asarray(0.0, dtype=dtype),
    )
    finite = (
        jnp.all(jnp.isfinite(candidate))
        & jnp.all(jnp.isfinite(law_velocity))
        & jnp.isfinite(residual)
    )
    certificate_scale = jnp.maximum(
        1.0,
        jnp.sqrt(jnp.sum(response.free_contact_velocity**2)),
    )
    certificate_tolerance = jnp.maximum(
        response.solver.absolute_tolerance
        + response.solver.relative_tolerance * certificate_scale,
        jnp.finfo(dtype).eps
        * max(64, 8 * int(np.sqrt(response.free_contact_velocity.size)))
        * certificate_scale,
    )
    converged = residual <= certificate_tolerance
    preliminary_success = (
        converged
        & finite
        & (cone_defect <= certificate_tolerance)
        & (complementarity <= certificate_tolerance)
        & (minimum_impulse >= -certificate_tolerance)
        & (minimum_velocity >= -certificate_tolerance)
    )
    accepted = jnp.where(preliminary_success, candidate, jnp.zeros_like(candidate))
    generalized_impulses: list[PyTree[Array]] = []
    velocity_updates: list[PyTree[Array]] = []
    post_velocities: list[PyTree[Array]] = []
    duality_residuals: list[Array] = []
    duality_scales: list[Array] = []
    contact_dual = DualSpace(response.delassus_operator.target)
    for block in response.blocks:
        effort = dual_transpose(block.velocity_operator).mv(accepted)
        update = block.inverse_mass_operator.mv(effort)
        post = jax.tree.map(lambda free, delta: free + delta, block.free_velocity, update)
        local_update = block.velocity_operator.mv(update)
        contact_power = contact_dual.pair(accepted, local_update)
        generalized_power = DualSpace(block.velocity_operator.source).pair(effort, update)
        difference = contact_power - generalized_power
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(contact_power), jnp.abs(generalized_power))
        )
        generalized_impulses.append(effort)
        velocity_updates.append(update)
        post_velocities.append(post)
        duality_residuals.append(jnp.abs(difference))
        duality_scales.append(scale)
    duality_residual = jnp.max(jnp.stack(tuple(duality_residuals)), initial=0.0)
    duality_scale = jnp.max(jnp.stack(tuple(duality_scales)), initial=1.0)
    duality_tolerance = jnp.finfo(dtype).eps * max(
        64,
        8 * response.delassus_operator.source.size,
        *(8 * block.velocity_operator.source.size for block in response.blocks),
    )
    duality_valid = duality_residual <= duality_tolerance * duality_scale
    successful = preliminary_success & duality_valid
    accepted = jnp.where(successful, accepted, jnp.zeros_like(accepted))
    if generalized_impulses:
        generalized_impulses = [
            jax.tree.map(lambda value: jnp.where(successful, value, 0.0), effort)
            for effort in generalized_impulses
        ]
        velocity_updates = [
            jax.tree.map(lambda value: jnp.where(successful, value, 0.0), update)
            for update in velocity_updates
        ]
        post_velocities = [
            jax.tree.map(
                lambda free, update: free + update,
                block.free_velocity,
                update,
            )
            for block, update in zip(response.blocks, velocity_updates, strict=True)
        ]
    post_contact = action(accepted) + response.free_contact_velocity
    slip_velocity = post_contact[:, 1:]
    slip_norm = jnp.sqrt(jnp.sum(slip_velocity * slip_velocity, axis=-1))
    impulse_tangent_norm = jnp.sqrt(jnp.sum(accepted[:, 1:] ** 2, axis=-1))
    sticking = (
        valid
        & (slip_norm <= certificate_tolerance)
        & (
            impulse_tangent_norm
            <= response.static_friction * jnp.maximum(accepted[:, 0], 0.0)
            + certificate_tolerance
        )
    )
    evidence = CompositeContactResponseEvidence(
        converged,
        iterations,
        residual,
        complementarity,
        cone_defect,
        minimum_impulse,
        minimum_velocity,
        duality_residual,
        duality_scale,
        duality_valid,
        finite,
        successful,
        ~successful,
        successful,
        backend,
        response.response_id,
    )
    return CompositeContactResult(
        accepted,
        candidate,
        tuple(generalized_impulses),
        tuple(velocity_updates),
        tuple(post_velocities),
        post_contact,
        sticking,
        slip_velocity,
        response.witnesses.route_keys,
        evidence,
        response.response_id,
    )


__all__ = [
    "build_capsule_contact_velocity_operator",
    "build_rod_contact_velocity_operator",
    "CompositeContactParticipantBlock",
    "CompositeContactResponse",
    "CompositeContactResponseEvidence",
    "CompositeContactResult",
    "prepare_composite_contact_block",
    "PreparedRodContactSearch",
    "RodContactCCDEvidence",
    "RodContactCCDPlan",
    "RodContactCCDResult",
    "RodContactCCDStatus",
    "RodContactManifoldState",
    "RodContactManifoldTransition",
    "RodContactSearchEvidence",
    "RodContactSearchFailure",
    "RodContactSearchPlan",
    "RodContactSearchResult",
    "RodContactSearchRoute",
    "RodContactWitnessBatch",
]

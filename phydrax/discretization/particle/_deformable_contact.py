#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


class DeformableContactRouteKind(IntEnum):
    """Geometry carried by one node-to-surface contact candidate."""

    PLANE = 0
    NODE = 1
    SEGMENT = 2
    TRIANGLE = 3


def _route_keys(
    kind: np.ndarray,
    query: np.ndarray,
    surface: np.ndarray,
    /,
) -> np.ndarray:
    mask = (1 << 64) - 1
    keys = np.empty(kind.shape, dtype=np.int64)
    for slot in range(kind.size):
        value = 1469598103934665603
        for item in (slot, int(kind[slot]), int(query[slot]), *surface[slot].tolist()):
            value ^= (int(item) + 0x9E3779B97F4A7C15) & mask
            value = (value * 1099511628211) & mask
        keys[slot] = value & ((1 << 63) - 1)
    return keys


def _closest_on_segment(
    point: Array,
    first: Array,
    second: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    edge = second - first
    squared_length = jnp.sum(edge * edge, axis=-1)
    nondegenerate = squared_length > tolerance * tolerance
    safe_squared_length = jnp.where(nondegenerate, squared_length, 1.0)
    raw_coordinate = jnp.sum((point - first) * edge, axis=-1) / safe_squared_length
    coordinate = jnp.clip(raw_coordinate, 0.0, 1.0)
    weights = jnp.stack((1.0 - coordinate, coordinate), axis=-1)
    witness = first + coordinate[:, None] * edge
    branch_margin = jnp.minimum(jnp.abs(raw_coordinate), jnp.abs(1.0 - raw_coordinate))
    return witness, weights, nondegenerate, branch_margin, raw_coordinate


def _triangle_geometry(
    point: Array,
    first: Array,
    second: Array,
    third: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    edge_first = second - first
    edge_second = third - first
    cross = jnp.cross(edge_first, edge_second)
    double_area = jnp.sqrt(jnp.sum(cross * cross, axis=-1))
    nondegenerate = double_area > tolerance
    safe_area = jnp.where(nondegenerate, double_area, 1.0)
    face_normal = cross / safe_area[:, None]
    plane_gap = jnp.sum((point - first) * face_normal, axis=-1)
    face_witness = point - plane_gap[:, None] * face_normal

    dot00 = jnp.sum(edge_first * edge_first, axis=-1)
    dot01 = jnp.sum(edge_first * edge_second, axis=-1)
    dot11 = jnp.sum(edge_second * edge_second, axis=-1)
    rhs0 = jnp.sum((face_witness - first) * edge_first, axis=-1)
    rhs1 = jnp.sum((face_witness - first) * edge_second, axis=-1)
    determinant = dot00 * dot11 - dot01 * dot01
    safe_determinant = jnp.where(nondegenerate, determinant, 1.0)
    second_weight = (dot11 * rhs0 - dot01 * rhs1) / safe_determinant
    third_weight = (dot00 * rhs1 - dot01 * rhs0) / safe_determinant
    first_weight = 1.0 - second_weight - third_weight
    face_weights = jnp.stack((first_weight, second_weight, third_weight), axis=-1)
    face_valid = nondegenerate & jnp.all(face_weights >= 0.0, axis=-1)

    witness_ab, weights_ab, _, margin_ab, _ = _closest_on_segment(
        point, first, second, tolerance
    )
    witness_bc, weights_bc, _, margin_bc, _ = _closest_on_segment(
        point, second, third, tolerance
    )
    witness_ca, weights_ca, _, margin_ca, _ = _closest_on_segment(
        point, third, first, tolerance
    )
    triangle_weights_ab = jnp.stack(
        (weights_ab[:, 0], weights_ab[:, 1], jnp.zeros_like(weights_ab[:, 0])),
        axis=-1,
    )
    triangle_weights_bc = jnp.stack(
        (jnp.zeros_like(weights_bc[:, 0]), weights_bc[:, 0], weights_bc[:, 1]),
        axis=-1,
    )
    triangle_weights_ca = jnp.stack(
        (weights_ca[:, 1], jnp.zeros_like(weights_ca[:, 0]), weights_ca[:, 0]),
        axis=-1,
    )
    witnesses = jnp.stack((face_witness, witness_ab, witness_bc, witness_ca), axis=1)
    weights = jnp.stack(
        (face_weights, triangle_weights_ab, triangle_weights_bc, triangle_weights_ca),
        axis=1,
    )
    squared_distances = jnp.sum((point[:, None, :] - witnesses) ** 2, axis=-1)
    squared_distances = squared_distances.at[:, 0].set(
        jnp.where(face_valid, squared_distances[:, 0], jnp.finfo(point.dtype).max)
    )
    selected = jax.lax.stop_gradient(jnp.argmin(squared_distances, axis=-1))
    selection = jax.nn.one_hot(selected, 4, dtype=point.dtype)
    witness = contract("nk,nkd->nd", selection, witnesses)
    interpolation = contract("nk,nkj->nj", selection, weights)
    interpolation_sum = jnp.sum(interpolation, axis=-1, keepdims=True)
    interpolation = interpolation / jnp.where(
        interpolation_sum > 0.0, interpolation_sum, 1.0
    )

    displacement = point - witness
    distance = jnp.sqrt(jnp.maximum(jnp.sum(displacement * displacement, axis=-1), 0.0))
    sign = jnp.where(plane_gap >= 0.0, 1.0, -1.0)
    normal = jnp.where(
        (distance > tolerance)[:, None],
        sign[:, None]
        * displacement
        / jnp.where(distance > tolerance, distance, 1.0)[:, None],
        face_normal,
    )
    gap = sign * distance
    ordered = jnp.sort(squared_distances, axis=-1)
    selection_margin = jnp.maximum(ordered[:, 1] - ordered[:, 0], 0.0)
    edge_margins = jnp.stack(
        (
            jnp.min(jnp.maximum(face_weights, 0.0), axis=-1),
            margin_ab,
            margin_bc,
            margin_ca,
        ),
        axis=-1,
    )
    coordinate_margin = jnp.sum(selection * edge_margins, axis=-1)
    feature_margin = jnp.minimum(coordinate_margin, selection_margin)
    return witness, interpolation, normal, gap, nondegenerate, feature_margin


class DeformableContactPlan(StrictModule, NonTrainableState):
    """Fixed candidate topology for node/segment/triangle surface contact."""

    query_indices: Array
    surface_indices: Array
    route_kinds: Array
    plane_points: Array
    plane_normals: Array
    plane_velocities: Array
    candidate_mask: Array
    route_keys: Array
    ambient_dimension: int = eqx.field(static=True)
    contact_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    geometry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        query_indices: ArrayLike,
        surface_indices: ArrayLike,
        route_kinds: ArrayLike,
        /,
        *,
        ambient_dimension: int,
        contact_capacity: int,
        plane_points: ArrayLike | None = None,
        plane_normals: ArrayLike | None = None,
        plane_velocities: ArrayLike | None = None,
        candidate_mask: ArrayLike | None = None,
        route_keys: ArrayLike | None = None,
        activation_distance: float = 0.0,
        geometry_tolerance: float = 1.0e-12,
        plan_id: str | None = None,
    ):
        query = np.asarray(query_indices)
        surface = np.asarray(surface_indices)
        kinds = np.asarray(route_kinds)
        if (
            query.ndim != 1
            or query.size == 0
            or not np.issubdtype(query.dtype, np.integer)
        ):
            raise TypeError("query_indices must be a nonempty rank-1 integer array.")
        candidate_count = int(query.size)
        if surface.shape != (candidate_count, 3) or not np.issubdtype(
            surface.dtype, np.integer
        ):
            raise TypeError(
                "surface_indices must be an integer (candidate_count, 3) array."
            )
        if kinds.shape != (candidate_count,) or not np.issubdtype(
            kinds.dtype, np.integer
        ):
            raise TypeError("route_kinds must be a candidate-count integer array.")
        dimension = int(ambient_dimension)
        if dimension not in (2, 3):
            raise ValueError(
                "Deformable surface contact requires dimension two or three."
            )
        capacity = int(contact_capacity)
        if capacity <= 0:
            raise ValueError("contact_capacity must be positive.")
        activation = float(activation_distance)
        tolerance = float(geometry_tolerance)
        if not np.isfinite(activation) or activation < 0.0:
            raise ValueError("activation_distance must be finite and nonnegative.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("geometry_tolerance must be finite and positive.")
        active = (
            np.ones((candidate_count,), dtype=bool)
            if candidate_mask is None
            else np.asarray(candidate_mask, dtype=bool)
        )
        if active.shape != (candidate_count,):
            raise ValueError("candidate_mask must have candidate-count shape.")
        allowed = np.asarray([int(value) for value in DeformableContactRouteKind])
        if np.any(active & ~np.isin(kinds, allowed)):
            raise ValueError("Active route_kinds contain an unknown route kind.")
        if np.any(active & (query < 0)):
            raise ValueError("Active query indices must be nonnegative.")

        plane_point = (
            np.zeros((candidate_count, dimension), dtype=float)
            if plane_points is None
            else np.asarray(plane_points, dtype=float)
        )
        plane_normal = (
            np.zeros((candidate_count, dimension), dtype=float)
            if plane_normals is None
            else np.asarray(plane_normals, dtype=float)
        )
        plane_velocity = (
            np.zeros((candidate_count, dimension), dtype=float)
            if plane_velocities is None
            else np.asarray(plane_velocities, dtype=float)
        )
        expected_vector_shape = (candidate_count, dimension)
        if (
            plane_point.shape != expected_vector_shape
            or plane_normal.shape != expected_vector_shape
            or plane_velocity.shape != expected_vector_shape
        ):
            raise ValueError("Plane data must have (candidate_count, dimension) shape.")
        if not np.all(np.isfinite(plane_point)) or not np.all(
            np.isfinite(plane_velocity)
        ):
            raise ValueError("Plane points and velocities must be finite.")

        plane = active & (kinds == int(DeformableContactRouteKind.PLANE))
        node = active & (kinds == int(DeformableContactRouteKind.NODE))
        segment = active & (kinds == int(DeformableContactRouteKind.SEGMENT))
        triangle = active & (kinds == int(DeformableContactRouteKind.TRIANGLE))
        normal_norm = np.linalg.norm(plane_normal, axis=-1)
        if np.any(plane & (~np.isfinite(normal_norm) | (normal_norm <= tolerance))):
            raise ValueError("Plane routes require finite nonzero normals.")
        if np.any(plane & np.any(surface != -1, axis=-1)):
            raise ValueError("Plane routes use -1 in every surface-index slot.")
        arity_valid = (
            (node & (surface[:, 0] >= 0) & np.all(surface[:, 1:] == -1, axis=-1))
            | (segment & np.all(surface[:, :2] >= 0, axis=-1) & (surface[:, 2] == -1))
            | (triangle & np.all(surface >= 0, axis=-1))
            | plane
            | ~active
        )
        if not np.all(arity_valid):
            raise ValueError("Surface-index padding does not match route arity.")
        if np.any(segment) and dimension != 2:
            raise ValueError("Oriented segment routes are two-dimensional.")
        if np.any(triangle) and dimension != 3:
            raise ValueError("Triangle routes are three-dimensional.")

        if route_keys is None:
            keys = _route_keys(
                kinds.astype(np.int64), query.astype(np.int64), surface.astype(np.int64)
            )
        else:
            keys = np.asarray(route_keys)
            if keys.shape != (candidate_count,) or not np.issubdtype(
                keys.dtype, np.integer
            ):
                raise TypeError("route_keys must be a candidate-count integer array.")
            keys = keys.astype(np.int64, copy=False)
        if np.any(active & (keys < 0)) or np.unique(keys[active]).size != int(
            np.sum(active)
        ):
            raise ValueError("Active route keys must be unique and nonnegative.")

        normalized_plane_normal = plane_normal / np.where(
            normal_norm[:, None] > tolerance, normal_norm[:, None], 1.0
        )
        generated = canonical_fingerprint(
            {
                "kind": "deformable-contact-plan",
                "arrays": array_tree_fingerprint(
                    {
                        "query_indices": query,
                        "surface_indices": surface,
                        "route_kinds": kinds,
                        "plane_points": plane_point,
                        "plane_normals": normalized_plane_normal,
                        "plane_velocities": plane_velocity,
                        "candidate_mask": active,
                        "route_keys": keys,
                    }
                ),
                "ambient_dimension": dimension,
                "contact_capacity": capacity,
                "activation_distance": activation,
                "geometry_tolerance": tolerance,
            }
        )
        self.query_indices = jnp.asarray(query, dtype=jnp.int32)
        self.surface_indices = jnp.asarray(surface, dtype=jnp.int32)
        self.route_kinds = jnp.asarray(kinds, dtype=jnp.int32)
        self.plane_points = jnp.asarray(plane_point)
        self.plane_normals = jnp.asarray(normalized_plane_normal)
        self.plane_velocities = jnp.asarray(plane_velocity)
        self.candidate_mask = jnp.asarray(active)
        self.route_keys = jnp.asarray(keys, dtype=jnp.int64)
        self.ambient_dimension = dimension
        self.contact_capacity = capacity
        self.activation_distance = activation
        self.geometry_tolerance = tolerance
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self,
        query_nodes: ParticleDiscretization,
        surface_nodes: ParticleDiscretization | None = None,
        /,
    ) -> PreparedDeformableContact:
        return PreparedDeformableContact(self, query_nodes, surface_nodes)


class DeformableContactEvaluation(StrictModule):
    query_indices: Array
    query_weights: Array
    surface_indices: Array
    surface_weights: Array
    plane_velocity: Array
    route_kinds: Array
    gap: Array
    normal: Array
    query_witness: Array
    surface_witness: Array
    relative_velocity: Array
    contact_keys: Array
    valid: Array
    validity_margin: Array
    feature_margin: Array
    candidate_count: Array
    overflow_count: Array
    overflow: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class DeformableContactTransposeResult(StrictModule):
    query_action: Array
    surface_action: Array
    plane_action: Array
    balance_residual: Array
    balance_valid: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedDeformableContact(StrictModule, NonTrainableState):
    """Prepared fixed-capacity surface evaluator and exact interpolation transpose."""

    plan: DeformableContactPlan
    query_nodes: ParticleDiscretization
    surface_nodes: ParticleDiscretization
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DeformableContactPlan,
        query_nodes: ParticleDiscretization,
        surface_nodes: ParticleDiscretization | None,
        /,
    ):
        if not isinstance(plan, DeformableContactPlan):
            raise TypeError("plan must be DeformableContactPlan.")
        if not isinstance(query_nodes, ParticleDiscretization):
            raise TypeError("query_nodes must be ParticleDiscretization.")
        surface = query_nodes if surface_nodes is None else surface_nodes
        if not isinstance(surface, ParticleDiscretization):
            raise TypeError("surface_nodes must be ParticleDiscretization or None.")
        if (
            query_nodes.ambient_dimension != plan.ambient_dimension
            or surface.ambient_dimension != plan.ambient_dimension
        ):
            raise ValueError(
                "Contact plan and node supports must have matching dimension."
            )
        active = np.asarray(plan.candidate_mask)
        query = np.asarray(plan.query_indices)
        surface_index = np.asarray(plan.surface_indices)
        discrete = active & (
            np.asarray(plan.route_kinds) != int(DeformableContactRouteKind.PLANE)
        )
        if np.any(query[active] >= query_nodes.capacity):
            raise ValueError("A query route exceeds query-node capacity.")
        if np.any(surface_index[discrete] >= surface.capacity):
            raise ValueError("A surface route exceeds surface-node capacity.")
        active_query_endpoints = np.asarray(query_nodes.active_mask)[query[active]]
        used_surface_indices = surface_index[discrete]
        used_surface_indices = used_surface_indices[used_surface_indices >= 0]
        active_surface_endpoints = np.asarray(surface.active_mask)[used_surface_indices]
        if np.any(~active_query_endpoints) or np.any(~active_surface_endpoints):
            raise ValueError("Active contact routes require active node endpoints.")
        self.plan = plan
        self.query_nodes = query_nodes
        self.surface_nodes = surface
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-deformable-contact",
                "plan": plan.plan_id,
                "query_nodes": query_nodes.prepared_id,
                "surface_nodes": surface.prepared_id,
            }
        )

    @property
    def capacity(self) -> int:
        return self.plan.contact_capacity

    @property
    def ambient_dimension(self) -> int:
        return self.plan.ambient_dimension

    def _vectors(self, name: str, value: ArrayLike, capacity: int, /) -> Array:
        array = jnp.asarray(value, dtype=self.query_nodes.safe_masses.dtype)
        expected = (capacity, self.ambient_dimension)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}.")
        return array

    def evaluate(
        self,
        query_position: ArrayLike,
        query_velocity: ArrayLike,
        surface_position: ArrayLike,
        surface_velocity: ArrayLike,
        /,
    ) -> DeformableContactEvaluation:
        query_position_ = self._vectors(
            "query_position", query_position, self.query_nodes.capacity
        )
        query_velocity_ = self._vectors(
            "query_velocity", query_velocity, self.query_nodes.capacity
        )
        surface_position_ = self._vectors(
            "surface_position", surface_position, self.surface_nodes.capacity
        )
        surface_velocity_ = self._vectors(
            "surface_velocity", surface_velocity, self.surface_nodes.capacity
        )
        plan = self.plan
        safe_query = jnp.clip(plan.query_indices, 0, self.query_nodes.capacity - 1)
        safe_surface = jnp.clip(plan.surface_indices, 0, self.surface_nodes.capacity - 1)
        query_point = query_position_[safe_query]
        query_rate = query_velocity_[safe_query]
        surface_point = surface_position_[safe_surface]
        surface_rate = surface_velocity_[safe_surface]
        candidate_count = plan.query_indices.shape[0]
        dtype = query_point.dtype
        tolerance = plan.geometry_tolerance

        zero_weights = jnp.zeros((candidate_count, 3), dtype=dtype)
        plane_gap = jnp.sum(
            (query_point - plan.plane_points.astype(dtype))
            * plan.plane_normals.astype(dtype),
            axis=-1,
        )
        plane_witness = query_point - plane_gap[:, None] * plan.plane_normals.astype(
            dtype
        )
        plane_geometry_margin = jnp.full((candidate_count,), 1.0, dtype=dtype)

        node_displacement = query_point - surface_point[:, 0]
        node_distance = jnp.sqrt(jnp.sum(node_displacement * node_displacement, axis=-1))
        node_valid = node_distance > tolerance
        node_normal = (
            node_displacement / jnp.where(node_valid, node_distance, 1.0)[:, None]
        )
        node_weights = zero_weights.at[:, 0].set(1.0)

        if self.ambient_dimension == 2:
            segment_witness, segment_pair_weights, segment_valid, segment_feature, _ = (
                _closest_on_segment(
                    query_point,
                    surface_point[:, 0],
                    surface_point[:, 1],
                    tolerance,
                )
            )
            segment_weights = zero_weights.at[:, :2].set(segment_pair_weights)
            segment_edge = surface_point[:, 1] - surface_point[:, 0]
            segment_length = jnp.sqrt(jnp.sum(segment_edge * segment_edge, axis=-1))
            segment_face_normal = (
                jnp.stack((-segment_edge[:, 1], segment_edge[:, 0]), axis=-1)
                / jnp.where(segment_valid, segment_length, 1.0)[:, None]
            )
            segment_displacement = query_point - segment_witness
            segment_distance = jnp.sqrt(
                jnp.sum(segment_displacement * segment_displacement, axis=-1)
            )
            segment_plane_gap = jnp.sum(
                segment_displacement * segment_face_normal, axis=-1
            )
            segment_sign = jnp.where(segment_plane_gap >= 0.0, 1.0, -1.0)
            segment_normal = jnp.where(
                (segment_distance > tolerance)[:, None],
                segment_sign[:, None]
                * segment_displacement
                / jnp.where(segment_distance > tolerance, segment_distance, 1.0)[:, None],
                segment_face_normal,
            )
            segment_gap = segment_sign * segment_distance
            triangle_witness = jnp.zeros_like(query_point)
            triangle_weights = zero_weights
            triangle_normal = jnp.zeros_like(query_point)
            triangle_gap = jnp.zeros((candidate_count,), dtype=dtype)
            triangle_valid = jnp.zeros((candidate_count,), dtype=bool)
            triangle_feature = jnp.zeros((candidate_count,), dtype=dtype)
            triangle_geometry_margin = jnp.zeros((candidate_count,), dtype=dtype)
        else:
            (
                triangle_witness,
                triangle_weights,
                triangle_normal,
                triangle_gap,
                triangle_valid,
                triangle_feature,
            ) = _triangle_geometry(
                query_point,
                surface_point[:, 0],
                surface_point[:, 1],
                surface_point[:, 2],
                tolerance,
            )
            triangle_double_area = jnp.sqrt(
                jnp.sum(
                    jnp.cross(
                        surface_point[:, 1] - surface_point[:, 0],
                        surface_point[:, 2] - surface_point[:, 0],
                    )
                    ** 2,
                    axis=-1,
                )
            )
            triangle_geometry_margin = triangle_double_area - tolerance
            segment_witness = jnp.zeros_like(query_point)
            segment_weights = zero_weights
            segment_normal = jnp.zeros_like(query_point)
            segment_gap = jnp.zeros((candidate_count,), dtype=dtype)
            segment_valid = jnp.zeros((candidate_count,), dtype=bool)
            segment_feature = jnp.zeros((candidate_count,), dtype=dtype)
            segment_length = jnp.zeros((candidate_count,), dtype=dtype)

        kind = plan.route_kinds
        is_plane = kind == int(DeformableContactRouteKind.PLANE)
        is_node = kind == int(DeformableContactRouteKind.NODE)
        is_segment = kind == int(DeformableContactRouteKind.SEGMENT)
        is_triangle = kind == int(DeformableContactRouteKind.TRIANGLE)
        surface_weights = jnp.where(
            is_node[:, None],
            node_weights,
            jnp.where(
                is_segment[:, None],
                segment_weights,
                jnp.where(is_triangle[:, None], triangle_weights, zero_weights),
            ),
        )
        surface_witness = jnp.where(
            is_plane[:, None],
            plane_witness,
            jnp.where(
                is_node[:, None],
                surface_point[:, 0],
                jnp.where(is_segment[:, None], segment_witness, triangle_witness),
            ),
        )
        normal = jnp.where(
            is_plane[:, None],
            plan.plane_normals.astype(dtype),
            jnp.where(
                is_node[:, None],
                node_normal,
                jnp.where(is_segment[:, None], segment_normal, triangle_normal),
            ),
        )
        gap = jnp.where(
            is_plane,
            plane_gap,
            jnp.where(
                is_node, node_distance, jnp.where(is_segment, segment_gap, triangle_gap)
            ),
        )
        geometry_valid = jnp.where(
            is_plane,
            jnp.ones((candidate_count,), dtype=bool),
            jnp.where(
                is_node, node_valid, jnp.where(is_segment, segment_valid, triangle_valid)
            ),
        )
        geometry_margin = jnp.where(
            is_plane,
            plane_geometry_margin,
            jnp.where(
                is_node,
                node_distance - tolerance,
                jnp.where(
                    is_segment,
                    segment_length - tolerance,
                    triangle_geometry_margin,
                ),
            ),
        )
        feature_margin = jnp.where(
            is_plane | is_node,
            jnp.ones((candidate_count,), dtype=dtype),
            jnp.where(is_segment, segment_feature, triangle_feature),
        )
        surface_active = jnp.all(
            jnp.where(
                plan.surface_indices >= 0,
                self.surface_nodes.active_mask[safe_surface],
                True,
            ),
            axis=-1,
        )
        topology_valid = (
            plan.candidate_mask
            & self.query_nodes.active_mask[safe_query]
            & (is_plane | surface_active)
        )
        selected_surface_rate = jnp.sum(
            surface_weights[:, :, None] * surface_rate, axis=1
        )
        target_rate = jnp.where(
            is_plane[:, None], plan.plane_velocities.astype(dtype), selected_surface_rate
        )
        relative_velocity = query_rate - target_rate
        finite_candidate = (
            jnp.all(jnp.isfinite(query_point), axis=-1)
            & jnp.all(jnp.isfinite(query_rate), axis=-1)
            & jnp.all(jnp.isfinite(surface_witness), axis=-1)
            & jnp.all(jnp.isfinite(target_rate), axis=-1)
            & jnp.all(jnp.isfinite(normal), axis=-1)
            & jnp.isfinite(gap)
            & jnp.isfinite(geometry_margin)
            & jnp.isfinite(feature_margin)
        )
        candidate_finite = jnp.all((~topology_valid) | finite_candidate)
        geometry_successful = jnp.all((~topology_valid) | geometry_valid)
        activation_margin = jnp.asarray(plan.activation_distance, dtype=dtype) - gap
        candidate_valid = (
            topology_valid
            & geometry_valid
            & finite_candidate
            & (activation_margin >= 0.0)
        )
        actual_count = jnp.sum(candidate_valid, dtype=jnp.int32)
        overflow_count = jnp.maximum(actual_count - plan.contact_capacity, 0)
        overflow = overflow_count > 0
        selected = jax.lax.stop_gradient(
            jnp.nonzero(candidate_valid, size=plan.contact_capacity, fill_value=0)[0]
        )
        valid = jnp.arange(plan.contact_capacity, dtype=jnp.int32) < jnp.minimum(
            actual_count, plan.contact_capacity
        )
        selected_query = jnp.where(valid, plan.query_indices[selected], -1)
        selected_query_weights = jnp.where(valid, 1.0, 0.0).astype(dtype)
        selected_surface = jnp.where(valid[:, None], plan.surface_indices[selected], -1)
        selected_weights = jnp.where(valid[:, None], surface_weights[selected], 0.0)
        selected_kind = jnp.where(
            valid, plan.route_kinds[selected], int(DeformableContactRouteKind.PLANE)
        )
        selected_gap = jnp.where(valid, gap[selected], 0.0)
        selected_normal = jnp.where(valid[:, None], normal[selected], 0.0)
        selected_query_witness = jnp.where(valid[:, None], query_point[selected], 0.0)
        selected_surface_witness = jnp.where(
            valid[:, None], surface_witness[selected], 0.0
        )
        selected_relative_velocity = jnp.where(
            valid[:, None], relative_velocity[selected], 0.0
        )
        selected_plane_velocity = jnp.where(
            (valid & is_plane[selected])[:, None], target_rate[selected], 0.0
        )
        selected_validity_margin = jnp.where(
            valid,
            jnp.maximum(
                jnp.minimum(activation_margin[selected], geometry_margin[selected]), 0.0
            ),
            0.0,
        )
        selected_feature_margin = jnp.where(
            valid, jnp.maximum(feature_margin[selected], 0.0), 0.0
        )
        payload_finite = (
            jnp.all(jnp.isfinite(selected_gap))
            & jnp.all(jnp.isfinite(selected_normal))
            & jnp.all(jnp.isfinite(selected_query_witness))
            & jnp.all(jnp.isfinite(selected_surface_witness))
            & jnp.all(jnp.isfinite(selected_relative_velocity))
            & jnp.all(jnp.isfinite(selected_query_weights))
            & jnp.all(jnp.isfinite(selected_weights))
            & jnp.all(jnp.isfinite(selected_validity_margin))
            & jnp.all(jnp.isfinite(selected_feature_margin))
            & candidate_finite
        )
        successful = ~overflow & geometry_successful & payload_finite
        return DeformableContactEvaluation(
            selected_query,
            selected_query_weights,
            selected_surface,
            selected_weights,
            selected_plane_velocity,
            selected_kind,
            selected_gap,
            selected_normal,
            selected_query_witness,
            selected_surface_witness,
            selected_relative_velocity,
            jnp.where(valid, plan.route_keys[selected], 0),
            valid,
            selected_validity_margin,
            selected_feature_margin,
            actual_count,
            overflow_count,
            overflow,
            payload_finite,
            successful,
            self.prepared_id,
        )

    def _require_evaluation(self, evaluation: DeformableContactEvaluation, /) -> None:
        if not isinstance(evaluation, DeformableContactEvaluation):
            raise TypeError("evaluation must be DeformableContactEvaluation.")
        if evaluation.prepared_id != self.prepared_id:
            raise ValueError("evaluation belongs to a different prepared contact plan.")

    def interpolate(
        self,
        evaluation: DeformableContactEvaluation,
        query_values: ArrayLike,
        surface_values: ArrayLike,
        /,
        *,
        plane_values: ArrayLike | None = None,
    ) -> Array:
        """Apply the contact relative-interpolation map used by ``transpose``."""
        self._require_evaluation(evaluation)
        query = self._vectors("query_values", query_values, self.query_nodes.capacity)
        surface = self._vectors(
            "surface_values", surface_values, self.surface_nodes.capacity
        )
        plane = (
            evaluation.plane_velocity
            if plane_values is None
            else jnp.asarray(plane_values, dtype=query.dtype)
        )
        expected_plane_shape = (self.capacity, self.ambient_dimension)
        if plane.shape != expected_plane_shape:
            raise ValueError(f"plane_values must have shape {expected_plane_shape}.")
        safe_query = jnp.clip(evaluation.query_indices, 0, self.query_nodes.capacity - 1)
        safe_surface = jnp.clip(
            evaluation.surface_indices, 0, self.surface_nodes.capacity - 1
        )
        surface_interpolated = jnp.sum(
            evaluation.surface_weights[:, :, None] * surface[safe_surface], axis=1
        )
        is_plane = evaluation.route_kinds == int(DeformableContactRouteKind.PLANE)
        target = jnp.where(is_plane[:, None], plane, surface_interpolated)
        query_interpolated = evaluation.query_weights[:, None] * query[safe_query]
        return jnp.where(evaluation.valid[:, None], query_interpolated - target, 0.0)

    def transpose(
        self,
        evaluation: DeformableContactEvaluation,
        route_action: ArrayLike,
        /,
    ) -> DeformableContactTransposeResult:
        """Apply the exact algebraic transpose of relative interpolation."""
        self._require_evaluation(evaluation)
        action = jnp.asarray(route_action, dtype=self.query_nodes.safe_masses.dtype)
        expected = (self.capacity, self.ambient_dimension)
        if action.shape != expected:
            raise ValueError(f"route_action must have shape {expected}.")
        active_action = jnp.where(evaluation.valid[:, None], action, 0.0)
        safe_query = jnp.clip(evaluation.query_indices, 0, self.query_nodes.capacity - 1)
        safe_surface = jnp.clip(
            evaluation.surface_indices, 0, self.surface_nodes.capacity - 1
        )
        query_contribution = evaluation.query_weights[:, None] * active_action
        query_action = (
            jnp.zeros(
                (self.query_nodes.capacity, self.ambient_dimension), dtype=action.dtype
            )
            .at[safe_query]
            .add(query_contribution)
        )
        is_plane = evaluation.route_kinds == int(DeformableContactRouteKind.PLANE)
        surface_contribution = (
            -evaluation.surface_weights[:, :, None] * active_action[:, None, :]
        )
        surface_contribution = jnp.where(
            (~is_plane)[:, None, None], surface_contribution, 0.0
        )
        surface_action = (
            jnp.zeros(
                (self.surface_nodes.capacity, self.ambient_dimension), dtype=action.dtype
            )
            .at[safe_surface.reshape((-1,))]
            .add(surface_contribution.reshape((-1, self.ambient_dimension)))
        )
        plane_action = jnp.where(is_plane[:, None], -active_action, 0.0)
        balance_residual = (
            jnp.sum(query_action, axis=0)
            + jnp.sum(surface_action, axis=0)
            + jnp.sum(plane_action, axis=0)
        )
        balance_scale = jnp.maximum(
            1.0,
            jnp.linalg.norm(query_action)
            + jnp.linalg.norm(surface_action)
            + jnp.linalg.norm(plane_action),
        )
        balance_tolerance = (
            jnp.finfo(action.dtype).eps * max(32, 4 * self.capacity) * balance_scale
        )
        balance_valid = jnp.linalg.norm(balance_residual) <= balance_tolerance
        finite = (
            jnp.all(jnp.isfinite(query_action))
            & jnp.all(jnp.isfinite(surface_action))
            & jnp.all(jnp.isfinite(plane_action))
            & jnp.all(jnp.isfinite(balance_residual))
        )
        return DeformableContactTransposeResult(
            query_action,
            surface_action,
            plane_action,
            balance_residual,
            balance_valid,
            finite,
            evaluation.successful & finite & balance_valid,
            self.prepared_id,
        )


__all__ = [
    "DeformableContactEvaluation",
    "DeformableContactPlan",
    "DeformableContactRouteKind",
    "DeformableContactTransposeResult",
    "PreparedDeformableContact",
]

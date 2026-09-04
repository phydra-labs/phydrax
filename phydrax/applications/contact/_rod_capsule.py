#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ... import ein
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._distance import edge_edge_distance
from ...discretization.contact._guarantee import (
    ContactCapability,
    ContactGuaranteeLevel,
)
from ...discretization.contact._implicit_geometry import PlaneContactGeometry
from ...discretization.contact._participant import (
    AbstractContactParticipant,
    ParticipantTrajectoryBounds,
)
from ...discretization.contact._surface import (
    CollisionFeatureKind,
    CollisionFeaturePolicy,
    CollisionSurfacePlan,
    ContactPairPolicy,
)
from ...linalg import AbstractVectorSpace, ArraySpace
from ..solid_mechanics._rod_dynamics import (
    _quaternion_rotation_matrix,
    PreparedRod,
)
from ..solid_mechanics._rod_reduction import PreparedReducedRod


def _nonnegative_integer(value: int, name: str, /) -> int:
    values = np.asarray(value)
    if values.shape != () or not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must be one integer.")
    result = int(values)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _feature_ids(
    value: ArrayLike | None,
    count: int,
    default_start: int,
    name: str,
    /,
) -> np.ndarray:
    values = (
        np.arange(default_start, default_start + count, dtype=np.int64)
        if value is None
        else np.asarray(value)
    )
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must contain one integer per feature.")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0) or np.unique(values).size != count:
        raise ValueError(f"{name} must contain unique nonnegative values.")
    if np.max(values, initial=0) > 1_500_000_000:
        raise ValueError(f"{name} exceed the collision-free route-key range.")
    return values


def _segment_metric(
    value: ArrayLike | float,
    count: int,
    name: str,
    /,
    *,
    positive: bool,
) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError(f"{name} values must be real numeric data.")
    values = raw.astype(float, copy=False)
    if values.shape == () and not positive:
        values = np.full((count,), float(values), dtype=float)
    if values.shape != (count,):
        qualifier = (
            "one constant circular value per segment"
            if positive
            else "scalar or one value per segment"
        )
        raise ValueError(f"{name} must provide {qualifier}.")
    invalid_sign = values <= 0.0 if positive else values < 0.0
    if np.any(~np.isfinite(values)) or np.any(invalid_sign):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} values must be finite and {qualifier}.")
    return values


def _node_maximum(
    segment_values: np.ndarray, segments: np.ndarray, node_count: int, /
) -> np.ndarray:
    values = np.zeros((node_count,), dtype=segment_values.dtype)
    np.maximum.at(values, segments[:, 0], segment_values)
    np.maximum.at(values, segments[:, 1], segment_values)
    return values


def _canonical_edge_order(
    segments: np.ndarray, node_feature_ids: np.ndarray, /
) -> np.ndarray:
    canonical = np.sort(segments, axis=1)
    keys = np.sort(node_feature_ids[canonical], axis=1)
    return np.lexsort((keys[:, 1], keys[:, 0]))


class RodCapsuleGeometryPlan(StrictModule, NonTrainableState):
    """Exact constant-radius circular capsules around native rod segments.

    Radius data is deliberately one-dimensional: each native segment owns one
    finite, positive radius. Endpoint radii, elliptical axes, and nonzero proxy
    error are rejected rather than silently converted to centerline contact.
    """

    segment_radii: Array
    node_feature_ids: Array
    segment_feature_ids: Array
    segment_solver_clearance: Array
    participant_label: int = eqx.field(static=True)
    body_id: int = eqx.field(static=True)
    material_id: int = eqx.field(static=True)
    patch_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        segment_radii: ArrayLike,
        /,
        *,
        participant_id: int,
        body_id: int,
        material_id: int,
        patch_id: int,
        node_feature_ids: ArrayLike | None = None,
        segment_feature_ids: ArrayLike | None = None,
        solver_clearance: ArrayLike | float = 0.0,
        proxy_error: ArrayLike | float = 0.0,
    ):
        raw_radii = np.asarray(segment_radii)
        if raw_radii.ndim != 1 or raw_radii.size == 0:
            raise ValueError(
                "segment_radii must provide one constant circular radius per segment; "
                "tapered and noncircular data are unsupported."
            )
        count = int(raw_radii.size)
        radii = _segment_metric(
            raw_radii,
            count,
            "segment_radii",
            positive=True,
        )
        clearance = _segment_metric(
            solver_clearance,
            count,
            "solver_clearance",
            positive=False,
        )
        error = _segment_metric(
            proxy_error,
            count,
            "proxy_error",
            positive=False,
        )
        if np.any(error != 0.0):
            raise ValueError(
                "Exact rod capsules require zero proxy_error; an approximate proxy "
                "must use a separately qualified geometry."
            )
        node_ids = _feature_ids(
            node_feature_ids,
            count + 1,
            0,
            "node_feature_ids",
        )
        segment_ids = _feature_ids(
            segment_feature_ids,
            count,
            count + 1,
            "segment_feature_ids",
        )
        if np.intersect1d(node_ids, segment_ids).size:
            raise ValueError("Node and segment feature IDs must be globally distinct.")
        participant = _nonnegative_integer(participant_id, "participant_id")
        body = _nonnegative_integer(body_id, "body_id")
        material = _nonnegative_integer(material_id, "material_id")
        patch = _nonnegative_integer(patch_id, "patch_id")
        content = {
            "segment_radii": radii,
            "node_feature_ids": node_ids,
            "segment_feature_ids": segment_ids,
            "solver_clearance": clearance,
            "participant_id": participant,
            "body_id": body,
            "material_id": material,
            "patch_id": patch,
            "proxy_error": error,
        }
        self.segment_radii = jnp.asarray(radii)
        self.node_feature_ids = jnp.asarray(node_ids, dtype=jnp.int64)
        self.segment_feature_ids = jnp.asarray(segment_ids, dtype=jnp.int64)
        self.segment_solver_clearance = jnp.asarray(clearance)
        self.participant_label = participant
        self.body_id = body
        self.material_id = material
        self.patch_id = patch
        self.plan_id = canonical_fingerprint(
            {
                "kind": "exact-constant-circular-rod-capsule-geometry-plan",
                "content": array_tree_fingerprint(content),
            }
        )

    @property
    def segment_count(self) -> int:
        return int(self.segment_radii.size)

    def prepare(self, rod: PreparedRod, /) -> PreparedRodCapsuleGeometry:
        if not isinstance(rod, PreparedRod):
            raise TypeError("rod must be a PreparedRod.")
        if rod.plan.dimension != 3:
            raise ValueError("Rod capsule geometry requires a spatial PreparedRod.")
        if rod.plan.segment_count != self.segment_count:
            raise ValueError("Capsule radii must match the native rod segment count.")

        dtype = np.dtype(rod.plan.rest_positions.dtype)
        radii = np.asarray(self.segment_radii, dtype=dtype)
        clearance = np.asarray(self.segment_solver_clearance, dtype=dtype)
        segments = np.asarray(rod.plan.segment_node_ids, dtype=np.int64)
        node_ids = np.asarray(self.node_feature_ids, dtype=np.int64)
        segment_ids = np.asarray(self.segment_feature_ids, dtype=np.int64)
        edge_order = _canonical_edge_order(segments, node_ids)
        node_radii = _node_maximum(radii, segments, rod.plan.node_count)
        node_clearance = _node_maximum(clearance, segments, rod.plan.node_count)
        feature_ids = np.concatenate((node_ids, segment_ids[edge_order]))
        feature_kinds = np.concatenate(
            (
                np.full(
                    (rod.plan.node_count,),
                    int(CollisionFeatureKind.VERTEX),
                    dtype=np.int32,
                ),
                np.full(
                    (rod.plan.segment_count,),
                    int(CollisionFeatureKind.EDGE),
                    dtype=np.int32,
                ),
            )
        )
        feature_count = feature_ids.size
        feature_policy = CollisionFeaturePolicy(
            feature_ids,
            feature_kinds,
            participant_ids=np.full(
                (feature_count,), self.participant_label, dtype=np.int64
            ),
            body_ids=np.full((feature_count,), self.body_id, dtype=np.int64),
            material_ids=np.full((feature_count,), self.material_id, dtype=np.int64),
            patch_ids=np.full((feature_count,), self.patch_id, dtype=np.int64),
            physical_radius=np.concatenate((node_radii, radii[edge_order])),
            solver_clearance=np.concatenate((node_clearance, clearance[edge_order])),
            proxy_error=np.zeros((feature_count,), dtype=dtype),
            provenance_id=canonical_fingerprint(
                {
                    "kind": "exact-native-rod-capsule-features",
                    "plan": self.plan_id,
                    "rod": rod.prepared_id,
                }
            ),
        )
        pair_policy = ContactPairPolicy(
            rod.plan.node_count,
            excluded_vertex_pairs=segments,
        )
        surface = CollisionSurfacePlan(
            node_ids,
            ambient_dimension=3,
            edges=segments,
            feature_policy=feature_policy,
            pair_policy=pair_policy,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-exact-native-rod-capsule-geometry",
                "plan": self.plan_id,
                "rod": rod.prepared_id,
                "surface": surface.topology_id,
                "edge_order": array_tree_fingerprint(edge_order),
            }
        )
        return PreparedRodCapsuleGeometry(
            self,
            rod,
            surface,
            jnp.asarray(radii),
            jnp.asarray(clearance),
            jnp.asarray(edge_order, dtype=jnp.int32),
            prepared_id,
        )


class RodCapsulePlaneWitness(StrictModule):
    segment_indices: Array
    axial_coordinates: Array
    centerline_witness: Array
    capsule_witness: Array
    plane_witness: Array
    normal: Array
    signed_centerline_distance: Array
    gap: Array
    feature_margin: Array
    finite: Array
    valid: Array
    geometry_id: str = eqx.field(static=True)


class RodCapsulePairWitness(StrictModule):
    segment_pairs: Array
    left_axial_coordinates: Array
    right_axial_coordinates: Array
    left_centerline_witness: Array
    right_centerline_witness: Array
    left_capsule_witness: Array
    right_capsule_witness: Array
    normal: Array
    centerline_distance: Array
    gap: Array
    feature: Array
    feature_margin: Array
    adjacent: Array
    finite: Array
    valid: Array
    geometry_id: str = eqx.field(static=True)


class RodCapsuleDualityEvidence(StrictModule):
    surface_power: Array
    native_power: Array
    reduced_power: Array
    native_residual: Array
    reduced_residual: Array
    scale: Array
    finite: Array
    valid: Array
    participant_id: str = eqx.field(static=True)


class PreparedRodCapsuleGeometry(StrictModule, NonTrainableState):
    """Prepared exact capsule geometry tied to one native rod discretization."""

    plan: RodCapsuleGeometryPlan
    rod: PreparedRod
    surface_plan: CollisionSurfacePlan
    segment_radii: Array
    segment_solver_clearance: Array
    surface_edge_order: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: RodCapsuleGeometryPlan,
        rod: PreparedRod,
        surface_plan: CollisionSurfacePlan,
        segment_radii: Array,
        segment_solver_clearance: Array,
        surface_edge_order: Array,
        prepared_id: str,
        /,
    ):
        if not isinstance(plan, RodCapsuleGeometryPlan):
            raise TypeError("plan must be a RodCapsuleGeometryPlan.")
        if not isinstance(rod, PreparedRod) or rod.plan.dimension != 3:
            raise TypeError("rod must be a spatial PreparedRod.")
        if not isinstance(surface_plan, CollisionSurfacePlan):
            raise TypeError("surface_plan must be a CollisionSurfacePlan.")
        if segment_radii.shape != (rod.plan.segment_count,):
            raise ValueError("Prepared capsule radii changed shape.")
        identifier = str(prepared_id)
        if not identifier:
            raise ValueError("prepared_id must be nonempty.")
        self.plan = plan
        self.rod = rod
        self.surface_plan = surface_plan
        self.segment_radii = segment_radii
        self.segment_solver_clearance = segment_solver_clearance
        self.surface_edge_order = surface_edge_order
        self.prepared_id = identifier

    def _configuration(
        self, configuration: tuple[ArrayLike, ArrayLike], /
    ) -> tuple[Array, Array]:
        self.rod.configuration_schema.validate(configuration)
        return jnp.asarray(configuration[0]), jnp.asarray(configuration[1])

    def _segment_indices(self, segment_indices: ArrayLike, /) -> Array:
        indices = jnp.asarray(segment_indices)
        if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("segment_indices must be one integer vector.")
        indices = indices.astype(jnp.int32)
        return eqx.error_if(
            indices,
            jnp.any((indices < 0) | (indices >= self.rod.plan.segment_count)),
            "segment_indices index outside the prepared rod.",
        )

    def capsule_plane_witness(
        self,
        configuration: tuple[ArrayLike, ArrayLike],
        segment_indices: ArrayLike,
        plane: PlaneContactGeometry,
        /,
    ) -> RodCapsulePlaneWitness:
        """Return exact one-sided witnesses for capsules against ``n·x=offset``."""
        if not isinstance(plane, PlaneContactGeometry) or plane.unit_normal.shape != (3,):
            raise TypeError("plane must be a three-dimensional PlaneContactGeometry.")
        positions, _ = self._configuration(configuration)
        indices = self._segment_indices(segment_indices)
        safe = jnp.clip(indices, 0, self.rod.plan.segment_count - 1)
        nodes = self.rod.plan.segment_node_ids[safe]
        first = positions[nodes[:, 0]]
        second = positions[nodes[:, 1]]
        normal = jnp.asarray(plane.unit_normal, dtype=positions.dtype)
        first_distance = ein.contract("ci,i->c", first, normal) - plane.offset
        second_distance = ein.contract("ci,i->c", second, normal) - plane.offset
        choose_second = jax.lax.stop_gradient(second_distance < first_distance)
        axial = choose_second.astype(positions.dtype)
        centerline = jnp.where(choose_second[:, None], second, first)
        signed = jnp.where(choose_second, second_distance, first_distance)
        radius = self.segment_radii[safe].astype(positions.dtype)
        capsule = centerline - radius[:, None] * normal
        plane_witness = centerline - signed[:, None] * normal
        segment_lengths = jnp.sqrt(jnp.sum((second - first) ** 2, axis=-1))
        finite = (
            jnp.all(jnp.isfinite(centerline), axis=-1)
            & jnp.isfinite(signed)
            & jnp.isfinite(segment_lengths)
        )
        valid = finite & (segment_lengths > self.rod.plan.minimum_segment_length)
        return RodCapsulePlaneWitness(
            indices,
            axial,
            centerline,
            capsule,
            plane_witness,
            jnp.broadcast_to(normal, centerline.shape),
            signed,
            signed - radius,
            jnp.abs(first_distance - second_distance),
            finite,
            valid,
            self.prepared_id,
        )

    def capsule_capsule_witness(
        self,
        configuration: tuple[ArrayLike, ArrayLike],
        segment_pairs: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
    ) -> RodCapsulePairWitness:
        """Return exact witnesses for nonadjacent constant-radius capsules."""
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        positions, _ = self._configuration(configuration)
        pairs = jnp.asarray(segment_pairs)
        if (
            pairs.ndim != 2
            or pairs.shape[1:] != (2,)
            or not jnp.issubdtype(pairs.dtype, jnp.integer)
        ):
            raise TypeError("segment_pairs must be an integer (contacts, 2) array.")
        flat = self._segment_indices(pairs.reshape((-1,)))
        pairs = flat.reshape(pairs.shape)
        safe = jnp.clip(pairs, 0, self.rod.plan.segment_count - 1)
        left_nodes = self.rod.plan.segment_node_ids[safe[:, 0]]
        right_nodes = self.rod.plan.segment_node_ids[safe[:, 1]]
        left_first = positions[left_nodes[:, 0]]
        left_second = positions[left_nodes[:, 1]]
        right_first = positions[right_nodes[:, 0]]
        right_second = positions[right_nodes[:, 1]]
        distance = edge_edge_distance(
            left_first,
            left_second,
            right_first,
            right_second,
            tolerance=tolerance,
        )
        left_axis = left_second - left_first
        right_axis = right_second - right_first
        left_squared_length = jnp.sum(left_axis * left_axis, axis=-1)
        right_squared_length = jnp.sum(right_axis * right_axis, axis=-1)
        left_coordinate = jnp.clip(
            jnp.sum((distance.left_witness - left_first) * left_axis, axis=-1)
            / jnp.where(left_squared_length > 0.0, left_squared_length, 1.0),
            0.0,
            1.0,
        )
        right_coordinate = jnp.clip(
            jnp.sum((distance.right_witness - right_first) * right_axis, axis=-1)
            / jnp.where(right_squared_length > 0.0, right_squared_length, 1.0),
            0.0,
            1.0,
        )
        centerline_distance = jnp.sqrt(jnp.maximum(distance.squared_distance, 0.0))
        normal = distance.normal
        left_radius = self.segment_radii[safe[:, 0]].astype(positions.dtype)
        right_radius = self.segment_radii[safe[:, 1]].astype(positions.dtype)
        left_capsule = distance.left_witness - left_radius[:, None] * normal
        right_capsule = distance.right_witness + right_radius[:, None] * normal
        adjacent = jnp.any(left_nodes[:, :, None] == right_nodes[:, None, :], axis=(1, 2))
        finite = (
            distance.finite
            & jnp.isfinite(centerline_distance)
            & jnp.all(jnp.isfinite(left_capsule), axis=-1)
            & jnp.all(jnp.isfinite(right_capsule), axis=-1)
        )
        valid = distance.nondegenerate & finite & ~adjacent
        return RodCapsulePairWitness(
            pairs,
            left_coordinate,
            right_coordinate,
            distance.left_witness,
            distance.right_witness,
            left_capsule,
            right_capsule,
            normal,
            centerline_distance,
            centerline_distance - left_radius - right_radius,
            distance.feature,
            distance.feature_margin,
            adjacent,
            finite,
            valid,
            self.prepared_id,
        )

    def _surface_routes(
        self,
        configuration: tuple[ArrayLike, ArrayLike],
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        positions, orientations = self._configuration(configuration)
        indices = self._segment_indices(segment_indices)
        raw_coordinates = jnp.asarray(axial_coordinates)
        raw_offsets = jnp.asarray(surface_offsets)
        if (
            not jnp.issubdtype(raw_coordinates.dtype, jnp.number)
            or jnp.iscomplexobj(raw_coordinates)
            or not jnp.issubdtype(raw_offsets.dtype, jnp.number)
            or jnp.iscomplexobj(raw_offsets)
        ):
            raise TypeError("Surface route coordinates and offsets must be real.")
        coordinates = raw_coordinates.astype(positions.dtype)
        offsets = raw_offsets.astype(positions.dtype)
        if coordinates.shape != indices.shape or offsets.shape != (indices.size, 3):
            raise ValueError(
                "axial_coordinates and surface_offsets must have shapes (contacts,) "
                "and (contacts, 3)."
            )
        safe = jnp.clip(indices, 0, self.rod.plan.segment_count - 1)
        nodes = self.rod.plan.segment_node_ids[safe]
        axis = positions[nodes[:, 1]] - positions[nodes[:, 0]]
        axis_norm = jnp.sqrt(jnp.sum(axis * axis, axis=-1))
        radial_norm = jnp.sqrt(jnp.sum(offsets * offsets, axis=-1))
        radius = self.segment_radii[safe].astype(positions.dtype)
        scale = jnp.maximum(1.0, jnp.maximum(axis_norm, radius))
        metric_tolerance = 512.0 * jnp.finfo(positions.dtype).eps * scale
        coordinate_tolerance = 512.0 * jnp.finfo(positions.dtype).eps
        unit_axis = axis / jnp.where(axis_norm > 0.0, axis_norm, 1.0)[:, None]
        axial_offset = jnp.sum(offsets * unit_axis, axis=-1)
        at_first = coordinates <= coordinate_tolerance
        at_second = coordinates >= 1.0 - coordinate_tolerance
        on_boundary = jnp.where(
            at_first,
            axial_offset <= metric_tolerance,
            jnp.where(
                at_second,
                axial_offset >= -metric_tolerance,
                jnp.abs(axial_offset) <= metric_tolerance,
            ),
        )
        invalid = (
            ~jnp.isfinite(coordinates)
            | jnp.any(~jnp.isfinite(offsets), axis=-1)
            | (coordinates < 0.0)
            | (coordinates > 1.0)
            | (axis_norm <= self.rod.plan.minimum_segment_length)
            | (jnp.abs(radial_norm - radius) > metric_tolerance)
            | ~on_boundary
        )
        offsets = eqx.error_if(
            offsets,
            jnp.any(invalid),
            "Surface routes must lie on the exact circular capsule boundary; "
            "centerline, tapered, and noncircular routes are invalid.",
        )
        return positions, orientations, safe, nodes, coordinates, offsets

    def surface_positions(
        self,
        configuration: tuple[ArrayLike, ArrayLike],
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        /,
    ) -> Array:
        positions, _, _, nodes, coordinates, offsets = self._surface_routes(
            configuration,
            segment_indices,
            axial_coordinates,
            surface_offsets,
        )
        return (
            (1.0 - coordinates)[:, None] * positions[nodes[:, 0]]
            + coordinates[:, None] * positions[nodes[:, 1]]
            + offsets
        )

    def surface_velocity(
        self,
        configuration: tuple[ArrayLike, ArrayLike],
        native_velocity: tuple[ArrayLike, ArrayLike],
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        /,
    ) -> Array:
        """Evaluate translational plus material-spin velocity at witnesses."""
        _, orientations, safe, nodes, coordinates, offsets = self._surface_routes(
            configuration,
            segment_indices,
            axial_coordinates,
            surface_offsets,
        )
        linear, angular = self.rod.velocity_space.validate(native_velocity)
        rotation = _quaternion_rotation_matrix(orientations[safe])
        world_angular = ein.contract("cij,cj->ci", rotation, angular[safe])
        axis_velocity = (1.0 - coordinates)[:, None] * linear[nodes[:, 0]] + coordinates[
            :, None
        ] * linear[nodes[:, 1]]
        return axis_velocity + jnp.cross(world_angular, offsets)

    def native_effort_pullback(
        self,
        configuration: tuple[ArrayLike, ArrayLike],
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        surface_efforts: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        """Pull witness forces to world nodal forces and material-frame moments."""
        positions, orientations, safe, nodes, coordinates, offsets = self._surface_routes(
            configuration,
            segment_indices,
            axial_coordinates,
            surface_offsets,
        )
        raw_efforts = jnp.asarray(surface_efforts)
        if not jnp.issubdtype(raw_efforts.dtype, jnp.number) or jnp.iscomplexobj(
            raw_efforts
        ):
            raise TypeError("surface_efforts must be real.")
        efforts = raw_efforts.astype(positions.dtype)
        if efforts.shape != offsets.shape:
            raise ValueError("surface_efforts must match surface_offsets.")
        efforts = eqx.error_if(
            efforts,
            jnp.any(~jnp.isfinite(efforts)),
            "surface_efforts must be finite.",
        )
        forces = jnp.zeros_like(positions)
        forces = forces.at[nodes[:, 0]].add((1.0 - coordinates)[:, None] * efforts)
        forces = forces.at[nodes[:, 1]].add(coordinates[:, None] * efforts)
        rotation = _quaternion_rotation_matrix(orientations[safe])
        world_moments = jnp.cross(offsets, efforts)
        material_moments = ein.contract("cji,cj->ci", rotation, world_moments)
        moments = jnp.zeros((self.rod.plan.segment_count, 3), dtype=positions.dtype)
        moments = moments.at[safe].add(material_moments)
        return self.rod.effort_from_load(forces, moments)


class ReducedRodCapsuleContactParticipant(AbstractContactParticipant):
    """Reduced rod participant with exact capsule-specific witness actions."""

    reduced: PreparedReducedRod
    geometry: PreparedRodCapsuleGeometry
    contact_space: ArraySpace
    _capabilities: ContactCapability = eqx.field(static=True)
    _participant_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduced: PreparedReducedRod,
        geometry: PreparedRodCapsuleGeometry,
        /,
    ):
        if not isinstance(reduced, PreparedReducedRod):
            raise TypeError("reduced must be a PreparedReducedRod.")
        if not isinstance(geometry, PreparedRodCapsuleGeometry):
            raise TypeError("geometry must be a PreparedRodCapsuleGeometry.")
        if reduced.rod.prepared_id != geometry.rod.prepared_id:
            raise ValueError(
                "Reduced rod and capsule geometry must own the same PreparedRod."
            )
        dtype = np.dtype(reduced.rod.plan.rest_positions.dtype)
        contact_space = ArraySpace(
            (reduced.rod.plan.node_count, 3),
            dtype=dtype,
            space_id=canonical_fingerprint(
                {
                    "kind": "reduced-rod-capsule-centerline-contact-velocity-space",
                    "reduction": reduced.prepared_id,
                    "geometry": geometry.prepared_id,
                }
            ),
        )
        self.reduced = reduced
        self.geometry = geometry
        self.contact_space = contact_space
        self._capabilities = (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.EFFORT_PULLBACK
        )
        self._participant_id = canonical_fingerprint(
            {
                "kind": "reduced-native-rod-exact-capsule-contact-participant",
                "reduction": reduced.prepared_id,
                "geometry": geometry.prepared_id,
                "source_space": reduced.coefficient_space.space_id,
                "effort_space": reduced.reduced_effort_space.space_id,
            }
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.reduced.coefficient_space

    @property
    def tangent_space(self) -> AbstractVectorSpace:
        return self.reduced.coefficient_space

    @property
    def contact_velocity_space(self) -> ArraySpace:
        return self.contact_space

    @property
    def surface_plan(self) -> CollisionSurfacePlan:
        return self.geometry.surface_plan

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def capabilities(self) -> ContactCapability:
        return self._capabilities

    def positions(self, state: PyTree, /) -> Array:
        coefficients = self.source_space.validate(state)
        return self.contact_velocity_space.validate(
            self.reduced.lift_configuration(coefficients)[0]
        )

    def velocities(self, state: PyTree, rates: PyTree, /) -> Array:
        coefficients = self.source_space.validate(state)
        rates_ = self.tangent_space.validate(rates)
        native = self.reduced.lift_velocity_operator(coefficients).mv(rates_)
        return self.contact_velocity_space.validate(native[0])

    def effort_pullback(self, state: PyTree, surface_effort: ArrayLike, /) -> Array:
        coefficients = self.source_space.validate(state)
        effort = self.contact_effort_space.validate(surface_effort)
        moments = jnp.zeros(
            (self.reduced.rod.plan.segment_count, 3),
            dtype=effort.dtype,
        )
        return self.effort_space.validate(
            self.reduced.pullback_loads(coefficients, effort, moments)
        )

    def trajectory_bounds(
        self, start_state: PyTree, end_state: PyTree, /
    ) -> ParticipantTrajectoryBounds:
        start = self.positions(start_state)
        end = self.positions(end_state)
        lower = jnp.minimum(start, end)
        upper = jnp.maximum(start, end)
        finite = jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper))
        return ParticipantTrajectoryBounds(
            lower,
            upper,
            jnp.asarray(int(ContactGuaranteeLevel.HEURISTIC), dtype=jnp.int32),
            finite,
            finite,
            self.participant_id,
        )

    def capsule_plane_witness(
        self,
        coefficients: ArrayLike,
        segment_indices: ArrayLike,
        plane: PlaneContactGeometry,
        /,
    ) -> RodCapsulePlaneWitness:
        coefficients_ = self.source_space.validate(coefficients)
        return self.geometry.capsule_plane_witness(
            self.reduced.lift_configuration(coefficients_),
            segment_indices,
            plane,
        )

    def capsule_capsule_witness(
        self,
        coefficients: ArrayLike,
        segment_pairs: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
    ) -> RodCapsulePairWitness:
        coefficients_ = self.source_space.validate(coefficients)
        return self.geometry.capsule_capsule_witness(
            self.reduced.lift_configuration(coefficients_),
            segment_pairs,
            tolerance=tolerance,
        )

    def surface_velocity(
        self,
        coefficients: ArrayLike,
        rates: ArrayLike,
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        /,
    ) -> Array:
        coefficients_ = self.source_space.validate(coefficients)
        rates_ = self.tangent_space.validate(rates)
        configuration = self.reduced.lift_configuration(coefficients_)
        native_velocity = self.reduced.lift_velocity_operator(coefficients_).mv(rates_)
        return self.geometry.surface_velocity(
            configuration,
            native_velocity,
            segment_indices,
            axial_coordinates,
            surface_offsets,
        )

    def surface_effort_pullback(
        self,
        coefficients: ArrayLike,
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        surface_efforts: ArrayLike,
        /,
    ) -> Array:
        coefficients_ = self.source_space.validate(coefficients)
        configuration = self.reduced.lift_configuration(coefficients_)
        native_effort = self.geometry.native_effort_pullback(
            configuration,
            segment_indices,
            axial_coordinates,
            surface_offsets,
            surface_efforts,
        )
        return self.effort_space.validate(
            self.reduced.pullback_loads(
                coefficients_,
                native_effort[0],
                native_effort[1],
            )
        )

    def surface_duality_evidence(
        self,
        coefficients: ArrayLike,
        rates: ArrayLike,
        segment_indices: ArrayLike,
        axial_coordinates: ArrayLike,
        surface_offsets: ArrayLike,
        surface_efforts: ArrayLike,
        /,
    ) -> RodCapsuleDualityEvidence:
        coefficients_ = self.source_space.validate(coefficients)
        rates_ = self.tangent_space.validate(rates)
        configuration = self.reduced.lift_configuration(coefficients_)
        native_velocity = self.reduced.lift_velocity_operator(coefficients_).mv(rates_)
        velocity = self.geometry.surface_velocity(
            configuration,
            native_velocity,
            segment_indices,
            axial_coordinates,
            surface_offsets,
        )
        efforts = jnp.asarray(surface_efforts, dtype=velocity.dtype)
        native_effort = self.geometry.native_effort_pullback(
            configuration,
            segment_indices,
            axial_coordinates,
            surface_offsets,
            efforts,
        )
        reduced_effort = self.reduced.pullback_loads(
            coefficients_, native_effort[0], native_effort[1]
        )
        surface_power = jnp.sum(efforts * velocity).real
        native_power = self.reduced.native_effort_space.pair(
            native_effort, native_velocity
        ).real
        reduced_power = self.reduced.reduced_effort_space.pair(
            reduced_effort, rates_
        ).real
        native_residual = surface_power - native_power
        reduced_residual = native_power - reduced_power
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(surface_power),
                jnp.maximum(jnp.abs(native_power), jnp.abs(reduced_power)),
            ),
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        surface_power,
                        native_power,
                        reduced_power,
                        native_residual,
                        reduced_residual,
                        scale,
                    )
                )
            )
        )
        tolerance = jnp.finfo(velocity.dtype).eps * max(64, 16 * velocity.shape[0])
        valid = (
            finite
            & (jnp.abs(native_residual) <= tolerance * scale)
            & (jnp.abs(reduced_residual) <= tolerance * scale)
        )
        return RodCapsuleDualityEvidence(
            surface_power,
            native_power,
            reduced_power,
            native_residual,
            reduced_residual,
            scale,
            finite,
            valid,
            self.participant_id,
        )


def prepare_reduced_rod_contact_participant(
    reduced: PreparedReducedRod,
    geometry: PreparedRodCapsuleGeometry,
    /,
) -> ReducedRodCapsuleContactParticipant:
    """Bind exact capsule geometry to the reduction's native mechanics map."""
    return ReducedRodCapsuleContactParticipant(reduced, geometry)


__all__ = [
    "PreparedRodCapsuleGeometry",
    "ReducedRodCapsuleContactParticipant",
    "RodCapsuleDualityEvidence",
    "RodCapsuleGeometryPlan",
    "RodCapsulePairWitness",
    "RodCapsulePlaneWitness",
    "prepare_reduced_rod_contact_participant",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...combinatorial import (
    BipartiteAssignmentSpace,
    BranchAndBoundSetPacking,
    CombinatorialCertification,
    CombinatorialStatus,
    GreedySetPacking,
    HungarianAssignment,
    LinearCombinatorialProblem,
    SetPackingSpace,
    solve_combinatorial,
)
from ..camera import CameraRig, pixels_to_rays, triangulate_weighted_rays
from ._types import AssociationEvidence, AssociationStatus, ParticleDetections


class TwoViewAssociationPlan(StrictModule, NonTrainableState):
    """Geometric, radiometric, ambiguity, and dummy-match policy."""

    maximum_ray_distance: float = eqx.field(static=True)
    parallel_tolerance: float = eqx.field(static=True)
    covariance_scale: float = eqx.field(static=True)
    intensity_weight: float = eqx.field(static=True)
    unmatched_cost: float = eqx.field(static=True)
    ambiguity_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_ray_distance: float,
        parallel_tolerance: float = 1e-10,
        covariance_scale: float = 1.0,
        intensity_weight: float = 0.0,
        unmatched_cost: float = 4.0,
        ambiguity_margin: float = 0.25,
    ):
        values = jnp.asarray(
            (
                maximum_ray_distance,
                parallel_tolerance,
                covariance_scale,
                intensity_weight,
                unmatched_cost,
                ambiguity_margin,
            )
        )
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Association plan values must be finite.")
        if maximum_ray_distance <= 0.0 or parallel_tolerance <= 0.0:
            raise ValueError("Ray-distance tolerances must be positive.")
        if (
            min(covariance_scale, intensity_weight, unmatched_cost, ambiguity_margin)
            < 0.0
        ):
            raise ValueError("Association cost weights must be nonnegative.")
        self.maximum_ray_distance = float(maximum_ray_distance)
        self.parallel_tolerance = float(parallel_tolerance)
        self.covariance_scale = float(covariance_scale)
        self.intensity_weight = float(intensity_weight)
        self.unmatched_cost = float(unmatched_cost)
        self.ambiguity_margin = float(ambiguity_margin)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-view-particle-association",
                "maximum_ray_distance": self.maximum_ray_distance,
                "parallel_tolerance": self.parallel_tolerance,
                "covariance_scale": self.covariance_scale,
                "intensity_weight": self.intensity_weight,
                "unmatched_cost": self.unmatched_cost,
                "ambiguity_margin": self.ambiguity_margin,
            }
        )


class TwoViewAssociationResult(StrictModule):
    """One-to-one matches with explicit dummies and ambiguity evidence."""

    matches_a_to_b: Array
    matches_b_to_a: Array
    matched_a: Array
    unmatched_a: Array
    unmatched_b: Array
    ambiguous_a: Array
    pair_cost: Array
    pair_valid: Array
    ray_distance: Array
    status: Array
    valid: Array
    evidence: AssociationEvidence
    assignment: object
    association_id: str = eqx.field(static=True)


class MultiViewAssociationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity N-camera candidate and set-packing policy."""

    camera_capacity: int = eqx.field(static=True)
    candidate_capacity: int = eqx.field(static=True)
    selected_capacity: int = eqx.field(static=True)
    min_views: int = eqx.field(static=True)
    maximum_ray_distance: float = eqx.field(static=True)
    parallel_tolerance: float = eqx.field(static=True)
    view_reward: float = eqx.field(static=True)
    exact_candidate_limit: int = eqx.field(static=True)
    maximum_nodes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        camera_capacity: int,
        candidate_capacity: int,
        selected_capacity: int,
        /,
        *,
        min_views: int = 2,
        maximum_ray_distance: float,
        parallel_tolerance: float = 1e-10,
        view_reward: float = 2.0,
        exact_candidate_limit: int = 32,
        maximum_nodes: int = 1_000_000,
    ):
        for name, value in (
            ("camera_capacity", camera_capacity),
            ("candidate_capacity", candidate_capacity),
            ("selected_capacity", selected_capacity),
            ("min_views", min_views),
            ("exact_candidate_limit", exact_candidate_limit),
            ("maximum_nodes", maximum_nodes),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if camera_capacity < 2 or candidate_capacity <= 0 or selected_capacity <= 0:
            raise ValueError("Camera, candidate, and selected capacities are invalid.")
        if not 2 <= min_views <= camera_capacity:
            raise ValueError("min_views must lie between two and camera_capacity.")
        if selected_capacity > candidate_capacity:
            raise ValueError("selected_capacity cannot exceed candidate_capacity.")
        if exact_candidate_limit <= 0 or maximum_nodes <= 0:
            raise ValueError("Set-packing limits must be positive.")
        if maximum_ray_distance <= 0.0 or parallel_tolerance <= 0.0:
            raise ValueError("Ray-distance tolerances must be positive.")
        if view_reward <= 0.0:
            raise ValueError("view_reward must be positive.")
        self.camera_capacity = int(camera_capacity)
        self.candidate_capacity = int(candidate_capacity)
        self.selected_capacity = int(selected_capacity)
        self.min_views = int(min_views)
        self.maximum_ray_distance = float(maximum_ray_distance)
        self.parallel_tolerance = float(parallel_tolerance)
        self.view_reward = float(view_reward)
        self.exact_candidate_limit = int(exact_candidate_limit)
        self.maximum_nodes = int(maximum_nodes)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiview-particle-association",
                "camera_capacity": self.camera_capacity,
                "candidate_capacity": self.candidate_capacity,
                "selected_capacity": self.selected_capacity,
                "min_views": self.min_views,
                "maximum_ray_distance": self.maximum_ray_distance,
                "parallel_tolerance": self.parallel_tolerance,
                "view_reward": self.view_reward,
                "exact_candidate_limit": self.exact_candidate_limit,
                "maximum_nodes": self.maximum_nodes,
            }
        )


class MultiViewAssociationResult(StrictModule):
    """Selected conflict-free camera tuples and their native triangulation evidence."""

    detection_indices: Array
    candidate_score: Array
    valid: Array
    triangulation: object
    status: Array


class _TupleCandidates(StrictModule):
    detection_indices: Array
    scores: Array
    valid: Array
    incidence: Array
    ray_origins: Array
    ray_directions: Array
    ray_valid: Array
    ray_weights: Array
    candidate_count: Array
    overflow_count: Array
    conflict_count: Array


def _capacity(detections: ParticleDetections, /) -> int:
    if detections.positions_rc.ndim != 2 or detections.positions_rc.shape[-1] != 2:
        raise ValueError("ParticleDetections.positions_rc must have shape (capacity, 2).")
    capacity = int(detections.positions_rc.shape[0])
    if detections.covariance_rc.shape != (capacity, 2, 2):
        raise ValueError("ParticleDetections.covariance_rc has an incompatible shape.")
    for name, value in (
        ("intensity", detections.intensity),
        ("radius", detections.radius),
        ("valid", detections.valid),
        ("status", detections.status),
    ):
        if value.shape != (capacity,):
            raise ValueError(f"ParticleDetections.{name} must have shape (capacity,).")
    return capacity


def _ray_distance(
    origins_a: Array,
    directions_a: Array,
    origins_b: Array,
    directions_b: Array,
    parallel_tolerance: float,
    /,
) -> tuple[Array, Array]:
    first_norm = jnp.sqrt(jnp.sum(directions_a * directions_a, axis=-1, keepdims=True))
    second_norm = jnp.sqrt(jnp.sum(directions_b * directions_b, axis=-1, keepdims=True))
    tiny = jnp.finfo(jnp.result_type(directions_a, directions_b)).tiny
    first = directions_a / jnp.maximum(first_norm, tiny)
    second = directions_b / jnp.maximum(second_norm, tiny)
    cross_direction = jnp.cross(first[:, None, :], second[None, :, :])
    cross_norm = jnp.sqrt(jnp.sum(cross_direction * cross_direction, axis=-1))
    displacement = origins_b[None, :, :] - origins_a[:, None, :]
    numerator = jnp.abs(contract("ijk,ijk->ij", displacement, cross_direction))
    distance = numerator / jnp.maximum(cross_norm, parallel_tolerance)
    finite = (
        jnp.all(jnp.isfinite(origins_a), axis=-1)[:, None]
        & jnp.all(jnp.isfinite(origins_b), axis=-1)[None, :]
        & jnp.all(jnp.isfinite(directions_a), axis=-1)[:, None]
        & jnp.all(jnp.isfinite(directions_b), axis=-1)[None, :]
    )
    return distance, finite & (cross_norm > parallel_tolerance)


def associate_two_view(
    detections_a: ParticleDetections,
    detections_b: ParticleDetections,
    origins_a: ArrayLike,
    directions_a: ArrayLike,
    origins_b: ArrayLike,
    directions_b: ArrayLike,
    plan: TwoViewAssociationPlan,
    /,
) -> TwoViewAssociationResult:
    """Associate two views through a square Hungarian problem with explicit dummies."""
    if not isinstance(detections_a, ParticleDetections) or not isinstance(
        detections_b, ParticleDetections
    ):
        raise TypeError("Both detection sets must be ParticleDetections.")
    da, db = _capacity(detections_a), _capacity(detections_b)
    valid_a = jnp.asarray(detections_a.valid, dtype=bool)
    valid_b = jnp.asarray(detections_b.valid, dtype=bool)
    oa, ra = jnp.asarray(origins_a), jnp.asarray(directions_a)
    ob, rb = jnp.asarray(origins_b), jnp.asarray(directions_b)
    if oa.shape != (da, 3) or ra.shape != (da, 3):
        raise ValueError("View-a rays must have shape (capacity_a, 3).")
    if ob.shape != (db, 3) or rb.shape != (db, 3):
        raise ValueError("View-b rays must have shape (capacity_b, 3).")
    dtype = jnp.result_type(oa, ra, ob, rb, float)
    oa, ra, ob, rb = (jnp.asarray(value, dtype=dtype) for value in (oa, ra, ob, rb))
    distance, geometry_valid = _ray_distance(oa, ra, ob, rb, plan.parallel_tolerance)
    variance = 1.0 + plan.covariance_scale * (
        jnp.trace(detections_a.covariance_rc, axis1=-2, axis2=-1)[:, None]
        + jnp.trace(detections_b.covariance_rc, axis1=-2, axis2=-1)[None, :]
    )
    tiny = jnp.finfo(dtype).tiny
    intensity_delta = (
        jnp.log(jnp.maximum(detections_a.intensity, tiny))[:, None]
        - jnp.log(jnp.maximum(detections_b.intensity, tiny))[None, :]
    )
    pair_cost = (
        distance / plan.maximum_ray_distance
    ) ** 2 / variance + plan.intensity_weight * intensity_delta**2
    pair_valid = (
        valid_a[:, None]
        & valid_b[None, :]
        & geometry_valid
        & (distance <= plan.maximum_ray_distance)
        & jnp.isfinite(pair_cost)
    )
    dimension = da + db
    costs = jnp.zeros((dimension, dimension), dtype=dtype)
    allowed = jnp.zeros((dimension, dimension), dtype=bool)
    costs = costs.at[:da, :db].set(jnp.where(pair_valid, pair_cost, 0.0))
    allowed = allowed.at[:da, :db].set(pair_valid)
    a_index = jnp.arange(da, dtype=jnp.int32)
    b_index = jnp.arange(db, dtype=jnp.int32)
    costs = costs.at[a_index, db + a_index].set(
        jnp.where(valid_a, plan.unmatched_cost, 0.0)
    )
    costs = costs.at[da + b_index, b_index].set(
        jnp.where(valid_b, plan.unmatched_cost, 0.0)
    )
    allowed = allowed.at[a_index, db + a_index].set(True)
    allowed = allowed.at[da + b_index, b_index].set(True)
    allowed = allowed.at[da:, db:].set(True)
    problem = LinearCombinatorialProblem(
        BipartiteAssignmentSpace(dimension, dimension, valid=allowed),
        costs,
        problem_id=f"two-view:{detections_a.detection_id}:{detections_b.detection_id}:{plan.plan_id}",
    )
    method = HungarianAssignment(maximum_dimension=dimension)
    assignment = method.solve(problem, method.plan(problem, CombinatorialCertification()))
    columns = assignment.decision.columns[:da]
    matched_a = valid_a & (columns >= 0) & (columns < db)
    matches_a_to_b = jnp.where(matched_a, columns, -1).astype(jnp.int32)
    matches_b_to_a = jnp.full((db,), -1, dtype=jnp.int32)
    for index in range(da):
        safe_column = jnp.clip(matches_a_to_b[index], 0, db - 1)
        matches_b_to_a = jax.lax.cond(
            matched_a[index],
            lambda value, column=safe_column, row=index: value.at[column].set(row),
            lambda value: value,
            matches_b_to_a,
        )
    unmatched_a = valid_a & ~matched_a
    unmatched_b = valid_b & (matches_b_to_a < 0)
    ordered = jnp.sort(jnp.where(pair_valid, pair_cost, jnp.inf), axis=-1)
    ambiguous = (
        matched_a
        & jnp.isfinite(ordered[:, 1])
        & ((ordered[:, 1] - ordered[:, 0]) <= plan.ambiguity_margin)
        if db > 1
        else jnp.zeros((da,), dtype=bool)
    )
    status = jnp.where(
        assignment.valid,
        int(AssociationStatus.SUCCESS),
        int(AssociationStatus.INFEASIBLE),
    ).astype(jnp.int32)
    evidence = AssociationEvidence(
        gated_pair_count=jnp.sum(pair_valid, dtype=jnp.int32),
        matched_count=jnp.sum(matched_a, dtype=jnp.int32),
        unmatched_a_count=jnp.sum(unmatched_a, dtype=jnp.int32),
        unmatched_b_count=jnp.sum(unmatched_b, dtype=jnp.int32),
        ambiguous_a_count=jnp.sum(ambiguous, dtype=jnp.int32),
        assignment_status=assignment.status,
        optimality_proven=assignment.certificate.optimality_proven,
        plan_id=plan.plan_id,
    )
    association_id = "association:" + canonical_fingerprint(
        {
            "a": detections_a.detection_id,
            "b": detections_b.detection_id,
            "plan": plan.plan_id,
        }
    )
    return TwoViewAssociationResult(
        matches_a_to_b,
        matches_b_to_a,
        matched_a,
        unmatched_a,
        unmatched_b,
        ambiguous,
        pair_cost,
        pair_valid,
        distance,
        status,
        assignment.valid,
        evidence,
        assignment,
        association_id,
    )


def _camera_rays(
    detections: tuple[ParticleDetections, ...], rig: CameraRig, /
) -> tuple[Array, Array, Array, Array]:
    origins, directions, valid, weights = [], [], [], []
    for camera_index, camera in enumerate(rig.cameras):
        result = pixels_to_rays(camera, detections[camera_index].positions_rc)
        detected = (
            jnp.asarray(detections[camera_index].valid, dtype=bool)
            & rig.camera_valid[camera_index]
        )
        ray_valid = detected & result.valid
        trace = jnp.trace(detections[camera_index].covariance_rc, axis1=-2, axis2=-1)
        precision = detections[camera_index].intensity / jnp.maximum(trace, 1e-12)
        origins.append(result.origins)
        directions.append(result.directions)
        valid.append(ray_valid)
        weights.append(jnp.where(ray_valid, precision, 0.0))
    return tuple(
        jnp.stack(values, axis=0) for values in (origins, directions, valid, weights)
    )


def _enumerate_candidates(
    detections: tuple[ParticleDetections, ...],
    ray_origins: Array,
    ray_directions: Array,
    ray_valid: Array,
    ray_weights: Array,
    plan: MultiViewAssociationPlan,
    /,
) -> _TupleCandidates:
    camera_count = plan.camera_capacity
    detection_capacity = _capacity(detections[0])
    grid_a = jnp.repeat(
        jnp.arange(detection_capacity, dtype=jnp.int32), detection_capacity
    )
    grid_b = jnp.tile(jnp.arange(detection_capacity, dtype=jnp.int32), detection_capacity)
    index_pool, valid_pool, score_pool, origin_pool = [], [], [], []
    direction_pool, ray_valid_pool, weight_pool = [], [], []
    for first_camera in range(camera_count - 1):
        for second_camera in range(first_camera + 1, camera_count):
            distance, geometric = _ray_distance(
                ray_origins[first_camera],
                ray_directions[first_camera],
                ray_origins[second_camera],
                ray_directions[second_camera],
                plan.parallel_tolerance,
            )
            base_distance = distance.reshape((-1,))
            base_valid = (
                ray_valid[first_camera, grid_a]
                & ray_valid[second_camera, grid_b]
                & geometric.reshape((-1,))
                & (base_distance <= plan.maximum_ray_distance)
            )
            count = detection_capacity * detection_capacity
            indices = jnp.full((count, camera_count), -1, dtype=jnp.int32)
            indices = indices.at[:, first_camera].set(grid_a)
            indices = indices.at[:, second_camera].set(grid_b)
            origins = jnp.zeros((count, camera_count, 3), dtype=ray_origins.dtype)
            directions = jnp.zeros_like(origins)
            valid = jnp.zeros((count, camera_count), dtype=bool)
            weights = jnp.zeros((count, camera_count), dtype=ray_weights.dtype)
            origins = origins.at[:, first_camera].set(ray_origins[first_camera, grid_a])
            origins = origins.at[:, second_camera].set(ray_origins[second_camera, grid_b])
            directions = directions.at[:, first_camera].set(
                ray_directions[first_camera, grid_a]
            )
            directions = directions.at[:, second_camera].set(
                ray_directions[second_camera, grid_b]
            )
            valid = valid.at[:, first_camera].set(base_valid)
            valid = valid.at[:, second_camera].set(base_valid)
            weights = weights.at[:, first_camera].set(ray_weights[first_camera, grid_a])
            weights = weights.at[:, second_camera].set(ray_weights[second_camera, grid_b])
            residual = base_distance
            view_count = jnp.full((count,), 2, dtype=jnp.int32)
            for camera in range(camera_count):
                if camera in (first_camera, second_camera):
                    continue
                first_distance, first_valid = _ray_distance(
                    ray_origins[first_camera, grid_a],
                    ray_directions[first_camera, grid_a],
                    ray_origins[camera],
                    ray_directions[camera],
                    plan.parallel_tolerance,
                )
                second_distance, second_valid = _ray_distance(
                    ray_origins[second_camera, grid_b],
                    ray_directions[second_camera, grid_b],
                    ray_origins[camera],
                    ray_directions[camera],
                    plan.parallel_tolerance,
                )
                consistency = jnp.maximum(first_distance, second_distance)
                admissible = (
                    first_valid
                    & second_valid
                    & ray_valid[camera][None, :]
                    & (consistency <= plan.maximum_ray_distance)
                )
                masked = jnp.where(admissible, consistency, jnp.inf)
                chosen = jnp.argmin(masked, axis=-1).astype(jnp.int32)
                chosen_distance = jnp.take_along_axis(masked, chosen[:, None], axis=-1)[
                    :, 0
                ]
                chosen_valid = jnp.isfinite(chosen_distance) & base_valid
                indices = indices.at[:, camera].set(jnp.where(chosen_valid, chosen, -1))
                origins = origins.at[:, camera].set(ray_origins[camera, chosen])
                directions = directions.at[:, camera].set(ray_directions[camera, chosen])
                valid = valid.at[:, camera].set(chosen_valid)
                weights = weights.at[:, camera].set(
                    jnp.where(chosen_valid, ray_weights[camera, chosen], 0.0)
                )
                residual = residual + jnp.where(chosen_valid, chosen_distance, 0.0)
                view_count = view_count + chosen_valid.astype(jnp.int32)
            candidate_valid = base_valid & (view_count >= plan.min_views)
            score = (
                plan.view_reward * view_count.astype(ray_origins.dtype)
                - residual / plan.maximum_ray_distance
            )
            index_pool.append(indices)
            valid_pool.append(candidate_valid)
            score_pool.append(score)
            origin_pool.append(origins)
            direction_pool.append(directions)
            ray_valid_pool.append(valid & candidate_valid[:, None])
            weight_pool.append(weights)
    indices = jnp.concatenate(index_pool, axis=0)
    valid = jnp.concatenate(valid_pool, axis=0)
    scores = jnp.concatenate(score_pool, axis=0)
    origins = jnp.concatenate(origin_pool, axis=0)
    directions = jnp.concatenate(direction_pool, axis=0)
    candidate_ray_valid = jnp.concatenate(ray_valid_pool, axis=0)
    candidate_weights = jnp.concatenate(weight_pool, axis=0)
    candidate_count = jnp.sum(valid, dtype=jnp.int32)
    take = min(plan.candidate_capacity, int(scores.shape[0]))
    selected_scores, selected_indices = jax.lax.top_k(
        jnp.where(valid, scores, -jnp.inf), take
    )
    arrays = (
        indices[selected_indices],
        origins[selected_indices],
        directions[selected_indices],
        candidate_ray_valid[selected_indices],
        candidate_weights[selected_indices],
    )
    selected_valid = jnp.isfinite(selected_scores)
    if take < plan.candidate_capacity:
        padding = plan.candidate_capacity - take
        arrays = (
            jnp.concatenate(
                (arrays[0], jnp.full((padding, camera_count), -1, dtype=jnp.int32))
            ),
            jnp.concatenate(
                (arrays[1], jnp.zeros((padding, camera_count, 3), dtype=origins.dtype))
            ),
            jnp.concatenate(
                (arrays[2], jnp.zeros((padding, camera_count, 3), dtype=directions.dtype))
            ),
            jnp.concatenate((arrays[3], jnp.zeros((padding, camera_count), dtype=bool))),
            jnp.concatenate(
                (
                    arrays[4],
                    jnp.zeros((padding, camera_count), dtype=candidate_weights.dtype),
                )
            ),
        )
        selected_scores = jnp.concatenate(
            (selected_scores, jnp.zeros((padding,), dtype=scores.dtype))
        )
        selected_valid = jnp.concatenate(
            (selected_valid, jnp.zeros((padding,), dtype=bool))
        )
    (
        selected_detection_indices,
        origins,
        directions,
        candidate_ray_valid,
        candidate_weights,
    ) = arrays
    selected_detection_indices = jnp.where(
        selected_valid[:, None], selected_detection_indices, -1
    )
    selected_scores = jnp.where(selected_valid, selected_scores, 0.0)
    incidence = jnp.zeros(
        (plan.candidate_capacity, camera_count * detection_capacity), dtype=bool
    )
    for camera in range(camera_count):
        detection_index = selected_detection_indices[:, camera]
        resource = camera * detection_capacity + jnp.clip(
            detection_index, 0, detection_capacity - 1
        )
        incidence = incidence | (
            jax.nn.one_hot(resource, camera_count * detection_capacity, dtype=bool)
            & (detection_index >= 0)[:, None]
        )
    conflict = (
        contract("kr,lr->kl", incidence.astype(jnp.int32), incidence.astype(jnp.int32))
        > 0
    )
    conflict = conflict & ~jnp.eye(plan.candidate_capacity, dtype=bool)
    conflict_count = jnp.sum(
        jnp.triu(conflict & selected_valid[:, None] & selected_valid[None, :], k=1),
        dtype=jnp.int32,
    )
    return _TupleCandidates(
        selected_detection_indices,
        selected_scores,
        selected_valid,
        incidence,
        origins,
        directions,
        candidate_ray_valid,
        candidate_weights,
        candidate_count,
        jnp.maximum(candidate_count - plan.candidate_capacity, 0),
        conflict_count,
    )


def associate_multiview(
    detections_by_camera: tuple[ParticleDetections, ...],
    rig: CameraRig,
    plan: MultiViewAssociationPlan,
    /,
) -> MultiViewAssociationResult:
    """Generate, conflict-pack, and triangulate fixed-capacity N-view tuples."""
    if not isinstance(rig, CameraRig):
        raise TypeError("rig must be a CameraRig.")
    if (
        len(detections_by_camera) != plan.camera_capacity
        or rig.capacity != plan.camera_capacity
    ):
        raise ValueError("Detection, rig, and association camera capacities must match.")
    if any(not isinstance(item, ParticleDetections) for item in detections_by_camera):
        raise TypeError("detections_by_camera must contain ParticleDetections values.")
    detection_capacity = _capacity(detections_by_camera[0])
    if any(_capacity(item) != detection_capacity for item in detections_by_camera):
        raise ValueError("All camera detection capacities must match.")
    origins, directions, ray_valid, ray_weights = _camera_rays(detections_by_camera, rig)
    candidates = _enumerate_candidates(
        detections_by_camera,
        origins,
        directions,
        ray_valid,
        ray_weights,
        plan,
    )
    packing_space = SetPackingSpace(
        candidates.incidence,
        valid=candidates.valid,
        maximum_selected=plan.selected_capacity,
    )
    problem = LinearCombinatorialProblem(
        packing_space,
        -candidates.scores,
        problem_id=f"particle-tuples:{plan.plan_id}:"
        + canonical_fingerprint(
            tuple(item.detection_id for item in detections_by_camera)
        ),
    )
    if plan.candidate_capacity <= plan.exact_candidate_limit:
        method = BranchAndBoundSetPacking(
            maximum_nodes=plan.maximum_nodes,
            maximum_candidates=plan.candidate_capacity,
        )
    else:
        method = GreedySetPacking(maximum_candidates=plan.candidate_capacity)
    packing = solve_combinatorial(problem, method)
    packed = packing.decision.selected & candidates.valid
    packed_scores = jnp.where(packed, candidates.scores, -jnp.inf)
    selected_scores, selected_candidate = jax.lax.top_k(
        packed_scores, plan.selected_capacity
    )
    selected_valid = jnp.isfinite(selected_scores)
    detection_indices = candidates.detection_indices[selected_candidate]
    selected_origins = candidates.ray_origins[selected_candidate]
    selected_directions = candidates.ray_directions[selected_candidate]
    selected_ray_valid = (
        candidates.ray_valid[selected_candidate] & selected_valid[:, None]
    )
    selected_weights = candidates.ray_weights[selected_candidate]
    triangulation = triangulate_weighted_rays(
        selected_origins,
        selected_directions,
        selected_ray_valid,
        selected_weights,
    )
    valid = selected_valid & triangulation.valid
    detection_indices = jnp.where(valid[:, None], detection_indices, -1)
    selected_scores = jnp.where(selected_valid, selected_scores, 0.0)
    packing_optimal = packing.status == int(CombinatorialStatus.OPTIMAL)
    status = jnp.where(
        candidates.overflow_count > 0,
        int(AssociationStatus.CANDIDATE_OVERFLOW),
        jnp.where(
            packing_optimal,
            int(AssociationStatus.SUCCESS),
            int(AssociationStatus.HEURISTIC_NOT_CERTIFIED),
        ),
    ).astype(jnp.int32)
    return MultiViewAssociationResult(
        detection_indices,
        selected_scores,
        valid,
        triangulation,
        status,
    )


__all__ = [
    "MultiViewAssociationPlan",
    "MultiViewAssociationResult",
    "TwoViewAssociationPlan",
    "TwoViewAssociationResult",
    "associate_multiview",
    "associate_two_view",
]

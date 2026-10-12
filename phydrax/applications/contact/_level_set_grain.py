#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact import (
    AbstractContactParticipant,
    ParticipantDualityEvidence,
)
from ...discretization.particle._dem_contact import LevelSetPenaltyNormalPlan
from ...discretization.particle._level_set import (
    evaluate_level_set_point_query,
    LevelSetPointQueryResult,
    PreparedLevelSetRigidShapeSet,
)
from ...discretization.particle._rigid_body import RigidBodyKinematics, RigidBodyLoad
from ...sparse import EdgeRelation, RelationExecutionPlan
from ...typing import checked


@final
class MPMLevelSetGrainContactEvaluation(StrictModule):
    """Atomic candidate; status 0 admits, 1 overflow, 2 query, 3 input,
    4 duality, and 5 nonfinite effort refuse all loads.

    Route areas are physical areas, not normalized quadrature probabilities.
    Geometry is retained on refusal so unsupported stencils remain observable.
    ``dissipated_power`` is the loss relative to the raw-gradient elastic
    traction, including the non-tension rebound clamp.
    """

    mpm_generalized_effort: PyTree[Array]
    mpm_surface_force: Array
    grain_load: RigidBodyLoad
    geometry: LevelSetPointQueryResult
    gap: Array
    potential: Array
    route_potential: Array
    route_force: Array
    dissipated_power: Array
    route_active: Array
    route_candidate: Array
    mpm_vertex_indices: Array
    mpm_vertex_ids: Array
    grain_indices: Array
    participant_duality: ParticipantDualityEvidence
    force_defect: Array
    moment_defect: Array
    candidate_count: Array
    active_count: Array
    overflow: Array
    query_valid: Array
    finite: Array
    successful: Array
    status: Array
    maximum_contact_routes: int = eqx.field(static=True)
    participant_id: str = eqx.field(static=True)


@final
class MPMLevelSetGrainContactParticipant(StrictModule, NonTrainableState):
    """One-sided physical MPM surface quadrature against posed grain fields.

    Elastic traction uses the raw interpolated gradient; normal damping and
    the non-tension clamp use its unit normal. Both bodies receive force at the
    exact MPM world position, including finite-radius surface proxies: no projected-point
    torque discrepancy is introduced. Unsupported candidates never commit
    partial loads, including when the fixed route capacity is exceeded.
    """

    mpm: AbstractContactParticipant
    grains: PreparedLevelSetRigidShapeSet
    normal: LevelSetPenaltyNormalPlan
    mpm_surface_area_weights: Array
    mpm_surface_radii: Array
    maximum_contact_routes: int = eqx.field(static=True)
    participant_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mpm: AbstractContactParticipant,
        grains: PreparedLevelSetRigidShapeSet,
        normal: LevelSetPenaltyNormalPlan,
        mpm_surface_area_weights: ArrayLike,
        mpm_surface_radii: ArrayLike,
        /,
        *,
        maximum_contact_routes: int,
        participant_id: str | None = None,
    ) -> None:
        if mpm.surface_plan.ambient_dimension != 3:
            raise ValueError(
                "MPM level-set grain contact requires 3-D surface positions."
            )
        if maximum_contact_routes < 1:
            raise ValueError("maximum_contact_routes must be positive.")
        if grains.bodies.particles.capacity == 0:
            raise ValueError("MPM grain contact requires nonempty grain capacity.")
        count = mpm.surface_plan.vertex_count
        if count < 1 or count * grains.bodies.particles.capacity > np.iinfo(np.int32).max:
            raise ValueError(
                "Candidate addressing requires a nonempty surface and representable fixed capacity."
            )
        area = np.asarray(mpm_surface_area_weights, dtype=np.float64)
        radii = np.asarray(mpm_surface_radii, dtype=np.float64)
        for name, value in (("surface area", area), ("surface radius", radii)):
            if (
                value.shape != (count,)
                or not np.all(np.isfinite(value))
                or np.any(value < 0)
            ):
                raise ValueError(
                    f"MPM {name} must be finite nonnegative per-vertex values."
                )
        vertex_materials = np.asarray(
            mpm.surface_plan.feature_policy.material_ids[:count]
        )
        grain_materials = np.asarray(grains.contact_material_ids)
        for coefficient in (normal.stiffness_per_area, normal.damping_per_area):
            if coefficient.ndim == 2 and (
                np.any(vertex_materials >= coefficient.shape[0])
                or np.any(grain_materials >= coefficient.shape[1])
            ):
                raise ValueError(
                    "Per-area coefficient table does not cover participant materials."
                )
        identity = participant_id or canonical_fingerprint(
            {
                "kind": "mpm-level-set-grain-contact",
                "mpm": mpm.participant_id,
                "grains": grains.shape_id,
                "area": array_tree_fingerprint(area),
                "radius": array_tree_fingerprint(radii),
                "capacity": maximum_contact_routes,
                "stiffness": array_tree_fingerprint(normal.stiffness_per_area),
                "damping": array_tree_fingerprint(normal.damping_per_area),
            }
        )
        self.mpm = mpm
        self.grains = grains
        self.normal = normal
        self.mpm_surface_area_weights = jnp.asarray(area, dtype=jnp.float64)
        self.mpm_surface_radii = jnp.asarray(radii, dtype=jnp.float64)
        self.maximum_contact_routes = maximum_contact_routes
        self.participant_id = identity

    def evaluate(
        self,
        mpm_state: PyTree[Any],
        mpm_rates: PyTree[Any],
        mpm_surface_mass: ArrayLike,
        grain_kinematics: RigidBodyKinematics,
        /,
    ) -> MPMLevelSetGrainContactEvaluation:
        positions = self.mpm.positions(mpm_state)
        velocities = self.mpm.velocities(mpm_state, mpm_rates)
        mass = jnp.asarray(mpm_surface_mass, dtype=positions.dtype)
        if mass.shape != (self.mpm.surface_plan.vertex_count,):
            raise ValueError("MPM surface mass must have one value per vertex.")
        bodies = self.grains.bodies
        body_count = bodies.particles.capacity
        node_live = (mass > 0) & (self.mpm_surface_area_weights > 0)
        input_valid = jnp.all(jnp.isfinite(mass) & (mass >= 0)) & jnp.all(
            ~node_live
            | (
                jnp.all(jnp.isfinite(positions), axis=-1)
                & jnp.all(jnp.isfinite(velocities), axis=-1)
            )
        )
        for value in (
            grain_kinematics.position,
            grain_kinematics.velocity,
            grain_kinematics.orientation,
            grain_kinematics.angular_velocity,
        ):
            input_valid = input_valid & jnp.all(
                ~bodies.particles.active_mask | jnp.all(jnp.isfinite(value), axis=-1)
            )
        # Stream the bounded dense reference search: never retain node×grain scratch.
        total = positions.shape[0] * body_count
        chunk = min(256, self.maximum_contact_routes)
        lane = jnp.arange(chunk, dtype=jnp.int32)

        def collect(block: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            count, slots = carry
            flat = block * chunk + lane
            source = jnp.minimum(flat // body_count, positions.shape[0] - 1)
            target = flat % body_count
            delta = positions[source] - grain_kinematics.position[target]
            bound = self.grains.bounding_radii[target] + self.mpm_surface_radii[source]
            candidates = (
                (flat < total)
                & node_live[source]
                & bodies.particles.active_mask[target]
                & (jnp.sum(delta**2, axis=1) <= bound**2)
            )
            found = jnp.sum(candidates, dtype=jnp.int32)
            local = jnp.nonzero(candidates, size=chunk, fill_value=0)[0]
            destination = count + lane
            destination = jnp.where(
                (lane < found) & (destination < self.maximum_contact_routes),
                destination,
                self.maximum_contact_routes,
            )
            slots = slots.at[destination].set(flat[local], mode="drop")
            return count + found, slots

        candidate_count, selected = jax.lax.fori_loop(
            0,
            (total + chunk - 1) // chunk,
            collect,
            (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.zeros((self.maximum_contact_routes,), dtype=jnp.int32),
            ),
        )
        node = selected // body_count
        grain = selected % body_count
        route_candidate = (
            jnp.arange(self.maximum_contact_routes, dtype=jnp.int32) < candidate_count
        )
        overflow = candidate_count > self.maximum_contact_routes
        points = positions[node]
        safe_points = self.grains.safe_query_points(grain_kinematics, grain)
        query_points = jnp.where(route_candidate[:, None], points, safe_points)
        geometry = evaluate_level_set_point_query(
            self.grains, grain_kinematics, grain, query_points
        )
        query_valid = jnp.all(~route_candidate | geometry.valid)
        gap = geometry.gap - self.mpm_surface_radii[node]
        active = route_candidate & geometry.valid & (gap < 0)
        overlap = jnp.where(active, -gap, 0.0)
        gradient = jnp.where(active[:, None], geometry.raw_gradient, 0.0)
        unit_normal = jnp.where(active[:, None], geometry.unit_normal, 0.0)
        arm = jnp.where(
            route_candidate[:, None], points - grain_kinematics.position[grain], 0.0
        )
        grain_velocity = grain_kinematics.velocity[grain] + jnp.cross(
            grain_kinematics.angular_velocity[grain], arm
        )
        relative_velocity = jnp.where(
            active[:, None], velocities[node] - grain_velocity, 0.0
        )
        normal_rate = jnp.sum(relative_velocity * unit_normal, axis=-1)
        area = self.mpm_surface_area_weights[node]
        left_material = self.mpm.surface_plan.feature_policy.material_ids[node]
        right_material = self.grains.contact_material_ids[grain]
        stiffness = self.normal.stiffness_per_area
        damping = self.normal.damping_per_area
        if stiffness.ndim == 2:
            stiffness = stiffness[left_material, right_material]
        if damping.ndim == 2:
            damping = damping[left_material, right_material]
        elastic_magnitude = stiffness * overlap * jnp.sqrt(jnp.sum(gradient**2, axis=-1))
        magnitude = jnp.maximum(elastic_magnitude - damping * normal_rate, 0.0)
        route_force = area[:, None] * magnitude[:, None] * unit_normal
        # The clamp removes only rebound traction, never stored elastic energy.
        dissipated_power = jnp.sum(area * (elastic_magnitude - magnitude) * normal_rate)
        route_potential = 0.5 * stiffness * overlap * overlap * area
        surface_force = _reduce_routes(node, route_force, active, positions.shape[0])
        grain_payload = jnp.concatenate(
            (-route_force, -jnp.cross(arm, route_force)), axis=1
        )
        grain_load = _reduce_routes(grain, grain_payload, active, body_count)
        force = grain_load[:, :3]
        torque = grain_load[:, 3:]
        pulled = self.mpm.effort_pullback(mpm_state, surface_force)
        duality = self.mpm.duality_evidence(mpm_state, mpm_rates, surface_force)
        finite = jnp.all(jnp.isfinite(route_force)) & jnp.all(
            jnp.isfinite(route_potential)
        )
        finite = finite & jnp.isfinite(dissipated_power)
        for leaf in jax.tree.leaves(pulled):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        finite = finite & jnp.all(jnp.isfinite(force)) & jnp.all(jnp.isfinite(torque))
        successful = ~overflow & query_valid & input_valid & finite & duality.valid
        status = jnp.where(
            overflow,
            1,
            jnp.where(
                ~query_valid,
                2,
                jnp.where(
                    ~input_valid,
                    3,
                    jnp.where(~finite, 5, jnp.where(~duality.valid, 4, 0)),
                ),
            ),
        ).astype(jnp.int32)
        force_defect = jnp.sum(surface_force, axis=0) + jnp.sum(force, axis=0)
        moment_defect = jnp.sum(
            jnp.cross(jnp.where(node_live[:, None], positions, 0.0), surface_force),
            axis=0,
        ) + jnp.sum(torque + jnp.cross(grain_kinematics.position, force), axis=0)
        return MPMLevelSetGrainContactEvaluation(
            jax.tree.map(lambda leaf: jnp.where(successful, leaf, 0.0), pulled),
            jnp.where(successful, surface_force, 0.0),
            RigidBodyLoad(
                jnp.where(successful, force, 0.0), jnp.where(successful, torque, 0.0)
            ),
            geometry,
            gap,
            jnp.where(successful, jnp.sum(route_potential), 0.0),
            jnp.where(successful, route_potential, 0.0),
            jnp.where(successful, route_force, 0.0),
            jnp.where(successful, dissipated_power, 0.0),
            active,
            route_candidate,
            node,
            self.mpm.surface_plan.vertex_ids[node],
            grain,
            duality,
            force_defect,
            moment_defect,
            candidate_count,
            jnp.sum(active, dtype=jnp.int32),
            overflow,
            query_valid,
            finite,
            successful,
            status,
            self.maximum_contact_routes,
            self.participant_id,
        )


def _reduce_routes(
    indices: Array, payload: Array, valid: Array, target_count: int, /
) -> Array:
    relation = EdgeRelation(
        jnp.arange(indices.size, dtype=jnp.int32),
        indices,
        source_size=indices.size,
        target_size=target_count,
        valid=valid,
    )
    execution = RelationExecutionPlan().prepare(relation)
    result, _ = execution.reduce(payload, accumulation="deterministic", output="dense")
    return result


__all__ = ["MPMLevelSetGrainContactEvaluation", "MPMLevelSetGrainContactParticipant"]

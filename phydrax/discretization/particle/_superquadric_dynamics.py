#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ._core import ParticleDiscretization
from ._dem_contact import (
    DEMContactBatch,
    DEMContactModelPlan,
    DEMContactResponse,
    PreparedDEMContactModel,
)
from ._dem_contact_state import (
    DEMContactEvaluationContext,
    DEMContactHistory,
    remap_dem_contact_history,
)
from ._dem_kernels import reduce_dem_contact
from ._neighborhood import (
    AbstractParticleNeighborhoodPlan,
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pair_state import match_particle_pair_keys, ParticlePairKeySpace
from ._precision import ParticleExecutionPolicy, ParticlePrecisionPolicy
from ._rigid_body import (
    PreparedRigidBodySet,
    rigid_body_kick_drift_kick,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._superquadric_contact import (
    PreparedSuperquadricSet,
    superquadric_pair_contact,
    SuperquadricContactPlan,
    SuperquadricContactResult,
    SuperquadricSetPlan,
)


_DEFAULT_STEP_INDEX = jnp.asarray(-1, dtype=jnp.int32)


class SuperquadricDEMPlan(StrictModule, NonTrainableState):
    shapes: SuperquadricSetPlan
    geometry: SuperquadricContactPlan
    contact: DEMContactModelPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shapes: SuperquadricSetPlan,
        geometry: SuperquadricContactPlan,
        contact: DEMContactModelPlan,
        /,
        *,
        plan_id: str | None = None,
    ):
        if not isinstance(shapes, SuperquadricSetPlan):
            raise TypeError("shapes must be a SuperquadricSetPlan.")
        if not isinstance(geometry, SuperquadricContactPlan):
            raise TypeError("geometry must be a SuperquadricContactPlan.")
        if not isinstance(contact, DEMContactModelPlan):
            raise TypeError("contact must be a DEMContactModelPlan.")
        generated = canonical_fingerprint(
            {
                "kind": "superquadric-dem-plan",
                "shapes": shapes.plan_id,
                "geometry": geometry.plan_id,
                "contact": contact.contact_model_id,
            }
        )
        self.shapes = shapes
        self.geometry = geometry
        self.contact = contact
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self,
        particles: ParticleDiscretization,
        materials: Any,
        neighborhood: AbstractParticleNeighborhoodPlan,
        /,
        *,
        execution: ParticleExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
    ):
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(neighborhood, AbstractParticleNeighborhoodPlan):
            raise TypeError("neighborhood must be an AbstractParticleNeighborhoodPlan.")
        shapes = self.shapes.prepare(particles)
        bodies = self.shapes.rigid_body_plan(particles).prepare(particles)
        prepared_neighborhood = neighborhood.prepare(particles)
        contact = self.contact.prepare(materials, 3)
        execution_ = (
            ParticleExecutionPolicy(realization=neighborhood.backend)
            if execution is None
            else execution
        )
        precision_ = (
            ParticlePrecisionPolicy(
                geometry_dtype=particles.plan.coordinate_dtype,
                evaluation_dtype=particles.plan.coordinate_dtype,
            )
            if precision is None
            else precision
        )
        return PreparedSuperquadricDEMDynamics(
            bodies,
            shapes,
            prepared_neighborhood,
            ParticlePairKeySpace(particles),
            contact,
            self.geometry,
            execution_,
            precision_,
            self.plan_id,
        )


class SuperquadricDEMState(StrictModule):
    kinematics: RigidBodyKinematics
    contact_history: DEMContactHistory


class SuperquadricDEMEvaluation(StrictModule):
    neighborhood: ParticleNeighborhoodState
    geometry: SuperquadricContactResult
    contact: DEMContactResponse
    load: RigidBodyLoad
    successful: Array
    prepared_id: str = eqx.field(static=True)


class SuperquadricDEMStepResult(StrictModule):
    candidate_state: SuperquadricDEMState
    accepted_state: SuperquadricDEMState
    evaluation: SuperquadricDEMEvaluation
    successful: Array


class PreparedSuperquadricDEMDynamics(StrictModule, NonTrainableState):
    bodies: PreparedRigidBodySet
    shapes: PreparedSuperquadricSet
    neighborhood: AbstractPreparedParticleNeighborhood
    pair_key_space: ParticlePairKeySpace
    contact_model: PreparedDEMContactModel
    geometry_plan: SuperquadricContactPlan
    execution: ParticleExecutionPolicy
    precision: ParticlePrecisionPolicy
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        bodies,
        shapes,
        neighborhood,
        pair_key_space,
        contact_model,
        geometry_plan,
        execution,
        precision,
        plan_id,
        /,
    ):
        if bodies.particles.prepared_id != shapes.particles.prepared_id:
            raise ValueError("Rigid body and superquadric populations do not match.")
        self.bodies = bodies
        self.shapes = shapes
        self.neighborhood = neighborhood
        self.pair_key_space = pair_key_space
        self.contact_model = contact_model
        self.geometry_plan = geometry_plan
        self.execution = execution
        self.precision = precision
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-superquadric-dem",
                "plan": plan_id,
                "bodies": bodies.prepared_id,
                "shapes": shapes.prepared_id,
                "neighborhood": neighborhood.prepared_id,
                "contact": contact_model.prepared_id,
                "execution": execution.policy_id,
                "precision": precision.policy_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        orientation: ArrayLike,
        angular_velocity: ArrayLike,
        /,
    ) -> SuperquadricDEMState:
        kinematics = self.bodies.kinematics(
            position,
            velocity,
            orientation,
            angular_velocity,
        )
        return SuperquadricDEMState(
            kinematics,
            self.contact_model.empty_history(
                self.neighborhood.pair_capacity,
                kinematics.position.dtype,
            ),
        )

    def evaluate(
        self,
        state: SuperquadricDEMState,
        step_size: Array,
        /,
        *,
        step_index: Array = _DEFAULT_STEP_INDEX,
    ) -> SuperquadricDEMEvaluation:
        if not isinstance(state, SuperquadricDEMState):
            raise TypeError("state must be a SuperquadricDEMState.")
        kinematics = state.kinematics
        neighborhood = self.neighborhood.build(kinematics.position)
        pairs = neighborhood.pair_relation
        keys = self.pair_key_space.keys(pairs)
        remap = match_particle_pair_keys(
            state.contact_history.pair_keys,
            state.contact_history.valid,
            keys.keys,
            keys.valid,
            maximum_key=max(self.pair_key_space.pair_count - 1, 0),
        )
        history = remap_dem_contact_history(
            state.contact_history,
            remap,
            keys.keys,
            keys.valid,
        )
        left = pairs.left_indices
        right = pairs.right_indices
        geometry = superquadric_pair_contact(
            self.geometry_plan,
            self.shapes,
            kinematics.position,
            kinematics.orientation,
            left,
            right,
        )
        left_contact_velocity = kinematics.velocity[left] + jnp.cross(
            kinematics.angular_velocity[left], geometry.left_arm
        )
        right_contact_velocity = kinematics.velocity[right] + jnp.cross(
            kinematics.angular_velocity[right], geometry.right_arm
        )
        relative_velocity = left_contact_velocity - right_contact_velocity
        normal_velocity = jnp.sum(relative_velocity * geometry.normal, axis=-1)
        tangential_velocity = (
            relative_velocity - normal_velocity[:, None] * geometry.normal
        )
        valid = pairs.valid & geometry.valid
        batch = DEMContactBatch(
            geometry.normal,
            geometry.gap,
            jnp.maximum(-geometry.gap, 0.0),
            geometry.effective_radius,
            geometry.left_arm,
            geometry.right_arm,
            normal_velocity,
            tangential_velocity,
            kinematics.angular_velocity[left],
            kinematics.angular_velocity[right],
            valid,
        )
        context = DEMContactEvaluationContext(
            keys.keys,
            keys.valid,
            remap.continued,
            self.bodies.inverse_masses[left],
            self.bodies.inverse_masses[right],
            self.shapes.bounding_radii[left],
            self.shapes.bounding_radii[right],
            self.shapes.material_ids[left],
            self.shapes.material_ids[right],
            jnp.asarray(step_size, dtype=kinematics.position.dtype),
            jnp.asarray(step_index, dtype=jnp.int32),
        )
        contact = self.contact_model.evaluate(batch, history, context)
        load = reduce_dem_contact(
            pairs,
            contact,
            particle_capacity=self.bodies.capacity,
            ambient_dimension=3,
            angular_dimension=3,
            execution=self.execution,
            precision=self.precision,
        )
        successful = (
            neighborhood.successful
            & keys.successful
            & remap.successful
            & jnp.all(~pairs.valid | geometry.valid)
            & contact.successful
        )
        return SuperquadricDEMEvaluation(
            neighborhood,
            geometry,
            contact,
            RigidBodyLoad(load.force, load.torque),
            successful,
            self.prepared_id,
        )

    def step(
        self,
        state: SuperquadricDEMState,
        time: Array,
        step_size: Array,
        /,
        *,
        step_index: Array = _DEFAULT_STEP_INDEX,
    ) -> SuperquadricDEMStepResult:
        first = self.evaluate(state, step_size, step_index=step_index)

        def load_function(next_time, kinematics, args):
            del next_time, args
            staged = SuperquadricDEMState(kinematics, state.contact_history)
            return self.evaluate(staged, step_size, step_index=step_index).load

        body_step = rigid_body_kick_drift_kick(
            self.bodies,
            state.kinematics,
            first.load,
            jnp.asarray(time),
            jnp.asarray(step_size),
            load_function,
            None,
        )
        staged = SuperquadricDEMState(body_step.kinematics, state.contact_history)
        final = self.evaluate(staged, step_size, step_index=step_index)
        candidate = SuperquadricDEMState(
            body_step.kinematics,
            final.contact.next_history,
        )
        successful = first.successful & body_step.successful & final.successful
        accepted = tree_where(successful, candidate, state)
        return SuperquadricDEMStepResult(candidate, accepted, final, successful)


__all__ = [
    "PreparedSuperquadricDEMDynamics",
    "SuperquadricDEMEvaluation",
    "SuperquadricDEMPlan",
    "SuperquadricDEMState",
    "SuperquadricDEMStepResult",
]

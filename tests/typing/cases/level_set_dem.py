"""Level-set, certified field query, packing and MPM-grain annotations."""

from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.applications.contact import (
    MPMLevelSetGrainContactEvaluation,
    MPMLevelSetGrainContactParticipant,
)
from phydrax.discretization import (
    AbstractContactParticipant,
    evaluate_level_set_point_query,
    LevelSetPenaltyNormalPlan,
    LevelSetPointQueryResult,
    LevelSetRigidShapeSetPlan,
    LevelSetTangentialContactPlan,
    ParticleDiscretization,
    PosedSignedFieldQueryResult,
    PreparedLevelSetRigidShapeSet,
    RandomSequentialPackingPlan,
    RigidBodyKinematics,
    RigidBodyLoad,
    RigidDEMPackingResult,
)
from phydrax.typing import PRNGKey


def level_set_contract(
    plan: LevelSetRigidShapeSetPlan,
    particles: ParticleDiscretization,
    kinematics: RigidBodyKinematics,
    indices: Array,
    points: Array,
    key: PRNGKey,
) -> None:
    shapes = plan.prepare(particles)
    assert_type(shapes, PreparedLevelSetRigidShapeSet)
    assert_type(
        evaluate_level_set_point_query(shapes, kinematics, indices, points),
        LevelSetPointQueryResult,
    )
    assert_type(
        shapes.evaluate_signed_field_query(kinematics, indices, points),
        PosedSignedFieldQueryResult,
    )
    packing = RandomSequentialPackingPlan(
        jnp.full((3,), -2.0, dtype=jnp.float64),
        jnp.full((3,), 2.0, dtype=jnp.float64),
        4,
        maximum_candidate_evaluations=1000,
    )
    assert_type(packing.initialize(shapes, key), RigidDEMPackingResult)


def grain_contact_contract(
    mpm: AbstractContactParticipant,
    grains: PreparedLevelSetRigidShapeSet,
    state: Array,
    rates: Array,
    mass: Array,
    area: Array,
    radius: Array,
    kinematics: RigidBodyKinematics,
) -> None:
    normal = LevelSetPenaltyNormalPlan(100.0, damping_per_area=2.0)
    assert_type(normal, LevelSetPenaltyNormalPlan)
    assert_type(LevelSetTangentialContactPlan(20.0), LevelSetTangentialContactPlan)
    participant = MPMLevelSetGrainContactParticipant(
        mpm, grains, normal, area, radius, maximum_contact_routes=8
    )
    result = participant.evaluate(state, rates, mass, kinematics)
    assert_type(result, MPMLevelSetGrainContactEvaluation)
    assert_type(result.grain_load, RigidBodyLoad)
    assert_type(result.mpm_surface_force, Array)

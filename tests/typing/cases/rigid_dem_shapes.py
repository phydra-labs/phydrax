"""Shared compliant shape/compiler/orientation public annotations."""

from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.discretization import (
    AbstractPreparedRigidContactShape,
    AbstractRigidContactShapePlan,
    CompliantContactShapeKind,
    CompliantDEMBodyProperties,
    CompliantDEMMethodPlan,
    DEMContactModelPlan,
    DEMRuntimeState,
    DEMStepEvaluation,
    DenseParticleNeighborhoodPlan,
    LinearSpringDashpotNormalPlan,
    ParticleDiscretization,
    PreparedCompliantDEMDynamics,
    PreparedRigidBodySet,
    RigidBodyKinematics,
    RigidContactGeometryBatch,
)
from phydrax.equations import (
    compile_discrete_element_problem,
    CompiledDiscreteElementProblem,
    DiscreteElementProblemIR,
)


def shape_compiler_contract(
    problem: DiscreteElementProblemIR,
    particles: ParticleDiscretization,
    shapes: AbstractRigidContactShapePlan,
    position: Array,
    velocity: Array,
    angular_velocity: Array,
    orientation: Array,
) -> None:
    method = CompliantDEMMethodPlan(
        DEMContactModelPlan(LinearSpringDashpotNormalPlan(100.0)),
        maximum_contact_routes=8,
    )
    compiled = compile_discrete_element_problem(
        problem,
        particles,
        shapes,
        method,
        neighborhood=DenseParticleNeighborhoodPlan(8),
    )
    assert_type(compiled, CompiledDiscreteElementProblem)
    assert_type(compiled.dynamics, PreparedCompliantDEMDynamics)
    assert_type(compiled.dynamics.shapes, AbstractPreparedRigidContactShape)
    assert_type(compiled.dynamics.bodies, PreparedRigidBodySet)
    assert_type(compiled.dynamics.shapes.shape_kind, CompliantContactShapeKind)
    state = compiled.initialize_state(
        0.0,
        position,
        velocity,
        angular_velocity,
        orientation=orientation,
    )
    assert_type(state, DEMRuntimeState)
    assert_type(state.kinematics, RigidBodyKinematics)
    assert_type(state.body_properties, CompliantDEMBodyProperties)
    assert_type(state.body_properties.shape_scale, Array)
    assert_type(state.body_properties.inertia_com, Array)
    assert_type(state.body_properties.inverse_inertia_com, Array)
    assert_type(state.body_properties.shape_id, str)
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-4),
        None,
    )
    assert_type(detail, DEMStepEvaluation)
    assert_type(detail.evaluation.geometry, RigidContactGeometryBatch)
    assert_type(detail.accepted_state, DEMRuntimeState)

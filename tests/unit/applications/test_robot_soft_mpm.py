#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.robotics._backend import RoboticsOperationRequirement
from phydrax.applications.robotics._soft_mpm import (
    MPMSoftCommand,
    MPMSoftObservationRequest,
    MPMSoftPlant,
    MPMSoftResolutionRequirement,
)
from phydrax.backends._types import BackendUnavailableError
from phydrax.discretization.mpm import MPMRunStatus, MPMRuntimeState
from phydrax.dynamics._plant import AbstractDiscretePlant, PlantStepContext


def _prepared_problem(*, stateful: bool = False, case_ndim: int = 0):
    dimension = 2
    particle_count = 4
    axes = tuple(
        phx.discretization.UniformAxisSpec(12, periodic=True, endpoint=False)
        for _ in range(dimension)
    )
    grid = phx.discretization.TensorGridPlan(axes, axis_names=("x", "y")).prepare(
        jnp.asarray(((0.0, 0.0), (1.0, 1.0)))
    )
    volume = jnp.full((particle_count,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count), volume, ambient_dimension=dimension
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        boundary="reject",
    ).prepare(particles)
    domain = phx.discretization.MPMParticleDomainPlan(
        jnp.asarray(((0.0, 0.0), (1.0, 1.0))),
        periodic=(True, True),
        support_margin=0.0,
    )

    if stateful:
        material = (
            phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(
                dimension
            )
        )
        material_parameters = phx.applications.solid_mechanics.MPMPhaseFieldParameters(
            phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
                2.0, 8.0
            ),
            1.0,
            0.1,
        )
        initial_history = jnp.broadcast_to(jnp.asarray((0.2, 0.3)), (particle_count, 2))
    else:
        material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(
            dimension
        )
        material_parameters = (
            phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
                2.0, 8.0
            )
        )
        initial_history = None

    def commanded_acceleration(time, position, velocity, command):
        del time, velocity
        return jnp.broadcast_to(
            jnp.asarray(command, dtype=position.dtype), position.shape
        )

    problem = phx.equations.MaterialPointProblemIR(
        "soft-robot-mpm",
        material,
        external_acceleration=commanded_acceleration,
        external_acceleration_id="soft-robot-uniform-body-acceleration-v1",
    )
    compiled = phx.equations.compile_material_point_problem(
        problem,
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        domain,
    )
    arguments = phx.equations.MaterialPointArguments(
        material_parameters, jnp.zeros((dimension,))
    )
    position = jnp.asarray(
        (
            (0.28, 0.31),
            (0.42, 0.36),
            (0.34, 0.49),
            (0.48, 0.52),
        )
    )
    velocity = jnp.broadcast_to(jnp.asarray((0.05, 0.0)), position.shape)
    runtime = compiled.initialize_state(
        position,
        velocity,
        volume,
        arguments,
        material_state=initial_history,
    )
    plant = MPMSoftPlant(compiled, runtime, arguments, case_ndim=case_ndim)
    return plant, compiled, runtime, arguments


def _context(state, step_size):
    return PlantStepContext(
        state.time,
        state.time + jnp.asarray(step_size, dtype=state.time.dtype),
        state.step_index,
    )


def test_mpm_soft_plant_routes_body_force_and_reports_motion_conservation_and_work():
    plant, _, initial_runtime, _ = _prepared_problem()
    assert isinstance(plant, AbstractDiscretePlant)
    assert plant.control_schema is not None
    assert plant.features.supports("body-force-command")
    assert plant.profile.capability("step").solvers == ("explicit-mpm",)
    assert plant.profile.capability("contact").supported is False

    reset = plant.reset(jax.random.key(12), plant.parameters)
    source = reset.accepted_state
    command = MPMSoftCommand(jnp.asarray((0.0, -0.2)))
    step = plant.step(_context(source, 0.001), source, command, plant.parameters)

    assert bool(step.successful)
    assert bool(step.evidence.command_routed)
    assert bool(step.evidence.native.successful)
    expected_x = initial_runtime.particles.position[:, 0] + 0.001 * 0.05
    np.testing.assert_allclose(
        step.accepted_state.payload.runtime.particles.position[:, 0],
        expected_x,
        rtol=1e-9,
        atol=1e-12,
    )
    assert jnp.all(
        step.accepted_state.payload.runtime.particles.position[:, 1]
        < initial_runtime.particles.position[:, 1]
    )
    transfer = step.evidence.native.diagnostics.transfer
    energy = step.evidence.native.diagnostics.energy
    np.testing.assert_allclose(transfer.particle_mass, 0.04, atol=1e-12)
    np.testing.assert_allclose(transfer.grid_mass, 0.04, atol=1e-12)
    assert transfer.relative_mass_defect < 1e-12
    assert transfer.relative_momentum_defect < 1e-10
    assert energy.external_work > 0.0

    request = MPMSoftObservationRequest(
        particle_mask=jnp.ones((plant.resolution.particle_capacity,), dtype=bool),
        grid_mask=jnp.ones(plant.resolution.grid_shape, dtype=bool),
        surface_normals=jnp.broadcast_to(
            jnp.asarray((0.0, 1.0)), plant.resolution.grid_shape + (2,)
        ),
    )
    observation = plant.observe(step.accepted_state, request)
    assert bool(observation.successful)
    assert observation.region is not None
    assert observation.surface is not None
    np.testing.assert_allclose(observation.region.mass, 0.04, atol=1e-12)
    np.testing.assert_allclose(observation.surface.mass, 0.04, atol=1e-12)
    assert observation.surface.external_force[1] < 0.0
    assert observation.surface.normal_force < 0.0
    assert observation.semantic_provenance_id == plant.semantic_provenance.semantic_id
    assert observation.numeric_revision_id == plant.numeric_revision.revision_id
    assert observation.state_schema_id == plant.state_schema.schema_id
    assert observation.execution_signature_id == plant.execution_signature.signature_id


def test_casewise_failure_rolls_back_particle_grid_material_history_and_metadata():
    plant, _, runtime, _ = _prepared_problem(stateful=True, case_ndim=1)
    assert jnp.any(runtime.particles.material_state != 0.0)
    keys = jax.random.split(jax.random.key(3), 2)
    reset = plant.reset(keys, plant.parameters, case_shape=(2,))
    source = reset.accepted_state
    context = PlantStepContext(
        source.time,
        jnp.asarray((0.001, 10.0)),
        source.step_index,
    )
    commands = MPMSoftCommand(jnp.asarray(((0.0, -0.2), (0.0, -0.2))))

    result = plant.step(context, source, commands, plant.parameters)

    np.testing.assert_array_equal(result.successful, jnp.asarray((True, False)))
    assert int(result.candidate_state.payload.runtime.last_status[1]) == int(
        MPMRunStatus.STABILITY_LIMIT_EXCEEDED
    )
    assert result.candidate_state.step_index[1] == 1
    assert result.accepted_state.step_index[1] == 0
    assert result.accepted_state.time[1] == source.time[1]
    np.testing.assert_array_equal(
        result.accepted_state.payload.runtime.particles.material_state[1],
        source.payload.runtime.particles.material_state[1],
    )
    for accepted, original in zip(
        jax.tree_util.tree_leaves(result.accepted_state.payload),
        jax.tree_util.tree_leaves(source.payload),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted[1], original[1])
    assert result.accepted_state.time[0] == context.target_time[0]
    assert result.accepted_state.step_index[0] == 1


def test_checkpoint_replay_is_deterministic_and_includes_particle_grid_history():
    plant, _, _, _ = _prepared_problem(stateful=True)
    source = plant.reset(jax.random.key(8), plant.parameters).accepted_state
    checkpoint = plant.checkpoint(source)
    command = MPMSoftCommand(jnp.asarray((0.0, -0.1)))

    first_context = _context(source, 0.001)
    first = plant.step(first_context, source, command, plant.parameters)
    first_digest = plant.state_digest(first.accepted_state)
    second_context = _context(first.accepted_state, 0.001)
    second = plant.step(second_context, first.accepted_state, command, plant.parameters)
    second_digest = plant.state_digest(second.accepted_state)

    replay = plant.replay(
        checkpoint,
        (first_context, second_context),
        (command, command),
        plant.parameters,
        expected_digests=(first_digest, second_digest),
    )

    assert bool(replay.successful)
    assert replay.matched
    assert replay.first_mismatch_step == -1
    assert plant.state_digest(replay.final_state) == second_digest
    np.testing.assert_array_equal(
        replay.final_state.payload.runtime.particles.material_state,
        second.accepted_state.payload.runtime.particles.material_state,
    )
    np.testing.assert_array_equal(
        replay.final_state.payload.grid.momentum,
        second.accepted_state.payload.grid.momentum,
    )


def test_resolution_and_unbound_capability_requirements_fail_closed():
    plant, compiled, runtime, arguments = _prepared_problem()
    assert plant.resolution.particle_capacity == 4
    assert plant.resolution.grid_shape == (12, 12)
    assert plant.resolution.grid_node_count == 144
    assert dict(plant.execution_signature.capacities)["particle_capacity"] == 4

    with pytest.raises(ValueError, match="particle capacity requirement mismatch"):
        plant.require_resolution(MPMSoftResolutionRequirement(particle_capacity=5))
    with pytest.raises(ValueError, match="grid shape requirement mismatch"):
        MPMSoftPlant(
            compiled,
            runtime,
            arguments,
            required_resolution=MPMSoftResolutionRequirement(grid_shape=(10, 10)),
        )
    with pytest.raises(BackendUnavailableError, match="adaptive-grid"):
        MPMSoftPlant(
            compiled,
            runtime,
            arguments,
            required_features=("amr",),
        )
    for feature in ("contact", "amr", "topology-change"):
        with pytest.raises(BackendUnavailableError, match=feature):
            plant.require_features((feature,))
    with pytest.raises(BackendUnavailableError, match="contact-free"):
        plant.profile.require((RoboticsOperationRequirement("contact"),))

    topology_runtime = MPMRuntimeState(
        runtime.particles,
        runtime.time,
        runtime.accepted_step,
        runtime.last_status,
        1,
        runtime.assignment_input,
        runtime.material_slots,
        runtime.body_ids,
        runtime.velocity_field_slots,
        runtime.storage_state,
        runtime.lifecycle_state,
    )
    with pytest.raises(BackendUnavailableError, match="mutated topology"):
        MPMSoftPlant(compiled, topology_runtime, arguments)

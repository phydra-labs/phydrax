#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.applications.robotics._soft_fem import (
    FEM_ATOMIC_REPLAY_CAPABILITY_ID,
    FEM_BODY_FORCE_ACTUATION_CAPABILITY_ID,
    FEM_CONTACT_CAPABILITY_ID,
    FEM_EXACT_CONTROL_CODEC_CAPABILITY_ID,
    FEM_EXACT_STATE_CODEC_CAPABILITY_ID,
    FEM_FIBER_ACTUATION_CAPABILITY_ID,
    FEM_FRACTURE_CAPABILITY_ID,
    FEM_HYPERELASTICITY_CAPABILITY_ID,
    FEM_LINEAR_ELASTICITY_CAPABILITY_ID,
    FEM_PRESSURE_ACTUATION_CAPABILITY_ID,
    FEM_REGION_DISPLACEMENT_SENSOR_CAPABILITY_ID,
    FEM_REGION_FORCE_SENSOR_CAPABILITY_ID,
    FEM_REMESH_CAPABILITY_ID,
    FEM_VISCOELASTICITY_CAPABILITY_ID,
    FEMSoftCommand,
    FEMSoftLoadLayout,
    FEMSoftLoads,
    FEMSoftParameters,
    FEMSoftPlant,
    FEMSoftSensorLayout,
    FEMSoftStepArguments,
)
from phydrax.applications.solid_mechanics._fem_dynamics import (
    FiniteElementDynamicsState,
    prepare_finite_element_dynamics,
)
from phydrax.dynamics import PlantStepContext
from phydrax.equations._finite_element_variational import (
    coefficient,
    FiniteElementForm,
    SourceAction,
)


def _mesh_problem(*, commanded=False):
    points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()
    elasticity = phx.equations.fem.linear_elasticity_form("u", 2.0, 1.0)
    if not commanded:
        form = elasticity
    else:

        def commanded_source(points, context):
            loads = context.user_args.user_args.loads
            pressure = loads.pressure("skin")
            fiber = loads.fiber_tension("fiber-a")
            zero = jnp.zeros_like(pressure)
            vector = loads.body_force("bulk") + jnp.stack((pressure, fiber, zero))
            return jnp.broadcast_to(vector, points.shape)

        source = SourceAction(
            "u",
            coefficient(
                commanded_source,
                coefficient_id="test-soft-fem-all-command-routes",
            ),
            action_id="manufactured-soft-command-source",
        )
        form = FiniteElementForm(
            "soft-commanded-linear-elasticity",
            "u",
            elasticity.actions + (source,),
        )
    return phx.equations.compile_finite_element_problem(form, discretization)


def _load_layout():
    return FEMSoftLoadLayout(
        pressure_region_ids=("skin",),
        fiber_routes=(("fiber-a", "skin"),),
        body_force_region_ids=("bulk",),
        spatial_dimension=3,
    )


def _sensor_layout():
    return FEMSoftSensorLayout(
        displacement_regions=(("tip", (1, 2, 3)),),
        force_region_ids=("skin",),
    )


def _parameters(dtype):
    return FEMSoftParameters({"load_scale": jnp.asarray(1.0, dtype=dtype)})


def _force_sensor(state, loads, parameters):
    scale = parameters["load_scale"]
    return scale * jnp.stack(
        (
            loads.pressure("skin"),
            loads.fiber_tension("fiber-a"),
            jnp.linalg.norm(loads.body_force("bulk")),
        )
    )[None, :].astype(state.displacement.dtype)


def _potential(compiled):
    def potential(time, displacement, args):
        del time, args
        return 0.5 * jnp.sum(displacement * compiled.residual(displacement))

    return potential


def _zero_work(previous, candidate, args):
    del previous, candidate, args
    return jnp.asarray(0.0)


def _plant(
    *,
    commanded=False,
    inverted=False,
    material=False,
    constitutive=FEM_LINEAR_ELASTICITY_CAPABILITY_ID,
    initial_velocity=None,
):
    compiled = _mesh_problem(commanded=commanded)
    displacement = jnp.zeros((4, 3))
    velocity = (
        jnp.zeros_like(displacement)
        if initial_velocity is None
        else jnp.broadcast_to(jnp.asarray(initial_velocity), displacement.shape)
    )
    transaction = None
    material_update = None
    material_update_id = None
    if material:
        transaction = phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    phx.equations.MaterialSiteId("visco-history"),
                    "unit-visco-history",
                    jnp.zeros((1, 2), dtype=displacement.dtype),
                ),
            )
        )

        def update(displacement, velocity, acceleration, time, dt, previous, args):
            del displacement, velocity, acceleration, time, args
            return previous.with_trials(
                {"visco-history": previous.states[0].committed + dt}
            )

        material_update = update
        material_update_id = "unit-visco-history-update"
    initial = FiniteElementDynamicsState(
        displacement,
        velocity,
        jnp.zeros_like(displacement),
        materials=transaction,
    )
    load_layout = _load_layout()
    parameters = _parameters(displacement.dtype)
    sample_args = FEMSoftStepArguments(
        FEMSoftLoads(load_layout.zero_command(displacement.dtype), load_layout),
        parameters.values,
    )

    determinant = None
    determinant_id = None
    if inverted:

        def determinant(time, displacement, args):
            del time, displacement, args
            return jnp.asarray((-1.0,))

        determinant_id = "manufactured-inverted-soft-cell"

    plan = prepare_finite_element_dynamics(
        compiled,
        initial,
        0.1,
        args=sample_args,
        determinant_evaluator=determinant,
        determinant_id=determinant_id,
        minimum_jacobian=0.0,
        material_update=material_update,
        material_update_id=material_update_id,
        potential_energy=None if commanded else _potential(compiled),
        potential_energy_id=None if commanded else "unit-linear-elastic-potential",
        external_work=None if commanded else _zero_work,
        external_work_id=None if commanded else "unit-zero-external-work",
    )
    return FEMSoftPlant(
        plan,
        initial,
        parameters,
        load_layout,
        _sensor_layout(),
        constitutive_capability_id=constitutive,
        initial_region_force=jnp.zeros((1, 3), dtype=displacement.dtype),
        region_force_evaluator=_force_sensor,
        region_force_evaluator_id="unit-command-force-sensor",
    )


def _reset(plant):
    return plant.reset(jax.random.key(7), plant.parameters).accepted_state


def _context(state, dt=0.1):
    return PlantStepContext(state.time, state.time + dt, state.step_index)


def _assert_tree_equal(first, second):
    first_leaves = jax.tree.leaves(first)
    second_leaves = jax.tree.leaves(second)
    assert len(first_leaves) == len(second_leaves)
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(first_leaves, second_leaves, strict=True)
    )


def test_reset_reports_complete_candidate_commit_and_current_sensor_evidence():
    plant = _plant()
    reset = plant.reset(jax.random.key(3), plant.parameters, initial_time=1.25)

    assert bool(reset.attempted)
    assert bool(reset.successful)
    _assert_tree_equal(reset.candidate_state.payload, plant.reset_fallback)
    _assert_tree_equal(reset.accepted_state.payload, plant.reset_fallback)
    assert reset.accepted_state.time == 1.25
    assert reset.accepted_state.step_index == 0
    assert bool(reset.evidence.observation.successful)
    assert reset.evidence.capability_ids == plant.capabilities.capability_ids


def test_manufactured_translation_preserves_zero_strain_energy_and_observation():
    plant = _plant(initial_velocity=(0.4, -0.2, 0.1))
    source = _reset(plant)
    command = plant.load_layout.zero_command(source.payload.displacement.dtype)
    result = plant.step(_context(source), source, command, plant.parameters)

    assert bool(result.successful)
    assert jnp.allclose(
        result.accepted_state.payload.displacement,
        0.1 * source.payload.velocity,
        atol=1.0e-7,
    )
    assert jnp.allclose(
        result.accepted_state.payload.velocity, source.payload.velocity, atol=1.0e-7
    )
    assert bool(result.evidence.dynamics.candidate.energy.available)
    assert bool(result.evidence.dynamics.candidate.energy.finite)
    assert bool(result.evidence.dynamics.candidate.energy.balanced)
    observation = plant.observe(result.accepted_state)
    assert observation.displacement.shape == (1, 3)
    assert observation.force.shape == (1, 3)
    assert bool(observation.successful)


def test_pressure_fiber_and_body_force_are_routed_by_name_into_native_fem_args():
    plant = _plant(commanded=True)
    source = _reset(plant)
    dtype = source.payload.displacement.dtype
    command = FEMSoftCommand(
        jnp.asarray((2.0,), dtype=dtype),
        jnp.asarray((3.0,), dtype=dtype),
        jnp.asarray(((0.0, 0.0, 4.0),), dtype=dtype),
    )
    result = plant.step(_context(source), source, command, plant.parameters)
    loads = result.evidence.loads

    assert bool(result.successful)
    assert loads.pressure("skin") == 2.0
    assert loads.fiber_tension("fiber-a") == 3.0
    assert jnp.array_equal(
        loads.body_force("bulk"), jnp.asarray((0.0, 0.0, 4.0), dtype=dtype)
    )
    assert jnp.linalg.norm(result.accepted_state.payload.displacement) > 0.0
    assert jnp.array_equal(
        result.accepted_state.payload.region_force,
        jnp.asarray(((2.0, 3.0, 4.0),), dtype=dtype),
    )


def test_constitutive_history_is_complete_and_committed_only_on_acceptance():
    plant = _plant(material=True, constitutive=FEM_VISCOELASTICITY_CAPABILITY_ID)
    source = _reset(plant)
    command = plant.load_layout.zero_command(source.payload.displacement.dtype)
    result = plant.step(_context(source), source, command, plant.parameters)

    assert bool(result.successful)
    assert len(source.payload.material_state) == 1
    assert jnp.allclose(result.accepted_state.payload.material_state[0], 0.1)
    second = plant.step(
        _context(result.accepted_state),
        result.accepted_state,
        command,
        plant.parameters,
    )
    assert bool(second.successful)
    assert jnp.allclose(second.accepted_state.payload.material_state[0], 0.2)
    assert (
        second.evidence.dynamics.previous.materials.states[0].state_version
        == plant.material_templates[0].state_version + 1
    )
    assert plant.capabilities.supports(FEM_VISCOELASTICITY_CAPABILITY_ID)


def test_failed_admissibility_keeps_candidate_evidence_and_rolls_back_every_atom():
    plant = _plant(inverted=True, initial_velocity=(0.4, 0.0, 0.0))
    source = _reset(plant)
    source_digest = plant.state_digest(source)
    command = plant.load_layout.zero_command(source.payload.displacement.dtype)
    result = plant.step(_context(source), source, command, plant.parameters)

    assert not bool(result.successful)
    assert not bool(result.evidence.dynamics.candidate.admissibility.jacobian_valid)
    assert jnp.linalg.norm(result.candidate_state.payload.displacement) > 0.0
    assert plant.state_digest(result.accepted_state) == source_digest
    _assert_tree_equal(result.accepted_state, source)


def test_checkpoint_replay_is_deterministic_and_matches_exact_digests():
    plant = _plant(initial_velocity=(0.2, -0.1, 0.05))
    source = _reset(plant)
    checkpoint = plant.checkpoint(source)
    command = plant.load_layout.zero_command(source.payload.displacement.dtype)
    first_context = _context(source)
    first = plant.step(first_context, source, command, plant.parameters)
    second_context = _context(first.accepted_state)
    second = plant.step(second_context, first.accepted_state, command, plant.parameters)
    expected = (
        plant.state_digest(first.accepted_state),
        plant.state_digest(second.accepted_state),
    )

    replay = plant.replay(
        checkpoint,
        (first_context, second_context),
        (command, command),
        plant.parameters,
        expected_digests=expected,
    )

    assert bool(replay.successful)
    assert replay.matched
    assert replay.first_mismatch_step == -1
    assert plant.state_digest(replay.final_state) == expected[-1]


def test_complete_state_and_control_codecs_round_trip_exactly():
    plant = _plant(material=True, constitutive=FEM_VISCOELASTICITY_CAPABILITY_ID)
    state = _reset(plant).payload
    dtype = state.displacement.dtype
    command = FEMSoftCommand(
        jnp.asarray((1.25,), dtype=dtype),
        jnp.asarray((2.5,), dtype=dtype),
        jnp.asarray(((3.0, 4.0, 5.0),), dtype=dtype),
    )

    decoded_state = plant.state_codec.decode_point(plant.state_codec.encode_point(state))
    decoded_command = plant.control_codec.decode_command(
        plant.control_codec.encode_command(command)
    )

    _assert_tree_equal(decoded_state, state)
    _assert_tree_equal(decoded_command, command)
    assert plant.capabilities.supports(FEM_EXACT_STATE_CODEC_CAPABILITY_ID)
    assert plant.capabilities.supports(FEM_EXACT_CONTROL_CODEC_CAPABILITY_ID)


def test_capability_manifest_is_explicit_and_rejects_unimplemented_physics():
    linear = _plant()
    expected = (
        FEM_LINEAR_ELASTICITY_CAPABILITY_ID,
        FEM_PRESSURE_ACTUATION_CAPABILITY_ID,
        FEM_FIBER_ACTUATION_CAPABILITY_ID,
        FEM_BODY_FORCE_ACTUATION_CAPABILITY_ID,
        FEM_REGION_DISPLACEMENT_SENSOR_CAPABILITY_ID,
        FEM_REGION_FORCE_SENSOR_CAPABILITY_ID,
        FEM_ATOMIC_REPLAY_CAPABILITY_ID,
    )

    linear.require_capabilities(expected)
    assert (
        len(
            {
                FEM_LINEAR_ELASTICITY_CAPABILITY_ID,
                FEM_HYPERELASTICITY_CAPABILITY_ID,
                FEM_VISCOELASTICITY_CAPABILITY_ID,
            }
        )
        == 3
    )
    for unsupported in (
        FEM_REMESH_CAPABILITY_ID,
        FEM_FRACTURE_CAPABILITY_ID,
        FEM_CONTACT_CAPABILITY_ID,
    ):
        with pytest.raises(ValueError, match="rejects unsupported"):
            linear.require_capabilities((unsupported,))

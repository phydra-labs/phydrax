from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.robotics._soft_observations import (
    prepare_soft_observation_plan,
    SoftObservationPlan,
    SoftReducedStateQueryPlan,
    SoftSensorPlan,
    SoftSensorState,
)
from phydrax.applications.solid_mechanics._rod_dynamics import prepare_rod, RodPlan
from phydrax.applications.solid_mechanics._rod_plant import prepare_reduced_rod_plant
from phydrax.applications.solid_mechanics._rod_reduced_basis import RodStrainBasisPlan
from phydrax.applications.solid_mechanics._rod_reduced_dynamics import (
    prepare_reduced_rod_dynamics,
    ReducedRodDenseCholeskyPlan,
)
from phydrax.applications.solid_mechanics._rod_reduced_integrators import (
    ReducedRodImplicitMidpoint,
    ReducedRodSemiImplicitVelocityEuler,
)
from phydrax.applications.solid_mechanics._rod_reduction import (
    prepare_reduced_rod,
    ReducedRodPlan,
)
from phydrax.applications.solid_mechanics._rod_tendon import (
    FrictionlessElasticTendonPlan,
    RodMaterialStation,
    TendonRoutePlan,
)
from phydrax.applications.solid_mechanics._rod_tendon_plant import (
    prepare_tendon_driven_rod_plant,
    TendonDrivenRodPlantState,
    TendonDrivenRodPlantStatus,
)
from phydrax.dynamics import PlantStepContext


def _tree_arrays_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        assert jnp.array_equal(left_leaf, right_leaf)


def _base_plant(route: str):
    dtype = jnp.float32
    rod = prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(
                ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
                dtype=dtype,
            ),
            jnp.broadcast_to(jnp.eye(3, dtype=dtype), (2, 3, 3)),
            jnp.ones((3,), dtype=dtype),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((0.2, 0.3, 0.4), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.broadcast_to(
                jnp.diag(jnp.asarray((40.0, 10.0, 10.0), dtype=dtype)),
                (2, 3, 3),
            ),
            jnp.diag(jnp.asarray((4.0, 5.0, 6.0), dtype=dtype))[None, ...],
        )
    )
    basis = RodStrainBasisPlan.shifted_legendre(
        0,
        dimension=3,
        components=("nu_x",),
        component_scales=jnp.ones((6,), dtype=dtype),
    )
    reduction = prepare_reduced_rod(rod, ReducedRodPlan(basis))
    dynamics = prepare_reduced_rod_dynamics(reduction, ReducedRodDenseCholeskyPlan())
    policy = (
        ReducedRodSemiImplicitVelocityEuler(
            maximum_step_size=1.0e-3, energy_balance_tolerance=1.0
        )
        if route == "semi-implicit"
        else ReducedRodImplicitMidpoint(
            maximum_step_size=1.0e-3, energy_balance_tolerance=1.0
        )
    )
    return prepare_reduced_rod_plant(dynamics, policy), reduction


def _plant(
    count: int = 1,
    *,
    route: str = "semi-implicit",
    sensor: bool = False,
):
    base, reduction = _base_plant(route)
    offsets = (0.0,) if count == 1 else (-0.02, 0.0, 0.02)
    tendons = tuple(
        FrictionlessElasticTendonPlan(
            TendonRoutePlan(
                (
                    RodMaterialStation(
                        0,
                        0.0,
                        jnp.asarray((0.0, offset, 0.0), dtype=jnp.float32),
                    ),
                    RodMaterialStation(
                        1,
                        1.0,
                        jnp.asarray((0.0, offset, 0.0), dtype=jnp.float32),
                    ),
                )
            ),
            40.0,
            free_length_bounds=(0.5, 1.5),
            payout_rate_bounds=(-0.2, 0.2),
            tendon_length_bounds=(0.5, 1.5),
            maximum_tension=40.0,
            power_tolerance=5.0e-2,
        ).prepare(reduction)
        for offset in offsets
    )
    observation = None
    if sensor:
        observation = prepare_soft_observation_plan(
            base,
            SoftObservationPlan(
                reduced_state=SoftReducedStateQueryPlan(),
                sensor=SoftSensorPlan(
                    "reduced-state-sensor",
                    noise_standard_deviation=0.0,
                    sample_period=0.0,
                ),
            ),
        )
    return prepare_tendon_driven_rod_plant(
        base,
        tendons,
        (0.95,) * count,
        external_effort_bounds=((-2.0,), (2.0,)),
        observation_plan=observation,
    )


def _reset(plant, seed: int = 7):
    parameters = plant.bind_parameters()
    result = plant.reset(jax.random.key(seed), parameters)
    assert bool(result.successful)
    return parameters, result.accepted_state


def _context(state, duration: float = 1.0e-4):
    step = jnp.asarray(duration, dtype=state.time.dtype)
    return PlantStepContext(state.time, state.time + step, state.step_index)


@pytest.mark.parametrize("count", (1, 3))
def test_one_and_three_tendon_commands_drive_exact_ledgers(count):
    plant = _plant(count)
    parameters, source = _reset(plant)
    rates = tuple(0.01 * (index + 1) for index in range(count))
    command = plant.command(rates, external_effort=(0.02,))
    result = plant.step(_context(source), source, command, parameters)

    assert bool(result.successful)
    assert int(result.status) == int(TendonDrivenRodPlantStatus.SUCCESS)
    assert isinstance(result.accepted_state.payload, TendonDrivenRodPlantState)
    assert len(command.tendon_commands) == count
    assert len(result.evidence.tendon_ledger.payout_evaluations) == count
    assert result.evidence.tendon_ledger.tendon_ids == plant.tendon_ids
    assert bool(result.evidence.tendon_ledger.valid)
    assert bool(result.evidence.tendon_ledger.balanced)
    assert result.evidence.tendon_ledger.source_tension.shape == (count,)
    assert result.evidence.tendon_ledger.candidate_tension.shape == (count,)
    assert result.evidence.integration_result.evidence.ledger.source_ids[-1] == (
        "external-reduced-command"
    )
    for before, after, rate in zip(
        source.payload.actuator_state.states,
        result.accepted_state.payload.actuator_state.states,
        rates,
        strict=True,
    ):
        assert after.free_length == pytest.approx(
            float(before.free_length) + 1.0e-4 * rate
        )


def test_bounds_failure_retains_candidate_but_rolls_back_every_committed_atom():
    plant = _plant()
    parameters, source = _reset(plant)
    result = plant.step(_context(source), source, plant.command((0.25,)), parameters)

    assert not bool(result.successful)
    assert int(result.status) == int(TendonDrivenRodPlantStatus.COMMAND_OUT_OF_BOUNDS)
    assert not bool(result.evidence.command_within_bounds)
    assert (
        result.candidate_state.payload.actuator_state.states[0].free_length
        != source.payload.actuator_state.states[0].free_length
    )
    _tree_arrays_equal(result.accepted_state.payload, source.payload)
    assert result.accepted_state.time == source.time
    assert result.accepted_state.step_index == source.step_index
    assert jnp.array_equal(
        jax.random.key_data(result.accepted_state.key),
        jax.random.key_data(source.key),
    )


@pytest.mark.parametrize(
    ("route", "expected"),
    (
        ("semi-implicit", "semi-implicit-velocity-euler"),
        ("implicit", "implicit-midpoint"),
    ),
)
def test_prepared_integrator_route_is_executed_without_passive_fallback(route, expected):
    plant = _plant(route=route)
    parameters, source = _reset(plant)
    result = plant.step(_context(source), source, plant.zero_command(), parameters)

    assert bool(result.successful)
    integration = result.evidence.integration_result
    assert integration.evidence.route == expected
    assert integration.policy_id == plant.base_plant.policy.policy_id
    if route == "implicit":
        assert integration.evidence.nonlinear_solve_evidence is not None
        assert bool(integration.evidence.nonlinear_solve_successful)
    else:
        assert integration.evidence.nonlinear_solve_evidence is None
        assert bool(integration.evidence.linear_solve_successful)


def test_sensor_sampling_is_part_of_reset_and_step_atomic_commit():
    plant = _plant(sensor=True)
    parameters, source = _reset(plant)

    assert isinstance(source.payload.sensor_state, SoftSensorState)
    assert bool(source.payload.sensor_state.initialized)
    assert source.payload.sensor_state.sample_epoch == 0

    result = plant.step(_context(source), source, plant.zero_command(), parameters)
    sensor = result.accepted_state.payload.sensor_state
    assert bool(result.successful)
    assert result.evidence.observation is not None
    assert bool(result.evidence.observation_valid)
    assert isinstance(sensor, SoftSensorState)
    assert sensor.sample_epoch == 1
    assert sensor.sample_timestamp == result.accepted_state.time
    assert jnp.array_equal(sensor.held_values, result.evidence.observation.values)

    encoded = plant.state_codec.encode_point(result.accepted_state.payload)
    decoded = plant.state_codec.decode_point(encoded)
    _tree_arrays_equal(decoded, result.accepted_state.payload)
    decoded_sensor = decoded.sensor_state
    assert isinstance(decoded_sensor, SoftSensorState)
    assert decoded_sensor.sample_epoch == 1
    assert bool(decoded_sensor.initialized)
    assert any("sample_epoch" in path for path in encoded.mode_paths)
    assert any("initialized" in path for path in encoded.mode_paths)


def test_late_sensor_invalidity_rolls_back_mechanics_actuator_sensor_clock_and_key():
    plant = _plant(sensor=True)
    parameters, source = _reset(plant)
    broken = eqx.tree_at(
        lambda value: value.observation_plan.sensor.noise_standard_deviation,
        plant,
        jnp.full_like(plant.observation_plan.sensor.noise_standard_deviation, jnp.nan),
    )
    result = broken.step(_context(source), source, broken.zero_command(), parameters)

    assert not bool(result.successful)
    assert int(result.status) == int(TendonDrivenRodPlantStatus.SENSOR_INVALID)
    assert not bool(result.evidence.observation_valid)
    assert not bool(result.evidence.observation.finite)
    assert jnp.any(~jnp.isfinite(result.candidate_state.payload.sensor_state.held_values))
    _tree_arrays_equal(result.accepted_state.payload, source.payload)
    assert result.accepted_state.time == source.time
    assert result.accepted_state.step_index == source.step_index
    assert jnp.array_equal(
        jax.random.key_data(result.accepted_state.key),
        jax.random.key_data(source.key),
    )


def test_exact_state_and_control_codecs_are_bound_to_plant_identities():
    plant = _plant(count=3)
    state_encoded = plant.state_codec.encode_point(plant.initial_state)
    state_decoded = plant.state_codec.decode_point(state_encoded)
    command = plant.command((0.01, -0.02, 0.03), external_effort=(0.5,))
    control_encoded = plant.control_codec.encode_command(command)
    control_decoded = plant.control_codec.decode_command(control_encoded)

    _tree_arrays_equal(state_decoded, plant.initial_state)
    _tree_arrays_equal(control_decoded, command)
    assert plant.state_codec.semantic_id == plant.semantic_provenance.semantic_id
    assert plant.control_codec.numeric_revision_id == plant.numeric_revision.revision_id
    assert control_encoded.vector.shape == (4,)

    other = _plant(count=1)
    with pytest.raises(ValueError, match="provenance"):
        other.state_codec.decode_point(state_encoded)
    with pytest.raises(ValueError, match="provenance"):
        other.control_codec.decode_command(control_encoded)


def test_rollout_checkpoint_and_replay_preserve_exact_accepted_trajectory():
    plant = _plant()
    parameters, source = _reset(plant, seed=23)
    checkpoint = plant.checkpoint(source)
    contexts = (
        PlantStepContext(source.time, source.time + 1.0e-4, 0),
        PlantStepContext(source.time + 1.0e-4, source.time + 2.0e-4, 1),
        PlantStepContext(source.time + 2.0e-4, source.time + 3.0e-4, 2),
    )
    commands = (
        plant.command((0.01,)),
        plant.command((0.0,), external_effort=(0.01,)),
        plant.command((-0.01,)),
    )
    direct = source
    digests = []
    for context, command in zip(contexts, commands, strict=True):
        step = plant.step(context, direct, command, parameters)
        assert bool(step.successful)
        direct = step.accepted_state
        digests.append(plant.state_digest(direct))

    replay = plant.replay(
        checkpoint,
        contexts,
        commands,
        parameters,
        expected_digests=tuple(digests),
    )
    assert plant.verify_checkpoint(checkpoint)
    assert bool(replay.successful)
    assert replay.matched
    assert len(replay.accepted_states) == len(contexts) + 1
    assert plant.state_digest(replay.accepted_states[0]) == plant.state_digest(source)
    assert plant.state_digest(replay.final_state) == plant.state_digest(direct)
    _tree_arrays_equal(replay.final_state.payload, direct.payload)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete-state native articulated plant lifecycle, history and replay."""

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.contact._articulated_step import ArticulatedActuationMode
from phydrax.applications.robotics._articulated_plant import (
    ArticulatedPlantCommand,
    ArticulatedPlantParameters,
    ArticulatedPlantStepEvidence,
    NativeArticulatedPlantPlan,
    prepare_articulated_plant,
    PreparedArticulatedPlant,
    REVISION_MISMATCH_STATUS,
)
from phydrax.dynamics import PlantRuntimeState, PlantStepContext
from tests._support.native_articulated import (
    GAP_TOLERANCE,
    LBVH_TRAVERSAL_VISITS,
    pendulum_model,
    route_budget,
    slider_model,
    standard_gravity,
    step_policy,
    zero_gravity,
)


def _plant(
    model: Any,
    /,
    *,
    case_ndim: int = 0,
    gravity: Any = None,
    mode: ArticulatedActuationMode = ArticulatedActuationMode.GENERALIZED_EFFORT,
    **keywords: Any,
) -> PreparedArticulatedPlant:
    # The plant lifecycle runs on the device LBVH route with explicit budgets.
    plan = NativeArticulatedPlantPlan(
        step_policy(),
        gravity=standard_gravity() if gravity is None else gravity,
        actuation_mode=mode,
        case_ndim=case_ndim,
        activation_distance=0.05,
        search_backend="lbvh",
        route_capacities=route_budget(model),
        maximum_traversal_visits=LBVH_TRAVERSAL_VISITS,
        **keywords,
    )
    return prepare_articulated_plant(model, plan)


def _keys(case_shape: tuple[int, ...]) -> jax.Array:
    if case_shape == ():
        return jax.random.key(0)
    return jax.random.split(jax.random.key(0), case_shape[0])


def _context(state: PlantRuntimeState, duration: float) -> PlantStepContext:
    return PlantStepContext(state.time, state.time + duration, state.step_index)


def _step(
    plant: PreparedArticulatedPlant,
    state: PlantRuntimeState,
    values: Any,
    duration: float = 0.01,
) -> Any:
    command = ArticulatedPlantCommand(jnp.asarray(values))
    return plant.step(_context(state, duration), state, command, plant.bind_parameters())


def _equal(left: Any, right: Any) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_batched_reset_accepts_admissible_cases_with_fresh_sensors() -> None:
    plant = _plant(slider_model(), case_ndim=1)
    reset = plant.reset(_keys((3,)), plant.bind_parameters(), case_shape=(3,))
    observation = plant.observe(reset.accepted_state)

    assert np.asarray(reset.successful).tolist() == [True, True, True]
    assert np.asarray(observation.fresh).tolist() == [True, True, True]
    assert observation.frame_transforms.shape == (3, 1, 4, 4)
    assert np.allclose(np.asarray(observation.frame_transforms)[:, 0, 2, 3], 0.1)


def test_invalid_commands_never_enter_mechanics_and_fail_casewise() -> None:
    plant = _plant(slider_model(), case_ndim=1)
    state = plant.reset(
        _keys((3,)), plant.bind_parameters(), case_shape=(3,)
    ).accepted_state
    commands = jnp.asarray([[30.0], [jnp.nan], [0.0]])
    result = _step(plant, state, commands)
    accepted = result.accepted_state

    assert np.asarray(result.successful).tolist() == [True, False, True]
    assert int(result.status[1]) != 0
    assert float(accepted.time[1]) == float(state.time[1])
    assert int(accepted.step_index[1]) == int(state.step_index[1])
    assert _equal(
        jax.tree.map(lambda leaf: leaf[1], accepted.payload),
        jax.tree.map(lambda leaf: leaf[1], state.payload),
    )
    # Thirty newtons upward exceed the slider weight; zero effort keeps it resting.
    assert float(accepted.payload.velocity[0, 0]) > 0.0
    assert abs(float(accepted.payload.velocity[2, 0])) <= 1.0e-9
    assert np.asarray(accepted.payload.command)[0].tolist() == [30.0]


def test_contact_birth_is_recorded_in_complete_history() -> None:
    plant = _plant(
        slider_model(), initial_lifted_configuration=[0.002], initial_velocity=[0.0]
    )
    state = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    impulses: list[float] = []
    for _ in range(4):
        result = _step(plant, state, [0.0])
        assert bool(result.successful)
        state = result.accepted_state
        impulses.append(float(jnp.sum(state.payload.sensors.contact_normal_impulse)))

    assert impulses[0] == 0.0
    assert impulses[-1] > 0.0
    assert float(state.payload.configuration[0]) >= -1.0e-9
    assert int(state.payload.epoch) == 4
    assert state.payload.contact_history is not None
    assert bool(jnp.any(state.payload.contact_history.active))


def test_checkpoint_replay_reproduces_manual_stepping_exactly() -> None:
    plant = _plant(
        slider_model(), initial_lifted_configuration=[0.001], initial_velocity=[0.0]
    )
    state = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    checkpoint = plant.checkpoint(state)
    contexts: list[PlantStepContext] = []
    commands: list[ArticulatedPlantCommand] = []
    digests: list[str] = []
    for value in (0.0, 5.0, 0.0):
        command = ArticulatedPlantCommand(jnp.asarray([value]))
        context = _context(state, 0.01)
        state = plant.step(
            context, state, command, plant.bind_parameters()
        ).accepted_state
        contexts.append(context)
        commands.append(command)
        digests.append(plant.state_digest(state))

    replay = plant.replay(
        checkpoint, contexts, commands, plant.bind_parameters(), expected_digests=digests
    )

    assert replay.matched
    assert bool(jnp.all(replay.successful))
    assert plant.verify_checkpoint(checkpoint)
    restored = plant.restore(plant.checkpoint(state))
    assert plant.state_digest(restored) == digests[-1]


def test_same_principal_state_with_different_winding_steps_differently() -> None:
    upper = 2.0 * math.pi + 0.05
    plant = _plant(
        pendulum_model(obstacle_angle=None, stop_bounds=(-10.0, upper)),
        gravity=zero_gravity(),
        initial_lifted_configuration=[2.0 * math.pi],
        initial_velocity=[1.0],
    )
    wound = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    assert int(wound.payload.winding[0]) == 1
    assert abs(float(wound.payload.configuration[0])) <= 1.0e-12
    unwound = eqx.tree_at(
        lambda state: state.payload.winding, wound, jnp.zeros_like(wound.payload.winding)
    )

    near_stop = _step(plant, wound, [0.0], duration=0.1)
    free = _step(plant, unwound, [0.0], duration=0.1)

    assert bool(near_stop.successful) and bool(free.successful)
    # Position-level law: the stop is reached at the endpoint, q1 = upper, so
    # the arrival velocity is the remaining gap over h (0.05 rad / 0.1 s),
    # within the certified gap floor over h; it is not a zero normal speed.
    gap_rate_tolerance = GAP_TOLERANCE / 0.1
    arrived = near_stop.accepted_state
    assert float(arrived.payload.velocity[0]) == pytest.approx(
        0.05 / 0.1, abs=gap_rate_tolerance
    )
    assert float(arrived.payload.stop_impulse[1]) > 0.0
    lifted = plant.observe(arrived).lifted_configuration
    assert float(lifted[0]) == pytest.approx(upper, abs=GAP_TOLERANCE)
    # The next request starts on the stop and the reaction stops the motion.
    resting = _step(plant, arrived, [0.0], duration=0.1)
    assert bool(resting.successful)
    assert abs(float(resting.accepted_state.payload.velocity[0])) <= gap_rate_tolerance
    assert float(resting.accepted_state.payload.stop_impulse[1]) > 0.0
    assert float(free.accepted_state.payload.velocity[0]) == pytest.approx(1.0)
    assert abs(float(free.accepted_state.payload.stop_impulse[1])) <= 1.0e-12


def test_observation_is_pure_and_refresh_restores_freshness() -> None:
    plant = _plant(slider_model())
    state = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    first = plant.observe(state)
    second = plant.observe(state)
    assert _equal(first, second)

    stale = eqx.tree_at(
        lambda value: value.payload.sensor_epoch,
        state,
        state.payload.sensor_epoch - 1,
    )
    assert not bool(plant.observe(stale).fresh)
    refreshed = plant.refresh(stale)

    assert bool(refreshed.successful)
    assert bool(refreshed.observation.fresh)
    assert _equal(refreshed.accepted_state.payload.sensors, state.payload.sensors)
    assert plant.state_digest(refreshed.accepted_state) == plant.state_digest(state)


def test_mismatched_history_revision_is_refused_without_mutation() -> None:
    plant = _plant(slider_model())
    state = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    stale = eqx.tree_at(
        lambda value: value.payload.geometry_revision,
        state,
        state.payload.geometry_revision + 1,
    )
    result = _step(plant, stale, [0.0])

    assert not bool(result.successful)
    assert int(result.status) == REVISION_MISMATCH_STATUS
    assert plant.state_digest(result.accepted_state) == plant.state_digest(stale)
    assert isinstance(result.evidence, ArticulatedPlantStepEvidence)
    assert not bool(result.evidence.revision_valid)


def test_inelastic_impact_through_plant_keeps_momentum_and_loss_ledger() -> None:
    mass, speed = 2.0, 1.0
    plant = _plant(
        slider_model(mass=mass),
        gravity=zero_gravity(),
        initial_lifted_configuration=[0.0],
        initial_velocity=[-speed],
    )
    state = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    result = _step(plant, state, [0.0])
    ledger = result.evidence.step.ledger

    assert bool(result.successful)
    assert abs(float(result.accepted_state.payload.velocity[0])) <= 1.0e-9
    assert float(ledger.correction_loss) == pytest.approx(
        0.5 * mass * speed**2, rel=1.0e-9
    )
    assert float(
        jnp.sum(result.accepted_state.payload.sensors.contact_normal_impulse)
    ) == (pytest.approx(mass * speed, rel=1.0e-9))


def test_reset_refuses_raw_uint32_keys_and_accepts_varying_float_times() -> None:
    plant = _plant(slider_model())
    with pytest.raises(TypeError, match="typed PRNG key"):
        plant.reset(jax.random.PRNGKey(0), plant.bind_parameters())
    for time in (0.0, 0.25):
        reset = plant.reset(_keys(()), plant.bind_parameters(), initial_time=time)
        assert bool(reset.successful)
        assert float(reset.accepted_state.time) == pytest.approx(time)
    with pytest.raises(ValueError):
        plant.reset(_keys(()), plant.bind_parameters(), case_shape=(2,))


def test_actuation_mode_must_match_declared_model_actuators() -> None:
    with pytest.raises(ValueError, match="MAPPED_MOTORS"):
        _plant(slider_model(), mode=ArticulatedActuationMode.MAPPED_MOTORS)
    autonomous = _plant(slider_model(), mode=ArticulatedActuationMode.AUTONOMOUS)
    assert autonomous.control_schema is None
    state = autonomous.reset(_keys(()), autonomous.bind_parameters()).accepted_state
    result = autonomous.step(
        _context(state, 0.01), state, None, autonomous.bind_parameters()
    )
    assert bool(result.successful)


def test_gravity_parameter_is_bound_to_plant_revision() -> None:
    plant = _plant(slider_model(), initial_lifted_configuration=[0.5])
    state = plant.reset(_keys(()), plant.bind_parameters()).accepted_state
    lunar = plant.bind_parameters(
        ArticulatedPlantParameters(jnp.asarray([0.0, 0.0, -1.62]))
    )
    result = plant.step(
        _context(state, 0.01), state, ArticulatedPlantCommand(jnp.asarray([0.0])), lunar
    )

    assert bool(result.successful)
    assert float(result.accepted_state.payload.velocity[0]) == pytest.approx(-0.0162)

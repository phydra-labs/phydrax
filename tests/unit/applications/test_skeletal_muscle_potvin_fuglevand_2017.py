import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._trainable import partition_trainable
from phydrax.applications.skeletal_muscle.motor_units import (
    potvin_fuglevand_2017_default_parameters,
    POTVIN_FUGLEVAND_2017_DOI,
    POTVIN_FUGLEVAND_2017_REFERENCE_SHA,
    PotvinFuglevand2017Parameters,
    PotvinFuglevand2017Plan,
    PotvinFuglevand2017State,
    PotvinFuglevand2017Status,
)
from phydrax.dynamics import DiscreteStepContext


def test_default_population_reproduces_published_endpoints():
    parameters = potvin_fuglevand_2017_default_parameters()
    runtime = PotvinFuglevand2017Plan().prepare(parameters)

    assert POTVIN_FUGLEVAND_2017_DOI == "10.1371/journal.pcbi.1005581"
    assert POTVIN_FUGLEVAND_2017_REFERENCE_SHA == (
        "15462f85106ed9ebde3d78ab6fe665c88bf8b32e"
    )
    np.testing.assert_allclose(
        np.asarray(parameters.recruitment_threshold)[[0, -1]], [1.0, 50.0]
    )
    np.testing.assert_allclose(
        np.asarray(parameters.rested_twitch_force)[[0, -1]], [1.0, 100.0]
    )
    np.testing.assert_allclose(
        np.asarray(parameters.resting_contraction_time_s)[[0, -1]], [0.09, 0.03]
    )
    np.testing.assert_allclose(
        np.asarray(parameters.maximum_firing_rate_hz)[[0, -1]], [35.0, 25.0]
    )
    np.testing.assert_allclose(
        np.asarray(parameters.nominal_twitch_force_loss_per_s)[[0, -1]],
        [0.000125, 2.25],
    )
    np.testing.assert_allclose(runtime.maximum_excitation(), 67.0)
    np.testing.assert_allclose(runtime.rested_maximum_force(), 2215.9811474699964)

def test_float32_population_preserves_requested_runtime_dtype():
    runtime = PotvinFuglevand2017Plan(dtype=np.float32).prepare()
    candidate = runtime.candidate(runtime.initialize(), 20.125, 0.1)

    assert bool(candidate.evidence.successful)
    assert candidate.candidate_state.current_twitch_force.dtype == jnp.float32
    assert candidate.output.motor_unit_force.dtype == jnp.float32
    assert candidate.output.total_force.dtype == jnp.float32


def test_recruitment_threshold_and_saturation_boundaries_are_exact():
    runtime = PotvinFuglevand2017Plan(
        central_adaptation=False, peripheral_fatigue=False
    ).prepare()
    state = runtime.initialize()

    below = runtime.evaluate(state, np.nextafter(1.0, 0.0))
    at = runtime.evaluate(state, 1.0)
    maximum = runtime.evaluate(state, 67.0)

    assert not bool(jnp.any(below.recruited))
    assert below.total_force == 0.0
    assert bool(at.recruited[0])
    assert at.unadapted_firing_rate_hz[0] == 8.0
    assert not bool(jnp.any(at.recruited[1:]))
    assert bool(jnp.all(maximum.recruited))
    assert bool(jnp.all(maximum.saturated))
    np.testing.assert_allclose(
        np.asarray(maximum.firing_rate_hz)[[0, -1]], [35.0, 25.0]
    )


def test_force_frequency_branches_are_continuous_at_point_four():
    parameters = PotvinFuglevand2017Parameters(
        jnp.asarray([1.0, 100.0]),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.1, 0.05]),
        jnp.asarray([20.0, 20.0]),
        jnp.asarray([0.0, 0.0]),
        minimum_firing_rate_hz=1.0,
        firing_rate_gain_hz=1.0,
        derecruitment_delta_hz=0.0,
        adaptation_scale=0.0,
        adaptation_time_constant_s=1.0,
        contraction_time_change_ratio=0.0,
    )
    runtime = PotvinFuglevand2017Plan(
        2, central_adaptation=False, peripheral_fatigue=False
    ).prepare(parameters)
    state = runtime.initialize()
    exact = runtime.evaluate(state, 4.0)
    above = runtime.evaluate(state, 4.0 + 1.0e-8)
    expected = 1.0 - np.exp(-2.0 * 0.4**3)

    np.testing.assert_allclose(exact.normalized_firing_rate[0], 0.4)
    np.testing.assert_allclose(exact.normalized_force[0], expected)
    np.testing.assert_allclose(above.normalized_force[0], expected, atol=1.0e-8)


def test_adaptation_uses_source_duration_and_tracks_time_since_first_recruitment():
    runtime = PotvinFuglevand2017Plan(peripheral_fatigue=False).prepare()
    source = runtime.initialize()
    first = runtime.candidate(source, 20.0, 0.1)

    assert bool(first.evidence.successful)
    assert bool(jnp.all(first.output.firing_rate_adaptation_hz == 0.0))
    recruited = first.output.recruited
    committed = first.commit()
    np.testing.assert_allclose(committed.recruitment_duration_s[recruited], 0.1)
    np.testing.assert_allclose(committed.recruitment_duration_s[~recruited], 0.0)

    second = runtime.candidate(committed, 20.0, 0.1)
    assert bool(jnp.any(second.output.firing_rate_adaptation_hz > 0.0))
    inactive = runtime.candidate(second.commit(), 0.0, 0.1).commit()
    np.testing.assert_allclose(inactive.recruitment_duration_s[recruited], 0.3)
    np.testing.assert_allclose(inactive.recruitment_duration_s[~recruited], 0.0)


def test_force_is_evaluated_before_peripheral_capacity_update():
    no_fatigue = PotvinFuglevand2017Plan(
        central_adaptation=False, peripheral_fatigue=False
    ).prepare()
    fatigue = PotvinFuglevand2017Plan(central_adaptation=False).prepare()
    source = fatigue.initialize()

    candidate = fatigue.candidate(source, 67.0, 0.1)
    baseline = no_fatigue.evaluate(no_fatigue.initialize(), 67.0)
    next_output = fatigue.evaluate(candidate.commit(), 67.0)

    np.testing.assert_allclose(candidate.output.total_force, baseline.total_force)
    assert candidate.output.total_force_capacity_fraction == 1.0
    assert candidate.commit().current_twitch_force[-1] < source.current_twitch_force[-1]
    assert next_output.total_force < candidate.output.total_force
    rested_weight = np.asarray(baseline.normalized_force)
    expected_capacity = np.sum(
        rested_weight * np.asarray(candidate.commit().current_twitch_force)
    ) / np.sum(rested_weight * np.asarray(fatigue.parameters.rested_twitch_force))
    np.testing.assert_allclose(
        next_output.total_force_capacity_fraction, expected_capacity
    )


def test_fatigue_mechanism_selections_are_static_and_independent():
    source = PotvinFuglevand2017Plan().prepare().initialize()
    neither = PotvinFuglevand2017Plan(
        central_adaptation=False, peripheral_fatigue=False
    ).prepare()
    central = PotvinFuglevand2017Plan(peripheral_fatigue=False).prepare()
    peripheral = PotvinFuglevand2017Plan(central_adaptation=False).prepare()

    neither_state = neither.candidate(source, 20.0, 0.1).commit()
    central_state = central.candidate(source, 20.0, 0.1).commit()
    peripheral_state = peripheral.candidate(source, 20.0, 0.1).commit()

    np.testing.assert_allclose(neither_state.recruitment_duration_s, 0.0)
    np.testing.assert_allclose(neither_state.current_twitch_force, source.current_twitch_force)
    assert bool(jnp.any(central_state.recruitment_duration_s > 0.0))
    np.testing.assert_allclose(central_state.current_twitch_force, source.current_twitch_force)
    np.testing.assert_allclose(peripheral_state.recruitment_duration_s, 0.0)
    assert bool(jnp.any(peripheral_state.current_twitch_force < source.current_twitch_force))


@pytest.mark.parametrize(
    ("excitation", "step_s", "status"),
    [
        (-1.0, 0.1, PotvinFuglevand2017Status.INVALID_EXCITATION),
        (68.0, 0.1, PotvinFuglevand2017Status.INVALID_EXCITATION),
        (20.0, 0.0, PotvinFuglevand2017Status.INVALID_STEP),
        (20.0, 0.11, PotvinFuglevand2017Status.INVALID_STEP),
    ],
)
def test_invalid_interval_inputs_roll_back(excitation, step_s, status):
    runtime = PotvinFuglevand2017Plan().prepare()
    source = runtime.initialize()
    candidate = runtime.candidate(source, excitation, step_s)

    assert not bool(candidate.evidence.successful)
    assert int(candidate.evidence.status) & int(status)
    committed = candidate.commit()
    np.testing.assert_array_equal(
        committed.recruitment_duration_s, source.recruitment_duration_s
    )
    np.testing.assert_array_equal(
        committed.current_twitch_force, source.current_twitch_force
    )

def test_direct_evaluation_refuses_inputs_outside_the_model_domain():
    runtime = PotvinFuglevand2017Plan().prepare()
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="outside the model domain",
    ):
        jax.block_until_ready(
            runtime.evaluate(runtime.initialize(), -1.0).total_force
        )


def test_invalid_state_and_trained_parameters_roll_back():
    runtime = PotvinFuglevand2017Plan().prepare()
    source = runtime.initialize()
    invalid_state = PotvinFuglevand2017State(
        source.recruitment_duration_s,
        source.current_twitch_force.at[0].set(jnp.nan),
    )
    state_candidate = runtime.candidate(invalid_state, 20.0, 0.1)
    assert int(state_candidate.evidence.status) & int(
        PotvinFuglevand2017Status.NONFINITE
    )
    assert int(state_candidate.evidence.status) & int(
        PotvinFuglevand2017Status.INVALID_STATE
    )

    invalid_runtime = eqx.tree_at(
        lambda value: value.parameters.adaptation_time_constant_s,
        runtime,
        jnp.asarray(-1.0),
    )
    parameter_candidate = invalid_runtime.candidate(source, 20.0, 0.1)
    assert int(parameter_candidate.evidence.status) & int(
        PotvinFuglevand2017Status.INVALID_PARAMETERS
    )
    np.testing.assert_array_equal(
        parameter_candidate.commit().current_twitch_force,
        source.current_twitch_force,
    )


def test_array_dynamics_view_matches_typed_candidate():
    runtime = PotvinFuglevand2017Plan().prepare()
    source = runtime.initialize()
    packed = runtime.pack_state(source)
    context = DiscreteStepContext(0.0, 0.1, 0)
    result = runtime.discrete_system.evaluate_result(
        context,
        packed,
        runtime.parameters,
        inputs=jnp.asarray([20.0]),
    )
    typed = runtime.candidate(source, 20.0, 0.1)

    np.testing.assert_allclose(result.candidate_state, runtime.pack_state(typed.candidate_state))
    np.testing.assert_allclose(result.accepted_state, runtime.pack_state(typed.commit()))
    assert bool(result.successful)
    assert int(result.status) == int(typed.evidence.status)


def test_jit_vmap_pathwise_gradient_and_parameter_partitioning():
    runtime = PotvinFuglevand2017Plan().prepare()
    state = runtime.initialize()
    compiled = eqx.filter_jit(runtime.candidate)(state, 20.125, 0.1)
    eager = runtime.candidate(state, 20.125, 0.1)
    np.testing.assert_allclose(compiled.output.total_force, eager.output.total_force)

    drives = jnp.asarray([10.125, 20.125, 30.125])
    batched = jax.vmap(lambda drive: runtime.evaluate(state, drive).total_force)(drives)
    assert batched.shape == (3,)
    assert bool(jnp.all(jnp.diff(batched) > 0.0))

    derivative = jax.grad(lambda drive: runtime.evaluate(state, drive).total_force)(20.125)
    assert jnp.isfinite(derivative)
    assert derivative > 0.0
    assert eager.evidence.minimum_recruitment_margin > 0.0
    assert eager.evidence.minimum_saturation_margin > 0.0

    trainable, fixed = partition_trainable(runtime)
    assert trainable.parameters.rested_twitch_force is not None
    assert fixed.plan is runtime.plan

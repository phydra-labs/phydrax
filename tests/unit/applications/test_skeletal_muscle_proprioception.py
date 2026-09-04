#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.proprioception import (
    MileusnicSpindle2006Plan,
    MileusnicSpindleInput,
    MileusnicSpindleStatus,
)


def _input(length=1.0, velocity=0.0, acceleration=0.0, dynamic=0.0, static=0.0):
    return MileusnicSpindleInput(length, velocity, acceleration, dynamic, static)


def test_published_feline_parameters_and_equilibrium_initialization():
    runtime = MileusnicSpindle2006Plan().prepare()
    state = runtime.initialize(_input())
    rates = runtime.rates(state, _input())
    output = runtime.output(state, _input())

    assert runtime.parameters.beta_zero.shape == (3,)
    assert runtime.parameters.primary_gain_pps.tolist() == [20_000.0, 10_000.0, 10_000.0]
    np.testing.assert_allclose(
        rates.branch_tension_rate_force_unit_per_s, 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        rates.branch_tension_acceleration_force_unit_per_s2, 0.0, atol=1.0e-9
    )
    assert output.primary_afferent_pps >= 0.0
    assert output.secondary_afferent_pps >= 0.0


def test_dynamic_and_static_gamma_drive_distinct_branches():
    runtime = MileusnicSpindle2006Plan().prepare()
    state = runtime.initialize(_input())
    dynamic = runtime._fusimotor_targets(_input(dynamic=70.0))
    static = runtime._fusimotor_targets(_input(static=70.0))

    assert dynamic[0] > 0.0
    assert dynamic[1] == 0.0
    assert dynamic[2] == 0.0
    assert static[0] == 0.0
    assert static[1] > 0.0
    assert static[2] > 0.0
    driven = runtime.output(
        runtime.initialize(_input(dynamic=70.0, static=70.0)),
        _input(dynamic=70.0, static=70.0),
    )
    assert driven.primary_afferent_pps > 0.0
    assert driven.secondary_afferent_pps > 0.0


def test_ramp_stretch_increases_primary_afferent_and_is_jittable():
    runtime = MileusnicSpindle2006Plan().prepare()
    resting_input = _input()
    state = runtime.initialize(resting_input)
    resting = runtime.output(state, resting_input).primary_afferent_pps
    ramp = _input(length=1.02, velocity=0.1, dynamic=70.0)

    @eqx.filter_jit
    def advance(value):
        candidate = runtime.candidate(value, ramp, 1.0e-4)
        return candidate.commit(), candidate.evidence.successful

    for _ in range(100):
        state, successful = advance(state)
        assert bool(successful)
    stretched = runtime.output(state, ramp)
    assert stretched.primary_afferent_pps > resting
    derivative = jax.grad(
        lambda length: runtime.output(
            state, _input(length=length, velocity=0.1, dynamic=70.0)
        ).primary_afferent_pps
    )(jnp.asarray(1.02))
    assert jnp.isfinite(derivative)


def test_invalid_input_and_step_roll_back_whole_state():
    runtime = MileusnicSpindle2006Plan().prepare()
    state = runtime.initialize(_input())
    invalid_input = runtime.candidate(state, _input(length=-1.0), 1.0e-4)
    invalid_step = runtime.candidate(state, _input(), 1.0e-3)

    assert not bool(invalid_input.evidence.successful)
    assert int(invalid_input.evidence.status) & int(MileusnicSpindleStatus.INVALID_INPUT)
    assert not bool(invalid_step.evidence.successful)
    assert int(invalid_step.evidence.status) & int(MileusnicSpindleStatus.INVALID_STEP)
    np.testing.assert_array_equal(
        invalid_input.commit().branch_tension_force_unit,
        state.branch_tension_force_unit,
    )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    prepare_replay_schedule,
)
from phydrax.applications.geophysics import (
    AcousticGrid,
    BoundedAcousticWavespeed,
    ConstantDensityAcousticPlan,
    PreparedAcousticSampling,
    PreparedSeismicObservation,
    ricker_wavelet,
    SeismicAcquisition,
    SeismicGaussianLikelihood,
)
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.units import KILOMETER, PASCAL


def _survey(steps=19, absorber=3):
    grid = AcousticGrid((13, 13), (0.25, 0.25))
    acquisition = SeismicAcquisition(grid, [[1.5, 1.5]], [[2.0, 1.5], [1.5, 2.0]])
    plan = ConstantDensityAcousticPlan(
        grid, 0.05, steps, 2.0, 2.0, absorber_cells=absorber
    )
    times = (jnp.arange(steps) + 0.5) * plan.time_step
    rates = ricker_wavelet(times, 3.0, delay=0.2)[:, None] * 0.01
    return plan, acquisition, rates


def test_prepared_sampling_affine_fields_and_exact_transpose():
    grid = AcousticGrid((7, 9), (0.002, 0.003), (0.01, -0.02), length_unit=KILOMETER)
    points = np.asarray([[11.2, -18.1], [17.3, 1.8], [22.0, 4.0]])
    sampling = PreparedAcousticSampling(grid, points)
    x, y = jnp.meshgrid(*grid.axis_nodes(), indexing="ij")
    field = 2.0 * x - 3.0 * y + 5.0
    np.testing.assert_allclose(
        sampling.apply(field), 2.0 * points[:, 0] - 3.0 * points[:, 1] + 5.0, rtol=2e-6
    )
    cotangent = jnp.asarray([0.7, -1.3, 2.1])
    lhs = jnp.sum(sampling.apply(field) * cotangent)
    rhs = jnp.sum(field * sampling.transpose(cotangent))
    np.testing.assert_allclose(lhs, rhs, rtol=2e-6)
    with pytest.raises(ValueError, match="within"):
        PreparedAcousticSampling(grid, [[9.0, 0.0]])


@pytest.mark.parametrize("dimensions", [2, 3])
def test_monopole_volume_rate_scaling_and_axis_symmetry(dimensions):
    grid = AcousticGrid((7,) * dimensions, (0.5,) * dimensions)
    centre = [1.5] * dimensions
    acquisition = SeismicAcquisition(grid, [centre], [centre])
    plan = ConstantDensityAcousticPlan(grid, 0.05, 3, 2.0, 3.0)
    initial = plan.initial_state()
    state = plan.step(initial, 2.0, acquisition, jnp.asarray([0.7]))
    expected_pressure = 3.0 * 2.0**2 * 0.7 * 0.05 / grid.cell_measure
    np.testing.assert_allclose(
        state.pressure[(3,) * dimensions], expected_pressure, rtol=2e-6
    )
    np.testing.assert_allclose(
        jnp.sum(state.pressure) * grid.cell_measure / (3.0 * 2.0**2),
        0.7 * 0.05,
        rtol=2e-6,
    )
    propagated = plan.step(state, 2.0, acquisition, jnp.asarray([0.0])).pressure
    neighbours = []
    for axis in range(dimensions):
        for sign in (-1, 1):
            index = [3] * dimensions
            index[axis] += sign
            neighbours.append(propagated[tuple(index)])
    assert float(neighbours[0]) > 0
    np.testing.assert_allclose(
        neighbours, jnp.full((2 * dimensions,), neighbours[0]), rtol=2e-6
    )


def test_rigid_wall_standing_wave_matches_discrete_dispersion():
    nx, steps, dt, speed = 32, 40, 0.1, 2.0
    grid = AcousticGrid((nx, 4), (1.0, 1.0), (0.5, 0.5))
    acquisition = SeismicAcquisition(grid, [[15.5, 1.5]], [[4.5, 1.5]])
    plan = ConstantDensityAcousticPlan(grid, dt, steps, speed, 1.0)
    mode = 3
    pressure = jnp.cos(mode * jnp.pi * (jnp.arange(nx) + 0.5) / nx)[:, None] * jnp.ones(
        (1, 4)
    )
    result = plan.simulate(
        speed,
        acquisition,
        jnp.zeros((steps, 1)),
        initial_state=plan.initial_state(pressure),
    )
    phase = 2 * np.arcsin(speed * dt * np.sin(mode * np.pi / (2 * nx)))
    np.testing.assert_allclose(
        result.final_state.pressure,
        pressure * np.cos(steps * phase),
        atol=4e-6,
        rtol=2e-5,
    )
    continuum = pressure * np.cos(steps * dt * speed * mode * np.pi / nx)
    assert float(jnp.max(jnp.abs(result.final_state.pressure - continuum))) < 0.02


def test_cfl_preparation_and_dynamic_material_fail_closed():
    grid = AcousticGrid((7, 7), (1.0, 1.0))
    with pytest.raises(ValueError, match="CFL"):
        ConstantDensityAcousticPlan(grid, 0.8, 2, 1.0)
    plan, acquisition, rates = _survey(steps=2)
    forward = eqx.filter_jit(
        lambda speed: plan.simulate(speed, acquisition, rates).traces.values
    )
    for speed in (0.0, -1.0, 2.1, jnp.nan):
        with pytest.raises(
            (ValueError, eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError),
            match="wavespeed",
        ):
            jax.block_until_ready(forward(speed))


def test_full_step_block_and_scheduled_discrete_adjoint_equivalence():
    plan, acquisition, rates = _survey()
    weights = jnp.linspace(-0.3, 1.2, 2 * (plan.step_count + 1)).reshape((2, -1))
    state = plan.initial_state()
    state_bytes = sum(
        value.size * value.dtype.itemsize
        for value in jax.tree.leaves(state)
        if eqx.is_array(value)
    )
    schedule = prepare_replay_schedule(
        plan.step_count,
        state_bytes,
        AdaptiveReplayPreparationPolicy(3 * state_bytes, 2 * plan.step_count),
    )

    def action(speed, mode, **kwargs):
        return plan.simulate(
            speed, acquisition, rates, replay=mode, **kwargs
        ).traces.values

    def loss(speed, mode, **kwargs):
        return jnp.sum(weights * action(speed, mode, **kwargs))

    speed = jnp.full(plan.grid.shape, 1.5)
    full = action(speed, "full")
    gradient = jax.grad(loss)(speed, "full")
    for mode, options in (
        ("step", {}),
        ("block", {"block_size": 7}),
        ("scheduled", {"schedule": schedule}),
    ):
        np.testing.assert_allclose(
            action(speed, mode, **options), full, atol=2e-7, rtol=3e-6
        )
        np.testing.assert_allclose(
            jax.grad(loss)(speed, mode, **options), gradient, atol=2e-7, rtol=3e-5
        )
    direction = jnp.cos(jnp.arange(speed.size).reshape(speed.shape))
    tangent = jax.jvp(
        lambda value: action(value, "block", block_size=7), (speed,), (direction,)
    )[1]
    np.testing.assert_allclose(
        jnp.sum(weights * tangent), jnp.sum(gradient * direction), atol=2e-7, rtol=3e-5
    )
    epsilon = 1e-3
    finite_difference = (
        loss(speed + epsilon * direction, "full")
        - loss(speed - epsilon * direction, "full")
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        finite_difference, jnp.sum(gradient * direction), atol=3e-6, rtol=0.02
    )


def test_checkpoint_continuation_preserves_split_absorber_state():
    plan, acquisition, _ = _survey(steps=9)
    full_plan = ConstantDensityAcousticPlan(
        plan.grid,
        plan.time_step,
        18,
        plan.maximum_wavespeed,
        plan.density,
        absorber_cells=plan.absorber_cells,
    )
    x, y = jnp.meshgrid(*plan.grid.axis_nodes(), indexing="ij")
    initial_pressure = jnp.exp(-((x - 0.8) ** 2 + (y - 1.1) ** 2) / 0.1)
    first = plan.simulate(
        1.5,
        acquisition,
        jnp.zeros((9, 1)),
        initial_state=plan.initial_state(initial_pressure),
    )
    checkpoint = plan.checkpoint(first.final_state, 1.5)
    resumed = plan.restart(
        checkpoint, acquisition, jnp.zeros((9, 1)), replay="block", block_size=4
    )
    full = full_plan.simulate(
        1.5,
        acquisition,
        jnp.zeros((18, 1)),
        initial_state=full_plan.initial_state(initial_pressure),
    )
    np.testing.assert_allclose(
        resumed.final_state.split_pressure,
        full.final_state.split_pressure,
        atol=1e-7,
        rtol=1e-6,
    )
    for resumed_velocity, full_velocity in zip(
        resumed.final_state.velocity, full.final_state.velocity, strict=True
    ):
        np.testing.assert_allclose(resumed_velocity, full_velocity, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(resumed.traces.support.coordinates[0], 9 * plan.time_step)
    wrong_plan = ConstantDensityAcousticPlan(
        plan.grid, 0.04, 9, 2.0, 2.0, absorber_cells=3
    )
    with pytest.raises(ValueError, match="different prepared plan"):
        wrong_plan.restart(checkpoint, acquisition, jnp.zeros((9, 1)))


def test_split_damping_reduces_late_box_energy_without_perfect_pml_claim():
    grid = AcousticGrid((41, 41), (0.5, 0.5))
    acquisition = SeismicAcquisition(grid, [[10.0, 10.0]], [[10.0, 10.0]])
    wall = ConstantDensityAcousticPlan(grid, 0.15, 240, 1.0, 1.0)
    layer = ConstantDensityAcousticPlan(
        grid, 0.15, 240, 1.0, 1.0, absorber_cells=8, absorber_strength=6.0
    )
    x, y = jnp.meshgrid(*grid.axis_nodes(), indexing="ij")
    pressure = jnp.exp(-((x - 10.0) ** 2 + (y - 10.0) ** 2) / 2.0)
    rates = jnp.zeros((240, 1))
    reflected = wall.simulate(
        1.0, acquisition, rates, initial_state=wall.initial_state(pressure)
    )
    absorbed = layer.simulate(
        1.0, acquisition, rates, initial_state=layer.initial_state(pressure)
    )
    reflected_energy = wall.energy(reflected.final_state, 1.0)
    absorbed_energy = layer.energy(absorbed.final_state, 1.0)
    assert float(absorbed_energy / reflected_energy) < 0.6
    assert float(reflected_energy / wall.energy(wall.initial_state(pressure), 1.0)) > 0.95


def test_masked_native_pressure_likelihood_gradient_and_time_transpose():
    plan, acquisition, rates = _survey()
    true = plan.simulate(1.35, acquisition, rates)
    active = np.ones(true.traces.values.shape, dtype=bool)
    active[0, 7:10] = False
    observed = SampledSeries(
        true.traces.support,
        jnp.where(active, true.traces.values, jnp.nan),
        value_valid=active,
        series_id="gapped-pressure",
    )
    observation = PreparedSeismicObservation(
        plan, acquisition, observed, amplitude_unit=PASCAL
    )
    messages = jnp.linspace(-1.0, 0.7, observation.target.size).reshape(
        observation.target.shape
    )
    np.testing.assert_allclose(
        jnp.sum(observation.sample(true.traces.values) * messages),
        jnp.sum(true.traces.values * observation.transpose(messages)),
        rtol=2e-6,
        atol=1e-8,
    )
    term = SeismicGaussianLikelihood(
        plan, acquisition, observed, rates, 0.03, replay="block", block_size=7
    )
    objective = lambda raw: term.log_prob(
        BoundedAcousticWavespeed(raw, minimum=1.0, maximum=2.0)
    )
    value, derivative = jax.value_and_grad(objective)(jnp.asarray(0.2))
    epsilon = 2e-3
    finite_difference = (objective(0.2 + epsilon) - objective(0.2 - epsilon)) / (
        2 * epsilon
    )
    assert np.isfinite(float(value)) and abs(float(derivative)) > 1e-5
    np.testing.assert_allclose(derivative, finite_difference, rtol=0.03, atol=0.003)
    assert float(term.log_prob(1.35)) > float(term.log_prob(1.7))
    outside_support = SeriesSupport(
        jnp.asarray([0.0, 2.0]), series_shape=(2,), coordinate_name="time_s"
    )
    with pytest.raises(ValueError, match="extrapolate"):
        PreparedSeismicObservation(
            plan,
            acquisition,
            SampledSeries(outside_support, jnp.zeros((2, 2)), series_id="outside"),
        )

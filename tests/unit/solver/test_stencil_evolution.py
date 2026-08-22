#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def _periodic_grid(points=64):
    return phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(
                points,
                periodic=True,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def test_staggered_acoustic_plan_prepares_locations_cfl_and_sensors():
    grid = _periodic_grid()
    acoustic = phx.solver.StaggeredAcousticPlan(
        grid,
        bulk_modulus=4.0,
        density=1.0,
        accuracy_order=4,
        sensor_indices=jnp.asarray([[0], [16], [32]], dtype=jnp.int32),
    ).prepare()
    pressure = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    velocity = (pressure / 2.0,)
    state = acoustic.pack(pressure, velocity)

    drift = acoustic.drift(jnp.asarray(0.0), state, None)

    assert drift.pressure.shape == acoustic.pressure_shape
    assert drift.velocity[0].shape == acoustic.velocity_shapes[0]
    assert acoustic.stable_dt > 0.0
    assert acoustic.stable_dt < (grid.axes[0].nodes[1] - grid.axes[0].nodes[0])
    assert jnp.allclose(acoustic.observe(state), pressure[jnp.asarray([0, 16, 32])])
    assert len(acoustic.discretization.locations) == 2


def test_staggered_leapfrog_has_bounded_energy_drift_on_periodic_medium():
    grid = _periodic_grid()
    acoustic = phx.solver.StaggeredAcousticPlan(
        grid,
        bulk_modulus=1.0,
        density=1.0,
        accuracy_order=4,
    ).prepare()
    pressure = jnp.sin(2.0 * jnp.pi * grid.axes[0].nodes)
    state = acoustic.pack(pressure, (pressure,))
    initial_energy = acoustic.energy(state)
    dt = 0.4 * acoustic.stable_dt
    time = jnp.asarray(0.0)
    for _ in range(100):
        state = acoustic.leapfrog_step(time, state, dt)
        time = time + dt
    final_energy = acoustic.energy(state)

    assert jnp.abs(final_energy - initial_energy) / initial_energy < 2e-2


def test_split_field_pml_profiles_are_nonnegative_and_decay_energy():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(41),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    acoustic = phx.solver.StaggeredAcousticPlan(
        grid,
        bulk_modulus=1.0,
        density=1.0,
        pml=phx.solver.SplitFieldPMLPlan(
            6,
            maximum_attenuation=5.0,
        ),
    ).prepare()
    state = acoustic.pack(
        jnp.ones(grid.cells().shape),
        (jnp.zeros(grid.faces("x").shape),),
    )
    drift = acoustic.drift(jnp.asarray(0.0), state, None)
    pressure_rate, _ = acoustic.unpack(drift)
    stepped = acoustic.leapfrog_step(
        jnp.asarray(0.0),
        state,
        0.25 * acoustic.stable_dt,
    )

    assert acoustic.pml is not None
    assert jnp.all(acoustic.damping >= 0.0)
    assert jnp.min(pressure_rate) < 0.0
    assert jnp.allclose(pressure_rate[10:-10], 0.0)
    assert acoustic.energy(stepped) < acoustic.energy(state)


def test_multidimensional_pml_damps_only_matching_pressure_split():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(21),
            phx.discretization.UniformCellAxisSpec(19),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    acoustic = phx.solver.StaggeredAcousticPlan(
        grid,
        bulk_modulus=1.0,
        density=1.0,
        pml=phx.solver.SplitFieldPMLPlan(
            (4, 0),
            maximum_attenuation=8.0,
        ),
    ).prepare()
    first = jnp.ones(acoustic.pressure_shape)
    state = acoustic.pack_split(
        (first, -first),
        tuple(jnp.zeros(shape) for shape in acoustic.velocity_shapes),
    )

    drift = acoustic.drift(jnp.asarray(0.0), state, None)
    pressure_rate, velocity_rate = acoustic.unpack_split(drift)

    assert jnp.allclose(state.pressure, 0.0)
    assert jnp.min(pressure_rate[0]) < 0.0
    assert jnp.allclose(pressure_rate[0][6:-6], 0.0)
    assert jnp.allclose(pressure_rate[1], 0.0)
    assert all(jnp.allclose(rate, 0.0) for rate in velocity_rate)


def test_split_field_pml_suppresses_outgoing_pulse_reflection():
    points = 96
    width = 16
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(points),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    acoustic = phx.solver.StaggeredAcousticPlan(
        grid,
        bulk_modulus=1.0,
        density=1.0,
        pml=phx.solver.SplitFieldPMLPlan(
            width,
            maximum_attenuation=60.0,
            polynomial_order=3,
        ),
    ).prepare()
    cell_x = grid.cells().coordinates_by_axis[0]
    face_x = grid.faces("x").coordinates_by_axis[0]
    pressure = jnp.exp(-(((cell_x - 0.3) / 0.04) ** 2))
    velocity = jnp.exp(-(((face_x - 0.3) / 0.04) ** 2))
    state = acoustic.pack(pressure, (velocity,))
    step = eqx.filter_jit(acoustic.leapfrog_step)
    dt = 0.7 * float(acoustic.stable_dt)
    time = jnp.asarray(0.0)

    for _ in range(int(1.2 / dt)):
        state = step(time, state, dt)
        time = time + dt

    reflected_amplitude = jnp.max(jnp.abs(state.pressure[width:-width]))
    assert reflected_amplitude < 1.5e-2

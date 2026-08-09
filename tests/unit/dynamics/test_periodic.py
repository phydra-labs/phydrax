#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_map_periodic_orbit_dense_matrix_free_and_floquet_modes():
    matrix = jnp.diag(jnp.asarray([1.2, 0.5, 0.2]))
    layout = phx.dynamics.StateLayout(
        (3,), component_names=("unstable", "stable", "strongly_stable")
    )
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, args: matrix @ state,
        state_layout=layout,
        system_id="linear-map",
    )
    problem = phx.dynamics.analysis.PeriodicOrbitProblem(
        phx.dynamics.DiscreteEvolution(system),
        kind="map",
        num_segments=1,
    )

    dense = phx.dynamics.analysis.solve_periodic_orbit(
        problem,
        jnp.asarray([0.7, -0.5, 0.3]),
        linear_method="dense",
    )
    matrix_free = phx.dynamics.analysis.solve_periodic_orbit(
        problem,
        jnp.asarray([0.7, -0.5, 0.3]),
        linear_method="matrix_free",
        krylov_tolerance=1e-11,
    )
    full = phx.dynamics.analysis.floquet_spectrum(dense, method="full")
    leading = phx.dynamics.analysis.floquet_spectrum(dense, method="leading", leading_k=1)

    assert bool(dense.valid)
    assert bool(matrix_free.valid)
    np.testing.assert_allclose(np.asarray(dense.initial_state), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(matrix_free.initial_state), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(full.multipliers))),
        np.asarray([0.2, 0.5, 1.2]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.abs(np.asarray(leading.multipliers[0])), 1.2, atol=1e-12
    )
    assert full.stability == "unstable"
    assert leading.stability == "unstable"


def test_flow_multiple_shooting_and_floquet_neutral_mode():
    layout = phx.dynamics.StateLayout((2,), component_names=("x", "y"))

    def radial_cycle(time, state, args):
        radius_squared = jnp.sum(state**2)
        growth = 1.0 - radius_squared
        return jnp.asarray(
            [
                growth * state[0] - state[1],
                state[0] + growth * state[1],
            ]
        )

    system = phx.dynamics.ContinuousSystem(
        radial_cycle,
        state_layout=layout,
        system_id="radial-limit-cycle",
    )
    evolution = phx.solver.DiffraxEvolution(
        system,
        rtol=1e-10,
        atol=1e-12,
        max_steps=4096,
    )
    phase = phx.dynamics.analysis.OrthogonalityPhaseCondition(
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([0.0, 1.0]),
        state_layout=layout,
    )
    problem = phx.dynamics.analysis.PeriodicOrbitProblem(
        evolution,
        kind="flow",
        num_segments=3,
        phase_condition=phase,
    )

    orbit = phx.dynamics.analysis.solve_periodic_orbit(
        problem,
        jnp.asarray([1.08, 0.0]),
        initial_period=6.1,
        max_iterations=12,
        rtol=1e-8,
        atol=1e-10,
    )
    floquet = phx.dynamics.analysis.floquet_spectrum(orbit)

    assert bool(orbit.valid)
    np.testing.assert_allclose(np.asarray(orbit.period), 2.0 * np.pi, atol=2e-7)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(orbit.initial_state)), 1.0, atol=2e-7
    )
    assert bool(floquet.valid)
    multipliers = np.sort(np.abs(np.asarray(floquet.multipliers)))
    np.testing.assert_allclose(multipliers[-1], 1.0, atol=2e-6)
    np.testing.assert_allclose(multipliers[0], np.exp(-4.0 * np.pi), rtol=3e-3)
    assert floquet.stability == "stable"
    assert int(floquet.neutral_index) >= 0

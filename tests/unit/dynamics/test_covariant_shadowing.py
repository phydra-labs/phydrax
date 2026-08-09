#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _linear_map(matrix, *, system_id):
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: matrix @ state,
        state_layout=phx.dynamics.StateLayout((matrix.shape[0],)),
        system_id=system_id,
    )
    return phx.dynamics.DiscreteEvolution(system)


def test_covariant_store_and_recompute_modes_match_diagonal_map():
    matrix = jnp.diag(jnp.asarray([2.0, 0.5]))
    evolution = _linear_map(matrix, system_id="clv-diagonal-map")
    grid = phx.dynamics.IterationGrid.from_steps(6, iteration_id="clv-grid")
    options = dict(
        initial_basis=jnp.eye(2),
        qr_interval=1,
        save_every=1,
        backward_discard=1,
    )

    stored = phx.dynamics.analysis.covariant_directions(
        evolution,
        jnp.asarray([0.3, -0.4]),
        grid,
        memory_mode="store",
        **options,
    )
    recomputed = phx.dynamics.analysis.covariant_directions(
        evolution,
        jnp.asarray([0.3, -0.4]),
        grid,
        memory_mode="recompute",
        **options,
    )

    assert bool(stored.valid)
    assert not bool(stored.converged)
    assert int(stored.status) == (
        phx.dynamics.analysis.COVARIANT_INSUFFICIENT_BACKWARD_DEPTH
    )
    assert float(jnp.max(stored.backward_convergence_drift[0])) < float(
        jnp.max(stored.backward_convergence_drift[-2])
    )
    assert stored.stored_frame_count == 7
    assert recomputed.stored_frame_count == 0
    assert recomputed.tangent_evaluations > stored.tangent_evaluations
    np.testing.assert_allclose(
        np.abs(np.asarray(stored.directions[:-1])),
        np.broadcast_to(np.eye(2), (6, 2, 2)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(recomputed.directions), np.asarray(stored.directions), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(stored.local_growth_rates[:-1]),
        np.broadcast_to(np.log([2.0, 0.5]), (6, 2)),
        atol=1e-12,
    )


def test_shadowing_boundary_reports_exact_inhomogeneous_tangent_candidate():
    evolution = _linear_map(jnp.asarray([[0.8]]), system_id="shadowing-map")
    grid = phx.dynamics.IterationGrid.from_steps(5, iteration_id="shadowing-grid")
    trajectory = phx.dynamics.evolve(evolution, jnp.asarray([2.0]), grid)
    problem = phx.dynamics.analysis.ShadowingSensitivityProblem(
        evolution,
        lambda state, source, target, args: jnp.ones((1,)),
        lambda coordinate, state, args: state[0],
        parameter_id="offset",
        observable_id="state",
        problem_id="linear-shadowing-boundary",
    )
    tangent = [jnp.asarray([0.0])]
    for _ in range(grid.num_steps):
        tangent.append(0.8 * tangent[-1] + 1.0)
    tangent_path = jnp.stack(tuple(tangent))

    candidate = phx.dynamics.analysis.evaluate_shadowing_candidate(
        problem,
        trajectory,
        tangent_path,
        boundary="free",
    )

    assert bool(candidate.valid)
    assert not candidate.boundary_enforced
    np.testing.assert_allclose(np.asarray(candidate.defects), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(candidate.observable_directional),
        np.asarray(tangent_path[:, 0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(candidate.mean_directional_response),
        np.mean(np.asarray(tangent_path[:, 0])),
        atol=1e-12,
    )
    assert candidate.least_squares_residual().shape == (grid.num_steps,)

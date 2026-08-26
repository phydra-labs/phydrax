import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_relative_equilibrium_residual_is_continuation_ready():
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64)

    def generator(state):
        return jnp.asarray([-state[1], state[0]])

    problem = phx.dynamics.analysis.RelativeEquilibriumProblem(
        lambda state, args: 2.0 * generator(state),
        (generator,),
        (lambda state, args: state[0] - 1.0,),
        space,
    )
    unknown = problem.pack(jnp.asarray([1.0, 0.0]), jnp.asarray([2.0]))
    residual = problem.as_nonlinear_problem().residual(unknown)

    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-12)


def test_edge_tracking_bisects_opposite_outcomes():
    layout = phx.dynamics.StateLayout((1,))
    system = phx.dynamics.DiscreteSystem(
        lambda step, state, args: state,
        state_layout=layout,
        system_id="identity-edge-map",
    )
    problem = phx.dynamics.analysis.EdgeTrackingProblem(
        phx.dynamics.DiscreteEvolution(system),
        lambda coordinate, state, args: state[0],
        0.0,
        1.0,
    )
    result = phx.dynamics.analysis.track_basin_edge(
        problem,
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
    )

    assert bool(result.valid)
    assert bool(result.converged)
    assert int(result.status) == phx.dynamics.analysis.EDGE_SUCCESS
    assert jnp.abs(result.edge_state[0]) < 1e-8
    assert result.upper_parameter - result.lower_parameter < 1e-8


def test_recurrence_seed_candidates_select_temporally_separated_minima():
    layout = phx.dynamics.StateLayout((1,))
    trajectory = phx.dynamics.TrajectoryData(
        jnp.arange(6.0),
        jnp.asarray([[0.0], [1.0], [0.1], [2.0], [0.05], [3.0]]),
        state_layout=layout,
        source_id="recurrence-candidates",
    )
    candidates = phx.dynamics.analysis.recurrence_seed_candidates(
        trajectory,
        2,
        minimum_separation=2,
    )

    assert jnp.all(candidates.valid)
    np.testing.assert_array_equal(np.asarray(candidates.source_indices), [0, 2])
    np.testing.assert_array_equal(np.asarray(candidates.target_indices), [4, 4])
    np.testing.assert_allclose(np.asarray(candidates.periods), [4.0, 2.0])
    np.testing.assert_allclose(np.asarray(candidates.distances), [0.05, 0.05])

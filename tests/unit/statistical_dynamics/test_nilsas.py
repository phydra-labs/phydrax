#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _affine_shadowing_case(*, memory_mode):
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: 0.5 * state + args["offset"],
        state_layout=phx.dynamics.StateLayout((1,)),
        system_id="adjoint-contracting-affine-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    args = {"offset": jnp.asarray([1.0]), "unused": jnp.asarray(0.3)}
    grid = phx.dynamics.IterationGrid.from_steps(
        100,
        iteration_id="adjoint-contracting-grid",
    )
    trajectory = phx.dynamics.evolve(
        evolution,
        jnp.asarray([2.0]),
        grid,
        args=args,
    )
    problem = phx.dynamics.analysis.ShadowingSensitivityProblem(
        evolution,
        lambda coordinate, state, parameters: state[0],
        parameter_id="affine-offset",
        observable_id="state-average",
        problem_id="adjoint-contracting-affine-shadowing",
    )
    result = (
        phx.statistical_dynamics.NILSASPlan(
            1,
            0,
            0,
            10,
            10,
            memory_mode=memory_mode,
        )
        .prepare(problem, trajectory, args=args)
        .solve()
    )
    return result


def test_nilsas_store_and_recompute_match_exact_discrete_gradient():
    stored = _affine_shadowing_case(memory_mode="store")
    recomputed = _affine_shadowing_case(memory_mode="recompute")
    expected = 2.0 - 4.0 * (1.0 - 0.5**101) / 101.0

    assert bool(stored.successful)
    assert bool(recomputed.successful)
    assert stored.adjoint_shadowing_path.shape == (101, 1)
    assert recomputed.adjoint_shadowing_path.shape == (0, 1)
    assert int(stored.parameter_action_count) == 100
    assert int(stored.replay_action_count) == 0
    assert int(recomputed.replay_action_count) == 100
    np.testing.assert_allclose(
        stored.parameter_gradient["offset"],
        expected,
        atol=1e-12,
    )
    np.testing.assert_allclose(stored.parameter_gradient["unused"], 0.0, atol=1e-12)
    for name in stored.parameter_gradient:
        np.testing.assert_allclose(
            recomputed.parameter_gradient[name],
            stored.parameter_gradient[name],
            atol=1e-12,
        )


def test_nilsas_enforces_declared_flow_neutral_constraint():
    matrix = jnp.diag(jnp.asarray([0.8, 0.5]))
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: matrix @ state + args,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="flow-neutral-affine-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    args = jnp.asarray([0.0, 1.0])
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 1.0, 5),
        time_id="flow-neutral-grid",
    )
    trajectory = phx.dynamics.evolve(
        evolution,
        jnp.asarray([0.0, 2.0]),
        grid,
        args=args,
    )
    problem = phx.dynamics.analysis.ShadowingSensitivityProblem(
        evolution,
        lambda coordinate, state, parameters: state[1],
        parameter_id="flow-offset",
        observable_id="second-state-average",
        problem_id="flow-neutral-shadowing",
        neutral_direction=lambda coordinate, state, parameters: jnp.asarray([1.0, 0.0]),
        time_dilation="flow",
    )
    result = (
        phx.statistical_dynamics.NILSASPlan(
            2,
            0,
            1,
            2,
            2,
            regularization=1.0e-12,
        )
        .prepare(
            problem,
            trajectory,
            args=args,
            terminal_basis=jnp.asarray([[1.0], [0.0]]),
        )
        .solve()
    )

    assert bool(result.successful)
    assert result.approximation == "finite-horizon-time-discrete-flow-adjoint-shadowing"
    np.testing.assert_allclose(result.neutral_constraint_residual, 0.0, atol=1e-10)

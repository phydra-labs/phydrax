#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _affine_shadowing_case(*, steps: int):
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: 0.5 * state + args,
        state_layout=phx.dynamics.StateLayout((1,)),
        system_id="contracting-affine-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    args = jnp.asarray([1.0])
    grid = phx.dynamics.IterationGrid.from_steps(
        steps,
        iteration_id=f"contracting-grid-{steps}",
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
        problem_id="contracting-affine-shadowing",
    )
    return problem, trajectory, args


def test_nilss_solves_matrix_free_parameter_directional_gradient():
    problem, trajectory, args = _affine_shadowing_case(steps=100)
    plan = phx.statistical_dynamics.NILSSPlan(1, 0, 0, 10, 10)

    result = plan.prepare(
        problem,
        trajectory,
        jnp.asarray([1.0]),
        args=args,
    ).solve()

    expected = 2.0 - 4.0 * (1.0 - 0.5**101) / 101.0
    assert bool(result.successful)
    assert result.segment_coefficients.shape == (10, 0)
    assert result.shadowing_tangent.shape == (101, 1)
    assert int(result.tangent_evaluations) == 100
    assert int(result.parameter_action_count) == 100
    np.testing.assert_allclose(result.continuity_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.candidate.defects, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.directional_gradient, expected, atol=1e-12)


def test_nilss_projects_flow_neutral_direction_and_records_time_dilation():
    matrix = jnp.diag(jnp.asarray([1.0, 0.5]))
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: matrix @ state + args,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="nilss-flow-neutral-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    args = jnp.zeros(2)
    trajectory = phx.dynamics.evolve(
        evolution,
        jnp.zeros(2),
        phx.dynamics.TimeGrid(
            jnp.linspace(0.0, 1.0, 5),
            time_id="nilss-flow-neutral-grid",
        ),
        args=args,
    )
    problem = phx.dynamics.analysis.ShadowingSensitivityProblem(
        evolution,
        lambda coordinate, state, parameters: state[1],
        parameter_id="flow-offset",
        observable_id="second-state-average",
        problem_id="nilss-flow-neutral-shadowing",
        neutral_direction=lambda coordinate, state, parameters: jnp.asarray([1.0, 0.0]),
        time_dilation="flow",
    )

    result = (
        phx.statistical_dynamics.NILSSPlan(2, 0, 0, 2, 2)
        .prepare(
            problem,
            trajectory,
            jnp.ones(2),
            args=args,
        )
        .solve()
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.time_dilation, -1.0, atol=1e-12)
    np.testing.assert_allclose(
        result.candidate.neutral_inner_product,
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.candidate.defects, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.directional_gradient, 1.296875, atol=1e-12)


def test_nilss_resource_and_rank_preflight_refuse_unsupported_runs():
    problem, trajectory, args = _affine_shadowing_case(steps=4)
    with pytest.raises(MemoryError, match="maximum_retained_bytes"):
        phx.statistical_dynamics.NILSSPlan(
            1,
            0,
            0,
            2,
            2,
            maximum_retained_bytes=1,
        ).prepare(
            problem,
            trajectory,
            jnp.asarray([1.0]),
            args=args,
        )

    rank_system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, parameters: state + parameters,
        state_layout=phx.dynamics.StateLayout((2,)),
        system_id="nilss-rank-refusal-map",
    )
    rank_evolution = phx.dynamics.DiscreteEvolution(rank_system)
    rank_args = jnp.zeros(2)
    rank_trajectory = phx.dynamics.evolve(
        rank_evolution,
        jnp.zeros(2),
        phx.dynamics.IterationGrid.from_steps(4, iteration_id="nilss-rank-grid"),
        args=rank_args,
    )
    rank_problem = phx.dynamics.analysis.ShadowingSensitivityProblem(
        rank_evolution,
        lambda coordinate, state, parameters: jnp.sum(state),
        parameter_id="rank-offset",
        observable_id="sum",
        problem_id="nilss-rank-refusal",
    )
    with pytest.raises(ValueError, match="lost numerical rank"):
        phx.statistical_dynamics.NILSSPlan(2, 2, 2, 2, 2).prepare(
            rank_problem,
            rank_trajectory,
            jnp.ones(2),
            args=rank_args,
            initial_basis=jnp.ones((2, 2)),
        )

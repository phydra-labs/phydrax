#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax.numpy as jnp

import phydrax as phx
from phydrax.optim._programming import _mixed_integer as mip


def test_failed_node_relaxation_invalidates_optimal_tree_with_incumbent(monkeypatch):
    program = phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([1.0]),
            bounds=phx.optim.Bounds(0.0, 1.0),
        ),
        binary_indices=(0,),
    )
    failed_result = SimpleNamespace(
        primal=jnp.asarray([0.5]),
        objective=jnp.asarray(jnp.nan),
        successful=jnp.asarray(False),
        status=jnp.asarray(
            int(phx.optim.ConvexProgramStatus.ITERATION_LIMIT), dtype=jnp.int32
        ),
    )
    incumbent_result = SimpleNamespace(
        primal=jnp.asarray([0.0]),
        objective=jnp.asarray(0.0),
        successful=jnp.asarray(True),
        status=jnp.asarray(int(phx.optim.ConvexProgramStatus.OPTIMAL), dtype=jnp.int32),
    )
    relaxations = iter((failed_result, incumbent_result))

    monkeypatch.setattr(mip, "prepare_convex_template", lambda *_: object())
    monkeypatch.setattr(mip, "bind_convex_numeric", lambda _, numeric: numeric)
    monkeypatch.setattr(
        mip,
        "solve_prepared_convex_program",
        lambda _: SimpleNamespace(result=next(relaxations)),
    )

    def branch_with_failed_node(problem, /, *, policy):
        del policy
        failed_node = problem.root()
        problem._solve(failed_node)
        incumbent = problem.root()
        problem._solve(incumbent)
        return SimpleNamespace(
            incumbent=incumbent,
            objective=jnp.asarray(0.0),
            global_lower_bound=jnp.asarray(0.0),
            absolute_gap=jnp.asarray(0.0),
            relative_gap=jnp.asarray(0.0),
            explored_nodes=jnp.asarray(2, dtype=jnp.int32),
            pruned_nodes=jnp.asarray(1, dtype=jnp.int32),
            frontier_size=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.asarray(
                int(phx.optim.MixedIntegerStatus.OPTIMAL), dtype=jnp.int32
            ),
        )

    monkeypatch.setattr(mip, "branch_and_bound", branch_with_failed_node)

    result = phx.optim.solve_mixed_integer_program(program)

    assert result.status == phx.optim.MixedIntegerStatus.RELAXATION_FAILURE
    assert result.integral
    assert jnp.array_equal(result.primal, incumbent_result.primal)
    assert result.relaxation_result is incumbent_result
    assert not result.successful


def _binary_implications(power_limit):
    return phx.optim.MixedIntegerProgram(
        phx.optim.LinearProgram(
            jnp.asarray([0.0, 0.0, 1.0]),
            inequality_matrix=jnp.asarray(
                [
                    [1.0, -1.0, 0.0],
                    [0.0, 0.5, -1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            inequality_rhs=jnp.asarray([0.0, 0.0, power_limit]),
            bounds=phx.optim.Bounds(
                jnp.asarray([1.0, 0.0, 0.0]),
                jnp.asarray([1.0, 1.0, jnp.inf]),
            ),
        ),
        binary_indices=(0, 1),
    )


def test_binary_implication_contradiction_has_original_coordinate_farkas_proof():
    program = _binary_implications(0.0)
    certificate = mip._linear_bound_certificate(program.relaxation, 1e-8)
    assert certificate is not None
    assert certificate.valid
    assert certificate.objective < -1e-8
    assert certificate.residual_norm <= 1e-8
    result = phx.optim.solve_mixed_integer_program(program)
    assert result.status == phx.optim.MixedIntegerStatus.INFEASIBLE
    assert not result.successful


def test_near_feasible_binary_implications_are_not_pruned_without_a_proof():
    program = _binary_implications(0.5 + 1e-6)
    assert mip._linear_bound_certificate(program.relaxation, 1e-8) is None
    result = phx.optim.solve_mixed_integer_program(program)
    assert result.successful
    assert jnp.allclose(result.primal, jnp.asarray([1.0, 1.0, 0.5]), atol=2e-6)

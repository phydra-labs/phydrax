#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _solve(space, costs, method):
    return phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            space, jnp.asarray(costs, dtype=float)
        ),
        method,
    )


def test_branch_and_bound_set_packing_solves_conflicts_and_certifies():
    space = phx.combinatorial.SetPackingSpace(
        jnp.asarray(
            [
                [True, True],
                [True, False],
                [False, True],
            ]
        )
    )
    result = _solve(
        space,
        [-5.0, -3.0, -3.0],
        phx.combinatorial.BranchAndBoundSetPacking(),
    )

    np.testing.assert_array_equal(result.decision.selected, [False, True, True])
    np.testing.assert_allclose(result.objective_value, -6.0)
    assert result.status == int(phx.combinatorial.CombinatorialStatus.OPTIMAL)
    assert result.certificate.optimality_proven
    assert result.certificate.absolute_gap == 0.0
    assert result.provenance.exact


def test_branch_and_bound_ties_infeasibility_and_budget_evidence_are_explicit():
    tied_space = phx.combinatorial.SetPackingSpace(
        jnp.asarray([[True], [True]]),
        minimum_selected=1,
        maximum_selected=1,
    )
    tied = _solve(
        tied_space,
        [-1.0, -1.0],
        phx.combinatorial.BranchAndBoundSetPacking(),
    )
    np.testing.assert_array_equal(tied.decision.selected, [True, False])
    assert tied.certificate.tie_available
    assert tied.certificate.tie_margin == 0.0

    infeasible_space = phx.combinatorial.SetPackingSpace(
        jnp.asarray([[True], [True]]),
        minimum_selected=2,
        maximum_selected=2,
    )
    infeasible = _solve(
        infeasible_space,
        [-1.0, -1.0],
        phx.combinatorial.BranchAndBoundSetPacking(),
    )
    assert infeasible.status == int(phx.combinatorial.CombinatorialStatus.INFEASIBLE)
    assert not infeasible.valid

    budget_space = phx.combinatorial.SetPackingSpace(jnp.zeros((3, 0), dtype=bool))
    budgeted = _solve(
        budget_space,
        [-1.0, -1.0, -1.0],
        phx.combinatorial.BranchAndBoundSetPacking(maximum_nodes=4),
    )
    assert budgeted.status == int(
        phx.combinatorial.CombinatorialStatus.MAXIMUM_STEPS_REACHED
    )
    assert budgeted.valid
    assert not budgeted.certificate.optimality_proven
    assert budgeted.certificate.gap_available
    np.testing.assert_array_equal(budgeted.decision.selected, [True, True, True])


def test_greedy_set_packing_distinguishes_feasible_and_certified_results():
    conflicted = phx.combinatorial.SetPackingSpace(
        jnp.asarray(
            [
                [True, True],
                [True, False],
                [False, True],
            ]
        )
    )
    heuristic = _solve(
        conflicted,
        [-5.0, -3.0, -3.0],
        phx.combinatorial.GreedySetPacking(),
    )
    np.testing.assert_array_equal(heuristic.decision.selected, [True, False, False])
    assert heuristic.status == int(phx.combinatorial.CombinatorialStatus.FEASIBLE)
    assert heuristic.valid
    assert not heuristic.certificate.optimality_proven
    assert heuristic.certificate.absolute_gap > 0.0
    assert not heuristic.provenance.exact

    independent = phx.combinatorial.SetPackingSpace(jnp.eye(3, dtype=bool))
    certified = _solve(
        independent,
        [-3.0, 1.0, -2.0],
        phx.combinatorial.GreedySetPacking(),
    )
    np.testing.assert_array_equal(certified.decision.selected, [True, False, True])
    assert certified.status == int(phx.combinatorial.CombinatorialStatus.OPTIMAL)
    assert certified.certificate.optimality_proven
    assert not certified.provenance.exact


def test_set_packing_outputs_have_stopped_ordinary_gradients():
    space = phx.combinatorial.SetPackingSpace(jnp.eye(2, dtype=bool))
    method = phx.combinatorial.BranchAndBoundSetPacking()

    gradient = jax.grad(
        lambda costs: (
            phx.combinatorial.solve_combinatorial(
                phx.combinatorial.LinearCombinatorialProblem(space, costs), method
            ).objective_value
        )
    )(jnp.asarray([-2.0, 1.0]))

    np.testing.assert_array_equal(gradient, jnp.zeros((2,)))

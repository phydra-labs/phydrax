#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _brute_assignment(costs, valid):
    rows, columns = costs.shape
    candidates = [
        assignment
        for assignment in permutations(range(columns), rows)
        if all(valid[row, column] for row, column in enumerate(assignment))
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda assignment: (
            sum(costs[row, column] for row, column in enumerate(assignment)),
            assignment,
        ),
    )


def test_hungarian_matches_enumeration_for_signed_rectangular_problem():
    costs = np.asarray(
        [
            [4.0, -1.0, 3.0, 8.0],
            [2.0, 0.0, 5.0, 7.0],
            [3.0, 2.0, -2.0, 4.0],
        ]
    )
    valid = np.asarray(
        [
            [True, True, True, False],
            [True, True, False, True],
            [True, True, True, True],
        ]
    )
    expected = _brute_assignment(costs, valid)
    space = phx.combinatorial.BipartiteAssignmentSpace(3, 4, valid=valid)
    result = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(space, jnp.asarray(costs)),
        phx.combinatorial.HungarianAssignment(),
    )

    assert expected is not None
    np.testing.assert_array_equal(result.decision.columns, jnp.asarray(expected))
    np.testing.assert_allclose(
        result.objective_value,
        sum(costs[row, column] for row, column in enumerate(expected)),
    )
    np.testing.assert_allclose(result.certificate.absolute_gap, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.certificate.dual_residual, 0.0, atol=1e-12)
    assert result.certificate.optimality_proven
    assert result.status == int(phx.combinatorial.CombinatorialStatus.OPTIMAL)


def test_hungarian_is_batched_jittable_and_breaks_ties_canonically():
    space = phx.combinatorial.BipartiteAssignmentSpace(2, 3)
    costs = jnp.asarray(
        [
            [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[5.0, 1.0, 2.0], [1.0, 4.0, 3.0]],
        ]
    )
    method = phx.combinatorial.HungarianAssignment()
    result = jax.jit(
        lambda value: phx.combinatorial.solve_combinatorial(
            phx.combinatorial.LinearCombinatorialProblem(space, value),
            method,
        )
    )(costs)

    np.testing.assert_array_equal(result.decision.columns, jnp.asarray([[0, 1], [1, 0]]))
    np.testing.assert_allclose(result.objective_value, jnp.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(jnp.sum(result.features, axis=-1), jnp.ones((2, 2)))
    assert bool(result.all_success)


def test_hungarian_reports_structural_and_mask_infeasibility():
    method = phx.combinatorial.HungarianAssignment()
    too_many_rows = phx.combinatorial.BipartiteAssignmentSpace(3, 2)
    structural = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            too_many_rows,
            jnp.zeros((3, 2)),
        ),
        method,
    )
    hall_space = phx.combinatorial.BipartiteAssignmentSpace(
        2,
        2,
        valid=jnp.asarray([[True, False], [True, False]]),
    )
    hall = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            hall_space,
            jnp.zeros((2, 2)),
        ),
        method,
    )

    for result in (structural, hall):
        assert result.status == int(phx.combinatorial.CombinatorialStatus.INFEASIBLE)
        assert not result.valid
        np.testing.assert_array_equal(
            result.decision.columns, -jnp.ones_like(result.decision.columns)
        )

    finite_space = phx.combinatorial.BipartiteAssignmentSpace(2, 2)
    nonfinite = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            finite_space,
            jnp.asarray([[jnp.nan, 0.0], [0.0, 1.0]]),
        ),
        method,
    )
    assert nonfinite.status == int(phx.combinatorial.CombinatorialStatus.NONFINITE_INPUT)
    np.testing.assert_array_equal(
        nonfinite.decision.columns,
        -jnp.ones((2,), dtype=jnp.int32),
    )
    np.testing.assert_array_equal(nonfinite.features, jnp.zeros((2, 2)))


def test_assignment_space_audits_duplicate_and_forbidden_columns():
    space = phx.combinatorial.BipartiteAssignmentSpace(
        2,
        3,
        valid=jnp.asarray([[True, True, True], [True, False, True]]),
    )

    duplicate = space.audit(phx.combinatorial.AssignmentDecision(jnp.asarray([0, 0])))
    forbidden = space.audit(phx.combinatorial.AssignmentDecision(jnp.asarray([0, 1])))
    feasible = space.audit(phx.combinatorial.AssignmentDecision(jnp.asarray([1, 2])))

    assert not duplicate.feasible
    assert not forbidden.feasible
    assert feasible.feasible

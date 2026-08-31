#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _space(source, target, vertices, balances, capacities, *, valid=None):
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(source, dtype=jnp.int32),
        jnp.asarray(target, dtype=jnp.int32),
        source_size=vertices,
        target_size=vertices,
        valid=valid,
    )
    return phx.combinatorial.CapacitatedFlowSpace(
        relation,
        jnp.asarray(balances, dtype=jnp.int32),
        jnp.asarray(capacities, dtype=jnp.int32),
    )


def _solve(space, costs, method=None):
    selected = phx.combinatorial.CycleCancelingMinCostFlow() if method is None else method
    return phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            space, jnp.asarray(costs, dtype=float)
        ),
        selected,
    )


def test_capacitated_min_cost_flow_builds_disjoint_multi_object_paths():
    space = _space(
        [0, 1, 0, 2],
        [1, 3, 2, 3],
        4,
        [2, 0, 0, -2],
        [1, 1, 1, 1],
    )
    result = _solve(space, [1.0, 1.0, 2.0, 1.0])

    np.testing.assert_array_equal(result.decision.flow, [1, 1, 1, 1])
    np.testing.assert_allclose(result.objective_value, 5.0)
    assert result.status == int(phx.combinatorial.CombinatorialStatus.OPTIMAL)
    assert result.certificate.feasible
    assert result.certificate.optimality_proven
    assert result.certificate.dual_available
    np.testing.assert_allclose(result.certificate.dual_residual, 0.0, atol=1e-6)


def test_min_cost_flow_uses_deterministic_edge_ties_and_negative_cycles():
    tied_space = _space([0, 0], [1, 1], 2, [1, -1], [1, 1])
    tied = _solve(tied_space, [0.0, 0.0])
    np.testing.assert_array_equal(tied.decision.flow, [1, 0])
    assert tied.provenance.tie_policy == "lowest-vertex-then-lowest-residual-edge"

    circulation_space = _space([0, 1], [1, 0], 2, [0, 0], [1, 1])
    circulation = _solve(circulation_space, [-2.0, 0.0])
    np.testing.assert_array_equal(circulation.decision.flow, [1, 1])
    np.testing.assert_allclose(circulation.objective_value, -2.0)
    assert circulation.certificate.optimality_proven


def test_min_cost_flow_reports_infeasible_and_budget_exhausted_states():
    infeasible_space = _space([0], [1], 3, [1, 0, -1], [1])
    infeasible = _solve(infeasible_space, [0.0])
    assert infeasible.status == int(phx.combinatorial.CombinatorialStatus.INFEASIBLE)
    assert not infeasible.valid
    assert not infeasible.certificate.feasible

    budget_space = _space(
        [0, 2, 3],
        [1, 3, 2],
        4,
        [1, -1, 0, 0],
        [1, 1, 1],
    )
    budgeted = _solve(
        budget_space,
        [0.0, -1.0, 0.0],
        phx.combinatorial.CycleCancelingMinCostFlow(maximum_iterations=1),
    )
    assert budgeted.status == int(
        phx.combinatorial.CombinatorialStatus.MAXIMUM_STEPS_REACHED
    )
    assert budgeted.valid
    assert budgeted.certificate.feasible
    assert not budgeted.certificate.optimality_proven
    np.testing.assert_array_equal(budgeted.decision.flow, [1, 0, 0])


def test_min_cost_flow_masks_edges_and_stops_ordinary_gradients():
    space = _space(
        [0, 0],
        [1, 1],
        2,
        [1, -1],
        [1, 100],
        valid=jnp.asarray([True, False]),
    )
    result = _solve(space, [2.0, -100.0])
    np.testing.assert_array_equal(result.decision.flow, [1, 0])

    method = phx.combinatorial.CycleCancelingMinCostFlow()
    gradient = jax.grad(
        lambda costs: (
            phx.combinatorial.solve_combinatorial(
                phx.combinatorial.LinearCombinatorialProblem(space, costs), method
            ).objective_value
        )
    )(jnp.asarray([2.0, -100.0]))
    np.testing.assert_array_equal(gradient, jnp.zeros((2,)))

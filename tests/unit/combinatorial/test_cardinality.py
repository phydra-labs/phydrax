#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _brute(costs, count, valid):
    candidates = [
        candidate
        for candidate in combinations(range(len(costs)), count)
        if all(valid[index] for index in candidate)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (sum(costs[index] for index in candidate), candidate),
    )


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_cardinality_matches_enumeration_and_certifies_boundary(count):
    costs = np.asarray([2.0, -1.0, -1.0, 3.0])
    valid = np.asarray([True, True, True, True])
    expected = _brute(costs, count, valid)
    space = phx.combinatorial.CardinalitySpace(4, count, valid=valid)
    result = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(space, jnp.asarray(costs)),
        phx.combinatorial.StableCardinalityOracle(),
    )

    assert expected is not None
    np.testing.assert_array_equal(result.decision.indices, jnp.asarray(expected))
    np.testing.assert_array_equal(
        result.features,
        jnp.asarray([index in expected for index in range(4)], dtype=float),
    )
    np.testing.assert_allclose(result.objective_value, costs[list(expected)].sum())
    assert result.status == int(phx.combinatorial.CombinatorialStatus.OPTIMAL)
    assert result.certificate.optimality_proven
    if 0 < count < 4:
        assert result.certificate.tie_available
        assert result.certificate.tie_margin >= 0.0
    else:
        assert not result.certificate.tie_available


def test_cardinality_masks_infeasibility_and_batches_under_jit():
    space = phx.combinatorial.CardinalitySpace(
        4,
        2,
        valid=jnp.asarray([True, False, True, True]),
    )
    costs = jnp.asarray(
        [
            [3.0, -100.0, 1.0, 2.0],
            [-2.0, -100.0, 4.0, 1.0],
        ]
    )
    method = phx.combinatorial.StableCardinalityOracle()
    result = jax.jit(
        lambda value: phx.combinatorial.solve_combinatorial(
            phx.combinatorial.LinearCombinatorialProblem(space, value),
            method,
        )
    )(costs)

    np.testing.assert_array_equal(result.decision.indices, jnp.asarray([[2, 3], [0, 3]]))
    np.testing.assert_allclose(result.objective_value, jnp.asarray([3.0, -1.0]))
    assert bool(result.all_success)

    infeasible_space = phx.combinatorial.CardinalitySpace(
        4,
        3,
        valid=jnp.asarray([True, False, True, False]),
    )
    infeasible = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            infeasible_space,
            jnp.arange(4.0),
        ),
        method,
    )
    assert infeasible.status == int(phx.combinatorial.CombinatorialStatus.INFEASIBLE)
    assert not infeasible.valid
    np.testing.assert_array_equal(infeasible.decision.indices, -jnp.ones((3,), dtype=int))

    nonfinite = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            space,
            jnp.asarray([jnp.nan, 0.0, 1.0, 2.0]),
        ),
        method,
    )
    assert nonfinite.status == int(phx.combinatorial.CombinatorialStatus.NONFINITE_INPUT)
    np.testing.assert_array_equal(
        nonfinite.decision.indices,
        -jnp.ones((2,), dtype=jnp.int32),
    )
    np.testing.assert_array_equal(nonfinite.features, jnp.zeros((4,)))


def test_cardinality_rejects_invalid_static_contracts():
    with pytest.raises(ValueError, match="positive"):
        phx.combinatorial.CardinalitySpace(0, 0)
    with pytest.raises(ValueError, match=r"\[0, size\]"):
        phx.combinatorial.CardinalitySpace(3, 4)
    with pytest.raises(ValueError, match="shape"):
        phx.combinatorial.CardinalitySpace(3, 1, valid=jnp.ones((2,), dtype=bool))

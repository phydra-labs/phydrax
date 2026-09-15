#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _quadratic_value(features, target):
    return 0.5 * jnp.sum((features - target) ** 2)


def test_integer_hull_quadratic_matches_exhaustive_cardinality_search():
    target = jnp.asarray([0.2, 0.8, 0.0, 0.4])
    space = phx.combinatorial.CardinalitySpace(4, 2)
    problem = phx.optim.IntegerHullProblem(
        phx.optim.MinimizationProblem(_quadratic_value),
        space,
        args=target,
        convexity=phx.optim.ConvexObjectiveEvidence(
            "construction",
            "squared-euclidean",
        ),
        problem_id="cardinality-quadratic",
    )
    policy = phx.optim.IntegerHullPolicy(
        phx.combinatorial.StableCardinalityOracle(),
        maximum_fw_steps=100,
        fw_tolerance=1e-8,
    )

    result = phx.optim.solve_integer_hull(problem, policy)

    exhaustive = []
    for selected in itertools.combinations(range(4), 2):
        features = jnp.asarray(
            [index in selected for index in range(4)],
            dtype=float,
        )
        exhaustive.append((float(_quadratic_value(features, target)), selected))
    expected_value, expected = min(exhaustive)
    assert result.successful
    np.testing.assert_array_equal(result.decision.indices, expected)
    np.testing.assert_allclose(result.objective, expected_value, atol=1e-8)
    assert result.global_lower_bound <= expected_value + 1e-8
    assert result.certificate.lower_bound_certified
    assert result.work.oracle_calls > 0
    assert result.work.fw_steps > 0


def test_integer_hull_root_with_integral_relaxation_finishes_without_branching():
    target = jnp.asarray([1.0, 0.0, 0.0])
    space = phx.combinatorial.CardinalitySpace(3, 1)
    problem = phx.optim.IntegerHullProblem(
        phx.optim.MinimizationProblem(_quadratic_value),
        space,
        args=target,
        convexity=phx.optim.ConvexObjectiveEvidence(
            "construction",
            "integral-quadratic",
        ),
    )

    result = phx.optim.solve_integer_hull(
        problem,
        phx.optim.IntegerHullPolicy(
            phx.combinatorial.StableCardinalityOracle(),
            fw_tolerance=1e-9,
        ),
    )

    assert result.successful
    assert result.work.explored_nodes == 1
    np.testing.assert_array_equal(result.decision.indices, [0])
    assert result.objective == 0.0

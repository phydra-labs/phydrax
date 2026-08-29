#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _space(*, valid=None):
    return phx.combinatorial.ExplicitDecisionSpace(
        {"label": jnp.asarray([[10], [20], [30]], dtype=jnp.int32)},
        {
            "linear": jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            "offset": jnp.asarray([0.0, 0.25, -0.5]),
        },
        valid=valid,
    )


def test_linear_problem_validates_cost_feature_contract_and_refresh():
    space = _space()
    problem = phx.combinatorial.LinearCombinatorialProblem(
        space,
        {"linear": jnp.asarray([2.0, 1.0]), "offset": jnp.asarray(1.0)},
        problem_id="catalog",
    )
    refreshed = problem.with_costs(
        {"linear": jnp.asarray([1.0, 3.0]), "offset": jnp.asarray(-2.0)}
    )

    assert problem.batch_shape == ()
    assert problem.structure_id == refreshed.structure_id == space.structure_id
    assert problem.problem_id == refreshed.problem_id == "catalog"

    with pytest.raises(ValueError, match="identical PyTree"):
        phx.combinatorial.LinearCombinatorialProblem(
            space,
            {"linear": jnp.ones((2,))},
        )
    with pytest.raises(ValueError, match="feature shape"):
        phx.combinatorial.LinearCombinatorialProblem(
            space,
            {"linear": jnp.ones((3,)), "offset": jnp.asarray(0.0)},
        )
    with pytest.raises(TypeError, match="floating dtype"):
        phx.combinatorial.LinearCombinatorialProblem(
            space,
            {"linear": jnp.ones((2,), dtype=jnp.int32), "offset": jnp.asarray(0)},
        )
    with pytest.raises(ValueError, match="preserve batch shape"):
        problem.with_costs(
            {
                "linear": jnp.ones((2, 2)),
                "offset": jnp.ones((2,)),
            }
        )


def test_explicit_oracle_keeps_decisions_features_and_stable_ties_distinct():
    space = _space()
    problem = phx.combinatorial.LinearCombinatorialProblem(
        space,
        {
            "linear": jnp.asarray([[2.0, 1.0], [2.0, 1.0]]),
            "offset": jnp.asarray([0.0, 1.0]),
        },
    )
    method = phx.combinatorial.ExhaustiveLinearOracle(batch_size=2)
    result = jax.jit(
        lambda declared: phx.combinatorial.solve_combinatorial(declared, method)
    )(problem)

    np.testing.assert_array_equal(result.decision.index, jnp.asarray([1, 1]))
    np.testing.assert_array_equal(
        result.decision.value["label"],
        jnp.asarray([[20], [20]], dtype=jnp.int32),
    )
    np.testing.assert_allclose(
        result.features["linear"],
        jnp.asarray([[0.0, 1.0], [0.0, 1.0]]),
    )
    np.testing.assert_allclose(result.objective_value, jnp.asarray([1.0, 1.25]))
    assert bool(result.all_success)
    assert result.provenance.tie_policy == "lowest-candidate-index"
    assert result.certificate.tie_available.shape == (2,)


def test_explicit_oracle_reports_infeasible_and_nonfinite_instances():
    method = phx.combinatorial.ExhaustiveLinearOracle(batch_size=4)
    infeasible = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            _space(valid=jnp.zeros((3,), dtype=bool)),
            {"linear": jnp.asarray([1.0, 2.0]), "offset": jnp.asarray(0.0)},
        ),
        method,
    )
    nonfinite = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            _space(),
            {"linear": jnp.asarray([jnp.nan, 2.0]), "offset": jnp.asarray(0.0)},
        ),
        method,
    )

    assert infeasible.status == int(phx.combinatorial.CombinatorialStatus.INFEASIBLE)
    assert infeasible.decision.index == -1
    assert not infeasible.valid
    assert nonfinite.status == int(phx.combinatorial.CombinatorialStatus.NONFINITE_INPUT)
    assert not nonfinite.valid


def test_public_combinatorial_exports_are_rooted_in_new_namespace():
    expected = {
        "DAGShortestPath",
        "ExhaustiveLinearOracle",
        "HungarianAssignment",
        "LinearCombinatorialProblem",
        "StableCardinalityOracle",
        "blackbox_solution",
    }

    assert phx.combinatorial is not None
    assert expected <= set(phx.combinatorial.__all__)
    assert "HungarianAssignment" not in phx.optim.__all__

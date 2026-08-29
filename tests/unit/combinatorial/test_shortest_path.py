#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _relation(source, target, vertices, *, valid=None):
    return phx.sparse.EdgeRelation(
        jnp.asarray(source, dtype=jnp.int32),
        jnp.asarray(target, dtype=jnp.int32),
        source_size=vertices,
        target_size=vertices,
        valid=valid,
    )


def test_dag_shortest_path_supports_signed_edges_and_certifies_dual():
    relation = _relation([0, 0, 1, 2, 1], [1, 2, 2, 3, 3], 4)
    space = phx.combinatorial.ShortestPathSpace(relation, 0, 3)
    result = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            space,
            jnp.asarray([2.0, 5.0, -1.0, 1.0, 10.0]),
        ),
        phx.combinatorial.DAGShortestPath(),
    )

    np.testing.assert_array_equal(result.decision.vertices, jnp.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(result.decision.edges, jnp.asarray([0, 2, 3]))
    assert result.decision.length == 4
    np.testing.assert_array_equal(
        result.features,
        jnp.asarray([1.0, 0.0, 1.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(result.objective_value, 2.0)
    np.testing.assert_allclose(result.certificate.absolute_gap, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.certificate.dual_residual, 0.0, atol=1e-12)
    assert result.certificate.optimality_proven


def test_dag_path_batches_ties_masks_and_unreachable_status():
    relation = _relation(
        [0, 0, 1, 2, 0],
        [1, 2, 3, 3, 3],
        4,
        valid=jnp.asarray([True, True, True, True, False]),
    )
    space = phx.combinatorial.ShortestPathSpace(relation, 0, 3)
    costs = jnp.asarray(
        [
            [1.0, 1.0, 1.0, 1.0, -100.0],
            [3.0, 1.0, 1.0, 1.0, -100.0],
        ]
    )
    method = phx.combinatorial.DAGShortestPath()
    result = jax.jit(
        lambda value: phx.combinatorial.solve_combinatorial(
            phx.combinatorial.LinearCombinatorialProblem(space, value),
            method,
        )
    )(costs)

    np.testing.assert_array_equal(
        result.decision.vertices,
        jnp.asarray([[0, 1, 3, -1], [0, 2, 3, -1]]),
    )
    np.testing.assert_allclose(result.objective_value, jnp.asarray([2.0, 2.0]))
    assert bool(result.all_success)

    disconnected_relation = _relation([0], [1], 3)
    disconnected_space = phx.combinatorial.ShortestPathSpace(
        disconnected_relation,
        0,
        2,
    )
    disconnected = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            disconnected_space,
            jnp.asarray([1.0]),
        ),
        method,
    )
    assert disconnected.status == int(phx.combinatorial.CombinatorialStatus.INFEASIBLE)
    assert disconnected.decision.length == 0

    nonfinite = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            space,
            jnp.asarray([jnp.nan, 1.0, 1.0, 1.0, 1.0]),
        ),
        method,
    )
    assert nonfinite.status == int(phx.combinatorial.CombinatorialStatus.NONFINITE_INPUT)
    np.testing.assert_array_equal(
        nonfinite.decision.vertices,
        -jnp.ones((4,), dtype=jnp.int32),
    )
    assert nonfinite.decision.length == 0
    np.testing.assert_array_equal(nonfinite.features, jnp.zeros((5,)))


def test_dag_shortest_path_handles_identity_path_and_rejects_cycles():
    empty = _relation([], [], 1)
    identity_space = phx.combinatorial.ShortestPathSpace(empty, 0, 0)
    identity = phx.combinatorial.solve_combinatorial(
        phx.combinatorial.LinearCombinatorialProblem(
            identity_space,
            jnp.asarray([], dtype=float),
        ),
        phx.combinatorial.DAGShortestPath(),
    )

    np.testing.assert_array_equal(identity.decision.vertices, jnp.asarray([0]))
    np.testing.assert_array_equal(
        identity.decision.edges, jnp.asarray([], dtype=jnp.int32)
    )
    assert identity.decision.length == 1
    assert identity.objective_value == 0.0
    assert identity.valid

    cycle = _relation([0, 1], [1, 0], 2)
    cyclic_space = phx.combinatorial.ShortestPathSpace(cycle, 0, 1)
    with pytest.raises(ValueError, match="acyclic"):
        phx.combinatorial.plan_combinatorial(
            phx.combinatorial.LinearCombinatorialProblem(
                cyclic_space,
                jnp.ones((2,)),
            ),
            phx.combinatorial.DAGShortestPath(),
        )


def test_path_audit_rejects_disconnected_decision():
    relation = _relation([0, 1], [1, 2], 3)
    space = phx.combinatorial.ShortestPathSpace(relation, 0, 2)
    invalid = phx.combinatorial.PathDecision(
        vertices=jnp.asarray([0, 2, -1]),
        edges=jnp.asarray([1, -1]),
        length=jnp.asarray(2),
    )

    assert not space.audit(invalid).feasible

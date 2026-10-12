"""Deterministic sketch oracles, without probabilistic error assertions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def graph() -> phx.sparse.EdgeRelation:
    return phx.sparse.EdgeRelation(
        np.array([0, 1, 2, 0], dtype=np.int32),
        np.array([1, 2, 3, 3], dtype=np.int32),
        source_size=5,
        target_size=5,
    )


def oracle(w: np.ndarray, signs: np.ndarray, *, square_root: bool = True) -> np.ndarray:
    incidence = np.zeros((4, 5), dtype=np.float64)
    for i, (u, v) in enumerate(zip([0, 1, 2, 0], [1, 2, 3, 3], strict=True)):
        incidence[i, u], incidence[i, v] = 1, -1
    lap = incidence.T @ (w[:, None] * incidence)
    rhs = (
        ((np.sqrt(w) if square_root else w)[None, :] * signs)
        @ incidence
        / np.sqrt(signs.shape[0])
    )
    result = np.zeros((signs.shape[0], 5), dtype=np.float64)
    result[:, 1:4] = np.linalg.solve(lap[1:4, 1:4], rhs[:, 1:4].T).T
    return result


@pytest.mark.strict_jax
def test_embedding_oracle_fault_block_components() -> None:
    w = np.array([0.5, 2.0, 3.0, 1.0], dtype=np.float64)
    key = jax.random.key(12)
    k = 7
    signs = np.stack(
        [
            np.asarray(
                jax.random.rademacher(jax.random.fold_in(key, r), (4,), dtype=jnp.float64)
            )
            for r in range(k)
        ]
    )
    expected = oracle(w, signs)
    assert np.max(np.abs(expected - oracle(w, signs, square_root=False))) > 0.1
    for block in (1, 3, k):
        padded = ((k + block - 1) // block) * block
        primal = (k * 5 + 5 * padded + (2 * 3 + 5) * block + 4 * 4 + 5 + 4) * 8
        primal += padded * 17
        budget = primal + (k * 5 + (3 + 4) * padded + (3 * 3 + 5 + 4) * block + 4) * 8
        plan = phx.graph.SketchedEffectiveResistancePlan(
            graph(), sketch_dimension=k, workspace_bytes=budget
        )
        assert plan.block_size == block
        estimate = plan.resource_estimate
        assert all(isinstance(value, int) for value in estimate.values())
        assert estimate["output_bytes"] == k * 5 * 8
        assert estimate["row_assembly_bytes"] == (4 * 4 + 5) * 8
        assert estimate["rhs_bytes"] == 3 * block * 8
        assert estimate["potential_bytes"] == estimate["rhs_bytes"]
        assert estimate["query_block_bytes"] == 5 * block * 8
        assert estimate["conductance_gradient_bytes"] == 4 * 8
        assert estimate["retained_potential_bytes"] == 3 * padded * 8
        assert estimate["retained_sign_bytes"] == 4 * padded * 8
        assert estimate["primal_workspace_bytes"] == primal
        assert estimate["derivative_workspace_bytes"] == budget
        assert estimate["budgeted_workspace_bytes"] == max(
            estimate["primal_workspace_bytes"], estimate["derivative_workspace_bytes"]
        )
        assert estimate["budgeted_workspace_bytes"] <= budget
        if block == 1:
            minimum = estimate["budgeted_workspace_bytes"]
            for insufficient in (minimum - 1, estimate["primal_workspace_bytes"]):
                with pytest.raises(ValueError, match="one RHS"):
                    phx.graph.SketchedEffectiveResistancePlan(
                        graph(), sketch_dimension=k, workspace_bytes=insufficient
                    )
        elif block == 3:
            full = phx.graph.SketchedEffectiveResistancePlan(
                graph(), sketch_dimension=k
            ).resource_estimate
            assert full["primal_workspace_bytes"] <= budget
            assert full["derivative_workspace_bytes"] > budget
        result = plan.embed(jnp.asarray(w), key)
        assert result.status == 0
        np.testing.assert_allclose(result.embedding, expected, atol=1e-8)
        np.testing.assert_array_equal(result.embedding[:, [0, 4]], 0)
        distance = result.resistance(
            jnp.array([0, 1], dtype=jnp.int32), jnp.array([3, 4], dtype=jnp.int32)
        )
        np.testing.assert_allclose(
            distance[0], np.sum((expected[:, 0] - expected[:, 3]) ** 2)
        )
        assert np.isinf(distance[1])


@pytest.mark.strict_jax
def test_fixed_key_gradient_invalid_budget_key() -> None:
    plan = phx.graph.SketchedEffectiveResistancePlan(graph(), sketch_dimension=5)
    key = jax.random.key(4)
    w = np.array([0.5, 2.0, 3.0, 1.0], dtype=np.float64)

    def objective(values: jax.Array) -> jax.Array:
        return plan.embed(values, key).resistance(
            jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32)
        )

    gradient = jax.grad(objective)(jnp.asarray(w))
    finite = np.zeros_like(w)
    for i in range(len(w)):
        delta = np.zeros_like(w)
        delta[i] = 1e-4
        finite[i] = (
            float(objective(jnp.asarray(w + delta)))
            - float(objective(jnp.asarray(w - delta)))
        ) / 2e-4
    np.testing.assert_allclose(gradient, finite, atol=1e-6)
    invalid = plan.embed(jnp.zeros(4, dtype=jnp.float64), key)
    assert invalid.status == phx.graph.EffectiveResistanceStatus.INVALID_CONDUCTANCE
    assert np.all(np.isnan(invalid.embedding))
    assert np.isnan(
        invalid.resistance(
            jnp.asarray(0, dtype=jnp.int32), jnp.asarray(4, dtype=jnp.int32)
        )
    )
    with pytest.raises(ValueError, match="one RHS"):
        phx.graph.SketchedEffectiveResistancePlan(
            graph(), sketch_dimension=5, workspace_bytes=1
        )
    with pytest.raises((TypeError, ValueError), match="key|PRNG"):
        plan.embed(jnp.asarray(w), jax.random.PRNGKey(4))

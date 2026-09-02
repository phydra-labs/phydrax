#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_finite_top_k_pareto_and_landscape_are_index_stable():
    space = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(jnp.asarray([3.0, 1.0, 1.0, 2.0]))
    )
    top = phx.optim.search_finite(
        lambda value: (value, jnp.asarray(True)),
        space,
        phx.optim.FiniteTopK(3),
        landscape=phx.optim.FiniteLandscapePolicy(retain=True),
    )
    assert top.exact
    assert jnp.array_equal(top.flat_indices, jnp.asarray([1, 2, 3]))
    assert jnp.array_equal(top.landscape_evaluated, jnp.ones((4,), dtype=bool))

    pareto = phx.optim.search_finite(
        lambda value: (jnp.stack((value, 4.0 - value)), jnp.asarray(True)),
        space,
        phx.optim.FinitePareto(2, 4),
    )
    direct = pareto.scores[pareto.valid]
    dominates = jnp.all(direct[:, None] <= direct[None, :], axis=-1) & jnp.any(
        direct[:, None] < direct[None, :], axis=-1
    )
    assert not jnp.any(dominates)


def test_mixed_differential_evolution_decodes_domain_members_and_guards_invalid():
    space = phx.optim.DifferentialEvolutionSpace(
        {
            "integer": phx.optim.DifferentialEvolutionInteger(0, 3),
            "category": phx.optim.DifferentialEvolutionCategorical(
                phx.optim.FiniteAxis(jnp.asarray([10.0, 20.0, 40.0]))
            ),
        }
    )
    search = phx.optim.DifferentialEvolutionSearch(8, 2)

    def valid(candidate):
        return candidate["integer"] >= 1

    def objective(candidate):
        return (candidate["integer"] - 2.0) ** 2 + (candidate["category"] != 20.0)

    result = phx.optim.search_differential_evolution(
        objective, space, search, key=jr.key(9), validity=valid
    )
    assert jnp.all(result.population["integer"] >= 0)
    assert jnp.all(result.population["integer"] <= 3)
    assert jnp.all(
        jnp.isin(result.population["category"], jnp.asarray([10.0, 20.0, 40.0]))
    )


def test_sparse_conic_program_retains_relation_and_native_method_capability():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    matrix = phx.sparse.SparseLinearMap(relation, jnp.asarray([1.0, 1.0]))
    program = phx.optim.ConicProgram(
        None,
        jnp.asarray([1.0, 1.0]),
        matrix,
        jnp.asarray([1.0, 1.0]),
        phx.optim.NonnegativeCone(2),
    )
    method = phx.optim.NativeHomogeneousConic()
    assert program.constraint_is_sparse
    assert method.capabilities.sparse
    assert program.constraint_matrix.sparse_storage().nnz == 2


def test_bounded_mixed_integer_program_rejects_unbounded_discrete_roles():
    relaxation = phx.optim.LinearProgram(jnp.asarray([1.0, 0.0]))
    try:
        phx.optim.MixedIntegerProgram(relaxation, integer_indices=(0,))
    except ValueError as error:
        assert "finite bounds" in str(error)
    else:
        raise AssertionError("unbounded integer coordinate was accepted")

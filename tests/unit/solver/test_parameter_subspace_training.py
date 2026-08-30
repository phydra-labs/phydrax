#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


def _output_target(model, *, key=None, iter_=None):
    del key, iter_
    return (model(jnp.asarray([1.0])) - 2.0) ** 2


def _solver():
    base = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        rwf=False,
        key=jr.key(0),
    )
    adapted, _ = phx.nn.parameters.adapt_low_rank(
        base,
        {".layers[0].weight": phx.nn.parameters.LowRankSpec(1)},
        key=jr.key(1),
    )
    model = adapted.add_model_loss(_output_target, label="output_target")
    domain = phx.domain.Interval1d(0.0, 1.0)
    return phx.solver.FunctionalSolver(
        functions={"u": domain.Model("x")(model)},
        terms=(),
    )


def test_functional_solver_optimizes_only_explicit_low_rank_factors():
    solver = _solver()
    subspace = phx.nn.parameters.low_rank_parameter_subspace(solver.functions)
    initial_loss = solver.loss(key=jr.key(2))
    initial_factors = tuple(jax.tree.leaves(subspace.initial))

    trained = solver.solve(
        num_iter=20,
        optim=optax.adamw(5e-2, weight_decay=0.1),
        parameter_subspace=subspace,
        keep_best=False,
        log_every=0,
    )
    final_loss = trained.loss(key=jr.key(2))
    rebased = subspace.rebase(trained.functions)
    final_factors = tuple(jax.tree.leaves(rebased.initial))

    assert final_loss < initial_loss
    assert any(
        not jnp.array_equal(initial, final)
        for initial, final in zip(initial_factors, final_factors, strict=True)
    )
    frozen_equal = eqx.tree_equal(subspace.frozen, rebased.frozen)
    assert bool(frozen_equal)


def test_functional_solver_requires_matching_low_rank_subspace():
    solver = _solver()
    with pytest.raises(ValueError, match="requires an explicit"):
        solver.solve(num_iter=1, optim=optax.sgd(1e-2), log_every=0)

    unrelated = phx.nn.parameters.ParameterSubspace(
        {"weight": jnp.ones((1,))},
        eqx.is_inexact_array,
    )
    with pytest.raises(ValueError, match="does not describe"):
        solver.solve(
            num_iter=1,
            optim=optax.sgd(1e-2),
            parameter_subspace=unrelated,
            log_every=0,
        )


def test_functional_solver_rejects_non_optax_subspace_backend():
    solver = _solver()
    subspace = phx.nn.parameters.low_rank_parameter_subspace(solver.functions)
    with pytest.raises(ValueError, match="unsupported by KFAC"):
        solver.solve(
            num_iter=1,
            optim=phx.optim.kfac(damping=1e-2),
            parameter_subspace=subspace,
            keep_best=False,
            log_every=0,
        )

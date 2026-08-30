#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


_FLOAT64_POLICY = phx.nn.operator.training.OperatorDTypePolicy(
    parameter_dtype="float64",
    compute_dtype="float64",
    reduction_dtype="float64",
)


def _dataset(cases=4, resolution=6):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, resolution),
        quadrature_weights=jnp.full((resolution,), 1.0 / resolution),
    )
    offsets = jnp.arange(cases, dtype=float)[:, None]
    values = offsets + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _adapted_operator(seed=0, *, resolution=6):
    latent = 4
    branch = phx.nn.models.MLP(
        in_size=resolution,
        out_size=latent,
        width_size=5,
        depth=1,
        rwf=False,
        key=jr.key(seed),
    )
    trunk = phx.nn.models.MLP(
        in_size="scalar",
        out_size=latent,
        width_size=5,
        depth=1,
        rwf=False,
        key=jr.key(seed + 1),
    )
    base = phx.nn.operator.architectures.DeepONet(
        branch=phx.nn.operator.architectures.FixedBranchEncoder(branch, latent),
        trunk=trunk,
        coord_dim=1,
        latent_size=latent,
        out_size="scalar",
        in_size=resolution,
        source_key="state",
    )
    paths = phx.nn.parameters.low_rank_sites(base)
    adapted, _ = phx.nn.parameters.adapt_low_rank(
        base,
        {path: phx.nn.parameters.LowRankSpec(1) for path in paths},
        key=jr.key(seed + 100),
    )
    return adapted, phx.nn.parameters.low_rank_parameter_subspace(adapted)


def test_fit_operator_updates_only_low_rank_factor_subspace():
    model, subspace = _adapted_operator()
    initial_factors = tuple(jax.tree.leaves(subspace.initial))
    result = phx.nn.operator.training.fit_operator(
        model,
        _dataset(),
        epochs=1,
        steps=2,
        batch_size=2,
        learning_rate=2e-2,
        parameter_subspace=subspace,
        dtype_policy=_FLOAT64_POLICY,
    )
    rebased = subspace.rebase(result.last_execution_model)
    final_factors = tuple(jax.tree.leaves(rebased.initial))

    assert any(
        not jnp.array_equal(initial, final)
        for initial, final in zip(initial_factors, final_factors, strict=True)
    )
    assert bool(eqx.tree_equal(subspace.frozen, rebased.frozen))
    assert jnp.isfinite(result.final_loss)


def test_fit_operator_requires_explicit_low_rank_subspace():
    model, _ = _adapted_operator()
    with pytest.raises(ValueError, match="requires an explicit"):
        phx.nn.operator.training.fit_operator(
            model,
            _dataset(),
            epochs=1,
            steps=1,
        )


def test_low_rank_operator_resume_matches_uninterrupted(tmp_path):
    model, subspace = _adapted_operator(seed=2)
    dataset = _dataset()
    common = {
        "epochs": 2,
        "batch_size": 2,
        "seed": 11,
        "checkpoint_every": 1,
        "parameter_subspace": subspace,
        "dtype_policy": _FLOAT64_POLICY,
    }
    uninterrupted = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        **common,
    )
    checkpoint = tmp_path / "low-rank-fit"
    phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=1,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    assert eqx.tree_equal(
        uninterrupted.last_execution_model,
        resumed.last_execution_model,
    )
    assert uninterrupted.history == resumed.history
    assert resumed.resumed_from_step == 1

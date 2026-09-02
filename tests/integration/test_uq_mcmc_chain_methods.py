#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax.uq._mcmc as mcmc_module


def _correlated_problem():
    precision = jnp.asarray([[4.0, 1.2], [1.2, 2.5]])
    center = jnp.asarray([0.4, -0.7])
    space = phx.uq.ParameterSpace(
        jnp.zeros((2,)),
        priors=phx.uq.Normal(0.0, 3.0),
    )
    return phx.uq.PosteriorProblem(
        space,
        lambda value: -0.5 * (value - center) @ precision @ (value - center),
        predict=lambda value, query: cx.Field(
            value[0] + value[1] * query,
            dims=("x",),
        ),
    )


def _assert_tree_equal(left, right):
    comparisons = jax.tree_util.tree_map(jnp.array_equal, left, right)
    assert all(jax.tree_util.tree_leaves(comparisons))


def _assert_tree_close(left, right, *, atol=1e-10):
    comparisons = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=atol),
        left,
        right,
    )
    assert all(jax.tree_util.tree_leaves(comparisons))


def test_vectorized_nuts_replays_and_matches_independent_sequential_chains(
    monkeypatch,
):
    problem = _correlated_problem()
    unstack_calls = 0
    original_unstack = mcmc_module._unstack_tree

    def tracked_unstack(*args, **kwargs):
        nonlocal unstack_calls
        unstack_calls += 1
        return original_unstack(*args, **kwargs)

    monkeypatch.setattr(mcmc_module, "_unstack_tree", tracked_unstack)
    settings: dict[str, Any] = dict(
        key=jr.key(200),
        num_chains=3,
        num_warmup=50,
        num_samples=60,
        initial_step_size=0.2,
        target_acceptance_rate=0.9,
        max_num_doublings=7,
    )
    sequential = phx.uq.sample_nuts(problem, **settings, chain_method="sequential")
    sequential_unstacks = unstack_calls
    unstack_calls = 0
    vectorized = phx.uq.sample_nuts(problem, **settings, chain_method="vectorized")
    vectorized_unstacks = unstack_calls
    unstack_calls = 0
    vectorized_replay = phx.uq.sample_nuts(
        problem,
        **settings,
        chain_method="vectorized",
    )
    unstack_calls = 0
    interleaved = phx.uq.sample_nuts(
        problem,
        **settings,
        chain_method="interleaved",
    )
    interleaved_unstacks = unstack_calls
    unstack_calls = 0
    interleaved_replay = phx.uq.sample_nuts(
        problem,
        **settings,
        chain_method="interleaved",
    )

    _assert_tree_equal(vectorized.samples, vectorized_replay.samples)
    _assert_tree_equal(
        vectorized.unconstrained_samples,
        vectorized_replay.unconstrained_samples,
    )
    _assert_tree_equal(interleaved.samples, interleaved_replay.samples)
    _assert_tree_equal(
        interleaved.unconstrained_samples,
        interleaved_replay.unconstrained_samples,
    )
    _assert_tree_close(sequential.samples, vectorized.samples)
    _assert_tree_close(vectorized.samples, interleaved.samples)
    _assert_tree_close(vectorized.final_states, interleaved.final_states)
    assert jnp.allclose(
        vectorized.acceptance_rate,
        interleaved.acceptance_rate,
        rtol=0.0,
        atol=1e-10,
    )
    assert jnp.allclose(
        vectorized.energy,
        interleaved.energy,
        rtol=0.0,
        atol=1e-10,
    )
    assert jnp.array_equal(
        vectorized.num_integration_steps,
        interleaved.num_integration_steps,
    )
    assert jnp.array_equal(
        vectorized.num_trajectory_expansions,
        interleaved.num_trajectory_expansions,
    )
    assert jnp.array_equal(vectorized.divergent, interleaved.divergent)
    assert jnp.array_equal(sequential.divergent, vectorized.divergent)
    assert jnp.array_equal(sequential.chain_keys, vectorized.chain_keys)
    assert vectorized.chain_method == "vectorized"
    assert sequential.chain_method == "sequential"
    assert interleaved.chain_method == "interleaved"
    assert sequential_unstacks > vectorized_unstacks
    assert vectorized_unstacks == interleaved_unstacks == 2
    assert not jnp.array_equal(vectorized.samples[0], vectorized.samples[1])
    assert vectorized.diagnostics.rhat.shape == (2,)
    assert vectorized.sample_memory_bytes == sequential.sample_memory_bytes
    assert vectorized.adaptation_duration_seconds > 0.0
    assert vectorized.sampling_duration_seconds > 0.0
    assert vectorized.samples_per_second > 0.0

    query = jnp.linspace(0.0, 1.0, 9)
    full = interleaved.predict(query)
    chunked = interleaved.predict(query, batch_size=17)
    assert isinstance(full, phx.uq.PredictiveField)
    assert isinstance(chunked, phx.uq.PredictiveField)
    assert jnp.array_equal(
        jnp.asarray(full.samples.data), jnp.asarray(chunked.samples.data)
    )
    assert full.samples.shape == (3, 60, 9)


def test_vectorized_hmc_preserves_fixed_trajectory_and_diagnostics():
    problem = _correlated_problem()
    settings: dict[str, Any] = dict(
        key=jr.key(201),
        num_integration_steps=6,
        num_chains=2,
        num_warmup=45,
        num_samples=50,
        initial_step_size=0.15,
        target_acceptance_rate=0.9,
    )
    sequential = phx.uq.sample_hmc(problem, **settings, chain_method="sequential")
    vectorized = phx.uq.sample_hmc(problem, **settings, chain_method="vectorized")

    _assert_tree_close(sequential.samples, vectorized.samples, atol=1e-6)
    assert jnp.all(vectorized.num_integration_steps == 6)
    assert len(vectorized.final_states) == 2
    assert len(vectorized.warmup) == 2
    assert all(warmup.num_integration_steps == 6 for warmup in vectorized.warmup)
    assert vectorized.log_density.shape == (2, 50)
    assert vectorized.acceptance_rate.shape == (2, 50)
    assert vectorized.divergent.shape == (2, 50)
    assert jnp.isfinite(vectorized.diagnostics.max_rhat)


@pytest.mark.parametrize("algorithm", ["nuts", "hmc"])
def test_chain_specific_initial_positions_are_checked_and_used(algorithm):
    problem = _correlated_problem()
    initial_positions = jnp.asarray([[-1.0, 0.25], [1.0, -0.25]])
    settings: dict[str, Any] = dict(
        key=jr.key(208),
        num_chains=2,
        num_warmup=8,
        num_samples=4,
        initial_positions=initial_positions,
        initial_step_size=0.1,
    )
    result = (
        phx.uq.sample_nuts(problem, **settings, max_num_doublings=4)
        if algorithm == "nuts"
        else phx.uq.sample_hmc(problem, **settings, num_integration_steps=3)
    )

    assert result.samples.shape == (2, 4, 2)
    assert jnp.all(jnp.isfinite(result.log_density))
    assert len(result.warmup) == 2


@pytest.mark.parametrize("algorithm", ["nuts", "hmc"])
def test_chain_specific_invalid_initial_positions_fail_before_warmup(algorithm):
    problem = _correlated_problem()
    settings: dict[str, Any] = dict(
        key=jr.key(209),
        num_chains=2,
        num_warmup=8,
        num_samples=4,
        initial_positions=jnp.asarray([[0.0, 0.0], [jnp.nan, 0.0]]),
        initial_step_size=0.1,
    )
    with pytest.raises(FloatingPointError, match="positions must be finite"):
        if algorithm == "nuts":
            phx.uq.sample_nuts(problem, **settings, max_num_doublings=4)
        else:
            phx.uq.sample_hmc(problem, **settings, num_integration_steps=3)


@pytest.mark.parametrize("algorithm", ["nuts", "hmc"])
def test_every_chain_specific_initial_log_density_is_checked(algorithm):
    space = phx.uq.ParameterSpace(
        jnp.zeros((1,)),
        priors=phx.uq.Normal(0.0, 1.0),
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda value: jnp.where(value[0] > 0.0, jnp.nan, -0.5 * value[0] ** 2),
    )
    settings: dict[str, Any] = dict(
        key=jr.key(210),
        num_chains=2,
        num_warmup=8,
        num_samples=4,
        initial_positions=jnp.asarray([[-1.0], [1.0]]),
        initial_step_size=0.1,
    )
    with pytest.raises(FloatingPointError, match="log density and gradient"):
        if algorithm == "nuts":
            phx.uq.sample_nuts(problem, **settings, max_num_doublings=4)
        else:
            phx.uq.sample_hmc(problem, **settings, num_integration_steps=3)


@pytest.mark.parametrize("chain_method", ["vectorized", "interleaved"])
def test_vectorized_nuts_supports_dense_and_diagonal_mass_adaptation(chain_method):
    problem = _correlated_problem()
    settings: dict[str, Any] = dict(
        key=jr.key(202),
        num_chains=2,
        num_warmup=55,
        num_samples=50,
        initial_step_size=0.2,
        target_acceptance_rate=0.9,
        max_num_doublings=7,
        chain_method=chain_method,
    )
    diagonal = phx.uq.sample_nuts(
        problem,
        **settings,
        kinetic=phx.uq.MCMCMassAdaptationPlan.diagonal(),
    )
    dense = phx.uq.sample_nuts(
        problem,
        **settings,
        kinetic=phx.uq.MCMCMassAdaptationPlan.blocks((("",),), max_block_size=2),
    )

    assert diagonal.samples.shape == dense.samples.shape == (2, 50, 2)
    assert diagonal.warmup[0].inverse_mass_matrix.shape == (2,)
    assert dense.warmup[0].inverse_mass_matrix.shape == (2, 2)
    assert jnp.all(jnp.isfinite(diagonal.samples))
    assert jnp.all(jnp.isfinite(dense.samples))
    assert diagonal.sample_memory_bytes == dense.sample_memory_bytes


def test_nuts_and_hmc_sample_every_separable_mlp_final_layer_subtree():
    model = phx.nn.layers.inference_mode(
        phx.nn.models.SeparableMLP(
            in_size=2,
            out_size="scalar",
            latent_size=2,
            width_size=3,
            depth=1,
            key=jr.key(203),
        )
    )
    final_layers = tuple(
        f".model.models[{index}].layers[{len(factor.layers) - 1}]"
        for index, factor in enumerate(model.model.models)
    )
    subspace = phx.nn.parameters.ParameterSubspace.from_subtree_paths(model, final_layers)
    expected_paths = (
        ".model.models[0].layers[1].weight",
        ".model.models[0].layers[1].bias",
        ".model.models[1].layers[1].weight",
        ".model.models[1].layers[1].bias",
    )

    assert subspace.leaf_paths == expected_paths

    priors = jax.tree_util.tree_map(
        lambda _: phx.uq.Normal(0.0, 1.0),
        subspace.initial,
    )
    space = phx.uq.ParameterSpace(subspace.initial, priors=priors)
    inputs = jnp.asarray(
        [
            [-0.8, -0.4],
            [-0.3, 0.2],
            [0.1, 0.5],
            [0.6, -0.2],
            [0.9, 0.7],
        ]
    )
    baseline = jax.vmap(model)(inputs)
    targets = baseline + jnp.asarray([0.02, -0.01, 0.03, -0.02, 0.01])

    def predict(selected):
        return jax.vmap(subspace.reconstruct(selected))(inputs)

    problem = phx.uq.PosteriorProblem(
        space,
        lambda selected: -0.5 * jnp.sum(((predict(selected) - targets) / 0.1) ** 2),
        predict=lambda selected: predict(selected),
    )
    problem.validate()

    hmc = phx.uq.sample_hmc(
        problem,
        key=jr.key(204),
        num_integration_steps=3,
        num_chains=2,
        num_warmup=15,
        num_samples=8,
        initial_step_size=0.02,
        chain_method="vectorized",
    )
    nuts = phx.uq.sample_nuts(
        problem,
        key=jr.key(205),
        num_chains=2,
        num_warmup=15,
        num_samples=8,
        initial_step_size=0.02,
        chain_method="interleaved",
    )

    for result in (hmc, nuts):
        assert result.log_density.shape == (2, 8)
        assert jnp.all(jnp.isfinite(result.log_density))
        assert all(
            leaf.shape[:2] == (2, 8) for leaf in jax.tree_util.tree_leaves(result.samples)
        )
        first_draw = jax.tree_util.tree_map(lambda leaf: leaf[0, 0], result.samples)
        assert jnp.all(jnp.isfinite(jax.vmap(subspace.reconstruct(first_draw))(inputs)))


def test_interleaved_chain_method_is_nuts_specific():
    problem = _correlated_problem()
    invalid_hmc_method: Any = "interleaved"
    invalid_nuts_method: Any = "unknown"

    with pytest.raises(ValueError, match="sequential.*vectorized"):
        phx.uq.sample_hmc(
            problem,
            key=jr.key(206),
            num_integration_steps=3,
            num_chains=2,
            num_warmup=8,
            num_samples=4,
            chain_method=invalid_hmc_method,
        )
    with pytest.raises(ValueError, match="interleaved"):
        phx.uq.sample_nuts(
            problem,
            key=jr.key(207),
            num_chains=2,
            num_warmup=8,
            num_samples=4,
            chain_method=invalid_nuts_method,
        )

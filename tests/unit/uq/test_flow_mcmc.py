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


def _small_config(**overrides):
    settings: dict[str, Any] = {
        "num_adaptation_rounds": 1,
        "num_local_adaptation_steps": 4,
        "num_global_adaptation_steps": 2,
        "num_stabilization_steps": 1,
        "num_local_steps": 1,
        "num_global_steps": 1,
        "history_capacity_per_chain": 4,
        "history_thinning": 1,
        "flow_layers": 1,
        "num_knots": 4,
        "nn_width": 8,
        "nn_depth": 1,
        "learning_rate": 1e-3,
        "max_epochs": 2,
        "max_patience": 2,
        "batch_size": 2,
        "validation_fraction": 0.25,
    }
    settings.update(overrides)
    return phx.uq.FlowNUTSConfig(**settings)


def _positive_problem():
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {"rate": jnp.asarray(0.0)},
            priors={"rate": phx.uq.LogNormal(0.0, 0.8)},
            bijectors={"rate": phx.uq.ExpBijector()},
        ),
        lambda parameters: -0.5 * ((parameters["rate"] - 1.5) / 0.25) ** 2,
        predict=lambda parameters, query: cx.Field(
            parameters["rate"] * query,
            dims=("query",),
        ),
    )


def _nested_constrained_problem():
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {
                "positive": jnp.asarray(0.0),
                "nested": {"bounded": jnp.asarray(0.0)},
            },
            priors={
                "positive": phx.uq.LogNormal(0.0, 0.8),
                "nested": {"bounded": phx.uq.Uniform(-1.0, 2.0)},
            },
            bijectors={
                "positive": phx.uq.ExpBijector(),
                "nested": {"bounded": phx.uq.SigmoidIntervalBijector(-1.0, 2.0)},
            },
        ),
        lambda parameters: (
            -0.5
            * ((parameters["positive"] - 1.2) ** 2 + parameters["nested"]["bounded"] ** 2)
        ),
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_adaptation_rounds", 0),
        ("num_stabilization_steps", -1),
        ("num_local_steps", 0),
        ("history_capacity_per_chain", 0),
        ("history_thinning", 5),
        ("learning_rate", 0.0),
        ("validation_fraction", 0.5),
    ],
)
def test_flow_nuts_config_rejects_invalid_controls(name, value):
    with pytest.raises(ValueError):
        _small_config(**{name: value})


def test_flow_nuts_preserves_transformed_parameters_and_result_contract():
    result = phx.uq.sample_flow_nuts(
        _positive_problem(),
        key=jr.key(10),
        num_chains=2,
        num_warmup=20,
        num_samples=8,
        target_acceptance_rate=0.9,
        max_num_doublings=5,
        config=_small_config(),
        chain_method="vectorized",
    )
    prediction = result.predict(jnp.asarray([1.0, 2.0]))

    assert result.algorithm == "flow_nuts"
    assert result.samples["rate"].shape == (2, 8)
    assert result.unconstrained_samples["rate"].shape == (2, 8)
    assert jnp.all(result.samples["rate"] > 0.0)
    assert result.log_density.shape == (2, 8)
    assert result.global_acceptance_rate.shape == (2, 8)
    assert result.global_accepted_count.shape == (2, 8)
    assert result.global_nonfinite_count.shape == (2, 8)
    assert result.adaptation_global_acceptance_rate.shape == (1, 2)
    assert result.adaptation_proposal_ess.shape == (1,)
    assert result.adaptation_history_size.shape == (1,)
    assert len(result.training_losses) == 1
    assert len(result.validation_losses) == 1
    assert len(result.flow_training_duration_seconds) == 1
    assert result.flow_parameter_memory_bytes > 0
    assert result.history_memory_bytes > 0
    assert prediction.samples.shape == (2, 8, 2)


def test_flow_nuts_roundtrips_nested_positive_and_bounded_parameters():
    result = phx.uq.sample_flow_nuts(
        _nested_constrained_problem(),
        key=jr.key(14),
        num_chains=2,
        num_warmup=12,
        num_samples=4,
        max_num_doublings=4,
        config=_small_config(max_epochs=1, max_patience=1),
        chain_method="vectorized",
    )

    assert result.samples["positive"].shape == (2, 4)
    assert result.samples["nested"]["bounded"].shape == (2, 4)
    assert jnp.all(result.samples["positive"] > 0.0)
    assert jnp.all(result.samples["nested"]["bounded"] > -1.0)
    assert jnp.all(result.samples["nested"]["bounded"] < 2.0)
    assert result.unconstrained_samples["nested"]["bounded"].shape == (2, 4)


def test_flow_nuts_requires_explicit_starts_for_custom_joint_prior():
    problem = phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.asarray(0.0),
            log_prior=lambda value: -0.5 * value**2,
        ),
        lambda value: -0.5 * (value - 0.3) ** 2,
    )

    with pytest.raises(ValueError, match="initial_positions"):
        phx.uq.sample_flow_nuts(
            problem,
            key=jr.key(11),
            num_chains=2,
            num_warmup=10,
            num_samples=4,
            config=_small_config(),
        )


def test_flow_nuts_validates_leading_chain_axes_before_warmup():
    problem = _positive_problem()

    with pytest.raises(ValueError, match="shape"):
        phx.uq.sample_flow_nuts(
            problem,
            key=jr.key(12),
            num_chains=2,
            num_warmup=10,
            num_samples=4,
            initial_positions={"rate": jnp.zeros((3,))},
            config=_small_config(),
        )


def test_flow_nuts_sequential_and_vectorized_methods_share_semantic_keys():
    common: dict[str, Any] = {
        "key": jr.key(13),
        "num_chains": 2,
        "num_warmup": 12,
        "num_samples": 4,
        "target_acceptance_rate": 0.9,
        "max_num_doublings": 4,
        "config": _small_config(max_epochs=1, max_patience=1),
    }

    sequential = phx.uq.sample_flow_nuts(
        _positive_problem(),
        chain_method="sequential",
        **common,
    )
    vectorized = phx.uq.sample_flow_nuts(
        _positive_problem(),
        chain_method="vectorized",
        **common,
    )
    sample_comparisons = jax.tree_util.tree_map(
        lambda left, right: jnp.allclose(left, right, rtol=1e-13, atol=1e-13),
        sequential.samples,
        vectorized.samples,
    )

    assert jnp.array_equal(sequential.chain_keys, vectorized.chain_keys)
    assert all(jax.tree_util.tree_leaves(sample_comparisons))
    assert jnp.allclose(
        sequential.log_density,
        vectorized.log_density,
        rtol=1e-13,
        atol=1e-13,
    )
    assert jnp.allclose(
        sequential.acceptance_rate,
        vectorized.acceptance_rate,
        rtol=1e-13,
        atol=1e-13,
    )
    assert jnp.array_equal(
        sequential.global_acceptance_rate,
        vectorized.global_acceptance_rate,
    )
    assert jnp.array_equal(
        sequential.global_accepted_count,
        vectorized.global_accepted_count,
    )
    assert jnp.array_equal(
        sequential.global_nonfinite_count,
        vectorized.global_nonfinite_count,
    )

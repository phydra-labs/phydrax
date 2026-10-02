from collections.abc import Mapping
from itertools import combinations, product
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from phydrax._randomized_residual_modes import RandomizedResidualLossMode
from phydrax._sampling._addressing import derive_key, SampleAddress
from phydrax.domain import DomainFunction
from phydrax.operators.differential._dimension_estimators import (
    dimension_sum_samples,
    DimensionOperatorSamples,
    DimensionSamplingPolicy,
)
from phydrax.operators.differential._stochastic_estimators import (
    stochastic_bilaplacian_samples,
    stochastic_divergence_samples,
    StochasticOperatorSamples,
    StochasticTracePolicy,
)
from phydrax.terms._randomized_quadratic import randomized_squared_mean
from phydrax.terms._randomized_residual import (
    RandomizedResidualBatch,
    RandomizedResidualSamples,
    RandomizedResidualTerm,
)
from phydrax.typing import PRNGKey


def _term(
    samples: RandomizedResidualSamples
    | DimensionOperatorSamples
    | StochasticOperatorSamples,
    mode: RandomizedResidualLossMode,
) -> RandomizedResidualTerm:
    def evaluate(
        functions: Mapping[str, DomainFunction], collocation: Any, key: PRNGKey
    ) -> RandomizedResidualSamples | DimensionOperatorSamples | StochasticOperatorSamples:
        del functions, collocation, key
        return samples

    return RandomizedResidualTerm(
        evaluate, collocation=(), sampling_mode="fixed", loss_mode=mode
    )


def test_finite_population_adapter_retains_error_and_refuses_iid_objective() -> None:
    contributions = jnp.asarray(
        [[1.0 + 2.0j, 2.0], [3.0 - 1.0j, -3.0], [-2.0 + 0.5j, 4.0], [7.0, 1.0]]
    )
    samples = dimension_sum_samples(
        lambda index: contributions[index], jr.key(21), DimensionSamplingPolicy(4, 2)
    )
    reference_variance = jnp.sum(jnp.abs(samples.values - samples.mean) ** 2, axis=0)
    expected_error = jnp.sqrt((1.0 - 2.0 / 4.0) * reference_variance / 2.0)
    diagnostics = _term(samples, "plug_in").diagnostics({})
    assert jnp.allclose(samples.standard_error, expected_error)
    assert jnp.allclose(
        diagnostics.mean_probe_standard_error, jnp.sqrt(jnp.sum(expected_error**2))
    )
    assert diagnostics.sampling_design == "finite_population"
    assert diagnostics.population_size == 4
    assert diagnostics.uncertainty_available
    with pytest.raises(ValueError, match="independent_product"):
        _term(samples, "u_statistic").loss({})


def test_finite_population_independent_products_are_unbiased_by_enumeration() -> None:
    population = jnp.asarray([1.0, -2.0, 4.0, 8.0])
    left_groups = tuple(
        population[jnp.asarray(indices)] * 4.0 for indices in combinations(range(4), 2)
    )
    right_groups = tuple(
        population[jnp.asarray(indices)] * 4.0 for indices in combinations(range(4), 3)
    )

    def objective(scale: Array) -> Array:
        return jnp.mean(
            jnp.stack(
                tuple(
                    randomized_squared_mean(
                        scale * left,
                        (),
                        "independent_product",
                        right=scale * right,
                        sampling_design="finite_population",
                        right_sampling_design="finite_population",
                    )
                    for left, right in product(left_groups, right_groups)
                )
            )
        )

    scale = jnp.asarray(0.7)
    exact = jnp.sum(population)
    assert jnp.allclose(objective(scale), scale**2 * exact**2)
    assert jnp.allclose(jax.grad(objective)(scale), 2.0 * scale * exact**2)


def test_singleton_independent_product_keeps_both_gradients_and_sign() -> None:
    def objective(left_parameter: Array, right_parameter: Array) -> Array:
        return randomized_squared_mean(
            jnp.asarray([left_parameter]),
            (),
            "independent_product",
            right=jnp.asarray([right_parameter - 1.0, right_parameter + 1.0]),
            sampling_design="iid",
            right_sampling_design="iid",
        )

    value = jnp.asarray(2.0)
    other = jnp.asarray(-3.0)
    assert jnp.allclose(objective(value, other), -6.0)
    gradients = jax.grad(objective, argnums=(0, 1))(value, other)
    assert jnp.allclose(gradients[0], -3.0)
    assert jnp.allclose(gradients[1], 2.0)


@pytest.mark.parametrize("mode", ["u_statistic", "independent_product"])
def test_unknown_law_cannot_enter_unbiased_modes(
    mode: RandomizedResidualLossMode,
) -> None:
    samples = RandomizedResidualSamples(jnp.asarray([1.0, 3.0]))
    with pytest.raises(ValueError, match="sampling_design|iid"):
        _term(samples, mode).loss({})
    stochastic = StochasticOperatorSamples(
        jnp.asarray([1.0, 3.0]), distribution="normal", dependence_ids=jnp.asarray([0, 1])
    )
    with pytest.raises(ValueError, match="sampling_design|iid"):
        _term(stochastic, mode).loss({})
    dimension = DimensionOperatorSamples(
        jnp.asarray([0, 1]), jnp.asarray([2.0, 6.0]), DimensionSamplingPolicy(2, 2)
    )
    with pytest.raises(ValueError, match="sampling_design|iid"):
        _term(dimension, mode).loss({})


def test_unknown_and_nonexact_singletons_report_unavailable_uncertainty() -> None:
    for samples in (
        RandomizedResidualSamples(jnp.asarray([2.0])),
        RandomizedResidualSamples(jnp.asarray([2.0]), sampling_design="iid"),
        StochasticOperatorSamples(
            jnp.asarray([2.0]), distribution="normal", sampling_design="iid"
        ),
        dimension_sum_samples(
            lambda index: jnp.asarray([1.0, 3.0])[index],
            jr.key(0),
            DimensionSamplingPolicy(2, 1),
        ),
    ):
        diagnostics = _term(samples, "plug_in").diagnostics({})
        assert not diagnostics.uncertainty_available
        assert jnp.isnan(diagnostics.mean_probe_standard_error)
        assert diagnostics.passed


def test_exact_singleton_has_zero_uncertainty_and_exact_objective() -> None:
    exact_dimension = dimension_sum_samples(
        lambda index: jnp.asarray(3.0), jr.key(0), DimensionSamplingPolicy(1, 1)
    )
    for samples in (
        exact_dimension,
        RandomizedResidualSamples(jnp.asarray([3.0]), sampling_design="exact"),
        StochasticOperatorSamples(
            jnp.asarray([3.0]), distribution="normal", sampling_design="exact"
        ),
    ):
        term = _term(samples, "u_statistic")
        diagnostics = term.diagnostics({})
        assert jnp.allclose(term.loss({}), 9.0)
        assert jnp.allclose(diagnostics.mean_probe_standard_error, 0.0)
        assert diagnostics.uncertainty_available
        assert diagnostics.passed


def test_exact_complex_vector_singleton_respects_masks_and_weights() -> None:
    values = jnp.asarray([[[1.0 + 2.0j, 2.0j], [3.0, 4.0j], [jnp.nan, jnp.nan]]])
    samples = RandomizedResidualSamples(
        values,
        sample_shape=(3,),
        event_shape=(2,),
        mask=jnp.asarray([True, True, False]),
        weights=jnp.asarray([1.0, 3.0, 100.0]),
        sampling_design="exact",
    )
    term = _term(samples, "u_statistic")
    assert jnp.allclose(term.loss({}), (9.0 + 3.0 * 25.0) / 4.0)
    assert term.diagnostics({}).passed


def test_independent_product_refuses_reused_keys_except_exact_groups() -> None:
    key = jr.key(8)
    batch = RandomizedResidualBatch((), key, key)
    iid = RandomizedResidualSamples(jnp.asarray([1.0, 3.0]), sampling_design="iid")
    with pytest.raises(Exception, match="distinct owner-generated"):
        _term(iid, "independent_product").loss({}, batch=batch)
    exact = RandomizedResidualSamples(jnp.asarray([2.0]), sampling_design="exact")
    assert jnp.allclose(_term(exact, "independent_product").loss({}, batch=batch), 4.0)


def test_proposal_probabilities_are_refused_but_contributions_differentiate() -> None:
    contributions = jnp.asarray([1.0, 2.0, 4.0])
    probabilities = contributions / jnp.sum(contributions)
    policy = DimensionSamplingPolicy(
        3, 4, sampling="importance", replace=True, probabilities=probabilities
    )

    def mean(scale: Array) -> Array:
        return dimension_sum_samples(
            lambda index: scale * contributions[index], jr.key(3), policy
        ).mean

    assert jnp.allclose(jax.grad(mean)(jnp.asarray(0.7)), 7.0)

    def changing_proposal(proposal: Array) -> Array:
        changed = eqx.tree_at(lambda item: item.probabilities, policy, proposal)
        return dimension_sum_samples(
            lambda index: contributions[index], jr.key(3), changed
        ).mean

    with pytest.raises(ValueError, match="proposal probabilities are fixed"):
        jax.grad(changing_proposal)(probabilities)
    with pytest.raises(ValueError, match="proposal probabilities are fixed"):
        jax.jvp(changing_proposal, (probabilities,), (jnp.ones_like(probabilities),))


def test_gaussian_bilaplacian_normalizes_pure_and_mixed_quartics() -> None:
    state = jnp.asarray([0.4, -0.3])
    key = jr.key(9)
    policy = StochasticTracePolicy(5, distribution="normal")
    probe_root = derive_key(
        key, SampleAddress("differential", "bilaplacian", role="probe")
    )
    probes = jax.vmap(
        lambda index: jr.normal(jr.fold_in(probe_root, index), (2,), dtype=state.dtype)
    )(jnp.arange(5, dtype=jnp.int32))

    def values(scale: Array) -> Array:
        samples = stochastic_bilaplacian_samples(
            lambda point: (
                scale
                * jnp.asarray(
                    [point[0] ** 4, (1.0 + 2.0j) * point[0] ** 2 * point[1] ** 2]
                )
            ),
            state,
            key,
            policy=policy,
        )
        return samples.values

    # E[v_i^4] = 3 and E[v_i^2 v_j^2] = 1: the exact targets are 24 and 8(1+2j).
    expected = jnp.stack(
        (
            8.0 * probes[:, 0] ** 4,
            8.0 * (1.0 + 2.0j) * probes[:, 0] ** 2 * probes[:, 1] ** 2,
        ),
        axis=-1,
    )
    assert jnp.allclose(jax.jit(values)(jnp.asarray(1.0)), expected, atol=1e-10)
    assert jnp.allclose(jax.jacfwd(values)(jnp.asarray(1.0)), expected, atol=1e-10)


def test_native_probe_singleton_preserves_value_without_claiming_error() -> None:
    samples = stochastic_divergence_samples(
        lambda state: 3.0 * state,
        jnp.asarray([0.4, -0.3]),
        jr.key(2),
        policy=StochasticTracePolicy(1),
    )
    assert jnp.allclose(samples.mean, 6.0)
    assert not samples.uncertainty_available
    assert jnp.isnan(samples.standard_error)


def test_sequential_contributions_differentiate_only_selected_valid_branches() -> None:
    policy = DimensionSamplingPolicy(2, 4, replace=True)

    def mean(scale: Array) -> Array:
        def contribution(index: Array) -> Array:
            signed = jnp.where(index == 0, scale, -scale)
            return jax.lax.switch(
                index,
                (lambda value: jnp.sqrt(value), lambda value: jnp.sqrt(-value)),
                signed,
            )

        return dimension_sum_samples(
            contribution, jr.key(4), policy, evaluation="sequential"
        ).mean

    scale = jnp.asarray(4.0)
    assert jnp.allclose(jax.jit(mean)(scale), 4.0)
    assert jnp.allclose(jax.jit(jax.grad(mean))(scale), 0.5)


def test_population_refuses_int32_index_overflow_before_sampling() -> None:
    with pytest.raises(ValueError, match="int32 population capacity"):
        DimensionSamplingPolicy(jnp.iinfo(jnp.int32).max + 1, 2, replace=True)


def test_gaussian_bilaplacian_refuses_incompatible_fourth_moment() -> None:
    with pytest.raises(ValueError, match="normal probes"):
        stochastic_bilaplacian_samples(
            lambda point: point[0] ** 4,
            jnp.asarray([0.4, -0.3], dtype=jnp.float64),
            jr.key(9),
            policy=StochasticTracePolicy(5),
        )


def test_gaussian_probe_addresses_preserve_prefix_when_count_changes() -> None:
    state = jnp.asarray([0.4, -0.3], dtype=jnp.float64)

    def function(point: Array) -> Array:
        return jnp.sum(point**4)

    short = stochastic_bilaplacian_samples(
        function, state, jr.key(5), policy=StochasticTracePolicy(2, distribution="normal")
    )
    longer = stochastic_bilaplacian_samples(
        function, state, jr.key(5), policy=StochasticTracePolicy(5, distribution="normal")
    )
    assert jnp.array_equal(short.values, longer.values[:2])


def test_gaussian_bilaplacian_refuses_nonfinite_primal_with_finite_derivative() -> None:
    def function(point: Array) -> Array:
        return jnp.asarray(jnp.inf, dtype=point.dtype) + jnp.sum(point**4)

    def evaluate(state: Array) -> Array:
        return stochastic_bilaplacian_samples(
            function,
            state,
            jr.key(5),
            policy=StochasticTracePolicy(2, distribution="normal"),
        ).mean

    with pytest.raises(eqx.EquinoxRuntimeError, match="Gaussian bilaplacian contraction"):
        eqx.filter_jit(evaluate)(jnp.asarray([0.4, -0.3], dtype=jnp.float64))

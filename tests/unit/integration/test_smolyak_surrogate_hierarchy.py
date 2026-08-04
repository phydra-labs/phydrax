#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _surrogate():
    uncertainty = phx.domain.ProbabilityDomain(
        phx.uq.Uniform(-1.0, 1.0),
        label="u",
    )
    fine = uncertainty.Function("u")(lambda u: u**4 + 0.2 * u)
    surrogate = phx.operators.interpolate_smolyak(
        fine,
        phx.operators.SmolyakInterpolationPlan(1, 3),
    )
    return surrogate


def _adapter(*, deterministic_base):
    surrogate = _surrogate()
    expectation = (
        phx.integration.smolyak_surrogate_expectation(
            surrogate,
            quadrature_level=5,
        ).value.data
        if deterministic_base
        else None
    )
    return phx.integration.SmolyakSurrogateHierarchyAdapter(
        surrogate,
        lambda u: u**4 + 0.2 * u,
        problem_id="quartic-uq",
        observable_id="quartic-output",
        hierarchy_id="quartic-smolyak-control",
        sampler_id="quartic-smolyak-sampler",
        fine_solver_id="analytic-fine",
        fine_approximation_id="quartic-exact",
        surrogate_expectation=expectation,
    )


def test_smolyak_control_hierarchy_recovers_fine_expectation():
    adapter = _adapter(deterministic_base=True)
    result = phx.integration.integrate(
        adapter.observable,
        adapter.target,
        phx.integration.MultilevelMonteCarloPlan(
            samples_per_level=(2, 8192),
            initial_samples=(2, 2),
            max_samples_per_level=(2, 8192),
            batch_size=8192,
        ),
        key=jr.key(61),
    )

    assert result.successful
    assert result.provenance.target == "quartic-smolyak-control"
    assert jnp.allclose(result.value, 0.2, atol=0.012)
    assert jnp.allclose(
        result.diagnostics.correction_means[0],
        adapter.surrogate_expectation,
    )
    assert result.diagnostics.correction_variance_norms[0] == 0.0


def test_smolyak_fine_correction_is_paired_prefix_stable_and_lower_variance():
    adapter = _adapter(deterministic_base=False)
    root_key = jr.key(62)
    whole = adapter.sample(1, jnp.arange(1024), root_key)
    first = adapter.sample(1, jnp.arange(512), root_key)
    second = adapter.sample(1, jnp.arange(512, 1024), root_key)

    assert jnp.allclose(
        whole.fine_samples,
        jnp.concatenate((first.fine_samples, second.fine_samples)),
    )
    assert jnp.allclose(
        whole.coarse_samples,
        jnp.concatenate((first.coarse_samples, second.coarse_samples)),
    )
    assert jnp.array_equal(whole.pair_ids, whole.sample_indices)
    correction = whole.fine_samples - whole.coarse_samples
    assert jnp.var(correction) < jnp.var(whole.fine_samples)
    assert jnp.all(whole.costs > 0.0)


def test_smolyak_base_and_correction_use_independent_level_namespaces():
    adapter = _adapter(deterministic_base=False)
    indices = jnp.arange(16)
    root_key = jr.key(63)
    base = adapter.sample(0, indices, root_key)
    correction = adapter.sample(1, indices, root_key)

    assert not jnp.allclose(base.fine_samples, correction.coarse_samples)
    assert adapter.hierarchy.refinement_axes == ("surrogate",)
    assert adapter.hierarchy.coupled

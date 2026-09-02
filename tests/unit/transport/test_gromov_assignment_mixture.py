import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.terms._transport_learning import audit_transport_map
from phydrax.transport._gaussian_mixture import (
    gaussian_mixture_transport_problem,
    solve_gaussian_mixture_transport,
)
from phydrax.transport._gromov import gromov_wasserstein_problem, GromovWasserstein


def _weighted(points, probabilities, name):
    return phx.integration.weighted(
        jnp.asarray(points, dtype=float),
        jnp.log(jnp.asarray(probabilities, dtype=float)),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance=name,
    )


def test_gromov_wasserstein_is_small_for_isometric_relabeling():
    source = _weighted([[0.0], [1.0]], [0.5, 0.5], "gw-source")
    target = _weighted([[1.0], [0.0]], [0.5, 0.5], "gw-target")
    relation = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    problem = gromov_wasserstein_problem(
        source,
        target,
        source_relation=relation,
        target_relation=relation,
    )
    result = GromovWasserstein(
        1.0e-3,
        max_outer_iterations=8,
        tolerance=1.0e-5,
    )(problem)

    assert result.approximation_kind == "finite-entropic-gw-local-solve"
    assert result.coupling.shape == (2, 2)
    assert result.objective < 1.0e-3
    assert jnp.all(jnp.isfinite(result.objective_history))


def test_fused_gromov_alpha_zero_uses_declared_feature_cost():
    source = _weighted([[0.0], [1.0]], [0.5, 0.5], "fgw-source")
    target = _weighted([[0.0], [2.0]], [0.5, 0.5], "fgw-target")
    relation = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    feature = jnp.asarray([[0.0, 4.0], [1.0, 1.0]])
    problem = gromov_wasserstein_problem(
        source,
        target,
        source_relation=relation,
        target_relation=2.0 * relation,
        feature_cost=feature,
        alpha=0.0,
    )
    result = GromovWasserstein(1.0e-3, max_outer_iterations=4)(problem)

    assert jnp.allclose(
        result.objective,
        jnp.sum(result.coupling * feature),
        atol=1.0e-6,
    )


def test_gromov_rejects_non_symmetric_relational_costs():
    target = _weighted([[0.0], [1.0]], [0.5, 0.5], "invalid-gw")
    with pytest.raises(ValueError, match="symmetric"):
        gromov_wasserstein_problem(
            target,
            target,
            source_relation=jnp.asarray([[0.0, 1.0], [0.0, 0.0]]),
            target_relation=jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
        )


def _single_gaussian(mean, variance):
    covariance = jnp.asarray([[[variance]]])
    return phx.ml.mixture.GaussianMixtureModel(
        jnp.asarray([1.0]),
        jnp.asarray([[mean]]),
        covariance,
        jnp.asarray([[[1.0 / variance]]]),
        jnp.asarray([jnp.log(variance)]),
        covariance_type="full",
    )


def test_single_gaussian_mixture_transport_matches_analytic_w2():
    problem = gaussian_mixture_transport_problem(
        _single_gaussian(0.0, 1.0),
        _single_gaussian(2.0, 4.0),
    )
    result = solve_gaussian_mixture_transport(
        problem,
        phx.transport.Sinkhorn(0.05, max_iterations=64),
    )

    assert result.approximation_kind == "exact-single-gaussian-w2"
    assert jnp.allclose(result.objective, 5.0, atol=1.0e-6)
    assert jnp.allclose(result.coupling, jnp.ones((1, 1)))


def test_learned_transport_audit_has_jit_and_gradient_contract():
    represented_cost = lambda value: (
        audit_transport_map(
            value,
            jnp.asarray(0.02),
            jnp.asarray(0.01),
            full_finite_pairs=True,
        ).represented_cost
    )
    compiled = jax.jit(represented_cost)(jnp.asarray(1.3))
    derivative = jax.grad(represented_cost)(jnp.asarray(1.3))
    audit = audit_transport_map(1.3, 0.02, 0.01, full_finite_pairs=True)

    assert jnp.allclose(compiled, 1.3)
    assert jnp.allclose(derivative, 1.0)
    assert audit.valid
    assert audit.semantics == "finite-dual-audit"

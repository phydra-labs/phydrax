#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_gaussian_prior_whitening_roundtrips_normal_and_lognormal_coordinates():
    space = phx.uq.ParameterSpace(
        {
            "coefficient": jnp.array([1.5, -0.5]),
            "rate": jnp.log(jnp.array(2.0)),
        },
        priors={
            "coefficient": phx.uq.Normal(0.5, 2.0),
            "rate": phx.uq.LogNormal(jnp.log(2.0), 0.3),
        },
        bijectors={
            "coefficient": phx.uq.IdentityBijector(),
            "rate": phx.uq.ExpBijector(),
        },
    )

    whitening = phx.uq.GaussianPriorWhitening.from_parameter_space(space)
    position = {
        "coefficient": jnp.array([1.5, -0.5]),
        "rate": jnp.log(jnp.array(2.0)),
    }
    whitened = whitening.whiten(position)

    assert jnp.allclose(whitened["coefficient"], jnp.array([0.5, -0.5]))
    assert jnp.allclose(whitened["rate"], 0.0)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda left, right: jnp.allclose(left, right),
            whitening.unwhiten(whitened),
            position,
        )
    )


def test_structured_whitening_and_transformed_covariance_match_analytic_delta_method():
    location = 0.3
    scale = 0.4
    space = phx.uq.ParameterSpace(
        jnp.asarray(location),
        priors=phx.uq.LogNormal(location, scale),
        bijectors=phx.uq.ExpBijector(),
    )
    problem = phx.uq.PosteriorProblem(space, lambda _: jnp.zeros(()))

    structured = phx.uq.fit_laplace(problem, curvature="diagonal")
    dense = phx.uq.fit_laplace(problem)
    probe = jnp.ones(())
    expected_unconstrained = scale**2
    expected_physical = (jnp.exp(location) * scale) ** 2

    assert structured.whitening is not None
    assert jnp.allclose(
        structured.covariance_vector_product(probe),
        expected_unconstrained,
        atol=1e-7,
    )
    assert jnp.allclose(
        structured.physical_covariance_vector_product(probe),
        expected_physical,
        atol=1e-7,
    )
    assert jnp.allclose(dense.physical_covariance(), expected_physical, atol=1e-7)
    assert jnp.allclose(dense.physical_correlation(), jnp.ones((1, 1)))

    draws = structured.sample_unconstrained(jr.key(4), num_samples=30_000)
    assert jnp.var(draws) == pytest.approx(expected_unconstrained, rel=0.04)


def test_ggn_fisher_curvature_matches_linear_gaussian_posterior():
    design = jnp.array([[1.0, 2.0], [0.5, -1.0], [2.0, 0.2]])
    target = jnp.array([0.4, -0.2, 1.0])
    noise_scale = 0.3
    precision = design.T @ design / noise_scale**2 + jnp.eye(2)
    mode = jnp.linalg.solve(precision, design.T @ target / noise_scale**2)

    def residual(parameters):
        return (design @ parameters - target) / noise_scale

    space = phx.uq.ParameterSpace(mode, priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameters: -0.5 * jnp.sum(residual(parameters) ** 2),
        gauss_newton_residual=residual,
    )
    result = phx.uq.fit_laplace(
        problem,
        curvature="full",
        likelihood_curvature="ggn",
    )
    covariance = jax.vmap(result.covariance_vector_product)(jnp.eye(2))

    assert result.likelihood_curvature == "ggn"
    assert result.approximate_memory_bytes > 0
    assert jnp.allclose(covariance, jnp.linalg.inv(precision), atol=2e-6)


def test_ggn_requires_an_explicit_normalized_residual_contract():
    space = phx.uq.ParameterSpace(jnp.zeros(2), priors=phx.uq.Normal(0.0, 1.0))
    problem = phx.uq.PosteriorProblem(space, lambda value: -0.5 * jnp.sum(value**2))

    with pytest.raises(ValueError, match="Gauss-Newton residual"):
        phx.uq.fit_laplace(
            problem,
            curvature="diagonal",
            likelihood_curvature="ggn",
        )


def test_named_parameter_subspace_selects_exact_array_leaves():
    model = {
        "encoder": {"weight": jnp.ones((2, 2)), "bias": jnp.zeros(2)},
        "head": {"weight": jnp.ones((1, 2)), "bias": jnp.zeros(1)},
        "label": "fixed",
    }
    paths = phx.nn.parameters.ParameterSubspace.array_leaf_paths(model)
    named = phx.nn.parameters.ParameterSubspace.from_leaf_paths(
        model, [paths[1], paths[3]]
    )

    assert named.leaf_paths == (paths[1], paths[3])
    with pytest.raises(ValueError, match="Unknown parameter leaf paths"):
        phx.nn.parameters.ParameterSubspace.from_leaf_paths(model, ["['missing']"])


def test_parameter_subspace_selects_disjoint_branched_subtrees_by_exact_path():
    model = {
        "branches": (
            {
                "body": jnp.ones((3, 2)),
                "head": {"weight": jnp.ones((2, 3)), "bias": jnp.zeros(2)},
            },
            {
                "body": jnp.ones((4, 2)),
                "head": {
                    "weight": jnp.ones((2, 4)),
                    "bias": jnp.zeros(2),
                    "scale": jnp.ones(2),
                },
            },
        ),
        "head_aux": jnp.ones(1),
    }
    subspace = phx.nn.parameters.ParameterSubspace.from_subtree_paths(
        model,
        [
            "['branches'][0]['head']",
            "['branches'][1]['head']",
        ],
    )

    assert subspace.leaf_paths == (
        "['branches'][0]['head']['bias']",
        "['branches'][0]['head']['weight']",
        "['branches'][1]['head']['bias']",
        "['branches'][1]['head']['scale']",
        "['branches'][1]['head']['weight']",
    )
    assert subspace.total_dimension == 20
    assert "['head_aux']" not in subspace.leaf_paths

    with pytest.raises(ValueError, match="Unknown parameter subtree paths"):
        phx.nn.parameters.ParameterSubspace.from_subtree_paths(
            model,
            ["['branches'][0]['hea']"],
        )
    with pytest.raises(ValueError, match="disjoint"):
        phx.nn.parameters.ParameterSubspace.from_subtree_paths(
            model,
            ["['branches'][0]", "['branches'][0]['head']"],
        )
    with pytest.raises(ValueError, match="distinct, non-empty"):
        phx.nn.parameters.ParameterSubspace.from_subtree_paths(model, [])

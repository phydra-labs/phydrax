#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_parameter_space_bijectors_density_and_gradient_are_consistent():
    initial = {
        "bounded": jnp.asarray(0.0),
        "free": jnp.asarray([-0.25, 0.5]),
        "positive": jnp.log(jnp.asarray(2.0)),
    }
    priors = {
        "bounded": phx.uq.Uniform(0.0, 1.0),
        "free": phx.uq.Normal(0.0, 2.0),
        "positive": phx.uq.LogNormal(jnp.log(2.0), 0.4),
    }
    space = phx.uq.ParameterSpace(
        initial,
        priors=priors,
        bijectors={
            "bounded": phx.uq.SigmoidIntervalBijector(0.0, 1.0),
            "free": phx.uq.IdentityBijector(),
            "positive": phx.uq.ExpBijector(),
        },
    )
    problem = phx.uq.PosteriorProblem(
        space,
        lambda physical: (
            -0.5
            * (physical["positive"] + physical["bounded"] + jnp.sum(physical["free"]))
            ** 2
        ),
    )

    physical = space.constrain(initial)
    reconstructed = space.unconstrain(physical)
    value, gradient = problem.validate()
    expected = (
        problem.log_likelihood(physical)
        + jnp.sum(priors["bounded"].log_prob(physical["bounded"]))
        + jnp.sum(priors["free"].log_prob(physical["free"]))
        + jnp.sum(priors["positive"].log_prob(physical["positive"]))
        + space.log_abs_det_jacobian(initial)
    )

    assert physical["bounded"] == 0.5
    assert physical["positive"] == 2.0
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.allclose, initial, reconstructed)
    )
    assert jnp.allclose(value, expected)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(gradient)
    )


def test_parameter_subspace_reconstructs_only_explicitly_selected_leaves():
    tree = {
        "feature": {"weight": jnp.arange(6.0).reshape(2, 3)},
        "last": {"bias": jnp.asarray([0.5]), "weight": jnp.ones((3, 1))},
    }
    subspace = phx.uq.ParameterSubspace(
        tree,
        {
            "feature": {"weight": False},
            "last": {"bias": True, "weight": True},
        },
    )
    updated = jax.tree_util.tree_map(
        lambda value: None if value is None else value + 2.0,
        subspace.initial,
        is_leaf=lambda value: value is None,
    )
    rebuilt = subspace.reconstruct(updated)

    assert subspace.total_dimension == 4
    assert len(subspace.leaf_paths) == 2
    assert jnp.array_equal(rebuilt["feature"]["weight"], tree["feature"]["weight"])
    assert jnp.array_equal(rebuilt["last"]["bias"], tree["last"]["bias"] + 2.0)
    assert jnp.array_equal(rebuilt["last"]["weight"], tree["last"]["weight"] + 2.0)


def test_supervised_likelihood_exposes_fixed_observations_and_log_probabilities():
    rows = jnp.linspace(0.0, 1.0, 6)[:, None]
    domain = phx.domain.DatasetDomain(rows)

    @domain.Function("data")
    def field(row):
        return 1.5 + 2.0 * row[0]

    targets = 1.5 + 2.0 * rows[:, 0]
    likelihood = phx.uq.GaussianLikelihood(0.2)
    constraint = phx.terms.SupervisedLikelihoodTerm(
        "u",
        domain.component(),
        targets,
        likelihood,
        sampling=phx.domain.PointSampling(3, design="uniform"),
    )

    observed = constraint.observed_batch()
    per_case = constraint.log_prob({"u": field}, batch=observed, key=jr.key(0))

    assert jnp.array_equal(observed.indices, jnp.arange(6))
    assert observed.target.shape == (6,)
    assert per_case.shape == (6,)
    assert jnp.allclose(per_case, likelihood.log_prob(targets, targets))
    assert jnp.allclose(
        constraint.loss({"u": field}, batch=observed, key=jr.key(1)),
        -jnp.mean(per_case),
    )

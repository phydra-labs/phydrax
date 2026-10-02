#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any, assert_never, get_args, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.typing import PRNGKey


_CallbackKind: TypeAlias = Literal[
    "predict", "observation_variance", "sample_observation", "gauss_newton_residual"
]


class _ArrayObservationMethods(phx.StrictModule):
    offset: Array

    def shift(self, position: Array, /) -> Array:
        return position + self.offset

    def sample(self, key: PRNGKey, position: Array, /) -> Array:
        return self.shift(position) + jr.normal(
            key, self.offset.shape, dtype=self.offset.dtype
        )


@pytest.mark.strict_jax
@pytest.mark.filterwarnings("error:A JAX array is being set as static:UserWarning")
@pytest.mark.parametrize("callback_kind", get_args(_CallbackKind))
def test_array_backed_callbacks_remain_dynamic_under_compilation(
    callback_kind: _CallbackKind,
) -> None:
    space = phx.uq.ParameterSpace(
        jnp.asarray(0.0, dtype=jnp.float64), priors=phx.uq.Normal(0.0, 1.0)
    )
    state = _ArrayObservationMethods(jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    position = jnp.asarray(0.75, dtype=jnp.float64)
    key = jr.key(0)

    def likelihood(value: Array) -> Array:
        return -0.5 * jnp.square(value)

    def problem(observation: _ArrayObservationMethods) -> phx.uq.PosteriorProblem:
        return phx.uq.PosteriorProblem(
            space,
            likelihood,
            predict=observation.shift if callback_kind == "predict" else None,
            observation_variance=(
                observation.shift if callback_kind == "observation_variance" else None
            ),
            sample_observation=(
                observation.sample if callback_kind == "sample_observation" else None
            ),
            gauss_newton_residual=(
                observation.shift if callback_kind == "gauss_newton_residual" else None
            ),
        )

    @eqx.filter_jit
    def evaluate(posterior: phx.uq.PosteriorProblem, value: Array) -> Array:
        match callback_kind:
            case "predict":
                return posterior.predict(value)
            case "observation_variance":
                return posterior.conditional_observation_variance(value)
            case "sample_observation":
                return posterior.sample_observation(key, value)
            case "gauss_newton_residual":
                return posterior.gauss_newton_residual(value)
            case _:
                assert_never(callback_kind)

    first = problem(state)
    noise = (
        jr.normal(key, state.offset.shape, dtype=state.offset.dtype)
        if callback_kind == "sample_observation"
        else jnp.zeros_like(state.offset)
    )
    expected = position + state.offset + noise
    np.testing.assert_allclose(evaluate(first, position), expected)

    updated = eqx.tree_at(
        lambda observation: observation.offset,
        state,
        jnp.asarray([4.0, 5.0], dtype=jnp.float64),
    )
    second = problem(updated)
    np.testing.assert_allclose(
        evaluate(second, position), position + updated.offset + noise
    )
    np.testing.assert_allclose(
        jax.grad(lambda value: jnp.sum(evaluate(second, value)))(position), 2.0
    )


def test_parameter_contracts() -> None:
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
    tree = {
        "feature": {"weight": jnp.arange(6.0).reshape(2, 3)},
        "last": {"bias": jnp.asarray([0.5]), "weight": jnp.ones((3, 1))},
    }
    subspace = phx.nn.parameters.ParameterSubspace(
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


def test_supervised_likelihood_exposes_fixed_observations_and_log_probabilities() -> None:
    rows = jnp.linspace(0.0, 1.0, 6)[:, None]
    domain = phx.domain.DatasetDomain(rows)

    @domain.Function("data")
    def field(row: Any) -> Any:
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

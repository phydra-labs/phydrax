import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax.objectives._randomized_residual import (
    RandomizedResidualObjective,
    RandomizedResidualSamples,
)


def _functions(parameter):
    domain = phx.domain.Interval1d(0.0, 1.0)
    return {"u": domain.Parameter(jnp.asarray([parameter]))}


def _parameter(functions):
    return jnp.asarray(functions["u"].func())[0]


def _noisy_evaluator(*, num_realizations, scale):
    def evaluate(functions, collocation, key):
        count = int(collocation["count"])
        noise = jr.normal(key, (num_realizations, count))
        values = _parameter(functions) + scale * noise
        return RandomizedResidualSamples(
            values,
            sample_shape=(count,),
            mask=collocation.get("mask"),
            weights=collocation.get("weights"),
        )

    return evaluate


def test_u_statistic_is_unbiased_for_noisy_residual_and_has_exact_gradient():
    parameter = jnp.asarray(0.7)
    objective = RandomizedResidualObjective(
        _noisy_evaluator(num_realizations=4096, scale=1.5),
        collocation={"count": 1},
        sampling_mode="fixed",
        loss_mode="u_statistic",
    )
    batch = objective.sample(key=jr.key(4))

    value = objective.loss(_functions(parameter), batch=batch)
    gradient = jax.grad(
        lambda value_: objective.loss(_functions(value_), batch=batch)
    )(parameter)

    assert jnp.allclose(value, parameter**2, atol=8e-2)
    assert jnp.allclose(gradient, 2.0 * parameter, atol=8e-2)
    assert objective.diagnostics(_functions(parameter), batch=batch).passed


def test_plugin_exposes_variance_bias_while_independent_product_is_unbiased():
    parameter = jnp.asarray(0.4)
    collocation = {"count": 8192}
    evaluator = _noisy_evaluator(num_realizations=4, scale=2.0)
    u_statistic = RandomizedResidualObjective(
        evaluator,
        collocation=collocation,
        sampling_mode="fixed",
        loss_mode="u_statistic",
    )
    plug_in = RandomizedResidualObjective(
        evaluator,
        collocation=collocation,
        sampling_mode="fixed",
        loss_mode="plug_in",
    )
    independent = RandomizedResidualObjective(
        evaluator,
        collocation=collocation,
        sampling_mode="fixed",
        loss_mode="independent_product",
    )
    shared_key = jr.key(8)

    unbiased = u_statistic.loss(_functions(parameter), key=shared_key)
    biased = plug_in.loss(_functions(parameter), key=shared_key)
    product = independent.loss(_functions(parameter), key=shared_key)

    assert jnp.allclose(unbiased, parameter**2, atol=8e-2)
    assert jnp.allclose(product, parameter**2, atol=8e-2)
    assert jnp.allclose(biased - unbiased, 1.0, atol=8e-2)


def test_vector_complex_residuals_masks_and_weights_reduce_correctly():
    collocation = {
        "residual": jnp.asarray(
            [[1.0 + 2.0j, 0.5 - 0.5j], [3.0 + 0.0j, 4.0j], [9.0, 9.0]]
        ),
        "mask": jnp.asarray([True, True, False]),
        "weights": jnp.asarray([1.0, 3.0, 100.0]),
    }

    def evaluator(functions, batch, key):
        del functions, key
        values = jnp.broadcast_to(batch["residual"], (3,) + batch["residual"].shape)
        return RandomizedResidualSamples(
            values,
            sample_shape=(3,),
            event_shape=(2,),
            mask=batch["mask"],
            weights=batch["weights"],
        )

    objective = RandomizedResidualObjective(
        evaluator,
        collocation=collocation,
        sampling_mode="fixed",
    )
    expected = (1.0 * (5.0 + 0.5) + 3.0 * (9.0 + 16.0)) / 4.0

    assert jnp.allclose(objective.loss({}, key=jr.key(0)), expected)


def test_resampled_collocation_is_materialized_once_per_optimizer_update():
    calls = []

    def sampler(key):
        calls.append(key)
        return {"target": jnp.asarray(1.0)}

    def evaluator(functions, batch, key):
        del key
        residual = _parameter(functions) - batch["target"]
        return RandomizedResidualSamples(jnp.stack((residual, residual)))

    objective = RandomizedResidualObjective(
        evaluator,
        collocation=sampler,
        sampling_mode="resample",
    )
    solver = phx.solver.FunctionalSolver(
        functions=_functions(0.0),
        constraints=(),
        objectives=(objective,),
    )

    trained = solver.solve(
        num_iter=5,
        optim=optax.sgd(0.1),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert len(calls) == 5
    assert _parameter(trained.functions) > 0.5


def test_zero_valid_mass_is_rejected():
    def evaluator(functions, batch, key):
        del functions, batch, key
        return RandomizedResidualSamples(
            jnp.ones((2, 3)),
            sample_shape=(3,),
            mask=jnp.zeros((3,), dtype=bool),
        )

    objective = RandomizedResidualObjective(
        evaluator,
        collocation={"points": jnp.ones((3, 1))},
        sampling_mode="fixed",
    )

    with pytest.raises(Exception, match="zero valid"):
        objective.loss({}, key=jr.key(0))

from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx
from phydrax.terms._randomized_residual import (
    RandomizedResidualSamples,
    RandomizedResidualTerm,
)


def _functions(parameter: Any) -> Any:
    domain = phx.domain.Interval1d(0.0, 1.0)
    return {"u": domain.Parameter(jnp.asarray([parameter]))}


def _parameter(functions: Any) -> Any:
    return jnp.asarray(functions["u"].func())[0]


def _noisy_evaluator(*, num_realizations: Any, scale: Any) -> Any:
    def evaluate(functions: Any, collocation: Any, key: Any) -> Any:
        count = int(collocation["count"])
        noise = jr.normal(key, (num_realizations, count))
        values = _parameter(functions) + scale * noise
        return RandomizedResidualSamples(
            values,
            sample_shape=(count,),
            mask=collocation.get("mask"),
            weights=collocation.get("weights"),
            sampling_design="iid",
        )

    return evaluate


def test_randomized_residual_objective_scenario_1() -> None:
    parameter = jnp.asarray(0.7)
    realizations = jnp.asarray(tuple(product((-1.0, 1.0), repeat=2)))

    def evaluator(functions: Any, collocation: Any, key: Any) -> Any:
        del key
        value = _parameter(functions)
        return RandomizedResidualSamples(
            value * (1.0 + collocation),
            sampling_design="iid",
        )

    unbiased = RandomizedResidualTerm(
        evaluator,
        collocation=realizations[0],
        sampling_mode="fixed",
        loss_mode="u_statistic",
    )
    biased = RandomizedResidualTerm(
        evaluator,
        collocation=realizations[0],
        sampling_mode="fixed",
        loss_mode="plug_in",
    )

    def enumerated_loss(value: Any, term: Any) -> Any:
        return jnp.mean(
            jnp.stack(
                tuple(
                    term.loss(
                        _functions(value),
                        batch=phx.terms.RandomizedResidualBatch(
                            noise, *jr.split(jr.key(4))
                        ),
                    )
                    for noise in realizations
                )
            )
        )

    assert jnp.allclose(enumerated_loss(parameter, unbiased), parameter**2)
    assert jnp.allclose(enumerated_loss(parameter, biased), 1.5 * parameter**2)
    assert jnp.allclose(
        jax.grad(lambda value: enumerated_loss(value, unbiased))(parameter),
        2.0 * parameter,
    )
    assert jnp.allclose(
        jax.grad(lambda value: enumerated_loss(value, biased))(parameter),
        3.0 * parameter,
    )
    objective = RandomizedResidualTerm(
        _noisy_evaluator(num_realizations=2, scale=1.0),
        collocation={"count": 1},
        sampling_mode="fixed",
    )
    solver = phx.solver.FunctionalSolver(
        functions=_functions(0.0),
        terms=(objective,),
    )

    with pytest.raises(ValueError, match="keep_best=False"):
        solver.solve(num_iter=1, optim=optax.sgd(0.1), log_every=0)


def test_vector_complex_residuals_masks_and_weights_reduce_correctly() -> None:
    collocation = {
        "residual": jnp.asarray(
            [[1.0 + 2.0j, 0.5 - 0.5j], [3.0 + 0.0j, 4.0j], [9.0, 9.0]]
        ),
        "mask": jnp.asarray([True, True, False]),
        "weights": jnp.asarray([1.0, 3.0, 100.0]),
    }

    def evaluator(functions: Any, batch: Any, key: Any) -> Any:
        del functions, key
        values = jnp.broadcast_to(batch["residual"], (3,) + batch["residual"].shape)
        return RandomizedResidualSamples(
            values,
            sample_shape=(3,),
            event_shape=(2,),
            mask=batch["mask"],
            weights=batch["weights"],
            sampling_design="exact",
        )

    objective = RandomizedResidualTerm(
        evaluator,
        collocation=collocation,
        sampling_mode="fixed",
    )
    expected = (1.0 * (5.0 + 0.5) + 3.0 * (9.0 + 16.0)) / 4.0

    assert jnp.allclose(objective.loss({}, key=jr.key(0)), expected)


def test_resampled_collocation_is_materialized_once_per_optimizer_update() -> None:
    calls = []

    def sampler(key: Any) -> Any:
        calls.append(key)
        return {"target": jnp.asarray(1.0)}

    def evaluator(functions: Any, batch: Any, key: Any) -> Any:
        del key
        residual = _parameter(functions) - batch["target"]
        return RandomizedResidualSamples(
            jnp.stack((residual, residual)), sampling_design="exact"
        )

    objective = RandomizedResidualTerm(
        evaluator,
        collocation=sampler,
        sampling_mode="resample",
    )
    solver = phx.solver.FunctionalSolver(
        functions=_functions(0.0),
        terms=(objective,),
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


def test_zero_valid_mass_is_rejected() -> None:
    def evaluator(functions: Any, batch: Any, key: Any) -> Any:
        del functions, batch, key
        return RandomizedResidualSamples(
            jnp.ones((2, 3)),
            sample_shape=(3,),
            mask=jnp.zeros((3,), dtype="bool"),
            sampling_design="exact",
        )

    objective = RandomizedResidualTerm(
        evaluator,
        collocation={"points": jnp.ones((3, 1))},
        sampling_mode="fixed",
    )

    with pytest.raises(Exception, match="zero valid"):
        objective.loss({}, key=jr.key(0))


def test_complex_vector_iid_objective_and_gradient_by_exact_enumeration() -> None:
    base = jnp.asarray([[1.0 + 2.0j, 2.0j], [3.0, 4.0j], [999.0, 999.0]])
    noise = jnp.asarray([[0.5 + 0.25j, 1.0j], [2.0j, -1.0], [0.0, 0.0]])
    mask = jnp.asarray([True, True, False])
    weights = jnp.asarray([1.0, 3.0, 100.0])
    signs = tuple(product((-1.0, 1.0), repeat=2))

    def evaluate(functions: Any, collocation: Any, key: Any) -> Any:
        del key
        values = _parameter(functions) * (
            base[None, ...] + collocation[:, None, None] * noise[None, ...]
        )
        return RandomizedResidualSamples(
            values,
            sample_shape=(3,),
            event_shape=(2,),
            mask=mask,
            weights=weights,
            sampling_design="iid",
        )

    term = RandomizedResidualTerm(
        evaluate, collocation=(), sampling_mode="fixed", loss_mode="u_statistic"
    )
    keys = jr.split(jr.key(12))

    def expectation(parameter: Any) -> Any:
        return jnp.mean(
            jnp.stack(
                tuple(
                    term.loss(
                        _functions(parameter),
                        batch=phx.terms.RandomizedResidualBatch(
                            jnp.asarray(realization), *keys
                        ),
                    )
                    for realization in signs
                )
            )
        )

    parameter = jnp.asarray(0.7)
    squared_norm = (9.0 + 3.0 * 25.0) / 4.0
    assert jnp.allclose(expectation(parameter), parameter**2 * squared_norm)
    assert jnp.allclose(jax.grad(expectation)(parameter), 2.0 * parameter * squared_norm)

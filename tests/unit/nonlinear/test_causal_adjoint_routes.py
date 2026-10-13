from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from phydrax import nonlinear as nl


def transition(matrix: Array, state: Array, driver: Array) -> Array:
    return jnp.tanh(matrix @ state + driver)


@pytest.mark.parametrize(
    "route", ["dense-associative", "action-scan", "checkpointed-action-scan"]
)
@pytest.mark.parametrize("horizon", [1, 5, 6])
def test_causal_routes_match_independent_recurrence(
    route: nl.CausalAdjointRoute, horizon: int
) -> None:
    matrix = jnp.asarray([[0.3, -0.2], [0.1, 0.4]], dtype=jnp.float64)
    initial = jnp.asarray([0.1, -0.1], dtype=jnp.float64)
    drivers = jnp.linspace(-0.2, 0.3, horizon * 2, dtype=jnp.float64).reshape(horizon, 2)
    policy = nl.CausalAdjointPolicy(route, checkpoint_block_size=2)

    def serial(parameter: Array, boundary: Array, forcing: Array) -> Array:
        def step(state: Array, driver: Array) -> tuple[Array, Array]:
            value = transition(parameter, state, driver)
            return value, value

        return jax.lax.scan(step, boundary, forcing)[1]

    def causal(parameter: Array, boundary: Array, forcing: Array) -> Array:
        result = nl.solve_causal_recurrence(
            nl.CausalRecurrenceProblem(
                transition, boundary, forcing, parameters=parameter
            ),
            method=nl.CausalNewton(),
            adjoint=policy,
            termination=nl.NonlinearTermination(
                absolute_residual=1e-12, relative_residual=1e-12, maximum_steps=12
            ),
        )
        assert result.adjoint_route == route
        assert result.adjoint_policy_id == policy.policy_id
        return result.states

    arguments = (matrix, initial, drivers)
    tangents = tuple(jnp.full_like(value, 0.03) for value in arguments)
    value, tangent = jax.jit(lambda *args: jax.jvp(causal, args, tangents))(*arguments)
    reference, reference_tangent = jax.jvp(serial, arguments, tangents)
    assert jnp.allclose(value, reference, atol=1e-11)
    assert jnp.allclose(tangent, reference_tangent, atol=1e-11)
    weights = jnp.arange(horizon * 2, dtype=jnp.float64).reshape(horizon, 2) + 1
    cotangents = jax.jit(lambda *args: jax.vjp(causal, *args)[1](weights))(*arguments)
    reference_cotangents = jax.vjp(serial, *arguments)[1](weights)
    for actual, expected in zip(cotangents, reference_cotangents, strict=True):
        assert jnp.allclose(actual, expected, atol=1e-11)
    assert jnp.allclose(
        jnp.sum(weights * tangent),
        sum(jnp.sum(t * c) for t, c in zip(tangents, cotangents, strict=True)),
        atol=1e-11,
    )


def test_dense_causal_adjoint_refuses_before_allocation() -> None:
    problem = nl.CausalRecurrenceProblem(
        transition, jnp.zeros(2), jnp.zeros((5, 2)), parameters=jnp.eye(2)
    )
    with pytest.raises(ValueError, match="action-scan.*checkpointed-action-scan"):
        nl.solve_causal_recurrence(
            problem, adjoint=nl.CausalAdjointPolicy(maximum_dense_bytes=1)
        )


def test_causal_adjoint_policy_identity_and_logical_storage() -> None:
    dense = nl.CausalAdjointPolicy()
    scan = nl.CausalAdjointPolicy("action-scan")
    checkpointed = nl.CausalAdjointPolicy("checkpointed-action-scan")
    assert len({dense.policy_id, scan.policy_id, checkpointed.policy_id}) == 3
    assert dense.transition_storage_bytes(8, 4, 8) == 8 * 4 * 4 * 8
    assert scan.transition_storage_bytes(8, 4, 8) == 0
    with pytest.raises(ValueError):
        nl.CausalAdjointPolicy("unknown")  # ty: ignore[invalid-argument-type]

# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from phydrax.domain import FunctionBinding, TimeInterval
from phydrax.operators.differential import dt_n
from phydrax.operators.differential._runtime import derivative_runtime_context
from phydrax.typing import PRNGKey


def test_derivative_context_preserves_changed_points_and_function_draws() -> None:
    domain = TimeInterval(0.0, 1.0)

    @domain.Function("t", binding=FunctionBinding(pass_key=True))
    def field(time: Array, *, key: PRNGKey | None = None) -> Array:
        if key is None:
            raise ValueError("A complete function draw requires a key.")
        amplitude = jr.uniform(key, (), dtype=time.dtype)
        return amplitude * time**3

    derivative = dt_n(field, var="t", order=2, backend="jet")
    first_key, second_key = jr.split(jr.key(13))
    first_point = jnp.asarray(0.25, dtype=jnp.float64)
    second_point = jnp.asarray(0.4, dtype=jnp.float64)
    first_amplitude = jr.uniform(first_key, (), dtype=jnp.float64)
    second_amplitude = jr.uniform(second_key, (), dtype=jnp.float64)
    with derivative_runtime_context():
        first = derivative.func(first_point, key=first_key)
        repeated = derivative.func(first_point, key=first_key)
        changed_point = derivative.func(second_point, key=first_key)
        changed_draw = derivative.func(first_point, key=second_key)
    with derivative_runtime_context():
        next_context = derivative.func(second_point, key=second_key)

    np.testing.assert_allclose(first, 6 * first_point * first_amplitude, rtol=1e-12)
    np.testing.assert_allclose(repeated, first, rtol=1e-12)
    np.testing.assert_allclose(
        changed_point, 6 * second_point * first_amplitude, rtol=1e-12
    )
    np.testing.assert_allclose(
        changed_draw, 6 * first_point * second_amplitude, rtol=1e-12
    )
    np.testing.assert_allclose(
        next_context, 6 * second_point * second_amplitude, rtol=1e-12
    )

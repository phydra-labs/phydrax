from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax.linalg as la
from phydrax._custom_root import custom_root


def _newton_sqrt(target: jax.Array, *, has_aux: bool = False) -> Any:
    def residual(value: jax.Array) -> jax.Array:
        return value**2 - target

    def solve(function: Callable[[jax.Array], jax.Array], guess: jax.Array) -> Any:
        root = jax.lax.fori_loop(0, 40, lambda _, x: 0.5 * (x + target / x), guess)
        return (root, jnp.int32(40)) if has_aux else root

    def tangent_solve(
        linearized: Callable[[jax.Array], jax.Array], right_hand_side: jax.Array
    ) -> jax.Array:
        return right_hand_side / linearized(jnp.ones_like(right_hand_side))

    return custom_root(
        residual, jnp.ones_like(target), solve, tangent_solve, has_aux=has_aux
    )


def test_vmapped_root_prepares_linearization_with_implicit_derivatives() -> None:
    targets = jnp.array([0.25, 4.0, 9.0], dtype=jnp.float64)
    tangent = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
    derivative = 0.5 / jnp.sqrt(targets)

    linearization = la.prepare_linearization(jax.vmap(_newton_sqrt), targets)

    np.testing.assert_allclose(linearization.primal, jnp.sqrt(targets), rtol=1e-14)
    np.testing.assert_allclose(
        linearization.jvp(tangent), derivative * tangent, rtol=1e-13
    )
    np.testing.assert_allclose(
        linearization.vjp(tangent), derivative * tangent, rtol=1e-13
    )


def test_vmapped_root_auxiliary_output_is_derivative_free() -> None:
    targets = jnp.array([1.0, 16.0], dtype=jnp.float64)
    tangent = jnp.array([2.0, 4.0], dtype=jnp.float64)

    (root, iterations), pushforward = jax.linearize(
        jax.vmap(lambda target: _newton_sqrt(target, has_aux=True)), targets
    )
    root_tangent, iterations_tangent = pushforward(tangent)

    np.testing.assert_allclose(root, jnp.array([1.0, 4.0]), rtol=1e-14)
    np.testing.assert_array_equal(iterations, jnp.array([40, 40], dtype=jnp.int32))
    np.testing.assert_allclose(root_tangent, jnp.array([1.0, 0.5]), rtol=1e-13)
    assert iterations_tangent.dtype == jax.dtypes.float0

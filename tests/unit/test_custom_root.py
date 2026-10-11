from collections.abc import Callable
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_pytree_root_shape_probe_preserves_captured_parameter_derivatives() -> None:
    def root(parameters: jax.Array) -> dict[str, jax.Array]:
        def residual(value: dict[str, jax.Array]) -> dict[str, jax.Array]:
            return {
                "left": value["left"] ** 2 - parameters[0],
                "right": value["right"] ** 2 - parameters[1],
            }

        def solve(
            _function: Callable[[dict[str, jax.Array]], dict[str, jax.Array]],
            _guess: dict[str, jax.Array],
        ) -> dict[str, jax.Array]:
            return {"left": jnp.sqrt(parameters[0]), "right": jnp.sqrt(parameters[1])}

        def tangent_solve(
            linearized: Callable[[dict[str, jax.Array]], dict[str, jax.Array]],
            right_hand_side: dict[str, jax.Array],
        ) -> dict[str, jax.Array]:
            diagonal = linearized(
                {
                    "left": jnp.asarray(1.0, dtype=jnp.float64),
                    "right": jnp.asarray(1.0, dtype=jnp.float64),
                }
            )
            return {
                "left": right_hand_side["left"] / diagonal["left"],
                "right": right_hand_side["right"] / diagonal["right"],
            }

        return custom_root(
            residual,
            {
                "left": jnp.asarray(1.0, dtype=jnp.float64),
                "right": jnp.asarray(1.0, dtype=jnp.float64),
            },
            solve,
            tangent_solve,
        )

    parameters = jnp.asarray([3.0, 12.0], dtype=jnp.float64)
    direction = jnp.asarray([0.7, -1.2], dtype=jnp.float64)
    value, tangent = jax.jvp(root, (parameters,), (direction,))
    cotangent = {
        "left": jnp.asarray(2.0, dtype=jnp.float64),
        "right": jnp.asarray(-0.3, dtype=jnp.float64),
    }
    _, pullback = jax.vjp(root, parameters)
    expected = np.sqrt(np.asarray(parameters))
    np.testing.assert_allclose(
        np.asarray([value["left"], value["right"]]), expected, rtol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray([tangent["left"], tangent["right"]]),
        0.5 * np.asarray(direction) / expected,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        pullback(cotangent)[0], 0.5 * np.asarray([2.0, -0.3]) / expected, rtol=1e-13
    )


@pytest.mark.parametrize(
    "residual",
    [
        pytest.param(lambda value: {"wrong": value}, id="changed-tree"),
        pytest.param(lambda value: object(), id="unsupported-static-output"),
    ],
)
def test_root_shape_probe_refuses_invalid_residual_outputs(
    residual: Callable[[jax.Array], Any],
) -> None:
    def solve(_function: Callable[[jax.Array], Any], _guess: jax.Array) -> NoReturn:
        raise RuntimeError("Invalid residual must be refused before primal solve.")

    def tangent_solve(_function: Callable[[jax.Array], Any], _rhs: jax.Array) -> NoReturn:
        raise RuntimeError("Invalid residual must be refused before tangent solve.")

    with pytest.raises(TypeError):
        custom_root(
            residual,
            jnp.asarray(1.0, dtype=jnp.float64),
            solve,
            tangent_solve,
        )

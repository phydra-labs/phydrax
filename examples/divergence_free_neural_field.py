"""Train a neural one-form potential; its vector field is star d psi in 3-D."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from phydrax.exterior import exterior_derivative_from_jacobian, FormType, hodge_star
from phydrax.optim import adam


type NeuralParameters = tuple[Array, Array, Array]


def potential(parameters: NeuralParameters, x: Array) -> Array:
    input_weights, bias, output_weights = parameters
    return output_weights @ jnp.tanh(input_weights @ x + bias)


def velocity(parameters: NeuralParameters, x: Array) -> Array:
    form_type = FormType(3, 1)
    jacobian = jax.jacfwd(potential, argnums=1)(parameters, x)
    curl_form = exterior_derivative_from_jacobian(jacobian, form_type)
    return hodge_star(
        curl_form,
        form_type.exterior_derivative_type(),
        jnp.eye(3, dtype=x.dtype),
        jnp.asarray(1.0, dtype=x.dtype),
    )


def target(x: Array) -> Array:
    return jnp.stack((-jnp.sin(x[1]), jnp.sin(x[0]), jnp.zeros((), dtype=x.dtype)))


def train(steps: int = 150) -> tuple[NeuralParameters, Array, Array]:
    keys = jax.random.split(jax.random.key(11), 3)
    parameters = (
        jax.random.normal(keys[0], (12, 3), dtype=jnp.float64) * 0.4,
        jnp.zeros((12,), dtype=jnp.float64),
        jax.random.normal(keys[1], (3, 12), dtype=jnp.float64) * 0.1,
    )
    points = jax.random.uniform(
        keys[2], (64, 3), dtype=jnp.float64, minval=-1.0, maxval=1.0
    )
    observations = jax.vmap(target)(points)
    optimizer = adam(0.02)
    state = optimizer.init(parameters)

    def objective(weights: NeuralParameters) -> Array:
        prediction = jax.vmap(velocity, in_axes=(None, 0))(weights, points)
        return jnp.mean(jnp.square(prediction - observations))

    def step(
        carry: tuple[NeuralParameters, optax.OptState], _: None
    ) -> tuple[tuple[NeuralParameters, optax.OptState], Array]:
        weights, optimizer_state = carry
        loss, gradients = jax.value_and_grad(objective)(weights)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, weights)
        return (eqx.apply_updates(weights, updates), optimizer_state), loss

    (parameters, _), losses = jax.jit(
        lambda weights, optimizer_state: jax.lax.scan(
            step, (weights, optimizer_state), None, length=steps
        )
    )(parameters, state)
    jacobians = jax.vmap(jax.jacfwd(velocity, argnums=1), in_axes=(None, 0))(
        parameters, points
    )
    divergence = jnp.trace(jacobians, axis1=-2, axis2=-1)
    return parameters, losses, divergence


def main() -> None:
    _, losses, divergence = train()
    print(
        f"Neural potential fitting: loss {float(losses[0]):.3e} -> {float(losses[-1]):.3e}"
    )
    print(f"AD divergence of star d psi: {float(jnp.max(jnp.abs(divergence))):.3e}")
    if not bool(losses[-1] < losses[0] * 0.1) or not bool(
        jnp.max(jnp.abs(divergence)) < 1e-11
    ):
        raise RuntimeError("Potential learning or structural solenoidality failed.")


if __name__ == "__main__":
    main()

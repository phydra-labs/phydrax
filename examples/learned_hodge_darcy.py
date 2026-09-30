"""Learn a positive Darcy Hodge from flux observations without a divergence penalty.

The mixed stream-function representation q = star_1^-1 d_1^T psi makes the
source-free Darcy conservation equation exact for every optimizer iterate.
The learned diagonal is the inverse-conductivity Riesz map; no material loss
or indefinite coefficient is admitted as a metric.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

from phydrax.discretization import (
    CellComplexTopology,
    CochainDiscretization,
    cubical_cell_complex,
    DiagonalHodge,
)
from phydrax.optim import adam


def darcy_flux(
    topology: CellComplexTopology, log_weights: Array, stream: Array
) -> tuple[Array, Array]:
    counts = tuple(entity.count for entity in topology.entity_sets)
    realization = CochainDiscretization(
        topology,
        (
            DiagonalHodge(jnp.ones((counts[0],), dtype=jnp.float64)),
            DiagonalHodge(jnp.exp(log_weights)),
            DiagonalHodge(jnp.ones((counts[2],), dtype=jnp.float64)),
        ),
        numeric_revision="darcy-training-binding",
    )
    flux = realization.codifferential(2, stream)
    divergence = -realization.codifferential(1, flux)
    return flux, divergence


def train(steps: int = 200) -> tuple[Array, Array, Array]:
    topology = cubical_cell_complex((4, 4), periodic=True).topology
    counts = tuple(entity.count for entity in topology.entity_sets)
    stream = jnp.sin(0.7 * jnp.arange(counts[2], dtype=jnp.float64))
    truth = 0.3 * jnp.cos(jnp.arange(counts[1], dtype=jnp.float64))
    observed, _ = darcy_flux(topology, truth, stream)
    parameters = jnp.zeros_like(truth)
    optimizer = adam(0.05)
    state = optimizer.init(parameters)

    def objective(weights: Array) -> Array:
        predicted, _ = darcy_flux(topology, weights, stream)
        return jnp.mean(jnp.square(predicted - observed))

    def step(
        carry: tuple[Array, optax.OptState], _: None
    ) -> tuple[tuple[Array, optax.OptState], Array]:
        weights, optimizer_state = carry
        loss, gradient = jax.value_and_grad(objective)(weights)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, weights)
        weights = eqx.apply_updates(weights, updates)
        _flux, divergence = darcy_flux(topology, weights, stream)
        evidence = jnp.stack((loss, jnp.max(jnp.abs(divergence))))
        return (weights, optimizer_state), evidence

    (parameters, _), history = jax.jit(
        lambda initial, initial_state: jax.lax.scan(
            step, (initial, initial_state), None, length=steps
        )
    )(parameters, state)
    return parameters, history[:, 0], history[:, 1]


def main() -> None:
    weights, losses, conservation = train()
    print(f"Darcy flux fitting: loss {float(losses[0]):.3e} -> {float(losses[-1]):.3e}")
    print(
        f"Maximum divergence during actual training: {float(jnp.max(conservation)):.3e}"
    )
    print(f"Minimum learned Hodge weight: {float(jnp.min(jnp.exp(weights))):.6f}")
    if not bool(losses[-1] < losses[0] * 1e-3) or not bool(jnp.max(conservation) < 1e-11):
        raise RuntimeError("Learned-Hodge fitting or conservation failed.")


if __name__ == "__main__":
    main()

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule


class SeparableHamiltonianResult(StrictModule):
    """Final canonical state of fixed-step Störmer–Verlet integration."""

    position: Array
    momentum: Array
    steps: int = eqx.field(static=True)
    step_size: Array
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        position: ArrayLike,
        momentum: ArrayLike,
        /,
        *,
        steps: int,
        step_size: ArrayLike,
    ):
        self.position = jnp.asarray(position)
        self.momentum = jnp.asarray(momentum)
        self.steps = int(steps)
        self.step_size = jnp.asarray(step_size)
        self.method_id = "stormer-verlet"


def stormer_verlet_step(
    position: ArrayLike,
    momentum: ArrayLike,
    potential_gradient: Callable[[Array], Array],
    kinetic_gradient: Callable[[Array], Array],
    step_size: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Advance one separable canonical Hamiltonian step."""
    if not callable(potential_gradient) or not callable(kinetic_gradient):
        raise TypeError("Hamiltonian gradients must be callable.")
    position_ = jnp.asarray(position)
    momentum_ = jnp.asarray(momentum)
    if position_.shape != momentum_.shape:
        raise ValueError("Canonical position and momentum must have one shape.")
    step = jnp.asarray(step_size, dtype=position_.real.dtype)
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    half_momentum = momentum_ - 0.5 * step * potential_gradient(position_)
    next_position = position_ + step * kinetic_gradient(half_momentum)
    next_momentum = half_momentum - 0.5 * step * potential_gradient(next_position)
    if next_position.shape != position_.shape or next_momentum.shape != momentum_.shape:
        raise ValueError("Hamiltonian gradients must preserve canonical state shape.")
    return next_position, next_momentum


def integrate_stormer_verlet(
    position: ArrayLike,
    momentum: ArrayLike,
    potential_gradient: Callable[[Array], Array],
    kinetic_gradient: Callable[[Array], Array],
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> SeparableHamiltonianResult:
    """Integrate a separable canonical Hamiltonian for a fixed number of steps."""
    count = int(steps)
    step = jnp.asarray(step_size, dtype=jnp.asarray(position).real.dtype)
    if count < 0:
        raise ValueError("steps must be non-negative.")
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    step = eqx.error_if(
        step,
        ~jnp.isfinite(step) | (step == 0.0),
        "step_size must be finite and nonzero.",
    )
    initial = (jnp.asarray(position), jnp.asarray(momentum))

    def advance(_, state: tuple[Array, Array]) -> tuple[Array, Array]:
        return stormer_verlet_step(
            state[0],
            state[1],
            potential_gradient,
            kinetic_gradient,
            step,
        )

    final_position, final_momentum = jax.lax.fori_loop(
        0,
        count,
        advance,
        initial,
    )
    return SeparableHamiltonianResult(
        final_position,
        final_momentum,
        steps=count,
        step_size=step,
    )


__all__ = [
    "SeparableHamiltonianResult",
    "integrate_stormer_verlet",
    "stormer_verlet_step",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ._temporal_precision import TemporalPrecisionPolicy


class SeparableHamiltonianResult(StrictModule):
    """Final canonical state of fixed-step Störmer–Verlet integration."""

    position: Array
    momentum: Array
    steps: int = eqx.field(static=True)
    step_size: Array
    precision_evidence: PrecisionEvidenceEnvelope
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        position: ArrayLike,
        momentum: ArrayLike,
        /,
        *,
        steps: int,
        step_size: ArrayLike,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        position_ = jnp.asarray(position)
        step_size_ = jnp.asarray(step_size)
        evidence = (
            TemporalPrecisionPolicy().evidence_for(position_, step_size_)
            if precision_evidence is None
            else precision_evidence
        )
        if not isinstance(evidence, PrecisionEvidenceEnvelope):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.position = position_
        self.momentum = jnp.asarray(momentum)
        self.steps = int(steps)
        self.step_size = step_size_
        self.precision_evidence = evidence
        self.method_id = "stormer-verlet"


def stormer_verlet_step(
    position: ArrayLike,
    momentum: ArrayLike,
    potential_gradient: Callable[[Array], Array],
    kinetic_gradient: Callable[[Array], Array],
    step_size: ArrayLike,
    /,
    *,
    precision: TemporalPrecisionPolicy | None = None,
) -> tuple[Array, Array]:
    """Advance one separable canonical Hamiltonian step."""
    if not callable(potential_gradient) or not callable(kinetic_gradient):
        raise TypeError("Hamiltonian gradients must be callable.")
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    position_ = jnp.asarray(position)
    momentum_ = jnp.asarray(momentum)
    if position_.shape != momentum_.shape:
        raise ValueError("Canonical position and momentum must have one shape.")
    if position_.dtype != momentum_.dtype:
        raise TypeError("Canonical position and momentum must have one dtype.")
    precision_.validate_state(position_)
    precision_.validate_state(momentum_)
    step = precision_.coefficient(jnp.asarray(step_size, dtype=position_.real.dtype))
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    staged_position = precision_.stage(position_)
    staged_momentum = precision_.stage(momentum_)
    half_momentum = precision_.stage(
        precision_.accumulation(staged_momentum)
        - precision_.accumulation(
            0.5 * step * precision_.stage(potential_gradient(staged_position))
        )
    )
    next_position = precision_.stage(
        precision_.accumulation(staged_position)
        + precision_.accumulation(
            step * precision_.stage(kinetic_gradient(half_momentum))
        )
    )
    next_momentum = precision_.stage(
        precision_.accumulation(half_momentum)
        - precision_.accumulation(
            0.5 * step * precision_.stage(potential_gradient(next_position))
        )
    )
    if next_position.shape != position_.shape or next_momentum.shape != momentum_.shape:
        raise ValueError("Hamiltonian gradients must preserve canonical state shape.")
    return (
        jnp.asarray(next_position, dtype=position_.dtype),
        jnp.asarray(next_momentum, dtype=momentum_.dtype),
    )


def integrate_stormer_verlet(
    position: ArrayLike,
    momentum: ArrayLike,
    potential_gradient: Callable[[Array], Array],
    kinetic_gradient: Callable[[Array], Array],
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    precision: TemporalPrecisionPolicy | None = None,
) -> SeparableHamiltonianResult:
    """Integrate a separable canonical Hamiltonian for a fixed number of steps."""
    count = int(steps)
    initial = (jnp.asarray(position), jnp.asarray(momentum))
    step = jnp.asarray(step_size, dtype=initial[0].real.dtype)
    if count < 0:
        raise ValueError("steps must be non-negative.")
    if step.shape != ():
        raise ValueError("step_size must be scalar.")
    step = eqx.error_if(
        step,
        ~jnp.isfinite(step) | (step == 0.0),
        "step_size must be finite and nonzero.",
    )
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    precision_.validate_state(initial[0])
    precision_.validate_state(initial[1])

    def advance(_, state: tuple[Array, Array]) -> tuple[Array, Array]:
        return stormer_verlet_step(
            state[0],
            state[1],
            potential_gradient,
            kinetic_gradient,
            step,
            precision=precision_,
        )

    final_position, final_momentum = jax.lax.fori_loop(
        0,
        count,
        advance,
        initial,
    )
    return SeparableHamiltonianResult(
        precision_.output(final_position),
        precision_.output(final_momentum),
        steps=count,
        step_size=step,
        precision_evidence=precision_.evidence_for(initial[0], step),
    )


__all__ = [
    "SeparableHamiltonianResult",
    "integrate_stormer_verlet",
    "stormer_verlet_step",
]

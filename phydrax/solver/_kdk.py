#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


class KDKCoefficients(StrictModule):
    first_kick: Array
    drift: Array
    second_kick: Array

    def __init__(
        self,
        first_kick: ArrayLike,
        drift: ArrayLike,
        second_kick: ArrayLike,
        /,
    ):
        first = jnp.asarray(first_kick).reshape(())
        drift_ = jnp.asarray(drift, dtype=first.dtype).reshape(())
        second = jnp.asarray(second_kick, dtype=first.dtype).reshape(())
        self.first_kick = first
        self.drift = drift_
        self.second_kick = second


class KDKProposal(StrictModule):
    positions: Array
    half_momenta: Array
    coefficients: KDKCoefficients
    finite: Array
    successful: Array


class KDKCompletion(StrictModule):
    positions: Array
    momenta: Array
    finite: Array
    successful: Array


class KDKTransactionPlan(StrictModule, NonTrainableState):
    periodic_box: tuple[float, ...] | None = eqx.field(static=True)

    def __init__(self, periodic_box: tuple[float, ...] | None = None, /):
        self.periodic_box = (
            None
            if periodic_box is None
            else tuple(float(value) for value in periodic_box)
        )

    def propose(
        self,
        positions: ArrayLike,
        momenta: ArrayLike,
        masses: ArrayLike,
        acceleration_start: ArrayLike,
        coefficients: KDKCoefficients,
        /,
    ) -> KDKProposal:
        position = jnp.asarray(positions)
        momentum = jnp.asarray(momenta, dtype=position.dtype)
        mass = jnp.asarray(masses, dtype=position.dtype)
        acceleration = jnp.asarray(acceleration_start, dtype=position.dtype)
        if (
            momentum.shape != position.shape
            or acceleration.shape != position.shape
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("KDK state, acceleration, and masses are incompatible.")
        half = momentum + coefficients.first_kick * mass[:, None] * acceleration
        candidate = position + coefficients.drift * half / mass[:, None]
        if self.periodic_box is not None:
            candidate = jnp.mod(
                candidate, jnp.asarray(self.periodic_box, dtype=candidate.dtype)
            )
        finite = jnp.all(jnp.isfinite(candidate)) & jnp.all(jnp.isfinite(half))
        return KDKProposal(candidate, half, coefficients, finite, finite)

    def complete(
        self,
        proposal: KDKProposal,
        masses: ArrayLike,
        acceleration_end: ArrayLike,
        /,
    ) -> KDKCompletion:
        mass = jnp.asarray(masses, dtype=proposal.positions.dtype)
        acceleration = jnp.asarray(acceleration_end, dtype=proposal.positions.dtype)
        if acceleration.shape != proposal.positions.shape or mass.shape != (
            proposal.positions.shape[0],
        ):
            raise ValueError("KDK endpoint acceleration/masses are incompatible.")
        momentum = (
            proposal.half_momenta
            + proposal.coefficients.second_kick * mass[:, None] * acceleration
        )
        finite = proposal.successful & jnp.all(jnp.isfinite(momentum))
        return KDKCompletion(proposal.positions, momentum, finite, finite)


__all__ = [
    "KDKCoefficients",
    "KDKCompletion",
    "KDKProposal",
    "KDKTransactionPlan",
]

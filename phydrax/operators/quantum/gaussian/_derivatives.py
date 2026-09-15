#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic AD derivatives of fixed-topology Gaussian integral kernels."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._basis import PreparedGaussianBasis
from ._integrals import (
    electron_repulsion_tensor,
    kinetic_matrix,
    nuclear_attraction_matrix,
    overlap_matrix,
)


class GaussianIntegralDerivativeResult(StrictModule):
    overlap: Array
    kinetic: Array
    nuclear_attraction: Array
    electron_repulsion: Array | None
    derivative_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    successful: Array


class GaussianIntegralDerivativePlan(StrictModule, NonTrainableState):
    derivative_order: int = eqx.field(static=True)
    include_electron_repulsion: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        derivative_order: int = 1,
        /,
        *,
        include_electron_repulsion: bool = False,
    ):
        order = int(derivative_order)
        if order not in (1, 2):
            raise ValueError("Gaussian integral derivative order must be one or two.")
        self.derivative_order = order
        self.include_electron_repulsion = bool(include_electron_repulsion)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-integral-derivative-plan",
                "derivative_order": order,
                "include_electron_repulsion": self.include_electron_repulsion,
            }
        )

    def evaluate(
        self,
        basis: PreparedGaussianBasis,
        positions: ArrayLike,
        nuclear_charges: ArrayLike,
        /,
    ) -> GaussianIntegralDerivativeResult:
        coordinate = jnp.asarray(positions)
        charges = jnp.asarray(nuclear_charges, dtype=coordinate.dtype)

        def differentiate(function):
            derivative = jax.jacfwd(function)
            return (
                derivative(coordinate)
                if self.derivative_order == 1
                else jax.jacfwd(derivative)(coordinate)
            )

        overlap = differentiate(lambda value: overlap_matrix(basis, value))
        kinetic = differentiate(lambda value: kinetic_matrix(basis, value))
        attraction = differentiate(
            lambda value: nuclear_attraction_matrix(basis, value, charges)
        )
        eri = (
            differentiate(lambda value: electron_repulsion_tensor(basis, value))
            if self.include_electron_repulsion
            else None
        )
        finite = (
            jnp.all(jnp.isfinite(overlap))
            & jnp.all(jnp.isfinite(kinetic))
            & jnp.all(jnp.isfinite(attraction))
            & (jnp.asarray(True) if eri is None else jnp.all(jnp.isfinite(eri)))
        )
        return GaussianIntegralDerivativeResult(
            overlap,
            kinetic,
            attraction,
            eri,
            self.derivative_order,
            self.plan_id,
            finite,
        )


__all__ = ["GaussianIntegralDerivativePlan", "GaussianIntegralDerivativeResult"]

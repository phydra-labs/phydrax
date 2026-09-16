#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator


class PowerAdjointTransferEvidence(StrictModule):
    source_power: Array
    target_power: Array
    power_defect: Array
    constant_defect: Array
    finite: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class PowerAdjointTransferPair(StrictModule, NonTrainableState):
    """Primal state transfer paired with its physical dual force transfer."""

    primal: AbstractLinearOperator
    dual: AbstractLinearOperator
    tolerance: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal: AbstractLinearOperator,
        dual: AbstractLinearOperator,
        /,
        *,
        tolerance: float = 1.0e-10,
        transfer_id: str,
    ):
        if not isinstance(primal, AbstractLinearOperator) or not isinstance(
            dual, AbstractLinearOperator
        ):
            raise TypeError("Power-adjoint transfer requires two linear operators.")
        if primal.source != dual.target or primal.target != dual.source:
            raise ValueError("Primal and dual transfer spaces must be reversed.")
        tolerance_ = float(tolerance)
        if not jnp.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Power-adjoint tolerance must be nonnegative.")
        identifier = str(transfer_id)
        if not identifier:
            raise ValueError("Power-adjoint transfer requires a stable ID.")
        self.primal = primal
        self.dual = dual
        self.tolerance = tolerance_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "power-adjoint-transfer-pair",
                "declared_id": identifier,
                "primal": primal.operator_id,
                "dual": dual.operator_id,
                "tolerance": tolerance_,
            }
        )

    def evaluate(
        self,
        source_velocity: ArrayLike,
        target_force: ArrayLike,
        /,
    ) -> PowerAdjointTransferEvidence:
        source = self.primal.source.validate(source_velocity)
        force = self.primal.target.validate(target_force)
        target_velocity = self.primal.mv(source)
        source_force = self.dual.mv(force)
        source_power = self.primal.source.inner(source, source_force)
        target_power = self.primal.target.inner(target_velocity, force)
        defect = target_power - source_power
        source_constant = jax.tree.map(
            lambda spec: jnp.ones(spec.shape, spec.dtype),
            self.primal.source.structure(),
        )
        target_constant = jax.tree.map(
            lambda spec: jnp.ones(spec.shape, spec.dtype),
            self.primal.target.structure(),
        )
        target_error = jax.tree.map(
            lambda left, right: left - right,
            self.primal.mv(source_constant),
            target_constant,
        )
        constant_defect = jnp.sqrt(
            jnp.real(self.primal.target.inner(target_error, target_error))
        )
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(source_power), jnp.abs(target_power)), 1.0
        )
        finite = (
            jnp.isfinite(source_power)
            & jnp.isfinite(target_power)
            & jnp.isfinite(constant_defect)
        )
        successful = (
            finite
            & (jnp.abs(defect) <= self.tolerance * scale)
            & (constant_defect <= self.tolerance)
        )
        return PowerAdjointTransferEvidence(
            source_power,
            target_power,
            defect,
            constant_defect,
            finite,
            successful,
            self.transfer_id,
        )


__all__ = ["PowerAdjointTransferEvidence", "PowerAdjointTransferPair"]

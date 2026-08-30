#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    PreconditionerProperties,
)


class LowOrderAuxiliaryOperatorPlan(StrictModule, NonTrainableState):
    interpolation: AbstractLinearOperator
    anterpolation: AbstractLinearOperator
    multiplicity_weight: object
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interpolation: AbstractLinearOperator,
        anterpolation: AbstractLinearOperator,
        multiplicity_weight: object,
        /,
    ):
        if not isinstance(interpolation, AbstractLinearOperator) or not isinstance(
            anterpolation, AbstractLinearOperator
        ):
            raise TypeError("Auxiliary transfers must be linear operators.")
        if not interpolation.source.compatible(
            anterpolation.target
        ) or not interpolation.target.compatible(anterpolation.source):
            raise ValueError(
                "Auxiliary interpolation/anterpolation spaces are incompatible."
            )
        weight = interpolation.source.validate(multiplicity_weight)
        leaves = jax.tree.leaves(weight)
        if any(bool(jnp.any(~jnp.isfinite(value) | (value <= 0.0))) for value in leaves):
            raise ValueError(
                "Auxiliary multiplicity weights must be positive and finite."
            )
        self.interpolation = interpolation
        self.anterpolation = anterpolation
        self.multiplicity_weight = weight
        self.plan_id = canonical_fingerprint(
            {
                "kind": "low-order-auxiliary-operator-plan",
                "interpolation": interpolation.operator_id,
                "anterpolation": anterpolation.operator_id,
                "high_space": interpolation.source.space_id,
                "low_space": interpolation.target.space_id,
            }
        )


class LowOrderAuxiliaryPreconditioner(AbstractPreconditioner):
    plan: LowOrderAuxiliaryOperatorPlan
    low_order_preconditioner: AbstractPreconditioner

    def __init__(
        self,
        plan: LowOrderAuxiliaryOperatorPlan,
        low_order_preconditioner: AbstractPreconditioner,
        /,
    ):
        if not isinstance(plan, LowOrderAuxiliaryOperatorPlan) or not isinstance(
            low_order_preconditioner, AbstractPreconditioner
        ):
            raise TypeError("Auxiliary preconditioner inputs are invalid.")
        if not low_order_preconditioner.space.compatible(plan.interpolation.target):
            raise ValueError("Low-order preconditioner acts on the wrong space.")
        self.plan = plan
        self.low_order_preconditioner = low_order_preconditioner
        self.space = plan.interpolation.source
        self.properties = PreconditionerProperties(
            linear=low_order_preconditioner.properties.certifies("linear"),
            stationary=low_order_preconditioner.properties.certifies("stationary"),
            evidence={
                name: "transformed"
                for name in ("linear", "stationary")
                if low_order_preconditioner.properties.certifies(name)
            },
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "low-order-auxiliary-preconditioner",
                "plan": plan.plan_id,
                "inner": low_order_preconditioner.preconditioner_id,
            }
        )

    def apply(
        self,
        residual: PyTree,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        checked = self.space.validate(residual)
        weighted = jax.tree.map(
            lambda value, weight: value * weight,
            checked,
            self.plan.multiplicity_weight,
        )
        low_residual = self.plan.interpolation.mv(weighted)
        low_correction = self.low_order_preconditioner.apply(
            low_residual,
            iteration=iteration,
        )
        high_correction = self.plan.anterpolation.mv(low_correction)
        return self.space.validate(
            jax.tree.map(
                lambda value, weight: value * weight,
                high_correction,
                self.plan.multiplicity_weight,
            )
        )


__all__ = [
    "LowOrderAuxiliaryOperatorPlan",
    "LowOrderAuxiliaryPreconditioner",
]

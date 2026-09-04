#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    DifferentiationPolicy,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    solve,
    TolerancePolicy,
)


CompositeGaugeProjector = Callable[[Array], Array]


class CompositeMACProjectionResult(StrictModule):
    velocity: object
    pressure: Array
    divergence_before: Array
    divergence_after: Array
    divergence_norm: Array
    linear: LinearSolveResult
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class CompositeMACProjectionPlan(StrictModule, NonTrainableState):
    """Composite-grid projection from compatible divergence/gradient operators."""

    divergence: AbstractLinearOperator
    gradient: AbstractLinearOperator
    inverse_momentum: AbstractLinearOperator
    gauge_project: CompositeGaugeProjector
    linear_policy: LinearSolvePolicy
    pressure_operator: FunctionLinearOperator
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        divergence: AbstractLinearOperator,
        gradient: AbstractLinearOperator,
        inverse_momentum: AbstractLinearOperator,
        gauge_project: CompositeGaugeProjector,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        tolerance: float = 1.0e-9,
    ):
        if not divergence.source.compatible(gradient.target) or not (
            divergence.target.compatible(gradient.source)
        ):
            raise ValueError("Composite divergence and gradient spaces are incompatible.")
        if not inverse_momentum.source.compatible(divergence.source) or not (
            inverse_momentum.target.compatible(divergence.source)
        ):
            raise ValueError("Composite inverse momentum must act on velocity space.")
        if not callable(gauge_project):
            raise TypeError("gauge_project must be callable.")
        tolerance_ = float(tolerance)
        if tolerance_ <= 0.0:
            raise ValueError("Composite projection tolerance must be positive.")
        policy = (
            LinearSolvePolicy(
                GMRES(restart=50),
                tolerance=TolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    max_steps=1000,
                ),
                differentiation=DifferentiationPolicy("mathematical"),
            )
            if linear_policy is None
            else linear_policy
        )

        def pressure_action(pressure):
            gauged = gauge_project(pressure)
            gradient_value = gradient.mv(gauged)
            inverse_value = inverse_momentum.mv(gradient_value)
            physical = jax.tree.map(lambda value: -value, divergence.mv(inverse_value))
            null_component = jax.tree.map(
                lambda value, projected: value - projected,
                pressure,
                gauged,
            )
            return jax.tree.map(
                lambda value, null: value + null,
                gauge_project(physical),
                null_component,
            )

        identifier = canonical_fingerprint(
            {
                "kind": "composite-mac-pressure-projection",
                "divergence": divergence.operator_id,
                "gradient": gradient.operator_id,
                "inverse_momentum": inverse_momentum.operator_id,
                "linear_method": policy.method.name,
                "tolerance": tolerance_,
            }
        )
        self.divergence = divergence
        self.gradient = gradient
        self.inverse_momentum = inverse_momentum
        self.gauge_project = gauge_project
        self.linear_policy = policy
        self.pressure_operator = FunctionLinearOperator(
            pressure_action,
            source=divergence.target,
            target=divergence.target,
            transpose_action=pressure_action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=f"composite-pressure/{identifier}",
        )
        self.tolerance = tolerance_
        self.plan_id = identifier

    def project(
        self,
        velocity,
        /,
        *,
        pressure: Array | None = None,
    ) -> CompositeMACProjectionResult:
        velocity_ = self.divergence.source.validate(velocity)
        divergence_before = self.divergence.mv(velocity_)
        right_hand_side = jax.tree.map(
            lambda value: -value,
            self.gauge_project(divergence_before),
        )
        initial = (
            self.divergence.target.zeros()
            if pressure is None
            else self.divergence.target.validate(pressure)
        )
        linear = solve(
            LinearSystem(
                self.pressure_operator,
                problem_id=f"composite-pressure/{self.plan_id}",
            ),
            right_hand_side,
            policy=self.linear_policy,
            initial_guess=initial,
        )
        pressure_value = self.gauge_project(linear.value)
        correction = self.inverse_momentum.mv(self.gradient.mv(pressure_value))
        projected = jax.tree.map(
            lambda value, update: value - update,
            velocity_,
            correction,
        )
        divergence_after = self.divergence.mv(projected)
        divergence_norm = jnp.sqrt(
            jnp.real(self.divergence.target.inner(divergence_after, divergence_after))
        )
        finite = jnp.isfinite(divergence_norm) & jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(projected)
                )
            )
        )
        scale = jnp.sqrt(
            jnp.real(self.divergence.target.inner(divergence_before, divergence_before))
        )
        accepted = (
            linear.successful
            & finite
            & (divergence_norm <= self.tolerance * jnp.maximum(scale, 1.0))
        )
        return CompositeMACProjectionResult(
            projected,
            pressure_value,
            divergence_before,
            divergence_after,
            divergence_norm,
            linear,
            finite,
            accepted,
            self.plan_id,
        )


__all__ = [
    "CompositeGaugeProjector",
    "CompositeMACProjectionPlan",
    "CompositeMACProjectionResult",
]

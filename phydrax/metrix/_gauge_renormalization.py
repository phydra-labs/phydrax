#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule


class GaugeRenormalizationEvidence(StrictModule):
    inverse_residual: Array
    function_residual: Array
    operator_residual: Array
    state_residual: Array
    finite: Array
    valid: Array
    gauge_kind: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        inverse_residual: ArrayLike,
        function_residual: ArrayLike,
        operator_residual: ArrayLike,
        state_residual: ArrayLike,
        finite: ArrayLike,
        valid: ArrayLike,
        gauge_kind: str,
    ):
        self.inverse_residual = jnp.asarray(inverse_residual)
        self.function_residual = jnp.asarray(function_residual)
        self.operator_residual = jnp.asarray(operator_residual)
        self.state_residual = jnp.asarray(state_residual)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.gauge_kind = str(gauge_kind)


class GaugeRenormalizationPlan(StrictModule):
    """Declared gauge action and optimizer-state transport for one fixed epoch.

    The reverse action callbacks receive the element returned by ``gauge_inverse``;
    they never receive an elementwise reciprocal manufactured by this plan.
    """

    action: Callable[[PyTree, Array], PyTree]
    inverse_action: Callable[[PyTree, Array], PyTree]
    state_transport: Callable[[PyTree, Array], PyTree]
    gauge_inverse: Callable[[Array], Array]
    inverse_state_transport: Callable[[PyTree, Array], PyTree]
    gauge_kind: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[PyTree, Array], PyTree],
        inverse_action: Callable[[PyTree, Array], PyTree],
        state_transport: Callable[[PyTree, Array], PyTree],
        /,
        *,
        gauge_inverse: Callable[[Array], Array],
        inverse_state_transport: Callable[[PyTree, Array], PyTree],
        gauge_kind: str,
        tolerance: float = 1e-8,
        plan_id: str,
    ):
        supported = (
            "complex_scalar",
            "quaternion_unit",
            "g2",
            "spin",
            "positive_scaling",
        )
        callables = (
            action,
            inverse_action,
            state_transport,
            gauge_inverse,
            inverse_state_transport,
        )
        if gauge_kind not in supported or not all(callable(value) for value in callables):
            raise ValueError(
                "Gauge action, inverse, and state transports must use a supported declared family."
            )
        if float(tolerance) <= 0.0 or not plan_id:
            raise ValueError("Gauge tolerance/id are invalid.")
        self.action = action
        self.inverse_action = inverse_action
        self.state_transport = state_transport
        self.gauge_inverse = gauge_inverse
        self.inverse_state_transport = inverse_state_transport
        self.gauge_kind = str(gauge_kind)
        self.tolerance = float(tolerance)
        self.plan_id = str(plan_id)

    @staticmethod
    def _tree_residual(left: PyTree, right: PyTree, /) -> Array:
        left_structure = jax.tree.structure(left)
        if left_structure != jax.tree.structure(right):
            return jnp.asarray(jnp.inf)
        leaves = tuple(
            jnp.max(jnp.abs(a - b))
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        )
        if not leaves:
            return jnp.asarray(0.0)
        return jnp.max(jnp.stack(leaves))

    def apply(
        self, parameters: PyTree, gauge: ArrayLike, optimizer_state: PyTree, /
    ) -> tuple[PyTree, PyTree]:
        gauge_ = jnp.asarray(gauge)
        if (
            jnp.any(~jnp.isfinite(gauge_))
            or (self.gauge_kind == "positive_scaling" and bool(jnp.any(gauge_ <= 0.0)))
            or (self.gauge_kind != "positive_scaling" and bool(jnp.all(gauge_ == 0)))
        ):
            raise ValueError(
                "Gauge element is nonfinite, zero, or outside the declared positive action."
            )
        return self.action(parameters, gauge_), self.state_transport(
            optimizer_state, gauge_
        )

    def _inverse_gauge(self, gauge: Array, /) -> Array:
        inverse = jnp.asarray(self.gauge_inverse(gauge))
        if inverse.shape != gauge.shape or bool(jnp.any(~jnp.isfinite(inverse))):
            raise ValueError(
                "Declared gauge inverse must be finite and preserve the gauge layout."
            )
        return inverse

    def evidence(
        self,
        parameters: PyTree,
        gauge: ArrayLike,
        optimizer_state: PyTree,
        function: Callable[[PyTree], Array],
        operator: Callable[[PyTree], Array],
        /,
    ) -> GaugeRenormalizationEvidence:
        gauge_ = jnp.asarray(gauge)
        transformed, state = self.apply(parameters, gauge_, optimizer_state)
        inverse_gauge = self._inverse_gauge(gauge_)
        reconstructed = self.inverse_action(transformed, inverse_gauge)
        inverse_residual = self._tree_residual(parameters, reconstructed)
        function_residual = jnp.max(jnp.abs(function(parameters) - function(transformed)))
        operator_residual = jnp.max(jnp.abs(operator(parameters) - operator(transformed)))
        reverse_state = self.inverse_state_transport(state, inverse_gauge)
        state_residual = self._tree_residual(optimizer_state, reverse_state)
        residuals = jnp.stack(
            (inverse_residual, function_residual, operator_residual, state_residual)
        )
        finite = jnp.all(jnp.isfinite(residuals))
        valid = finite & jnp.all(residuals <= self.tolerance)
        return GaugeRenormalizationEvidence(
            inverse_residual=inverse_residual,
            function_residual=function_residual,
            operator_residual=operator_residual,
            state_residual=state_residual,
            finite=finite,
            valid=valid,
            gauge_kind=self.gauge_kind,
        )


__all__ = ["GaugeRenormalizationEvidence", "GaugeRenormalizationPlan"]

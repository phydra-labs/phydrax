#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...metrix.algebra import AlgebraProductPlan, BracketingPlan


AnalyticityKind = Literal[
    "complex_holomorphic",
    "slice_regular",
    "left_fueter",
    "right_fueter",
    "left_monogenic",
    "right_monogenic",
    "certified_linear",
]


class AnalyticityOperator(StrictModule):
    action: Callable[[Callable[[Array], Array], Array], Array]
    kind: AnalyticityKind = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: AnalyticityKind,
        action: Callable[[Callable[[Array], Array], Array], Array],
        /,
        *,
        tolerance: float = 1e-6,
        operator_id: str,
    ):
        supported = (
            "complex_holomorphic",
            "slice_regular",
            "left_fueter",
            "right_fueter",
            "left_monogenic",
            "right_monogenic",
            "certified_linear",
        )
        if (
            kind not in supported
            or not callable(action)
            or float(tolerance) <= 0.0
            or not operator_id
        ):
            raise ValueError("Analyticity operator kind/action/tolerance/id are invalid.")
        self.action = action
        self.kind = kind
        self.tolerance = float(tolerance)
        self.operator_id = str(operator_id)

    def residual(
        self, function: Callable[[Array], Array], coordinates: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(self.action(function, jnp.asarray(coordinates)))
        return jnp.max(jnp.abs(value))


class AlgebraAnalyticLayer(StrictModule):
    weights: Array
    bias: Array
    product: AlgebraProductPlan
    side: Literal["left", "right"] = eqx.field(static=True)

    def __init__(
        self,
        weights: ArrayLike,
        bias: ArrayLike,
        product: AlgebraProductPlan,
        /,
        *,
        side: Literal["left", "right"],
    ):
        weights_ = jnp.asarray(weights)
        bias_ = jnp.asarray(bias, dtype=weights_.dtype)
        dimension = product.algebra.coordinate_dimension
        if (
            weights_.ndim != 3
            or weights_.shape[-1] != dimension
            or bias_.shape != (weights_.shape[0], dimension)
        ):
            raise ValueError("Algebra analytic weights/bias have incompatible shapes.")
        if side not in ("left", "right"):
            raise ValueError("Algebra analytic multiplication side must be explicit.")
        self.weights = weights_
        self.bias = bias_
        self.product = product
        self.side = side

    def __call__(self, inputs: ArrayLike, /) -> Array:
        value = jnp.asarray(inputs)
        if value.shape != (self.weights.shape[1], self.weights.shape[2]):
            raise ValueError("Algebra analytic layer input has the wrong shape.")

        def neuron(weights, bias):
            terms = jax.vmap(
                lambda weight, entry: (
                    self.product(weight, entry)
                    if self.side == "left"
                    else self.product(entry, weight)
                )
            )(weights, value)
            return jnp.sum(terms, axis=0) + bias

        return jax.vmap(neuron)(self.weights, self.bias)


class AnalyticityEvidence(StrictModule):
    residual: Array
    finite: Array
    valid: Array
    operator_kind: str = eqx.field(static=True)
    side: str = eqx.field(static=True)

    def __init__(
        self,
        residual: ArrayLike,
        finite: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        operator_kind: str,
        side: str,
    ):
        self.residual = jnp.asarray(residual)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.operator_kind = str(operator_kind)
        self.side = str(side)


class AlgebraAnalyticNetwork(StrictModule):
    """Operator-specific network with fixed algebra, side, and bracket semantics."""

    layers: tuple[AlgebraAnalyticLayer, ...]
    activation: Callable[[Array], Array]
    operator: AnalyticityOperator
    bracketing: BracketingPlan
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        layers: Sequence[AlgebraAnalyticLayer],
        activation: Callable[[Array], Array],
        operator: AnalyticityOperator,
        bracketing: BracketingPlan,
        /,
        *,
        network_id: str,
    ):
        layers_ = tuple(layers)
        if (
            not layers_
            or any(not isinstance(value, AlgebraAnalyticLayer) for value in layers_)
            or not callable(activation)
        ):
            raise TypeError(
                "AlgebraAnalyticNetwork requires layers and a callable operator-compatible activation."
            )
        plan_ids = {value.product.plan_id for value in layers_}
        sides = {value.side for value in layers_}
        if len(plan_ids) != 1 or len(sides) != 1:
            raise ValueError(
                "All algebra analytic layers must share one product plan and side."
            )
        if (
            not isinstance(operator, AnalyticityOperator)
            or not isinstance(bracketing, BracketingPlan)
            or not network_id
        ):
            raise ValueError("Network operator/bracketing/id are invalid.")
        self.layers = layers_
        self.activation = activation
        self.operator = operator
        self.bracketing = bracketing
        self.network_id = str(network_id)

    @property
    def side(self) -> str:
        return self.layers[0].side

    def __call__(self, inputs: ArrayLike, /) -> Array:
        value = jnp.asarray(inputs)
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = self.activation(value)
        return value

    def analyticity_evidence(self, coordinates: ArrayLike, /) -> AnalyticityEvidence:
        residual = self.operator.residual(self, coordinates)
        finite = jnp.isfinite(residual)
        return AnalyticityEvidence(
            residual,
            finite,
            finite & (residual <= self.operator.tolerance),
            operator_kind=self.operator.kind,
            side=self.side,
        )


__all__ = [
    "AlgebraAnalyticLayer",
    "AlgebraAnalyticNetwork",
    "AnalyticityEvidence",
    "AnalyticityKind",
    "AnalyticityOperator",
]

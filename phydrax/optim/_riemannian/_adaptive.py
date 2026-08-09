#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import StrictModule
from ._first_order import (
    AbstractRiemannianOptimizer,
    LearningRate,
    RiemannianStepMetrics,
)
from ._parameter_geometry import ParameterGeometry


class RiemannianAdamState(StrictModule):
    """Transported first moment and intrinsic scalar factor moments."""

    step: Array
    first_moment: PyTree[Array]
    second_moment: PyTree[Array]
    maximum_second_moment: PyTree[Array]
    metrics: RiemannianStepMetrics

    def __init__(
        self,
        step: Array,
        first_moment: PyTree[Array],
        second_moment: PyTree[Array],
        maximum_second_moment: PyTree[Array],
        metrics: RiemannianStepMetrics,
        /,
    ):
        self.step = step
        self.first_moment = first_moment
        self.second_moment = second_moment
        self.maximum_second_moment = maximum_second_moment
        self.metrics = metrics


def _factor_bounds(factors: PyTree[Any], /) -> tuple[Array, Array]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(factors))
    minimum = jnp.asarray(jnp.inf, dtype=leaves[0].dtype)
    maximum = jnp.asarray(0.0, dtype=leaves[0].dtype)
    for leaf in leaves:
        minimum = jnp.minimum(minimum, jnp.min(leaf, initial=jnp.inf))
        maximum = jnp.maximum(maximum, jnp.max(leaf, initial=0.0))
    return minimum, maximum


class RiemannianAdam(AbstractRiemannianOptimizer):
    """Adam with transported tangent momentum and intrinsic scalar factor moments."""

    parameter_geometry: ParameterGeometry
    learning_rate: LearningRate = eqx.field(static=True)
    max_gradient_norm: float | None = eqx.field(static=True)
    first_moment_decay: float = eqx.field(static=True)
    second_moment_decay: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    amsgrad: bool = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterGeometry,
        /,
        *,
        learning_rate: LearningRate = 1e-3,
        first_moment_decay: float = 0.9,
        second_moment_decay: float = 0.999,
        epsilon: float = 1e-8,
        amsgrad: bool = False,
        max_gradient_norm: float | None = None,
    ):
        if not isinstance(parameter_geometry, ParameterGeometry):
            raise TypeError("parameter_geometry must be a ParameterGeometry.")
        if isinstance(learning_rate, (int, float)):
            scalar = float(learning_rate)
            if not isfinite(scalar) or scalar <= 0.0:
                raise ValueError("learning_rate must be finite and positive.")
            resolved_learning_rate: LearningRate = scalar
        elif callable(learning_rate):
            resolved_learning_rate = learning_rate
        else:
            raise TypeError("learning_rate must be a positive scalar or callable.")
        if max_gradient_norm is None:
            clipping = None
        else:
            clipping = float(max_gradient_norm)
            if not isfinite(clipping) or clipping <= 0.0:
                raise ValueError("max_gradient_norm must be finite and positive.")
        first_decay = float(first_moment_decay)
        second_decay = float(second_moment_decay)
        epsilon_value = float(epsilon)
        if not isfinite(first_decay) or not 0.0 <= first_decay < 1.0:
            raise ValueError(
                "first_moment_decay must be finite and satisfy 0 <= decay < 1."
            )
        if not isfinite(second_decay) or not 0.0 <= second_decay < 1.0:
            raise ValueError(
                "second_moment_decay must be finite and satisfy 0 <= decay < 1."
            )
        if not isfinite(epsilon_value) or epsilon_value <= 0.0:
            raise ValueError("epsilon must be finite and positive.")
        if not isinstance(amsgrad, bool):
            raise TypeError("amsgrad must be a bool.")
        self.parameter_geometry = parameter_geometry
        self.learning_rate = resolved_learning_rate
        self.max_gradient_norm = clipping
        self.first_moment_decay = first_decay
        self.second_moment_decay = second_decay
        self.epsilon = epsilon_value
        self.amsgrad = amsgrad
        self.optimizer_id = "riemannian-adam"

    def _resolved_learning_rate(self, step: Array, /) -> Array:
        if isinstance(self.learning_rate, (int, float)):
            value = self.learning_rate
        else:
            value = self.learning_rate(step)
        rate = jnp.asarray(value)
        if rate.shape != ():
            raise ValueError("Riemannian optimizer learning rate must be scalar.")
        return eqx.error_if(
            rate,
            (~jnp.isfinite(rate)) | (rate < 0.0),
            "Riemannian optimizer learning rate must be finite and nonnegative.",
        )

    def _gradient_and_scale(
        self,
        gradients: PyTree[Any],
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], Array, Array]:
        rgradient = self.parameter_geometry.egrad_to_rgrad(parameters, gradients)
        gradient_norm = self.parameter_geometry.norm(parameters, rgradient)
        gradient_norm = eqx.error_if(
            gradient_norm,
            ~jnp.isfinite(gradient_norm),
            "Riemannian gradient norm is not finite.",
        )
        if self.max_gradient_norm is None:
            clipping_scale = jnp.asarray(1.0, dtype=gradient_norm.dtype)
        else:
            denominator = jnp.maximum(
                gradient_norm,
                jnp.finfo(gradient_norm.dtype).tiny,
            )
            clipping_scale = jnp.minimum(
                1.0,
                jnp.asarray(self.max_gradient_norm, dtype=gradient_norm.dtype)
                / denominator,
            )
        clipped = jax.tree.map(lambda leaf: clipping_scale * leaf, rgradient)
        return clipped, gradient_norm, clipping_scale

    def init(self, parameters: PyTree[Any], /) -> RiemannianAdamState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError("Initial parameters are outside ParameterGeometry.")
        zero = jnp.asarray(0.0)
        first = jax.tree.map(jnp.zeros_like, parameters)
        second = self.parameter_geometry._factor_moment_zeros(parameters)
        maximum = jax.tree.map(jnp.zeros_like, second)
        return RiemannianAdamState(
            jnp.asarray(0, dtype=jnp.int32),
            first,
            second,
            maximum,
            RiemannianStepMetrics(
                zero,
                zero,
                jnp.asarray(1.0),
                zero,
                zero,
            ),
        )

    def update(
        self,
        gradients: PyTree[Any],
        state: RiemannianAdamState,
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], RiemannianAdamState]:
        if not isinstance(state, RiemannianAdamState):
            raise TypeError("RiemannianAdam requires RiemannianAdamState.")
        gradient, gradient_norm, clipping_scale = self._gradient_and_scale(
            gradients,
            parameters,
        )
        first_decay = jnp.asarray(
            self.first_moment_decay,
            dtype=gradient_norm.dtype,
        )
        second_decay = jnp.asarray(
            self.second_moment_decay,
            dtype=gradient_norm.dtype,
        )
        first_moment = jax.tree.map(
            lambda previous, current: (
                first_decay * previous + (1.0 - first_decay) * current
            ),
            state.first_moment,
            gradient,
        )
        squared_norms = self.parameter_geometry._factor_squared_norms(
            parameters,
            gradient,
        )
        second_moment = jax.tree.map(
            lambda previous, current: (
                second_decay * previous + (1.0 - second_decay) * current
            ),
            state.second_moment,
            squared_norms,
        )
        maximum_second_moment = jax.tree.map(
            jnp.maximum,
            state.maximum_second_moment,
            second_moment,
        )
        effective_second_moment = (
            maximum_second_moment if self.amsgrad else second_moment
        )
        step_number = state.step + jnp.asarray(1, dtype=state.step.dtype)
        step_value = step_number.astype(gradient_norm.dtype)
        first_bias = 1.0 - first_decay**step_value
        second_bias = 1.0 - second_decay**step_value
        corrected_first = jax.tree.map(
            lambda moment: moment / first_bias,
            first_moment,
        )
        corrected_second = jax.tree.map(
            lambda moment: moment / second_bias,
            effective_second_moment,
        )
        epsilon = jnp.asarray(self.epsilon, dtype=gradient_norm.dtype)
        denominators = jax.tree.map(
            lambda moment: jnp.sqrt(jnp.maximum(moment, 0.0)) + epsilon,
            corrected_second,
        )
        inverse_denominators = jax.tree.map(
            lambda denominator: 1.0 / denominator,
            denominators,
        )
        direction = self.parameter_geometry._scale_tangent_factors(
            corrected_first,
            inverse_denominators,
        )
        rate = self._resolved_learning_rate(state.step)
        tangent_step = jax.tree.map(lambda leaf: -rate * leaf, direction)
        tangent_step_norm = self.parameter_geometry.norm(parameters, tangent_step)
        tangent_residual = self.parameter_geometry.maximum_tangent_residual(
            parameters,
            tangent_step,
        )
        destination = self.parameter_geometry.retract(parameters, tangent_step)
        transported_first = self.parameter_geometry.transport(
            parameters,
            tangent_step,
            destination,
            first_moment,
        )
        momentum_norm = self.parameter_geometry.norm(parameters, first_moment)
        transported_tangent_residual = (
            self.parameter_geometry.maximum_tangent_residual(
                destination,
                transported_first,
            )
        )
        transport_metric_distortion = (
            self.parameter_geometry.maximum_transport_metric_distortion(
                parameters,
                destination,
                first_moment,
                transported_first,
            )
        )
        denominator_minimum, denominator_maximum = _factor_bounds(denominators)
        metrics = RiemannianStepMetrics(
            rate,
            gradient_norm,
            clipping_scale,
            tangent_step_norm,
            momentum_norm,
            constraint_residual=self.parameter_geometry.maximum_constraint_residual(
                destination
            ),
            tangent_residual=tangent_residual,
            transported_tangent_residual=transported_tangent_residual,
            transport_metric_distortion=transport_metric_distortion,
            adaptive_denominator_minimum=denominator_minimum,
            adaptive_denominator_maximum=denominator_maximum,
        )
        return destination, RiemannianAdamState(
            step_number,
            transported_first,
            second_moment,
            maximum_second_moment,
            metrics,
        )

    def step_metrics(self, state: RiemannianAdamState, /) -> RiemannianStepMetrics:
        if not isinstance(state, RiemannianAdamState):
            raise TypeError("RiemannianAdam requires RiemannianAdamState.")
        return state.metrics


def riemannian_adam(
    parameter_geometry: ParameterGeometry,
    /,
    *,
    learning_rate: LearningRate = 1e-3,
    first_moment_decay: float = 0.9,
    second_moment_decay: float = 0.999,
    epsilon: float = 1e-8,
    amsgrad: bool = False,
    max_gradient_norm: float | None = None,
) -> RiemannianAdam:
    """Construct invariant factorwise Riemannian Adam or AMSGrad."""
    return RiemannianAdam(
        parameter_geometry,
        learning_rate=learning_rate,
        first_moment_decay=first_moment_decay,
        second_moment_decay=second_moment_decay,
        epsilon=epsilon,
        amsgrad=amsgrad,
        max_gradient_norm=max_gradient_norm,
    )


__all__ = [
    "RiemannianAdam",
    "RiemannianAdamState",
    "riemannian_adam",
]

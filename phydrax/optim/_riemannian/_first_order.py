#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._strict import AbstractAttribute, StrictModule
from ._parameter_geometry import ParameterGeometry


LearningRate = float | Callable[[Array], Array]


class RiemannianStepMetrics(StrictModule):
    """Scalar diagnostics produced by one geometric optimizer update."""

    learning_rate: Array
    gradient_norm: Array
    clipping_scale: Array
    tangent_step_norm: Array
    momentum_norm: Array
    constraint_residual: Array
    tangent_residual: Array
    transported_tangent_residual: Array
    transport_metric_distortion: Array
    line_search_evaluations: Array
    line_search_accepted: Array
    line_search_reduction: Array
    conjugacy_beta: Array
    history_pair_count: Array

    def __init__(
        self,
        learning_rate: Array,
        gradient_norm: Array,
        clipping_scale: Array,
        tangent_step_norm: Array,
        momentum_norm: Array,
        *,
        constraint_residual: Array | None = None,
        tangent_residual: Array | None = None,
        transported_tangent_residual: Array | None = None,
        transport_metric_distortion: Array | None = None,
        line_search_evaluations: Array | None = None,
        line_search_accepted: Array | None = None,
        line_search_reduction: Array | None = None,
        conjugacy_beta: Array | None = None,
        history_pair_count: Array | None = None,
    ):
        zero = jnp.zeros_like(jnp.asarray(gradient_norm))
        self.learning_rate = learning_rate
        self.gradient_norm = gradient_norm
        self.clipping_scale = clipping_scale
        self.tangent_step_norm = tangent_step_norm
        self.momentum_norm = momentum_norm
        self.constraint_residual = (
            zero if constraint_residual is None else constraint_residual
        )
        self.tangent_residual = zero if tangent_residual is None else tangent_residual
        self.transported_tangent_residual = (
            zero if transported_tangent_residual is None else transported_tangent_residual
        )
        self.transport_metric_distortion = (
            zero if transport_metric_distortion is None else transport_metric_distortion
        )
        self.line_search_evaluations = (
            jnp.asarray(0, dtype=jnp.int32)
            if line_search_evaluations is None
            else jnp.asarray(line_search_evaluations, dtype=jnp.int32)
        )
        self.line_search_accepted = (
            jnp.asarray(False)
            if line_search_accepted is None
            else jnp.asarray(line_search_accepted, dtype=bool)
        )
        self.line_search_reduction = (
            zero if line_search_reduction is None else line_search_reduction
        )
        self.conjugacy_beta = zero if conjugacy_beta is None else conjugacy_beta
        self.history_pair_count = (
            jnp.asarray(0, dtype=jnp.int32)
            if history_pair_count is None
            else jnp.asarray(history_pair_count, dtype=jnp.int32)
        )


class RiemannianSGDState(StrictModule):
    step: Array
    metrics: RiemannianStepMetrics

    def __init__(
        self,
        step: Array,
        metrics: RiemannianStepMetrics,
    ):
        self.step = step
        self.metrics = metrics


class RiemannianMomentumState(StrictModule):
    step: Array
    momentum: PyTree[Array]
    metrics: RiemannianStepMetrics

    def __init__(
        self,
        step: Array,
        momentum: PyTree[Array],
        metrics: RiemannianStepMetrics,
    ):
        self.step = step
        self.momentum = momentum
        self.metrics = metrics


class AbstractRiemannianOptimizer(StrictModule):
    """Nominal optimizer contract whose updates return manifold-valued parameters."""

    optimizer_id: AbstractAttribute[str]
    parameter_geometry: AbstractAttribute[ParameterGeometry]

    @abstractmethod
    def init(self, parameters: PyTree[Any], /) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        gradients: PyTree[Any],
        state: Any,
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], Any]:
        raise NotImplementedError

    @abstractmethod
    def step_metrics(self, state: Any, /) -> RiemannianStepMetrics:
        raise NotImplementedError


class RiemannianSGD(AbstractRiemannianOptimizer):
    """Riemannian gradient descent over a bound product parameter geometry."""

    parameter_geometry: ParameterGeometry
    learning_rate: LearningRate = eqx.field(static=True)
    max_gradient_norm: float | None = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterGeometry,
        /,
        *,
        learning_rate: LearningRate = 1e-2,
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
        self.parameter_geometry = parameter_geometry
        self.learning_rate = resolved_learning_rate
        self.max_gradient_norm = clipping
        self.optimizer_id = "riemannian-sgd"

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

    def init(self, parameters: PyTree[Any], /) -> RiemannianSGDState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError("Initial parameters are outside ParameterGeometry.")
        zero = jnp.asarray(0.0)
        return RiemannianSGDState(
            jnp.asarray(0, dtype=jnp.int32),
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
        state: RiemannianSGDState,
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], RiemannianSGDState]:
        if not isinstance(state, RiemannianSGDState):
            raise TypeError("RiemannianSGD requires RiemannianSGDState.")
        gradient, gradient_norm, clipping_scale = self._gradient_and_scale(
            gradients, parameters
        )
        rate = self._resolved_learning_rate(state.step)
        tangent_step = jax.tree.map(lambda leaf: -rate * leaf, gradient)
        tangent_step_norm = self.parameter_geometry.norm(parameters, tangent_step)
        tangent_residual = self.parameter_geometry.maximum_tangent_residual(
            parameters, tangent_step
        )
        destination = self.parameter_geometry.retract(parameters, tangent_step)
        constraint_residual = self.parameter_geometry.maximum_constraint_residual(
            destination
        )
        metrics = RiemannianStepMetrics(
            rate,
            gradient_norm,
            clipping_scale,
            tangent_step_norm,
            jnp.asarray(0.0, dtype=gradient_norm.dtype),
            constraint_residual=constraint_residual,
            tangent_residual=tangent_residual,
        )
        return destination, RiemannianSGDState(
            state.step + jnp.asarray(1, dtype=state.step.dtype),
            metrics,
        )

    def step_metrics(self, state: RiemannianSGDState, /) -> RiemannianStepMetrics:
        if not isinstance(state, RiemannianSGDState):
            raise TypeError("RiemannianSGD requires RiemannianSGDState.")
        return state.metrics


class RiemannianMomentum(AbstractRiemannianOptimizer):
    """Transported heavy-ball momentum on a product parameter manifold."""

    parameter_geometry: ParameterGeometry
    learning_rate: LearningRate = eqx.field(static=True)
    momentum_coefficient: float = eqx.field(static=True)
    max_gradient_norm: float | None = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterGeometry,
        /,
        *,
        learning_rate: LearningRate = 1e-2,
        momentum: float = 0.9,
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
        coefficient = float(momentum)
        if not isfinite(coefficient) or not 0.0 <= coefficient < 1.0:
            raise ValueError("momentum must be finite and satisfy 0 <= momentum < 1.")
        if max_gradient_norm is None:
            clipping = None
        else:
            clipping = float(max_gradient_norm)
            if not isfinite(clipping) or clipping <= 0.0:
                raise ValueError("max_gradient_norm must be finite and positive.")
        self.parameter_geometry = parameter_geometry
        self.learning_rate = resolved_learning_rate
        self.momentum_coefficient = coefficient
        self.max_gradient_norm = clipping
        self.optimizer_id = "riemannian-momentum"

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

    def init(self, parameters: PyTree[Any], /) -> RiemannianMomentumState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError("Initial parameters are outside ParameterGeometry.")
        zero = jnp.asarray(0.0)
        return RiemannianMomentumState(
            jnp.asarray(0, dtype=jnp.int32),
            jax.tree.map(jnp.zeros_like, parameters),
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
        state: RiemannianMomentumState,
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], RiemannianMomentumState]:
        if not isinstance(state, RiemannianMomentumState):
            raise TypeError("RiemannianMomentum requires RiemannianMomentumState.")
        gradient, gradient_norm, clipping_scale = self._gradient_and_scale(
            gradients, parameters
        )
        momentum = jax.tree.map(
            lambda previous, current: self.momentum_coefficient * previous + current,
            state.momentum,
            gradient,
        )
        momentum_norm = self.parameter_geometry.norm(parameters, momentum)
        rate = self._resolved_learning_rate(state.step)
        tangent_step = jax.tree.map(lambda leaf: -rate * leaf, momentum)
        tangent_step_norm = self.parameter_geometry.norm(parameters, tangent_step)
        tangent_residual = self.parameter_geometry.maximum_tangent_residual(
            parameters, tangent_step
        )
        destination = self.parameter_geometry.retract(parameters, tangent_step)
        transported = self.parameter_geometry.transport(
            parameters,
            tangent_step,
            destination,
            momentum,
        )
        constraint_residual = self.parameter_geometry.maximum_constraint_residual(
            destination
        )
        transported_tangent_residual = self.parameter_geometry.maximum_tangent_residual(
            destination, transported
        )
        transport_metric_distortion = (
            self.parameter_geometry.maximum_transport_metric_distortion(
                parameters,
                destination,
                momentum,
                transported,
            )
        )
        metrics = RiemannianStepMetrics(
            rate,
            gradient_norm,
            clipping_scale,
            tangent_step_norm,
            momentum_norm,
            constraint_residual=constraint_residual,
            tangent_residual=tangent_residual,
            transported_tangent_residual=transported_tangent_residual,
            transport_metric_distortion=transport_metric_distortion,
        )
        return destination, RiemannianMomentumState(
            state.step + jnp.asarray(1, dtype=state.step.dtype),
            transported,
            metrics,
        )

    def step_metrics(
        self,
        state: RiemannianMomentumState,
        /,
    ) -> RiemannianStepMetrics:
        if not isinstance(state, RiemannianMomentumState):
            raise TypeError("RiemannianMomentum requires RiemannianMomentumState.")
        return state.metrics


def riemannian_sgd(
    parameter_geometry: ParameterGeometry,
    /,
    *,
    learning_rate: LearningRate = 1e-2,
    max_gradient_norm: float | None = None,
) -> RiemannianSGD:
    """Construct metric-correct fixed-step Riemannian gradient descent."""
    return RiemannianSGD(
        parameter_geometry,
        learning_rate=learning_rate,
        max_gradient_norm=max_gradient_norm,
    )


def riemannian_momentum(
    parameter_geometry: ParameterGeometry,
    /,
    *,
    learning_rate: LearningRate = 1e-2,
    momentum: float = 0.9,
    max_gradient_norm: float | None = None,
) -> RiemannianMomentum:
    """Construct transported heavy-ball Riemannian momentum."""
    return RiemannianMomentum(
        parameter_geometry,
        learning_rate=learning_rate,
        momentum=momentum,
        max_gradient_norm=max_gradient_norm,
    )


__all__ = [
    "AbstractRiemannianOptimizer",
    "RiemannianMomentum",
    "RiemannianMomentumState",
    "RiemannianSGD",
    "RiemannianSGDState",
    "RiemannianStepMetrics",
    "riemannian_momentum",
    "riemannian_sgd",
]

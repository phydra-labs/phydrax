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

from .._strict import AbstractAttribute, StrictModule
from ._mirror_geometry import ParameterMirrorGeometry


MirrorLearningRate = float | Callable[[Array], Array]


class MirrorStepMetrics(StrictModule):
    """Coordinate diagnostics produced by one mirror-descent update."""

    learning_rate: Array
    coordinate_gradient_norm: Array
    dual_displacement_norm: Array
    bregman_step: Array
    constraint_residual: Array

    def __init__(
        self,
        learning_rate: Array,
        coordinate_gradient_norm: Array,
        dual_displacement_norm: Array,
        bregman_step: Array,
        constraint_residual: Array,
    ):
        self.learning_rate = jnp.asarray(learning_rate)
        self.coordinate_gradient_norm = jnp.asarray(coordinate_gradient_norm)
        self.dual_displacement_norm = jnp.asarray(dual_displacement_norm)
        self.bregman_step = jnp.asarray(bregman_step)
        self.constraint_residual = jnp.asarray(constraint_residual)


class MirrorDescentState(StrictModule):
    step: Array
    metrics: MirrorStepMetrics

    def __init__(self, step: Array, metrics: MirrorStepMetrics):
        if not isinstance(metrics, MirrorStepMetrics):
            raise TypeError("metrics must be MirrorStepMetrics.")
        self.step = jnp.asarray(step, dtype=jnp.int32)
        self.metrics = metrics


class AbstractMirrorOptimizer(StrictModule):
    """Optimizer contract whose updates translate declared dual coordinates."""

    optimizer_id: AbstractAttribute[str]
    parameter_geometry: AbstractAttribute[ParameterMirrorGeometry]

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
    def step_metrics(self, state: Any, /) -> MirrorStepMetrics:
        raise NotImplementedError


class MirrorDescent(AbstractMirrorOptimizer):
    """Fixed-step mirror descent over a separable parameter geometry."""

    parameter_geometry: ParameterMirrorGeometry
    learning_rate: MirrorLearningRate = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterMirrorGeometry,
        /,
        *,
        learning_rate: MirrorLearningRate = 1e-2,
    ):
        if not isinstance(parameter_geometry, ParameterMirrorGeometry):
            raise TypeError("parameter_geometry must be a ParameterMirrorGeometry.")
        if isinstance(learning_rate, (int, float)):
            scalar = float(learning_rate)
            if not isfinite(scalar) or scalar <= 0.0:
                raise ValueError("learning_rate must be finite and positive.")
            resolved_learning_rate: MirrorLearningRate = scalar
        elif callable(learning_rate):
            resolved_learning_rate = learning_rate
        else:
            raise TypeError("learning_rate must be a positive scalar or callable.")
        self.parameter_geometry = parameter_geometry
        self.learning_rate = resolved_learning_rate
        self.optimizer_id = "mirror-descent"

    def _resolved_learning_rate(self, step: Array, /) -> Array:
        value = (
            self.learning_rate
            if isinstance(self.learning_rate, (int, float))
            else self.learning_rate(step)
        )
        rate = jnp.asarray(value)
        if rate.shape != ():
            raise ValueError("Mirror-descent learning rate must be scalar.")
        if jnp.issubdtype(rate.dtype, jnp.complexfloating):
            raise TypeError("Mirror-descent learning rate must be real.")
        if not jnp.issubdtype(rate.dtype, jnp.inexact):
            rate = rate.astype(jnp.result_type(rate, 0.0))
        return eqx.error_if(
            rate,
            (~jnp.isfinite(rate)) | (rate < 0.0),
            "Mirror-descent learning rate must be finite and nonnegative.",
        )

    def init(self, parameters: PyTree[Any], /) -> MirrorDescentState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError("Initial parameters are outside ParameterMirrorGeometry.")
        zero = jnp.asarray(0.0)
        return MirrorDescentState(
            jnp.asarray(0, dtype=jnp.int32),
            MirrorStepMetrics(zero, zero, zero, zero, zero),
        )

    def update(
        self,
        gradients: PyTree[Any],
        state: MirrorDescentState,
        parameters: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], MirrorDescentState]:
        if not isinstance(state, MirrorDescentState):
            raise TypeError("MirrorDescent requires MirrorDescentState.")
        self.parameter_geometry.validate(parameters)
        coordinate_gradient_norm = self.parameter_geometry.coordinate_gradient_norm(
            gradients
        )
        coordinate_gradient_norm = eqx.error_if(
            coordinate_gradient_norm,
            ~jnp.isfinite(coordinate_gradient_norm),
            "Mirror coordinate-gradient norm is not finite.",
        )
        rate = self._resolved_learning_rate(state.step)
        dual_displacements = jax.tree.map(
            lambda gradient: (
                -jnp.asarray(
                    rate,
                    dtype=jnp.asarray(gradient).real.dtype,
                )
                * (
                    jnp.conj(gradient)
                    if jnp.issubdtype(
                        jnp.asarray(gradient).dtype,
                        jnp.complexfloating,
                    )
                    else gradient
                )
            ),
            gradients,
        )
        dual_displacement_norm = self.parameter_geometry.dual_displacement_norm(
            dual_displacements
        )
        destination = self.parameter_geometry.dual_translate(
            parameters,
            dual_displacements,
        )
        constraint_residual = self.parameter_geometry.maximum_constraint_residual(
            destination
        )
        constraint_residual = eqx.error_if(
            constraint_residual,
            ~jnp.isfinite(constraint_residual),
            "Mirror-descent destination is outside ParameterMirrorGeometry.",
        )
        bregman_step = self.parameter_geometry.bregman_step(parameters, destination)
        metrics = MirrorStepMetrics(
            rate,
            coordinate_gradient_norm,
            dual_displacement_norm,
            bregman_step,
            constraint_residual,
        )
        return destination, MirrorDescentState(state.step + 1, metrics)

    def step_metrics(self, state: MirrorDescentState, /) -> MirrorStepMetrics:
        if not isinstance(state, MirrorDescentState):
            raise TypeError("MirrorDescent requires MirrorDescentState.")
        return state.metrics


def mirror_descent(
    parameter_geometry: ParameterMirrorGeometry,
    /,
    *,
    learning_rate: MirrorLearningRate = 1e-2,
) -> MirrorDescent:
    """Construct fixed-step mirror descent over declared Legendre leaves."""
    return MirrorDescent(parameter_geometry, learning_rate=learning_rate)


__all__ = [
    "AbstractMirrorOptimizer",
    "MirrorDescent",
    "MirrorDescentState",
    "MirrorStepMetrics",
    "mirror_descent",
]

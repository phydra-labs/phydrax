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

from ._first_order import (
    AbstractRiemannianOptimizer,
    RiemannianStepMetrics,
)
from ._line_search import armijo_backtracking, ArmijoLineSearch
from ._parameter_geometry import ParameterGeometry


class AbstractRiemannianLineSearchOptimizer(AbstractRiemannianOptimizer):
    """Riemannian optimizer whose update consumes one frozen objective closure."""

    @abstractmethod
    def update(
        self,
        gradients: PyTree[Any],
        state: Any,
        parameters: PyTree[Any],
        /,
        *,
        value: Array | None = None,
        value_fn: Callable[[PyTree[Any]], Array] | None = None,
    ):
        raise NotImplementedError


class RiemannianConjugateGradientState(eqx.Module):
    step: Array
    previous_gradient: PyTree[Array]
    previous_direction: PyTree[Array]
    beta: Array
    line_search_evaluations: Array
    line_search_accepted: Array
    restarted: Array
    metrics: RiemannianStepMetrics


class RiemannianConjugateGradient(AbstractRiemannianLineSearchOptimizer):
    """Polak–Ribière+ conjugate gradient with transported state and Armijo search."""

    parameter_geometry: ParameterGeometry
    line_search: ArmijoLineSearch
    descent_tolerance: float = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterGeometry,
        /,
        *,
        line_search: ArmijoLineSearch | None = None,
        descent_tolerance: float = 1e-12,
    ):
        if not isinstance(parameter_geometry, ParameterGeometry):
            raise TypeError("parameter_geometry must be a ParameterGeometry.")
        tolerance = float(descent_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("descent_tolerance must be finite and non-negative.")
        self.parameter_geometry = parameter_geometry
        self.line_search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(self.line_search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch.")
        self.descent_tolerance = tolerance
        self.optimizer_id = "riemannian-conjugate-gradient"

    def init(self, parameters: PyTree[Any], /) -> RiemannianConjugateGradientState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError("Initial parameters are outside ParameterGeometry.")
        zero_tree = jax.tree.map(jnp.zeros_like, parameters)
        zero = jnp.asarray(0.0)
        return RiemannianConjugateGradientState(
            step=jnp.asarray(0, dtype=jnp.int32),
            previous_gradient=zero_tree,
            previous_direction=zero_tree,
            beta=zero,
            line_search_evaluations=jnp.asarray(0, dtype=jnp.int32),
            line_search_accepted=jnp.asarray(False),
            restarted=jnp.asarray(True),
            metrics=RiemannianStepMetrics(zero, zero, jnp.asarray(1.0), zero, zero),
        )

    def update(
        self,
        gradients: PyTree[Any],
        state: RiemannianConjugateGradientState,
        parameters: PyTree[Any],
        /,
        *,
        value: Array | None = None,
        value_fn: Callable[[PyTree[Any]], Array] | None = None,
    ) -> tuple[PyTree[Array], RiemannianConjugateGradientState]:
        if not isinstance(state, RiemannianConjugateGradientState):
            raise TypeError("RiemannianConjugateGradient requires its matching state.")
        if value is None or value_fn is None:
            raise TypeError(
                "RiemannianConjugateGradient.update requires value and value_fn."
            )
        gradient = self.parameter_geometry.egrad_to_rgrad(parameters, gradients)
        gradient_norm = self.parameter_geometry.norm(parameters, gradient)
        gradient_norm = eqx.error_if(
            gradient_norm,
            ~jnp.isfinite(gradient_norm),
            "Riemannian conjugate-gradient norm is not finite.",
        )
        difference = jax.tree.map(
            lambda current, previous: current - previous,
            gradient,
            state.previous_gradient,
        )
        numerator = self.parameter_geometry.inner(parameters, gradient, difference)
        denominator = self.parameter_geometry.inner(
            parameters, state.previous_gradient, state.previous_gradient
        )
        tiny = jnp.finfo(gradient_norm.dtype).tiny
        beta = jnp.maximum(0.0, numerator / jnp.maximum(denominator, tiny))
        beta = jnp.where(state.step > 0, beta, 0.0)
        candidate_direction = jax.tree.map(
            lambda current, previous: -current + beta * previous,
            gradient,
            state.previous_direction,
        )
        directional = self.parameter_geometry.inner(
            parameters, gradient, candidate_direction
        )
        descent_bound = (
            -jnp.asarray(self.descent_tolerance, dtype=gradient_norm.dtype)
            * gradient_norm**2
        )
        restarted = (
            (state.step == 0)
            | (~jnp.isfinite(directional))
            | (directional >= descent_bound)
        )
        direction = jax.tree.map(
            lambda candidate, current: jnp.where(restarted, -current, candidate),
            candidate_direction,
            gradient,
        )
        beta = jnp.where(restarted, 0.0, beta)
        result = armijo_backtracking(
            value_fn,
            self.parameter_geometry,
            parameters,
            jnp.asarray(value),
            gradient,
            direction,
            policy=self.line_search,
        )
        tangent_step = jax.tree.map(lambda leaf: result.rate * leaf, direction)
        tangent_step_norm = self.parameter_geometry.norm(parameters, tangent_step)
        tangent_residual = self.parameter_geometry.maximum_tangent_residual(
            parameters, tangent_step
        )
        transported_gradient = self.parameter_geometry.transport(
            parameters,
            tangent_step,
            result.parameters,
            gradient,
        )
        transported_direction = self.parameter_geometry.transport(
            parameters,
            tangent_step,
            result.parameters,
            direction,
        )
        transported_residual = self.parameter_geometry.maximum_tangent_residual(
            result.parameters, transported_direction
        )
        transport_distortion = (
            self.parameter_geometry.maximum_transport_metric_distortion(
                parameters,
                result.parameters,
                direction,
                transported_direction,
            )
        )
        metrics = RiemannianStepMetrics(
            result.rate,
            gradient_norm,
            jnp.asarray(1.0, dtype=gradient_norm.dtype),
            tangent_step_norm,
            jnp.asarray(0.0, dtype=gradient_norm.dtype),
            constraint_residual=self.parameter_geometry.maximum_constraint_residual(
                result.parameters
            ),
            tangent_residual=tangent_residual,
            transported_tangent_residual=transported_residual,
            transport_metric_distortion=transport_distortion,
            line_search_evaluations=result.evaluations,
            line_search_accepted=result.accepted,
            line_search_reduction=jnp.asarray(value) - result.value,
            conjugacy_beta=beta,
        )
        return result.parameters, RiemannianConjugateGradientState(
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
            previous_gradient=transported_gradient,
            previous_direction=transported_direction,
            beta=beta,
            line_search_evaluations=result.evaluations,
            line_search_accepted=result.accepted,
            restarted=restarted,
            metrics=metrics,
        )

    def step_metrics(
        self, state: RiemannianConjugateGradientState, /
    ) -> RiemannianStepMetrics:
        if not isinstance(state, RiemannianConjugateGradientState):
            raise TypeError("RiemannianConjugateGradient requires its matching state.")
        return state.metrics


class RiemannianLBFGSState(eqx.Module):
    step: Array
    s_history: PyTree[Array]
    y_history: PyTree[Array]
    rho: Array
    active: Array
    count: Array
    next_index: Array
    line_search_evaluations: Array
    line_search_accepted: Array
    pair_accepted: Array
    metrics: RiemannianStepMetrics


def _history_slot(history: PyTree[Array], index: Array, /) -> PyTree[Array]:
    return jax.tree.map(lambda leaf: leaf[index], history)


def _stack_history(slots: tuple[PyTree[Array], ...], /) -> PyTree[Array]:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *slots)


class RiemannianLBFGS(AbstractRiemannianLineSearchOptimizer):
    """Transported limited-memory Riemannian BFGS with Armijo globalization."""

    parameter_geometry: ParameterGeometry
    line_search: ArmijoLineSearch
    history_size: int = eqx.field(static=True)
    curvature_tolerance: float = eqx.field(static=True)
    descent_tolerance: float = eqx.field(static=True)
    optimizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_geometry: ParameterGeometry,
        /,
        *,
        history_size: int = 10,
        line_search: ArmijoLineSearch | None = None,
        curvature_tolerance: float = 1e-10,
        descent_tolerance: float = 1e-12,
    ):
        if not isinstance(parameter_geometry, ParameterGeometry):
            raise TypeError("parameter_geometry must be a ParameterGeometry.")
        size = int(history_size)
        curvature = float(curvature_tolerance)
        descent = float(descent_tolerance)
        if size <= 0:
            raise ValueError("history_size must be positive.")
        if not isfinite(curvature) or curvature < 0.0:
            raise ValueError("curvature_tolerance must be finite and non-negative.")
        if not isfinite(descent) or descent < 0.0:
            raise ValueError("descent_tolerance must be finite and non-negative.")
        self.parameter_geometry = parameter_geometry
        self.line_search = ArmijoLineSearch() if line_search is None else line_search
        if not isinstance(self.line_search, ArmijoLineSearch):
            raise TypeError("line_search must be an ArmijoLineSearch.")
        self.history_size = size
        self.curvature_tolerance = curvature
        self.descent_tolerance = descent
        self.optimizer_id = "riemannian-lbfgs"

    def init(self, parameters: PyTree[Any], /) -> RiemannianLBFGSState:
        self.parameter_geometry.validate(parameters)
        if not bool(self.parameter_geometry.contains(parameters)):
            raise ValueError("Initial parameters are outside ParameterGeometry.")
        history = jax.tree.map(
            lambda leaf: jnp.zeros((self.history_size,) + leaf.shape, dtype=leaf.dtype),
            parameters,
        )
        zero = jnp.asarray(0.0)
        return RiemannianLBFGSState(
            step=jnp.asarray(0, dtype=jnp.int32),
            s_history=history,
            y_history=history,
            rho=jnp.zeros((self.history_size,)),
            active=jnp.zeros((self.history_size,), dtype=bool),
            count=jnp.asarray(0, dtype=jnp.int32),
            next_index=jnp.asarray(0, dtype=jnp.int32),
            line_search_evaluations=jnp.asarray(0, dtype=jnp.int32),
            line_search_accepted=jnp.asarray(False),
            pair_accepted=jnp.asarray(False),
            metrics=RiemannianStepMetrics(zero, zero, jnp.asarray(1.0), zero, zero),
        )

    def _history_coefficients(
        self,
        parameters: PyTree[Any],
        s_history: PyTree[Array],
        y_history: PyTree[Array],
        active: Array,
        /,
    ) -> tuple[Array, Array]:
        curvatures = jnp.stack(
            tuple(
                self.parameter_geometry.inner(
                    parameters,
                    _history_slot(s_history, jnp.asarray(index, dtype=jnp.int32)),
                    _history_slot(y_history, jnp.asarray(index, dtype=jnp.int32)),
                )
                for index in range(self.history_size)
            )
        )
        scales = jnp.stack(
            tuple(
                self.parameter_geometry.norm(
                    parameters,
                    _history_slot(s_history, jnp.asarray(index, dtype=jnp.int32)),
                )
                * self.parameter_geometry.norm(
                    parameters,
                    _history_slot(y_history, jnp.asarray(index, dtype=jnp.int32)),
                )
                for index in range(self.history_size)
            )
        )
        thresholds = (
            jnp.asarray(self.curvature_tolerance, dtype=curvatures.dtype) * scales
        )
        valid = (
            active
            & jnp.isfinite(curvatures)
            & jnp.isfinite(scales)
            & (curvatures > thresholds)
        )
        safe_curvatures = jnp.where(valid, curvatures, 1.0)
        return jnp.where(valid, 1.0 / safe_curvatures, 0.0), valid

    def _direction(
        self,
        parameters: PyTree[Any],
        gradient: PyTree[Any],
        state: RiemannianLBFGSState,
        /,
    ) -> PyTree[Array]:
        q = gradient
        rho, active = self._history_coefficients(
            parameters,
            state.s_history,
            state.y_history,
            state.active,
        )
        alphas = jnp.zeros((self.history_size,), dtype=rho.dtype)
        for offset in range(self.history_size):
            index = (state.next_index - 1 - offset) % self.history_size
            s_value = _history_slot(state.s_history, index)
            y_value = _history_slot(state.y_history, index)
            alpha = rho[index] * self.parameter_geometry.inner(parameters, s_value, q)
            alpha = jnp.where(active[index], alpha, 0.0)
            alphas = alphas.at[index].set(alpha)
            q = jax.tree.map(
                lambda q_leaf, y_leaf: q_leaf - alpha * y_leaf,
                q,
                y_value,
            )

        newest = (state.next_index - 1) % self.history_size
        newest_s = _history_slot(state.s_history, newest)
        newest_y = _history_slot(state.y_history, newest)
        sy = self.parameter_geometry.inner(parameters, newest_s, newest_y)
        yy = self.parameter_geometry.inner(parameters, newest_y, newest_y)
        gamma = sy / jnp.maximum(yy, jnp.finfo(state.rho.dtype).tiny)
        gamma = jnp.where(active[newest] & jnp.isfinite(gamma), gamma, 1.0)
        result = jax.tree.map(lambda leaf: gamma * leaf, q)

        oldest = state.next_index
        for offset in range(self.history_size):
            index = (oldest + offset) % self.history_size
            s_value = _history_slot(state.s_history, index)
            y_value = _history_slot(state.y_history, index)
            beta = rho[index] * self.parameter_geometry.inner(parameters, y_value, result)
            active_pair = active[index]
            coefficient = jnp.where(active_pair, alphas[index] - beta, 0.0)
            result = jax.tree.map(
                lambda result_leaf, s_leaf: result_leaf + coefficient * s_leaf,
                result,
                s_value,
            )
        return jax.tree.map(lambda leaf: -leaf, result)

    def _transport_history(
        self,
        state: RiemannianLBFGSState,
        parameters: PyTree[Any],
        tangent_step: PyTree[Any],
        destination: PyTree[Any],
        /,
    ) -> tuple[PyTree[Array], PyTree[Array]]:
        s_slots = tuple(
            self.parameter_geometry.transport(
                parameters,
                tangent_step,
                destination,
                _history_slot(state.s_history, jnp.asarray(index, dtype=jnp.int32)),
            )
            for index in range(self.history_size)
        )
        y_slots = tuple(
            self.parameter_geometry.transport(
                parameters,
                tangent_step,
                destination,
                _history_slot(state.y_history, jnp.asarray(index, dtype=jnp.int32)),
            )
            for index in range(self.history_size)
        )
        return _stack_history(s_slots), _stack_history(y_slots)

    def update(
        self,
        gradients: PyTree[Any],
        state: RiemannianLBFGSState,
        parameters: PyTree[Any],
        /,
        *,
        value: Array | None = None,
        value_fn: Callable[[PyTree[Any]], Array] | None = None,
    ) -> tuple[PyTree[Array], RiemannianLBFGSState]:
        if not isinstance(state, RiemannianLBFGSState):
            raise TypeError("RiemannianLBFGS requires RiemannianLBFGSState.")
        if value is None or value_fn is None:
            raise TypeError("RiemannianLBFGS.update requires value and value_fn.")
        gradient = self.parameter_geometry.egrad_to_rgrad(parameters, gradients)
        gradient_norm = self.parameter_geometry.norm(parameters, gradient)
        gradient_norm = eqx.error_if(
            gradient_norm,
            ~jnp.isfinite(gradient_norm),
            "Riemannian L-BFGS gradient norm is not finite.",
        )
        quasi_newton_direction = self._direction(parameters, gradient, state)
        directional = self.parameter_geometry.inner(
            parameters, gradient, quasi_newton_direction
        )
        descent_bound = (
            -jnp.asarray(self.descent_tolerance, dtype=gradient_norm.dtype)
            * gradient_norm**2
        )
        restart = (~jnp.isfinite(directional)) | (directional >= descent_bound)
        direction = jax.tree.map(
            lambda proposed, current: jnp.where(restart, -current, proposed),
            quasi_newton_direction,
            gradient,
        )
        result = armijo_backtracking(
            value_fn,
            self.parameter_geometry,
            parameters,
            jnp.asarray(value),
            gradient,
            direction,
            policy=self.line_search,
        )
        tangent_step = jax.tree.map(lambda leaf: result.rate * leaf, direction)
        destination = result.parameters
        transported_s, transported_y = self._transport_history(
            state, parameters, tangent_step, destination
        )
        transported_rho, transported_active = self._history_coefficients(
            destination,
            transported_s,
            transported_y,
            state.active,
        )
        transported_gradient = self.parameter_geometry.transport(
            parameters, tangent_step, destination, gradient
        )
        destination_egrad = jax.grad(value_fn)(destination)
        destination_gradient = self.parameter_geometry.egrad_to_rgrad(
            destination, destination_egrad
        )
        displacement = self.parameter_geometry.transport(
            parameters, tangent_step, destination, tangent_step
        )
        gradient_difference = jax.tree.map(
            lambda current, previous: current - previous,
            destination_gradient,
            transported_gradient,
        )
        curvature = self.parameter_geometry.inner(
            destination, displacement, gradient_difference
        )
        pair_scale = self.parameter_geometry.norm(
            destination, displacement
        ) * self.parameter_geometry.norm(destination, gradient_difference)
        threshold = (
            jnp.asarray(self.curvature_tolerance, dtype=gradient_norm.dtype) * pair_scale
        )
        pair_accepted = (
            result.accepted
            & jnp.isfinite(curvature)
            & jnp.isfinite(pair_scale)
            & (curvature > threshold)
        )
        index = state.next_index
        old_s = _history_slot(transported_s, index)
        old_y = _history_slot(transported_y, index)
        inserted_s = jax.tree.map(
            lambda history, previous, new: history.at[index].set(
                jnp.where(pair_accepted, new, previous)
            ),
            transported_s,
            old_s,
            displacement,
        )
        inserted_y = jax.tree.map(
            lambda history, previous, new: history.at[index].set(
                jnp.where(pair_accepted, new, previous)
            ),
            transported_y,
            old_y,
            gradient_difference,
        )
        rho_value = 1.0 / jnp.maximum(curvature, jnp.finfo(gradient_norm.dtype).tiny)
        rho = transported_rho.at[index].set(
            jnp.where(pair_accepted, rho_value, transported_rho[index])
        )
        active = transported_active.at[index].set(
            jnp.where(pair_accepted, True, transported_active[index])
        )
        count = jnp.sum(active, dtype=jnp.int32)
        next_index = jnp.where(
            pair_accepted,
            (state.next_index + 1) % self.history_size,
            state.next_index,
        )
        tangent_step_norm = self.parameter_geometry.norm(parameters, tangent_step)
        tangent_residual = self.parameter_geometry.maximum_tangent_residual(
            parameters, tangent_step
        )
        transported_residual = self.parameter_geometry.maximum_tangent_residual(
            destination, displacement
        )
        transport_distortion = (
            self.parameter_geometry.maximum_transport_metric_distortion(
                parameters, destination, tangent_step, displacement
            )
        )
        metrics = RiemannianStepMetrics(
            result.rate,
            gradient_norm,
            jnp.asarray(1.0, dtype=gradient_norm.dtype),
            tangent_step_norm,
            jnp.asarray(0.0, dtype=gradient_norm.dtype),
            constraint_residual=self.parameter_geometry.maximum_constraint_residual(
                destination
            ),
            tangent_residual=tangent_residual,
            transported_tangent_residual=transported_residual,
            transport_metric_distortion=transport_distortion,
            line_search_evaluations=result.evaluations,
            line_search_accepted=result.accepted,
            line_search_reduction=jnp.asarray(value) - result.value,
            history_pair_count=count,
        )
        return destination, RiemannianLBFGSState(
            step=state.step + jnp.asarray(1, dtype=state.step.dtype),
            s_history=inserted_s,
            y_history=inserted_y,
            rho=rho,
            active=active,
            count=count,
            next_index=next_index,
            line_search_evaluations=result.evaluations,
            line_search_accepted=result.accepted,
            pair_accepted=pair_accepted,
            metrics=metrics,
        )

    def step_metrics(self, state: RiemannianLBFGSState, /) -> RiemannianStepMetrics:
        if not isinstance(state, RiemannianLBFGSState):
            raise TypeError("RiemannianLBFGS requires RiemannianLBFGSState.")
        return state.metrics


def riemannian_conjugate_gradient(
    parameter_geometry: ParameterGeometry,
    /,
    **kwargs: Any,
) -> RiemannianConjugateGradient:
    """Construct transported Polak–Ribière+ conjugate gradient."""
    return RiemannianConjugateGradient(parameter_geometry, **kwargs)


def riemannian_lbfgs(
    parameter_geometry: ParameterGeometry,
    /,
    **kwargs: Any,
) -> RiemannianLBFGS:
    """Construct transported limited-memory Riemannian BFGS."""
    return RiemannianLBFGS(parameter_geometry, **kwargs)


__all__ = [
    "AbstractRiemannianLineSearchOptimizer",
    "RiemannianConjugateGradient",
    "RiemannianConjugateGradientState",
    "RiemannianLBFGS",
    "RiemannianLBFGSState",
    "riemannian_conjugate_gradient",
    "riemannian_lbfgs",
]

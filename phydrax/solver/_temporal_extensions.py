#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded explicit, implicit, exponential, and parallel time integrations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..linalg import ArraySpace, DenseLinearOperator, matrix_phi1_action
from ._radau_iia import RadauIIAMethod
from ._temporal_method import TemporalMethodCapabilities


RHS = Callable[[Array, Array, object], Array]


@dataclass(frozen=True, slots=True)
class FixedStepTemporalResult:
    times: Array
    states: Array
    finite: Array
    method_id: str


class RKCMethod:
    """First-order stabilized Runge–Kutta–Chebyshev method."""

    def __init__(self, stages: int = 8, /):
        stages_ = int(stages)
        if stages_ < 2:
            raise ValueError("RKC stages must be at least two.")
        self.stages = stages_
        self.method_id = f"temporal:rkc1:stages-{stages_}"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="erk",
            order=1,
            adaptive=False,
            stage_abscissae=tuple((index / stages_) ** 2 for index in range(stages_ + 1)),
            causal_stage_extent=1.0,
            verified=True,
            method_id=self.method_id,
        )

    def step(self, rhs: RHS, time: Array, state: Array, step: Array, args=None) -> Array:
        stages = self.stages
        scaled = step / float(stages * stages)
        previous = state
        current = state + scaled * rhs(time, state, args)
        for stage in range(2, stages + 1):
            stage_time = time + ((stage - 1) / stages) ** 2 * step
            following = (
                2.0 * current - previous + 2.0 * scaled * rhs(stage_time, current, args)
            )
            previous, current = current, following
        return current


_AB_COEFFICIENTS = {
    2: (3.0 / 2.0, -1.0 / 2.0),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}
_AM_COEFFICIENTS = {
    2: (1.0 / 2.0, 1.0 / 2.0),
    3: (5.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0),
    4: (9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0),
}


class AdamsBashforthMoultonMethod:
    """Fixed-step PECE Adams predictor-corrector of orders two through four."""

    def __init__(self, order: int = 4, /):
        order_ = int(order)
        if order_ not in _AB_COEFFICIENTS:
            raise ValueError("Adams Bashforth-Moulton order must be two, three, or four.")
        self.order = order_
        self.method_id = f"temporal:adams-bashforth-moulton:order-{order_}"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="linear-multistep",
            order=order_,
            embedded_order=order_ - 1,
            adaptive=False,
            history_depth=order_,
            verified=True,
            method_id=self.method_id,
        )

    @staticmethod
    def _rk4(rhs: RHS, time: Array, state: Array, step: Array, args) -> Array:
        first = rhs(time, state, args)
        second = rhs(time + 0.5 * step, state + 0.5 * step * first, args)
        third = rhs(time + 0.5 * step, state + 0.5 * step * second, args)
        fourth = rhs(time + step, state + step * third, args)
        return state + step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0

    def integrate(self, rhs: RHS, times: ArrayLike, initial: ArrayLike, /, *, args=None):
        times_ = jnp.asarray(times)
        state = jnp.asarray(initial)
        _uniform_times(times_)
        states = [state]
        derivatives = [jnp.asarray(rhs(times_[0], state, args))]
        for index in range(1, times_.size):
            step = times_[index] - times_[index - 1]
            if len(derivatives) < self.order:
                state = self._rk4(rhs, times_[index - 1], state, step, args)
            else:
                recent = derivatives[-self.order :][::-1]
                predictor = state + step * sum(
                    coefficient * derivative
                    for coefficient, derivative in zip(
                        _AB_COEFFICIENTS[self.order], recent, strict=True
                    )
                )
                predicted = jnp.asarray(rhs(times_[index], predictor, args))
                corrector_terms = (predicted, *recent[: self.order - 1])
                state = state + step * sum(
                    coefficient * derivative
                    for coefficient, derivative in zip(
                        _AM_COEFFICIENTS[self.order], corrector_terms, strict=True
                    )
                )
            derivatives.append(jnp.asarray(rhs(times_[index], state, args)))
            states.append(state)
        stacked = jnp.stack(states)
        return FixedStepTemporalResult(
            times_, stacked, jnp.all(jnp.isfinite(stacked)), self.method_id
        )


class ExponentialRosenbrockEulerMethod:
    """Second-order exponential Rosenbrock-Euler with a native phi-one action."""

    def __init__(self):
        self.method_id = "temporal:exponential-rosenbrock-euler"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="exponential",
            order=2,
            adaptive=False,
            verified=True,
            method_id=self.method_id,
        )

    def step(self, rhs: RHS, time: Array, state: Array, step: Array, args=None) -> Array:
        value = jnp.asarray(rhs(time, state, args))
        jacobian = jax.jacfwd(lambda current: rhs(time, current, args))(state)
        space = ArraySpace(state.shape, dtype=state.dtype)
        operator = DenseLinearOperator(jacobian, source=space, target=space)
        action = matrix_phi1_action(operator, value, step)
        return state + step * action.value


class RadauIIAIntegrator:
    """Dense-array fixed-step Radau IIA with bounded Newton stage solves."""

    def __init__(
        self,
        stage_count: int = 3,
        /,
        *,
        maximum_newton_steps: int = 8,
        residual_tolerance: float = 1.0e-10,
    ):
        tableau = RadauIIAMethod(stage_count)
        steps = int(maximum_newton_steps)
        tolerance = float(residual_tolerance)
        if steps <= 0 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Radau Newton limits must be finite and positive.")
        self.tableau = tableau
        self.maximum_newton_steps = steps
        self.residual_tolerance = tolerance
        self.method_id = f"temporal:radau-iia-integrator:{tableau.method_id}"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode", "implicit-residual"),
            method_class="irk",
            order=tableau.order,
            stage_order=stage_count,
            adaptive=False,
            stage_abscissae=tuple(float(value) for value in np.asarray(tableau.nodes)),
            a_stable=True,
            l_stable=True,
            stiffly_accurate=True,
            verified=True,
            method_id=self.method_id,
        )

    def step(self, rhs: RHS, time: Array, state: Array, step: Array, args=None):
        stages = self.tableau.stage_count
        initial_derivative = jnp.asarray(rhs(time, state, args))
        stage_values = jnp.broadcast_to(initial_derivative, (stages, *state.shape))

        def residual(values):
            states = state + step * jnp.tensordot(self.tableau.matrix, values, axes=1)
            evaluated = jax.vmap(
                lambda node, value: rhs(time + step * node, value, args)
            )(self.tableau.nodes, states)
            return values - evaluated

        for _ in range(self.maximum_newton_steps):
            defect = residual(stage_values)
            flat_defect = defect.reshape((-1,))
            jacobian = jax.jacfwd(lambda value: residual(value).reshape((-1,)))(
                stage_values
            )
            jacobian = jacobian.reshape((flat_defect.size, flat_defect.size))
            correction = jnp.linalg.solve(jacobian, -flat_defect).reshape(
                stage_values.shape
            )
            stage_values = stage_values + correction
        defect_norm = jnp.linalg.norm(residual(stage_values))
        next_state = state + step * jnp.tensordot(
            self.tableau.weights, stage_values, axes=1
        )
        accepted = jnp.isfinite(defect_norm) & (defect_norm <= self.residual_tolerance)
        return jnp.where(accepted, next_state, state), defect_norm, accepted


class IMEXBDF2Integrator:
    """Fixed-step second-order IMEX-BDF with extrapolated explicit forcing."""

    def __init__(
        self,
        *,
        maximum_newton_steps: int = 8,
        residual_tolerance: float = 1.0e-10,
    ):
        self.maximum_newton_steps = int(maximum_newton_steps)
        self.residual_tolerance = float(residual_tolerance)
        if self.maximum_newton_steps <= 0 or self.residual_tolerance <= 0.0:
            raise ValueError("IMEX-BDF Newton limits must be positive.")
        self.method_id = "temporal:imex-bdf2"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("additive-ode", "split-residual"),
            method_class="bdf",
            order=2,
            adaptive=False,
            history_depth=2,
            a_stable=True,
            stiffly_accurate=True,
            verified=True,
            method_id=self.method_id,
        )

    def step(
        self,
        explicit_rhs: RHS,
        implicit_rhs: RHS,
        time: Array,
        previous: Array,
        current: Array,
        step: Array,
        args=None,
    ):
        explicit_current = explicit_rhs(time, current, args)
        explicit_previous = explicit_rhs(time - step, previous, args)
        base = (
            4.0 * current / 3.0
            - previous / 3.0
            + 2.0 * step * (2.0 * explicit_current - explicit_previous) / 3.0
        )
        candidate = base + 2.0 * step * implicit_rhs(time + step, current, args) / 3.0

        def residual(value):
            return (
                value - base - 2.0 * step * implicit_rhs(time + step, value, args) / 3.0
            )

        for _ in range(self.maximum_newton_steps):
            defect = residual(candidate)
            jacobian = jax.jacfwd(residual)(candidate)
            candidate = candidate + jnp.linalg.solve(
                jacobian.reshape((defect.size, defect.size)), -defect.reshape((-1,))
            ).reshape(candidate.shape)
        defect_norm = jnp.linalg.norm(residual(candidate))
        accepted = jnp.isfinite(defect_norm) & (defect_norm <= self.residual_tolerance)
        return jnp.where(accepted, candidate, current), defect_norm, accepted


def parareal(
    coarse: Callable[[Array, Array, object], Array],
    fine: Callable[[Array, Array, object], Array],
    times: ArrayLike,
    initial: ArrayLike,
    iterations: int,
    /,
    *,
    args=None,
) -> FixedStepTemporalResult:
    """Deterministic Parareal iteration over a fixed coarse time partition."""

    times_ = jnp.asarray(times)
    _increasing_times(times_)
    count = times_.size - 1
    iteration_count = int(iterations)
    if iteration_count < 0:
        raise ValueError("Parareal iterations must be non-negative.")
    states = [jnp.asarray(initial)]
    for index in range(count):
        states.append(coarse(times_[index], states[-1], args))
    current = states
    for _ in range(iteration_count):
        fine_values = tuple(
            fine(times_[index], current[index], args) for index in range(count)
        )
        next_states = [jnp.asarray(initial)]
        for index in range(count):
            next_states.append(
                coarse(times_[index], next_states[-1], args)
                + fine_values[index]
                - coarse(times_[index], current[index], args)
            )
        current = next_states
    stacked = jnp.stack(current)
    return FixedStepTemporalResult(
        times_,
        stacked,
        jnp.all(jnp.isfinite(stacked)),
        f"temporal:parareal:iterations-{iteration_count}",
    )


def _increasing_times(times: Array) -> None:
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a vector with at least two entries.")
    host = np.asarray(times)
    if not np.all(np.isfinite(host)) or np.any(np.diff(host) <= 0.0):
        raise ValueError("times must be finite and strictly increasing.")


def _uniform_times(times: Array) -> None:
    _increasing_times(times)
    differences = np.diff(np.asarray(times))
    tolerance = 64.0 * np.finfo(differences.dtype).eps * max(1.0, abs(differences[0]))
    if not np.allclose(differences, differences[0], rtol=0.0, atol=tolerance):
        raise ValueError("This fixed-step temporal method requires uniform times.")


__all__ = [
    "AdamsBashforthMoultonMethod",
    "ExponentialRosenbrockEulerMethod",
    "FixedStepTemporalResult",
    "IMEXBDF2Integrator",
    "RKCMethod",
    "RadauIIAIntegrator",
    "parareal",
]

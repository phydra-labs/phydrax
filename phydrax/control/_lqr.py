#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite- and infinite-horizon linear-quadratic regulation."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ._parameterization import (
    _case_shape,
    _coefficient_array,
    _query,
    AbstractControlParameterization,
)
from ._problem import _identifier
from ._riccati import (
    _require_positive_definite,
    _require_positive_semidefinite,
    _require_shape,
    AlgebraicRiccatiDiagnostics,
    RiccatiStatus,
    solve_continuous_are,
    solve_discrete_are,
)


class AffineFeedbackPolicy(AbstractControlParameterization):
    """Affine state feedback ``u(t, x) = K(t)x + k(t)``.

    A finite policy carries one gain per interval of ``time_grid``. An infinite
    policy has no time grid and carries one constant gain. Coefficients are a
    scalar-shaped, intentionally ignored token because the optimized gains are
    already stored by the policy. Feedback policies cannot be sampled without
    a state trajectory; use :meth:`evaluate` online through ``ControlProblem``.
    """

    feedback_gain: Array
    feedforward: Array
    time_grid: TimeGrid | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    finite_horizon: bool = eqx.field(static=True)

    def __init__(
        self,
        feedback_gain: ArrayLike,
        feedforward: ArrayLike,
        /,
        *,
        time_grid: TimeGrid | None,
        state_size: int,
        case_shape: Sequence[int] = (),
        policy_id: str,
        _allow_nonfinite: bool = False,
    ):
        cases = _case_shape(case_shape)
        if not isinstance(state_size, int) or state_size <= 0:
            raise ValueError("state_size must be a positive integer.")
        gain = jnp.asarray(feedback_gain)
        bias = jnp.asarray(feedforward)
        finite = time_grid is not None
        if finite:
            if not isinstance(time_grid, TimeGrid):
                raise TypeError("time_grid must be a TimeGrid or None.")
            horizon = time_grid.num_steps
            if gain.ndim < 3:
                raise ValueError("Finite feedback gains must include a time axis.")
            expected_gain_prefix = cases + (horizon,)
            expected_bias_prefix = cases + (horizon,)
        else:
            expected_gain_prefix = cases
            expected_bias_prefix = cases
        if tuple(gain.shape[:-2]) != expected_gain_prefix or gain.shape[-1] != state_size:
            expected = expected_gain_prefix + ("control_size", state_size)
            raise ValueError(
                f"feedback_gain must have shape {expected}; got {gain.shape}."
            )
        control_size = int(gain.shape[-2])
        expected_bias = expected_bias_prefix + (control_size,)
        if tuple(bias.shape) != expected_bias:
            raise ValueError(
                f"feedforward must have shape {expected_bias}; got {bias.shape}."
            )
        if not jnp.issubdtype(gain.dtype, jnp.inexact):
            gain = gain.astype(float)
        if not jnp.issubdtype(bias.dtype, jnp.inexact):
            bias = bias.astype(float)
        if not _allow_nonfinite:
            gain = eqx.error_if(
                gain, jnp.any(~jnp.isfinite(gain)), "feedback_gain must be finite."
            )
            bias = eqx.error_if(
                bias, jnp.any(~jnp.isfinite(bias)), "feedforward must be finite."
            )
        self.feedback_gain = gain
        self.feedforward = bias
        self.time_grid = time_grid
        self.state_shape = (state_size,)
        self.case_shape = cases
        self.finite_horizon = finite
        self.control_shape = (control_size,)
        self.parameter_shape = ()
        self.parameterization_id = _identifier(policy_id, "policy_id")
        self.approximation_id = (
            "control:affine-state-feedback:piecewise-constant"
            if finite
            else "control:affine-state-feedback:stationary"
        )

    def _selected_gain(self, time: Array, /) -> tuple[Array, Array]:
        if not self.finite_horizon:
            return self.feedback_gain, self.feedforward
        assert self.time_grid is not None
        time = eqx.error_if(
            time,
            (time < self.time_grid.t0) | (time > self.time_grid.t1),
            "Feedback-policy time lies outside its physical grid.",
        )
        index = jnp.searchsorted(self.time_grid.times, time, side="right") - 1
        index = jnp.minimum(index, self.time_grid.num_steps - 1)
        axis = len(self.case_shape)
        return (
            jnp.take(self.feedback_gain, index, axis=axis),
            jnp.take(self.feedforward, index, axis=axis),
        )

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        cases = _case_shape(case_shape)
        if cases != self.case_shape:
            raise ValueError(
                f"case_shape must match the policy case shape {self.case_shape}; got {cases}."
            )
        _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        if query.shape != ():
            raise ValueError("AffineFeedbackPolicy.evaluate requires a scalar time.")
        if state is None:
            raise ValueError("AffineFeedbackPolicy.evaluate requires the current state.")
        state_ = jnp.asarray(state)
        expected_state = cases + self.state_shape
        if tuple(state_.shape) != expected_state:
            raise ValueError(
                f"Feedback state must have shape {expected_state}; got {state_.shape}."
            )
        if not jnp.issubdtype(state_.dtype, jnp.inexact):
            state_ = state_.astype(float)
        state_ = eqx.error_if(
            state_, jnp.any(~jnp.isfinite(state_)), "Feedback state must be finite."
        )
        gain, bias = self._selected_gain(query)
        return jnp.einsum("...ij,...j->...i", gain, state_) + bias

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        del coefficients, times, case_shape
        raise ValueError(
            "AffineFeedbackPolicy cannot be sampled without states; evaluate it "
            "online or roll it out through ControlProblem."
        )

    def __call__(self, time: ArrayLike, state: ArrayLike, args=None) -> Array:
        del args
        coefficients = jnp.zeros(self.case_shape, dtype=self.feedback_gain.dtype)
        return self.evaluate(coefficients, time, case_shape=self.case_shape, state=state)


class QuadraticValueFunction(StrictModule):
    """Quadratic value ``V(t,x)=xᵀP(t)x/2+p(t)ᵀx+s(t)``."""

    matrices: Array
    linear: Array
    constants: Array
    time_grid: TimeGrid | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    finite_horizon: bool = eqx.field(static=True)

    def __init__(
        self,
        matrices: ArrayLike,
        linear: ArrayLike,
        constants: ArrayLike,
        /,
        *,
        time_grid: TimeGrid | None,
        case_shape: Sequence[int] = (),
    ):
        cases = _case_shape(case_shape)
        matrix = jnp.asarray(matrices)
        vector = jnp.asarray(linear)
        scalar = jnp.asarray(constants)
        finite = time_grid is not None
        if matrix.ndim < 2 or matrix.shape[-2] != matrix.shape[-1]:
            raise ValueError("Value matrices must end in square matrix dimensions.")
        n = int(matrix.shape[-1])
        prefix = cases + ((time_grid.num_times,) if finite else ())
        if tuple(matrix.shape) != prefix + (n, n):
            raise ValueError(
                f"Value matrices must have shape {prefix + (n, n)}; got {matrix.shape}."
            )
        if tuple(vector.shape) != prefix + (n,):
            raise ValueError(
                f"Value linear terms must have shape {prefix + (n,)}; got {vector.shape}."
            )
        if tuple(scalar.shape) != prefix:
            raise ValueError(
                f"Value constants must have shape {prefix}; got {scalar.shape}."
            )
        dtype = jnp.result_type(matrix, vector, scalar, float)
        self.matrices = matrix.astype(dtype)
        self.linear = vector.astype(dtype)
        self.constants = scalar.astype(dtype)
        self.time_grid = time_grid
        self.state_shape = (n,)
        self.case_shape = cases
        self.finite_horizon = finite

    def evaluate(self, time: ArrayLike, state: ArrayLike, /) -> Array:
        query = _query(time)
        if query.shape != ():
            raise ValueError("QuadraticValueFunction.evaluate requires a scalar time.")
        state_ = jnp.asarray(state)
        expected = self.case_shape + self.state_shape
        if tuple(state_.shape) != expected:
            raise ValueError(f"state must have shape {expected}; got {state_.shape}.")
        if self.finite_horizon:
            assert self.time_grid is not None
            query = eqx.error_if(
                query,
                (query < self.time_grid.t0) | (query > self.time_grid.t1),
                "Value-function time lies outside its physical grid.",
            )
            index = jnp.searchsorted(self.time_grid.times, query, side="right") - 1
            index = jnp.minimum(index, self.time_grid.num_times - 1)
            axis = len(self.case_shape)
            matrix = jnp.take(self.matrices, index, axis=axis)
            linear = jnp.take(self.linear, index, axis=axis)
            constant = jnp.take(self.constants, index, axis=axis)
        else:
            matrix = self.matrices
            linear = self.linear
            constant = self.constants
        return (
            0.5 * jnp.einsum("...i,...ij,...j->...", state_, matrix, state_)
            + jnp.einsum("...i,...i->...", linear, state_)
            + constant
        )

    __call__ = evaluate


class FiniteHorizonLQRDiagnostics(StrictModule):
    """Per-stage Riccati/KKT residuals and effective-Hessian conditioning."""

    riccati_residuals: Array
    kkt_residuals: Array
    control_condition_numbers: Array
    maximum_riccati_residual: Array
    maximum_kkt_residual: Array
    maximum_control_condition_number: Array
    finite: Array
    converged: Array
    status: Array
    method: str = eqx.field(static=True)


class FiniteHorizonLQRResult(StrictModule):
    """Finite-horizon affine feedback, value coefficients, and diagnostics."""

    policy: AffineFeedbackPolicy
    value: QuadraticValueFunction
    diagnostics: FiniteHorizonLQRDiagnostics
    valid: Array
    status: Array

    @property
    def feedback_gain(self) -> Array:
        return self.policy.feedback_gain

    @property
    def feedforward(self) -> Array:
        return self.policy.feedforward


class InfiniteHorizonLQRResult(StrictModule):
    """Stationary LQR feedback and algebraic Riccati evidence.

    The policy retains the raw diagnosed gain for every case so the result has a
    transform-stable PyTree structure. A gain belonging to an invalid case is
    failure evidence only and may be non-finite; inspect ``valid`` before use.
    """

    policy: AffineFeedbackPolicy
    value: QuadraticValueFunction
    diagnostics: AlgebraicRiccatiDiagnostics
    valid: Array
    status: Array

    @property
    def feedback_gain(self) -> Array:
        return self.policy.feedback_gain

    @property
    def value_matrix(self) -> Array:
        return self.value.matrices


def _finite_inputs(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_cost: ArrayLike,
    dynamics_bias: ArrayLike | None,
    state_control_cross: ArrayLike | None,
    state_linear: ArrayLike | None,
    control_linear: ArrayLike | None,
    stage_constants: ArrayLike | None,
    terminal_linear: ArrayLike | None,
    terminal_constant: ArrayLike,
    cost_tolerance: float,
    /,
):
    a = jnp.asarray(dynamics_matrices)
    if a.ndim < 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError(
            "dynamics_matrices must have shape case_shape + (horizon, n, n)."
        )
    case_shape = tuple(a.shape[:-3])
    horizon = int(a.shape[-3])
    n = int(a.shape[-1])
    if horizon < 1:
        raise ValueError("Finite-horizon LQR requires at least one stage.")
    b = jnp.asarray(control_matrices)
    if (
        b.ndim < 3
        or tuple(b.shape[:-3]) != case_shape
        or b.shape[-3] != horizon
        or b.shape[-2] != n
    ):
        raise ValueError(
            "control_matrices must have shape case_shape + (horizon, n, m); "
            f"got {b.shape}."
        )
    m = int(b.shape[-1])
    a = _require_shape(a, case_shape + (horizon, n, n), "dynamics_matrices")
    b = _require_shape(b, case_shape + (horizon, n, m), "control_matrices")
    q = _require_shape(state_costs, case_shape + (horizon, n, n), "state_costs")
    r = _require_shape(control_costs, case_shape + (horizon, m, m), "control_costs")
    q_terminal = _require_shape(
        terminal_state_cost, case_shape + (n, n), "terminal_state_cost"
    )
    dtype = jnp.result_type(a, b, q, r, q_terminal, float)
    zeros = lambda shape: jnp.zeros(shape, dtype=dtype)
    c = (
        zeros(case_shape + (horizon, n))
        if dynamics_bias is None
        else _require_shape(dynamics_bias, case_shape + (horizon, n), "dynamics_bias")
    )
    cross = (
        zeros(case_shape + (horizon, n, m))
        if state_control_cross is None
        else _require_shape(
            state_control_cross,
            case_shape + (horizon, n, m),
            "state_control_cross",
        )
    )
    q_linear = (
        zeros(case_shape + (horizon, n))
        if state_linear is None
        else _require_shape(state_linear, case_shape + (horizon, n), "state_linear")
    )
    r_linear = (
        zeros(case_shape + (horizon, m))
        if control_linear is None
        else _require_shape(control_linear, case_shape + (horizon, m), "control_linear")
    )
    constants = (
        zeros(case_shape + (horizon,))
        if stage_constants is None
        else _require_shape(stage_constants, case_shape + (horizon,), "stage_constants")
    )
    q_terminal_linear = (
        zeros(case_shape + (n,))
        if terminal_linear is None
        else _require_shape(terminal_linear, case_shape + (n,), "terminal_linear")
    )
    terminal_constant_value = jnp.asarray(terminal_constant)
    if terminal_constant_value.shape == () and case_shape:
        terminal_constant_value = jnp.broadcast_to(terminal_constant_value, case_shape)
    terminal_constant_ = _require_shape(
        terminal_constant_value, case_shape, "terminal_constant"
    )
    q = _require_positive_semidefinite(q, "state_costs", cost_tolerance)
    r = _require_positive_definite(r, "control_costs", cost_tolerance)
    q_terminal = _require_positive_semidefinite(
        q_terminal, "terminal_state_cost", cost_tolerance
    )
    stage_hessian = jnp.concatenate(
        (
            jnp.concatenate((q, cross), axis=-1),
            jnp.concatenate((jnp.swapaxes(cross, -1, -2), r), axis=-1),
        ),
        axis=-2,
    )
    stage_hessian = _require_positive_semidefinite(
        stage_hessian, "joint stage costs", cost_tolerance
    )
    q = stage_hessian[..., :n, :n]
    cross = stage_hessian[..., :n, n:]
    r = stage_hessian[..., n:, n:]
    values = (
        a,
        b,
        q,
        r,
        q_terminal,
        c,
        cross,
        q_linear,
        r_linear,
        constants,
        q_terminal_linear,
        terminal_constant_,
    )
    return tuple(value.astype(dtype) for value in values), case_shape, horizon, n, m


def finite_horizon_lqr(
    dynamics_matrices: ArrayLike,
    control_matrices: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_cost: ArrayLike,
    /,
    *,
    dynamics_bias: ArrayLike | None = None,
    state_control_cross: ArrayLike | None = None,
    state_linear: ArrayLike | None = None,
    control_linear: ArrayLike | None = None,
    stage_constants: ArrayLike | None = None,
    terminal_linear: ArrayLike | None = None,
    terminal_constant: ArrayLike = 0.0,
    time_grid: TimeGrid | None = None,
    policy_id: str = "lqr:finite-horizon",
    tolerance: float = 1e-9,
    cost_tolerance: float = 1e-10,
) -> FiniteHorizonLQRResult:
    """Solve a time-varying affine finite-horizon discrete LQR problem.

    Dynamics are ``x[t+1] = A[t]x[t] + B[t]u[t] + c[t]``. Stage
    costs use the convention ``xᵀQx/2 + uᵀRu/2 + xᵀNu + qᵀx + rᵀu + d``.
    Every stage array has an explicit time axis; no time broadcasting occurs.
    """
    if tolerance <= 0.0 or cost_tolerance < 0.0:
        raise ValueError("tolerance must be positive and cost_tolerance non-negative.")
    values, case_shape, horizon, n, _ = _finite_inputs(
        dynamics_matrices,
        control_matrices,
        state_costs,
        control_costs,
        terminal_state_cost,
        dynamics_bias,
        state_control_cross,
        state_linear,
        control_linear,
        stage_constants,
        terminal_linear,
        terminal_constant,
        cost_tolerance,
    )
    (
        a,
        b,
        q,
        r,
        q_terminal,
        c,
        cross,
        q_linear,
        r_linear,
        constants,
        q_terminal_linear,
        terminal_constant_,
    ) = values
    if time_grid is None:
        time_grid = TimeGrid(
            jnp.arange(horizon + 1, dtype=a.dtype), time_id=f"{policy_id}:time"
        )
    elif not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid or None.")
    if time_grid.num_steps != horizon:
        raise ValueError(
            f"time_grid must contain {horizon + 1} times for this LQR horizon."
        )

    def to_time_major(value: Array, payload_rank: int) -> Array:
        return jnp.moveaxis(value, -(payload_rank + 1), 0)

    inputs = (
        to_time_major(a, 2),
        to_time_major(b, 2),
        to_time_major(q, 2),
        to_time_major(r, 2),
        to_time_major(c, 1),
        to_time_major(cross, 2),
        to_time_major(q_linear, 1),
        to_time_major(r_linear, 1),
        to_time_major(constants, 0),
    )

    def step(carry, stage):
        p_next, linear_next, constant_next = carry
        a_t, b_t, q_t, r_t, c_t, cross_t, q_t_linear, r_t_linear, d_t = stage
        p_b = p_next @ b_t
        control_hessian = r_t + jnp.swapaxes(b_t, -1, -2) @ p_b
        control_hessian = _require_positive_definite(
            control_hessian,
            "effective control Hessian in the Riccati recursion",
            cost_tolerance,
        )
        state_control = jnp.swapaxes(b_t, -1, -2) @ p_next @ a_t + jnp.swapaxes(
            cross_t, -1, -2
        )
        affine_next = jnp.einsum("...ij,...j->...i", p_next, c_t) + linear_next
        control_affine = r_t_linear + jnp.einsum("...ji,...j->...i", b_t, affine_next)
        feedback = -jnp.linalg.solve(control_hessian, state_control)
        feedforward = -jnp.linalg.solve(control_hessian, control_affine[..., None])[
            ..., 0
        ]
        p_raw = (
            q_t
            + jnp.swapaxes(a_t, -1, -2) @ p_next @ a_t
            + jnp.swapaxes(state_control, -1, -2) @ feedback
        )
        p_current = 0.5 * (p_raw + jnp.swapaxes(p_raw, -1, -2))
        linear_current = (
            q_t_linear
            + jnp.einsum("...ji,...j->...i", a_t, affine_next)
            + jnp.einsum("...ji,...j->...i", state_control, feedforward)
        )
        constant_current = (
            d_t
            + constant_next
            + 0.5 * jnp.einsum("...i,...ij,...j->...", c_t, p_next, c_t)
            + jnp.einsum("...i,...i->...", linear_next, c_t)
            + 0.5 * jnp.einsum("...i,...i->...", control_affine, feedforward)
        )
        stationarity_matrix = control_hessian @ feedback + state_control
        stationarity_vector = (
            jnp.einsum("...ij,...j->...i", control_hessian, feedforward) + control_affine
        )
        kkt_residual = jnp.sqrt(
            jnp.sum(jnp.square(stationarity_matrix), axis=(-2, -1))
            + jnp.sum(jnp.square(stationarity_vector), axis=-1)
        )
        value_matrix_residual = p_current - p_raw
        value_linear_residual = linear_current - (
            q_t_linear
            + jnp.einsum("...ji,...j->...i", a_t, affine_next)
            + jnp.einsum("...ji,...j->...i", state_control, feedforward)
        )
        value_constant_residual = constant_current - (
            d_t
            + constant_next
            + 0.5 * jnp.einsum("...i,...ij,...j->...", c_t, p_next, c_t)
            + jnp.einsum("...i,...i->...", linear_next, c_t)
            + 0.5 * jnp.einsum("...i,...i->...", control_affine, feedforward)
        )
        riccati_residual = jnp.sqrt(
            jnp.sum(jnp.square(value_matrix_residual), axis=(-2, -1))
            + jnp.sum(jnp.square(value_linear_residual), axis=-1)
            + jnp.square(value_constant_residual)
        )
        condition = jnp.linalg.cond(control_hessian)
        output = (
            p_current,
            linear_current,
            constant_current,
            feedback,
            feedforward,
            riccati_residual,
            kkt_residual,
            condition,
        )
        return (p_current, linear_current, constant_current), output

    _, outputs = jax.lax.scan(
        step,
        (q_terminal, q_terminal_linear, terminal_constant_),
        inputs,
        reverse=True,
    )
    (
        p_stages,
        linear_stages,
        constant_stages,
        feedback,
        feedforward,
        riccati_residuals,
        kkt_residuals,
        conditions,
    ) = outputs
    p_all = jnp.concatenate((p_stages, q_terminal[None, ...]), axis=0)
    linear_all = jnp.concatenate((linear_stages, q_terminal_linear[None, ...]), axis=0)
    constant_all = jnp.concatenate(
        (constant_stages, terminal_constant_[None, ...]), axis=0
    )
    p_all = jnp.moveaxis(p_all, 0, -3)
    linear_all = jnp.moveaxis(linear_all, 0, -2)
    constant_all = jnp.moveaxis(constant_all, 0, -1)
    feedback = jnp.moveaxis(feedback, 0, -3)
    feedforward = jnp.moveaxis(feedforward, 0, -2)
    riccati_residuals = jnp.moveaxis(riccati_residuals, 0, -1)
    kkt_residuals = jnp.moveaxis(kkt_residuals, 0, -1)
    conditions = jnp.moveaxis(conditions, 0, -1)
    maximum_riccati = jnp.max(riccati_residuals, axis=-1)
    maximum_kkt = jnp.max(kkt_residuals, axis=-1)
    maximum_condition = jnp.max(conditions, axis=-1)
    finite = (
        jnp.all(jnp.isfinite(p_all), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(linear_all), axis=(-2, -1))
        & jnp.all(jnp.isfinite(constant_all), axis=-1)
        & jnp.all(jnp.isfinite(feedback), axis=(-3, -2, -1))
        & jnp.all(jnp.isfinite(feedforward), axis=(-2, -1))
        & jnp.all(jnp.isfinite(riccati_residuals), axis=-1)
        & jnp.all(jnp.isfinite(kkt_residuals), axis=-1)
        & jnp.all(jnp.isfinite(conditions), axis=-1)
        & jnp.isfinite(maximum_riccati)
        & jnp.isfinite(maximum_kkt)
        & jnp.isfinite(maximum_condition)
    )
    converged = finite & (maximum_riccati <= tolerance) & (maximum_kkt <= tolerance)
    status = jnp.where(
        ~finite,
        int(RiccatiStatus.NONFINITE),
        jnp.where(
            converged,
            int(RiccatiStatus.SUCCESS),
            int(RiccatiStatus.NONCONVERGED),
        ),
    ).astype(jnp.int32)
    policy = AffineFeedbackPolicy(
        feedback,
        feedforward,
        time_grid=time_grid,
        state_size=n,
        case_shape=case_shape,
        policy_id=policy_id,
    )
    value = QuadraticValueFunction(
        p_all,
        linear_all,
        constant_all,
        time_grid=time_grid,
        case_shape=case_shape,
    )
    diagnostics = FiniteHorizonLQRDiagnostics(
        riccati_residuals=riccati_residuals,
        kkt_residuals=kkt_residuals,
        control_condition_numbers=conditions,
        maximum_riccati_residual=maximum_riccati,
        maximum_kkt_residual=maximum_kkt,
        maximum_control_condition_number=maximum_condition,
        finite=finite,
        converged=converged,
        status=status,
        method="sequential-riccati",
    )
    return FiniteHorizonLQRResult(
        policy=policy,
        value=value,
        diagnostics=diagnostics,
        valid=converged,
        status=status,
    )


def continuous_lqr(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    /,
    *,
    s: ArrayLike | None = None,
    policy_id: str = "lqr:continuous-infinite-horizon",
    tolerance: float = 1e-9,
    pbh_tolerance: float = 1e-9,
    cost_tolerance: float = 1e-10,
) -> InfiniteHorizonLQRResult:
    """Solve a continuous-time infinite-horizon LQR problem."""
    riccati = solve_continuous_are(
        a,
        b,
        q,
        r,
        s=s,
        tolerance=tolerance,
        pbh_tolerance=pbh_tolerance,
        cost_tolerance=cost_tolerance,
    )
    a_ = jnp.asarray(a)
    b_ = jnp.asarray(b)
    r_ = jnp.asarray(r)
    case_shape = tuple(a_.shape[:-2])
    n = int(a_.shape[-1])
    m = int(b_.shape[-1])
    s_ = (
        jnp.zeros(case_shape + (n, m), dtype=riccati.matrix.dtype)
        if s is None
        else jnp.asarray(s)
    )
    gain = -jnp.linalg.solve(
        r_,
        jnp.swapaxes(b_, -1, -2) @ riccati.matrix + jnp.swapaxes(s_, -1, -2),
    )
    policy = AffineFeedbackPolicy(
        gain,
        jnp.zeros(case_shape + (m,), dtype=gain.dtype),
        time_grid=None,
        state_size=n,
        case_shape=case_shape,
        policy_id=policy_id,
        _allow_nonfinite=True,
    )
    value = QuadraticValueFunction(
        riccati.matrix,
        jnp.zeros(case_shape + (n,), dtype=riccati.matrix.dtype),
        jnp.zeros(case_shape, dtype=riccati.matrix.dtype),
        time_grid=None,
        case_shape=case_shape,
    )
    return InfiniteHorizonLQRResult(
        policy=policy,
        value=value,
        diagnostics=riccati.diagnostics,
        valid=riccati.valid,
        status=riccati.status,
    )


def discrete_lqr(
    a: ArrayLike,
    b: ArrayLike,
    q: ArrayLike,
    r: ArrayLike,
    /,
    *,
    s: ArrayLike | None = None,
    policy_id: str = "lqr:discrete-infinite-horizon",
    tolerance: float = 1e-9,
    pbh_tolerance: float = 1e-9,
    cost_tolerance: float = 1e-10,
    max_iterations: int = 512,
) -> InfiniteHorizonLQRResult:
    """Solve a discrete-time infinite-horizon LQR problem."""
    riccati = solve_discrete_are(
        a,
        b,
        q,
        r,
        s=s,
        tolerance=tolerance,
        pbh_tolerance=pbh_tolerance,
        cost_tolerance=cost_tolerance,
        max_iterations=max_iterations,
    )
    a_ = jnp.asarray(a)
    b_ = jnp.asarray(b)
    r_ = jnp.asarray(r)
    case_shape = tuple(a_.shape[:-2])
    n = int(a_.shape[-1])
    m = int(b_.shape[-1])
    s_ = (
        jnp.zeros(case_shape + (n, m), dtype=riccati.matrix.dtype)
        if s is None
        else jnp.asarray(s)
    )
    control_hessian = r_ + jnp.swapaxes(b_, -1, -2) @ riccati.matrix @ b_
    gain = -jnp.linalg.solve(
        control_hessian,
        jnp.swapaxes(b_, -1, -2) @ riccati.matrix @ a_ + jnp.swapaxes(s_, -1, -2),
    )
    policy = AffineFeedbackPolicy(
        gain,
        jnp.zeros(case_shape + (m,), dtype=gain.dtype),
        time_grid=None,
        state_size=n,
        case_shape=case_shape,
        policy_id=policy_id,
        _allow_nonfinite=True,
    )
    value = QuadraticValueFunction(
        riccati.matrix,
        jnp.zeros(case_shape + (n,), dtype=riccati.matrix.dtype),
        jnp.zeros(case_shape, dtype=riccati.matrix.dtype),
        time_grid=None,
        case_shape=case_shape,
    )
    return InfiniteHorizonLQRResult(
        policy=policy,
        value=value,
        diagnostics=riccati.diagnostics,
        valid=riccati.valid,
        status=riccati.status,
    )

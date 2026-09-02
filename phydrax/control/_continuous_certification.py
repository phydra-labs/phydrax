#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import comb
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._direct_collocation import DirectCollocationResult


class ControlSegmentInterpolant(StrictModule, NonTrainableState):
    """Fixed-degree power-basis state/control polynomials on normalized segments."""

    times: Array
    state_coefficients: Array
    control_coefficients: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    interpolant_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        state_coefficients: ArrayLike,
        control_coefficients: ArrayLike,
        /,
        *,
        case_shape: Sequence[int],
        state_shape: Sequence[int],
        control_shape: Sequence[int],
        interpolant_id: str,
    ):
        times_ = jnp.asarray(times)
        state_ = jnp.asarray(state_coefficients)
        control_ = jnp.asarray(control_coefficients)
        cases = tuple(int(value) for value in case_shape)
        states = tuple(int(value) for value in state_shape)
        controls = tuple(int(value) for value in control_shape)
        if times_.ndim != 1 or times_.size < 2:
            raise ValueError(
                "segment interpolant times must be rank-one with at least two entries."
            )
        intervals = times_.size - 1
        prefix = cases + (intervals,)
        if (
            state_.ndim < len(prefix) + 2
            or state_.shape[: len(prefix)] != prefix
            or state_.shape[-len(states) :] != states
        ):
            raise ValueError(
                "state coefficients do not match declared case/interval/state axes."
            )
        if (
            control_.ndim < len(prefix) + 2
            or control_.shape[: len(prefix)] != prefix
            or control_.shape[-len(controls) :] != controls
        ):
            raise ValueError(
                "control coefficients do not match declared case/interval/control axes."
            )
        if not isinstance(interpolant_id, str) or not interpolant_id:
            raise ValueError("interpolant_id must be non-empty.")
        self.times = times_
        self.state_coefficients = state_
        self.control_coefficients = control_
        self.case_shape = cases
        self.state_shape = states
        self.control_shape = controls
        self.interpolant_id = interpolant_id

    @property
    def interval_count(self) -> int:
        return self.times.size - 1

    @property
    def state_degree(self) -> int:
        return self.state_coefficients.shape[len(self.case_shape) + 1] - 1

    @property
    def control_degree(self) -> int:
        return self.control_coefficients.shape[len(self.case_shape) + 1] - 1

    def evaluate(
        self, interval: int, normalized_time: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        theta = jnp.asarray(normalized_time)
        case_rank = len(self.case_shape)
        state_coefficients = jnp.take(self.state_coefficients, interval, axis=case_rank)
        control_coefficients = jnp.take(
            self.control_coefficients, interval, axis=case_rank
        )
        flat_theta = theta.reshape((-1,))

        def state_at(value):
            powers = jnp.power(value, jnp.arange(self.state_degree + 1))
            weights = powers.reshape(
                (1,) * case_rank + (self.state_degree + 1,) + (1,) * len(self.state_shape)
            )
            return jnp.sum(state_coefficients * weights, axis=case_rank)

        def control_at(value):
            powers = jnp.power(value, jnp.arange(self.control_degree + 1))
            weights = powers.reshape(
                (1,) * case_rank
                + (self.control_degree + 1,)
                + (1,) * len(self.control_shape)
            )
            return jnp.sum(control_coefficients * weights, axis=case_rank)

        state = jnp.moveaxis(jax.vmap(state_at)(flat_theta), 0, case_rank)
        control = jnp.moveaxis(jax.vmap(control_at)(flat_theta), 0, case_rank)
        state = state.reshape(self.case_shape + theta.shape + self.state_shape)
        control = control.reshape(self.case_shape + theta.shape + self.control_shape)
        time = self.times[interval] + theta * (
            self.times[interval + 1] - self.times[interval]
        )
        return time, state, control


def direct_collocation_interpolant(
    result: DirectCollocationResult, /
) -> ControlSegmentInterpolant:
    """Expose the current theta result's represented linear/held segment family."""

    if not isinstance(result, DirectCollocationResult):
        raise TypeError("result must be a DirectCollocationResult.")
    trajectory = result.trajectory
    interval_axis = len(trajectory.case_shape)
    left = jnp.take(
        trajectory.states, jnp.arange(trajectory.time_grid.num_steps), axis=interval_axis
    )
    right = jnp.take(
        trajectory.states,
        jnp.arange(1, trajectory.time_grid.num_times),
        axis=interval_axis,
    )
    state_coefficients = jnp.stack((left, right - left), axis=interval_axis + 1)
    control_coefficients = jnp.expand_dims(trajectory.controls, axis=interval_axis + 1)
    return ControlSegmentInterpolant(
        trajectory.time_grid.times,
        state_coefficients,
        control_coefficients,
        case_shape=trajectory.case_shape,
        state_shape=trajectory.state_shape,
        control_shape=trajectory.control_shape,
        interpolant_id=f"direct-collocation-segments:{result.result_id}",
    )


class AbstractPathConstraintEnvelope(StrictModule):
    """Typed interval bound provider; custom inheritance alone certifies nothing."""

    envelope_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)

    @abc.abstractmethod
    def bounds(
        self,
        interpolant: ControlSegmentInterpolant,
        residual: Callable[[Array, Array, Array, Any], Array],
        interval: int,
        args: Any,
        /,
    ) -> tuple[Array, Array, Array]:
        raise NotImplementedError


class AffineBernsteinPathEnvelope(AbstractPathConstraintEnvelope, NonTrainableState):
    """Convex-hull bounds for an affine residual over represented polynomials."""

    state_weights: Array
    control_weights: Array
    bias: Array

    def __init__(
        self,
        state_weights: ArrayLike,
        control_weights: ArrayLike,
        bias: ArrayLike = 0.0,
        /,
        *,
        envelope_id: str,
    ):
        self.state_weights = jnp.asarray(state_weights)
        self.control_weights = jnp.asarray(control_weights)
        self.bias = jnp.asarray(bias).reshape(())
        self.envelope_id = envelope_id
        self.conservative = True

    def bounds(self, interpolant, residual, interval, args, /):
        del residual, args
        case_rank = len(interpolant.case_shape)
        state = jnp.take(interpolant.state_coefficients, interval, axis=case_rank)
        control = jnp.take(interpolant.control_coefficients, interval, axis=case_rank)
        degree = max(interpolant.state_degree, interpolant.control_degree)
        state = jnp.pad(
            state,
            [(0, 0)] * case_rank
            + [(0, degree - interpolant.state_degree)]
            + [(0, 0)] * len(interpolant.state_shape),
        )
        control = jnp.pad(
            control,
            [(0, 0)] * case_rank
            + [(0, degree - interpolant.control_degree)]
            + [(0, 0)] * len(interpolant.control_shape),
        )
        flat_state = state.reshape(interpolant.case_shape + (degree + 1, -1))
        flat_control = control.reshape(interpolant.case_shape + (degree + 1, -1))
        power = oe.contract(
            "...di,i->...d", flat_state, self.state_weights.reshape((-1,))
        ) + oe.contract(
            "...di,i->...d", flat_control, self.control_weights.reshape((-1,))
        )
        power = power.at[..., 0].add(self.bias)
        transform = np.zeros((degree + 1, degree + 1), dtype=float)
        for k in range(degree + 1):
            for j in range(k + 1):
                transform[k, j] = comb(k, j) / comb(degree, j)
        bernstein = oe.contract(
            "kj,...j->...k", jnp.asarray(transform, dtype=power.dtype), power
        )
        finite = jnp.all(jnp.isfinite(bernstein), axis=-1)
        return jnp.min(bernstein, axis=-1), jnp.max(bernstein, axis=-1), finite


class LipschitzPathEnvelope(AbstractPathConstraintEnvelope, NonTrainableState):
    """Sample-plus-declared total-time derivative bound over each segment."""

    derivative_bound: Callable[[Array, Array, Array, Any], Array] | Array
    sample_count: int = eqx.field(static=True)

    def __init__(
        self,
        derivative_bound: Callable[[Array, Array, Array, Any], Array] | ArrayLike,
        /,
        *,
        sample_count: int = 3,
        envelope_id: str,
    ):
        if not callable(derivative_bound):
            derivative_bound = jnp.asarray(derivative_bound)
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 2
        ):
            raise ValueError("sample_count must be an integer of at least two.")
        self.derivative_bound = derivative_bound
        self.sample_count = sample_count
        self.envelope_id = envelope_id
        self.conservative = True

    def bounds(self, interpolant, residual, interval, args, /):
        theta = jnp.linspace(0.0, 1.0, self.sample_count)
        time, state, control = interpolant.evaluate(interval, theta)
        case_count = int(np.prod(interpolant.case_shape)) if interpolant.case_shape else 1
        state_cases = state.reshape(
            (case_count, self.sample_count) + interpolant.state_shape
        )
        control_cases = control.reshape(
            (case_count, self.sample_count) + interpolant.control_shape
        )

        def evaluate_case(states, controls):
            return jax.vmap(
                lambda t, x, u: jnp.asarray(residual(t, x, u, args)).reshape(())
            )(time, states, controls)

        values = jax.vmap(evaluate_case)(state_cases, control_cases)
        if callable(self.derivative_bound):

            def derivative_case(states, controls):
                return jax.vmap(
                    lambda t, x, u: jnp.asarray(
                        self.derivative_bound(t, x, u, args)
                    ).reshape(())
                )(time, states, controls)

            derivative = jax.vmap(derivative_case)(state_cases, control_cases)
            bound = jnp.max(derivative, axis=-1)
        else:
            bound = jnp.broadcast_to(
                jnp.asarray(self.derivative_bound).reshape(()), (case_count,)
            )
        spacing = jnp.diff(time)
        variation = bound[:, None] * spacing[None, :]
        adjacent_difference = jnp.diff(values, axis=-1)
        finite = (
            jnp.all(jnp.isfinite(values), axis=-1)
            & jnp.isfinite(bound)
            & (bound >= 0)
            & jnp.all(jnp.isfinite(spacing) & (spacing > 0))
            & jnp.all(jnp.abs(adjacent_difference) <= variation, axis=-1)
        )
        pair_sum = values[:, :-1] + values[:, 1:]
        lower_between = 0.5 * (pair_sum - variation)
        upper_between = 0.5 * (pair_sum + variation)
        lower = jnp.min(jnp.concatenate((values, lower_between), axis=-1), axis=-1)
        upper = jnp.max(jnp.concatenate((values, upper_between), axis=-1), axis=-1)
        return (
            lower.reshape(interpolant.case_shape),
            upper.reshape(interpolant.case_shape),
            finite.reshape(interpolant.case_shape),
        )


class CertifiedPathConstraint(StrictModule, NonTrainableState):
    residual: Callable[[Array, Array, Array, Any], Array]
    envelope: AbstractPathConstraintEnvelope
    constraint_id: str = eqx.field(static=True)

    def __init__(self, residual, envelope, /, *, constraint_id: str):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if not isinstance(envelope, AbstractPathConstraintEnvelope):
            raise TypeError("envelope must be an AbstractPathConstraintEnvelope.")
        self.residual = residual
        self.envelope = envelope
        self.constraint_id = canonical_fingerprint(
            {
                "kind": "certified-path-constraint",
                "user_id": constraint_id,
                "envelope": envelope.envelope_id,
            }
        )

    def __call__(self, time, state, control, args, /):
        return self.residual(time, state, control, args)


class ContinuousPathConstraintCertificate(StrictModule, NonTrainableState):
    lower_bounds: Array
    upper_bounds: Array
    margins: Array
    active: Array
    finite: Array
    interval_certified: Array
    certified: Array
    status: Array
    constraint_ids: tuple[str, ...] = eqx.field(static=True)
    interpolant_id: str = eqx.field(static=True)


def certify_continuous_path_constraints(
    result: DirectCollocationResult | ControlSegmentInterpolant,
    constraints: Sequence[CertifiedPathConstraint],
    /,
    *,
    tolerance: float = 0.0,
    args: Any = None,
) -> ContinuousPathConstraintCertificate:
    """Certify every represented interval or fail closed without relabeling samples."""

    interpolant = (
        direct_collocation_interpolant(result)
        if isinstance(result, DirectCollocationResult)
        else result
    )
    if not isinstance(interpolant, ControlSegmentInterpolant):
        raise TypeError("result must expose a typed ControlSegmentInterpolant.")
    items = tuple(constraints)
    if not items or any(
        not isinstance(value, CertifiedPathConstraint) for value in items
    ):
        raise ValueError(
            "constraints must be a nonempty sequence of CertifiedPathConstraint values."
        )
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    runtime_args = (
        result.compilation.problem.args
        if isinstance(result, DirectCollocationResult) and args is None
        else args
    )
    lower_rows = []
    upper_rows = []
    finite_rows = []
    for constraint in items:
        lowers = []
        uppers = []
        finite = []
        for interval in range(interpolant.interval_count):
            lower, upper, valid = constraint.envelope.bounds(
                interpolant, constraint.residual, interval, runtime_args
            )
            lowers.append(lower)
            uppers.append(upper)
            finite.append(valid & constraint.envelope.conservative)
        lower_rows.append(jnp.stack(tuple(lowers), axis=-1))
        upper_rows.append(jnp.stack(tuple(uppers), axis=-1))
        finite_rows.append(jnp.stack(tuple(finite), axis=-1))
    constraint_axis = len(interpolant.case_shape)
    lower = jnp.stack(tuple(lower_rows), axis=constraint_axis)
    upper = jnp.stack(tuple(upper_rows), axis=constraint_axis)
    finite = jnp.stack(tuple(finite_rows), axis=constraint_axis)
    active = jnp.ones_like(finite, dtype=bool)
    interval_certified = active & finite & (upper <= tolerance_)
    certified = jnp.all(interval_certified, axis=(-2, -1))
    return ContinuousPathConstraintCertificate(
        lower,
        upper,
        tolerance_ - upper,
        active,
        finite,
        interval_certified,
        certified,
        jnp.where(certified, 0, -1).astype(jnp.int32),
        tuple(value.constraint_id for value in items),
        interpolant.interpolant_id,
    )


__all__ = [
    "AbstractPathConstraintEnvelope",
    "AffineBernsteinPathEnvelope",
    "CertifiedPathConstraint",
    "ContinuousPathConstraintCertificate",
    "ControlSegmentInterpolant",
    "LipschitzPathEnvelope",
    "certify_continuous_path_constraints",
    "direct_collocation_interpolant",
]

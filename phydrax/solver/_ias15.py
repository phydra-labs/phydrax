#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike
from scipy.special import roots_jacobi

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


_Acceleration: TypeAlias = Callable[[Array, Array, Array, Any], Array]
# (time, position, velocity, step size, path valid)
_IntervalCarry: TypeAlias = tuple[Array, Array, Array, Array, Array]
# (position, velocity, valid, accepted steps, rejected steps)
_IntervalOutput: TypeAlias = tuple[Array, Array, Array, Array, Array]
# (time, position, velocity, step size, accepted steps, rejected steps, active)
_StepState: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array]


def _radau_coefficients(
    stage_count: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    # P_n + P_(n-1) has the endpoint -1 and the real Jacobi (0, 1) roots.
    # Use the orthogonal-polynomial provider rather than a general companion
    # eigensolve, whose dtype may be complex even for these real roots.
    interior, _ = roots_jacobi(stage_count - 1, 0.0, 1.0)
    nodes = 0.5 * (np.concatenate(([-1.0], interior)) + 1.0)
    velocity_matrix = np.zeros((stage_count, stage_count))
    position_matrix = np.zeros_like(velocity_matrix)
    final_velocity = np.zeros((stage_count,))
    final_position = np.zeros((stage_count,))
    for column in range(stage_count):
        polynomial = np.poly1d([1.0])
        denominator = 1.0
        for other in range(stage_count):
            if other != column:
                polynomial *= np.poly1d([1.0, -nodes[other]])
                denominator *= nodes[column] - nodes[other]
        polynomial /= denominator
        integral = np.polyint(polynomial)
        weighted_integral = np.polyint(np.poly1d([-1.0, 1.0]) * polynomial)
        for row, node in enumerate(nodes):
            velocity_matrix[row, column] = integral(node) - integral(0.0)
            kernel = np.polyint(np.poly1d([-1.0, node]) * polynomial)
            position_matrix[row, column] = kernel(node) - kernel(0.0)
        final_velocity[column] = integral(1.0) - integral(0.0)
        final_position[column] = weighted_integral(1.0) - weighted_integral(0.0)
    return nodes, velocity_matrix, position_matrix, final_velocity, final_position


class IAS15Result(StrictModule):
    times: Array
    position: Array
    velocity: Array
    valid: Array
    status: Array
    accepted_steps: Array
    rejected_steps: Array
    plan_id: str = eqx.field(static=True)


class IAS15Plan(StrictModule, NonTrainableState):
    """Eight-stage, fifteenth-order Gauss–Radau adaptive second-order solver."""

    nodes: Array
    velocity_matrix: Array
    position_matrix: Array
    final_velocity: Array
    final_position: Array
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    maximum_steps_per_interval: int = eqx.field(static=True)
    corrector_iterations: int = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 1.0e-12,
        absolute_tolerance: float = 1.0e-14,
        maximum_steps_per_interval: int = 1024,
        corrector_iterations: int = 8,
        minimum_step: float = 1.0e-15,
    ) -> None:
        if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
            raise ValueError("IAS15 tolerances must be positive.")
        if maximum_steps_per_interval <= 0 or corrector_iterations <= 0:
            raise ValueError("IAS15 iteration capacities must be positive.")
        nodes, velocity, position, final_velocity, final_position = _radau_coefficients(8)
        self.nodes = jnp.asarray(nodes)
        self.velocity_matrix = jnp.asarray(velocity)
        self.position_matrix = jnp.asarray(position)
        self.final_velocity = jnp.asarray(final_velocity)
        self.final_position = jnp.asarray(final_position)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.maximum_steps_per_interval = int(maximum_steps_per_interval)
        self.corrector_iterations = int(corrector_iterations)
        self.minimum_step = float(minimum_step)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ias15-gauss-radau-plan",
                "rtol": float(relative_tolerance),
                "atol": float(absolute_tolerance),
                "maximum_steps": int(maximum_steps_per_interval),
                "corrector_iterations": int(corrector_iterations),
            }
        )

    def _collocation_step(
        self,
        acceleration: _Acceleration,
        time: Array,
        position: Array,
        velocity: Array,
        step_size: Array,
        args: Any,
    ) -> tuple[Array, Array, Array]:
        initial_acceleration = acceleration(time, position, velocity, args)
        accelerations = jnp.broadcast_to(
            initial_acceleration, (self.nodes.size, *initial_acceleration.shape)
        )

        def correct(_: Array, values: Array) -> Array:
            stage_position = (
                position
                + self.nodes[:, None] * step_size * velocity
                + step_size**2 * contract("ij,jd->id", self.position_matrix, values)
            )
            stage_velocity = velocity + step_size * contract(
                "ij,jd->id", self.velocity_matrix, values
            )
            return jax.vmap(
                lambda node, q, v: acceleration(time + node * step_size, q, v, args)
            )(self.nodes, stage_position, stage_velocity)

        accelerations = jax.lax.fori_loop(
            0, self.corrector_iterations, correct, accelerations
        )
        next_position = (
            position
            + step_size * velocity
            + step_size**2 * contract("i,id->d", self.final_position, accelerations)
        )
        next_velocity = velocity + step_size * contract(
            "i,id->d", self.final_velocity, accelerations
        )
        return next_position, next_velocity, accelerations

    def _adaptive_step(
        self,
        acceleration: _Acceleration,
        time: Array,
        position: Array,
        velocity: Array,
        step_size: Array,
        args: Any,
    ) -> tuple[Array, Array, Array, Array, Array]:
        full_position, full_velocity, _ = self._collocation_step(
            acceleration, time, position, velocity, step_size, args
        )
        half_position, half_velocity, _ = self._collocation_step(
            acceleration, time, position, velocity, 0.5 * step_size, args
        )
        refined_position, refined_velocity, _ = self._collocation_step(
            acceleration,
            time + 0.5 * step_size,
            half_position,
            half_velocity,
            0.5 * step_size,
            args,
        )
        error = jnp.maximum(
            jnp.max(jnp.abs(refined_position - full_position)),
            jnp.max(jnp.abs(refined_velocity - full_velocity)),
        )
        scale = self.absolute_tolerance + self.relative_tolerance * jnp.maximum(
            jnp.max(jnp.abs(refined_position)), jnp.max(jnp.abs(refined_velocity))
        )
        normalized = error / scale
        accepted = normalized <= 1.0
        factor = jnp.clip(
            0.9 * jnp.where(normalized > 0.0, normalized ** (-1.0 / 16.0), 2.0), 0.2, 2.0
        )
        return refined_position, refined_velocity, accepted, factor, normalized

    def solve(
        self,
        acceleration: _Acceleration,
        initial_position: ArrayLike,
        initial_velocity: ArrayLike,
        save_times: ArrayLike,
        args: Any = None,
        /,
    ) -> IAS15Result:
        if not callable(acceleration):
            raise TypeError("acceleration must be callable.")
        position0 = jnp.asarray(initial_position)
        velocity0 = jnp.asarray(initial_velocity, dtype=position0.dtype)
        times = jnp.asarray(save_times, dtype=position0.dtype)
        if position0.ndim != 1 or velocity0.shape != position0.shape:
            raise ValueError("IAS15 position and velocity must be matching vectors.")
        if times.ndim != 1 or times.size < 2:
            raise ValueError("IAS15 save_times must be a vector with at least two nodes.")

        def interval(
            carry: _IntervalCarry, target: Array
        ) -> tuple[_IntervalCarry, _IntervalOutput]:
            time, position, velocity, initial_step, path_valid = carry

            def condition(state: _StepState) -> Array:
                current, _, _, _, steps, _, active = state
                return (
                    active
                    & (current < target)
                    & (steps < self.maximum_steps_per_interval)
                )

            def body(state: _StepState) -> _StepState:
                current, q, v, step_size, steps, rejected, active = state
                proposed = jnp.minimum(step_size, target - current)
                next_q, next_v, accepted, factor, _ = self._adaptive_step(
                    acceleration, current, q, v, proposed, args
                )
                finite = jnp.all(jnp.isfinite(next_q)) & jnp.all(jnp.isfinite(next_v))
                accepted = accepted & finite
                return (
                    jnp.where(accepted, current + proposed, current),
                    jnp.where(accepted, next_q, q),
                    jnp.where(accepted, next_v, v),
                    jnp.maximum(proposed * factor, self.minimum_step),
                    steps + accepted.astype(jnp.int32),
                    rejected + (~accepted).astype(jnp.int32),
                    active & (proposed >= self.minimum_step),
                )

            initial_state = (
                time,
                position,
                velocity,
                initial_step,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                path_valid,
            )
            final = jax.lax.while_loop(condition, body, initial_state)
            reached = final[0] >= target
            valid = final[6] & reached
            return (final[0], final[1], final[2], final[3], valid), (
                final[1],
                final[2],
                valid,
                final[4],
                final[5],
            )

        initial_step = times[1] - times[0]
        (_, _, _, _, completed), outputs = jax.lax.scan(
            interval,
            (times[0], position0, velocity0, initial_step, jnp.asarray(True)),
            times[1:],
        )
        positions = jnp.concatenate((position0[None], outputs[0]), axis=0)
        velocities = jnp.concatenate((velocity0[None], outputs[1]), axis=0)
        valid = jnp.concatenate((jnp.asarray(True)[None], outputs[2]))
        status = jnp.where(valid, 0, 1).astype(jnp.int32)
        accepted = jnp.concatenate((jnp.asarray(0, dtype=jnp.int32)[None], outputs[3]))
        rejected = jnp.concatenate((jnp.asarray(0, dtype=jnp.int32)[None], outputs[4]))
        return IAS15Result(
            times, positions, velocities, valid, status, accepted, rejected, self.plan_id
        )


__all__ = ["IAS15Plan", "IAS15Result"]

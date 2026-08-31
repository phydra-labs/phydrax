#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.vortex._wake import VortexWakePlan, VortexWakeState
from ..operators.integral.vortex._filament3d import (
    PreparedFilamentVelocity3D,
    regularized_filament_velocity_3d,
)
from ._vortex_lattice import SteadyVortexLatticePlan, VortexLatticeResult


class UVLMState(StrictModule):
    wake: VortexWakeState
    bound_circulation: Array
    time: Array
    step_index: Array


class UVLMStepResult(StrictModule):
    state: UVLMState
    bound: VortexLatticeResult
    emitted_circulation: Array
    wake_capacity_remaining: Array
    wake_conservation_residual: Array
    successful: Array
    method_id: str = eqx.field(static=True)


class UnsteadyVortexLatticePlan(StrictModule, NonTrainableState):
    """Accepted-step fixed-capacity UVLM with explicit wake emission."""

    bound: SteadyVortexLatticePlan
    wake: VortexWakePlan
    method_id: str = eqx.field(static=True)

    def __init__(self, bound: SteadyVortexLatticePlan, wake: VortexWakePlan, /):
        if not isinstance(bound, SteadyVortexLatticePlan):
            raise TypeError("bound must be SteadyVortexLatticePlan.")
        if not isinstance(wake, VortexWakePlan):
            raise TypeError("wake must be VortexWakePlan.")
        if wake.source_count != bound.surface.panel_count:
            raise ValueError("Wake source count must match lifting panels.")
        self.bound = bound
        self.wake = wake
        self.method_id = canonical_fingerprint(
            {
                "kind": "unsteady-vortex-lattice",
                "bound": bound.solver_id,
                "wake": wake.plan_id,
            }
        )

    def initialize(self, /) -> UVLMState:
        circulation = jnp.zeros(
            (self.bound.surface.panel_count,),
            dtype=self.bound.surface.control_point.dtype,
        )
        wake = self.wake.initialize(circulation)
        return UVLMState(
            wake,
            circulation,
            jnp.asarray(0.0, dtype=circulation.dtype),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def _wake_velocity(self, state: VortexWakeState, targets: Array, /) -> Array:
        if state.start.shape[0] == 0:
            return jnp.zeros_like(targets)
        return PreparedFilamentVelocity3D(state.as_filaments()).evaluate(targets).velocity

    def _bound_velocity(self, circulation: Array, targets: Array, /) -> Array:
        surface = self.bound.surface
        far_left = (
            surface.bound_start + self.bound.wake_length * self.bound.wake_direction
        )
        far_right = surface.bound_end + self.bound.wake_length * self.bound.wake_direction
        starts = jnp.stack(
            (far_left, surface.bound_start, surface.bound_end), axis=1
        ).reshape((-1, 3))
        ends = jnp.stack(
            (surface.bound_start, surface.bound_end, far_right), axis=1
        ).reshape((-1, 3))
        gamma = jnp.repeat(circulation, 3)
        core = jnp.full_like(gamma, self.bound.core_radius)
        return regularized_filament_velocity_3d(targets, starts, ends, gamma, core)

    def step(
        self,
        state: UVLMState,
        freestream_velocity: ArrayLike,
        time_step: ArrayLike,
        /,
    ) -> UVLMStepResult:
        freestream = jnp.asarray(
            freestream_velocity, dtype=self.bound.surface.control_point.dtype
        )
        if freestream.shape == (3,):
            freestream_control = jnp.broadcast_to(
                freestream, (self.bound.surface.panel_count, 3)
            )
        elif freestream.shape == (self.bound.surface.panel_count, 3):
            freestream_control = freestream
        else:
            raise ValueError("freestream_velocity has an invalid UVLM shape.")
        dt = jnp.asarray(time_step, dtype=freestream_control.dtype)
        if dt.shape != ():
            raise ValueError("time_step must be scalar.")
        wake_induced = self._wake_velocity(state.wake, self.bound.surface.control_point)
        bound_result = self.bound.solve(freestream_control + wake_induced)
        transition = self.wake.shed(
            state.wake,
            self.bound.surface.trailing_start,
            self.bound.surface.trailing_end,
            bound_result.circulation,
            dt,
        )
        accepted_wake = transition.accepted
        start_velocity = (
            self._wake_velocity(accepted_wake, accepted_wake.start)
            + self._bound_velocity(bound_result.circulation, accepted_wake.start)
            + jnp.broadcast_to(freestream_control[0], accepted_wake.start.shape)
        )
        end_velocity = (
            self._wake_velocity(accepted_wake, accepted_wake.end)
            + self._bound_velocity(bound_result.circulation, accepted_wake.end)
            + jnp.broadcast_to(freestream_control[0], accepted_wake.end.shape)
        )
        active = accepted_wake.active[:, None]
        convected = VortexWakeState(
            accepted_wake.start + jnp.where(active, dt * start_velocity, 0.0),
            accepted_wake.end + jnp.where(active, dt * end_velocity, 0.0),
            accepted_wake.circulation,
            accepted_wake.core_radius,
            accepted_wake.age,
            accepted_wake.active,
            accepted_wake.bound_circulation,
            accepted_wake.step_index,
        )
        successful = (
            bound_result.successful
            & transition.successful
            & jnp.isfinite(dt)
            & (dt > 0.0)
        )
        next_state = UVLMState(
            convected,
            bound_result.circulation,
            state.time + dt,
            state.step_index + 1,
        )
        remaining = self.wake.segment_capacity - jnp.sum(
            convected.active, dtype=jnp.int32
        )
        return UVLMStepResult(
            next_state,
            bound_result,
            transition.emitted_circulation,
            remaining,
            transition.circulation_residual,
            successful,
            self.method_id,
        )


__all__ = ["UVLMState", "UVLMStepResult", "UnsteadyVortexLatticePlan"]

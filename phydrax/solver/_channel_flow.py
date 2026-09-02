#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization.spectral import (
    ChannelStokesDiagnostics,
    PreparedChannelStokesSolver,
)
from ..equations._channel_flow import CompiledChannelFlowDynamics
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._temporal_method import TemporalMethodCapabilities


CHANNEL_FLOW_SUCCESS = 0
CHANNEL_FLOW_INITIAL_CONSTRAINT = -1
CHANNEL_FLOW_STOKES_FAILURE = -2


class ChannelSBDF2Method(StrictModule, NonTrainableState):
    """Fixed-step semi-implicit BDF2 with backward-Euler initialization."""

    capabilities: TemporalMethodCapabilities
    method_id: str = eqx.field(static=True)

    def __init__(self):
        identifier = canonical_fingerprint(
            {
                "kind": "channel-sbdf2-method-v2",
                "startup": "backward-euler",
                "failure_policy": "atomic-retain-complete-history",
            }
        )
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("additive-ode",),
            method_class="bdf",
            order=2,
            adaptive=False,
            history_depth=2,
            stage_abscissae=(1.0,),
            causal_stage_extent=1.0,
            noise_requirement="none",
            method_id=identifier,
        )
        self.method_id = identifier

    def prepare(
        self,
        dynamics: CompiledChannelFlowDynamics,
        step_size: ArrayLike,
        /,
    ) -> PreparedChannelSBDF2Method:
        if not isinstance(dynamics, CompiledChannelFlowDynamics):
            raise TypeError("dynamics must be CompiledChannelFlowDynamics.")
        raw_step = np.asarray(step_size)
        if np.iscomplexobj(raw_step):
            raise TypeError("Channel SBDF2 step_size must be real.")
        if raw_step.shape != () or not np.isfinite(raw_step) or float(raw_step) <= 0.0:
            raise ValueError("Channel SBDF2 step_size must be finite and positive.")
        step = float(raw_step)
        return PreparedChannelSBDF2Method(
            self,
            dynamics,
            step,
            dynamics.prepare_stokes(1.0 / step),
            dynamics.prepare_stokes(3.0 / (2.0 * step)),
        )


class ChannelSBDF2State(StrictModule):
    """Complete immutable restart state for startup and multistep transitions."""

    previous_velocity: Array
    current_velocity: Array
    previous_nonlinear_rhs: Array
    current_nonlinear_rhs: Array
    current_pressure: Array
    pressure_gradient: Array
    history_count: Array

    def __init__(
        self,
        previous_velocity: ArrayLike,
        current_velocity: ArrayLike,
        previous_nonlinear_rhs: ArrayLike,
        current_nonlinear_rhs: ArrayLike,
        current_pressure: ArrayLike,
        pressure_gradient: ArrayLike,
        history_count: ArrayLike,
        /,
    ):
        previous = jnp.asarray(previous_velocity)
        current = jnp.asarray(current_velocity)
        previous_rhs = jnp.asarray(previous_nonlinear_rhs)
        current_rhs = jnp.asarray(current_nonlinear_rhs)
        pressure = jnp.asarray(current_pressure)
        gradient = jnp.asarray(pressure_gradient)
        count = jnp.asarray(history_count)
        if (
            current.ndim < 1
            or current.shape[-1] != 3
            or previous.shape != current.shape
            or previous_rhs.shape != current.shape
            or current_rhs.shape != current.shape
            or pressure.shape != current.shape[:-1]
            or gradient.shape != (2,)
        ):
            raise ValueError("ChannelSBDF2State fields have incompatible shapes.")
        if not all(
            jnp.issubdtype(value.dtype, jnp.inexact)
            for value in (
                previous,
                current,
                previous_rhs,
                current_rhs,
                pressure,
                gradient,
            )
        ):
            raise TypeError("ChannelSBDF2State fields must have inexact dtypes.")
        if not all(
            jnp.issubdtype(value.dtype, jnp.complexfloating)
            for value in (previous, current, previous_rhs, current_rhs, pressure)
        ):
            raise TypeError(
                "Channel velocity, nonlinear RHS, and pressure history must be complex."
            )
        if jnp.issubdtype(gradient.dtype, jnp.complexfloating):
            raise TypeError("Channel pressure_gradient history must be real.")
        if count.shape != () or not jnp.issubdtype(count.dtype, jnp.integer):
            raise TypeError("history_count must be one integer scalar array.")
        count = eqx.error_if(
            count,
            count < 0,
            "Channel SBDF2 history_count must be nonnegative.",
        )
        self.previous_velocity = previous
        self.current_velocity = current
        self.previous_nonlinear_rhs = previous_rhs
        self.current_nonlinear_rhs = current_rhs
        self.current_pressure = pressure
        self.pressure_gradient = gradient
        self.history_count = count


class _ChannelSBDF2Transition(StrictModule):
    fixed_step: FixedStepResult
    diagnostics: ChannelStokesDiagnostics
    status: Array


class PreparedChannelSBDF2Method(AbstractFixedStepMethod):
    """Channel SBDF2 bound to one dynamics compilation and exact time step."""

    dynamics: CompiledChannelFlowDynamics
    backward_euler: PreparedChannelStokesSolver
    bdf2: PreparedChannelStokesSolver
    capabilities: TemporalMethodCapabilities
    _required_step_size: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: ChannelSBDF2Method,
        dynamics: CompiledChannelFlowDynamics,
        step_size: float,
        backward_euler: PreparedChannelStokesSolver,
        bdf2: PreparedChannelStokesSolver,
        /,
    ):
        self.dynamics = dynamics
        self.backward_euler = backward_euler
        self.bdf2 = bdf2
        self.capabilities = method.capabilities
        self._required_step_size = float(step_size)
        self.method_id = canonical_fingerprint(
            {
                "kind": "prepared-channel-sbdf2-method-v1",
                "method": method.method_id,
                "dynamics": dynamics.compilation_id,
                "step_size": self.required_step_size,
                "backward_euler": backward_euler.prepared_id,
                "bdf2": bdf2.prepared_id,
            }
        )

    @property
    def required_step_size(self) -> float:
        return self._required_step_size

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def initialize(
        self,
        initial_state: ArrayLike,
        time: ArrayLike,
        args: Any = None,
        /,
    ) -> ChannelSBDF2State:
        velocity = self.dynamics.admissible_modes(initial_state)
        nonlinear = self.dynamics.nonlinear(jnp.asarray(time), velocity, args)
        pressure = jnp.zeros(
            self.dynamics.discretization.modal_shape,
            dtype=velocity.dtype,
        )
        gradient = jnp.zeros((2,), dtype=velocity.real.dtype)
        return ChannelSBDF2State(
            velocity,
            velocity,
            nonlinear,
            nonlinear,
            pressure,
            gradient,
            jnp.asarray(0, dtype=jnp.int32),
        )

    def _solve_arrays(
        self,
        state: ChannelSBDF2State,
        step: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        def startup(_):
            solved = self.backward_euler.solve(
                state.current_velocity / step + state.current_nonlinear_rhs
            )
            diagnostics = solved.diagnostics
            return (
                solved.velocity,
                solved.pressure,
                solved.pressure_gradient,
                diagnostics.momentum_constraint_residual,
                diagnostics.divergence_norm,
                diagnostics.wall_residual,
                diagnostics.pressure_gauge_residual,
                diagnostics.bulk_velocity,
                diagnostics.failed,
            )

        def multistep(_):
            right_hand_side = (
                (4.0 * state.current_velocity - state.previous_velocity) / (2.0 * step)
                + 2.0 * state.current_nonlinear_rhs
                - state.previous_nonlinear_rhs
            )
            solved = self.bdf2.solve(right_hand_side)
            diagnostics = solved.diagnostics
            return (
                solved.velocity,
                solved.pressure,
                solved.pressure_gradient,
                diagnostics.momentum_constraint_residual,
                diagnostics.divergence_norm,
                diagnostics.wall_residual,
                diagnostics.pressure_gauge_residual,
                diagnostics.bulk_velocity,
                diagnostics.failed,
            )

        return jax.lax.cond(
            state.history_count == 0,
            startup,
            multistep,
            operand=None,
        )

    def step_with_diagnostics(
        self,
        step_index: Array,
        time: Array,
        state: ChannelSBDF2State,
        step_size: Array,
        args: Any,
        /,
    ) -> _ChannelSBDF2Transition:
        del step_index
        if not isinstance(state, ChannelSBDF2State):
            raise TypeError("state must be a ChannelSBDF2State.")
        if state.current_velocity.shape != self.dynamics.state_shape:
            raise ValueError("Channel SBDF2 state shape does not match its dynamics.")
        step = jnp.asarray(
            step_size,
            dtype=state.current_velocity.real.dtype,
        ).reshape(())
        declared = jnp.asarray(self.required_step_size, dtype=step.dtype)
        step = eqx.error_if(
            step,
            ~(jnp.isfinite(step) & (step == declared)),
            "Channel SBDF2 step_size must exactly equal its prepared value.",
        )
        start = jnp.asarray(time, dtype=step.dtype).reshape(())
        incoming = self.dynamics.state_diagnostics(state.current_velocity)
        (
            velocity,
            pressure,
            pressure_gradient,
            momentum_residual,
            divergence_norm,
            wall_residual,
            pressure_gauge_residual,
            bulk_velocity,
            stokes_failed,
        ) = self._solve_arrays(state, step)
        successful = incoming.valid & ~stokes_failed
        nonlinear = jax.lax.cond(
            successful,
            lambda _: self.dynamics.nonlinear(start + step, velocity, args),
            lambda _: state.current_nonlinear_rhs,
            operand=None,
        )
        candidate = ChannelSBDF2State(
            state.current_velocity,
            velocity,
            state.current_nonlinear_rhs,
            nonlinear,
            pressure,
            pressure_gradient,
            state.history_count + jnp.asarray(1, dtype=state.history_count.dtype),
        )
        accepted = tree_where(successful, candidate, state)
        diagnostics = ChannelStokesDiagnostics(
            momentum_constraint_residual=momentum_residual,
            divergence_norm=divergence_norm,
            wall_residual=wall_residual,
            pressure_gauge_residual=pressure_gauge_residual,
            bulk_velocity=bulk_velocity,
            failed=stokes_failed,
        )
        invalid_status = jnp.where(
            state.history_count == 0,
            jnp.asarray(CHANNEL_FLOW_INITIAL_CONSTRAINT, dtype=jnp.int32),
            jnp.asarray(CHANNEL_FLOW_STOKES_FAILURE, dtype=jnp.int32),
        )
        status = jnp.where(
            incoming.valid,
            jnp.where(
                stokes_failed,
                jnp.asarray(CHANNEL_FLOW_STOKES_FAILURE, dtype=jnp.int32),
                jnp.asarray(CHANNEL_FLOW_SUCCESS, dtype=jnp.int32),
            ),
            invalid_status,
        )
        fixed_step = FixedStepResult(
            candidate_state=candidate,
            accepted_state=accepted,
            successful=successful,
            residual=momentum_residual,
            iterations=jnp.asarray(1, dtype=jnp.int32),
            work=jnp.asarray(1, dtype=jnp.int32),
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.zeros(
                (),
                dtype=state.current_velocity.real.dtype,
            ),
        )
        return _ChannelSBDF2Transition(fixed_step, diagnostics, status)

    def step(
        self,
        step_index: Array,
        time: Array,
        state: ChannelSBDF2State,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        return self.step_with_diagnostics(
            step_index,
            time,
            state,
            step_size,
            args,
        ).fixed_step


class ChannelFlowDiagnosticsHistory(StrictModule):
    stokes_residual: Array
    divergence_norm: Array
    wall_residual: Array
    pressure_gauge_residual: Array
    bulk_velocity: Array
    kinetic_energy: Array
    valid: Array
    status: Array


class ChannelFlowSolution(StrictModule):
    """Accepted velocity, pressure, forcing, and constraint evidence at every step."""

    times: Array
    velocity: Array
    pressure: Array
    pressure_gradient: Array
    diagnostics: ChannelFlowDiagnosticsHistory
    method: ChannelSBDF2Method
    dynamics: CompiledChannelFlowDynamics
    solver_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.all(self.diagnostics.valid) & jnp.all(
            self.diagnostics.status == CHANNEL_FLOW_SUCCESS
        )


def solve_channel_sbdf2(
    dynamics: CompiledChannelFlowDynamics,
    initial_state: ArrayLike,
    times: ArrayLike,
    /,
    *,
    method: ChannelSBDF2Method | None = None,
    args: Any = None,
) -> ChannelFlowSolution:
    """Integrate by scanning the same prepared atomic transition used for restart."""
    if not isinstance(dynamics, CompiledChannelFlowDynamics):
        raise TypeError("dynamics must be CompiledChannelFlowDynamics.")
    selected = ChannelSBDF2Method() if method is None else method
    if not isinstance(selected, ChannelSBDF2Method):
        raise TypeError("method must be ChannelSBDF2Method or None.")
    raw_saved = jnp.asarray(times)
    if jnp.iscomplexobj(raw_saved):
        raise TypeError("times must be real.")
    saved = raw_saved.astype(jnp.result_type(raw_saved.dtype, jnp.float32))
    saved_host = np.asarray(saved)
    if (
        saved.ndim != 1
        or saved.size < 2
        or np.any(~np.isfinite(saved_host))
        or np.any(np.diff(saved_host) <= 0.0)
    ):
        raise ValueError("times must be a finite strictly increasing rank-one grid.")
    durations = np.diff(saved_host)
    if not np.allclose(durations, durations[0], rtol=1e-12, atol=1e-14):
        raise ValueError("ChannelSBDF2Method requires one fixed time step.")

    prepared = selected.prepare(dynamics, float(durations[0]))
    initial_history = prepared.initialize(initial_state, saved[0], args)
    initial = initial_history.current_velocity
    initial_evidence = dynamics.state_diagnostics(initial)
    initial_valid = initial_evidence.valid
    initial_status = jnp.where(
        initial_valid,
        jnp.asarray(CHANNEL_FLOW_SUCCESS, dtype=jnp.int32),
        jnp.asarray(CHANNEL_FLOW_INITIAL_CONSTRAINT, dtype=jnp.int32),
    )
    step = jnp.asarray(
        prepared.required_step_size,
        dtype=initial.real.dtype,
    )
    starts = saved[:-1]
    indices = jnp.arange(int(saved.size) - 1, dtype=jnp.int32)

    def advance(
        carry: tuple[ChannelSBDF2State, Array, Array],
        data: tuple[Array, Array],
    ):
        state, cumulative_valid, latched_status = carry
        step_index, time = data
        transition = prepared.step_with_diagnostics(
            step_index,
            time,
            state,
            step,
            args,
        )
        result = transition.fixed_step
        following = tree_where(cumulative_valid, result.accepted_state, state)
        valid = cumulative_valid & result.successful
        status = jnp.where(
            cumulative_valid,
            transition.status,
            latched_status,
        )
        reported_gradient = jnp.where(
            (following.history_count == 0) & ~valid,
            jnp.full((2,), jnp.nan, dtype=following.pressure_gradient.dtype),
            following.pressure_gradient,
        )
        diagnostics = transition.diagnostics
        output = (
            following.current_velocity,
            following.current_pressure,
            reported_gradient,
            diagnostics.momentum_constraint_residual,
            diagnostics.divergence_norm,
            diagnostics.wall_residual,
            diagnostics.pressure_gauge_residual,
            diagnostics.bulk_velocity,
            dynamics.state_diagnostics(following.current_velocity).kinetic_energy,
            valid,
            status,
        )
        return (following, valid, status), output

    _, advanced = jax.lax.scan(
        advance,
        (initial_history, initial_valid, initial_status),
        (indices, starts),
    )
    initial_output_gradient = jnp.full(
        (2,),
        jnp.nan,
        dtype=initial.real.dtype,
    )
    diagnostics = ChannelFlowDiagnosticsHistory(
        stokes_residual=jnp.concatenate(
            (jnp.asarray([jnp.nan], dtype=advanced[3].dtype), advanced[3]),
            axis=0,
        ),
        divergence_norm=jnp.concatenate(
            (initial_evidence.divergence_norm[None], advanced[4]),
            axis=0,
        ),
        wall_residual=jnp.concatenate(
            (initial_evidence.wall_residual[None], advanced[5]),
            axis=0,
        ),
        pressure_gauge_residual=jnp.concatenate(
            (jnp.asarray([jnp.nan], dtype=advanced[6].dtype), advanced[6]),
            axis=0,
        ),
        bulk_velocity=jnp.concatenate(
            (jnp.full((1, 2), jnp.nan, dtype=initial.real.dtype), advanced[7]),
            axis=0,
        ),
        kinetic_energy=jnp.concatenate(
            (initial_evidence.kinetic_energy[None], advanced[8]),
            axis=0,
        ),
        valid=jnp.concatenate((initial_valid[None], advanced[9]), axis=0),
        status=jnp.concatenate((initial_status[None], advanced[10]), axis=0),
    )
    return ChannelFlowSolution(
        times=saved,
        velocity=jnp.concatenate((initial[None, ...], advanced[0]), axis=0),
        pressure=jnp.concatenate(
            (initial_history.current_pressure[None, ...], advanced[1]),
            axis=0,
        ),
        pressure_gradient=jnp.concatenate(
            (initial_output_gradient[None, ...], advanced[2]),
            axis=0,
        ),
        diagnostics=diagnostics,
        method=selected,
        dynamics=dynamics,
        solver_id=canonical_fingerprint(
            {
                "kind": "channel-sbdf2-solve-v3",
                "method": prepared.method_id,
                "dynamics": dynamics.compilation_id,
                "step_size": prepared.required_step_size,
                "steps": int(saved.size) - 1,
                "failure_policy": "atomic-retain-complete-history",
            }
        ),
    )


__all__ = [
    "CHANNEL_FLOW_INITIAL_CONSTRAINT",
    "CHANNEL_FLOW_STOKES_FAILURE",
    "CHANNEL_FLOW_SUCCESS",
    "ChannelFlowDiagnosticsHistory",
    "ChannelFlowSolution",
    "ChannelSBDF2Method",
    "ChannelSBDF2State",
    "PreparedChannelSBDF2Method",
    "solve_channel_sbdf2",
]

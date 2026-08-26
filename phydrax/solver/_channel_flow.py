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
from ..equations._channel_flow import CompiledChannelFlowDynamics
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
                "kind": "channel-sbdf2-method-v1",
                "startup": "backward-euler",
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
    """Integrate one constrained channel state on a uniform increasing time grid."""
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

    initial = dynamics.admissible_modes(initial_state)
    initial_evidence = dynamics.state_diagnostics(initial)
    initial_valid = initial_evidence.valid
    initial_status = jnp.where(
        initial_valid,
        jnp.asarray(CHANNEL_FLOW_SUCCESS, dtype=jnp.int32),
        jnp.asarray(CHANNEL_FLOW_INITIAL_CONSTRAINT, dtype=jnp.int32),
    )
    initial_pressure = jnp.zeros(dynamics.discretization.modal_shape, dtype=initial.dtype)
    initial_gradient = jnp.full((2,), jnp.nan, dtype=initial.real.dtype)
    step = jnp.asarray(durations[0], dtype=initial.real.dtype)
    nonlinear_initial = dynamics.nonlinear(saved[0], initial, args)
    backward_euler = dynamics.prepare_stokes(1.0 / step)
    first_candidate = backward_euler.solve(initial / step + nonlinear_initial)
    first_valid = initial_valid & first_candidate.successful
    first_velocity = jnp.where(first_valid, first_candidate.velocity, initial)
    first_pressure = jnp.where(
        first_valid, first_candidate.pressure, initial_pressure
    )
    first_gradient = jnp.where(
        first_valid, first_candidate.pressure_gradient, initial_gradient
    )
    first_status = jnp.where(
        initial_valid,
        jnp.where(
            first_candidate.successful,
            jnp.asarray(CHANNEL_FLOW_SUCCESS, dtype=jnp.int32),
            jnp.asarray(CHANNEL_FLOW_STOKES_FAILURE, dtype=jnp.int32),
        ),
        initial_status,
    )
    first_energy = dynamics.state_diagnostics(first_velocity).kinetic_energy
    nonlinear_first = dynamics.nonlinear(saved[1], first_velocity, args)
    first_diagnostics = first_candidate.diagnostics
    first_output = (
        first_velocity,
        first_pressure,
        first_gradient,
        first_diagnostics.momentum_constraint_residual,
        first_diagnostics.divergence_norm,
        first_diagnostics.wall_residual,
        first_diagnostics.pressure_gauge_residual,
        first_diagnostics.bulk_velocity,
        first_energy,
        first_valid,
        first_status,
    )

    bdf2 = dynamics.prepare_stokes(3.0 / (2.0 * step))

    def advance(carry, time):
        (
            previous,
            current,
            previous_nonlinear,
            current_nonlinear,
            current_pressure,
            current_gradient,
            cumulative_valid,
            latched_status,
        ) = carry
        right_hand_side = (
            (4.0 * current - previous) / (2.0 * step)
            + 2.0 * current_nonlinear
            - previous_nonlinear
        )
        candidate = bdf2.solve(right_hand_side)
        accepted = cumulative_valid & candidate.successful
        following_velocity = jnp.where(accepted, candidate.velocity, current)
        following_pressure = jnp.where(accepted, candidate.pressure, current_pressure)
        following_gradient = jnp.where(
            accepted, candidate.pressure_gradient, current_gradient
        )
        following_nonlinear = jax.lax.cond(
            accepted,
            lambda _: dynamics.nonlinear(time + step, candidate.velocity, args),
            lambda _: current_nonlinear,
            operand=None,
        )
        next_status = jnp.where(
            cumulative_valid,
            jnp.where(
                candidate.successful,
                jnp.asarray(CHANNEL_FLOW_SUCCESS, dtype=jnp.int32),
                jnp.asarray(CHANNEL_FLOW_STOKES_FAILURE, dtype=jnp.int32),
            ),
            latched_status,
        )
        next_carry = (
            jnp.where(accepted, current, previous),
            following_velocity,
            jnp.where(accepted, current_nonlinear, previous_nonlinear),
            following_nonlinear,
            following_pressure,
            following_gradient,
            accepted,
            next_status,
        )
        diagnostics = candidate.diagnostics
        output = (
            following_velocity,
            following_pressure,
            following_gradient,
            diagnostics.momentum_constraint_residual,
            diagnostics.divergence_norm,
            diagnostics.wall_residual,
            diagnostics.pressure_gauge_residual,
            diagnostics.bulk_velocity,
            dynamics.state_diagnostics(following_velocity).kinetic_energy,
            accepted,
            next_status,
        )
        return next_carry, output

    if int(saved.size) > 2:
        _, scanned = jax.lax.scan(
            advance,
            (
                initial,
                first_velocity,
                nonlinear_initial,
                nonlinear_first,
                first_pressure,
                first_gradient,
                first_valid,
                first_status,
            ),
            saved[1:-1],
        )
        advanced = tuple(
            jnp.concatenate((first_output[index][None, ...], scanned[index]), axis=0)
            for index in range(len(first_output))
        )
    else:
        advanced = tuple(value[None, ...] for value in first_output)

    diagnostics = ChannelFlowDiagnosticsHistory(
        stokes_residual=jnp.concatenate(
            (jnp.asarray([jnp.nan], dtype=advanced[3].dtype), advanced[3]), axis=0
        ),
        divergence_norm=jnp.concatenate(
            (initial_evidence.divergence_norm[None], advanced[4]), axis=0
        ),
        wall_residual=jnp.concatenate(
            (initial_evidence.wall_residual[None], advanced[5]), axis=0
        ),
        pressure_gauge_residual=jnp.concatenate(
            (jnp.asarray([jnp.nan], dtype=advanced[6].dtype), advanced[6]), axis=0
        ),
        bulk_velocity=jnp.concatenate(
            (jnp.full((1, 2), jnp.nan, dtype=initial.real.dtype), advanced[7]),
            axis=0,
        ),
        kinetic_energy=jnp.concatenate(
            (initial_evidence.kinetic_energy[None], advanced[8]), axis=0
        ),
        valid=jnp.concatenate((initial_valid[None], advanced[9]), axis=0),
        status=jnp.concatenate((initial_status[None], advanced[10]), axis=0),
    )
    return ChannelFlowSolution(
        times=saved,
        velocity=jnp.concatenate((initial[None, ...], advanced[0]), axis=0),
        pressure=jnp.concatenate((initial_pressure[None, ...], advanced[1]), axis=0),
        pressure_gradient=jnp.concatenate(
            (initial_gradient[None, ...], advanced[2]), axis=0
        ),
        diagnostics=diagnostics,
        method=selected,
        dynamics=dynamics,
        solver_id=canonical_fingerprint(
            {
                "kind": "channel-sbdf2-solve-v2",
                "method": selected.method_id,
                "dynamics": dynamics.compilation_id,
                "step_size": float(durations[0]),
                "steps": int(saved.size) - 1,
                "failure_policy": "retain-last-accepted",
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
    "solve_channel_sbdf2",
]

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
    valid: Array


class ChannelFlowSolution(StrictModule):
    """Velocity, pressure, mean forcing, and constraint evidence at every step."""

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
        return jnp.all(self.diagnostics.valid)


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
    step = jnp.asarray(durations[0], dtype=initial.real.dtype)
    nonlinear_initial = dynamics.nonlinear(saved[0], initial, args)
    backward_euler = dynamics.prepare_stokes(1.0 / step)
    first = backward_euler.solve(initial / step + nonlinear_initial)
    nonlinear_first = dynamics.nonlinear(saved[1], first.velocity, args)

    def advance(carry, time):
        previous, current, previous_nonlinear, current_nonlinear = carry
        right_hand_side = (
            (4.0 * current - previous) / (2.0 * step)
            + 2.0 * current_nonlinear
            - previous_nonlinear
        )
        following = bdf2.solve(right_hand_side)
        following_nonlinear = dynamics.nonlinear(time + step, following.velocity, args)
        next_carry = (
            current,
            following.velocity,
            current_nonlinear,
            following_nonlinear,
        )
        output = _stokes_output(following)
        return next_carry, output

    bdf2 = dynamics.prepare_stokes(3.0 / (2.0 * step))
    first_output = _stokes_output(first)
    if int(saved.size) > 2:
        _, scanned = jax.lax.scan(
            advance,
            (initial, first.velocity, nonlinear_initial, nonlinear_first),
            saved[1:-1],
        )
        advanced_velocity = jnp.concatenate(
            (first.velocity[None, ...], scanned[0]), axis=0
        )
        advanced_pressure = jnp.concatenate(
            (first.pressure[None, ...], scanned[1]), axis=0
        )
        advanced_gradient = jnp.concatenate(
            (first.pressure_gradient[None, ...], scanned[2]), axis=0
        )
        diagnostic_values = tuple(
            jnp.concatenate(
                (first_output[index + 3][None, ...], scanned[index + 3]),
                axis=0,
            )
            for index in range(6)
        )
    else:
        advanced_velocity = first.velocity[None, ...]
        advanced_pressure = first.pressure[None, ...]
        advanced_gradient = first.pressure_gradient[None, ...]
        diagnostic_values = tuple(
            first_output[index + 3][None, ...] for index in range(6)
        )
    initial_pressure = jnp.zeros(dynamics.discretization.modal_shape, dtype=initial.dtype)
    initial_gradient = jnp.full(
        (2,),
        jnp.nan,
        dtype=initial.real.dtype,
    )
    initial_valid = jnp.all(jnp.isfinite(initial))
    diagnostics = ChannelFlowDiagnosticsHistory(
        stokes_residual=jnp.concatenate(
            (
                jnp.asarray([jnp.nan], dtype=diagnostic_values[0].dtype),
                diagnostic_values[0],
            ),
            axis=0,
        ),
        divergence_norm=jnp.concatenate(
            (
                jnp.asarray([jnp.nan], dtype=diagnostic_values[1].dtype),
                diagnostic_values[1],
            ),
            axis=0,
        ),
        wall_residual=jnp.concatenate(
            (
                jnp.asarray([jnp.nan], dtype=diagnostic_values[2].dtype),
                diagnostic_values[2],
            ),
            axis=0,
        ),
        pressure_gauge_residual=jnp.concatenate(
            (
                jnp.asarray([jnp.nan], dtype=diagnostic_values[3].dtype),
                diagnostic_values[3],
            ),
            axis=0,
        ),
        bulk_velocity=jnp.concatenate(
            (jnp.full((1, 2), jnp.nan, dtype=initial.dtype), diagnostic_values[4]),
            axis=0,
        ),
        valid=jnp.concatenate((initial_valid[None], diagnostic_values[5]), axis=0),
    )
    return ChannelFlowSolution(
        times=saved,
        velocity=jnp.concatenate((initial[None, ...], advanced_velocity), axis=0),
        pressure=jnp.concatenate(
            (initial_pressure[None, ...], advanced_pressure), axis=0
        ),
        pressure_gradient=jnp.concatenate(
            (initial_gradient[None, ...], advanced_gradient), axis=0
        ),
        diagnostics=diagnostics,
        method=selected,
        dynamics=dynamics,
        solver_id=canonical_fingerprint(
            {
                "kind": "channel-sbdf2-solve-v1",
                "method": selected.method_id,
                "dynamics": dynamics.compilation_id,
                "step_size": float(durations[0]),
                "steps": int(saved.size) - 1,
            }
        ),
    )


def _stokes_output(result, /):
    diagnostics = result.diagnostics
    return (
        result.velocity,
        result.pressure,
        result.pressure_gradient,
        diagnostics.momentum_constraint_residual,
        diagnostics.divergence_norm,
        diagnostics.wall_residual,
        diagnostics.pressure_gauge_residual,
        diagnostics.bulk_velocity,
        result.successful,
    )


__all__ = [
    "ChannelFlowDiagnosticsHistory",
    "ChannelFlowSolution",
    "ChannelSBDF2Method",
    "solve_channel_sbdf2",
]

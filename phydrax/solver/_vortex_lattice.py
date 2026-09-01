#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.vortex._lifting import PreparedLiftingSurface
from ..linalg import (
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve as solve_linear,
)
from ..operators.integral.vortex._filament3d import regularized_filament_velocity_3d


class VortexLatticeResult(StrictModule):
    circulation: Array
    control_velocity: Array
    panel_force: Array
    total_force: Array
    residual_norm: Array
    linear_result: LinearSolveResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class SteadyVortexLatticePlan(StrictModule, NonTrainableState):
    """Prepared horseshoe-vortex lifting-surface solve."""

    surface: PreparedLiftingSurface
    wake_direction: Array
    wake_length: float = eqx.field(static=True)
    core_radius: float = eqx.field(static=True)
    density: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedLiftingSurface,
        wake_direction: ArrayLike,
        /,
        *,
        wake_length: float,
        core_radius: float,
        density: float = 1.0,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(surface, PreparedLiftingSurface):
            raise TypeError("surface must be PreparedLiftingSurface.")
        direction = jnp.asarray(wake_direction, dtype=surface.bound_start.dtype)
        if direction.shape != (3,):
            raise ValueError("wake_direction must have shape (3,).")
        norm = jnp.linalg.norm(direction)
        direction = eqx.error_if(
            direction, ~jnp.isfinite(norm) | (norm <= 0.0), "Wake direction is invalid."
        )
        length = float(wake_length)
        core = float(core_radius)
        density_ = float(density)
        if (
            not math.isfinite(length)
            or length <= 0.0
            or not math.isfinite(core)
            or core <= 0.0
            or not math.isfinite(density_)
            or density_ <= 0.0
        ):
            raise ValueError("VLM wake/core/density values must be finite and positive.")
        if linear_policy is not None and not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        self.surface = surface
        self.wake_direction = direction / norm
        self.wake_length = length
        self.core_radius = core
        self.density = density_
        self.linear_policy = linear_policy
        self.solver_id = canonical_fingerprint(
            {
                "kind": "steady-vortex-lattice",
                "surface": surface.surface_id,
                "wake_length": length,
                "core_radius": core,
                "density": density_,
            }
        )

    def influence_velocity(self, /) -> Array:
        target = self.surface.control_point
        far_left = self.surface.bound_start + self.wake_length * self.wake_direction
        far_right = self.surface.bound_end + self.wake_length * self.wake_direction
        columns = []
        for panel in range(self.surface.panel_count):
            starts = jnp.stack(
                (
                    far_left[panel],
                    self.surface.bound_start[panel],
                    self.surface.bound_end[panel],
                )
            )
            ends = jnp.stack(
                (
                    self.surface.bound_start[panel],
                    self.surface.bound_end[panel],
                    far_right[panel],
                )
            )
            columns.append(
                regularized_filament_velocity_3d(
                    target,
                    starts,
                    ends,
                    jnp.ones((3,), dtype=target.dtype),
                    jnp.full((3,), self.core_radius, dtype=target.dtype),
                )
            )
        return jnp.stack(tuple(columns), axis=1)

    def solve(self, freestream_velocity: ArrayLike, /) -> VortexLatticeResult:
        freestream = jnp.asarray(
            freestream_velocity, dtype=self.surface.control_point.dtype
        )
        if freestream.shape == (3,):
            freestream = jnp.broadcast_to(freestream, (self.surface.panel_count, 3))
        if freestream.shape != (self.surface.panel_count, 3):
            raise ValueError(
                "freestream_velocity must have shape (3,) or (panel_count, 3)."
            )
        influence = self.influence_velocity()
        matrix = contract("tjc,tc->tj", influence, self.surface.normal)
        right_hand_side = -jnp.sum(freestream * self.surface.normal, axis=-1)
        linear = solve_linear(
            LinearSystem(
                DenseLinearOperator(matrix), problem_id=f"{self.solver_id}:circulation"
            ),
            right_hand_side,
            policy=self.linear_policy,
        )
        circulation = jnp.asarray(linear.value)
        induced = contract("tjc,j->tc", influence, circulation)
        control_velocity = freestream + induced
        residual = matrix @ circulation - right_hand_side
        residual_norm = jnp.linalg.norm(residual)
        bound_vector = self.surface.bound_end - self.surface.bound_start
        panel_force = (
            self.density
            * circulation[:, None]
            * jnp.cross(control_velocity, bound_vector)
        )
        total_force = jnp.sum(panel_force, axis=0)
        successful = linear.successful & jnp.all(jnp.isfinite(panel_force))
        return VortexLatticeResult(
            circulation,
            control_velocity,
            panel_force,
            total_force,
            residual_norm,
            linear,
            successful,
            self.solver_id,
        )


__all__ = ["SteadyVortexLatticePlan", "VortexLatticeResult"]

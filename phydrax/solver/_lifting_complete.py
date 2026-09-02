#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.vortex._lifting_complete import PreparedMultiLiftingSurface
from ..linalg import (
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve as solve_linear,
)
from ..operators.integral.vortex._filament3d import regularized_filament_velocity_3d
from ._vortex_loads import (
    KuttaJoukowskiLoadPlan,
    TrefftzInducedDragPlan,
    VortexLoadResult,
)


class LiftingConstraintEvidence(StrictModule):
    normal_residual: Array
    kutta_residual: Array
    kelvin_residual: Array
    trailing_edge_owner: Array
    successful: Array


class CompleteLiftingResult(StrictModule):
    circulation: Array
    control_velocity: Array
    load: VortexLoadResult
    induced_drag: Array
    constraints: LiftingConstraintEvidence
    linear_result: LinearSolveResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class CompleteLiftingSystemPlan(StrictModule, NonTrainableState):
    surface: PreparedMultiLiftingSurface
    method: str = eqx.field(static=True)
    wake_direction: Array
    wake_length: float = eqx.field(static=True)
    core_radius: float = eqx.field(static=True)
    density: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedMultiLiftingSurface,
        method: str = "horseshoe-vlm",
        /,
        *,
        wake_direction: ArrayLike = (1.0, 0.0, 0.0),
        wake_length: float = 50.0,
        core_radius: float = 0.02,
        density: float = 1.0,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(surface, PreparedMultiLiftingSurface):
            raise TypeError("surface must be PreparedMultiLiftingSurface.")
        method_ = str(method)
        if method_ not in ("horseshoe-vlm", "ring-vlm", "lifting-line"):
            raise ValueError("Unsupported lifting method.")
        direction = jnp.asarray(wake_direction, dtype=surface.control_point.dtype)
        norm = jnp.linalg.norm(direction)
        if (
            direction.shape != (3,)
            or float(wake_length) <= 0.0
            or float(core_radius) <= 0.0
            or float(density) <= 0.0
        ):
            raise ValueError("Lifting wake/core/density controls are invalid.")
        direction = eqx.error_if(
            direction, ~jnp.isfinite(norm) | (norm <= 0.0), "Wake direction is invalid."
        )
        self.surface, self.method = surface, method_
        self.wake_direction, self.wake_length = direction / norm, float(wake_length)
        self.core_radius, self.density, self.linear_policy = (
            float(core_radius),
            float(density),
            linear_policy,
        )
        self.solver_id = canonical_fingerprint(
            {
                "kind": "complete-lifting-system",
                "surface": surface.system_id,
                "method": method_,
                "wake_length": self.wake_length,
                "core_radius": self.core_radius,
                "density": self.density,
            }
        )

    def _segments(self, panel: int, /) -> tuple[Array, Array]:
        surface = self.surface
        if self.method in ("horseshoe-vlm", "lifting-line"):
            far_left = surface.bound_start[panel] + self.wake_length * self.wake_direction
            far_right = surface.bound_end[panel] + self.wake_length * self.wake_direction
            start = jnp.stack(
                (far_left, surface.bound_start[panel], surface.bound_end[panel])
            )
            end = jnp.stack(
                (surface.bound_start[panel], surface.bound_end[panel], far_right)
            )
        else:
            start = jnp.stack(
                (
                    surface.bound_start[panel],
                    surface.bound_end[panel],
                    surface.trailing_end[panel],
                    surface.trailing_start[panel],
                )
            )
            end = jnp.stack(
                (
                    surface.bound_end[panel],
                    surface.trailing_end[panel],
                    surface.trailing_start[panel],
                    surface.bound_start[panel],
                )
            )
        return start, end

    def influence_velocity(self, targets: ArrayLike | None = None, /) -> Array:
        target = (
            self.surface.control_point
            if targets is None
            else jnp.asarray(targets, dtype=self.surface.control_point.dtype)
        )
        if target.ndim != 2 or target.shape[1] != 3:
            raise ValueError("Lifting influence targets require shape (N,3).")
        columns = []
        for panel in range(self.surface.panel_count):
            start, end = self._segments(panel)
            count = int(start.shape[0])
            columns.append(
                regularized_filament_velocity_3d(
                    target,
                    start,
                    end,
                    jnp.ones((count,), dtype=target.dtype),
                    jnp.full((count,), self.core_radius, dtype=target.dtype),
                )
            )
        return jnp.stack(tuple(columns), axis=1)

    def solve(
        self,
        incident_velocity: ArrayLike,
        /,
        *,
        wake_velocity: ArrayLike | None = None,
        previous_bound_circulation: ArrayLike | None = None,
        shed_circulation: ArrayLike | None = None,
        reference_point: ArrayLike = (0.0, 0.0, 0.0),
    ) -> CompleteLiftingResult:
        incident = jnp.asarray(incident_velocity, dtype=self.surface.control_point.dtype)
        if incident.shape == (3,):
            incident = jnp.broadcast_to(incident, (self.surface.panel_count, 3))
        wake = (
            jnp.zeros_like(incident)
            if wake_velocity is None
            else jnp.asarray(wake_velocity, dtype=incident.dtype)
        )
        if (
            incident.shape != (self.surface.panel_count, 3)
            or wake.shape != incident.shape
        ):
            raise ValueError("Lifting incident/wake velocities have invalid shapes.")
        relative = incident + wake - self.surface.body_velocity
        influence = self.influence_velocity()
        matrix = contract("tjc,tc->tj", influence, self.surface.normal)
        rhs = -jnp.sum(relative * self.surface.normal, axis=-1)
        linear = solve_linear(
            LinearSystem(
                DenseLinearOperator(matrix), problem_id=f"{self.solver_id}:circulation"
            ),
            rhs,
            policy=self.linear_policy,
        )
        circulation = jnp.asarray(linear.value)
        induced = contract("tjc,j->tc", influence, circulation)
        control_velocity = relative + induced
        normal_residual = matrix @ circulation - rhs
        previous = (
            jnp.zeros_like(circulation)
            if previous_bound_circulation is None
            else jnp.asarray(previous_bound_circulation, dtype=circulation.dtype)
        )
        shed = (
            previous - circulation
            if shed_circulation is None
            else jnp.asarray(shed_circulation, dtype=circulation.dtype)
        )
        if previous.shape != circulation.shape or shed.shape != circulation.shape:
            raise ValueError("Lifting Kelvin arrays must have panel-count shape.")
        kelvin = previous - circulation - shed
        # Every trailing edge has one explicit owner; Kutta residual is its normal-flow residual.
        kutta = normal_residual[self.surface.trailing_edge_owner]
        tolerance = (
            512
            * jnp.finfo(circulation.dtype).eps
            * jnp.maximum(jnp.linalg.norm(rhs), 1.0)
        )
        constraint_success = (jnp.linalg.norm(normal_residual) <= tolerance) & (
            jnp.linalg.norm(kelvin) <= tolerance
        )
        constraints = LiftingConstraintEvidence(
            normal_residual,
            kutta,
            kelvin,
            self.surface.trailing_edge_owner,
            constraint_success,
        )
        load = KuttaJoukowskiLoadPlan(self.density).evaluate(
            self.surface, circulation, control_velocity, reference_point=reference_point
        )
        downwash = jnp.sum(induced * self.surface.normal, axis=-1)
        _, induced_drag = TrefftzInducedDragPlan(self.density).evaluate(
            circulation, downwash, self.surface.span_width
        )
        successful = linear.successful & constraint_success & load.finite
        return CompleteLiftingResult(
            circulation,
            control_velocity,
            load,
            induced_drag,
            constraints,
            linear,
            successful,
            self.solver_id,
        )


__all__ = [
    "CompleteLiftingResult",
    "CompleteLiftingSystemPlan",
    "LiftingConstraintEvidence",
]

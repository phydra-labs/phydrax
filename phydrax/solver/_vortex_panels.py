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
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    solve as solve_linear,
)
from ..operators.integral.vortex._panels2d import (
    constant_panel_velocity_2d,
    FlowPanelGeometry2D,
    panel_influence_matrix_2d,
    RigidPanelMotion2D,
)


class VortexPanelResult2D(StrictModule):
    sheet_strength: Array
    surface_velocity: Array
    pressure_coefficient: Array
    panel_force: Array
    total_force: Array
    boundary_residual_norm: Array
    constraint_residual: Array
    linear_result: LinearSolveResult
    successful: Array
    solver_id: str = eqx.field(static=True)


class VortexPanelFlowPlan2D(StrictModule, NonTrainableState):
    """Constant vortex-panel impermeability solve with explicit closure row."""

    reference: FlowPanelGeometry2D
    prescribed_circulation: float = eqx.field(static=True)
    trailing_edge_panels: tuple[int, int] | None = eqx.field(static=True)
    density: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: FlowPanelGeometry2D,
        /,
        *,
        prescribed_circulation: float = 0.0,
        trailing_edge_panels: tuple[int, int] | None = None,
        density: float = 1.0,
        linear_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(geometry, FlowPanelGeometry2D):
            raise TypeError("geometry must be FlowPanelGeometry2D.")
        count = int(geometry.length.size)
        trailing = None
        if trailing_edge_panels is not None:
            trailing = tuple(int(value) for value in trailing_edge_panels)
            if (
                len(trailing) != 2
                or trailing[0] == trailing[1]
                or any(value < 0 or value >= count for value in trailing)
            ):
                raise ValueError(
                    "trailing_edge_panels must identify two distinct panels."
                )
        if density <= 0.0:
            raise ValueError("Panel-flow density must be positive.")
        policy = LinearSolvePolicy(DenseSVD()) if linear_policy is None else linear_policy
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        self.reference = geometry
        self.prescribed_circulation = float(prescribed_circulation)
        self.trailing_edge_panels = trailing
        self.density = float(density)
        self.linear_policy = policy
        self.solver_id = canonical_fingerprint(
            {
                "kind": "vortex-panel-flow-2d",
                "geometry": geometry.geometry_id,
                "prescribed_circulation": self.prescribed_circulation,
                "trailing_edge_panels": trailing,
                "density": self.density,
            }
        )

    def solve(
        self,
        background_velocity: ArrayLike,
        /,
        *,
        motion: RigidPanelMotion2D | None = None,
    ) -> VortexPanelResult2D:
        if motion is None:
            geometry = self.reference
            body_velocity = jnp.zeros_like(geometry.control)
        else:
            if not isinstance(motion, RigidPanelMotion2D):
                raise TypeError("motion must be RigidPanelMotion2D or None.")
            geometry, body_velocity = motion.realize(self.reference)
        background = jnp.asarray(background_velocity, dtype=geometry.control.dtype)
        if background.shape == (2,):
            background = jnp.broadcast_to(background, geometry.control.shape)
        if background.shape != geometry.control.shape:
            raise ValueError(
                "background_velocity must have shape (2,) or (panel_count, 2)."
            )
        normal_matrix, _ = panel_influence_matrix_2d(
            geometry,
            kind="vortex",
        )
        relative = background - body_velocity
        rhs = -jnp.sum(relative * geometry.normal, axis=-1)
        matrix = normal_matrix
        if self.trailing_edge_panels is None:
            closure = geometry.length
            closure_rhs = jnp.asarray(self.prescribed_circulation, dtype=rhs.dtype)
        else:
            first, second = self.trailing_edge_panels
            closure = (
                jnp.zeros_like(geometry.length).at[first].set(1.0).at[second].set(1.0)
            )
            closure_rhs = jnp.asarray(0.0, dtype=rhs.dtype)
        row = int(geometry.length.size - 1)
        constrained_matrix = matrix.at[row].set(closure)
        constrained_rhs = rhs.at[row].set(closure_rhs)
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(constrained_matrix),
                problem_id=f"{self.solver_id}:sheet",
            ),
            constrained_rhs,
            policy=self.linear_policy,
        )
        strength = jnp.asarray(linear.value)
        induced_control = constant_panel_velocity_2d(
            geometry.control, geometry, strength, kind="vortex"
        )
        surface_velocity = relative + induced_control
        tangential_speed = jnp.sum(surface_velocity * geometry.tangent, axis=-1)
        reference_speed_squared = jnp.maximum(
            jnp.mean(jnp.sum(background * background, axis=-1)),
            jnp.finfo(background.dtype).tiny,
        )
        pressure = 1.0 - tangential_speed**2 / reference_speed_squared
        dynamic_pressure = 0.5 * self.density * reference_speed_squared
        panel_force = (
            -dynamic_pressure
            * pressure[:, None]
            * geometry.normal
            * geometry.length[:, None]
        )
        residual = jnp.sum(surface_velocity * geometry.normal, axis=-1)
        boundary_residual = jnp.linalg.norm(residual)
        constraint_residual = closure @ strength - closure_rhs
        successful = linear.successful & jnp.all(jnp.isfinite(panel_force))
        return VortexPanelResult2D(
            strength,
            surface_velocity,
            pressure,
            panel_force,
            jnp.sum(panel_force, axis=0),
            boundary_residual,
            constraint_residual,
            linear,
            successful,
            self.solver_id,
        )


__all__ = ["VortexPanelFlowPlan2D", "VortexPanelResult2D"]

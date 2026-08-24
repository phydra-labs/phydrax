#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve as solve_linear,
)
from ..operators.integral.layer_potential import (
    BoundaryLayerApproximationReport,
    BoundaryPanelization2D,
    double_layer_principal_value_matrix,
    LaplaceLayerPotential2D,
)


class InteriorLaplaceDirichletResult(StrictModule):
    """Solved double-layer density and independently certified interior potential."""

    density: Array
    potential: LaplaceLayerPotential2D
    linear_result: LinearSolveResult
    approximation: BoundaryLayerApproximationReport
    boundary_residual_norm: Array
    valid: Array

    def __init__(
        self,
        *,
        density: Array,
        potential: LaplaceLayerPotential2D,
        linear_result: LinearSolveResult,
        approximation: BoundaryLayerApproximationReport,
        boundary_residual_norm: Array,
    ):
        if not isinstance(potential, LaplaceLayerPotential2D):
            raise TypeError("potential must be LaplaceLayerPotential2D.")
        if not isinstance(linear_result, LinearSolveResult):
            raise TypeError("linear_result must be LinearSolveResult.")
        if not isinstance(approximation, BoundaryLayerApproximationReport):
            raise TypeError("approximation must be BoundaryLayerApproximationReport.")
        residual = jnp.asarray(boundary_residual_norm)
        self.density = jnp.asarray(density)
        self.potential = potential
        self.linear_result = linear_result
        self.approximation = approximation
        self.boundary_residual_norm = residual
        self.valid = (
            linear_result.successful
            & linear_result.diagnostics.finite
            & jnp.isfinite(residual)
        )


def solve_interior_laplace_dirichlet_2d(
    panelization: BoundaryPanelization2D,
    boundary_values: ArrayLike,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
) -> InteriorLaplaceDirichletResult:
    """Solve the interior Dirichlet problem with an outward-normal double layer."""

    if not isinstance(panelization, BoundaryPanelization2D):
        raise TypeError("panelization must be BoundaryPanelization2D.")
    values = jnp.asarray(boundary_values, dtype=float)
    if values.shape != (panelization.node_count,):
        raise ValueError("boundary_values must contain one value per source node.")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("boundary_values must be finite.")
    policy = LinearSolvePolicy(DenseLU()) if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    principal_value = double_layer_principal_value_matrix(panelization)
    trace_matrix = principal_value - 0.5 * jnp.eye(
        panelization.node_count,
        dtype=principal_value.dtype,
    )
    problem = LinearSystem(
        DenseLinearOperator(trace_matrix),
        problem_id="interior-laplace-dirichlet-double-layer-2d",
    )
    linear_result = solve_linear(problem, values, policy=policy)
    density = jnp.asarray(linear_result.value)
    residual = jnp.linalg.norm(trace_matrix @ density - values)
    potential = LaplaceLayerPotential2D(
        panelization,
        kind="double",
        density=density,
    )
    approximation = BoundaryLayerApproximationReport(
        panelization=panelization,
        kernel_id=potential.kernel.kernel_id,
        density_space="quadrature-node-values",
        trace_policy="interior-minus-half-jump-local-diagonal",
    )
    return InteriorLaplaceDirichletResult(
        density=density,
        potential=potential,
        linear_result=linear_result,
        approximation=approximation,
        boundary_residual_norm=residual,
    )


__all__ = [
    "InteriorLaplaceDirichletResult",
    "solve_interior_laplace_dirichlet_2d",
]

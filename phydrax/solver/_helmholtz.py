#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..integration import AdaptiveQuadraturePlan
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve as solve_linear,
)
from ..operators.integral.layer_potential import (
    BoundaryOperatorAssemblyReport,
    BoundaryPanelization2D,
    HelmholtzCombinedField2D,
    HelmholtzLayerKernel2D,
)
from ..operators.integral.layer_potential._quadrature2d import (
    evaluate_helmholtz_single_layer_self_panel_block_2d,
)


class ExteriorHelmholtzDirichletResult2D(StrictModule):
    """Solved outgoing Brakhage--Werner density and combined field."""

    density: Array
    potential: HelmholtzCombinedField2D
    linear_result: LinearSolveResult
    assembly_report: BoundaryOperatorAssemblyReport
    discretization: object
    coupling: float
    boundary_residual_norm: Array
    valid: Array

    def __init__(
        self,
        *,
        density: Array,
        potential: HelmholtzCombinedField2D,
        linear_result: LinearSolveResult,
        assembly_report: BoundaryOperatorAssemblyReport,
        discretization: object,
        coupling: float,
        boundary_residual_norm: Array,
    ):
        if not isinstance(potential, HelmholtzCombinedField2D):
            raise TypeError("potential must be HelmholtzCombinedField2D.")
        if not isinstance(linear_result, LinearSolveResult):
            raise TypeError("linear_result must be LinearSolveResult.")
        if not isinstance(assembly_report, BoundaryOperatorAssemblyReport):
            raise TypeError(
                "assembly_report must be BoundaryOperatorAssemblyReport."
            )
        residual = jnp.asarray(boundary_residual_norm)
        self.density = jnp.asarray(density)
        self.potential = potential
        self.linear_result = linear_result
        self.assembly_report = assembly_report
        self.discretization = discretization
        self.coupling = float(coupling)
        self.boundary_residual_norm = residual
        self.valid = (
            linear_result.successful
            & linear_result.diagnostics.finite
            & assembly_report.accuracy_supported
            & jnp.isfinite(residual)
        )


def _helmholtz_trace_matrices_2d(
    panelization: BoundaryPanelization2D,
    kernel: HelmholtzLayerKernel2D,
    quadrature: AdaptiveQuadraturePlan,
    /,
) -> tuple[Array, Array, BoundaryOperatorAssemblyReport]:
    targets = panelization.points
    sources = panelization.points
    normals = panelization.normals
    single = jax.vmap(
        lambda target: jax.vmap(kernel.value, in_axes=(None, 0))(target, sources)
    )(targets)
    double = jax.vmap(
        lambda target: jax.vmap(
            kernel.source_normal_derivative,
            in_axes=(None, 0, 0),
        )(target, sources, normals)
    )(targets)
    order = panelization.quadrature_order
    block_status = []
    block_errors = []
    block_evaluations = []
    for panel_id in range(panelization.panel_count):
        start = panel_id * order
        stop = start + order
        self_weights = evaluate_helmholtz_single_layer_self_panel_block_2d(
            panelization,
            kernel,
            panel_id,
            panelization.references[start:stop, 0],
            quadrature,
        )
        single = single.at[start:stop, start:stop].set(self_weights.value)
        block_status.append(self_weights.status)
        block_errors.append(
            jnp.inf
            if self_weights.error_estimate is None
            else self_weights.error_estimate
        )
        block_evaluations.append(self_weights.num_evaluations)
    policy_id = (
        f"adaptive:{type(quadrature.rule).__name__}:"
        f"abs={quadrature.absolute_tolerance}:"
        f"rel={quadrature.relative_tolerance}:"
        f"max_intervals={quadrature.max_intervals}"
    )
    assembly = BoundaryOperatorAssemblyReport(
        panelization=panelization,
        kernel_id=kernel.kernel_id,
        policy_id=policy_id,
        trace_policy="exterior-brakhage-werner-single-self-product-double-shift",
        block_status=jnp.stack(block_status),
        block_errors=jnp.stack(block_errors),
        block_evaluations=jnp.stack(block_evaluations),
    )
    if not bool(assembly.accuracy_supported):
        raise ValueError(
            "Helmholtz singular trace assembly failed its quadrature contract."
        )
    step = 1.0 / (
        panelization.panels_per_chart * panelization.quadrature_order * 1_000_000.0
    )
    reference = panelization.references[:, 0]
    direction = jnp.where(reference + step < 1.0, 1.0, -1.0)
    shifted_reference = (reference + direction * step)[:, None]
    shifted = panelization.atlas.frame(
        panelization.chart_indices,
        shifted_reference,
    )
    diagonal = jax.vmap(kernel.source_normal_derivative)(
        targets,
        shifted.origin,
        shifted.normal,
    )
    indices = jnp.arange(panelization.node_count)
    double = double.at[indices, indices].set(diagonal)
    weights = panelization.weights[None, :]
    return single * weights, double * weights, assembly
def solve_exterior_helmholtz_dirichlet_2d(
    panelization: BoundaryPanelization2D,
    boundary_values: ArrayLike,
    wavenumber: float,
    /,
    *,
    quadrature: AdaptiveQuadraturePlan,
    eta: float | None = None,
    linear: LinearSolvePolicy | None = None,
) -> ExteriorHelmholtzDirichletResult2D:
    """Solve an exterior Dirichlet trace with the outgoing Brakhage--Werner field."""
    if not isinstance(panelization, BoundaryPanelization2D):
        raise TypeError("panelization must be BoundaryPanelization2D.")
    if not isinstance(quadrature, AdaptiveQuadraturePlan):
        raise TypeError("quadrature must be an AdaptiveQuadraturePlan.")
    values = jnp.asarray(boundary_values, dtype=complex)
    if values.shape != (panelization.node_count,):
        raise ValueError("boundary_values must contain one value per source node.")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("boundary_values must be finite.")
    kernel = HelmholtzLayerKernel2D(wavenumber)
    coupling = max(float(kernel.wavenumber), 1.0) if eta is None else float(eta)
    if not jnp.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("eta must be finite and positive.")
    single, double, assembly_report = _helmholtz_trace_matrices_2d(
        panelization,
        kernel,
        quadrature,
    )
    trace_matrix = double + 0.5 * jnp.eye(
        panelization.node_count,
        dtype=double.dtype,
    ) - 1j * coupling * single
    problem = LinearSystem(
        DenseLinearOperator(trace_matrix),
        problem_id="exterior-helmholtz-dirichlet-brakhage-werner-2d",
    )
    policy = LinearSolvePolicy(DenseLU()) if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    linear_result = solve_linear(problem, values, policy=policy)
    density = jnp.asarray(linear_result.value)
    potential = HelmholtzCombinedField2D(
        panelization,
        kernel.wavenumber,
        density,
        eta=coupling,
    )
    residual = jnp.linalg.norm(trace_matrix @ density - values)
    return ExteriorHelmholtzDirichletResult2D(
        density=density,
        potential=potential,
        linear_result=linear_result,
        assembly_report=assembly_report,
        discretization=potential.discretization_report(),
        coupling=coupling,
        boundary_residual_norm=residual,
    )


__all__ = [
    "ExteriorHelmholtzDirichletResult2D",
    "solve_exterior_helmholtz_dirichlet_2d",
]

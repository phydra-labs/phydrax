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
    HelmholtzCombinedField3D,
    HelmholtzLayerKernel3D,
    HelmholtzLayerPotential3D,
    SurfacePanelization3D,
)
from ..operators.integral.layer_potential._quadrature3d import (
    evaluate_double_layer_self_triangle_3d,
    evaluate_single_layer_self_triangle_3d,
)


class ExteriorHelmholtzDirichletResult3D(StrictModule):
    """Solved 3D outgoing Brakhage--Werner density and field."""

    density: Array
    potential: HelmholtzCombinedField3D
    linear_result: LinearSolveResult
    assembly_report: BoundaryOperatorAssemblyReport
    boundary_residual_norm: Array
    valid: Array

    def __init__(
        self,
        *,
        density: Array,
        potential: HelmholtzCombinedField3D,
        linear_result: LinearSolveResult,
        assembly_report: BoundaryOperatorAssemblyReport,
        boundary_residual_norm: Array,
    ):
        if not isinstance(potential, HelmholtzCombinedField3D):
            raise TypeError("potential must be HelmholtzCombinedField3D.")
        if not isinstance(linear_result, LinearSolveResult):
            raise TypeError("linear_result must be LinearSolveResult.")
        if not isinstance(assembly_report, BoundaryOperatorAssemblyReport):
            raise TypeError("assembly_report must be BoundaryOperatorAssemblyReport.")
        residual = jnp.asarray(boundary_residual_norm)
        self.density = jnp.asarray(density)
        self.potential = potential
        self.linear_result = linear_result
        self.assembly_report = assembly_report
        self.boundary_residual_norm = residual
        self.valid = (
            linear_result.successful
            & linear_result.diagnostics.finite
            & assembly_report.accuracy_supported
            & jnp.isfinite(residual)
        )


def _trace_matrices_3d(
    panelization: SurfacePanelization3D,
    kernel: HelmholtzLayerKernel3D,
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
    weights = panelization.weights[None, :]
    single = single * weights
    double = double * weights
    zero_density = jnp.zeros((panelization.node_count,), dtype=complex)
    single_base = HelmholtzLayerPotential3D(
        panelization,
        kernel.wavenumber,
        kind="single",
        density=zero_density,
    )
    double_base = HelmholtzLayerPotential3D(
        panelization,
        kernel.wavenumber,
        kind="double",
        density=zero_density,
    )
    status = []
    errors = []
    evaluations = []
    for panel_id in range(panelization.panel_count):
        start = panel_id * panelization.nodes_per_panel
        stop = start + panelization.nodes_per_panel
        for target_index in range(start, stop):
            single_values = []
            double_values = []
            for source_index in range(start, stop):
                density = jnp.zeros_like(zero_density).at[source_index].set(1.0 + 0.0j)
                single_estimate = evaluate_single_layer_self_triangle_3d(
                    single_base.with_density(density),
                    panel_id,
                    panelization.references[target_index],
                    quadrature,
                )
                double_estimate = evaluate_double_layer_self_triangle_3d(
                    double_base.with_density(density),
                    panel_id,
                    panelization.references[target_index],
                    quadrature,
                )
                single_values.append(single_estimate.value)
                double_values.append(double_estimate.value)
                status.extend((single_estimate.status, double_estimate.status))
                errors.extend(
                    (
                        jnp.inf
                        if single_estimate.error_estimate is None
                        else single_estimate.error_estimate,
                        jnp.inf
                        if double_estimate.error_estimate is None
                        else double_estimate.error_estimate,
                    )
                )
                evaluations.extend(
                    (single_estimate.num_evaluations, double_estimate.num_evaluations)
                )
            single = single.at[target_index, start:stop].set(jnp.stack(single_values))
            double = double.at[target_index, start:stop].set(jnp.stack(double_values))
    report = BoundaryOperatorAssemblyReport(
        panelization=panelization,
        kernel_id=kernel.kernel_id,
        policy_id=(
            f"adaptive:{type(quadrature.rule).__name__}:"
            f"abs={quadrature.absolute_tolerance}:rel={quadrature.relative_tolerance}:"
            f"max_intervals={quadrature.max_intervals}"
        ),
        trace_policy="3d-exterior-brakhage-werner-single-and-double-duffy",
        block_status=jnp.stack(status),
        block_errors=jnp.stack(errors),
        block_evaluations=jnp.stack(evaluations),
    )
    if not bool(report.accuracy_supported):
        raise ValueError("3D Helmholtz singular assembly failed its quadrature contract.")
    return single, double, report


def solve_exterior_helmholtz_dirichlet_3d(
    panelization: SurfacePanelization3D,
    boundary_values: ArrayLike,
    wavenumber: float,
    /,
    *,
    quadrature: AdaptiveQuadraturePlan,
    eta: float | None = None,
    linear: LinearSolvePolicy | None = None,
) -> ExteriorHelmholtzDirichletResult3D:
    """Solve 3D exterior Dirichlet CFIE with explicit Duffy self policy."""
    if not isinstance(panelization, SurfacePanelization3D):
        raise TypeError("panelization must be SurfacePanelization3D.")
    if not isinstance(quadrature, AdaptiveQuadraturePlan):
        raise TypeError("quadrature must be an AdaptiveQuadraturePlan.")
    values = jnp.asarray(boundary_values, dtype=complex)
    if values.shape != (panelization.node_count,):
        raise ValueError("boundary_values must match surface node count.")
    kernel = HelmholtzLayerKernel3D(wavenumber)
    coupling = max(kernel.wavenumber, 1.0) if eta is None else float(eta)
    if not jnp.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("eta must be finite and positive.")
    single, double, report = _trace_matrices_3d(panelization, kernel, quadrature)
    trace = double + 0.5 * jnp.eye(panelization.node_count, dtype=double.dtype) - 1j * coupling * single
    problem = LinearSystem(
        DenseLinearOperator(trace),
        problem_id="exterior-helmholtz-dirichlet-brakhage-werner-3d",
    )
    policy = LinearSolvePolicy(DenseLU()) if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be a LinearSolvePolicy or None.")
    linear_result = solve_linear(problem, values, policy=policy)
    density = jnp.asarray(linear_result.value)
    potential = HelmholtzCombinedField3D(
        panelization,
        kernel.wavenumber,
        density,
        eta=coupling,
    )
    residual = jnp.linalg.norm(trace @ density - values)
    return ExteriorHelmholtzDirichletResult3D(
        density=density,
        potential=potential,
        linear_result=linear_result,
        assembly_report=report,
        boundary_residual_norm=residual,
    )


__all__ = [
    "ExteriorHelmholtzDirichletResult3D",
    "solve_exterior_helmholtz_dirichlet_3d",
]

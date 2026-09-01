#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare as prepare_linear,
    solve,
)
from ..operators.integral.layer_potential._helmholtz3d import (
    HelmholtzCombinedField3D,
    HelmholtzLayerPotential3D,
)
from ..operators.integral.layer_potential._laplace3d import LaplaceLayerPotential3D
from ..operators.integral.layer_potential._scalar_calderon3d import (
    ScalarCalderonAssemblyReport3D,
)
from ..operators.integral.layer_potential._scalar_formulations3d import (
    ScalarBoundaryFormulation3D,
    ScalarBoundaryFormulationMetadata3D,
)


ScalarBoundaryPotential3D = (
    LaplaceLayerPotential3D | HelmholtzLayerPotential3D | HelmholtzCombinedField3D
)


class ScalarBoundarySolveResult3D(StrictModule):
    """One discrete closed-surface scalar BIE solve with bounded evidence.

    ``metadata`` states the exact 3D PDE, two-dimensional closed triangular
    geometry, formulation, trace jumps, provider, precision, compatibility,
    gauge, resonance risk, resource/error evidence, and non-goals.
    ``assembly_report`` records the underlying quadrature evidence. The result
    is a DP0 discrete solve and does not claim continuum certification.
    """

    solution: Array
    right_hand_side: Array
    boundary_data: Array
    boundary_dirichlet: Array | None
    boundary_neumann: Array | None
    linear_result: LinearSolveResult
    potential: ScalarBoundaryPotential3D | None
    compatibility_residual: Array
    gauge_residual: Array
    finite: Array
    valid: Array
    metadata: ScalarBoundaryFormulationMetadata3D
    assembly_report: ScalarCalderonAssemblyReport3D


def _default_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def _component_integrals(
    formulation: ScalarBoundaryFormulation3D,
    values: Array,
    /,
) -> Array:
    calderon = formulation.calderon
    return jnp.stack(
        tuple(
            jnp.sum(
                jnp.where(
                    calderon.face_component_ids == component,
                    calderon.face_areas * values,
                    0.0,
                )
            )
            for component in range(calderon.component_count)
        )
    )


def _component_means(
    formulation: ScalarBoundaryFormulation3D,
    values: Array,
    /,
) -> Array:
    calderon = formulation.calderon
    measures = _component_integrals(
        formulation,
        jnp.ones((calderon.face_count,), dtype=calderon.space.dtype),
    )
    return _component_integrals(formulation, values) / measures


def _check_compatibility(
    formulation: ScalarBoundaryFormulation3D,
    boundary_data: Array,
    tolerance: float,
    /,
) -> Array:
    calderon = formulation.calderon
    residual = jnp.zeros((calderon.component_count,), dtype=calderon.space.dtype)
    if (
        calderon.kernel.family != "laplace"
        or formulation.metadata.formulation_name
        != "interior-Neumann-direct-Calderon-trace"
    ):
        return residual
    residual = _component_integrals(formulation, boundary_data)
    scales = jnp.stack(
        tuple(
            jnp.maximum(
                jnp.sum(
                    jnp.where(
                        calderon.face_component_ids == component,
                        calderon.face_areas * jnp.abs(boundary_data),
                        0.0,
                    )
                ),
                1.0,
            )
            for component in range(calderon.component_count)
        )
    )
    if not bool(jnp.all(jnp.abs(residual) <= tolerance * scales)):
        raise ValueError(
            "Pure interior Laplace Neumann data violates the per-component "
            "zero-flux compatibility condition."
        )
    return residual


def _traces(
    formulation: ScalarBoundaryFormulation3D,
    solution: Array,
    boundary_data: Array,
    /,
) -> tuple[Array | None, Array | None]:
    calderon = formulation.calderon
    representation = formulation.metadata.representation
    side = formulation.metadata.side
    if representation == "direct-trace":
        return solution, boundary_data
    if representation == "single-layer":
        boundary_dirichlet = calderon.single_layer.mv(solution)
        jump = calderon.trace_convention.single_layer_neumann_jump(side)
        boundary_neumann = calderon.adjoint_double_layer.mv(solution) + jump * solution
        return boundary_dirichlet, boundary_neumann
    if representation == "double-layer":
        jump = calderon.trace_convention.double_layer_dirichlet_jump(side)
        return calderon.double_layer.mv(solution) + jump * solution, None
    if representation == "combined-field":
        return formulation.operator.mv(solution), None
    raise ValueError("Unknown scalar layer representation.")


def _potential(
    formulation: ScalarBoundaryFormulation3D,
    solution: Array,
    /,
) -> ScalarBoundaryPotential3D | None:
    calderon = formulation.calderon
    if calderon.kernel.family == "modified-helmholtz":
        return None
    representation = formulation.metadata.representation
    if representation == "single-layer":
        return calderon.single_layer_potential(solution)
    if representation == "double-layer":
        return calderon.double_layer_potential(solution)
    if representation == "combined-field":
        coupling = formulation.metadata.coupling_parameter
        if coupling is None:
            raise ValueError("Combined-field formulation lacks its coupling parameter.")
        return calderon.combined_field_potential(solution, eta=coupling)
    return None


def solve_scalar_boundary_3d(
    formulation: ScalarBoundaryFormulation3D,
    boundary_data: ArrayLike,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
    compatibility_tolerance: float = 1.0e-10,
) -> ScalarBoundarySolveResult3D:
    """Solve one prepared scalar BIE, enforcing declared Neumann compatibility."""
    if not isinstance(formulation, ScalarBoundaryFormulation3D):
        raise TypeError("formulation must be ScalarBoundaryFormulation3D.")
    tolerance = float(compatibility_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("compatibility_tolerance must be finite and nonnegative.")
    calderon = formulation.calderon
    data = jnp.asarray(boundary_data, dtype=calderon.space.dtype)
    data = calderon.space.validate(data)
    compatibility_residual = _check_compatibility(formulation, data, tolerance)
    policy = _default_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError(
            "Scalar boundary solves currently require differentiation mode 'none'."
        )
    right_hand_side = formulation.right_hand_side(data)
    problem = LinearSystem(
        formulation.operator,
        problem_id=formulation.metadata.formulation_id,
    )
    prepared_linear = prepare_linear(problem, policy)
    linear_result = solve(prepared_linear, right_hand_side)
    solution = calderon.space.validate(jnp.asarray(linear_result.value))
    boundary_dirichlet, boundary_neumann = _traces(formulation, solution, data)
    gauge_residual = jnp.zeros((calderon.component_count,), dtype=calderon.space.dtype)
    if (
        calderon.kernel.family == "laplace"
        and formulation.metadata.formulation_name
        == "interior-Neumann-direct-Calderon-trace"
    ):
        gauge_residual = _component_means(formulation, solution)
    potential = _potential(formulation, solution)
    trace_finite = (
        True if boundary_dirichlet is None else jnp.all(jnp.isfinite(boundary_dirichlet))
    ) & (True if boundary_neumann is None else jnp.all(jnp.isfinite(boundary_neumann)))
    finite = (
        jnp.all(jnp.isfinite(solution))
        & jnp.all(jnp.isfinite(right_hand_side))
        & jnp.all(jnp.isfinite(compatibility_residual))
        & jnp.all(jnp.isfinite(gauge_residual))
        & trace_finite
    )
    valid = (
        calderon.assembly_report.accuracy_supported
        & linear_result.successful
        & linear_result.diagnostics.finite
        & finite
    )
    return ScalarBoundarySolveResult3D(
        solution=solution,
        right_hand_side=right_hand_side,
        boundary_data=data,
        boundary_dirichlet=boundary_dirichlet,
        boundary_neumann=boundary_neumann,
        linear_result=linear_result,
        potential=potential,
        compatibility_residual=compatibility_residual,
        gauge_residual=gauge_residual,
        finite=finite,
        valid=valid,
        metadata=formulation.metadata,
        assembly_report=calderon.assembly_report,
    )


def solve_laplace_boundary_3d(
    formulation: ScalarBoundaryFormulation3D,
    boundary_data: ArrayLike,
    /,
    **kwargs,
) -> ScalarBoundarySolveResult3D:
    """Named Laplace path; reject a formulation from any other kernel family."""
    if not isinstance(formulation, ScalarBoundaryFormulation3D):
        raise TypeError("formulation must be ScalarBoundaryFormulation3D.")
    if formulation.calderon.kernel.family != "laplace":
        raise ValueError("solve_laplace_boundary_3d requires a Laplace formulation.")
    return solve_scalar_boundary_3d(formulation, boundary_data, **kwargs)


def solve_helmholtz_boundary_3d(
    formulation: ScalarBoundaryFormulation3D,
    boundary_data: ArrayLike,
    /,
    **kwargs,
) -> ScalarBoundarySolveResult3D:
    """Named outgoing Helmholtz path with raw/CFIE risk retained in metadata."""
    if not isinstance(formulation, ScalarBoundaryFormulation3D):
        raise TypeError("formulation must be ScalarBoundaryFormulation3D.")
    if formulation.calderon.kernel.family != "outgoing-helmholtz":
        raise ValueError(
            "solve_helmholtz_boundary_3d requires an outgoing Helmholtz formulation."
        )
    return solve_scalar_boundary_3d(formulation, boundary_data, **kwargs)


__all__ = [
    "ScalarBoundarySolveResult3D",
    "solve_helmholtz_boundary_3d",
    "solve_laplace_boundary_3d",
    "solve_scalar_boundary_3d",
]

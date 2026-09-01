#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
)
from ..operators.integral.layer_potential._maxwell3d import (
    MaxwellEFIEAssemblyReport3D,
    PreparedMaxwellEFIE3D,
)


class PECEFIEResult3D(StrictModule):
    """One finite dense PEC EFIE current solution within the declared envelope."""

    coefficients: Array
    surface_current_at_centroids: Array
    surface_divergence: Array
    right_hand_side: Array
    residual_norm: Array
    relative_residual: Array
    linear_result: LinearSolveResult
    assembly_report: MaxwellEFIEAssemblyReport3D
    valid: Array
    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)


def solve_pec_efie_3d(
    prepared: PreparedMaxwellEFIE3D,
    incident_electric: ArrayLike,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
) -> PECEFIEResult3D:
    """Solve n x (E_inc + E_scat)=0 in the prepared mass-lumped RWG EFIE."""
    if not isinstance(prepared, PreparedMaxwellEFIE3D):
        raise TypeError("prepared must be PreparedMaxwellEFIE3D.")
    if not bool(
        prepared.assembly_report.finite
        & prepared.assembly_report.discrete_accuracy_supported
    ):
        raise ValueError(
            "Prepared Maxwell EFIE lacks finite supported discrete evidence."
        )
    policy = (
        LinearSolvePolicy(
            DenseLU(),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
        )
        if linear is None
        else linear
    )
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError("PEC EFIE foundation requires differentiation mode 'none'.")
    if not isinstance(policy.method, DenseLU):
        raise ValueError(
            "Bounded PEC EFIE foundation requires the explicit DenseLU method."
        )
    rhs = prepared.incident_rhs(incident_electric)
    linear_result = solve(
        LinearSystem(
            prepared.operator,
            problem_id=f"pec-maxwell-efie-3d:{prepared.prepared_id}",
        ),
        rhs,
        policy=policy,
    )
    coefficients = jnp.asarray(linear_result.value)
    residual = prepared.operator.mv(coefficients) - rhs
    residual_norm = jnp.linalg.norm(residual)
    rhs_norm = jnp.linalg.norm(rhs)
    relative_residual = residual_norm / jnp.maximum(
        rhs_norm, jnp.finfo(rhs_norm.dtype).tiny
    )
    current = prepared.current_space.current_at_centroids(coefficients)
    divergence = prepared.current_space.surface_divergence(coefficients)
    finite = (
        jnp.all(jnp.isfinite(coefficients))
        & jnp.all(jnp.isfinite(current))
        & jnp.all(jnp.isfinite(divergence))
        & jnp.isfinite(residual_norm)
        & jnp.isfinite(relative_residual)
    )
    valid = (
        finite
        & linear_result.successful
        & linear_result.diagnostics.finite
        & prepared.assembly_report.discrete_accuracy_supported
    )
    return PECEFIEResult3D(
        coefficients=coefficients,
        surface_current_at_centroids=current,
        surface_divergence=divergence,
        right_hand_side=rhs,
        residual_norm=residual_norm,
        relative_residual=relative_residual,
        linear_result=linear_result,
        assembly_report=prepared.assembly_report,
        valid=valid,
        ambient_dimension=3,
        pde="source-free exterior time-harmonic Maxwell scattering by a perfect electric conductor",
        geometry="one connected oriented closed genus-zero piecewise-planar triangular boundary",
        formulation="dense mass-lumped RWG electric field integral equation with exp(-i omega t)",
        provider="Phydra DenseLU over PreparedMaxwellEFIE3D",
        precision=str(coefficients.dtype),
        resource_evidence=f"one dense solve with {prepared.current_space.size} complex unknowns",
        error_evidence="algebraic residual and substrate quadrature evidence; no continuum discretization bound",
        non_goals=(
            "low-frequency or dense-discretization stabilization",
            "multiply connected conductors",
            "BC/RBC, Calderon, or CFIE formulations",
            "interior-resonance immunity",
            "continuum certification",
        ),
    )


__all__ = ["PECEFIEResult3D", "solve_pec_efie_3d"]

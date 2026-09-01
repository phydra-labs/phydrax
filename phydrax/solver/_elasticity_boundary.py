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
from ..operators.integral.layer_potential._elasticity3d import (
    ElasticityBoundaryContract3D,
    ElasticityLayerPotential3D,
    ElasticityNullspaceMetadata3D,
    ElasticitySingleLayerDP0AssemblyReport3D,
    ElasticitySingleLayerDP0Galerkin3D,
)


class ElasticityDirichletResult3D(StrictModule):
    """Bounded static-isotropic closed-triangle Kelvin Dirichlet result.

    The three-dimensional displacement trace is solved with the prepared DP0
    single-layer operator. ``traction_density`` uses outward Cauchy traction
    components. Provider, precision, resource/error evidence, geometry envelope,
    and non-goals are carried verbatim in ``contract``; the report explicitly
    does not certify continuum discretization error.
    """

    traction_density: Array
    prescribed_displacement: Array
    potential: ElasticityLayerPotential3D
    linear_result: LinearSolveResult
    assembly_report: ElasticitySingleLayerDP0AssemblyReport3D
    nullspace: ElasticityNullspaceMetadata3D
    contract: ElasticityBoundaryContract3D = eqx.field(static=True)
    boundary_residual_norm: Array
    finite: Array
    valid: Array
    formulation: str = eqx.field(static=True)


def _default_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        DenseLU(),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )


def solve_elasticity_interior_displacement_dirichlet_3d(
    galerkin: ElasticitySingleLayerDP0Galerkin3D,
    boundary_displacement: ArrayLike,
    /,
    *,
    linear: LinearSolvePolicy | None = None,
) -> ElasticityDirichletResult3D:
    """Solve one interior displacement Dirichlet trace by a Kelvin single layer."""
    if not isinstance(galerkin, ElasticitySingleLayerDP0Galerkin3D):
        raise TypeError("galerkin must be ElasticitySingleLayerDP0Galerkin3D.")
    if not bool(galerkin.assembly_report.accuracy_supported):
        raise ValueError("Elasticity Galerkin quadrature evidence does not meet policy.")
    values = jnp.asarray(boundary_displacement, dtype=galerkin.face_areas.dtype)
    if values.shape != (galerkin.face_count, 3):
        raise ValueError("boundary_displacement must have shape (face_count, 3).")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("boundary_displacement must be finite.")
    policy = _default_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError(
            "Elasticity boundary solves require differentiation mode 'none'."
        )

    right = values.reshape(-1)
    linear_result = solve(
        LinearSystem(
            galerkin.strong_operator,
            problem_id="static-elasticity-interior-displacement-dirichlet-3d",
        ),
        right,
        policy=policy,
    )
    density_flat = jnp.asarray(linear_result.value)
    density = density_flat.reshape((galerkin.face_count, 3))
    residual = galerkin.strong_operator.mv(density_flat) - right
    residual_norm = jnp.linalg.norm(residual)
    finite = (
        jnp.all(jnp.isfinite(density))
        & jnp.all(jnp.isfinite(residual))
        & jnp.isfinite(residual_norm)
    )
    valid = (
        galerkin.assembly_report.accuracy_supported
        & linear_result.successful
        & linear_result.diagnostics.finite
        & finite
    )
    return ElasticityDirichletResult3D(
        traction_density=density,
        prescribed_displacement=values,
        potential=galerkin.potential(density),
        linear_result=linear_result,
        assembly_report=galerkin.assembly_report,
        nullspace=galerkin.nullspace,
        contract=galerkin.contract,
        boundary_residual_norm=residual_norm,
        finite=finite,
        valid=valid,
        formulation=(
            "interior displacement Dirichlet trace represented by a static Kelvin "
            "single layer; no Neumann, contact, fracture, or dynamic route"
        ),
    )


__all__ = [
    "ElasticityDirichletResult3D",
    "solve_elasticity_interior_displacement_dirichlet_3d",
]

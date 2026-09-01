#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
)
from ..operators.integral.layer_potential._stokes3d import (
    StokesBoundaryContract3D,
    StokesLayerPotential3D,
    StokesNullspaceMetadata3D,
    StokesSingleLayerDP0AssemblyReport3D,
    StokesSingleLayerDP0Galerkin3D,
)


class StokesDirichletResult3D(StrictModule):
    """Bounded steady-Stokes closed-triangle velocity Dirichlet result.

    The three-dimensional incompressible velocity trace is represented by a
    DP0 Stokes single layer. The density is constrained to be surface-L2
    orthogonal to the outward-normal null density, fixing the one-dimensional
    interior pressure gauge. Provider, precision, resources, quadrature error,
    geometry envelope, and non-goals are carried in ``contract`` and
    ``assembly_report``; no continuum accuracy is claimed.
    """

    force_density: Array
    prescribed_velocity: Array
    potential: StokesLayerPotential3D
    linear_result: LinearSolveResult
    assembly_report: StokesSingleLayerDP0AssemblyReport3D
    nullspace: StokesNullspaceMetadata3D
    contract: StokesBoundaryContract3D = eqx.field(static=True)
    boundary_flux: Array
    flux_compatibility_multiplier: Array
    density_gauge_residual: Array
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


def solve_stokes_interior_velocity_dirichlet_3d(
    galerkin: StokesSingleLayerDP0Galerkin3D,
    boundary_velocity: ArrayLike,
    /,
    *,
    flux_tolerance: float = 1.0e-10,
    linear: LinearSolvePolicy | None = None,
) -> StokesDirichletResult3D:
    """Solve a flux-compatible interior velocity trace with a pressure gauge."""
    if not isinstance(galerkin, StokesSingleLayerDP0Galerkin3D):
        raise TypeError("galerkin must be StokesSingleLayerDP0Galerkin3D.")
    if not bool(galerkin.assembly_report.accuracy_supported):
        raise ValueError("Stokes Galerkin quadrature evidence does not meet policy.")
    values = jnp.asarray(boundary_velocity, dtype=galerkin.face_areas.dtype)
    if values.shape != (galerkin.face_count, 3):
        raise ValueError("boundary_velocity must have shape (face_count, 3).")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError("boundary_velocity must be finite.")
    tolerance = float(flux_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("flux_tolerance must be finite and nonnegative.")
    policy = _default_policy() if linear is None else linear
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear must be LinearSolvePolicy or None.")
    if policy.differentiation.mode != "none":
        raise ValueError("Stokes boundary solves require differentiation mode 'none'.")

    velocity_flat = values.reshape(-1)
    flux_vector = galerkin.nullspace.boundary_flux_functional
    flux = jnp.dot(flux_vector, velocity_flat)
    flux_scale = 1.0 + jnp.linalg.norm(flux_vector) * jnp.linalg.norm(velocity_flat)
    if not bool(jnp.abs(flux) <= tolerance * flux_scale):
        raise ValueError(
            "Interior incompressible Dirichlet velocity must have zero total outward flux."
        )
    if not isinstance(galerkin.weak_operator, DenseLinearOperator):
        raise TypeError(
            "Stokes bounded Dirichlet preparation requires a dense weak operator."
        )
    weak = galerkin.weak_operator.matrix
    density_gauge = flux_vector.astype(weak.dtype)
    dimension = weak.shape[0]
    kkt = jnp.zeros((dimension + 1, dimension + 1), dtype=weak.dtype)
    kkt = kkt.at[:dimension, :dimension].set(weak)
    kkt = kkt.at[:dimension, dimension].set(density_gauge)
    kkt = kkt.at[dimension, :dimension].set(density_gauge)
    mass = jnp.repeat(galerkin.face_areas, 3)
    right = jnp.concatenate((mass * velocity_flat, jnp.zeros((1,), dtype=weak.dtype)))
    linear_result = solve(
        LinearSystem(
            DenseLinearOperator(kkt),
            problem_id="steady-stokes-interior-velocity-dirichlet-gauged-3d",
        ),
        right,
        policy=policy,
    )
    solution = jnp.asarray(linear_result.value)
    density_flat = solution[:dimension]
    density = density_flat.reshape((galerkin.face_count, 3))
    compatibility_multiplier = solution[dimension]
    gauge_residual = jnp.dot(density_gauge, density_flat)
    residual = galerkin.strong_operator.mv(density_flat) - velocity_flat
    residual_norm = jnp.linalg.norm(residual)
    finite = (
        jnp.all(jnp.isfinite(density))
        & jnp.isfinite(flux)
        & jnp.isfinite(compatibility_multiplier)
        & jnp.isfinite(gauge_residual)
        & jnp.isfinite(residual_norm)
    )
    valid = (
        galerkin.assembly_report.accuracy_supported
        & linear_result.successful
        & linear_result.diagnostics.finite
        & finite
    )
    return StokesDirichletResult3D(
        force_density=density,
        prescribed_velocity=values,
        potential=galerkin.potential(density),
        linear_result=linear_result,
        assembly_report=galerkin.assembly_report,
        nullspace=galerkin.nullspace,
        contract=galerkin.contract,
        boundary_flux=flux,
        flux_compatibility_multiplier=compatibility_multiplier,
        density_gauge_residual=gauge_residual,
        boundary_residual_norm=residual_norm,
        finite=finite,
        valid=valid,
        formulation=(
            "interior velocity Dirichlet trace represented by a Stokes single layer; "
            "zero-flux compatibility and outward-normal density gauge enforced by KKT"
        ),
    )


__all__ = [
    "StokesDirichletResult3D",
    "solve_stokes_interior_velocity_dirichlet_3d",
]

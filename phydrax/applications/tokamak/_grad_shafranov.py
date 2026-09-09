#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse fixed-boundary Grad-Shafranov solve for prescribed toroidal current."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DifferentiationPolicy,
    FailurePolicy,
    GMRES,
    JacobiPreconditionerBuilder,
    LinearSolvePolicy,
    LinearSystem,
    PreconditioningPolicy,
    solve_checked,
    TolerancePolicy,
)
from ...sparse import EdgeRelation, SparseCoordinateOperator
from ._conventions import TokamakMagneticConvention


DEFAULT_VACUUM_PERMEABILITY_H_M = 1.25663706127e-6


class FixedBoundaryEquilibriumResult(StrictModule):
    poloidal_flux_wb_per_rad: Array
    toroidal_current_density_a_m2: Array
    total_plasma_current_a: Array
    pde_residual_norm: Array
    linear_residual_norm: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class FixedBoundaryGradShafranovPlan:
    r_m: np.ndarray
    z_m: np.ndarray
    convention: TokamakMagneticConvention
    vacuum_permeability_h_m: float = DEFAULT_VACUUM_PERMEABILITY_H_M
    relative_tolerance: float = 1.0e-10
    maximum_steps: int = 512
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        r = np.array(self.r_m, dtype=np.float64, copy=True)
        z = np.array(self.z_m, dtype=np.float64, copy=True)
        if r.ndim != 1 or z.ndim != 1 or r.size < 3 or z.size < 3:
            raise ValueError(
                "Grad-Shafranov coordinates require at least three points per axis."
            )
        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(z)) or np.any(r <= 0.0):
            raise ValueError("Grad-Shafranov coordinates must be finite with R > 0.")
        dr = np.diff(r)
        dz = np.diff(z)
        if np.any(dr <= 0.0) or np.any(dz <= 0.0):
            raise ValueError("Grad-Shafranov coordinates must increase strictly.")
        tolerance = 512.0 * np.finfo(float).eps
        if not np.allclose(
            dr, dr[0], rtol=tolerance, atol=tolerance * max(1.0, abs(dr[0]))
        ):
            raise ValueError("Initial Grad-Shafranov support requires a uniform R axis.")
        if not np.allclose(
            dz, dz[0], rtol=tolerance, atol=tolerance * max(1.0, abs(dz[0]))
        ):
            raise ValueError("Initial Grad-Shafranov support requires a uniform Z axis.")
        if not isinstance(self.convention, TokamakMagneticConvention):
            raise TypeError("convention must be TokamakMagneticConvention.")
        if (
            self.convention.convention_id
            != TokamakMagneticConvention.canonical().convention_id
        ):
            raise ValueError(
                "Native Grad-Shafranov solve requires the canonical Phydrax convention."
            )
        permeability = float(self.vacuum_permeability_h_m)
        relative = float(self.relative_tolerance)
        steps = int(self.maximum_steps)
        if not math.isfinite(permeability) or permeability <= 0.0:
            raise ValueError("vacuum_permeability_h_m must be finite and positive.")
        if not math.isfinite(relative) or relative <= 0.0 or steps < 1:
            raise ValueError("Grad-Shafranov solve controls are invalid.")
        r.setflags(write=False)
        z.setflags(write=False)
        object.__setattr__(self, "r_m", r)
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "vacuum_permeability_h_m", permeability)
        object.__setattr__(self, "relative_tolerance", relative)
        object.__setattr__(self, "maximum_steps", steps)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "fixed-boundary-grad-shafranov-plan",
                    "r_m": array_tree_fingerprint(r),
                    "z_m": array_tree_fingerprint(z),
                    "convention": self.convention.convention_id,
                    "vacuum_permeability_h_m": permeability,
                    "relative_tolerance": relative,
                    "maximum_steps": steps,
                    "source": "prescribed-toroidal-current-density",
                }
            ),
        )

    def prepare(self) -> PreparedFixedBoundaryGradShafranov:
        nr = self.r_m.size
        nz = self.z_m.size
        interior_nr = nr - 2
        interior_nz = nz - 2
        size = interior_nr * interior_nz
        dr = float(self.r_m[1] - self.r_m[0])
        dz = float(self.z_m[1] - self.z_m[0])
        source_indices = []
        target_indices = []
        coefficients = []
        boundary_rows = []
        boundary_z = []
        boundary_r = []
        boundary_coefficients = []

        def interior_index(j, i):
            return (j - 1) * interior_nr + (i - 1)

        for j in range(1, nz - 1):
            for i in range(1, nr - 1):
                row = interior_index(j, i)
                radius = self.r_m[i]
                center = 2.0 / dr**2 + 2.0 / dz**2
                neighbors = (
                    (j, i - 1, -(1.0 / dr**2 + 1.0 / (2.0 * radius * dr))),
                    (j, i + 1, -(1.0 / dr**2 - 1.0 / (2.0 * radius * dr))),
                    (j - 1, i, -1.0 / dz**2),
                    (j + 1, i, -1.0 / dz**2),
                )
                source_indices.append(row)
                target_indices.append(row)
                coefficients.append(center)
                for neighbor_j, neighbor_i, coefficient in neighbors:
                    if 0 < neighbor_i < nr - 1 and 0 < neighbor_j < nz - 1:
                        source_indices.append(interior_index(neighbor_j, neighbor_i))
                        target_indices.append(row)
                        coefficients.append(coefficient)
                    else:
                        boundary_rows.append(row)
                        boundary_z.append(neighbor_j)
                        boundary_r.append(neighbor_i)
                        boundary_coefficients.append(coefficient)
        relation = EdgeRelation(
            jnp.asarray(source_indices, dtype=jnp.int32),
            jnp.asarray(target_indices, dtype=jnp.int32),
            source_size=size,
            target_size=size,
        )
        space = ArraySpace((size,), dtype=np.float64)
        operator = SparseCoordinateOperator(
            relation,
            jnp.asarray(coefficients),
            source=space,
            target=space,
            operator_id=self.plan_id,
        )
        policy = LinearSolvePolicy(
            GMRES(restart=min(64, size)),
            tolerance=TolerancePolicy(
                relative=self.relative_tolerance,
                absolute=self.relative_tolerance,
                max_steps=self.maximum_steps,
            ),
            preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
            differentiation=DifferentiationPolicy("mathematical"),
            failure=FailurePolicy("status"),
        )
        return PreparedFixedBoundaryGradShafranov(
            jnp.asarray(self.r_m),
            jnp.asarray(self.z_m),
            operator,
            jnp.asarray(boundary_rows, dtype=jnp.int32),
            jnp.asarray(boundary_z, dtype=jnp.int32),
            jnp.asarray(boundary_r, dtype=jnp.int32),
            jnp.asarray(boundary_coefficients),
            self.vacuum_permeability_h_m,
            dr,
            dz,
            policy,
            self.plan_id,
        )


class PreparedFixedBoundaryGradShafranov(StrictModule, NonTrainableState):
    r_m: Array
    z_m: Array
    operator: SparseCoordinateOperator
    boundary_rows: Array
    boundary_z: Array
    boundary_r: Array
    boundary_coefficients: Array
    vacuum_permeability_h_m: float = eqx.field(static=True)
    dr_m: float = eqx.field(static=True)
    dz_m: float = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def solve(
        self,
        toroidal_current_density_a_m2: ArrayLike,
        boundary_flux_wb_per_rad: ArrayLike,
        /,
    ) -> FixedBoundaryEquilibriumResult:
        current = jnp.asarray(toroidal_current_density_a_m2, dtype=jnp.float64)
        boundary = jnp.asarray(boundary_flux_wb_per_rad, dtype=current.dtype)
        shape = (self.z_m.size, self.r_m.size)
        if current.shape != shape or boundary.shape != shape:
            raise ValueError(
                f"Current density and boundary flux must have shape {shape}."
            )
        boundary_mask = jnp.ones(shape, dtype=bool).at[1:-1, 1:-1].set(False)
        domain_valid = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(boundary))
            & jnp.all(jnp.where(boundary_mask, current == 0.0, True))
        )
        interior_current = current[1:-1, 1:-1].reshape((-1,))
        radii = jnp.broadcast_to(self.r_m[1:-1], current[1:-1, 1:-1].shape).reshape((-1,))
        rhs = self.vacuum_permeability_h_m * radii * interior_current
        boundary_values = boundary[self.boundary_z, self.boundary_r]
        rhs = rhs.at[self.boundary_rows].add(
            -self.boundary_coefficients * boundary_values
        )
        linear, evidence = solve_checked(
            LinearSystem(self.operator),
            rhs,
            policy=self.linear_policy,
        )
        candidate = boundary.at[1:-1, 1:-1].set(
            linear.value.reshape((self.z_m.size - 2, self.r_m.size - 2))
        )
        residual = self.operator.mv(linear.value) - rhs
        pde_residual = jnp.sqrt(jnp.vdot(residual, residual))
        total_current = jnp.sum(current[1:-1, 1:-1]) * self.dr_m * self.dz_m
        finite = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.isfinite(pde_residual)
            & jnp.isfinite(total_current)
        )
        successful = domain_valid & linear.successful & evidence.valid & finite
        return FixedBoundaryEquilibriumResult(
            candidate,
            current,
            total_current,
            pde_residual,
            evidence.true_residual_norm,
            finite,
            domain_valid,
            successful,
            successful,
            self.plan_id,
        )


__all__ = [
    "DEFAULT_VACUUM_PERMEABILITY_H_M",
    "FixedBoundaryEquilibriumResult",
    "FixedBoundaryGradShafranovPlan",
    "PreparedFixedBoundaryGradShafranov",
]

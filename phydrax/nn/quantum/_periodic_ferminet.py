#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resource-admitted finite periodic determinant amplitude with twist covariance."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ...operators.quantum._amplitude import LogAmplitude
from ...operators.quantum._electronic_advanced import ElectronicVMCResourcePlan


class PeriodicFermiNet(StrictModule):
    """Finite reciprocal-feature determinant network in fractional coordinates.

    This is a bounded periodic Slater/Jastrow amplitude, not an unrestricted
    thermodynamic-limit or continuum-complete representation.
    """

    reciprocal_vectors: Array
    orbital_coefficients: Array
    determinant_coefficients: Array
    twist: Array
    pair_jastrow_strength: Array
    resource_plan: ElectronicVMCResourcePlan
    electron_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        reciprocal_vectors: ArrayLike,
        orbital_coefficients: ArrayLike,
        determinant_coefficients: ArrayLike,
        /,
        *,
        twist: ArrayLike,
        pair_jastrow_strength: ArrayLike = 0.0,
        resource_plan: ElectronicVMCResourcePlan,
    ):
        reciprocal = jnp.asarray(reciprocal_vectors)
        coefficients = jnp.asarray(orbital_coefficients)
        mixing = jnp.asarray(determinant_coefficients)
        twist_ = jnp.asarray(twist)
        if reciprocal.ndim != 2 or reciprocal.shape[0] < 1:
            raise ValueError("reciprocal_vectors must have shape (modes, dimension).")
        if coefficients.ndim != 3 or coefficients.shape[2] != reciprocal.shape[0]:
            raise ValueError(
                "orbital_coefficients require (determinants,electrons,modes)."
            )
        determinants, electrons = map(int, coefficients.shape[:2])
        if mixing.shape != (determinants,) or twist_.shape != (reciprocal.shape[1],):
            raise ValueError(
                "determinant coefficients/twist dimensions are inconsistent."
            )
        if not isinstance(resource_plan, ElectronicVMCResourcePlan):
            raise TypeError("resource_plan must be ElectronicVMCResourcePlan.")
        if (
            resource_plan.electron_count != electrons
            or resource_plan.determinant_count != determinants
        ):
            raise ValueError("PeriodicFermiNet counts must match resource_plan.")
        dtype = jnp.result_type(coefficients.dtype, mixing.dtype, 1j)
        self.reciprocal_vectors = reciprocal
        self.orbital_coefficients = coefficients.astype(dtype)
        self.determinant_coefficients = mixing.astype(dtype)
        self.twist = twist_
        self.pair_jastrow_strength = jnp.asarray(pair_jastrow_strength)
        self.resource_plan = resource_plan
        self.electron_count = electrons
        self.determinant_count = determinants
        self.spatial_dimension = int(reciprocal.shape[1])
        self.claim = "finite-reciprocal-feature-periodic-amplitude"

    def __call__(self, fractional_coordinates: ArrayLike, /) -> LogAmplitude:
        coordinates = jnp.asarray(fractional_coordinates)
        if coordinates.shape != (self.electron_count, self.spatial_dimension):
            raise ValueError(
                "fractional_coordinates shape must match the admitted finite system."
            )
        phases = 2.0 * jnp.pi * (coordinates @ self.reciprocal_vectors.T)
        features = jnp.exp(1j * phases)
        determinant_logs = []
        determinant_phases = []
        for coefficients in self.orbital_coefficients:
            orbitals = contract("im,jm->ij", features, coefficients)
            prepared = factorize(
                DenseLinearOperator(orbitals),
                FactorizationPolicy("lu"),
            )
            determinant_logs.append(prepared.log_abs_determinant())
            determinant_phases.append(prepared.determinant_sign())
        logs = jnp.stack(determinant_logs)
        phases_ = jnp.stack(determinant_phases)
        reference = jnp.max(logs)
        mixture = jnp.sum(
            self.determinant_coefficients * phases_ * jnp.exp(logs - reference)
        )
        minimum_image = coordinates[:, None, :] - coordinates[None, :, :]
        minimum_image = minimum_image - jnp.round(minimum_image)
        pair_mask = jnp.triu(
            jnp.ones((self.electron_count, self.electron_count), dtype=bool),
            k=1,
        )
        pair_distance = jnp.sqrt(jnp.sum(minimum_image * minimum_image, axis=-1))
        jastrow = self.pair_jastrow_strength * jnp.sum(
            jnp.where(pair_mask, pair_distance, 0.0)
        )
        twist_phase = contract("d,id->", self.twist, coordinates)
        magnitude = jnp.abs(mixture)
        phase = jnp.where(
            magnitude > 0.0,
            mixture / magnitude * jnp.exp(1j * twist_phase),
            1.0 + 0.0j,
        )
        log_abs = reference + jnp.log(magnitude) + jnp.real(jastrow)
        valid = (
            self.resource_plan.valid
            & jnp.all(jnp.isfinite(coordinates))
            & jnp.isfinite(log_abs)
            & (magnitude > 0.0)
        )
        return LogAmplitude(log_abs, phase, valid=valid)


__all__ = ["PeriodicFermiNet"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit molecular integral definitions and J/K representation admission.

Fixed factors and screened quartets are distinct Hamiltonian definitions. They
must not inherit nuclear derivatives or response from the exact dense kernels.
cuEST definitions also bind the actual dynamic numerical recipe. Its prepared
source explicitly binds one geometry before repeated J/K and contracted-force
operations; native SCF owns that binding, never an end-to-end vendor SCF call.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....ein import contract
from ._basis import PreparedGaussianBasis
from ._cuest import CuestIntegralPlan, PreparedCuestIntegrals
from ._direct import PreparedDirectJK
from ._factorization import FactorizedERITensor
from ._integrals import (
    electron_repulsion_tensor,
    kinetic_matrix,
    nuclear_attraction_matrix,
    nuclear_repulsion_energy,
    overlap_matrix,
)


class MolecularOneElectronResult(StrictModule):
    overlap: Array
    core_hamiltonian: Array
    nuclear_repulsion: Array
    successful: Array
    definition_id: str = eqx.field(static=True)
    geometry: Array
    kinetic: Array
    nuclear_attraction: Array


class MolecularDensityFactorizationEvidence(StrictModule):
    """PSD certificate; unmeasured eigenvalue/magnitude are NaN, never fabricated."""

    minimum_eigenvalue: Array
    reconstruction_residual: Array
    tolerance: Array
    negative_roundoff_magnitude: Array
    factor_bytes: Array
    used_eigensolver: Array
    minimum_eigenvalue_lower_bound: Array


class MolecularJKResult(StrictModule):
    coulomb: Array
    exchange: Array
    successful: Array
    definition_id: str = eqx.field(static=True)
    geometry: Array
    factorization: MolecularDensityFactorizationEvidence | None = None


def molecular_integral_definition_id(
    basis: PreparedGaussianBasis,
    source: FactorizedERITensor | PreparedDirectJK | CuestIntegralPlan | None = None,
    /,
) -> str:
    """Bind an explicitly selected integral source to the canonical AO basis."""
    if isinstance(source, CuestIntegralPlan):
        return source.definition_id(basis)
    plan = MolecularIntegralPlan(source)
    plan.validate_basis(basis)
    return canonical_fingerprint(
        {
            "kind": "molecular-integral-definition",
            "basis": basis.prepared_id,
            "plan": plan.plan_id,
        }
    )


class MolecularIntegralPlan(StrictModule, NonTrainableState):
    """Select exact dense, screened direct, fixed factors, or explicit cuEST DF.

    Passing factors explicitly binds their AO ordering to the prepared basis at
    the consumer boundary; equal dimensions alone are not an identity claim.
    Fixed factors have no moving-geometry derivative contract. Direct screening
    retains its own geometry validity test and does not imply exact response.
    """

    source: FactorizedERITensor | PreparedDirectJK | CuestIntegralPlan | None
    representation: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: FactorizedERITensor | PreparedDirectJK | CuestIntegralPlan | None = None,
        /,
    ) -> None:
        if source is None:
            representation, source_id = "exact", None
        elif isinstance(source, FactorizedERITensor):
            representation, source_id = "factorized", source.tensor_id
        elif isinstance(source, PreparedDirectJK):
            representation, source_id = "direct", source.prepared_id
        elif isinstance(source, CuestIntegralPlan):
            representation, source_id = "density-fitting", source.plan_id
        else:
            raise TypeError(
                "Integral source must be fixed factors, prepared direct J/K, cuEST, or None."
            )
        self.source = source
        self.representation = representation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-integral-plan",
                "representation": representation,
                "source": source_id,
            }
        )

    def validate_basis(self, basis: PreparedGaussianBasis, /) -> None:
        if not isinstance(basis, PreparedGaussianBasis):
            raise TypeError("Molecular integrals require a prepared Gaussian basis.")
        if isinstance(self.source, FactorizedERITensor):
            if self.source.orbital_count != basis.basis_function_count:
                raise ValueError("Factorized ERIs do not align with the Gaussian basis.")
        elif isinstance(self.source, PreparedDirectJK):
            if self.source.basis.prepared_id != basis.prepared_id:
                raise ValueError("Direct J/K belongs to another Gaussian basis.")
        elif isinstance(self.source, CuestIntegralPlan):
            self.source.validate_basis(basis)

    def definition_id(self, basis: PreparedGaussianBasis, /) -> str:
        self.validate_basis(basis)
        return molecular_integral_definition_id(basis, self.source)

    def require_exact(self, operation: str, /) -> None:
        if self.source is not None:
            raise NotImplementedError(
                f"{operation} requires the same integral definition as the mean-field energy; "
                f"the {self.representation} representation does not supply this operation."
            )

    def prepare(self, basis: PreparedGaussianBasis, /) -> PreparedMolecularIntegrals:
        return PreparedMolecularIntegrals(self, basis)

    def one_electron(
        self, basis: PreparedGaussianBasis, positions: ArrayLike, charges: ArrayLike, /
    ) -> MolecularOneElectronResult:
        positions, charges = jnp.asarray(positions), jnp.asarray(charges)
        if isinstance(self.source, CuestIntegralPlan):
            return self.source.one_electron(basis, positions, charges)
        definition = self.definition_id(basis)
        overlap = overlap_matrix(basis, positions)
        kinetic = kinetic_matrix(basis, positions)
        attraction = nuclear_attraction_matrix(basis, positions, charges)
        core = kinetic + attraction
        nuclear = nuclear_repulsion_energy(positions, charges)
        return MolecularOneElectronResult(
            overlap,
            core,
            nuclear,
            jnp.all(jnp.isfinite(overlap))
            & jnp.all(jnp.isfinite(core))
            & jnp.isfinite(nuclear),
            definition,
            positions,
            kinetic,
            attraction,
        )

    def dense_tensor(
        self, basis: PreparedGaussianBasis, positions: ArrayLike, /
    ) -> Array | None:
        self.validate_basis(basis)
        return (
            electron_repulsion_tensor(basis, jnp.asarray(positions))
            if self.source is None
            else None
        )

    def jk(
        self,
        basis: PreparedGaussianBasis,
        positions: ArrayLike,
        density: ArrayLike,
        dense: Array | None,
        /,
    ) -> MolecularJKResult:
        positions, density = jnp.asarray(positions), jnp.asarray(density)
        if isinstance(self.source, CuestIntegralPlan):
            return self.source.jk(basis, positions, density)
        definition = self.definition_id(basis)
        if isinstance(self.source, FactorizedERITensor):
            coulomb = self.source.coulomb(density)
            exchange = self.source.exchange(density)
            successful = jnp.all(jnp.isfinite(coulomb)) & jnp.all(jnp.isfinite(exchange))
        elif isinstance(self.source, PreparedDirectJK):
            result = self.source.evaluate(positions, density)
            coulomb, exchange, successful = (
                result.coulomb,
                result.exchange,
                result.successful,
            )
        else:
            if dense is None:
                raise RuntimeError("Exact dense integral source is unexpectedly absent.")
            coulomb = contract("cd,abcd->ab", density, dense)
            exchange = contract("cd,acbd->ab", density, dense)
            successful = jnp.all(jnp.isfinite(coulomb)) & jnp.all(jnp.isfinite(exchange))
        return MolecularJKResult(coulomb, exchange, successful, definition, positions)


class PreparedMolecularIntegrals(StrictModule, NonTrainableState):
    """Canonical basis-bound integral source; numerical leaves stay dynamic."""

    plan: MolecularIntegralPlan
    basis: PreparedGaussianBasis
    vendor: PreparedCuestIntegrals | None
    definition_id: str = eqx.field(static=True)

    def __init__(
        self, plan: MolecularIntegralPlan, basis: PreparedGaussianBasis, /
    ) -> None:
        plan.validate_basis(basis)
        self.plan = plan
        self.basis = basis
        self.vendor = (
            plan.source.prepare(basis)
            if isinstance(plan.source, CuestIntegralPlan)
            else None
        )
        self.definition_id = (
            self.vendor.definition_id
            if self.vendor is not None
            else plan.definition_id(basis)
        )

    def one_electron(
        self, positions: ArrayLike, charges: ArrayLike, /
    ) -> MolecularOneElectronResult:
        if self.vendor is not None:
            return self.vendor.bind_geometry(positions, charges).one_electron()
        return self.plan.one_electron(self.basis, positions, charges)

    def dense_tensor(self, positions: ArrayLike, /) -> Array | None:
        return (
            electron_repulsion_tensor(self.basis, jnp.asarray(positions))
            if self.plan.source is None
            else None
        )

    def jk(
        self,
        positions: ArrayLike,
        density: ArrayLike,
        dense: Array | None,
        /,
        *,
        factors: Array | None = None,
    ) -> MolecularJKResult:
        if self.vendor is not None:
            coordinate = jnp.asarray(positions)
            return self.vendor.bind_geometry(
                coordinate, jnp.zeros((coordinate.shape[0],), dtype=jnp.float64)
            ).jk(density, factors=factors)
        return self.plan.jk(self.basis, positions, density, dense)


__all__ = [
    "MolecularDensityFactorizationEvidence",
    "MolecularIntegralPlan",
    "MolecularJKResult",
    "MolecularOneElectronResult",
    "PreparedMolecularIntegrals",
]

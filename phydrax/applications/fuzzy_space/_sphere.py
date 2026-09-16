#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Two-particle lowest-Landau-level fuzzy-sphere pseudopotential model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...operators.quantum.lattice import (
    prepare_su2_sector_basis,
    PreparedSU2SectorBasis,
    SU2CouplingTreePlan,
    SU2SectorResourcePolicy,
)
from ...tensor_network import su2_fusion


FuzzyParticleStatistics: TypeAlias = Literal["boson", "fermion"]


class FuzzySphereTwoParticlePlan(StrictModule):
    """Rotationally invariant two-particle Haldane pseudopotentials on S²_F."""

    pseudopotentials: tuple[tuple[int, float], ...] = eqx.field(static=True)
    allowed_total_twice_spins: tuple[int, ...] = eqx.field(static=True)
    statistics: FuzzyParticleStatistics = eqx.field(static=True)
    twice_monopole_flux: int = eqx.field(static=True)
    orbital_count: int = eqx.field(static=True)
    resources: SU2SectorResourcePolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        twice_monopole_flux: int,
        statistics: FuzzyParticleStatistics,
        pseudopotentials: Mapping[int, float],
        resources: SU2SectorResourcePolicy,
        /,
    ):
        flux = int(twice_monopole_flux)
        statistics_value = str(statistics)
        values = tuple(
            sorted((int(spin), float(value)) for spin, value in pseudopotentials.items())
        )
        if flux < 1 or statistics_value not in {"boson", "fermion"}:
            raise ValueError("Fuzzy-sphere flux/statistics are invalid.")
        if not isinstance(resources, SU2SectorResourcePolicy):
            raise TypeError("resources must be SU2SectorResourcePolicy.")
        all_spins = su2_fusion(flux, flux)
        allowed = tuple(
            total
            for total in all_spins
            if (((2 * flux - total) // 2) % 2 == 0) == (statistics_value == "boson")
        )
        supplied = tuple(spin for spin, _ in values)
        if supplied != allowed:
            raise ValueError(
                "Pseudopotentials must specify every and only statistics-allowed pair spin."
            )
        if any(not np.isfinite(value) for _, value in values):
            raise ValueError("Pseudopotential values must be finite.")
        self.pseudopotentials = values
        self.allowed_total_twice_spins = allowed
        self.statistics = statistics_value
        self.twice_monopole_flux = flux
        self.orbital_count = flux + 1
        self.resources = resources
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-particle-fuzzy-sphere-pseudopotential-plan",
                "twice_monopole_flux": flux,
                "statistics": statistics_value,
                "pseudopotentials": values,
                "allowed_total_twice_spins": allowed,
                "resources": resources.policy_id,
                "convention": "lowest-landau-level-condon-shortley-pair-spin",
            }
        )


class FuzzySphereQualificationEvidence(StrictModule):
    transform_orthonormality_residual: Array
    physical_projector_idempotence_residual: Array
    rotational_commutator_residual: Array
    exchange_sector_complete: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class PreparedFuzzySphereTwoParticleModel(StrictModule):
    plan: FuzzySphereTwoParticlePlan = eqx.field(static=True)
    sector_bases: tuple[PreparedSU2SectorBasis, ...]
    physical_transform: Array
    product_hamiltonian: Array
    physical_hamiltonian: Array
    total_spin_squared: Array
    evidence: FuzzySphereQualificationEvidence
    prepared_id: str = eqx.field(static=True)


class FuzzySphereSpectrumResult(StrictModule):
    energies: Array
    total_twice_spins: Array
    magnetic_twice_projections: Array
    degeneracies: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_fuzzy_sphere_two_particle(
    plan: FuzzySphereTwoParticlePlan, /
) -> PreparedFuzzySphereTwoParticleModel:
    if not isinstance(plan, FuzzySphereTwoParticlePlan):
        raise TypeError("plan must be FuzzySphereTwoParticlePlan.")
    sector_bases = tuple(
        prepare_su2_sector_basis(
            SU2CouplingTreePlan(
                ("particle-0", "particle-1"),
                (plan.twice_monopole_flux, plan.twice_monopole_flux),
                total,
                plan.resources,
            )
        )
        for total in plan.allowed_total_twice_spins
    )
    transform = np.concatenate(
        tuple(np.asarray(value.transform) for value in sector_bases), axis=0
    )
    physical_dimension = int(transform.shape[0])
    if physical_dimension**2 > plan.resources.maximum_matrix_elements:
        raise ValueError("Fuzzy-sphere physical Hamiltonian exceeds matrix admission.")
    energies = np.concatenate(
        tuple(
            np.full((basis.plan.sector_dimension,), dict(plan.pseudopotentials)[total])
            for total, basis in zip(
                plan.allowed_total_twice_spins, sector_bases, strict=True
            )
        )
    )
    physical_hamiltonian = np.diag(energies)
    product_hamiltonian = np.conj(transform.T) @ physical_hamiltonian @ transform
    spin_squared_values = np.concatenate(
        tuple(
            np.full(
                (basis.plan.sector_dimension,),
                0.25 * total * (total + 2),
            )
            for total, basis in zip(
                plan.allowed_total_twice_spins, sector_bases, strict=True
            )
        )
    )
    total_spin_squared = np.diag(spin_squared_values)
    gram = transform @ np.conj(transform.T)
    projector = np.conj(transform.T) @ transform
    orthonormality = float(
        np.linalg.norm(gram - np.eye(physical_dimension))
        / max(1.0, float(np.linalg.norm(np.eye(physical_dimension))))
    )
    idempotence = float(
        np.linalg.norm(projector @ projector - projector)
        / max(1.0, float(np.linalg.norm(projector)))
    )
    commutator = (
        physical_hamiltonian @ total_spin_squared
        - total_spin_squared @ physical_hamiltonian
    )
    rotational = float(
        np.linalg.norm(commutator) / max(1.0, float(np.linalg.norm(physical_hamiltonian)))
    )
    expected_dimension = (
        plan.orbital_count
        * (plan.orbital_count + (1 if plan.statistics == "boson" else -1))
    ) // 2
    complete = physical_dimension == expected_dimension
    finite = bool(
        np.all(np.isfinite(transform))
        and np.all(np.isfinite(product_hamiltonian))
        and np.all(np.isfinite(physical_hamiltonian))
    )
    accepted = (
        finite
        and complete
        and orthonormality <= 1e-11
        and idempotence <= 1e-11
        and rotational <= 1e-11
    )
    evidence = FuzzySphereQualificationEvidence(
        transform_orthonormality_residual=jnp.asarray(orthonormality),
        physical_projector_idempotence_residual=jnp.asarray(idempotence),
        rotational_commutator_residual=jnp.asarray(rotational),
        exchange_sector_complete=jnp.asarray(complete),
        finite=jnp.asarray(finite),
        accepted=jnp.asarray(accepted),
        plan_id=plan.plan_id,
        claim="finite-two-particle-lowest-landau-level-fuzzy-sphere-reference",
    )
    return PreparedFuzzySphereTwoParticleModel(
        plan=plan,
        sector_bases=sector_bases,
        physical_transform=jnp.asarray(transform),
        product_hamiltonian=jnp.asarray(product_hamiltonian),
        physical_hamiltonian=jnp.asarray(physical_hamiltonian),
        total_spin_squared=jnp.asarray(total_spin_squared),
        evidence=evidence,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-two-particle-fuzzy-sphere-model",
                "plan": plan.plan_id,
                "transform": array_tree_fingerprint(transform),
                "hamiltonian": array_tree_fingerprint(physical_hamiltonian),
            }
        ),
    )


def fuzzy_sphere_spectrum(
    prepared: PreparedFuzzySphereTwoParticleModel, /
) -> FuzzySphereSpectrumResult:
    if not isinstance(prepared, PreparedFuzzySphereTwoParticleModel):
        raise TypeError("prepared must be PreparedFuzzySphereTwoParticleModel.")
    energies = []
    spins = []
    projections = []
    degeneracies = []
    values = dict(prepared.plan.pseudopotentials)
    for total in prepared.plan.allowed_total_twice_spins:
        degeneracy = total + 1
        for projection in range(-total, total + 1, 2):
            energies.append(values[total])
            spins.append(total)
            projections.append(projection)
            degeneracies.append(degeneracy)
    energy_array = jnp.asarray(energies)
    return FuzzySphereSpectrumResult(
        energies=energy_array,
        total_twice_spins=jnp.asarray(spins, dtype=jnp.int32),
        magnetic_twice_projections=jnp.asarray(projections, dtype=jnp.int32),
        degeneracies=jnp.asarray(degeneracies, dtype=jnp.int32),
        finite=jnp.all(jnp.isfinite(energy_array)),
        prepared_id=prepared.prepared_id,
        claim="finite-cutoff-two-particle-spectrum-no-continuum-cft-identification",
    )


__all__ = [
    "FuzzyParticleStatistics",
    "FuzzySphereQualificationEvidence",
    "FuzzySphereSpectrumResult",
    "FuzzySphereTwoParticlePlan",
    "PreparedFuzzySphereTwoParticleModel",
    "fuzzy_sphere_spectrum",
    "prepare_fuzzy_sphere_two_particle",
]

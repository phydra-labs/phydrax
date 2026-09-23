#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit finite multi-Landau-level and multicomponent sphere ED."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.quantum import AbelianGroup, FermionModeOrder
from ...operators.quantum.lattice import (
    FixedAbelianChargeBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    PreparedQuantumLattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)
from ._identity import HallChargeSector, MonopoleLandauLevel, MonopoleOrbitalKey


@dataclass(frozen=True, slots=True)
class ProjectedOrbitalTerm:
    creators: tuple[int, ...]
    annihilators: tuple[int, ...]
    coefficient: complex
    term_id: str

    def __init__(
        self,
        creators: Sequence[int],
        annihilators: Sequence[int],
        coefficient: complex,
    ):
        creators_ = tuple(int(value) for value in creators)
        annihilators_ = tuple(int(value) for value in annihilators)
        coefficient_ = complex(coefficient)
        if (
            not creators_
            or len(creators_) != len(annihilators_)
            or len(set(creators_)) != len(creators_)
            or len(set(annihilators_)) != len(annihilators_)
            or not np.isfinite(coefficient_)
        ):
            raise ValueError("Projected orbital term is invalid.")
        identifier = canonical_fingerprint(
            {
                "kind": "projected-orbital-term",
                "creators": creators_,
                "annihilators": annihilators_,
                "coefficient": (coefficient_.real, coefficient_.imag),
            }
        )
        object.__setattr__(self, "creators", creators_)
        object.__setattr__(self, "annihilators", annihilators_)
        object.__setattr__(self, "coefficient", coefficient_)
        object.__setattr__(self, "term_id", identifier)


class MultiLandauLevelSpherePlan(StrictModule, NonTrainableState):
    particle_count: int = eqx.field(static=True)
    manifolds: tuple[MonopoleLandauLevel, ...]
    sector: HallChargeSector
    terms: tuple[ProjectedOrbitalTerm, ...] = eqx.field(static=True)
    maximum_basis_dimension: int = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    maximum_table_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_count: int,
        manifolds: Sequence[MonopoleLandauLevel],
        sector: HallChargeSector,
        terms: Sequence[ProjectedOrbitalTerm],
        /,
        *,
        maximum_basis_dimension: int = 100_000,
        maximum_terms: int = 2_000_000,
        maximum_table_bytes: int = 128 * 1024 * 1024,
    ):
        particles = int(particle_count)
        manifolds_ = tuple(manifolds)
        terms_ = tuple(terms)
        maximum_basis = int(maximum_basis_dimension)
        maximum_terms_ = int(maximum_terms)
        maximum_table = int(maximum_table_bytes)
        if not manifolds_ or any(
            not isinstance(value, MonopoleLandauLevel) for value in manifolds_
        ):
            raise TypeError("manifolds must contain MonopoleLandauLevel values.")
        if not isinstance(sector, HallChargeSector):
            raise TypeError("sector must be HallChargeSector.")
        if particles < 1 or particles > sum(value.orbital_count for value in manifolds_):
            raise ValueError("Multi-level particle count is outside the orbital roster.")
        if len({value.manifold_id for value in manifolds_}) != len(manifolds_):
            raise ValueError("Multi-level manifold identities must be unique.")
        fluxes = {value.twice_monopole_strength for value in manifolds_}
        if len(fluxes) != 1:
            raise ValueError(
                "All explicit manifolds must share physical monopole strength."
            )
        orbital_count = sum(value.orbital_count for value in manifolds_)
        if any(
            not isinstance(value, ProjectedOrbitalTerm)
            or any(
                index < 0 or index >= orbital_count
                for index in value.creators + value.annihilators
            )
            for value in terms_
        ):
            raise ValueError("Projected terms reference unavailable orbitals.")
        if len({value.term_id for value in terms_}) != len(terms_):
            raise ValueError("Projected orbital term identities must be unique.")
        if maximum_basis < 1 or maximum_terms_ < len(terms_) or maximum_table < 1:
            raise ValueError("Multi-level resource limits are invalid.")
        self.particle_count = particles
        self.manifolds = manifolds_
        self.sector = sector
        self.terms = terms_
        self.maximum_basis_dimension = maximum_basis
        self.maximum_terms = maximum_terms_
        self.maximum_table_bytes = maximum_table
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multi-landau-level-sphere-plan",
                "particle_count": particles,
                "manifolds": tuple(value.manifold_id for value in self.manifolds),
                "sector": sector.sector_id,
                "terms": tuple(
                    {
                        "id": value.term_id,
                        "creators": value.creators,
                        "annihilators": value.annihilators,
                        "coefficient": (value.coefficient.real, value.coefficient.imag),
                    }
                    for value in terms_
                ),
                "resources": (maximum_basis, maximum_terms_, maximum_table),
            }
        )

    @property
    def orbitals(self) -> tuple[MonopoleOrbitalKey, ...]:
        return tuple(key for manifold in self.manifolds for key in manifold.orbital_keys)


class PreparedMultiLandauLevelSphere(StrictModule, NonTrainableState):
    plan: MultiLandauLevelSpherePlan
    prepared_lattice: PreparedQuantumLattice
    basis: FixedAbelianChargeBasis
    operator: QuantumSectorOperator
    prepared_id: str = eqx.field(static=True)


def _mode_charge(label: str, orbital: MonopoleOrbitalKey, /) -> int:
    if label == "particle-number":
        return 1
    if label == "twice-projection":
        return orbital.twice_orbital_projection
    if label == "twice-spin":
        return (
            0
            if orbital.component.twice_spin_projection is None
            else orbital.component.twice_spin_projection
        )
    if label == "landau-excitation":
        return orbital.landau_level
    prefix = "component:"
    if label.startswith(prefix):
        return int(orbital.component.component_id == label[len(prefix) :])
    raise ValueError(f"Unsupported multi-level sector charge {label!r}.")


def prepare_multi_landau_level_sphere(
    plan: MultiLandauLevelSpherePlan,
    /,
) -> PreparedMultiLandauLevelSphere:
    if not isinstance(plan, MultiLandauLevelSpherePlan):
        raise TypeError("plan must be MultiLandauLevelSpherePlan.")
    orbitals = plan.orbitals
    labels = tuple(key.orbital_id for key in orbitals)
    mode_order = FermionModeOrder(labels)
    charge_labels = tuple(key for key, _ in plan.sector.targets)
    moduli = dict(plan.sector.moduli)
    group = AbelianGroup(tuple(moduli[key] for key in charge_labels))
    spaces = tuple(
        LocalSpacePlan(
            label,
            ("empty", "occupied"),
            charge_labels,
            (
                (0,) * len(charge_labels),
                tuple(_mode_charge(charge, orbital) for charge in charge_labels),
            ),
            statistics="fermion",
            fermion_mode_label=label,
        )
        for label, orbital in zip(labels, orbitals, strict=True)
    )
    annihilation = tuple(
        LocalOperatorPlan(
            space,
            "annihilate",
            ((0.0, 1.0), (0.0, 0.0)),
            tuple(-int(value) for value in np.asarray(space.charges[1])),
        )
        for space in spaces
    )
    creation = tuple(value.adjoint() for value in annihilation)
    lattice_terms = []
    manifold_by_identity = {
        (value.landau_level, value.component.key_id): value for value in plan.manifolds
    }
    for index, orbital in enumerate(orbitals):
        manifold = manifold_by_identity[(orbital.landau_level, orbital.component.key_id)]
        if manifold.one_body_energy != 0.0:
            lattice_terms.append(
                QuantumLatticeTerm(
                    (creation[index], annihilation[index]),
                    coefficient=manifold.one_body_energy,
                    label=f"one-body:{orbital.orbital_id}",
                )
            )
    for term in plan.terms:
        factors = tuple(creation[index] for index in term.creators) + tuple(
            annihilation[index] for index in reversed(term.annihilators)
        )
        lattice_terms.append(
            QuantumLatticeTerm(
                factors,
                coefficient=term.coefficient,
                add_adjoint=term.creators != term.annihilators,
                label=term.term_id,
            )
        )
    specification = QuantumLatticeSpecification(
        spaces,
        lattice_terms,
        fermion_mode_order=mode_order,
    )
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=2 * plan.maximum_terms,
        maximum_factors_per_term=max(
            2, 2 * max((len(value.creators) for value in plan.terms), default=1)
        ),
        maximum_branches_per_input=32 * plan.maximum_terms,
        maximum_sector_dimension=plan.maximum_basis_dimension,
        maximum_workspace_bytes=max(64 * 1024 * 1024, 4096 * plan.maximum_terms),
    )
    prepared = prepare_quantum_lattice(specification, resources)
    local_vectors = tuple(
        (
            (0,) * len(charge_labels),
            tuple(_mode_charge(charge, orbital) for charge in charge_labels),
        )
        for orbital in orbitals
    )
    basis = FixedAbelianChargeBasis(
        labels,
        local_vectors,
        charge_labels,
        tuple(value for _, value in plan.sector.targets),
        group,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=plan.maximum_basis_dimension,
            maximum_table_bytes=plan.maximum_table_bytes,
        ),
    )
    charge_map = SectorChargeMap(basis, basis, 0)
    operator = QuantumSectorOperator(prepared, charge_map)
    return PreparedMultiLandauLevelSphere(
        plan,
        prepared,
        basis,
        operator,
        canonical_fingerprint(
            {
                "kind": "prepared-multi-landau-level-sphere",
                "plan": plan.plan_id,
                "lattice": prepared.prepared_id,
                "basis": basis.basis_id,
            }
        ),
    )


__all__ = [
    "MultiLandauLevelSpherePlan",
    "PreparedMultiLandauLevelSphere",
    "ProjectedOrbitalTerm",
    "prepare_multi_landau_level_sphere",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Magnetic-torus projected Hamiltonians and modular momentum sectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

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
from ._identity import HallComponentKey, SPIN_POLARIZED_ELECTRON


class MagneticTorusGeometry(StrictModule, NonTrainableState):
    flux_quanta: int = eqx.field(static=True)
    modular_parameter: complex = eqx.field(static=True)
    twists: tuple[float, float] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        flux_quanta: int,
        modular_parameter: complex = 1.0j,
        twists: tuple[float, float] = (0.0, 0.0),
    ):
        flux = int(flux_quanta)
        modular = complex(modular_parameter)
        twist = tuple(float(value) for value in twists)
        if (
            flux < 1
            or not np.isfinite(modular)
            or modular.imag <= 0.0
            or len(twist) != 2
            or any(not isfinite(value) for value in twist)
        ):
            raise ValueError(
                "Magnetic torus flux, modular parameter, or twists are invalid."
            )
        self.flux_quanta = flux
        self.modular_parameter = modular
        self.twists = twist
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "magnetic-torus-geometry",
                "flux_quanta": flux,
                "modular_parameter": (modular.real, modular.imag),
                "twists": twist,
            }
        )


@dataclass(frozen=True, slots=True)
class TorusOrbitalTerm:
    creators: tuple[int, ...]
    annihilators: tuple[int, ...]
    coefficient: complex
    winding: tuple[int, int]
    term_id: str

    def __init__(
        self,
        creators: Sequence[int],
        annihilators: Sequence[int],
        coefficient: complex,
        winding: tuple[int, int],
        term_id: str,
    ):
        creators_ = tuple(int(value) for value in creators)
        annihilators_ = tuple(int(value) for value in annihilators)
        coefficient_ = complex(coefficient)
        winding_ = tuple(int(value) for value in winding)
        identifier = str(term_id).strip()
        if (
            not creators_
            or len(creators_) != len(annihilators_)
            or len(winding_) != 2
            or not np.isfinite(coefficient_)
            or not identifier
        ):
            raise ValueError("Torus orbital term is invalid.")
        object.__setattr__(self, "creators", creators_)
        object.__setattr__(self, "annihilators", annihilators_)
        object.__setattr__(self, "coefficient", coefficient_)
        object.__setattr__(self, "winding", winding_)
        object.__setattr__(self, "term_id", identifier)


class TorusProjectedPlan(StrictModule, NonTrainableState):
    particle_count: int = eqx.field(static=True)
    geometry: MagneticTorusGeometry
    component: HallComponentKey
    momentum_sector: int = eqx.field(static=True)
    terms: tuple[TorusOrbitalTerm, ...] = eqx.field(static=True)
    maximum_basis_dimension: int = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_count: int,
        geometry: MagneticTorusGeometry,
        momentum_sector: int,
        terms: Sequence[TorusOrbitalTerm],
        /,
        *,
        component: HallComponentKey = SPIN_POLARIZED_ELECTRON,
        maximum_basis_dimension: int = 100_000,
        maximum_terms: int = 2_000_000,
    ):
        particles = int(particle_count)
        momentum = int(momentum_sector)
        terms_ = tuple(terms)
        if not isinstance(geometry, MagneticTorusGeometry) or not isinstance(
            component, HallComponentKey
        ):
            raise TypeError("geometry and component have invalid types.")
        if particles < 1 or particles > geometry.flux_quanta:
            raise ValueError("Torus particle count is outside the orbital roster.")
        if any(
            not isinstance(value, TorusOrbitalTerm)
            or any(
                index < 0 or index >= geometry.flux_quanta
                for index in value.creators + value.annihilators
            )
            for value in terms_
        ):
            raise ValueError("Torus terms reference unavailable orbitals.")
        if maximum_basis_dimension < 1 or maximum_terms < len(terms_):
            raise ValueError("Torus resource limits are invalid.")
        self.particle_count = particles
        self.geometry = geometry
        self.component = component
        self.momentum_sector = momentum % geometry.flux_quanta
        self.terms = terms_
        self.maximum_basis_dimension = int(maximum_basis_dimension)
        self.maximum_terms = int(maximum_terms)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "torus-projected-plan",
                "particle_count": particles,
                "geometry": geometry.geometry_id,
                "component": component.key_id,
                "momentum_sector": self.momentum_sector,
                "terms": tuple(value.term_id for value in terms_),
            }
        )


class PreparedTorusHamiltonian(StrictModule, NonTrainableState):
    plan: TorusProjectedPlan
    prepared_lattice: PreparedQuantumLattice
    basis: FixedAbelianChargeBasis
    operator: QuantumSectorOperator
    prepared_id: str = eqx.field(static=True)


def prepare_torus_hamiltonian(plan: TorusProjectedPlan, /) -> PreparedTorusHamiltonian:
    if not isinstance(plan, TorusProjectedPlan):
        raise TypeError("plan must be TorusProjectedPlan.")
    flux = plan.geometry.flux_quanta
    labels = tuple(f"torus-{index}" for index in range(flux))
    mode_order = FermionModeOrder(labels)
    spaces = tuple(
        LocalSpacePlan(
            label,
            ("empty", "occupied"),
            ("particle-number", "magnetic-momentum"),
            ((0, 0), (1, index % flux)),
            statistics="fermion",
            fermion_mode_label=label,
        )
        for index, label in enumerate(labels)
    )
    annihilation = tuple(
        LocalOperatorPlan(
            space,
            "annihilate",
            ((0.0, 1.0), (0.0, 0.0)),
            (-1, -index),
        )
        for index, space in enumerate(spaces)
    )
    creation = tuple(value.adjoint() for value in annihilation)
    lattice_terms = []
    for term in plan.terms:
        phase = np.exp(
            1.0j
            * (
                term.winding[0] * plan.geometry.twists[0]
                + term.winding[1] * plan.geometry.twists[1]
            )
        )
        factors = tuple(creation[index] for index in term.creators) + tuple(
            annihilation[index] for index in reversed(term.annihilators)
        )
        lattice_terms.append(
            QuantumLatticeTerm(
                factors,
                coefficient=term.coefficient * phase,
                add_adjoint=term.creators != term.annihilators or term.winding != (0, 0),
                label=term.term_id,
            )
        )
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(
            spaces,
            lattice_terms,
            fermion_mode_order=mode_order,
        ),
        QuantumLatticeResourcePolicy(
            maximum_terms=2 * plan.maximum_terms,
            maximum_factors_per_term=max(
                2, 2 * max((len(value.creators) for value in plan.terms), default=1)
            ),
            maximum_branches_per_input=32 * plan.maximum_terms,
            maximum_sector_dimension=plan.maximum_basis_dimension,
            maximum_workspace_bytes=max(64 * 1024 * 1024, 4096 * plan.maximum_terms),
        ),
    )
    group = AbelianGroup((None, flux))
    basis = FixedAbelianChargeBasis(
        labels,
        tuple(((0, 0), (1, index)) for index in range(flux)),
        ("particle-number", "magnetic-momentum"),
        (plan.particle_count, plan.momentum_sector),
        group,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=plan.maximum_basis_dimension,
            maximum_table_bytes=128 * 1024 * 1024,
        ),
    )
    operator = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
    return PreparedTorusHamiltonian(
        plan,
        prepared,
        basis,
        operator,
        canonical_fingerprint(
            {
                "kind": "prepared-torus-hamiltonian",
                "plan": plan.plan_id,
                "lattice": prepared.prepared_id,
                "basis": basis.basis_id,
            }
        ),
    )


__all__ = [
    "MagneticTorusGeometry",
    "PreparedTorusHamiltonian",
    "TorusOrbitalTerm",
    "TorusProjectedPlan",
    "prepare_torus_hamiltonian",
]

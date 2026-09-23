#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite orbital-cylinder fractional Hall Hamiltonians."""

from __future__ import annotations

from math import factorial, isfinite, pi, sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.quantum import FermionModeOrder
from ...operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    PreparedQuantumLattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)


class HallCylinderPlan(StrictModule, NonTrainableState):
    orbital_count: int = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)
    circumference_in_magnetic_lengths: float = eqx.field(static=True)
    pseudopotentials: tuple[tuple[int, float], ...] = eqx.field(static=True)
    maximum_orbital_separation: int = eqx.field(static=True)
    coefficient_tolerance: float = eqx.field(static=True)
    maximum_terms: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        orbital_count: int,
        particle_count: int,
        circumference_in_magnetic_lengths: float,
        pseudopotentials: dict[int, float],
        /,
        *,
        maximum_orbital_separation: int | None = None,
        coefficient_tolerance: float = 1.0e-12,
        maximum_terms: int = 1_000_000,
    ):
        orbitals = int(orbital_count)
        particles = int(particle_count)
        circumference = float(circumference_in_magnetic_lengths)
        channels = tuple(
            sorted(
                (int(index), float(value)) for index, value in pseudopotentials.items()
            )
        )
        separation = (
            orbitals - 1
            if maximum_orbital_separation is None
            else int(maximum_orbital_separation)
        )
        tolerance = float(coefficient_tolerance)
        maximum = int(maximum_terms)
        if orbitals < 2 or particles < 1 or particles > orbitals:
            raise ValueError("Cylinder orbital and particle counts are invalid.")
        if not isfinite(circumference) or circumference <= 0.0:
            raise ValueError("Cylinder circumference must be positive finite.")
        if (
            not channels
            or any(
                index < 0 or index % 2 == 0 or not isfinite(value)
                for index, value in channels
            )
            or separation < 1
            or separation >= orbitals
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or maximum < 1
        ):
            raise ValueError(
                "Cylinder pseudopotentials or resource controls are invalid."
            )
        self.orbital_count = orbitals
        self.particle_count = particles
        self.circumference_in_magnetic_lengths = circumference
        self.pseudopotentials = channels
        self.maximum_orbital_separation = separation
        self.coefficient_tolerance = tolerance
        self.maximum_terms = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hall-cylinder-plan",
                "orbital_count": orbitals,
                "particle_count": particles,
                "circumference_in_magnetic_lengths": circumference,
                "pseudopotentials": channels,
                "maximum_orbital_separation": separation,
                "coefficient_tolerance": tolerance,
                "maximum_terms": maximum,
            }
        )


class PreparedHallCylinderHamiltonian(StrictModule, NonTrainableState):
    plan: HallCylinderPlan
    specification: QuantumLatticeSpecification
    prepared_lattice: PreparedQuantumLattice
    omitted_coefficient_norm: jnp.ndarray
    retained_term_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


def _pair_amplitude(relative_channel: int, separation: int, kappa: float, /) -> float:
    coordinate = kappa * separation
    coefficients = np.zeros((relative_channel + 1,), dtype=np.float64)
    coefficients[-1] = 1.0
    hermite = np.polynomial.hermite.hermval(coordinate, coefficients)
    normalization = sqrt((2.0**relative_channel) * factorial(relative_channel) * sqrt(pi))
    return float(hermite * np.exp(-0.5 * coordinate**2) / normalization)


def prepare_hall_cylinder_hamiltonian(
    plan: HallCylinderPlan,
    /,
) -> PreparedHallCylinderHamiltonian:
    if not isinstance(plan, HallCylinderPlan):
        raise TypeError("plan must be HallCylinderPlan.")
    labels = tuple(f"orbital-{index}" for index in range(plan.orbital_count))
    order = FermionModeOrder(labels)
    spaces = tuple(
        LocalSpacePlan(
            label,
            ("empty", "occupied"),
            ("particle-number", "orbital-momentum"),
            ((0, 0), (1, index)),
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
    creation = tuple(operator.adjoint() for operator in annihilation)
    pairs_by_center: dict[int, list[tuple[int, int]]] = {}
    for first in range(plan.orbital_count):
        for second in range(first + 1, plan.orbital_count):
            if second - first <= plan.maximum_orbital_separation:
                pairs_by_center.setdefault(first + second, []).append((first, second))
    kappa = 2.0 * pi / plan.circumference_in_magnetic_lengths
    terms = []
    omitted_squared = 0.0
    for center in sorted(pairs_by_center):
        pairs = pairs_by_center[center]
        amplitudes = {
            pair: tuple(
                _pair_amplitude(channel, pair[1] - pair[0], kappa)
                for channel, _ in plan.pseudopotentials
            )
            for pair in pairs
        }
        for row_index, row in enumerate(pairs):
            for column_index in range(row_index + 1):
                column = pairs[column_index]
                coefficient = sum(
                    potential
                    * amplitudes[row][channel_index]
                    * amplitudes[column][channel_index]
                    for channel_index, (_, potential) in enumerate(plan.pseudopotentials)
                )
                if abs(coefficient) <= plan.coefficient_tolerance:
                    omitted_squared += abs(coefficient) ** 2
                    continue
                diagonal = row == column
                terms.append(
                    QuantumLatticeTerm(
                        (
                            creation[row[0]],
                            creation[row[1]],
                            annihilation[column[1]],
                            annihilation[column[0]],
                        ),
                        coefficient=0.5 * coefficient if diagonal else coefficient,
                        add_adjoint=True,
                        label=f"pair-{row[0]}-{row[1]}:{column[0]}-{column[1]}",
                    )
                )
    if not terms:
        raise ValueError("Cylinder interaction preparation produced no retained terms.")
    if len(terms) > plan.maximum_terms:
        raise ValueError("Cylinder interaction exceeds maximum_terms.")
    specification = QuantumLatticeSpecification(
        spaces,
        terms,
        fermion_mode_order=order,
    )
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=2 * plan.maximum_terms,
        maximum_factors_per_term=4,
        maximum_branches_per_input=32 * plan.maximum_terms,
        maximum_sector_dimension=1 << min(plan.orbital_count, 30),
        maximum_workspace_bytes=max(64 * 1024 * 1024, 2048 * plan.maximum_terms),
    )
    prepared = prepare_quantum_lattice(specification, resources)
    return PreparedHallCylinderHamiltonian(
        plan,
        specification,
        prepared,
        jnp.asarray(sqrt(omitted_squared)),
        len(terms),
        canonical_fingerprint(
            {
                "kind": "prepared-hall-cylinder-hamiltonian",
                "plan": plan.plan_id,
                "lattice": prepared.prepared_id,
                "retained_term_count": len(terms),
                "omitted_coefficient_norm": sqrt(omitted_squared),
            }
        ),
    )


__all__ = [
    "HallCylinderPlan",
    "PreparedHallCylinderHamiltonian",
    "prepare_hall_cylinder_hamiltonian",
]

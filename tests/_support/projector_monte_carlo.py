#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent hand matrices, not dense lowerings of the package physics.

Bosons use the textbook two-site Bose-Hubbard N=2 matrix. Fermions use
exterior products |01>,|02>,|12>: replacing the last factor in |12> by
mode 0 crosses one occupied factor and therefore contributes a minus sign.
The flux control is a three-site one-particle ring with oriented hopping.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice._address import QuantumConfigurationDomain
from phydrax.operators.quantum.lattice._column import QuantumLatticeColumnOperator
from phydrax.operators.quantum.lattice._column_compile import (
    prepare_quantum_lattice_columns,
    QuantumColumnResourcePolicy,
)
from phydrax.operators.quantum.lattice._model import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)


@dataclass(frozen=True)
class ProjectorFiniteControl:
    specification: QuantumLatticeSpecification
    coordinates: npt.NDArray[np.int32]
    matrix: npt.NDArray[np.complex128]
    ground_energy: float
    ground_vector: npt.NDArray[np.complex128]
    provenance: str


def two_boson_two_site_control(
    *, hopping: float = 1.0, interaction: float = 2.0
) -> ProjectorFiniteControl:
    spaces = tuple(LocalSpacePlan.boson(site, 3) for site in ("left", "right"))
    annihilate = np.asarray(
        ((0, 1, 0), (0, 0, np.sqrt(2)), (0, 0, 0)), dtype=np.complex128
    )
    create = tuple(
        LocalOperatorPlan(space, "create", annihilate.T, (1,)) for space in spaces
    )
    destroy = tuple(
        LocalOperatorPlan(space, "annihilate", annihilate, (-1,)) for space in spaces
    )
    interaction_ops = tuple(
        LocalOperatorPlan(
            space,
            "pair-occupation",
            np.diag(np.asarray((0, 0, 1), dtype=np.complex128)),
            (0,),
        )
        for space in spaces
    )
    terms = (
        QuantumLatticeTerm(
            (create[0], destroy[1]),
            coefficient=-hopping,
            add_adjoint=True,
            label="boson-hop",
        ),
        QuantumLatticeTerm(
            (interaction_ops[0],), coefficient=interaction, label="left-interaction"
        ),
        QuantumLatticeTerm(
            (interaction_ops[1],), coefficient=interaction, label="right-interaction"
        ),
    )
    off_diagonal = -np.sqrt(2) * hopping
    matrix = np.asarray(
        (
            (interaction, off_diagonal, 0),
            (off_diagonal, 0, off_diagonal),
            (0, off_diagonal, interaction),
        ),
        dtype=np.complex128,
    )
    if hopping == 0:
        energy = min(interaction, 0)
        vector = np.asarray(
            (0, 1, 0) if interaction >= 0 else (1, 0, 0), dtype=np.complex128
        )
    else:
        discriminant = np.sqrt(interaction * interaction + 16 * hopping * hopping)
        energy = (
            -8 * hopping * hopping / (interaction + discriminant)
            if interaction >= 0
            else (interaction - discriminant) / 2
        )
        vector = np.asarray(
            (np.sqrt(2) * hopping, interaction - energy, np.sqrt(2) * hopping),
            dtype=np.complex128,
        )
        vector /= np.linalg.norm(vector)
    return ProjectorFiniteControl(
        QuantumLatticeSpecification(spaces, terms),
        np.asarray(((2, 0), (1, 1), (0, 2)), dtype=np.int32),
        matrix,
        float(energy),
        vector,
        "analytic-two-boson-two-site-bose-hubbard",
    )


def fermionic_exterior_control(*, hopping: float = 0.75) -> ProjectorFiniteControl:
    order = FermionModeOrder(("a", "b", "c"))
    spaces = tuple(LocalSpacePlan.fermion(site, site) for site in order.labels)
    create_matrix = np.asarray(((0, 0), (1, 0)), dtype=np.complex128)
    create = LocalOperatorPlan(spaces[0], "create", create_matrix, (1,))
    destroy = LocalOperatorPlan(spaces[2], "annihilate", create_matrix.T, (-1,))
    term = QuantumLatticeTerm(
        (create, destroy), coefficient=hopping, add_adjoint=True, label="exterior-hop"
    )
    matrix = np.asarray(
        ((0, 0, -hopping), (0, 0, 0), (-hopping, 0, 0)), dtype=np.complex128
    )
    vector = np.asarray((1, 0, 1), dtype=np.complex128) / np.sqrt(2)
    return ProjectorFiniteControl(
        QuantumLatticeSpecification(spaces, (term,), fermion_mode_order=order),
        np.asarray(((1, 1, 0), (1, 0, 1), (0, 1, 1)), dtype=np.int32),
        matrix,
        -hopping,
        vector,
        "three-mode-two-particle-exterior-sign",
    )


def complex_flux_control(
    *, hopping: float = 1.0, phase: float = 0.4
) -> ProjectorFiniteControl:
    spaces = tuple(LocalSpacePlan.boson(site, 2) for site in ("a", "b", "c"))
    create_matrix = np.asarray(((0, 0), (1, 0)), dtype=np.complex128)
    create = tuple(
        LocalOperatorPlan(space, "create", create_matrix, (1,)) for space in spaces
    )
    destroy = tuple(
        LocalOperatorPlan(space, "annihilate", create_matrix.T, (-1,)) for space in spaces
    )
    forward = -hopping * np.exp(1j * phase)
    terms = tuple(
        QuantumLatticeTerm(
            (create[target], destroy[source]),
            coefficient=forward,
            add_adjoint=True,
            label=f"flux-{source}-{target}",
        )
        for source, target in ((0, 1), (1, 2), (2, 0))
    )
    matrix = np.asarray(
        (
            (0, np.conj(forward), forward),
            (forward, 0, np.conj(forward)),
            (np.conj(forward), forward, 0),
        ),
        dtype=np.complex128,
    )
    momenta = np.asarray((0, 2 * np.pi / 3, 4 * np.pi / 3), dtype=np.float64)
    energies = -2 * hopping * np.cos(phase - momenta)
    selected = np.argmin(energies)
    vector = np.exp(1j * momenta[selected] * np.arange(3, dtype=np.float64)).astype(
        np.complex128
    ) / np.sqrt(3)
    return ProjectorFiniteControl(
        QuantumLatticeSpecification(spaces, terms),
        np.eye(3, dtype=np.int32),
        matrix,
        float(energies[selected]),
        vector,
        "analytic-three-site-one-particle-flux-ring",
    )


def control_operator(control: ProjectorFiniteControl) -> QuantumLatticeColumnOperator:
    domain = QuantumConfigurationDomain(
        control.specification,
        species_ids=("particle",) * len(control.specification.spaces),
    )
    resources = QuantumColumnResourcePolicy(
        maximum_monomials=32,
        maximum_factors_per_monomial=4,
        maximum_transition_table_bytes=100_000,
        maximum_raw_routes=256,
        maximum_column_targets=256,
        maximum_workspace_bytes=1_000_000,
    )
    return QuantumLatticeColumnOperator(
        prepare_quantum_lattice_columns(control.specification, resources), domain
    )


def control_keys(
    operator: QuantumLatticeColumnOperator, control: ProjectorFiniteControl
) -> Array:
    return jnp.stack(
        tuple(
            operator.domain.address(coordinate).key_words
            for coordinate in control.coordinates
        )
    )

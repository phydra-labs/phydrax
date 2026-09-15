#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.operators.quantum import CARPolynomial, FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    lower_quantum_lattice_to_vmc,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)
from phydrax.solver._local_hamiltonian import materialize_local_hamiltonian
from phydrax.solver._quantum_lattice import (
    LocalHamiltonianQuantumLatticePolicy,
    lower_quantum_lattice_to_local_hamiltonian,
)
from phydrax.tensor_network._quantum_lattice import (
    lower_quantum_lattice_to_mpo,
    QuantumLatticeMPOPolicy,
)


def _case():
    order = FermionModeOrder(("a", "b", "c"))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate = create.T
    by_site = {
        space.site_id: (
            LocalOperatorPlan(space, "create", create, (1,)),
            LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
        )
        for space in spaces
    }
    term = QuantumLatticeTerm(
        (by_site["a"][0], by_site["c"][1]),
        coefficient=0.75,
        add_adjoint=True,
        label="long-range-hopping",
    )
    specification = QuantumLatticeSpecification(spaces, (term,), fermion_mode_order=order)
    prepared = prepare_quantum_lattice(
        specification,
        QuantumLatticeResourcePolicy(
            maximum_terms=8,
            maximum_factors_per_term=4,
            maximum_branches_per_input=64,
            maximum_sector_dimension=32,
            maximum_workspace_bytes=100_000,
        ),
    )
    basis = FixedCardinalityFermionBasis(
        order,
        2,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=32, maximum_table_bytes=20_000
        ),
    )
    sector = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
    return order, prepared, basis, sector


def _sector_matrix(operator):
    identity = jnp.eye(operator.source.size, dtype=jnp.complex128)
    return jnp.stack(tuple(operator.mv(column) for column in identity), axis=1)


def test_local_hamiltonian_mpo_and_vmc_lowerers_preserve_one_car_operator():
    order, prepared, basis, sector = _case()
    full_reference = CARPolynomial(
        order,
        (
            (0.75, (("a", "create"), ("c", "annihilate"))),
            (0.75, (("c", "create"), ("a", "annihilate"))),
        ),
    ).dense_matrix(maximum_elements=512)

    local = lower_quantum_lattice_to_local_hamiltonian(
        prepared,
        LocalHamiltonianQuantumLatticePolicy(maximum_term_matrix_elements=128),
    )
    local_matrix = materialize_local_hamiltonian(local.hamiltonian)
    np.testing.assert_allclose(local_matrix, full_reference)
    assert local.evidence.exact

    mpo = lower_quantum_lattice_to_mpo(
        prepared,
        QuantumLatticeMPOPolicy(maximum_bond_dimension=4, maximum_tensor_elements=1_000),
    )
    np.testing.assert_allclose(
        mpo.operator.to_dense(maximum_elements=512), full_reference
    )
    assert bool(mpo.evidence.hermitian)

    vmc = lower_quantum_lattice_to_vmc(sector)
    matrix = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    for row in range(basis.dimension):
        configuration = basis.coordinate(row)
        matrix[row, row] += complex(vmc.diagonal(configuration))
        connections = vmc.connections(configuration)
        for connected, element, valid in zip(
            np.asarray(connections.configurations),
            np.asarray(connections.matrix_elements),
            np.asarray(connections.valid),
            strict=True,
        ):
            if valid:
                matrix[row, int(basis.rank(connected))] += element
    np.testing.assert_allclose(matrix, _sector_matrix(sector))

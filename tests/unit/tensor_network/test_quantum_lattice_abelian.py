import numpy as np

from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.tensor_network import (
    lower_quantum_lattice_to_abelian_mpo,
    lower_quantum_lattice_to_mpo,
    QuantumLatticeAbelianMPOPolicy,
    QuantumLatticeMPOPolicy,
)


def test_abelian_lowering_matches_exact_dense_mpo():
    labels = ("left", "right")
    order = FermionModeOrder(labels)
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in labels)
    number = tuple(
        LocalOperatorPlan(
            space,
            "number",
            ((0.0, 0.0), (0.0, 1.0)),
            (0,),
        )
        for space in spaces
    )
    specification = QuantumLatticeSpecification(
        spaces,
        (
            QuantumLatticeTerm((number[0],), coefficient=1.0, label="left-number"),
            QuantumLatticeTerm((number[1],), coefficient=2.0, label="right-number"),
        ),
        fermion_mode_order=order,
    )
    prepared = prepare_quantum_lattice(
        specification,
        QuantumLatticeResourcePolicy(
            maximum_terms=8,
            maximum_factors_per_term=2,
            maximum_branches_per_input=32,
            maximum_sector_dimension=16,
            maximum_workspace_bytes=1_000_000,
        ),
    )
    dense = lower_quantum_lattice_to_mpo(
        prepared,
        QuantumLatticeMPOPolicy(
            maximum_bond_dimension=8,
            maximum_tensor_elements=1_000,
        ),
    ).operator.to_dense()
    abelian = lower_quantum_lattice_to_abelian_mpo(
        prepared,
        QuantumLatticeAbelianMPOPolicy(
            maximum_bond_dimension=8,
            maximum_tensor_elements=1_000,
        ),
    ).operator.to_dense()

    np.testing.assert_allclose(abelian, dense, atol=1.0e-12)

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax.operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from phydrax.tensor_network._quantum_lattice import (
    lower_quantum_lattice_to_mpo,
    QuantumLatticeMPOPolicy,
)


def test_mpo_lowerer_preserves_heterogeneous_spin_boson_product():
    spin = LocalSpacePlan.spin("spin", 1)
    boson = LocalSpacePlan.boson("phonon", 3)
    sigma_z = np.diag((-1.0, 1.0))
    number = np.diag((0.0, 1.0, 2.0))
    term = QuantumLatticeTerm(
        (
            LocalOperatorPlan(spin, "sz", sigma_z, (0,)),
            LocalOperatorPlan(boson, "number", number, (0,)),
        ),
        coefficient=0.4,
        label="spin-boson-density",
    )
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification((spin, boson), (term,)),
        QuantumLatticeResourcePolicy(
            maximum_terms=2,
            maximum_factors_per_term=2,
            maximum_branches_per_input=8,
            maximum_sector_dimension=16,
            maximum_workspace_bytes=10_000,
        ),
    )
    result = lower_quantum_lattice_to_mpo(
        prepared,
        QuantumLatticeMPOPolicy(maximum_bond_dimension=2, maximum_tensor_elements=100),
    )
    np.testing.assert_allclose(
        result.operator.to_dense(maximum_elements=100),
        0.4 * np.kron(sigma_z, number),
    )
    assert bool(result.evidence.hermitian)

    with pytest.raises(ValueError, match="maximum_tensor_elements"):
        lower_quantum_lattice_to_mpo(
            prepared,
            QuantumLatticeMPOPolicy(maximum_bond_dimension=2, maximum_tensor_elements=1),
        )

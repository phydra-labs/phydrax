#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.linalg import LinearCapabilityError, MaterializationPolicy, materialize
from phydrax.linalg.eigen import (
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    RestartedLanczos,
)
from phydrax.operators.quantum import (
    CARPolynomial,
    FermionicFockBasis,
    FermionModeOrder,
)
from phydrax.operators.quantum.lattice import (
    FixedBosonNumberBasis,
    FixedCardinalityFermionBasis,
    FixedSpinProjectionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)


def _resources():
    return QuantumLatticeResourcePolicy(
        maximum_terms=16,
        maximum_factors_per_term=4,
        maximum_branches_per_input=128,
        maximum_sector_dimension=64,
        maximum_workspace_bytes=100_000,
    )


def _basis_resources():
    return SectorBasisResourcePolicy(maximum_dimension=64, maximum_table_bytes=20_000)


def _three_mode_hopping():
    order = FermionModeOrder(("a", "b", "c"))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate = create.T
    operators = {
        space.site_id: {
            "create": LocalOperatorPlan(space, "create", create, (1,)),
            "annihilate": LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
        }
        for space in spaces
    }
    hopping = QuantumLatticeTerm(
        (operators["a"]["create"], operators["c"]["annihilate"]),
        coefficient=1.0,
        add_adjoint=True,
        label="a-c-hopping",
    )
    specification = QuantumLatticeSpecification(
        spaces, (hopping,), fermion_mode_order=order
    )
    return order, prepare_quantum_lattice(specification, _resources())


def _matrix(operator):
    identity = jnp.eye(operator.source.size, dtype=jnp.complex128)
    return jnp.stack(tuple(operator.mv(column) for column in identity), axis=1)


def test_matrix_free_sector_action_matches_independent_full_fock_car_signs():
    order, prepared = _three_mode_hopping()
    sector = FixedCardinalityFermionBasis(order, 2, resources=_basis_resources())
    operator = QuantumSectorOperator(prepared, SectorChargeMap(sector, sector, 0))
    sector_matrix = _matrix(operator)

    full_basis = FermionicFockBasis(order)
    full = CARPolynomial(
        order,
        (
            (1.0, (("a", "create"), ("c", "annihilate"))),
            (1.0, (("c", "create"), ("a", "annihilate"))),
        ),
    ).dense_matrix(maximum_elements=512)
    indices = np.asarray(
        [
            full_basis.basis_index(np.asarray(sector.coordinate(index)).tolist())
            for index in range(sector.dimension)
        ]
    )
    reference = np.asarray(full)[np.ix_(indices, indices)]
    np.testing.assert_allclose(sector_matrix, reference)
    with pytest.raises(LinearCapabilityError, match="does not support"):
        materialize(operator, MaterializationPolicy(max_entries=16, max_bytes=1_024))
    assert sector_matrix[2, 0] == -1.0


def test_charge_map_certification_rejects_wrong_target_sector():
    order = FermionModeOrder(("a", "b"))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    creation = LocalOperatorPlan(spaces[0], "create-a", ((0.0, 0.0), (1.0, 0.0)), (1,))
    specification = QuantumLatticeSpecification(
        spaces,
        (QuantumLatticeTerm((creation,), label="source-probe"),),
        fermion_mode_order=order,
    )
    prepared = prepare_quantum_lattice(specification, _resources())
    source = FixedCardinalityFermionBasis(order, 0, resources=_basis_resources())
    target = FixedCardinalityFermionBasis(order, 1, resources=_basis_resources())
    mapped = QuantumSectorOperator(prepared, SectorChargeMap(source, target, 1))
    np.testing.assert_allclose(mapped.mv(jnp.asarray([1.0 + 0.0j])), [0.0, 1.0])
    with pytest.raises(ValueError, match="unique charge map"):
        QuantumSectorOperator(prepared, SectorChargeMap(target, target, 0))


def test_direct_spin_and_boson_sectors_preserve_ladder_normalizations():
    spin_spaces = (LocalSpacePlan.spin("i", 1), LocalSpacePlan.spin("j", 1))
    spin_raise = ((0.0, 0.0), (1.0, 0.0))
    spin_lower = np.asarray(spin_raise).T
    spin_term = QuantumLatticeTerm(
        (
            LocalOperatorPlan(spin_spaces[0], "raise", spin_raise, (2,)),
            LocalOperatorPlan(spin_spaces[1], "lower", spin_lower, (-2,)),
        ),
        coefficient=1.0,
        add_adjoint=True,
        label="spin-exchange",
    )
    spin_prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(spin_spaces, (spin_term,)), _resources()
    )
    spin_basis = FixedSpinProjectionBasis(
        ("i", "j"), (1, 1), 0, resources=_basis_resources()
    )
    spin_operator = QuantumSectorOperator(
        spin_prepared, SectorChargeMap(spin_basis, spin_basis, 0)
    )
    np.testing.assert_allclose(_matrix(spin_operator), ((0.0, 1.0), (1.0, 0.0)))

    boson_spaces = (LocalSpacePlan.boson("x", 3), LocalSpacePlan.boson("y", 3))
    create = np.diag(np.sqrt((1.0, 2.0)), k=-1)
    annihilate = create.T
    boson_term = QuantumLatticeTerm(
        (
            LocalOperatorPlan(boson_spaces[0], "create", create, (1,)),
            LocalOperatorPlan(boson_spaces[1], "annihilate", annihilate, (-1,)),
        ),
        coefficient=1.0,
        add_adjoint=True,
        label="boson-hopping",
    )
    boson_prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(boson_spaces, (boson_term,)), _resources()
    )
    boson_basis = FixedBosonNumberBasis(
        ("x", "y"), (3, 3), 2, resources=_basis_resources()
    )
    boson_operator = QuantumSectorOperator(
        boson_prepared, SectorChargeMap(boson_basis, boson_basis, 0)
    )
    root_two = np.sqrt(2.0)
    np.testing.assert_allclose(
        _matrix(boson_operator),
        ((0.0, root_two, 0.0), (root_two, 0.0, root_two), (0.0, root_two, 0.0)),
    )


def test_existing_linalg_eigensolve_consumes_matrix_free_sector_operator_directly():
    order, prepared = _three_mode_hopping()
    sector = FixedCardinalityFermionBasis(order, 2, resources=_basis_resources())
    operator = QuantumSectorOperator(prepared, SectorChargeMap(sector, sector, 0))
    result = eigensolve(
        Eigenproblem(operator),
        policy=EigenSolvePolicy(
            RestartedLanczos(subspace_dimension=3, restart_dimension=2),
            count=2,
            max_steps=8,
            key=jnp.asarray((0, 7), dtype=jnp.uint32),
        ),
    )
    np.testing.assert_allclose(result.eigenvalues, (-1.0, 0.0), atol=1e-8)
    assert bool(jnp.all(result.converged))

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.linalg import (
    DifferentiationPolicy,
    IdentityLinearOperator,
    MatrixFunctionPolicy,
)
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
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
from phydrax.solver._thermal_pure_quantum import (
    prepare_thermal_pure_quantum,
    thermal_pure_quantum,
    ThermalPureQuantumPlan,
)


def _diagonal_fermion_operators():
    order = FermionModeOrder(("a", "b"))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate = create.T
    number_terms = []
    for energy, space in zip((1.0, 2.0), spaces, strict=True):
        number_terms.append(
            QuantumLatticeTerm(
                (
                    LocalOperatorPlan(space, "create", create, (1,)),
                    LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
                ),
                coefficient=energy,
                label=f"number:{space.site_id}",
            )
        )
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=8,
        maximum_factors_per_term=4,
        maximum_branches_per_input=64,
        maximum_sector_dimension=16,
        maximum_workspace_bytes=50_000,
    )
    specification = QuantumLatticeSpecification(
        spaces, number_terms, fermion_mode_order=order
    )
    prepared = prepare_quantum_lattice(specification, resources)
    basis = FixedCardinalityFermionBasis(
        order,
        1,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=16, maximum_table_bytes=10_000
        ),
    )
    hamiltonian = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
    number_a = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, (number_terms[0],), fermion_mode_order=order),
        resources,
    )
    observable = QuantumSectorOperator(number_a, SectorChargeMap(basis, basis, 0))
    return hamiltonian, observable


def _plan(beta, maximum_retained_bytes=100_000, maximum_workspace_bytes=100_000):
    return ThermalPureQuantumPlan(
        beta,
        probe_count=3,
        observable_count=1,
        matrix_function=MatrixFunctionPolicy(
            "lanczos",
            max_dimension=2,
            error_tolerance=1e-11,
            differentiation=DifferentiationPolicy("none"),
        ),
        maximum_retained_bytes=maximum_retained_bytes,
        maximum_workspace_bytes=maximum_workspace_bytes,
    )


def test_tpq_beta_zero_trace_and_semantic_prng_replay_are_exact():
    hamiltonian, _ = _diagonal_fermion_operators()
    identity = IdentityLinearOperator(hamiltonian.source)
    prepared = prepare_thermal_pure_quantum(_plan(0.0), hamiltonian)
    first = thermal_pure_quantum(prepared, (identity,), key=jr.key(91))
    replay = thermal_pure_quantum(prepared, (identity,), key=jr.key(91))
    np.testing.assert_allclose(first.partition_estimate, hamiltonian.source.size)
    np.testing.assert_allclose(first.observable_estimates, (1.0,))
    np.testing.assert_array_equal(first.raw_probes, replay.raw_probes)
    np.testing.assert_array_equal(first.thermal_vectors, replay.thermal_vectors)
    assert bool(first.valid)


def test_tpq_uses_ratio_of_probe_sums_for_canonical_observable():
    hamiltonian, number_a = _diagonal_fermion_operators()
    beta = 0.7
    result = thermal_pure_quantum(
        prepare_thermal_pure_quantum(_plan(beta), hamiltonian),
        (number_a,),
        key=jr.key(13),
    )
    expected_partition = np.exp(-beta) + np.exp(-2.0 * beta)
    expected_occupation = np.exp(-beta) / expected_partition
    np.testing.assert_allclose(result.partition_estimate, expected_partition, atol=1e-9)
    np.testing.assert_allclose(
        result.observable_estimates, (expected_occupation,), atol=1e-9
    )
    ratio_of_sums = jnp.sum(result.observable_numerators, axis=0) / jnp.sum(
        result.norm_weights
    )
    np.testing.assert_allclose(result.observable_estimates, ratio_of_sums)


def test_tpq_preparation_refuses_retained_or_krylov_workspace():
    hamiltonian, _ = _diagonal_fermion_operators()
    with pytest.raises(ValueError, match="maximum_retained_bytes"):
        prepare_thermal_pure_quantum(_plan(0.0, maximum_retained_bytes=1), hamiltonian)
    with pytest.raises(ValueError, match="maximum_workspace_bytes"):
        prepare_thermal_pure_quantum(_plan(0.0, maximum_workspace_bytes=1), hamiltonian)

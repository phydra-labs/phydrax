#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.operators.periodic._family import PeriodicFiniteRealization
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    assess_sign_free_stochastic_candidate,
    FermionInteractionPlan,
    FermionInteractionTerm,
    FixedCardinalityFermionBasis,
    FixedSpinProjectionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    periodic_finite_to_fermion_lattice,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    refresh_quantum_lattice,
    SectorBasisResourcePolicy,
    SectorChargeMap,
    SignFreeStochasticCandidatePlan,
)


def _compiler_resources(**updates):
    values = {
        "maximum_terms": 32,
        "maximum_factors_per_term": 4,
        "maximum_branches_per_input": 128,
        "maximum_sector_dimension": 64,
        "maximum_workspace_bytes": 100_000,
    }
    values.update(updates)
    return QuantumLatticeResourcePolicy(**values)


def _sector_resources():
    return SectorBasisResourcePolicy(maximum_dimension=64, maximum_table_bytes=20_000)


def test_periodic_bridge_requires_explicit_order_and_interaction_contract():
    realization = PeriodicFiniteRealization(
        (0, 0, 1, 1),
        (0, 1, 0, 1),
        (1.0, -0.5, -0.5, 2.0),
        supercell_shape=(1,),
        periodic_axes=(False,),
        twists=(0.0,),
        output_size=2,
        input_size=2,
        maximum_dense_entries=4,
        source_prepared_id="two-orbital-finite",
    )
    order = FermionModeOrder(("a", "b"))
    interactions = FermionInteractionPlan(
        (
            FermionInteractionTerm(
                (
                    ("a", "create"),
                    ("a", "annihilate"),
                    ("b", "create"),
                    ("b", "annihilate"),
                ),
                3.0,
                label="density-interaction",
            ),
        ),
        provenance_id="supplied-density-interaction",
        units_id="energy-unit",
        maximum_terms=1,
        maximum_operations_per_term=4,
    )
    specification = periodic_finite_to_fermion_lattice(
        realization, order, ("a", "b"), interactions
    )
    prepared = prepare_quantum_lattice(specification, _compiler_resources())
    basis = FixedCardinalityFermionBasis(order, 1, resources=_sector_resources())
    operator = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
    columns = jnp.eye(basis.dimension, dtype=jnp.complex128)
    matrix = jnp.stack(tuple(operator.mv(column) for column in columns), axis=1)
    np.testing.assert_allclose(matrix, [[2.0, -0.5], [-0.5, 1.0]])
    filled = FixedCardinalityFermionBasis(order, 2, resources=_sector_resources())
    filled_operator = QuantumSectorOperator(prepared, SectorChargeMap(filled, filled, 0))
    np.testing.assert_allclose(filled_operator.mv(jnp.asarray((1.0 + 0.0j,))), (6.0,))
    with pytest.raises(ValueError, match="exactly equal"):
        periodic_finite_to_fermion_lattice(realization, order, ("b", "a"), interactions)


def test_compiler_refuses_branch_work_before_preparation():
    space = LocalSpacePlan.spin("s", 2)
    raising = LocalOperatorPlan(
        space,
        "raise",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (2,),
    )
    term = QuantumLatticeTerm((raising,), coefficient=1.0, label="raise")
    specification = QuantumLatticeSpecification((space,), (term,))
    with pytest.raises(ValueError, match="branch count"):
        prepare_quantum_lattice(
            specification,
            _compiler_resources(maximum_branches_per_input=2),
        )


def test_numeric_refresh_preserves_structure_and_updates_sector_action():
    space = LocalSpacePlan.spin("s", 1)
    sz = LocalOperatorPlan(space, "sz", ((-1.0, 0.0), (0.0, 1.0)), (0,))
    initial_specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((sz,), coefficient=1.0, label="field"),)
    )
    initial = prepare_quantum_lattice(initial_specification, _compiler_resources())
    refreshed_specification = QuantumLatticeSpecification(
        (space,), (QuantumLatticeTerm((sz,), coefficient=2.0, label="field"),)
    )
    refreshed = refresh_quantum_lattice(initial, refreshed_specification)
    basis = FixedSpinProjectionBasis(("s",), (1,), 1, resources=_sector_resources())
    charge_map = SectorChargeMap(basis, basis, 0)
    initial_operator = QuantumSectorOperator(initial, charge_map)
    refreshed_operator = QuantumSectorOperator(refreshed, charge_map)
    vector = jnp.asarray((1.0 + 0.0j,))
    np.testing.assert_allclose(initial_operator.mv(vector), (1.0,))
    np.testing.assert_allclose(refreshed_operator.mv(vector), (2.0,))
    assert refreshed.plan.plan_id == initial.plan.plan_id
    assert int(refreshed.numeric_version) == int(initial.numeric_version) + 1


def test_sign_free_candidate_retains_raw_chain_and_refuses_sign_cancellation():
    plan = SignFreeStochasticCandidatePlan(
        chain_count=2,
        draw_count=4,
        chain_state_width=2,
        observable_count=1,
        maximum_expansion_order=3,
        maximum_autocorrelation_lag=2,
        minimum_signed_effective_samples=2.0,
        sign_tolerance=1e-12,
        maximum_raw_bytes=10_000,
        method_id="bounded-stoquastic-control",
    )
    chain = jnp.arange(16.0).reshape((2, 4, 2))
    orders = jnp.asarray(((0, 1, 1, 2), (0, 1, 2, 3)))
    observations = jnp.arange(8.0).reshape((2, 4, 1))
    accepted = jnp.asarray(((True, False, True, True), (True, True, False, True)))
    sign_free = assess_sign_free_stochastic_candidate(
        plan, chain, orders, observations, jnp.ones((2, 4)), accepted
    )
    assert bool(sign_free.successful)
    np.testing.assert_array_equal(sign_free.raw_observables, observations)
    np.testing.assert_array_equal(sign_free.raw_chain, chain)
    np.testing.assert_array_equal(sign_free.raw_orders, orders)
    np.testing.assert_array_equal(sign_free.order_histogram, (2, 3, 2, 1))
    cancelling = assess_sign_free_stochastic_candidate(
        plan,
        chain,
        orders,
        observations,
        jnp.asarray(((1, -1, 1, -1), (1, -1, 1, -1))),
        accepted,
    )
    assert not bool(cancelling.sign_free)
    assert not bool(cancelling.successful)

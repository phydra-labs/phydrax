#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.applications.lattice_field._hamiltonian_gauge import (
    compact_u1_gauss_network,
    compact_u1_hamiltonian,
    CompactU1GaugeModel2D,
    compile_gauge_preserving_product_formula,
    estimate_gauge_hamiltonian_resources,
    execute_gauge_preserving_program,
    gauge_hamiltonian_mpo,
    GaugeSimulationResourcePolicy,
    periodic_schwinger_flux_values,
    periodic_schwinger_gauss_network,
    periodic_schwinger_hamiltonian,
    PeriodicSchwingerModel,
)
from phydrax.discretization._cell_complex import polygonal_cell_complex
from phydrax.linalg._materialization import MaterializationPolicy
from phydrax.operators.quantum._gauge_constraints import (
    gauss_commutator_evidence,
    GaussSectorResourcePolicy,
    prepare_sparse_physical_sector,
)
from phydrax.solver._local_hamiltonian import materialize_local_hamiltonian
from phydrax.tensor_network._core import MatrixProductState
from phydrax.tensor_network._mpo import apply_mpo


def test_periodic_schwinger_global_flux_theta_and_gauss_commutator():
    model = PeriodicSchwingerModel(
        2,
        maximum_flux=1,
        lattice_spacing=0.5,
        mass=0.2,
        gauge_coupling=0.7,
        theta_angle=jnp.pi,
        global_flux_sector=2,
    )
    assert jnp.allclose(
        periodic_schwinger_flux_values(model, jnp.asarray([0, 2])),
        jnp.asarray([1.5, 3.5]),
    )
    network = periodic_schwinger_gauss_network(model)
    hamiltonian = periodic_schwinger_hamiltonian(model)
    dense = materialize_local_hamiltonian(
        hamiltonian,
        policy=MaterializationPolicy(max_entries=2000, max_bytes=1 << 20),
    )
    evidence = gauss_commutator_evidence(
        network,
        dense,
        operator_id=hamiltonian.hamiltonian_id,
        maximum_elements=10000,
    )
    assert evidence.gauge_invariant
    assert evidence.maximum_residual == 0.0


def test_periodic_theta_shift_matches_neighboring_global_flux_sector():
    parameters = {
        "maximum_flux": 1,
        "lattice_spacing": 0.5,
        "mass": 0.2,
        "gauge_coupling": 0.7,
    }
    first = PeriodicSchwingerModel(2, theta_angle=0.3, global_flux_sector=0, **parameters)
    shifted = PeriodicSchwingerModel(
        2, theta_angle=0.3 + 2.0 * jnp.pi, global_flux_sector=-1, **parameters
    )
    first_dense = materialize_local_hamiltonian(
        periodic_schwinger_hamiltonian(first),
        policy=MaterializationPolicy(max_entries=2000, max_bytes=1 << 20),
    )
    shifted_dense = materialize_local_hamiltonian(
        periodic_schwinger_hamiltonian(shifted),
        policy=MaterializationPolicy(max_entries=2000, max_bytes=1 << 20),
    )
    assert jnp.allclose(first_dense, shifted_dense)


def test_compact_2d_dense_mpo_mps_and_gauss_evolution_agree():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    model = CompactU1GaugeModel2D(
        topology,
        maximum_flux=1,
        electric_coupling=0.8,
        magnetic_coupling=0.4,
    )
    network = compact_u1_gauss_network(model)
    hamiltonian = compact_u1_hamiltonian(model)
    dense = materialize_local_hamiltonian(
        hamiltonian,
        policy=MaterializationPolicy(max_entries=2000, max_bytes=1 << 20),
    )
    mpo = gauge_hamiltonian_mpo(
        hamiltonian,
        maximum_dense_elements=2000,
        maximum_bond_dimension=100,
    )
    assert mpo.evidence.valid
    assert jnp.allclose(mpo.operator.to_dense(), dense)

    local_zero_flux = jnp.asarray((0.0, 1.0, 0.0), dtype=complex)
    mps = MatrixProductState(
        tuple(local_zero_flux[None, :, None] for _ in range(model.link_count))
    )
    applied, compression = apply_mpo(
        mpo.operator,
        mps,
        maximum_bond_dimension=100,
        normalize=False,
    )
    assert compression.valid
    assert jnp.allclose(applied.to_dense(), dense @ mps.to_dense())

    sector = prepare_sparse_physical_sector(
        network,
        resources=GaussSectorResourcePolicy(
            maximum_hilbert_dimension=100,
            maximum_dense_elements=100000,
            maximum_basis_states=100,
        ),
    )
    initial = (
        jnp.zeros((network.physical_dimension,), dtype=complex)
        .at[int(sector.subspace.basis_indices[0])]
        .set(1.0)
    )
    prepared = compile_gauge_preserving_product_formula(
        hamiltonian,
        network,
        0.01,
        step_count=2,
        order=2,
        resources=GaugeSimulationResourcePolicy(
            maximum_dense_elements=10000,
            maximum_program_operations=100,
        ),
    )
    result = execute_gauge_preserving_program(prepared, network, initial)
    assert prepared.evidence.gauge_preserving
    assert result.evidence.valid
    assert result.evidence.final_leakage <= 1e-12


def test_product_formula_lcu_and_qubitization_counts_are_explicit():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    hamiltonian = compact_u1_hamiltonian(
        CompactU1GaugeModel2D(
            topology,
            maximum_flux=1,
            electric_coupling=0.8,
            magnetic_coupling=0.4,
        )
    )
    evidence = estimate_gauge_hamiltonian_resources(
        hamiltonian,
        1.0,
        1e-2,
        product_formula_order=2,
        product_formula_steps=5,
    )
    assert evidence.valid
    assert evidence.term_count == 4
    assert evidence.product_formula_exponentials == 40
    assert evidence.lcu_select_qubits == 2
    assert evidence.lcu_prepare_amplitudes == 4
    assert evidence.qubitization_queries > 0

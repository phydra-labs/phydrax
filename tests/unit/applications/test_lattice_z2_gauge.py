#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _triangle_model(*, charges=None):
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]]), None, 3
    )
    return phx.applications.lattice_field.Z2GaugeModel(
        topology,
        electric_coupling=0.4,
        magnetic_coupling=1.2,
        external_charges=charges,
    )


def test_z2_gauss_sector_uses_binary_edge_basis_and_constraint_rank():
    model = _triangle_model()
    sector = phx.applications.lattice_field.prepare_z2_gauss_sector(model)
    occupations = jnp.asarray([[0, 0, 0], [1, 1, 1]], dtype=jnp.int32)

    assert model.valid
    assert sector.constraint_rank == 2
    assert sector.expected_dimension == 2
    np.testing.assert_array_equal(sector.subspace.basis_indices, [0, 7])
    np.testing.assert_array_equal(
        phx.applications.lattice_field.z2_gauss_eigenvalues(model, occupations),
        np.ones((2, 3), dtype=int),
    )


def test_z2_hamiltonian_commutes_with_every_gauss_generator():
    model = _triangle_model()
    hamiltonian = phx.applications.lattice_field.z2_gauge_hamiltonian(model)
    dense = phx.solver.materialize_local_hamiltonian(hamiltonian)

    for term in phx.applications.lattice_field.z2_gauss_terms(model):
        generator = phx.solver.materialize_local_hamiltonian(
            phx.solver.LocalHamiltonian(model.layout, (term,))
        )
        assert jnp.max(jnp.abs(dense @ generator - generator @ dense)) < 1e-12
        assert jnp.max(jnp.abs(generator @ generator - jnp.eye(8))) < 1e-12


def test_z2_boundary_loop_matches_magnetic_support():
    model = _triangle_model()
    paths = phx.discretization.prepare_cell_boundary_paths(model.topology).paths
    loop = phx.applications.lattice_field.z2_loop_operator(model, paths, 0)
    loop_dense = phx.solver.materialize_local_hamiltonian(
        phx.solver.LocalHamiltonian(model.layout, (loop,))
    )
    assert jnp.max(jnp.abs(loop_dense @ loop_dense - jnp.eye(8))) < 1e-12


def test_z2_charge_consistency_and_resource_limits_fail_before_allocation():
    inconsistent = _triangle_model(charges=jnp.asarray([1, 0, 0]))
    with pytest.raises(ValueError, match="inconsistent"):
        phx.applications.lattice_field.prepare_z2_gauss_sector(inconsistent)
    with pytest.raises(ValueError, match="maximum_basis_states"):
        phx.applications.lattice_field.prepare_z2_gauss_sector(
            _triangle_model(), maximum_basis_states=4
        )


def test_z2_homology_uses_exact_gf2_topology():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    topology = phx.discretization.StructuredCochainBridge(grid).cochain.topology
    model = phx.applications.lattice_field.Z2GaugeModel(
        topology,
        electric_coupling=0.2,
        magnetic_coupling=0.7,
    )
    homology = phx.applications.lattice_field.z2_homology(model)

    assert homology.dimensions[:3] == (1, 2, 1)

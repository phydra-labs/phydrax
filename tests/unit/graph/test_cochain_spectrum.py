#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _path() -> phx.discretization.CochainDiscretization:
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((3, 1)),
        edges={"conductance": jnp.ones((4,))},
        senders=jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        receivers=jnp.asarray([1, 0, 2, 1], dtype=jnp.int32),
        n_node=jnp.asarray([3], dtype=jnp.int32),
        n_edge=jnp.asarray([4], dtype=jnp.int32),
    )
    return phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")


def test_diagonal_hodge_path_has_analytic_spectrum_and_native_evidence() -> None:
    realization = _path()
    basis = phx.exterior.hodge_laplacian_eigenbasis(realization, 0)
    np.testing.assert_allclose(basis.eigenvalues, [0.0, 1.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(basis.analysis @ basis.synthesis, np.eye(3), atol=1e-9)
    assert basis.eigen_solve is not None
    assert int(basis.eigen_solve.status) == int(phx.linalg.eigen.EigenSolveStatus.SUCCESS)
    assert bool(jnp.all(basis.eigen_solve.diagnostics.converged))


def test_sparse_lower_hodge_uses_full_lower_mass_solve() -> None:
    realization = _path()
    lower_mass = np.asarray([[2.0, 0.4, 0.0], [0.4, 3.0, 0.25], [0.0, 0.25, 4.0]])
    rows, columns = np.triu_indices(3)
    keep = lower_mass[rows, columns] != 0
    lower_hodge = phx.discretization.SparseHodge(
        rows[keep], columns[keep], lower_mass[rows[keep], columns[keep]], 3
    )
    upper_mass = np.asarray([[1.25, 0.3], [0.3, 2.0]])
    upper_rows, upper_columns = np.triu_indices(2)
    upper_hodge = phx.discretization.SparseHodge(
        upper_rows, upper_columns, upper_mass[upper_rows, upper_columns], 2
    )
    sparse = phx.discretization.CochainDiscretization(
        realization.topology, (lower_hodge, upper_hodge)
    )
    basis = phx.exterior.hodge_laplacian_eigenbasis(sparse, 1, part="lower")
    # Independent weak pencil from the known oriented path, not production assembly.
    differential = np.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    factor = np.linalg.cholesky(upper_mass)
    expected = np.linalg.eigvalsh(
        factor.T @ differential @ np.linalg.solve(lower_mass, differential.T) @ factor
    )
    np.testing.assert_allclose(basis.eigenvalues, expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(basis.analysis @ basis.synthesis, np.eye(2), atol=1e-8)
    assert basis.eigen_solve is not None
    assert int(basis.eigen_solve.status) == int(phx.linalg.eigen.EigenSolveStatus.SUCCESS)


def test_degenerate_eigenspace_cannot_be_split() -> None:
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((4, 1)),
        edges={"conductance": jnp.ones((8,))},
        senders=jnp.asarray([0, 1, 1, 2, 2, 3, 3, 0], dtype=jnp.int32),
        receivers=jnp.asarray([1, 0, 2, 1, 3, 2, 0, 3], dtype=jnp.int32),
        n_node=jnp.asarray([4], dtype=jnp.int32),
        n_edge=jnp.asarray([8], dtype=jnp.int32),
    )
    realization = phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")
    with pytest.raises(ValueError):
        phx.exterior.hodge_laplacian_eigenbasis(realization, 0, count=2)
    basis = phx.exterior.hodge_laplacian_eigenbasis(realization, 0, count=3)
    np.testing.assert_allclose(basis.eigenvalues, [0.0, 2.0, 2.0], atol=1e-9)
    assert basis.report is not None
    assert basis.report.next_eigenvalue == pytest.approx(4.0)


def test_relative_spectrum_excludes_boundary_zero_modes() -> None:
    realization = _path()
    relative = phx.discretization.CochainDiscretization(
        realization.topology,
        tuple(
            phx.discretization.DiagonalHodge(realization.hodge_diagonal(k))
            for k in (0, 1)
        ),
        boundary_masks=(np.asarray([True, False, True]), np.asarray([False, False])),
    )
    basis = phx.exterior.hodge_laplacian_eigenbasis(relative, 0, boundary="relative")
    np.testing.assert_allclose(basis.eigenvalues, [2.0], atol=1e-9)
    np.testing.assert_array_equal(basis.active_mask, [False, True, False])
    np.testing.assert_allclose(basis.eigenfunctions[jnp.asarray([0, 2])], 0.0, atol=0.0)
    harmonic, report = phx.exterior.validate_harmonic_cohomology(
        relative, 0, boundary="relative"
    )
    assert harmonic.dimension == 0
    assert report.exact_dimension == 0
    assert bool(report.complete)


def test_inactive_storage_coordinates_do_not_add_harmonic_modes() -> None:
    vertices = phx.discretization.EntitySet(
        "active-vertices", 0, np.arange(3), active_mask=np.asarray([True, True, False])
    )
    edges = phx.discretization.EntitySet(
        "active-edges", 1, np.arange(2), active_mask=np.asarray([True, False])
    )
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1, 1, 2]), np.asarray([0, 0, 1, 1]), source_size=3, target_size=2
    )
    incidence = phx.discretization.OrientedIncidence(
        1, vertices, edges, relation, np.asarray([-1.0, 1.0, -1.0, 1.0])
    )
    topology = phx.discretization.CellComplexTopology((vertices, edges), (incidence,))
    realization = phx.discretization.CochainDiscretization(
        topology,
        (
            phx.discretization.DiagonalHodge(jnp.ones((3,))),
            phx.discretization.DiagonalHodge(jnp.ones((2,))),
        ),
    )
    basis = phx.exterior.hodge_laplacian_eigenbasis(realization, 0)
    np.testing.assert_allclose(basis.eigenvalues, [0.0, 2.0], atol=1e-9)
    np.testing.assert_array_equal(basis.active_mask, [True, True, False])
    np.testing.assert_allclose(basis.eigenfunctions[2], 0.0, atol=0.0)
    harmonic, report = phx.exterior.validate_harmonic_cohomology(realization, 0)
    assert harmonic.dimension == report.exact_dimension == 1
    assert bool(report.complete)


def test_interval_sparse_gram_publishes_true_metric_and_hodge_kernel() -> None:
    topology = phx.discretization.simplicial_cell_complex(
        (np.asarray([[0], [1]], dtype=np.int32), np.asarray([[0, 1]], dtype=np.int32)),
        topology_id="interval:full-gram",
    )
    mass = np.asarray([[2.0, 0.2], [0.2, 2.0]])
    rows, columns = np.triu_indices(2)
    realization = phx.discretization.CochainDiscretization(
        topology,
        (
            phx.discretization.SparseHodge(rows, columns, mass[rows, columns], 2),
            phx.discretization.DiagonalHodge(jnp.ones(1)),
        ),
    )
    basis = phx.exterior.hodge_laplacian_eigenbasis(realization, 0)
    np.testing.assert_allclose(basis.eigenvalues, [0.0, 10.0 / 9.0], atol=1e-9)
    synthesis = np.asarray(basis.synthesis)
    normalized_mass = mass / np.trace(mass)
    np.testing.assert_allclose(
        synthesis.T @ normalized_mass @ synthesis, np.eye(2), atol=1e-9
    )
    np.testing.assert_allclose(basis.analysis, synthesis.T @ normalized_mass, atol=1e-9)
    np.testing.assert_allclose(basis.analysis @ synthesis, np.eye(2), atol=1e-9)
    probe = jnp.asarray([0.3, -0.7])
    np.testing.assert_allclose(
        basis.analysis_metric.mv(probe), normalized_mass @ probe, atol=1e-9
    )
    assert basis.quadrature_weights is None
    with pytest.raises(ValueError):
        _ = basis.probability_measure
    with pytest.raises(ValueError):
        phx.discretization.EigenbasisDiscretization(basis)
    assert basis.eigen_solve is not None
    assert int(basis.eigen_solve.status) == int(phx.linalg.eigen.EigenSolveStatus.SUCCESS)
    assert bool(jnp.all(basis.eigen_solve.diagnostics.converged))
    assert basis.report is not None
    assert basis.report.orthonormality_residual < 1e-9
    assert basis.report.source_id == basis.decomposition_id
    spectra = phx.exterior.hodge_sector_spectra(realization, 0)
    assert spectra.harmonic is not None and spectra.coexact is not None
    assert spectra.exact is None
    for result in spectra.solve_results:
        assert int(result.status) == int(phx.linalg.eigen.EigenSolveStatus.SUCCESS)
    kernel = phx.kernels.HodgeSpectralKernel(
        spectra,
        harmonic_multiplier=phx.kernels.HeatSpectralMultiplier(0.0),
        coexact_multiplier=phx.kernels.HeatSpectralMultiplier(0.0),
        normalize_sectors=False,
    )
    entities = jnp.asarray([[0.0], [1.0]])
    expected_covariance = np.trace(mass) * np.linalg.inv(mass)
    np.testing.assert_allclose(
        kernel.matrix(entities, entities), expected_covariance, atol=1e-9
    )
    np.testing.assert_allclose(
        kernel.features(entities) @ kernel.features(entities).T,
        expected_covariance,
        atol=1e-9,
    )
    product = phx.metrix.product_laplacian_eigenbasis((basis, basis), num_modes=None)
    product_metric = np.kron(normalized_mass, normalized_mass)
    np.testing.assert_allclose(
        product.analysis,
        np.asarray(product.synthesis).T @ product_metric,
        atol=1e-9,
    )
    np.testing.assert_allclose(product.analysis @ product.synthesis, np.eye(4), atol=1e-9)
    product_probe = jnp.asarray([0.2, -0.3, 0.4, 0.8])
    np.testing.assert_allclose(
        product.analysis_metric.mv(product_probe),
        product_metric @ product_probe,
        atol=1e-9,
    )
    assert product.quadrature_weights is None

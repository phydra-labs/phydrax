#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _graph(node_count, senders, receivers, weights, *, edge_mask=None):
    senders = jnp.asarray(senders, dtype=jnp.int32)
    receivers = jnp.asarray(receivers, dtype=jnp.int32)
    return phx.graph.GraphIR(
        nodes=jnp.zeros((node_count, 1)),
        edges={"conductance": jnp.asarray(weights, dtype=float)},
        senders=senders,
        receivers=receivers,
        n_node=jnp.asarray([node_count], dtype=jnp.int32),
        n_edge=jnp.asarray([senders.size], dtype=jnp.int32),
        edge_mask=edge_mask,
    )


def _path_graph():
    return _graph(3, [0, 1, 1, 2], [1, 0, 2, 1], [1.0, 1.0, 1.0, 1.0])


def _cycle_graph():
    return _graph(
        4,
        [0, 1, 1, 2, 2, 3, 3, 0],
        [1, 0, 2, 1, 3, 2, 0, 3],
        jnp.ones((8,)),
    )


def test_graph_to_cochain_complex_canonicalizes_reciprocal_edges():
    complex_ir = phx.graph.graph_to_cochain_complex(
        _path_graph(),
        edge_weight_key="conductance",
    )
    incidence = complex_ir.incidences[0]

    assert complex_ir.cell_counts == (3, 2)
    assert np.array_equal(np.asarray(incidence.lower_indices), [0, 1, 1, 2])
    assert np.array_equal(np.asarray(incidence.upper_indices), [0, 0, 1, 1])
    assert np.array_equal(np.asarray(incidence.signs), [-1.0, 1.0, -1.0, 1.0])
    assert jnp.allclose(complex_ir.hodge_stars[0], 1.0)
    assert jnp.allclose(complex_ir.hodge_stars[1], 1.0)


def test_graph_to_cochain_complex_aggregates_parallel_reciprocal_edges():
    graph = _graph(
        2,
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1.0, 2.0, 1.5, 1.5],
    )
    complex_ir = phx.graph.graph_to_cochain_complex(
        graph,
        edge_weight_key="conductance",
    )

    assert complex_ir.cell_counts == (2, 1)
    assert jnp.allclose(complex_ir.hodge_stars[1], jnp.asarray([3.0]))


def test_graph_to_cochain_complex_validates_edge_semantics_and_measure():
    missing_reverse = _graph(2, [0], [1], [1.0])
    inconsistent = _graph(2, [0, 1], [1, 0], [1.0, 2.0])
    self_loop = _graph(2, [0], [0], [1.0])
    nonpositive = _graph(2, [0, 1], [1, 0], [0.0, 0.0])

    with pytest.raises(ValueError, match="reverse"):
        phx.graph.graph_to_cochain_complex(missing_reverse, edge_weight_key="conductance")
    with pytest.raises(ValueError, match="inconsistent"):
        phx.graph.graph_to_cochain_complex(inconsistent, edge_weight_key="conductance")
    with pytest.raises(ValueError, match="Self-loops"):
        phx.graph.graph_to_cochain_complex(self_loop, edge_weight_key="conductance")
    with pytest.raises(ValueError, match="strictly positive"):
        phx.graph.graph_to_cochain_complex(nonpositive, edge_weight_key="conductance")
    with pytest.raises(ValueError, match="strictly positive"):
        phx.graph.graph_to_cochain_complex(
            _graph(3, [0, 1], [1, 0], [1.0, 1.0]),
            edge_weight_key="conductance",
            node_measure="degree",
        )


def test_undirected_once_and_edge_mask_have_explicit_semantics():
    graph = _graph(
        3,
        [0, 1, 0],
        [1, 2, 2],
        [2.0, 3.0, 100.0],
        edge_mask=jnp.asarray([True, True, False]),
    )
    complex_ir = phx.graph.graph_to_cochain_complex(
        graph,
        edge_weight_key="conductance",
        edge_semantics="undirected_once",
    )

    assert complex_ir.cell_counts == (3, 2)
    assert jnp.allclose(complex_ir.hodge_stars[1], jnp.asarray([2.0, 3.0]))


def test_path_cochain_eigenspectrum_recovers_analytic_modes():
    complex_ir = phx.graph.graph_to_cochain_complex(
        _path_graph(), edge_weight_key="conductance"
    )
    basis = phx.graph.cochain_laplacian_eigenbasis(
        complex_ir,
        0,
        num_modes=None,
    )

    assert jnp.allclose(basis.eigenvalues, jnp.asarray([0.0, 1.0, 3.0]), atol=1e-10)
    assert basis.report.exact
    assert basis.zero_mode_count == 1
    assert jnp.allclose(
        basis.eigenfunctions.T
        @ (basis.probability_measure[:, None] * basis.eigenfunctions),
        jnp.eye(3),
        atol=1e-10,
    )


def test_disconnected_isolated_node_remains_a_true_zero_mode():
    complex_ir = phx.graph.graph_to_cochain_complex(
        _graph(3, [0, 1], [1, 0], [1.0, 1.0]),
        edge_weight_key="conductance",
    )
    basis = phx.graph.cochain_laplacian_eigenbasis(
        complex_ir,
        0,
        num_modes=None,
    )

    assert basis.zero_mode_count == 2


def test_truncation_rejects_a_cut_through_degenerate_cycle_modes():
    complex_ir = phx.graph.graph_to_cochain_complex(
        _cycle_graph(), edge_weight_key="conductance"
    )

    with pytest.raises(ValueError, match="degenerate eigenspace"):
        phx.graph.cochain_laplacian_eigenbasis(complex_ir, 0, num_modes=2)

    basis = phx.graph.cochain_laplacian_eigenbasis(complex_ir, 0, num_modes=3)
    assert jnp.allclose(basis.eigenvalues, jnp.asarray([0.0, 2.0, 2.0]), atol=1e-10)
    assert not basis.report.exact
    assert basis.report.next_eigenvalue == pytest.approx(4.0)


def test_sparse_tail_is_explicitly_uncertified_for_product_construction():
    node_count = 8
    forward = jnp.arange(node_count - 1, dtype=jnp.int32)
    backward = forward + 1
    graph = _graph(
        node_count,
        jnp.concatenate((forward, backward)),
        jnp.concatenate((backward, forward)),
        jnp.ones((2 * (node_count - 1),)),
    )
    complex_ir = phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")
    basis = phx.graph.cochain_laplacian_eigenbasis(
        complex_ir,
        0,
        num_modes=2,
        dense_threshold=1,
    )

    assert basis.report.method_id == "sparse-eigsh"
    assert not basis.report.tail_certified
    with pytest.raises(ValueError, match="certified spectral tail"):
        phx.metrix.product_laplacian_eigenbasis(
            (basis, basis),
            num_modes=2,
        )


def test_zero_classification_uses_operator_scale_for_every_solver_path():
    graph = _graph(
        6,
        [0, 1, 2, 3, 4, 5],
        [1, 0, 3, 2, 5, 4],
        [0.1, 0.1, 0.2, 0.2, 1e12, 1e12],
    )
    complex_ir = phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")

    for dense_threshold in (1, 100):
        with pytest.raises(ValueError, match="degenerate eigenspace"):
            phx.graph.cochain_laplacian_eigenbasis(
                complex_ir,
                0,
                num_modes=4,
                dense_threshold=dense_threshold,
            )


def test_harmonic_nullity_uses_the_full_operator_scale_under_sparse_solves():
    graph = _graph(
        8,
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6],
        [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e8, 1e8, 1e8, 1e8, 1e8, 1e8],
    )
    complex_ir = phx.graph.graph_to_cochain_complex(graph, edge_weight_key="conductance")

    with pytest.raises(ValueError, match="exceeds max_modes"):
        phx.graph.compute_harmonic_subspace(
            complex_ir,
            max_modes=2,
            tolerance=1e-9,
            dense_threshold=1,
        )


def test_relative_boundary_basis_zeroes_inactive_rows():
    complex_ir = phx.graph.graph_to_cochain_complex(
        _path_graph(), edge_weight_key="conductance"
    )
    relative = phx.graph.CochainComplexIR(
        complex_ir.cell_counts,
        complex_ir.incidences,
        complex_ir.hodge_stars,
        boundary_masks=(
            jnp.asarray([True, False, True]),
            jnp.asarray([False, False]),
        ),
    )
    basis = phx.graph.cochain_laplacian_eigenbasis(
        relative,
        0,
        num_modes=None,
        boundary_policy="relative",
    )

    assert basis.mode_count == 1
    assert jnp.array_equal(basis.active_mask, jnp.asarray([False, True, False]))
    assert jnp.allclose(basis.eigenfunctions[jnp.asarray([0, 2])], 0.0)

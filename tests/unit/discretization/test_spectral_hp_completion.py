#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    AnisotropicHPattern,
    compact_hp_forest,
    CompatibleAuxiliaryMultigrid,
    CompatibleMortarPlan,
    CompatibleTraceConstraint,
    ConservativeMovingInterfaceTransfer,
    GeometryOrderAdaptation,
    HybridMortarPlan,
    HybridReferenceFamily,
    HybridRefinementPlan,
    initial_finite_element_hp_topology,
    LevelSetCutQuadrature,
    NIrregularMortarPlan,
    refine_anisotropic_hp_cells,
    resize_hp_forest,
    SimplexModalFamily,
    SimplexSBPPlan,
    tensor_hcurl_family,
    tensor_hdiv_family,
    TensorDeRhamComplex,
    TensorDeRhamTransferPlan,
    TensorPiolaMap,
    UnfittedAggregationPlan,
)


def _quad_mesh():
    return CellMesh(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (
            CellBlock(
                "quad",
                "quadrilateral",
                jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
                global_ids=jnp.asarray((10,), dtype=jnp.int64),
            ),
        ),
    )


def test_anisotropic_h_resize_and_compaction_preserve_forest_geometry():
    topology, geometry = initial_finite_element_hp_topology(_quad_mesh(), (3, 2), 8)
    pattern = AnisotropicHPattern(2, (0,))
    refined = refine_anisotropic_hp_cells(
        topology,
        geometry,
        jnp.asarray((10,), dtype=jnp.int64),
        pattern,
    )
    assert refined.topology.active_count == 2
    assert np.count_nonzero(np.asarray(refined.topology.child_valid)[0]) == 2
    resized_topology, resized_geometry = resize_hp_forest(
        refined.topology, refined.geometry, 16
    )
    compacted, compacted_geometry, routes = compact_hp_forest(
        resized_topology, resized_geometry
    )
    assert compacted.capacity == 16
    assert compacted.active_count == 2
    assert compacted_geometry.geometry_id
    assert np.count_nonzero(np.asarray(routes) >= 0) == 3


def test_geometry_order_nirregular_and_cut_quadrature_contracts():
    source_nodes = np.asarray(((0.0,), (1.0,)))
    target_nodes = np.asarray(((0.0,), (0.5,), (1.0,)))
    coordinates = np.asarray((((0.0, 0.0), (1.0, 0.2)),))
    adaptation = GeometryOrderAdaptation(
        source_nodes, target_nodes, (1,), (2,), coordinates
    )
    target = adaptation.apply(coordinates)
    assert target.shape == (1, 3, 2)
    assert adaptation.curvature_indicator >= 0.0

    mortar = NIrregularMortarPlan(
        source_nodes,
        (np.asarray(((0.0,), (0.25,))), np.asarray(((0.5,), (0.75,), (1.0,)))),
        (np.full((2,), 0.25), np.asarray((0.25, 0.25, 0.0))),
    )
    assert len(mortar.coarse_to_patch) == 2
    assert mortar.reproduction_error == 0.0

    points = jnp.linspace(-1.0, 1.0, 9)[:, None]
    cut = LevelSetCutQuadrature(points, jnp.ones((9,)), points[:, 0])
    assert 0.0 < cut.volume_fraction < 1.0


def test_tensor_de_rham_piola_and_simplex_hybrid_families_are_exact():
    for dimension in (2, 3):
        complex_ = TensorDeRhamComplex(2, dimension)
        np.testing.assert_allclose(np.asarray(complex_.grad_curl_defect), 0.0)
        np.testing.assert_allclose(np.asarray(complex_.curl_div_defect), 0.0)

    jacobian = jnp.asarray(((2.0, 0.0), (0.0, 3.0)))
    value = jnp.asarray((1.0, 2.0))
    covariant = TensorPiolaMap("covariant").apply(jacobian, value)
    contravariant = TensorPiolaMap("contravariant").apply(jacobian, value)
    np.testing.assert_allclose(np.asarray(covariant), (0.5, 2.0 / 3.0))
    np.testing.assert_allclose(np.asarray(contravariant), (1.0 / 3.0, 1.0))

    triangle = SimplexModalFamily("triangle", 3)
    tetrahedron = SimplexModalFamily("tetrahedron", 2)
    np.testing.assert_allclose(
        np.asarray(triangle.tabulate(triangle.nodes)),
        np.eye(triangle.nodes.shape[0]),
        atol=2.0e-12,
    )
    assert tetrahedron.nodes.shape[0] == 10
    hcurl = tensor_hcurl_family("hexahedron", 3)
    hdiv = tensor_hdiv_family("hexahedron", 3)
    sample_points = jnp.asarray(((0.2, 0.3, 0.4), (0.7, 0.5, 0.1)))
    assert hcurl.tabulate(sample_points).shape == (
        2,
        hcurl.local_dof_count,
        3,
    )
    assert hdiv.tabulate(sample_points).shape == (
        2,
        hdiv.local_dof_count,
        3,
    )
    assert hcurl.mapping == "covariant_piola"
    assert hdiv.mapping == "contravariant_piola"
    assert HybridReferenceFamily("prism", 2).nodes.shape[1] == 3
    assert HybridReferenceFamily("pyramid", 2).nodes.shape[1] == 3
    prism = HybridReferenceFamily("prism", 2)
    pyramid = HybridReferenceFamily("pyramid", 2)
    np.testing.assert_allclose(
        np.asarray(prism.tabulate(prism.nodes)),
        np.eye(prism.nodes.shape[0]),
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        np.asarray(pyramid.tabulate(pyramid.nodes)),
        np.eye(pyramid.nodes.shape[0]),
        atol=2.0e-11,
    )


def test_compatible_transfers_hybrid_mortars_and_auxiliary_correction():
    source = TensorDeRhamComplex(1, 2)
    target = TensorDeRhamComplex(2, 2)
    transfer = TensorDeRhamTransferPlan(source, target)
    assert transfer.commuting_gradient_error < 1.0e-12
    assert transfer.commuting_curl_error < 1.0e-12

    nodes = jnp.asarray(((0.0,), (0.5,), (1.0,)))
    left = CompatibleTraceConstraint("tangential", nodes, nodes)
    right = CompatibleTraceConstraint("tangential", nodes, nodes)
    compatible = CompatibleMortarPlan(left, right, jnp.eye(3), jnp.eye(3))
    assert compatible.commuting_error == 0.0
    auxiliary = CompatibleAuxiliaryMultigrid(
        jnp.eye(3), jnp.diag(jnp.asarray((1.0, 2.0, 4.0)))
    )
    np.testing.assert_allclose(
        np.asarray(auxiliary.apply(jnp.asarray((1.0, 2.0, 4.0)))),
        1.0,
    )

    hybrid = HybridMortarPlan(
        jnp.asarray(((0.0,), (0.5,), (1.0,))),
        jnp.asarray(((0.0,), (1.0,))),
        jnp.linspace(0.0, 1.0, 5)[:, None],
        1,
    )
    assert hybrid.reproduction_error < 1.0e-12
    refinement = HybridRefinementPlan(
        "prism",
        (
            (jnp.asarray((0.0, 0.0, 0.0)), jnp.asarray((0.5, 1.0, 1.0))),
            (jnp.asarray((0.5, 0.0, 0.0)), jnp.asarray((1.0, 1.0, 1.0))),
        ),
    )
    assert len(refinement.child_maps) == 2


def test_simplex_sbp_unfitted_and_interface_transfers_conserve():
    family = SimplexModalFamily("triangle", 3)
    sbp = SimplexSBPPlan(family)
    assert sbp.polynomial_derivative_error < 1.0e-12

    aggregation = UnfittedAggregationPlan(
        jnp.asarray((0.05, 0.8)),
        jnp.asarray(((1, -1), (0, -1)), dtype=jnp.int32),
    )
    content = jnp.asarray((2.0, 3.0))
    aggregated = aggregation.aggregate(content)
    np.testing.assert_allclose(jnp.sum(aggregated), jnp.sum(content))
    np.testing.assert_allclose(np.asarray(aggregated), (0.0, 5.0))

    transfer = ConservativeMovingInterfaceTransfer(
        jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
        jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
        jnp.ones((2,)),
    )
    np.testing.assert_allclose(
        np.asarray(transfer.apply(jnp.asarray((2.0, 4.0)))),
        (2.0, 4.0),
    )

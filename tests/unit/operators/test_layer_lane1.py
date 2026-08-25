import jax.numpy as jnp
import pytest

import phydrax as phx


def _circle_panelization(*, panels=4, order=6, partition=None):
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    return phx.operators.BoundaryPanelization2D(
        geometry.boundary_atlas,
        panels_per_chart=None if partition is not None else panels,
        quadrature_order=order,
        geometry=geometry,
        partition=partition,
    )


def test_adaptive_layer_reports_global_panel_accuracy_and_regimes():
    panelization = _circle_panelization(panels=4, order=6)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="double",
        density=jnp.ones((panelization.node_count,)),
    )
    plan = phx.operators.LayerEvaluationPlan2D(
        "adaptive",
        accuracy_clearance=1e-4,
        near_ratio=3.0,
        adaptive_plan=phx.integration.AdaptiveQuadraturePlan(
            absolute_tolerance=1e-7,
            relative_tolerance=1e-7,
            max_intervals=12,
            throw=False,
        ),
    )
    result = phx.operators.evaluate_layer_potential(
        potential,
        jnp.asarray([[0.0, 0.0], [0.99, 0.0]]),
        plan,
        target_side="interior",
    )

    assert result.evaluation_report.near_panel_count > 0
    assert result.evaluation_report.far_panel_count > 0
    assert result.evaluation_report.error_kind == "adaptive-embedded-rule"
    assert bool(result.evaluation_report.accuracy_supported)
    assert jnp.allclose(result.values, -1.0, atol=2e-6)


def test_single_layer_self_panel_regularization_is_finite():
    panelization = _circle_panelization(panels=2, order=4)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="single",
        density=jnp.ones((panelization.node_count,)),
    )
    plan = phx.operators.LayerEvaluationPlan2D(
        "adaptive",
        adaptive_plan=phx.integration.AdaptiveQuadraturePlan(
            absolute_tolerance=1e-5,
            relative_tolerance=1e-5,
            max_intervals=16,
            throw=False,
        ),
    )
    result = phx.operators.evaluate_layer_potential(
        potential,
        panelization.points[0],
        plan,
        target_side="boundary",
    )

    assert jnp.isfinite(result.values)
    assert result.evaluation_report.failed_panel_count == 0
    assert result.evaluation_report.error_estimate < 1e-4


def test_kress_partition_preserves_support_and_measure():
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    topology = phx.operators.BoundaryCornerTopology2D(
        geometry.boundary_atlas.num_charts,
        ((0, "start"), (0, "end")),
        interior_angles=(jnp.pi, jnp.pi),
    )
    partition = phx.operators.BoundaryPanelPartition2D(
        geometry.boundary_atlas,
        8,
        grading="kress",
        grading_order=3,
        corner_topology=topology,
    )
    panelization = _circle_panelization(partition=partition)

    assert partition.breakpoints[0][1] < 1.0 / 8.0
    assert panelization.partition.partition_id == partition.partition_id
    assert jnp.allclose(panelization.boundary_measure, 2.0 * jnp.pi, atol=1e-10)


def test_qbx_boundary_average_is_reported_separately():
    panelization = _circle_panelization(panels=4, order=8)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="double",
        density=jnp.ones((panelization.node_count,)),
    )
    result = phx.operators.evaluate_layer_potential(
        potential,
        panelization.points[0],
        phx.operators.LayerEvaluationPlan2D(
            "qbx",
            qbx_order=4,
            qbx_radius_factor=0.25,
            adaptive_plan=phx.integration.AdaptiveQuadraturePlan(
                absolute_tolerance=1e-3,
                relative_tolerance=1e-3,
                throw=False,
            ),
        ),
        target_side="boundary",
    )

    assert jnp.isfinite(result.values)
    assert (
        result.evaluation_report.error_kind
        == "qbx-coefficient-quadrature-and-truncation"
    )
    assert result.evaluation_report.near_panel_count == 1
    assert float(result.values) == pytest.approx(-0.5, abs=2e-2)

def test_barycentric_self_weights_are_partition_unity_and_nodal_delta():
    panelization = _circle_panelization(panels=1, order=5)
    from phydrax._interpolation import barycentric_basis

    nodes = panelization.references[: panelization.quadrature_order, 0]
    differences = nodes[:, None] - nodes[None, :]
    weights = jnp.reciprocal(jnp.prod(differences + jnp.eye(nodes.size), axis=1))
    basis = jnp.stack([barycentric_basis(node, nodes, weights) for node in nodes])

    assert jnp.allclose(jnp.sum(basis, axis=1), 1.0)
    assert jnp.allclose(basis, jnp.eye(nodes.size), atol=1e-12)


def test_helmholtz_cfie_requires_and_reports_explicit_self_policy():
    panelization = _circle_panelization(panels=2, order=4)
    quadrature = phx.integration.AdaptiveQuadraturePlan(
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
        max_intervals=16,
        throw=False,
    )
    result = phx.solver.solve_exterior_helmholtz_dirichlet_2d(
        panelization,
        jnp.zeros((panelization.node_count,), dtype=complex),
        2.0,
        quadrature=quadrature,
    )

    assert bool(result.valid)
    assert result.assembly_report.corrected_block_count == panelization.panel_count
    assert bool(result.assembly_report.accuracy_supported)
    assert jnp.all(jnp.isfinite(result.density))


def test_3d_surface_layer_uses_continuous_target_evidence():
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    panelization = phx.operators.SurfacePanelization3D(
        geometry.boundary_atlas,
        quadrature_order=3,
        geometry=geometry,
    )
    potential = phx.operators.LaplaceLayerPotential3D(
        panelization,
        kind="single",
        density=jnp.ones((panelization.node_count,)),
    )
    values, report = phx.operators.evaluate_laplace_layer_3d(
        potential,
        jnp.asarray([[0.0, 0.0, 0.0]]),
        target_side="interior",
    )

    assert jnp.isfinite(values[0])
    assert bool(report.pde_membership_valid)
    assert report.target_fingerprint


def test_3d_qbx_directional_expansion_covers_real_and_complex_layers():
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    panelization = phx.operators.SurfacePanelization3D(
        geometry.boundary_atlas,
        quadrature_order=2,
        geometry=geometry,
    )
    triangle_plan = phx.integration.AdaptiveTrianglePlan(
        phx.integration.CubatureRule("triangle", 1),
        phx.integration.CubatureRule("triangle", 2),
        absolute_tolerance=10.0,
        relative_tolerance=10.0,
        max_cells=1,
        throw=False,
    )
    targets = jnp.asarray([[0.0, 0.0, 0.0]])
    density = jnp.ones((panelization.node_count,))

    single = phx.operators.evaluate_qbx_3d(
        phx.operators.LaplaceLayerPotential3D(
            panelization,
            kind="single",
            density=density,
        ),
        targets,
        target_side="interior",
        order=1,
        radius_factor=0.05,
        triangle_plan=triangle_plan,
    )
    double = phx.operators.evaluate_qbx_3d(
        phx.operators.LaplaceLayerPotential3D(
            panelization,
            kind="double",
            density=density,
        ),
        targets,
        target_side="interior",
        order=1,
        radius_factor=0.05,
        triangle_plan=triangle_plan,
    )
    helmholtz = phx.operators.evaluate_qbx_3d(
        phx.operators.HelmholtzLayerPotential3D(
            panelization,
            2.0,
            kind="single",
            density=density.astype(complex),
        ),
        targets,
        target_side="interior",
        order=1,
        radius_factor=0.05,
        triangle_plan=triangle_plan,
    )

    assert jnp.allclose(single.values, 0.5, atol=1e-2)
    assert jnp.allclose(double.values, -0.5, atol=1e-2)
    for result in (single, double, helmholtz):
        assert result.values.shape == (1,)
        assert jnp.all(jnp.isfinite(result.values))
        assert jnp.isfinite(result.error_estimate)
        assert int(result.status) == 0
        assert bool(result.accuracy_supported)
        assert int(result.num_evaluations) > 0


def test_helmholtz_qbx_uses_directional_hankel_expansion():
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    panelization = _circle_panelization(panels=1, order=2)
    potential = phx.operators.HelmholtzLayerPotential2D(
        panelization,
        2.0,
        kind="single",
        density=jnp.ones((panelization.node_count,), dtype=complex),
    )
    result = phx.operators.evaluate_layer_potential(
        potential,
        panelization.points[0][None, :],
        phx.operators.LayerEvaluationPlan2D(
            "qbx",
            qbx_order=1,
            qbx_radius_factor=0.05,
            adaptive_plan=phx.integration.AdaptiveQuadraturePlan(
                absolute_tolerance=10.0,
                relative_tolerance=10.0,
                max_intervals=1,
                throw=False,
            ),
        ),
        target_side="boundary",
    )

    assert result.values.shape == (1,)
    assert jnp.all(jnp.isfinite(result.values))
    assert jnp.isfinite(result.evaluation_report.error_estimate)
    assert bool(result.evaluation_report.accuracy_supported)


def test_direct_near_far_backend_matches_direct_representation():
    panelization = _circle_panelization()
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="double",
        density=jnp.ones((panelization.node_count,)),
    )
    targets = jnp.asarray([[0.0, 0.0], [0.5, 0.1]])
    backend = phx.operators.DirectNearFarReferenceBackend2D()
    reference = backend.evaluate(potential, targets, near_ratio=3.0)

    assert backend.backend_id == "direct-near-far-reference-2d-v1"
    assert jnp.allclose(reference.values, potential._evaluate_direct(targets))
    assert bool(reference.accuracy_supported)
    assert reference.near_panel_count + reference.far_panel_count == (
        targets.shape[0] * panelization.panel_count
    )


def test_double_layer_diagonal_limit_is_analytic_and_orientation_aware():
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    uniform = _circle_panelization(panels=4, order=4)
    topology = phx.operators.BoundaryCornerTopology2D(
        geometry.boundary_atlas.num_charts,
        ((0, "start"), (0, "end")),
    )
    graded_partition = phx.operators.BoundaryPanelPartition2D(
        geometry.boundary_atlas,
        4,
        grading="kress",
        corner_topology=topology,
    )
    graded = phx.operators.BoundaryPanelization2D(
        geometry.boundary_atlas,
        quadrature_order=4,
        geometry=geometry,
        partition=graded_partition,
    )
    expected = -1.0 / (4.0 * jnp.pi)
    for panelization in (uniform, graded):
        matrix = phx.operators.double_layer_principal_value_matrix(panelization)
        assert jnp.allclose(
            jnp.diag(matrix) / panelization.weights,
            expected,
            atol=1e-10,
        )

    atlas = geometry.boundary_atlas
    reversed_atlas = phx.geometry.BoundaryAtlas(
        atlas.mapping,
        source_entity_ids=atlas.source_entity_ids,
        source_id="reversed-circle-diagonal-test",
        physical_tags=atlas.physical_tags,
        orientation=-atlas.orientation,
        seam_owner=atlas.seam_owner,
        trim_domains=atlas.trim_domains,
    )
    reversed_panelization = phx.operators.BoundaryPanelization2D(
        reversed_atlas,
        panels_per_chart=4,
        quadrature_order=4,
    )
    reversed_matrix = phx.operators.double_layer_principal_value_matrix(
        reversed_panelization
    )
    assert jnp.allclose(
        jnp.diag(reversed_matrix) / reversed_panelization.weights,
        -expected,
        atol=1e-10,
    )


def test_rcip_uses_nonzero_nested_corner_hierarchy():
    coarse = jnp.asarray(((2.0, 0.2), (0.1, 1.5)))
    fine = (
        jnp.asarray(
            (
                (2.0, 0.1, 0.0, 0.0),
                (0.0, 1.8, 0.1, 0.0),
                (0.0, 0.0, 2.2, 0.2),
                (0.0, 0.0, 0.1, 1.7),
            )
        ),
        jnp.eye(8) * 2.0,
    )
    restriction_0 = jnp.asarray(
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    )
    restriction_1 = jnp.concatenate(
        (jnp.eye(4), jnp.zeros((4, 4))),
        axis=1,
    )
    preconditioner = phx.operators.RCIPPreconditioner2D(
        coarse,
        fine,
        (restriction_0, restriction_1),
        (restriction_0.T, restriction_1.T),
        topology_id="corner-hierarchy",
    )
    applied = preconditioner.apply(jnp.asarray((1.0, -0.5)))

    assert preconditioner.levels == 2
    assert len(preconditioner.compressed_inverses) == 3
    assert jnp.all(jnp.isfinite(applied))
    assert preconditioner.preconditioner_id


def test_laplace_fmm_order_sweep_matches_direct_far_field():
    panelization = _circle_panelization(panels=8, order=4)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="single",
        density=jnp.linspace(0.5, 1.5, panelization.node_count),
    )
    targets = jnp.stack(
        (jnp.linspace(2.8, 3.2, 32), jnp.zeros((32,))),
        axis=-1,
    )
    direct = potential._evaluate_direct(targets)
    errors = []
    for expansion_order in (2, 4, 8):
        backend = phx.operators.LaplaceFMMBackend2D(
            potential,
            expansion_order=expansion_order,
            leaf_size=8,
            opening_angle=0.5,
        )
        evaluation = backend.evaluate(
            potential,
            targets,
            absolute_tolerance=1e-5,
        )
        errors.append(float(jnp.max(jnp.abs(evaluation.values - direct))))
        assert evaluation.m2m_translations > 0
        assert evaluation.m2l_translations > 0
        assert evaluation.l2l_translations > 0
    assert errors[-1] <= errors[0]


def test_fmm_mixed_excluded_leaf_is_kept_in_near_correction():
    panelization = _circle_panelization(panels=8, order=4)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="single",
        density=jnp.ones((panelization.node_count,)),
    )
    backend = phx.operators.LaplaceFMMBackend2D(
        potential,
        expansion_order=6,
        leaf_size=4,
    )
    _, _, _, near_sources, _ = backend.local_expansions(
        potential,
        jnp.asarray([[2.8, 0.0], [3.2, 0.0]]),
        excluded_source_indices=(0,),
    )
    near_blocks = tuple(
        tuple(int(index) for index in source_block)
        for target_sources in near_sources
        for source_block in target_sources
    )

    assert any(0 in block and any(index != 0 for index in block) for block in near_blocks)


def test_global_qbx_fmm_reports_independent_error_channels():
    panelization = _circle_panelization(panels=4, order=4)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="single",
        density=jnp.ones((panelization.node_count,)),
    )
    backend = phx.operators.LaplaceFMMBackend2D(
        potential,
        expansion_order=4,
        leaf_size=4,
    )
    evaluation = phx.operators.evaluate_global_qbx_fmm_2d(
        potential,
        backend,
        panelization.points[0][None, :],
        target_side="boundary",
        expansion_order=2,
        radius_factor=0.05,
        adaptive_plan=phx.integration.AdaptiveQuadraturePlan(
            absolute_tolerance=1e-2,
            relative_tolerance=1e-2,
            max_intervals=12,
            throw=False,
        ),
    )

    direct = phx.operators.evaluate_layer_potential(
        potential,
        panelization.points[0][None, :],
        phx.operators.LayerEvaluationPlan2D(
            "qbx",
            qbx_order=2,
            qbx_radius_factor=0.05,
            adaptive_plan=phx.integration.AdaptiveQuadraturePlan(
                absolute_tolerance=1e-2,
                relative_tolerance=1e-2,
                max_intervals=12,
                throw=False,
            ),
        ),
        target_side="boundary",
    )
    assert jnp.allclose(evaluation.values, direct.values, atol=2e-2)
    assert evaluation.m2l_translations > 0
    assert evaluation.l2l_translations > 0
    assert evaluation.near_panel_count > 0
    assert jnp.all(
        jnp.isfinite(
            jnp.asarray(
                (
                    evaluation.coefficient_quadrature_error,
                    evaluation.fmm_truncation_error,
                    evaluation.expansion_truncation_error,
                    evaluation.error_estimate,
                )
            )
        )
    )
    assert evaluation.error_estimate >= evaluation.fmm_truncation_error

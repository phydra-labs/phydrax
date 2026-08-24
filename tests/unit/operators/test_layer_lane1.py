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
    assert result.evaluation_report.error_kind == "qbx-taylor-truncation"
    assert result.evaluation_report.near_panel_count == 1
    assert float(result.values) == pytest.approx(-0.5, abs=2e-2)

def test_barycentric_self_weights_are_partition_unity_and_nodal_delta():
    panelization = _circle_panelization(panels=1, order=5)
    from phydrax._interpolation import barycentric_basis

    nodes = panelization.references[:, 0]
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


def test_direct_near_far_backend_matches_direct_representation():
    panelization = _circle_panelization()
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="double",
        density=jnp.ones((panelization.node_count,)),
    )
    targets = jnp.asarray([[0.0, 0.0], [0.5, 0.1]])
    backend = phx.operators.DirectNearFarBackend2D()
    accelerated = backend.evaluate(potential, targets, near_ratio=3.0)

    assert backend.backend_id == "direct-near-far-2d-v1"
    assert jnp.allclose(accelerated.values, potential._evaluate_direct(targets))
    assert bool(accelerated.accuracy_supported)
    assert accelerated.near_panel_count + accelerated.far_panel_count == (
        targets.shape[0] * panelization.panel_count
    )

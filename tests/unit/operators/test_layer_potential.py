#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _circle_panelization(*, panels=8, order=8):
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile()
    return phx.operators.BoundaryPanelization2D(
        geometry.boundary_atlas,
        panels_per_chart=panels,
        quadrature_order=order,
        geometry=geometry,
    )


def _point_batch(domain, points):
    layout = phx.domain.SampleLayout((("x",),)).canonicalize(domain.labels)
    axis_names = layout.axis_names
    assert axis_names is not None
    return phx.domain.PointBatch(
        {
            "x": cx.Field(
                jnp.asarray(points, dtype=float).reshape((-1, 2)),
                dims=(axis_names[0], None),
            )
        },
        layout,
    )


def test_panelization_rejects_same_named_geometrically_distinct_support():
    source_geometry = phx.geometry.Circle(
        (0.0, 0.0),
        1.0,
        feature_id="shared-layer-support",
    ).compile()
    different_geometry = phx.geometry.Circle(
        (0.25, -0.1),
        1.2,
        feature_id="shared-layer-support",
    ).compile()
    phx.operators.BoundaryPanelization2D(
        source_geometry.boundary_atlas,
        panels_per_chart=2,
        quadrature_order=2,
        geometry=source_geometry,
    )
    with pytest.raises(ValueError, match="same support"):
        phx.operators.BoundaryPanelization2D(
            source_geometry.boundary_atlas,
            panels_per_chart=2,
            quadrature_order=2,
            geometry=different_geometry,
        )


def test_panelization_measure_and_reports_separate_pde_from_accuracy():
    panelization = _circle_panelization()
    assert jnp.allclose(panelization.boundary_measure, 2.0 * jnp.pi, atol=2e-11)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="double",
        density=jnp.ones((panelization.node_count,)),
    )
    domain = phx.domain.GeometryDomain(phx.geometry.Circle((0.0, 0.0), 1.0).compile())
    field = domain.Model("x")(potential)
    certificate = phx.equations.trial_space_certificate(field)
    assert certificate.exactness == "algebraic"
    assert certificate.validity_region == "off-singular-support"
    assert certificate.singular_support_id == panelization.source_support_id
    assert "minimum_declared_clearance" not in vars(certificate)

    interior_values, interior_report = potential.evaluate_with_report(
        jnp.asarray([[0.0, 0.0], [0.2, -0.1]]),
        target_side="interior",
        accuracy_clearance=0.1,
    )
    assert bool(interior_report.pde_membership_valid)
    assert bool(interior_report.accuracy_supported)
    assert jnp.allclose(interior_values, -1.0, atol=2e-10)
    interior_batch = _point_batch(domain, jnp.asarray([[0.0, 0.0], [0.2, -0.1]]))
    audit = phx.equations.audit_trial_space(
        field,
        interior_batch,
        admissibility=interior_report,
    )
    assert bool(audit.valid)
    assert bool(audit.pde_membership_valid)
    assert bool(audit.evaluation_accuracy_supported)
    assert audit.admissibility_report_id == interior_report.report_id
    with pytest.raises(TypeError, match="admissibility evidence"):
        phx.equations.audit_trial_space(field, interior_batch)
    mismatched_batch = _point_batch(
        domain,
        jnp.asarray([[0.0, 0.0], [0.25, -0.1]]),
    )
    with pytest.raises(ValueError, match="does not match the audit batch"):
        phx.equations.audit_trial_space(
            field,
            mismatched_batch,
            admissibility=interior_report,
        )

    other_geometry = phx.geometry.Circle(
        (0.0, 0.0),
        1.0,
        feature_id="different-layer-support",
    ).compile()
    other_panelization = phx.operators.BoundaryPanelization2D(
        other_geometry.boundary_atlas,
        panels_per_chart=2,
        quadrature_order=2,
        geometry=other_geometry,
    )
    other_support_report = phx.operators.LayerPotentialTargetReport(
        jnp.asarray([[0.0, 0.0], [0.2, -0.1]]),
        other_panelization,
        target_side="interior",
    )
    assert bool(other_support_report.pde_membership_valid)
    with pytest.raises(ValueError, match="certified singular support"):
        phx.equations.audit_trial_space(
            field,
            interior_batch,
            admissibility=other_support_report,
        )

    near_report = phx.operators.LayerPotentialTargetReport(
        jnp.asarray([[0.999, 0.0]]),
        panelization,
        target_side="interior",
        accuracy_clearance=0.01,
    )
    assert bool(near_report.pde_membership_valid)
    assert not bool(near_report.accuracy_supported)

    collision_report = phx.operators.LayerPotentialTargetReport(
        panelization.points[0],
        panelization,
        target_side="interior",
    )
    assert not bool(collision_report.pde_membership_valid)
    collision_batch = _point_batch(domain, panelization.points[0])
    with pytest.raises(ValueError, match="singular support"):
        phx.equations.audit_trial_space(
            field,
            collision_batch,
            admissibility=collision_report,
        )

    nonnode_boundary = jnp.asarray([[jnp.cos(0.123), jnp.sin(0.123)]])
    assert not bool(
        jnp.any(jnp.all(nonnode_boundary[:, None, :] == panelization.points, axis=-1))
    )
    boundary_report = phx.operators.LayerPotentialTargetReport(
        nonnode_boundary,
        panelization,
        target_side="interior",
    )
    assert bool(boundary_report.intersects_singular_support)
    assert not bool(boundary_report.pde_membership_valid)
    with pytest.raises(ValueError, match="singular support"):
        phx.equations.audit_trial_space(
            field,
            _point_batch(domain, nonnode_boundary),
            admissibility=boundary_report,
        )

    bare_panelization = phx.operators.BoundaryPanelization2D(
        panelization.atlas,
        panels_per_chart=2,
        quadrature_order=2,
    )
    with pytest.raises(TypeError, match="certified geometry"):
        phx.operators.LayerPotentialTargetReport(
            jnp.asarray([[0.0, 0.0]]),
            bare_panelization,
            target_side="interior",
        )
    approximation = potential.approximation_report()
    assert approximation.approximation_id
    assert approximation.panelization_id == panelization.panelization_id


def test_finite_layer_sum_is_harmonic_independently_of_quadrature_accuracy():
    panelization = _circle_panelization(panels=3, order=3)
    density = jnp.linspace(-0.7, 1.1, panelization.node_count)
    potential = phx.operators.LaplaceLayerPotential2D(
        panelization,
        kind="single",
        density=density,
    )
    point = jnp.asarray([0.15, -0.25])
    residual = jnp.trace(jax.hessian(potential)(point))
    assert jnp.allclose(residual, 0.0, atol=5e-10)

    refined = phx.operators.LaplaceLayerPotential2D(
        _circle_panelization(panels=6, order=6),
        kind="single",
    )
    refined_density = jnp.linspace(-0.7, 1.1, refined.panelization.node_count)
    refined = refined.with_density(refined_density)
    refined_residual = jnp.trace(jax.hessian(refined)(point))
    assert jnp.allclose(refined_residual, 0.0, atol=5e-10)


def test_interior_circle_dirichlet_double_layer_recovers_constant_solution():
    panelization = _circle_panelization(panels=8, order=8)
    result = phx.solver.solve_interior_laplace_dirichlet_2d(
        panelization,
        jnp.ones((panelization.node_count,)),
    )
    assert bool(result.valid)
    assert float(result.boundary_residual_norm) < 1e-9
    targets = jnp.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [-0.4, 0.3],
        ]
    )
    values, report = result.potential.evaluate_with_report(
        targets,
        target_side="interior",
        accuracy_clearance=0.1,
    )
    assert bool(report.pde_membership_valid)
    assert bool(report.accuracy_supported)
    assert jnp.allclose(values, 1.0, atol=3e-3, rtol=3e-3)

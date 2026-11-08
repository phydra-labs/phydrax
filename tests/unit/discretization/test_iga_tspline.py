#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import TMesh2D
from phydrax.discretization.iga import (
    IsogeometricQuadraturePolicy,
    TSplineGeometryState,
    TSplineH1FieldSpec,
    TSplineIsogeometricPlan,
    TSplineQueryRoute,
    UVTrimLoop,
    UVTrimmedSurface,
)
from phydrax.discretization.iga._tspline import TSplineReferenceActions
from phydrax.geometry.brep import TSplineSurfacePatch


def _mesh(cells_per_axis: int = 2) -> TMesh2D:
    breaks = tuple(Fraction(index, cells_per_axis) for index in range(cells_per_axis + 1))
    return TMesh2D.tensor_product(breaks, breaks)


def _surface_patch(
    mesh: TMesh2D, weights: np.ndarray | None = None
) -> TSplineSurfacePatch:
    greville_points = tuple(
        anchor.local_knots.greville_parameter for anchor in mesh.anchors
    )
    exact_points = tuple((u, v, u * v) for u, v in greville_points)
    points = np.asarray([[float(value) for value in point] for point in exact_points])
    raw_weights = np.ones((mesh.coefficient_count,)) if weights is None else weights
    exact_weights = tuple(Fraction.from_float(float(value)) for value in raw_weights)
    return TSplineSurfacePatch(
        mesh,
        points,
        raw_weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )


def _outer_trim() -> UVTrimmedSurface:
    return UVTrimmedSurface(UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))))


def _plan(
    mesh: TMesh2D,
    fields: TSplineH1FieldSpec,
    *,
    trim: UVTrimmedSurface | None = None,
) -> TSplineIsogeometricPlan:
    geometry = TSplineGeometryState(_surface_patch(mesh))
    return TSplineIsogeometricPlan(
        geometry,
        fields,
        trim=_outer_trim() if trim is None else trim,
        quadrature_policy=IsogeometricQuadraturePolicy(6),
    )


def test_cad_surface_bank_and_iga_map_share_one_authoritative_source() -> None:
    mesh = _mesh()
    patch = _surface_patch(mesh)
    geometry = TSplineGeometryState(patch)
    field = TSplineH1FieldSpec(
        "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
    )
    prepared = TSplineIsogeometricPlan(
        geometry,
        field,
        trim=_outer_trim(),
        quadrature_policy=IsogeometricQuadraturePolicy(5),
    ).prepare(numeric_version="shared-cad-source")
    points = np.asarray(((0.13, 0.29), (0.71, 0.38), (0.46, 0.84)))
    route = prepared.prepare_query_route(points)

    assert isinstance(route, TSplineQueryRoute)
    assert route.trim_id == prepared.trim.surface_id
    assert prepared.default_runtime.source_definition_id == patch.source_definition_id
    assert prepared.default_runtime.surface_patch is patch
    np.testing.assert_allclose(
        prepared.evaluate_geometry_route(route, maximum_order=1).values,
        np.column_stack((points, points[:, 0] * points[:, 1])),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert prepared.default_runtime.exact_homogeneous_control_points == (
        patch.exact_target_homogeneous
    )


def test_local_refinement_prolongs_geometry_and_weighted_field_separately() -> None:
    mesh = _mesh(4)
    field_weights_exact = tuple(
        Fraction(1) + Fraction(anchor.anchor_id + 1, 97) for anchor in mesh.anchors
    )
    field_weights = np.asarray([float(value) for value in field_weights_exact])
    field = TSplineH1FieldSpec(
        "rational-field",
        field_weights,
        exact_weights=field_weights_exact,
        anchor_count=mesh.coefficient_count,
    )
    plan = _plan(mesh, field)
    source_prepared = plan.prepare(numeric_version="source")
    marked = mesh.cells[len(mesh.cells) // 2]
    refinement = plan.refine((marked.cell_id,))
    target_plan = refinement.target_plan
    target_prepared = target_plan.prepare(numeric_version="target")
    exact_coefficients = tuple(
        (Fraction((index % 7) - 3, 11),) for index in range(mesh.coefficient_count)
    )
    source_coefficients = jnp.asarray(
        [float(row[0]) for row in exact_coefficients], dtype=jnp.float64
    )
    assert not np.allclose(source_coefficients, source_coefficients[0])
    target_coefficients, target_weights = refinement.prolong_field(
        "rational-field", source_coefficients
    )
    assert target_weights is not None

    query_points = np.asarray(((0.117, 0.223), (0.529, 0.681), (0.883, 0.347)))
    source_route = source_prepared.prepare_query_route(query_points)
    target_route = target_prepared.prepare_query_route(query_points)
    source_values = source_prepared.evaluate_field_route(
        "rational-field", source_route, source_coefficients
    )
    target_values = target_prepared.evaluate_field_route(
        "rational-field", target_route, target_coefficients
    )
    assert np.all(np.isfinite(np.asarray(source_values)))
    assert not np.allclose(np.asarray(source_values), 7.0 / 3.0)
    np.testing.assert_allclose(target_values, source_values, rtol=2.0e-12, atol=2.0e-12)
    target_field_weights = target_plan.fields[0].weights
    assert target_field_weights is not None
    np.testing.assert_allclose(
        target_weights,
        target_field_weights,
        rtol=0.0,
        atol=0.0,
    )
    geometry_weights = np.asarray(target_plan.geometry.weights)
    np.testing.assert_array_equal(
        geometry_weights, np.ones((target_plan.geometry.mesh.coefficient_count,))
    )
    assert not np.allclose(np.asarray(target_weights), geometry_weights)

    exact_values, exact_weights = refinement.prolong_exact_field(
        "rational-field", exact_coefficients
    )
    assert exact_weights == target_plan.fields[0].exact_weights
    np.testing.assert_allclose(
        np.asarray(target_coefficients),
        np.asarray([float(row[0]) for row in exact_values]),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    scalar_exact_values, scalar_exact_weights = refinement.prolong_exact_field(
        "rational-field",
        tuple(Fraction(7, 3) for _ in range(mesh.coefficient_count)),
    )
    assert all(value == Fraction(7, 3) for value in scalar_exact_values)
    assert scalar_exact_weights == exact_weights
    assert target_plan.geometry.surface_patch.source_mesh.mesh_id == mesh.mesh_id
    assert len(target_plan.geometry.surface_patch.transfer_plans) == 1
    target_plan.geometry.surface_patch.validate_exact_transfer_chain()
    second_marked = target_plan.geometry.mesh.cells[
        len(target_plan.geometry.mesh.cells) // 2
    ]
    second_refinement = target_plan.refine((second_marked.cell_id,))
    second_patch = second_refinement.target_plan.geometry.surface_patch
    assert len(second_patch.transfer_plans) == 2
    second_patch.validate_exact_transfer_chain()


def test_trim_routes_are_admitted_once_then_reused_in_jax() -> None:
    mesh = _mesh()
    hole = UVTrimLoop(((0.3, 0.3), (0.3, 0.6), (0.6, 0.6), (0.6, 0.3)))
    trim = UVTrimmedSurface(
        UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        (hole,),
    )
    prepared = _plan(
        mesh,
        TSplineH1FieldSpec(
            "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
        ),
        trim=trim,
    ).prepare()
    points = jnp.asarray(((0.17, 0.23), (0.77, 0.81)))
    route = prepared.prepare_query_route(points)
    coefficients = jnp.arange(mesh.coefficient_count, dtype=jnp.float64)
    expected = prepared.evaluate_field_route("u", route, coefficients)
    actual = jax.jit(lambda values: prepared.evaluate_field_route("u", route, values))(
        coefficients
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)

    with pytest.raises(ValueError, match="outside the trimmed source surface"):
        prepared.prepare_query_route(((0.45, 0.45),))

    with pytest.raises(TypeError, match="real numeric coordinates"):
        prepared.prepare_query_route(np.asarray(((0.45 + 0.1j, 0.2),)))

    foreign = _plan(
        mesh,
        TSplineH1FieldSpec(
            "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
        ),
    ).prepare()
    with pytest.raises(ValueError, match="does not match this prepared trimmed surface"):
        foreign.evaluate_geometry_route(route)


def test_raw_weight_gauge_is_a_fixed_topology_jvp_vjp_and_hessian_null_direction() -> (
    None
):
    mesh = _mesh()
    geometry = TSplineGeometryState(_surface_patch(mesh))
    plan = TSplineIsogeometricPlan(
        geometry,
        TSplineH1FieldSpec(
            "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
        ),
        trim=_outer_trim(),
        quadrature_policy=IsogeometricQuadraturePolicy(5),
    )
    prepared = plan.prepare(numeric_version="gauge")
    runtime = prepared.default_runtime
    points = jnp.asarray(((0.27, 0.41), (0.72, 0.62)))
    route = prepared.prepare_query_route(points)
    source_points = runtime.surface_patch.control_points
    source_weights = runtime.surface_patch.weights

    def mapped(gauge: jax.Array) -> jax.Array:
        control_points, weights = runtime.realize_parameter_bank(
            source_points,
            source_weights * gauge,
            source_authority_id=runtime.source_authority_id,
        )
        return prepared.basis_realization.evaluate_rational(
            route.parameters,
            control_points,
            weights,
            cell_ids=route.cell_ids,
            maximum_order=0,
        ).values

    gauge = jnp.asarray(1.0)
    tangent = jax.jvp(mapped, (gauge,), (jnp.ones_like(gauge),))[1]
    pullback = jax.vjp(mapped, gauge)[1](jnp.ones_like(mapped(gauge)))[0]
    curvature = jax.hessian(lambda scale: jnp.sum(mapped(scale)))(gauge)
    scaled_geometry = geometry.rebind_numeric_bank(
        source_points,
        source_weights * 8.5,
        revision="gauge-rescaled",
    )
    np.testing.assert_allclose(
        np.asarray(scaled_geometry.surface_patch.evaluate(points)),
        np.asarray(geometry.surface_patch.evaluate(points)),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(tangent, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(pullback, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(curvature, 0.0, atol=2.0e-13)


def test_numeric_source_changes_require_a_fresh_exact_revision_and_degenerate_maps_refuse() -> (
    None
):
    mesh = _mesh()
    geometry = TSplineGeometryState(_surface_patch(mesh))
    field = TSplineH1FieldSpec(
        "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
    )
    prepared = TSplineIsogeometricPlan(
        geometry,
        field,
        trim=_outer_trim(),
        quadrature_policy=IsogeometricQuadraturePolicy(5),
    ).prepare()
    moved_points = geometry.surface_patch.control_points + jnp.asarray((0.0, 0.0, 0.125))
    edited_patch = eqx.tree_at(
        lambda source: source.control_points,
        geometry.surface_patch,
        moved_points,
    )
    with pytest.raises(ValueError, match="Concrete source-bank edits"):
        edited_patch.evaluate(jnp.asarray(((0.3, 0.4),)))
    with pytest.raises(ValueError, match="rebind_numeric_bank"):
        prepared.default_runtime.realize_parameter_bank(
            moved_points,
            geometry.surface_patch.weights,
            source_authority_id=geometry.source_authority_id,
        )

    with pytest.raises(ValueError, match="new source revision"):
        geometry.rebind_numeric_bank(
            moved_points,
            geometry.surface_patch.weights,
            revision=geometry.revision,
        )
    rebound = geometry.rebind_numeric_bank(
        moved_points,
        geometry.surface_patch.weights,
        revision="translated-surface",
    )
    refreshed = prepared.prepare_runtime(rebound, numeric_version="translated")
    assert refreshed.source_definition_id == rebound.source_definition_id
    assert refreshed.source_definition_id != geometry.source_definition_id

    collapsed = TSplineGeometryState(
        mesh,
        np.zeros((mesh.coefficient_count, 3)),
        np.ones((mesh.coefficient_count,)),
        tuple(anchor.anchor_id for anchor in mesh.anchors),
    )
    assert collapsed.surface_patch.is_c1_along(0, 0.5, 0.1, 0.9)
    with pytest.raises(ValueError, match="physical-map certificates"):
        TSplineIsogeometricPlan(
            collapsed,
            field,
            trim=_outer_trim(),
            quadrature_policy=IsogeometricQuadraturePolicy(5),
        )


def test_reference_hessian_and_transpose_share_the_exact_local_route() -> None:
    mesh = _mesh(1)
    realization = mesh.realize()
    extraction = realization.extractions[0]
    points = np.asarray(((0.17, 0.31), (0.58, 0.77), (0.84, 0.22)))
    values = np.asarray(extraction.evaluate_reference(points))
    gradients = np.stack(
        (
            np.asarray(extraction.evaluate_reference(points, derivative=(1, 0))),
            np.asarray(extraction.evaluate_reference(points, derivative=(0, 1))),
        ),
        axis=-1,
    )
    hessians = np.empty((points.shape[0], extraction.local_width, 2, 2))
    for first_axis, second_axis, derivative in (
        (0, 0, (2, 0)),
        (0, 1, (1, 1)),
        (1, 0, (1, 1)),
        (1, 1, (0, 2)),
    ):
        hessians[:, :, first_axis, second_axis] = np.asarray(
            extraction.evaluate_reference(points, derivative=derivative)
        )
    exact_weights = tuple(
        Fraction(1) + Fraction(row + 1, 19) for row in range(mesh.coefficient_count)
    )
    action = TSplineReferenceActions(
        values[None, ...],
        gradients[None, ...],
        hessians[None, ...],
        np.asarray(extraction.coefficient_rows)[None, :],
        np.asarray([float(value) for value in exact_weights]),
        field_weights_from_geometry=False,
        query_shape=(points.shape[0],),
        topology_id=mesh.mesh_id,
        geometry_layout_id="tspline-adjoint-layout",
        structural_id=realization.realization_id,
        global_coefficient_count=mesh.coefficient_count,
        maximum_derivative_order=2,
        is_trace=False,
    )
    coefficients = jnp.asarray(np.linspace(-0.7, 1.1, extraction.local_width))[None, :]
    hessian_cotangent = jnp.asarray(
        np.linspace(0.3, 1.4, points.shape[0] * 4).reshape((1, points.shape[0], 2, 2))
    )
    forward = action.reference_hessian(None, coefficients)
    transpose = action.reference_hessian_transpose(None, hessian_cotangent)

    assert forward.shape == hessian_cotangent.shape
    np.testing.assert_allclose(forward, jnp.swapaxes(forward, -1, -2), atol=2.0e-13)
    np.testing.assert_allclose(
        jnp.vdot(forward, hessian_cotangent),
        jnp.vdot(coefficients, transpose),
        rtol=2.0e-13,
        atol=2.0e-13,
    )

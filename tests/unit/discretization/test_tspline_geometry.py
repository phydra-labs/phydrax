#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation._tspline import certify_tspline_geometry, TMesh2D
from phydrax.discretization import iga
from phydrax.geometry.brep._tspline import TSplineSurfacePatch


def _control_points(mesh: TMesh2D) -> tuple[tuple[Fraction, ...], ...]:
    rows = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        rows.append((u, v, u * v / 7))
    return tuple(rows)


def test_exact_physical_map_certificate_distinguishes_regular_and_degenerate_maps() -> (
    None
):
    mesh = TMesh2D.tensor_product((0, Fraction(1, 2), 1), (0, Fraction(1, 3), 1))
    realization = mesh.realize()
    controls = _control_points(mesh)
    weights = tuple(Fraction(1) for _ in mesh.anchors)

    regular = certify_tspline_geometry(
        realization,
        np.asarray(controls, dtype=object),
        np.asarray(weights, dtype=object),
    )
    assert regular.passed
    assert regular.denominator_positive
    assert regular.regular
    assert regular.embedded
    assert regular.projection_axes == (0, 1)
    assert regular.minimum_monotonicity_margin > 0

    collapsed = tuple((u + v, 2 * (u + v), Fraction(0)) for u, v, _ in controls)
    degenerate = certify_tspline_geometry(
        realization,
        np.asarray(collapsed, dtype=object),
        np.asarray(weights, dtype=object),
    )
    assert not degenerate.passed
    assert not degenerate.regular
    assert not degenerate.embedded
    assert degenerate.failure_reasons


def test_raw_nonuniform_weights_have_fixed_topology_jvp_vjp_hessian_and_gauge_null() -> (
    None
):
    mesh = TMesh2D.tensor_product((0, Fraction(1, 2), 1), (0, Fraction(1, 2), 1))
    realization = mesh.realize()
    controls = jnp.asarray(
        [[float(value) for value in row] for row in _control_points(mesh)],
        dtype=jnp.float64,
    )
    greville_v = jnp.asarray(
        [float(anchor.local_knots.greville_parameter[1]) for anchor in mesh.anchors],
        dtype=jnp.float64,
    )
    weights = 1.0 + greville_v / 3.0
    query = jnp.asarray(((0.19, 0.23), (0.63, 0.74)), dtype=jnp.float64)
    assert not jnp.all(weights == weights[0])

    def values(control_points: jax.Array, raw_weights: jax.Array) -> jax.Array:
        return realization.evaluate_rational(
            query, control_points, raw_weights, maximum_order=0
        ).values

    direction = (
        jnp.arange(controls.size, dtype=controls.dtype).reshape(controls.shape) / 17.0
    )
    jvp_value, jvp_tangent = jax.jvp(
        lambda scale: values(controls + scale * direction, weights),
        (jnp.asarray(0.0),),
        (jnp.asarray(1.0),),
    )
    assert jnp.all(jnp.isfinite(jvp_value))
    assert jnp.all(jnp.isfinite(jvp_tangent))

    value, pullback = jax.vjp(values, controls, weights)
    control_cotangent, weight_cotangent = pullback(jnp.ones_like(value))
    assert jnp.all(jnp.isfinite(control_cotangent))
    assert jnp.all(jnp.isfinite(weight_cotangent))

    weight_direction = jnp.linspace(-0.2, 0.3, weights.size, dtype=weights.dtype)
    second = jax.hessian(
        lambda scale: jnp.sum(values(controls, weights + scale * weight_direction))
    )(jnp.asarray(0.0))
    assert jnp.isfinite(second)

    gauge_tangent = jax.jvp(
        lambda scale: values(controls, scale * weights),
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )[1]
    np.testing.assert_allclose(np.asarray(gauge_tangent), 0.0, rtol=0.0, atol=2e-12)


def test_refined_cad_source_keeps_control_and_weight_derivatives_active() -> None:
    mesh = TMesh2D.tensor_product((0, Fraction(1, 2), 1), (0, Fraction(1, 2), 1))
    exact_points = tuple(
        (
            anchor.local_knots.greville_parameter[0],
            anchor.local_knots.greville_parameter[1],
            Fraction(0),
        )
        for anchor in mesh.anchors
    )
    exact_weights = tuple(
        Fraction(1) + anchor.local_knots.greville_parameter[1] / 5
        for anchor in mesh.anchors
    )
    source_points = np.asarray(
        [[float(value) for value in row] for row in exact_points], dtype=np.float64
    )
    source_weights = np.asarray([float(value) for value in exact_weights])
    patch = TSplineSurfacePatch(
        mesh,
        source_points,
        source_weights,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    plan = iga.TSplineIsogeometricPlan(
        patch,
        iga.TSplineH1FieldSpec("u", weights_from_geometry=True),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    refinement = plan.refine((mesh.cells[0].cell_id,))
    prepared = refinement.target_plan.prepare(numeric_version="source-derivatives")
    route = prepared.prepare_query_route(jnp.asarray(((0.31, 0.42),)))
    controls = jnp.asarray(source_points)
    weights = jnp.asarray(source_weights)

    def evaluate(source_controls: jax.Array, source_weights_: jax.Array) -> jax.Array:
        target_controls, target_weights = prepared.default_runtime.realize_parameter_bank(
            source_controls,
            source_weights_,
            source_authority_id=patch.source_authority_id,
        )
        return jnp.sum(
            prepared.basis_realization.evaluate_rational(
                route.parameters,
                target_controls,
                target_weights,
                cell_ids=route.cell_ids,
                maximum_order=0,
            ).values
        )

    control_gradient, weight_gradient = jax.grad(evaluate, argnums=(0, 1))(
        controls, weights
    )
    assert jnp.linalg.norm(control_gradient) > 0.0
    assert jnp.linalg.norm(weight_gradient) > 0.0
    _, control_tangent = jax.jvp(
        lambda value: evaluate(value, weights),
        (controls,),
        (jnp.ones_like(controls),),
    )
    assert jnp.isfinite(control_tangent)

    gauge_tangent = jax.jvp(
        lambda scale: evaluate(controls, scale * weights),
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )[1]
    np.testing.assert_allclose(np.asarray(gauge_tangent), 0.0, rtol=0.0, atol=2e-11)
    gauge_hessian = jax.hessian(lambda scale: evaluate(controls, scale * weights))(
        jnp.asarray(1.0)
    )
    np.testing.assert_allclose(np.asarray(gauge_hessian), 0.0, rtol=0.0, atol=2e-10)


def test_runtime_recomputes_retained_physical_geometry_certificates() -> None:
    mesh = TMesh2D.tensor_product((0, 1), (0, 1))
    exact_points = _control_points(mesh)
    exact_weights = tuple(Fraction(1) for _ in mesh.anchors)
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray(exact_points, dtype=np.float64),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    geometry = iga.TSplineGeometryState(patch)
    runtime = iga.TSplineRuntimeData(geometry, numeric_version="source")
    forged_exact = replace(
        runtime.exact_geometry_certificate,
        projection_jacobian_bounds=(),
    )
    forged_numeric = replace(
        runtime.geometry_certificate,
        projection_jacobian_bounds=(),
    )

    with pytest.raises(ValueError, match="Retained exact geometry evidence"):
        iga.TSplineRuntimeData(
            geometry,
            numeric_version="forged-exact",
            exact_geometry_certificate=forged_exact,
            geometry_certificate=runtime.geometry_certificate,
        )
    with pytest.raises(ValueError, match="Retained numeric geometry evidence"):
        iga.TSplineRuntimeData(
            geometry,
            numeric_version="forged-numeric",
            exact_geometry_certificate=runtime.exact_geometry_certificate,
            geometry_certificate=forged_numeric,
        )

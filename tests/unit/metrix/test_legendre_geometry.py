#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _full_support(chart, support_id):
    return phx.metrix.ChartSupport(
        chart,
        lambda point: jnp.all(jnp.isfinite(point), axis=-1),
        support_id=support_id,
    )


def _quadratic_geometry():
    diagonal = jnp.asarray([2.0, 4.0])
    primal_chart = phx.metrix.CoordinateChart("quadratic-primal", ("x", "y"))
    dual_chart = phx.metrix.CoordinateChart("quadratic-dual", ("u", "v"))
    hessian = phx.metrix.HessianGeometry(
        lambda point: 0.5 * jnp.sum(diagonal * point**2),
        chart=primal_chart,
    )
    return phx.metrix.LegendreGeometry(
        hessian,
        lambda dual: dual / diagonal,
        primal_support=_full_support(primal_chart, "quadratic-primal-support"),
        dual_support=_full_support(dual_chart, "quadratic-dual-support"),
        geometry_id="quadratic-legendre",
    )


def _negative_entropy_geometry(dimension=3):
    primal_chart = phx.metrix.CoordinateChart(
        "positive-primal",
        tuple(f"x{index}" for index in range(dimension)),
    )
    dual_chart = phx.metrix.CoordinateChart(
        "log-dual",
        tuple(f"y{index}" for index in range(dimension)),
    )
    hessian = phx.metrix.HessianGeometry(
        lambda point: jnp.sum(point * (jnp.log(point) - 1.0)),
        chart=primal_chart,
    )
    return phx.metrix.LegendreGeometry(
        hessian,
        jnp.exp,
        primal_support=phx.metrix.ChartSupport(
            primal_chart,
            lambda point: jnp.all(point > 0.0, axis=-1),
            support_id="positive-orthant",
        ),
        dual_support=_full_support(dual_chart, "finite-log-coordinates"),
        geometry_id="negative-entropy",
    )


def test_quadratic_legendre_geometry_matches_closed_form_with_batches():
    geometry = _quadratic_geometry()
    diagonal = jnp.asarray([2.0, 4.0])
    left = jnp.asarray([[0.4, -0.7], [1.2, 0.3]])
    right = jnp.asarray([[-0.2, 0.5], [0.6, -0.8]])

    dual = geometry.dual_coordinates(left)
    assert jnp.allclose(dual, diagonal * left)
    assert jnp.allclose(geometry.inverse_dual_coordinates(dual), left)

    expected = 0.5 * jnp.sum(diagonal * (left - right) ** 2, axis=-1)
    assert jnp.allclose(geometry.bregman_divergence(left, right), expected)
    assert jnp.allclose(
        geometry.dual_potential(dual),
        0.5 * jnp.sum(dual**2 / diagonal, axis=-1),
    )

    right_dual = geometry.dual_coordinates(right)
    assert jnp.allclose(
        geometry.dual_bregman_divergence(dual, right_dual),
        geometry.bregman_divergence(right, left),
    )
    assert jnp.allclose(geometry.fenchel_young_gap(left, dual), 0.0)


def test_information_operators_propagate_compute_precision_and_evidence():
    geometry = _quadratic_geometry()
    point = jnp.asarray([0.4, -0.7], dtype=jnp.float32)
    vector = jnp.ones((2,), dtype=jnp.float32)
    with jax.enable_x64():
        precision = phx.metrix.GeometryPrecisionPolicy(
            coordinate_dtype=jnp.float32,
            compute_dtype=jnp.float64,
            accumulation_dtype=jnp.float64,
            decision_dtype=jnp.float64,
            output_dtype=jnp.float64,
        )
        operators = (
            geometry.hessian_geometry.information_operator(
                point,
                damping=0.25,
                precision=precision,
            ),
            geometry.information_operator(
                point,
                damping=0.25,
                precision=precision,
            ),
        )
        for operator in operators:
            assert operator.space.dtype == jnp.dtype(jnp.float64)
            assert operator.mv(vector).dtype == jnp.dtype(jnp.float64)
            assert jnp.allclose(operator.mv(vector), jnp.asarray([2.25, 4.25]))
            assert dict(operator.precision_evidence.observed) == {
                "accumulation": "float64",
                "certification": "float64",
                "compute": "float64",
                "output": "float64",
                "storage": "float32",
            }
            assert operator.precision_evidence.domain == "geometry"
            assert operator.precision_evidence.provider == "phydrax-geometry"


def test_legendre_validation_keeps_storage_and_compute_precision_distinct():
    geometry = _quadratic_geometry()
    points = jnp.asarray([[0.4, -0.7], [1.2, 0.3]], dtype=jnp.float32)
    with jax.enable_x64():
        precision = phx.metrix.GeometryPrecisionPolicy(
            coordinate_dtype=jnp.float32,
            compute_dtype=jnp.float64,
            accumulation_dtype=jnp.float64,
            decision_dtype=jnp.float64,
            output_dtype=jnp.float64,
        )
        report = phx.metrix.validate_legendre_geometry(
            geometry,
            points,
            precision=precision,
        )

    expected = {
        "accumulation": "float64",
        "certification": "float64",
        "compute": "float64",
        "output": "float64",
        "storage": "float32",
    }
    assert bool(report.valid)
    assert dict(report.precision_evidence.observed) == expected
    assert dict(report.metric_validation.precision_evidence.observed) == expected


def test_negative_entropy_divergence_and_dual_translation_are_exact_and_jittable():
    geometry = _negative_entropy_geometry()
    left = jnp.asarray([0.3, 1.4, 2.2])
    right = jnp.asarray([0.7, 0.8, 1.1])
    displacement = jnp.asarray([-0.2, 0.4, -0.1])

    expected_divergence = jnp.sum(
        left * jnp.log(left / right) - left + right
    )
    assert jnp.allclose(
        geometry.bregman_divergence(left, right),
        expected_divergence,
    )

    translated = jax.jit(geometry.dual_translate)(left, displacement)
    expected_translation = left * jnp.exp(displacement)
    assert jnp.allclose(translated, expected_translation)

    jacobian = jax.jacrev(lambda point: geometry.dual_translate(point, displacement))(
        left
    )
    assert jnp.allclose(jacobian, jnp.diag(jnp.exp(displacement)))


def test_legendre_geometry_validation_certifies_roundtrips_and_rejects_wrong_inverse():
    geometry = _quadratic_geometry()
    primal = jnp.asarray([[0.3, -0.4], [1.1, 0.8]])
    dual = geometry.dual_coordinates(primal)

    report = phx.metrix.validate_legendre_geometry(
        geometry,
        primal,
        dual_points=dual,
        roundtrip_tolerance=1e-5,
        jacobian_tolerance=1e-5,
        conjugacy_tolerance=1e-5,
    )
    assert bool(report.valid)
    assert bool(report.metric_validation.valid)
    assert report.maximum_primal_roundtrip_error < 1e-6
    assert report.maximum_dual_roundtrip_error < 1e-6
    assert report.maximum_jacobian_inverse_error < 1e-6

    wrong = phx.metrix.LegendreGeometry(
        geometry.hessian_geometry,
        lambda value: value,
        primal_support=geometry.primal_support,
        dual_support=geometry.dual_support,
        geometry_id="wrong-quadratic-inverse",
    )
    invalid = phx.metrix.validate_legendre_geometry(
        wrong,
        primal,
        raise_on_error=False,
    )
    assert not bool(invalid.valid)
    assert invalid.maximum_primal_roundtrip_error > 0.1
    with pytest.raises(ValueError, match="Legendre geometry validation failed"):
        phx.metrix.validate_legendre_geometry(wrong, primal)


def test_legendre_validation_rejects_inverse_outputs_outside_primal_support():
    primal_chart = phx.metrix.CoordinateChart("positive-quadratic-primal", ("x",))
    dual_chart = phx.metrix.CoordinateChart("positive-quadratic-dual", ("u",))
    geometry = phx.metrix.LegendreGeometry(
        phx.metrix.HessianGeometry(
            lambda point: 0.5 * point[0] ** 2,
            chart=primal_chart,
        ),
        lambda dual: dual,
        primal_support=phx.metrix.ChartSupport(
            primal_chart,
            lambda point: point[..., 0] > 0.0,
            support_id="strictly-positive-quadratic",
        ),
        dual_support=_full_support(dual_chart, "finite-quadratic-dual"),
        geometry_id="positive-identity-quadratic",
    )
    negative_dual = jnp.asarray([[-0.5]])
    report = phx.metrix.validate_legendre_geometry(
        geometry,
        jnp.asarray([[1.0]]),
        dual_points=negative_dual,
        raise_on_error=False,
    )

    assert not bool(report.valid)
    assert not bool(report.primal_support_valid)
    assert bool(report.dual_support_valid)
    with pytest.raises(Exception, match="outside Legendre primal support"):
        geometry.inverse_dual_coordinates(negative_dual)


def test_legendre_geometry_rejects_domains_shapes_and_chart_mismatches():
    geometry = _negative_entropy_geometry()
    with pytest.raises(Exception, match="outside Legendre primal support"):
        geometry.dual_coordinates(jnp.asarray([0.4, 0.0, 0.8]))
    with pytest.raises(ValueError, match="equal shapes"):
        geometry.fenchel_young_gap(
            jnp.asarray([0.4, 0.7, 0.8]),
            jnp.asarray([[0.0, 0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="match the primal point shape"):
        geometry.dual_translate(
            jnp.asarray([0.4, 0.7, 0.8]),
            jnp.asarray([[0.0, 0.0, 0.0]]),
        )

    incompatible_chart = phx.metrix.CoordinateChart(
        "other-primal",
        ("x0", "x1", "x2"),
    )
    with pytest.raises(ValueError, match="must use the Hessian geometry chart"):
        phx.metrix.LegendreGeometry(
            geometry.hessian_geometry,
            jnp.exp,
            primal_support=_full_support(incompatible_chart, "wrong-primal"),
            dual_support=geometry.dual_support,
            geometry_id="invalid-legendre",
        )


def test_legendre_validation_reports_nonconvex_metric_without_global_claims():
    primal_chart = phx.metrix.CoordinateChart("nonconvex-primal", ("x",))
    dual_chart = phx.metrix.CoordinateChart("nonconvex-dual", ("u",))
    geometry = phx.metrix.LegendreGeometry(
        phx.metrix.HessianGeometry(
            lambda point: -(point[0] ** 2),
            chart=primal_chart,
        ),
        lambda dual: -0.5 * dual,
        primal_support=_full_support(primal_chart, "nonconvex-primal-support"),
        dual_support=_full_support(dual_chart, "nonconvex-dual-support"),
        geometry_id="nonconvex-candidate",
    )
    report = phx.metrix.validate_legendre_geometry(
        geometry,
        jnp.asarray([[0.2], [0.7]]),
        raise_on_error=False,
    )
    assert not bool(report.valid)
    assert report.metric_validation.minimum_eigenvalue < 0.0

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _chart(name):
    return phx.metrix.CoordinateChart(name, ("t", "r", "theta", "phi"))


@pytest.mark.parametrize(
    ("convention", "timelike_sign"),
    (("mostly_plus", -1.0), ("mostly_minus", 1.0)),
)
def test_minkowski_reference_is_flat_in_both_sign_conventions(
    convention,
    timelike_sign,
):
    metric = phx.metrix.minkowski_metric(
        _chart("minkowski"),
        convention=convention,
    )
    point = jnp.array([0.2, 0.4, 1.1, -0.3])
    time_vector = jnp.array([1.0, 0.0, 0.0, 0.0])

    assert jnp.allclose(
        metric.bilinear(time_vector, time_vector, point),
        timelike_sign,
    )
    assert jnp.allclose(phx.metrix.riemann_tensor(metric, point), 0.0)
    assert jnp.allclose(phx.metrix.ricci_tensor(metric, point), 0.0)
    assert jnp.allclose(phx.metrix.scalar_curvature(metric, point), 0.0)
    assert jnp.allclose(phx.metrix.einstein_tensor(metric, point), 0.0)
    assert jnp.allclose(phx.metrix.adm_extrinsic_curvature(metric, point), 0.0)
    assert phx.metrix.adm_constraint_residuals(
        metric,
        point,
        einstein_coupling=0.0,
    ).maximum_absolute == 0.0


@pytest.mark.parametrize(
    ("convention", "scalar_sign"),
    (("mostly_plus", 1.0), ("mostly_minus", -1.0)),
)
def test_curved_flrw_reference_matches_friedmann_tensors_and_bianchi_identity(
    convention,
    scalar_sign,
):
    chart = _chart("flrw")

    def scale_factor(time):
        return 1.0 + 0.2 * time + 0.1 * time**2

    spatial_curvature = 1
    metric = phx.metrix.flrw_metric(
        scale_factor,
        chart=chart,
        spatial_curvature=spatial_curvature,
        convention=convention,
    )
    point = jnp.array([0.4, 0.3, 1.1, 0.2])
    time, radius, polar, _ = point
    scale = scale_factor(time)
    scale_derivative = 0.2 + 0.2 * time
    scale_second_derivative = 0.2
    hubble = scale_derivative / scale
    expected_scalar = 6.0 * (
        scale_second_derivative / scale
        + hubble**2
        + spatial_curvature / scale**2
    )
    spatial_pressure_factor = -(
        2.0 * scale * scale_second_derivative
        + scale_derivative**2
        + spatial_curvature
    )
    expected_einstein = jnp.diag(
        jnp.array(
            [
                3.0 * (hubble**2 + spatial_curvature / scale**2),
                spatial_pressure_factor / (1.0 - spatial_curvature * radius**2),
                spatial_pressure_factor * radius**2,
                spatial_pressure_factor * radius**2 * jnp.sin(polar) ** 2,
            ]
        )
    )

    assert jnp.allclose(
        phx.metrix.scalar_curvature(metric, point),
        scalar_sign * expected_scalar,
        atol=1e-10,
    )
    assert jnp.allclose(
        phx.metrix.einstein_tensor(metric, point),
        expected_einstein,
        atol=1e-10,
    )

    contravariant = phx.metrix.TensorType(
        ("contravariant", "contravariant")
    )

    def raised_einstein(coordinates):
        inverse = metric.inverse(coordinates)
        return inverse @ phx.metrix.einstein_tensor(metric, coordinates) @ inverse

    derivative = phx.metrix.covariant_derivative(
        raised_einstein,
        metric,
        contravariant,
        point,
    )
    assert jnp.allclose(jnp.einsum("ijj->i", derivative), 0.0, atol=5e-10)


def test_schwarzschild_reference_is_vacuum_with_exact_kretschmann_scalar():
    chart = _chart("schwarzschild")
    mass = 1.2
    metric = phx.metrix.schwarzschild_metric(mass, chart=chart)
    point = jnp.array([0.0, 4.0, 1.1, 0.2])
    radius = point[1]
    matrix = metric(point)
    inverse = metric.inverse(point)
    riemann = phx.metrix.riemann_tensor(metric, point)
    lowered_riemann = jnp.einsum("al,lkij->akij", matrix, riemann)
    kretschmann = jnp.einsum(
        "abcd,ae,bf,cg,dh,efgh->",
        lowered_riemann,
        inverse,
        inverse,
        inverse,
        inverse,
        lowered_riemann,
    )
    decomposition = phx.metrix.decompose_adm_metric(metric, point)
    factor = 1.0 - 2.0 * mass / radius

    assert jnp.allclose(phx.metrix.ricci_tensor(metric, point), 0.0, atol=1e-10)
    assert jnp.allclose(phx.metrix.scalar_curvature(metric, point), 0.0, atol=1e-10)
    assert jnp.allclose(phx.metrix.einstein_tensor(metric, point), 0.0, atol=1e-10)
    assert jnp.allclose(kretschmann, 48.0 * mass**2 / radius**6, rtol=1e-10)
    assert jnp.allclose(decomposition.lapse, jnp.sqrt(factor))
    assert jnp.allclose(decomposition.shift, 0.0)
    assert jnp.allclose(
        jnp.diag(decomposition.spatial_metric),
        jnp.array(
            [
                1.0 / factor,
                radius**2,
                radius**2 * jnp.sin(point[2]) ** 2,
            ]
        ),
    )
    assert jnp.allclose(phx.metrix.adm_extrinsic_curvature(metric, point), 0.0)
    assert phx.metrix.adm_constraint_residuals(
        metric,
        point,
        einstein_coupling=0.0,
    ).maximum_absolute < 1e-10

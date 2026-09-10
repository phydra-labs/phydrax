#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _resource_limits():
    return phx.interchange.ResourceLimits(10_000, 2, 100, 100, 2)


def _astrodynamics_context():
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsContext(
        astro.AstrodynamicsScaleContract.si(),
        astro.ReferenceEpoch(astro.TimeInstant(astro.JulianDate(2451545.0), "TT")),
        astro.FrameDefinition("earth", "icrf", pseudo_inertial=True),
    )


def _astrodynamics_provenance(context):
    astro = phx.applications.astrodynamics
    return astro.AstrodynamicsDataProvenance(
        producer="test",
        producer_version="1",
        source_id="closed-form-degree-two-order-one",
        checksum="sha256:test",
        license_id="test-data",
        frame_id=context.frame.frame_id,
        epoch_id=context.epoch.epoch_id,
        scale_id=context.scale.scale_id,
        differentiability="constant",
    )


def _degree_two_order_one_value_gradient(position, prefactor, cosine, sine):
    position = np.asarray(position, dtype=float)
    radius = np.linalg.norm(position)
    x, y, z = position
    linear = cosine * x + sine * y
    product = z * linear
    product_gradient = np.asarray([cosine * z, sine * z, linear])
    value = prefactor * product / radius**5
    gradient = prefactor * (
        product_gradient / radius**5 - 5.0 * product * position / radius**7
    )
    return value, gradient


def test_astrodynamics_uses_unnormalized_condon_shortley_harmonics():
    astro = phx.applications.astrodynamics
    context = _astrodynamics_context()
    cosine_value, sine_value = 0.35, -0.22
    cosine = jnp.zeros((3, 3)).at[2, 1].set(cosine_value)
    sine = jnp.zeros((3, 3)).at[2, 1].set(sine_value)
    mu, reference_radius = 4.6, 1.4
    field = astro.SphericalHarmonicGravityField(
        cosine,
        sine,
        mu,
        reference_radius,
        context,
        _astrodynamics_provenance(context),
        maximum_degree=2,
        maximum_order=1,
    )
    gravity = astro.SphericalHarmonicGravity(field)
    position = jnp.asarray([2.2, -1.3, 1.7])

    result = gravity.evaluate(0.0, jnp.concatenate((position, jnp.zeros(3))))
    expected_potential, potential_gradient = _degree_two_order_one_value_gradient(
        position,
        3.0 * mu * reference_radius**2,
        cosine_value,
        sine_value,
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.potential, expected_potential, rtol=2e-12)
    np.testing.assert_allclose(
        result.acceleration, -potential_gradient, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize(
    ("normalization", "normalization_factor"),
    (
        ("unnormalized", 1.0),
        ("schmidt", np.sqrt(1.0 / 3.0)),
        ("fully_normalized", np.sqrt(5.0 / 3.0)),
    ),
)
def test_geophysical_gravity_honors_declared_normalization(
    tmp_path, normalization, normalization_factor
):
    cosine_value, sine_value = 0.35, -0.22
    gravitational_constant, reference_radius = 7.0, 1.4
    coefficient_file = tmp_path / f"degree_2_order_1_{normalization}.gfc"
    coefficient_file.write_text(
        "modelname DEGREE_2_ORDER_1\n"
        f"earth_gravity_constant {gravitational_constant}\n"
        f"radius {reference_radius}\n"
        "max_degree 2\n"
        f"norm {normalization}\n"
        "tide_system tide_free\n"
        "end_of_head\n"
        f"gfc 2 1 {cosine_value} {sine_value}\n"
    )
    model = phx.interchange.read_icgem_gfc(
        coefficient_file.name,
        trusted_root=tmp_path,
        limits=_resource_limits(),
    )
    plan = phx.applications.geophysics.SphericalHarmonicGravityPlan(model)
    position = jnp.asarray([2.2, -1.3, 1.7])

    result = plan.evaluate(position[None, :])
    expected_potential, expected_acceleration = _degree_two_order_one_value_gradient(
        position,
        3.0 * gravitational_constant * reference_radius**2 * normalization_factor,
        cosine_value,
        sine_value,
    )

    assert bool(result.finite)
    np.testing.assert_allclose(result.potential_m2_s2, [expected_potential], rtol=2e-12)
    np.testing.assert_allclose(
        result.acceleration_m_s2[0],
        expected_acceleration,
        rtol=2e-12,
        atol=2e-12,
    )


def test_geomagnetic_schmidt_harmonics_apply_secular_epoch_offsets(tmp_path):
    epoch, reference_radius = 2025.0, 1.4
    g, h, secular_g, secular_h = 120.0, -45.0, -3.5, 4.0
    coefficient_file = tmp_path / "degree_2_order_1.cof"
    coefficient_file.write_text(f"2 1 {g} {h} {secular_g} {secular_h}\n")
    model = phx.interchange.read_geomagnetic_coefficients(
        coefficient_file.name,
        trusted_root=tmp_path,
        limits=_resource_limits(),
        model_name="synthetic-epoch-model",
        epoch_decimal_year=epoch,
        reference_radius_m=reference_radius,
    )
    plan = phx.applications.geophysics.SphericalHarmonicMagneticPlan(model)
    position = np.asarray([2.2, -1.3, 1.7])
    years = np.asarray([epoch, epoch + 2.25])
    positions = np.broadcast_to(position, (years.size, 3))
    reference_direction = np.asarray([2.0, -1.0, 2.0]) / 3.0

    result = plan.evaluate(
        positions,
        years,
        reference_direction=reference_direction,
    )
    expected_fields = []
    for year in years:
        elapsed = year - epoch
        effective_g = g + elapsed * secular_g
        effective_h = h + elapsed * secular_h
        _, potential_gradient = _degree_two_order_one_value_gradient(
            position,
            np.sqrt(3.0) * reference_radius**4,
            effective_g,
            effective_h,
        )
        expected_fields.append(-potential_gradient)
    expected_fields = np.asarray(expected_fields)

    assert bool(result.finite)
    np.testing.assert_allclose(result.field_nT, expected_fields, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        result.total_field_anomaly_nT,
        expected_fields @ reference_direction,
        rtol=2e-12,
        atol=2e-12,
    )

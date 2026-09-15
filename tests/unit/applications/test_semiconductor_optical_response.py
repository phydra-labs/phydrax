#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.semiconductor._optical_response import (
    ActiveRegionOpticalProjection,
    evaluate_semiconductor_optical_response,
    LinearizedCarrierOpticalResponsePlan,
    SemiconductorOpticalResponseStatus,
    TabulatedCarrierOpticalResponsePlan,
)


def _linear(**overrides):
    options = {
        "reference_carrier_pair_density": 2.0e24,
        "reference_temperature": 300.0,
        "differential_refractive_index": -2.0e-27,
        "thermo_optic_coefficient": 2.5e-4,
        "background_internal_loss": 250.0,
        "carrier_internal_loss_cross_section": 1.0e-22,
        "density_range": (0.5e24, 4.0e24),
        "temperature_range": (280.0, 340.0),
        "angular_frequency_range": (0.9e15, 1.1e15),
        "active_volume": 8.0e-17,
        "confinement_factor": 0.3,
        "provenance": "synthetic analytic semiconductor optical fixture",
        "model_id": "linear-optical-fixture",
    }
    options.update(overrides)
    return LinearizedCarrierOpticalResponsePlan(
        1.5e24,
        4.0e-21,
        1.0e15,
        **options,
    )


def test_linearized_response_preserves_transparency_and_power_field_factor_two():
    plan = _linear()
    transparent = evaluate_semiconductor_optical_response(plan, 1.5e24, 300.0, 1.0e15)
    populated = evaluate_semiconductor_optical_response(plan, 2.0e24, 300.0, 1.0e15)

    np.testing.assert_allclose(transparent.modal_power_gain, 0.0, atol=0.0)
    np.testing.assert_allclose(populated.modal_power_gain, 600.0, rtol=1.0e-14)
    np.testing.assert_allclose(populated.field_gain, 0.5 * populated.modal_power_gain)
    np.testing.assert_allclose(populated.refractive_index_change, 0.0, atol=0.0)
    assert populated.internal_loss >= 0.0
    assert populated.successful
    assert populated.evidence.model_provenance == plan.provenance
    assert populated.evidence.model_id == plan.model_id


def test_active_region_projection_is_volume_and_mode_explicit():
    projection = ActiveRegionOpticalProjection(
        jnp.asarray((1.0, 2.0, 1.0)) * 1.0e-18,
        jnp.asarray((0.0, 1.0, 0.5)),
        jnp.asarray((1.0, 3.0, 2.0)),
        support_id="three-control-volumes",
        mode_id="fundamental-mode",
        provenance="synthetic mode-volume projection",
    )
    density = jnp.asarray((1.0, 2.0, 4.0)) * 1.0e24
    temperature = jnp.asarray((290.0, 300.0, 320.0))
    result = evaluate_semiconductor_optical_response(
        _linear(), density, temperature, 1.0e15, projection=projection
    )
    expected_weights = np.asarray((0.0, 6.0, 1.0)) / 7.0

    np.testing.assert_allclose(projection.projection_weights, expected_weights)
    np.testing.assert_allclose(
        result.carrier_pair_density, np.dot(expected_weights, np.asarray(density))
    )
    np.testing.assert_allclose(
        result.lattice_temperature, np.dot(expected_weights, np.asarray(temperature))
    )
    np.testing.assert_allclose(result.active_volume, 2.5e-18)
    np.testing.assert_allclose(result.confinement_factor, 7.0 / 9.0)
    assert result.evidence.projection_id == projection.projection_id
    assert result.evidence.projection_provenance == projection.provenance
    assert result.successful


def test_tabulated_response_interpolates_all_three_physical_axes_without_extrapolation():
    density = np.asarray((1.0, 3.0)) * 1.0e24
    temperature = np.asarray((290.0, 330.0))
    frequency = np.asarray((0.9, 1.1)) * 1.0e15
    d, t, w = np.meshgrid(density, temperature, frequency, indexing="ij")
    gain = 1.0e-21 * (d - 2.0e24) + 2.0 * (t - 300.0) + 1.0e-12 * (w - 1.0e15)
    index = -1.0e-27 * (d - 2.0e24) + 1.0e-4 * (t - 300.0)
    loss = 100.0 + 1.0e-22 * d
    plan = TabulatedCarrierOpticalResponsePlan(
        density,
        temperature,
        frequency,
        gain,
        index,
        loss,
        active_volume=2.0e-16,
        confinement_factor=0.4,
        provenance="synthetic affine trilinear table",
        model_id="tabulated-optical-fixture",
    )
    result = evaluate_semiconductor_optical_response(plan, 2.0e24, 310.0, 1.0e15)

    np.testing.assert_allclose(result.material_power_gain, 20.0, atol=1.0e-12)
    np.testing.assert_allclose(result.modal_power_gain, 8.0, atol=1.0e-12)
    np.testing.assert_allclose(result.refractive_index_change, 1.0e-3, rtol=1.0e-12)
    np.testing.assert_allclose(result.internal_loss, 300.0, rtol=1.0e-12)
    assert result.successful

    rejected = evaluate_semiconductor_optical_response(plan, 4.0e24, 310.0, 1.0e15)
    assert not rejected.successful
    assert rejected.status & int(
        SemiconductorOpticalResponseStatus.CARRIER_OUTSIDE_SUPPORT
    )
    assert np.isnan(rejected.modal_power_gain)


def test_support_failures_remain_explicit_and_provenance_is_retained():
    plan = _linear()
    rejected = evaluate_semiconductor_optical_response(
        plan,
        jnp.asarray((2.0e24, 2.0e24)),
        jnp.asarray((300.0, 350.0)),
        1.2e15,
    )

    assert np.all(
        rejected.status
        & int(SemiconductorOpticalResponseStatus.FREQUENCY_OUTSIDE_SUPPORT)
    )
    assert rejected.status[1] & int(
        SemiconductorOpticalResponseStatus.TEMPERATURE_OUTSIDE_SUPPORT
    )
    assert np.isnan(rejected.field_gain).all()
    assert rejected.evidence.model_provenance == plan.provenance

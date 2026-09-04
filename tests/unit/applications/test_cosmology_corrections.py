import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _linear_power():
    scale = cosmology.CosmologyScaleContract(
        cosmology.CODE_COSMOLOGY_SCALE.length_unit,
        cosmology.CODE_COSMOLOGY_SCALE.mass_unit,
        cosmology.CODE_COSMOLOGY_SCALE.time_unit,
    )
    background = cosmology.FLRWBackground(1.0, 0.3, scale=scale)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="test-linear-power",
        numerical_policy_id="test-grid",
        physics_policy_id="linear-total-matter",
        scale_id=scale.scale_id,
        source_kind="native",
        differentiation="native-parameter",
    )
    power = cosmology.MatterPowerTable(
        [0.5, 1.0],
        [1.0, 2.0, 4.0],
        [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]],
        cosmology.MatterPowerDescriptor("total_matter", "total_matter"),
        scale,
        provenance,
        background.realization,
    )
    return background, power


def test_multiplicative_correction_preserves_grid_and_chains_provenance():
    _, power = _linear_power()
    card = cosmology.CorrectionModelCard(
        name="test-boost",
        model_version="current",
        source_reference="independent-test",
        calibration_id="test-calibration",
        denominator_stage="linear",
        output_stage="nonlinear",
        scale_factor_domain=(0.5, 1.0),
        wavenumber_domain=(1.0, 4.0),
        expected_error="exact fixture",
        license_id="internal-test",
    )
    plan = cosmology.MultiplicativeMatterPowerCorrectionPlan(
        power.scale_factors,
        power.wavenumbers,
        2.0 * jnp.ones_like(power.power_values),
        card,
        differentiation="native-parameter",
    )
    identity = plan.apply(power, strength=0.0)
    np.testing.assert_allclose(identity.power.power_values, power.power_values)
    result = plan.apply(power, strength=0.5)
    np.testing.assert_allclose(result.power.power_values, 1.5 * power.power_values)
    assert result.power.descriptor.stage == "nonlinear"
    assert result.power.provenance.parent_product_ids == (power.provenance.provenance_id,)
    assert bool(result.successful)
    derivative = jax.grad(
        lambda strength: jnp.sum(plan.apply(power, strength=strength).power.power_values)
    )(jnp.asarray(0.5))
    np.testing.assert_allclose(derivative, jnp.sum(power.power_values))


def test_correction_rejects_wrong_denominator_and_domain():
    _, power = _linear_power()
    card = cosmology.CorrectionModelCard(
        name="bad-domain",
        model_version="current",
        source_reference="test",
        calibration_id="test",
        denominator_stage="nonlinear",
        output_stage="nonlinear",
        scale_factor_domain=(0.6, 1.0),
        wavenumber_domain=(1.0, 4.0),
        expected_error="not applicable",
        license_id="internal-test",
    )
    plan = cosmology.MultiplicativeMatterPowerCorrectionPlan(
        power.scale_factors,
        power.wavenumbers,
        jnp.ones_like(power.power_values),
        card,
    )
    with pytest.raises(ValueError, match="denominator stage"):
        plan.apply(power)

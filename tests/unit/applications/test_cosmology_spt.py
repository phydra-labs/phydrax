import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_one_loop_spt_returns_bounded_nonlinear_power():
    background = cosmology.FLRWBackground(1.0, 1.0)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="spt-test",
        numerical_policy_id="spt-input-grid",
        physics_policy_id="linear-total-matter",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiation=cosmology.DifferentiationContract.native(),
    )
    k = jnp.geomspace(1.0e-4, 4.0, 512)
    linear_values = 1.0e-4 * k / (1.0 + k**4)
    power = cosmology.MatterPowerTable(
        [0.5, 1.0],
        k,
        jnp.stack((0.25 * linear_values, linear_values)),
        cosmology.MatterPowerDescriptor("total_matter", "total_matter"),
        background.scale,
        provenance,
        background.realization,
    )
    output_k = jnp.geomspace(0.5, 1.0, 8)
    plan = cosmology.OneLoopEdSSPTPlan(
        output_k,
        radial_order=24,
        angular_order=24,
        radial_ratio_domain=(0.5, 2.0),
        maximum_relative_correction=0.9,
    )
    result = plan.evaluate(background, power, 1.0)
    assert bool(result.successful)
    assert result.power.descriptor.stage == "nonlinear"
    assert result.power.provenance.parent_product_ids == (power.provenance.provenance_id,)
    assert jnp.all(jnp.isfinite(result.evidence.p22))
    assert jnp.all(jnp.isfinite(result.evidence.p13))
    assert jnp.all(result.evidence.relative_correction <= 0.9)
    np.testing.assert_allclose(result.power.wavenumbers, output_k)

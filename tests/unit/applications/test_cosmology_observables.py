import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _context():
    scale = cosmology.CosmologyScaleContract(
        cosmology.CODE_COSMOLOGY_SCALE.length_unit,
        cosmology.CODE_COSMOLOGY_SCALE.mass_unit,
        cosmology.CODE_COSMOLOGY_SCALE.time_unit,
    )
    background = cosmology.FLRWBackground(1.0, 1.0, scale=scale)
    growth = cosmology.FLRWGrowthPlan(jnp.geomspace(1.0e-2, 1.0, 32)).solve(background)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="survey-test-power",
        numerical_policy_id="survey-test-grid",
        physics_policy_id="linear-total-matter",
        scale_id=scale.scale_id,
        source_kind="native",
        differentiation="native-parameter",
    )
    k = jnp.geomspace(0.05, 2000.0, 512)
    power = cosmology.MatterPowerTable(
        [0.5, 1.0],
        k,
        jnp.ones((2, k.size)),
        cosmology.MatterPowerDescriptor("total_matter", "total_matter"),
        scale,
        provenance,
        background.realization,
    )
    return background, growth, power


def test_limber_density_bias_scaling_and_lensing_finiteness():
    background, _, power = _context()
    distance = cosmology.FLRWDistancePlan(light_speed=1.0, order=64)
    grid = cosmology.RadialGrid(jnp.linspace(0.1, 0.9, 48))
    distribution = cosmology.RedshiftDistribution(
        grid, jnp.exp(-(((grid.redshifts - 0.5) / 0.15) ** 2)), "bin-0"
    )
    first = cosmology.LinearDensityTracer(distribution, 1.0)
    second = cosmology.LinearDensityTracer(distribution, 2.0)
    plan = cosmology.LimberAngularPowerPlan([10, 20, 50], 2)
    prediction = plan.predict(background, distance, power, (first, second))
    assert bool(prediction.successful)
    np.testing.assert_allclose(prediction.values[2], 4.0 * prediction.values[0])
    np.testing.assert_allclose(prediction.values[1], 2.0 * prediction.values[0])

    lensing = cosmology.LensingConvergenceTracer(distribution)
    lensing_plan = cosmology.LimberAngularPowerPlan([20, 50], 1)
    lensing_prediction = lensing_plan.predict(background, distance, power, (lensing,))
    assert bool(lensing_prediction.successful)
    assert jnp.all(lensing_prediction.values >= 0.0)


def test_linear_kaiser_multipoles_and_ap_identity():
    background, growth, power = _context()
    distance = cosmology.FLRWDistancePlan(light_speed=1.0, order=64)
    plan = cosmology.LinearRSDMultipolePlan(jnp.asarray([0.1, 0.2, 0.5]), mu_order=64)
    result = plan.predict(
        background,
        background,
        distance,
        growth,
        power,
        2.0,
        0.5,
    )
    np.testing.assert_allclose(result.alpha_perpendicular, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.alpha_parallel, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.monopole, 4.0 + 4.0 / 3.0 + 1.0 / 5.0, rtol=1e-10)
    np.testing.assert_allclose(result.quadrupole, 8.0 / 3.0 + 4.0 / 7.0, rtol=1e-10)
    np.testing.assert_allclose(result.hexadecapole, 8.0 / 35.0, rtol=1e-10)
    assert bool(result.successful)

"""Advanced background, products, corrections, halos, surveys, and CMB contracts."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmo = phx.applications.cosmology
    scale = cosmo.CosmologyScaleContract("Mpc", "mass", "Gyr")
    background = cosmo.FLRWBackground(
        1.0,
        0.3,
        dark_energy_w0=-0.9,
        dark_energy_wa=0.1,
        scale=scale,
    )
    curved = cosmo.FLRWBackground(
        1.0,
        0.3,
        curvature_density=0.02,
        dark_energy_w0=-0.9,
        dark_energy_wa=0.1,
        scale=scale,
    )
    distance_plan = cosmo.FLRWDistancePlan(light_speed=1.0, order=64)
    curved_distance = distance_plan.evaluate(curved, jnp.asarray([0.5, 1.0]))
    growth = cosmo.FLRWGrowthPlan(jnp.geomspace(1.0e-2, 1.0, 48)).solve(background)
    provenance = cosmo.CosmologyProductProvenance(
        producer="advanced-example",
        producer_version="native",
        model_form_id=background.model_form_id,
        request_id="advanced-example-power",
        numerical_policy_id="advanced-example-grid",
        physics_policy_id="linear-total-matter",
        scale_id=scale.scale_id,
        source_kind="native",
        differentiability="native-parameter",
    )
    k = jnp.geomspace(0.05, 2000.0, 512)
    linear = cosmo.MatterPowerTable(
        [0.5, 1.0],
        k,
        jnp.stack((0.25 / (1.0 + k**2), 1.0 / (1.0 + k**2))),
        cosmo.MatterPowerDescriptor("total_matter", "total_matter"),
        scale,
        provenance,
        background.realization,
    )
    card = cosmo.CorrectionModelCard(
        name="example-smooth-boost",
        model_version="native-example",
        source_reference="demonstration-only",
        calibration_id="none",
        denominator_stage="linear",
        output_stage="nonlinear",
        scale_factor_domain=(0.5, 1.0),
        wavenumber_domain=(0.05, 2000.0),
        expected_error="not a calibrated physical model",
        license_id="internal-example",
    )
    boost = 1.0 + 0.1 * (k[None, :] / (1.0 + k[None, :]))
    boost = jnp.broadcast_to(boost, linear.power_values.shape)
    nonlinear = cosmo.MultiplicativeMatterPowerCorrectionPlan(
        linear.scale_factors,
        linear.wavenumbers,
        boost,
        card,
        differentiability="native-parameter",
    ).apply(linear)
    variance = cosmo.LinearVariancePlan(1.0).sigma(
        background, linear, jnp.asarray([0.01, 0.1, 1.0]), 1.0
    )
    radial = cosmo.RadialGrid(jnp.linspace(0.1, 0.9, 48))
    distribution = cosmo.RedshiftDistribution(
        radial,
        jnp.exp(-(((radial.redshifts - 0.5) / 0.15) ** 2)),
        "example-bin",
    )
    tracer = cosmo.LinearDensityTracer(distribution, 1.5)
    angular = cosmo.LimberAngularPowerPlan([20, 50, 100], 1).predict(
        background,
        distance_plan,
        nonlinear.power,
        (tracer,),
    )
    primordial = cosmo.PrimordialPowerLaw(2.1e-9, 0.965, 0.05)
    force_plan = cosmo.PeriodicImageForcePlan(
        (1.0, 1.0, 1.0), 1.0, softening=0.02, image_shells=1
    )
    positions = jnp.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    masses = jnp.ones((2,))
    reference_force = force_plan.acceleration(positions, masses)
    force_evidence = force_plan.qualify(positions, masses, reference_force)
    print("curved_radial_distance", curved_distance.radial_comoving_distance)
    print("curved_transverse_distance", curved_distance.transverse_comoving_distance)
    print("growth_today", growth.first_order_growth[-1])
    print("nonlinear_successful", bool(nonlinear.successful))
    print("halo_sigma", variance)
    print("angular_prediction", angular.values)
    print("primordial_at_pivot", primordial.scalar_power(0.05))
    print("force_qualification", bool(force_evidence.successful))


if __name__ == "__main__":
    main()

import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _context():
    background = cosmology.FLRWBackground(1.0, 1.0)
    growth = cosmology.FLRWGrowthPlan(jnp.geomspace(1.0e-2, 1.0, 32)).solve(background)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="halo-model-test",
        numerical_policy_id="halo-model-grid",
        physics_policy_id="linear-total-matter",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiation=cosmology.DifferentiationContract.native(),
    )
    k = jnp.geomspace(1.0e-2, 100.0, 512)
    values = 1.0e-2 * k / (1.0 + k**3)
    power = cosmology.MatterPowerTable(
        [0.5, 1.0],
        k,
        jnp.stack((0.25 * values, values)),
        cosmology.MatterPowerDescriptor("total_matter", "total_matter"),
        background.scale,
        provenance,
        background.realization,
    )
    return background, growth, power


def test_smooth_collapse_and_calibrated_halo_triplet():
    background, growth, power = _context()
    collapse = cosmology.SmoothComponentSphericalCollapsePlan(
        steps=256, bisection_iterations=32
    ).solve(background, growth, 1.0)
    assert bool(collapse.successful)
    np.testing.assert_allclose(collapse.linear_threshold, 1.686, rtol=5e-2)

    variance = cosmology.LinearVariancePlan(1.0)
    triplet_plan = cosmology.TinkerDuffy200mPlan(
        variance,
        mass_domain=(1.0e-4, 1.0),
        pivot_mass=0.1,
    )
    masses = jnp.geomspace(1.0e-4, 1.0, 32)
    triplet = triplet_plan.evaluate(background, power, masses, 1.0)
    assert bool(triplet.successful)
    assert jnp.all(triplet.mass_function_dndlnm >= 0.0)
    assert jnp.all(triplet.concentration > 0.0)


def test_halo_model_catalog_and_zheng_expectation():
    background, _, power = _context()
    variance = cosmology.LinearVariancePlan(1.0)
    triplet = cosmology.TinkerDuffy200mPlan(
        variance,
        mass_domain=(1.0e-4, 1.0),
        pivot_mass=0.1,
    )
    definition = cosmology.SphericalOverdensityMassDefinition(200.0, "mean_matter")
    profile = cosmology.NFWProfile(definition, quadrature_order=32)
    model = cosmology.MatterHaloModel200mPlan(triplet, profile)
    result = model.evaluate(
        background,
        power,
        jnp.geomspace(1.0e-4, 1.0, 32),
        jnp.geomspace(0.05, 2.0, 12),
        1.0,
    )
    assert bool(result.successful)
    assert jnp.all(result.total >= 0.0)

    artifact = cosmology.ScientificArtifactEnvelope(
        artifact_kind="halo-catalog",
        content_digest="fixture",
        producer="test",
        producer_version="current",
        build_id="build",
        license_id="internal",
        resource_id="resource",
        status="complete",
    )
    catalog = cosmology.HaloCatalog(
        [1, 2],
        [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [0.1, 0.2],
        [True, True],
        definition,
        1.0,
        (1.0, 1.0, 1.0),
        artifact,
    )
    assert catalog.catalog_id
    hod = cosmology.Zheng07OccupationExpectation200m(-1.0, 0.2, 0.01, 0.1, 1.0).evaluate(
        catalog.masses
    )
    assert bool(hod.successful)
    assert jnp.all(hod.total_mean >= hod.central_probability)

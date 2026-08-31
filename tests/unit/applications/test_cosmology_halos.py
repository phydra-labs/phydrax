import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _power(background):
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="halo-test-power",
        numerical_policy_id="halo-test-grid",
        physics_policy_id="linear-total-matter",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiability="native-parameter",
    )
    k = jnp.geomspace(1.0e-2, 20.0, 256)
    values = jnp.stack((k**-1.0, 2.0 * k**-1.0))
    return cosmology.MatterPowerTable(
        [0.5, 1.0],
        k,
        values,
        cosmology.MatterPowerDescriptor("total_matter", "total_matter"),
        background.scale,
        provenance,
        background.realization,
    )


def test_spherical_overdensity_mass_radius_inverse_and_eds_constants():
    background = cosmology.FLRWBackground(1.0, 1.0)
    definition = cosmology.SphericalOverdensityMassDefinition(200.0, "critical")
    mass = jnp.asarray([1.0, 10.0, 100.0])
    radius = definition.radius(background, mass, 1.0, 1.0)
    recovered = definition.mass(background, radius, 1.0, 1.0)
    np.testing.assert_allclose(recovered, mass, rtol=1e-12)
    collapse = cosmology.SphericalCollapseEdS()
    np.testing.assert_allclose(collapse.linear_threshold, 1.68647019984, rtol=1e-11)
    np.testing.assert_allclose(collapse.virial_overdensity, 18.0 * np.pi**2)


def test_linear_variance_and_nfw_normalization_are_finite():
    background = cosmology.FLRWBackground(1.0, 1.0)
    power = _power(background)
    variance = cosmology.LinearVariancePlan(1.0).sigma(
        background, power, jnp.asarray([0.01, 0.1, 1.0]), 1.0
    )
    assert jnp.all(jnp.isfinite(variance))
    assert jnp.all(variance > 0.0)

    definition = cosmology.SphericalOverdensityMassDefinition(200.0, "critical")
    profile = cosmology.NFWProfile(definition, quadrature_order=96)
    np.testing.assert_allclose(profile.enclosed_mass_fraction(1.0, 5.0), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        profile.fourier(jnp.asarray(0.0), 1.0, 5.0), 1.0, rtol=1e-10
    )
    assert profile.fourier(jnp.asarray(10.0), 1.0, 5.0) < 1.0

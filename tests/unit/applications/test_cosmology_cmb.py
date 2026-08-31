import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def _table():
    background = cosmology.FLRWBackground(1.0, 0.3)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test-cmb",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="test-cmb-request",
        numerical_policy_id="test-cmb-grid",
        physics_policy_id="raw-unlensed-cmb",
        scale_id=background.scale.scale_id,
        source_kind="external",
        differentiability="constant",
    )
    spectra = jnp.zeros((1, 2, 4, 4))
    spectra = spectra.at[0, :, 0, 0].set(jnp.asarray([2.0, 3.0]))
    spectra = spectra.at[0, :, 1, 1].set(jnp.asarray([1.0, 1.5]))
    spectra = spectra.at[0, :, 2, 2].set(jnp.asarray([0.1, 0.2]))
    spectra = spectra.at[0, :, 3, 3].set(jnp.asarray([0.01, 0.02]))
    spectra = spectra.at[0, :, 0, 1].set(jnp.asarray([0.5, 0.6]))
    spectra = spectra.at[0, :, 1, 0].set(jnp.asarray([0.5, 0.6]))
    return cosmology.CmbSpectrumTable(
        [2, 3],
        spectra,
        ("scalar",),
        provenance,
        background.realization,
    )


def test_cmb_raw_dl_temperature_and_packing_conventions():
    table = _table()
    d_ell = table.d_ell()
    factor = 2.0 * 3.0 / (2.0 * np.pi)
    np.testing.assert_allclose(d_ell[0, 0, 0, 0], factor * 2.0)
    np.testing.assert_allclose(d_ell[0, 0, 3, 3], table.spectra[0, 0, 3, 3])
    scaled = table.temperature_scaled(2.0)
    np.testing.assert_allclose(scaled[0, :, 0, 0], 4.0 * table.spectra[0, :, 0, 0])
    np.testing.assert_allclose(scaled[0, :, 0, 3], 2.0 * table.spectra[0, :, 0, 3])
    plan = cosmology.CmbSpectrumTransformPlan((0,), ((0, 0), (0, 1), (3, 3)))
    packed = plan.pack(table)
    assert packed.shape == (3, 2)
    np.testing.assert_allclose(packed[0], [2.0, 3.0])


def test_primordial_power_law_amplitude_and_gradients():
    primordial = cosmology.PrimordialPowerLaw(2.1e-9, 0.965, 0.05)
    np.testing.assert_allclose(primordial.scalar_power(0.05), 2.1e-9)
    np.testing.assert_allclose(primordial.tensor_power(0.05), 0.0)

    def value(amplitude):
        return cosmology.PrimordialPowerLaw(amplitude, 0.965, 0.05).scalar_power(0.1)

    derivative = jax.grad(value)(jnp.asarray(2.1e-9))
    assert derivative > 0.0


def test_cmb_bandpower_response_uses_canonical_packed_theory():
    table = _table()
    transform = cosmology.CmbSpectrumTransformPlan((0,), ((0, 0),))
    windows = jnp.asarray([[[1.0, 0.0]]])
    response = cosmology.CmbBandpowerResponsePlan(
        transform,
        windows,
        jnp.asarray([2.0]),
        jnp.eye(1),
        expected_temperature_unit="dimensionless-thermodynamic",
        response_id="synthetic-bandpower",
    ).evaluate(table)
    assert bool(response.valid)
    np.testing.assert_allclose(response.predicted_bandpowers, jnp.asarray([2.0]))
    np.testing.assert_allclose(response.log_likelihood, 0.0)

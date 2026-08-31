import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_cmb_table_convention_and_response_likelihood():
    cosmology = phx.applications.cosmology
    scale = cosmology.CODE_COSMOLOGY_SCALE
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="1",
        model_id="synthetic-cmb",
        numerical_policy_id="exact",
        scale_id=scale.scale_id,
        source_kind="external",
        differentiability="constant",
    )
    ell = jnp.asarray([2.0, 3.0, 4.0])
    spectra = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]])
    table = cosmology.CMBAngularPowerTable(
        ell,
        spectra,
        ("TT", "TE"),
        scale,
        provenance,
        convention="Cl",
        units="uK2",
        lensed=False,
    )
    converted = table.converted("Dl")
    factor = ell * (ell + 1.0) / (2.0 * jnp.pi)
    np.testing.assert_allclose(converted.spectra, spectra * factor[None, :])

    windows = jnp.zeros((2, 2, 3))
    windows = windows.at[0, 0, 0].set(1.0)
    windows = windows.at[1, 1, 2].set(1.0)
    observed = jnp.asarray([factor[0], 0.125 * factor[2]])
    response = cosmology.CMBResponsePlan(
        windows,
        observed,
        jnp.eye(2),
        ("TT", "TE"),
        expected_convention="Dl",
        expected_units="uK2",
        response_id="synthetic",
    ).evaluate(table)
    assert bool(response.valid)
    np.testing.assert_allclose(response.residual, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(response.log_likelihood, 0.0, atol=1.0e-14)

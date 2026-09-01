import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def test_massive_neutrino_species_and_request_identity():
    scale = cosmology.CosmologyScaleContract("Mpc", "solar_mass", "Gyr")
    species = (
        cosmology.MassiveNeutrinoSpecies(0.05),
        cosmology.MassiveNeutrinoSpecies(0.01, degeneracy=2.0),
    )
    request = cosmology.LinearTheoryRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=species,
    )
    assert request.request_id
    assert "massive_neutrino_mass_0" in request.realization.parameter_names
    assert "massive_neutrino_mass_1" in request.realization.parameter_names
    assert "massive_neutrino_temperature_ratio_1" in request.realization.parameter_names
    assert "massive_neutrino_degeneracy_1" in request.realization.parameter_names
    changed = cosmology.LinearTheoryRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=(cosmology.MassiveNeutrinoSpecies(0.06),),
    )
    assert request.request_id != changed.request_id
    with pytest.raises(ValueError, match="invalid"):
        cosmology.MassiveNeutrinoSpecies(-0.1)


def test_concrete_linear_theory_backend_returns_named_constant_products(tmp_path):
    root = Path(__file__).parents[2]
    worker = root / "_linear_theory_worker.py"
    scale = cosmology.CosmologyScaleContract("Mpc", "solar_mass", "Gyr")
    request = cosmology.LinearTheoryRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=(cosmology.MassiveNeutrinoSpecies(0.05),),
    )
    build = cosmology.BackendBuildManifest(
        backend="class",
        release="fixture",
        application=sys.executable,
        arguments=(str(worker), "{request}", "{output}"),
        license_id="internal-test",
    )
    backend = cosmology.ClassLinearTheoryBackend(
        build,
        cosmology.LinearTheoryResourcePolicy(timeout_seconds=60.0),
        str(tmp_path),
    )
    result = backend.run(request).products
    assert result.return_code == 0
    assert result.thermodynamics is not None
    assert result.power.descriptor.left_field == "cold_baryon"
    assert not result.power.provenance.differentiation.query_coordinates
    np.testing.assert_allclose(result.power.evaluate([1.0, 3.0], 0.75), [0.625, 1.875])
    np.testing.assert_allclose(
        result.transfer.evaluate("density/total_matter", [1.0, 3.0], 0.75),
        [1.5, 4.5],
    )
    np.testing.assert_allclose(
        jax.grad(lambda k: result.power.evaluate(k, 0.75))(jnp.asarray(1.5)),
        0.0,
    )

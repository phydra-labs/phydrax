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
    request = cosmology.CosmologyModelRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=species,
    )
    assert request.request_id
    assert request.realization.parameter_names[-2:] == (
        "massive_neutrino_mass_0",
        "massive_neutrino_mass_1",
    )
    changed = cosmology.CosmologyModelRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=(cosmology.MassiveNeutrinoSpecies(0.06),),
    )
    assert request.request_id != changed.request_id
    with pytest.raises(ValueError, match="invalid"):
        cosmology.MassiveNeutrinoSpecies(-0.1)


def test_subprocess_linear_theory_backend_returns_named_constant_products():
    root = Path(__file__).parents[2]
    worker = root / "_linear_theory_worker.py"
    scale = cosmology.CosmologyScaleContract("Mpc", "solar_mass", "Gyr")
    request = cosmology.CosmologyModelRequest(
        scale,
        hubble_constant=70.0,
        baryon_density=0.05,
        cold_dark_matter_density=0.25,
        neutrinos=(cosmology.MassiveNeutrinoSpecies(0.05),),
    )
    backend = cosmology.SubprocessCosmologyModelBackend(
        sys.executable,
        arguments=(str(worker), "{request}", "{output}"),
        backend_name="fixture-linear-theory",
        backend_version="current",
        numerical_policy_id="fixture-policy",
    )
    assert backend.availability().available
    result = backend.run(request)
    assert result.return_code == 0
    assert result.thermodynamics is not None
    assert result.power.descriptor.left_field == "cold_baryon"
    assert result.power.provenance.differentiability == "constant"
    np.testing.assert_allclose(result.power.evaluate([1.0, 3.0], 0.75), [0.625, 1.875])
    np.testing.assert_allclose(
        result.transfer.evaluate("density/total_matter", [1.0, 3.0], 0.75),
        [1.5, 4.5],
    )
    np.testing.assert_allclose(
        jax.grad(lambda k: result.power.evaluate(k, 0.75))(jnp.asarray(1.5)),
        0.0,
    )

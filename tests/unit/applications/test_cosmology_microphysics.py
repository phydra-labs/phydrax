import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_primordial_microphysics_conserves_nuclei_charge_and_updates_energy():
    artifact = cosmology.ScientificArtifactEnvelope(
        artifact_kind="primordial-rate-table",
        content_digest="fixture-rates",
        producer="test",
        producer_version="current",
        build_id="fixture",
        license_id="internal",
        resource_id="static",
        status="complete",
    )
    rates = jnp.zeros((len(cosmology.PRIMORDIAL_PROCESSES), 2, 3))
    rates = rates.at[6].set(1.0e-2)
    rates = rates.at[7].set(5.0e-3)
    rates = rates.at[8].set(2.0e-3)
    rates = rates.at[9].set(1.0e-4)
    rates = rates.at[10].set(2.0e-4)
    table = cosmology.PrimordialRateTable([0.1, 1.0, 10.0], [0.5, 1.0], rates, artifact)
    plan = cosmology.PrimordialMicrophysicsPlan(
        table, maximum_iterations=12, tolerance=1.0e-9
    )
    species = cosmology.PrimordialSpeciesState(
        [[0.9, 0.1, 0.09, 0.01, 0.0, 0.11]], [1.0], 0.5
    )
    result = plan.advance(species, 0.01)
    assert bool(result.successful)
    assert result.ledger.hydrogen_nuclei_defect < 1e-10
    assert result.ledger.helium_nuclei_defect < 1e-10
    assert result.ledger.charge_defect < 1e-10
    assert jnp.all(result.state.number_densities >= 0.0)
    assert jnp.all(jnp.isfinite(result.state.internal_energy))

    gas = cosmology.ComovingEulerState(jnp.asarray([[1.0, 0.0, 1.0]]), jnp.asarray(0.5))
    updated_gas, updated = plan.apply_to_gas(gas, species, 0.01)
    assert bool(updated.successful)
    np.testing.assert_allclose(
        updated_gas.cell_average[..., -1], updated.state.internal_energy
    )

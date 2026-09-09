import hashlib

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _manifest(payload=b"fusion-data"):
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-fusion-data",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )


def _fusion_plan():
    deuterium = phx.nuclear.NuclearSpeciesKey.from_nuclide(phx.nuclear.NuclideKey(1, 2))
    tritium = phx.nuclear.NuclearSpeciesKey.from_nuclide(phx.nuclear.NuclideKey(1, 3))
    helium4 = phx.nuclear.NuclearSpeciesKey.from_nuclide(phx.nuclear.NuclideKey(2, 4))
    neutron = phx.nuclear.NuclearSpeciesKey.from_particle(
        phx.nuclear.NuclearParticleKind.NEUTRON
    )
    table = phx.nuclear.NuclearSpeciesTable(
        (deuterium, tritium, helium4, neutron),
        np.asarray(
            [
                3.3435837724e-27,
                5.0073567446e-27,
                6.6446573357e-27,
                1.67492749804e-27,
            ]
        ),
        _manifest(b"masses"),
    )
    data = phx.nuclear.NuclearDataProvenance(
        _manifest(),
        "https://example.invalid/fusion",
        "synthetic-reactivity",
        "release",
        "dt",
    )
    participant = phx.nuclear.NuclearReactionParticipant
    channel = phx.nuclear.NuclearReactionChannel.from_species_table(
        "d-t",
        (participant(deuterium), participant(tritium)),
        (participant(helium4), participant(neutron)),
        table,
        data,
    )
    kev_to_j = float(
        phx.units.conversion_factor(phx.units.KILOELECTRONVOLT, phx.units.JOULE)
    )
    reactivity = phx.nuclear.TabulatedMaxwellianReactivity(
        kev_to_j * np.asarray([1.0, 5.0, 10.0, 25.0]),
        np.asarray([1.0e-24, 1.0e-23, 3.0e-23, 6.0e-23]),
        channel,
    )
    return phx.nuclear.ThermalFusionReactionPlan(channel, reactivity, table), channel


def test_dt_fusion_closes_particle_charge_and_energy_ledgers():
    plan, channel = _fusion_plan()
    thermal = 10.0 * float(
        phx.units.conversion_factor(phx.units.KILOELECTRONVOLT, phx.units.JOULE)
    )
    result = plan.evaluate(
        np.asarray([1.0e19, 2.0e19]),
        np.asarray([1.5e19, 2.5e19]),
        np.asarray([thermal, thermal]),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.total_power_density_w_m3,
        result.reaction_rate_density_m3_s * channel.q_value_j,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        np.sum(result.products.product_power_density_w_m3, axis=-1),
        result.total_power_density_w_m3,
    )


def test_fusion_reactivity_rejects_out_of_support_temperature():
    plan, _ = _fusion_plan()
    result = plan.evaluate(
        np.asarray([1.0e19]), np.asarray([1.0e19]), np.asarray([1.0e-30])
    )
    assert not bool(result.successful)
    assert not bool(result.support_valid[0])


def test_fusion_source_is_differentiable_inside_reactivity_support():
    plan, _ = _fusion_plan()
    thermal = 8.0 * float(
        phx.units.conversion_factor(phx.units.KILOELECTRONVOLT, phx.units.JOULE)
    )

    def power(density):
        return plan.evaluate(density, 2.0e19, thermal).total_power_density_w_m3

    derivative = jax.grad(power)(jnp.asarray(1.0e19))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0

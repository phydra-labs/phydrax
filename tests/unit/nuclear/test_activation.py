import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _data(name="activation"):
    payload = name.encode()
    reference = phx.qualification.ReferenceArtifactManifest(
        name,
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
    return phx.nuclear.NuclearDataProvenance(
        reference,
        f"https://example.invalid/{name}",
        "synthetic-library",
        "release",
        name,
    )


def _decay_network():
    cobalt = phx.nuclear.NuclideKey(27, 60)
    nickel = phx.nuclear.NuclideKey(28, 60)
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0], phx.units.MEGAELECTRONVOLT, source_id="decay-groups"
    )
    decay_rate = math.log(2.0) / 10.0
    transition = phx.nuclear.InventoryTransition(
        "beta-minus",
        cobalt,
        ((nickel, 1.0),),
        decay_rate,
        np.asarray([0.0]),
        1.0e-13,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        _data(),
    )
    return (
        phx.nuclear.ActivationNetworkPlan(
            (cobalt, nickel), groups, (transition,), error_tolerance=1.0e-12
        ).prepare(),
        decay_rate,
    )


def test_activation_matches_two_member_bateman_decay_chain():
    network, decay_rate = _decay_network()
    initial = network.inventory(np.asarray([1.0, 0.0]))
    result = network.step(initial, np.asarray([0.0]), 5.0)
    expected_parent = math.exp(-decay_rate * 5.0)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted.amounts_mol,
        [expected_parent, 1.0 - expected_parent],
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.activity_bq[0],
        phx.nuclear.AVOGADRO_PER_MOL * expected_parent * decay_rate,
        rtol=1.0e-11,
    )
    assert result.decay_heat_w > 0.0


def test_flux_driven_capture_uses_group_integrated_scalar_flux():
    hydrogen = phx.nuclear.NuclideKey(1, 1)
    deuterium = phx.nuclear.NuclideKey(1, 2)
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0, 2.0], phx.units.MEGAELECTRONVOLT, source_id="capture-groups"
    )
    transition = phx.nuclear.InventoryTransition(
        "capture",
        hydrogen,
        ((deuterium, 1.0),),
        0.0,
        np.asarray([1.0e-28, 2.0e-28]),
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _data("capture"),
    )
    network = phx.nuclear.ActivationNetworkPlan(
        (hydrogen, deuterium), groups, (transition,)
    ).prepare()
    flux = jnp.asarray([3.0e18, 4.0e18])
    rate = network.transition_rates(flux)
    np.testing.assert_allclose(rate, [1.1e-9])


def test_activation_is_differentiable_with_respect_to_flux():
    hydrogen = phx.nuclear.NuclideKey(1, 1)
    deuterium = phx.nuclear.NuclideKey(1, 2)
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0], phx.units.MEGAELECTRONVOLT, source_id="gradient-groups"
    )
    transition = phx.nuclear.InventoryTransition(
        "capture",
        hydrogen,
        ((deuterium, 1.0),),
        0.0,
        np.asarray([1.0e-28]),
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        _data("gradient-capture"),
    )
    network = phx.nuclear.ActivationNetworkPlan(
        (hydrogen, deuterium), groups, (transition,)
    ).prepare()
    initial = network.inventory([1.0, 0.0])

    def product(flux):
        return network.step(initial, jnp.asarray([flux]), 1.0).accepted.amounts_mol[1]

    derivative = jax.grad(product)(jnp.asarray(1.0e28))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0


def test_irradiation_schedule_tracks_valid_prefix():
    network, _ = _decay_network()
    schedule = phx.nuclear.IrradiationSchedulePlan(
        np.asarray([1.0, 2.0]), np.zeros((2, 1))
    )
    result = schedule.run(network, network.inventory([1.0, 0.0]))
    assert result.amounts_mol.shape == (3, 2)
    assert result.times_s[-1] == 3.0
    assert bool(np.all(result.valid_prefix))

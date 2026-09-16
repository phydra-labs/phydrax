import hashlib

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.cosmology._dark_radiation import DarkRadiationLedgerPlan
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_kernels import TwoBodyDifferentialKernelPlan
from phydrax.applications.cosmology._sidm_reactions import (
    DarkReactionStatus,
    DarkTwoBodyReactionPlan,
    InelasticSIDMPlan,
    write_inelastic_sidm_checkpoint,
)
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPacketState
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _kernel(
    first,
    second,
    speeds,
    total_cross_section,
    *,
    cosines=(-1.0, 0.0, 1.0),
    azimuths=None,
    differential=None,
):
    cosines = np.asarray(cosines)
    if differential is None:
        differential = np.full(
            (len(speeds), cosines.size), total_cross_section / (4.0 * np.pi)
        )
    payload = TwoBodyDifferentialKernelPlan.canonical_table_bytes(
        speeds, cosines, differential, azimuths=azimuths
    )
    digest = hashlib.sha256(payload).hexdigest()
    lineage = (f"synthetic:{first.species_id}:{second.species_id}",)
    manifest = ReferenceArtifactManifest(
        f"sidm-reaction-kernel-{digest[:16]}",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="test-synthetic-kernel",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"speed": 1.0, "cross_section": 1.0},
        uncertainty={"relative_cross_section": 1.0e-12},
        lineage_ids=lineage,
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="synthetic-sidm-differential-kernel",
        content_digest=digest,
        producer="phydrax-test",
        producer_version="1",
        build_id=f"sidm-reaction-kernel-{digest[:16]}",
        license_id=manifest.license_id,
        resource_id=manifest.manifest_id,
        status="complete",
        parent_artifact_ids=lineage,
    )
    return TwoBodyDifferentialKernelPlan(
        first,
        second,
        speeds,
        cosines,
        differential,
        azimuths=azimuths,
        source_artifact=artifact,
        reference_manifest=manifest,
        commercial_use=True,
        redistribution=True,
        training_use=True,
        export=True,
    )


def _plan(*, radiation_capacity=2):
    common = dict(charge_names=("dark",), charges=(0.0,))
    a = DarkSectorSpeciesPlan("a", 1.0, **common)
    b = DarkSectorSpeciesPlan("b", 1.0, **common)
    c = DarkSectorSpeciesPlan("c", 0.9, internal_energy=10.25, **common)
    d = DarkSectorSpeciesPlan("d", 0.9, internal_energy=10.25, **common)
    outgoing_speed = np.sqrt(1.0 / 0.45)
    speeds = np.asarray((0.0, outgoing_speed, 2.0, 5.0))
    forward = _kernel(a, b, speeds, 1.0)
    reverse = _kernel(c, d, speeds, 1.0 / 0.45)
    channel = DarkTwoBodyReactionPlan(
        (a, b),
        (c, d),
        forward,
        reverse,
        conserved_charge_names=("dark",),
        speed_of_light=10.0,
        maximum_speed_fraction=0.2,
        detailed_balance_tolerance=2.0e-12,
    )
    return InelasticSIDMPlan(
        (channel,),
        DarkRadiationLedgerPlan(radiation_capacity, speed_of_light=10.0),
    )


def _state(plan, left_velocity, right_velocity, *, weights=(1.0, 1.0), capacity=3):
    positions = jnp.zeros((capacity, 3), dtype=jnp.float64)
    active = jnp.arange(capacity) < 2
    microscopic = jnp.where(active, 1.0, 0.0)
    weight = jnp.zeros((capacity,), dtype=jnp.float64).at[:2].set(jnp.asarray(weights))
    macro = microscopic * weight
    velocity = jnp.zeros((capacity, 3), dtype=jnp.float64)
    velocity = velocity.at[0, 0].set(left_velocity)
    velocity = velocity.at[1, 0].set(right_velocity)
    packets = WeightedSIDMPacketState(
        positions,
        microscopic,
        weight,
        macro,
        macro[:, None] * 0.5 * velocity,
        active,
        jnp.arange(101, 101 + capacity, dtype=jnp.int64),
        jnp.full((capacity,), -1, dtype=jnp.int64),
        jnp.where(active, 0, -1).astype(jnp.int32),
        jnp.asarray(0.5),
    )
    return plan.initialize(
        packets,
        jnp.asarray((0, 1) + (-1,) * (capacity - 2), dtype=jnp.int32),
    )


def test_endothermic_threshold_is_closed_and_transaction_rolls_back():
    plan = _plan()
    state = _state(plan, 0.25, -0.25)

    result = eqx.filter_jit(plan.react)(state, 0, 1, jr.key(4))

    assert not bool(result.successful)
    assert int(result.evidence.status) == int(DarkReactionStatus.THRESHOLD_CLOSED)
    assert not bool(result.evidence.threshold_open)
    assert bool(result.evidence.rolled_back)
    assert bool(eqx.tree_equal(result.accepted_state, state))


def test_forward_then_reverse_has_recoil_detailed_balance_and_full_closure():
    plan = _plan()
    initial = _state(plan, 1.0, -1.0)

    forward = plan.react(initial, 0, 1, jr.key(7))
    assert bool(forward.successful)
    assert bool(forward.evidence.forward)
    assert bool(forward.evidence.threshold_open)
    assert bool(forward.evidence.detailed_balance_valid)
    np.testing.assert_allclose(
        forward.evidence.detailed_balance_forward,
        forward.evidence.detailed_balance_reverse,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        forward.evidence.rest_mass_energy_change, -20.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        forward.evidence.internal_energy_change, 20.5, atol=1.0e-12
    )
    np.testing.assert_allclose(forward.evidence.kinetic_energy_change, -0.5, atol=1.0e-12)
    np.testing.assert_array_equal(forward.evidence.radiation_energy_change, 0.0)
    np.testing.assert_allclose(forward.evidence.total_energy_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        forward.evidence.physical_momentum_defect, 0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(forward.evidence.charge_defect, 0.0, atol=1.0e-12)
    assert bool(forward.evidence.pm_mass_consistent)
    assert bool(forward.evidence.dynamic_pm_mass_required)
    np.testing.assert_allclose(
        forward.evidence.gravitational_mass_change, -0.2, atol=1.0e-14
    )
    np.testing.assert_array_equal(forward.accepted_state.species_indices[:2], (2, 3))

    reverse = plan.react(forward.accepted_state, 0, 1, jr.key(8))
    assert bool(reverse.successful)
    assert not bool(reverse.evidence.forward)
    assert float(reverse.evidence.outgoing_relative_speed) > float(
        forward.evidence.outgoing_relative_speed
    )
    np.testing.assert_allclose(reverse.evidence.total_energy_defect, 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(reverse.accepted_state.species_indices[:2], (0, 1))


def test_partial_rates_select_only_matching_open_channel_with_rate_weights():
    base = _plan()
    first = base.channels[0]
    common = dict(charge_names=("dark",), charges=(0.0,))
    e = DarkSectorSpeciesPlan("e", 0.9, internal_energy=10.25, **common)
    f = DarkSectorSpeciesPlan("f", 0.9, internal_energy=10.25, **common)
    recoil_speed = np.sqrt(1.0 / 0.45)
    speeds = np.asarray((0.0, recoil_speed, 2.0, 5.0))
    second = DarkTwoBodyReactionPlan(
        first.incoming_species,
        (e, f),
        _kernel(first.incoming_species[0], first.incoming_species[1], speeds, 3.0),
        _kernel(e, f, speeds, 3.0 / 0.45),
        conserved_charge_names=("dark",),
        speed_of_light=10.0,
        maximum_speed_fraction=0.2,
        detailed_balance_tolerance=2.0e-12,
    )
    plan = InelasticSIDMPlan(
        (first, second), DarkRadiationLedgerPlan(2, speed_of_light=10.0)
    )
    state = _state(plan, 1.0, -1.0)

    rates = plan.partial_rates(state, 0, 1)
    np.testing.assert_allclose(rates.partial_rates, (2.0, 6.0, 0.0, 0.0))
    result = plan.react(state, 0, 1, jr.key(44))

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.evidence.channel_probabilities, (0.25, 0.75, 0.0, 0.0)
    )
    selected = int(result.evidence.selected_channel)
    assert selected in (0, 1)
    expected_species = (2, 3) if selected == 0 else (4, 5)
    np.testing.assert_array_equal(
        result.accepted_state.species_indices[:2], expected_species
    )


def test_thermal_equilibrium_ratio_includes_mass_degeneracy_and_energy_gap():
    plan = _plan()
    channel = plan.channels[0]
    thermal_energy = 2.0
    expected = (0.9 * 0.9) ** 1.5 * np.exp(-0.5 / thermal_energy)

    np.testing.assert_allclose(
        channel.thermal_equilibrium_ratio(thermal_energy), expected, rtol=2.0e-14
    )


def test_weight_split_preserves_ids_and_records_child_lineage():
    plan = _plan()
    state = _state(plan, 1.0, -1.0, weights=(2.0, 1.0), capacity=3)

    result = plan.react(state, 0, 1, jr.key(12))

    assert bool(result.successful)
    assert bool(result.evidence.split_required)
    assert int(result.evidence.child_slot) == 2
    assert bool(result.evidence.stable_ids_preserved)
    assert bool(result.evidence.lineage_valid)
    np.testing.assert_array_equal(
        result.accepted_state.packets.packet_ids,
        state.packets.packet_ids,
    )
    assert int(result.accepted_state.packets.parent_packet_ids[2]) == 101
    assert int(result.accepted_state.packets.lineage_depth[2]) == 1
    assert int(result.accepted_state.species_indices[2]) == 0
    assert float(result.accepted_state.packets.weights[2]) == 1.0
    np.testing.assert_allclose(result.evidence.total_energy_defect, 0.0, atol=2.0e-12)


def test_full_capacity_split_and_relativistic_input_refuse_without_mutation():
    plan = _plan()
    full = _state(plan, 1.0, -1.0, weights=(2.0, 1.0), capacity=2)
    exhausted = plan.react(full, 0, 1, jr.key(15))
    assert not bool(exhausted.successful)
    assert int(exhausted.evidence.status) == int(DarkReactionStatus.CAPACITY_EXHAUSTED)
    assert bool(eqx.tree_equal(exhausted.accepted_state, full))

    relativistic = _state(plan, 2.1, -2.1)
    refused = plan.react(relativistic, 0, 1, jr.key(16))
    assert not bool(refused.successful)
    assert int(refused.evidence.status) == int(DarkReactionStatus.RELATIVISTIC_REFUSED)
    assert bool(refused.evidence.relativistic_refusal)
    assert bool(eqx.tree_equal(refused.accepted_state, relativistic))


def test_full_angular_law_not_one_random_angle_controls_microreversibility():
    base = _plan()
    first = base.channels[0]
    speeds = np.asarray((0.0, np.sqrt(1.0 / 0.45), 2.0, 5.0))
    azimuths = np.linspace(0.0, 2.0 * np.pi, 5)
    forward_shape = np.asarray((1.0, 1.5, 1.0, 0.5, 1.0))
    reverse_shape = np.asarray((1.0, 0.5, 1.0, 1.5, 1.0))
    forward_values = np.broadcast_to(
        forward_shape[None, None, :] / (4.0 * np.pi), (4, 3, 5)
    )
    reverse_values = np.broadcast_to(
        reverse_shape[None, None, :] / (0.45 * 4.0 * np.pi), (4, 3, 5)
    )
    channel = DarkTwoBodyReactionPlan(
        first.incoming_species,
        first.outgoing_species,
        _kernel(
            *first.incoming_species,
            speeds,
            1.0,
            azimuths=azimuths,
            differential=forward_values,
        ),
        _kernel(
            *first.outgoing_species,
            speeds,
            1.0 / 0.45,
            azimuths=azimuths,
            differential=reverse_values,
        ),
        conserved_charge_names=("dark",),
        speed_of_light=10.0,
        maximum_speed_fraction=0.2,
        detailed_balance_tolerance=2.0e-12,
    )
    plan = InelasticSIDMPlan((channel,), DarkRadiationLedgerPlan(1, speed_of_light=10.0))
    state = _state(plan, 1.0, -1.0)

    result = plan.react(state, 0, 1, jr.key(101))

    assert not bool(result.successful)
    assert bool(result.evidence.integrated_balance_valid)
    assert not bool(result.evidence.detailed_balance_valid)
    assert int(result.evidence.status) == int(DarkReactionStatus.DETAILED_BALANCE_FAILURE)
    assert bool(result.evidence.rolled_back)
    assert bool(eqx.tree_equal(result.accepted_state, state))


def test_large_capacity_lineage_is_depth_ordered_and_cycle_safe(tmp_path):
    plan = _plan()
    large = _state(plan, 1.0, -1.0, capacity=8192)
    assert bool(plan.state_valid(large))

    self_parent = eqx.tree_at(
        lambda value: value.packets.parent_packet_ids,
        large,
        large.packets.parent_packet_ids.at[0].set(large.packets.packet_ids[0]),
    )
    self_parent = eqx.tree_at(
        lambda value: value.packets.lineage_depth,
        self_parent,
        self_parent.packets.lineage_depth.at[0].set(1),
    )
    assert not bool(plan.state_valid(self_parent))
    with pytest.raises(ValueError, match="valid accepted"):
        write_inelastic_sidm_checkpoint(tmp_path / "self-parent.phx", plan, self_parent)

    cycle = eqx.tree_at(
        lambda value: value.packets.parent_packet_ids,
        large,
        large.packets.parent_packet_ids.at[0]
        .set(large.packets.packet_ids[1])
        .at[1]
        .set(large.packets.packet_ids[0]),
    )
    cycle = eqx.tree_at(
        lambda value: value.packets.lineage_depth,
        cycle,
        cycle.packets.lineage_depth.at[0].set(2).at[1].set(1),
    )
    assert not bool(plan.state_valid(cycle))

    inactive_parent = eqx.tree_at(
        lambda value: value.packets.parent_packet_ids,
        large,
        large.packets.parent_packet_ids.at[0].set(large.packets.packet_ids[2]),
    )
    inactive_parent = eqx.tree_at(
        lambda value: value.packets.lineage_depth,
        inactive_parent,
        inactive_parent.packets.lineage_depth.at[0].set(1),
    )
    assert not bool(plan.state_valid(inactive_parent))


def test_tiny_unit_mass_redistribution_requires_dynamic_pm_refresh():
    common = dict(charge_names=("dark",), charges=(0.0,))
    a = DarkSectorSpeciesPlan("tiny-a", 1.0e-30, **common)
    b = DarkSectorSpeciesPlan("tiny-b", 1.0e-30, **common)
    c = DarkSectorSpeciesPlan("tiny-c", 1.5e-30, **common)
    d = DarkSectorSpeciesPlan("tiny-d", 0.5e-30, **common)
    outgoing_speed = np.sqrt(2.0e-30 / 0.375e-30)
    speeds = np.asarray((0.0, 2.0, outgoing_speed, 5.0))
    channel = DarkTwoBodyReactionPlan(
        (a, b),
        (c, d),
        _kernel(a, b, speeds, 1.0),
        _kernel(c, d, speeds, 4.0 / 3.0),
        conserved_charge_names=("dark",),
        speed_of_light=100.0,
        maximum_speed_fraction=0.1,
        detailed_balance_tolerance=2.0e-12,
    )
    plan = InelasticSIDMPlan((channel,), DarkRadiationLedgerPlan(1, speed_of_light=100.0))
    microscopic = jnp.asarray((1.0e-30, 1.0e-30))
    velocities = jnp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    packets = WeightedSIDMPacketState(
        jnp.zeros((2, 3)),
        microscopic,
        jnp.ones((2,)),
        microscopic,
        microscopic[:, None] * 0.5 * velocities,
        jnp.ones((2,), dtype=bool),
        jnp.asarray((501, 502), dtype=jnp.int64),
        jnp.full((2,), -1, dtype=jnp.int64),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray(0.5),
    )
    state = plan.initialize(packets, jnp.asarray((0, 1), dtype=jnp.int32))

    result = plan.react(state, 0, 1, jr.key(102))

    assert bool(result.successful)
    np.testing.assert_allclose(result.evidence.gravitational_mass_change, 0.0)
    np.testing.assert_allclose(
        result.evidence.packet_microscopic_mass_change,
        (0.5e-30, -0.5e-30),
        rtol=2.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.evidence.packet_gravitational_mass_change,
        (0.5e-30, -0.5e-30),
        rtol=2.0e-14,
        atol=0.0,
    )
    assert bool(result.evidence.dynamic_pm_mass_required)


def test_units_and_identical_final_state_normalization_are_explicit():
    plan = _plan()
    channel = plan.channels[0]
    with pytest.raises(ValueError, match="energy/momentum/position units"):
        InelasticSIDMPlan(
            (channel,),
            DarkRadiationLedgerPlan(
                1,
                speed_of_light=10.0,
                energy_unit="different-energy",
            ),
        )

    a, b = channel.incoming_species
    identical = DarkSectorSpeciesPlan(
        "identical-final",
        0.9,
        internal_energy=10.25,
        charge_names=("dark",),
        charges=(0.0,),
    )
    speeds = np.asarray((0.0, np.sqrt(1.0 / 0.45), 2.0, 5.0))
    forward = _kernel(a, b, speeds, 1.0)
    reverse = TwoBodyDifferentialKernelPlan.constant_isotropic(identical, 1.0 / 0.45)
    with pytest.raises(ValueError, match="forward_outgoing_convention"):
        DarkTwoBodyReactionPlan(
            (a, b),
            (identical, identical),
            forward,
            reverse,
            conserved_charge_names=("dark",),
            speed_of_light=10.0,
            maximum_speed_fraction=0.2,
        )

    bound = DarkTwoBodyReactionPlan(
        (a, b),
        (identical, identical),
        forward,
        reverse,
        conserved_charge_names=("dark",),
        forward_outgoing_convention="labelled-full-sphere",
        speed_of_light=10.0,
        maximum_speed_fraction=0.2,
    )
    assert bound.forward_outgoing_convention == "labelled-full-sphere"
    assert (
        bound.forward_outgoing_convention
        == bound.reverse_kernel.identical_particle_convention
    )

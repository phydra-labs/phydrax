import hashlib

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax._array_archive as archive_module
from phydrax._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from phydrax.applications.cosmology._dark_radiation import (
    DarkRadiationLedgerPlan,
    DarkRadiationPacket,
    DarkRadiationStatus,
)
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._sidm_kernels import TwoBodyDifferentialKernelPlan
from phydrax.applications.cosmology._sidm_reactions import (
    DarkTwoBodyReactionPlan,
    InelasticSIDMPlan,
    read_inelastic_sidm_checkpoint,
    write_inelastic_sidm_checkpoint,
)
from phydrax.applications.cosmology._sidm_weighted import WeightedSIDMPacketState
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest


def _workflow():
    species = tuple(
        DarkSectorSpeciesPlan(
            name,
            mass,
            internal_energy=internal,
            charge_names=("dark",),
            charges=(0.0,),
        )
        for name, mass, internal in (
            ("a", 1.0, 0.0),
            ("b", 1.0, 0.0),
            ("c", 0.9, 10.25),
            ("d", 0.9, 10.25),
        )
    )
    recoil_speed = np.sqrt(1.0 / 0.45)
    speed_nodes = np.asarray((0.0, recoil_speed, 2.0, 5.0))

    def kernel(first, second, cross_section):
        cosines = np.asarray((-1.0, 0.0, 1.0))
        differential = np.full((4, 3), cross_section / (4.0 * np.pi))
        payload = TwoBodyDifferentialKernelPlan.canonical_table_bytes(
            speed_nodes, cosines, differential
        )
        digest = hashlib.sha256(payload).hexdigest()
        lineage = (f"synthetic:{first.species_id}:{second.species_id}",)
        manifest = ReferenceArtifactManifest(
            f"inelastic-workflow-kernel-{digest[:16]}",
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
            build_id=f"inelastic-workflow-kernel-{digest[:16]}",
            license_id=manifest.license_id,
            resource_id=manifest.manifest_id,
            status="complete",
            parent_artifact_ids=lineage,
        )
        return TwoBodyDifferentialKernelPlan(
            first,
            second,
            speed_nodes,
            cosines,
            differential,
            source_artifact=artifact,
            reference_manifest=manifest,
            commercial_use=True,
            redistribution=True,
            training_use=True,
            export=True,
        )

    reaction = DarkTwoBodyReactionPlan(
        species[:2],
        species[2:],
        kernel(species[0], species[1], 1.0),
        kernel(species[2], species[3], 1.0 / 0.45),
        conserved_charge_names=("dark",),
        speed_of_light=10.0,
        maximum_speed_fraction=0.2,
        detailed_balance_tolerance=2.0e-12,
    )
    plan = InelasticSIDMPlan((reaction,), DarkRadiationLedgerPlan(1, speed_of_light=10.0))
    velocities = jnp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    active = jnp.asarray((True, True, False))
    microscopic = jnp.asarray((1.0, 1.0, 0.0))
    weights = jnp.asarray((1.0, 1.0, 0.0))
    macro = microscopic * weights
    packets = WeightedSIDMPacketState(
        jnp.zeros((3, 3)),
        microscopic,
        weights,
        macro,
        macro[:, None] * 0.5 * velocities,
        active,
        jnp.asarray((901, 902, 903), dtype=jnp.int64),
        jnp.full((3,), -1, dtype=jnp.int64),
        jnp.asarray((0, 0, -1), dtype=jnp.int32),
        jnp.asarray(0.5),
    )
    return plan, plan.initialize(packets, jnp.asarray((0, 1, -1), dtype=jnp.int32))


def test_reaction_radiation_capacity_and_restart_are_one_atomic_workflow(
    tmp_path, monkeypatch
):
    plan, initial = _workflow()
    reacted = eqx.filter_jit(plan.react)(initial, 0, 1, jr.key(21))
    assert bool(reacted.successful)
    assert bool(reacted.evidence.dynamic_pm_mass_required)
    assert bool(reacted.evidence.detailed_balance_valid)

    retained_state = eqx.tree_at(
        lambda value: value.packets.canonical_momenta,
        reacted.accepted_state,
        reacted.accepted_state.packets.canonical_momenta * jnp.sqrt(0.5),
    )
    packet = DarkRadiationPacket(
        packet_id=jnp.asarray(7001, dtype=jnp.int64),
        species_id=jnp.asarray(5, dtype=jnp.int32),
        source_event_id=jnp.asarray(8001, dtype=jnp.int64),
        parent_ids=jnp.asarray((-1, -1), dtype=jnp.int64),
        physical_energy=jnp.asarray(0.25),
        physical_momentum=jnp.zeros((3,)),
        comoving_position=jnp.asarray((9.0, 9.0, 9.0)),
        emission_scale_factor=jnp.asarray(9.0),
    )
    teleported = eqx.tree_at(
        lambda value: value.packets.positions,
        retained_state,
        retained_state.packets.positions.at[0, 0].set(0.25),
    )
    refused_teleport = plan.export_radiation(
        reacted.accepted_state,
        teleported,
        packet,
        jnp.asarray((0, 1), dtype=jnp.int32),
    )
    assert not bool(refused_teleport.successful)
    assert not bool(refused_teleport.evidence.positions_preserved)
    assert bool(refused_teleport.evidence.rolled_back)
    assert bool(eqx.tree_equal(refused_teleport.accepted_state, reacted.accepted_state))

    invalid_source = eqx.tree_at(
        lambda value: value.packets.parent_packet_ids,
        reacted.accepted_state,
        reacted.accepted_state.packets.parent_packet_ids.at[0].set(901),
    )
    invalid_source = eqx.tree_at(
        lambda value: value.packets.lineage_depth,
        invalid_source,
        invalid_source.packets.lineage_depth.at[0].set(1),
    )
    refused_source = plan.export_radiation(
        invalid_source,
        retained_state,
        packet,
        jnp.asarray((0, 1), dtype=jnp.int32),
    )
    assert not bool(refused_source.successful)
    assert not bool(refused_source.evidence.source_state_valid)
    assert bool(refused_source.evidence.rolled_back)
    assert bool(eqx.tree_equal(refused_source.accepted_state, invalid_source))

    maximum_epoch = jnp.iinfo(jnp.int64).max
    max_epoch_source = eqx.tree_at(
        lambda value: value.reaction_epoch,
        reacted.accepted_state,
        jnp.asarray(maximum_epoch, dtype=jnp.int64),
    )
    max_epoch_retained = eqx.tree_at(
        lambda value: value.reaction_epoch,
        retained_state,
        jnp.asarray(maximum_epoch, dtype=jnp.int64),
    )
    refused_epoch = plan.export_radiation(
        max_epoch_source,
        max_epoch_retained,
        packet,
        jnp.asarray((0, 1), dtype=jnp.int32),
    )
    assert not bool(refused_epoch.successful)
    assert not bool(refused_epoch.evidence.candidate_state_valid)
    assert bool(refused_epoch.evidence.rolled_back)
    assert bool(eqx.tree_equal(refused_epoch.accepted_state, max_epoch_source))

    radiated = eqx.filter_jit(plan.export_radiation)(
        reacted.accepted_state,
        retained_state,
        packet,
        jnp.asarray((0, 1), dtype=jnp.int32),
    )
    assert bool(radiated.successful)
    assert bool(radiated.evidence.conservation_valid)
    assert bool(radiated.evidence.export.four_momentum_balanced)
    assert bool(
        jnp.all(
            jnp.abs(radiated.evidence.four_momentum_defect)
            <= radiated.evidence.export.four_momentum_tolerance
        )
    )
    assert int(radiated.accepted_state.radiation.packet_ids[0]) == 7001
    assert int(radiated.accepted_state.radiation.source_event_ids[0]) == 8001
    np.testing.assert_array_equal(
        radiated.accepted_state.radiation.parent_ids[0], (901, 902)
    )
    np.testing.assert_array_equal(
        radiated.accepted_state.radiation.comoving_position[0], (0.0, 0.0, 0.0)
    )
    assert float(radiated.accepted_state.radiation.emission_scale_factors[0]) == 0.5
    np.testing.assert_allclose(
        radiated.accepted_state.packets.canonical_momenta,
        retained_state.packets.canonical_momenta,
    )

    second_retained = eqx.tree_at(
        lambda value: value.packets.canonical_momenta,
        radiated.accepted_state,
        radiated.accepted_state.packets.canonical_momenta * jnp.sqrt(0.5),
    )
    exhausted = plan.export_radiation(
        radiated.accepted_state,
        second_retained,
        DarkRadiationPacket(
            packet_id=jnp.asarray(7002, dtype=jnp.int64),
            species_id=packet.species_id,
            source_event_id=jnp.asarray(8002, dtype=jnp.int64),
            parent_ids=packet.parent_ids,
            physical_energy=jnp.asarray(0.125),
            physical_momentum=jnp.zeros((3,)),
            comoving_position=packet.comoving_position,
            emission_scale_factor=packet.emission_scale_factor,
        ),
        jnp.asarray((0, 1), dtype=jnp.int32),
    )
    assert not bool(exhausted.successful)
    assert int(exhausted.evidence.export.status) == int(
        DarkRadiationStatus.CAPACITY_EXHAUSTED
    )
    assert bool(exhausted.evidence.rolled_back)
    assert bool(eqx.tree_equal(exhausted.accepted_state, radiated.accepted_state))

    checkpoint = tmp_path / "inelastic-sidm.phx"
    write_inelastic_sidm_checkpoint(checkpoint, plan, radiated.accepted_state)
    restored = read_inelastic_sidm_checkpoint(checkpoint, plan, radiated.accepted_state)
    assert bool(eqx.tree_equal(restored, radiated.accepted_state))
    wrong_support = eqx.tree_at(
        lambda value: value.packets.packet_ids,
        radiated.accepted_state,
        radiated.accepted_state.packets.packet_ids.at[0].set(9999),
    )
    with pytest.raises(ValueError, match="particle support"):
        read_inelastic_sidm_checkpoint(checkpoint, plan, wrong_support)

    manifest, archived_arrays = read_array_archive(checkpoint)
    archive_manifest = {key: value for key, value in manifest.items() if key != "arrays"}
    shaped_name = next(
        name
        for name, value in archived_arrays.items()
        if value.ndim > 0 and value.shape[0] > 1
    )
    wrong_shape_arrays = dict(archived_arrays)
    wrong_shape_arrays[shaped_name] = wrong_shape_arrays[shaped_name][:-1]
    wrong_shape = write_array_archive(
        tmp_path / "wrong-shape.phx",
        manifest=archive_manifest,
        arrays=wrong_shape_arrays,
    )
    floating_name = next(
        name
        for name, value in archived_arrays.items()
        if np.issubdtype(value.dtype, np.floating)
    )
    wrong_dtype_arrays = dict(archived_arrays)
    replacement_dtype = (
        np.float32
        if wrong_dtype_arrays[floating_name].dtype != np.dtype(np.float32)
        else np.float64
    )
    wrong_dtype_arrays[floating_name] = wrong_dtype_arrays[floating_name].astype(
        replacement_dtype
    )
    wrong_dtype = write_array_archive(
        tmp_path / "wrong-dtype.phx",
        manifest=archive_manifest,
        arrays=wrong_dtype_arrays,
    )
    extra_member_arrays = dict(archived_arrays)
    extra_member_arrays["state/unexpected"] = np.asarray(0, dtype=np.int8)
    extra_member = write_array_archive(
        tmp_path / "extra-member.phx",
        manifest=archive_manifest,
        arrays=extra_member_arrays,
    )

    def fail_if_numpy_loads(*args, **kwargs):
        raise AssertionError("np.load must not run before inventory admission")

    monkeypatch.setattr(archive_module.np, "load", fail_if_numpy_loads)
    for corrupt in (wrong_shape, wrong_dtype, extra_member):
        with pytest.raises(ArrayArchiveCorruptionError, match="template"):
            read_inelastic_sidm_checkpoint(corrupt, plan, radiated.accepted_state)

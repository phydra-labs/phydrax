#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_radiation import (
    DarkRadiationLedgerPlan,
    DarkRadiationPacket,
)
from phydrax.applications.cosmology._dark_radiation_transport import (
    average_packets_to_m1,
    DarkRadiationLedgerSourceAdapter,
    packet_gravity_source,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.equations._dark_radiation_moments import (
    CosmologicalMultigroupM1System,
)
from phydrax.metrix import ADMGridGeometry, RelativityConvention
from phydrax.solver._dark_radiation_packets import DarkRadiationPacketPlan
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan


def _units():
    return RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )


def _frame(units, *, time, scale_factor, token):
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(token),
        chart_id="workflow-flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="workflow-cell",
        geometry_lineage_id="workflow-flrw",
    )
    return LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        jnp.zeros((4,)),
        jnp.asarray(time),
        jnp.asarray(scale_factor),
        observer_id="workflow-observer",
        orientation_id="right-handed-future",
    )


def _source_ledger():
    plan = DarkRadiationLedgerPlan(2, speed_of_light=1.0)
    ledger = plan.empty()
    for packet_id, event_id, position, direction in (
        (101, 31, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        (102, 32, (0.5, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ):
        packet = DarkRadiationPacket(
            packet_id,
            7,
            event_id,
            (11, 12),
            jnp.asarray(1.0),
            jnp.asarray(direction),
            jnp.asarray(position),
            jnp.asarray(1.0),
        )
        retained = jnp.asarray((3.0, 0.0, 0.0, 0.0))
        source = retained + jnp.concatenate(
            (packet.physical_energy[None], packet.physical_momentum)
        )
        exported = plan.export(ledger, packet, source, retained)
        assert bool(exported.successful)
        ledger = exported.accepted_ledger
    return plan, ledger


def test_reaction_packets_material_events_epoch_moments_and_gravity_close():
    units = _units()
    frame0 = _frame(units, time=0.0, scale_factor=1.0, token=1)
    frame1 = _frame(units, time=0.25, scale_factor=1.1, token=2)
    epoch_plan = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=2,
        product_capacity=1,
        radiation_capacity=2,
        work_capacity=1,
        frontier_capacity=1,
        packet_width=1,
        event_width=1,
        product_width=1,
        radiation_width=20,
        work_width=1,
        frontier_width=1,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
    )
    packet_plan = DarkRadiationPacketPlan(
        2,
        2,
        units,
        jnp.asarray((0.1, 2.0)),
        jnp.asarray((-4.0, -4.0, -4.0)),
        jnp.asarray((4.0, 4.0, 4.0)),
        epoch_plan=epoch_plan,
    )
    ledger_plan, ledger = _source_ledger()
    adapted = DarkRadiationLedgerSourceAdapter(ledger_plan, packet_plan).adapt(
        ledger,
        packet_plan.empty(frame0, epoch_manifest_id="a" * 64),
        frame0,
        jnp.asarray((1, 1), dtype=jnp.int32),
        jnp.asarray(((1.0, 0.4, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
        jnp.asarray((0.05, 0.05)),
        jnp.asarray(((7, 11), (13, 17)), dtype=jnp.uint32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((1.0, 1.0)),
        target_state_id="workflow-packets",
    )
    assert bool(adapted.evidence.successful)

    # The first packet samples a scattering slab; the second samples an
    # absorbing sphere. Geometry ownership remains with the caller and enters
    # transport only as explicit physical opacities.
    slab_mask = jnp.abs(adapted.admission.accepted_state.comoving_position[:, 0]) < 0.1
    sphere_mask = (
        jnp.linalg.norm(
            adapted.admission.accepted_state.comoving_position
            - jnp.asarray((0.5, 0.0, 0.0)),
            axis=-1,
        )
        < 0.1
    )
    step = packet_plan.advance(
        adapted.admission.accepted_state,
        frame0,
        frame1,
        sphere_mask.astype(jnp.float64) * 4.0,
        slab_mask.astype(jnp.float64) * 4.0,
        jnp.diag(jnp.asarray((1.0, 0.5, 0.5, 0.5))),
        next_epoch_manifest_id="b" * 64,
        end_frame_realization_id=frame1.realization_id(),
    )
    assert bool(step.successful)
    assert int(step.evidence.event_count) == 2
    assert int(jnp.sum(step.accepted_state.active_mask)) == 1
    assert bool(step.exchange.exact_opposite)
    np.testing.assert_allclose(step.evidence.four_force_residual, 0.0, atol=0.0)

    epoch = packet_plan.to_dark_sector_epoch_state(
        step.accepted_state,
        parent_epoch_manifest_id="a" * 64,
    )
    assert epoch.epoch_sequence == 1
    assert int(epoch.resident_counts[3]) == 1
    np.testing.assert_allclose(epoch.conservation_residual, 0.0, atol=0.0)

    m1 = CosmologicalMultigroupM1System(
        packet_plan.group_edges,
        physical_light_speed=1.0,
        reduced_light_speed=0.5,
    )
    moments = average_packets_to_m1(
        packet_plan,
        step.accepted_state,
        m1,
        jnp.asarray((0, 0)),
        jnp.asarray((8.0,)),
        target_state_id="workflow-m1",
    )
    assert bool(moments.receipt.accepted)
    np.testing.assert_allclose(moments.receipt.conservation_defect, 0.0, atol=1e-12)
    assert moments.frame_realization_id == frame1.realization_id()

    gravity = packet_gravity_source(
        packet_plan,
        step.accepted_state,
        frame1,
        step.exchange,
        jnp.asarray(8.0),
    )
    assert bool(gravity.projection.all_active_valid)
    assert gravity.projection.snapshot_token == frame1.frame_token
    assert gravity.frame_realization_id == frame1.realization_id()
    assert gravity.exchange.source_state_id == gravity.source_state_id

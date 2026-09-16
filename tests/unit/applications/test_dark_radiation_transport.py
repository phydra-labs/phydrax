#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
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
    DarkRadiationM1Checkpoint,
    hierarchy_gravity_source,
    packet_gravity_source,
    packet_transport_profile,
    read_dark_radiation_m1_checkpoint,
    vet_stress_energy_projection,
    write_dark_radiation_m1_checkpoint,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.equations._dark_radiation_moments import (
    CosmologicalMultigroupM1System,
    DarkRadiationBoltzmannHierarchyPlan,
    DarkRadiationConversionReceipt,
    DarkRadiationFourForce,
    DarkRadiationVETPlan,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.solver._dark_radiation_packets import DarkRadiationPacketPlan
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan


def _setup():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(17),
        chart_id="flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat-lineage",
    )
    chart = CoordinateChart("flat", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        source_id="transport-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="transport-observer",
        orientation_id="right-handed-future",
    )
    packet_plan = DarkRadiationPacketPlan(
        2,
        2,
        units,
        jnp.asarray((0.1, 2.0, 4.0)),
        jnp.asarray((-2.0, -2.0, -2.0)),
        jnp.asarray((2.0, 2.0, 2.0)),
        epoch_plan=DarkSectorEpochPlan(
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
        ),
    )
    return units, frame, packet_plan


def _ledger():
    plan = DarkRadiationLedgerPlan(2, speed_of_light=1.0)
    empty = plan.empty()
    packet = DarkRadiationPacket(
        101,
        7,
        41,
        (11, 12),
        jnp.asarray(1.0),
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.zeros((3,)),
        jnp.asarray(1.0),
    )
    retained = jnp.asarray((2.0, 0.0, 0.0, 0.0))
    source = retained + jnp.asarray((1.0, 1.0, 0.0, 0.0))
    result = plan.export(empty, packet, source, retained)
    assert bool(result.successful)
    return plan, result.accepted_ledger


def test_ledger_adapter_creates_distinct_packet_state_with_explicit_frame_and_support():
    _, frame, packet_plan = _setup()
    ledger_plan, ledger = _ledger()
    target = packet_plan.empty(frame, epoch_manifest_id="a" * 64)
    adapter = DarkRadiationLedgerSourceAdapter(ledger_plan, packet_plan)
    result = adapter.adapt(
        ledger,
        target,
        frame,
        jnp.asarray((1, 0), dtype=jnp.int32),
        jnp.asarray(((1.0, 0.2, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))),
        jnp.asarray((0.5, 0.0)),
        jnp.asarray(((2, 3), (0, 0)), dtype=jnp.uint32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((2.0, 0.0)),
        target_state_id="packet-state-0",
    )
    assert bool(result.evidence.successful)
    assert bool(result.admission.accepted_state.active_mask[0])
    assert result.admission.accepted_state.frame_id == frame.frame_id
    assert result.admission.accepted_state.unit_contract_id == frame.units.contract_id
    np.testing.assert_allclose(result.evidence.four_momentum_defect, 0.0, atol=1e-7)


def test_packet_moment_receipt_and_stress_energy_are_conservative():
    units, frame, packet_plan = _setup()
    ledger_plan, ledger = _ledger()
    state = (
        DarkRadiationLedgerSourceAdapter(ledger_plan, packet_plan)
        .adapt(
            ledger,
            packet_plan.empty(frame, epoch_manifest_id="a" * 64),
            frame,
            jnp.asarray((1, 0), dtype=jnp.int32),
            jnp.asarray(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))),
            jnp.asarray((1.0, 0.0)),
            jnp.asarray(((2, 3), (0, 0)), dtype=jnp.uint32),
            jnp.asarray((0, 0), dtype=jnp.int32),
            jnp.asarray((1.0, 0.0)),
            target_state_id="packet-state-0",
        )
        .admission.accepted_state
    )
    m1 = CosmologicalMultigroupM1System(
        packet_plan.group_edges,
        physical_light_speed=1.0,
        reduced_light_speed=0.5,
    )
    moments = average_packets_to_m1(
        packet_plan,
        state,
        m1,
        jnp.asarray((0, 0)),
        jnp.asarray((2.0,)),
        target_state_id="m1-state-0",
    )
    assert bool(moments.receipt.accepted)
    np.testing.assert_allclose(moments.receipt.conservation_defect, 0.0, atol=1e-7)
    assert moments.moment_state.shape == (1, 8)

    exchange = DarkRadiationFourForce.paired(
        jnp.zeros((4,)),
        state.coordinate_time,
        state.frame_token,
        source_state_id=state.epoch_manifest_id,
        frame_id=frame.frame_id,
        frame_realization_id=state.frame_realization_id,
        unit_contract_id=units.contract_id,
    )
    gravity = packet_gravity_source(packet_plan, state, frame, exchange, 2.0)
    assert bool(gravity.projection.all_active_valid)
    assert gravity.projection.geometry_lineage_id == frame.geometry.geometry_lineage_id
    assert gravity.source_state_id == state.epoch_manifest_id
    np.testing.assert_allclose(gravity.projection.energy_density, 0.5)
    np.testing.assert_allclose(gravity.projection.stress_covariant[0, 0], 0.5)


def test_hierarchy_gravity_requires_explicit_linearization_receipt():
    units, frame, _ = _setup()
    plan = DarkRadiationBoltzmannHierarchyPlan(
        jnp.asarray((0.2,)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        3,
        frame=frame,
        closure_tolerance=1.0,
    )
    state = plan.initialize(jnp.zeros(plan.shape), state_id="hierarchy-stage")
    receipt = DarkRadiationConversionReceipt(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        source_state_id="m1-stage",
        target_state_id=state.state_id,
        source_representation="multigroup-m1",
        target_representation="linear-boltzmann-hierarchy",
        operation="background-linearization-and-momentum-projection",
        differentiation_policy="differentiable-fixed-quadrature",
    )
    exchange = DarkRadiationFourForce.paired(
        jnp.zeros((4,)),
        state.conformal_time,
        state.frame_token,
        source_state_id=state.state_id,
        frame_id=state.frame_id,
        frame_realization_id=state.frame_realization_id,
        unit_contract_id=units.contract_id,
    )
    gravity = hierarchy_gravity_source(
        plan,
        state,
        receipt,
        frame,
        exchange,
        jnp.asarray(2.0),
        jnp.asarray((1.0,)),
        jnp.asarray(((1.0, 0.0, 0.0),)),
    )
    assert bool(gravity.projection.all_active_valid)
    np.testing.assert_allclose(gravity.projection.energy_density, 2.0)
    np.testing.assert_allclose(
        gravity.projection.stress_covariant, 2.0 * jnp.eye(3) / 3.0
    )


def test_vet_shadow_tensor_couples_to_exact_gravity_snapshot():
    _, frame, _ = _setup()
    plan = DarkRadiationVETPlan(
        jnp.asarray(
            (
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
            )
        ),
        jnp.ones((4,)),
        maximum_iterations=24,
        residual_tolerance=1.0e-6,
    )
    result = plan.formal_solve(
        jnp.asarray((3.0, 0.1, 0.1, 0.1)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(0.25),
        source_state_id="vet-shadow",
    )
    projection = vet_stress_energy_projection(
        result,
        jnp.asarray(2.0),
        jnp.asarray((0.5, 0.0, 0.0)),
        frame,
        source_state_id="vet-shadow",
    )
    assert bool(projection.all_active_valid)
    assert projection.snapshot_token == frame.frame_token
    np.testing.assert_allclose(
        jnp.trace(projection.stress_covariant),
        projection.energy_density,
        rtol=1.0e-6,
    )


def test_distributed_m1_checkpoint_and_profile_products_are_separate(tmp_path):
    _, frame, _ = _setup()
    state = jnp.asarray(((1.0, 0.0, 0.0, 0.0), (2.0, 0.5, 0.0, 0.0)))
    checkpoint = DarkRadiationM1Checkpoint(
        state,
        jnp.zeros_like(state),
        jnp.asarray((100, 101), dtype=jnp.int64),
        jnp.asarray((0, 1), dtype=jnp.int32),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray(3),
        jnp.zeros_like(state),
        jnp.asarray(8),
        frame=frame,
        system_id="m1-system",
        topology_id="amr-topology-4",
        partition_id="two-shards",
        epoch_manifest_id="8" * 64,
    )
    path = write_dark_radiation_m1_checkpoint(tmp_path / "m1.npz", checkpoint)
    restored = read_dark_radiation_m1_checkpoint(path, checkpoint)
    assert bool(eqx.tree_equal(restored, checkpoint))
    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert int(restored.collective_count) == 3

    profile = packet_transport_profile()
    assert profile.checkpoint_product != profile.output_product
    assert "checkpoint-restart-identity" in profile.production_evidence
    assert "event-capacity-exhaustion" in profile.refusal_conditions

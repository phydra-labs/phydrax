#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.solver._dark_radiation_packets import (
    DarkRadiationInteractionKind,
    DarkRadiationPacketPlan,
    read_dark_radiation_packet_checkpoint,
    write_dark_radiation_packet_checkpoint,
)
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan


def _units():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1)
    return RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )


def _frame(units, *, time, scale_factor, snapshot):
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(snapshot),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-zone",
        geometry_lineage_id="flat-flrw",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="packet-observer",
    )
    return LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(time),
        jnp.asarray(scale_factor),
        observer_id="packet-observer",
        orientation_id="right-handed-future",
    )


def _epoch_plan(*, capacity=2, event_capacity=2):
    return DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=event_capacity,
        product_capacity=1,
        radiation_capacity=capacity,
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


def _plan_and_state(*, capacity=2, event_capacity=2):
    units = _units()
    frame = _frame(units, time=0.0, scale_factor=1.0, snapshot=1)
    plan = DarkRadiationPacketPlan(
        capacity,
        event_capacity,
        units,
        jnp.asarray((0.1, 1.5, 3.0)),
        jnp.asarray((-10.0, -10.0, -10.0)),
        jnp.asarray((10.0, 10.0, 10.0)),
        epoch_plan=_epoch_plan(capacity=capacity, event_capacity=event_capacity),
    )
    state = plan.empty(frame, epoch_manifest_id="a" * 64)
    admission = plan.admit(
        state,
        jnp.asarray((101,), dtype=jnp.int64),
        jnp.asarray((7,), dtype=jnp.int32),
        jnp.asarray(((11, 12),), dtype=jnp.int64),
        jnp.asarray((31,), dtype=jnp.int64),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray(((0.0, 0.0, 0.0),)),
        jnp.asarray(((1.0, 1.0, 0.0, 0.0),)),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((1.0, 0.2, 0.0, 0.0),)),
        jnp.asarray((0.1,)),
        jr.split(jr.key(19), 1),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((2.0,)),
        jnp.asarray((True,)),
    )
    assert bool(admission.successful)
    return units, frame, plan, admission.accepted_state


def test_minkowski_and_flrw_scattering_propagation_redshift_and_polarization():
    units, frame0, plan, state = _plan_and_state()
    frame1 = _frame(units, time=1.0, scale_factor=2.0, snapshot=2)
    mueller = jnp.diag(jnp.asarray((1.0, 0.5, 0.5, 0.5)))

    result = eqx.filter_jit(plan.advance)(
        state,
        frame0,
        frame1,
        jnp.zeros((2,)),
        jnp.asarray((1.0, 0.0)),
        mueller,
        next_epoch_manifest_id="b" * 64,
        end_frame_realization_id=frame1.realization_id(),
    )

    assert bool(result.successful)
    assert int(result.events.interaction_kind[0]) == int(
        DarkRadiationInteractionKind.SCATTERING
    )
    assert int(result.events.parent_packet_ids[0]) == 101
    assert int(result.events.child_packet_ids[0]) == 101
    np.testing.assert_allclose(
        result.accepted_state.tetrad_four_momentum[0, 0], 0.5, rtol=1e-6
    )
    np.testing.assert_allclose(result.accepted_state.stokes[0, 1], 0.1, rtol=1e-6)
    assert result.accepted_state.comoving_position[0, 0] > 0.0
    assert bool(result.exchange.exact_opposite)
    np.testing.assert_array_equal(
        result.exchange.radiation_four_force, -result.exchange.matter_four_force
    )
    assert bool(plan.valid(result.accepted_state))


def test_absorption_event_four_force_and_event_capacity_rollback():
    units, frame0, plan, state = _plan_and_state(event_capacity=1)
    frame1 = _frame(units, time=0.5, scale_factor=1.0, snapshot=2)
    absorbed = plan.advance(
        state,
        frame0,
        frame1,
        jnp.asarray((2.0, 0.0)),
        jnp.zeros((2,)),
        jnp.eye(4),
        next_epoch_manifest_id="c" * 64,
        end_frame_realization_id=frame1.realization_id(),
    )
    assert bool(absorbed.successful)
    assert int(absorbed.events.interaction_kind[0]) == int(
        DarkRadiationInteractionKind.ABSORPTION
    )
    assert int(absorbed.events.child_packet_ids[0]) == -1
    assert not bool(absorbed.accepted_state.active_mask[0])
    np.testing.assert_allclose(absorbed.exchange.matter_four_force[0], 4.0, rtol=1e-6)
    np.testing.assert_allclose(
        absorbed.events.four_momentum_to_matter[0],
        absorbed.exchange.matter_four_force * 0.5,
        rtol=1.0e-6,
    )

    second = plan.admit(
        state,
        jnp.asarray((102,), dtype=jnp.int64),
        jnp.asarray((7,), dtype=jnp.int32),
        jnp.asarray(((11, 12),), dtype=jnp.int64),
        jnp.asarray((32,), dtype=jnp.int64),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray(((0.0, 0.0, 0.0),)),
        jnp.asarray(((1.0, 1.0, 0.0, 0.0),)),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((1.0, 0.0, 0.0, 0.0),)),
        jnp.asarray((0.1,)),
        jnp.asarray(((29, 31),), dtype=jnp.uint32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        jnp.asarray((True,)),
    ).accepted_state
    overflow = plan.advance(
        second,
        frame0,
        frame1,
        jnp.asarray((2.0, 2.0)),
        jnp.zeros((2,)),
        jnp.eye(4),
        next_epoch_manifest_id="d" * 64,
        end_frame_realization_id=frame1.realization_id(),
    )
    assert not bool(overflow.successful)
    assert bool(overflow.evidence.rolled_back)
    assert bool(eqx.tree_equal(overflow.accepted_state, second))
    assert not bool(jnp.any(overflow.events.mask))


def test_capacity_and_exact_checkpoint_restart_preserve_rng_and_epoch(tmp_path):
    _, _, plan, state = _plan_and_state(capacity=1)
    full = plan.admit(
        state,
        jnp.asarray((202,), dtype=jnp.int64),
        jnp.asarray((8,), dtype=jnp.int32),
        jnp.asarray(((21, 22),), dtype=jnp.int64),
        jnp.asarray((41,), dtype=jnp.int64),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.zeros((1, 3)),
        jnp.asarray(((1.0, 1.0, 0.0, 0.0),)),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((1.0, 0.0, 0.0, 0.0),)),
        jnp.asarray((0.2,)),
        jnp.asarray(((3, 4),), dtype=jnp.uint32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1.0,)),
        jnp.asarray((True,)),
    )
    assert not bool(full.successful)
    assert bool(full.evidence.rolled_back)

    path = write_dark_radiation_packet_checkpoint(tmp_path / "packets.npz", plan, state)
    restored = read_dark_radiation_packet_checkpoint(path, plan, state)
    assert bool(eqx.tree_equal(restored, state))
    np.testing.assert_array_equal(restored.rng_keys, state.rng_keys)
    assert restored.epoch_manifest_id == state.epoch_manifest_id
    assert restored.epoch_plan_id == state.epoch_plan_id

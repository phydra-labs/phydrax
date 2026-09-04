from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax.applications.solid_mechanics._rod_dynamics import (
    prepare_rod,
    RodPlan,
    RodState,
)
from phydrax.applications.solid_mechanics._rod_loads import (
    ReducedRodLoadBundle,
    RodLoad,
    RodLoadLedger,
)


def _spatial_rod():
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 2.0))),
            jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
            jnp.ones((3,)),
            jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
            jnp.broadcast_to(jnp.diag(jnp.asarray((4.0, 5.0, 6.0))), (2, 3, 3)),
            jnp.broadcast_to(jnp.diag(jnp.asarray((2.0, 3.0, 4.0))), (1, 3, 3)),
        )
    )


def test_load_ledger_preserves_source_frame_unit_and_channel_semantics():
    contact = RodLoad(
        jnp.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.0))),
        jnp.asarray(((0.0, 0.0, 0.5), (0.0, 0.0, 0.0))),
        source_id="contact:tip",
        power_channel="contact",
        force_unit="N",
        moment_unit="N*m",
    )
    motor = RodLoad(
        jnp.zeros((3, 3)),
        jnp.asarray(((0.3, -0.1, 0.2), (-0.2, 0.4, 0.1))),
        source_id="motor:distributed",
        power_channel="actuation",
        force_unit="N",
        moment_unit="N*m",
    )
    ledger = RodLoadLedger((contact, motor))

    assert ledger.source_ids == ("contact:tip", "motor:distributed")
    assert ledger.channel_names == ("contact", "actuation")
    assert ledger.force_frame == "world"
    assert ledger.moment_frame == "material"
    assert ledger.force_unit == "N"
    assert ledger.moment_unit == "N*m"


def test_load_power_evidence_uses_native_dual_pairing_and_named_channels():
    rod = _spatial_rod()
    linear_velocity = jnp.asarray(((0.1, -0.2, 0.3), (0.4, 0.2, -0.1), (-0.3, 0.5, 0.6)))
    body_angular_velocity = jnp.asarray(((0.2, 0.4, -0.1), (-0.3, 0.1, 0.5)))
    state = RodState(
        rod.plan.rest_positions,
        linear_velocity,
        rod.rest_orientations,
        body_angular_velocity,
    )
    first = RodLoad(
        jnp.asarray(((1.0, 0.0, 2.0), (0.5, -1.0, 0.0), (0.0, 0.4, -0.2))),
        jnp.asarray(((0.3, 0.1, -0.4), (0.0, -0.2, 0.5))),
        source_id="fluid",
        power_channel="environment",
    )
    second = RodLoad(
        jnp.asarray(((0.0, 0.0, 0.0), (-0.2, 0.1, 0.4), (0.7, 0.0, 0.0))),
        jnp.asarray(((0.1, -0.3, 0.2), (0.4, 0.1, -0.1))),
        source_id="contact",
        power_channel="environment",
    )
    third = RodLoad(
        jnp.zeros((3, 3)),
        jnp.asarray(((0.5, 0.0, 0.0), (-0.2, 0.0, 0.0))),
        source_id="tendon",
        power_channel="actuation",
    )
    ledger = RodLoadLedger((first, second, third))

    evidence = ledger.power_from_state(rod, state)
    direct = sum(
        jnp.sum(load.forces * linear_velocity)
        + jnp.sum(load.moments * body_angular_velocity)
        for load in ledger.loads
    )
    total_effort = ledger.total_effort(rod)

    assert evidence.valid
    assert evidence.total_power == pytest.approx(direct)
    assert evidence.paired_power == pytest.approx(
        rod.effort_space.pair(total_effort, rod.velocity_from_state(state))
    )
    assert evidence.power_for_channel("environment") == pytest.approx(
        evidence.source_power[0] + evidence.source_power[1]
    )
    assert evidence.power_for_channel("actuation") == pytest.approx(
        evidence.source_power[2]
    )
    assert evidence.absolute_pairing_error == pytest.approx(0.0, abs=2.0e-6)


def test_spatial_load_rejects_world_or_quaternion_storage_moments():
    with pytest.raises(ValueError, match="material frame"):
        RodLoad(
            jnp.zeros((3, 3)),
            jnp.zeros((2, 3)),
            source_id="bad-frame",
            power_channel="external",
            moment_frame="world",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Rod moments"):
        RodLoad(
            jnp.zeros((3, 3)),
            jnp.zeros((2, 4)),
            source_id="bad-shape",
            power_channel="external",
        )


def test_reduced_bundle_keeps_native_source_order_and_channel_aggregation():
    first = RodLoad(
        jnp.zeros((3, 3)),
        jnp.zeros((2, 3)),
        source_id="fluid",
        power_channel="environment",
    )
    second = RodLoad(
        jnp.zeros((3, 3)),
        jnp.zeros((2, 3)),
        source_id="motor",
        power_channel="actuation",
    )
    ledger = RodLoadLedger((first, second))
    source_efforts = jnp.asarray(((1.0, -2.0, 0.5), (-0.3, 0.4, 0.2)))
    bundle = ReducedRodLoadBundle(ledger, source_efforts, "reduction:test")

    assert bundle.source_ids == ledger.source_ids
    assert jnp.array_equal(bundle.effort_for_source("motor"), source_efforts[1])
    assert jnp.array_equal(bundle.effort_for_channel("environment"), source_efforts[0])
    assert jnp.allclose(bundle.total_effort(), jnp.sum(source_efforts, axis=0))

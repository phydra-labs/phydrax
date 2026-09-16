import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cosmology._dark_radiation import (
    DarkRadiationLedgerPlan,
    DarkRadiationPacket,
    DarkRadiationStatus,
)


def _packet(packet_id=41):
    return DarkRadiationPacket(
        packet_id=jnp.asarray(packet_id, dtype=jnp.int64),
        species_id=jnp.asarray(7, dtype=jnp.int32),
        source_event_id=jnp.asarray(3001, dtype=jnp.int64),
        parent_ids=jnp.asarray((101, 211), dtype=jnp.int64),
        physical_energy=jnp.asarray(1.0),
        physical_momentum=jnp.asarray((0.25, -0.125, 0.0)),
        comoving_position=jnp.asarray((0.2, 0.3, 0.4)),
        emission_scale_factor=jnp.asarray(0.75),
    )


def test_packet_export_preserves_four_momentum_identity_and_time_level():
    plan = DarkRadiationLedgerPlan(2, speed_of_light=1.0)
    initial = plan.empty()
    packet = _packet()
    retained = jnp.asarray((4.0, 0.5, 0.25, -0.125))
    source = retained + jnp.concatenate(
        (packet.physical_energy[None], packet.physical_momentum)
    )

    result = eqx.filter_jit(plan.export)(initial, packet, source, retained)

    assert bool(result.successful)
    assert int(result.evidence.status) == int(DarkRadiationStatus.SUCCESS)
    assert bool(result.evidence.four_momentum_balanced)
    assert bool(
        jnp.all(
            jnp.abs(result.evidence.four_momentum_defect)
            <= result.evidence.four_momentum_tolerance
        )
    )
    np.testing.assert_array_equal(result.accepted_ledger.packet_ids, (41, -1))
    np.testing.assert_array_equal(result.accepted_ledger.species_ids, (7, -1))
    np.testing.assert_array_equal(result.accepted_ledger.source_event_ids, (3001, -1))
    np.testing.assert_array_equal(result.accepted_ledger.parent_ids[0], (101, 211))
    np.testing.assert_array_equal(
        result.accepted_ledger.physical_momentum[0], packet.physical_momentum
    )
    assert float(result.accepted_ledger.physical_energy[0]) == 1.0
    assert float(result.accepted_ledger.emission_scale_factors[0]) == 0.75
    assert bool(result.evidence.ledger_unchanged_on_failure)


def test_capacity_and_four_momentum_failures_roll_back_every_ledger_leaf():
    plan = DarkRadiationLedgerPlan(1)
    initial = plan.empty()
    first_packet = _packet(10)
    retained = jnp.asarray((2.0, 0.0, 0.0, 0.0))
    source = retained + jnp.concatenate(
        (first_packet.physical_energy[None], first_packet.physical_momentum)
    )
    first = plan.export(initial, first_packet, source, retained)
    assert bool(first.successful)

    full = plan.export(first.accepted_ledger, _packet(11), source, retained)
    assert not bool(full.successful)
    assert int(full.evidence.status) == int(DarkRadiationStatus.CAPACITY_EXHAUSTED)
    assert bool(eqx.tree_equal(full.accepted_ledger, first.accepted_ledger))
    assert bool(full.evidence.ledger_unchanged_on_failure)

    imbalance = plan.export(
        initial,
        first_packet,
        source.at[0].add(0.5),
        retained,
    )
    assert not bool(imbalance.successful)
    assert int(imbalance.evidence.status) == int(
        DarkRadiationStatus.FOUR_MOMENTUM_IMBALANCE
    )
    assert bool(eqx.tree_equal(imbalance.accepted_ledger, initial))
    assert bool(imbalance.evidence.ledger_unchanged_on_failure)


def test_runtime_dtype_tolerance_and_causal_four_vectors_fail_closed():
    plan = DarkRadiationLedgerPlan(1)
    ledger = plan.empty(dtype=jnp.float32)
    packet = DarkRadiationPacket(
        51,
        3,
        9001,
        (101, 211),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray((0.01, 0.0, 0.0), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
    )
    retained = jnp.asarray((0.5, 0.02, 0.0, 0.0), dtype=jnp.float32)
    source = retained + jnp.concatenate(
        (packet.physical_energy[None], packet.physical_momentum)
    )
    accepted = plan.export(ledger, packet, source, retained)

    assert bool(accepted.successful)
    assert accepted.evidence.four_momentum_tolerance.dtype == jnp.float32
    assert bool(
        jnp.all(
            accepted.evidence.four_momentum_tolerance
            >= 64.0 * jnp.finfo(jnp.float32).eps * accepted.evidence.four_momentum_scale
        )
    )

    impossible = plan.export(
        ledger,
        DarkRadiationPacket(
            52,
            3,
            9002,
            (101, 211),
            jnp.asarray(2.0, dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
        ),
        jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
        jnp.asarray((-1.0, 0.0, 0.0, 0.0), dtype=jnp.float32),
    )
    assert not bool(impossible.successful)
    assert bool(impossible.evidence.source_future_directed)
    assert not bool(impossible.evidence.retained_future_directed)
    assert bool(eqx.tree_equal(impossible.accepted_ledger, ledger))

    with pytest.raises(TypeError, match="identical floating dtype"):
        plan.export(plan.empty(), packet, source, retained)

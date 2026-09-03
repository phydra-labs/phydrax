from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


mn = phx.applications.solid_mechanics.member_network


def _epochs(count: int):
    epoch = jnp.zeros((count,), dtype=jnp.int32)
    return {
        "accepted": jnp.ones((count,), dtype=bool),
        "topology_epoch": epoch,
        "contact_epoch": epoch,
        "fracture_epoch": epoch,
        "mode_epoch": epoch,
        "unilateral_epoch": epoch,
    }


def test_conservative_and_damped_accepted_histories_close_independent_terms():
    conservative = mn.member_energy_work_evidence(
        jnp.asarray((0.5, 0.25, 0.0)),
        jnp.asarray((0.5, 0.75, 1.0)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        **_epochs(3),
    )
    assert conservative.available
    assert conservative.balanced
    assert jnp.allclose(conservative.algorithmic_defect, 0.0)

    times = jnp.asarray((0.0, 0.2, 0.4))
    kinetic = 0.5 * jnp.exp(-2.0 * times)
    exact_damping = kinetic[:-1] - kinetic[1:]
    damped = mn.member_energy_work_evidence(
        kinetic,
        jnp.zeros_like(kinetic),
        exact_damping,
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        tolerance=1.0e-7,
        **_epochs(3),
    )
    assert damped.available
    assert damped.balanced
    assert jnp.allclose(damped.damping_work, exact_damping)

    inelastic = mn.member_energy_work_evidence(
        jnp.asarray((1.0, 0.0)),
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.ones((1,)),
        jnp.zeros((1,)),
        **_epochs(2),
    )
    assert inelastic.balanced
    assert inelastic.material_contact_work[0] == pytest.approx(1.0)


def test_energy_ledger_detects_wrong_external_work_sign_and_refines():
    correct = mn.member_energy_work_evidence(
        jnp.asarray((0.0, 1.0)),
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        jnp.ones((1,)),
        **_epochs(2),
    )
    wrong_sign = mn.member_energy_work_evidence(
        jnp.asarray((0.0, 1.0)),
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        -jnp.ones((1,)),
        **_epochs(2),
    )
    assert correct.balanced
    assert not wrong_sign.balanced
    assert wrong_sign.algorithmic_defect[0] == pytest.approx(2.0)

    def left_rule(step):
        times = jnp.arange(0.0, 0.4 + 0.5 * step, step)
        kinetic = 0.5 * jnp.exp(-2.0 * times)
        damping = step * jnp.exp(-2.0 * times[:-1])
        return mn.member_energy_work_evidence(
            kinetic,
            jnp.zeros_like(kinetic),
            damping,
            jnp.zeros_like(damping),
            jnp.zeros_like(damping),
            tolerance=0.0,
            **_epochs(times.size),
        )

    coarse = left_rule(0.2)
    fine = left_rule(0.1)
    assert fine.maximum_relative_defect < coarse.maximum_relative_defect


@pytest.mark.parametrize(
    "epoch_name",
    (
        "topology_epoch",
        "contact_epoch",
        "fracture_epoch",
        "unilateral_epoch",
        "mode_epoch",
    ),
)
def test_energy_evidence_is_unavailable_across_epoch_switches(epoch_name):
    epochs = _epochs(3)
    epochs[epoch_name] = jnp.asarray((0, 0, 1), dtype=jnp.int32)
    evidence = mn.member_energy_work_evidence(
        jnp.asarray((1.0, 1.0, 1.0)),
        jnp.zeros((3,)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        **epochs,
    )
    assert not evidence.available
    assert not evidence.balanced
    assert not evidence.interval_available[-1]
    assert not evidence.epoch_consistent[-1]


def test_energy_evidence_rejects_unaccepted_history_samples():
    epochs = _epochs(2)
    epochs["accepted"] = jnp.asarray((True, False))
    evidence = mn.member_energy_work_evidence(
        jnp.ones((2,)),
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        **epochs,
    )
    assert not evidence.available
    assert not evidence.balanced


def test_outgoing_work_requires_an_explicit_traction_velocity_port():
    port = mn.TractionVelocityPortHistory(
        jnp.ones((1, 1, 1)),
        jnp.ones((1, 1, 1)),
        jnp.ones((1,)),
        jnp.ones((1,)),
        port_id="right-boundary-outgoing",
    )
    evidence = mn.member_energy_work_evidence(
        jnp.asarray((1.0, 0.0)),
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        traction_velocity_port=port,
        **_epochs(2),
    )
    assert evidence.outgoing_port_work[0] == pytest.approx(1.0)
    assert evidence.traction_velocity_port_id == "right-boundary-outgoing"
    assert evidence.balanced

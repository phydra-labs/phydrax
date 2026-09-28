#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


two_phase_api = phx.applications.two_phase_flow
N = 16


@pytest.fixture(scope="module")
def two_phase() -> two_phase_api.PreparedIncompressibleTwoPhaseVOF:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(N, periodic=True),
            phx.discretization.UniformCellAxisSpec(N, periodic=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=1.0,
        liquid_viscosity=0.0,
        gas_viscosity=0.0,
    )
    return two_phase_api.IncompressibleTwoPhaseVOFPlan(discretization, material).prepare()


@pytest.fixture(scope="module")
def plan(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> two_phase_api.BubbleComponentPlan:
    return two_phase_api.BubbleComponentPlan(
        two_phase,
        component_capacity=8,
        maximum_rounds=64,
        pair_capacity=16,
        vent_sides=((1, "upper"),),
    )


def _coordinates() -> tuple[np.ndarray, np.ndarray]:
    centers = (np.arange(N) + 0.5) / N
    return np.meshgrid(centers, centers, indexing="ij")


def _field(*disks: tuple[float, float, float], atmosphere: bool = True) -> np.ndarray:
    x, y = _coordinates()
    alpha = np.ones((N, N))
    for cx, cy, radius in disks:
        alpha[(x - cx) ** 2 + (y - cy) ** 2 < radius**2] = 0.0
    if atmosphere:
        alpha[:, -2:] = 0.0
    return alpha


BUBBLES = ((0.3, 0.4, 0.12), (0.7, 0.4, 0.12))


def test_initial_identities_separate_atmosphere_from_bubbles(
    plan: two_phase_api.BubbleComponentPlan,
) -> None:
    alpha = _field(*BUBBLES)
    state = plan.initial_state(jnp.asarray(alpha))
    ids = np.asarray(state.slot_ids)
    volume = np.asarray(state.labels.volume)
    assert ids[:3].tolist() == [two_phase_api.ATMOSPHERE_ID, 1, 2]
    assert np.all(ids[3:] == -1)
    assert int(state.next_id) == 3
    np.testing.assert_allclose(volume[0], 2.0 / N, rtol=1e-14)
    np.testing.assert_allclose(np.sum(volume), np.sum(1.0 - alpha) / N**2, rtol=1e-14)
    assert np.asarray(state.labels.atmosphere)[:3].tolist() == [True, False, False]


def test_moving_bubbles_keep_identity(plan: two_phase_api.BubbleComponentPlan) -> None:
    state = plan.initial_state(jnp.asarray(_field(*BUBBLES)))
    moved = _field((0.32, 0.4, 0.12), (0.7, 0.44, 0.12))
    proposal = plan.propose(state, jnp.asarray(moved))
    assert int(proposal.evidence.status) == two_phase_api.BubbleIdentityStatus.CONTINUED
    assert not bool(proposal.evidence.topology_changed)
    assert bool(proposal.evidence.derivative_available)
    assert np.asarray(proposal.slot_ids)[:3].tolist() == [0, 1, 2]
    assert int(proposal.commit(state).epoch) == 0


def test_merge_split_and_lineage_journal(plan: two_phase_api.BubbleComponentPlan) -> None:
    state = plan.initial_state(jnp.asarray(_field(*BUBBLES)))
    x, y = _coordinates()
    bridged = _field(*BUBBLES)
    bridged[(y > 0.35) & (y < 0.45) & (x > 0.3) & (x < 0.7)] = 0.0
    merge = plan.propose(state, jnp.asarray(bridged))
    assert bool(merge.evidence.topology_changed)
    assert not bool(merge.evidence.derivative_available)
    records = two_phase_api.transition_records(merge.event, state)
    assert [(record.kind, record.parent_ids, record.child_ids) for record in records] == [
        ("merge", (1, 2), (3,))
    ]
    # Both parents lie entirely inside the merged gas: the observed overlaps
    # are the parents' gas volumes; the bridge is new gas without a parent.
    np.testing.assert_allclose(
        [volume for _, _, volume in records[0].overlaps],
        records[0].parent_volumes,
        rtol=1e-14,
    )
    assert records[0].child_volumes[0] > sum(records[0].parent_volumes)
    merged = merge.commit(state)
    assert int(merged.epoch) == 1
    journal = two_phase_api.BubbleTransitionJournal().extend(records)

    split = plan.propose(merged, jnp.asarray(_field(*BUBBLES)))
    split_records = two_phase_api.transition_records(split.event, merged)
    assert [
        (record.kind, record.parent_ids, record.child_ids) for record in split_records
    ] == [("split", (3,), (4, 5))]
    journal = journal.extend(split_records)
    assert journal.lineage(5) == (1, 2, 3)
    replay = two_phase_api.BubbleTransitionJournal().extend(records).extend(split_records)
    assert replay.journal_id == journal.journal_id


def test_create_vanish_vent_and_entrain_events(
    plan: two_phase_api.BubbleComponentPlan,
) -> None:
    state = plan.initial_state(jnp.asarray(_field(*BUBBLES)))
    x, y = _coordinates()

    def kinds(
        previous: two_phase_api.BubbleComponentState, alpha: np.ndarray
    ) -> list[tuple[str, tuple[int, ...], tuple[int, ...]]]:
        proposal = plan.propose(previous, jnp.asarray(alpha))
        return [
            (record.kind, record.parent_ids, record.child_ids)
            for record in two_phase_api.transition_records(proposal.event, previous)
        ]

    assert kinds(state, _field(*BUBBLES, (0.5, 0.1, 0.06))) == [("create", (), (3,))]
    assert kinds(state, _field(BUBBLES[0])) == [("vanish", (2,), ())]
    vented = _field(*BUBBLES)
    vented[(x > 0.65) & (x < 0.75) & (y > 0.4)] = 0.0
    assert kinds(state, vented) == [("vent", (2,), (0,))]

    thick = _field(*BUBBLES)
    thick[:, -4:] = 0.0
    deep = plan.initial_state(jnp.asarray(thick))
    entrained = _field(*BUBBLES)
    entrained[:, -2:] = 1.0
    entrained[:, -1:] = 0.0
    entrained[(x > 0.1) & (x < 0.3) & (y > 0.75) & (y < 0.82)] = 0.0
    assert kinds(deep, entrained) == [("entrain", (0,), (3,))]


def test_colors_keep_touching_bubbles_apart(
    plan: two_phase_api.BubbleComponentPlan,
) -> None:
    touching = _field((0.38, 0.4, 0.12), (0.62, 0.4, 0.12), atmosphere=False)
    x, _ = _coordinates()
    gas = touching < 1.0
    color = np.where(gas & (x > 0.5), 1, 0)
    assert int(plan.label(jnp.asarray(touching)).labeling.count) == 1
    colored = plan.label(jnp.asarray(touching), color=jnp.asarray(color))
    assert int(colored.labeling.count) == 2


def test_component_capacity_overflow_is_refused(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> None:
    plan = two_phase_api.BubbleComponentPlan(
        two_phase, component_capacity=2, maximum_rounds=64, pair_capacity=8
    )
    state = plan.initial_state(jnp.asarray(_field((0.3, 0.3, 0.1), atmosphere=False)))
    proposal = plan.propose(
        state,
        jnp.asarray(
            _field((0.2, 0.2, 0.08), (0.7, 0.2, 0.08), (0.5, 0.7, 0.08), atmosphere=False)
        ),
    )
    assert (
        int(proposal.evidence.status)
        == two_phase_api.BubbleIdentityStatus.LABELING_FAILED
    )
    assert bool(proposal.evidence.overflow)


def test_vent_sides_must_be_nonperiodic(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> None:
    with pytest.raises(ValueError, match="nonperiodic"):
        two_phase_api.BubbleComponentPlan(
            two_phase,
            component_capacity=4,
            maximum_rounds=8,
            pair_capacity=4,
            vent_sides=((0, "lower"),),
        )

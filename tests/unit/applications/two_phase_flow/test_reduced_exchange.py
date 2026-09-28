#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.bubble_dynamics as bd
from phydrax.applications import two_phase_flow as tpf


def _state(
    bubble_ids: tuple[int, ...],
    amounts: tuple[float, ...],
    energies: tuple[float, ...],
    volumes: tuple[float, ...],
    *,
    epoch: int,
) -> tpf.BubbleCompartmentState:
    count = len(bubble_ids)
    centroids = np.column_stack(
        (
            np.arange(1, count + 1, dtype=np.float64),
            np.arange(11, 11 + count, dtype=np.float64),
            np.arange(21, 21 + count, dtype=np.float64),
        )
    )
    return tpf.BubbleCompartmentState(
        bubble_id=jnp.asarray(bubble_ids, dtype=jnp.int32),
        amount=jnp.asarray(amounts, dtype=jnp.float64),
        internal_energy=jnp.asarray(energies, dtype=jnp.float64),
        internal=jnp.asarray(
            np.column_stack((np.asarray(amounts), np.asarray(energies))),
            dtype=jnp.float64,
        ),
        volume=jnp.asarray(volumes, dtype=jnp.float64),
        pressure=jnp.full((count,), 101325.0, dtype=jnp.float64),
        centroid=jnp.asarray(centroids, dtype=jnp.float64),
        epoch=jnp.asarray(epoch, dtype=jnp.int32),
    )


def _evaluation(count: int, *, supported: bool = True) -> tpf.BubbleCompartmentEvaluation:
    return tpf.BubbleCompartmentEvaluation(
        pressure=jnp.full((count,), 101325.0, dtype=jnp.float64),
        temperature=jnp.full((count,), 300.0, dtype=jnp.float64),
        compliance=jnp.full((count,), 1.0e-9, dtype=jnp.float64),
        admissible=jnp.full((count,), supported, dtype=jnp.bool_),
    )


def _transition(
    epoch: int,
    kind: tpf.BubbleTransitionKind,
    parents: tuple[int, ...],
    children: tuple[int, ...],
    parent_volumes: tuple[float, ...],
    child_volumes: tuple[float, ...],
) -> tpf.BubbleTransitionRecord:
    return tpf.BubbleTransitionRecord(
        epoch,
        kind,
        parent_ids=parents,
        child_ids=children,
        parent_volumes=parent_volumes,
        child_volumes=child_volumes,
        child_slots=tuple(range(len(children))),
        overlaps=(),
    )


def test_resolved_evidence_preserves_content_and_never_fabricates_missing_vectors() -> (
    None
):
    transition = _transition(3, "vanish", (7,), (), (8.0e-15,), ())
    state = _state((7,), (2.5e-12,), (7.5e-7,), (8.0e-15,), epoch=2)
    source = tuple(np.asarray(leaf).copy() for leaf in jax.tree.leaves(state))

    incomplete = tpf.resolved_bubble_evidence_records(
        transition,
        state,
        time=1.25,
        source_realization_id="resolved-run-4",
        law_id="caloric-gas",
    )[0]
    assert incomplete.bubble_id == 7
    assert incomplete.parent_ids == (7,)
    assert incomplete.child_ids == ()
    assert incomplete.gas_amount == 2.5e-12
    assert incomplete.internal_energy == 7.5e-7
    assert incomplete.translational_momentum is None
    assert incomplete.liquid_impulse is None
    assert incomplete.ambient_state is None
    assert incomplete.status is tpf.BubbleExchangeStatus.HANDOFF_INCOMPLETE
    assert (
        incomplete.missing_invariants & tpf.BubbleExchangeInvariant.TRANSLATIONAL_MOMENTUM
    )
    assert incomplete.missing_invariants & tpf.BubbleExchangeInvariant.LIQUID_IMPULSE
    assert incomplete.missing_invariants & tpf.BubbleExchangeInvariant.AMBIENT_STATE
    assert not incomplete.handoff_ready

    complete_evidence = tpf.resolved_bubble_evidence_records(
        transition,
        state,
        time=1.25,
        source_realization_id="resolved-run-4",
        law_id="caloric-gas",
        evaluation=_evaluation(1),
        environment=bd.BubbleEnvironment(101325.0, 300.0),
        translational_momentum={7: np.zeros(3)},
        liquid_impulse={7: np.array((1.0e-9, -2.0e-9, 3.0e-9))},
    )[0]
    assert complete_evidence.gas_amount == incomplete.gas_amount
    assert complete_evidence.internal_energy == incomplete.internal_energy
    assert complete_evidence.translational_momentum == (0.0, 0.0, 0.0)
    assert (
        complete_evidence.missing_invariants
        == tpf.BubbleExchangeInvariant.FLOW_FIELD_REMOVAL
    )
    assert complete_evidence.status is tpf.BubbleExchangeStatus.EVIDENCE_ONLY
    assert not complete_evidence.handoff_ready

    replay = tpf.resolved_bubble_evidence_records(
        transition,
        state,
        time=1.25,
        source_realization_id="resolved-run-4",
        law_id="caloric-gas",
        evaluation=_evaluation(1),
        environment=bd.BubbleEnvironment(101325.0, 300.0),
        liquid_impulse={7: np.array((1.0e-9, -2.0e-9, 3.0e-9))},
        translational_momentum={7: np.zeros(3)},
    )[0]
    assert replay.record_id == complete_evidence.record_id
    for before, after in zip(source, jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(before, np.asarray(after))

    request = tpf.reduced_bubble_request(
        complete_evidence, "single-bubble", "keller-miksis"
    )
    assert request.reason == "vanish"
    assert request.bubble_ids == (7,)
    assert not request.accepted
    assert not request.handoff_ready
    assert request.missing_invariants == complete_evidence.missing_invariants


def test_merge_split_and_entrain_lineage_is_canonical_and_conservative() -> None:
    environment = bd.BubbleEnvironment(101325.0, 300.0)
    merge = _transition(
        4,
        "merge",
        (9, 3),
        (11,),
        (3.0e-15, 5.0e-15),
        (8.0e-15,),
    )
    parents = _state(
        (9, 3),
        (4.0e-12, 6.0e-12),
        (9.0e-7, 1.1e-6),
        (3.0e-15, 5.0e-15),
        epoch=3,
    )
    merged_evidence = tpf.resolved_bubble_evidence_records(
        merge,
        parents,
        time=2.0,
        source_realization_id="merge-run",
        law_id="caloric-gas",
        evaluation=_evaluation(2),
        environment=environment,
    )
    assert tuple(record.bubble_id for record in merged_evidence) == (3, 9)
    assert all(record.parent_ids == (3, 9) for record in merged_evidence)
    assert all(record.child_ids == (11,) for record in merged_evidence)
    assert sum(record.gas_amount for record in merged_evidence) == pytest.approx(1.0e-11)
    assert sum(record.internal_energy for record in merged_evidence) == pytest.approx(
        2.0e-6
    )

    split = _transition(
        5,
        "split",
        (11,),
        (15, 14),
        (8.0e-15,),
        (3.0e-15, 5.0e-15),
    )
    children = _state(
        (15, 14),
        (3.75e-12, 6.25e-12),
        (7.5e-7, 1.25e-6),
        (3.0e-15, 5.0e-15),
        epoch=5,
    )
    split_evidence = tpf.resolved_bubble_evidence_records(
        split,
        children,
        time=2.1,
        source_realization_id="split-run",
        law_id="caloric-gas",
        evaluation=_evaluation(2),
        environment=environment,
    )
    assert tuple(record.bubble_id for record in split_evidence) == (14, 15)
    assert all(record.parent_ids == (11,) for record in split_evidence)
    assert all(record.child_ids == (14, 15) for record in split_evidence)
    assert sum(record.gas_amount for record in split_evidence) == pytest.approx(1.0e-11)
    assert sum(record.internal_energy for record in split_evidence) == pytest.approx(
        2.0e-6
    )

    unsupported = tpf.resolved_bubble_evidence_records(
        split,
        children,
        time=2.1,
        source_realization_id="split-unsupported-run",
        law_id="caloric-gas",
        evaluation=_evaluation(2, supported=False),
        environment=environment,
    )[0]
    support_request = tpf.reduced_bubble_request(
        unsupported, "single-bubble", "rayleigh-plesset"
    )
    assert support_request.reason == "support_loss"

    entrain = _transition(6, "entrain", (0,), (18,), (1.0e-14,), (1.0e-14,))
    entrained = _state((18,), (8.0e-12,), (1.7e-6,), (1.0e-14,), epoch=6)
    entrained_evidence = tpf.resolved_bubble_evidence_records(
        entrain,
        entrained,
        time=2.2,
        source_realization_id="entrain-run",
        law_id="caloric-gas",
        evaluation=_evaluation(1),
        environment=environment,
    )[0]
    request = tpf.reduced_bubble_request(
        entrained_evidence, "single-bubble", "rayleigh-plesset"
    )
    assert request.reason == "entrain"
    assert request.parent_ids == (0,)
    assert request.child_ids == (18,)


def _radial_model(gas: bd.AbstractBubbleGasLaw | None = None) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        "rayleigh_plesset",
        bd.PolytropicBubbleGasLaw(1.4) if gas is None else gas,
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        bd.CleanBubbleInterfaceLaw(0.072),
        bd.BubbleEnvironment(101325.0, 293.15),
        liquid_density=998.0,
        liquid_sound_speed=1481.0,
    )


def test_reduced_overlap_and_support_failures_create_requests_only() -> None:
    group = bd.BubbleSpeciesGroup(
        _radial_model(),
        np.full(2, 10.0e-6),
        np.array(((0.0, 0.0, 0.0), (15.0e-6, 0.0, 0.0))),
        bubble_ids=(5, 2),
    )
    overlap_result = bd.solve_bubble_cloud(
        bd.BubbleCloudPlan(
            (group,), bd.ConstantPressureDrive(0.0), np.array((1.0e-7,))
        ).prepare()
    )
    source_radii = np.asarray(overlap_result.terminal_state.groups[0].radius).copy()
    overlap = tpf.reduced_bubble_failure_from_result(
        overlap_result,
        "resolved-vof",
        source_realization_id="cloud-overlap-run",
    )
    assert overlap.reason == "overlap"
    assert overlap.bubble_ids == (2, 5)
    assert overlap.source_status is bd.BubbleDynamicsStatus.OVERLAP
    assert not overlap.accepted
    assert overlap.missing_invariants & tpf.BubbleExchangeInvariant.INTERNAL_ENERGY
    assert (
        overlap.missing_invariants & tpf.BubbleExchangeInvariant.FLOW_FIELD_INITIALIZATION
    )
    assert not overlap.handoff_ready
    np.testing.assert_array_equal(
        source_radii, np.asarray(overlap_result.terminal_state.groups[0].radius)
    )

    times = np.linspace(0.0, 5.0e-6, 21)
    sampled = bd.SampledPressureDrive(times[:11], 1.0e4 * np.sin(4.0e5 * times[:11]))
    support_result = bd.solve_single_bubble(
        bd.SingleBubblePlan(_radial_model(), sampled, times).prepare(1.0e-5)
    )
    support = tpf.reduced_bubble_failure_from_result(
        support_result,
        "resolved-vof",
        source_realization_id=support_result.realization_id(),
        bubble_id=77,
    )
    assert support.reason == "support_loss"
    assert support.bubble_ids == (77,)
    assert support.source_status is bd.BubbleDynamicsStatus.SUPPORT_EXIT
    assert support.missing_invariants & tpf.BubbleExchangeInvariant.LAW_SUPPORT
    assert not support.handoff_ready


def test_invalid_compartment_invariants_are_refused() -> None:
    transition = _transition(1, "vanish", (4,), (), (1.0e-12,), ())
    invalid = _state((4,), (1.0,), (-1.0,), (1.0e-12,), epoch=0)
    with pytest.raises(ValueError, match="internal_energy"):
        tpf.resolved_bubble_evidence_records(
            transition,
            invalid,
            time=0.0,
            source_realization_id="bad-run",
            law_id="caloric-gas",
        )

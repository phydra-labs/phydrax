#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


two_phase_api = phx.applications.two_phase_flow
N = 24
SUB = 6


@pytest.fixture(scope="module")
def two_phase() -> two_phase_api.PreparedIncompressibleTwoPhaseVOF:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(N, periodic=True) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=1.0, gas_density=1.0, liquid_viscosity=0.0, gas_viscosity=0.0
    )
    return two_phase_api.IncompressibleTwoPhaseVOFPlan(discretization, material).prepare()


def _disk(cx: float, cy: float, radius: float) -> np.ndarray:
    """Subsampled gas fraction of a periodic disk."""

    centers = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(centers, centers, indexing="ij")
    offsets = ((np.arange(SUB) + 0.5) / SUB - 0.5) / N
    inside = np.zeros((N, N))
    for dx in offsets:
        for dy in offsets:
            px = (x + dx - cx + 0.5) % 1.0 - 0.5
            py = (y + dy - cy + 0.5) % 1.0 - 0.5
            inside += px**2 + py**2 < radius**2
    return inside / SUB**2


def _plans(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF, markers: int
) -> tuple[two_phase_api.MultiMarkerPlan, two_phase_api.BubbleComponentPlan]:
    return (
        two_phase_api.MultiMarkerPlan(
            two_phase,
            marker_capacity=markers,
            component_capacity=6,
            pair_capacity=16,
            proximity_radius=3,
        ),
        two_phase_api.BubbleComponentPlan(
            two_phase, component_capacity=6, maximum_rounds=64, pair_capacity=16
        ),
    )


def test_markers_follow_the_step_fluxes_and_keep_touching_bubbles_apart(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> None:
    first, second = _disk(0.37, 0.5, 0.13), _disk(0.63, 0.5, 0.13)
    alpha = 1.0 - first - second
    color = np.where(second > first, 1, 0)
    markers, identity = _plans(two_phase, 3)
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase)
    velocity = tuple(
        jnp.full(layout.shape, value)
        for layout, value in zip(
            two_phase.plan.discretization.face_layouts, (1.0, 0.4), strict=True
        )
    )
    state = method.initial_continuation(
        two_phase.initial_state(jnp.asarray(alpha), velocity)
    )
    marker_state = markers.initial_state(jnp.asarray(alpha), jnp.asarray(color))
    identities = identity.initial_state(
        jnp.asarray(alpha), color=markers.color(marker_state)
    )
    totals = np.asarray(jnp.sum(marker_state.content, axis=(1, 2)))
    journal = two_phase_api.BubbleTransitionJournal()
    step = eqx.filter_jit(method.step)
    for index in range(24):
        result = step(
            jnp.asarray(index), jnp.asarray(0.0), state, jnp.asarray(0.25 / N), None
        )
        assert bool(result.successful)
        state = result.accepted_state
        transported = markers.transport(marker_state, state.fluxes)
        assert bool(transported.successful)
        marker_state = transported.state
        proposal = identity.propose(
            identities, two_phase.alpha(state.state), color=markers.color(marker_state)
        )
        assert (
            int(proposal.evidence.status) == two_phase_api.BubbleIdentityStatus.CONTINUED
        )
        records = two_phase_api.transition_records(proposal.event, identities)
        assert records == ()
        journal = journal.extend(records)
        identities = proposal.commit(identities)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(marker_state.content, axis=(1, 2))), totals, rtol=1e-12
    )
    gas = two_phase.plan.discretization.cell_volumes * (
        1.0 - two_phase.alpha(state.state)
    )
    np.testing.assert_allclose(
        np.asarray(jnp.sum(marker_state.content, axis=0)),
        np.asarray(gas),
        atol=1e-12 / N**2,
    )
    assert sorted(np.asarray(identities.slot_ids)[:2].tolist()) == [1, 2]
    assert journal.records == ()
    assert journal.lineage(1) == ()
    assert journal.lineage(2) == ()
    assert int(identity.label(two_phase.alpha(state.state)).labeling.count) == 1


def test_recoloring_resolves_same_color_conflicts_conservatively(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> None:
    alpha = 1.0 - _disk(0.3, 0.5, 0.12) - _disk(0.62, 0.5, 0.12)
    markers, identity = _plans(two_phase, 3)
    state = markers.initial_state(jnp.asarray(alpha), jnp.zeros((N, N), dtype=jnp.int32))
    identities = identity.initial_state(jnp.asarray(alpha), color=markers.color(state))
    proximity = markers.proximity(identities.labels)
    assert bool(jnp.any(proximity.pair_active & proximity.same_color))
    result = markers.recolor(state, identities.labels, identities.slot_ids, proximity)
    assert result.status is two_phase_api.MarkerRecolorStatus.COMMITTED
    assert result.recolored == ((2, 0, 1),)
    repeated = markers.recolor(state, identities.labels, identities.slot_ids, proximity)
    assert repeated.transaction_id == result.transaction_id
    np.testing.assert_array_equal(
        np.asarray(jnp.sum(result.state.content, axis=0)),
        np.asarray(jnp.sum(state.content, axis=0)),
    )
    recolored = identity.initial_state(
        jnp.asarray(alpha), color=markers.color(result.state)
    )
    after = markers.proximity(recolored.labels)
    assert not bool(jnp.any(after.pair_active & after.same_color))
    again = markers.recolor(result.state, recolored.labels, recolored.slot_ids, after)
    assert again.status is two_phase_api.MarkerRecolorStatus.NO_CONFLICT


def test_recoloring_refuses_instead_of_aliasing(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> None:
    alpha = (
        1.0 - _disk(0.3, 0.4, 0.08) - _disk(0.55, 0.4, 0.08) - _disk(0.425, 0.63, 0.08)
    )
    markers, identity = _plans(two_phase, 2)
    state = markers.initial_state(jnp.asarray(alpha), jnp.zeros((N, N), dtype=jnp.int32))
    identities = identity.initial_state(jnp.asarray(alpha), color=markers.color(state))
    proximity = markers.proximity(identities.labels)
    result = markers.recolor(state, identities.labels, identities.slot_ids, proximity)
    assert result.status is two_phase_api.MarkerRecolorStatus.MARKER_CAPACITY_EXCEEDED
    np.testing.assert_array_equal(
        np.asarray(result.state.content), np.asarray(state.content)
    )


def test_ruptured_pair_takes_the_kept_color_and_stays_exempt(
    two_phase: two_phase_api.PreparedIncompressibleTwoPhaseVOF,
) -> None:
    first, second = _disk(0.3, 0.5, 0.12), _disk(0.62, 0.5, 0.12)
    alpha = 1.0 - first - second
    markers, identity = _plans(two_phase, 3)
    state = markers.initial_state(
        jnp.asarray(alpha), jnp.asarray(np.where(second > first, 1, 0))
    )
    identities = identity.initial_state(jnp.asarray(alpha), color=markers.color(state))
    proximity = markers.proximity(identities.labels)
    result = markers.recolor(
        state,
        identities.labels,
        identities.slot_ids,
        proximity,
        merge_pairs=((1, 2),),
    )
    assert result.recolored == ((2, 1, 0),)
    merged = identity.initial_state(jnp.asarray(alpha), color=markers.color(result.state))
    after = markers.proximity(merged.labels)
    assert bool(jnp.any(after.pair_active & after.same_color))
    kept = markers.recolor(
        result.state, merged.labels, merged.slot_ids, after, exempt_pairs=((1, 2),)
    )
    assert kept.status is two_phase_api.MarkerRecolorStatus.NO_CONFLICT

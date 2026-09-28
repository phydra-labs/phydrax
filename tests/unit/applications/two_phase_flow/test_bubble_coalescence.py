#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.two_phase_flow._coalescence import (
    film_equivalent_radius,
    FilmContactLedger,
    FilmContactObservation,
    FilmContactStatus,
    FilmContactUpdate,
    FilmDrainageCoalescencePlan,
)


SIGMA = 0.05
MU_C = 1.0e-3
MU_D = 2.0e-3
H0 = 1.0e-6
HC = 5.0e-8
RADIUS = 5.0e-4
LOAD = 1.0e-5

CASES = (
    ("immobile", "axisymmetric"),
    ("partially-mobile", "axisymmetric"),
    ("fully-mobile", "axisymmetric"),
    ("immobile", "planar"),
    ("fully-mobile", "planar"),
)


def _plan(
    regime: Any,
    geometry: Any,
    *,
    capacity: int = 2,
    id_upper_bound: int = 100,
    **rupture: float,
) -> FilmDrainageCoalescencePlan:
    partial = regime == "partially-mobile"
    return FilmDrainageCoalescencePlan(
        regime=regime,
        geometry=geometry,
        pair_capacity=capacity,
        id_upper_bound=id_upper_bound,
        continuous_viscosity=MU_C,
        surface_tension=SIGMA,
        initial_film_thickness=H0,
        dispersed_viscosity=MU_D if partial else None,
        **(rupture or {"critical_thickness": HC}),
    )


def _observation(
    pairs: Sequence[tuple[int, int] | None],
    *,
    in_contact: Sequence[bool] | None = None,
    load: float = LOAD,
    work: float = 0.0,
) -> FilmContactObservation:
    count = len(pairs)
    contact = [pair is not None for pair in pairs] if in_contact is None else in_contact
    return FilmContactObservation(
        first_id=np.asarray([-1 if p is None else p[0] for p in pairs], dtype=np.int32),
        second_id=np.asarray([-1 if p is None else p[1] for p in pairs], dtype=np.int32),
        valid=np.asarray([p is not None for p in pairs]),
        in_contact=np.asarray(contact, dtype=np.bool_),
        load=np.full(count, load),
        equivalent_radius=np.full(count, RADIUS),
        near_contact_work=np.full(count, work),
    )


@eqx.filter_jit
def _advance(
    plan: FilmDrainageCoalescencePlan,
    ledger: FilmContactLedger,
    observation: FilmContactObservation,
    step: jax.Array,
) -> FilmContactUpdate:
    return plan.advance(ledger, observation, step)


@eqx.filter_jit
def _run(
    plan: FilmDrainageCoalescencePlan,
    observation: FilmContactObservation,
    steps: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    def body(
        ledger: FilmContactLedger, step: jax.Array
    ) -> tuple[FilmContactLedger, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        update = plan.advance(ledger, observation, step)
        slot = update.observation_slots[1]
        record = (
            update.ledger.film_thickness[slot],
            update.ledger.status[slot],
            update.merge_time[slot],
            update.successful,
        )
        return update.ledger, record

    ledger = FilmContactLedger.empty(plan.pair_capacity, jnp.float64)
    return jax.lax.scan(body, ledger, steps)[1]


def _slot_of(update: FilmContactUpdate, pair: tuple[int, int]) -> int:
    ledger = update.ledger
    first = np.asarray(ledger.first_id)
    second = np.asarray(ledger.second_id)
    matches = np.flatnonzero((first == pair[0]) & (second == pair[1]))
    assert matches.size == 1
    return int(matches[0])


@pytest.mark.parametrize(("regime", "geometry"), CASES)
def test_repeated_advance_integrates_closed_form_exactly(
    regime: Any, geometry: Any
) -> None:
    plan = _plan(regime, geometry)
    drain = float(plan.drainage_time(H0, HC, LOAD, RADIUS))
    steps = np.concatenate((np.full(200, drain / 1000.0), np.full(3, 0.3 * drain)))
    thickness, status, merge_time, successful = _run(
        plan, _observation([None, (3, 7)]), jnp.asarray(steps)
    )
    elapsed = np.cumsum(steps)
    before = slice(0, -1)
    expected = np.asarray(plan.thickness_after(H0, elapsed[before], LOAD, RADIUS))
    np.testing.assert_allclose(np.asarray(thickness)[before], expected, rtol=1e-10)
    assert np.all(np.asarray(status)[before] == FilmContactStatus.DRAINING)
    assert int(status[-1]) == FilmContactStatus.MERGED
    assert np.isclose(float(thickness[-1]), HC, rtol=1e-12)
    np.testing.assert_allclose(elapsed[-2] + float(merge_time[-1]), drain, rtol=1e-10)
    assert bool(np.all(np.asarray(successful)))

    probe = 0.4 * drain
    delta = 1.0e-5 * drain
    difference = (
        plan.thickness_after(H0, probe + delta, LOAD, RADIUS)
        - plan.thickness_after(H0, probe - delta, LOAD, RADIUS)
    ) / (2.0 * delta)
    rate = plan.drainage_rate(plan.thickness_after(H0, probe, LOAD, RADIUS), LOAD, RADIUS)
    np.testing.assert_allclose(float(rate), float(difference), rtol=1e-7)
    assert float(rate) < 0.0


def test_drainage_times_match_the_published_constant_force_laws() -> None:
    inverse_squares = 1.0 / HC**2 - 1.0 / H0**2
    axisymmetric_film = math.sqrt(LOAD * RADIUS / (2.0 * math.pi * SIGMA))
    expected = {
        ("immobile", "axisymmetric"): 3.0
        * MU_C
        * LOAD
        * RADIUS**2
        / (16.0 * math.pi * SIGMA**2)
        * inverse_squares,
        ("partially-mobile", "axisymmetric"): (1.0 / HC - 1.0 / H0)
        * 3.0
        * MU_D
        * RADIUS
        * axisymmetric_film
        / (4.0 * math.sqrt(3.0) * 0.66 * SIGMA),
        ("fully-mobile", "axisymmetric"): math.log(H0 / HC)
        * 3.0
        * MU_C
        * RADIUS
        / (2.0 * SIGMA),
        ("immobile", "planar"): inverse_squares
        * MU_C
        * LOAD**2
        * RADIUS**3
        / (2.0 * SIGMA**3),
        ("fully-mobile", "planar"): math.log(H0 / HC) * 4.0 * MU_C * RADIUS / SIGMA,
    }
    for case, reference in expected.items():
        plan = _plan(*case)
        assert math.isclose(
            float(plan.drainage_time(H0, HC, LOAD, RADIUS)), reference, rel_tol=1e-12
        )


@pytest.mark.parametrize(
    ("regime", "geometry", "ratio"),
    (
        ("immobile", "axisymmetric", 2.0),
        ("partially-mobile", "axisymmetric", math.sqrt(2.0)),
        ("fully-mobile", "axisymmetric", 1.0),
        ("immobile", "planar", 4.0),
        ("fully-mobile", "planar", 1.0),
    ),
)
def test_drainage_time_force_scaling(regime: Any, geometry: Any, ratio: float) -> None:
    plan = _plan(regime, geometry)
    single = float(plan.drainage_time(H0, HC, LOAD, RADIUS))
    double = float(plan.drainage_time(H0, HC, 2.0 * LOAD, RADIUS))
    assert math.isclose(double / single, ratio, rel_tol=1e-12)


def test_equivalent_radius_and_hamaker_rupture_thickness() -> None:
    assert math.isclose(float(film_equivalent_radius(RADIUS, RADIUS)), RADIUS)
    assert math.isclose(float(film_equivalent_radius(RADIUS, jnp.inf)), 2.0 * RADIUS)
    hamaker = 1.0e-20
    plan = _plan("immobile", "axisymmetric", hamaker_constant=hamaker)
    expected = (hamaker * RADIUS / (8.0 * math.pi * SIGMA)) ** (1.0 / 3.0)
    assert math.isclose(float(plan.critical_thickness(RADIUS)), expected, rel_tol=1e-12)
    validity = plan.validity(LOAD, RADIUS)
    film = math.sqrt(LOAD * RADIUS / (2.0 * math.pi * SIGMA))
    assert math.isclose(float(validity.slope_ratio), film / RADIUS, rel_tol=1e-12)
    assert validity.plug_flow_ratio is None

    partial = _plan("partially-mobile", "axisymmetric", hamaker_constant=hamaker)
    plug = partial.validity(LOAD, RADIUS)
    assert plug.plug_flow_ratio is not None
    expected_plug = MU_D / MU_C * expected / (4.0 * math.sqrt(3.0 * 0.66) * film)
    assert math.isclose(float(plug.plug_flow_ratio), expected_plug, rel_tol=1e-12)
    assert float(plug.validity_margin) == max(film / RADIUS, float(plug.plug_flow_ratio))


@pytest.mark.parametrize("contact_steps", (4, 12))
def test_pair_merges_only_when_drainage_is_shorter_than_contact(
    contact_steps: int,
) -> None:
    plan = _plan("immobile", "axisymmetric")
    drain = float(plan.drainage_time(H0, HC, LOAD, RADIUS))
    step = drain / 7.5
    touching = _observation([(2, 9), None])
    apart = _observation([(2, 9), None], in_contact=[False, False])
    merged_away = _observation([None, None])
    ledger = FilmContactLedger.empty(2, jnp.float64)
    events: list[tuple[bool, bool, float]] = []
    for index in range(contact_steps + 2):
        if any(event[0] for event in events):
            observation = merged_away
        else:
            observation = touching if index < contact_steps else apart
        update = _advance(plan, ledger, observation, jnp.asarray(step))
        ledger = update.ledger
        events.append(
            (
                bool(jnp.any(update.merge_proposals)),
                bool(jnp.any(update.release_events)),
                float(jnp.max(update.merge_time)),
            )
        )
        assert bool(update.successful)
    merges = [index for index, event in enumerate(events) if event[0]]
    releases = [index for index, event in enumerate(events) if event[1]]
    if contact_steps * step > drain:
        assert merges == [7] and releases == []
        assert math.isclose(7 * step + events[7][2], drain, rel_tol=1e-10)
    else:
        assert merges == [] and releases == [contact_steps]
    assert np.all(np.asarray(ledger.status) == FilmContactStatus.INACTIVE)


def test_ledger_identity_is_independent_of_observation_order() -> None:
    plan = _plan("fully-mobile", "axisymmetric", capacity=4)
    step = jnp.asarray(1.0e-6)
    pairs = [(1, 2), (3, 40), (5, 6), None]
    reordered = [None, (5, 6), (1, 2), (3, 40)]
    empty = FilmContactLedger.empty(4, jnp.float64)
    first = _advance(plan, empty, _observation(pairs, work=1.0e-9), step)
    slots = {pair: _slot_of(first, pair) for pair in ((1, 2), (3, 40), (5, 6))}
    second = _advance(plan, first.ledger, _observation(reordered, work=1.0e-9), step)
    direct = _advance(plan, first.ledger, _observation(pairs, work=1.0e-9), step)
    for pair, slot in slots.items():
        assert _slot_of(second, pair) == slot
    for field in ("first_id", "second_id", "status", "film_thickness", "contact_age"):
        np.testing.assert_array_equal(
            np.asarray(getattr(second.ledger, field)),
            np.asarray(getattr(direct.ledger, field)),
        )
    expected = float(plan.thickness_after(H0, 2.0e-6, LOAD, RADIUS))
    thickness = np.asarray(second.ledger.film_thickness)[list(slots.values())]
    np.testing.assert_allclose(thickness, expected, rtol=1e-12)
    work = np.asarray(second.ledger.near_contact_work)[list(slots.values())]
    np.testing.assert_allclose(work, 2.0e-9, rtol=1e-12)
    assert bool(second.successful)


def test_capacity_overflow_refuses_without_aliasing_and_slots_are_reused() -> None:
    plan = _plan("immobile", "axisymmetric", capacity=2)
    step = jnp.asarray(1.0e-3)
    empty = FilmContactLedger.empty(2, jnp.float64)
    full = _advance(plan, empty, _observation([(9, 10), (1, 2)]), step)
    assert bool(full.successful)
    released = _slot_of(full, (9, 10))
    kept = _slot_of(full, (1, 2))

    crowded = _advance(
        plan, full.ledger, _observation([(1, 2), (3, 4)], work=1.0e-9), step
    )
    assert bool(crowded.capacity_overflow)
    assert not bool(crowded.successful)
    np.testing.assert_array_equal(np.asarray(crowded.refused_observations), [False, True])
    assert int(crowded.observation_slots[1]) == -1
    assert math.isclose(float(crowded.untracked_near_contact_work), 1.0e-9)
    assert int(crowded.ledger.status[released]) == FilmContactStatus.RELEASED
    assert _slot_of(crowded, (1, 2)) == kept

    reused = _advance(plan, crowded.ledger, _observation([(1, 2), (3, 4)]), step)
    assert bool(reused.successful)
    assert _slot_of(reused, (3, 4)) == released
    assert _slot_of(reused, (1, 2)) == kept
    np.testing.assert_array_equal(
        np.asarray(reused.ledger.status),
        [FilmContactStatus.DRAINING, FilmContactStatus.DRAINING],
    )
    fresh = float(plan.thickness_after(H0, 1.0e-3, LOAD, RADIUS))
    assert math.isclose(
        float(reused.ledger.film_thickness[released]), fresh, rel_tol=1e-12
    )


def test_overflow_refusal_follows_pair_keys_not_observation_slots() -> None:
    plan = _plan("immobile", "axisymmetric", capacity=2)
    step = jnp.asarray(1.0e-3)
    empty = FilmContactLedger.empty(2, jnp.float64)
    seeded = _advance(plan, empty, _observation([(1, 2), None]), step)
    refused = []
    for order in ([(5, 6), (3, 4)], [(3, 4), (5, 6)]):
        update = _advance(plan, seeded.ledger, _observation(order), step)
        assert bool(update.capacity_overflow)
        index = int(np.flatnonzero(np.asarray(update.refused_observations))[0])
        refused.append(order[index])
        assert _slot_of(update, (3, 4)) != _slot_of(seeded, (1, 2))
        assert int(update.ledger.status[_slot_of(seeded, (1, 2))]) == (
            FilmContactStatus.RELEASED
        )
    assert refused == [(5, 6), (5, 6)]


def test_invalid_and_duplicate_observations_are_reported() -> None:
    plan = _plan("fully-mobile", "planar")
    empty = FilmContactLedger.empty(2, jnp.float64)
    step = jnp.asarray(1.0e-6)
    reversed_ids = _advance(plan, empty, _observation([(4, 2), None]), step)
    assert int(reversed_ids.invalid_observations) == 1
    assert not bool(reversed_ids.successful)
    assert np.all(np.asarray(reversed_ids.ledger.status) == FilmContactStatus.INACTIVE)
    duplicate = _advance(plan, empty, _observation([(2, 4), (2, 4)]), step)
    assert int(duplicate.duplicate_observations) == 1
    assert not bool(duplicate.successful)
    assert int(jnp.sum(duplicate.ledger.status == FilmContactStatus.DRAINING)) == 1
    unloaded = _advance(plan, empty, _observation([(2, 4), None], load=0.0), step)
    assert bool(unloaded.failed[_slot_of(unloaded, (2, 4))])
    assert not bool(unloaded.successful)


def test_construction_refusals() -> None:
    with pytest.raises(ValueError, match="planar"):
        _plan("partially-mobile", "planar")
    with pytest.raises(ValueError, match="exactly one"):
        _plan("immobile", "axisymmetric", hamaker_constant=1.0e-20, critical_thickness=HC)
    with pytest.raises(ValueError, match="exactly one"):
        FilmDrainageCoalescencePlan(
            regime="immobile",
            geometry="axisymmetric",
            pair_capacity=2,
            id_upper_bound=10,
            continuous_viscosity=MU_C,
            surface_tension=SIGMA,
            initial_film_thickness=H0,
        )
    with pytest.raises(ValueError, match="critical_thickness"):
        _plan("immobile", "planar", hamaker_constant=1.0e-20)
    with pytest.raises(ValueError, match="regime"):
        _plan("mobile", "axisymmetric")

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._sampling import derive_key, SampleAddress


_MAX_WORD = np.uint32(0xFFFFFFFF)
_ADDRESS = SampleAddress("test", "particle-identity", role="event")


def _plan(capacity: int, /) -> phx.discretization.ParticlePopulationPlan:
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity), jnp.ones((capacity,)), ambient_dimension=1
    ).prepare()
    return phx.discretization.ParticlePopulationPlan(particles)


def _request(
    event_ids: tuple[int, ...],
    /,
    *,
    parents: tuple[jax.Array, jax.Array] | None = None,
) -> phx.discretization.ParticleAllocationRequest:
    width = len(event_ids)
    return phx.discretization.ParticleAllocationRequest(
        jnp.asarray(event_ids, dtype=jnp.int64),
        jnp.ones((width,)),
        jnp.ones((width,), dtype=jnp.bool_),
        parents=parents,
    )


def _identity(state: phx.discretization.ParticlePopulationState, slot: int, /) -> int:
    return (int(state.id_hi[slot]) << 32) | int(state.id_lo[slot])


def _with_counter(
    state: phx.discretization.ParticlePopulationState, hi: int, lo: int, /
) -> phx.discretization.ParticlePopulationState:
    return eqx.tree_at(
        lambda value: (value.next_id_hi, value.next_id_lo),
        state,
        (jnp.asarray(hi, dtype=jnp.uint32), jnp.asarray(lo, dtype=jnp.uint32)),
    )


def test_initial_population_numbers_active_particles_in_slot_order() -> None:
    state = _plan(6).initialize(
        active_mask=jnp.asarray([True, False, True, True, False, False]),
        masses=jnp.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
    )
    assert [_identity(state, slot) for slot in (0, 2, 3)] == [0, 1, 2]
    assert (int(state.next_id_hi), int(state.next_id_lo)) == (0, 3)
    vacant = jnp.asarray([1, 4, 5])
    np.testing.assert_array_equal(state.id_hi[vacant], _MAX_WORD)
    np.testing.assert_array_equal(state.id_lo[vacant], _MAX_WORD)
    assert not bool(jnp.any(state.has_parent))


def test_allocation_assigns_identities_in_event_order_not_request_order() -> None:
    plan = _plan(5)
    state = plan.initialize(
        active_mask=jnp.asarray([True, False, False, False, False]),
        masses=jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    result = plan.allocate(state, _request((7, 3, 5)))
    assert bool(result.successful)
    slots = [int(slot) for slot in result.slots]
    accepted = result.accepted_state
    # Event 3 is created first, then 5, then 7.
    assert [_identity(accepted, slot) for slot in slots] == [3, 1, 2]
    assert (int(accepted.next_id_hi), int(accepted.next_id_lo)) == (0, 4)


def test_identities_stay_unique_across_slot_reuse_and_deactivation_cycles() -> None:
    plan = _plan(4)
    state = plan.initialize(
        active_mask=jnp.asarray([True, True, False, False]),
        masses=jnp.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    survivor = _identity(state, 0)
    issued = [survivor, _identity(state, 1)]
    reused_slots: list[int] = []
    for cycle in range(5):
        allocated = plan.allocate(state, _request((2 * cycle, 2 * cycle + 1)))
        assert bool(allocated.successful)
        slots = [int(slot) for slot in allocated.slots]
        reused_slots.extend(slots)
        issued.extend(_identity(allocated.accepted_state, slot) for slot in slots)
        # Retire everything except the original slot-0 particle.
        removed = plan.deactivate(
            allocated.accepted_state, jnp.asarray([False, True, True, True])
        )
        assert bool(removed.successful)
        state = removed.accepted_state
        assert _identity(state, 0) == survivor
    assert len(issued) == len(set(issued)) == 12
    assert len(set(reused_slots)) < len(reused_slots)
    assert int(state.incarnation[1]) >= 4


def test_allocation_records_parent_lineage_through_parent_retirement() -> None:
    plan = _plan(4)
    state = plan.initialize(
        active_mask=jnp.asarray([True, True, False, False]),
        masses=jnp.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    parent_ids = (state.id_hi[jnp.asarray([1, 0])], state.id_lo[jnp.asarray([1, 0])])
    children = plan.allocate(state, _request((0, 1), parents=parent_ids))
    assert bool(children.successful)
    child_slots = [int(slot) for slot in children.slots]
    accepted = children.accepted_state
    parents = [
        (int(accepted.parent_hi[slot]) << 32) | int(accepted.parent_lo[slot])
        for slot in child_slots
    ]
    assert parents == [_identity(state, 1), _identity(state, 0)]
    assert bool(jnp.all(accepted.has_parent[jnp.asarray(child_slots)]))
    retired = plan.deactivate(accepted, jnp.asarray([True, True, False, False]))
    orphan_lineage = retired.accepted_state
    np.testing.assert_array_equal(orphan_lineage.parent_hi, accepted.parent_hi)
    np.testing.assert_array_equal(orphan_lineage.parent_lo, accepted.parent_lo)
    np.testing.assert_array_equal(orphan_lineage.id_lo, accepted.id_lo)


def test_flip_split_children_descend_from_their_cell_receiver() -> None:
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(6), jnp.ones((6,)), ambient_dimension=2
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particles).initialize(
        active_mask=jnp.asarray([True, True, False, False, False, False]),
        masses=jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    flip_state = phx.discretization.flip.FLIPParticleState(
        jnp.asarray([[0.2, 0.2], [0.25, 0.2], *([[0.0, 0.0]] * 4)]),
        jnp.asarray([[1.0, 0.0], [1.0, 0.0], *([[0.0, 0.0]] * 4)]),
    )
    result = phx.discretization.flip.FLIPReseedingPlan(
        2,
        target_per_cell=4,
        minimum_per_cell=2,
        maximum_per_cell=5,
        maximum_events=4,
    ).apply(
        population,
        flip_state,
        jnp.asarray([0, 0, -1, -1, -1, -1]),
        jnp.asarray([[0.2, 0.2], [0.8, 0.8]]),
    )
    assert bool(result.successful)
    accepted = result.accepted_population
    children = np.flatnonzero(np.asarray(result.inserted))
    assert children.tolist() == [2, 3]
    assert [_identity(accepted, int(slot)) for slot in children] == [2, 3]
    np.testing.assert_array_equal(accepted.parent_hi[children], population.id_hi[0])
    np.testing.assert_array_equal(accepted.parent_lo[children], population.id_lo[0])
    assert [_identity(accepted, slot) for slot in (0, 1)] == [0, 1]
    assert (int(accepted.next_id_hi), int(accepted.next_id_lo)) == (0, 4)


def test_field_ionization_electron_descends_from_the_ionized_ion() -> None:
    ion_support = phx.discretization.ParticleSetPlan(
        jnp.arange(3), jnp.ones((3,)), ambient_dimension=3
    ).prepare()
    electron_support = phx.discretization.ParticleSetPlan(
        jnp.arange(10, 12), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    ions = phx.discretization.ParticlePopulationPlan(ion_support).initialize()
    electron_plan = phx.discretization.ParticlePopulationPlan(electron_support)
    electrons = electron_plan.initialize(
        active_mask=jnp.asarray([True, False]), masses=jnp.asarray([1.0, 0.0])
    )
    ion_model = phx.discretization.pic.PICChargeModelPlan(
        1.0,
        "ions",
        minimum_charge_number=0,
        maximum_charge_number=2,
        initial_charge_number=0,
    )
    electron_model = phx.discretization.pic.PICChargeModelPlan(
        -1.0,
        "electrons",
        minimum_charge_number=1,
        maximum_charge_number=1,
        initial_charge_number=1,
    )
    # Only ion slot 2 sees a field; its ionization probability is 1 - exp(-10).
    result = phx.discretization.pic.ionization.FieldIonizationPlan(
        1.0,
        field_power=1.0,
        ionization_energy=0.1,
        maximum_probability=1.0,
        maximum_events=1,
    ).apply(
        ion_model,
        ions,
        ion_model.initialize(ions),
        phx.discretization.pic.PICParticleState(jnp.zeros((3, 3)), jnp.zeros((3, 3))),
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0e8, 0.0, 0.0]]),
        electron_model,
        electron_plan,
        electrons,
        electron_model.initialize(electrons),
        phx.discretization.pic.PICParticleState(jnp.zeros((2, 3)), jnp.zeros((2, 3))),
        jr.key(0),
        1.0e-7,
        1,
    )
    assert bool(result.successful)
    assert int(result.event_count) == 1
    born = result.electron_population
    assert bool(born.active[1]) and bool(born.has_parent[1])
    assert _identity(born, 1) == 1
    assert (int(born.parent_hi[1]) << 32) | int(born.parent_lo[1]) == _identity(ions, 2)
    assert not bool(born.has_parent[0])


def test_population_update_births_receive_fresh_identities() -> None:
    plan = _plan(4)
    state = plan.initialize(
        active_mask=jnp.asarray([True, True, False, False]),
        masses=jnp.asarray([1.0, 1.0, 0.0, 0.0]),
    )
    updated = phx.discretization.update_particle_population(
        state,
        jnp.asarray([False, True, True, True]),
        jnp.asarray([0.0, 1.0, 2.0, 3.0]),
    )
    assert [_identity(updated, slot) for slot in (0, 1, 2, 3)] == [0, 1, 2, 3]
    assert (int(updated.next_id_hi), int(updated.next_id_lo)) == (0, 4)
    reborn = phx.discretization.update_particle_population(
        updated,
        jnp.asarray([True, True, True, True]),
        jnp.asarray([1.0, 1.0, 2.0, 3.0]),
    )
    assert _identity(reborn, 0) == 4


def test_identity_counter_carries_from_low_into_high_word() -> None:
    plan = _plan(3)
    state = _with_counter(
        plan.initialize(
            active_mask=jnp.asarray([True, False, False]),
            masses=jnp.asarray([1.0, 0.0, 0.0]),
        ),
        0,
        0xFFFFFFFF,
    )
    result = plan.allocate(state, _request((0, 1)))
    assert bool(result.successful)
    slots = [int(slot) for slot in result.slots]
    assert [_identity(result.accepted_state, slot) for slot in slots] == [
        0xFFFFFFFF,
        1 << 32,
    ]
    assert int(result.accepted_state.next_id_hi) == 1
    assert int(result.accepted_state.next_id_lo) == 1


def test_allocation_refuses_the_reserved_identity() -> None:
    plan = _plan(4)
    state = _with_counter(
        plan.initialize(
            active_mask=jnp.asarray([True, False, False, False]),
            masses=jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
        0xFFFFFFFF,
        0xFFFFFFFD,
    )
    refused = plan.allocate(state, _request((0, 1, 2)))
    assert not bool(refused.successful)
    assert int(refused.status) == int(
        phx.discretization.ParticlePopulationStatus.IDENTITY_EXHAUSTED
    )
    assert not bool(jnp.any(refused.allocated))
    for accepted, original in zip(
        jax.tree.leaves(refused.accepted_state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(accepted, original)
    last = plan.allocate(state, _request((0, 1)))
    assert bool(last.successful)
    assert _identity(last.accepted_state, int(last.slots[1])) == (1 << 64) - 2


def test_assignment_rejects_ranks_that_are_not_a_permutation() -> None:
    state = _plan(3).initialize(
        active_mask=jnp.asarray([False, False, False]), masses=jnp.zeros((3,))
    )
    no_parent = jnp.full((3,), _MAX_WORD, dtype=jnp.uint32)
    created = jnp.asarray([True, True, False])
    _, dense = phx.discretization.assign_particle_identities(
        state, created, jnp.asarray([1, 0, 0], dtype=jnp.int32), no_parent, no_parent
    )
    _, repeated = phx.discretization.assign_particle_identities(
        state, created, jnp.asarray([0, 0, 1], dtype=jnp.int32), no_parent, no_parent
    )
    _, gapped = phx.discretization.assign_particle_identities(
        state, created, jnp.asarray([0, 2, 0], dtype=jnp.int32), no_parent, no_parent
    )
    assert bool(dense)
    assert not bool(repeated)
    assert not bool(gapped)


def test_identity_keys_follow_particles_under_slot_permutation() -> None:
    plan = _plan(5)
    state = plan.initialize(
        active_mask=jnp.asarray([True, True, True, False, True]),
        masses=jnp.asarray([1.0, 2.0, 3.0, 0.0, 4.0]),
    )
    root = jr.key(11)

    def keys(population: phx.discretization.ParticlePopulationState) -> jax.Array:
        return jr.key_data(
            jax.vmap(lambda hi, lo: derive_key(root, _ADDRESS, 9, hi, lo, 2))(
                population.id_hi, population.id_lo
            )
        )

    permutation = jnp.asarray([4, 2, 0, 3, 1])
    per_slot = ("active", "mass", "incarnation", "ever_occupied", "retired")
    per_slot += ("id_hi", "id_lo", "parent_hi", "parent_lo")
    permuted = eqx.tree_at(
        lambda value: tuple(getattr(value, name) for name in per_slot),
        state,
        tuple(getattr(state, name)[permutation] for name in per_slot),
    )
    np.testing.assert_array_equal(keys(permuted), keys(state)[permutation])
    reference = np.asarray(keys(state))
    active = np.asarray(state.active)
    assert len({tuple(row) for row in reference[active]}) == int(active.sum())


def test_reallocated_slot_draws_a_different_key_than_its_previous_occupant() -> None:
    plan = _plan(2)
    state = plan.initialize(
        active_mask=jnp.asarray([True, False]), masses=jnp.asarray([1.0, 0.0])
    )
    before = derive_key(jr.key(3), _ADDRESS, 0, state.id_hi[0], state.id_lo[0], 0)
    removed = plan.deactivate(state, jnp.asarray([True, False])).accepted_state
    reused = plan.allocate(removed, _request((0, 1)))
    assert int(reused.slots[0]) == 0
    after_state = reused.accepted_state
    after = derive_key(
        jr.key(3), _ADDRESS, 0, after_state.id_hi[0], after_state.id_lo[0], 0
    )
    assert not np.array_equal(jr.key_data(before), jr.key_data(after))


@pytest.mark.parametrize(
    ("index", "error"),
    [
        pytest.param(-1, ValueError, id="negative-host"),
        pytest.param(2**32, ValueError, id="host-beyond-word"),
        pytest.param(True, TypeError, id="host-bool"),
        pytest.param(jnp.asarray(1.0), TypeError, id="float-array"),
        pytest.param(jnp.asarray([1, 2], dtype=jnp.uint32), ValueError, id="non-scalar"),
    ],
)
def test_derive_key_refuses_indices_that_are_not_one_exact_word(
    index: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        derive_key(jr.key(0), _ADDRESS, index)  # ty: ignore[invalid-argument-type]


def test_derive_key_folds_word_boundaries_exactly() -> None:
    root = jr.key(5)
    maximum = derive_key(root, _ADDRESS, 2**32 - 1)
    np.testing.assert_array_equal(
        jr.key_data(maximum),
        jr.key_data(derive_key(root, _ADDRESS, jnp.asarray(_MAX_WORD))),
    )
    assert not np.array_equal(
        jr.key_data(derive_key(root, _ADDRESS, 0, 1)),
        jr.key_data(derive_key(root, _ADDRESS, 1, 0)),
    )

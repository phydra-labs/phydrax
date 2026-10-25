#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generation support, sampler lanes, filters and bounded resumable streams.

Independent controls: proposals of explicit test samplers are pure functions
of ``derive_key(root, address(campaign, support, sampler), ordinal)`` and are
recomputed directly; charge balance is checked against brute-force
enumeration of oxidation-state assignments; minimum distances against an
explicit NumPy image sum.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._identity import NumericRevision
from phydrax.atomistic._conditioning import AtomisticConditioning
from phydrax.atomistic._generation import (
    AbstractAtomisticGenerationSink,
    AbstractPreparedAtomisticSampler,
    atomistic_proposal_id,
    AtomisticCellAdmission,
    AtomisticCellBounds,
    AtomisticChargeBalanceQualifier,
    AtomisticChemistryAdmission,
    AtomisticCompositionBounds,
    AtomisticGeneratedProposal,
    AtomisticGenerationState,
    AtomisticGenerationStopReason,
    AtomisticGenerationStreamPlan,
    AtomisticGenerationSupport,
    AtomisticMemoryGenerationSink,
    AtomisticMinimumDistanceFilter,
    AtomisticPayloadSpec,
    AtomisticProposalBlock,
    AtomisticProposalStatus,
    AtomisticSamplerFamily,
    AtomisticWeightedRoute,
    map_admitted_lanes,
    PreparedAtomisticGenerationStream,
    proposal_address,
    proposal_keys,
    role_keys,
)
from phydrax.typing import ConvertibleToArray
from phydrax.units import ANGSTROM


CAMPAIGN = "generation-stream-test"
PAYLOAD = AtomisticPayloadSpec((("draw", (), "float64"), ("remaining", (), "int32")))


def _support(capacity: int = 3) -> AtomisticGenerationSupport:
    species = (1, 8) + (0,) * (capacity - 2)
    mask = (True, True) + (False,) * (capacity - 2)
    return AtomisticGenerationSupport(
        atom_capacity=capacity,
        species_vocabulary=(1, 8),
        vocabulary_masses=(1.008, 15.999),
        mass_provenance="test masses",
        chemistry=AtomisticChemistryAdmission.FIXED,
        length_unit=ANGSTROM,
        fixed_species=species,
        fixed_mask=mask,
    )


def _draw(keys: jax.Array, capacity: int) -> jax.Array:
    return jax.vmap(lambda key: 4.0 * jr.uniform(key, (capacity, 3), dtype=jnp.float64))(
        role_keys(keys, "positions")
    )


class _OneShot(AbstractPreparedAtomisticSampler):
    """Uniform positions; ordinals ``= 4 mod 5`` report a sampler failure."""

    support: AtomisticGenerationSupport
    payload_spec: AtomisticPayloadSpec
    sampler_id: str = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)

    def __init__(self, name: str = "one-shot", capacity: int = 3) -> None:
        self.support = _support(capacity)
        self.payload_spec = PAYLOAD
        self.sampler_id = name
        self.stateful = False
        self.idempotent = True

    def vacant_method(self, lanes: int) -> jax.Array:
        return jnp.zeros((lanes, self.support.atom_capacity, 3), dtype=jnp.float64)

    def initialize_method(
        self,
        keys: jax.Array,
        ordinals: jax.Array,
        conditioning: AtomisticConditioning | None,
        selected: jax.Array,
    ) -> jax.Array:
        return _draw(keys, self.support.atom_capacity)

    def emit_method(
        self,
        method: jax.Array,
        ordinals: jax.Array,
    ) -> AtomisticProposalBlock:
        status = jnp.where(
            ordinals % 5 == 4, int(AtomisticProposalStatus.SAMPLER_FAILURE), 0
        )
        mask = jnp.asarray(self.support.fixed_mask)
        return AtomisticProposalBlock(
            ordinals,
            jnp.where(mask[None, :, None], method, 0.0),
            jnp.broadcast_to(jnp.asarray(self.support.fixed_species), method.shape[:2]),
            jnp.broadcast_to(mask, method.shape[:2]),
            None,
            status,
            (method[:, 0, 0], jnp.zeros(ordinals.shape, dtype=jnp.int32)),
            sampler_id=self.sampler_id,
            payload_names=self.payload_spec.names,
        )

    def signature_facts(self) -> dict[str, str | int]:
        return {"kind": "test-one-shot", "atoms": self.support.atom_capacity}

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(self.sampler_id, {"scale": np.asarray(4.0)})


class _Countdown(AbstractPreparedAtomisticSampler):
    """Stateful: ordinal ``o`` needs ``1 + o mod 4`` advance steps."""

    support: AtomisticGenerationSupport
    payload_spec: AtomisticPayloadSpec
    sampler_id: str = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)

    def __init__(self) -> None:
        self.support = _support()
        self.payload_spec = PAYLOAD
        self.sampler_id = "countdown"
        self.stateful = True
        self.idempotent = False

    def vacant_method(self, lanes: int) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros((lanes, 3, 3), dtype=jnp.float64),
            jnp.zeros((lanes,), dtype=jnp.int32),
        )

    def initialize_method(
        self,
        keys: jax.Array,
        ordinals: jax.Array,
        conditioning: AtomisticConditioning | None,
        selected: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _draw(keys, 3), (1 + ordinals % 4).astype(jnp.int32)

    def advance_method(
        self,
        method: tuple[jax.Array, jax.Array],
        running: jax.Array,
        steps: int,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        positions, remaining = method
        taken = jnp.minimum(remaining, steps)
        moved = positions + 0.01 * taken[:, None, None].astype(positions.dtype)
        remaining = remaining - taken
        return (moved, remaining), remaining == 0

    def emit_method(
        self,
        method: tuple[jax.Array, jax.Array],
        ordinals: jax.Array,
    ) -> AtomisticProposalBlock:
        positions, remaining = method
        mask = jnp.asarray(self.support.fixed_mask)
        return AtomisticProposalBlock(
            ordinals,
            jnp.where(mask[None, :, None], positions, 0.0),
            jnp.broadcast_to(
                jnp.asarray(self.support.fixed_species), positions.shape[:2]
            ),
            jnp.broadcast_to(mask, positions.shape[:2]),
            None,
            jnp.zeros(ordinals.shape, dtype=jnp.int32),
            (positions[:, 0, 0], remaining),
            sampler_id=self.sampler_id,
            payload_names=self.payload_spec.names,
        )

    def signature_facts(self) -> dict[str, str | int]:
        return {"kind": "test-countdown"}

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(self.sampler_id, {"rate": np.asarray(0.01)})


def _run(
    stream: AtomisticGenerationStreamPlan,
    *,
    target: int,
    budget: int,
    root: int = 7,
    first: int = 0,
    sink: AtomisticMemoryGenerationSink | None = None,
    transitions: int = 200,
) -> tuple[
    PreparedAtomisticGenerationStream,
    AtomisticGenerationState,
    AtomisticMemoryGenerationSink,
]:
    runtime = stream.prepare()
    state = runtime.start(
        jr.key(root), target=target, attempt_budget=budget, first_ordinal=first
    )
    sink = AtomisticMemoryGenerationSink("test") if sink is None else sink
    state = runtime.run(state, sink, maximum_transitions=transitions)
    return runtime, state, sink


def _direct(
    sampler: _OneShot,
    ordinals: Sequence[int],
    root: int = 7,
) -> np.ndarray:
    """Independent recomputation of one-shot proposals from their address."""
    address = proposal_address(
        CAMPAIGN, sampler.support.support_id, sampler.sampler_id, role="proposal"
    )
    keys = proposal_keys(jr.key(root), address, jnp.asarray(ordinals, dtype=jnp.int64))
    return np.asarray(_draw(keys, sampler.support.atom_capacity))


_EVALUATED: list[int] = []


def _record_lane(lane: jax.Array) -> None:
    _EVALUATED.append(int(lane))


def _lane_square(lane: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
    jax.debug.callback(_record_lane, lane)
    return value * value, lane


@pytest.mark.parametrize("chunk", [1, 2, 3, 5])
def test_admitted_lane_map_reads_only_admitted_lanes_in_bounded_chunks(
    chunk: int,
) -> None:
    lanes = 5
    selected = jnp.asarray([False, True, False, True, True])
    lane_ids = jnp.arange(lanes, dtype=jnp.int32)
    # Vacant lanes carry poison inputs: reading one would surface a NaN.
    values = jnp.where(selected, jnp.arange(lanes, dtype=jnp.float64) + 1.0, jnp.nan)
    vacant = (jnp.full((lanes,), -1.0), jnp.full((lanes,), -1, dtype=jnp.int32))

    @eqx.filter_jit
    def mapped(
        mask: jax.Array,
        ids: jax.Array,
        inputs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return map_admitted_lanes(
            _lane_square, mask, (ids, inputs), vacant, lane_chunk=chunk
        )

    _EVALUATED.clear()
    squares, owners = mapped(selected, lane_ids, values)
    jax.effects_barrier()

    np.testing.assert_array_equal(squares, [-1.0, 4.0, -1.0, 16.0, 25.0])
    np.testing.assert_array_equal(owners, [-1, 1, -1, 3, 4])
    # Only admitted lanes are evaluated; padding of the last bounded chunk
    # repeats an admitted lane and is discarded.
    assert set(_EVALUATED) <= {1, 3, 4}
    assert len(_EVALUATED) == -(-3 // min(chunk, lanes)) * min(chunk, lanes)
    _EVALUATED.clear()
    empty = mapped(jnp.zeros((lanes,), dtype=jnp.bool_), lane_ids, values)
    jax.effects_barrier()
    assert not _EVALUATED
    np.testing.assert_array_equal(empty[0], vacant[0])
    np.testing.assert_array_equal(empty[1], vacant[1])


def test_proposals_are_ordinal_addressed_independent_of_lanes_and_chunks() -> None:
    sampler = _OneShot()
    runs = []
    for lanes in (1, 3):
        stream = AtomisticGenerationStreamPlan(
            sampler,
            campaign_id=CAMPAIGN,
            lanes_per_member=lanes,
            pending_capacity=4 * lanes,
        )
        _, state, sink = _run(stream, target=100, budget=8)
        assert state.stop_reason is AtomisticGenerationStopReason.BUDGET_EXHAUSTED
        runs.append(sink.records)
    first, second = runs
    assert [item.ordinal for item in first] == list(range(8))
    assert [item.proposal_id for item in first] == [item.proposal_id for item in second]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)
    expected = _direct(sampler, range(8))
    for item in first:
        np.testing.assert_array_equal(item.positions, expected[item.ordinal][:2])
        assert item.proposal_id == atomistic_proposal_id(
            CAMPAIGN, sampler.support.support_id, sampler.sampler_id, item.ordinal
        )
    stream = AtomisticGenerationStreamPlan(
        sampler, campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    )
    _, _, low = _run(stream, target=100, budget=4)
    _, _, high = _run(stream, target=100, budget=4, first=4)
    chunked = low.records + high.records
    assert [item.proposal_id for item in chunked] == [item.proposal_id for item in first]


def test_rejections_are_published_with_reasons_and_shortfall_is_explicit() -> None:
    stream = AtomisticGenerationStreamPlan(
        _OneShot(), campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    )
    runtime, state, sink = _run(stream, target=10, budget=10)
    statuses = [item.status for item in sink.records]
    assert len(statuses) == 10
    assert statuses[4] == statuses[9] == int(AtomisticProposalStatus.SAMPLER_FAILURE)
    assert sum(status == 0 for status in statuses) == 8
    assert state.stop_reason is AtomisticGenerationStopReason.BUDGET_EXHAUSTED
    assert runtime.shortfall(state) == 2
    assert int(state.cursor.committed_rejected) == 2


def test_target_cut_publishes_exactly_the_first_accepted_in_ordinal_order() -> None:
    stream = AtomisticGenerationStreamPlan(
        _OneShot(), campaign_id=CAMPAIGN, lanes_per_member=3, pending_capacity=6
    )
    runtime, state, sink = _run(stream, target=5, budget=50)
    records = sink.records
    # Ordinal 4 is rejected: the fifth acceptance is ordinal 5; the reject stays.
    assert [item.ordinal for item in records] == [0, 1, 2, 3, 4, 5]
    assert sum(item.accepted for item in records) == 5
    assert state.stop_reason is AtomisticGenerationStopReason.TARGET_REACHED
    assert runtime.finished(state) and runtime.shortfall(state) == 0


@pytest.mark.parametrize(
    ("target", "budget", "reason"),
    [
        (0, 10, AtomisticGenerationStopReason.ZERO_TARGET),
        (3, 0, AtomisticGenerationStopReason.ZERO_BUDGET),
    ],
)
def test_zero_work_returns_empty_results_without_transitions(
    target: int,
    budget: int,
    reason: AtomisticGenerationStopReason,
) -> None:
    stream = AtomisticGenerationStreamPlan(
        _OneShot(), campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    )
    runtime, state, sink = _run(stream, target=target, budget=budget)
    assert sink.records == ()
    assert state.stop_reason is reason
    assert int(state.cursor.transitions) == 0 and int(state.cursor.next_ordinal) == 0
    assert not bool(np.any(np.asarray(state.lanes[0].active)))
    assert runtime.finished(state)


def test_backpressure_bounds_pending_work_and_resumes() -> None:
    class _Refusing(AbstractAtomisticGenerationSink):
        @property
        def sink_id(self) -> str:
            return "blocked"

        def publish(self, proposals: Sequence[AtomisticGeneratedProposal], /) -> None:
            return None

        def published(self, proposal_ids: Sequence[str], /) -> tuple[bool, ...]:
            return (False,) * len(proposal_ids)

    stream = AtomisticGenerationStreamPlan(
        _OneShot(), campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    )
    runtime = stream.prepare()
    state = runtime.start(jr.key(7), target=100, attempt_budget=20)
    state = runtime.run(state, _Refusing(), maximum_transitions=10)
    # Nothing durable: the watermark stays and admission stops at the bank bound.
    assert int(state.cursor.watermark) == 0
    assert int(state.cursor.next_ordinal) == 4
    sink = AtomisticMemoryGenerationSink("open")
    state = runtime.run(state, sink, maximum_transitions=50)
    assert [item.ordinal for item in sink.records] == list(range(20))


def test_stateful_restart_from_checkpoint_is_exact(tmp_path: Path) -> None:
    stream = AtomisticGenerationStreamPlan(
        _Countdown(), campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    )
    _, reference_state, reference = _run(stream, target=100, budget=9)
    runtime = stream.prepare()
    state = runtime.start(jr.key(7), target=100, attempt_budget=9)
    sink = AtomisticMemoryGenerationSink("test")
    state = runtime.run(state, sink, maximum_transitions=3)
    assert bool(np.any(np.asarray(state.lanes[0].active)))
    path = tmp_path / "generation.npz"
    runtime.write_checkpoint(path, state)
    restored = runtime.read_checkpoint(path)
    jax.tree.map(
        np.testing.assert_array_equal, jax.device_get(restored), jax.device_get(state)
    )
    resumed = runtime.run(restored, sink, maximum_transitions=200)
    assert [item.proposal_id for item in sink.records] == [
        item.proposal_id for item in reference.records
    ]
    for left, right in zip(sink.records, reference.records, strict=True):
        np.testing.assert_array_equal(left.positions, right.positions)
    assert int(resumed.cursor.watermark) == int(reference_state.cursor.watermark) == 9


def test_checkpoint_refuses_another_stream_identity(tmp_path: Path) -> None:
    runtime = AtomisticGenerationStreamPlan(
        _Countdown(), campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    ).prepare()
    path = tmp_path / "generation.npz"
    runtime.write_checkpoint(path, runtime.start(jr.key(1), target=2, attempt_budget=4))
    other = AtomisticGenerationStreamPlan(
        _Countdown(),
        campaign_id=CAMPAIGN + "-other",
        lanes_per_member=2,
        pending_capacity=4,
    ).prepare()
    with pytest.raises(ValueError):
        other.read_checkpoint(path)


def test_reconcile_commits_durable_proposals_once_after_a_crash() -> None:
    stream = AtomisticGenerationStreamPlan(
        _OneShot(), campaign_id=CAMPAIGN, lanes_per_member=2, pending_capacity=4
    )
    runtime = stream.prepare()
    state = runtime.start(jr.key(7), target=100, attempt_budget=4)
    state = runtime.segment(state, maximum_transitions=3)
    sink = AtomisticMemoryGenerationSink("store")
    ready = runtime.ready(state)
    sink.publish(ready)  # durable, then "crash" before commit
    state = runtime.reconcile(state, sink)
    assert int(state.cursor.watermark) == len(ready)
    state = runtime.run(state, sink, maximum_transitions=20)
    ids = [item.proposal_id for item in sink.records]
    assert len(ids) == len(set(ids)) == 4


def test_weighted_family_routes_by_ordinal_independent_of_lanes() -> None:
    small, large = _OneShot("small", 3), _OneShot("large", 4)
    route = AtomisticWeightedRoute((0.3, 0.7), provenance="test prior")
    members = []
    for lanes in (1, 2):
        stream = AtomisticGenerationStreamPlan(
            AtomisticSamplerFamily((small, large), route=route),
            campaign_id=CAMPAIGN,
            lanes_per_member=lanes,
            pending_capacity=4 * lanes,
        )
        _, _, sink = _run(stream, target=100, budget=12)
        members.append([(item.ordinal, item.member) for item in sink.records])
    assert members[0] == members[1]
    expected = np.asarray(route.members(jr.key(7), jnp.arange(12, dtype=jnp.int64)))
    assert [member for _, member in members[0]] == expected.tolist()


def test_minimum_distance_filter_retains_rejected_proposals() -> None:
    minimum = 1.5
    stream = AtomisticGenerationStreamPlan(
        _OneShot(),
        campaign_id=CAMPAIGN,
        lanes_per_member=2,
        pending_capacity=4,
        filters=(AtomisticMinimumDistanceFilter(minimum),),
    )
    _, _, sink = _run(stream, target=100, budget=12)
    assert len(sink.records) == 12
    for item in sink.records:
        distance = float(np.linalg.norm(item.positions[0] - item.positions[1]))
        np.testing.assert_allclose(item.filter_values[0], distance, rtol=1e-12)
        if item.ordinal % 5 == 4:
            continue
        rejected = distance < minimum
        assert bool(item.filter_bits & 1) == rejected
        assert (
            bool(item.status & int(AtomisticProposalStatus.GEOMETRY_INVALID)) == rejected
        )
        assert (
            bool(item.status & int(AtomisticProposalStatus.FILTER_REJECTED)) == rejected
        )


def _periodic_block(
    positions: ConvertibleToArray,
    cell: ConvertibleToArray,
    species: ConvertibleToArray,
) -> AtomisticProposalBlock:
    positions = jnp.asarray(positions, dtype=jnp.float64)[None]
    return AtomisticProposalBlock(
        jnp.zeros((1,), dtype=jnp.int64),
        positions,
        jnp.asarray(species, dtype=jnp.int32)[None],
        jnp.ones(positions.shape[:2], dtype=jnp.bool_),
        jnp.asarray(cell, dtype=jnp.float64)[None],
        jnp.zeros((1,), dtype=jnp.int32),
        (),
        sampler_id="test",
        payload_names=(),
    )


def test_periodic_minimum_distance_counts_self_images_against_brute_force() -> None:
    cell = np.asarray([[4.0, 0.0, 0.0], [1.3, 3.9, 0.0], [0.4, 0.7, 4.2]])
    positions = np.asarray([[0.1, 0.2, 0.3], [3.9, 3.6, 4.0], [2.0, 2.0, 2.0]])
    evidence = AtomisticMinimumDistanceFilter(2.0, image_extent=3).evaluate(
        _periodic_block(positions, cell, (8, 8, 8))
    )
    best = np.inf
    for n in itertools.product(range(-4, 5), repeat=3):
        shift = np.asarray(n) @ cell
        for i in range(3):
            for j in range(3):
                if i == j and not any(n):
                    continue
                best = min(
                    best, float(np.linalg.norm(positions[j] - positions[i] + shift))
                )
    np.testing.assert_allclose(float(evidence.value[0]), best, rtol=1e-12)
    assert bool(evidence.keep[0]) == (best >= 2.0)
    # A single atom in a thin cell meets its own image.
    thin = np.diag([1.2, 6.0, 6.0])
    lonely = AtomisticMinimumDistanceFilter(1.5, image_extent=2).evaluate(
        _periodic_block([[0.0, 0.0, 0.0]], thin, (8,))
    )
    np.testing.assert_allclose(float(lonely.value[0]), 1.2, rtol=1e-12)
    assert not bool(lonely.keep[0])


def test_charge_balance_qualifier_matches_brute_force_assignments() -> None:
    states = {8: (-2,), 26: (2, 3), 11: (1,)}
    qualifier = AtomisticChargeBalanceQualifier(
        states, provenance="test states", maximum_atoms=6
    )
    cases = [
        (26, 26, 8, 8, 8, 0),
        (26, 8, 8, 0, 0, 0),
        (11, 11, 8, 0, 0, 0),
        (26, 8, 11, 8, 0, 0),
    ]
    species = np.asarray(cases, dtype=np.int32)
    mask = species > 0
    block = AtomisticProposalBlock(
        jnp.arange(len(cases), dtype=jnp.int64),
        jnp.zeros(species.shape + (3,), dtype=jnp.float64),
        species,
        mask,
        None,
        jnp.zeros((len(cases),), dtype=jnp.int32),
        (),
        sampler_id="test",
        payload_names=(),
    )
    evidence = qualifier.evaluate(block)
    for row, case in enumerate(cases):
        atoms = [number for number in case if number]
        balanced = any(
            sum(choice) == 0
            for choice in itertools.product(*(states[number] for number in atoms))
        )
        assert bool(evidence.keep[row]) == balanced


def test_support_admission_refuses_inconsistent_chemistry_and_cells() -> None:
    with pytest.raises(ValueError, match="Variable composition infers no topology"):
        AtomisticGenerationSupport(
            atom_capacity=4,
            species_vocabulary=(8, 26),
            vocabulary_masses=(15.999, 55.845),
            mass_provenance="test",
            chemistry=AtomisticChemistryAdmission.VARIABLE_COMPOSITION,
            length_unit=ANGSTROM,
            composition=AtomisticCompositionBounds(
                {8: (1, 3), 26: (1, 2)}, minimum_atoms=2, maximum_atoms=4
            ),
            topology="fixed",
        )
    bounds = AtomisticCellBounds(
        minimum_height=2.0,
        maximum_condition=50.0,
        minimum_volume=5.0,
        maximum_volume=500.0,
    )
    support = AtomisticGenerationSupport(
        atom_capacity=4,
        species_vocabulary=(8, 26),
        vocabulary_masses=(15.999, 55.845),
        mass_provenance="test",
        chemistry=AtomisticChemistryAdmission.VARIABLE_COMPOSITION,
        length_unit=ANGSTROM,
        composition=AtomisticCompositionBounds(
            {8: (1, 3), 26: (1, 2)}, minimum_atoms=2, maximum_atoms=4
        ),
        cell_admission=AtomisticCellAdmission.VARIABLE_CELL,
        cell_bounds=bounds,
    )
    good = _periodic_block(np.zeros((4, 3)), 4.0 * np.eye(3), (26, 8, 8, 8))
    flat = _periodic_block(np.zeros((4, 3)), np.diag([4.0, 4.0, 1.0]), (26, 8, 8, 8))
    unknown = _periodic_block(np.zeros((4, 3)), 4.0 * np.eye(3), (26, 7, 8, 8))
    assert int(support.admit(good)[0]) == 0
    assert int(support.admit(flat)[0]) & int(AtomisticProposalStatus.CELL_INVALID)
    assert int(support.admit(unknown)[0]) & int(AtomisticProposalStatus.SPECIES_INVALID)

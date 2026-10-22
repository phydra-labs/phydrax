# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Restart packet boundaries retain typed history and status attribution."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import AdaptiveSimplexStatus
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexState,
)
from phydrax.meshing._restart_distribution import (
    _empty,
    _invalid_evidence,
    _invalid_reason_summary,
    _merge_packet,
    _packet_conflict,
    _repack,
    _source_vertex_activity,
    packet,
    SimplexRestartInvalidReason,
    SimplexRestartRecordDtype,
    SimplexRestartRecordRole,
    validate_restart_duplicate_conflicts,
    validate_restart_invalid_reasons,
)


@pytest.mark.parametrize("capacity", (3, 2))
def test_restart_packet_required_capacity_and_one_less(capacity: int) -> None:
    packet = {
        "ids": jnp.asarray([7, 11, 13], dtype=jnp.int64),
        "rng": jnp.asarray([3, 5, 8], dtype=jnp.uint32),
        "history": jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "valid": jnp.asarray([True, False, True]),
    }
    original = {name: np.asarray(value).copy() for name, value in packet.items()}
    result, status = _merge_packet(_empty(packet, capacity), packet, jnp.ones(3, bool))
    assert int(status) == (0 if capacity == 3 else int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED))
    for name, value in packet.items():
        np.testing.assert_array_equal(result[name], original[name][:capacity])
        np.testing.assert_array_equal(value, original[name])
        assert result[name].dtype == value.dtype


@pytest.mark.parametrize("name", ("rng", "history", "valid"))
def test_restart_packet_conflicting_history_is_invalid_not_capacity(name: str) -> None:
    packet = {
        "ids": jnp.asarray([7], dtype=jnp.int64),
        "rng": jnp.asarray([3], dtype=jnp.uint32),
        "history": jnp.asarray([[1.0, 2.0]]),
        "valid": jnp.asarray([True]),
    }
    resident, status = _merge_packet(_empty(packet, 2), packet, jnp.ones(1, bool))
    assert int(status) == 0
    incoming = dict(packet)
    incoming[name] = ~packet[name] if name == "valid" else packet[name] + 1
    before = {key: np.asarray(value).copy() for key, value in resident.items()}
    _, status = _merge_packet(resident, incoming, jnp.ones(1, bool))
    assert int(status) == int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    for key, value in resident.items():
        np.testing.assert_array_equal(value, before[key])


@pytest.mark.parametrize("reason", tuple(SimplexRestartInvalidReason))
def test_restart_reason_each_bit_preserves_scalar_status(
    reason: SimplexRestartInvalidReason,
) -> None:
    clear = _invalid_evidence(jnp.asarray(False), reason)
    refused = _invalid_evidence(jnp.asarray(True), reason)
    assert int(clear) == 0
    assert int(refused & 255) == int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    assert int(refused >> 8) == int(reason)
    masks = jnp.asarray([0, int(reason)], dtype=jnp.int64)
    status = jnp.full((2,), int(AdaptiveSimplexStatus.INVALID_GEOMETRY), jnp.int32)
    validate_restart_invalid_reasons(status, masks)
    mask, counts, first = _invalid_reason_summary(masks)
    assert int(mask) == int(reason)
    expected = np.asarray([bit == reason for bit in SimplexRestartInvalidReason])
    np.testing.assert_array_equal(counts, expected.astype(np.int32))
    np.testing.assert_array_equal(first, np.where(expected, 1, -1))


def test_restart_reason_combination_and_ordinary_success() -> None:
    reasons = (
        SimplexRestartInvalidReason.CELL_AUTHORITY,
        SimplexRestartInvalidReason.DUPLICATE_RECORD_CONFLICT,
        SimplexRestartInvalidReason.VERTEX_COUNT,
    )
    combined = jnp.int64(0)
    for reason in reasons:
        combined |= _invalid_evidence(jnp.asarray(True), reason)
    assert int(combined >> 8) == sum(map(int, reasons))
    assert int(combined & 255) == int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    masks = jnp.zeros(3, jnp.int64)
    validate_restart_invalid_reasons(jnp.zeros(3, jnp.int32), masks)
    mask, counts, first = _invalid_reason_summary(masks)
    assert int(mask) == 0
    np.testing.assert_array_equal(counts, np.zeros(len(SimplexRestartInvalidReason)))
    np.testing.assert_array_equal(first, -np.ones(len(SimplexRestartInvalidReason)))


@pytest.mark.parametrize(
    ("status", "masks"),
    (([0], [1]), ([32], [0]), ([32], [1 << 34]), ([32], [-1])),
)
def test_restart_reason_refuses_inconsistent_or_unknown_evidence(
    status: list[int], masks: list[int],
) -> None:
    with pytest.raises(ValueError, match="known bits consistent with status"):
        validate_restart_invalid_reasons(
            jnp.asarray(status, jnp.int32), jnp.asarray(masks, jnp.int64)
        )


@pytest.mark.parametrize("name", ("rng", "history", "valid"))
def test_restart_duplicate_reason_preserves_packets_and_status(name: str) -> None:
    incoming = {
        "ids": jnp.asarray([7], jnp.int64),
        "rng": jnp.asarray([3], jnp.uint32),
        "history": jnp.asarray([[1.0, 2.0]]),
        "valid": jnp.asarray([True]),
    }
    resident, _ = _merge_packet(_empty(incoming, 2), incoming, jnp.ones(1, bool))
    incoming[name] = ~incoming[name] if name == "valid" else incoming[name] + 1
    ordinary, status = _merge_packet(resident, incoming, jnp.ones(1, bool))
    diagnosed, flags = _merge_packet(
        resident, incoming, jnp.ones(1, bool), diagnostic=True
    )
    assert int(flags & 255) == int(status)
    assert int(flags >> 8) == int(SimplexRestartInvalidReason.DUPLICATE_RECORD_CONFLICT)
    for key in ordinary:
        np.testing.assert_array_equal(ordinary[key], diagnosed[key])
        assert ordinary[key].dtype == diagnosed[key].dtype


@pytest.fixture
def restart_raw() -> tuple[AdaptiveSimplexLayout, AdaptiveSimplexState]:
    layout = AdaptiveSimplexLayout(
        2, 2, vertex_capacity=4, cell_capacity=2,
        protected_edge_capacity=2, maximum_closure_iterations=8,
        maximum_coarsening_passes=8,
    )
    cells = np.asarray([[0, 1, 2]], dtype=np.int32)
    state = adaptive_simplex_state(
        layout,
        coordinates=np.asarray([[0., 0.], [1., 0.], [0., 1.]]),
        vertex_ids=np.arange(3, dtype=np.int64),
        vertex_active=np.ones(3, dtype=bool),
        vertex_parents=np.full((3, 2), -1, dtype=np.int32),
        vertex_protected=np.zeros(3, dtype=bool),
        cells=cells, tuples=cells, tags=np.asarray([2], dtype=np.int32),
        blocks=np.zeros(1, dtype=np.int32),
        generations=np.zeros(1, dtype=np.int32),
        parents=np.full(1, -1, dtype=np.int32),
        children=np.full((1, 2), -1, dtype=np.int32),
        bisection_vertices=np.full(1, -1, dtype=np.int32),
        cell_ids=np.asarray([0], dtype=np.int64),
        cell_active=np.ones(1, dtype=bool),
        cell_classes=np.zeros(1, dtype=np.int32),
        facet_classes=np.zeros((1, 3), dtype=np.int32),
        protected_edges=np.empty((0, 2), dtype=np.int32),
        next_vertex_id=3, next_cell_id=1,
    )
    return layout, state


@pytest.mark.parametrize(
    ("field", "index", "value", "reason"),
    (
        ("parents", (0,), -2, SimplexRestartInvalidReason.CELL_PARENT_RANGE),
        ("mesh.cells", (0, 0), 4, SimplexRestartInvalidReason.CELL_VERTEX_RANGE),
        ("tuples", (0, 0), 4, SimplexRestartInvalidReason.TUPLE_VERTEX_RANGE),
        ("bisection_vertices", (0,), 4, SimplexRestartInvalidReason.BISECTION_VERTEX_RANGE),
        ("children", (0, 0), 2, SimplexRestartInvalidReason.CELL_CHILD_RANGE),
        ("vertex_parents", (0, 0), 4, SimplexRestartInvalidReason.VERTEX_PARENT_RANGE),
        ("mesh.cells", (0, 0), -1, SimplexRestartInvalidReason.REQUIRED_VERTEX_SLOT),
        ("mesh.vertex_ids", (0,), -1, SimplexRestartInvalidReason.REQUIRED_VERTEX_ID),
        ("parents", (0,), 2, SimplexRestartInvalidReason.CELL_PARENT_ID),
        ("children", (0, 0), 1, SimplexRestartInvalidReason.CELL_CHILD_ID),
        ("bisection_vertices", (0,), 3, SimplexRestartInvalidReason.BISECTION_VERTEX_ID),
        ("vertex_parents", (0, 0), 3, SimplexRestartInvalidReason.VERTEX_PARENT_ID),
        ("protected_codes", (0,), 3, SimplexRestartInvalidReason.PROTECTED_ENDPOINT_ID),
        ("mesh.coordinates", (0, 0), float("nan"), SimplexRestartInvalidReason.NONFINITE_COORDINATE),
        ("protected_codes", (0,), -1, SimplexRestartInvalidReason.PROTECTED_CODE_RANGE),
        ("cursors", (2,), 2, SimplexRestartInvalidReason.ALLOCATOR_CURSOR),
    ),
)
def test_restart_packet_predicate_attribution(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
    field: str,
    index: tuple[int, ...],
    value: int | float,
    reason: SimplexRestartInvalidReason,
) -> None:
    _, raw = restart_raw

    def leaf(state: AdaptiveSimplexState) -> jax.Array:
        if field.startswith("mesh."):
            return getattr(state.mesh, field.removeprefix("mesh."))
        return getattr(state, field)

    changed = eqx.tree_at(leaf, raw, leaf(raw).at[index].set(value))
    ordinary = packet(changed, (), ())
    diagnosed = packet(changed, (), (), diagnostic=True)
    assert int(diagnosed[2] >> 8) & int(reason)
    assert int(diagnosed[2] & 255) == int(ordinary[2]) == 32
    for plain, evidence in zip(ordinary[:2], diagnosed[:2], strict=True):
        for name in plain:
            np.testing.assert_array_equal(plain[name], evidence[name])


def test_restart_repack_success_has_exact_zero_mask(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
) -> None:
    layout, raw = restart_raw
    old = jax.tree_util.tree_map(lambda value: value[None], raw)
    cells = jnp.where(old.mesh.cell_ids >= 0, 0, -1).astype(jnp.int32)
    vertices = jnp.where(old.mesh.vertex_ids >= 0, 0, -1).astype(jnp.int32)
    result = _repack(layout, 1, old, cells, vertices, jnp.zeros_like(cells), (), ())
    np.testing.assert_array_equal(result[5], [0])
    np.testing.assert_array_equal(result[6], [0])
    np.testing.assert_array_equal(result[0].mesh.cell_ids, old.mesh.cell_ids)
    np.testing.assert_array_equal(result[0].mesh.vertex_ids, old.mesh.vertex_ids)
    assert int(packet(raw, (), (), diagnostic=True)[2]) == 0


@pytest.mark.parametrize(
    "reason",
    (
        SimplexRestartInvalidReason.CELL_AUTHORITY,
        SimplexRestartInvalidReason.VERTEX_AUTHORITY,
        SimplexRestartInvalidReason.CELL_OWNER_RANGE,
        SimplexRestartInvalidReason.VERTEX_OWNER_RANGE,
        SimplexRestartInvalidReason.SAVED_STATUS,
        SimplexRestartInvalidReason.FOREST_ROOT,
        SimplexRestartInvalidReason.PROPOSED_OWNER_RANGE,
    ),
)
def test_restart_repack_authority_and_forest_reason(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
    reason: SimplexRestartInvalidReason,
) -> None:
    layout, raw = restart_raw
    old = jax.tree_util.tree_map(lambda value: value[None], raw)
    cells = jnp.where(old.mesh.cell_ids >= 0, 0, -1).astype(jnp.int32)
    vertices = jnp.where(old.mesh.vertex_ids >= 0, 0, -1).astype(jnp.int32)
    requested = jnp.zeros_like(cells)
    match reason:
        case SimplexRestartInvalidReason.CELL_AUTHORITY | SimplexRestartInvalidReason.CELL_OWNER_RANGE:
            cells = cells.at[0, 0].set(1)
        case SimplexRestartInvalidReason.VERTEX_AUTHORITY | SimplexRestartInvalidReason.VERTEX_OWNER_RANGE:
            vertices = vertices.at[0, 0].set(1)
        case SimplexRestartInvalidReason.SAVED_STATUS:
            old = eqx.tree_at(lambda state: state.clocks, old, old.clocks.at[0, 2].set(1))
        case SimplexRestartInvalidReason.FOREST_ROOT:
            old = eqx.tree_at(lambda state: state.parents, old, old.parents.at[0, 0].set(2))
        case SimplexRestartInvalidReason.PROPOSED_OWNER_RANGE:
            requested = requested.at[0, 0].set(1)
    result = _repack(layout, 1, old, cells, vertices, requested, (), ())
    assert int(result[5][0]) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    assert int(result[6][0]) & int(reason)
    validate_restart_invalid_reasons(result[5], result[6])


def test_restart_reason_summary_high_bits_do_not_overcount_or_overflow() -> None:
    known = sum(int(reason) for reason in SimplexRestartInvalidReason)
    masks = jnp.full((1024,), known, dtype=jnp.int64)
    mask, counts, first = _invalid_reason_summary(masks)
    assert int(mask) == known
    assert mask.dtype == jnp.int64
    np.testing.assert_array_equal(counts, np.full(len(SimplexRestartInvalidReason), 1024, dtype=np.int32))
    np.testing.assert_array_equal(first, np.zeros(len(SimplexRestartInvalidReason), dtype=np.int32))


@pytest.mark.skipif(jax.device_count() < 2, reason="requires two real JAX devices")
def test_restart_reason_summary_sharded_rank_reduction() -> None:
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    low = int(SimplexRestartInvalidReason.CELL_AUTHORITY)
    high = int(SimplexRestartInvalidReason.VERTEX_COUNT)
    placement = NamedSharding(
        Mesh(np.asarray(jax.devices()[:2]), ("rank",)), PartitionSpec("rank")
    )
    masks = jax.device_put(
        np.asarray([0, high, low, high | low, 0, high], dtype=np.int64), placement
    )
    mask, counts, first = jax.device_get(_invalid_reason_summary(masks))
    assert int(mask) == low | high
    expected_counts = np.zeros(len(SimplexRestartInvalidReason), dtype=np.int32)
    expected_counts[0], expected_counts[32] = 2, 3
    expected_first = np.full(len(SimplexRestartInvalidReason), -1, dtype=np.int32)
    expected_first[0], expected_first[32] = 2, 1
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(first, expected_first)


@pytest.mark.parametrize("role", tuple(SimplexRestartRecordRole))
def test_duplicate_witness_every_record_field(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
    role: SimplexRestartRecordRole,
) -> None:
    _, raw = restart_raw
    cell, vertex, _ = packet(
        raw, (jnp.zeros((2, 2), jnp.uint32),), (jnp.zeros((4, 2), jnp.float64),)
    )
    source = cell if role == SimplexRestartRecordRole.CELL else vertex
    source = {name: value[:1] for name, value in source.items()}
    source["_source_rank"] = jnp.asarray([0], jnp.int32)
    other = dict(source)
    other["_source_rank"] = jnp.asarray([1], jnp.int32)
    selected = jnp.ones(1, jnp.bool_)
    fields = sorted(name for name in source if name != "_source_rank")
    np.testing.assert_array_equal(_packet_conflict(source, other, selected, role), -1)
    assert int(_merge_packet(source, other, selected)[1]) == 0
    for name in fields:
        if name == "ids":
            continue
        changed = dict(other)
        value = changed[name]
        changed[name] = ~value if value.dtype == jnp.bool_ else value + 1
        expected = [
            int(role), int(source["ids"][0]), fields.index(name), 0,
            int(SimplexRestartRecordDtype[value.dtype.name.upper()]), 0, 1, 1,
        ]
        for left, right in ((source, changed), (changed, source)):
            witness = _packet_conflict(left, right, selected, role)
            np.testing.assert_array_equal(witness, expected)
            _, flags = _merge_packet(left, right, selected, diagnostic=True)
            assert int(flags & 255) == int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
            validate_restart_duplicate_conflicts((flags >> 8)[None], witness[None], 2)
            with pytest.raises(ValueError, match="closed schema"):
                validate_restart_duplicate_conflicts(
                    (flags >> 8)[None], witness.at[7].set(99)[None], 2
                )


def test_duplicate_witness_exact_equality_nan_and_signed_zero() -> None:
    source = {
        "ids": jnp.asarray([7], jnp.int64),
        "coordinates": jnp.asarray([[-0.0, float("nan")]], jnp.float64),
        "_source_rank": jnp.asarray([0], jnp.int32),
    }
    incoming = {**source, "coordinates": source["coordinates"].at[0, 0].set(0.0)}
    witness = _packet_conflict(
        source, incoming, jnp.ones(1, bool), SimplexRestartRecordRole.VERTEX
    )
    assert int(witness[3]) == 1
    assert int(_merge_packet(source, incoming, jnp.ones(1, bool))[1]) == 32


def test_duplicate_witness_permutation_and_rank_determinism() -> None:
    source = {
        "ids": jnp.asarray([7, 3], jnp.int64),
        "coordinates": jnp.zeros((2, 2), jnp.float64),
        "_source_rank": jnp.zeros(2, jnp.int32),
    }
    incoming = {
        "ids": jnp.asarray([3, 7], jnp.int64),
        "coordinates": jnp.ones((2, 2), jnp.float64),
        "_source_rank": jnp.ones(2, jnp.int32),
    }
    expected = [2, 3, 0, 0, 5, 0, 1, 1]
    for reverse_source in (False, True):
        for reverse_incoming in (False, True):
            left = {key: value[::-1] if reverse_source else value for key, value in source.items()}
            right = {key: value[::-1] if reverse_incoming else value for key, value in incoming.items()}
            witnesses = jax.vmap(
                lambda selected: _packet_conflict(left, right, selected, SimplexRestartRecordRole.VERTEX)
            )(jnp.ones((6, 2), jnp.bool_))
            np.testing.assert_array_equal(witnesses, np.tile(expected, (6, 1)))


@pytest.mark.parametrize("activity", ((True, False), (False, True), (False, False), (True, True)))
@pytest.mark.parametrize("owner", (0, 1))
@pytest.mark.parametrize("events", ((3, 1), (-1, 1), (1, 1)))
def test_restart_shared_vertex_partition_activity(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
    activity: tuple[bool, bool],
    owner: int,
    events: tuple[int, int],
) -> None:
    layout, raw = restart_raw
    old = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), raw)
    masks = (old.mesh.vertex_ids >= 0) & jnp.asarray(activity)[:, None]
    old = eqx.tree_at(lambda state: state.mesh.vertex_active, old, masks)
    removal = jnp.where((old.mesh.vertex_ids >= 0) & ~masks, jnp.asarray(events)[:, None], -1)
    old = eqx.tree_at(
        lambda state: (state.vertex_removal, state.clocks), old,
        (removal.astype(jnp.int32), old.clocks.at[:, 1].set(jnp.asarray([4, 2], jnp.int32))),
    )
    cells = jnp.where(old.mesh.cell_ids >= 0, owner, -1).astype(jnp.int32)
    vertices = jnp.where(old.mesh.vertex_ids >= 0, owner, -1).astype(jnp.int32)
    result = _repack(layout, 1, old, cells, vertices, jnp.zeros_like(cells), (), ())
    np.testing.assert_array_equal(result[5], [0])
    np.testing.assert_array_equal(result[6], [0])
    np.testing.assert_array_equal(result[7], -np.ones((1, 8), dtype=np.int64))
    expected = (result[0].mesh.vertex_ids >= 0) & any(activity)
    np.testing.assert_array_equal(result[0].mesh.vertex_active, expected)
    np.testing.assert_array_equal(old.mesh.vertex_active, masks)
    expected_removal = jnp.where((result[0].mesh.vertex_ids >= 0) & ~expected, 1, -1)
    np.testing.assert_array_equal(result[0].vertex_removal, expected_removal)
    np.testing.assert_array_equal(result[0].clocks[:, 1], [4])
    np.testing.assert_array_equal(old.vertex_removal, removal)
    # Renumbering source ranks preserves scientific results and refusal evidence.
    permuted = jax.tree_util.tree_map(lambda value: value[::-1], old)
    permuted_cells = jnp.where(cells[::-1] >= 0, 1 - cells[::-1], -1)
    permuted_vertices = jnp.where(vertices[::-1] >= 0, 1 - vertices[::-1], -1)
    replay = _repack(
        layout, 1, permuted, permuted_cells, permuted_vertices,
        jnp.zeros_like(cells), (), (),
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(result[0]),
        jax.tree_util.tree_leaves(replay[0]), strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(replay[6], result[6])


@pytest.mark.parametrize("corruption", ("missing_owner", "owner_range", "active_id", "scientific_bool"))
def test_restart_partition_activity_does_not_hide_invalid_sources(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
    corruption: str,
) -> None:
    layout, raw = restart_raw
    old = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), raw)
    old = eqx.tree_at(
        lambda state: state.mesh.vertex_active, old,
        old.mesh.vertex_active.at[1].set(False),
    )
    cells = jnp.where(old.mesh.cell_ids >= 0, 0, -1).astype(jnp.int32)
    vertices = jnp.where(old.mesh.vertex_ids >= 0, 0, -1).astype(jnp.int32)
    history = ()
    reason = SimplexRestartInvalidReason.VERTEX_AUTHORITY
    if corruption == "missing_owner":
        vertices = vertices.at[0, 0].set(1)
    elif corruption == "owner_range":
        vertices = vertices.at[:, 0].set(2)
        reason = SimplexRestartInvalidReason.VERTEX_OWNER_RANGE
    elif corruption == "active_id":
        old = eqx.tree_at(
            lambda state: state.mesh.vertex_active, old,
            old.mesh.vertex_active.at[0, -1].set(True),
        )
        reason = SimplexRestartInvalidReason.REQUIRED_VERTEX_ID
    else:
        history = (jnp.zeros_like(old.mesh.vertex_active).at[1, 0].set(True),)
        reason = SimplexRestartInvalidReason.DUPLICATE_RECORD_CONFLICT
    result = _repack(
        layout, 1, old, cells, vertices, jnp.zeros_like(cells), (), history,
    )
    assert int(result[5][0]) == int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    assert int(result[6][0]) & int(reason)
    validate_restart_duplicate_conflicts(result[6], result[7], 2)


def test_source_vertex_activity_requires_boolean_id_aligned_masks() -> None:
    identifiers = jnp.asarray([[2, 7], [7, 2]], jnp.int64)
    active = jnp.asarray([[False, True], [False, False]])
    np.testing.assert_array_equal(
        _source_vertex_activity(identifiers, active), [[False, True], [True, False]],
    )
    for invalid in (active[:, :1], active.astype(jnp.int32)):
        with pytest.raises(ValueError, match="Boolean with the source ID shape"):
            _source_vertex_activity(identifiers, invalid)


@pytest.mark.parametrize(
    ("activity", "events", "clocks"),
    (
        ((False, False), (-1, -1), (0, 0)),
        ((False, False), (-2, 1), (3, 3)),
        ((False, False), (3, 1), (3, 3)),
        ((False, False), (0, 1), (-1, 3)),
        ((True, False), (0, 1), (3, 3)),
        ((True, False), (-1, 3), (3, 3)),
    ),
)
def test_restart_removal_history_refuses_invalid_events(
    restart_raw: tuple[AdaptiveSimplexLayout, AdaptiveSimplexState],
    activity: tuple[bool, bool],
    events: tuple[int, int],
    clocks: tuple[int, int],
) -> None:
    layout, raw = restart_raw
    old = jax.tree_util.tree_map(lambda value: jnp.stack((value, value)), raw)
    present = old.mesh.vertex_ids >= 0
    old = eqx.tree_at(
        lambda state: (state.mesh.vertex_active, state.vertex_removal, state.clocks),
        old,
        (
            present & jnp.asarray(activity)[:, None],
            jnp.where(present, jnp.asarray(events, jnp.int32)[:, None], -1),
            old.clocks.at[:, 1].set(jnp.asarray(clocks, jnp.int32)),
        ),
    )
    cells = jnp.where(old.mesh.cell_ids >= 0, 0, -1).astype(jnp.int32)
    vertices = jnp.where(present, 0, -1).astype(jnp.int32)
    result = _repack(layout, 1, old, cells, vertices, jnp.zeros_like(cells), (), ())
    assert int(result[5][0]) == int(AdaptiveSimplexStatus.INVALID_GEOMETRY)
    assert int(result[6][0]) == int(SimplexRestartInvalidReason.VERTEX_REMOVAL_HISTORY)
    np.testing.assert_array_equal(result[7], -np.ones((1, 8), dtype=np.int64))

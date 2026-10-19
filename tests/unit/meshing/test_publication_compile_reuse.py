# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Publication table kernels reuse compilation with changing dynamic contents."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellGeometryStorageProjection,
    CellMesh,
)
from phydrax.discretization._adaptive_simplex import MaskedSimplexMesh
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.geometry._collective_domain_coverage import _epoch_cell_inventory
from phydrax.meshing import _device_adaptation as adaptation_owner
from phydrax.meshing._collective_geometry import _geometry_bank_membership
from phydrax.meshing._collective_organization import _original_vertex_ancestry_values
from phydrax.meshing._device_adaptation import (
    _compact_logical_values,
    _first_appearance_order,
    _logical_boundary_masks,
    _logical_multiblock_topology_id,
    _LogicalPresentation,
    canonical_rows,
)
from phydrax.meshing._distribution import lexicographic_bound
from phydrax.meshing._entity_keys import entity_keys, key_positions
from phydrax.meshing._publication_lowering import (
    _addressable_arrays,
    _project,
    addressable,
)
from tests._support import cell_meshes
from tests._support.forced_devices import run_on_forced_devices
from tools.compile_counter import CompileCounter


def test_identical_publication_tables_compile_nothing_when_warm() -> None:
    keys = jax.device_put(np.asarray(((2,), (0,), (1,), (2,)), dtype=np.int64))
    valid = jax.device_put(np.asarray((True, True, True, False), dtype=np.bool_))
    vertices = jax.device_put(np.asarray(((0,), (1,), (2,), (-1,)), dtype=np.int64))
    edges = jax.device_put(np.asarray(((0, 1), (0, 2), (1, 2), (-1, -1)), dtype=np.int64))
    cells = jax.device_put(np.asarray(((0, 1, 2), (0, 0, 0)), dtype=np.int64))
    tables = (vertices, edges, cells[:, :1])

    def publish() -> tuple[
        jax.Array,
        jax.Array,
        int,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        tuple[jax.Array, ...],
    ]:
        rows, order, count = canonical_rows(keys, valid)
        searchable = vertices[:3]
        return (
            rows,
            order,
            count,
            key_positions(searchable, keys),
            lexicographic_bound(searchable, keys, False),
            lexicographic_bound(searchable, keys, True),
            _first_appearance_order(keys, valid, vertices, 3),
            _logical_boundary_masks(cells, 1, tables, (3, 3, 1)),
        )

    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(publish())
    with CompileCounter() as warm:
        second = jax.block_until_ready(publish())
    assert cold.count > 0
    assert warm.count == 0
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first[0], ((0,), (1,), (2,), (-1,)))
    np.testing.assert_array_equal(first[3], (2, 0, 1, 2))
    np.testing.assert_array_equal(first[4], (2, 0, 1, 2))
    np.testing.assert_array_equal(first[5], (3, 1, 2, 3))
    assert first[2] == 3
    for mask, expected in zip(
        first[7],
        ((True, True, True, False), (True, True, True, False), (False, False)),
        strict=True,
    ):
        np.testing.assert_array_equal(mask, expected)


def test_publication_bounds_do_not_specialize_on_dynamic_selector() -> None:
    table = jax.device_put(np.asarray(((0,), (1,), (2,)), dtype=np.int64))
    queries = jax.device_put(np.asarray(((1,), (3,)), dtype=np.int64))
    jax.clear_caches()
    jax.block_until_ready(lexicographic_bound(table, queries, False))
    with CompileCounter() as compiler:
        result = jax.block_until_ready(lexicographic_bound(table, queries, True))
    assert compiler.count == 0
    np.testing.assert_array_equal(result, (2, 3))


def test_publication_tables_keep_active_counts_dynamic() -> None:
    keys = jax.device_put(np.asarray(((2,), (0,), (1,), (2,)), dtype=np.int64))
    first_active = jax.device_put(np.asarray((True, True, True, False), dtype=np.bool_))
    next_active = jax.device_put(np.asarray((False, True, True, False), dtype=np.bool_))
    jax.clear_caches()
    assert canonical_rows(keys, first_active)[2] == 3
    with CompileCounter() as compiler:
        rows, _, count = canonical_rows(keys, next_active)
    assert compiler.count == 0
    assert count == 2
    np.testing.assert_array_equal(rows, ((0,), (1,), (-1,), (-1,)))


@pytest.mark.parametrize("missing", (False, True), ids=("foreign-key", "missing-entity"))
def test_first_appearance_retains_publication_refusals(missing: bool) -> None:
    table = jax.device_put(np.asarray(((0,), (1,), (-1,)), dtype=np.int64))
    keys = jax.device_put(np.asarray(((0,), (0 if missing else 2,)), dtype=np.int64))
    valid = jax.device_put(np.asarray((True, True), dtype=np.bool_))
    with pytest.raises(
        ValueError,
        match="missing a logical simplex entity" if missing else "absent entity key",
    ):
        _first_appearance_order(keys, valid, table, 2)


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
@pytest.mark.parametrize("source_kind", ("certified-mesh", "native-domain"))
def test_second_identical_collective_publication_compiles_zero(source_kind: str) -> None:
    receipt = run_on_forced_devices(
        "tests._support.distributed_meshing.publication_compile",
        "repeated_publication",
        (source_kind,),
        devices=2,
        env={},
    )
    assert receipt["cold_count"] > 0
    assert receipt["warm_count"] == 0, receipt["warm_callsites"]
    assert receipt["raw_partition_warm_count"] == 0
    assert len(receipt["result_ids"]) == 2


def test_ancestry_publication_keeps_rank_and_epoch_identifiers_dynamic() -> None:
    first_ids = jax.device_put(np.asarray((11, 23, 47, 59), dtype=np.int64))
    next_ids = jax.device_put(np.asarray((101, 103, 107, 109), dtype=np.int64))
    parents = jax.device_put(
        np.asarray(((-1, -1), (-1, -1), (-1, -1), (0, 1)), dtype=np.int32)
    )
    cursor = jax.device_put(np.asarray(4, dtype=np.int64))
    jax.clear_caches()
    jax.block_until_ready(
        _original_vertex_ancestry_values(first_ids, parents, cursor, 3, None)
    )
    with CompileCounter() as compiler:
        ids, weights, passed = jax.block_until_ready(
            _original_vertex_ancestry_values(next_ids, parents, cursor, 3, None)
        )
    assert compiler.count == 0
    assert bool(passed)
    np.testing.assert_array_equal(ids[3], (101, 103, -1))
    np.testing.assert_array_equal(weights[3], (0.5, 0.5, 0.0))


def test_geometry_bank_projection_compiles_zero_when_warm() -> None:
    mesh = cell_meshes.unit_square()
    geometry = CellGeometrySpec.affine(mesh)
    storage = cell_meshes.storage(mesh, geometry)
    banks = dict(storage.logical_arrays)
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in mesh.blocks)
    )
    queries = np.stack((cell_ids, cell_ids))
    banks["closure/cell_ids"] = jax.device_put(queries)
    banks["closure/cell_valid"] = jax.device_put(np.ones(queries.shape, dtype=np.bool_))
    arrays = tuple(sorted(banks.items()))
    identity = cell_geometry_id(geometry)
    count = geometry.coordinates.shape[0]
    jax.clear_caches()
    first = CellGeometryStorageProjection(arrays, identity, count)
    with CompileCounter(capture_stack=True) as compiler:
        repeated = CellGeometryStorageProjection(arrays, identity, count)
        jax.block_until_ready(repeated.projected_arrays)
    assert compiler.count == 0, [event.to_record() for event in compiler.events]
    assert first.source_content_id == repeated.source_content_id
    for (name, left), (other, right) in zip(
        first.projected_arrays, repeated.projected_arrays, strict=True
    ):
        assert name == other
        np.testing.assert_array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        )


def test_compact_topology_keeps_owner_state_dynamic_when_warm() -> None:
    coordinates = np.asarray(
        ((-0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    mesh = MaskedSimplexMesh(
        np.stack((coordinates, coordinates)),
        np.tile(np.arange(4, dtype=np.int64), (2, 1)),
        np.ones((2, 4), dtype=np.bool_),
        np.asarray((((0, 1, 2), (0, 0, 0)), ((0, 2, 3), (0, 0, 0))), dtype=np.int32),
        np.asarray(((11, -1), (23, -1)), dtype=np.int64),
        np.asarray(((True, False), (True, False)), dtype=np.bool_),
        np.full((2, 2, 3), -1, dtype=np.int32),
    )
    first_owners = jax.device_put(np.asarray(((0, -1), (1, -1)), dtype=np.int32))
    next_owners = jax.device_put(np.asarray(((1, -1), (0, -1)), dtype=np.int32))
    jax.clear_caches()
    first = jax.block_until_ready(
        _compact_logical_values(mesh, first_owners, first_owners)
    )
    with CompileCounter() as compiler:
        repeated = jax.block_until_ready(
            _compact_logical_values(mesh, next_owners, next_owners)
        )
    assert compiler.count == 0
    assert int(first[2]) == int(repeated[2]) == 2
    assert int(first[7]) == int(repeated[7]) == 4
    np.testing.assert_array_equal(first[0], ((11,), (23,), (-1,), (-1,)))
    np.testing.assert_array_equal(first[8][:4], (0, 0, 0, 1))
    np.testing.assert_array_equal(repeated[8][:4], (0, 1, 0, 0))
    np.testing.assert_array_equal(
        np.asarray(first[9][:4]).view(np.uint64), coordinates.view(np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(first[9]).view(np.uint64), np.asarray(repeated[9]).view(np.uint64)
    )


def test_multiblock_topology_identity_preserves_serial_id_and_warm_zero() -> None:
    mesh = CellMesh(
        np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64),
        (
            CellBlock(
                "lower",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((11,), dtype=np.int64),
            ),
            CellBlock(
                "upper",
                "triangle",
                np.asarray(((0, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((23,), dtype=np.int64),
            ),
        ),
    )
    tables = []
    identities = []
    counts = tuple(mesh.entity_set(degree).count for degree in range(3))
    for degree, capacity in enumerate((8, 8, 4)):
        keys = entity_keys(mesh, degree)
        order = np.lexsort(keys.T[::-1])
        table = np.full((capacity, keys.shape[1]), -1, dtype=np.int64)
        table[: keys.shape[0]] = keys[order]
        ids = np.full(capacity, -1, dtype=np.int64)
        ids[: keys.shape[0]] = np.asarray(mesh.entity_set(degree).entity_ids)[order]
        tables.append(jax.device_put(table))
        identities.append(jax.device_put(ids))
    cell_rows = jax.device_put(
        np.asarray(((0, 1, 2), (0, 2, 3), (0, 0, 0), (0, 0, 0)), dtype=np.int64)
    )
    presentation = _LogicalPresentation(
        ("lower", "upper"),
        ("triangle", "triangle"),
        jax.device_put(np.asarray((0, 1, -1, -1), dtype=np.int64)),
        (1, 1),
        jax.device_put(np.arange(4, dtype=np.int64)),
    )
    args = (
        2,
        tables[0],
        tables[-1],
        cell_rows,
        tuple(tables),
        tuple(identities),
        counts,
        presentation,
    )
    jax.clear_caches()
    identity = _logical_multiblock_topology_id(*args)
    with CompileCounter(capture_stack=True) as compiler:
        repeated = _logical_multiblock_topology_id(*args)
    assert compiler.count == 0, [event.to_record() for event in compiler.events]
    assert identity == repeated == mesh.topology_id


def test_compile_counter_distinguishes_static_output_shape_signatures() -> None:
    values = jax.device_put(np.arange(8, dtype=np.int32))
    jax.clear_caches()
    with CompileCounter(capture_equations=True) as compiler:
        first = jax.lax.broadcast_in_dim(values, (8, 1), (0,))
        second = jax.lax.broadcast_in_dim(values, (1, 8), (1,))
        jax.block_until_ready((first, second))
    assert compiler.count == 2
    left, right = compiler.events
    assert left.signature == right.signature
    assert left.lowering_equations is not None
    assert right.lowering_equations is not None
    assert left.lowering_equations != right.lowering_equations
    assert left.lowering_layouts != right.lowering_layouts
    np.testing.assert_array_equal(first[:, 0], values)
    np.testing.assert_array_equal(second[0], values)


@jax.jit
def _used_signature_values(values: jax.Array, unused: jax.Array) -> jax.Array:
    return values + 1


def test_compile_counter_retains_dead_frontend_input_shapes() -> None:
    values = jax.device_put(np.arange(8, dtype=np.int32))
    first_unused = jax.device_put(np.zeros(2, dtype=np.int32))
    second_unused = jax.device_put(np.zeros(3, dtype=np.int32))
    with CompileCounter(capture_equations=True) as compiler:
        first = _used_signature_values(values, first_unused)
        second = _used_signature_values(values, second_unused)
        jax.block_until_ready((first, second))
    assert compiler.count == 2
    left, right = compiler.events
    assert left.signature == right.signature
    assert left.lowering_equations == right.lowering_equations
    assert left.lowering_layouts != right.lowering_layouts
    np.testing.assert_array_equal(first, values + 1)
    np.testing.assert_array_equal(second, first)


def test_geometry_bank_membership_keeps_active_count_dynamic() -> None:
    original_ids = jax.device_put(np.asarray((9, 7), dtype=np.int64))
    sources = jax.device_put(np.asarray((7, 9, 19, -1), dtype=np.int64))
    jax.clear_caches()
    first = jax.block_until_ready(_geometry_bank_membership(original_ids, sources, 3))
    with CompileCounter() as compiler:
        repeated = jax.block_until_ready(
            _geometry_bank_membership(original_ids, sources, 1)
        )
    assert compiler.count == 0
    np.testing.assert_array_equal(first[3], (True, True, False, False))
    np.testing.assert_array_equal(repeated[3], (True, False, False, False))
    assert int(first[4]) == 2
    assert int(repeated[4]) == 1


def test_epoch_cell_inventory_keeps_expected_count_dynamic() -> None:
    identifiers = jax.device_put(np.asarray((11, 7, -1, -1), dtype=np.int64))
    active = jax.device_put(np.asarray((True, True, False, False), dtype=np.bool_))
    jax.clear_caches()
    order, passed = jax.block_until_ready(_epoch_cell_inventory(identifiers, active, 2))
    with CompileCounter() as compiler:
        repeated_order, rejected = jax.block_until_ready(
            _epoch_cell_inventory(identifiers, active, 3)
        )
    assert compiler.count == 0
    assert bool(passed)
    assert not bool(rejected)
    np.testing.assert_array_equal(order, (1, 0, 2, 3))
    np.testing.assert_array_equal(repeated_order, order)


def test_publication_projection_keeps_accepted_counts_dynamic() -> None:
    tables = tuple(
        jax.device_put(np.asarray(rows, dtype=np.int64))
        for rows in (
            ((0,), (1,), (2,), (3,)),
            ((0, 1), (0, 2), (1, 2), (-1, -1)),
            ((11,), (13,)),
        )
    )
    identities = tuple(
        jax.device_put(np.asarray(values, dtype=np.int64))
        for values in ((0, 1, 2, 3), (0, 1, 2, -1), (11, 13))
    )
    owners = tuple(
        jax.device_put(np.zeros(value.shape, dtype=np.int32)) for value in identities
    )
    arrays = {
        "closure/cell_vertices": jax.device_put(
            np.asarray((((0, 1, 2),), ((0, 1, 2),)), dtype=np.int64)
        ),
        "closure/cell_ids": jax.device_put(np.asarray(((11,), (11,)), dtype=np.int64)),
        "closure/cell_valid": jax.device_put(np.ones((2, 1), dtype=np.bool_)),
        "closure/cell_owner": jax.device_put(np.zeros((2, 1), dtype=np.int32)),
        "coordinates": jax.device_put(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0)), dtype=np.float64)
        ),
        "marker/membership": jax.device_put(
            np.asarray((True, False, True, False), dtype=np.bool_)
        ),
    }
    jax.clear_caches()
    first, valid = jax.block_until_ready(
        _project(
            arrays,
            tables,
            identities,
            owners,
            counts=(3, 3, 1),
            vertex_capacity=4,
            organization_degrees=(("marker/membership", 0),),
        )
    )
    with CompileCounter() as compiler:
        repeated, next_valid = jax.block_until_ready(
            _project(
                arrays,
                tables,
                identities,
                owners,
                counts=(4, 3, 2),
                vertex_capacity=4,
                organization_degrees=(("marker/membership", 0),),
            )
        )
    assert compiler.count == 0
    assert bool(valid) and bool(next_valid)
    for name, value in first.items():
        np.testing.assert_array_equal(repeated[name], value)


def test_retained_protection_kernels_preserve_codes_and_reuse_dynamic_scopes() -> None:
    vertex_ids = jax.device_put(
        np.asarray(((10, 20, 30, 40, -1, -1, -1, -1),), dtype=np.int64)
    )
    cells = jax.device_put(
        np.asarray((((0, 1, 2), (1, 3, 2), (0, 0, 0), (0, 0, 0)),), dtype=np.int32)
    )
    active = jax.device_put(np.asarray(((True, True, False, False),)))
    flags = jax.device_put(
        np.asarray(((True, False, False, False, False, False, False, False),))
    )
    identifiers = jax.device_put(
        np.asarray((400, 100, 300, 200, -1, -1, -1, -1), dtype=np.int64)
    )
    source_keys = jax.device_put(
        np.asarray(
            (
                (10, 20),
                (20, 30),
                (20, 40),
                (30, 40),
                (-1, -1),
                (-1, -1),
                (-1, -1),
                (-1, -1),
            ),
            dtype=np.int64,
        )
    )
    selected = jax.device_put(np.asarray((100, 200), dtype=np.int64))
    different = jax.device_put(np.asarray((300, 400), dtype=np.int64))
    count = jax.device_put(np.asarray(4, dtype=np.int64))
    capacity = jax.device_put(np.asarray(8, dtype=np.int64))
    sentinel = np.iinfo(np.int64).max
    prior = jax.device_put(
        np.asarray(((1, sentinel, sentinel, sentinel),), dtype=np.int64)
    )

    def prepare(selection: jax.Array) -> tuple[jax.Array, ...]:
        slots, edges, edge_flags = adaptation_owner._retained_protection_edges(
            vertex_ids, cells
        )
        keys, missing = adaptation_owner._retained_scope_keys(
            identifiers, count, selection, source_keys
        )
        protected = adaptation_owner._retained_vertex_scope_flags(vertex_ids, flags, keys)
        edge_flags = adaptation_owner._retained_edge_scope_flags(edges, edge_flags, keys)
        codes, used = adaptation_owner._retained_protection_values(
            slots, edge_flags, active, prior, capacity
        )
        return (
            protected,
            adaptation_owner._retained_protection_bucket(codes, capacity=4),
            used,
            missing,
        )

    expected = jax.block_until_ready(prepare(selected))
    with CompileCounter() as compiler:
        identical = jax.block_until_ready(prepare(selected))
        changed = jax.block_until_ready(prepare(different))
    assert compiler.count == 0
    np.testing.assert_array_equal(
        expected[0], ((True, True, True, True, False, False, False, False),)
    )
    # Stable unique-prefix packing retains duplicate tail bytes in the bucket.
    np.testing.assert_array_equal(expected[1], ((1, 10, 19, 10),))
    assert int(expected[2]) == 3
    assert not bool(expected[3])
    for first, second in zip(expected, identical, strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(changed[1], ((1, 11, 1, sentinel),))
    assert int(changed[2]) == 2
    assert not bool(changed[3])


@pytest.mark.parametrize("valid_count", (0, 1, 2))
def test_retained_scope_keys_exclude_padding_and_allow_real_maximum_ids(
    valid_count: int,
) -> None:
    sentinel = np.iinfo(np.int64).max
    identifiers = jax.device_put(np.asarray((17, sentinel, -1, -1), dtype=np.int64))
    keys = jax.device_put(
        np.asarray(((1, 2), (3, 4), (-1, -1), (-1, -1)), dtype=np.int64)
    )
    selected = jax.device_put(np.asarray((sentinel,), dtype=np.int64))
    found, missing = adaptation_owner._retained_scope_keys(
        identifiers,
        jax.device_put(np.asarray(valid_count, dtype=np.int64)),
        selected,
        keys,
    )
    assert bool(missing) == (valid_count != 2)
    if valid_count == 2:
        np.testing.assert_array_equal(found, ((3, 4),))


def test_retained_coordinate_projection_reuses_dynamic_count_and_preserves_bits() -> None:
    identifiers = jax.device_put(np.asarray((10, 20, 30, -1), dtype=np.int64))
    vertices = jax.device_put(np.asarray(((20, 30, 99, -1),), dtype=np.int64))
    active = jax.device_put(np.asarray(((True, True, True, False),)))
    source = jax.device_put(
        np.asarray(((1.0, 2.0), (-0.0, 4.0), (5.0, -0.0), (0.0, 0.0)), dtype=np.float64)
    )
    prior = jax.device_put(
        np.asarray((((9.0, 8.0), (7.0, 6.0), (5.0, 4.0), (-0.0, 2.0)),), dtype=np.float64)
    )
    two = jax.device_put(np.asarray(2, dtype=np.int64))
    three = jax.device_put(np.asarray(3, dtype=np.int64))

    def coordinates(count: jax.Array) -> jax.Array:
        return adaptation_owner._canonical_retained_coordinate_values(
            identifiers, count, vertices, active, source, prior
        )

    first = coordinates(two).block_until_ready()
    with CompileCounter() as compiler:
        identical = coordinates(two).block_until_ready()
        changed = coordinates(three).block_until_ready()
    assert compiler.count == 0
    expected = np.asarray((((-0.0, 4.0), (7.0, 6.0), (5.0, 4.0), (-0.0, 2.0)),))
    np.testing.assert_array_equal(
        np.asarray(first).view(np.uint64), expected.view(np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(identical).view(np.uint64), expected.view(np.uint64)
    )
    expected[0, 1] = (5.0, -0.0)
    np.testing.assert_array_equal(
        np.asarray(changed).view(np.uint64), expected.view(np.uint64)
    )


def test_addressable_publication_arrays_fuse_local_projection_and_reuse_indices() -> None:
    bits = np.asarray(
        (0, 0x8000000000000000, 0x7FF8000000000001, 0x7FF8000000000002),
        dtype=np.uint64,
    )
    floating = np.tile(bits.view(np.float64), (2, 1))
    values = (
        jax.device_put(floating),
        jax.device_put(np.arange(12, dtype=np.int64).reshape((2, 3, 2))),
        jax.device_put(np.asarray(((True, False), (False, True)), dtype=np.bool_)),
    )
    reference = tuple(addressable(value, 1) for value in values)
    jax.clear_caches()
    with CompileCounter() as cold:
        jax.block_until_ready(_addressable_arrays(values, 0))
    with CompileCounter() as warm:
        result = jax.block_until_ready(_addressable_arrays(values, 1))
    assert cold.count == 1
    assert warm.count == 0
    for expected, actual in zip(reference, result, strict=True):
        assert actual.sharding == expected.sharding
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    with pytest.raises(ValueError, match="process-addressable"):
        _addressable_arrays(values, 2)

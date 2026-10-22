# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Publication table kernels reuse compilation with changing dynamic contents."""

from __future__ import annotations

from itertools import combinations

import jax
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellGeometryStorageProjection,
    CellMesh,
    CellMeshStorage,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    form_element,
    lagrange_element,
)
from phydrax.discretization._adaptive_simplex import MaskedSimplexMesh
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization.fem._de_rham import _boundary_entity_closure
from phydrax.geometry._collective_domain_coverage import _epoch_cell_inventory
from phydrax.meshing import _device_adaptation as adaptation_owner
from phydrax.meshing import _result as result_owner
from phydrax.meshing._collective_geometry import _geometry_bank_membership
from phydrax.meshing._collective_organization import _original_vertex_ancestry_values
from phydrax.meshing._device_adaptation import (
    _compact_logical_values,
    _first_appearance_order,
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
    logical_boundary_masks,
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
            logical_boundary_masks(cells, 1, tables, (3, 3, 1)),
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
    assert receipt["constructor_fields_preserved"]


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
            ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)),
            ((11,), (13,)),
        )
    )
    identities = tuple(
        jax.device_put(np.asarray(values, dtype=np.int64))
        for values in ((0, 1, 2, 3), (0, 1, 2, 3, 4), (11, 13))
    )
    owners = tuple(
        jax.device_put(np.zeros(value.shape, dtype=np.int32)) for value in identities
    )
    arrays = {
        "cell_vertices": jax.device_put(
            np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int64)
        ),
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
                counts=(4, 5, 2),
                vertex_capacity=4,
                organization_degrees=(("marker/membership", 0),),
            )
        )
    assert compiler.count == 0
    assert bool(valid) and bool(next_valid)
    for name, value in first.items():
        if name == "entity/1/physical_boundary":
            np.testing.assert_array_equal(value, ((True, True, True),) * 2)
            np.testing.assert_array_equal(repeated[name], ((True, True, False),) * 2)
        else:
            np.testing.assert_array_equal(repeated[name], value)


def test_restart_publication_fuses_global_incidence_and_retains_refusals() -> None:
    ids = jax.device_put(np.asarray((11, 23, -1), dtype=np.int64))
    cells = jax.device_put(np.asarray(((0, 1, 2), (1, 3, 2), (-1, -1, -1)), dtype=np.int64))
    facets = jax.device_put(
        np.asarray(((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (-1, -1)), dtype=np.int64)
    )
    targets = jax.device_put(np.asarray(((11, 23, -1),), dtype=np.int64))
    first_active = jax.device_put(np.asarray(((True, False, False),)))
    next_active = jax.device_put(np.asarray(((True, True, False),)))
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            adaptation_owner._restart_publication_values(
                ids, cells, facets, 1, 3, targets, first_active
            )
        )
    assert cold.count == 1
    np.testing.assert_array_equal(first[0], (((True, True, True), (False,) * 3, (False,) * 3),))
    np.testing.assert_array_equal(first[1], ((11, -1, -1),))
    assert bool(first[2]) and bool(first[3])
    with CompileCounter() as warm:
        second = jax.block_until_ready(
            adaptation_owner._restart_publication_values(
                ids, cells, facets, 2, 5, targets, next_active
            )
        )
    assert warm.count == 0
    np.testing.assert_array_equal(
        second[0], (((False, True, True), (True, False, True), (False,) * 3),)
    )
    np.testing.assert_array_equal(second[1], ((11, 23, -1),))
    assert bool(second[2]) and bool(second[3])
    missing_cells = adaptation_owner._restart_publication_values(
        ids, cells, facets, 2, 5, targets.at[0, 1].set(99), next_active
    )
    assert not bool(missing_cells[2])
    missing_facets = adaptation_owner._restart_publication_values(
        ids, cells, facets, 2, 2, targets, next_active
    )
    assert not bool(missing_facets[3])


@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("order", ((1, 5, 4), (4, 1, 5)))
@pytest.mark.parametrize("family", ("H1", "Hdiv", "Hcurl"))
def test_global_boundary_projection_retains_nonresident_physical_facets(
    degree: int, order: tuple[int, ...], family: str
) -> None:
    points = np.asarray([(x, y) for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)])
    vertex_ids = np.asarray((97, 43, 211, 19, 151, 73, 307, 131, 59), dtype=np.int64)
    cells = np.asarray(
        [
            triangle
            for lower in (0, 1, 3, 4)
            for triangle in ((lower, lower + 1, lower + 4), (lower, lower + 4, lower + 3))
        ],
        dtype=np.int32,
    )
    dense = CellMesh(
        points,
        (CellBlock("cells", "triangle", cells),),
        vertex_global_ids=vertex_ids,
    )
    tables, identities = [], []
    for dimension in range(3):
        keys = entity_keys(dense, dimension)
        indices = np.lexsort(keys.T[::-1])
        tables.append(jax.device_put(keys[indices]))
        identities.append(jax.device_put(np.asarray(dense.entity_set(dimension).entity_ids)[indices]))
    counts = tuple(entity.count for entity in dense.topology.entity_sets)
    selected = np.asarray(order)
    # This resident triangle touches the global boundary only at two vertices.
    # None of its three edges is a physical facet of the global mesh.
    cell_id = np.asarray(dense.blocks[0].global_ids)[3]
    arrays = {
        "cell_vertices": jax.device_put(vertex_ids[cells]),
        "coordinates": jax.device_put(points[np.argsort(vertex_ids)]),
        "closure/cell_vertices": jax.device_put(vertex_ids[selected][None, None, :]),
        "closure/cell_ids": jax.device_put(np.asarray(((cell_id,),))),
        "closure/cell_valid": jax.device_put(np.ones((1, 1), dtype=np.bool_)),
        "closure/cell_owner": jax.device_put(np.zeros((1, 1), dtype=np.int32)),
    }
    projected, passed = _project(
        arrays,
        tuple(tables),
        tuple(identities),
        tuple(jax.device_put(np.zeros(value.shape, dtype=np.int32)) for value in identities),
        counts=counts,
        vertex_capacity=3,
        organization_degrees=(),
    )
    assert bool(passed)
    probe = CellMesh(
        points[selected],
        (
            CellBlock(
                "cells",
                "triangle",
                np.asarray(((0, 1, 2),)),
                global_ids=np.asarray((cell_id,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids[selected],
    )
    ids, boundary = [], []
    for dimension in range(3):
        active = np.asarray(projected[f"entity/{dimension}/valid"])[0]
        keys = np.asarray(projected[f"entity/{dimension}/keys"])[0][active]
        rows = np.asarray(
            key_positions(jax.device_put(keys), jax.device_put(entity_keys(probe, dimension)))
        )
        ids.append(np.asarray(projected[f"entity/{dimension}/ids"])[0][active][rows])
        boundary.append(
            np.asarray(projected[f"entity/{dimension}/physical_boundary"])[0][active][rows]
        )
    assert not np.any(boundary[1])
    np.testing.assert_array_equal(boundary[0], selected != 4)
    storage = CellMeshStorage(
        counts,
        ids,
        tuple(np.zeros(value.shape, dtype=np.int32) for value in ids),
        partition_index=0,
        partition_count=1,
        logical_topology_id=dense.topology_id,
        logical_geometry_id=dense.geometry_id,
        evidence_id=dense.topology_id,
        logical_arrays=(("coordinates", dense.coordinates),),
        local_coordinates=probe.coordinates,
        local_blocks=probe.blocks,
        local_physical_boundary_entities=boundary,
    )
    restored = CellMeshStorage.from_stored_fields(storage)
    assert restored.storage_id == storage.storage_id
    mesh = CellMesh(probe.coordinates, probe.blocks, storage=restored)
    trace_boundary = _boundary_entity_closure(mesh, None)
    for actual, expected_mask in zip(trace_boundary, boundary, strict=True):
        np.testing.assert_array_equal(actual, expected_mask)
    element = (
        lagrange_element("triangle", degree)
        if family == "H1"
        else form_element(
            "triangle",
            1,
            degree,
            twist="twisted" if family == "Hdiv" else "untwisted",
            proxy="flux" if family == "Hdiv" else "circulation",
        )
    )
    dofs = FiniteElementPlan(mesh, FiniteElementFieldSpec("u", element)).prepare().dof_maps[0]
    expected = np.zeros(dofs.boundary_dof_mask.shape, dtype=np.bool_)
    if family == "H1":
        expected[:3] = selected != 4
    np.testing.assert_array_equal(dofs.boundary_dof_mask, expected)


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


def test_block_admission_fuses_primitives_and_keeps_block_count_dynamic() -> None:
    identifiers = jax.device_put(np.asarray(((5, 7, 9, 11, -1, 13),), dtype=np.int64))
    active = jax.device_put(np.asarray(((True, False, False, False, False, True),)))
    retired = jax.device_put(np.asarray(((False, False, True, True, False, False),)))
    blocks = jax.device_put(np.asarray(((0, 1, -9, -8, -7, 1),), dtype=np.int32))
    invalid = jax.device_put(np.asarray(((-1, 1, -9, -8, -7, 1),), dtype=np.int32))
    jax.clear_caches()
    with CompileCounter() as cold:
        passed = result_owner._invalid_collective_blocks(
            identifiers, active, retired, blocks, 2
        ).block_until_ready()
    with CompileCounter() as warm:
        reduced = result_owner._invalid_collective_blocks(
            identifiers, active, retired, blocks, 1
        ).block_until_ready()
        altered = result_owner._invalid_collective_blocks(
            identifiers, active, retired, invalid, 2
        ).block_until_ready()
    assert cold.count == 1
    assert warm.count == 0
    assert not bool(passed)
    assert bool(reduced) and bool(altered)


def test_uniform_owner_join_reuses_compilation_and_checks_every_source_leaf() -> None:
    raw = np.asarray(((11, 13, 21, -1),) * 2, dtype=np.int64)
    published = jax.device_put(np.asarray(((5, 21, -1, -1),) * 2, dtype=np.int64))
    active = jax.device_put(np.asarray(((True, True, False, False),) * 2))
    owners = np.asarray(((3, 1, 2, -1), (1, 0, 3, -1)), dtype=np.int32)
    roots = jax.device_put(np.asarray((5,), dtype=np.int64))
    siblings = jax.device_put(np.asarray(((11, 13),), dtype=np.int64))
    missing_ids = raw.copy()
    missing_ids[1, 1] = 17
    missing_owners = owners.copy()
    missing_owners[0, 1] = -1
    variants = tuple(
        (jax.device_put(ids), jax.device_put(values))
        for ids, values in (
            (raw, owners),
            (raw, owners + np.int32(2)),
            (missing_ids, owners),
            (raw, missing_owners),
        )
    )

    def join(index: int) -> tuple[jax.Array, jax.Array]:
        ids, values = variants[index]
        return result_owner._uniform_publication_owner_values(
            ids, published, active, values, roots, siblings
        )

    jax.clear_caches()
    with CompileCounter() as cold:
        first, missing = jax.block_until_ready(join(0))
    with CompileCounter() as warm:
        changed, changed_missing = jax.block_until_ready(join(1))
        _, absent_child = jax.block_until_ready(join(2))
        _, absent_owner = jax.block_until_ready(join(3))
    assert cold.count == 1
    assert warm.count == 0
    assert not bool(missing) and not bool(changed_missing)
    assert bool(absent_child) and bool(absent_owner)
    expected = np.asarray(((1, 2, -1, -1), (0, 3, -1, -1)), dtype=np.int32)
    assert np.asarray(first).tobytes() == expected.tobytes()
    expected[:, :2] += 2
    assert np.asarray(changed).tobytes() == expected.tobytes()


@pytest.mark.parametrize("capacity", (4, 6))
def test_class_content_order_preserves_allocated_tail_bytes(capacity: int) -> None:
    identifiers = np.asarray(((31, -1, 11), (23, 19, -1)), dtype=np.int64)
    active = np.asarray(((True, False, True), (True, False, False)))
    changed_active = np.asarray(((False, False, True), (True, True, False)))
    classes = np.arange(6, dtype=np.int32).reshape((2, 3))
    facets = np.arange(18, dtype=np.int32).reshape((2, 3, 3))
    arrays = tuple(jax.device_put(value) for value in (identifiers, classes, facets))
    masks = tuple(jax.device_put(value) for value in (active, changed_active))

    def prepare(index: int) -> tuple[jax.Array, jax.Array]:
        ids, cells, faces = arrays
        return result_owner._publication_class_values(
            ids, masks[index], cells, faces, capacity=capacity
        )

    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(prepare(0))
    with CompileCounter() as warm:
        changed = jax.block_until_ready(prepare(1))
    assert cold.count == 1
    assert warm.count == 0
    for result, mask in ((first, active), (changed, changed_active)):
        order = np.argsort(
            np.where(mask.ravel(), identifiers.ravel(), np.iinfo(np.int64).max),
            kind="stable",
        )[:capacity]
        for actual, expected in zip(
            result, (classes.ravel()[order], facets.reshape((-1, 3))[order]), strict=True
        ):
            assert np.asarray(actual).tobytes() == expected.tobytes()


@pytest.mark.parametrize("width,degree", ((3, 1), (4, 1), (4, 2)))
def test_entity_preparation_keeps_canonical_padding_and_epoch_counts_dynamic(
    width: int, degree: int
) -> None:
    rows = np.asarray(
        (tuple(range(width)), tuple(range(1, width + 1)), (0,) * width), dtype=np.int64
    )
    owners = np.asarray((1, 0, -1), dtype=np.int32)
    device_rows, device_owners = jax.device_put(rows), jax.device_put(owners)
    columns = np.asarray(tuple(combinations(range(width), degree + 1)), dtype=np.int32)
    keys = np.sort(rows[:, columns], axis=2).reshape((-1, degree + 1))

    def prepare(count: int) -> tuple[jax.Array, ...]:
        return adaptation_owner._logical_entity_values(
            device_rows, count, device_owners, degree=degree
        )

    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(prepare(2))
    with CompileCounter() as warm:
        changed = jax.block_until_ready(prepare(1))
    assert cold.count == 1
    assert warm.count == 0
    for result, count in ((first, 2), (changed, 1)):
        valid = np.repeat(np.arange(rows.shape[0]) < count, columns.shape[0])
        sentinel = np.iinfo(np.int64).max
        padded = np.where(valid[:, None], keys, sentinel)
        order = np.lexsort(padded.T[::-1])
        ordered = padded[order]
        fresh = np.concatenate(
            (np.asarray((True,)), np.any(ordered[1:] != ordered[:-1], axis=1))
        ) & np.all(ordered != sentinel, axis=1)
        used = np.count_nonzero(fresh)
        selected = np.full(keys.shape[0], keys.shape[0] - 1, dtype=np.int64)
        selected[:used] = np.flatnonzero(fresh)
        table = np.where(
            (np.arange(keys.shape[0]) < used)[:, None], ordered[selected], -1
        )
        routing = np.repeat(owners, columns.shape[0])[order[selected]]
        assert int(result[2]) == used
        assert np.asarray(result[0]).tobytes() == table.tobytes()
        assert np.asarray(result[1]).tobytes() == routing.tobytes()


def test_presentation_group_join_counts_only_valid_dynamic_prefix() -> None:
    table = jax.device_put(np.asarray(((10,), (20,)), dtype=np.int64))
    groups = jax.device_put(np.asarray((1, 0), dtype=np.int64))
    keys = jax.device_put(np.asarray(((20,), (10,), (-1,), (20,)), dtype=np.int64))
    order = jax.device_put(np.asarray((1, 3, 0, 2), dtype=np.int64))
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            adaptation_owner._presentation_group_values(table, groups, keys, order, 3)
        )
    with CompileCounter() as warm:
        changed = jax.block_until_ready(
            adaptation_owner._presentation_group_values(table, groups, keys, order, 1)
        )
        missing = jax.block_until_ready(
            adaptation_owner._presentation_group_values(table, groups, keys, order, 4)
        )
    assert cold.count == 1
    assert warm.count == 0
    np.testing.assert_array_equal(first[0], (1, 0, 0, -1))
    np.testing.assert_array_equal(first[1], (True, True, True, False))
    np.testing.assert_array_equal(first[2], (2, 1))
    np.testing.assert_array_equal(changed[2], (0, 1))
    # An unknown active presentation still leaves the host coverage test short.
    assert int(np.sum(missing[2])) == 3


def test_closure_coordinate_join_preserves_bits_and_rejects_active_missing_vertex() -> None:
    identifiers = jax.device_put(np.asarray(((10,), (20,), (30,), (-1,)), dtype=np.int64))
    bits = np.asarray(
        ((0, 0x8000000000000000), (0x7FF8000000000001, 0x7FF8000000000002),
         (0x3FF0000000000000, 0), (0, 0)),
        dtype=np.uint64,
    )
    coordinates = jax.device_put(bits.view(np.float64))
    vertices = jax.device_put(np.asarray((((20, 30, 10), (99, -1, -1)),), dtype=np.int64))
    inactive = jax.device_put(np.asarray(((True, False),)))
    active = jax.device_put(np.asarray(((True, True),)))
    jax.clear_caches()
    with CompileCounter() as cold:
        first, missing = jax.block_until_ready(
            adaptation_owner._canonical_neighborhood_coordinate_values(
                identifiers, coordinates, vertices, inactive
            )
        )
    with CompileCounter() as warm:
        repeated, rejected = jax.block_until_ready(
            adaptation_owner._canonical_neighborhood_coordinate_values(
                identifiers, coordinates, vertices, active
            )
        )
    assert cold.count == 1
    assert warm.count == 0
    assert not bool(missing) and bool(rejected)
    expected = bits[np.asarray((((1, 2, 0), (0, 0, 0)),))]
    assert np.asarray(first).tobytes() == expected.tobytes()
    assert np.asarray(repeated).tobytes() == expected.tobytes()


def test_entity_receipt_leaf_ordering_fuses_gathers_without_changing_bytes() -> None:
    identifiers = tuple(
        jax.device_put(np.asarray(value, dtype=np.int64))
        for value in ((11, 7, -1), (29, 23, 19, -1))
    )
    masks = tuple(
        jax.device_put(np.asarray(value, dtype=np.bool_))
        for value in ((True, False, False), (False, True, True, False))
    )
    orders = tuple(
        jax.device_put(np.asarray(value, dtype=np.int64))
        for value in ((1, 0, 2), (2, 1, 0, 3))
    )
    changed_orders = tuple(
        jax.device_put(np.arange(value.shape[0], dtype=np.int64)) for value in identifiers
    )
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            adaptation_owner._ordered_entity_set_values(identifiers, masks, orders)
        )
    with CompileCounter() as warm:
        changed = jax.block_until_ready(
            adaptation_owner._ordered_entity_set_values(identifiers, masks, changed_orders)
        )
    assert cold.count == 1
    assert warm.count == 0
    for result, selected in ((first, orders), (changed, changed_orders)):
        for (ids, boundary, active), source, mask, order in zip(
            result, identifiers, masks, selected, strict=True
        ):
            assert np.asarray(ids).tobytes() == np.asarray(source)[order].tobytes()
            assert np.asarray(boundary).tobytes() == np.asarray(mask)[order].tobytes()
            np.testing.assert_array_equal(active, np.ones(source.shape, dtype=np.bool_))


def test_counter_preparation_preserves_exact_work_and_monotonicity_checks() -> None:
    from phydrax.discretization._adaptive_simplex import AdaptiveSimplexCounter

    baseline = np.arange(20, dtype=np.int64).reshape((2, 10))
    counters = baseline + 2
    negative = baseline.copy()
    negative[0, 0] = -1
    backwards = counters.copy()
    backwards[1, 9] = baseline[1, 9] - 1
    inputs = tuple(
        (jax.device_put(before), jax.device_put(after))
        for before, after in ((baseline, counters), (negative, counters), (baseline, backwards))
    )
    jax.clear_caches()
    with CompileCounter() as cold:
        valid, work = jax.block_until_ready(
            adaptation_owner._partitioned_counter_values(*inputs[0])
        )
    with CompileCounter() as warm:
        invalid_baseline, _ = jax.block_until_ready(
            adaptation_owner._partitioned_counter_values(*inputs[1])
        )
        invalid_delta, _ = jax.block_until_ready(
            adaptation_owner._partitioned_counter_values(*inputs[2])
        )
    assert cold.count == 1
    assert warm.count == 0
    assert bool(valid) and not bool(invalid_baseline) and not bool(invalid_delta)
    index = AdaptiveSimplexCounter.BISECTIONS
    assert int(work) == int(np.sum(counters[:, index] - baseline[:, index]))

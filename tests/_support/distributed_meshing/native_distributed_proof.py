# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._fingerprint import logical_array_value_collection_digest, tree_content_equal
from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex_parts,
    refine_adaptive_simplex_parts,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._collective_geometry import build_collective_geometry_witness
from phydrax.meshing._collective_organization import build_collective_organization_witness
from phydrax.meshing._device_adaptation import canonical_rows
from phydrax.meshing._distribution import (
    compiled_bisection_checks,
    expand_partitioned_simplex_neighborhood,
    expand_simplex_neighborhood_workset,
    simplex_neighborhood_seed,
    SimplexNeighborhoodWorkset,
)
from phydrax.meshing._distribution_migration import validate_simplex_predecessor_witness
from phydrax.meshing._entity_keys import facet_packets
from phydrax.meshing.providers._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import NativePublicationPolicy
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
)


_CONTRACT_KEYS = (
    "proof_145",
    "proof_146",
    "proof_152",
    "proof_154",
    "proof_155",
    "proof_156",
    "proof_158",
    "proof_171",
    "proof_172",
    "proof_37",
    "proof_39",
    "proof_41",
    "proof_46",
    "proof_49",
    "proof_50",
    "proof_54",
    "proof_69",
    "proof_70",
    "proof_72",
    "proof_74",
    "proof_81",
    "proof_82",
    "proof_86",
    "proof_90",
    "proof_94",
    "proof_97",
)


def _dense_predecessor_oracle(
    state: AdaptiveSimplexState, initial: AdaptiveSimplexState
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the former dense predicate independently of production labels."""
    active = np.asarray(state.mesh.cell_active)
    initial_active = np.asarray(initial.mesh.cell_active)
    parents = np.asarray(state.parents) // 2
    cells = active.size

    def nearest(slot: int, selected: np.ndarray) -> int:
        for _ in range(cells):
            if selected[slot]:
                return slot
            slot = int(parents[slot])
            if slot < 0:
                return -1
        raise AssertionError("genuine predecessor forest must have acyclic parents")

    roots = np.asarray([nearest(row, initial_active) for row in range(cells)])
    ancestors = np.asarray([nearest(row, active) for row in range(cells)])
    lanes = np.arange(cells)
    support = (
        (
            (roots[:, None] == lanes[None, :])
            | ((roots[:, None] < 0) & (lanes[:, None] == ancestors[None, :]))
        )
        & initial_active[None, :]
        & active[:, None]
    )
    return support, np.where(support, np.asarray(initial.mesh.cell_ids)[None, :], -1)


def _assert_seed_multisets(
    state: AdaptiveSimplexState,
    initial: AdaptiveSimplexState,
    flags: jax.Array,
    rank: int,
    width: int,
) -> SimplexNeighborhoodWorkset:
    support, dense = _dense_predecessor_oracle(state, initial)
    seed = simplex_neighborhood_seed(
        state, initial, flags, jnp.asarray(rank, jnp.int32), 32, source_width=width
    )
    active = np.asarray(state.mesh.cell_active)
    slots = np.flatnonzero(active)
    slots = slots[np.argsort(np.asarray(state.mesh.cell_ids)[slots], kind="stable")]
    actual = np.asarray(seed.source_cell_ids)
    for row, slot in enumerate(slots):
        # Cardinality, not ID sign or uniqueness, defines the selected prefix.
        expected = np.sort(dense[slot, support[slot]], kind="stable")
        np.testing.assert_array_equal(actual[row, : expected.size], expected)
        np.testing.assert_array_equal(actual[row, expected.size :], -1)
    np.testing.assert_array_equal(actual[len(slots) :], -1)
    return seed


def _compact_predecessor_contract(
    parts: AdaptiveSimplexParts,
    source: AdaptiveSimplexState,
    first: AdaptiveSimplexState,
    second: AdaptiveSimplexState,
    coarse: AdaptiveSimplexState,
    exterior: jax.Array,
    first_exterior: jax.Array,
    archive_path: str,
) -> dict[str, Any]:
    widths = []
    coarsened_work = None
    for state, initial, flags in (
        (source, source, exterior),
        (first, source, exterior),
        (second, first, first_exterior),
        (coarse, first, first_exterior),
    ):
        local = [
            (
                jax.tree_util.tree_map(lambda value: value[rank], state),
                jax.tree_util.tree_map(lambda value: value[rank], initial),
            )
            for rank in range(2)
        ]
        expected_width = max(
            1,
            *(int(_dense_predecessor_oracle(current, before)[0].sum(axis=1).max())
              for current, before in local),
        )
        work = expand_partitioned_simplex_neighborhood(
            parts, state, initial, flags,
            halo_width=1, cell_capacity=32, vertex_capacity=32,
        )
        assert work.source_cell_ids.shape == (2, 32, expected_width)
        np.testing.assert_array_equal(work.status, 0)
        widths.append(expected_width)
        by_id = {}
        for rank, (current, before) in enumerate(local):
            seed = _assert_seed_multisets(current, before, flags[rank], rank, expected_width)
            np.testing.assert_array_equal(seed.status, 0)
            for row in np.flatnonzero(np.asarray(seed.cell_valid)):
                by_id[int(seed.cell_ids[row])] = np.asarray(seed.source_cell_ids[row])
        for rank, row in np.argwhere(np.asarray(work.cell_valid)):
            np.testing.assert_array_equal(
                work.source_cell_ids[rank, row], by_id[int(work.cell_ids[rank, row])]
            )
        coarsened_work = work
    assert widths == [1, 1, 1, 2]
    assert coarsened_work is not None
    validate_simplex_predecessor_witness(parts, coarse, first, coarsened_work)
    forged_width = eqx.tree_at(
        lambda value: value.source_cell_ids,
        coarsened_work,
        coarsened_work.source_cell_ids[..., :1],
    )
    np.testing.assert_array_equal(forged_width.status, 0)
    with pytest.raises(ValueError, match="predecessor"):
        validate_simplex_predecessor_witness(parts, coarse, first, forged_width)
    rank, row = np.argwhere(np.asarray(coarsened_work.cell_valid))[0]
    for identifier in (-7, int(coarsened_work.source_cell_ids[rank, row, 1])):
        malformed = eqx.tree_at(
            lambda value: value.source_cell_ids,
            coarsened_work,
            coarsened_work.source_cell_ids.at[rank, row, 0].set(identifier),
        )
        np.testing.assert_array_equal(malformed.status, 0)
        with pytest.raises(ValueError, match="predecessor"):
            validate_simplex_predecessor_witness(parts, coarse, first, malformed)

    # The same actual coarsening support at K-1 must refuse on every rank,
    # including a rank with no local undersized row.
    forged = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *(
            simplex_neighborhood_seed(
                jax.tree_util.tree_map(
                    lambda value: value[rank], coarse if rank == 0 else first
                ),
                jax.tree_util.tree_map(lambda value: value[rank], first),
                first_exterior[rank], jnp.asarray(rank, jnp.int32), 32,
                source_width=1,
            )
            for rank in range(2)
        ),
    )
    assert int(forged.status[0]) & int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    assert int(forged.status[1]) == 0
    refused = expand_simplex_neighborhood_workset(
        parts, forged, halo_width=1, vertex_capacity=32
    )
    assert np.all(
        np.asarray(refused.status) & int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    )

    local_current = jax.tree_util.tree_map(lambda value: value[0], coarse)
    local_initial = jax.tree_util.tree_map(lambda value: value[0], first)
    active_slots = np.flatnonzero(np.asarray(local_initial.mesh.cell_active))
    assert active_slots.size == 2
    for identifiers in ((41, 41), (-7, 41), (np.iinfo(np.int64).max, 41)):
        malformed = eqx.tree_at(
            lambda value: value.mesh.cell_ids, local_initial,
            local_initial.mesh.cell_ids.at[active_slots].set(jnp.asarray(identifiers)),
        )
        _assert_seed_multisets(
            local_current, malformed, first_exterior[0], 0, 2
        )
    unsupported = eqx.tree_at(
        lambda value: value.mesh.cell_active,
        local_initial,
        jnp.zeros_like(local_initial.mesh.cell_active),
    )
    missing = simplex_neighborhood_seed(
        local_current, unsupported, first_exterior[0], jnp.asarray(0, jnp.int32),
        32, source_width=1,
    )
    assert int(missing.status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY)

    # A representation change is an honest identity change, never an entity edit.
    dense_shape = eqx.tree_at(
        lambda value: value.source_cell_ids,
        coarsened_work,
        jnp.pad(coarsened_work.source_cell_ids, ((0, 0), (0, 0), (0, 30)),
                constant_values=-1),
    )
    compact_id = logical_array_value_collection_digest(coarsened_work.arrays())
    dense_id = logical_array_value_collection_digest(dense_shape.arrays())
    assert compact_id != dense_id
    for name, value in coarsened_work.arrays().items():
        if name != "source_cell_ids":
            other = dense_shape.arrays()[name]
            assert value.dtype == other.dtype and value.shape == other.shape
            assert np.asarray(value).tobytes() == np.asarray(other).tobytes()
    receipt = write_meshing_source_closure(
        Path(archive_path), coarsened_work, limits=ArrayArchiveLimits()
    )
    restored = read_meshing_source_closure(
        Path(archive_path), expected_content_id=receipt.content_id,
        limits=ArrayArchiveLimits(),
    )
    assert tree_content_equal(coarsened_work, restored)
    assert logical_array_value_collection_digest(restored.arrays()) == compact_id
    return {
        "widths": widths,
        "dense_oracle_complete": True,
        "duplicates_negative_and_max_preserved": True,
        "undersized_collective_refused": True,
        "forged_zero_status_width_refused": True,
        "malformed_and_duplicate_witness_refused": True,
        "unsupported_refused": True,
        "representation_id_changed": compact_id != dense_id,
        "entity_and_affine_bytes_unchanged": True,
        "canonical_archive_roundtrip": True,
    }


def predecessor_forest_proof(coordinate_kind: str, archive_path: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rounded = coordinate_kind == "rounded"
    if rounded:
        points = np.asarray([[0.1, 0.2], [0.4, 0.2], [0.3, 0.7], [0.0, 0.7]])
    rows = np.asarray([[0, 1, 2], [2, 3, 0]], np.int32)
    layout = AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=32,
        cell_capacity=32,
        protected_edge_capacity=1,
        maximum_closure_iterations=64,
        maximum_coarsening_passes=64,
    )

    def piece(rank: int) -> AdaptiveSimplexState:
        ids = np.unique(rows[rank]).astype(np.int64)
        cells = np.searchsorted(ids, rows[rank : rank + 1]).astype(np.int32)
        return adaptive_simplex_state(
            layout,
            coordinates=points[ids],
            vertex_ids=ids,
            vertex_active=np.ones(3, np.bool_),
            vertex_parents=np.full((3, 2), -1, np.int32),
            vertex_protected=np.zeros(3, np.bool_),
            cells=cells,
            tuples=cells,
            tags=np.full(1, 2, np.int32),
            blocks=np.zeros(1, np.int32),
            generations=np.zeros(1, np.int32),
            parents=np.full(1, -1, np.int32),
            children=np.full((1, 2), -1, np.int32),
            bisection_vertices=np.full(1, -1, np.int32),
            cell_ids=np.asarray([rank + 10], np.int64),
            cell_active=np.ones(1, np.bool_),
            cell_classes=np.full(1, 7, np.int32),
            facet_classes=np.asarray([[3, 0, 3]], np.int32),
            protected_edges=np.empty((0, 2), np.int32),
            next_vertex_id=4,
            next_cell_id=12,
        )

    source = jax.tree_util.tree_map(lambda *v: jnp.stack(v), piece(0), piece(1))
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0, 1), (1, 0)))
    exterior = (
        jnp.zeros(source.mesh.cells.shape, np.bool_)
        .at[:, 0, :]
        .set(jnp.asarray([True, False, True]))
    )
    first = refine_adaptive_simplex_parts(
        layout, parts, source, source.mesh.cell_active
    ).state
    checks.setdefault("proof_37", []).append(
        bool(
            np.all(np.asarray(compiled_bisection_checks(parts, first, source, exterior)))
        )
    )
    first_exterior = first.facet_classes != 0
    checks.setdefault("proof_39", []).append(bool(np.any(np.asarray(first.parents) >= 0)))
    second = refine_adaptive_simplex_parts(
        layout, parts, first, first.mesh.cell_active
    ).state
    checks.setdefault("proof_41", []).append(
        bool(
            np.all(
                np.asarray(
                    compiled_bisection_checks(parts, second, first, first_exterior)
                )
            )
        )
    )

    def packets(
        state: AdaptiveSimplexState, initial: AdaptiveSimplexState, flags: jax.Array
    ) -> SimplexNeighborhoodWorkset:
        return expand_partitioned_simplex_neighborhood(
            parts,
            state,
            initial,
            flags,
            halo_width=1,
            cell_capacity=32,
            vertex_capacity=32,
        )

    work = packets(second, first, first_exterior)
    checks.setdefault("proof_46", []).append(bool(np.all(np.asarray(work.status) == 0)))
    initial_ids = np.asarray(first.mesh.cell_ids)[np.asarray(first.mesh.cell_active)]
    valid = np.asarray(work.cell_valid)
    checks.setdefault("proof_49", []).append(
        bool(np.all(np.isin(np.asarray(work.root_cell_ids)[valid], initial_ids)))
    )
    checks.setdefault("proof_50", []).append(
        bool(not np.any(np.isin(np.asarray(work.root_cell_ids)[valid], [10, 11])))
    )
    support = np.asarray(work.source_vertex_ids)[valid]
    checks.setdefault("proof_54", []).append(bool(np.any(support == 4)))
    field = np.arange(64, dtype=np.float64) ** 2
    values = np.sum(
        np.where(support >= 0, field[np.maximum(support, 0)], 0)
        * np.asarray(work.source_weights)[valid],
        axis=2,
    )
    vertices = np.asarray(work.cell_vertices)[valid]
    expected = {}
    for rank in range(2):
        count = int(second.cursors[rank, 0])
        identifiers = np.asarray(second.mesh.vertex_ids[rank])
        local = field[np.maximum(identifiers, 0)].copy()
        for slot in range(int(first.cursors[rank, 0]), count):
            a, b = np.asarray(second.vertex_parents[rank, slot])
            local[slot] = 0.5 * local[a] + 0.5 * local[b]
        expected.update(((int(identifiers[slot]), local[slot]) for slot in range(count)))
    for vertex in np.unique(vertices):
        copies = values[vertices == vertex]
        _record_array(checks, "proof_69", copies, copies[0])
        _record_array(checks, "proof_70", copies, expected[int(vertex)])
    coarse = coarsen_adaptive_simplex_parts(
        layout, parts, first, first.mesh.cell_active
    ).state
    checks.setdefault("proof_72", []).append(
        bool(
            np.all(
                np.asarray(
                    compiled_bisection_checks(parts, coarse, first, first_exterior)
                )
            )
        )
    )
    work = packets(coarse, first, first_exterior)
    checks.setdefault("proof_74", []).append(bool(np.all(np.asarray(work.status) == 0)))
    checks["compact_predecessors"] = _compact_predecessor_contract(
        parts, source, first, second, coarse, exterior, first_exterior, archive_path
    )
    for rank in range(2):
        for lane in np.flatnonzero(np.asarray(work.cell_valid[rank])):
            identifier = int(work.cell_ids[rank, lane])
            owner = identifier - 10
            children = np.asarray(first.mesh.cell_ids[owner])[
                np.asarray(first.mesh.cell_active[owner])
            ]
            actual = np.asarray(work.source_cell_ids[rank, lane])
            _record_array(
                checks, "proof_81", np.sort(actual[actual >= 0]), np.sort(children)
            )
            checks.setdefault("proof_82", []).append(
                bool(int(work.root_cell_ids[rank, lane]) == -1)
            )
            ids = np.asarray(work.source_vertex_ids[rank, lane])
            weights = np.asarray(work.source_weights[rank, lane])
            reconstructed = np.sum(
                np.where(ids >= 0, field[np.maximum(ids, 0)], 0) * weights, axis=1
            )
            _record_array(
                checks,
                "proof_86",
                reconstructed,
                field[np.asarray(work.cell_vertices[rank, lane])],
            )
    bad = eqx.tree_at(lambda s: s.parents, second, second.parents.at[1, 1].set(-1))
    verdicts = np.asarray(compiled_bisection_checks(parts, bad, first, first_exterior))
    checks.setdefault("proof_90", []).append(bool(not np.all(verdicts[:, 0])))
    again = refine_adaptive_simplex_parts(
        layout, parts, coarse, coarse.mesh.cell_active
    ).state
    coarse_exterior = coarse.facet_classes != 0
    checks.setdefault("proof_94", []).append(
        bool(
            np.all(
                np.asarray(
                    compiled_bisection_checks(parts, again, coarse, coarse_exterior)
                )
            )
        )
    )
    retired = int(np.flatnonzero(np.asarray(coarse.retired[1]))[0])
    bad = eqx.tree_at(
        lambda s: s.mesh.cell_ids, again, again.mesh.cell_ids.at[1, retired].add(100)
    )
    checks.setdefault("proof_97", []).append(
        bool(
            not np.all(
                np.asarray(
                    compiled_bisection_checks(parts, bad, coarse, coarse_exterior)
                )[:, 0]
            )
        )
    )
    print(
        "PROOF: accepted forest refine/coarsen; complete source children; nonlinear ghost field ancestry; owner corruption rejected"
    )
    M = phx.meshing
    D = phx.discretization
    mesh = D.CellMesh(
        points,
        (
            D.CellBlock(
                "domain", "triangle", rows, global_ids=np.asarray([10, 11], np.int64)
            ),
        ),
        vertex_global_ids=np.arange(4, dtype=np.int64),
    )

    def scope(degree: int, ids: list[int]) -> M.MeshingScope:
        return M.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            M.MeshingEntityKind.MESH,
            degree,
            mesh.entity_set(degree).entity_set_id,
            np.asarray(ids, np.int64),
        )

    all_cells = scope(2, [10, 11])
    zones = (M.MeshZone("material", M.MeshZoneRole.USER, all_cells),)
    labels = (M.MeshLabel("left", scope(2, [10])),)
    attributes = (
        M.MeshAttribute(
            "marker",
            M.MeshAttributeRole.MARKER,
            all_cells,
            np.asarray([19, 23], np.int64),
        ),
    )
    seed = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    loop = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], np.int32)
    domain = phx.geometry.PiecewiseLinearDomain(
        points,
        loop,
        np.tile(np.asarray([0, -1], np.int32), (4, 1)),
        ("domain",),
        source_id="proof-source",
    )
    limits = M.MeshingLimits()
    with (
        NativeExecutionScope(limits.native_limits()) as native,
        native.host_workspace() as workspace,
    ):
        original = publish_native_result(
            mesh,
            seed.coordinate_contract,
            M.MeshingComplianceReport("proof-source"),
            (),
            seed.provider,
            {"kind": "scientific-proof-source"},
            NativeCertificationRequest(
                M.MeshCertificationSchedule("volume_plc"),
                "proof-source",
                "r0",
                limits,
                domain=domain,
                cell_regions=np.zeros(2, np.int32),
            ),
            policy=NativePublicationPolicy(
                M.CellMeshAuditPolicy(
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT
                ),
                M.MeshingDerivativeMode.FIXED_TOPOLOGY_EXACT,
                (),
                (),
            ),
            organization=NativePublicationOrganization(
                zones=zones, labels=labels, attributes=attributes
            ),
            execution=NativePublicationExecution(native, workspace, None, None),
        )
    placement = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))

    def logical(
        state: AdaptiveSimplexState,
    ) -> tuple[
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        tuple[jax.Array, ...],
        tuple[int, ...],
    ]:
        vkeys, vorder, nv = canonical_rows(
            state.mesh.vertex_ids.reshape((-1, 1)), state.mesh.vertex_active.reshape(-1)
        )
        ckeys, corder, nc = canonical_rows(
            state.mesh.cell_ids.reshape((-1, 1)), state.mesh.cell_active.reshape(-1)
        )
        fkeys, _ = jax.vmap(facet_packets)(state)
        fkeys, forder, nf = canonical_rows(
            fkeys.reshape((-1, 2)), jnp.repeat(state.mesh.cell_active.reshape(-1), 3)
        )
        ids = (
            vkeys[:, 0],
            jnp.where(
                jnp.arange(fkeys.shape[0]) < nf,
                jnp.arange(fkeys.shape[0], dtype=jnp.int64) + 1000,
                -1,
            ),
            ckeys[:, 0],
        )
        owners = (vorder // 32, forder // (32 * 3), corder // 32)
        return (
            tuple((jax.device_put(a, placement) for a in (vkeys, fkeys, ckeys))),
            tuple((jax.device_put(a, placement) for a in ids)),
            tuple((jax.device_put(a.astype(jnp.int32), placement) for a in owners)),
            (nv, nf, nc),
        )

    for state in (first, second, coarse):
        keys, ids, owners, counts = logical(state)
        with coordinate_reference_budget() as budget:
            bank, organization_id = build_collective_organization_witness(
                original,
                state,
                keys,
                ids,
                counts,
                uniform_refinement=None,
                budget=budget,
            )
        bank = dict(bank)
        roots = np.asarray(bank["organization/cell/source_ids"])[: counts[-1]]
        _record_array(
            checks,
            "proof_145",
            np.asarray(bank["organization/label/0/membership"])[: counts[-1]],
            roots == 10,
        )
        _record_array(
            checks,
            "proof_146",
            np.asarray(bank["organization/attribute/0/values"])[: counts[-1]],
            np.where(roots == 10, 19, 23),
        )
        with NativeExecutionScope(
            NativeExecutionLimits(
                maximum_work_units=40960000,
                maximum_geometry_queries=200000,
                maximum_managed_bytes=81920000,
                maximum_seconds=120,
                maximum_cavity_cells=M.MeshingLimits().maximum_cavity_cells,
            )
        ) as native:
            remaining = native.remaining()
            with (
                native.host_workspace(),
                native.batch(
                    min(40960000, remaining.remaining_work_units),
                    bytes_upper=min(81920000, remaining.remaining_managed_bytes),
                ) as ticket,
                CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=40960000,
                    maximum_memory_bytes=81920000,
                ).activate() as budget,
            ):
                geometry, geometry_id = build_collective_geometry_witness(
                    original,
                    state,
                    tuple(bank.items()),
                    ids,
                    owners,
                    counts,
                    budget=budget,
                )
        geometry = dict(geometry)
        _record_array(
            checks,
            "proof_152",
            np.asarray(geometry["geometry/coordinates"])[:4],
            np.asarray(original.geometry.coordinates),
        )
        if state is coarse:
            checks.setdefault("proof_154", []).append(
                bool(geometry_id == cell_geometry_id(original.geometry))
            )
            _record_array(
                checks,
                "proof_155",
                np.asarray(geometry["geometry/action_counts/domain"])[:2],
                np.zeros(2, dtype=np.int32),
            )
            _record_array(
                checks,
                "proof_156",
                np.asarray(geometry["geometry/action_weights/domain"])[:2],
                0.0,
            )
        else:
            checks.setdefault("proof_158", []).append(
                bool(geometry_id != cell_geometry_id(original.geometry))
            )
    if rounded:
        keys, ids, owners, counts = logical(first)
        with coordinate_reference_budget() as budget:
            bank, _ = build_collective_organization_witness(
                original,
                first,
                keys,
                ids,
                counts,
                uniform_refinement=None,
                budget=budget,
            )
        bank = dict(bank)
        row = int(np.flatnonzero(np.asarray(ids[0]) == 4)[0])
        support = np.asarray(bank["organization/vertex/source_ids"])[row]
        weights = np.asarray(bank["organization/vertex/source_weights"])[row]
        exact = sum(
            (
                Fraction.from_float(float(points[i, 0])) * Fraction.from_float(float(w))
                for i, w in zip(support, weights, strict=True)
                if i >= 0
            ),
            Fraction(),
        )
        actual = first.mesh.coordinates[
            0, int(np.flatnonzero(np.asarray(first.mesh.vertex_ids[0]) == 4)[0]), 0
        ]
        checks.setdefault("proof_171", []).append(
            bool(exact != Fraction.from_float(float(actual)))
        )
        checks.setdefault("proof_172", []).append(
            bool(
                exact
                == (
                    Fraction.from_float(float(points[0, 0]))
                    + Fraction.from_float(float(points[2, 0]))
                )
                / 2
            )
        )
    print(
        "PROOF: global source occurrences and exact original coefficient maps; coarsen restores scientific field"
    )
    return checks

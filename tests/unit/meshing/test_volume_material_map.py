# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Cavity-local chronological material chains and their FE/FV transport."""
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction as F
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    meshcore_provider_installed,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization.fem._topology_transfer import SourceRealizationFieldSemantics
from phydrax.discretization.finite_volume._automatic_remap import (
    prepare_unstructured_conservative_remap,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from phydrax.exterior._form_type import FormProxy
from phydrax.geometry._exact_polyhedral_geometry import ExactTetrahedronOverlap
from phydrax.geometry._implicit_cavity_atlas import (
    build_implicit_cavity_atlas,
    CavitySnapshot,
    ImplicitCavityAtlas,
    prepare_implicit_cavity_edit,
)
from phydrax.geometry._supermesh import CommonRefinementPolicy, CommonRefinementStatus
from phydrax.geometry._volume_material_map import (
    _weight_pair,
    volume_material_edit_pieces,
    volume_material_edit_weights,
    VolumeMaterialChain,
    VolumeMaterialEdit,
    VolumeMaterialEditPieces,
    VolumeMaterialMapError,
    VolumeMaterialStatus,
)
from phydrax.meshing._projected_material_map import prepare_volume_material_fe_transfers


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_provider_installed(), reason="meshcore provider is not installed"
    ),
]

_WORK = 20_000_000
_BYTES = 256 * 1024**2
# Binary64-exact corners: a lifted projected insertion (vertex 4), then a
# domain-preserving split of cell 23 at an interior vertex (5).
_POINTS = np.asarray((
    (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.25, 0.25, 0.125), (0.3125, 0.0625, -0.21875),
))
_INSERTED = ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4))
_SPLIT = ((5, 1, 2, 4), (0, 5, 2, 4), (0, 1, 5, 4), (0, 1, 2, 5))


_SPHERE_BYTES = 2 * 1024**3


@contextmanager
def _budget(work: int = _WORK, memory: int = _BYTES) -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(work, queries_upper=_WORK, bytes_upper=memory) as ticket:
        budget = CoordinateEnclosureBudget(ticket, maximum_work_units=work, maximum_memory_bytes=memory)
        with budget.activate():
            yield budget


def _exact(row: np.ndarray) -> tuple[F, F, F]:
    return F(float(row[0])), F(float(row[1])), F(float(row[2]))


def _atlas(
    budget: CoordinateEnclosureBudget, *, points: np.ndarray = _POINTS[:4],
    cells: tuple[tuple[int, int, int, int], ...] = ((0, 1, 2, 3),), cell_ids: tuple[int, ...] = (17,),
) -> ImplicitCavityAtlas:
    """The lifted insertion over source cavity ``cells`` (local rows of ``points``)."""
    vertices = tuple(_exact(row) for row in points)
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 1, "ca"))
    before = CavitySnapshot(vertices, cells, ((1, 2, 3),), ("source-face",), cell_ids, rim)
    after = CavitySnapshot(
        (*vertices[:4], _exact(_POINTS[4])), _INSERTED, ((1, 2, 4), (2, 3, 4), (3, 1, 4)),
        ("source-face",) * 3, (23, 24, 25), rim,
    )
    edit = prepare_implicit_cavity_edit(
        before, after, source_id="original-source", discovery_id="original-discovery",
        chronology_index=0, predecessor_id="", budget=budget,
    )
    return build_implicit_cavity_atlas(
        edit, chart_axes=(0, 1), anchor=(F(1, 4), F(1, 4), F(-1, 4)), budget=budget,
    )


def _edit(
    removed: tuple[int, ...], removed_rows: tuple[tuple[int, ...], ...],
    created: tuple[int, ...], created_rows: tuple[tuple[int, ...], ...],
    atlas: ImplicitCavityAtlas | None = None, points: np.ndarray = _POINTS,
) -> VolumeMaterialEdit:
    old, new = np.asarray(removed_rows), np.asarray(created_rows)
    return _curved_edit(
        np.asarray(removed), old, _POINTS[old], np.asarray(created), new, points[new], atlas,
    )


def _curved_edit(
    removed: np.ndarray, removed_vertices: np.ndarray, removed_points: np.ndarray,
    created: np.ndarray, created_vertices: np.ndarray, created_points: np.ndarray,
    atlas: ImplicitCavityAtlas | None,
) -> VolumeMaterialEdit:
    """One edit; a committed atlas supplies its cones, anchor, parents and chart ID."""
    if atlas is None:
        return VolumeMaterialEdit(
            removed, removed_vertices, removed_points, created, created_vertices, created_points,
        )
    return VolumeMaterialEdit(
        removed, removed_vertices, removed_points, created, created_vertices, created_points,
        source_charts=atlas.old_tets,
        target_charts=atlas.new_tets,
        chart_anchor=atlas.anchor,
        source_parents=tuple(
            (parent.cell_id, parent.vertices, parent.orientation)
            for parent in atlas.source_boundary_parents
        ),
        target_parents=tuple(
            (parent.cell_id, parent.vertices, parent.orientation)
            for parent in atlas.target_boundary_parents
        ),
        chart_id=atlas.chart_id,
    )


def _endpoints(*, permuted: bool = False) -> tuple[CellMesh, CellMesh]:
    source = CellMesh.from_tetrahedra(
        _POINTS[:4], np.asarray(((0, 1, 2, 3),)), cell_global_ids=np.asarray((17,)),
    )
    rows = np.asarray((*_INSERTED[1:], *_SPLIT))
    if permuted:
        # An even corner rotation: the same positive cells, listed otherwise.
        rows = rows[:, (0, 2, 3, 1)]
    target = CellMesh.from_tetrahedra(
        _POINTS, rows, cell_global_ids=np.asarray((24, 25, 26, 27, 28, 29)),
    )
    return source, target


def _chain(
    budget: CoordinateEnclosureBudget, *, split: VolumeMaterialEdit | None = None,
    permuted: bool = False,
) -> VolumeMaterialChain:
    atlas = _atlas(budget)
    source, target = _endpoints(permuted=permuted)
    edits = (
        _edit((17,), ((0, 1, 2, 3),), (23, 24, 25), _INSERTED, atlas),
        split if split is not None else _edit((23,), _INSERTED[:1], (26, 27, 28, 29), _SPLIT),
    )
    return VolumeMaterialChain(
        source, CellGeometrySpec.affine(source), target, CellGeometrySpec.affine(target), edits,
        original_source_id="original-source", discovery_id="original-discovery",
        maximum_work_units=_WORK, maximum_scratch_bytes=_BYTES,
        atlases={atlas.chart_id: atlas}, budget=budget,
    )


def _many_cavity_history(
    budget: CoordinateEnclosureBudget, cycles: int
) -> tuple[
    CellMesh, CellMesh, tuple[VolumeMaterialEdit, ...], dict[str, ImplicitCavityAtlas]
]:
    """A lifted insertion followed by repeated exact splits and collapses."""
    atlas = _atlas(budget)
    source = CellMesh.from_tetrahedra(
        _POINTS[:4], np.asarray(((0, 1, 2, 3),)), cell_global_ids=np.asarray((17,))
    )
    edits = [_edit((17,), ((0, 1, 2, 3),), (23, 24, 25), _INSERTED, atlas)]
    current = 23
    next_identity = 100
    for _ in range(cycles):
        split_ids = tuple(range(next_identity, next_identity + 4))
        edits.append(_edit((current,), _INSERTED[:1], split_ids, _SPLIT))
        next_identity += 4
        collapsed = next_identity
        edits.append(_edit(split_ids, _SPLIT, (collapsed,), _INSERTED[:1]))
        next_identity += 1
        current = collapsed
    target = CellMesh.from_tetrahedra(
        _POINTS,
        np.asarray(_INSERTED),
        cell_global_ids=np.asarray((current, 24, 25)),
    )
    return source, target, tuple(edits), {atlas.chart_id: atlas}


def _volumes(mesh: CellMesh) -> np.ndarray:
    corners = np.asarray(mesh.coordinates)[np.concatenate([np.asarray(b.vertices) for b in mesh.blocks])]
    return np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0


def _fv_endpoints(
    chain: VolumeMaterialChain,
) -> tuple[UnstructuredFiniteVolumeDiscretization, UnstructuredFiniteVolumeDiscretization]:
    source, target = (
        UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare(cell_geometry=geometry)
        for mesh, geometry in (
            (chain.source_mesh, chain.source_geometry), (chain.target_mesh, chain.target_geometry),
        )
    )
    return source, target


def _exact_rows(
    chain: VolumeMaterialChain, budget: CoordinateEnclosureBudget,
) -> dict[int, dict[int, F]]:
    """Independent exact chronological composition: content measure per original source row."""
    source_ids = np.asarray(chain.source_mesh.blocks[0].global_ids).tolist()
    rows: dict[int, dict[int, F]] = {}
    for edit in chain.edits:
        pieces = volume_material_edit_pieces(edit, budget=budget)
        removed = np.asarray(edit.source_cell_ids).tolist()
        created = np.asarray(edit.target_cell_ids).tolist()
        old = [
            rows.pop(cell) if cell in rows else {source_ids.index(cell): volume}
            for cell, volume in zip(removed, pieces.source_volumes, strict=True)
        ]
        new: list[dict[int, F]] = [{} for _ in created]
        for (target, source, _), volume in zip(
            np.asarray(pieces.routes).tolist(), pieces.volumes, strict=True
        ):
            share = volume / pieces.source_volumes[source]
            for original, measure in old[source].items():
                new[target][original] = new[target].get(original, F(0)) + share * measure
        rows.update(zip(created, new, strict=True))
    return rows


def test_live_state_promotion_is_exact_and_does_not_readmit_storage() -> None:
    shared = (F(1, 3), F(2, 5), F(7, 11))
    state = {
        17: ((0, shared), (1, (F(0), F(1), F(0))), (2, (F(1), F(0), F(0)))),
        18: ((0, shared), (1, (F(0), F(1), F(0))), (3, (F(0), F(0), F(1)))),
    }

    def promote(storage_limit: int | None) -> tuple[int, int]:
        with _budget() as budget:
            storage = budget.live_object_storage_upper(state)
            starting = budget.retained_basis_bytes + budget.temporary_bytes_upper

            def transition() -> None:
                with budget.live_storage() as live:
                    with budget.temporary_scope() as source:
                        budget.reserve(0, storage)
                        admitted = budget.temporary_bytes_upper
                        live.promote_temporary_storage(source, storage)
                        assert budget.temporary_bytes_upper == admitted
                    assert live.bound == storage

            with budget.measurement() as measurement:
                if storage_limit is None:
                    transition()
                else:
                    with budget.bound_stage(_WORK, starting + storage_limit):
                        transition()
            return storage, measurement.peak_bytes_upper

    required, measured = promote(None)
    assert measured == required
    assert promote(required) == (required, required)
    with pytest.raises(CoordinateEnclosureResourceError):
        promote(required - 1)
    with _budget() as budget:
        with budget.live_storage() as live:
            with budget.temporary_scope() as outer:
                with pytest.raises(RuntimeError, match="source credit"):
                    live.promote_temporary_storage(outer, 1)
                with budget.temporary_scope():
                    with pytest.raises(RuntimeError, match="innermost"):
                        live.promote_temporary_storage(outer, 1)
                foreign = CoordinateEnclosureBudget(
                    budget.ticket,
                    maximum_work_units=budget.maximum_work_units,
                    maximum_memory_bytes=budget.maximum_memory_bytes,
                )
                with foreign.temporary_scope() as foreign_scope:
                    with pytest.raises(ValueError, match="destination budget"):
                        live.promote_temporary_storage(foreign_scope, 1)
            with pytest.raises(RuntimeError, match="innermost"):
                live.promote_temporary_storage(outer, 1)


def test_chain_edit_workspace_is_released_while_history_outputs_remain_charged() -> None:
    def measure(cycles: int) -> tuple[int, VolumeMaterialChain, int]:
        with _budget() as budget:
            source, target, edits, atlases = _many_cavity_history(budget, cycles)
            retained_before = budget.retained_basis_bytes
            with budget.measurement() as measurement:
                chain = VolumeMaterialChain(
                    source,
                    CellGeometrySpec.affine(source),
                    target,
                    CellGeometrySpec.affine(target),
                    edits,
                    original_source_id="original-source",
                    discovery_id="original-discovery",
                    maximum_work_units=_WORK,
                    maximum_scratch_bytes=_BYTES,
                    atlases=atlases,
                    budget=budget,
                )
                budget.retain_owner(chain)
            retained = budget.retained_basis_bytes - retained_before
            return measurement.peak_bytes_upper, chain, retained

    short_peak, _, short_retained = measure(1)
    many_peak, _, many_retained = measure(17)
    assert many_retained > short_retained
    assert many_peak >= many_retained
    # Only the additional retained graph is allowed to grow the history peak.
    assert many_peak <= short_peak + (many_retained - short_retained)


def test_chain_weights_are_exact_per_edit_partitions() -> None:
    with _budget() as budget:
        chain = _chain(budget)
        offsets, targets, sources, mantissas, exponents, errors = volume_material_edit_weights(chain)
        for index, edit in enumerate(chain.edits):
            pieces = volume_material_edit_pieces(edit, budget=budget)
            exact: dict[tuple[int, int], F] = {}
            for (target, source, _), volume in zip(
                np.asarray(pieces.routes).tolist(), pieces.volumes, strict=True
            ):
                exact[(target, source)] = exact.get((target, source), F(0)) + volume
            rows = range(int(offsets[index]), int(offsets[index + 1]))
            assert sorted(exact) == [(int(targets[r]), int(sources[r])) for r in rows]
            covered = [F(0)] * len(pieces.source_volumes)
            for r in rows:
                value = exact[(int(targets[r]), int(sources[r]))] / pieces.target_volumes[int(targets[r])]
                scale = F(2) ** int(exponents[r])
                # Each published mantissa encloses the exact density weight.
                assert 0.5 <= mantissas[r] < 1.0
                assert abs(F(float(mantissas[r])) * scale - value) <= F(float(errors[r])) * scale
                covered[int(sources[r])] += value * pieces.target_volumes[int(targets[r])]
            # The weights times target volumes partition every source exactly.
            assert covered == list(pieces.source_volumes)


@pytest.mark.parametrize(
    "value",
    (F(1), F(3, 4), F(1, 3), F(2**1100, 3), F(1, 3 * 2**1100), F(2**53 - 1, 2**53), F(10**400, 7)),
)
def test_weight_pairs_never_overflow_and_enclose_their_rational(value: F) -> None:
    mantissa, exponent, error = _weight_pair(value)
    scale = F(2) ** exponent
    assert 0.5 <= mantissa < 1.0
    assert abs(F(mantissa) * scale - value) <= F(error) * scale
    assert F(error) <= F(2) ** -53


def test_factorized_material_remap_conserves_and_matches_the_exact_composition() -> None:
    with _budget() as budget:
        chain = _chain(budget)
        exact = _exact_rows(chain, budget)
        source, target = _fv_endpoints(chain)
        prepared = prepare_unstructured_conservative_remap(
            source, target, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        )
    plan = prepared.plan
    assert plan is not None
    assert plan.measure_semantics == "source-physical-volume-material-content"
    values = np.asarray(((2.0, -3.0),))
    contents = np.asarray(source.cell_volumes)[:, None] * values
    carried = np.asarray(plan.apply_content(contents))
    np.testing.assert_allclose(np.sum(carried, axis=0), contents[0], rtol=0.0, atol=1e-15)
    # Unit density carries each target's exact old-frame content.
    content = np.asarray(plan.apply_content(np.asarray(source.cell_volumes)))
    target_ids = np.asarray(chain.target_mesh.blocks[0].global_ids).tolist()
    for position, cell in enumerate(target_ids):
        reference = float(sum(exact[cell].values(), F(0)))
        assert abs(content[position] - reference) <= 1e-14 * reference
    # Cell averages: content divided by the bound target volume.
    averages = np.asarray(plan.apply(np.ones((1,))))
    np.testing.assert_allclose(
        averages * np.asarray(target.cell_volumes), content, rtol=1e-14, atol=0.0
    )


def test_recreated_cell_id_from_an_independent_branch_keeps_chronology() -> None:
    # Cell 23 is split away (edit 1); the independent cell 24 is relabelled as
    # a new cell 23 (edit 2, the same dependency level), which edit 3 removes.
    # Only birth-occurrence tokens let edit 3 read edit 2's cell 23.
    target_mesh = CellMesh.from_tetrahedra(
        _POINTS,
        np.asarray((_INSERTED[2], *_SPLIT, _INSERTED[1])),
        cell_global_ids=np.asarray((25, 26, 27, 28, 29, 30)),
    )
    with _budget() as budget:
        atlas = _atlas(budget)
        source_mesh, _ = _endpoints()
        chain = VolumeMaterialChain(
            source_mesh, CellGeometrySpec.affine(source_mesh), target_mesh,
            CellGeometrySpec.affine(target_mesh),
            (
                _edit((17,), ((0, 1, 2, 3),), (23, 24, 25), _INSERTED, atlas),
                _edit((23,), _INSERTED[:1], (26, 27, 28, 29), _SPLIT),
                _edit((24,), _INSERTED[1:2], (23,), _INSERTED[1:2]),
                _edit((23,), _INSERTED[1:2], (30,), _INSERTED[1:2]),
            ),
            original_source_id="original-source", discovery_id="original-discovery",
            maximum_work_units=_WORK, maximum_scratch_bytes=_BYTES,
            atlases={atlas.chart_id: atlas}, budget=budget,
        )
        exact = _exact_rows(chain, budget)
        source, target = _fv_endpoints(chain)
        plan = prepare_unstructured_conservative_remap(
            source, target, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        ).plan
    assert plan is not None
    content = np.asarray(plan.apply_content(np.asarray(source.cell_volumes)))
    for position, cell in enumerate((25, 26, 27, 28, 29, 30)):
        reference = float(sum(exact[cell].values(), F(0)))
        assert abs(content[position] - reference) <= 1e-14 * reference


def test_explicit_csr_view_materializes_the_exact_final_rows() -> None:
    with _budget() as budget:
        chain = _chain(budget)
        exact = _exact_rows(chain, budget)
        source, target = _fv_endpoints(chain)
        plan = prepare_unstructured_conservative_remap(
            source, target, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        ).plan
        assert plan is not None
        view = plan.prepare_csr_view(phx.linalg.SparseAssemblyPolicy(), budget=budget)
    offsets = np.asarray(view.target_offsets)
    indices = np.asarray(view.source_indices)
    measures = np.asarray(view.intersection_measures)
    target_ids = np.asarray(chain.target_mesh.blocks[0].global_ids).tolist()
    for position, cell in enumerate(target_ids):
        start, stop = int(offsets[position]), int(offsets[position + 1])
        assert indices[start:stop].tolist() == sorted(exact[cell])
        for original, measure in zip(indices[start:stop].tolist(), measures[start:stop].tolist(), strict=True):
            assert abs(F(measure) - exact[cell][original]) <= F(1e-14) * exact[cell][original]
    # A policy below the materialized row count refuses before allocation.
    with _budget() as budget, pytest.raises(phx.linalg.LinearCapabilityError):
        plan.prepare_csr_view(phx.linalg.SparseAssemblyPolicy(max_nnz=1), budget=budget)


@pytest.mark.parametrize("material", (True, False), ids=("material", "routed"))
def test_csr_view_uses_measured_caller_budget_without_reserving_policy_limits(
    material: bool,
) -> None:
    from phydrax.discretization.finite_volume._unstructured_remap import (
        UnstructuredConservativeRemapPlan,
    )

    with _budget() as budget:
        chain = _chain(budget)
        source, target = _fv_endpoints(chain)
        if material:
            plan = prepare_unstructured_conservative_remap(
                source,
                target,
                provenance="volume-material-chain-test",
                volume_material_chain=chain,
                budget=budget,
            ).plan
            assert plan is not None
        else:
            plan = UnstructuredConservativeRemapPlan(
                source,
                source,
                np.asarray((0, 1), dtype=np.int32),
                np.asarray((0,), dtype=np.int32),
                source.cell_volumes,
                method="physical-cell-overlap",
                provenance="volume-material-chain-test",
            )
    policy = phx.linalg.SparseAssemblyPolicy(
        max_nnz=1_000_000,
        max_bytes=1 << 30,
        max_contributions=4_000_000,
        max_workspace_bytes=1 << 31,
    )
    with _budget() as measurement:
        measurement.retain_owner(plan)
        plan.prepare_csr_view(policy, budget=measurement)
        required = measurement.peak_bytes_upper
    with _budget(memory=required) as sufficient:
        sufficient.retain_owner(plan)
        view = plan.prepare_csr_view(policy, budget=sufficient)
    np.testing.assert_allclose(
        np.sum(view.intersection_measures), np.sum(plan.source_volumes),
        rtol=1e-12, atol=0.0,
    )
    with _budget(memory=required - 1) as insufficient, pytest.raises(
        CoordinateEnclosureResourceError
    ):
        insufficient.retain_owner(plan)
        plan.prepare_csr_view(policy, budget=insufficient)


def test_explicit_view_admission_refuses_rows_or_plans_changed_after_construction() -> None:
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _fv_endpoints(chain)
        plan = prepare_unstructured_conservative_remap(
            source, target, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        ).plan
        assert plan is not None
        policy = phx.linalg.SparseAssemblyPolicy()
        view = plan.prepare_csr_view(policy, budget=budget)
        equivalent = plan.prepare_csr_view(policy, budget=budget)
    # Legitimate and equivalently rebuilt views are admitted with one identity.
    # The plan records its first-order mantissa-gradient replay evidence.
    assert plan.material_replay_evidence is not None
    assert plan.material_schedule is not None
    length, blocks, peak, operations = plan.material_replay_evidence
    assert length * blocks >= plan.material_schedule.pair_sources.shape[0]
    assert operations == 2 * length * blocks and peak > 0
    view.require_bound(plan)
    equivalent.require_bound(plan)
    assert equivalent.view_id == view.view_id
    # Replacing a numeric leaf keeps every carried identifier, yet admission
    # recomputes the content and refuses.
    forged_rows = eqx.tree_at(
        lambda value: value.intersection_measures, view, view.intersection_measures * 2.0,
    )
    assert forged_rows.view_id == view.view_id
    with pytest.raises(ValueError, match="rows derived from its plan"):
        forged_rows.require_bound(plan)
    assert plan.volume_material_chain is not None
    forged_plan = eqx.tree_at(
        lambda value: value.volume_material_chain.weight_mantissas,
        plan,
        np.asarray(plan.volume_material_chain.weight_mantissas) * 0.75,
    )
    assert forged_plan.plan_id == plan.plan_id
    with pytest.raises(ValueError, match="admitted plan's content"):
        view.require_bound(forged_plan)
    forged_view_plan = eqx.tree_at(lambda value: value.plan, view, forged_plan)
    with pytest.raises(ValueError, match="admitted plan's content"):
        forged_view_plan.require_bound(plan)


def _split_chain(budget: CoordinateEnclosureBudget) -> VolumeMaterialChain:
    """A certified identity chart over a nested split of cell 23 at vertex 5."""
    corners = np.asarray((0, 1, 2, 4))
    local = {int(vertex): row for row, vertex in enumerate((0, 1, 2, 4, 5))}
    source = CellMesh.from_tetrahedra(
        _POINTS[corners], np.asarray(((0, 1, 2, 3),)),
        vertex_global_ids=corners, cell_global_ids=np.asarray((23,)),
    )
    target = CellMesh.from_tetrahedra(
        _POINTS[np.asarray((0, 1, 2, 4, 5))],
        np.asarray([[local[vertex] for vertex in row] for row in _SPLIT]),
        vertex_global_ids=np.asarray((0, 1, 2, 4, 5)),
        cell_global_ids=np.asarray((26, 27, 28, 29)),
    )
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 1, "ca"))
    patch = ((1, 2, 3),)
    points = tuple(_exact(row) for row in _POINTS[np.asarray((0, 1, 2, 4, 5))])
    before = CavitySnapshot(
        points[:4], ((0, 1, 2, 3),), patch, ("source-face",), (23,), rim,
    )
    after = CavitySnapshot(
        points,
        tuple((local[a], local[b], local[c], local[d]) for a, b, c, d in _SPLIT),
        patch, ("source-face",), (26, 27, 28, 29), rim,
    )
    prepared = prepare_implicit_cavity_edit(
        before, after, source_id="original-source", discovery_id="original-discovery",
        chronology_index=0, predecessor_id="", budget=budget,
    )
    atlas = build_implicit_cavity_atlas(
        prepared, chart_axes=(0, 1), anchor=_exact(_POINTS[5]), budget=budget,
    )
    return VolumeMaterialChain(
        source, CellGeometrySpec.affine(source), target, CellGeometrySpec.affine(target),
        (_edit((23,), _INSERTED[:1], (26, 27, 28, 29), _SPLIT, atlas),),
        original_source_id="original-source", discovery_id="original-discovery",
        maximum_work_units=_WORK, maximum_scratch_bytes=_BYTES,
        atlases={atlas.chart_id: atlas}, budget=budget,
    )


def test_amr_admits_a_material_prolongation_view_only_through_its_content_binding() -> None:
    from phydrax.discretization.finite_volume._unstructured_amr import (
        UnstructuredAMRHierarchyPlan,
    )
    from phydrax.discretization.finite_volume._unstructured_remap import (
        PreparedRemapCSRView,
        UnstructuredConservativeRemapPlan,
    )

    with _budget() as budget:
        chain = _split_chain(budget)
        coarse, fine = _fv_endpoints(chain)
        prolongation = prepare_unstructured_conservative_remap(
            coarse, fine, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        ).plan
        assert prolongation is not None
        policy = phx.linalg.SparseAssemblyPolicy()
        view = prolongation.prepare_csr_view(policy, budget=budget)
        rebuilt = prolongation.prepare_csr_view(policy, budget=budget)
    restriction = UnstructuredConservativeRemapPlan(
        fine,
        coarse,
        np.asarray((0, fine.cell_count), dtype=np.int32),
        np.arange(fine.cell_count, dtype=np.int32),
        fine.cell_volumes,
        method="nested-volume-restriction",
        provenance="volume-material-chain-test",
    )

    def hierarchy(
        plan: UnstructuredConservativeRemapPlan, prolongation_view: PreparedRemapCSRView
    ) -> UnstructuredAMRHierarchyPlan:
        return UnstructuredAMRHierarchyPlan(
            coarse, fine, plan, restriction,
            maximum_refined_cells=1, prolongation_view=prolongation_view,
        )

    admitted = hierarchy(prolongation, view)
    # An equivalently rebuilt view is the same scientific binding.
    equivalent = hierarchy(prolongation, rebuilt)
    assert equivalent.plan_id == admitted.plan_id
    assert equivalent.coverage_evidence_id == admitted.coverage_evidence_id
    np.testing.assert_array_equal(np.asarray(admitted.fine_parent_cells), np.zeros(fine.cell_count))
    # A live view whose rows were replaced keeps its carried identity but is refused.
    forged_view = eqx.tree_at(
        lambda value: value.intersection_measures, view, view.intersection_measures * 0.5,
    )
    assert forged_view.view_id == view.view_id
    with pytest.raises(ValueError, match="rows derived from its plan"):
        hierarchy(prolongation, forged_view)
    # A prolongation whose weights were replaced keeps its plan_id but is refused.
    assert prolongation.volume_material_chain is not None
    forged_plan = eqx.tree_at(
        lambda value: value.volume_material_chain.weight_mantissas,
        prolongation,
        np.asarray(prolongation.volume_material_chain.weight_mantissas) * 0.75,
    )
    assert forged_plan.plan_id == prolongation.plan_id
    with pytest.raises(ValueError, match="admitted plan's content"):
        hierarchy(forged_plan, view)


def test_material_plan_and_explicit_view_survive_array_archive(tmp_path: Path) -> None:
    from phydrax._array_archive import read_array_archive, write_array_archive
    from phydrax._model._structure import (
        model_from_logical_array_recipe,
        model_recipe_array_values,
        model_recipe_from_wire_json,
        model_recipe_wire_json,
        model_structure_recipe,
    )
    from phydrax.discretization.finite_volume._unstructured_remap import (
        PreparedRemapCSRView,
        UnstructuredConservativeRemapPlan,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _fv_endpoints(chain)
        plan = prepare_unstructured_conservative_remap(
            source, target, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        ).plan
        assert plan is not None
        view = plan.prepare_csr_view(phx.linalg.SparseAssemblyPolicy(), budget=budget)
    contents = np.asarray(source.cell_volumes)[:, None] * np.asarray([[2.0, -3.0]])
    expected = np.asarray(plan.apply_content(contents))
    values = (plan, view)
    recipe = model_structure_recipe(values)
    arrays = model_recipe_array_values(values, recipe, prefix="material-transfer")
    path = write_array_archive(
        tmp_path / "material-transfer.zip",
        manifest={"recipe": model_recipe_wire_json(recipe)},
        arrays=arrays,
    )
    manifest, stored = read_array_archive(path)
    restored = model_from_logical_array_recipe(
        model_recipe_from_wire_json(manifest["recipe"]), stored, prefix="material-transfer"
    )
    restored_plan, restored_view = restored
    assert isinstance(restored_plan, UnstructuredConservativeRemapPlan)
    assert isinstance(restored_view, PreparedRemapCSRView)
    # Cold restoration keeps the content binding: the restored view admits
    # against the restored plan with the same identity, and a fresh
    # derivation from that plan reproduces its rows.
    restored_view.require_bound(restored_plan)
    assert restored_view.view_id == view.view_id
    assert restored_view.plan_content == view.plan_content
    rebuilt = PreparedRemapCSRView(restored_plan, restored_view.policy)
    assert rebuilt.view_id == view.view_id
    for name in ("target_offsets", "source_indices", "target_routes", "intersection_measures"):
        np.testing.assert_array_equal(
            np.asarray(getattr(rebuilt, name)), np.asarray(getattr(view, name))
        )
    np.testing.assert_array_equal(np.asarray(restored_plan.apply_content(contents)), expected)
    np.testing.assert_allclose(
        np.sum(np.asarray(restored_plan.apply_content(contents)), axis=0),
        np.sum(contents, axis=0), rtol=1e-14, atol=0.0,
    )
    target_contents = np.bincount(
        np.asarray(restored_view.target_routes),
        weights=np.asarray(restored_view.intersection_measures),
        minlength=target.cell_volumes.size,
    )
    dynamic = np.asarray(restored_view.apply_fixed_combinatorics(
        np.asarray([[2.0, -3.0]]), restored_view.intersection_measures,
        source.cell_volumes, target_contents,
    ))
    np.testing.assert_allclose(
        dynamic, np.broadcast_to(np.asarray([2.0, -3.0]), dynamic.shape),
        rtol=1e-14, atol=0.0,
    )


def test_float32_material_remap_certifies_its_own_precision() -> None:
    # With JAX x64 off the report actions run in binary32 and certify binary32
    # bounds: the default 1e-10 coverage tolerance refuses them, while an
    # explicit tolerance looser than binary32 roundoff admits the plan.
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _fv_endpoints(chain)
    with phx.config.context(phx.config.NumericsConfig(enable_x64=False)):
        with _budget() as budget, pytest.raises(ValueError, match="insufficient certified accuracy"):
            prepare_unstructured_conservative_remap(
                source, target, provenance="volume-material-chain-test",
                volume_material_chain=chain, budget=budget,
            )
        with _budget() as budget:
            plan = prepare_unstructured_conservative_remap(
                source, target, provenance="volume-material-chain-test",
                policy=CommonRefinementPolicy(coverage_tolerance=1e-5),
                volume_material_chain=chain, budget=budget,
            ).plan
        assert plan is not None
        contents = np.asarray(source.cell_volumes, dtype=np.float32)[:, None] * np.float32(2.0)
        carried = np.asarray(plan.apply_content(contents))
    assert carried.dtype == np.float32
    np.testing.assert_allclose(np.sum(carried, axis=0), np.sum(contents, axis=0), rtol=1e-5, atol=0.0)


def test_material_remap_refuses_weights_whose_bounds_would_widen_coverage() -> None:
    with _budget() as budget:
        chain = _chain(budget)
        # Mantissa bounds of half the mantissa make every weight uncertain by a factor two.
        coarse = eqx.tree_at(
            lambda value: value.weight_mantissa_errors, chain, np.asarray(chain.weight_mantissas) / 2.0,
        )
        source, target = _fv_endpoints(chain)
        with pytest.raises(ValueError, match="insufficient certified accuracy"):
            prepare_unstructured_conservative_remap(
                source, target, provenance="volume-material-chain-test",
                volume_material_chain=coarse, budget=budget,
            )


def test_material_accuracy_gate_encloses_binary64_aggregation() -> None:
    from phydrax.discretization.finite_volume._unstructured_remap import (
        require_certified_material_accuracy,
    )

    tolerance = 2.0**-10
    content = np.asarray((1025.0,))
    sources, exact_volumes = np.asarray((2.0**20,)), np.zeros(1)
    covered = np.zeros(1)
    # A nearest-rounded check would admit the bound 1 + 2**-52 against
    # 2**-10 * 1025; the certified content 1025 - (1 + 2**-52), rounded down,
    # leaves an allowance below one.
    with pytest.raises(ValueError, match="insufficient certified accuracy"):
        require_certified_material_accuracy(
            content, np.asarray((1.0 + 2.0**-52,)), covered, sources, exact_volumes, tolerance,
        )
    require_certified_material_accuracy(
        content, np.asarray((0.5,)), covered, sources, exact_volumes, tolerance,
    )


def test_compressive_material_remap_judges_target_accuracy_by_carried_content() -> None:
    # Collapsing a tall lifted spike back onto its base compresses about 1e9:
    # each target carries the spike's old-frame content, not the small new
    # physical volume, so certified bounds are judged against that content.
    tall = np.asarray(((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.25, 0.25, 2.0**30)))
    exact = tuple(_exact(row) for row in tall)
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 1, "ca"))
    spike = np.asarray(_INSERTED)
    base = np.asarray(((0, 1, 2, 3),))
    with _budget() as budget:
        before = CavitySnapshot(
            exact, _INSERTED, ((1, 2, 4), (2, 3, 4), (3, 1, 4)), ("source-face",) * 3, (23, 24, 25), rim,
        )
        after = CavitySnapshot(exact[:4], ((0, 1, 2, 3),), ((1, 2, 3),), ("source-face",), (17,), rim)
        atlas = build_implicit_cavity_atlas(
            prepare_implicit_cavity_edit(
                before, after, source_id="original-source", discovery_id="original-discovery",
                chronology_index=0, predecessor_id="", budget=budget,
            ),
            chart_axes=(0, 1), anchor=(F(1, 4), F(1, 4), F(-1, 4)), budget=budget,
        )
        source_mesh = CellMesh.from_tetrahedra(tall, spike, cell_global_ids=np.asarray((23, 24, 25)))
        target_mesh = CellMesh.from_tetrahedra(tall[:4], base, cell_global_ids=np.asarray((17,)))
        edit = _curved_edit(
            np.asarray((23, 24, 25)), spike, tall[spike], np.asarray((17,)), base, tall[base], atlas,
        )
        chain = VolumeMaterialChain(
            source_mesh, CellGeometrySpec.affine(source_mesh), target_mesh,
            CellGeometrySpec.affine(target_mesh), (edit,),
            original_source_id="original-source", discovery_id="original-discovery",
            maximum_work_units=_WORK, maximum_scratch_bytes=_BYTES,
            atlases={atlas.chart_id: atlas}, budget=budget,
        )
        source, target = _fv_endpoints(chain)
        prepared = prepare_unstructured_conservative_remap(
            source, target, provenance="volume-material-chain-test",
            volume_material_chain=chain, budget=budget,
        )
    assert prepared.plan is not None
    target_volume = float(np.asarray(target.cell_volumes)[0])
    carried_content = np.asarray(prepared.plan.apply_content(np.asarray(source.cell_volumes)))
    # Strong compression: the carried content dwarfs the new physical volume.
    assert carried_content[0] > 1e5 * target_volume
    # The explicit view materializes the same old-frame content rows; its
    # construction never imposes the physical-volume target check.
    view = prepared.plan.prepare_csr_view(phx.linalg.SparseAssemblyPolicy())
    row_contents = np.add.reduceat(
        np.asarray(view.intersection_measures), np.asarray(view.target_offsets)[:-1]
    )
    np.testing.assert_allclose(row_contents, carried_content, rtol=1e-12, atol=0.0)
    contents = np.asarray(source.cell_volumes)[:, None] * np.asarray(((2.0,), (2.0,), (2.0,)))
    carried = np.asarray(prepared.plan.apply_content(contents))
    np.testing.assert_allclose(np.sum(carried, axis=0), np.sum(contents, axis=0), rtol=1e-12, atol=0.0)


def test_identity_edit_must_preserve_its_oriented_cavity_boundary() -> None:
    moved = _POINTS.copy()
    # Relocating the lifted boundary vertex is not a domain-preserving edit.
    moved[4] = (0.25, 0.25, 0.0625)
    relocation = _edit((23,), _INSERTED[:1], (23,), _INSERTED[:1], points=moved)
    with _budget() as budget, pytest.raises(VolumeMaterialMapError) as failure:
        _chain(budget, split=relocation)
    assert failure.value.status is VolumeMaterialStatus.COVERAGE_GAP


def test_chain_must_reproduce_the_target_by_identity_and_exact_geometry() -> None:
    stale = _edit((23,), _INSERTED[:1], (26, 27, 28, 30), _SPLIT)
    with _budget() as budget, pytest.raises(VolumeMaterialMapError) as failure:
        _chain(budget, split=stale)
    assert failure.value.status is VolumeMaterialStatus.FOREIGN_ENDPOINT


def test_cold_reproof_rebinds_identity_and_refuses_changed_geometry() -> None:
    with _budget() as budget:
        chain = _chain(budget)

    def reprove(limit: int) -> tuple[VolumeMaterialChain, int]:
        # One request envelope; only the reproof stage bound varies.
        with _budget() as budget:
            start = budget.work_units
            with budget.bound_stage(limit, _BYTES):
                restored = VolumeMaterialChain.from_stored_fields(chain, budget=budget)
            return restored, budget.work_units - start

    restored, required = reprove(_WORK)
    assert restored.chain_id == chain.chain_id
    for name in ("piece_routes", "piece_simplex_vertices", "piece_vertex_offsets", "piece_vertices"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(chain, name))
    for name in (
        "weight_offsets", "weight_targets", "weight_sources", "weight_mantissas",
        "weight_exponents", "weight_mantissa_errors",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(chain, name))
    # The measured reproof allowance is exactly sufficient; one unit less refuses.
    assert reprove(required)[0].chain_id == chain.chain_id
    with pytest.raises(CoordinateEnclosureResourceError):
        reprove(required - 1)
    shifted = np.asarray(chain.edits[1].target_points).copy()
    shifted[0, 0, 0] = np.nextafter(shifted[0, 0, 0], 1.0)
    forged = eqx.tree_at(lambda value: value.edits[1].target_points, chain, shifted)
    with _budget() as budget, pytest.raises(VolumeMaterialMapError):
        VolumeMaterialChain.from_stored_fields(forged, budget=budget)


_P1 = ("u", "density", "history")


def _spaces(
    chain: VolumeMaterialChain,
    fields: tuple[phx.discretization.FiniteElementFieldSpec, ...],
    budget: CoordinateEnclosureBudget,
) -> tuple[phx.discretization.FiniteElementDiscretization, ...]:
    return tuple(
        phx.discretization.FiniteElementPlan(mesh, fields, coordinate_spec=geometry).prepare(budget=budget)
        for mesh, geometry in (
            (chain.source_mesh, chain.source_geometry), (chain.target_mesh, chain.target_geometry),
        )
    )


def _vertex_rows(space: phx.discretization.FiniteElementDiscretization, field: int) -> np.ndarray:
    """P1 DOF of every mesh vertex row, from the actual cell routes."""
    cells = np.concatenate([np.asarray(b.vertices) for b in space.mesh.blocks])
    routes = np.concatenate([np.asarray(r) for r in space.dof_maps[field].cell_dofs])
    result = np.full((space.mesh.coordinates.shape[0],), -1, dtype=np.int64)
    result[cells.ravel()] = routes.ravel()
    return result


def test_p1_roles_match_the_independent_numpy_replay() -> None:
    from phydrax.meshing.qualification._envelope import material_chain_reference

    semantics: dict[str, SourceRealizationFieldSemantics] = {
        "u": "intensive", "density": "conservative-density", "history": "intensive",
    }
    fields = tuple(
        phx.discretization.FiniteElementFieldSpec(name, phx.discretization.lagrange_element("tetrahedron", 1))
        for name in _P1
    )
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _spaces(chain, fields, budget)
        transfers = prepare_volume_material_fe_transfers(
            chain, source, target, semantics=semantics, budget=budget,
        )
    assert all(transfer.evidence.passed for transfer in transfers)
    assert transfers[0].transfer.preserves_constants and transfers[2].transfer.preserves_constants
    assert transfers[1].transfer.conservative
    coordinates = np.asarray(chain.source_mesh.coordinates)
    authored = {
        "u": 1.0 + coordinates[:, 0] - 2.0 * coordinates[:, 2],
        "density": 2.0 + coordinates[:, 1],
        "history": coordinates[:, 0] + 0.5,
    }
    source_ids = np.asarray(chain.source_mesh.vertex_global_ids).tolist()
    target_ids = np.asarray(chain.target_mesh.vertex_global_ids).tolist()
    with _budget() as budget:
        reference = material_chain_reference(
            chain,
            {name: {v: authored[name][row:row + 1] for row, v in enumerate(source_ids)} for name in _P1},
            semantics,
            budget=budget,
        )
    assert reference.coverage_ratio <= 1.0
    assert np.isfinite(reference.condition_sum) and reference.condition_sum >= 1.0
    for index, (name, transfer) in enumerate(zip(_P1, transfers, strict=True)):
        values = np.zeros((source.dof_maps[index].global_dof_count,))
        values[_vertex_rows(source, index)] = authored[name]
        carried = np.asarray(transfer.transfer.apply(values))[_vertex_rows(target, index)]
        expected = np.asarray([reference.values[name][v][0] for v in target_ids])
        np.testing.assert_allclose(carried, expected, rtol=0.0, atol=1e-12)
        assert np.all(np.isfinite(reference.l2_errors[name]))
    density = np.asarray(transfers[1].transfer.apply(np.full((4,), 2.0)))
    target_cells = np.concatenate([np.asarray(b.vertices) for b in chain.target_mesh.blocks])
    np.testing.assert_allclose(
        _volumes(chain.target_mesh) @ density[_vertex_rows(target, 1)][target_cells].mean(axis=1),
        2.0 / 6.0, rtol=1e-13, atol=0.0,
    )
    np.testing.assert_allclose(np.asarray(transfers[0].transfer.apply(np.full((4,), 3.0))), 3.0, rtol=0.0, atol=1e-13)


def _content(
    space: phx.discretization.FiniteElementDiscretization, field: int, values: np.ndarray,
) -> np.ndarray:
    from phydrax.discretization.fem._topology_transfer import _cell_integrals

    measures = np.asarray(_cell_integrals(space, field, mass=False)[0])
    return np.tensordot(values, measures, axes=(0, 0))


def _projected_constant(
    space: phx.discretization.FiniteElementDiscretization, field: int, vector: np.ndarray,
) -> np.ndarray:
    """Independent dense L2 coefficients of one constant vector field."""
    from phydrax.discretization.fem._topology_transfer import _cell_integrals

    measures, triples = _cell_integrals(space, field, mass=True)
    assert triples is not None
    size = space.dof_maps[field].global_dof_count
    mass = np.zeros((size, size))
    np.add.at(mass, (triples[0], triples[1]), triples[2])
    return np.linalg.solve(mass, np.asarray(measures) @ vector)


def _cone_maps(chain: VolumeMaterialChain) -> list[tuple[np.ndarray, float, float]]:
    """Independent float cone maps ``(L, J, |C|)`` of the curved edit."""
    edit = chain.edits[0]
    result = []
    for origin, image in zip(edit.source_charts, edit.target_charts, strict=True):
        old = np.asarray(origin, dtype=np.float64)
        new = np.asarray(image, dtype=np.float64)
        linear = np.linalg.solve((old[1:] - old[:1]), (new[1:] - new[:1])).T
        result.append((linear, float(np.linalg.det(linear)), float(abs(np.linalg.det(old[1:] - old[:1]))) / 6.0))
    return result


@pytest.mark.parametrize(("form_degree", "proxy"), ((1, "circulation"), (2, "flux")))
def test_piola_material_fields_carry_their_pulled_vector_content(form_degree: int, proxy: FormProxy) -> None:
    twist = "twisted" if form_degree == 2 else None
    element = phx.discretization.form_element("tetrahedron", form_degree, 1, twist=twist, proxy=proxy)
    fields = (phx.discretization.FiniteElementFieldSpec("w", element),)
    vector = np.asarray((0.5, -1.0, 2.0))
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _spaces(chain, fields, budget)
        (transfer,) = prepare_volume_material_fe_transfers(
            chain, source, target, semantics={"w": "material-compatible"}, budget=budget,
        )
    assert transfer.evidence.passed
    assert transfer.semantics == ("covariant-piola" if form_degree == 1 else "contravariant-piola")
    carried = np.asarray(transfer.transfer.apply(_projected_constant(source, 0, vector)))
    # Pullbacks: L^-T w (one-forms, measure J) or L w / J (fluxes).
    expected = sum(
        (jacobian * np.linalg.solve(linear.T, vector) if form_degree == 1 else linear @ vector) * volume
        for linear, jacobian, volume in _cone_maps(chain)
    )
    np.testing.assert_allclose(_content(target, 0, carried), expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("element", (
    phx.discretization.lagrange_element("tetrahedron", 2),
    phx.discretization.discontinuous_element("tetrahedron", 1),
))
def test_higher_order_scalar_spaces_preserve_constants_and_density(
    element: phx.discretization.FiniteElementSpec,
) -> None:
    fields = tuple(phx.discretization.FiniteElementFieldSpec(name, element) for name in ("u", "density"))
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _spaces(chain, fields, budget)
        intensive, density = prepare_volume_material_fe_transfers(
            chain, source, target,
            semantics={"u": "intensive", "density": "conservative-density"}, budget=budget,
        )
    assert intensive.evidence.passed and density.evidence.passed
    size = source.dof_maps[0].global_dof_count
    np.testing.assert_allclose(np.asarray(intensive.transfer.apply(np.full((size,), 3.0))), 3.0, rtol=0.0, atol=1e-12)
    values = np.linspace(0.5, 2.0, source.dof_maps[1].global_dof_count)
    carried = np.asarray(density.transfer.apply(values))
    np.testing.assert_allclose(
        _content(target, 1, carried), _content(source, 1, values), rtol=1e-13, atol=0.0,
    )


def test_local_projection_refuses_mismatched_semantics() -> None:
    fields = (phx.discretization.FiniteElementFieldSpec("u", phx.discretization.lagrange_element("tetrahedron", 1)),)
    with _budget() as budget:
        chain = _chain(budget)
        source, target = _spaces(chain, fields, budget)
        with pytest.raises(VolumeMaterialMapError) as failure:
            prepare_volume_material_fe_transfers(
                chain, source, target, semantics={"u": "material-compatible"}, budget=budget,
            )
    assert failure.value.status is VolumeMaterialStatus.UNSUPPORTED_SPACE


@pytest.fixture(scope="module")
def sphere_chain() -> tuple[phx.meshing.CellMeshingResult, VolumeMaterialChain]:
    """A genuine lifted insertion and its collapse on a certified implicit sphere."""
    from phydrax.units import METER
    from tests._support.implicit_meshing import (
        IMPLICIT_DOMAIN,
        implicit_query,
        implicit_specification,
    )

    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 0.371).compile(coordinate_unit=METER)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(5) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(1.2 * IMPLICIT_DOMAIN)
    source = phx.meshing.NativeImplicitSource(
        geometry, grid, "implicit", "state", discovery=implicit_query(geometry),
    )
    scope = implicit_specification().boundary_scope
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))),
        scope, phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.2, strength=phx.meshing.SizeControlStrength.SOFT),),
    )
    result = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("implicit_restricted_delaunay"),
    ).plan(source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()).execute()
    assert isinstance(result, phx.meshing.CellMeshingResult)
    assert result.certification is not None and result.certification.passed
    query = result.certification.request.source
    assert isinstance(query, phx.geometry.ImplicitProjectionBoundarySource) and query.discovery is not None
    mesh = result.mesh
    rows = np.asarray(mesh.blocks[0].vertices)
    points = np.asarray(mesh.coordinates)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cell_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    owners: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for cell, row in enumerate(rows):
        for opposite in range(4):
            owners.setdefault(tuple(sorted(np.delete(row, opposite).tolist())), []).append((cell, int(row[opposite])))
    best = None
    for face, incident in owners.items():
        if len(incident) != 1:
            continue
        cell, apex = incident[0]
        normal = np.cross(points[face[1]] - points[face[0]], points[face[2]] - points[face[0]])
        if normal @ (points[apex] - points[face[0]]) > 0.0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        if best is None or normal[2] > best[0]:
            best = (normal[2], cell, apex, face, normal)
    assert best is not None
    _, cell, apex, face, normal = best
    a, b, c = face
    if np.linalg.det(np.stack([points[v] - points[apex] for v in (a, b, c)])) < 0.0:
        a, b = b, a
    local = np.asarray((apex, a, b, c))
    edge = min(np.linalg.norm(points[u] - points[v]) for u, v in ((a, b), (b, c), (c, a)))
    lifted = points[[a, b, c]].mean(axis=0) + 0.05 * edge * normal
    exact = tuple(
        (F(float(point[0])), F(float(point[1])), F(float(point[2])))
        for point in np.concatenate((points[local], lifted[None]))
    )
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 1, "ca"))
    created = tuple(int(cell_ids.max()) + offset for offset in (1, 2, 3))
    before = CavitySnapshot(exact[:4], ((0, 1, 2, 3),), ((1, 2, 3),), ("source-face",), (int(cell_ids[cell]),), rim)
    after = CavitySnapshot(
        exact, _INSERTED, ((1, 2, 4), (2, 3, 4), (3, 1, 4)), ("source-face",) * 3, created, rim,
    )
    anchor = tuple(sum((point[axis] for point in exact[:4]), F(0)) / 4 for axis in range(3))
    # Exact clipping of real binary64 cones needs the old sphere fixture's scratch.
    with _budget(memory=_SPHERE_BYTES) as budget:
        atlases = []
        for index, (old, new) in enumerate(((before, after), (after, before))):
            edit = prepare_implicit_cavity_edit(
                old, new, source_id=query.source_id, discovery_id=query.discovery.query_id,
                chronology_index=index, predecessor_id="" if index == 0 else atlases[0].chart_id,
                budget=budget,
            )
            atlases.append(build_implicit_cavity_atlas(edit, chart_axes=(0, 1), anchor=anchor, budget=budget))
        bank = np.concatenate((points[local], lifted[None]))
        identifiers = np.concatenate((vertex_ids[local], (int(vertex_ids.max()) + 1,)))
        old_rows, new_rows = np.asarray(((0, 1, 2, 3),)), np.asarray(_INSERTED)
        edits = (
            _curved_edit(
                np.asarray((cell_ids[cell],)), identifiers[old_rows], bank[old_rows],
                np.asarray(created), identifiers[new_rows], bank[new_rows], atlases[0],
            ),
            _curved_edit(
                np.asarray(created), identifiers[new_rows], bank[new_rows],
                np.asarray((cell_ids[cell],)), identifiers[old_rows], bank[old_rows], atlases[1],
            ),
        )
        chain = VolumeMaterialChain(
            mesh, result.geometry, mesh, result.geometry, edits,
            original_source_id=query.source_id, discovery_id=query.discovery.query_id,
            maximum_work_units=_WORK, maximum_scratch_bytes=_SPHERE_BYTES,
            atlases={atlas.chart_id: atlas for atlas in atlases}, budget=budget,
        )
    return result, chain


def test_material_transition_rejects_foreign_exact_source_geometry(
    sphere_chain: tuple[phx.meshing.CellMeshingResult, VolumeMaterialChain],
) -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._exact_algebraic_geometry import (
        ExactAlgebraicCellGeometrySource,
    )
    from phydrax.discretization._exact_rational_geometry import rational_source_layouts
    from phydrax.meshing._lineage import CellMeshTransition, identity_lineage
    from phydrax.meshing._lineage_records import MeshTransitionKind

    result, chain = sphere_chain
    mesh = result.mesh
    with _budget(memory=_SPHERE_BYTES) as budget:
        transition = CellMeshTransition(
            mesh.mesh_id, mesh.topology_id, result, identity_lineage(mesh, mesh),
            MeshTransitionKind.REMESH, source_geometry=result.geometry,
            volume_material_chain=chain, budget=budget,
        )
        assert CellMeshTransition.from_stored_fields(transition, budget=budget).transition_id == transition.transition_id
    elements = {block.name: coordinate_lagrange_element("tetrahedron", 1) for block in mesh.blocks}
    routes = {block.name: block.vertices for block in mesh.blocks}
    row, axis = np.argwhere(np.asarray(mesh.coordinates) != 0.0)[0]
    spacing = F(abs(float(np.spacing(float(mesh.coordinates[row, axis])))))
    # A positive rational displacement below half an ulp keeps the rounded
    # carrier and changes only the exact volume realization.
    exact = [[F(float(value)) for value in point] for point in np.asarray(mesh.coordinates)]
    exact[row][axis] += spacing / 8
    np.testing.assert_array_equal(np.asarray(exact, dtype=np.float64), mesh.coordinates)
    foreign = CellGeometrySpec(
        elements, routes, mesh.coordinates,
        exact_source=ExactAlgebraicCellGeometrySource(
            tuple(tuple(point) for point in exact), rational_source_layouts(elements, routes),
            source_identity="foreign-exact-material-source",
            maximum_work=_WORK, maximum_bits=4096, budget=None,
        ),
        budget=None,
    )
    assert cell_geometry_id(foreign) != cell_geometry_id(chain.source_geometry)
    with pytest.raises(ValueError, match="actual endpoint meshes and geometry"):
        CellMeshTransition(
            transition.source_mesh_id, transition.source_topology_id,
            transition.target, transition.lineage, transition.transition_kind,
            source_geometry=foreign, volume_material_chain=chain,
        )
    forged = eqx.tree_at(lambda value: value.source_geometry, transition, foreign)
    with pytest.raises(ValueError, match="actual endpoint meshes and geometry"):
        CellMeshTransition.from_stored_fields(forged, budget=None)
    with pytest.raises(ValueError, match="actual endpoint meshes and geometry"):
        CellMeshTransition(
            transition.source_mesh_id, transition.source_topology_id,
            transition.target, transition.lineage, transition.transition_kind,
            volume_material_chain=chain,
        )


def test_nonbinary_chart_coverage_uses_exact_source_not_rounded_carrier() -> None:
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._exact_algebraic_geometry import (
        ExactAlgebraicCellGeometrySource,
    )
    from phydrax.discretization._exact_rational_geometry import rational_source_layouts
    from phydrax.geometry._supermesh import (
        ExactTetrahedralRefinementError,
        prepare_common_refinement,
    )

    with _budget() as budget:
        points = (
            (F(0), F(0), F(0)),
            (F(1), F(0), F(0)),
            (F(0), F(1), F(0)),
            (F(0), F(0), F(1)),
            (F(1, 3), F(2, 3), F(0)),
        )
        source = phx.discretization.CellMesh.from_tetrahedra(np.asarray(points[:4], dtype=np.float64), np.asarray(((0, 1, 2, 3),)))
        target = phx.discretization.CellMesh.from_tetrahedra(np.asarray(points, dtype=np.float64), np.asarray(((0, 1, 4, 3), (0, 4, 2, 3))))
        name = target.blocks[0].name
        elements = {name: coordinate_lagrange_element("tetrahedron", 1)}
        routes = {name: target.blocks[0].vertices}
        exact = ExactAlgebraicCellGeometrySource(
            points, rational_source_layouts(elements, routes),
            source_identity="nonbinary-material-chart", maximum_work=20_000_000,
            maximum_bits=4096, budget=budget,
        )
        geometry = CellGeometrySpec(elements, routes, target.coordinates, exact_source=exact, budget=budget)
        refinement = prepare_common_refinement(
            source, target, source_geometry=CellGeometrySpec.affine(source),
            target_geometry=geometry, policy=CommonRefinementPolicy(overlap_simplices=True),
            budget=budget,
        )
        assert refinement.succeeded
        np.testing.assert_array_equal(refinement.evidence.source_coverage_defects, 0.)
        np.testing.assert_array_equal(
            refinement.target_measures,
            np.asarray((float(F(1, 9)), float(F(1, 18)))),
        )
        with pytest.raises(ExactTetrahedralRefinementError) as failure:
            prepare_common_refinement(
                source, target, source_geometry=CellGeometrySpec.affine(source),
                target_geometry=CellGeometrySpec.affine(target),
                policy=CommonRefinementPolicy(overlap_simplices=True), budget=budget,
            )
        assert failure.value.status == CommonRefinementStatus.COVERAGE_GAP


def test_complete_volume_mismatch_refuses_before_candidate_admission() -> None:
    from phydrax.geometry._supermesh import (
        ExactTetrahedralRefinementError,
        prepare_common_refinement,
    )

    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.float64,
    )
    source = CellMesh.from_tetrahedra(vertices, np.asarray(((0, 1, 2, 3),)))
    target_vertices = np.concatenate((vertices, ((0.5, 0., 0.),)))
    target_vertices[:, 0] *= 0.5
    target = CellMesh.from_tetrahedra(
        target_vertices, np.asarray(((0, 4, 2, 3), (4, 1, 2, 3))),
    )
    with _budget() as budget:
        with pytest.raises(ExactTetrahedralRefinementError) as failure:
            prepare_common_refinement(
                source, target,
                source_geometry=CellGeometrySpec.affine(source),
                target_geometry=CellGeometrySpec.affine(target),
                policy=CommonRefinementPolicy(
                    overlap_simplices=True, maximum_candidate_pairs=1,
                ),
                budget=budget,
            )
        assert failure.value.status == CommonRefinementStatus.COVERAGE_GAP
        assert "equal total volumes" in str(failure.value)


def test_equal_total_volume_does_not_certify_local_coverage() -> None:
    from phydrax.geometry._supermesh import (
        ExactTetrahedralRefinementError,
        prepare_common_refinement,
    )

    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.float64,
    )
    source = CellMesh.from_tetrahedra(vertices, np.asarray(((0, 1, 2, 3),)))
    target = source.with_coordinates(vertices + 2., numeric_version="disjoint")
    with _budget() as budget:
        with pytest.raises(ExactTetrahedralRefinementError) as failure:
            prepare_common_refinement(
                source, target,
                source_geometry=CellGeometrySpec.affine(source),
                target_geometry=CellGeometrySpec.affine(target),
                policy=CommonRefinementPolicy(overlap_simplices=True), budget=budget,
            )
        assert failure.value.status == CommonRefinementStatus.COVERAGE_GAP
        assert "equal total volumes" not in str(failure.value)


def test_cell_associated_layout_names_cell_entities_and_local_positions() -> None:
    from phydrax.discretization.fem._assembly import (
        lifted_dof_entities,
        lifted_finite_element_dof_layout,
    )

    _, target = _endpoints()
    element = phx.discretization.discontinuous_element("tetrahedron", 1)
    layout = lifted_finite_element_dof_layout(target, (element,), ())
    degrees, entities, positions, widths = lifted_dof_entities(target, layout)
    width = element.local_dof_count
    cells = sum(block.cell_count for block in target.blocks)
    np.testing.assert_array_equal(degrees, 3)
    np.testing.assert_array_equal(entities, np.repeat(np.arange(cells), width))
    np.testing.assert_array_equal(positions, np.tile(np.arange(width), cells))
    np.testing.assert_array_equal(widths[3], width)


@pytest.mark.parametrize("element", (
    phx.discretization.lagrange_element("tetrahedron", 4),
    phx.discretization.discontinuous_element("tetrahedron", 3),
))
def test_permuted_target_corners_carry_the_same_fields(
    element: phx.discretization.FiniteElementSpec,
) -> None:
    from phydrax.discretization.fem._topology_transfer import _overlap_basis

    fields = tuple(phx.discretization.FiniteElementFieldSpec(name, element) for name in ("u", "density"))
    weights = np.asarray(((0.1, 0.2, 0.3, 0.4), (0.25, 0.25, 0.25, 0.25), (0.7, 0.1, 0.1, 0.1)))
    evaluated = []
    with _budget() as budget:
        for permuted in (False, True):
            chain = _chain(budget, permuted=permuted)
            source, target = _spaces(chain, fields, budget)
            transfers = prepare_volume_material_fe_transfers(
                chain, source, target,
                semantics={"u": "intensive", "density": "conservative-density"}, budget=budget,
            )
            assert all(transfer.evidence.passed for transfer in transfers)
            mesh = target.mesh
            corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
            ids = np.asarray(mesh.vertex_global_ids)[np.asarray(mesh.blocks[0].vertices)]
            # Identical physical points per cell: corners ordered by vertex ID.
            ordered = np.take_along_axis(corners, np.argsort(ids, axis=1)[:, :, None], axis=1)
            points = weights @ ordered
            cells = np.arange(points.shape[0])
            values = []
            for index, transfer in enumerate(transfers):
                carried = np.asarray(transfer.transfer.apply(
                    np.linspace(0.5, 2.0, source.dof_maps[index].global_dof_count)
                ))
                routes, valid, basis, _ = _overlap_basis(target, index, cells, points)
                values.append(np.einsum("cqi,ci->cq", basis[..., 0] * valid[:, None, :], carried[routes]))
            evaluated.append(values)
    for actual, expected in zip(evaluated[1], evaluated[0], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-11)


def test_reference_clipping_keeps_thin_positive_slivers_and_drops_contacts() -> None:
    from phydrax.meshing.qualification._envelope import _reference_clip

    unit = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))[None]
    for thickness in (1e-4, 1e-6):
        # The corner beyond x = 1 - thickness is a genuine sliver of volume t^3/6.
        sliver = _reference_clip(unit, np.asarray((-1.0, 0.0, 0.0)), -(1.0 - thickness))
        volume = np.sum(np.abs(np.linalg.det(sliver[:, 1:] - sliver[:, :1]))) / 6.0
        np.testing.assert_allclose(volume, thickness**3 / 6.0, rtol=1e-6)
    contact = _reference_clip(unit, np.asarray((1.0, 0.0, 0.0)), 0.0)
    assert np.sum(np.abs(np.linalg.det(contact[:, 1:] - contact[:, :1]))) == 0.0


type _ArrangementInputs = tuple[
    tuple[tuple[F, F, F], ...],
    tuple[tuple[F, F, F, F], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _unit_arrangement() -> _ArrangementInputs:
    """The unit tetrahedron, one job ``x <= 1/4`` and its child ``y <= 1/4``."""
    zero, one = F(0), F(1)
    points = ((zero, zero, zero), (one, zero, zero), (zero, one, zero), (zero, zero, one))
    planes = (
        (one, zero, zero, zero),
        (zero, one, zero, zero),
        (zero, zero, one, zero),
        (one, one, one, -one),
        (one, zero, zero, F(-1, 4)),
        (zero, one, zero, F(-1, 4)),
    )
    return (
        points,
        planes,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        # Slot k is the outward face opposite corner k.
        np.asarray(((4, -1, -2, -3),), dtype=np.int32),
        np.asarray((-1, 0), dtype=np.int32),
        np.asarray((0, 1, 2), dtype=np.int64),
        np.asarray((5, 6), dtype=np.int32),
    )


_ARRANGEMENT_FIELDS = (
    "vertex_words", "vertex_word_offsets", "vertex_signs", "vertex_input_index",
    "cell_face_offsets", "face_offsets", "face_vertices", "face_planes",
)


def test_exact_arrangement_welds_faces_and_charges_native_work_once() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import (
        arrange_tetrahedra,
        ArrangementVertices,
        loop_volume_contribution,
    )

    inputs = _unit_arrangement()
    planes = inputs[1]
    with _budget() as host:
        expected = arrange_tetrahedra(*inputs, budget=host)
        required = host.work_units
        vertices = ArrangementVertices(expected, budget=host)
        offsets = np.asarray(expected.cell_face_offsets)
        loops = np.asarray(expected.face_offsets)
        volumes = []
        for job in range(2):
            six = sum(
                (
                    loop_volume_contribution(
                        tuple(
                            vertices[vertex]
                            for vertex in np.asarray(expected.face_vertices)[loops[face]:loops[face + 1]].tolist()
                        ),
                        planes[abs(int(expected.face_planes[face])) - 1],
                        budget=host,
                    )
                    for face in range(int(offsets[job]), int(offsets[job + 1]))
                ),
                F(0),
            )
            volumes.append(six / 6)
        exact = [vertices[vertex] for vertex in range(vertices.count)]
    # Exact face sums: the unit simplex below x = 1/4, then also below y = 1/4.
    assert volumes == [F(37, 384), F(3, 64)]
    coordinates = [tuple(F(value, row[3]) for value in row[:3]) for row in exact]
    # Welded: one published vertex per exact point, input points named by index.
    assert len(set(coordinates)) == len(coordinates)
    for coordinate, index in zip(coordinates, np.asarray(expected.vertex_input_index).tolist(), strict=True):
        assert (index >= 0) == (coordinate in inputs[0])
        if index >= 0:
            assert inputs[0][index] == coordinate
    with NativeExecutionScope(
        NativeExecutionLimits(required, 2, 0, _BYTES, np.inf)
    ) as scope:
        scope.observe_managed_admission()
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=required,
                maximum_memory_bytes=_BYTES,
            )
            with budget.activate():
                actual = arrange_tetrahedra(*inputs, budget=budget)
        for name in _ARRANGEMENT_FIELDS:
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        assert budget.work_units == required
        assert budget.native_charged_work_units == required
    assert scope.evidence is not None
    assert scope.evidence.status is MeshcoreStatus.OK
    assert scope.evidence.work_counters.construction_work == required


def test_exact_arrangement_records_partial_native_refusal() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import arrange_tetrahedra

    with pytest.raises(MeshcoreError) as refusal:
        with (
            NativeExecutionScope(
                NativeExecutionLimits(1000, 2, 0, _BYTES, np.inf)
            ) as scope,
            scope.ticket_view() as ticket,
        ):
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=1000,
                maximum_memory_bytes=_BYTES,
            )
            with budget.activate(), budget.bound_stage(3, _BYTES):
                arrange_tetrahedra(*_unit_arrangement(), budget=budget)
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert scope.evidence is not None
    partial = scope.evidence.work_counters.construction_work
    assert partial <= 3
    assert budget.work_units == budget.native_charged_work_units == partial


def _rational(row: tuple[int, int, int, int]) -> tuple[F, F, F]:
    return F(row[0], row[3]), F(row[1], row[3]), F(row[2], row[3])


def _piece_integrals(
    pieces: VolumeMaterialEditPieces, position: int,
) -> tuple[F, tuple[F, ...], tuple[tuple[F, ...], ...]]:
    """Exact volume, first and second moments of one piece from its positive simplices."""
    points = [_rational(row) for row in pieces.exact_vertices]
    volume, first = F(0), [F(0)] * 3
    second = [[F(0)] * 3 for _ in range(3)]
    start, stop = (int(value) for value in pieces.simplex_offsets[position:position + 2])
    for row in np.asarray(pieces.simplices)[start:stop].tolist():
        corners = [points[index] for index in row]
        u, v, w = ([corner[a] - corners[0][a] for a in range(3)] for corner in corners[1:])
        determinant = (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )
        # Pulling simplices are positive, never degenerate fan slivers.
        assert determinant > 0
        measure = determinant / 6
        sums = [sum((corner[a] for corner in corners), F(0)) for a in range(3)]
        volume += measure
        for a in range(3):
            first[a] += measure * sums[a] / 4
            for b in range(3):
                second[a][b] += measure * (
                    sums[a] * sums[b] + sum((corner[a] * corner[b] for corner in corners), F(0))
                ) / 20
    # The triangulation partitions the piece whose exact volume its faces sum.
    assert volume == pieces.volumes[position]
    return volume, tuple(first), tuple(tuple(row) for row in second)


def _split(created: tuple[int, ...], rows: tuple[tuple[int, ...], ...]) -> VolumeMaterialEdit:
    return _edit((23,), _INSERTED[:1], created, rows)


# A second binary64-exact interior split vertex of cell 23, barycentric (1/8, 1/8, 1/2, 1/4).
_RESPLIT_POINTS = np.concatenate((_POINTS, ((0.5625, 0.0625, -0.09375),)))
_RESPLIT = ((6, 1, 2, 4), (0, 6, 2, 4), (0, 1, 6, 4), (0, 1, 2, 6))


@pytest.mark.parametrize(("removed", "removed_rows", "created", "created_rows", "clipped"), (
    # Every target lies inside the source and is published whole.
    ((23,), _INSERTED[:1], (26, 27, 28, 29), _SPLIT, False),
    # Every source lies inside the target and is published whole.
    ((26, 27, 28, 29), _SPLIT, (23,), _INSERTED[:1], False),
    # Two different interior splits: overlapping pairs clip either cell.
    ((26, 27, 28, 29), _SPLIT, (30, 31, 32, 33), _RESPLIT, True),
))
def test_identity_pieces_match_independent_exact_overlaps(
    removed: tuple[int, ...], removed_rows: tuple[tuple[int, ...], ...],
    created: tuple[int, ...], created_rows: tuple[tuple[int, ...], ...], clipped: bool,
) -> None:
    from phydrax.geometry._exact_polyhedral_geometry import tetrahedron_overlap

    edit = _edit(removed, removed_rows, created, created_rows, points=_RESPLIT_POINTS)
    with _budget() as budget:
        pieces = volume_material_edit_pieces(edit, budget=budget)
        _, _, sources = edit.exact_cells("source", budget=budget)
        _, _, targets = edit.exact_cells("target", budget=budget)
        routes = {
            tuple(route): position for position, route in enumerate(np.asarray(pieces.routes).tolist())
        }
        partial = False
        for target, target_tet in enumerate(targets):
            for source, source_tet in enumerate(sources):
                oracle = tetrahedron_overlap(source_tet, target_tet, second_moments=True, budget=budget)
                assert isinstance(oracle, ExactTetrahedronOverlap) and oracle.second_moment is not None
                position = routes.get((target, source, -1))
                if position is None:
                    assert oracle.volume == 0
                    continue
                volume, first, second = _piece_integrals(pieces, position)
                assert (volume, first, second) == (
                    oracle.volume, tuple(oracle.first_moment), tuple(tuple(row) for row in oracle.second_moment),
                )
                partial |= volume < min(pieces.source_volumes[source], pieces.target_volumes[target])
    assert partial is clipped


def _solve(linear: tuple[tuple[F, ...], ...], right: tuple[F, ...]) -> tuple[F, ...]:
    """Exact Cramer solution of one 3x3 system."""
    def det(rows: tuple[tuple[F, ...], ...]) -> F:
        return (
            rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
            - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
            + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
        )

    whole = det(linear)
    return tuple(
        det(tuple(
            tuple(right[row] if column == replaced else linear[row][column] for column in range(3))
            for row in range(3)
        )) / whole
        for replaced in range(3)
    )


def test_curved_pieces_match_mapped_independent_exact_overlaps() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import tetrahedron_overlap

    with _budget() as budget:
        edit = _edit((17,), ((0, 1, 2, 3),), (23, 24, 25), _INSERTED, _atlas(budget))
        pieces = volume_material_edit_pieces(edit, budget=budget)
        _, _, targets = edit.exact_cells("target", budget=budget)
        routes = {
            tuple(route): position for position, route in enumerate(np.asarray(pieces.routes).tolist())
        }
        assert routes and all(chart >= 0 for _, _, chart in routes)
        for target, target_tet in enumerate(targets):
            for chart, (image, cone_map) in enumerate(zip(edit.target_charts, pieces.charts, strict=True)):
                # The single source cell is the cavity: s ∩ C_k ∩ A_k^-1(t) = A_k^-1(I_k ∩ t).
                oracle = tetrahedron_overlap(image, target_tet, budget=budget)
                assert isinstance(oracle, ExactTetrahedronOverlap)
                position = routes.get((target, 0, chart))
                if position is None:
                    assert oracle.volume == 0
                    continue
                volume, first, _ = _piece_integrals(pieces, position)
                assert volume * cone_map.jacobian == oracle.volume
                pulled = _solve(
                    cone_map.linear,
                    tuple(
                        oracle.first_moment[a] - cone_map.offset[a] * oracle.volume
                        for a in range(3)
                    ),
                )
                assert first == tuple(value / cone_map.jacobian for value in pulled)


def test_split_source_cone_pieces_match_independent_exact_overlaps() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import tetrahedron_overlap

    # Cavity 17 split at an off-anchor interior vertex (barycentric 1/8, 1/8, 1/2, 1/4),
    # local index 4 in the snapshot and global vertex 6 in the edit: cones cross sources.
    points = np.concatenate((_POINTS, ((0.5, 0.25, -0.125),)))
    local = ((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4))
    rows = np.asarray(tuple(tuple(6 if vertex == 4 else vertex for vertex in row) for row in local))
    removed = (40, 41, 42, 43)
    with _budget() as budget:
        atlas = _atlas(budget, points=points[[0, 1, 2, 3, 6]], cells=local, cell_ids=removed)
        created = np.asarray(_INSERTED)
        edit = _curved_edit(
            np.asarray(removed), rows, points[rows], np.asarray((23, 24, 25)), created, points[created], atlas,
        )
        pieces = volume_material_edit_pieces(edit, budget=budget)
        _, _, sources = edit.exact_cells("source", budget=budget)
        _, _, targets = edit.exact_cells("target", budget=budget)
        zero = (F(0), (F(0), F(0), F(0)))
        by_cone: dict[tuple[int, int], tuple[F, tuple[F, ...]]] = {}
        by_image: dict[tuple[int, int], tuple[F, tuple[F, ...]]] = {}
        for position, (target, source, chart) in enumerate(np.asarray(pieces.routes).tolist()):
            volume, first, _ = _piece_integrals(pieces, position)
            for totals, key in ((by_cone, (source, chart)), (by_image, (target, chart))):
                total, moment = totals.get(key, zero)
                totals[key] = total + volume, tuple(a + b for a, b in zip(moment, first, strict=True))
        crossed = False
        for chart, (origin, image, cone_map) in enumerate(
            zip(edit.source_charts, edit.target_charts, pieces.charts, strict=True)
        ):
            # The targets partition the image cone: summed over targets, s ∩ C_k is whole.
            overlaps: list[F] = []
            for source, source_tet in enumerate(sources):
                oracle = tetrahedron_overlap(source_tet, origin, budget=budget)
                assert isinstance(oracle, ExactTetrahedronOverlap)
                assert by_cone.get((source, chart), zero) == (oracle.volume, tuple(oracle.first_moment))
                overlaps.append(oracle.volume)
            # Some cone and source each reach beyond the other: neither is published whole.
            crossed |= any(
                0 < volume < min(pieces.source_volumes[source], sum(overlaps, F(0)))
                for source, volume in enumerate(overlaps)
            )
            # The sources partition the cone: summed over sources, A_k^-1(I_k ∩ t) is whole.
            for target, target_tet in enumerate(targets):
                oracle = tetrahedron_overlap(image, target_tet, budget=budget)
                assert isinstance(oracle, ExactTetrahedronOverlap)
                volume, first = by_image.get((target, chart), zero)
                assert volume * cone_map.jacobian == oracle.volume
                pulled = _solve(
                    cone_map.linear,
                    tuple(oracle.first_moment[a] - cone_map.offset[a] * oracle.volume for a in range(3)),
                )
                assert first == tuple(value / cone_map.jacobian for value in pulled)
    assert crossed


def test_edit_cell_and_corner_order_keep_exact_material_regions() -> None:
    orders = (
        ((26, 27, 28, 29), _SPLIT),
        # Reversed cell order and an even corner rotation of every created cell.
        ((29, 28, 27, 26), tuple(tuple(row[i] for i in (0, 2, 3, 1)) for row in _SPLIT[::-1])),
    )
    regions = []
    with _budget() as budget:
        for created, rows in orders:
            pieces = volume_material_edit_pieces(_split(created, rows), budget=budget)
            regions.append({
                (created[target], source, chart): _piece_integrals(pieces, position)
                for position, (target, source, chart) in enumerate(np.asarray(pieces.routes).tolist())
            })
    # Internal triangulations may differ; exact per-label regions may not.
    assert regions[0] == regions[1]


@pytest.mark.parametrize(("element", "semantics"), (
    (phx.discretization.lagrange_element("tetrahedron", 4), ("intensive", "conservative-density")),
    (phx.discretization.discontinuous_element("tetrahedron", 3), ("intensive", "conservative-density")),
    (phx.discretization.form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"), ("material-compatible",)),
    (phx.discretization.form_element("tetrahedron", 1, 1, proxy="circulation"), ("material-compatible",)),
))
def test_cold_restored_chain_carries_identical_fields(
    element: phx.discretization.FiniteElementSpec, semantics: tuple[SourceRealizationFieldSemantics, ...],
) -> None:
    names = tuple(f"field{index}" for index in range(len(semantics)))
    fields = tuple(phx.discretization.FiniteElementFieldSpec(name, element) for name in names)
    carried = []
    with _budget() as budget:
        chain = _chain(budget)
        restored = VolumeMaterialChain.from_stored_fields(chain, budget=budget)
        for value in (chain, restored):
            source, target = _spaces(value, fields, budget)
            transfers = prepare_volume_material_fe_transfers(
                value, source, target, semantics=dict(zip(names, semantics, strict=True)), budget=budget,
            )
            assert all(transfer.evidence.passed for transfer in transfers)
            carried.append([
                np.asarray(transfer.transfer.apply(
                    np.linspace(0.5, 2.0, source.dof_maps[index].global_dof_count)
                ))
                for index, transfer in enumerate(transfers)
            ])
    for actual, expected in zip(carried[1], carried[0], strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_exact_determinant_cache_preserves_sub_float_changes_and_orientation() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import star_determinant

    zero = (F(0), F(0), F(0))
    x = (F(1), F(0), F(0))
    y = (F(0), F(1), F(0))
    z = (F(0), F(0), F(1))
    height = F(1) + F(1, 2**80)
    assert float(height) == 1.0
    with _budget() as budget:
        assert star_determinant((zero, x, y, z), budget=budget) == 1
        assert star_determinant(
            (zero, x, y, (F(0), F(0), height)), budget=budget
        ) == height
        assert star_determinant((zero, y, x, z), budget=budget) == -1
        retained = budget.retained_basis_bytes
        copied_zero = tuple(F(value.numerator, value.denominator) for value in zero)
        assert star_determinant(
            (copied_zero, x, y, z),
            budget=budget,
        ) == 1
        assert budget.retained_basis_bytes == retained


def test_exact_determinant_history_fits_a_bounded_scalar_cache() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import star_determinant

    zero = (F(0), F(0), F(0))
    x = (F(1), F(0), F(0))
    y = (F(0), F(1), F(0))
    with _budget(memory=256 * 1024) as budget:
        for height in range(1, 101):
            assert star_determinant(
                (zero, x, y, (F(0), F(0), F(height))), budget=budget
            ) == height, f"tetrahedron height {height}"
        retained = budget.retained_basis_bytes
        assert star_determinant(
            (zero, x, y, (F(0), F(0), F(1))), budget=budget
        ) == 1
        assert budget.retained_basis_bytes == retained

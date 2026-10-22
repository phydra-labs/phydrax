"""Concrete-incidence admission and retained polyhedral topology lifetimes."""

import gc
import math
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
import phydrax.meshing._polyhedral_generation as owner
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from phydrax.discretization import CellMesh, PeriodicCell, PeriodicMeshTopology
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.polyhedral_meshes import (
    periodic_fv_public_inputs,
    POLYHEDRAL_CUBE_LOOPS,
)


def _scope() -> NativeExecutionScope:
    return NativeExecutionScope(
        NativeExecutionLimits(1 << 30, 1 << 26, 1 << 24, 1 << 30, math.inf)
    )


def _cube() -> tuple[np.ndarray, list[list[np.ndarray]]]:
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    return points, [[np.asarray(loop, dtype=np.int64) for loop in POLYHEDRAL_CUBE_LOOPS]]


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("short", (False, True))
def test_concrete_incidence_exact_headroom_before_builder(
    periodic: bool, short: bool,
) -> None:
    """R/R-1 is real pool headroom, including both native token headers.

    The canonical builders are real: no patched allocator, admission method,
    topology constructor or output. Neither builder is entered on R-1.
    """
    points, cells = _cube()
    lifted = CellMesh.from_polyhedra(points, cells) if periodic else None
    required, retained = (
        owner._periodic_topology_storage_bound(8, 1, 6, 24, 12 * 4 + 6 * 16 + 64, 3)
        if periodic else owner._polyhedral_mesh_storage_bound(8, 1, 6, 24)
    )
    records = 512 + 64 + (16 * (8 + 24 + 6 + 1) if periodic else 0)
    retained_total = retained + 1024 + 512 * records
    reached_builder = False
    scope = _scope()
    expected = pytest.raises(MeshcoreError) if short else nullcontext()
    with expected as refusal:
        with scope, owner._PolyhedralTopologyStorage(scope) as storage:
            storage.workspace.reserve_empty()
            with NativeHostStorageWorkspace(scope) as ballast:
                ballast.reserve_empty()
                available = scope.remaining().remaining_managed_bytes
                ballast.set_bound(available - required + int(short))
                assert scope.remaining().remaining_managed_bytes == required - int(short)
                storage.workspace.set_bound(required)
                reached_builder = True
                if periodic:
                    assert lifted is not None
                    with coordinate_reference_budget() as budget:
                        result = PeriodicMeshTopology(
                            lifted, PeriodicCell(np.eye(3)), np.zeros(8, dtype=np.int32),
                            points.astype(np.int32), budget=budget,
                        )
                else:
                    result = CellMesh.from_polyhedra(points, cells)
                storage.retain(retained, result, record_count_upper=records)
                assert storage.workspace.bound == retained_total
                assert scope.remaining().remaining_managed_bytes >= required - retained_total
    assert reached_builder is not short
    if short:
        assert refusal is not None
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        assert scope.evidence is not None


def test_retained_mesh_storage_outlives_mesh_when_provider_keeps_leaf() -> None:
    points, cells = _cube()
    required, retained = owner._polyhedral_mesh_storage_bound(8, 1, 6, 24)
    records = 512 + 64
    retained_total = retained + 1024 + 512 * records
    with _scope() as scope:
        with owner._PolyhedralTopologyStorage(scope) as storage:
            storage.workspace.set_bound(required)
            mesh = CellMesh.from_polyhedra(points, cells)
            leaf = owner.require_polyhedral_connectivity(mesh).face_vertex_values
            workspace = storage.workspace
            storage.retain(retained, mesh, record_count_upper=records)
        del mesh, storage
        gc.collect()
        assert workspace.bound == retained_total
        del leaf
        gc.collect()
        assert workspace.bound == 0


@pytest.mark.parametrize("shortfall", (0, 1))
def test_retention_work_admitted_before_traversal_and_binding(
    shortfall: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = np.arange(4, dtype=np.int64)
    records = 4096  # Safety capacity is deliberately much larger than actual work.
    required_work = 2  # Argument tuple and the single fresh array owner.
    inspected: list[int] = []
    original = owner.retained_host_storage_fanout_upper

    def fanout(value: object) -> int:
        inspected.append(id(value))
        return original(value)

    monkeypatch.setattr(owner, "retained_host_storage_fanout_upper", fanout)
    expected = pytest.raises(MeshcoreError) if shortfall else nullcontext()
    with expected as refused, NativeExecutionScope(
        NativeExecutionLimits(required_work - shortfall, 1024, 1024, 1 << 24, math.inf)
    ) as scope:
        with owner._PolyhedralTopologyStorage(scope) as storage:
            storage.retain(value.nbytes, value, record_count_upper=records)
            assert scope.remaining().remaining_work_units == 0
        assert storage.workspace.bound > 0
        del value
        gc.collect()
        assert storage.workspace.bound == 0
    assert len(inspected) == required_work - shortfall
    if shortfall:
        assert refused is not None
        assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        assert storage.workspace.bound == 0


@dataclass
class _RetentionOwner:
    children: object


def test_retention_deduplicates_actual_visits_and_keeps_only_weak_owners() -> None:
    value = np.arange(4, dtype=np.int64)
    root = _RetentionOwner([value] * 1000)
    identity = value.copy()
    with _scope() as scope:
        before = scope.remaining().remaining_work_units
        with owner._PolyhedralTopologyStorage(scope) as storage:
            storage.retain(value.nbytes, root, root, value, record_count_upper=16)
        # Argument tuple, root, list and array: aliases do not earn new visits.
        assert before - scope.remaining().remaining_work_units == 4
        assert storage._remaining == 2
        np.testing.assert_array_equal(value, identity)
        del root
        gc.collect()
        assert storage._remaining == 1 and storage.workspace.bound > 0
        del value
        gc.collect()
        assert storage._remaining == 0 and storage.workspace.bound == 0


def test_retention_rejects_oversized_graph_before_inspecting_unadmitted_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _RetentionOwner([np.arange(1) for _ in range(32)])
    inspected: list[int] = []
    original = owner.retained_host_storage_fanout_upper

    def fanout(value: object) -> int:
        inspected.append(id(value))
        return original(value)

    monkeypatch.setattr(owner, "retained_host_storage_fanout_upper", fanout)
    with _scope() as scope, owner._PolyhedralTopologyStorage(scope) as storage:
        before = scope.remaining().remaining_work_units
        with pytest.raises(RuntimeError, match="retention_records"):
            storage.retain(256, root, record_count_upper=4)
        # Tuple/root/list plus four distinct arrays: the fourth array cannot
        # acquire a fifth weak-owner record and is refused before expansion.
        assert before - scope.remaining().remaining_work_units == 7
        assert len(inspected) == 6
        assert storage._remaining == 0 and storage.workspace.bound == 0


@pytest.mark.parametrize("failure_at", ("attachment", "commit"))
def test_retention_rolls_back_every_partial_finalizer(
    monkeypatch: pytest.MonkeyPatch, failure_at: str,
) -> None:
    value = np.arange(4, dtype=np.int64)
    root = _RetentionOwner(value)
    attached = []
    finalize = owner.weakref.finalize
    set_bound = NativeHostStorageWorkspace.set_bound

    def attach(value: object, callback: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(getattr(callback, "__self__", None), owner._PolyhedralTopologyStorage):
            return finalize(value, callback, *args, **kwargs)
        if failure_at == "attachment" and attached:
            raise RuntimeError("refused retention attachment")
        result = finalize(value, callback)
        attached.append(result)
        return result

    with _scope() as scope, owner._PolyhedralTopologyStorage(scope) as storage:
        def resize(workspace: NativeHostStorageWorkspace, amount: int) -> None:
            if failure_at == "commit" and workspace is storage.workspace and storage._remaining:
                raise RuntimeError("refused retention commit")
            set_bound(workspace, amount)

        monkeypatch.setattr(owner.weakref, "finalize", attach)
        monkeypatch.setattr(NativeHostStorageWorkspace, "set_bound", resize)
        with pytest.raises(RuntimeError, match="refused retention"):
            storage.retain(value.nbytes, root, record_count_upper=16)
        assert attached and not any(finalizer.alive for finalizer in attached)
        assert storage._remaining == 0 and storage.workspace.bound == 0
    del root, value
    gc.collect()
    assert storage._remaining == 0


def test_exact_seam_subdivision_and_reversed_expansion() -> None:
    points = tuple(
        tuple(map(Fraction, point))
        for point in ((0, 0, 0), (2, 0, 0), (0, 2, 0), (1, 0, 0))
    )
    loop = (0, 1, 2)
    with coordinate_reference_budget() as budget:
        edges = owner._seam_segment_subdivision(points, (loop,), budget=budget)
    assert edges[(0, 1)] == (0, 3, 1)
    assert owner._expanded_seam_loop(loop, edges) == (0, 3, 1, 2)
    assert owner._expanded_seam_loop(loop[::-1], edges) == (2, 1, 3, 0)


@pytest.mark.parametrize("periodic", (False, True))
def test_actual_provider_publication_is_deterministic(
    periodic: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    active: list[owner._PolyhedralTopologyStorage] = []
    builds = {"mesh": 0, "periodic": 0, "bank": 0, "subdivision": 0, "orbits": 0}
    original_enter = owner._PolyhedralTopologyStorage.__enter__
    original_exit = owner._PolyhedralTopologyStorage.__exit__
    original_mesh = CellMesh.from_polyhedra
    original_periodic = PeriodicMeshTopology.__init__
    original_bank = owner._periodic_seam_bank
    original_subdivision = owner._seam_segment_subdivision
    original_orbits = owner.seam_vertex_orbits

    def enter(storage: owner._PolyhedralTopologyStorage) -> owner._PolyhedralTopologyStorage:
        result = original_enter(storage)
        active.append(storage)
        return result

    def exit_(storage: owner._PolyhedralTopologyStorage, *exc: object) -> None:
        assert active.pop() is storage
        original_exit(storage, *exc)

    def mesh_builder(cls: type[CellMesh], coordinates: Any, cells: Any, **kwargs: Any) -> CellMesh:
        loops = sum(len(cell) for cell in cells)
        corners = sum(len(loop) for cell in cells for loop in cell)
        required, _ = owner._polyhedral_mesh_storage_bound(
            len(coordinates), len(cells), loops, corners
        )
        assert active and active[-1].workspace.bound >= required
        builds["mesh"] += 1
        return original_mesh(coordinates, cells, **kwargs)

    def periodic_builder(self: PeriodicMeshTopology, lifted: CellMesh, *args: Any, **kwargs: Any) -> None:
        connectivity = owner.require_polyhedral_connectivity(lifted)
        offsets = np.asarray(connectivity.face_vertex_offsets)
        squares = 4 * connectivity.edge_count + sum(
            int(b - a) ** 2 for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ) + sum(block.cell_count * block.vertices.shape[1] ** 2 for block in lifted.blocks)
        required, _ = owner._periodic_topology_storage_bound(
            connectivity.vertex_count, connectivity.cell_count,
            len(connectivity.cell_face_values), len(connectivity.face_vertex_values),
            squares, args[0].rank,
        )
        assert active and active[-1].workspace.bound >= required
        builds["periodic"] += 1
        original_periodic(self, lifted, *args, **kwargs)
    def seam_bank(*args: Any, **kwargs: Any) -> Any:
        assert active and active[-1].workspace.bound > 0
        bank = original_bank(*args, **kwargs)
        assert len(bank.points) == len(bank.parents) == len(bank.carriers)
        assert len(bank.vertex_pairs) > 0
        assert all(len(loop) >= 3 for loops in bank.face_parts.values() for loop in loops)
        builds["bank"] += 1
        return bank

    def subdivision(*args: Any, **kwargs: Any) -> Any:
        assert active and active[-1].workspace.bound > 0
        edges = original_subdivision(*args, **kwargs)
        assert all(row[0] == edge[0] and row[-1] == edge[1] for edge, row in edges.items())
        builds["subdivision"] += 1
        return edges

    def orbits(*args: Any, **kwargs: Any) -> Any:
        representatives, shifts = original_orbits(*args, **kwargs)
        points, pairs, identifiers, preparation = args
        assert representatives.shape == (len(points),)
        assert shifts.shape == (len(points), len(preparation.orders))
        for source_vertex, target_vertex, _ in pairs:
            assert representatives[source_vertex] == representatives[target_vertex]
        assert np.all(identifiers[representatives] <= identifiers)
        builds["orbits"] += 1
        return representatives, shifts


    monkeypatch.setattr(owner._PolyhedralTopologyStorage, "__enter__", enter)
    monkeypatch.setattr(owner._PolyhedralTopologyStorage, "__exit__", exit_)
    monkeypatch.setattr(CellMesh, "from_polyhedra", classmethod(mesh_builder))
    monkeypatch.setattr(PeriodicMeshTopology, "__init__", periodic_builder)
    monkeypatch.setattr(owner, "_periodic_seam_bank", seam_bank)
    monkeypatch.setattr(owner, "_seam_segment_subdivision", subdivision)
    monkeypatch.setattr(owner, "seam_vertex_orbits", orbits)
    source, specification, options, _ = periodic_fv_public_inputs("translation")
    if not periodic:
        specification = phx.meshing.VolumeMeshingSpec(
            specification.target, specification.boundary_scope,
            phx.meshing.VolumeFillStrategy.POLYHEDRAL,
            size_controls=specification.size_controls,
        )
    provider = phx.meshing.NativeMeshingProvider(options)
    results = [
        provider.plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        ).execute()
        for _ in range(2)
    ]
    first, second = results
    assert builds["mesh"] >= 2
    for name in ("periodic", "bank", "subdivision", "orbits"):
        assert builds[name] >= 2 if periodic else builds[name] == 0
    assert first.result_id == second.result_id
    assert first.mesh.mesh_id == second.mesh.mesh_id
    np.testing.assert_array_equal(first.mesh.coordinates, second.mesh.coordinates)
    np.testing.assert_array_equal(
        first.mesh.connectivity.face_vertex_values,
        second.mesh.connectivity.face_vertex_values,
    )
    assert (first.mesh.periodic_topology is not None) is periodic
    if periodic:
        assert first.mesh.periodic_topology.periodic_topology_id == (
            second.mesh.periodic_topology.periodic_topology_id
        )

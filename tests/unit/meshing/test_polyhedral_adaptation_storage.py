"""Concrete adaptation admission, actual builders, and extracted-alias lifetimes."""

import gc
import math
from contextlib import nullcontext
from fractions import Fraction
from typing import Any, Never

import numpy as np
import pytest

import phydrax.meshing._polyhedral_adaptation as owner
import phydrax.meshing._polyhedral_adaptation_shared as shared
import phydrax.meshing._polyhedral_generation as generation
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
)
from phydrax.discretization import (
    CellMesh,
    PeriodicCell,
    PeriodicMeshTopology,
    PolyhedralConnectivity,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.polyhedral_meshes import boxes


def _scope() -> NativeExecutionScope:
    return NativeExecutionScope(
        NativeExecutionLimits(1 << 30, 1 << 26, 1 << 24, 1 << 30, math.inf)
    )


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("short", (False, True))
def test_adaptation_bound_actual_builder_exact_pool_headroom(
    periodic: bool, short: bool,
) -> None:
    """R/R-1 uses real allocator tokens, not a substituted admission function."""
    source = boxes((0.0, 1.0))
    points = np.asarray(source.coordinates)
    loops = shared.polyhedral_face_loops(source)
    reached_builder = False
    scope = _scope()
    expected = pytest.raises(MeshcoreError) if short else nullcontext()
    with expected as refusal:
        with scope:
            incidence = owner._polyhedral_incidence(source, scope=scope)
            assert incidence.corners == 24
            assert incidence.squared_arities == 4 * 12 + 6 * 16 + 64
            bound = owner._polyhedral_adaptation_storage_bound(
                incidence._replace(rank=3 if periodic else 0)
            )
            with generation._PolyhedralTopologyStorage(scope) as storage:
                storage.workspace.reserve_empty()
                with NativeHostStorageWorkspace(scope) as ballast:
                    ballast.reserve_empty()
                    available = scope.remaining().remaining_managed_bytes
                    ballast.set_bound(available - bound.construction + int(short))
                    assert scope.remaining().remaining_managed_bytes == (
                        bound.construction - int(short)
                    )
                    storage.workspace.set_bound(bound.construction)
                    reached_builder = True
                    target = CellMesh.from_polyhedra(points, loops)
                    if periodic:
                        with coordinate_reference_budget() as budget:
                            descriptor = PeriodicMeshTopology(
                                target, PeriodicCell(np.eye(3)),
                                np.zeros(len(points), dtype=np.int32),
                                points.astype(np.int32), budget=budget,
                            )
                        storage.retain(
                            bound.retained, descriptor,
                            record_count_upper=bound.records,
                        )
                    else:
                        storage.retain(
                            bound.retained, target,
                            record_count_upper=bound.records,
                        )
    assert reached_builder is not short
    if short:
        assert refusal is not None
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize("operation", ("plane", "agglomerate"))
def test_actual_adaptation_builders_are_deterministic_and_preserve_source(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = boxes((0.0, 0.5, 1.0))
    assert isinstance(source.connectivity, PolyhedralConnectivity)
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    corners = np.asarray(source.connectivity.face_vertex_values).copy()
    active = []
    builds = 0
    original_enter = generation._PolyhedralTopologyStorage.__enter__
    original_exit = generation._PolyhedralTopologyStorage.__exit__
    original_builder = CellMesh.from_polyhedra

    def enter(storage: generation._PolyhedralTopologyStorage) -> generation._PolyhedralTopologyStorage:
        result = original_enter(storage)
        active.append(storage)
        return result

    def exit_(storage: generation._PolyhedralTopologyStorage, *exc: object) -> None:
        assert active.pop() is storage
        original_exit(storage, *exc)

    def builder(cls: type[CellMesh], points: Any, cells: Any, **kwargs: Any) -> CellMesh:
        nonlocal builds
        required, _ = generation._polyhedral_mesh_storage_bound(
            len(points), len(cells), sum(len(cell) for cell in cells),
            sum(len(loop) for cell in cells for loop in cell),
        )
        assert active and active[-1].workspace.bound >= required
        builds += 1
        return original_builder(points, cells, **kwargs)

    monkeypatch.setattr(generation._PolyhedralTopologyStorage, "__enter__", enter)
    monkeypatch.setattr(generation._PolyhedralTopologyStorage, "__exit__", exit_)
    monkeypatch.setattr(CellMesh, "from_polyhedra", classmethod(builder))
    results = []
    for _ in range(2):
        with _scope() as scope:
            if operation == "plane":
                support = {
                    int(identifier): {int(identifier)}
                    for identifier in np.asarray(source.vertex_global_ids)
                }
                with coordinate_reference_budget() as budget:
                    result = owner._split(
                        source, np.asarray([1.0, 0.0, 0.0, 0.25]), set(),
                        0.0, 8, 64, support, scope=scope, budget=budget,
                    ).mesh
            else:
                result = owner._agglomerate(
                    source, ((20, 21),), np.zeros(2, dtype=np.int64), set(),
                    scope=scope,
                )
            results.append(result)
    assert builds == 2
    assert results[0].mesh_id == results[1].mesh_id
    np.testing.assert_array_equal(results[0].coordinates, results[1].coordinates)
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)
    np.testing.assert_array_equal(source.connectivity.face_vertex_values, corners)


@pytest.mark.parametrize("operation", ("plane", "agglomerate"))
def test_adapted_topology_storage_follows_extracted_leaf(
    operation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspaces = []
    original_enter = generation._PolyhedralTopologyStorage.__enter__

    def enter(storage: generation._PolyhedralTopologyStorage) -> generation._PolyhedralTopologyStorage:
        result = original_enter(storage)
        workspaces.append(storage.workspace)
        return result

    monkeypatch.setattr(generation._PolyhedralTopologyStorage, "__enter__", enter)
    source = boxes((0.0, 0.5, 1.0))
    with _scope() as scope:
        if operation == "plane":
            support = {
                int(identifier): {int(identifier)}
                for identifier in np.asarray(source.vertex_global_ids)
            }
            with coordinate_reference_budget() as budget:
                target = owner._split(
                    source, np.asarray([1.0, 0.0, 0.0, 0.25]), set(),
                    0.0, 8, 64, support, scope=scope, budget=budget,
                ).mesh
        else:
            target = owner._agglomerate(
                source, ((20, 21),), np.zeros(2, dtype=np.int64), set(), scope=scope,
            )
        assert isinstance(target.connectivity, PolyhedralConnectivity)
        leaf = target.connectivity.face_vertex_values
        assert len(workspaces) == 1
        workspace = workspaces[0]
        retained = workspace.bound
        assert retained > 0
        del target
        gc.collect()
        assert workspace.bound == retained
        del leaf
        gc.collect()
        assert workspace.bound == 0


@pytest.mark.parametrize("disjoint", (False, True))
def test_exact_stratum_storage_uses_existing_scalar_rates_and_releases_temporaries(
    disjoint: bool,
) -> None:
    unit = Fraction(1, 1 << 257)
    source = np.asarray(
        [(Fraction(0), Fraction(0), Fraction(0)),
         (unit, Fraction(0), Fraction(0))], dtype=object,
    )
    lower, upper = ((2 * unit, 3 * unit) if disjoint else (unit / 4, unit / 2))
    target = np.asarray(
        [(lower, Fraction(0), Fraction(0)),
         (upper, Fraction(0), Fraction(0))], dtype=object,
    )
    with coordinate_reference_budget() as budget:
        work = budget.work_units
        temporary = budget.temporary_bytes_upper
        remaining = [1024]
        assert owner._stratum_overlap(
            source, target, 1, remaining, budget=budget
        ) is not disjoint
        # Existing comparison/lift/coefficient tariffs, charged only for the
        # reached broad phase or the complete exact curve predicate.
        assert budget.work_units - work == (42 if disjoint else 106)
        assert remaining == [1024]
        assert budget.temporary_bytes_upper == temporary


@pytest.mark.parametrize("circuit, required", (("halfspace", 7), ("section", 3)))
def test_plane_scalar_circuits_charge_actual_operations_before_arithmetic(
    circuit: str, required: int,
) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.meshing import _polyhedral_plane_restriction as restriction

    a = (Fraction(0), Fraction(0), Fraction(0))
    b = (Fraction(2), Fraction(0), Fraction(0))
    point = (Fraction(1), Fraction(0), Fraction(0))
    plane = (Fraction(1), Fraction(0), Fraction(0), Fraction(-2))
    with coordinate_reference_budget() as budget:
        before = budget.work_units
        if circuit == "halfspace":
            assert restriction._outside_plane(plane, 1, point, budget=budget) is False
        else:
            assert restriction._section_parameter(point, a, b, 0, budget=budget) == Fraction(1, 2)
        assert budget.work_units - before == required

    class ArithmeticMustNotRun(Fraction):
        def __mul__(self, b: Any) -> Never:
            raise AssertionError("arithmetic preceded work admission")

        def __sub__(self, b: Any) -> Never:
            raise AssertionError("arithmetic preceded work admission")

    trap = ArithmeticMustNotRun(1)
    with coordinate_reference_budget() as budget:
        with budget.bound_stage(required - 1, budget.maximum_memory_bytes):
            with pytest.raises(CoordinateEnclosureResourceError):
                if circuit == "halfspace":
                    restriction._outside_plane((trap, *plane[1:]), 1, point, budget=budget)
                else:
                    restriction._section_parameter((trap, *point[1:]), a, b, 0, budget=budget)
        assert budget.work_units == 0


@pytest.mark.parametrize("shortfall", (0, 1))
def test_exact_curve_stratum_actual_work_r_and_r_minus_one(shortfall: int) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    source = np.asarray(
        [(Fraction(0), Fraction(0), Fraction(0)),
         (Fraction(1), Fraction(0), Fraction(0))], dtype=object,
    )
    target = np.asarray(
        [(Fraction(1, 4), Fraction(0), Fraction(0)),
         (Fraction(1, 2), Fraction(0), Fraction(0))], dtype=object,
    )
    with coordinate_reference_budget() as budget:
        temporary = budget.temporary_bytes_upper
        with budget.bound_stage(106 - shortfall, budget.maximum_memory_bytes):
            if shortfall:
                with pytest.raises(CoordinateEnclosureResourceError) as failure:
                    owner._stratum_overlap(source, target, 1, [1024], budget=budget)
                assert failure.value.resource == "coefficient_work"
                assert failure.value.limit == 105
                assert failure.value.requested == 106
                assert failure.value.completed == 96
            else:
                assert owner._stratum_overlap(
                    source, target, 1, [1024], budget=budget
                )
                assert budget.work_units == 106
        assert budget.temporary_bytes_upper == temporary


@pytest.mark.parametrize("operation", ("plane", "agglomerate", "regenerate"))
@pytest.mark.parametrize("shortfall", (0, 1))
@pytest.mark.parametrize("nested", (False, True))
def test_current_candidate_cavity_admission_precedes_construction(
    operation: str, shortfall: int, nested: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R/R-1 uses the actual replacement workset, including split descendants."""
    from dataclasses import replace

    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing._plc_source import PiecewiseLinearComplex

    source = boxes(
        tuple(np.linspace(0.0, 1.0, 33)) if operation == "regenerate"
        else (0.0, 0.5, 1.0)
    )
    source_id = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    reached = []
    if operation == "plane":
        required = 3
        request = owner.PolyhedralMeshAdaptation(
            owner.PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=((1, 0, 0, 0.25), (0, 1, 0, 0.5)),
        )
        name, module = "_clip_plane_split", owner
    elif operation == "agglomerate":
        required = 2
        request = owner.PolyhedralMeshAdaptation(
            owner.PolyhedralAdaptationOperation.AGGLOMERATE,
            agglomerations=((20, 21),),
        )
        name, module = "_build_agglomeration", owner
    else:
        required = source.connectivity.cell_count
        topology = reference_cell_topology("hexahedron")
        complex_ = PiecewiseLinearComplex(
            np.asarray(topology.vertices, dtype=np.float64), topology.entities[2],
            np.zeros(6, dtype=np.int64), np.asarray([[-1, 0]], dtype=np.int64),
            ("material",),
        )
        request = owner.PolyhedralMeshAdaptation(
            owner.PolyhedralAdaptationOperation.REGENERATE,
            complex_=complex_, sites=((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)),
        )
        name, module = "generate_polyhedral_volume", generation
    original = getattr(module, name)

    def build(*args: Any, **kwargs: Any) -> Any:
        reached.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, build)
    limits = replace(
        _scope().limits, maximum_cavity_cells=required - shortfall
    )
    scope = NativeExecutionScope(limits)
    expected = pytest.raises(shared.PolyhedralAdaptationError) if shortfall else nullcontext()
    with expected as refusal:
        with NativeExecutionScope(limits) if nested else nullcontext():
            with scope:
                scope.observe_managed_admission()
                result = owner.adapt_polyhedral_mesh(source, request, scope=scope)
    if shortfall:
        assert refusal is not None
        assert len(reached) == (1 if operation == "plane" else 0)
        native = refusal.value.__cause__
        assert isinstance(native, MeshcoreError)
        assert native.status is MeshcoreStatus.CAPACITY_EXCEEDED
        assert native.batch_requested_upper is not None
        assert native.batch_requested_upper[2] == required
        assert native.batch_refusal_evidence is not None
        assert native.batch_refusal_evidence[2] == required - 1
        achieved = dict(refusal.value.evidence.achieved)
        assert achieved["native_status"] == float(MeshcoreStatus.CAPACITY_EXCEEDED)
        assert achieved["native_requested:cavity_cells"] == required
        assert achieved["native_remaining:cavity_cells"] == required - 1
    else:
        assert len(reached) == (2 if operation == "plane" else 1)
        assert result.common_refinement.succeeded
        evidence = scope.completed_evidence()
        # A nested receipt cannot independently observe the enclosing pool.
        # Its explicit partial observation must not become a fabricated peak.
        assert (evidence.managed_admission_peak_bytes_upper is None) is nested
    assert source.mesh_id == source_id
    np.testing.assert_array_equal(source.coordinates, coordinates)


@pytest.mark.parametrize("shortfall", (0, 1))
def test_exact_plane_descendant_cavity_precedes_intersections(
    shortfall: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.meshing import _polyhedral_plane_restriction as restriction
    from phydrax.meshing._plc_source import PiecewiseLinearComplex

    topology = reference_cell_topology("hexahedron")
    domain = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64), topology.entities[2],
        np.zeros(6, dtype=np.int64), np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    with _scope() as scope:
        source = generation.generate_polyhedral_volume(
            domain, sites=((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), scope=scope,
        )
    intersected = set()
    original = restriction._PlaneCut.intersect

    def intersect(cut: restriction._PlaneCut, first: int, second: int) -> int:
        intersected.add(cut.plane[1])
        return original(cut, first, second)

    monkeypatch.setattr(restriction._PlaneCut, "intersect", intersect)
    limits = replace(_scope().limits, maximum_cavity_cells=3 - shortfall)
    expected = pytest.raises(shared.PolyhedralAdaptationError) if shortfall else nullcontext()
    with expected as refusal, NativeExecutionScope(limits) as scope:
        outcome = owner.adapt_polyhedral_mesh(
            source.mesh,
            owner.PolyhedralMeshAdaptation(
                owner.PolyhedralAdaptationOperation.PLANE_SPLIT,
                planes=((1, 0, 0, 0.25), (0, 1, 0, 0.5)),
            ),
            source_geometry=source.geometry, scope=scope,
        )
    if shortfall:
        assert refusal is not None
        assert intersected == {Fraction(0)}
        native = refusal.value.__cause__
        assert isinstance(native, MeshcoreError)
        assert native.batch_requested_upper == (0, 0, 3, 0)
        assert dict(refusal.value.evidence.achieved)["native_remaining:cavity_cells"] == 2
    else:
        assert intersected == {Fraction(0), Fraction(1)}
        assert outcome.common_refinement.succeeded


@pytest.mark.parametrize("axis", (
    "vertices", "edges", "faces", "cells", "connectivity_entries", "data_bytes",
))
@pytest.mark.parametrize("shortfall", (0, 1))
@pytest.mark.parametrize("split", (False, True))
def test_candidate_output_capacity_precedes_topology_builder(
    axis: str, shortfall: int, split: bool,
) -> None:
    from phydrax.meshing import MeshingFailure, MeshingFailureCategory, MeshingLimits

    source = boxes((0.0, 0.5, 1.0))
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    v, c, f, k = (32, 4, 28, 288) if split else (12, 2, 12, 48)
    slots = 64 * (k + f) + 16 * (v + c + 1)
    required = {
        "vertices": v, "edges": k, "faces": f, "cells": c,
        "connectivity_entries": slots, "data_bytes": 8 * slots,
    }[axis]
    limits = MeshingLimits(**{f"maximum_{axis}": required - shortfall})
    entered = False
    expected = pytest.raises(MeshingFailure) if shortfall else nullcontext()
    with expected as refused, _scope() as scope:
        with owner._polyhedral_topology_phase(
            scope, source, split=split, periodic_topology=False, limits=limits
        ):
            entered = True
            target = CellMesh.from_polyhedra(
                source.coordinates, shared.polyhedral_face_loops(source)
            )
            np.testing.assert_array_equal(target.coordinates, source.coordinates)
    assert entered is (not shortfall)
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)
    if shortfall:
        assert refused is not None
        assert refused.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        assert dict(refused.value.evidence.requested)[f"maximum_{axis}"] == required - 1
        assert dict(refused.value.evidence.achieved)[f"required_{axis}"] == required
        assert dict(refused.value.evidence.achieved)[f"remaining_{axis}"] == required - 1

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry transitions of curved coordinate maps through native adaptation.

Nested-transition oracles use analytic shears ``x -> x + a q(y, z)`` of straight
simplex meshes: child maps are algebraic restrictions and measures stay unchanged.
Source-realization oracles instead use ``(x, y) -> (x, y(1 + x/4))``:
the physical measure changes from 1 to 9/8, density inventory remains 2, and
material history/flux follow their reference pullback and Piola laws.
"""

from collections.abc import Iterator
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh, FiniteElementSpec
from phydrax.discretization._cell_geometry_transfer import (
    transition_displaced_cell_geometry,
)
from phydrax.discretization._cell_geometry_transition_budget import (
    charged_geometry_transition,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransitionError,
    CellGeometryTransitionPolicy,
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._bisection import BisectionHierarchy
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    EntityRelations,
    TopologyEditBlock,
)
from tests._support.cell_meshes import kuhn_cube
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.geometry_transfer_cases import (
    apply_shear,
    curved_shear_source,
    shear_term,
    transfer_triangles,
)


_CONTRACT = phx.SpatialCoordinateContract.si()

# Unit-coordinate accuracy requirements allocate 8, 64, 512, and 8192 binary64
# ulps to affine motion, source interpolation, repeated restriction, and measure
# reduction. The retained decimal thresholds are stricter than those allocations;
# the analytic shear and unit-determinant measure provide independent references.
_MOTION_ROUNDING = 1e-15
_SOURCE_ROUNDING = 1e-14
_RESTRICTION_ROUNDING = 1e-13
_MEASURE_ROUNDING = 1e-12
# Piola reconstruction and solve certificates have separate unit-scale accuracy
# allocations of 2**14 and 2**20 eps; retain their stricter decimal requirements.
_PIOLA_ROUNDING = 2e-12
_SOLVE_ACCURACY = 1e-10


def _adapt(source: Any, refine: Any = (), coarsen: Any = (), **options: Any) -> Any:
    hierarchy = options.pop("hierarchy", None)
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION, **options)
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _map(result: Any, row: int, reference: np.ndarray) -> np.ndarray:
    geometry = result.geometry
    for element, routes in zip(geometry.elements, geometry.geometry_dofs, strict=True):
        if row < routes.shape[0]:
            route = np.asarray(routes)[row]
            values, _ = element.tabulate(reference)
            return np.asarray(values) @ np.asarray(geometry.coordinates)[route]
        row -= routes.shape[0]
    raise ValueError(
        f"fixture.geometry.row must index a cell; got unmatched remainder {row}"
    )


def _max_shear_deviation(result: Any, term: Any, samples: int) -> float:
    """Largest distance between each cell's map and the algebraic sheared simplex."""

    mesh = result.mesh
    dimension = mesh.topological_dimension
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    rng = np.random.default_rng(7)
    worst = 0.0
    for row in range(corners.shape[0]):
        barycentric = rng.dirichlet(np.ones(dimension + 1), size=samples)
        # The shear keeps y and z, so its inverse subtracts the same term.
        straight = apply_shear(corners[row], term, -1.0)
        exact = apply_shear(barycentric @ straight, term, 1.0)
        mapped = _map(result, row, barycentric[:, 1:])
        worst = max(worst, float(np.max(np.abs(mapped - exact))))
    return worst


@pytest.mark.parametrize(
    ("kind", "degree"),
    [("triangle", 2), ("triangle", 3), ("tetrahedron", 2), ("tetrahedron", 3)],
    ids=["p2-triangle", "p3-triangle", "p2-tetrahedron", "p3-tetrahedron"],
)
def test_native_bisection_restricts_curved_maps_exactly(kind: str, degree: int) -> None:
    dimension = 2 if kind == "triangle" else 3
    term = shear_term(degree, dimension)
    source = curved_shear_source(
        transfer_triangles(2)
        if dimension == 2
        else kuhn_cube(1, orientation_swap=(0, 1)),
        degree,
        term,
    )
    assert _max_shear_deviation(source, term, 5) < _SOURCE_ROUNDING
    refined = _adapt(source, (0, 1))

    target = refined.target
    transition = refined.geometry_transition
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert target.mesh.blocks[0].cell_count > source.mesh.blocks[0].cell_count
    assert target.geometry.elements[0].element_id == (
        source.geometry.elements[0].element_id
    )
    assert _max_shear_deviation(target, term, 9) < _RESTRICTION_ROUNDING
    assert transition.evidence.kind == "nested_restriction"
    assert transition.evidence.exact
    # Unit-determinant shear: the mapped measure is the straight unit measure.
    assert transition.evidence.measure_exact
    assert abs(transition.evidence.target_measure - 1.0) < _MEASURE_ROUNDING
    # The certified mesh corners are the map's vertex nodes, not straight midpoints.
    vertex_rows = np.asarray(target.mesh.blocks[0].vertices)
    node_rows = np.asarray(target.geometry.geometry_dofs[0])[
        :, [entity[0] for entity in target.geometry.elements[0].entity_dofs[0]]
    ]
    np.testing.assert_allclose(
        np.asarray(target.mesh.coordinates)[vertex_rows],
        np.asarray(target.geometry.coordinates)[node_rows],
        rtol=0.0,
        atol=0.0,
    )


def test_parent_witnesses_evaluate_the_source_map() -> None:
    term = shear_term(3, 2)
    source = curved_shear_source(transfer_triangles(2), 3, term)
    refined = _adapt(source, (0, 3, 5))
    parents = refined.parent_cells
    corners = refined.parent_reference_vertices
    rng = np.random.default_rng(3)
    for row in range(parents.size):
        barycentric = rng.dirichlet(np.ones(3), size=6)
        np.testing.assert_allclose(
            _map(refined.target, row, barycentric[:, 1:]),
            _map(source, int(parents[row]), barycentric @ corners[row]),
            rtol=0.0,
            atol=_SOURCE_ROUNDING,
        )


def test_restored_parents_coarsen_back_to_the_source_map() -> None:
    term = shear_term(2, 2)
    source = curved_shear_source(transfer_triangles(2), 2, term)
    refined = _adapt(source, (0,))
    new_cells = np.setdiff1d(
        np.asarray(refined.target.mesh.blocks[0].global_ids),
        np.asarray(source.mesh.blocks[0].global_ids),
    )
    restored = _adapt(refined.target, (), new_cells, hierarchy=refined.hierarchy)

    evidence = restored.geometry_transition.evidence
    assert evidence.kind == "coarsening_interpolation"
    assert evidence.exact
    assert evidence.approximation_bound <= evidence.rounding_slack
    assert restored.target.mesh.topology_id == source.mesh.topology_id
    np.testing.assert_allclose(
        np.asarray(restored.target.geometry.coordinates),
        np.asarray(source.geometry.coordinates),
        rtol=0.0,
        atol=_SOURCE_ROUNDING,
    )
    with pytest.raises(ValueError, match="MeshAdaptationResult.transition"):
        restored.parent_cells


def _wavy_refinement() -> tuple[Any, Any, np.ndarray]:
    """A refined mesh whose fine map is not a restriction of any coarse P2 map."""

    affine = phx.meshing.certify_cell_mesh(
        phx.meshing.canonicalize_cell_mesh(transfer_triangles(2)), _CONTRACT
    )
    refined = _adapt(affine, (0,))
    mesh = refined.target.mesh
    straight = straight_geometry(mesh, 2)
    points = np.asarray(straight.coordinates)
    bumped = points.copy()
    bumped[:, 0] += 0.02 * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    geometry = CellGeometrySpec(
        dict(zip((block.name for block in mesh.blocks), straight.elements, strict=True)),
        dict(
            zip(
                (block.name for block in mesh.blocks), straight.geometry_dofs, strict=True
            )
        ),
        bumped,
    )
    corners = np.asarray(mesh.coordinates).copy()
    for block, coordinate_element, route in zip(
        mesh.blocks, straight.elements, straight.geometry_dofs, strict=True
    ):
        assert isinstance(coordinate_element, FiniteElementSpec)
        corner_weights, _ = coordinate_element.tabulate(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
        )
        corners[np.asarray(block.vertices)] = (
            np.asarray(corner_weights) @ bumped[np.asarray(route)]
        )
    curved = phx.meshing.certify_cell_mesh(
        mesh.with_coordinates(corners, numeric_version="bumped"),
        _CONTRACT,
        geometry=geometry,
    )
    new_cells = np.setdiff1d(
        np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks]),
        np.concatenate([np.asarray(block.global_ids) for block in affine.mesh.blocks]),
    )
    return curved, refined.hierarchy, new_cells


def test_curved_coarsening_refuses_an_inexact_coarse_map_by_default() -> None:
    curved, hierarchy, new_cells = _wavy_refinement()
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _adapt(curved, (), new_cells, hierarchy=hierarchy)
    assert refusal.value.reason == "approximation_bound"
    assert refusal.value.measured > refusal.value.limit


def test_curved_coarsening_follows_its_declared_approximation_bound() -> None:
    curved, hierarchy, new_cells = _wavy_refinement()
    policy = CellGeometryTransitionPolicy(
        coarsening="bounded_interpolation", coarsening_tolerance=1e-2
    )
    coarse = _adapt(
        curved, (), new_cells, hierarchy=hierarchy, geometry_transition=policy
    )
    evidence = coarse.geometry_transition.evidence
    assert not evidence.exact
    assert 0.0 < evidence.approximation_bound <= evidence.approximation_tolerance
    # The certified sup bound dominates the deviation seen at fine-map nodes:
    # coarse nodes lie on the fine map, so sample the coarse map at fine nodes.
    witnesses = coarse.geometry_transition
    fine_ids = np.asarray(witnesses.coarsened_cell_ids)
    coarse_ids = np.asarray(witnesses.coarsened_into_ids)
    corners = np.asarray(witnesses.coarsened_reference_vertices)
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in curved.mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in coarse.target.mesh.blocks]
    )
    nodes = np.asarray(curved.geometry.elements[0].reference_nodes)
    barycentric = np.concatenate((1.0 - nodes.sum(axis=1, keepdims=True), nodes), axis=1)
    observed = 0.0
    for fine, parent, reference in zip(fine_ids, coarse_ids, corners, strict=True):
        fine_row = int(np.flatnonzero(source_ids == fine)[0])
        coarse_row = int(np.flatnonzero(target_ids == parent)[0])
        observed = max(
            observed,
            float(
                np.max(
                    np.abs(
                        _map(coarse.target, coarse_row, barycentric @ reference)
                        - _map(curved, fine_row, nodes)
                    )
                )
            ),
        )
    assert observed <= evidence.approximation_bound


def test_metric_route_still_refuses_curved_sources_it_cannot_carry() -> None:
    term = shear_term(2, 2)
    source = curved_shear_source(transfer_triangles(2), 2, term)
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    metric = phx.meshing.MeshMetricField(
        phx.meshing.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            vertices.entity_ids,
        ),
        np.broadcast_to(16.0 * np.eye(2), (vertices.count, 2, 2)),
        minimum_size=0.25,
        maximum_size=0.25,
    )
    with pytest.raises(ValueError, match="no geometry transition"):
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(metric),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_2D),
        )


def test_displacement_keeps_curvature_and_moves_vertex_nodes() -> None:
    term = shear_term(2, 2)
    source = curved_shear_source(transfer_triangles(2), 2, term)
    offset = np.asarray((0.25, -0.5))
    moved = source.mesh.with_coordinates(
        np.asarray(source.mesh.coordinates) + offset, numeric_version="translated"
    )
    transition = transition_displaced_cell_geometry(source.mesh, source.geometry, moved)
    # A rigid translation moves every node, curved or not, by the same offset.
    np.testing.assert_allclose(
        np.asarray(transition.geometry.coordinates),
        np.asarray(source.geometry.coordinates) + offset,
        rtol=0.0,
        atol=_MOTION_ROUNDING,
    )
    assert transition.evidence.kind == "vertex_displacement"
    assert abs(transition.evidence.target_measure - 1.0) < _MEASURE_ROUNDING


@pytest.fixture(scope="module")
def displaced_measure_source() -> Any:
    from phydrax.meshing.qualification import native_planar_part

    part, _ = native_planar_part("displaced-measures", 1.0, size=0.2)
    return part.carrier


def _earned_displaced_measures(source: Any) -> tuple[Any, Any, Any, Any]:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
    from phydrax.meshing._measurements import NativeExecutionRecord
    from phydrax.meshing._native_preparation import native_preparation_ledger

    coordinates = np.asarray(source.mesh.coordinates).copy()
    free = np.flatnonzero(np.asarray(source.geometry.exact_source.vertex_strata) == 0)
    assert free.size
    coordinates[free[0], 0] += 0.001
    target = source.mesh.with_coordinates(coordinates, numeric_version="issued-motion")
    limits = phx.meshing.MeshingLimits(maximum_work_units=10_000_000)
    collector = CoordinateOperandCollector(capture_measures=True)
    with NativeExecutionScope(limits.native_limits()) as scope:
        collector.begin(
            scope, "issued-displacement", native_preparation_ledger(scope).overlap(scope, None)
        )
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            budget.capture_coordinate_operands = True
            budget.capture_measure_operands = True
            with budget.activate(), ticket.native_target():
                transition = transition_displaced_cell_geometry(
                    source.mesh, source.geometry, target, budget=budget
                )
                collector.capture(
                    target, transition.geometry, budget, scope,
                    input_owners=(source.mesh, source.geometry),
                )
    receipt = NativeExecutionRecord(
        scope.completed_evidence(), owner_id="issued-displacement"
    )
    collector.complete(receipt)
    assert collector.prepared is not None
    return target, transition, collector.prepared, receipt


def _replay_displaced_measures(
    source: Any,
    target: Any,
    transition: Any,
    operands: Any,
    receipt: Any,
    *,
    warm: bool = True,
    work: int = 10_000_000,
    storage: int = 1_000_000_000,
) -> tuple[Any, Any]:
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.meshing._native_preparation import native_preparation_ledger

    limits = phx.meshing.MeshingLimits(
        maximum_work_units=work, maximum_scratch_bytes=storage
    )
    with NativeExecutionScope(limits.native_limits()) as scope:
        scope.observe_managed_admission()
        native_preparation_ledger(scope).import_receipt(scope, receipt)
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=storage
            )
            with budget.activate(), ticket.native_target():
                if warm:
                    operands.require(operands.mesh, operands.geometry, receipt, budget)
                rebuilt = transition_displaced_cell_geometry(
                    source.mesh, source.geometry, target, budget=budget,
                    _issued_transition=transition,
                )
    return rebuilt, scope.completed_evidence()


def test_displaced_measure_issue_reconstructs_bitwise_and_rejects_replay(
    displaced_measure_source: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.discretization import _cell_geometry_transfer as transfer_owner
    from phydrax.lifecycle._meshing_source_records import content_equal

    source = displaced_measure_source
    target, issued, operands, receipt = _earned_displaced_measures(source)
    original = transfer_owner.mapped_measure_owner.certified_cell_measures
    calls: list[object] = []

    def measured(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        transfer_owner.mapped_measure_owner, "certified_cell_measures", measured
    )
    rebuilt, _ = _replay_displaced_measures(source, target, issued, operands, receipt)
    assert calls == []
    assert rebuilt is not issued and rebuilt.geometry is not issued.geometry
    assert rebuilt.evidence is not issued.evidence
    assert id(rebuilt) not in transfer_owner._ISSUED_DISPLACED_MEASURES
    assert content_equal(rebuilt, issued, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
    cold, _ = _replay_displaced_measures(
        source, target, issued, operands, receipt, warm=False
    )
    assert len(calls) == 2
    assert content_equal(cold, issued, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
    # A reconstructed equal transition cannot forward the original numeric seal.
    replay, _ = _replay_displaced_measures(source, target, rebuilt, operands, receipt)
    assert len(calls) == 4
    assert content_equal(replay, issued, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
    with coordinate_reference_budget(
        phx.meshing.MeshingLimits(maximum_work_units=10_000_000),
        queries_upper=10_000_000,
    ) as budget:
        budget.capture_coordinate_operands = True
        budget.capture_measure_operands = True
        standalone = transition_displaced_cell_geometry(
            source.mesh, source.geometry, target, budget=budget
        )
    assert id(standalone) not in transfer_owner._ISSUED_DISPLACED_MEASURES
    _replay_displaced_measures(source, target, standalone, operands, receipt)
    assert len(calls) == 8


@pytest.mark.parametrize(
    "fault",
    (
        "copy", "replace", "transition-field", "evidence-field", "reference-bank",
        "source-copy", "geometry-copy", "target-copy", "target-geometry",
    ),
)
def test_displaced_measure_issue_misses_changed_actual_bindings(
    displaced_measure_source: Any, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from dataclasses import replace

    import jax.numpy as jnp

    from phydrax.discretization import _cell_geometry_transfer as transfer_owner

    source = displaced_measure_source
    target, issued, operands, receipt = _earned_displaced_measures(source)
    transition = issued
    if fault == "copy":
        transition = copy(issued)
    elif fault == "replace":
        transition = replace(issued)
    elif fault == "transition-field":
        object.__setattr__(issued, "source_geometry_id", "foreign")
    elif fault == "evidence-field":
        object.__setattr__(issued.evidence, "target_measure", issued.evidence.target_measure + 1.0)
    elif fault == "reference-bank":
        object.__setattr__(
            issued, "parent_reference_vertices", issued.parent_reference_vertices + jnp.asarray(0.125)
        )
    elif fault == "source-copy":
        source = copy(source)
        object.__setattr__(source, "mesh", copy(source.mesh))
    elif fault == "geometry-copy":
        source = copy(source)
        object.__setattr__(source, "geometry", copy(source.geometry))
    elif fault == "target-copy":
        target = copy(target)
    elif fault == "target-geometry":
        coordinates = np.asarray(target.coordinates).copy()
        free = np.flatnonzero(np.asarray(source.geometry.exact_source.vertex_strata) == 0)
        coordinates[free[0], 0] += 0.001
        target = target.with_coordinates(coordinates, numeric_version="other-motion")
    else:
        raise AssertionError(fault)
    original = transfer_owner.mapped_measure_owner.certified_cell_measures
    calls: list[object] = []

    def measured(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(
        transfer_owner.mapped_measure_owner, "certified_cell_measures", measured
    )
    _replay_displaced_measures(source, target, transition, operands, receipt)
    assert len(calls) == 2


def test_displaced_measure_issue_rejects_unpaid_request_and_releases_weak_owner(
    displaced_measure_source: Any,
) -> None:
    import gc
    from weakref import ref

    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization import _cell_geometry_transfer as transfer_owner

    source = displaced_measure_source
    target, issued, operands, receipt = _earned_displaced_measures(source)
    identity, weak = id(issued), ref(issued)
    assert identity in transfer_owner._ISSUED_DISPLACED_MEASURES
    limits = phx.meshing.MeshingLimits()
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with pytest.raises(ValueError, match="already be paid"):
                operands.require(target, operands.geometry, receipt, budget)
    del issued
    gc.collect()
    assert weak() is None
    assert identity not in transfer_owner._ISSUED_DISPLACED_MEASURES


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_displaced_measure_issue_exact_request_boundaries(
    displaced_measure_source: Any, resource: str,
) -> None:
    from phydrax._meshcore import current_native_execution_scope, MeshcoreError
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    source = displaced_measure_source
    target, issued, operands, receipt = _earned_displaced_measures(source)
    _, measured = _replay_displaced_measures(source, target, issued, operands, receipt)
    required = (
        measured.work_counters.construction_work if resource == "work"
        else measured.managed_admission_peak_bytes_upper
    )
    assert required is not None and required > 0
    _replay_displaced_measures(
        source, target, issued, operands, receipt, **{resource: required}
    )
    with pytest.raises((MeshcoreError, CellGeometryTransitionError, CoordinateEnclosureResourceError)):
        _replay_displaced_measures(
            source, target, issued, operands, receipt, **{resource: required - 1}
        )
    assert current_native_execution_scope() is None


def test_geometry_transition_refuses_beyond_its_evaluation_budget() -> None:
    source = curved_shear_source(transfer_triangles(2), 2, shear_term(2, 2))
    policy = CellGeometryTransitionPolicy(maximum_evaluations=10)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _adapt(source, (0,), geometry_transition=policy)
    assert refusal.value.reason == "resource_limit"


def test_incomplete_topology_edit_refuses_atomically(
    budget: CoordinateEnclosureBudget,
) -> None:

    source = transfer_triangles(1)
    ids = np.asarray(source.vertex_global_ids)
    before = np.asarray(source.coordinates).copy()
    block = source.blocks[0]
    witness = NestedReferenceWitnesses(
        np.asarray(block.global_ids)[:1],
        np.asarray(block.global_ids)[:1],
        np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
    )
    relations = tuple(
        EntityRelations(
            dimension,
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0, 1), dtype=np.int64),
            np.zeros(0, dtype=np.int32),
        )
        for dimension in range(3)
    )
    edit = CellTopologyEdit(
        "nested_refinement",
        before.copy(),
        ids,
        (
            TopologyEditBlock(
                block.name,
                block.cell_kind,
                block.cell_kind,
                np.asarray(block.vertices),
                np.asarray(block.global_ids),
            ),
        ),
        ids[:, None],
        np.ones((ids.size, 1), dtype=np.float64),
        np.ones((ids.size, 1), dtype=np.bool_),
        relations,
        refinement=witness,
    )
    with pytest.raises(ValueError, match=r"CellTopologyEdit\.witnesses"):
        assemble_topology_edit(source, edit, numeric_version="refused", budget=budget)
    np.testing.assert_array_equal(source.coordinates, before)
    np.testing.assert_array_equal(edit.coordinates, before)


def _coordinate_field(discretization: Any, result: Any) -> np.ndarray:
    """Physical-coordinate coefficients at the field's own nodal functionals."""

    routes = np.asarray(discretization.dof_maps[0].cell_dofs[0])
    nodes = np.asarray(result.geometry.geometry_dofs[0])
    values = np.zeros((discretization.dof_maps[0].global_dof_count, 2))
    weights, _ = result.geometry.elements[0].tabulate(
        discretization.elements[0][0].reference_nodes
    )
    local = np.asarray(weights) @ np.asarray(result.geometry.coordinates)[nodes]
    values[routes.reshape(-1)] = local.reshape((-1, 2))
    return values


def _discretization(result: Any) -> Any:
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 2)
    )
    return phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()


def test_nested_field_transfer_reproduces_curved_coordinate_fields() -> None:
    source = curved_shear_source(transfer_triangles(2), 2, shear_term(2, 2))
    refined = _adapt(source, (0, 4))
    coarse = _discretization(source)
    fine = _discretization(refined.target)

    with pytest.raises(ValueError, match=r"require_affine_simplices\.coordinate_element"):
        phx.discretization.prepare_nested_field_transfer(
            coarse, fine, refined.parent_cells, field_name="u"
        )
    transfer = phx.discretization.prepare_nested_field_transfer(
        coarse,
        fine,
        refined.parent_cells,
        field_name="u",
        parent_reference_vertices=refined.parent_reference_vertices,
        parent_reference_homogeneous=refined.parent_reference_homogeneous,
    )

    # Physical coordinates are P2 fields of an isoparametric P2 map whose
    # coefficients are the geometry nodes; the algebraic nested transfer must carry
    # the coarse coordinate field onto the refined one.
    np.testing.assert_allclose(
        np.asarray(transfer.transfer.apply(_coordinate_field(coarse, source))),
        _coordinate_field(fine, refined.target),
        rtol=0.0,
        atol=_RESTRICTION_ROUNDING,
    )
    assert transfer.evidence.passed
    assert transfer.geometry.relation == "exact-restriction"
    assert transfer.transfer.preserves_constants
    assert not transfer.transfer.conservative


def _source_realization_case() -> tuple[
    CellMesh,
    CellGeometrySpec,
    CellMesh,
    CellGeometrySpec,
    phx.discretization.SourceGeometryRealization,
]:
    source = phx.meshing.canonicalize_cell_mesh(transfer_triangles(1))
    old = CellGeometrySpec.affine(source)
    layout = straight_geometry(source, 2)

    def moved(values: np.ndarray) -> np.ndarray:
        result = np.array(values, dtype=np.float64, copy=True)
        result[:, 1] *= 1.0 + result[:, 0] / 4.0
        return result

    target = source.with_coordinates(
        moved(np.asarray(source.coordinates)), numeric_version="source-realized"
    )
    names = tuple(block.name for block in source.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, layout.elements, strict=True)),
        dict(zip(names, layout.geometry_dofs, strict=True)),
        moved(np.asarray(layout.coordinates)),
    )
    old_validity = phx.discretization.certify_cell_geometry_validity(old, mesh=source)
    new_validity = phx.discretization.certify_cell_geometry_validity(
        geometry, mesh=target
    )
    old_embedding = phx.geometry.certify_global_embedding(source, old, old_validity)
    new_embedding = phx.geometry.certify_global_embedding(target, geometry, new_validity)
    realization = phx.discretization.prepare_source_geometry_realization(
        source,
        old,
        target,
        geometry,
        source_embedding=old_embedding,
        target_embedding=new_embedding,
        policy=CellGeometryTransitionPolicy(
            reconstruction="source_realization", reconstruction_tolerance=0.3
        ),
        maximum_storage_bytes=16 * 1024**2,
    )
    return source, old, target, geometry, realization


def test_source_realization_records_changed_physical_support_and_enclosed_measures() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    transition = realization.transition
    assert source.topology_id == target.topology_id
    assert transition.source_geometry_id != transition.target_geometry_id
    assert transition.evidence.source_measure == pytest.approx(1.0, abs=_SOURCE_ROUNDING)
    assert transition.evidence.target_measure == pytest.approx(
        9.0 / 8.0, abs=_SOURCE_ROUNDING
    )
    assert not transition.evidence.exact
    assert transition.evidence.coverage_defect is None
    assert 0.0 < transition.evidence.approximation_bound <= 0.3
    assert transition.evidence.source_measure_error_bound is not None
    assert transition.evidence.target_measure_error_bound is not None
    assert transition.evidence.source_measure_error_bound < _MEASURE_ROUNDING
    assert transition.evidence.target_measure_error_bound < _MEASURE_ROUNDING


def test_source_realization_density_projection_conserves_inventory_not_constant_values() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    field = phx.discretization.FiniteElementFieldSpec(
        "density", phx.discretization.discontinuous_element("triangle", 0)
    )
    before = phx.discretization.FiniteElementPlan(
        source, field, coordinate_spec=old
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=new
    ).prepare()
    transfer = phx.discretization.prepare_source_realization_field_transfer(
        before,
        after,
        realization,
        field_name="density",
        source_geometry=old,
        target_geometry=new,
        semantics="conservative-density",
    )
    values = np.full(before.dof_maps[0].global_dof_count, 2.0, dtype=np.float64)
    carried = np.asarray(transfer.transfer.apply(values))
    expected = np.empty_like(carried)
    for row, route in enumerate(np.asarray(after.dof_maps[0].cell_dofs[0])):
        expected[route] = (
            2.0
            * np.asarray(realization.source_cell_measures)[row]
            / np.asarray(realization.target_cell_measures)[row]
        )
    np.testing.assert_allclose(carried, expected, rtol=0.0, atol=_RESTRICTION_ROUNDING)
    assert not np.allclose(carried, 2.0, rtol=0.0, atol=_RESTRICTION_ROUNDING)
    assert float(np.dot(carried, np.asarray(transfer.target_measures))) == pytest.approx(
        2.0, abs=_RESTRICTION_ROUNDING
    )
    assert transfer.geometry.relation == "source-realization"
    assert transfer.geometry.coverage_defect is None
    assert transfer.evidence.passed
    assert transfer.evidence.bound("content") < _SOLVE_ACCURACY
    assert transfer.transfer.conservative
    assert not transfer.transfer.preserves_constants


def test_source_realization_intensive_history_is_material_not_world_point_interpolation() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    field = phx.discretization.FiniteElementFieldSpec(
        "history", phx.discretization.lagrange_element("triangle", 1)
    )
    before = phx.discretization.FiniteElementPlan(
        source, field, coordinate_spec=old
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=new
    ).prepare()
    transfer = phx.discretization.prepare_source_realization_field_transfer(
        before,
        after,
        realization,
        field_name="history",
        source_geometry=old,
        target_geometry=new,
        semantics="intensive",
    )
    values = before.dof_maps[0].dof_coordinates[:, 1] + 0.5
    carried = np.asarray(transfer.transfer.apply(values))
    np.testing.assert_allclose(carried, values, rtol=0.0, atol=0.0)
    assert not np.allclose(
        carried, np.asarray(after.dof_maps[0].dof_coordinates)[:, 1] + 0.5
    )
    assert transfer.semantics == "material-pullback"
    assert not transfer.transfer.conservative


def test_source_realization_flux_retains_material_cochain_and_actual_piola_values() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    element = phx.discretization.form_element(
        "triangle", 1, 1, twist="twisted", proxy="flux"
    )
    field = phx.discretization.FiniteElementFieldSpec("flux", element)
    before = phx.discretization.FiniteElementPlan(
        source, field, coordinate_spec=old
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=new
    ).prepare()
    transfer = phx.discretization.prepare_source_realization_field_transfer(
        before,
        after,
        realization,
        field_name="flux",
        source_geometry=old,
        target_geometry=new,
        semantics="material-compatible",
    )
    values = np.linspace(0.3, 1.4, before.dof_maps[0].global_dof_count)
    carried = np.asarray(transfer.transfer.apply(values))
    points = np.asarray(((0.2, 0.3), (0.6, 0.1)), dtype=np.float64)

    def physical_values(
        space: phx.discretization.FiniteElementDiscretization, coefficients: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = space.evaluate_block_geometry(
            "flux", 0, space.default_runtime.coordinates, points, np.ones(points.shape[0])
        )
        dof_map = space.dof_maps[0]
        local = np.einsum(
            "cij,cj->ci",
            np.asarray(dof_map.cell_transforms[0]),
            coefficients[np.asarray(dof_map.cell_dofs[0])],
        )
        basis = np.asarray(geometry.basis_values).reshape(
            (local.shape[0], points.shape[0], local.shape[1], 2)
        )
        return np.asarray(geometry.physical_points), np.einsum(
            "cqiv,ci->cqv", basis, local
        )

    old_points, old_values = physical_values(before, values)
    _, new_values = physical_values(after, carried)
    determinant = 1.0 + old_points[..., 0] / 4.0
    expected = np.empty_like(old_values)
    expected[..., 0] = old_values[..., 0] / determinant
    expected[..., 1] = old_values[..., 1] + old_points[..., 1] * old_values[..., 0] / (
        4.0 * determinant
    )
    np.testing.assert_allclose(
        new_values, expected, rtol=_PIOLA_ROUNDING, atol=_PIOLA_ROUNDING
    )
    assert not np.allclose(
        new_values, old_values, rtol=_SOLVE_ACCURACY, atol=_SOLVE_ACCURACY
    )
    assert transfer.evidence.passed


def test_scientific_block_authority_survives_hierarchy_archive(tmp_path: Path) -> None:

    _, hierarchy, _ = _wavy_refinement()
    receipt = write_meshing_source_closure(tmp_path / "scientific-hierarchy", hierarchy)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, BisectionHierarchy)
    assert restored.hierarchy_id == hierarchy.hierarchy_id
    np.testing.assert_array_equal(
        restored.scientific_cell_ids, hierarchy.scientific_cell_ids
    )
    np.testing.assert_array_equal(
        restored.scientific_block_ids, hierarchy.scientific_block_ids
    )
    assert np.unique(np.asarray(restored.scientific_block_ids)).size == 1


def test_geometry_transition_nested_phases_share_one_allowance_origin(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(37)
    policy = CellGeometryTransitionPolicy(maximum_evaluations=5)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        with charged_geometry_transition(policy, budget=budget):
            budget.reserve(3)
            with charged_geometry_transition(policy, budget=budget):
                budget.reserve(2)
            with charged_geometry_transition(policy, budget=budget):
                budget.reserve(1)
    assert refusal.value.reason == "resource_limit"
    assert refusal.value.measured == 6
    assert refusal.value.limit == 5
    assert budget.work_units == 42
    # Closing a refused transition releases its stage, never request work.
    with charged_geometry_transition(
        CellGeometryTransitionPolicy(maximum_evaluations=1), budget=budget
    ):
        budget.reserve(1)
    assert budget.work_units == 43


def test_geometry_transition_still_obeys_the_remaining_request_allowance() -> None:
    with coordinate_reference_budget(
        phx.meshing.MeshingLimits(maximum_work_units=39)
    ) as budget:
        budget.reserve(37)
        with pytest.raises(CellGeometryTransitionError) as refusal:
            with charged_geometry_transition(
                CellGeometryTransitionPolicy(maximum_evaluations=5), budget=budget
            ):
                budget.reserve(3)
        assert refusal.value.measured == 3
        assert refusal.value.limit == 2
        assert budget.work_units == 37


def test_geometry_transition_preserves_declared_source_local_work_cap(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._exact_plc_geometry import ExactPlcBudget

    budget.reserve(37)
    source_work = ExactPlcBudget(1, 4096, budget=budget)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        with charged_geometry_transition(
            CellGeometryTransitionPolicy(maximum_evaluations=5), budget=budget
        ):
            source_work.charge(2)
    assert refusal.value.measured == 2
    assert refusal.value.limit == 1
    assert budget.work_units == 37


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger

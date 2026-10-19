#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import assert_never, Literal, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    NativeExecutionWorkCounters,
    TetMesh3D,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem import (
    discontinuous_element,
    form_element,
    lagrange_element,
    prepare_l2_projection_target,
    prepare_projection_field_transfer,
)
from phydrax.geometry import CommonRefinementPolicy, prepare_common_refinement
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._metric import metric_simplex_quality
from phydrax.meshing._tetra_metric import (
    _metric_relocation_merit,
    constraints,
    execute_native_tetra_metric,
    execute_tetra_metric_adaptation,
    MetricMemoryEvidence,
    MetricOptimizerEvidence,
    MetricRemeshingEvidence,
    MetricRemeshingStatus,
    TetraMetricSource,
)
from phydrax.meshing._topology_edit import assemble_topology_edit
from tests._support.cell_meshes import kuhn_cube_arrays
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.metric_meshes import (
    TETRA_DG_TOLERANCE,
    TETRA_FIELD_TOLERANCE,
    TETRA_GEOMETRY_TOLERANCE,
    TETRA_VOLUME_TOLERANCE,
)


_METRIC_LIMITS = NativeExecutionLimits(
    1_000_000_000, 10_000_000, 512, 8_000_000_000, np.inf
)


@contextmanager
def _metric_request(
    limits: NativeExecutionLimits, /, *, mesh: TetMesh3D | None = None
) -> Iterator[tuple[NativeExecutionScope, CoordinateEnclosureBudget]]:
    ticket = HostBatchTicket(0)
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_managed_bytes,
    )
    with ticket, budget.activate():
        pass
    with NativeExecutionScope(limits, mesh=mesh) as scope:
        yield scope, budget


def _box() -> CellMesh:
    points, tetrahedra = kuhn_cube_arrays(orientation_swap=(0, 1))
    points *= np.asarray((2.0, 0.5, 1.0), dtype=np.float64)
    return CellMesh(points, (CellBlock("box", "tetrahedron", tetrahedra),))


class _PublicMetricRemesh(NamedTuple):
    source: phx.meshing.CellMeshingResult
    result: phx.meshing.MeshAdaptationResult


@pytest.fixture(scope="module")
def public_metric_remesh() -> _PublicMetricRemesh:
    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    source = phx.meshing.certify_cell_mesh(_box(), phx.SpatialCoordinateContract.si())
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (8, 3, 3)),
        minimum_size=0.1,
        maximum_size=2.0,
    )
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MetricMeshAdaptation(metric),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D,
                maximum_passes=12,
                relocation=False,
            ),
        )
    )
    return _PublicMetricRemesh(source, result)


def _assert_anisotropic_unit_mesh(mesh: CellMesh) -> None:
    points = np.asarray(mesh.coordinates)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = points[cells]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.all(volumes > 0)
    np.testing.assert_allclose(
        np.sum(volumes), 1.0, atol=TETRA_GEOMETRY_TOLERANCE, rtol=0
    )
    # Independent Euclidean lengths in the declared anisotropic pullback.
    edges = np.unique(
        np.sort(cells[:, tuple(combinations(range(4), 2))].reshape((-1, 2)), axis=1),
        axis=0,
    )
    delta = (points[edges[:, 1]] - points[edges[:, 0]]) * np.asarray(
        (0.5, 2.0, 1.0), dtype=np.float64
    )
    lengths = np.linalg.norm(delta, axis=1)
    assert np.count_nonzero((lengths < 1 / np.sqrt(2.0)) | (lengths > np.sqrt(2.0))) == 0


def test_public_metric_remesh_publishes_domain_and_size_evidence(
    public_metric_remesh: _PublicMetricRemesh,
) -> None:
    source, result = public_metric_remesh
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert source.audit.passed and result.target.audit.passed
    evidence = result.evidence
    assert isinstance(evidence, MetricRemeshingEvidence)
    mesh = result.target.mesh
    _assert_anisotropic_unit_mesh(mesh)
    assert evidence.out_of_range_edges == 0
    assert (
        evidence.work_units
        == evidence.native_work_units + evidence.optimizer.host_work_units
    )
    assert evidence.native_work_units > 0
    assert evidence.optimizer.refused_proposals <= sum(
        count for _, count in evidence.optimizer.refusal_counts
    )
    execution = evidence.execution_evidence
    assert execution is not None
    execution.require_valid()
    field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
    space = FiniteElementPlan(
        mesh, field, coordinate_spec=result.target.geometry
    ).prepare()
    integrated = sum(
        float(np.sum(np.asarray(block.measure)))
        for block in space.evaluate_geometry("u", result.target.geometry.coordinates)
    )
    np.testing.assert_allclose(integrated, 1.0, atol=TETRA_GEOMETRY_TOLERANCE, rtol=0)


def test_public_metric_guard_rejects_unrefined_source(
    public_metric_remesh: _PublicMetricRemesh,
) -> None:
    corrupted = eqx.tree_at(
        lambda result: result.target,
        public_metric_remesh.result,
        public_metric_remesh.source,
    )
    with pytest.raises(AssertionError):
        _assert_anisotropic_unit_mesh(corrupted.target.mesh)


def test_metric_quality_is_regular_after_anisotropic_coordinate_pullback() -> None:
    regular = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, np.sqrt(3.0) / 2.0, 0.0),
            (0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)),
        ),
        dtype=np.float64,
    )
    points = regular * np.asarray((2.0, 0.5, 1.0), dtype=np.float64)
    metric = np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (4, 3, 3))
    quality = metric_simplex_quality(
        metric, points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    np.testing.assert_allclose(
        quality, 1.0, rtol=TETRA_GEOMETRY_TOLERANCE, atol=TETRA_GEOMETRY_TOLERANCE
    )


def test_unit_edges_do_not_hide_tetrahedral_slivers() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0e-9)),
        dtype=np.float64,
    )
    edges = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    assert np.all((lengths >= 1.0) & (lengths <= np.sqrt(2.0) + TETRA_GEOMETRY_TOLERANCE))
    quality = np.asarray(
        metric_simplex_quality(
            np.broadcast_to(np.eye(3), (4, 3, 3)),
            points,
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        )
    )
    assert quality[0] < 1.0e-5


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_anisotropic_box_remesh_keeps_exact_domain_and_p1_source_coverage() -> None:
    source = _box()
    coordinates = np.asarray(source.coordinates).copy()
    metric = np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (8, 3, 3))
    with _metric_request(_METRIC_LIMITS) as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            source,
            metric,
            maximum_passes=12,
            relocation=False,
            scope=scope,
            budget=budget,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    with coordinate_reference_budget() as budget:
        target, lineage, stencil = assemble_topology_edit(
            source, outcome.edit, numeric_version="box-remesh", budget=budget
        )
    corners = np.asarray(target.coordinates)[np.asarray(target.blocks[0].vertices)]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.all(volumes > 0.0)
    assert np.sum(volumes) == pytest.approx(1.0, abs=TETRA_GEOMETRY_TOLERANCE)
    assert outcome.evidence.out_of_range_edges == 0
    assert lineage.source_topology_id == source.topology_id
    sources = np.asarray(outcome.edit.stencil_sources)
    weights = np.asarray(outcome.edit.stencil_weights)
    source_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source.vertex_global_ids))
    }
    reproduced = np.sum(
        coordinates[
            np.asarray(
                [[source_rows[int(value)] for value in row] for row in sources],
                dtype=np.int32,
            )
        ]
        * weights[..., None],
        axis=1,
    )
    np.testing.assert_allclose(
        reproduced, target.coordinates, atol=TETRA_GEOMETRY_TOLERANCE
    )
    np.testing.assert_array_equal(source.coordinates, coordinates)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_repeated_boundary_splits_retain_native_source_facet_lineage() -> None:
    source = _box()
    metric = np.broadcast_to(np.diag((1.0, 16.0, 4.0)), (8, 3, 3))
    with _metric_request(_METRIC_LIMITS) as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            source,
            metric,
            maximum_passes=12,
            relocation=True,
            scope=scope,
            budget=budget,
        )
    with coordinate_reference_budget() as budget:
        target, lineage, _ = assemble_topology_edit(
            source, outcome.edit, numeric_version="boundary-refinement", budget=budget
        )
    old_connectivity, new_connectivity = source.connectivity, target.connectivity
    assert isinstance(old_connectivity, phx.discretization.TetrahedralConnectivity)
    assert isinstance(new_connectivity, phx.discretization.TetrahedralConnectivity)
    old_ids = np.asarray(source.entity_set(2).entity_ids)[
        np.asarray(old_connectivity.boundary_faces)
    ]
    new_ids = np.asarray(target.entity_set(2).entity_ids)[
        np.asarray(new_connectivity.boundary_faces)
    ]
    assert new_ids.size > old_ids.size
    record = lineage.entity_lineage(2)
    assert not np.intersect1d(record.created_target_ids, new_ids).size
    relevant = np.isin(np.asarray(record.target_global_ids), new_ids)
    assert np.all(np.isin(np.asarray(record.source_global_ids)[relevant], old_ids))
    np.testing.assert_array_equal(
        np.unique(np.asarray(record.target_global_ids)[relevant]), np.sort(new_ids)
    )


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_exhausted_metric_work_keeps_source_and_reports_unmet_edges() -> None:
    source = _box()
    before = np.asarray(source.coordinates).copy()
    with _metric_request(_METRIC_LIMITS) as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            source,
            np.broadcast_to(np.diag((1.0, 16.0, 4.0)), (8, 3, 3)),
            maximum_operations=0,
            maximum_passes=4,
            scope=scope,
            budget=budget,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.RESOURCE_LIMIT
    assert outcome.evidence.out_of_range_edges > 0
    assert outcome.evidence.splits == 0
    np.testing.assert_array_equal(outcome.edit.coordinates, before)
    np.testing.assert_array_equal(source.coordinates, before)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
@pytest.mark.parametrize("family", ["h1", "dg", "hcurl", "hdiv"])
def test_repeated_native_remesh_transfers_each_declared_field_family(
    family: Literal["h1", "dg", "hcurl", "hdiv"],
) -> None:
    match family:
        case "h1":
            element = lagrange_element("tetrahedron", 1)
        case "dg":
            element = discontinuous_element("tetrahedron", 0)
        case "hcurl":
            element = form_element("tetrahedron", 1, 1, proxy="circulation")
        case "hdiv":
            element = form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux")
        case unknown:
            assert_never(unknown)

    def field(points: Array, args: object) -> Array:
        del args
        if family == "hcurl":
            return jnp.asarray((0.5, -0.2, 1.3), dtype=jnp.float64) + jnp.cross(
                jnp.asarray((-0.7, 0.9, 0.4), dtype=jnp.float64),
                points,
            )
        if family == "hdiv":
            return jnp.asarray((0.5, -0.2, 1.3), dtype=jnp.float64) + 2.0 * points
        if family == "dg":
            return 1.0 + points[..., 0]
        return 0.75 + 2.0 * points[..., 0] - 0.3 * points[..., 1] + points[..., 2]

    def dg_content(
        discretization: FiniteElementDiscretization, coefficients: Array
    ) -> float:
        total = 0.0
        for block, routes in zip(
            discretization.mesh.blocks, discretization.dof_maps[0].cell_dofs, strict=True
        ):
            corners = np.asarray(discretization.mesh.coordinates)[
                np.asarray(block.vertices)
            ]
            volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
            averages = np.asarray(coefficients)[np.asarray(routes)[:, 0]]
            total += float(volumes @ averages)
        return total

    mesh = _box()
    specification = FiniteElementFieldSpec("u", element)
    prepared = FiniteElementPlan(mesh, specification).prepare()
    values = prepared.project("u", field)
    for epoch, scale in enumerate((1.0, 4.0), start=1):
        metric = np.broadcast_to(
            scale * np.diag((0.25, 4.0, 1.0)), (mesh.coordinates.shape[0], 3, 3)
        )
        with _metric_request(_METRIC_LIMITS) as (scope, budget):
            outcome = execute_tetra_metric_adaptation(
                mesh,
                metric,
                maximum_passes=16,
                relocation=False,
                scope=scope,
                budget=budget,
            )
        assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
        with coordinate_reference_budget() as budget:
            target, _, _ = assemble_topology_edit(
                mesh, outcome.edit, numeric_version=f"field-remesh:{epoch}", budget=budget
            )
        successor = FiniteElementPlan(target, specification).prepare()
        common = prepare_common_refinement(
            mesh, target, policy=CommonRefinementPolicy(overlap_simplices=True)
        )
        transfer = prepare_projection_field_transfer(
            prepared,
            prepare_l2_projection_target(successor, field_name="u"),
            common,
            field_name="u",
        )
        source_content = dg_content(prepared, values) if family == "dg" else 0.0
        source_minimum, source_maximum = float(np.min(values)), float(np.max(values))
        values = transfer.transfer.apply(values)
        assert transfer.evidence.passed
        assert transfer.evidence.defect("coverage") <= transfer.evidence.tolerance
        assert transfer.evidence.defect("reproduction") <= transfer.evidence.tolerance
        if family in ("hcurl", "hdiv"):
            assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
        if family == "dg":
            assert dg_content(successor, values) == pytest.approx(
                source_content, abs=TETRA_DG_TOLERANCE
            )
            assert np.min(values) >= source_minimum - TETRA_DG_TOLERANCE
            assert np.max(values) <= source_maximum + TETRA_DG_TOLERANCE
        else:
            np.testing.assert_allclose(
                values,
                successor.project("u", field),
                atol=TETRA_FIELD_TOLERANCE,
                rtol=TETRA_FIELD_TOLERANCE,
            )
        mesh, prepared = target, successor


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_metric_ring_reconnection_improves_quality_without_changing_boundary() -> None:
    radius, height = 0.75, 0.6
    points = np.asarray(
        (
            (0.0, 0.0, -height),
            (0.0, 0.0, height),
            (radius, 0.0, 0.0),
            (-radius / 2, radius * np.sqrt(3.0) / 2, 0.0),
            (-radius / 2, -radius * np.sqrt(3.0) / 2, 0.0),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3), (0, 1, 3, 4), (0, 1, 4, 2)), dtype=np.int32)
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    source = CellMesh.from_tetrahedra(points, cells)
    expected = np.asarray(((0, 2, 3, 4), (1, 2, 3, 4)), dtype=np.int32)

    def mean_ratio(tetrahedra: np.ndarray) -> np.ndarray:
        corners = points[tetrahedra]
        determinant = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
        edges = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
        squares = np.sum(
            (corners[:, edges[:, 1]] - corners[:, edges[:, 0]]) ** 2, axis=-1
        )
        return 6.0 * (2.0 * determinant**2) ** (1.0 / 3.0) / np.sum(squares, axis=1)

    requested = float((np.min(mean_ratio(cells)) + np.min(mean_ratio(expected))) / 2.0)
    with _metric_request(_METRIC_LIMITS) as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3), (5, 3, 3)),
            minimum_metric_quality=requested,
            relocation=False,
            maximum_passes=4,
            scope=scope,
            budget=budget,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            source, outcome.edit, numeric_version="ring-remesh", budget=budget
        )
    actual_cells = np.asarray(target.blocks[0].vertices)
    assert {tuple(row) for row in np.sort(actual_cells, axis=1)} == {
        tuple(row) for row in np.sort(expected, axis=1)
    }
    assert outcome.evidence.minimum_metric_quality >= requested
    assert outcome.evidence.out_of_range_edges == 0
    old_faces = np.sort(
        cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3)), axis=1
    )
    new_faces = np.sort(
        actual_cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3)),
        axis=1,
    )
    old_keys, old_uses = np.unique(old_faces, axis=0, return_counts=True)
    new_keys, new_uses = np.unique(new_faces, axis=0, return_counts=True)
    np.testing.assert_array_equal(old_keys[old_uses == 1], new_keys[new_uses == 1])


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_source_location_budget_refuses_atomically_with_resource_evidence() -> None:
    source = _box()
    before = np.asarray(source.coordinates).copy()
    with (
        pytest.raises(MeshingFailure) as caught,
        _metric_request(_METRIC_LIMITS) as (scope, budget),
    ):
        execute_tetra_metric_adaptation(
            source,
            np.broadcast_to(np.diag((0.25, 4.0, 1.0)), (8, 3, 3)),
            maximum_location_pairs=0,
            scope=scope,
            budget=budget,
        )
    assert caught.value.evidence.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert caught.value.evidence.requested == (("source_location_candidates", 0.0),)
    assert caught.value.evidence.achieved[0][1] > 0.0
    np.testing.assert_array_equal(source.coordinates, before)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_rotated_decimal_constraint_splits_at_an_exact_nonmidpoint() -> None:
    points = np.asarray(
        ((-1.6, 1.4, 0.9), (0.6, 0.5, -0.3), (1.0, 2.0, 1.0), (2.0, 0.0, 3.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    if np.linalg.det(points[cells[0, 1:]] - points[cells[0, :1]]) < 0.0:
        cells = cells[:, (1, 0, 2, 3)]
    faces = cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3))
    segments = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    source_volume = abs(np.linalg.det(points[cells[0, 1:]] - points[cells[0, :1]])) / 6.0
    native = TetMesh3D(
        points,
        cells,
        np.zeros((1,), dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        segments,
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    try:
        construction = native.edge_split_point(0, 1, work_limit=10000)
        assert construction is not None
        point, fraction = construction.position, construction.parameter
        assert 0.0 < fraction < 1.0
        assert fraction != 0.5
        inserted = native.split_edge(
            0, 1, point, source_fraction=fraction, work_limit=10000
        )
        assert inserted is not None
        target = native.arrays()
        corners = target.points[target.tetrahedra]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        assert np.all(volumes > 0.0)
        assert np.sum(volumes) == pytest.approx(
            source_volume, abs=TETRA_GEOMETRY_TOLERANCE
        )
        np.testing.assert_array_equal(target.points[:4], points)
        np.testing.assert_array_equal(target.points[inserted], point)
    finally:
        native.close()


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_coordinated_relocation_commits_once_and_refuses_the_entire_cavity_union() -> (
    None
):
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.2, 0.2),
        ),
        dtype=np.float64,
    )
    cells = np.asarray(
        ((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)), dtype=np.int32
    )
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    segments = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    native = TetMesh3D(
        points,
        cells,
        np.zeros((4,), dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        segments,
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
        max_vertices=32,
        max_tetrahedra=128,
    )
    try:
        inserted = native.split_edge(0, 4, points[4] / 2.0, work_limit=100000)
        assert inserted is not None
        before = native.arrays()
        rows = np.asarray((4, inserted), dtype=np.int32)
        positions = before.points[rows] + np.asarray((0.01, 0.01, 0.01), dtype=np.float64)
        assert native.relocate_vertices(rows, positions, work_limit=100000)
        accepted = native.arrays()
        np.testing.assert_array_equal(accepted.points[rows], positions)
        np.testing.assert_array_equal(accepted.points[:4], points[:4])
        np.testing.assert_array_equal(accepted.tetrahedra, before.tetrahedra)
        np.testing.assert_array_equal(accepted.faces, before.faces)
        np.testing.assert_array_equal(accepted.face_sources, before.face_sources)
        corners = accepted.points[accepted.tetrahedra]
        volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
        assert np.all(volumes > 0.0)
        assert np.sum(volumes) == pytest.approx(1.0 / 6.0, abs=TETRA_VOLUME_TOLERANCE)

        inverted = positions.copy()
        inverted[0] += 0.001
        inverted[1] = -1.0
        assert not native.relocate_vertices(rows, inverted, work_limit=100000)
        np.testing.assert_array_equal(native.arrays().points, accepted.points)
        assert not native.relocate_vertices(
            rows,
            positions + 0.001,
            radius_edge_bound=0.5,
            work_limit=100000,
        )
        np.testing.assert_array_equal(native.arrays().points, accepted.points)
        with pytest.raises(MeshcoreError) as caught:
            native.relocate_vertices(rows, positions + 0.001, work_limit=0)
        assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        np.testing.assert_array_equal(native.arrays().points, accepted.points)
        np.testing.assert_array_equal(native.arrays().tetrahedra, accepted.tetrahedra)
    finally:
        native.close()


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_batch_curve_construction_retains_exact_decimal_source_and_reports_refused_rows() -> (
    None
):
    points = np.asarray(
        ((-1.6, 1.4, 0.9), (0.6, 0.5, -0.3), (1.0, 2.0, 1.0), (2.0, 0.0, 3.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    if np.linalg.det(points[cells[0, 1:]] - points[cells[0, :1]]) < 0.0:
        cells = cells[:, (1, 0, 2, 3)]
    faces = cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape((-1, 3))
    native = TetMesh3D(
        points,
        cells,
        np.zeros((1,), dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32),
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    try:
        seed = native.edge_split_point(0, 1, work_limit=10000)
        assert seed is not None
        inserted = native.split_edge(
            0, 1, seed.position, source_fraction=seed.parameter, work_limit=10000
        )
        assert inserted is not None
        before = native.arrays()
        constructed, coordinates, mask = native.construct_curve_points(
            np.asarray((inserted, 0), dtype=np.int32),
            np.asarray(((0, 1), (1, 2)), dtype=np.int32),
            np.asarray(
                (points[0] + 0.625 * (points[1] - points[0]), points[0]), dtype=np.float64
            ),
            work_limit=10000,
        )
        np.testing.assert_array_equal(mask, np.asarray((True, False), dtype=np.bool_))
        assert np.all(np.isnan(constructed[1])) and np.isnan(coordinates[1])
        axis = int(np.argmax(np.abs(points[1] - points[0])))
        assert coordinates[0] == constructed[0, axis]
        source_delta = [
            Fraction(float(points[1, k])) - Fraction(float(points[0, k]))
            for k in range(3)
        ]
        target_delta = [
            Fraction(float(constructed[0, k])) - Fraction(float(points[0, k]))
            for k in range(3)
        ]
        for first, second in combinations(range(3), 2):
            assert (
                target_delta[first] * source_delta[second]
                == target_delta[second] * source_delta[first]
            )
        assert (
            min(points[0, axis], points[1, axis])
            < constructed[0, axis]
            < max(points[0, axis], points[1, axis])
        )
        reversed_points, reversed_coordinates, reversed_mask = (
            native.construct_curve_points(
                np.asarray((inserted, 0), dtype=np.int32),
                np.asarray(((1, 0), (2, 1)), dtype=np.int32),
                np.asarray(
                    (points[0] + 0.625 * (points[1] - points[0]), points[0]),
                    dtype=np.float64,
                ),
                work_limit=10000,
            )
        )
        np.testing.assert_array_equal(reversed_mask, mask)
        np.testing.assert_array_equal(reversed_points, constructed)
        np.testing.assert_array_equal(reversed_coordinates, coordinates)
        np.testing.assert_array_equal(native.arrays().points, before.points)
        np.testing.assert_array_equal(native.arrays().tetrahedra, before.tetrahedra)
        np.testing.assert_array_equal(
            native.arrays().segment_sources, before.segment_sources
        )
    finally:
        native.close()


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_metric_epoch_checks_original_host_deadline_before_preparation() -> None:

    mesh = _box()
    points = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices)
    regions = np.zeros((cells.shape[0],), dtype=np.int32)
    faces, face_ids, segments, segment_ids = constraints(
        mesh,
        cells,
        regions,
        None,
        None,
        np.zeros((mesh.entity_set(1).count,), dtype=np.bool_),
    )
    native = TetMesh3D(
        points,
        cells,
        regions,
        faces,
        face_ids,
        segments,
        segment_ids,
        boundary_policy="fixed",
    )
    try:
        limits = NativeExecutionLimits(1000000, 1000000, 512, 8000000000, np.inf)
        with _metric_request(limits, mesh=native) as (scope, budget):
            source = TetraMetricSource(
                points,
                cells,
                np.arange(points.shape[0], dtype=np.int64),
                np.arange(cells.shape[0], dtype=np.int64),
                budget=budget,
            )
            before = native.arrays()
            work = native.work_units()
            with pytest.raises(MeshingFailure) as caught:
                execute_native_tetra_metric(
                    native,
                    source,
                    np.broadcast_to(np.eye(3, dtype=np.float64), (points.shape[0], 3, 3)),
                    fixed_vertices=np.zeros((points.shape[0],), dtype=np.bool_),
                    protected_edges=set(),
                    maximum_passes=16,
                    topology_operations=True,
                    relocation=True,
                    minimum_metric_quality=0.05,
                    lower_metric_length=0.0,
                    upper_metric_length=np.sqrt(2.0),
                    maximum_vertices=4096,
                    maximum_cells=4096,
                    maximum_operations=100000,
                    maximum_work_units=1000000,
                    maximum_location_pairs=1000000,
                    maximum_cavity_cells=512,
                    maximum_cavity_work=1000000,
                    operation_started=monotonic() - 2.0,
                    maximum_wall_seconds=1.0,
                    scope=scope,
                    budget=budget,
                )
            assert caught.value.evidence.category is MeshingFailureCategory.TIMED_OUT
            assert caught.value.evidence.requested == (("maximum_wall_seconds", 1.0),)
            assert dict(caught.value.evidence.achieved)["elapsed_seconds"] > 1.0
            np.testing.assert_array_equal(native.arrays().points, before.points)
            np.testing.assert_array_equal(native.arrays().tetrahedra, before.tetrahedra)
            assert native.work_units() == work
    finally:
        native.close()


def test_metric_evidence_composes_optimizer_and_memory_counters() -> None:
    optimizer = MetricOptimizerEvidence(
        method_id="bounded-lm",
        iterations=2,
        evaluations=3,
        constraint_evaluations=4,
        proposal_evaluations=5,
        refused_proposals=1,
        refusal_counts=(("source_support", 1),),
        host_work_units=7,
    )
    memory = MetricMemoryEvidence(
        native_counters=NativeExecutionMemoryCounters(100, 10, 20, 5, 4, 0),
        optimizer_scratch_required_bytes=64,
    )
    evidence = MetricRemeshingEvidence(
        MetricRemeshingStatus.COMPLETE,
        passes=1,
        counts=(1, 0, 0, 0),
        rejected_operations=1,
        work_units=10,
        native_work_units=3,
        lengths=np.asarray([1.0], dtype=np.float64),
        quality=np.asarray([1.0], dtype=np.float64),
        optimizer=optimizer,
        memory=memory,
    )
    evidence.validate_restored()
    assert evidence.optimizer is optimizer
    assert evidence.memory is memory
    assert evidence.optimizer.evaluations == 3
    assert evidence.optimizer.constraint_evaluations == 4
    assert evidence.optimizer.proposal_evaluations == 5
    assert evidence.optimizer.refusal_counts == (("source_support", 1),)
    assert (
        evidence.work_units
        == evidence.native_work_units + evidence.optimizer.host_work_units
    )
    assert evidence.memory.native_counters is memory.native_counters
    assert memory.native_counters is not None
    assert dict(memory.native_counters.items()) == {
        "limit_bytes": 100,
        "live_bytes": 10,
        "peak_bytes": 20,
        "requested_bytes": 5,
        "allocations": 4,
        "refusals": 0,
    }
    assert evidence.memory.optimizer_scratch_required_bytes == 64


@pytest.mark.parametrize(
    "constructor",
    (
        pytest.param(lambda: MetricOptimizerEvidence(iterations=-1), id="iterations"),
        pytest.param(lambda: MetricOptimizerEvidence(evaluations=-1), id="evaluations"),
        pytest.param(
            lambda: MetricOptimizerEvidence(constraint_evaluations=-1), id="constraint"
        ),
        pytest.param(
            lambda: MetricOptimizerEvidence(proposal_evaluations=-1), id="proposal"
        ),
        pytest.param(lambda: MetricOptimizerEvidence(refused_proposals=-1), id="refused"),
        pytest.param(lambda: MetricOptimizerEvidence(host_work_units=-1), id="host-work"),
    ),
)
def test_metric_optimizer_evidence_refuses_negative_counters(
    constructor: Callable[[], MetricOptimizerEvidence],
) -> None:
    with pytest.raises(ValueError, match="MetricOptimizerEvidence.counters"):
        constructor()


def test_metric_memory_evidence_refuses_wrong_native_counter_owner() -> None:
    with pytest.raises(ValueError, match="MetricMemoryEvidence.native_counters"):
        MetricMemoryEvidence(native_counters=NativeExecutionWorkCounters())  # ty: ignore[invalid-argument-type]


def test_metric_evidence_archive_round_trip_preserves_composed_fields(
    tmp_path: Path,
) -> None:
    evidence = MetricRemeshingEvidence(
        MetricRemeshingStatus.COMPLETE,
        passes=1,
        counts=(0, 0, 0, 0),
        rejected_operations=0,
        work_units=5,
        native_work_units=2,
        lengths=np.asarray([1.0], dtype=np.float64),
        quality=np.asarray([1.0], dtype=np.float64),
        optimizer=MetricOptimizerEvidence(
            method_id="bounded-lm",
            evaluations=2,
            host_work_units=3,
            refusal_counts=(("source_support", 1),),
        ),
        memory=MetricMemoryEvidence(
            native_counters=NativeExecutionMemoryCounters(100, 10, 20, 5, 4, 0),
            optimizer_scratch_required_bytes=64,
        ),
    )
    receipt = write_meshing_source_closure(tmp_path / "metric-evidence", evidence)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, MetricRemeshingEvidence)
    assert restored.evidence_id == evidence.evidence_id
    assert restored.optimizer.method_id == evidence.optimizer.method_id
    assert restored.optimizer.evaluations == evidence.optimizer.evaluations
    assert restored.optimizer.host_work_units == evidence.optimizer.host_work_units
    assert restored.optimizer.refusal_counts == evidence.optimizer.refusal_counts
    assert restored.memory.native_counters is not None
    assert evidence.memory.native_counters is not None
    assert (
        restored.memory.native_counters.items() == evidence.memory.native_counters.items()
    )
    assert (
        restored.memory.optimizer_scratch_required_bytes
        == evidence.memory.optimizer_scratch_required_bytes
    )


@pytest.mark.parametrize("band", ((1.0 / np.sqrt(2.0), np.sqrt(2.0)), (0.8, 1.3)))
def test_metric_size_descent_preserves_the_declared_quality_floor(
    band: tuple[float, float],
) -> None:
    old = np.asarray((0.5, 1.6, 1.0), dtype=np.float64)
    improved = np.asarray((0.6, 1.5, 1.0), dtype=np.float64)
    assert _metric_relocation_merit(old, improved, 0.7, 0.6, 0.05, band)
    assert not _metric_relocation_merit(old, improved, 0.7, 0.04, 0.05, band)
    assert not _metric_relocation_merit(improved, old, 0.6, 0.7, 0.05, band)
    assert not _metric_relocation_merit(old, old, 0.7, 0.7, 0.05, band)
    assert _metric_relocation_merit(old, old, 0.6, 0.7, 0.05, band)


def test_metric_epoch_can_monotonically_repair_an_initial_quality_deficit() -> None:
    lengths = np.asarray((1.0,), dtype=np.float64)
    band = (0.8, 1.3)
    assert _metric_relocation_merit(lengths, lengths, 0.02, 0.03, 0.05, band)
    assert not _metric_relocation_merit(lengths, lengths, 0.03, 0.02, 0.05, band)
    assert _metric_relocation_merit(lengths, lengths, 0.03, 0.06, 0.05, band)
    assert not _metric_relocation_merit(lengths, lengths, 0.06, 0.04, 0.05, band)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_unrepaired_initial_quality_deficit_cannot_complete_the_metric_epoch() -> None:
    source = _box()
    before = np.asarray(source.coordinates).copy()
    with _metric_request(_METRIC_LIMITS) as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            source,
            np.broadcast_to(
                np.eye(3, dtype=np.float64), (source.coordinates.shape[0], 3, 3)
            ),
            minimum_metric_quality=1.0,
            topology_operations=False,
            relocation=False,
            scope=scope,
            budget=budget,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.STALLED
    assert outcome.evidence.minimum_metric_quality < 1.0
    assert outcome.evidence.operation_attempts == 0
    np.testing.assert_array_equal(source.coordinates, before)


@pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)
def test_partial_quality_repair_remains_an_uncommitted_metric_epoch() -> None:
    corners = 1.4 * np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, np.sqrt(3.0) / 2.0, 0.0),
            (0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)),
        ),
        dtype=np.float64,
    )
    center = np.mean(corners, axis=0)
    points = np.concatenate((corners, (center + 0.15 * (corners[0] - center))[None]))
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    cells = np.column_stack((np.full((4,), 4, dtype=np.int32), faces))
    negative = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0.0
    cells[negative] = cells[negative][:, (1, 0, 2, 3)]
    source = CellMesh.from_tetrahedra(points, cells)
    edge_pairs = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
    determinant = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])
    squares = np.sum(
        (points[cells[:, edge_pairs[:, 1]]] - points[cells[:, edge_pairs[:, 0]]]) ** 2,
        axis=-1,
    )
    initial_quality = float(
        np.min(6.0 * (2.0 * determinant**2) ** (1.0 / 3.0) / np.sum(squares, axis=1))
    )
    with _metric_request(_METRIC_LIMITS) as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            source,
            np.broadcast_to(np.eye(3, dtype=np.float64), (5, 3, 3)),
            minimum_metric_quality=0.95,
            topology_operations=False,
            relocation=True,
            maximum_passes=1,
            scope=scope,
            budget=budget,
        )
    assert outcome.evidence.relocations == 1
    assert initial_quality < outcome.evidence.minimum_metric_quality < 0.95
    assert outcome.evidence.status is MetricRemeshingStatus.PASS_LIMIT
    np.testing.assert_array_equal(source.coordinates, points)

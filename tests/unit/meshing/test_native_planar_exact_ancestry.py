#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Planar rational ancestry and bounded binary64 carriers."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from fractions import Fraction
from time import monotonic

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import MeshcoreError, NativeExecutionScope
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellGeometrySpec, CellMesh, ExactPlcCellGeometrySource
from phydrax.discretization._cell_geometry_transfer import is_affine_cell_geometry
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import ConstrainedDelaunayTriangulation, SegmentMesh
from phydrax.meshing import MeshingScope, SizeControlStrength, UniformSizeControl
from phydrax.meshing._contracts import MeshingLimits, SurfaceMeshingSpec
from phydrax.meshing._plc_planar_support import cell_maps
from phydrax.meshing._sizing import SizeStatistic
from phydrax.meshing.providers._native_planar_domain import (
    constraint_coordinate_source,
    PreparedPlanarDomain,
    source_carrier_evidence,
    split_constraints,
    triangulate,
)
from phydrax.meshing.providers._native_sources import NativePlanarSource
from tests._support.native_provider_cases import plate, stage_report


@pytest.fixture
def coordinate_budget() -> Iterator[Callable[[int, int], CoordinateEnclosureBudget]]:
    with ExitStack() as resources:

        def create(work: int, memory: int) -> CoordinateEnclosureBudget:
            ticket = resources.enter_context(
                HostBatchTicket(work, queries_upper=work, bytes_upper=memory)
            )
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            resources.enter_context(budget.activate())
            return budget

        yield create


@pytest.fixture
def reference_budget(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> CoordinateEnclosureBudget:
    limits = MeshingLimits()
    return coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)


def _source(
    bound: float, *, budget: CoordinateEnclosureBudget
) -> ExactPlcCellGeometrySource:
    return ExactPlcCellGeometrySource(
        np.asarray(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0, 0, 0, 1), dtype=np.int64),
        np.asarray((-1, -1, -1, 0), dtype=np.int64),
        np.asarray(
            ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0 / 3.0, 0.0)), dtype=np.float64
        ),
        domain_source_id="planar",
        domain_source_revision="decimal",
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.asarray((7,), dtype=np.int64),
        source_segment_bounds=np.asarray((bound,), dtype=np.float64),
        vertex_parameter_ratios=(((0, 1), (0, 1)),) * 3 + (((1, 3), (0, 1)),),
        budget=budget,
    )


def test_non_rne_planar_carrier_retains_exact_rational_coordinate(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    prepared = source.prepare(carrier, budget=reference_budget)
    assert prepared.vertices[3] == (Fraction(1, 3), Fraction(1, 3))
    assert prepared.maximum_rounding_error > 0
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    assert geometry.source_coordinates(budget=reference_budget)[3] == prepared.vertices[3]
    assert is_affine_cell_geometry(mesh, geometry, budget=reference_budget)
    assert source.exact_vertex_parameters()[3] == (Fraction(1, 3), Fraction(0))
    with pytest.raises(
        ValueError, match=r"ExactPlcCellGeometrySource\.source_row_bounds"
    ):
        _source(0.0, budget=reference_budget).prepare(carrier, budget=reference_budget)
    with pytest.raises(
        ValueError, match=r"ExactPlcCellGeometrySource\.source_row_bounds"
    ):
        _source(1e-18, budget=reference_budget).prepare(carrier, budget=reference_budget)


def test_exact_witness_renewal_requires_explicit_subset_ratios(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    source = _source(1e-15, budget=reference_budget)
    unchanged = source.with_witnesses(
        source.vertex_strata,
        source.vertex_rows,
        source.vertex_parameters,
        budget=reference_budget,
    )
    assert unchanged.source_id == source.source_id
    with pytest.raises(
        ValueError, match=r"ExactPlcCellGeometrySource\.vertex_parameter_ratios"
    ):
        source.with_witnesses(
            source.vertex_strata[:3],
            source.vertex_rows[:3],
            source.vertex_parameters[:3],
            budget=reference_budget,
        )
    ratios = source.vertex_parameter_ratios
    assert ratios is not None
    subset = source.with_witnesses(
        source.vertex_strata[:3],
        source.vertex_rows[:3],
        source.vertex_parameters[:3],
        vertex_parameter_ratios=ratios[:3],
        budget=reference_budget,
    )
    assert subset.vertex_parameter_ratios == ratios[:3]


def test_exact_witness_cache_is_content_bound_and_request_scoped(
    reference_budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    budget = coordinate_budget(10000, 1 << 20)
    with budget.activate():
        first = source.prepare(carrier, budget=budget)
        work = budget.work_units
        assert work > 0
        assert source.prepare(carrier.copy(), budget=budget) is first
        assert budget.work_units == work
        moved = carrier.copy()
        moved[3, 0] = np.nextafter(moved[3, 0], 1.0)
        changed = source.prepare(moved, budget=budget)
        assert changed is not first
        assert changed.maximum_rounding_error > first.maximum_rounding_error
        assert budget.work_units > work
        with pytest.raises(
            ValueError, match=r"ExactPlcCellGeometrySource\.source_row_bounds"
        ):
            _source(0.0, budget=reference_budget).prepare(carrier, budget=budget)
    following = coordinate_budget(10000, 1 << 20)
    with following.activate():
        assert source.prepare(carrier, budget=following) is not first
        assert following.work_units == work


def test_constraint_split_preserves_source_identity_and_exact_dyadic_parameter() -> None:
    vertices = np.asarray(((0.2, 0.2), (0.5, 1.7)), dtype=np.float64)
    points, pieces, owners, point_sources, parameters = split_constraints(
        vertices,
        np.asarray(((0, 1),), dtype=np.int64),
        0.2,
        np.empty((0, 2, 2), dtype=np.float64),
        np.empty((0,), dtype=np.float64),
        float("inf"),
        MeshingLimits(),
    )
    assert np.array_equal(point_sources[:2], (-1, -1))
    assert np.all(point_sources[2:] == 0)
    assert np.all(owners == 0)
    assert pieces.shape[0] == points.shape[0] - 1
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in vertices)
    witnessed = [
        tuple(
            a + Fraction(float(t)) * (b - a)
            for a, b in zip(exact[0], exact[1], strict=True)
        )
        for t in parameters[2:]
    ]
    assert any(
        tuple(Fraction(float(value)) for value in point) != witness
        for point, witness in zip(points[2:], witnessed, strict=True)
    )
    assert all(0 < Fraction(float(t)) < 1 for t in parameters[2:])


def test_native_constrained_steiner_points_publish_composable_witnesses() -> None:
    triangulation = ConstrainedDelaunayTriangulation(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32),
        max_area=0.03,
        max_steiner=100,
    )
    constrained = np.flatnonzero(triangulation.constraint_sources >= 0)
    assert constrained.size > 0
    for vertex in constrained.tolist():
        parents = triangulation.constraint_parents[vertex]
        assert np.all(parents >= 0)
        assert np.all(parents < vertex)
        assert 0 < triangulation.constraint_ratios[vertex] < 1
    assert np.all(triangulation.constraint_sources[:4] == -1)


def test_planar_hole_carrier_evidence_names_each_source_bound_and_deviation(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    source, template = plate(
        limits=MeshingLimits(
            maximum_vertices=20_000,
            maximum_edges=160_000,
            maximum_faces=160_000,
            maximum_cells=160_000,
            maximum_connectivity_entries=1_280_000,
            maximum_data_bytes=20_480_000,
            maximum_work_units=640_000,
            maximum_cavity_cells=20_000,
            maximum_geometry_queries=1_280_000,
            maximum_scratch_bytes=81_920_000,
            maximum_wall_seconds=600.0,
        ),
    )
    # Preserve the registered tools/_meshing_cases.py planar-hole-feature
    # source revision and embedded-segment identity while sharing its geometry.
    # Source: tests/data/meshing/corpus.json, name=planar-hole-feature.
    revision = "fb13d4042f6bb9faa6581aa4399c848a8c4c899f88d621a6abcbecf3f4f54c2e"
    embedded = source.embedded
    if embedded is None:
        raise AssertionError(
            "fixture.plate.embedded must contain the protected segment; got None"
        )
    source = NativePlanarSource(
        source.region,
        revision,
        embedded=SegmentMesh(
            embedded.vertices,
            jnp.asarray(embedded.edges, dtype=jnp.int32),
            source_id="embedded-feature",
        ),
    )
    scope = MeshingScope(
        template.scope.source_id,
        revision,
        template.scope.entity_kind,
        template.scope.entity_dimension,
        template.scope.entity_set_id,
        template.scope.entity_ids,
    )
    request = SurfaceMeshingSpec(
        template.target,
        scope,
        planar_embedding=template.planar_embedding,
        size_controls=(
            UniformSizeControl(
                scope,
                0.2,
                maximum_size=1.5 * 0.2,
                strength=SizeControlStrength.HARD,
            ),
        ),
        size_compliance=template.size_compliance,
        quality_target=template.quality_target,
        limits=template.limits,
    )
    prepared = PreparedPlanarDomain(source, request)
    triangulation, _ = triangulate(prepared, monotonic(), request)
    used = np.unique(triangulation.triangles)
    authority, source_deviations, carrier_deviations = constraint_coordinate_source(
        source, prepared, triangulation, used, budget=reference_budget
    )
    vertex_ids = np.arange(used.size, dtype=np.int64)
    requested, achieved = source_carrier_evidence(
        prepared, authority, vertex_ids, carrier_deviations, source_deviations
    )
    bounds, deviations = dict(requested), dict(achieved)
    rows = np.asarray(authority.vertex_rows, dtype=np.int64)
    assert np.any(carrier_deviations > 0.0)
    for vertex in np.flatnonzero(rows >= 0).tolist():
        edge = int(rows[vertex])
        key = f"source:edge:{edge}:vertex:{vertex_ids[vertex]}"
        bound = bounds[f"{key}:maximum_deviation"]
        deviation = deviations[f"{key}:certified_deviation"]
        assert bound == prepared.source_bounds[edge]
        assert deviation == carrier_deviations[vertex]
        assert 0.0 <= deviation <= bound


def test_planar_cell_maps_reuse_proof_and_invalidate_changed_carriers(
    reference_budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    budget = coordinate_budget(10000, 1 << 20)
    with budget.activate():
        first = cell_maps(mesh, geometry, budget=budget)
        work = budget.work_units
        assert work > 0
        assert cell_maps(mesh, geometry, budget=budget) is first
        assert budget.work_units == work
        moved = carrier.copy()
        moved[3, 0] = np.nextafter(moved[3, 0], 1.0)
        successor = mesh.with_coordinates(moved, numeric_version="moved-carrier")
        new_geometry = geometry.with_coordinates(moved)
        assert cell_maps(successor, new_geometry, budget=budget) is not first
        assert budget.work_units > work


@pytest.mark.parametrize(
    ("statistics", "relative", "minimum"),
    (
        (("p50", "p95"), 1e-12, None),
        (("p50",), 1e-12, None),
        (("p50", "p95"), 0.5, None),
        (("p50",), 0.5, 0.9),
    ),
    ids=("tight-multiple", "tight-median", "multiple-statistics", "minimum-size"),
)
def test_uniform_planar_strategy_preserves_other_hard_size_contracts(
    statistics: tuple[SizeStatistic, ...], relative: float, minimum: float | None
) -> None:
    m = phx.meshing
    source = m.NativePlanarSource(
        phx.geometry.PlanarMeshRegion(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0))),
            ((0, 1, 2),),
            feature_id="equilateral",
        ),
        "r1",
    )
    scope = m.MeshingScope(
        source.source_id,
        "r1",
        m.MeshingEntityKind.GEOMETRY,
        2,
        "equilateral-region",
        np.asarray((0,), dtype=np.int64),
    )
    request = SurfaceMeshingSpec(
        m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            UniformSizeControl(
                scope,
                1.0,
                minimum_size=minimum,
                maximum_size=1.0,
                strength=SizeControlStrength.HARD,
            ),
        ),
        size_compliance=m.SizeCompliancePolicy(
            relative_tolerance=relative, target_statistics=statistics
        ),
    )
    result = (
        m.NativeMeshingProvider(m.NativeMeshingOptions("planar_constrained_delaunay"))
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    assert result.mesh.coordinates.shape[0] == 3
    assert result.mesh.blocks[0].vertices.shape[0] == 1
    assert result.compliance.passed
    assert result.certification is not None and result.certification.passed


@pytest.fixture
def thin_channel_seed_request() -> tuple[PreparedPlanarDomain, SurfaceMeshingSpec]:
    from tools._meshing_cases import meshing_case, prepare_case

    source, request, _ = prepare_case(
        meshing_case("planar-thin-channel"), 4, 20000, 120.0
    )
    return PreparedPlanarDomain(source, request), request


@pytest.mark.parametrize(
    ("work", "scratch", "expected_work", "expected_bytes"),
    ((100, 1_000_000, 675, None), (10_000, 4096, 0, 5296)),
    ids=("edge-wise-work", "scratch-before-allocation"),
)
def test_planar_distribution_refuses_before_filtering_or_site_allocation(
    thin_channel_seed_request: tuple[PreparedPlanarDomain, SurfaceMeshingSpec],
    monkeypatch: pytest.MonkeyPatch,
    work: int,
    scratch: int,
    expected_work: int,
    expected_bytes: int | None,
) -> None:
    from phydrax.meshing.providers import _native_planar_domain as owner

    prepared, template = thin_channel_seed_request
    request = SurfaceMeshingSpec(
        template.target,
        template.scope,
        planar_embedding=template.planar_embedding,
        size_controls=template.size_controls,
        size_compliance=template.size_compliance,
        quality_target=template.quality_target,
        limits=MeshingLimits(maximum_work_units=work, maximum_scratch_bytes=scratch),
    )

    def no_dispatch(*args: object, **kwargs: object) -> np.ndarray:
        pytest.fail("an inadmissible seed batch allocated or dispatched")

    monkeypatch.setattr(owner, "exact_orient2d", no_dispatch)
    monkeypatch.setattr(owner, "_segment_distances", no_dispatch)
    monkeypatch.setattr(NativeExecutionScope, "allocate_host_array", no_dispatch)
    with pytest.raises(MeshcoreError) as caught:
        owner._distribution_seeds(prepared, request)
    demand = caught.value.batch_requested_upper
    assert demand is not None
    assert demand[0] == expected_work
    if expected_bytes is not None:
        assert demand[3] == expected_bytes


def test_planar_distribution_chunks_and_charges_the_shared_work_owner(
    thin_channel_seed_request: tuple[PreparedPlanarDomain, SurfaceMeshingSpec],
) -> None:
    from phydrax.meshing.providers import _native_planar_domain as owner

    prepared, request = thin_channel_seed_request
    reference = owner._distribution_seeds(prepared, request)
    # The active parent, not only the request's declared limit, constrains the
    # chunk. This allowance fits one proposal plus retained output, not 27.
    limits = MeshingLimits(maximum_work_units=1000, maximum_scratch_bytes=8000)
    with NativeExecutionScope(limits.native_limits()) as scope:
        actual = owner._distribution_seeds(prepared, request)
        np.testing.assert_array_equal(actual, reference)
    evidence = scope.completed_evidence()
    assert evidence.externally_charged_work == 675
    assert evidence.work_counters.construction_work == 675
    assert evidence.host_storage_peak_bytes_upper <= 8000
    assert evidence.memory_counters.refusals == 0


@pytest.mark.parametrize(
    ("work", "scratch", "quantity", "demand"),
    ((100, 1_000_000, "work_units", 675), (10_000, 4096, "scratch_bytes", 5296)),
    ids=("work", "scratch"),
)
def test_planar_distribution_refusal_translates_to_meshing_failure(
    thin_channel_seed_request: tuple[PreparedPlanarDomain, SurfaceMeshingSpec],
    work: int,
    scratch: int,
    quantity: str,
    demand: int,
) -> None:
    prepared, template = thin_channel_seed_request
    request = SurfaceMeshingSpec(
        template.target,
        template.scope,
        planar_embedding=template.planar_embedding,
        size_controls=template.size_controls,
        size_compliance=template.size_compliance,
        quality_target=template.quality_target,
        limits=MeshingLimits(maximum_work_units=work, maximum_scratch_bytes=scratch),
    )
    with pytest.raises(phx.meshing.MeshingFailure) as caught:
        triangulate(prepared, monotonic(), request, distribution_sites=True)
    failure = caught.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.evidence.stage == "surface_meshing"
    assert dict(failure.evidence.requested)["maximum_work_units"] == work
    assert dict(failure.evidence.requested)["maximum_scratch_bytes"] == scratch
    assert dict(failure.evidence.achieved)[quantity] == demand
    assert isinstance(failure.__cause__, MeshcoreError)
    assert failure.__cause__.batch_refusal_evidence is not None


def test_compliant_planar_generation_retains_established_mesh_and_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.providers import _native_planar_domain as owner

    m = phx.meshing
    source = NativePlanarSource(
        phx.geometry.PlanarMeshRegion(
            np.asarray(
                (
                    (0, 0),
                    (2, 0),
                    (2, 2),
                    (0, 2),
                    (0.5, 0.5),
                    (0.5, 1.5),
                    (1.5, 1.5),
                    (1.5, 0.5),
                ),
                dtype=np.float64,
            ),
            ((0, 1, 2, 3), (4, 5, 6, 7)),
            feature_id="plate",
        ),
        "r1",
    )
    scope = MeshingScope(
        "plate",
        "r1",
        m.MeshingEntityKind.GEOMETRY,
        2,
        "plate-region",
        np.asarray((0,), dtype=np.int64),
    )
    request = SurfaceMeshingSpec(
        m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            UniformSizeControl(
                scope, 1.0, maximum_size=1.5, strength=SizeControlStrength.HARD
            ),
        ),
        size_compliance=m.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
    )
    prepared = PreparedPlanarDomain(source, request)
    reference, inserted = triangulate(prepared, monotonic(), request)
    assert inserted == 0 and reference.points.shape[0] == 12
    assert reference.triangles.shape[0] == 12

    def no_seeding(*args: object, **kwargs: object) -> np.ndarray:
        pytest.fail("an already-compliant construction must not be replaced")

    monkeypatch.setattr(owner, "_distribution_seeds", no_seeding)
    result = (
        m.NativeMeshingProvider(m.NativeMeshingOptions("planar_constrained_delaunay"))
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    assert result.mesh.coordinates.shape[0] == 12
    assert result.mesh.blocks[0].vertices.shape[0] == 12
    # Preserved-source reference: 12 unit boundary edges and 12 diagonal
    # interior edges of length sqrt(1/2), not a newly relabeled mesh.
    assert dict(result.compliance.achieved)[
        f"size:{request.size_controls[0].control_id}:p50_edge"
    ] == 0.5 * (1.0 + np.sqrt(0.5))
    inspection = stage_report(result, m.MeshingStageKind.SOURCE_INSPECTION)
    generation = stage_report(result, m.MeshingStageKind.SURFACE_MESHING)
    assert inspection.output_ids == (prepared.prepared_id,)
    assert generation.input_ids == inspection.output_ids
    assert generation.output_ids[0] == reference.evidence.evidence_id
    assert result.compliance.passed
    assert result.certification is not None and result.certification.passed


def test_planar_repair_native_consumer_spends_its_reserved_nested_allowance() -> None:
    from phydrax.meshing.providers import _native_planar_domain as owner
    from tools._meshing_cases import meshing_case, prepare_case

    source, request, _ = prepare_case(
        meshing_case("planar-thin-channel"), 4, 20000, 120.0
    )
    prepared = PreparedPlanarDomain(source, request)
    initial, _ = triangulate(prepared, monotonic(), request)
    initial_work = int(initial.work_evidence[0])
    del initial
    outer_limits = MeshingLimits(
        maximum_work_units=request.limits.maximum_work_units - 1,
        maximum_scratch_bytes=81_920_000,
    )
    with NativeExecutionScope(outer_limits.native_limits()) as parent:
        with NativeExecutionScope(request.limits.native_limits()) as child:
            with child.host_workspace() as workspace:
                selected, triangulation, inserted = owner.select_triangulation(
                    source, request, prepared, monotonic(), child, workspace
                )
        remaining = parent.remaining().remaining_work_units
    assert selected.prepared_id != prepared.prepared_id
    assert triangulation.points.shape[0] == 67
    assert triangulation.triangles.shape[0] == 87
    assert inserted == 18
    # Real native interior preparation must spend through its reservation: its
    # ticket reserves all root work, so an un-targeted native consumer refuses.
    # The two CDT attempts and the interior CDT debit the native work ledger;
    # only median classification, split preparation and lattice visits are host.
    host_work = 3978 + 612 + 675
    evidence = parent.completed_evidence()
    child_evidence = child.completed_evidence()
    assert evidence.externally_charged_work == host_work
    assert child_evidence.externally_charged_work == host_work
    assert (
        evidence.work_counters.construction_work
        == child_evidence.work_counters.construction_work
    )
    assert evidence.work_counters.construction_work > (
        host_work + initial_work + int(triangulation.work_evidence[0])
    )
    assert (
        remaining
        == outer_limits.maximum_work_units - evidence.work_counters.construction_work
    )
    assert evidence.work_counters.work_refusals == 0
    assert evidence.memory_counters.refusals == 0


def test_planar_repair_preparation_demand_refuses_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.providers import _native_planar_domain as owner
    from tools._meshing_cases import meshing_case, prepare_case

    source, request, _ = prepare_case(
        meshing_case("planar-thin-channel"), 4, 20000, 120.0
    )
    prepared = PreparedPlanarDomain(source, request)
    initial, _ = triangulate(prepared, monotonic(), request)
    initial_work = int(initial.work_evidence[0])
    del initial

    def no_preparation(*args: object, **kwargs: object) -> np.ndarray:
        pytest.fail("unadmitted balanced preparation dispatched")

    monkeypatch.setattr(owner, "interior_point", no_preparation)
    outer_limits = MeshingLimits(
        maximum_work_units=initial_work + 4200, maximum_scratch_bytes=81_920_000
    )
    parent = NativeExecutionScope(outer_limits.native_limits())
    with pytest.raises(phx.meshing.MeshingFailure) as caught, parent:
        with NativeExecutionScope(request.limits.native_limits()) as child:
            with child.host_workspace() as workspace:
                owner.select_triangulation(
                    source, request, prepared, monotonic(), child, workspace
                )
    failure = caught.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.evidence.stage == "surface_meshing"
    assert dict(failure.evidence.achieved)["work_units"] == 612
    assert isinstance(failure.__cause__, MeshcoreError)
    assert failure.__cause__.batch_requested_upper is not None
    assert failure.__cause__.batch_requested_upper[0] == 612
    evidence = parent.completed_evidence()
    assert evidence.externally_charged_work == 3978
    assert evidence.work_counters.construction_work == initial_work + 3978
    assert evidence.work_counters.work_refusals == 1


def test_planar_initial_native_work_refusal_retains_public_prefix() -> None:
    from tools._meshing_cases import meshing_case, prepare_case

    source, template, options = prepare_case(
        meshing_case("planar-thin-channel"), 4, 20000, 120.0
    )
    request = SurfaceMeshingSpec(
        template.target,
        template.scope,
        planar_embedding=template.planar_embedding,
        size_controls=template.size_controls,
        size_compliance=template.size_compliance,
        quality_target=template.quality_target,
        limits=MeshingLimits(maximum_work_units=1, maximum_scratch_bytes=81_920_000),
    )
    with pytest.raises(phx.meshing.MeshingFailure) as caught:
        phx.meshing.NativeMeshingProvider(options).plan(
            source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
        ).execute()
    assert caught.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(caught.value.evidence.requested)["maximum_work_units"] == 1
    prefix = dict(caught.value.evidence.achieved)
    assert prefix["native_planar:work_units"] == 1
    assert prefix["native_planar:work_refusals"] == 1
    cause = caught.value.__cause__
    while isinstance(cause, phx.meshing.MeshingFailure):
        cause = cause.__cause__
    assert isinstance(cause, MeshcoreError)

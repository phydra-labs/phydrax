#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Test selected CAD source fidelity across independent native chart chains."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as p
from phydrax._meshcore import MeshcoreError, NativeExecutionScope
from phydrax.discretization import CellGeometrySpec, CellMesh, reference_cell_topology
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.geometry import MeshingDomain, MeshingDomainBoundarySource
from phydrax.geometry._affine_chart_coverage import (
    _Preparation,
    prove_affine_chart_chain,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartCover
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing._domain import compile_surface_domain
from phydrax.meshing._surface_construction import SurfaceConstruction
from phydrax.meshing._surface_generation import generate_surface
from phydrax.meshing.providers._native_schedules import NativeSurfaceSchedule


_NativeBoxSource = tuple[MeshingDomain, SurfaceConstruction, CellMesh, CellGeometrySpec]

# Physical requests for this unit-box fixture, not numerical oracle tolerances.
_UNIT_BOX_FIDELITY_TOLERANCE = 1.0e-10
# Empirical upper-inflation cap retained from this module's rounded quadratic
# box regression; changing it would change that regression's acceptance bound.
_UNIT_BOX_PROXY_INFLATION_CAP = 1.0e-15


@pytest.fixture(scope="module")
def native_box_source() -> _NativeBoxSource:
    model = p.geometry.brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=p.SpatialCoordinateContract.si()
    )
    domain = MeshingDomain.from_brep(model)
    scope = p.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        p.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    specification = p.meshing.SurfaceMeshingSpec(
        p.meshing.CellMeshingTarget(
            2, 3, p.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            p.meshing.UniformSizeControl(
                scope,
                1.0,
                strength=p.meshing.SizeControlStrength.SOFT,
            ),
        ),
    )
    limits = specification.limits
    with NativeExecutionScope(limits.native_limits()) as execution:
        with execution.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
        ) as ticket:
            owner = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with owner.activate():
                compiled = compile_surface_domain(domain, specification, budget=owner)
        with owner.retained_storage(lambda count: execution.batch(0, bytes_upper=count)):
            construction = generate_surface(
                compiled,
                NativeSurfaceSchedule(),
                limits,
                0.0,
                scope=execution,
                budget=owner,
            )
    points = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    mesh = p.discretization.CellMesh(
        points,
        (
            p.discretization.CellBlock(
                "original-box",
                "hexahedron",
                np.arange(8)[None],
            ),
        ),
    )
    geometry = p.discretization.CellGeometrySpec.affine(mesh)
    return domain, construction, mesh, geometry


def test_scoped_original_source_fidelity_preserves_original_patch_identity(
    native_box_source: _NativeBoxSource,
) -> None:
    domain, construction, mesh, geometry = native_box_source
    charts = construction.chart_triangulations
    points = np.asarray(mesh.coordinates)
    connectivity = mesh.connectivity
    assert isinstance(connectivity, HexahedralConnectivity)
    faces = np.asarray(connectivity.faces)
    bottom_slot = int(np.flatnonzero(np.all(points[faces, 2] == 0.0, axis=1))[0])
    facets = np.asarray([mesh.entity_set(2).entity_ids[bottom_slot]], dtype=np.int64)
    centers = domain.evaluate(
        np.arange(len(domain.patches), dtype=np.int64),
        np.full((len(domain.patches), 2), 0.5),
    )
    bottom, top = int(np.argmin(centers[:, 2])), int(np.argmax(centers[:, 2]))
    original = MeshingDomainBoundarySource(domain, (bottom,), chart_triangulations=charts)
    wrong = MeshingDomainBoundarySource(domain, (top,), chart_triangulations=charts)
    positive = p.geometry.certify_source_fidelity(
        mesh,
        geometry,
        original,
        tolerance=_UNIT_BOX_FIDELITY_TOLERANCE,
        target_facet_ids=facets,
    )
    negative = p.geometry.certify_source_fidelity(
        mesh,
        geometry,
        wrong,
        tolerance=_UNIT_BOX_FIDELITY_TOLERANCE,
        target_facet_ids=facets,
    )
    assert positive.status == "certified"
    assert 0.0 < positive.mesh_to_source_upper <= positive.tolerance
    assert 0.0 < positive.source_to_mesh_upper <= positive.tolerance
    assert negative.status == "violated"
    assert negative.mesh_to_source_lower > negative.tolerance
    assert positive.source_scope_id != negative.source_scope_id
    assert (
        positive.target_facet_ids == negative.target_facet_ids == tuple(facets.tolist())
    )
    missing = MeshingDomainBoundarySource(
        domain,
        (bottom,),
        chart_triangulations=tuple(chart for chart in charts if chart[0] != bottom),
    )
    assert not missing.boundary_chart_cover(128).complete
    duplicated = MeshingDomainBoundarySource(
        domain,
        (bottom,),
        chart_triangulations=(
            *charts,
            next(chart for chart in charts if chart[0] == bottom),
        ),
    )
    assert not duplicated.boundary_chart_cover(128).complete


def test_full_original_facet_fidelity_prepared_cold_and_one_under_caps(
    native_box_source: _NativeBoxSource,
) -> None:
    domain, construction, mesh, geometry = native_box_source
    charts = construction.chart_triangulations
    original = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches))), chart_triangulations=charts
    )
    facets = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    before = np.array(mesh.coordinates, copy=True)
    cold = p.geometry.certify_source_fidelity(
        mesh,
        geometry,
        original,
        tolerance=_UNIT_BOX_FIDELITY_TOLERANCE,
        target_facet_ids=facets,
    )
    warm = p.geometry.certify_source_fidelity(
        mesh,
        geometry,
        original,
        tolerance=_UNIT_BOX_FIDELITY_TOLERANCE,
        target_facet_ids=facets,
    )
    assert cold.status == warm.status == "certified"
    assert cold.binding.binding_id == warm.binding.binding_id
    assert cold.source_scope_id == warm.source_scope_id
    assert cold.mesh_to_source_upper == warm.mesh_to_source_upper
    assert cold.source_to_mesh_upper == warm.source_to_mesh_upper
    assert cold.chart_coverage is not None
    target_owner, = {
        row.owner_id for row in cold.chart_coverage.affine_chain_resources
        if row.role == "target"
    }
    counts = {
        row.metric: row.used for row in cold.chart_coverage.affine_chain_resources
        if row.role == "target" and row.owner_id == target_owner
    }
    assert (
        counts["total_work"] >= counts["coefficient_work"] > 0
    )
    assert counts["native_peak_bytes"] == 0
    assert counts["native_clip_work"] == 0
    for limits in (
        p.geometry.MeshCertificateLimits(
            maximum_work_units=counts["total_work"] - 1
        ),
        p.geometry.MeshCertificateLimits(
            maximum_scratch_bytes=counts["coefficient_bytes_upper"] - 1
        ),
    ):
        refused = p.geometry.certify_source_fidelity(
            mesh,
            geometry,
            original,
            tolerance=_UNIT_BOX_FIDELITY_TOLERANCE,
            target_facet_ids=facets,
            limits=limits,
        )
        assert refused.status == "unresolved"
        assert any(
            finding.check == "source_affine_chain_resource_budget"
            for finding in refused.findings
        )
        assert refused.target_facet_ids == cold.target_facet_ids
        assert refused.source_scope_id == cold.source_scope_id
    np.testing.assert_array_equal(mesh.coordinates, before)


def test_rounded_quadratic_box_charges_certified_proxy_deviation(
    native_box_source: _NativeBoxSource,
) -> None:

    domain, construction, _, _ = native_box_source
    # Two hexahedral layers split at z = 0.2: binary64 midpoints such as
    # (0.2 + 1) / 2 are rounded, so each quadratic face map is only certified
    # to lie within a nonzero deviation of its affine proxy.
    unit = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    base = unit[unit[:, 2] == 0.0]
    points = np.concatenate([base + (0.0, 0.0, z) for z in (0.0, 0.2, 1.0)])
    vertices = np.asarray([[*range(0, 8)], [*range(4, 12)]], dtype=np.int32)
    mesh = p.discretization.CellMesh(
        points, (p.discretization.CellBlock("layers", "hexahedron", vertices),)
    )
    geometry = straight_geometry(mesh, 2)
    source = MeshingDomainBoundarySource(
        domain,
        tuple(range(len(domain.patches))),
        resolution=4,
        chart_triangulations=construction.chart_triangulations,
    )
    cover = source.boundary_chart_cover(
        p.geometry.MeshCertificateLimits().maximum_source_samples
    )
    cover_bound = float(np.max(cover.deviation_bounds))
    certified = p.geometry.certify_source_fidelity(
        mesh, geometry, source, tolerance=_UNIT_BOX_FIDELITY_TOLERANCE
    )
    assert certified.status == "certified"
    assert certified.mesh_to_source_semantics == "certified"
    assert certified.source_to_mesh_semantics == "certified"
    assert certified.mesh_to_source_upper == certified.source_to_mesh_upper
    assert (
        cover_bound
        < certified.mesh_to_source_upper
        < cover_bound + _UNIT_BOX_PROXY_INFLATION_CAP
    )
    # Moving one face node off the x = 0 plane is charged through the same
    # proxy deviation, so the bowed map cannot certify within tolerance.
    coordinates = np.array(geometry.coordinates, copy=True)
    node = int(
        np.flatnonzero(
            (coordinates[:, 0] == 0.0)
            & (np.arange(coordinates.shape[0]) >= points.shape[0])
            & (coordinates[:, 2] > 0.2)
        )[0]
    )
    coordinates[node, 0] = -1e-6
    bowed = p.geometry.certify_source_fidelity(
        mesh,
        geometry.with_coordinates(coordinates),
        source,
        tolerance=_UNIT_BOX_FIDELITY_TOLERANCE,
    )
    assert bowed.status != "certified"
    assert bowed.mesh_to_source_upper > bowed.tolerance


@pytest.mark.parametrize("reverse", (False, True), ids=("forward", "reversed"))
@pytest.mark.parametrize("change", ("none", "gap", "overlap"))
def test_independent_affine_tilings_keep_exact_contact_and_refusal_decisions(
    reverse: bool, change: str
) -> None:
    points = np.asarray(
        ((0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 2), (0.5, 0.5, 2)),
        dtype=np.float64,
    )
    source_rows = points[np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)]
    target_rows = points[
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int64)
    ]
    if reverse:
        source_rows = source_rows[:, ::-1]
        target_rows = target_rows[:, ::-1]
    if change == "gap":
        target_rows = target_rows[:-1]
    elif change == "overlap":
        target_rows = np.concatenate((target_rows, target_rows[:1]), axis=0)
    cover = SourceBoundaryChartCover(
        source_rows,
        np.zeros(source_rows.shape[0], dtype=np.float64),
        "certified",
        True,
        "independent-square",
        "original-square",
    )
    proxies = [(triangle, 0.0, 0.0) for triangle in target_rows]
    bound, proved, separation = prove_affine_chart_chain(
        proxies, cover, p.geometry.MeshCertificateLimits(), 0.0,
        source_identity=(cover.source_id, cover.source_revision),
    )
    assert bound == (0.0 if change == "none" else None)
    assert separation is None
    assert proved.source_id == cover.source_id
    assert proved.source_revision == cover.source_revision
    np.testing.assert_array_equal(proved.simplices, cover.simplices)
    counts = {
        row.metric: row.used for row in proved.affine_chain_resources
        if row.role == "target" and row.owner_id == cover.cover_id
    }
    assert counts["total_work"] > 0
    if change == "none":
        with pytest.raises((CoordinateEnclosureResourceError, MeshcoreError)):
            prove_affine_chart_chain(
                proxies,
                cover,
                p.geometry.MeshCertificateLimits(
                    maximum_work_units=counts["total_work"] - 1
                ),
                0.0,
                source_identity=(cover.source_id, cover.source_revision),
            )


def test_independent_affine_grid_tiles_under_inherited_request_allowance() -> None:
    subdivisions = 4
    source_rows = np.asarray(
        (
            ((0, 0, 2), (1, 0, 2), (1, 1, 2)),
            ((0, 0, 2), (1, 1, 2), (0, 1, 2)),
        ),
        dtype=np.float64,
    )
    target_rows = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            a = (i / subdivisions, j / subdivisions, 2.0)
            b = ((i + 1) / subdivisions, j / subdivisions, 2.0)
            c = ((i + 1) / subdivisions, (j + 1) / subdivisions, 2.0)
            d = (i / subdivisions, (j + 1) / subdivisions, 2.0)
            target_rows.extend((np.asarray((a, b, d)), np.asarray((b, c, d))))
    cover = SourceBoundaryChartCover(
        source_rows,
        np.zeros(2, dtype=np.float64),
        "certified",
        True,
        "independent-square",
        "original-square",
    )
    # The lifecycle's capacity-20000 request declares 32 work units per slot.
    limits = p.geometry.MeshCertificateLimits(maximum_work_units=640_000)
    parent_work = 17
    with NativeExecutionScope(limits.native_limits()) as parent:
        parent.charge(work=parent_work)
        initial = parent.remaining().remaining_work_units
        bound, proved, separation = prove_affine_chart_chain(
            [(triangle, 0.0, 0.0) for triangle in target_rows], cover, limits, 0.0,
            source_identity=(cover.source_id, cover.source_revision),
        )
        consumed = initial - parent.remaining().remaining_work_units
    counts = {
        row.metric: row.used for row in proved.affine_chain_resources
        if row.role == "target" and row.owner_id == cover.cover_id
    }
    assert bound == 0.0
    assert separation is None
    assert proved.complete
    assert proved.source_id == cover.source_id
    assert proved.source_revision == cover.source_revision
    np.testing.assert_array_equal(proved.simplices, source_rows)
    assert 0 < counts["total_work"] == consumed
    assert counts["total_work"] <= limits.maximum_work_units - parent_work
    assert parent.evidence is not None
    assert parent.evidence.work_counters.construction_work == parent_work + consumed


def test_request_proved_chain_areas_reuse_exact_pairs_without_changing_decisions() -> None:
    points = np.asarray(
        ((0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 2), (0.5, 0.5, 2)),
        dtype=np.float64,
    )
    source_rows = points[np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)]
    target_rows = points[
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int64)
    ]
    cover = SourceBoundaryChartCover(
        source_rows,
        np.zeros(source_rows.shape[0], dtype=np.float64),
        "certified",
        True,
        "independent-square",
        "original-square",
    )
    proxies = [(triangle, 0.0, 0.0) for triangle in target_rows]
    limits = p.geometry.MeshCertificateLimits()
    earned: dict = {}
    first, first_cover, first_separation = prove_affine_chart_chain(
        proxies, cover, limits, 0.0, earned_areas=earned,
        source_identity=(cover.source_id, cover.source_revision),
    )
    assert first == 0.0 and first_separation is None
    assert earned and all(area >= 0 for area in earned.values())
    replayed: dict = {}
    second, second_cover, second_separation = prove_affine_chart_chain(
        proxies, cover, limits, 0.0, areas=earned, earned_areas=replayed,
        source_identity=(cover.source_id, cover.source_revision),
    )
    assert (second, second_separation) == (first, first_separation)
    assert not replayed
    work = {
        name: {
            row.metric: row.used for row in proved.affine_chain_resources
            if row.role == "target" and row.owner_id == cover.cover_id
        }
        for name, proved in (("first", first_cover), ("second", second_cover))
    }
    assert work["second"]["exact_clip_calls"] == 0
    assert work["second"]["total_work"] < work["first"]["total_work"]
    # Reused areas never mask a changed tiling: the gap still refuses.
    gap, _, _ = prove_affine_chart_chain(
        proxies[:-1], cover, limits, 0.0, areas=earned,
        source_identity=(cover.source_id, cover.source_revision),
    )
    assert gap is None


def test_affine_plane_profile_reuse_preserves_first_arithmetic() -> None:
    from phydrax.geometry._affine_chart_coverage import _plane
    from tests._support.coordinate_references import coordinate_reference_budget

    points = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(3), Fraction(0)),
    )
    results = []
    charges = []
    for prepared in (False, True):
        with coordinate_reference_budget() as budget, budget.live_storage() as cache, budget.live_storage() as sides:
            preparation = _Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            for point in points:
                preparation.prepare_points(point)
            if prepared:
                preparation.prepare_points(points)
            before = budget.work_units
            results.append(_plane(points, preparation))
            charges.append(budget.work_units - before)
            before = budget.work_units
            assert _plane(points, preparation) == results[-1]
            assert budget.work_units - before == charges[-1]
    assert results[0] == results[1]
    assert charges[0] - charges[1] == 2


def test_affine_vertex_preparation_reuses_full_binary64_content() -> None:
    limits = p.geometry.MeshCertificateLimits(
        maximum_work_units=2000, maximum_scratch_bytes=128_000
    )
    row = np.asarray((0.1, 0.25, 0.5), dtype=np.float64)
    with NativeExecutionScope(limits.native_limits()) as native:
        ticket = native.batch(
            limits.maximum_work_units, bytes_upper=limits.maximum_scratch_bytes
        )
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with (
            ticket,
            ticket.native_target(),
            budget.activate(),
            budget.live_storage() as cache_storage,
            budget.live_storage() as side_storage,
        ):
            preparation = _Preparation(
                budget=budget,
                cache_storage=cache_storage,
                side_storage=side_storage,
            )
            point = preparation.vertex(row, 64)
            assert point == (Fraction.from_float(0.1), Fraction(1, 4), Fraction(1, 2))
            before = budget.work_units
            repeated = preparation.vertex(np.array(row, copy=True), 64)
            assert repeated == point
            assert budget.work_units - before == row.size
            row[0] = 0.2
            changed = preparation.vertex(row, 64)
            assert changed == (Fraction.from_float(0.2), Fraction(1, 4), Fraction(1, 2))
            assert point[0] == Fraction.from_float(0.1)


@pytest.mark.parametrize("change", ("none", "duplicate", "owner", "revision", "sampled"))
def test_exact_chain_identity_is_local_to_authenticated_scope(change: str) -> None:
    square = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    rows = square[np.asarray(((0, 1, 2), (0, 2, 3)))]
    if change == "duplicate":
        rows = np.asarray((rows[0], rows[0]))
    cover = SourceBoundaryChartCover(
        rows, np.zeros(2), "sampled" if change == "sampled" else "certified",
        True, "scoped-material-a", "revision-a", restriction_ids=("restriction-a",),
    )
    proxies = [(row[::-1].copy(), 0.0, 0.0) for row in reversed(rows)]
    identity = (
        "scoped-material-b" if change == "owner" else cover.source_id,
        "revision-b" if change == "revision" else cover.source_revision,
    )
    if change in ("owner", "revision"):
        with pytest.raises(ValueError, match="authenticated scoped source owner"):
            prove_affine_chart_chain(
                proxies, cover, p.geometry.MeshCertificateLimits(), 0.0,
                source_identity=identity,
            )
        return
    bound, proved, separation = prove_affine_chart_chain(
        proxies, cover, p.geometry.MeshCertificateLimits(), 0.0,
        source_identity=identity,
    )
    assert bound == (0.0 if change == "none" else None)
    assert separation is None
    assert proved.restriction_ids == cover.restriction_ids
    if change == "none":
        counts = {
            row.metric: row.used for row in proved.affine_chain_resources
            if row.role == "target" and row.owner_id == cover.cover_id
        }
        assert counts["identity_rows"] == 2
        assert counts["identity_key_bytes_upper"] > 0
        assert counts["total_work"] > 0


def test_exact_chain_identity_preserves_candidate_capacity() -> None:
    rows = np.asarray((
        ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        ((0, 0, 0), (1, 1, 0), (0, 1, 0)),
    ), dtype=float)
    cover = SourceBoundaryChartCover(
        rows, np.zeros(2), "certified", True, "scoped-material", "revision",
    )
    bound, proved, separation = prove_affine_chart_chain(
        [(row, 0.0, 0.0) for row in rows], cover,
        p.geometry.MeshCertificateLimits(maximum_candidate_pairs=1), 0.0,
        source_identity=(cover.source_id, cover.source_revision),
    )
    assert bound is None and separation is None
    assert proved is cover


def test_completed_source_topology_import_matches_fresh_and_exact_work_cap(
    native_box_source: _NativeBoxSource, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import copy

    from phydrax.geometry import _affine_chart_coverage as affine
    from phydrax.geometry._domain_boundary import _authenticated_chart_topology
    from tests._support.coordinate_references import coordinate_reference_budget

    domain, construction, _, _ = native_box_source
    source = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches))),
        chart_triangulations=construction.chart_triangulations,
    )
    limits = p.geometry.MeshCertificateLimits()
    cover = source.boundary_chart_cover(limits.maximum_source_samples)
    assert cover.complete
    raw_cover = copy(cover)
    with coordinate_reference_budget() as budget:
        topology = _authenticated_chart_topology(cover, budget)
        assert topology is not None
        assert _authenticated_chart_topology(raw_cover, budget) is None
    # A subsequent cover still earns its own source proof; only numeric
    # connectivity survives the completed ancestor request.
    next_cover = source.boundary_chart_cover(limits.maximum_source_samples)
    with coordinate_reference_budget() as budget:
        assert _authenticated_chart_topology(next_cover, budget) is topology
    proxies = [(row[::-1].copy(), 0.0, 0.0) for row in reversed(cover.simplices)]
    mapping_results = []
    key_imports = []
    mapping = affine._chain_identity_mapping

    def observe_mapping(*args: Any, **kwargs: Any) -> tuple[int, ...] | None:
        key_imports.append(kwargs.get("source_keys") is not None)
        result = mapping(*args, **kwargs)
        mapping_results.append(result)
        return result

    monkeypatch.setattr(affine, "_chain_identity_mapping", observe_mapping)
    results = [
        prove_affine_chart_chain(
            proxies, candidate, limits, 1.0,
            source_identity=(source.source_id, source.source_revision),
        )
        for candidate in (raw_cover, next_cover)
    ]
    assert key_imports == [False, True]
    assert mapping_results[0] == mapping_results[1]
    raw, prepared = results
    assert raw[0] == prepared[0]
    assert raw[2] == prepared[2]
    assert raw[0] is not None
    for field in ("complete", "semantics", "restriction_ids", "findings"):
        assert getattr(raw[1], field) == getattr(prepared[1], field)
    counts = [
        {
            row.metric: row.used for row in result[1].affine_chain_resources
            if row.role == "target" and row.owner_id == candidate.cover_id
        }
        for result, candidate in zip(results, (raw_cover, next_cover), strict=True)
    ]
    for name in (
        "embedding_obligations", "embedding_components",
        "embedding_holes", "embedding_boundary_edges",
    ):
        assert counts[0][name] == counts[1][name]
    work = counts[1]["total_work"]
    assert work < counts[0]["total_work"]
    # Dry admission may reserve a larger arithmetic bound than the completed
    # work it ultimately charges; the exact cap must cover that requirement.
    work = int(counts[1]["required_work"])
    exact = prove_affine_chart_chain(
        proxies, next_cover, p.geometry.MeshCertificateLimits(maximum_work_units=work), 1.0,
        source_identity=(source.source_id, source.source_revision),
    )
    assert exact[0] == prepared[0]
    with pytest.raises((CoordinateEnclosureResourceError, MeshcoreError)):
        prove_affine_chart_chain(
            proxies, next_cover,
            p.geometry.MeshCertificateLimits(maximum_work_units=work - 1), 1.0,
            source_identity=(source.source_id, source.source_revision),
        )


@pytest.mark.parametrize(
    "change",
    ("source-owner", "source-bytes", "source-order", "source-metadata",
     "cover-owner", "cover-bytes", "cover-order", "cover-metadata"),
)
def test_source_topology_issue_reauthenticates_live_arrays(
    native_box_source: _NativeBoxSource, change: str,
) -> None:
    from phydrax.geometry._domain_boundary import _authenticated_chart_topology
    from tests._support.coordinate_references import coordinate_reference_budget

    domain, construction, _, _ = native_box_source
    source = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches))),
        chart_triangulations=construction.chart_triangulations,
    )
    cover = source.boundary_chart_cover(p.geometry.MeshCertificateLimits().maximum_source_samples)
    with coordinate_reference_budget() as budget:
        assert _authenticated_chart_topology(cover, budget) is not None
    if change == "source-metadata":
        object.__setattr__(source, "resolution", source.resolution + 1)
    elif change == "cover-metadata":
        object.__setattr__(cover, "restriction_ids", ("foreign-restriction",))
    elif change == "source-order":
        object.__setattr__(source, "chart_triangulations", tuple(reversed(source.chart_triangulations)))
    elif change == "cover-order":
        object.__setattr__(cover, "simplices", cover.simplices[::-1])
    elif change.startswith("source"):
        records = list(source.chart_triangulations)
        record = records[0]
        points = np.array(record[2], copy=True)
        if change == "source-bytes":
            points[0, 0] += 0.125
        points.setflags(write=False)
        records[0] = (
            record[0], record[1], points, record[3], record[4], record[5],
            record[6], record[7], record[8], record[9],
        )
        object.__setattr__(source, "chart_triangulations", tuple(records))
    else:
        simplices = np.array(cover.simplices, copy=True)
        if change == "cover-bytes":
            simplices[0, 0, 0] += 0.125
        simplices.setflags(write=False)
        object.__setattr__(cover, "simplices", simplices)
    with coordinate_reference_budget() as budget:
        assert _authenticated_chart_topology(cover, budget) is None


def test_source_topology_archive_and_clone_reissue_only_after_cold_proof(
    native_box_source: _NativeBoxSource,
) -> None:
    from copy import copy

    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.geometry._domain_boundary import (
        _authenticated_chart_topology,
        _SOURCE_TOPOLOGY,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
    from tests._support.coordinate_references import coordinate_reference_budget

    domain, construction, _, _ = native_box_source
    source = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches))),
        chart_triangulations=construction.chart_triangulations,
    )
    limits = p.geometry.MeshCertificateLimits()
    cover = source.boundary_chart_cover(limits.maximum_source_samples)
    with coordinate_reference_budget() as budget:
        original = _authenticated_chart_topology(cover, budget)
    assert original is not None
    register_meshing_source_artifacts()
    recipe = model_structure_recipe((source, cover))
    archived = model_from_array_recipe(
        recipe, model_recipe_array_values((source, cover), recipe, prefix="topology"),
        prefix="topology",
    )
    restored_source, restored_cover = archived
    assert isinstance(restored_source, MeshingDomainBoundarySource)
    assert isinstance(restored_cover, SourceBoundaryChartCover)
    with coordinate_reference_budget() as budget:
        assert _authenticated_chart_topology(restored_cover, budget) is None
    for candidate in (copy(source), restored_source):
        assert id(candidate) not in _SOURCE_TOPOLOGY
        cold_cover = candidate.boundary_chart_cover(limits.maximum_source_samples)
        assert cold_cover.complete
        with coordinate_reference_budget() as budget:
            cold = _authenticated_chart_topology(cold_cover, budget)
        assert cold is not None and cold is not original
        np.testing.assert_array_equal(cold_cover.simplices, cover.simplices)


def test_source_topology_issuers_follow_source_and_cover_alias_lifetimes(
    native_box_source: _NativeBoxSource,
) -> None:
    import gc
    import weakref

    from phydrax.geometry._domain_boundary import (
        _authenticated_chart_topology,
        _COVER_TOPOLOGY,
        _SOURCE_TOPOLOGY,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    domain, construction, _, _ = native_box_source
    source = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches))),
        chart_triangulations=construction.chart_triangulations,
    )
    cover = source.boundary_chart_cover(p.geometry.MeshCertificateLimits().maximum_source_samples)
    source_id, cover_id = id(source), id(cover)
    source_ref, cover_ref = weakref.ref(source), weakref.ref(cover)
    issue_ref = weakref.ref(_SOURCE_TOPOLOGY[source_id])
    bank_ref = weakref.ref(_SOURCE_TOPOLOGY[source_id].topology.planes[0].rows)
    source_alias, cover_alias = source, cover
    del source, cover
    gc.collect()
    assert source_ref() is source_alias and cover_ref() is cover_alias
    assert issue_ref() is not None and bank_ref() is not None
    del source_alias
    gc.collect()
    assert source_ref() is None and source_id not in _SOURCE_TOPOLOGY
    # Keeping the cover alive must not keep a dead source's schedule alive.
    assert issue_ref() is None and bank_ref() is None
    with coordinate_reference_budget() as budget:
        assert _authenticated_chart_topology(cover_alias, budget) is None
    del cover_alias
    gc.collect()
    assert cover_ref() is None and cover_id not in _COVER_TOPOLOGY


@pytest.mark.parametrize("stage", ("binding", "exporter"))
def test_source_topology_failed_issue_publishes_no_partial_owner(
    native_box_source: _NativeBoxSource, stage: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import copy

    from phydrax.geometry import (
        _affine_chart_coverage as affine,
        _domain_boundary as issuer,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    domain, construction, _, _ = native_box_source
    source = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches))),
        chart_triangulations=construction.chart_triangulations,
    )
    original = source.boundary_chart_cover(p.geometry.MeshCertificateLimits().maximum_source_samples)
    source, cover = copy(source), copy(original)
    definition = issuer._topology_definition

    def fail_binding(
        value: MeshingDomainBoundarySource | SourceBoundaryChartCover,
        budget: CoordinateEnclosureBudget,
    ) -> str:
        if value is cover:
            raise ValueError("cover binding refused")
        return definition(value, budget)

    def fail_exporter(*args: object, **kwargs: object) -> None:
        raise ValueError("source exporter refused")

    if stage == "binding":
        monkeypatch.setattr(issuer, "_topology_definition", fail_binding)
    else:
        monkeypatch.setattr(affine, "_prepare_source_planar_topology", fail_exporter)
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError, match="refused"):
            issuer._issue_chart_topology(source, cover, budget)
        assert id(source) not in issuer._SOURCE_TOPOLOGY
        assert id(cover) not in issuer._COVER_TOPOLOGY
        assert not budget.retained_owner_objects

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Test selected CAD source fidelity across independent native chart chains."""

from fractions import Fraction

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
    counts = {name: used for name, used, _ in cold.chart_coverage.resource_counts}
    assert (
        counts["affine_chain_total_work"] >= counts["affine_chain_coefficient_work"] > 0
    )
    assert counts["affine_chain_native_peak_bytes"] > 0
    assert counts["affine_chain_native_clip_work"] > 0
    for limits in (
        p.geometry.MeshCertificateLimits(
            maximum_work_units=counts["affine_chain_total_work"] - 1
        ),
        p.geometry.MeshCertificateLimits(
            maximum_scratch_bytes=counts["affine_chain_coefficient_bytes_upper"] - 1
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
        proxies, cover, p.geometry.MeshCertificateLimits(), 0.0
    )
    assert bound == (0.0 if change == "none" else None)
    assert separation is None
    assert proved.source_id == cover.source_id
    assert proved.source_revision == cover.source_revision
    np.testing.assert_array_equal(proved.simplices, cover.simplices)
    counts = {name: used for name, used, _ in proved.resource_counts}
    assert counts["affine_chain_total_work"] > 0
    if change == "none":
        with pytest.raises((CoordinateEnclosureResourceError, MeshcoreError)):
            prove_affine_chart_chain(
                proxies,
                cover,
                p.geometry.MeshCertificateLimits(
                    maximum_work_units=counts["affine_chain_total_work"] - 1
                ),
                0.0,
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
            [(triangle, 0.0, 0.0) for triangle in target_rows], cover, limits, 0.0
        )
        consumed = initial - parent.remaining().remaining_work_units
    counts = {name: used for name, used, _ in proved.resource_counts}
    assert bound == 0.0
    assert separation is None
    assert proved.complete
    assert proved.source_id == cover.source_id
    assert proved.source_revision == cover.source_revision
    np.testing.assert_array_equal(proved.simplices, source_rows)
    assert 0 < counts["affine_chain_total_work"] == consumed
    assert counts["affine_chain_total_work"] <= limits.maximum_work_units - parent_work
    assert parent.evidence is not None
    assert parent.evidence.work_counters.construction_work == parent_work + consumed


def test_affine_exact_actions_reuse_full_inputs_and_preserve_singular_status() -> None:
    limits = p.geometry.MeshCertificateLimits(
        maximum_work_units=2000, maximum_scratch_bytes=128_000
    )
    matrix = ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(3)))
    right = ((Fraction(1),), (Fraction(1),))
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
                ticket,
                budget=budget,
                cache_storage=cache_storage,
                side_storage=side_storage,
            )
            solved = preparation.exact_actions(matrix, right)
            assert solved.actions == ((Fraction(1, 2),), (Fraction(1, 3),))
            assert solved.determinant == 6
            assert solved.rank == 2
            assert solved.status == "exact"
            before = budget.work_units
            repeated = preparation.exact_actions(
                tuple(tuple(Fraction(value) for value in row) for row in matrix),
                tuple(tuple(Fraction(value) for value in row) for row in right),
            )
            assert repeated == solved
            assert budget.work_units - before == 1
            changed = preparation.exact_actions(matrix, ((Fraction(1),), (Fraction(2),)))
            assert changed.actions == ((Fraction(1, 2),), (Fraction(2, 3),))
            assert changed.input_id != solved.input_id
            singular = preparation.exact_actions(
                ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(0))), right
            )
            assert singular.actions is None
            assert singular.rank == 1
            assert singular.status == "singular"


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
                ticket,
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Original selected CAD source fidelity across independent native chart chains."""

import numpy as np
import pytest

import phydrax as p
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.discretization._hexahedral import HexahedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._meshing_domain import MeshingDomain, MeshingDomainBoundarySource
from phydrax.meshing._domain import compile_surface_domain
from phydrax.meshing._surface_generation import generate_surface, SurfaceConstruction
from phydrax.meshing.providers._native_options import NativeSurfaceSchedule


_NativeBoxSource = tuple[MeshingDomain, SurfaceConstruction, CellMesh, CellGeometrySpec]


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
    construction = generate_surface(
        compile_surface_domain(domain, specification),
        NativeSurfaceSchedule(),
        specification.limits,
        0.0,
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
        mesh, geometry, original, tolerance=1e-10, target_facet_ids=facets
    )
    negative = p.geometry.certify_source_fidelity(
        mesh, geometry, wrong, tolerance=1e-10, target_facet_ids=facets
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
        mesh, geometry, original, tolerance=1e-10, target_facet_ids=facets
    )
    warm = p.geometry.certify_source_fidelity(
        mesh, geometry, original, tolerance=1e-10, target_facet_ids=facets
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
            tolerance=1e-10,
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
    from phydrax.meshing._curving import _straight_geometry

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
    geometry = _straight_geometry(mesh, 2)
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
        mesh, geometry, source, tolerance=1e-10
    )
    assert certified.status == "certified"
    assert certified.mesh_to_source_semantics == "certified"
    assert certified.source_to_mesh_semantics == "certified"
    assert certified.mesh_to_source_upper == certified.source_to_mesh_upper
    assert cover_bound < certified.mesh_to_source_upper < cover_bound + 1e-15
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
        mesh, geometry.with_coordinates(coordinates), source, tolerance=1e-10
    )
    assert bowed.status != "certified"
    assert bowed.mesh_to_source_upper > bowed.tolerance

import itertools
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh, PolyhedralConnectivity
from phydrax.discretization.finite_volume._polyhedral import (
    prepare_polyhedral_finite_volume_geometry,
)
from phydrax.discretization.vem._polyhedral import (
    prepare_polyhedral_h1_virtual_element_3d,
)
from phydrax.geometry._supermesh import CommonRefinementStatus, prepare_common_refinement
from phydrax.geometry._triangulation import RestrictedPowerDiagram
from phydrax.meshing import CellMeshingResult, MeshingScope, PiecewiseLinearComplex
from phydrax.meshing._controls import PatchControl, PeriodicConstraint
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
    PolyhedralGenerationError,
    prepare_periodic_polyhedral_seam_fragments,
)
from phydrax.meshing.providers._native_polyhedral import (
    execute_polyhedral_route,
    PreparedPolyhedralVolume,
)


M = phx.meshing
_LOOPS = (
    (0, 2, 3, 1),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 6, 7, 3),
    (0, 4, 6, 2),
    (1, 3, 7, 5),
)


def _voxel_plc(
    cubes: Sequence[tuple[int, int, int]],
    regions: Sequence[int] | None = None,
) -> PiecewiseLinearComplex:
    regions = [0] * len(cubes) if regions is None else regions
    vertices: list[tuple[int, int, int]] = []
    ids: dict[tuple[int, int, int], int] = {}
    faces: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    for cube, region in zip(cubes, regions, strict=True):
        corners = []
        for i in range(8):
            point = (
                cube[0] + (i & 1),
                cube[1] + ((i >> 1) & 1),
                cube[2] + ((i >> 2) & 1),
            )
            if point not in ids:
                ids[point] = len(vertices)
                vertices.append(point)
            corners.append(ids[point])
        for local in _LOOPS:
            face = tuple(corners[i] for i in local)
            faces.setdefault(tuple(sorted(face)), []).append((face, region))
    polygons, incidence = [], []
    for entries in faces.values():
        loop, region = entries[0]
        other = entries[1][1] if len(entries) == 2 else -1
        if other == region:
            continue
        polygons.append(loop)
        incidence.append((other, region))
    # Semantic groups deliberately span noncoplanar faces: they must not be
    # merged into a single invalid polygon by restricted assembly.
    pairs, facet_ids = np.unique(incidence, axis=0, return_inverse=True)
    return M.PiecewiseLinearComplex(
        vertices,
        polygons,
        facet_ids,
        pairs,
        tuple(f"material-{r}" for r in sorted(set(regions))),
    )


def _periodic_cube_source() -> tuple[
    PiecewiseLinearComplex, tuple[PeriodicConstraint, ...]
]:
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=float
    )
    domain = PiecewiseLinearComplex(
        points, _LOOPS, np.arange(6), [(-1, 0)] * 6, ("material-0",)
    )
    constraints = []
    for axis, source_facet, target_facet in ((0, 4, 5), (1, 2, 3), (2, 0, 1)):
        source = M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            2,
            f"source-{axis}",
            [source_facet],
        )
        target = M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            2,
            f"target-{axis}",
            [target_facet],
        )
        matrix = np.eye(4)
        matrix[axis, 3] = 1
        constraints.append(
            M.PeriodicConstraint(
                source,
                target,
                matrix,
                tolerance=0,
                source_entity_ids=np.asarray([source_facet], dtype=np.int64),
                orientations=np.asarray([-1], dtype=np.int64),
            )
        )
    return domain, tuple(constraints)


def _publish(
    complex_: PiecewiseLinearComplex,
    *,
    sites: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    schedule: NativePolyhedralSchedule | None = None,
    periodic_constraints: tuple[PeriodicConstraint, ...] = (),
    patch_controls: tuple[PatchControl, ...] = (),
    region_controls: tuple[M.RegionControl, ...] = (),
) -> CellMeshingResult:
    source = (
        M.NativePlcSource(complex_, "polyhedral-domain", "r1")
        if sites is None
        else M.NativePolyhedralSource(
            complex_, "polyhedral-domain", "r1", sites=sites, weights=weights
        )
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-facets",
        np.arange(complex_.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        scope,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            M.UniformSizeControl(scope, 10.0, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=region_controls,
        periodic_constraints=periodic_constraints,
        patch_controls=patch_controls,
    )
    prepared = PreparedPolyhedralVolume(source, specification, schedule)
    return execute_polyhedral_route(
        source,
        specification,
        prepared,
        phx.SpatialCoordinateContract.si(),
        M.NativeMeshingProvider.info(),
        "polyhedral-test-plan",
    )


def test_nonconvex_material_coverage_reciprocity_and_consumers() -> None:
    domain = _voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0])
    result = _publish(domain)
    assert isinstance(result, M.CellMeshingResult)
    assert result.certification is not None
    assert result.certification.passed
    fv = prepare_polyhedral_finite_volume_geometry(
        result.mesh, cell_geometry=result.geometry
    )
    vem = prepare_polyhedral_h1_virtual_element_3d(
        result.mesh, cell_geometry=result.geometry
    )
    assert float(jnp.sum(fv.cell_volumes)) == pytest.approx(3.0, abs=1e-12)
    assert vem.evidence.minimum_rank_margin > 0
    assert vem.evidence.maximum_reproduction_defect < 1e-10
    ones = jnp.ones(vem.dof_count)
    np.testing.assert_allclose(vem.mv(ones), 0, atol=1e-10)
    x = jnp.asarray(result.mesh.coordinates)[:, 0]
    assert float(jnp.vdot(x, vem.mv(x))) == pytest.approx(3.0, abs=1e-10)
    c = result.mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError(
            "Native polyhedral publication requires root PolyhedralConnectivity."
        )
    signs, faces = np.asarray(c.cell_face_sign_values), np.asarray(c.cell_face_values)
    sums = np.bincount(faces, weights=signs, minlength=c.face_count)
    counts = np.bincount(faces, minlength=c.face_count)
    np.testing.assert_array_equal(sums[counts == 2], 0)
    assert {zone.name for zone in result.zones} == {"material-0", "material-1"}
    assert any(label.name == "interface" for label in result.labels)


def test_canonical_weighted_regeneration_preserves_source_strata_and_inventory() -> None:
    from phydrax.discretization.finite_volume._unstructured import (
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.discretization.finite_volume._unstructured_remap import (
        UnstructuredConservativeRemapPlan,
    )

    domain = _voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0])
    source = _publish(
        domain,
        sites=np.asarray(
            [[0.25, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]], dtype=np.float64
        ),
        weights=np.asarray([0.03, -0.02, 0.01], dtype=np.float64),
    )
    original_id = source.result_id
    request = M.PolyhedralMeshAdaptation(
        M.PolyhedralAdaptationOperation.REGENERATE,
        complex_=domain,
        sites=np.asarray(
            [[0.5, 0.25, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]], dtype=np.float64
        ),
        weights=np.asarray([0.02, -0.01, 0.03], dtype=np.float64),
    )
    adapted = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            request,
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
                audit_policy=M.CellMeshAuditPolicy(
                    require_complete_association=True,
                    watertight_boundary=M.CellMeshAuditDisposition.REJECT,
                ),
            ),
        )
    )
    target = adapted.target
    assert target.certification is not None and target.certification.passed
    assert {patch.name for patch in target.patches} == {
        patch.name for patch in source.patches
    }
    assert {label.name for label in target.labels} == {
        label.name for label in source.labels
    }
    assert {(zone.name, zone.material_id, zone.region_role) for zone in target.zones} == {
        (zone.name, zone.material_id, zone.region_role) for zone in source.zones
    }
    for dimension in range(4):
        previous = next(
            value
            for value in source.associations
            if value.target_entity_set_id
            == source.mesh.entity_set(dimension).entity_set_id
        )
        current = next(
            value
            for value in target.associations
            if value.target_entity_set_id
            == target.mesh.entity_set(dimension).entity_set_id
        )
        # New cut vertices may lie on different source edges after a site move;
        # the fixed source points and the owning higher-dimensional strata persist.
        previous_strata = {
            identity
            for identity, source_dimension in zip(
                previous.source_entity_ids,
                np.asarray(previous.source_dimensions),
                strict=True,
            )
            if dimension != 0 or source_dimension == 0
        }
        current_strata = {
            identity
            for identity, source_dimension in zip(
                current.source_entity_ids,
                np.asarray(current.source_dimensions),
                strict=True,
            )
            if dimension != 0 or source_dimension == 0
        }
        assert current_strata == previous_strata
        assert current.source_id == previous.source_id
        assert current.source_revision == previous.source_revision
    common = adapted.common_refinement
    assert common is not None and common.status is CommonRefinementStatus.SUCCESS
    old = UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh).prepare(
        cell_geometry=source.geometry
    )
    new = UnstructuredFiniteVolumePlan.from_cell_mesh(target.mesh).prepare(
        cell_geometry=target.geometry
    )
    remap = UnstructuredConservativeRemapPlan(
        old,
        new,
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
        intersection_error_bounds=common.volume_error_bounds,
    )
    assert source.certification is not None
    values = np.where(
        np.asarray(source.certification.request.cell_regions) == 0, 2.0, 5.0
    )
    carried = np.asarray(remap.apply(values))
    for region in (0, 1):
        before = np.asarray(source.certification.request.cell_regions) == region
        after = np.asarray(target.certification.request.cell_regions) == region
        assert np.dot(
            carried[after], np.asarray(new.cell_volumes)[after]
        ) == pytest.approx(
            np.dot(values[before], np.asarray(old.cell_volumes)[before]),
            abs=1e-11,
        )
    vem = prepare_polyhedral_h1_virtual_element_3d(
        target.mesh, cell_geometry=target.geometry
    )
    assert vem.evidence.minimum_rank_margin > 0
    assert vem.evidence.maximum_reproduction_defect < 1e-10
    assert source.result_id == original_id


def test_regenerated_partial_facet_label_tracks_physical_overlap() -> None:
    from phydrax.meshing._lineage import identity_lineage
    from phydrax.meshing._polyhedral_adaptation import PolyhedralAdaptationEvidence

    domain = _voxel_plc([(0, 0, 0)])
    published = _publish(domain)
    mesh = published.mesh
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("The source requires genuine packed polyhedral faces.")
    offsets = np.asarray(connectivity.face_vertex_offsets)
    vertices = np.asarray(connectivity.face_vertex_values)
    coordinates = np.asarray(mesh.coordinates)
    selected = [
        face
        for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
        if np.all(coordinates[vertices[a:b], 0] == 0)
    ]
    scope = M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        mesh.entity_set(2).entity_set_id,
        np.asarray(connectivity.face_global_ids)[selected],
        local_mesh=mesh,
    )
    audit_policy = M.CellMeshAuditPolicy(
        require_complete_association=True,
        watertight_boundary=M.CellMeshAuditDisposition.REJECT,
    )
    assert published.certification is not None
    source = M.certify_cell_mesh(
        mesh,
        published.coordinate_contract,
        geometry=published.geometry,
        audit_policy=audit_policy,
        patches=published.patches,
        zones=published.zones,
        labels=(*published.labels, M.MeshLabel("inlet", scope)),
        associations=published.associations,
        certification_inputs=published.certification.request,
        lineage=identity_lineage(mesh, mesh),
    )
    request = M.PolyhedralMeshAdaptation(
        M.PolyhedralAdaptationOperation.REGENERATE,
        complex_=domain,
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.125, -0.125], dtype=np.float64),
    )
    adapted = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            request,
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_POLYHEDRAL, audit_policy=audit_policy
            ),
        )
    )
    target = adapted.target
    assert target.certification is not None and target.certification.passed
    fv = prepare_polyhedral_finite_volume_geometry(target.mesh)
    np.testing.assert_allclose(np.sort(fv.cell_volumes), [0.25, 0.75], atol=1e-12)
    inlet = next(label for label in target.labels if label.name == "inlet")
    c = target.mesh.connectivity
    if not isinstance(c, PolyhedralConnectivity):
        raise TypeError("Regeneration must preserve genuine packed polyhedral faces.")
    rows = np.flatnonzero(
        np.isin(np.asarray(c.face_global_ids), np.asarray(inlet.scope.entity_ids))
    )
    target_points = np.asarray(target.mesh.coordinates)
    for face in rows:
        a, b = np.asarray(c.face_vertex_offsets)[face : face + 2]
        assert np.all(target_points[np.asarray(c.face_vertex_values)[a:b], 0] == 0)
    evidence = adapted.evidence
    assert isinstance(evidence, PolyhedralAdaptationEvidence)
    np.testing.assert_array_equal(evidence.site_weights, request.weights)
    np.testing.assert_array_equal(evidence.site_points, request.sites)


def test_coincident_carrier_bisector_and_conservative_nonnested_remap() -> None:
    domain = _voxel_plc([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [0, 1, 0])
    source = _publish(domain)
    target = _publish(
        domain, sites=np.asarray([[0.5, 0.25, 0.5], [0.5, 1.75, 0.5]], dtype=np.float64)
    )
    assert source.certification is not None and target.certification is not None
    assert source.certification.passed and target.certification.passed
    refinement = prepare_common_refinement(
        source.mesh,
        target.mesh,
        source_geometry=source.geometry,
        target_geometry=target.geometry,
    )
    assert refinement.status is CommonRefinementStatus.SUCCESS
    source_connectivity = source.mesh.connectivity
    target_connectivity = target.mesh.connectivity
    if not isinstance(source_connectivity, PolyhedralConnectivity):
        raise TypeError("Native polyhedral source requires root PolyhedralConnectivity.")
    if not isinstance(target_connectivity, PolyhedralConnectivity):
        raise TypeError("Native polyhedral target requires root PolyhedralConnectivity.")
    assert source_connectivity.cell_count == 2
    assert target_connectivity.cell_count == 3
    inventory = np.array([2.0, 5.0])
    overlaps = np.asarray(refinement.volumes)
    masses = np.bincount(
        np.asarray(refinement.target_cells),
        weights=overlaps * inventory[np.asarray(refinement.source_cells)],
        minlength=target_connectivity.cell_count,
    )
    target_average = masses / np.asarray(refinement.target_measures)
    assert np.dot(
        target_average, np.asarray(refinement.target_measures)
    ) == pytest.approx(
        np.dot(inventory, np.asarray(refinement.source_measures)), abs=1e-12
    )
    assert np.all((target_average >= 2) & (target_average <= 5))


def test_disconnected_components_and_hole_decomposition() -> None:
    disconnected = generate_polyhedral_volume(_voxel_plc([(0, 0, 0), (3, 0, 0)]))
    connectivity = disconnected.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "Native polyhedral construction requires root PolyhedralConnectivity."
        )
    assert connectivity.cell_count == 2
    np.testing.assert_array_equal(disconnected.cell_sites, [0, 0])
    np.testing.assert_array_equal(disconnected.component_ids, [0, 1])
    ring = [
        (x, y, 0) for x, y in itertools.product(range(3), repeat=2) if (x, y) != (1, 1)
    ]
    result = _publish(_voxel_plc(ring))
    assert result.certification is not None
    assert result.certification.passed
    fv = prepare_polyhedral_finite_volume_geometry(result.mesh)
    assert float(jnp.sum(fv.cell_volumes)) == pytest.approx(8.0, abs=1e-12)
    assert np.max(np.asarray(result.mesh.coordinates)[:, 0]) == 3
    # Every star quadrature point remains outside the hole (no convex-hull fill).
    q = np.asarray(fv.cell_quadrature_points)[np.asarray(fv.cell_quadrature_valid)]
    assert not np.any((q[:, 0] > 1) & (q[:, 0] < 2) & (q[:, 1] > 1) & (q[:, 1] < 2))


def test_hidden_duplicate_sites_and_required_survival() -> None:
    domain = _voxel_plc([(0, 0, 0)])
    construction = generate_polyhedral_volume(
        domain, sites=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], weights=[0, -100]
    )
    np.testing.assert_array_equal(construction.cell_sites, [0])
    np.testing.assert_allclose(construction.achieved_site_volumes, [1, 0], atol=1e-12)
    with pytest.raises(PolyhedralGenerationError, match="required_empty_sites"):
        generate_polyhedral_volume(
            domain,
            sites=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
            weights=[0, -100],
            schedule=NativePolyhedralSchedule(require_every_site=True),
        )


def test_weight_optimization_meets_requested_site_volumes() -> None:
    construction = generate_polyhedral_volume(
        _voxel_plc([(0, 0, 0)]),
        sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
        schedule=NativePolyhedralSchedule(target_site_volumes=(0.25, 0.75)),
    )
    np.testing.assert_allclose(
        construction.achieved_site_volumes, [0.25, 0.75], atol=1e-6
    )
    assert construction.optimization_relative_residual <= 1e-6


def test_six_hundred_collinear_sites_cover_without_all_pairs_limit() -> None:
    points = np.asarray(
        [[(i >> axis) & 1 for axis in range(3)] for i in range(8)], dtype=np.float64
    )
    tets = []
    for permutation in itertools.permutations(range(3)):
        corner = np.zeros(3, dtype=int)
        tet = [0]
        for axis in permutation:
            corner[axis] = 1
            tet.append(int(corner @ np.array([1, 2, 4])))
        if np.linalg.det(points[tet[1:]] - points[tet[0]]) < 0:
            tet[0], tet[1] = tet[1], tet[0]
        tets.append(tet)
    facets = []
    for tet in tets:
        row = []
        for opposite in range(4):
            face = points[[tet[k] for k in range(4) if k != opposite]]
            token = -1
            for axis in range(3):
                for side in (0, 1):
                    if np.all(face[:, axis] == side):
                        token = 2 * axis + side
            row.append(token)
        facets.append(row)
    sites = np.column_stack(
        (np.linspace(0.01, 0.99, 600), np.full(600, 0.371), np.full(600, 0.419))
    )
    diagram = RestrictedPowerDiagram(
        sites, np.zeros(600), points, tets, np.zeros(6, dtype=int), facets
    )
    data = diagram.construction
    assert np.unique(data.cell_sites).size == 600
    assert np.sum(data.cell_volumes) == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(
        np.sum(data.cell_moments, axis=0), [0.5, 0.5, 0.5], atol=1e-10
    )


def test_weighted_collinear_sites_retain_their_restricted_partition() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    diagram = RestrictedPowerDiagram(
        [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
        [0.03, -0.02],
        points,
        [[0, 1, 2, 3]],
        [0],
        [[0, 1, 2, 3]],
    )
    data = diagram.construction
    volumes = np.bincount(data.cell_sites, weights=data.cell_volumes, minlength=2)
    # The weighted bisector is x=0.55; the right tetrahedron has edge scale 0.45.
    np.testing.assert_allclose(volumes, [1 / 6 - 0.45**3 / 6, 0.45**3 / 6], atol=1e-12)
    assert np.sum(volumes) == pytest.approx(1 / 6, abs=1e-12)


def test_weighted_cube_collinear_faces_admit_fv_and_vem() -> None:
    result = _publish(
        _voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.03, -0.02], dtype=np.float64),
    )
    fv = prepare_polyhedral_finite_volume_geometry(
        result.mesh, cell_geometry=result.geometry
    )
    vem = prepare_polyhedral_h1_virtual_element_3d(
        result.mesh, cell_geometry=result.geometry
    )
    np.testing.assert_allclose(np.sort(fv.cell_volumes), [0.45, 0.55], atol=1e-12)
    x = np.asarray(result.mesh.coordinates)[:, 0]
    assert float(jnp.vdot(x, vem.mv(x))) == pytest.approx(1.0, abs=1e-10)
    assert vem.evidence.minimum_rank_margin > 0
    left = jnp.linspace(0.1, 1.0, vem.dof_count)
    right = jnp.linspace(1.0, -0.2, vem.dof_count)
    assert float(jnp.vdot(left, vem.mv(right))) == pytest.approx(
        float(jnp.vdot(vem.transpose_mv(left), right)), abs=1e-10
    )
    quadrature_moment = np.sum(
        np.asarray(fv.cell_quadrature_points)
        * np.asarray(fv.cell_quadrature_weights)[..., None],
        axis=(0, 1),
    )
    np.testing.assert_allclose(quadrature_moment, [0.5, 0.5, 0.5], atol=1e-12)


def test_nonconvex_face_triangulation_preserves_polynomial_geometry() -> None:
    base = np.array([[0, 0], [2, 0], [2, 2], [1, 1.5], [0, 2]], dtype=np.float64)
    n = base.shape[0]
    points = np.vstack(
        (np.column_stack((base, np.zeros(n))), np.column_stack((base, np.ones(n))))
    )
    faces = [
        np.arange(n)[::-1],
        np.arange(n, 2 * n),
        *[np.array([i, (i + 1) % n, (i + 1) % n + n, i + n]) for i in range(n)],
    ]
    mesh = phx.discretization.CellMesh.from_polyhedra(points, [faces])
    fv = prepare_polyhedral_finite_volume_geometry(mesh)
    vem = prepare_polyhedral_h1_virtual_element_3d(mesh)
    assert float(fv.cell_volumes[0]) == pytest.approx(3.5, abs=1e-12)
    assert float(vem.evidence.cell_volumes[0]) == pytest.approx(3.5, abs=1e-12)
    x = jnp.asarray(mesh.coordinates)[:, 0]
    assert float(jnp.vdot(x, vem.mv(x))) == pytest.approx(3.5, abs=1e-10)
    q = np.asarray(fv.face_quadrature_points)
    valid = np.asarray(fv.face_quadrature_valid)
    face_centers = np.asarray(fv.face_centers)
    # The top/bottom notch is excluded by constrained-domain triangulation.
    cap = np.abs(np.asarray(fv.face_area_vectors)[:, 2]) > 0
    cap_points = q[cap][valid[cap]]
    assert not np.any((cap_points[:, 1] > 1.5 + 0.5 * np.abs(cap_points[:, 0] - 1)))
    np.testing.assert_allclose(face_centers[cap, 0], 1, atol=1e-12)


def test_exact_weighted_plane_restriction_preserves_source_and_consumer_volume() -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry_validity import (
        certify_cell_geometry_validity,
    )
    from phydrax.meshing._polyhedral_adaptation import adapt_polyhedral_mesh
    from phydrax.meshing._topology_edit import assemble_topology_edit

    source = _publish(
        _voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.03, -0.02], dtype=np.float64),
    )
    plane = np.asarray([1.0, 0.3, 0.2, 0.8], dtype=np.float64)
    outcome = adapt_polyhedral_mesh(
        source.mesh,
        M.PolyhedralMeshAdaptation(
            M.PolyhedralAdaptationOperation.PLANE_SPLIT, planes=plane[None, :]
        ),
        source_geometry=source.geometry,
    )
    mesh, _, _ = assemble_topology_edit(
        source.mesh, outcome.edit, numeric_version="exact-plane-test"
    )
    geometry = outcome.geometry
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    assert certify_cell_geometry_validity(geometry, mesh=mesh).all_certified
    restricted = geometry.exact_source
    if not isinstance(
        restricted, phx.discretization.ExactPowerCellGeometryRestrictionSource
    ):
        raise AssertionError("The plane restriction lost its original exact source.")
    coordinates = restricted.prepare(mesh.coordinates).vertices
    normal = tuple(Fraction(float(value)) for value in plane[:3])
    offset = Fraction(float(plane[3]))
    for vertex in np.flatnonzero(np.asarray(restricted.vertex_plane_ids) >= 0):
        assert (
            sum(
                (
                    value * coefficient
                    for value, coefficient in zip(
                        coordinates[vertex], normal, strict=True
                    )
                ),
                Fraction(0),
            )
            == offset
        )
    fv = prepare_polyhedral_finite_volume_geometry(mesh, cell_geometry=geometry)
    vem = prepare_polyhedral_h1_virtual_element_3d(mesh, cell_geometry=geometry)
    assert float(jnp.sum(fv.cell_volumes)) == pytest.approx(1.0, abs=1e-12)
    assert float(jnp.sum(vem.evidence.cell_volumes)) == pytest.approx(1.0, abs=1e-12)
    assert vem.evidence.maximum_reproduction_defect < 1e-10


def test_exact_power_geometry_persistence_roundtrip_keeps_power_source() -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.lifecycle._meshing_sources import (
        register_meshing_source_artifacts,
        validate_meshing_source_closure,
    )

    register_meshing_source_artifacts()
    source = _publish(
        _voxel_plc([(0, 0, 0)]),
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.asarray([0.03, -0.02], dtype=np.float64),
    )
    geometry = source.geometry
    if not isinstance(
        geometry.exact_source, phx.discretization.ExactPowerCellGeometrySource
    ):
        raise AssertionError(
            "Weighted polyhedral publication lost its exact power source."
        )
    recipe = model_structure_recipe(geometry)
    restored = model_from_array_recipe(
        recipe,
        model_recipe_array_values(geometry, recipe, prefix="power"),
        prefix="power",
    )
    validate_meshing_source_closure(restored)
    if not isinstance(
        restored.exact_source, phx.discretization.ExactPowerCellGeometrySource
    ):
        raise AssertionError("The restored geometry lost its exact power source kind.")
    assert restored.exact_source.source_id == geometry.exact_source.source_id
    assert cell_geometry_id(restored) == cell_geometry_id(geometry)
    assert restored.source_coordinates() == geometry.source_coordinates()
    fv = prepare_polyhedral_finite_volume_geometry(source.mesh, cell_geometry=restored)
    reference = prepare_polyhedral_finite_volume_geometry(
        source.mesh, cell_geometry=geometry
    )
    np.testing.assert_array_equal(
        np.asarray(fv.cell_volumes), np.asarray(reference.cell_volumes)
    )


def test_exact_power_source_refuses_corrupt_witness_carrier_rank_and_budgets() -> None:
    owner = phx.discretization.ExactPowerCellGeometrySource
    sites = np.asarray([[0.0, 0.0, 0.0], [0.7, 0.6, 0.8]], dtype=np.float64)
    weights = np.asarray([0.03, -0.02], dtype=np.float64)
    carrier = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    offsets = np.asarray([0, 1, 2, 3, 4, 6, 8, 10], dtype=np.int64)
    equal = np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    witnesses = np.asarray(
        [
            [0, -1, -1, -1],
            [1, -1, -1, -1],
            [2, -1, -1, -1],
            [3, -1, -1, -1],
            [0, 3, -1, -1],
            [1, 3, -1, -1],
            [2, 3, -1, -1],
        ],
        dtype=np.int32,
    )
    source = owner(sites, weights, carrier, tets, offsets, equal, witnesses)
    rounded = source.prepare().rounded_vertices
    wrong_rne = rounded.copy()
    wrong_rne[4, 2] = np.nextafter(wrong_rne[4, 2], np.inf)
    with pytest.raises(ValueError):
        source.prepare(wrong_rne)
    wrong_owner = equal.copy()
    wrong_owner[3] = 0
    with pytest.raises(ValueError):
        owner(sites, weights, carrier, tets, offsets, wrong_owner, witnesses).prepare(
            rounded
        )
    wrong_rank = witnesses.copy()
    wrong_rank[4] = [0, 1, 3, -1]
    with pytest.raises(ValueError):
        owner(sites, weights, carrier, tets, offsets, equal, wrong_rank).prepare(rounded)
    flat_carrier = carrier.copy()
    flat_carrier[3] = [1.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        owner(sites, weights, flat_carrier, tets, offsets, equal, witnesses).prepare()
    with pytest.raises(ValueError):
        owner(
            sites, weights, carrier, tets, offsets, equal, witnesses, maximum_work=1
        ).prepare()
    with pytest.raises(ValueError):
        owner(
            sites, weights, carrier, tets, offsets, equal, witnesses, maximum_bits=1
        ).prepare()
    with pytest.raises(ValueError):
        owner(
            sites,
            weights,
            carrier,
            tets,
            offsets,
            equal,
            witnesses,
            maximum_condition=1.0,
        ).prepare()


def _independent_oriented_source_integrals(
    construction: PolyhedralConstruction,
) -> tuple[list[Fraction], list[list[Fraction]]]:
    """Rational divergence oracle, independent of native moments and FV/VEM."""
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    points = np.asarray(construction.geometry.source_coordinates(), dtype=object)
    face_offsets = np.asarray(c.face_vertex_offsets)
    face_vertices = np.asarray(c.face_vertex_values)
    cell_offsets = np.asarray(c.cell_face_offsets)
    cell_faces = np.asarray(c.cell_face_values)
    signs = np.asarray(c.cell_face_sign_values)
    volumes, moments = [], []
    for start, stop in zip(cell_offsets[:-1], cell_offsets[1:], strict=True):
        volume = Fraction(0)
        moment = [Fraction(0) for _ in range(3)]
        closed_area = [Fraction(0) for _ in range(3)]
        for face, sign in zip(cell_faces[start:stop], signs[start:stop], strict=True):
            a, b = face_offsets[face : face + 2]
            polygon = points[face_vertices[a:b]]
            for index in range(1, len(polygon) - 1):
                p, q, r = polygon[0], polygon[index], polygon[index + 1]
                cross = (
                    q[1] * r[2] - q[2] * r[1],
                    q[2] * r[0] - q[0] * r[2],
                    q[0] * r[1] - q[1] * r[0],
                )
                tetrahedron = (
                    int(sign) * sum(p[axis] * cross[axis] for axis in range(3)) / 6
                )
                volume += tetrahedron
                u, v = q - p, r - p
                area = (
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                )
                for axis in range(3):
                    closed_area[axis] += int(sign) * area[axis] / 2
                    moment[axis] += tetrahedron * (p[axis] + q[axis] + r[axis]) / 4
        assert volume > 0
        assert closed_area == [0, 0, 0]  # Constant-vector flux closure per cell.
        volumes.append(volume)
        moments.append(moment)
    face_counts = np.bincount(cell_faces, minlength=c.face_count)
    face_signs = np.bincount(cell_faces, weights=signs, minlength=c.face_count)
    assert np.all((face_counts == 1) | (face_counts == 2))
    np.testing.assert_array_equal(face_signs[face_counts == 2], 0)
    return volumes, moments


@pytest.mark.parametrize(
    "cubes",
    [
        [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        [(0, 0, 0), (3, 0, 0)],
        [(x, y, 0) for x, y in itertools.product(range(3), repeat=2) if (x, y) != (1, 1)],
    ],
)
def test_declared_domain_independent_exact_volume_moment_and_flux(
    cubes: Sequence[tuple[int, int, int]],
) -> None:
    construction = generate_polyhedral_volume(
        _voxel_plc(cubes),
        sites=[[0.25, 0.5, 0.5], [2.75, 0.5, 0.5]],
        weights=[0.125, -0.25],
    )
    volumes, moments = _independent_oriented_source_integrals(construction)
    assert sum(volumes) == len(cubes)
    for axis in range(3):
        expected = sum(
            (Fraction(cube[axis]) + Fraction(1, 2) for cube in cubes), Fraction(0)
        )
        assert sum(moment[axis] for moment in moments) == expected
    points = np.asarray(construction.geometry.source_coordinates(), dtype=object)
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    offsets, vertices = (
        np.asarray(c.cell_vertex_offsets),
        np.asarray(c.cell_vertex_values),
    )
    sites = [[Fraction(float(x)) for x in row] for row in construction.diagram.points]
    weights = [Fraction(float(x)) for x in construction.diagram.weights]
    for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        owner = int(construction.cell_sites[cell])
        for point in points[vertices[a:b]]:
            powers = [
                sum((point[axis] - site[axis]) ** 2 for axis in range(3)) - weight
                for site, weight in zip(sites, weights, strict=True)
            ]
            assert powers[owner] == min(powers)


def test_concave_declared_prism_and_thin_feature_exact_oracle() -> None:
    base = np.asarray([[0, 0], [2, 0], [2, 2], [1, 1.5], [0, 2]], dtype=float)
    height = 2.0**-12
    n = len(base)
    points = np.vstack(
        (
            np.column_stack((base, np.zeros(n))),
            np.column_stack((base, np.full(n, height))),
        )
    )
    polygons = [
        np.arange(n)[::-1],
        np.arange(n, 2 * n),
        *[np.asarray([i, (i + 1) % n, (i + 1) % n + n, i + n]) for i in range(n)],
    ]
    domain = PiecewiseLinearComplex(
        points, polygons, np.zeros(len(polygons), dtype=int), [[-1, 0]], ("material-0",)
    )
    construction = generate_polyhedral_volume(domain)
    volumes, moments = _independent_oriented_source_integrals(construction)
    assert sum(volumes) == Fraction(7, 2) * Fraction(height)
    assert sum(moment[0] for moment in moments) == sum(volumes)
    assert sum(moment[2] for moment in moments) == sum(volumes) * Fraction(height) / 2
    assert np.all(construction.feature_coverage_gap == 0)
    assert np.all(construction.feature_maximum_distance == 0)


def test_required_site_survival_is_not_overridden_by_loose_volume_target() -> None:
    with pytest.raises(PolyhedralGenerationError, match="required_empty_sites"):
        generate_polyhedral_volume(
            _voxel_plc([(0, 0, 0)]),
            sites=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 2.0**-10]],
            weights=[0, -100],
            schedule=NativePolyhedralSchedule(
                require_every_site=True,
                target_site_volumes=(0.5, 0.5),
                relative_volume_tolerance=2,
            ),
        )


def test_unmet_cell_quality_is_a_bounded_failure() -> None:
    with pytest.raises(
        PolyhedralGenerationError, match="refinement_unresolved"
    ) as caught:
        generate_polyhedral_volume(
            _voxel_plc([(0, 0, 0)]),
            schedule=NativePolyhedralSchedule(
                maximum_cell_diameter=0.01,
                maximum_refinement_steps=0,
            ),
        )
    assert caught.value.entities
    assert isinstance(caught.value.achieved, dict)
    assert caught.value.achieved["diameter_upper"] > 0.01


def test_native_clipping_refusal_preserves_status_failure_and_counters() -> None:
    from phydrax._meshcore import MeshcoreStatus, regular_3d, RestrictedPowerFailure
    from phydrax.meshing.providers._native_polyhedral import _construction_failure

    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    sites = np.asarray(
        [[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6]],
        dtype=np.float64,
    )
    weights = np.zeros(4, dtype=np.float64)
    prepared, _ = regular_3d(sites, weights, max_tetrahedra=64)
    accepted = RestrictedPowerDiagram(
        sites,
        weights,
        points,
        [[0, 1, 2, 3]],
        [0],
        [[0, 1, 2, 3]],
        max_pieces=64,
    )
    clipping_capacity = len(accepted.construction.piece_sites) - 1
    assert len(prepared) <= clipping_capacity
    with pytest.raises(RestrictedPowerFailure) as caught:
        RestrictedPowerDiagram(
            sites,
            weights,
            points,
            [[0, 1, 2, 3]],
            [0],
            [[0, 1, 2, 3]],
            max_pieces=clipping_capacity,
        )
    native = caught.value
    assert native.status == MeshcoreStatus.CAPACITY_EXCEEDED
    failure = _construction_failure(native, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    evidence = dict(failure.achieved)
    assert evidence["native_status"] == native.status
    for index, value in enumerate(native.failure):
        assert evidence[f"power_failure_{index}"] == value
    for index, value in enumerate(native.counters):
        assert evidence[f"power_counter_{index}"] == value


def test_exact_publication_budget_refusal_preserves_actual_allowance() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.meshing.providers._native_polyhedral import _construction_failure

    ledger = CoordinateEnclosureBudget(1, 1024)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        ledger.reserve(2, 0)
    failure = _construction_failure(caught.value, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.requested)["coefficient_work"] == 1
    assert dict(failure.achieved) == {"requested": 2, "completed": 0}


def test_thin_material_partition_has_exact_interface_and_region_measures() -> None:
    source = _voxel_plc([(0, 0, 0), (1, 0, 0)], [0, 1])
    height = 2.0**-10
    points = np.array(source.vertices, copy=True)
    points[:, 2] *= height
    polygons = [
        source.polygon_vertices[a:b]
        for a, b in zip(
            source.polygon_offsets[:-1], source.polygon_offsets[1:], strict=True
        )
    ]
    domain = PiecewiseLinearComplex(
        points,
        polygons,
        source.polygon_facets,
        source.facet_regions,
        source.region_ids,
    )
    construction = generate_polyhedral_volume(
        domain,
        sites=[[0.25, 0.5, height / 2], [1.75, 0.5, height / 2]],
        weights=[0.125, 0],
    )
    volumes, moments = _independent_oriented_source_integrals(construction)
    for region in (0, 1):
        cells = np.flatnonzero(construction.cell_regions == region)
        assert sum(volumes[cell] for cell in cells) == Fraction(height)
        assert sum(moments[cell][0] for cell in cells) == Fraction(height) * Fraction(
            2 * region + 1, 2
        )
    assert construction.interface_faces.size > 0
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    faces = np.asarray(c.cell_face_values)
    offsets = np.asarray(c.cell_face_offsets)
    signs = np.asarray(c.cell_face_sign_values)
    for face in construction.interface_faces:
        incident = []
        orientation = []
        for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            positions = np.flatnonzero(faces[a:b] == face)
            if positions.size:
                incident.append(int(construction.cell_regions[cell]))
                orientation.append(int(signs[a + positions[0]]))
        assert sorted(incident) == [0, 1]
        assert sum(orientation) == 0
    assert np.all(construction.feature_coverage_gap == 0)


def _unequal_cube_seam_fixture() -> tuple[
    CellMesh, CellGeometrySpec, np.ndarray, MeshingScope, MeshingScope, np.ndarray
]:
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=float
    )
    faces = [*_LOOPS[:4], (0, 4, 6), (0, 6, 2), _LOOPS[5]]
    mesh = phx.discretization.CellMesh.from_polyhedra(
        points, [[np.asarray(face, dtype=np.int64) for face in faces]]
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    c = mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    offsets, values = np.asarray(c.face_vertex_offsets), np.asarray(c.face_vertex_values)
    facets = np.full(c.face_count, -1, dtype=np.int64)
    for face, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        corners = set(map(int, values[a:b]))
        if corners.issubset(_LOOPS[4]):
            facets[face] = 4
        if corners.issubset(_LOOPS[5]):
            facets[face] = 5
    source = M.MeshingScope(
        "cube-seam", "r1", M.MeshingEntityKind.GEOMETRY, 2, "left", [4]
    )
    target = M.MeshingScope(
        "cube-seam", "r1", M.MeshingEntityKind.GEOMETRY, 2, "right", [5]
    )
    matrix = np.eye(4)
    matrix[0, 3] = 1
    return mesh, geometry, facets, source, target, matrix


def test_periodic_unequal_seam_fragments_reconstruct_both_exact_parent_actions() -> None:
    mesh, geometry, facets, source, target, matrix = _unequal_cube_seam_fixture()
    constraint = M.PeriodicConstraint(
        source,
        target,
        matrix,
        tolerance=0,
        source_entity_ids=np.asarray([4], dtype=np.int64),
        orientations=np.asarray([-1], dtype=np.int64),
    )
    fragments = prepare_periodic_polyhedral_seam_fragments(
        mesh,
        geometry,
        facets,
        [constraint],
        maximum_work=1_000_000,
        maximum_scratch_bytes=16_000_000,
    )
    assert fragments
    assert sum(fragment.projected_measure for fragment in fragments) == 1
    points = np.asarray(geometry.source_coordinates(), dtype=object)
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    face_ids = np.asarray(connectivity.cell_face_values)
    cell_signs = np.asarray(connectivity.cell_face_sign_values)
    outward_sign = dict(zip(map(int, face_ids), map(int, cell_signs), strict=True))
    np.testing.assert_array_equal(constraint.orientations, [-1])
    for fragment in fragments:
        assert fragment.constraint_id == constraint.constraint_id
        # Connectivity stores canonical loops, independently of outward cell signs.
        assert (
            fragment.relative_orientation
            * outward_sign[fragment.source_face]
            * outward_sign[fragment.target_face]
        ) == -1
        for point, source_weights, target_weights in zip(
            fragment.vertices,
            fragment.source_coefficients,
            fragment.target_coefficients,
            strict=True,
        ):
            original = tuple(
                sum(
                    weight * points[vertex, axis]
                    for vertex, weight in zip(
                        fragment.source_triangle, source_weights, strict=True
                    )
                )
                for axis in range(3)
            )
            transformed = (original[0] + 1, original[1], original[2])
            target_point = tuple(
                sum(
                    weight * points[vertex, axis]
                    for vertex, weight in zip(
                        fragment.target_triangle, target_weights, strict=True
                    )
                )
                for axis in range(3)
            )
            assert transformed == point == target_point
            assert sum(source_weights) == sum(target_weights) == 1
    source_faces = set(np.flatnonzero(facets == 4))
    assert {fragment.source_face for fragment in fragments} == source_faces
    for face in source_faces:
        assert sum(
            fragment.projected_measure
            for fragment in fragments
            if fragment.source_face == face
        ) == Fraction(1, 2)


def test_periodic_seam_tolerance_does_not_replace_original_source_transform() -> None:
    mesh, geometry, facets, source, target, matrix = _unequal_cube_seam_fixture()
    matrix[0, 3] += 2.0**-30
    constraint = M.PeriodicConstraint(source, target, matrix, tolerance=1e-6)
    with pytest.raises(PolyhedralGenerationError, match="periodic_seam_exact_coverage"):
        prepare_periodic_polyhedral_seam_fragments(
            mesh,
            geometry,
            facets,
            [constraint],
            maximum_work=1_000_000,
            maximum_scratch_bytes=16_000_000,
        )
    assert float(np.asarray(constraint.transform)[0, 3]) == 1 + 2.0**-30


def test_periodic_seam_common_refinement_refuses_original_work_capacity() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureResourceError,
    )

    mesh, geometry, facets, source, target, matrix = _unequal_cube_seam_fixture()
    constraint = M.PeriodicConstraint(source, target, matrix, tolerance=0)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        prepare_periodic_polyhedral_seam_fragments(
            mesh,
            geometry,
            facets,
            [constraint],
            maximum_work=1,
            maximum_scratch_bytes=16_000_000,
        )
    assert caught.value.limit == 1
    assert caught.value.requested > caught.value.completed


@pytest.mark.parametrize("reverse", [False, True])
def test_canonical_exact_clip_polygon_keeps_measure_and_oriented_loop(
    reverse: bool,
) -> None:
    from phydrax.geometry._planar_coverage import (
        intersection,
        intersection_polygon,
        signed_measure,
    )

    subject = tuple(
        tuple(Fraction(x) for x in point) for point in [(0, 0), (2, 0), (2, 1), (0, 1)]
    )
    target = tuple(
        tuple(Fraction(x) for x in point) for point in [(1, -1), (3, -1), (3, 2), (1, 2)]
    )
    if reverse:
        subject = subject[::-1]
    polygon = intersection_polygon(subject, target)
    assert abs(signed_measure(polygon)) == intersection(subject, target) == 1
    assert signed_measure(polygon) * signed_measure(subject) > 0
    assert (
        intersection_polygon(
            ((Fraction(0),), (Fraction(1),)), ((Fraction(2),), (Fraction(3),))
        )
        == ()
    )


def test_periodic_image_capacity_refusal_keeps_original_source_bits_and_limit() -> None:
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.geometry._triangulation import (
        PeriodicPowerImageCapacityRefusal,
        PeriodicPowerPreparation,
    )
    from phydrax.meshing.providers._native_polyhedral import _construction_failure

    points = np.asarray([[0.25, -0.0, 0.25]])
    weights = np.asarray([-0.0])
    carrier = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(PeriodicPowerImageCapacityRefusal) as caught:
        PeriodicPowerPreparation(
            points, weights, carrier, PeriodicCell(np.eye(3)), maximum_images=1
        )
    error = caught.value
    np.testing.assert_array_equal(error.points.view(np.uint64), points.view(np.uint64))
    np.testing.assert_array_equal(error.weights.view(np.uint64), weights.view(np.uint64))
    failure = _construction_failure(error, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.requested)["site_images"] == error.maximum_images == 1
    assert dict(failure.achieved)["requested_site_images"] == error.requested_images > 1
    assert dict(failure.achieved)["completed_site_images"] == error.completed_images


def test_periodic_adjacency_refusal_keeps_source_identity_and_actual_work_bound() -> None:
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.geometry._triangulation import (
        PeriodicPowerPreparation,
        PeriodicPowerSourceRefusal,
    )
    from phydrax.meshing.providers._native_polyhedral import _construction_failure

    points = np.asarray([[0.25, 0.25, 0.25]])
    weights = np.asarray([0.0])
    carrier = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    group = PeriodicCell(np.eye(3))
    preparation = PeriodicPowerPreparation(
        points,
        weights,
        carrier,
        group,
        maximum_images=4096,
        maximum_work_units=2_000_000,
    )
    with pytest.raises(PeriodicPowerSourceRefusal) as caught:
        RestrictedPowerDiagram(
            points,
            weights,
            carrier,
            [[0, 1, 2, 3]],
            [0],
            [[0, 1, 2, 3]],
            periodic_group=group,
            periodic_preparation=preparation,
            work_limit=1,
        )
    error = caught.value
    assert error.preparation.source_id == preparation.source_id
    failure = _construction_failure(error, "volume_fill")
    assert failure.category == M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert preparation.source_id in str(failure)
    assert dict(failure.requested)["periodic_call_work_units"] == error.maximum_work == 1
    assert dict(failure.achieved)["periodic_requested_work_units"] == error.requested_work
    assert dict(failure.achieved)["periodic_completed_work_units"] == error.completed_work


def test_periodic_power_preparation_cold_archive_preserves_original_law_and_refuses_corruption(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys
    from copy import copy

    from phydrax._array_archive import ArrayArchiveLimits
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.geometry._triangulation import PeriodicPowerPreparation
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        validate_meshing_source_closure,
        write_meshing_source_closure,
    )

    points = np.asarray([[0.25, -0.0, 0.25]])
    weights = np.asarray([-0.0])
    carrier = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    preparation = PeriodicPowerPreparation(
        points,
        weights,
        carrier,
        PeriodicCell(np.eye(3)),
        maximum_images=257,
        maximum_work_units=1 << 26,
    )
    limits = ArrayArchiveLimits()
    receipt = write_meshing_source_closure(
        tmp_path / "periodic-power",
        (preparation, preparation),
        limits=limits,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from phydrax.lifecycle._meshing_sources import read_meshing_source_closure; "
            "from phydrax.geometry._triangulation import PeriodicPowerPreparation; "
            "v=read_meshing_source_closure(sys.argv[1], expected_content_id=sys.argv[2]); "
            "assert type(v[0]) is PeriodicPowerPreparation; "
            "assert v[0] is v[1]; "
            "v[0].validate_restored()",
            str(tmp_path / "periodic-power"),
            receipt.content_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    restored = read_meshing_source_closure(
        tmp_path / "periodic-power",
        expected_content_id=receipt.content_id,
        limits=limits,
    )
    owner = restored[0]
    assert owner is restored[1]
    assert type(owner) is PeriodicPowerPreparation
    assert owner.preparation_id == preparation.preparation_id
    assert owner.maximum_images == preparation.maximum_images
    assert owner.maximum_work_units == preparation.maximum_work_units
    for name in ("points", "weights", "domain_points"):
        np.testing.assert_array_equal(
            np.asarray(getattr(owner, name)).view(np.uint64),
            np.asarray(getattr(preparation, name)).view(np.uint64),
        )
    corrupted = copy(owner)
    actions = np.array(owner.image_exponents, copy=True)
    actions[0, 0] += 1
    object.__setattr__(corrupted, "image_exponents", actions)
    with pytest.raises(ValueError):
        validate_meshing_source_closure(corrupted, limits=limits)
    corrupted = copy(owner)
    object.__setattr__(corrupted, "carrier_id", "not-the-original-carrier")
    with pytest.raises(ValueError):
        validate_meshing_source_closure(corrupted, limits=limits)


def _source_face_area_vector(
    construction: PolyhedralConstruction, face: int
) -> tuple[Fraction, ...]:
    c = construction.mesh.connectivity
    assert isinstance(c, PolyhedralConnectivity)
    points = np.asarray(construction.geometry.source_coordinates(), dtype=object)
    a, b = np.asarray(c.face_vertex_offsets)[face : face + 2]
    polygon = points[np.asarray(c.face_vertex_values)[a:b]]
    vector = [Fraction(0)] * 3
    for index in range(1, len(polygon) - 1):
        u, v = polygon[index] - polygon[0], polygon[index + 1] - polygon[0]
        vector[0] += (u[1] * v[2] - u[2] * v[1]) / 2
        vector[1] += (u[2] * v[0] - u[0] * v[2]) / 2
        vector[2] += (u[0] * v[1] - u[1] * v[0]) / 2
    owner = int(np.asarray(c.face_owner)[face])
    first, last = np.asarray(c.cell_face_offsets)[owner : owner + 2]
    position = np.flatnonzero(np.asarray(c.cell_face_values)[first:last] == face)
    sign = int(np.asarray(c.cell_face_sign_values)[first + position[0]])
    return tuple(sign * value for value in vector)


def test_full_periodic_power_source_independent_measure_faces_and_flux() -> None:
    domain, constraints = _periodic_cube_source()
    construction = generate_polyhedral_volume(
        domain,
        sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
        weights=[0, 0],
        periodic_constraints=constraints,
    )
    volumes, moments = _independent_oriented_source_integrals(construction)
    assert sum(volumes) == 1
    assert (
        tuple(sum(moment[axis] for moment in moments) for axis in range(3))
        == (Fraction(1, 2),) * 3
    )
    assert construction.mesh.periodic_topology is not None
    assert construction.seam_face_pairs is not None
    assert construction.seam_vertex_pairs is not None
    for source_face, target_face, generator in construction.seam_face_pairs:
        source = _source_face_area_vector(construction, int(source_face))
        target = _source_face_area_vector(construction, int(target_face))
        matrix = np.asarray(constraints[generator].transform)
        rotated = tuple(
            sum(
                Fraction(float(matrix[axis, column])) * source[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        assert rotated == tuple(-value for value in target)
        velocity = (Fraction(2), Fraction(-3), Fraction(5))
        target_velocity = tuple(
            sum(
                Fraction(float(matrix[axis, column])) * velocity[column]
                for column in range(3)
            )
            for axis in range(3)
        )
        assert (
            sum(
                source[axis] * velocity[axis] + target[axis] * target_velocity[axis]
                for axis in range(3)
            )
            == 0
        )
    assert np.all(construction.feature_coverage_gap == 0)
    assert np.all(construction.feature_maximum_distance == 0)


def test_public_periodic_power_route_retains_original_source_and_patch_conformity() -> (
    None
):
    domain, constraints = _periodic_cube_source()
    patch = PatchControl("periodic-left", constraints[0].source_scope, ("material-0",))
    region = M.RegionControl(
        M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            3,
            "material-regions",
            [0],
        ),
        "material-0",
        "material-0",
        M.RegionRole.SOLID,
    )
    result = _publish(
        domain,
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
        periodic_constraints=constraints,
        patch_controls=(patch,),
        region_controls=(region,),
    )
    assert result.certification is not None and result.certification.passed
    assert result.mesh.periodic_topology is not None
    assert any(value.name == "periodic" for value in result.labels)
    named = next(value for value in result.patches if value.name == patch.name)
    assert named.scope.entity_ids.size
    assert named.adjacent_zone_ids == tuple(zone.zone_id for zone in result.zones)
    source = result.geometry.exact_source
    assert isinstance(source, phx.discretization.ExactPowerCellGeometryLinearActionSource)
    assert isinstance(source.parent, phx.discretization.ExactPowerCellGeometrySource)
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    assert isinstance(source.periodic_preparation, PeriodicPowerPreparation)
    assert isinstance(source.parent.periodic_preparation, PeriodicPowerPreparation)
    np.testing.assert_array_equal(
        source.parent.site_points, [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
    )
    np.testing.assert_array_equal(source.parent.site_weights, [0, 0])
    assert (
        source.periodic_preparation.source_id
        == source.parent.periodic_preparation.source_id
    )


def test_sheared_periodic_power_publication_uses_exact_source_not_double_rne_lifts() -> (
    None
):
    from phydrax.discretization._periodic_topology import PeriodicMeshTopology

    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=float
    )
    points[:, 0] += points[:, 1]
    domain = PiecewiseLinearComplex(
        points, _LOOPS, np.arange(6), [(-1, 0)] * 6, ("material-0",)
    )
    source_scope = MeshingScope(
        "sheared-power", "r1", M.MeshingEntityKind.GEOMETRY, 2, "source", [5]
    )
    target_scope = MeshingScope(
        "sheared-power", "r1", M.MeshingEntityKind.GEOMETRY, 2, "target", [4]
    )
    matrix = np.eye(4)
    matrix[0, 3] = -1
    constraint = PeriodicConstraint(
        source_scope,
        target_scope,
        matrix,
        tolerance=0,
        source_entity_ids=np.asarray([5], dtype=np.int64),
        orientations=np.asarray([-1], dtype=np.int64),
    )
    construction = generate_polyhedral_volume(
        domain,
        sites=[[0, 0, 0.5], [0, 3, 0.5]],
        weights=[0, 7],
        periodic_constraints=[constraint],
    )
    volumes, moments = _independent_oriented_source_integrals(construction)
    assert sum(volumes) == 1
    assert tuple(sum(moment[axis] for moment in moments) for axis in range(3)) == (
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 2),
    )
    original = construction.mesh
    topology = original.periodic_topology
    assert topology is not None
    exact = construction.geometry.source_coordinates()
    chosen = next(
        index
        for index, point in enumerate(exact)
        if point[0] == Fraction(4, 3) and point[1] == Fraction(1, 3)
    )
    representatives = np.array(topology.vertex_representatives, copy=True)
    shifts = np.array(topology.vertex_shifts, copy=True)
    root = representatives[chosen]
    members = np.flatnonzero(representatives == root)
    origin = shifts[chosen].copy()
    representatives[members] = chosen
    shifts[members] -= origin
    assert isinstance(original.connectivity, PolyhedralConnectivity)
    assert isinstance(
        construction.geometry.exact_source,
        phx.discretization.ExactPowerCellGeometryLinearActionSource,
    )
    lifted = CellMesh(
        original.coordinates,
        original.blocks,
        vertex_global_ids=original.vertex_global_ids,
        polyhedral_connectivity=original.connectivity,
        numeric_version=original.numeric_version,
    )
    geometry = CellGeometrySpec.power(lifted, construction.geometry.exact_source)
    with pytest.raises(ValueError):
        PeriodicMeshTopology(lifted, topology.cell, representatives, shifts)
    renewed = PeriodicMeshTopology(
        lifted, topology.cell, representatives, shifts, actual_geometry=geometry
    )
    assert renewed.actual_geometry is not None
    for member in members:
        assert renewed.vertex_representatives[member] == chosen


def _assert_original_certificate_corruption_refuses(
    publication: CellMeshingResult,
) -> None:
    """Exercise pre-renewal authentication with real reports, without new queries."""
    from copy import copy

    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_source_families import (
        require_same_native_scientific_theorem,
    )

    original = publication.certification
    assert original is not None and original.embedding is not None

    def corrupt_path[T](value: T, path: tuple[str, ...], replacement: object) -> T:
        altered = copy(value)
        if len(path) == 1:
            object.__setattr__(altered, path[0], replacement)
        else:
            object.__setattr__(
                altered,
                path[0],
                corrupt_path(getattr(value, path[0]), path[1:], replacement),
            )
        return altered

    changes = []
    for name, maximum in (
        ("source_expression_work_units", original.request.limits.maximum_work_units),
        ("source_expression_peak_bytes", original.request.limits.maximum_scratch_bytes),
    ):
        changes.extend(
            (
                corrupt_path(original, ("embedding", name), -1),
                corrupt_path(original, ("embedding", name), maximum + 1),
            )
        )
    changes.extend(
        (
            corrupt_path(original, ("embedding", "certificate_id"), "0" * 64),
            corrupt_path(original, ("embedding", "binding", "binding_id"), "0" * 64),
            corrupt_path(original, ("embedding", "binding", "limits_id"), "0" * 64),
            corrupt_path(
                original, ("embedding", "cell_count"), original.embedding.cell_count + 1
            ),
            corrupt_path(original, ("request", "request_id"), "0" * 64),
            corrupt_path(original, ("report_id",), "0" * 64),
        )
    )
    outcomes = list(original.outcomes)
    index = next(
        i for i, value in enumerate(outcomes) if value.check == "global_embedding"
    )
    outcomes[index] = corrupt_path(outcomes[index], ("outcome_id",), "0" * 64)
    changes.append(corrupt_path(original, ("outcomes",), tuple(outcomes)))
    for altered in changes:
        with pytest.raises(ValueError):
            require_same_native_scientific_theorem(
                altered,
                original,
                publication.mesh,
                publication.geometry,
                publication.audit,
                limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
            )
    coordinates = np.asarray(publication.geometry.coordinates).copy()
    coordinates.view(np.uint64).reshape(-1)[0] ^= np.uint64(1)
    altered_geometry = corrupt_path(
        publication.geometry, ("coordinates",), jnp.asarray(coordinates)
    )
    with pytest.raises(ValueError):
        require_same_native_scientific_theorem(
            original,
            original,
            publication.mesh,
            altered_geometry,
            publication.audit,
            limits=DEFAULT_ARRAY_ARCHIVE_LIMITS,
        )


def _periodic_fv_public_inputs(
    action_kind: str,
) -> tuple[
    M.NativePolyhedralSource,
    M.VolumeMeshingSpec,
    M.NativeMeshingOptions,
    np.ndarray,
]:
    """The actual authored provider inputs shared by FV and contact diagnostics."""
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=float
    )
    domain = PiecewiseLinearComplex(
        points, _LOOPS, np.arange(6), [(-1, 0)] * 6, ("material",)
    )
    source_scope = M.MeshingScope(
        "polyhedral-domain", "r1", M.MeshingEntityKind.GEOMETRY, 2, "left", [4]
    )
    target_scope = M.MeshingScope(
        "polyhedral-domain",
        "r1",
        M.MeshingEntityKind.GEOMETRY,
        2,
        "target",
        [2 if action_kind == "rotation" else 5],
    )
    action = np.eye(4)
    action[0, 3] = 1
    if action_kind == "rotation":
        action = np.asarray(
            (
                (0.0, 1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
    elif action_kind == "screw":
        action[1, 1] = action[2, 2] = -1
        action[1:3, 3] = 1
    constraints = (
        M.PeriodicConstraint(
            source_scope,
            target_scope,
            action,
            tolerance=0,
            source_entity_ids=np.asarray([4], dtype=np.int64),
            orientations=np.asarray([-1], dtype=np.int64),
        ),
    )
    if action_kind == "translation":
        domain, constraints = _periodic_cube_source()
    source = M.NativePolyhedralSource(
        domain,
        "polyhedral-domain",
        "r1",
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        weights=np.zeros(2, dtype=np.float64),
    )
    root_scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "polyhedral-facets",
        np.arange(domain.facet_count),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("polyhedron",))),
        root_scope,
        M.VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            M.UniformSizeControl(root_scope, 10.0, strength=M.SizeControlStrength.SOFT),
        ),
        periodic_constraints=constraints,
    )
    return source, specification, M.NativeMeshingOptions("plc_restricted_power"), action


@pytest.mark.parametrize("action_kind", ["translation", "rotation", "screw"])
def test_native_periodic_polyhedral_fv_real_quotient_flux_and_adjoint(
    action_kind: str, tmp_path: Path
) -> None:
    import jax

    from phydrax.discretization import UnstructuredFiniteVolumePlan
    from phydrax.discretization.finite_volume._diffusion_boundary import (
        HybridDiffusionBoundary,
    )
    from phydrax.discretization.finite_volume._hybrid_diffusion import (
        HybridMimeticDiffusion,
    )

    native_policy = phx.linalg.LinearSolvePolicy(phx.linalg.ConjugateGradient())
    generation_source, specification, options, action = _periodic_fv_public_inputs(
        action_kind
    )
    domain = generation_source.complex
    constraints = specification.periodic_constraints
    construction = generate_polyhedral_volume(
        domain,
        sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]],
        weights=[0.0, 0.0],
        periodic_constraints=constraints,
        source_id="polyhedral-domain",
    )
    original_coordinates = np.asarray(construction.mesh.coordinates).copy()
    original_source = construction.geometry.source_coordinates()
    with pytest.raises(ValueError):
        prepare_polyhedral_finite_volume_geometry(
            construction.mesh,
            cell_geometry=construction.geometry,
            maximum_workset_entries=1,
        )
    np.testing.assert_array_equal(
        np.asarray(construction.mesh.coordinates).view(np.uint64),
        original_coordinates.view(np.uint64),
    )
    assert construction.geometry.source_coordinates() == original_source
    velocity = (0.0, 0.0, 1.0) if action_kind == "rotation" else (1.0, 0.0, 0.0)
    system = phx.equations.ScalarConservationSystem(
        3,
        lambda value, axis, args: velocity[axis] * value,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], abs(velocity[axis])),
        system_id="source-periodic-polyhedral-fv-advection",
    )
    prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
        construction.mesh,
        component_names=system.component_names,
    ).prepare(
        cell_geometry=construction.geometry,
    )
    assert construction.seam_face_pairs is not None
    pairs = construction.seam_face_pairs[:, :2]
    active = np.asarray(prepared.face_block.active_mask)
    for left, right in pairs:
        rows = np.flatnonzero(
            np.isin(np.asarray(prepared.source_face_indices), (left, right))
        )
        assert rows.size == 1
        assert bool(active[rows[0]])
        assert int(prepared.neighbor_cells[rows[0]]) >= 0
        assert int(prepared.boundary_patch_ids[rows[0]]) == -1
    volumes, moments = _independent_oriented_source_integrals(construction)
    np.testing.assert_allclose(
        prepared.cell_volumes, [float(value) for value in volumes], atol=1e-13
    )
    np.testing.assert_allclose(
        prepared.cell_centers,
        [
            [float(value / volume) for value in moment]
            for volume, moment in zip(volumes, moments, strict=True)
        ],
        atol=1e-13,
    )
    diffusion = HybridMimeticDiffusion(prepared)
    constant = jnp.ones((prepared.cell_count,))
    traces = jnp.ones((diffusion.face_count,))
    local = diffusion.local_fluxes(constant, traces, jnp.eye(3))
    np.testing.assert_allclose(local, 0, atol=1e-13)
    mask = active & (np.asarray(prepared.neighbor_cells) >= 0)
    rates = jnp.asarray(np.arange(diffusion.face_count, dtype=float) * mask)
    result = diffusion.cell_divergence(rates)
    np.testing.assert_allclose(jnp.sum(result), 0, atol=1e-13)
    direction = jnp.linspace(-0.5, 0.5, diffusion.face_count)
    cotangent = jnp.arange(prepared.cell_count, dtype=float) + 0.25
    _, tangent = jax.jvp(diffusion.cell_divergence, (rates,), (direction,))
    _, pullback = jax.vjp(diffusion.cell_divergence, rates)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(pullback(cotangent)[0], direction),
        atol=1e-13,
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        prepared.boundary_patch_names,
        {"boundary": phx.discretization.ExtrapolationBoundary()},
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
        system,
        prepared,
        method,
        boundaries,
    )
    state = jnp.ones(prepared.state_shape)
    history = []
    for step in range(4):
        residual = dynamics(jnp.asarray(step * 0.01), state)
        np.testing.assert_allclose(residual, 0, atol=1e-13)
        state = state + 0.01 * residual
        history.append(state)
    np.testing.assert_allclose(jnp.stack(history), 1, atol=1e-13)
    varying = jnp.arange(prepared.cell_count, dtype=float)[:, None] + 1
    residual = dynamics(jnp.asarray(0.0), varying)
    np.testing.assert_allclose(
        jnp.sum(prepared.cell_volumes[:, None] * residual), 0, atol=1e-13
    )
    manufactured_history = []
    if action_kind != "rotation":
        # Independent two-half-cell upwind balance: du/dt=-4(u-mean).
        assert prepared.cell_count == 2
        order = np.argsort(np.asarray(prepared.cell_centers)[:, 0])
        manufactured = jnp.zeros((2, 1)).at[order].set(jnp.asarray(((1.0,), (2.0,))))
        initial_mode = manufactured - 1.5
        np.testing.assert_allclose(
            dynamics(jnp.asarray(0.0), manufactured),
            -4 * initial_mode,
            atol=1e-12,
        )
        for step in range(4):
            manufactured = manufactured + 0.01 * dynamics(
                jnp.asarray(step * 0.01), manufactured
            )
            np.testing.assert_allclose(
                manufactured,
                1.5 + initial_mode * (1 - 4 * 0.01) ** (step + 1),
                atol=1e-12,
            )
            manufactured_history.append(manufactured)
        # The same source-frame mimetic operator gives the independent
        # two-half-cell diffusion mode du/dt=-16(u-mean), with no seam law.
        diffused = jnp.zeros((2,)).at[order].set(jnp.asarray((1.0, 2.0)))
        diffusion_mode = diffused - 1.5
        for step in range(4):
            owner_values = diffused[diffusion.owner_cells]
            neighbor_values = diffused[jnp.maximum(diffusion.neighbor_cells, 0)]
            face_values = jnp.where(
                diffusion.neighbor_cells >= 0,
                (owner_values + neighbor_values) / 2,
                owner_values,
            )
            face_rates = diffusion.face_fluxes(diffused, face_values, jnp.eye(3))
            rate = -diffusion.cell_divergence(face_rates) / prepared.cell_volumes
            np.testing.assert_allclose(rate, -16 * (diffused - 1.5), atol=1e-12)
            diffused = diffused + 0.01 * rate
            np.testing.assert_allclose(
                diffused, 1.5 + diffusion_mode * (1 - 16 * 0.01) ** (step + 1), atol=1e-12
            )
        neumann = HybridDiffusionBoundary(prepared)
        # Authored piecewise physical forcing -u''=(-.25,+.25), integrated
        # over the two exact half cells. It is not manufactured from A*u.
        loads = jnp.zeros((2,)).at[order].set(jnp.asarray((-0.125, 0.125)))
        pinned_system, pinned_rhs = diffusion.pinned_linear_system(
            jnp.eye(3),
            neumann,
            source=loads,
            pin_cell=int(order[0]),
            pin_value=0.0,
        )
        solved = phx.linalg.solve(
            pinned_system,
            pinned_rhs,
            policy=native_policy,
        )
        assert bool(solved.successful)
        np.testing.assert_allclose(solved.value[order[0]], 0, atol=1e-12)
        physical_residual = diffusion.residual(
            solved.value[:2],
            solved.value[2:],
            jnp.eye(3),
            neumann,
            source=loads,
        )
        native_residual = pinned_system.operator.mv(solved.value) - pinned_rhs
        native_limit = (
            native_policy.tolerance.absolute
            + native_policy.tolerance.relative * jnp.linalg.norm(pinned_rhs)
        )
        assert float(jnp.linalg.norm(native_residual)) <= float(native_limit)
        # Exact source compatibility and physical flux cancellation recover the
        # removed row: r_pin=-sum(r_free). Hence ||r||₂≤sqrt(n)||r_free||₂.
        np.testing.assert_allclose(jnp.sum(loads), 0, atol=0)
        np.testing.assert_allclose(jnp.sum(physical_residual), 0, atol=1e-13)
        free_rows = np.arange(physical_residual.size) != order[0]
        np.testing.assert_allclose(
            physical_residual[free_rows], native_residual[free_rows], atol=1e-12
        )
        assert float(jnp.linalg.norm(physical_residual)) <= float(
            jnp.sqrt(physical_residual.size) * native_limit
        )
        # The independently integrated piecewise quadratic differs between
        # quarter/three-quarter centers by .25/16; the FV error stays below .05.
        assert (
            abs(float(solved.value[order[1]] - solved.value[order[0]]) - 0.25 / 16) < 0.05
        )
    trace_boundary = HybridDiffusionBoundary(prepared)
    trace_system, cell_rhs_action, trace_offset = (
        diffusion.trace_reconstruction_operators(
            jnp.eye(3),
            trace_boundary,
        )
    )
    trace_prepared = phx.linalg.prepare(
        trace_system,
        native_policy,
    )
    actual_histories = manufactured_history[:3] if manufactured_history else history[:3]
    for cell_values in (
        varying[:, 0],
        *(value[:, 0] for value in actual_histories),
        jnp.ones(prepared.cell_count),
    ):
        actual_rhs = cell_rhs_action.mv(cell_values) + trace_offset
        trace_solved = phx.linalg.solve(
            trace_prepared,
            actual_rhs,
        )
        assert bool(trace_solved.successful)
        local_flux = diffusion.local_fluxes(cell_values, trace_solved.value, jnp.eye(3))
        physical_face_residual = trace_boundary.face_residual(
            trace_solved.value,
            diffusion.continuity_residual(local_flux),
        )
        native_face_residual = trace_system.operator.mv(trace_solved.value) - actual_rhs
        np.testing.assert_allclose(
            physical_face_residual, native_face_residual, atol=1e-12
        )
        native_limit = (
            native_policy.tolerance.absolute
            + native_policy.tolerance.relative * jnp.linalg.norm(actual_rhs)
        )
        assert float(jnp.linalg.norm(physical_face_residual)) <= float(native_limit)
    one = jnp.ones(diffusion.face_count)
    np.testing.assert_allclose(
        trace_system.operator.mv(one),
        cell_rhs_action.mv(jnp.ones(prepared.cell_count)) + trace_offset,
        atol=1e-12,
    )
    assert float(jnp.max(jnp.abs(trace_solved.value - one))) < 0.05
    face_cotangent = jnp.linspace(-1.0, 1.0, diffusion.face_count)
    cell_direction = jnp.linspace(0.2, 0.7, prepared.cell_count)
    np.testing.assert_allclose(
        jnp.vdot(face_cotangent, cell_rhs_action.mv(cell_direction)),
        jnp.vdot(cell_rhs_action.transpose_mv(face_cotangent), cell_direction),
        atol=1e-12,
    )

    # Conserved momentum is a genuine system-owned bundle representation.
    euler = phx.equations.EulerSystem(3)
    vector_prepared = UnstructuredFiniteVolumePlan.from_cell_mesh(
        construction.mesh,
        component_names=euler.component_names,
    ).prepare(cell_geometry=construction.geometry)
    vector_dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
        euler,
        vector_prepared,
        method,
        boundaries,
    )
    conserved_constant = (
        (1.0, 0.0, 0.0, 0.2, 3.0)
        if action_kind == "rotation"
        else (1.0, 0.2, 0.0, 0.0, 3.0)
    )
    vector_state = jnp.broadcast_to(
        jnp.asarray(conserved_constant), vector_prepared.state_shape
    )
    np.testing.assert_allclose(
        vector_dynamics(jnp.asarray(0.0), vector_state), 0, atol=1e-12
    )
    action_matrix = euler.proper_isometry_state_matrix(jnp.asarray(action[:3, :3]))
    np.testing.assert_allclose(action_matrix[0], (1.0, 0.0, 0.0, 0.0, 0.0), atol=0)
    np.testing.assert_allclose(action_matrix[-1], (0.0, 0.0, 0.0, 0.0, 1.0), atol=0)
    np.testing.assert_allclose(action_matrix[1:4, 1:4], action[:3, :3], atol=0)
    flux, _ = vector_dynamics.face_fluxes(jnp.asarray(0.0), vector_state)
    expected = vector_dynamics._content_rate_from_fluxes(flux)
    from phydrax.discretization._conservation_ledger import ConservationStageLedger

    block = vector_dynamics.stage_rate_block_templates[0].with_flux_rate(
        flux * vector_prepared.face_measures[:, None],
    )
    ledger = ConservationStageLedger(
        [block],
        jnp.zeros_like(vector_state),
        jnp.ones((vector_prepared.cell_count,), dtype=bool),
        geometry_family_id=vector_prepared.geometry_id,
        geometry_layout_id=vector_prepared.topology_id,
        geometry_version=jnp.asarray(0),
        evidence_policy_id=vector_prepared.geometry_id,
        evidence_version=jnp.asarray(0),
        topology_epoch_id=vector_prepared.topology_id,
    )
    np.testing.assert_allclose(ledger.scatter_content_rate(), expected, atol=1e-12)
    source_sum, boundary_sum, net_sum = ledger.conservation_sums()
    np.testing.assert_allclose(
        net_sum, source_sum - boundary_sum + ledger.frame_exchange_sum(), atol=1e-12
    )
    raw_direction = jnp.arange(flux.size, dtype=float).reshape(flux.shape) / flux.size
    cell_cotangent = (
        jnp.arange(vector_state.size, dtype=float).reshape(vector_state.shape)
        / vector_state.size
    )
    scatter = lambda values: vector_dynamics._content_rate_from_fluxes(values)
    _, tangent = jax.jvp(scatter, (flux,), (raw_direction,))
    _, pullback = jax.vjp(scatter, flux)
    np.testing.assert_allclose(
        jnp.vdot(cell_cotangent, tangent),
        jnp.vdot(pullback(cell_cotangent)[0], raw_direction),
        atol=1e-12,
    )
    vector_history = []
    for step in range(4):
        vector_state = vector_state + 0.01 * vector_dynamics(
            jnp.asarray(step * 0.01), vector_state
        )
        vector_history.append(vector_state)
    import subprocess
    import sys

    from phydrax._differentiation import BranchDifferentiationPolicy
    from phydrax._frozendict import frozendict
    from phydrax.discretization._views import FieldTracePolicy
    from phydrax.lifecycle._meshing_field_records import (
        MeshingFieldDeclaration,
        MeshingFieldStateRole,
        prepare_meshing_field_index_binding,
    )
    from phydrax.lifecycle._meshing_sources import write_meshing_source_closure

    # The cold root is the actual public certified publication, not an arbitrary
    # mesh/array dictionary or a fabricated report around the direct constructor.
    publication = (
        M.NativeMeshingProvider(options)
        .plan(
            generation_source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert publication.certification is not None and publication.certification.passed
    np.testing.assert_array_equal(
        publication.mesh.entity_set(3).entity_ids,
        construction.mesh.entity_set(3).entity_ids,
    )
    assert publication.geometry.source_coordinates() == original_source
    _assert_original_certificate_corruption_refuses(publication)
    restored_owner = UnstructuredFiniteVolumePlan.from_cell_mesh(
        publication.mesh,
        component_names=euler.component_names,
    ).prepare(cell_geometry=publication.geometry)
    np.testing.assert_array_equal(
        restored_owner.cell_centers, vector_prepared.cell_centers
    )
    binding = prepare_meshing_field_index_binding(
        publication,
        "field/global_coefficient_ids",
        field_space=restored_owner.cell_space,
    )
    arrays = {
        "vector_state": vector_state,
        "vector_history": jnp.moveaxis(jnp.stack(vector_history), 0, 1),
        "scalar_state": state,
        "scalar_history": jnp.moveaxis(jnp.stack(history), 0, 1),
        "density": jnp.ones((prepared.cell_count,)),
        "state_epoch": jnp.asarray(4, dtype=jnp.int64),
    }
    if manufactured_history:
        arrays["manufactured_history"] = jnp.moveaxis(
            jnp.stack(manufactured_history), 0, 1
        )
    density_unit = phx.units.derived_unit(
        "kg/m^3", ((phx.units.KILOGRAM, 1), (phx.units.METER, -3))
    )
    momentum_unit = phx.units.derived_unit(
        "kg/(m^2*s)",
        ((phx.units.KILOGRAM, 1), (phx.units.METER, -2), (phx.units.SECOND, -1)),
    )
    units = (density_unit, momentum_unit, momentum_unit, momentum_unit, phx.units.PASCAL)
    roles = tuple(
        MeshingFieldStateRole(
            name,
            "state",
            "coefficients"
            if name == "vector_state"
            else "stage-history"
            if name.endswith("history")
            else "material-history",
            tuple(value.shape),
            np.dtype(value.dtype),
            4,
            units
            if name in ("vector_state", "vector_history")
            else (density_unit,)
            if name == "density"
            else (phx.units.ONE,),
            index_binding=binding,
        )
        for name, value in arrays.items()
        if name != "state_epoch"
    ) + (
        MeshingFieldStateRole(
            "state_epoch", "state", "state-epoch", (), np.dtype("int64"), 4
        ),
    )
    reconstruction = phx.discretization.PiecewiseConstantReconstruction()
    declaration = MeshingFieldDeclaration(
        part_name="generation",
        owner="finite_volume",
        topology_id=publication.mesh.topology_id,
        geometry_layout_id=publication.geometry.geometry_layout_id,
        field_space_ids=frozendict({"state": restored_owner.cell_space.field_space_id}),
        value_units=frozendict({"state": units}),
        maximum_derivative_orders=frozendict({"state": 0}),
        branch_policy=BranchDifferentiationPolicy.SMOOTH,
        trace_policy=FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version=restored_owner.numeric_version,
        finite_volume_field_name="state",
        finite_volume_component_names=euler.component_names,
        finite_volume_coordinate_policy="accepted-map",
        reconstruction_policy=reconstruction,
        reconstruction_policy_id=reconstruction.plan_id,
        finite_volume_system=euler,
        finite_volume_flux_policy=phx.discretization.RusanovFluxPlan(),
        finite_volume_boundaries=frozendict(
            {
                name: phx.discretization.ExtrapolationBoundary()
                for name in restored_owner.boundary_patch_names
            }
        ),
    )
    report = publication.certification
    receipt = write_meshing_source_closure(
        tmp_path / "periodic-fv",
        {
            "certification_inputs": report.request,
            "report": report,
            "associations": publication.associations,
            "generation_part": M.MeshPart("generation", publication),
            "generation_source": generation_source,
            "generation_specification": specification,
            "generation_options": options,
            "accepted_data": {"fields": arrays},
            "field_declarations": {"generation": declaration},
        },
    )
    cold = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy as np, jax.numpy as jnp; "
            "from phydrax.lifecycle._meshing_sources import read_meshing_source_closure,recertify_restored_meshing_source; "
            "from phydrax.lifecycle._meshing_field_records import prepare_meshing_field_owner,prepare_meshing_field_dynamics; "
            "from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS; "
            "from phydrax.lifecycle._meshing_source_families import require_same_native_scientific_theorem; "
            "from phydrax.discretization.finite_volume._hybrid_diffusion import HybridMimeticDiffusion; "
            "v=read_meshing_source_closure(sys.argv[1],expected_content_id=sys.argv[2]); "
            "p=v['generation_part'].carrier; "
            "r=recertify_restored_meshing_source(v['certification_inputs'],p.mesh,p.geometry,p.audit); "
            "old,current=require_same_native_scientific_theorem(v['report'],r,p.mesh,p.geometry,p.audit,limits=DEFAULT_ARRAY_ARCHIVE_LIMITS); "
            "assert old is v['report'] and current is r; "
            "decl=v['field_declarations']['generation']; "
            "d,_=prepare_meshing_field_owner(decl,p); f=prepare_meshing_field_dynamics(decl,p); "
            "a=v['accepted_data']['fields']; h=HybridMimeticDiffusion(d); "
            "np.testing.assert_allclose(h.local_fluxes(a['scalar_state'][:,0],jnp.ones(h.face_count),jnp.eye(3)),0,atol=1e-13); "
            "np.testing.assert_array_equal(a['scalar_history'],np.ones((d.cell_count,4,1))); "
            "np.testing.assert_array_equal(a['density'],np.ones(d.cell_count)); "
            "assert d.face_measures.size<p.mesh.connectivity.face_count; "
            "np.testing.assert_allclose(f(jnp.asarray(.04),a['vector_state']),0,atol=1e-12); "
            "np.testing.assert_allclose(a['vector_history'],np.broadcast_to(a['vector_state'][:,None,:],a['vector_history'].shape),atol=1e-12)",
            str(tmp_path / "periodic-fv"),
            receipt.content_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert cold.returncode == 0, cold.stdout + cold.stderr

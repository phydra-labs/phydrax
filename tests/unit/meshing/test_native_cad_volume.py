from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.geometry._approximate_source_coverage import (
    _cad_chart_vertex_key,
    cad_material_sides,
)
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._domain_boundary import (
    exact_plane_chart,
    MeshingDomainBoundarySource,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry._surface_source_support import SurfaceSourceCharts
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cylinder,
    brep_sphere,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._projection_contracts import brep_entity_id
from phydrax.meshing._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    FeatureKind,
    ProtectedFeature,
    RegionControl,
    RegionSeed,
)
from phydrax.meshing._model_codec import write_model_artifact
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule, VolumeConstruction
from phydrax.meshing.providers import _native_cad_volume as cad_volume
from phydrax.meshing.providers._native_cad_volume import (
    NativeCadVolumeSource,
    prepare_native_cad_volume,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions


pytestmark = pytest.mark.meshcore


def _cad_surface() -> tuple[MeshingDomain, CellMeshingResult]:
    contract = phx.SpatialCoordinateContract.si()
    model = brep_box(
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((1.0, 1.0, 1.0), dtype=np.float64),
        coordinate_contract=contract,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.05,
            angular_deflection=0.1,
            trim_samples_per_edge=3,
        ),
        source_id="native-cad-volume-box",
    )
    tessellation = model.tessellation
    assert tessellation is not None
    domain = MeshingDomain.from_brep(model)
    points = np.asarray(tessellation.vertices, dtype=np.float64)
    triangles = np.asarray(tessellation.triangles, dtype=np.int64)
    face_indices = np.asarray(tessellation.face_ids, dtype=np.int64)
    vertex_dimensions = np.asarray(tessellation.vertex_source_dimensions, dtype=np.int64)
    vertex_indices = np.asarray(tessellation.vertex_source_indices, dtype=np.int64)
    vertex_parameters = np.asarray(tessellation.vertex_parameters, dtype=np.float64)
    mesh = phx.discretization.CellMesh.from_triangles(
        points,
        triangles,
        vertex_global_ids=np.arange(points.shape[0], dtype=np.int64),
        cell_global_ids=np.arange(triangles.shape[0], dtype=np.int64),
        numeric_version="native-cad-volume-surface",
    )
    source_id = model.source_id
    revision = model.source_revision
    source_paths_by_dimension: list[dict[int, tuple[str, ...]]] = []
    for dimension in range(3):
        indices = np.asarray(domain.source_indices[dimension], dtype=np.int64)
        paths = domain.source_occurrences[dimension]
        path_by_index = {
            int(index): tuple(path)
            for index, path in zip(indices.tolist(), paths, strict=True)
        }
        if len(path_by_index) != indices.size:
            raise ValueError("The BRep test source has ambiguous entity occurrences.")
        source_paths_by_dimension.append(path_by_index)
    vertex_occurrence_paths = tuple(
        source_paths_by_dimension[int(dimension)][int(index)]
        for dimension, index in zip(
            vertex_dimensions.tolist(), vertex_indices.tolist(), strict=True
        )
    )
    vertex_entity_ids = tuple(
        brep_entity_id(
            source_id,
            revision,
            int(dimension),
            int(index),
            occurrence_path=path,
        )
        for dimension, index, path in zip(
            vertex_dimensions.tolist(),
            vertex_indices.tolist(),
            vertex_occurrence_paths,
            strict=True,
        )
    )
    vertex_association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source_id,
        revision,
        mesh.entity_set(0).entity_set_id,
        np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64),
        vertex_entity_ids,
        np.zeros((points.shape[0],), dtype=np.float64),
        exact=True,
        source_dimensions=vertex_dimensions.astype(np.int8),
        source_indices=vertex_indices.astype(np.int32),
        source_roles=tuple(
            {
                0: GeometrySourceEntityRole.VERTEX,
                1: GeometrySourceEntityRole.EDGE,
                2: GeometrySourceEntityRole.FACET,
            }[int(dimension)]
            for dimension in vertex_dimensions
        ),
        source_occurrence_paths=vertex_occurrence_paths,
        parameters=vertex_parameters,
    )

    edge_faces: dict[tuple[int, int], set[int]] = {}
    for face, triangle in enumerate(triangles.tolist()):
        for first, second in combinations(triangle, 2):
            edge_faces.setdefault((min(first, second), max(first, second)), set()).add(
                face
            )
    source_curve_by_vertex_pair: dict[tuple[int, int], int] = {}
    source_edge_vertices = np.asarray(model.geometry.edge_vertices, dtype=np.int64)
    curve_row_by_source = {
        int(source_index): row
        for row, source_index in enumerate(domain.source_indices[1])
    }
    for edge, vertices in enumerate(source_edge_vertices.tolist()):
        first_vertex, second_vertex = int(vertices[0]), int(vertices[1])
        source_curve_by_vertex_pair[
            (min(first_vertex, second_vertex), max(first_vertex, second_vertex))
        ] = edge
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        mesh_edges = np.asarray(connectivity.entities[1], dtype=np.int64)
    elif isinstance(connectivity, PolygonalConnectivity):
        mesh_edges = np.asarray(connectivity.edges, dtype=np.int64)
    else:
        raise TypeError("BRep surface mesh lacks canonical triangle edge incidence.")
    mesh_edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_entity_ids: list[str] = []
    edge_dimensions: list[int] = []
    edge_indices: list[int] = []
    edge_orientations: list[int] = []
    edge_paths: list[tuple[str, ...]] = []
    edge_roles: list[GeometrySourceEntityRole] = []
    edge_parameter_intervals: list[tuple[Fraction, Fraction] | None] = []
    edge_residuals = np.zeros((mesh_edges.shape[0],), dtype=np.float64)
    edge_set_rows = {
        (
            min(int(vertices[0]), int(vertices[1])),
            max(int(vertices[0]), int(vertices[1])),
        ): row
        for row, vertices in enumerate(mesh_edges.tolist())
    }

    def curve_parameter(vertex_row: int, curve_index: int) -> float:
        dimension = int(vertex_dimensions[vertex_row])
        source_index = int(vertex_indices[vertex_row])
        if dimension == 1 and source_index == curve_index:
            return float(vertex_parameters[vertex_row, 0])
        if dimension == 0:
            endpoints = source_edge_vertices[curve_index]
            parameter_range = np.asarray(model.geometry.edge_ranges)[curve_index]
            if source_index == endpoints[0]:
                return float(parameter_range[0])
            if source_index == endpoints[1]:
                return float(parameter_range[1])
        raise ValueError("A tessellated curve edge lost its exact endpoint parameters.")

    for edge_key, edge_row in edge_set_rows.items():
        first, second = (int(value) for value in mesh_edges[edge_row])
        first_dimension = int(vertex_dimensions[first])
        second_dimension = int(vertex_dimensions[second])
        first_source = int(vertex_indices[first])
        second_source = int(vertex_indices[second])
        curve_index = None
        if first_dimension == second_dimension == 1 and first_source == second_source:
            curve_index = first_source
        elif first_dimension == second_dimension == 0:
            curve_index = source_curve_by_vertex_pair.get(
                (min(first_source, second_source), max(first_source, second_source))
            )
        elif first_dimension == 0 and second_dimension == 1:
            if first_source in source_edge_vertices[second_source]:
                curve_index = second_source
        elif first_dimension == 1 and second_dimension == 0:
            if second_source in source_edge_vertices[first_source]:
                curve_index = first_source
        if curve_index is not None:
            curve_path = source_paths_by_dimension[1][int(curve_index)]
            edge_dimensions.append(1)
            edge_indices.append(int(curve_index))
            edge_entity_ids.append(
                brep_entity_id(
                    source_id,
                    revision,
                    1,
                    int(curve_index),
                    occurrence_path=curve_path,
                )
            )
            edge_roles.append(GeometrySourceEntityRole.EDGE)
            edge_paths.append(curve_path)
            first_parameter = curve_parameter(first, int(curve_index))
            second_parameter = curve_parameter(second, int(curve_index))
            curve_row = curve_row_by_source[int(curve_index)]
            patch, loop, position = domain.curve_owners[curve_row]
            owner = domain.patches[patch].loops[loop][position]
            owner_first = Fraction.from_float(float(owner.first))
            owner_last = Fraction.from_float(float(owner.last))
            first_exact = (Fraction.from_float(first_parameter) - owner_first) / (
                owner_last - owner_first
            )
            second_exact = (Fraction.from_float(second_parameter) - owner_first) / (
                owner_last - owner_first
            )
            edge_parameter_intervals.append(
                (first_exact, second_exact)
                if first_exact < second_exact
                else (second_exact, first_exact)
            )
            edge_orientations.append(1 if second_parameter > first_parameter else -1)
        else:
            incident_faces = edge_faces[edge_key]
            incident_face_indices = {int(face_indices[face]) for face in incident_faces}
            if len(incident_face_indices) != 1:
                endpoint_sources = tuple(
                    (
                        int(vertex_dimensions[vertex]),
                        int(vertex_indices[vertex]),
                    )
                    for vertex in (first, second)
                )
                raise ValueError(
                    f"Mesh edge {edge_key} spans source faces "
                    f"{tuple(sorted(incident_face_indices))} but endpoint witnesses "
                    f"{endpoint_sources} do not identify one exact BRep curve."
                )
            face_index = next(iter(incident_face_indices))
            face_path = source_paths_by_dimension[2][face_index]
            edge_dimensions.append(2)
            edge_indices.append(face_index)
            edge_entity_ids.append(
                brep_entity_id(
                    source_id, revision, 2, face_index, occurrence_path=face_path
                )
            )
            edge_roles.append(GeometrySourceEntityRole.FACET)
            edge_paths.append(face_path)
            edge_orientations.append(0)
            edge_parameter_intervals.append(None)
    curve_rows = np.flatnonzero(np.asarray(edge_dimensions, dtype=np.int64) == 1)
    edge_association = GeometryAssociation(
        GeometryAssociationKind.CURVE,
        source_id,
        revision,
        mesh.entity_set(1).entity_set_id,
        mesh_edge_ids[curve_rows],
        tuple(edge_entity_ids[int(row)] for row in curve_rows.tolist()),
        edge_residuals[curve_rows],
        exact=False,
        source_dimensions=np.asarray(edge_dimensions, dtype=np.int8)[curve_rows],
        source_indices=np.asarray(edge_indices, dtype=np.int32)[curve_rows],
        source_roles=tuple(edge_roles[int(row)] for row in curve_rows.tolist()),
        source_occurrence_paths=tuple(
            edge_paths[int(row)] for row in curve_rows.tolist()
        ),
        source_parameter_intervals=tuple(
            edge_parameter_intervals[int(row)] for row in curve_rows.tolist()
        ),
        orientations=np.asarray(edge_orientations, dtype=np.int8)[curve_rows],
    )
    face_global_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
        )
    }
    face_source_indices = np.asarray(
        [int(face_indices[face_rows[int(identifier)]]) for identifier in face_global_ids],
        dtype=np.int32,
    )
    face_source_paths = tuple(
        source_paths_by_dimension[2][int(index)] for index in face_source_indices.tolist()
    )
    face_sources = tuple(
        brep_entity_id(source_id, revision, 2, int(index), occurrence_path=path)
        for index, path in zip(
            face_source_indices.tolist(), face_source_paths, strict=True
        )
    )
    patch_by_source_index = {
        int(source_index): patch
        for patch, source_index in enumerate(domain.source_indices[2])
    }
    root_patches = np.asarray(
        [patch_by_source_index[int(face)] for face in face_indices], dtype=np.int64
    )
    root_parameters = np.asarray(tessellation.parameters, dtype=np.float64)
    first_edges = root_parameters[:, 1] - root_parameters[:, 0]
    second_edges = root_parameters[:, 2] - root_parameters[:, 0]
    root_orientations = np.where(
        first_edges[:, 0] * second_edges[:, 1] - first_edges[:, 1] * second_edges[:, 0]
        > 0.0,
        1,
        -1,
    )
    face_orientations_by_cell = np.asarray(
        [
            int(domain.patches[int(patch)].orientation) * int(orientation)
            for patch, orientation in zip(root_patches, root_orientations, strict=True)
        ],
        dtype=np.int8,
    )
    face_orientations = np.asarray(
        [
            face_orientations_by_cell[face_rows[int(identifier)]]
            for identifier in face_global_ids
        ],
        dtype=np.int8,
    )
    face_parameters = np.asarray(tessellation.parameters, dtype=np.float64).mean(axis=1)
    face_parameters_by_id = {
        int(identifier): face_parameters[row]
        for row, identifier in enumerate(np.asarray(mesh.blocks[0].global_ids))
    }
    face_association = GeometryAssociation(
        GeometryAssociationKind.BREP,
        source_id,
        revision,
        mesh.entity_set(2).entity_set_id,
        face_global_ids,
        face_sources,
        np.zeros((face_global_ids.size,), dtype=np.float64),
        exact=True,
        source_dimensions=np.full((face_global_ids.size,), 2, dtype=np.int8),
        source_indices=face_source_indices,
        source_roles=(GeometrySourceEntityRole.FACET,) * face_global_ids.size,
        source_occurrence_paths=face_source_paths,
        parameters=np.asarray(
            [face_parameters_by_id[int(identifier)] for identifier in face_global_ids],
            dtype=np.float64,
        ),
        orientations=face_orientations,
    )
    source_scope_ids_by_key = {
        domain.source_entity_key_for_scope(2, int(scope_id)): int(scope_id)
        for scope_id in domain.scope_entity_ids(2)
    }
    patch_scope_ids = np.asarray(
        [
            source_scope_ids_by_key[
                (domain.entity_id(2, patch), domain.source_occurrences[2][patch])
            ]
            for patch in range(len(domain.patches))
        ],
        dtype=np.int64,
    )
    source_root_cell_ids = patch_scope_ids[root_patches]
    source_piece_ids = tuple(
        (int(patch), int(domain.source_indices[2][int(patch)])) for patch in root_patches
    )
    boundary_source = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches)))
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    surface_source = SurfaceSourceCharts(
        domain,
        mesh,
        geometry,
        contract,
        root_parameters,
        root_patches,
        (vertex_association, edge_association, face_association),
        boundary_source,
        source_root_cell_ids=source_root_cell_ids,
        source_piece_ids=source_piece_ids,
    )
    surface = phx.meshing.certify_cell_mesh(
        mesh,
        contract,
        geometry=geometry,
        associations=(vertex_association, edge_association, face_association),
        surface_source=surface_source,
    )
    return domain, surface


def test_surface_source_projected_recipe_reconstructs_canonical_owners() -> None:
    from phydrax._model._structure import model_archive_projection
    from phydrax.discretization._cell_geometry_transition_types import cell_geometry_id
    from phydrax.geometry._surface_source_archive import restore_surface_source_charts

    _, surface = _cad_surface()
    source = surface.surface_source
    assert source is not None
    recipe = model_archive_projection(source)
    restored = restore_surface_source_charts(recipe)
    assert restored.source_chart_id == source.source_chart_id
    assert restored.mesh.mesh_id == source.mesh.mesh_id
    assert cell_geometry_id(restored.geometry) == cell_geometry_id(source.geometry)
    assert tuple(item.association_id for item in restored.associations) == tuple(
        item.association_id for item in source.associations
    )
    restored.require_current()
    with pytest.raises(TypeError, match="CellGeometrySpec"):
        restore_surface_source_charts(recipe._replace(geometry=None))


def test_exact_plane_trim_does_not_use_authored_chart_junctions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _domain_boundary

    domain = MeshingDomain.from_brep(
        brep_cylinder(0.5, 1.0, coordinate_contract=phx.SpatialCoordinateContract.si())
    )
    patch, plane = next(
        (patch, chart)
        for patch in range(len(domain.patches))
        if (chart := exact_plane_chart(domain, patch)) is not None
    )

    def reject_authored(*args: object) -> None:
        raise AssertionError("Alternate chart must not inspect authored UV roots.")

    monkeypatch.setattr(
        _domain_boundary, "_rooted_patch_chart_junctions", reject_authored
    )
    assert (
        _domain_boundary._patch_chart_polygon(domain, patch, trim_loops=plane.loops)
        is not None
    )
    with pytest.raises(AssertionError, match="authored UV roots"):
        _domain_boundary._patch_chart_polygon(domain, patch)


def _parametric_surface_charts(model: BRepModel, /) -> SurfaceSourceCharts:
    contract = phx.SpatialCoordinateContract.si()
    domain = MeshingDomain.from_brep(model)
    scope = MeshingScope.whole(model, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.3),),
        maximum_deviation=0.05,
    )
    result = (
        phx.meshing.NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
        .plan(domain, specification, coordinate_contract=contract)
        .execute()
    )
    assert isinstance(result.surface_source, SurfaceSourceCharts)
    return result.surface_source


@pytest.fixture(scope="module")
def cylinder_charts() -> SurfaceSourceCharts:
    contract = phx.SpatialCoordinateContract.si()
    return _parametric_surface_charts(
        brep_cylinder(0.5, 1.0, coordinate_contract=contract)
    )


@pytest.fixture(scope="module")
def sphere_charts() -> SurfaceSourceCharts:
    # Each pole side is the rooted latitude AffinePCurve(ConicRootLinePCurve),
    # not a straight chart line.
    contract = phx.SpatialCoordinateContract.si()
    return _parametric_surface_charts(brep_sphere(0.5, coordinate_contract=contract))


@pytest.mark.parametrize("charts_fixture", ("cylinder_charts", "sphere_charts"))
def test_cad_pole_lens_uses_exact_collapsed_corner_only(
    charts_fixture: str, request: pytest.FixtureRequest
) -> None:
    from phydrax.geometry._domain_boundary import (
        _is_rooted_pole_line,
        _rooted_patch_chart_endpoints,
    )
    from phydrax.geometry._meshing_domain import PatchCurveUse, PatchPoleUse
    from phydrax.geometry._surface_source_support import _cad_pole_lens_endpoint
    from phydrax.geometry.brep._patches import ExactLineCurve, LineCurve
    from phydrax.geometry.brep._source_shell_admissibility import _single_shell_topology

    charts = request.getfixturevalue(charts_fixture)
    assert isinstance(charts, SurfaceSourceCharts)
    domain = charts.domain
    assert domain.brep_authority is not None
    assert _single_shell_topology(domain.brep_authority, domain)
    for patch, surface in enumerate(domain.patches):
        for loop_index, uses in enumerate(surface.loops):
            for index, use in enumerate(uses):
                if not isinstance(use, PatchCurveUse):
                    continue
                previous = (index - 1) % len(uses)
                pole = uses[previous]
                if not isinstance(pole, PatchPoleUse):
                    continue
                assert (charts_fixture == "sphere_charts") is (
                    _is_rooted_pole_line(pole)
                    and not isinstance(pole.pcurve, (ExactLineCurve, LineCurve))
                )
                endpoints = _rooted_patch_chart_endpoints(domain, patch)[loop_index]
                start, stop = endpoints[previous]
                midpoint = ((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2)
                parameter = Fraction(0 if use.sense > 0 else 1)
                cache = {}
                for candidate, candidate_parameter in (
                    (midpoint, parameter),
                    ((midpoint[0], midpoint[1] + 1), parameter),
                    (midpoint, Fraction(1, 2)),
                ):
                    assert _cad_pole_lens_endpoint(
                        charts, patch, loop_index, index, candidate, candidate_parameter,
                        endpoints_by_patch=cache,
                    ) == _cad_pole_lens_endpoint(
                        charts, patch, loop_index, index, candidate, candidate_parameter,
                    )
                assert cache[patch] == _rooted_patch_chart_endpoints(domain, patch)
                for candidate_cache in (None, {}):
                    with pytest.raises(IndexError):
                        _cad_pole_lens_endpoint(
                            charts, patch, len(surface.loops), index, midpoint, parameter,
                            endpoints_by_patch=candidate_cache,
                        )
                junction = endpoints[index][0]
                assert (
                    _cad_pole_lens_endpoint(
                        charts, patch, loop_index, index, midpoint, parameter
                    )
                    == junction
                )
                off_side = (midpoint[0], midpoint[1] + 1)
                assert (
                    _cad_pole_lens_endpoint(
                        charts, patch, loop_index, index, off_side, parameter
                    )
                    == off_side
                )
                assert (
                    _cad_pole_lens_endpoint(
                        charts, patch, loop_index, index, midpoint, Fraction(1, 2)
                    )
                    == midpoint
                )
                return
    pytest.fail("The source has no regular coedge following a pole side.")


@pytest.mark.parametrize("reverse", (False, True))
def test_cad_plane_seam_side_uses_oriented_source_chart(reverse: bool) -> None:
    from phydrax.geometry._domain_boundary import _TURN, ExactPlaneChart
    from phydrax.geometry._surface_source_support import _cad_plane_seam_side

    zero, one, two = Fraction(0), Fraction(1), Fraction(2)
    turn = Fraction.from_float(_TURN)
    plane = ExactPlaneChart(
        "plane",
        (zero, zero, zero),
        (one, zero, zero),
        (zero, one, zero),
        (),
        zero,
        1,
    )
    target = ((one, zero), (two, zero), (one, -one))
    edge = (1, 0) if reverse else (0, 1)
    parameters = (one, zero) if reverse else (zero, one)
    for angle, interior, expected in ((zero, one, True), (turn, turn - one, False)):
        root = ((interior, one), (angle, one), (angle, two))
        assert (
            _cad_plane_seam_side(
                plane,
                root,
                0,
                (root[1], root[2]),
                target,
                edge,
                (zero, one),
                parameters,
            )
            is expected
        )
    other_side = ((one, zero), (two, zero), (one, one))
    root = ((turn - one, one), (turn, one), (turn, two))
    assert _cad_plane_seam_side(
        plane,
        root,
        0,
        (root[1], root[2]),
        other_side,
        edge,
        (zero, one),
        parameters,
    )
    pole_root = ((one, one), (zero, zero), (zero, one))
    pole_target = ((zero, zero), (one, zero), (one, -one))
    assert _cad_plane_seam_side(
        plane,
        pole_root,
        0,
        (pole_root[1], pole_root[2]),
        pole_target,
        edge,
        (zero, one),
        parameters,
    )


@pytest.mark.parametrize("charts_fixture", ("cylinder_charts", "sphere_charts"))
def test_cad_pole_fan_interior_edge_requires_exact_common_fiber(
    charts_fixture: str, request: pytest.FixtureRequest
) -> None:
    from phydrax.geometry._domain_boundary import _rooted_patch_chart_endpoints
    from phydrax.geometry._meshing_domain import PatchPoleUse
    from phydrax.geometry._surface_source_support import (
        _validate_cad_chart_edge_incidence,
    )

    charts = request.getfixturevalue(charts_fixture)
    assert isinstance(charts, SurfaceSourceCharts)
    domain = charts.domain
    for patch, surface in enumerate(domain.patches):
        for loop_index, loop in enumerate(surface.loops):
            for use_index, use in enumerate(loop):
                if not isinstance(use, PatchPoleUse):
                    continue
                start, stop = _rooted_patch_chart_endpoints(domain, patch)[loop_index][
                    use_index
                ]
                one = ((2 * start[0] + stop[0]) / 3, (2 * start[1] + stop[1]) / 3)
                two = ((start[0] + 2 * stop[0]) / 3, (start[1] + 2 * stop[1]) / 3)
                apex = (one[0], one[1] + 1)
                rows = [
                    (identifier, slot)
                    for row, identifier in enumerate(charts.cell_global_ids)
                    if int(charts.root_patches[row]) == patch
                    for slot in range(3)
                    if charts.boundary_occurrences.size == 0
                    or tuple(charts.boundary_occurrences[row, slot]) == (-1, -1, -1)
                ]
                first = rows[0]
                second = next(row for row in rows if row[0] != first[0])
                adjacent: dict[
                    int,
                    tuple[
                        tuple[
                            int, int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]
                        ],
                        ...,
                    ],
                ] = {
                    1: (
                        (first[0], first[1], one, apex),
                        (second[0], second[1], two, apex),
                    )
                }
                _validate_cad_chart_edge_incidence(charts, adjacent, set())
                forged: dict[
                    int,
                    tuple[
                        tuple[
                            int, int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]
                        ],
                        ...,
                    ],
                ] = {
                    1: (
                        (first[0], first[1], one, apex),
                        (second[0], second[1], (two[0], two[1] + 1), apex),
                    )
                }
                with pytest.raises(ValueError, match="unbound or unproved"):
                    _validate_cad_chart_edge_incidence(charts, forged, set())
                return
    pytest.fail("The source has no pole side.")


def test_cad_periodic_coedge_pair_requires_both_occurrences_and_opposite_sense(
    cylinder_charts: SurfaceSourceCharts,
) -> None:
    from phydrax.geometry._approximate_source_coverage import (
        _cad_validate_coedge_pair,
        SourceCoverageUnavailable,
    )
    from phydrax.geometry._meshing_domain import PatchCurveUse

    uses = [
        use
        for patch in cylinder_charts.domain.patches
        for loop in patch.loops
        for use in loop
        if isinstance(use, PatchCurveUse)
    ]
    authored_first, authored_opposite = next(
        (first_use, second_use)
        for first_use in uses
        for second_use in uses
        if first_use.curve == second_use.curve and first_use.sense == -second_use.sense
    )
    domain = cylinder_charts.domain
    curve = (
        domain.entity_id(1, authored_first.curve),
        domain.source_occurrences[1][authored_first.curve],
    )
    first = (curve, Fraction(0), Fraction(1, 2), authored_first)
    opposite = (curve, Fraction(0), Fraction(1, 2), authored_opposite)
    _cad_validate_coedge_pair(first, opposite, 1, 1)
    with pytest.raises(SourceCoverageUnavailable, match="lacks bound"):
        _cad_validate_coedge_pair(first, None, 1, 1)
    with pytest.raises(SourceCoverageUnavailable, match="same authored"):
        _cad_validate_coedge_pair(
            first, (curve, Fraction(0), Fraction(1, 3), opposite[3]), 1, 1
        )
    with pytest.raises(SourceCoverageUnavailable, match="same authored"):
        _cad_validate_coedge_pair(
            first,
            (
                (curve[0], (*curve[1], "forged")),
                Fraction(0),
                Fraction(1, 2),
                authored_opposite,
            ),
            1,
            1,
        )
    with pytest.raises(SourceCoverageUnavailable, match="opposite"):
        _cad_validate_coedge_pair(first, first, 1, 1)


def test_cad_root_curve_ancestry_replays_complete_source_chains() -> None:
    from phydrax.geometry._surface_source_support import validate_cad_curve_ancestry

    domain, surface = _cad_surface()
    charts = surface.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    bindings = validate_cad_curve_ancestry(charts, 0.1)
    assert bindings
    assert {binding.curve_row for rows in bindings.values() for binding in rows} == set(
        range(len(domain.curves))
    )


def test_cad_trim_homotopy_replays_box_root_chains() -> None:
    from phydrax.geometry._approximate_source_coverage import _cad_trim_reference_loops
    from phydrax.geometry._meshing_domain import SourcePreparation
    from phydrax.geometry._surface_source_support import validate_cad_curve_ancestry
    from phydrax.geometry.brep._curve_approximation_topology import (
        certify_trim_reference_homotopy,
    )

    domain, surface = _cad_surface()
    charts = surface.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    bindings = validate_cad_curve_ancestry(charts, 0.1)
    preparation = SourcePreparation(domain)
    for patch in range(len(domain.patches)):
        loops = _cad_trim_reference_loops(charts, bindings, patch)
        assert certify_trim_reference_homotopy(loops, preparation)


def _assert_curve_ancestry(
    source: NativeCadVolumeSource, construction: NativeMeshConstruction, *, fixed: bool
) -> None:
    witness = construction.certification.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    assert witness.cad_plc is source.plc_source.complex
    roots = np.asarray(witness.cad_root_cell_ids, dtype=np.int64)
    assert roots.shape == np.asarray(witness.entity_global_ids).shape
    assert set(roots.tolist()).issubset({-1, *source.surface_source.cell_global_ids})
    association = witness.cad_curve_association
    assert association is not None
    assert association.association_kind is GeometryAssociationKind.CURVE
    assert (
        association.target_entity_set_id == construction.mesh.entity_set(1).entity_set_id
    )
    assert any(
        value.association_id == association.association_id
        for value in construction.organization.associations
    )
    (brep_edges,) = tuple(
        value
        for value in construction.organization.associations
        if value.association_kind is GeometryAssociationKind.BREP
        and value.target_entity_set_id == association.target_entity_set_id
    )
    brep_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(brep_edges.target_global_ids))
    }
    for row, identifier in enumerate(np.asarray(association.target_global_ids)):
        brep_row = brep_rows[int(identifier)]
        assert (
            brep_edges.source_entity_ids[brep_row] == association.source_entity_ids[row]
        )
        assert (
            brep_edges.source_occurrence_paths[brep_row]
            == association.source_occurrence_paths[row]
        )
        assert int(brep_edges.orientations[brep_row]) == int(
            association.orientations[row]
        )
    (original,) = tuple(
        value
        for value in source.surface.associations
        if value.association_kind is GeometryAssociationKind.CURVE
        and value.target_entity_set_id == source.surface.mesh.entity_set(1).entity_set_id
    )
    assert association.provenance is GeometryAssociationProvenance.LINEAGE
    assert association.parent_association_id == original.association_id
    assert np.all(np.asarray(association.parent_dimensions, dtype=np.int64) == 1)
    original_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(original.target_global_ids))
    }
    for row, parent_id in enumerate(np.asarray(association.parent_ids)):
        parent_row = original_rows[int(parent_id)]
        parent_interval = original.source_parameter_intervals[parent_row]
        interval = association.source_parameter_intervals[row]
        assert parent_interval is not None and interval is not None
        assert parent_interval[0] <= interval[0] < interval[1] <= parent_interval[1]
        assert (
            association.source_entity_ids[row] == original.source_entity_ids[parent_row]
        )
        assert (
            association.source_occurrence_paths[row]
            == original.source_occurrence_paths[parent_row]
        )
    source_intervals = tuple(
        zip(
            original.source_entity_ids,
            original.source_occurrence_paths,
            original.source_parameter_intervals,
            strict=True,
        )
    )
    target_intervals = tuple(
        zip(
            association.source_entity_ids,
            association.source_occurrence_paths,
            association.source_parameter_intervals,
            strict=True,
        )
    )
    assert set(np.asarray(association.orientations).tolist()).issubset({-1, 1})
    matched = 0
    for entity_id, path, root in source_intervals:
        assert root is not None
        children = sorted(
            interval
            for child_id, child_path, interval in target_intervals
            if child_id == entity_id
            and child_path == path
            and interval is not None
            and root[0] <= interval[0] < interval[1] <= root[1]
        )
        assert children
        assert children[0][0] == root[0]
        assert children[-1][1] == root[1]
        assert all(first[1] == second[0] for first, second in zip(children, children[1:]))
        if fixed:
            assert children == [root]
        matched += len(children)
    assert matched == len(target_intervals)
    original_connectivity = source.surface.mesh.connectivity
    target_connectivity = construction.mesh.connectivity
    assert isinstance(original_connectivity, PolygonalConnectivity)
    assert isinstance(target_connectivity, SimplicialConnectivity)
    original_edges = np.asarray(original_connectivity.edges, dtype=np.int64)
    target_edges = np.asarray(target_connectivity.entities[1], dtype=np.int64)
    original_edge_rows = {
        int(identifier): row
        for row, identifier in enumerate(source.surface.mesh.entity_set(1).entity_ids)
    }
    target_edge_rows = {
        int(identifier): row
        for row, identifier in enumerate(construction.mesh.entity_set(1).entity_ids)
    }
    original_points = np.asarray(source.surface.mesh.coordinates)
    target_points = np.asarray(construction.mesh.coordinates)
    original_orientations = np.asarray(original.orientations, dtype=np.int8)
    target_orientations = np.asarray(association.orientations, dtype=np.int8)
    for row, (entity_id, path, interval) in enumerate(target_intervals):
        assert interval is not None
        parents = [
            parent
            for parent, (source_id, source_path, root) in enumerate(source_intervals)
            if source_id == entity_id
            and source_path == path
            and root is not None
            and root[0] <= interval[0] < interval[1] <= root[1]
        ]
        assert len(parents) == 1
        parent = parents[0]
        original_row = original_edge_rows[int(original.target_global_ids[parent])]
        target_row = target_edge_rows[int(association.target_global_ids[row])]
        original_vertices = original_points[original_edges[original_row]]
        target_vertices = target_points[target_edges[target_row]]
        direction = float(
            np.dot(
                original_vertices[1] - original_vertices[0],
                target_vertices[1] - target_vertices[0],
            )
        )
        assert (
            direction * int(original_orientations[parent]) * int(target_orientations[row])
            > 0
        )


def test_quadric_boundary_meets_fixed_tetrahedral_face_quality(tmp_path: Path) -> None:
    from phydrax.geometry._approximate_source_coverage import _cad_trim_reference_loops
    from phydrax.geometry._meshing_domain import SourcePreparation
    from phydrax.geometry._surface_source_support import validate_cad_curve_ancestry
    from phydrax.geometry.brep._curve_approximation_topology import (
        certify_trim_reference_homotopy,
    )
    from phydrax.geometry.brep._source_embedding import certify_cad_trim_sheet
    from phydrax.geometry.brep._source_shell_admissibility import (
        certify_single_source_shell,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _quadric_case

    model, _, _, _ = _quadric_case("iges", tmp_path)
    domain = MeshingDomain.from_brep(model)
    scope = MeshingScope.whole(model, 2)
    deviation = 0.125**2 / (4.0 * 0.6)
    specification = phx.meshing.SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(
                scope, 0.125, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            ProtectedFeature(scope, FeatureKind.SURFACE, maximum_deviation=deviation),
        ),
        maximum_deviation=deviation,
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=float(np.nextafter(np.arcsin(0.25), np.inf))
        ),
    )
    result = (
        phx.meshing.NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
        .plan(domain, specification, coordinate_contract=model.coordinate_contract)
        .execute()
    )
    assert result.compliance.passed
    assert isinstance(result.surface_source, SurfaceSourceCharts)
    curve_ancestry = validate_cad_curve_ancestry(result.surface_source, deviation)
    assert curve_ancestry
    preparation = SourcePreparation(domain)
    for patch in range(len(domain.patches)):
        assert certify_cad_trim_sheet(result.surface_source, patch, deviation) is not None
        assert certify_trim_reference_homotopy(
            _cad_trim_reference_loops(result.surface_source, curve_ancestry, patch),
            preparation,
        )
    assert certify_single_source_shell(result.surface_source, deviation) is not None
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    triangles = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in result.mesh.blocks)
    )
    corners = coordinates[triangles]
    lengths = np.stack(
        tuple(
            np.linalg.norm(corners[:, (axis + 1) % 3] - corners[:, axis], axis=1)
            for axis in range(3)
        ),
        axis=1,
    )
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    assert np.all(areas > 0.0)
    assert (
        np.max(np.prod(lengths, axis=1) / (4.0 * areas * np.min(lengths, axis=1))) <= 2.0
    )


def test_native_cad_volume_keeps_original_brep_query_and_explicit_tolerance(
    tmp_path: Path,
) -> None:
    contract = phx.SpatialCoordinateContract.si()
    domain, surface = _cad_surface()
    source = NativeCadVolumeSource(
        domain,
        surface,
        maximum_deviation=0.1,
        boundary="fixed",
    )
    fidelity_source = source.fidelity_source
    if not isinstance(fidelity_source, MeshingDomainBoundarySource):
        raise AssertionError("CAD volume source must retain its original BRep query.")

    assert len(source.source_triangle_parameter_exact) == len(source.source_triangle_ids)
    for exact_triangle, numeric_triangle in zip(
        source.source_triangle_parameter_exact,
        source.source_triangle_parameters,
        strict=True,
    ):
        assert all(
            isinstance(value, Fraction) for point in exact_triangle for value in point
        )
        np.testing.assert_array_equal(
            np.asarray([[float(value) for value in point] for point in exact_triangle]),
            numeric_triangle,
        )
    assert source.plc_source.complex.boundary == "fixed"
    assert source.plc_source.complex.geometry is None
    assert source.plc_source.complex.cad_source_charts is source.surface_source
    assert source.plc_source.complex.cad_embedding is not None
    assert source.plc_source.complex.cad_fidelity is not None
    assert source.plc_source.complex.cad_embedding.status == "certified"
    assert source.plc_source.complex.cad_fidelity.status == "certified"
    assert source.plc_source.complex.cad_fidelity_tolerance == 0.1
    assert source.plc_source.complex.region_ids == tuple(
        domain.entity_id(3, row) for row in range(len(domain.regions))
    )
    _, source_side_signs = cad_material_sides(domain)
    expected_facet_regions = np.full((len(domain.patches), 2), -1, dtype=np.int64)
    for patch_row, signs in enumerate(source_side_signs):
        for region_row, source_sign in signs:
            boundary_sign = domain.patches[patch_row].orientation * source_sign
            side = 1 if boundary_sign > 0 else 0
            expected_facet_regions[patch_row, side] = region_row
    np.testing.assert_array_equal(
        source.plc_source.complex.facet_regions, expected_facet_regions
    )
    assert fidelity_source.source_id == domain.source_id
    assert fidelity_source.source_revision == domain.source_revision
    assert fidelity_source.boundary_chart_cover().complete

    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    corner_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        0,
        domain.scope_entity_set_id(0),
        np.asarray(domain.scope_entity_ids(0)[:1], dtype=np.int64),
    )
    curve_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        domain.scope_entity_set_id(1),
        np.asarray(domain.scope_entity_ids(1)[:1], dtype=np.int64),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        protected_features=(
            ProtectedFeature(
                scope,
                FeatureKind.SURFACE,
                maximum_deviation=0.05,
            ),
            ProtectedFeature(
                curve_scope,
                FeatureKind.CURVE,
                maximum_deviation=0.05,
            ),
            ProtectedFeature(
                corner_scope,
                FeatureKind.CORNER,
                maximum_deviation=0.05,
            ),
        ),
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                domain.regions[0].name,
                "solid-material",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.05,
    )
    prepared = prepare_native_cad_volume(
        source,
        specification,
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
        ),
        contract,
    )
    construction = prepared.execute()

    (region_evidence,) = construction.organization.region_boundary_evidence
    assert region_evidence.control_id is None
    assert region_evidence.material_id == "solid-material"
    assert region_evidence.role is RegionRole.SOLID
    assert region_evidence.source_scope.source_id == domain.source_id
    assert region_evidence.source_scope.source_revision == domain.source_revision
    assert region_evidence.source_scope.entity_set_id == domain.scope_entity_set_id(3)
    source_region_key = domain.source_entity_key_for_scope(
        3, int(region_evidence.source_scope.entity_ids[0])
    )
    assert (region_evidence.source_region_id, region_evidence.source_occurrence_path) == (
        source_region_key[0],
        source_region_key[1],
    )
    assert region_evidence.boundary_label.label_id in {
        label.label_id for label in construction.organization.labels
    }
    region_evidence.require_current(
        construction.mesh,
        construction.organization.labels,
        construction.organization.patches,
    )
    patch_by_id = {patch.patch_id: patch for patch in construction.organization.patches}
    for patch_id, side in region_evidence.patch_sides:
        patch = patch_by_id[patch_id]
        source_side = 0 if side > 0 else 1
        assert patch.source_adjacent_region_ids is not None
        assert patch.source_adjacent_region_occurrence_paths is not None
        assert patch.source_adjacent_region_ids[source_side] == (
            region_evidence.source_region_id
        )
        assert patch.source_adjacent_region_occurrence_paths[source_side] == (
            region_evidence.source_occurrence_path
        )
    assert construction.certification.domain is domain
    assert construction.certification.fidelity_source is fidelity_source
    assert construction.certification.fidelity_tolerance == 0.05
    assert construction.geometry is not None
    witness = construction.certification.fidelity_witness
    assert witness is not None
    assert isinstance(witness, SourceBoundaryChartWitness)
    assert witness.source_query_id == fidelity_source.source_scope_id
    witness.require_current(construction.mesh, construction.geometry, fidelity_source)
    from phydrax.geometry._approximate_source_coverage import (
        _cad_target_root_partition,
        SourceCoverageUnavailable,
    )

    patch_rows = [
        domain.resolve_entity_id(2, entity, occurrence_path=path)
        for entity, path in zip(
            witness.source_entity_ids, witness.occurrence_paths, strict=True
        )
    ]
    ancestry = _cad_target_root_partition(
        domain, construction.mesh, witness, patch_rows, 0.05
    )
    assert ancestry.vertex_keys
    assert ancestry.edge_uses
    roots = np.asarray(witness.cad_root_cell_ids, dtype=np.int64).copy()
    first = next(row for row, root in enumerate(roots) if root >= 0)
    second = next(
        row for row, root in enumerate(roots) if root >= 0 and root != roots[first]
    )
    roots[first], roots[second] = roots[second], roots[first]
    reordered = eqx.tree_at(lambda value: value.cad_root_cell_ids, witness, roots)
    with pytest.raises(SourceCoverageUnavailable):
        _cad_target_root_partition(domain, construction.mesh, reordered, patch_rows, 0.05)
    with pytest.raises(ValueError):
        reordered.require_current(
            construction.mesh, construction.geometry, fidelity_source
        )
    curve_association = witness.cad_curve_association
    assert curve_association is not None
    changed_curve = eqx.tree_at(
        lambda value: value.orientations,
        curve_association,
        -curve_association.orientations,
    )
    changed_witness = eqx.tree_at(
        lambda value: value.cad_curve_association, witness, changed_curve
    )
    with pytest.raises((SourceCoverageUnavailable, ValueError)):
        _cad_target_root_partition(
            domain, construction.mesh, changed_witness, patch_rows, 0.05
        )
    with pytest.raises(ValueError):
        changed_witness.require_current(
            construction.mesh, construction.geometry, fidelity_source
        )
    _assert_curve_ancestry(source, construction, fixed=True)
    archive_path = tmp_path / "cad-chart-witness.phx"
    write_model_artifact(archive_path, witness, limits=None)
    worker = (
        Path(__file__).resolve().parents[1]
        / "interchange"
        / "_native_cad_archive_worker.py"
    )
    restored = subprocess.run(
        [
            sys.executable,
            str(worker),
            "read-cad-chart-witness",
            str(archive_path),
            witness.witness_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert "cad-chart-witness-cold-restored" in restored.stdout
    assert (
        construction.certification.protected_features == specification.protected_features
    )
    assert construction.certification.feature_source is domain
    scoped = construction.certification.scoped_fidelity
    assert len(scoped) == 3
    surface_query, surface_witness, surface_tolerance = next(
        row
        for row in scoped
        if isinstance(row[1], SourceBoundaryChartWitness) and row[1].chart_dimension == 2
    )
    if not isinstance(surface_query, MeshingDomainBoundarySource):
        raise AssertionError("CAD surface witness must use the original BRep query.")
    if not isinstance(surface_witness, SourceBoundaryChartWitness):
        raise AssertionError("CAD surface scope must retain its chart witness.")
    assert surface_tolerance == 0.05
    assert surface_witness.source_query_id == surface_query.source_scope_id
    assert set(surface_witness.source_entity_ids) == {
        domain.source_entity_key_for_scope(2, int(identifier))[0]
        for identifier in scope.entity_ids.tolist()
    }
    curve_query, curve_witness, curve_tolerance = next(
        row
        for row in scoped
        if isinstance(row[1], SourceBoundaryChartWitness) and row[1].chart_dimension == 1
    )
    if not isinstance(curve_query, ParametricCurveBoundarySource):
        raise AssertionError("CAD curve feature must retain its parametric query.")
    if not isinstance(curve_witness, SourceBoundaryChartWitness):
        raise AssertionError("CAD curve feature must retain its chart witness.")
    assert curve_tolerance == 0.05
    assert curve_witness.source_query_id == curve_query.query_id
    assert set(curve_witness.source_entity_ids) == {
        domain.source_entity_key_for_scope(1, int(identifier))[0]
        for identifier in curve_scope.entity_ids.tolist()
    }
    point_query, point_witness, point_tolerance = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryPointWitness)
    )
    if not isinstance(point_query, MeshingDomainBoundarySource):
        raise AssertionError("CAD corner feature must retain its point query.")
    if not isinstance(point_witness, SourceBoundaryPointWitness):
        raise AssertionError("CAD corner feature must retain its point witness.")
    assert point_tolerance == 0.05
    assert point_witness.source_query_id == point_query.source_scope_id
    assert point_witness.source_entity_ids == (
        domain.source_entity_key_for_scope(0, int(corner_scope.entity_ids[0]))[0],
    )
    exterior_face_ids = set(
        np.asarray(construction.mesh.entity_set(2).entity_ids, dtype=np.int64)[
            np.asarray(construction.mesh.boundary_masks[2], dtype=np.bool_)
        ].tolist()
    )
    assert exterior_face_ids.issubset(
        set(np.asarray(witness.entity_global_ids, dtype=np.int64).tolist())
    )
    assert construction.organization.associations
    assert all(
        association.association_kind
        in (GeometryAssociationKind.BREP, GeometryAssociationKind.CURVE)
        for association in construction.organization.associations
    )
    complex_ = source.plc_source.complex
    authored_curve_sources = {
        (int(edge_index), tuple(path))
        for edge_index, path in zip(
            np.asarray(source.segment_brep_indices, dtype=np.int64).tolist(),
            complex_.segment_occurrence_paths,
            strict=True,
        )
    }
    edge_entity_set_id = construction.mesh.entity_set(1).entity_set_id
    edge_associations = tuple(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == edge_entity_set_id
        and association.association_kind is GeometryAssociationKind.BREP
    )
    if len(edge_associations) != 1:
        raise AssertionError("CAD volume must publish one BRep edge association.")
    edge_association = edge_associations[0]
    edge_dimensions = np.asarray(edge_association.source_dimensions, dtype=np.int64)
    edge_indices = np.asarray(edge_association.source_indices, dtype=np.int64)
    assert np.any(edge_dimensions == 1)
    assert np.any(edge_dimensions == 2)
    for dimension, source_index, path, role in zip(
        edge_dimensions.tolist(),
        edge_indices.tolist(),
        edge_association.source_occurrence_paths,
        edge_association.source_roles,
        strict=True,
    ):
        if dimension == 1:
            assert (source_index, path) in authored_curve_sources
            assert role is GeometrySourceEntityRole.EDGE
        elif dimension == 2:
            assert role is GeometrySourceEntityRole.FACET


def test_native_cad_region_control_identity_is_preserved_in_boundary_evidence() -> None:
    contract = phx.SpatialCoordinateContract.si()
    domain, surface = _cad_surface()
    source = NativeCadVolumeSource(
        domain,
        surface,
        maximum_deviation=0.1,
        boundary="fixed",
    )
    boundary_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    region_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        domain.scope_entity_set_id(3),
        np.asarray(domain.scope_entity_ids(3)[:1], dtype=np.int64),
    )
    region_name = domain.regions[0].name
    control = RegionControl(
        region_scope,
        region_name,
        "controlled-solid",
        RegionRole.SOLID,
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
        ),
        boundary_scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        region_controls=(control,),
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                region_name,
                "seed-material",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.05,
    )
    prepared = prepare_native_cad_volume(
        source,
        specification,
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
        ),
        contract,
    )

    construction = prepared.execute()

    (evidence,) = construction.organization.region_boundary_evidence
    assert evidence.control_id == control.control_id
    assert evidence.source_scope.scope_id == control.scope.scope_id
    assert evidence.material_id == "controlled-solid"
    assert evidence.role is RegionRole.SOLID
    assert evidence.source_occurrence_path == domain.region_source_occurrences[0]
    evidence.require_current(
        construction.mesh,
        construction.organization.labels,
        construction.organization.patches,
    )


def test_conforming_cad_box_splits_declared_patch_planes_with_exact_chart_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each planar box face is a declared exact plane: conforming refinement
    # places facet split points on it (TetMesh witness stratum 3), and the
    # chart witness maps every boundary vertex through the patch's one exact
    # planar chart.
    contract = phx.SpatialCoordinateContract.si()
    domain, surface = _cad_surface()
    source = NativeCadVolumeSource(
        domain, surface, maximum_deviation=0.1, boundary="conforming"
    )
    planes = np.asarray(source.plc_source.complex.facet_planes, dtype=np.float64)
    assert planes.shape == (len(domain.patches), 3, 3)
    assert np.all(np.isfinite(planes))
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 0.3),),
        region_seeds=(
            RegionSeed(
                np.asarray((0.5, 0.5, 0.5), dtype=np.float64),
                domain.regions[0].name,
                "solid-material",
                RegionRole.SOLID,
            ),
        ),
        maximum_deviation=0.05,
    )
    generated = []
    generate = cad_volume.generate_plc_volume

    def capture(*args: Any, **kwargs: Any) -> VolumeConstruction:
        result = generate(*args, **kwargs)
        generated.append(result)
        return result

    monkeypatch.setattr(cad_volume, "generate_plc_volume", capture)
    construction = prepare_native_cad_volume(
        source,
        specification,
        NativeMeshingOptions(
            "cad_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=2.0, minimum_dihedral_degrees=0.0
            ),
        ),
        contract,
    ).execute()
    witness = construction.certification.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    assert construction.geometry is not None
    witness.require_current(
        construction.mesh, construction.geometry, source.fidelity_source
    )
    _assert_curve_ancestry(source, construction, fixed=False)
    (raw_construction,) = generated
    reordered_source = eqx.tree_at(
        lambda value: value.source_triangle_cell_ids,
        source,
        np.roll(np.asarray(source.source_triangle_cell_ids), 1),
    )
    with pytest.raises(ValueError, match="canonical PLC row bindings"):
        cad_volume._cad_volume_chart_witness(
            reordered_source,
            raw_construction,
            construction.geometry,
            witness.cad_curve_association,
        )
    authored_ids = set(source.plc_source.complex.segment_ids)
    source_row = next(
        row
        for row, identifier in enumerate(raw_construction.segment_ids)
        if identifier in authored_ids
    )
    corrupt_endpoints = np.array(raw_construction.source_segment_vertices, copy=True)
    corrupt_endpoints[source_row] = corrupt_endpoints[source_row, ::-1]
    with pytest.raises(ValueError, match="canonical endpoints"):
        cad_volume._cad_curve_association(
            source, replace(raw_construction, source_segment_vertices=corrupt_endpoints)
        )
    target_row = next(
        row
        for row, parent in enumerate(raw_construction.segment_source_edges)
        if int(parent) == source_row
    )
    endpoint = int(raw_construction.segments[target_row, 0])
    input_vertices = np.array(raw_construction.input_vertices, copy=True)
    witness_strata = np.array(raw_construction.witness_strata, copy=True)
    input_vertices[endpoint] = -1
    witness_strata[endpoint] = 0
    with pytest.raises(ValueError, match="endpoint lacks"):
        cad_volume._cad_curve_association(
            source,
            replace(
                raw_construction,
                input_vertices=input_vertices,
                witness_strata=witness_strata,
            ),
        )
    keep = np.arange(raw_construction.segments.shape[0]) != target_row
    with pytest.raises(ValueError, match="partition"):
        cad_volume._cad_curve_association(
            source,
            replace(
                raw_construction,
                segments=raw_construction.segments[keep],
                segment_source_edges=raw_construction.segment_source_edges[keep],
                segment_source_ids=tuple(
                    value
                    for row, value in enumerate(raw_construction.segment_source_ids)
                    if row != target_row
                ),
                segment_source_occurrence_paths=tuple(
                    value
                    for row, value in enumerate(
                        raw_construction.segment_source_occurrence_paths
                    )
                    if row != target_row
                ),
            ),
        )

    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    block = construction.mesh.blocks[0]
    cell_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(block.global_ids).tolist())
    }
    cells = np.asarray(block.vertices, dtype=np.int64)
    patch_by_entity = {domain.entity_id(2, patch): patch for patch in range(len(planes))}
    surface_points = {
        tuple(point) for point in np.asarray(surface.mesh.coordinates).tolist()
    }
    face_interior_vertices = 0
    for cell_id, chart_id, references, parameters in zip(
        np.asarray(witness.cell_global_ids).tolist(),
        witness.source_chart_ids,
        np.asarray(witness.target_reference_vertices),
        np.asarray(witness.source_parameters),
        strict=True,
    ):
        patch = domain.patches[patch_by_entity[chart_id]].surface
        for reference, uv in zip(references, parameters, strict=True):
            local = 0 if not np.any(reference) else int(np.argmax(reference)) + 1
            point = coordinates[cells[cell_rows[int(cell_id)], local]]
            np.testing.assert_allclose(
                np.asarray(patch.evaluate(np.asarray(uv))), point, rtol=0.0, atol=1e-12
            )
            if (
                tuple(point.tolist()) not in surface_points
                and np.count_nonzero((point > 0.0) & (point < 1.0)) == 2
            ):
                face_interior_vertices += 1
    # Refinement split the declared faces inside their planes.
    assert face_interior_vertices > 0


def test_cad_sphere_volume_certifies_collapsed_poles_and_its_periodic_seam() -> None:
    # The sphere's chart has two collapsed pole sides and a seam whose u = 2*pi
    # use is a native-period gauge of the u = 0 line. Pole fans leave chart
    # slivers between their apex chart points; every pole chart point names the
    # one pole corner, and seam points on either side name one curve point.
    contract = phx.SpatialCoordinateContract.si()
    model = brep_sphere(0.5, coordinate_contract=contract, source_id="cad-volume-sphere")
    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.3),),
        maximum_deviation=0.05,
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            NativeMeshingOptions(
                "cad_tetrahedral",
                volume_schedule=NativeVolumeSchedule(
                    radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
                ),
            )
        )
        .plan(model, specification)
        .execute()
    )

    assert result.certification is not None and result.certification.passed
    witness = result.certification.inputs.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    domain = MeshingDomain.from_brep(model)
    (patch,) = range(len(domain.patches))
    pole = Fraction.from_float(np.pi / 2.0)
    two_pi = Fraction.from_float(2.0 * np.pi)
    points = {
        (point[0], point[1])
        for row in range(witness.cell_global_ids.size)
        for point in witness.source_parameter_chart(row)
    }
    for side in (-pole, pole):
        pole_points = {point for point in points if point[1] == side}
        assert len(pole_points) > 1
        (label,) = {_cad_chart_vertex_key(domain, patch, point) for point in pole_points}
        assert label[0] == "corner"
    seam = [point for point in points if point[0] == two_pi and abs(point[1]) < pole]
    assert seam
    for point in seam:
        label = _cad_chart_vertex_key(domain, patch, point)
        assert label[0] == "curve"
        assert label == _cad_chart_vertex_key(domain, patch, (Fraction(0), point[1]))


def test_cad_cylinder_volume_charts_planar_caps_in_their_exact_plane() -> None:
    # Each cap revolves a radial profile line: its authored polar chart has a
    # pole and a seam, but the carrier proves it exactly planar. The cap is a
    # declared PLC plane, and its witness rows use the exact affine plane chart
    # (pole and seam interior), whose trim points take their rim curve and corner
    # labels from the lateral face.
    contract = phx.SpatialCoordinateContract.si()
    model = brep_cylinder(
        0.5, 1.0, coordinate_contract=contract, source_id="cad-volume-cylinder"
    )
    boundary = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(boundary, 0.3),),
        maximum_deviation=0.05,
    )
    result = (
        phx.meshing.NativeMeshingProvider(
            NativeMeshingOptions(
                "cad_tetrahedral",
                volume_schedule=NativeVolumeSchedule(
                    radius_edge_bound=8.0, minimum_dihedral_degrees=0.0
                ),
            )
        )
        .plan(model, specification)
        .execute()
    )

    assert result.certification is not None and result.certification.passed
    assert result.compliance.issues == ()
    witness = result.certification.inputs.fidelity_witness
    assert isinstance(witness, SourceBoundaryChartWitness)
    domain = MeshingDomain.from_brep(model)
    planes = {
        chart.chart_id: chart
        for chart in (
            exact_plane_chart(domain, patch) for patch in range(len(domain.patches))
        )
        if chart is not None
    }
    assert len(planes) == 2
    assert set(planes) <= set(witness.source_chart_ids)
    from phydrax.geometry._approximate_source_coverage import (
        _cad_target_root_partition,
        SourceCoverageUnavailable,
    )

    assert witness.cad_plc is not None
    charts = witness.cad_plc.cad_source_charts
    assert charts is not None
    from phydrax.geometry._meshing_domain import PatchCurveUse
    from phydrax.geometry._surface_source_support import validate_cad_curve_ancestry

    occurrences = np.array(charts.boundary_occurrences, copy=True)
    swapped = False
    for row, slot in np.argwhere(occurrences[:, :, 0] >= 0):
        patch, loop, use_index = (int(value) for value in occurrences[row, slot])
        uses = charts.domain.patches[patch].loops[loop]
        use = uses[use_index]
        if not isinstance(use, PatchCurveUse):
            continue
        alternatives = [
            index
            for index, other in enumerate(uses)
            if index != use_index
            and isinstance(other, PatchCurveUse)
            and other.curve == use.curve
        ]
        if alternatives:
            occurrences[row, slot, 2] = alternatives[0]
            swapped = True
            break
    assert swapped
    forged_charts = SurfaceSourceCharts(
        charts.domain,
        charts.mesh,
        charts.geometry,
        charts.coordinate_contract,
        charts.root_parameters,
        charts.root_patches,
        charts.associations,
        charts.boundary_source,
        source_root_cell_ids=charts.source_root_cell_ids,
        source_piece_ids=charts.source_piece_ids,
        root_parameter_exact=charts.root_parameter_exact,
        boundary_occurrences=occurrences,
    )
    with pytest.raises(ValueError, match="authored coedge occurrence"):
        validate_cad_curve_ancestry(forged_charts, 1e4)
    patch_rows = [
        domain.resolve_entity_id(2, entity, occurrence_path=path)
        for entity, path in zip(
            witness.source_entity_ids, witness.occurrence_paths, strict=True
        )
    ]
    plane_row = next(
        row for row, chart_id in enumerate(witness.source_chart_ids) if chart_id in planes
    )
    root_row = next(
        row
        for row, patch in enumerate(charts.root_patches)
        if int(patch) == patch_rows[plane_row]
    )
    roots = np.asarray(witness.cad_root_cell_ids, dtype=np.int64).copy()
    assert roots[plane_row] == -1
    roots[plane_row] = charts.cell_global_ids[root_row]
    forged = eqx.tree_at(lambda value: value.cad_root_cell_ids, witness, roots)
    with pytest.raises(SourceCoverageUnavailable, match="nonlinear root triangle"):
        _cad_target_root_partition(domain, result.mesh, forged, patch_rows, 0.05)
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    block = result.mesh.blocks[0]
    cell_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(block.global_ids).tolist())
    }
    cells = np.asarray(block.vertices, dtype=np.int64)
    plane_rows = 0
    for row, chart_id in enumerate(witness.source_chart_ids):
        chart = planes.get(chart_id)
        if chart is None:
            continue
        plane_rows += 1
        origin, first, second = (
            np.asarray([float(value) for value in axis], dtype=np.float64)
            for axis in (chart.origin, chart.first_axis, chart.second_axis)
        )
        references = np.asarray(witness.target_reference_vertices)[row]
        for reference, (s, t) in zip(
            references, witness.source_parameter_chart(row), strict=True
        ):
            local = 0 if not np.any(reference) else int(np.argmax(reference)) + 1
            point = coordinates[
                cells[cell_rows[int(witness.cell_global_ids[row])], local]
            ]
            np.testing.assert_allclose(
                origin + float(s) * first + float(t) * second, point, atol=1.0e-12
            )
    assert plane_rows > 0

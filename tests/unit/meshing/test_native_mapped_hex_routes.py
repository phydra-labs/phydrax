from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import PerCellRestrictedCellGeometryElement
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_enclosure import add, axes, multiply, scale
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import lagrange_element
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_boundary import (
    _mapped_root_entity_banks,
    MappedReferenceBoundarySource,
)
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry._source_fidelity import (
    _surface_target_chart_rows,
    certify_source_fidelity,
    validate_source_fidelity_witness,
)
from phydrax.meshing._association import GeometryAssociationKind
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import PatchControl, RegionControl, RegionSeed
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.meshing.providers._native_dual import (
    prepare_native_mapped_balanced_grid_hex,
    prepare_native_mapped_frame_grid_hex,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_sources import (
    NativeMappedHexSource,
    NativePlcSource,
)
from phydrax.meshing.providers._native_volume import prepare_native_plc_volume
from phydrax.units import METER


pytestmark = pytest.mark.meshcore


_SOURCE_ID = "mapped-unit-cube"
_SOURCE_REVISION = "mapped-unit-cube-r1"
_SCALE = 2.0
_TRANSLATION = np.asarray((1.0, -2.0, 0.5), dtype=np.float64)


def _mapped_cube_source(
    *, curved: bool = False, shared_facets: bool = False
) -> tuple[NativeMappedHexSource, MappedReferenceDomain]:
    vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    triangles = tuple(
        np.asarray(face, dtype=np.int64)
        for face in (
            (0, 3, 2),
            (0, 2, 1),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        )
    )
    segments = np.asarray(
        tuple(
            tuple(sorted(edge))
            for edge in reference_cell_topology("hexahedron").entities[1]
        ),
        dtype=np.int64,
    )
    facet_count = len(triangles) // 2 if shared_facets else len(triangles)
    complex_ = PiecewiseLinearComplex(
        vertices,
        triangles,
        np.repeat(np.arange(facet_count, dtype=np.int64), 2)
        if shared_facets
        else np.arange(facet_count, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (facet_count, 1)),
        ("solid",),
        segments=segments,
        vertex_ids=tuple(f"cube-vertex-{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"cube-facet-{row}" for row in range(facet_count)),
        facet_occurrence_paths=((),) * facet_count,
        segment_ids=tuple(f"cube-edge-{row}" for row in range(segments.shape[0])),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
    )
    reference_domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(triangles, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(triangles), 1)),
        ("solid",),
        source_id=_SOURCE_ID,
    )
    reference_mesh = CellMesh(
        vertices,
        (CellBlock("root", "hexahedron", np.arange(8, dtype=np.int32)[None, :]),),
    )
    if curved:
        element = lagrange_element("hexahedron", 2)
        reference_nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        mapped_coordinates = np.column_stack(
            (
                2.0 * reference_nodes[:, 0] + 1.0,
                2.0 * reference_nodes[:, 1] - 2.0,
                2.0 * reference_nodes[:, 2]
                + 0.5
                + reference_nodes[:, 0] * reference_nodes[:, 1] / 8.0,
            )
        )
        source_geometry = CellGeometrySpec(
            {"root": element},
            {"root": np.arange(element.local_dof_count, dtype=np.int64)[None, :]},
            mapped_coordinates,
        )
    else:
        base_geometry = CellGeometrySpec.affine(reference_mesh)
        mapped_coordinates = _SCALE * vertices + _TRANSLATION
        source_geometry = CellGeometrySpec(
            dict(zip(base_geometry.block_names, base_geometry.elements, strict=True)),
            {
                name: np.asarray(route)
                for name, route in zip(
                    base_geometry.block_names, base_geometry.geometry_dofs, strict=True
                )
            },
            mapped_coordinates,
        )
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id=_SOURCE_ID,
        source_revision=_SOURCE_REVISION,
        reference_domain_vertex_root_ids=np.asarray(
            reference_mesh.vertex_global_ids, dtype=np.int64
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    reference = NativePlcSource(complex_, _SOURCE_ID, _SOURCE_REVISION)
    return NativeMappedHexSource(
        reference,
        domain,
        plc_vertex_root_global_ids=np.asarray(
            reference_mesh.vertex_global_ids, dtype=np.int64
        ),
    ), domain


def _mapped_plc_tetra_source(
    *, curved: bool = False
) -> tuple[NativePlcSource, MappedReferenceDomain]:
    vertices = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    triangles = tuple(
        np.asarray(face, dtype=np.int64)
        for face in ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    )
    root_vertex_ids = np.asarray((31, 37, 41, 43), dtype=np.int64)
    root_cell_ids = np.asarray((901,), dtype=np.int64)
    reference_domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(triangles, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(triangles), 1)),
        ("bulk",),
        source_id="mapped-native-plc-tetra",
    )
    root_mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "root",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=root_cell_ids,
            ),
        ),
        vertex_global_ids=root_vertex_ids,
    )
    if curved:
        # The quadratic map x + (yz, xz, xy)/8 is authored exactly as a per-cell
        # chart of the vertex-bound P1 root, so the PLC binds one coordinate row
        # per source vertex.
        u, v, w = axes(3)
        chart = (
            add(u, scale(multiply(v, w), Fraction(1, 8))),
            add(v, scale(multiply(u, w), Fraction(1, 8))),
            add(w, scale(multiply(u, v), Fraction(1, 8))),
        )
        element = PerCellRestrictedCellGeometryElement(
            lagrange_element("tetrahedron", 1),
            "tetrahedron",
            root_cell_ids,
            root_cell_ids,
            (chart,),
        )
        source_geometry = CellGeometrySpec(
            {"root": element},
            {"root": np.arange(4, dtype=np.int64)[None, :]},
            vertices,
        )
    else:
        source_geometry = CellGeometrySpec.affine(root_mesh)
    source_revision = "r1"
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id=reference_domain.source_id,
        source_revision=source_revision,
        reference_domain_vertex_root_ids=root_vertex_ids,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    edge_rows = np.asarray(
        tuple(
            tuple(sorted(int(value) for value in edge))
            for edge in reference_cell_topology("tetrahedron").entities[1]
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        triangles,
        np.arange(len(triangles), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(triangles), 1)),
        ("bulk",),
        segments=edge_rows,
        vertex_ids=tuple(str(value) for value in root_vertex_ids.tolist()),
        vertex_occurrence_paths=((),) * len(root_vertex_ids),
        facet_ids=tuple(f"tetra-facet-{row}" for row in range(len(triangles))),
        facet_occurrence_paths=((),) * len(triangles),
        segment_ids=tuple(f"tetra-edge-{row}" for row in range(edge_rows.shape[0])),
        segment_occurrence_paths=((),) * edge_rows.shape[0],
        region_occurrence_paths=((),),
        boundary="fixed",
        geometry=source_geometry,
    )
    source = NativePlcSource(
        complex_,
        domain.source_id,
        source_revision,
        mapped_domain=domain,
        maximum_deviation=0.25,
    )
    return source, domain


def _mapped_two_root_source() -> tuple[NativeMappedHexSource, MappedReferenceDomain]:
    local_vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    local_faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    vertex_rows: dict[tuple[float, float, float], int] = {}
    coordinates: list[tuple[float, float, float]] = []
    cells = []
    for shift in range(2):
        cell = []
        for point in local_vertices:
            coordinate = (float(point[0] + shift), float(point[1]), float(point[2]))
            row = vertex_rows.get(coordinate)
            if row is None:
                row = len(coordinates)
                vertex_rows[coordinate] = row
                coordinates.append(coordinate)
            cell.append(row)
        cells.append(cell)
    vertices = np.asarray(coordinates, dtype=np.float64)
    cell_rows = np.asarray(cells, dtype=np.int32)
    reference_mesh = CellMesh(
        vertices,
        tuple(
            CellBlock(
                f"root-{row}",
                "hexahedron",
                cell_rows[row : row + 1],
                global_ids=np.asarray((identifier,), dtype=np.int64),
            )
            for row, identifier in enumerate((17, 29))
        ),
        vertex_global_ids=np.arange(vertices.shape[0], dtype=np.int64) * 7 + 100,
    )
    face_occurrences: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for cell in cell_rows:
        for local_face in local_faces:
            cycle = tuple(int(cell[index]) for index in local_face)
            face_occurrences.setdefault(tuple(sorted(cycle)), []).append(cycle)
    boundary_cycles = tuple(
        cycles[0] for cycles in face_occurrences.values() if len(cycles) == 1
    )
    triangle_rows = tuple(
        triangle
        for cycle in boundary_cycles
        for triangle in ((cycle[0], cycle[1], cycle[2]), (cycle[0], cycle[2], cycle[3]))
    )
    triangles = tuple(np.asarray(triangle, dtype=np.int64) for triangle in triangle_rows)
    segments = np.asarray(
        sorted(
            {
                tuple(sorted((cycle[index], cycle[(index + 1) % 4])))
                for cycle in boundary_cycles
                for index in range(4)
            }
        ),
        dtype=np.int64,
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        triangles,
        np.arange(len(triangles), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(triangles), 1)),
        ("solid",),
        segments=segments,
        vertex_ids=tuple(f"two-root-vertex-{row}" for row in range(vertices.shape[0])),
        vertex_occurrence_paths=((),) * vertices.shape[0],
        facet_ids=tuple(f"two-root-facet-{row}" for row in range(len(triangles))),
        facet_occurrence_paths=((),) * len(triangles),
        segment_ids=tuple(f"two-root-edge-{row}" for row in range(segments.shape[0])),
        segment_occurrence_paths=((),) * segments.shape[0],
        region_occurrence_paths=((),),
    )
    reference_domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(triangles, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(triangles), 1)),
        ("solid",),
        source_id=_SOURCE_ID,
    )
    base_geometry = CellGeometrySpec.affine(reference_mesh)
    matrix = np.diag(np.asarray((2.0, 1.0, 3.0), dtype=np.float64))
    mapped_coordinates = vertices @ matrix.T + _TRANSLATION
    source_geometry = CellGeometrySpec(
        dict(zip(base_geometry.block_names, base_geometry.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(
                base_geometry.block_names, base_geometry.geometry_dofs, strict=True
            )
        },
        mapped_coordinates,
    )
    domain = MappedReferenceDomain(
        reference_domain,
        reference_mesh,
        source_geometry,
        np.asarray((0, 0), dtype=np.int64),
        source_id=_SOURCE_ID,
        source_revision=_SOURCE_REVISION,
        reference_domain_vertex_root_ids=np.asarray(
            reference_mesh.vertex_global_ids, dtype=np.int64
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    reference = NativePlcSource(complex_, _SOURCE_ID, _SOURCE_REVISION)
    return NativeMappedHexSource(
        reference,
        domain,
        plc_vertex_root_global_ids=np.asarray(
            reference_mesh.vertex_global_ids, dtype=np.int64
        ),
    ), domain


def _request(
    source: NativeMappedHexSource,
    /,
    *,
    seed_point: np.ndarray | None = None,
    geometry_order: int = 1,
    target_size: float = 2.0,
    fill_strategy: VolumeFillStrategy = VolumeFillStrategy.SIMPLEX,
) -> VolumeMeshingSpec:
    boundary = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    point = (
        _SCALE * np.asarray((0.5, 0.5, 0.5), dtype=np.float64) + _TRANSLATION
        if seed_point is None
        else np.asarray(seed_point, dtype=np.float64)
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("hexahedron",)),
            geometry_order=geometry_order,
        ),
        boundary,
        fill_strategy,
        size_controls=(
            UniformSizeControl(
                boundary,
                target_size,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_seeds=(
            RegionSeed(
                point,
                "solid",
                "mapped-cube-material",
                RegionRole.SOLID,
            ),
        ),
    )


def test_mapped_plc_auto_binding_requires_matching_source_entity_keys() -> None:
    source, domain = _mapped_cube_source()
    with pytest.raises(ValueError, match="source owner's entity set"):
        NativePlcSource(
            source.reference.complex,
            source.reference.source_id,
            source.reference.source_revision,
            mapped_domain=domain,
        )


def test_mapped_grid_hex_rejects_a_different_physical_frame() -> None:
    source, _domain = _mapped_cube_source()
    with pytest.raises(ValueError, match="coordinate contract must bind"):
        prepare_native_mapped_balanced_grid_hex(
            source,
            _request(source),
            NativeMeshingOptions("mapped_balanced_grid_hex"),
            SpatialCoordinateContract(METER, reference_frame="local-frame"),
        )


@pytest.mark.parametrize(
    "fill_strategy", (VolumeFillStrategy.SIMPLEX, VolumeFillStrategy.MULTIZONE)
)
@pytest.mark.parametrize(
    ("route", "prepare"),
    (
        ("mapped_balanced_grid_hex", prepare_native_mapped_balanced_grid_hex),
        ("mapped_frame_grid_hex", prepare_native_mapped_frame_grid_hex),
    ),
)
def test_mapped_grid_hex_routes_execute_with_root_map_and_source_evidence(
    route: Literal["mapped_balanced_grid_hex", "mapped_frame_grid_hex"],
    fill_strategy: VolumeFillStrategy,
    prepare: Callable[..., Any],
) -> None:
    source, domain = _mapped_cube_source()
    complex_ = source.reference.complex
    relation = source.root_relation
    np.testing.assert_array_equal(
        relation.segment_source_entity_ids, complex_.plc_edge_entity_ids
    )
    assert relation.segment_source_entity_keys == tuple(
        source.reference.source_entity_key_for_scope(1, int(identifier))[0]
        for identifier in relation.segment_source_entity_ids.tolist()
    )
    assert relation.segment_source_occurrence_paths == tuple(
        source.reference.source_entity_key_for_scope(1, int(identifier))[1]
        for identifier in relation.segment_source_entity_ids.tolist()
    )
    expected_facet_ids = np.asarray(complex_.facet_entity_ids, dtype=np.int64)[
        np.asarray(relation.polygon_source_facet_rows, dtype=np.int64)
    ]
    np.testing.assert_array_equal(relation.polygon_source_entity_ids, expected_facet_ids)
    assert relation.polygon_source_entity_keys == tuple(
        source.reference.source_entity_key_for_scope(2, int(identifier))[0]
        for identifier in expected_facet_ids.tolist()
    )
    assert relation.polygon_source_occurrence_paths == tuple(
        source.reference.source_entity_key_for_scope(2, int(identifier))[1]
        for identifier in expected_facet_ids.tolist()
    )
    construction = prepare(
        source,
        _request(source, fill_strategy=fill_strategy),
        NativeMeshingOptions(route),
        SpatialCoordinateContract.si(),
    ).execute()

    assert isinstance(construction, NativeMeshConstruction)
    assert all(block.cell_kind == "hexahedron" for block in construction.mesh.blocks)
    np.testing.assert_allclose(
        np.min(np.asarray(construction.mesh.coordinates), axis=0), _TRANSLATION
    )
    np.testing.assert_allclose(
        np.max(np.asarray(construction.mesh.coordinates), axis=0),
        _TRANSLATION + _SCALE,
    )
    assert construction.certification.domain is domain
    assert construction.certification.source_id == _SOURCE_ID
    assert construction.certification.source_revision == _SOURCE_REVISION
    assert construction.certification.cell_regions is not None
    assert construction.certification.cell_regions.shape == (
        construction.mesh.connectivity.cell_count,
    )
    assert np.all(construction.certification.cell_regions == 0)
    assert construction.geometry.restriction_source is not None
    elements, _routes, _coordinates = construction.geometry.resolve(construction.mesh)
    assert isinstance(elements[0], PerCellRestrictedCellGeometryElement)
    associations = construction.organization.associations
    assert associations
    assert all(
        association.association_kind is GeometryAssociationKind.MAPPED_REFERENCE
        and association.source_id == _SOURCE_ID
        and association.source_revision == _SOURCE_REVISION
        for association in associations
    )
    assert construction.reference_maps
    assert construction.reference_maps[0].source_reference_chart_arguments is not None
    assert np.all(
        construction.reference_maps[0].source_cell_ids == domain.root_cell_ids[0]
    )
    first_block = construction.mesh.blocks[0]
    first_cell_id = np.asarray(first_block.global_ids, dtype=np.int64)[:1]
    reference_vertices = np.asarray(
        reference_cell_topology("hexahedron").vertices, dtype=np.float64
    )
    mapped_corners = construction.geometry.evaluate(
        construction.mesh,
        first_cell_id,
        reference_vertices[np.newaxis, :, :],
    )
    np.testing.assert_allclose(
        mapped_corners[0],
        np.asarray(construction.mesh.coordinates)[
            np.asarray(first_block.vertices, dtype=np.int64)[0]
        ],
    )


@pytest.mark.parametrize(
    ("route", "prepare"),
    (
        ("mapped_balanced_grid_hex", prepare_native_mapped_balanced_grid_hex),
        ("mapped_frame_grid_hex", prepare_native_mapped_frame_grid_hex),
    ),
)
def test_mapped_grid_hex_routes_execute_nonuniform_affine_multiple_roots(
    route: Literal["mapped_balanced_grid_hex", "mapped_frame_grid_hex"],
    prepare: Callable[..., Any],
) -> None:
    source, domain = _mapped_two_root_source()
    assert isinstance(domain.reference_mesh.connectivity, PolyhedralConnectivity)
    matrix = np.diag(np.asarray((2.0, 1.0, 3.0), dtype=np.float64))
    seed = matrix @ np.asarray((0.5, 0.5, 0.5), dtype=np.float64) + _TRANSLATION
    construction = prepare(
        source,
        _request(source, seed_point=seed),
        NativeMeshingOptions(route),
        SpatialCoordinateContract.si(),
    ).execute()
    assert isinstance(construction, NativeMeshConstruction)
    assert isinstance(construction.mesh.connectivity, PolyhedralConnectivity)
    np.testing.assert_allclose(
        np.min(np.asarray(construction.mesh.coordinates), axis=0), _TRANSLATION
    )
    np.testing.assert_allclose(
        np.max(np.asarray(construction.mesh.coordinates), axis=0),
        _TRANSLATION + np.asarray((4.0, 1.0, 3.0)),
    )
    assert construction.certification.domain is domain
    assert construction.geometry.restriction_source is not None
    assert construction.reference_maps
    np.testing.assert_array_equal(
        np.unique(
            np.concatenate(
                [
                    np.asarray(mapping.source_cell_ids, dtype=np.int64)
                    for mapping in construction.reference_maps
                ]
            )
        ),
        np.asarray((17, 29), dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("route", "prepare"),
    (
        ("mapped_balanced_grid_hex", prepare_native_mapped_balanced_grid_hex),
        ("mapped_frame_grid_hex", prepare_native_mapped_frame_grid_hex),
    ),
)
def test_mapped_grid_hex_routes_restrict_curved_source_maps(
    route: Literal["mapped_balanced_grid_hex", "mapped_frame_grid_hex"],
    prepare: Callable[..., Any],
) -> None:
    source, domain = _mapped_cube_source(curved=True)
    seed = np.asarray((2.0, -1.0, 1.53125), dtype=np.float64)
    construction = prepare(
        source,
        _request(source, seed_point=seed, geometry_order=2),
        NativeMeshingOptions(route),
        SpatialCoordinateContract.si(),
    ).execute()

    assert isinstance(construction, NativeMeshConstruction)
    np.testing.assert_allclose(
        np.min(np.asarray(construction.mesh.coordinates), axis=0),
        np.asarray((1.0, -2.0, 0.5)),
    )
    np.testing.assert_allclose(
        np.max(np.asarray(construction.mesh.coordinates), axis=0),
        np.asarray((3.0, 0.0, 2.625)),
    )
    assert construction.certification.domain is domain
    assert construction.geometry.restriction_source is not None
    elements, _routes, _coordinates = construction.geometry.resolve(construction.mesh)
    assert isinstance(elements[0], PerCellRestrictedCellGeometryElement)
    assert construction.reference_maps


@pytest.mark.parametrize("curved", (False, True))
def test_mapped_grid_hex_certifies_exact_restriction_without_requested_deviation(
    curved: bool,
) -> None:
    source, _domain = _mapped_cube_source(curved=curved)
    seed = np.asarray((2.0, -1.0, 1.53125), dtype=np.float64) if curved else None
    specification = _request(source, seed_point=seed, geometry_order=2 if curved else 1)
    assert specification.maximum_deviation is None
    prepared = prepare_native_mapped_balanced_grid_hex(
        source,
        specification,
        NativeMeshingOptions("mapped_balanced_grid_hex"),
        SpatialCoordinateContract.si(),
    )
    assert prepared.execute().certification.fidelity_tolerance == 0.0

    result = (
        NativeMeshingProvider(NativeMeshingOptions("mapped_balanced_grid_hex"))
        .plan(source, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )

    assert result.certification is not None
    fidelity = result.certification.fidelity
    assert fidelity is not None
    assert fidelity.status == "certified"
    assert fidelity.mesh_to_source_upper == 0.0
    assert fidelity.source_to_mesh_upper == 0.0


def test_mapped_grid_hex_preserves_multiple_charts_per_source_face() -> None:
    source, _ = _mapped_cube_source(shared_facets=True)
    query = source.boundary_query()
    # Charts are exact source-triangle/root-fan-triangle overlay pieces. The
    # canonical root face (0, 3, 7, 4) fans across diagonal 0-7 while its source
    # triangles share diagonal 3-4, so that face carries four charts; the other
    # five faces' diagonals coincide and carry two each.
    assert len(query.face_chart_ids) == 14
    assert len(set(query.face_chart_ids)) == 14
    assert len(set(query.face_scope_keys)) == 6
    result = (
        NativeMeshingProvider(NativeMeshingOptions("mapped_balanced_grid_hex"))
        .plan(
            source, _request(source), coordinate_contract=SpatialCoordinateContract.si()
        )
        .execute()
    )
    assert result.certification is not None
    assert result.certification.passed
    assert result.certification.fidelity is not None
    assert result.certification.fidelity.status == "certified"


def test_mapped_grid_hex_retains_root_region_and_patch_controls() -> None:
    source, _domain = _mapped_cube_source()
    complex_ = source.reference.complex
    relation = source.root_relation
    root_face_id = int(relation.polygon_root_face_global_ids[0])
    polygon_rows = np.flatnonzero(
        np.asarray(relation.polygon_root_face_global_ids) == root_face_id
    )
    facet_rows = np.asarray(relation.polygon_source_facet_rows, dtype=np.int64)[
        polygon_rows
    ]
    patch_entity_ids = np.sort(
        np.asarray(complex_.facet_entity_ids, dtype=np.int64)[facet_rows]
    )
    patch_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        patch_entity_ids,
    )
    region_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    base = _request(source)
    specification = VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=base.size_controls,
        region_seeds=base.region_seeds,
        region_controls=(
            RegionControl(
                region_scope, "solid", "mapped-cube-material", RegionRole.SOLID
            ),
        ),
        patch_controls=(PatchControl("mapped-root-face", patch_scope, ("solid",)),),
    )
    construction = prepare_native_mapped_balanced_grid_hex(
        source,
        specification,
        NativeMeshingOptions("mapped_balanced_grid_hex"),
        SpatialCoordinateContract.si(),
    ).execute()

    assert len(construction.organization.zones) == 1
    zone = construction.organization.zones[0]
    assert zone.name == "solid"
    assert zone.material_id == "mapped-cube-material"
    assert zone.region_role is RegionRole.SOLID
    assert len(construction.organization.patches) == 1
    patch = construction.organization.patches[0]
    assert patch.name == "mapped-root-face"
    assert patch.connected
    assert patch.adjacent_zone_ids == (zone.zone_id,)


def _mapped_surface_chart_witness(
    source: NativeMappedHexSource | NativePlcSource,
    domain: MappedReferenceDomain,
    geometry: CellGeometrySpec,
) -> tuple[MappedReferenceBoundarySource, SourceBoundaryChartWitness]:
    mesh = domain.reference_mesh
    query = source.boundary_query()
    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError(
            "Mapped root witness construction requires MappedReferenceBoundarySource."
        )
    chart_ids = tuple(query.face_chart_ids)
    rows = tuple(query.chart_row(2, chart_id) for chart_id in chart_ids)
    source_keys = tuple(query.chart_source_key(2, chart_id) for chart_id in chart_ids)
    reference_tokens = tuple(
        query.polygon_root_reference_rational_tokens[row] for row in rows
    )
    references = np.asarray(
        [
            [[float(Fraction(*value)) for value in point] for point in row]
            for row in reference_tokens
        ],
        dtype=np.float64,
    )
    parameters = np.broadcast_to(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (len(rows), 3, 2),
    )
    cell_ids = np.asarray(
        [query.polygon_root_cell_global_ids[row] for row in rows], dtype=np.int64
    )
    face_ids = np.asarray(
        [query.polygon_root_face_global_ids[row] for row in rows], dtype=np.int64
    )
    local_faces = np.asarray(
        [query.polygon_root_local_face_indices[row] for row in rows], dtype=np.int64
    )

    def make(orientations: np.ndarray) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=query.query_id,
            chart_dimension=2,
            reference_dimension=3,
            cell_global_ids=cell_ids,
            entity_global_ids=face_ids,
            local_entity_indices=local_faces,
            target_reference_vertices=references,
            target_reference_rational_tokens=reference_tokens,
            source_chart_ids=chart_ids,
            source_entity_ids=tuple(key for key, _ in source_keys),
            occurrence_paths=tuple(path for _, path in source_keys),
            source_parameters=parameters,
            orientations=orientations,
        )

    preliminary = make(np.ones((len(rows),), dtype=np.int8))
    _, _, target_orientation = _surface_target_chart_rows(
        mesh, geometry, None, preliminary, full_source=True
    )
    orientation_by_chart = {
        chart_id: target_sign
        for chart_id, target_sign in zip(
            preliminary.source_chart_ids, target_orientation, strict=True
        )
    }
    orientations = np.asarray(
        [orientation_by_chart[chart_id] for chart_id in chart_ids], dtype=np.int8
    )
    return query, make(orientations)


def test_mapped_fidelity_target_map_reuse_preserves_exact_maps_and_refusal() -> None:
    from collections import OrderedDict

    from phydrax.geometry._mapped_reference_interpolation import (
        _coordinate_expression,
        _mapped_chart_maps,
    )

    source, domain = _mapped_cube_source(curved=True)
    mesh, geometry = domain.reference_mesh, domain.source_geometry
    query, witness = _mapped_surface_chart_witness(source, domain, geometry)
    elements, routes, _ = geometry.resolve(mesh)
    coordinates = geometry.source_coordinates()
    cache = OrderedDict()
    preparations = {}
    for row in (0, 1):
        source_base = tuple(
            _coordinate_expression(value)
            for value in query.chart_coordinate_expressions(
                2, witness.source_chart_ids[row]
            )
        )
        args = (
            mesh,
            witness,
            row,
            (0, 0),
            source_base,
            elements,
            routes,
            coordinates,
            None,
        )
        expected = _mapped_chart_maps(*args, OrderedDict(), {})
        assert _mapped_chart_maps(*args, cache, preparations) == expected
    malformed = source_base[:2]
    args = (mesh, witness, 0, (0, 0), malformed, elements, routes, coordinates, None)
    for window in (cache, OrderedDict()):
        with pytest.raises(ValueError, match="three physical dimensions"):
            _mapped_chart_maps(*args, window, {})


def test_mapped_root_source_query_certifies_exact_affine_boundary_and_normals() -> None:
    source, domain = _mapped_cube_source()
    query, witness = _mapped_surface_chart_witness(source, domain, domain.source_geometry)
    assert query.coordinate_contract is domain.coordinate_contract

    certificate = certify_source_fidelity(
        domain.reference_mesh,
        domain.source_geometry,
        query,
        tolerance=0.0,
        maximum_normal_deviation=0.0,
        witness=witness,
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0
    assert certificate.maximum_normal_deviation_bound == 0.0
    certificate.require_current(domain.reference_mesh, domain.source_geometry, query)

    source_vertex_id = int(source.reference.complex.vertex_entity_ids[0])
    source_key = source.source_entity_key_for_scope(0, source_vertex_id)
    root_vertex_id = int(source.root_relation.vertex_root_global_ids[0])
    point_witness = SourceBoundaryPointWitness(
        mesh_id=domain.reference_mesh.mesh_id,
        geometry_id=cell_geometry_id(domain.source_geometry),
        topology_id=domain.reference_mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.query_id,
        vertex_global_ids=np.asarray((root_vertex_id,), dtype=np.int64),
        source_entity_ids=(source_key[0],),
        occurrence_paths=(source_key[1],),
    )
    point_certificate = certify_source_fidelity(
        domain.reference_mesh,
        domain.source_geometry,
        query,
        tolerance=0.0,
        witness=point_witness,
    )
    assert point_certificate.status == "certified"
    assert point_certificate.mesh_to_source_upper == 0.0


def test_mapped_root_curved_source_certifies_explicit_bounded_error() -> None:
    source, domain = _mapped_cube_source(curved=True)
    root_geometry = domain.source_geometry
    coordinates = np.array(root_geometry.coordinates, dtype=np.float64, copy=True)
    coordinates[:, 2] += 0.125
    geometry = CellGeometrySpec(
        dict(zip(root_geometry.block_names, root_geometry.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(
                root_geometry.block_names, root_geometry.geometry_dofs, strict=True
            )
        },
        coordinates,
    )
    query, witness = _mapped_surface_chart_witness(source, domain, geometry)

    certificate = certify_source_fidelity(
        domain.reference_mesh,
        geometry,
        query,
        tolerance=0.25,
        witness=witness,
    )
    exact = certify_source_fidelity(
        domain.reference_mesh,
        geometry,
        query,
        tolerance=0.0,
        witness=witness,
    )

    assert certificate.status == "certified"
    assert 0.0 < certificate.mesh_to_source_upper <= 0.25
    assert 0.0 < certificate.source_to_mesh_upper <= 0.25
    assert exact.status == "violated"


def test_native_plc_source_revalidates_tetra_root_boundary_and_curved_bounds() -> None:
    affine_source, affine_domain = _mapped_plc_tetra_source()
    query, witness = _mapped_surface_chart_witness(
        affine_source, affine_domain, affine_domain.source_geometry
    )
    relation = affine_source.root_relation
    assert relation is not None
    assert query.authority is affine_source
    root_face_ids = np.asarray(affine_domain.reference_mesh.entity_set(2).entity_ids)
    _, cell_faces, _ = _mapped_root_entity_banks(affine_domain.reference_mesh)
    root_cell_rows = {
        int(identifier): row for row, identifier in enumerate(affine_domain.root_cell_ids)
    }
    for face_id, cell_ids, local_faces in zip(
        relation.polygon_root_face_global_ids,
        relation.polygon_root_cell_global_ids,
        relation.polygon_root_local_face_indices,
        strict=True,
    ):
        incident = tuple(
            (int(cell_id), int(local_face))
            for cell_id, local_face in zip(cell_ids, local_faces, strict=True)
            if int(cell_id) >= 0
        )
        assert incident
        for cell_id, local_face in incident:
            root_row = root_cell_rows[cell_id]
            assert int(root_face_ids[cell_faces[root_row, local_face]]) == int(face_id)
    for entity_id, key, occurrence in zip(
        relation.polygon_source_entity_ids,
        relation.polygon_source_entity_keys,
        relation.polygon_source_occurrence_paths,
        strict=True,
    ):
        assert affine_source.source_entity_key_for_scope(2, int(entity_id)) == (
            key,
            occurrence,
        )
    certificate = certify_source_fidelity(
        affine_domain.reference_mesh,
        affine_domain.source_geometry,
        query,
        tolerance=0.0,
        maximum_normal_deviation=0.0,
        witness=witness,
    )
    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    query.require_current()
    certificate.require_current(
        affine_domain.reference_mesh, affine_domain.source_geometry, query
    )
    affine_source.validate_restored()

    curved_source, curved_domain = _mapped_plc_tetra_source(curved=True)
    root_geometry = curved_domain.source_geometry
    coordinates = np.array(root_geometry.coordinates, dtype=np.float64, copy=True)
    coordinates[:, 2] += 0.125
    candidate_geometry = CellGeometrySpec(
        dict(zip(root_geometry.block_names, root_geometry.elements, strict=True)),
        {
            name: np.asarray(route)
            for name, route in zip(
                root_geometry.block_names, root_geometry.geometry_dofs, strict=True
            )
        },
        coordinates,
    )
    curved_query, curved_witness = _mapped_surface_chart_witness(
        curved_source, curved_domain, candidate_geometry
    )
    bounded = certify_source_fidelity(
        curved_domain.reference_mesh,
        candidate_geometry,
        curved_query,
        tolerance=0.25,
        witness=curved_witness,
    )
    unqualified = certify_source_fidelity(
        curved_domain.reference_mesh,
        candidate_geometry,
        curved_query,
        tolerance=0.0,
        witness=curved_witness,
    )
    assert bounded.status == "certified"
    assert 0.0 < bounded.mesh_to_source_upper <= 0.25
    assert unqualified.status == "violated"


def _slice_mapped_chart_witness(
    witness: SourceBoundaryChartWitness,
    rows: tuple[int, ...],
    /,
    *,
    occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
) -> SourceBoundaryChartWitness:
    selected_face_vertices = (
        np.empty((0, 3), dtype=np.int64)
        if np.asarray(witness.target_face_vertex_global_ids).size == 0
        else np.asarray(witness.target_face_vertex_global_ids)[list(rows)]
    )
    return SourceBoundaryChartWitness(
        mesh_id=witness.mesh_id,
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        source_id=witness.source_id,
        source_revision=witness.source_revision,
        source_query_id=witness.source_query_id,
        chart_dimension=witness.chart_dimension,
        reference_dimension=witness.reference_dimension,
        cell_global_ids=np.asarray(witness.cell_global_ids)[list(rows)],
        entity_global_ids=np.asarray(witness.entity_global_ids)[list(rows)],
        local_entity_indices=np.asarray(witness.local_entity_indices)[list(rows)],
        target_reference_vertices=np.asarray(witness.target_reference_vertices)[
            list(rows)
        ],
        target_reference_rational_tokens=tuple(
            witness.target_reference_rational_tokens[row] for row in rows
        ),
        target_face_vertex_global_ids=selected_face_vertices,
        source_chart_ids=tuple(witness.source_chart_ids[row] for row in rows),
        source_entity_ids=tuple(witness.source_entity_ids[row] for row in rows),
        occurrence_paths=(
            tuple(witness.occurrence_paths[row] for row in rows)
            if occurrence_paths is None
            else occurrence_paths
        ),
        source_parameters=np.asarray(witness.source_parameters)[list(rows)],
        source_parameter_rational_tokens=tuple(
            witness.source_parameter_rational_tokens[row] for row in rows
        ),
        orientations=np.asarray(witness.orientations)[list(rows)],
        implicit_cell_keys=np.asarray(witness.implicit_cell_keys)[list(rows)]
        if witness.implicit_cell_keys.shape[0]
        else None,
        implicit_root_edge_keys=np.asarray(witness.implicit_root_edge_keys)[list(rows)]
        if witness.implicit_root_edge_keys.shape[0]
        else None,
    )


@pytest.mark.parametrize(
    ("field", "path"),
    (
        ("polygon_root_reference_rational_tokens", (0, 0, 0, 0)),
        ("segment_root_reference_rational_tokens", (0, 0, 0)),
        ("segment_source_parameter_rational_tokens", (0, 0)),
    ),
)
def test_mapped_source_rejects_relation_token_tamper_with_original_ids(
    field: str,
    path: tuple[int, ...],
) -> None:
    from dataclasses import replace

    import equinox as eqx

    source, _ = _mapped_cube_source()
    relation = source.root_relation

    def alter(value: tuple, indices: tuple[int, ...]) -> tuple:
        if not indices:
            numerator, denominator = value
            return (numerator + denominator, denominator)
        rows = list(value)
        rows[indices[0]] = alter(rows[indices[0]], indices[1:])
        return tuple(rows)

    forged_relation = replace(relation, **{field: alter(getattr(relation, field), path)})
    forged = eqx.tree_at(lambda value: value.root_relation, source, forged_relation)
    assert forged.root_relation.relation_id == relation.relation_id
    assert forged.binding_id == source.binding_id
    with pytest.raises(ValueError, match="relation"):
        forged.validate_restored()


def test_mapped_root_face_partition_accepts_nonvertex_corners_and_rejects_tamper() -> (
    None
):
    source, domain = _mapped_cube_source()
    query = source.boundary_query()
    triangles = np.asarray(
        [
            [[Fraction(*value) for value in point] for point in row]
            for row in query.polygon_root_reference_rational_tokens
        ],
        dtype=object,
    )
    triangle = triangles[0]
    midpoint = (triangle[0] + triangle[1]) / 2
    pieces = np.stack(
        (
            np.stack((triangle[0], midpoint, triangle[2])),
            np.stack((midpoint, triangle[1], triangle[2])),
        )
    )
    refs = np.concatenate((pieces, triangles[1:]))

    def repeat_first(values: np.ndarray) -> np.ndarray:
        return np.concatenate((values[:1], values))

    def validate(points: np.ndarray) -> None:
        MappedReferenceBoundarySource._validate_root_relation(
            domain,
            repeat_first(query.polygon_root_face_global_ids),
            repeat_first(query.polygon_root_cell_global_ids),
            repeat_first(query.polygon_root_local_face_indices),
            points,
            repeat_first(query.polygon_orientation_signs),
            query.segment_root_dimensions,
            query.segment_root_entity_global_ids,
            query.segment_root_cell_global_ids,
            query.segment_root_local_entity_indices,
            tuple(
                tuple(tuple(Fraction(*value) for value in point) for point in row)
                for row in query.segment_root_reference_rational_tokens
            ),
            query.segment_orientations,
            False,
        )

    validate(refs)
    changed = refs.copy()
    changed[1] = changed[0]
    with pytest.raises(ValueError):
        validate(changed)


def test_mapped_root_chart_validator_rejects_omission_and_occurrence_change() -> None:
    source, domain = _mapped_cube_source()
    query, witness = _mapped_surface_chart_witness(source, domain, domain.source_geometry)
    with pytest.raises(ValueError):
        validate_source_fidelity_witness(
            domain.reference_mesh,
            domain.source_geometry,
            query,
            _slice_mapped_chart_witness(
                witness, tuple(range(len(witness.source_chart_ids) - 1))
            ),
        )

    changed_paths = list(witness.occurrence_paths)
    changed_paths[0] = ("wrong-source-occurrence",)
    with pytest.raises(ValueError):
        validate_source_fidelity_witness(
            domain.reference_mesh,
            domain.source_geometry,
            query,
            _slice_mapped_chart_witness(
                witness,
                tuple(range(len(witness.source_chart_ids))),
                occurrence_paths=tuple(changed_paths),
            ),
        )


def _mapped_curve_chart_witness(
    source: NativeMappedHexSource,
    domain: MappedReferenceDomain,
    geometry: CellGeometrySpec,
    *,
    include_facet_carriers: bool = False,
) -> tuple[MappedReferenceBoundarySource, SourceBoundaryChartWitness]:
    mesh = domain.reference_mesh
    edge_scope = (
        None
        if include_facet_carriers
        else tuple(
            zip(
                source.complex.segment_ids,
                source.complex.segment_occurrence_paths,
                strict=True,
            )
        )
    )
    query = source.boundary_query(edge_scope_keys=edge_scope)
    if not isinstance(query, MappedReferenceBoundarySource):
        raise TypeError(
            "Mapped curve witness construction requires MappedReferenceBoundarySource."
        )
    selected = set(query.selected_source_keys(1))
    chart_ids = tuple(
        chart_id
        for chart_id, key, path in zip(
            query.edge_chart_ids,
            query.segment_source_entity_keys,
            query.segment_source_occurrence_paths,
            strict=True,
        )
        if (key, path) in selected
    )
    rows = tuple(query.chart_row(1, chart_id) for chart_id in chart_ids)
    source_keys = tuple(query.chart_source_key(1, chart_id) for chart_id in chart_ids)
    reference_tokens = tuple(
        query.segment_root_reference_rational_tokens[row] for row in rows
    )
    references = np.asarray(
        [
            [[float(Fraction(*value)) for value in point] for point in row]
            for row in reference_tokens
        ],
        dtype=np.float64,
    )
    source_parameters = np.broadcast_to(
        np.asarray(((0.0,), (1.0,)), dtype=np.float64), (len(rows), 2, 1)
    )
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=query.query_id,
        chart_dimension=1,
        reference_dimension=mesh.topological_dimension,
        cell_global_ids=np.asarray(
            [query.segment_root_cell_global_ids[row] for row in rows], dtype=np.int64
        ),
        entity_global_ids=np.asarray(
            [query.segment_root_entity_global_ids[row] for row in rows], dtype=np.int64
        ),
        local_entity_indices=np.asarray(
            [query.segment_root_local_entity_indices[row] for row in rows], dtype=np.int64
        ),
        target_reference_vertices=references,
        target_reference_rational_tokens=reference_tokens,
        source_chart_ids=chart_ids,
        source_entity_ids=tuple(key for key, _ in source_keys),
        occurrence_paths=tuple(path for _, path in source_keys),
        source_parameters=source_parameters,
        orientations=np.asarray(
            [query.segment_orientations[row] for row in rows], dtype=np.int8
        ),
    )
    return query, witness


def test_mapped_root_curve_query_certifies_exact_affine_source_charts() -> None:
    source, domain = _mapped_cube_source()
    query, witness = _mapped_curve_chart_witness(source, domain, domain.source_geometry)

    certificate = certify_source_fidelity(
        domain.reference_mesh,
        domain.source_geometry,
        query,
        tolerance=0.0,
        witness=witness,
    )

    assert certificate.status == "certified"
    assert certificate.mesh_to_source_upper == 0.0
    assert certificate.source_to_mesh_upper == 0.0


def test_mapped_curve_witness_refuses_facet_carrier_as_target_edge() -> None:
    source, domain = _mapped_cube_source()
    query, witness = _mapped_curve_chart_witness(
        source, domain, domain.source_geometry, include_facet_carriers=True
    )

    with pytest.raises(ValueError):
        certify_source_fidelity(
            domain.reference_mesh,
            domain.source_geometry,
            query,
            tolerance=0.0,
            witness=witness,
        )


def test_mapped_physical_improvement_keeps_unmet_quality_for_native_noop() -> None:
    from phydrax._meshcore import MeshcoreStatus, TetMesh3D, TetMeshEditStatus
    from phydrax.meshing._layer_core_mapped import _propose_mapped_physical_improvement
    from phydrax.meshing._volume_generation import _unmet_rows

    _, base = _mapped_plc_tetra_source()
    reference = base.reference_mesh
    geometry = base.source_geometry
    physical_points = np.asarray(geometry.coordinates) * np.asarray((1.0, 1.0, 0.05))
    stretched_geometry = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        physical_points,
    )
    assert base.reference_domain_vertex_root_ids is not None
    domain = MappedReferenceDomain(
        base.reference_domain,
        reference,
        stretched_geometry,
        base.cell_regions,
        source_id=base.source_id,
        source_revision=base.source_revision,
        reference_domain_vertex_root_ids=np.asarray(
            base.reference_domain_vertex_root_ids
        ),
        coordinate_contract=base.coordinate_contract,
    )
    points = np.asarray(reference.coordinates)
    tetrahedra = np.asarray(reference.blocks[0].vertices)
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    segments = np.empty((0, 2), dtype=np.int32)
    segment_sources = np.empty((0,), dtype=np.int32)
    vertex_ids = np.asarray((31, 37, 41, 43), dtype=np.int64)
    cell_ids = np.asarray((901,), dtype=np.int64)
    schedule = NativeVolumeSchedule(minimum_dihedral_degrees=60.0)

    def make_state(coordinates: np.ndarray) -> TetMesh3D:
        return TetMesh3D(
            coordinates,
            tetrahedra,
            np.asarray((0,), dtype=np.int32),
            faces,
            np.arange(4, dtype=np.int32),
            segments,
            segment_sources,
            protected_vertices=np.ones(4, dtype=np.bool_),
            vertex_ids=vertex_ids,
            cell_ids=cell_ids,
        )

    quality_args = {
        "radius_edge_bound": schedule.radius_edge_bound,
        "min_dihedral_degrees": schedule.minimum_dihedral_degrees,
        "minimum_relative_determinant": schedule.minimum_relative_determinant,
    }
    with make_state(physical_points) as physical:
        original_identity = physical.identity
        original_unmet = _unmet_rows(physical.evaluate_quality(**quality_args))
        assert original_unmet
        run = physical.improve(
            radius_edge_bound=schedule.radius_edge_bound,
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            minimum_relative_determinant=schedule.minimum_relative_determinant,
        )
        assert run.edit_status is TetMeshEditStatus.APPLIED
        assert run.status in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED)
        assert physical.identity == original_identity
        assert _unmet_rows(physical.evaluate_quality(**quality_args)) == original_unmet

    with make_state(points) as source:
        original_identity = source.identity
        arrays = source.arrays()
        candidate_quality = source.evaluate_candidate_quality(
            arrays.points, vertex_ids=arrays.vertex_ids, **quality_args
        )
        consumer_unmet = _unmet_rows(candidate_quality)
        assert consumer_unmet
        proposal = _propose_mapped_physical_improvement(
            domain,
            arrays,
            np.asarray((0,), dtype=np.int64),
            vertex_ids,
            schedule,
        )
        # The consumer retains its original unmet evidence when no proposal exists.
        assert proposal is None
        assert _unmet_rows(candidate_quality) == consumer_unmet
        assert source.identity == original_identity
        np.testing.assert_array_equal(source.arrays().points, points)


@pytest.mark.parametrize("curved", (False, True))
def test_mapped_plc_tetra_volume_publishes_exact_reference_maps(
    curved: bool,
) -> None:
    source, domain = _mapped_plc_tetra_source(curved=curved)
    boundary_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    region_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
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
        maximum_deviation=source.maximum_deviation,
        region_controls=(
            RegionControl(region_scope, "bulk", "bulk-material", RegionRole.SOLID),
        ),
        region_seeds=(
            RegionSeed(
                np.asarray((0.1, 0.1, 0.1), dtype=np.float64),
                "bulk",
                "bulk-material",
                RegionRole.SOLID,
            ),
        ),
    )
    schedule = NativeVolumeSchedule(
        radius_edge_bound=8.0,
        minimum_dihedral_degrees=0.0,
    )
    construction = prepare_native_plc_volume(
        source,
        specification,
        NativeMeshingOptions("plc_tetrahedral", volume_schedule=schedule),
        domain.coordinate_contract,
    ).execute()

    assert construction.source_mesh is domain.reference_mesh
    assert construction.source_geometry is domain.source_geometry
    assert len(construction.reference_maps) == 1
    reference_map = construction.reference_maps[0]
    block = construction.mesh.blocks[0]
    assert reference_map.target_cell_kind == block.cell_kind
    assert reference_map.source_cell_kind == "tetrahedron"
    np.testing.assert_array_equal(reference_map.target_cell_ids, block.global_ids)
    np.testing.assert_array_equal(
        reference_map.source_cell_ids,
        np.full((block.cell_count,), 901, dtype=np.int64),
    )
    assert reference_map.source_reference_exact_vertices is not None
    assert len(reference_map.source_reference_exact_vertices) == block.cell_count
    assert reference_map.source_reference_chart_arguments is not None
    assert construction.certification.domain is domain
    assert construction.certification.fidelity_witness is not None
    assert construction.certification.fidelity_tolerance == source.maximum_deviation


def _mapped_rectangular_split_source() -> NativeMappedHexSource:
    """A 1-by-3/4 source face whose diagonal crosses a quarter-grid line at 1/3."""
    template, _ = _mapped_cube_source()
    complex_ = template.complex
    corners = np.asarray(complex_.vertices, dtype=np.float64).copy()
    corners[:, 1] *= 0.75
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygons = tuple(
        np.asarray(
            complex_.polygon_vertices[offsets[row] : offsets[row + 1]], dtype=np.int64
        )
        for row in range(len(offsets) - 1)
    )
    reference = PiecewiseLinearDomain(
        corners,
        np.asarray(polygons, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(polygons), 1)),
        ("solid",),
        source_id="mapped-rectangular-split",
    )
    source_complex = PiecewiseLinearComplex(
        corners,
        polygons,
        np.asarray(complex_.polygon_facets, dtype=np.int64),
        np.asarray(complex_.facet_regions, dtype=np.int64),
        ("solid",),
        segments=np.asarray(complex_.segments, dtype=np.int64),
        vertex_ids=complex_.vertex_ids,
        vertex_occurrence_paths=complex_.vertex_occurrence_paths,
        facet_ids=complex_.facet_ids,
        facet_occurrence_paths=complex_.facet_occurrence_paths,
        segment_ids=complex_.segment_ids,
        segment_occurrence_paths=complex_.segment_occurrence_paths,
        region_occurrence_paths=complex_.region_occurrence_paths,
    )
    xs = (0.0, 0.25, 0.5, 0.75, 1.0)
    ys = (0.0, 0.25, 0.5, 0.75)
    points = np.asarray([(x, y, z) for z in (0.0, 1.0) for y in ys for x in xs])
    point_rows = {tuple(point): row for row, point in enumerate(points)}
    topology = reference_cell_topology("hexahedron")
    cells = np.asarray(
        [
            tuple(
                point_rows[(xs[i + int(u)], ys[j + int(v)], float(w))]
                for u, v, w in topology.vertices
            )
            for j in range(len(ys) - 1)
            for i in range(len(xs) - 1)
        ],
        dtype=np.int32,
    )
    mesh = CellMesh(points, (CellBlock("root", "hexahedron", cells),))
    corner_ids = np.asarray(
        [mesh.vertex_global_ids[point_rows[tuple(point)]] for point in corners],
        dtype=np.int64,
    )
    domain = MappedReferenceDomain(
        reference,
        mesh,
        CellGeometrySpec.affine(mesh),
        np.zeros(len(cells), dtype=np.int64),
        source_id="mapped-rectangular-split",
        source_revision="r1",
        reference_domain_vertex_root_ids=corner_ids,
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    source = NativePlcSource(source_complex, domain.source_id, domain.source_revision)
    return NativeMappedHexSource(source, domain, plc_vertex_root_global_ids=corner_ids)


def test_original_frame_query_restricts_whole_facets_to_root_pieces() -> None:
    from phydrax.discretization._coordinate_enclosure import evaluate_expression
    from tools.meshing_qualification.scenarios.native_family_matrix import (
        _frame_sweep_source,
    )

    case = _frame_sweep_source(4)
    query = case.source.fidelity_source
    assert isinstance(query, MappedReferenceBoundarySource)
    assert len(query.authority.complex.polygon_facets) == 12
    assert len(query.face_chart_ids) > 12
    assert set(query.polygon_source_facet_rows.tolist()) == set(range(12))
    assert query.boundary_chart_cover() is not None
    for row, chart_id in enumerate(query.face_chart_ids):
        refs = tuple(
            tuple(Fraction(*value) for value in point)
            for point in query.polygon_root_reference_rational_tokens[row]
        )
        cell = int(query.polygon_root_cell_global_ids[row])
        expected = query.domain.source_geometry.evaluate_exact(
            query.domain.reference_mesh, np.asarray((cell,), dtype=np.int64), (refs,)
        )[0]
        expressions = query.chart_coordinate_expressions(2, chart_id)
        actual = tuple(
            tuple(evaluate_expression(expression, uv) for expression in expressions)
            for uv in (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
        )
        assert actual == tuple(tuple(point) for point in expected)
        facet = int(query.polygon_source_facet_rows[row])
        assert query.chart_source_key(
            2, chart_id
        ) == query.authority.source_entity_key_for_scope(
            2, int(query.authority.complex.facet_entity_ids[facet])
        )


def test_mapped_split_atlas_retains_nonbinary_vertices_and_rejects_invalid_cover() -> (
    None
):
    source = _mapped_rectangular_split_source()
    query = source.boundary_query()
    assert len(source.complex.polygon_facets) == 12
    assert query.boundary_chart_cover() is not None
    assert any(
        denominator == 3
        for row in query.polygon_root_reference_rational_tokens
        for point in row
        for _, denominator in point
    )
    refs = tuple(
        tuple(tuple(Fraction(*value) for value in point) for point in row)
        for row in query.polygon_root_reference_rational_tokens
    )
    segment_refs = tuple(
        tuple(tuple(Fraction(*value) for value in point) for point in row)
        for row in query.segment_root_reference_rational_tokens
    )
    faces = query.polygon_root_face_global_ids
    cells = query.polygon_root_cell_global_ids
    locals_ = query.polygon_root_local_face_indices
    signs = query.polygon_orientation_signs

    def validate(
        face_rows: np.ndarray,
        cell_rows: np.ndarray,
        local_rows: np.ndarray,
        charts: tuple[tuple[tuple[Fraction, ...], ...], ...],
        orientations: np.ndarray,
    ) -> None:
        MappedReferenceBoundarySource._validate_root_relation(
            query.domain,
            face_rows,
            cell_rows,
            local_rows,
            charts,
            orientations,
            query.segment_root_dimensions,
            query.segment_root_entity_global_ids,
            query.segment_root_cell_global_ids,
            query.segment_root_local_entity_indices,
            segment_refs,
            query.segment_orientations,
            False,
        )

    wrong_signs = np.asarray(signs).copy()
    wrong_signs[0] *= -1
    with pytest.raises(ValueError):
        validate(faces, cells, locals_, refs, wrong_signs)
    wrong_cells = np.asarray(cells).copy()
    wrong_cells[0] = next(
        int(cell) for cell in query.domain.root_cell_ids if int(cell) != int(cells[0])
    )
    with pytest.raises(ValueError):
        validate(faces, wrong_cells, locals_, refs, signs)
    with pytest.raises(ValueError):
        validate(faces[:-1], cells[:-1], locals_[:-1], refs[:-1], signs[:-1])
    with pytest.raises(ValueError):
        validate(
            np.concatenate((faces, faces[:1])),
            np.concatenate((cells, cells[:1])),
            np.concatenate((locals_, locals_[:1])),
            (*refs, refs[0]),
            np.concatenate((signs, signs[:1])),
        )

    complex_ = source.complex
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    polygons = tuple(
        np.asarray(
            complex_.polygon_vertices[offsets[row] : offsets[row + 1]], dtype=np.int64
        )
        for row in range(len(offsets) - 1)
    )
    wrong_regions = np.asarray(complex_.facet_regions).copy()
    wrong_regions[0, 1] = 1
    wrong_complex = PiecewiseLinearComplex(
        complex_.vertices,
        polygons,
        complex_.polygon_facets,
        wrong_regions,
        ("solid", "wrong-material"),
        segments=complex_.segments,
        vertex_ids=complex_.vertex_ids,
        vertex_occurrence_paths=complex_.vertex_occurrence_paths,
        facet_ids=complex_.facet_ids,
        facet_occurrence_paths=complex_.facet_occurrence_paths,
        segment_ids=complex_.segment_ids,
        segment_occurrence_paths=complex_.segment_occurrence_paths,
        region_occurrence_paths=((), ()),
    )
    wrong_source = NativePlcSource(
        wrong_complex, source.source_id, source.source_revision
    )
    with pytest.raises(ValueError):
        NativeMappedHexSource(
            wrong_source,
            source.domain,
            plc_vertex_root_global_ids=source.root_relation.vertex_root_global_ids,
        )


def test_mapped_domain_codec_rebuilds_coordinate_elements_and_refuses_tampering() -> None:
    from copy import copy

    from phydrax._model._structure import model_archive_projection
    from phydrax.geometry._mesh_certificates import _same_tree_content
    from phydrax.meshing._archive_layer_columns import LayerRecordArchiveRecipe
    from phydrax.meshing._model_codec import restore_model_archive_value

    _, original = _mapped_cube_source()
    projected = model_archive_projection(original)
    restored = restore_model_archive_value(projected)
    assert restored.domain_id == original.domain_id
    assert _same_tree_content(restored.reference_mesh, original.reference_mesh)
    assert _same_tree_content(restored.source_geometry, original.source_geometry)
    elements, routes, coordinates = restored.source_geometry.resolve(
        restored.reference_mesh
    )
    assert elements[0].cell_kind == "hexahedron"
    assert len(routes) == 1
    np.testing.assert_array_equal(coordinates, original.source_geometry.coordinates)

    stored_geometry = projected.source_geometry
    assert type(stored_geometry) is LayerRecordArchiveRecipe
    fields = dict(stored_geometry.fields)
    changed = np.asarray(fields["coordinates"], dtype=np.float64).copy()
    changed[0, 0] += 0.125
    fields["coordinates"] = changed
    changed_geometry = stored_geometry._replace(fields=fields)
    changed_domain = copy(projected)
    object.__setattr__(changed_domain, "source_geometry", changed_geometry)
    with pytest.raises(ValueError):
        restore_model_archive_value(changed_domain)

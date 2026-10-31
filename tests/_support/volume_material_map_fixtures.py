from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    PerCellRestrictedCellGeometryElement,
    ScalarCoordinateElement,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._nested_reference_types import (
    ExactReferenceChartExpression,
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry._domain_certification import DomainCoverageCertificate
from phydrax.geometry._domain_coverage import (
    certify_domain_coverage,
    PiecewiseLinearDomain,
)
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.geometry._volume_material_map import (
    CommonRootVolumeWitness,
    MaterialSourceMapCapability,
)


FractionPoint = tuple[Fraction, Fraction, Fraction]
ExactTetrahedron = tuple[FractionPoint, FractionPoint, FractionPoint, FractionPoint]
ROOT_REFERENCE_TETRAHEDRON: ExactTetrahedron = (
    (Fraction(0), Fraction(0), Fraction(0)),
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)
CHILD_REFERENCE_TETRAHEDRA: tuple[ExactTetrahedron, ExactTetrahedron] = (
    (
        ROOT_REFERENCE_TETRAHEDRON[0],
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        ROOT_REFERENCE_TETRAHEDRON[2],
        ROOT_REFERENCE_TETRAHEDRON[3],
    ),
    (
        (Fraction(1, 2), Fraction(0), Fraction(0)),
        ROOT_REFERENCE_TETRAHEDRON[1],
        ROOT_REFERENCE_TETRAHEDRON[2],
        ROOT_REFERENCE_TETRAHEDRON[3],
    ),
)


def make_curved_root() -> tuple[CellMesh, CellGeometrySpec, np.ndarray]:
    element = lagrange_element("tetrahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = np.column_stack(
        (nodes[:, 0], nodes[:, 1], nodes[:, 2] + 0.2 * nodes[:, 0] * nodes[:, 1])
    )
    corners = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        corners,
        (
            CellBlock(
                "root",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),)),
                global_ids=np.asarray((7,)),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"root": element},
        {"root": np.arange(element.local_dof_count, dtype=np.int32)[None, :]},
        coordinates,
    )
    return mesh, geometry, coordinates


def affine_reference_chart(
    vertices: ExactTetrahedron,
) -> tuple[dict[tuple[int, ...], Fraction], ...]:
    origin = vertices[0]
    result = []
    for axis in range(3):
        terms = {(0, 0, 0): origin[axis]}
        for column in range(3):
            coefficient = vertices[column + 1][axis] - origin[axis]
            if coefficient:
                exponent = tuple(int(index == column) for index in range(3))
                terms[exponent] = coefficient
        result.append(terms)
    return tuple(result)


def reference_map_witness(
    target_ids: tuple[int, ...],
    root_ids: tuple[int, ...],
    cells: tuple[ExactTetrahedron, ...],
    /,
    *,
    chart_arguments: Sequence[Sequence[ExactReferenceChartExpression]] | None = None,
) -> ReferenceCellMapWitnesses:
    return ReferenceCellMapWitnesses(
        np.asarray(target_ids, dtype=np.int64),
        np.asarray(root_ids, dtype=np.int64),
        "tetrahedron",
        "tetrahedron",
        np.asarray(
            [[[float(value) for value in point] for point in cell] for cell in cells]
        ),
        cells,
        source_reference_chart_arguments=(
            None
            if chart_arguments is None
            else freeze_reference_chart_arguments(chart_arguments)
        ),
    )


def with_root_restriction(
    root_mesh: CellMesh,
    root_geometry: CellGeometrySpec,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    reference_map: ReferenceCellMapWitnesses,
    /,
) -> CellGeometrySpec:
    """Retain the exact child→root chart ancestry of one restricted child mesh."""
    (block,) = mesh.blocks
    (root_block,) = root_mesh.blocks
    parents = np.asarray(reference_map.source_cell_ids, dtype=np.int64)
    root_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(root_block.global_ids))
    }
    root_vertices = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(root_block.vertices, dtype=np.int64)
    ]
    restriction = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry),
        root_mesh.topology_id,
        {block.name: parents},
        {
            block.name: np.stack(
                [root_vertices[root_rows[int(parent)]] for parent in parents]
            )
        },
        block_source_blocks={block.name: root_block.name},
        block_parent_reference_maps={block.name: reference_map},
    )
    return CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        geometry.coordinates,
        restriction_source=restriction,
    )


def make_root_restriction(
    root_mesh: CellMesh,
    root_geometry: CellGeometrySpec,
    /,
    *,
    cells: tuple[ExactTetrahedron, ...] = CHILD_REFERENCE_TETRAHEDRA,
    cell_ids: tuple[int, ...] = (10, 11),
    block_name: str = "refined",
    duplicate_vertices: bool = False,
) -> tuple[CellMesh, CellGeometrySpec, ReferenceCellMapWitnesses]:
    if duplicate_vertices:
        reference_vertices = [point for cell in cells for point in cell]
        connectivity = np.arange(4 * len(cells), dtype=np.int32).reshape((len(cells), 4))
    else:
        reference_vertices = []
        vertex_rows: dict[FractionPoint, int] = {}
        connectivity_rows = []
        for cell in cells:
            row = []
            for point in cell:
                if point not in vertex_rows:
                    vertex_rows[point] = len(reference_vertices)
                    reference_vertices.append(point)
                row.append(vertex_rows[point])
            connectivity_rows.append(row)
        connectivity = np.asarray(connectivity_rows, dtype=np.int32)
    points = np.asarray(reference_vertices, dtype=np.float64)
    root_ids = np.full(points.shape[0], 7, dtype=np.int64)
    mesh_points = root_geometry.evaluate(root_mesh, root_ids, points[:, None, :])[:, 0, :]
    mesh = CellMesh(
        mesh_points,
        (
            CellBlock(
                block_name, "tetrahedron", connectivity, global_ids=np.asarray(cell_ids)
            ),
        ),
    )

    root_element = root_geometry.elements[0]
    if not isinstance(root_element, ScalarCoordinateElement):
        raise TypeError("Root restrictions require an owning scalar coordinate element.")
    element = PerCellRestrictedCellGeometryElement(
        root_element,
        "tetrahedron",
        cell_ids,
        tuple(7 for _ in cells),
        tuple(affine_reference_chart(cell) for cell in cells),
    )
    route = np.asarray(root_geometry.geometry_dofs[0], dtype=np.int32)
    geometry = CellGeometrySpec(
        {block_name: element},
        {block_name: np.tile(route, (len(cells), 1))},
        root_geometry.coordinates,
    )
    witness = reference_map_witness(
        tuple(range(len(cells))), tuple(7 for _ in cells), cells
    )
    return mesh, geometry, witness


def make_material_coverage_pair(
    root_mesh: CellMesh,
    root_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    region_id: str,
    occurrence_path: tuple[str, ...],
) -> tuple[MappedReferenceDomain, DomainCoverageCertificate, DomainCoverageCertificate]:
    vertices = np.asarray(root_mesh.coordinates, dtype=np.float64)
    topology = reference_cell_topology("tetrahedron")
    center = np.mean(vertices, axis=0)
    facets = []
    for face in topology.entities[2]:
        oriented = [int(value) for value in face]
        points = vertices[np.asarray(oriented, dtype=np.int64)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        if float(np.dot(normal, np.mean(points, axis=0) - center)) < 0.0:
            oriented[1], oriented[2] = oriented[2], oriented[1]
        facets.append(tuple(oriented))
    source_id = "native-cad-source"
    source_revision = "revision-4"
    reference_domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (len(facets), 1)),
        (region_id,),
        source_id=source_id,
    )
    root_vertex_ids = tuple(
        int(value) for value in np.asarray(root_mesh.vertex_global_ids)
    )
    root_cell_count = sum(block.cell_count for block in root_mesh.blocks)
    domain = MappedReferenceDomain(
        reference_domain,
        root_mesh,
        root_geometry,
        np.zeros((root_cell_count,), dtype=np.int64),
        source_id=source_id,
        source_revision=source_revision,
        coordinate_contract=SpatialCoordinateContract.si(),
        root_vertex_occurrence_paths=tuple(occurrence_path for _ in root_vertex_ids),
        root_cell_occurrence_paths=tuple(occurrence_path for _ in range(root_cell_count)),
        reference_domain_vertex_root_ids=np.asarray(root_vertex_ids, dtype=np.int64),
    )

    def certify_endpoint(
        mesh: CellMesh, geometry: CellGeometrySpec, /
    ) -> DomainCoverageCertificate:
        cell_count = sum(block.cell_count for block in mesh.blocks)
        regions = np.zeros((cell_count,), dtype=np.int64)
        validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        embedding = certify_global_embedding(mesh, geometry, validity)
        return certify_domain_coverage(
            mesh,
            geometry,
            domain,
            regions,
            embedding=embedding,
        )

    return (
        domain,
        certify_endpoint(root_mesh, root_geometry),
        certify_endpoint(target_mesh, target_geometry),
    )


def make_material_source(
    domain: MappedReferenceDomain,
    /,
    *,
    entity_id: str = "solid-cell",
    region_id: str = "aluminum",
    occurrence_path: tuple[str, ...] = ("assembly-1", "occurrence-9"),
) -> MaterialSourceMapCapability:
    return MaterialSourceMapCapability(
        domain.source_id,
        domain.source_revision,
        domain.domain_id,
        (entity_id,),
        (region_id,),
        occurrence_paths=(occurrence_path,),
    )


def make_curved_material_refinement(
    *,
    gap: bool = False,
    duplicate: bool = False,
    mismatched_source: bool = False,
    exact_chart_arguments: bool = False,
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    CellGeometrySpec,
    CellMesh,
    CellGeometrySpec,
    ReferenceCellMapWitnesses,
    CommonRootVolumeWitness | None,
]:
    root_mesh, root_geometry, controls = make_curved_root()
    source_geometry = root_geometry
    if mismatched_source:
        changed = np.asarray(controls).copy()
        changed[4, 2] += 0.125
        source_geometry = CellGeometrySpec(
            {"root": root_geometry.elements[0]},
            {"root": np.asarray(root_geometry.geometry_dofs[0])},
            changed,
        )
    cells = (
        (CHILD_REFERENCE_TETRAHEDRA[0],)
        if gap
        else (
            (ROOT_REFERENCE_TETRAHEDRON, ROOT_REFERENCE_TETRAHEDRON)
            if duplicate
            else CHILD_REFERENCE_TETRAHEDRA
        )
    )
    cell_ids = (10,) if gap else (10, 11)
    target_mesh, target_geometry, target_maps = make_root_restriction(
        root_mesh,
        root_geometry,
        cells=cells,
        cell_ids=cell_ids,
        duplicate_vertices=duplicate,
    )
    exact_maps = reference_map_witness(
        tuple(range(len(cells))),
        tuple(7 for _ in cells),
        cells,
        chart_arguments=tuple(affine_reference_chart(cell) for cell in cells),
    )
    # Coverage of a child mesh requires its exact chart ancestry to the root.
    target_geometry = with_root_restriction(
        root_mesh, root_geometry, target_mesh, target_geometry, exact_maps
    )
    if exact_chart_arguments:
        target_maps = exact_maps
    source_maps = reference_map_witness((0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,))
    witness = None
    if not mismatched_source:
        witness = CommonRootVolumeWitness(
            root_mesh,
            root_geometry,
            root_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            {"root": source_maps},
            {target_mesh.blocks[0].name: target_maps},
        )
    return (
        root_mesh,
        root_geometry,
        source_geometry,
        target_mesh,
        target_geometry,
        target_maps,
        witness,
    )


def make_curved_coarsening_witness(
    root_mesh: CellMesh,
    root_geometry: CellGeometrySpec,
    refined_mesh: CellMesh,
    refined_geometry: CellGeometrySpec,
    refined_reference_map: ReferenceCellMapWitnesses,
    /,
) -> CommonRootVolumeWitness:
    return CommonRootVolumeWitness(
        root_mesh,
        root_geometry,
        refined_mesh,
        refined_geometry,
        root_mesh,
        root_geometry,
        {refined_mesh.blocks[0].name: refined_reference_map},
        {"root": reference_map_witness((0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,))},
    )

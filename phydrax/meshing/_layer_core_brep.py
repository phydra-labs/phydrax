#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-face layer controls on a native BRep volume.

The accepted parametric boundary binds every surface cell to one source-face chart.
A source-face layer control resolves through those charts to wall cells of the
inward-oriented boundary. The layers grow from that wall; unselected cells are the
fixed adjacent surfaces along which rim columns slide. The fixed core boundary is
the layer cap plus every unselected surface cell, whose rim vertices move to the
top of their columns so that the core boundary conforms to the cap.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import phydrax.meshing._association as association_owner
import phydrax.meshing._boundary_layer as boundary_layer_owner
import phydrax.meshing._layer_core_controls as controls_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing.providers._native_sources as sources_owner

from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..geometry._meshing_domain import MeshingDomain
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..typing import checked
from ._contracts import VolumeMeshingSpec
from ._result import CellMeshingResult


def _target_association(
    associations: tuple[association_owner.GeometryAssociation, ...],
    entity_set_id: str,
    /,
) -> association_owner.GeometryAssociation:
    matches = tuple(
        value for value in associations if value.target_entity_set_id == entity_set_id
    )
    if len(matches) != 1:
        raise ValueError(
            "The accepted boundary must publish one exact association per wall stratum."
        )
    return matches[0]


def _rebind_association(
    parent: association_owner.GeometryAssociation,
    target: CellMesh,
    dimension: int,
    orientation_sign: int,
    /,
) -> association_owner.GeometryAssociation:
    entity_set = target.entity_set(dimension)
    if parent.target_entity_set_id == entity_set.entity_set_id:
        return parent
    targets = np.asarray(entity_set.entity_ids, dtype=np.int64)
    parent_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(parent.target_global_ids, dtype=np.int64).tolist()
        )
    }
    if set(parent_rows) != set(targets.tolist()):
        raise ValueError("The inward wall must keep every associated boundary entity.")
    rows = np.asarray(
        [parent_rows[int(identifier)] for identifier in targets.tolist()],
        dtype=np.int64,
    )
    return association_owner.inherit_association_rows(
        parent,
        entity_set,
        targets,
        rows,
        residuals=np.asarray(parent.residuals, dtype=np.float64)[rows],
        orientations=orientation_sign
        * np.asarray(parent.orientations, dtype=np.int8)[rows],
    )


@checked
def inward_wall_charts(
    boundary: CellMeshingResult, /
) -> tuple[SurfaceSourceCharts, association_owner.GeometryAssociation]:
    """Reverse the accepted outward boundary and carry its exact source charts."""
    charts = boundary.surface_source
    if charts is None:
        raise ValueError("The accepted CAD boundary must retain its exact source charts.")
    charts.require_root(boundary.mesh, boundary.geometry)
    mesh = boundary.mesh
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Source-face layer walls require a triangular CAD boundary.")
    wall = CellMesh(
        mesh.coordinates,
        tuple(
            CellBlock(
                block.name,
                "triangle",
                np.asarray(block.vertices, dtype=np.int64)[:, (0, 2, 1)],
                global_ids=np.asarray(block.global_ids, dtype=np.int64),
            )
            for block in mesh.blocks
        ),
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        numeric_version=mesh.numeric_version,
    )
    cells = _rebind_association(
        _target_association(charts.associations, mesh.entity_set(2).entity_set_id),
        wall,
        2,
        -1,
    )
    vertices = _rebind_association(
        _target_association(charts.associations, mesh.entity_set(0).entity_set_id),
        wall,
        0,
        1,
    )
    exact = tuple((chart[0], chart[2], chart[1]) for chart in charts.root_parameter_exact)
    inward = SurfaceSourceCharts(
        charts.domain,
        wall,
        CellGeometrySpec.affine(wall),
        charts.coordinate_contract,
        np.asarray(charts.root_parameters, dtype=np.float64)[:, (0, 2, 1)],
        np.asarray(charts.root_patches, dtype=np.int64),
        (cells, vertices),
        charts.boundary_source,
        source_root_cell_ids=np.asarray(charts.source_root_cell_ids, dtype=np.int64),
        source_piece_ids=charts.source_piece_ids,
        root_parameter_exact=exact,
        boundary_occurrences=(
            None
            if charts.boundary_occurrences.size == 0
            else np.asarray(charts.boundary_occurrences, dtype=np.int64)[:, (0, 2, 1)]
        ),
    )
    return inward, vertices


def _triangle_rows(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle vertex rows and cell IDs in block order."""
    if any(block.cell_kind != "triangle" for block in mesh.blocks):
        raise ValueError("Layer cap and wall surfaces must be triangular.")
    rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    return rows, ids


class FixedCore(NamedTuple):
    """The fixed core PLC and its exact ancestry in the charted wall.

    ``vertex_layer_ids`` maps PLC vertices to cap layer vertices (or ``-1``),
    ``vertex_wall_rows`` maps the remaining PLC vertices to unmoved wall rows (or
    ``-1``), and ``polygon_wall_cells`` maps each fixed surface polygon to its wall
    cell (``-1`` for cap polygons, which come first).
    """

    complex_: plc_source_owner.PiecewiseLinearComplex
    vertex_layer_ids: np.ndarray
    vertex_wall_rows: np.ndarray
    polygon_wall_cells: np.ndarray
    facet_keys: tuple[tuple[str, int], ...]
    cap_count: int


def grown_wall_cells(layers: boundary_layer_owner.BoundaryLayerMesh, /) -> np.ndarray:
    """Return the wall cells that grew layers, from the accepted front rows."""
    kinds = np.asarray(layers.front_row_source_entity_kinds)
    return np.unique(
        np.asarray(layers.front_row_source_global_ids, dtype=np.int64)[kinds == "face"]
    )


@checked
def fixed_core(
    domain: MeshingDomain,
    layers: boundary_layer_owner.BoundaryLayerMesh,
    charts: SurfaceSourceCharts,
    /,
) -> FixedCore:
    """Bind the cap and the unselected wall cells as one fixed core PLC.

    Every rim vertex of an unselected cell moves to the cap vertex that tops its
    single column; the cell keeps its outward orientation. One PLC facet groups
    the cap over each source face and one the fixed remainder of each source face;
    every polygon edge is an explicit segment so that recovery keeps each polygon.
    """
    wall = charts.mesh
    if layers.source_wall is not wall and layers.source_wall.mesh_id != wall.mesh_id:
        raise ValueError("The accepted layers must grow from the charted wall.")
    cap = layers.cap
    if cap is None:
        raise ValueError("The accepted layers left no free cap for the fixed core.")
    cap_rows, _ = _triangle_rows(cap)
    cap_layer_rows = np.asarray(layers.cap_vertices, dtype=np.int64)
    layer_vertex_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    layer_sources = np.asarray(layers.vertex_source_vertex_ids, dtype=np.int64)
    wall_rows, wall_cell_ids = _triangle_rows(wall)
    wall_vertex_ids = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    grown = np.isin(wall_cell_ids, grown_wall_cells(layers))
    fixed_rows = wall_rows[~grown]
    fixed_cell_ids = wall_cell_ids[~grown]
    rim = np.intersect1d(
        np.unique(wall_rows[grown].reshape(-1)), np.unique(fixed_rows.reshape(-1))
    )
    cap_of_wall_vertex: dict[int, list[int]] = {}
    for cap_row, layer_row in enumerate(cap_layer_rows.tolist()):
        source_vertex = int(layer_sources[layer_row])
        if source_vertex >= 0:
            cap_of_wall_vertex.setdefault(source_vertex, []).append(cap_row)
    plc_row_of_wall: dict[int, int] = {}
    for wall_row in rim.tolist():
        rows = cap_of_wall_vertex.get(int(wall_vertex_ids[wall_row]), [])
        if len(rows) != 1:
            raise ValueError(
                "Each rim vertex must end in one cap vertex on its adjacent surface."
            )
        plc_row_of_wall[int(wall_row)] = rows[0]
    unmoved = np.setdiff1d(np.unique(fixed_rows.reshape(-1)), rim)
    cap_count = cap_rows.shape[0]
    cap_vertex_count = cap_layer_rows.size
    plc_row_of_wall.update(
        {
            int(wall_row): cap_vertex_count + offset
            for offset, wall_row in enumerate(unmoved.tolist())
        }
    )
    # The wall is oriented inward; reversing fixed cells restores the outward
    # source orientation, so the core lies on their negative side. The cap is
    # oriented away from the layers, into the core.
    fixed_polygons = np.asarray(
        [[plc_row_of_wall[int(value)] for value in row[::-1]] for row in fixed_rows],
        dtype=np.int64,
    ).reshape((-1, 3))
    polygons = np.concatenate((cap_rows, fixed_polygons))
    points = np.concatenate(
        (
            np.asarray(layers.mesh.coordinates, dtype=np.float64)[cap_layer_rows],
            np.asarray(wall.coordinates, dtype=np.float64)[unmoved],
        )
    )
    patch_of_cell = dict(
        zip(
            charts.cell_global_ids,
            np.asarray(charts.root_patches, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    wall_cell_of_vertices = {
        tuple(sorted(int(value) for value in wall_vertex_ids[row])): int(cell)
        for row, cell in zip(wall_rows, wall_cell_ids.tolist(), strict=True)
    }
    try:
        cap_patches = [
            patch_of_cell[
                wall_cell_of_vertices[
                    tuple(
                        sorted(int(value) for value in layer_sources[cap_layer_rows[row]])
                    )
                ]
            ]
            for row in cap_rows
        ]
    except KeyError as error:
        raise ValueError(
            "Every cap facet must cap exactly one charted wall cell."
        ) from error
    fixed_patches = [patch_of_cell[int(cell)] for cell in fixed_cell_ids.tolist()]
    facet_keys = tuple(
        sorted({("cap", patch) for patch in cap_patches})
        + sorted({("face", patch) for patch in fixed_patches})
    )
    facet_of_key = {key: row for row, key in enumerate(facet_keys)}
    source_id = domain.source_id
    vertex_ids = (
        *(
            f"{source_id}:layer-vertex:{int(layer_vertex_ids[row])}"
            for row in cap_layer_rows.tolist()
        ),
        *(
            f"{source_id}:surface-vertex:{int(wall_vertex_ids[row])}"
            for row in unmoved.tolist()
        ),
    )
    polygon_facets = np.asarray(
        [facet_of_key[("cap", patch)] for patch in cap_patches]
        + [facet_of_key[("face", patch)] for patch in fixed_patches],
        dtype=np.int64,
    )
    facet_paths = tuple(
        tuple(domain.source_occurrences[2][patch]) for _, patch in facet_keys
    )
    edge_paths: dict[tuple[int, int], set[tuple[str, ...]]] = {}
    for polygon, facet in zip(polygons.tolist(), polygon_facets.tolist(), strict=True):
        for corner in range(3):
            edge = tuple(sorted((polygon[corner], polygon[(corner + 1) % 3])))
            edge_paths.setdefault((edge[0], edge[1]), set()).add(facet_paths[facet])
    if any(len(paths) != 1 for paths in edge_paths.values()):
        raise ValueError("A fixed core edge joins different source occurrences.")
    edges = np.asarray(sorted(edge_paths), dtype=np.int64).reshape((-1, 2))
    complex_ = plc_source_owner.PiecewiseLinearComplex(
        points,
        tuple(tuple(int(value) for value in row) for row in polygons),
        polygon_facets,
        np.asarray(
            [(0, -1) if side == "cap" else (-1, 0) for side, _ in facet_keys],
            dtype=np.int64,
        ).reshape((-1, 2)),
        (domain.regions[0].name,),
        segments=edges,
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=tuple(
            f"{source_id}:layer-{side}:{domain.entity_id(2, patch)}"
            for side, patch in facet_keys
        ),
        facet_occurrence_paths=facet_paths,
        segment_ids=tuple(
            f"{vertex_ids[first]}~{vertex_ids[second]}" for first, second in edges
        ),
        segment_occurrence_paths=tuple(
            next(iter(edge_paths[(int(first), int(second))])) for first, second in edges
        ),
        region_occurrence_paths=(domain.region_source_occurrences[0],),
        boundary="fixed",
    )
    return FixedCore(
        complex_,
        np.concatenate((cap_layer_rows, np.full(unmoved.size, -1, dtype=np.int64))),
        np.concatenate((np.full(cap_vertex_count, -1, dtype=np.int64), unmoved)),
        np.concatenate((np.full(cap_count, -1, dtype=np.int64), fixed_cell_ids)),
        facet_keys,
        cap_count,
    )


@checked
def brep_layer_core_source(
    domain: MeshingDomain,
    boundary: CellMeshingResult,
    specification: VolumeMeshingSpec,
    /,
) -> sources_owner.NativeLayerCoreSource:
    """Grow the source-face layer control and bind the fixed conforming core."""
    if len(specification.layer_controls) != 1:
        raise ValueError("A native BRep volume grows exactly one layer control.")
    if len(domain.regions) != 1:
        raise ValueError("Source-face layer controls require a single-solid BRep.")
    charts, vertex_association = inward_wall_charts(boundary)
    if charts.domain.domain_id != domain.domain_id:
        raise ValueError(
            "The accepted boundary charts must bind the volume source domain."
        )
    control = controls_owner.wall_layer_control(specification.layer_controls[0], charts)
    layers = boundary_layer_owner.prepare_boundary_layers(
        charts.mesh, control, wall_association=vertex_association
    )
    core = fixed_core(domain, layers, charts)
    return sources_owner.NativeLayerCoreSource(
        layers,
        core.complex_,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=core.vertex_layer_ids,
        cap_polygon_ids=np.arange(core.cap_count, dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
        region_entity_ids=np.asarray(domain.region_source_indices[:1], dtype=np.int64),
        region_occurrence_paths=(domain.region_source_occurrences[0],),
        region_ids=(domain.regions[0].name,),
        core_region_map=np.zeros((1,), dtype=np.int64),
        source_boundary_scope=specification.boundary_scope,
        source_domain=domain,
        fidelity_source=charts.boundary_source,
        source_surface_charts=charts,
        core_facet_source_ids=tuple(
            None if side == "cap" else domain.entity_id(2, patch)
            for side, patch in core.facet_keys
        ),
        core_facet_source_occurrence_paths=tuple(
            None if side == "cap" else tuple(domain.source_occurrences[2][patch])
            for side, patch in core.facet_keys
        ),
    )


__all__ = [
    "FixedCore",
    "brep_layer_core_source",
    "fixed_core",
    "grown_wall_cells",
    "inward_wall_charts",
]

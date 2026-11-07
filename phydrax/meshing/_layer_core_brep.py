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
from .providers._native_schedules import NativeSurfaceSchedule


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
    if parent.target_entity_set_id == entity_set.entity_set_id and orientation_sign == 1:
        return parent
    if parent.source_boundary_witness_id is not None:
        raise ValueError("Rebound face pieces require a source-bound target witness.")
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
    return association_owner.GeometryAssociation(
        parent.association_kind,
        parent.source_id,
        parent.source_revision,
        entity_set.entity_set_id,
        targets,
        tuple(parent.source_entity_ids[int(row)] for row in rows),
        np.asarray(parent.residuals)[rows],
        resolved=np.asarray(parent.resolved)[rows],
        ambiguous=np.asarray(parent.ambiguous)[rows],
        exact=parent.exact,
        source_dimensions=np.asarray(parent.source_dimensions)[rows],
        source_indices=np.asarray(parent.source_indices)[rows],
        source_roles=tuple(parent.source_roles[int(row)] for row in rows),
        source_occurrence_paths=tuple(
            parent.source_occurrence_paths[int(row)] for row in rows
        ),
        source_region_sides=tuple(parent.source_region_sides[int(row)] for row in rows),
        source_parameter_intervals=tuple(
            parent.source_parameter_intervals[int(row)] for row in rows
        ),
        target_parameter_intervals=tuple(
            None
            if (interval := parent.target_parameter_intervals[int(row)]) is None
            else (
                interval
                if dimension != 1 or orientation_sign == 1
                else (1 - interval[1], 1 - interval[0])
            )
            for row in rows
        ),
        source_parameter_points=tuple(
            parent.source_parameter_points[int(row)] for row in rows
        ),
        parameters=np.asarray(parent.parameters)[rows],
        orientations=orientation_sign
        * np.asarray(parent.orientations, dtype=np.int8)[rows],
        parent_association_id=parent.association_id,
        provenance=association_owner.GeometryAssociationProvenance.LINEAGE,
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
    edge_tables = tuple(
        value
        for value in charts.associations
        if value.target_entity_set_id == mesh.entity_set(1).entity_set_id
    )
    edge_associations = []
    for parent in edge_tables:
        target_ids = np.asarray(parent.target_global_ids, dtype=np.int64)
        if not np.all(np.isin(target_ids, wall.entity_set(1).entity_ids)):
            raise ValueError("The inward wall must retain every associated edge.")
        edge_associations.append(
            association_owner.GeometryAssociation(
                parent.association_kind,
                parent.source_id,
                parent.source_revision,
                wall.entity_set(1).entity_set_id,
                target_ids,
                parent.source_entity_ids,
                np.asarray(parent.residuals),
                resolved=np.asarray(parent.resolved),
                ambiguous=np.asarray(parent.ambiguous),
                exact=parent.exact,
                source_dimensions=np.asarray(parent.source_dimensions),
                source_indices=np.asarray(parent.source_indices),
                source_roles=parent.source_roles,
                source_occurrence_paths=parent.source_occurrence_paths,
                source_region_sides=parent.source_region_sides,
                source_parameter_intervals=parent.source_parameter_intervals,
                target_parameter_intervals=parent.target_parameter_intervals,
                source_parameter_points=parent.source_parameter_points,
                parameters=np.asarray(parent.parameters),
                orientations=np.asarray(parent.orientations),
                parent_association_id=parent.association_id,
                provenance=association_owner.GeometryAssociationProvenance.LINEAGE,
            )
        )
    exact = tuple((chart[0], chart[2], chart[1]) for chart in charts.root_parameter_exact)
    inward = SurfaceSourceCharts(
        charts.domain,
        wall,
        CellGeometrySpec.affine(wall),
        charts.coordinate_contract,
        np.asarray(charts.root_parameters, dtype=np.float64)[:, (0, 2, 1)],
        np.asarray(charts.root_patches, dtype=np.int64),
        (cells, vertices, *edge_associations),
        charts.boundary_source,
        source_root_cell_ids=np.asarray(charts.source_root_cell_ids, dtype=np.int64),
        source_piece_ids=charts.source_piece_ids,
        root_parameter_exact=exact,
        trim_roots=charts.trim_roots,
        boundary_occurrences=(
            None
            if charts.boundary_occurrences.size == 0
            else np.asarray(charts.boundary_occurrences, dtype=np.int64)[:, (0, 2, 1)]
        ),
    )
    return inward, vertices


@checked
def restrict_wall_charts(
    charts: SurfaceSourceCharts, wall: CellMesh, /
) -> SurfaceSourceCharts:
    """Restrict accepted surface charts to an exactly incident oriented wall."""
    from ..geometry._domain_boundary import MeshingDomainBoundarySource
    from ._association_routes import ordered_vertices, target_dimension

    charts.require_root(charts.mesh, charts.geometry)
    if any(block.cell_kind != "triangle" for block in wall.blocks):
        raise ValueError("A restricted BRep wall must contain triangular cells.")
    parent_vertices = np.asarray(charts.mesh.vertex_global_ids, dtype=np.int64)
    wall_vertices = np.asarray(wall.vertex_global_ids, dtype=np.int64)
    parent_rows = {int(identifier): row for row, identifier in enumerate(parent_vertices)}
    parent_points = np.asarray(charts.mesh.coordinates, dtype=np.float64)
    wall_points = np.asarray(wall.coordinates, dtype=np.float64)
    for row, identifier in enumerate(wall_vertices):
        source_row = parent_rows.get(int(identifier))
        if source_row is None or not np.array_equal(
            wall_points[row].view(np.uint64), parent_points[source_row].view(np.uint64)
        ):
            raise ValueError(
                "Restricted wall vertices must retain exact source coordinates."
            )
    parent_cells = {
        int(identifier): tuple(int(parent_vertices[int(vertex)]) for vertex in row)
        for block in charts.mesh.blocks
        for identifier, row in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64).tolist(),
            strict=True,
        )
    }
    chart_rows = {
        int(identifier): row for row, identifier in enumerate(charts.cell_global_ids)
    }
    selected_ids = [
        int(identifier)
        for block in wall.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    ]
    permutations: list[tuple[int, ...]] = []
    for block in wall.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64).tolist(),
            strict=True,
        ):
            old = parent_cells.get(int(identifier))
            new = tuple(int(wall_vertices[int(vertex)]) for vertex in vertices)
            if old is None or len(old) != 3 or set(old) != set(new):
                raise ValueError(
                    "Restricted wall cells must retain exact parent vertex incidence."
                )
            permutations.append(tuple(old.index(vertex) for vertex in new))
    rows = np.asarray(
        [chart_rows[identifier] for identifier in selected_ids], dtype=np.int64
    )
    signs = {
        identifier: (
            -1
            if sum(
                permutation[i] > permutation[j] for i in range(3) for j in range(i + 1, 3)
            )
            % 2
            else 1
        )
        for identifier, permutation in zip(selected_ids, permutations, strict=True)
    }
    parameters = np.stack(
        [
            np.asarray(charts.root_parameters[row])[np.asarray(permutation)]
            for row, permutation in zip(rows, permutations, strict=True)
        ]
    )
    exact = tuple(
        tuple(charts.root_parameter_exact[int(row)][index] for index in permutation)
        for row, permutation in zip(rows, permutations, strict=True)
    )
    patches = np.asarray(charts.root_patches, dtype=np.int64)[rows]
    occurrences = (
        None
        if charts.boundary_occurrences.size == 0
        else np.stack(
            [
                np.asarray(charts.boundary_occurrences[row])[np.asarray(permutation)]
                for row, permutation in zip(rows, permutations, strict=True)
            ]
        )
    )
    parent_edges = ordered_vertices(charts.mesh, 1)
    wall_edges = ordered_vertices(wall, 1)
    if parent_edges is None or wall_edges is None:
        raise ValueError(
            "Restricted wall edge ancestry requires exact simplex incidence."
        )
    parent_edge_ids = np.asarray(charts.mesh.entity_set(1).entity_ids, dtype=np.int64)
    wall_edge_ids = np.asarray(wall.entity_set(1).entity_ids, dtype=np.int64)
    edge_parents: dict[tuple[int, ...], tuple[int, tuple[int, ...]]] = {}
    for edge_id, edge in zip(parent_edge_ids, parent_edges, strict=True):
        ordered = tuple(int(parent_vertices[int(row)]) for row in edge)
        key = tuple(sorted(ordered))
        if key in edge_parents:
            raise ValueError("Parent surface has repeated edge vertex incidence.")
        edge_parents[key] = (int(edge_id), ordered)
    edge_map: dict[int, tuple[int, int]] = {}
    for edge_id, edge in zip(wall_edge_ids, wall_edges, strict=True):
        ordered = tuple(int(wall_vertices[int(row)]) for row in edge)
        parent = edge_parents.get(tuple(sorted(ordered)))
        if parent is None:
            raise ValueError("A wall edge has no exact parent surface incidence.")
        edge_map[int(parent[0])] = (int(edge_id), 1 if ordered == parent[1] else -1)
    tables: list[association_owner.GeometryAssociation] = []
    for parent in charts.associations:
        dimension = target_dimension(charts.mesh, parent)
        parent_ids = np.asarray(parent.target_global_ids, dtype=np.int64)
        target_ids = set(
            np.asarray(wall.entity_set(dimension).entity_ids, dtype=np.int64).tolist()
        )
        chosen: list[tuple[int, int, int]] = []
        for row, identifier in enumerate(parent_ids.tolist()):
            if dimension == 1:
                match = edge_map.get(int(identifier))
                if match is not None:
                    chosen.append((row, match[0], match[1]))
            elif int(identifier) in target_ids:
                chosen.append(
                    (
                        row,
                        int(identifier),
                        signs[int(identifier)] if dimension == 2 else 1,
                    )
                )
        if not chosen:
            continue
        parent_rows = np.asarray([item[0] for item in chosen], dtype=np.int64)
        ids = np.asarray([item[1] for item in chosen], dtype=np.int64)
        if (
            parent.target_entity_set_id == wall.entity_set(dimension).entity_set_id
            and np.array_equal(ids, parent_ids)
            and all(item[2] == 1 for item in chosen)
        ):
            tables.append(parent)
            continue
        if parent.source_boundary_witness_id is not None:
            raise ValueError(
                "Restricted face pieces require a source-bound target witness."
            )
        orientation = np.asarray(parent.orientations, dtype=np.int8)[
            parent_rows
        ] * np.asarray([item[2] for item in chosen], dtype=np.int8)
        tables.append(
            association_owner.GeometryAssociation(
                parent.association_kind,
                parent.source_id,
                parent.source_revision,
                wall.entity_set(dimension).entity_set_id,
                ids,
                tuple(parent.source_entity_ids[int(row)] for row in parent_rows),
                np.asarray(parent.residuals)[parent_rows],
                resolved=np.asarray(parent.resolved)[parent_rows],
                ambiguous=np.asarray(parent.ambiguous)[parent_rows],
                exact=parent.exact,
                source_dimensions=np.asarray(parent.source_dimensions)[parent_rows],
                source_indices=np.asarray(parent.source_indices)[parent_rows],
                source_roles=tuple(parent.source_roles[int(row)] for row in parent_rows),
                source_occurrence_paths=tuple(
                    parent.source_occurrence_paths[int(row)] for row in parent_rows
                ),
                source_region_sides=tuple(
                    parent.source_region_sides[int(row)] for row in parent_rows
                ),
                source_parameter_intervals=tuple(
                    parent.source_parameter_intervals[int(row)] for row in parent_rows
                ),
                target_parameter_intervals=tuple(
                    None
                    if (interval := parent.target_parameter_intervals[int(row)]) is None
                    else (
                        interval
                        if dimension != 1 or chosen[index][2] == 1
                        else (1 - interval[1], 1 - interval[0])
                    )
                    for index, row in enumerate(parent_rows)
                ),
                source_parameter_points=tuple(
                    parent.source_parameter_points[int(row)] for row in parent_rows
                ),
                parameters=np.asarray(parent.parameters)[parent_rows],
                orientations=orientation,
                parent_association_id=parent.association_id,
                provenance=association_owner.GeometryAssociationProvenance.LINEAGE,
            )
        )
    return SurfaceSourceCharts(
        charts.domain,
        wall,
        CellGeometrySpec.affine(wall),
        charts.coordinate_contract,
        parameters,
        patches,
        tuple(tables),
        MeshingDomainBoundarySource(charts.domain, tuple(sorted(set(patches.tolist())))),
        source_root_cell_ids=np.asarray(charts.source_root_cell_ids, dtype=np.int64)[
            rows
        ],
        source_piece_ids=tuple(charts.source_piece_ids[int(row)] for row in rows),
        root_parameter_exact=exact,
        boundary_occurrences=occurrences,
        trim_roots=charts.trim_roots,
    )


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
    ``vertex_wall_rows`` maps other retained vertices to their source-wall rows
    (or ``-1``), and ``polygon_wall_cells`` names the unique source wall cell
    when one exists. Cap polygons come first and carry ``-1``; an arranged side
    polygon spanning multiple wall cells also carries ``-1`` and requires
    independent exact source-image overlap.
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


def _planar_fixed_core(
    domain: MeshingDomain,
    layers: boundary_layer_owner.BoundaryLayerMesh,
    charts: SurfaceSourceCharts,
    /,
) -> FixedCore | None:
    """Build a cap-preserving source-face arrangement when exact strata admit it."""
    from fractions import Fraction

    from ._layer_core_brep_plc import remaining_brep_core_plc_rows
    from ._layer_core_cad_footprints import layer_planar_cad_footprints
    from ._layer_core_edge_policy import _segment_relation

    footprints = layer_planar_cad_footprints(domain, charts, layers)
    if footprints is None:
        return None
    rows = remaining_brep_core_plc_rows(domain, charts, layers, footprints)
    if any(
        layer < 0 and wall < 0
        for layer, wall in zip(rows.vertex_layer_ids, rows.vertex_wall_rows, strict=True)
    ):
        raise ValueError(
            "A planar core arrangement vertex lacks retained layer or source-wall ancestry."
        )
    layer_ids = np.asarray(layers.mesh.vertex_global_ids, dtype=np.int64)
    wall_ids = np.asarray(charts.mesh.vertex_global_ids, dtype=np.int64)
    vertex_ids = tuple(
        f"{domain.source_id}:layer-vertex:{int(layer_ids[layer])}"
        if layer >= 0
        else f"{domain.source_id}:surface-vertex:{int(wall_ids[wall])}"
        for layer, wall in zip(rows.vertex_layer_ids, rows.vertex_wall_rows, strict=True)
    )
    facet_paths = tuple(
        tuple(domain.source_occurrences[2][patch]) for _, patch in rows.facet_keys
    )
    edge_paths: dict[tuple[int, int], set[tuple[str, ...]]] = {}
    for polygon, facet in zip(rows.polygons, rows.polygon_facets, strict=True):
        for index, first in enumerate(polygon):
            last = polygon[(index + 1) % 3]
            edge = (min(first, last), max(first, last))
            edge_paths.setdefault(edge, set()).add(facet_paths[facet])
    if any(len(paths) != 1 for paths in edge_paths.values()):
        raise ValueError("A planar core edge joins different source occurrences.")
    constrained: list[tuple[int, int]] = []
    for edge in sorted(edge_paths):
        intervals: list[tuple[Fraction, Fraction]] = []
        for curve in footprints.strata.segments:
            interval, crossing = _segment_relation(
                rows.points[edge[0]], rows.points[edge[1]], curve.first, curve.last
            )
            if crossing:
                raise ValueError("A planar core edge crosses an original CAD curve.")
            if interval is not None:
                intervals.append(interval)
        if not intervals:
            continue
        cursor = Fraction(0)
        for lower, upper in sorted(intervals):
            if lower > cursor:
                break
            cursor = max(cursor, upper)
        if cursor != 1:
            raise ValueError(
                "A planar core edge only partially covers a source CAD curve."
            )
        constrained.append(edge)
    complex_ = plc_source_owner.PiecewiseLinearComplex(
        np.asarray(
            [[float(value) for value in point] for point in rows.points], dtype=np.float64
        ),
        rows.polygons,
        np.asarray(rows.polygon_facets, dtype=np.int64),
        np.asarray(
            [(0, -1) if side == "cap" else (-1, 0) for side, _ in rows.facet_keys],
            dtype=np.int64,
        ),
        (domain.regions[0].name,),
        segments=np.asarray(constrained, dtype=np.int64).reshape((-1, 2)),
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=((),) * len(rows.points),
        facet_ids=tuple(
            f"{domain.source_id}:layer-{side}:{domain.entity_id(2, patch)}"
            for side, patch in rows.facet_keys
        ),
        facet_occurrence_paths=facet_paths,
        segment_ids=tuple(
            f"{vertex_ids[first]}~{vertex_ids[last]}" for first, last in constrained
        ),
        segment_occurrence_paths=tuple(
            next(iter(edge_paths[edge])) for edge in constrained
        ),
        region_occurrence_paths=(domain.region_source_occurrences[0],),
        boundary="fixed",
    )
    return FixedCore(
        complex_,
        np.asarray(rows.vertex_layer_ids, dtype=np.int64),
        np.asarray(rows.vertex_wall_rows, dtype=np.int64),
        np.asarray(rows.polygon_wall_cells, dtype=np.int64),
        rows.facet_keys,
        rows.cap_count,
    )


@checked
def fixed_core(
    domain: MeshingDomain,
    layers: boundary_layer_owner.BoundaryLayerMesh,
    charts: SurfaceSourceCharts,
    /,
) -> FixedCore:
    """Bind the unchanged cap to the remaining original source-face boundary.

    Exact planar source charts are arranged against actual layer footprints;
    otherwise the source wall's existing fixed-cell ancestry is retained.
    """
    wall = charts.mesh
    if layers.source_wall is not wall and layers.source_wall.mesh_id != wall.mesh_id:
        raise ValueError("The accepted layers must grow from the charted wall.")
    cap = layers.cap
    if cap is None:
        raise ValueError("The accepted layers left no free cap for the fixed core.")
    planar = _planar_fixed_core(domain, layers, charts)
    if planar is not None:
        return planar
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
    source_boundary_schedule: NativeSurfaceSchedule,
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
    policy = boundary_layer_owner.BoundaryLayerPolicy()
    layers = boundary_layer_owner.prepare_boundary_layers(
        charts.mesh, control, wall_association=vertex_association, policy=policy
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
        source_boundary_charts=boundary.surface_source,
        source_boundary_schedule=source_boundary_schedule,
        source_wall_charts=charts,
        noncap_refinement=(
            "conforming"
            if np.any(core.polygon_wall_cells[core.cap_count :] < 0)
            else "fixed"
        ),
        layer_control=control,
        layer_policy=policy,
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

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent authored-wall source proof and exact layer ancestry replay."""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np

from ..geometry._surface_source_support import (
    SurfaceSourceCharts,
    _cad_root_edges,
    validate_cad_curve_ancestry,
)
from ..geometry.brep._projection_contracts import BRepProjectionStatus
from ._association import (
    _brep_association,
    _Classes,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from .providers._native_sources import NativeLayerCoreSource


class _WallAuthority(NamedTuple):
    charts: SurfaceSourceCharts
    parent: GeometryAssociation
    curve_bindings: dict[int, tuple[_CadCurveUseBinding, ...]]
    edge_vertices: dict[int, tuple[int, int]]
    vertex_classes: dict[int, tuple[int, int, Fraction | None]]
    projection: NativeBRepProjection | None
    actual_chart_rows: tuple[
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...
    ] | None


def _same_rows(left: GeometryAssociation, right: GeometryAssociation, /) -> bool:
    """Compare all association claims independent of row order."""
    if (
        left.association_kind != right.association_kind
        or left.source_id != right.source_id
        or left.source_revision != right.source_revision
        or left.target_entity_set_id != right.target_entity_set_id
        or left.source_boundary_witness_id != right.source_boundary_witness_id
        or left.provenance != right.provenance
        or left.parent_association_id != right.parent_association_id
    ):
        return False
    left_ids = np.asarray(left.target_global_ids, dtype=np.int64)
    right_ids = np.asarray(right.target_global_ids, dtype=np.int64)
    order_left = np.argsort(left_ids, kind="stable")
    order_right = np.argsort(right_ids, kind="stable")
    if not np.array_equal(left_ids[order_left], right_ids[order_right]):
        return False
    for name in (
        "source_dimensions", "source_indices", "residuals", "resolved",
        "ambiguous", "parameters", "orientations", "parent_dimensions", "parent_ids",
    ):
        if not np.array_equal(
            np.asarray(getattr(left, name))[order_left],
            np.asarray(getattr(right, name))[order_right], equal_nan=True,
        ):
            return False
    for name in (
        "source_entity_ids", "source_roles", "source_occurrence_paths",
        "source_region_sides", "source_parameter_intervals", "source_parameter_points",
        "target_parameter_intervals", "target_reference_pieces",
    ):
        left_values, right_values = getattr(left, name), getattr(right, name)
        if tuple(left_values[int(row)] for row in order_left) != tuple(
            right_values[int(row)] for row in order_right
        ):
            return False
    return left.exact == right.exact


def _validate_chart_association(
    charts: SurfaceSourceCharts,
    parent: GeometryAssociation,
    tolerance: float,
    authority: NativeLayerCoreSource | None,
    /,
) -> _WallAuthority:
    """Reprove chart-anchored vertex claims, optionally using the original query."""
    charts.require_root(charts.mesh, charts.geometry)
    from ..geometry._source_fidelity import reprove_actual_planar_chart_rows

    actual_chart_rows = reprove_actual_planar_chart_rows(charts)
    domain = charts.domain
    if (parent.source_id, parent.source_revision) != (domain.source_id, domain.source_revision):
        raise ValueError("Wall association changes its original source revision.")
    parent.validate_target(charts.mesh.entity_set(0))
    if parent.association_kind is GeometryAssociationKind.BREP:
        bindings = validate_cad_curve_ancestry(charts, tolerance)
    elif parent.association_kind is GeometryAssociationKind.SURFACE:
        from ..geometry._surface_source_support import validate_surface_curve_ancestry

        bindings = validate_surface_curve_ancestry(charts, tolerance)
    else:
        raise ValueError("Wall source association must bind BRep or surface strata.")
    edge_vertices, _ = _cad_root_edges(charts)
    mesh = charts.mesh
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    classes: dict[int, set[tuple[int, int, Fraction | tuple[Fraction, Fraction] | None]]] = {}
    chart_rows = {int(identifier): row for row, identifier in enumerate(charts.cell_global_ids)}
    for block in mesh.blocks:
        if block.cell_kind not in ("triangle", "quadrilateral"):
            raise ValueError("Wall source proof requires triangular or quadrilateral charts.")
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True):
            row = chart_rows[int(identifier)]
            patch = int(charts.root_patches[row])
            for local, vertex in enumerate(vertices):
                vertex_id = int(vertex_ids[int(vertex)])
                uv = tuple(charts.root_parameter_exact[row][local])
                classes.setdefault(vertex_id, set()).add((2, patch, uv))
    curve_classes: dict[int, set[tuple[int, int, Fraction | None]]] = {}
    for edge_id, uses in bindings.items():
        endpoints = edge_vertices[edge_id]
        for use in uses:
            for endpoint, parameter in zip(endpoints, use.parameters, strict=True):
                vertex_id = int(vertex_ids[int(endpoint)])
                if parameter == 0:
                    item = (0, int(domain.curves[use.curve_row].start), None)
                elif parameter == 1:
                    item = (0, int(domain.curves[use.curve_row].end), None)
                else:
                    item = (1, int(use.curve_row), parameter)
                curve_classes.setdefault(vertex_id, set()).add(item)
    rows = {int(identifier): row for row, identifier in enumerate(parent.target_global_ids)}
    dimensions = np.asarray(parent.source_dimensions, dtype=np.int64)
    indices = np.asarray(parent.source_indices, dtype=np.int64)
    resolved = np.asarray(parent.resolved, dtype=np.bool_)
    ambiguous = np.asarray(parent.ambiguous, dtype=np.bool_)
    proved = {}
    for identifier in vertex_ids.tolist():
        row = rows.get(int(identifier))
        if row is None or not resolved[row] or ambiguous[row]:
            raise ValueError("Every authored wall vertex must have a resolved source class.")
        possibilities = curve_classes.get(int(identifier))
        if possibilities:
            if len(possibilities) != 1:
                raise ValueError("Wall curve incidences disagree on a source vertex class.")
            dimension, source_row, parameter = next(iter(possibilities))
        else:
            incident = classes.get(int(identifier), set())
            patches = {item[1] for item in incident}
            if len(patches) != 1:
                raise ValueError("A wall vertex has no unique authored source patch.")
            dimension, source_row, parameter = 2, next(iter(patches)), None
        expected_role = (GeometrySourceEntityRole.VERTEX, GeometrySourceEntityRole.EDGE,
                         GeometrySourceEntityRole.FACET)[dimension]
        if (
            int(dimensions[row]) != dimension
            or int(indices[row]) != int(domain.source_indices[dimension][source_row])
            or parent.source_entity_ids[row] != domain.entity_id(dimension, source_row)
            or parent.source_occurrence_paths[row] != tuple(domain.source_occurrences[dimension][source_row])
            or parent.source_roles[row] not in (None, expected_role)
        ):
            raise ValueError("Authored wall vertex changes its independently proved source class.")
        sides = parent.source_region_sides[row]
        if sides != (None, None):
            permitted = []
            for patch in {item[1] for item in classes.get(int(identifier), set())}:
                region_sides = [None, None]
                for region_row, region in enumerate(domain.regions):
                    for region_patch, orientation in region.boundary:
                        if region_patch != patch:
                            continue
                        slot = 0 if orientation > 0 else 1
                        region_id = domain.entity_id(3, region_row)
                        if region_sides[slot] not in (None, region_id):
                            raise ValueError("Wall source patch has conflicting material sides.")
                        region_sides[slot] = region_id
                permitted.append(tuple(region_sides))
            if not any(
                all(value is None or value == candidate[slot]
                    for slot, value in enumerate(sides))
                for candidate in permitted
            ):
                raise ValueError("Authored wall vertex changes its source material sides.")
        proved[int(identifier)] = (dimension, source_row, parameter)
    projection = None
    if authority is not None and authority.source_wall_projection is not None:
        if parent.association_kind is not GeometryAssociationKind.BREP:
            raise ValueError("A BRep wall projection cannot prove a surface association.")
        if authority.source_wall_association_policy is None:
            raise ValueError("Projected wall source is missing its original association policy.")
        # Reprepare from retained original query inputs, not cached projection arrays.
        restored = NativeLayerCoreSource.from_stored_fields(authority)
        projection = restored.source_wall_projection
        if projection is None:
            raise ValueError("Projected wall source lost its original query.")
        proved_classes = _Classes(
            np.asarray([proved[int(identifier)][0] for identifier in vertex_ids], dtype=np.int64),
            np.asarray([
                domain.source_indices[proved[int(identifier)][0]][proved[int(identifier)][1]]
                for identifier in vertex_ids
            ], dtype=np.int64),
            np.full(vertex_ids.shape, BRepProjectionStatus.UNIQUE, dtype=np.int8),
            tuple(
                tuple(domain.source_occurrences[proved[int(identifier)][0]][proved[int(identifier)][1]])
                for identifier in vertex_ids
            ),
        )
        replayed = _brep_association(
            projection, mesh, 0, proved_classes,
            np.asarray(mesh.coordinates, dtype=np.float64),
            authority.source_wall_association_policy,
            provenance=GeometryAssociationProvenance.CLASSIFICATION,
            parent_association_id=None,
        )
        if np.any(
            np.asarray(replayed.residuals, dtype=np.float64)
            > authority.source_wall_association_policy.classification_tolerance
        ):
            raise ValueError("Authored wall vertex lies outside its original classification tolerance.")
        if not _same_rows(parent, replayed):
            raise ValueError("Authored projected wall rows differ from their original source query.")
    else:
        root_tables = tuple(
            item for item in charts.associations
            if item.target_entity_set_id == mesh.entity_set(0).entity_set_id
        )
        if len(root_tables) != 1 or not _same_rows(parent, root_tables[0]):
            raise ValueError("Wall rows must retain the independently charted source association.")
        parameters = np.asarray(parent.parameters, dtype=np.float64)
        for identifier, (dimension, source_row, parameter) in proved.items():
            row = rows[identifier]
            if dimension == 0:
                valid = bool(np.all(np.isnan(parameters[row])))
            elif dimension == 1:
                if not isinstance(parameter, Fraction):
                    raise ValueError("Wall curve vertex lacks its exact source parameter.")
                expected = float(
                    domain.curve_native_parameter(source_row, parameter)
                    if parent.association_kind is GeometryAssociationKind.BREP
                    else parameter
                )
                valid = parameters[row, 0] == expected and np.isnan(parameters[row, 1])
            else:
                valid = any(
                    item[1] == source_row and isinstance(item[2], tuple) and np.array_equal(
                        parameters[row], np.asarray(tuple(float(v) for v in item[2]))
                    )
                    for item in classes[identifier]
                )
            if not valid:
                raise ValueError("Wall vertex parameter differs from its exact source chart.")
    return _WallAuthority(charts, parent, bindings, edge_vertices, proved, projection, actual_chart_rows)


def validate_wall_source(source: NativeLayerCoreSource, tolerance: float, /) -> _WallAuthority:
    """Reprove original wall rows without using proximity to choose their class."""
    authority = source if source.reference_source is None else source.reference_source
    parent = authority.layers.wall_association
    charts = authority.source_wall_charts
    if parent is None or charts is None:
        raise ValueError("Wall source replay requires the original association and wall charts.")
    charts.require_root(authority.layers.source_wall, charts.geometry)
    return _validate_chart_association(charts, parent, tolerance, authority)


def validate_full_boundary_charts(source: NativeLayerCoreSource, tolerance: float, /) -> None:
    """Independently reprove the retained full source atlas and its vertex rows."""
    from ..geometry._source_fidelity import certify_source_fidelity

    charts = source.source_boundary_charts
    if charts is None or source.fidelity_source is None:
        raise ValueError("Full wall-source replay requires its original boundary atlas and query.")
    charts.require_root(charts.mesh, charts.geometry)
    charts.require_bound(charts.mesh, charts.geometry, source.fidelity_source)
    for table in charts.associations:
        if (
            table.exact
            or table.provenance is not GeometryAssociationProvenance.PROVIDER
            or table.parent_association_id is not None
            or np.any(np.asarray(table.parent_dimensions, dtype=np.int8) != -1)
            or np.any(np.asarray(table.parent_ids, dtype=np.int64) != -1)
            or table.source_boundary_witness_id is not None
            or any(value is not None for value in table.target_reference_pieces)
            or any(value is not None for value in table.target_parameter_intervals)
        ):
            raise ValueError("Full source atlas lineage requires its original parent authority.")
    vertex_tables = tuple(
        item for item in charts.associations
        if item.target_entity_set_id == charts.mesh.entity_set(0).entity_set_id
    )
    if len(vertex_tables) != 1:
        raise ValueError("The full source atlas requires one exact vertex association.")
    vertex_table = vertex_tables[0]
    if (
        vertex_table.exact
        or np.any(np.asarray(vertex_table.residuals, dtype=np.float64) != 0)
        or np.any(np.asarray(vertex_table.orientations, dtype=np.int8) != 0)
    ):
        raise ValueError("Full source atlas vertex evidence changes its native provider origin.")
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
    )
    if any(
        int(dimension) not in (0, 1, 2) or role is not roles[int(dimension)]
        for dimension, role in zip(
            np.asarray(vertex_table.source_dimensions), vertex_table.source_roles, strict=True
        )
    ):
        raise ValueError("Full source vertex roles differ from native source strata.")
    root_proof = _validate_chart_association(charts, vertex_tables[0], tolerance, None)
    face_tables = tuple(
        item for item in charts.associations
        if item.target_entity_set_id == charts.mesh.entity_set(2).entity_set_id
    )
    face_rows = {}
    for table in face_tables:
        for row, identifier in enumerate(table.target_global_ids):
            if int(identifier) in face_rows:
                raise ValueError("The full source atlas repeats a face association.")
            face_rows[int(identifier)] = (table, row)
    if set(face_rows) != set(charts.cell_global_ids):
        raise ValueError("The full source atlas must associate every original face.")
    if not charts.mesh.blocks or len({block.cell_kind for block in charts.mesh.blocks}) != 1:
        raise ValueError("The full source atlas must retain one surface cell family.")
    kind = charts.mesh.blocks[0].cell_kind
    coordinates = np.asarray(charts.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate([np.asarray(block.vertices, dtype=np.int64) for block in charts.mesh.blocks])
    parameters = np.asarray(charts.root_parameters, dtype=np.float64)
    patches = np.asarray(charts.root_patches, dtype=np.int64)
    if kind == "triangle":
        centers = np.mean(parameters, axis=1)
        weights = np.asarray(((0.5, 0.5, 0.0), (0.0, 0.5, 0.5),
                              (0.5, 0.0, 0.5), (1.0 / 3.0,) * 3))
        sample_parameters = np.einsum("qv,cvd->cqd", weights, parameters)
        sample_points = charts.domain.evaluate(
            np.repeat(patches, len(weights)), sample_parameters.reshape((-1, 2))
        ).reshape((-1, len(weights), 3))
        chord_points = np.einsum("qv,cvd->cqd", weights, coordinates[cells])
        expected_residuals = np.max(np.linalg.norm(sample_points - chord_points, axis=2), axis=1)
        normals, _ = charts.domain.oriented_normals(patches, centers)
        face_points = coordinates[cells]
        normals_mesh = np.cross(
            face_points[:, 1] - face_points[:, 0],
            face_points[:, 2] - face_points[:, 0],
        )
        expected_orientations = np.sign(np.sum(normals_mesh * normals, axis=1)).astype(np.int8)
    elif kind == "quadrilateral":
        centers = np.asarray([
            [float(sum((point[axis] for point in chart), Fraction(0)) / len(chart))
             for axis in range(2)]
            for chart in charts.root_parameter_exact
        ])
        expected_residuals = np.zeros(len(charts.cell_global_ids), dtype=np.float64)
        expected_orientations = np.ones(len(charts.cell_global_ids), dtype=np.int8)
    else:
        raise ValueError("The full source atlas requires triangle or quadrilateral charts.")
    curve_tables = tuple(
        table for table in charts.associations
        if table.association_kind is GeometryAssociationKind.CURVE
        and table.target_entity_set_id == charts.mesh.entity_set(1).entity_set_id
    )
    if curve_tables and source.source_boundary_schedule is None:
        raise ValueError("Full source curve residual replay requires its original surface schedule.")
    if curve_tables:
        from .providers._native_surface import _curve_edge_samples

        schedule = source.source_boundary_schedule
        if schedule is None:
            raise ValueError("The original boundary surface schedule is absent.")
        chart_rows = {int(value): row for row, value in enumerate(charts.cell_global_ids)}
        for table in curve_tables:
            if table.exact:
                raise ValueError("Native source curve rows cannot claim exact sampled residuals.")
            if np.any(~np.isnan(np.asarray(table.parameters, dtype=np.float64))):
                raise ValueError("Native source curve rows cannot claim unrelated point parameters.")
            for row, identifier in enumerate(table.target_global_ids):
                uses = root_proof.curve_bindings.get(int(identifier))
                if not uses:
                    raise ValueError("A full source curve edge lacks its proved coedge ancestry.")
                keys = {(int(use.curve_row), tuple(use.parameters)) for use in uses}
                if len(keys) != 1:
                    raise ValueError("A full source curve edge has conflicting exact source parameters.")
                curve, endpoints = next(iter(keys))
                vertices = root_proof.edge_vertices[int(identifier)]
                first, last = (0, 1) if endpoints[0] < endpoints[1] else (1, 0)
                lower, upper = endpoints[first], endpoints[last]
                if kind == "triangle":
                    expected_residual = _curve_edge_samples(
                        charts.domain, coordinates[vertices[first]], coordinates[vertices[last]],
                        curve, lower, upper, schedule.curve_schedule.fidelity_samples,
                    )
                else:
                    expected_residual = 0.0
                if float(np.asarray(table.residuals)[row]) != expected_residual:
                    raise ValueError("A full source curve changes its original sampled residual.")
                possible = []
                for use in uses:
                    patch = int(charts.root_patches[chart_rows[int(use.root_cell_id)]])
                    sides = [None, None]
                    for region_row, region in enumerate(charts.domain.regions):
                        for source_patch, orientation in region.boundary:
                            if source_patch == patch:
                                slot = 0 if orientation > 0 else 1
                                region_id = charts.domain.entity_id(3, region_row)
                                if sides[slot] not in (None, region_id):
                                    raise ValueError("A source coedge has conflicting material-side incidence.")
                                sides[slot] = region_id
                    possible.append(tuple(sides))
                claimed = table.source_region_sides[row]
                if claimed != (None, None) and not any(
                    all(value is None or value == allowed[slot]
                        for slot, value in enumerate(claimed))
                    for allowed in possible
                ):
                    raise ValueError("A full source curve changes its original material sides.")
    for chart_row, identifier in enumerate(charts.cell_global_ids):
        table, row = face_rows[int(identifier)]
        patch = int(charts.root_patches[chart_row])
        if (
            table.source_entity_ids[row] != charts.domain.entity_id(2, patch)
            or int(np.asarray(table.source_dimensions)[row]) != 2
            or int(np.asarray(table.source_indices)[row]) != charts.domain.source_indices[2][patch]
            or table.source_occurrence_paths[row] != tuple(charts.domain.source_occurrences[2][patch])
            or table.source_roles[row] is not GeometrySourceEntityRole.FACET
            or not bool(np.asarray(table.resolved)[row])
            or bool(np.asarray(table.ambiguous)[row])
        ):
            raise ValueError("The full source atlas changes a source face or occurrence.")
        expected = centers[chart_row]
        if not np.array_equal(np.asarray(table.parameters)[row], expected):
            raise ValueError("The full source atlas changes its original face parameter.")
        if (
            float(np.asarray(table.residuals)[row]) != float(expected_residuals[chart_row])
            or int(np.asarray(table.orientations)[row]) != int(expected_orientations[chart_row])
        ):
            raise ValueError("The full source atlas changes its original face residual or orientation.")
        permitted = [None, None]
        for region_row, region in enumerate(charts.domain.regions):
            for source_patch, orientation in region.boundary:
                if source_patch != patch:
                    continue
                slot = 0 if orientation > 0 else 1
                identifier_ = charts.domain.entity_id(3, region_row)
                if permitted[slot] not in (None, identifier_):
                    raise ValueError("A full source face has conflicting material-side incidence.")
                permitted[slot] = identifier_
        if any(value is not None and value != permitted[slot]
               for slot, value in enumerate(table.source_region_sides[row])):
            raise ValueError("The full source atlas changes its original face material sides.")
    fidelity = certify_source_fidelity(
        charts.mesh, charts.geometry, charts.boundary_source,
        tolerance=tolerance, witness=charts.chart_witness,
    )
    if fidelity.status != "certified":
        raise ValueError("The full source atlas lacks certified source fidelity.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..discretization import CellMesh
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._exact_algebraic_geometry import ExactScalar
    from ..geometry._source_boundary_records import SourceBoundaryChartWitness, SourceBoundaryQuery
    from ..geometry._surface_source_support import _CadCurveUseBinding
    from ..geometry.brep._projection import NativeBRepProjection
    from ._contracts import VolumeMeshingSpec
    from ._layer_source_association_replay import LayerAssociationReplay

type _WallIncidence = tuple[dict[int, int], dict[int, int], dict[int, int]]
type _WallPoint = tuple[Fraction, ...]
type _WallChartRow = tuple[
    int, int, int, int, int, tuple[int, ...], tuple[_WallPoint, ...],
    tuple[_WallPoint, ...], int | None,
]


def _authored_wall_incidence(
    source: NativeLayerCoreSource, wall: _WallAuthority, /
) -> _WallIncidence:
    """Bind root wall cells and edges to authored layer entities by vertex ancestry."""
    from ._topology_edit import entity_keys

    authored = source if source.reference_source is None else source.reference_source
    layer = authored.layers
    root_mesh = wall.charts.mesh
    mesh = layer.mesh
    wall_rows = np.asarray(layer.wall_vertices, dtype=np.int64)
    root_ids = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
    layer_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    if wall_rows.shape != root_ids.shape or np.any(wall_rows < -1) or np.any(wall_rows >= len(layer_ids)):
        raise ValueError("Authored wall vertex ancestry has incompatible extents.")
    active = wall_rows >= 0
    if len(np.unique(wall_rows[active])) != np.count_nonzero(active):
        raise ValueError("Authored wall vertices must have distinct layer descendants.")
    if not np.array_equal(
        np.asarray(root_mesh.coordinates, dtype=np.float64)[active].view(np.uint64),
        np.asarray(mesh.coordinates, dtype=np.float64)[wall_rows[active]].view(np.uint64),
    ):
        raise ValueError("Authored wall vertex ancestry changes source coordinates.")
    vertices = {
        int(root_ids[row]): int(layer_ids[index])
        for row, index in enumerate(wall_rows) if index >= 0
    }
    def incidence(dimension: int) -> dict[tuple[int, ...], int]:
        return {
            tuple(int(value) for value in key if value >= 0): int(identifier)
            for key, identifier in zip(
                entity_keys(mesh, dimension), mesh.entity_set(dimension).entity_ids,
                strict=True,
            )
        }
    layer_faces, layer_edges = incidence(2), incidence(1)
    faces = {}
    for block in root_mesh.blocks:
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
            root_vertices = tuple(int(root_ids[int(row)]) for row in rows)
            if not all(vertex in vertices for vertex in root_vertices):
                continue
            key = tuple(sorted(vertices[vertex] for vertex in root_vertices))
            face = layer_faces.get(key)
            if face is None or face in faces:
                raise ValueError("Authored wall chart lacks a unique layer face.")
            faces[face] = int(identifier)
    edges = {}
    for identifier, rows in wall.edge_vertices.items():
        endpoints = tuple(int(root_ids[row]) for row in rows)
        if not all(vertex in vertices for vertex in endpoints):
            continue
        key = tuple(sorted(vertices[vertex] for vertex in endpoints))
        edge = layer_edges.get(key)
        if edge is None or edge in edges:
            raise ValueError("Authored wall edge lacks a unique layer edge.")
        edges[edge] = int(identifier)
    if not faces:
        raise ValueError("Authored layer has no source wall chart faces.")
    return vertices, edges, faces


def _quad_wall_weights(
    u: Fraction, v: Fraction, diagonal: int, /
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Exact piecewise-affine weights on the authored quad diagonal."""
    if not (0 <= u <= 1 and 0 <= v <= 1):
        raise ValueError("Target wall reference point leaves its authored quadrilateral.")
    if diagonal == 1:
        rotated = _quad_wall_weights(v, 1 - u, 0)
        return (rotated[3], rotated[0], rotated[1], rotated[2])
    if diagonal != 0:
        raise ValueError("A quadrilateral wall chart has an invalid diagonal.")
    if u >= v:
        return (1 - u, u - v, v, Fraction(0))
    return (1 - v, Fraction(0), u, v - u)


def _target_wall_charts(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    layer: LayerAssociationReplay,
    wall: _WallAuthority,
    incidence: _WallIncidence,
    /,
) -> tuple[_WallChartRow, ...]:
    """Map target wall-face corners to exact authored wall-chart coordinates."""
    from ..discretization._nested_reference import nested_cell_reference_maps
    from ..discretization._reference_cell import reference_cell_topology
    from ._layer_curving_records import (
        _barycentric_3d, _fraction, _fraction3, _prepare_barycentric_solver_3d,
    )
    from ._volume_entity_containment import _cell_entities, reference_entity_containment

    authored = source if source.reference_source is None else source.reference_source
    original = authored.layers.mesh
    vertices, _, faces = incidence
    checked = reference_entity_containment(mesh, original, layer.reference_maps)
    if tuple(checked) != tuple(layer.containment):
        raise ValueError("Layer containment differs from its exact reference maps.")

    def cells(value: CellMesh) -> dict[int, tuple[str, tuple[int, ...]]]:
        vertex_ids = np.asarray(value.vertex_global_ids, dtype=np.int64)
        return {
            int(identifier): (block.cell_kind, tuple(int(vertex_ids[int(row)]) for row in rows))
            for block in value.blocks
            for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
        }

    target_cells, original_cells = cells(mesh), cells(original)
    target_entities, original_entities = _cell_entities(mesh), _cell_entities(original)
    references = {}
    for group in layer.reference_maps:
        for item in nested_cell_reference_maps(group):
            key = (item.target_cell_id, item.source_cell_id)
            if key in references:
                raise ValueError("Layer cell ancestry contains duplicate reference maps.")
            references[key] = item
    root_ids = np.asarray(wall.charts.mesh.vertex_global_ids, dtype=np.int64)
    roots = {}
    chart_rows = {int(identifier): row for row, identifier in enumerate(wall.charts.cell_global_ids)}
    for block in wall.charts.mesh.blocks:
        if block.cell_kind not in ("triangle", "quadrilateral"):
            raise ValueError("Authored wall reference charts must be triangles or quadrilaterals.")
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
            root = int(identifier)
            roots[root] = tuple(int(root_ids[int(row)]) for row in rows)
    output = []
    for relation in checked:
        if relation.fine_dimension != 2 or relation.coarse_dimension != 2:
            continue
        root = faces.get(relation.coarse_id)
        if root is None:
            continue
        target_kind, target_vertices = target_cells[relation.fine_cell_id]
        coarse_kind, coarse_vertices = original_cells[relation.coarse_cell_id]
        target_topology = reference_cell_topology(target_kind)
        coarse_topology = reference_cell_topology(coarse_kind)
        try:
            local_face = target_entities[relation.fine_cell_id][1][2].index(relation.fine_id)
            coarse_face = original_entities[relation.coarse_cell_id][1][2].index(relation.coarse_id)
            root_vertices = roots[root]
            root_layer_vertices = tuple(vertices[value] for value in root_vertices)
            coarse_slots = tuple(coarse_vertices.index(value) for value in root_layer_vertices)
            if set(coarse_slots) != set(coarse_topology.entities[2][coarse_face]):
                raise ValueError("Wall root face differs from its authored layer incidence.")
            reference = references[(relation.fine_cell_id, relation.coarse_cell_id)]
            chart_row = chart_rows[root]
        except (KeyError, ValueError) as exc:
            raise ValueError("Wall face ancestry does not bind its exact layer chart.") from exc
        corners = tuple(
            _fraction3(tuple(Fraction(float(value)) for value in coarse_topology.vertices[slot]))
            for slot in coarse_slots
        )
        diagonal = None
        if len(corners) == 4:
            face_slots = coarse_topology.entities[2][coarse_face]
            order = tuple(coarse_slots.index(slot) for slot in face_slots)
            if any((order[(i + 1) % 4] - order[i]) % 4 not in (1, 3) for i in range(4)):
                raise ValueError("Authored quadrilateral wall incidence changes its boundary order.")
            diagonal = order[0] % 2
        if len(corners) == 3:
            solver = _prepare_barycentric_solver_3d(corners)
            def weights_at(point: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
                return tuple(_fraction(value) for value in _barycentric_3d(
                    _fraction3(point), corners, solver=solver
                ))
        else:
            basis = (corners[0], corners[1], corners[3])
            if any(corners[2][axis] != corners[1][axis] + corners[3][axis] - corners[0][axis]
                   for axis in range(3)):
                raise ValueError("Authored layer quadrilateral is not an affine reference face.")
            solver = _prepare_barycentric_solver_3d(basis)
            def weights_at(point: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
                _, u, v = _barycentric_3d(_fraction3(point), basis, solver=solver)
                if diagonal is None:
                    raise ValueError("A quadrilateral wall chart lacks its diagonal.")
                return _quad_wall_weights(_fraction(u), _fraction(v), diagonal)
        local_vertices = target_topology.entities[2][local_face]
        target_ids = tuple(target_vertices[slot] for slot in local_vertices)
        barycentric = tuple(weights_at(reference.source_reference_vertices[slot])
                            for slot in local_vertices)
        if any(
            any(not isinstance(value, Fraction) or value < 0 for value in weights)
            or sum(weights, Fraction(0)) != 1
            for weights in barycentric
        ):
            raise ValueError("Target wall face leaves its exact authored root chart.")
        parameters = (
            wall.charts.root_parameter_exact[chart_row]
            if wall.actual_chart_rows is None else wall.actual_chart_rows[chart_row]
        )
        uv = tuple(
            tuple(sum((weights[index] * parameters[index][axis] for index in range(len(parameters))), Fraction(0))
                  for axis in range(2))
            for weights in barycentric
        )
        if wall.actual_chart_rows is not None:
            from ._layer_curving_records import _require_reoriented_wall_side

            area = ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
                    - (uv[1][1] - uv[0][1]) * (uv[2][0] - uv[0][0]))
            _require_reoriented_wall_side(
                wall.charts, chart_row, wall.actual_chart_rows[chart_row], area
            )
        output.append((relation.fine_cell_id, relation.fine_id, local_face, root,
                       chart_row, target_ids, barycentric, uv, diagonal))
    if not output:
        raise ValueError("No target face retains exact authored wall ancestry.")
    return tuple(output)


def _wall_witness_query(
    source: NativeLayerCoreSource, witness: SourceBoundaryChartWitness, /
) -> SourceBoundaryQuery:
    """Derive the grown-wall query and require the witness's exact source scope."""
    authored = source if source.reference_source is None else source.reference_source
    charts = authored.source_wall_charts
    if charts is not None:
        from ._layer_curving import layer_wall_boundary_query

        selected = layer_wall_boundary_query(authored.layers, charts)
        if (
            (witness.source_id, witness.source_revision)
            != (selected.source_id, selected.source_revision)
            or witness.source_query_id != selected.source_scope_id
        ):
            raise ValueError("Wall witness does not bind the canonical grown-wall source scope.")
        return selected
    query = source.fidelity_source
    if query is None:
        raise ValueError("Wall replay requires its original source boundary query.")
    identity = getattr(query, "source_scope_id", None)
    if identity is None:
        identity = getattr(query, "source_query_id", getattr(query, "query_id", None))
    if (
        identity is None
        or witness.source_query_id != identity
        or (witness.source_id, witness.source_revision)
        != (query.source_id, query.source_revision)
    ):
        raise ValueError("Wall witness does not bind the retained source query.")
    return query


def _validate_target_wall_witness(
    query: SourceBoundaryQuery,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
    wall: _WallAuthority,
    rows: tuple[_WallChartRow, ...],
    /,
) -> None:
    """Reprove renewed target charts against the authored layer-wall maps."""
    from ..discretization._reference_cell import reference_cell_topology
    from ..geometry._source_boundary_records import SourceBoundaryChartWitness
    from ._layer_curving_records import (
        _barycentric_3d, _fraction, _fraction3, _prepare_barycentric_solver_3d,
    )

    if not isinstance(witness, SourceBoundaryChartWitness):
        raise ValueError("Wall replay requires its renewed source boundary chart witness.")
    witness.require_current(mesh, geometry, query)
    if witness.chart_dimension != 2 or witness.reference_dimension != 3:
        raise ValueError("Wall replay requires volume-face source charts.")
    cells = {
        int(identifier): block.cell_kind
        for block in mesh.blocks
        for identifier in block.global_ids
    }
    domain = wall.charts.domain
    expected = {}
    by_face = {}
    for cell_id, face_id, local_face, root, chart_row, vertex_ids, barycentric, uv, diagonal in rows:
        key = (cell_id, face_id, local_face)
        value = (root, chart_row, vertex_ids, barycentric, uv, diagonal)
        if key in expected and expected[key] != value:
            raise ValueError("One target wall face has conflicting authored chart maps.")
        expected[key] = value
        by_face.setdefault(face_id, set()).add(key)
    seen = set()
    for index, (cell_id, face_id, local_face) in enumerate(zip(
        witness.cell_global_ids, witness.entity_global_ids,
        witness.local_entity_indices, strict=True,
    )):
        key = (int(cell_id), int(face_id), int(local_face))
        if int(face_id) not in by_face:
            continue
        item = expected.get(key)
        if item is None:
            raise ValueError("Wall witness changes the target face owner or local incidence.")
        _, chart_row, _, root_weights, uv, diagonal = item
        patch = int(wall.charts.root_patches[chart_row])
        if (
            witness.source_entity_ids[index] != domain.entity_id(2, patch)
            or tuple(witness.occurrence_paths[index]) != tuple(domain.source_occurrences[2][patch])
        ):
            raise ValueError("Wall witness changes the authored patch occurrence.")
        topology = reference_cell_topology(cells[key[0]])
        slots = topology.entities[2][key[2]]
        if len(slots) < 3:
            raise ValueError("Target wall face has fewer than three vertices.")
        corners = tuple(
            _fraction3(tuple(Fraction(float(value)) for value in topology.vertices[slot]))
            for slot in slots
        )
        solver = _prepare_barycentric_solver_3d(corners[:3])
        root_count = len(root_weights[0])
        if root_count == 3:
            coordinates = root_weights
        elif root_count == 4:
            coordinates = tuple((weights[1] + weights[2], weights[2] + weights[3])
                                for weights in root_weights)
        else:
            raise ValueError("Wall source chart has unsupported reference corners.")
        def root_coordinates(point: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
            weights = _barycentric_3d(_fraction3(point), corners[:3], solver=solver)
            return tuple(
                _fraction(sum((weights[i] * coordinates[i][axis] for i in range(3)), Fraction(0)))
                for axis in range(len(coordinates[0]))
            )
        for point, declared in zip(corners, coordinates, strict=True):
            if root_coordinates(point) != declared:
                raise ValueError("Target wall face changes its affine root reference map.")
        root_parameters = (
            wall.charts.root_parameter_exact[chart_row]
            if wall.actual_chart_rows is None else wall.actual_chart_rows[chart_row]
        )
        target_chart = witness.target_reference_chart(index)
        reference_points = tuple(root_coordinates(point) for point in target_chart)
        if root_count == 4:
            mixed = tuple(root_parameters[0][axis] - root_parameters[1][axis]
                          + root_parameters[2][axis] - root_parameters[3][axis]
                          for axis in range(2))
            signs = tuple((point[0] - point[1]) if diagonal == 0
                          else (point[0] + point[1] - 1) for point in reference_points)
            if any(mixed) and min(signs) < 0 < max(signs):
                raise ValueError("A wall witness piece crosses a non-affine root chart diagonal.")
            if diagonal is None:
                raise ValueError("A quadrilateral wall chart lacks its diagonal.")
            chart_weights = tuple(
                _quad_wall_weights(point[0], point[1], diagonal) for point in reference_points
            )
        else:
            chart_weights = reference_points
        for weights, actual in zip(
            chart_weights, witness.source_parameter_chart(index), strict=True
        ):
            expected_uv = tuple(
                sum((weights[i] * root_parameters[i][axis] for i in range(root_count)), Fraction(0))
                for axis in range(2)
            )
            if expected_uv != actual:
                raise ValueError("Wall witness source parameters differ from authored ancestry.")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Renewed source witness does not cover every target wall face.")


def replay_layer_wall_associations(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
    layer: LayerAssociationReplay,
    maximum_deviation: float | None,
    /,
) -> tuple[GeometryAssociation, ...]:
    """Rebuild authored-wall source associations from current exact layer ancestry."""
    from ._layer_wall_source_records import wall_records
    query = _wall_witness_query(source, witness)
    if maximum_deviation is not None and (
        not np.isfinite(maximum_deviation) or maximum_deviation < 0.0
    ):
        raise ValueError("Wall replay deviation must be finite and nonnegative.")

    tolerances = tuple(
        value for value in (source.maximum_deviation, maximum_deviation)
        if value is not None
    )
    if tolerances:
        proof_bound = min(tolerances)
        if proof_bound == 0.0 and source.maximum_deviation is None:
            from ..geometry._source_fidelity import certify_source_fidelity

            exact = certify_source_fidelity(mesh, geometry, query, tolerance=0.0, witness=witness)
            if exact.status != "certified":
                raise ValueError("The source and target do not prove exact zero wall deviation.")
    else:
        from ..geometry._meshing_domain import MeshingDomain
        from ..geometry._source_fidelity import certify_source_fidelity
        original_domain = (
            source.source_domain
            if source.source_domain is not None
            else None if source.source_boundary_charts is None
            else source.source_boundary_charts.domain
        )

        if (
            not isinstance(original_domain, MeshingDomain)
            or original_domain.accuracy != "exact_represented"
            or source.fidelity_source is None
        ):
            raise ValueError("Wall replay requires an explicit bound or an exact source theorem.")
        validate_full_boundary_charts(source, 0.0)
        exact = certify_source_fidelity(
            mesh, geometry, query, tolerance=0.0, witness=witness
        )
        if exact.status != "certified":
            raise ValueError("The source and target do not prove exact zero wall deviation.")
        proof_bound = 0.0
    wall = validate_wall_source(source, proof_bound)
    incidence = _authored_wall_incidence(source, wall)
    rows = _target_wall_charts(source, mesh, layer, wall, incidence)
    _validate_target_wall_witness(query, mesh, geometry, witness, wall, rows)
    return wall_records(source, mesh, geometry, witness, wall, rows)

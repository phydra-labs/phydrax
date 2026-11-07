#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Issue wall association rows from proved exact root-chart containment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, cast, TYPE_CHECKING

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_transition_types import _geometry_map_expressions
from ..discretization._coordinate_enclosure import evaluate_expression
from ..discretization._exact_algebraic_geometry import (
    coefficient_rounding_error,
    ExactScalar,
)
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._domain_boundary import MeshingDomainBoundarySource
from ..geometry._meshing_domain import MeshingDomain, PatchCurveUse, PatchPoleUse
from ..geometry._rational import fraction_endpoint
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryQuery,
)
from ..geometry.brep._projection import NativeBRepProjection
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ._association import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
    source_boundary_association_piece_rows,
)
from ._layer_curving_records import _barycentric_3d, _prepare_barycentric_solver_3d
from ._topology_edit import entity_keys


if TYPE_CHECKING:
    from ._layer_wall_source_replay import _WallAuthority, _WallChartRow
    from .providers._native_sources import NativeLayerCoreSource

type _RegionSides = tuple[str | None, str | None]
type _Representative = tuple[np.ndarray, Fraction]


def _class_residuals(
    domain: MeshingDomain,
    points: Sequence[_Representative],
    classes: Sequence[tuple[int, int]],
    query_cache: dict[tuple[int, int], SourceBoundaryQuery],
    resolution: int,
    projection: NativeBRepProjection | None,
    /,
) -> np.ndarray:
    """Evaluate certified distance bounds only on already selected source strata."""
    from ..geometry.brep._projection_contracts import BRepProjectionStatus
    from .providers._native_surface import _domain_curve_source_query

    result = np.empty((len(points),), dtype=np.float64)
    rounded_points = np.asarray([value[0] for value in points], dtype=np.float64)
    groups = {}
    for row, (dimension, source_row) in enumerate(classes):
        groups.setdefault((dimension, source_row), []).append(row)
    curve_scope_ids = (
        {
            key: int(identifier)
            for key, identifier in zip(
                domain.scope_entity_key_banks[1], domain.scope_entity_ids(1), strict=True
            )
        }
        if projection is None and any(dimension == 1 for dimension, _ in groups)
        else {}
    )
    for (dimension, source_row), rows in groups.items():
        if projection is not None:
            projected = projection.project(
                rounded_points[rows],
                np.full(len(rows), dimension, dtype=np.int8),
                np.full(
                    len(rows),
                    domain.source_indices[dimension][source_row],
                    dtype=np.int64,
                ),
                occurrence_paths=(
                    tuple(domain.source_occurrences[dimension][source_row]),
                )
                * len(rows),
            )
            upper = np.asarray(projected.residuals, dtype=np.float64)
            status = np.asarray(projected.status, dtype=np.int8)
            if (
                upper.shape != (len(rows),)
                or not np.all(np.isfinite(upper))
                or not np.all(
                    np.isin(
                        status, (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.SEAM)
                    )
                )
            ):
                raise ValueError(
                    "Wall strata lack a resolved class-bound source projection."
                )
            result[rows] = [
                fraction_endpoint(
                    Fraction.from_float(float(value)) + points[row][1], upper=True
                )
                for row, value in zip(rows, upper, strict=True)
            ]
            continue
        key = (dimension, source_row)
        query = query_cache.get(key)
        if query is None:
            if dimension == 1:
                source_key = (
                    domain.entity_id(1, source_row),
                    tuple(domain.source_occurrences[1][source_row]),
                )
                scope_id = curve_scope_ids.get(source_key)
                if scope_id is None:
                    raise ValueError(
                        "A source curve has no occurrence-qualified scope identity."
                    )
                query = _domain_curve_source_query(
                    domain, np.asarray([scope_id], dtype=np.int64)
                )
            elif dimension == 2:
                query = MeshingDomainBoundarySource(
                    domain, (source_row,), resolution=resolution
                )
            else:
                raise ValueError(
                    "A new wall entity must belong to a source curve or patch."
                )
            query_cache[key] = query
        distance = query.boundary_distance(rounded_points[rows])
        upper = np.asarray(distance.upper, dtype=np.float64)
        if (
            distance.semantics != "certified"
            or upper.shape != (len(rows),)
            or not np.all(np.isfinite(upper))
        ):
            raise ValueError(
                "Wall strata require certified class-bound source distances."
            )
        result[rows] = [
            fraction_endpoint(
                Fraction.from_float(float(value)) + points[row][1], upper=True
            )
            for row, value in zip(rows, upper, strict=True)
        ]
    return result


def wall_records(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
    wall: _WallAuthority,
    rows: tuple[_WallChartRow, ...],
    /,
) -> tuple[GeometryAssociation, ...]:
    """Rebuild all wall vertices, edges and faces from exact chart restrictions."""
    if source.reference_source is not None:
        raise ValueError(
            "Mapped wall replay requires class-bound physical-source distance authority."
        )
    charts, parent = wall.charts, wall.parent
    domain, root = charts.domain, charts.mesh
    query_cache = {}
    root_ids = np.asarray(root.vertex_global_ids, dtype=np.int64)
    root_cells = {
        int(identifier): tuple(int(value) for value in vertices)
        for block in root.blocks
        for identifier, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    root_edges = {
        frozenset(int(value) for value in vertices): int(identifier)
        for identifier, vertices in wall.edge_vertices.items()
    }
    parent_rows = {
        int(identifier): row for row, identifier in enumerate(parent.target_global_ids)
    }
    parent_parameters = np.asarray(parent.parameters, dtype=np.float64)
    parent_residuals = np.asarray(parent.residuals, dtype=np.float64)
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
    )
    curve_tables = tuple(
        item
        for item in charts.associations
        if item.association_kind is GeometryAssociationKind.CURVE
        and item.target_entity_set_id == root.entity_set(1).entity_set_id
    )
    curve_rows = {}
    for table in curve_tables:
        for row, identifier in enumerate(table.target_global_ids):
            if int(identifier) in curve_rows:
                raise ValueError("Root wall curve edge has duplicate source rows.")
            curve_rows[int(identifier)] = (table, row)
    face_tables = tuple(
        item
        for item in charts.associations
        if item.target_entity_set_id == root.entity_set(2).entity_set_id
    )
    root_face_rows = {}
    for table in face_tables:
        for row, identifier in enumerate(table.target_global_ids):
            if int(identifier) in root_face_rows:
                raise ValueError("Root wall face has duplicate source rows.")
            root_face_rows[int(identifier)] = (table, row)
    elements, routes, _ = geometry.resolve(mesh)
    geometry_maps = _geometry_map_expressions(mesh, geometry, (elements, routes))
    cell_kinds = {
        int(identifier): block.cell_kind
        for block in mesh.blocks
        for identifier in block.global_ids
    }
    witness_rows = {}
    for index, (cell, face) in enumerate(
        zip(witness.cell_global_ids, witness.entity_global_ids, strict=True)
    ):
        witness_rows.setdefault((int(cell), int(face)), []).append(index)

    def reference_position(
        cell_id: int, local_face: int, positions: tuple[int, ...]
    ) -> tuple[Fraction, Fraction, Fraction]:
        topology = reference_cell_topology(cell_kinds[int(cell_id)])
        slots = topology.entities[2][int(local_face)]
        chosen = tuple(slots[position] for position in positions)
        coordinates = tuple(
            sum(
                (Fraction(float(topology.vertices[slot][axis])) for slot in chosen),
                Fraction(0),
            )
            / len(chosen)
            for axis in range(3)
        )
        return coordinates[0], coordinates[1], coordinates[2]

    def representative(
        cell_id: int, local_face: int, positions: tuple[int, ...]
    ) -> _Representative:
        reference = reference_position(cell_id, local_face, positions)
        exact = tuple(
            evaluate_expression(expression, reference)
            for expression in geometry_maps.expressions(int(cell_id))
        )
        rounded = np.asarray([float(value) for value in exact], dtype=np.float64)
        if not np.all(np.isfinite(rounded)):
            raise ValueError("Wall target representative is not finite.")
        error = fraction_sqrt_interval(
            sum(
                (
                    coefficient_rounding_error(value, float(shadow)) ** 2
                    for value, shadow in zip(exact, rounded, strict=True)
                ),
                Fraction(0),
            )
        )[1]
        return rounded, error

    def source_uv(
        cell_id: int, face_id: int, reference: tuple[Fraction, Fraction, Fraction]
    ) -> tuple[ExactScalar, ExactScalar]:
        values = set()
        for index in witness_rows.get((int(cell_id), int(face_id)), ()):
            target = cast(
                tuple[tuple[Fraction, Fraction, Fraction], ...],
                witness.target_reference_chart(index),
            )
            weights = _barycentric_3d(
                reference, target, solver=_prepare_barycentric_solver_3d(target)
            )
            if any(value < 0 for value in weights):
                continue
            source_chart = witness.source_parameter_chart(index)
            values.add(
                (
                    sum((weights[i] * source_chart[i][0] for i in range(3)), Fraction(0)),
                    sum((weights[i] * source_chart[i][1] for i in range(3)), Fraction(0)),
                )
            )
        if len(values) != 1:
            raise ValueError(
                "A wall representative lacks one exact source-chart parameter."
            )
        return next(iter(values))

    def contains(upper: tuple[Any, ...], lower: tuple[Any, ...]) -> bool:
        upper_dim, upper_row = upper[:2]
        lower_dim, lower_row = lower[:2]
        if upper_dim == lower_dim:
            return upper_row == lower_row
        if upper_dim == 1 and lower_dim == 0:
            curve = domain.curves[upper_row]
            return lower_row in (curve.start, curve.end)
        if upper_dim != 2:
            return False
        uses = tuple(use for loop in domain.patches[upper_row].loops for use in loop)
        if lower_dim == 1:
            return any(
                isinstance(use, PatchCurveUse) and use.curve == lower_row for use in uses
            )
        if lower_dim == 0:
            return any(
                (isinstance(use, PatchPoleUse) and use.corner == lower_row)
                or (
                    isinstance(use, PatchCurveUse)
                    and lower_row
                    in (domain.curves[use.curve].start, domain.curves[use.curve].end)
                )
                for use in uses
            )
        return False

    def patch_region_sides(patch: int) -> _RegionSides:
        result = [None, None]
        for region_row, region in enumerate(domain.regions):
            for region_patch, orientation in region.boundary:
                if region_patch == patch:
                    slot = 0 if orientation > 0 else 1
                    identifier = domain.entity_id(3, region_row)
                    if result[slot] not in (None, identifier):
                        raise ValueError(
                            "A source patch has conflicting material-side incidence."
                        )
                    result[slot] = identifier
        return result[0], result[1]

    def require_sides(
        side: _RegionSides, possibilities: Iterable[_RegionSides]
    ) -> _RegionSides:
        if side != (None, None) and not any(
            all(
                value is None or value == allowed[index]
                for index, value in enumerate(side)
            )
            for allowed in possibilities
        ):
            raise ValueError("Authored wall metadata changes its source material sides.")
        return side

    def merge_sides(first: _RegionSides, second: _RegionSides) -> _RegionSides:
        result = []
        for left, right in zip(first, second, strict=True):
            if left is not None and right is not None and left != right:
                raise ValueError(
                    "Wall source rows have conflicting material-side metadata."
                )
            result.append(left if left is not None else right)
        return result[0], result[1]

    def root_face_side(root_cell: int, patch: int) -> _RegionSides:
        source_row = root_face_rows.get(int(root_cell))
        if source_row is None:
            raise ValueError("An authored wall face lacks its source association.")
        table, row = source_row
        if (
            table.source_entity_ids[row] != domain.entity_id(2, patch)
            or table.source_occurrence_paths[row]
            != tuple(domain.source_occurrences[2][patch])
            or table.source_roles[row] is not GeometrySourceEntityRole.FACET
        ):
            raise ValueError("Authored wall face changes its source patch occurrence.")
        return require_sides(table.source_region_sides[row], (patch_region_sides(patch),))

    chart_rows_by_id = {
        int(identifier): row for row, identifier in enumerate(charts.cell_global_ids)
    }

    def curve_side_possibilities(root_edge: int, curve: int) -> set[_RegionSides]:
        return {
            patch_region_sides(
                int(charts.root_patches[chart_rows_by_id[int(binding.root_cell_id)]])
            )
            for binding in wall.curve_bindings.get(int(root_edge), ())
            if int(binding.curve_row) == int(curve)
        }

    def curve_sides(root_edge: int, curve: int) -> _RegionSides:
        source_row = curve_rows.get(int(root_edge))
        if source_row is None:
            raise ValueError("An authored wall curve lacks its source association.")
        table, row = source_row
        if (
            table.source_entity_ids[row] != domain.entity_id(1, curve)
            or table.source_occurrence_paths[row]
            != tuple(domain.source_occurrences[1][curve])
            or table.source_roles[row] is not GeometrySourceEntityRole.EDGE
        ):
            raise ValueError("Authored wall curve changes its source occurrence.")
        possible = curve_side_possibilities(root_edge, curve)
        return require_sides(table.source_region_sides[row], possible)

    curve_vertex_sides = {}
    curve_vertex_possibilities = {}
    for edge_id, bindings in wall.curve_bindings.items():
        for binding in bindings:
            side = curve_sides(edge_id, binding.curve_row)
            for vertex, parameter in zip(
                wall.edge_vertices[edge_id], binding.parameters, strict=True
            ):
                if parameter in (0, 1):
                    continue
                key = (int(root_ids[int(vertex)]), int(binding.curve_row))
                possible = curve_side_possibilities(edge_id, binding.curve_row)
                previous = curve_vertex_possibilities.get(key)
                curve_vertex_possibilities[key] = (
                    possible if previous is None else previous & possible
                )
                curve_vertex_sides[key] = merge_sides(
                    curve_vertex_sides.get(key, (None, None)), side
                )
                require_sides(curve_vertex_sides[key], curve_vertex_possibilities[key])

    def root_curve(
        root_vertices: tuple[int, ...],
        first_weights: tuple[Fraction, ...],
        second_weights: tuple[Fraction, ...] | None = None,
    ) -> tuple[int, Fraction, Fraction | None, int] | None:
        candidates = set()
        for index in range(len(root_vertices)):
            positions = (index, (index + 1) % len(root_vertices))
            if any(
                first_weights[other] != 0
                or (second_weights is not None and second_weights[other] != 0)
                for other in range(len(root_vertices))
                if other not in positions
            ):
                continue
            edge_id = root_edges.get(
                frozenset(root_vertices[position] for position in positions)
            )
            if edge_id is None:
                continue
            uses = wall.curve_bindings.get(edge_id, ())
            if not uses:
                continue
            canonical = wall.edge_vertices[edge_id]
            slots = tuple(root_vertices.index(int(vertex)) for vertex in canonical)
            for use in uses:
                first = sum(
                    (first_weights[slots[i]] * use.parameters[i] for i in range(2)),
                    Fraction(0),
                )
                last = (
                    None
                    if second_weights is None
                    else sum(
                        (second_weights[slots[i]] * use.parameters[i] for i in range(2)),
                        Fraction(0),
                    )
                )
                candidates.add((int(use.curve_row), first, last, int(edge_id)))
        if len(candidates) > 1:
            raise ValueError(
                "Exact wall curve uses disagree on source occurrence or parameter."
            )
        return next(iter(candidates)) if candidates else None

    def root_support(
        root_cell: int, root_vertices: tuple[int, ...], weights: tuple[Fraction, ...]
    ) -> tuple[int, int]:
        nonzero = {index for index, value in enumerate(weights) if value != 0}
        if len(nonzero) == 1:
            return 0, int(root_ids[root_vertices[next(iter(nonzero))]])
        edges = {
            root_edges[
                frozenset(
                    (
                        root_vertices[index],
                        root_vertices[(index + 1) % len(root_vertices)],
                    )
                )
            ]
            for index in range(len(root_vertices))
            if nonzero <= {index, (index + 1) % len(root_vertices)}
            and frozenset(
                (root_vertices[index], root_vertices[(index + 1) % len(root_vertices)])
            )
            in root_edges
        }
        if len(edges) > 1:
            raise ValueError("A target wall vertex has ambiguous authored edge support.")
        return (1, next(iter(edges))) if edges else (2, int(root_cell))

    def root_edge_support(
        root_cell: int,
        root_vertices: tuple[int, ...],
        first: tuple[Fraction, ...],
        last: tuple[Fraction, ...],
    ) -> tuple[int, int]:
        nonzero = {index for index, value in enumerate(first) if value != 0}
        nonzero.update(index for index, value in enumerate(last) if value != 0)
        edges = {
            root_edges[
                frozenset(
                    (
                        root_vertices[index],
                        root_vertices[(index + 1) % len(root_vertices)],
                    )
                )
            ]
            for index in range(len(root_vertices))
            if nonzero <= {index, (index + 1) % len(root_vertices)}
            and frozenset(
                (root_vertices[index], root_vertices[(index + 1) % len(root_vertices)])
            )
            in root_edges
        }
        if len(edges) > 1:
            raise ValueError("A target wall edge has ambiguous authored support.")
        return (1, next(iter(edges))) if edges else (2, int(root_cell))

    vertices = {}
    vertex_sides = {}
    vertex_possibilities = {}
    vertex_support = {}
    vertex_representatives = {}
    face_values = {}
    face_row_bank = {}
    for (
        cell_id,
        face_id,
        local_face,
        root_cell,
        chart_row,
        target_ids,
        barycentric,
        uv,
        diagonal,
    ) in rows:
        patch = int(charts.root_patches[chart_row])
        path = tuple(domain.source_occurrences[2][patch])
        face_key = (patch, path)
        previous = face_values.setdefault(int(face_id), face_key)
        if previous != face_key:
            raise ValueError("A wall face has conflicting authored source occurrences.")
        face_row_bank.setdefault(int(face_id), []).append(
            (cell_id, local_face, root_cell, chart_row, target_ids, barycentric, uv)
        )
        root_vertices = root_cells[int(root_cell)]
        for slot, (identifier, weights, parameter_uv) in enumerate(
            zip(target_ids, barycentric, uv, strict=True)
        ):
            representative_key = (int(cell_id), int(local_face), slot)
            previous = vertex_representatives.get(int(identifier))
            if previous is None or representative_key < previous:
                vertex_representatives[int(identifier)] = representative_key
            unit = [index for index, weight in enumerate(weights) if weight == 1]
            original = int(root_ids[root_vertices[unit[0]]]) if len(unit) == 1 else None
            support = root_support(root_cell, root_vertices, weights)
            previous_support = vertex_support.get(int(identifier))
            if previous_support is None or support[0] < previous_support[0]:
                vertex_support[int(identifier)] = support
            elif support[0] == previous_support[0] and support != previous_support:
                raise ValueError("A target wall vertex has conflicting authored support.")
            if original is not None:
                dimension, source_row, exact_parameter = wall.vertex_classes[original]
                candidate_side = parent.source_region_sides[parent_rows[original]]
                candidate_possible = (
                    curve_vertex_possibilities.get((original, source_row), set())
                    if dimension == 1
                    else {patch_region_sides(patch)}
                )
                if dimension == 1:
                    candidate_side = merge_sides(
                        candidate_side,
                        curve_vertex_sides.get((original, source_row), (None, None)),
                    )
            else:
                curve = root_curve(root_vertices, weights)
                if curve is None:
                    dimension, source_row, exact_parameter = 2, patch, tuple(parameter_uv)
                    candidate_side = root_face_side(root_cell, patch)
                    candidate_possible = {patch_region_sides(patch)}
                else:
                    dimension, source_row, exact_parameter = 1, curve[0], curve[1]
                    candidate_side = curve_sides(curve[3], curve[0])
                    candidate_possible = curve_side_possibilities(curve[3], curve[0])
            candidate = (int(dimension), int(source_row), exact_parameter, original)
            old = vertices.get(int(identifier))
            if old is None:
                vertices[int(identifier)] = candidate
                vertex_sides[int(identifier)] = candidate_side
                vertex_possibilities[int(identifier)] = candidate_possible
            elif candidate[0] < old[0]:
                if not contains(old, candidate):
                    raise ValueError(
                        "A wall vertex source class is outside its incident source closure."
                    )
                vertices[int(identifier)] = candidate
                vertex_sides[int(identifier)] = candidate_side
                vertex_possibilities[int(identifier)] = candidate_possible
            elif candidate[0] == old[0]:
                if candidate[:3] != old[:3]:
                    raise ValueError(
                        "A target wall vertex has conflicting exact source classes or parameters."
                    )
                vertex_sides[int(identifier)] = merge_sides(
                    vertex_sides[int(identifier)], candidate_side
                )
                if candidate[0] != 0:
                    vertex_possibilities[int(identifier)] &= candidate_possible
                    require_sides(
                        vertex_sides[int(identifier)],
                        vertex_possibilities[int(identifier)],
                    )
                if old[3] is not None and original is not None and old[3] != original:
                    raise ValueError(
                        "A target wall vertex welds distinct authored source vertices."
                    )
                if old[3] is None and original is not None:
                    vertices[int(identifier)] = candidate
            elif not contains(candidate, old):
                raise ValueError(
                    "A wall vertex source class is outside its incident source closure."
                )
    if not vertices:
        raise ValueError("No target vertices retain the authored wall.")
    ids = np.asarray(sorted(vertices), dtype=np.int64)
    selected = [vertices[int(identifier)] for identifier in ids]
    numeric = np.full((len(ids), 2), np.nan, dtype=np.float64)
    residuals = np.empty(len(ids), dtype=np.float64)
    source_roles = []
    region_sides = []
    intervals = []
    parameter_points = []
    orientations = np.zeros(len(ids), dtype=np.int8)
    fresh = []
    for row, (dimension, source_row, parameter, original) in enumerate(selected):
        if original is not None:
            parent_row = parent_rows[original]
            numeric[row] = parent_parameters[parent_row]
            residuals[row] = parent_residuals[parent_row]
            source_roles.append(parent.source_roles[parent_row])
            region_sides.append(parent.source_region_sides[parent_row])
            intervals.append(parent.source_parameter_intervals[parent_row])
            parameter_points.append(parent.source_parameter_points[parent_row])
            orientations[row] = np.asarray(parent.orientations, dtype=np.int8)[parent_row]
        else:
            if dimension == 1:
                curve_parameter = cast(Fraction, parameter)
                numeric[row, 0] = float(
                    domain.curve_native_parameter(source_row, curve_parameter)
                    if parent.association_kind is GeometryAssociationKind.BREP
                    else curve_parameter
                )
            elif dimension == 2:
                numeric[row] = tuple(
                    float(value) for value in cast(tuple[Fraction, ...], parameter)
                )
            else:
                raise ValueError("A created wall vertex lacks original corner ancestry.")
            source_roles.append(roles[dimension])
            region_sides.append(vertex_sides[int(ids[row])])
            intervals.append(None)
            parameter_points.append(None)
            fresh.append(row)
    if fresh:
        fresh_points = [
            representative(cell, face, (slot,))
            for row in fresh
            for cell, face, slot in (vertex_representatives[int(ids[row])],)
        ]
        distances = _class_residuals(
            domain,
            fresh_points,
            [(selected[row][0], selected[row][1]) for row in fresh],
            query_cache,
            charts.boundary_source.resolution,
            wall.projection,
        )
        residuals[np.asarray(fresh, dtype=np.int64)] = distances
    policy = source.source_wall_association_policy
    if policy is None and source.reference_source is not None:
        policy = source.reference_source.source_wall_association_policy
    if (
        policy is not None
        and policy.maximum_residual is not None
        and np.any(residuals > policy.maximum_residual)
    ):
        raise ValueError("Wall source residual exceeds its authored association policy.")
    vertex_table = GeometryAssociation(
        parent.association_kind,
        parent.source_id,
        parent.source_revision,
        mesh.entity_set(0).entity_set_id,
        ids,
        tuple(
            domain.entity_id(dimension, source_row)
            for dimension, source_row, _, _ in selected
        ),
        residuals,
        resolved=np.ones(len(ids), dtype=np.bool_),
        ambiguous=np.zeros(len(ids), dtype=np.bool_),
        exact=parent.exact and bool(np.all(residuals == 0)),
        source_dimensions=np.asarray([item[0] for item in selected], dtype=np.int8),
        source_indices=np.asarray(
            [domain.source_indices[item[0]][item[1]] for item in selected], dtype=np.int64
        ),
        source_roles=tuple(source_roles),
        source_occurrence_paths=tuple(
            tuple(domain.source_occurrences[item[0]][item[1]]) for item in selected
        ),
        source_region_sides=tuple(region_sides),
        source_parameter_intervals=tuple(intervals),
        source_parameter_points=tuple(parameter_points),
        parameters=numeric,
        orientations=orientations,
        parent_dimensions=np.asarray(
            [vertex_support[int(identifier)][0] for identifier in ids], dtype=np.int8
        ),
        parent_ids=np.asarray(
            [vertex_support[int(identifier)][1] for identifier in ids], dtype=np.int64
        ),
        parent_association_id=parent.association_id,
        provenance=GeometryAssociationProvenance.LINEAGE,
    )
    point_rows = [row for row, item in enumerate(selected) if item[0] == 1]
    curve_vertex_table = None
    if point_rows:
        if any(not isinstance(selected[row][2], Fraction) for row in point_rows):
            raise ValueError(
                "Wall curve vertices require exact normalized source parameters."
            )
        curve_vertex_table = GeometryAssociation(
            GeometryAssociationKind.CURVE,
            parent.source_id,
            parent.source_revision,
            mesh.entity_set(0).entity_set_id,
            ids[point_rows],
            tuple(domain.entity_id(1, selected[row][1]) for row in point_rows),
            residuals[point_rows],
            exact=False,
            source_dimensions=np.ones(len(point_rows), dtype=np.int8),
            source_indices=np.asarray(
                [domain.source_indices[1][selected[row][1]] for row in point_rows],
                dtype=np.int64,
            ),
            source_roles=(GeometrySourceEntityRole.EDGE,) * len(point_rows),
            source_occurrence_paths=tuple(
                tuple(domain.source_occurrences[1][selected[row][1]])
                for row in point_rows
            ),
            source_region_sides=tuple(vertex_sides[int(ids[row])] for row in point_rows),
            source_parameter_points=tuple(
                cast(Fraction, selected[row][2]) for row in point_rows
            ),
            parameters=np.asarray(
                [(float(cast(Fraction, selected[row][2])), np.nan) for row in point_rows]
            ),
            orientations=np.zeros(len(point_rows), dtype=np.int8),
            parent_dimensions=np.asarray(
                [vertex_support[int(ids[row])][0] for row in point_rows], dtype=np.int8
            ),
            parent_ids=np.asarray(
                [vertex_support[int(ids[row])][1] for row in point_rows], dtype=np.int64
            ),
            provenance=GeometryAssociationProvenance.CLASSIFICATION,
        )
    target_edges = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(
            entity_keys(mesh, 1), mesh.entity_set(1).entity_ids, strict=True
        )
    }
    edge_values = {}
    edge_side_values = {}
    edge_possibilities = {}
    edge_representatives = {}
    edge_support = {}
    for (
        cell_id,
        face_id,
        local_face,
        root_cell,
        chart_row,
        target_ids,
        barycentric,
        uv,
        diagonal,
    ) in rows:
        patch = int(charts.root_patches[chart_row])
        root_vertices = root_cells[int(root_cell)]
        for slot in range(len(target_ids)):
            other = (slot + 1) % len(target_ids)
            first, last = int(target_ids[slot]), int(target_ids[other])
            key = tuple(sorted((first, last)))
            edge_id = target_edges.get(key)
            if edge_id is None:
                raise ValueError(
                    "A target wall face edge has no canonical mesh identity."
                )
            representative_key = (int(cell_id), int(local_face), slot, other)
            if (
                edge_id not in edge_representatives
                or representative_key < edge_representatives[edge_id]
            ):
                edge_representatives[edge_id] = representative_key
            weights0, weights1 = barycentric[slot], barycentric[other]
            if first != key[0]:
                weights0, weights1 = weights1, weights0
            support = root_edge_support(root_cell, root_vertices, weights0, weights1)
            previous_support = edge_support.get(edge_id)
            if previous_support is None or support[0] < previous_support[0]:
                edge_support[edge_id] = support
            elif support[0] == previous_support[0] and support != previous_support:
                raise ValueError("A target wall edge has conflicting authored support.")
            curve = root_curve(root_vertices, weights0, weights1)
            midpoint = source_uv(
                cell_id, face_id, reference_position(cell_id, local_face, (slot, other))
            )
            if curve is None:
                candidate = (2, patch, None, None, midpoint, None)
                candidate_side = root_face_side(root_cell, patch)
                candidate_possible = {patch_region_sides(patch)}
            else:
                curve_row, t0, t1, old_edge = curve
                if t0 == t1:
                    raise ValueError("A target source-curve edge has zero exact extent.")
                candidate = (1, curve_row, t0, t1, None, old_edge)
                candidate_side = curve_sides(old_edge, curve_row)
                candidate_possible = curve_side_possibilities(old_edge, curve_row)
            old = edge_values.get(edge_id)
            if old is None:
                edge_values[edge_id] = candidate
                edge_side_values[edge_id] = candidate_side
                edge_possibilities[edge_id] = candidate_possible
            elif candidate[0] < old[0]:
                if not contains(old, candidate):
                    raise ValueError(
                        "A wall edge source class is outside its incident source closure."
                    )
                edge_values[edge_id] = candidate
                edge_side_values[edge_id] = candidate_side
                edge_possibilities[edge_id] = candidate_possible
            elif candidate[0] == old[0] and candidate[:5] != old[:5]:
                raise ValueError(
                    "A target wall edge has conflicting authored source classes."
                )
            elif candidate[0] == old[0]:
                edge_side_values[edge_id] = merge_sides(
                    edge_side_values[edge_id], candidate_side
                )
                edge_possibilities[edge_id] &= candidate_possible
                require_sides(edge_side_values[edge_id], edge_possibilities[edge_id])
            elif candidate[0] > old[0] and not contains(candidate, old):
                raise ValueError(
                    "A wall edge source class is outside its incident source closure."
                )

    records = [vertex_table]
    if curve_vertex_table is not None:
        records.append(curve_vertex_table)
    for dimension in (1, 2):
        if dimension == 1:
            for source_dimension in (1, 2):
                selected_edges = [
                    (identifier, value)
                    for identifier, value in sorted(edge_values.items())
                    if value[0] == source_dimension
                ]
                if not selected_edges:
                    continue
                edge_ids = np.asarray(
                    [item[0] for item in selected_edges], dtype=np.int64
                )
                edge_classes = [(item[1][0], item[1][1]) for item in selected_edges]
                edge_points = [
                    representative(cell, face, (first, last))
                    for identifier, _ in selected_edges
                    for cell, face, first, last in (edge_representatives[identifier],)
                ]
                edge_residuals = _class_residuals(
                    domain,
                    edge_points,
                    edge_classes,
                    query_cache,
                    charts.boundary_source.resolution,
                    wall.projection,
                )
                if (
                    policy is not None
                    and policy.maximum_residual is not None
                    and np.any(edge_residuals > policy.maximum_residual)
                ):
                    raise ValueError(
                        "Wall edge residual exceeds its authored association policy."
                    )
                signs, edge_intervals, edge_sides, edge_parameters = [], [], [], []
                for edge_id, (
                    source_dim,
                    source_row,
                    first,
                    last,
                    midpoint,
                    old_edge,
                ) in selected_edges:
                    if source_dim == 1:
                        edge_sides.append(edge_side_values[edge_id])
                        first = cast(Fraction, first)
                        last = cast(Fraction, last)
                        edge_intervals.append((min(first, last), max(first, last)))
                        owner_patch, owner_loop, owner_position = (
                            int(v) for v in domain.curve_owners[source_row]
                        )
                        owner = domain.patches[owner_patch].loops[owner_loop][
                            owner_position
                        ]
                        gauge = (
                            (1 if owner.last > owner.first else -1)
                            if parent.association_kind is GeometryAssociationKind.BREP
                            else 1
                        )
                        signs.append((1 if last > first else -1) * gauge)
                        edge_parameters.append((np.nan, np.nan))
                    else:
                        edge_sides.append(edge_side_values[edge_id])
                        edge_intervals.append(None)
                        signs.append(0)
                        edge_parameters.append(
                            tuple(
                                float(value)
                                for value in cast(
                                    tuple[ExactScalar, ExactScalar], midpoint
                                )
                            )
                        )
                records.append(
                    GeometryAssociation(
                        GeometryAssociationKind.CURVE
                        if source_dimension == 1
                        else parent.association_kind,
                        parent.source_id,
                        parent.source_revision,
                        mesh.entity_set(1).entity_set_id,
                        edge_ids,
                        tuple(domain.entity_id(dim, row) for dim, row in edge_classes),
                        edge_residuals,
                        exact=False,
                        source_dimensions=np.asarray(
                            [item[0] for item in edge_classes], dtype=np.int8
                        ),
                        source_indices=np.asarray(
                            [
                                domain.source_indices[dim][row]
                                for dim, row in edge_classes
                            ],
                            dtype=np.int64,
                        ),
                        source_roles=tuple(roles[dim] for dim, _ in edge_classes),
                        source_occurrence_paths=tuple(
                            tuple(domain.source_occurrences[dim][row])
                            for dim, row in edge_classes
                        ),
                        source_region_sides=tuple(edge_sides),
                        source_parameter_intervals=tuple(edge_intervals),
                        parameters=np.asarray(edge_parameters, dtype=np.float64),
                        orientations=np.asarray(signs, dtype=np.int8),
                        parent_dimensions=np.asarray(
                            [edge_support[int(identifier)][0] for identifier in edge_ids],
                            dtype=np.int8,
                        ),
                        parent_ids=np.asarray(
                            [edge_support[int(identifier)][1] for identifier in edge_ids],
                            dtype=np.int64,
                        ),
                        provenance=GeometryAssociationProvenance.CLASSIFICATION,
                    )
                )
        else:
            face_ids = np.asarray(sorted(face_values), dtype=np.int64)
            face_classes = [
                (2, face_values[int(identifier)][0]) for identifier in face_ids
            ]
            witness_pieces = source_boundary_association_piece_rows(witness)
            face_support = []
            face_parents = []
            face_sides, face_parameters, face_orientations, face_points = [], [], [], []
            for identifier in face_ids:
                entries = face_row_bank[int(identifier)]
                first = entries[0]
                parent_faces = {int(entry[2]) for entry in entries}
                if len(parent_faces) != 1:
                    raise ValueError(
                        "A target wall face has conflicting authored face support."
                    )
                face_parents.append(next(iter(parent_faces)))
                target_ids = first[4]
                patch = face_values[int(identifier)][0]
                support_key = (
                    int(identifier),
                    domain.entity_id(2, patch),
                    tuple(domain.source_occurrences[2][patch]),
                )
                if support_key not in witness_pieces:
                    raise ValueError(
                        "A wall face lacks its occurrence-qualified exact target support."
                    )
                face_support.append(witness_pieces[support_key])
                side = (None, None)
                for entry in entries:
                    side = merge_sides(side, root_face_side(int(entry[2]), patch))
                face_sides.append(side)
                center = reference_position(
                    first[0], first[1], tuple(range(len(target_ids)))
                )
                face_parameters.append(
                    tuple(
                        float(value)
                        for value in source_uv(first[0], int(identifier), center)
                    )
                )
                face_points.append(
                    representative(first[0], first[1], tuple(range(len(target_ids))))
                )
                direction = tuple(int(value) for value in target_ids)

                def cycle(values: tuple[int, ...]) -> tuple[int, ...]:
                    start = values.index(min(values))
                    return values[start:] + values[:start]

                sign = 1 if cycle(direction) <= cycle(tuple(reversed(direction))) else -1
                orientations = {
                    int(witness.orientations[index]) * sign
                    for index, (cell, face) in enumerate(
                        zip(
                            witness.cell_global_ids,
                            witness.entity_global_ids,
                            strict=True,
                        )
                    )
                    if int(cell) == int(first[0]) and int(face) == int(identifier)
                }
                if len(orientations) != 1:
                    raise ValueError(
                        "Target wall face charts disagree on source orientation."
                    )
                face_orientations.append(next(iter(orientations)))
            face_residuals = _class_residuals(
                domain,
                face_points,
                face_classes,
                query_cache,
                charts.boundary_source.resolution,
                wall.projection,
            )
            if (
                policy is not None
                and policy.maximum_residual is not None
                and np.any(face_residuals > policy.maximum_residual)
            ):
                raise ValueError(
                    "Wall face residual exceeds its authored association policy."
                )
            records.append(
                GeometryAssociation(
                    parent.association_kind,
                    parent.source_id,
                    parent.source_revision,
                    mesh.entity_set(2).entity_set_id,
                    face_ids,
                    tuple(domain.entity_id(dim, row) for dim, row in face_classes),
                    face_residuals,
                    exact=False,
                    source_dimensions=np.full(len(face_ids), 2, dtype=np.int8),
                    source_indices=np.asarray(
                        [domain.source_indices[2][row] for _, row in face_classes],
                        dtype=np.int64,
                    ),
                    source_roles=(GeometrySourceEntityRole.FACET,) * len(face_ids),
                    source_occurrence_paths=tuple(
                        tuple(domain.source_occurrences[2][row])
                        for _, row in face_classes
                    ),
                    target_reference_pieces=tuple(face_support),
                    source_boundary_witness_id=witness.witness_id,
                    source_region_sides=tuple(face_sides),
                    parameters=np.asarray(face_parameters, dtype=np.float64),
                    orientations=np.asarray(face_orientations, dtype=np.int8),
                    parent_dimensions=np.full(len(face_ids), 2, dtype=np.int8),
                    parent_ids=np.asarray(face_parents, dtype=np.int64),
                    provenance=GeometryAssociationProvenance.CLASSIFICATION,
                )
            )
    return tuple(records)

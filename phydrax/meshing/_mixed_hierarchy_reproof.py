#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reprove mixed refinement trees against a bounded root-cell partition."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..discretization import CellGeometrySpec, CellMesh, reference_cell_topology
from ..discretization._cell_geometry_transition_types import (
    _geometry_map_expressions,
    _require_exact_reference_group,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import (
    axes,
    compose_expression,
    expression_equal,
)
from ..discretization._nested_reference_types import (
    evaluate_reference_chart_arguments,
    exact_reference_cell_vertices,
    ExactReferenceChartExpression,
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
    ReferenceCellPartition,
    validate_reference_cell_partition,
)
from ._lineage import MeshLineage, VertexInterpolationStencil
from ._mixed_adaptation import (
    _affine_chart,
    _exact_child_to_parent_affine_map,
    _source_root_charts,
    _templates,
    MixedAdaptationHierarchy,
    MixedCellContentTransfer,
)
from ._result import CellMeshingResult
from ._topology_edit import entity_keys


if TYPE_CHECKING:
    from ..discretization._cell_geometry_transition_types import CellGeometryTransition
    from ._adaptation import _AdaptationConstraints, MarkedMeshAdaptation


def _cell_rows(mesh: CellMesh, /) -> dict[int, tuple[str, str, tuple[int, ...]]]:
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = {}
    for block in mesh.blocks:
        for identifier, row in zip(
            np.asarray(block.global_ids, dtype=np.int64),
            np.asarray(block.vertices, dtype=np.int64),
            strict=True,
        ):
            key = int(identifier)
            if key in cells:
                raise ValueError("Mixed hierarchy endpoint repeats a cell ID.")
            cells[key] = (block.name, block.cell_kind, tuple(map(int, vertex_ids[row])))
    return cells


def _root_point(
    kind: str, vertices: tuple[int, ...], point: tuple[Fraction, ...], /
) -> tuple[tuple[int, Fraction], ...]:
    x, y, z = point
    if kind in ("tetrahedron", "simplex:3"):
        weights = (1 - x - y - z, x, y, z)
    elif kind == "prism":
        triangle = (1 - x - y, x, y)
        weights = tuple(value * (1 - z) for value in triangle) + tuple(
            value * z for value in triangle
        )
    elif kind == "hexahedron":
        weights = tuple(
            (x if corner[0] else 1 - x)
            * (y if corner[1] else 1 - y)
            * (z if corner[2] else 1 - z)
            for corner in reference_cell_topology(kind).vertices
        )
    elif kind == "pyramid":
        height = 1 - z
        u, v = (
            ((x - z / 2) / height, (y - z / 2) / height)
            if height
            else (Fraction(1, 2), Fraction(1, 2))
        )
        weights = (
            height * (1 - u) * (1 - v),
            height * u * (1 - v),
            height * u * v,
            height * (1 - u) * v,
            z,
        )
    else:
        raise ValueError("Mixed hierarchy contains an unsupported cell family.")
    if len(weights) != len(vertices) or sum(weights, Fraction(0)) != 1:
        raise ValueError("Root vertex stencil does not form a partition of unity.")
    return tuple(
        sorted(
            (identifier, value)
            for identifier, value in zip(vertices, weights, strict=True)
            if value
        )
    )


def require_rooted_mixed_hierarchy_geometry(
    root: CellMesh,
    target: CellMesh,
    root_partition: ReferenceCellPartition,
    hierarchy: MixedAdaptationHierarchy,
    /,
    *,
    root_classes: dict[int, int],
    root_layers: dict[int, tuple[int, int]],
) -> None:
    """Prove the complete retained parent/leaf graph and exact root stencils.

    The caller must independently bind ``root_partition`` to the actual root and
    target coordinate maps and authenticate the root classes and layer controls.
    Retired keys and IDs are checked against the issued vertex bank and active
    topology; periodic quotient action is separately reproved against the root.
    """
    if not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError("A rooted mixed hierarchy is required.")
    if root.topological_dimension != 3 or target.topological_dimension != 3:
        raise ValueError("Rooted mixed hierarchy endpoints must be volume meshes.")
    validate_reference_cell_partition(root_partition)
    root_cells = _cell_rows(root)
    target_cells = _cell_rows(target)
    forest = {int(record.cell_id): record for record in hierarchy.forest_roots}
    if len(forest) != len(hierarchy.forest_roots) or not forest:
        raise ValueError("Mixed hierarchy needs unique initial forest roots.")
    if set(root_classes) != set(forest) or not set(root_layers) <= set(forest):
        raise ValueError(
            "Root classes and layer controls must bind the initial forest cells."
        )
    grouped_roots = {}
    for record in forest.values():
        root_cell = root_cells.get(record.root_cell_id)
        if (
            root_cell is None
            or root_cell[1] != record.root_cell_kind
            or root_cell[2] != record.root_vertices
        ):
            raise ValueError("Mixed forest root names an absent original root cell.")
        grouped_roots.setdefault((record.cell_kind, record.root_cell_kind), []).append(
            record
        )
    forest_maps = []
    for (kind, root_kind), records in sorted(grouped_roots.items()):
        records.sort(key=lambda item: item.cell_id)
        exact = tuple(record.root_reference_vertices for record in records)
        forest_maps.append(
            ReferenceCellMapWitnesses(
                np.asarray([record.cell_id for record in records], dtype=np.int64),
                np.asarray([record.root_cell_id for record in records], dtype=np.int64),
                kind,
                root_kind,
                np.asarray(exact, dtype=np.float64),
                exact,
                freeze_reference_chart_arguments(
                    tuple(record.root_reference_chart_arguments for record in records)
                ),
            )
        )
    validate_reference_cell_partition(ReferenceCellPartition(tuple(forest_maps)))
    if {record.root_cell_id for record in forest.values()} != set(root_cells):
        raise ValueError("Mixed initial forest does not cover the original root.")
    leaves = {int(cell.cell_id): cell for cell in hierarchy.active_cells}
    siblings = {int(record.parent_id): record for record in hierarchy.siblings}
    if len(leaves) != len(hierarchy.active_cells) or len(siblings) != len(
        hierarchy.siblings
    ):
        raise ValueError("Mixed hierarchy repeats a cell identity.")
    if set(leaves) != set(target_cells) or set(leaves) & set(siblings):
        raise ValueError("Mixed hierarchy leaves must be exactly the target cells.")
    for identifier, cell in leaves.items():
        if target_cells[identifier] != (cell.block_name, cell.cell_kind, cell.vertices):
            raise ValueError(
                "Mixed hierarchy leaf connectivity differs from the target mesh."
            )
    maps = {}
    for group in root_partition.maps:
        exact = exact_reference_cell_vertices(group)
        arguments = group.source_reference_chart_arguments
        if arguments is None:
            raise ValueError("Mixed root ancestry must retain complete exact charts.")
        for target_id, root_id, vertices, chart in zip(
            group.target_cell_ids, group.source_cell_ids, exact, arguments, strict=True
        ):
            key = int(target_id)
            if key in maps:
                raise ValueError("Root partition repeats a target cell.")
            maps[key] = (
                int(root_id),
                group.target_cell_kind,
                group.source_cell_kind,
                vertices,
                chart,
            )
    if set(maps) != set(leaves) or {value[0] for value in maps.values()} != set(
        root_cells
    ):
        raise ValueError("Root partition must cover every actual root and target cell.")
    issued = set(forest)
    parent_of: dict[int, int] = {}
    templates_by_kind = {}
    for parent, record in siblings.items():
        if (
            len(record.child_ids) != len(record.child_reference_vertices)
            or not record.child_ids
        ):
            raise ValueError("Mixed sibling children and reference maps must align.")
        if len(set(record.child_ids)) != len(record.child_ids):
            raise ValueError("Mixed sibling children must be unique.")
        if record.cell_kind not in templates_by_kind:
            templates_by_kind[record.cell_kind] = {
                tuple(
                    tuple(
                        tuple(Fraction(float(value)) for value in point)
                        for point in child
                    )
                    for child in template.children
                )
                for axial in (False, True)
                for template in _templates(record.cell_kind, axial)
                if template.index != 0
            }
        if record.child_reference_vertices not in templates_by_kind[record.cell_kind]:
            raise ValueError(
                "Mixed sibling reference maps are not a declared exact template."
            )
        for child in record.child_ids:
            child = int(child)
            if child <= parent or child in forest or child in parent_of:
                raise ValueError("Mixed hierarchy reuses a root or child cell identity.")
            parent_of[child] = parent
            issued.add(child)
    if set(siblings) | set(leaves) != issued:
        raise ValueError("Mixed hierarchy has disconnected or undeclared cells.")
    if hierarchy.next_cell_id <= max(issued | set(root_cells), default=-1):
        raise ValueError("Mixed hierarchy cell allocator would reuse an issued ID.")
    root_points: dict[int, tuple[tuple[int, Fraction], ...]] = {}
    reached: set[int] = set()
    active: set[int] = set()

    def visit(
        identifier: int,
        root_id: int,
        name: str,
        kind: str,
        vertices: tuple[int, ...],
        positions: tuple[tuple[Fraction, ...], ...],
        root_chart: tuple[ExactReferenceChartExpression, ...],
        cell_class: int,
        layer: tuple[int, int] | None,
    ) -> None:
        if identifier in active or identifier in reached:
            raise ValueError("Mixed hierarchy contains a cycle or a repeated child.")
        active.add(identifier)
        reached.add(identifier)
        for vertex, point in zip(vertices, positions, strict=True):
            key = _root_point(root_cells[root_id][1], root_cells[root_id][2], point)
            previous = root_points.setdefault(vertex, key)
            if previous != key:
                raise ValueError(
                    "A mixed hierarchy vertex has conflicting exact root stencils."
                )
        if identifier in leaves:
            cell = leaves[identifier]
            expected_layer = (
                (cell.column_id, cell.interval_index)
                if cell.column_id is not None
                else None
            )
            if (
                cell.block_name,
                cell.cell_kind,
                cell.vertices,
                cell.cell_class,
                expected_layer,
            ) != (name, kind, vertices, cell_class, layer):
                raise ValueError("Mixed leaf changes its parent class or layer ancestry.")
            retained = maps[identifier]
            if retained[:4] != (
                root_id,
                kind,
                root_cells[root_id][1],
                positions,
            ) or not all(
                expression_equal(first, second)
                for first, second in zip(retained[4], root_chart, strict=True)
            ):
                raise ValueError(
                    "Mixed leaf chart differs from its exact root partition."
                )
        else:
            record = siblings.get(identifier)
            if record is None:
                raise ValueError("Mixed hierarchy graph omits an active descendant.")
            record_layer = (
                (record.column_id, record.interval_index)
                if record.column_id is not None
                else None
            )
            if (
                record.block_name,
                record.cell_kind,
                record.parent_vertices,
                record.cell_class,
                record_layer,
            ) != (name, kind, vertices, cell_class, layer):
                raise ValueError("Mixed sibling changes its exact parent ancestry.")
            axial = kind in ("prism", "hexahedron") and any(
                max(point[2] for point in child) - min(point[2] for point in child) < 1
                for child in record.child_reference_vertices
            )
            for child_id, chart in zip(
                record.child_ids, record.child_reference_vertices, strict=True
            ):
                child_id = int(child_id)
                child = leaves.get(child_id)
                nested = siblings.get(child_id)
                if child is not None:
                    child_vertices = child.vertices
                elif nested is not None:
                    child_vertices = nested.parent_vertices
                else:
                    raise ValueError("Mixed hierarchy omits a declared child.")
                local_chart = _affine_chart(
                    *_exact_child_to_parent_affine_map(kind, chart)
                )
                child_root_chart = tuple(
                    compose_expression(expression, local_chart)
                    for expression in root_chart
                )
                child_positions = tuple(
                    evaluate_reference_chart_arguments(
                        kind,
                        child_root_chart,
                        tuple(Fraction(float(value)) for value in point),
                    )
                    for point in reference_cell_topology(kind).vertices
                )
                if any(
                    not isinstance(value, Fraction)
                    for point in child_positions
                    for value in point
                ):
                    raise ValueError(
                        "Mixed root reference coordinates must be exact rationals."
                    )
                rational_positions = tuple(
                    tuple(value for value in point if isinstance(value, Fraction))
                    for point in child_positions
                )
                child_layer = layer
                if layer is not None and axial:
                    midpoint = sum((point[2] for point in chart), Fraction(0)) / len(
                        chart
                    )
                    child_layer = (
                        layer[0],
                        2 * layer[1] + int(midpoint >= Fraction(1, 2)),
                    )
                visit(
                    child_id,
                    root_id,
                    name,
                    kind,
                    child_vertices,
                    rational_positions,
                    child_root_chart,
                    cell_class,
                    child_layer,
                )
        active.remove(identifier)

    for identifier, record in forest.items():
        layer = None
        if record.column_id is not None:
            if record.interval_index is None:
                raise ValueError("Mixed forest root has incomplete layer ancestry.")
            layer = (record.column_id, record.interval_index)
        if record.cell_class != root_classes[identifier] or layer != root_layers.get(
            identifier
        ):
            raise ValueError(
                "Mixed forest root changes its accepted class or layer controls."
            )
        visit(
            identifier,
            record.root_cell_id,
            record.block_name,
            record.cell_kind,
            record.vertices,
            record.root_reference_vertices,
            record.root_reference_chart_arguments,
            root_classes[identifier],
            layer,
        )
    if reached != issued:
        raise ValueError("Mixed hierarchy has unreachable parent or leaf cells.")
    target_vertices = set(map(int, np.asarray(target.vertex_global_ids, dtype=np.int64)))
    actual_points = dict(hierarchy.vertex_points)
    if len(actual_points) != len(hierarchy.vertex_points):
        raise ValueError("Mixed hierarchy repeats a vertex identity.")
    root_vertex_ids = set(map(int, np.asarray(root.vertex_global_ids, dtype=np.int64)))
    roots_by_vertex: dict[int, list[tuple[str, tuple[int, ...]]]] = {}
    for _, kind, vertices in root_cells.values():
        for vertex in vertices:
            roots_by_vertex.setdefault(vertex, []).append((kind, vertices))
    for identifier, key in hierarchy.vertex_points:
        if (
            identifier < 0
            or not key
            or tuple(sorted(key)) != key
            or len({vertex for vertex, _ in key}) != len(key)
            or any(
                vertex not in root_vertex_ids
                or not isinstance(weight, Fraction)
                or weight <= 0
                for vertex, weight in key
            )
            or sum((weight for _, weight in key), Fraction(0)) != 1
        ):
            raise ValueError("Mixed hierarchy has a malformed root vertex stencil.")
        support = dict(key)
        for kind, vertices in roots_by_vertex.get(key[0][0], ()):
            if not set(support).issubset(vertices):
                continue
            corners = reference_cell_topology(kind).vertices
            reference = tuple(
                sum(
                    (
                        support.get(vertex, Fraction(0)) * Fraction(float(corner[axis]))
                        for vertex, corner in zip(vertices, corners, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            if _root_point(kind, vertices, reference) == key:
                break
        else:
            raise ValueError(
                "Mixed retired vertex has no exact supporting root-cell chart."
            )
    if not target_vertices <= set(actual_points) or any(
        actual_points[vertex] != root_points[vertex] for vertex in target_vertices
    ):
        raise ValueError(
            "Mixed hierarchy active vertex stencils differ from exact root ancestry."
        )
    if hierarchy.next_vertex_id <= max(set(actual_points) | root_vertex_ids, default=-1):
        raise ValueError("Mixed hierarchy vertex allocator would reuse an issued ID.")
    for degree in (1, 2):
        keys = entity_keys(target, degree)
        ids = np.asarray(target.entity_set(degree).entity_ids, dtype=np.int64)
        current = {
            tuple(int(vertex) for vertex in key if vertex >= 0): int(identifier)
            for key, identifier in zip(keys, ids, strict=True)
        }
        retired = [
            (key, identifier)
            for d, key, identifier in hierarchy.retired_entities
            if d == degree
        ]
        if len({key for key, _ in retired}) != len(retired) or len(
            {identifier for _, identifier in retired}
        ) != len(retired):
            raise ValueError("Mixed hierarchy reuses a retired entity key or ID.")
        if any(
            len(key) < (2 if degree == 1 else 3)
            or tuple(sorted(set(key))) != key
            or not set(key) <= set(actual_points)
            or identifier < 0
            or key in current
            or identifier in current.values()
            for key, identifier in retired
        ):
            raise ValueError(
                "Mixed hierarchy retired entities conflict with retained topology."
            )
        issued_by_id = {identifier: key for key, identifier in current.items()}
        issued_by_id.update({identifier: key for key, identifier in retired})
        for key, identifier in zip(
            entity_keys(root, degree),
            np.asarray(root.entity_set(degree).entity_ids, dtype=np.int64),
            strict=True,
        ):
            root_key = tuple(int(vertex) for vertex in key if vertex >= 0)
            current_key = issued_by_id.get(int(identifier))
            if current_key is not None and current_key != root_key:
                raise ValueError("Mixed hierarchy reuses an original root entity ID.")
    if any(degree not in (1, 2) for degree, _, _ in hierarchy.retired_entities):
        raise ValueError("Mixed hierarchy has an unsupported retired entity degree.")
    if hierarchy.active_lifted_topology_id != target.lifted_topology_id:
        raise ValueError("Mixed hierarchy is bound to another target topology.")


def _returned_root_geometry_id(
    source: CellMeshingResult, hierarchy: MixedAdaptationHierarchy, /
) -> str:
    """Bind a fully coarsened endpoint to its retained root reference frame."""
    from ..geometry._surface_source_archive import _geometry

    # Reconstruct cached restriction/layout identities before using the retained
    # label. Matching endpoint labels alone do not authenticate root provenance.
    geometry = _geometry(source.geometry, source.mesh)
    restriction = geometry.restriction_source
    if restriction is None:
        raise ValueError("Mixed source hierarchy lost its retained root restriction.")
    charts = _source_root_charts(source.mesh, geometry, hierarchy, source.mesh)
    identity = axes(source.mesh.topological_dimension)
    for identifier, (_, kind, vertices) in _cell_rows(source.mesh).items():
        chart = charts[identifier]
        if (
            chart.root_cell_id != identifier
            or chart.root_cell_kind != kind
            or chart.root_vertices != vertices
            or len(chart.arguments) != len(identity)
            or not all(
                expression_equal(actual, expected)
                for actual, expected in zip(chart.arguments, identity, strict=True)
            )
        ):
            raise ValueError("Mixed source has not returned to its retained root cells.")
    return restriction.source_geometry_id


def _root_partition(
    root: CellMesh,
    root_geometry: CellGeometrySpec,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    ancestral_geometry_id: str | None = None,
) -> ReferenceCellPartition:
    """Reprove the complete current-to-root map against actual coordinate maps."""
    restriction = geometry.restriction_source
    groups = []
    if mesh.mesh_id == root.mesh_id and cell_geometry_id(geometry) == cell_geometry_id(
        root_geometry
    ):
        for block in mesh.blocks:
            corners = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            exact = tuple(corners for _ in range(block.cell_count))
            ids = np.asarray(block.global_ids, dtype=np.int64)
            groups.append(
                ReferenceCellMapWitnesses(
                    ids,
                    ids,
                    block.cell_kind,
                    block.cell_kind,
                    np.asarray(exact, dtype=np.float64),
                    exact,
                    freeze_reference_chart_arguments(
                        tuple(axes(root.topological_dimension) for _ in ids)
                    ),
                )
            )
    else:
        if restriction is None or (
            restriction.source_topology_id != root.topology_id
            or restriction.source_geometry_id
            != (
                cell_geometry_id(root_geometry)
                if ancestral_geometry_id is None
                else ancestral_geometry_id
            )
        ):
            raise ValueError(
                "Mixed geometry does not retain the independently proved root."
            )
        by_name = restriction.block_parent_reference_maps
        if set(by_name) != {block.name for block in mesh.blocks}:
            raise ValueError("Mixed root maps do not cover every current block.")
        for block in mesh.blocks:
            witness = by_name[block.name]
            if witness is None:
                raise ValueError("Mixed root ancestry lacks an exact reference map.")
            rows = np.asarray(witness.target_cell_ids, dtype=np.int64)
            ids = np.asarray(block.global_ids, dtype=np.int64)
            if not np.array_equal(rows, np.arange(ids.size)):
                raise ValueError("Mixed root map is not aligned to its actual block.")
            groups.append(witness._replace(target_cell_ids=ids))
    partition = ReferenceCellPartition(tuple(groups))
    validate_reference_cell_partition(partition)
    root_resolved = root_geometry.resolve(root)
    current_resolved = geometry.resolve(mesh)
    root_maps = _geometry_map_expressions(
        root, root_geometry, (root_resolved[0], root_resolved[1])
    )
    current_maps = _geometry_map_expressions(
        mesh, geometry, (current_resolved[0], current_resolved[1])
    )
    for group in groups:
        _require_exact_reference_group(
            root_maps, current_maps, group, exact_reference_cell_vertices(group)
        )
    return partition


def _require_lineage_and_stencil(
    source: CellMesh,
    target: CellMesh,
    hierarchy: MixedAdaptationHierarchy,
    lineage: MeshLineage,
    stencil: VertexInterpolationStencil,
    /,
) -> None:
    """Bind all lineage rows and exact vertex actions to their actual endpoints."""
    from ._lineage import EntityLineageKind

    if (
        lineage.source_topology_id != source.topology_id
        or lineage.target_topology_id != target.topology_id
    ):
        raise ValueError("Mixed lineage has different topology endpoints.")
    if tuple(record.dimension for record in lineage.entities) != tuple(range(4)):
        raise ValueError("Mixed lineage must cover every topological dimension.")
    for degree, record in enumerate(lineage.entities):
        source_set = source.entity_set(degree)
        target_set = target.entity_set(degree)
        if (record.source_entity_set_id, record.target_entity_set_id) != (
            source_set.entity_set_id,
            target_set.entity_set_id,
        ):
            raise ValueError("Mixed lineage uses another entity set.")
        sources = set(map(int, np.asarray(source_set.entity_ids)))
        targets = set(map(int, np.asarray(target_set.entity_ids)))
        rows = tuple(
            zip(
                map(int, np.asarray(record.source_global_ids)),
                map(int, np.asarray(record.target_global_ids)),
                map(int, np.asarray(record.relation_kinds)),
                strict=True,
            )
        )
        created = tuple(map(int, np.asarray(record.created_target_ids)))
        deleted = tuple(map(int, np.asarray(record.deleted_source_ids)))
        if (
            len(set(rows)) != len(rows)
            or len(set(created)) != len(created)
            or len(set(deleted)) != len(deleted)
            or {row[0] for row in rows} | set(deleted) != sources
            or {row[1] for row in rows} | set(created) != targets
            or {row[0] for row in rows} & set(deleted)
            or {row[1] for row in rows} & set(created)
        ):
            raise ValueError("Mixed lineage does not partition its actual entity sets.")
        source_keys = {
            int(identifier): tuple(int(v) for v in key if v >= 0)
            for identifier, key in zip(
                np.asarray(source_set.entity_ids),
                entity_keys(source, degree),
                strict=True,
            )
        }
        target_keys = {
            int(identifier): tuple(int(v) for v in key if v >= 0)
            for identifier, key in zip(
                np.asarray(target_set.entity_ids),
                entity_keys(target, degree),
                strict=True,
            )
        }
        for first, second, kind in rows:
            if first not in source_keys or second not in target_keys:
                raise ValueError("Mixed lineage references an absent endpoint entity.")
            if (
                kind == int(EntityLineageKind.PRESERVED)
                and source_keys[first] != target_keys[second]
            ):
                raise ValueError("Preserved mixed lineage changes an entity's topology.")
    if (
        stencil.source_entity_set_id != source.entity_set(0).entity_set_id
        or stencil.target_entity_set_id != target.entity_set(0).entity_set_id
    ):
        raise ValueError("Mixed stencil names another vertex set.")
    source_ids = set(map(int, np.asarray(source.vertex_global_ids)))
    target_ids = set(map(int, np.asarray(target.vertex_global_ids)))
    stencil_targets = tuple(map(int, np.asarray(stencil.target_global_ids)))
    if (
        len(set(stencil_targets)) != len(stencil_targets)
        or set(stencil_targets) != target_ids
    ):
        raise ValueError("Mixed stencil must cover every target vertex exactly once.")
    root_points = dict(hierarchy.vertex_points)
    source_rows = np.asarray(stencil.source_global_ids, dtype=np.int64)
    valid_rows = np.asarray(stencil.valid, dtype=np.bool_)
    for row in range(len(stencil_targets)):
        total = Fraction(0)
        for source_id, valid, coefficient in zip(
            source_rows[row], valid_rows[row], stencil.exact_weights[row], strict=True
        ):
            if not valid:
                continue
            source_id = int(source_id)
            if (
                source_id not in source_ids
                or source_id not in root_points
                or coefficient < 0
            ):
                raise ValueError("Mixed stencil references an invalid source vertex.")
            total += coefficient
        if total != 1:
            raise ValueError("Mixed stencil must preserve constants exactly.")


def validate_mixed_restoration(
    *,
    source: CellMeshingResult,
    target: CellMeshingResult,
    geometry_transition: CellGeometryTransition,
    hierarchy: MixedAdaptationHierarchy,
    request: MarkedMeshAdaptation,
    lineage: MeshLineage,
    stencil: VertexInterpolationStencil,
    constraints: _AdaptationConstraints,
) -> MixedCellContentTransfer:
    """Reprove a retained mixed adaptation without executing its request."""
    root_owner = target.layer_core_root_geometry
    source_root = source.layer_core_root_geometry
    source_hierarchy = request.hierarchy
    if source_hierarchy is not None and not isinstance(
        source_hierarchy, MixedAdaptationHierarchy
    ):
        raise TypeError("Mixed restoration requires mixed source hierarchy ancestry.")
    ancestral_geometry_id = None
    if root_owner is None and source_root is None:
        root, root_geometry = source.mesh, source.geometry
        if source_hierarchy is not None:
            ancestral_geometry_id = _returned_root_geometry_id(source, source_hierarchy)
    else:
        if (
            root_owner is None
            or source_root is None
            or root_owner.root_id != source_root.root_id
        ):
            raise ValueError("Mixed endpoints have different retained root authorities.")
        root, root_geometry = root_owner.mesh, root_owner.geometry
    source_partition = _root_partition(root, root_geometry, source.mesh, source.geometry)
    target_partition = _root_partition(
        root,
        root_geometry,
        target.mesh,
        target.geometry,
        ancestral_geometry_id=ancestral_geometry_id,
    )
    root_cells = _cell_rows(root)
    source_cells = _cell_rows(source.mesh)
    source_to_root: dict[int, int] = {}
    source_charts = {}
    bank = dict(hierarchy.vertex_points)
    for group in source_partition.maps:
        if group.source_reference_chart_arguments is None:
            raise ValueError("Mixed source lacks complete exact root charts.")
        for source_id, root_id, corners, chart in zip(
            group.target_cell_ids,
            group.source_cell_ids,
            exact_reference_cell_vertices(group),
            group.source_reference_chart_arguments,
            strict=True,
        ):
            source_id, root_id = int(source_id), int(root_id)
            if (
                source_id in source_to_root
                or root_id not in root_cells
                or source_id not in source_cells
            ):
                raise ValueError("Mixed source root map has invalid cell identities.")
            source_to_root[source_id] = root_id
            source_charts[source_id] = (corners, chart)
            root_kind, root_vertices = root_cells[root_id][1:]
            for vertex, point in zip(source_cells[source_id][2], corners, strict=True):
                if bank.get(vertex) != _root_point(root_kind, root_vertices, point):
                    raise ValueError(
                        "Mixed source vertex bank differs from its root chart."
                    )
    if set(source_to_root) != set(source_cells):
        raise ValueError("Mixed source root map omits cells.")
    ordered_source_ids = [
        int(identifier)
        for block in source.mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    ]
    classes = np.asarray(constraints.cell_classes, dtype=np.int64)
    if classes.shape != (len(ordered_source_ids),):
        raise ValueError("Mixed source material classes do not cover its cells.")
    source_classes = dict(zip(ordered_source_ids, map(int, classes), strict=True))
    forest = {record.cell_id: record for record in hierarchy.forest_roots}
    root_classes = {
        identifier: record.cell_class for identifier, record in forest.items()
    }
    root_layers = {
        identifier: (record.column_id, record.interval_index)
        for identifier, record in forest.items()
        if record.column_id is not None and record.interval_index is not None
    }
    if source_hierarchy is not None:
        if source_hierarchy.forest_roots != hierarchy.forest_roots:
            raise ValueError("Mixed transition changes the retained initial forest.")
        source_active = {
            record.cell_id: record for record in source_hierarchy.active_cells
        }
        if set(source_active) != set(source_classes) or any(
            source_active[identifier].cell_class != cell_class
            for identifier, cell_class in source_classes.items()
        ):
            raise ValueError("Mixed source hierarchy changes certified material classes.")
        require_rooted_mixed_hierarchy_geometry(
            root,
            source.mesh,
            source_partition,
            source_hierarchy,
            root_classes=root_classes,
            root_layers=root_layers,
        )
    else:
        if set(forest) != set(source_cells):
            raise ValueError("Initial mixed forest must be the actual source cells.")
        initial_layers = {}
        if request.layer_columns is not None:
            columns = request.layer_columns
            initial_layers = {
                int(identifier): (int(column), int(interval))
                for identifier, column, interval in zip(
                    np.asarray(columns.cell_ids),
                    np.asarray(columns.column_ids),
                    np.asarray(columns.interval_indices),
                    strict=True,
                )
            }
        if root_classes != source_classes or root_layers != initial_layers:
            raise ValueError(
                "Initial mixed forest changes source classes or layer controls."
            )
        for identifier, record in forest.items():
            corners, chart = source_charts[identifier]
            if (
                source_cells[identifier]
                != (record.block_name, record.cell_kind, record.vertices)
                or source_to_root[identifier] != record.root_cell_id
                or record.root_reference_vertices != corners
                or not all(
                    expression_equal(first, second)
                    for first, second in zip(
                        record.root_reference_chart_arguments, chart, strict=True
                    )
                )
            ):
                raise ValueError(
                    "Initial mixed forest differs from the actual source root chart."
                )
    require_rooted_mixed_hierarchy_geometry(
        root,
        target.mesh,
        target_partition,
        hierarchy,
        root_classes=root_classes,
        root_layers=root_layers,
    )
    geometry_transition.require_bound(
        source.mesh, source.geometry, target.mesh, target.geometry
    )
    _require_lineage_and_stencil(source.mesh, target.mesh, hierarchy, lineage, stencil)
    cell_transfer = _require_immediate_relations(
        source.mesh,
        target.mesh,
        hierarchy,
        source_hierarchy,
        geometry_transition,
        lineage,
        stencil,
    )
    if (source.mesh.periodic_topology is None) != (
        hierarchy.periodic_quotient_history is None
    ):
        raise ValueError("Mixed periodic ancestry is missing or unexpected.")
    if hierarchy.periodic_quotient_history is not None:
        from ._periodic import reprove_periodic_quotient_history

        reprove_periodic_quotient_history(
            root,
            source.mesh,
            hierarchy.periodic_quotient_history,
            hierarchy.vertex_points,
            root_geometry,
        )
    return cell_transfer


def _require_immediate_relations(
    source: CellMesh,
    target: CellMesh,
    hierarchy: MixedAdaptationHierarchy,
    prior: MixedAdaptationHierarchy | None,
    geometry: CellGeometryTransition,
    lineage: MeshLineage,
    stencil: VertexInterpolationStencil,
    /,
) -> MixedCellContentTransfer:
    """Derive exact entity lineage from retained maps, not the original request."""
    from ._mixed_adaptation import (
        _cell_transfer,
        _entity_relations,
        _TargetCell,
        MixedCoarseningMap,
    )
    from ._topology_edit import _entity_lineage

    target_cells = _cell_rows(target)
    source_cells = _cell_rows(source)
    stencils = {}
    for row, identifier in enumerate(
        np.asarray(stencil.target_global_ids, dtype=np.int64)
    ):
        stencils[int(identifier)] = tuple(
            sorted(
                (int(source_id), coefficient)
                for source_id, valid, coefficient in zip(
                    np.asarray(stencil.source_global_ids)[row],
                    np.asarray(stencil.valid)[row],
                    stencil.exact_weights[row],
                    strict=True,
                )
                if valid and coefficient
            )
        )
    active = {cell.cell_id: cell for cell in hierarchy.active_cells}
    sibling_by_parent = {record.parent_id: record for record in hierarchy.siblings}
    prior_siblings = (
        {} if prior is None else {record.parent_id: record for record in prior.siblings}
    )
    direct = geometry.reference_map
    groups = (
        ()
        if direct is None
        else (
            (direct,) if isinstance(direct, ReferenceCellMapWitnesses) else tuple(direct)
        )
    )
    target_sources: dict[int, tuple[int, ...]] = {}
    target_charts = {}
    coarsening_maps = []
    for group in groups:
        if group.source_reference_chart_arguments is None:
            raise ValueError("Mixed immediate maps require complete exact charts.")
        for target_id, source_id, corners, chart in zip(
            group.target_cell_ids,
            group.source_cell_ids,
            exact_reference_cell_vertices(group),
            group.source_reference_chart_arguments,
            strict=True,
        ):
            target_id, source_id = int(target_id), int(source_id)
            if target_id in target_sources or target_id not in target_cells:
                raise ValueError("Mixed immediate maps repeat or invent target cells.")
            if target_id == source_id:
                identity = tuple(
                    tuple(Fraction(float(v)) for v in p)
                    for p in reference_cell_topology(group.target_cell_kind).vertices
                )
                if corners != identity:
                    raise ValueError(
                        "A preserved mixed cell must retain its identity chart."
                    )
            else:
                record = sibling_by_parent.get(source_id)
                if record is None or target_id not in record.child_ids:
                    raise ValueError(
                        "Mixed refinement map lacks its exact retained sibling."
                    )
                index = record.child_ids.index(target_id)
                if corners != record.child_reference_vertices[index]:
                    raise ValueError(
                        "Mixed refinement map differs from the sibling partition."
                    )
            expected_chart = _affine_chart(
                *_exact_child_to_parent_affine_map(group.target_cell_kind, corners)
            )
            if not all(
                expression_equal(first, second)
                for first, second in zip(chart, expected_chart, strict=True)
            ):
                raise ValueError("Mixed immediate chart differs from its exact template.")
            if source_id not in source_cells:
                raise ValueError("Mixed immediate map names an absent source cell.")
            source_kind, source_vertices = source_cells[source_id][1:]
            for vertex, point in zip(target_cells[target_id][2], corners, strict=True):
                if stencils[vertex] != _root_point(source_kind, source_vertices, point):
                    raise ValueError(
                        "Mixed vertex stencil differs from its exact immediate chart."
                    )
            target_sources[target_id] = (source_id,)
            target_charts[target_id] = corners
    coarsen_sources: dict[int, tuple[int, ...]] = {}
    partition = geometry.coarsening_partition
    if partition is not None:
        validate_reference_cell_partition(partition)
        found: dict[int, list[tuple[int, tuple[tuple[Fraction, ...], ...]]]] = {}
        for group in partition.maps:
            if group.source_reference_chart_arguments is None:
                raise ValueError("Mixed coarsening requires complete exact charts.")
            for source_id, target_id, corners, chart in zip(
                group.target_cell_ids,
                group.source_cell_ids,
                exact_reference_cell_vertices(group),
                group.source_reference_chart_arguments,
                strict=True,
            ):
                expected_chart = _affine_chart(
                    *_exact_child_to_parent_affine_map(group.target_cell_kind, corners)
                )
                if not all(
                    expression_equal(first, second)
                    for first, second in zip(chart, expected_chart, strict=True)
                ):
                    raise ValueError(
                        "Mixed coarsening chart differs from its exact template."
                    )
                found.setdefault(int(target_id), []).append((int(source_id), corners))
        for target_id, rows in found.items():
            record = prior_siblings.get(target_id)
            if (
                record is None
                or target_id in target_sources
                or target_id not in target_cells
            ):
                raise ValueError("Mixed coarsening lacks its retained source sibling.")
            actual = {identifier: corners for identifier, corners in rows}
            expected = dict(
                zip(record.child_ids, record.child_reference_vertices, strict=True)
            )
            if len(actual) != len(rows) or actual != expected:
                raise ValueError(
                    "Mixed coarsening does not restore the complete sibling partition."
                )
            for vertex in target_cells[target_id][2]:
                if stencils[vertex] != ((vertex, Fraction(1)),):
                    raise ValueError("Mixed coarsening changes a restored parent vertex.")
            coarsen_sources[target_id] = tuple(record.child_ids)
            target_sources[target_id] = tuple(record.child_ids)
            target_charts[target_id] = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(record.cell_kind).vertices
            )
            coarsening_maps.append(
                MixedCoarseningMap(
                    target_id,
                    record.block_name,
                    record.cell_kind,
                    np.asarray(record.child_ids, dtype=np.int64),
                    record.child_reference_vertices,
                )
            )
    if set(target_sources) != set(target_cells):
        raise ValueError("Mixed immediate maps do not cover every target cell.")
    targets = []
    for identifier in sorted(target_cells):
        cell = active[identifier]
        sources = target_sources[identifier]
        if identifier in coarsen_sources or sources == (identifier,):
            weights = tuple(Fraction(1) for _ in sources)
        else:
            weights = (Fraction(1, len(sibling_by_parent[sources[0]].child_ids)),)
        targets.append(
            _TargetCell(
                identifier,
                cell.block_name,
                cell.cell_kind,
                cell.vertices,
                cell.cell_class,
                cell.column_id,
                cell.interval_index,
                sources,
                weights,
                target_charts[identifier],
            )
        )
    relations = _entity_relations(
        source, target, targets, coarsen_sources, tuple(coarsening_maps)
    )
    for expected, actual in zip(relations, lineage.entities, strict=True):
        rebuilt = _entity_lineage(source, target, expected)
        if rebuilt.lineage_id != actual.lineage_id:
            raise ValueError(
                "Stored mixed lineage differs from exact endpoint containment."
            )
    return _cell_transfer(np.asarray(sorted(_cell_rows(source)), dtype=np.int64), targets)

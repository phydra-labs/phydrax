#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-line and partition proofs for layer/core stratum replay."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import cast, TYPE_CHECKING

import numpy as np

from ..discretization import CellMesh
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._coordinate_enclosure import (
    add,
    compose_expression,
    constant,
    evaluate_expression,
    Expression,
    expression_equal,
    scale,
)
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
)
from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
from ._association import (
    _LayerImageSupportPreparation,
    _source_boundary_image_face_support,
    GeometryAssociation,
    GeometryAssociationKind,
    GeometryAssociationProvenance,
    GeometrySourceEntityRole,
)
from ._association_routes import incidence_pairs
from ._contracts import VolumeMeshingSpec
from ._layer_core_association import _core_source_identity
from ._layer_core_carrier_witness import NativeLayerCoreCarrierWitness
from ._layer_core_topology import prepare_identity
from ._layer_source_association_replay import _maps, LayerAssociationReplay
from ._topology_edit import entity_keys
from .providers._native_sources import NativeLayerCoreSource


if TYPE_CHECKING:
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry


def mapped_source_edge_bindings(
    source: NativeLayerCoreSource, /
) -> dict[int, tuple[int, int]]:
    """Bind core edges to full-PLC edge rows by declared root vertex identities.

    The returned sign relates core edge endpoint order to the full-PLC row.
    This establishes endpoint ancestry only: callers must still prove source
    facet incidence, complete edge image, and interval coverage.
    """
    query = source.fidelity_source
    domain = source.mapped_domain
    reference = source.reference_source
    if (
        not isinstance(query, MappedReferenceBoundarySource)
        or domain is None
        or reference is None
    ):
        raise ValueError("Mapped core edge binding requires its full source authority.")
    query.require_current()
    relation = query.authority.root_relation
    roots = domain.reference_domain_vertex_root_ids
    if relation is None or roots is None:
        raise ValueError(
            "Mapped core edge binding requires declared root vertex ancestry."
        )
    mapping, _, _, _ = prepare_identity(
        reference.layers,
        reference.complex,
        reference.vertex_layer_ids,
        reference.cap_polygon_ids,
    )
    if np.any(mapping < 0) or np.any(mapping >= len(roots)):
        raise ValueError("Core PLC vertices do not bind the mapped reference domain.")
    core_roots = np.asarray(roots, dtype=np.int64)[mapping]
    full_roots = np.asarray(relation.vertex_root_global_ids, dtype=np.int64)
    full_edges = np.asarray(query.authority.complex.plc_edge_vertices, dtype=np.int64)
    if full_edges.ndim != 2 or full_edges.shape[1] != 2:
        raise ValueError("Full source edge connectivity must have two endpoints per row.")
    if np.any(full_edges < 0) or np.any(full_edges >= len(full_roots)):
        raise ValueError("A full source edge names an absent root vertex.")
    by_roots: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for row, edge in enumerate(full_edges):
        endpoints = (int(full_roots[edge[0]]), int(full_roots[edge[1]]))
        if endpoints[0] == endpoints[1]:
            raise ValueError("A full source edge collapses in root vertex identity.")
        key = (min(endpoints), max(endpoints))
        if key in by_roots:
            raise ValueError("Full source root endpoints have ambiguous edge occurrence.")
        by_roots[key] = (row, endpoints)
    result: dict[int, tuple[int, int]] = {}
    for row, edge in enumerate(reference.complex.plc_edge_vertices):
        endpoints = (int(core_roots[edge[0]]), int(core_roots[edge[1]]))
        match = by_roots.get((min(endpoints), max(endpoints)))
        if match is None:
            continue
        full_row, full_endpoints = match
        result[row] = (full_row, 1 if endpoints == full_endpoints else -1)
    return result


def mapped_source_edge_curve(
    source: NativeLayerCoreSource, full_row: int, orientation: int, /
) -> tuple[tuple[tuple[Fraction, Fraction], tuple[Expression, ...]], ...]:
    """Return exact piece-local source curves in original core-edge parameters."""
    query = source.fidelity_source
    if not isinstance(query, MappedReferenceBoundarySource):
        raise ValueError("Mapped edge curves require the full source and root geometry.")
    complex_ = query.authority.complex
    if not 0 <= full_row < len(complex_.plc_edge_entity_ids):
        raise ValueError("Mapped edge curve names an absent original source edge.")
    if orientation not in (-1, 1):
        raise ValueError("Mapped edge orientation must be positive or negative.")
    key = query.authority.source_entity_key_for_scope(
        1, int(complex_.plc_edge_entity_ids[full_row])
    )
    pieces = []
    for row, chart_id in enumerate(query.edge_chart_ids):
        if query.chart_source_key(1, chart_id) != key:
            continue
        first, last = (
            Fraction(*token)
            for token in query.segment_source_parameter_rational_tokens[row]
        )
        if orientation < 0:
            first, last = 1 - first, 1 - last
        if first == last:
            raise ValueError("A mapped source edge piece has zero parameter extent.")
        pieces.append(
            ((first, last), tuple(query.chart_coordinate_expressions(1, chart_id)))
        )
    if not pieces:
        raise ValueError("Mapped edge has no exact source chart pieces.")
    require_interval_partition(tuple(interval for interval, _ in pieces))
    return tuple(pieces)


def mapped_core_edge_curve(
    source: NativeLayerCoreSource, edge_row: int, /
) -> tuple[Expression, ...]:
    """Resolve an internal core edge through full-source vertex-root ancestry."""
    query, domain, authored = (
        source.fidelity_source,
        source.mapped_domain,
        source.reference_source,
    )
    if (
        not isinstance(query, MappedReferenceBoundarySource)
        or domain is None
        or authored is None
    ):
        raise ValueError(
            "Mapped core edges require their original full-source root authority."
        )
    relation = query.authority.root_relation
    roots = domain.reference_domain_vertex_root_ids
    if relation is None or roots is None:
        raise ValueError("Mapped core edges require declared vertex-root identities.")
    mapping, _, _, _ = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    if not 0 <= edge_row < len(authored.complex.plc_edge_vertices):
        raise ValueError("A mapped core edge row is absent from its source.")
    edge = authored.complex.plc_edge_vertices[edge_row]
    if any(not 0 <= int(mapping[vertex]) < len(roots) for vertex in edge):
        raise ValueError("A mapped core edge endpoint has no reference-domain root.")
    endpoints = tuple(int(roots[int(mapping[vertex])]) for vertex in edge)
    full_roots = set(int(value) for value in relation.vertex_root_global_ids)
    if endpoints[0] == endpoints[1] or not set(endpoints) <= full_roots:
        raise ValueError("Core edge endpoints do not bind distinct full-source roots.")
    root_mesh = domain.reference_mesh
    global_vertices = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
    root_maps = _maps(root_mesh, domain.source_geometry)
    curve: tuple[Expression, ...] | None = None
    for block in root_mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
            cell_vertices = tuple(
                int(value) for value in global_vertices[np.asarray(rows)]
            )
            if not set(endpoints) <= set(cell_vertices):
                continue
            first, last = (
                topology.vertices[cell_vertices.index(value)] for value in endpoints
            )
            arguments = tuple(
                add(
                    constant(Fraction.from_float(float(a)), 1),
                    scale(
                        {(1,): Fraction(1)},
                        Fraction.from_float(float(b)) - Fraction.from_float(float(a)),
                    ),
                )
                for a, b in zip(first, last, strict=True)
            )
            candidate = tuple(
                compose_expression(value, arguments)
                for value in root_maps.expressions(int(identifier))
            )
            if curve is None:
                curve = candidate
            elif not all(
                expression_equal(a, b) for a, b in zip(curve, candidate, strict=True)
            ):
                raise ValueError(
                    "Incident full-source roots disagree on an internal edge curve."
                )
    if curve is None:
        raise ValueError("No full-source root cell contains the declared core edge.")
    return curve


def core_edge_images(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    core_cell_ids: Sequence[int],
    /,
    *,
    root: LayerCoreRootGeometry | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> dict[
    int, tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...]
]:
    """Resolve core edge charts in canonical global-vertex order.

    Mapped reference images are reconstructed through the retained root atlas,
    and the complete physical cell map is checked against that same atlas.
    Every incident core cell is retained so later carrier proofs can refuse
    contradictory charts rather than choosing an arbitrary owner.
    """
    target_maps = _maps(mesh, geometry)
    domain = source.mapped_domain
    root_reference = None
    local_maps = None
    if domain is not None:
        from ._layer_core_mapped_segment_overlay import mapped_local_reference_maps

        local_maps = mapped_local_reference_maps(
            source, mesh, geometry, root, computed_root=computed_root
        )
        root_reference = _maps(
            domain.reference_mesh, CellGeometrySpec.affine(domain.reference_mesh)
        )
    ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    keys = entity_keys(mesh, 1)
    by_key = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(keys, ids, strict=True)
    }
    if len(by_key) != len(ids):
        raise ValueError("Core edge incidence contains ambiguous vertex keys.")
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    core = set(int(value) for value in core_cell_ids)
    found: set[int] = set()
    result: dict[
        int, list[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None]]
    ] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        for cell_value, vertex_rows in zip(block.global_ids, block.vertices, strict=True):
            cell = int(cell_value)
            if cell not in core:
                continue
            if cell in found:
                raise ValueError("A core cell has ambiguous block incidence.")
            found.add(cell)
            physical = target_maps.expressions(cell)
            root_image = None
            if local_maps is not None:
                assert root_reference is not None
                original_cell, original_arguments = local_maps[cell]
                root_image = tuple(
                    compose_expression(component, original_arguments)
                    for component in root_reference.expressions(original_cell)
                )
            cell_vertices = vertices[np.asarray(vertex_rows, dtype=np.int64)]
            for edge in topology.entities[1]:
                if len(edge) != 2:
                    raise ValueError("Core source replay requires two-ended cell edges.")
                a, b = (int(edge[0]), int(edge[1]))
                if cell_vertices[a] > cell_vertices[b]:
                    a, b = b, a
                key = (int(cell_vertices[a]), int(cell_vertices[b]))
                edge_id = by_key.get(key)
                if edge_id is None:
                    raise ValueError("A core cell edge is absent from target incidence.")
                first = topology.vertices[a]
                last = topology.vertices[b]
                arguments = tuple(
                    add(
                        constant(Fraction.from_float(float(x)), 1),
                        scale(
                            {(1,): Fraction(1)},
                            Fraction.from_float(float(y)) - Fraction.from_float(float(x)),
                        ),
                    )
                    for x, y in zip(first, last, strict=True)
                )
                image = tuple(compose_expression(value, arguments) for value in physical)
                reference_image = (
                    None
                    if root_image is None
                    else tuple(
                        compose_expression(value, arguments) for value in root_image
                    )
                )
                result.setdefault(edge_id, []).append((cell, image, reference_image))
    if found != core:
        raise ValueError("Core edge replay names absent target cells.")
    return {edge: tuple(rows) for edge, rows in result.items()}


def _triangle_line_interval(
    first: tuple[Fraction, ...],
    last: tuple[Fraction, ...],
    triangle: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[Fraction, Fraction] | None:
    """Clip a reference segment to a triangle with exact barycentric inequalities."""
    origin = triangle[0]
    u = tuple(b - a for a, b in zip(origin, triangle[1], strict=True))
    v = tuple(b - a for a, b in zip(origin, triangle[2], strict=True))
    axes = next(
        (
            (i, j)
            for i in range(len(origin))
            for j in range(i + 1, len(origin))
            if u[i] * v[j] - u[j] * v[i]
        ),
        None,
    )
    if axes is None:
        raise ValueError("A source chart triangle is degenerate.")
    i, j = axes
    determinant = u[i] * v[j] - u[j] * v[i]

    def weights(
        point: tuple[Fraction, ...],
    ) -> tuple[Fraction, Fraction, Fraction] | None:
        delta = tuple(value - base for value, base in zip(point, origin, strict=True))
        a = (delta[i] * v[j] - delta[j] * v[i]) / determinant
        b = (u[i] * delta[j] - u[j] * delta[i]) / determinant
        if any(origin[k] + a * u[k] + b * v[k] != point[k] for k in range(len(origin))):
            return None
        return 1 - a - b, a, b

    left, right = weights(first), weights(last)
    if left is None or right is None:
        return None
    lower, upper = Fraction(0), Fraction(1)
    for a, b in zip(left, right, strict=True):
        delta = b - a
        if not delta:
            if a < 0:
                return None
        elif delta > 0:
            lower = max(lower, -a / delta)
        else:
            upper = min(upper, -a / delta)
    return (lower, upper) if lower <= upper else None


def _source_chart_support(
    mesh: CellMesh, witness: SourceBoundaryChartWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]]:
    """Collect exact target vertices and positive-length edges of each chart piece."""
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in vertices[np.asarray(rows)]),
        )
        for block in mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    result: dict[
        tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]
    ] = {}
    for row, (cell_id, face_id, local_face) in enumerate(
        zip(
            witness.cell_global_ids,
            witness.entity_global_ids,
            witness.local_entity_indices,
            strict=True,
        )
    ):
        cell = cells.get(int(cell_id))
        if cell is None:
            raise ValueError("A source chart names an absent target cell.")
        kind, global_vertices = cell
        topology = reference_cell_topology(kind)
        face_index = int(local_face)
        if not 0 <= face_index < len(topology.entities[2]):
            raise ValueError("A source chart names an absent target face.")
        face = topology.entities[2][face_index]
        if len(face) not in (3, 4):
            raise ValueError(
                "Core source chart support requires triangular or quadrilateral target faces."
            )
        chart = witness.target_reference_chart(row)
        if len(chart) != 3:
            raise ValueError(
                "Core source chart support requires triangular chart pieces."
            )
        key = (
            int(face_id),
            witness.source_entity_ids[row],
            tuple(witness.occurrence_paths[row]),
        )
        covered_vertices, covered_edges = result.setdefault(key, (set(), set()))
        local_points = {
            int(global_vertices[index]): tuple(
                Fraction.from_float(float(value)) for value in topology.vertices[index]
            )
            for index in face
        }
        for vertex, point in local_points.items():
            interval = _triangle_line_interval(point, point, chart)
            if interval is not None:
                covered_vertices.add(vertex)
        face_vertices = set(int(index) for index in face)
        for local_edge in topology.entities[1]:
            if not set(local_edge) <= face_vertices:
                continue
            first, last = sorted(int(global_vertices[index]) for index in local_edge)
            interval = _triangle_line_interval(
                local_points[first], local_points[last], chart
            )
            if interval is not None and interval[0] < interval[1]:
                covered_edges.add((first, last))
    return result


def _source_face_support(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness,
    preparation: _LayerImageSupportPreparation,
    /,
) -> dict[tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]]:
    preparation.require(
        preparation.source,
        mesh,
        geometry,
        preparation.core_cell_ids,
        root=preparation.root,
        computed_root=preparation.computed_root,
    )
    cached = preparation.face_support.get(id(witness))
    if cached is not None and cached[0] is witness:
        return {
            key: (set(vertices), set(edges))
            for key, (vertices, edges) in cached[1].items()
        }
    if isinstance(witness, SourceBoundaryFaceImageWitness):
        result = _source_boundary_image_face_support(mesh, geometry, witness, preparation)
    else:
        result = _source_chart_support(mesh, witness)
    preparation.face_support[id(witness)] = (
        witness,
        {
            key: (frozenset(vertices), frozenset(edges))
            for key, (vertices, edges) in result.items()
        },
    )
    return result


def incident_core_source_facets(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    replay: LayerAssociationReplay,
    face_tables: Sequence[GeometryAssociation],
    chart_support: dict[
        tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]
    ],
    /,
) -> dict[int, frozenset[int]]:
    """Derive edge-to-source-facet candidates from independently proved faces."""
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    facet_rows: dict[tuple[str, tuple[str, ...]], set[int]] = {}
    facet_identity: dict[int, tuple[str, tuple[str, ...]]] = {}
    for row, (identifier, path) in enumerate(
        zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)
    ):
        if (
            authored.source_boundary_scope is not None
            and authored.core_facet_source_ids[row] is None
        ):
            continue
        identity = _core_source_identity(authored, 2, str(identifier), tuple(path))
        facet_identity[row] = identity
        facet_rows.setdefault(identity, set()).add(row)
    face_facets: dict[int, set[int]] = {}
    face_set_id = mesh.entity_set(2).entity_set_id
    for table in face_tables:
        if table.target_entity_set_id != face_set_id:
            raise ValueError("Core carrier input is not a target face association.")
        for identifier, entity, path in zip(
            table.target_global_ids,
            table.source_entity_ids,
            table.source_occurrence_paths,
            strict=True,
        ):
            rows = facet_rows.get((entity, tuple(path)))
            if rows is None:
                raise ValueError("A core face association lacks an authored facet.")
            face_facets.setdefault(int(identifier), set()).update(rows)

    mapping, caps, polygons, _ = prepare_identity(
        authored.layers, complex_, authored.vertex_layer_ids, authored.cap_polygon_ids
    )
    layer_mesh = authored.layers.mesh
    layer_ids = np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64)
    layer_face_ids = np.asarray(layer_mesh.entity_set(2).entity_ids, dtype=np.int64)
    layer_faces = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(
            entity_keys(layer_mesh, 2), layer_face_ids, strict=True
        )
    }
    cap_facets: dict[int, set[int]] = {}
    fine_cap_facets: dict[int, set[int]] = {}
    for polygon in caps:
        index = int(polygon)
        key = tuple(sorted(int(value) for value in layer_ids[mapping[polygons[index]]]))
        face_id = layer_faces.get(key)
        if face_id is None:
            raise ValueError("An authored core cap has no layer face incidence.")
        cap_facets.setdefault(face_id, set()).add(int(complex_.polygon_facets[index]))
    for relation in replay.containment:
        if relation.fine_dimension == 2 and relation.coarse_dimension == 2:
            rows = cap_facets.get(int(relation.coarse_id))
            if rows:
                fine_cap_facets.setdefault(int(relation.fine_id), set()).update(rows)
                face_facets.setdefault(int(relation.fine_id), set()).update(rows)

    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_keys = tuple(
        tuple(int(value) for value in key if value >= 0) for key in entity_keys(mesh, 1)
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_edges: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for edge, face in incidence_pairs(mesh, 1, 2):
        face_edges[int(face_ids[face])].append((int(edge_ids[edge]), edge_keys[edge]))
    result: dict[int, set[int]] = {}
    for face_id, rows in face_facets.items():
        pairs = face_edges.get(face_id)
        if not pairs:
            raise ValueError("A proved core face is absent from mesh edge incidence.")
        unconditional = fine_cap_facets.get(face_id, set()) & rows
        groups: dict[tuple[str, tuple[str, ...]], set[int]] = {}
        for facet in rows - unconditional:
            identity = facet_identity.get(facet)
            if identity is None:
                identity = _core_source_identity(
                    authored,
                    2,
                    str(complex_.facet_ids[facet]),
                    tuple(complex_.facet_occurrence_paths[facet]),
                )
                facet_identity[facet] = identity
            groups.setdefault(identity, set()).add(facet)
        support_groups = []
        for (identity, path), facets in groups.items():
            support = chart_support.get((face_id, identity, path))
            if support is None:
                raise ValueError("A source facet carrier lacks a validated chart piece.")
            support_groups.append((facets, support[1]))
        for edge_id, pair in pairs:
            if unconditional:
                result.setdefault(edge_id, set()).update(unconditional)
            for facets, supported_edges in support_groups:
                if tuple(sorted(pair)) in supported_edges:
                    result.setdefault(edge_id, set()).update(facets)
    return {edge: frozenset(rows) for edge, rows in result.items()}


def mapped_internal_facet_carriers(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    replay: LayerAssociationReplay,
    root: LayerCoreRootGeometry | None,
    validated_scoped: tuple[ValidatedSourceFidelityWitness, ...],
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> dict[int, frozenset[int]]:
    """Prove target-edge incidence in original same-material root facets."""
    from ..discretization._nested_reference import _chart_from_corners
    from ._layer_core_facet_authority import MappedCoreFacetAuthority
    from ._layer_core_mapped_segment_overlay import mapped_local_reference_maps

    domain, authored, query = (
        source.mapped_domain,
        source.reference_source,
        source.fidelity_source,
    )
    if domain is None:
        return {}
    if authored is None or not isinstance(query, MappedReferenceBoundarySource):
        raise ValueError("Mapped internal facets require the full source authority.")
    relation, roots = (
        query.authority.root_relation,
        domain.reference_domain_vertex_root_ids,
    )
    if relation is None or roots is None:
        raise ValueError("Mapped internal facets require exact original root identities.")
    mapping, _, polygons, _ = prepare_identity(
        authored.layers,
        authored.complex,
        authored.vertex_layer_ids,
        authored.cap_polygon_ids,
    )
    complex_ = authored.complex
    full_roots = frozenset(int(value) for value in relation.vertex_root_global_ids)
    by_vertex: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for polygon, facet_value in zip(polygons, complex_.polygon_facets, strict=True):
        facet = int(facet_value)
        first, last = (int(value) for value in complex_.facet_regions[facet])
        if first < 0 or first != last:
            continue
        indices = tuple(int(mapping[int(value)]) for value in polygon)
        if any(index < 0 or index >= len(roots) for index in indices):
            raise ValueError("An internal source facet lacks reference-domain vertices.")
        vertices = tuple(int(roots[index]) for index in indices)
        if (
            len(vertices) != 3
            or len(set(vertices)) != 3
            or not set(vertices) <= full_roots
        ):
            raise ValueError(
                "An internal source facet lacks distinct full-source root vertices."
            )
        by_vertex[vertices[0]].append((facet, vertices))
    if not by_vertex:
        return {}
    facet_rows = frozenset(facet for values in by_vertex.values() for facet, _ in values)
    proved = any(
        isinstance(token.source, MappedReferenceBoundarySource)
        and isinstance(token.source.authority, MappedCoreFacetAuthority)
        and token.binds(mesh, geometry, token.source)
        and token.source.authority.source.binding_id == source.binding_id
        and facet_rows <= set(token.source.authority.facet_rows)
        for token in validated_scoped
    )
    if not proved:
        MappedCoreFacetAuthority(source, tuple(sorted(facet_rows)))
    root_mesh = domain.reference_mesh
    root_vertices = np.asarray(root_mesh.vertex_global_ids, dtype=np.int64)
    root_cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in root_vertices[np.asarray(rows)]),
        )
        for block in root_mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    local_maps = mapped_local_reference_maps(
        source, mesh, geometry, root, computed_root=computed_root
    )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edges = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(entity_keys(mesh, 1), edge_ids, strict=True)
    }
    global_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    core = frozenset(int(value) for value in replay.core_cell_ids)
    result: dict[int, set[int]] = {}
    for block in mesh.blocks:
        topology = reference_cell_topology(block.cell_kind)
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
            cell = int(identifier)
            if cell not in core:
                continue
            root_id, arguments = local_maps[cell]
            original = root_cells.get(root_id)
            if original is None:
                raise ValueError(
                    "Mapped internal facet names an absent source root cell."
                )
            root_kind, root_global = original
            root_topology = reference_cell_topology(root_kind)
            positions = {value: index for index, value in enumerate(root_global)}
            candidates: list[tuple[int, tuple[tuple[Fraction, ...], ...]]] = []
            for vertex in root_global:
                for facet, original_vertices in by_vertex.get(vertex, ()):
                    if not all(value in positions for value in original_vertices):
                        continue
                    local_indices = {positions[value] for value in original_vertices}
                    if not any(
                        local_indices <= set(face) for face in root_topology.entities[2]
                    ):
                        continue
                    triangle = tuple(
                        tuple(
                            Fraction.from_float(float(value))
                            for value in root_topology.vertices[positions[item]]
                        )
                        for item in original_vertices
                    )
                    candidates.append((facet, triangle))
            if not candidates:
                continue
            cell_vertices = global_vertices[np.asarray(rows)]
            for local_edge in topology.entities[1]:
                corners = tuple(
                    tuple(
                        Fraction.from_float(float(value))
                        for value in topology.vertices[index]
                    )
                    for index in local_edge
                )
                edge_chart = _chart_from_corners("interval", corners)
                image = tuple(
                    compose_expression(value, edge_chart) for value in arguments
                )
                endpoints = tuple(
                    tuple(
                        evaluate_expression(value, (Fraction(point),)) for value in image
                    )
                    for point in (0, 1)
                )
                if any(
                    not isinstance(value, Fraction)
                    for point in endpoints
                    for value in point
                ):
                    raise ValueError(
                        "Mapped internal edge charts require rational root coordinates."
                    )
                endpoints = cast(tuple[tuple[Fraction, ...], ...], endpoints)
                for axis, value in enumerate(image):
                    expected = add(
                        constant(endpoints[0][axis], 1),
                        scale(
                            {(1,): Fraction(1)}, endpoints[1][axis] - endpoints[0][axis]
                        ),
                    )
                    if not expression_equal(value, expected):
                        raise ValueError(
                            "Mapped internal edge ancestry must be affine in root coordinates."
                        )
                key = tuple(sorted(int(cell_vertices[index]) for index in local_edge))
                edge_id = edges.get(key)
                if edge_id is None:
                    raise ValueError(
                        "An internal core edge is absent from mesh incidence."
                    )
                for facet, triangle in candidates:
                    interval = _triangle_line_interval(
                        endpoints[0], endpoints[1], triangle
                    )
                    if interval is not None and interval[0] < interval[1]:
                        result.setdefault(edge_id, set()).add(facet)
    return {edge: frozenset(rows) for edge, rows in result.items()}


def _source_edge_spatial_candidates(
    exact_points: tuple[tuple[Fraction, ...], ...],
    source_edges: Sequence[Sequence[int]],
    images: dict[
        int, tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...]
    ],
    *,
    mapped: bool,
) -> dict[int, set[int]]:
    """Conservatively filter finite segment support, never proving a chart."""
    from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
    from ..geometry._rational import fraction_down, fraction_up

    if not exact_points or len(source_edges) == 0:
        return {}
    dimension = len(exact_points[0])
    if dimension != 3:
        # The unmapped exact overlap owner has its own dimensional refusal.
        return {}
    # Degenerate rows must still reach the exact routine and its refusal.
    degenerate: set[int] = set()
    source_rows: list[int] = []
    source_lower: list[tuple[float, ...]] = []
    source_upper: list[tuple[float, ...]] = []
    for row, (first_row, last_row) in enumerate(source_edges):
        first, last = exact_points[first_row], exact_points[last_row]
        if first == last:
            degenerate.add(row)
            continue
        source_rows.append(row)
        source_lower.append(
            tuple(fraction_down(min(a, b)) for a, b in zip(first, last, strict=True))
        )
        source_upper.append(
            tuple(fraction_up(max(a, b)) for a, b in zip(first, last, strict=True))
        )
    # Float32 BVHs remain conservative even when JAX x64 is disabled.
    # Fall back to the exact loop for coordinates outside its finite range.
    float_limit = float(np.finfo(np.float32).max) / 2
    if any(
        abs(value) > float_limit
        for box in (*source_lower, *source_upper)
        for value in box
    ):
        return {}

    result: dict[int, set[int]] = {}
    target_edges: list[int] = []
    target_lower: list[tuple[float, ...]] = []
    target_upper: list[tuple[float, ...]] = []
    for edge, edge_images in images.items():
        bounds: list[tuple[tuple[Fraction, ...], ...]] = []
        try:
            for _, physical, reference in edge_images:
                chart = reference if mapped else physical
                if chart is None or len(chart) != dimension:
                    raise ValueError
                endpoints = tuple(
                    tuple(
                        evaluate_expression(component, (Fraction(t),))
                        for component in chart
                    )
                    for t in (0, 1)
                )
                if any(
                    not isinstance(value, Fraction)
                    for point in endpoints
                    for value in point
                ):
                    raise ValueError
                bounds.append(cast(tuple[tuple[Fraction, ...], ...], endpoints))
            lower = tuple(
                fraction_down(
                    min(point[axis] for endpoints in bounds for point in endpoints)
                )
                for axis in range(dimension)
            )
            upper = tuple(
                fraction_up(
                    max(point[axis] for endpoints in bounds for point in endpoints)
                )
                for axis in range(dimension)
            )
            if any(
                not np.isfinite(value) or abs(value) > float_limit
                for value in (*lower, *upper)
            ):
                raise ValueError
        except (ArithmeticError, TypeError, ValueError, IndexError):
            # The exact owner retains responsibility for malformed charts and
            # their ordered errors. An empty image likewise needs no filtering.
            continue
        result[edge] = set(degenerate)
        target_edges.append(edge)
        target_lower.append(lower)
        target_upper.append(upper)
    if source_rows and target_edges:
        source_tree = prepare_bvh(
            np.asarray(source_lower), np.asarray(source_upper), dtype=np.float32
        )
        target_tree = prepare_bvh(
            np.asarray(target_lower), np.asarray(target_upper), dtype=np.float32
        )
        for source_block, target_block in bvh_overlap_pair_blocks(
            source_tree, target_tree, include_touching=True
        ):
            for source_index, target_index in zip(
                source_block, target_block, strict=True
            ):
                result[target_edges[int(target_index)]].add(
                    source_rows[int(source_index)]
                )
    return result


def match_core_source_edges(
    source: NativeLayerCoreSource,
    images: dict[
        int, tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...]
    ],
    carriers: dict[int, frozenset[int]],
    /,
) -> dict[
    int, tuple[tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]
]:
    """Prove incident mesh edges against complete authored PLC edge images.

    Multiple source rows remain distinct; a caller must apply the independently
    declared active-carrier policy and prove the resulting interval partitions.
    """
    from .providers._native_layer_overlap import exact_collinear_segment_partitions

    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    exact_points = _exact_source_points(source)
    bindings: dict[int, tuple[int, int]] = {}
    if source.mapped_domain is not None:
        bindings = mapped_source_edge_bindings(source)
    by_facet: dict[int, set[int]] = {}
    for row, facets in enumerate(complex_.plc_edge_facets):
        for facet in facets:
            by_facet.setdefault(int(facet), set()).add(row)
    spatial_candidates = _source_edge_spatial_candidates(
        exact_points,
        complex_.plc_edge_vertices,
        {
            edge: edge_images
            for edge, edge_images in images.items()
            if any(facet in by_facet for facet in carriers.get(edge, ()))
        },
        mapped=source.mapped_domain is not None,
    )
    curves: dict[
        int, tuple[tuple[tuple[Fraction, Fraction], tuple[Expression, ...]], ...]
    ] = {}
    result: dict[
        int, tuple[tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]
    ] = {}
    for edge_id, edge_images in images.items():
        candidates: set[int] = set()
        for facet in carriers.get(edge_id, ()):
            candidates.update(by_facet.get(facet, ()))
        if edge_id in spatial_candidates:
            candidates.intersection_update(spatial_candidates[edge_id])
        matches: list[
            tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
        ] = []
        for row in sorted(candidates):
            first_row, last_row = complex_.plc_edge_vertices[row]
            first, last = exact_points[first_row], exact_points[last_row]
            intervals = []
            target_intervals = []
            for _, physical, reference in edge_images:
                chart = physical if source.mapped_domain is None else reference
                if chart is None:
                    raise ValueError(
                        "A mapped edge lacks its exact root-reference chart."
                    )
                if source.mapped_domain is not None:
                    interval = source_line_interval(chart, first, last)
                    target_interval = (
                        (Fraction(0), Fraction(1)) if interval is not None else None
                    )
                else:
                    # Establish the whole target chart is affine on the original
                    # source line before intersecting either finite support.
                    line = source_line_interval(chart, first, last, bounded=False)
                    if line is None:
                        interval = target_interval = None
                    else:
                        target_segment = tuple(
                            tuple(
                                evaluate_expression(component, (Fraction(t),))
                                for component in chart
                            )
                            for t in (0, 1)
                        )
                        pieces = exact_collinear_segment_partitions(
                            ((first, last),), (target_segment,)
                        )
                        interval = pieces[0].source_interval if pieces else None
                        target_interval = pieces[0].target_interval if pieces else None
                intervals.append(interval)
                target_intervals.append(target_interval)
            if all(value is None for value in intervals):
                continue
            if (
                not intervals
                or any(value is None or value != intervals[0] for value in intervals)
                or any(
                    value is None or value != target_intervals[0]
                    for value in target_intervals
                )
            ):
                raise ValueError(
                    "Incident core cells disagree on source-edge parameters."
                )
            interval = intervals[0]
            assert interval is not None
            if source.mapped_domain is not None:
                binding = bindings.get(row)
                if row not in curves:
                    curves[row] = (
                        (
                            (
                                (Fraction(0), Fraction(1)),
                                mapped_core_edge_curve(source, row),
                            ),
                        )
                        if binding is None
                        else mapped_source_edge_curve(source, *binding)
                    )
                if any(
                    not source_curve_piecewise_interval(physical, curves[row], interval)
                    for _, physical, _ in edge_images
                ):
                    raise ValueError("A mapped edge does not equal its source curve.")
            target_interval = target_intervals[0]
            assert target_interval is not None
            matches.append((row, interval, target_interval))
        if matches:
            result[edge_id] = tuple(matches)
    return result


def core_edge_association_tables(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    matches: dict[
        int, tuple[tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]
    ],
    active_source_rows: Sequence[int],
    /,
    *,
    required_source_rows: Sequence[int] | None = None,
) -> tuple[GeometryAssociation, ...]:
    """Publish exact source/target intervals; required rows must be complete."""
    authored = source if source.reference_source is None else source.reference_source
    source_edges = authored.complex.plc_edge_ids
    source_paths = authored.complex.plc_edge_occurrence_paths
    active = tuple(int(row) for row in active_source_rows)
    if len(set(active)) != len(active) or any(
        row < 0 or row >= len(source_edges) for row in active
    ):
        raise ValueError("Active core carrier rows are invalid or ambiguous.")
    required = frozenset(active if required_source_rows is None else required_source_rows)
    if not required <= set(active):
        raise ValueError(
            "Required source edge rows must belong to the recovered inventory."
        )
    by_source: dict[
        int, list[tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]]]
    ] = {row: [] for row in active}
    for target, choices in matches.items():
        target_supports = sorted(
            (target_interval, row)
            for row, _, target_interval in choices
            if row in by_source
        )
        if any(
            left[0][1] > right[0][0]
            for left, right in zip(target_supports, target_supports[1:])
        ):
            raise ValueError("Distinct source curves overlap on a target-edge interior.")
        seen: set[int] = set()
        for row, interval, target_interval in choices:
            if row not in by_source:
                continue
            if row in seen:
                raise ValueError("A target edge repeats a source carrier occurrence.")
            seen.add(row)
            by_source[row].append((target, interval, target_interval))
    records = []
    for row in active:
        values = sorted(by_source[row])
        if row in required and any(
            target_interval != (Fraction(0), Fraction(1))
            for _, _, target_interval in values
        ):
            raise ValueError(
                "A required source edge occupies only part of a target edge."
            )
        if row in required:
            try:
                require_interval_partition(tuple(interval for _, interval, _ in values))
            except ValueError as error:
                raise ValueError(
                    f"Core source edge {row} ({source_edges[row]!r}) has incomplete "
                    f"or overlapping target interval coverage: {values!r}."
                ) from error
        elif values:
            intervals = sorted(
                (min(interval), max(interval)) for _, interval, _ in values
            )
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError("Optional recovered source-edge intervals overlap.")
        else:
            continue
        identity, path = _core_source_identity(
            authored, 1, str(source_edges[row]), tuple(source_paths[row])
        )
        ids = np.asarray([target for target, _, _ in values], dtype=np.int64)
        records.append(
            GeometryAssociation(
                GeometryAssociationKind.PIECEWISE_LINEAR,
                source.source_id,
                source.source_revision,
                mesh.entity_set(1).entity_set_id,
                ids,
                (identity,) * len(values),
                np.zeros(ids.shape, dtype=np.float64),
                exact=True,
                source_roles=(GeometrySourceEntityRole.EDGE,) * len(values),
                source_occurrence_paths=(path,) * len(values),
                source_parameter_intervals=tuple(
                    (min(first, last), max(first, last)) for _, (first, last), _ in values
                ),
                target_parameter_intervals=tuple(
                    target_interval for _, _, target_interval in values
                ),
                orientations=np.asarray(
                    [1 if last > first else -1 for _, (first, last), _ in values],
                    dtype=np.int8,
                ),
                provenance=GeometryAssociationProvenance.CLASSIFICATION,
            )
        )
    return tuple(records)


def _exact_source_points(
    source: NativeLayerCoreSource, /
) -> tuple[tuple[Fraction, ...], ...]:
    """Return authored PLC vertices in the coordinates of the core edge charts."""
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    points = np.asarray(complex_.vertices)
    if source.mapped_domain is not None:
        mapping, _, _, _ = prepare_identity(
            authored.layers, complex_, authored.vertex_layer_ids, authored.cap_polygon_ids
        )
        points = np.asarray(source.mapped_domain.reference_domain.vertices)[mapping]
    return tuple(
        tuple(Fraction.from_float(float(value)) for value in point) for point in points
    )


def _shared_facet_vertex_stratum(
    source: NativeLayerCoreSource,
    facet_keys: frozenset[tuple[str, tuple[str, ...]]],
    vertex_edges: Sequence[tuple[int, int]],
    images: dict[
        int, tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...]
    ],
    exact_points: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[int, Fraction | None] | None:
    """Locate a vertex of several closed source facets on their shared PLC stratum.

    Distinct facets of an authored PLC meet only in shared PLC edges and vertices,
    so a vertex exactly supported by every facet chart lies on that intersection.
    The exact vertex image of every incident core edge chart (given as edge ID and
    chart endpoint) is compared with every PLC edge incident to all those facets.
    The result is ``(edge_row, parameter)`` for an edge interior, ``(-1 -
    vertex_row, None)`` for a PLC vertex, and ``None`` when the exact intersection
    is not one stratum.
    """
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    facet_identity: dict[int, tuple[str, tuple[str, ...]]] = {}
    for row, (identifier, path) in enumerate(
        zip(complex_.facet_ids, complex_.facet_occurrence_paths, strict=True)
    ):
        if (
            authored.source_boundary_scope is not None
            and authored.core_facet_source_ids[row] is None
        ):
            continue
        facet_identity[row] = _core_source_identity(
            authored, 2, str(identifier), tuple(path)
        )
    mapped = source.mapped_domain is not None
    point: tuple[Fraction, ...] | None = None
    for edge_id, endpoint in vertex_edges:
        for _, physical, reference in images.get(edge_id, ()):
            chart = reference if mapped else physical
            if chart is None:
                raise ValueError("A mapped edge lacks its exact root-reference chart.")
            value = tuple(
                evaluate_expression(component, (Fraction(endpoint),))
                for component in chart
            )
            if not all(isinstance(item, Fraction) for item in value):
                return None
            exact = cast(tuple[Fraction, ...], value)
            if point is None:
                point = exact
            elif exact != point:
                raise ValueError("Incident core edge charts disagree on a vertex image.")
    if point is None:
        return None
    strata: set[tuple[int, Fraction | None]] = set()
    for row, facets in enumerate(complex_.plc_edge_facets):
        incident = {facet_identity[int(f)] for f in facets if int(f) in facet_identity}
        if not facet_keys <= incident:
            continue
        first_row, last_row = (int(value) for value in complex_.plc_edge_vertices[row])
        first, last = exact_points[first_row], exact_points[last_row]
        direction = tuple(b - a for a, b in zip(first, last, strict=True))
        axis = next((i for i, value in enumerate(direction) if value), None)
        if axis is None or len(point) != len(first):
            raise ValueError("An authored source edge must have distinct endpoints.")
        parameter = (point[axis] - first[axis]) / direction[axis]
        if not 0 <= parameter <= 1 or any(
            origin + parameter * delta != coordinate
            for origin, delta, coordinate in zip(first, direction, point, strict=True)
        ):
            continue
        if parameter == 0:
            strata.add((-1 - first_row, None))
        elif parameter == 1:
            strata.add((-1 - last_row, None))
        else:
            strata.add((row, parameter))
    if len(strata) != 1:
        return None
    return next(iter(strata))


def core_vertex_association_table(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    replay: LayerAssociationReplay,
    matches: dict[
        int, tuple[tuple[int, tuple[Fraction, Fraction], tuple[Fraction, Fraction]], ...]
    ],
    active_source_rows: Sequence[int],
    face_tables: Sequence[GeometryAssociation],
    region_table: GeometryAssociation,
    chart_support: dict[
        tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]
    ],
    /,
    *,
    cap_coverage_proved: bool,
    images: dict[
        int, tuple[tuple[int, tuple[Expression, ...], tuple[Expression, ...] | None], ...]
    ],
) -> GeometryAssociation | None:
    """Classify core vertices from source edge, face, and material incidence.

    A vertex exactly supported by several distinct source facets without a lower
    recovered stratum lies on their shared PLC edge or vertex: it is located there
    exactly, and a derived-edge interior keeps the derived-edge classification.
    """
    authored = source if source.reference_source is None else source.reference_source
    complex_ = authored.complex
    active = frozenset(int(row) for row in active_source_rows)
    edges = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(
            entity_keys(mesh, 1), mesh.entity_set(1).entity_ids, strict=True
        )
    }
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells: dict[int, tuple[int, ...]] = {}
    for block in mesh.blocks:
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
            cells[int(identifier)] = tuple(
                int(value) for value in vertices[np.asarray(rows)]
            )
    core = frozenset(int(value) for value in replay.core_cell_ids)
    if not core <= cells.keys():
        raise ValueError("Core vertex replay names absent cells.")
    core_vertices = {vertex for cell in core for vertex in cells[cell]}
    # A lower rank is the more specific source stratum.
    classes: dict[
        int,
        tuple[int, str, tuple[str, ...], Fraction | None, tuple[str | None, str | None]],
    ] = {}
    derived_edge_vertices: set[int] = set()

    def install(
        vertex: int,
        value: tuple[
            int, str, tuple[str, ...], Fraction | None, tuple[str | None, str | None]
        ],
    ) -> None:
        if vertex not in core_vertices:
            return
        previous = classes.get(vertex)
        if previous is None or value[0] < previous[0]:
            classes[vertex] = value
        elif value[0] == previous[0] and value != previous:
            raise ValueError(
                "A core vertex has ambiguous occurrence-qualified source strata."
            )

    for edge_id, choices in matches.items():
        key = edges.get(edge_id)
        if key is None or len(key) != 2:
            raise ValueError("A matched core edge has invalid target incidence.")
        for row, (first, last), (target_first, target_last) in choices:
            endpoints = complex_.plc_edge_vertices[row]
            endpoint_values = (
                (key[0], first) if target_first == 0 else None,
                (key[1], last) if target_last == 1 else None,
            )
            for item in endpoint_values:
                if item is None:
                    continue
                vertex, parameter = item
                if parameter in (0, 1):
                    source_vertex = endpoints[int(parameter)]
                    identity, path = _core_source_identity(
                        authored,
                        0,
                        str(complex_.vertex_ids[source_vertex]),
                        tuple(complex_.vertex_occurrence_paths[source_vertex]),
                    )
                    install(vertex, (0, identity, path, None, (None, None)))
                elif row in active and row >= len(complex_.segments):
                    derived_edge_vertices.add(vertex)
                elif row in active:
                    identity, path = _core_source_identity(
                        authored,
                        1,
                        str(complex_.plc_edge_ids[row]),
                        tuple(complex_.plc_edge_occurrence_paths[row]),
                    )
                    install(vertex, (1, identity, path, parameter, (None, None)))
    faces = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(
            entity_keys(mesh, 2), mesh.entity_set(2).entity_ids, strict=True
        )
    }
    from ._layer_core_boundary_replay import replay_core_interface_faces

    cap_table = replay_core_interface_faces(
        source, mesh, replay, require_coverage=not cap_coverage_proved
    )
    cap_rows = (
        set()
        if cap_table is None
        else {
            (int(face), identity, tuple(path), tuple(sides), int(orientation))
            for face, identity, path, sides, orientation in zip(
                cap_table.target_global_ids,
                cap_table.source_entity_ids,
                cap_table.source_occurrence_paths,
                cap_table.source_region_sides,
                cap_table.orientations,
                strict=True,
            )
        }
    )
    face_candidates: dict[
        int,
        set[
            tuple[
                int, str, tuple[str, ...], Fraction | None, tuple[str | None, str | None]
            ]
        ],
    ] = defaultdict(set)
    for table in face_tables:
        if table.target_entity_set_id != mesh.entity_set(2).entity_set_id:
            raise ValueError("A core vertex carrier is not a target face table.")
        for face, identity, path, sides, orientation in zip(
            table.target_global_ids,
            table.source_entity_ids,
            table.source_occurrence_paths,
            table.source_region_sides,
            table.orientations,
            strict=True,
        ):
            key = faces.get(int(face))
            if key is None:
                raise ValueError("A core vertex carrier names an absent face.")
            support = chart_support.get((int(face), identity, tuple(path)))
            if (
                support is None
                and (int(face), identity, tuple(path), tuple(sides), int(orientation))
                not in cap_rows
            ):
                raise ValueError(
                    "A core vertex face carrier lacks exact chart or cap support."
                )
            covered_vertices = set(key) if support is None else support[0]
            for vertex in key:
                if (
                    vertex in core_vertices
                    and vertex in covered_vertices
                    and vertex not in derived_edge_vertices
                ):
                    face_candidates[vertex].add(
                        (2, identity, tuple(path), None, tuple(sides))
                    )
    # Face strata are installed after every lower recovered stratum is known.
    shared = {
        vertex
        for vertex, candidates in face_candidates.items()
        if vertex not in classes
        and len({(value[1], value[2]) for value in candidates}) > 1
    }
    vertex_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    exact_points: tuple[tuple[Fraction, ...], ...] = ()
    if shared:
        # Core edge charts run from the lower to the higher global vertex ID.
        for edge_id, key in edges.items():
            if len(key) == 2:
                for vertex in key:
                    if vertex in shared:
                        vertex_edges[vertex].append(
                            (edge_id, 0 if vertex == min(key) else 1)
                        )
        exact_points = _exact_source_points(source)
    for vertex, candidates in face_candidates.items():
        if vertex not in shared:
            for value in sorted(candidates, key=repr):
                install(vertex, value)
            continue
        stratum = _shared_facet_vertex_stratum(
            source,
            frozenset((value[1], value[2]) for value in candidates),
            vertex_edges[vertex],
            images,
            exact_points,
        )
        if stratum is None:
            raise ValueError(
                "A core vertex has ambiguous occurrence-qualified source strata."
            )
        row, parameter = stratum
        if parameter is None:
            source_vertex = -1 - row
            identity, path = _core_source_identity(
                authored,
                0,
                str(complex_.vertex_ids[source_vertex]),
                tuple(complex_.vertex_occurrence_paths[source_vertex]),
            )
            install(vertex, (0, identity, path, None, (None, None)))
        elif row not in active:
            raise ValueError(
                "A core vertex lies on an inactive source edge of several facets."
            )
        elif row >= len(complex_.segments):
            classes.pop(vertex, None)
            derived_edge_vertices.add(vertex)
        else:
            identity, path = _core_source_identity(
                authored,
                1,
                str(complex_.plc_edge_ids[row]),
                tuple(complex_.plc_edge_occurrence_paths[row]),
            )
            install(vertex, (1, identity, path, parameter, (None, None)))
    if region_table.target_entity_set_id != mesh.entity_set(3).entity_set_id:
        raise ValueError("Core vertex material incidence requires a cell region table.")
    regions = {
        int(cell): (identity, tuple(path))
        for cell, identity, path in zip(
            region_table.target_global_ids,
            region_table.source_entity_ids,
            region_table.source_occurrence_paths,
            strict=True,
        )
    }
    if not core <= regions.keys():
        raise ValueError("Core cell material classes are incomplete.")
    for cell in core:
        identity, path = regions[cell]
        for vertex in cells[cell]:
            if vertex not in derived_edge_vertices:
                install(vertex, (3, identity, path, None, (None, None)))
    if not classes:
        return None
    ids = np.asarray(sorted(classes), dtype=np.int64)
    values = tuple(classes[int(identifier)] for identifier in ids)
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        mesh.entity_set(0).entity_set_id,
        ids,
        tuple(value[1] for value in values),
        np.zeros(ids.shape, dtype=np.float64),
        exact=True,
        source_roles=tuple(roles[value[0]] for value in values),
        source_occurrence_paths=tuple(value[2] for value in values),
        source_region_sides=tuple(value[4] for value in values),
        source_parameter_points=tuple(value[3] for value in values),
        orientations=np.zeros(ids.shape, dtype=np.int8),
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
    )


def source_line_interval(
    image: Sequence[Expression],
    first: Sequence[Fraction],
    last: Sequence[Fraction],
    /,
    *,
    bounded: bool = True,
) -> tuple[Fraction, Fraction] | None:
    """Prove an entire one-dimensional chart is an affine source-line restriction.

    The caller must separately establish source-carrier incidence. This routine
    proves geometry only; coincident curves do not establish source identity.
    Endpoints and every coordinate of the complete image are compared exactly.
    """
    if not image or len(image) != len(first) or len(first) != len(last):
        raise ValueError("Source line and image dimensions must agree.")
    direction = tuple(b - a for a, b in zip(first, last, strict=True))
    axis = next((i for i, value in enumerate(direction) if value), None)
    if axis is None:
        raise ValueError("An authored source edge must have distinct endpoints.")
    endpoints = tuple(evaluate_expression(image[axis], (Fraction(t),)) for t in (0, 1))
    if not isinstance(endpoints[0], Fraction) or not isinstance(endpoints[1], Fraction):
        raise ValueError("Source-line endpoints must have rational parameters.")
    start = (endpoints[0] - first[axis]) / direction[axis]
    end = (endpoints[1] - first[axis]) / direction[axis]
    if start == end or (bounded and not (0 <= start <= 1 and 0 <= end <= 1)):
        return None
    parameter: Expression = {(0,): start, (1,): end - start}
    for component, origin, delta in zip(image, first, direction, strict=True):
        expected = add(constant(origin, 1), scale(parameter, delta))
        if not expression_equal(component, expected):
            return None
    return start, end


def source_curve_interval(
    image: Sequence[Expression],
    source_curve: Sequence[Expression],
    interval: tuple[Fraction, Fraction],
    /,
) -> bool:
    """Compare the whole physical image with an exact root-source curve chart."""
    if len(image) != len(source_curve):
        raise ValueError("Source curve and target image dimensions must agree.")
    start, end = interval
    if start == end or not (0 <= start <= 1 and 0 <= end <= 1):
        raise ValueError("A source curve interval must be nondegenerate and bounded.")
    argument: Expression = {(0,): start, (1,): end - start}
    return all(
        expression_equal(target, compose_expression(source, (argument,)))
        for target, source in zip(image, source_curve, strict=True)
    )


def source_curve_piecewise_interval(
    image: Sequence[Expression],
    pieces: Sequence[tuple[tuple[Fraction, Fraction], tuple[Expression, ...]]],
    interval: tuple[Fraction, Fraction],
    /,
) -> bool:
    """Compare every exact overlap with its own source-piece parameterization."""
    start, end = interval
    if start == end or not (0 <= start <= 1 and 0 <= end <= 1):
        raise ValueError("A source curve interval must be nondegenerate and bounded.")
    lower, upper = sorted(interval)
    covered = []
    for (first, last), curve in pieces:
        if first == last:
            raise ValueError("A source curve piece must be nondegenerate.")
        left, right = max(lower, min(first, last)), min(upper, max(first, last))
        if left >= right:
            continue
        target_first, target_last = (
            (left - start) / (end - start),
            (right - start) / (end - start),
        )
        source_first, source_last = (
            (left - first) / (last - first),
            (right - first) / (last - first),
        )
        argument: Expression = {(0,): target_first, (1,): target_last - target_first}
        restricted = tuple(compose_expression(value, (argument,)) for value in image)
        if not source_curve_interval(restricted, curve, (source_first, source_last)):
            return False
        covered.append((left, right))
    cursor = lower
    for left, right in sorted(covered):
        if left != cursor:
            return False
        cursor = right
    return cursor == upper


def require_interval_partition(
    intervals: Sequence[tuple[Fraction, Fraction]],
    /,
) -> None:
    """Require a gap-free, nonoverlapping exact partition of a source edge."""
    ordered = sorted((min(a, b), max(a, b)) for a, b in intervals)
    cursor = Fraction(0)
    for lower, upper in ordered:
        if lower != cursor or upper <= lower or upper > 1:
            raise ValueError("Source-edge intervals do not form an exact partition.")
        cursor = upper
    if cursor != 1:
        raise ValueError("Source-edge intervals do not cover the complete source edge.")


def _replay_core_strata(
    source: NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    primary_token: ValidatedSourceFidelityWitness,
    replay: LayerAssociationReplay,
    face_tables: Sequence[GeometryAssociation],
    region_table: GeometryAssociation,
    carrier_inventory: NativeLayerCoreCarrierWitness,
    validated_scoped: tuple[ValidatedSourceFidelityWitness, ...],
    root: LayerCoreRootGeometry | None,
    preparation: _LayerImageSupportPreparation,
    /,
    *,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> tuple[GeometryAssociation, ...]:
    """Independently rebuild core edge and vertex strata from source authority."""
    if not isinstance(carrier_inventory, NativeLayerCoreCarrierWitness):
        raise TypeError("Core stratum replay requires an accepted carrier witness.")
    if not isinstance(primary_token, ValidatedSourceFidelityWitness) or (
        primary_token.source is not source.fidelity_source
    ):
        raise ValueError("Core stratum replay requires its original source query.")
    from ._layer_core_boundary_replay import (
        _require_layer_fidelity_token,
        core_source_facet_keys,
        require_scoped_cap_coverage,
    )

    witness = _require_layer_fidelity_token(mesh, geometry, primary_token, source)
    if not isinstance(
        witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
    ):
        raise ValueError("Core stratum replay requires a source face witness.")

    cap_coverage_proved = source.cap_refinement == "conforming"
    if cap_coverage_proved:
        require_scoped_cap_coverage(mesh, geometry, validated_scoped, source)
    carrier_inventory.require_current(source, specification)
    active = carrier_inventory.original_source_edge_rows
    from ._layer_core_edge_policy import required_core_source_edge_rows

    required = set(required_core_source_edge_rows(source, specification))
    if not required <= set(active):
        raise ValueError(
            "Native carrier inventory omits an authored or requested source edge."
        )
    preparation.require(
        source,
        mesh,
        geometry,
        replay.core_cell_ids,
        root=root,
        computed_root=computed_root,
    )
    images = preparation.images
    chart_support = _source_face_support(mesh, geometry, witness, preparation)
    source_keys = core_source_facet_keys(source)
    for token in validated_scoped:
        if not isinstance(
            token.witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        ) or (token.source.source_id, token.source.source_revision) != (
            source.source_id,
            source.source_revision,
        ):
            continue
        keys = set(
            zip(
                token.witness.source_entity_ids,
                token.witness.occurrence_paths,
                strict=True,
            )
        )
        if not keys & source_keys:
            continue
        scoped_witness = _require_layer_fidelity_token(mesh, geometry, token, source)
        assert isinstance(
            scoped_witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        )
        for key, (vertices, edges) in _source_face_support(
            mesh, geometry, scoped_witness, preparation
        ).items():
            existing = chart_support.setdefault(key, (set(), set()))
            existing[0].update(vertices)
            existing[1].update(edges)
    carriers = incident_core_source_facets(
        source, mesh, replay, face_tables, chart_support
    )
    for edge_id, facets in mapped_internal_facet_carriers(
        source,
        mesh,
        geometry,
        replay,
        root,
        validated_scoped,
        computed_root=computed_root,
    ).items():
        carriers[edge_id] = carriers.get(edge_id, frozenset()) | facets
    matches = match_core_source_edges(source, images, carriers)
    edges = core_edge_association_tables(
        source,
        mesh,
        matches,
        active,
        required_source_rows=tuple(sorted(required)),
    )
    vertices = core_vertex_association_table(
        source,
        mesh,
        replay,
        matches,
        active,
        face_tables,
        region_table,
        chart_support,
        cap_coverage_proved=cap_coverage_proved,
        images=images,
    )
    return (() if vertices is None else (vertices,)) + edges

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Original same-material core facets as independently bound mapped charts."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._source_identity import source_entity_id
from ._layer_core_topology import prepare_identity
from ._topology_edit import entity_keys
from .providers._native_sources import NativeLayerCoreSource, _MappedPlcRootRelation
from ._plc_source import PiecewiseLinearComplex

if TYPE_CHECKING:
    from ..discretization import CellGeometrySpec, CellMesh
    from ..geometry._source_boundary_records import SourceBoundaryChartWitness
    from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry


@final
class MappedCoreFacetAuthority(StrictModule, NonTrainableState):
    """Reconstruct selected original PLC facet charts against the retained roots.

    This owner describes source strata, not physical material boundaries. Each
    selected facet must have the same material on both sides and be a triangle
    carried by one actual root face with two cells of that material.
    """

    source: NativeLayerCoreSource
    facet_rows: tuple[int, ...] = eqx.field(static=True)
    domain: MappedReferenceDomain
    complex: PiecewiseLinearComplex
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    root_relation: _MappedPlcRootRelation
    binding_id: str = eqx.field(static=True)

    def __init__(
        self, source: NativeLayerCoreSource, facet_rows: tuple[int, ...], /
    ) -> None:
        if (
            type(source) is not NativeLayerCoreSource
            or source.reference_source is None
            or source.mapped_domain is None
        ):
            raise ValueError(
                "Internal mapped facets require their actual original core source and mapped root."
            )
        source.validate_restored()
        domain, authored = source.mapped_domain, source.reference_source
        complex_ = authored.complex
        if (
            not facet_rows
            or tuple(sorted(set(facet_rows))) != facet_rows
            or any(
                type(row) is not int or row < 0 or row >= len(complex_.facet_ids)
                for row in facet_rows
            )
        ):
            raise ValueError(
                "Internal core facet rows must be a nonempty canonical source selection."
            )
        mapping, _, polygons, _ = prepare_identity(
            authored.layers, complex_, authored.vertex_layer_ids, authored.cap_polygon_ids
        )
        root_ids = domain.reference_domain_vertex_root_ids
        if root_ids is None or any(
            int(index) < 0 or int(index) >= len(root_ids) for index in mapping
        ):
            raise ValueError(
                "Internal core facets lack complete original root-vertex ancestry."
            )
        plc_roots = np.asarray(
            [root_ids[int(index)] for index in mapping], dtype=np.int64
        )
        mesh = domain.reference_mesh
        root_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        root_rows = {int(value): row for row, value in enumerate(root_vertex_ids)}
        if len(set(plc_roots.tolist())) != len(plc_roots) or any(
            int(value) not in root_rows for value in plc_roots
        ):
            raise ValueError(
                "Internal core PLC vertices do not bind distinct root vertices."
            )
        for index, identifier in enumerate(plc_roots):
            if not np.array_equal(
                np.asarray(complex_.vertices[index]),
                np.asarray(mesh.coordinates[root_rows[int(identifier)]]),
            ):
                raise ValueError(
                    "Internal core PLC vertex changed its root reference position."
                )
            if tuple(complex_.vertex_occurrence_paths[index]) != tuple(
                domain.root_vertex_occurrence_paths[root_rows[int(identifier)]]
            ):
                raise ValueError("Internal core PLC vertex changed its root occurrence.")
        face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        face_by_key = {
            tuple(int(value) for value in key if value >= 0): int(identifier)
            for key, identifier in zip(entity_keys(mesh, 2), face_ids, strict=True)
        }
        regions = dict(
            zip(
                domain.root_cell_ids,
                np.asarray(domain.cell_regions, dtype=np.int64),
                strict=True,
            )
        )
        owners: dict[int, list[tuple[int, int, str, tuple[int, ...]]]] = {}
        for block in mesh.blocks:
            topology = reference_cell_topology(block.cell_kind)
            for identifier, rows in zip(block.global_ids, block.vertices, strict=True):
                cell = int(identifier)
                vertices = tuple(int(root_vertex_ids[int(row)]) for row in rows)
                for local, face in enumerate(topology.entities[2]):
                    key = tuple(sorted(vertices[int(index)] for index in face))
                    face_id = face_by_key.get(key)
                    if face_id is None:
                        raise ValueError(
                            "Root cell face is absent from canonical root incidence."
                        )
                    owners.setdefault(face_id, []).append(
                        (cell, local, block.cell_kind, vertices)
                    )
        polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
        rows_out: list[int] = []
        faces_out: list[int] = []
        cells_out: list[tuple[int, int]] = []
        locals_out: list[tuple[int, int]] = []
        refs_out: list[tuple[tuple[tuple[Fraction, ...], ...], ...]] = []
        signs_out: list[tuple[int, int]] = []
        from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions
        from ..geometry._planar_primitives import signed_area2

        root_points = {
            int(identifier): tuple(
                Fraction.from_float(float(value)) for value in mesh.coordinates[row]
            )
            for row, identifier in enumerate(root_vertex_ids)
        }
        for facet in facet_rows:
            sides = tuple(int(value) for value in complex_.facet_regions[facet])
            if sides[0] < 0 or sides[0] != sides[1]:
                raise ValueError(
                    "A scoped internal core facet must have identical declared material sides."
                )
            polygon_rows = np.flatnonzero(polygon_facets == facet)
            if polygon_rows.size == 0:
                raise ValueError(
                    "An internal core facet has no original triangle charts."
                )
            source_triangles = tuple(
                tuple(
                    root_points[int(plc_roots[int(value)])]
                    for value in polygons[int(polygon)]
                )
                for polygon in polygon_rows
            )
            material = (
                complex_.region_ids[sides[0]],
                complex_.region_occurrence_paths[sides[0]],
            )
            targets = []
            for face, incident in sorted(owners.items()):
                if len(incident) != 2 or any(
                    (
                        domain.region_ids[int(regions[cell])],
                        domain.reference_domain.region_occurrence_paths[
                            int(regions[cell])
                        ],
                    )
                    != material
                    for cell, _, _, _ in incident
                ):
                    continue
                cell, local, kind, vertices = min(incident, key=lambda item: item[0])
                topology = reference_cell_topology(kind)
                face_vertices = tuple(
                    vertices[int(index)] for index in topology.entities[2][local]
                )
                for index in range(1, len(face_vertices) - 1):
                    targets.append(
                        (
                            face,
                            cell,
                            local,
                            kind,
                            vertices,
                            (
                                face_vertices[0],
                                face_vertices[index],
                                face_vertices[index + 1],
                            ),
                        )
                    )
            target_triangles = tuple(
                tuple(root_points[identifier] for identifier in triangle)
                for _, _, _, _, _, triangle in targets
            )
            pieces = exact_coplanar_triangle_partitions(
                source_triangles,
                target_triangles,
                require_source=True,
                require_target=False,
            )
            for piece in pieces:
                face, cell, local, kind, vertices, triangle = targets[piece.target_row]
                topology = reference_cell_topology(kind)
                reference_vertices = tuple(
                    tuple(Fraction(value) for value in point)
                    for point in topology.vertices
                )
                triangle_reference = tuple(
                    reference_vertices[vertices.index(value)] for value in triangle
                )
                order = (0, 1, 2) if signed_area2(piece.source_chart) > 0 else (0, 2, 1)
                reference = tuple(
                    tuple(
                        triangle_reference[0][axis]
                        + piece.target_chart[index][0]
                        * (triangle_reference[1][axis] - triangle_reference[0][axis])
                        + piece.target_chart[index][1]
                        * (triangle_reference[2][axis] - triangle_reference[0][axis])
                        for axis in range(3)
                    )
                    for index in order
                )
                face_reference = tuple(
                    reference_vertices[int(index)]
                    for index in topology.entities[2][local]
                )

                def cross(
                    first: tuple[Fraction, ...], second: tuple[Fraction, ...]
                ) -> tuple[Fraction, ...]:
                    return (
                        first[1] * second[2] - first[2] * second[1],
                        first[2] * second[0] - first[0] * second[2],
                        first[0] * second[1] - first[1] * second[0],
                    )

                def difference(
                    first: tuple[Fraction, ...], second: tuple[Fraction, ...]
                ) -> tuple[Fraction, ...]:
                    return tuple(a - b for a, b in zip(first, second, strict=True))

                normal = cross(
                    difference(face_reference[1], face_reference[0]),
                    difference(face_reference[-1], face_reference[0]),
                )
                triangle_normal = cross(
                    difference(reference[1], reference[0]),
                    difference(reference[2], reference[0]),
                )
                orientation = sum(
                    (a * b for a, b in zip(normal, triangle_normal, strict=True)),
                    Fraction(0),
                )
                if orientation == 0:
                    raise ValueError(
                        "An original core triangle is degenerate in its root face."
                    )
                zero = (Fraction(0),) * 3
                reference_sides = (reference, (zero, zero, zero))
                rows_out.append(facet)
                faces_out.append(face)
                cells_out.append((cell, -1))
                locals_out.append((local, -1))
                refs_out.append(reference_sides)
                signs_out.append((1 if orientation > 0 else -1, 0))
        entity_ids = np.asarray(
            [int(complex_.facet_entity_ids[row]) for row in rows_out], dtype=np.int64
        )
        paths = tuple(tuple(complex_.facet_occurrence_paths[row]) for row in rows_out)
        keys = tuple(
            source_entity_id(
                source.source_id,
                source.source_revision,
                "face",
                str(complex_.facet_ids[row]),
                occurrence_path=path,
            )
            for row, path in zip(rows_out, paths, strict=True)
        )
        empty_i = np.empty((0,), dtype=np.int64)
        from ..geometry._source_boundary_records import rational_chart_tokens

        reference_tokens = tuple(
            tuple(rational_chart_tokens((side,))[0] for side in row) for row in refs_out
        )
        relation_payload = {
            "kind": "mapped-original-core-facet-relation",
            "source": source.binding_id,
            "domain": domain.domain_id,
            "facet_rows": facet_rows,
            "vertex_roots": plc_roots,
            "facet_faces": faces_out,
            "facet_cells": cells_out,
            "facet_locals": locals_out,
            "facet_references": reference_tokens,
            "facet_signs": signs_out,
            "facet_keys": keys,
            "facet_occurrences": paths,
        }
        relation_id = canonical_fingerprint(relation_payload)
        self.source = source
        self.facet_rows = facet_rows
        self.domain = domain
        self.complex = complex_
        self.source_id, self.source_revision = source.source_id, source.source_revision
        self.root_relation = _MappedPlcRootRelation(
            plc_roots,
            empty_i,
            empty_i,
            empty_i,
            empty_i,
            (),
            empty_i,
            np.asarray(rows_out, dtype=np.int64),
            np.asarray(faces_out, dtype=np.int64),
            np.asarray(cells_out, dtype=np.int64),
            np.asarray(locals_out, dtype=np.int64),
            reference_tokens,
            np.asarray(signs_out, dtype=np.int8),
            empty_i,
            (),
            (),
            entity_ids,
            keys,
            paths,
            (),
            relation_id,
        )
        self.binding_id = canonical_fingerprint(
            {
                "kind": "mapped-original-core-facet-authority",
                "source": source.binding_id,
                "relation": relation_id,
            }
        )

    def validate_restored(self, /) -> None:
        from ..geometry._mesh_certificates import _same_tree_content

        rebuilt = type(self)(self.source, self.facet_rows)
        if (
            rebuilt.binding_id != self.binding_id
            or self.domain.domain_id != rebuilt.domain.domain_id
            or self.complex.complex_id != rebuilt.complex.complex_id
            or (self.source_id, self.source_revision)
            != (rebuilt.source_id, rebuilt.source_revision)
            or not _same_tree_content(self.root_relation, rebuilt.root_relation)
        ):
            raise ValueError(
                "Original core facet authority no longer matches its source and roots."
            )

    def require_current(self, source: NativeLayerCoreSource, /) -> None:
        """Reconstruct the original core/root relation against the actual owner."""
        if (
            type(source) is not NativeLayerCoreSource
            or source.binding_id != self.source.binding_id
        ):
            raise ValueError("Original core facet authority binds a different source.")
        self.validate_restored()
        rebuilt = type(self)(source, self.facet_rows)
        if (
            rebuilt.binding_id != self.binding_id
            or rebuilt.root_relation.relation_id != self.root_relation.relation_id
        ):
            raise ValueError(
                "Original core facet authority changed its source or root relation."
            )

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if dimension != 2:
            return np.empty((0,), dtype=np.int64)
        return np.unique(self.root_relation.polygon_source_entity_ids)

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        if dimension != 2:
            raise ValueError("Original core facet authority only exposes face scope.")
        for identifier, key, path in zip(
            self.root_relation.polygon_source_entity_ids,
            self.root_relation.polygon_source_entity_keys,
            self.root_relation.polygon_source_occurrence_paths,
            strict=True,
        ):
            if int(identifier) == entity_id:
                return key, path
        raise ValueError(
            "Original core facet scope does not contain the requested entity."
        )

    def boundary_query(self, /) -> MappedReferenceBoundarySource:
        return MappedReferenceBoundarySource(self)


def issue_mapped_core_facet_witness(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    root: LayerCoreRootGeometry | None = None,
    /,
    *,
    query: MappedReferenceBoundarySource | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> (
    tuple[
        MappedReferenceBoundarySource,
        SourceBoundaryChartWitness,
        ValidatedSourceFidelityWitness,
    ]
    | None
):
    """Issue scoped charts by exact overlap in the independently retained root."""
    from fractions import Fraction

    from ..discretization._cell_geometry_validity import cell_geometry_id
    from ..geometry._planar_primitives import signed_area2
    from ..geometry._source_boundary_records import (
        SourceBoundaryChartWitness,
        rational_chart_tokens,
    )
    from ..geometry._source_fidelity import (
        _face_reference_chart,
        validate_source_fidelity_witness,
    )
    from ._layer_core_mapped_segment_overlay import (
        affine_mapped_reference_triangle,
        mapped_local_reference_maps,
    )
    from .providers._native_layer import _layer_output_boundary_faces
    from ..geometry._coplanar_overlap import exact_coplanar_triangle_partitions

    if source.mapped_domain is None or source.reference_source is None:
        if query is not None:
            raise ValueError(
                "A retained internal facet query requires its actual mapped core source."
            )
        return None
    authored = source.reference_source
    facet_regions = np.asarray(authored.complex.facet_regions, dtype=np.int64)
    if facet_regions.ndim != 2 or facet_regions.shape[1] != 2:
        raise ValueError("Original core facet material sides must be an N-by-2 bank.")
    facet_rows = tuple(
        row
        for row in range(facet_regions.shape[0])
        if int(facet_regions[row, 0]) >= 0
        and int(facet_regions[row, 0]) == int(facet_regions[row, 1])
        and authored.core_facet_source_ids[row] is None
    )
    if not facet_rows:
        if query is not None:
            raise ValueError(
                "A retained internal facet query no longer has original source facets."
            )
        return None
    if query is None:
        authority = MappedCoreFacetAuthority(source, facet_rows)
        query = authority.boundary_query()
    else:
        if (
            type(query) is not MappedReferenceBoundarySource
            or type(query.authority) is not MappedCoreFacetAuthority
        ):
            raise ValueError(
                "Internal facet renewal requires its original scoped core query."
            )
        authority = query.authority
        if source is not authority.source:
            source.validate_restored()
        expected_keys = tuple(
            sorted(
                set(
                    zip(
                        authority.root_relation.polygon_source_entity_keys,
                        authority.root_relation.polygon_source_occurrence_paths,
                        strict=True,
                    )
                )
            )
        )
        if (
            authority.source.binding_id != source.binding_id
            or authority.facet_rows != facet_rows
            or query.face_scope_keys != expected_keys
            or query.edge_scope_keys
        ):
            raise ValueError(
                "Internal facet renewal changed its original source or facet scope."
            )
    local_maps = mapped_local_reference_maps(source, mesh, geometry, root, computed_root=computed_root)
    owners = {cell: value[0] for cell, value in local_maps.items()}
    target_maps = {cell: value[1] for cell, value in local_maps.items()}
    source_triangles = tuple(
        tuple(tuple(Fraction(*value) for value in point) for point in reference)
        for reference in query.polygon_root_reference_rational_tokens
    )
    target_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    source_rows: list[tuple[tuple[Fraction, ...], ...]] = []
    cell_ids: list[int] = []
    face_ids: list[int] = []
    local_ids: list[int] = []
    chart_ids: list[str] = []
    entity_ids: list[str] = []
    paths: list[tuple[str, ...]] = []
    signs: list[int] = []
    output_faces = _layer_output_boundary_faces(mesh, include_internal=True)
    selected_roots = frozenset(int(value) for value in query.polygon_root_cell_global_ids)
    triangles_by_root: dict[int, list[tuple[tuple[Fraction, ...], ...]]] = {}
    metadata_by_root: dict[
        int, list[tuple[int, int, int, str, tuple[tuple[Fraction, ...], ...]]]
    ] = {}
    seen_by_root: dict[int, set[int]] = {}
    for incident in output_faces.values():
        for face, cell, local, kind, order, references in incident:
            owner = owners.get(int(cell))
            if owner not in selected_roots:
                continue
            seen = seen_by_root.setdefault(owner, set())
            if int(face) in seen:
                continue
            seen.add(int(face))
            triangles = (
                (order,)
                if len(order) == 3
                else (
                    ((order[0], order[1], order[2]), (order[0], order[2], order[3]))
                    if len(order) == 4
                    else ()
                )
            )
            for triangle in triangles:
                chart = tuple(references[vertex] for vertex in triangle)
                triangles_by_root.setdefault(owner, []).append(
                    affine_mapped_reference_triangle(target_maps, int(cell), chart)
                )
                metadata_by_root.setdefault(owner, []).append(
                    (int(face), int(cell), int(local), kind, chart)
                )
    for source_row, (source_triangle, source_root) in enumerate(
        zip(source_triangles, query.polygon_root_cell_global_ids, strict=True)
    ):
        target_triangles = triangles_by_root.get(int(source_root), ())
        metadata = metadata_by_root.get(int(source_root), ())
        pieces = exact_coplanar_triangle_partitions(
            (source_triangle,),
            tuple(target_triangles),
            require_source=True,
            require_target=False,
        )
        for piece in pieces:
            face, cell, local, kind, chart = metadata[piece.target_row]
            target = tuple(
                tuple(
                    chart[0][axis]
                    + uv[0] * (chart[1][axis] - chart[0][axis])
                    + uv[1] * (chart[2][axis] - chart[0][axis])
                    for axis in range(3)
                )
                for uv in piece.target_chart
            )
            projected, _ = _face_reference_chart(kind, local, target)
            source_rows.append(piece.source_chart)
            target_rows.append(target)
            cell_ids.append(cell)
            face_ids.append(face)
            local_ids.append(local)
            chart_ids.append(query.face_chart_ids[source_row])
            entity_ids.append(query.polygon_source_entity_keys[source_row])
            paths.append(query.polygon_source_occurrence_paths[source_row])
            signs.append(
                (1 if signed_area2(piece.source_chart) > 0 else -1)
                * (1 if signed_area2(projected[0]) > 0 else -1)
            )
    if not target_rows:
        raise ValueError(
            "Original internal core facets have no exact target face overlap."
        )
    witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=query.source_id,
        source_revision=query.source_revision,
        source_query_id=query.source_query_id,
        chart_dimension=2,
        reference_dimension=3,
        cell_global_ids=np.asarray(cell_ids, dtype=np.int64),
        entity_global_ids=np.asarray(face_ids, dtype=np.int64),
        local_entity_indices=np.asarray(local_ids, dtype=np.int32),
        target_reference_vertices=np.asarray(target_rows, dtype=np.float64),
        target_reference_rational_tokens=rational_chart_tokens(target_rows),
        source_chart_ids=tuple(chart_ids),
        source_entity_ids=tuple(entity_ids),
        occurrence_paths=tuple(paths),
        source_parameters=np.asarray(source_rows, dtype=np.float64),
        source_parameter_rational_tokens=rational_chart_tokens(source_rows),
        orientations=np.asarray(signs, dtype=np.int8),
    )
    token = validate_source_fidelity_witness(mesh, geometry, query, witness)
    return query, witness, token

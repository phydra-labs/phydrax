# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact periodic mapped refinements and inverse common-subcomplex proofs."""

from __future__ import annotations

from fractions import Fraction
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    affine_map,
    RationalPolynomial,
)
from ..discretization._nested_reference import (
    _reference_shape_weights,
    nested_cell_reference_maps,
)
from ..discretization._nested_reference_types import (
    exact_reference_vertices,
    NestedReferenceWitnesses,
    ReferenceCellMapWitnesses,
    validate_nested_reference_witnesses,
    validate_reference_cell_map_witnesses,
)
from ..discretization._periodic_exact import (
    identification_id,
    identification_orders,
    reduce_periodic_exponents,
)
from ..discretization._periodic_topology import (
    _entity_vertex_rows,
    PeriodicMeshTopology,
)
from ..discretization._reference_cell import (
    reference_cell_topology,
    ReferenceCellTopology,
)
from ..typing import parse
from ._embedding_state import PeriodicPartitionDirection
from ._periodic_restriction import (
    _convex_cell,
    _ConvexCell,
    _parent_partition,
    _restriction_planes,
)


if TYPE_CHECKING:
    from ._domain_coverage import PiecewiseLinearDomain
    from ._global_embedding import GlobalEmbeddingCertificate
    from ._mapped_reference_domain import MappedReferenceDomain


type _Key = tuple[tuple[int, Fraction], ...]
type _Point = tuple[Fraction, Fraction, Fraction]


class _MappedPiece(NamedTuple):
    fine_cell_id: int
    coarse_cell_id: int
    fine_cell_kind: str
    coarse_cell_kind: str
    coarse_reference_vertices: tuple[tuple[Fraction, ...], ...]


class _MappedRestrictionUnavailable(ValueError):
    """Exact mapped partition premises do not establish the requested theorem."""


def _cells(mesh: CellMesh) -> dict[int, tuple[str, tuple[int, ...]]]:
    result: dict[int, tuple[str, tuple[int, ...]]] = {}
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for block in mesh.blocks:
        if block.cell_kind in ("polygon", "polyhedron"):
            return {}
        topology = reference_cell_topology(block.cell_kind)
        rows = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        identifiers = np.asarray(block.global_ids, dtype=np.int64)
        if (
            topology.dimension != 3
            or rows.shape[1] != len(topology.vertices)
            or not np.all(valid)
        ):
            return {}
        for identifier, row in zip(identifiers.tolist(), rows.tolist(), strict=True):
            cell_id = int(identifier)
            if cell_id in result:
                return {}
            result[cell_id] = (
                block.cell_kind,
                tuple(int(vertex_ids[int(vertex)]) for vertex in row),
            )
    return result


def _reference_points(cell_kind: str) -> tuple[_Point, ...] | None:
    topology = reference_cell_topology(cell_kind)
    if topology.dimension != 3:
        return None
    result = []
    for point in topology.vertices:
        if len(point) != 3:
            return None
        result.append(
            (
                Fraction(float(point[0])),
                Fraction(float(point[1])),
                Fraction(float(point[2])),
            )
        )
    return tuple(result)


def _reference_weight_key(
    cell_kind: str,
    parent_vertices: tuple[int, ...],
    point: _Point,
) -> _Key | None:
    if (
        cell_kind == "pyramid"
        and point[2] == 1
        and (point[0], point[1]) != (Fraction(1, 2), Fraction(1, 2))
    ):
        return None
    weights = _reference_shape_weights(cell_kind, point)
    if (
        len(weights) != len(parent_vertices)
        or any(value < 0 for value in weights)
        or sum(weights, Fraction(0)) != 1
    ):
        return None
    return tuple(
        sorted(
            (identifier, weight)
            for identifier, weight in zip(parent_vertices, weights, strict=True)
            if weight
        )
    )


def _physical_boundary(mesh: CellMesh) -> set[frozenset[int]] | None:
    topology = mesh.periodic_topology
    if topology is None or topology.topological_dimension != 3:
        return None
    quotient_faces = topology.quotient.entity_sets[2]
    boundary = np.asarray(quotient_faces.subset("boundary").mask, dtype=np.bool_)
    quotient_offsets = np.asarray(topology.quotient_offsets, dtype=np.int64)
    representatives = np.asarray(topology.representatives, dtype=np.int64)
    lifted_face_rows = _entity_vertex_rows(mesh)[2]
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    first = int(quotient_offsets[2])
    stop = int(quotient_offsets[3])
    if boundary.shape != (stop - first,) or representatives.size < stop:
        return None
    result: set[frozenset[int]] = set()
    for row in np.flatnonzero(boundary):
        lifted_row = int(representatives[first + int(row)])
        if lifted_row < 0 or lifted_row >= len(lifted_face_rows):
            return None
        result.add(
            frozenset(
                int(vertex_ids[int(vertex)]) for vertex in lifted_face_rows[lifted_row]
            )
        )
    return result


def _reference_parent_tiling(
    parents: dict[int, _ConvexCell],
    children: tuple[tuple[int, _ConvexCell], ...],
) -> bool:
    """Prove every reference parent is exactly the union of its children."""
    grouped: dict[int, list[tuple[int, _ConvexCell]]] = {
        identifier: [] for identifier in parents
    }
    for parent_id, child in children:
        if parent_id not in grouped:
            return False
        grouped[parent_id].append((parent_id, child))
    for parent_id, parent in parents.items():
        local_children = tuple(grouped[parent_id])
        if not local_children:
            return False
        planes = _restriction_planes({parent_id: parent}, local_children)
        if planes is None:
            if len(local_children) != 1:
                return False
            planes = ()
        if _parent_partition({parent_id: parent}, local_children, planes) is None:
            return False
    return True


def _nested_pieces(
    witness: NestedReferenceWitnesses,
    fine_cells: dict[int, tuple[str, tuple[int, ...]]],
    coarse_cells: dict[int, tuple[str, tuple[int, ...]]],
) -> tuple[_MappedPiece, ...] | None:
    validate_nested_reference_witnesses(witness, dimension=3)
    fine_ids = np.asarray(witness.fine_cell_ids, dtype=np.int64)
    coarse_ids = np.asarray(witness.coarse_cell_ids, dtype=np.int64)
    exact = exact_reference_vertices(witness)
    pieces = []
    for fine_id_value, coarse_id_value, corners in zip(
        fine_ids.tolist(), coarse_ids.tolist(), exact, strict=True
    ):
        fine_id, coarse_id = int(fine_id_value), int(coarse_id_value)
        fine_cell = fine_cells.get(fine_id)
        coarse_cell = coarse_cells.get(coarse_id)
        if fine_cell is None or coarse_cell is None:
            return None
        if len(corners) != len(reference_cell_topology(fine_cell[0]).vertices):
            return None
        pieces.append(
            _MappedPiece(
                fine_id,
                coarse_id,
                fine_cell[0],
                coarse_cell[0],
                corners,
            )
        )
    return tuple(pieces)


def _reference_map_groups(
    value: ReferenceCellMapWitnesses | tuple[ReferenceCellMapWitnesses, ...] | None,
) -> tuple[ReferenceCellMapWitnesses, ...] | None:
    if isinstance(value, ReferenceCellMapWitnesses):
        return (value,)
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(group, ReferenceCellMapWitnesses) for group in value)
    ):
        return None
    return value


def _reference_map_pieces(
    groups: tuple[ReferenceCellMapWitnesses, ...],
    fine_cells: dict[int, tuple[str, tuple[int, ...]]],
    coarse_cells: dict[int, tuple[str, tuple[int, ...]]],
) -> tuple[_MappedPiece, ...] | None:
    pieces = []
    for group in groups:
        validate_reference_cell_map_witnesses(group)
        for mapped in nested_cell_reference_maps(group):
            fine_cell = fine_cells.get(mapped.target_cell_id)
            coarse_cell = coarse_cells.get(mapped.source_cell_id)
            if (
                fine_cell is None
                or coarse_cell is None
                or mapped.target_cell_kind != fine_cell[0]
                or mapped.source_cell_kind != coarse_cell[0]
                or mapped.arguments is None
            ):
                return None
            arguments = tuple(
                argument if isinstance(argument, RationalPolynomial) else dict(argument)
                for argument in mapped.arguments
            )
            affine = affine_map(arguments, 3)
            if affine is None:
                return None
            offset, matrix = affine
            if any(
                not isinstance(value, Fraction)
                for value in (*offset, *(entry for row in matrix for entry in row))
            ):
                return None
            pieces.append(
                _MappedPiece(
                    mapped.target_cell_id,
                    mapped.source_cell_id,
                    mapped.target_cell_kind,
                    mapped.source_cell_kind,
                    mapped.source_reference_vertices,
                )
            )
    return tuple(pieces)


def _transition_pieces(
    transition: CellGeometryTransition,
) -> tuple[PeriodicPartitionDirection, tuple[_MappedPiece, ...]] | None:
    source_cells = _cells(transition.source_mesh)
    target_cells = _cells(transition.target_mesh)
    if not source_cells or not target_cells:
        return None
    if transition.refinement is not None and transition.coarsening is not None:
        return None
    if transition.refinement is not None:
        pieces = _nested_pieces(transition.refinement, target_cells, source_cells)
        return None if pieces is None else ("refine", pieces)
    if transition.coarsening is not None:
        pieces = _nested_pieces(transition.coarsening, source_cells, target_cells)
        return None if pieces is None else ("coarsen", pieces)
    groups = _reference_map_groups(transition.reference_map)
    if groups is None:
        return None
    pieces = _reference_map_pieces(groups, target_cells, source_cells)
    return None if pieces is None else ("refine", pieces)


def _reference_parent_data(
    coarse_cells: dict[int, tuple[str, tuple[int, ...]]],
) -> (
    tuple[
        dict[int, _ConvexCell],
        dict[int, ReferenceCellTopology],
        dict[frozenset[int], set[int]],
    ]
    | None
):
    parents: dict[int, _ConvexCell] = {}
    topologies: dict[int, ReferenceCellTopology] = {}
    incident: dict[frozenset[int], set[int]] = {}
    for identifier, (cell_kind, vertices) in coarse_cells.items():
        topology = reference_cell_topology(cell_kind)
        reference = _reference_points(cell_kind)
        if topology.dimension != 3 or len(topology.entities) < 3 or reference is None:
            return None
        faces = tuple(
            tuple(int(vertex) for vertex in face) for face in topology.entities[2]
        )
        parent = _convex_cell(identifier, faces, dict(enumerate(reference)))
        if parent is None:
            return None
        parents[identifier] = parent
        topologies[identifier] = topology
        for level in topology.entities:
            for entity in level:
                support = frozenset(vertices[int(local)] for local in entity)
                incident.setdefault(support, set()).add(identifier)
    return parents, topologies, incident


def _reference_piece_keys(
    coarse_cells: dict[int, tuple[str, tuple[int, ...]]],
    fine_cells: dict[int, tuple[str, tuple[int, ...]]],
    pieces: tuple[_MappedPiece, ...],
    parent_topologies: dict[int, ReferenceCellTopology],
    incident: dict[frozenset[int], set[int]],
    coarse_vertex_ids: set[int],
    fine_vertex_ids: set[int],
) -> tuple[dict[int, _Key], tuple[tuple[int, _ConvexCell], ...]] | None:
    keys: dict[int, _Key] = {}
    key_vertices: dict[_Key, int] = {}
    key_parents: dict[_Key, set[int]] = {}
    children: list[tuple[int, _ConvexCell]] = []
    for piece in pieces:
        parent_id, child_id = piece.coarse_cell_id, piece.fine_cell_id
        parent_cell = coarse_cells.get(parent_id)
        fine_cell = fine_cells.get(child_id)
        parent_topology = parent_topologies.get(parent_id)
        if (
            parent_cell is None
            or fine_cell is None
            or parent_topology is None
            or piece.coarse_cell_kind != parent_cell[0]
            or piece.fine_cell_kind != fine_cell[0]
        ):
            return None
        parent_vertices = parent_cell[1]
        fine_vertices = fine_cell[1]
        fine_topology = reference_cell_topology(piece.fine_cell_kind)
        if len(piece.coarse_reference_vertices) != len(fine_topology.vertices) or len(
            fine_vertices
        ) != len(fine_topology.vertices):
            return None
        local_points: dict[int, _Point] = {}
        for local, (identifier, point) in enumerate(
            zip(fine_vertices, piece.coarse_reference_vertices, strict=True)
        ):
            if len(point) != 3 or any(not isinstance(value, Fraction) for value in point):
                return None
            exact_point = (point[0], point[1], point[2])
            local_points[local] = exact_point
            key = _reference_weight_key(
                piece.coarse_cell_kind, parent_vertices, exact_point
            )
            if key is None:
                return None
            support = frozenset(identifier_ for identifier_, _ in key)
            if support not in incident:
                return None
            previous = keys.setdefault(identifier, key)
            if previous != key or key_vertices.setdefault(key, identifier) != identifier:
                return None
            if len(key) == 1 and (key[0][1] != 1 or identifier != key[0][0]):
                return None
            key_parents.setdefault(key, set()).add(parent_id)
        faces = tuple(
            tuple(int(vertex) for vertex in face) for face in fine_topology.entities[2]
        )
        child = _convex_cell(child_id, faces, local_points)
        if child is None:
            return None
        children.append((parent_id, child))

    if set(keys) != fine_vertex_ids:
        return None
    if any(
        keys.get(identifier) != ((identifier, Fraction(1)),)
        for identifier in coarse_vertex_ids
    ):
        return None
    for key, parents_for_key in key_parents.items():
        support = frozenset(identifier for identifier, _ in key)
        if parents_for_key != incident[support]:
            return None
    return keys, tuple(children)


def _periodic_vertex_keys_match(
    coarse: CellMesh,
    fine: CellMesh,
    keys: dict[int, _Key],
) -> bool:
    coarse_periodic, fine_periodic = coarse.periodic_topology, fine.periodic_topology
    if (
        not isinstance(coarse_periodic, PeriodicMeshTopology)
        or not isinstance(fine_periodic, PeriodicMeshTopology)
        or identification_id(coarse_periodic.cell)
        != identification_id(fine_periodic.cell)
    ):
        return False
    coarse_ids = np.asarray(coarse.vertex_global_ids, dtype=np.int64)
    fine_ids = np.asarray(fine.vertex_global_ids, dtype=np.int64)
    coarse_rows = {int(identifier): row for row, identifier in enumerate(coarse_ids)}
    coarse_roots = np.asarray(coarse_periodic.vertex_representatives, dtype=np.int64)
    coarse_shifts = np.asarray(coarse_periodic.vertex_shifts, dtype=np.int64)
    coarse_fixed = np.asarray(coarse_periodic.vertex_fixed_generators, dtype=np.bool_)
    fine_roots = np.asarray(fine_periodic.vertex_representatives, dtype=np.int64)
    fine_shifts = np.asarray(fine_periodic.vertex_shifts, dtype=np.int64)
    orders = identification_orders(coarse_periodic.cell)
    quotient_keys: dict[int, tuple[tuple[int, tuple[int, ...], Fraction], ...]] = {}
    quotient_vertices: dict[tuple[tuple[int, tuple[int, ...], Fraction], ...], int] = {}
    for row, identifier_value in enumerate(fine_ids):
        identifier = int(identifier_value)
        normalized_values = []
        for coarse_identifier, weight in keys[identifier]:
            coarse_row = coarse_rows.get(coarse_identifier)
            if coarse_row is None:
                return False
            winding = np.asarray(
                reduce_periodic_exponents(
                    coarse_shifts[coarse_row] - fine_shifts[row], orders
                ),
                dtype=np.int64,
            ).copy()
            winding[coarse_fixed[coarse_row]] = 0
            normalized_values.append(
                (
                    int(coarse_ids[coarse_roots[coarse_row]]),
                    tuple(int(value) for value in winding),
                    weight,
                )
            )
        normalized = tuple(sorted(normalized_values))
        representative = int(fine_ids[fine_roots[row]])
        if (
            quotient_keys.setdefault(representative, normalized) != normalized
            or quotient_vertices.setdefault(normalized, representative) != representative
        ):
            return False
    return True


def _physical_boundary_record(
    coarse: CellMesh,
    fine: CellMesh,
    keys: dict[int, _Key],
) -> tuple[tuple[int, ...], ...] | None:
    physical_coarse = _physical_boundary(coarse)
    physical_fine = _physical_boundary(fine)
    if physical_coarse is None or physical_fine is None:
        return None
    for face in physical_fine:
        support = frozenset(
            identifier for vertex in face for identifier, _ in keys[vertex]
        )
        if not any(support <= coarse_face for coarse_face in physical_coarse):
            return None
    return tuple(sorted(tuple(sorted(face)) for face in physical_fine))


def _reference_partition(
    coarse: CellMesh,
    fine: CellMesh,
    pieces: tuple[_MappedPiece, ...],
    direction: PeriodicPartitionDirection,
) -> str | None:
    """Prove exact oriented parent tilings and all shared lower strata."""
    coarse_cells = _cells(coarse)
    fine_cells = _cells(fine)
    if not coarse_cells or not fine_cells:
        return None
    declared = {piece.fine_cell_id: piece.coarse_cell_id for piece in pieces}
    if (
        len(declared) != len(pieces)
        or set(declared) != set(fine_cells)
        or not set(declared.values()) <= set(coarse_cells)
    ):
        return None
    if set(declared.values()) != set(coarse_cells):
        return None
    parent_data = _reference_parent_data(coarse_cells)
    if parent_data is None:
        return None
    parents, parent_topologies, incident = parent_data
    keys_and_children = _reference_piece_keys(
        coarse_cells,
        fine_cells,
        pieces,
        parent_topologies,
        incident,
        set(np.asarray(coarse.vertex_global_ids, dtype=np.int64).tolist()),
        set(np.asarray(fine.vertex_global_ids, dtype=np.int64).tolist()),
    )
    if keys_and_children is None:
        return None
    keys, children = keys_and_children
    if not _reference_parent_tiling(parents, children):
        return None
    if not _periodic_vertex_keys_match(coarse, fine, keys):
        return None
    boundary = _physical_boundary_record(coarse, fine, keys)
    if boundary is None:
        return None
    records = tuple(
        (
            identifier,
            tuple(
                (vertex, weight.numerator, weight.denominator) for vertex, weight in key
            ),
        )
        for identifier, key in sorted(keys.items())
    )
    reference_pieces = tuple(
        (
            piece.fine_cell_id,
            piece.coarse_cell_id,
            piece.fine_cell_kind,
            piece.coarse_cell_kind,
            tuple(
                tuple((value.numerator, value.denominator) for value in point)
                for point in piece.coarse_reference_vertices
            ),
        )
        for piece in sorted(pieces, key=lambda value: value.fine_cell_id)
    )
    return canonical_fingerprint(
        {
            "kind": "periodic-mapped-common-subcomplex",
            "vertices": records,
            "reference_pieces": reference_pieces,
            "physical_boundary": boundary,
            "direction": direction,
        }
    )


def _regions(values: tuple[int, ...] | None, count: int) -> tuple[int, ...] | None:
    if values is None or not isinstance(values, tuple) or len(values) != count:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in values
    ):
        return None
    return tuple(int(value) for value in values)


def _region_by_cell(mesh: CellMesh, regions: tuple[int, ...]) -> dict[int, int]:
    result: dict[int, int] = {}
    offset = 0
    for block in mesh.blocks:
        for identifier in np.asarray(block.global_ids, dtype=np.int64).tolist():
            result[int(identifier)] = regions[offset]
            offset += 1
    return result


def _mapped_restriction_identity(
    transition: CellGeometryTransition,
    source_validity: CellValidityCertificate,
    source_embedding: GlobalEmbeddingCertificate,
    validity: CellValidityCertificate,
    source_regions: tuple[int, ...],
    target_regions: tuple[int, ...],
    source_domain: PiecewiseLinearDomain | MappedReferenceDomain,
    region_namespace_id: str,
    facet_trace_certificate_id: str,
    partition_id: str,
    subcomplex_id: str,
    direction: PeriodicPartitionDirection,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "periodic-mapped-restriction-evidence",
            "source_mesh": transition.source_mesh.mesh_id,
            "source_geometry": cell_geometry_id(transition.source_geometry),
            "source_validity": source_validity.certificate_id,
            "source_embedding": source_embedding.certificate_id,
            "target_mesh": transition.target_mesh.mesh_id,
            "target_geometry": cell_geometry_id(transition.target_geometry),
            "target_validity": validity.certificate_id,
            "transition": transition.transition_id,
            "source_regions": source_regions,
            "target_regions": target_regions,
            "source_domain": source_domain.domain_id,
            "region_namespace": region_namespace_id,
            "facet_traces": facet_trace_certificate_id,
            "partition": partition_id,
            "subcomplex": subcomplex_id,
            "direction": direction,
        }
    )


def _prove(
    transition: CellGeometryTransition,
    source_validity: CellValidityCertificate,
    source_embedding: GlobalEmbeddingCertificate,
    validity: CellValidityCertificate,
    source_regions: tuple[int, ...],
    target_regions: tuple[int, ...],
    source_domain: PiecewiseLinearDomain | MappedReferenceDomain,
    region_namespace_id: str,
    facet_trace_certificate_id: str,
) -> tuple[str, str, PeriodicPartitionDirection] | None:
    from ._domain_coverage import certify_domain_coverage
    from ._embedding_cells import prepare_embedding_cells
    from ._global_embedding import (
        _periodic_facet_trace_certificate_id,
        GlobalEmbeddingCertificate,
    )

    source_mesh, source_geometry = transition.source_mesh, transition.source_geometry
    target_mesh, target_geometry = transition.target_mesh, transition.target_geometry
    transition.require_bound(source_mesh, source_geometry, target_mesh, target_geometry)
    relation = _transition_pieces(transition)
    if relation is None:
        return None
    direction, pieces = relation
    first, second = source_mesh.periodic_topology, target_mesh.periodic_topology
    if (
        not isinstance(first, PeriodicMeshTopology)
        or not isinstance(second, PeriodicMeshTopology)
        or source_mesh.topological_dimension != 3
        or source_mesh.ambient_dimension != 3
        or target_mesh.topological_dimension != 3
        or target_mesh.ambient_dimension != 3
        or identification_id(first.cell) != identification_id(second.cell)
        or source_embedding.status != "certified"
        or source_embedding.findings
        or not source_validity.all_certified
        or not validity.all_certified
        or not facet_trace_certificate_id
    ):
        return None
    target_cells = prepare_embedding_cells(target_mesh, target_geometry)
    current_trace_id = _periodic_facet_trace_certificate_id(
        target_mesh, target_geometry, target_cells, second
    )
    if current_trace_id is None or current_trace_id != facet_trace_certificate_id:
        return None
    source_validity.require_current(source_geometry, mesh=source_mesh)
    validity.require_current(target_geometry, mesh=target_mesh)
    source_certificate = GlobalEmbeddingCertificate.from_stored_fields(source_embedding)
    source_certificate.binding.require(source_mesh, source_geometry)
    if (
        source_certificate.status != "certified"
        or source_certificate.findings
        or source_certificate.validity_certificate_id != source_validity.certificate_id
        or source_certificate.periodic is None
        or source_certificate.periodic.periodic_topology_id != first.periodic_topology_id
    ):
        return None
    if source_domain.domain_id != region_namespace_id:
        return None
    source_count = sum(block.cell_count for block in source_mesh.blocks)
    target_count = sum(block.cell_count for block in target_mesh.blocks)
    source_regions_ = _regions(source_regions, source_count)
    target_regions_ = _regions(target_regions, target_count)
    if source_regions_ is None or target_regions_ is None:
        return None
    coverage = certify_domain_coverage(
        source_mesh,
        source_geometry,
        source_domain,
        np.asarray(source_regions_, dtype=np.int64),
        embedding=source_certificate,
    )
    if coverage.status != "certified" or coverage.findings:
        return None
    inverse = direction == "coarsen"
    coarse, fine = (target_mesh, source_mesh) if inverse else (source_mesh, target_mesh)
    coarse_regions, fine_regions = (
        (target_regions_, source_regions_)
        if inverse
        else (source_regions_, target_regions_)
    )
    coarse_by_id = _region_by_cell(coarse, coarse_regions)
    fine_by_id = _region_by_cell(fine, fine_regions)
    for piece in pieces:
        if fine_by_id.get(piece.fine_cell_id) != coarse_by_id.get(piece.coarse_cell_id):
            return None
    subcomplex_id = _reference_partition(coarse, fine, pieces, direction)
    if subcomplex_id is None:
        return None
    partition_id = canonical_fingerprint(
        {
            "kind": "periodic-mapped-oriented-reference-partition",
            "transition": transition.transition_id,
            "source_embedding": source_certificate.certificate_id,
            "source_coverage": coverage.certificate_id,
            "derived_target_regions": target_regions_,
            "namespace": region_namespace_id,
            "facet_trace": facet_trace_certificate_id,
            "direction": direction,
            "pieces": tuple(
                (
                    piece.fine_cell_id,
                    piece.coarse_cell_id,
                    piece.fine_cell_kind,
                    piece.coarse_cell_kind,
                    tuple(
                        tuple((value.numerator, value.denominator) for value in point)
                        for point in piece.coarse_reference_vertices
                    ),
                )
                for piece in sorted(pieces, key=lambda value: value.fine_cell_id)
            ),
        }
    )
    return partition_id, subcomplex_id, direction


@final
class PeriodicMappedRestrictionEvidence(StrictModule, NonTrainableState):
    """Retain current endpoint owners and the re-proved periodic subdivision."""

    transition: CellGeometryTransition
    source_validity: CellValidityCertificate
    source_embedding: GlobalEmbeddingCertificate
    validity: CellValidityCertificate
    source_regions: tuple[int, ...] = eqx.field(static=True)
    target_regions: tuple[int, ...] = eqx.field(static=True)
    source_domain: PiecewiseLinearDomain | MappedReferenceDomain
    region_namespace_id: str = eqx.field(static=True)
    facet_trace_certificate_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    subcomplex_id: str = eqx.field(static=True)
    direction: PeriodicPartitionDirection = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: CellGeometryTransition,
        source_validity: CellValidityCertificate,
        source_embedding: GlobalEmbeddingCertificate,
        validity: CellValidityCertificate,
        source_regions: tuple[int, ...],
        target_regions: tuple[int, ...],
        source_domain: PiecewiseLinearDomain | MappedReferenceDomain,
        region_namespace_id: str,
        facet_trace_certificate_id: str,
        /,
    ) -> None:
        from ._domain_coverage import PiecewiseLinearDomain
        from ._global_embedding import GlobalEmbeddingCertificate
        from ._mapped_reference_domain import MappedReferenceDomain

        if not isinstance(transition, CellGeometryTransition):
            raise TypeError("transition must be CellGeometryTransition.")
        if not isinstance(source_validity, CellValidityCertificate):
            raise TypeError("source_validity must be CellValidityCertificate.")
        if not isinstance(source_embedding, GlobalEmbeddingCertificate):
            raise TypeError("source_embedding must be GlobalEmbeddingCertificate.")
        if not isinstance(validity, CellValidityCertificate):
            raise TypeError("validity must be CellValidityCertificate.")
        if not isinstance(source_domain, (PiecewiseLinearDomain, MappedReferenceDomain)):
            raise TypeError(
                "source_domain must be an exact PiecewiseLinearDomain or MappedReferenceDomain."
            )
        source_mesh = transition.source_mesh
        target_mesh = transition.target_mesh
        source_regions_ = _regions(
            source_regions, sum(block.cell_count for block in source_mesh.blocks)
        )
        target_regions_ = _regions(
            target_regions, sum(block.cell_count for block in target_mesh.blocks)
        )
        if source_regions_ is None or target_regions_ is None:
            raise ValueError(
                "Mapped restriction region labels must align with endpoint cells."
            )
        namespace = canonical_identifier(region_namespace_id, "region_namespace_id")
        trace_id = canonical_identifier(
            facet_trace_certificate_id, "facet_trace_certificate_id"
        )
        if source_domain.domain_id != namespace:
            raise ValueError(
                "region_namespace_id must bind the retained exact source domain."
            )
        canonical_transition = CellGeometryTransition.from_stored_fields(transition)
        proved = _prove(
            canonical_transition,
            source_validity,
            source_embedding,
            validity,
            source_regions_,
            target_regions_,
            source_domain,
            namespace,
            trace_id,
        )
        if proved is None:
            raise _MappedRestrictionUnavailable(
                "Exact periodic mapped partition or common-subcomplex proof is unresolved."
            )
        partition_id, subcomplex_id, direction = proved
        self.transition = canonical_transition
        self.source_validity = source_validity
        self.source_embedding = source_embedding
        self.validity = validity
        self.source_regions = source_regions_
        self.target_regions = target_regions_
        self.source_domain = source_domain
        self.region_namespace_id = namespace
        self.facet_trace_certificate_id = trace_id
        self.partition_id = canonical_identifier(partition_id, "partition_id")
        self.subcomplex_id = canonical_identifier(subcomplex_id, "subcomplex_id")
        self.direction = parse(direction, PeriodicPartitionDirection, "direction")
        self.evidence_id = _mapped_restriction_identity(
            self.transition,
            self.source_validity,
            self.source_embedding,
            self.validity,
            self.source_regions,
            self.target_regions,
            self.source_domain,
            self.region_namespace_id,
            self.facet_trace_certificate_id,
            self.partition_id,
            self.subcomplex_id,
            self.direction,
        )

    @classmethod
    def from_stored_fields(
        cls: type[PeriodicMappedRestrictionEvidence],
        value: PeriodicMappedRestrictionEvidence,
        /,
    ) -> PeriodicMappedRestrictionEvidence:
        if type(value) is not cls:
            raise TypeError("value must be PeriodicMappedRestrictionEvidence.")
        rebuilt = cls(
            value.transition,
            value.source_validity,
            value.source_embedding,
            value.validity,
            value.source_regions,
            value.target_regions,
            value.source_domain,
            value.region_namespace_id,
            value.facet_trace_certificate_id,
        )
        if (
            rebuilt.partition_id != value.partition_id
            or rebuilt.subcomplex_id != value.subcomplex_id
            or rebuilt.direction != value.direction
            or rebuilt.evidence_id != value.evidence_id
        ):
            raise ValueError(
                "Stored PeriodicMappedRestrictionEvidence does not reproduce its current proof."
            )
        return rebuilt


def certify_periodic_mapped_restriction(
    transition: CellGeometryTransition | None,
    source_validity: CellValidityCertificate | None,
    source_embedding: GlobalEmbeddingCertificate | None,
    validity: CellValidityCertificate | None,
    source_regions: tuple[int, ...] | None,
    target_regions: tuple[int, ...] | None,
    source_domain: PiecewiseLinearDomain | MappedReferenceDomain | None,
    region_namespace_id: str | None,
    facet_trace_certificate_id: str | None,
) -> PeriodicMappedRestrictionEvidence | None:
    """Certify exact periodic mapped refinement or inverse coarsening evidence."""
    if (
        transition is None
        or source_validity is None
        or source_embedding is None
        or validity is None
        or source_regions is None
        or target_regions is None
        or source_domain is None
        or region_namespace_id is None
        or facet_trace_certificate_id is None
    ):
        return None
    try:
        return PeriodicMappedRestrictionEvidence(
            transition,
            source_validity,
            source_embedding,
            validity,
            source_regions,
            target_regions,
            source_domain,
            region_namespace_id,
            facet_trace_certificate_id,
        )
    except _MappedRestrictionUnavailable:
        return None


__all__ = [
    "PeriodicMappedRestrictionEvidence",
    "certify_periodic_mapped_restriction",
]

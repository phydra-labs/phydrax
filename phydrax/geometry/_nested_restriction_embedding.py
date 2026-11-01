#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Global embedding of a nested simplicial restriction from its certified source.

Theorem (nested restriction). Let ``S`` be a simplicial mesh whose exact
coordinate map ``F_S`` carries a certified global embedding certificate: every
cell map is injective and distinct cells meet exactly in their shared mesh
entities. Let ``T`` be a simplicial mesh of the same dimension and let a
certified exact :class:`CellGeometryTransition` bind ``(S, F_S)`` to
``(T, F_T)`` by affine reference maps ``A_t`` (one parent ``p(t)`` per target
cell) with ``F_T|_t = F_S|_{p(t)} o A_t``. Suppose:

1. every ``A_t`` has positive determinant;
2. for each parent ``p`` the determinants of its children sum to the parent's
   reference volume (``sum det A_t = 1`` on the unit simplex);
3. every oriented child facet whose corners do not lie in one facet of the
   parent reference cell occurs exactly twice among the children of that
   parent, with opposite induced orientations;
4. every other child facet lies in exactly one parent facet ``f``; it occurs
   once for each target-bearing source cell incident to ``f``, once per such
   cell;
5. each target vertex has one exact source position, its nonzero barycentric
   weights on source vertex IDs, the same from every parent of every incident
   child, and distinct target vertices have distinct positions.

Then ``F_T`` is a global embedding of ``T``. By (3) the children of ``p`` form
an oriented chain whose boundary lies in the parent's boundary, so their
affine images have one degree on the open parent cell and degree zero
outside; positive determinants (1) and the volume identity (2) make that
degree one, hence the children tile the parent with disjoint interiors and,
again by (3), meet facet to facet. Conditions (4) and (5) make the induced
subdivisions of every shared source entity identical, so target cells in
different parents meet exactly in shared target entities inside the shared
source entity, on which ``F_S`` is injective. Each ``F_T|_t`` is injective as
the composition of an injective affine map and the injective ``F_S|_p``.

The certifier answers ``None`` when a premise does not hold; the caller then
proves the target from scratch, so the verdict is never weakened.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_transition_types import (
    _nested_reference_groups,
    _reference_map_groups,
    CellGeometryTransition,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._nested_reference_types import exact_reference_cell_vertices
from ..discretization._reference_cell import reference_cell_topology


if TYPE_CHECKING:
    from ._global_embedding import GlobalEmbeddingCertificate


NESTED_RESTRICTION_ROUTE = "nested-restriction"
_SIMPLEX_KINDS = {"interval": 1, "triangle": 2, "tetrahedron": 3}
_ZERO = Fraction(0)
_ONE = Fraction(1)


class NestedRestrictionPremise(StrictModule):
    """A certified source embedding and the exact transition to the target."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    source_embedding: GlobalEmbeddingCertificate
    transition: CellGeometryTransition

    def __init__(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        source_embedding: GlobalEmbeddingCertificate,
        transition: CellGeometryTransition,
        /,
    ) -> None:
        from ._global_embedding import GlobalEmbeddingCertificate

        if not isinstance(source_mesh, CellMesh):
            raise TypeError("source_mesh must be CellMesh.")
        if not isinstance(source_geometry, CellGeometrySpec):
            raise TypeError("source_geometry must be CellGeometrySpec.")
        if not isinstance(source_embedding, GlobalEmbeddingCertificate):
            raise TypeError("source_embedding must be GlobalEmbeddingCertificate.")
        if not isinstance(transition, CellGeometryTransition):
            raise TypeError("transition must be CellGeometryTransition.")
        source_embedding.require_current(source_mesh, source_geometry)
        self.source_mesh = source_mesh
        self.source_geometry = source_geometry
        self.source_embedding = source_embedding
        self.transition = transition


@dataclass(frozen=True, slots=True)
class NestedRestrictionEmbeddingEvidence:
    """Exact per-parent tiling evidence of one nested-restriction embedding."""

    route: str
    mesh_id: str
    topology_id: str
    geometry_id: str
    validity_certificate_id: str
    source_mesh_id: str
    source_geometry_id: str
    source_embedding_certificate_id: str
    transition_id: str
    parent_count: int
    child_count: int
    parent_interior_facet_count: int
    shared_source_facet_count: int
    source_boundary_facet_count: int
    evidence_id: str


def _determinant(rows: list[list[Fraction]], /) -> Fraction:
    if len(rows) == 1:
        return rows[0][0]
    if len(rows) == 2:
        return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    return (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )


def _permutation_sign(values: tuple[int, ...], /) -> int:
    sign = 1
    items = list(values)
    for index in range(len(items)):
        for other in range(index + 1, len(items)):
            if items[other] < items[index]:
                sign = -sign
    return sign


def _unit_simplex(kind: str, dimension: int, /) -> bool:
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    return np.array_equal(
        vertices, np.vstack((np.zeros((1, dimension)), np.eye(dimension)))
    )


def certify_nested_restriction_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    premise: NestedRestrictionPremise,
    /,
) -> NestedRestrictionEmbeddingEvidence | None:
    """Prove the module theorem's premises exactly, or ``None`` if one fails.

    The source certificate and the transition are checked against their bound
    inputs; nothing of the source proof is repeated.
    """
    source_mesh, source_geometry = premise.source_mesh, premise.source_geometry
    source_embedding = premise.source_embedding
    transition = premise.transition
    source_embedding.require_current(source_mesh, source_geometry)
    transition.require_bound(source_mesh, source_geometry, mesh, geometry)
    dimension = mesh.topological_dimension
    if (
        not validity.all_certified
        or source_embedding.status != "certified"
        or source_embedding.findings
        or mesh.periodic_topology is not None
        or source_mesh.periodic_topology is not None
        or source_mesh.topological_dimension != dimension
        or transition.evidence.kind != "nested_restriction"
        or (transition.reference_map is None and transition.refinement is None)
        or transition.coarsening is not None
        or any(
            _SIMPLEX_KINDS.get(block.cell_kind) != dimension
            for block in (*mesh.blocks, *source_mesh.blocks)
        )
    ):
        return None
    kind = next(iter({block.cell_kind for block in mesh.blocks}), None)
    if kind is None or not _unit_simplex(kind, dimension):
        return None
    corner_count = dimension + 1

    # Source cells: parent ID -> source vertex IDs; source facets -> incidence.
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    parents: dict[int, tuple[int, ...]] = {}
    for block in source_mesh.blocks:
        rows = source_vertex_ids[np.asarray(block.vertices, dtype=np.int64)].tolist()
        for identifier, row in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(), rows, strict=True
        ):
            parents[identifier] = tuple(row)

    # Target cells in mesh order with their reference maps.
    target_vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    children: dict[int, tuple[int, ...]] = {}
    for block in mesh.blocks:
        rows = target_vertex_ids[np.asarray(block.vertices, dtype=np.int64)].tolist()
        for identifier, row in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(), rows, strict=True
        ):
            children[identifier] = tuple(row)
    mapped: dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]] = {}
    # The transition proved the target maps equal the source maps through
    # these groups: its reference maps, or else its nested refinement maps.
    groups = (
        _reference_map_groups(transition.reference_map)
        if transition.reference_map is not None
        else _nested_reference_groups(transition.refinement, source_mesh, mesh)
    )
    for group in groups:
        if (
            group.target_cell_kind != kind
            or group.source_cell_kind != kind
            or group.source_reference_chart_arguments is not None
        ):
            return None
        exact = group.source_reference_exact_vertices
        if exact is None:
            exact = exact_reference_cell_vertices(group)
        for target_id, source_id, corners in zip(
            np.asarray(group.target_cell_ids, dtype=np.int64).tolist(),
            np.asarray(group.source_cell_ids, dtype=np.int64).tolist(),
            exact,
            strict=True,
        ):
            if target_id in mapped:
                return None
            mapped[target_id] = (source_id, corners)
    if mapped.keys() != children.keys():
        return None

    # (5) One exact source position per target vertex, injective on vertices.
    position_by_vertex: dict[int, tuple[tuple[int, Fraction], ...]] = {}
    vertex_by_position: dict[tuple[tuple[int, Fraction], ...], int] = {}
    seen_corners: dict[tuple[int, int], tuple[Fraction, ...]] = {}
    volume_by_parent: dict[int, Fraction] = {}
    # Oriented child facets: sorted vertex IDs -> [(parent, sign)].
    facets: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    # Parent facet (sorted source vertex IDs) carrying each boundary facet.
    boundary_carrier: dict[tuple[int, ...], tuple[int, ...]] = {}
    for target_id, vertices in children.items():
        parent_id, corners = mapped[target_id]
        parent_vertices = parents.get(parent_id)
        if parent_vertices is None or len(corners) != corner_count:
            return None
        zero_slots: list[frozenset[int]] = []
        for vertex_id, corner in zip(vertices, corners, strict=True):
            known = seen_corners.get((parent_id, vertex_id))
            if known is None:
                weights = (_ONE - sum(corner, _ZERO), *corner)
                position = tuple(
                    sorted(
                        (source_vertex, weight)
                        for source_vertex, weight in zip(
                            parent_vertices, weights, strict=True
                        )
                        if weight
                    )
                )
                previous = position_by_vertex.get(vertex_id)
                if previous is None:
                    if vertex_by_position.setdefault(position, vertex_id) != vertex_id:
                        return None
                    position_by_vertex[vertex_id] = position
                elif previous != position:
                    return None
                seen_corners[parent_id, vertex_id] = corner
            elif known != corner:
                return None
            zero_slots.append(
                frozenset(
                    slot
                    for slot, weight in enumerate((_ONE - sum(corner, _ZERO), *corner))
                    if not weight
                )
            )
        # (1) Positive orientation and (2) the per-parent volume sum.
        origin = corners[0]
        determinant = _determinant(
            [
                [corners[row][axis] - origin[axis] for axis in range(dimension)]
                for row in range(1, corner_count)
            ]
        )
        if determinant <= 0:
            return None
        volume_by_parent[parent_id] = volume_by_parent.get(parent_id, _ZERO) + determinant
        # (3)/(4) Oriented facets: the facet opposite local corner ``i``.
        for omitted in range(corner_count):
            local = tuple(index for index in range(corner_count) if index != omitted)
            facet_vertices = tuple(vertices[index] for index in local)
            key = tuple(sorted(facet_vertices))
            sign = (-1 if omitted % 2 else 1) * _permutation_sign(facet_vertices)
            facets.setdefault(key, []).append((parent_id, sign))
            shared_zero = frozenset.intersection(*(zero_slots[index] for index in local))
            if shared_zero:
                if len(shared_zero) != 1:
                    return None
                slot = next(iter(shared_zero))
                carrier = tuple(
                    sorted(
                        vertex
                        for position, vertex in enumerate(parent_vertices)
                        if position != slot
                    )
                )
                if boundary_carrier.setdefault(key, carrier) != carrier:
                    return None
    if any(volume != _ONE for volume in volume_by_parent.values()):
        return None

    # Source facet incidence among the parents bearing target cells.
    incident: dict[tuple[int, ...], set[int]] = {}
    for parent_id in volume_by_parent:
        parent_vertices = parents[parent_id]
        for omitted in range(corner_count):
            incident.setdefault(
                tuple(
                    sorted(
                        vertex
                        for position, vertex in enumerate(parent_vertices)
                        if position != omitted
                    )
                ),
                set(),
            ).add(parent_id)
    interior_count = shared_count = boundary_count = 0
    for key, occurrences in facets.items():
        carrier = boundary_carrier.get(key)
        if carrier is None:
            if (
                len(occurrences) != 2
                or occurrences[0][0] != occurrences[1][0]
                or occurrences[0][1] != -occurrences[1][1]
            ):
                return None
            interior_count += 1
            continue
        owners = incident.get(carrier, set())
        occurrence_parents = [parent for parent, _ in occurrences]
        if (
            len(occurrence_parents) != len(set(occurrence_parents))
            or set(occurrence_parents) != owners
        ):
            return None
        if len(owners) == 1:
            boundary_count += 1
        else:
            shared_count += 1
    payload = {
        "kind": "nested-restriction-embedding-evidence",
        "route": NESTED_RESTRICTION_ROUTE,
        "mesh": mesh.mesh_id,
        "topology": mesh.topology_id,
        "geometry": validity.geometry_id,
        "validity": validity.certificate_id,
        "source_mesh": source_mesh.mesh_id,
        "source_geometry": cell_geometry_id(source_geometry),
        "source_embedding": source_embedding.certificate_id,
        "transition": transition.transition_id,
        "parents": len(volume_by_parent),
        "children": len(children),
        "parent_interior_facets": interior_count,
        "shared_source_facets": shared_count,
        "source_boundary_facets": boundary_count,
    }
    return NestedRestrictionEmbeddingEvidence(
        NESTED_RESTRICTION_ROUTE,
        mesh.mesh_id,
        mesh.topology_id,
        validity.geometry_id,
        validity.certificate_id,
        source_mesh.mesh_id,
        str(payload["source_geometry"]),
        source_embedding.certificate_id,
        transition.transition_id,
        len(volume_by_parent),
        len(children),
        interior_count,
        shared_count,
        boundary_count,
        canonical_fingerprint(payload),
    )


__all__ = [
    "NESTED_RESTRICTION_ROUTE",
    "NestedRestrictionEmbeddingEvidence",
    "NestedRestrictionPremise",
    "certify_nested_restriction_embedding",
]

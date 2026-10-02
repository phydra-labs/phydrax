#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mesh-side attachments of authoritative geometry interfaces.

An attachment links exact entities of one named, revision-bound
:class:`MeshPart` to declared authoritative geometry entities through the
carrier's certified :class:`GeometryAssociation`. It does not classify
entities itself: classification, residuals, and the orientation relation of
each entity's canonical vertex order come from the association owner.

A *sided* attachment additionally records which side of the authoritative
oriented normal the attached cells lie on. The authoritative normal is the
oriented B-Rep face normal in three dimensions and the B-Rep edge tangent
rotated clockwise, ``(t_y, -t_x)``, in two dimensions; ``orientation = +1``
means the outward normal of the attached side equals that normal (the side is
the one the normal points out of) and ``-1`` means it is opposed.
"""

from __future__ import annotations

from typing import assert_never, final

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..geometry.surface import InterfaceSide
from ..typing import checked
from ._assembly import MeshPart
from ._association import (
    _centroids,
    _entity_rows,
    _incidence_pairs,
    _ordered_vertices,
    GeometryAssociation,
)
from ._organization import MeshLabel, MeshPatch, MeshZone
from ._result import CellMeshingResult
from ._scope import MeshingScope


type MeshAttachmentScope = MeshingScope | MeshPatch | MeshZone | MeshLabel


def _carrier(part: MeshPart, /) -> CellMeshingResult:
    if not isinstance(part, MeshPart):
        raise TypeError("part must be MeshPart.")
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise ValueError(
            "Interface attachments require a certified cell carrier with geometry "
            "associations."
        )
    return carrier


def _member(
    value: MeshPatch | MeshZone | MeshLabel, carrier: CellMeshingResult, /
) -> tuple[str, set[str]]:
    """Identity of one organization record and of the carrier's records of its kind."""
    match value:
        case MeshPatch():
            return value.patch_id, {member.patch_id for member in carrier.patches}
        case MeshZone():
            return value.zone_id, {member.zone_id for member in carrier.zones}
        case MeshLabel():
            return value.label_id, {member.label_id for member in carrier.labels}
        case _:
            assert_never(value)


def _part_scope(
    part: MeshPart,
    carrier: CellMeshingResult,
    value: MeshAttachmentScope,
    name: str,
    /,
) -> tuple[MeshingScope, tuple[str, ...]]:
    """Part-bound scope and the organization evidence it came from."""
    match value:
        case MeshingScope():
            part.require_scope(value)
            return value, ()
        case MeshPatch() | MeshZone() | MeshLabel():
            identifier, members = _member(value, carrier)
            if identifier not in members:
                raise ValueError(
                    f"{name} is not organization evidence of the part's certified carrier."
                )
            scope = value.scope
            return (
                part.scope(
                    scope.entity_dimension,
                    np.asarray(scope.entity_ids),
                    entity_set_id=scope.entity_set_id,
                ),
                (identifier,),
            )
        case _:
            raise TypeError(
                f"{name} must be MeshingScope, MeshPatch, MeshZone, or MeshLabel."
            )


def _entity_ids(values: tuple[str, ...], /) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, tuple):
        raise TypeError("geometry_entity_ids must be a tuple of entity IDs.")
    identifiers = tuple(str(value).strip() for value in values)
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("geometry_entity_ids must contain non-empty entity IDs.")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("geometry_entity_ids must be unique.")
    return tuple(sorted(identifiers))


def _witness_rows(
    carrier: CellMeshingResult,
    association: GeometryAssociation,
    scope: MeshingScope,
    entity_ids: tuple[str, ...],
    tolerance: float,
    /,
) -> tuple[np.ndarray, float]:
    """Certified association rows of the scope, classified on the declared entities."""
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if association.association_id not in {
        value.association_id for value in carrier.associations
    }:
        raise ValueError(
            "The geometry association is not certified evidence of the part's carrier."
        )
    if association.target_entity_set_id != scope.entity_set_id:
        raise ValueError("The geometry association classifies another entity set.")
    rows = association.target_rows(np.asarray(scope.entity_ids))
    if not np.all(np.asarray(association.resolved)[rows]) or np.any(
        np.asarray(association.ambiguous)[rows]
    ):
        raise ValueError(
            "Attached entities have unresolved or ambiguous geometry classification."
        )
    classes = {association.source_entity_ids[row] for row in rows.tolist()}
    declared = set(entity_ids)
    if not classes <= declared:
        raise ValueError(
            "Attached entities are classified on geometry outside the declared "
            f"interface entities: {sorted(classes - declared)!r}."
        )
    if not declared <= classes:
        raise ValueError(
            "Declared interface entities have no attached mesh entity: "
            f"{sorted(declared - classes)!r}."
        )
    residual = float(np.max(np.asarray(association.residuals)[rows]))
    if residual > tolerance:
        raise ValueError(
            f"Attached geometry residual {residual:.3e} exceeds tolerance {tolerance:.3e}."
        )
    return rows, residual


def _uniform_sign(relation: np.ndarray, message: str, /) -> int:
    if np.all(relation == 1):
        return 1
    if np.all(relation == -1):
        return -1
    raise ValueError(message)


def _outward_relation(
    mesh: CellMesh, facet_rows: np.ndarray, cell_rows: np.ndarray, /
) -> np.ndarray:
    """Sign of each facet's canonical normal against its attached cell's outward normal."""
    top = mesh.topological_dimension
    ordered = _ordered_vertices(mesh, top - 1)
    if mesh.ambient_dimension != top or top not in (2, 3) or ordered is None:
        raise ValueError(
            "Sided attachments require simplex facets of a volume carrier in its own "
            "ambient dimension."
        )
    pairs = _incidence_pairs(mesh, top - 1, top)
    pairs = pairs[np.isin(pairs[:, 0], facet_rows) & np.isin(pairs[:, 1], cell_rows)]
    counts = np.bincount(pairs[:, 0], minlength=mesh.entity_set(top - 1).count)
    if np.any(counts[facet_rows] != 1):
        raise ValueError(
            "Every attached facet must bound exactly one cell of the attached side."
        )
    adjacent = np.empty_like(counts)
    adjacent[pairs[:, 0]] = pairs[:, 1]
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    corners = points[ordered[facet_rows]]
    first = corners[:, 1] - corners[:, 0]
    if top == 2:
        normal = np.stack((first[:, 1], -first[:, 0]), axis=1)
    else:
        normal = np.cross(first, corners[:, 2] - corners[:, 0])
    cells = _centroids(
        mesh, _incidence_pairs(mesh, 0, top)[:, ::-1], mesh.entity_set(top).count
    )[adjacent[facet_rows]]
    # A simplex centroid lies strictly inside, so the facet-centroid offset
    # has a positive component along the cell's outward facet normal.
    return np.sign(np.sum(normal * (corners.mean(axis=1) - cells), axis=1)).astype(
        np.int64
    )


def _orientation(
    carrier: CellMeshingResult,
    scope: MeshingScope,
    association: GeometryAssociation,
    rows: np.ndarray,
    region: MeshingScope | None,
    carrier_side: InterfaceSide | None,
    /,
) -> int:
    """Relation of the attached side's outward normal to the authoritative normal."""
    mesh = carrier.mesh
    if region is None and carrier_side is None:
        return 0
    if region is not None and carrier_side is not None:
        raise ValueError("Declare either region or carrier_side, not both.")
    witness = np.asarray(association.orientations, dtype=np.int64)[rows]
    if np.any(witness == 0):
        raise ValueError(
            "The geometry association records no orientation for the attached entities."
        )
    top = mesh.topological_dimension
    if carrier_side is not None:
        if not isinstance(carrier_side, InterfaceSide):
            raise TypeError("carrier_side must be InterfaceSide or None.")
        if scope.entity_dimension != top or mesh.ambient_dimension != top + 1:
            raise ValueError(
                "carrier_side declares the side of a codimension-one carrier's own cells."
            )
        relation = _uniform_sign(
            witness, "Attached carrier cells disagree with the authoritative normal."
        )
        return relation if carrier_side is InterfaceSide.MINUS else -relation
    if not isinstance(region, MeshingScope):
        raise TypeError("region must resolve to a MeshingScope.")
    if scope.entity_dimension != top - 1 or region.entity_dimension != top:
        raise ValueError("A sided region attaches facets to the cells on one side.")
    facet_rows = _entity_rows(mesh, top - 1, np.asarray(scope.entity_ids))
    cell_rows = _entity_rows(mesh, top, np.asarray(region.entity_ids))
    return _uniform_sign(
        witness * _outward_relation(mesh, facet_rows, cell_rows),
        "Attached facets disagree on the side of the authoritative normal.",
    )


@final
class MeshInterfaceAttachment(StrictModule, NonTrainableState):
    """Exact entities of one mesh part attached to authoritative geometry entities.

    ``scope`` holds the attached entities and ``region`` the cells of the attached
    side (sided volume attachments only). ``organization_ids`` name the certified
    patch/zone/label evidence the scopes came from. ``association_id`` is the
    witness: a certified association of the part's carrier classifying every
    attached entity on the declared ``geometry_entity_ids`` of
    ``geometry_source_revision`` within ``tolerance``. ``orientation`` relates
    the attached side's outward normal to the authoritative oriented normal
    (``+1`` aligned, ``-1`` opposed, ``0`` for an unsided attachment).
    """

    part_name: str = eqx.field(static=True)
    part_revision: str = eqx.field(static=True)
    scope: MeshingScope
    region: MeshingScope | None
    carrier_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    organization_ids: tuple[str, ...] = eqx.field(static=True)
    geometry_source_id: str = eqx.field(static=True)
    geometry_source_revision: str = eqx.field(static=True)
    geometry_entity_ids: tuple[str, ...] = eqx.field(static=True)
    association_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_residual: float = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    attachment_id: str = eqx.field(static=True)

    def __init__(
        self,
        part: MeshPart,
        scope: MeshAttachmentScope,
        association: GeometryAssociation,
        geometry_entity_ids: tuple[str, ...],
        /,
        *,
        tolerance: float,
        region: MeshAttachmentScope | None = None,
        carrier_side: InterfaceSide | None = None,
    ) -> None:
        carrier = _carrier(part)
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        entity_ids = _entity_ids(geometry_entity_ids)
        attached, attached_members = _part_scope(part, carrier, scope, "scope")
        side, side_members = (
            (None, ()) if region is None else _part_scope(part, carrier, region, "region")
        )
        rows, residual = _witness_rows(
            carrier, association, attached, entity_ids, tolerance_
        )
        orientation = _orientation(
            carrier, attached, association, rows, side, carrier_side
        )
        organization = tuple(sorted({*attached_members, *side_members}))
        self.part_name = part.name
        self.part_revision = part.part_id
        self.scope = attached
        self.region = side
        self.carrier_dimension = carrier.mesh.topological_dimension
        self.ambient_dimension = carrier.mesh.ambient_dimension
        self.organization_ids = organization
        self.geometry_source_id = association.source_id
        self.geometry_source_revision = association.source_revision
        self.geometry_entity_ids = entity_ids
        self.association_id = association.association_id
        self.tolerance = tolerance_
        self.maximum_residual = residual
        self.orientation = orientation
        self.attachment_id = canonical_fingerprint(
            {
                "kind": "mesh-interface-attachment",
                "part": part.name,
                "part_revision": part.part_id,
                "scope": attached.scope_id,
                "region": None if side is None else side.scope_id,
                "organization": organization,
                "geometry_source_id": association.source_id,
                "geometry_source_revision": association.source_revision,
                "geometry_entity_ids": entity_ids,
                "association": association.association_id,
                "tolerance": tolerance_,
                "maximum_residual": residual,
                "orientation": orientation,
            }
        )

    @property
    def sided(self) -> bool:
        return self.orientation != 0

    @checked
    def require_current(self, part: MeshPart, /) -> None:
        """Refuse a part that is not the exact revision this attachment binds."""
        if part.name != self.part_name:
            raise ValueError(
                f"Mesh part {part.name!r} does not own attachment of {self.part_name!r}."
            )
        if part.part_id != self.part_revision:
            raise ValueError(
                f"Interface attachment of mesh part {self.part_name!r} is stale: the "
                "part revision changed."
            )

    def oriented_simplices(
        self, part: MeshPart, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Entity rows, simplex corners, and authoritative normals of the attachment.

        Rows index the carrier's entity set ``scope.entity_set_id`` in the order
        of ``scope.entity_ids``; ``corners`` has shape ``(entities, d, d)`` (segment
        ends in two dimensions, triangle corners in three). ``normals`` are the
        unit authoritative oriented normals: each entity's canonical vertex-order
        normal (tangent rotated clockwise in 2-D, ``(v1 - v0) x (v2 - v0)`` in
        3-D) times the association's orientation witness, so the attached side's
        outward normal has sign ``orientation`` against them.
        """
        self.require_current(part)
        carrier = _carrier(part)
        mesh = carrier.mesh
        dimension = self.scope.entity_dimension
        ambient = mesh.ambient_dimension
        ordered = _ordered_vertices(mesh, dimension)
        if ambient not in (2, 3) or dimension != ambient - 1 or ordered is None:
            raise ValueError(
                "Oriented attachment geometry requires codimension-one simplex "
                "entities in two or three dimensions."
            )
        association = None
        for value in carrier.associations:
            if value.association_id == self.association_id:
                association = value
        if association is None:
            raise ValueError(
                "The attachment's geometry association is not evidence of the part."
            )
        ids = np.asarray(self.scope.entity_ids, dtype=np.int64)
        rows = _entity_rows(mesh, dimension, ids)
        witness = np.asarray(association.orientations, dtype=np.int64)[
            association.target_rows(ids)
        ]
        corners = np.asarray(mesh.coordinates, dtype=np.float64)[ordered[rows]]
        first = corners[:, 1] - corners[:, 0]
        if ambient == 2:
            normal = np.stack((first[:, 1], -first[:, 0]), axis=1)
        else:
            normal = np.cross(first, corners[:, 2] - corners[:, 0])
        normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
        return rows, corners, witness[:, None].astype(np.float64) * normal


__all__ = ["MeshInterfaceAttachment"]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Defines mesh-side attachments to declared geometry interfaces.

An attachment links every requested entity of one named, revision-bound
:class:`MeshPart` to declared geometry entities through the carrier's certified
:class:`GeometryAssociation`. Classification, residuals, and the orientation
relation of each entity's vertex order are provided by the association holder.

A *sided* attachment also records which side of the declared oriented normal
contains the attached cells. The oriented normal is the B-Rep face normal in
three dimensions and the B-Rep edge tangent rotated clockwise,
``(t_y, -t_x)``, in two dimensions. ``orientation = +1`` means the attached
side's outward normal equals that normal, so the side is the one from which the
normal points outward; ``orientation = -1`` means the normals are opposed.
"""

from __future__ import annotations

from typing import assert_never, final

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
from phydrax.meshing._association_state import entity_rows, GeometryAssociation

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_finite_float
from ..geometry.surface import InterfaceSide
from ..typing import checked
from ._association import centroids
from ._association_routes import incidence_pairs, ordered_vertices
from ._lineage import inherit_scope
from ._lineage_records import MeshLineage
from ._mesh_part import MeshPart
from ._organization_records import MeshLabel, MeshPatch, MeshZone
from ._result import CellMeshingResult
from ._scope import MeshingScope


type MeshAttachmentScope = MeshingScope | MeshPatch | MeshZone | MeshLabel


def _carrier(part: MeshPart, /) -> CellMeshingResult:
    carrier = part.carrier
    if not isinstance(carrier, CellMeshingResult):
        raise ValueError(
            "MeshInterfaceAttachment.part.carrier must be CellMeshingResult with geometry associations; got another carrier kind."
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
                    f"{'MeshInterfaceAttachment.' + name} must identify a scope, patch, zone, or label in the carrier organization; got an absent identifier."
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
                f"{'MeshInterfaceAttachment.' + name} must resolve to a MeshingScope, MeshPatch, MeshZone, or MeshLabel; got an unsupported part-scope value."
            )


def _entity_ids(values: tuple[str, ...], /) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, tuple):
        raise TypeError(
            "MeshInterfaceAttachment.geometry_entity_ids must be a tuple of entity IDs; got a string or non-tuple value."
        )
    identifiers = tuple(str(value).strip() for value in values)
    if not identifiers or any(not value for value in identifiers):
        raise ValueError(
            "MeshInterfaceAttachment.geometry_entity_ids must contain at least one entity ID and every ID must be non-empty; got an empty tuple or empty ID."
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            "MeshInterfaceAttachment.geometry_entity_ids must contain unique entity IDs; got duplicate IDs."
        )
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
    if association.association_id not in {
        value.association_id for value in carrier.associations
    }:
        raise ValueError(
            "MeshInterfaceAttachment.association must be listed in MeshInterfaceAttachment.carrier.associations; got an association outside the carrier."
        )
    if association.target_entity_set_id != scope.entity_set_id:
        raise ValueError(
            "MeshInterfaceAttachment.association.target_entity_set_id must equal scope.entity_set_id; got an association for another entity set."
        )
    rows = association.target_rows(np.asarray(scope.entity_ids))
    if not np.all(np.asarray(association.resolved)[rows]) or np.any(
        np.asarray(association.ambiguous)[rows]
    ):
        raise ValueError(
            "MeshInterfaceAttachment.association.resolved must be true and MeshInterfaceAttachment.association.ambiguous must be false for every attached row; got an unresolved or ambiguous row."
        )
    classes = {association.source_entities.entity_id(row) for row in rows.tolist()}
    declared = set(entity_ids)
    if not classes <= declared:
        raise ValueError(
            f"MeshInterfaceAttachment.association must classify attached entities on declared geometry entities; got undeclared classifications {sorted(classes - declared)!r}."
        )
    if not declared <= classes:
        raise ValueError(
            f"MeshInterfaceAttachment.association must cover declared interface entities; got unclassified entities {sorted(declared - classes)!r}"
        )
    residual = float(np.max(np.asarray(association.residuals)[rows]))
    if residual > tolerance:
        raise ValueError(
            f"MeshInterfaceAttachment.association.residuals must be no greater than MeshInterfaceAttachment.tolerance; got residual {residual:.3e} and tolerance {tolerance:.3e}."
        )
    return rows, residual


def _uniform_sign(relation: np.ndarray, message: str, /) -> int:
    if np.all(relation == 1):
        return 1
    if np.all(relation == -1):
        return -1
    raise ValueError(message)


def _outward_relation(
    mesh: cell_mesh_owner.CellMesh, facet_rows: np.ndarray, cell_rows: np.ndarray, /
) -> np.ndarray:
    """Returns the sign of every facet's vertex-order normal relative to its attached
    cell's outward normal.
    """
    top = mesh.topological_dimension
    ordered = ordered_vertices(mesh, top - 1)
    if mesh.ambient_dimension != top or top not in (2, 3) or ordered is None:
        raise ValueError(
            "MeshInterfaceAttachment.region must select cells of a full-dimensional simplex volume carrier; got incompatible dimensions or topology."
        )
    pairs = incidence_pairs(mesh, top - 1, top)
    pairs = pairs[np.isin(pairs[:, 0], facet_rows) & np.isin(pairs[:, 1], cell_rows)]
    counts = np.bincount(pairs[:, 0], minlength=mesh.entity_set(top - 1).count)
    if np.any(counts[facet_rows] != 1):
        raise ValueError(
            "MeshInterfaceAttachment.scope must attach each facet to one cell of region; got a different attached-side cell count."
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
    cells = centroids(
        mesh, incidence_pairs(mesh, 0, top)[:, ::-1], mesh.entity_set(top).count
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
    """Returns the relation of the attached side's outward normal to the declared oriented
    normal.
    """
    mesh = carrier.mesh
    if region is None and carrier_side is None:
        return 0
    if region is not None and carrier_side is not None:
        raise ValueError(
            "MeshInterfaceAttachment.region must be omitted when carrier_side is specified; got both side declarations."
        )
    witness = np.asarray(association.orientations, dtype=np.int64)[rows]
    if np.any(witness == 0):
        raise ValueError(
            "MeshInterfaceAttachment.association.orientation must be nonzero for every attached entity; got a zero orientation."
        )
    top = mesh.topological_dimension
    if carrier_side is not None:
        if not isinstance(carrier_side, InterfaceSide):
            raise TypeError(
                "MeshInterfaceAttachment.carrier_side must be an InterfaceSide or None; got another value type."
            )
        if scope.entity_dimension != top or mesh.ambient_dimension != top + 1:
            raise ValueError(
                "MeshInterfaceAttachment.carrier_side must describe the side of a codimension-one carrier's cells; got incompatible entity and ambient dimensions."
            )
        relation = _uniform_sign(
            witness,
            "MeshInterfaceAttachment.carrier_side must select cells with a uniform geometry normal relation; got disagreeing carrier-cell orientations.",
        )
        return relation if carrier_side is InterfaceSide.MINUS else -relation
    if region is None:
        raise TypeError(
            "MeshInterfaceAttachment.region must resolve to a MeshingScope; got a non-MeshingScope region."
        )
    if scope.entity_dimension != top - 1 or region.entity_dimension != top:
        raise ValueError(
            "MeshInterfaceAttachment.region must be a top-dimensional cell scope paired with a codimension-one attachment scope; got incompatible scope dimensions."
        )
    facet_rows = entity_rows(mesh, top - 1, np.asarray(scope.entity_ids))
    cell_rows = entity_rows(mesh, top, np.asarray(region.entity_ids))
    return _uniform_sign(
        witness * _outward_relation(mesh, facet_rows, cell_rows),
        "MeshInterfaceAttachment.region must select facets with a uniform geometry normal relation; got disagreeing facet-side orientations.",
    )


@final
class MeshInterfaceAttachment(StrictModule, NonTrainableState):
    """Represents every requested entity of one mesh part attached to declared geometry
    entities.

    ``scope`` contains the attached entities, and ``region`` contains the cells on
    the attached side for sided volume attachments. ``organization_ids`` identify
    the certified patch, zone, or label evidence from which the scopes came.
    ``association_id`` identifies a certified association of the part's carrier
    that classifies every attached entity on the declared ``geometry_entity_ids``
    of ``geometry_source_revision`` within ``tolerance``. ``orientation`` relates
    the attached side's outward normal to the declared oriented normal: ``+1`` for
    aligned, ``-1`` for opposed, and ``0`` for an unsided attachment.
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

    @checked
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
        tolerance_ = nonnegative_finite_float(
            tolerance, "MeshInterfaceAttachment.tolerance"
        )
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
        """Refuses a part unless its revision is the revision bound by this attachment."""
        if part.name != self.part_name:
            raise ValueError(
                f"MeshInterfaceAttachment.part_name must match the supplied mesh part name; got {part.name!r} for attachment {self.part_name!r}."
            )
        if part.part_id != self.part_revision:
            raise ValueError(
                f"MeshInterfaceAttachment.part_revision must match the supplied part revision; got a stale attachment for {self.part_name!r}."
            )

    @checked
    def oriented_simplices(
        self, part: MeshPart, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns entity rows, simplex corners, and declared oriented normals for every
        requested attachment entity.

        Rows index the carrier's entity set ``scope.entity_set_id`` in the order of
        ``scope.entity_ids``. ``corners`` has shape ``(entities, d, d)``: segment ends
        in two dimensions and triangle corners in three dimensions. ``normals`` are
        unit declared oriented normals. Each is the entity's vertex-order normal—the
        tangent rotated clockwise in 2-D or ``(v1 - v0) x (v2 - v0)`` in 3-D—multiplied
        by the association's orientation relation, so the attached side's outward
        normal has sign ``orientation`` relative to it.
        """
        self.require_current(part)
        carrier = _carrier(part)
        mesh = carrier.mesh
        dimension = self.scope.entity_dimension
        ambient = mesh.ambient_dimension
        ordered = ordered_vertices(mesh, dimension)
        if ambient not in (2, 3) or dimension != ambient - 1 or ordered is None:
            raise ValueError(
                f"MeshInterfaceAttachment.geometry must use codimension-one simplex entities in two or three dimensions; got entity dimension {dimension}, ambient dimension {ambient}, and ordered vertices available={ordered is not None}"
            )
        association = None
        for value in carrier.associations:
            if value.association_id == self.association_id:
                association = value
        if association is None:
            raise ValueError(
                "MeshInterfaceAttachment.association must belong to the attached part's carrier; got an association outside the part."
            )
        ids = np.asarray(self.scope.entity_ids, dtype=np.int64)
        rows = entity_rows(mesh, dimension, ids)
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


@checked
def transition_interface_attachment(
    attachment: MeshInterfaceAttachment,
    source: MeshPart,
    target: MeshPart,
    lineage: MeshLineage,
    /,
) -> MeshInterfaceAttachment:
    """Rebinds an attachment to the successor revision of its part.

    ``source`` must be the revision bound by the attachment, and ``lineage`` must
    be the topology transition from its carrier to the carrier of ``target``. The
    function remaps every requested entity in the attached and sided scopes through
    the lineage, requires the successor carrier's single association for the same
    geometry source revision to classify them again, and requires the side relation
    to remain unchanged. It refuses the rebind for mixed membership, a missing or
    ambiguous successor association, or a changed side, and does not copy a stale
    attachment.
    """

    attachment.require_current(source)
    carrier = _carrier(target)
    if target.name != source.name:
        raise ValueError(
            "transition_interface_attachment.target.name must equal transition_interface_attachment.source.name; got a changed part name."
        )
    if (
        lineage.source_topology_id != _carrier(source).mesh.topology_id
        or lineage.target_topology_id != carrier.mesh.topology_id
    ):
        raise ValueError(
            "transition_interface_attachment.lineage must join the source and target carrier topology IDs; got a mismatched source or target topology ID."
        )
    mesh = carrier.mesh
    remapped = inherit_scope(attachment.scope, lineage, mesh, "attachment scope")
    scope = target.scope(remapped.entity_dimension, remapped.entity_ids)
    region = None
    if attachment.region is not None:
        side = inherit_scope(attachment.region, lineage, mesh, "attachment region")
        region = target.scope(side.entity_dimension, side.entity_ids)
    associations = tuple(
        value
        for value in carrier.associations
        if value.source_id == attachment.geometry_source_id
        and value.source_revision == attachment.geometry_source_revision
        and value.target_entity_set_id == remapped.entity_set_id
    )
    if len(associations) != 1:
        raise ValueError(
            f"transition_interface_attachment.target.associations must contain one association for the attachment geometry revision and entities; got {len(associations)} matches."
        )
    sides: tuple[InterfaceSide | None, ...] = (
        (None,)
        if attachment.orientation == 0 or region is not None
        else (InterfaceSide.MINUS, InterfaceSide.PLUS)
    )
    for side in sides:
        candidate = MeshInterfaceAttachment(
            target,
            scope,
            associations[0],
            attachment.geometry_entity_ids,
            tolerance=attachment.tolerance,
            region=region,
            carrier_side=side,
        )
        if candidate.orientation == attachment.orientation:
            return candidate
    raise ValueError(
        "transition_interface_attachment.side_relation must remain equal between source and successor attachments; got a changed successor side relation."
    )


__all__ = ["MeshInterfaceAttachment", "transition_interface_attachment"]

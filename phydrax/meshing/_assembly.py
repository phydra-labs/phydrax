#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from math import prod
from typing import TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellGeometrySpec, PointCloudPlan, PreparedTensorGrid
from ..discretization.iga import IsogeometricPlan
from ..typing import checked
from ._canonical import certify_cell_mesh
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._lineage import identity_lineage, inherit_mesh_organization, inherit_scope
from ._organization import MeshAttribute, MeshAttributeRole
from ._result import CellMeshingResult
from ._scope import _contains_ids, MeshingEntityKind, MeshingScope


if TYPE_CHECKING:
    from ._coupling import MeshCoupling
    from ._interface_binding import MeshInterfaceAttachment


class MeshCarrierKind(StrEnum):
    CELL = "cell"
    TENSOR = "tensor"
    POINT = "point"
    SPLINE = "spline"


MeshCarrier: TypeAlias = (
    CellMeshingResult | PreparedTensorGrid | PointCloudPlan | IsogeometricPlan
)


class MeshPart(StrictModule, NonTrainableState):
    """Named immutable carrier; assembly never tessellates compact representations.

    ``name`` is the stable source identity and ``part_id`` its exact revision.
    Coordinates of every part must already use the assembly coordinate contract.
    """

    name: str = eqx.field(static=True)
    carrier: MeshCarrier
    coordinate_contract: SpatialCoordinateContract
    carrier_kind: MeshCarrierKind = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    part_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        carrier: MeshCarrier,
        /,
        *,
        coordinate_contract: SpatialCoordinateContract | None = None,
    ) -> None:
        name_ = str(name).strip()
        if not name_:
            raise ValueError("Mesh part names must be non-empty.")
        logical_values_id = None
        if isinstance(carrier, CellMeshingResult):
            contract = (
                carrier.coordinate_contract
                if coordinate_contract is None
                else coordinate_contract
            )
            if (
                not isinstance(contract, SpatialCoordinateContract)
                or contract.spatial_id != carrier.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Cell part coordinates must retain their certified contract."
                )
            kind = MeshCarrierKind.CELL
            intrinsic, ambient = (
                carrier.mesh.topological_dimension,
                carrier.mesh.ambient_dimension,
            )
            storage = carrier.mesh.storage
            if storage is None:
                identity = carrier.result_id
                values = (carrier.mesh, carrier.geometry)
            else:
                collective = carrier.collective_evidence
                if collective is None:
                    raise ValueError(
                        "Distributed mesh parts require consumed collective publication evidence."
                    )
                collective.require_passed()
                if (
                    collective.mesh_id != carrier.mesh.mesh_id
                    or collective.topology_id != carrier.mesh.topology_id
                    or collective.topology_id != storage.logical_topology_id
                    or collective.geometry_id != carrier.mesh.geometry_id
                    or collective.geometry_id != storage.logical_geometry_id
                    or collective.evidence_id != storage.evidence_id
                    or collective.partition_count != storage.partition_count
                    or collective.global_entity_counts != storage.global_entity_counts
                    or carrier.geometry.storage_id != storage.storage_id
                    or carrier.geometry.logical_geometry_id
                    != storage.logical_coordinate_geometry_id
                    or not carrier.geometry.geometry_layout_id
                ):
                    raise ValueError(
                        "Distributed mesh-part geometry/logical publication bindings are stale."
                    )
                # Rank-local audits, lowered slots and shard payloads are not
                # global scientific identity. This consumed witness binds the
                # accepted logical publication without fetching global arrays.
                logical_values_id = canonical_fingerprint(
                    {
                        "kind": "collective-part-numerical-values",
                        "topology": storage.logical_topology_id,
                        "geometry": storage.logical_geometry_id,
                        "coordinate_layout": carrier.geometry.geometry_layout_id,
                        "coverage": storage.evidence_id,
                        "global_entity_counts": storage.global_entity_counts,
                    }
                )
                identity = carrier.result_id
                values = None
        else:
            contract = coordinate_contract
            if not isinstance(contract, SpatialCoordinateContract):
                raise TypeError(
                    "Compact carriers require an explicit SpatialCoordinateContract."
                )
            if isinstance(carrier, PreparedTensorGrid):
                kind = MeshCarrierKind.TENSOR
                intrinsic = ambient = len(carrier.axis_names)
                identity, values = carrier.prepared_id, carrier
            elif isinstance(carrier, PointCloudPlan):
                kind = MeshCarrierKind.POINT
                intrinsic, ambient = 0, carrier.points.shape[1]
                identity, values = carrier.plan_id, carrier
            elif isinstance(carrier, IsogeometricPlan):
                kind = MeshCarrierKind.SPLINE
                intrinsic, ambient = (
                    carrier.basis.parametric_dimension,
                    carrier.geometry.ambient_dimension,
                )
                identity, values = carrier.plan_id, carrier
            else:
                raise TypeError("Unsupported mesh carrier.")
        self.name = name_
        self.carrier = carrier
        self.coordinate_contract = contract
        self.carrier_kind = kind
        self.intrinsic_dimension = intrinsic
        self.ambient_dimension = ambient
        self.part_id = canonical_fingerprint(
            {
                "kind": "mesh-part",
                "name": name_,
                "carrier_kind": kind.value,
                "carrier": identity,
                "values": array_tree_fingerprint(values)
                if logical_values_id is None
                else logical_values_id,
                "coordinates": contract.spatial_id,
            }
        )

    def entity_binding(
        self, dimension: int, /, *, entity_set_id: str | None = None
    ) -> tuple[str, np.ndarray]:
        """Resolve native global IDs, requiring the layout for ambiguous tensor faces."""
        carrier = self.carrier
        if isinstance(carrier, CellMeshingResult):
            entities = carrier.mesh.entity_set(dimension)
            identifier = entities.entity_set_id
            ids = np.asarray(entities.entity_ids)[np.asarray(entities.active_mask)]
        elif isinstance(carrier, PreparedTensorGrid):
            layouts = tuple(
                layout
                for layout in carrier.entity_layouts
                if layout.axis_entities.count("interval") == dimension
                and (entity_set_id is None or layout.entity_set_id == entity_set_id)
            )
            if len(layouts) != 1:
                raise ValueError(
                    "Tensor entity dimension requires one exact entity_set_id."
                )
            identifier = layouts[0].entity_set_id
            ids = np.arange(prod(layouts[0].shape), dtype=np.int64)
        elif isinstance(carrier, PointCloudPlan):
            if dimension != 0:
                raise ValueError("Point carriers only expose point entities.")
            identifier = canonical_fingerprint(
                {"kind": "mesh-part-points", "plan": carrier.plan_id}
            )
            ids = np.arange(carrier.points.shape[0], dtype=np.int64)
        else:
            if dimension != self.intrinsic_dimension:
                raise ValueError(
                    "Spline carriers expose native positive-span entities, not coefficient vertices."
                )
            identifier = carrier.topology.topology_id
            ids = np.arange(carrier.topology.cell_count, dtype=np.int64)
        if entity_set_id is not None and entity_set_id != identifier:
            raise ValueError("Entity set does not belong to this mesh part.")
        return identifier, ids

    def scope(
        self,
        dimension: int,
        entity_ids: ArrayLike,
        /,
        *,
        entity_set_id: str | None = None,
    ) -> MeshingScope:
        identifier, _ = self.entity_binding(dimension, entity_set_id=entity_set_id)
        scope = MeshingScope(
            self.name,
            self.part_id,
            MeshingEntityKind.MESH,
            dimension,
            identifier,
            entity_ids,
        )
        self.require_scope(scope)
        return scope

    @checked
    def require_scope(self, scope: MeshingScope, /) -> None:
        if (
            scope.source_id != self.name
            or scope.source_revision != self.part_id
            or scope.entity_kind != MeshingEntityKind.MESH
        ):
            raise ValueError("Mesh scope is stale or belongs to another part.")
        _, ids = self.entity_binding(
            scope.entity_dimension, entity_set_id=scope.entity_set_id
        )
        if not np.all(np.isin(np.asarray(scope.entity_ids), ids)):
            raise ValueError("Mesh scope contains unknown or inactive entity IDs.")

    def point_coordinates(self, scope: MeshingScope, /) -> Array:
        """Gather actual point entities, without turning spans or cells into vertices."""
        self.require_scope(scope)
        if scope.entity_dimension != 0:
            raise ValueError("Point coordinates require a zero-dimensional scope.")
        carrier = self.carrier
        ids = np.asarray(scope.entity_ids)
        if isinstance(carrier, CellMeshingResult):
            vertex_ids = np.asarray(carrier.mesh.vertex_global_ids)
            rows = {int(value): index for index, value in enumerate(vertex_ids)}
            return carrier.mesh.coordinates[
                jnp.asarray([rows[int(value)] for value in ids])
            ]
        if isinstance(carrier, PointCloudPlan):
            return carrier.points[jnp.asarray(ids)]
        if isinstance(carrier, PreparedTensorGrid):
            layout = carrier.vertices()
            indices = np.unravel_index(ids, layout.shape)
            return jnp.stack(
                tuple(
                    axis[jnp.asarray(index)]
                    for axis, index in zip(
                        layout.coordinates_by_axis, indices, strict=True
                    )
                ),
                axis=-1,
            )
        raise ValueError("Spline spans do not expose point coordinates.")

    def with_coordinates(
        self,
        coordinates: ArrayLike,
        /,
        *,
        motion_id: str,
        geometry: CellGeometrySpec | None = None,
    ) -> MeshPart:
        """Recertify a cell part at moved vertex coordinates in mesh row order.

        Topology and scientific entity IDs are unchanged; organization scopes
        rebind through identity lineage. Non-vertex-aligned coordinate maps need
        the explicit successor ``geometry``. Source-bound certificates,
        classifications and associations require a fully recertified successor
        part registered through the overset owner's ``reregister`` handoff.
        ``motion_id`` names the numeric revision; unchanged coordinates return
        this part itself.
        """
        carrier = self.carrier
        if not isinstance(carrier, CellMeshingResult):
            raise TypeError("Only certified cell mesh parts can move.")
        points = np.asarray(coordinates)
        if points.dtype.kind not in "iuf":
            raise TypeError("Part motion coordinates must be real arrays.")
        points = points.astype(np.float64)
        current = np.asarray(carrier.mesh.coordinates, dtype=np.float64)
        if points.shape != current.shape or not np.all(np.isfinite(points)):
            raise ValueError(
                f"Motion of {self.name!r} requires finite coordinates of shape "
                f"{current.shape}."
            )
        if geometry is not None and not isinstance(geometry, CellGeometrySpec):
            raise TypeError("Motion geometry must be CellGeometrySpec or None.")
        if np.array_equal(points, current) and geometry is None:
            return self
        elements, routes, old_coordinates = carrier.geometry.resolve(carrier.mesh)
        vertex_aligned = (
            np.array_equal(np.asarray(old_coordinates), current)
            and all(
                np.array_equal(np.asarray(route), np.asarray(block.vertices))
                for route, block in zip(routes, carrier.mesh.blocks, strict=True)
            )
            and all(getattr(element, "degree", 1) == 1 for element in elements)
        )
        if geometry is None and not vertex_aligned:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Non-vertex-aligned part motion requires the complete successor coordinate map.",
                stage="motion",
            )
        if (
            carrier.boundary is not None
            or carrier.associations
            or carrier.certification is not None
            or carrier.region_evidence is not None
            or any(
                attribute.role is MeshAttributeRole.GEOMETRY_CLASSIFICATION
                for attribute in carrier.attributes
            )
        ):
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                f"Motion of {self.name!r} requires a recertified successor part "
                "for its source geometry/classification/region obligations.",
            )
        mesh = carrier.mesh.with_coordinates(
            points,
            numeric_version=canonical_fingerprint(
                {
                    "kind": "mesh-part-motion",
                    "motion": str(motion_id),
                    "mesh": carrier.mesh.mesh_id,
                    "coordinates": array_tree_fingerprint(points),
                }
            ),
        )
        lineage = identity_lineage(carrier.mesh, mesh)
        patches, zones, labels = inherit_mesh_organization(carrier, mesh, lineage)
        attributes = tuple(
            MeshAttribute(
                attribute.name,
                attribute.role,
                inherit_scope(attribute.scope, lineage, mesh, attribute.name),
                attribute.global_values,
                unit=attribute.unit,
            )
            for attribute in carrier.attributes
        )
        return MeshPart(
            self.name,
            certify_cell_mesh(
                mesh,
                self.coordinate_contract,
                geometry=geometry,
                patches=patches,
                zones=zones,
                labels=labels,
                attributes=attributes,
            ),
        )


class MeshAssembly(StrictModule, NonTrainableState):
    """Named parts and revision-bound coupling overlays, with no implicit welding."""

    parts: tuple[MeshPart, ...]
    couplings: tuple[MeshCoupling, ...]
    coordinate_contract: SpatialCoordinateContract
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self, parts: tuple[MeshPart, ...], /, *, couplings: tuple[MeshCoupling, ...] = ()
    ) -> None:
        from ._coupling import MeshCoupling, OversetCoupling

        values = tuple(parts)
        overlays = tuple(couplings)
        if not values or not all(isinstance(part, MeshPart) for part in values):
            raise ValueError("Mesh assemblies require MeshPart values.")
        if len({part.name for part in values}) != len(values):
            raise ValueError("Every assembly part must have one unique name.")
        values = tuple(sorted(values, key=lambda part: part.name))
        contract = values[0].coordinate_contract
        if any(
            part.coordinate_contract.spatial_id != contract.spatial_id
            or part.ambient_dimension != values[0].ambient_dimension
            for part in values
        ):
            raise ValueError(
                "Assembly parts must use one coordinate contract and ambient dimension."
            )
        if not all(isinstance(overlay, MeshCoupling) for overlay in overlays):
            raise TypeError("couplings must contain MeshCoupling values.")
        if len({overlay.coupling_id for overlay in overlays}) != len(overlays):
            raise ValueError("Assembly coupling overlays must be unique.")
        by_name = {part.name: part for part in values}
        for overlay in overlays:
            for scope in (overlay.source_scope, overlay.target_scope):
                if scope.source_id not in by_name:
                    raise ValueError("Coupling endpoint is not owned by this assembly.")
                by_name[scope.source_id].require_scope(scope)
        receptors: dict[tuple[str, str], list[MeshingScope]] = {}
        holes: dict[tuple[str, str], list[MeshingScope]] = {}
        hole_scopes: dict[tuple[str, str], str] = {}
        for overlay in overlays:
            if not isinstance(overlay, OversetCoupling):
                continue
            key = (overlay.target_scope.source_id, overlay.target_scope.entity_set_id)
            owned = receptors.setdefault(key, [])
            if any(
                bool(
                    jnp.any(
                        _contains_ids(
                            previous.global_entity_ids,
                            overlay.target_scope.global_entity_ids,
                        )
                    )
                )
                for previous in owned
            ):
                raise ValueError(
                    "Every overset receptor must have exactly one donor overlay."
                )
            owned.append(overlay.target_scope)
            if overlay.hole_scope is not None:
                by_name[overlay.hole_scope.source_id].require_scope(overlay.hole_scope)
                # One part has one blanking: every overlay into it names it alike.
                previous = hole_scopes.setdefault(key, overlay.hole_scope.scope_id)
                if previous != overlay.hole_scope.scope_id:
                    raise ValueError(
                        "Overset overlays into one receptor entity set must share one "
                        "hole scope."
                    )
                holes.setdefault(key, []).append(overlay.hole_scope)
        for key, scopes in receptors.items():
            if any(
                bool(
                    jnp.any(
                        _contains_ids(
                            scope.global_entity_ids,
                            hole.global_entity_ids,
                        )
                    )
                )
                for scope in scopes
                for hole in holes.get(key, ())
            ):
                raise ValueError("Overset receptor and hole ownership must be disjoint.")
        for overlay in overlays:
            if not isinstance(overlay, OversetCoupling):
                continue
            if overlay.field_query is not None:
                support = overlay.support_cell_scope
                if support is None:
                    raise ValueError(
                        "Query-mode overset donors require an explicit support-cell scope."
                    )
                key = (support.source_id, support.entity_set_id)
                used = support.global_entity_ids
            else:
                key = (overlay.source_scope.source_id, overlay.source_scope.entity_set_id)
                weights = overlay.donor_weights
                if weights is None:
                    raise ValueError(
                        "Explicit overset vertex stencils require their owning donor weights."
                    )
                used = jnp.where(
                    weights > 0,
                    overlay.donor_ids,
                    -1,
                ).reshape(-1)
            forbidden = (*receptors.get(key, ()), *holes.get(key, ()))
            if any(
                bool(
                    jnp.any(
                        _contains_ids(
                            scope.global_entity_ids,
                            used,
                        )
                    )
                )
                for scope in forbidden
            ):
                raise ValueError("Overset donors cannot be assembly receptors or holes.")
        overlays = tuple(sorted(overlays, key=lambda overlay: overlay.coupling_id))
        self.parts = values
        self.couplings = overlays
        self.coordinate_contract = contract
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "mesh-assembly",
                "parts": [part.part_id for part in values],
                "couplings": [overlay.coupling_id for overlay in overlays],
                "coordinates": contract.spatial_id,
            }
        )

    def part(self, name: str, /) -> MeshPart:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"Unknown assembly part {name!r}.")

    def require_attachment(self, attachment: MeshInterfaceAttachment, /) -> MeshPart:
        """Return the current owning part of an interface attachment.

        The assembly stays a carrier container: attachments of different parts
        are validated against the current part revisions without welding them.
        """
        from ._interface_binding import MeshInterfaceAttachment

        if not isinstance(attachment, MeshInterfaceAttachment):
            raise TypeError("attachment must be MeshInterfaceAttachment.")
        for part in self.parts:
            if part.name == attachment.part_name:
                attachment.require_current(part)
                return part
        raise ValueError(
            f"Interface attachment part {attachment.part_name!r} is not owned by this "
            "assembly."
        )


__all__ = ["MeshAssembly", "MeshCarrier", "MeshCarrierKind", "MeshPart"]

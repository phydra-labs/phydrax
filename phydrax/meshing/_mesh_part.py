#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._point_cloud as point_cloud_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization._tensor_support as tensor_support_owner
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.iga import IsogeometricPlan
from ..typing import checked
from ._canonical import certify_cell_mesh
from ._lineage import identity_lineage, inherit_mesh_organization, inherit_scope
from ._organization_records import MeshAttribute, MeshAttributeRole
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope


class MeshCarrierKind(StrEnum):
    CELL = "cell"
    TENSOR = "tensor"
    POINT = "point"
    SPLINE = "spline"


type MeshCarrier = (
    CellMeshingResult
    | tensor_support_owner.PreparedTensorGrid
    | point_cloud_owner.PointCloudPlan
    | IsogeometricPlan
)


def _vertex_linear_element(
    element: reference_cell_types_owner.CellGeometryElement, /
) -> bool:
    """Admit vertex descriptors or degree-one scalar maps for coordinate motion."""
    match element:
        case cell_geometry_owner.CellVertexGeometryElement():
            return True
        case _:
            scalar = cell_geometry_owner.require_scalar_coordinate_element(
                element, "MeshPart.geometry"
            )
            return scalar.degree == 1


class MeshPart(StrictModule, NonTrainableState):
    """Named immutable carrier; assembly never tessellates compact representations.

    ``name`` is the stable source identity and ``part_id`` its bound revision.
    Coordinates of every part must already use the assembly coordinate contract.
    """

    name: str = eqx.field(static=True)
    carrier: MeshCarrier
    coordinate_contract: SpatialCoordinateContract
    carrier_kind: MeshCarrierKind = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    part_id: str = eqx.field(static=True)

    @checked
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
            raise ValueError(f"MeshPart.name must be nonempty; got {name_!r}")
        logical_values_id = None
        match carrier:
            case CellMeshingResult():
                contract = (
                    carrier.coordinate_contract
                    if coordinate_contract is None
                    else coordinate_contract
                )
                if contract.spatial_id != carrier.coordinate_contract.spatial_id:
                    raise ValueError(
                        f"MeshPart.coordinate_contract must match the certified carrier contract; got {type(contract).__name__}"
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
                            "MeshPart.carrier must provide collective publication evidence when distributed; got None"
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
                            f"MeshPart.carrier must match logical publication and geometry storage bindings; got result {carrier.result_id!r}"
                        )
                    # Rank-local audits, lowered slots and shard payloads are not
                    # global scientific identity. Collective evidence binds the
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
            case _:
                contract = coordinate_contract
                if contract is None:
                    raise TypeError(
                        f"MeshPart.coordinate_contract must be SpatialCoordinateContract for compact carriers; got {type(contract).__name__}"
                    )
                match carrier:
                    case tensor_support_owner.PreparedTensorGrid():
                        kind = MeshCarrierKind.TENSOR
                        intrinsic = ambient = len(carrier.axis_names)
                        identity, values = carrier.prepared_id, carrier
                    case point_cloud_owner.PointCloudPlan():
                        kind = MeshCarrierKind.POINT
                        intrinsic, ambient = 0, carrier.points.shape[1]
                        identity, values = carrier.plan_id, carrier
                    case IsogeometricPlan():
                        kind = MeshCarrierKind.SPLINE
                        intrinsic, ambient = (
                            carrier.basis.parametric_dimension,
                            carrier.geometry.ambient_dimension,
                        )
                        identity, values = carrier.plan_id, carrier
                    case _:
                        raise TypeError(
                            f"MeshPart.carrier must be a cell, tensor, point, or spline carrier; got {type(carrier).__name__}"
                        )
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
        match carrier:
            case CellMeshingResult():
                entities = carrier.mesh.entity_set(dimension)
                identifier = entities.entity_set_id
                ids = np.asarray(entities.entity_ids)[np.asarray(entities.active_mask)]
            case tensor_support_owner.PreparedTensorGrid():
                layouts = tuple(
                    layout
                    for layout in carrier.entity_layouts
                    if layout.axis_entities.count("interval") == dimension
                    and (entity_set_id is None or layout.entity_set_id == entity_set_id)
                )
                if len(layouts) != 1:
                    raise ValueError(
                        f"MeshPart.entity_set_id must select one tensor layout at the requested dimension; got {len(layouts)} layouts for {dimension}"
                    )
                identifier = layouts[0].entity_set_id
                ids = np.arange(prod(layouts[0].shape), dtype=np.int64)
            case point_cloud_owner.PointCloudPlan():
                if dimension != 0:
                    raise ValueError(
                        f"MeshPart.entity_dimension must be zero for point carriers; got {dimension}"
                    )
                identifier = canonical_fingerprint(
                    {"kind": "mesh-part-points", "plan": carrier.plan_id}
                )
                ids = np.arange(carrier.points.shape[0], dtype=np.int64)
            case IsogeometricPlan():
                if dimension != self.intrinsic_dimension:
                    raise ValueError(
                        f"MeshPart.entity_dimension must select positive-span spline entities; got {dimension} for dimension {self.intrinsic_dimension}"
                    )
                identifier = carrier.topology.topology_id
                ids = np.arange(carrier.topology.cell_count, dtype=np.int64)
            case _:
                raise TypeError(
                    f"MeshPart.carrier must provide cell, tensor, point, or spline entity bindings; got {type(carrier).__name__}"
                )
        if entity_set_id is not None and entity_set_id != identifier:
            raise ValueError(
                f"MeshPart.entity_set_id must match the selected carrier layout; got {entity_set_id!r} instead of {identifier!r}"
            )
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
            raise ValueError(
                f"MeshPart.scope must select this part revision and mesh entities; got {(scope.source_id, scope.source_revision, scope.entity_kind)!r}"
            )
        _, ids = self.entity_binding(
            scope.entity_dimension, entity_set_id=scope.entity_set_id
        )
        if not np.all(np.isin(np.asarray(scope.entity_ids), ids)):
            raise ValueError(
                "MeshPart.scope must contain active carrier entity IDs; got absent or inactive IDs"
            )

    def point_coordinates(self, scope: MeshingScope, /) -> Array:
        """Gather point coordinates from zero-dimensional carrier entities."""
        self.require_scope(scope)
        if scope.entity_dimension != 0:
            raise ValueError(
                f"MeshPart.point_coordinates must use a zero-dimensional scope; got {scope.entity_dimension}"
            )
        carrier = self.carrier
        ids = np.asarray(scope.entity_ids)
        if isinstance(carrier, CellMeshingResult):
            vertex_ids = np.asarray(carrier.mesh.vertex_global_ids)
            rows = {int(value): index for index, value in enumerate(vertex_ids)}
            return carrier.mesh.coordinates[
                jnp.asarray([rows[int(value)] for value in ids])
            ]
        if isinstance(carrier, point_cloud_owner.PointCloudPlan):
            return carrier.points[jnp.asarray(ids)]
        if isinstance(carrier, tensor_support_owner.PreparedTensorGrid):
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
        raise ValueError(
            f"MeshPart.point_coordinates must use a cell, point, or tensor carrier; got {self.carrier_kind.value}"
        )

    @checked
    def with_coordinates(
        self,
        coordinates: ArrayLike,
        /,
        *,
        motion_id: str,
        geometry: cell_geometry_owner.CellGeometrySpec | None = None,
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
            raise TypeError(
                f"MeshPart.carrier must be CellMeshingResult for coordinate motion; got {type(carrier).__name__}"
            )
        points = np.asarray(coordinates)
        if points.dtype.kind not in "iuf":
            raise TypeError(
                f"MeshPart.coordinates must be real arrays; got {points.dtype}"
            )
        points = points.astype(np.float64)
        current = np.asarray(carrier.mesh.coordinates, dtype=np.float64)
        if points.shape != current.shape or not np.all(np.isfinite(points)):
            raise ValueError(
                f"MeshPart.coordinates must be finite with shape {current.shape}; got {points.shape} for part {self.name!r}"
            )
        if np.array_equal(points, current) and geometry is None:
            return self
        elements, routes, old_coordinates = carrier.geometry.resolve(carrier.mesh)
        vertex_aligned = (
            np.array_equal(np.asarray(old_coordinates), current)
            and all(
                np.array_equal(np.asarray(route), np.asarray(block.vertices))
                for route, block in zip(routes, carrier.mesh.blocks, strict=True)
            )
            and all(_vertex_linear_element(element) for element in elements)
        )
        if geometry is None and not vertex_aligned:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "MeshPart.geometry must supply the successor coordinate map for non-vertex-aligned motion; got None",
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
                f"MeshPart.carrier must be recertified for source geometry, classification, or region obligations after motion; got source-bound part {self.name!r}",
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


__all__ = ["MeshCarrier", "MeshCarrierKind", "MeshPart"]

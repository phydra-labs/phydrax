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

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PointCloudPlan, PreparedTensorGrid
from ..discretization._view_support import OwnerLocalCellGeometry
from ..discretization.iga import IsogeometricPlan
from ..typing import checked, ConvertibleToArray
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope


if TYPE_CHECKING:
    from ._coupling import MeshCoupling
    from ._interface_binding import MeshInterfaceAttachment


class MeshCarrierKind(StrEnum):
    CELL = "cell"
    TENSOR = "tensor"
    POINT = "point"
    SPLINE = "spline"


MeshCarrier: TypeAlias = (
    CellMeshingResult
    | OwnerLocalCellGeometry
    | PreparedTensorGrid
    | PointCloudPlan
    | IsogeometricPlan
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
        global_part_id: str | None = None
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
            identity = carrier.result_id
            values = (carrier.mesh, carrier.geometry)
        elif isinstance(carrier, OwnerLocalCellGeometry):
            if name_ != carrier.part_name:
                raise ValueError("Owner-local geometry must retain its global part name.")
            contract = (
                carrier.coordinate_contract
                if coordinate_contract is None
                else coordinate_contract
            )
            if (
                not isinstance(contract, SpatialCoordinateContract)
                or contract.spatial_id != carrier.coordinate_contract_id
            ):
                raise ValueError(
                    "Owner-local geometry changed its certified coordinate frame."
                )
            kind = MeshCarrierKind.CELL
            intrinsic, ambient = (
                carrier.topological_dimension,
                int(np.asarray(carrier.coordinates).shape[1]),
            )
            identity, values = carrier.part_id, carrier.local_view_id
            global_part_id = carrier.part_id
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
        self.part_id = (
            global_part_id
            if global_part_id is not None
            else canonical_fingerprint(
                {
                    "kind": "mesh-part",
                    "name": name_,
                    "carrier_kind": kind.value,
                    "carrier": identity,
                    "values": array_tree_fingerprint(values),
                    "coordinates": contract.spatial_id,
                }
            )
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
        elif isinstance(carrier, OwnerLocalCellGeometry):
            identifier, ids = carrier.entity_binding(dimension)
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
        entity_ids: ConvertibleToArray,
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
        ids = np.asarray(scope.entity_ids)
        carrier = self.carrier
        if isinstance(carrier, OwnerLocalCellGeometry):
            return carrier.point_coordinates(ids)
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


class MeshPartManifest(StrictModule, NonTrainableState):
    """Scientific global part identity; local publication data stays in each rank view.

    The compact owner-local representation intentionally uses new manifest and
    assembly/PyTree identities rather than reusing authority-bearing container IDs.
    """

    name: str = eqx.field(static=True)
    part_id: str = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract
    source_topology_id: str | None = eqx.field(static=True)
    source_geometry_id: str | None = eqx.field(static=True)
    intrinsic_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    distribution_id: str | None = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        part_id: str | None,
        coordinate_contract: SpatialCoordinateContract,
        intrinsic_dimension: int,
        ambient_dimension: int,
        /,
        *,
        source_topology_id: str | None = None,
        source_geometry_id: str | None = None,
        distribution_id: str | None = None,
    ) -> None:
        name_ = str(name).strip()
        if not name_ or not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise ValueError("Part manifests require a name and coordinate contract.")
        part_id_ = "" if part_id is None else str(part_id).strip()
        topology_id = source_topology_id
        geometry_id = source_geometry_id
        intrinsic = int(intrinsic_dimension)
        ambient = int(ambient_dimension)
        if (
            not part_id_
            or isinstance(intrinsic_dimension, (bool, np.bool_))
            or isinstance(ambient_dimension, (bool, np.bool_))
            or intrinsic < 0
            or ambient <= 0
            or ambient < intrinsic
            or (topology_id is not None and not str(topology_id))
            or (geometry_id is not None and not str(geometry_id))
            or distribution_id is not None
            and (not isinstance(distribution_id, str) or not distribution_id)
        ):
            raise ValueError("Part manifest identities or dimensions are invalid.")
        topology_id = None if topology_id is None else str(topology_id)
        geometry_id = None if geometry_id is None else str(geometry_id)
        identity = canonical_fingerprint(
            {
                "kind": "mesh-part-manifest",
                "name": name_,
                "part": part_id_,
                "coordinates": coordinate_contract.spatial_id,
                "source_topology": topology_id,
                "source_geometry": geometry_id,
                "distribution": distribution_id,
                "dimensions": (intrinsic, ambient),
            }
        )
        self.name, self.part_id = name_, part_id_
        self.coordinate_contract = coordinate_contract
        self.source_topology_id, self.source_geometry_id = topology_id, geometry_id
        self.intrinsic_dimension, self.ambient_dimension = intrinsic, ambient
        self.distribution_id = distribution_id
        self.manifest_id = identity

    @classmethod
    def from_part(cls, part: MeshPart, /) -> MeshPartManifest:
        if not isinstance(part, MeshPart):
            raise TypeError("part must be MeshPart.")
        from ..discretization._cell_geometry_validity import cell_geometry_id

        carrier = part.carrier
        if isinstance(carrier, OwnerLocalCellGeometry):
            return cls(
                part.name,
                part.part_id,
                part.coordinate_contract,
                part.intrinsic_dimension,
                part.ambient_dimension,
                source_topology_id=carrier.source_topology_id,
                source_geometry_id=carrier.source_geometry_id,
                distribution_id=carrier.distribution_id,
            )
        if isinstance(carrier, CellMeshingResult):
            return cls(
                part.name,
                part.part_id,
                part.coordinate_contract,
                part.intrinsic_dimension,
                part.ambient_dimension,
                source_topology_id=carrier.mesh.topology_id,
                source_geometry_id=cell_geometry_id(carrier.geometry),
            )
        return cls(
            part.name,
            part.part_id,
            part.coordinate_contract,
            part.intrinsic_dimension,
            part.ambient_dimension,
        )


class MeshAssembly(StrictModule, NonTrainableState):
    """Global part identity plus either complete serial parts or one local view."""

    parts: tuple[MeshPart, ...]
    part_manifest: tuple[MeshPartManifest, ...]
    couplings: tuple[MeshCoupling, ...]
    coordinate_contract: SpatialCoordinateContract
    owner_local: bool = eqx.field(static=True)
    local_rank: int | None = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)
    assembly_view_id: str = eqx.field(static=True)

    def __init__(
        self,
        parts: tuple[MeshPart, ...],
        /,
        *,
        couplings: tuple[MeshCoupling, ...] = (),
        global_parts: tuple[MeshPartManifest, ...] | None = None,
        local_rank: int | None = None,
    ) -> None:
        from ._coupling import MeshCoupling, OversetCoupling

        values = tuple(parts)
        overlays = tuple(couplings)
        owner_local = local_rank is not None
        if owner_local:
            if (
                isinstance(local_rank, bool)
                or not isinstance(local_rank, (int, np.integer))
                or local_rank < 0
            ):
                raise ValueError(
                    "Owner-local assembly rank must be a non-negative integer."
                )
            if global_parts is None or not global_parts:
                raise ValueError(
                    "Owner-local assemblies require the global part manifest."
                )
            if overlays:
                raise ValueError(
                    "Owner-local assemblies prepare coupling routes collectively."
                )
            if any(
                not isinstance(part, MeshPart)
                or not isinstance(part.carrier, OwnerLocalCellGeometry)
                or part.carrier.owner_rank != int(local_rank)
                for part in values
            ):
                raise ValueError(
                    "Owner-local assemblies accept only this rank's compact cell views."
                )
        elif (
            not values
            or not all(isinstance(part, MeshPart) for part in values)
            or any(isinstance(part.carrier, OwnerLocalCellGeometry) for part in values)
        ):
            raise ValueError("Serial mesh assemblies require complete MeshPart carriers.")
        if len({part.name for part in values}) != len(values):
            raise ValueError("Every local assembly part must have one unique name.")
        values = tuple(sorted(values, key=lambda part: part.name))
        manifest = (
            tuple(global_parts)
            if global_parts is not None
            else tuple(MeshPartManifest.from_part(part) for part in values)
        )
        if not manifest or any(
            not isinstance(item, MeshPartManifest) for item in manifest
        ):
            raise ValueError(
                "Mesh assemblies require a non-empty typed global part manifest."
            )
        manifest = tuple(sorted(manifest, key=lambda item: item.name))
        if len({item.name for item in manifest}) != len(manifest):
            raise ValueError("Global mesh part manifest names must be unique.")
        manifest_by_name = {item.name: item for item in manifest}
        if owner_local and any(
            item.distribution_id is None
            or item.source_topology_id is None
            or item.source_geometry_id is None
            for item in manifest
        ):
            raise ValueError(
                "Owner-local manifests require current directory and source identities."
            )
        local_names = {part.name for part in values}
        manifest_names = set(manifest_by_name)
        if owner_local and local_names != manifest_names:
            raise ValueError(
                "Every owner-local manifest part requires a local view, including empty views."
            )
        if not local_names.issubset(manifest_by_name):
            raise ValueError("A local part is absent from the global manifest.")
        if not owner_local and local_names != manifest_names:
            raise ValueError("Serial assembly parts must cover the global part manifest.")
        contract = manifest[0].coordinate_contract
        if any(
            item.coordinate_contract.spatial_id != contract.spatial_id
            or item.ambient_dimension != manifest[0].ambient_dimension
            for item in manifest
        ):
            raise ValueError(
                "Global mesh parts must share one coordinate contract and ambient dimension."
            )
        for part in values:
            identity = manifest_by_name[part.name]
            carrier = part.carrier
            if (
                part.part_id != identity.part_id
                or part.coordinate_contract.spatial_id
                != identity.coordinate_contract.spatial_id
                or part.intrinsic_dimension != identity.intrinsic_dimension
                or part.ambient_dimension != identity.ambient_dimension
            ):
                raise ValueError(
                    "Local part content differs from its global manifest identity."
                )
            if isinstance(carrier, OwnerLocalCellGeometry) and (
                identity.distribution_id != carrier.distribution_id
                or identity.source_topology_id != carrier.source_topology_id
                or identity.source_geometry_id != carrier.source_geometry_id
            ):
                raise ValueError(
                    "Local cell view differs from its global scientific manifest."
                )
        if not all(isinstance(overlay, MeshCoupling) for overlay in overlays):
            raise TypeError("couplings must contain MeshCoupling values.")
        if len({overlay.coupling_id for overlay in overlays}) != len(overlays):
            raise ValueError("Assembly coupling overlays must be unique.")
        by_name = {part.name: part for part in values}
        for overlay in overlays:
            for scope in (overlay.source_scope, overlay.target_scope):
                if scope.source_id not in by_name:
                    raise ValueError("Coupling endpoint is not local to this assembly.")
                by_name[scope.source_id].require_scope(scope)
        receptors: dict[tuple[str, str], set[int]] = {}
        holes: dict[tuple[str, str], set[int]] = {}
        for overlay in overlays:
            if not isinstance(overlay, OversetCoupling):
                continue
            key = (overlay.target_scope.source_id, overlay.target_scope.entity_set_id)
            ids = set(np.asarray(overlay.target_scope.entity_ids).tolist())
            owned = receptors.setdefault(key, set())
            if owned.intersection(ids):
                raise ValueError(
                    "Every overset receptor must have exactly one donor overlay."
                )
            owned.update(ids)
            if overlay.hole_scope is not None:
                by_name[overlay.hole_scope.source_id].require_scope(overlay.hole_scope)
                holes.setdefault(key, set()).update(
                    np.asarray(overlay.hole_scope.entity_ids).tolist()
                )
        for key, ids in receptors.items():
            if ids.intersection(holes.get(key, set())):
                raise ValueError("Overset receptor and hole ownership must be disjoint.")
        for overlay in overlays:
            if not isinstance(overlay, OversetCoupling):
                continue
            key = (overlay.source_scope.source_id, overlay.source_scope.entity_set_id)
            forbidden = receptors.get(key, set()) | holes.get(key, set())
            if overlay.field_query is not None:
                used = set(np.asarray(overlay.source_scope.entity_ids).tolist())
            else:
                assert overlay.donor_ids is not None
                assert overlay.donor_weights is not None
                used = set(
                    np.asarray(overlay.donor_ids)[
                        np.asarray(overlay.donor_weights) > 0
                    ].tolist()
                )
            if used.intersection(forbidden):
                raise ValueError("Overset donors cannot be assembly receptors or holes.")
        overlays = tuple(sorted(overlays, key=lambda overlay: overlay.coupling_id))
        self.parts, self.part_manifest, self.couplings = values, manifest, overlays
        self.coordinate_contract = contract
        self.owner_local = owner_local
        self.local_rank = None if not owner_local else int(local_rank)
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "mesh-assembly",
                "parts": [item.manifest_id for item in manifest],
                "couplings": [overlay.coupling_id for overlay in overlays],
                "coordinates": contract.spatial_id,
            }
        )
        self.assembly_view_id = canonical_fingerprint(
            {
                "kind": "mesh-assembly-local-view",
                "assembly": self.assembly_id,
                "rank": self.local_rank,
                "parts": [
                    (
                        part.name,
                        part.part_id,
                        getattr(part.carrier, "local_view_id", None),
                    )
                    for part in values
                ],
            }
        )

    @property
    def global_part_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.part_manifest)

    def declared_part(self, name: str, /) -> MeshPartManifest:
        for identity in self.part_manifest:
            if identity.name == name:
                return identity
        raise KeyError(f"Unknown assembly part {name!r}.")

    def part(self, name: str, /) -> MeshPart:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"Assembly part {name!r} has no local carrier on this rank.")

    def require_attachment(self, attachment: MeshInterfaceAttachment, /) -> MeshPart:
        """Return the current local owner of an interface attachment."""
        from ._interface_binding import MeshInterfaceAttachment

        if not isinstance(attachment, MeshInterfaceAttachment):
            raise TypeError("attachment must be MeshInterfaceAttachment.")
        for part in self.parts:
            if part.name == attachment.part_name:
                attachment.require_current(part)
                return part
        raise ValueError(
            f"Interface attachment part {attachment.part_name!r} is not local to this assembly."
        )


__all__ = [
    "MeshAssembly",
    "MeshCarrier",
    "MeshCarrierKind",
    "MeshPart",
    "MeshPartManifest",
]

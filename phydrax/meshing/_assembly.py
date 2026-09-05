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
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PointCloudPlan, PreparedTensorGrid
from ..discretization.iga import IsogeometricPlan
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope


if TYPE_CHECKING:
    from ._coupling import MeshCoupling


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
    ):
        name_ = str(name).strip()
        if not name_:
            raise ValueError("Mesh part names must be non-empty.")
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
                "values": array_tree_fingerprint(values),
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

    def require_scope(self, scope: MeshingScope, /) -> None:
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
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


class MeshAssembly(StrictModule, NonTrainableState):
    """Named parts and revision-bound coupling overlays, with no implicit welding."""

    parts: tuple[MeshPart, ...]
    couplings: tuple[MeshCoupling, ...]
    coordinate_contract: SpatialCoordinateContract
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self, parts: tuple[MeshPart, ...], /, *, couplings: tuple[MeshCoupling, ...] = ()
    ):
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
            used = set(
                np.asarray(overlay.donor_ids)[
                    np.asarray(overlay.donor_weights) > 0
                ].tolist()
            )
            if used.intersection(forbidden):
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


__all__ = ["MeshAssembly", "MeshCarrier", "MeshCarrierKind", "MeshPart"]

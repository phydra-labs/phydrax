#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared mesh-certificate identity bindings and phase measurements."""

from __future__ import annotations

from typing import final, Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked, Dim, parse
from . import _embedding_state as embedding_state_owner
from ._embedding_state import MeshCertificateFinding, MeshCertificateLimits


type MeshCertificateStatus = Literal["certified", "violated", "unresolved"]


class MeshCellDim(Dim, minimum=1):
    """Cells of one certified mesh."""


@final
class MeshCertificateBinding(StrictModule, NonTrainableState):
    """Identity of everything a mesh certificate was computed from."""

    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_scope: embedding_state_owner.CoordinateMapScope = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    source_revision: str | None = eqx.field(static=True)
    junction_vertices: tuple[int, ...] = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_scope: embedding_state_owner.CoordinateMapScope,
        limits: MeshCertificateLimits,
        /,
        *,
        source_id: str | None = None,
        source_revision: str | None = None,
        junction_vertices: tuple[int, ...] = (),
    ) -> None:
        scope = parse(
            coordinate_scope, embedding_state_owner.CoordinateMapScope, "coordinate_scope"
        )
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.geometry_id = cell_geometry_id(geometry)
        self.coordinate_scope = scope
        self.source_id = source_id
        self.source_revision = source_revision
        self.junction_vertices = junction_vertices
        self.limits_id = limits.limits_id
        self.binding_id = canonical_fingerprint(
            {
                "kind": "mesh-certificate-binding",
                "mesh": mesh.mesh_id,
                "topology": mesh.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "geometry": self.geometry_id,
                "coordinate_scope": scope,
                "source": source_id,
                "source_revision": source_revision,
                "junction_vertices": junction_vertices,
                "limits": limits.limits_id,
            }
        )

    @checked
    def require(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        """Refuse reuse for another mesh topology, coordinate array or layout."""

        if (
            mesh.mesh_id != self.mesh_id
            or mesh.topology_id != self.topology_id
            or geometry.geometry_layout_id != self.geometry_layout_id
            or cell_geometry_id(geometry) != self.geometry_id
        ):
            raise ValueError(
                "MeshCertificateBinding.geometry must match its mesh, topology, coordinate array and layout; "
                f"got mesh={mesh.mesh_id!r}, geometry={cell_geometry_id(geometry)!r}"
            )


def certificate_status(
    findings: tuple[MeshCertificateFinding, ...], /
) -> MeshCertificateStatus:
    if any(finding.status == "violated" for finding in findings):
        return "violated"
    if findings:
        return "unresolved"
    return "certified"


@checked
def active_expression_work(*, budget: CoordinateEnclosureBudget) -> tuple[int, int, int]:
    """Measured work, additional storage peak, and admission bound of this phase."""
    measurement = budget.current_measurement
    return (
        measurement.work_units,
        measurement.peak_bytes_upper,
        measurement.required_work_units,
    )


__all__ = [
    "MeshCellDim",
    "MeshCertificateBinding",
    "MeshCertificateStatus",
    "active_expression_work",
    "certificate_status",
]

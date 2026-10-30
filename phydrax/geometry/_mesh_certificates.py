#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Input-bound mesh certificate identities and mathematical status."""

from __future__ import annotations

import math
from typing import Literal

import jax

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..typing import checked, parse
from ._embedding_state import (
    CoordinateMapScope,
    MeshCertificateFinding,
)
from ._source_boundary_records import SourceBoundaryQuery


type MeshCertificateStatus = Literal["certified", "violated", "unresolved"]


def _same_tree_structure(
    first: jax.tree_util.PyTreeDef, second: jax.tree_util.PyTreeDef, /
) -> bool:
    """Compare static metadata with the same missing-value semantics as leaves."""
    if first == second:
        return True
    first_data, second_data = first.node_data(), second.node_data()
    if first_data is None or second_data is None:
        return first_data is None and second_data is None
    if first_data[0] is not second_data[0] or not _same_tree_content(
        first_data[1], second_data[1]
    ):
        return False
    first_children, second_children = first.children(), second.children()
    return len(first_children) == len(second_children) and all(
        _same_tree_structure(left, right)
        for left, right in zip(first_children, second_children, strict=True)
    )


def _same_tree_content(first: object, second: object, /) -> bool:
    """Compare immutable scientific records by static tree structure and every leaf."""
    if first is second:
        return True
    if type(first) is not type(second):
        return False
    first_leaves, first_definition = jax.tree_util.tree_flatten(first)
    second_leaves, second_definition = jax.tree_util.tree_flatten(second)
    if len(first_leaves) != len(second_leaves) or not _same_tree_structure(
        first_definition, second_definition
    ):
        return False
    import numpy as np

    for left, right in zip(first_leaves, second_leaves, strict=True):
        if isinstance(left, (jax.Array, np.ndarray, np.generic)) or isinstance(
            right, (jax.Array, np.ndarray, np.generic)
        ):
            left_array, right_array = np.asarray(left), np.asarray(right)
            if (
                left_array.shape != right_array.shape
                or left_array.dtype != right_array.dtype
            ):
                return False
            try:
                equal = np.array_equal(left_array, right_array, equal_nan=True)
            except TypeError:
                equal = np.array_equal(left_array, right_array)
            if not equal:
                return False
        else:
            if (
                isinstance(left, float)
                and isinstance(right, float)
                and math.isnan(left)
                and math.isnan(right)
            ):
                continue
            try:
                equal = left == right
            except (TypeError, ValueError):
                return False
            if not isinstance(equal, bool) or not equal:
                return False
    return True


class MeshCertificateBinding(StrictModule, NonTrainableState):
    """Retain the actual immutable scientific inputs of one certificate request."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    source: SourceBoundaryQuery | None
    mesh_id: str
    topology_id: str
    geometry_layout_id: str
    geometry_id: str
    coordinate_scope: CoordinateMapScope
    source_id: str | None
    source_revision: str | None
    junction_vertices: tuple[int, ...]
    binding_id: str

    @checked
    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_scope: CoordinateMapScope,
        /,
        *,
        source: SourceBoundaryQuery | None = None,
        source_id: str | None = None,
        source_revision: str | None = None,
        junction_vertices: tuple[int, ...] = (),
    ) -> None:
        scope = parse(coordinate_scope, CoordinateMapScope, "coordinate_scope")
        if source is not None and not isinstance(source, StrictModule):
            raise TypeError("source must be an immutable source-query record.")
        if source is not None:
            source_id_ = canonical_identifier(source.source_id, "source_id")
            source_revision_ = canonical_identifier(
                source.source_revision, "source_revision"
            )
            if (
                source_id is not None
                and canonical_identifier(source_id, "source_id") != source_id_
            ):
                raise ValueError("source_id must match the supplied source authority.")
            if (
                source_revision is not None
                and canonical_identifier(source_revision, "source_revision")
                != source_revision_
            ):
                raise ValueError(
                    "source_revision must match the supplied source authority."
                )
        else:
            source_id_ = (
                None
                if source_id is None
                else canonical_identifier(source_id, "source_id")
            )
            source_revision_ = (
                None
                if source_revision is None
                else canonical_identifier(source_revision, "source_revision")
            )
            if (source_id_ is None) != (source_revision_ is None):
                raise ValueError(
                    "source_id and source_revision must be supplied together."
                )
        junctions = tuple(junction_vertices)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in junctions
        ):
            raise ValueError(
                "junction_vertices must contain nonnegative integer global IDs."
            )
        if tuple(sorted(set(junctions))) != junctions:
            raise ValueError("junction_vertices must be unique and increasing.")
        geometry_id = cell_geometry_id(geometry)
        self.mesh = mesh
        self.geometry = geometry
        self.source = source
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.geometry_id = geometry_id
        self.coordinate_scope = scope
        self.source_id = source_id_
        self.source_revision = source_revision_
        self.junction_vertices = junctions
        self.binding_id = canonical_fingerprint(
            {
                "kind": "mesh-certificate-binding",
                "mesh": self.mesh_id,
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "geometry": self.geometry_id,
                "coordinate_scope": scope,
                "source": source_id_,
                "source_revision": source_revision_,
                "junction_vertices": junctions,
            }
        )

    @checked
    def matches(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        source: SourceBoundaryQuery | None = None,
    ) -> bool:
        """Accept a freshly reconstructed record only when all retained data match."""
        if (
            not _same_tree_content(mesh, self.mesh)
            or not _same_tree_content(geometry, self.geometry)
            or mesh.mesh_id != self.mesh_id
            or mesh.topology_id != self.topology_id
            or geometry.geometry_layout_id != self.geometry_layout_id
        ):
            return False
        current_geometry_id = cell_geometry_id(geometry)
        if current_geometry_id != self.geometry_id:
            return False
        if source is None:
            return self.source is None
        return (
            self.source is not None
            and _same_tree_content(source, self.source)
            and source.source_id == self.source_id
            and source.source_revision == self.source_revision
        )

    @checked
    def require(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        source: SourceBoundaryQuery | None = None,
    ) -> None:
        if not self.matches(mesh, geometry, source=source):
            raise ValueError(
                "MeshCertificateBinding must match the original immutable mesh, coordinate map, and source authority."
            )


def certificate_status(
    findings: tuple[MeshCertificateFinding, ...], /
) -> MeshCertificateStatus:
    if any(finding.status == "violated" for finding in findings):
        return "violated"
    if findings:
        return "unresolved"
    return "certified"


__all__ = [
    "MeshCertificateBinding",
    "MeshCertificateStatus",
    "certificate_status",
]

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scientific coordinate-scope identity shared by preparation caches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh


def _geometry_id(geometry: CellGeometrySpec, /) -> str:
    # Coordinate identity → geometry validity → source enclosure → coordinate identity.
    from ._cell_geometry_validity import cell_geometry_id

    return cell_geometry_id(geometry)


def coordinate_source_bank_key(geometry: CellGeometrySpec, /) -> tuple[str, str]:
    """Bind the source bank to its declared geometry and current numerical tree."""
    geometry_id = _geometry_id(geometry)
    live_tree_id = (
        geometry.storage_id
        if geometry.storage_id is not None
        else canonical_fingerprint(array_tree_fingerprint(geometry))
    )
    return geometry_id, live_tree_id


def coordinate_scope_key(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[str, str, str]:
    """Scientific source ownership and every live coordinate/reference bank."""

    geometry_id = _geometry_id(geometry)
    live_tree_id = (
        geometry.storage_id
        if geometry.storage_id is not None
        else canonical_fingerprint(array_tree_fingerprint((mesh, geometry)))
    )
    return mesh.mesh_id, geometry_id, live_tree_id

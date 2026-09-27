#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..discretization import CellMesh
from ._complex import CellSubcomplex
from ._integer import ExactIntegerCOO
from ._maps import CellularChainMap


if TYPE_CHECKING:
    from ..meshing._lineage import MeshLineage


def finite_element_topology_transfer(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    lineage: MeshLineage,
    degree_maps: Sequence[ExactIntegerCOO],
    /,
) -> CellularChainMap:
    """Bind mesh-adaptation lineage to an independently verified exact chain map."""
    # Lazy: phydrax.meshing transitively imports packages that import topology.
    from ..meshing._lineage import MeshLineage

    if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError("Finite-element topology transfer requires two CellMesh values.")
    if not isinstance(lineage, MeshLineage):
        raise TypeError("lineage must be a MeshLineage.")
    if lineage.source_topology_id != source_mesh.topology_id:
        raise ValueError("Mesh lineage source topology does not match the source mesh.")
    if lineage.target_topology_id != target_mesh.topology_id:
        raise ValueError("Mesh lineage target topology does not match the target mesh.")
    source = CellSubcomplex.full(source_mesh.topology)
    target = CellSubcomplex.full(target_mesh.topology)
    return CellularChainMap(
        source,
        target,
        degree_maps,
        map_id=f"finite-element-topology-transfer:{lineage.lineage_id}",
    )


__all__ = ["finite_element_topology_transfer"]

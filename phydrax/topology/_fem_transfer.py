#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

from ..discretization import CellMesh
from ..discretization.fem import FiniteElementAdaptationMap, FiniteElementHPLineage
from ._complex import CellSubcomplex
from ._integer import ExactIntegerCOO
from ._maps import CellularChainMap


def finite_element_topology_transfer(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    lineage: FiniteElementHPLineage | FiniteElementAdaptationMap,
    degree_maps: Sequence[ExactIntegerCOO],
    /,
) -> CellularChainMap:
    """Bind FE lineage to an independently verified exact cellular chain map."""
    if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError("Finite-element topology transfer requires two CellMesh values.")
    if isinstance(lineage, FiniteElementHPLineage):
        source_topology_id = lineage.source_topology_id
        target_topology_id = lineage.target_topology_id
        lineage_id = lineage.lineage_id
    elif isinstance(lineage, FiniteElementAdaptationMap):
        source_topology_id = lineage.source_mesh.topology_id
        target_topology_id = lineage.target_mesh.topology_id
        lineage_id = lineage.adaptation_id
    else:
        raise TypeError("lineage must be an FE refinement or adaptation map.")
    if source_topology_id != source_mesh.topology_id:
        raise ValueError("FE lineage source topology does not match the source mesh.")
    if target_topology_id != target_mesh.topology_id:
        raise ValueError("FE lineage target topology does not match the target mesh.")
    source = CellSubcomplex.full(source_mesh.topology)
    target = CellSubcomplex.full(target_mesh.topology)
    return CellularChainMap(
        source,
        target,
        degree_maps,
        map_id=f"finite-element-topology-transfer:{lineage_id}",
    )


__all__ = ["finite_element_topology_transfer"]

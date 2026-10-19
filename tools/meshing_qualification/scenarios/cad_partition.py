# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.gmsh import _three_dimensional_partition


def qualify_cad_partition() -> dict[str, object]:

    with TemporaryDirectory(prefix="phydrax-cad-partition-") as temporary:
        result = _three_dimensional_partition(Path(temporary))
    if result.topological_dimension != 3 or not result.named_solid_entity_ids:
        raise qualification_failure(
            "qualify_cad_partition",
            ("CAD partition did not produce solid regions."),
            {"result": result},
        )
    if not result.association_graph.graph_id or not result.revision.revision_id:
        raise qualification_failure(
            "qualify_cad_partition",
            ("CAD partition omitted revision association evidence."),
            {"result": result},
        )
    return {
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solid_regions": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "face_patches": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_cad_partition()

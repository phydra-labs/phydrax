# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.gmsh import _planar_partition


def qualify_planar_semantic() -> dict[str, object]:

    with TemporaryDirectory(prefix="phydrax-planar-semantic-") as temporary:
        embedding, result = _planar_partition(Path(temporary) / "planar.brep")
    if (
        result.topological_dimension != 2
        or result.model.topology.num_solids != 0
        or not result.named_region_entity_ids
        or not result.named_patch_entity_ids
    ):
        raise qualification_failure(
            "qualify_planar_semantic",
            ("Planar partition did not preserve two-dimensional semantics."),
            {"result": result},
        )
    return {
        "embedding": embedding.embedding_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "region_entity_kind": result.region_entity_kind,
        "patch_entity_kind": result.patch_entity_kind,
        "regions": {name: len(ids) for name, ids in result.named_region_entity_ids},
        "patches": {name: len(ids) for name, ids in result.named_patch_entity_ids},
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_planar_semantic()

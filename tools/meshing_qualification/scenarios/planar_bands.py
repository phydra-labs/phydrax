# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.gmsh import _planar_partition, _region_controls


def qualify_planar_bands() -> dict[str, object]:

    with TemporaryDirectory(prefix="phydrax-planar-bands-") as temporary:
        embedding, partition = _planar_partition(Path(temporary) / "planar.brep")
        patch = next(
            value for value in partition.patches if len(value.adjacent_region_ids) == 2
        )
        schedule = phx.meshing.LayerSchedule.geometric(2, 0.05, growth_rate=1.0)
        control = phx.meshing.PlanarBandControl(
            patch.name,
            {name: schedule for name in patch.adjacent_region_ids},
            0.1,
        )
        source = phx.meshing.prepare_planar_bands(
            phx.meshing.PlanarBandPlan(partition, embedding, (control,)),
            destination=Path(temporary) / "bands.brep",
        )
        provider = phx.meshing.GmshProvider()
        scope = provider.whole_scope(source, 2)
        band_patch_controls = tuple(
            phx.meshing.PatchControl(
                band_patch.name,
                provider.entity_scope(source, band_patch.entity_ids),
                band_patch.adjacent_region_ids,
            )
            for band_patch in source.partition.patches
            if len(set(band_patch.adjacent_region_ids))
            == len(band_patch.adjacent_region_ids)
        )
        specification = phx.meshing.SurfaceMeshingSpec(
            phx.meshing.CellMeshingTarget(
                2,
                2,
                phx.meshing.CellFamilyPolicy(required=("quadrilateral",)),
            ),
            scope,
            planar_embedding=embedding,
            size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
            region_controls=_region_controls(provider, source.partition),
            patch_controls=band_patch_controls,
        )
        mesh_result = provider.plan(source, specification).execute()
    if not mesh_result.audit.passed or not mesh_result.compliance.passed:
        raise qualification_failure(
            "qualify_planar_bands",
            ("Planar band mesh failed audit or compliance."),
            {"mesh_result": mesh_result},
        )
    achieved = dict(mesh_result.compliance.achieved)
    front_residuals = {}
    for layer in source.layer_partitions:
        key = f"planar_band:{layer.control_id}:{layer.region_name}:front:{layer.layer_index + 1}"
        residual = achieved[f"{key}:maximum_residual"]
        if residual > 1.0e-12:
            raise qualification_failure(
                "qualify_planar_bands",
                ("Planar band front departed from its CAD level."),
                {"residual": residual},
            )
        front_residuals[key] = residual
    return {
        "plan_id": source.plan_id,
        "result_id": source.result_id,
        "mesh_result_id": mesh_result.result_id,
        "partition_revision": source.partition.revision.revision_id,
        "layer_partitions": len(source.layer_partitions),
        "front_residuals": front_residuals,
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_planar_bands()

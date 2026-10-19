# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import _contract, qualification_failure


def _planar_region(name: str, x0: float, x1: float) -> phx.geometry.PlanarMeshRegion:
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=name,
    )


def _planar_partition(destination: Path) -> Any:

    contract = _contract()
    embedding = phx.geometry.PlanarEmbedding(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    operands = (
        phx.geometry.PlanarPartitionOperand(
            "left",
            _planar_region("left", 0.0, 1.0),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.PlanarPartitionOperand(
            "right",
            _planar_region("right", 1.0, 2.0),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.PlanarPartitionPlan(
        contract,
        embedding,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return embedding, phx.geometry.partition_planar(plan, destination=destination)


def _persist_partition_operand(
    shape: Any, path: Path, contract: phx.SpatialCoordinateContract
) -> Any:
    return phx.interchange.persist_occt_shape(
        shape,
        path,
        coordinate_contract=contract,
    )


def _three_dimensional_partition(destination: Path) -> Any:

    # Optional provider imports are admitted only for this fixture workflow.
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    contract = _contract()
    left_shape = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
    right_shape = BRepPrimAPI_MakeBox(gp_Pnt(1.0, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
    operands = (
        phx.geometry.BRepPartitionOperand(
            "left",
            _persist_partition_operand(left_shape, destination / "left.brep", contract),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.BRepPartitionOperand(
            "right",
            _persist_partition_operand(right_shape, destination / "right.brep", contract),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.BRepPartitionPlan(
        contract,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return phx.geometry.partition_brep(plan, destination=destination / "composite.brep")


def _region_controls(provider: Any, partition: Any, model: Any = None) -> Any:
    source = partition.model if model is None else model
    return tuple(
        phx.meshing.RegionControl(
            provider.entity_scope(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in partition.regions
    )


def _patch_controls(provider: Any, partition: Any, model: Any = None) -> Any:
    source = partition.model if model is None else model
    return tuple(
        phx.meshing.PatchControl(
            patch.name,
            provider.entity_scope(source, patch.entity_ids),
            patch.adjacent_region_ids,
        )
        for patch in partition.patches
    )


def _semantic_gmsh_specification(
    provider: Any,
    partition: Any,
    model: Any = None,
    /,
    *,
    scoped_size: bool,
) -> Any:
    source = partition.model if model is None else model
    whole = provider.whole_scope(source, 3)
    controls = (
        phx.meshing.UniformSizeControl(
            whole,
            0.35,
            strength=(
                phx.meshing.SizeControlStrength.SOFT
                if scoped_size
                else phx.meshing.SizeControlStrength.HARD
            ),
        ),
    )
    if scoped_size:
        local = provider.entity_scope(source, partition.regions[0].entity_ids)
        controls = (
            *controls,
            phx.meshing.UniformSizeControl(local, 0.16),
        )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=1,
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=controls,
        region_controls=_region_controls(provider, partition, source),
        patch_controls=_patch_controls(provider, partition, source),
    )
    return specification, controls


def _require_semantic_gmsh_result(result: Any, partition: Any) -> None:

    if not result.audit.passed or not result.compliance.passed:
        raise qualification_failure(
            "_require_semantic_gmsh_result",
            ("Semantic Gmsh result failed audit or compliance."),
            {"result": result},
        )
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    expected_regions = {region.name for region in partition.regions}
    if set(zones) != expected_regions:
        raise qualification_failure(
            "_require_semantic_gmsh_result",
            ("Semantic Gmsh result lost region ownership."),
            {"expected_regions": expected_regions, "zones": zones},
        )
    expected_cells = np.asarray(result.mesh.entity_set(3).entity_ids)
    observed_cells = np.concatenate(
        tuple(np.asarray(zone.scope.entity_ids) for zone in zones.values())
    )
    if np.unique(observed_cells).size != observed_cells.size or not np.array_equal(
        np.sort(observed_cells), np.sort(expected_cells)
    ):
        raise qualification_failure(
            "_require_semantic_gmsh_result",
            ("Semantic Gmsh region zones are not exclusive and exhaustive."),
            {"expected_cells": expected_cells, "observed_cells": observed_cells},
        )
    patches = {patch.name: patch for patch in result.patches}
    expected_patches = {patch.name for patch in partition.patches}
    if set(patches) != expected_patches:
        raise qualification_failure(
            "_require_semantic_gmsh_result",
            ("Semantic Gmsh result lost declared patches."),
            {"expected_patches": expected_patches, "patches": patches},
        )


def qualify_gmsh() -> dict[str, object]:

    # Optional provider imports are admitted only for this fixture workflow.
    import build123d as bd

    with TemporaryDirectory(prefix="phydrax-gmsh-semantic-") as temporary:
        root = Path(temporary)
        partition = _three_dimensional_partition(root)
        provider = phx.meshing.GmshProvider()
        specification, _controls = _semantic_gmsh_specification(
            provider,
            partition,
            scoped_size=False,
        )
        plan = provider.plan(partition.model, specification)
        result = plan.execute()
        _require_semantic_gmsh_result(result, partition)

        replayed_model = phx.interchange.import_occt_brep(
            partition.model.source_id,
            coordinate_contract=partition.model.coordinate_contract,
        )
        replayed_specification, _ = _semantic_gmsh_specification(
            provider,
            partition,
            replayed_model,
            scoped_size=False,
        )
        replayed_plan = provider.plan(replayed_model, replayed_specification)
        replayed = replayed_plan.execute()
        _require_semantic_gmsh_result(replayed, partition)
        replay_preserved = (
            partition.model.source_revision == replayed_model.source_revision
            and plan.plan_id == replayed_plan.plan_id
            and result.result_id == replayed.result_id
        )
        if not replay_preserved:
            raise qualification_failure(
                "qualify_gmsh",
                ("Native BRep replay changed semantic Gmsh identity."),
                {"replay_preserved": replay_preserved},
            )

        scoped_specification, scoped_controls = _semantic_gmsh_specification(
            provider,
            partition,
            scoped_size=True,
        )
        scoped = provider.plan(partition.model, scoped_specification).execute()
        _require_semantic_gmsh_result(scoped, partition)
        scoped_evidence = dict(scoped.compliance.achieved)
        if any(
            f"size:{control.control_id}:minimum_edge" not in scoped_evidence
            or f"size:{control.control_id}:maximum_edge" not in scoped_evidence
            for control in scoped_controls
        ):
            raise qualification_failure(
                "qualify_gmsh",
                ("Scoped Gmsh sizing lacks per-control evidence."),
                {
                    "scoped_controls": scoped_controls,
                    "scoped_evidence": scoped_evidence,
                },
            )

        phx.interchange.persist_occt_shape(
            bd.Box(3.0, 1.0, 1.0).wrapped,
            partition.model.source_id,
            coordinate_contract=partition.model.coordinate_contract,
            overwrite=True,
        )
        stale_rejected = False
        try:
            plan.execute()
        except phx.meshing.MeshingFailure as failure:
            if failure.category is not phx.meshing.MeshingFailureCategory.INVALID_SOURCE:
                raise
            stale_rejected = True
        if not stale_rejected:
            raise qualification_failure(
                "qualify_gmsh",
                ("Semantic Gmsh plan accepted replaced BRep bytes."),
                {"stale_rejected": stale_rejected},
            )

    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "result_id": result.result_id,
        "plan_id": plan.plan_id,
        "source_revision": partition.model.source_revision,
        "regions": len(partition.regions),
        "patches": len(partition.patches),
        "cells": result.mesh.entity_set(3).count,
        "replay_preserved": replay_preserved,
        "stale_rejected": stale_rejected,
        "scoped_size_controls": [control.control_id for control in scoped_controls],
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_gmsh()

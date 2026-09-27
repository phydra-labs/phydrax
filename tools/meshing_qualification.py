#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._meshcore import meshcore_available, meshcore_identity


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _runtime_identity() -> dict[str, object]:
    """Build, environment, backend, topology, and precision identity of this run."""
    meshcore = meshcore_identity() if meshcore_available() else "unavailable"
    build_id = canonical_fingerprint(
        {
            "kind": "meshing-qualification-build",
            "phydrax": importlib.metadata.version("phydrax"),
            "phydrax_path": str(Path(phx.__file__).resolve().parent),
            "meshcore": meshcore,
        }
    )
    environment_id = canonical_fingerprint(
        {
            "kind": "meshing-qualification-environment",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        }
    )
    identity = phx.qualification.QualificationRuntimeIdentity(
        build_id,
        environment_id,
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return {**identity.to_record(), "meshcore": meshcore}


def _measured(call: Callable[[], object], /) -> tuple[object, dict[str, float]]:
    """Run one stage; report wall seconds and the process peak resident bytes."""
    started = perf_counter()
    value = call()
    elapsed = perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports ru_maxrss in bytes, Linux in KiB.
    scale = 1 if sys.platform == "darwin" else 1024
    return value, {"wall_seconds": elapsed, "peak_resident_bytes": peak * scale}


def _quality_summary(report: phx.meshing.CellQualityReport, /) -> dict[str, object]:
    return {
        "minimum_mean_ratio": report.minimum_mean_ratio,
        "maximum_aspect_ratio": report.maximum_aspect_ratio,
        "minimum_scaled_jacobian": report.minimum_scaled_jacobian,
        "minimum_angle": report.minimum_angle,
        "maximum_angle": report.maximum_angle,
        "sampled_invalid_count": report.sampled_invalid_count,
    }


def _result_identity(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    return {
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "vertices": result.mesh.coordinates.shape[0],
        "cells": sum(block.cell_count for block in result.mesh.blocks),
        "audit_passed": result.audit.passed,
    }


def _missing_dependency(dependency: str, detail: str, /) -> dict[str, object]:
    """Explicit record of a scenario that cannot run in this environment."""
    return {
        "status": "missing-dependency",
        "dependency": dependency,
        "detail": detail,
        "runtime": _runtime_identity(),
    }


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
    return phx.geometry.persist_occt_shape(
        shape,
        path,
        coordinate_contract=contract,
    )


def _three_dimensional_partition(destination: Path) -> Any:
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
        raise RuntimeError("Semantic Gmsh result failed audit or compliance.")
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    expected_regions = {region.name for region in partition.regions}
    if set(zones) != expected_regions:
        raise RuntimeError("Semantic Gmsh result lost exact region ownership.")
    expected_cells = np.asarray(result.mesh.entity_set(3).entity_ids)
    observed_cells = np.concatenate(
        tuple(np.asarray(zone.scope.entity_ids) for zone in zones.values())
    )
    if np.unique(observed_cells).size != observed_cells.size or not np.array_equal(
        np.sort(observed_cells), np.sort(expected_cells)
    ):
        raise RuntimeError("Semantic Gmsh region zones are not exclusive and exhaustive.")
    patches = {patch.name: patch for patch in result.patches}
    expected_patches = {patch.name for patch in partition.patches}
    if set(patches) != expected_patches:
        raise RuntimeError("Semantic Gmsh result lost exact declared patches.")


def qualify_gmsh() -> dict[str, object]:
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

        replayed_model = phx.geometry.import_brep(
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
            raise RuntimeError("Native BRep replay changed semantic Gmsh identity.")

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
            raise RuntimeError("Scoped Gmsh sizing lacks per-control evidence.")

        phx.geometry.persist_occt_shape(
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
            raise RuntimeError("Semantic Gmsh plan accepted replaced BRep bytes.")

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


def _layout_bytes() -> bytes:
    # ty: ignore[unresolved-import]
    import gdstk

    with TemporaryDirectory(prefix="phydrax-layout-fixture-") as temporary:
        path = Path(temporary) / "qualification.gds"
        library = gdstk.Library(unit=1.0e-3, precision=1.0e-9)
        top = library.new_cell("TOP")
        top.add(gdstk.rectangle((0.0, 0.0), (2.0, 1.0), layer=1, datatype=0))
        library.write_gds(path)
        return path.read_bytes()


def _decode_layout() -> Any:
    contract = _contract()
    limits = phx.interchange.ResourceLimits(
        max_bytes=1_000_000,
        max_depth=4,
        max_nodes=1_000,
        max_attributes=1_000,
        max_losses=0,
    )
    policy = phx.interchange.LayoutImportPolicy(
        phx.interchange.LayoutFormat.GDSII,
        contract,
        limits,
        top_cell="TOP",
    )
    return phx.interchange.decode_layout_bytes(
        _layout_bytes(), policy, source_path="qualification.gds"
    )


def _hybrid_face_indices(source: Any, axis: int, coordinate: float) -> Any:
    points = np.asarray(source.mesh_vertices)
    triangles = points[np.asarray(source.mesh_faces)]
    face_ids = np.asarray(source.triangle_face_ids)
    return tuple(
        np.unique(
            face_ids[
                np.all(
                    np.isclose(triangles[:, :, axis], coordinate),
                    axis=1,
                )
            ]
        )
    )


def _hybrid_gmsh_result(destination: Path, *, schedule: Any = None) -> Any:
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    lower = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.0), 1.0, 1.0, 0.3).Shape()
    upper = BRepPrimAPI_MakeBox(gp_Pnt(0.0, 0.0, 0.3), 1.0, 1.0, 0.7).Shape()
    splitter = BOPAlgo_Splitter()
    splitter.AddArgument(lower)
    splitter.AddArgument(upper)
    splitter.Perform()
    if splitter.HasErrors():
        raise RuntimeError("OCCT failed to build the hybrid qualification fixture.")
    source = phx.geometry.persist_occt_shape(
        splitter.Shape(),
        destination,
        coordinate_contract=_contract(),
        linear_deflection=0.03,
        angular_deflection=0.15,
    )
    source_face = _hybrid_face_indices(source, 2, 0.0)
    target_face = _hybrid_face_indices(source, 2, 0.3)
    if len(source_face) != 1 or len(target_face) != 1:
        raise RuntimeError("Hybrid qualification could not resolve its exact caps.")
    swept = source.topology.face_solids[source_face[0]][0]
    core = next(index for index in range(source.topology.num_solids) if index != swept)
    provider = phx.meshing.GmshProvider()
    swept_scope = provider.entity_scope(source, source.solid_ids[swept])
    core_scope = provider.entity_scope(source, source.solid_ids[core])
    layer_schedule = (
        # ty: ignore[invalid-argument-type]
        phx.meshing.LayerSchedule((0.08, 0.10, 0.12)) if schedule is None else schedule
    )
    if not isinstance(layer_schedule, phx.meshing.LayerSchedule):
        raise TypeError("schedule must be LayerSchedule or None.")
    control = phx.meshing.BoundaryLayerControl(
        provider.entity_scope(source, source.face_ids[source_face[0]]),
        layer_schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=swept_scope,
        cap_scope=provider.entity_scope(source, source.face_ids[target_face[0]]),
    )
    whole = provider.whole_scope(source, 3)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allow_mixed=True,
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SWEEP,
        size_controls=(phx.meshing.UniformSizeControl(whole, 0.24),),
        region_controls=(
            phx.meshing.RegionControl(
                swept_scope,
                "swept-slab",
                "swept-material",
                phx.meshing.RegionRole.SOLID,
            ),
            phx.meshing.RegionControl(
                core_scope,
                "tetra-core",
                "core-material",
                phx.meshing.RegionRole.SOLID,
            ),
        ),
        patch_controls=(
            phx.meshing.PatchControl(
                "slab-core",
                provider.entity_scope(source, source.face_ids[target_face[0]]),
                ("swept-slab", "tetra-core"),
            ),
        ),
        layer_controls=(control,),
    )
    return provider.plan(source, specification).execute(), control


def _mesh_scope(
    mesh: phx.discretization.CellMesh, dimension: int, ids: np.ndarray
) -> Any:
    entities = mesh.entity_set(dimension)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        dimension,
        entities.entity_set_id,
        ids,
    )


def qualify_manifold() -> dict[str, object]:
    import manifold3d

    def cube(offset: Any) -> Any:
        arrays = manifold3d.Manifold.cube().translate(offset).to_mesh64()
        return phx.geometry.SurfaceModel.from_triangles(
            arrays.vert_properties[:, :3],
            arrays.tri_verts,
            phx.geometry.SurfaceMetadata(
                source_id=str(offset),
                source_revision="0",
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                provenance=("qualification",),
            ),
        )

    result = phx.meshing.ManifoldProvider().execute(
        (cube((0.0, 0.0, 0.0)), cube((0.5, 0.0, 0.0))),
        phx.meshing.SurfaceBooleanOperation.DIFFERENCE,
    )
    points = np.asarray(result.mesh.coordinates)[
        np.asarray(result.mesh.blocks[0].vertices)
    ]
    volume = float(
        np.sum(np.sum(points[:, 0] * np.cross(points[:, 1], points[:, 2]), axis=1)) / 6.0
    )
    if not np.isclose(volume, 0.5, atol=1e-12) or not result.audit.passed:
        raise RuntimeError("Manifold difference failed solid-volume qualification.")
    covered = np.concatenate(
        [np.asarray(item.target_global_ids) for item in result.associations]
    )
    if not np.array_equal(
        np.sort(covered), np.sort(np.asarray(result.mesh.entity_set(2).entity_ids))
    ):
        raise RuntimeError("Manifold result lacks complete source-face ancestry.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "volume": volume,
        "source_associations": len(result.associations),
        "audit_passed": result.audit.passed,
    }


def qualify_volume(
    provider_name: str, executable: str, worker: str | None
) -> dict[str, object]:
    contract = phx.SpatialCoordinateContract.si()
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    faces = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    expected_volume = 1.0 / 6.0
    surface = phx.geometry.SurfaceModel.from_triangles(
        points,
        faces,
        phx.geometry.SurfaceMetadata(
            source_id="qualification-tetrahedron",
            source_revision="0",
            coordinate_contract=contract,
            provenance=("qualification",),
        ),
    )
    if provider_name == "mmg":
        source = phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh.from_tetrahedra(
                points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
            contract,
        )
        with phx.meshing.MmgProvider(executable=worker) as provider:
            metric = phx.meshing.MeshMetricField(
                provider.vertex_scope(source.mesh),
                np.tile(np.eye(3) * 16.0, (4, 1, 1)),
                minimum_size=0.25,
                maximum_size=0.25,
            )
            result = provider.adapt(source, metric=metric).mesh
    elif provider_name == "ftetwild":
        provider = phx.meshing.FTetWildProvider(
            phx.meshing.FTetWildOptions(envelope_distance=0.005, maximum_iterations=20)
        )
        scope = provider.whole_scope(surface)
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            deterministic=False,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope, 0.3, strength=phx.meshing.SizeControlStrength.SOFT
                ),
            ),
        )
        result = provider.plan(surface, specification).execute()
    elif provider_name == "vorocrust":
        with phx.meshing.VoroCrustProvider(executable, worker) as provider:
            result = provider.execute(surface, phx.meshing.VoroCrustOptions(1.0))
    else:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        with TemporaryDirectory(prefix="phydrax-gmsh-qualification-") as temporary:
            path = Path(temporary) / "cube.brep"
            coordinate_contract = _contract()
            phx.geometry.persist_occt_shape(
                BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(),
                path,
                coordinate_contract=coordinate_contract,
            )
            source = phx.geometry.BRepSource(
                phx.geometry.import_brep(
                    path,
                    coordinate_contract=coordinate_contract,
                    linear_deflection=0.05,
                    angular_deflection=0.2,
                )
            )
            provider = phx.meshing.GmshProvider()
            scope = provider.whole_scope(source, 3)
            specification = phx.meshing.VolumeMeshingSpec(
                phx.meshing.CellMeshingTarget(
                    3,
                    3,
                    phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
                    geometry_order=2,
                ),
                scope,
                phx.meshing.VolumeFillStrategy.SIMPLEX,
                size_controls=(
                    phx.meshing.UniformSizeControl(scope, 0.3, maximum_growth_rate=2.0),
                ),
            )
            result = provider.plan(source, specification).execute()
            expected_volume = 1.0
    volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    if not np.isclose(volume, expected_volume, rtol=0.03) or not result.audit.passed:
        raise RuntimeError(f"{provider_name} failed enclosed-volume qualification.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "cells": result.mesh.entity_set(3).count,
        "volume": volume,
        "audit_passed": result.audit.passed,
        "quality_scope": result.audit.quality_scope,
    }


def qualify_poisson() -> dict[str, object]:
    count = 400
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    angles = np.arange(count) * np.pi * (3.0 - np.sqrt(5.0))
    normals = np.column_stack(
        (np.sqrt(1.0 - z * z) * np.cos(angles), np.sqrt(1.0 - z * z) * np.sin(angles), z)
    )
    source = phx.meshing.OrientedPointCloud(
        normals,
        normals,
        phx.SpatialCoordinateContract.si(),
        source_id="sphere",
        source_revision="0",
    )
    result = phx.meshing.PoissonProvider().execute(
        source, phx.meshing.PoissonReconstructionSpec(depth=5)
    )
    error = float(
        np.max(np.abs(np.linalg.norm(np.asarray(result.mesh.coordinates), axis=1) - 1.0))
    )
    if error > 0.1 or not result.audit.passed:
        raise RuntimeError("Poisson reconstruction failed sphere qualification.")
    return {
        "provider": result.provider.name,
        "version": result.runtime.actual_version,
        "maximum_radius_error": error,
        "audit_passed": result.audit.passed,
    }


def qualify_cad_partition() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-cad-partition-") as temporary:
        result = _three_dimensional_partition(Path(temporary))
    if result.topological_dimension != 3 or not result.named_solid_entity_ids:
        raise RuntimeError("CAD partition did not produce exact solid regions.")
    if not result.association_graph.graph_id or not result.revision.revision_id:
        raise RuntimeError("CAD partition omitted revision association evidence.")
    return {
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solid_regions": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "face_patches": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


def qualify_composite_heat_3d() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-composite-heat-3d-") as temporary:
        partition = _three_dimensional_partition(Path(temporary))
        provider = phx.meshing.GmshProvider()
        boundary_scope = provider.whole_scope(partition.model, 3)
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            boundary_scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(phx.meshing.UniformSizeControl(boundary_scope, 0.2),),
            region_controls=_region_controls(provider, partition),
            patch_controls=_patch_controls(provider, partition),
        )
        result = provider.plan(partition.model, specification).execute()
    if not result.audit.passed:
        raise RuntimeError("Composite three-dimensional heat mesh failed audit.")
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature",
        phx.discretization.lagrange_element("tetrahedron", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(
        result.mesh,
        field,
    ).prepare()
    domains = {
        name: discretization.integration_domain(
            "cell",
            phx.meshing.resolve_mesh_scope(result.mesh, zone.scope),
        )
        for name, zone in zones.items()
    }
    exterior_patches = tuple(
        patch for patch in result.patches if len(patch.adjacent_zone_ids) == 1
    )
    exterior_scope = exterior_patches[0].scope
    for patch in exterior_patches[1:]:
        exterior_scope = exterior_scope | patch.scope
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "temperature",
        boundary_selection=phx.meshing.resolve_mesh_scope(
            result.mesh,
            exterior_scope,
        ),
    )
    form = phx.equations.FiniteElementForm(
        "two-region-volume-diffusion",
        "temperature",
        (
            phx.equations.DiffusionAction(
                "temperature",
                1.0,
                action_id="left-volume-conductivity",
                domain=domains["left"],
            ),
            phx.equations.DiffusionAction(
                "temperature",
                2.0,
                action_id="right-volume-conductivity",
                domain=domains["right"],
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: jnp.where(
            points[..., 0] <= 1.0,
            points[..., 0],
            1.0 + 0.5 * (points[..., 0] - 1.0),
        ),
    )
    system, right_hand_side = compiled.linear_system()
    solved = phx.linalg.solve(system, right_hand_side)
    solution = np.asarray(compiled.expand(solved.value))
    coordinates = np.asarray(result.mesh.coordinates)
    expected = np.where(
        coordinates[:, 0] <= 1.0,
        coordinates[:, 0],
        1.0 + 0.5 * (coordinates[:, 0] - 1.0),
    )
    error = float(np.max(np.abs(solution - expected)))
    if not bool(np.all(np.asarray(solved.successful))) or error > 1.0e-6:
        raise RuntimeError(
            "Composite volume diffusion failed its exact interface solution."
        )
    return {
        "mesh_id": result.mesh.mesh_id,
        "cells": result.mesh.entity_set(3).count,
        "minimum_quality": float(result.quality.minimum_mean_ratio),
        "revision": partition.revision.revision_id,
        "association_graph": partition.association_graph.graph_id,
        "region_control_count": len(specification.region_controls),
        "patch_control_count": len(specification.patch_controls),
        "dofs": discretization.dof_maps[0].global_dof_count,
        "maximum_solution_error": error,
    }


def qualify_planar_semantic() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-planar-semantic-") as temporary:
        embedding, result = _planar_partition(Path(temporary) / "planar.brep")
    if (
        result.topological_dimension != 2
        or result.model.topology.num_solids != 0
        or not result.named_region_entity_ids
        or not result.named_patch_entity_ids
    ):
        raise RuntimeError("Planar partition did not preserve two-dimensional semantics.")
    return {
        "embedding": embedding.embedding_id,
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "region_entity_kind": result.region_entity_kind,
        "patch_entity_kind": result.patch_entity_kind,
        "regions": {name: len(ids) for name, ids in result.named_region_entity_ids},
        "patches": {name: len(ids) for name, ids in result.named_patch_entity_ids},
    }


def qualify_composite_heat_2d() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-composite-heat-2d-") as temporary:
        embedding, partition = _planar_partition(Path(temporary) / "planar.brep")
        provider = phx.meshing.GmshProvider()
        scope = provider.whole_scope(partition.model, 2)
        specification = phx.meshing.SurfaceMeshingSpec(
            phx.meshing.CellMeshingTarget(
                2,
                2,
                phx.meshing.CellFamilyPolicy(required=("triangle",)),
            ),
            scope,
            planar_embedding=embedding,
            size_controls=(phx.meshing.UniformSizeControl(scope, 0.1),),
            region_controls=_region_controls(provider, partition),
            patch_controls=_patch_controls(provider, partition),
        )
        result = provider.plan(partition.model, specification).execute()
    if not result.audit.passed or result.boundary is not None:
        raise RuntimeError("Composite two-dimensional heat mesh failed planar audit.")
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    if set(zones) != {"left", "right"}:
        raise RuntimeError("Planar meshing lost exact material-region zones.")
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature",
        phx.discretization.lagrange_element("triangle", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(
        result.mesh,
        field,
    ).prepare()
    domains = {
        name: discretization.integration_domain(
            "cell",
            phx.meshing.resolve_mesh_scope(result.mesh, zone.scope),
        )
        for name, zone in zones.items()
    }
    exterior_patches = tuple(
        patch for patch in result.patches if len(patch.adjacent_zone_ids) == 1
    )
    exterior_scope = exterior_patches[0].scope
    for patch in exterior_patches[1:]:
        exterior_scope = exterior_scope | patch.scope
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "temperature",
        boundary_selection=phx.meshing.resolve_mesh_scope(
            result.mesh,
            exterior_scope,
        ),
    )
    form = phx.equations.FiniteElementForm(
        "two-region-planar-diffusion",
        "temperature",
        (
            phx.equations.DiffusionAction(
                "temperature",
                1.0,
                action_id="left-conductivity",
                domain=domains["left"],
            ),
            phx.equations.DiffusionAction(
                "temperature",
                2.0,
                action_id="right-conductivity",
                domain=domains["right"],
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: jnp.where(
            points[..., 0] <= 1.0,
            points[..., 0],
            1.0 + 0.5 * (points[..., 0] - 1.0),
        ),
    )
    system, right_hand_side = compiled.linear_system()
    solved = phx.linalg.solve(system, right_hand_side)
    solution = np.asarray(compiled.expand(solved.value))
    coordinates = np.asarray(result.mesh.coordinates)
    expected = np.where(
        coordinates[:, 0] <= 1.0,
        coordinates[:, 0],
        1.0 + 0.5 * (coordinates[:, 0] - 1.0),
    )
    error = float(np.max(np.abs(solution - expected)))
    if not bool(np.all(np.asarray(solved.successful))) or error > 1.0e-6:
        raise RuntimeError(
            "Composite planar diffusion failed its exact interface solution."
        )
    return {
        "mesh_id": result.mesh.mesh_id,
        "cells": result.mesh.entity_set(2).count,
        "minimum_quality": float(result.quality.minimum_mean_ratio),
        "revision": partition.revision.revision_id,
        "association_graph": partition.association_graph.graph_id,
        "dofs": discretization.dof_maps[0].global_dof_count,
        "maximum_solution_error": error,
    }


def qualify_layout_observable() -> dict[str, object]:
    result = _decode_layout()
    region = result.model.regions[0]
    expected_layer = phx.interchange.LayoutLayerKey(1, 0)
    if (
        not result.report.valid
        or region.layer != expected_layer
        or result.source_digest != result.resource_manifest.content_sha256
        or result.model.coordinate_contract != _contract()
    ):
        raise RuntimeError(
            "Layout decoder omitted observable identity or layer evidence."
        )
    return {
        "layout_model_id": result.model.model_id,
        "source_digest": result.source_digest,
        "layer": [region.layer.layer, region.layer.datatype],
        "occurrence_path": list(region.occurrence_path),
        "provenance_id": region.provenance_id,
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
    }


def qualify_layout_stack() -> dict[str, object]:
    decoded = _decode_layout()
    with TemporaryDirectory(prefix="phydrax-layout-stack-") as temporary:
        result = phx.geometry.lower_process_stack(
            phx.geometry.ProcessStack(
                decoded.model.coordinate_contract,
                (
                    phx.geometry.StackRegion(
                        "metal",
                        decoded.model.regions[0].geometry,
                        phx.geometry.ZInterval(0.0, 0.8),
                        20,
                    ),
                ),
            ),
            Path(temporary) / "stack.brep",
            operand_directory=Path(temporary) / "operands",
        )
    if not result.named_solid_entity_ids or not result.association_graph.graph_id:
        raise RuntimeError("Process stack omitted persisted CAD identity evidence.")
    return {
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solids": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "faces": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


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
        raise RuntimeError("Planar band mesh failed audit or compliance.")
    achieved = dict(mesh_result.compliance.achieved)
    front_residuals = {}
    for layer in source.layer_partitions:
        key = f"planar_band:{layer.control_id}:{layer.region_name}:front:{layer.layer_index + 1}"
        residual = achieved[f"{key}:maximum_residual"]
        if residual > 1.0e-12:
            raise RuntimeError("Planar band front departed from its exact CAD level.")
        front_residuals[key] = residual
    return {
        "plan_id": source.plan_id,
        "result_id": source.result_id,
        "mesh_result_id": mesh_result.result_id,
        "partition_revision": source.partition.revision.revision_id,
        "layer_partitions": len(source.layer_partitions),
        "front_residuals": front_residuals,
    }


def qualify_hybrid_layer() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-hybrid-layer-") as temporary:
        result, control = _hybrid_gmsh_result(Path(temporary) / "hybrid-slab.brep")
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError("Generated hybrid layer mesh failed audit or compliance.")
    block_kinds = tuple(block.cell_kind for block in result.mesh.blocks)
    if block_kinds != ("prism", "tetrahedron"):
        raise RuntimeError("Generated hybrid layer mesh lost its required cell families.")
    achieved = dict(result.compliance.achieved)
    return {
        "mesh_id": result.mesh.mesh_id,
        "result_id": result.result_id,
        "control_id": control.control_id,
        "schedule_id": control.schedule.schedule_id,
        "thicknesses": list(control.schedule.thicknesses),
        "block_kinds": block_kinds,
        "cells": result.mesh.entity_set(3).count,
        "maximum_layer_thickness_error": achieved[
            f"layer:{control.control_id}:maximum_thickness_residual"
        ],
        "audit_passed": result.audit.passed,
    }


def qualify_hybrid_diffusion() -> dict[str, object]:
    with TemporaryDirectory(prefix="phydrax-hybrid-diffusion-") as temporary:
        generated, _ = _hybrid_gmsh_result(Path(temporary) / "hybrid-slab.brep")
    mesh = generated.mesh
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(
                block.cell_kind,
                1,
            )
            for block in mesh.blocks
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    cells = mesh.entity_set(3)
    scope = _mesh_scope(mesh, 3, np.asarray(cells.entity_ids))
    domain = discretization.integration_domain(
        "cell", phx.meshing.resolve_mesh_scope(mesh, scope)
    )
    form = phx.equations.FiniteElementForm(
        "hybrid-diffusion",
        "u",
        (phx.equations.DiffusionAction("u", 1.0, domain=domain),),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    system, rhs = compiled.linear_system()
    dof_count = discretization.dof_maps[0].global_dof_count
    constant = np.ones(dof_count)
    null_residual = float(np.max(np.abs(np.asarray(system.operator.mv(constant)))))
    coordinates = np.asarray(discretization.vertices)
    values = coordinates[:, 0]
    energy = float(np.vdot(values, np.asarray(system.operator.mv(values))))
    rhs_norm = float(np.max(np.abs(np.asarray(rhs))))
    if not (
        np.isclose(null_residual, 0.0, atol=1.0e-12)
        and np.isclose(rhs_norm, 0.0, atol=1.0e-12)
        and np.isclose(energy, 1.0, atol=1.0e-10)
    ):
        raise RuntimeError("Hybrid diffusion lost its P1 null mode or volume energy.")
    return {
        "mesh_id": mesh.mesh_id,
        "scope": scope.scope_id,
        "dofs": dof_count,
        "null_residual": null_residual,
        "rhs_norm": rhs_norm,
        "x_gradient_energy": energy,
        "generated_result_id": generated.result_id,
    }


def _distribution_lattice(cells_per_axis: int, /) -> phx.discretization.CellMesh:
    """Counterclockwise lattice triangles of the unit square."""
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((cells_per_axis + 1) ** 2, dtype=np.int32).reshape(
        (cells_per_axis + 1, cells_per_axis + 1)
    )
    lower, right = index[:-1, :-1].reshape((-1,)), index[1:, :-1].reshape((-1,))
    upper, left = index[1:, 1:].reshape((-1,)), index[:-1, 1:].reshape((-1,))
    triangles = np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )
    return phx.discretization.CellMesh.from_triangles(points, triangles)


def _distribution_cells(
    mesh: phx.discretization.CellMesh, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native-order global IDs, signed areas, and centroids of triangle cells."""
    triangles = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = np.asarray(mesh.coordinates)[triangles]
    edges = corners[:, 1:] - corners[:, :1]
    areas = 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    return ids, areas, corners.mean(axis=1)


def _distribution_edge_cut(
    mesh: phx.discretization.CellMesh, owner: np.ndarray, /
) -> int:
    """Edge-sharing cell pairs with different owners, recounted from the triangles."""
    triangles = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int32)
    edges = np.sort(triangles[:, corners], axis=2).reshape((-1, 2)).astype(np.int64)
    keys = edges[:, 0] * mesh.coordinates.shape[0] + edges[:, 1]
    order = np.argsort(keys, kind="stable")
    cells = np.repeat(np.arange(triangles.shape[0]), 3)[order]
    shared = keys[order][1:] == keys[order][:-1]
    return int(np.count_nonzero(owner[cells[:-1][shared]] != owner[cells[1:][shared]]))


def _distribution_rows(native: np.ndarray, identifiers: np.ndarray, /) -> np.ndarray:
    """Rows of ``identifiers`` in ``native``; every identifier must be present."""
    order = np.argsort(native, kind="stable")
    located = np.searchsorted(native, identifiers, sorter=order)
    rows = order[np.minimum(located, native.size - 1)]
    if not np.array_equal(native[rows], identifiers):
        raise RuntimeError("Lineage identifiers are missing from the native cell order.")
    return rows


def _distribution_retained_bytes(tree: object, /) -> int:
    """Bytes of the distinct array leaves a prepared value retains."""
    leaves = {
        id(leaf): leaf
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, (jax.Array, np.ndarray))
    }
    return sum(leaf.nbytes for leaf in leaves.values())


def _distribution_route(
    part: phx.meshing.MeshPart,
    policy: phx.meshing.MeshPartitionPolicy,
    weights: np.ndarray,
    bound: float,
    /,
    **ownership: np.ndarray,
) -> tuple[phx.meshing.MeshDistribution, dict[str, object]]:
    """Partition ``part`` and check its evidence against an independent recount."""
    distribution, stage = _measured(
        lambda: phx.meshing.prepare_mesh_distribution(
            part, policy=policy, cell_weights=weights, **ownership
        )
    )
    # ty: ignore[unresolved-attribute]
    owner = np.asarray(distribution.partition.cell_owner)
    # ty: ignore[unresolved-attribute]
    evidence = distribution.evidence
    part_weights = np.bincount(owner, weights=weights, minlength=policy.part_count)
    # ty: ignore[unresolved-attribute]
    halos = tuple(np.asarray(rows) for rows in distribution.halo_rows)
    # ty: ignore[unresolved-attribute]
    edge_cut = _distribution_edge_cut(part.carrier.mesh, owner)
    ghost_owned = any(np.any(owner[rows] == rank) for rank, rows in enumerate(halos))
    if (
        np.any(part_weights <= 0.0)
        or not np.allclose(np.asarray(evidence.part_weights), part_weights, rtol=1e-14)
        or float(evidence.imbalance) > bound + 1e-12
        or int(evidence.edge_cut) != edge_cut
        or int(evidence.halo_replicas) != sum(rows.size for rows in halos)
        or ghost_owned
    ):
        raise RuntimeError(
            f"{policy.kind.value} distribution evidence failed its recount."
        )
    # ty: ignore[invalid-return-type]
    return distribution, {
        "status": "passed",
        # ty: ignore[unresolved-attribute]
        "distribution_id": distribution.distribution_id,
        "evidence_id": evidence.evidence_id,
        "provenance": evidence.provenance,
        "part_weights": part_weights.tolist(),
        "imbalance": float(evidence.imbalance),
        "imbalance_bound": float(bound),
        "edge_cut": edge_cut,
        "halo_replicas": int(evidence.halo_replicas),
        "resources": {
            **stage,
            "retained_bytes": _distribution_retained_bytes(distribution),
        },
    }


def _distribution_transition(
    source: phx.meshing.MeshDistribution,
    target_part: phx.meshing.MeshPart,
    lineage: phx.meshing.MeshLineage,
    policy: phx.meshing.MeshPartitionPolicy,
    weights: np.ndarray,
    source_areas: np.ndarray,
    density: np.ndarray,
    /,
) -> dict[str, object]:
    """Carry ``source`` through ``lineage``; check migration and cell-data transfer."""
    transition, stage = _measured(
        lambda: phx.meshing.prepare_distribution_transition(
            source, target_part, lineage, policy=policy, cell_weights=weights
        )
    )
    # ty: ignore[unresolved-attribute]
    target = transition.target
    # ty: ignore[unresolved-attribute]
    mesh = target_part.carrier.mesh
    target_ids, target_areas, _ = _distribution_cells(mesh)
    source_ids = np.asarray(source.cell_global_ids)
    cells = next(record for record in lineage.entities if record.dimension == 2)
    lineage_sources = np.asarray(cells.source_global_ids)
    lineage_targets = np.asarray(cells.target_global_ids)
    parents = np.full(target_ids.shape, -1, dtype=np.int64)
    parents[_distribution_rows(target_ids, lineage_targets)] = _distribution_rows(
        source_ids, lineage_sources
    )
    # ty: ignore[unresolved-attribute]
    if np.any(parents < 0) or not np.all(np.asarray(transition.target_defined)):
        raise RuntimeError("Bisection lineage left a target cell without a parent.")
    # ty: ignore[unresolved-attribute]
    moved = np.asarray(transition.transfer(jnp.asarray(density)))
    copy_error = float(np.max(np.abs(moved - density[parents])))
    integral_before = float(np.sum(density * source_areas))
    integral_after = float(np.sum(moved * target_areas))
    residual = abs(integral_after - integral_before) / abs(integral_before)
    parts = policy.part_count
    sender = np.asarray(source.partition.cell_owner, dtype=np.int64)[parents]
    receiver = np.asarray(target.partition.cell_owner, dtype=np.int64)
    remote = sender != receiver
    counts = np.bincount(sender * parts + receiver, minlength=parts * parts)
    # ty: ignore[unresolved-attribute]
    slots = np.arange(transition.send.capacity)
    # ty: ignore[unresolved-attribute]
    sent = np.concatenate([np.asarray(transition.sent_by(rank)) for rank in range(parts)])
    received = np.concatenate(
        # ty: ignore[unresolved-attribute]
        [np.asarray(transition.received_by(rank)) for rank in range(parts)]
    )
    preserved = np.asarray(cells.relation_kinds) == int(
        phx.meshing.EntityLineageKind.PRESERVED
    )
    bound = (
        1.0 + parts * np.max(weights) / np.sum(weights)
        # ty: ignore[unresolved-attribute]
        if transition.rebalanced
        else policy.maximum_imbalance
    )
    checks = {
        "exact_parent_copy": copy_error == 0.0,
        "integral_preserved": residual <= 1e-12,
        "migration_counts": np.array_equal(
            # ty: ignore[unresolved-attribute]
            np.asarray(transition.migration_counts),
            counts.reshape((parts, parts)),
        ),
        # ty: ignore[unresolved-attribute]
        "migrated_cells": int(transition.migrated_cells) == np.count_nonzero(remote),
        "migration_volume": np.isclose(
            # ty: ignore[unresolved-attribute]
            float(transition.migration_volume),
            np.sum(weights[remote]),
            rtol=1e-14,
            atol=0.0,
        ),
        "message_slots": np.array_equal(np.sort(sent), slots)
        and np.array_equal(np.sort(received), slots),
        "imbalance_bound": float(target.evidence.imbalance) <= bound + 1e-12,
        "edge_cut": int(target.evidence.edge_cut)
        == _distribution_edge_cut(mesh, receiver),
        "source_ids_unchanged": np.array_equal(
            # ty: ignore[unresolved-attribute]
            np.asarray(transition.source.cell_global_ids),
            source_ids,
        ),
        "target_ids_native": np.array_equal(
            np.asarray(target.cell_global_ids), target_ids
        ),
        "preserved_ids_unchanged": np.array_equal(
            lineage_sources[preserved], lineage_targets[preserved]
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Distribution transition failed checks: {failed}.")
    return {
        "status": "passed",
        # ty: ignore[unresolved-attribute]
        "transition_id": transition.transition_id,
        # ty: ignore[unresolved-attribute]
        "lineage_id": transition.lineage_id,
        "source_distribution_id": source.distribution_id,
        "target_distribution_id": target.distribution_id,
        "target_provenance": target.evidence.provenance,
        "maximum_imbalance": policy.maximum_imbalance,
        # ty: ignore[unresolved-attribute]
        "rebalanced": transition.rebalanced,
        "imbalance_after": float(target.evidence.imbalance),
        "imbalance_bound": float(bound),
        "edge_cut_after": int(target.evidence.edge_cut),
        "halo_replicas_after": int(target.evidence.halo_replicas),
        "migration_counts": counts.reshape((parts, parts)).tolist(),
        # ty: ignore[unresolved-attribute]
        "migrated_cells": int(transition.migrated_cells),
        # ty: ignore[unresolved-attribute]
        "migration_volume": float(transition.migration_volume),
        "message_slots": int(slots.size),
        "preserved_cells": int(np.count_nonzero(preserved)),
        "checks_passed": sorted(checks),
        "transfer": {
            "maximum_parent_copy_error": copy_error,
            "integral_before": integral_before,
            "integral_after": integral_after,
            "relative_integral_residual": residual,
        },
        "resources": {
            **stage,
            "retained_bytes": _distribution_retained_bytes(transition),
        },
    }


def qualify_distribution() -> dict[str, object]:
    from phydrax.meshing._metis import metis_identity

    kind = phx.meshing.MeshPartitionKind
    parts = 4
    source, certify_stage = _measured(
        lambda: phx.meshing.certify_cell_mesh(_distribution_lattice(8), _contract())
    )
    # ty: ignore[invalid-argument-type]
    part = phx.meshing.MeshPart("distribution-square", source)
    # ty: ignore[unresolved-attribute]
    cell_ids, areas, centroids = _distribution_cells(source.mesh)
    # The right half costs twice as much, so balanced parts differ in cell count.
    weights = np.where(centroids[:, 0] > 0.5, 2.0, 1.0)
    curve_bound = 1.0 + parts * np.max(weights) / np.sum(weights)
    distributions, routes = {}, {}
    for route in (kind.MORTON, kind.HILBERT):
        distributions[route], routes[route.value] = _distribution_route(
            part, phx.meshing.MeshPartitionPolicy(route, parts), weights, curve_bound
        )
    # Loading the shared library is the only way to learn whether METIS is usable.
    try:
        metis = metis_identity()
    except phx.meshing.MetisUnavailableError as error:
        routes[kind.GRAPH.value] = {
            "status": "missing-dependency",
            "dependency": "METIS",
            "detail": f"{error} Install METIS 5 and set PHYDRAX_METIS_LIBRARY.",
        }
    else:
        graph = phx.meshing.MeshPartitionPolicy(kind.GRAPH, parts, maximum_imbalance=1.1)
        _, record = _distribution_route(part, graph, weights, graph.maximum_imbalance)
        routes[kind.GRAPH.value] = {**record, "metis": metis}
    # Horizontal strips stand in for ownership computed by an external partitioner.
    strips = np.minimum(np.floor(centroids[:, 1] * parts), parts - 1).astype(np.int32)
    strip_weights = np.bincount(strips, weights=weights, minlength=parts)
    provided, routes[kind.PROVIDER.value] = _distribution_route(
        part,
        phx.meshing.MeshPartitionPolicy(kind.PROVIDER, parts),
        weights,
        np.max(strip_weights) / np.mean(strip_weights),
        ownership=strips,
    )
    if not np.array_equal(np.asarray(provided.partition.cell_owner), strips):
        raise RuntimeError("The PROVIDER route did not keep the supplied ownership.")

    marked = cell_ids[np.all(centroids < 0.3, axis=1)]
    adaptation, adaptation_stage = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                # ty: ignore[invalid-argument-type]
                source,
                phx.meshing.MarkedMeshAdaptation(marked),
                policy=phx.meshing.MeshAdaptationPolicy(
                    phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
                ),
            )
        )
    )
    # ty: ignore[unresolved-attribute]
    target = adaptation.target
    # ty: ignore[unresolved-attribute]
    if adaptation.lineage is None or not target.audit.passed:
        raise RuntimeError("Native bisection did not refine the marked subregion.")
    target_part = phx.meshing.MeshPart(part.name, target)
    _, _, target_centroids = _distribution_cells(target.mesh)
    target_weights = np.where(target_centroids[:, 0] > 0.5, 2.0, 1.0)
    density = 1.0 + centroids[:, 0] + 2.0 * centroids[:, 1] ** 2
    hilbert = distributions[kind.HILBERT]
    transitions = {
        name: _distribution_transition(
            hilbert,
            target_part,
            # ty: ignore[unresolved-attribute]
            adaptation.lineage,
            phx.meshing.MeshPartitionPolicy(
                kind.HILBERT, parts, maximum_imbalance=tolerance
            ),
            target_weights,
            areas,
            density,
        )
        for name, tolerance in (("inherited", 2.0), ("rebalanced", 1.05))
    }
    inherited, rebalanced = transitions["inherited"], transitions["rebalanced"]
    if (
        inherited["rebalanced"]
        or inherited["migrated_cells"] != 0
        or not rebalanced["rebalanced"]
        or rebalanced["migrated_cells"] == 0
    ):
        raise RuntimeError("Transition ownership ignored the maximum-imbalance rule.")
    return {
        "status": "passed",
        "source": {
            # ty: ignore[invalid-argument-type]
            **_result_identity(source),
            "part_id": part.part_id,
            "distribution_id": hilbert.distribution_id,
        },
        "target": {
            **_result_identity(target),
            "part_id": target_part.part_id,
            # ty: ignore[unresolved-attribute]
            "adaptation_result_id": adaptation.result_id,
            # ty: ignore[unresolved-attribute]
            "lineage_id": adaptation.lineage.lineage_id,
        },
        "runtime": _runtime_identity(),
        "quality": {
            # ty: ignore[unresolved-attribute]
            "before": _quality_summary(source.quality),
            "after": _quality_summary(target.quality),
        },
        "routes": routes,
        "adaptation": {
            # ty: ignore[unresolved-attribute]
            "route": adaptation.route.value,
            # ty: ignore[unresolved-attribute]
            "status": adaptation.status.value,
            "marked_cells": int(marked.size),
        },
        "transitions": transitions,
        "transfer": {name: value["transfer"] for name, value in transitions.items()},
        "conservation": {
            # ty: ignore[not-subscriptable]
            name: value["transfer"]["relative_integral_residual"]
            for name, value in transitions.items()
        },
        "resources": {
            "certification": certify_stage,
            "adaptation": adaptation_stage,
            "routes": {
                name: value["resources"]
                for name, value in routes.items()
                if "resources" in value
            },
            "transitions": {
                name: value["resources"] for name, value in transitions.items()
            },
        },
    }


def _forest_balance(topology: phx.discretization.ForestHierarchyTopology, /) -> Any:
    """Level histogram and the largest level jump across shared faces."""
    levels = topology.leaf_levels().astype(np.int64)
    workset = topology.workset
    valid = np.asarray(workset.face_valid)
    minus = levels[np.asarray(workset.face_minus)[valid]]
    plus = levels[np.asarray(workset.face_plus)[valid]]
    return {
        "topology_id": topology.topology_id,
        "epoch": topology.epoch.index,
        "leaves": topology.leaf_count,
        "faces": topology.face_count,
        "level_histogram": np.bincount(
            levels, minlength=topology.plan.maximum_level + 1
        ).tolist(),
        "maximum_face_level_jump": int(np.max(np.abs(minus - plus))),
        "hanging_faces": int(
            np.count_nonzero(np.asarray(workset.face_coarse_fine)[valid])
        ),
    }


def qualify_forest_amr() -> dict[str, object]:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    plan = phx.discretization.ForestPlan(
        grid,
        maximum_level=4,
        balance=phx.discretization.AMRBalanceStencil.FACE,
        maximum_leaf_capacity=1 << 12,
    )
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    initial, initialize_stage = _measured(lambda: compiler.initialize(1))
    # ty: ignore[unresolved-attribute]
    topology = initial.topology
    finest = np.asarray(plan.root_shape, dtype=np.int64) << plan.maximum_level
    # Drive one interior point to the maximum level; every step forces 2:1
    # closure refinements of coarser face neighbors.
    point = (finest * 3 // 8)[None, :]
    refinement_stages, closures = [], 0
    for _ in range(plan.maximum_level - 1):
        marks = np.zeros((topology.signature.leaf_capacity,), dtype=np.int8)
        marks[topology.locate_cells([plan.maximum_level], point)] = 1
        refined, stage = _measured(
            lambda topology=topology, marks=marks: compiler.adapt(topology, marks)
        )
        # ty: ignore[unresolved-attribute]
        if not refined.status.successful:
            # ty: ignore[unresolved-attribute]
            raise RuntimeError(f"Forest refinement failed: {refined.status.message}")
        # ty: ignore[unresolved-attribute]
        closures += refined.evidence.balance_refinements
        refinement_stages.append(stage)
        # ty: ignore[unresolved-attribute]
        topology = refined.topology
    source = topology
    # One mixed epoch: refine near the center, coarsen every family on the right.
    levels = source.leaf_levels().astype(np.int64)
    shift = plan.maximum_level - levels
    lower = source.leaf_coordinates().astype(np.int64) << shift[:, None]
    centers = (lower + (1 << shift)[:, None] / 2.0) / finest
    marks = np.zeros((source.signature.leaf_capacity,), dtype=np.int8)
    marks[: source.leaf_count] = np.where(centers[:, 0] > 0.75, -1, 0)
    marks[source.locate_cells([plan.maximum_level], (finest * 5 // 8)[None, :])] = 1
    adapted, adapt_stage = _measured(lambda: compiler.adapt(source, marks))
    # ty: ignore[unresolved-attribute]
    evidence = adapted.evidence
    # ty: ignore[unresolved-attribute]
    target = adapted.topology
    before, after = _forest_balance(source), _forest_balance(target)
    if (
        # ty: ignore[unresolved-attribute]
        not adapted.status.successful
        or evidence.coarsened_families == 0
        or evidence.requested_refinements == 0
        or closures == 0
        or max(before["maximum_face_level_jump"], after["maximum_face_level_jump"]) > 1
    ):
        raise RuntimeError("Forest adaptation did not refine, coarsen, and stay 2:1.")

    transition, transition_stage = _measured(
        lambda: phx.discretization.ForestFieldTransition(source, target)
    )
    # ty: ignore[unresolved-attribute]
    routes = transition.routes
    capacity = source.signature.leaf_capacity
    rng = np.random.default_rng(11)
    field = jnp.asarray(
        np.where(
            np.asarray(source.workset.leaf_valid),
            3.0 + rng.normal(size=(capacity,)),
            0.0,
        )
    )
    transferred, apply_stage = _measured(
        lambda: jax.block_until_ready(routes.apply(field))
    )
    content_before = float(jnp.sum(routes.source_measures * field))
    # ty: ignore[unresolved-attribute]
    content_after = float(jnp.sum(routes.target_measures * transferred.values))
    residual = abs(content_after - content_before) / abs(content_before)
    # ty: ignore[unresolved-attribute]
    reported = float(jnp.max(jnp.abs(transferred.conservation_residual)))
    constant = np.asarray(routes.apply(jnp.full((capacity,), 2.5)).values)
    constant_error = float(
        np.max(np.abs(constant[np.asarray(target.workset.leaf_valid)] - 2.5))
    )
    if (
        # ty: ignore[unresolved-attribute]
        not bool(transferred.successful)
        # ty: ignore[unresolved-attribute]
        or not transition.properties.conservative
        # ty: ignore[unresolved-attribute]
        or not transition.properties.constant_preserving
        or residual > 1e-13
        or reported > 1e-12 * abs(content_before)
        or constant_error > 1e-14
    ):
        raise RuntimeError("Forest transfer failed conservation or constants.")

    partition_plan = phx.discretization.ForestPartitionPlan(4)
    source_partition, source_partition_stage = _measured(
        lambda: partition_plan.prepare(source)
    )
    target_partition, target_partition_stage = _measured(
        lambda: partition_plan.prepare(target)
    )
    migration, migration_stage = _measured(
        # ty: ignore[unresolved-attribute]
        lambda: source_partition.migration_to(target_partition, transition=transition)
    )
    # ty: ignore[unresolved-attribute]
    migrated = target_partition.unpack(migration.migrate(source_partition.pack(field)))
    # ty: ignore[unresolved-attribute]
    migration_error = float(jnp.max(jnp.abs(migrated - transferred.values)))
    round_trip = bool(
        # ty: ignore[unresolved-attribute]
        jnp.array_equal(source_partition.unpack(source_partition.pack(field)), field)
    )
    partitions = {
        # ty: ignore[unresolved-attribute]
        "source": source_partition.evidence,
        # ty: ignore[unresolved-attribute]
        "target": target_partition.evidence,
    }
    if (
        migration_error > 1e-12
        or not round_trip
        or any(min(value.ghost_counts) == 0 for value in partitions.values())
    ):
        raise RuntimeError("Distributed forest migration disagreed with the transfer.")
    return {
        "status": "passed",
        "source": {
            "plan_id": plan.plan_id,
            "topology_id": source.topology_id,
            "epoch_id": source.epoch.epoch_id,
            "leaves": source.leaf_count,
            # ty: ignore[unresolved-attribute]
            "partition_id": source_partition.partition_id,
        },
        "target": {
            "topology_id": target.topology_id,
            "epoch_id": target.epoch.epoch_id,
            "leaves": target.leaf_count,
            # ty: ignore[unresolved-attribute]
            "adapt_result_id": adapted.result_id,
            "adapt_evidence_id": evidence.evidence_id,
            # ty: ignore[unresolved-attribute]
            "partition_id": target_partition.partition_id,
        },
        "runtime": _runtime_identity(),
        "quality": {"before": before, "after": after},
        "balance": {
            "stencil": plan.balance.value,
            "closure_refinements_while_refining": closures,
            "closure_refinements_mixed_epoch": evidence.balance_refinements,
            "balance_vetoed_families": evidence.balance_vetoed_families,
            "depth_limited_refinements": evidence.depth_limited_refinements,
        },
        "adaptation": {
            "requested_refinements": evidence.requested_refinements,
            "requested_coarsenings": evidence.requested_coarsenings,
            "coarsened_families": evidence.coarsened_families,
            # ty: ignore[unresolved-attribute]
            "retained_leaves": transition.retained_leaves,
            # ty: ignore[unresolved-attribute]
            "prolonged_leaves": transition.prolonged_leaves,
            # ty: ignore[unresolved-attribute]
            "restricted_leaves": transition.restricted_leaves,
        },
        "transfer": {
            # ty: ignore[unresolved-attribute]
            "transition_id": transition.transition_id,
            # ty: ignore[unresolved-attribute]
            "conservative": transition.properties.conservative,
            # ty: ignore[unresolved-attribute]
            "constant_preserving": transition.properties.constant_preserving,
            "constant_error": constant_error,
            "distributed_migration_error": migration_error,
        },
        "conservation": {
            "content_before": content_before,
            "content_after": content_after,
            "relative_residual": residual,
            "reported_residual": reported,
        },
        "distribution": {
            "parts": partition_plan.part_count,
            "ghost_stencil": partition_plan.ghost_stencil.value,
            # ty: ignore[unresolved-attribute]
            "migration_id": migration.migration_id,
            # ty: ignore[unresolved-attribute]
            "moved_leaves": migration.moved_leaves,
            **{
                name: {
                    "evidence_id": value.evidence_id,
                    "part_leaf_counts": list(value.part_leaf_counts),
                    "part_costs": list(value.part_costs),
                    "ghost_counts": list(value.ghost_counts),
                    "maximum_imbalance": value.maximum_imbalance,
                    "cut_faces": value.cut_faces,
                }
                for name, value in partitions.items()
            },
        },
        "resources": {
            "initialize": initialize_stage,
            "refinements": refinement_stages,
            "mixed_adaptation": adapt_stage,
            "transition": transition_stage,
            "transfer_apply": apply_stage,
            "source_partition": source_partition_stage,
            "target_partition": target_partition_stage,
            "migration": migration_stage,
            "retained_bytes": {
                "source_topology": _distribution_retained_bytes(source),
                "target_topology": _distribution_retained_bytes(target),
                "transition": _distribution_retained_bytes(transition),
                "source_partition": _distribution_retained_bytes(source_partition),
                "target_partition": _distribution_retained_bytes(target_partition),
                "migration": _distribution_retained_bytes(migration),
            },
        },
    }


_OMEGA_H_WORKER_HINT = (
    "Build native/providers/omega_h against an Omega_h CMake package (MPI-enabled "
    "for omega-h-distributed; docs/guides_meshing.md), then pass --worker, set "
    "PHYDRAX_OMEGA_H_WORKER, or put phydrax-omega-h-worker on PATH."
)


def _omega_h_worker(worker: str | None, /) -> str | None:
    """Resolve the Omega_h worker exactly as ``OmegaHProvider`` does."""
    return shutil.which(
        worker or os.environ.get("PHYDRAX_OMEGA_H_WORKER", "phydrax-omega-h-worker")
    )


def _qualify_omega_h_ranks(worker: str | None, ranks: int, /) -> dict[str, object]:
    """Metric adaptation with LINEAR and CONSERVE fields on ``ranks`` MPI ranks."""
    executable = _omega_h_worker(worker)
    if executable is None:
        return _missing_dependency("phydrax-omega-h-worker", _OMEGA_H_WORKER_HINT)
    if ranks > 1 and shutil.which("mpiexec") is None:
        return _missing_dependency(
            "mpiexec",
            "Distributed Omega_h adaptation launches its worker under the MPI "
            "launcher of the MPI implementation Omega_h was built with.",
        )
    cells_per_axis = 4
    mesh = _distribution_lattice(cells_per_axis)
    cell_ids, areas, centroids = _distribution_cells(mesh)
    zones = tuple(
        phx.meshing.MeshZone(
            name, phx.meshing.MeshZoneRole.REGION, _mesh_scope(mesh, 2, cell_ids[mask])
        )
        for name, mask in (
            ("left", centroids[:, 0] < 0.5),
            ("right", centroids[:, 0] > 0.5),
        )
    )
    source, certify_stage = _measured(
        lambda: phx.meshing.certify_cell_mesh(mesh, _contract(), zones=zones)
    )
    vertex_order = np.argsort(np.asarray(mesh.vertex_global_ids))
    vertex_ids = np.asarray(mesh.vertex_global_ids)[vertex_order]
    points = np.asarray(mesh.coordinates)[vertex_order]
    matrix = np.eye(2) * (2.0 * cells_per_axis) ** 2
    matrix[0, 1] = matrix[1, 0] = 0.15 * matrix[0, 0]
    metric = phx.meshing.MeshMetricField(
        _mesh_scope(mesh, 0, vertex_ids),
        np.broadcast_to(matrix, (points.shape[0], 2, 2)),
        minimum_size=1.0 / (3.0 * cells_per_axis),
        maximum_size=1.0 / cells_per_axis,
        maximum_anisotropy=2.0,
    )
    gradient = np.asarray((1.0, 2.0))
    cell_order = np.argsort(cell_ids)
    density = 1.0 + np.where(centroids[:, 0] < 0.5, 0.0, 4.0) + centroids[:, 1] ** 2
    fields = (
        phx.meshing.OmegaHField(
            "potential",
            phx.meshing.OmegaHFieldTransfer.LINEAR,
            _mesh_scope(mesh, 0, vertex_ids),
            points @ gradient,
        ),
        phx.meshing.OmegaHField(
            "density",
            phx.meshing.OmegaHFieldTransfer.CONSERVE,
            _mesh_scope(mesh, 2, cell_ids[cell_order]),
            density[cell_order],
        ),
    )
    with phx.meshing.OmegaHProvider(executable) as provider:
        result, adapt_stage = _measured(
            lambda: provider.execute(
                # ty: ignore[invalid-argument-type]
                source,
                metric,
                fields=fields,
                ranks=ranks,
                gather=True,
            )
        )
    # ty: ignore[unresolved-attribute]
    target = result.target
    target_ids, target_areas, _ = _distribution_cells(target.mesh)
    # ty: ignore[unresolved-attribute]
    potential, transferred = result.fields
    target_points = np.asarray(target.mesh.coordinates)[
        np.argsort(np.asarray(target.mesh.vertex_global_ids))
    ]
    linear_error = float(
        np.max(np.abs(np.asarray(potential.values) - target_points @ gradient))
    )
    rows = _distribution_rows(np.asarray(transferred.scope.entity_ids), target_ids)
    mass_before = float(np.sum(density * areas))
    mass_after = float(np.sum(np.asarray(transferred.values)[rows] * target_areas))
    mass_residual = abs(mass_after - mass_before) / abs(mass_before)
    # ty: ignore[unresolved-attribute]
    conserve = result.evidence.fields[1]
    reported_before = float(np.sum(np.asarray(conserve.integral_before)))
    reported_after = float(np.sum(np.asarray(conserve.integral_after)))
    left = next(zone for zone in target.zones if zone.name == "left")
    left_area = float(
        np.sum(target_areas[np.isin(target_ids, np.asarray(left.scope.entity_ids))])
    )
    partitions = [
        {
            "rank": partition.rank,
            "partition_id": partition.partition_id,
            "owned_cells": int(np.count_nonzero(np.asarray(partition.cell_owned))),
            "ghost_cells": int(np.count_nonzero(np.asarray(partition.cell_ghosts))),
            "owned_vertices": int(np.count_nonzero(np.asarray(partition.vertex_owned))),
            "ghost_vertices": int(np.count_nonzero(np.asarray(partition.vertex_ghosts))),
        }
        # ty: ignore[unresolved-attribute]
        for partition in result.partitions
    ]
    owned = [value["owned_cells"] for value in partitions]
    checks = {
        "audit": target.audit.passed,
        "positive_cells": bool(np.all(target_areas > 0.0)),
        "domain_area": bool(np.isclose(np.sum(target_areas), 1.0, rtol=0.0, atol=1e-12)),
        "refined": target_ids.size > cell_ids.size,
        # ty: ignore[unresolved-attribute]
        "lineage_unknown": result.lineage_status == "unknown",
        "zone_area": bool(np.isclose(left_area, 0.5, rtol=0.0, atol=1e-12)),
        "linear_reproduction": linear_error <= 1e-12,
        "conservation": mass_residual <= 1e-12,
        "reported_conservation": np.isclose(
            reported_after, reported_before, rtol=1e-12, atol=0.0
        ),
        "ownership": sum(owned) == target_ids.size and min(owned) > 0,
        "ghosts": ranks == 1 or all(value["ghost_cells"] > 0 for value in partitions),
        # ty: ignore[unresolved-attribute]
        "ranks": result.evidence.ranks == ranks and len(partitions) == ranks,
        # ty: ignore[unresolved-attribute]
        "distribution": (result.distribution is None) == (ranks == 1),
        # ty: ignore[unresolved-attribute]
        "metric": bool(np.all(np.linalg.eigvalsh(np.asarray(result.metric.values)) > 0)),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Omega_h qualification failed checks: {failed}.")
    record = {
        "status": "passed",
        # ty: ignore[invalid-argument-type]
        "source": _result_identity(source),
        "target": {
            **_result_identity(target),
            # ty: ignore[unresolved-attribute]
            "omega_h_result_id": result.result_id,
            # ty: ignore[unresolved-attribute]
            "lineage_status": result.lineage_status,
        },
        "runtime": {
            **_runtime_identity(),
            # ty: ignore[unresolved-attribute]
            "provider_version": result.provider.version,
            # ty: ignore[unresolved-attribute]
            "actual_version": result.runtime.actual_version,
            "worker_executable": executable,
            # ty: ignore[unresolved-attribute]
            "worker_identity": result.evidence.identity_id,
            # ty: ignore[unresolved-attribute]
            "worker_session": result.evidence.session_id,
            # ty: ignore[unresolved-attribute]
            "ranks": result.evidence.ranks,
        },
        "quality": {
            # ty: ignore[unresolved-attribute]
            "before": _quality_summary(source.quality),
            "after": _quality_summary(target.quality),
            "omega_h_metric_quality": [
                # ty: ignore[unresolved-attribute]
                result.evidence.minimum_quality,
                # ty: ignore[unresolved-attribute]
                result.evidence.maximum_quality,
            ],
            "omega_h_metric_length": [
                # ty: ignore[unresolved-attribute]
                result.evidence.minimum_length,
                # ty: ignore[unresolved-attribute]
                result.evidence.maximum_length,
            ],
            # ty: ignore[unresolved-attribute]
            "iterations": result.evidence.iterations,
        },
        "transfer": {
            # ty: ignore[unresolved-attribute]
            "methods": {value.name: value.method for value in result.evidence.fields},
            "linear_reproduction_error": linear_error,
        },
        "conservation": {
            "integral_before": mass_before,
            "integral_after": mass_after,
            "relative_residual": mass_residual,
            "omega_h_integral_before": reported_before,
            "omega_h_integral_after": reported_after,
        },
        "classification": {"left_zone_area": left_area, "zones": len(target.zones)},
        "partitions": partitions,
        "checks_passed": sorted(checks),
        "resources": {
            "certification": certify_stage,
            "adaptation": adapt_stage,
            # ty: ignore[unresolved-attribute]
            "worker_peak_rss_bytes": result.evidence.peak_rss_bytes,
            "target_retained_bytes": _distribution_retained_bytes(target),
        },
    }
    # ty: ignore[unresolved-attribute]
    if result.distribution is not None:
        # ty: ignore[unresolved-attribute]
        distribution = result.distribution.evidence
        record["distribution"] = {
            # ty: ignore[unresolved-attribute]
            "distribution_id": result.distribution.distribution_id,
            "provenance": distribution.provenance,
            "part_weights": np.asarray(distribution.part_weights).tolist(),
            "imbalance": float(distribution.imbalance),
            "edge_cut": int(distribution.edge_cut),
            "halo_replicas": int(distribution.halo_replicas),
        }
    return record


def qualify_omega_h(worker: str | None) -> dict[str, object]:
    return _qualify_omega_h_ranks(worker, 1)


def qualify_omega_h_distributed(worker: str | None) -> dict[str, object]:
    return _qualify_omega_h_ranks(worker, 2)


def _metric_lattice(
    count: int, lower: float, upper: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and counterclockwise triangles of a ``count x count`` square lattice."""
    axis = np.linspace(lower, upper, count + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((count + 1) ** 2, dtype=np.int32).reshape((count + 1, count + 1))
    first, second = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    third, fourth = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate(
        (
            np.stack((first, second, third), axis=1),
            np.stack((first, third, fourth), axis=1),
        )
    )
    return points, triangles


def _layer_metric_tensors(points: np.ndarray, /) -> np.ndarray:
    """Diagonal internal-layer metric: size 0.04 across x = 1/2 growing to 0.3.

    Every other ambient axis (along the layer, and the normal of an embedded
    planar face) requests size 0.25.
    """
    sizes = np.full(points.shape, 0.25, dtype=np.float64)
    sizes[:, 0] = 0.04 + 0.52 * np.abs(points[:, 0] - 0.5)
    return np.eye(points.shape[1], dtype=np.float64) * sizes[:, None, :] ** -2


def _vertex_scope_rows(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Vertex rows in the sorted global-ID row order of a vertex scope."""
    return np.argsort(np.asarray(mesh.vertex_global_ids), kind="stable")


def _layer_metric(mesh: phx.discretization.CellMesh, /) -> phx.meshing.MeshMetricField:
    rows = _vertex_scope_rows(mesh)
    return phx.meshing.MeshMetricField(
        _mesh_scope(mesh, 0, np.asarray(mesh.vertex_global_ids, dtype=np.int64)),
        _layer_metric_tensors(np.asarray(mesh.coordinates, dtype=np.float64)[rows]),
        minimum_size=0.04,
        maximum_size=0.31,
        maximum_anisotropy=6.5,
    )


def _triangle_areas(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    spatial = np.pad(points, ((0, 0), (0, 3 - points.shape[1])))
    first = spatial[cells[:, 1]] - spatial[cells[:, 0]]
    second = spatial[cells[:, 2]] - spatial[cells[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(first, second), axis=1)


def _mesh_triangles(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    return np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])


def _vertex_dual_areas(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Barycentric dual areas: exact P1 integration weights of the vertex rows."""
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = _mesh_triangles(mesh)
    areas = np.zeros((points.shape[0],), dtype=np.float64)
    shares = np.repeat(_triangle_areas(points, cells)[:, None] / 3.0, 3, axis=1)
    np.add.at(areas, cells, shares)
    return areas


def _metric_unit_measurements(
    mesh: phx.discretization.CellMesh, metric: phx.meshing.MeshMetricField, /
) -> dict[str, object]:
    """Riemannian edge lengths and metric mean-ratio quality of a planar mesh."""
    rows = _vertex_scope_rows(mesh)
    inverse = np.empty_like(rows)
    inverse[rows] = np.arange(rows.shape[0])
    points = np.asarray(mesh.coordinates, dtype=np.float64)[rows]
    values = np.asarray(metric.values, dtype=np.float64)
    lengths = np.asarray(
        phx.meshing.metric_edge_lengths(
            values,
            points,
            # ty: ignore[unresolved-attribute]
            inverse[np.asarray(mesh.connectivity.edges)],
        )
    )
    cells = inverse[_mesh_triangles(mesh)]
    cell_metric = np.asarray(
        phx.meshing.interpolate_mesh_metric(
            values[cells], np.full(cells.shape, 1.0 / 3.0, dtype=np.float64)
        )
    )
    corners = points[cells]
    sides = corners[:, (1, 2, 0)] - corners
    stretched = (cell_metric[:, None] @ sides[..., None])[..., 0]
    squared = np.sum(sides * stretched, axis=(1, 2))
    determinant = cell_metric[:, 0, 0] * cell_metric[:, 1, 1] - cell_metric[:, 0, 1] ** 2
    areas = _triangle_areas(points, cells)
    quality = 4.0 * np.sqrt(3.0) * areas * np.sqrt(determinant) / squared
    unit = (lengths >= 1.0 / np.sqrt(2.0)) & (lengths <= np.sqrt(2.0))
    return {
        "edges": lengths.shape[0],
        "minimum_length": float(np.min(lengths)),
        "mean_length": float(np.mean(lengths)),
        "maximum_length": float(np.max(lengths)),
        "unit_fraction": float(np.mean(unit)),
        "minimum_metric_quality": float(np.min(quality)),
        "mean_metric_quality": float(np.mean(quality)),
    }


def _linear_probe(points: np.ndarray, /) -> np.ndarray:
    return 2.0 + 3.0 * points[:, 0] - 1.5 * points[:, 1]


def _adaptation_transfer_evidence(
    adaptation: phx.meshing.MeshAdaptationResult, /
) -> dict[str, object]:
    """Measured constant/linear reproduction and P1 integrals of the P1 transfer."""
    transfer = adaptation.transfer
    if transfer is None:
        raise RuntimeError(f"The {adaptation.route.value} adaptation has no transfer.")
    source = adaptation.source.mesh
    target = adaptation.target.mesh
    source_points = np.asarray(source.coordinates, dtype=np.float64)
    target_points = np.asarray(target.coordinates, dtype=np.float64)
    source_values = _linear_probe(source_points)
    transferred = np.asarray(transfer.apply(jnp.asarray(source_values)))
    constant = np.asarray(
        transfer.apply(jnp.ones((source_points.shape[0],), dtype=jnp.float64))
    )
    source_area = _vertex_dual_areas(source)
    target_area = _vertex_dual_areas(target)
    return {
        "transfer_id": transfer.transfer_id,
        "preserves_constants": transfer.preserves_constants,
        "preserves_linear": transfer.preserves_linear,
        "conservative": transfer.conservative,
        "positivity_preserving": transfer.positivity_preserving,
        "constant_error": float(np.max(np.abs(constant - 1.0))),
        "linear_error": float(np.max(np.abs(transferred - _linear_probe(target_points)))),
        "area_residual": abs(float(np.sum(target_area) - np.sum(source_area))),
        "linear_integral_residual": abs(
            float(np.dot(target_area, transferred) - np.dot(source_area, source_values))
        ),
    }


def _metric_operation_evidence(evidence: Any, /) -> dict[str, object]:
    """Fields shared by `LocalMetricEvidence` and `DeviceMetricEvidence`."""
    return {
        "evidence_id": evidence.evidence_id,
        "passes": evidence.passes,
        "splits": evidence.splits,
        "collapses": evidence.collapses,
        "flips": evidence.flips,
        "relocations": evidence.relocations,
        "rejected_uncertain": evidence.rejected_uncertain,
        "rejected_invalid": evidence.rejected_invalid,
        "converged": evidence.converged,
        "stalled": evidence.stalled,
        "measured_edges": evidence.measured_edges,
        "out_of_range_edges": evidence.out_of_range_edges,
        "minimum_metric_length": evidence.minimum_metric_length,
        "maximum_metric_length": evidence.maximum_metric_length,
        "unit_fraction": evidence.unit_fraction,
        "minimum_metric_quality": evidence.minimum_metric_quality,
    }


def qualify_metric_adaptation() -> dict[str, object]:
    points, triangles = _metric_lattice(6, 0.0, 1.0)
    source, certify_record = _measured(
        lambda: phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh.from_triangles(points, triangles), _contract()
        )
    )
    # ty: ignore[unresolved-attribute]
    metric = _layer_metric(source.mesh)
    request = phx.meshing.MetricMeshAdaptation(metric)
    native_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    )
    native, native_record = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            # ty: ignore[invalid-argument-type]
            phx.meshing.prepare_mesh_adaptation(source, request, policy=native_policy)
        )
    )
    device_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_METRIC_2D,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=256, cell_capacity=512
        ),
    )
    prepared, prepare_record = _measured(
        lambda: phx.meshing.prepare_device_metric_adaptation(
            # ty: ignore[invalid-argument-type]
            source,
            request,
            policy=device_policy,
        )
    )
    update, adapt_record = _measured(
        lambda: jax.block_until_ready(
            # ty: ignore[unresolved-attribute]
            phx.meshing.adapt_device_metric(prepared.layout, prepared.state)
        )
    )
    # ty: ignore[unresolved-attribute]
    report = jax.device_get(update.report)
    device, commit_record = _measured(
        # ty: ignore[invalid-argument-type, unresolved-attribute]
        lambda: phx.meshing.commit_device_metric_adaptation(prepared, update.state)
    )
    relocation, relocation_record = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                # ty: ignore[invalid-argument-type]
                source,
                phx.meshing.RelocationMeshAdaptation(metric),
                policy=native_policy,
            )
        )
    )
    adapted = {"native": native, "device": device}
    # ty: ignore[unresolved-attribute]
    if any(result.metric is None for result in (*adapted.values(), relocation)):
        raise RuntimeError("A metric adaptation did not bind its metric to the target.")
    # ty: ignore[unresolved-attribute]
    before = _metric_unit_measurements(source.mesh, metric)
    after = {
        # ty: ignore[unresolved-attribute]
        name: _metric_unit_measurements(result.target.mesh, result.metric)
        for name, result in adapted.items()
    }
    # ty: ignore[unresolved-attribute]
    relocated = _metric_unit_measurements(relocation.target.mesh, relocation.metric)
    transfer = {
        # ty: ignore[invalid-argument-type]
        name: _adaptation_transfer_evidence(result)
        for name, result in adapted.items()
    }
    # ty: ignore[invalid-argument-type]
    relocation_transfer = _adaptation_transfer_evidence(relocation)
    checks = {
        # ty: ignore[unresolved-attribute]
        "native target audited": native.target.audit.passed,
        # ty: ignore[unresolved-attribute]
        "device target audited": device.target.audit.passed,
        # ty: ignore[unresolved-attribute]
        "relocation target audited": relocation.target.audit.passed,
        "device passes did not fail": not bool(report.failed),
        "relocation kept the topology": (
            # ty: ignore[unresolved-attribute]
            relocation.target.mesh.topology_id == source.mesh.topology_id
            # ty: ignore[unresolved-attribute]
            and relocation.evidence.relocations > 0
            # ty: ignore[unresolved-attribute]
            and relocation.evidence.splits == relocation.evidence.collapses == 0
        ),
    }
    for name, result in adapted.items():
        checks[f"{name} moved toward the unit mesh"] = (
            # ty: ignore[unsupported-operator]
            after[name]["unit_fraction"] > before["unit_fraction"]
            # ty: ignore[unresolved-attribute]
            and result.evidence.unit_fraction >= 0.9
        )
        checks[f"{name} transfer reproduces linear fields"] = (
            transfer[name]["preserves_linear"]
            # ty: ignore[unsupported-operator]
            and transfer[name]["linear_error"] <= 1.0e-12
            # ty: ignore[unsupported-operator]
            and transfer[name]["constant_error"] <= 1.0e-12
        )
        checks[f"{name} target covers the source area"] = (
            # ty: ignore[unsupported-operator]
            transfer[name]["area_residual"] <= 1.0e-12
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Metric adaptation qualification failed: {failed}.")
    return {
        "status": "passed",
        "checks": sorted(checks),
        # ty: ignore[invalid-argument-type]
        "source": {**_result_identity(source), "metric_id": metric.metric_id},
        "target": {
            name: {
                # ty: ignore[unresolved-attribute]
                **_result_identity(result.target),
                # ty: ignore[unresolved-attribute]
                "metric_id": result.metric.metric_id,
            }
            for name, result in (*adapted.items(), ("relocation", relocation))
        },
        "runtime": {
            **_runtime_identity(),
            # ty: ignore[unresolved-attribute]
            "native_predicate_mode": native.evidence.predicate_mode.value,
            # ty: ignore[unresolved-attribute]
            "device_layout_id": prepared.layout.signature_id,
        },
        "quality": {
            # ty: ignore[unresolved-attribute]
            "before": {**_quality_summary(source.quality), "metric": before},
            "after": {
                # ty: ignore[unresolved-attribute]
                name: {**_quality_summary(result.target.quality), "metric": after[name]}
                for name, result in adapted.items()
            },
            "relocation": {
                # ty: ignore[unresolved-attribute]
                **_quality_summary(relocation.target.quality),
                "metric": relocated,
            },
        },
        "transfer": {**transfer, "relocation": relocation_transfer},
        "conservation": {
            name: {
                "area_residual": record["area_residual"],
                "linear_integral_residual": record["linear_integral_residual"],
            }
            for name, record in (*transfer.items(), ("relocation", relocation_transfer))
        },
        "routes": {
            "native": {
                # ty: ignore[unresolved-attribute]
                "adaptation_status": native.status.value,
                # ty: ignore[unresolved-attribute]
                "evidence": _metric_operation_evidence(native.evidence),
            },
            "device": {
                # ty: ignore[unresolved-attribute]
                "adaptation_status": device.status.value,
                "evidence": {
                    # ty: ignore[unresolved-attribute]
                    **_metric_operation_evidence(device.evidence),
                    # ty: ignore[unresolved-attribute]
                    "rejected_cavity": device.evidence.rejected_cavity,
                    # ty: ignore[unresolved-attribute]
                    "status_flags": int(device.evidence.status),
                    # ty: ignore[unresolved-attribute]
                    "status_names": device.evidence.status.name,
                },
                "report": {
                    "status_flags": int(report.status),
                    "passes": int(report.passes),
                    "unit_fraction": float(report.unit_fraction),
                },
            },
            "relocation": {
                # ty: ignore[unresolved-attribute]
                "adaptation_status": relocation.status.value,
                # ty: ignore[unresolved-attribute]
                "evidence": _metric_operation_evidence(relocation.evidence),
            },
        },
        "resources": {
            "certify": certify_record,
            "native_adaptation": native_record,
            "device_prepare": prepare_record,
            "device_adapt_cold": adapt_record,
            "device_commit": commit_record,
            "relocation": relocation_record,
            "device_capacity": {
                # ty: ignore[unresolved-attribute]
                "vertices": prepared.layout.vertex_capacity,
                # ty: ignore[unresolved-attribute]
                "cells": prepared.layout.cell_capacity,
            },
            "device_state_bytes": sum(
                leaf.nbytes
                # ty: ignore[unresolved-attribute]
                for leaf in jax.tree.leaves(prepared.state)
            ),
        },
    }


def _gmsh_metric_run(
    square: Any, control: phx.meshing.BackgroundMetricControl, options: Any, /
) -> tuple[phx.meshing.CellMeshingResult, dict[str, object]]:
    provider = phx.meshing.GmshProvider(options)
    scope = provider.whole_scope(square, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
    )
    plan, plan_record = _measured(
        lambda: provider.plan(square, specification, background_metric=control)
    )
    # ty: ignore[unresolved-attribute]
    result, execute_record = _measured(plan.execute)
    # ty: ignore[unresolved-attribute]
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError(f"Gmsh {control.mode.value} metric mesh failed its audit.")
    prefix = f"background_metric:{control.control_id}:"
    compliance = {
        key.removeprefix(prefix): value
        # ty: ignore[unresolved-attribute]
        for key, value in result.compliance.achieved
        if key.startswith(prefix)
    }
    if sorted(compliance) != [
        "maximum_edge_length",
        "mean_edge_length",
        "minimum_edge_length",
        "unit_edge_fraction",
    ]:
        raise RuntimeError("Gmsh compliance lacks the background metric edge lengths.")
    # ty: ignore[unresolved-attribute]
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    # ty: ignore[unresolved-attribute]
    edges = np.asarray(result.mesh.connectivity.edges)
    vectors = points[edges[:, 1]] - points[edges[:, 0]]
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    diagonal = np.diagonal(_layer_metric_tensors(midpoints), axis1=1, axis2=2)
    analytic = np.sqrt(np.sum(vectors**2 * diagonal, axis=1))
    # ty: ignore[unresolved-attribute]
    area = float(np.sum(_triangle_areas(points, _mesh_triangles(result.mesh))))
    extent = np.mean(np.abs(vectors), axis=0)
    # ty: ignore[invalid-return-type]
    return result, {
        "mode": control.mode.value,
        "control_id": control.control_id,
        # ty: ignore[unresolved-attribute]
        "plan_id": plan.plan_id,
        "algorithm_2d": options.algorithm_2d.value,
        "compliance_metric_edge_lengths": compliance,
        "analytic_metric_edge_lengths": {
            "minimum": float(np.min(analytic)),
            "mean": float(np.mean(analytic)),
            "maximum": float(np.max(analytic)),
            "unit_fraction": float(
                np.mean((analytic >= 1.0 / np.sqrt(2.0)) & (analytic <= np.sqrt(2.0)))
            ),
        },
        "mean_edge_extent_y_over_x": float(extent[1] / extent[0]),
        "area": area,
        "resources": {"plan": plan_record, "execute": execute_record},
    }


def qualify_gmsh_metric() -> dict[str, object]:
    from importlib.util import find_spec

    if find_spec("gmsh") is None:
        return _missing_dependency(
            "gmsh",
            "Install the optional Gmsh Python package: "
            "pip install 'phydrax[meshing-gmsh]'.",
        )
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.gp import gp_Pnt

    contract = _contract()
    grid, triangles = _metric_lattice(8, -0.01, 1.01)
    background = phx.discretization.CellMesh(
        np.concatenate((grid, np.zeros((grid.shape[0], 1), dtype=np.float64)), axis=1),
        (phx.discretization.CellBlock("triangles", "triangle", triangles),),
    )
    metric = _layer_metric(background)
    certified_background, background_record = _measured(
        lambda: phx.meshing.certify_cell_mesh(background, contract)
    )
    controls = {
        mode: phx.meshing.BackgroundMetricControl(
            background, metric, contract, mode=mode, maximum_metric_edge_length=2.0
        )
        for mode in phx.meshing.BackgroundMetricMode
    }
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
        polygon.Add(gp_Pnt(x, y, 0.0))
    polygon.Close()
    face = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    with TemporaryDirectory(prefix="phydrax-gmsh-metric-") as temporary:
        square = phx.geometry.persist_occt_shape(
            face,
            Path(temporary) / "square.brep",
            coordinate_contract=contract,
            linear_deflection=0.01,
            angular_deflection=0.2,
        )
        isotropic, isotropic_record = _gmsh_metric_run(
            square,
            controls[phx.meshing.BackgroundMetricMode.ISOTROPIC],
            phx.meshing.GmshOptions(),
        )
        anisotropic, anisotropic_record = _gmsh_metric_run(
            square,
            controls[phx.meshing.BackgroundMetricMode.ANISOTROPIC],
            phx.meshing.GmshOptions(algorithm_2d=phx.meshing.GmshSurfaceAlgorithm.BAMG),
        )
    results = {"isotropic": isotropic, "anisotropic": anisotropic}
    records = {"isotropic": isotropic_record, "anisotropic": anisotropic_record}
    # ty: ignore[not-subscriptable]
    unit = anisotropic_record["compliance_metric_edge_lengths"]["unit_edge_fraction"]
    checks = {
        # ty: ignore[unresolved-attribute]
        "background certified": certified_background.audit.passed,
        "isotropic mode refines every direction": (
            isotropic.mesh.entity_set(2).count > anisotropic.mesh.entity_set(2).count
        ),
        "anisotropic cells stretch along the layer": (
            # ty: ignore[unsupported-operator]
            anisotropic_record["mean_edge_extent_y_over_x"] > 1.5
        ),
        "anisotropic mesh is near unit": unit >= 0.8,
    }
    for name, record in records.items():
        # ty: ignore[unsupported-operator]
        checks[f"{name} mesh covers the face"] = abs(record["area"] - 1.0) <= 1.0e-12
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Gmsh metric qualification failed: {failed}.")
    return {
        "status": "passed",
        "checks": sorted(checks),
        "source": {
            "brep_source_revision": square.source_revision,
            # ty: ignore[invalid-argument-type]
            "background": _result_identity(certified_background),
            "metric_id": metric.metric_id,
        },
        "target": {name: _result_identity(result) for name, result in results.items()},
        "runtime": {
            **_runtime_identity(),
            "provider": anisotropic.provider.name,
            "provider_version": anisotropic.runtime.actual_version,
        },
        "quality": {
            # ty: ignore[unresolved-attribute]
            "before": _quality_summary(certified_background.quality),
            "after": {
                name: _quality_summary(result.quality) for name, result in results.items()
            },
        },
        "conservation": {
            # ty: ignore[unsupported-operator]
            name: {"area": record["area"], "area_residual": abs(record["area"] - 1.0)}
            for name, record in records.items()
        },
        "routes": {
            name: {key: value for key, value in record.items() if key != "resources"}
            for name, record in records.items()
        },
        "resources": {
            "background_certify": background_record,
            **{name: record["resources"] for name, record in records.items()},
        },
    }


def _layer_wall_mesh(points: np.ndarray, triangles: np.ndarray, /) -> Any:
    return phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "wall",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
                global_ids=np.arange(triangles.shape[0], dtype=np.int64),
            ),
        ),
    )


def _layer_box_wall(half_width: float, count: int, /) -> Any:
    """Outward-oriented structured triangulation of a centered box surface."""
    grid = np.linspace(-half_width, half_width, count + 1)
    u, v = (value.ravel() for value in np.meshgrid(grid, grid, indexing="ij"))
    lower = (np.arange(count)[:, None] * (count + 1) + np.arange(count)).ravel()
    quads = np.stack((lower, lower + count + 1, lower + count + 2, lower + 1), axis=1)
    local = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    points = []
    triangles = []
    for axis in range(3):
        tangent = [value for value in range(3) if value != axis]
        for bound in (-half_width, half_width):
            face = np.empty((u.size, 3), dtype=np.float64)
            face[:, axis] = bound
            face[:, tangent[0]] = u
            face[:, tangent[1]] = v
            triangles.append(local + len(points) * u.size)
            points.append(face)
    coordinates, inverse = np.unique(
        np.round(np.concatenate(points), 12), axis=0, return_inverse=True
    )
    triangles = inverse.reshape(-1)[np.concatenate(triangles)]
    corners = coordinates[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    inward = np.sum(normals * corners.mean(axis=1), axis=1) < 0.0
    triangles[inward] = triangles[inward][:, ::-1]
    return _layer_wall_mesh(coordinates, triangles)


def _layer_plate(count: int, height: float, /, *, flip: bool) -> Any:
    """Unit-square plate at ``height`` growing along +z (or -z when flipped)."""
    grid = np.linspace(0.0, 1.0, count + 1)
    x, y = (value.ravel() for value in np.meshgrid(grid, grid, indexing="ij"))
    points = np.stack((x, y, np.full(x.size, height)), axis=1)
    lower = (np.arange(count)[:, None] * (count + 1) + np.arange(count)).ravel()
    quads = np.stack((lower, lower + count + 1, lower + count + 2, lower + 1), axis=1)
    triangles = np.concatenate((quads[:, (0, 1, 2)], quads[:, (0, 2, 3)]))
    return points, triangles[:, ::-1] if flip else triangles


def _advancing_layer_control(wall: Any, schedule: Any, /, **options: Any) -> Any:
    cells = wall.entity_set(2)
    return phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope(
            wall.mesh_id,
            wall.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            2,
            cells.entity_set_id,
            cells.entity_ids,
        ),
        schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        **options,
    )


def _layer_enclosed_volume(points: np.ndarray, triangles: np.ndarray, /) -> float:
    """Divergence-theorem volume enclosed by outward-oriented triangles."""
    corners = points[triangles]
    return float(np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2])) / 6.0)


def _layer_block_rows(mesh: Any, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )


def _layer_retained_bytes(tree: Any, /) -> int:
    leaves = {
        id(leaf): leaf for leaf in jax.tree.leaves(tree) if isinstance(leaf, jax.Array)
    }
    return sum(leaf.nbytes for leaf in leaves.values())


def _layer_entity_rows(identifiers: Any, requested: Any, /) -> np.ndarray:
    """Rows of the unique ``identifiers`` holding every ``requested`` ID."""
    values = np.asarray(identifiers, dtype=np.int64)
    wanted = np.asarray(requested, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    rows = order[np.searchsorted(values[order], wanted).clip(0, values.size - 1)]
    if not np.array_equal(values[rows], wanted):
        raise RuntimeError("Requested entity IDs are absent from the mesh.")
    return rows


def _layer_evidence(evidence: Any, /) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "requested_thicknesses": list(evidence.requested_thicknesses),
        "measured_thicknesses": np.asarray(evidence.achieved_thicknesses).tolist(),
        "minimum_thicknesses": np.asarray(evidence.minimum_thicknesses).tolist(),
        "maximum_thicknesses": np.asarray(evidence.maximum_thicknesses).tolist(),
        "measured_growth_rates": np.asarray(evidence.achieved_growth_rates).tolist(),
        "layer_active": np.asarray(evidence.layer_active).tolist(),
        "columns": evidence.column_count,
        "fan_columns": evidence.fan_column_count,
        "corner_patches": evidence.corner_patch_count,
        "convex_ridges": evidence.convex_ridge_count,
        "concave_ridges": evidence.concave_ridge_count,
        "rim_vertices": evidence.rim_vertex_count,
        "minimum_visibility": evidence.minimum_visibility,
        "maximum_stretch": evidence.maximum_stretch,
        "unconverged_visibility": evidence.unconverged_visibility_count,
        "collision_policy": evidence.collision_policy.value,
        "predicted_collision_vertices": evidence.predicted_collision_vertex_count,
        "detected_collisions": evidence.detected_collision_count,
        "resolution_iterations": evidence.resolution_iterations,
        "reduced_vertices": evidence.reduced_vertex_count,
        "terminated_vertices": evidence.terminated_vertex_count,
        "merged_vertices": evidence.merged_vertex_count,
        "minimum_scale": evidence.minimum_scale,
        "cell_counts": dict(evidence.cell_counts),
        "certified_valid_cells": evidence.certified_valid_count,
    }


def _layer_certificate_counts(certificate: Any, /) -> dict[str, object]:
    return {
        "certificate_id": certificate.certificate_id,
        "cells": certificate.status.shape[0],
        "certified_valid": certificate.certified_valid_count,
        "invalid": certificate.invalid_count,
        "unresolved": certificate.unresolved_count,
    }


def _layer_quality(mesh: Any, /) -> Any:
    return phx.meshing.summarize_cell_quality(phx.meshing.evaluate_cell_quality(mesh))


def qualify_boundary_layer() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact layer certification requires the native meshcore library; build "
            "it and set PHYDRAX_MESHCORE_LIBRARY to the shared library path.",
        )
    schedule = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)
    wall = _layer_box_wall(0.3, 2)
    control = _advancing_layer_control(
        wall,
        schedule,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
    )
    layers, prepare = _measured(
        lambda: phx.meshing.prepare_boundary_layers(wall, control)
    )
    before, wall_quality = _measured(lambda: _layer_quality(wall))
    # ty: ignore[unresolved-attribute]
    after, layer_quality = _measured(lambda: _layer_quality(layers.mesh))
    certificate, certification = _measured(
        # ty: ignore[unresolved-attribute]
        lambda: phx.discretization.certify_cell_geometry_validity(layers.mesh)
    )
    # ty: ignore[unresolved-attribute]
    evidence = layers.evidence
    # ty: ignore[unresolved-attribute]
    cells = sum(block.cell_count for block in layers.mesh.blocks)
    requested = np.asarray(schedule.thicknesses, dtype=np.float64)
    measured = np.stack(
        (
            np.asarray(evidence.achieved_thicknesses),
            np.asarray(evidence.minimum_thicknesses),
            np.asarray(evidence.maximum_thicknesses),
        )
    )
    thickness_error = float(np.max(np.abs(measured - requested) / requested))
    growth_error = float(np.max(np.abs(np.asarray(evidence.achieved_growth_rates) - 1.2)))
    # ty: ignore[unresolved-attribute]
    layer_volume = float(np.sum(np.asarray(after.evaluation.measures)))
    enclosed = _layer_enclosed_volume(
        # ty: ignore[unresolved-attribute]
        np.asarray(layers.cap.coordinates),
        # ty: ignore[unresolved-attribute]
        _layer_block_rows(layers.cap),
    ) - _layer_enclosed_volume(np.asarray(wall.coordinates), _layer_block_rows(wall))
    volume_residual = abs(layer_volume - enclosed) / enclosed
    if (
        # ty: ignore[unresolved-attribute]
        not layers.closed_cap
        # ty: ignore[unresolved-attribute]
        or layers.validity.certified_valid_count != cells
        # ty: ignore[unresolved-attribute]
        or certificate.certified_valid_count != cells
        or evidence.convex_ridge_count != 24
        or evidence.corner_patch_count != 8
        or evidence.detected_collision_count != 0
        or thickness_error > 1.0e-9
        or growth_error > 1.0e-9
        # ty: ignore[unresolved-attribute]
        or after.sampled_invalid_count != 0
        or volume_residual > 1.0e-10
    ):
        raise RuntimeError(
            "Native layers missed the schedule, certification, or volume closure."
        )

    gap = 0.05
    lower, lower_triangles = _layer_plate(4, 0.0, flip=False)
    upper, upper_triangles = _layer_plate(4, gap, flip=True)
    channel = _layer_wall_mesh(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )
    reduction = _advancing_layer_control(
        channel,
        schedule,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
        minimum_thickness_fraction=0.2,
    )
    reduced, reduce = _measured(
        lambda: phx.meshing.prepare_boundary_layers(channel, reduction)
    )
    reduced_certificate, reduced_certification = _measured(
        # ty: ignore[unresolved-attribute]
        lambda: phx.discretization.certify_cell_geometry_validity(reduced.mesh)
    )
    # The cap faces away from its layers: lower-front faces point along +z.
    # ty: ignore[unresolved-attribute]
    cap_points = np.asarray(reduced.cap.coordinates)
    # ty: ignore[unresolved-attribute]
    cap_corners = cap_points[_layer_block_rows(reduced.cap)]
    upward = (
        np.cross(
            cap_corners[:, 1] - cap_corners[:, 0], cap_corners[:, 2] - cap_corners[:, 0]
        )[:, 2]
        > 0.0
    )
    clearance = float(
        np.min(cap_corners[~upward][..., 2]) - np.max(cap_corners[upward][..., 2])
    )
    # ty: ignore[unresolved-attribute]
    reduced_evidence = reduced.evidence
    # ty: ignore[unresolved-attribute]
    reduced_cells = sum(block.cell_count for block in reduced.mesh.blocks)
    if (
        reduced_evidence.predicted_collision_vertex_count != 50
        or reduced_evidence.reduced_vertex_count != 50
        or not 0.2 <= reduced_evidence.minimum_scale < 1.0
        or clearance <= 0.0
        or np.max(np.abs(np.asarray(reduced_evidence.achieved_growth_rates) - 1.2))
        > 1.0e-9
        # ty: ignore[unresolved-attribute]
        or reduced_certificate.certified_valid_count != reduced_cells
    ):
        raise RuntimeError("REDUCE_THICKNESS did not resolve the narrow-gap collision.")
    return {
        "status": "passed",
        "source": {
            "wall_mesh_id": wall.mesh_id,
            "topology_id": wall.topology_id,
            "vertices": wall.coordinates.shape[0],
            "cells": wall.blocks[0].cell_count,
            "control_id": control.control_id,
            "schedule_id": schedule.schedule_id,
        },
        "target": {
            # ty: ignore[unresolved-attribute]
            "result_id": layers.result_id,
            # ty: ignore[unresolved-attribute]
            "mesh_id": layers.mesh.mesh_id,
            # ty: ignore[unresolved-attribute]
            "topology_id": layers.mesh.topology_id,
            # ty: ignore[unresolved-attribute]
            "vertices": layers.mesh.coordinates.shape[0],
            "cells": cells,
            # ty: ignore[unresolved-attribute]
            "cap_mesh_id": layers.cap.mesh_id,
            # ty: ignore[unresolved-attribute]
            "closed_cap": layers.closed_cap,
        },
        "runtime": _runtime_identity(),
        "quality": {
            # ty: ignore[invalid-argument-type]
            "before": _quality_summary(before),
            # ty: ignore[invalid-argument-type]
            "after": _quality_summary(after),
        },
        "conservation": {
            "layer_cell_volume": layer_volume,
            "cap_minus_wall_enclosed_volume": enclosed,
            "relative_volume_residual": volume_residual,
        },
        "certificates": {
            # ty: ignore[unresolved-attribute]
            "layer_generation": _layer_certificate_counts(layers.validity),
            "independent": _layer_certificate_counts(certificate),
        },
        "boundary_layer": {
            **_layer_evidence(evidence),
            "maximum_relative_thickness_error": thickness_error,
            "maximum_growth_rate_error": growth_error,
        },
        "collision": {
            "gap": gap,
            "wall_faces": channel.blocks[0].cell_count,
            # ty: ignore[unresolved-attribute]
            "result_id": reduced.result_id,
            "front_clearance": clearance,
            "certificate": _layer_certificate_counts(reduced_certificate),
            **_layer_evidence(reduced_evidence),
        },
        "resources": {
            "prepare_boundary_layers": prepare,
            "wall_quality": wall_quality,
            "layer_quality": layer_quality,
            "independent_certification": certification,
            "collision_prepare_boundary_layers": reduce,
            "collision_certification": reduced_certification,
            "retained_boundary_layer_bytes": _layer_retained_bytes(layers),
        },
    }


def _gmsh_layered_route(
    provider: Any, source: Any, route: Any, corner: Any, schedule: Any, tolerance: Any, /
) -> Any:
    """Plan, execute, and independently audit one layered Gmsh volume route."""
    points = np.asarray(source.mesh_vertices)
    face_ids = np.asarray(source.triangle_face_ids)
    inner = np.all(np.linalg.norm(points[np.asarray(source.mesh_faces)], axis=2) < 0.5, 1)
    wall_faces = tuple(source.face_ids[index] for index in np.unique(face_ids[inner]))
    whole = provider.whole_scope(source, 3)
    control = phx.meshing.BoundaryLayerControl(
        provider.entity_scope(source, wall_faces),
        schedule,
        route=route,
        volume_scope=whole,
        corner=corner,
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("pyramid", "hexahedron"),
                allow_mixed=True,
            ),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(whole, 0.25, maximum_growth_rate=10.0),
        ),
        layer_controls=(control,),
    )
    plan, planning = _measured(lambda: provider.plan(source, specification))
    # ty: ignore[unresolved-attribute]
    result, execution = _measured(plan.execute)
    certificate, certification = _measured(
        # ty: ignore[unresolved-attribute]
        lambda: phx.discretization.certify_cell_geometry_validity(result.mesh)
    )
    # ty: ignore[unresolved-attribute]
    mesh = result.mesh
    connectivity = mesh.connectivity
    owners = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    face_entities = mesh.entity_set(2).entity_ids
    patches = {
        patch.name: _layer_entity_rows(face_entities, patch.scope.entity_ids)
        # ty: ignore[unresolved-attribute]
        for patch in result.patches
    }
    # ty: ignore[unresolved-attribute]
    zones = {zone.name: zone.scope.entity_ids for zone in result.zones}
    if set(zones) != {"boundary-layer", "core"} or set(patches) != {
        "wall",
        "layer-core-interface",
        "outer",
    }:
        raise RuntimeError("Layered Gmsh result lost its zones or patches.")
    cell_entities = mesh.entity_set(3).entity_ids
    zone_of = np.full((cell_entities.shape[0],), -1, dtype=np.int8)
    zone_of[_layer_entity_rows(cell_entities, zones["boundary-layer"])] = 0
    zone_of[_layer_entity_rows(cell_entities, zones["core"])] = 1

    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    entry_cells = np.repeat(np.arange(zone_of.size), np.diff(cell_face_offsets))
    entry_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    entry_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)

    def triangles(faces: Any, zones: Any) -> Any:
        """Triangles of ``faces`` oriented out of their incident cells in ``zones``."""
        selected = np.isin(entry_faces, faces) & np.isin(zone_of[entry_cells], zones)
        rows = entry_faces[selected]
        if rows.size != faces.size or np.any(offsets[rows + 1] - offsets[rows] != 3):
            raise RuntimeError("Layered Gmsh wall and interface faces must be triangles.")
        vertices = values[offsets[rows][:, None] + np.arange(3)]
        return np.where(entry_signs[selected][:, None] > 0.0, vertices, vertices[:, ::-1])

    interface = patches["layer-core-interface"]
    boundary = np.flatnonzero(neighbors < 0)
    conforming = (
        np.all(zone_of >= 0)
        and np.array_equal(
            np.sort(
                np.stack((zone_of[owners[interface]], zone_of[neighbors[interface]]), 1),
                1,
            ),
            np.tile((0, 1), (interface.size, 1)),
        )
        and np.array_equal(
            np.sort(np.concatenate((patches["wall"], patches["outer"]))), boundary
        )
    )
    coordinates = np.asarray(mesh.coordinates)
    # ty: ignore[unresolved-attribute]
    evaluation = result.quality.evaluation
    measures = np.asarray(evaluation.measures)
    measure_rows = _layer_entity_rows(evaluation.cell_global_ids, cell_entities)
    cell_measures = measures[measure_rows]
    domain_volume = float(np.sum(cell_measures))
    boundary_volume = _layer_enclosed_volume(
        coordinates,
        triangles(np.concatenate((patches["wall"], patches["outer"])), (0, 1)),
    )
    wall_triangles = triangles(patches["wall"], (0,))
    layer_volume = float(np.sum(cell_measures[zone_of == 0]))
    layer_enclosed = _layer_enclosed_volume(
        coordinates, np.concatenate((wall_triangles, triangles(interface, (0,))))
    )
    domain_residual = abs(domain_volume - boundary_volume) / boundary_volume
    layer_residual = abs(layer_volume - layer_enclosed) / layer_enclosed
    wall_vertices, wall_inverse = np.unique(wall_triangles, return_inverse=True)
    before = _layer_quality(
        _layer_wall_mesh(coordinates[wall_vertices], wall_inverse.reshape((-1, 3)))
    )
    key = f"layer:{control.control_id}:"
    achieved = {
        name.removeprefix(key): value
        # ty: ignore[unresolved-attribute]
        for name, value in result.compliance.achieved
        if name.startswith(key)
        or name in ("fixed_boundary_bitwise", "layer_interface_compliance")
    }
    requested_values = np.asarray(schedule.thicknesses, dtype=np.float64)
    thickness_error = float(
        np.max(
            np.abs(
                np.asarray(
                    [
                        achieved[f"thickness:{index}"]
                        for index in range(schedule.layer_count)
                    ]
                )
                - requested_values
            )
            / requested_values
        )
    )
    growth_error = float(
        np.max(
            np.abs(
                np.asarray(
                    [
                        achieved[f"growth:{index}"]
                        for index in range(1, schedule.layer_count)
                    ]
                )
                - np.asarray(schedule.growth_rates)
            )
        )
    )
    cells = sum(block.cell_count for block in mesh.blocks)
    if (
        # ty: ignore[unresolved-attribute]
        not result.audit.passed
        # ty: ignore[unresolved-attribute]
        or not result.compliance.passed
        # ty: ignore[unresolved-attribute]
        or not all(association.complete for association in result.associations)
        or not conforming
        or achieved["fixed_boundary_bitwise"] != 1.0
        or achieved["layer_interface_compliance"] != 1.0
        or achieved["layer_count"] != schedule.layer_count
        or thickness_error > tolerance
        or growth_error > tolerance * schedule.growth_rates[0]
        # ty: ignore[unresolved-attribute]
        or certificate.certified_valid_count != cells
        # ty: ignore[unresolved-attribute]
        or result.quality.sampled_invalid_count != 0
        or domain_residual > 1.0e-10
        or layer_residual > 1.0e-10
    ):
        raise RuntimeError(f"Gmsh {route.value} layers failed audit or compliance.")
    return {
        # ty: ignore[unresolved-attribute]
        "plan_id": plan.plan_id,
        "control_id": control.control_id,
        # ty: ignore[invalid-argument-type]
        "target": _result_identity(result),
        "provider": {
            # ty: ignore[unresolved-attribute]
            "version": result.runtime.actual_version,
            # ty: ignore[unresolved-attribute]
            "runtime_id": result.runtime.runtime_id,
            # ty: ignore[unresolved-attribute]
            "deterministic": result.runtime.deterministic,
        },
        "stages": {
            stage.stage.value: stage.status.value
            # ty: ignore[unresolved-attribute]
            for stage in result.trace.stages
        },
        "cell_counts": {block.cell_kind: block.cell_count for block in mesh.blocks},
        "zone_cells": {name: ids.shape[0] for name, ids in zones.items()},
        "patch_faces": {name: rows.size for name, rows in patches.items()},
        "organization_conforming": bool(conforming),
        "compliance": {
            **achieved,
            "maximum_relative_thickness_error": thickness_error,
            "maximum_growth_rate_error": growth_error,
            "tolerance": tolerance,
        },
        "certificate": _layer_certificate_counts(certificate),
        "quality": {
            "before": _quality_summary(before),
            # ty: ignore[unresolved-attribute]
            "after": _quality_summary(result.quality),
        },
        "conservation": {
            "domain_cell_volume": domain_volume,
            "domain_boundary_enclosed_volume": boundary_volume,
            "domain_relative_residual": domain_residual,
            "layer_cell_volume": layer_volume,
            "layer_boundary_enclosed_volume": layer_enclosed,
            "layer_relative_residual": layer_residual,
        },
        "resources": {
            "plan": planning,
            "execute": execution,
            "independent_certification": certification,
            "retained_mesh_bytes": _layer_retained_bytes(mesh),
        },
    }


def qualify_gmsh_boundary_layer() -> dict[str, object]:
    from importlib.util import find_spec

    if find_spec("gmsh") is None:
        return _missing_dependency(
            "gmsh",
            "Install the optional provider: pip install 'phydrax[meshing-gmsh]'.",
        )
    if find_spec("OCP") is None:
        return _missing_dependency(
            "OCP", "Install OpenCascade bindings (cadquery-ocp) for B-Rep persistence."
        )
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact layer certification requires the native meshcore library; build "
            "it and set PHYDRAX_MESHCORE_LIBRARY to the shared library path.",
        )
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt

    schedule = phx.meshing.LayerSchedule.geometric(3, 0.02, growth_rate=1.25)
    provider = phx.meshing.GmshProvider()
    with TemporaryDirectory(prefix="phydrax-gmsh-boundary-layer-") as temporary:
        box = BRepPrimAPI_MakeBox(gp_Pnt(-1.0, -1.0, -1.0), 2.0, 2.0, 2.0).Shape()
        ball = BRepPrimAPI_MakeSphere(gp_Pnt(0.0, 0.0, 0.0), 0.4).Shape()
        source = phx.geometry.persist_occt_shape(
            BRepAlgoAPI_Cut(box, ball).Shape(),
            Path(temporary) / "ball-in-box.brep",
            coordinate_contract=_contract(),
            linear_deflection=0.01,
            angular_deflection=0.1,
        )
        routes = {
            "advancing": _gmsh_layered_route(
                provider,
                source,
                phx.meshing.BoundaryLayerRoute.ADVANCING,
                phx.meshing.BoundaryLayerCornerPolicy.FAN,
                schedule,
                1.0e-6,
            ),
            "provider": _gmsh_layered_route(
                provider,
                source,
                phx.meshing.BoundaryLayerRoute.PROVIDER,
                phx.meshing.BoundaryLayerCornerPolicy.SMOOTH,
                schedule,
                2.0e-2,
            ),
        }
    sections = ("quality", "conservation", "resources")
    return {
        "status": "passed",
        "source": {
            "model_id": source.model_id,
            "source_id": source.report.source_id,
            "source_revision": source.report.source_revision,
            "faces": len(source.face_ids),
            "schedule_id": schedule.schedule_id,
        },
        "target": {name: record["target"] for name, record in routes.items()},
        "runtime": {
            **_runtime_identity(),
            "provider_version": routes["advancing"]["provider"]["version"],
        },
        "quality": {
            "before": {
                name: record["quality"]["before"] for name, record in routes.items()
            },
            "after": {
                name: record["quality"]["after"] for name, record in routes.items()
            },
        },
        "conservation": {name: record["conservation"] for name, record in routes.items()},
        "resources": {name: record["resources"] for name, record in routes.items()},
        "routes": {
            name: {key: value for key, value in record.items() if key not in sections}
            for name, record in routes.items()
        },
    }


_MESHCORE_DETAIL = "Set PHYDRAX_MESHCORE_LIBRARY or install phydrax[meshcore]."


def _unit_lattice_cells(
    cells_per_axis: int, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Unit square/cube lattice points and quadrilateral/hexahedron corners."""
    from tools.meshing_benchmarks import _HEXAHEDRON_CORNERS

    count = cells_per_axis
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(*(axis,) * dimension, indexing="ij"), axis=-1).reshape(
        (-1, dimension)
    )
    index = np.arange((count + 1) ** dimension, dtype=np.int32).reshape(
        (count + 1,) * dimension
    )
    offsets = np.unique(_HEXAHEDRON_CORNERS[:, :dimension], axis=0, return_index=True)[1]
    corners = _HEXAHEDRON_CORNERS[np.sort(offsets), :dimension]
    cells = np.stack(
        [
            index[tuple(slice(start, start + count) for start in corner)].reshape((-1,))
            for corner in corners
        ],
        axis=1,
    )
    return points, cells


def _jittered_cube_tetrahedra(
    cells_per_axis: int, seed: int, /
) -> phx.discretization.CellMesh:
    """Kuhn tetrahedra of the unit cube with interior vertices jittered in place."""
    from tools.meshing_benchmarks import _kuhn_tetrahedra

    points, tetrahedra = _kuhn_tetrahedra(cells_per_axis)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    jitter = np.random.default_rng(seed).uniform(-0.1, 0.1, points.shape)
    points[interior] += jitter[interior] / cells_per_axis
    return phx.discretization.CellMesh.from_tetrahedra(points, tetrahedra)


def _supermesh_case(
    source: phx.discretization.CellMesh, target: phx.discretization.CellMesh, /
) -> dict[str, object]:
    """Certify one common refinement and recount its measure and moment ledgers."""
    contract = _contract()
    source_result, source_certify = _measured(
        lambda: phx.meshing.certify_cell_mesh(source, contract)
    )
    target_result, target_certify = _measured(
        lambda: phx.meshing.certify_cell_mesh(target, contract)
    )
    policy = phx.geometry.CommonRefinementPolicy(overlap_simplices=True)
    refinement, refinement_record = _measured(
        lambda: phx.geometry.prepare_common_refinement(source, target, policy=policy)
    )
    # ty: ignore[unresolved-attribute]
    evidence = refinement.evidence
    # ty: ignore[unresolved-attribute]
    if not refinement.succeeded:
        raise RuntimeError(
            # ty: ignore[unresolved-attribute]
            f"Common refinement failed: {refinement.status.name}: {evidence.reason}."
        )
    # ty: ignore[unresolved-attribute]
    dimension = refinement.dimension
    # ty: ignore[unresolved-attribute]
    volumes = np.asarray(refinement.volumes, dtype=np.float64)
    # ty: ignore[unresolved-attribute]
    moments = np.asarray(refinement.first_moments, dtype=np.float64)
    ledgers = {}
    for role, cells, measures, first in (
        (
            "source",
            # ty: ignore[unresolved-attribute]
            refinement.source_cells,
            # ty: ignore[unresolved-attribute]
            refinement.source_measures,
            # ty: ignore[unresolved-attribute]
            refinement.source_first_moments,
        ),
        (
            "target",
            # ty: ignore[unresolved-attribute]
            refinement.target_cells,
            # ty: ignore[unresolved-attribute]
            refinement.target_measures,
            # ty: ignore[unresolved-attribute]
            refinement.target_first_moments,
        ),
    ):
        cells = np.asarray(cells, dtype=np.int64)
        measures = np.asarray(measures, dtype=np.float64)
        first = np.asarray(first, dtype=np.float64)
        covered = np.zeros_like(measures)
        np.add.at(covered, cells, volumes)
        covered_moments = np.zeros_like(first)
        np.add.at(covered_moments, cells, moments)
        total = float(np.sum(measures))
        ledgers[role] = {
            "measure": total,
            "total_measure_relative_defect": abs(float(np.sum(volumes)) - total) / total,
            "maximum_cell_measure_relative_defect": float(
                np.max(np.abs(covered - measures) / measures)
            ),
            # Cell centroid integrals: the overlap first moments of each cell sum
            # to the cell's own first moment.
            "maximum_cell_first_moment_relative_defect": float(
                np.max(np.abs(covered_moments - first)) / np.max(np.abs(first))
            ),
        }
    # The unit square/cube has measure one and centroid (1/2, ..., 1/2).
    domain = {
        "measure_defect": abs(float(np.sum(volumes)) - 1.0),
        "first_moment_defect": float(np.max(np.abs(np.sum(moments, axis=0) - 0.5))),
    }
    # ty: ignore[unresolved-attribute]
    offsets = np.asarray(refinement.simplex_offsets, dtype=np.int64)
    # ty: ignore[unresolved-attribute]
    simplices = np.asarray(refinement.simplices, dtype=np.float64)
    signed = np.linalg.det(simplices[:, 1:] - simplices[:, :1]) / np.prod(
        np.arange(1, dimension + 1)
    )
    partition = np.zeros_like(volumes)
    np.add.at(partition, np.repeat(np.arange(volumes.size), np.diff(offsets)), signed)
    partition_defect = float(np.max(np.abs(partition - volumes) / volumes))
    defects = (
        *(
            value
            for ledger in ledgers.values()
            for key, value in ledger.items()
            if key != "measure"
        ),
        *domain.values(),
        partition_defect,
        evidence.maximum_relative_source_defect,
        evidence.maximum_relative_target_defect,
    )
    if (
        max(defects) > 1.0e-12
        or evidence.source_gap_count
        or evidence.target_gap_count
        or evidence.source_double_count
        or evidence.target_double_count
        or evidence.uncertain_predicate_count
        or evidence.invalid_cell_count
        or evidence.intersection_failure_count
        # ty: ignore[unresolved-attribute]
        or not (source_result.audit.passed and target_result.audit.passed)
    ):
        raise RuntimeError(
            f"Common refinement ledgers failed: {ledgers}, {domain}, "
            f"partition {partition_defect}, evidence {evidence.reason}."
        )
    return {
        # ty: ignore[invalid-argument-type]
        "source": {**_result_identity(source_result), "mesh_id": source.mesh_id},
        # ty: ignore[invalid-argument-type]
        "target": {**_result_identity(target_result), "mesh_id": target.mesh_id},
        "identities": {
            # ty: ignore[unresolved-attribute]
            "refinement_id": refinement.refinement_id,
            "evidence_id": evidence.evidence_id,
            "policy_id": policy.policy_id,
            # ty: ignore[unresolved-attribute]
            "source_topology_id": refinement.source_topology_id,
            # ty: ignore[unresolved-attribute]
            "target_topology_id": refinement.target_topology_id,
        },
        "quality": {
            # ty: ignore[unresolved-attribute]
            "source": _quality_summary(source_result.quality),
            # ty: ignore[unresolved-attribute]
            "target": _quality_summary(target_result.quality),
        },
        "coverage": {
            # ty: ignore[unresolved-attribute]
            "status": refinement.status.name,
            "reason": evidence.reason,
            "maximum_relative_source_defect": evidence.maximum_relative_source_defect,
            "maximum_relative_target_defect": evidence.maximum_relative_target_defect,
            "source_gap_count": evidence.source_gap_count,
            "target_gap_count": evidence.target_gap_count,
            "source_double_count": evidence.source_double_count,
            "target_double_count": evidence.target_double_count,
        },
        "conservation": {
            **ledgers,
            "unit_domain": domain,
            "simplex_partition_maximum_relative_defect": partition_defect,
        },
        "pairs": {
            "candidate": evidence.candidate_pair_count,
            "accepted": evidence.accepted_pair_count,
            "piece_pairs": evidence.piece_pair_count,
            # ty: ignore[unresolved-attribute]
            "entries": refinement.entry_count,
            "overlap_simplices": simplices.shape[0],
        },
        "predicates": {
            "mode": policy.predicate_mode.value,
            "uncertain": evidence.uncertain_predicate_count,
            "invalid_cells": evidence.invalid_cell_count,
            "intersection_failures": evidence.intersection_failure_count,
        },
        "resources": {
            "source_certify": source_certify,
            "target_certify": target_certify,
            "common_refinement": refinement_record,
            "retained_bytes": evidence.retained_bytes,
            "working_bytes": evidence.working_bytes,
        },
    }


def qualify_supermesh() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency("phydrax-meshcore", _MESHCORE_DETAIL)
    from tools.meshing_benchmarks import _jittered_unit_square_triangles

    square_points, quadrilaterals = _unit_lattice_cells(5, 2)
    cube_points, hexahedra = _unit_lattice_cells(2, 3)
    meshes = {
        "triangles-quadrilaterals-2d": (
            _jittered_unit_square_triangles(6, 1),
            phx.discretization.CellMesh(
                square_points,
                (phx.discretization.CellBlock("quads", "quadrilateral", quadrilaterals),),
            ),
        ),
        "triangles-triangles-2d": (
            _jittered_unit_square_triangles(6, 1),
            _jittered_unit_square_triangles(5, 2),
        ),
        "tetrahedra-tetrahedra-3d": (
            _jittered_cube_tetrahedra(3, 3),
            _jittered_cube_tetrahedra(2, 4),
        ),
        "hexahedra-tetrahedra-3d": (
            phx.discretization.CellMesh(
                cube_points,
                (phx.discretization.CellBlock("hexahedra", "hexahedron", hexahedra),),
            ),
            _jittered_cube_tetrahedra(3, 3),
        ),
    }
    cases = {name: _supermesh_case(*pair) for name, pair in meshes.items()}
    sections = ("source", "target", "quality", "conservation", "resources")
    return {
        "status": "passed",
        "source": {name: case["source"] for name, case in cases.items()},
        "target": {name: case["target"] for name, case in cases.items()},
        "runtime": _runtime_identity(),
        "quality": {
            # ty: ignore[not-subscriptable]
            "before": {name: case["quality"]["source"] for name, case in cases.items()},
            # ty: ignore[not-subscriptable]
            "after": {name: case["quality"]["target"] for name, case in cases.items()},
        },
        "conservation": {name: case["conservation"] for name, case in cases.items()},
        "resources": {name: case["resources"] for name, case in cases.items()},
        "cases": {
            name: {key: value for key, value in case.items() if key not in sections}
            for name, case in cases.items()
        },
    }


def _remap_cell_averages(discretization: Any, function: Any, /) -> np.ndarray:
    """Quadrature cell averages of ``function`` on one FV discretization."""
    weights = np.where(
        np.asarray(discretization.cell_quadrature_valid),
        np.asarray(discretization.cell_quadrature_weights),
        0.0,
    )
    values = function(np.asarray(discretization.cell_quadrature_points))
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


def _remap_linear(points: np.ndarray, /) -> np.ndarray:
    return 0.5 + 2.0 * points[..., 0] - 3.0 * points[..., 1]


def qualify_remap_p1() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency("phydrax-meshcore", _MESHCORE_DETAIL)
    from tools.meshing_benchmarks import _jittered_unit_square_triangles, _remap_field

    discretization = phx.discretization
    contract = _contract()
    points, quadrilaterals = _unit_lattice_cells(6, 2)
    triangles = _jittered_unit_square_triangles(5, 3)
    source, source_record = _measured(
        lambda: discretization.UnstructuredFiniteVolumePlan(
            points, quadrilaterals=quadrilaterals
        ).prepare()
    )
    target, target_record = _measured(
        lambda: discretization.UnstructuredFiniteVolumePlan(
            np.asarray(triangles.coordinates),
            triangles=np.asarray(triangles.blocks[0].vertices),
        ).prepare()
    )
    # ty: ignore[unresolved-attribute]
    source_result = phx.meshing.certify_cell_mesh(source.mesh, contract)
    # ty: ignore[unresolved-attribute]
    target_result = phx.meshing.certify_cell_mesh(target.mesh, contract)
    prepared, refinement_record = _measured(
        lambda: discretization.prepare_unstructured_conservative_remap(
            # ty: ignore[invalid-argument-type]
            source,
            # ty: ignore[invalid-argument-type]
            target,
            provenance="qualification-remap-p1",
        )
    )
    # ty: ignore[unresolved-attribute]
    if not prepared.succeeded:
        # ty: ignore[unresolved-attribute]
        raise RuntimeError(f"P0 remap preparation failed: {prepared.reason}.")
    limited_plan, limited_record = _measured(
        lambda: discretization.UnstructuredSecondOrderRemapPlan(
            # ty: ignore[unresolved-attribute]
            prepared.plan,
            # ty: ignore[unresolved-attribute]
            prepared.refinement,
            # ty: ignore[invalid-argument-type]
            source,
        )
    )
    unlimited_plan, unlimited_record = _measured(
        lambda: discretization.UnstructuredSecondOrderRemapPlan(
            # ty: ignore[unresolved-attribute]
            prepared.plan,
            # ty: ignore[unresolved-attribute]
            prepared.refinement,
            # ty: ignore[invalid-argument-type]
            source,
            limiter=discretization.UnstructuredRemapLimiter.NONE,
        )
    )
    # ty: ignore[unresolved-attribute]
    source_volumes = np.asarray(source.cell_volumes, dtype=np.float64)
    # ty: ignore[unresolved-attribute]
    target_volumes = np.asarray(target.cell_volumes, dtype=np.float64)
    smooth = jnp.asarray(_remap_cell_averages(source, _remap_field))
    smooth_exact = _remap_cell_averages(target, _remap_field)
    linear = jnp.asarray(_remap_cell_averages(source, _remap_linear))
    linear_exact = _remap_cell_averages(target, _remap_linear)
    # ty: ignore[unresolved-attribute]
    centers = np.asarray(source.cell_centers)
    step = jnp.asarray(np.where(centers[:, 0] + 0.3 * centers[:, 1] > 0.55, 2.0, -1.0))
    constant = jnp.ones_like(smooth)

    def relative_l1(values: Any, exact: Any) -> Any:
        return float(
            np.sum(target_volumes * np.abs(np.asarray(values) - exact))
            / np.sum(target_volumes * np.abs(exact))
        )

    def mass_residual(source_values: Any, target_values: Any) -> Any:
        return abs(
            float(np.sum(target_volumes * np.asarray(target_values)))
            - float(np.sum(source_volumes * np.asarray(source_values)))
        ) / float(np.sum(source_volumes * np.abs(np.asarray(source_values))))

    p0, p0_record = _measured(
        lambda: jax.block_until_ready(
            {
                # ty: ignore[unresolved-attribute]
                name: prepared.plan.apply(values)
                for name, values in (
                    ("constant", constant),
                    ("smooth", smooth),
                    ("linear", linear),
                    ("step", step),
                )
            }
        )
    )
    p1, p1_record = _measured(
        lambda: jax.block_until_ready(
            {
                # ty: ignore[unresolved-attribute]
                name: limited_plan.apply(values)
                for name, values in (
                    ("constant", constant),
                    ("smooth", smooth),
                    ("linear", linear),
                    ("step", step),
                )
            }
        )
    )
    # ty: ignore[unresolved-attribute]
    unlimited_linear = unlimited_plan.apply(linear)
    # ty: ignore[not-subscriptable, unresolved-attribute]
    p0_defect = prepared.plan.conservation_defect(smooth, p0["smooth"])
    fv = {
        "p0": {
            # ty: ignore[not-subscriptable]
            "constant_error": float(np.max(np.abs(np.asarray(p0["constant"]) - 1.0))),
            # ty: ignore[not-subscriptable]
            "smooth_mass_residual": mass_residual(smooth, p0["smooth"]),
            # ty: ignore[not-subscriptable]
            "step_mass_residual": mass_residual(step, p0["step"]),
            "plan_conservation_defect": float(np.max(np.abs(np.asarray(p0_defect)))),
            # ty: ignore[not-subscriptable]
            "smooth_relative_l1_error": relative_l1(p0["smooth"], smooth_exact),
            # ty: ignore[not-subscriptable]
            "linear_relative_l1_error": relative_l1(p0["linear"], linear_exact),
            # ty: ignore[not-subscriptable]
            "step_bounds": [float(jnp.min(p0["step"])), float(jnp.max(p0["step"]))],
        },
        "limited_p1": {
            "constant_error": float(
                # ty: ignore[not-subscriptable]
                np.max(np.abs(np.asarray(p1["constant"].values) - 1.0))
            ),
            # ty: ignore[not-subscriptable]
            "smooth_mass_residual": mass_residual(smooth, p1["smooth"].values),
            # ty: ignore[not-subscriptable]
            "step_mass_residual": mass_residual(step, p1["step"].values),
            "step_conservation_residual_after": float(
                # ty: ignore[not-subscriptable]
                np.abs(np.asarray(p1["step"].conservation_residual_after))
            ),
            # ty: ignore[not-subscriptable]
            "step_restored": bool(p1["step"].restored),
            # ty: ignore[not-subscriptable]
            "smooth_relative_l1_error": relative_l1(p1["smooth"].values, smooth_exact),
            # ty: ignore[not-subscriptable]
            "linear_relative_l1_error": relative_l1(p1["linear"].values, linear_exact),
            "step_bounds": [
                # ty: ignore[not-subscriptable]
                float(jnp.min(p1["step"].values)),
                # ty: ignore[not-subscriptable]
                float(jnp.max(p1["step"].values)),
            ],
            # ty: ignore[not-subscriptable]
            "step_limited_cells": int(p1["step"].limited_count),
            # ty: ignore[not-subscriptable]
            "step_minimum_limiter_factor": float(p1["step"].minimum_limiter_factor),
            # ty: ignore[unresolved-attribute]
            "gradient_minimum_rank": int(np.min(np.asarray(limited_plan.gradient_rank))),
            "gradient_maximum_condition": float(
                # ty: ignore[unresolved-attribute]
                np.max(np.asarray(limited_plan.gradient_condition))
            ),
        },
        "unlimited_p1": {
            "linear_maximum_error": float(
                np.max(np.abs(np.asarray(unlimited_linear.values) - linear_exact))
            ),
            "linear_mass_residual": mass_residual(linear, unlimited_linear.values),
        },
    }
    step_bounds = fv["limited_p1"]["step_bounds"]
    if (
        max(
            fv["p0"]["constant_error"],
            fv["p0"]["smooth_mass_residual"],
            fv["p0"]["step_mass_residual"],
            fv["limited_p1"]["constant_error"],
            fv["limited_p1"]["smooth_mass_residual"],
            fv["limited_p1"]["step_mass_residual"],
            fv["unlimited_p1"]["linear_maximum_error"],
            fv["unlimited_p1"]["linear_mass_residual"],
        )
        > 1.0e-12
        or step_bounds[0] < -1.0 - 1.0e-12
        or step_bounds[1] > 2.0 + 1.0e-12
        or not fv["limited_p1"]["step_restored"]
        or fv["limited_p1"]["step_limited_cells"] == 0
        or fv["limited_p1"]["smooth_relative_l1_error"]
        >= fv["p0"]["smooth_relative_l1_error"]
    ):
        raise RuntimeError(f"Finite-volume remap qualification failed: {fv}.")

    # FEM L2 projection of a continuous P1 field between jittered triangle meshes.
    fem_source_mesh = _jittered_unit_square_triangles(7, 1)
    fem_target_mesh = _jittered_unit_square_triangles(5, 2)

    def p1_space(mesh: Any) -> Any:
        return discretization.FiniteElementPlan(
            mesh,
            discretization.FiniteElementFieldSpec(
                "u", discretization.lagrange_element("triangle", 1)
            ),
        ).prepare()

    fem_source, fem_target = p1_space(fem_source_mesh), p1_space(fem_target_mesh)
    fem_refinement, fem_refinement_record = _measured(
        lambda: phx.geometry.prepare_common_refinement(
            fem_source_mesh,
            fem_target_mesh,
            policy=phx.geometry.CommonRefinementPolicy(overlap_simplices=True),
        )
    )
    # ty: ignore[unresolved-attribute]
    if not fem_refinement.succeeded:
        # ty: ignore[unresolved-attribute]
        raise RuntimeError(f"L2 projection refinement failed: {fem_refinement.status}.")
    prepared_target, target_record = _measured(
        lambda: discretization.prepare_l2_projection_target(fem_target, field_name="u")
    )
    transfer, transfer_record = _measured(
        lambda: discretization.prepare_l2_projection_transfer(
            fem_source,
            # ty: ignore[invalid-argument-type]
            prepared_target,
            # ty: ignore[invalid-argument-type]
            fem_refinement,
            field_name="u",
        )
    )
    source_dofs = np.asarray(fem_source.dof_maps[0].dof_coordinates)
    target_dofs = np.asarray(fem_target.dof_maps[0].dof_coordinates)
    source_weights = np.asarray(
        fem_source.mass.mv(jnp.ones((fem_source.dof_maps[0].global_dof_count,)))
    )
    target_weights = np.asarray(
        fem_target.mass.mv(jnp.ones((fem_target.dof_maps[0].global_dof_count,)))
    )
    fem_smooth = _remap_field(source_dofs)
    projected, projection_record = _measured(
        lambda: jax.block_until_ready(
            (
                # ty: ignore[unresolved-attribute]
                transfer.apply(jnp.ones_like(jnp.asarray(fem_smooth))),
                # ty: ignore[unresolved-attribute]
                transfer.apply(jnp.asarray(_remap_linear(source_dofs))),
                # ty: ignore[unresolved-attribute]
                transfer.apply(jnp.asarray(fem_smooth)),
            )
        )
    )
    source_integral = float(np.dot(source_weights, fem_smooth))
    fem = {
        "flags": {
            # ty: ignore[unresolved-attribute]
            "preserves_constants": transfer.preserves_constants,
            # ty: ignore[unresolved-attribute]
            "preserves_linear": transfer.preserves_linear,
            # ty: ignore[unresolved-attribute]
            "conservative": transfer.conservative,
            # ty: ignore[unresolved-attribute]
            "positivity_preserving": transfer.positivity_preserving,
        },
        # ty: ignore[not-subscriptable]
        "constant_error": float(np.max(np.abs(np.asarray(projected[0]) - 1.0))),
        "linear_maximum_error": float(
            # ty: ignore[not-subscriptable]
            np.max(np.abs(np.asarray(projected[1]) - _remap_linear(target_dofs)))
        ),
        "smooth_integral_relative_residual": abs(
            # ty: ignore[not-subscriptable]
            float(np.dot(target_weights, np.asarray(projected[2]))) - source_integral
        )
        / abs(source_integral),
        "smooth_nodal_maximum_error": float(
            # ty: ignore[not-subscriptable]
            np.max(np.abs(np.asarray(projected[2]) - _remap_field(target_dofs)))
        ),
        # ty: ignore[unresolved-attribute]
        "target_mass_condition": float(prepared_target.mass_condition.value),
        # ty: ignore[unresolved-attribute]
        "factorization_status": int(prepared_target.factorization.status),
    }
    if (
        not (
            # ty: ignore[unresolved-attribute]
            transfer.preserves_constants
            # ty: ignore[unresolved-attribute]
            and transfer.preserves_linear
            # ty: ignore[unresolved-attribute]
            and transfer.conservative
        )
        # ty: ignore[unresolved-attribute]
        or transfer.positivity_preserving
        or max(
            fem["constant_error"],
            fem["linear_maximum_error"],
            fem["smooth_integral_relative_residual"],
        )
        > 1.0e-10
    ):
        raise RuntimeError(f"L2 projection qualification failed: {fem}.")
    # ty: ignore[unresolved-attribute]
    evidence = prepared.evidence
    return {
        "status": "passed",
        "source": {
            "finite_volume": {
                **_result_identity(source_result),
                # ty: ignore[unresolved-attribute]
                "prepared_id": source.prepared_id,
                # ty: ignore[unresolved-attribute]
                "geometry_id": source.geometry_id,
            },
            "finite_element": {
                "mesh_id": fem_source_mesh.mesh_id,
                "topology_id": fem_source_mesh.topology_id,
                "prepared_id": fem_source.prepared_id,
                "dofs": source_dofs.shape[0],
            },
        },
        "target": {
            "finite_volume": {
                **_result_identity(target_result),
                # ty: ignore[unresolved-attribute]
                "prepared_id": target.prepared_id,
                # ty: ignore[unresolved-attribute]
                "geometry_id": target.geometry_id,
            },
            "finite_element": {
                "mesh_id": fem_target_mesh.mesh_id,
                "topology_id": fem_target_mesh.topology_id,
                "prepared_id": fem_target.prepared_id,
                "dofs": target_dofs.shape[0],
            },
        },
        "identities": {
            # ty: ignore[unresolved-attribute]
            "refinement_id": prepared.refinement.refinement_id,
            # ty: ignore[unresolved-attribute]
            "remap_id": prepared.remap_id,
            # ty: ignore[unresolved-attribute]
            "p0_plan_id": prepared.plan.plan_id,
            # ty: ignore[unresolved-attribute]
            "limited_p1_plan_id": limited_plan.plan_id,
            # ty: ignore[unresolved-attribute]
            "unlimited_p1_plan_id": unlimited_plan.plan_id,
            # ty: ignore[unresolved-attribute]
            "l2_refinement_id": fem_refinement.refinement_id,
            # ty: ignore[unresolved-attribute]
            "l2_transfer_id": transfer.transfer_id,
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(source_result.quality),
            "after": _quality_summary(target_result.quality),
        },
        "conservation": {
            "finite_volume": fv,
            "coverage": {
                "maximum_relative_source_defect": evidence.maximum_relative_source_defect,
                "maximum_relative_target_defect": evidence.maximum_relative_target_defect,
                # ty: ignore[unresolved-attribute]
                "entries": prepared.refinement.entry_count,
            },
        },
        "transfer": {"l2_projection": fem},
        "resources": {
            "source_prepare": source_record,
            "target_prepare": target_record,
            "refinement_and_p0_plan": refinement_record,
            "limited_p1_plan": limited_record,
            "unlimited_p1_plan": unlimited_record,
            "p0_apply": p0_record,
            "limited_p1_apply": p1_record,
            "l2_refinement": fem_refinement_record,
            "l2_target_prepare": target_record,
            "l2_transfer_prepare": transfer_record,
            "l2_apply": projection_record,
            "refinement_retained_bytes": evidence.retained_bytes,
            "refinement_working_bytes": evidence.working_bytes,
            # ty: ignore[unresolved-attribute]
            "l2_refinement_retained_bytes": fem_refinement.evidence.retained_bytes,
            # ty: ignore[unresolved-attribute]
            "l2_refinement_working_bytes": fem_refinement.evidence.working_bytes,
        },
    }


def _bisection_meshes() -> dict[str, phx.discretization.CellMesh]:
    """A 4 x 4 right-triangle lattice of the unit square and 2^3 Kuhn cubes."""
    from tools.meshing_benchmarks import _kuhn_tetrahedra, _planar_triangles

    axis = np.linspace(0.0, 1.0, 5)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    spatial, tetrahedra = _kuhn_tetrahedra(2)
    return {
        "triangles": phx.discretization.CellMesh.from_triangles(
            points, _planar_triangles(4)
        ),
        "tetrahedra": phx.discretization.CellMesh.from_tetrahedra(spatial, tetrahedra),
    }


def _bisection_cell_ids(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    return np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)


def _bisection_corner_cells(mesh: phx.discretization.CellMesh, /) -> np.ndarray:
    """Cells touching the vertex nearest the origin (repeated local refinement)."""
    corner = np.argmin(np.linalg.norm(np.asarray(mesh.coordinates), axis=1))
    touching = np.any(np.asarray(mesh.blocks[0].vertices) == corner, axis=1)
    return np.sort(_bisection_cell_ids(mesh)[touching])


def _bisection_adapt(
    route: phx.meshing.MeshAdaptationRoute,
    source: phx.meshing.CellMeshingResult,
    refine: np.ndarray,
    coarsen: np.ndarray,
    hierarchy: Any,
    /,
    *,
    device_policy: phx.discretization.AdaptiveSimplexPolicy | None = None,
) -> tuple[phx.meshing.MeshAdaptationResult, dict[str, dict[str, float]]]:
    request = phx.meshing.MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    policy = phx.meshing.MeshAdaptationPolicy(route, device_policy=device_policy)
    prepared, preparation = _measured(
        lambda: phx.meshing.prepare_mesh_adaptation(source, request, policy=policy)
    )
    # ty: ignore[invalid-argument-type]
    result, execution = _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared))
    # ty: ignore[unresolved-attribute]
    if not result.target.audit.passed:
        raise RuntimeError(f"{route.name} produced a target that failed its audit.")
    # ty: ignore[invalid-return-type]
    return result, {"prepare": preparation, "execute": execution}


def _bisection_retained_bytes(tree: Any, /) -> int:
    """Bytes of the distinct array leaves an adaptation artifact retains."""
    leaves = {
        id(leaf): leaf
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, (jax.Array, np.ndarray))
    }
    return sum(leaf.nbytes for leaf in leaves.values())


def _bisection_lineage(result: phx.meshing.MeshAdaptationResult, /) -> dict[str, object]:
    """Every entity is an identity, a lineage relation, or created/deleted."""
    source, target = result.source.mesh, result.target.mesh
    lineage = result.lineage
    # ty: ignore[unresolved-attribute]
    dimensions = tuple(record.dimension for record in lineage.entities)
    width = np.asarray(source.blocks[0].vertices).shape[1]
    complete = (
        dimensions == tuple(range(width))
        # ty: ignore[unresolved-attribute]
        and lineage.source_topology_id == source.topology_id
        # ty: ignore[unresolved-attribute]
        and lineage.target_topology_id == target.topology_id
    )
    relations = {}
    # ty: ignore[unresolved-attribute]
    for record in lineage.entities:
        before = np.asarray(source.entity_set(record.dimension).entity_ids)
        after = np.asarray(target.entity_set(record.dimension).entity_ids)
        created = np.asarray(record.created_target_ids)
        deleted = np.asarray(record.deleted_source_ids)
        covered_after = np.concatenate(
            (before, np.asarray(record.target_global_ids), created)
        )
        covered_before = np.concatenate(
            (after, np.asarray(record.source_global_ids), deleted)
        )
        complete = complete and bool(
            np.all(np.isin(after, covered_after))
            and np.all(np.isin(before, covered_before))
            and np.all(np.isin(created, after) & ~np.isin(created, before))
            and np.all(np.isin(deleted, before) & ~np.isin(deleted, after))
        )
        relations[f"dimension_{record.dimension}"] = {
            "relations": record.relation_kinds.shape[0],
            "created": created.shape[0],
            "deleted": deleted.shape[0],
        }
    if not complete:
        raise RuntimeError("Bisection lineage does not account for every entity.")
    # ty: ignore[unresolved-attribute]
    return {"lineage_id": lineage.lineage_id, "complete": complete, **relations}


def _bisection_evidence(evidence: Any, /) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "requested_refinements": evidence.requested_refinements,
        "accepted_refinements": evidence.accepted_refinements,
        "rejected_refinements": evidence.rejected_refinement_ids.shape[0],
        "admissibility_tests": evidence.admissibility_tests,
        "bisections": evidence.bisections,
        "closure_iterations": evidence.closure_iterations,
        "created_vertices": evidence.created_vertices,
        "maximum_generation": evidence.maximum_generation,
        "initially_compatible": evidence.initially_compatible,
        "incompatible_facets": evidence.incompatible_facets,
        "uniform_refinement_applied": evidence.uniform_refinement_applied,
        "requested_coarsenings": evidence.requested_coarsenings,
        "coarsened_vertices": evidence.coarsened_vertices,
        "coarsening_passes": evidence.coarsening_passes,
        "restored_cells": evidence.restored_cells,
        "rejected_coarsenings": evidence.rejected_coarsening_ids.shape[0],
    }


def _bisection_p1_integral(mesh: phx.discretization.CellMesh, values: Any, /) -> float:
    """Exact integral of the P1 interpolant of vertex values."""
    cells = np.asarray(mesh.blocks[0].vertices)
    corners = np.asarray(mesh.coordinates)[cells]
    edges = corners[:, 1:] - corners[:, :1]
    width = cells.shape[1]
    measures = np.abs(np.linalg.det(edges)) / (2.0 if width == 3 else 6.0)
    return float(np.sum(measures * np.mean(np.asarray(values)[cells], axis=1)))


def _bisection_transfer(result: phx.meshing.MeshAdaptationResult, /) -> dict[str, object]:
    """Claims, P1 linear reproduction, and P1 integral change of one transfer."""
    transfer = result.transfer
    source = np.asarray(result.source.mesh.coordinates)
    target = np.asarray(result.target.mesh.coordinates)
    slope = np.linspace(0.5, 1.5, source.shape[1])
    # ty: ignore[unresolved-attribute]
    linear = np.asarray(transfer.apply(source @ slope + 0.25))
    reproduction = float(np.max(np.abs(linear - (target @ slope + 0.25))))
    # ty: ignore[unresolved-attribute]
    constant = np.asarray(transfer.apply(np.ones((source.shape[0],))))
    smooth = np.sin(3.0 * source[:, 0]) * np.cos(2.0 * source[:, -1])
    # ty: ignore[unresolved-attribute]
    moved = np.asarray(transfer.apply(smooth))
    residual = _bisection_p1_integral(result.target.mesh, moved) - _bisection_p1_integral(
        result.source.mesh, smooth
    )
    if (
        # ty: ignore[unresolved-attribute]
        not transfer.preserves_constants
        # ty: ignore[unresolved-attribute]
        or not transfer.preserves_linear
        or reproduction > 1.0e-12
        or float(np.max(np.abs(constant - 1.0))) > 1.0e-14
    ):
        raise RuntimeError("The bisection transfer does not reproduce P1 fields.")
    return {
        # ty: ignore[unresolved-attribute]
        "transfer_id": transfer.transfer_id,
        # ty: ignore[unresolved-attribute]
        "preserves_constants": transfer.preserves_constants,
        # ty: ignore[unresolved-attribute]
        "preserves_linear": transfer.preserves_linear,
        # ty: ignore[unresolved-attribute]
        "conservative": transfer.conservative,
        # ty: ignore[unresolved-attribute]
        "positivity_preserving": transfer.positivity_preserving,
        "linear_reproduction_error": reproduction,
        "constant_reproduction_error": float(np.max(np.abs(constant - 1.0))),
        "p1_integral_residual": residual,
        "retained_bytes": _bisection_retained_bytes(transfer),
    }


def _bisection_shape_classes(
    source: phx.meshing.CellMeshingResult, rounds: int, cycles: int, /
) -> dict[str, object]:
    """Uniform class bound, then repeated corner refinement against it.

    Newest-vertex (Maubach) bisection generates finitely many similarity
    classes; the first uniform generations visit them all, so locally refined
    meshes never fall below the uniform bound.
    """
    route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    angle = source.quality.minimum_angle
    mean_ratio = source.quality.minimum_mean_ratio
    current, hierarchy = source, None
    for _ in range(rounds):
        cells = np.sort(_bisection_cell_ids(current.mesh))
        # ty: ignore[invalid-argument-type]
        result, _ = _bisection_adapt(route, current, cells, (), hierarchy)
        current, hierarchy = result.target, result.hierarchy
        angle = min(angle, current.quality.minimum_angle)
        mean_ratio = min(mean_ratio, current.quality.minimum_mean_ratio)
    current, hierarchy = source, None
    angles, ratios = [], []
    for _ in range(cycles):
        marks = _bisection_corner_cells(current.mesh)
        # ty: ignore[invalid-argument-type]
        result, _ = _bisection_adapt(route, current, marks, (), hierarchy)
        current, hierarchy = result.target, result.hierarchy
        angles.append(current.quality.minimum_angle)
        ratios.append(current.quality.minimum_mean_ratio)
    bounded = min(angles) >= angle * (1.0 - 1.0e-9) and min(ratios) >= 0.999 * mean_ratio
    # ty: ignore[unresolved-attribute]
    if not bounded or result.evidence.maximum_generation < cycles:
        raise RuntimeError("Repeated bisection left the uniform shape-class bound.")
    return {
        "uniform_rounds": rounds,
        "uniform_minimum_angle_bound": angle,
        "uniform_minimum_mean_ratio_bound": mean_ratio,
        "corner_cycles": cycles,
        "corner_minimum_angles": angles,
        "corner_minimum_mean_ratios": ratios,
        "minimum_angle_ratio_to_source": min(angles) / source.quality.minimum_angle,
        "minimum_mean_ratio_ratio_to_source": min(ratios)
        / source.quality.minimum_mean_ratio,
        # ty: ignore[unresolved-attribute]
        "maximum_generation": result.evidence.maximum_generation,
        "final_cells": current.mesh.blocks[0].cell_count,
        "bounded": bounded,
    }


def _bisection_round_trip(
    mesh: phx.discretization.CellMesh, rounds: int, cycles: int, /
) -> dict[str, object]:
    route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    source, certification = _measured(
        lambda: phx.meshing.certify_cell_mesh(mesh, _contract())
    )
    marks = np.sort(_bisection_cell_ids(mesh))[::3]
    # ty: ignore[invalid-argument-type]
    first, first_stages = _bisection_adapt(route, source, marks, (), None)
    corner = _bisection_corner_cells(first.target.mesh)
    second, second_stages = _bisection_adapt(
        route,
        first.target,
        corner,
        # ty: ignore[invalid-argument-type]
        (),
        first.hierarchy,
    )
    everything = np.sort(_bisection_cell_ids(second.target.mesh))
    restored, restore_stages = _bisection_adapt(
        route,
        second.target,
        # ty: ignore[invalid-argument-type]
        (),
        everything,
        second.hierarchy,
    )
    back = restored.target.mesh
    identical = (
        back.topology_id == mesh.topology_id
        and np.array_equal(np.asarray(back.coordinates), np.asarray(mesh.coordinates))
        and np.array_equal(
            np.asarray(back.vertex_global_ids), np.asarray(mesh.vertex_global_ids)
        )
        and np.array_equal(_bisection_cell_ids(back), _bisection_cell_ids(mesh))
    )
    statuses = (first.status, second.status, restored.status)
    complete = phx.meshing.MeshAdaptationStatus.COMPLETE
    if not identical or first.status is not complete or second.status is not complete:
        raise RuntimeError("The bisection refine/coarsen round trip is not exact.")
    refinement = _bisection_transfer(first)
    # ty: ignore[invalid-argument-type]
    if not refinement["conservative"] or abs(refinement["p1_integral_residual"]) > 1e-13:
        raise RuntimeError("Refinement transfer does not conserve P1 integrals.")
    return {
        # ty: ignore[invalid-argument-type]
        "source": _result_identity(source),
        "refined": _result_identity(second.target),
        "target": _result_identity(restored.target),
        "statuses": [status.value for status in statuses],
        "restores_source_topology": identical,
        "quality": {
            # ty: ignore[unresolved-attribute]
            "before": _quality_summary(source.quality),
            "refined": _quality_summary(second.target.quality),
            "after": _quality_summary(restored.target.quality),
        },
        "transfer": {
            "refinement": refinement,
            "local_refinement": _bisection_transfer(second),
            "coarsening": _bisection_transfer(restored),
        },
        "lineage": {
            "refinement": _bisection_lineage(first),
            "local_refinement": _bisection_lineage(second),
            "coarsening": _bisection_lineage(restored),
        },
        "evidence": {
            "refinement": _bisection_evidence(first.evidence),
            "local_refinement": _bisection_evidence(second.evidence),
            "coarsening": _bisection_evidence(restored.evidence),
        },
        # ty: ignore[unresolved-attribute]
        "hierarchy_id": second.hierarchy.hierarchy_id,
        # ty: ignore[invalid-argument-type]
        "shape_classes": _bisection_shape_classes(source, rounds, cycles),
        "resources": {
            "certify": certification,
            "refinement": first_stages,
            "local_refinement": second_stages,
            "coarsening": restore_stages,
            "hierarchy_retained_bytes": _bisection_retained_bytes(second.hierarchy),
            "target_mesh_retained_bytes": _bisection_retained_bytes(second.target.mesh),
        },
    }


def qualify_bisection() -> dict[str, object]:
    """NATIVE_BISECTION refine/coarsen round trips of triangles and tetrahedra."""
    meshes = _bisection_meshes()
    dimensions = {
        "triangles": _bisection_round_trip(meshes["triangles"], 4, 12),
        "tetrahedra": _bisection_round_trip(meshes["tetrahedra"], 3, 6),
    }
    return {
        "status": "passed",
        "route": phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION.value,
        "source": {name: record["source"] for name, record in dimensions.items()},
        "target": {name: record["target"] for name, record in dimensions.items()},
        "quality": {
            name: {
                # ty: ignore[not-subscriptable]
                "before": record["quality"]["before"],
                # ty: ignore[not-subscriptable]
                "after": record["quality"]["after"],
            }
            for name, record in dimensions.items()
        },
        "transfer": {name: record["transfer"] for name, record in dimensions.items()},
        "conservation": {
            name: {
                stage: values["p1_integral_residual"]
                # ty: ignore[unresolved-attribute]
                for stage, values in record["transfer"].items()
            }
            for name, record in dimensions.items()
        },
        "resources": {name: record["resources"] for name, record in dimensions.items()},
        "dimensions": dimensions,
        "runtime": _runtime_identity(),
    }


def _bisection_target_bytes(
    result: phx.meshing.MeshAdaptationResult, /
) -> dict[str, object]:
    """Committed arrays and identities compared byte for byte across routes."""
    mesh = result.target.mesh
    block = mesh.blocks[0]
    return {
        "status": result.status.value,
        "topology_id": mesh.topology_id,
        "coordinates": np.asarray(mesh.coordinates).tobytes(),
        "vertex_global_ids": np.asarray(mesh.vertex_global_ids).tobytes(),
        "cells": np.asarray(block.vertices).tobytes(),
        "cell_global_ids": np.asarray(block.global_ids).tobytes(),
        # ty: ignore[unresolved-attribute]
        "lineage_id": result.lineage.lineage_id,
        # ty: ignore[unresolved-attribute]
        "hierarchy_id": result.hierarchy.hierarchy_id,
        # ty: ignore[unresolved-attribute]
        "stencil_id": result.stencil.stencil_id,
        # ty: ignore[unresolved-attribute]
        "transfer_id": result.transfer.transfer_id,
        "rejected_coarsening_ids": np.asarray(
            # ty: ignore[unresolved-attribute]
            result.evidence.rejected_coarsening_ids
        ).tobytes(),
    }


def _bisection_same_target(host: Any, device: Any, /) -> list[str]:
    first, second = _bisection_target_bytes(host), _bisection_target_bytes(device)
    differences = [key for key in first if first[key] != second[key]]
    if differences:
        raise RuntimeError(
            f"DEVICE_BISECTION differs from NATIVE_BISECTION: {differences}"
        )
    return list(first)


def _bisection_simplex_report(report: Any, /) -> dict[str, object]:
    status = phx.discretization.AdaptiveSimplexStatus(int(report.status))
    return {
        "status": int(status),
        "flags": [
            flag.name
            for flag in phx.discretization.AdaptiveSimplexStatus
            if flag & status
        ],
        "failed": bool(report.failed),
        "requested": int(report.requested),
        "accepted": int(report.accepted),
        "rejected": int(np.count_nonzero(np.asarray(report.rejected))),
        "operations": int(report.operations),
        "iterations": int(report.iterations),
        "vertices": int(report.vertices),
        "uncertain_cells": int(report.uncertain_cells),
        "invalid_cells": int(report.invalid_cells),
        "minimum_quality": float(report.minimum_quality),
    }


def _bisection_device_parity(
    mesh: phx.discretization.CellMesh,
    capacity: tuple[int, int],
    /,
) -> dict[str, object]:
    """Host and device routes for the same marks, plus one explicit device epoch."""
    host_route = phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    device_route = phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION
    device_policy = phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=capacity[0], cell_capacity=capacity[1]
    )
    source = phx.meshing.certify_cell_mesh(mesh, _contract())
    marks = np.sort(_bisection_cell_ids(mesh))[::3]
    # ty: ignore[invalid-argument-type]
    host, host_first = _bisection_adapt(host_route, source, marks, (), None)
    device, device_first = _bisection_adapt(
        device_route,
        source,
        marks,
        # ty: ignore[invalid-argument-type]
        (),
        None,
        device_policy=device_policy,
    )
    compared = _bisection_same_target(host, device)
    refine = _bisection_corner_cells(host.target.mesh)
    coarsen = np.setdiff1d(np.sort(_bisection_cell_ids(host.target.mesh))[-24:], refine)
    host_second, host_stages = _bisection_adapt(
        host_route, host.target, refine, coarsen, host.hierarchy
    )
    device_second, device_stages = _bisection_adapt(
        device_route,
        device.target,
        refine,
        coarsen,
        device.hierarchy,
        device_policy=device_policy,
    )
    _bisection_same_target(host_second, device_second)
    policy = phx.meshing.MeshAdaptationPolicy(device_route, device_policy=device_policy)
    epoch, preparation = _measured(
        lambda: phx.meshing.prepare_adaptive_simplex(
            device.target,
            policy=policy,
            # ty: ignore[invalid-argument-type]
            hierarchy=device.hierarchy,
        )
    )
    # ty: ignore[unresolved-attribute]
    layout = epoch.layout
    refined, refinement = _measured(
        lambda: jax.block_until_ready(
            phx.discretization.refine_adaptive_simplex(
                layout,
                # ty: ignore[unresolved-attribute]
                epoch.state,
                # ty: ignore[unresolved-attribute]
                epoch.cell_marks(refine),
            )
        )
    )
    coarsened, coarsening = _measured(
        lambda: jax.block_until_ready(
            phx.discretization.coarsen_adaptive_simplex(
                layout,
                # ty: ignore[unresolved-attribute]
                refined.state,
                # ty: ignore[unresolved-attribute]
                epoch.cell_marks(coarsen),
            )
        )
    )
    committed, commit = _measured(
        # ty: ignore[invalid-argument-type, unresolved-attribute]
        lambda: phx.meshing.commit_adaptive_simplex(epoch, coarsened.state)
    )
    _bisection_same_target(host_second, committed)
    reports = {
        # ty: ignore[unresolved-attribute]
        "refine": _bisection_simplex_report(refined.report),
        # ty: ignore[unresolved-attribute]
        "coarsen": _bisection_simplex_report(coarsened.report),
    }
    if any(report["failed"] for report in reports.values()):
        raise RuntimeError("The explicit device epoch refused an operation.")
    return {
        "source": _result_identity(source),
        "target": _result_identity(device_second.target),
        "statuses": {
            "first": [host.status.value, device.status.value],
            "second": [host_second.status.value, device_second.status.value],
            # ty: ignore[unresolved-attribute]
            "explicit_epoch": committed.status.value,
        },
        "byte_identical_fields": compared,
        "quality": {
            "before": _quality_summary(source.quality),
            "after": _quality_summary(device_second.target.quality),
        },
        "transfer": {
            "first": _bisection_transfer(device),
            "second": _bisection_transfer(device_second),
        },
        "lineage": _bisection_lineage(device_second),
        "evidence": {
            "host": _bisection_evidence(host_second.evidence),
            "device": _bisection_evidence(device_second.evidence),
        },
        "device_reports": reports,
        "layout": {
            "signature_id": layout.signature_id,
            "dimension": layout.dimension,
            "vertex_capacity": layout.vertex_capacity,
            "cell_capacity": layout.cell_capacity,
            "protected_edge_capacity": layout.protected_edge_capacity,
            "maximum_closure_iterations": layout.maximum_closure_iterations,
            "maximum_coarsening_passes": layout.maximum_coarsening_passes,
        },
        "resources": {
            "host_first": host_first,
            "device_first": device_first,
            "host_second": host_stages,
            "device_second": device_stages,
            "epoch_prepare": preparation,
            "epoch_refine": refinement,
            "epoch_coarsen": coarsening,
            "epoch_commit": commit,
            # ty: ignore[unresolved-attribute]
            "epoch_state_bytes": _bisection_retained_bytes(coarsened.state),
            "transfer_retained_bytes": _bisection_retained_bytes(device_second.transfer),
        },
    }


def _bisection_metric_measurements(evidence: Any, /) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "passes": evidence.passes,
        "splits": evidence.splits,
        "collapses": evidence.collapses,
        "flips": evidence.flips,
        "relocations": evidence.relocations,
        "rejected_uncertain": evidence.rejected_uncertain,
        "rejected_invalid": evidence.rejected_invalid,
        "converged": evidence.converged,
        "stalled": evidence.stalled,
        "measured_edges": evidence.measured_edges,
        "out_of_range_edges": evidence.out_of_range_edges,
        "minimum_metric_length": evidence.minimum_metric_length,
        "maximum_metric_length": evidence.maximum_metric_length,
        "unit_fraction": evidence.unit_fraction,
        "minimum_metric_quality": evidence.minimum_metric_quality,
    }


def _bisection_metric_comparison(
    mesh: phx.discretization.CellMesh, /
) -> dict[str, object]:
    """Host and device planar metric adaptation toward one isotropic size 0.2."""
    source = phx.meshing.certify_cell_mesh(mesh, _contract())
    vertices = np.asarray(source.mesh.entity_set(0).entity_ids)
    metric = phx.meshing.MeshMetricField(
        _mesh_scope(source.mesh, 0, vertices),
        np.broadcast_to(np.eye(2) / 0.2**2, (vertices.shape[0], 2, 2)),
        minimum_size=1.0e-3,
        maximum_size=10.0,
    )
    request = phx.meshing.MetricMeshAdaptation(metric)
    native_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
    )
    native, native_seconds = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(source, request, policy=native_policy)
        )
    )
    device_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_METRIC_2D,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=256, cell_capacity=512
        ),
    )
    prepared, preparation = _measured(
        lambda: phx.meshing.prepare_device_metric_adaptation(
            source, request, policy=device_policy
        )
    )
    update, adaptation = _measured(
        lambda: jax.block_until_ready(
            # ty: ignore[unresolved-attribute]
            phx.meshing.adapt_device_metric(prepared.layout, prepared.state)
        )
    )
    # ty: ignore[unresolved-attribute]
    report = jax.device_get(update.report)
    device, commit = _measured(
        # ty: ignore[invalid-argument-type, unresolved-attribute]
        lambda: phx.meshing.commit_device_metric_adaptation(prepared, update.state)
    )
    complete = phx.meshing.MeshAdaptationStatus.COMPLETE
    if (
        bool(report.failed)
        # ty: ignore[unresolved-attribute]
        or device.status is not complete
        # ty: ignore[unresolved-attribute]
        or not device.evidence.converged
        # ty: ignore[unresolved-attribute]
        or not native.target.audit.passed
        # ty: ignore[unresolved-attribute]
        or not device.target.audit.passed
    ):
        raise RuntimeError("DEVICE_METRIC_2D did not reach a certified unit mesh.")
    status = phx.discretization.AdaptiveSimplexStatus(int(report.status))
    return {
        "source": _result_identity(source),
        "target": {
            # ty: ignore[unresolved-attribute]
            "native": _result_identity(native.target),
            # ty: ignore[unresolved-attribute]
            "device": _result_identity(device.target),
        },
        # ty: ignore[unresolved-attribute]
        "statuses": {"native": native.status.value, "device": device.status.value},
        "quality": {
            "before": _quality_summary(source.quality),
            "after": {
                # ty: ignore[unresolved-attribute]
                "native": _quality_summary(native.target.quality),
                # ty: ignore[unresolved-attribute]
                "device": _quality_summary(device.target.quality),
            },
        },
        "measurements": {
            # ty: ignore[unresolved-attribute]
            "native": _bisection_metric_measurements(native.evidence),
            # ty: ignore[unresolved-attribute]
            "device": _bisection_metric_measurements(device.evidence),
        },
        "device_report": {
            "status": int(status),
            "flags": [
                flag.name
                for flag in phx.discretization.AdaptiveSimplexStatus
                if flag & status
            ],
            "passes": int(report.passes),
            "splits": int(report.splits),
            "collapses": int(report.collapses),
            "flips": int(report.flips),
            "relocations": int(report.relocations),
            "rejected_cavity": int(report.rejected_cavity),
            "unit_fraction": float(report.unit_fraction),
            "minimum_metric_quality": float(report.minimum_metric_quality),
        },
        "layout": {
            # ty: ignore[unresolved-attribute]
            "signature_id": prepared.layout.signature_id,
            # ty: ignore[unresolved-attribute]
            "vertex_capacity": prepared.layout.vertex_capacity,
            # ty: ignore[unresolved-attribute]
            "cell_capacity": prepared.layout.cell_capacity,
            # ty: ignore[unresolved-attribute]
            "record_capacity": prepared.layout.record_capacity,
            # ty: ignore[unresolved-attribute]
            "ball_width": prepared.layout.ball_width,
        },
        "transfer": {
            # ty: ignore[invalid-argument-type]
            "native": _bisection_transfer(native),
            # ty: ignore[invalid-argument-type]
            "device": _bisection_transfer(device),
        },
        "resources": {
            "native_adaptation": native_seconds,
            "device_prepare": preparation,
            "device_adapt": adaptation,
            "device_commit": commit,
            # ty: ignore[unresolved-attribute]
            "device_state_bytes": _bisection_retained_bytes(update.state),
        },
    }


def qualify_device_adaptation() -> dict[str, object]:
    """DEVICE_BISECTION commits NATIVE_BISECTION byte for byte; metric routes agree."""
    meshes = _bisection_meshes()
    bisection = {
        "triangles": _bisection_device_parity(meshes["triangles"], (256, 512)),
        "tetrahedra": _bisection_device_parity(meshes["tetrahedra"], (256, 1024)),
    }
    metric = _bisection_metric_comparison(meshes["triangles"])
    return {
        "status": "passed",
        "source": {
            **{name: record["source"] for name, record in bisection.items()},
            "metric": metric["source"],
        },
        "target": {
            **{name: record["target"] for name, record in bisection.items()},
            "metric": metric["target"],
        },
        "quality": {
            **{name: record["quality"] for name, record in bisection.items()},
            "metric": metric["quality"],
        },
        "transfer": {
            **{name: record["transfer"] for name, record in bisection.items()},
            "metric": metric["transfer"],
        },
        "resources": {
            **{name: record["resources"] for name, record in bisection.items()},
            "metric": metric["resources"],
        },
        "bisection": bisection,
        "metric": metric,
        "runtime": _runtime_identity(),
    }


_CAD_CURVING_ASSOCIATION = phx.meshing.AssociationPropagationPolicy(
    classification_tolerance=1.0e-8
)


def _cad_curving_volume(mesh: Any, geometry: Any) -> float:
    """Volume of mapped P1/P2/P3 tetrahedra.

    The degree-aware finite-element rule integrates their polynomial Jacobian
    determinant exactly.
    """
    field = phx.discretization.FiniteElementFieldSpec(
        "x", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=geometry
    ).prepare()
    blocks = discretization.evaluate_geometry("x", geometry.coordinates)
    return float(sum(jnp.sum(block.measure) for block in blocks))


def _cad_curving_certificate(certificate: Any) -> dict[str, object]:
    lower = np.asarray(certificate.determinant_lower)
    upper = np.asarray(certificate.determinant_upper)
    return {
        "certificate_id": certificate.certificate_id,
        "certified_valid": certificate.certified_valid_count,
        "invalid": certificate.invalid_count,
        "unresolved": certificate.unresolved_count,
        "minimum_determinant_lower_bound": float(np.min(lower)),
        "minimum_lower_to_upper_ratio": float(np.min(lower / upper)),
        "depth_histogram": np.bincount(np.asarray(certificate.depth)).tolist(),
        "retained_bytes": sum(
            array.nbytes
            for array in (
                certificate.status,
                certificate.determinant_lower,
                certificate.determinant_upper,
                certificate.depth,
            )
        ),
    }


def _cad_curving_association(mesh: Any, projection: Any) -> Any:
    """Vertex association plus completeness and ambiguity of every entity level."""
    vertices, vertex_record = _measured(
        lambda: phx.meshing.associate_mesh_vertices(
            mesh, projection, policy=_CAD_CURVING_ASSOCIATION
        )
    )
    entities, entity_record = _measured(
        lambda: tuple(
            phx.meshing.associate_mesh_entities(
                mesh,
                # ty: ignore[invalid-argument-type]
                vertices,
                projection,
                dimension,
                policy=_CAD_CURVING_ASSOCIATION,
            )
            for dimension in (1, 2, 3)
        )
    )
    counts = {
        str(dimension): {
            # ty: ignore[unresolved-attribute]
            "association_id": association.association_id,
            # ty: ignore[unresolved-attribute]
            "rows": association.target_global_ids.shape[0],
            # ty: ignore[unresolved-attribute]
            "resolved": int(np.count_nonzero(np.asarray(association.resolved))),
            # ty: ignore[unresolved-attribute]
            "ambiguous": int(np.count_nonzero(np.asarray(association.ambiguous))),
            # ty: ignore[unresolved-attribute]
            "maximum_residual": float(np.max(np.asarray(association.residuals))),
        }
        # ty: ignore[not-iterable]
        for dimension, association in enumerate((vertices, *entities))
    }
    if any(
        entry["resolved"] != entry["rows"] or entry["ambiguous"]
        for entry in counts.values()
    ):
        raise RuntimeError("cad-curving: a mesh entity has no unique B-Rep class.")
    tolerance = _CAD_CURVING_ASSOCIATION.classification_tolerance
    if counts["0"]["maximum_residual"] > tolerance:
        raise RuntimeError("cad-curving: a mesh vertex lies off its B-Rep class.")
    return vertices, counts, {"vertices": vertex_record, "entities": entity_record}


def _cad_curving_degree(
    mesh: Any, association: Any, projection: Any, degree: Any, exact_volume: Any
) -> Any:
    """Curve, independently certify, and measure one geometry degree."""
    policy = phx.meshing.HighOrderCurvingPolicy(degree=degree)
    curved, curving_record = _measured(
        lambda: phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    )
    # ty: ignore[unresolved-attribute]
    if curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED:
        # ty: ignore[unresolved-attribute]
        raise RuntimeError(f"cad-curving: P{degree} ended with {curved.status}.")
    certificate, certification_record = _measured(
        lambda: phx.discretization.certify_cell_geometry_validity(
            # ty: ignore[unresolved-attribute]
            curved.geometry,
            mesh=mesh,
        )
    )
    valid = phx.discretization.CellValidityStatus.CERTIFIED_VALID
    # ty: ignore[unresolved-attribute]
    if not np.all(np.asarray(certificate.status) == valid):
        raise RuntimeError(f"cad-curving: a P{degree} cell is not certified valid.")
    # ty: ignore[unresolved-attribute]
    if certificate.certificate_id != curved.evidence.certificate.certificate_id:
        raise RuntimeError("cad-curving: recertification differs from curving evidence.")
    # ty: ignore[unresolved-attribute]
    if curved.evidence.maximum_residual > policy.residual_tolerance:
        raise RuntimeError(f"cad-curving: P{degree} nodes lie off the B-Rep.")
    straight = phx.meshing.verify_curved_geometry(
        # ty: ignore[unresolved-attribute]
        curved.straight,
        mesh,
        association,
        projection,
        policy=policy,
    )
    if straight.accepted:
        raise RuntimeError("cad-curving: straight nodes lie on the curved B-Rep.")
    volumes, volume_record = _measured(
        lambda: (
            # ty: ignore[unresolved-attribute]
            _cad_curving_volume(mesh, curved.straight),
            # ty: ignore[unresolved-attribute]
            _cad_curving_volume(mesh, curved.geometry),
        )
    )
    # ty: ignore[not-iterable]
    errors = tuple(abs(volume - exact_volume) / exact_volume for volume in volumes)
    if not errors[1] < 0.25 * errors[0]:
        raise RuntimeError(f"cad-curving: P{degree} did not reduce the volume error.")
    # ty: ignore[unresolved-attribute]
    geometry = curved.geometry
    return {
        "target": {
            # ty: ignore[unresolved-attribute]
            "result_id": curved.result_id,
            "geometry_layout_id": geometry.geometry_layout_id,
            # ty: ignore[unresolved-attribute]
            "evidence_id": curved.evidence.evidence_id,
            "geometry_nodes": geometry.coordinates.shape[0],
        },
        "quality": {
            "before": {
                "minimum_scaled_jacobian": straight.minimum_scaled_jacobian,
                "maximum_distortion": straight.maximum_distortion,
                "certificate": _cad_curving_certificate(straight.certificate),
            },
            "after": {
                # ty: ignore[unresolved-attribute]
                "minimum_scaled_jacobian": curved.evidence.minimum_scaled_jacobian,
                # ty: ignore[unresolved-attribute]
                "maximum_distortion": curved.evidence.maximum_distortion,
                "certificate": _cad_curving_certificate(certificate),
            },
        },
        "conservation": {
            "exact_volume": exact_volume,
            # ty: ignore[not-subscriptable]
            "straight_volume": volumes[0],
            # ty: ignore[not-subscriptable]
            "curved_volume": volumes[1],
            "straight_relative_error": errors[0],
            "curved_relative_error": errors[1],
            "straight_node_residual": straight.maximum_residual,
            # ty: ignore[unresolved-attribute]
            "curved_node_residual": curved.evidence.maximum_residual,
            "residual_tolerance": policy.residual_tolerance,
        },
        "curving": {
            # ty: ignore[unresolved-attribute]
            "status": curved.status.value,
            "policy_id": policy.policy_id,
            # ty: ignore[unresolved-attribute]
            "relaxation_rounds": len(curved.minimizations),
            # ty: ignore[unresolved-attribute]
            "constrained_nodes": int(np.count_nonzero(curved.evidence.constrained)),
        },
        "resources": {
            "curving": curving_record,
            "certification": certification_record,
            "volume": volume_record,
            "geometry_retained_bytes": geometry.coordinates.nbytes
            + sum(routes.nbytes for routes in geometry.geometry_dofs),
        },
    }


def _cad_curving_solid(
    name: Any, shape: Any, exact_volume: Any, size: Any, directory: Path
) -> Any:
    """Gmsh P1 tetrahedra of one OCCT solid curved to P2 and P3 on its B-Rep."""
    contract = phx.SpatialCoordinateContract.si()
    path = directory / f"{name}.brep"
    phx.geometry.persist_occt_shape(shape, path, coordinate_contract=contract)
    model = phx.geometry.import_brep(
        path, coordinate_contract=contract, linear_deflection=0.05, angular_deflection=0.2
    )
    source = phx.geometry.BRepSource(model)
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 3)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    result, meshing_record = _measured(
        lambda: provider.plan(source, specification).execute()
    )
    # ty: ignore[unresolved-attribute]
    if not result.audit.passed:
        raise RuntimeError(f"cad-curving: the Gmsh {name} mesh failed its audit.")
    projection, projection_record = _measured(
        lambda: phx.geometry.prepare_brep_projection(model, path)
    )
    association, counts, association_records = _cad_curving_association(
        # ty: ignore[unresolved-attribute]
        result.mesh,
        projection,
    )
    # ty: ignore[unresolved-attribute]
    straight_certificate = phx.discretization.certify_cell_geometry_validity(result.mesh)
    degrees = {
        f"P{degree}": _cad_curving_degree(
            # ty: ignore[unresolved-attribute]
            result.mesh,
            association,
            projection,
            degree,
            exact_volume,
        )
        for degree in (2, 3)
    }
    return {
        "source": {
            "source_id": model.source_id,
            "source_revision": model.source_revision,
            "source_digest": model.source_digest,
            "model_id": model.model_id,
            # ty: ignore[unresolved-attribute]
            "projection_id": projection.projection_id,
            # ty: ignore[unresolved-attribute]
            "brep_entity_counts": list(projection.entity_counts),
            # ty: ignore[invalid-argument-type]
            "straight": _result_identity(result),
        },
        "target": {key: record["target"] for key, record in degrees.items()},
        "quality": {
            "before": {
                # ty: ignore[unresolved-attribute]
                **_quality_summary(result.quality),
                "certificate": _cad_curving_certificate(straight_certificate),
            },
            "after": {key: record["quality"] for key, record in degrees.items()},
        },
        "association": counts,
        "conservation": {key: record["conservation"] for key, record in degrees.items()},
        "curving": {key: record["curving"] for key, record in degrees.items()},
        "resources": {
            "meshing": meshing_record,
            "projection": projection_record,
            "association": association_records,
            **{key: record["resources"] for key, record in degrees.items()},
        },
        "provider": {
            # ty: ignore[unresolved-attribute]
            "provider": result.provider.name,
            # ty: ignore[unresolved-attribute]
            "version": result.runtime.actual_version,
        },
    }


def qualify_cad_curving() -> dict[str, object]:
    """Certified P2/P3 curving of Gmsh tetrahedra onto OCCT cylinder and sphere B-Reps."""
    from importlib.util import find_spec

    if find_spec("OCP") is None:
        return _missing_dependency(
            "OCP",
            "OCP (cadquery-ocp-novtk) is a core phydrax dependency; reinstall phydrax.",
        )
    if find_spec("gmsh") is None:
        return _missing_dependency(
            "gmsh",
            "Install the optional Gmsh Python package: "
            "pip install 'phydrax[meshing-gmsh]'.",
        )
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere

    with TemporaryDirectory(prefix="phydrax-cad-curving-") as temporary:
        directory = Path(temporary)
        solids = {
            "cylinder": _cad_curving_solid(
                "cylinder",
                BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape(),
                2.0 * np.pi,
                0.7,
                directory,
            ),
            "sphere": _cad_curving_solid(
                "sphere",
                BRepPrimAPI_MakeSphere(1.0).Shape(),
                4.0 / 3.0 * np.pi,
                0.6,
                directory,
            ),
        }
    fields = ("source", "target", "quality", "association", "conservation", "curving")
    return {
        "status": "passed",
        **{
            field: {name: solid[field] for name, solid in solids.items()}
            for field in fields
        },
        "resources": {name: solid["resources"] for name, solid in solids.items()},
        "runtime": {
            **_runtime_identity(),
            "providers": {name: solid["provider"] for name, solid in solids.items()},
        },
    }


_PREDICATE_ARITY = {"orient2d": 3, "orient3d": 4, "incircle": 4, "insphere": 5}
_PREDICATE_OFFSET = 2.0**40


def _predicate_integers(arrays: tuple[np.ndarray, ...], /) -> tuple[np.ndarray, ...]:
    """Float64 coordinates as exact Python integers on one common binary scale."""
    mantissas, exponents = [], []
    for array in arrays:
        fraction, exponent = np.frexp(array)
        mantissas.append(np.ldexp(fraction, 53).astype(np.int64))
        exponents.append(exponent.astype(np.int64) - 53)
    scale = min(
        np.min(np.where(mantissa != 0, exponent, np.iinfo(np.int64).max))
        for mantissa, exponent in zip(mantissas, exponents, strict=True)
    )
    return tuple(
        np.left_shift(
            mantissa.astype(np.object_),
            np.where(mantissa != 0, exponent - scale, 0).astype(np.object_),
        )
        for mantissa, exponent in zip(mantissas, exponents, strict=True)
    )


def _integer_determinant(rows: list[list[np.ndarray]], /) -> np.ndarray:
    """Laplace expansion of a small matrix whose entries are integer object arrays."""
    if len(rows) == 1:
        return rows[0][0]
    total = np.zeros(rows[0][0].shape, dtype=np.object_)
    for column, entry in enumerate(rows[0]):
        minor = [row[:column] + row[column + 1 :] for row in rows[1:]]
        term = entry * _integer_determinant(minor)
        total = total + term if column % 2 == 0 else total - term
    return total


def _reference_predicate_signs(
    name: str, arrays: tuple[np.ndarray, ...], /
) -> np.ndarray:
    """Predicate signs in the phydrax convention from exact integer arithmetic."""
    points = _predicate_integers(arrays)
    width = arrays[0].shape[-1]
    match name:
        case "orient2d" | "orient3d":
            # det[b - a, c - a(, d - a)]
            determinant = _integer_determinant(
                [
                    [point[:, axis] - points[0][:, axis] for axis in range(width)]
                    for point in points[1:]
                ]
            )
        case "incircle" | "insphere":
            # Lifted rows (p - q, |p - q|^2) against the query q; insphere is
            # positive inside the sphere of a positively oriented tetrahedron.
            rows = [
                [point[:, axis] - points[-1][:, axis] for axis in range(width)]
                for point in points[:-1]
            ]
            determinant = _integer_determinant(
                [row + [sum(value * value for value in row)] for row in rows]
            )
            if name == "insphere":
                determinant = -determinant
        case _:
            raise ValueError(f"Unknown predicate {name!r}.")
    return (determinant > 0).astype(np.int8) - (determinant < 0).astype(np.int8)


def _ulp_steps(values: np.ndarray, steps: np.ndarray, /) -> np.ndarray:
    """Move each coordinate by ``steps`` in {-1, 0, 1} units in the last place."""
    return np.where(
        steps > 0,
        np.nextafter(values, np.inf),
        np.where(steps < 0, np.nextafter(values, -np.inf), values),
    )


def _integer_shell(radius: int, dimension: int, /) -> np.ndarray:
    """Integer points on the circle/sphere of an integer radius, by polar angle."""
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(*(axis,) * dimension, indexing="ij"), axis=-1)
    grid = grid.reshape((-1, dimension))
    shell = grid[np.sum(grid * grid, axis=1) == radius * radius].astype(np.float64)
    return shell[np.argsort(np.arctan2(shell[:, 1], shell[:, 0]), kind="stable")]


def _shell_rows(
    shell: np.ndarray,
    simplices: np.ndarray,
    offset: float,
    rng: np.random.Generator,
    /,
) -> tuple[np.ndarray, ...]:
    """Every shell point queried against every simplex, queries moved by one ulp.

    The shell is translated by ``offset``; every coordinate is then nonzero, so a
    one-ulp move never creates a subnormal outside the meshcore exact domain.
    """
    count = shell.shape[0]
    corners = np.repeat(simplices, count, axis=0)
    query = shell[np.tile(np.arange(count), simplices.shape[0])] + offset
    query = _ulp_steps(query, rng.integers(-1, 2, size=query.shape))
    return tuple(
        shell[corners[:, index]] + offset for index in range(simplices.shape[1])
    ) + (query,)


def _predicate_families(
    rng: np.random.Generator, /
) -> dict[str, dict[str, tuple[np.ndarray, ...]]]:
    """Random rows plus adversarial near-degenerate rows for every predicate."""

    def random(arity: int, width: int) -> tuple[np.ndarray, ...]:
        return tuple(rng.standard_normal((2048, width)) for _ in range(arity))

    # Kettner et al.: points near the line y = x and the plane z = y, moved by
    # single ulps of 0.5.
    steps = np.arange(16) * 2.0**-53
    near_line = 0.5 + np.stack(np.meshgrid(steps, steps, indexing="ij"), axis=-1)
    near_line = near_line.reshape((-1, 2))
    steps = np.arange(8) * 2.0**-53
    near_plane = 0.5 + np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1)
    near_plane = near_plane.reshape((-1, 3))
    # Exactly collinear/coplanar integer points far from the origin; the last point
    # moves by at most one ulp of 2**40.
    line = rng.integers(-64, 65, size=(3, 512))
    far_line = _PREDICATE_OFFSET + np.stack((line, 2 * line + 1), axis=-1).astype(
        np.float64
    )
    far_line[2, :, 1] = _ulp_steps(far_line[2, :, 1], rng.integers(-1, 2, size=512))
    plane = rng.integers(-64, 65, size=(4, 512, 2))
    far_plane = _PREDICATE_OFFSET + np.concatenate(
        (plane, plane[..., :1] + 2 * plane[..., 1:] + 3), axis=-1
    ).astype(np.float64)
    far_plane[3, :, 2] = _ulp_steps(far_plane[3, :, 2], rng.integers(-1, 2, size=512))
    # Cocircular lattice: every counterclockwise triple of the 12 integer points of
    # the circle of radius 5. Cospherical lattice: positively oriented tetrahedra
    # of the 30 integer points of the sphere of radius 3.
    circle = _integer_shell(5, 2)
    index = np.arange(circle.shape[0])
    first, second, third = np.meshgrid(index, index, index, indexing="ij")
    ordered = (first < second) & (second < third)
    triples = np.stack((first[ordered], second[ordered], third[ordered]), axis=1)
    sphere = _integer_shell(3, 3)
    tetrahedra = np.argsort(rng.random((128, sphere.shape[0])), axis=1)[:, :4]
    orientation = _reference_predicate_signs(
        "orient3d", tuple(sphere[tetrahedra[:, index]] for index in range(4))
    )
    tetrahedra = np.where(
        (orientation < 0)[:, None], tetrahedra[:, (1, 0, 2, 3)], tetrahedra
    )[orientation != 0]
    return {
        "orient2d": {
            "random": random(3, 2),
            "collinear_ulp": (
                near_line,
                np.broadcast_to(np.asarray((12.0, 12.0)), near_line.shape),
                np.broadcast_to(np.asarray((24.0, 24.0)), near_line.shape),
            ),
            "large_offset": tuple(far_line),
        },
        "orient3d": {
            "random": random(4, 3),
            "coplanar_ulp": (
                np.broadcast_to(np.asarray((1.0, 12.0, 12.0)), near_plane.shape),
                np.broadcast_to(np.asarray((24.0, 24.0, 24.0)), near_plane.shape),
                np.broadcast_to(np.asarray((-7.0, 3.0, 3.0)), near_plane.shape),
                near_plane,
            ),
            "large_offset": tuple(far_plane),
        },
        "incircle": {
            "random": random(4, 2),
            "cocircular_lattice": _shell_rows(circle, triples, 0.5, rng),
            "large_offset": _shell_rows(circle, triples, _PREDICATE_OFFSET, rng),
        },
        "insphere": {
            "random": random(5, 3),
            "cospherical_lattice": _shell_rows(sphere, tetrahedra, 0.5, rng),
            "large_offset": _shell_rows(sphere, tetrahedra, _PREDICATE_OFFSET, rng),
        },
    }


def _predicate_family_record(
    routes: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: np.ndarray,
    perturbed: np.ndarray,
    /,
) -> dict[str, object]:
    """Certainty, correctness, agreement, and degeneracy counts of one family."""
    rows = reference.size
    exact = routes["exact"][0]
    record: dict[str, object] = {"rows": rows}
    for route, (signs, certain) in routes.items():
        record[route] = {
            "certain": int(np.count_nonzero(certain)),
            "uncertain_fraction": 1.0 - int(np.count_nonzero(certain)) / rows,
            "certified_zeros": int(np.count_nonzero(certain & (signs == 0))),
            "wrong_certain_signs": int(np.count_nonzero(certain & (signs != reference))),
        }
    pairs = (("filtered", "device"), ("filtered", "exact"), ("device", "exact"))
    record["certain_disagreements"] = {
        f"{first}_{second}": int(
            np.count_nonzero(
                routes[first][1]
                & routes[second][1]
                & (routes[first][0] != routes[second][0])
            )
        )
        for first, second in pairs
    }
    record["exact_signs"] = {
        "negative": int(np.count_nonzero(exact == -1)),
        "zero": int(np.count_nonzero(exact == 0)),
        "positive": int(np.count_nonzero(exact == 1)),
    }
    record["reference_zeros"] = int(np.count_nonzero(reference == 0))
    record["symbolic"] = {
        "zeros": int(np.count_nonzero(perturbed == 0)),
        "changed_strict_signs": int(
            np.count_nonzero((exact != 0) & (perturbed != exact))
        ),
    }
    return record


def _qualify_predicate(
    name: str, families: dict[str, tuple[np.ndarray, ...]], /
) -> tuple[dict[str, object], dict[str, object], np.ndarray, int]:
    """Evaluate one predicate in every mode on all families against the reference."""
    from phydrax._meshcore import (
        symbolic_incircle,
        symbolic_insphere,
        symbolic_orient2d,
        symbolic_orient3d,
    )

    function, symbolic = {
        "orient2d": (phx.geometry.orient2d, symbolic_orient2d),
        "orient3d": (phx.geometry.orient3d, symbolic_orient3d),
        "incircle": (phx.geometry.incircle, symbolic_incircle),
        "insphere": (phx.geometry.insphere, symbolic_insphere),
    }[name]
    mode = phx.geometry.PredicateMode
    arity = _PREDICATE_ARITY[name]
    arrays = tuple(
        np.concatenate([rows[index] for rows in families.values()])
        for index in range(arity)
    )
    bounds = np.cumsum([0, *(rows[0].shape[0] for rows in families.values())])
    filtered, filtered_stage = _measured(lambda: function(*arrays, mode=mode.FILTERED))
    device_entry = jax.jit(lambda *values: function(*values, mode=mode.FILTERED_DEVICE))
    device_arrays = tuple(jnp.asarray(array) for array in arrays)
    _, device_cold = _measured(
        lambda: jax.block_until_ready(device_entry(*device_arrays))
    )
    device, device_warm = _measured(
        lambda: jax.block_until_ready(device_entry(*device_arrays))
    )
    exact, exact_stage = _measured(lambda: function(*arrays, mode=mode.EXACT))
    reference, reference_stage = _measured(
        lambda: np.concatenate(
            [_reference_predicate_signs(name, rows) for rows in families.values()]
        )
    )
    swapped = function(arrays[1], arrays[0], *arrays[2:], mode=mode.EXACT).signs
    # Distinct ids: the index-ordered perturbation used inside the triangulations.
    # ty: ignore[too-many-positional-arguments]
    perturbed = symbolic(*arrays, np.arange(arity, dtype=np.int64))
    routes = {
        # ty: ignore[unresolved-attribute]
        "filtered": (filtered.signs, filtered.certain),
        # ty: ignore[unresolved-attribute]
        "device": (np.asarray(device.signs), np.asarray(device.certain)),
        # ty: ignore[unresolved-attribute]
        "exact": (exact.signs, exact.certain),
    }
    records = {
        family: _predicate_family_record(
            {
                route: (signs[start:stop], certain[start:stop])
                for route, (signs, certain) in routes.items()
            },
            # ty: ignore[not-subscriptable]
            reference[start:stop],
            perturbed[start:stop],
        )
        for family, start, stop in zip(families, bounds[:-1], bounds[1:], strict=True)
    }
    resources = {
        "filtered": filtered_stage,
        "device_cold": device_cold,
        "device_warm": device_warm,
        "exact": exact_stage,
        # ty: ignore[unresolved-attribute]
        "exact_native_rows": int(np.count_nonzero(~filtered.certain)),
        "integer_reference": reference_stage,
    }
    # ty: ignore[invalid-return-type, unresolved-attribute]
    return records, resources, exact.signs, int(np.count_nonzero(swapped != -exact.signs))


def _predicate_failures(name: str, family: str, record: dict, /) -> list[str]:
    """Failed checks of one family record (empty when the family qualifies)."""
    routes = ("filtered", "device", "exact")
    failures = [
        f"{route} certified {record[route]['wrong_certain_signs']} wrong signs"
        for route in routes
        if record[route]["wrong_certain_signs"]
    ]
    failures += [
        f"{pair} disagree on {count} certain rows"
        for pair, count in record["certain_disagreements"].items()
        if count
    ]
    if record["exact"]["certain"] != record["rows"]:
        failures.append("EXACT left uncertain rows")
    if record["exact_signs"]["zero"] != record["reference_zeros"]:
        failures.append("EXACT zeros differ from the exact degeneracies")
    if record["symbolic"]["zeros"] or record["symbolic"]["changed_strict_signs"]:
        failures.append("symbolic perturbation left ties or changed a strict sign")
    if family == "random":
        if min(record["filtered"]["certain"], record["device"]["certain"]) < (
            0.9 * record["rows"]
        ):
            failures.append("filters left more than 10% of random rows uncertain")
    elif min(record["exact_signs"].values()) == 0:
        failures.append("adversarial rows miss an exact zero or a strict sign")
    return [f"{name}/{family}: {failure}" for failure in failures]


def qualify_predicates() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "EXACT predicates and the symbolic tie-breaking require the native "
            "meshcore library; install phydrax[meshcore] or set "
            "PHYDRAX_MESHCORE_LIBRARY to the shared library path.",
        )
    from phydrax._meshcore import meshcore_exact_domain

    families = _predicate_families(np.random.default_rng(20260926))
    records, resources, exact_signs, antisymmetry = {}, {}, {}, {}
    for name, rows in families.items():
        records[name], resources[name], exact_signs[name], antisymmetry[name] = (
            _qualify_predicate(name, rows)
        )
    failures = [
        failure
        for name, per_family in records.items()
        for family, record in per_family.items()
        for failure in _predicate_failures(name, family, record)
    ]
    failures += [
        f"{name}: {count} exact signs are not antisymmetric under argument exchange"
        for name, count in antisymmetry.items()
        if count
    ]
    if failures:
        raise RuntimeError("Predicate qualification failed: " + "; ".join(failures))
    flat = [record for per_family in records.values() for record in per_family.values()]
    rows = sum(record["rows"] for record in flat)

    def total(route: str, key: str) -> int:
        return sum(record[route][key] for record in flat)

    return {
        "status": "passed",
        "source": {
            "inputs_id": canonical_fingerprint(families),
            "rows": {
                name: {family: record["rows"] for family, record in per_family.items()}
                for name, per_family in records.items()
            },
        },
        "target": {"exact_signs_id": canonical_fingerprint(exact_signs), "rows": rows},
        "runtime": {
            **_runtime_identity(),
            "meshcore_exact_exponent_range": list(meshcore_exact_domain()),
        },
        "quality": {
            "before": {
                "filtered_certain_fraction": total("filtered", "certain") / rows,
                "device_certain_fraction": total("device", "certain") / rows,
            },
            "after": {
                "exact_certain_fraction": total("exact", "certain") / rows,
                "exact_wrong_signs": total("exact", "wrong_certain_signs"),
            },
        },
        "conservation": {
            "antisymmetry_violations": antisymmetry,
            "wrong_certain_signs": {
                route: total(route, "wrong_certain_signs")
                for route in ("filtered", "device", "exact")
            },
            "certain_disagreements": sum(
                sum(record["certain_disagreements"].values()) for record in flat
            ),
        },
        "degeneracy_semantics": {
            # EXACT reports exact zeros; only the triangulations break ties, by
            # index-ordered symbolic perturbation.
            "exact_mode": "exact-zero",
            "triangulation_ties": "index-ordered-symbolic-perturbation",
            "exact_zero_rows": sum(record["exact_signs"]["zero"] for record in flat),
            "reference_zero_rows": sum(record["reference_zeros"] for record in flat),
            "symbolic_zero_rows": sum(record["symbolic"]["zeros"] for record in flat),
        },
        "resources": resources,
        "predicates": records,
    }


def _simplex_measures(points: np.ndarray, simplices: np.ndarray, /) -> np.ndarray:
    """Signed triangle areas or tetrahedron volumes."""
    edges = points[simplices[:, 1:]] - points[simplices[:, :1]]
    if points.shape[1] == 2:
        return 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    return np.sum(edges[:, 0] * np.cross(edges[:, 1], edges[:, 2]), axis=1) / 6.0


def _delaunay_record(points: np.ndarray, /) -> dict[str, object]:
    """Exact orientation, empty-ball, hull-coverage, and repeat checks of one set."""
    from scipy.spatial import ConvexHull

    exact = phx.geometry.PredicateMode.EXACT
    triangulation, stage = _measured(lambda: phx.geometry.DelaunayTriangulation(points))
    # ty: ignore[unresolved-attribute]
    simplices = triangulation.simplices
    corners = tuple(points[simplices[:, index]] for index in range(simplices.shape[1]))
    # Every (simplex, point) pair.
    balls = tuple(corner[:, None, :] for corner in corners)
    if points.shape[1] == 2:
        orientation = phx.geometry.orient2d(*corners, mode=exact)
        # ty: ignore[too-many-positional-arguments]
        inside = phx.geometry.incircle(*balls, points[None], mode=exact)
    else:
        orientation = phx.geometry.orient3d(*corners, mode=exact)
        # ty: ignore[too-many-positional-arguments]
        inside = phx.geometry.insphere(*balls, points[None], mode=exact)
    hull = ConvexHull(points).volume
    measure = float(np.sum(_simplex_measures(points, simplices)))
    repeated = phx.geometry.DelaunayTriangulation(points)
    # ty: ignore[unresolved-attribute]
    evidence = triangulation.evidence
    record = {
        "points": points.shape[0],
        "simplices": simplices.shape[0],
        "simplices_id": canonical_fingerprint(simplices),
        "evidence_id": evidence.evidence_id,
        "route": evidence.route,
        "status": evidence.status,
        "predicate_mode": evidence.predicate_mode.value,
        "positive_orientations": int(np.count_nonzero(orientation.signs == 1)),
        "in_ball_checks": inside.signs.size,
        "empty_ball_violations": int(np.count_nonzero(inside.signs > 0)),
        # Zero signs beyond each simplex's own vertices are cocircular/cospherical ties.
        "cospherical_ties": int(np.count_nonzero(inside.signs == 0)) - simplices.size,
        "hull_measure": hull,
        "measure": measure,
        "relative_hull_residual": abs(measure - hull) / hull,
        "repeat_equal": bool(
            np.array_equal(repeated.simplices, simplices)
            and repeated.evidence.evidence_id == evidence.evidence_id
        ),
        "resources": stage,
    }
    if (
        not (np.all(orientation.certain) and np.all(inside.certain))
        or record["positive_orientations"] != simplices.shape[0]
        or record["empty_ball_violations"]
        or record["relative_hull_residual"] > 1.0e-12
        or not record["repeat_equal"]
        or evidence.status != "ok"
    ):
        raise RuntimeError(f"{evidence.route} verification failed: {record}")
    return record


def _duplicate_record(rng: np.random.Generator, dimension: int, /) -> dict[str, object]:
    """Duplicated points map to their first copy and never become vertices or cells."""
    count, copies = 48, 8
    base = rng.random((count, dimension))
    chosen = np.sort(rng.choice(count, size=copies, replace=False))
    points = np.concatenate((base, base[chosen]))
    triangulation = phx.geometry.DelaunayTriangulation(points)
    unique = phx.geometry.DelaunayTriangulation(base)
    voronoi = phx.geometry.VoronoiDiagram(
        points,
        box_lower=np.zeros((dimension,), dtype=np.float64),
        box_upper=np.ones((dimension,), dtype=np.float64),
    )
    measures = voronoi.cells.measures
    record = {
        "input_points": points.shape[0],
        "duplicate_count": triangulation.evidence.duplicate_count,
        "vertex_count": triangulation.evidence.vertex_count,
        "vertex_map_to_first_copy": bool(
            np.array_equal(
                triangulation.vertex_map, np.concatenate((np.arange(count), chosen))
            )
        ),
        "duplicates_in_simplices": int(
            np.count_nonzero(triangulation.simplices >= count)
        ),
        "simplices_equal_unique_input": bool(
            np.array_equal(triangulation.simplices, unique.simplices)
        ),
        "voronoi_duplicate_count": voronoi.evidence.duplicate_count,
        "voronoi_duplicate_measure": float(np.sum(measures[count:])),
        "voronoi_measure_sum": float(np.sum(measures)),
    }
    if (
        record["duplicate_count"] != copies
        or record["voronoi_duplicate_count"] != copies
        or not record["vertex_map_to_first_copy"]
        or record["duplicates_in_simplices"]
        or not record["simplices_equal_unique_input"]
        or record["voronoi_duplicate_measure"] != 0.0
        or abs(record["voronoi_measure_sum"] - 1.0) > 1.0e-12
    ):
        raise RuntimeError(f"{dimension}D duplicate handling failed: {record}")
    return record


def _segment_conformity(
    triangulation: phx.geometry.ConstrainedDelaunayTriangulation,
    segments: np.ndarray,
    /,
) -> dict[str, object]:
    """Exact collinearity and length coverage of the edges carrying each segment."""
    points, triangles = triangulation.points, triangulation.triangles
    cells, corners = np.nonzero(triangulation.segment_ids >= 0)
    edges = np.sort(
        np.stack(
            (triangles[cells, (corners + 1) % 3], triangles[cells, (corners + 2) % 3]),
            axis=1,
        ),
        axis=1,
    )
    carried = np.unique(
        np.column_stack((triangulation.segment_ids[cells, corners], edges)), axis=0
    )
    owner, first, second = carried.T
    start, stop = points[segments[owner, 0]], points[segments[owner, 1]]
    on_line = phx.geometry.orient2d(
        np.concatenate((start, start)),
        np.concatenate((stop, stop)),
        np.concatenate((points[first], points[second])),
        mode=phx.geometry.PredicateMode.EXACT,
    )
    lengths = np.linalg.norm(points[second] - points[first], axis=1)
    covered = np.bincount(owner, weights=lengths, minlength=segments.shape[0])
    expected = np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]], axis=1)
    return {
        "segments": segments.shape[0],
        "carried_edges": carried.shape[0],
        "uncovered_segments": int(np.count_nonzero(covered == 0.0)),
        "off_segment_endpoints": int(np.count_nonzero(on_line.signs != 0)),
        "maximum_relative_length_defect": float(
            np.max(np.abs(covered - expected) / expected)
        ),
    }


def _constrained_record() -> tuple[
    dict[str, object], phx.meshing.CellMeshingResult, phx.meshing.CellMeshingResult
]:
    """Plate with a hole and an interior crack: CDT, refinement, and certification."""
    points = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.75, 0.75),
            (1.25, 0.75),
            (1.25, 1.25),
            (0.75, 1.25),
            (0.25, 0.4),
            (1.75, 0.4),
        )
    )
    loop = np.arange(4, dtype=np.int32)
    ring = np.stack((loop, (loop + 1) % 4), axis=1)
    segments = np.concatenate((ring, 4 + ring, np.asarray(((8, 9),), dtype=np.int32)))
    holes = np.asarray(((1.0, 1.0),))
    min_angle, max_area = 28.0, 0.02

    def triangulate(max_steiner: int, refine: bool) -> Any:
        return phx.geometry.ConstrainedDelaunayTriangulation(
            points,
            segments,
            holes=holes,
            min_angle=min_angle if refine else 0.0,
            max_area=max_area if refine else np.inf,
            max_steiner=max_steiner,
        )

    unrefined, unrefined_stage = _measured(lambda: triangulate(0, False))
    refined, refined_stage = _measured(lambda: triangulate(4000, True))
    limited = triangulate(1, True)
    repeated = triangulate(4000, True)
    # ty: ignore[unresolved-attribute]
    areas = _simplex_measures(refined.points, refined.triangles)
    orientation = phx.geometry.orient2d(
        # ty: ignore[unresolved-attribute]
        *(refined.points[refined.triangles[:, index]] for index in range(3)),
        mode=phx.geometry.PredicateMode.EXACT,
    )

    def certify(triangulation: Any) -> Any:
        return phx.meshing.certify_cell_mesh(
            phx.discretization.CellMesh.from_triangles(
                triangulation.points, triangulation.triangles
            ),
            _contract(),
        )

    before, before_stage = _measured(lambda: certify(unrefined))
    after, after_stage = _measured(lambda: certify(refined))
    plate_area = 4.0 - 0.25
    record = {
        # ty: ignore[unresolved-attribute]
        "status": refined.evidence.status,
        # ty: ignore[unresolved-attribute]
        "route": refined.evidence.route,
        # ty: ignore[unresolved-attribute]
        "predicate_mode": refined.evidence.predicate_mode.value,
        "requested_minimum_angle_degrees": min_angle,
        # ty: ignore[unresolved-attribute]
        "minimum_angle_degrees": refined.evidence.minimum_angle_degrees,
        # ty: ignore[unresolved-attribute]
        "unrefined_minimum_angle_degrees": unrefined.evidence.minimum_angle_degrees,
        # ty: ignore[unresolved-attribute]
        "certified_minimum_angle_degrees": float(np.degrees(after.quality.minimum_angle)),
        "requested_maximum_area": max_area,
        "maximum_area": float(np.max(areas)),
        "relative_area_residual": abs(float(np.sum(areas)) - plate_area) / plate_area,
        "positive_orientations": int(np.count_nonzero(orientation.signs == 1)),
        # ty: ignore[unresolved-attribute]
        "triangles": refined.triangles.shape[0],
        # ty: ignore[unresolved-attribute]
        "unrefined_triangles": unrefined.triangles.shape[0],
        # ty: ignore[unresolved-attribute]
        "steiner_points": refined.evidence.steiner_count,
        # ty: ignore[invalid-argument-type]
        "conformity": _segment_conformity(refined, segments),
        # ty: ignore[invalid-argument-type]
        "unrefined_conformity": _segment_conformity(unrefined, segments),
        "limited_status": limited.evidence.status,
        "limited_steiner_points": limited.evidence.steiner_count,
        # ty: ignore[unresolved-attribute]
        "evidence_id": refined.evidence.evidence_id,
        "repeat_equal": bool(
            # ty: ignore[unresolved-attribute]
            repeated.evidence.evidence_id == refined.evidence.evidence_id
            # ty: ignore[unresolved-attribute]
            and np.array_equal(repeated.points, refined.points)
            # ty: ignore[unresolved-attribute]
            and np.array_equal(repeated.triangles, refined.triangles)
        ),
        "resources": {
            "unrefined": unrefined_stage,
            "refined": refined_stage,
            "certify_unrefined": before_stage,
            "certify_refined": after_stage,
        },
    }
    conformity = (record["conformity"], record["unrefined_conformity"])
    if (
        record["status"] != "ok"
        or record["minimum_angle_degrees"] < min_angle
        or record["maximum_area"] > max_area
        or record["relative_area_residual"] > 1.0e-12
        or record["positive_orientations"] != record["triangles"]
        or any(
            item["uncovered_segments"]
            or item["off_segment_endpoints"]
            # ty: ignore[unsupported-operator]
            or item["maximum_relative_length_defect"] > 1.0e-12
            for item in conformity
        )
        or record["limited_status"] != "refinement_limit"
        or record["limited_steiner_points"] > 1
        or not record["repeat_equal"]
        # ty: ignore[unresolved-attribute]
        or not (before.audit.passed and after.audit.passed)
        or record["certified_minimum_angle_degrees"] < min_angle * (1.0 - 1.0e-12)
    ):
        raise RuntimeError(f"Constrained Delaunay qualification failed: {record}")
    # ty: ignore[invalid-return-type]
    return record, before, after


def _reciprocal_faces(cells: phx.geometry.DiagramCells, /) -> bool:
    """Every interior face (owner, neighbor) has the opposite face (neighbor, owner)."""
    owners = np.repeat(np.arange(cells.cell_count), np.diff(cells.cell_face_offsets))
    interior = cells.face_labels >= 0
    forward = np.unique(
        np.stack((owners[interior], cells.face_labels[interior]), axis=1), axis=0
    )
    return bool(np.array_equal(forward, np.unique(forward[:, ::-1], axis=0)))


def _diagram_record(
    diagram: Any, repeated: Any, lower: np.ndarray, upper: np.ndarray, stage: dict, /
) -> dict[str, object]:
    """Box coverage, reciprocity, centroid, and repeat checks of one diagram."""
    cells, evidence = diagram.cells, diagram.evidence
    box = float(np.prod(upper - lower))
    total = float(np.sum(cells.measures))
    occupied = cells.measures > 0.0
    centroids = cells.centroids[occupied]
    record = {
        "route": evidence.route,
        "predicate_mode": evidence.predicate_mode.value,
        "evidence_id": evidence.evidence_id,
        "measures_id": canonical_fingerprint(cells.measures),
        "cells": cells.cell_count,
        "empty_cells": int(np.count_nonzero(~occupied)),
        "redundant_generators": evidence.redundant_count,
        "dual_simplices": evidence.simplex_count,
        "box_measure": box,
        "measure_sum": total,
        "relative_box_residual": abs(total - box) / box,
        "reciprocal_faces": _reciprocal_faces(cells),
        "centroids_inside_box": bool(np.all((centroids >= lower) & (centroids <= upper))),
        "repeat_equal": bool(
            repeated.evidence.evidence_id == evidence.evidence_id
            and np.array_equal(repeated.cells.vertices, cells.vertices)
        ),
        "resources": stage,
    }
    if (
        record["relative_box_residual"] > 1.0e-12
        or np.any(cells.measures < 0.0)
        or not record["reciprocal_faces"]
        or not record["centroids_inside_box"]
        or not record["repeat_equal"]
    ):
        raise RuntimeError(f"{evidence.route} qualification failed: {record}")
    return record


def _diagram_pair(
    rng: np.random.Generator, dimension: int, count: int, /
) -> dict[str, dict[str, object]]:
    """Voronoi and weighted power diagrams of one generator set in a box."""
    lower = np.zeros((dimension,), dtype=np.float64)
    upper = np.asarray((1.0, 2.0, 1.0)[:dimension], dtype=np.float64)
    generators = lower + (upper - lower) * rng.random((count, dimension))
    weights = 0.01 * rng.random(count)
    box = {"box_lower": lower, "box_upper": upper}
    voronoi, voronoi_stage = _measured(
        lambda: phx.geometry.VoronoiDiagram(generators, **box)
    )
    power, power_stage = _measured(
        # ty: ignore[invalid-argument-type]
        lambda: phx.geometry.PowerDiagram(generators, weights, **box)
    )
    unweighted = phx.geometry.PowerDiagram(
        generators,
        np.zeros((count,), dtype=np.float64),
        # ty: ignore[invalid-argument-type]
        **box,
    )
    voronoi_record = _diagram_record(
        voronoi,
        phx.geometry.VoronoiDiagram(generators, **box),
        lower,
        upper,
        voronoi_stage,
    )
    # ty: ignore[unresolved-attribute]
    deviation = float(np.max(np.abs(unweighted.cells.measures - voronoi.cells.measures)))
    voronoi_record["unweighted_power_maximum_deviation"] = deviation
    # ty: ignore[unresolved-attribute]
    if deviation > 1.0e-10 * float(np.max(voronoi.cells.measures)):
        raise RuntimeError("Unweighted power cells differ from the Voronoi cells.")
    return {
        f"voronoi_{dimension}d": voronoi_record,
        f"power_{dimension}d": _diagram_record(
            power,
            # ty: ignore[invalid-argument-type]
            phx.geometry.PowerDiagram(generators, weights, **box),
            lower,
            upper,
            power_stage,
        ),
    }


def qualify_triangulation() -> dict[str, object]:
    if not meshcore_available():
        return _missing_dependency(
            "meshcore",
            "Exact triangulations and diagrams require the native meshcore library; "
            "install phydrax[meshcore] or set PHYDRAX_MESHCORE_LIBRARY to the shared "
            "library path.",
        )
    rng = np.random.default_rng(20260927)
    lattice_2d = np.stack(np.meshgrid(*(np.arange(6.0),) * 2, indexing="ij"), axis=-1)
    lattice_3d = np.stack(np.meshgrid(*(np.arange(4.0),) * 3, indexing="ij"), axis=-1)
    delaunay = {
        "random_2d": _delaunay_record(rng.random((200, 2))),
        "lattice_2d": _delaunay_record(lattice_2d.reshape((-1, 2))),
        "random_3d": _delaunay_record(rng.random((60, 3))),
        "lattice_3d": _delaunay_record(lattice_3d.reshape((-1, 3))),
    }
    duplicates = {
        f"{dimension}d": _duplicate_record(rng, dimension) for dimension in (2, 3)
    }
    constrained, unrefined, refined = _constrained_record()
    diagrams = {**_diagram_pair(rng, 2, 64), **_diagram_pair(rng, 3, 40)}
    # A zero-weight generator inside the heavy triangle is redundant: no cell.
    hidden = phx.geometry.PowerDiagram(
        np.asarray(((0.2, 0.2), (0.8, 0.2), (0.5, 0.8), (0.5, 0.4))),
        np.asarray((1.0, 1.0, 1.0, 0.0)),
        box_lower=(0.0, 0.0),
        box_upper=(1.0, 1.0),
    )
    redundant = {
        "redundant_count": hidden.evidence.redundant_count,
        # ty: ignore[unresolved-attribute]
        "dual_vertex_map": hidden.dual_vertex_map.tolist(),
        "redundant_measure": float(hidden.cells.measures[3]),
        "measure_sum": float(np.sum(hidden.cells.measures)),
    }
    if (
        redundant["redundant_count"] != 1
        or redundant["dual_vertex_map"][3] != -1
        or redundant["redundant_measure"] != 0.0
        or abs(redundant["measure_sum"] - 1.0) > 1.0e-12
    ):
        raise RuntimeError(f"Redundant power generator was not hidden: {redundant}")
    resources = {
        **{
            f"delaunay_{name}": record.pop("resources")
            for name, record in delaunay.items()
        },
        **{
            f"constrained_{name}": stage
            # ty: ignore[unresolved-attribute]
            for name, stage in constrained.pop("resources").items()
        },
        **{name: record.pop("resources") for name, record in diagrams.items()},
    }
    return {
        "status": "passed",
        "source": {
            "constrained_unrefined": _result_identity(unrefined),
            "delaunay_simplices": {
                name: record["simplices_id"] for name, record in delaunay.items()
            },
        },
        "target": {
            "constrained_refined": _result_identity(refined),
            "evidence": {
                **{name: record["evidence_id"] for name, record in delaunay.items()},
                "constrained": constrained["evidence_id"],
                **{name: record["evidence_id"] for name, record in diagrams.items()},
            },
        },
        "runtime": _runtime_identity(),
        "quality": {
            "before": _quality_summary(unrefined.quality),
            "after": _quality_summary(refined.quality),
        },
        "conservation": {
            "delaunay_relative_hull_residual": {
                name: record["relative_hull_residual"]
                for name, record in delaunay.items()
            },
            "constrained_relative_area_residual": constrained["relative_area_residual"],
            "diagram_relative_box_residual": {
                name: record["relative_box_residual"] for name, record in diagrams.items()
            },
        },
        "determinism": {
            **{name: record["repeat_equal"] for name, record in delaunay.items()},
            "constrained": constrained["repeat_equal"],
            **{name: record["repeat_equal"] for name, record in diagrams.items()},
        },
        "resources": resources,
        "delaunay": delaunay,
        "duplicates": duplicates,
        "constrained": constrained,
        "diagrams": diagrams,
        "redundant_power_generator": redundant,
    }


_SCENARIOS: dict[str, Callable[[argparse.Namespace], dict[str, object]]] = {
    "manifold": lambda args: qualify_manifold(),
    "mmg": lambda args: qualify_volume("mmg", args.executable, args.worker),
    "ftetwild": lambda args: qualify_volume("ftetwild", args.executable, args.worker),
    "gmsh": lambda args: qualify_gmsh(),
    "vorocrust": lambda args: qualify_volume("vorocrust", args.executable, args.worker),
    "poisson": lambda args: qualify_poisson(),
    "cad-partition": lambda args: qualify_cad_partition(),
    "composite-heat-3d": lambda args: qualify_composite_heat_3d(),
    "planar-semantic": lambda args: qualify_planar_semantic(),
    "composite-heat-2d": lambda args: qualify_composite_heat_2d(),
    "layout-stack": lambda args: qualify_layout_stack(),
    "layout-observable": lambda args: qualify_layout_observable(),
    "planar-bands": lambda args: qualify_planar_bands(),
    "hybrid-layer": lambda args: qualify_hybrid_layer(),
    "hybrid-diffusion": lambda args: qualify_hybrid_diffusion(),
    "distribution": lambda args: qualify_distribution(),
    "forest-amr": lambda args: qualify_forest_amr(),
    "omega-h": lambda args: qualify_omega_h(args.worker),
    "omega-h-distributed": lambda args: qualify_omega_h_distributed(args.worker),
    "metric-adaptation": lambda args: qualify_metric_adaptation(),
    "gmsh-metric": lambda args: qualify_gmsh_metric(),
    "boundary-layer": lambda args: qualify_boundary_layer(),
    "gmsh-boundary-layer": lambda args: qualify_gmsh_boundary_layer(),
    "supermesh": lambda args: qualify_supermesh(),
    "remap-p1": lambda args: qualify_remap_p1(),
    "bisection": lambda args: qualify_bisection(),
    "device-adaptation": lambda args: qualify_device_adaptation(),
    "cad-curving": lambda args: qualify_cad_curving(),
    "predicates": lambda args: qualify_predicates(),
    "triangulation": lambda args: qualify_triangulation(),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(_SCENARIOS), default="manifold")
    parser.add_argument("--executable", default="vc_mesh")
    parser.add_argument("--worker", default=None)
    args = parser.parse_args()
    print(json.dumps(_SCENARIOS[args.scenario](args), indent=2))


if __name__ == "__main__":
    main()

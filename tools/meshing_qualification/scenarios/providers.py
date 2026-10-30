from __future__ import annotations


import argparse


import importlib.metadata


import json


import os


import platform


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


from phydrax.geometry.brep import BRepTessellationPolicy
from phydrax.interchange import (
    CadImportPolicy,
    persist_occt_shape,
    read_brep_text,
)
_CAD_CURVING_ASSOCIATION_POLICY = phx.meshing.AssociationPropagationPolicy(
    classification_tolerance=1.0e-8
)



def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _measured(call: Callable[[], object], /) -> tuple[object, dict[str, float]]:
    """Run one stage; report wall seconds and the process peak resident bytes."""
    if os.name != "posix":
        raise RuntimeError("Meshing resource qualification requires a POSIX host.")
    import resource

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


def _three_dimensional_partition() -> Any:
    contract = _contract()
    operands = (
        phx.geometry.BRepPartitionOperand(
            "left",
            phx.geometry.brep_box(
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                coordinate_contract=contract,
                source_id="left",
            ),
            phx.geometry.BRepPartitionRole.REGION,
        ),
        phx.geometry.BRepPartitionOperand(
            "right",
            phx.geometry.brep_box(
                (1.0, 0.0, 0.0),
                (2.0, 1.0, 1.0),
                coordinate_contract=contract,
                source_id="right",
            ),
            phx.geometry.BRepPartitionRole.REGION,
        ),
    )
    plan = phx.geometry.BRepPartitionPlan(
        contract,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=("right", "left")),
    )
    return phx.geometry.partition_brep(plan)


def _region_controls(provider: Any, partition: Any, model: Any = None) -> Any:
    source = partition.model if model is None else model
    return tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope.from_geometry(source, region.entity_ids),
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
            phx.meshing.MeshingScope.from_geometry(source, patch.entity_ids),
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
    whole = phx.meshing.MeshingScope.whole(source, 3)
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
        local = phx.meshing.MeshingScope.from_geometry(
            source, partition.regions[0].entity_ids
        )
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
    with TemporaryDirectory(prefix="phydrax-gmsh-semantic-"):
        partition = _three_dimensional_partition()
        provider = phx.meshing.GmshProvider()
        specification, _controls = _semantic_gmsh_specification(
            provider,
            partition,
            scoped_size=False,
        )
        plan = provider.plan(partition.model, specification)
        result = plan.execute()
        _require_semantic_gmsh_result(result, partition)

        replayed_specification, _ = _semantic_gmsh_specification(
            provider,
            partition,
            scoped_size=False,
        )
        replayed_plan = provider.plan(partition.model, replayed_specification)
        replayed = replayed_plan.execute()
        _require_semantic_gmsh_result(replayed, partition)
        replay_preserved = (
            partition.model.source_revision == partition.revision.revision_id
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

        source_revision = partition.model.source_revision
        geometry_id = partition.model.geometry.geometry_id
        plan.execute()
        if (
            partition.model.source_revision != source_revision
            or partition.model.geometry.geometry_id != geometry_id
        ):
            raise RuntimeError("Native Gmsh qualification changed its immutable CAD source.")

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
        "scoped_size_controls": [control.control_id for control in scoped_controls],
    }


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
    source = persist_occt_shape(
        splitter.Shape(),
        destination,
        coordinate_contract=_contract(),
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.03,
            angular_deflection=0.15,
        ),
    )
    source_face = _hybrid_face_indices(source, 2, 0.0)
    target_face = _hybrid_face_indices(source, 2, 0.3)
    if len(source_face) != 1 or len(target_face) != 1:
        raise RuntimeError("Hybrid qualification could not resolve its exact caps.")
    swept = source.topology.face_solids[source_face[0]][0]
    core = next(index for index in range(source.topology.num_solids) if index != swept)
    provider = phx.meshing.GmshProvider()
    swept_scope = phx.meshing.MeshingScope.from_geometry(source, source.solid_ids[swept])
    core_scope = phx.meshing.MeshingScope.from_geometry(source, source.solid_ids[core])
    layer_schedule = (
        phx.meshing.LayerSchedule((0.08, 0.10, 0.12)) if schedule is None else schedule
    )
    if not isinstance(layer_schedule, phx.meshing.LayerSchedule):
        raise TypeError("schedule must be LayerSchedule or None.")
    control = phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope.from_geometry(source, source.face_ids[source_face[0]]),
        layer_schedule,
        route=phx.meshing.BoundaryLayerRoute.EXACT_SWEEP,
        volume_scope=swept_scope,
        cap_scope=phx.meshing.MeshingScope.from_geometry(
            source, source.face_ids[target_face[0]]
        ),
    )
    whole = phx.meshing.MeshingScope.whole(source, 3)
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
                phx.meshing.MeshingScope.from_geometry(
                    source, source.face_ids[target_face[0]]
                ),
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
        scope = phx.meshing.MeshingScope.whole(surface, 2)
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
            persist_occt_shape(
                BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(),
                path,
                coordinate_contract=coordinate_contract,
            )
            model = read_brep_text(
                path,
                CadImportPolicy(
                    coordinate_contract,
                    tessellation=BRepTessellationPolicy(
                        linear_deflection=0.05,
                        angular_deflection=0.2,
                    ),
                ),
                trusted_root=Path(temporary),
                source_length_unit=coordinate_contract.length_unit,
            ).model
            source = phx.geometry.BRepSource(model)
            provider = phx.meshing.GmshProvider()
            scope = phx.meshing.MeshingScope.whole(source, 3)
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


def _omega_h_worker(worker: str | None, /) -> str | None:
    """Resolve the Omega_h worker exactly as ``OmegaHProvider`` does."""
    configured = (
        phx.config.snapshot().workers.get("omega_h") if worker is None else worker
    )
    return shutil.which("phydrax-omega-h-worker" if configured is None else configured)


def _qualify_omega_h_ranks(worker: str | None, ranks: int, /) -> dict[str, object]:
    """Metric adaptation with LINEAR and CONSERVE fields on ``ranks`` MPI ranks."""
    executable = _omega_h_worker(worker)
    if executable is None:
        return _missing_dependency(
            "phydrax-omega-h-worker",
            "Build native/providers/omega_h against an Omega_h CMake package; configure PHYDRAX_OMEGA_H_WORKER or put phydrax-omega-h-worker on PATH.",
        )
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


def _gmsh_metric_run(
    square: Any, control: phx.meshing.BackgroundMetricControl, options: Any, /
) -> tuple[phx.meshing.CellMeshingResult, dict[str, object]]:
    provider = phx.meshing.GmshProvider(options)
    scope = phx.meshing.MeshingScope.whole(square, 2)
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
        square = persist_occt_shape(
            face,
            Path(temporary) / "square.brep",
            coordinate_contract=contract,
            tessellation=BRepTessellationPolicy(
                linear_deflection=0.01,
                angular_deflection=0.2,
            ),
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


def _gmsh_layered_route(
    provider: Any, source: Any, route: Any, corner: Any, schedule: Any, tolerance: Any, /
) -> Any:
    """Plan, execute, and independently audit one layered Gmsh volume route."""
    points = np.asarray(source.mesh_vertices)
    face_ids = np.asarray(source.triangle_face_ids)
    inner = np.all(np.linalg.norm(points[np.asarray(source.mesh_faces)], axis=2) < 0.5, 1)
    wall_faces = tuple(source.face_ids[index] for index in np.unique(face_ids[inner]))
    whole = phx.meshing.MeshingScope.whole(source, 3)
    control = phx.meshing.BoundaryLayerControl(
        phx.meshing.MeshingScope.from_geometry(source, wall_faces),
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
        source = persist_occt_shape(
            BRepAlgoAPI_Cut(box, ball).Shape(),
            Path(temporary) / "ball-in-box.brep",
            coordinate_contract=_contract(),
            tessellation=BRepTessellationPolicy(
                linear_deflection=0.01,
                angular_deflection=0.1,
            ),
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
            "source_revision": source.source_revision,
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
            mesh, projection, policy=_CAD_CURVING_ASSOCIATION_POLICY
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
                policy=_CAD_CURVING_ASSOCIATION_POLICY,
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
    tolerance = _CAD_CURVING_ASSOCIATION_POLICY.classification_tolerance
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
    persist_occt_shape(shape, path, coordinate_contract=contract)
    model = read_brep_text(
        path,
        CadImportPolicy(
            contract,
            tessellation=BRepTessellationPolicy(
                linear_deflection=0.05,
                angular_deflection=0.2,
            ),
        ),
        trusted_root=directory,
        source_length_unit=contract.length_unit,
    ).model
    source = phx.geometry.BRepSource(model)
    provider = phx.meshing.GmshProvider()
    scope = phx.meshing.MeshingScope.whole(source, 3)
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
        lambda: phx.geometry.prepare_brep_projection(model)
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




from importlib.util import find_spec

from .._runtime import runtime_identity as _runtime_identity


def _configured_worker(name: str, explicit: str | None, default: str, /) -> str | None:
    configured = explicit or phx.config.snapshot().workers.get(name) or os.environ.get({"mmg": "PHYDRAX_MMG_WORKER", "vorocrust": "PHYDRAX_VOROCRUST_WORKER"}.get(name, "")) or default
    return shutil.which(configured) if configured else None


def _missing(dependencies: tuple[str, ...], /) -> dict[str, object] | None:
    absent = tuple(name for name in dependencies if find_spec(name) is None)
    if not absent:
        return None
    return {
        "status": "missing-dependency",
        "dependency": ", ".join(absent),
        "detail": "Install the explicitly optional provider dependencies to run this comparison.",
        "runtime": _runtime_identity(),
    }


def run_optional(name: str, args: argparse.Namespace, /) -> dict[str, object]:
    package_dependencies = {
        "manifold": ("manifold3d",),
        "gmsh": ("gmsh",),
        "gmsh-metric": ("gmsh",),
        "gmsh-boundary-layer": ("gmsh", "OCP"),
        "cad-curving": ("gmsh", "OCP"),
        "poisson": ("open3d",),
        "ftetwild": ("wildmeshing",),
        "planar-bands": ("gmsh",),
    }
    if name in package_dependencies:
        missing = _missing(package_dependencies[name])
        if missing is not None:
            return missing
    if name == "mmg" and _configured_worker("mmg", args.worker, "phydrax-mmg-worker") is None:
        return {"status": "missing-dependency", "dependency": "phydrax-mmg-worker", "detail": "Build the MMG worker and set PHYDRAX_MMG_WORKER or put phydrax-mmg-worker on PATH.", "runtime": _runtime_identity()}
    if name == "vorocrust":
        worker = _configured_worker("vorocrust", args.worker, "phydrax-vorocrust-worker")
        executable = shutil.which(args.executable)
        if worker is None or executable is None:
            missing_names = tuple(key for key, value in (("phydrax-vorocrust-worker", worker), (args.executable, executable)) if value is None)
            return {"status": "missing-dependency", "dependency": ", ".join(missing_names), "detail": "Build the VoroCrust worker and install its upstream vc_mesh executable; configure PHYDRAX_VOROCRUST_WORKER when needed.", "runtime": _runtime_identity()}
    def run_planar_bands() -> dict[str, object]:
        from .native_layers import qualify_planar_bands

        return qualify_planar_bands()

    calls = {
        "manifold": lambda: qualify_manifold(),
        "mmg": lambda: qualify_volume("mmg", args.executable, args.worker),
        "ftetwild": lambda: qualify_volume("ftetwild", args.executable, args.worker),
        "gmsh": lambda: qualify_gmsh(),
        "vorocrust": lambda: qualify_volume("vorocrust", args.executable, args.worker),
        "poisson": lambda: qualify_poisson(),
        "omega-h": lambda: qualify_omega_h(args.worker),
        "omega-h-distributed": lambda: qualify_omega_h_distributed(args.worker),
        "gmsh-metric": lambda: qualify_gmsh_metric(),
        "gmsh-boundary-layer": lambda: qualify_gmsh_boundary_layer(),
        "cad-curving": lambda: qualify_cad_curving(),
        "planar-bands": lambda: run_planar_bands(),
    }
    if name not in calls:
        raise ValueError(f"Unknown optional qualification scenario {name!r}.")
    record = calls[name]()
    record.setdefault("status", "passed")
    return record

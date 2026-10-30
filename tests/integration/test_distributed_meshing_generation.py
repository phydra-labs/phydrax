#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Three-process owner-local initial meshing generation contracts."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="requires POSIX process groups"
)


_REPOSITORY_ROOT = Path(__file__).parents[2]
_PROCESS_COUNT = 3
_PROCESS_TIMEOUT_SECONDS = 900.0


def _launch_workers(scenario: str, /) -> tuple[dict[str, Any], ...]:
    import phydrax as phx
    from phydrax._config_types import RuntimeConfig
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchPlan,
    )

    environment = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        environment.pop(name, None)
    environment = phx.config.environment(
        RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=environment,
    )
    previous_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(_REPOSITORY_ROOT)
        if not previous_path
        else os.pathsep.join((str(_REPOSITORY_ROOT), previous_path))
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{listener.getsockname()[1]}"
    results = launch_local_processes(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            scenario,
        ),
        LocalProcessLaunchPlan(_PROCESS_COUNT, address),
        environment=environment,
        cwd=str(_REPOSITORY_ROOT),
        timeout_seconds=_PROCESS_TIMEOUT_SECONDS,
    )
    records: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda value: value.process_id):
        if result.returncode != 0:
            raise RuntimeError(
                f"Distributed meshing {scenario} rank {result.process_id} exited with {result.returncode}: "
                f"{result.stderr.decode('utf-8', errors='replace')}"
            )
        lines = [
            line
            for line in result.stdout.decode("utf-8", errors="strict").splitlines()
            if line.lstrip().startswith("{")
        ]
        if not lines:
            raise RuntimeError(
                f"Distributed meshing {scenario} rank {result.process_id} returned no JSON evidence."
            )
        record = json.loads(lines[-1])
        if not isinstance(record, dict) or record.get("rank") != result.process_id:
            raise RuntimeError(
                f"Distributed meshing {scenario} rank {result.process_id} returned malformed evidence."
            )
        records.append(record)
    return tuple(records)


def _triangle_metrics(
    points: np.ndarray, triangles: np.ndarray, /
) -> tuple[float, float, float]:
    rows = np.asarray(triangles, dtype=np.int64)
    corners = points[rows]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    if points.shape[1] == 2:
        twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        area_values = 0.5 * np.abs(twice_area)
    else:
        cross = np.cross(first, second)
        area_values = 0.5 * np.linalg.norm(cross, axis=1)
    edge_01 = np.linalg.norm(corners[:, 1] - corners[:, 0], axis=1)
    edge_12 = np.linalg.norm(corners[:, 2] - corners[:, 1], axis=1)
    edge_20 = np.linalg.norm(corners[:, 0] - corners[:, 2], axis=1)
    if np.any(edge_01 <= 0.0) or np.any(edge_12 <= 0.0) or np.any(edge_20 <= 0.0):
        raise ValueError("Triangle quality requires three distinct physical vertices.")
    angles = np.stack(
        (
            np.arccos(
                np.clip(
                    (edge_01**2 + edge_20**2 - edge_12**2) / (2.0 * edge_01 * edge_20),
                    -1.0,
                    1.0,
                )
            ),
            np.arccos(
                np.clip(
                    (edge_01**2 + edge_12**2 - edge_20**2) / (2.0 * edge_01 * edge_12),
                    -1.0,
                    1.0,
                )
            ),
            np.arccos(
                np.clip(
                    (edge_12**2 + edge_20**2 - edge_01**2) / (2.0 * edge_12 * edge_20),
                    -1.0,
                    1.0,
                )
            ),
        ),
        axis=1,
    )
    edges = np.concatenate(
        (
            rows[:, (0, 1)],
            rows[:, (1, 2)],
            rows[:, (2, 0)],
        ),
        axis=0,
    )
    unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    boundary_length = np.sum(
        np.linalg.norm(
            points[boundary_edges[:, 1]] - points[boundary_edges[:, 0]], axis=1
        )
    )
    return float(np.sum(area_values)), float(np.min(angles)), float(boundary_length)


def _planar_hole_worker() -> dict[str, Any]:
    import jax

    if not jax.distributed.is_initialized():
        jax.distributed.initialize(
            coordinator_address=os.environ["PHYDRAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["PHYDRAX_NUM_PROCESSES"]),
            process_id=int(os.environ["PHYDRAX_PROCESS_ID"]),
        )
    try:
        from phydrax._execution_resources import ExecutionGroupSpec
        from phydrax._execution_runtime import ExecutionGroup
        from phydrax._physical import SpatialCoordinateContract
        from phydrax.geometry.simplicial import PlanarMeshRegion
        from phydrax.graph import GraphPartitionPlan
        from phydrax.meshing._contracts import (
            CellFamilyPolicy,
            CellMeshingTarget,
            MeshQualityTarget,
            SurfaceMeshingSpec,
        )
        from phydrax.meshing._distributed_planar_generation import (
            execute_distributed_planar_initial,
            PlanarTileSchedule,
            prepare_distributed_planar_generation,
        )
        from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
        from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
        from phydrax.meshing.providers._native_options import NativeMeshingOptions
        from phydrax.meshing.providers._native_planar import prepare_native_planar
        from phydrax.meshing.providers._native_sources import NativePlanarSource

        rank = jax.process_index()
        if jax.process_count() != _PROCESS_COUNT:
            raise RuntimeError(
                "Planar generation requires the real three-process test group."
            )
        region = PlanarMeshRegion(
            np.asarray(
                (
                    (0.0, 0.0),
                    (4.0, 0.0),
                    (4.0, 4.0),
                    (0.0, 4.0),
                    (1.4, 1.4),
                    (1.4, 1.8),
                    (1.8, 1.8),
                    (1.8, 1.4),
                ),
                dtype=np.float64,
            ),
            ((0, 1, 2, 3), (4, 5, 6, 7)),
            feature_id="distributed-planar-hole",
        )
        source = NativePlanarSource(
            region,
            "planar-r1",
            vertex_entity_ids=np.arange(101, 109, dtype=np.int64),
            edge_entity_ids=np.arange(201, 209, dtype=np.int64),
        )
        scope = MeshingScope(
            source.source_id,
            source.source_revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            source.scope_entity_ids(2),
        )
        specification = SurfaceMeshingSpec(
            CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                UniformSizeControl(scope, 0.6, strength=SizeControlStrength.SOFT),
            ),
            quality_target=MeshQualityTarget(
                minimum_angle=float(np.deg2rad(10.0)), hard=True
            ),
        )
        coordinate_contract = SpatialCoordinateContract.si()
        options = NativeMeshingOptions("planar_constrained_delaunay")
        serial_construction = prepare_native_planar(
            source,
            specification,
            options,
            coordinate_contract,
        ).execute()
        serial_mesh = serial_construction.mesh
        serial_triangles = np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int64) for block in serial_mesh.blocks
            )
        )
        serial_metrics = _triangle_metrics(
            np.asarray(serial_mesh.coordinates, dtype=np.float64),
            serial_triangles,
        )
        source_vertices = np.asarray(region.vertices, dtype=np.float64)
        source_edges = np.asarray(region.edges, dtype=np.int64)
        expected_boundary = float(
            np.sum(
                np.linalg.norm(
                    source_vertices[source_edges[:, 1]]
                    - source_vertices[source_edges[:, 0]],
                    axis=1,
                )
            )
        )
        devices = tuple(
            sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
        )
        members = tuple(sorted({device.process_index for device in devices}))
        if members != (0, 1, 2):
            raise RuntimeError(
                f"Planar generation observed unexpected process members {members}."
            )
        group = ExecutionGroup(
            ExecutionGroupSpec(
                "three-process-planar-generation",
                members,
                tuple((device.process_index, device.id) for device in devices),
                mesh_axes=(("planar_owner", len(devices)),),
            ),
            devices,
        )
        prepared = prepare_distributed_planar_generation(
            source,
            specification,
            PlanarTileSchedule(tile_span_sizes=2),
            group,
            partition_plan=GraphPartitionPlan(
                _PROCESS_COUNT,
                target_shares=np.asarray((1.0, 1.0, 1.0), dtype=np.float64),
                maximum_imbalance=2.0,
                require_nonempty=True,
            ),
        )
        generated = execute_distributed_planar_initial(prepared)
        local_metrics = _triangle_metrics(
            np.asarray(generated.coordinates, dtype=np.float64),
            np.asarray(generated.triangles, dtype=np.int64),
        )
        source_edge_mask = np.asarray(generated.edge_sources, dtype=np.int64) >= 0
        boundary_edges = np.asarray(generated.edge_vertices, dtype=np.int64)[
            source_edge_mask
        ]
        distributed_boundary = float(
            np.sum(
                np.linalg.norm(
                    generated.coordinates[boundary_edges[:, 1]]
                    - generated.coordinates[boundary_edges[:, 0]],
                    axis=1,
                )
            )
        )
        seam_edges: dict[str, set[tuple[int, int]]] = {}
        for pair, seam in zip(
            np.asarray(generated.edge_vertices, dtype=np.int64).tolist(),
            np.asarray(generated.edge_seams, dtype=np.int64).tolist(),
            strict=True,
        ):
            if seam >= 0:
                first_id = int(generated.vertex_ids[int(pair[0])])
                last_id = int(generated.vertex_ids[int(pair[1])])
                identifiers = (min(first_id, last_id), max(first_id, last_id))
                seam_edges.setdefault(str(seam), set()).add(identifiers)
        known_owner_by_tile = {
            int(tile): int(owner)
            for tile, owner in zip(
                prepared.known_tile_ids,
                prepared.known_tile_owners,
                strict=True,
            )
        }
        process_members = prepared.execution_group.spec.process_indices
        seam_owner_ranks: dict[str, list[int]] = {}
        for seam, tile_rows in enumerate(prepared.tiles.seam_tiles.tolist()):
            tile_ids = [int(prepared.tiles.tile_ids[tile_row]) for tile_row in tile_rows]
            if all(tile_id in known_owner_by_tile for tile_id in tile_ids):
                seam_owner_ranks[str(int(prepared.tiles.seam_ids[seam]))] = sorted(
                    {
                        process_members[known_owner_by_tile[tile_id]]
                        for tile_id in tile_ids
                    }
                )
        return {
            "rank": rank,
            "process_count": jax.process_count(),
            "prepared_id": prepared.prepared_id,
            "construction_id": generated.compiled_id,
            "tile_count": int(prepared.tiles.tile_ids.size),
            "local_tile_count": int(generated.local_tiles.size),
            "global_cell_count": generated.global_cell_count,
            "local_cell_count": int(generated.triangles.shape[0]),
            "local_area": local_metrics[0],
            "distributed_boundary_length": distributed_boundary,
            "distributed_minimum_angle_degrees": generated.minimum_angle_degrees,
            "serial_area": serial_metrics[0],
            "serial_boundary_length": serial_metrics[2],
            "serial_minimum_angle_degrees": float(np.rad2deg(serial_metrics[1])),
            "expected_area": 15.84,
            "expected_boundary_length": expected_boundary,
            "quality_target_degrees": 10.0,
            "seam_rounds": generated.seam_rounds,
            "seam_owner_ranks": seam_owner_ranks,
            "seam_edges": {
                seam: [list(edge) for edge in sorted(edges)]
                for seam, edges in seam_edges.items()
            },
        }
    finally:
        jax.distributed.shutdown()


def _sphere_request() -> tuple[Any, Any, Any, Any]:
    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry._meshing_domain import (
        MeshingDomain,
        MeshingDomainCurve,
        MeshingSurfacePatch,
        PatchCurveUse,
        PatchPoleUse,
    )
    from phydrax.geometry.brep._patches import LineCurve, SpherePatch
    from phydrax.meshing._contracts import (
        CellFamilyPolicy,
        CellMeshingTarget,
        SurfaceMeshingSpec,
    )
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
    from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
    from phydrax.meshing.providers._native_options import NativeMeshingOptions
    from phydrax.meshing.providers._native_schedules import NativeSurfaceSchedule

    sphere = SpherePatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    longitude_edges = tuple(index * 2.0 * np.pi / 3.0 for index in range(4))
    half_pi = 0.5 * np.pi
    patches = []
    for patch in range(3):
        lower, upper = longitude_edges[patch : patch + 2]
        step = upper - lower
        loops = (
            (
                PatchPoleUse(
                    0,
                    LineCurve((lower, -half_pi), (step, 0.0)),
                    0.0,
                    1.0,
                ),
                PatchCurveUse(
                    (patch + 1) % 3,
                    LineCurve((upper, -half_pi), (0.0, np.pi)),
                    0.0,
                    1.0,
                ),
                PatchPoleUse(
                    1,
                    LineCurve((upper, half_pi), (-step, 0.0)),
                    0.0,
                    1.0,
                ),
                PatchCurveUse(
                    patch,
                    LineCurve((lower, -half_pi), (0.0, np.pi)),
                    0.0,
                    1.0,
                    sense=-1,
                ),
            ),
        )
        patches.append(MeshingSurfacePatch(sphere, loops))
    domain = MeshingDomain(
        tuple(patches),
        tuple(MeshingDomainCurve(0, 1) for _ in range(3)),
        2,
        source_id="distributed-three-patch-sphere",
        source_revision="sphere-r1",
        tolerance=1.0e-12,
        corner_points=np.asarray(
            ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)), dtype=np.float64
        ),
    )
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.75, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=0.06,
    )
    return (
        domain,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(
                spacing_fraction=0.65,
                minimum_edge_samples=3,
            ),
        ),
        SpatialCoordinateContract.si(),
    )


def _surface_sphere_worker() -> dict[str, Any]:
    import jax

    if not jax.distributed.is_initialized():
        jax.distributed.initialize(
            coordinator_address=os.environ["PHYDRAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["PHYDRAX_NUM_PROCESSES"]),
            process_id=int(os.environ["PHYDRAX_PROCESS_ID"]),
        )
    try:
        from phydrax._execution_resources import ExecutionGroupSpec
        from phydrax._execution_runtime import ExecutionGroup
        from phydrax.graph import GraphPartitionPlan
        from phydrax.meshing._distributed_generation import (
            execute_distributed_surface_initial,
            prepare_distributed_surface_generation,
        )

        rank = jax.process_index()
        if jax.process_count() != _PROCESS_COUNT:
            raise RuntimeError(
                "Surface generation requires the real three-process test group."
            )
        domain, specification, options, coordinate_contract = _sphere_request()
        print(f"surface-worker rank={rank} phase=source-ready", file=sys.stderr, flush=True)
        devices = tuple(
            sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
        )
        members = tuple(sorted({device.process_index for device in devices}))
        if members != (0, 1, 2):
            raise RuntimeError(
                f"Surface generation observed unexpected process members {members}."
            )
        group = ExecutionGroup(
            ExecutionGroupSpec(
                "three-process-surface-generation",
                members,
                tuple((device.process_index, device.id) for device in devices),
                mesh_axes=(("surface_owner", len(devices)),),
            ),
            devices,
        )
        print(f"surface-worker rank={rank} phase=group-ready", file=sys.stderr, flush=True)
        print(
            f"surface-worker rank={rank} phase=prepare-start",
            file=sys.stderr,
            flush=True,
        )
        prepared = prepare_distributed_surface_generation(
            domain,
            specification,
            options,
            coordinate_contract,
            group,
            partition_plan=GraphPartitionPlan(
                _PROCESS_COUNT,
                target_shares=np.asarray((1.0, 1.0, 1.0), dtype=np.float64),
                require_nonempty=True,
            ),
            axis_name="sphere_surface_generation_test",
        )
        print(
            f"surface-worker rank={rank} phase=prepare-done",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"surface-worker rank={rank} phase=execute-start",
            file=sys.stderr,
            flush=True,
        )
        generated = execute_distributed_surface_initial(prepared)
        print(
            f"surface-worker rank={rank} phase=execute-done",
            file=sys.stderr,
            flush=True,
        )
        construction = generated.construction
        if construction is None:
            raise RuntimeError(
                "A nonempty graph partition produced an empty sphere owner."
            )
        local_metrics = _triangle_metrics(
            np.asarray(construction.vertices, dtype=np.float64),
            np.asarray(construction.triangles, dtype=np.int64),
        )
        shared_curve_ids = {
            int(value) for value in np.asarray(generated.shared_curve_ids).tolist()
        }
        scope_curve_ids = np.asarray(domain.scope_entity_ids(1), dtype=np.int64)
        sample_rows = {
            str(int(scope_curve_ids[curve])): [
                float(value)
                for value in np.asarray(parameters, dtype=np.float64).tolist()
            ]
            for curve, parameters in construction.curve_parameters
            if int(scope_curve_ids[curve]) in shared_curve_ids
        }
        chain_rows = dict(construction.curve_vertex_chains())
        shared_vertex_rows = {
            str(int(scope_curve_ids[curve])): [
                generated.vertex_ids[int(vertex)]
                for vertex in chain_rows[curve].tolist()
            ]
            for curve in chain_rows
            if int(scope_curve_ids[curve]) in shared_curve_ids
        }
        return {
            "rank": rank,
            "process_count": jax.process_count(),
            "prepared_id": generated.prepared_id,
            "patch_owner_ranks": np.asarray(generated.patch_owner_ranks).tolist(),
            "local_patch_count": int(
                np.count_nonzero(
                    np.asarray(generated.patch_owner_ranks, dtype=np.int32) == rank
                )
            ),
            "local_cell_count": len(generated.cell_ids),
            "local_area": local_metrics[0],
            "local_minimum_angle": local_metrics[1],
            "shared_curve_ids": sorted(shared_curve_ids),
            "shared_curve_samples": sample_rows,
            "shared_curve_vertex_ids": shared_vertex_rows,
        }
    finally:
        jax.distributed.shutdown()


def _device_cube_worker() -> dict[str, Any]:
    import jax

    if not jax.distributed.is_initialized():
        jax.distributed.initialize(
            coordinator_address=os.environ["PHYDRAX_COORDINATOR_ADDRESS"],
            num_processes=int(os.environ["PHYDRAX_NUM_PROCESSES"]),
            process_id=int(os.environ["PHYDRAX_PROCESS_ID"]),
        )
    try:
        from phydrax.meshing._device_generation import (
            DeviceGenerationLayout,
            DeviceGenerationStatus,
            evaluate_device_generation_candidates,
            execute_device_generation_round,
            prepare_device_generation,
        )

        rank = jax.process_index()
        if jax.process_count() != _PROCESS_COUNT:
            raise RuntimeError(
                "Device generation requires the real three-process test group."
            )
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            ),
            dtype=np.float64,
        )
        cells = np.asarray(
            (
                (0, 1, 2, 6),
                (0, 2, 3, 6),
                (0, 3, 7, 6),
                (0, 7, 4, 6),
                (0, 4, 5, 6),
                (0, 5, 1, 6),
            ),
            dtype=np.int32,
        )
        prepared = prepare_device_generation(
            DeviceGenerationLayout(vertex_capacity=8, cell_capacity=6),
            points=points,
            cells=cells,
            vertex_ids=np.arange(points.shape[0], dtype=np.int64),
            cell_ids=np.arange(100, 106, dtype=np.int64),
            cell_classes=np.zeros((cells.shape[0],), dtype=np.int64),
            target_sizes=np.full((cells.shape[0],), 1.1, dtype=np.float64),
        )
        seed_generation_id = prepared.generation_id
        candidates = evaluate_device_generation_candidates(prepared)
        update = execute_device_generation_round(prepared, candidates=candidates)
        evidence = update.evidence
        if evidence.status is not DeviceGenerationStatus.COMPLETE:
            raise RuntimeError(f"Cube generation failed with status {evidence.status!s}.")
        mesh = update.prepared.mesh
        active_cells = np.asarray(mesh.cell_active, dtype=np.bool_)
        active_vertices = np.asarray(mesh.vertex_active, dtype=np.bool_)
        vertex_count = int(np.count_nonzero(active_vertices))
        generated_points = np.asarray(mesh.coordinates, dtype=np.float64)[:vertex_count]
        generated_cells = np.asarray(mesh.cells, dtype=np.int32)[active_cells]
        corners = generated_points[generated_cells]
        signed_six_volumes = np.einsum(
            "ij,ij->i",
            corners[:, 1] - corners[:, 0],
            np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 0]),
        )
        volumes = signed_six_volumes / 6.0
        facet_rows = np.asarray(
            tuple(
                tuple(index for index in range(4) if index != opposite)
                for opposite in range(4)
            ),
            dtype=np.int32,
        )
        facets = generated_cells[:, facet_rows].reshape((-1, 3))
        unique_facets, facet_counts = np.unique(
            np.sort(facets, axis=1), axis=0, return_counts=True
        )
        boundary_facets = unique_facets[facet_counts == 1]
        boundary_corners = generated_points[boundary_facets]
        boundary_areas = 0.5 * np.linalg.norm(
            np.cross(
                boundary_corners[:, 1] - boundary_corners[:, 0],
                boundary_corners[:, 2] - boundary_corners[:, 0],
            ),
            axis=1,
        )
        minimum_volume = float(np.min(volumes))
        if (
            evidence.inserted_vertices != 6
            or evidence.accepted_cells != generated_cells.shape[0]
            or evidence.exact_predicate_count < generated_cells.shape[0]
            or minimum_volume <= 0.0
            or not np.isclose(np.sum(volumes), 1.0, rtol=0.0, atol=1.0e-13)
            or not np.isclose(np.sum(boundary_areas), 6.0, rtol=0.0, atol=1.0e-13)
        ):
            raise RuntimeError(
                "Committed device cube violates its volume, boundary, or exact-predicate contract."
            )
        return {
            "rank": rank,
            "process_count": jax.process_count(),
            "generation_id": update.prepared.generation_id,
            "seed_generation_id": seed_generation_id,
            "device_candidate_count": int(np.asarray(candidates.count)),
            "inserted_vertices": evidence.inserted_vertices,
            "accepted_cells": evidence.accepted_cells,
            "exact_predicate_count": evidence.exact_predicate_count,
            "volume": float(np.sum(volumes)),
            "boundary_area": float(np.sum(boundary_areas)),
            "minimum_cell_volume": minimum_volume,
        }
    finally:
        jax.distributed.shutdown()


def test_three_process_planar_generation_matches_serial_hole_invariants() -> None:
    records = _launch_workers("planar-hole")
    assert tuple(record["rank"] for record in records) == (0, 1, 2)
    assert all(record["process_count"] == _PROCESS_COUNT for record in records)
    assert len({record["prepared_id"] for record in records}) == 1
    assert len({record["construction_id"] for record in records}) == 1
    tile_count = records[0]["tile_count"]
    assert tile_count > _PROCESS_COUNT
    assert all(record["tile_count"] == tile_count for record in records)
    assert sum(record["local_tile_count"] for record in records) == tile_count
    global_cells = records[0]["global_cell_count"]
    assert all(record["global_cell_count"] == global_cells for record in records)
    assert sum(record["local_cell_count"] for record in records) == global_cells
    expected_area = 15.84
    expected_boundary = 17.6
    serial_area = records[0]["serial_area"]
    serial_boundary = records[0]["serial_boundary_length"]
    serial_quality = records[0]["serial_minimum_angle_degrees"]
    distributed_area = sum(record["local_area"] for record in records)
    distributed_boundary = sum(
        record["distributed_boundary_length"] for record in records
    )
    distributed_quality = records[0]["distributed_minimum_angle_degrees"]
    assert np.isclose(serial_area, expected_area, rtol=0.0, atol=1.0e-12)
    assert np.isclose(distributed_area, serial_area, rtol=0.0, atol=1.0e-12)
    assert np.isclose(serial_boundary, expected_boundary, rtol=0.0, atol=1.0e-12)
    assert np.isclose(distributed_boundary, serial_boundary, rtol=0.0, atol=1.0e-12)
    assert serial_quality >= 10.0
    assert distributed_quality >= 10.0
    assert all(
        np.isclose(
            record["distributed_minimum_angle_degrees"],
            distributed_quality,
            rtol=0.0,
            atol=1.0e-12,
        )
        for record in records
    )
    expected_seam_owners: dict[str, list[int]] = {}
    for record in records:
        for seam, owners in record["seam_owner_ranks"].items():
            previous = expected_seam_owners.get(seam)
            assert previous is None or previous == owners
            expected_seam_owners[seam] = owners
    assert expected_seam_owners
    for seam, owners in expected_seam_owners.items():
        copies = {
            int(record["rank"]): record["seam_edges"][seam]
            for record in records
            if seam in record["seam_edges"]
        }
        assert sorted(copies) == owners
        assert copies[owners[0]]
        assert all(copies[owner] == copies[owners[0]] for owner in owners[1:])


def test_three_process_surface_generation_reconciles_sphere_seams() -> None:
    records = _launch_workers("surface-sphere")
    assert tuple(record["rank"] for record in records) == (0, 1, 2)
    assert all(record["process_count"] == _PROCESS_COUNT for record in records)
    owner_rows = {tuple(record["patch_owner_ranks"]) for record in records}
    assert len(owner_rows) == 1
    assert set(next(iter(owner_rows))) == {0, 1, 2}
    assert all(record["local_patch_count"] == 1 for record in records)
    assert all(record["local_cell_count"] > 0 for record in records)
    assert len({record["prepared_id"] for record in records}) == 1
    from phydrax.meshing.providers._native_surface import prepare_native_surface

    domain, specification, options, contract = _sphere_request()
    serial_construction = prepare_native_surface(
        domain, specification, options, contract
    ).execute()
    serial_triangles = np.concatenate(
        tuple(
            np.asarray(block.vertices, dtype=np.int64)
            for block in serial_construction.mesh.blocks
        )
    )
    serial_area, serial_angle, serial_boundary = _triangle_metrics(
        np.asarray(serial_construction.mesh.coordinates, dtype=np.float64),
        serial_triangles,
    )
    assert serial_angle > 0.0
    assert serial_boundary <= 1.0e-12
    assert np.isclose(
        sum(record["local_area"] for record in records),
        serial_area,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.isclose(
        min(record["local_minimum_angle"] for record in records),
        serial_angle,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    for curve_id in records[0]["shared_curve_ids"]:
        copies = [
            (
                record["shared_curve_samples"][str(curve_id)],
                record["shared_curve_vertex_ids"][str(curve_id)],
            )
            for record in records
            if str(curve_id) in record["shared_curve_samples"]
        ]
        assert len(copies) == 2
        assert copies[0] == copies[1]
        assert copies[0][0][0] == 0.0 and copies[0][0][-1] == 1.0


def test_three_process_device_generation_commits_a_conforming_cube() -> None:
    records = _launch_workers("device-cube")
    assert tuple(record["rank"] for record in records) == (0, 1, 2)
    assert all(record["process_count"] == _PROCESS_COUNT for record in records)
    assert all(record["device_candidate_count"] == 6 for record in records)
    assert all(record["inserted_vertices"] == 6 for record in records)
    assert all(record["accepted_cells"] == 24 for record in records)
    assert all(record["exact_predicate_count"] >= 24 for record in records)
    assert all(record["minimum_cell_volume"] > 0.0 for record in records)
    assert all(
        np.isclose(record["volume"], 1.0, rtol=0.0, atol=1.0e-13) for record in records
    )
    assert all(
        np.isclose(record["boundary_area"], 6.0, rtol=0.0, atol=1.0e-13)
        for record in records
    )
    assert len({record["generation_id"] for record in records}) == 1
    assert all(
        record["seed_generation_id"] != record["generation_id"] for record in records
    )


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    match sys.argv[2]:
        case "device-cube":
            record = _device_cube_worker()
        case "planar-hole":
            record = _planar_hole_worker()
        case "surface-sphere":
            record = _surface_sphere_worker()
        case _:
            raise ValueError(
                f"Unknown distributed meshing worker scenario {sys.argv[2]!r}."
            )
    print(json.dumps(record, sort_keys=True))

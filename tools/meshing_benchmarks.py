#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from collections.abc import Callable
from fractions import Fraction
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from typing import Any, TypedDict

import equinox as eqx
import jax
import numpy as np

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._bvh import (
    bvh_nearest_items,
    bvh_overlap_pairs,
    bvh_overlap_pairs_host,
    BVHBuildKind,
    BVHBuildPolicy,
    prepare_bvh,
    refit_packed_bvh_bounds,
)
from phydrax._execution_sampling import process_peak_memory
from phydrax._jax_config import configuration_values, set_compilation_cache
from phydrax._meshcore import PlcRecovery3D, TetMeshArrays
from phydrax.discretization._cell_geometry_validity import (
    CellValidityCertificate,
    ComputedCellValidity,
    ValidatedCellValidityParent,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._mesh_certificates import _same_tree_content
from phydrax.geometry.brep import brep_box
from phydrax.meshing._quality import _standard_block_quality


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _timed(call: Any) -> Any:
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


def _timed_host[T](call: Callable[[], T], /) -> tuple[T, dict[str, float]]:
    started, cpu_started = perf_counter(), process_time()
    result = jax.block_until_ready(call())
    return result, {
        "wall_seconds": perf_counter() - started,
        "cpu_seconds": process_time() - cpu_started,
    }


def _validity_decisions(certificate: CellValidityCertificate, /) -> dict[str, object]:
    """Report proof depths, not an inferred interval-filter or reuse hit rate."""
    summary = certificate.stage_summary()
    total = certificate.status.shape[0]
    return {
        "cells_per_proof": total,
        "counts_by_depth": summary,
        "shares_by_depth": {
            name: {
                depth: count / total if total else 0.0
                for depth, count in sorted(counts.items())
            }
            for name, counts in summary.items()
        },
        "depth_semantics": (
            "-1: unresolved; 0: initial proof; positive: subdivision depth. "
            "Depth zero does not distinguish affine, interval, and exact screens. "
            "These are cell-decision shares, not CPU shares or proof-reuse counts."
        ),
    }


def _range(values: Any) -> dict[str, float]:
    samples = np.asarray(tuple(values), dtype="float64")
    if samples.ndim != 1 or not samples.size:
        raise ValueError("Benchmark ranges require one nonempty scalar sequence.")
    return {"minimum": float(np.min(samples)), "maximum": float(np.max(samples))}


def _integer_range(values: Any) -> dict[str, int]:
    samples = np.asarray(tuple(values), dtype=np.int64)
    if samples.ndim != 1 or not samples.size:
        raise ValueError("Benchmark ranges require one nonempty integer sequence.")
    return {"minimum": int(np.min(samples)), "maximum": int(np.max(samples))}


def _planar_region(name: str, x0: float, x1: float) -> phx.geometry.PlanarMeshRegion:
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=name,
    )


def _planar_partition(count: int) -> Any:
    contract = _contract()
    embedding = phx.geometry.PlanarEmbedding(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    region_names = tuple(f"region-{index}" for index in range(count))
    operands = tuple(
        phx.geometry.PlanarPartitionOperand(
            name,
            _planar_region(name, float(index), float(index + 1)),
            phx.geometry.BRepPartitionRole.REGION,
        )
        for index, name in enumerate(region_names)
    )
    plan = phx.geometry.PlanarPartitionPlan(
        contract,
        embedding,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=tuple(reversed(region_names))),
    )
    return embedding, plan, phx.geometry.partition_planar(plan)


def _cad_partition(count: int) -> Any:
    contract = _contract()
    region_names = tuple(f"region-{index}" for index in range(count))
    operands = tuple(
        phx.geometry.BRepPartitionOperand(
            name,
            brep_box(
                (float(index), 0.0, 0.0),
                (float(index + 1), 1.0, 1.0),
                coordinate_contract=contract,
                source_id=name,
            ),
            phx.geometry.BRepPartitionRole.REGION,
        )
        for index, name in enumerate(region_names)
    )
    plan = phx.geometry.BRepPartitionPlan(
        contract,
        operands,
        phx.geometry.BRepPartitionPolicy(region_precedence=tuple(reversed(region_names))),
    )
    return plan, phx.geometry.partition_brep(plan)


def _layout_bytes(count: int) -> bytes:
    # ty: ignore[unresolved-import]
    import gdstk

    with TemporaryDirectory(prefix="phydrax-layout-benchmark-") as temporary:
        path = Path(temporary) / "benchmark.gds"
        library = gdstk.Library(unit=1.0e-3, precision=1.0e-9)
        top = library.new_cell("TOP")
        for index in range(count):
            top.add(
                gdstk.rectangle(
                    (float(index), 0.0),
                    (float(index + 1), 1.0),
                    layer=1 + index % 2,
                    datatype=0,
                )
            )
        library.write_gds(path)
        return path.read_bytes()


def benchmark_native_meshing(resolution: int, repeats: int) -> dict[str, object]:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel()))
    lower = (
        np.arange(resolution)[:, None] * (resolution + 1) + np.arange(resolution)
    ).ravel()
    faces = np.concatenate(
        (
            np.column_stack((lower, lower + resolution + 1, lower + 1)),
            np.column_stack((lower + 1, lower + resolution + 1, lower + resolution + 2)),
        )
    )
    mesh, construction_seconds = _timed(
        lambda: phx.discretization.CellMesh.from_triangles(points, faces)
    )
    result, certification_seconds = _timed(
        lambda: phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )
    quality = jax.jit(
        lambda coordinates: phx.meshing.evaluate_cell_quality(mesh, coordinates)
    )
    started = perf_counter()
    jax.block_until_ready(quality(mesh.coordinates))
    quality_compile_seconds = perf_counter() - started
    started = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(quality(mesh.coordinates))
    quality_seconds = (perf_counter() - started) / repeats
    adaptation, refinement_seconds = _timed(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                result,
                phx.meshing.MarkedMeshAdaptation(
                    np.asarray(result.mesh.blocks[0].global_ids)[:1]
                ),
                policy=phx.meshing.MeshAdaptationPolicy(
                    phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
                ),
            )
        )
    )
    transition = adaptation.transition
    assert transition is not None and transition.vertex_stencil is not None
    transferred = transition.vertex_stencil.apply(
        result.mesh.vertex_global_ids, result.mesh.coordinates[:, 0]
    )
    error = float(
        np.max(
            np.abs(
                np.asarray(transferred)
                - np.asarray(transition.target.mesh.coordinates[:, 0])
            )
        )
    )
    if not result.audit.passed or error > 1e-12:
        raise RuntimeError(
            "Native meshing benchmark failed certification or linear transfer."
        )
    return {
        "status": "passed",
        "resolution": resolution,
        "stages_seconds": {
            "construction": construction_seconds,
            "certification": certification_seconds,
            "quality_compile": quality_compile_seconds,
            "quality_steady": quality_seconds,
            "refinement": refinement_seconds,
        },
        "identity": {
            "mesh": mesh.mesh_id,
            "certified_mesh": result.mesh.mesh_id,
            "quality_report": result.quality.report_id,
            "transition": transition.transition_id,
        },
        "quality": {
            "audit_passed": result.audit.passed,
            "minimum_measure": result.quality.minimum_measure,
            "minimum_mean_ratio": result.quality.minimum_mean_ratio,
            "linear_transfer_error": error,
        },
        "counts": {"vertices": len(points), "cells": len(faces)},
    }


def benchmark_cad_partition(resolution: int) -> dict[str, object]:
    count = max(2, resolution)
    (plan, result), total_seconds = _timed(lambda: _cad_partition(count))
    if result.topological_dimension != 3 or not result.named_solid_entity_ids:
        raise RuntimeError("CAD partition benchmark did not create solid regions.")
    return {
        "resolution": resolution,
        "stages_seconds": {"partition": total_seconds},
        "identity": {
            "plan": plan.plan_id,
            "model": result.model.model_id,
            "revision": result.revision.revision_id,
            "association_graph": result.association_graph.graph_id,
        },
        "quality": {
            "topological_dimension": result.topological_dimension,
            "solid_region_count": len(result.named_solid_entity_ids),
            "face_patch_count": len(result.named_face_entity_ids),
            "history_certificate": result.report.history_certificate_id,
        },
    }


def benchmark_planar_semantic(resolution: int) -> dict[str, object]:
    count = max(2, resolution)
    with TemporaryDirectory(prefix="phydrax-planar-semantic-benchmark-"):
        (embedding, plan, partition), partition_seconds = _timed(
            lambda: _planar_partition(count)
        )
        provider = phx.meshing.GmshProvider()
        scope = phx.meshing.MeshingScope.whole(partition.model, 2)
        regions = tuple(
            phx.meshing.RegionControl(
                phx.meshing.MeshingScope.from_geometry(
                    partition.model, region.entity_ids
                ),
                region.name,
                f"material:{region.name}",
                phx.meshing.RegionRole.SOLID,
            )
            for region in partition.regions
        )
        patches = tuple(
            phx.meshing.PatchControl(
                patch.name,
                phx.meshing.MeshingScope.from_geometry(partition.model, patch.entity_ids),
                patch.adjacent_region_ids,
            )
            for patch in partition.patches
        )
        specification = phx.meshing.SurfaceMeshingSpec(
            phx.meshing.CellMeshingTarget(
                2,
                2,
                phx.meshing.CellFamilyPolicy(required=("triangle",)),
            ),
            scope,
            planar_embedding=embedding,
            size_controls=(phx.meshing.UniformSizeControl(scope, 0.5),),
            region_controls=regions,
            patch_controls=patches,
        )
        mesh_result, meshing_seconds = _timed(
            provider.plan(partition, specification).execute
        )
    if (
        partition.topological_dimension != 2
        or partition.model.topology.num_solids
        or not mesh_result.audit.passed
        or not mesh_result.compliance.passed
    ):
        raise RuntimeError("Planar benchmark failed semantic mesh qualification.")
    return {
        "resolution": resolution,
        "stages_seconds": {
            "partition": partition_seconds,
            "gmsh_meshing": meshing_seconds,
        },
        "identity": {
            "embedding": embedding.embedding_id,
            "plan": plan.plan_id,
            "model": partition.model.model_id,
            "revision": partition.revision.revision_id,
            "association_graph": partition.association_graph.graph_id,
            "mesh_result": mesh_result.result_id,
        },
        "quality": {
            "topological_dimension": partition.topological_dimension,
            "region_entity_kind": partition.region_entity_kind,
            "patch_entity_kind": partition.patch_entity_kind,
            "region_count": len(partition.regions),
            "patch_count": len(partition.patches),
            "minimum_mean_ratio": mesh_result.quality.minimum_mean_ratio,
        },
        "counts": {
            "vertices": mesh_result.mesh.entity_set(0).count,
            "cells": mesh_result.mesh.entity_set(2).count,
        },
    }


def benchmark_layout_decode(resolution: int) -> dict[str, object]:
    input_limits = phx.interchange.ResourceLimits(
        max_bytes=1_000_000,
        max_depth=4,
        max_nodes=10_000,
        max_attributes=10_000,
        max_losses=0,
    )
    policy = phx.interchange.LayoutImportPolicy(
        phx.interchange.LayoutFormat.GDSII,
        _contract(),
        input_limits,
        top_cell="TOP",
    )
    payload, encoding_seconds = _timed(lambda: _layout_bytes(max(1, resolution)))
    result, decoding_seconds = _timed(
        lambda: phx.interchange.decode_layout_bytes(
            payload, policy, source_path="benchmark.gds"
        )
    )
    if (
        not result.report.valid
        or result.source_digest != result.resource_manifest.content_sha256
    ):
        raise RuntimeError("Layout benchmark lost decoder validity or source identity.")
    return {
        "resolution": resolution,
        "stages_seconds": {
            "fixture_encoding": encoding_seconds,
            "decode": decoding_seconds,
        },
        "identity": {
            "layout_model": result.model.model_id,
            "source_digest": result.source_digest,
            "manifest": result.resource_manifest.manifest_id,
        },
        "quality": {
            "report_valid": result.report.valid,
            "region_count": len(result.model.regions),
            "coordinate_contract": result.model.coordinate_contract.spatial_id,
            "observed_losses": result.resource_manifest.observed_losses,
        },
    }


def benchmark_planar_band(resolution: int) -> dict[str, object]:
    embedding, _plan, partition = _planar_partition(2)
    patch = next(
        value for value in partition.patches if len(value.adjacent_region_ids) == 2
    )
    schedule = phx.meshing.LayerSchedule.geometric(
        max(1, resolution), 0.02, growth_rate=1.0
    )
    control = phx.meshing.PlanarBandControl(
        patch.name,
        {name: schedule for name in patch.adjacent_region_ids},
        0.1,
    )
    plan = phx.meshing.PlanarBandPlan(partition, embedding, (control,))
    band_result, preparation_seconds = _timed(
        lambda: phx.meshing.prepare_planar_bands(plan)
    )
    provider = phx.meshing.GmshProvider()
    scope = phx.meshing.MeshingScope.whole(band_result.partition.model, 2)
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope.from_geometry(
                band_result.partition.model, region.entity_ids
            ),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in band_result.partition.regions
    )
    patches = tuple(
        phx.meshing.PatchControl(
            value.name,
            phx.meshing.MeshingScope.from_geometry(
                band_result.partition.model, value.entity_ids
            ),
            value.adjacent_region_ids,
        )
        for value in band_result.partition.patches
        if len(set(value.adjacent_region_ids)) == len(value.adjacent_region_ids)
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
        region_controls=regions,
        patch_controls=patches,
    )
    mesh_result, meshing_seconds = _timed(
        provider.plan(band_result, specification).execute
    )
    if (
        not band_result.layer_partitions
        or not mesh_result.audit.passed
        or not mesh_result.compliance.passed
    ):
        raise RuntimeError("Planar-band benchmark failed exact mesh qualification.")
    achieved = dict(mesh_result.compliance.achieved)
    residuals = tuple(
        achieved[
            f"planar_band:{layer.control_id}:{layer.region_name}:front:{layer.layer_index + 1}:maximum_residual"
        ]
        for layer in band_result.layer_partitions
    )
    return {
        "status": "passed",
        "resolution": resolution,
        "stages_seconds": {
            "in_memory_preparation": preparation_seconds,
            "gmsh_meshing": meshing_seconds,
        },
        "identity": {
            "plan": band_result.plan_id,
            "result": band_result.result_id,
            "mesh_result": mesh_result.result_id,
            "partition_revision": band_result.partition.revision.revision_id,
        },
        "quality": {
            "layer_partition_count": len(band_result.layer_partitions),
            "schedule": schedule.schedule_id,
            "control": control.control_id,
            "maximum_front_residual": max(residuals),
            "minimum_mean_ratio": mesh_result.quality.minimum_mean_ratio,
        },
        "counts": {
            "vertices": mesh_result.mesh.entity_set(0).count,
            "cells": mesh_result.mesh.entity_set(2).count,
        },
    }


def benchmark_hybrid_slab(resolution: int) -> dict[str, object]:
    from tools.meshing_qualification.scenarios.providers import _hybrid_gmsh_result

    layer_count = max(1, resolution)
    schedule = phx.meshing.LayerSchedule.geometric(
        layer_count,
        0.3 / layer_count,
        growth_rate=1.0,
    )
    with TemporaryDirectory(prefix="phydrax-hybrid-slab-benchmark-") as temporary:
        (result, control), generation_seconds = _timed(
            lambda: _hybrid_gmsh_result(
                Path(temporary) / "hybrid-slab.brep",
                schedule=schedule,
            )
        )
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
            for block in result.mesh.blocks
        },
    )
    discretization, preparation_seconds = _timed(
        lambda: phx.discretization.FiniteElementPlan(result.mesh, field).prepare()
    )
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError("Hybrid slab benchmark failed mesh audit or compliance.")
    achieved = dict(result.compliance.achieved)
    return {
        "resolution": resolution,
        "stages_seconds": {
            "gmsh_generation": generation_seconds,
            "p1_preparation": preparation_seconds,
        },
        "identity": {
            "mesh": result.mesh.mesh_id,
            "result": result.result_id,
            "quality_report": result.quality.report_id,
            "schedule": control.schedule.schedule_id,
            "prepared": discretization.prepared_id,
        },
        "quality": {
            "audit_passed": result.audit.passed,
            "minimum_measure": result.quality.minimum_measure,
            "minimum_mean_ratio": result.quality.minimum_mean_ratio,
            "block_kinds": [block.cell_kind for block in result.mesh.blocks],
            "maximum_thickness_residual": achieved[
                f"layer:{control.control_id}:maximum_thickness_residual"
            ],
        },
        "counts": {
            "vertices": result.mesh.coordinates.shape[0],
            "cells": result.mesh.entity_set(3).count,
            "dofs": discretization.dof_maps[0].global_dof_count,
        },
    }


@eqx.filter_jit
def _realize_implicit_surface(plan: Any, state: Any) -> Any:
    return plan.realize(state)


def benchmark_implicit_discovery(resolution: int) -> dict[str, object]:
    """Scale dual-contouring discovery with the lattice points per axis."""

    points_per_axis = 4 * resolution + 1
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(points_per_axis) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jax.numpy.asarray([[-1.4, -1.4, -1.4], [1.4, 1.4, 1.4]]))
    geometry = phx.geometry.Sphere(
        (0.013, -0.007, 0.011),
        0.75,
        feature_id="benchmark-sphere",
    ).compile()
    plan, discovery_seconds = _timed(
        lambda: phx.geometry.discover_implicit_surface(
            geometry,
            grid,
            policy=phx.geometry.ImplicitSurfacePolicy(
                projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            ),
            source_id="benchmark-sphere",
        )
    )
    cold, cold_seconds = _timed(
        lambda: jax.block_until_ready(_realize_implicit_surface(plan, geometry.state))
    )
    warm, warm_seconds = _timed(
        lambda: jax.block_until_ready(_realize_implicit_surface(plan, geometry.state))
    )
    if not bool(np.asarray(cold.accepted)) or not bool(np.asarray(warm.accepted)):
        raise RuntimeError("Implicit discovery benchmark realization was rejected.")
    return {
        "resolution": resolution,
        "stages_seconds": {
            "discovery": discovery_seconds,
            "realization_cold": cold_seconds,
            "realization_warm": warm_seconds,
        },
        "identity": {
            "plan": plan.plan_id,
            "topology": plan.topology_id,
        },
        "quality": {
            "minimum_face_area": float(np.asarray(warm.evidence.minimum_face_area)),
            "minimum_orientation_margin": float(
                np.asarray(warm.evidence.minimum_orientation_margin)
            ),
        },
        "counts": {
            "lattice_points": points_per_axis**3,
            "crossings": plan.projection.anchors.shape[0],
            "vertices": plan.base_vertices.shape[0],
            "faces": plan.faces.shape[0],
            "intersection_candidates": plan.intersection_pairs.shape[0],
        },
    }


def _clustered_boxes(count: int, seed: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Boxes around Gaussian clusters, so hierarchy quality matters."""
    rng = np.random.default_rng(seed)
    centers = rng.random((max(1, count // 256), 3))
    lower = centers[rng.integers(0, centers.shape[0], count)] + 0.02 * (
        rng.standard_normal((count, 3))
    )
    return lower, lower + 0.004 + 0.004 * rng.random((count, 3))


@eqx.filter_jit
def _refit_and_query(bvh: Any, lower: Any, upper: Any, queries: Any) -> Any:
    refitted = refit_packed_bvh_bounds(bvh, lower, upper)
    return bvh_nearest_items(refitted, queries, k=4)


def benchmark_lbvh(resolution: int) -> dict[str, object]:
    """Scale every BVH build strategy, refit, and exact kNN with the item count."""
    item_count = 1024 * resolution
    lower, upper = _clustered_boxes(item_count, resolution)
    moved_lower = lower + 0.001
    queries = np.random.default_rng(7).random((1024, 3))
    stages: dict[str, float] = {}
    quality: dict[str, int] = {}
    for kind in BVHBuildKind:
        policy = BVHBuildPolicy(kind, leaf_size=8)
        bvh, stages[f"build_{kind.value}"] = _timed(
            lambda policy=policy: prepare_bvh(lower, upper, policy=policy)
        )
        _, stages[f"refit_query_cold_{kind.value}"] = _timed(
            lambda bvh=bvh: jax.block_until_ready(
                _refit_and_query(bvh, moved_lower, moved_lower + 0.006, queries)
            )
        )
        _, stages[f"refit_query_warm_{kind.value}"] = _timed(
            lambda bvh=bvh: jax.block_until_ready(
                _refit_and_query(bvh, moved_lower, moved_lower + 0.006, queries)
            )
        )
        quality[f"depth_{kind.value}"] = bvh.max_depth
        quality[f"nodes_{kind.value}"] = bvh.node_count
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "quality": quality,
        "counts": {"items": item_count, "queries": queries.shape[0]},
    }


@eqx.filter_jit
def _device_overlap_pairs(first: Any, second: Any, *, capacity: Any) -> Any:
    return bvh_overlap_pairs(first, second, capacity=capacity)


def benchmark_overlap_pairs(resolution: int) -> dict[str, object]:
    """Scale exact host and bounded device BVH pair search with the item count."""
    item_count = 1024 * resolution
    first_lower, first_upper = _clustered_boxes(item_count, resolution)
    # The second set jitters the first, so both share clusters and overlap densely.
    second_lower = first_lower + 0.004 * np.random.default_rng(
        resolution
    ).standard_normal(first_lower.shape)
    second_upper = second_lower + (first_upper - first_lower)
    policy = BVHBuildPolicy(BVHBuildKind.SAH, leaf_size=8)
    first = prepare_bvh(first_lower, first_upper, policy=policy, dtype=np.float64)
    second = prepare_bvh(second_lower, second_upper, policy=policy, dtype=np.float64)
    (host_first, _), host_seconds = _timed(lambda: bvh_overlap_pairs_host(first, second))
    capacity = max(1, 2 * host_first.shape[0])
    _, device_cold = _timed(
        lambda: jax.block_until_ready(
            _device_overlap_pairs(first, second, capacity=capacity)
        )
    )
    device, device_warm = _timed(
        lambda: jax.block_until_ready(
            _device_overlap_pairs(first, second, capacity=capacity)
        )
    )
    if bool(device.overflow) or int(device.count) != host_first.shape[0]:
        raise RuntimeError("Device overlap pairs disagree with the exact host search.")
    return {
        "resolution": resolution,
        "stages_seconds": {
            "host": host_seconds,
            "device_cold": device_cold,
            "device_warm": device_warm,
        },
        "counts": {"items": item_count, "pairs": host_first.shape[0]},
    }


_HEXAHEDRON_CORNERS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
)
_KUHN_TETRAHEDRA = np.asarray(
    ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
)


def _kuhn_tetrahedra(cells_per_axis: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Positively oriented Kuhn tetrahedra of a unit-cube lattice."""
    count = cells_per_axis
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    index = np.arange((count + 1) ** 3, dtype=np.int32).reshape((count + 1,) * 3)
    corners = np.stack(
        [
            index[a : a + count, b : b + count, c : c + count].reshape((-1,))
            for a, b, c in _HEXAHEDRON_CORNERS
        ],
        axis=1,
    )
    tetrahedra = corners[:, _KUHN_TETRAHEDRA].reshape((-1, 4))
    first, second, third, fourth = (points[tetrahedra[:, i]] for i in range(4))
    volume = np.sum(np.cross(second - first, third - first) * (fourth - first), axis=1)
    tetrahedra[volume < 0.0] = tetrahedra[volume < 0.0][:, [1, 0, 2, 3]]
    return points, tetrahedra


def _planar_triangles(cells_per_axis: int, /) -> np.ndarray:
    """Counterclockwise triangles of a square lattice disk."""
    index = np.arange((cells_per_axis + 1) ** 2, dtype=np.int32).reshape(
        (cells_per_axis + 1, cells_per_axis + 1)
    )
    lower = index[:-1, :-1].reshape((-1,))
    right = index[1:, :-1].reshape((-1,))
    upper = index[1:, 1:].reshape((-1,))
    left = index[:-1, 1:].reshape((-1,))
    return np.concatenate(
        (np.stack((lower, right, upper), axis=1), np.stack((lower, upper, left), axis=1))
    )


def _array_bytes(tree: Any) -> int:
    """Logical bytes in distinct retained array leaves, excluding runtime overhead."""
    leaves = {id(leaf): leaf for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))}
    return sum(leaf.nbytes for leaf in leaves.values())


def benchmark_incidence(resolution: int) -> dict[str, object]:
    """Scale canonical incidence construction with about `resolution` cells."""
    points, tetrahedra = _kuhn_tetrahedra(max(1, round((resolution / 6) ** (1 / 3))))
    triangles = _planar_triangles(max(1, round((resolution / 2) ** 0.5)))
    vertex_count = points.shape[0]
    connectivity, connectivity_seconds = _timed(
        lambda: phx.discretization.tetrahedral_connectivity(tetrahedra, vertex_count)
    )
    # The first complex at a new size also compiles its shape-specialized
    # relation checks; the warm build isolates construction.
    _, topology_cold_seconds = _timed(
        lambda: phx.discretization.tetrahedral_cell_complex(tetrahedra, vertex_count)
    )
    topology, topology_seconds = _timed(
        lambda: phx.discretization.tetrahedral_cell_complex(tetrahedra, vertex_count)
    )
    mesh, mesh_seconds = _timed(
        lambda: phx.discretization.CellMesh(
            points,
            (phx.discretization.CellBlock("tetrahedra", "tetrahedron", tetrahedra),),
        )
    )
    moved, moved_seconds = _timed(
        lambda: mesh.with_coordinates(1.5 * points, numeric_version="1")
    )
    if moved.topology_id != mesh.topology_id:
        raise RuntimeError("Coordinate refresh changed the mesh topology.")
    surface, surface_seconds = _timed(
        lambda: phx.geometry.simplicial.TriangleTopology(triangles)
    )
    return {
        "resolution": resolution,
        "stages_seconds": {
            "tetrahedral_connectivity": connectivity_seconds,
            "tetrahedral_cell_complex_cold": topology_cold_seconds,
            "tetrahedral_cell_complex": topology_seconds,
            "cell_mesh": mesh_seconds,
            "with_coordinates": moved_seconds,
            "triangle_topology": surface_seconds,
        },
        "retained_bytes": {
            "tetrahedral_connectivity": _array_bytes(connectivity),
            "tetrahedral_cell_complex": _array_bytes(topology),
            "cell_mesh": _array_bytes(mesh),
            "triangle_topology": _array_bytes(surface),
        },
        "counts": {
            "vertices": vertex_count,
            "tetrahedra": tetrahedra.shape[0],
            "faces": np.asarray(connectivity.faces).shape[0],
            "edges": np.asarray(connectivity.edges).shape[0],
            "triangles": triangles.shape[0],
            "triangle_boundary_loops": surface.boundary_loop_offsets.shape[0] - 1,
        },
    }


def _jittered_unit_square_triangles(cells_per_axis: int, seed: int, /) -> Any:
    """Counterclockwise lattice triangles with interior vertices jittered in place."""
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    jitter = np.random.default_rng(seed).uniform(-0.2, 0.2, points.shape)
    points[interior] += jitter[interior] / cells_per_axis
    return phx.discretization.CellMesh.from_triangles(
        points, _planar_triangles(cells_per_axis)
    )


def _cube_tetrahedra(cells_per_axis: int, /) -> Any:
    points, tetrahedra = _kuhn_tetrahedra(cells_per_axis)
    return phx.discretization.CellMesh.from_tetrahedra(points, tetrahedra)


def benchmark_supermesh(resolution: int) -> dict[str, object]:
    """Scale certified common refinement with about `resolution` cells per mesh.

    Both dimensions refine two non-matching meshes of the unit square/cube: a
    jittered triangle lattice against a coarser one, and Kuhn tetrahedra against
    a coarser Kuhn lattice.
    """
    planar = max(2, round((resolution / 2) ** 0.5))
    spatial = max(2, round((resolution / 6) ** (1 / 3)))
    meshes = {
        "2d": (
            _jittered_unit_square_triangles(planar, 1),
            _jittered_unit_square_triangles(max(1, planar - 1), 2),
        ),
        "3d": (_cube_tetrahedra(spatial), _cube_tetrahedra(max(1, spatial - 1))),
    }
    stages, counts = {}, {}
    for name, (source, target) in meshes.items():
        refinement, seconds = _timed(
            lambda source=source, target=target: phx.geometry.prepare_common_refinement(
                source, target
            )
        )
        if not refinement.succeeded:
            raise RuntimeError(f"{name} common refinement failed: {refinement.status}.")
        evidence = refinement.evidence
        stages[name] = seconds
        counts[name] = {
            "source_cells": refinement.source_cell_count,
            "target_cells": refinement.target_cell_count,
            "candidate_pairs": evidence.candidate_pair_count,
            "piece_pairs": evidence.piece_pair_count,
            "entries": refinement.entry_count,
            "retained_bytes": evidence.retained_bytes,
            "working_bytes": evidence.working_bytes,
            "maximum_relative_defect": max(
                evidence.maximum_relative_source_defect,
                evidence.maximum_relative_target_defect,
            ),
        }
    return {"resolution": resolution, "stages_seconds": stages, "counts": counts}


def _remap_field(points: np.ndarray, /) -> np.ndarray:
    x, y = points[..., 0], points[..., 1]
    return np.sin(2.0 * np.pi * x) * np.cos(np.pi * y) + x * x


def _finite_volume_averages(discretization: Any, /) -> np.ndarray:
    weights = np.where(
        np.asarray(discretization.cell_quadrature_valid),
        np.asarray(discretization.cell_quadrature_weights),
        0.0,
    )
    values = _remap_field(np.asarray(discretization.cell_quadrature_points))
    return np.sum(weights * values, axis=1) / np.sum(weights, axis=1)


@eqx.filter_jit
def _first_order_remap(plan: Any, values: Any) -> Any:
    return plan.apply(values)


@eqx.filter_jit
def _second_order_remap(plan: Any, values: Any) -> Any:
    return plan.apply(values)


def _p1_space(mesh: Any, /) -> Any:
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()


def _overlap_refinement(source_mesh: Any, target_mesh: Any, /) -> Any:
    refinement = phx.geometry.prepare_common_refinement(
        source_mesh,
        target_mesh,
        policy=phx.geometry.CommonRefinementPolicy(overlap_simplices=True),
    )
    if not refinement.succeeded:
        raise RuntimeError(f"L2 projection refinement failed: {refinement.status}.")
    return refinement


def _l2_projection_residuals(
    source: Any, target: Any, transfer: Any, /
) -> dict[str, float]:
    """Conservation, Galerkin orthogonality, and transpose duality of one transfer."""
    values = jax.numpy.asarray(
        _remap_field(np.asarray(source.dof_maps[0].dof_coordinates))
    )
    dual = jax.numpy.asarray(
        np.random.default_rng(0).normal(size=(transfer.target_size,))
    )
    projected = transfer.apply(values)
    source_mass = np.asarray(source.mass.mv(jax.numpy.ones((transfer.source_size,))))
    target_mass = np.asarray(target.mass.mv(jax.numpy.ones((transfer.target_size,))))
    integral = float(source_mass @ np.asarray(values))
    mixed = np.asarray(transfer.primal.mixed_mass.mv(values))
    galerkin = mixed - np.asarray(target.mass.mv(projected))
    pairing = float(jax.numpy.vdot(projected, dual))
    return {
        "conservation_relative": abs(
            float(target_mass @ np.asarray(projected)) - integral
        )
        / abs(integral),
        "galerkin_orthogonality_relative": float(
            np.max(np.abs(galerkin)) / np.max(np.abs(mixed))
        ),
        "transpose_duality_relative": abs(
            pairing - float(jax.numpy.vdot(values, transfer.pullback(dual)))
        )
        / float(jax.numpy.linalg.norm(projected) * jax.numpy.linalg.norm(dual)),
    }


_L2_PROJECTION_KERNELS = (
    "_factor_target_mass",
    "_mass_condition",
    "_mixed_mass_action",
    "_target_mass_solve",
    "_tabulate_values",
)


def _benchmark_l2_projection(
    target_mesh: Any, moved_mesh: Any, source_mesh: Any, /
) -> dict:
    """Prepared-target P1 Galerkin L2 projection stages onto a triangle target.

    The target is prepared cold (symbolic and numeric factorization), refreshed
    for ``moved_mesh`` (same topology, moved interior vertices) on the retained
    symbolic plan, and shared by the transfers from ``source_mesh``.
    ``refresh_new_compilations`` counts the compilations the refresh and the
    refreshed transfer add to the module-level factorization, condition,
    target-solve, and bucketed tabulation kernels (zero when the refresh reuses
    every compiled kernel).
    """
    from phydrax.discretization.fem import _topology_transfer as projection_kernels

    def kernel_compilations() -> Any:
        return {
            name: getattr(projection_kernels, name)._cached._cache_size()
            for name in _L2_PROJECTION_KERNELS
        }

    source, target, moved = map(_p1_space, (source_mesh, target_mesh, moved_mesh))
    refinements = (
        _overlap_refinement(source_mesh, target_mesh),
        _overlap_refinement(source_mesh, moved_mesh),
    )
    prepared, target_prepare = _timed(
        lambda: phx.discretization.prepare_l2_projection_target(target, field_name="u")
    )
    transfer, transfer_prepare = _timed(
        lambda: phx.discretization.prepare_l2_projection_transfer(
            source, prepared, refinements[0], field_name="u"
        )
    )
    points = np.asarray(source.dof_maps[0].dof_coordinates)
    block = jax.numpy.asarray(
        np.stack((_remap_field(points), points[:, 0], np.ones(points.shape[0])), -1)
    )
    mixed_mass = transfer.primal.mixed_mass
    compiled = {
        "l2_mixed_mass_action_block3": _compiled_cost(
            projection_kernels._mixed_mass_action,
            mixed_mass.relation,
            mixed_mass.coefficients,
            block,
        ),
        "l2_target_solve_block3": _compiled_cost(
            projection_kernels._target_mass_solve,
            prepared.factorization,
            jax.numpy.zeros((transfer.target_size, block.shape[1])),
        ),
    }
    _, action_first = _timed(lambda: jax.block_until_ready(transfer.apply(block)))
    _, action_warm = _timed(lambda: jax.block_until_ready(transfer.apply(block)))
    # Snapshot only after every original-target action has compiled and run.
    before_refresh = kernel_compilations()
    refreshed, target_refresh = _timed(
        lambda: phx.discretization.refresh_l2_projection_target(prepared, moved)
    )
    refreshed_transfer, refreshed_transfer_prepare = _timed(
        lambda: phx.discretization.prepare_l2_projection_transfer(
            source, refreshed, refinements[1], field_name="u"
        )
    )
    _, refreshed_action_first = _timed(
        lambda: jax.block_until_ready(refreshed_transfer.apply(block))
    )
    _, refreshed_action_warm = _timed(
        lambda: jax.block_until_ready(refreshed_transfer.apply(block))
    )
    compilations = {
        name: count - before_refresh[name]
        for name, count in kernel_compilations().items()
    }
    residuals = (
        _l2_projection_residuals(source, target, transfer),
        _l2_projection_residuals(source, moved, refreshed_transfer),
    )
    return {
        "stages_seconds": {
            "l2_target_prepare": target_prepare,
            "l2_transfer_prepare": transfer_prepare,
            "l2_target_refresh": target_refresh,
            "l2_refreshed_transfer_prepare": refreshed_transfer_prepare,
            "l2_action_first": action_first,
            "l2_action_warm": action_warm,
            "l2_refreshed_action_first": refreshed_action_first,
            "l2_refreshed_action_warm": refreshed_action_warm,
        },
        "compiled": compiled,
        "refresh_new_compilations": compilations,
        "counts": {
            "l2_source_dofs": transfer.source_size,
            "l2_target_dofs": transfer.target_size,
            "l2_mixed_mass_entries": mixed_mass.relation.capacity,
            "l2_target_mass_entries": prepared.mass.relation.capacity,
            "l2_factor_entries": int(prepared.factorization.diagnostics.factor_nonzeros),
        },
        "retained_bytes": {
            "l2_factor_values": prepared.factorization.factor_values.nbytes,
            "l2_factorization_plan": _array_bytes(prepared.factorization.plan),
            "l2_mixed_mass": _array_bytes(mixed_mass),
            "l2_refinement": refinements[0].evidence.retained_bytes,
        },
        "errors": {
            f"l2_{name}": max(record[name] for record in residuals)
            for name in residuals[0]
        },
    }


def benchmark_remap(resolution: int) -> dict[str, object]:
    """Scale P0 and limited P1 conservative remap with about `resolution` cells.

    A smooth field moves from a structured quadrilateral mesh to a jittered
    triangle mesh of the unit square; errors are volume-weighted L1 errors of the
    target averages. The P1 Galerkin L2 projection onto the same triangle target
    (from a finer jittered triangle source) reports cold and refreshed target
    preparation, transfer assembly, first and warmed actions, factor bytes, and
    retained mixed-mass bytes (``l2_*`` entries).
    """
    cells = max(2, round(resolution**0.5))
    axis = np.linspace(0.0, 1.0, cells + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    index = np.arange((cells + 1) ** 2, dtype=np.int32).reshape((cells + 1,) * 2)
    quadrilaterals = np.stack(
        (index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1
    ).reshape((-1, 4))
    source = phx.discretization.UnstructuredFiniteVolumePlan(
        points, quadrilaterals=quadrilaterals
    ).prepare()
    triangle_cells = max(2, round((resolution / 2) ** 0.5))
    triangles = _jittered_unit_square_triangles(triangle_cells, 3)
    target = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(triangles.coordinates),
        triangles=np.asarray(triangles.blocks[0].vertices),
    ).prepare()
    values = jax.numpy.asarray(_finite_volume_averages(source))
    exact = _finite_volume_averages(target)
    volumes = np.asarray(target.cell_volumes)
    remap, remap_seconds = _timed(
        lambda: phx.discretization.prepare_unstructured_conservative_remap(
            source, target, provenance="meshing-benchmark"
        )
    )
    if not remap.succeeded:
        raise RuntimeError(f"Remap preparation failed: {remap.status}.")
    _, first_cold = _timed(
        lambda: jax.block_until_ready(_first_order_remap(remap.plan, values))
    )
    first, first_warm = _timed(
        lambda: jax.block_until_ready(_first_order_remap(remap.plan, values))
    )

    def second_order_plan() -> Any:
        return phx.discretization.UnstructuredSecondOrderRemapPlan(
            remap.plan, remap.refinement, source
        )

    _, second_prepare_cold = _timed(second_order_plan)
    plan, second_prepare_warm = _timed(second_order_plan)
    _, second_cold = _timed(
        lambda: jax.block_until_ready(_second_order_remap(plan, values))
    )
    second, second_warm = _timed(
        lambda: jax.block_until_ready(_second_order_remap(plan, values))
    )
    projection = _benchmark_l2_projection(
        triangles,
        _jittered_unit_square_triangles(triangle_cells, 4),
        _jittered_unit_square_triangles(triangle_cells + 2, 1),
    )
    scale = float(np.sum(volumes * np.abs(exact)))
    return {
        "resolution": resolution,
        "stages_seconds": {
            "refinement_and_plan": remap_seconds,
            "first_order_cold": first_cold,
            "first_order_warm": first_warm,
            "second_order_prepare_cold": second_prepare_cold,
            "second_order_prepare_warm": second_prepare_warm,
            "second_order_cold": second_cold,
            "second_order_warm": second_warm,
            **projection["stages_seconds"],
        },
        "compiled": projection["compiled"],
        "refresh_new_compilations": projection["refresh_new_compilations"],
        "retained_bytes": projection["retained_bytes"],
        "counts": {
            "source_cells": source.cell_count,
            "target_cells": target.cell_count,
            "entries": remap.refinement.entry_count,
            "limited_cells": int(second.limited_count),
            **projection["counts"],
        },
        "errors": {
            "first_order_relative_l1": float(
                np.sum(volumes * np.abs(np.asarray(first) - exact)) / scale
            ),
            "second_order_relative_l1": float(
                np.sum(volumes * np.abs(np.asarray(second.values) - exact)) / scale
            ),
            "second_order_conservation_residual": float(
                np.max(np.abs(np.asarray(second.conservation_residual_after)))
            ),
            **projection["errors"],
        },
    }


def _semantic_gmsh_specification(provider: Any, partition: Any) -> Any:
    source = partition.model
    whole = phx.meshing.MeshingScope.whole(source, 3)
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope.from_geometry(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in partition.regions
    )
    patches = tuple(
        phx.meshing.PatchControl(
            patch.name,
            phx.meshing.MeshingScope.from_geometry(source, patch.entity_ids),
            patch.adjacent_region_ids,
        )
        for patch in partition.patches
    )
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        whole,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                whole,
                0.8,
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        region_controls=regions,
        patch_controls=patches,
    )


def _semantic_gmsh_plan(provider: Any, partition: Any) -> Any:
    specification = _semantic_gmsh_specification(provider, partition)
    return specification, provider.plan(partition.model, specification)


def benchmark_gmsh_semantic(
    region_count: int,
    repeats: int,
    /,
) -> dict[str, object]:
    provider = phx.meshing.GmshProvider()
    timings = {"partition": [], "plan": [], "execute": []}
    samples = []
    for repeat in range(repeats):
        jax.clear_caches()
        (_, partition), partition_seconds = _timed(lambda: _cad_partition(region_count))
        timings["partition"].append(partition_seconds)
        (_specification, plan), plan_seconds = _timed(
            lambda partition=partition: _semantic_gmsh_plan(provider, partition)
        )
        timings["plan"].append(plan_seconds)
        result, execute_seconds = _timed(plan.execute)
        timings["execute"].append(execute_seconds)
        if not result.audit.passed or not result.compliance.passed:
            raise RuntimeError("Semantic Gmsh benchmark failed audit or compliance.")
        samples.append(
            {
                "mesh": result.mesh.mesh_id,
                "result": result.result_id,
                "plan": plan.plan_id,
                "source_revision": partition.model.source_revision,
                "cells": result.mesh.entity_set(3).count,
                "regions": len(
                    tuple(
                        zone
                        for zone in result.zones
                        if zone.role is phx.meshing.MeshZoneRole.REGION
                    )
                ),
                "patches": len(result.patches),
                "minimum_jacobian": float(
                    dict(result.compliance.achieved)[
                        "minimum_curved_jacobian_determinant"
                    ]
                ),
            }
        )
    return {
        "status": "passed",
        "regions": region_count,
        "repeats": repeats,
        "timings_seconds": {
            name: {"samples": values, "range": _range(values)}
            for name, values in timings.items()
        },
        "samples": samples,
        "ranges": {
            "cells": _integer_range([sample["cells"] for sample in samples]),
            "regions": _integer_range([sample["regions"] for sample in samples]),
            "patches": _integer_range([sample["patches"] for sample in samples]),
            "minimum_jacobian": _range(
                [sample["minimum_jacobian"] for sample in samples]
            ),
        },
    }


def _benchmark_gmsh_semantic_case(
    resolution: int,
    repeats: int,
    /,
) -> dict[str, object]:
    return benchmark_gmsh_semantic(max(2, resolution), repeats)


def _comparison_result_record(
    provider_name: str, plan: Any, result: Any, partition: Any, /
) -> dict[str, object]:
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError(f"{provider_name} comparison failed audit or compliance.")
    region_count = sum(
        zone.role is phx.meshing.MeshZoneRole.REGION for zone in result.zones
    )
    if region_count != len(partition.regions):
        raise RuntimeError(f"{provider_name} comparison lost semantic source regions.")
    measured_volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    expected_volume = float(len(partition.regions))
    relative_volume_error = abs(measured_volume - expected_volume) / expected_volume
    if relative_volume_error > 5.0e-8:
        raise RuntimeError(
            f"{provider_name} comparison lost source volume: {relative_volume_error:.6g}."
        )
    runtime = getattr(result, "runtime", None)
    return {
        "provider": provider_name,
        "provider_version": getattr(runtime, "actual_version", None),
        "plan": plan.plan_id,
        "resolved_route": getattr(
            getattr(plan, "route", None),
            "value",
            str(getattr(plan, "route", "")),
        ),
        "result": result.result_id,
        "audit_passed": result.audit.passed,
        "compliance_passed": result.compliance.passed,
        "source_revision": partition.model.source_revision,
        "regions": region_count,
        "patches": len(result.patches),
        "cells": result.mesh.entity_set(3).count,
        "measured_volume": measured_volume,
        "relative_volume_error": relative_volume_error,
        "minimum_mean_ratio": result.quality.minimum_mean_ratio,
        "logical_retained_array_bytes": _array_bytes(result),
    }


def benchmark_native_gmsh_comparison(resolution: int, repeats: int) -> dict[str, object]:
    """Compare native and optional Gmsh plans on identical partitioned CAD."""

    if resolution < 1 or repeats < 1:
        raise ValueError("resolution and repeats must be positive.")
    missing = _missing_gmsh_dependencies()
    if missing:
        return {
            "scenario": "native-versus-gmsh-cad-meshing",
            "status": "missing_dependency",
            "resolution": resolution,
            "repeats": repeats,
            "missing_dependencies": missing,
            "native_fallback": False,
        }
    from phydrax._meshcore import meshcore_identity

    region_count = max(2, resolution)
    native_provider = phx.meshing.NativeMeshingProvider()
    gmsh_provider = phx.meshing.GmshProvider()
    timings = {
        "partition": [],
        "request_preparation": [],
        "native_plan_preparation": [],
        "native_execution": [],
        "gmsh_plan_preparation": [],
        "gmsh_execution": [],
    }
    samples = []
    for _ in range(repeats):
        jax.clear_caches()
        partition, elapsed = _timed(lambda: _cad_partition(region_count)[1])
        timings["partition"].append(elapsed)
        specification, elapsed = _timed(
            lambda partition=partition: _semantic_gmsh_specification(
                gmsh_provider, partition
            )
        )
        timings["request_preparation"].append(elapsed)
        native_plan, elapsed = _timed(
            lambda partition=partition, specification=specification: native_provider.plan(
                partition.model, specification
            )
        )
        timings["native_plan_preparation"].append(elapsed)
        gmsh_plan, elapsed = _timed(
            lambda partition=partition, specification=specification: gmsh_provider.plan(
                partition.model, specification
            )
        )
        timings["gmsh_plan_preparation"].append(elapsed)
        source_revision = partition.model.source_revision
        native_result, elapsed = _timed(native_plan.execute)
        timings["native_execution"].append(elapsed)
        gmsh_result, elapsed = _timed(gmsh_plan.execute)
        timings["gmsh_execution"].append(elapsed)
        if partition.model.source_revision != source_revision:
            raise RuntimeError("A CAD provider mutated the shared comparison source.")
        samples.append(
            {
                "source_revision": source_revision,
                "request": specification.specification_id,
                "native": _comparison_result_record(
                    "native", native_plan, native_result, partition
                ),
                "gmsh": _comparison_result_record(
                    "gmsh", gmsh_plan, gmsh_result, partition
                ),
                "same_source": native_plan.source.source_revision
                == gmsh_plan.source.source_revision
                == source_revision,
                "same_request": native_plan.specification.specification_id
                == gmsh_plan.specification.specification_id
                == specification.specification_id,
            }
        )
    return {
        "scenario": "native-versus-gmsh-cad-meshing",
        "status": "passed",
        "resolution": resolution,
        "regions_per_sample": region_count,
        "repeats": repeats,
        "explicit_optional_comparison": True,
        "provider_versions": {
            "native_meshcore": meshcore_identity(),
            "gmsh": samples[0]["gmsh"]["provider_version"],
        },
        "phase_seconds_samples": timings,
        "phase_timing_ranges_seconds": {
            name: _range(values) for name, values in timings.items()
        },
        "samples": samples,
    }


def _lower_compile(compiled_entry: Any, *arguments: Any) -> tuple[Any, dict[str, object]]:
    """Lower and compile one real JAX entry and retain its executable."""
    lowered, lowering = _timed(lambda: compiled_entry.lower(*arguments))
    compiled, compile_seconds = _timed(lowered.compile)
    memory = compiled.compiled.memory_analysis()
    return compiled, {
        "lowering_seconds": lowering,
        "compile_seconds": compile_seconds,
        "temporary_bytes": memory.temp_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "argument_bytes": memory.argument_size_in_bytes,
        "generated_code_bytes": memory.generated_code_size_in_bytes,
    }


def _compiled_cost(compiled_entry: Any, *arguments: Any) -> dict[str, object]:
    """Report compiler-measured lowering, compilation, and storage bytes."""
    _, observations = _lower_compile(compiled_entry, *arguments)
    return observations


def _native_volume_specification(source: Any, target_size: float, /) -> Any:
    scope = phx.meshing.MeshingScope.whole(source, 2)
    return phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(phx.meshing.UniformSizeControl(scope, target_size),),
        maximum_deviation=0.02,
    )


def _native_volume_result_record(
    source: Any,
    plan: Any,
    result: phx.meshing.CellMeshingResult,
    expected_volume: float,
    /,
) -> dict[str, object]:
    measured_volume = float(np.sum(np.asarray(result.quality.evaluation.measures)))
    relative_volume_error = abs(measured_volume - expected_volume) / expected_volume
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError("Native CAD volume benchmark failed audit or compliance.")
    if relative_volume_error > 5.0e-8:
        raise RuntimeError(
            "Native CAD volume differs from its exact source measure: "
            f"{relative_volume_error:.6g}."
        )
    if not result.region_boundary_evidence:
        raise RuntimeError("Native CAD volume omitted source-region boundary evidence.")
    if any(
        reference_cell_topology(block.cell_kind).family != "tetrahedron"
        for block in result.mesh.blocks
    ):
        raise RuntimeError("Native CAD volume did not preserve its required tetrahedra.")
    if result.certification is None:
        raise RuntimeError("Native CAD volume omitted its certification report.")
    return {
        "status": "passed",
        "identity": {
            "source": source.source_id,
            "source_revision": source.source_revision,
            "source_geometry": source.geometry.geometry_id,
            "plan": plan.plan_id,
            "route": getattr(plan.route, "value", str(plan.route)),
            "result": result.result_id,
        },
        "correctness": {
            "audit_passed": result.audit.passed,
            "compliance_passed": result.compliance.passed,
            "expected_volume": expected_volume,
            "measured_volume": measured_volume,
            "relative_volume_error": relative_volume_error,
            "source_region_boundary_count": len(result.region_boundary_evidence),
        },
        "counts": {
            "vertices": result.mesh.entity_set(0).count,
            "cells": result.mesh.entity_set(3).count,
            "source_faces": source.topology.num_faces,
            "source_solids": source.topology.num_solids,
        },
        "quality": {
            "minimum_measure": result.quality.minimum_measure,
            "minimum_mean_ratio": result.quality.minimum_mean_ratio,
        },
        "validity_decisions": _validity_decisions(
            result.certification.embedding.validity
        ),
        "logical_retained_array_bytes": _array_bytes((source, plan, result)),
    }


def _changing_box_source(resolution: int, sample: int, repeats: int, /) -> Any:
    extent = float(resolution) + float(sample + 1) / float(repeats + 1)
    return brep_box(
        (0.0, 0.0, 0.0),
        (extent, 1.0, 1.0),
        coordinate_contract=_contract(),
        source_id="benchmark:native-changing-box",
    )


def benchmark_native_cad_meshing(resolution: int, repeats: int) -> dict[str, object]:
    """Scale a real native B-Rep volume route for fixed and changing sources."""
    if resolution < 1 or repeats < 1:
        raise ValueError("resolution and repeats must be positive.")
    provider = phx.meshing.NativeMeshingProvider()
    fixed_source, source_seconds = _timed(
        lambda: brep_box(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            coordinate_contract=_contract(),
            source_id="benchmark:native-fixed-box",
        )
    )
    fixed_specification = _native_volume_specification(fixed_source, 2.0 / resolution)
    fixed_plan, preparation_seconds = _timed(
        lambda: provider.plan(fixed_source, fixed_specification)
    )
    fixed_result, first_execution_seconds = _timed(fixed_plan.execute)
    fixed_record = _native_volume_result_record(
        fixed_source, fixed_plan, fixed_result, 1.0
    )
    warm_execution_seconds = []
    for _ in range(repeats):
        warmed, elapsed = _timed(fixed_plan.execute)
        _native_volume_result_record(fixed_source, fixed_plan, warmed, 1.0)
        warm_execution_seconds.append(elapsed)

    changing_records = []
    changing_source_seconds = []
    changing_preparation_seconds = []
    changing_execution_seconds = []
    for sample in range(repeats):
        jax.clear_caches()
        source, elapsed = _timed(
            lambda sample=sample: _changing_box_source(resolution, sample, repeats)
        )
        changing_source_seconds.append(elapsed)
        target_volume = float(resolution) + float(sample + 1) / float(repeats + 1)
        specification = _native_volume_specification(source, 0.5)
        plan, elapsed = _timed(
            lambda source=source, specification=specification: provider.plan(
                source, specification
            )
        )
        changing_preparation_seconds.append(elapsed)
        result, elapsed = _timed(plan.execute)
        changing_execution_seconds.append(elapsed)
        changing_record = _native_volume_result_record(
            source, plan, result, target_volume
        )
        changing_records.append(changing_record)

    return {
        "scenario": "native-cad-volume-meshing",
        "status": "passed",
        "resolution": resolution,
        "repeats": repeats,
        "fixed_geometry": {
            "phases_seconds": {
                "source_construction": source_seconds,
                "provider_preparation": preparation_seconds,
                "first_prepared_execution": first_execution_seconds,
                "warm_prepared_execution_samples": warm_execution_seconds,
                "warm_prepared_execution_range": _range(warm_execution_seconds),
            },
            "result": fixed_record,
        },
        "changing_geometry": {
            "phases_seconds_samples": {
                "source_construction": changing_source_seconds,
                "provider_preparation": changing_preparation_seconds,
                "execution": changing_execution_seconds,
            },
            "timing_ranges_seconds": {
                "source_construction": _range(changing_source_seconds),
                "provider_preparation": _range(changing_preparation_seconds),
                "execution": _range(changing_execution_seconds),
            },
            "samples": changing_records,
        },
    }


def _native_array_bytes(record: PlcRecovery3D | TetMeshArrays, /) -> int:
    return sum(value.nbytes for value in record if isinstance(value, np.ndarray))


def _native_record_fingerprint(record: PlcRecovery3D | TetMeshArrays, /) -> str:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint

    arrays = {
        name: array_tree_fingerprint(value)
        for name in record._fields
        if isinstance((value := getattr(record, name)), np.ndarray)
    }
    if isinstance(record, TetMeshArrays):
        identity: dict[str, object] = {
            "generation": record.generation,
            "identity_bound": record.identity_bound,
            "source_id": record.source_id,
            "source_revision": record.source_revision,
            "source_map_id": record.source_map_id,
            "frame_id": record.frame_id,
        }
    else:
        identity = {"status": record.status.name}
    return canonical_fingerprint({"arrays": arrays, "identity": identity})


def _native_source_bank_fingerprint(arrays: TetMeshArrays, /) -> str:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint

    names = (
        "source_points",
        "source_faces",
        "source_face_group_rows",
        "source_face_ids",
        "source_face_tolerances",
        "source_segments",
        "source_segment_group_rows",
        "source_segment_ids",
        "source_segment_tolerances",
    )
    content = {name: array_tree_fingerprint(getattr(arrays, name)) for name in names}
    return canonical_fingerprint(
        {
            "source": content,
            "identity": {
                "source_id": arrays.source_id,
                "source_revision": arrays.source_revision,
                "source_map_id": arrays.source_map_id,
                "frame_id": arrays.frame_id,
            },
        }
    )


def _native_source_banks_equal(first: TetMeshArrays, second: TetMeshArrays, /) -> bool:
    names = (
        "source_points",
        "source_faces",
        "source_face_group_rows",
        "source_face_ids",
        "source_face_tolerances",
        "source_segments",
        "source_segment_group_rows",
        "source_segment_ids",
        "source_segment_tolerances",
    )
    return (
        first.source_id,
        first.source_revision,
        first.source_map_id,
        first.frame_id,
    ) == (
        second.source_id,
        second.source_revision,
        second.source_map_id,
        second.frame_id,
    ) and all(
        np.array_equal(getattr(first, name), getattr(second, name)) for name in names
    )


def _native_original_witnesses_equal(
    before: TetMeshArrays, after: TetMeshArrays, /
) -> bool:
    after_rows = {
        int(identifier): row for row, identifier in enumerate(after.vertex_ids.tolist())
    }
    witness_names = (
        "witness_strata",
        "witness_entities",
        "witness_parameters",
        "witness_deviations",
    )
    for before_row, identifier in enumerate(before.vertex_ids.tolist()):
        after_row = after_rows.get(int(identifier))
        if after_row is None or any(
            not np.array_equal(
                getattr(before, name)[before_row],
                getattr(after, name)[after_row],
            )
            for name in witness_names
        ):
            return False
    return True


class _NativeRefinementSample(TypedDict):
    status: str
    resolution: int
    target_size: float
    radius_edge_bound: float
    surface_spacing_fraction: float
    source_surface_target_edge_size: float
    phases_seconds: dict[str, float]
    identity: dict[str, object]
    refinement: dict[str, object]
    correctness: dict[str, object]
    counts: dict[str, object]
    logical_array_bytes: dict[str, int]


def _native_brep_plc_refinement_sample(
    resolution: int, target_size: float, /
) -> _NativeRefinementSample:
    from examples.native_brep_workflow import (
        DEFAULT_MAXIMUM_DEVIATION,
        DEFAULT_SURFACE_SPACING_FRACTION,
        DEFAULT_VOLUME_SIZE,
        prepare_workflow,
    )
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax._meshcore import (
        meshcore_identity,
        MeshcoreStatus,
        PlcBoundaryPolicy,
        recover_plc_3d,
        TetMesh3D,
        TetMeshEditKind,
        TetMeshEditStatus,
    )
    from phydrax.meshing._canonical import _accept_native_construction, _NATIVE_PROVIDER
    from phydrax.meshing._volume_generation import (
        _region_seed_points,
        _size_field,
        _source_rows,
        _source_tables,
    )
    from phydrax.meshing.providers._native import (
        _PreparedDirectBRepCadVolume,
    )
    from phydrax.meshing.providers._native_cad_volume import (
        _translate_specification,
        NativeCadVolumeSource,
    )

    if not np.isfinite(target_size) or target_size <= 0.0:
        raise ValueError("PLC refinement target size must be finite and positive.")
    surface_spacing_fraction = (
        DEFAULT_SURFACE_SPACING_FRACTION
        / float(resolution)
        * (DEFAULT_VOLUME_SIZE / target_size)
    )
    if not 0.0 < surface_spacing_fraction < 1.0:
        raise ValueError(
            "The initial PLC resolution must produce a valid surface schedule."
        )
    (model, specification, plan), public_plan_preparation_seconds = _timed(
        lambda: prepare_workflow(
            size=target_size,
            maximum_deviation=DEFAULT_MAXIMUM_DEVIATION,
            surface_spacing_fraction=surface_spacing_fraction,
        )
    )
    prepared = plan.prepared
    if not isinstance(prepared, _PreparedDirectBRepCadVolume):
        raise TypeError("The authored BRep workflow did not prepare direct CAD volume.")
    surface_schedule = prepared.boundary_options.surface_schedule
    volume_schedule = plan.route_options.volume_schedule
    if surface_schedule is None or volume_schedule is None:
        raise RuntimeError("The authored CAD plan omitted its resolved native schedules.")
    if prepared.specification.specification_id != specification.specification_id:
        raise RuntimeError("The prepared CAD route changed its authored volume request.")

    boundary_construction, boundary_execution_seconds = _timed(
        prepared.boundary_preparation.execute
    )
    boundary_result, boundary_acceptance_seconds = _timed(
        lambda: _accept_native_construction(
            boundary_construction,
            prepared.boundary_specification,
            _NATIVE_PROVIDER,
            meshcore_identity(),
            prepared.boundary_plan_id,
            "parametric_surface",
        )
    )
    if not boundary_result.audit.passed or not boundary_result.compliance.passed:
        raise RuntimeError("Public BRep surface preparation failed acceptance.")
    cad_source, cad_source_preparation_seconds = _timed(
        lambda: NativeCadVolumeSource(
            prepared.domain,
            boundary_result,
            maximum_deviation=None,
            boundary="fixed",
        )
    )
    plc_specification, scope_translation_seconds = _timed(
        lambda: _translate_specification(cad_source, specification)
    )
    complex_ = cad_source.plc_source.complex
    facet_tolerances = np.full((complex_.facet_count,), np.inf, dtype=np.float64)
    segment_tolerances = np.full((complex_.segments.shape[0],), np.inf, dtype=np.float64)
    for feature in plc_specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        match feature.scope.entity_dimension, feature.feature_kind.value:
            case 2, "surface" | "material_interface":
                rows = _source_rows(identifiers, complex_.facet_entity_ids, "facet")
                facet_tolerances[rows] = np.minimum(
                    facet_tolerances[rows], feature.maximum_deviation
                )
            case 1, "curve":
                rows = _source_rows(identifiers, complex_.segment_entity_ids, "segment")
                segment_tolerances[rows] = np.minimum(
                    segment_tolerances[rows], feature.maximum_deviation
                )
            case _:
                continue
    facet_tolerances[~np.isfinite(facet_tolerances)] = 0.0
    segment_tolerances[~np.isfinite(segment_tolerances)] = 0.0
    native_boundary_policy = (
        PlcBoundaryPolicy.FIXED
        if complex_.boundary == "fixed"
        else PlcBoundaryPolicy.CONFORMING
    )
    (seeds, seed_regions), region_seed_preparation_seconds = _timed(
        lambda: _region_seed_points(complex_, plc_specification)
    )
    recovery, recovery_seconds = _timed(
        lambda: recover_plc_3d(
            complex_.vertices,
            complex_.polygon_offsets,
            complex_.polygon_vertices,
            complex_.polygon_facets,
            complex_.facet_regions,
            segments=complex_.segments,
            seeds=seeds,
            seed_regions=seed_regions,
            facet_tolerances=facet_tolerances,
            segment_tolerances=segment_tolerances,
            boundary_policy=native_boundary_policy,
        )
    )
    if recovery.status is not MeshcoreStatus.OK:
        raise RuntimeError(f"Native BRep PLC recovery returned {recovery.status.name}.")
    tables, source_table_preparation_seconds = _timed(
        lambda: _source_tables(complex_, plc_specification, recovery)
    )
    face_rows = np.asarray(recovery.face_sources, dtype=np.int64)
    edge_rows = np.asarray(recovery.segment_sources, dtype=np.int64)
    if np.any((face_rows < 0) | (face_rows >= tables.facet_labels.size)) or np.any(
        (edge_rows < 0) | (edge_rows >= tables.edge_labels.size)
    ):
        raise RuntimeError("Native BRep recovery lost its source facet/edge ancestry.")
    live_face_sources = tables.facet_labels[face_rows]
    live_segment_sources = tables.edge_labels[edge_rows]
    source = tables.source
    # The direct BRep volume route fills with the default edge-length convention.
    sizes, size_field_preparation_seconds = _timed(
        lambda: _size_field(
            complex_,
            plc_specification,
            recovery,
            tables,
            size_convention="edge-length",
        )
    )
    if sizes is not None and sizes.shape != (recovery.points.shape[0],):
        raise RuntimeError("Native BRep volume sizing does not align with PLC points.")

    points = np.asarray(source.points, dtype=np.float64)
    source_triangles = np.asarray(source.faces, dtype=np.int32)
    source_face_group_rows = source.face_group_rows
    source_face_ids = source.face_ids
    segments = source.segments
    source_segment_ids = source.segment_ids
    facet_regions = np.asarray(complex_.facet_regions, dtype=np.int32)
    input_counts = {
        "input_points": int(complex_.vertices.shape[0]),
        "input_polygons": int(complex_.polygon_facets.size),
        "input_facets": int(complex_.facet_regions.shape[0]),
        "input_segments": int(complex_.segments.shape[0]),
        "accepted_surface_vertices": boundary_result.mesh.entity_set(0).count,
        "accepted_surface_triangles": boundary_result.mesh.entity_set(2).count,
    }
    source_id = cad_source.source_id
    source_revision = cad_source.source_revision
    source_map_id = ""
    frame_id = plan.coordinate_contract.spatial_id
    source_vertex_authority = f"{len(complex_.vertex_ids)} BRep PLC source vertex IDs"
    source_identity_fingerprint = canonical_fingerprint(
        {
            "kind": "native-brep-workflow-source",
            "workflow_plan": plan.plan_id,
            "boundary_plan": prepared.boundary_plan_id,
            "boundary_result": boundary_result.result_id,
            "cad_source_binding": cad_source.binding_id,
            "source_id": source_id,
            "source_revision": source_revision,
            "coordinate_frame": frame_id,
        }
    )
    source_oracle_fingerprint = canonical_fingerprint(
        {
            "domain": prepared.domain.domain_id,
            "plc_complex": complex_.complex_id,
            "source_facets": complex_.facet_ids,
            "facet_occurrences": complex_.facet_occurrence_paths,
            "source_segments": complex_.segment_ids,
            "segment_occurrences": complex_.segment_occurrence_paths,
            "source_triangle_ids": cad_source.source_triangle_ids,
            "source_triangle_patches": array_tree_fingerprint(
                cad_source.source_triangle_patches
            ),
            "source_triangle_parameters": array_tree_fingerprint(
                cad_source.source_triangle_parameters
            ),
            "facet_regions": array_tree_fingerprint(facet_regions),
        }
    )
    source_preparation_phases = {
        "cad_to_plc_scope_translation": scope_translation_seconds,
        "public_brep_and_volume_plan_preparation": public_plan_preparation_seconds,
        "boundary_surface_execution": boundary_execution_seconds,
        "boundary_surface_acceptance": boundary_acceptance_seconds,
        "native_cad_volume_source_preparation": cad_source_preparation_seconds,
        "region_seed_preparation": region_seed_preparation_seconds,
        "plc_recovery": recovery_seconds,
        "source_table_preparation": source_table_preparation_seconds,
        "volume_size_field_preparation": size_field_preparation_seconds,
    }
    radius_edge_bound = volume_schedule.radius_edge_bound
    state, state_preparation_seconds = _timed(
        lambda: TetMesh3D(
            recovery.points,
            recovery.tetrahedra,
            recovery.tetrahedron_regions,
            recovery.faces,
            live_face_sources,
            recovery.segments,
            live_segment_sources,
            boundary_policy=native_boundary_policy,
            protection_radii=recovery.protection_radii,
            source=source,
            source_id=source_id,
            source_revision=source_revision,
            source_map_id=source_map_id,
            frame_id=frame_id,
        )
    )
    try:
        before, before_snapshot_seconds = _timed(state.arrays)
        refinement, refinement_seconds = _timed(
            lambda: state.refine(radius_edge_bound=radius_edge_bound, sizes=sizes)
        )
        after, after_snapshot_seconds = _timed(state.arrays)
        evidence, source_evidence_seconds = _timed(state.source_evidence)
        unmet, unmet_evidence_seconds = _timed(state.unmet)
        verification_started = perf_counter()

        if refinement.edit_status != TetMeshEditStatus.APPLIED:
            raise RuntimeError(
                f"Persistent PLC refinement returned {refinement.edit_status.name}."
            )
        if refinement.status not in (MeshcoreStatus.OK, MeshcoreStatus.UNRESOLVED):
            raise RuntimeError(
                f"Persistent PLC refinement returned {refinement.status.name}."
            )
        if not before.identity_bound or not after.identity_bound:
            raise RuntimeError("Persistent PLC state lost its explicit source identity.")
        if not _native_source_banks_equal(before, after):
            raise RuntimeError("Persistent PLC refinement changed owned source banks.")
        if not np.array_equal(before.source_points, points):
            raise RuntimeError(
                "Persistent PLC state lost its original source point bank."
            )
        if not np.array_equal(before.source_faces, source_triangles):
            raise RuntimeError("Persistent PLC state lost its source triangle atlas.")
        if not np.array_equal(before.source_face_group_rows, source_face_group_rows):
            raise RuntimeError("Persistent PLC state changed source facet groups.")
        if not np.array_equal(before.source_segments, segments):
            raise RuntimeError("Persistent PLC state lost its source segment bank.")
        if not np.array_equal(before.source_segment_ids, source_segment_ids):
            raise RuntimeError("Persistent PLC state changed source segment identities.")
        allowed_region_labels = np.unique(facet_regions.reshape(-1))
        if not np.isin(after.tetrahedron_regions, allowed_region_labels).all():
            raise RuntimeError("Persistent PLC state invented a foreign material region.")
        if (
            not np.isin(live_face_sources, after.face_sources).all()
            or not np.isin(after.face_sources, source_face_ids).all()
        ):
            raise RuntimeError("Refined PLC facets lost their source entity ancestry.")
        if (
            not np.isin(live_segment_sources, after.segment_sources).all()
            or not np.isin(after.segment_sources, source_segment_ids).all()
        ):
            raise RuntimeError("Refined PLC segments lost their source entity ancestry.")
        if not np.array_equal(
            after.witness_deviations, evidence.witness_deviations
        ) or not np.array_equal(after.witness_entities, evidence.witness_entities):
            raise RuntimeError(
                "Persistent source evidence differs from retained witnesses."
            )
        if not _native_original_witnesses_equal(before, after):
            raise RuntimeError("Refinement changed an existing source witness.")
        requested_source_bound = max(
            float(np.max(before.source_face_tolerances, initial=0.0)),
            float(np.max(before.source_segment_tolerances, initial=0.0)),
        )
        achieved_source_bound = float(np.max(after.witness_deviations, initial=0.0))
        if (
            evidence.requested_bound != requested_source_bound
            or evidence.achieved_bound != achieved_source_bound
            or not np.isfinite(evidence.achieved_bound)
        ):
            raise RuntimeError(
                "Source deviation evidence differs from its retained banks."
            )
        if refinement.unmet_cells != np.unique(unmet.cell_ids).size:
            raise RuntimeError("Refinement run status and unmet evidence disagree.")
        if unmet.values.size and not np.all(np.isfinite(unmet.values)):
            raise RuntimeError("Refinement unmet evidence contains nonfinite values.")

        old_vertex_ids = set(int(value) for value in before.vertex_ids.tolist())
        old_cell_ids = set(int(value) for value in before.cell_ids.tolist())
        new_vertex_rows = np.flatnonzero(~np.isin(after.vertex_ids, before.vertex_ids))
        new_cell_rows = np.flatnonzero(~np.isin(after.cell_ids, before.cell_ids))
        vertex_ancestry_arities = np.diff(after.vertex_ancestry_offsets)[new_vertex_rows]
        stellar_centers = 0
        for row in new_vertex_rows.tolist():
            first = int(after.vertex_ancestry_offsets[row])
            last = int(after.vertex_ancestry_offsets[row + 1])
            parent_ids = after.vertex_ancestor_ids[first:last]
            weights = after.vertex_ancestry_weights[first:last]
            if (
                not parent_ids.size
                or not set(int(value) for value in parent_ids.tolist()).issubset(
                    old_vertex_ids
                )
                or not np.isfinite(weights).all()
                or not np.isclose(float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12)
            ):
                raise RuntimeError(
                    "Refined vertex ancestry is incomplete or inconsistent."
                )
            if parent_ids.size == 4:
                if not np.array_equal(weights, np.full((4,), 0.25, dtype=np.float64)):
                    raise RuntimeError(
                        "Stellar center ancestry is not exact quarter weighting."
                    )
                stellar_centers += 1
        before_regions = {
            int(cell_id): int(region)
            for cell_id, region in zip(
                before.cell_ids.tolist(),
                before.tetrahedron_regions.tolist(),
                strict=True,
            )
        }
        for row in new_cell_rows.tolist():
            first = int(after.cell_ancestry_offsets[row])
            last = int(after.cell_ancestry_offsets[row + 1])
            parents = after.cell_ancestor_ids[first:last]
            edit_kind = int(after.cell_ancestry_edit_kinds[row])
            if (
                parents.size != 1
                or int(parents[0]) not in old_cell_ids
                or int(after.tetrahedron_regions[row])
                != before_regions.get(int(parents[0]))
                or edit_kind != int(TetMeshEditKind.REFINE_CELL)
            ):
                raise RuntimeError(
                    "Refined cell lacks its immediate parent/edit ancestry."
                )
        if refinement.inserted_vertices != new_vertex_rows.size:
            raise RuntimeError(
                "Refinement inserted-vertex count differs from live topology."
            )

        source_before_fingerprint = _native_source_bank_fingerprint(before)
        source_after_fingerprint = _native_source_bank_fingerprint(after)
        if source_before_fingerprint != source_after_fingerprint:
            raise RuntimeError(
                "Persistent PLC source identity changed during refinement."
            )
        source_evidence_fingerprint = canonical_fingerprint(
            {
                "requested_bound": evidence.requested_bound,
                "achieved_bound": evidence.achieved_bound,
                "witness_strata": array_tree_fingerprint(evidence.witness_strata),
                "witness_entities": array_tree_fingerprint(evidence.witness_entities),
                "witness_parameters": array_tree_fingerprint(evidence.witness_parameters),
                "witness_deviations": array_tree_fingerprint(evidence.witness_deviations),
            }
        )
        unmet_fingerprint = canonical_fingerprint(
            {
                "cell_ids": array_tree_fingerprint(unmet.cell_ids),
                "criteria": array_tree_fingerprint(unmet.criteria),
                "reasons": array_tree_fingerprint(unmet.reasons),
                "values": array_tree_fingerprint(unmet.values),
            }
        )
        verification_seconds = perf_counter() - verification_started
        return {
            "status": "passed",
            "resolution": resolution,
            "target_size": target_size,
            "radius_edge_bound": radius_edge_bound,
            "surface_spacing_fraction": surface_schedule.spacing_fraction,
            "source_surface_target_edge_size": (
                surface_schedule.spacing_fraction * target_size
            ),
            "phases_seconds": {
                **source_preparation_phases,
                "persistent_state_preparation": state_preparation_seconds,
                "pre_refinement_snapshot": before_snapshot_seconds,
                "persistent_refinement": refinement_seconds,
                "post_refinement_snapshot": after_snapshot_seconds,
                "source_evidence": source_evidence_seconds,
                "unmet_evidence": unmet_evidence_seconds,
                "postcondition_validation": verification_seconds,
            },
            "identity": {
                "source_definition": source_identity_fingerprint,
                "region_boundary_oracle": source_oracle_fingerprint,
                "source_id": after.source_id,
                "source_vertex_authority": source_vertex_authority,
                "source_revision": after.source_revision,
                "source_map_id": after.source_map_id,
                "frame_id": after.frame_id,
                "public_plan": plan.plan_id,
                "volume_specification": specification.specification_id,
                "plc_specification": plc_specification.specification_id,
                "surface_schedule": surface_schedule.schedule_id,
                "volume_schedule": volume_schedule.schedule_id,
                "surface_spacing_fraction": surface_schedule.spacing_fraction,
                "boundary_plan": prepared.boundary_plan_id,
                "boundary_result": boundary_result.result_id,
                "cad_source_binding": cad_source.binding_id,
                "plc_complex_id": complex_.complex_id,
                "source_banks_before": source_before_fingerprint,
                "source_banks_after": source_after_fingerprint,
                "source_evidence": source_evidence_fingerprint,
                "unmet_evidence": unmet_fingerprint,
                "post_refinement_state": _native_record_fingerprint(after),
            },
            "refinement": {
                "radius_edge_bound": radius_edge_bound,
                "volume_schedule": volume_schedule.schedule_id,
                "edit_status": refinement.edit_status.name,
                "recovery_status": recovery.status.name,
                "status": refinement.status.name,
                "inserted_vertices": refinement.inserted_vertices,
                "unmet_cells": refinement.unmet_cells,
                "unmet_evidence_count": int(unmet.cell_ids.size),
                "unmet_criteria": {
                    str(int(criterion)): int(count)
                    for criterion, count in zip(
                        *np.unique(unmet.criteria, return_counts=True),
                        strict=True,
                    )
                },
                "unmet_maximum": float(np.max(unmet.values, initial=0.0)),
            },
            "correctness": {
                "source_banks_unchanged": True,
                "source_identity_bound": True,
                "source_triangle_atlas_preserved": True,
                "source_entity_ancestry_preserved": True,
                "source_witnesses_preserved": True,
                "vertex_ancestry_valid": True,
                "stellar_center_count": stellar_centers,
                "cell_parent_and_edit_ancestry_valid": True,
                "new_vertices": int(new_vertex_rows.size),
                "new_cells": int(new_cell_rows.size),
                "vertex_ancestry_arities": {
                    str(int(arity)): int(count)
                    for arity, count in zip(
                        *np.unique(vertex_ancestry_arities, return_counts=True),
                        strict=True,
                    )
                },
            },
            "counts": {
                **input_counts,
                "recovered_points": int(recovery.points.shape[0]),
                "initial_tetrahedra": int(before.tetrahedra.shape[0]),
                "final_points": int(after.points.shape[0]),
                "final_tetrahedra": int(after.tetrahedra.shape[0]),
                "source_faces": int(after.source_faces.shape[0]),
                "source_segments": int(after.source_segments.shape[0]),
                "tetrahedron_regions": {
                    str(int(region)): int(count)
                    for region, count in zip(
                        *np.unique(after.tetrahedron_regions, return_counts=True),
                        strict=True,
                    )
                },
            },
            "logical_array_bytes": {
                "recovery_output": _native_array_bytes(recovery),
                "pre_refinement_snapshot": _native_array_bytes(before),
                "post_refinement_snapshot": _native_array_bytes(after),
                "source_evidence": sum(
                    value.nbytes
                    for value in (
                        evidence.witness_strata,
                        evidence.witness_entities,
                        evidence.witness_parameters,
                        evidence.witness_deviations,
                    )
                ),
                "unmet_evidence": sum(
                    value.nbytes
                    for value in (
                        unmet.cell_ids,
                        unmet.criteria,
                        unmet.reasons,
                        unmet.values,
                    )
                ),
            },
        }
    finally:
        state.close()


def benchmark_native_brep_plc_refinement(
    resolution: int,
    repeats: int,
    /,
    *,
    target_size: float | None = None,
) -> dict[str, object]:
    """Scale the public profile-extrusion PLC and its independent fill size."""
    from examples.native_brep_workflow import DEFAULT_VOLUME_SIZE

    if resolution < 1 or repeats < 1:
        raise ValueError("resolution and repeats must be positive.")
    target_size = DEFAULT_VOLUME_SIZE if target_size is None else float(target_size)
    if not np.isfinite(target_size) or target_size <= 0.0:
        raise ValueError("PLC refinement target size must be finite and positive.")
    samples = []
    for _ in range(repeats):
        jax.clear_caches()
        samples.append(_native_brep_plc_refinement_sample(resolution, target_size))
    phase_names = tuple(samples[0]["phases_seconds"])
    return {
        "scenario": "native-brep-plc-refinement",
        "status": "passed",
        "resolution": resolution,
        "target_size": target_size,
        "repeats": repeats,
        "source_preparation": "examples.native_brep_workflow.prepare_workflow",
        "phase_seconds_samples": {
            name: [sample["phases_seconds"][name] for sample in samples]
            for name in phase_names
        },
        "phase_timing_ranges_seconds": {
            name: _range([sample["phases_seconds"][name] for sample in samples])
            for name in phase_names
        },
        "samples": samples,
    }


def _native_tspline_patch(resolution: int, /) -> tuple[Any, dict[str, float]]:
    from phydrax._interpolation import TMesh2D
    from phydrax.geometry.brep._tspline import TSplineSurfacePatch

    if resolution < 4:
        raise ValueError("Native T-spline scaling requires at least four knot intervals.")
    breaks = tuple(Fraction(index, resolution) for index in range(resolution + 1))
    source_mesh, tensor_seconds = _timed(lambda: TMesh2D.tensor_product(breaks, breaks))
    marked = min(
        source_mesh.cells,
        key=lambda cell: (
            abs(float((cell.parameter_bounds[0] + cell.parameter_bounds[1]) / 2) - 0.5)
            + abs(float((cell.parameter_bounds[2] + cell.parameter_bounds[3]) / 2) - 0.5)
        ),
    )
    refinement, refinement_seconds = _timed(lambda: source_mesh.refine((marked.cell_id,)))
    if not refinement.certificate.passed or refinement.refined_mesh.t_junction_count == 0:
        raise RuntimeError("Native T-spline local refinement lost its ASTS witness.")
    orientations = {
        (extension.axis, extension.missing_direction)
        for extension in refinement.refined_mesh.t_junction_extensions
    }
    expected_orientations = {
        ("u", "left"),
        ("u", "right"),
        ("v", "down"),
        ("v", "up"),
    }
    if not orientations.issuperset(expected_orientations):
        raise RuntimeError("Native T-spline refinement lost a T-junction orientation.")
    exact_control_points = tuple(
        (u, v, u * v)
        for u, v, _ in (
            anchor.local_knots.greville_parameter for anchor in source_mesh.anchors
        )
    )
    exact_weights = tuple(
        Fraction(1) + Fraction(2) * u * (1 - u) * v * (1 - v)
        for u, v, _ in exact_control_points
    )
    if len(set(exact_weights)) == 1 or any(weight <= 0 for weight in exact_weights):
        raise RuntimeError(
            "Native T-spline fixture requires positive nonuniform weights."
        )
    control_points = np.asarray(
        [[float(value) for value in row] for row in exact_control_points],
        dtype=np.float64,
    )
    weights = np.asarray([float(value) for value in exact_weights], dtype=np.float64)
    surface, patch_seconds = _timed(
        lambda: TSplineSurfacePatch(
            refinement.refined_mesh,
            control_points,
            weights,
            exact_control_points=exact_control_points,
            exact_weights=exact_weights,
            source_mesh=source_mesh,
            transfer_plans=(refinement.transfer_plan,),
            transfer_meshes=(source_mesh, refinement.refined_mesh),
        )
    )
    return surface, {
        "tensor_mesh": tensor_seconds,
        "local_refinement": refinement_seconds,
        "source_patch": patch_seconds,
    }


def _native_tspline_brep_model(surface: Any, contract: Any, /) -> Any:
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.geometry.brep._model import BRepGeometry, BRepImportReport, BRepModel
    from phydrax.geometry.brep._patches import LineCurve

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    curves = tuple(
        LineCurve(points[first], points[last] - points[first]) for first, last in edges
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 1.0), (-1.0, 0.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    geometry = BRepGeometry(
        vertex_points=points,
        curves=curves,
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (4, 1)),
        edge_vertices=edges,
        pcurves=pcurves,
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    source_digest = canonical_fingerprint(
        {
            "kind": "native-tspline-benchmark-source",
            "surface_definition": surface.source_definition_id,
            "boundary_points": points.tolist(),
        }
    )
    import_policy_id = canonical_fingerprint({"kind": "native-tspline-benchmark-source"})
    report = BRepImportReport(
        source_id="benchmark:native-tspline-cad",
        source_digest=source_digest,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id=import_policy_id,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    return BRepModel(
        geometry=geometry,
        patches=(surface,),
        parameter_bounds=np.asarray((((0.0, 0.0), (1.0, 1.0)),)),
        orientation=np.asarray((1.0,)),
        trim_domains=None,
        physical_tags=("analysis-surface",),
        coordinate_contract=contract,
        report=report,
    )


def _unit_tspline_source(points: Any, arguments: Any, /) -> Any:
    del arguments
    return jax.numpy.ones(points.shape[:-1], dtype=points.dtype)


@eqx.filter_jit
def _solve_tspline_poisson(problem: Any) -> tuple[Any, Any, Any, Any]:
    operator, right_hand_side = problem.linear_system()
    solve = phx.linalg.solve(operator, right_hand_side)
    residual = jax.numpy.linalg.norm(problem.residual(solve.value))
    rhs_norm = jax.numpy.linalg.norm(right_hand_side)
    return solve.value, solve.successful, residual, rhs_norm


def benchmark_native_tspline_cad_analysis(
    resolution: int, repeats: int
) -> dict[str, object]:
    """Scale ASTS refinement, native CAD surface meshing, and IGA analysis."""
    from phydrax.discretization import iga
    from phydrax.geometry._meshing_domain import MeshingDomain
    from phydrax.geometry._source_fidelity import (
        certify_source_fidelity,
        validate_source_fidelity_witness,
    )

    if resolution < 4 or repeats < 1:
        raise ValueError(
            "T-spline resolution must be at least four and repeats positive."
        )
    contract = phx.SpatialCoordinateContract.si()
    surface, source_phases = _native_tspline_patch(resolution)
    model, cad_source_seconds = _timed(
        lambda: _native_tspline_brep_model(surface, contract)
    )
    domain, domain_preparation_seconds = _timed(lambda: MeshingDomain.from_brep(model))
    scope = phx.meshing.MeshingScope.whole(domain, 2)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2,
            3,
            phx.meshing.CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(phx.meshing.UniformSizeControl(scope, 1.0 / resolution),),
        maximum_deviation=0.025,
    )
    provider = phx.meshing.NativeMeshingProvider(
        phx.meshing.NativeMeshingOptions("parametric_surface")
    )
    plan, provider_preparation_seconds = _timed(
        lambda: provider.plan(domain, specification, coordinate_contract=contract)
    )
    native_result, native_execution_seconds = _timed(plan.execute)
    surface_source = native_result.surface_source
    certification = native_result.certification
    if surface_source is None or certification is None or certification.fidelity is None:
        raise RuntimeError("Native T-spline meshing omitted source-chart fidelity.")
    if (
        certification.inputs.source is None
        or certification.inputs.fidelity_tolerance is None
        or certification.inputs.fidelity_witness is None
        or certification.inputs.fidelity_witness.witness_id
        != surface_source.chart_witness.witness_id
    ):
        raise RuntimeError("T-spline result lost its source-bound fidelity premises.")
    validate_source_fidelity_witness(
        native_result.mesh,
        native_result.geometry,
        certification.inputs.source,
        surface_source.chart_witness,
    )
    fidelity = certify_source_fidelity(
        native_result.mesh,
        native_result.geometry,
        certification.inputs.source,
        tolerance=certification.inputs.fidelity_tolerance,
        maximum_normal_deviation=certification.inputs.maximum_normal_deviation,
        witness=surface_source.chart_witness,
    )
    if (
        fidelity.status != "certified"
        or fidelity.certificate_id != certification.fidelity.certificate_id
        or any(len(piece) != 3 for piece in surface_source.source_piece_ids)
    ):
        raise RuntimeError("T-spline CAD source fidelity or ancestry did not revalidate.")
    if not native_result.audit.passed or not native_result.compliance.passed:
        raise RuntimeError("Native T-spline surface meshing failed audit or compliance.")

    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    iga_plan, iga_plan_seconds = _timed(
        lambda: iga.TSplineIsogeometricPlan(
            iga.TSplineGeometryState(surface),
            trim=trim,
            quadrature_policy=iga.IsogeometricQuadraturePolicy(6),
        )
    )
    marked = min(
        surface.mesh.cells,
        key=lambda cell: (
            abs(float((cell.parameter_bounds[0] + cell.parameter_bounds[1]) / 2) - 0.5)
            + abs(float((cell.parameter_bounds[2] + cell.parameter_bounds[3]) / 2) - 0.5)
        ),
    )
    refinement, iga_refinement_seconds = _timed(
        lambda: iga_plan.refine((marked.cell_id,))
    )
    target_plan = refinement.target_plan
    target_mesh = target_plan.geometry.mesh
    full_tensor_count = (len(target_mesh.axis_breaks[0]) - 1) * (
        len(target_mesh.axis_breaks[1]) - 1
    )
    if (
        target_mesh.t_junction_count == 0
        or target_mesh.cell_count >= full_tensor_count
        or target_plan.geometry.surface_patch.source_mesh.mesh_id
        != surface.source_mesh.mesh_id
        or not target_plan.exact_geometry_certificate.passed
        or not target_plan.geometry_certificate.passed
    ):
        raise RuntimeError("T-spline IGA refinement lost local source-map correctness.")
    prepared, iga_preparation_seconds = _timed(target_plan.prepare)
    source_term = phx.equations.coefficient(
        _unit_tspline_source,
        coefficient_id="native-tspline-benchmark-unit-source",
    )
    form = phx.equations.FiniteElementForm(
        "native-tspline-benchmark-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source_term),
        ),
    )
    finite_element_problem, equation_preparation_seconds = _timed(
        lambda: phx.equations.compile_finite_element_problem(
            form,
            prepared,
            constraint=prepared.homogeneous_trace_constraint("u"),
            execution_policy=phx.equations.FiniteElementExecutionPolicy(
                realization="matrix_free", local_kernel="dense"
            ),
        )
    )
    executable, compiler = _lower_compile(_solve_tspline_poisson, finite_element_problem)
    solved, first_solve_seconds = _timed(
        lambda: jax.block_until_ready(executable(finite_element_problem))
    )
    warm_solve_seconds = []
    for _ in range(repeats):
        _, elapsed = _timed(
            lambda: jax.block_until_ready(executable(finite_element_problem))
        )
        warm_solve_seconds.append(elapsed)
    solution, successful, residual_norm, rhs_norm = solved
    normalized_residual = float(np.asarray(residual_norm)) / max(
        float(np.asarray(rhs_norm)), 1.0
    )
    solution_norm = float(np.linalg.norm(np.asarray(solution)))
    solve_successful = bool(np.all(np.asarray(successful)))
    if not solve_successful or normalized_residual > 2.0e-8 or solution_norm == 0.0:
        raise RuntimeError(
            "Native T-spline CAD-derived Poisson field failed its residual oracle."
        )

    return {
        "scenario": "native-tspline-cad-analysis",
        "status": "passed",
        "resolution": resolution,
        "repeats": repeats,
        "identity": {
            "source": model.source_id,
            "source_revision": model.source_revision,
            "source_geometry": model.geometry.geometry_id,
            "source_t_mesh": surface.source_mesh.mesh_id,
            "source_patch": surface.source_definition_id,
            "native_plan": plan.plan_id,
            "native_result": native_result.result_id,
            "source_fidelity": certification.fidelity.certificate_id,
            "fidelity_revalidation": fidelity.certificate_id,
            "iga_plan": target_plan.plan_id,
            "iga_prepared": prepared.prepared_id,
        },
        "phases_seconds": {
            **source_phases,
            "cad_source_construction": cad_source_seconds,
            "meshing_domain_preparation": domain_preparation_seconds,
            "native_provider_preparation": provider_preparation_seconds,
            "native_execution": native_execution_seconds,
            "iga_plan_preparation": iga_plan_seconds,
            "iga_local_refinement": iga_refinement_seconds,
            "iga_discretization_preparation": iga_preparation_seconds,
            "equation_preparation": equation_preparation_seconds,
            "lowering_seconds": compiler["lowering_seconds"],
            "compilation_seconds": compiler["compile_seconds"],
            "first_solve_seconds": first_solve_seconds,
            "warm_solve_samples_seconds": warm_solve_seconds,
            "warm_solve_range_seconds": _range(warm_solve_seconds),
        },
        "compiler_memory_bytes": {
            "temporary": compiler["temporary_bytes"],
            "output": compiler["output_bytes"],
            "arguments": compiler["argument_bytes"],
            "generated_code": compiler["generated_code_bytes"],
        },
        "source_correctness": {
            "audit_passed": native_result.audit.passed,
            "compliance_passed": native_result.compliance.passed,
            "fidelity_status": fidelity.status,
            "source_chart_witness": surface_source.chart_witness.witness_id,
            "source_piece_ancestry_count": len(surface_source.source_piece_ids),
            "source_piece_ancestry_arity": sorted(
                {len(piece) for piece in surface_source.source_piece_ids}
            ),
        },
        "field_correctness": {
            "solve_successful": solve_successful,
            "normalized_residual": normalized_residual,
            "solution_norm": solution_norm,
        },
        "counts": {
            "source_cells": surface.mesh.cell_count,
            "source_t_junctions": surface.mesh.t_junction_count,
            "source_anchors": len(surface.mesh.anchors),
            "meshed_vertices": native_result.mesh.entity_set(0).count,
            "meshed_triangles": native_result.mesh.entity_set(2).count,
            "iga_cells": target_mesh.cell_count,
            "iga_t_junctions": target_mesh.t_junction_count,
            "full_tensor_cells": full_tensor_count,
        },
        "logical_retained_array_bytes": _array_bytes(
            (surface, model, native_result, prepared, finite_element_problem)
        ),
    }


def benchmark_device_bisection(resolution: int) -> dict[str, object]:
    """Five device adapt cycles inside one bucket of ``1000 * resolution`` slots.

    A disk of marks moves across a triangulated square: each cycle refines the
    cells inside the disk and coarsens every other cell, then assembles the
    masked P1 operators on the device layout. Reports lowering/compile time,
    compiler memory, warm execution per cycle, and the compilation count of
    every entry point after the cycles (one compilation per entry means no
    recompilation inside the bucket).
    """
    from phydrax.discretization import _adaptive_simplex as simplex
    from phydrax.discretization.fem import _generic as finite_element

    capacity = 1000 * resolution
    side = max(2, round((capacity / 32) ** 0.5))
    axis = np.linspace(0.0, 1.0, side + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    mesh = phx.discretization.CellMesh.from_triangles(points, _planar_triangles(side))
    source = phx.meshing.certify_cell_mesh(mesh, _contract())
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=capacity, cell_capacity=capacity
        ),
    )
    prepared, preparation = _timed(
        lambda: phx.meshing.prepare_adaptive_simplex(source, policy=policy)
    )
    layout, state = prepared.layout, prepared.state
    empty = jax.numpy.zeros((layout.cell_capacity,), dtype=jax.numpy.bool_)
    plan = phx.discretization.MaskedFiniteElementPlan(state.mesh)
    entries = {
        "refine": simplex._compiled_refine,
        "coarsen": simplex._compiled_coarsen,
        "masked_fe_assembly": finite_element.assemble_masked_finite_element,
    }
    costs = {
        "refine": _compiled_cost(entries["refine"], layout, state, empty),
        "coarsen": _compiled_cost(entries["coarsen"], layout, state, empty),
        "masked_fe_assembly": _compiled_cost(
            entries["masked_fe_assembly"], plan, state.mesh
        ),
    }
    cycles = []
    for cycle in range(5):
        center = jax.numpy.asarray((0.2 + 0.15 * cycle, 0.5))
        centroids = jax.numpy.mean(state.mesh.coordinates[state.mesh.cells], axis=1)
        inside = jax.numpy.linalg.norm(centroids - center, axis=1) < 0.12
        refine_marks = inside & state.mesh.cell_active
        coarsen_marks = ~inside & state.mesh.cell_active
        refined, refine_seconds = _timed(
            lambda state=state, refine_marks=refine_marks: jax.block_until_ready(
                simplex.refine_adaptive_simplex(layout, state, refine_marks)
            )
        )
        coarsened, coarsen_seconds = _timed(
            lambda refined=refined, coarsen_marks=coarsen_marks: jax.block_until_ready(
                simplex.coarsen_adaptive_simplex(layout, refined.state, coarsen_marks)
            )
        )
        state = coarsened.state
        _, assembly_seconds = _timed(
            lambda state=state: jax.block_until_ready(
                finite_element.assemble_masked_finite_element(plan, state.mesh)
            )
        )
        cycles.append(
            {
                "refine_seconds": refine_seconds,
                "coarsen_seconds": coarsen_seconds,
                "masked_fe_assembly_seconds": assembly_seconds,
                "refine_status": int(refined.report.status),
                "coarsen_status": int(coarsened.report.status),
                "bisections": int(refined.report.operations),
                "restored_cells": int(coarsened.report.operations),
                "active_cells": int(jax.numpy.sum(state.mesh.cell_active)),
                "allocated_cells": int(state.cursors[1]),
            }
        )
    # ty: ignore[unresolved-attribute]
    compilations = {name: entry._cached._cache_size() for name, entry in entries.items()}
    result, commit_seconds = _timed(
        lambda: phx.meshing.commit_adaptive_simplex(prepared, state)
    )
    warm = cycles[1:]
    return {
        "resolution": resolution,
        "capacity": {"vertices": capacity, "cells": capacity},
        "source_cells": 2 * side * side,
        "compiled": costs,
        "cycles": cycles,
        "compilations_after_cycles": compilations,
        "stages_seconds": {
            "prepare": preparation,
            "warm_refine_mean": float(np.mean([c["refine_seconds"] for c in warm])),
            "warm_coarsen_mean": float(np.mean([c["coarsen_seconds"] for c in warm])),
            "warm_masked_fe_assembly_mean": float(
                np.mean([c["masked_fe_assembly_seconds"] for c in warm])
            ),
            "commit": commit_seconds,
        },
        "logical_state_bytes": _array_bytes(state),
        "committed_cells": int(result.target.audit.entity_counts[-1]),
    }


def benchmark_local_metric(resolution: int) -> dict[str, object]:
    """Anisotropic metric adaptation of a lattice of about ``resolution`` triangles.

    The analytic layer metric (size 0.04 across x = 1/2 growing to 0.3, 0.25
    along y) is fixed, so the unit-mesh target is independent of ``resolution``
    while the source lattice and the split/collapse work to reach the target scale
    with it. ``NATIVE_METRIC_2D`` is host-only: certification and metric binding
    are host preparation and the route is timed as host execution.
    ``DEVICE_METRIC_2D`` runs the same request in one compiled executable
    (``adapt_device_metric``) inside a bucket whose capacities grow with the
    source: host preparation, lowering, compile, cold and warm execution, the
    host commit, compiler memory, and retained device state bytes.
    """
    from phydrax.meshing import _device_metric as device_metric

    side = max(2, round((resolution / 2) ** 0.5))
    axis = np.linspace(0.0, 1.0, side + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    mesh = phx.discretization.CellMesh.from_triangles(points, _planar_triangles(side))

    def prepare_host() -> Any:
        source = phx.meshing.certify_cell_mesh(mesh, _contract())
        certified = source.mesh
        rows = np.argsort(np.asarray(certified.vertex_global_ids), kind="stable")
        sizes = np.full((rows.shape[0], 2), 0.25, dtype=np.float64)
        coordinates = np.asarray(certified.coordinates, dtype=np.float64)[rows]
        sizes[:, 0] = 0.04 + 0.52 * np.abs(coordinates[:, 0] - 0.5)
        vertices = certified.entity_set(0)
        scope = phx.meshing.MeshingScope(
            certified.mesh_id,
            certified.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            np.asarray(certified.vertex_global_ids, dtype=np.int64),
        )
        metric = phx.meshing.MeshMetricField(
            scope,
            np.eye(2, dtype=np.float64) * sizes[:, None, :] ** -2,
            minimum_size=0.04,
            maximum_size=0.31,
            maximum_anisotropy=6.5,
        )
        return source, phx.meshing.MetricMeshAdaptation(metric)

    (source, request), host_preparation = _timed(prepare_host)
    native, native_execution = _timed(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source,
                request,
                policy=phx.meshing.MeshAdaptationPolicy(
                    phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_2D
                ),
            )
        )
    )
    vertex_count = points.shape[0]
    cell_count = 2 * side * side
    capacity = {"vertices": 256 + 2 * vertex_count, "cells": 512 + 8 * cell_count}
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.DEVICE_METRIC_2D,
        device_policy=phx.discretization.AdaptiveSimplexPolicy(
            vertex_capacity=capacity["vertices"], cell_capacity=capacity["cells"]
        ),
    )
    prepared, device_preparation = _timed(
        lambda: phx.meshing.prepare_device_metric_adaptation(
            source, request, policy=policy
        )
    )
    layout, state = prepared.layout, prepared.state
    cost = _compiled_cost(device_metric._compiled_adapt, layout, state)
    cold, cold_seconds = _timed(
        lambda: jax.block_until_ready(phx.meshing.adapt_device_metric(layout, state))
    )
    warm, warm_seconds = _timed(
        lambda: jax.block_until_ready(phx.meshing.adapt_device_metric(layout, state))
    )
    report = jax.device_get(warm.report)
    if bool(report.failed):
        raise RuntimeError(
            f"Device metric passes failed with status {int(report.status)}."
        )
    device, commit_seconds = _timed(
        lambda: phx.meshing.commit_device_metric_adaptation(prepared, warm.state)
    )
    if not native.target.audit.passed or not device.target.audit.passed:
        raise RuntimeError("A local metric adaptation target failed certification.")
    return {
        "resolution": resolution,
        "source": {"vertices": vertex_count, "cells": cell_count},
        "capacity": capacity,
        "native": {
            "status": native.status.value,
            "vertices": native.target.mesh.coordinates.shape[0],
            "cells": native.target.mesh.entity_set(2).count,
            "passes": native.evidence.passes,
            "splits": native.evidence.splits,
            "collapses": native.evidence.collapses,
            "flips": native.evidence.flips,
            "relocations": native.evidence.relocations,
            "unit_fraction": native.evidence.unit_fraction,
        },
        "device": {
            "status": device.status.value,
            "status_flags": int(report.status),
            "vertices": device.target.mesh.coordinates.shape[0],
            "cells": device.target.mesh.entity_set(2).count,
            "passes": int(report.passes),
            "splits": int(report.splits),
            "collapses": int(report.collapses),
            "flips": int(report.flips),
            "relocations": int(report.relocations),
            "unit_fraction": float(report.unit_fraction),
            "cold_warm_identical": bool(
                np.array_equal(
                    np.asarray(cold.state.mesh.coordinates),
                    np.asarray(warm.state.mesh.coordinates),
                )
            ),
        },
        "compiled": {"adapt_device_metric": cost},
        # ty: ignore[unresolved-attribute]
        "compilations_after_calls": device_metric._compiled_adapt._cached._cache_size(),
        "stages_seconds": {
            "host_preparation": host_preparation,
            "native_host_execution": native_execution,
            "device_host_preparation": device_preparation,
            "device_lowering": cost["lowering_seconds"],
            "device_compile": cost["compile_seconds"],
            "device_cold_execution": cold_seconds,
            "device_warm_execution": warm_seconds,
            "device_host_commit": commit_seconds,
        },
        "retained_bytes": {
            "device_state": _array_bytes(state),
            "native_target": _array_bytes(native.target),
            "device_target": _array_bytes(device.target),
        },
    }


@eqx.filter_jit
def _boundary_layer_cell_quality(coordinates: Any, connectivity: Any) -> Any:
    return {
        kind: _standard_block_quality(kind, coordinates[rows], None)
        for kind, rows in connectivity.items()
    }


def benchmark_boundary_layer(resolution: int) -> dict[str, object]:
    """Grow native advancing layers from a box wall of about `resolution` triangles.

    The closed box wall has convex ridges and corners, so growth exercises fan
    columns and corner patches. Wall construction and `prepare_boundary_layers`
    (visibility QP, proximity probes, exact certification) are host-only NumPy
    stages timed as host preparation and host execution. The compiled consumer is
    the sampled quality evaluation of every layer cell through one module-level
    entry, which reports lowering, compile, cold, and warm execution.
    """
    from tools.meshing_qualification.scenarios.providers import (
        _advancing_layer_control,
        _layer_box_wall,
    )

    count = max(1, round((resolution / 12) ** 0.5))
    schedule = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)
    wall, wall_seconds = _timed(lambda: _layer_box_wall(0.3, count))
    control = _advancing_layer_control(
        wall,
        schedule,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
    )
    layers, prepare_seconds = _timed(
        lambda: phx.meshing.prepare_boundary_layers(wall, control)
    )
    coordinates = layers.mesh.coordinates
    connectivity = {
        block.cell_kind: jax.numpy.asarray(block.vertices, dtype=jax.numpy.int32)
        for block in layers.mesh.blocks
    }
    compiled = _compiled_cost(_boundary_layer_cell_quality, coordinates, connectivity)
    _, cold_seconds = _timed(
        lambda: jax.block_until_ready(
            _boundary_layer_cell_quality(coordinates, connectivity)
        )
    )
    quality, warm_seconds = _timed(
        lambda: jax.block_until_ready(
            _boundary_layer_cell_quality(coordinates, connectivity)
        )
    )
    evidence = layers.evidence
    cells = sum(block.cell_count for block in layers.mesh.blocks)
    requested = np.asarray(schedule.thicknesses, dtype=np.float64)
    thickness_error = float(
        np.max(np.abs(np.asarray(evidence.achieved_thicknesses) - requested) / requested)
    )
    scaled_jacobian = float(
        min(np.min(np.asarray(block.scaled_jacobian)) for block in quality.values())
    )
    if (
        layers.validity.certified_valid_count != cells
        or not all(bool(np.all(block.sampled_valid)) for block in quality.values())
        or scaled_jacobian <= 0.0
        or thickness_error > 1.0e-9
    ):
        raise RuntimeError("Boundary-layer benchmark lost certification or schedule.")
    return {
        "resolution": resolution,
        "stages_seconds": {
            "wall_preparation": wall_seconds,
            "prepare_boundary_layers": prepare_seconds,
            "quality_lowering": compiled["lowering_seconds"],
            "quality_compile": compiled["compile_seconds"],
            "quality_cold": cold_seconds,
            "quality_warm": warm_seconds,
        },
        "compiled": {"layer_cell_quality": compiled},
        "retained_bytes": {
            "boundary_layer_mesh": _array_bytes(layers),
            "layer_cell_quality": _array_bytes(quality),
        },
        "counts": {
            "wall_faces": wall.blocks[0].cell_count,
            "wall_vertices": wall.coordinates.shape[0],
            "layer_vertices": coordinates.shape[0],
            "layer_cells": cells,
            "cells_by_kind": dict(evidence.cell_counts),
            "columns": evidence.column_count,
            "fan_columns": evidence.fan_column_count,
            "corner_patches": evidence.corner_patch_count,
        },
        "evidence": {
            "result_id": layers.result_id,
            "maximum_relative_thickness_error": thickness_error,
            "minimum_scaled_jacobian": scaled_jacobian,
        },
    }


@eqx.filter_jit
def _bisection_transfer_apply(transfer: Any, values: Any) -> Any:
    return transfer.apply(values)


def benchmark_host_bisection(resolution: int) -> dict[str, object]:
    """Host NATIVE_BISECTION of a marked disk/ball on about `resolution` cells.

    A right-triangle lattice of the unit square and Kuhn tetrahedra of the unit
    cube, each of about `resolution` cells (at least three lattice cells per
    axis, so the ball holds cells: every Kuhn tetrahedron of a 2^3 lattice has
    its centroid 0.47 from the center), are certified and every cell whose
    centroid lies within 0.3 of the domain center is marked for refinement.
    `prepare_mesh_adaptation` and `execute_mesh_adaptation` are host-only NumPy
    routes, timed as host preparation and host execution (execution includes the
    conformity closure, target certification, lineage, and transfer binding).
    The compiled consumer is the returned sparse P1 transfer applied to a vertex
    field through the module-level `_bisection_transfer_apply` entry: lowering,
    compile, cold and warm execution, compiler bytes, and retained transfer bytes.
    """
    planar = max(3, round((resolution / 2) ** 0.5))
    spatial = max(3, round((resolution / 6) ** (1 / 3)))
    axis = np.linspace(0.0, 1.0, planar + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    meshes = {
        "triangles": phx.discretization.CellMesh.from_triangles(
            points, _planar_triangles(planar)
        ),
        "tetrahedra": phx.discretization.CellMesh.from_tetrahedra(
            *_kuhn_tetrahedra(spatial)
        ),
    }
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    )
    stages, compiled, retained, counts = {}, {}, {}, {}
    for name, mesh in meshes.items():
        source, certify_seconds = _timed(
            lambda mesh=mesh: phx.meshing.certify_cell_mesh(mesh, _contract())
        )
        coordinates = np.asarray(mesh.coordinates)
        centroids = coordinates[np.asarray(mesh.blocks[0].vertices)].mean(axis=1)
        inside = np.linalg.norm(centroids - 0.5, axis=1) < 0.3
        marks = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)[inside]
        request = phx.meshing.MarkedMeshAdaptation(marks)
        prepared, prepare_seconds = _timed(
            lambda source=source, request=request: phx.meshing.prepare_mesh_adaptation(
                source, request, policy=policy
            )
        )
        result, execute_seconds = _timed(
            lambda prepared=prepared: phx.meshing.execute_mesh_adaptation(prepared)
        )
        if (
            result.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
            or not result.target.audit.passed
        ):
            raise RuntimeError(f"Host bisection of the {name} failed.")
        transfer = result.transfer
        slope = np.linspace(0.5, 1.5, coordinates.shape[1])
        values = jax.numpy.asarray(coordinates @ slope)
        compiled[name] = _compiled_cost(_bisection_transfer_apply, transfer, values)
        moved, cold_seconds = _timed(
            lambda transfer=transfer, values=values: jax.block_until_ready(
                _bisection_transfer_apply(transfer, values)
            )
        )
        _, warm_seconds = _timed(
            lambda transfer=transfer, values=values: jax.block_until_ready(
                _bisection_transfer_apply(transfer, values)
            )
        )
        expected = np.asarray(result.target.mesh.coordinates) @ slope
        if float(np.max(np.abs(np.asarray(moved) - expected))) > 1.0e-12:
            raise RuntimeError(f"The {name} bisection transfer lost P1 reproduction.")
        stages |= {
            f"{name}_host_certify": certify_seconds,
            f"{name}_host_prepare": prepare_seconds,
            f"{name}_host_execute": execute_seconds,
            f"{name}_transfer_lowering": compiled[name]["lowering_seconds"],
            f"{name}_transfer_compile": compiled[name]["compile_seconds"],
            f"{name}_transfer_cold": cold_seconds,
            f"{name}_transfer_warm": warm_seconds,
        }
        retained[name] = {
            "transfer": _array_bytes(transfer),
            "hierarchy": _array_bytes(result.hierarchy),
            "target_mesh": _array_bytes(result.target.mesh),
        }
        counts[name] = {
            "source_vertices": coordinates.shape[0],
            "source_cells": mesh.blocks[0].cell_count,
            "marked_cells": marks.shape[0],
            "target_vertices": result.target.mesh.coordinates.shape[0],
            "target_cells": result.target.mesh.blocks[0].cell_count,
            "bisections": result.evidence.bisections,
            "closure_iterations": result.evidence.closure_iterations,
            "transfer_entries": transfer.primal.coefficients.size,
        }
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "compiled": compiled,
        "retained_bytes": retained,
        "counts": counts,
    }


@eqx.filter_jit
def _migrate_cell_data(transition: Any, values: Any) -> Any:
    return transition.transfer(values)


def _lattice_square(cells_per_axis: int, /) -> phx.meshing.CellMeshingResult:
    """Certified lattice triangles of the unit square."""
    axis = np.linspace(0.0, 1.0, cells_per_axis + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    return phx.meshing.certify_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            points, _planar_triangles(cells_per_axis)
        ),
        _contract(),
    )


def benchmark_repartition(resolution: int) -> dict[str, object]:
    """Partition about ``resolution`` triangles, refine a corner, and migrate.

    Host-only NumPy stages: certification, MORTON/HILBERT ownership (GRAPH too
    when METIS loads) with ghost construction, native bisection of the cells
    in one corner, and the HILBERT distribution transition onto the refined
    revision. The compiled consumer is the transition's ``transfer`` (send
    gather plus receive reduction) through ``_migrate_cell_data``: lowering,
    compile, cold and warm execution, and compiler memory.
    """
    from phydrax.meshing._metis import metis_identity

    side = max(2, round((resolution / 2) ** 0.5))
    source, certification_seconds = _timed(lambda: _lattice_square(side))
    part = phx.meshing.MeshPart("repartition", source)
    kind = phx.meshing.MeshPartitionKind
    parts = 4
    routes = [kind.MORTON, kind.HILBERT]
    # Loading the shared library is the only way to learn whether METIS is usable.
    try:
        metis = metis_identity()
    except phx.meshing.MetisUnavailableError as error:
        metis = f"missing-dependency: {error}"
    else:
        routes.append(kind.GRAPH)
    stages, distributions, partitions = {}, {}, {}
    stages["host_certification"] = certification_seconds
    for route in routes:
        policy = phx.meshing.MeshPartitionPolicy(route, parts, maximum_imbalance=1.1)
        distribution, seconds = _timed(
            lambda policy=policy: phx.meshing.prepare_mesh_distribution(
                part, policy=policy
            )
        )
        evidence = distribution.evidence
        distributions[route] = distribution
        stages[f"host_partition_{route.value}"] = seconds
        partitions[route.value] = {
            "distribution_id": distribution.distribution_id,
            "provenance": evidence.provenance,
            "imbalance": float(evidence.imbalance),
            "edge_cut": int(evidence.edge_cut),
            "halo_replicas": int(evidence.halo_replicas),
        }
    block = source.mesh.blocks[0]
    centroids = np.asarray(source.mesh.coordinates)[np.asarray(block.vertices)].mean(
        axis=1
    )
    marked = np.asarray(block.global_ids)[np.all(centroids < 0.3, axis=1)]
    adaptation, stages["host_refinement"] = _timed(
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source,
                phx.meshing.MarkedMeshAdaptation(marked),
                policy=phx.meshing.MeshAdaptationPolicy(
                    phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
                ),
            )
        )
    )
    if adaptation.lineage is None:
        raise RuntimeError("Repartition benchmark refinement left the mesh unchanged.")
    hilbert = distributions[kind.HILBERT]
    transition, stages["host_transition"] = _timed(
        lambda: phx.meshing.prepare_distribution_transition(
            hilbert,
            phx.meshing.MeshPart(part.name, adaptation.target),
            adaptation.lineage,
            policy=phx.meshing.MeshPartitionPolicy(kind.HILBERT, parts),
        )
    )
    values = jax.numpy.linspace(1.0, 2.0, hilbert.cell_global_ids.shape[0])
    compiled = {
        "migrate_cell_data": _compiled_cost(_migrate_cell_data, transition, values)
    }
    moved, cold_seconds = _timed(
        lambda: jax.block_until_ready(_migrate_cell_data(transition, values))
    )
    warm_seconds = [
        _timed(lambda: jax.block_until_ready(_migrate_cell_data(transition, values)))[1]
        for _ in range(3)
    ]
    reference = np.asarray(transition.transfer(values))
    if not np.array_equal(np.asarray(moved), reference) or not np.all(
        reference[np.asarray(transition.target_defined)] >= 1.0
    ):
        raise RuntimeError("Compiled migration disagreed with the eager transfer.")
    stages.update(
        lowering=compiled["migrate_cell_data"]["lowering_seconds"],
        compile=compiled["migrate_cell_data"]["compile_seconds"],
        cold_execution=cold_seconds,
        warm_execution=float(np.mean(warm_seconds)),
    )
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "compiled": compiled,
        "retained_bytes": {
            **{
                f"distribution_{route.value}": _array_bytes(value)
                for route, value in distributions.items()
            },
            "transition": _array_bytes(transition),
        },
        "partitions": partitions,
        "metis": metis,
        "counts": {
            "source_cells": hilbert.cell_global_ids.shape[0],
            "target_cells": transition.target.cell_global_ids.shape[0],
            "marked_cells": marked.shape[0],
            "parts": parts,
            "rebalanced": transition.rebalanced,
            "migrated_cells": int(transition.migrated_cells),
            "migration_volume": float(transition.migration_volume),
            "message_slots": transition.send.capacity,
            "imbalance_before": float(hilbert.evidence.imbalance),
            "imbalance_after": float(transition.target.evidence.imbalance),
            "edge_cut_after": int(transition.target.evidence.edge_cut),
            "halo_replicas_after": int(transition.target.evidence.halo_replicas),
        },
    }


_PROVIDER_WORKERS: tuple[tuple[phx.config.WorkerSetting, str], ...] = (
    ("mmg", "phydrax-mmg-worker"),
    ("omega_h", "phydrax-omega-h-worker"),
)


def _provider_worker() -> tuple[str, str] | None:
    """First installed provider worker, resolved exactly as its provider does."""
    workers = phx.config.snapshot().workers
    for provider, default in _PROVIDER_WORKERS:
        configured = workers.get(provider)
        located = shutil.which(default if configured is None else configured)
        if located is not None:
            return provider, located
    return None


@eqx.filter_jit
def _adapted_quality(coordinates: Any, connectivity: Any) -> Any:
    return {
        kind: _standard_block_quality(kind, coordinates[rows], None)
        for kind, rows in connectivity.items()
    }


def benchmark_provider_worker(resolution: int) -> dict[str, object]:
    """Cold (fresh launch per call) versus persistent (reused session) worker calls.

    A real Mmg worker (preferred) or Omega_h worker adapts about ``resolution``
    lattice triangles of the unit square to a uniform metric of half the lattice
    spacing. Host-only stages: preparation, worker launch with its identity
    probe, and every worker call (exchange write, native adaptation, verified
    exchange read, certification). The compiled consumer is quality evaluation
    of the adapted mesh through ``_adapted_quality``.
    """
    route = _provider_worker()
    if route is None:
        raise RuntimeError("No provider worker is installed.")
    provider_name, executable = route
    side = max(2, round((resolution / 2) ** 0.5))
    size = 0.5 / side

    def prepare() -> Any:
        source = _lattice_square(side)
        scope = phx.meshing.MmgProvider.vertex_scope(source.mesh)
        metric = phx.meshing.MeshMetricField(
            scope,
            np.tile(np.eye(2) / size**2, (scope.entity_ids.shape[0], 1, 1)),
            minimum_size=size,
            maximum_size=size,
        )
        if provider_name == "mmg":
            plan = phx.meshing.MmgProvider(executable=executable).plan(
                source, metric=metric
            )
            return source, plan, plan._encoding.arrays
        from phydrax.meshing.providers._omega_h import _carrier

        arrays = _carrier(source, metric, (), phx.meshing.OmegaHOptions()).arrays
        return source, metric, arrays

    (source, request, arrays), preparation_seconds = _timed(prepare)

    def open_provider() -> Any:
        if provider_name == "mmg":
            provider = phx.meshing.MmgProvider(executable=executable)
            return provider, provider.worker
        provider = phx.meshing.OmegaHProvider(executable)
        return provider, provider.worker(1)

    def execute(provider: Any) -> Any:
        if provider_name == "mmg":
            return provider.execute(request).mesh
        return provider.execute(source, request).target

    provider, worker = open_provider()
    with provider:
        _, cold_launch = _timed(lambda: worker.identity)
        _, cold_call = _timed(lambda: execute(provider))
        cold_calls = list(worker.session().calls)
    provider, worker = open_provider()
    with provider:
        identity, persistent_launch = _timed(lambda: worker.identity)
        adapted, persistent_first = _timed(lambda: execute(provider))
        warm_calls = [_timed(lambda: execute(provider))[1] for _ in range(3)]
        calls = list(worker.session().calls)
    if (
        worker.launches != 1
        or len({call["session_id"] for call in calls}) != 1
        or [call["sequence"] for call in calls] != [1, 2, 3, 4]
        or not adapted.audit.passed
        or adapted.mesh.entity_set(2).count <= source.mesh.entity_set(2).count
    ):
        raise RuntimeError("Persistent provider worker relaunched or did not refine.")
    mesh = adapted.mesh
    connectivity = {
        block.cell_kind: jax.numpy.asarray(block.vertices, dtype=jax.numpy.int32)
        for block in mesh.blocks
    }
    compiled = {
        "adapted_quality": _compiled_cost(
            _adapted_quality, mesh.coordinates, connectivity
        )
    }
    _, cold_quality = _timed(
        lambda: jax.block_until_ready(_adapted_quality(mesh.coordinates, connectivity))
    )
    _, warm_quality = _timed(
        lambda: jax.block_until_ready(_adapted_quality(mesh.coordinates, connectivity))
    )
    cold_seconds = cold_launch + cold_call
    warm_seconds = float(np.mean(warm_calls))
    return {
        "resolution": resolution,
        "provider": provider_name,
        "worker": {
            "executable": identity.executable,
            "executable_sha256": identity.executable_sha256,
            "reported": dict(identity.reported),
            "ranks": identity.ranks,
            "memory_enforcement": identity.memory_enforcement,
            "identity_id": identity.identity_id,
        },
        "stages_seconds": {
            "host_preparation": preparation_seconds,
            "cold_launch_and_identity": cold_launch,
            "cold_first_call": cold_call,
            "persistent_launch_and_identity": persistent_launch,
            "persistent_first_call": persistent_first,
            "persistent_warm_call": warm_seconds,
            "lowering": compiled["adapted_quality"]["lowering_seconds"],
            "compile": compiled["adapted_quality"]["compile_seconds"],
            "cold_execution": cold_quality,
            "warm_execution": warm_quality,
        },
        "latency": {
            "cold_call_seconds": cold_seconds,
            "persistent_warm_call_seconds": warm_seconds,
            "cold_over_persistent": cold_seconds / warm_seconds,
        },
        "compiled": compiled,
        "exchange_bytes": {
            "input": sum(value.nbytes for value in arrays.values()),
            "output": [call["output_bytes"] for call in calls],
        },
        "worker_peak_rss_bytes": max(
            call["peak_rss_bytes"] for call in (*cold_calls, *calls)
        ),
        "calls": {"cold": cold_calls, "persistent": calls},
        "retained_bytes": _array_bytes(adapted),
        "counts": {
            "source_cells": source.mesh.entity_set(2).count,
            "target_cells": mesh.entity_set(2).count,
            "persistent_calls": len(calls),
            "launches": worker.launches,
        },
    }


def _provider_worker_case(resolution: int, repeats: int, /) -> dict[str, object]:
    """Missing-dependency record unless a real provider worker is installed."""
    if _provider_worker() is None:
        return {
            "status": "missing-dependency",
            "dependency": "phydrax-mmg-worker or phydrax-omega-h-worker",
            "detail": (
                "Build native/providers/mmg against Mmg 5.8 or native/providers/"
                "omega_h against Omega_h (docs/guides_meshing.md), then set "
                "PHYDRAX_MMG_WORKER or PHYDRAX_OMEGA_H_WORKER, or put the worker "
                "on PATH."
            ),
            "resolution": resolution,
        }
    return _repeated_case(benchmark_provider_worker, resolution, repeats)


def benchmark_optimization(resolution: int) -> dict[str, object]:
    """Target-matrix optimization of about `resolution` jittered triangles and tets.

    A jittered unit-square triangle lattice and a jittered Kuhn tetrahedral lattice
    of about `resolution` cells each are optimized toward their unjittered lattice
    shapes (``SHAPE_SIZE``, default ``ProjectedLBFGS``) with fixed boundary
    vertices. Plan construction is host-only preparation. ``optimize_cell_mesh`` is
    host-orchestrated (energy and inversion checks, audit, certification) around
    the library's module-level compiled minimization entry, whose lowering and
    compile are measured explicitly; cold and warm execution time the public call
    (the cold call compiles), and warm minimization times the compiled entry alone.
    """
    from phydrax.meshing._optimization import _minimize_free_coordinates

    planar = max(3, round((resolution / 2) ** 0.5))
    spatial = max(3, round((resolution / 6) ** (1 / 3)))
    axis = np.linspace(0.0, 1.0, planar + 1)
    square = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))
    cube, tetrahedra = _kuhn_tetrahedra(spatial)
    interior = np.all((cube > 0.0) & (cube < 1.0), axis=1)
    jitter = np.random.default_rng(7).uniform(-0.15, 0.15, cube.shape) / spatial
    meshes = {
        "2d": (_jittered_unit_square_triangles(planar, 5), square),
        "3d": (
            phx.discretization.CellMesh.from_tetrahedra(
                cube + np.where(interior[:, None], jitter, 0.0), tetrahedra
            ),
            cube,
        ),
    }
    contract = _contract()
    stages, compiled, retained, counts, quality, minimization = {}, {}, {}, {}, {}, {}
    for name, (mesh, lattice) in meshes.items():
        fixed = np.any((lattice == 0.0) | (lattice == 1.0), axis=1)
        plan, preparation = _timed(
            lambda mesh=mesh, lattice=lattice, fixed=fixed: (
                phx.meshing.TargetMatrixOptimizationPlan(
                    mesh, target_coordinates=lattice, fixed_vertices=fixed
                )
            )
        )
        free = jax.numpy.asarray(np.flatnonzero(~fixed), dtype=jax.numpy.int32)
        arguments = (
            plan.energy,
            plan.method,
            plan.termination,
            jax.numpy.asarray(mesh.coordinates, dtype=jax.numpy.float64),
            free,
            plan.lower[free],
            plan.upper[free],
        )
        cost = _compiled_cost(_minimize_free_coordinates, *arguments)
        result, cold = _timed(
            lambda plan=plan: phx.meshing.optimize_cell_mesh(plan, contract)
        )
        _, warm = _timed(lambda plan=plan: phx.meshing.optimize_cell_mesh(plan, contract))
        _, warm_minimization = _timed(
            lambda arguments=arguments: jax.block_until_ready(
                _minimize_free_coordinates(*arguments)
            )
        )
        before = float(
            np.min(np.asarray(phx.meshing.evaluate_cell_quality(mesh).mean_ratios))
        )
        if not result.accepted or result.result.quality.minimum_mean_ratio <= before:
            raise RuntimeError(
                f"{name} optimization failed or did not improve the minimum mean "
                f"ratio: {result.status.value}."
            )
        stages.update(
            {
                f"{name}_host_preparation": preparation,
                f"{name}_lowering": cost["lowering_seconds"],
                f"{name}_compile": cost["compile_seconds"],
                f"{name}_cold_execution": cold,
                f"{name}_warm_execution": warm,
                f"{name}_warm_minimization": warm_minimization,
            }
        )
        compiled[name] = {"minimize_free_coordinates": cost}
        retained[name] = {
            "plan": _array_bytes(plan),
            "result": _array_bytes(result.result),
        }
        counts[name] = {
            "cells": mesh.blocks[0].cell_count,
            "vertices": mesh.coordinates.shape[0],
            "free_vertices": free.shape[0],
        }
        after = result.result.quality
        quality[name] = {
            "minimum_mean_ratio_before": before,
            "minimum_mean_ratio_after": after.minimum_mean_ratio,
            "minimum_scaled_jacobian_after": after.minimum_scaled_jacobian,
        }
        minimization[name] = {
            "status": result.status.value,
            "optimizer_status": result.optimizer_status.name,
            "iterations": result.iterations,
            "accepted_steps": result.accepted_steps,
            "initial_objective": result.initial_objective,
            "final_objective": result.final_objective,
        }
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "compiled": compiled,
        "retained_bytes": retained,
        "counts": counts,
        "quality": quality,
        "minimization": minimization,
    }


def _squeezed(points: np.ndarray, /) -> np.ndarray:
    """Gaussian contraction toward the unit-cube center.

    Its radial stretch ``1 - 1.2 exp(-8 r^2) (1 - 16 r^2)`` inverts a small core
    and nearly degenerates a surrounding shell, so certification exercises valid,
    invalid, and subdivided cells.
    """
    offset = points - 0.5
    weight = np.exp(-8.0 * np.sum(offset * offset, axis=1, keepdims=True))
    return points - 1.2 * offset * weight


def _cube_hexahedra(cells_per_axis: int, /) -> phx.discretization.CellMesh:
    count = cells_per_axis
    axis = np.linspace(0.0, 1.0, count + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )
    index = np.arange((count + 1) ** 3, dtype=np.int32).reshape((count + 1,) * 3)
    corners = np.stack(
        [
            index[a : a + count, b : b + count, c : c + count].reshape((-1,))
            for a, b, c in _HEXAHEDRON_CORNERS
        ],
        axis=1,
    )
    return phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hexahedra", "hexahedron", corners),)
    )


@eqx.filter_jit
def _mapped_cell_measures(discretization: Any, coordinates: Any) -> Any:
    """Mapped measure of every cell and the reference-cell volume of every block."""
    return tuple(
        (block.measure, jax.numpy.sum(block.reference_weights))
        for block in discretization.evaluate_geometry("x", coordinates)
    )


def _certification_route(mesh: Any, degree: int, name: str, /) -> Any:
    """Host certification and compiled measure consumer of one curved geometry."""
    from phydrax.meshing._curving import _straight_geometry

    def prepare() -> Any:
        straight = _straight_geometry(mesh, degree)
        names = straight.block_names
        return phx.discretization.CellGeometrySpec(
            dict(zip(names, straight.elements, strict=True)),
            dict(zip(names, straight.geometry_dofs, strict=True)),
            _squeezed(np.asarray(straight.coordinates, dtype=np.float64)),
        )

    geometry, preparation = _timed(prepare)
    certificate, certification = _timed(
        lambda: phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "x", phx.discretization.lagrange_element(mesh.blocks[0].cell_kind, 1)
    )
    discretization, consumer_preparation = _timed(
        lambda: phx.discretization.FiniteElementPlan(
            mesh, field, coordinate_spec=geometry
        ).prepare()
    )
    coordinates = geometry.coordinates
    compiled = _compiled_cost(_mapped_cell_measures, discretization, coordinates)
    ((measures, reference),), cold = _timed(
        lambda: jax.block_until_ready(_mapped_cell_measures(discretization, coordinates))
    )
    _, warm = _timed(
        lambda: jax.block_until_ready(_mapped_cell_measures(discretization, coordinates))
    )
    # A certified cell's measure is the integral of its positive determinant, so
    # it lies in the certificate's determinant enclosure times the reference volume.
    certified = (
        np.asarray(certificate.status)
        == phx.discretization.CellValidityStatus.CERTIFIED_VALID
    )
    measure = np.asarray(measures)[certified]
    determinant_lower = np.asarray(certificate.determinant_lower)[certified]
    lower = determinant_lower * float(reference)
    upper = np.asarray(certificate.determinant_upper)[certified] * float(reference)
    slack = 1.0e-9 * upper
    outside = (measure < lower - slack) | (measure > upper + slack)
    if np.any(lower <= 0.0) or np.any(outside):
        raise RuntimeError(f"{name}: a certified measure leaves its enclosure.")
    depth = np.asarray(certificate.depth)
    stages = {
        "host_preparation": preparation,
        "host_certification": certification,
        "consumer_host_preparation": consumer_preparation,
        "consumer_lowering": compiled["lowering_seconds"],
        "consumer_compile": compiled["compile_seconds"],
        "consumer_cold": cold,
        "consumer_warm": warm,
    }
    route = {
        "cells": depth.shape[0],
        "geometry_nodes": coordinates.shape[0],
        "certified_valid": certificate.certified_valid_count,
        "invalid": certificate.invalid_count,
        "unresolved": certificate.unresolved_count,
        "depth_histogram": np.bincount(depth).tolist(),
        "validity_decisions": _validity_decisions(certificate),
        "minimum_certified_determinant_lower_bound": (
            float(np.min(determinant_lower)) if determinant_lower.size else None
        ),
        "certified_mapped_volume": float(np.sum(measure)),
        "certificate_retained_bytes": _array_bytes(certificate),
        "geometry_retained_bytes": _array_bytes(geometry),
    }
    return stages, compiled, route


def _incremental_validity_record(
    mesh: phx.discretization.CellMesh,
    geometry: phx.discretization.CellGeometrySpec,
    computed: ComputedCellValidity,
    /,
) -> dict[str, object]:
    """Measure one local coordinate edit with separate parent replay and reuse."""
    coordinates = np.array(geometry.coordinates, dtype=np.float64, copy=True)
    coordinates[0, 0] += 1.0e-4
    target = phx.discretization.CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        coordinates,
    )
    affected = sum(
        np.count_nonzero(np.any(np.asarray(route) == 0, axis=1))
        for route in geometry.geometry_dofs
    )
    validated, parent_replay = _timed_host(
        lambda: ValidatedCellValidityParent(computed.certificate)
    )
    full, full_cost = _timed_host(
        lambda: phx.discretization.certify_cell_geometry_validity(target, mesh=mesh)
    )
    incremental, incremental_cost = _timed_host(
        lambda: phx.discretization.certify_cell_geometry_validity(
            target, mesh=mesh, parent=validated
        )
    )
    live, live_cost = _timed_host(
        lambda: phx.discretization.certify_cell_geometry_validity(
            target, mesh=mesh, parent=computed
        )
    )
    if not full.all_certified:
        raise RuntimeError("The local benchmark edit produced invalid geometry.")
    if not _same_tree_content(full, incremental) or not _same_tree_content(full, live):
        raise RuntimeError("Incremental validity differs from the complete target proof.")
    if full.geometry_id == computed.certificate.geometry_id:
        raise RuntimeError("The benchmark target did not change its coordinate map.")
    return {
        "target_geometry": full.geometry_id,
        "target_certificate": full.certificate_id,
        "edited_coordinate_rows": 1,
        "incident_cells": int(affected),
        "unaffected_cells": mesh.connectivity.cell_count - int(affected),
        "complete_certificate_parity": True,
        "phases": {
            "parent_replay": parent_replay,
            "full_target": full_cost,
            "validated_parent_target": incremental_cost,
            "live_computed_parent_target": live_cost,
        },
        "validity_decisions": _validity_decisions(full),
        "timing_scope": (
            "Synchronized host wall and process CPU time. Parent replay is separate; "
            "live reuse requires the actual computed artifact from this operation. "
            "Incident-cell counts describe the edit, not measured proof-cache hits."
        ),
    }


def benchmark_global_embedding(resolution: int) -> dict[str, object]:
    """Scale exact global contact certification on a conforming cube lattice."""
    mesh, mesh_preparation = _timed(lambda: _cube_tetrahedra(resolution))
    geometry, geometry_preparation = _timed(
        lambda: phx.discretization.CellGeometrySpec.affine(mesh)
    )
    computed, local_cost = _timed_host(
        lambda: ComputedCellValidity(
            geometry, mesh, phx.discretization.CellValidityPolicy(), None
        )
    )
    validity = computed.certificate
    if not validity.all_certified:
        raise RuntimeError("Embedding benchmark requires positive source tetrahedra.")
    certificate, cold = _timed(
        lambda: phx.geometry.certify_global_embedding(mesh, geometry, validity)
    )
    repeated, warm = _timed(
        lambda: phx.geometry.certify_global_embedding(mesh, geometry, validity)
    )
    handed_on, handoff_cost = _timed_host(
        lambda: phx.geometry.certify_global_embedding(
            mesh, geometry, validity, validated_validity=computed
        )
    )
    if (
        certificate.status != "certified"
        or not _same_tree_content(repeated, certificate)
        or not _same_tree_content(handed_on, certificate)
    ):
        raise RuntimeError("Embedding benchmark did not reproduce a certified proof.")
    incremental = _incremental_validity_record(mesh, geometry, computed)
    return {
        "resolution": resolution,
        "counts": {
            "cells_per_axis": resolution,
            "cells": mesh.connectivity.cell_count,
            "vertices": mesh.coordinates.shape[0],
            "boundary_facets": certificate.boundary_facet_count,
        },
        "identity": {
            "mesh": mesh.mesh_id,
            "geometry": validity.geometry_id,
            "validity": validity.certificate_id,
            "embedding": certificate.certificate_id,
        },
        "stages_seconds": {
            "mesh_preparation": mesh_preparation,
            "geometry_preparation": geometry_preparation,
            "local_validity": local_cost["wall_seconds"],
            "global_embedding_cold": cold,
            "global_embedding_warm": warm,
            "global_embedding_live_handoff": handoff_cost["wall_seconds"],
        },
        "host_cpu_seconds": {
            "local_validity": local_cost["cpu_seconds"],
            "global_embedding_live_handoff": handoff_cost["cpu_seconds"],
        },
        "validity_decisions": _validity_decisions(validity),
        "incremental_validity": incremental,
        "proof_handoff": {
            "complete_certificate_parity": True,
            "standalone_embedding": "replays supplied validity independently",
            "live_embedding": "uses the actual operation-local computed validity",
        },
        "retained_bytes": {
            "mesh": _array_bytes(mesh),
            "geometry": _array_bytes(geometry),
            "embedding": _array_bytes(certificate),
        },
        "compiler": None,
        "execution": "host-exact; no compiled certification kernel",
    }


def benchmark_high_order_certification(resolution: int) -> dict[str, object]:
    """Bernstein validity certification of P2/P3 tetrahedra and Q2 hexahedra.

    About ``resolution`` Kuhn tetrahedra (P2 and P3) and hexahedra (Q2) of the
    unit cube get straight Lagrange geometry nodes moved by the smooth
    `_squeezed` map. Geometry preparation and `certify_cell_geometry_validity`
    (adaptive Bernstein subdivision) are host-only NumPy and timed as host
    stages. The compiled consumer is the exactly integrated mapped cell measure
    of the curved geometry (`FiniteElementDiscretization.evaluate_geometry`)
    through the module-level `_mapped_cell_measures` entry: lowering, compile,
    cold, and warm execution plus compiler bytes.
    """
    tetrahedra = _cube_tetrahedra(max(2, round((resolution / 6) ** (1 / 3))))
    hexahedra = _cube_hexahedra(max(2, round(resolution ** (1 / 3))))
    stages, compiled, routes = {}, {}, {}
    for name, mesh, degree in (
        ("p2_tetrahedra", tetrahedra, 2),
        ("p3_tetrahedra", tetrahedra, 3),
        ("q2_hexahedra", hexahedra, 2),
    ):
        route_stages, compiled[name], routes[name] = _certification_route(
            mesh, degree, name
        )
        stages.update({f"{name}_{key}": value for key, value in route_stages.items()})
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "compiled": compiled,
        "routes": routes,
    }


@eqx.filter_jit
def _device_orient2d(a: Any, b: Any, c: Any) -> Any:
    return phx.geometry.orient2d(a, b, c, mode=phx.geometry.PredicateMode.FILTERED_DEVICE)


@eqx.filter_jit
def _device_orient3d(a: Any, b: Any, c: Any, d: Any) -> Any:
    return phx.geometry.orient3d(
        a, b, c, d, mode=phx.geometry.PredicateMode.FILTERED_DEVICE
    )


@eqx.filter_jit
def _device_incircle(a: Any, b: Any, c: Any, d: Any) -> Any:
    return phx.geometry.incircle(
        a, b, c, d, mode=phx.geometry.PredicateMode.FILTERED_DEVICE
    )


@eqx.filter_jit
def _device_insphere(a: Any, b: Any, c: Any, d: Any, e: Any) -> Any:
    return phx.geometry.insphere(
        a, b, c, d, e, mode=phx.geometry.PredicateMode.FILTERED_DEVICE
    )


_PREDICATE_ROUTES = {
    "orient2d": (_device_orient2d, phx.geometry.orient2d, 3, 2),
    "orient3d": (_device_orient3d, phx.geometry.orient3d, 4, 3),
    "incircle": (_device_incircle, phx.geometry.incircle, 4, 2),
    "insphere": (_device_insphere, phx.geometry.insphere, 5, 3),
}


def _predicate_queries(count: int, seed: int, /) -> dict[str, tuple[np.ndarray, ...]]:
    """Random predicate rows; a quarter are rounded exact degeneracies.

    The nearly degenerate quarter rounds points on a segment, a plane, the unit
    circle, or the unit sphere, so the floating-point filters leave rows uncertain.
    """
    rng = np.random.default_rng(seed)
    near = count // 4
    rows = {
        name: rng.standard_normal((arity, count, width))
        for name, (_, _, arity, width) in _PREDICATE_ROUTES.items()
    }
    weights = rng.random((2, near, 1))
    for name in ("orient2d", "orient3d"):
        points = rows[name]
        spans = points[1:-1, :near] - points[0, :near]
        points[-1, :near] = points[0, :near] + np.sum(
            weights[: spans.shape[0]] * spans, axis=0
        )
    for name in ("incircle", "insphere"):
        points = rows[name][:, :near]
        rows[name][:, :near] = points / np.linalg.norm(points, axis=-1, keepdims=True)
    return {name: tuple(points) for name, points in rows.items()}


def _predicate_stages(
    name: str, host: tuple[np.ndarray, ...], device: tuple, /
) -> tuple[dict[str, float], dict[str, object], dict[str, int], int]:
    """Compiled device filter plus host FILTERED and EXACT routes of one predicate."""
    entry, predicate, _, _ = _PREDICATE_ROUTES[name]
    mode = phx.geometry.PredicateMode
    compiled = _compiled_cost(entry, *device)
    _, cold = _timed(lambda: jax.block_until_ready(entry(*device)))
    result, warm = _timed(lambda: jax.block_until_ready(entry(*device)))
    filtered, filtered_seconds = _timed(lambda: predicate(*host, mode=mode.FILTERED))
    exact, exact_seconds = _timed(lambda: predicate(*host, mode=mode.EXACT))
    signs, certain = np.asarray(result.signs), np.asarray(result.certain)
    wrong = np.count_nonzero(certain & (signs != exact.signs)) + np.count_nonzero(
        filtered.certain & (filtered.signs != exact.signs)
    )
    if wrong or not np.all(exact.certain):
        raise RuntimeError(f"{name}: a filter certified a sign that EXACT contradicts.")
    stages = {
        f"{name}_lowering": compiled["lowering_seconds"],
        f"{name}_compile": compiled["compile_seconds"],
        f"{name}_device_cold": cold,
        f"{name}_device_warm": warm,
        f"{name}_filtered_host": filtered_seconds,
        f"{name}_exact_host": exact_seconds,
    }
    counts = {
        "queries": signs.size,
        "device_uncertain": int(np.count_nonzero(~certain)),
        "filtered_uncertain": int(np.count_nonzero(~filtered.certain)),
        "exact_zero": int(np.count_nonzero(exact.signs == 0)),
    }
    # ty: ignore[invalid-return-type]
    return stages, compiled, counts, _array_bytes(result)


def benchmark_predicates(resolution: int) -> dict[str, object]:
    """Filtered and exact predicates on about `resolution` queries each.

    Every predicate evaluates `resolution` rows, a quarter nearly degenerate.
    FILTERED_DEVICE runs through one module-level compiled entry per predicate
    (lowering, compile, cold, and warm execution with compiler memory); the
    host-only NumPy FILTERED route and the meshcore EXACT route are timed as host
    execution. Delaunay 2D/3D construction of `resolution` random points is a
    host-only meshcore stage timed as host preparation.
    """
    from phydrax._meshcore import meshcore_available

    if not meshcore_available():
        raise RuntimeError(
            "The predicates benchmark requires the native meshcore library; set "
            "PHYDRAX_MESHCORE_LIBRARY or install phydrax[meshcore]."
        )
    count = max(16, resolution)
    queries, preparation = _timed(lambda: _predicate_queries(count, resolution))
    device, transfer = _timed(
        lambda: jax.block_until_ready(
            {
                name: tuple(jax.numpy.asarray(array) for array in arrays)
                for name, arrays in queries.items()
            }
        )
    )
    stages = {"query_preparation": preparation, "device_transfer": transfer}
    compiled, counts, retained = {}, {}, {}
    for name in _PREDICATE_ROUTES:
        predicate_stages, compiled[name], counts[name], retained[name] = (
            _predicate_stages(name, queries[name], device[name])
        )
        stages.update(predicate_stages)
    rng = np.random.default_rng(resolution)
    planar_points, spatial_points = rng.random((count, 2)), rng.random((count, 3))
    planar, stages["delaunay_2d_host"] = _timed(
        lambda: phx.geometry.DelaunayTriangulation(planar_points)
    )
    spatial, stages["delaunay_3d_host"] = _timed(
        lambda: phx.geometry.DelaunayTriangulation(spatial_points)
    )
    retained["delaunay_2d"] = _array_bytes(planar)
    retained["delaunay_3d"] = _array_bytes(spatial)
    counts["delaunay_2d"] = {"points": count, "triangles": planar.simplices.shape[0]}
    counts["delaunay_3d"] = {"points": count, "tetrahedra": spatial.simplices.shape[0]}
    return {
        "resolution": resolution,
        "stages_seconds": stages,
        "compiled": compiled,
        "retained_bytes": retained,
        "counts": counts,
    }


def _installed_version(distribution: str, /) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _missing_gmsh_dependencies() -> tuple[str, ...]:
    return tuple(
        dependency for dependency in ("gmsh", "OCP") if find_spec(dependency) is None
    )


def _environment_identity() -> dict[str, object]:
    from phydrax._meshcore import meshcore_available, meshcore_identity

    available = meshcore_available()
    devices = [
        {
            "platform": device.platform,
            "kind": str(getattr(device, "device_kind", "")),
            "id": int(device.id),
        }
        for device in jax.devices()
    ]
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "phydrax": {
            "distribution_version": _installed_version("phydrax"),
            "module_origin": str(Path(phx.__file__).resolve()),
        },
        "packages": {
            name: _installed_version(name)
            for name in ("numpy", "jax", "equinox", "gmsh", "cadquery-ocp")
        },
        "jax": {
            "backend": jax.default_backend(),
            "devices": devices,
            "x64_enabled": bool(pxconfig.x64_enabled()),
            "compilation_cache_enabled": bool(pxconfig.compilation_cache_enabled()),
            "compilation_cache_directory": configuration_values().get(
                "jax_compilation_cache_dir"
            ),
        },
        "meshcore": {
            "available": available,
            "identity": meshcore_identity() if available else None,
        },
    }


def _periodic_power_component_lookup(
    full: Any, /
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Resolve each site/tet witness to its actual full-result component."""
    by_cell: list[list[int]] = [[] for _ in range(len(full.cell_sites))]
    for cell, tet in zip(full.piece_cells, full.piece_tets, strict=True):
        by_cell[int(cell)].append(int(tet))
    lookup = {}
    for cell, pieces in enumerate(by_cell):
        component = tuple(sorted(pieces))
        for tet in component:
            key = (int(full.cell_sites[cell]), tet)
            if key in lookup:
                raise ValueError("A full power site/tet has multiple component owners.")
            lookup[key] = component
    return lookup


def _periodic_power_selected_signature(
    result: Any,
    selected: np.ndarray,
    components: dict[tuple[int, int], tuple[int, ...]],
    /,
) -> tuple[Any, ...]:
    """Compare selected components without relying on native row numbering."""
    sites = np.asarray(result.cell_sites, dtype=np.int64)
    face_offsets = np.asarray(result.face_offsets, dtype=np.int64)
    face_vertices = np.asarray(result.face_vertices, dtype=np.int64)
    face_cells = np.asarray(result.face_cells, dtype=np.int64)
    witness_offsets = np.asarray(result.vertex_site_offsets, dtype=np.int64)
    witnesses = np.asarray(result.vertex_sites, dtype=np.int64)
    carriers = np.asarray(result.vertex_carriers, dtype=np.int64)
    vertex_bits = np.ascontiguousarray(result.vertices).view(np.uint64).reshape((-1, 3))
    piece_rows: list[list[int]] = [[] for _ in range(len(sites))]
    for cell, tet in zip(result.piece_cells, result.piece_tets, strict=True):
        piece_rows[int(cell)].append(int(tet))
    remote_sites = np.asarray(result.face_remote_sites, dtype=np.int64)
    remote_tets = np.asarray(result.face_remote_tets, dtype=np.int64)
    facets = np.asarray(result.face_facets, dtype=np.int64)
    selected_cells = np.flatnonzero(np.isin(sites, selected))
    face_rows: dict[int, list[tuple[Any, ...]]] = {
        int(cell): [] for cell in selected_cells
    }
    for face, owners in enumerate(face_cells):
        selected_owners = tuple(int(owner) for owner in owners if int(owner) in face_rows)
        if not selected_owners:
            continue
        vertices = []
        for vertex in face_vertices[face_offsets[face] : face_offsets[face + 1]]:
            row = int(vertex)
            vertices.append(
                (
                    tuple(int(value) for value in vertex_bits[row]),
                    tuple(
                        int(value)
                        for value in witnesses[
                            witness_offsets[row] : witness_offsets[row + 1]
                        ]
                    ),
                    tuple(int(value) for value in carriers[row]),
                )
            )
        forward = tuple(vertices)
        reverse = tuple(reversed(vertices))
        forward_key = min(forward[i:] + forward[:i] for i in range(len(forward)))
        reverse_key = min(reverse[i:] + reverse[:i] for i in range(len(reverse)))
        face_key = min(forward_key, reverse_key)
        orientation = 1 if forward_key < reverse_key else -1
        for owner in selected_owners:
            sign = orientation if int(owners[0]) == owner else -orientation
            other = int(owners[1]) if int(owners[0]) == owner else int(owners[0])
            if other >= 0:
                if not piece_rows[other]:
                    raise ValueError(
                        "A power face neighbor has no carrier component witness."
                    )
                neighbor = (int(sites[other]), piece_rows[other][0])
            elif remote_sites[face] >= 0:
                neighbor = (int(remote_sites[face]), int(remote_tets[face]))
            else:
                neighbor = None
            if neighbor is None:
                if facets[face] < 0:
                    raise ValueError(
                        "A power face lacks both exterior and remote evidence."
                    )
                counterpart: tuple[Any, ...] = ("exterior", int(facets[face]))
            else:
                component = components.get(neighbor)
                if component is None:
                    raise ValueError("A remote power face has no full-result component.")
                counterpart = ("cell", neighbor[0], component, int(facets[face]))
            face_rows[owner].append((face_key, sign, counterpart))
    signatures = []
    for cell in selected_cells:
        signatures.append(
            (
                int(sites[cell]),
                int(result.cell_regions[cell]),
                tuple(sorted(piece_rows[int(cell)])),
                tuple(sorted(face_rows[int(cell)])),
                float(result.cell_volumes[cell]).hex(),
                tuple(float(value).hex() for value in result.cell_moments[cell]),
                float(result.cell_second_moments[cell]).hex(),
            )
        )
    return tuple(sorted(signatures))


def _periodic_power_prepared_kernel(
    points: np.ndarray,
    point_ids: np.ndarray,
    weights: np.ndarray,
    lattice: np.ndarray,
    owned: np.ndarray,
    repeats: int,
    /,
) -> dict[str, object]:
    """Prepare one immutable global-carrier input and compare native targets."""
    from phydrax import _meshcore as meshcore
    from phydrax.meshing.providers import _native_periodic_power as power

    radius = power._lattice_covering_radius(lattice)
    inverse, inverse_time = _timed_host(lambda: power._exact_lattice_inverse(lattice))
    lower = power._outward_bound(np.min(points, axis=0) - radius, -np.inf)
    upper = power._outward_bound(np.max(points, axis=0) + radius, np.inf)
    (image_owners, image_windings), image_time = _timed_host(
        lambda: power._power_images(points, point_ids, inverse, lower, upper, radius, 0)
    )
    (coordinate_offsets, coordinate_components), coordinates_time = _timed_host(
        lambda: power._exact_image_coordinates(
            points, lattice, image_owners, image_windings
        )
    )
    (neighbor_offsets, neighbors), graph_time = _timed_host(
        lambda: power._image_radius_neighbor_csr(
            image_owners,
            image_windings,
            coordinate_offsets,
            coordinate_components,
            radius,
        )
    )
    (carrier_points, carrier_tets, carrier_facets), carrier_time = _timed_host(
        lambda: power._box_tetrahedra(lower, upper)
    )
    selected = np.asarray(
        [
            power._image_row(image_owners, image_windings, int(row))
            for row in np.flatnonzero(owned)
        ],
        dtype=np.int32,
    )
    args = (
        points,
        weights,
        image_owners,
        coordinate_offsets,
        coordinate_components,
        neighbor_offsets,
        neighbors,
        np.zeros(image_owners.size, dtype=np.int8),
        carrier_points,
        carrier_tets,
        np.zeros(carrier_tets.shape[0], dtype=np.int32),
        carrier_facets,
    )
    full, full_time = _timed_host(lambda: meshcore.restricted_power_cells_exact(*args))
    selected_result, selected_time = _timed_host(
        lambda: meshcore.restricted_power_cells_exact(*args, selected_sites=selected)
    )
    component_lookup, component_time = _timed_host(
        lambda: _periodic_power_component_lookup(full)
    )
    selected_signature, selected_signature_time = _timed_host(
        lambda: _periodic_power_selected_signature(
            selected_result, selected, component_lookup
        )
    )

    def compare() -> bool:
        return (
            _periodic_power_selected_signature(full, selected, component_lookup)
            == selected_signature
        )

    matched, comparison_time = _timed_host(compare)
    if not matched:
        raise ValueError("Selected native cells differ from the full exact construction.")
    warm_kernel_times: dict[str, list[dict[str, float]]] = {"full": [], "selected": []}
    for name, selection in (("full", None), ("selected", selected)):
        for _ in range(repeats):
            result, elapsed = _timed_host(
                lambda selection=selection: meshcore.restricted_power_cells_exact(
                    *args, selected_sites=selection
                )
            )
            if (
                _periodic_power_selected_signature(result, selected, component_lookup)
                != selected_signature
            ):
                raise ValueError(
                    "Warm native power cells changed their selected geometry."
                )
            warm_kernel_times[name].append(elapsed)

    def output_size(result: Any) -> dict[str, int]:
        remote = np.asarray(result.face_remote_sites, dtype=np.int64)
        return {
            "cells": int(result.cell_sites.size),
            "faces": int(result.face_facets.size),
            "pieces": int(result.piece_sites.size),
            "represented_remote_image_sites": int(np.unique(remote[remote >= 0]).size),
            "logical_array_bytes": sum(
                int(value.nbytes) for value in result if isinstance(value, np.ndarray)
            ),
        }

    return {
        "selected_images": int(selected.size),
        "derived_image_rows": int(image_owners.size),
        "derived_image_logical_bytes": sum(
            int(value.nbytes)
            for value in (
                image_owners,
                image_windings,
                coordinate_offsets,
                coordinate_components,
            )
        ),
        "derived_neighbor_logical_bytes": int(neighbor_offsets.nbytes + neighbors.nbytes),
        "carrier_input_logical_bytes": int(
            carrier_points.nbytes
            + carrier_tets.nbytes
            + carrier_facets.nbytes
            + args[10].nbytes
        ),
        "prepare": {
            "lattice_inverse": inverse_time,
            "images": image_time,
            "exact_image_coordinates": coordinates_time,
            "neighbor_csr": graph_time,
            "carrier": carrier_time,
        },
        "full_first": full_time,
        "selected_first": selected_time,
        "warm": warm_kernel_times,
        "component_resolution": component_time,
        "selected_signature": selected_signature_time,
        "output_comparison": comparison_time,
        "selected_matches_full": matched,
        "full_output": output_size(full),
        "selected_output": output_size(selected_result),
    }


def benchmark_periodic_power_selected(
    resolution: int, repeats: int, /
) -> dict[str, object]:
    """Measure the real owner-local candidate and a matched native kernel ablation.

    Resolution is the authored site count. The current JAX process group is the
    owner group; launch one process per owner through the normal runtime launcher.
    Global authored sites and carrier are explicit benchmark inputs. Expanded
    images and adjacency are derived construction inputs, not authored data.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    from phydrax import _meshcore as meshcore
    from phydrax._execution_runtime import ExecutionGroup, ExecutionGroupSpec
    from phydrax._fingerprint import canonical_fingerprint
    from phydrax.discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from phydrax.discretization._periodic_cell import PeriodicCell
    from phydrax.meshing import PeriodicPointOrbits
    from phydrax.meshing.providers import (
        _native_periodic_power as power,
        NativePeriodicSource,
    )

    if resolution < 1 or repeats < 1:
        raise ValueError("resolution and repeats must be positive.")
    members = tuple(range(jax.process_count()))
    rank = members.index(jax.process_index())
    devices = tuple(
        min(
            (device for device in jax.devices() if device.process_index == owner),
            key=lambda device: device.id,
        )
        for owner in members
    )
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "periodic-power-benchmark",
            members,
            tuple((device.process_index, device.id) for device in devices),
            mesh_axes=(("periodic_power_benchmark", len(members)),),
        ),
        devices,
    )

    def authored() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = np.arange(resolution, dtype=np.int64)
        points = np.column_stack(
            (
                (rows + 0.5) / resolution,
                np.full(resolution, 0.125),
                np.full(resolution, 0.125),
            )
        )
        return (
            points,
            rows + 100,
            np.zeros(resolution, dtype=np.float64),
            np.diag((1.0, 0.25, 0.25)),
            (rows * len(members) // resolution).astype(np.int32),
        )

    (points, point_ids, weights, lattice, owners), authored_time = _timed_host(authored)
    cell = PeriodicCell(lattice)
    owned = owners == rank
    source, source_time = _timed_host(
        lambda: NativePeriodicSource(
            PeriodicPointOrbits(cell, points[owned], point_ids[owned]),
            "periodic-power-benchmark",
            "periodic-power-benchmark-r1",
            site_weights=weights[owned],
            point_owner_ranks=owners[owned],
        )
    )
    manifest = canonical_fingerprint(
        ("periodic-power-benchmark", resolution, points, point_ids, weights, lattice)
    )

    candidate_phases: dict[str, list[dict[str, float]]] = {}
    native_candidate_outputs: list[dict[str, int]] = []

    def instrument(original: Any, name: str) -> Any:
        def measured(*args: Any, **kwargs: Any) -> Any:
            started, cpu_started = perf_counter(), process_time()
            try:
                result = original(*args, **kwargs)
            finally:
                candidate_phases.setdefault(name, []).append(
                    {
                        "wall_seconds": perf_counter() - started,
                        "cpu_seconds": process_time() - cpu_started,
                    }
                )
            if name == "native_restriction":
                native_candidate_outputs.append(
                    {
                        "cells": int(result.cell_sites.size),
                        "pieces": int(result.piece_sites.size),
                        "faces": int(result.face_facets.size),
                        "logical_array_bytes": sum(
                            int(value.nbytes)
                            for value in result
                            if isinstance(value, np.ndarray)
                        ),
                    }
                )
            return result

        return measured

    with ExitStack() as stack:
        for name in (
            "_power_images",
            "_exact_image_coordinates",
            "_image_radius_neighbor_csr",
            "_owner_local_power_support_exchange",
            "_owner_local_power_candidate_support",
            "_polyhedral_quotient_from_power",
            "_native_power_face_evidence",
        ):
            stack.enter_context(
                patch.object(power, name, instrument(getattr(power, name), name))
            )
        stack.enter_context(
            patch.object(
                meshcore,
                "restricted_power_cells_exact",
                instrument(meshcore.restricted_power_cells_exact, "native_restriction"),
            )
        )
        stack.enter_context(
            patch.object(
                ExactPowerCellGeometrySource,
                "__init__",
                instrument(
                    ExactPowerCellGeometrySource.__init__, "source_map_construction"
                ),
            )
        )
        candidate, cold_candidate_time = _timed_host(
            lambda: power._owner_local_periodic_power_candidate(
                source, rank, group, manifest, axis_name="periodic_power_benchmark"
            )
        )
        cold_identity = (
            None if candidate["mesh"] is None else candidate["mesh"].topology_id,
            None
            if candidate["source_map"] is None
            else candidate["source_map"].source_map_id,
        )
        warm_candidate_times = []
        for _ in range(repeats):
            candidate, elapsed = _timed_host(
                lambda: power._owner_local_periodic_power_candidate(
                    source, rank, group, manifest, axis_name="periodic_power_benchmark"
                )
            )
            identity = (
                None if candidate["mesh"] is None else candidate["mesh"].topology_id,
                None
                if candidate["source_map"] is None
                else candidate["source_map"].source_map_id,
            )
            if identity != cold_identity:
                raise ValueError(
                    "Warm periodic power construction changed its exact identity."
                )
            warm_candidate_times.append(elapsed)

    kernel_ablation = _periodic_power_prepared_kernel(
        points, point_ids, weights, lattice, owned, repeats
    )

    peak = process_peak_memory()
    if peak.value_bytes is None:
        raise RuntimeError(
            f"Periodic power benchmark has no process peak memory: {peak.unavailable_reason}"
        )
    process_peak_bytes = peak.value_bytes
    source_map = candidate["source_map"]
    local_mesh = candidate["mesh"]
    support_rows = int(np.asarray(candidate["candidate"]["point_ids"]).size)
    return {
        "status": "passed",
        "scenario": "periodic-power-selected",
        "resolution_authored_site_count": resolution,
        "repeats": repeats,
        "owner_rank": rank,
        "owner_count": len(members),
        "source_manifest": manifest,
        "timing_scope": {
            "candidate": "owner-local candidate construction, not public provider publication",
            "kernel": "native full versus selected restriction on the same prepared inputs",
            "candidate_phases": "inclusive instrumented intervals; nested phases must not be summed",
            "cold": "first call in this process after case cache clearing, not a fresh process",
            "warm": "subsequent calls in this process using the same authored or prepared inputs",
        },
        "authored": {
            "global_sites": resolution,
            "owned_sites": int(np.count_nonzero(owned)),
            "global_logical_array_bytes": sum(
                int(value.nbytes) for value in (points, point_ids, weights, lattice)
            ),
            "owner_placement_logical_array_bytes": int(owners.nbytes),
        },
        "candidate": {
            "cold": cold_candidate_time,
            "warm": warm_candidate_times,
            "phase_samples": candidate_phases,
            "native_output_samples": native_candidate_outputs,
            "support_site_rows": support_rows,
            "ghost_support_site_rows": support_rows - int(np.count_nonzero(owned)),
            "retained_support_source_logical_array_bytes": _array_bytes(
                candidate["source"]
            ),
            "resident_cells": 0
            if local_mesh is None
            else int(local_mesh.entity_set(3).count),
            "source_map_id": None if source_map is None else source_map.source_map_id,
            "retained_source_map_logical_array_bytes": 0
            if source_map is None
            else _array_bytes(source_map),
            "retained_mesh_logical_array_bytes": 0
            if local_mesh is None
            else _array_bytes(local_mesh),
            "exact_source_site_rows": 0
            if source_map is None
            else len(source_map.exact_site_points),
            "exact_vertex_rows": 0
            if source_map is None
            else len(source_map.exact_vertices),
        },
        "kernel_ablation": kernel_ablation,
        "preparation": {"authored": authored_time, "owner_source": source_time},
        "process_peak_rss_bytes": process_peak_bytes,
        "memory_scope": "process lifetime peak RSS; array bytes are logical and exclude Python objects",
    }


def _run_case(name: str, resolution: int, repeats: int, /) -> dict[str, object]:
    jax.clear_caches()
    if name in {
        "gmsh-semantic",
        "gmsh-planar-semantic",
        "gmsh-planar-band",
        "gmsh-hybrid-slab",
    }:
        missing = _missing_gmsh_dependencies()
        if missing:
            return {
                "scenario": name,
                "status": "missing_dependency",
                "resolution": resolution,
                "repeats": repeats,
                "missing_dependencies": missing,
                "native_fallback": False,
            }
    return _CASES[name](resolution, repeats)


def _repeated_case(
    runner: Any,
    resolution: int,
    repeats: int,
    /,
) -> dict[str, object]:
    samples = []
    for _ in range(repeats):
        jax.clear_caches()
        samples.append(runner(resolution))
    timing_names = tuple(samples[0]["stages_seconds"])
    return {
        "status": "passed",
        "resolution": resolution,
        "repeats": repeats,
        "samples": samples,
        "timing_ranges_seconds": {
            name: _range([sample["stages_seconds"][name] for sample in samples])
            for name in timing_names
        },
    }


_CASES = {
    "periodic-power-selected": benchmark_periodic_power_selected,
    "native-meshing": lambda resolution, repeats: benchmark_native_meshing(
        resolution, repeats
    ),
    "native-cad-meshing": lambda resolution, repeats: benchmark_native_cad_meshing(
        resolution, repeats
    ),
    "native-brep-plc-refinement": benchmark_native_brep_plc_refinement,
    "native-tspline-cad-analysis": lambda resolution, repeats: (
        benchmark_native_tspline_cad_analysis(resolution, repeats)
    ),
    "gmsh-semantic": _benchmark_gmsh_semantic_case,
    "optional-gmsh-comparison": benchmark_native_gmsh_comparison,
    "cad-partition": lambda resolution, repeats: _repeated_case(
        benchmark_cad_partition,
        resolution,
        repeats,
    ),
    "gmsh-planar-semantic": lambda resolution, repeats: _repeated_case(
        benchmark_planar_semantic,
        resolution,
        repeats,
    ),
    "layout-decode": lambda resolution, repeats: _repeated_case(
        benchmark_layout_decode,
        resolution,
        repeats,
    ),
    "gmsh-planar-band": lambda resolution, repeats: _repeated_case(
        benchmark_planar_band,
        resolution,
        repeats,
    ),
    "gmsh-hybrid-slab": lambda resolution, repeats: _repeated_case(
        benchmark_hybrid_slab,
        resolution,
        repeats,
    ),
    "implicit-discovery": lambda resolution, repeats: _repeated_case(
        benchmark_implicit_discovery,
        resolution,
        repeats,
    ),
    "lbvh": lambda resolution, repeats: _repeated_case(
        benchmark_lbvh,
        resolution,
        repeats,
    ),
    "overlap-pairs": lambda resolution, repeats: _repeated_case(
        benchmark_overlap_pairs,
        resolution,
        repeats,
    ),
    "incidence": lambda resolution, repeats: _repeated_case(
        benchmark_incidence,
        resolution,
        repeats,
    ),
    "supermesh": lambda resolution, repeats: _repeated_case(
        benchmark_supermesh,
        resolution,
        repeats,
    ),
    "remap": lambda resolution, repeats: _repeated_case(
        benchmark_remap,
        resolution,
        repeats,
    ),
    "device-bisection": lambda resolution, repeats: _repeated_case(
        benchmark_device_bisection,
        resolution,
        repeats,
    ),
    "local-metric": lambda resolution, repeats: _repeated_case(
        benchmark_local_metric,
        resolution,
        repeats,
    ),
    "boundary-layer": lambda resolution, repeats: _repeated_case(
        benchmark_boundary_layer, resolution, repeats
    ),
    "host-bisection": lambda resolution, repeats: _repeated_case(
        benchmark_host_bisection,
        resolution,
        repeats,
    ),
    "repartition": lambda resolution, repeats: _repeated_case(
        benchmark_repartition,
        resolution,
        repeats,
    ),
    "provider-worker": lambda resolution, repeats: _provider_worker_case(
        resolution, repeats
    ),
    "optimization": lambda resolution, repeats: _repeated_case(
        benchmark_optimization,
        resolution,
        repeats,
    ),
    "high-order-certification": lambda resolution, repeats: _repeated_case(
        benchmark_high_order_certification,
        resolution,
        repeats,
    ),
    "global-embedding": lambda resolution, repeats: _repeated_case(
        benchmark_global_embedding, resolution, repeats
    ),
    "predicates": lambda resolution, repeats: _repeated_case(
        benchmark_predicates,
        resolution,
        repeats,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=[8, 16],
        help="Scale each case; periodic-power-selected uses the authored site count.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--plc-target-size",
        type=float,
        help="Explicit volume refinement size for native-brep-plc-refinement.",
    )
    parser.add_argument(
        "--gmsh-semantic",
        action="store_true",
        help="Also run semantic Gmsh scaling at 2, 8, and 32 regions.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(_CASES),
        help="Run one named scaling case; defaults to native meshing only.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.resolution):
        parser.error("resolutions and repeats must be positive")
    selected = ("native-meshing",) if args.case is None else tuple(args.case)
    if args.plc_target_size is not None:
        if "native-brep-plc-refinement" not in selected:
            parser.error(
                "--plc-target-size is only valid with --case native-brep-plc-refinement"
            )
        if not np.isfinite(args.plc_target_size) or args.plc_target_size <= 0.0:
            parser.error("--plc-target-size must be finite and positive")
    set_compilation_cache(False)
    output: dict[str, object] = {
        "status": "completed",
        "environment": _environment_identity(),
        "cache_policy": {
            "process_local_jax": "cleared before each case and independent cold sample; retained for labeled warm samples",
            "persistent_jax": "disabled; lowering and compilation do not reuse persistent compiler artifacts",
            "prepared_execution": "same immutable native plan or compiled executable reused for warm samples",
        },
    }
    for name in selected:
        if name == "native-brep-plc-refinement":
            samples = []
            for value in args.resolution:
                jax.clear_caches()
                samples.append(
                    benchmark_native_brep_plc_refinement(
                        value, args.repeats, target_size=args.plc_target_size
                    )
                )
            output[name] = samples
        else:
            output[name] = [
                _run_case(name, value, args.repeats) for value in args.resolution
            ]
    if args.gmsh_semantic:
        missing = _missing_gmsh_dependencies()
        if missing:
            output["gmsh_semantic"] = [
                {
                    "scenario": "gmsh-semantic",
                    "status": "missing_dependency",
                    "resolution": value,
                    "repeats": args.repeats,
                    "missing_dependencies": missing,
                    "native_fallback": False,
                }
                for value in (2, 8, 32)
            ]
        else:
            output["gmsh_semantic"] = []
            for value in (2, 8, 32):
                jax.clear_caches()
                output["gmsh_semantic"].append(
                    benchmark_gmsh_semantic(value, args.repeats)
                )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

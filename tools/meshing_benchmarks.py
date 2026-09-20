#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import jax
import numpy as np

import phydrax as phx


def _contract() -> phx.SpatialCoordinateContract:
    return phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _timed(call):
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


def _range(values) -> dict[str, float]:
    samples = np.asarray(tuple(values), dtype="float64")
    if samples.ndim != 1 or not samples.size:
        raise ValueError("Benchmark ranges require one nonempty scalar sequence.")
    return {"minimum": float(np.min(samples)), "maximum": float(np.max(samples))}


def _integer_range(values) -> dict[str, int]:
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


def _planar_partition(count: int, destination: Path):
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
    return embedding, plan, phx.geometry.partition_planar(plan, destination=destination)


def _persist_shape(shape, path: Path, contract: phx.SpatialCoordinateContract):
    return phx.geometry.persist_occt_shape(shape, path, coordinate_contract=contract)


def _cad_partition(count: int, destination: Path):
    import build123d as bd

    contract = _contract()
    region_names = tuple(f"region-{index}" for index in range(count))
    operands = []
    for index, name in enumerate(region_names):
        shape = bd.Box(1.0, 1.0, 1.0).translate((float(index), 0.0, 0.0))
        operands.append(
            phx.geometry.BRepPartitionOperand(
                name,
                _persist_shape(shape.wrapped, destination / f"{name}.brep", contract),
                phx.geometry.BRepPartitionRole.REGION,
            )
        )
    plan = phx.geometry.BRepPartitionPlan(
        contract,
        tuple(operands),
        phx.geometry.BRepPartitionPolicy(region_precedence=tuple(reversed(region_names))),
    )
    return plan, phx.geometry.partition_brep(
        plan, destination=destination / "result.brep"
    )


def _layout_bytes(count: int) -> bytes:
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
    transition, refinement_seconds = _timed(
        lambda: phx.meshing.refine_triangle_mesh(
            mesh,
            np.asarray(mesh.blocks[0].global_ids)[:1],
            phx.SpatialCoordinateContract.si(),
        )[0]
    )
    stencil = transition.vertex_stencil
    assert stencil is not None
    transferred = stencil.apply(mesh.vertex_global_ids, mesh.coordinates[:, 0])
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
    with TemporaryDirectory(prefix="phydrax-cad-partition-benchmark-") as temporary:
        root = Path(temporary)
        (plan, result), total_seconds = _timed(lambda: _cad_partition(count, root))
    if result.topological_dimension != 3 or not result.named_solid_entity_ids:
        raise RuntimeError("CAD partition benchmark did not create solid regions.")
    return {
        "resolution": resolution,
        "stages_seconds": {"persist_and_partition": total_seconds},
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
    with TemporaryDirectory(prefix="phydrax-planar-semantic-benchmark-") as temporary:
        root = Path(temporary) / "planar.brep"
        (embedding, plan, partition), partition_seconds = _timed(
            lambda: _planar_partition(count, root)
        )
        provider = phx.meshing.GmshProvider()
        scope = provider.whole_scope(partition, 2)
        regions = tuple(
            phx.meshing.RegionControl(
                provider.entity_scope(partition, region.entity_ids),
                region.name,
                f"material:{region.name}",
                phx.meshing.RegionRole.SOLID,
            )
            for region in partition.regions
        )
        patches = tuple(
            phx.meshing.PatchControl(
                patch.name,
                provider.entity_scope(partition, patch.entity_ids),
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
    limits = phx.interchange.ResourceLimits(
        max_bytes=1_000_000,
        max_depth=4,
        max_nodes=10_000,
        max_attributes=10_000,
        max_losses=0,
    )
    policy = phx.interchange.LayoutImportPolicy(
        phx.interchange.LayoutFormat.GDSII,
        _contract(),
        limits,
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
    with TemporaryDirectory(prefix="phydrax-planar-band-benchmark-") as temporary:
        root = Path(temporary)
        embedding, _plan, partition = _planar_partition(2, root / "planar.brep")
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
            lambda: phx.meshing.prepare_planar_bands(
                plan, destination=root / "bands.brep"
            )
        )
        provider = phx.meshing.GmshProvider()
        scope = provider.whole_scope(band_result, 2)
        regions = tuple(
            phx.meshing.RegionControl(
                provider.entity_scope(band_result, region.entity_ids),
                region.name,
                f"material:{region.name}",
                phx.meshing.RegionRole.SOLID,
            )
            for region in band_result.partition.regions
        )
        patches = tuple(
            phx.meshing.PatchControl(
                value.name,
                provider.entity_scope(band_result, value.entity_ids),
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
        "resolution": resolution,
        "stages_seconds": {
            "preparation": preparation_seconds,
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
    from tools.meshing_qualification import _hybrid_gmsh_result

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


def _semantic_gmsh_specification(provider, partition):
    source = partition.model
    whole = provider.whole_scope(source, 3)
    regions = tuple(
        phx.meshing.RegionControl(
            provider.entity_scope(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in partition.regions
    )
    patches = tuple(
        phx.meshing.PatchControl(
            patch.name,
            provider.entity_scope(source, patch.entity_ids),
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


def _semantic_gmsh_plan(provider, partition):
    specification = _semantic_gmsh_specification(provider, partition)
    return specification, provider.plan(partition.model, specification)


def benchmark_gmsh_semantic(
    region_count: int,
    repeats: int,
    directory: Path,
    /,
) -> dict[str, object]:
    provider = phx.meshing.GmshProvider()
    timings = {"partition": [], "plan": [], "execute": []}
    samples = []
    for repeat in range(repeats):
        root = directory / f"regions-{region_count}-repeat-{repeat}"
        root.mkdir(parents=True, exist_ok=False)
        (_, partition), partition_seconds = _timed(
            lambda root=root: _cad_partition(region_count, root)
        )
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
    with TemporaryDirectory(prefix="phydrax-gmsh-semantic-benchmark-") as temporary:
        return benchmark_gmsh_semantic(
            max(2, resolution),
            repeats,
            Path(temporary),
        )


def _repeated_case(
    runner,
    resolution: int,
    repeats: int,
    /,
) -> dict[str, object]:
    samples = [runner(resolution) for _ in range(repeats)]
    timing_names = tuple(samples[0]["stages_seconds"])
    return {
        "resolution": resolution,
        "repeats": repeats,
        "samples": samples,
        "timing_ranges_seconds": {
            name: _range([sample["stages_seconds"][name] for sample in samples])
            for name in timing_names
        },
    }


_CASES = {
    "native-meshing": lambda resolution, repeats: benchmark_native_meshing(
        resolution, repeats
    ),
    "gmsh-semantic": _benchmark_gmsh_semantic_case,
    "cad-partition": lambda resolution, repeats: _repeated_case(
        benchmark_cad_partition,
        resolution,
        repeats,
    ),
    "planar-semantic": lambda resolution, repeats: _repeated_case(
        benchmark_planar_semantic,
        resolution,
        repeats,
    ),
    "layout-decode": lambda resolution, repeats: _repeated_case(
        benchmark_layout_decode,
        resolution,
        repeats,
    ),
    "planar-band": lambda resolution, repeats: _repeated_case(
        benchmark_planar_band,
        resolution,
        repeats,
    ),
    "hybrid-slab": lambda resolution, repeats: _repeated_case(
        benchmark_hybrid_slab,
        resolution,
        repeats,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--repeats", type=int, default=5)
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
    output = {
        name: [_CASES[name](value, args.repeats) for value in args.resolution]
        for name in selected
    }
    if args.gmsh_semantic:
        with TemporaryDirectory(prefix="phydrax-gmsh-semantic-benchmark-") as temporary:
            directory = Path(temporary)
            output["gmsh_semantic"] = [
                benchmark_gmsh_semantic(value, args.repeats, directory)
                for value in (2, 8, 32)
            ]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

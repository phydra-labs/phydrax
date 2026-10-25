#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from time import perf_counter


_IMPORT_STARTED = perf_counter()
import argparse
import json
import os
import shutil
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, final

import equinox as eqx
import jax
import numpy as np

import phydrax as phx
from phydrax import canonical_fingerprint
from phydrax.meshing import (
    BisectionCompatibility,
    meshcore_available,
    metis_identity,
    metis_partition,
)
from phydrax.meshing.qualification import (
    benchmark_adapted_quality,
    benchmark_bisection_compilation_counts,
    benchmark_boundary_layer_quality,
    benchmark_device_metric_compilation_count,
    benchmark_projection_compilation_counts,
    BenchmarkProviderKind,
    execute_benchmark_device_metric,
    execute_benchmark_optimization,
    measure_bisection_compilation,
    measure_projection_compilation,
    prepare_benchmark_provider_input,
    prepare_squeezed_benchmark_geometry,
    qualify_bvh_scaling,
    qualify_overlap_pair_scaling,
)
from phydrax.units import METER
from tools._meshing_cases import (
    CASE_NAMES,
    original_wall_layer_schedule,
    POLYHEDRAL_CASE_NAMES,
)
from tools.compile_counter import CompileCounter
from tools.meshing_qualification._campaign import (
    _benchmark_registered_native_corpus,
    _native_driver_controls,
    benchmark_native_lifecycle,
    load_native_campaign_registration,
)
from tools.meshing_qualification._controls import (
    _NATIVE_DRIVER_ARGUMENTS,
    _periodic_qualification_profile_contracts,
    add_case_arguments,
    BenchmarkControls,
    FAMILY_CASES,
    HYBRID_PROFILES,
    SURFACE_CASES,
)
from tools.meshing_qualification._fixtures import (
    _HEXAHEDRON_CORNERS,
    _jittered_unit_square_triangles,
    _kuhn_tetrahedra,
    _planar_triangles,
    _remap_field,
)
from tools.meshing_qualification._records import _corpus_profiles
from tools.meshing_qualification._runtime import (
    _contract,
    peak_resident_bytes,
    qualification_failure,
)
from tools.meshing_qualification.brep_performance import (
    benchmark_brep_containment,
    benchmark_brep_containment_campaign,
    benchmark_brep_interval_program,
    benchmark_brep_moments,
    benchmark_brep_orientation_audit,
    benchmark_brep_preconditioners,
    benchmark_brep_quadratic_bounds,
    benchmark_cell_validity,
    benchmark_homotopy_arithmetic,
)
from tools.meshing_qualification.kernel_performance import (
    benchmark_fresh_imports,
    benchmark_parallel_batches,
)
from tools.meshing_qualification.scenarios.boundary_layer import (
    _advancing_layer_control,
    _layer_box_wall,
)
from tools.meshing_qualification.scenarios.layout import _hybrid_gmsh_result


_IMPORT_SECONDS = perf_counter() - _IMPORT_STARTED


def _measured_benchmark_case(
    name: str, call: Callable[[], dict[str, object]]
) -> dict[str, object]:
    """Attach compile-diagnostic timing metadata without replacing case evidence.

    Compile counts and seconds come from completed JAX compiler events; import
    seconds measure this module's first import. Peak RSS is process-lifetime
    high-water RSS in bytes, not an incremental per-case allocation. Legacy
    microcases do not enable an accounting profiler: their accounting seconds
    are None with an explicit reason code. Lifecycle and diagnostic-capable
    cases publish measured accounting only in a separate process launched with
    PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS=1; those timings are not headline
    samples.
    """
    stage = f"benchmark_case.{name}"
    started = perf_counter()
    with CompileCounter(stage=stage) as compiler:
        result: dict[str, Any] = call()
    measured = {
        "wall_seconds": perf_counter() - started,
        "timing_policy": "compile-instrumented-diagnostic-not-headline",
        "eager_xla_compile_count": compiler.count,
        "eager_xla_compile_seconds": compiler.seconds,
        "compile_events": [event.to_record() for event in compiler.events],
        "python_resource_accounting_seconds": None,
        "python_resource_accounting_measurement": "unmeasured-clean-timing-no-accounting-profiler",
        "peak_resident_bytes": peak_resident_bytes(),
        "peak_resident_unit": "bytes",
    }
    return {
        **result,
        "import_seconds": _IMPORT_SECONDS,
        "peak_resident_unit": "bytes",
        "stage_measurements": {
            **result.get("stage_measurements", {}),
            stage: measured,
        },
    }


def _clean_benchmark_case(
    name: str, call: Callable[[], dict[str, object]]
) -> dict[str, object]:
    """Measure clean timing, or collect compiler diagnostics in a separate run."""
    if os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") == "1":
        return _measured_benchmark_case(name, call)
    started = perf_counter()
    result: dict[str, Any] = call()
    measured = {
        "wall_seconds": perf_counter() - started,
        "timing_policy": "clean-no-compile-or-accounting-instrumentation",
        "eager_xla_compile_count": None,
        "eager_xla_compile_seconds": None,
        "compile_events": None,
        "compile_measurement": "unmeasured-run-separate-diagnostic-invocation",
        "python_resource_accounting_seconds": None,
        "python_resource_accounting_measurement": "unmeasured-clean-timing-no-accounting-profiler",
        "peak_resident_bytes": peak_resident_bytes(),
        "peak_resident_unit": "bytes",
    }
    return {
        **result,
        "import_seconds": _IMPORT_SECONDS,
        "peak_resident_unit": "bytes",
        "stage_measurements": {
            **result.get("stage_measurements", {}),
            f"benchmark_case.{name}": measured,
        },
    }


def _timed[T](call: Callable[[], T]) -> tuple[T, float]:
    started = perf_counter()
    result = call()
    return result, perf_counter() - started


def _range(values: Any) -> dict[str, float]:
    samples = np.asarray(tuple(values), dtype="float64")
    if samples.ndim != 1 or not samples.size:
        raise ValueError(
            f"_range.samples must be a nonempty one-dimensional scalar sequence; got {samples!r}."
        )
    return {"minimum": float(np.min(samples)), "maximum": float(np.max(samples))}


def _integer_range(values: Any) -> dict[str, int]:
    samples = np.asarray(tuple(values), dtype=np.int64)
    if samples.ndim != 1 or not samples.size:
        raise ValueError(
            f"_integer_range.samples must be a nonempty one-dimensional integer sequence; got {samples!r}."
        )
    return {"minimum": int(np.min(samples)), "maximum": int(np.max(samples))}


def _planar_region(name: str, x0: float, x1: float) -> phx.geometry.PlanarMeshRegion:
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=name,
    )


def _planar_partition(count: int, destination: Path) -> Any:
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


def _persist_shape(
    shape: Any, path: Path, contract: phx.SpatialCoordinateContract
) -> Any:
    return phx.interchange.persist_occt_shape(shape, path, coordinate_contract=contract)


def _cad_partition(count: int, destination: Path) -> Any:

    # Optional provider imports are admitted only for this fixture workflow.
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    contract = _contract()
    region_names = tuple(f"region-{index}" for index in range(count))
    operands = []
    for index, name in enumerate(region_names):
        shape = BRepPrimAPI_MakeBox(gp_Pnt(float(index), 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
        operands.append(
            phx.geometry.BRepPartitionOperand(
                name,
                _persist_shape(shape, destination / f"{name}.brep", contract),
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
    # The optional dependency is admitted when this workflow executes.
    import gdstk  # ty: ignore[unresolved-import]

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
        raise qualification_failure(
            "benchmark_native_meshing",
            ("Native meshing benchmark failed certification or linear transfer."),
            {"error": error, "result": result},
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
        raise qualification_failure(
            "benchmark_cad_partition",
            ("CAD partition benchmark did not create solid regions."),
            {"result": result},
        )
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
        raise qualification_failure(
            "benchmark_planar_semantic",
            ("Planar benchmark failed semantic mesh qualification."),
            {"mesh_result": mesh_result, "partition": partition},
        )
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
        raise qualification_failure(
            "benchmark_layout_decode",
            ("Layout benchmark lost decoder validity or source identity."),
            {"result": result},
        )
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
        raise qualification_failure(
            "benchmark_planar_band",
            ("Planar-band benchmark failed mesh qualification."),
            {"band_result": band_result, "mesh_result": mesh_result},
        )
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
        raise qualification_failure(
            "benchmark_hybrid_slab",
            ("Hybrid slab benchmark failed mesh audit or compliance."),
            {"result": result},
        )
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
    ).compile(coordinate_unit=METER)
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
        raise qualification_failure(
            "benchmark_implicit_discovery",
            ("Implicit discovery benchmark realization was rejected."),
            {"cold": cold, "warm": warm},
        )
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


def _array_bytes(tree: Any) -> int:
    """Bytes of the distinct array leaves a result retains."""
    leaves = {id(leaf): leaf for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))}
    return sum(leaf.nbytes for leaf in leaves.values())


def benchmark_incidence(resolution: int) -> dict[str, object]:
    """Scale incidence construction with about `resolution` cells."""
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
        raise qualification_failure(
            "benchmark_incidence",
            ("Coordinate refresh changed the mesh topology."),
            {"mesh": mesh, "moved": moved},
        )
    surface, surface_seconds = _timed(
        # ty: ignore[invalid-argument-type]
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
            raise qualification_failure(
                "benchmark_supermesh",
                (f"{name} common refinement failed: {refinement.status}."),
                {"refinement": refinement},
            )
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
        raise qualification_failure(
            "_overlap_refinement",
            (f"L2 projection refinement failed: {refinement.status}."),
            {"refinement": refinement},
        )
    return refinement


def _l2_projection_operator(
    transfer: phx.discretization.FiniteElementTopologyTransfer, /
) -> phx.discretization.FiniteElementL2Projection:
    """Return the Galerkin operator required for mixed-mass benchmark evidence."""
    match transfer.primal:
        case phx.discretization.FiniteElementL2Projection() as projection:
            return projection
        case primal:
            raise qualification_failure(
                "_benchmark_l2_projection",
                (
                    "L2 projection transfer.primal must expose Galerkin mixed mass; "
                    f"got {type(primal).__name__}."
                ),
                {"transfer": transfer},
            )


def _l2_projection_residuals(
    source: Any,
    target: Any,
    transfer: phx.discretization.FiniteElementTopologyTransfer,
    /,
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
    mixed = np.asarray(_l2_projection_operator(transfer).mixed_mass.mv(values))
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
    projection = _l2_projection_operator(transfer)
    mixed_mass = projection.mixed_mass
    compiled = measure_projection_compilation(
        prepared, projection, block, transfer.target_size, compiler=_compiled_cost
    )
    _, action_first = _timed(lambda: jax.block_until_ready(transfer.apply(block)))
    _, action_warm = _timed(lambda: jax.block_until_ready(transfer.apply(block)))
    # Snapshot only after every source-target action has compiled and run.
    before_refresh = benchmark_projection_compilation_counts()
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
        for name, count in benchmark_projection_compilation_counts().items()
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
        raise qualification_failure(
            "benchmark_remap.remap.status",
            "be SUCCESS",
            {"status": remap.status, "reason": remap.reason},
        )
    remap_plan = remap.plan
    if remap_plan is None:
        raise qualification_failure(
            "benchmark_remap.remap.plan",
            "be present after successful preparation",
            {"remap": remap},
        )
    match remap.route:
        case phx.geometry.PreparedCommonRefinement() as refinement:
            pass
        case route:
            raise qualification_failure(
                "benchmark_remap.remap.route",
                "be a common refinement for second-order reconstruction",
                {"route": type(route).__name__},
            )
    _, first_cold = _timed(
        lambda: jax.block_until_ready(_first_order_remap(remap_plan, values))
    )
    first, first_warm = _timed(
        lambda: jax.block_until_ready(_first_order_remap(remap_plan, values))
    )

    def second_order_plan() -> phx.discretization.UnstructuredSecondOrderRemapPlan:
        return phx.discretization.UnstructuredSecondOrderRemapPlan(
            remap_plan, refinement, source
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
            "entries": refinement.entry_count,
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


def _semantic_gmsh_plan(provider: Any, partition: Any) -> Any:
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
            raise qualification_failure(
                "benchmark_gmsh_semantic",
                ("Semantic Gmsh benchmark failed audit or compliance."),
                {"result": result},
            )
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


def _compiled_cost(compiled_entry: Any, *arguments: Any) -> dict[str, object]:
    """Lowering and compile time plus compiler memory of one compiled entry."""
    lowered, lowering = _timed(lambda: compiled_entry.lower(*arguments))
    compiled, compile_seconds = _timed(lowered.compile)
    # Equinox wraps the JAX stages; the compiler memory lives on the JAX object.
    memory = compiled.compiled.memory_analysis()
    return {
        "lowering_seconds": lowering,
        "compile_seconds": compile_seconds,
        "temporary_bytes": memory.temp_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "argument_bytes": memory.argument_size_in_bytes,
        "generated_code_bytes": memory.generated_code_size_in_bytes,
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
    costs = measure_bisection_compilation(
        layout, state, plan, empty, compiler=_compiled_cost
    )
    cycles = []
    for cycle in range(5):
        center = jax.numpy.asarray((0.2 + 0.15 * cycle, 0.5))
        centroids = jax.numpy.mean(state.mesh.coordinates[state.mesh.cells], axis=1)
        inside = jax.numpy.linalg.norm(centroids - center, axis=1) < 0.12
        refine_marks = inside & state.mesh.cell_active
        coarsen_marks = ~inside & state.mesh.cell_active
        refined, refine_seconds = _timed(
            lambda state=state, refine_marks=refine_marks: jax.block_until_ready(
                phx.discretization.refine_adaptive_simplex(layout, state, refine_marks)
            )
        )
        coarsened, coarsen_seconds = _timed(
            lambda refined=refined, coarsen_marks=coarsen_marks: jax.block_until_ready(
                phx.discretization.coarsen_adaptive_simplex(
                    layout, refined.state, coarsen_marks
                )
            )
        )
        state = coarsened.state
        _, assembly_seconds = _timed(
            lambda state=state: jax.block_until_ready(
                phx.discretization.assemble_masked_finite_element(plan, state.mesh)
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
    compilations = benchmark_bisection_compilation_counts()
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
    execution = execute_benchmark_device_metric(
        layout, state, measure=_timed, compiler=_compiled_cost
    )
    cost, cold, warm = execution.cost, execution.cold, execution.warm
    cold_seconds, warm_seconds = execution.cold_seconds, execution.warm_seconds
    report = jax.device_get(warm.report)
    if bool(report.failed):
        raise qualification_failure(
            "benchmark_local_metric",
            (f"Device metric passes failed with status {int(report.status)}."),
            {"report": report},
        )
    device, commit_seconds = _timed(
        lambda: phx.meshing.commit_device_metric_adaptation(prepared, warm.state)
    )
    if not native.target.audit.passed or not device.target.audit.passed:
        raise qualification_failure(
            "benchmark_local_metric",
            ("A local metric adaptation target failed certification."),
            {"device": device, "native": native},
        )
    match native.evidence:
        case phx.meshing.LocalMetricEvidence() as native_evidence:
            pass
        case evidence:
            raise qualification_failure(
                "benchmark_local_metric",
                (
                    "Native metric adaptation.evidence must be LocalMetricEvidence; "
                    f"got {type(evidence).__name__}."
                ),
                {"native": native},
            )
    return {
        "resolution": resolution,
        "source": {"vertices": vertex_count, "cells": cell_count},
        "capacity": capacity,
        "native": {
            "status": native.status.value,
            "vertices": native.target.mesh.coordinates.shape[0],
            "cells": native.target.mesh.entity_set(2).count,
            "passes": native_evidence.passes,
            "splits": native_evidence.splits,
            "collapses": native_evidence.collapses,
            "flips": native_evidence.flips,
            "relocations": native_evidence.relocations,
            "unit_fraction": native_evidence.unit_fraction,
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
        "compilations_after_calls": benchmark_device_metric_compilation_count(),
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


def benchmark_boundary_layer(resolution: int) -> dict[str, object]:
    """Grow native advancing layers from a box wall of about `resolution` triangles.

    The closed box wall has convex ridges and corners, so growth exercises fan
    columns and corner patches. Wall construction and `prepare_boundary_layers`
    (visibility QP, proximity probes, certification) are host-only NumPy
    stages timed as host preparation and host execution. The compiled consumer is
    the sampled quality evaluation of every layer cell through one module-level
    entry, which reports lowering, compile, cold, and warm execution.
    """

    count = max(1, round((resolution / 12) ** 0.5))
    schedule = original_wall_layer_schedule()
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
    compiled = _compiled_cost(benchmark_boundary_layer_quality, coordinates, connectivity)
    _, cold_seconds = _timed(
        lambda: jax.block_until_ready(
            benchmark_boundary_layer_quality(coordinates, connectivity)
        )
    )
    quality, warm_seconds = _timed(
        lambda: jax.block_until_ready(
            benchmark_boundary_layer_quality(coordinates, connectivity)
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
        raise qualification_failure(
            "benchmark_boundary_layer",
            ("Boundary-layer benchmark lost certification or schedule."),
            {
                "cells": cells,
                "layers": layers,
                "quality": quality,
                "scaled_jacobian": scaled_jacobian,
                "thickness_error": thickness_error,
            },
        )
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
            raise qualification_failure(
                "benchmark_host_bisection",
                (f"Host bisection of the {name} failed."),
                {"result": result},
            )
        transfer = result.transfer
        if transfer is None:
            raise qualification_failure(
                "benchmark_host_bisection",
                (f"Host bisection of the {name} must provide a P1 transfer; got None."),
                {"result": result},
            )
        match result.evidence:
            case phx.meshing.BisectionEvidence() as bisection_evidence:
                pass
            case evidence:
                raise qualification_failure(
                    "benchmark_host_bisection",
                    (
                        "Host bisection adaptation.evidence must be BisectionEvidence; "
                        f"got {type(evidence).__name__}."
                    ),
                    {"result": result},
                )
        match transfer.primal:
            case phx.sparse.SparseLinearMap() as primal:
                transfer_entries = primal.coefficients.size
            case primal:
                raise qualification_failure(
                    "benchmark_host_bisection",
                    (
                        "Host bisection transfer.primal must expose sparse coefficients; "
                        f"got {type(primal).__name__}."
                    ),
                    {"result": result},
                )
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
            raise qualification_failure(
                "benchmark_host_bisection",
                (f"The {name} bisection transfer lost P1 reproduction."),
                {"expected": expected, "moved": moved},
            )
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
            "bisections": bisection_evidence.bisections,
            "closure_iterations": bisection_evidence.closure_iterations,
            "transfer_entries": transfer_entries,
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

    Host-only NumPy stages: certification, MORTON/HILBERT/native GRAPH
    ownership (METIS comparison too when it loads) with ghost construction,
    native bisection of the cells in one corner, and the HILBERT distribution
    transition onto the refined revision. The compiled consumer is the
    transition's ``transfer`` (send gather plus receive reduction) through
    ``_migrate_cell_data``: lowering,
    compile, cold and warm execution, and compiler memory.
    """

    side = max(2, round((resolution / 2) ** 0.5))
    source, certification_seconds = _timed(lambda: _lattice_square(side))
    part = phx.meshing.MeshPart("repartition", source)
    kind = phx.meshing.MeshPartitionKind
    parts = 4
    routes = [kind.MORTON, kind.HILBERT, kind.GRAPH]
    # METIS is an explicit comparison route; loading its library is the only
    # way to learn whether it is usable.
    try:
        metis = metis_identity()
    except phx.meshing.MetisUnavailableError as error:
        metis = f"missing-dependency: {error}"
    else:
        routes.append(kind.METIS)
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
    lineage = adaptation.lineage
    if lineage is None:
        raise qualification_failure(
            "benchmark_repartition",
            ("Repartition benchmark refinement left the mesh unchanged."),
            {"adaptation": adaptation},
        )
    hilbert = distributions[kind.HILBERT]
    transition, stages["host_transition"] = _timed(
        lambda: phx.meshing.prepare_distribution_transition(
            hilbert,
            phx.meshing.MeshPart(part.name, adaptation.target),
            lineage,
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
        raise qualification_failure(
            "benchmark_repartition",
            ("Compiled migration disagreed with the eager transfer."),
            {"moved": moved, "reference": reference, "transition": transition},
        )
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


_PROVIDER_WORKERS: tuple[tuple[BenchmarkProviderKind, str], ...] = (
    ("mmg", "phydrax-mmg-worker"),
    ("omega_h", "phydrax-omega-h-worker"),
)


def _provider_worker() -> tuple[BenchmarkProviderKind, str] | None:
    """First installed provider worker, resolved exactly as its provider does."""
    workers = phx.config.snapshot().workers
    for provider, default in _PROVIDER_WORKERS:
        configured = workers.get(provider)
        located = shutil.which(default if configured is None else configured)
        if located is not None:
            return provider, located
    return None


def benchmark_provider_worker(resolution: int) -> dict[str, object]:
    """Cold (fresh launch per call) versus persistent (reused session) worker calls.

    A real Mmg worker (preferred) or Omega_h worker adapts about ``resolution``
    lattice triangles of the unit square to a uniform metric of half the lattice
    spacing. Host-only stages: preparation, worker launch with its identity
    probe, and every worker call (exchange write, native adaptation, verified
    exchange read, certification). The compiled consumer is quality evaluation
    of the adapted mesh through ``benchmark_adapted_quality``.
    """
    route = _provider_worker()
    if route is None:
        raise qualification_failure(
            "benchmark_provider_worker",
            ("No provider worker is installed."),
            {"route": route},
        )
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
        prepared = prepare_benchmark_provider_input(
            source, metric, provider_name, executable
        )
        return source, prepared.request, prepared.arrays

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
        raise qualification_failure(
            "benchmark_provider_worker",
            ("Persistent provider worker relaunched or did not refine."),
            {"adapted": adapted, "calls": calls, "source": source, "worker": worker},
        )
    mesh = adapted.mesh
    connectivity = {
        block.cell_kind: jax.numpy.asarray(block.vertices, dtype=jax.numpy.int32)
        for block in mesh.blocks
    }
    compiled = {
        "adapted_quality": _compiled_cost(
            benchmark_adapted_quality, mesh.coordinates, connectivity
        )
    }
    _, cold_quality = _timed(
        lambda: jax.block_until_ready(
            benchmark_adapted_quality(mesh.coordinates, connectivity)
        )
    )
    _, warm_quality = _timed(
        lambda: jax.block_until_ready(
            benchmark_adapted_quality(mesh.coordinates, connectivity)
        )
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
        execution = execute_benchmark_optimization(
            plan, mesh, free, contract, measure=_timed, compiler=_compiled_cost
        )
        cost, result = execution.cost, execution.result
        cold, warm = execution.cold_seconds, execution.warm_seconds
        warm_minimization = execution.minimization_seconds
        before = float(
            np.min(np.asarray(phx.meshing.evaluate_cell_quality(mesh).mean_ratios))
        )
        optimized = result.result
        if (
            not result.accepted
            or optimized is None
            or optimized.quality.minimum_mean_ratio <= before
        ):
            raise qualification_failure(
                "benchmark_optimization",
                (
                    f"{name} optimization failed or did not improve the minimum mean "
                    f"ratio: {result.status.value}."
                ),
                {"before": before, "result": result},
            )
        optimizer_status = result.optimizer_status
        if optimizer_status is None:
            raise qualification_failure(
                "benchmark_optimization",
                (
                    f"{name} optimization.optimizer_status must report native "
                    "termination evidence; got None."
                ),
                {"result": result},
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
            "result": _array_bytes(optimized),
        }
        counts[name] = {
            "cells": mesh.blocks[0].cell_count,
            "vertices": mesh.coordinates.shape[0],
            "free_vertices": free.shape[0],
        }
        after = optimized.quality
        quality[name] = {
            "minimum_mean_ratio_before": before,
            "minimum_mean_ratio_after": after.minimum_mean_ratio,
            "minimum_scaled_jacobian_after": after.minimum_scaled_jacobian,
        }
        minimization[name] = {
            "status": result.status.value,
            "optimizer_status": optimizer_status.name,
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
    geometry, preparation = _timed(
        lambda: prepare_squeezed_benchmark_geometry(mesh, degree)
    )
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
        raise qualification_failure(
            "_certification_route",
            (f"{name}: a certified measure leaves its enclosure."),
            {"lower": lower, "outside": outside},
        )
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
        "minimum_certified_determinant_lower_bound": float(np.min(determinant_lower)),
        "certified_mapped_volume": float(np.sum(measure)),
        "certificate_retained_bytes": _array_bytes(certificate),
        "geometry_retained_bytes": _array_bytes(geometry),
    }
    return stages, compiled, route


def benchmark_high_order_certification(resolution: int) -> dict[str, object]:
    """Bernstein validity certification of P2/P3 tetrahedra and Q2 hexahedra.

    About ``resolution`` Kuhn tetrahedra (P2 and P3) and hexahedra (Q2) of the
    unit cube get straight Lagrange geometry nodes moved by the smooth
    Gaussian squeeze. Geometry preparation and `certify_cell_geometry_validity`
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
    """Generate random predicate rows with one quarter rounded to zero determinant.

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
        raise qualification_failure(
            "_predicate_stages",
            (f"{name}: a filter certified a sign that EXACT contradicts."),
            {"exact": exact, "wrong": wrong},
        )
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

    if not meshcore_available():
        raise qualification_failure(
            "benchmark_predicates",
            (
                "The predicates benchmark requires the native meshcore library; set "
                "PHYDRAX_MESHCORE_LIBRARY or install phydrax[meshcore]."
            ),
            {"meshcore_available": meshcore_available},
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


_PARTITION_PART_COUNTS = (2, 8, 32, 64)
_PARTITION_IMBALANCE = 1.03


def _partition_csr(count: int, pairs: np.ndarray, /) -> phx.graph.WeightedCSRGraph:
    """Unit-weight symmetric CSR graph of ``count`` vertices and undirected pairs."""
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    order = np.lexsort((columns, rows))
    offsets = np.zeros((count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=count), out=offsets[1:])
    return phx.graph.WeightedCSRGraph(offsets, columns[order].astype(np.int32))


def _grid_pairs(shape: tuple[int, ...], /) -> np.ndarray:
    """Axis-neighbor vertex pairs of a structured grid."""
    index = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    pairs = []
    for axis in range(len(shape)):
        lower = [slice(None)] * len(shape)
        upper = [slice(None)] * len(shape)
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        pairs.append(
            np.stack(
                (index[tuple(lower)].reshape((-1,)), index[tuple(upper)].reshape((-1,))),
                axis=1,
            )
        )
    return np.concatenate(pairs)


def _simplex_dual_pairs(simplices: np.ndarray, /) -> np.ndarray:
    """Pairs of simplices sharing a facet."""
    count, corners = simplices.shape
    facets = np.concatenate(
        [
            np.sort(np.delete(simplices, corner, axis=1), axis=1)
            for corner in range(corners)
        ]
    )
    owners = np.tile(np.arange(count, dtype=np.int64), corners)
    order = np.lexsort(facets.T[::-1])
    facets, owners = facets[order], owners[order]
    shared = np.flatnonzero(np.all(facets[1:] == facets[:-1], axis=1))
    return np.stack((owners[shared], owners[shared + 1]), axis=1)


def _freudenthal_tetrahedra(cells_per_axis: int, /) -> np.ndarray:
    """Six Freudenthal tetrahedra per cube of a structured lattice."""
    side = cells_per_axis + 1
    index = np.arange(side**3, dtype=np.int64).reshape((side, side, side))
    base = index[:-1, :-1, :-1].reshape((-1,))
    steps = np.asarray((side * side, side, 1), dtype=np.int64)
    tetrahedra = []
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        path = np.cumsum(steps[list(permutation)])
        tetrahedra.append(np.stack((base, *(base + offset for offset in path)), axis=1))
    return np.concatenate(tetrahedra)


def _partition_graphs(resolution: int, /) -> dict[str, phx.graph.WeightedCSRGraph | str]:
    """Deterministic unit, vertex-weighted and disconnected mesh campaigns."""
    rng = np.random.default_rng(resolution)
    square = max(2, round(resolution**0.5))
    cube = max(2, round(resolution ** (1.0 / 3.0)))
    planar_points = rng.random((max(8, resolution // 2), 2))
    spatial_points = rng.random((max(8, resolution // 6), 3))

    def dual(points: np.ndarray) -> phx.graph.WeightedCSRGraph:
        cells = np.asarray(
            phx.geometry.DelaunayTriangulation(points).simplices, dtype=np.int64
        )
        return _partition_csr(cells.shape[0], _simplex_dual_pairs(cells))

    def lattice(dimension: int) -> phx.graph.WeightedCSRGraph:
        cells = (
            _planar_triangles(max(2, round((resolution / 2) ** 0.5))).astype(np.int64)
            if dimension == 2
            else _freudenthal_tetrahedra(max(2, round((resolution / 6) ** (1.0 / 3.0))))
        )
        return _partition_csr(cells.shape[0], _simplex_dual_pairs(cells))

    factories = {
        "grid-2d": lambda: _partition_csr(square * square, _grid_pairs((square, square))),
        "grid-3d": lambda: _partition_csr(cube**3, _grid_pairs((cube, cube, cube))),
        "delaunay-dual-2d": lambda: dual(planar_points),
        "delaunay-dual-3d": lambda: dual(spatial_points),
        "mesh-dual-2d": lambda: lattice(2),
        "mesh-dual-3d": lambda: lattice(3),
    }
    graphs: dict[str, phx.graph.WeightedCSRGraph | str] = {}
    for family, construct in factories.items():
        try:
            graph = construct()
        except Exception as error:
            # Campaign failures are observations, not successful provider fallbacks.
            for suffix in ("", "-weighted", "-disconnected"):
                graphs[family + suffix] = f"{type(error).__name__}: {error}"
            continue
        graphs[family] = graph
        graphs[family + "-weighted"] = phx.graph.WeightedCSRGraph(
            graph.offsets,
            graph.neighbors,
            edge_weights=graph.edge_weights,
            vertex_weights=rng.integers(1, 9, graph.vertex_count, dtype=np.int64),
        )
        graphs[family + "-disconnected"] = phx.graph.WeightedCSRGraph(
            np.concatenate((graph.offsets, graph.offsets[1:] + graph.neighbors.size)),
            np.concatenate((graph.neighbors, graph.neighbors + graph.vertex_count)),
            edge_weights=np.tile(graph.edge_weights, 2),
        )
    return graphs


def _partition_measure(
    graph: phx.graph.WeightedCSRGraph, parts: np.ndarray, part_count: int, /
) -> dict[str, float | int | bool]:
    """Independent acceptance of capacities and required nonempty parts."""
    if parts.shape != (graph.vertex_count,) or parts.dtype.kind not in "iu":
        raise ValueError(
            f"_partition_measure.parts must have one integer owner per graph vertex; got {(parts.shape, parts.dtype)!r}."
        )
    if np.any((parts < 0) | (parts >= part_count)):
        raise ValueError(
            f"_partition_measure.parts must contain owner indices in [0, part_count); got {parts!r}."
        )
    rows = np.repeat(
        np.arange(graph.vertex_count, dtype=np.int64), np.diff(graph.offsets)
    )
    crossing = parts[rows] != parts[graph.neighbors]
    weights = np.zeros((part_count,), dtype=np.int64)
    np.add.at(weights, parts, graph.vertex_weights)
    counts = np.bincount(parts, minlength=part_count)
    target = int(np.sum(graph.vertex_weights)) / part_count
    capacity = min(
        int(np.sum(graph.vertex_weights)),
        max(int(np.floor(_PARTITION_IMBALANCE * target)), int(np.ceil(target))),
    )
    return {
        "edge_cut": int(np.sum(graph.edge_weights[crossing])) // 2,
        "imbalance": float(np.max(weights)) / target,
        "capacity": capacity,
        "maximum_part_weight": int(np.max(weights)),
        "empty_parts": int(np.count_nonzero(counts == 0)),
        "accepted": bool(np.all(weights <= capacity) and np.all(counts > 0)),
    }


def _partition_route(
    graph: phx.graph.WeightedCSRGraph, plan: phx.graph.GraphPartitionPlan, route: str, /
) -> dict[str, Any]:
    """Measure execution/audit phases and truthful process/Python memory."""

    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    before_bytes = tracemalloc.get_traced_memory()[0]
    started = perf_counter()
    record: dict[str, Any] = {}
    phase = "execute_with_boundary_evidence"
    phase_started = started
    try:
        if route == "native":
            result = phx.graph.partition_graph(graph, plan)
            owners = np.asarray(result.parts)
            record["status"] = result.evidence.status
            work = result.evidence.work
            record["work"] = {
                "coarsening_levels": work.coarsening_levels,
                "coarsest_vertices": work.coarsest_vertex_count,
                "matched_pairs": work.matched_pairs,
                "bisection_trials": work.bisection_trials,
                "refinement_passes": work.refinement_passes,
                "moves_committed": work.moves_committed,
                "moves_rolled_back": work.moves_rolled_back,
                "balance_moves": work.balance_moves,
                "nonempty_repairs": work.nonempty_repairs,
                "adjacency_visits": work.adjacency_visits,
                "candidate_evaluations": work.candidate_evaluations,
            }
        elif route == "metis":
            owners = metis_partition(
                np.asarray(graph.offsets),
                np.asarray(graph.neighbors),
                np.asarray(graph.vertex_weights),
                plan.part_count,
                imbalance=plan.maximum_imbalance,
                seed=0,
            )
            record["status"] = "returned"
        else:
            raise ValueError(
                f"_partition_route.route must identify a partition comparison route; got {route!r}."
            )
        execution_seconds = perf_counter() - started
        record["stages_seconds"] = {"execute_with_boundary_evidence": execution_seconds}
        phase = "independent_audit"
        phase_started = perf_counter()
        audit_started = perf_counter()
        record |= _partition_measure(graph, owners, plan.part_count)
        if route == "native":
            record["accepted"] = record["accepted"] and record["status"] == "balanced"
        record["stages_seconds"] = {
            "execute_with_boundary_evidence": execution_seconds,
            "independent_audit": perf_counter() - audit_started,
        }
    except Exception as error:
        record.setdefault("stages_seconds", {})[phase] = perf_counter() - phase_started
        record |= {
            "status": "failed",
            "accepted": False,
            "failure_phase": phase,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        current, peak = tracemalloc.get_traced_memory()
        record["seconds"] = perf_counter() - started
        record["memory_bytes"] = {
            "python_retained_delta": current - before_bytes,
            "python_traced_peak_cumulative": peak,
            "process_peak_rss_cumulative": peak_resident_bytes(),
            "graph_retained": sum(
                array.nbytes
                for array in (
                    graph.offsets,
                    graph.neighbors,
                    graph.edge_weights,
                    graph.vertex_weights,
                )
            ),
        }
        if not tracing:
            tracemalloc.stop()
    return record


def benchmark_graph_partition(resolution: int) -> dict[str, object]:
    """Record every case/failure at fixed tolerance; never drop quality tails."""

    try:
        metis = metis_identity()
    except phx.meshing.MetisUnavailableError as error:
        metis = f"missing-dependency: {error}"
        compare = False
    else:
        compare = True
    graphs, construction_seconds = _timed(lambda: _partition_graphs(resolution))
    cases: list[dict[str, Any]] = []
    ratios: list[float] = []
    zero_baseline_losses = 0
    for family, graph in graphs.items():
        for part_count in _PARTITION_PART_COUNTS:
            case: dict[str, Any] = {"family": family, "parts": part_count}
            cases.append(case)
            if isinstance(graph, str):
                case |= {
                    "construction_error": graph,
                    "failure_phase": "graph_construction",
                    "native": {"status": "not_run", "accepted": False},
                    "metis": {"status": "not_run", "accepted": False},
                }
                continue
            case |= {
                "vertices": graph.vertex_count,
                "edges": graph.neighbors.size // 2,
                "graph_id": graph.graph_id,
            }
            plan = phx.graph.GraphPartitionPlan(
                part_count,
                maximum_imbalance=_PARTITION_IMBALANCE,
                work_limit=max(
                    100_000, 100_000 * (graph.vertex_count + graph.neighbors.size)
                ),
            )
            case["plan_id"] = plan.plan_id
            case["native"] = _partition_route(graph, plan, "native")
            case["metis"] = (
                _partition_route(graph, plan, "metis")
                if compare
                else {"status": "unavailable", "accepted": False}
            )
            case["cut_ratio"] = None
            if case["native"]["accepted"] and case["metis"]["accepted"]:
                native_cut, baseline_cut = (
                    case["native"]["edge_cut"],
                    case["metis"]["edge_cut"],
                )
                if baseline_cut:
                    ratio = native_cut / baseline_cut
                elif native_cut == 0:
                    ratio = 1.0
                else:
                    zero_baseline_losses += 1
                    case["zero_baseline_cut_loss"] = True
                    continue
                ratios.append(ratio)
                case["cut_ratio"] = ratio
    all_accepted = all(
        case["native"]["accepted"] and case["metis"]["accepted"] for case in cases
    )
    median = float(np.median(ratios)) if ratios else None
    maximum = float(np.max(ratios)) if ratios and zero_baseline_losses == 0 else None
    return {
        "resolution": resolution,
        "stages_seconds": {"graph_construction": construction_seconds},
        "metis": metis,
        "imbalance_tolerance": _PARTITION_IMBALANCE,
        "memory_scope": "Python traced bytes exclude native allocations; process RSS is cumulative, not per-phase.",
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "comparable_cases": len(ratios),
            "native_not_accepted": sum(not case["native"]["accepted"] for case in cases),
            "metis_not_accepted": sum(not case["metis"]["accepted"] for case in cases),
            "median_cut_ratio": median,
            "maximum_cut_ratio": maximum,
            "zero_baseline_cut_losses": zero_baseline_losses,
            "cases_above_three_percent": sum(ratio > 1.03 for ratio in ratios)
            + zero_baseline_losses,
            "quality_target_met": bool(
                all_accepted
                and zero_baseline_losses == 0
                and median is not None
                and maximum is not None
                and median <= 1.0
                and maximum <= 1.03
            ),
        },
    }


def benchmark_scope_construction(resolution: int, repeats: int, /) -> dict[str, object]:
    """Measure scope admission separately from import and diagnostic profiling."""
    if resolution < 1 or repeats < 1:
        raise ValueError(
            "ScopeBenchmark.controls must contain positive capacity and repeats; "
            f"got {(resolution, repeats)!r}."
        )
    identifiers = np.arange(resolution, dtype=np.int64)

    def construct() -> phx.meshing.MeshingScope:
        return phx.meshing.MeshingScope(
            "benchmark-scope",
            "fixed-scope-revision",
            phx.meshing.MeshingEntityKind.MESH,
            2,
            "scope-entities",
            identifiers,
        )

    started = perf_counter()
    scope = construct()
    first_seconds = perf_counter() - started
    constructions_per_sample = 1000
    batches = []
    for _ in range(repeats):
        started = perf_counter()
        for _ in range(constructions_per_sample):
            scope = construct()
        batches.append(perf_counter() - started)
    samples = np.asarray(batches, dtype=np.float64) / constructions_per_sample
    diagnostic_mode = os.environ.get("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS") == "1"
    diagnostics: dict[str, Any] = {}
    if diagnostic_mode:
        from tools.meshing_qualification._runtime import StageLedger

        StageLedger(diagnostics).run("warm_scope_diagnostic", construct)
    return {
        "resolution": resolution,
        "repeats": repeats,
        "capacity_bucket": resolution,
        "scope_id": scope.scope_id,
        "warmup_constructions": 1,
        "constructions_per_sample": constructions_per_sample,
        "measured_construction_count": repeats * constructions_per_sample,
        "first_construction_seconds": first_seconds,
        "warm_batch_samples_seconds": batches,
        "warm_construction_samples_seconds": samples.tolist(),
        "warm_construction_median_seconds": float(np.median(samples)),
        "diagnostic_measurements": diagnostics.get("stage_measurements"),
        "accounting_measurement": "measured-separate-diagnostic-process"
        if diagnostic_mode
        else "unmeasured-clean-timing-no-accounting-profiler",
        "timing_is_headline": not diagnostic_mode,
        "diagnostic_timing_is_headline": False,
    }


def _repeated_case(
    runner: Any,
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
    "fresh-imports": benchmark_fresh_imports,
    "parallel-batches": benchmark_parallel_batches,
    "scope-construction": benchmark_scope_construction,
    "brep-containment": benchmark_brep_containment,
    "brep-moments": benchmark_brep_moments,
    "brep-orientation-audit": benchmark_brep_orientation_audit,
    "brep-preconditioners": benchmark_brep_preconditioners,
    "brep-quadratic-bounds": benchmark_brep_quadratic_bounds,
    "brep-interval-program": benchmark_brep_interval_program,
    "cell-validity": benchmark_cell_validity,
    "homotopy-arithmetic": benchmark_homotopy_arithmetic,
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
    "implicit-discovery": lambda resolution, repeats: _repeated_case(
        benchmark_implicit_discovery,
        resolution,
        repeats,
    ),
    "lbvh": lambda resolution, repeats: _repeated_case(
        lambda value: qualify_bvh_scaling(value, measure=_timed),
        resolution,
        repeats,
    ),
    "overlap-pairs": lambda resolution, repeats: _repeated_case(
        lambda value: qualify_overlap_pair_scaling(
            value, measure=_timed, failure=qualification_failure
        ),
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
    "graph-partition": lambda resolution, repeats: _repeated_case(
        benchmark_graph_partition,
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
    "predicates": lambda resolution, repeats: _repeated_case(
        benchmark_predicates,
        resolution,
        repeats,
    ),
}


def _benchmark_parser(
    profiles: dict[str, dict[str, Any]],
) -> argparse.ArgumentParser:
    """Declare benchmark CLI controls and source/profile selectors."""
    family_choices, family_default = FAMILY_CASES, "dual-hex"
    surface_choices, surface_default = SURFACE_CASES, "sphere"
    hybrid_choices, hybrid_default = HYBRID_PROFILES, "unit-box"
    hybrid_core_default = 0.1
    parser = argparse.ArgumentParser()
    add_case_arguments(parser)
    parser.add_argument("--resolution", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--benchmark-warm-repeats",
        type=int,
        default=5,
        help="Warm solver samples per cold benchmark child; --repeats selects cold attempts.",
    )
    parser.add_argument(
        "--corpus-case", choices=CASE_NAMES, default="planar-hole-feature"
    )
    parser.add_argument(
        "--corpus-profile",
        nargs="+",
        choices=tuple(profiles),
        default=None,
        help="Benchmark only these profiles while retaining registered gaps.",
    )
    parser.add_argument(
        "--polyhedral-source",
        choices=POLYHEDRAL_CASE_NAMES,
        nargs="+",
        default=list(POLYHEDRAL_CASE_NAMES),
    )
    parser.add_argument("--capacity", type=int, nargs="+", default=[20000])
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-case whole-workflow deadline in seconds; stages and native calls share it.",
    )
    parser.add_argument("--target-error", type=float, default=0.05)
    parser.add_argument("--adaptation-rounds", type=int, default=1)
    parser.add_argument("--comparison", choices=("triangle",), default=None)
    parser.add_argument("--cad-format", choices=("step", "iges", "brep"), default="step")
    parser.add_argument(
        "--periodic-case",
        choices=tuple(_periodic_qualification_profile_contracts()),
        default=BenchmarkControls().periodic_case,
    )
    parser.add_argument("--maximum-memory-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--maximum-condition", type=float, default=1.0e12)
    parser.add_argument("--family-case", choices=family_choices, default=family_default)
    parser.add_argument(
        "--surface-case", choices=surface_choices, default=surface_default
    )
    parser.add_argument(
        "--hybrid-profile", choices=hybrid_choices, default=hybrid_default
    )
    parser.add_argument("--hybrid-archive")
    parser.add_argument("--hybrid-content-id")
    parser.add_argument("--hybrid-core-size", type=float, default=hybrid_core_default)
    parser.add_argument(
        "--hybrid-region-densities", type=float, nargs="+", default=None
    )
    parser.add_argument("--hybrid-material-site", default=None)
    parser.add_argument("--layer-count", type=int, default=2)
    parser.add_argument("--first-thickness", type=float, default=0.1)
    parser.add_argument("--layer-growth", type=float, default=1.0)
    parser.add_argument("--geometry-order", type=int, choices=(2, 3, 4, 6, 10), default=2)
    parser.add_argument("--implicit-radius", type=float, default=0.371)
    parser.add_argument("--implicit-fidelity", type=float, default=0.02)
    parser.add_argument("--implicit-maximum-level", type=int, default=9)
    parser.add_argument("--implicit-edge-isolation-depth", type=int, default=8)
    parser.add_argument("--implicit-root-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--implicit-refinement-rounds", type=int, default=8)
    parser.add_argument("--implicit-improvement-passes", type=int, default=8)
    parser.add_argument("--implicit-metric-passes", type=int, default=16)
    parser.add_argument("--distributed-parts", type=int, default=2)
    parser.add_argument("--restart-parts", type=int, default=1)
    parser.add_argument("--distributed-processes", type=int, default=2)
    parser.add_argument(
        "--distributed-source",
        choices=("native-unit-square",),
        default="native-unit-square",
    )
    parser.add_argument("--distributed-halo-width", type=int, default=2)
    parser.add_argument("--distributed-halo-capacity", type=int, default=20000)
    parser.add_argument("--distributed-cavity-capacity", type=int, default=20000)
    parser.add_argument("--distributed-slack", type=int, default=2)
    parser.add_argument(
        "--distributed-host-packet-bytes", type=int, default=64 * 1024 * 1024
    )
    parser.add_argument("--distributed-checkpoint-root", default=None)
    parser.add_argument(
        "--distributed-bisection-compatibility",
        type=BisectionCompatibility,
        choices=tuple(BisectionCompatibility),
        default=BisectionCompatibility.REJECT,
        help="Initial matching-label preparation, charged to the unchanged request limits.",
    )
    parser.add_argument("--image-size", type=int, default=2)
    parser.add_argument("--transport-steps", type=int, default=5)
    parser.add_argument("--transport-dt", type=float, default=0.01)
    parser.add_argument("--balance-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--oracle-capacity", type=int, default=2048)
    parser.add_argument("--overset-motion-steps", type=int, default=2)
    parser.add_argument("--overset-wall-pairs", type=int, default=500000)
    parser.add_argument("--overset-donor-pairs", type=int, default=500000)
    parser.add_argument("--overset-overlap-pairs", type=int, default=500000)
    parser.add_argument(
        "--gmsh-semantic",
        action="store_true",
        help="Also run semantic Gmsh scaling at 2, 8, and 32 regions.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            *_CASES,
            "native-lifecycle",
            "native-polyhedral-lifecycle",
            "native-corpus",
            *_NATIVE_DRIVER_ARGUMENTS,
        ),
        help="Run one named scaling case; defaults to native meshing only.",
    )
    return parser


@final
class BenchmarkExecution(phx.StrictModule):
    """Selected campaigns and their ordered benchmark results."""

    selected: tuple[str, ...] = eqx.field(static=True)
    output: dict[str, dict[str, object] | list[dict[str, object]]]


def _run_benchmark_cases(
    args: BenchmarkControls, profiles: dict[str, dict[str, Any]]
) -> BenchmarkExecution:
    """Execute requested scaling campaigns in CLI selection order."""
    selected = ("native-meshing",) if args.case is None else tuple(args.case)
    output: dict[str, dict[str, object] | list[dict[str, object]]] = {}
    for name in selected:
        if name == "native-corpus":
            output[name] = _benchmark_registered_native_corpus(args, profiles)
            continue
        if name == "brep-containment":
            output[name] = benchmark_brep_containment_campaign(
                args.resolution,
                args.repeats,
                run_case=lambda resolution: _clean_benchmark_case(
                    name,
                    lambda: benchmark_brep_containment(resolution, args.repeats),
                ),
            )
            continue
        if name in _NATIVE_DRIVER_ARGUMENTS:
            controls = _native_driver_controls(name, args)
            output[name] = [
                benchmark_native_lifecycle(
                    name,
                    resolution,
                    capacity,
                    args.timeout,
                    args.repeats,
                    args.target_error,
                    args.adaptation_rounds,
                    args.comparison,
                    scenario_override=name,
                    benchmark_warm_repeats=args.benchmark_warm_repeats,
                    controls=controls,
                )
                for resolution in args.resolution
                for capacity in args.capacity
            ]
            continue
        if name == "native-polyhedral-lifecycle":
            polyhedral_controls: dict[str, str | int | float | tuple[float, ...]] = {
                key: value for key, value in (
                    ("maximum_work_units", args.maximum_work_units),
                    ("maximum_geometry_queries", args.maximum_geometry_queries),
                    ("maximum_scratch_bytes", args.maximum_scratch_bytes),
                ) if value is not None
            }
            output[name] = [
                benchmark_native_lifecycle(
                    source,
                    sites,
                    capacity,
                    args.timeout,
                    args.repeats,
                    args.target_error,
                    0,
                    args.comparison,
                    benchmark_warm_repeats=args.benchmark_warm_repeats,
                    controls=polyhedral_controls,
                )
                for source in args.polyhedral_source
                for sites in args.resolution
                for capacity in args.capacity
            ]
            continue
        if name == "native-lifecycle":
            planar_controls: dict[str, str | int | float | tuple[float, ...]] = {
                key: value for key, value in (
                    ("maximum_work_units", args.maximum_work_units),
                    ("maximum_geometry_queries", args.maximum_geometry_queries),
                    ("maximum_scratch_bytes", args.maximum_scratch_bytes),
                ) if value is not None
            }
            output[name] = [
                benchmark_native_lifecycle(
                    args.corpus_case,
                    resolution,
                    capacity,
                    args.timeout,
                    args.repeats,
                    args.target_error,
                    args.adaptation_rounds,
                    args.comparison,
                    benchmark_warm_repeats=args.benchmark_warm_repeats,
                    controls=planar_controls,
                )
                for resolution in args.resolution
                for capacity in args.capacity
            ]
        else:
            output[name] = [
                _CASES[name](value, args.repeats)
                if name in ("fresh-imports", "parallel-batches")
                else _clean_benchmark_case(
                    name, lambda: _CASES[name](value, args.repeats)
                )
                for value in args.resolution
            ]
    if args.gmsh_semantic:
        with TemporaryDirectory(prefix="phydrax-gmsh-semantic-benchmark-") as temporary:
            directory = Path(temporary)
            output["gmsh_semantic"] = [
                _clean_benchmark_case(
                    "gmsh_semantic",
                    lambda: benchmark_gmsh_semantic(value, args.repeats, directory),
                )
                for value in (2, 8, 32)
            ]
    return BenchmarkExecution(selected=selected, output=output)


def main() -> None:

    profiles = _corpus_profiles()
    parser = _benchmark_parser(profiles)
    args = BenchmarkControls.from_namespace(parser.parse_args())
    if (
        args.repeats < 1
        or args.benchmark_warm_repeats < 1
        or any(value < 1 for value in (*args.resolution, *args.capacity))
        or args.timeout <= 0.0
        or args.target_error <= 0.0
        or args.adaptation_rounds < 0
    ):
        parser.error(
            "resolutions, capacities, repeats, timeout and error must be positive; rounds nonnegative"
        )
    execution = _run_benchmark_cases(args, profiles)
    registration = load_native_campaign_registration()
    suite: dict[str, object] = {
        "kind": "meshing-benchmark-suite",
        "campaign_registration_id": registration["registration_id"],
        "selected_cases": list(execution.selected),
        "campaigns": execution.output,
    }
    print(json.dumps({**suite, "suite_id": canonical_fingerprint(suite)}, indent=2))


if __name__ == "__main__":
    main()

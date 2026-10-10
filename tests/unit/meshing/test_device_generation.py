#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
)
from phydrax.meshing._device_generation import (
    DeviceGenerationLayout,
    DeviceGenerationStatus,
    evaluate_device_generation_candidates,
    execute_device_generation_round,
    prepare_device_generation,
    PreparedDeviceGeneration,
)
from phydrax.meshing._device_tetra_metric import (
    adapt_device_tetra_metric,
    DeviceTetraMetricLayout,
    prepare_device_tetra_metric,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native exact meshcore is unavailable"
)
_POINTS = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
_CELL = np.asarray(((0, 1, 2, 3),), dtype=np.int32)


def _tetra(
    *, vertices: int = 8, cells: int = 16, candidates: int = 8, work: int = 1000
) -> PreparedDeviceGeneration:
    layout = DeviceGenerationLayout(
        vertex_capacity=vertices,
        cell_capacity=cells,
        candidate_capacity=candidates,
        maximum_work_units=work,
    )
    return prepare_device_generation(
        layout,
        points=_POINTS,
        cells=_CELL,
        vertex_ids=np.asarray((11, 21, 31, 41), dtype=np.int64),
        cell_ids=np.asarray((501,), dtype=np.int64),
        cell_classes=np.asarray((7,), dtype=np.int64),
        target_sizes=np.asarray((0.5,), dtype=np.float64),
    )


def _volumes(prepared: PreparedDeviceGeneration) -> np.ndarray:
    mesh = prepared.mesh
    corners = np.asarray(mesh.coordinates)[
        np.asarray(mesh.cells)[np.asarray(mesh.cell_active)]
    ]
    return np.linalg.det(np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)) / 6.0


def test_tetra_generation_preserves_coverage_regions_and_semantic_ids() -> None:
    source = _tetra()
    update = execute_device_generation_round(source)
    assert update.evidence.status is DeviceGenerationStatus.COMPLETE
    assert update.evidence.accepted_vertices == 1
    np.testing.assert_allclose(
        _volumes(update.prepared), np.full((4,), 1.0 / 24.0), rtol=0, atol=1e-15
    )
    assert np.sum(_volumes(update.prepared)) == pytest.approx(1.0 / 6.0)
    mesh = update.prepared.mesh
    np.testing.assert_array_equal(np.asarray(mesh.vertex_ids)[:5], (11, 21, 31, 41, 42))
    active = np.asarray(mesh.cell_active)
    np.testing.assert_array_equal(np.asarray(mesh.cell_ids)[active], (502, 503, 504, 505))
    np.testing.assert_array_equal(
        np.asarray(update.prepared.cell_classes)[active], np.full((4,), 7)
    )
    # Four internal star faces pair; each original boundary face survives once.
    assert np.sum(np.asarray(mesh.boundary_facets)) == 4


@pytest.mark.parametrize(
    "vertices,cells,work,status",
    [
        pytest.param(
            4, 16, 1000, DeviceGenerationStatus.CAPACITY_EXCEEDED, id="vertex-capacity"
        ),
        pytest.param(
            8, 1, 1000, DeviceGenerationStatus.CAPACITY_EXCEEDED, id="cell-capacity"
        ),
        pytest.param(8, 16, 1, DeviceGenerationStatus.WORK_LIMIT, id="topology-work"),
    ],
)
def test_generation_resource_refusal_is_atomic(
    vertices: int, cells: int, work: int, status: DeviceGenerationStatus
) -> None:
    source = _tetra(vertices=vertices, cells=cells, candidates=min(cells, 8), work=work)
    update = execute_device_generation_round(source)
    assert update.evidence.status is status
    assert update.prepared is source
    assert update.evidence.accepted_vertices == 0
    np.testing.assert_array_equal(np.asarray(source.mesh.cell_ids)[:1], (501,))


def test_candidate_overflow_does_not_silently_truncate_requested_cells() -> None:
    source = execute_device_generation_round(_tetra()).prepared
    layout = DeviceGenerationLayout(
        vertex_capacity=8, cell_capacity=16, candidate_capacity=1, maximum_work_units=1000
    )
    source = eqx.tree_at(lambda prepared: prepared.layout, source, layout)
    batch = evaluate_device_generation_candidates(source)
    assert int(np.asarray(batch.count)) == 4
    update = execute_device_generation_round(source, candidates=batch)
    assert update.evidence.status is DeviceGenerationStatus.CAPACITY_EXCEEDED
    assert update.prepared is source


def test_surface_generation_inserts_projected_geometry_and_preserves_feature_edges() -> (
    None
):
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    layout = DeviceGenerationLayout(
        vertex_capacity=8, cell_capacity=8, candidate_capacity=4, maximum_work_units=1000
    )
    source = prepare_device_generation(
        layout,
        points=points,
        cells=np.asarray(((0, 1, 2),), dtype=np.int32),
        vertex_ids=np.asarray((10, 20, 30), dtype=np.int64),
        cell_ids=np.asarray((100,), dtype=np.int64),
        cell_classes=np.asarray((5,), dtype=np.int64),
        target_sizes=np.asarray((0.5,), dtype=np.float64),
        charts=points[:, :2].copy(),
        normals=np.tile(np.asarray((0, 0, 1), dtype=np.float64), (3, 1)),
        constrained=np.ones((1, 3), dtype=np.bool_),
    )
    batch = evaluate_device_generation_candidates(source)
    chart_points = np.asarray(batch.charts)
    physical = np.column_stack((chart_points, np.zeros((4,), dtype=np.float64)))
    update = execute_device_generation_round(
        source,
        candidates=batch,
        surface_points=physical,
        surface_normals=np.tile(np.asarray((0, 0, 1), dtype=np.float64), (4, 1)),
    )
    assert update.evidence.status is DeviceGenerationStatus.COMPLETE
    assert update.evidence.geometry_queries == 0
    assert update.evidence.accepted_vertices == 1
    active = np.asarray(update.prepared.mesh.cell_active)
    assert np.sum(np.asarray(update.prepared.constrained)[active]) == 3
    triangles = np.asarray(update.prepared.mesh.cells)[active]
    xy = np.asarray(update.prepared.charts)[triangles]
    area = (
        (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
        - (xy[:, 1, 1] - xy[:, 0, 1]) * (xy[:, 2, 0] - xy[:, 0, 0])
    ) * 0.5
    assert np.all(area > 0)
    assert np.sum(area) == pytest.approx(0.5)


def _adaptive(layout: AdaptiveSimplexLayout) -> AdaptiveSimplexState:
    return adaptive_simplex_state(
        layout,
        coordinates=_POINTS,
        vertex_ids=np.asarray((11, 21, 31, 41), dtype=np.int64),
        vertex_active=np.ones((4,), dtype=np.bool_),
        vertex_parents=np.full((4, 2), -1, dtype=np.int32),
        vertex_protected=np.zeros((4,), dtype=np.bool_),
        cells=_CELL,
        tuples=_CELL,
        tags=np.asarray((3,), dtype=np.int32),
        blocks=np.zeros((1,), dtype=np.int32),
        generations=np.zeros((1,), dtype=np.int32),
        parents=np.full((1,), -1, dtype=np.int32),
        children=np.full((1, 2), -1, dtype=np.int32),
        bisection_vertices=np.full((1,), -1, dtype=np.int32),
        cell_ids=np.asarray((501,), dtype=np.int64),
        cell_active=np.ones((1,), dtype=np.bool_),
        cell_classes=np.asarray((7,), dtype=np.int32),
        facet_classes=np.ones((1, 4), dtype=np.int32),
        protected_edges=np.empty((0, 2), dtype=np.int32),
        next_vertex_id=42,
        next_cell_id=502,
    )


def _metric_layout(*, vertices: int = 16, cells: int = 32) -> DeviceTetraMetricLayout:
    return DeviceTetraMetricLayout(
        AdaptiveSimplexLayout(
            3,
            3,
            vertex_capacity=vertices,
            cell_capacity=cells,
            protected_edge_capacity=1,
            maximum_closure_iterations=32,
            maximum_coarsening_passes=32,
        )
    )


def test_tetra_metric_refine_then_complete_family_coarsen_restores_domain() -> None:
    layout = _metric_layout()
    state = prepare_device_tetra_metric(
        layout,
        _adaptive(layout.simplex),
        np.tile(4.0 * np.eye(3, dtype=np.float64), (16, 1, 1)),
    )
    refined = adapt_device_tetra_metric(layout, state)
    assert refined.evidence.committed
    assert int(np.asarray(refined.report.refinement.operations)) > 0
    live = np.asarray(refined.state.simplex.mesh.cell_active)
    points = np.asarray(refined.state.simplex.mesh.coordinates)
    cells = np.asarray(refined.state.simplex.mesh.cells)[live]
    determinants = np.linalg.det(
        np.swapaxes(points[cells][:, 1:] - points[cells][:, :1], -1, -2)
    )
    assert np.all(determinants > 0)
    assert np.sum(determinants) == pytest.approx(1.0)
    small = eqx.tree_at(
        lambda value: value.metric,
        refined.state,
        jnp.tile(0.01 * jnp.eye(3, dtype=jnp.float64), (16, 1, 1)),
    )
    restored = adapt_device_tetra_metric(layout, small)
    assert restored.evidence.committed
    active = np.asarray(restored.state.simplex.mesh.cell_active)
    np.testing.assert_array_equal(
        np.asarray(restored.state.simplex.mesh.cells)[active], _CELL
    )
    np.testing.assert_array_equal(
        np.asarray(restored.state.simplex.mesh.cell_ids)[active], (501,)
    )


def test_tetra_metric_capacity_failure_rolls_back_both_phases_and_metrics() -> None:
    layout = _metric_layout(vertices=4, cells=1)
    source = prepare_device_tetra_metric(
        layout,
        _adaptive(layout.simplex),
        np.tile(4.0 * np.eye(3, dtype=np.float64), (4, 1, 1)),
    )
    update = adapt_device_tetra_metric(layout, source)
    assert not update.evidence.committed
    assert update.evidence.status & int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    np.testing.assert_array_equal(
        np.asarray(update.state.simplex.mesh.cells), np.asarray(source.simplex.mesh.cells)
    )
    np.testing.assert_array_equal(
        np.asarray(update.state.metric), np.asarray(source.metric)
    )


def test_uncertain_thin_tetra_rejects_rounded_degenerate_star_before_publication() -> (
    None
):
    points = np.asarray(
        ((0, 0, 0), (1, 1, 0), (1, np.nextafter(1.0, 2.0), 0), (0, 0, 1)),
        dtype=np.float64,
    )
    layout = DeviceGenerationLayout(
        vertex_capacity=8, cell_capacity=8, candidate_capacity=4, maximum_work_units=1000
    )
    source = prepare_device_generation(
        layout,
        points=points,
        cells=_CELL,
        vertex_ids=np.asarray((11, 21, 31, 41), dtype=np.int64),
        cell_ids=np.asarray((501,), dtype=np.int64),
        cell_classes=np.asarray((7,), dtype=np.int64),
        target_sizes=np.asarray((0.5,), dtype=np.float64),
    )
    batch = evaluate_device_generation_candidates(source)
    assert int(np.asarray(batch.uncertain)) == 1
    update = execute_device_generation_round(source, candidates=batch)
    assert update.evidence.status is DeviceGenerationStatus.INVALID_GEOMETRY
    assert update.prepared is source
    assert update.evidence.exact_resolution_count > 0
    assert update.evidence.accepted_vertices == 0


# GENERATION: target cells originate in owning constrained charts, not a
# recovered global mesh. These scenarios exercise actual process asymmetry.
def _run_initial_scenario(scenario: str) -> list[dict[str, object]]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        PHYDRAX_CPU_COLLECTIVES="gloo",
    )
    root = Path(__file__).parents[3]
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _INITIAL_GENERATION_SCRIPT,
                str(rank),
                coordinator,
                scenario,
            ],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in range(2)
    ]
    reports: list[dict[str, object]] = []
    try:
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=180)
            if worker.returncode:
                pytest.fail(stdout + "\n" + stderr, pytrace=False)
            reports.append(json.loads(stdout))
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    return reports


@pytest.mark.parametrize(
    "scenario", ("shared_constraint", "empty_owner", "one_owner_overflow")
)
def test_native_distributed_initial_source_patch_construction(scenario: str) -> None:
    reports = _run_initial_scenario(scenario)
    if scenario == "one_owner_overflow":
        assert [report["rejected"] for report in reports] == [True, True]
    elif scenario == "empty_owner":
        assert reports[1]["empty"] is True
        assert reports[0]["area"] == pytest.approx(2.0)
        assert reports[0]["global_cells"] == reports[0]["local_cells"]
    else:
        assert [report["area"] for report in reports] == pytest.approx([1.0, 1.0])
        assert reports[0]["shared_ids"] == reports[1]["shared_ids"]
        assert reports[0]["shared_points"] == reports[1]["shared_points"]


def _report_number(report: dict[str, object], name: str) -> float:
    value = report[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a numerical process report value.")
    return float(value)


def _report_zones(report: dict[str, object]) -> list[list[str]]:
    values = report["zones"]
    if not isinstance(values, list):
        raise TypeError("zones must be a process report sequence.")
    result = []
    for row in values:
        if not isinstance(row, list) or len(row) != 4:
            raise TypeError(
                "A zone report must contain its four declared identity fields."
            )
        name, material, role, identity = row
        if not (
            isinstance(name, str)
            and isinstance(material, str)
            and isinstance(role, str)
            and isinstance(identity, str)
        ):
            raise TypeError("Zone report identity fields must be strings.")
        result.append([name, material, role, identity])
    return result


def test_native_initial_publication_reaches_finite_element_affine_solve() -> None:
    reports = _run_initial_scenario("accepted_consumer")
    assert reports[0]["result_id"] == reports[1]["result_id"]
    assert [report["owned_area"] for report in reports] == pytest.approx(
        [1.0, 1.0], abs=1.0e-12
    )
    assert all(_report_number(report, "solve_error") < 1.0e-9 for report in reports)
    assert reports[0]["organization"] == reports[1]["organization"]
    assert all(_report_number(report, "locally_empty_scopes") > 0 for report in reports)


@pytest.mark.parametrize("scenario", ("remote_source_failure", "physical_overlap"))
def test_native_initial_theorem_collectively_rejects_invalid_geometry(
    scenario: str,
) -> None:
    reports = _run_initial_scenario(scenario)
    assert [report["rejected"] for report in reports] == [True, True]
    assert reports[0]["failure_id"] == reports[1]["failure_id"]
    if scenario == "remote_source_failure":
        assert reports[0]["local_category"] == 0
        assert _report_number(reports[1], "local_category") > 0
    else:
        assert _report_number(reports[0], "native_intersection_count") > 0
        assert reports[1]["native_intersection_count"] == 0


def test_native_initial_empty_owner_publishes_actual_zero_resident_storage() -> None:
    reports = _run_initial_scenario("accepted_empty_owner")
    assert reports[1]["empty"] is True
    assert reports[0]["result_id"] == reports[1]["result_id"]
    assert reports[0]["global_cells"] == reports[1]["global_cells"]
    assert reports[0]["owned_area"] == pytest.approx(2.0, abs=1.0e-12)


def test_native_initial_authored_material_and_interface_inventory_is_global() -> None:
    reports = _run_initial_scenario("accepted_organization")
    assert reports[0]["organization"] == reports[1]["organization"]
    assert reports[0]["zones"] == reports[1]["zones"]
    assert sorted(row[:3] for row in _report_zones(reports[0])) == [
        ["left", "material-left", "user"],
        ["right", "material-right", "user"],
    ]


def test_native_initial_local_audit_rejection_does_not_strand_another_owner() -> None:
    reports = _run_initial_scenario("local_audit_failure")
    assert [report["rejected"] for report in reports] == [True, True]


def test_native_initial_passive_storage_rejects_self_issued_forged_coefficient_banks() -> (
    None
):
    reports = _run_initial_scenario("storage_bank_forgery")
    assert [report["forged_bank_rejected"] for report in reports] == [True, True]
    assert reports[1]["passive_constructor_rejected"] is True
    assert [report["independent_buffer_admitted"] for report in reports] == [True, True]


_INITIAL_GENERATION_SCRIPT = r"""
import json
import sys
import jax
jax.distributed.initialize(coordinator_address=sys.argv[2], num_processes=2,
                           process_id=int(sys.argv[1]), local_device_ids=[0])
import numpy as np
import phydrax as phx
from examples._native_surface_sources import folded_plates
from phydrax.meshing._distributed_generation import execute_distributed_surface_initial
M = phx.meshing
rank, scenario = jax.process_index(), sys.argv[3]
domain = folded_plates().domain
if scenario == "physical_overlap":
    from phydrax.geometry._meshing_domain import MeshingSurfacePatch, MeshingDomain
    from phydrax.geometry.brep._patches import PlanePatch
    second = MeshingSurfacePatch(
        PlanePatch((0, 0, 0), (-1, 0, 0), (0, 1, 0)), domain.patches[1].loops)
    domain = MeshingDomain(
        (domain.patches[0], second), domain.curves, domain.corner_points.shape[0],
        source_id="overlapping-plates", source_revision="r1")
scope = M.MeshingScope(domain.source_id, domain.source_revision,
                      M.MeshingEntityKind.GEOMETRY, 2, domain.entity_set_id(2),
                      np.arange(len(domain.patches), dtype=np.int64))
region_controls = ()
patch_controls = ()
if scenario == "accepted_organization":
    selected = tuple(M.MeshingScope(
        domain.source_id, domain.source_revision, M.MeshingEntityKind.GEOMETRY,
        2, domain.entity_set_id(2), np.asarray((index,), dtype=np.int64)) for index in range(2))
    region_controls = tuple(M.RegionControl(
        selected[index], name, f"material-{name}", M.RegionRole.USER)
        for index, name in enumerate(("left", "right")))
    edge_scope = M.MeshingScope(
        domain.source_id, domain.source_revision, M.MeshingEntityKind.GEOMETRY,
        1, domain.entity_set_id(1), np.asarray((0,), dtype=np.int64))
    patch_controls = (M.PatchControl("shared-interface", edge_scope, ("left", "right")),)
specification = M.SurfaceMeshingSpec(
    M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))), scope,
    size_controls=(M.UniformSizeControl(scope, .35, strength=M.SizeControlStrength.SOFT),),
    region_controls=region_controls, patch_controls=patch_controls,
    quality_target=M.MeshQualityTarget(minimum_angle=.25, hard=True),
)
owners = np.asarray((0, 0) if scenario in ("empty_owner", "accepted_empty_owner", "storage_bank_forgery") else (0, 1), dtype=np.int32)
small = scenario == "one_owner_overflow" and rank == 1
layout = M.DeviceGenerationLayout(
    vertex_capacity=8 if small else 1000, cell_capacity=16 if small else 2000,
    candidate_capacity=8 if small else 64, maximum_work_units=2_000_000,
)
schedule = M.NativeSurfaceSchedule(quality_angle_degrees=float(np.rad2deg(.25)))
prepared = M.prepare_distributed_surface_generation(
    M.NativeSurfaceSource(domain), specification, schedule, layout, owners,
    maximum_metadata_bytes=10_000,
)
try:
    if scenario in ("shared_constraint", "empty_owner", "one_owner_overflow"):
        generated = execute_distributed_surface_initial(prepared, minimum_angle=.25)
except M.MeshingFailure as error:
    assert scenario == "one_owner_overflow"
    assert error.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    print(json.dumps({"rejected": True}), flush=True)
else:
    assert scenario != "one_owner_overflow"
    if scenario in ("accepted_consumer", "accepted_empty_owner", "accepted_organization",
                    "remote_source_failure", "physical_overlap", "local_audit_failure", "storage_bank_forgery"):
        import equinox as eqx
        import jax.numpy as jnp
        from dataclasses import replace
        from phydrax.meshing._initial_certification import InitialCollectiveMeshEvidence
        from phydrax.meshing._result import require_original_meshing_source
        provider = M.NativeMeshingProvider(M.NativeMeshingOptions(
            "parametric_surface", surface_schedule=schedule))
        if scenario == "local_audit_failure" and rank == 1:
            from typing import Any
            import phydrax.meshing._audit as audit_module
            original_audit = audit_module.audit_cell_mesh
            def strict_audit(*args: Any, **kwargs: Any) -> Any:
                kwargs["policy"] = audit_module.CellMeshAuditPolicy(
                    require_complete_association=True, minimum_mean_ratio=.999999)
                return original_audit(*args, **kwargs)
            audit_module.audit_cell_mesh = strict_audit
        if scenario == "remote_source_failure" and rank == 1:
            import phydrax.meshing._distributed_generation as generation_module
            original_generation = generation_module.execute_distributed_surface_initial
            def corrupt_source(
                prepared: generation_module.PreparedDistributedSurfaceGeneration, /, *,
                minimum_angle: float, maximum_normal_angle: float = np.inf,
            ) -> generation_module.DistributedSurfaceConstruction:
                generated = original_generation(prepared, minimum_angle=minimum_angle,
                                                maximum_normal_angle=maximum_normal_angle)
                original = generated.construction
                points = original.vertices.copy()
                points[np.flatnonzero(original.vertex_source_dimensions == 2)[0], 0] += .125
                return replace(generated, construction=eqx.tree_at(
                    lambda value: value.vertices, original, points))
            generation_module.execute_distributed_surface_initial = corrupt_source
        try:
            plan = provider.plan(
                M.NativeSurfaceSource(domain), specification,
                coordinate_contract=phx.SpatialCoordinateContract(phx.units.METER),
                initial_partition=prepared)
            target = plan.execute()
        except M.MeshingFailure as error:
            assert scenario in ("remote_source_failure", "physical_overlap", "local_audit_failure")
            findings = dict(error.evidence.logical_findings)
            report = {"rejected": True, "failure_id": error.evidence.evidence_id}
            if scenario == "remote_source_failure":
                raw = findings["initial/failed_owner_category"]
                assert not raw.is_fully_addressable
                report["local_category"] = int(np.asarray(raw.addressable_shards[0].data)[0])
            elif scenario == "physical_overlap":
                raw = findings["initial/cross_owner_findings"]
                assert not raw.is_fully_addressable
                records = np.asarray(raw.addressable_shards[0].data)
                report["native_intersection_count"] = int(np.sum(records[:, 2] == 1))
            print(json.dumps(report), flush=True)
        else:
            if scenario == "storage_bank_forgery":
                from phydrax.meshing._result import CollectiveMeshStorageBinding
                from phydrax.discretization._cell_geometry import CellGeometryStorageProjection
                evidence = target.collective_evidence
                storage = target.mesh.storage
                copied = tuple((name, value.copy()) for name, value in storage.logical_arrays)
                restored_binding = CollectiveMeshStorageBinding(evidence, copied, storage.global_entity_counts)
                copied_projection = CellGeometryStorageProjection(
                    copied, storage.logical_coordinate_geometry_id, storage.global_coordinate_count,
                    source_elements=dict(storage.geometry_projection.source_elements))
                copied_storage = eqx.tree_at(
                    lambda value: (value.logical_arrays, value.geometry_projection),
                    storage, (copied, copied_projection))
                copied_mesh = eqx.tree_at(lambda value: value.storage, target.mesh, copied_storage)
                restored_binding.require_storage(copied_mesh, evidence)
                copied_geometry = copied_storage.restore_geometry()
                np.testing.assert_array_equal(copied_geometry.coordinates, target.geometry.coordinates)
                copied_geometry.resolve(copied_mesh)
                forged = dict(storage.logical_arrays)
                forged["geometry/coordinates"] = forged["geometry/coordinates"].at[0, 0].add(.125)
                forged_arrays = tuple(sorted(forged.items()))
                forged_projection = CellGeometryStorageProjection(
                    forged_arrays, storage.logical_coordinate_geometry_id, storage.global_coordinate_count,
                    source_elements=dict(storage.geometry_projection.source_elements))
                rejected = False
                try:
                    CollectiveMeshStorageBinding(evidence, forged_arrays, storage.global_entity_counts)
                except ValueError:
                    rejected = True
                assert rejected
                local_rejected = False
                if rank == 1:
                    assert target.mesh.entity_set(2).count == 0
                    forged_storage = eqx.tree_at(
                        lambda value: (value.logical_arrays, value.geometry_projection),
                        storage, (forged_arrays, forged_projection))
                    forged_mesh = eqx.tree_at(lambda value: value.storage, target.mesh, forged_storage)
                    forged_geometry = forged_storage.restore_geometry()
                    scopes = tuple((record.scope._scope_projection.membership_name, record.scope._scope_projection)
                        for record in (*target.patches, *target.zones, *target.labels))
                    try:
                        M.CellMeshingResult(
                            forged_mesh, forged_geometry, target.coordinate_contract, target.audit,
                            target.quality, target.compliance, target.trace, target.provider, target.runtime,
                            target.derivative_mode, target.provenance,
                            patches=target.patches, zones=target.zones, labels=target.labels,
                            collective_evidence=evidence, scope_projections=scopes,
                            storage_binding=target.storage_binding)
                    except ValueError:
                        local_rejected = True
                    assert local_rejected
                print(json.dumps({"forged_bank_rejected": rejected,
                    "passive_constructor_rejected": local_rejected, "independent_buffer_admitted": True}), flush=True)
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices("initial-storage-security")
                jax.distributed.shutdown()
                sys.exit(0)
            assert scenario.startswith("accepted_")
            assert isinstance(target, M.CellMeshingResult)
            assert target.provider.provider_id == provider.info().provider_id
            if scenario == "accepted_empty_owner":
                assert target.mesh.storage is not None
                empty = target.mesh.entity_set(2).count == 0
                if empty:
                    assert rank == 1
                    assert target.mesh.coordinates.shape == (0, 3)
                    assert target.mesh.blocks == ()
                    assert target.geometry.coordinates.shape == (0, 3)
                    assert all(entities.count == 0 for entities in target.mesh.topology.entity_sets)
                    assert all(record.scope.entity_ids.size == 0 for record in (*target.patches, *target.labels))
                    assert all(record.scope.global_entity_ids.size > 0 for record in (*target.patches, *target.labels))
                    target.audit.require_passed()
                    area = 0.
                else:
                    D = phx.discretization
                    fe = D.FiniteElementPlan(target.mesh, D.FiniteElementFieldSpec(
                        "u", D.lagrange_element("triangle", 1))).prepare()
                    area = float(jnp.sum(fe.block_geometries[0][0].measure))
                print(json.dumps({"empty": empty, "owned_area": area,
                    "result_id": target.result_id,
                    "global_cells": target.mesh.storage.global_entity_counts[2]}), flush=True)
                from jax.experimental import multihost_utils
                multihost_utils.sync_global_devices("initial-empty-consumer")
                jax.distributed.shutdown()
                sys.exit(0)
            assert target is not None
            evidence = require_original_meshing_source(target)
            assert isinstance(evidence, InitialCollectiveMeshEvidence)
            assert evidence.compiled.domain.domain_id == domain.domain_id
            assert evidence.specification.specification_id == specification.specification_id
            assert not any(name.startswith("epoch/") for name, _ in evidence.logical_arrays)
            from jax.experimental import multihost_utils
            assert len(target.patches) == len(domain.patches) + (scenario == "accepted_organization")
            assert len(target.labels) == len(domain.curves)
            organization = []
            locally_empty = 0
            for degree, records, association in (
                (2, tuple(patch for patch in target.patches if patch.scope.entity_dimension == 2), target.associations[0]),
                (1, target.labels, target.associations[1]),
            ):
                resident = np.asarray(target.mesh.entity_set(degree).entity_ids)
                owned_ids = resident[np.asarray(target.mesh.storage.entity_owned[degree])]
                for record in records:
                    source_index = int(record.name.split(":")[1])
                    source_id = domain.entity_id(degree, source_index)
                    selected = np.asarray(association.target_global_ids)[
                        np.asarray([source == source_id for source in association.source_entity_ids])]
                    np.testing.assert_array_equal(record.scope.entity_ids, np.sort(selected))
                    locally_empty += record.scope.entity_ids.size == 0
                    own_members = np.intersect1d(selected, owned_ids)
                    packet = np.full((layout.cell_capacity,), -1, dtype=np.int64)
                    packet[:own_members.size] = own_members
                    gathered = np.asarray(multihost_utils.process_allgather(packet, tiled=False))
                    reference_ids = np.unique(gathered[gathered >= 0])
                    assert record.scope.global_entity_ids.shape == reference_ids.shape
                    assert bool(jnp.all(record.scope.global_entity_ids == jnp.asarray(reference_ids)))
                    organization.append((record.name, record.scope.scope_id, reference_ids.size))
            zones = []
            if scenario == "accepted_organization":
                face_association = target.associations[0]
                for zone in target.zones:
                    index = 0 if zone.name == "left" else 1
                    selected = np.asarray(face_association.target_global_ids)[np.asarray(
                        [source == domain.entity_id(2, index) for source in face_association.source_entity_ids])]
                    np.testing.assert_array_equal(zone.scope.entity_ids, np.sort(selected))
                    zones.append((zone.name, zone.material_id, zone.region_role.value, zone.zone_id))
                interface = next(patch for patch in target.patches if patch.name == "shared-interface")
                assert set(interface.adjacent_zone_ids) == {zone.zone_id for zone in target.zones}
            D = phx.discretization
            fe = D.FiniteElementPlan(
                target.mesh, D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1))).prepare()
            ones = jnp.ones((fe.dof_maps[0].global_dof_count,), dtype=jnp.float64)
            np.testing.assert_allclose(fe.stiffness.mv(ones), 0., atol=1.e-11)
            points = fe.dof_maps[0].dof_coordinates
            expected = 1. + points[:, 0] + 2. * points[:, 1] - points[:, 2]
            solved = phx.linalg.solve(
                phx.linalg.LinearSystem(fe.mass), fe.mass.mv(expected),
                policy=phx.linalg.LinearSolvePolicy(
                    phx.linalg.PCG(), tolerance=phx.linalg.TolerancePolicy(
                        relative=1.e-12, absolute=1.e-14, max_steps=128)))
            assert bool(jnp.all(solved.successful))
            np.testing.assert_allclose(solved.value, expected, atol=1.e-9, rtol=1.e-9)
            area = jnp.sum(jnp.where(target.mesh.storage.entity_owned[2],
                                    fe.block_geometries[0][0].measure, 0.))
            print(json.dumps({"result_id": target.result_id, "owned_area": float(area),
                              "solve_error": float(jnp.max(jnp.abs(solved.value - expected))),
                              "organization": organization, "locally_empty_scopes": locally_empty,
                              "zones": zones}), flush=True)
        jax.distributed.shutdown()
        sys.exit(0)
    construction = generated.construction
    if construction is None:
        assert scenario == "empty_owner" and rank == 1
        assert generated.vertex_ids.size == 0 and generated.cell_ids.size == 0
        print(json.dumps({"empty": True}), flush=True)
    else:
        assert set(construction.triangle_patches.tolist()) == set(
            np.flatnonzero(owners == rank).tolist())
        corners = construction.vertices[construction.triangles]
        area = np.sum(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                             corners[:, 2] - corners[:, 0]), axis=1) / 2)
        assert not np.any(construction.unresolved)
        shared_rows = construction.curve_edges[construction.curve_edge_curves == 0]
        shared = np.unique(shared_rows)
        assert np.all(construction.vertices[shared, 0] == 0.)
        assert np.all(construction.vertices[shared, 2] == 0.)
        assert np.sum(np.linalg.norm(np.diff(
            construction.vertices[shared_rows], axis=1)[:, 0], axis=1)) == 1.
        if scenario == "shared_constraint":
            assert generated.neighbor_packet_count == 1
            assert generated.neighbor_transferred_bytes > 0
            assert generated.shared_curve_ids.tolist() == [0]
        print(json.dumps({
            "area": float(area), "local_cells": construction.triangles.shape[0],
            "global_cells": generated.global_cell_count,
            "shared_ids": generated.vertex_ids[shared].tolist(),
            "shared_points": construction.vertices[shared].tolist(),
        }), flush=True)
jax.distributed.shutdown()
"""

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import equinox as eqx
import jax
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
from phydrax.meshing import (
    adapt_device_tetra_metric,
    DeviceTetraMetricLayout,
    MeshingFailureCategory,
    prepare_device_tetra_metric,
)
from phydrax.meshing._device_generation import (
    DeviceGenerationLayout,
    DeviceGenerationStatus,
    evaluate_device_generation_candidates,
    execute_device_generation_round,
    prepare_device_generation,
    PreparedDeviceGeneration,
)
from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native exact meshcore is unavailable"
)
_POINTS = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64)
_CELL = np.asarray(((0, 1, 2, 3),), dtype=np.int32)

# Empirical `test_tetra_generation_preserves_coverage_regions_and_semantic_ids`: matched max |delta|=6.93889e-18; joint allowance fraction=0.00693889, margin=99.3061%.
GENERATED_TETRAHEDRON_VOLUME_ATOL = 1e-15
# Empirical `_assert_initial_consumer`: matched max |delta|=6.66134e-16; joint allowance fraction=6.66134e-05, margin=99.9933%.
INITIAL_STIFFNESS_CONSTANT_ACTION_ATOL = 1e-11
# Empirical `_assert_initial_consumer`: matched max |delta|=1.79524e-11; joint allowance fraction=0.0179524, margin=98.2048%.
INITIAL_AFFINE_SOLUTION_ATOL = 1e-9
# Empirical `_assert_initial_consumer`: matched max |delta|=1.79524e-11; joint allowance fraction=0.0179524, margin=98.2048%.
INITIAL_AFFINE_SOLUTION_RTOL = 1e-9
# Empirical `test_native_initial_publication_reaches_finite_element_affine_solve`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
INITIAL_OWNED_AREA_ATOL = 1e-12


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
        _volumes(update.prepared),
        np.full((4,), 1.0 / 24.0),
        rtol=0,
        atol=GENERATED_TETRAHEDRON_VOLUME_ATOL,
    )
    assert np.sum(_volumes(update.prepared)) == pytest.approx(1.0 / 6.0)
    mesh = update.prepared.mesh
    np.testing.assert_array_equal(np.asarray(mesh.vertex_ids)[:5], (11, 21, 31, 41, 42))
    active = np.asarray(mesh.cell_active)
    np.testing.assert_array_equal(np.asarray(mesh.cell_ids)[active], (502, 503, 504, 505))
    np.testing.assert_array_equal(
        np.asarray(update.prepared.cell_classes)[active], np.full((4,), 7)
    )
    # Four internal star faces pair; each source boundary face survives once.
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


def test_surface_reconnect_validation_refusal_is_work_limit_without_publication() -> None:
    # The topology reservation fits, but the remaining native allowance cannot pay
    # the reconnect's mandatory source validation (10 units for one triangle).
    points = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    roomy = DeviceGenerationLayout(
        vertex_capacity=8, cell_capacity=8, candidate_capacity=4, maximum_work_units=1000
    )
    source = prepare_device_generation(
        roomy,
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
    count = int(np.asarray(batch.count))
    assert count >= 1
    # One source row plus two created triangles per candidate are reserved.
    reserved = 1 + 2 * count
    tight = DeviceGenerationLayout(
        vertex_capacity=8,
        cell_capacity=8,
        candidate_capacity=4,
        maximum_work_units=reserved + 5,
    )
    source = eqx.tree_at(lambda prepared: prepared.layout, source, tight)
    chart_points = np.asarray(batch.charts)
    update = execute_device_generation_round(
        source,
        candidates=batch,
        surface_points=np.column_stack(
            (chart_points, np.zeros((chart_points.shape[0],), dtype=np.float64))
        ),
        surface_normals=np.tile(
            np.asarray((0, 0, 1), dtype=np.float64), (chart_points.shape[0], 1)
        ),
    )
    assert update.evidence.status is DeviceGenerationStatus.WORK_LIMIT
    assert update.prepared is source
    assert update.evidence.accepted_vertices == 0


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


@pytest.mark.parametrize(
    "dimension, ambient_dimension", ((2, 2), (2, 3)), ids=("planar", "embedded")
)
def test_tetra_metric_layout_refuses_nonphysical_tetrahedral_axes(
    dimension: int, ambient_dimension: int
) -> None:
    simplex = AdaptiveSimplexLayout(
        dimension,
        ambient_dimension,
        vertex_capacity=16,
        cell_capacity=32,
        protected_edge_capacity=1,
        maximum_closure_iterations=32,
        maximum_coarsening_passes=32,
    )
    with pytest.raises(ValueError, match="DeviceTetraMetricLayout.simplex"):
        DeviceTetraMetricLayout(simplex)


@pytest.mark.parametrize(
    "lower, upper",
    ((0.0, 2.0), (1.0, 1.0), (2.0, 1.0), (np.nan, 2.0), (1.0, np.inf)),
    ids=("zero", "equal", "reversed", "nan", "infinite"),
)
def test_tetra_metric_layout_refuses_invalid_hysteresis(
    lower: float, upper: float
) -> None:
    simplex = _metric_layout().simplex
    with pytest.raises(ValueError, match="DeviceTetraMetricLayout.thresholds"):
        DeviceTetraMetricLayout(simplex, lower_length=lower, upper_length=upper)


@pytest.mark.parametrize(
    "shape", ((3, 3), (15, 3, 3), (16, 2, 2)), ids=("rank", "capacity", "tensor_axes")
)
def test_tetra_metric_preparation_preserves_host_metadata_type_refusal(
    shape: tuple[int, ...],
) -> None:
    layout = _metric_layout()
    with pytest.raises(TypeError, match="prepare_device_tetra_metric.metric"):
        prepare_device_tetra_metric(
            layout, _adaptive(layout.simplex), np.ones(shape, dtype=np.float64)
        )


@pytest.mark.parametrize(
    "tensor",
    (
        np.diag(np.asarray((-1.0, 1.0, 1.0), dtype=np.float64)),
        np.asarray(((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64),
        np.diag(np.asarray((np.inf, 1.0, 1.0), dtype=np.float64)),
    ),
    ids=("indefinite", "nonsymmetric", "nonfinite"),
)
def test_tetra_metric_preparation_checks_inactive_vertex_metrics(
    tensor: np.ndarray,
) -> None:
    layout = _metric_layout()
    source = _adaptive(layout.simplex)
    metric = np.tile(np.eye(3, dtype=np.float64), (layout.simplex.vertex_capacity, 1, 1))
    metric[-1] = tensor
    assert not bool(np.asarray(source.mesh.vertex_active)[-1])
    with pytest.raises(ValueError, match="prepare_device_tetra_metric.metric"):
        prepare_device_tetra_metric(layout, source, metric)


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


# Target cells retain their constrained source-chart identity across owners;
# these scenarios exercise process-asymmetric publication.
def _run_initial_scenario(
    scenario: str, arguments: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def run(rank: int) -> dict[str, Any]:
        return run_on_forced_devices(
            "tests._support.distributed_meshing.device_generation",
            scenario,
            arguments,
            devices=1,
            env=gloo_rank_environment(coordinator, rank),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(run, range(2)))


def test_native_initial_serial_boundary_projection_retains_exact_arrays_and_ids() -> None:
    from tests._support.distributed_meshing.device_generation import (
        boundary_projection_record,
    )

    assert boundary_projection_record()["identical"] is True


def test_native_initial_foreign_curve_boundary_retains_exact_arrays_and_ids() -> None:
    reports = _run_initial_scenario("foreign_curve_boundary")
    assert [report["identical"] for report in reports] == [True, True]


@pytest.mark.parametrize(
    "scenario", ("shared_constraint", "empty_owner", "one_owner_overflow")
)
def test_native_distributed_initial_source_patch_construction(scenario: str) -> None:
    reports = _run_initial_scenario(scenario)
    if scenario == "one_owner_overflow":
        assert [report["rejected"] for report in reports] == [True, True]
        assert [report["category"] for report in reports] == [
            MeshingFailureCategory.RESOURCE_EXHAUSTED.value,
            MeshingFailureCategory.RESOURCE_EXHAUSTED.value,
        ]
    elif scenario == "empty_owner":
        assert reports[1]["empty"] is True
        assert reports[0]["area"] == pytest.approx(2.0)
        assert reports[0]["global_cells"] == reports[0]["local_cells"]
    else:
        assert [report["area"] for report in reports] == pytest.approx([1.0, 1.0])
        assert reports[0]["shared_ids"] == reports[1]["shared_ids"]
        assert reports[0]["shared_points"] == reports[1]["shared_points"]
    if scenario != "one_owner_overflow":
        for rank, report in enumerate(reports):
            if report["empty"]:
                assert scenario == "empty_owner" and rank == 1
                assert report["vertex_count"] == report["cell_count"] == 0
            else:
                assert report["triangle_patches"] == report["expected_patches"]
                assert not np.any(report["unresolved"])
                assert np.all(np.asarray(report["shared_x"]) == 0.0)
                assert np.all(np.asarray(report["shared_z"]) == 0.0)
                assert report["shared_length"] == 1.0
                if scenario == "shared_constraint":
                    assert report["neighbor_packet_count"] == 1
                    assert report["neighbor_transferred_bytes"] > 0
                    assert report["shared_curve_ids"] == [0]


def _report_number(report: dict[str, object], name: str) -> float:
    value = report[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"process_report.{name} must be numerical; got a Boolean or nonnumeric value"
        )
    return float(value)


def _report_zones(report: dict[str, object]) -> list[list[str]]:
    values = report["zones"]
    if not isinstance(values, list):
        raise TypeError(
            "process_report.zones must be a sequence; got a nonsequence value"
        )
    result = []
    for row in values:
        if not isinstance(row, list) or len(row) != 4:
            raise TypeError(
                "process_report.zones.row must contain four identity fields; got an incompatible row"
            )
        name, material, role, identity = row
        if not (
            isinstance(name, str)
            and isinstance(material, str)
            and isinstance(role, str)
            and isinstance(identity, str)
        ):
            raise TypeError(
                "process_report.zones.row must contain string identity fields; got a nonstring field"
            )
        result.append([name, material, role, identity])
    return result


def _assert_initial_consumer(report: dict[str, Any]) -> None:
    assert report["target_kind"]
    assert report["provider_id"] == report["expected_provider_id"]
    assert report["collective_source_kind"]
    assert report["domain_id"] == report["expected_domain_id"]
    assert report["specification_id"] == report["expected_specification_id"]
    assert not report["epoch_arrays"]
    assert report["patch_count"] == report["expected_patch_count"]
    assert report["label_count"] == report["expected_label_count"]
    for membership in report["memberships"]:
        np.testing.assert_array_equal(membership["local_ids"], membership["selected_ids"])
        assert membership["global_shape"] == membership["reference_shape"]
        assert membership["global_members_match"]
    for membership in report["zone_memberships"]:
        np.testing.assert_array_equal(membership["local_ids"], membership["selected_ids"])
    assert report["interface_ids"] == report["expected_interface_ids"]
    np.testing.assert_allclose(
        report["stiffness_ones"], 0.0, atol=INITIAL_STIFFNESS_CONSTANT_ACTION_ATOL
    )
    assert report["solve_successful"]
    np.testing.assert_allclose(
        report["solved"],
        report["expected_solution"],
        atol=INITIAL_AFFINE_SOLUTION_ATOL,
        rtol=INITIAL_AFFINE_SOLUTION_RTOL,
    )


def test_native_initial_publication_reaches_finite_element_affine_solve() -> None:
    reports = _run_initial_scenario("accepted_consumer")
    assert reports[0]["result_id"] == reports[1]["result_id"]
    assert [report["owned_area"] for report in reports] == pytest.approx(
        [1.0, 1.0], abs=INITIAL_OWNED_AREA_ATOL
    )
    assert all(
        _report_number(report, "solve_error") < INITIAL_AFFINE_SOLUTION_ATOL
        for report in reports
    )
    assert reports[0]["organization"] == reports[1]["organization"]
    assert all(_report_number(report, "locally_empty_scopes") > 0 for report in reports)
    for report in reports:
        _assert_initial_consumer(report)


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
        assert all(not report["category_fully_addressable"] for report in reports)
    else:
        assert _report_number(reports[0], "native_intersection_count") > 0
        assert reports[1]["native_intersection_count"] == 0
        assert all(not report["intersection_fully_addressable"] for report in reports)


def test_native_initial_empty_owner_publishes_actual_zero_resident_storage() -> None:
    reports = _run_initial_scenario("accepted_empty_owner")
    assert reports[1]["empty"] is True
    assert reports[0]["result_id"] == reports[1]["result_id"]
    assert reports[0]["global_cells"] == reports[1]["global_cells"]
    assert reports[0]["owned_area"] == pytest.approx(2.0, abs=INITIAL_OWNED_AREA_ATOL)
    for report in reports:
        assert report["target_kind"]
        assert report["provider_id"] == report["expected_provider_id"]
        assert report["storage_present"]
        if report["empty"]:
            assert report["rank"] == 1
            assert report["coordinate_shape"] == [0, 3]
            assert report["block_count"] == 0
            assert report["geometry_shape"] == [0, 3]
            assert all(count == 0 for count in report["entity_counts"])
            assert all(count == 0 for count in report["scope_local_counts"])
            assert all(count > 0 for count in report["scope_global_counts"])
            assert report["audit_passed"]


def test_native_initial_authored_material_and_interface_inventory_is_global() -> None:
    reports = _run_initial_scenario("accepted_organization")
    assert reports[0]["organization"] == reports[1]["organization"]
    assert reports[0]["zones"] == reports[1]["zones"]
    assert sorted(row[:3] for row in _report_zones(reports[0])) == [
        ["left", "material-left", "user"],
        ["right", "material-right", "user"],
    ]
    for report in reports:
        _assert_initial_consumer(report)


def test_native_initial_local_audit_rejection_does_not_strand_another_owner() -> None:
    reports = _run_initial_scenario("local_audit_failure")
    assert [report["rejected"] for report in reports] == [True, True]


@pytest.fixture(scope="module")
def storage_bank_reports() -> list[dict[str, Any]]:
    return _run_initial_scenario("storage_bank_forgery")


def test_native_initial_storage_binding_rejects_forged_coefficient_banks(
    storage_bank_reports: list[dict[str, Any]],
) -> None:
    assert [report["forged_bank_rejected"] for report in storage_bank_reports] == [
        True,
        True,
    ]


def test_native_initial_passive_constructor_rejects_forged_coefficient_banks(
    storage_bank_reports: list[dict[str, Any]],
) -> None:
    assert storage_bank_reports[1]["passive_constructor_rejected"] is True
    assert storage_bank_reports[1]["local_cell_count"] == 0


def test_native_initial_storage_admits_independent_identical_buffers(
    storage_bank_reports: list[dict[str, Any]],
) -> None:
    assert [report["independent_buffer_admitted"] for report in storage_bank_reports] == [
        True,
        True,
    ]
    for report in storage_bank_reports:
        np.testing.assert_array_equal(
            report["copied_coordinates"], report["target_coordinates"]
        )


def test_generation_layout_neighborhood_depth_changes_its_static_signature() -> None:
    controls = dict(
        vertex_capacity=8, cell_capacity=8, candidate_capacity=4, maximum_work_units=1000
    )
    default = DeviceGenerationLayout(**controls)
    deeper = DeviceGenerationLayout(**controls, neighborhood_depth=2)
    assert default.neighborhood_depth == 1
    assert deeper.neighborhood_depth == 2
    assert jax.tree_util.tree_structure(default) != jax.tree_util.tree_structure(deeper)


@pytest.mark.parametrize("depth", (0, -1))
def test_generation_layout_rejects_nonpositive_neighborhood_depth(depth: int) -> None:
    with pytest.raises(ValueError, match="neighborhood_depth"):
        DeviceGenerationLayout(
            vertex_capacity=8, cell_capacity=8, candidate_capacity=4,
            maximum_work_units=1000, neighborhood_depth=depth,
        )


@pytest.fixture(scope="module")
def live_publication_reports() -> list[dict[str, Any]]:
    return _run_initial_scenario("initial_graph_live_publication")


def test_native_two_rank_live_plc_consumes_actual_fully_proven_packed_object(
    live_publication_reports: list[dict[str, Any]],
) -> None:
    assert len(live_publication_reports) == 2
    for report in live_publication_reports:
        live = report["live_publication"]
        assert live["same_actual_packed_object"]
        assert live["constructor_content_scans"] == 1
        assert live["counts"] == dict(raw=1, packed=1, physical=1, plc=1, restore=0)
        assert report["bytewise_fields_history_rng_clock"]
        assert report["source_raw_unchanged"]
        assert report["moved_cells"] > 0
        assert report["published_cells"] == report["initial_cells"]
    assert live_publication_reports[0]["target_result"] == live_publication_reports[1]["target_result"]


@pytest.mark.parametrize(
    "corruption",
    (
        "foreign_evidence", "foreign_scope", "foreign_budget", "layout",
        "evidence_epoch", "storage_epoch", "stored_content_id",
        "storage_array_owner", "packed_array_owner", "packed_object",
        "partitioned_layout", "raw_array_owner", "missing_epoch_bank",
        "unissued_view", "direct_view_constructor", "arbitrary_caller_data",
        "readonly_source_bytes", "expired_request",
        "mesh_signature", "ambient_dimension", "current_constructor",
        "empty_owners", "empty_bindings", "subclass_view",
    ),
)
def test_native_two_rank_live_publication_refuses_rebound_or_unissued_inputs(
    live_publication_reports: list[dict[str, Any]], corruption: str
) -> None:
    assert all(
        report["live_publication"]["refusals"][corruption]
        for report in live_publication_reports
    )


@pytest.fixture(scope="module")
def initial_graph_migration_reports() -> list[dict[str, Any]]:
    return _run_initial_scenario("initial_graph_migration")


def test_native_initial_graph_migration_owns_genuine_initial_theorem(
    initial_graph_migration_reports: list[dict[str, Any]],
) -> None:
    reports = initial_graph_migration_reports
    for name in ("initial_evidence", "source_result", "target_result", "migration_record"):
        assert reports[0][name] == reports[1][name]
        assert reports[0][name]
    assert reports[0]["target_result"] != reports[0]["source_result"]
    assert all(report["initial_owned_cells"] > 0 for report in reports)
    assert reports[0]["initial_owned_cells"] != reports[1]["initial_owned_cells"]
    assert sum(report["initial_owned_cells"] for report in reports) == reports[0]["initial_cells"]
    owned_ids = [identifier for report in reports for identifier in report["owned_ids"]]
    assert len(owned_ids) == len(set(owned_ids)) == reports[0]["initial_cells"]
    assert all(report["owned_ids"] for report in reports)
    for report in reports:
        assert report["published_cells"] == report["initial_cells"]
        assert report["proposal_weights"] == [
            len(owner["owned_ids"]) for owner in reports
        ]
        assert report["imbalance"] <= 1.05
        assert report["moved_cells"] > 0
        assert report["source_raw_unchanged"]


def test_native_initial_graph_migration_preserves_actual_fields_rng_clock_and_history(
    initial_graph_migration_reports: list[dict[str, Any]],
) -> None:
    assert all(
        report["bytewise_fields_history_rng_clock"]
        for report in initial_graph_migration_reports
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "source_mutation", "raw_mutation", "lost_cell", "missing_owner",
        "graph_measurements", "physical_placement",
    ),
)
def test_native_initial_graph_migration_refuses_changed_source_or_lost_ownership(
    initial_graph_migration_reports: list[dict[str, Any]],
    corruption: str,
) -> None:
    assert all(
        report["refusals"][corruption] for report in initial_graph_migration_reports
    )


@pytest.mark.parametrize("corner_packets", (False, True), ids=("cells", "corners"))
def test_native_initial_graph_migration_extracts_reordered_compact_fields(
    corner_packets: bool,
) -> None:
    from tools.meshing_qualification.scenarios.distributed import (
        _distributed_graph_compact_values,
    )

    base = 2**54
    query = np.asarray((base + 3, base + 1, base + 2), dtype=np.int64)
    if corner_packets:
        ids = np.asarray(
            ((base + 2, base + 1, base + 3), (base + 3, base + 2, base + 1)),
            dtype=np.int64,
        )
    else:
        ids = np.asarray((base + 2, base + 3, base + 1), dtype=np.int64)
    values = np.stack(
        (np.where(ids == base + 1, -0.0, 1.25), (ids - base).astype(np.float64)),
        axis=-1,
    )
    expected = np.asarray(((1.25, 3.0), (-0.0, 1.0), (1.25, 2.0)))
    actual = _distributed_graph_compact_values(ids, values, query)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_native_initial_graph_migration_refuses_missing_compact_field_identity() -> None:
    from tools.meshing_qualification.scenarios.distributed import (
        _distributed_graph_compact_values,
    )

    with pytest.raises(ValueError, match="cover every published compact identity"):
        _distributed_graph_compact_values(
            np.asarray((11, 29), dtype=np.int64),
            np.asarray((3.0, 7.0)),
            np.asarray((11, 23, 29), dtype=np.int64),
        )


def test_native_initial_graph_migration_refuses_shared_corner_byte_conflict() -> None:
    from tools.meshing_qualification.scenarios.distributed import (
        _distributed_graph_compact_values,
    )

    with pytest.raises(ValueError, match="identical shared packet bytes"):
        _distributed_graph_compact_values(
            np.asarray(((11, 29, 31), (29, 37, 31)), dtype=np.int64),
            np.asarray(((3.0, -0.0, 7.0), (0.0, 9.0, 7.0))),
            np.asarray((11, 29, 31, 37), dtype=np.int64),
        )


@pytest.mark.parametrize("failed_owner", (0, 1))
def test_native_migration_record_collective_reconstruction_refuses_and_recovers(
    failed_owner: int,
) -> None:
    reports = _run_initial_scenario(
        "initial_graph_collective_record_validation", (str(failed_owner),)
    )
    expected = {
        "source", "workset", "state", "transport", "record",
        "source-optional", "state-optional",
    }
    for report in reports:
        assert set(report["collective_record_refusals"]) == expected
        assert all(report["collective_record_refusals"].values())
        assert report["bytewise_fields_history_rng_clock"]
        assert report["source_raw_unchanged"]
    assert reports[0]["migration_record"] == reports[1]["migration_record"]

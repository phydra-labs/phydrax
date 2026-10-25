"""White-box accounting and cleanup contracts of surface metric measurement."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import (
    BSplineSurfacePatch,
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.linalg._small_batched import prepare_exact_small_linear_actions
from phydrax.meshing import (
    _surface_generation as generation,
    _surface_metric as surface_metric,
    MeshingLimits,
    SizeControlStrength,
)
from phydrax.meshing._domain import compile_surface_domain
from phydrax.meshing.providers._native_curve import QueryLedger
from tests._support.coordinate_references import native_meshing_request
from tests._support.meshing_fixtures import surface_mesh, surface_specification
from tests._support.surface_generation_sources import accounting_spline_domain
from tests._support.surface_metric import (
    native_coordinate_resources,
    prepare_sphere_metric_source,
    sphere_native_scope,
    SphereMetricSource,
)


def _chart_state(
    domain: MeshingDomain,
    ledger: QueryLedger,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> generation._Patch:
    charts = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    with generation._source_bound_scope(ledger, 0, scope=scope, budget=budget):
        points = domain.evaluate(np.zeros((4,), dtype=np.int64), charts, budget=budget)
        normals, regular = domain.oriented_normals(
            np.zeros((4,), dtype=np.int64), charts, budget=budget
        )
    assert np.all(regular)
    return generation._Patch(
        0,
        charts,
        points,
        normals,
        np.arange(4, dtype=np.int64),
        np.full((4,), -1, dtype=np.int64),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        np.asarray(((True, False, True), (True, True, False)), dtype=np.bool_),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.column_stack(
            (
                np.zeros(4, dtype=np.float64),
                np.arange(4, dtype=np.int64),
                np.zeros(4, dtype=np.float64),
                np.ones(4, dtype=np.float64),
            )
        ),
        np.zeros((4,), dtype=np.float64),
        np.zeros((4,), dtype=np.bool_),
        np.full((4, 2), -1, dtype=np.int64),
        np.zeros((4, 2), dtype=np.int64),
    )


def test_source_metric_work_is_charged_once_and_cached_cells_reuse_bounds() -> None:
    domain = accounting_spline_domain()
    limits = MeshingLimits()
    compiled = compile_surface_domain(
        domain, surface_specification(domain, 0.4, limits=limits)
    )
    cache = generation._MetricsCache()
    with native_meshing_request(limits) as (scope, budget, _):
        ledger = QueryLedger(limits, budget=budget)
        state = _chart_state(domain, ledger, scope, budget)
        before_source, before_work = budget.work_units, ledger.work
        before_temporary = budget.temporary_bytes_upper
        first = generation._Metrics(
            domain, state, ledger, compiled, cache, scope=scope, budget=budget
        )
        source_work = budget.work_units - before_source
        assert source_work > 0
        assert ledger.work - before_work == source_work
        assert limits.maximum_work_units - ledger.work == (
            limits.maximum_work_units - before_work - source_work
        )
        assert budget.temporary_bytes_upper == before_temporary
        before_source, before_work = budget.work_units, ledger.work
        second = generation._Metrics(
            domain, state, ledger, compiled, cache, scope=scope, budget=budget
        )
        assert budget.work_units == before_source
        assert ledger.work == before_work
        np.testing.assert_array_equal(second.deviation_bound, first.deviation_bound)
        np.testing.assert_array_equal(second.normal_bound, first.normal_bound)
        assert budget.temporary_bytes_upper == before_temporary
        assert budget.ticket.settled
    assert budget.storage_ticket is None


def test_surface_generation_retains_complete_bounded_native_admission() -> None:
    domain = accounting_spline_domain()
    limits = MeshingLimits()
    result = surface_mesh(domain, surface_specification(domain, 0.35, limits=limits))
    assert result.certification is not None and result.certification.passed
    receipt = result.execution_evidence
    assert receipt is not None
    receipt.require_valid()
    peak = receipt.total_managed_admission_peak_bytes_upper
    assert peak is not None
    assert 0 < int(np.asarray(peak)) <= limits.maximum_scratch_bytes


def test_coarse_hard_equilateral_target_does_not_seed_undersized_edges() -> None:
    # Every authored edge has length sqrt(2). Its unique junction pair needs
    # one interval, and the target spans both chart axes, requiring no seeds.
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    loop = tuple(
        PatchCurveUse(row, LineCurve(head, uv[(row + 1) % 3] - head), 0.0, 1.0)
        for row, head in enumerate(uv)
    )
    patch = BSplineSurfacePatch(
        np.asarray(
            (
                ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
                ((1.0, 1.0, 0.0), (2.0, 1.0, 1.0)),
            ),
            dtype=np.float64,
        ),
        np.ones((2, 2), dtype=np.float64),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        1,
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(patch, (loop,)),),
        tuple(MeshingDomainCurve(row, (row + 1) % 3) for row in range(3)),
        3,
        source_id="coarse-hard-equilateral",
        source_revision="authored",
    )
    target = np.sqrt(2.0)
    request = surface_specification(domain, target, strength=SizeControlStrength.HARD)
    result = surface_mesh(domain, request)
    assert result.compliance.passed
    assert result.mesh.coordinates.shape == (3, 3)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    assert cells.shape == (1, 3)
    points = np.asarray(result.mesh.coordinates)[cells[0]]
    lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
    np.testing.assert_array_equal(lengths, np.full((3,), target, dtype=np.float64))
    achieved = dict(result.compliance.achieved)
    key = f"size:{request.size_controls[0].control_id}"
    assert achieved[f"{key}:p50_edge"] == target
    assert achieved[f"{key}:p95_edge"] == target
    assert result.execution_evidence is not None
    print(
        f"Coarse target mesh: vertices={result.mesh.coordinates.shape[0]}, "
        f"cells={cells.shape[0]}, "
        f"work={int(np.asarray(result.execution_evidence.total_work_units))}, "
        f"queries={int(np.asarray(result.execution_evidence.total_geometry_queries))}, "
        f"mesh_id={result.mesh.mesh_id}, request_id={request.specification_id}"
    )


def _unfiltered_sphere_location(
    background: surface_metric._SphereMetricBackground,
    patch: int,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, Fraction], ...]:
    """Reference the ordered cone traversal without a predicate filter."""
    right = tuple((Fraction(float(value)),) for value in direction)
    for row in np.flatnonzero(np.asarray(background.atlas.patches) == patch):
        with budget.temporary_scope():
            action = prepare_exact_small_linear_actions(
                background.matrices[row], right, budget=budget
            )
            assert action.actions is not None
            homogeneous = tuple(value[0] for value in action.actions)
            denominator = sum(homogeneous, Fraction(0))
            if denominator <= 0 or any(value < 0 for value in homogeneous):
                continue
            return tuple(
                sorted(
                    (int(background.vertex_ids[vertex]), value / denominator)
                    for vertex, value in zip(
                        background.corner_rows[row], homogeneous, strict=True
                    )
                    if value
                )
            )
    raise AssertionError(
        "sphere_location.reference must lie in an authored sphere cone; got an uncovered direction"
    )


def test_sphere_location_filter_keeps_exact_keys_and_reduces_charged_work() -> None:
    fixture = prepare_sphere_metric_source(1.1)
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (fixture.source.mesh.coordinates.shape[0], 3, 3)
    )
    with sphere_native_scope(fixture.policy.limits) as scope:
        with (
            native_coordinate_resources(scope, fixture.policy.limits) as budget,
            budget.activate(),
        ):
            _, background = surface_metric._sphere_initial_state(
                fixture.source,
                metric,
                fixture.atlas,
                fixture.transfer,
                fixture.policy,
                0.32,
                budget=budget,
            )
            rows = np.linspace(0, background.corner_rows.shape[0] - 1, 5, dtype=np.int64)
            reference_work = filtered_work = 0
            for row in rows:
                rays = np.asarray(fixture.atlas.directions)[row]
                # Include a shared vertex, shared edge, and an interior ray.
                for direction in (
                    rays[0],
                    np.mean(rays[:2], axis=0),
                    np.mean(rays, axis=0),
                ):
                    patch = int(np.asarray(fixture.atlas.patches)[row])
                    before = budget.work_units
                    expected = _unfiltered_sphere_location(
                        background, patch, direction, budget
                    )
                    reference_work += budget.work_units - before
                    before = budget.work_units
                    actual = surface_metric._sphere_location(
                        background, patch, direction, budget=budget
                    )
                    filtered_work += budget.work_units - before
                    assert actual == expected
                    if np.array_equal(direction, rays[0]):
                        identifier = int(
                            np.asarray(fixture.atlas.physical_corner_global_ids)[row, 0]
                        )
                        assert actual == ((identifier, Fraction(1)),)
            assert filtered_work < reference_work
            print(
                f"Sphere location charged work: {reference_work} -> {filtered_work}; "
                f"exact keys preserved: {3 * rows.size}"
            )


def test_repeated_sphere_proposal_reuses_earned_source_point_and_exact_key() -> None:
    fixture = prepare_sphere_metric_source(1.1)
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (fixture.source.mesh.coordinates.shape[0], 3, 3)
    )
    with sphere_native_scope(fixture.policy.limits) as scope:
        with (
            native_coordinate_resources(scope, fixture.policy.limits) as budget,
            budget.activate(),
        ):
            state, background = surface_metric._sphere_initial_state(
                fixture.source,
                metric,
                fixture.atlas,
                fixture.transfer,
                fixture.policy,
                0.32,
                budget=budget,
            )
            vertex = next(
                row for row, dimension in enumerate(state.dimensions) if dimension == 2
            )
            owner = np.flatnonzero(np.any(state.cells == vertex, axis=1))[0]
            patch = int(state.patches[owner])
            parameters = state.parameters[vertex]
            before_work = budget.work_units
            before_queries = budget.ticket.remaining_geometry_queries
            expected = surface_metric._sphere_point(
                fixture.domain,
                background,
                patch,
                state.dimensions[vertex],
                state.indices[vertex],
                parameters,
                budget=budget,
            )
            first_work = budget.work_units - before_work
            assert before_queries - budget.ticket.remaining_geometry_queries == 1
            before_work = budget.work_units
            before_queries = budget.ticket.remaining_geometry_queries
            actual = surface_metric._sphere_point(
                fixture.domain,
                background,
                patch,
                state.dimensions[vertex],
                state.indices[vertex],
                parameters.copy(),
                budget=budget,
            )
            repeated_work = budget.work_units - before_work
            assert before_queries == budget.ticket.remaining_geometry_queries
            for observed, reference in zip(actual[:3], expected[:3], strict=True):
                np.testing.assert_array_equal(observed, reference)
            assert actual[3] == expected[3]
            assert repeated_work < first_work
            print(
                f"Repeated sphere proposal work: {first_work} -> {repeated_work}; "
                "source queries: 1 -> 0"
            )


def test_sphere_local_edge_merit_matches_the_full_carrier() -> None:
    fixture = prepare_sphere_metric_source(1.1)
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (fixture.source.mesh.coordinates.shape[0], 3, 3)
    )
    with sphere_native_scope(fixture.policy.limits) as scope:
        with (
            native_coordinate_resources(scope, fixture.policy.limits) as budget,
            budget.activate(),
        ):
            state, _ = surface_metric._sphere_initial_state(
                fixture.source,
                metric,
                fixture.atlas,
                fixture.transfer,
                fixture.policy,
                0.32,
                budget=budget,
            )
            edges, lengths = surface_metric._sphere_lengths(state)
            for vertex in (0, state.points.shape[0] // 2, state.points.shape[0] - 1):
                local = np.any(edges == vertex, axis=1)
                selected, actual = surface_metric._sphere_lengths(
                    state, edges=edges[local]
                )
                np.testing.assert_array_equal(selected, edges[local])
                np.testing.assert_array_equal(actual, lengths[local])
                assert surface_metric._unit_defect(actual) == surface_metric._unit_defect(
                    lengths[local]
                )


def _nonimproving_sphere_vertex(
    fixture: SphereMetricSource,
    state: surface_metric._SphereMetricState,
    background: surface_metric._SphereMetricBackground,
    budget: CoordinateEnclosureBudget,
) -> int:
    """Select a real proposal rejected by the unchanged numerical merit."""
    edges, old_lengths = surface_metric._sphere_lengths(state)
    for vertex in range(state.points.shape[0]):
        if vertex in state.fixed or state.dimensions[vertex] != 2:
            continue
        rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
        if (
            not rows.size
            or np.unique(state.patches[rows]).size != 1
            or np.unique(state.classes[rows]).size != 1
            or any(vertex in edge for edge in state.features)
        ):
            continue
        patch = int(state.patches[rows[0]])
        neighbors = np.unique(state.cells[rows])
        neighbors = neighbors[neighbors != vertex]
        parameters = surface_metric._sphere_radial_parameters(
            np.mean(state.directions[neighbors], axis=0), fixture.domain, patch,
            budget=budget,
        )
        point, direction, metric, _ = surface_metric._sphere_point(
            fixture.domain,
            background,
            patch,
            2,
            state.indices[vertex],
            parameters,
            budget=budget,
        )
        old_quality = surface_metric._sphere_quality(
            state, state.cells[rows], state.patches[rows], background
        )
        original = (
            state.points[vertex].copy(),
            state.directions[vertex].copy(),
            state.metric[vertex].copy(),
        )
        state.points[vertex], state.directions[vertex], state.metric[vertex] = (
            point,
            direction,
            metric,
        )
        try:
            new_quality = surface_metric._sphere_quality(
                state, state.cells[rows], state.patches[rows], background
            )
            _, new_lengths = surface_metric._sphere_lengths(state)
            local = np.any(edges == vertex, axis=1)
            improves = np.min(new_quality) >= np.min(
                old_quality
            ) and surface_metric._unit_defect(
                new_lengths[local]
            ) < surface_metric._unit_defect(old_lengths[local])
        finally:
            state.points[vertex], state.directions[vertex], state.metric[vertex] = (
                original
            )
        if not improves:
            return vertex
    raise ValueError(
        "surface_metric.fixture.relocation must contain a nonimproving free proposal; "
        "got no eligible vertex."
    )


def test_nonimproving_sphere_relocation_preserves_carrier_with_less_proof_work() -> None:
    fixture = prepare_sphere_metric_source(1.1)
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64), (fixture.source.mesh.coordinates.shape[0], 3, 3)
    )
    with sphere_native_scope(fixture.policy.limits) as scope:
        with (
            native_coordinate_resources(scope, fixture.policy.limits) as budget,
            budget.activate(),
            budget.temporary_scope(),
        ):
            state, background = surface_metric._sphere_initial_state(
                fixture.source,
                metric,
                fixture.atlas,
                fixture.transfer,
                fixture.policy,
                0.32,
                budget=budget,
            )
            vertex = _nonimproving_sphere_vertex(fixture, state, background, budget)
            rows = np.flatnonzero(np.any(state.cells == vertex, axis=1))
            patch = int(state.patches[rows[0]])
            neighbors = np.unique(state.cells[rows])
            neighbors = neighbors[neighbors != vertex]
            parameters = surface_metric._sphere_radial_parameters(
                np.mean(state.directions[neighbors], axis=0), fixture.domain, patch,
                budget=budget,
            )
            before = budget.work_units
            point, direction, _, _ = surface_metric._sphere_point(
                fixture.domain,
                background,
                patch,
                2,
                state.indices[vertex],
                parameters,
                budget=budget,
            )
            budget.reserve(0, state.points.nbytes + state.directions.nbytes)
            points, directions = state.points.copy(), state.directions.copy()
            points[vertex], directions[vertex] = point, direction
            surface_metric._sphere_legal(
                fixture.domain,
                state.cells[rows],
                state.patches[rows],
                directions,
                points=points,
                background=background,
                maximum_fidelity=state.maximum_fidelity,
                budget=budget,
            )
            reference_work = budget.work_units - before
            current, fresh = surface_metric._sphere_initial_state(
                fixture.source,
                metric,
                fixture.atlas,
                fixture.transfer,
                fixture.policy,
                0.32,
                budget=budget,
            )
            original = (
                current.points.copy(),
                current.directions.copy(),
                current.metric.copy(),
            )
            before = budget.work_units
            changed = surface_metric._sphere_relocate(
                current, vertex, fixture.domain, fresh, fixture.transfer, budget=budget
            )
            actual_work = budget.work_units - before
            assert not changed
            for actual, expected in zip(
                (current.points, current.directions, current.metric),
                original,
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)
            assert actual_work < reference_work
            print(
                f"Rejected relocation work: {reference_work} -> {actual_work}; vertex={vertex}"
            )


def test_all_rejected_native_refinement_round_retains_carrier_and_refuses_hard_size() -> (
    None
):
    from time import perf_counter

    import pytest

    from phydrax.meshing import (
        MeshingFailure,
        MeshingFailureCategory,
        NativeSurfaceSchedule,
        SizeCompliancePolicy,
    )

    domain = accounting_spline_domain()
    limits = MeshingLimits()
    request = surface_specification(
        domain, 0.9, limits=limits, strength=SizeControlStrength.HARD
    )
    compiled = compile_surface_domain(domain, request)
    # This valid authored schedule requires more separation than the real
    # size-only proposals have. Native rejection must not fabricate progress
    # or publish a carrier that still violates the unchanged hard request.
    schedule = NativeSurfaceSchedule(spacing_fraction=0.99)
    policy = generation._RefinementPolicy(
        float(compiled.patch_sizes[0]),
        float(compiled.patch_deviations[0]),
        float(np.radians(schedule.quality_angle_degrees)),
        np.inf,
        schedule,
        0.0,
        SizeCompliancePolicy(),
    )
    source_id, source_revision, domain_id = (
        domain.source_id,
        domain.source_revision,
        domain.domain_id,
    )
    source_corners = np.asarray(domain.corner_points).copy()
    request_id = request.specification_id
    totals = generation.RefinementTotals()
    with native_meshing_request(limits) as (scope, budget, _):
        ledger = QueryLedger(limits, budget=budget)
        state = _chart_state(domain, ledger, scope, budget)
        before = {
            field: getattr(state, field).copy()
            for field in state.__slots__
            if field != "index"
        }
        work_before = ledger.work
        with pytest.raises(
            MeshingFailure, match="SurfaceConstruction.physical_criteria"
        ) as failed:
            generation._refine(
                domain,
                state,
                policy,
                limits,
                ledger,
                perf_counter(),
                totals,
                compiled,
                scope=scope,
                budget=budget,
            )
        failure = failed.value
        assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
        assert failure.evidence.provider_code == "surface_refinement_incomplete"
        requested = dict(failure.evidence.requested)
        achieved = dict(failure.evidence.achieved)
        assert requested["maximum_rounds"] == schedule.maximum_rounds == 64
        assert totals.rounds == achieved["rounds"] == 1
        assert totals.inserted == 0
        assert totals.refused == 2
        assert achieved["unresolved_triangles"] == achieved["oversized_triangles"] == 2
        assert achieved["invalid_triangles"] == achieved["misoriented_triangles"] == 0
        assert achieved["work_units"] == ledger.work > work_before
        assert ledger.work < limits.maximum_work_units
        assert achieved["geometry_queries"] == ledger.queries
        assert ledger.queries < limits.maximum_geometry_queries
        assert state.index == 0
        for field, original in before.items():
            if field not in ("triangles", "constrained"):
                np.testing.assert_array_equal(getattr(state, field), original)
        # Reconnection's real final Delaunay sweep may rotate/flip cells even
        # when every insertion rejects. It must preserve the authored region
        # and constrained scientific edges, not the old storage-slot order.
        assert state.triangles.shape == before["triangles"].shape
        assert state.constrained.shape == before["constrained"].shape
        boundary = {
            tuple(
                sorted(
                    (
                        int(triangle[(corner + 1) % 3]),
                        int(triangle[(corner + 2) % 3]),
                    )
                )
            )
            for triangle, flags in zip(state.triangles, state.constrained, strict=True)
            for corner, constrained in enumerate(flags)
            if constrained
        }
        assert boundary == {
            tuple(sorted(map(int, edge))) for edge in before["boundary_edges"]
        }
        charts = state.charts[state.triangles]
        first, second = charts[:, 1] - charts[:, 0], charts[:, 2] - charts[:, 0]
        oriented_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        assert np.all(oriented_area > 0.0)
        assert np.sum(oriented_area) == 2.0
        # With the four unchanged authored corner rows, these are precisely
        # the two nonoverlapping triangulations covering the unit square.
        cells = {tuple(sorted(map(int, triangle))) for triangle in state.triangles}
        assert cells in (
            {(0, 1, 2), (0, 2, 3)},
            {(0, 1, 3), (1, 2, 3)},
        )
    assert (domain.source_id, domain.source_revision, domain.domain_id) == (
        source_id,
        source_revision,
        domain_id,
    )
    np.testing.assert_array_equal(domain.corner_points, source_corners)
    assert request.specification_id == request_id



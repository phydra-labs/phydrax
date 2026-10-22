from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields
from fractions import Fraction
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import MeshcoreError, MeshcoreStatus
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellMesh
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep._patches import SpherePatch, surface_differential, TorusPatch
from phydrax.meshing import _surface_metric as owner
from phydrax.meshing.qualification import (
    AnalyticSurface,
    capped_cylinder_surface,
    sphere_surface,
    torus_surface,
)
from tests._support.surface_metric import surface_resources
from tools.compile_counter import CompileCounter


def metric_grid(
    surface: AnalyticSurface, resolution: int, patch: int = 0,
) -> owner.SurfaceState:
    axis = 0.2 * np.linspace(0.0, 1.0, resolution + 1, dtype=np.float64) ** 1.4
    u, v = np.meshgrid(axis, axis, indexing="ij")
    charts = np.column_stack((u.ravel(), v.ravel()))
    cells = np.asarray(
        [triangle for i in range(resolution) for j in range(resolution)
         for a in (i * (resolution + 1) + j,)
         for triangle in ((a, a + resolution + 1, a + resolution + 2),
                          (a, a + resolution + 2, a + 1))],
        dtype=np.int32,
    )
    if surface.domain.patches[patch].orientation < 0:
        cells = cells[:, (0, 2, 1)]
    points = surface.domain.evaluate(np.full(charts.shape[0], patch, dtype=np.int32), charts)
    metric = np.broadcast_to(np.diag((20.0, 25.0, 30.0)), (points.shape[0], 3, 3)).copy()
    metric *= (1.0 + charts[:, 0, None, None])
    cell_ids = np.arange(10, 10 + cells.shape[0], dtype=np.int64)
    vertex_ids = np.arange(100, 100 + points.shape[0], dtype=np.int64)
    return owner.SurfaceState(
        points, metric, cells, np.full(cells.shape[0], patch, dtype=np.int32), charts[cells],
        np.zeros(cells.shape[0], dtype=np.int64), cell_ids, vertex_ids,
        [{int(value)} for value in cell_ids], [{int(value)} for value in vertex_ids],
        int(cell_ids[-1]) + 1, int(vertex_ids[-1]) + 1,
    )


def full_measure_lengths(
    lengths: Callable[..., tuple[np.ndarray, np.ndarray]],
    domain: MeshingDomain,
    state: owner.SurfaceState,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-cutover edge-only caller also evaluated unused quality/fidelity."""
    edges, result = lengths(domain, state, budget=budget)
    owner._mapped_quality(
        domain, state.metric, state.cells, state.patches, state.charts, budget=budget
    )
    for patch in np.unique(state.patches).tolist():
        domain.interpolation_bounds(
            patch, state.charts[state.patches == patch], budget=budget,
            reserve_queries=lambda count: owner._consume_native_work(
                geometry_queries=count, budget=budget
            ),
        )
    return edges, result


def serial_relocation_measures(
    domain: MeshingDomain,
    local: owner.SurfaceState,
    trial: owner.SurfaceState,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float, bool]:
    before = float(np.min(owner._mapped_quality(
        domain, local.metric, local.cells, local.patches, local.charts, budget=budget
    )))
    after = float(np.min(owner._mapped_quality(
        domain, trial.metric, trial.cells, trial.patches, trial.charts, budget=budget
    )))
    improved = (
        owner._unit_defect(owner._surface_lengths(domain, trial, budget=budget)[1])
        < owner._unit_defect(owner._surface_lengths(domain, local, budget=budget)[1])
        - 1.0e-12
    )
    return before, after, improved


def derivative_batched_relocation_measures(
    domain: MeshingDomain,
    local: owner.SurfaceState,
    trial: owner.SurfaceState,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float, bool]:
    """Prior candidate-local derivative batch, with four metric kernel calls."""
    edges, first, _ = owner._edge_table(local)
    cells = local.cells.shape[0]
    edge_samples = 8 * edges.shape[0]
    with budget.temporary_scope():
        budget.reserve(0, 256 * (2 * cells + 2 * edge_samples))
        velocity, charts, patches = owner._surface_edge_coordinates(local, edges, first)
        trial_velocity, trial_charts, _ = owner._surface_edge_coordinates(trial, edges, first)
        differential = owner._surface_differentials(
            domain,
            np.concatenate((local.patches, trial.patches, np.tile(np.repeat(patches, 8), 2))),
            np.concatenate((np.mean(local.charts, axis=1), np.mean(trial.charts, axis=1),
                            charts.reshape((-1, 2)), trial_charts.reshape((-1, 2)))),
            budget=budget,
        )
        before = float(np.min(owner._quality_from_differential(
            local.metric, local.cells, local.charts, differential[:cells],
        )))
        after = float(np.min(owner._quality_from_differential(
            trial.metric, trial.cells, trial.charts, differential[cells:2 * cells],
        )))
        local_derivative = differential[2 * cells:2 * cells + edge_samples].reshape((-1, 8, 3, 2))
        trial_derivative = differential[2 * cells + edge_samples:].reshape((-1, 8, 3, 2))
        improved = (
            owner._unit_defect(owner._lengths_from_differential(
                trial, edges, trial_velocity, trial_derivative,
            ))
            < owner._unit_defect(owner._lengths_from_differential(
                local, edges, velocity, local_derivative,
            )) - 1.0e-12
        )
        budget.ticket.checkpoint()
        return before, after, improved


def ordinary_surface_differential(patch: Any, parameters: jax.Array) -> jax.Array:
    """Independent eager AD reference; never dispatch through the candidate."""
    points = jnp.asarray(parameters, dtype=jnp.float64)
    return jax.vmap(jax.jacfwd(patch.evaluate))(points.reshape((-1, 2))).reshape(
        (*points.shape[:-1], 3, 2)
    )


def assert_same_float_bits(actual: Any, expected: Any) -> None:
    left, right = np.asarray(actual), np.asarray(expected)
    assert left.dtype == right.dtype == np.float64
    np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))


@pytest.mark.parametrize("builder", (sphere_surface, torus_surface))
def test_metric_derivative_program_keeps_geometry_and_points_dynamic(
    builder: Callable[[], AnalyticSurface],
) -> None:
    surface = builder().domain.patches[0].surface
    points = jnp.asarray(((0.03, 0.07), (0.11, 0.19)), dtype=jnp.float64)
    changed_points = points + 0.01
    if isinstance(surface, SpherePatch):
        changed_surface = eqx.tree_at(lambda value: value.radius, surface, surface.radius * 1.1)
    else:
        assert isinstance(surface, TorusPatch)
        changed_surface = eqx.tree_at(
            lambda value: value.major_radius, surface, surface.major_radius * 1.1
        )
    first = owner._metric_surface_differential(surface, points)
    jax.block_until_ready(first)
    with CompileCounter() as compiler:
        repeated = owner._metric_surface_differential(surface, points)
        moved = owner._metric_surface_differential(surface, changed_points)
        translated = owner._metric_surface_differential(changed_surface, points)
        jax.block_until_ready((repeated, moved, translated))
    assert compiler.count == 0
    np.testing.assert_array_equal(repeated, first)
    for actual, patch, parameters in (
        (first, surface, points), (moved, surface, changed_points),
        (translated, changed_surface, points),
    ):
        assert_same_float_bits(actual, ordinary_surface_differential(patch, parameters))
    assert not np.array_equal(first, moved)
    assert not np.array_equal(first, translated)


@pytest.mark.parametrize("resolution", (2, 4, 8))
def test_surface_edge_work_scales_with_edges_without_repeating_cell_proofs(
    resolution: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = torus_surface()
    state = metric_grid(surface, resolution)
    with surface_resources() as (_, budget):
        before = budget.ticket.remaining_geometry_queries
        expected = owner._measure(surface.domain, state, budget=budget)[0]
        full_queries = before - budget.ticket.remaining_geometry_queries
        before = budget.ticket.remaining_geometry_queries
        edges, actual = owner._surface_lengths(surface.domain, state, budget=budget)
        edge_queries = before - budget.ticket.remaining_geometry_queries
    np.testing.assert_array_equal(actual, expected)
    assert edge_queries == 8 * len(edges)
    assert full_queries > edge_queries

    def forbid_proof(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("edge-only numerics must not repeat cell verdicts")

    monkeypatch.setattr(owner, "_mapped_quality", forbid_proof)
    monkeypatch.setattr(MeshingDomain, "interpolation_bounds", forbid_proof)
    state.metric *= 1.21
    with surface_resources() as (_, budget):
        _, changed = owner._surface_lengths(surface.domain, state, budget=budget)
    np.testing.assert_allclose(changed, actual * 1.1, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize(
    ("builder", "patch_index"),
    ((sphere_surface, 0), (torus_surface, 0), (capped_cylinder_surface, 0),
     (capped_cylinder_surface, 1), (capped_cylinder_surface, 2)),
)
def test_surface_relocation_preserves_candidates_order_and_current_state(
    builder: Callable[[], AnalyticSurface], patch_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = builder()
    initial = metric_grid(surface, 4, patch_index)
    boundary = np.zeros((5, 5), dtype=np.bool_)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    fixed = boundary.ravel()
    source = owner.SurfaceSource(
        initial.cells, initial.charts, initial.patches, initial.cell_ids,
        initial.vertex_ids, initial.metric, initial.charts,
    )
    edges_only = owner._surface_lengths
    legal = owner._legal
    observations: list[list[tuple[np.ndarray, np.ndarray, bool]]] = [[], []]
    routes: list[Counter[str]] = [Counter(), Counter()]
    measures = owner._surface_relocation_measures
    states, counts, queries = [], [], []
    for mode in range(2):
        state = deepcopy(initial)

        def observe_legal(*args: Any, **kwargs: Any) -> bool:
            routes[mode]["legal"] += 1
            verdict = legal(*args, **kwargs)
            observations[mode].append((args[1].copy(), args[3].copy(), verdict))
            return verdict

        def observe_measures(*args: Any, **kwargs: Any) -> Any:
            routes[mode]["candidate_measures"] += 1
            return measures(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_legal", observe_legal)
            patch.setattr(owner, "_surface_relocation_measures", observe_measures)
            if mode == 0:
                patch.setattr(owner, "_metric_surface_differential", ordinary_surface_differential)
                patch.setattr(owner, "_surface_relocation_measures", serial_relocation_measures)

                def legacy_lengths(*args: Any, **kwargs: Any) -> Any:
                    return full_measure_lengths(edges_only, *args, **kwargs)

                patch.setattr(owner, "_surface_lengths", legacy_lengths)
            with surface_resources() as (_, budget):
                before = budget.ticket.remaining_geometry_queries
                counts.append(owner.relocate(
                    surface.domain, state, fixed, set(), 0.1, state.points.shape[0],
                    100_000, source, {}, set(), 0.05, budget=budget,
                ))
                queries.append(before - budget.ticket.remaining_geometry_queries)
        states.append(state)
    assert counts[0] == counts[1]
    assert counts[1][0] > 0
    assert len(observations[0]) == len(observations[1])
    assert routes[1]["legal"] > 0
    assert routes[1]["candidate_measures"] > 0
    for reference, actual in zip(observations[0], observations[1], strict=True):
        np.testing.assert_array_equal(actual[0], reference[0])
        assert_same_float_bits(actual[1], reference[1])
        assert actual[2] == reference[2]
    for name in ("points", "metric", "charts"):
        assert_same_float_bits(getattr(states[1], name), getattr(states[0], name))
    for name in ("cells", "cell_ids", "vertex_ids"):
        np.testing.assert_array_equal(getattr(states[1], name), getattr(states[0], name))
    assert states[1].vertex_sources == states[0].vertex_sources
    assert states[1].cell_kinds == states[0].cell_kinds
    assert states[1].relocated_vertices == states[0].relocated_vertices
    assert queries[1] < queries[0]
    np.testing.assert_array_equal(initial.points, metric_grid(surface, 4, patch_index).points)


def test_surface_lengths_checks_original_deadline_after_numerical_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = sphere_surface()
    state = metric_grid(surface, 2)
    derivative = owner._metric_surface_differential
    evaluated = False
    with surface_resources() as (_, budget):
        ticket = budget.ticket
        checkpoint = type(ticket).checkpoint

        def evaluate(*args: Any, **kwargs: Any) -> Any:
            nonlocal evaluated
            result = derivative(*args, **kwargs)
            evaluated = True
            return result

        def expire(current: Any) -> None:
            if current is ticket and evaluated:
                raise MeshcoreError(MeshcoreStatus.TIMEOUT, "test original deadline")
            checkpoint(current)

        monkeypatch.setattr(owner, "_metric_surface_differential", evaluate)
        with monkeypatch.context() as patch:
            patch.setattr(type(ticket), "checkpoint", expire)
            with pytest.raises(MeshcoreError, match="original deadline"):
                owner._surface_lengths(surface.domain, state, budget=budget)
    np.testing.assert_array_equal(state.points, metric_grid(surface, 2).points)


def test_surface_numerical_bank_is_admitted_before_execution_and_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = sphere_surface()
    state = metric_grid(surface, 2)
    small = CoordinateEnclosureBudget(
        HostBatchTicket(100_000, queries_upper=100_000, bytes_upper=1024),
        maximum_work_units=100_000, maximum_memory_bytes=1024,
    )

    def forbid_evaluation(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the derivative bank must be admitted before evaluation")

    with monkeypatch.context() as patch:
        patch.setattr(owner, "_metric_surface_differential", forbid_evaluation)
        with pytest.raises(CoordinateEnclosureResourceError):
            owner._surface_lengths(surface.domain, state, budget=small)
    budget = CoordinateEnclosureBudget(
        HostBatchTicket(100_000, queries_upper=100_000, bytes_upper=65_536),
        maximum_work_units=100_000, maximum_memory_bytes=65_536,
    )
    edges, first = owner._surface_lengths(surface.domain, state, budget=budget)
    peak = budget.peak_bytes_upper
    assert peak == 80 * 8 * edges.shape[0]
    assert budget.temporary_bytes_upper == 0
    _, repeated = owner._surface_lengths(surface.domain, state, budget=budget)
    assert budget.peak_bytes_upper == peak
    assert budget.temporary_bytes_upper == 0
    assert_same_float_bits(repeated, first)


@pytest.mark.parametrize("reference_kind", ("derivative", "material-routing"))
def test_surface_metric_preserves_operation_order_and_remeshing_evidence(
    monkeypatch: pytest.MonkeyPatch, reference_kind: str,
) -> None:
    domain = torus_surface().domain
    counts = (8, 4)
    parameters = np.asarray(
        [(2 * np.pi * i / counts[0], 2 * np.pi * j / counts[1])
         for i in range(counts[0]) for j in range(counts[1])],
        dtype=np.float64,
    )
    cells, charts = [], []
    for i in range(counts[0]):
        for j in range(counts[1]):
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            for local in ((0, 1, 2), (0, 2, 3)):
                cells.append([
                    (corners[k][0] % counts[0]) * counts[1] + corners[k][1] % counts[1]
                    for k in local
                ])
                charts.append([
                    (2 * np.pi * corners[k][0] / counts[0],
                     2 * np.pi * corners[k][1] / counts[1]) for k in local
                ])
    points = domain.evaluate(np.zeros(parameters.shape[0], dtype=np.int32), parameters)
    mesh = CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))
    patches = np.zeros(len(cells), dtype=np.int32)
    cell_charts = np.asarray(charts, dtype=np.float64)
    fixed = np.any(parameters == 0.0, axis=1)
    metric = np.broadcast_to(
        np.eye(3, dtype=np.float64) / 0.9**2, (mesh.coordinates.shape[0], 3, 3)
    )
    lengths = owner._surface_lengths
    operation = owner._stage_surface_metric_operation
    material_stencil = owner._material_chart_stencil
    orders: list[list[tuple[str, tuple[int, ...]]]] = [[], []]
    outcomes = []
    for mode in range(2):

        def observe_operation(*args: Any, **kwargs: Any) -> Any:
            orders[mode].append((args[4], args[5]))
            return operation(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(owner, "_stage_surface_metric_operation", observe_operation)
            if mode == 0 and reference_kind == "material-routing":
                def fresh_material(*args: Any, **kwargs: Any) -> Any:
                    kwargs["cache"] = None
                    return material_stencil(*args, **kwargs)

                patch.setattr(owner, "_material_chart_stencil", fresh_material)
            elif mode == 0:
                patch.setattr(owner, "_metric_surface_differential", ordinary_surface_differential)
                patch.setattr(owner, "_surface_relocation_measures", serial_relocation_measures)

                def legacy_lengths(*args: Any, **kwargs: Any) -> Any:
                    return full_measure_lengths(lengths, *args, **kwargs)

                patch.setattr(owner, "_surface_lengths", legacy_lengths)
            with surface_resources() as (scope, budget):
                outcomes.append(owner.execute_surface_metric_adaptation(
                    mesh, metric, domain,
                    source_id=domain.source_id,
                    source_revision=domain.source_revision,
                    cell_patches=patches, cell_charts=cell_charts,
                    fixed_vertices=fixed,
                    maximum_fidelity=1.0, maximum_passes=1,
                    budget=budget, scope=scope,
                ))
    assert orders[0] == orders[1]
    assert orders[1]
    reference, actual = outcomes
    for name in (
        "status", "passes", "splits", "collapses", "flips", "relocations",
        "rejected_operations", "work_units", "criterion", "metric_quality_floor",
        "resource_message", "resource_requested", "resource_achieved",
    ):
        assert getattr(actual.evidence, name) == getattr(reference.evidence, name)
    assert tree_content_equal(actual.evidence, reference.evidence)
    assert_same_float_bits(actual.metric, reference.metric)
    assert_same_float_bits(
        actual.geometry.edit.coordinates, reference.geometry.edit.coordinates
    )
    np.testing.assert_array_equal(
        actual.geometry.edit.vertex_global_ids, reference.geometry.edit.vertex_global_ids
    )
    np.testing.assert_array_equal(
        actual.geometry.cell_patches, reference.geometry.cell_patches
    )
    assert_same_float_bits(actual.geometry.cell_charts, reference.geometry.cell_charts)
    np.testing.assert_array_equal(
        actual.geometry.cell_material_charts, reference.geometry.cell_material_charts
    )


@pytest.mark.parametrize("patch_index", (0, 1, 2))
@pytest.mark.parametrize("resolution", (2, 4, 8))
def test_capped_candidate_metric_lanes_preserve_bits_and_resources(
    patch_index: int, resolution: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = capped_cylinder_surface()
    local = metric_grid(surface, resolution, patch_index)
    trial = deepcopy(local)
    trial.charts[:, 1, 0] += 0.003
    trial.metric *= 1.01
    interpolation = owner._dependency_phydrax_meshing__metric.interpolate_mesh_metric
    calls: Counter[str] = Counter()

    def observe(*args: Any, **kwargs: Any) -> Any:
        calls["metric"] += 1
        return interpolation(*args, **kwargs)

    with surface_resources() as (_, budget):
        start = budget.ticket.remaining_geometry_queries
        expected = derivative_batched_relocation_measures(
            surface.domain, local, trial, budget=budget,
        )
        expected_queries = start - budget.ticket.remaining_geometry_queries
    with monkeypatch.context() as patch, surface_resources() as (_, budget):
        patch.setattr(owner._dependency_phydrax_meshing__metric, "interpolate_mesh_metric", observe)
        start = budget.ticket.remaining_geometry_queries
        actual = owner._surface_relocation_measures(surface.domain, local, trial, budget=budget)
        assert start - budget.ticket.remaining_geometry_queries == expected_queries
        assert budget.temporary_bytes_upper == 0
    assert_same_float_bits(actual[:2], expected[:2])
    assert actual[2] == expected[2]
    assert calls["metric"] == 4


@pytest.mark.parametrize("patch_index", (0, 1, 2))
def test_capped_relocation_structured_derivative_preserves_current_candidate_execution(
    patch_index: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace only the differential, not relocation, metric arithmetic or lanes."""
    assert owner._metric_surface_differential is surface_differential
    surface = capped_cylinder_surface()
    initial = metric_grid(surface, 4, patch_index)
    pristine = deepcopy(initial)
    boundary = np.zeros((5, 5), dtype=np.bool_)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    source = owner.SurfaceSource(
        initial.cells, initial.charts, initial.patches, initial.cell_ids,
        initial.vertex_ids, initial.metric, initial.charts,
    )
    legal = owner._legal
    measures = owner._surface_relocation_measures
    orders: list[list[str]] = [[], []]
    legal_observations: list[list[tuple[np.ndarray, np.ndarray, bool]]] = [[], []]
    measure_observations: list[list[tuple[float, float, bool]]] = [[], []]
    states, counts, resources = [], [], []
    for mode in range(2):
        state = deepcopy(initial)

        def observe_legal(*args: Any, **kwargs: Any) -> bool:
            orders[mode].append("legal")
            verdict = legal(*args, **kwargs)
            legal_observations[mode].append((args[1].copy(), args[3].copy(), verdict))
            return verdict

        def observe_measures(*args: Any, **kwargs: Any) -> tuple[float, float, bool]:
            orders[mode].append("measures")
            result = measures(*args, **kwargs)
            measure_observations[mode].append(result)
            return result

        with monkeypatch.context() as patch, surface_resources() as (_, budget):
            initial_queries = budget.ticket.remaining_geometry_queries
            patch.setattr(owner, "_legal", observe_legal)
            patch.setattr(owner, "_surface_relocation_measures", observe_measures)
            if mode == 0:
                patch.setattr(owner, "_metric_surface_differential", ordinary_surface_differential)
            counts.append(owner.relocate(
                surface.domain, state, boundary.ravel(), set(), 0.1,
                state.points.shape[0], 100_000, source, {}, set(), 0.05, budget=budget,
            ))
            resources.append((
                budget.work_units, budget.native_charged_work_units,
                initial_queries - budget.ticket.remaining_geometry_queries,
                budget.peak_bytes_upper,
                budget.temporary_bytes_upper,
                budget.ticket.remaining_work_units,
                budget.ticket.remaining_geometry_queries,
            ))
            assert budget.temporary_bytes_upper == 0
        states.append(state)

    assert counts[0] == counts[1]
    assert counts[1][0] > 0
    assert orders[0] == orders[1]
    assert "legal" in orders[1] and "measures" in orders[1]
    assert resources[0] == resources[1]
    assert resources[1][2] > 0
    for reference, actual in zip(legal_observations[0], legal_observations[1], strict=True):
        np.testing.assert_array_equal(actual[0], reference[0])
        assert_same_float_bits(actual[1], reference[1])
        assert actual[2] == reference[2]
    for reference, actual in zip(measure_observations[0], measure_observations[1], strict=True):
        assert_same_float_bits(actual[:2], reference[:2])
        assert actual[2] == reference[2]
    for actual, reference in ((states[1], states[0]), (initial, pristine)):
        for field in fields(owner.SurfaceState):
            left, right = getattr(actual, field.name), getattr(reference, field.name)
            if isinstance(left, np.ndarray):
                if left.dtype == np.float64:
                    assert_same_float_bits(left, right)
                else:
                    np.testing.assert_array_equal(left, right)
            else:
                assert left == right, field.name


def material_route_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    triangle = np.asarray(
        ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)),
         (Fraction(0), Fraction(1))), dtype=object,
    )
    return (
        np.asarray(((0, 1, 2),) * 3, dtype=np.int32),
        np.asarray((triangle, triangle, triangle), dtype=object),
        np.asarray((0, 0, 1), dtype=np.int32),
        np.asarray((20, 10, 30), dtype=np.int64),
        np.asarray((100, 101, 102), dtype=np.int64),
        np.asarray((0,), dtype=np.int32),
        np.asarray(((Fraction(1, 4), Fraction(1, 3)),), dtype=object),
    )


def assert_same_stencil(
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    np.testing.assert_array_equal(actual[0], expected[0])
    assert_same_float_bits(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize(
    "mutation", ("uv", "patch", "charts", "cells", "patches", "cell_ids", "vertex_ids", "frame", "role"),
)
def test_material_routing_authenticates_exact_owners_and_immutable_values(
    mutation: str,
) -> None:
    arrays = material_route_fixture()
    with surface_resources() as (_, budget):
        with owner.prepared_material_routing(("source", "frame", "material"), budget=budget) as bank:
            expected = owner.chart_stencil(*arrays, 10_000, budget=budget)
            actual = owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
            assert_same_stencil(actual, expected)
            # Outputs never alias the immutable numeric bank.
            actual[0][:] = -1
            actual[1][:] = -1
            start, storage = budget.work_units, bank.storage.bound
            replay = owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
            assert_same_stencil(replay, expected)
            assert budget.work_units == start
            assert bank.storage.bound == storage
            assert all(isinstance(entry[0], tuple) and isinstance(entry[1], tuple)
                       for entry in bank.entries.values())
            if mutation == "uv":
                arrays[6][0, 0] += Fraction(1, 2**100)
                assert float(arrays[6][0, 0]) == 0.25
            elif mutation == "patch":
                arrays[5][0] = 1
            elif mutation == "charts":
                arrays[1][1, 1, 0] = Fraction(2)
            elif mutation == "cells":
                arrays[0][1] = (2, 1, 0)
            elif mutation == "patches":
                arrays[2][1] = 1
            elif mutation == "cell_ids":
                arrays[3][0] = 5
            elif mutation == "vertex_ids":
                arrays[4][0] = 200
            elif mutation == "frame":
                bank.frame = ("source", "changed-frame", "material")
            else:
                bank.frame = ("source", "frame", "changed-role")
            start = budget.work_units
            changed = owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
            assert budget.work_units > start
            fresh = owner.chart_stencil(*arrays, 10_000, budget=budget)
            assert_same_stencil(changed, fresh)
            assert len(bank.entries) == (2 if mutation in ("uv", "patch") else 1)
        assert not bank.entries
        assert bank.storage.bound == 0
        assert budget.temporary_bytes_upper == 0


def test_material_routing_cold_replay_exact_work_boundary_and_cleanup() -> None:
    arrays = material_route_fixture()
    samples = []
    for _ in range(2):
        with surface_resources() as (_, budget):
            with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
                start = budget.work_units
                result = owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
                samples.append((result, budget.work_units - start, bank.storage.bound))
            assert budget.temporary_bytes_upper == 0
    assert_same_stencil(samples[0][0], samples[1][0])
    assert samples[0][1:] == samples[1][1:]
    required = samples[0][1]
    assert required > 0
    for allowance in (required, required - 1):
        budget = CoordinateEnclosureBudget(
            HostBatchTicket(allowance, bytes_upper=1_000_000),
            maximum_work_units=allowance, maximum_memory_bytes=1_000_000,
        )
        if allowance == required:
            with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
                result = owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
                assert_same_stencil(result, samples[0][0])
                assert budget.work_units == required
                assert_same_stencil(
                    owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget),
                    result,
                )
        else:
            with pytest.raises(CoordinateEnclosureResourceError):
                with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
                    owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
        assert not bank.entries
        assert bank.storage.bound == budget.temporary_bytes_upper == 0
    with surface_resources() as (_, budget):
        with pytest.raises(RuntimeError, match="rollback"):
            with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
                owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
                raise RuntimeError("rollback")
        assert not bank.entries
        assert bank.storage.bound == budget.temporary_bytes_upper == 0


def test_material_routing_storage_refusal_and_cross_request_are_closed() -> None:
    arrays = material_route_fixture()
    budget = CoordinateEnclosureBudget(
        HostBatchTicket(100_000, bytes_upper=1024),
        maximum_work_units=100_000, maximum_memory_bytes=1024,
    )
    with pytest.raises(CoordinateEnclosureResourceError):
        with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
            owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
    assert not bank.entries
    assert budget.temporary_bytes_upper == 0
    with surface_resources() as (_, first), surface_resources() as (_, second):
        with owner.prepared_material_routing(("source", "frame"), budget=first) as bank:
            with pytest.raises(ValueError, match="request budget"):
                owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=second)


@pytest.mark.parametrize("maximum_fidelity", (0.1, 0.0))
def test_material_routing_survives_split_relocation_and_final_assembly(
    maximum_fidelity: float,
) -> None:
    surface = torus_surface()
    initial = metric_grid(surface, 4)
    initial.material_charts = np.asarray(
        [Fraction(float(value)) for value in initial.charts.flat], dtype=object,
    ).reshape(initial.charts.shape)
    mesh = CellMesh.from_triangles(
        initial.points, initial.cells, vertex_global_ids=initial.vertex_ids,
    )
    # Assembly binds the actual mesh's cell identities.
    initial.cell_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    initial.parents = [{int(value)} for value in initial.cell_ids]
    initial.next_cell_id = int(np.max(initial.cell_ids)) + 1
    source = owner.SurfaceSource(
        initial.cells, initial.material_charts, initial.patches, initial.cell_ids,
        initial.vertex_ids, initial.metric, initial.charts,
    )
    fixed = np.zeros((5, 5), dtype=np.bool_)
    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    states, counts, outputs, work = [], [], [], []
    for cached in (False, True):
        state = deepcopy(initial)
        with surface_resources() as (_, budget):
            with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
                route = bank if cached else None
                edges, first, _ = owner._edge_table(state)
                rows, local = first // 3, first % 3
                assert state.material_charts is not None
                midpoints = np.mean(
                    state.material_charts[rows[:, None], owner._LOCAL_EDGES[local]], axis=1,
                )
                owner.chart_stencil(
                    source.cells, source.charts, source.patches, source.cell_ids,
                    source.vertex_ids, state.patches[rows], midpoints, 100_000,
                    material_cache=route, budget=budget,
                )
                edge = np.asarray((6, 12), dtype=np.int32)
                assert owner.split(surface.domain, state, edge, state.metric[6], budget=budget)
                # A stale carrier edge is rejected without invalidating source locations.
                assert not owner.split(
                    surface.domain, state, edge, state.metric[6], budget=budget,
                )
                pass_counts = []
                for _ in range(2):
                    pass_counts.append(owner.relocate(
                        surface.domain, state, fixed.ravel(), set(), maximum_fidelity,
                        state.points.shape[0], 100_000, source, {}, set(), 0.05,
                        material_routing=route, budget=budget,
                    ))
                outputs.append(owner.assemble_surface(
                    mesh, state, source.cells, source.patches, source.charts, 100_000,
                    material_routing=route, budget=budget,
                ))
                counts.append(pass_counts)
                work.append(budget.work_units)
            assert not bank.entries
            assert bank.storage.bound == 0
        states.append(state)
    def same_material_bits(left: jax.Array | np.ndarray, right: jax.Array | np.ndarray) -> bool:
        first, second = np.asarray(left), np.asarray(right)
        if first.dtype != object:
            first, second = (np.ascontiguousarray(value).view(np.uint8) for value in (first, second))
        return bool(np.array_equal(first, second))

    assert tree_content_equal(states[0], states[1], array_equal=same_material_bits)
    assert tree_content_equal(outputs[0], outputs[1], array_equal=same_material_bits)
    assert counts[0] == counts[1]
    assert work[1] < work[0]
    if maximum_fidelity == 0.0:
        assert all(accepted == 0 and attempts > 0 for accepted, attempts in counts[1])
    else:
        assert sum(accepted for accepted, _ in counts[1]) > 0


def test_material_routing_hits_preserve_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = material_route_fixture()
    with surface_resources() as (_, budget):
        with pytest.raises(MeshcoreError, match="material deadline"):
            with owner.prepared_material_routing(("source", "frame"), budget=budget) as bank:
                owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
                ticket = budget.ticket
                checkpoint = type(ticket).checkpoint

                def expire(current: Any) -> None:
                    if current is ticket:
                        raise MeshcoreError(MeshcoreStatus.TIMEOUT, "material deadline")
                    checkpoint(current)

                with monkeypatch.context() as patch:
                    patch.setattr(type(ticket), "checkpoint", expire)
                    owner.chart_stencil(*arrays, 10_000, material_cache=bank, budget=budget)
        assert not bank.entries
        assert bank.storage.bound == budget.temporary_bytes_upper == 0

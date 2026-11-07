from __future__ import annotations

import copy
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
import phydrax.meshing.providers._native_periodic as native_periodic
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import exact_periodic_element
from phydrax.discretization._view_support import OwnerLocalCellGeometry
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    MeshingFailureCategory,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import PeriodicConstraint, RegionControl
from phydrax.meshing._local_metric import (
    execute_local_metric_adaptation,
    LocalMetricOutcome,
)
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._periodic import (
    periodic_delaunay_mesh,
    periodic_entity_orbit_map,
    PeriodicPointOrbits,
    stage_periodic_topology_edit,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeCompliancePolicy, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_periodic import (
    NativePeriodicSource,
    prepare_native_periodic,
)


def test_exact_size_statistic_compares_interpolated_lengths() -> None:
    from phydrax.meshing._device_adaptation import (
        _owner_local_size_statistic_compare,
        OwnerLocalPeriodicSizeStatistic,
    )

    def statistic(
        lower: int, upper: int, fraction: Fraction
    ) -> OwnerLocalPeriodicSizeStatistic:
        return OwnerLocalPeriodicSizeStatistic(
            "control",
            "p50",
            1.0,
            Fraction(lower),
            Fraction(upper),
            fraction,
            0.0,
            True,
            False,
            False,
        )

    before = statistic(1, 9, Fraction(1, 10))
    after = statistic(1, 4, Fraction(3, 5))
    equal = statistic(4, 4, Fraction(1, 2))
    assert _owner_local_size_statistic_compare(after, before) > 0
    assert _owner_local_size_statistic_compare(before, after) < 0
    assert _owner_local_size_statistic_compare(equal, equal) == 0
    assert _owner_local_size_statistic_compare(statistic(2, 2, Fraction(0)), equal) < 0


def _point_source() -> PeriodicPointOrbits:
    cell = PeriodicCell(np.asarray(((1.0, 0.2), (0.3, 1.1)), dtype=np.float64))
    fractional = np.asarray(
        ((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77)),
        dtype=np.float64,
    )
    return PeriodicPointOrbits(
        cell,
        np.asarray(cell.cartesian(fractional), dtype=np.float64),
        np.asarray((91, 4, 53, 17), dtype=np.int64),
    )


def _specification(
    source_id: str,
    source_revision: str,
    entity_kind: MeshingEntityKind,
    vertex_entity_set_id: str,
    vertex_ids: np.ndarray,
    domain_entity_set_id: str,
    domain_entity_ids: np.ndarray,
    /,
    *,
    periodic_constraints: tuple[PeriodicConstraint, ...] = (),
    region_controls: tuple[RegionControl, ...] = (),
) -> SurfaceMeshingSpec:
    vertex_scope = MeshingScope(
        source_id,
        source_revision,
        entity_kind,
        0,
        vertex_entity_set_id,
        vertex_ids,
    )
    domain_scope = MeshingScope(
        source_id,
        source_revision,
        entity_kind,
        2,
        domain_entity_set_id,
        domain_entity_ids,
    )
    return SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        domain_scope,
        size_controls=(UniformSizeControl(vertex_scope, 0.35),),
        region_controls=region_controls,
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
        periodic_constraints=periodic_constraints,
    )


def _periodic_vertex_pair(
    mesh: CellMesh,
    source_id: str,
    source_revision: str,
    /,
) -> PeriodicConstraint:
    topology = mesh.periodic_topology
    assert topology is not None
    orbit_map = periodic_entity_orbit_map(mesh, 0)
    ids = np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64)
    windings = np.asarray(orbit_map.winding_shifts, dtype=np.int64)
    lifted_to_quotient = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    entity_set_id = mesh.entity_set(0).entity_set_id
    for quotient_row in range(orbit_map.quotient_entity_ids.size):
        members = np.flatnonzero(lifted_to_quotient == quotient_row)
        for source_row in members:
            for target_row in members:
                exponents = windings[target_row] - windings[source_row]
                if not np.any(exponents):
                    continue
                source_scope = MeshingScope(
                    source_id,
                    source_revision,
                    MeshingEntityKind.MESH,
                    0,
                    entity_set_id,
                    ids[[source_row]],
                )
                target_scope = MeshingScope(
                    source_id,
                    source_revision,
                    MeshingEntityKind.MESH,
                    0,
                    entity_set_id,
                    ids[[target_row]],
                )
                transform = np.asarray(
                    exact_periodic_element(topology.cell, exponents), dtype=np.float64
                )
                return PeriodicConstraint(source_scope, target_scope, transform)
    raise AssertionError(
        "Periodic point orbits did not contain a nontrivial winding pair."
    )


def _stalled_periodic_adaptation(
    monkeypatch: pytest.MonkeyPatch, run: Callable[[], object], /
) -> tuple[CellMesh, LocalMetricOutcome, MeshingFailure]:
    """Run a periodic route that must stall, recording its local adaptation."""
    calls: list[tuple[CellMesh, LocalMetricOutcome]] = []

    def recorded(
        mesh: CellMesh, metric: np.ndarray, /, **controls: Any
    ) -> LocalMetricOutcome:
        outcome = execute_local_metric_adaptation(mesh, metric, **controls)
        calls.append((mesh, outcome))
        return outcome

    monkeypatch.setattr(native_periodic, "execute_local_metric_adaptation", recorded)
    with pytest.raises(MeshingFailure) as raised:
        run()
    assert len(calls) == 1
    return calls[0][0], calls[0][1], raised.value


def _assert_honest_stall(
    failure: MeshingFailure,
    mesh: CellMesh,
    outcome: LocalMetricOutcome,
    expected: tuple[tuple[float, str], ...],
    /,
) -> CellMesh:
    """The stall is reported edge by edge and its candidate keeps every contract."""
    assert failure.category is MeshingFailureCategory.COMPLIANCE_FAILED
    assert failure.stage == "periodic-metric-adaptation"
    assert "status=stalled" in str(failure)
    assert failure.evidence.checkpoint == "periodic-metric-adaptation:stalled"
    assert failure.evidence.logical_findings == tuple(
        f"stalled_edge_{index}: {reason}" for index, (_, reason) in enumerate(expected)
    )
    achieved = dict(failure.evidence.achieved_quantities)
    evidence = outcome.evidence
    assert evidence.stalled and not evidence.converged
    assert evidence.out_of_range_edges == len(expected)
    assert len(evidence.stalled_edges) == len(expected)
    for index, ((length, reason), (expected_length, expected_reason)) in enumerate(
        zip(evidence.stalled_edges, expected, strict=True)
    ):
        assert reason == expected_reason
        assert length == pytest.approx(expected_length, abs=5.0e-4)
        assert achieved[f"stalled_edge_{index}_metric_length"] == length

    # The stalled candidate is still a valid periodic edit: exact lifted orbits
    # and windings, and an exact convex P1 transfer stencil for every vertex.
    target = stage_periodic_topology_edit(mesh, outcome.edit)
    assert target is not None
    topology = target.periodic_topology
    assert topology is not None
    topology.require_lifted(target.topology, target.lifted_topology_id, target.blocks)
    topology.require_coordinates(target.coordinates)
    assert np.any(np.asarray(periodic_entity_orbit_map(target, 0).winding_shifts))
    edit = outcome.edit
    assert edit.vertex_orbit_keys is not None and edit.vertex_windings is not None
    exact_weights = edit.stencil_exact_weights
    assert exact_weights is not None
    for row in exact_weights:
        assert all(weight >= 0 for weight in row)
        assert sum(row, Fraction(0)) == 1
    weights = np.where(
        np.asarray(edit.stencil_valid), np.asarray(edit.stencil_weights), 0.0
    )
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, rtol=0.0, atol=1.0e-12)
    return target


def test_native_periodic_route_adapts_skew_winding_orbits_to_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_orbits = _point_source()
    source_id, revision = "periodic-sites", "periodic-sites-revision-1"
    source = NativePeriodicSource(source_orbits, source_id, revision)
    specification = _specification(
        source_id,
        revision,
        MeshingEntityKind.GEOMETRY,
        source.scope_entity_set_id(0),
        source.scope_entity_ids(0),
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )

    # The skew orbits adapt until one short birth edge remains whose collapse
    # would create an edge above the split threshold in either direction (the
    # classic split/collapse hysteresis) and whose flip is invalid: an honest
    # stall, reported as such rather than published as a size-compliant mesh.
    mesh, outcome, failure = _stalled_periodic_adaptation(
        monkeypatch,
        lambda: prepare_native_periodic(
            source,
            specification,
            NativeMeshingOptions("periodic_delaunay"),
            phx.SpatialCoordinateContract.si(),
        ).execute(),
    )
    target = _assert_honest_stall(
        failure,
        mesh,
        outcome,
        (
            (
                0.6532,
                "collapse=hysteresis+hysteresis;optimal=periodic;flip=invalid",
            ),
        ),
    )
    vertex_map = periodic_entity_orbit_map(target, 0)
    assert vertex_map.quotient_entity_ids.size > source_orbits.point_ids.size
    assert source.cell_regions is None


def test_native_periodic_route_binds_constraints_and_materials_by_quotient_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_orbits = _point_source()
    mesh = periodic_delaunay_mesh(source_orbits).mesh
    cell_map = periodic_entity_orbit_map(mesh, mesh.topological_dimension)
    quotient_count = cell_map.quotient_entity_ids.size
    quotient_regions = np.arange(quotient_count, dtype=np.int32) % 2
    lifted_regions = quotient_regions[
        np.asarray(cell_map.lifted_to_quotient, dtype=np.int64)
    ]
    cell_ids = np.asarray(
        mesh.entity_set(mesh.topological_dimension).entity_ids, dtype=np.int64
    )
    region_by_id = {
        int(identifier): int(region)
        for identifier, region in zip(cell_ids, lifted_regions, strict=True)
    }
    block_regions = np.concatenate(
        tuple(
            np.asarray(
                [region_by_id[int(identifier)] for identifier in block.global_ids],
                dtype=np.int32,
            )
            for block in mesh.blocks
        )
    )
    source_id, revision = mesh.mesh_id, mesh.numeric_version
    topology = mesh.periodic_topology
    assert topology is not None
    quotient_vertices = topology.quotient.entity_sets[0]
    source = NativePeriodicSource(mesh, source_id, revision, cell_regions=block_regions)
    periodic_constraint = _periodic_vertex_pair(mesh, source_id, revision)
    region_controls = tuple(
        RegionControl(
            MeshingScope(
                source_id,
                revision,
                MeshingEntityKind.MESH,
                mesh.topological_dimension,
                mesh.entity_set(mesh.topological_dimension).entity_set_id,
                cell_ids[lifted_regions == code],
            ),
            f"periodic-region-{code}",
            f"periodic-material-{code}",
            RegionRole.SOLID,
        )
        for code in (0, 1)
    )
    specification = _specification(
        source_id,
        revision,
        MeshingEntityKind.MESH,
        quotient_vertices.entity_set_id,
        np.asarray(quotient_vertices.entity_ids, dtype=np.int64),
        source.scope_entity_set_id(mesh.topological_dimension),
        source.scope_entity_ids(mesh.topological_dimension),
        periodic_constraints=(periodic_constraint,),
        region_controls=region_controls,
    )
    # The regular passes stall on two short birth edges (each collapse creates
    # an edge above the split threshold or removes a curve vertex off its
    # curve); the stall escape collapses them, the next splits repair the long
    # edges, and the route meets the unit-edge criterion.
    calls: list[tuple[CellMesh, LocalMetricOutcome]] = []

    def recorded(
        mesh: CellMesh, metric: np.ndarray, /, **controls: Any
    ) -> LocalMetricOutcome:
        outcome = execute_local_metric_adaptation(mesh, metric, **controls)
        calls.append((mesh, outcome))
        return outcome

    monkeypatch.setattr(native_periodic, "execute_local_metric_adaptation", recorded)
    prepare_native_periodic(
        source,
        specification,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    ).execute()
    assert len(calls) == 1
    adapted_mesh, outcome = calls[0]
    evidence = outcome.evidence
    assert evidence.converged and not evidence.stalled and not evidence.stalled_edges
    assert evidence.collapses > 0 and evidence.splits > 0
    target = stage_periodic_topology_edit(adapted_mesh, outcome.edit)
    assert target is not None
    topology = target.periodic_topology
    assert topology is not None
    topology.require_lifted(target.topology, target.lifted_topology_id, target.blocks)
    topology.require_coordinates(target.coordinates)

    # Materials follow the quotient cells through the adapted lineage: both
    # codes survive and every target cell orbit carries one material.
    values = native_periodic._target_cell_regions(
        adapted_mesh, target, outcome.edit, block_regions
    )
    assert np.unique(values).size == 2
    target_cells = periodic_entity_orbit_map(target, target.topological_dimension)
    for orbit in range(target_cells.quotient_entity_ids.size):
        rows = np.flatnonzero(
            np.asarray(target_cells.lifted_to_quotient, dtype=np.int64) == orbit
        )
        assert np.unique(values[rows]).size == 1
    assert values.size > block_regions.size

    invalid_transform = np.asarray(periodic_constraint.transform, dtype=np.float64).copy()
    invalid_transform[0, -1] += 0.01
    invalid_constraint = PeriodicConstraint(
        periodic_constraint.source_scope,
        periodic_constraint.target_scope,
        invalid_transform,
    )
    invalid_specification = _specification(
        source_id,
        revision,
        MeshingEntityKind.MESH,
        quotient_vertices.entity_set_id,
        np.asarray(quotient_vertices.entity_ids, dtype=np.int64),
        source.scope_entity_set_id(mesh.topological_dimension),
        source.scope_entity_ids(mesh.topological_dimension),
        periodic_constraints=(invalid_constraint,),
        region_controls=region_controls,
    )
    with pytest.raises(ValueError, match="not uniquely determined"):
        prepare_native_periodic(
            source,
            invalid_specification,
            NativeMeshingOptions("periodic_delaunay"),
            phx.SpatialCoordinateContract.si(),
        )
    for invalid_source_id, invalid_revision in (
        ("foreign-periodic-source", revision),
        (source_id, "stale-periodic-revision"),
    ):
        invalid_source_scope, invalid_target_scope = (
            MeshingScope(
                invalid_source_id,
                invalid_revision,
                MeshingEntityKind.MESH,
                0,
                scope.entity_set_id,
                np.asarray(scope.entity_ids, dtype=np.int64),
            )
            for scope in (
                periodic_constraint.source_scope,
                periodic_constraint.target_scope,
            )
        )
        invalid_constraint = PeriodicConstraint(
            invalid_source_scope, invalid_target_scope, periodic_constraint.transform
        )
        # The specification itself refuses a constraint bound to another source
        # or revision, before any provider can consume it.
        with pytest.raises(
            ValueError, match="Meshing controls must share the top-level source binding"
        ):
            _specification(
                source_id,
                revision,
                MeshingEntityKind.MESH,
                quotient_vertices.entity_set_id,
                np.asarray(quotient_vertices.entity_ids, dtype=np.int64),
                source.scope_entity_set_id(mesh.topological_dimension),
                source.scope_entity_ids(mesh.topological_dimension),
                periodic_constraints=(invalid_constraint,),
                region_controls=region_controls,
            )
    # A constraint whose two scopes disagree on the revision is refused outright.
    stale_scope = MeshingScope(
        source_id,
        "stale-periodic-revision",
        MeshingEntityKind.MESH,
        0,
        quotient_vertices.entity_set_id,
        np.asarray(periodic_constraint.source_scope.entity_ids, dtype=np.int64),
    )
    with pytest.raises(
        ValueError, match="Periodic scopes must share one source revision"
    ):
        PeriodicConstraint(
            stale_scope, periodic_constraint.target_scope, periodic_constraint.transform
        )


def test_owner_local_skew_periodic_partition_uses_integer_work() -> None:
    from fractions import Fraction

    import jax

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from phydrax.graph import GraphPartitionPlan
    from phydrax.meshing._distribution import partition_owner_local_graph

    if jax.process_count() != 1:
        pytest.skip("This skew periodic graph-load regression uses one owner.")
    process = int(jax.process_index())
    device = next(device for device in jax.devices() if device.process_index == process)
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "skew-periodic-integer-work",
            (process,),
            ((process, device.id),),
            mesh_axes=(("periodic_work", 1),),
        ),
        (device,),
    )
    source_orbits = _point_source()
    basis = np.asarray(source_orbits.cell.vectors, dtype=np.float64)
    assert basis[0, 1] != 0.0 and basis[1, 0] != 0.0
    source_id = "skew-periodic-work-source"
    source_revision = "skew-periodic-work-revision"
    source = NativePeriodicSource(
        source_orbits,
        source_id,
        source_revision,
        point_owner_ranks=np.zeros(
            np.asarray(source_orbits.point_ids).shape, dtype=np.int32
        ),
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
        CellMeshingTarget(
            2,
            2,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(UniformSizeControl(scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    graph_plan = GraphPartitionPlan(
        1,
        maximum_imbalance=1.0,
        require_nonempty=True,
        refinement_passes=0,
    )
    assembly = prepare_native_periodic(
        source,
        specification,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    ).execute(
        execution_group=group,
        graph_partition_plan=graph_plan,
    )
    assert assembly.owner_local
    view = assembly.parts[0].carrier
    assert isinstance(view, OwnerLocalCellGeometry)
    from phydrax.meshing._device_adaptation import OwnerLocalPeriodicAdaptiveResult
    from phydrax.meshing.providers._native_periodic import (
        _adapt_owner_local_periodic_assembly,
    )

    assert isinstance(view.adaptive_result, OwnerLocalPeriodicAdaptiveResult)
    assert view.adaptive_result.live_forest is not None
    retained = view.adaptive_result.live_forest.records
    assert np.any(~np.asarray(retained.cell_field_map()["active"]))
    renewed = OwnerLocalCellGeometry.from_stored_fields(
        view,
        group,
        current_specification=specification,
        graph_partition_plan=graph_plan,
        axis_name="periodic_live_family_restore",
    )
    assert renewed.adaptive_result is not None
    assert renewed.adaptive_result.live_forest is not None
    assert (
        renewed.adaptive_result.live_forest.forest_id
        == view.adaptive_result.live_forest.forest_id
    )
    from phydrax.meshing._distribution_migration import (
        _partition_owner_local_periodic_families,
    )

    families = _partition_owner_local_periodic_families(assembly, graph_plan, group)
    assert families.result.feasible
    np.testing.assert_array_equal(families.result.entity_ids, view.owned_cell_ids)
    np.testing.assert_array_equal(families.result.owners, 0)
    coarse_specification = SurfaceMeshingSpec(
        specification.target,
        scope,
        size_controls=(UniformSizeControl(scope, 0.7),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    coarse_assembly = _adapt_owner_local_periodic_assembly(
        assembly, coarse_specification, group
    )
    coarse_view = coarse_assembly.parts[0].carrier
    assert isinstance(coarse_view, OwnerLocalCellGeometry)
    assert (
        np.asarray(coarse_view.owned_cell_ids).size < np.asarray(view.owned_cell_ids).size
    )
    assert coarse_view.adaptive_result is not None
    lineage = coarse_view.adaptive_result.lineage
    assert lineage is not None
    assert np.any(
        np.asarray(lineage.entity_lineage(2).relation_kinds)
        == int(phx.meshing.EntityLineageKind.COARSENED_INTO)
    )
    cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
    neighbor_ids = np.asarray(view.cell_neighbor_ids, dtype=np.int64)
    neighbor_owner_ranks = np.asarray(view.cell_neighbor_owner_ranks, dtype=np.int32)
    assert np.all(np.isin(neighbor_ids, cell_ids))
    owner_by_cell_id = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            cell_ids.tolist(),
            np.asarray(view.cell_owner_ranks, dtype=np.int32).tolist(),
            strict=True,
        )
    }
    assert np.array_equal(
        np.asarray(
            [owner_by_cell_id[int(identifier)] for identifier in neighbor_ids],
            dtype=np.int32,
        ),
        neighbor_owner_ranks,
    )
    assert len(view.blocks) == 1
    block = view.blocks[0]
    assert block.cell_kind == "triangle"
    assert view.exact_vertex_coordinates is not None
    exact_vertex_bank = tuple(view.exact_vertex_coordinates)
    block_vertices = np.asarray(block.vertices, dtype=np.int64)
    block_vertex_valid = np.asarray(block.vertex_valid, dtype=np.bool_)
    block_cell_ids = np.asarray(block.cell_ids, dtype=np.int64)
    exact_by_id: dict[int, Fraction] = {}
    for row, cell_id in enumerate(block_cell_ids.tolist()):
        local_vertices = block_vertices[row][block_vertex_valid[row]]
        assert local_vertices.size == 3
        points = tuple(exact_vertex_bank[int(vertex)] for vertex in local_vertices)
        origin = points[0]
        first = tuple(points[1][axis] - origin[axis] for axis in range(2))
        second = tuple(points[2][axis] - origin[axis] for axis in range(2))
        measure = (first[0] * second[1] - first[1] * second[0]) / 2
        assert measure > 0
        exact_by_id[cell_id] = measure
    owned_ids = np.asarray(view.owned_cell_ids, dtype=np.int64)
    work = np.asarray(view.owned_cell_weights)
    assert owned_ids.size > 0
    assert work.dtype == np.dtype(np.int64)
    assert work.shape == owned_ids.shape
    assert np.array_equal(work, np.ones(owned_ids.shape, dtype=np.int64))
    owned_measures = tuple(exact_by_id[int(identifier)] for identifier in owned_ids)
    assert all(isinstance(value, Fraction) and value > 0 for value in owned_measures)
    assert any(Fraction.from_float(float(value)) != value for value in owned_measures)
    assert any(value != Fraction(1) for value in owned_measures)

    evidence = partition_owner_local_graph(
        assembly,
        graph_plan,
        group,
        axis_name="skew_periodic_integer_work_view",
    )
    assert evidence.result.feasible
    assert evidence.result.plan_id == graph_plan.plan_id
    assert np.array_equal(
        np.asarray(evidence.result.entity_ids, dtype=np.int64),
        owned_ids,
    )


@pytest.fixture(scope="module")
def periodic_transfer_geometry() -> tuple[OwnerLocalCellGeometry, OwnerLocalCellGeometry]:
    import jax

    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.meshing.providers._native_periodic import (
        _adapt_owner_local_periodic_assembly,
    )

    if jax.process_count() != 1:
        pytest.skip("The paired periodic transfer regression uses one owner.")
    device = jax.devices()[0]
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "periodic-transfer-claims",
            (0,),
            ((0, device.id),),
            mesh_axes=(("periodic_transfer", 1),),
        ),
        (device,),
    )
    orbits = _point_source()
    source = NativePeriodicSource(
        orbits,
        "periodic-transfer-source",
        "periodic-transfer-revision",
        point_owner_ranks=np.zeros(orbits.point_ids.shape, dtype=np.int32),
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    target = CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",)))
    fine_spec = SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(UniformSizeControl(scope, 0.35),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    coarse_spec = SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(UniformSizeControl(scope, 0.7),),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
    )
    fine = prepare_native_periodic(
        source,
        fine_spec,
        NativeMeshingOptions("periodic_delaunay"),
        phx.SpatialCoordinateContract.si(),
    ).execute(execution_group=group)
    coarse = _adapt_owner_local_periodic_assembly(fine, coarse_spec, group)
    fine_view, coarse_view = fine.parts[0].carrier, coarse.parts[0].carrier
    assert isinstance(fine_view, OwnerLocalCellGeometry)
    assert isinstance(coarse_view, OwnerLocalCellGeometry)
    restored = OwnerLocalCellGeometry.from_stored_fields(
        coarse_view,
        group,
        current_specification=coarse_spec,
        graph_partition_plan=None,
        axis_name="periodic_transfer_restore",
    )
    return fine_view, restored


def test_periodic_transfer_replays_linear_geometry(
    periodic_transfer_geometry: tuple[OwnerLocalCellGeometry, OwnerLocalCellGeometry],
) -> None:
    from phydrax.meshing._device_adaptation import (
        _require_owner_local_periodic_transfer,
        OwnerLocalPeriodicAdaptiveResult,
    )

    source, target = periodic_transfer_geometry
    result = target.adaptive_result
    assert isinstance(result, OwnerLocalPeriodicAdaptiveResult)
    _require_owner_local_periodic_transfer(result, source, target)
    stencil = result.vertex_stencil
    assert stencil is not None
    assert source.exact_vertex_coordinates is not None
    assert target.exact_vertex_coordinates is not None
    source_points = dict(
        zip(
            map(int, np.asarray(source.vertex_ids)),
            source.exact_vertex_coordinates,
            strict=True,
        )
    )
    target_points = dict(
        zip(
            map(int, np.asarray(target.vertex_ids)),
            target.exact_vertex_coordinates,
            strict=True,
        )
    )
    rows = np.asarray(stencil.source_global_ids)
    valid = np.asarray(stencil.valid)
    reproduced = {
        int(identifier): tuple(
            sum(
                (
                    source_points[int(rows[row, column])][axis]
                    * stencil.exact_weights[row][column]
                    for column in np.flatnonzero(valid[row])
                ),
                Fraction(0),
            )
            for axis in range(2)
        )
        for row, identifier in enumerate(np.asarray(stencil.target_global_ids))
    }
    assert reproduced == {
        identifier: target_points[identifier] for identifier in reproduced
    }


@pytest.mark.parametrize("claim", ("stencil", "entity-lineage", "measures"))
def test_periodic_transfer_rejects_rehashed_false_claims(
    periodic_transfer_geometry: tuple[OwnerLocalCellGeometry, OwnerLocalCellGeometry],
    claim: str,
) -> None:
    from phydrax.meshing._device_adaptation import (
        _require_owner_local_periodic_transfer,
        OwnerLocalPeriodicAdaptiveResult,
    )
    from phydrax.meshing._lineage import (
        EntityLineage,
        MeshLineage,
        VertexInterpolationStencil,
    )

    source, target = periodic_transfer_geometry
    result = target.adaptive_result
    assert isinstance(result, OwnerLocalPeriodicAdaptiveResult)
    if claim == "stencil":
        stencil = result.vertex_stencil
        assert stencil is not None
        rows = np.asarray(stencil.source_global_ids).copy()
        column = int(np.flatnonzero(np.asarray(stencil.valid)[0])[0])
        rows[0, column] = next(
            int(identifier)
            for identifier in source.vertex_ids
            if int(identifier) != rows[0, column]
        )
        changed = result._replace(
            vertex_stencil=VertexInterpolationStencil(
                stencil.source_entity_set_id,
                stencil.target_entity_set_id,
                stencil.target_global_ids,
                rows,
                stencil.weights,
                stencil.valid,
                exact_weights=stencil.exact_weights,
            )
        )
    elif claim == "entity-lineage":
        lineage = result.lineage
        assert lineage is not None
        entity = lineage.entity_lineage(0)
        rows = np.asarray(entity.source_global_ids).copy()
        rows[[0, 1]] = rows[[1, 0]]
        replacement = EntityLineage(
            0,
            entity.source_entity_set_id,
            entity.target_entity_set_id,
            rows,
            entity.target_global_ids,
            entity.relation_kinds,
            created_target_ids=entity.created_target_ids,
            deleted_source_ids=entity.deleted_source_ids,
        )
        changed = result._replace(
            lineage=MeshLineage(
                lineage.source_topology_id,
                lineage.target_topology_id,
                tuple(
                    replacement if row.dimension == 0 else row for row in lineage.entities
                ),
            )
        )
    else:
        changed = result._replace(
            source_cell_measures=tuple(
                2 * value for value in result.source_cell_measures
            ),
            target_cell_measures=tuple(
                2 * value for value in result.target_cell_measures
            ),
        )
    changed = changed._replace(result_id=changed._computed_result_id())
    altered = eqx.tree_at(
        lambda value: value.adaptive_result,
        target,
        changed,
        is_leaf=lambda value: isinstance(value, OwnerLocalPeriodicAdaptiveResult),
    )
    with pytest.raises(ValueError):
        _require_owner_local_periodic_transfer(changed, source, altered)


def test_algebraic_periodic_delaunay_result_restores_from_archive(
    tmp_path: Path,
) -> None:
    from phydrax.discretization._exact_algebraic_geometry import (
        ExactPeriodicAlgebraicSource,
    )
    from phydrax.geometry._surface_source_archive import _geometry
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )
    from phydrax.meshing._result import CellMeshingResult

    # Four seeds exhausting one coset of the half-unit lattice with a hard,
    # zero-tolerance target admit the exact equilateral orbit-band carrier.
    cell = PeriodicCell(np.eye(2, dtype=np.float64))
    points = np.asarray(
        ((0.125, 0.125), (0.625, 0.125), (0.125, 0.625), (0.625, 0.625)),
        dtype=np.float64,
    )
    orbits = PeriodicPointOrbits(cell, points, np.asarray((7, 3, 11, 5), dtype=np.int64))
    source_id, revision = "algebraic-sites", "algebraic-sites-revision-1"
    source = NativePeriodicSource(orbits, source_id, revision)
    vertex_scope = MeshingScope(
        source_id,
        revision,
        MeshingEntityKind.GEOMETRY,
        0,
        source.scope_entity_set_id(0),
        source.scope_entity_ids(0),
    )
    domain_scope = MeshingScope(
        source_id,
        revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
        domain_scope,
        size_controls=(UniformSizeControl(vertex_scope, 0.5),),
        size_compliance=SizeCompliancePolicy(),
    )
    result = (
        phx.meshing.NativeMeshingProvider(NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    witness = result.geometry.source_map
    assert type(witness) is ExactPeriodicAlgebraicSource

    path = tmp_path / "algebraic-periodic-result.phx"
    write_model_artifact(path, result, limits=None)
    restored = read_model_artifact(
        path,
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    assert restored.result_id == result.result_id
    assert restored.mesh.mesh_id == result.mesh.mesh_id
    assert restored.geometry.source_mesh_id == result.geometry.source_mesh_id
    restored_witness = restored.geometry.source_map
    assert type(restored_witness) is ExactPeriodicAlgebraicSource
    assert restored_witness.source_map_id == witness.source_map_id
    assert restored_witness.coefficients == witness.coefficients

    # The stored witness is rebuilt from its exact leaves, never trusted as stored.
    forged = copy.copy(witness)
    object.__setattr__(forged, "source_map_id", "forged")
    with pytest.raises(ValueError, match="does not reproduce its source leaves"):
        _geometry(
            eqx.tree_at(lambda value: value.source_map, result.geometry, forged),
            result.mesh,
        )

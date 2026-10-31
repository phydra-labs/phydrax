from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

import phydrax as phx
import phydrax.meshing.providers._native_periodic as native_periodic
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import exact_periodic_element
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
        ((0.6631, "collapse=hysteresis+hysteresis;flip=invalid"),),
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
    # Two short birth edges remain: each collapse either creates an edge above
    # the split threshold (hysteresis) or would remove a curve vertex off its
    # curve (inadmissible), and neither flip is valid/improving. The route
    # reports this stall instead of publishing the unmet candidate.
    stalled_mesh, outcome, failure = _stalled_periodic_adaptation(
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
        stalled_mesh,
        outcome,
        (
            (0.6156, "collapse=hysteresis+inadmissible;flip=invalid"),
            (0.7057, "collapse=hysteresis+inadmissible;flip=not-improving"),
        ),
    )

    # Materials follow the quotient cells through the stalled candidate's lineage:
    # both codes survive and every target cell orbit carries one material.
    values = native_periodic._target_cell_regions(
        stalled_mesh, target, outcome.edit, block_regions
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

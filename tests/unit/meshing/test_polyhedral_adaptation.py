#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
from phydrax.discretization import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._transfer import TransferGeometryBinding
from phydrax.discretization.vem._polyhedral import (
    prepare_polyhedral_h1_virtual_element_3d,
)
from phydrax.geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
)
from phydrax.lifecycle._meshing_source_records import (
    reconstruct_stored_fields,
    register_meshing_source_types,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
)
from phydrax.meshing._periodic import bind_periodic_topology_edit
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_adaptation import (
    adapt_polyhedral_mesh,
    PolyhedralAdaptationOperation,
    PolyhedralMeshAdaptation,
)
from phydrax.meshing._polyhedral_adaptation_shared import PolyhedralAdaptationError
from phydrax.meshing._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
)
from phydrax.meshing._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    require_periodic_nonnested_geometry,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.polyhedral_meshes import (
    boxes,
    common_refinement_remap,
    periodic_adaptation_source,
    periodic_fv_public_inputs,
)


# Retained acceptance envelopes from this module's pre-remediation unit-cube
# constant-inventory/common-overlap and degree-one periodic edge-trace checks.
# These empirical regression values are unchanged by the lifecycle decomposition.
FV_TRACE_REGRESSION_ABS = 1e-13
FV_INVENTORY_REGRESSION_ABS = 1e-12


POLYHEDRAL_REFERENCE_LIMITS = MeshingLimits(
    maximum_cells=1 << 24,
    maximum_vertices=1 << 22,
    maximum_work_units=1 << 26,
    maximum_scratch_bytes=8_000_000_000,
)


def test_native_plane_split_closes_faces_and_transfers_fv_inventory() -> None:
    source = boxes((0.0, 1.0))
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=[[1, 0, 0, 0.5]]
    )
    outcome = adapt_polyhedral_mesh(source, request, numeric_version="split")
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        target, _, stencil = assemble_topology_edit(
            source, outcome.edit, numeric_version="split", budget=budget
        )
    connectivity = target.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            f"polyhedral_adaptation_test.target.connectivity must be PolyhedralConnectivity; got {type(connectivity).__name__}"
        )
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    assert outcome.common_refinement.target_mesh_id == target.mesh_id
    assert connectivity.cell_count == 2
    assert np.count_nonzero(np.asarray(connectivity.face_neighbor) >= 0) == 1
    assert stencil is None
    plan = common_refinement_remap(source, target, outcome)
    averages = np.asarray(plan.apply(np.asarray([3.0], dtype=np.float64)))
    assert np.allclose(averages, 3.0, rtol=0, atol=FV_TRACE_REGRESSION_ABS)
    assert np.dot(
        averages, np.asarray(outcome.common_refinement.target_measures)
    ) == pytest.approx(3.0, abs=FV_INVENTORY_REGRESSION_ABS)


def test_plane_split_vertex_cap_retains_accepted_geometry_and_topology() -> None:
    source = boxes((0.0, 1.0))
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=[[1, 0, 0, 0.5]]
    )
    with pytest.raises(
        PolyhedralAdaptationError, match=r"polyhedral_adaptation\.maximum_vertices"
    ):
        adapt_polyhedral_mesh(
            source, request, maximum_vertices=source.coordinates.shape[0]
        )
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)


def test_connected_agglomeration_removes_only_interior_faces_and_conserves() -> None:
    source = boxes((0.0, 0.5, 1.0))
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=((20, 21),)
    )
    outcome = adapt_polyhedral_mesh(source, request, numeric_version="coarse")
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        target, _, _ = assemble_topology_edit(
            source, outcome.edit, numeric_version="coarse", budget=budget
        )
    assert target.connectivity.cell_count == 1
    plan = common_refinement_remap(source, target, outcome)
    assert np.allclose(
        plan.apply(np.asarray([2.0, 4.0], dtype=np.float64)),
        [3.0],
        atol=FV_TRACE_REGRESSION_ABS,
    )
    assert np.asarray(outcome.common_refinement.target_measures)[0] == pytest.approx(
        1.0, abs=FV_INVENTORY_REGRESSION_ABS
    )


def test_exact_agglomeration_keeps_source_rows_after_interior_vertex_removal() -> None:

    topology = reference_cell_topology("hexahedron")
    complex_ = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    with NativeExecutionScope(NativePolyhedralSchedule().native_limits()) as scope:
        construction = generate_polyhedral_volume(
            complex_,
            sites=np.asarray(
                [
                    [x, y, z]
                    for x in (0.25, 0.75)
                    for y in (0.25, 0.75)
                    for z in (0.25, 0.75)
                ],
                dtype=np.float64,
            ),
            scope=scope,
        )
    source = construction.mesh
    connectivity = source.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            f"polyhedral_adaptation_test.source.connectivity must be PolyhedralConnectivity; got {type(connectivity).__name__}"
        )
    ids = tuple(int(value) for value in np.asarray(connectivity.cell_global_ids))
    outcome = adapt_polyhedral_mesh(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=(ids,)
        ),
        source_geometry=construction.geometry,
        numeric_version="exact-agglomeration",
    )
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        target, _, _ = assemble_topology_edit(
            source, outcome.edit, numeric_version="exact-agglomeration", budget=budget
        )
    source_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source.vertex_global_ids))
    }
    expected = np.asarray(source.coordinates)[
        [
            source_rows[int(identifier)]
            for identifier in np.asarray(target.vertex_global_ids)
        ]
    ]
    assert target.coordinates.shape[0] < source.coordinates.shape[0]
    np.testing.assert_array_equal(target.coordinates, expected)
    outcome.geometry.resolve(target)
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    plan = common_refinement_remap(source, target, outcome)
    np.testing.assert_allclose(
        plan.apply(np.ones(len(ids))), [1.0], rtol=0, atol=FV_TRACE_REGRESSION_ABS
    )


def test_regeneration_uses_nonnested_overlap_not_site_parent_interpolation() -> None:
    source = boxes((0.0, 0.5, 1.0))
    topology = reference_cell_topology("hexahedron")
    coordinates = np.asarray(topology.vertices, dtype=np.float64)
    complex_ = PiecewiseLinearComplex(
        coordinates,
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=complex_,
        sites=[[0.5, 0.25, 0.5], [0.5, 0.75, 0.5]],
    )
    outcome = adapt_polyhedral_mesh(source, request, numeric_version="regenerated")
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        target, _, stencil = assemble_topology_edit(
            source, outcome.edit, numeric_version="regenerated", budget=budget
        )
    assert stencil is None
    assert outcome.construction is not None
    assert outcome.common_refinement.target_mesh_id == target.mesh_id
    counts = np.diff(np.asarray(outcome.common_refinement.target_offsets))
    assert np.all(counts >= 2)
    plan = common_refinement_remap(source, target, outcome)
    averages = np.asarray(plan.apply(np.asarray([2.0, 4.0], dtype=np.float64)))
    assert np.allclose(averages, 3.0, rtol=0, atol=FV_INVENTORY_REGRESSION_ABS)
    assert np.dot(
        averages, np.asarray(outcome.common_refinement.target_measures)
    ) == pytest.approx(3.0, abs=FV_INVENTORY_REGRESSION_ABS)


def test_region_crossing_agglomeration_and_feature_removal_rollback() -> None:
    source = boxes((0.0, 0.5, 1.0))
    connectivity = source.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            f"polyhedral_adaptation_test.source.connectivity must be PolyhedralConnectivity; got {type(connectivity).__name__}"
        )
    original = source.mesh_id
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=((20, 21),)
    )
    with pytest.raises(
        PolyhedralAdaptationError, match=r"polyhedral_adaptation\.cell_classes"
    ):
        adapt_polyhedral_mesh(source, request, cell_classes=[0, 1])
    interior = np.flatnonzero(np.asarray(connectivity.face_neighbor) >= 0)
    face_ids = np.asarray(connectivity.face_global_ids)[interior]
    with pytest.raises(
        PolyhedralAdaptationError, match=r"polyhedral_adaptation\.protected_face_ids"
    ):
        adapt_polyhedral_mesh(source, request, protected_face_ids=face_ids)
    assert source.mesh_id == original


def test_regeneration_native_allocation_refusal_preserves_source() -> None:

    source = boxes((0.0, 1.0))
    assert isinstance(source.connectivity, PolyhedralConnectivity)
    original_id = source.mesh_id
    original_coordinates = np.asarray(source.coordinates).copy()
    original_incidence = np.asarray(source.connectivity.face_vertex_values).copy()
    topology = reference_cell_topology("hexahedron")
    complex_ = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=complex_,
        sites=[[0.5, 0.5, 0.5]],
    )
    with pytest.raises(PolyhedralAdaptationError) as failure:
        adapt_polyhedral_mesh(source, request, maximum_scratch_bytes=1)
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    # The first real preallocation boundary may be the native host-token
    # header or the coordinate bank. Neither may pretend a later generator
    # allocation was attempted, and both must retain actual resource evidence.
    refusal = failure.value.__cause__
    if isinstance(refusal, CoordinateEnclosureResourceError):
        assert refusal.resource == "polynomial_storage"
        assert refusal.limit == 1
        assert refusal.requested > refusal.limit
        assert 0 <= refusal.completed <= refusal.limit
        assert refusal.requested > refusal.completed
        assert refusal.phase_evidence is not None
    else:
        assert isinstance(refusal, MeshcoreError)
        assert refusal.status is MeshcoreStatus.CAPACITY_EXCEEDED
        assert "execution_reserve_host_storage" in str(refusal)
        assert refusal.memory_counters is not None
        assert refusal.memory_counters.limit_bytes == 1
        assert refusal.memory_counters.refusals > 0
        assert refusal.memory_counters.requested_bytes > refusal.memory_counters.limit_bytes
        assert refusal.memory_counters.allocations == 0
        assert refusal.memory_counters.peak_bytes == 0
    assert source.mesh_id == original_id
    np.testing.assert_array_equal(source.coordinates, original_coordinates)
    np.testing.assert_array_equal(
        source.connectivity.face_vertex_values, original_incidence
    )


def test_common_refinement_replay_authenticates_retained_coverage() -> None:
    source = boxes((0.0, 1.0))
    target = boxes((0.0, 0.5))
    common = prepare_common_refinement(
        source,
        target,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.PARTIAL),
    )
    assert common.status is CommonRefinementStatus.SUCCESS
    assert common.evidence.source_gap_count == 1
    register_meshing_source_types()
    measure_bindings = {
        "source_measures": common.source_measures,
        "target_measures": common.target_measures,
    }
    evidence = reconstruct_stored_fields(common.evidence, bindings=measure_bindings)
    assert evidence.evidence_id == common.evidence.evidence_id
    assert (
        evidence.maximum_relative_source_defect
        == common.evidence.maximum_relative_source_defect
    )
    np.testing.assert_array_equal(
        evidence.source_coverage_defects, common.evidence.source_coverage_defects
    )
    np.testing.assert_array_equal(
        evidence.source_coverage_tolerances, common.evidence.source_coverage_tolerances
    )
    with pytest.raises(
        ValueError,
        match=r"CommonRefinementEvidence\.stored_fields.*got changed evidence",
    ):
        reconstruct_stored_fields(
            common.evidence,
            bindings={
                **measure_bindings,
                "source_measures": common.source_measures * 2.0,
            },
        )
    mesh_bindings = {"source_mesh": source, "target_mesh": target}
    rebuilt = reconstruct_stored_fields(common, bindings=mesh_bindings)
    assert rebuilt.refinement_id == common.refinement_id
    assert rebuilt.evidence.evidence_id == common.evidence.evidence_id
    np.testing.assert_array_equal(rebuilt.volumes, common.volumes)
    np.testing.assert_array_equal(rebuilt.target_offsets, common.target_offsets)
    changed = eqx.tree_at(
        lambda refinement: refinement.volumes,
        common,
        common.volumes * 0.5,
    )
    with pytest.raises(
        ValueError, match=r"stored_fields\.PreparedCommonRefinement.*changed volumes"
    ):
        reconstruct_stored_fields(changed, bindings=mesh_bindings)


def test_periodic_regeneration_restart_preserves_authored_constraint_controls(
    tmp_path: Path,
) -> None:

    M = phx.meshing
    topology = reference_cell_topology("hexahedron")
    complex_ = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.arange(6, dtype=np.int64),
        np.asarray([[-1, 0]] * 6, dtype=np.int64),
        ("material",),
    )
    source_scope = M.MeshingScope(
        "restart-cube", "r1", M.MeshingEntityKind.GEOMETRY, 2, "left", [4]
    )
    target_scope = M.MeshingScope(
        "restart-cube", "r1", M.MeshingEntityKind.GEOMETRY, 2, "right", [5]
    )
    transform = np.eye(4)
    transform[0, 3] = 1
    constraint = M.PeriodicConstraint(
        source_scope,
        target_scope,
        transform,
        tolerance=0,
        conforming_required=True,
        source_entity_ids=np.asarray([4], dtype=np.int64),
        orientations=np.asarray([-1], dtype=np.int8),
    )
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=complex_,
        sites=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64),
        periodic_constraints=(constraint,),
    )
    receipt = write_meshing_source_closure(tmp_path / "periodic-regeneration", request)
    restored = read_meshing_source_closure(
        tmp_path / "periodic-regeneration", expected_content_id=receipt.content_id
    )
    assert isinstance(restored, PolyhedralMeshAdaptation)
    assert restored.request_id == request.request_id
    assert restored.periodic_constraints[0].constraint_id == constraint.constraint_id
    np.testing.assert_array_equal(restored.periodic_constraints[0].transform, transform)
    np.testing.assert_array_equal(restored.periodic_constraints[0].orientations, [-1])
    with pytest.raises(
        ValueError, match=r"PolyhedralMeshAdaptation\.periodic_constraints"
    ):
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=[[1, 0, 0, 0.5]],
            periodic_constraints=(constraint,),
        )


def test_periodic_regeneration_certifies_actual_new_source_and_rejects_forged_coverage(
    tmp_path: Path,
) -> None:

    domain, constraints, construction = periodic_adaptation_source()
    source = construction.mesh
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=domain,
        sites=[[0.5, 0.25, 0.5], [0.5, 0.75, 0.5]],
        periodic_constraints=constraints,
    )
    outcome = adapt_polyhedral_mesh(
        source,
        request,
        source_geometry=construction.geometry,
        numeric_version="periodic-new-source",
    )
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        target, _, _ = assemble_topology_edit(
            source, outcome.edit, numeric_version="periodic-new-source", budget=budget
        )
    assert target.periodic_topology is not None
    assert target.periodic_topology.actual_geometry is not None
    assert outcome.common_refinement.target_mesh_id == target.mesh_id
    assert outcome.edit.periodic_orbits is not None
    authority = outcome.edit.periodic_orbits.nonnested_geometry
    assert authority is not None
    assert (
        construction.geometry.exact_source is not None
        and outcome.geometry.exact_source is not None
    )
    assert (
        construction.geometry.exact_source.source_id
        != outcome.geometry.exact_source.source_id
    )
    assert not np.intersect1d(source.vertex_global_ids, target.vertex_global_ids).size
    transferred = common_refinement_remap(source, target, outcome).apply(
        np.asarray([2.0, 4.0])
    )
    np.testing.assert_allclose(transferred, 3.0, rtol=0, atol=FV_INVENTORY_REGRESSION_ABS)
    assert source.periodic_topology is not None
    forged = eqx.tree_at(
        lambda common: common.volumes,
        authority.common_refinement,
        authority.common_refinement.volumes * 0.5,
    )

    queries_upper = cell_geometry_preparation_query_upper(
        authority.source_geometry
    ) + cell_geometry_preparation_query_upper(authority.target_geometry)

    with (
        coordinate_reference_budget(queries_upper=queries_upper) as budget,
        pytest.raises(ValueError, match=r"CellTopologyEdit\.nonnested_geometry"),
    ):
        require_periodic_nonnested_geometry(
            source.periodic_topology,
            target,
            authority._replace(common_refinement=forged),
            budget=budget,
        )
    changed_geometry = eqx.tree_at(
        lambda geometry: geometry.coordinates,
        authority.target_geometry,
        authority.target_geometry.coordinates + 0.125,
    )
    with (
        coordinate_reference_budget(queries_upper=queries_upper) as budget,
        pytest.raises(ValueError, match=r"\.rounded_vertices"),
    ):
        require_periodic_nonnested_geometry(
            source.periodic_topology,
            target,
            authority._replace(target_geometry=changed_geometry),
            budget=budget,
        )
    receipt = write_meshing_source_closure(
        tmp_path / "nonnested-periodic-edit", outcome.edit
    )
    cold_edit = read_meshing_source_closure(
        tmp_path / "nonnested-periodic-edit", expected_content_id=receipt.content_id
    )
    assert isinstance(cold_edit, CellTopologyEdit)
    assert (
        cold_edit.periodic_orbits is not None
        and cold_edit.periodic_orbits.nonnested_geometry is not None
    )
    cold_authority = cold_edit.periodic_orbits.nonnested_geometry
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        cold_target, _, _ = assemble_topology_edit(
            cold_authority.source,
            cold_edit,
            numeric_version=cold_authority.target.numeric_version,
            budget=budget,
        )
    assert cold_target.mesh_id == target.mesh_id
    np.testing.assert_array_equal(cold_target.coordinates, target.coordinates)
    np.testing.assert_array_equal(
        cold_authority.common_refinement.volumes, outcome.common_refinement.volumes
    )

    other_source_epoch = source.with_coordinates(
        source.coordinates, numeric_version="other-source-epoch"
    )
    np.testing.assert_array_equal(other_source_epoch.coordinates, source.coordinates)
    np.testing.assert_array_equal(
        other_source_epoch.vertex_global_ids, source.vertex_global_ids
    )
    with pytest.raises(ValueError, match=r"CellTopologyEdit\.nonnested_geometry"):
        with coordinate_reference_budget(
            POLYHEDRAL_REFERENCE_LIMITS,
            queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
        ) as budget:
            bind_periodic_topology_edit(
                other_source_epoch, target, outcome.edit.periodic_orbits, budget=budget
            )
    other_target_epoch = target.with_coordinates(
        target.coordinates, numeric_version="other-target-epoch"
    )
    np.testing.assert_array_equal(other_target_epoch.coordinates, target.coordinates)
    np.testing.assert_array_equal(
        other_target_epoch.vertex_global_ids, target.vertex_global_ids
    )
    with (
        coordinate_reference_budget(queries_upper=queries_upper) as budget,
        pytest.raises(ValueError, match=r"CellTopologyEdit\.nonnested_geometry"),
    ):
        require_periodic_nonnested_geometry(
            source.periodic_topology, other_target_epoch, authority, budget=budget
        )


def test_periodic_plane_split_and_connected_agglomeration_preserve_exact_source_and_fv_inventory() -> (
    None
):
    _, _, construction = periodic_adaptation_source()
    source = construction.mesh
    refined = adapt_polyhedral_mesh(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=[[1, 0, 0, 0.25], [1, 0, 0, 0.75]],
        ),
        source_geometry=construction.geometry,
        numeric_version="periodic-split",
    )
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        fine, _, trace = assemble_topology_edit(
            source, refined.edit, numeric_version="periodic-split", budget=budget
        )
    assert trace is not None
    source_x = np.asarray(source.coordinates)[:, 0]
    nodal = np.asarray(
        trace.apply(source.vertex_global_ids, 0.5 * np.minimum(source_x, 1 - source_x))
    )
    fine_x = np.asarray(fine.coordinates)[:, 0]
    # The independent periodic tent field is affine on the source cells' edges,
    # including source-authored seam subdivision vertices.
    np.testing.assert_allclose(
        nodal, 0.5 * np.minimum(fine_x, 1 - fine_x), rtol=0, atol=FV_TRACE_REGRESSION_ABS
    )
    assert fine.periodic_topology is not None

    source_vem = prepare_polyhedral_h1_virtual_element_3d(
        source, cell_geometry=construction.geometry
    )
    fine_vem = prepare_polyhedral_h1_virtual_element_3d(
        fine, cell_geometry=refined.geometry
    )
    geometry_binding = TransferGeometryBinding(
        cell_geometry_id(construction.geometry),
        cell_geometry_id(refined.geometry),
        "topology-correspondence",
        source_topology_id=source.topology_id,
        target_topology_id=fine.topology_id,
        coverage_defect=None,
    )
    field_transfer = source_vem.vertex_field_transfer(
        fine_vem,
        trace.as_transfer(
            source.vertex_global_ids,
            source_topology_id=source.topology_id,
            target_topology_id=fine.topology_id,
        ),
        geometry_binding,
    )
    assert source.periodic_topology is not None
    source_rows = np.asarray(source.periodic_topology.orbit_representatives(0))
    fine_rows = np.asarray(fine.periodic_topology.orbit_representatives(0))
    quotient_values = 0.5 * np.minimum(source_x[source_rows], 1 - source_x[source_rows])
    np.testing.assert_allclose(
        field_transfer.primal_operator.mv(quotient_values),
        nodal[fine_rows],
        rtol=0,
        atol=FV_TRACE_REGRESSION_ABS,
    )
    assert fine.connectivity.cell_count > source.connectivity.cell_count
    values = np.asarray(
        common_refinement_remap(source, fine, refined).apply(np.asarray([2.0, 4.0]))
    )
    common = refined.common_refinement
    groups = []
    for source_row in range(common.source_cell_count):
        rows = np.unique(
            np.asarray(common.target_cells)[np.asarray(common.source_cells) == source_row]
        )
        groups.append(
            tuple(int(value) for value in np.asarray(common.target_cell_global_ids)[rows])
        )
    coarse = adapt_polyhedral_mesh(
        fine,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=tuple(groups)
        ),
        source_geometry=refined.geometry,
        numeric_version="periodic-agglomerated",
    )
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        target, _, restriction = assemble_topology_edit(
            fine, coarse.edit, numeric_version="periodic-agglomerated", budget=budget
        )
    assert restriction is not None
    current_x = np.asarray(target.coordinates)[:, 0]
    np.testing.assert_allclose(
        restriction.apply(fine.vertex_global_ids, nodal),
        0.5 * np.minimum(current_x, 1 - current_x),
        rtol=0,
        atol=FV_TRACE_REGRESSION_ABS,
    )
    assert target.periodic_topology is not None
    assert target.periodic_topology.actual_geometry is not None
    assert target.connectivity.cell_count == source.connectivity.cell_count
    restored = np.asarray(common_refinement_remap(fine, target, coarse).apply(values))
    np.testing.assert_allclose(
        restored, [2.0, 4.0], rtol=0, atol=FV_INVENTORY_REGRESSION_ABS
    )
    np.testing.assert_allclose(
        np.dot(restored, np.asarray(coarse.common_refinement.target_measures)),
        np.dot(
            np.asarray([2.0, 4.0]), np.asarray(refined.common_refinement.source_measures)
        ),
        rtol=0,
        atol=FV_INVENTORY_REGRESSION_ABS,
    )


@pytest.mark.parametrize(
    ("change", "field"),
    [
        pytest.param("domain", r"CellTopologyEdit\.nonnested_geometry", id="domain"),
        pytest.param("group", r"prepare_periodic_edit_target\.source", id="group"),
        pytest.param(
            "source-revision",
            r"CellTopologyEdit\.nonnested_geometry",
            id="source-revision",
        ),
    ],
)
def test_periodic_regeneration_rejects_changed_source_controls_atomically(
    change: str,
    field: str,
) -> None:

    M = phx.meshing
    domain, constraints, construction = periodic_adaptation_source()
    source = construction.mesh
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    if change == "domain":
        domain = PiecewiseLinearComplex(
            domain.vertices,
            tuple(
                domain.polygon_vertices[first:second]
                for first, second in zip(
                    domain.polygon_offsets[:-1], domain.polygon_offsets[1:], strict=True
                )
            ),
            domain.polygon_facets,
            domain.facet_regions,
            ("different-material",),
            segments=domain.segments,
            boundary=domain.boundary,
        )
    elif change == "group":
        constraints = constraints[:2]
    else:
        revised = []
        for constraint in constraints:
            first, second = constraint.source_scope, constraint.target_scope
            source_scope = M.MeshingScope(
                first.source_id,
                "r2",
                first.entity_kind,
                first.entity_dimension,
                first.entity_set_id,
                first.entity_ids,
            )
            target_scope = M.MeshingScope(
                second.source_id,
                "r2",
                second.entity_kind,
                second.entity_dimension,
                second.entity_set_id,
                second.entity_ids,
            )
            revised.append(
                M.PeriodicConstraint(
                    source_scope,
                    target_scope,
                    constraint.transform,
                    tolerance=constraint.tolerance,
                    conforming_required=constraint.conforming_required,
                    source_entity_ids=constraint.source_entity_ids,
                    orientations=constraint.orientations,
                )
            )
        constraints = tuple(revised)
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=domain,
        sites=[[0.5, 0.25, 0.5], [0.5, 0.75, 0.5]],
        periodic_constraints=constraints,
    )
    with pytest.raises(ValueError, match=field):
        adapt_polyhedral_mesh(
            source,
            request,
            source_geometry=construction.geometry,
            numeric_version="rejected-authority",
        )
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)


def test_crossing_plane_cut_does_not_claim_unknown_virtual_face_trace() -> None:

    topology = reference_cell_topology("hexahedron")
    domain = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    with NativeExecutionScope(NativePolyhedralSchedule().native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain, sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], scope=scope
        )
    source = construction.mesh
    identity = source.mesh_id
    outcome = adapt_polyhedral_mesh(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=[[1, 0, 0, 0.25], [0, 1, 0, 0.25]],
        ),
        source_geometry=construction.geometry,
        numeric_version="unknown-face-trace",
    )
    with coordinate_reference_budget(
        POLYHEDRAL_REFERENCE_LIMITS,
        queries_upper=POLYHEDRAL_REFERENCE_LIMITS.maximum_geometry_queries,
    ) as budget:
        _, _, nodal = assemble_topology_edit(
            source, outcome.edit, numeric_version="unknown-face-trace", budget=budget
        )
    # Crossing planes create nodes inside the source faces. Their geometric
    # convex coefficients do not evaluate a virtual H1 field.
    assert nodal is None
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    assert source.mesh_id == identity


def test_polyhedral_edit_rejects_foreign_exact_geometry_before_changing_source() -> None:

    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64)
    domain = PiecewiseLinearComplex(
        points,
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    other_domain = PiecewiseLinearComplex(
        points + np.asarray([3.0, 0.0, 0.0]),
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    with NativeExecutionScope(NativePolyhedralSchedule().native_limits()) as scope:
        construction = generate_polyhedral_volume(
            domain, sites=[[0.5, 0.5, 0.5]], scope=scope
        )
    with NativeExecutionScope(NativePolyhedralSchedule().native_limits()) as scope:
        foreign = generate_polyhedral_volume(
            other_domain, sites=[[3.5, 0.5, 0.5]], scope=scope
        )
    source = construction.mesh
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    with pytest.raises(ValueError, match=r"CellGeometrySpec\.resolve\.inputs"):
        adapt_polyhedral_mesh(
            source,
            PolyhedralMeshAdaptation(
                PolyhedralAdaptationOperation.PLANE_SPLIT, planes=[[1, 0, 0, 0.25]]
            ),
            source_geometry=foreign.geometry,
        )
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)


@pytest.fixture(scope="module")
def canonical_periodic_power_source() -> tuple[phx.meshing.CellMeshingResult, phx.meshing.MeshingLimits]:
    source, specification, options, _ = periodic_fv_public_inputs("translation")
    result = phx.meshing.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()
    return result, specification.limits


def _canonical_polyhedral_policy(
    limits: phx.meshing.MeshingLimits,
) -> phx.meshing.MeshAdaptationPolicy:
    return phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
        limits=limits,
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        ),
    )


def _canonical_polyhedral_inventory_transfer(
    source: phx.meshing.CellMeshingResult,
    adaptation: phx.meshing.MeshAdaptationResult,
    values: np.ndarray,
) -> np.ndarray:
    from phydrax.discretization import UnstructuredFiniteVolumePlan
    from phydrax.discretization.finite_volume._unstructured_remap import (
        UnstructuredConservativeRemapPlan,
    )

    common = adaptation.common_refinement
    assert common is not None and common.succeeded
    source_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(source.mesh).prepare(
        cell_geometry=source.geometry
    )
    target_fv = UnstructuredFiniteVolumePlan.from_cell_mesh(adaptation.target.mesh).prepare(
        cell_geometry=adaptation.target.geometry
    )
    plan = UnstructuredConservativeRemapPlan(
        source_fv, target_fv, common.target_offsets, common.source_cells, common.volumes,
        method="common-refinement", provenance=common.refinement_id,
    )
    transferred = np.asarray(plan.apply(values))
    np.testing.assert_allclose(
        np.dot(transferred, np.asarray(target_fv.cell_volumes)),
        np.dot(values, np.asarray(source_fv.cell_volumes)),
        rtol=0, atol=FV_INVENTORY_REGRESSION_ABS,
    )
    return transferred


def _require_canonical_polyhedral_source_associations(
    source: phx.meshing.CellMeshingResult,
    target: phx.meshing.CellMeshingResult,
) -> None:
    assert target.certification is not None and target.certification.passed
    assert target.certification.coverage is not None
    assert target.certification.coverage.status == "certified"
    assert target.geometry.exact_source is not None
    assert target.mesh.periodic_topology is not None
    target.audit.require_passed()
    for dimension in range(4):
        previous = next(
            row for row in source.associations
            if row.target_entity_set_id == source.mesh.entity_set(dimension).entity_set_id
        )
        current = next(
            row for row in target.associations
            if row.target_entity_set_id == target.mesh.entity_set(dimension).entity_set_id
        )
        assert (current.source_id, current.source_revision) == (
            previous.source_id, previous.source_revision
        )
    assert {label.name for label in target.labels} == {label.name for label in source.labels}
    assert {patch.name for patch in target.patches} == {patch.name for patch in source.patches}


def test_canonical_periodic_polyhedral_split_coarsen_preserves_associations_and_inventory(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, phx.meshing.MeshingLimits],
    tmp_path: Path,
) -> None:
    source, limits = canonical_periodic_power_source
    source_id = source.result_id
    coordinates = np.asarray(source.mesh.coordinates).copy()
    policy = _canonical_polyhedral_policy(limits)
    split = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            PolyhedralMeshAdaptation(
                PolyhedralAdaptationOperation.PLANE_SPLIT,
                planes=((1, 0, 0, 0.25), (1, 0, 0, 0.75)),
            ),
            policy=policy,
        )
    )
    assert split.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert split.target.mesh.connectivity.cell_count > source.mesh.connectivity.cell_count
    _require_canonical_polyhedral_source_associations(source, split.target)
    values = _canonical_polyhedral_inventory_transfer(source, split, np.asarray((2.0, 4.0)))
    common = split.common_refinement
    assert common is not None
    groups = tuple(
        tuple(
            int(value) for value in np.asarray(common.target_cell_global_ids)[
                np.unique(np.asarray(common.target_cells)[np.asarray(common.source_cells) == row])
            ]
        )
        for row in range(common.source_cell_count)
    )
    coarse = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            split.target,
            PolyhedralMeshAdaptation(
                PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=groups
            ),
            policy=policy,
        )
    )
    assert coarse.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert coarse.target.mesh.connectivity.cell_count == source.mesh.connectivity.cell_count
    _require_canonical_polyhedral_source_associations(source, coarse.target)
    _canonical_polyhedral_inventory_transfer(split.target, coarse, values)
    refined_certificate = split.target.certification
    coarse_certificate = coarse.target.certification
    assert refined_certificate is not None and refined_certificate.embedding is not None
    assert coarse_certificate is not None and coarse_certificate.embedding is not None
    partition = coarse_certificate.embedding.periodic_restriction
    assert partition is not None and partition.direction == "coarsen"
    assert partition.source_mesh_id == split.target.mesh.mesh_id
    assert (
        partition.source_embedding_certificate_id
        == refined_certificate.embedding.certificate_id
    )
    fine_connectivity = split.target.mesh.connectivity
    coarse_connectivity = coarse.target.mesh.connectivity
    assert isinstance(fine_connectivity, PolyhedralConnectivity)
    assert isinstance(coarse_connectivity, PolyhedralConnectivity)
    assert set(partition.child_cell_ids) == set(
        np.asarray(fine_connectivity.cell_global_ids).tolist()
    )
    assert set(partition.partition_cell_ids) == set(
        np.asarray(coarse_connectivity.cell_global_ids).tolist()
    )
    receipt = write_meshing_source_closure(tmp_path / "inverse-partition", partition)
    restored = read_meshing_source_closure(
        tmp_path / "inverse-partition", expected_content_id=receipt.content_id
    )
    assert type(restored) is type(partition)
    assert restored.evidence_id == partition.evidence_id
    assert restored.direction == "coarsen"
    assert source.result_id == source_id
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)


def test_canonical_periodic_polyhedral_admission_rejects_stripped_source_authority(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, phx.meshing.MeshingLimits],
) -> None:
    source, limits = canonical_periodic_power_source
    stripped = eqx.tree_at(lambda result: result.geometry.exact_source, source, None)
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=((1, 0, 0, 0.25),)
    )
    with pytest.raises((ValueError, TypeError, MeshingFailure)):
        phx.meshing.prepare_mesh_adaptation(
            stripped, request, policy=_canonical_polyhedral_policy(limits)
        )


def test_public_regeneration_publication_keeps_actual_source_and_target_aliases(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, phx.meshing.MeshingLimits],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise REGENERATE itself, not an independently generated surrogate."""
    from contextlib import contextmanager

    import phydrax.meshing._adaptation as publication_owner
    import phydrax.meshing._polyhedral_adaptation as construction_owner

    source, limits = canonical_periodic_power_source
    authority, specification, _, _ = periodic_fv_public_inputs("translation")
    source_id = source.result_id
    source_geometry = source.geometry.exact_source
    source_coordinates = np.asarray(source.mesh.coordinates).copy()
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=authority.complex,
        sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
        periodic_constraints=specification.periodic_constraints,
    )
    prepared = phx.meshing.prepare_mesh_adaptation(
        source, request, policy=_canonical_polyhedral_policy(limits)
    )
    produced = []
    publication_tokens = []
    original_adapt = construction_owner.adapt_polyhedral_mesh
    original_storage = publication_owner._polyhedral_publication_storage

    def observe_adapt(*args: Any, **kwargs: Any) -> Any:
        result = original_adapt(*args, **kwargs)
        produced.append(result)
        return result

    @contextmanager
    def observe_storage(*args: Any, **kwargs: Any) -> Any:
        with original_storage(*args, **kwargs) as admitted:
            assert admitted is not None
            storage, retained, records = admitted
            assert storage.workspace.bound >= retained + 1024 + 512 * records
            scope = args[2]
            assert (
                scope._limits_record.maximum_managed_bytes
                - scope.remaining().remaining_managed_bytes
                >= storage.workspace.bound
            )
            publication_tokens.append(storage.workspace)
            yield admitted
        # Each output keeps the real native reservation after lexical exit.
        assert storage.workspace.bound == retained + 1024 + 512 * records
        assert (
            scope._limits_record.maximum_managed_bytes
            - scope.remaining().remaining_managed_bytes
            >= storage.workspace.bound
        )

    monkeypatch.setattr(construction_owner, "adapt_polyhedral_mesh", observe_adapt)
    monkeypatch.setattr(
        publication_owner, "_polyhedral_publication_storage", observe_storage
    )
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector

    collector = CoordinateOperandCollector()
    result = phx.meshing.execute_mesh_adaptation(
        prepared, coordinate_operands=collector
    )
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    assert len(produced) == 1
    actual = produced[0]
    assert actual.construction is not None
    assert result.transition is not None
    assert result.transition.target is result.target
    assert result.transition.lineage is result.lineage
    assert result.target.geometry.exact_source is actual.geometry.exact_source
    assert result.common_refinement is actual.common_refinement
    assert result.target.mesh.mesh_id != source.mesh.mesh_id
    assert result.target.mesh.topology_id == result.lineage.target_topology_id
    assert result.lineage.source_topology_id == source.mesh.topology_id
    assert result.target.execution_evidence is not None
    record = result.target.execution_evidence
    assert record.preparation_evidence is prepared.preparation_evidence
    assert record.history_evidence == (source.execution_evidence,)
    assert {"vertices", "edges", "faces", "cells", "connectivity_entries", "data_bytes", "cavity_cells", "wall_seconds"} <= set(
        result.target.runtime.enforced_limits
    )
    assert {"work_units", "scratch_bytes"} <= set(
        result.target.runtime.unenforced_limits
    )
    assert record.total_managed_admission_peak_bytes_upper is not None
    assert collector.prepared is not None
    assert collector.prepared.mesh is result.target.mesh
    assert collector.prepared.geometry is result.target.geometry
    assert collector.prepared.earning_receipt is record
    from phydrax.meshing.qualification import execute_native_consumer

    def consume_current_target(budget: Any) -> str:
        # Only this actual adaptation's newly earned target bank is eligible.
        # Generation/history never pays another adaptation's source proof.
        assert collector.prepared is not None
        collector.prepared.require(
            result.target.mesh, result.target.geometry, record, budget,
            coordinate_contract=result.target.coordinate_contract,
        )
        budget.reserve(1)
        return result.target.mesh.mesh_id

    continued = execute_native_consumer(
        result.target, limits, consume_current_target,
        failure_factory=lambda stage, message, details: RuntimeError(
            f"{stage}: {message}; {details}"
        ),
        observe_managed_admission=True,
    )
    assert continued.value == result.target.mesh.mesh_id
    assert continued.result.execution_evidence is not None
    assert continued.result.execution_evidence.preparation_evidence is record
    assert continued.result.mesh is result.target.mesh
    assert continued.result.geometry is result.target.geometry
    assert publication_tokens and all(token.bound > 0 for token in publication_tokens)
    _require_canonical_polyhedral_source_associations(source, result.target)
    assert source.result_id == source_id
    assert source.geometry.exact_source is source_geometry
    np.testing.assert_array_equal(source.mesh.coordinates, source_coordinates)


@pytest.mark.parametrize("short", (False, True))
def test_publication_bound_admits_real_native_exact_headroom(
    short: bool,
) -> None:
    """The actual publication bound rejects R-1 before target construction."""
    import math
    from contextlib import nullcontext

    import phydrax.meshing._adaptation as publication_owner
    import phydrax.meshing._polyhedral_generation as topology_owner
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeHostStorageWorkspace,
    )

    source = boxes((0.0, 1.0))
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=((1, 0, 0, 0.5),)
    )
    outcome = adapt_polyhedral_mesh(source, request)
    # The bound only needs scientific source topology and organization banks.
    canonical = phx.meshing.certify_cell_mesh(
        source, phx.SpatialCoordinateContract.si()
    )
    required, _, _ = publication_owner._polyhedral_publication_bounds(
        canonical, outcome.edit
    )
    scope = NativeExecutionScope(
        NativeExecutionLimits(1 << 30, 1 << 26, 1 << 24, 1 << 30, math.inf)
    )
    entered_builder = False
    expectation = pytest.raises(MeshcoreError) if short else nullcontext()
    with expectation as refusal:
        with scope, topology_owner._PolyhedralTopologyStorage(scope) as storage:
            storage.workspace.reserve_empty()
            with NativeHostStorageWorkspace(scope) as ballast:
                ballast.reserve_empty()
                available = scope.remaining().remaining_managed_bytes
                ballast.set_bound(available - required + int(short))
                assert scope.remaining().remaining_managed_bytes == required - int(short)
                storage.workspace.set_bound(required)
            # Release only the artificial ballast: the admitted publication
            # token stays live while the real coordinate owner admits its own
            # independent exact-work storage.
            entered_builder = True
            with coordinate_reference_budget() as budget:
                target, lineage, _ = assemble_topology_edit(
                    source, outcome.edit, numeric_version="bounded", budget=budget
                )
            assert target.topology_id == lineage.target_topology_id
    assert entered_builder is not short
    if short:
        assert refusal is not None
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_public_regeneration_early_resource_refusal_preserves_accepted_source(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, phx.meshing.MeshingLimits],
) -> None:
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureResourceError,
    )

    source, _ = canonical_periodic_power_source
    authority, specification, _, _ = periodic_fv_public_inputs("translation")
    source_id = source.result_id
    source_coordinates = np.asarray(source.mesh.coordinates).copy()
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=authority.complex,
        sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
        periodic_constraints=specification.periodic_constraints,
    )
    with pytest.raises(CoordinateEnclosureResourceError) as refusal:
        phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source,
                request,
                policy=_canonical_polyhedral_policy(
                    MeshingLimits(maximum_scratch_bytes=1)
                ),
            )
        )
    # Current source preparation admits its real host bank before native work.
    # A one-byte cap must preserve that earlier typed refusal, not force a
    # later C++ refusal after the source operands have already escaped admission.
    assert refusal.value.limit == 1
    assert refusal.value.requested > refusal.value.limit
    assert refusal.value.completed == 0
    assert source.result_id == source_id
    np.testing.assert_array_equal(source.mesh.coordinates, source_coordinates)


@pytest.mark.parametrize("shortfall", (0, 1))
def test_public_regeneration_root_data_receipt_boundary(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch, shortfall: int,
) -> None:
    """Prove candidate admission R separately from final retained-data R."""
    import dataclasses

    import phydrax.meshing._adaptation as owner
    import phydrax.meshing._polyhedral_generation as generation
    from phydrax._model._structure import model_array_bytes
    from phydrax.meshing.providers._native_publication import _require_retained_data
    from tools.meshing_qualification._controls import MeshingLimitControls

    source, limits = canonical_periodic_power_source
    source_id = source.result_id
    source_coordinates = np.asarray(source.mesh.coordinates).copy()
    candidate_bounds: list[int] = []
    original_admit = generation._admit_polyhedral_mesh_capacity

    def observe_admit(
        vertices: int, cells: int, loops: int, corners: int, limits: MeshingLimits,
    ) -> None:
        candidate_bounds.append(
            generation._polyhedral_numeric_publication_bounds(
                vertices, cells, loops, corners
            )[1]
        )
        original_admit(vertices, cells, loops, corners, limits)

    monkeypatch.setattr(generation, "_admit_polyhedral_mesh_capacity", observe_admit)
    authority, specification, _, _ = periodic_fv_public_inputs("translation")
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.REGENERATE,
        complex_=authority.complex,
        sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
        periodic_constraints=specification.periodic_constraints,
    )
    reference = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source, request, policy=_canonical_polyhedral_policy(limits)
        )
    )
    retained = model_array_bytes(reference)
    assert retained > model_array_bytes(reference.target)
    # The final union is independently admitted exactly; it is not the
    # request's R when an earlier conservative candidate envelope dominates.
    retained_limits = dataclasses.replace(
        MeshingLimitControls.from_limits(limits),
        maximum_data_bytes=retained,
    ).to_limits()
    _require_retained_data(reference, retained_limits)
    with pytest.raises(MeshingFailure) as union_refusal:
        _require_retained_data(
            reference,
            dataclasses.replace(
                MeshingLimitControls.from_limits(limits),
                maximum_data_bytes=retained - 1,
            ).to_limits(),
        )
    assert union_refusal.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    required = max(retained, *candidate_bounds)
    capped = dataclasses.replace(
        MeshingLimitControls.from_limits(limits),
        maximum_data_bytes=required - shortfall,
    ).to_limits()
    prepared = phx.meshing.prepare_mesh_adaptation(
        source, request, policy=_canonical_polyhedral_policy(capped)
    )
    records = []
    original = owner.NativeExecutionRecord

    def observe_record(*args: Any, **kwargs: Any) -> Any:
        record = original(*args, **kwargs)
        records.append(record)
        return record

    monkeypatch.setattr(owner, "NativeExecutionRecord", observe_record)
    if shortfall:
        with pytest.raises(MeshingFailure) as failure:
            phx.meshing.execute_mesh_adaptation(prepared)
        assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        assert dict(failure.value.achieved)["required_data_bytes"] == required
        assert dict(failure.value.achieved)["remaining_data_bytes"] == required - 1
        assert not records
    else:
        result = phx.meshing.execute_mesh_adaptation(prepared)
        assert model_array_bytes(result) == retained
        # Changing authored limits changes prepared_id and numeric_version;
        # compare scientific topology/IDs rather than that request-bound mesh ID.
        assert result.target.mesh.topology_id == reference.target.mesh.topology_id
        np.testing.assert_array_equal(
            result.target.mesh.vertex_global_ids, reference.target.mesh.vertex_global_ids
        )
        np.testing.assert_array_equal(
            result.target.mesh.coordinates, reference.target.mesh.coordinates
        )
        assert len(records) == 1
        assert result.transition is not None
        assert result.transition.target is result.target
    assert source.result_id == source_id
    np.testing.assert_array_equal(source.mesh.coordinates, source_coordinates)


def test_root_publication_rechecks_original_clock_after_receipt_binding(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-scope receipt work cannot renew the request or earlier parent clock."""
    import phydrax.meshing._adaptation as owner

    source, limits = canonical_periodic_power_source
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=((1, 0, 0, 0.5),),
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    original = owner._AdaptationPublicationStorage.bind_receipt
    original_clock = owner.time.monotonic

    def expire_after_binding(*args: Any) -> Any:
        outcome = original(*args)
        monkeypatch.setattr(
            owner.time, "monotonic",
            lambda: original_clock() + limits.maximum_wall_seconds + 1.0,
        )
        return outcome

    monkeypatch.setattr(
        owner._AdaptationPublicationStorage, "bind_receipt", expire_after_binding
    )
    with pytest.raises(MeshingFailure) as failure:
        phx.meshing.execute_mesh_adaptation(prepared)
    assert failure.value.category is MeshingFailureCategory.TIMED_OUT


def test_public_polyhedral_preparation_clock_includes_receipt_publication(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._adaptation as owner

    source, limits = canonical_periodic_power_source
    original = owner._remember_power_witnesses
    clock = owner.time.monotonic

    def expire_before_publication(*args: Any, **kwargs: Any) -> Any:
        monkeypatch.setattr(
            owner.time, "monotonic",
            lambda: clock() + limits.maximum_wall_seconds + 1.0,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "_remember_power_witnesses", expire_before_publication)
    with pytest.raises(MeshingFailure) as failure:
        phx.meshing.prepare_mesh_adaptation(
            source,
            PolyhedralMeshAdaptation(
                PolyhedralAdaptationOperation.PLANE_SPLIT,
                planes=((1, 0, 0, 0.5),),
            ),
            policy=_canonical_polyhedral_policy(limits),
        )
    assert failure.value.category is MeshingFailureCategory.TIMED_OUT


def test_public_polyhedral_execution_preserves_earlier_parent_expiry(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
) -> None:
    import dataclasses
    import time

    import phydrax.meshing._adaptation as owner
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus

    source, limits = canonical_periodic_power_source
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=((1, 0, 0, 0.5),),
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    remaining, preparation = owner._preparation_execution_limits(prepared)
    assert preparation is prepared.preparation_evidence
    assert remaining.maximum_seconds < limits.maximum_wall_seconds
    assert remaining.maximum_work_units <= limits.maximum_work_units
    parent_limits = dataclasses.replace(
        limits.native_limits(), maximum_seconds=0.001
    )
    with pytest.raises(MeshcoreError) as failure:
        with NativeExecutionScope(parent_limits):
            time.sleep(0.01)
            phx.meshing.execute_mesh_adaptation(prepared)
    assert failure.value.status is MeshcoreStatus.TIMEOUT


@pytest.mark.parametrize("shortfall", (0, 1))
def test_ended_regeneration_publication_uses_real_exact_headroom(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch, shortfall: int,
) -> None:
    """R/R-1 admits the real completion token before any final receipt allocation."""
    import gc
    from contextlib import nullcontext

    import phydrax.meshing._adaptation as owner
    from phydrax._meshcore import (
        current_native_execution_scope,
        NativeHostStorageWorkspace,
    )

    source, limits = canonical_periodic_power_source
    authority, specification, _, _ = periodic_fv_public_inputs("translation")
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.REGENERATE,
            complex_=authority.complex,
            sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
            periodic_constraints=specification.periodic_constraints,
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    admissions = []
    original_admit = owner._AdaptationPublicationStorage.admit

    def measure(storage: Any, *args: Any) -> None:
        original_admit(storage, *args)
        admissions.append(storage.workspace.bound)

    monkeypatch.setattr(owner._AdaptationPublicationStorage, "admit", measure)
    reference = phx.meshing.execute_mesh_adaptation(prepared)
    assert reference.elapsed_seconds >= 0
    original_id = reference.result_id
    required = admissions.pop()
    del reference
    gc.collect()
    storages = []
    tokens = []
    records = []
    original_record = owner.NativeExecutionRecord

    def admit_exact(storage: Any, *args: Any) -> None:
        storages.append(storage)
        scope = storage.workspace._budget
        storage.workspace.reserve_empty()
        with NativeHostStorageWorkspace(scope) as ballast:
            ballast.reserve_empty()
            available = scope.remaining().remaining_managed_bytes
            ballast.set_bound(available - required + shortfall)
            assert scope.remaining().remaining_managed_bytes == required - shortfall
            original_admit(storage, *args)
            assert storage.workspace.bound == required
            tokens.append(storage.workspace._reservation.value)
        # Only artificial ballast closes. The admitted publication token stays
        # held across real scope settlement, scalar allocation and root binding.

    def observe_record(evidence: Any, *args: Any, **kwargs: Any) -> Any:
        storage = storages[-1]
        assert current_native_execution_scope() is None
        assert storage.workspace._budget.evidence is not None
        assert storage.workspace._reservation.value == tokens[-1]
        assert storage.workspace.bound == required
        record = original_record(evidence, *args, **kwargs)
        records.append(record)
        return record

    monkeypatch.setattr(owner._AdaptationPublicationStorage, "admit", admit_exact)
    monkeypatch.setattr(owner, "NativeExecutionRecord", observe_record)
    original_fingerprint = owner.canonical_fingerprint

    def live_fingerprint(*args: Any, **kwargs: Any) -> Any:
        assert current_native_execution_scope() is not None
        return original_fingerprint(*args, **kwargs)

    # All adaptation identities must be computed before scope end; ended
    # publication only rebinds the prevalidated root and its receipt.
    monkeypatch.setattr(owner, "canonical_fingerprint", live_fingerprint)
    expected = pytest.raises(MeshcoreError) if shortfall else nullcontext()
    with expected as refusal:
        result = phx.meshing.execute_mesh_adaptation(prepared)
    storage = storages[-1]
    if shortfall:
        assert refusal is not None
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        assert not records and not tokens
        assert storage.workspace.bound == 0
        assert storage.workspace._reservation.value is None
    else:
        assert len(records) == 1
        assert result.result_id == original_id
        assert result.target.execution_evidence is records[0]
        assert result.transition is not None
        assert result.transition.target is result.target
        assert storage.workspace._reservation.value == tokens[0]
        assert storage.workspace.bound == required
        assert records[0].preparation_evidence is prepared.preparation_evidence
        assert records[0].history_evidence == (source.execution_evidence,)


@pytest.mark.parametrize("stage", ("record", "binding", "handover"))
def test_ended_regeneration_publication_failure_explicitly_rolls_back(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """Every final-publication failure closes the one live token exactly once."""
    import phydrax.meshing._adaptation as owner
    from phydrax._meshcore import current_native_execution_scope

    source, limits = canonical_periodic_power_source
    source_id = source.result_id
    coordinates = np.asarray(source.mesh.coordinates).copy()
    authority, specification, _, _ = periodic_fv_public_inputs("translation")
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.REGENERATE,
            complex_=authority.complex,
            sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
            periodic_constraints=specification.periodic_constraints,
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    storages = []
    closes = []
    original_admit = owner._AdaptationPublicationStorage.admit
    original_close = owner._AdaptationPublicationStorage.close

    def admit(storage: Any, *args: Any) -> None:
        original_admit(storage, *args)
        storages.append(storage)

    def close(storage: Any) -> None:
        closes.append(storage)
        original_close(storage)

    class PublicationInterrupted(RuntimeError):
        pass

    monkeypatch.setattr(owner._AdaptationPublicationStorage, "admit", admit)
    monkeypatch.setattr(owner._AdaptationPublicationStorage, "close", close)
    if stage in ("record", "binding"):
        target = owner if stage == "record" else owner._AdaptationPublicationStorage
        name = "NativeExecutionRecord" if stage == "record" else "bind_receipt"

        def interrupt(*args: Any, **kwargs: Any) -> None:
            assert current_native_execution_scope() is None
            assert storages[-1].workspace.bound > 0
            raise PublicationInterrupted(stage)

        monkeypatch.setattr(target, name, interrupt)
    else:
        original_handover = owner._AdaptationPublicationStorage.handover

        def interrupt_handover(storage: Any, *args: Any) -> None:
            original_finalize = owner.weakref.finalize
            registrations = 0

            def finalize(*args: Any, **kwargs: Any) -> Any:
                nonlocal registrations
                registrations += 1
                if registrations == 2:
                    raise PublicationInterrupted(stage)
                return original_finalize(*args, **kwargs)

            with monkeypatch.context() as registration_patch:
                registration_patch.setattr(owner.weakref, "finalize", finalize)
                original_handover(storage, *args)

        monkeypatch.setattr(
            owner._AdaptationPublicationStorage, "handover", interrupt_handover
        )
    with pytest.raises(PublicationInterrupted, match=stage):
        phx.meshing.execute_mesh_adaptation(prepared)
    assert len(storages) == 1
    storage = storages[0]
    assert closes == [storage]
    assert storage.workspace.bound == 0
    assert storage.workspace._reservation.value is None
    assert storage._finalizers == []
    assert source.result_id == source_id
    np.testing.assert_array_equal(source.mesh.coordinates, coordinates)


def test_ended_regeneration_token_follows_bound_roots_and_extracted_receipt_leaf(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc

    import phydrax.meshing._adaptation as owner

    source, limits = canonical_periodic_power_source
    authority, specification, _, _ = periodic_fv_public_inputs("translation")
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.REGENERATE,
            complex_=authority.complex,
            sites=((0.5, 0.25, 0.5), (0.5, 0.75, 0.5)),
            periodic_constraints=specification.periodic_constraints,
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    storages = []
    tokens = []
    original_admit = owner._AdaptationPublicationStorage.admit

    def admit(storage: Any, *args: Any) -> None:
        original_admit(storage, *args)
        storages.append(storage)
        tokens.append(storage.workspace._reservation.value)

    monkeypatch.setattr(owner._AdaptationPublicationStorage, "admit", admit)
    result = phx.meshing.execute_mesh_adaptation(prepared)
    storage = storages[0]
    retained = storage.workspace.bound
    assert retained > 0
    assert storage.workspace._budget.completed_evidence().host_storage_live_bytes_upper >= retained
    target, transition = result.target, result.transition
    record = target.execution_evidence
    assert record is not None and transition is not None
    work, memory = record.work_counters, record.memory_counters
    leaf = record.receipt_id
    assert transition.target is target
    del result, record
    gc.collect()
    assert storage.workspace._reservation.value == tokens[0]
    del target
    gc.collect()
    assert storage.workspace.bound == retained
    del transition
    gc.collect()
    assert storage.workspace.bound == retained
    del work, memory
    gc.collect()
    assert storage.workspace.bound == retained
    del leaf
    gc.collect()
    assert storage.workspace.bound == 0
    assert storage.workspace._reservation.value is None


@pytest.mark.parametrize("stage", ("handover", "source_issue"))
def test_publication_expiry_after_handover_rolls_back(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    import phydrax.meshing._adaptation as owner

    source, limits = canonical_periodic_power_source
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT, planes=((1, 0, 0, 0.5),),
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    storages = []
    original_admit = owner._AdaptationPublicationStorage.admit
    original_clock = owner.time.monotonic

    def admit(storage: Any, *args: Any) -> None:
        original_admit(storage, *args)
        storages.append(storage)

    cls = (
        owner._AdaptationPublicationStorage if stage == "handover"
        else owner.NativeCompletedSourceIssue
    )
    name = "handover" if stage == "handover" else "complete"
    original = getattr(cls, name)

    def expire_after_commit(*args: Any) -> None:
        original(*args)
        monkeypatch.setattr(
            owner.time, "monotonic",
            lambda: original_clock() + limits.maximum_wall_seconds + 1,
        )

    monkeypatch.setattr(owner._AdaptationPublicationStorage, "admit", admit)
    monkeypatch.setattr(cls, name, expire_after_commit)
    with pytest.raises(MeshingFailure) as failure:
        phx.meshing.execute_mesh_adaptation(prepared)
    assert failure.value.category is MeshingFailureCategory.TIMED_OUT
    assert dict(failure.value.evidence.requested)["maximum_wall_seconds"] == limits.maximum_wall_seconds
    assert dict(failure.value.evidence.achieved)["remaining_seconds"] < 0
    assert len(storages) == 1
    assert storages[0].workspace.bound == 0
    assert storages[0]._finalizers == []


def test_live_preparation_clock_survives_without_periodic_receipt(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absolute clock belongs to every live request, not periodic proofs."""
    import phydrax.meshing._adaptation as owner

    source, limits = canonical_periodic_power_source
    prepared = phx.meshing.prepare_mesh_adaptation(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT, planes=((1, 0, 0, 0.5),),
        ),
        policy=_canonical_polyhedral_policy(limits),
    )
    prepared = owner.PreparedMeshAdaptation(
        source, prepared.request, prepared.policy, prepared.constraints,
        preparation_evidence=prepared.preparation_evidence,
    )
    expires_at = owner.time.monotonic() + 1
    owner._remember_power_witnesses(prepared, (), publication_expires_at=expires_at)
    monkeypatch.setattr(owner.time, "monotonic", lambda: expires_at + 1)
    native_limits, evidence = owner._preparation_execution_limits(prepared)
    assert native_limits.maximum_seconds == 0
    assert evidence is prepared.preparation_evidence
    with pytest.raises(MeshcoreError) as refusal:
        phx.meshing.execute_mesh_adaptation(prepared)
    assert refusal.value.status is MeshcoreStatus.TIMEOUT


@pytest.mark.parametrize("distributed", (False, True))
@pytest.mark.parametrize("nested", (False, True))
def test_polyhedral_runtime_keeps_unproved_variants_partial(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    distributed: bool, nested: bool,
) -> None:
    from contextlib import nullcontext

    source, limits = canonical_periodic_power_source
    partition = phx.meshing.MeshPartitionPolicy(
        phx.meshing.MeshPartitionKind.MORTON, 1
    )
    distribution = (
        phx.meshing.prepare_mesh_distribution(
            phx.meshing.MeshPart("polyhedral", source), policy=partition
        ) if distributed else None
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_POLYHEDRAL,
        limits=limits,
        audit_policy=_canonical_polyhedral_policy(limits).audit_policy,
        distribution=distribution,
        partition_policy=partition if distributed else None,
    )
    expected = (
        pytest.raises(ValueError, match="omit polyhedral attributes and distribution")
        if distributed else nullcontext()
    )
    with (
        expected,
        NativeExecutionScope(limits.native_limits()) if nested else nullcontext(),
    ):
        result = phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                source,
                PolyhedralMeshAdaptation(
                    PolyhedralAdaptationOperation.PLANE_SPLIT,
                    planes=((1, 0, 0, 0.25),),
                ),
                policy=policy,
            )
        )
    if distributed:
        # Existing scientific support remains an explicit preparation refusal,
        # not a successful route whose unowned distribution is called bounded.
        return
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    enforced = result.target.runtime.enforced_limits
    partial = result.target.runtime.unenforced_limits
    assert "cavity_cells" in enforced and "cavity_cells" not in partial
    assert ("wall_seconds" in enforced) is (not nested and not distributed)
    assert ("wall_seconds" in partial) is (nested or distributed)
    assert "data_bytes" in enforced and "data_bytes" not in partial
    assert {"work_units", "geometry_queries", "scratch_bytes"} <= set(partial)
    assert result.target.execution_evidence is not None
    if not nested:
        assert result.target.execution_evidence.total_managed_admission_peak_bytes_upper is not None


def test_adaptation_root_binding_admits_work_before_end(
    canonical_periodic_power_source: tuple[phx.meshing.CellMeshingResult, MeshingLimits],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R/R-1 covers root discovery and binding, with no ended graph traversal."""
    import dataclasses
    from contextlib import nullcontext

    import jax

    import phydrax.meshing._adaptation as owner
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeHostStorageWorkspace,
    )

    source, limits = canonical_periodic_power_source
    children, _ = jax.tree_util.tree_flatten_with_path(
        source, is_leaf=lambda value: value is not source
    )
    required = len(dataclasses.fields(source)) + 2 * len(children) + 1
    for shortfall in (0, 1):
        bounded = dataclasses.replace(
            limits.native_limits(), maximum_work_units=required - shortfall,
            maximum_geometry_queries=0,
        )
        expectation = pytest.raises(MeshcoreError) if shortfall else nullcontext()
        with expectation as caught, NativeExecutionScope(bounded) as scope:
            with NativeHostStorageWorkspace(scope) as storage:
                storage.set_bound(4096 * len(dataclasses.fields(source)))
                binding = owner._AdaptationRootBinding(
                    source, ("execution_evidence",), scope
                )
                assert scope.remaining().remaining_work_units == 0
        if shortfall:
            assert caught is not None
            assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            continue

        def forbid_discovery(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Ended root binding must not traverse or fingerprint.")

        with monkeypatch.context() as patch:
            patch.setattr(jax.tree_util, "tree_flatten_with_path", forbid_discovery)
            patch.setattr(owner, "canonical_fingerprint", forbid_discovery)
            result = binding.bind(source.execution_evidence)
        assert result.result_id == source.result_id
        assert result.mesh is source.mesh
        assert result.geometry is source.geometry
        assert result.execution_evidence is source.execution_evidence
        assert scope.completed_evidence().externally_charged_geometry_queries == 0

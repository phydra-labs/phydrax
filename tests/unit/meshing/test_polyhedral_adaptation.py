#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import cell_geometry_preparation_query_upper
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
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
    original_id = source.mesh_id
    original_coordinates = np.asarray(source.coordinates).copy()
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
    with pytest.raises(MeshingFailure) as failure:
        adapt_polyhedral_mesh(source, request, maximum_scratch_bytes=1)
    assert failure.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.value.evidence.provider_code == "scratch_byte_budget"
    assert source.mesh_id == original_id
    np.testing.assert_array_equal(source.coordinates, original_coordinates)


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
        match=r"stored_fields\.CommonRefinementEvidence.*maximum_relative_source_defect",
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

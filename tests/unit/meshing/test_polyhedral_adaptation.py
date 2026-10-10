#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from pathlib import Path

import numpy as np
import pytest

from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._unstructured_remap import (
    UnstructuredConservativeRemapPlan,
)
from phydrax.geometry._supermesh import CommonRefinementStatus
from phydrax.meshing import PeriodicConstraint
from phydrax.meshing._polyhedral_adaptation import (
    adapt_polyhedral_mesh,
    PolyhedralAdaptationError,
    PolyhedralAdaptationOperation,
    PolyhedralAdaptationOutcome,
    PolyhedralMeshAdaptation,
)
from phydrax.meshing._polyhedral_generation import PolyhedralConstruction
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.meshing._volume_generation import PiecewiseLinearComplex


def _boxes(xs: tuple[float, ...]) -> CellMesh:
    topology = reference_cell_topology("hexahedron")
    coordinates = np.asarray(
        [[x, y, z] for x in xs for z in (0.0, 1.0) for y in (0.0, 1.0)], dtype=np.float64
    )
    cells = []
    for interval in range(len(xs) - 1):
        vertices = [
            4 * (interval + int(x)) + 2 * int(z) + int(y) for x, y, z in topology.vertices
        ]
        cells.append(
            tuple(
                np.asarray([vertices[index] for index in face], dtype=np.int32)
                for face in topology.entities[2]
            )
        )
    return CellMesh.from_polyhedra(
        coordinates,
        cells,
        vertex_global_ids=np.arange(40, 40 + coordinates.shape[0], dtype=np.int64),
        cell_global_ids=np.arange(20, 20 + len(cells), dtype=np.int64),
    )


def _remap(
    source: CellMesh, target: CellMesh, outcome: PolyhedralAdaptationOutcome
) -> UnstructuredConservativeRemapPlan:
    common = outcome.common_refinement
    return UnstructuredConservativeRemapPlan(
        UnstructuredFiniteVolumePlan.from_cell_mesh(source).prepare(),
        UnstructuredFiniteVolumePlan.from_cell_mesh(target).prepare(),
        common.target_offsets,
        common.source_cells,
        common.volumes,
        method="common-refinement",
        provenance=common.refinement_id,
    )


def test_native_plane_split_closes_faces_and_transfers_fv_inventory() -> None:
    source = _boxes((0.0, 1.0))
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=[[1, 0, 0, 0.5]]
    )
    outcome = adapt_polyhedral_mesh(source, request, numeric_version="split")
    target, _, stencil = assemble_topology_edit(
        source, outcome.edit, numeric_version="split"
    )
    connectivity = target.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Plane splitting must preserve polyhedral connectivity.")
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    assert outcome.common_refinement.target_mesh_id == target.mesh_id
    assert connectivity.cell_count == 2
    assert np.count_nonzero(np.asarray(connectivity.face_neighbor) >= 0) == 1
    assert stencil is None
    plan = _remap(source, target, outcome)
    averages = np.asarray(plan.apply(np.asarray([3.0], dtype=np.float64)))
    assert np.allclose(averages, 3.0, rtol=0, atol=1e-13)
    assert np.dot(
        averages, np.asarray(outcome.common_refinement.target_measures)
    ) == pytest.approx(3.0, abs=1e-12)


def test_plane_split_vertex_cap_retains_accepted_geometry_and_topology() -> None:
    source = _boxes((0.0, 1.0))
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.PLANE_SPLIT, planes=[[1, 0, 0, 0.5]]
    )
    with pytest.raises(PolyhedralAdaptationError, match="vertex budget"):
        adapt_polyhedral_mesh(
            source, request, maximum_vertices=source.coordinates.shape[0]
        )
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)


def test_connected_agglomeration_removes_only_interior_faces_and_conserves() -> None:
    source = _boxes((0.0, 0.5, 1.0))
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=((20, 21),)
    )
    outcome = adapt_polyhedral_mesh(source, request, numeric_version="coarse")
    target, _, _ = assemble_topology_edit(source, outcome.edit, numeric_version="coarse")
    assert target.connectivity.cell_count == 1
    plan = _remap(source, target, outcome)
    assert np.allclose(
        plan.apply(np.asarray([2.0, 4.0], dtype=np.float64)), [3.0], atol=1e-13
    )
    assert np.asarray(outcome.common_refinement.target_measures)[0] == pytest.approx(
        1.0, abs=1e-12
    )


def test_exact_agglomeration_keeps_source_rows_after_interior_vertex_removal() -> None:
    from phydrax.meshing._polyhedral_generation import generate_polyhedral_volume

    topology = reference_cell_topology("hexahedron")
    complex_ = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    construction = generate_polyhedral_volume(
        complex_,
        sites=np.asarray(
            [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)],
            dtype=np.float64,
        ),
    )
    source = construction.mesh
    connectivity = source.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Exact agglomeration requires polyhedral connectivity.")
    ids = tuple(int(value) for value in np.asarray(connectivity.cell_global_ids))
    outcome = adapt_polyhedral_mesh(
        source,
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=(ids,)
        ),
        source_geometry=construction.geometry,
        numeric_version="exact-agglomeration",
    )
    target, _, _ = assemble_topology_edit(
        source, outcome.edit, numeric_version="exact-agglomeration"
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
    plan = _remap(source, target, outcome)
    np.testing.assert_allclose(plan.apply(np.ones(len(ids))), [1.0], rtol=0, atol=1e-13)


def test_regeneration_uses_nonnested_overlap_not_site_parent_interpolation() -> None:
    source = _boxes((0.0, 0.5, 1.0))
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
    target, _, stencil = assemble_topology_edit(
        source, outcome.edit, numeric_version="regenerated"
    )
    assert stencil is None
    assert outcome.construction is not None
    assert outcome.common_refinement.target_mesh_id == target.mesh_id
    counts = np.diff(np.asarray(outcome.common_refinement.target_offsets))
    assert np.all(counts >= 2)
    plan = _remap(source, target, outcome)
    averages = np.asarray(plan.apply(np.asarray([2.0, 4.0], dtype=np.float64)))
    assert np.allclose(averages, 3.0, rtol=0, atol=1e-12)
    assert np.dot(
        averages, np.asarray(outcome.common_refinement.target_measures)
    ) == pytest.approx(3.0, abs=1e-12)


def test_region_crossing_agglomeration_and_feature_removal_rollback() -> None:
    source = _boxes((0.0, 0.5, 1.0))
    connectivity = source.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("The adaptation source must have polyhedral connectivity.")
    original = source.mesh_id
    request = PolyhedralMeshAdaptation(
        PolyhedralAdaptationOperation.AGGLOMERATE, agglomerations=((20, 21),)
    )
    with pytest.raises(PolyhedralAdaptationError, match="class"):
        adapt_polyhedral_mesh(source, request, cell_classes=[0, 1])
    interior = np.flatnonzero(np.asarray(connectivity.face_neighbor) >= 0)
    face_ids = np.asarray(connectivity.face_global_ids)[interior]
    with pytest.raises(PolyhedralAdaptationError, match="protected"):
        adapt_polyhedral_mesh(source, request, protected_face_ids=face_ids)
    assert source.mesh_id == original


def test_regeneration_native_allocation_refusal_preserves_source() -> None:
    from phydrax.meshing import MeshingFailure, MeshingFailureCategory

    source = _boxes((0.0, 1.0))
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


def test_periodic_regeneration_restart_preserves_authored_constraint_controls(
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

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
    with pytest.raises(ValueError, match="belong to site regeneration"):
        PolyhedralMeshAdaptation(
            PolyhedralAdaptationOperation.PLANE_SPLIT,
            planes=[[1, 0, 0, 0.5]],
            periodic_constraints=(constraint,),
        )


def _periodic_adaptation_source() -> tuple[
    PiecewiseLinearComplex,
    tuple[PeriodicConstraint, ...],
    PolyhedralConstruction,
]:
    import phydrax as phx
    from phydrax.meshing._polyhedral_generation import generate_polyhedral_volume

    M = phx.meshing
    points = np.asarray(
        [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float64
    )
    loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    domain = PiecewiseLinearComplex(
        points, loops, np.arange(6), [(-1, 0)] * 6, ("material-0",)
    )
    controls = []
    for axis, first, second in ((0, 4, 5), (1, 2, 3), (2, 0, 1)):
        source = M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            2,
            f"source-{axis}",
            [first],
        )
        target = M.MeshingScope(
            "polyhedral-domain",
            "r1",
            M.MeshingEntityKind.GEOMETRY,
            2,
            f"target-{axis}",
            [second],
        )
        matrix = np.eye(4)
        matrix[axis, 3] = 1
        controls.append(
            M.PeriodicConstraint(
                source,
                target,
                matrix,
                tolerance=0,
                source_entity_ids=np.asarray([first], dtype=np.int64),
                orientations=np.asarray([-1], dtype=np.int8),
            )
        )
    constraints = tuple(controls)
    construction = generate_polyhedral_volume(
        domain,
        sites=np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64),
        periodic_constraints=constraints,
    )
    return domain, constraints, construction


def test_periodic_regeneration_certifies_actual_new_source_and_rejects_forged_coverage(
    tmp_path: Path,
) -> None:
    import equinox as eqx

    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._topology_edit import (
        CellTopologyEdit,
        require_periodic_nonnested_geometry,
    )

    domain, constraints, construction = _periodic_adaptation_source()
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
    target, _, _ = assemble_topology_edit(
        source, outcome.edit, numeric_version="periodic-new-source"
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
    transferred = _remap(source, target, outcome).apply(np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(transferred, 3.0, rtol=0, atol=1e-12)
    assert source.periodic_topology is not None
    forged = eqx.tree_at(
        lambda common: common.volumes,
        authority.common_refinement,
        authority.common_refinement.volumes * 0.5,
    )
    with pytest.raises(ValueError, match="certify|payload"):
        require_periodic_nonnested_geometry(
            source.periodic_topology, target, authority._replace(common_refinement=forged)
        )
    changed_geometry = eqx.tree_at(
        lambda geometry: geometry.coordinates,
        authority.target_geometry,
        authority.target_geometry.coordinates + 0.125,
    )
    with pytest.raises(ValueError):
        require_periodic_nonnested_geometry(
            source.periodic_topology,
            target,
            authority._replace(target_geometry=changed_geometry),
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
    cold_target, _, _ = assemble_topology_edit(
        cold_authority.source,
        cold_edit,
        numeric_version=cold_authority.target.numeric_version,
    )
    assert cold_target.mesh_id == target.mesh_id
    np.testing.assert_array_equal(cold_target.coordinates, target.coordinates)
    np.testing.assert_array_equal(
        cold_authority.common_refinement.volumes, outcome.common_refinement.volumes
    )
    from phydrax.meshing._periodic import bind_periodic_topology_edit

    other_source_epoch = source.with_coordinates(
        source.coordinates, numeric_version="other-source-epoch"
    )
    np.testing.assert_array_equal(other_source_epoch.coordinates, source.coordinates)
    np.testing.assert_array_equal(
        other_source_epoch.vertex_global_ids, source.vertex_global_ids
    )
    with pytest.raises(ValueError, match="numeric/scientific source epoch"):
        bind_periodic_topology_edit(
            other_source_epoch, target, outcome.edit.periodic_orbits
        )
    other_target_epoch = target.with_coordinates(
        target.coordinates, numeric_version="other-target-epoch"
    )
    np.testing.assert_array_equal(other_target_epoch.coordinates, target.coordinates)
    np.testing.assert_array_equal(
        other_target_epoch.vertex_global_ids, target.vertex_global_ids
    )
    with pytest.raises(ValueError, match="stale mesh/geometry/scientific"):
        require_periodic_nonnested_geometry(
            source.periodic_topology, other_target_epoch, authority
        )


def test_periodic_plane_split_and_connected_agglomeration_preserve_exact_source_and_fv_inventory() -> (
    None
):
    _, _, construction = _periodic_adaptation_source()
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
    fine, _, trace = assemble_topology_edit(
        source, refined.edit, numeric_version="periodic-split"
    )
    assert trace is not None
    source_x = np.asarray(source.coordinates)[:, 0]
    nodal = np.asarray(
        trace.apply(source.vertex_global_ids, 0.5 * np.minimum(source_x, 1 - source_x))
    )
    fine_x = np.asarray(fine.coordinates)[:, 0]
    # This independent periodic tent field is affine on each original cell's
    # source edges, including any source-authored seam subdivision vertices.
    np.testing.assert_allclose(
        nodal, 0.5 * np.minimum(fine_x, 1 - fine_x), rtol=0, atol=1e-13
    )
    assert fine.periodic_topology is not None
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id
    from phydrax.discretization._transfer import TransferGeometryBinding
    from phydrax.discretization.vem._polyhedral import (
        prepare_polyhedral_h1_virtual_element_3d,
    )

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
        atol=1e-13,
    )
    assert fine.connectivity.cell_count > source.connectivity.cell_count
    values = np.asarray(_remap(source, fine, refined).apply(np.asarray([2.0, 4.0])))
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
    target, _, restriction = assemble_topology_edit(
        fine, coarse.edit, numeric_version="periodic-agglomerated"
    )
    assert restriction is not None
    current_x = np.asarray(target.coordinates)[:, 0]
    np.testing.assert_allclose(
        restriction.apply(fine.vertex_global_ids, nodal),
        0.5 * np.minimum(current_x, 1 - current_x),
        rtol=0,
        atol=1e-13,
    )
    assert target.periodic_topology is not None
    assert target.periodic_topology.actual_geometry is not None
    assert target.connectivity.cell_count == source.connectivity.cell_count
    restored = np.asarray(_remap(fine, target, coarse).apply(values))
    np.testing.assert_allclose(restored, [2.0, 4.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.dot(restored, np.asarray(coarse.common_refinement.target_measures)),
        np.dot(
            np.asarray([2.0, 4.0]), np.asarray(refined.common_refinement.source_measures)
        ),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("change", ["domain", "group", "source-revision"])
def test_periodic_regeneration_rejects_changed_original_authority_atomically(
    change: str,
) -> None:
    import phydrax as phx

    M = phx.meshing
    domain, constraints, construction = _periodic_adaptation_source()
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
    with pytest.raises(
        ValueError, match="group|domain/facets/materials/periodic controls"
    ):
        adapt_polyhedral_mesh(
            source,
            request,
            source_geometry=construction.geometry,
            numeric_version="rejected-authority",
        )
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)


def test_crossing_plane_cut_does_not_claim_unknown_virtual_face_trace() -> None:
    from phydrax.meshing._polyhedral_generation import generate_polyhedral_volume

    topology = reference_cell_topology("hexahedron")
    domain = PiecewiseLinearComplex(
        np.asarray(topology.vertices, dtype=np.float64),
        topology.entities[2],
        np.zeros(6, dtype=np.int64),
        np.asarray([[-1, 0]], dtype=np.int64),
        ("material",),
    )
    construction = generate_polyhedral_volume(
        domain, sites=[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]
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
    _, _, nodal = assemble_topology_edit(
        source, outcome.edit, numeric_version="unknown-face-trace"
    )
    # The crossing plane creates nodes inside original source faces. Their
    # coordinate convex coefficients do not evaluate a virtual H1 field.
    assert nodal is None
    assert outcome.common_refinement.status is CommonRefinementStatus.SUCCESS
    assert source.mesh_id == identity


def test_polyhedral_edit_rejects_foreign_exact_geometry_before_changing_source() -> None:
    from phydrax.meshing._polyhedral_generation import generate_polyhedral_volume

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
    construction = generate_polyhedral_volume(domain, sites=[[0.5, 0.5, 0.5]])
    foreign = generate_polyhedral_volume(other_domain, sites=[[3.5, 0.5, 0.5]])
    source = construction.mesh
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    with pytest.raises(ValueError, match="carrier differs"):
        adapt_polyhedral_mesh(
            source,
            PolyhedralMeshAdaptation(
                PolyhedralAdaptationOperation.PLANE_SPLIT, planes=[[1, 0, 0, 0.25]]
            ),
            source_geometry=foreign.geometry,
        )
    assert source.mesh_id == identity
    np.testing.assert_array_equal(source.coordinates, coordinates)

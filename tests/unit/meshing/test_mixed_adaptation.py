#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from pathlib import Path

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    CellGeometryTransitionError,
    reconstruct_parametric_surface_cell_geometry,
    transition_nested_cell_geometry,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import FiniteElementSpec
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._mixed_adaptation import (
    adapt_mixed_mesh,
    MixedAdaptationEvidence,
    MixedAdaptationHierarchy,
    MixedLayerColumns,
    MixedTemplateClosureError,
)
from phydrax.meshing._topology_edit import assemble_topology_edit


def _curved_cell(kind: str) -> tuple[CellMesh, CellGeometrySpec]:
    corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    corners[:, 1] += (
        0.05
        * corners[:, 0]
        * (1 - corners[:, 0] if corners.shape[1] == 2 else corners[:, 2])
    )
    mesh = CellMesh(
        corners,
        (
            CellBlock(
                "volume",
                kind,
                np.arange(corners.shape[0], dtype=np.int32)[None],
                global_ids=np.asarray([17], dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(100, 100 + corners.shape[0], dtype=np.int64),
    )
    element = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[:, 1] += (
        0.05 * nodes[:, 0] * (1 - nodes[:, 0] if nodes.shape[1] == 2 else nodes[:, 2])
    )
    geometry = CellGeometrySpec(
        {"volume": element},
        {"volume": np.arange(nodes.shape[0], dtype=np.int32)[None]},
        nodes,
    )
    return mesh, geometry


@pytest.mark.parametrize(
    "kind", ["quadrilateral", "hexahedron", "prism", "pyramid", "tetrahedron"]
)
def test_curved_template_restriction_and_complete_parent_restore(kind: str) -> None:
    source, geometry = _curved_cell(kind)
    before = np.asarray(geometry.coordinates).copy()
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    target, _, stencil = assemble_topology_edit(
        source, refined.edit, numeric_version="refined"
    )
    transition = transition_nested_cell_geometry(
        source,
        geometry,
        target,
        CellGeometrySpec.affine(target),
        refinement=refined.edit.refinement,
    )
    assert transition.evidence.exact
    assert stencil is not None
    probe = np.asarray([[0.25, 0.25, 0.2]], dtype=np.float64)[
        :, : source.topological_dimension
    ]
    assert refined.edit.refinement is not None
    witnesses = dict(
        zip(
            refined.edit.refinement.fine_cell_ids,
            refined.edit.refinement.fine_reference_vertices,
            strict=True,
        )
    )
    reference = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    # Children are exact restrictions of the source basis, never child nodal
    # interpolants: the parent element and its coefficients are retained
    # bitwise, and only the affine child reference action is new.
    np.testing.assert_array_equal(np.asarray(transition.geometry.coordinates), before)
    for block, element, route in zip(
        target.blocks,
        transition.geometry.elements,
        transition.geometry.geometry_dofs,
        strict=True,
    ):
        assert isinstance(element, RestrictedCellGeometryElement)
        assert element.source_element.element_id == geometry.elements[0].element_id
        assert element.cell_kind == block.cell_kind
        values, _ = element.tabulate(probe)
        result = (
            np.asarray(values)
            @ np.asarray(transition.geometry.coordinates)[np.asarray(route)[0]]
        )
        chart, _, _, _ = np.linalg.lstsq(
            np.column_stack((reference, np.ones(reference.shape[0], dtype=np.float64))),
            witnesses[np.asarray(block.global_ids)[0]][: reference.shape[0]],
            rcond=None,
        )
        parent = (
            np.column_stack((probe, np.ones(probe.shape[0], dtype=np.float64))) @ chart
        )
        expected = parent.copy()
        expected[:, 1] += (
            0.05
            * parent[:, 0]
            * (1 - parent[:, 0] if parent.shape[1] == 2 else parent[:, 2])
        )
        assert np.allclose(result, expected, rtol=0, atol=2e-14)
    source_ids = np.concatenate([np.asarray(block.global_ids) for block in target.blocks])
    coarse = adapt_mixed_mesh(
        target,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=source_ids,
        hierarchy=refined.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(
        target, coarse.edit, numeric_version="restored"
    )
    restored_map = transition_nested_cell_geometry(
        target,
        transition.geometry,
        restored,
        CellGeometrySpec.affine(restored),
        refinement=coarse.edit.refinement,
        coarsening=coarse.edit.coarsening,
    )
    assert np.array_equal(np.asarray(restored.blocks[0].global_ids), [17])
    assert np.array_equal(
        np.asarray(restored.vertex_global_ids), np.asarray(source.vertex_global_ids)
    )
    assert np.array_equal(np.asarray(restored_map.geometry.coordinates), before)
    assert np.array_equal(np.asarray(geometry.coordinates), before)
    assert restored_map.evidence.exact


def test_hard_first_interval_allows_tangential_but_forbids_axial_change() -> None:
    mesh, _ = _curved_cell("prism")
    columns = MixedLayerColumns(
        np.asarray([17], dtype=np.int64),
        np.asarray([4], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        axial_refinement=True,
        hard_first_thickness=True,
    )
    outcome = adapt_mixed_mesh(
        mesh, refine_cell_ids=np.asarray([17], dtype=np.int64), layer_columns=columns
    )
    assert outcome.edit.refinement is not None
    corners = outcome.edit.refinement.fine_reference_vertices[:, :6]
    assert np.all(np.min(corners[..., 2], axis=1) == 0)
    assert np.all(np.max(corners[..., 2], axis=1) == 1)
    assert outcome.evidence.schedule_changed_cell_ids.size == 0


def test_archived_layer_hierarchy_restores_parent_and_first_interval(
    tmp_path: Path,
) -> None:
    source, _ = _curved_cell("prism")
    columns = MixedLayerColumns(
        np.asarray([17], dtype=np.int64),
        np.asarray([4], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        axial_refinement=True,
        hard_first_thickness=True,
    )
    refined = adapt_mixed_mesh(
        source,
        refine_cell_ids=np.asarray([17], dtype=np.int64),
        layer_columns=columns,
    )
    fine, _, _ = assemble_topology_edit(
        source, refined.edit, numeric_version="archive-fine"
    )
    archive = tmp_path / "layer-hierarchy"
    receipt = write_meshing_source_closure(archive, refined.hierarchy)
    hierarchy = read_meshing_source_closure(
        archive, expected_content_id=receipt.content_id
    )
    if not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError("The restored hierarchy lost its owning type.")
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarse = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=hierarchy,
    )
    target, _, _ = assemble_topology_edit(
        fine, coarse.edit, numeric_version="archive-coarse"
    )
    np.testing.assert_array_equal(
        target.blocks[0].global_ids, source.blocks[0].global_ids
    )
    np.testing.assert_array_equal(target.vertex_global_ids, source.vertex_global_ids)
    np.testing.assert_array_equal(target.coordinates, source.coordinates)
    repeated = adapt_mixed_mesh(
        target,
        refine_cell_ids=np.asarray([17], dtype=np.int64),
        hierarchy=coarse.hierarchy,
    )
    if repeated.edit.refinement is None:
        raise ValueError("The restored layer failed to produce its refinement witnesses.")
    corners = repeated.edit.refinement.fine_reference_vertices[:, :6]
    assert np.all(np.min(corners[..., 2], axis=1) == 0)
    assert np.all(np.max(corners[..., 2], axis=1) == 1)
    assert repeated.evidence.schedule_changed_cell_ids.size == 0


@pytest.mark.parametrize("failure", ["partial", "protected", "missing-ancestry"])
def test_layer_column_coarsening_rejects_partial_interval_cohort(failure: str) -> None:
    corners = np.asarray(reference_cell_topology("prism").vertices, dtype=np.float64)
    points = np.concatenate((corners, corners + np.asarray([3.0, 0.0, 0.0])))
    source = CellMesh(
        points,
        (
            CellBlock(
                "layers",
                "prism",
                np.arange(12, dtype=np.int32).reshape(2, 6),
                global_ids=np.asarray([17, 18], dtype=np.int64),
            ),
        ),
    )
    columns = MixedLayerColumns(
        np.asarray([17, 18], dtype=np.int64),
        np.asarray([4, 4], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int32),
        axial_refinement=False,
        hard_first_thickness=True,
    )
    refined = adapt_mixed_mesh(
        source, refine_cell_ids=np.asarray([17]), layer_columns=columns
    )
    fine, _, _ = assemble_topology_edit(
        source, refined.edit, numeric_version="column-fine"
    )
    first, second = refined.hierarchy.records
    first_ids = np.asarray(first.child_ids, dtype=np.int64)
    all_ids = np.concatenate((first_ids, np.asarray(second.child_ids, dtype=np.int64)))
    marks = first_ids if failure == "partial" else all_ids
    hierarchy = refined.hierarchy
    if failure == "missing-ancestry":
        hierarchy = MixedAdaptationHierarchy(
            (first,),
            next_cell_id=hierarchy.next_cell_id,
            next_vertex_id=hierarchy.next_vertex_id,
            retired_entities=hierarchy.retired_entities,
            quotient_entities=hierarchy.quotient_entities,
            layer_columns=hierarchy.layer_columns,
            identity_cells=hierarchy.identity_cells,
        )
    outcome = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=marks,
        hierarchy=hierarchy,
        protected_cell_ids=np.asarray([second.child_ids[0]], dtype=np.int64)
        if failure == "protected"
        else None,
    )
    retained, _, _ = assemble_topology_edit(
        fine, outcome.edit, numeric_version="column-retained"
    )
    assert not outcome.evidence.coarsened_cell_ids.size
    np.testing.assert_array_equal(
        outcome.evidence.rejected_coarsening_ids, np.sort(marks)
    )
    assert retained.topology_id == fine.topology_id
    complete = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=all_ids,
        hierarchy=refined.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(
        fine, complete.edit, numeric_version="column-restored"
    )
    assert restored.topology_id == source.topology_id
    assert complete.hierarchy.layer_columns is not None
    np.testing.assert_array_equal(
        complete.hierarchy.layer_columns.interval_indices, [0, 1]
    )


def test_explicit_first_interval_schedule_revision_is_evidenced() -> None:
    mesh, _ = _curved_cell("prism")
    columns = MixedLayerColumns(
        np.asarray([17], dtype=np.int64),
        np.asarray([4], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        axial_refinement=True,
        hard_first_thickness=True,
        allow_schedule_change=True,
    )
    outcome = adapt_mixed_mesh(
        mesh, refine_cell_ids=np.asarray([17], dtype=np.int64), layer_columns=columns
    )
    assert np.array_equal(outcome.evidence.schedule_changed_cell_ids, [17])
    assert outcome.edit.refinement is not None
    assert np.all(
        np.ptp(outcome.edit.refinement.fine_reference_vertices[:, :6, 2], axis=1) == 0.5
    )


def test_incomplete_siblings_and_protected_refinement_preserve_source() -> None:
    mesh, _ = _curved_cell("hexahedron")
    before = np.asarray(mesh.coordinates).copy()
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray([17], dtype=np.int64))
    target, _, _ = assemble_topology_edit(mesh, outcome.edit, numeric_version="refined")
    marked = np.asarray([target.blocks[0].global_ids[0]], dtype=np.int64)
    incomplete = adapt_mixed_mesh(
        target,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=marked,
        hierarchy=outcome.hierarchy,
    )
    assert np.array_equal(incomplete.evidence.rejected_coarsening_ids, marked)
    with pytest.raises(MixedTemplateClosureError, match="protected"):
        adapt_mixed_mesh(
            mesh,
            refine_cell_ids=np.asarray([17], dtype=np.int64),
            protected_cell_ids=np.asarray([17], dtype=np.int64),
        )
    assert np.array_equal(np.asarray(mesh.coordinates), before)


def test_hex_pyramid_shared_quadrilateral_has_complete_oriented_closure() -> None:
    cube = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    points = np.concatenate((cube, np.asarray([[0.5, 0.5, 2]], dtype=np.float64)))
    points[:, 1] += 0.05 * points[:, 0] * points[:, 2]
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "hex",
                "hexahedron",
                np.arange(8, dtype=np.int32)[None],
                global_ids=np.asarray([10], dtype=np.int64),
            ),
            CellBlock(
                "transition",
                "pyramid",
                np.asarray([[4, 5, 6, 7, 8]], dtype=np.int32),
                global_ids=np.asarray([11], dtype=np.int64),
            ),
        ),
    )
    elements, routes, coefficients = {}, {}, []
    offset = 0
    for block in mesh.blocks:
        element = coordinate_lagrange_element(block.cell_kind, 2)
        nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
        if block.cell_kind == "pyramid":
            nodes[:, 2] += 1.0
        nodes[:, 1] += 0.05 * nodes[:, 0] * nodes[:, 2]
        elements[block.name] = element
        routes[block.name] = np.arange(offset, offset + nodes.shape[0], dtype=np.int32)[
            None
        ]
        coefficients.append(nodes)
        offset += nodes.shape[0]
    geometry = CellGeometrySpec(elements, routes, np.concatenate(coefficients))
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray([10], dtype=np.int64))
    target, _, _ = assemble_topology_edit(mesh, outcome.edit, numeric_version="closed")
    restriction = transition_nested_cell_geometry(
        mesh,
        geometry,
        target,
        CellGeometrySpec.affine(target),
        refinement=outcome.edit.refinement,
    )
    assert restriction.evidence.exact
    assert restriction.evidence.target_measure == pytest.approx(4.0 / 3.0, abs=1e-11)
    connectivity = target.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    offsets = np.asarray(connectivity.face_vertex_offsets)
    vertices = np.asarray(connectivity.face_vertex_values)
    cap_faces = np.asarray(
        [
            np.all(
                np.asarray(target.coordinates)[
                    vertices[offsets[face] : offsets[face + 1]], 2
                ]
                == 1.0
            )
            for face in range(connectivity.face_count)
        ]
    )
    assert np.count_nonzero(cap_faces) == 4
    assert np.all(np.asarray(connectivity.face_neighbor)[cap_faces] >= 0)
    assert set(block.cell_kind for block in target.blocks) == {"hexahedron", "pyramid"}
    assert np.array_equal(outcome.evidence.closure_cell_ids, [11])
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in target.blocks])
    coarse = adapt_mixed_mesh(
        target,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=outcome.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(target, coarse.edit, numeric_version="coarse")
    assert set(
        int(value) for block in restored.blocks for value in np.asarray(block.global_ids)
    ) == {10, 11}
    restoration = transition_nested_cell_geometry(
        target,
        restriction.geometry,
        restored,
        CellGeometrySpec.affine(restored),
        refinement=coarse.edit.refinement,
        coarsening=coarse.edit.coarsening,
    )
    assert restoration.evidence.exact
    assert np.array_equal(
        np.asarray(restoration.geometry.coordinates), np.asarray(geometry.coordinates)
    )


def _red_shared_face_source(
    mixed: bool, permuted: bool
) -> tuple[CellMesh, CellGeometrySpec]:
    from phydrax.meshing._canonical import canonicalize_cell_mesh

    if mixed:
        points = np.concatenate(
            (
                np.asarray(reference_cell_topology("prism").vertices, dtype=np.float64),
                np.asarray(((0.0, 0.0, 2.0),), dtype=np.float64),
            )
        )
        blocks = (
            CellBlock(
                "prism",
                "prism",
                np.arange(6, dtype=np.int32)[None],
                global_ids=np.asarray((11,), dtype=np.int64),
            ),
            CellBlock(
                "cap",
                "tetrahedron",
                np.asarray(((3, 4, 5, 6),), dtype=np.int32),
                global_ids=np.asarray((37,), dtype=np.int64),
            ),
        )
    else:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ),
            dtype=np.float64,
        )
        cells = ((1, 2, 0, 3), (2, 1, 0, 4)) if permuted else ((0, 1, 2, 3), (0, 2, 1, 4))
        blocks = (
            CellBlock(
                "tets",
                "tetrahedron",
                np.asarray(cells, dtype=np.int32),
                global_ids=np.asarray((11, 37), dtype=np.int64),
            ),
        )
    original = points.copy()
    points[:, 1] += points[:, 0] * points[:, 2] / 16
    mesh = CellMesh(
        points,
        blocks,
        vertex_global_ids=np.asarray((103, 71, 509, 211, 13, 17, 401), dtype=np.int64)[
            : points.shape[0]
        ],
    )
    mesh = canonicalize_cell_mesh(mesh)
    elements, routes, coefficients = {}, {}, []
    cursor = 0
    for block in mesh.blocks:
        element = coordinate_lagrange_element(block.cell_kind, 2)
        nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        local = []
        for row in np.asarray(block.vertices):
            corners = original[row]
            matrix = (
                (corners[1:] - corners[0])
                if block.cell_kind == "tetrahedron"
                else np.stack(
                    (
                        corners[1] - corners[0],
                        corners[2] - corners[0],
                        corners[3] - corners[0],
                    )
                )
            )
            values = nodes @ matrix + corners[0]
            values[:, 1] += values[:, 0] * values[:, 2] / 16
            local.append(np.arange(cursor, cursor + nodes.shape[0], dtype=np.int32))
            coefficients.append(values)
            cursor += nodes.shape[0]
        elements[block.name] = element
        routes[block.name] = np.stack(local)
    return mesh, CellGeometrySpec(elements, routes, np.concatenate(coefficients))


@pytest.mark.parametrize(
    ("mixed", "permuted"), ((False, False), (False, True), (True, False))
)
def test_red_tetrahedral_shared_face_and_mixed_interface_lifecycle(
    mixed: bool, permuted: bool
) -> None:
    import phydrax as phx

    M = phx.meshing
    mesh, geometry = _red_shared_face_source(mixed, permuted)
    source = M.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    limits = M.MeshingLimits(maximum_cells=100_000, maximum_work_units=20_000_000)
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_MIXED, limits=limits)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(np.asarray((11, 37), dtype=np.int64)),
            policy=policy,
        )
    )

    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    assert source.audit.passed and refined.target.audit.passed
    assert isinstance(refined.evidence, MixedAdaptationEvidence)
    assert 0 < refined.evidence.work_units <= limits.maximum_work_units
    execution = refined.evidence.execution_evidence
    assert execution is not None
    assert int(np.asarray(execution.work)[0]) >= refined.evidence.work_units
    assert int(np.asarray(execution.host_storage_peak_bytes_upper)) > 0
    assert (
        int(np.asarray(execution.host_storage_peak_bytes_upper))
        <= limits.maximum_scratch_bytes
    )
    np.testing.assert_array_equal(refined.evidence.refined_cell_ids, (11, 37))
    assert refined.lineage is not None and refined.transition is not None
    assert refined.lineage.source_topology_id == source.mesh.topology_id
    assert refined.lineage.target_topology_id == refined.target.mesh.topology_id
    assert (
        refined.geometry_transition is not None
        and refined.geometry_transition.evidence.exact
    )
    assert isinstance(refined.hierarchy, MixedAdaptationHierarchy)
    assert all(len(record.child_ids) == 8 for record in refined.hierarchy.records)
    fine = refined.target.mesh
    assert fine.entity_set(3).count == 16
    connectivity = fine.connectivity
    if isinstance(connectivity, TetrahedralConnectivity):
        rows = np.asarray(connectivity.faces)
    elif isinstance(connectivity, PolyhedralConnectivity):
        offsets, vertices = (
            np.asarray(connectivity.face_vertex_offsets),
            np.asarray(connectivity.face_vertex_values),
        )
        rows = np.asarray(
            [
                vertices[start:stop]
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
                if stop - start == 3
            ],
            dtype=np.int64,
        )
    else:
        raise TypeError(
            "The red mixed consumer lost canonical triangular face incidence."
        )
    height = 1.0 if mixed else 0.0
    shared = np.all(np.asarray(fine.coordinates)[rows, 2] == height, axis=1)
    assert np.count_nonzero(shared) == 4
    if isinstance(connectivity, TetrahedralConnectivity):
        assert np.all(~np.asarray(connectivity.boundary_faces)[shared])
    else:
        triangular = np.diff(np.asarray(connectivity.face_vertex_offsets)) == 3
        assert np.all(np.asarray(connectivity.face_neighbor)[triangular][shared] >= 0)

    # Consume real central-child connectivity. Its common edge must be the
    # lexicographically first pair of opposite source midpoint-edge identities,
    # even when the original local tetrahedron numbering is permuted.
    fine_cells = {
        int(identifier): np.asarray(fine.vertex_global_ids)[row]
        for block in fine.blocks
        for identifier, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    fine_positions = {
        int(identifier): point
        for identifier, point in zip(
            np.asarray(fine.vertex_global_ids), np.asarray(fine.coordinates), strict=True
        )
    }
    source_positions = {
        int(identifier): point
        for identifier, point in zip(
            np.asarray(mesh.vertex_global_ids), np.asarray(mesh.coordinates), strict=True
        )
    }
    for record in refined.hierarchy.records:
        if record.cell_kind != "tetrahedron":
            continue
        parent = record.parent_vertices
        candidates = tuple(
            tuple(
                sorted(
                    (
                        tuple(sorted((parent[a], parent[b]))),
                        tuple(sorted((parent[c], parent[d]))),
                    )
                )
            )
            for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
        )
        expected_keys = min(candidates)
        central = [
            set(fine_cells[child].tolist())
            for child in record.child_ids
            if not set(fine_cells[child].tolist()) & set(parent)
        ]
        assert len(central) == 4
        diagonal = set.intersection(*central)
        assert len(diagonal) == 2
        # Pull back the independently authored shear to compare the actual
        # midpoint construction in the original affine reference partition.
        actual_points = [fine_positions[identifier].copy() for identifier in diagonal]
        for point in actual_points:
            point[1] -= point[0] * point[2] / 16
        # Source midpoint carriers of a curved edge need their exact pullback,
        # not the average of already curved endpoint coordinates.
        expected_points = []
        for first, second in expected_keys:
            ends = [source_positions[identifier].copy() for identifier in (first, second)]
            for point in ends:
                point[1] -= point[0] * point[2] / 16
            expected_points.append((ends[0] + ends[1]) / 2)
        assert sorted(tuple(point) for point in actual_points) == sorted(
            tuple(point) for point in expected_points
        )

    children = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarse = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            refined.target,
            M.MarkedMeshAdaptation((), children, hierarchy=refined.hierarchy),
            policy=policy,
        )
    )
    assert coarse.status is M.MeshAdaptationStatus.COMPLETE
    assert coarse.target.audit.passed
    assert isinstance(coarse.evidence, MixedAdaptationEvidence)
    removed_children = np.asarray(
        [child for record in refined.hierarchy.records for child in record.child_ids],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        coarse.evidence.coarsened_cell_ids, np.sort(removed_children)
    )
    assert not coarse.evidence.rejected_coarsening_ids.size
    assert (
        coarse.geometry_transition is not None
        and coarse.geometry_transition.evidence.exact
    )
    assert coarse.target.mesh.topology_id == source.mesh.topology_id
    np.testing.assert_array_equal(
        coarse.target.geometry.coordinates, source.geometry.coordinates
    )
    np.testing.assert_array_equal(
        coarse.target.mesh.vertex_global_ids, source.mesh.vertex_global_ids
    )
    np.testing.assert_array_equal(coarse.target.mesh.coordinates, source.mesh.coordinates)


@pytest.mark.parametrize("resource", ("cells", "work"))
def test_red_tetrahedral_original_resource_refusal_leaves_source_unchanged(
    resource: str,
) -> None:
    mesh, _ = _red_shared_face_source(False, False)
    coordinates = np.asarray(mesh.coordinates).copy()
    topology_id = mesh.topology_id
    with pytest.raises(MixedTemplateClosureError, match="budget"):
        adapt_mixed_mesh(
            mesh,
            refine_cell_ids=np.asarray((11, 37), dtype=np.int64),
            maximum_cells=15 if resource == "cells" else 100_000,
            maximum_work_units=1 if resource == "work" else 20_000_000,
        )
    assert mesh.topology_id == topology_id
    np.testing.assert_array_equal(mesh.coordinates, coordinates)


def test_red_tetrahedral_original_host_scratch_refusal_leaves_source_unchanged() -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus

    mesh, _ = _red_shared_face_source(False, False)
    coordinates = np.asarray(mesh.coordinates).copy()
    topology_id = mesh.topology_id
    with pytest.raises(MeshcoreError) as caught:
        adapt_mixed_mesh(
            mesh,
            refine_cell_ids=np.asarray((11, 37), dtype=np.int64),
            maximum_cells=100_000,
            maximum_work_units=20_000_000,
            maximum_scratch_bytes=1,
        )
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.memory_evidence is not None
    assert caught.value.memory_evidence.shape == (6,)
    assert mesh.topology_id == topology_id
    np.testing.assert_array_equal(mesh.coordinates, coordinates)


def _periodic_template_source(
    kind: str, rotated: bool
) -> tuple[CellMesh, CellGeometrySpec]:
    from phydrax.discretization import (
        PeriodicCell,
        PeriodicIsometryGroup,
        PeriodicMeshTopology,
    )

    points = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    identification: PeriodicCell | PeriodicIsometryGroup
    if rotated:
        generator = np.eye(4, dtype=np.float64)
        generator[:2, :2] = ((0, -1), (1, 0))
        identification = PeriodicIsometryGroup(generator[None])
        if kind == "pyramid":
            points[4, :2] = 0
        roots = {
            "tetrahedron": (0, 1, 1, 3),
            "prism": (0, 1, 1, 3, 4, 4),
            "hexahedron": (0, 1, 2, 1, 4, 5, 6, 5),
            "pyramid": (0, 1, 2, 1, 4),
        }[kind]
        shifts = np.asarray(
            [int(index != root) for index, root in enumerate(roots)], dtype=np.int32
        )[:, None]
    else:
        identification = PeriodicCell(np.eye(points.shape[1], dtype=np.float64))
        roots = (0,) * points.shape[0]
        shifts = points.astype(np.int32)
    block = CellBlock(
        "periodic",
        kind,
        np.arange(points.shape[0], dtype=np.int32)[None],
        global_ids=np.asarray([17], dtype=np.int64),
    )
    mesh = CellMesh(
        points,
        (block,),
        vertex_global_ids=np.arange(100, 100 + points.shape[0], dtype=np.int64),
    )
    periodic = PeriodicMeshTopology(
        mesh, identification, np.asarray(roots, dtype=np.int32), shifts
    )
    mesh = CellMesh(
        points,
        (block,),
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=periodic,
    )
    element = coordinate_lagrange_element(kind, 3)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    if rotated:
        if kind == "pyramid":
            nodes[:, :2] -= nodes[:, 2, None] / 2
        nodes[:, :2] *= 1 + 0.03 * nodes[:, 2, None] * (1 - nodes[:, 2, None])
    elif kind == "hexahedron":
        nodes[:, 2] += 0.03 * np.prod(nodes * (1 - nodes), axis=1)
    else:
        nodes[:, 1] += 0.03 * np.prod(nodes * (1 - nodes), axis=1)
    geometry = CellGeometrySpec(
        {"periodic": element},
        {"periodic": np.arange(nodes.shape[0], dtype=np.int32)[None]},
        nodes,
    )
    return mesh, geometry


@pytest.mark.parametrize(
    ("kind", "rotated"),
    (
        ("quadrilateral", False),
        ("hexahedron", False),
        ("tetrahedron", True),
        ("prism", True),
        ("hexahedron", True),
        ("pyramid", True),
    ),
    ids=(
        "quad-torus",
        "hex-torus",
        "rotated-tet",
        "rotated-prism",
        "rotated-hex",
        "rotated-pyramid",
    ),
)
def test_periodic_mixed_exact_source_and_scientific_topology_roundtrip(
    kind: str, rotated: bool
) -> None:
    source, geometry = _periodic_template_source(kind, rotated)
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    fine, _, _ = assemble_topology_edit(source, refined.edit, numeric_version="fine")
    restricted = transition_nested_cell_geometry(
        source,
        geometry,
        fine,
        CellGeometrySpec.affine(fine),
        refinement=refined.edit.refinement,
    )
    fine = fine.with_coordinates(restricted.vertex_coordinates, numeric_version="fine")
    assert restricted.evidence.exact
    fine_periodic = fine.periodic_topology
    assert fine_periodic is not None
    for lower, upper in zip(
        fine_periodic.quotient.incidences[:-1],
        fine_periodic.quotient.incidences[1:],
        strict=True,
    ):
        assert np.all((lower.scipy_boundary() @ upper.scipy_boundary()).data == 0)
    ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarsened = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=ids,
        hierarchy=refined.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(
        fine, coarsened.edit, numeric_version="restored"
    )
    restoration = transition_nested_cell_geometry(
        fine,
        restricted.geometry,
        restored,
        CellGeometrySpec.affine(restored),
        refinement=coarsened.edit.refinement,
        coarsening=coarsened.edit.coarsening,
    )
    assert restoration.evidence.exact
    assert restored.topology_id == source.topology_id
    source_periodic, restored_periodic = (
        source.periodic_topology,
        restored.periodic_topology,
    )
    assert source_periodic is not None and restored_periodic is not None
    assert restored_periodic.periodic_topology_id == source_periodic.periodic_topology_id
    np.testing.assert_array_equal(restoration.geometry.coordinates, geometry.coordinates)
    np.testing.assert_array_equal(restoration.vertex_coordinates, source.coordinates)
    for degree in range(1, source.topological_dimension):
        np.testing.assert_array_equal(
            restored_periodic.quotient.entities(degree).entity_ids,
            source_periodic.quotient.entities(degree).entity_ids,
        )


@pytest.mark.parametrize(
    "failure",
    ("partial-family", "work", "wrong-cycle", "stale-source-orbit", "stale-allocation"),
)
def test_periodic_mixed_refusal_retains_the_source_epoch(failure: str) -> None:
    source, _ = _periodic_template_source("hexahedron", False)
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    fine, _, _ = assemble_topology_edit(source, refined.edit, numeric_version="fine")
    before = np.asarray(fine.coordinates).copy()
    identity = fine.topology_id
    if failure == "partial-family":
        with pytest.raises(MixedTemplateClosureError, match="complete"):
            adapt_mixed_mesh(
                fine,
                refine_cell_ids=np.zeros(0, dtype=np.int64),
                coarsen_cell_ids=np.asarray(
                    [fine.blocks[0].global_ids[0]], dtype=np.int64
                ),
                hierarchy=refined.hierarchy,
            )
    elif failure == "work":
        with pytest.raises(MixedTemplateClosureError, match="work budget"):
            adapt_mixed_mesh(
                fine,
                refine_cell_ids=np.asarray(
                    [fine.blocks[0].global_ids[0]], dtype=np.int64
                ),
                hierarchy=refined.hierarchy,
                maximum_work_units=1,
            )
    elif failure == "wrong-cycle":
        witness = refined.edit.periodic_orbits
        assert witness is not None
        shifts = witness.vertex_shifts.copy()
        shifts[-1, 0] += 1
        invalid_edit = refined.edit._replace(
            periodic_orbits=witness._replace(vertex_shifts=shifts)
        )
        with pytest.raises(ValueError, match="representative|image|lift|shift"):
            assemble_topology_edit(source, invalid_edit, numeric_version="invalid")
    elif failure == "stale-source-orbit":
        record = refined.hierarchy.records[0]
        invalid_record = record._replace(
            periodic_roots=(record.periodic_roots[0] + 1, *record.periodic_roots[1:])
        )
        hierarchy = MixedAdaptationHierarchy(
            (invalid_record,),
            next_cell_id=refined.hierarchy.next_cell_id,
            next_vertex_id=refined.hierarchy.next_vertex_id,
            quotient_entities=refined.hierarchy.quotient_entities,
        )
        with pytest.raises(
            MixedTemplateClosureError, match="source-corner orbits changed"
        ):
            adapt_mixed_mesh(
                fine,
                refine_cell_ids=np.zeros(0, dtype=np.int64),
                coarsen_cell_ids=np.concatenate(
                    [np.asarray(block.global_ids) for block in fine.blocks]
                ),
                hierarchy=hierarchy,
            )
    else:
        banks = refined.hierarchy.quotient_entities
        invalid_bank = banks[0]._replace(allocator_next_id=banks[0].allocator_next_id + 1)
        hierarchy = MixedAdaptationHierarchy(
            refined.hierarchy.records,
            next_cell_id=refined.hierarchy.next_cell_id,
            next_vertex_id=refined.hierarchy.next_vertex_id,
            quotient_entities=(invalid_bank, *banks[1:]),
        )
        with pytest.raises(ValueError, match="allocation cursor"):
            adapt_mixed_mesh(
                fine,
                refine_cell_ids=np.zeros(0, dtype=np.int64),
                coarsen_cell_ids=np.concatenate(
                    [np.asarray(block.global_ids) for block in fine.blocks]
                ),
                hierarchy=hierarchy,
            )
    assert fine.topology_id == identity
    np.testing.assert_array_equal(fine.coordinates, before)


def test_inverse_periodic_mixed_coarsening_retains_explicit_hard_vertex_scope() -> None:
    import phydrax as phx

    M = phx.meshing
    mesh, _ = _periodic_template_source("hexahedron", False)
    source = M.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    refinement = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(np.asarray((17,), dtype=np.int64)),
            policy=M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_MIXED),
        )
    )
    fine = refinement.target
    born = np.setdiff1d(
        np.asarray(fine.mesh.vertex_global_ids), np.asarray(source.mesh.vertex_global_ids)
    )
    protected = M.MeshingScope(
        fine.mesh.mesh_id,
        fine.mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        0,
        fine.mesh.entity_set(0).entity_set_id,
        born[:1],
    )
    children = np.concatenate(
        [np.asarray(block.global_ids) for block in fine.mesh.blocks]
    )
    prepared = M.prepare_mesh_adaptation(
        fine,
        M.MarkedMeshAdaptation((), children, hierarchy=refinement.hierarchy),
        policy=M.MeshAdaptationPolicy(
            M.MeshAdaptationRoute.NATIVE_MIXED, protected_scopes=(protected,)
        ),
    )
    before = np.asarray(fine.mesh.coordinates).copy()
    with pytest.raises(MixedTemplateClosureError, match="complete"):
        M.execute_mesh_adaptation(prepared)
    np.testing.assert_array_equal(fine.mesh.coordinates, before)
    assert fine.mesh.topology_id == refinement.target.mesh.topology_id


def test_periodic_mixed_retired_identities_are_not_reused_after_restore() -> None:
    source, _ = _periodic_template_source("hexahedron", False)
    first = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    fine, _, _ = assemble_topology_edit(source, first.edit, numeric_version="fine")
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarsened = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=first.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(
        fine, coarsened.edit, numeric_version="restored"
    )
    second = adapt_mixed_mesh(
        restored,
        refine_cell_ids=np.asarray([17], dtype=np.int64),
        hierarchy=coarsened.hierarchy,
    )
    again, _, _ = assemble_topology_edit(restored, second.edit, numeric_version="again")
    repeated_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in again.blocks]
    )
    assert not np.intersect1d(repeated_ids, fine_ids).size
    original_vertices = np.asarray(source.vertex_global_ids)
    born = np.setdiff1d(np.asarray(fine.vertex_global_ids), original_vertices)
    reborn = np.setdiff1d(np.asarray(again.vertex_global_ids), original_vertices)
    assert not np.intersect1d(born, reborn).size
    for previous, current in zip(
        coarsened.hierarchy.quotient_entities,
        second.hierarchy.quotient_entities,
        strict=True,
    ):
        known = dict(zip(current.entity_keys, current.entity_global_ids, strict=True))
        assert all(
            known[key] == identifier
            for key, identifier in zip(
                previous.entity_keys, previous.entity_global_ids, strict=True
            )
        )
        assert current.allocator_next_id >= previous.allocator_next_id


@pytest.mark.parametrize("variant", ("valid-chain", "bare-advanced-bank", "forged-prior"))
def test_periodic_staged_allocation_requires_the_real_prior_chain(variant: str) -> None:
    from phydrax.meshing._periodic import periodic_vertex_orbit_witness
    from phydrax.meshing._topology_edit import _build_mesh

    source, _ = _periodic_template_source("quadrilateral", False)
    original_coordinates = np.asarray(source.coordinates).copy()
    first = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    previous, _, _ = assemble_topology_edit(
        source, first.edit, numeric_version="first-stage"
    )
    first_witness = first.edit.periodic_orbits
    assert first_witness is not None
    # A different legal source partition retires the first stage's born entities.
    # Fresh vertex/cell IDs come from its actual high-water, not currently live IDs.
    candidate = adapt_mixed_mesh(
        source,
        refine_cell_ids=np.asarray([17], dtype=np.int64),
        protected_edge_keys=np.asarray([[100, 101]], dtype=np.int64),
        hierarchy=MixedAdaptationHierarchy(
            next_cell_id=first.hierarchy.next_cell_id,
            next_vertex_id=first.hierarchy.next_vertex_id,
        ),
    )
    raw = candidate.edit.periodic_orbits
    assert raw is not None
    plain = _build_mesh(candidate.edit, "second-stage", None)
    witness = periodic_vertex_orbit_witness(
        source,
        plain,
        raw.vertex_representative_ids,
        raw.vertex_shifts,
        None,
        retained_quotient_entities=first_witness.quotient_entities,
        allocation_prior=(previous, first_witness),
    )
    if variant == "bare-advanced-bank":
        invalid = witness._replace(allocation_prior=None)
        with pytest.raises(ValueError, match="allocation cursor"):
            assemble_topology_edit(
                source,
                candidate.edit._replace(periodic_orbits=invalid),
                numeric_version="refused",
            )
    elif variant == "forged-prior":
        banks = first_witness.quotient_entities
        forged_bank = banks[0]._replace(allocator_next_id=banks[0].allocator_next_id + 1)
        forged = first_witness._replace(quotient_entities=(forged_bank, *banks[1:]))
        invalid = witness._replace(allocation_prior=(previous, forged))
        with pytest.raises(ValueError, match="allocator high-water"):
            assemble_topology_edit(
                source,
                candidate.edit._replace(periodic_orbits=invalid),
                numeric_version="refused",
            )
    else:
        target, _, _ = assemble_topology_edit(
            source,
            candidate.edit._replace(periodic_orbits=witness),
            numeric_version="second-stage",
        )
        old = dict(
            zip(
                first_witness.quotient_entities[0].entity_keys,
                first_witness.quotient_entities[0].entity_global_ids,
                strict=True,
            )
        )
        bank = witness.quotient_entities[0]
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        periodic = target.periodic_topology
        assert periodic is not None
        born_keys = sorted(set(periodic.entity_keys(1)) - set(old))
        start = first_witness.quotient_entities[0].allocator_next_id
        assert [known[key] for key in born_keys] == list(
            range(start, bank.allocator_next_id)
        )
        assert all(known[key] == identifier for key, identifier in old.items())
        # A third real partition forces assembly to replay both retained stages.
        third = adapt_mixed_mesh(
            source,
            refine_cell_ids=np.asarray([17], dtype=np.int64),
            hierarchy=MixedAdaptationHierarchy(
                next_cell_id=candidate.hierarchy.next_cell_id,
                next_vertex_id=candidate.hierarchy.next_vertex_id,
            ),
        )
        third_raw = third.edit.periodic_orbits
        assert third_raw is not None
        third_plain = _build_mesh(third.edit, "third-stage", None)
        third_witness = periodic_vertex_orbit_witness(
            source,
            third_plain,
            third_raw.vertex_representative_ids,
            third_raw.vertex_shifts,
            None,
            retained_quotient_entities=witness.quotient_entities,
            allocation_prior=(target, witness),
        )
        final, _, _ = assemble_topology_edit(
            source,
            third.edit._replace(periodic_orbits=third_witness),
            numeric_version="third-stage",
        )
        final_periodic = final.periodic_topology
        assert final_periodic is not None
        final_bank = third_witness.quotient_entities[0]
        final_ids = dict(
            zip(final_bank.entity_keys, final_bank.entity_global_ids, strict=True)
        )
        final_born = sorted(set(final_periodic.entity_keys(1)) - set(known))
        assert [final_ids[key] for key in final_born] == list(
            range(bank.allocator_next_id, final_bank.allocator_next_id)
        )
        assert all(final_ids[key] == identifier for key, identifier in known.items())
    np.testing.assert_array_equal(source.coordinates, original_coordinates)


def test_nonnested_surface_reconstruction_preserves_degree_and_hard_fidelity() -> None:
    from examples._native_surface_sources import sphere
    from phydrax.meshing._curving import _straight_geometry

    domain = sphere().domain
    uv = np.asarray([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], dtype=np.float64)
    points = domain.evaluate(np.zeros(4, dtype=np.int32), uv)
    source = CellMesh.from_triangles(
        points,
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        cell_global_ids=np.asarray([10, 11], dtype=np.int64),
    )
    cells = np.asarray([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
    target = CellMesh.from_triangles(
        points, cells, cell_global_ids=np.asarray([20, 21], dtype=np.int64)
    )
    geometry = _straight_geometry(source, 2)
    layout = _straight_geometry(target, 2)
    before = np.asarray(geometry.coordinates).copy()
    result = reconstruct_parametric_surface_cell_geometry(
        source,
        geometry,
        target,
        layout,
        domain,
        domain_id=domain.domain_id,
        cell_ids=np.asarray([20, 21], dtype=np.int64),
        cell_patches=np.zeros(2, dtype=np.int32),
        cell_charts=uv[cells],
        cell_geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
        cell_occurrence_paths=(domain.source_occurrences[2][0],) * 2,
        maximum_fidelity=0.1,
    )
    for element in result.geometry.elements:
        assert isinstance(element, FiniteElementSpec)
        assert element.degree == 2
    assert not result.exact
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(result.geometry.coordinates), axis=1), 1.0, atol=2e-14
    )
    probes = np.asarray([[0.2, 0.2], [0.5, 0.1], [0.3, 0.3]], dtype=np.float64)
    element = result.geometry.elements[0]
    assert isinstance(element, FiniteElementSpec)
    for cell in range(2):
        values, _ = element.tabulate(probes)
        mapped = (
            np.asarray(values)
            @ np.asarray(result.geometry.coordinates)[
                np.asarray(result.geometry.geometry_dofs[0])[cell]
            ]
        )
        chart = np.column_stack((1 - probes.sum(axis=1), probes)) @ uv[cells[cell]]
        expected = np.column_stack(
            (
                np.cos(chart[:, 0]) * np.cos(chart[:, 1]),
                np.sin(chart[:, 0]) * np.cos(chart[:, 1]),
                np.sin(chart[:, 1]),
            )
        )
        error = np.max(np.linalg.norm(mapped - expected, axis=1))
        assert error < 2e-4
        assert error <= np.asarray(result.fidelity_bounds)[cell]
    with pytest.raises(CellGeometryTransitionError, match="fidelity"):
        reconstruct_parametric_surface_cell_geometry(
            source,
            geometry,
            target,
            layout,
            domain,
            domain_id=domain.domain_id,
            cell_ids=np.asarray([20, 21], dtype=np.int64),
            cell_patches=np.zeros(2, dtype=np.int32),
            cell_charts=uv[cells],
            cell_geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
            cell_occurrence_paths=(domain.source_occurrences[2][0],) * 2,
            maximum_fidelity=1e-8,
        )
    assert np.array_equal(np.asarray(geometry.coordinates), before)


def test_retained_rational_reconstruction_uses_continuous_source_expressions() -> None:
    from fractions import Fraction

    import phydrax as phx
    from phydrax.discretization._cell_geometry import (
        PolynomialComposedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from phydrax.discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        coordinate_expressions,
        expression_compose,
        expression_parts,
        scale,
    )
    from phydrax.geometry._surface_source_support import prepare_surface_source_root_atlas

    patch = phx.geometry.BSplineSurfacePatch(
        np.asarray(
            (((0.0, 0.0, 0.0), (0.0, 1.0, 0.05)), ((1.0, 0.0, 0.1), (1.0, 1.0, 0.2)))
        ),
        np.asarray(((1.0, 0.9), (1.1, 1.0))),
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        1,
    )
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    uses = tuple(
        phx.geometry.PatchCurveUse(
            row,
            phx.geometry.LineCurve(uv[row], uv[(row + 1) % 4] - uv[row]),
            0.0,
            1.0,
        )
        for row in range(4)
    )
    domain = phx.geometry.MeshingDomain(
        (phx.geometry.MeshingSurfacePatch(patch, (uses,)),),
        tuple(phx.geometry.MeshingDomainCurve(row, (row + 1) % 4) for row in range(4)),
        4,
        source_id="retained-rational-reconstruction",
        source_revision="authored",
    )
    atlas = prepare_surface_source_root_atlas(
        domain,
        (0,),
        phx.SpatialCoordinateContract(phx.units.METER),
        maximum_support_queries=100_000,
    )
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    target = CellMesh.from_triangles(
        domain.evaluate(np.zeros(4, dtype=np.int32), uv), cells
    )
    ids = np.concatenate([np.asarray(block.global_ids) for block in target.blocks])
    result = reconstruct_parametric_surface_cell_geometry(
        atlas.mesh,
        atlas.geometry,
        target,
        None,
        domain,
        domain_id=domain.domain_id,
        cell_ids=ids,
        cell_patches=np.zeros(2, dtype=np.int32),
        cell_charts=uv[cells],
        cell_geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
        cell_occurrence_paths=(domain.source_occurrences[2][0],) * 2,
        maximum_fidelity=0.0,
        source_atlas=atlas,
    )
    np.testing.assert_array_equal(
        result.target_mesh.vertex_global_ids, target.vertex_global_ids
    )
    np.testing.assert_array_equal(
        np.asarray(result.geometry.coordinates).view(np.uint64),
        np.asarray(atlas.geometry.coordinates).view(np.uint64),
    )
    root = atlas.geometry.elements[0]
    assert isinstance(root, SplineCellGeometryElement)
    source = coordinate_expressions(root, np.asarray(atlas.geometry.coordinates))
    assert source is not None
    variables = axes(2)
    for row, element in enumerate(result.geometry.elements):
        assert isinstance(element, PolynomialComposedCellGeometryElement)
        assert element.source_element.element_id == root.element_id
        arguments = tuple(
            add(
                constant(Fraction(float(uv[cells[row, 0], axis])), 2),
                add(
                    scale(
                        variables[0],
                        Fraction(
                            float(uv[cells[row, 1], axis] - uv[cells[row, 0], axis])
                        ),
                    ),
                    scale(
                        variables[1],
                        Fraction(
                            float(uv[cells[row, 2], axis] - uv[cells[row, 0], axis])
                        ),
                    ),
                ),
            )
            for axis in range(2)
        )
        actual = coordinate_expressions(element, np.asarray(result.geometry.coordinates))
        assert actual is not None
        assert tuple(expression_parts(value, 2) for value in actual) == tuple(
            expression_parts(expression_compose(value, arguments), 2) for value in source
        )
    np.testing.assert_array_equal(result.fidelity_bounds, np.zeros(2))
    assert (
        result.evaluation_count > 24
    )  # Includes actual coefficient work, not just chart bookkeeping.
    assert result.node_owner_cells is None and result.node_owner_locals is None
    with pytest.raises(ValueError, match="retained original source atlas"):
        reconstruct_parametric_surface_cell_geometry(
            atlas.mesh,
            atlas.geometry,
            target,
            None,
            domain,
            domain_id=domain.domain_id,
            cell_ids=ids,
            cell_patches=np.zeros(2, dtype=np.int32),
            cell_charts=uv[cells],
            cell_geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
            cell_occurrence_paths=(domain.source_occurrences[2][0],) * 2,
            maximum_fidelity=0.0,
        )


def test_native_material_pieces_preserve_nonrepresentable_exact_partitions() -> None:
    from fractions import Fraction

    from phydrax.discretization._surface_chart_deformation import (
        _material_point,
        _prepare_exact_material_pieces,
    )

    zero = Fraction(0)
    a, b, c, d = (
        (zero, zero),
        (Fraction(1, 3), zero),
        (Fraction(1, 3), Fraction(1, 7)),
        (zero, Fraction(1, 7)),
    )
    source, target = ((a, b, c), (a, c, d)), ((a, b, d), (b, c, d))
    pieces, work, pairs, retained = _prepare_exact_material_pieces(
        source,
        target,
        np.asarray((17, 91), dtype=np.int64),
        np.asarray((53, 211), dtype=np.int64),
        16,
        1_000_000,
        64 * 1024**2,
    )
    assert pieces and work > 0 and pairs > 0 and retained > 0
    old_areas, new_areas = [zero, zero], [zero, zero]
    for piece in pieces:
        old_areas[piece.source_row] += piece.source_reference_area
        new_areas[piece.target_row] += piece.target_reference_area
        assert tuple(
            _material_point(source[piece.source_row], point)
            for point in piece.exact_source_reference_vertices
        ) == tuple(
            _material_point(target[piece.target_row], point)
            for point in piece.exact_target_reference_vertices
        )
        assert piece.source_cell_global_id == (17, 91)[piece.source_row]
        assert piece.target_cell_global_id == (53, 211)[piece.target_row]
    assert old_areas == new_areas == [Fraction(1, 2), Fraction(1, 2)]
    assert (
        Fraction(float(b[0])) != b[0]
    )  # Native topology retained the original rational value.


def test_native_material_partition_refuses_a_real_gap() -> None:
    from fractions import Fraction

    from phydrax.discretization._surface_chart_deformation import (
        _prepare_exact_material_pieces,
    )

    source = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    target = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1, 2), Fraction(0)),
            (Fraction(0), Fraction(1, 2)),
        ),
    )
    with pytest.raises(ValueError, match="gap or double cover"):
        _prepare_exact_material_pieces(
            source,
            target,
            np.asarray((17,), dtype=np.int64),
            np.asarray((53,), dtype=np.int64),
            4,
            1_000_000,
            64 * 1024**2,
        )


def test_native_material_embedding_rejects_balanced_gap_and_double_cover() -> None:
    from fractions import Fraction

    from phydrax.discretization._surface_chart_deformation import (
        _prepare_exact_material_pieces,
    )

    source = (
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
    )
    half = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1, 2)),
    )
    # Net material area matches the source, but one half is doubled and the
    # other is absent. Matching total/piece reference areas alone is not proof.
    with pytest.raises(ValueError, match="gap or double cover"):
        _prepare_exact_material_pieces(
            source,
            (half, half),
            np.asarray((17,), dtype=np.int64),
            np.asarray((53, 211), dtype=np.int64),
            16,
            1_000_000,
            64 * 1024**2,
        )


@pytest.mark.parametrize("scope", ["complete", "incomplete", "unknown", "changed-class"])
def test_mixed_coarsening_authenticates_unchanged_source_cells(
    scope: str, tmp_path: Path
) -> None:
    corners = np.asarray(reference_cell_topology("prism").vertices, dtype=np.float64)
    source = CellMesh(
        np.concatenate((corners, corners + np.asarray([3.0, 0.0, 0.0]))),
        (
            CellBlock(
                "volume",
                "prism",
                np.arange(12, dtype=np.int32).reshape(2, 6),
                global_ids=np.asarray([17, 18], dtype=np.int64),
            ),
        ),
    )
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    fine, _, _ = assemble_topology_edit(
        source, refined.edit, numeric_version="identity-fine"
    )
    receipt = write_meshing_source_closure(
        tmp_path / "identity-hierarchy", refined.hierarchy
    )
    hierarchy = read_meshing_source_closure(
        tmp_path / "identity-hierarchy", expected_content_id=receipt.content_id
    )
    assert isinstance(hierarchy, MixedAdaptationHierarchy)
    children = np.asarray(hierarchy.records[0].child_ids, dtype=np.int64)
    marks = np.concatenate((children[:1] if scope == "incomplete" else children, [18]))
    if scope == "unknown":
        hierarchy = MixedAdaptationHierarchy(
            hierarchy.records,
            next_cell_id=hierarchy.next_cell_id,
            next_vertex_id=hierarchy.next_vertex_id,
            retired_entities=hierarchy.retired_entities,
        )
    classes = np.concatenate(
        [
            np.asarray(
                [
                    1 if int(identifier) == 18 and scope == "changed-class" else 0
                    for identifier in np.asarray(block.global_ids)
                ],
                dtype=np.int64,
            )
            for block in fine.blocks
        ]
    )
    outcome = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=marks,
        hierarchy=hierarchy,
        cell_classes=classes,
    )
    expected_rejected = (
        children[:1]
        if scope == "incomplete"
        else np.asarray(
            [18] if scope in ("unknown", "changed-class") else [], dtype=np.int64
        )
    )
    np.testing.assert_array_equal(
        outcome.evidence.rejected_coarsening_ids, expected_rejected
    )
    np.testing.assert_array_equal(
        outcome.evidence.coarsened_cell_ids,
        np.zeros(0, dtype=np.int64) if scope == "incomplete" else np.sort(children),
    )
    target, _, _ = assemble_topology_edit(
        fine, outcome.edit, numeric_version="identity-coarse"
    )
    if scope == "complete":
        assert target.topology_id == source.topology_id
    elif scope == "incomplete":
        assert target.topology_id == fine.topology_id


def test_quad_parent_restore_keeps_independent_reference_corner_banks() -> None:
    from phydrax.discretization._cell_geometry_transfer import NestedReferenceWitnesses

    source, geometry = _curved_cell("quadrilateral")
    original = np.asarray(geometry.coordinates).view(np.uint64).copy()
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray((17,), dtype=np.int64))
    fine, _, _ = assemble_topology_edit(source, refined.edit, numeric_version="refined")
    fine_geometry = transition_nested_cell_geometry(
        source,
        geometry,
        fine,
        CellGeometrySpec.affine(fine),
        refinement=refined.edit.refinement,
    )
    ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarse = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=ids,
        hierarchy=refined.hierarchy,
    )
    restored, _, _ = assemble_topology_edit(fine, coarse.edit, numeric_version="restored")
    assert coarse.edit.coarsening is not None
    witness = coarse.edit.coarsening
    # Four actual fine quadrilateral corners are independent of the generic
    # target parent-bank capacity. No source coefficient or SCI row is changed.
    corners = len(reference_cell_topology("quadrilateral").vertices)
    actual = NestedReferenceWitnesses(
        witness.fine_cell_ids,
        witness.coarse_cell_ids,
        witness.fine_reference_vertices[:, :corners],
    )
    transition = transition_nested_cell_geometry(
        fine,
        fine_geometry.geometry,
        restored,
        CellGeometrySpec.affine(restored),
        refinement=coarse.edit.refinement,
        coarsening=actual,
    )
    assert transition.parent_reference_vertices.shape[1] == 8
    assert transition.coarsened_reference_vertices.shape[1] == corners
    assert transition.evidence.exact
    np.testing.assert_array_equal(
        np.asarray(transition.geometry.coordinates).view(np.uint64), original
    )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import json
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionScope,
)
from phydrax._model._structure import model_from_structure_recipe, model_structure_recipe
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_complex import (
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    reconstruct_parametric_surface_cell_geometry,
    transition_nested_cell_geometry,
)
from phydrax.discretization._cell_geometry_transition_types import (
    CellGeometryTransitionError,
    NestedReferenceWitnesses,
)
from phydrax.discretization._chart_reconstruction import SurfaceGeometryReconstruction
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    constant,
    coordinate_expressions,
    expression_compose,
    expression_parts,
    scale,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import FiniteElementSpec
from phydrax.geometry._surface_source_support import (
    prepare_surface_source_root_atlas,
    SurfaceSourceRootAtlas,
)
from phydrax.lifecycle import register_meshing_source_artifacts
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing import _mixed_adaptation as mixed_owner
from phydrax.meshing._mixed_adaptation import (
    adapt_mixed_mesh,
    MixedAdaptationEvidence,
    MixedAdaptationHierarchy,
    MixedLayerColumns,
    MixedRetiredEntityBank,
    MixedSiblingBank,
    MixedTemplateClosureError,
    Sibling,
)
from phydrax.meshing._periodic import (
    PeriodicConstructionOrbits,
    periodic_vertex_orbit_witness,
)
from phydrax.meshing._topology_edit import assemble_topology_edit, build_mesh
from phydrax.meshing.qualification import sphere_surface
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mapped_cells import curved_volume
from tests._support.mixed_adaptation import (
    periodic_template_source,
    red_shared_face_source,
)
from tests._support.recipes import recipe_field_position


# Retained regression envelopes from this module's pre-remediation quadratic
# pullback/unit-sphere checks and the independent 4/3 hex-pyramid measure check.
# These acceptance values are unchanged; they are not resource or fidelity inputs.
COORDINATE_REGRESSION_ABS = 2e-14
MEASURE_REGRESSION_ABS = 1e-11
# Retained envelope from the unit-sphere, degree-two reconstruction probes on
# the authored 0.1-radian chart; distinct from its caller's fidelity allowance.
SPHERE_RECONSTRUCTION_REGRESSION_BOUND = 2e-4

@pytest.fixture
def template_request_budget() -> Iterator[
    tuple[NativeExecutionScope, CoordinateEnclosureBudget, NativeBatchTicket]
]:
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=1 << 20, maximum_scratch_bytes=1 << 26
    )
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(1 << 20, bytes_upper=1 << 26) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=1 << 20, maximum_memory_bytes=1 << 26
            )
            with budget.activate(), budget.temporary_scope():
                yield scope, budget, ticket


@pytest.mark.parametrize(
    "kind", ("triangle", "quadrilateral", "tetrahedron", "prism", "hexahedron", "pyramid")
)
def test_prepared_template_facts_preserve_every_signature_and_protection(
    kind: str,
    template_request_budget: tuple[
        NativeExecutionScope, CoordinateEnclosureBudget, NativeBatchTicket
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope, budget, ticket = template_request_budget
    topology = reference_cell_topology(kind)
    if kind == "triangle":
        # Triangle point keys are a supported reference helper, not a native
        # mixed-template generation route. Exercise an authored exact split.
        templates = (
            mixed_owner._Template(
                "identity", np.asarray(topology.vertices, dtype=np.float64)[None], None
            ),
            mixed_owner._Template(
                "reference-edge-split",
                np.asarray(
                    (((0, 0), (0.5, 0), (0, 1)), ((0.5, 0), (1, 0), (0, 1))),
                    dtype=np.float64,
                ),
                None,
            ),
        )
    else:
        templates = mixed_owner._templates(
            kind, True, scope=scope, budget=budget, ticket=ticket
        )
    ids = np.arange(len(topology.vertices), dtype=np.int64)[::-1] * 19 + 103
    cell = mixed_owner.Cell(17, "cell", kind, ids, 0)
    cache: mixed_owner._TemplateClosureCache = {}
    edges = {
        tuple(sorted(int(ids[index]) for index in edge))
        for edge in topology.entities[1]
    }
    faces = {
        tuple(sorted(int(ids[index]) for index in facet))
        for facet in topology.entities[topology.dimension - 1]
    }
    identity = mixed_owner._facet_signatures(
        kind, ids, templates[0], scope=scope, budget=budget
    )
    expected = []
    for template in templates:
        signature = mixed_owner._facet_signatures(
            kind, ids, template, scope=scope, budget=budget
        )
        split_edge = any(
            len(point) == 2 and tuple(identifier for identifier, _ in point) in edges
            for child in template.children
            for point in mixed_owner._point_keys(
                kind, ids, child, scope=scope, budget=budget
            )
        )
        expected.append((signature, not split_edge and signature == identity))
        assert mixed_owner._prepared_facet_signatures(
            cell, template, cache, None, scope=scope, budget=budget
        ) == signature
    assert len(cache) == len(templates)

    def forbid_repreparation(*args, **kwargs):
        raise AssertionError("Retained exact template facts must not be recomputed")

    monkeypatch.setattr(mixed_owner, "_point_keys", forbid_repreparation)
    for template, (signature, permitted) in zip(templates, expected, strict=True):
        assert mixed_owner._prepared_facet_signatures(
            cell, template, cache, None, scope=scope, budget=budget
        ) == signature
        assert mixed_owner._permitted_subdivision(
            cell, template, edges, faces, references=cache,
            identity_template=templates[0], scope=scope, budget=budget,
        ) == permitted


@pytest.mark.parametrize("kind", ("tetrahedron", "prism", "hexahedron", "pyramid"))
@pytest.mark.parametrize("rotated", (False, True))
def test_prepared_template_facts_preserve_periodic_facet_placement(
    kind: str,
    rotated: bool,
    template_request_budget: tuple[
        NativeExecutionScope, CoordinateEnclosureBudget, NativeBatchTicket
    ],
) -> None:
    scope, budget, ticket = template_request_budget
    if rotated:
        mesh, _ = periodic_template_source(kind, True)
    else:
        # Keep every distinct fractional representative of the unit lattice;
        # a pyramid apex is not an integer translate of a base corner.
        points = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        block = CellBlock(
            "periodic", kind, np.arange(points.shape[0], dtype=np.int32)[None],
            global_ids=np.asarray([17], dtype=np.int64),
        )
        mesh = CellMesh(
            points, (block,),
            vertex_global_ids=np.arange(100, 100 + points.shape[0], dtype=np.int64),
        )
        representatives: dict[tuple[float, ...], int] = {}
        roots = np.asarray(
            [
                representatives.setdefault(tuple(point - np.floor(point)), index)
                for index, point in enumerate(points)
            ],
            dtype=np.int32,
        )
        shifts = (points - points[roots]).astype(np.int32)
        periodic = phx.discretization.PeriodicMeshTopology(
            mesh, phx.discretization.PeriodicCell(np.eye(3)), roots, shifts
        )
        mesh = CellMesh(
            points, (block,), vertex_global_ids=mesh.vertex_global_ids,
            periodic_topology=periodic,
        )
    block = mesh.blocks[0]
    ids = np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)[0]]
    cell = mixed_owner.Cell(int(block.global_ids[0]), block.name, kind, ids, 0)
    orbits = PeriodicConstructionOrbits(mesh)
    cache: mixed_owner._TemplateClosureCache = {}
    for template in mixed_owner._templates(
        kind, True, scope=scope, budget=budget, ticket=ticket
    ):
        assert mixed_owner._prepared_facet_signatures(
            cell, template, cache, orbits, scope=scope, budget=budget
        ) == mixed_owner._periodic_facet_signatures(
            orbits, cell, template, scope=scope, budget=budget
        )


@pytest.mark.parametrize("mixed", (False, True))
def test_prepared_template_facts_preserve_closure_ids_and_canonical_work(
    mixed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, _ = red_shared_face_source(mixed, True)
    requested = np.asarray((37,), dtype=np.int64)
    actual = adapt_mixed_mesh(mesh, refine_cell_ids=requested)

    def direct(cell, template, cache, orbits, *, scope, budget):
        if orbits is None:
            return mixed_owner._facet_signatures(
                cell.kind, cell.vertices, template, scope=scope, budget=budget
            )
        return mixed_owner._periodic_facet_signatures(
            orbits, cell, template, scope=scope, budget=budget
        )

    monkeypatch.setattr(mixed_owner, "_prepared_facet_signatures", direct)
    reference = adapt_mixed_mesh(mesh, refine_cell_ids=requested)
    assert tree_content_equal(actual.edit, reference.edit)
    assert tree_content_equal(actual.hierarchy, reference.hierarchy)
    assert actual.evidence.closure_steps == reference.evidence.closure_steps
    assert actual.evidence.work_units == reference.evidence.work_units



@pytest.mark.parametrize(
    "kind", ["quadrilateral", "hexahedron", "prism", "pyramid", "tetrahedron"]
)
def test_curved_template_restriction_and_complete_parent_restore(kind: str) -> None:
    source, geometry = curved_volume(kind)
    before = np.asarray(geometry.coordinates).copy()
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    with coordinate_reference_budget() as budget:
        target, _, stencil = assemble_topology_edit(
            source, refined.edit, numeric_version="refined", budget=budget
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
    fine_ids = np.asarray(refined.edit.refinement.fine_cell_ids, dtype=np.int64)
    reference_vertices = np.asarray(
        refined.edit.refinement.fine_reference_vertices, dtype=np.float64
    )
    assert reference_vertices.ndim == 3
    assert reference_vertices.shape[0] == fine_ids.size
    witnesses = {
        int(fine_ids[index]): reference_vertices[index, :, :]
        for index in range(fine_ids.size)
    }
    reference = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    # Children restrict the source basis rather than interpolate child nodal data:
    # the parent element and its coefficients are retained bitwise, and only
    # the affine child reference action is new.
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
        assert np.allclose(result, expected, rtol=0, atol=COORDINATE_REGRESSION_ABS)
    source_ids = np.concatenate([np.asarray(block.global_ids) for block in target.blocks])
    coarse = adapt_mixed_mesh(
        target,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=source_ids,
        hierarchy=refined.hierarchy,
    )
    with coordinate_reference_budget() as budget:
        restored, _, _ = assemble_topology_edit(
            target, coarse.edit, numeric_version="restored", budget=budget
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
    mesh, _ = curved_volume("prism")
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
    source, _ = curved_volume("prism")
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
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="archive-fine", budget=budget
        )
    archive = tmp_path / "layer-hierarchy"
    receipt = write_meshing_source_closure(archive, refined.hierarchy)
    hierarchy = read_meshing_source_closure(
        archive, expected_content_id=receipt.content_id
    )
    if not isinstance(hierarchy, MixedAdaptationHierarchy):
        raise TypeError(
            f"mixed_adaptation_test.hierarchy must be MixedAdaptationHierarchy; got {type(hierarchy).__name__}"
        )
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarse = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=hierarchy,
    )
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            fine, coarse.edit, numeric_version="archive-coarse", budget=budget
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
        raise ValueError(
            "mixed_adaptation_test.refinement must contain repeated layer reference corners; got None"
        )
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
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="column-fine", budget=budget
        )
    first, second = refined.hierarchy.siblings
    first_ids = np.asarray(first.child_ids, dtype=np.int64)
    all_ids = np.concatenate((first_ids, np.asarray(second.child_ids, dtype=np.int64)))
    marks = first_ids if failure == "partial" else all_ids
    hierarchy = refined.hierarchy
    if failure == "missing-ancestry":
        hierarchy = MixedAdaptationHierarchy(
            (first,),
            next_cell_id=hierarchy.next_cell_id,
            next_vertex_id=hierarchy.next_vertex_id,
            retired_entities=tuple(hierarchy.retired),
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
    with coordinate_reference_budget() as budget:
        retained, _, _ = assemble_topology_edit(
            fine, outcome.edit, numeric_version="column-retained", budget=budget
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
    with coordinate_reference_budget() as budget:
        restored, _, _ = assemble_topology_edit(
            fine, complete.edit, numeric_version="column-restored", budget=budget
        )
    assert restored.topology_id == source.topology_id
    assert complete.hierarchy.layer_columns is not None
    np.testing.assert_array_equal(
        complete.hierarchy.layer_columns.interval_indices, [0, 1]
    )


def test_explicit_first_interval_schedule_revision_is_evidenced() -> None:
    mesh, _ = curved_volume("prism")
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
    mesh, _ = curved_volume("hexahedron")
    before = np.asarray(mesh.coordinates).copy()
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.asarray([17], dtype=np.int64))
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version="refined", budget=budget
        )
    marked = np.asarray([target.blocks[0].global_ids[0]], dtype=np.int64)
    incomplete = adapt_mixed_mesh(
        target,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=marked,
        hierarchy=outcome.hierarchy,
    )
    assert np.array_equal(incomplete.evidence.rejected_coarsening_ids, marked)
    with pytest.raises(
        MixedTemplateClosureError, match=r"mixed_adaptation\.protected_cell_ids"
    ):
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
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version="closed", budget=budget
        )
    restriction = transition_nested_cell_geometry(
        mesh,
        geometry,
        target,
        CellGeometrySpec.affine(target),
        refinement=outcome.edit.refinement,
    )
    assert restriction.evidence.exact
    assert restriction.evidence.target_measure == pytest.approx(
        4.0 / 3.0, abs=MEASURE_REGRESSION_ABS
    )
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
    with coordinate_reference_budget() as budget:
        restored, _, _ = assemble_topology_edit(
            target, coarse.edit, numeric_version="coarse", budget=budget
        )
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


class _RedLifecycle(NamedTuple):
    mixed: bool
    mesh: CellMesh
    source: phx.meshing.CellMeshingResult
    policy: phx.meshing.MeshAdaptationPolicy
    refined: phx.meshing.MeshAdaptationResult


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((False, False), id="tetrahedra"),
        pytest.param((False, True), id="permuted-tetrahedra"),
        pytest.param((True, False), id="prism-tetrahedron"),
    ],
)
def red_lifecycle(request: pytest.FixtureRequest) -> _RedLifecycle:
    mixed, permuted = request.param
    M = phx.meshing
    mesh, geometry = red_shared_face_source(mixed, permuted)
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
    return _RedLifecycle(mixed, mesh, source, policy, refined)


def _assert_red_refinement_cell_count(mesh: CellMesh) -> None:
    assert mesh.entity_set(3).count == 16


def test_red_lifecycle_guard_rejects_unrefined_source(
    red_lifecycle: _RedLifecycle,
) -> None:
    # A valid source is nevertheless a wrong output for the refinement builder.
    with pytest.raises(AssertionError):
        _assert_red_refinement_cell_count(red_lifecycle.source.mesh)


def test_red_tetrahedral_refinement_evidence(red_lifecycle: _RedLifecycle) -> None:
    M = phx.meshing
    source, refined = red_lifecycle.source, red_lifecycle.refined
    limits = red_lifecycle.policy.limits
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    assert source.audit.passed and refined.target.audit.passed
    assert isinstance(refined.evidence, MixedAdaptationEvidence)
    assert 0 < refined.evidence.work_units <= limits.maximum_work_units
    execution = refined.evidence.execution_evidence
    assert execution is not None
    assert (
        int(np.asarray(execution.work_counters.construction_work))
        >= refined.evidence.work_units
    )
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
    assert np.all(refined.hierarchy.siblings.child_counts == 8)
    fine = refined.target.mesh
    _assert_red_refinement_cell_count(fine)


def test_red_tetrahedral_shared_face_closure(red_lifecycle: _RedLifecycle) -> None:
    mixed = red_lifecycle.mixed
    fine = red_lifecycle.refined.target.mesh
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
            f"mixed_adaptation_test.connectivity must have triangular face incidence; got {type(connectivity).__name__}"
        )
    height = 1.0 if mixed else 0.0
    shared = np.all(np.asarray(fine.coordinates)[rows, 2] == height, axis=1)
    assert np.count_nonzero(shared) == 4
    if isinstance(connectivity, TetrahedralConnectivity):
        assert np.all(~np.asarray(connectivity.boundary_faces)[shared])
    else:
        triangular = np.diff(np.asarray(connectivity.face_vertex_offsets)) == 3
        assert np.all(np.asarray(connectivity.face_neighbor)[triangular][shared] >= 0)


def test_red_tetrahedral_diagonal_identity(red_lifecycle: _RedLifecycle) -> None:
    mesh, refined = red_lifecycle.mesh, red_lifecycle.refined
    fine = refined.target.mesh
    assert isinstance(refined.hierarchy, MixedAdaptationHierarchy)
    # Consume real central-child connectivity. Its common edge must be the
    # lexicographically first pair of opposite source midpoint-edge identities,
    # even when the source local tetrahedron numbering is permuted.
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
    for record in refined.hierarchy.siblings:
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
        # Pull back the authored shear to compare the generated midpoint
        # construction in the source affine reference partition.
        actual_points = [fine_positions[identifier].copy() for identifier in diagonal]
        for point in actual_points:
            point[1] -= point[0] * point[2] / 16
        # Curved-edge midpoint coordinates require the source pullback, not
        # the average of the curved endpoint coordinates.
        expected_points = []
        for first, second in expected_keys:
            ends = [source_positions[identifier].copy() for identifier in (first, second)]
            for point in ends:
                point[1] -= point[0] * point[2] / 16
            expected_points.append((ends[0] + ends[1]) / 2)
        assert sorted(tuple(point) for point in actual_points) == sorted(
            tuple(point) for point in expected_points
        )


def test_red_tetrahedral_parent_restoration(red_lifecycle: _RedLifecycle) -> None:
    M = phx.meshing
    source, refined, policy = (
        red_lifecycle.source,
        red_lifecycle.refined,
        red_lifecycle.policy,
    )
    fine = refined.target.mesh
    assert isinstance(refined.hierarchy, MixedAdaptationHierarchy)
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
    removed_children = refined.hierarchy.siblings.all_child_ids()
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


@pytest.mark.parametrize(
    ("maximum_cells", "maximum_work_units", "field"),
    [
        pytest.param(15, 20_000_000, r"mixed_adaptation\.maximum_cells", id="cells"),
        pytest.param(100_000, 1, r"mixed_adaptation\.maximum_work_units", id="work"),
    ],
)
def test_red_tetrahedral_resource_refusal_leaves_source_unchanged(
    maximum_cells: int,
    maximum_work_units: int,
    field: str,
) -> None:
    mesh, _ = red_shared_face_source(False, False)
    coordinates = np.asarray(mesh.coordinates).copy()
    topology_id = mesh.topology_id
    with pytest.raises(MixedTemplateClosureError, match=field):
        adapt_mixed_mesh(
            mesh,
            refine_cell_ids=np.asarray((11, 37), dtype=np.int64),
            maximum_cells=maximum_cells,
            maximum_work_units=maximum_work_units,
        )
    assert mesh.topology_id == topology_id
    np.testing.assert_array_equal(mesh.coordinates, coordinates)


def test_red_tetrahedral_original_host_scratch_refusal_leaves_source_unchanged() -> None:

    mesh, _ = red_shared_face_source(False, False)
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
    assert caught.value.memory_counters is not None
    assert caught.value.memory_counters.limit_bytes == 1
    assert mesh.topology_id == topology_id
    np.testing.assert_array_equal(mesh.coordinates, coordinates)


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
    source, geometry = periodic_template_source(kind, rotated)
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="fine", budget=budget
        )
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
    with coordinate_reference_budget() as budget:
        restored, _, _ = assemble_topology_edit(
            fine, coarsened.edit, numeric_version="restored", budget=budget
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
    source, _ = periodic_template_source("hexahedron", False)
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="fine", budget=budget
        )
    before = np.asarray(fine.coordinates).copy()
    identity = fine.topology_id
    if failure == "partial-family":
        with pytest.raises(
            MixedTemplateClosureError, match=r"mixed_adaptation\.coarsen_cell_ids"
        ):
            adapt_mixed_mesh(
                fine,
                refine_cell_ids=np.zeros(0, dtype=np.int64),
                coarsen_cell_ids=np.asarray(
                    [fine.blocks[0].global_ids[0]], dtype=np.int64
                ),
                hierarchy=refined.hierarchy,
            )
    elif failure == "work":
        with pytest.raises(
            MixedTemplateClosureError, match=r"mixed_adaptation\.maximum_work_units"
        ):
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
        with (
            coordinate_reference_budget() as budget,
            pytest.raises(ValueError, match=r"PeriodicMeshTopology\.vertex_shifts"),
        ):
            assemble_topology_edit(
                source, invalid_edit, numeric_version="invalid", budget=budget
            )
    elif failure == "stale-source-orbit":
        record = refined.hierarchy.siblings[0]
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
            MixedTemplateClosureError,
            match=r"mixed_adaptation\.hierarchy\.periodic_roots",
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
            tuple(refined.hierarchy.siblings),
            next_cell_id=refined.hierarchy.next_cell_id,
            next_vertex_id=refined.hierarchy.next_vertex_id,
            quotient_entities=(invalid_bank, *banks[1:]),
        )
        with pytest.raises(ValueError, match=r"allocate_periodic_identity_banks\.target"):
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

    M = phx.meshing
    mesh, _ = periodic_template_source("hexahedron", False)
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
    with pytest.raises(
        MixedTemplateClosureError, match=r"mixed_adaptation\.coarsen_cell_ids"
    ):
        M.execute_mesh_adaptation(prepared)
    np.testing.assert_array_equal(fine.mesh.coordinates, before)
    assert fine.mesh.topology_id == refinement.target.mesh.topology_id


def test_periodic_mixed_retired_identities_are_not_reused_after_restore() -> None:
    source, _ = periodic_template_source("hexahedron", False)
    first = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, first.edit, numeric_version="fine", budget=budget
        )
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarsened = adapt_mixed_mesh(
        fine,
        refine_cell_ids=np.zeros(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=first.hierarchy,
    )
    with coordinate_reference_budget() as budget:
        restored, _, _ = assemble_topology_edit(
            fine, coarsened.edit, numeric_version="restored", budget=budget
        )
    second = adapt_mixed_mesh(
        restored,
        refine_cell_ids=np.asarray([17], dtype=np.int64),
        hierarchy=coarsened.hierarchy,
    )
    with coordinate_reference_budget() as budget:
        again, _, _ = assemble_topology_edit(
            restored, second.edit, numeric_version="again", budget=budget
        )
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

    source, _ = periodic_template_source("quadrilateral", False)
    original_coordinates = np.asarray(source.coordinates).copy()
    first = adapt_mixed_mesh(source, refine_cell_ids=np.asarray([17], dtype=np.int64))
    with coordinate_reference_budget() as budget:
        previous, _, _ = assemble_topology_edit(
            source, first.edit, numeric_version="first-stage", budget=budget
        )
    first_witness = first.edit.periodic_orbits
    assert first_witness is not None
    # A different legal source partition retires the first stage's born entities.
    # Fresh vertex/cell IDs use its allocator high-water rather than live IDs.
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
    with coordinate_reference_budget() as budget:
        plain = build_mesh(candidate.edit, "second-stage", None, budget=budget)
        witness = periodic_vertex_orbit_witness(
            source,
            plain,
            raw.vertex_representative_ids,
            raw.vertex_shifts,
            None,
            retained_quotient_entities=first_witness.quotient_entities,
            allocation_prior=(previous, first_witness),
            budget=budget,
        )
    if variant == "bare-advanced-bank":
        invalid = witness._replace(allocation_prior=None)
        with (
            coordinate_reference_budget() as budget,
            pytest.raises(ValueError, match=r"allocate_periodic_identity_banks\.target"),
        ):
            assemble_topology_edit(
                source,
                candidate.edit._replace(periodic_orbits=invalid),
                numeric_version="refused",
                budget=budget,
            )
    elif variant == "forged-prior":
        banks = first_witness.quotient_entities
        forged_bank = banks[0]._replace(allocator_next_id=banks[0].allocator_next_id + 1)
        forged = first_witness._replace(quotient_entities=(forged_bank, *banks[1:]))
        invalid = witness._replace(allocation_prior=(previous, forged))
        with (
            coordinate_reference_budget() as budget,
            pytest.raises(ValueError, match=r"bind_periodic_target\.source"),
        ):
            assemble_topology_edit(
                source,
                candidate.edit._replace(periodic_orbits=invalid),
                numeric_version="refused",
                budget=budget,
            )
    else:
        with coordinate_reference_budget() as budget:
            target, _, _ = assemble_topology_edit(
                source,
                candidate.edit._replace(periodic_orbits=witness),
                numeric_version="second-stage",
                budget=budget,
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
        with coordinate_reference_budget() as budget:
            third_plain = build_mesh(third.edit, "third-stage", None, budget=budget)
            third_witness = periodic_vertex_orbit_witness(
                source,
                third_plain,
                third_raw.vertex_representative_ids,
                third_raw.vertex_shifts,
                None,
                retained_quotient_entities=witness.quotient_entities,
                allocation_prior=(target, witness),
                budget=budget,
            )
        with coordinate_reference_budget() as budget:
            final, _, _ = assemble_topology_edit(
                source,
                third.edit._replace(periodic_orbits=third_witness),
                numeric_version="third-stage",
                budget=budget,
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

    domain = sphere_surface().domain
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
    geometry = straight_geometry(source, 2)
    layout = straight_geometry(target, 2)
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
        np.linalg.norm(np.asarray(result.geometry.coordinates), axis=1),
        1.0,
        atol=COORDINATE_REGRESSION_ABS,
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
        assert error < SPHERE_RECONSTRUCTION_REGRESSION_BOUND
        assert error <= np.asarray(result.fidelity_bounds)[cell]
    with pytest.raises(
        CellGeometryTransitionError, match=r"surface_reconstruction\.fidelity"
    ):
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


class _RationalReconstruction(NamedTuple):
    atlas: SurfaceSourceRootAtlas
    target: CellMesh
    result: SurfaceGeometryReconstruction
    domain: phx.geometry.MeshingDomain
    uv: np.ndarray
    cells: np.ndarray
    ids: np.ndarray


@pytest.fixture(scope="module")
def rational_reconstruction() -> _RationalReconstruction:
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
    return _RationalReconstruction(atlas, target, result, domain, uv, cells, ids)


def test_retained_rational_reconstruction_preserves_source_identity(
    rational_reconstruction: _RationalReconstruction,
) -> None:
    atlas, target, result = (
        rational_reconstruction.atlas,
        rational_reconstruction.target,
        rational_reconstruction.result,
    )
    np.testing.assert_array_equal(
        result.target_mesh.vertex_global_ids, target.vertex_global_ids
    )
    np.testing.assert_array_equal(
        np.asarray(result.geometry.coordinates).view(np.uint64),
        np.asarray(atlas.geometry.coordinates).view(np.uint64),
    )


def test_retained_rational_reconstruction_uses_continuous_source_expressions(
    rational_reconstruction: _RationalReconstruction,
) -> None:
    atlas, result = rational_reconstruction.atlas, rational_reconstruction.result
    uv, cells = rational_reconstruction.uv, rational_reconstruction.cells
    root = atlas.geometry.elements[0]
    assert isinstance(root, SplineCellGeometryElement)
    with coordinate_reference_budget() as budget:
        source = coordinate_expressions(
            root, np.asarray(atlas.geometry.coordinates), budget=budget
        )
        assert source is not None
        variables = axes(2, budget=budget)
        for row, element in enumerate(result.geometry.elements):
            assert isinstance(element, PolynomialComposedCellGeometryElement)
            assert element.source_element.element_id == root.element_id
            arguments = tuple(
                add(
                    constant(Fraction(float(uv[cells[row, 0], axis])), 2, budget=budget),
                    add(
                        scale(
                            variables[0],
                            Fraction(
                                float(uv[cells[row, 1], axis] - uv[cells[row, 0], axis])
                            ),
                            budget=budget,
                        ),
                        scale(
                            variables[1],
                            Fraction(
                                float(uv[cells[row, 2], axis] - uv[cells[row, 0], axis])
                            ),
                            budget=budget,
                        ),
                        budget=budget,
                    ),
                    budget=budget,
                )
                for axis in range(2)
            )
            actual = coordinate_expressions(
                element, np.asarray(result.geometry.coordinates), budget=budget
            )
            assert actual is not None
            assert tuple(
                expression_parts(value, 2, budget=budget) for value in actual
            ) == tuple(
                expression_parts(
                    expression_compose(value, arguments, budget=budget), 2, budget=budget
                )
                for value in source
            )


def test_retained_rational_reconstruction_reports_fidelity_and_work(
    rational_reconstruction: _RationalReconstruction,
) -> None:
    result = rational_reconstruction.result
    np.testing.assert_array_equal(result.fidelity_bounds, np.zeros(2))
    assert (
        result.evaluation_count > 24
    )  # Includes coefficient arithmetic in addition to chart bookkeeping.
    assert result.node_owner_cells is None and result.node_owner_locals is None


def test_retained_rational_reconstruction_requires_source_atlas(
    rational_reconstruction: _RationalReconstruction,
) -> None:
    atlas, target, domain = (
        rational_reconstruction.atlas,
        rational_reconstruction.target,
        rational_reconstruction.domain,
    )
    ids, uv, cells = (
        rational_reconstruction.ids,
        rational_reconstruction.uv,
        rational_reconstruction.cells,
    )
    with pytest.raises(ValueError, match=r"surface_reconstruction\.source_atlas"):
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


def _assert_rational_fem_consumer(state: _RationalReconstruction) -> None:
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        state.result.target_mesh, field, coordinate_spec=state.result.geometry
    ).prepare()
    # Closed-form rational bilinear Bernstein map, from the four authored
    # control points/weights, independent of spline tabulation or expressions.
    controls = np.asarray(
        (
            ((0.0, 0.0, 0.0), (0.0, 1.0, 0.05)),
            ((1.0, 0.0, 0.1), (1.0, 1.0, 0.2)),
        ),
        dtype=np.float64,
    ).reshape((4, 3))
    weights = np.asarray((1.0, 0.9, 1.1, 1.0), dtype=np.float64)
    geometries = space.evaluate_geometry("u", state.result.geometry.coordinates)
    # The two triangles are published as separate chart-composed blocks.
    assert len(geometries) == 2
    for row, block in enumerate(geometries):
        reference = np.asarray(block.reference_points)
        corners = state.uv[state.cells[row]]
        uv = corners[0] + reference @ (corners[1:] - corners[0])
        u, v = uv.T
        basis = np.column_stack(((1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v))
        rational = basis * weights
        expected = rational @ controls / np.sum(rational, axis=1)[:, None]
        np.testing.assert_allclose(
            np.asarray(block.physical_points)[0],
            expected,
            atol=COORDINATE_REGRESSION_ABS,
            rtol=0,
        )


def test_retained_rational_reconstruction_is_consumed_by_public_fem(
    rational_reconstruction: _RationalReconstruction,
) -> None:
    _assert_rational_fem_consumer(rational_reconstruction)


def test_rational_reconstruction_guard_rejects_changed_chart(
    rational_reconstruction: _RationalReconstruction,
) -> None:
    corrupted = rational_reconstruction._replace(
        uv=rational_reconstruction.uv + np.asarray((0.125, 0.0), dtype=np.float64)
    )
    with pytest.raises(AssertionError):
        _assert_rational_fem_consumer(corrupted)


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
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="identity-fine", budget=budget
        )
    receipt = write_meshing_source_closure(
        tmp_path / "identity-hierarchy", refined.hierarchy
    )
    hierarchy = read_meshing_source_closure(
        tmp_path / "identity-hierarchy", expected_content_id=receipt.content_id
    )
    assert isinstance(hierarchy, MixedAdaptationHierarchy)
    children = np.asarray(hierarchy.siblings[0].child_ids, dtype=np.int64)
    marks = np.concatenate((children[:1] if scope == "incomplete" else children, [18]))
    if scope == "unknown":
        hierarchy = MixedAdaptationHierarchy(
            tuple(hierarchy.siblings),
            next_cell_id=hierarchy.next_cell_id,
            next_vertex_id=hierarchy.next_vertex_id,
            retired_entities=tuple(hierarchy.retired),
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
    with coordinate_reference_budget() as budget:
        target, _, _ = assemble_topology_edit(
            fine, outcome.edit, numeric_version="identity-coarse", budget=budget
        )
    if scope == "complete":
        assert target.topology_id == source.topology_id
    elif scope == "incomplete":
        assert target.topology_id == fine.topology_id


def test_quad_parent_restore_keeps_independent_reference_corner_banks() -> None:

    source, geometry = curved_volume("quadrilateral")
    original = np.asarray(geometry.coordinates).view(np.uint64).copy()
    refined = adapt_mixed_mesh(source, refine_cell_ids=np.asarray((17,), dtype=np.int64))
    with coordinate_reference_budget() as budget:
        fine, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="refined", budget=budget
        )
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
    with coordinate_reference_budget() as budget:
        restored, _, _ = assemble_topology_edit(
            fine, coarse.edit, numeric_version="restored", budget=budget
        )
    assert coarse.edit.coarsening is not None
    witness = coarse.edit.coarsening
    # Four fine quadrilateral corners are independent of the target parent-bank
    # capacity. Source coefficients and SCI rows remain unchanged.
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


@pytest.fixture
def stored_columns() -> MixedLayerColumns:
    register_meshing_source_artifacts()
    return MixedLayerColumns([17, 18], [4, 4], [0, 1], axial_refinement=True)


@pytest.fixture
def stored_hierarchy(stored_columns: MixedLayerColumns) -> MixedAdaptationHierarchy:
    return MixedAdaptationHierarchy(
        next_cell_id=19,
        next_vertex_id=108,
        layer_columns=stored_columns,
        identity_cells=((17, "volume", "prism", (100, 101, 102, 103, 104, 105), 0),),
    )


def test_layer_columns_stored_fields_preserve_ancestry_and_identity(
    stored_columns: MixedLayerColumns,
) -> None:
    restored = MixedLayerColumns.from_stored_fields(stored_columns)
    assert restored is not stored_columns
    assert restored.cell_ids == stored_columns.cell_ids
    assert restored.column_ids == stored_columns.column_ids
    assert restored.interval_indices == stored_columns.interval_indices
    assert restored.hard_first_thickness == stored_columns.hard_first_thickness
    assert restored.axial_refinement == stored_columns.axial_refinement
    assert restored.allow_schedule_change == stored_columns.allow_schedule_change
    assert restored.columns_id == stored_columns.columns_id


def test_layer_columns_stored_fields_refuse_negative_interval(
    stored_columns: MixedLayerColumns,
) -> None:
    recipe = model_structure_recipe(stored_columns)
    recipe["items"][recipe_field_position(recipe, "interval_indices")] = (
        model_structure_recipe((-1, 1))
    )
    malformed = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match=r"MixedLayerColumns\.ancestry"):
        MixedLayerColumns.from_stored_fields(malformed)


def test_layer_columns_stored_fields_refuse_forged_identity(
    stored_columns: MixedLayerColumns,
) -> None:
    recipe = model_structure_recipe(stored_columns)
    recipe["items"][recipe_field_position(recipe, "columns_id")] = model_structure_recipe(
        "forged"
    )
    malformed = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match=r"MixedLayerColumns\.columns_id"):
        MixedLayerColumns.from_stored_fields(malformed)


def test_mixed_hierarchy_stored_fields_preserve_allocator_and_identity(
    stored_hierarchy: MixedAdaptationHierarchy,
) -> None:
    restored = MixedAdaptationHierarchy.from_stored_fields(stored_hierarchy)
    assert restored is not stored_hierarchy
    assert tuple(restored.siblings) == tuple(stored_hierarchy.siblings)
    assert restored.siblings.bank_id == stored_hierarchy.siblings.bank_id
    assert restored.next_cell_id == stored_hierarchy.next_cell_id
    assert restored.next_vertex_id == stored_hierarchy.next_vertex_id
    assert tuple(restored.retired) == tuple(stored_hierarchy.retired)
    assert restored.retired.bank_id == stored_hierarchy.retired.bank_id
    assert restored.quotient_entities == stored_hierarchy.quotient_entities
    assert restored.layer_columns is stored_hierarchy.layer_columns
    assert restored.identity_cells == stored_hierarchy.identity_cells
    assert restored.hierarchy_id == stored_hierarchy.hierarchy_id


def test_mixed_hierarchy_stored_fields_refuse_negative_allocator(
    stored_hierarchy: MixedAdaptationHierarchy,
) -> None:
    recipe = model_structure_recipe(stored_hierarchy)
    recipe["items"][recipe_field_position(recipe, "next_cell_id")] = (
        model_structure_recipe(-1)
    )
    malformed = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match=r"MixedAdaptationHierarchy\.next_ids"):
        MixedAdaptationHierarchy.from_stored_fields(malformed)


def test_mixed_hierarchy_stored_fields_refuse_forged_identity(
    stored_hierarchy: MixedAdaptationHierarchy,
) -> None:
    recipe = model_structure_recipe(stored_hierarchy)
    recipe["items"][recipe_field_position(recipe, "hierarchy_id")] = (
        model_structure_recipe("forged")
    )
    malformed = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match=r"MixedAdaptationHierarchy\.hierarchy_id"):
        MixedAdaptationHierarchy.from_stored_fields(malformed)


_BANK_TEMPLATE = (
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
)


def test_mixed_sibling_bank_preserves_rows_and_refuses_forged_padding() -> None:
    signed = (((-0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), _BANK_TEMPLATE[1])
    periodic = Sibling(
        9,
        "b",
        "quadrilateral",
        (4, 5, 6, 7),
        (20, 21),
        signed,
        1,
        3,
        None,
        (4, 5, 4, 5),
        ((0, 0), (1, 0), (1, 1), (0, 1)),
    )
    plain = Sibling(
        3, "a", "triangle", (0, 1, 2), (10, 11, 12), _BANK_TEMPLATE, 0, None, 2, (), ()
    )
    bank = MixedSiblingBank((periodic, plain))
    assert tuple(bank) == (plain, periodic)
    assert bank.block_kinds == (("a", "triangle"), ("b", "quadrilateral"))
    # Signed zeros are distinct exact templates, ordered after equal values by bits.
    assert [np.signbit(template[0][0][0]) for template in bank.reference_templates] == [
        False,
        True,
    ]
    assert np.signbit(bank[-1].references[0][0][0])
    assert bank.parent_rows(np.asarray([9, 4, 3])).tolist() == [1, -1, 0]
    assert bank.all_child_ids().tolist() == [10, 11, 12, 20, 21]
    assert bank.child_ids.shape == (2, 3) and bank.periodic_shifts.shape == (2, 4, 2)
    assert not bank.child_ids.flags.writeable
    restored = MixedSiblingBank.from_stored_fields(bank)
    assert restored.bank_id == bank.bank_id and tuple(restored) == tuple(bank)
    children = np.array(bank.child_ids)
    children[1, 2] = 99
    forged = eqx.tree_at(lambda value: value.child_ids, bank, children)
    with pytest.raises(ValueError, match=r"MixedSiblingBank\.columns"):
        MixedSiblingBank.from_stored_fields(forged)
    with pytest.raises(ValueError, match=r"MixedSiblingBank\.periodic_shifts"):
        MixedSiblingBank((periodic._replace(periodic_shifts=((0, 0),)),))
    with pytest.raises(ValueError, match=r"MixedSiblingBank\.parent_ids"):
        MixedSiblingBank((plain, plain._replace(child_ids=(13,))))
    with pytest.raises(ValueError, match=r"MixedSiblingBank\.child_ids"):
        MixedSiblingBank((plain._replace(child_ids=(1 << 63,)),))


def test_mixed_retired_entity_bank_keeps_tuple_order_and_key_padding() -> None:
    rows = ((2, (5, 6, 7, -1), 40), (1, (0, 1), 30), (2, (5, 6, 7), 41))
    bank = MixedRetiredEntityBank(rows)
    assert tuple(bank) == tuple(sorted(rows))
    assert bank[-1] == (2, (5, 6, 7, -1), 40)
    assert bank.keys.shape == (3, 4) and bank.key_lengths.tolist() == [2, 3, 4]
    assert MixedRetiredEntityBank.from_stored_fields(bank).bank_id == bank.bank_id
    lengths = np.array(bank.key_lengths)
    lengths[0] = 1
    forged = eqx.tree_at(lambda value: value.key_lengths, bank, lengths)
    with pytest.raises(ValueError, match=r"MixedRetiredEntityBank\.columns"):
        MixedRetiredEntityBank.from_stored_fields(forged)


def test_mixed_hierarchy_recipe_size_is_independent_of_row_count() -> None:
    register_meshing_source_artifacts()

    def recipe_bytes(count: int) -> int:
        hierarchy = MixedAdaptationHierarchy(
            tuple(
                Sibling(
                    index,
                    "block",
                    "quadrilateral",
                    tuple(range(4 * index, 4 * index + 4)),
                    (1_000_000 + 2 * index, 1_000_001 + 2 * index),
                    _BANK_TEMPLATE,
                    0,
                    None,
                    None,
                    (),
                    (),
                )
                for index in range(count)
            ),
            next_cell_id=2_000_000,
            next_vertex_id=4_000_000,
            retired_entities=tuple(
                (1, (index, index + 1), 1_000_000 + index) for index in range(count)
            ),
        )
        restored = MixedAdaptationHierarchy.from_stored_fields(hierarchy)
        assert restored.hierarchy_id == hierarchy.hierarchy_id
        return len(json.dumps(model_structure_recipe(hierarchy)))

    assert abs(recipe_bytes(1024) - recipe_bytes(2)) < 512

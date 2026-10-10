#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry transitions of curved coordinate maps through native adaptation.

Nested-transition oracles use analytic shears ``x -> x + a q(y, z)`` of straight
simplex meshes: child maps are exact restrictions and measures stay unchanged.
Source-realization oracles instead use ``(x, y) -> (x, y(1 + x/4))``:
the physical measure changes from 1 to 9/8, density inventory remains 2, and
material history/flux follow their actual reference pullback and Piola laws.
"""

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellGeometrySpec, CellMesh, FiniteElementSpec
from phydrax.discretization._cell_geometry_transfer import (
    CellGeometryTransitionError,
    CellGeometryTransitionPolicy,
    transition_displaced_cell_geometry,
)
from phydrax.meshing import (
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshAdaptationStatus,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from phydrax.meshing._curving import _straight_geometry


_CONTRACT = phx.SpatialCoordinateContract.si()
_SHEAR = 0.1


def _shear(degree: int, dimension: int) -> Callable[[np.ndarray], np.ndarray]:
    """``x + a q`` with ``q`` a monomial of total ``degree`` in the other axes."""

    def term(points: np.ndarray) -> np.ndarray:
        if dimension == 2:
            return points[..., 1] ** degree
        return points[..., 1] ** (degree - 1) * points[..., 2]

    return term


def _apply(
    points: np.ndarray, term: Callable[[np.ndarray], np.ndarray], sign: float
) -> np.ndarray:
    moved = np.array(points, dtype=np.float64, copy=True)
    moved[..., 0] += sign * _SHEAR * term(points)
    return moved


def _triangles(size: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, size + 1)
    points = np.stack(np.meshgrid(axis, axis), axis=-1).reshape((-1, 2))
    cells = []
    for j, i in itertools.product(range(size), range(size)):
        a = j * (size + 1) + i
        cells.extend(((a, a + 1, a + size + 2), (a, a + size + 2, a + size + 1)))
    return CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32))


def _tetrahedra(size: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, size + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    points = points.reshape((-1, 3))
    cells = []
    for origin in itertools.product(range(size), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = list(origin)
            path = [(corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2]]
            for direction in order:
                corner[direction] += 1
                path.append((corner[0] * (size + 1) + corner[1]) * (size + 1) + corner[2])
            cells.append(path)
    cells_ = np.asarray(cells, dtype=np.int32)
    edges = points[cells_[:, 1:]] - points[cells_[:, :1]]
    negative = np.linalg.det(edges) < 0.0
    cells_[negative] = cells_[negative][:, [1, 0, 2, 3]]
    return CellMesh.from_tetrahedra(points, cells_)


def _curved(mesh: CellMesh, degree: int, term: Callable[[np.ndarray], np.ndarray]) -> Any:
    """Certified result whose coordinate map is the shear of the straight mesh."""

    mesh = phx.meshing.canonicalize_cell_mesh(mesh)
    straight = _straight_geometry(mesh, degree)
    names = [block.name for block in mesh.blocks]
    geometry = CellGeometrySpec(
        dict(zip(names, straight.elements, strict=True)),
        dict(zip(names, straight.geometry_dofs, strict=True)),
        _apply(np.asarray(straight.coordinates), term, 1.0),
    )
    moved = mesh.with_coordinates(
        _apply(np.asarray(mesh.coordinates), term, 1.0), numeric_version="sheared"
    )
    return phx.meshing.certify_cell_mesh(moved, _CONTRACT, geometry=geometry)


def _adapt(source: Any, refine: Any = (), coarsen: Any = (), **options: Any) -> Any:
    hierarchy = options.pop("hierarchy", None)
    policy = MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION, **options)
    request = MarkedMeshAdaptation(
        np.asarray(refine, dtype=np.int64),
        np.asarray(coarsen, dtype=np.int64),
        hierarchy=hierarchy,
    )
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )


def _map(result: Any, row: int, reference: np.ndarray) -> np.ndarray:
    geometry = result.geometry
    for element, routes in zip(geometry.elements, geometry.geometry_dofs, strict=True):
        if row < routes.shape[0]:
            route = np.asarray(routes)[row]
            values, _ = element.tabulate(reference)
            return np.asarray(values) @ np.asarray(geometry.coordinates)[route]
        row -= routes.shape[0]
    raise ValueError("The requested geometry cell row is absent.")


def _max_shear_deviation(result: Any, term: Any, samples: int) -> float:
    """Largest distance between each cell's map and the exact sheared simplex."""

    mesh = result.mesh
    dimension = mesh.topological_dimension
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    rng = np.random.default_rng(7)
    worst = 0.0
    for row in range(corners.shape[0]):
        barycentric = rng.dirichlet(np.ones(dimension + 1), size=samples)
        # The shear keeps y and z, so its inverse subtracts the same term.
        straight = _apply(corners[row], term, -1.0)
        exact = _apply(barycentric @ straight, term, 1.0)
        mapped = _map(result, row, barycentric[:, 1:])
        worst = max(worst, float(np.max(np.abs(mapped - exact))))
    return worst


@pytest.mark.parametrize(
    ("kind", "degree"),
    [("triangle", 2), ("triangle", 3), ("tetrahedron", 2), ("tetrahedron", 3)],
    ids=["p2-triangle", "p3-triangle", "p2-tetrahedron", "p3-tetrahedron"],
)
def test_native_bisection_restricts_curved_maps_exactly(kind: str, degree: int) -> None:
    dimension = 2 if kind == "triangle" else 3
    term = _shear(degree, dimension)
    source = _curved(_triangles(2) if dimension == 2 else _tetrahedra(1), degree, term)
    assert _max_shear_deviation(source, term, 5) < 1e-14
    refined = _adapt(source, (0, 1))

    target = refined.target
    transition = refined.geometry_transition
    assert refined.status is MeshAdaptationStatus.COMPLETE
    assert target.mesh.blocks[0].cell_count > source.mesh.blocks[0].cell_count
    assert target.geometry.elements[0].element_id == (
        source.geometry.elements[0].element_id
    )
    assert _max_shear_deviation(target, term, 9) < 1e-13
    assert transition.evidence.kind == "nested_restriction"
    assert transition.evidence.exact
    # Unit-determinant shear: the mapped measure is the straight unit measure.
    assert transition.evidence.measure_exact
    assert abs(transition.evidence.target_measure - 1.0) < 1e-12
    # The certified mesh corners are the map's vertex nodes, not straight midpoints.
    vertex_rows = np.asarray(target.mesh.blocks[0].vertices)
    node_rows = np.asarray(target.geometry.geometry_dofs[0])[
        :, [entity[0] for entity in target.geometry.elements[0].entity_dofs[0]]
    ]
    np.testing.assert_allclose(
        np.asarray(target.mesh.coordinates)[vertex_rows],
        np.asarray(target.geometry.coordinates)[node_rows],
        rtol=0.0,
        atol=0.0,
    )


def test_parent_witnesses_evaluate_the_source_map() -> None:
    term = _shear(3, 2)
    source = _curved(_triangles(2), 3, term)
    refined = _adapt(source, (0, 3, 5))
    parents = refined.parent_cells
    corners = refined.parent_reference_vertices
    rng = np.random.default_rng(3)
    for row in range(parents.size):
        barycentric = rng.dirichlet(np.ones(3), size=6)
        np.testing.assert_allclose(
            _map(refined.target, row, barycentric[:, 1:]),
            _map(source, int(parents[row]), barycentric @ corners[row]),
            rtol=0.0,
            atol=1e-14,
        )


def test_restored_parents_coarsen_back_to_the_source_map() -> None:
    term = _shear(2, 2)
    source = _curved(_triangles(2), 2, term)
    refined = _adapt(source, (0,))
    new_cells = np.setdiff1d(
        np.asarray(refined.target.mesh.blocks[0].global_ids),
        np.asarray(source.mesh.blocks[0].global_ids),
    )
    restored = _adapt(refined.target, (), new_cells, hierarchy=refined.hierarchy)

    evidence = restored.geometry_transition.evidence
    assert evidence.kind == "coarsening_interpolation"
    assert evidence.exact
    assert evidence.approximation_bound <= evidence.rounding_slack
    assert restored.target.mesh.topology_id == source.mesh.topology_id
    np.testing.assert_allclose(
        np.asarray(restored.target.geometry.coordinates),
        np.asarray(source.geometry.coordinates),
        rtol=0.0,
        atol=1e-14,
    )
    with pytest.raises(ValueError, match="no parent"):
        restored.parent_cells


def _wavy_refinement() -> tuple[Any, Any, np.ndarray]:
    """A refined mesh whose fine map is not a restriction of any coarse P2 map."""

    affine = phx.meshing.certify_cell_mesh(
        phx.meshing.canonicalize_cell_mesh(_triangles(2)), _CONTRACT
    )
    refined = _adapt(affine, (0,))
    mesh = refined.target.mesh
    straight = _straight_geometry(mesh, 2)
    points = np.asarray(straight.coordinates)
    bumped = points.copy()
    bumped[:, 0] += 0.02 * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    geometry = CellGeometrySpec(
        dict(zip((block.name for block in mesh.blocks), straight.elements, strict=True)),
        dict(
            zip(
                (block.name for block in mesh.blocks), straight.geometry_dofs, strict=True
            )
        ),
        bumped,
    )
    corners = np.asarray(mesh.coordinates).copy()
    for block, coordinate_element, route in zip(
        mesh.blocks, straight.elements, straight.geometry_dofs, strict=True
    ):
        assert isinstance(coordinate_element, FiniteElementSpec)
        corner_weights, _ = coordinate_element.tabulate(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
        )
        corners[np.asarray(block.vertices)] = (
            np.asarray(corner_weights) @ bumped[np.asarray(route)]
        )
    curved = phx.meshing.certify_cell_mesh(
        mesh.with_coordinates(corners, numeric_version="bumped"),
        _CONTRACT,
        geometry=geometry,
    )
    new_cells = np.setdiff1d(
        np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks]),
        np.concatenate([np.asarray(block.global_ids) for block in affine.mesh.blocks]),
    )
    return curved, refined.hierarchy, new_cells


def test_curved_coarsening_refuses_an_inexact_coarse_map_by_default() -> None:
    curved, hierarchy, new_cells = _wavy_refinement()
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _adapt(curved, (), new_cells, hierarchy=hierarchy)
    assert refusal.value.reason == "approximation_bound"
    assert refusal.value.measured > refusal.value.limit


def test_curved_coarsening_follows_its_declared_approximation_bound() -> None:
    curved, hierarchy, new_cells = _wavy_refinement()
    policy = CellGeometryTransitionPolicy(
        coarsening="bounded_interpolation", coarsening_tolerance=1e-2
    )
    coarse = _adapt(
        curved, (), new_cells, hierarchy=hierarchy, geometry_transition=policy
    )
    evidence = coarse.geometry_transition.evidence
    assert not evidence.exact
    assert 0.0 < evidence.approximation_bound <= evidence.approximation_tolerance
    # The certified sup bound dominates the deviation seen at fine-map nodes:
    # coarse nodes lie on the fine map, so sample the coarse map at fine nodes.
    witnesses = coarse.geometry_transition
    fine_ids = np.asarray(witnesses.coarsened_cell_ids)
    coarse_ids = np.asarray(witnesses.coarsened_into_ids)
    corners = np.asarray(witnesses.coarsened_reference_vertices)
    source_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in curved.mesh.blocks]
    )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in coarse.target.mesh.blocks]
    )
    nodes = np.asarray(curved.geometry.elements[0].reference_nodes)
    barycentric = np.concatenate((1.0 - nodes.sum(axis=1, keepdims=True), nodes), axis=1)
    observed = 0.0
    for fine, parent, reference in zip(fine_ids, coarse_ids, corners, strict=True):
        fine_row = int(np.flatnonzero(source_ids == fine)[0])
        coarse_row = int(np.flatnonzero(target_ids == parent)[0])
        observed = max(
            observed,
            float(
                np.max(
                    np.abs(
                        _map(coarse.target, coarse_row, barycentric @ reference)
                        - _map(curved, fine_row, nodes)
                    )
                )
            ),
        )
    assert observed <= evidence.approximation_bound


def test_metric_route_still_refuses_curved_sources_it_cannot_carry() -> None:
    term = _shear(2, 2)
    source = _curved(_triangles(2), 2, term)
    mesh = source.mesh
    vertices = mesh.entity_set(0)
    metric = phx.meshing.MeshMetricField(
        phx.meshing.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            0,
            vertices.entity_set_id,
            vertices.entity_ids,
        ),
        np.broadcast_to(16.0 * np.eye(2), (vertices.count, 2, 2)),
        minimum_size=0.25,
        maximum_size=0.25,
    )
    with pytest.raises(ValueError, match="no geometry transition"):
        prepare_mesh_adaptation(
            source,
            MetricMeshAdaptation(metric),
            policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_METRIC_2D),
        )


def test_displacement_keeps_curvature_and_moves_vertex_nodes() -> None:
    term = _shear(2, 2)
    source = _curved(_triangles(2), 2, term)
    offset = np.asarray((0.25, -0.5))
    moved = source.mesh.with_coordinates(
        np.asarray(source.mesh.coordinates) + offset, numeric_version="translated"
    )
    transition = transition_displaced_cell_geometry(source.mesh, source.geometry, moved)
    # A rigid translation moves every node, curved or not, by the same offset.
    np.testing.assert_allclose(
        np.asarray(transition.geometry.coordinates),
        np.asarray(source.geometry.coordinates) + offset,
        rtol=0.0,
        atol=1e-15,
    )
    assert transition.evidence.kind == "vertex_displacement"
    assert abs(transition.evidence.target_measure - 1.0) < 1e-12


def test_geometry_transition_refuses_beyond_its_evaluation_budget() -> None:
    source = _curved(_triangles(2), 2, _shear(2, 2))
    policy = CellGeometryTransitionPolicy(maximum_evaluations=10)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _adapt(source, (0,), geometry_transition=policy)
    assert refusal.value.reason == "resource_limit"


def test_incomplete_topology_edit_refuses_atomically() -> None:
    from phydrax.discretization._cell_geometry_transfer import NestedReferenceWitnesses
    from phydrax.meshing._topology_edit import (
        assemble_topology_edit,
        CellTopologyEdit,
        EntityRelations,
        TopologyEditBlock,
    )

    source = _triangles(1)
    ids = np.asarray(source.vertex_global_ids)
    before = np.asarray(source.coordinates).copy()
    block = source.blocks[0]
    witness = NestedReferenceWitnesses(
        np.asarray(block.global_ids)[:1],
        np.asarray(block.global_ids)[:1],
        np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
    )
    relations = tuple(
        EntityRelations(
            dimension,
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0, 1), dtype=np.int64),
            np.zeros(0, dtype=np.int32),
        )
        for dimension in range(3)
    )
    edit = CellTopologyEdit(
        "nested_refinement",
        before.copy(),
        ids,
        (
            TopologyEditBlock(
                block.name,
                block.cell_kind,
                block.cell_kind,
                np.asarray(block.vertices),
                np.asarray(block.global_ids),
            ),
        ),
        ids[:, None],
        np.ones((ids.size, 1), dtype=np.float64),
        np.ones((ids.size, 1), dtype=np.bool_),
        relations,
        refinement=witness,
    )
    with pytest.raises(ValueError, match="Every target cell"):
        assemble_topology_edit(source, edit, numeric_version="refused")
    np.testing.assert_array_equal(source.coordinates, before)
    np.testing.assert_array_equal(edit.coordinates, before)


def _coordinate_field(discretization: Any, result: Any) -> np.ndarray:
    """Physical-coordinate coefficients at the field's own nodal functionals."""

    routes = np.asarray(discretization.dof_maps[0].cell_dofs[0])
    nodes = np.asarray(result.geometry.geometry_dofs[0])
    values = np.zeros((discretization.dof_maps[0].global_dof_count, 2))
    weights, _ = result.geometry.elements[0].tabulate(
        discretization.elements[0][0].reference_nodes
    )
    local = np.asarray(weights) @ np.asarray(result.geometry.coordinates)[nodes]
    values[routes.reshape(-1)] = local.reshape((-1, 2))
    return values


def _discretization(result: Any) -> Any:
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 2)
    )
    return phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()


def test_nested_field_transfer_reproduces_curved_coordinate_fields() -> None:
    source = _curved(_triangles(2), 2, _shear(2, 2))
    refined = _adapt(source, (0, 4))
    coarse = _discretization(source)
    fine = _discretization(refined.target)

    with pytest.raises(ValueError, match="affine geometry"):
        phx.discretization.prepare_nested_field_transfer(
            coarse, fine, refined.parent_cells, field_name="u"
        )
    transfer = phx.discretization.prepare_nested_field_transfer(
        coarse,
        fine,
        refined.parent_cells,
        field_name="u",
        parent_reference_vertices=refined.parent_reference_vertices,
    )

    # Physical coordinates are P2 fields of an isoparametric P2 map whose
    # coefficients are the geometry nodes; the exact nested transfer must carry
    # the coarse coordinate field onto the refined one.
    np.testing.assert_allclose(
        np.asarray(transfer.transfer.apply(_coordinate_field(coarse, source))),
        _coordinate_field(fine, refined.target),
        rtol=0.0,
        atol=1e-13,
    )
    assert transfer.evidence.passed
    assert transfer.geometry.relation == "exact-restriction"
    assert transfer.transfer.preserves_constants
    assert not transfer.transfer.conservative


def _source_realization_case() -> tuple[
    CellMesh,
    CellGeometrySpec,
    CellMesh,
    CellGeometrySpec,
    phx.discretization.SourceGeometryRealization,
]:
    source = phx.meshing.canonicalize_cell_mesh(_triangles(1))
    old = CellGeometrySpec.affine(source)
    layout = _straight_geometry(source, 2)

    def moved(values: np.ndarray) -> np.ndarray:
        result = np.array(values, dtype=np.float64, copy=True)
        result[:, 1] *= 1.0 + result[:, 0] / 4.0
        return result

    target = source.with_coordinates(
        moved(np.asarray(source.coordinates)), numeric_version="source-realized"
    )
    names = tuple(block.name for block in source.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, layout.elements, strict=True)),
        dict(zip(names, layout.geometry_dofs, strict=True)),
        moved(np.asarray(layout.coordinates)),
    )
    old_validity = phx.discretization.certify_cell_geometry_validity(old, mesh=source)
    new_validity = phx.discretization.certify_cell_geometry_validity(
        geometry, mesh=target
    )
    old_embedding = phx.geometry.certify_global_embedding(source, old, old_validity)
    new_embedding = phx.geometry.certify_global_embedding(target, geometry, new_validity)
    realization = phx.discretization.prepare_source_geometry_realization(
        source,
        old,
        target,
        geometry,
        source_embedding=old_embedding,
        target_embedding=new_embedding,
        policy=CellGeometryTransitionPolicy(
            reconstruction="source_realization", reconstruction_tolerance=0.3
        ),
        maximum_storage_bytes=16 * 1024**2,
    )
    return source, old, target, geometry, realization


def test_source_realization_records_changed_physical_support_and_enclosed_measures() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    transition = realization.transition
    assert source.topology_id == target.topology_id
    assert transition.source_geometry_id != transition.target_geometry_id
    assert transition.evidence.source_measure == pytest.approx(1.0, abs=1e-14)
    assert transition.evidence.target_measure == pytest.approx(9.0 / 8.0, abs=1e-14)
    assert not transition.evidence.exact
    assert transition.evidence.coverage_defect is None
    assert 0.0 < transition.evidence.approximation_bound <= 0.3
    assert transition.evidence.source_measure_error_bound is not None
    assert transition.evidence.target_measure_error_bound is not None
    assert transition.evidence.source_measure_error_bound < 1e-12
    assert transition.evidence.target_measure_error_bound < 1e-12


def test_source_realization_density_projection_conserves_inventory_not_constant_values() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    field = phx.discretization.FiniteElementFieldSpec(
        "density", phx.discretization.discontinuous_element("triangle", 0)
    )
    before = phx.discretization.FiniteElementPlan(
        source, field, coordinate_spec=old
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=new
    ).prepare()
    transfer = phx.discretization.prepare_source_realization_field_transfer(
        before,
        after,
        realization,
        field_name="density",
        source_geometry=old,
        target_geometry=new,
        semantics="conservative-density",
    )
    values = np.full(before.dof_maps[0].global_dof_count, 2.0, dtype=np.float64)
    carried = np.asarray(transfer.transfer.apply(values))
    expected = np.empty_like(carried)
    for row, route in enumerate(np.asarray(after.dof_maps[0].cell_dofs[0])):
        expected[route] = (
            2.0
            * np.asarray(realization.source_cell_measures)[row]
            / np.asarray(realization.target_cell_measures)[row]
        )
    np.testing.assert_allclose(carried, expected, rtol=0.0, atol=1e-13)
    assert not np.allclose(carried, 2.0, rtol=0.0, atol=1e-13)
    assert float(np.dot(carried, np.asarray(transfer.target_measures))) == pytest.approx(
        2.0, abs=1e-13
    )
    assert transfer.geometry.relation == "source-realization"
    assert transfer.geometry.coverage_defect is None
    assert transfer.evidence.passed
    assert transfer.evidence.bound("content") < 1e-10
    assert transfer.transfer.conservative
    assert not transfer.transfer.preserves_constants


def test_source_realization_intensive_history_is_material_not_world_point_interpolation() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    field = phx.discretization.FiniteElementFieldSpec(
        "history", phx.discretization.lagrange_element("triangle", 1)
    )
    before = phx.discretization.FiniteElementPlan(
        source, field, coordinate_spec=old
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=new
    ).prepare()
    transfer = phx.discretization.prepare_source_realization_field_transfer(
        before,
        after,
        realization,
        field_name="history",
        source_geometry=old,
        target_geometry=new,
        semantics="intensive",
    )
    values = before.dof_maps[0].dof_coordinates[:, 1] + 0.5
    carried = np.asarray(transfer.transfer.apply(values))
    np.testing.assert_allclose(carried, values, rtol=0.0, atol=0.0)
    assert not np.allclose(
        carried, np.asarray(after.dof_maps[0].dof_coordinates)[:, 1] + 0.5
    )
    assert transfer.semantics == "material-pullback"
    assert not transfer.transfer.conservative


def test_source_realization_flux_retains_material_cochain_and_actual_piola_values() -> (
    None
):
    source, old, target, new, realization = _source_realization_case()
    element = phx.discretization.form_element(
        "triangle", 1, 1, twist="twisted", proxy="flux"
    )
    field = phx.discretization.FiniteElementFieldSpec("flux", element)
    before = phx.discretization.FiniteElementPlan(
        source, field, coordinate_spec=old
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        target, field, coordinate_spec=new
    ).prepare()
    transfer = phx.discretization.prepare_source_realization_field_transfer(
        before,
        after,
        realization,
        field_name="flux",
        source_geometry=old,
        target_geometry=new,
        semantics="material-compatible",
    )
    values = np.linspace(0.3, 1.4, before.dof_maps[0].global_dof_count)
    carried = np.asarray(transfer.transfer.apply(values))
    points = np.asarray(((0.2, 0.3), (0.6, 0.1)), dtype=np.float64)

    def physical_values(
        space: phx.discretization.FiniteElementDiscretization, coefficients: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = space.evaluate_block_geometry(
            "flux", 0, space.default_runtime.coordinates, points, np.ones(points.shape[0])
        )
        dof_map = space.dof_maps[0]
        local = np.einsum(
            "cij,cj->ci",
            np.asarray(dof_map.cell_transforms[0]),
            coefficients[np.asarray(dof_map.cell_dofs[0])],
        )
        basis = np.asarray(geometry.basis_values).reshape(
            (local.shape[0], points.shape[0], local.shape[1], 2)
        )
        return np.asarray(geometry.physical_points), np.einsum(
            "cqiv,ci->cqv", basis, local
        )

    old_points, old_values = physical_values(before, values)
    _, new_values = physical_values(after, carried)
    determinant = 1.0 + old_points[..., 0] / 4.0
    expected = np.empty_like(old_values)
    expected[..., 0] = old_values[..., 0] / determinant
    expected[..., 1] = old_values[..., 1] + old_points[..., 1] * old_values[..., 0] / (
        4.0 * determinant
    )
    np.testing.assert_allclose(new_values, expected, rtol=2e-12, atol=2e-12)
    assert not np.allclose(new_values, old_values, rtol=1e-10, atol=1e-10)
    assert transfer.evidence.passed


@pytest.mark.parametrize(
    ("kind", "expected"), (("triangle", (409, 400)), ("quadrilateral", (1059, 400)))
)
def test_high_degree_sqrt_density_keeps_actual_weighted_reference_moments(
    kind: str,
    expected: tuple[int, int],
) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry_transfer import (
        _certified_sqrt_polynomial_integral_state,
    )
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        multiply,
        Polynomial,
    )

    root: Polynomial = {
        (0, 0): Fraction(1),
        (3, 0): Fraction(1, 10),
        (0, 3): Fraction(1, 10),
    }
    weight: Polynomial = {(0, 0): Fraction(1), (1, 0): Fraction(1), (0, 1): Fraction(2)}
    ledger = CoordinateEnclosureBudget(1_000_000, 16 * 1024**2)
    with ledger.activate():
        value, error = _certified_sqrt_polynomial_integral_state(
            multiply(root, root),
            weight,
            kind,
            Fraction(1, 10**11),
            1e-11,
            10000,
            32,
            [0, 1_000_000],
        )
    assert abs(Fraction(value) - Fraction(*expected)) <= Fraction(error)
    assert error <= 1e-11 * float(Fraction(*expected))


@pytest.mark.parametrize("increment", ((1, 10**11), (1, 2)))
def test_area_reuse_encloses_changed_gram_or_integrates_it_fresh(
    increment: tuple[int, int],
) -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry_transfer import (
        _reuse_embedded_polynomial_area,
    )
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        multiply,
        Polynomial,
    )

    delta = Fraction(*increment)
    root: Polynomial = {(0, 0): Fraction(2), (1, 0): delta}
    reference: tuple[Polynomial, Fraction, float, float] = (
        {(0, 0): Fraction(4)},
        Fraction(4),
        1.0,
        0.0,
    )
    ledger = CoordinateEnclosureBudget(1_000_000, 16 * 1024**2)
    with ledger.activate():
        (value, error), prepared = _reuse_embedded_polynomial_area(
            multiply(root, root),
            "triangle",
            [reference],
            Fraction(1, 10**10),
            1e-10,
            10000,
            32,
            [0, 1_000_000],
        )
    assert abs(Fraction(value) - (Fraction(1) + delta / 6)) <= Fraction(error)
    assert error <= max(1e-10, 1e-10 * (value - error))
    if delta < Fraction(1, 1000):
        assert value == 1.0
        assert error > 0.0
        assert prepared is None
    else:
        assert value > 1.08
        assert prepared is not None


def test_exact_polynomial_square_preserves_cancelled_consumer_values_under_work_cap() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        evaluate,
        multiply,
        Polynomial,
    )

    polynomial: Polynomial = {
        (a, b): Fraction((-1) ** (a + b) * (a + 1), 2 ** (a + b + 3))
        for a in range(11)
        for b in range(11 - a)
    }
    ledger = CoordinateEnclosureBudget(2800, 16 * 1024**2)
    with ledger.activate():
        squared = multiply(polynomial, polynomial)
    for point in (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(-2, 5)),
        (Fraction(7, 6), Fraction(2, 3)),
    ):
        assert evaluate(squared, point) == evaluate(polynomial, point) ** 2
    cancellation: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(-2),
        (2, 0): Fraction(1),
    }
    squared = multiply(cancellation, cancellation)
    assert evaluate(squared, (Fraction(1), Fraction(0))) == 0
    assert evaluate(squared, (Fraction(3), Fraction(0))) == 16


def test_scientific_block_authority_survives_hierarchy_archive(tmp_path: Path) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._bisection import BisectionHierarchy

    _, hierarchy, _ = _wavy_refinement()
    receipt = write_meshing_source_closure(tmp_path / "scientific-hierarchy", hierarchy)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, BisectionHierarchy)
    assert restored.hierarchy_id == hierarchy.hierarchy_id
    np.testing.assert_array_equal(
        restored.scientific_cell_ids, hierarchy.scientific_cell_ids
    )
    np.testing.assert_array_equal(
        restored.scientific_block_ids, hierarchy.scientific_block_ids
    )
    assert np.unique(np.asarray(restored.scientific_block_ids)).size == 1


@pytest.mark.parametrize("coarsening", (False, True))
def test_exact_affine_correspondence_bounds_actual_corner_difference(
    coarsening: bool,
) -> None:
    from phydrax.discretization._cell_geometry_transfer import (
        _plc_nested_approximation,
        NestedReferenceWitnesses,
    )
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    source = _tetrahedra(1)
    coordinates = np.asarray(source.coordinates).copy()
    coordinates[0, 0] += 0.125
    target = source.with_coordinates(coordinates, numeric_version="perturbed-map")
    ids = np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
    reference = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0)
    witness = NestedReferenceWitnesses(
        ids,
        ids,
        np.broadcast_to(reference, (ids.size, 4, 3)).copy(),
    )
    empty = NestedReferenceWitnesses(
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 4, 3), dtype=np.float64),
    )
    ledger = CoordinateEnclosureBudget(100_000, 16 * 1024**2)
    with ledger.activate():
        bounds = _plc_nested_approximation(
            source,
            CellGeometrySpec.affine(source),
            target,
            CellGeometrySpec.affine(target),
            empty if coarsening else witness,
            witness if coarsening else empty,
        )
    assert bounds == ((0.0, 0.125) if coarsening else (0.125, 0.0))
    assert ledger.work_units > 0


def test_constant_embedded_area_work_is_independent_of_prior_candidates() -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry_transfer import (
        _reuse_embedded_polynomial_area,
    )
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    def measure(candidate_count: int) -> tuple[tuple[float, float], int]:
        ledger = CoordinateEnclosureBudget(1_000_000, 64 * 1024**2)
        work = [0, 1_000_000]
        with ledger.activate():
            _, reference = _reuse_embedded_polynomial_area(
                {(0, 0): Fraction(1), (2, 0): Fraction(1, 4)},
                "triangle",
                [],
                Fraction(1, 1000),
                0.0,
                1000,
                32,
                work,
            )
            if reference is None:
                raise RuntimeError(
                    "The positive varying Gram must earn its reference proof."
                )
            starting_work = ledger.work_units
            result, _ = _reuse_embedded_polynomial_area(
                {(0, 0): Fraction(2)},
                "triangle",
                [reference] * candidate_count,
                Fraction(1, 10**10),
                0.0,
                1000,
                32,
                work,
            )
        return result, ledger.work_units - starting_work

    empty, empty_work = measure(0)
    crowded, crowded_work = measure(100)
    assert crowded == empty
    assert crowded_work == empty_work > 0
    value, error = crowded
    assert abs(value - np.sqrt(2.0) / 2.0) <= error

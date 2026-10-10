#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import jax
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import (
    _certified_cell_measures,
    _certify_nested_geometry_pairs,
    _integrate_mapped_polynomial,
    _mapped_density_expression,
    CellGeometryTransition,
    CellGeometryTransitionError,
    CellGeometryTransitionPolicy,
    transition_nested_cell_geometry,
)
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._nested_reference import _NestedReferencePair
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem import (
    discontinuous_element,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementRuntimeData,
    FiniteElementSpec,
)
from phydrax.discretization.fem._generic import _degree_aware_reference_rule
from phydrax.discretization.fem._local_provider import (
    _require_tabulated_coordinate_element,
    FiniteElementGeometryActions,
)
from phydrax.discretization.fem._topology_transfer import (
    _prepare_mapped_nested_dg_transfer,
)
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._plc_mapped_support import plc_mapped_carrier
from phydrax.meshing._topology_edit import assemble_topology_edit


jax.config.update("jax_enable_x64", True)


def _curved(kind: str) -> tuple[CellMesh, CellGeometrySpec]:
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=float).copy()
    vertices[:, 1] += 0.05 * vertices[:, 0] * vertices[:, 2]
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "volume", kind, np.arange(len(vertices))[None], global_ids=np.array([17])
            ),
        ),
        vertex_global_ids=np.arange(100, 100 + len(vertices)),
    )
    element = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(element.reference_nodes).copy()
    nodes[:, 1] += 0.05 * nodes[:, 0] * nodes[:, 2]
    geometry = CellGeometrySpec(
        {"volume": element}, {"volume": np.arange(len(nodes))[None]}, nodes
    )
    return mesh, geometry


def _space(
    mesh: CellMesh, geometry: CellGeometrySpec, degree: int
) -> FiniteElementDiscretization:
    return FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "u",
            {
                block.name: discontinuous_element(block.cell_kind, degree)
                for block in mesh.blocks
            },
            component_shape=(2,),
        ),
        coordinate_spec=geometry,
    ).prepare()


def _content(
    space: FiniteElementDiscretization,
    geometry: CellGeometrySpec,
    coefficients: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Independent physical quadrature, not the transfer's reported measures."""
    total = np.zeros(2)
    elements, routes, coordinates = geometry.resolve(space.mesh)
    for field, coordinate, route, dofs in zip(
        space.elements[0], elements, routes, space.dof_maps[0].cell_dofs, strict=True
    ):
        points, weights = _degree_aware_reference_rule(field.cell_kind, 14)
        basis, _ = field.tabulate(points)
        _, gradients = _require_tabulated_coordinate_element(coordinate).tabulate(points)
        for cell, field_route in zip(np.asarray(route), np.asarray(dofs), strict=True):
            jacobian = np.einsum(
                "mD,qmd->qDd", np.asarray(coordinates)[cell], np.asarray(gradients)
            )
            density = (
                np.linalg.det(jacobian)
                if jacobian.shape[-2] == jacobian.shape[-1]
                else np.linalg.norm(np.cross(jacobian[..., 0], jacobian[..., 1]), axis=-1)
            )
            values = np.asarray(basis) @ coefficients[field_route]
            total += np.einsum("q,q,qk->k", np.asarray(weights), density, values)
    return total


@pytest.mark.parametrize(
    "kind,volume",
    [("prism", 0.5), ("hexahedron", 1.0), ("pyramid", 1 / 3), ("tetrahedron", 1 / 6)],
)
def test_curved_measures_have_outward_integration_error(kind: str, volume: float) -> None:
    mesh, geometry = _curved(kind)
    values, errors, exact = _certified_cell_measures(mesh, geometry)
    assert exact
    assert abs(values[0] - volume) <= errors[0]
    assert 0 < errors[0] < 1e-14


@pytest.mark.parametrize("kind", ["prism", "hexahedron", "pyramid", "tetrahedron"])
@pytest.mark.parametrize("degree", [0, 1])
def test_actual_scalar_dg_identity_and_component_transpose(
    kind: str, degree: int
) -> None:
    mesh, geometry = _curved(kind)
    space = _space(mesh, geometry, degree)
    prepared = _prepare_mapped_nested_dg_transfer(
        space,
        space,
        geometry,
        geometry,
        field_name="u",
        parent_cells=np.array([0]),
        parent_reference_vertices=np.asarray(reference_cell_topology(kind).vertices)[
            None
        ],
    )
    n = space.dof_maps[0].global_dof_count
    coefficients = np.asarray(
        np.column_stack((np.arange(n) + 1.0, np.arange(n) ** 2 - 0.3)), dtype=np.float64
    )
    dual = np.column_stack((np.arange(n) * 0.2 - 1, np.arange(n) * 0.1 + 2))
    transferred = np.asarray(prepared.transfer.apply(coefficients))
    np.testing.assert_allclose(transferred, coefficients, atol=1e-11)
    np.testing.assert_allclose(
        np.vdot(transferred, dual),
        np.vdot(coefficients, prepared.transfer.pullback(dual)),
        atol=1e-11,
    )
    content_defect = np.abs(
        _content(space, geometry, transferred) - _content(space, geometry, coefficients)
    )
    content_bound = prepared.evidence.bound("content") * np.sum(
        np.abs(coefficients), axis=0
    )
    assert np.all(content_defect <= content_bound + 1e-12)
    assert prepared.evidence.passed


@pytest.mark.parametrize("kind", ["prism", "hexahedron", "pyramid", "tetrahedron"])
@pytest.mark.parametrize("degree", [0, 1])
def test_curved_native_refine_and_complete_patch_coarsen(kind: str, degree: int) -> None:
    source_mesh, source_geometry = _curved(kind)
    before = np.asarray(source_geometry.coordinates).copy()
    outcome = adapt_mixed_mesh(source_mesh, refine_cell_ids=np.array([17]))
    fine_mesh, _, _ = assemble_topology_edit(
        source_mesh, outcome.edit, numeric_version="dg-fine"
    )
    restriction = transition_nested_cell_geometry(
        source_mesh,
        source_geometry,
        fine_mesh,
        CellGeometrySpec.affine(fine_mesh),
        refinement=outcome.edit.refinement,
    )
    source, fine = (
        _space(source_mesh, source_geometry, degree),
        _space(fine_mesh, restriction.geometry, degree),
    )
    forward = _prepare_mapped_nested_dg_transfer(
        source,
        fine,
        source_geometry,
        restriction.geometry,
        field_name="u",
        geometry_transition=restriction,
    )
    n = source.dof_maps[0].global_dof_count
    coefficients = np.asarray(
        np.column_stack((np.arange(n) + 0.4, np.arange(n) ** 2 + 2.0)), dtype=np.float64
    )
    fine_values = np.asarray(forward.transfer.apply(coefficients))
    np.testing.assert_allclose(
        _content(fine, restriction.geometry, fine_values),
        _content(source, source_geometry, coefficients),
        atol=2e-11,
    )
    ids = np.concatenate([np.asarray(block.global_ids) for block in fine_mesh.blocks])
    coarsening = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.empty(0, dtype=np.int64),
        coarsen_cell_ids=ids,
        hierarchy=outcome.hierarchy,
    )
    coarse_mesh, _, _ = assemble_topology_edit(
        fine_mesh, coarsening.edit, numeric_version="dg-coarse"
    )
    restoration = transition_nested_cell_geometry(
        fine_mesh,
        restriction.geometry,
        coarse_mesh,
        CellGeometrySpec.affine(coarse_mesh),
        refinement=coarsening.edit.refinement,
        coarsening=coarsening.edit.coarsening,
    )
    coarse = _space(coarse_mesh, restoration.geometry, degree)
    reverse = _prepare_mapped_nested_dg_transfer(
        fine,
        coarse,
        restriction.geometry,
        restoration.geometry,
        field_name="u",
        geometry_transition=restoration,
    )
    restored = np.asarray(reverse.transfer.apply(fine_values))
    np.testing.assert_allclose(restored, coefficients, atol=2e-10)
    np.testing.assert_allclose(
        _content(coarse, restoration.geometry, restored),
        _content(source, source_geometry, coefficients),
        atol=2e-11,
    )
    np.testing.assert_array_equal(source_geometry.coordinates, before)
    np.testing.assert_array_equal(coarse_mesh.blocks[0].global_ids, [17])


@pytest.mark.parametrize("kind", ["hexahedron", "tetrahedron"])
def test_mixed_transition_charges_actual_measure_work_to_its_budget(kind: str) -> None:
    source_mesh, source_geometry = _curved(kind)
    outcome = adapt_mixed_mesh(source_mesh, refine_cell_ids=np.array([17]))
    fine_mesh, _, _ = assemble_topology_edit(
        source_mesh, outcome.edit, numeric_version="budget-fine"
    )

    def transition(maximum_evaluations: int) -> CellGeometryTransition:
        return transition_nested_cell_geometry(
            source_mesh,
            source_geometry,
            fine_mesh,
            CellGeometrySpec.affine(fine_mesh),
            refinement=outcome.edit.refinement,
            policy=CellGeometryTransitionPolicy(maximum_evaluations=maximum_evaluations),
        )

    used = transition(1 << 26).evidence.evaluation_count
    exact = transition(used)
    assert exact.evidence.evaluation_count == used
    with pytest.raises(CellGeometryTransitionError) as refusal:
        transition(used - 1)
    assert refusal.value.reason == "resource_limit"
    assert refusal.value.measured > refusal.value.limit == used - 1


def test_refined_carrier_corners_are_rounded_exact_map_images() -> None:
    vertices = (
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        + np.asarray((0.1, 0.2, 0.3)) / 3.0
    )
    mesh = CellMesh(
        vertices,
        (CellBlock("v", "tetrahedron", np.arange(4)[None], global_ids=np.array([17])),),
    )
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.array([17]))
    fine, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="rounded-fine"
    )
    transition = transition_nested_cell_geometry(
        mesh,
        CellGeometrySpec.affine(mesh),
        fine,
        CellGeometrySpec.affine(fine),
        refinement=outcome.edit.refinement,
    )
    published = np.asarray(transition.vertex_coordinates)
    exact_midpoints = tuple(
        tuple(
            (
                Fraction(float(vertices[first, axis]))
                + Fraction(float(vertices[second, axis]))
            )
            / 2
            for axis in range(3)
        )
        for first, second in reference_cell_topology("tetrahedron").entities[1]
    )
    # The hierarchy's edge-midpoint children publish the correctly rounded
    # image of every exact affine reference corner.
    assert any(
        Fraction(float(value)) != value
        for midpoint in exact_midpoints
        for value in midpoint
    )
    midpoint_rows = []
    for exact in exact_midpoints:
        expected = np.asarray([float(value) for value in exact], dtype=np.float64)
        matches = np.flatnonzero(np.all(published == expected, axis=1))
        assert matches.size == 1
        midpoint_rows.append(int(matches[0]))
    perturbed = published.copy()
    perturbed[midpoint_rows[0], 0] = np.nextafter(perturbed[midpoint_rows[0], 0], np.inf)
    with pytest.raises(ValueError, match="correctly rounded"):
        plc_mapped_carrier(
            fine.with_coordinates(perturbed, numeric_version="perturbed-carrier"),
            transition.geometry,
        )


def test_wrong_geometry_and_scientific_root_leave_source_unchanged() -> None:
    mesh, geometry = _curved("hexahedron")
    before = np.asarray(geometry.coordinates).copy()
    pair = (_NestedReferencePair(0, 0, False, np.eye(3), np.zeros(3)),)
    wrong_nodes = before.copy()
    wrong_nodes[:, 1] += 0.01
    wrong = CellGeometrySpec(
        {"volume": geometry.elements[0]},
        {"volume": geometry.geometry_dofs[0]},
        wrong_nodes,
    )
    with pytest.raises(ValueError, match="same physical map"):
        _certify_nested_geometry_pairs(mesh, geometry, mesh, wrong, pair)
    root = CellGeometryRestrictionSource(
        "unrelated-geometry",
        mesh.topology_id,
        {"volume": np.array([17])},
        {
            "volume": np.asarray(mesh.vertex_global_ids)[
                np.asarray(mesh.blocks[0].vertices)
            ]
        },
    )
    coordinate = _require_tabulated_coordinate_element(geometry.elements[0])
    rooted = CellGeometrySpec(
        {
            "volume": RestrictedCellGeometryElement(
                coordinate, "hexahedron", np.eye(3), np.zeros(3)
            )
        },
        {"volume": geometry.geometry_dofs[0]},
        before,
        restriction_source=root,
    )
    with pytest.raises(ValueError, match="scientific root"):
        _certify_nested_geometry_pairs(mesh, geometry, mesh, rooted, pair)
    np.testing.assert_array_equal(geometry.coordinates, before)


@pytest.mark.parametrize("degree", [0, 1])
def test_connected_native_hex_pyramid_keeps_content_and_inventory(degree: int) -> None:
    cube = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=float)
    vertices = np.concatenate((cube, [[0.5, 0.5, 2.0]]))
    vertices[:, 1] += 0.05 * vertices[:, 0] * vertices[:, 2]
    mesh = CellMesh(
        vertices,
        (
            CellBlock("hex", "hexahedron", np.arange(8)[None], global_ids=np.array([10])),
            CellBlock(
                "cap", "pyramid", np.array([[4, 5, 6, 7, 8]]), global_ids=np.array([11])
            ),
        ),
    )
    elements: dict[str, FiniteElementSpec] = {}
    routes: dict[str, NDArray[np.int64]] = {}
    coordinates: list[NDArray[np.float64]] = []
    offset = 0
    for block in mesh.blocks:
        element = coordinate_lagrange_element(block.cell_kind, 2)
        nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
        if block.cell_kind == "pyramid":
            nodes[:, 2] += 1
        nodes[:, 1] += 0.05 * nodes[:, 0] * nodes[:, 2]
        elements[block.name], routes[block.name] = (
            element,
            np.arange(offset, offset + len(nodes), dtype=np.int64)[None],
        )
        coordinates.append(nodes)
        offset += len(nodes)
    geometry = CellGeometrySpec(elements, routes, np.concatenate(coordinates))
    source = _space(mesh, geometry, degree)
    count = source.dof_maps[0].global_dof_count
    coefficients = np.asarray(
        np.column_stack((np.arange(count) * 0.3 + 0.8, np.arange(count) ** 2 * 0.1 - 2)),
        dtype=np.float64,
    )
    before = np.asarray(geometry.coordinates).copy()
    outcome = adapt_mixed_mesh(mesh, refine_cell_ids=np.array([10]))
    fine_mesh, _, _ = assemble_topology_edit(
        mesh, outcome.edit, numeric_version="mixed-dg-fine"
    )
    restriction = transition_nested_cell_geometry(
        mesh,
        geometry,
        fine_mesh,
        CellGeometrySpec.affine(fine_mesh),
        refinement=outcome.edit.refinement,
    )
    fine = _space(fine_mesh, restriction.geometry, degree)
    forward = _prepare_mapped_nested_dg_transfer(
        source,
        fine,
        geometry,
        restriction.geometry,
        field_name="u",
        geometry_transition=restriction,
    )
    fine_values = np.asarray(forward.transfer.apply(coefficients))
    expected = _content(source, geometry, coefficients)
    np.testing.assert_allclose(
        _content(fine, restriction.geometry, fine_values), expected, atol=2e-10
    )
    fine_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in fine_mesh.blocks]
    )
    incomplete = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.empty(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids[:1],
        hierarchy=outcome.hierarchy,
    )
    np.testing.assert_array_equal(
        incomplete.evidence.rejected_coarsening_ids, fine_ids[:1]
    )
    stale_geometry = CellGeometrySpec(
        {
            block.name: element
            for block, element in zip(
                fine_mesh.blocks, restriction.geometry.elements, strict=True
            )
        },
        {
            block.name: route
            for block, route in zip(
                fine_mesh.blocks, restriction.geometry.geometry_dofs, strict=True
            )
        },
        np.asarray(restriction.geometry.coordinates) + 0.001,
    )
    with pytest.raises(ValueError, match="actual source/target maps"):
        _prepare_mapped_nested_dg_transfer(
            source,
            fine,
            geometry,
            stale_geometry,
            field_name="u",
            geometry_transition=restriction,
        )
    coarsening = adapt_mixed_mesh(
        fine_mesh,
        refine_cell_ids=np.empty(0, dtype=np.int64),
        coarsen_cell_ids=fine_ids,
        hierarchy=outcome.hierarchy,
    )
    coarse_mesh, _, _ = assemble_topology_edit(
        fine_mesh, coarsening.edit, numeric_version="mixed-dg-coarse"
    )
    restoration = transition_nested_cell_geometry(
        fine_mesh,
        restriction.geometry,
        coarse_mesh,
        CellGeometrySpec.affine(coarse_mesh),
        refinement=coarsening.edit.refinement,
        coarsening=coarsening.edit.coarsening,
    )
    coarse = _space(coarse_mesh, restoration.geometry, degree)
    reverse = _prepare_mapped_nested_dg_transfer(
        fine,
        coarse,
        restriction.geometry,
        restoration.geometry,
        field_name="u",
        geometry_transition=restoration,
    )
    restored = np.asarray(reverse.transfer.apply(fine_values))
    np.testing.assert_allclose(
        _content(coarse, restoration.geometry, restored), expected, atol=2e-10
    )
    source_cells = {
        int(identifier): coefficients[route]
        for block, cell_routes in zip(
            source.mesh.blocks, source.dof_maps[0].cell_dofs, strict=True
        )
        for identifier, route in zip(
            np.asarray(block.global_ids), np.asarray(cell_routes), strict=True
        )
    }
    for block, cell_routes in zip(
        coarse.mesh.blocks, coarse.dof_maps[0].cell_dofs, strict=True
    ):
        for identifier, route in zip(
            np.asarray(block.global_ids), np.asarray(cell_routes), strict=True
        ):
            np.testing.assert_allclose(
                restored[route], source_cells[int(identifier)], atol=2e-9
            )
    assert {
        int(value)
        for block in coarse_mesh.blocks
        for value in np.asarray(block.global_ids)
    } == {10, 11}
    assert {block.cell_kind for block in coarse_mesh.blocks} == {"hexahedron", "pyramid"}
    np.testing.assert_array_equal(geometry.coordinates, before)


def test_curved_pyramid_apex_chart_cancellation_integrates_nested_restrictions() -> None:
    _, geometry = _curved("pyramid")
    root = _require_tabulated_coordinate_element(geometry.elements[0])
    local = np.asarray(geometry.coordinates)
    total, error = 0.0, 0.0
    for x in (0.0, 0.5):
        for y in (0.0, 0.5):
            matrix = np.array([[0.5, 0, 0.25 - x], [0, 0.5, 0.25 - y], [0, 0, 1.0]])
            child = RestrictedCellGeometryElement(
                root, "pyramid", matrix, np.array([x, y, 0.0])
            )
            value, bound = _integrate_mapped_polynomial(
                _mapped_density_expression(child, local), "pyramid"
            )
            assert abs(value - 1 / 12) <= bound
            total += value
            error += bound
            grandchild = RestrictedCellGeometryElement(
                child, "pyramid", matrix, np.array([x, y, 0.0])
            )
            grandvalue, grandbound = _integrate_mapped_polynomial(
                _mapped_density_expression(grandchild, local), "pyramid"
            )
            assert abs(grandvalue - 1 / 48) <= grandbound
    assert abs(total - 1 / 3) <= error


def test_incomplete_reference_patch_refuses_without_changing_source() -> None:
    mesh, geometry = _curved("hexahedron")
    source = _space(mesh, geometry, 1)
    before = np.asarray(geometry.coordinates).copy()
    coefficients = np.arange(source.dof_maps[0].global_dof_count * 2.0).reshape(-1, 2)
    values_before = coefficients.copy()
    corners = np.asarray(reference_cell_topology("hexahedron").vertices).copy()
    corners[:, 0] *= 0.5
    with pytest.raises(ValueError, match="complete parent measure"):
        _prepare_mapped_nested_dg_transfer(
            source,
            source,
            geometry,
            geometry,
            field_name="u",
            parent_cells=np.array([0]),
            parent_reference_vertices=corners[None],
        )
    np.testing.assert_array_equal(geometry.coordinates, before)
    np.testing.assert_array_equal(coefficients, values_before)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("hexahedron", 1.05),
        ("prism", 31 / 60),
        ("pyramid", 0.35),
        ("tetrahedron", 41 / 240),
    ],
)
def test_nonlinear_density_matches_independent_physical_content(
    kind: str, expected: float
) -> None:
    mesh, old_geometry = _curved(kind)
    controls = np.asarray(old_geometry.coordinates).copy()
    coordinate = old_geometry.elements[0]
    if not isinstance(coordinate, FiniteElementSpec):
        raise TypeError(
            "The curved fixture requires canonical finite-element coordinate nodes."
        )
    reference = np.asarray(coordinate.reference_nodes)
    controls[:, 1] += 0.05 * reference[:, 1] ** 2
    corners = np.asarray(mesh.coordinates).copy()
    corners[:, 1] += 0.05 * np.asarray(reference_cell_topology(kind).vertices)[:, 1] ** 2
    curved_mesh = CellMesh(corners, mesh.blocks, vertex_global_ids=mesh.vertex_global_ids)
    geometry = CellGeometrySpec(
        {"volume": old_geometry.elements[0]},
        {"volume": old_geometry.geometry_dofs[0]},
        controls,
    )
    space = _space(curved_mesh, geometry, 1)
    prepared = _prepare_mapped_nested_dg_transfer(
        space,
        space,
        geometry,
        geometry,
        field_name="u",
        parent_cells=np.array([0]),
        parent_reference_vertices=np.asarray(reference_cell_topology(kind).vertices)[
            None
        ],
    )
    n = space.dof_maps[0].global_dof_count
    values = np.asarray(
        np.column_stack((np.arange(n) ** 2 + 0.4, np.arange(n) * 0.3 - 1)),
        dtype=np.float64,
    )
    independently_integrated = _content(space, geometry, values)
    np.testing.assert_allclose(
        np.asarray(prepared.source_measures) @ values,
        independently_integrated,
        atol=2e-12,
    )
    measures, errors, exact = _certified_cell_measures(curved_mesh, geometry)
    assert exact
    np.testing.assert_allclose(measures, [expected], atol=2e-15, rtol=0)
    assert np.all(errors > 0)


def _axis_plane_quad(axis: int, orientation: float) -> tuple[CellMesh, CellGeometrySpec]:
    element = coordinate_lagrange_element("quadrilateral", 2)
    active = [coordinate for coordinate in range(3) if coordinate != axis]

    def physical(reference: ArrayLike) -> NDArray[np.float64]:
        reference = np.asarray(reference, dtype=np.float64)
        points = np.empty((len(reference), 3))
        points[:, axis] = 2.0
        points[:, active[0]] = orientation * reference[:, 0]
        points[:, active[1]] = (
            reference[:, 1]
            + 0.1 * reference[:, 1] ** 2
            + 0.05 * reference[:, 0] * reference[:, 1]
        )
        return points

    corners = physical(
        np.asarray(reference_cell_topology("quadrilateral").vertices, dtype=np.float64)
    )
    mesh = CellMesh(
        corners,
        (
            CellBlock(
                "face",
                "quadrilateral",
                np.array([[0, 1, 2, 3]]),
                global_ids=np.array([17]),
            ),
        ),
    )
    controls = physical(element.reference_nodes)
    geometry = CellGeometrySpec(
        {"face": element}, {"face": np.arange(len(controls))[None]}, controls
    )
    return mesh, geometry


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("orientation", [-1.0, 1.0])
def test_embedded_axis_plane_quad_exact_area_nonconstant_content_and_transpose(
    axis: int,
    orientation: float,
) -> None:
    mesh, geometry = _axis_plane_quad(axis, orientation)
    measures, errors, exact = _certified_cell_measures(mesh, geometry)
    assert exact
    np.testing.assert_allclose(measures, [9 / 8], atol=2e-15, rtol=0)
    space = _space(mesh, geometry, 1)
    prepared = _prepare_mapped_nested_dg_transfer(
        space,
        space,
        geometry,
        geometry,
        field_name="u",
        parent_cells=np.array([0]),
        parent_reference_vertices=np.asarray(
            reference_cell_topology("quadrilateral").vertices
        )[None],
    )
    coefficients = np.asarray([[0.2, 1.0], [1.4, -0.8], [-0.9, 1.6], [2.3, 0.5]])
    dual = np.asarray([[1.0, -0.4], [-0.7, 0.8], [0.3, 2.1], [0.9, -0.2]])
    values = np.asarray(prepared.transfer.apply(coefficients))
    np.testing.assert_allclose(values, coefficients, atol=2e-12)
    np.testing.assert_allclose(
        np.asarray(prepared.source_measures) @ coefficients,
        _content(space, geometry, coefficients),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        np.vdot(values, dual),
        np.vdot(coefficients, prepared.transfer.pullback(dual)),
        atol=2e-12,
    )


def _parabolic_surface(kind: str) -> tuple[CellMesh, CellGeometrySpec]:
    element = coordinate_lagrange_element(kind, 2)
    reference = np.asarray(element.reference_nodes)
    controls = np.column_stack((reference, 0.5 * reference[:, 0] ** 2))
    corners = np.asarray(reference_cell_topology(kind).vertices)
    physical = np.column_stack((corners, 0.5 * corners[:, 0] ** 2))
    mesh = CellMesh(
        physical,
        (
            CellBlock(
                "surface", kind, np.arange(len(corners))[None], global_ids=np.array([17])
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"surface": element}, {"surface": np.arange(len(controls))[None]}, controls
    )
    return mesh, geometry


@pytest.mark.parametrize("kind", ["triangle", "quadrilateral"])
def test_embedded_sqrt_gram_area_encloses_known_parabolic_oracle(kind: str) -> None:
    import math

    mesh, geometry = _parabolic_surface(kind)
    before = np.asarray(geometry.coordinates).copy()
    values, errors, exact = _certified_cell_measures(
        mesh, geometry, absolute_tolerance=1e-10, relative_tolerance=0
    )
    expected = (math.sqrt(2) + math.asinh(1)) / 2
    if kind == "triangle":
        expected -= (2 * math.sqrt(2) - 1) / 3
    assert not exact
    assert abs(values[0] - expected) <= errors[0] + 2e-15
    assert 0 < errors[0] <= 1e-10
    np.testing.assert_array_equal(geometry.coordinates, before)


def test_embedded_measure_work_exhaustion_is_atomic() -> None:
    mesh, geometry = _parabolic_surface("quadrilateral")
    before = np.asarray(geometry.coordinates).copy()
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _certified_cell_measures(mesh, geometry, maximum_work=1, relative_tolerance=0)
    assert refusal.value.reason == "resource_limit"
    assert refusal.value.measured > refusal.value.limit == 1
    np.testing.assert_array_equal(geometry.coordinates, before)


def test_embedded_stale_root_refusal_is_atomic() -> None:
    mesh, geometry = _parabolic_surface("quadrilateral")
    before = np.asarray(geometry.coordinates).copy()
    root = CellGeometryRestrictionSource(
        "unrelated-source",
        "stale-topology",
        {"surface": np.array([17])},
        {
            "surface": np.asarray(mesh.vertex_global_ids)[
                np.asarray(mesh.blocks[0].vertices)
            ]
        },
    )
    rooted = CellGeometrySpec(
        {
            "surface": RestrictedCellGeometryElement(
                _require_tabulated_coordinate_element(geometry.elements[0]),
                "quadrilateral",
                np.eye(2),
                np.zeros(2),
            )
        },
        {"surface": geometry.geometry_dofs[0]},
        before,
        restriction_source=root,
    )
    with pytest.raises(ValueError):
        _certify_nested_geometry_pairs(
            mesh,
            geometry,
            mesh,
            rooted,
            (_NestedReferencePair(0, 0, False, np.eye(2), np.zeros(2)),),
        )
    np.testing.assert_array_equal(geometry.coordinates, before)


@pytest.mark.parametrize("surface_name", ["sphere", "torus"])
def test_authoritative_curved_surface_sources_have_enclosed_physical_area(
    surface_name: Literal["sphere", "torus"],
) -> None:
    from examples._native_surface_sources import sphere, torus
    from phydrax.discretization._cell_geometry_transfer import (
        reconstruct_parametric_surface_cell_geometry,
    )
    from phydrax.meshing._curving import _straight_geometry

    source = sphere() if surface_name == "sphere" else torus()
    domain = source.domain
    charts = np.asarray([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]])
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = CellMesh.from_triangles(
        domain.evaluate(np.zeros(4, dtype=np.int32), charts), cells
    )
    layout = _straight_geometry(mesh, 2)
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    reconstructed = reconstruct_parametric_surface_cell_geometry(
        mesh,
        layout,
        mesh,
        layout,
        domain,
        domain_id=domain.domain_id,
        cell_ids=ids,
        cell_patches=np.zeros(2, dtype=np.int32),
        cell_charts=charts[cells],
        cell_geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
        cell_occurrence_paths=(domain.source_occurrences[2][0],) * 2,
        maximum_fidelity=0.1,
    )
    before = np.asarray(reconstructed.geometry.coordinates).copy()
    values, errors, exact = _certified_cell_measures(
        mesh, reconstructed.geometry, absolute_tolerance=1e-10, relative_tolerance=0
    )
    space = _space(mesh, reconstructed.geometry, 0)
    independent = _content(space, reconstructed.geometry, np.ones((2, 2)))[0]
    assert not exact
    assert abs(np.sum(values) - independent) <= np.sum(errors) + 1e-12
    assert np.sum(errors) <= 1e-10
    corners = np.asarray(mesh.coordinates)[cells]
    chord_area = 0.5 * np.sum(
        np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
        )
    )
    assert abs(np.sum(values) - chord_area) > 1e-6
    np.testing.assert_array_equal(reconstructed.geometry.coordinates, before)


def test_regular_embedded_map_resolves_coarse_bernstein_sign_with_relative_budget() -> (
    None
):
    element = coordinate_lagrange_element("quadrilateral", 2)

    def physical(reference: ArrayLike) -> NDArray[np.float64]:
        reference = np.asarray(reference, dtype=np.float64)
        return np.column_stack(
            (
                reference[:, 0],
                0.1 * reference[:, 1],
                (reference[:, 0] - 0.5) * reference[:, 1],
            )
        )

    corners = np.asarray(reference_cell_topology("quadrilateral").vertices)
    mesh = CellMesh(
        physical(corners),
        (CellBlock("surface", "quadrilateral", np.array([[0, 1, 2, 3]])),),
    )
    geometry = CellGeometrySpec(
        {"surface": element},
        {"surface": np.arange(element.local_dof_count)[None]},
        physical(element.reference_nodes),
    )
    values, errors, exact = _certified_cell_measures(
        mesh, geometry, absolute_tolerance=0, relative_tolerance=1e-10
    )
    nodes, weights = np.polynomial.legendre.leggauss(64)
    v = (nodes + 1) / 2
    a_squared = 0.1**2 * (1 + v**2)
    # Analytically integrate sqrt(a(v)^2 + (u-.5)^2) over u first.
    inner = 0.5 * np.sqrt(a_squared + 0.25) + a_squared * np.arcsinh(
        0.5 / np.sqrt(a_squared)
    )
    expected = 0.5 * np.dot(weights, inner)
    assert not exact
    assert abs(values[0] - expected) <= errors[0] + 1e-13
    assert errors[0] <= 1e-10 * values[0]


def test_embedded_rank_deficient_source_does_not_use_positive_mesh_corner_area() -> None:
    mesh, geometry = _parabolic_surface("triangle")
    before = np.asarray(geometry.coordinates).copy()
    collapsed = before.copy()
    collapsed[:, 1:] = 0
    deficient = CellGeometrySpec(
        {"surface": geometry.elements[0]},
        {"surface": geometry.geometry_dofs[0]},
        collapsed,
    )
    with pytest.raises(ValueError, match="no positive Gram measure"):
        _certified_cell_measures(mesh, deficient)
    np.testing.assert_array_equal(geometry.coordinates, before)


@pytest.mark.parametrize("signed_shift", [None, -0.5])
def test_signed_weighted_gram_integral_preserves_cancellation_and_relative_error(
    signed_shift: float | None,
) -> None:
    import math
    from fractions import Fraction

    from phydrax.discretization._cell_geometry_transfer import (
        _certified_sqrt_polynomial_integral,
    )

    gram: Polynomial = {(0, 0): Fraction(1), (2, 0): Fraction(1)}
    weight: Polynomial = {(1, 0): Fraction(-1 if signed_shift is None else 1)}
    if signed_shift is not None:
        weight[(0, 0)] = Fraction(signed_shift)
    value, error = _certified_sqrt_polynomial_integral(
        gram, weight, "quadrilateral", absolute_tolerance=0, relative_tolerance=1e-10
    )
    first_moment = (2 * math.sqrt(2) - 1) / 3
    expected = (
        -first_moment
        if signed_shift is None
        else first_moment - (math.sqrt(2) + math.asinh(1)) / 4
    )
    assert abs(value - expected) <= error + 2e-15
    assert error <= 1e-10 * abs(value)


def test_constant_gram_signed_quadratic_triangle_weight_has_exact_zero_moment() -> None:
    from fractions import Fraction

    from phydrax.discretization._cell_geometry_transfer import (
        _certified_sqrt_polynomial_integral,
    )

    # Vertex P2 shape x(2x-1) integrates to zero, even on irrational-area planes.
    gram: Polynomial = {(0, 0): Fraction(2)}
    weight: Polynomial = {(2, 0): Fraction(2), (1, 0): Fraction(-1)}
    value, error = _certified_sqrt_polynomial_integral(
        gram, weight, "triangle", absolute_tolerance=0, relative_tolerance=1e-12
    )
    assert value == 0 and error == 0


def test_real_scalar_dg_local_functional_on_curved_hex_preserves_components_and_gradient() -> (
    None
):
    import jax.numpy as jnp

    from phydrax.equations import FiniteElementFunctional

    coordinate = coordinate_lagrange_element("hexahedron", 2)

    def physical(reference: ArrayLike) -> NDArray[np.float64]:
        reference = np.asarray(reference, dtype=np.float64)
        u, v, w = reference.T
        return np.column_stack(
            (
                u + 0.05 * u * v * w,
                v + 0.07 * u**2 * v**2,
                w + 0.03 * v**2 * w**2 + 0.02 * u * w,
            )
        )

    corners = physical(
        np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    )
    mesh = CellMesh(corners, (CellBlock("volume", "hexahedron", np.arange(8)[None]),))
    geometry = CellGeometrySpec(
        {"volume": coordinate},
        {"volume": np.arange(coordinate.local_dof_count)[None]},
        physical(coordinate.reference_nodes),
    )
    space = _space(mesh, geometry, 1)
    coefficients = np.column_stack(
        (np.arange(8) * 0.3 + 0.2, np.arange(8) ** 2 * 0.1 - 0.5)
    )
    functional = FiniteElementFunctional(
        "actual-curved-dg-energy",
        "u",
        lambda values, gradients, points, context: jnp.sum(values**2, axis=-1),
    )
    points, weights = _degree_aware_reference_rule("hexahedron", 8)
    basis, _ = space.elements[0][0].tabulate(points)
    _, gradients = coordinate.tabulate(points)
    jacobian = np.einsum(
        "mD,qmd->qDd", np.asarray(geometry.coordinates), np.asarray(gradients)
    )
    physical_weights = np.asarray(weights) * np.linalg.det(jacobian)
    basis = np.asarray(basis)
    values = basis @ coefficients
    expected = np.sum(physical_weights * np.sum(values**2, axis=-1))
    expected_gradient = 2 * basis.T @ (physical_weights[:, None] * values)
    np.testing.assert_allclose(
        functional.evaluate(space, coefficients), expected, atol=2e-11
    )
    derivative = jax.grad(lambda state: functional.evaluate(space, state))(
        jnp.asarray(coefficients)
    )
    np.testing.assert_allclose(derivative, expected_gradient, atol=2e-11)


def _actual_metric_action(
    coordinates: NDArray[np.float64],
    kind: str,
) -> tuple[FiniteElementGeometryActions, FiniteElementRuntimeData]:
    mesh = CellMesh(
        coordinates, (CellBlock("cells", kind, np.arange(len(coordinates))[None]),)
    )
    geometry = CellGeometrySpec.affine(mesh)
    elements, routes, values = geometry.resolve(mesh)
    dimension = mesh.topological_dimension
    coordinate = _require_tabulated_coordinate_element(elements[0])
    basis, gradients = coordinate.tabulate(np.full((1, dimension), 0.2))
    reference_volume = 1 / 6 if kind == "tetrahedron" else 0.5
    action = FiniteElementGeometryActions(
        geometry.geometry_layout_id,
        "cell",
        basis,
        gradients,
        routes[0],
        np.asarray([reference_volume]),
    )
    runtime = FiniteElementRuntimeData(
        mesh,
        values,
        numeric_version="metric-regression",
        geometry_layout_id=geometry.geometry_layout_id,
    )
    return action, runtime


def test_actual_thin_nonorthogonal_local_metric_uses_jacobian_rank_and_differentiates() -> (
    None
):
    import equinox as eqx
    import jax.numpy as jnp

    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1e-8, 0.0], [0.0, 0.0, 1.0]]
    )
    action, runtime = _actual_metric_action(coordinates, "tetrahedron")
    metric = action.realize(runtime)
    np.testing.assert_allclose(metric.physical_weights, [[1e-8 / 6]], rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        metric.physical_gradient(jnp.asarray([[[1.0, 1.0, 0.0]]])),
        [[[1.0, 0.0, 0.0]]],
        atol=1e-12,
    )

    def volume(nodes: Array) -> Array:
        moved = eqx.tree_at(lambda data: data.coordinates, runtime, nodes)
        return jnp.sum(action.realize(moved).physical_weights)

    direction = jnp.zeros_like(runtime.coordinates).at[2, 1].set(1.0)
    _, tangent = jax.jvp(volume, (runtime.coordinates,), (direction,))
    gradient = jax.grad(volume)(runtime.coordinates)
    np.testing.assert_allclose(tangent, 1 / 6, rtol=1e-12)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), 1 / 6, rtol=1e-12)
    np.testing.assert_array_equal(runtime.coordinates, coordinates)


def test_actual_embedded_local_metric_keeps_tangent_gradient_and_area() -> None:
    import jax.numpy as jnp

    coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    action, runtime = _actual_metric_action(coordinates, "triangle")
    metric = action.realize(runtime)
    np.testing.assert_allclose(metric.physical_weights, [[np.sqrt(2) / 2]], rtol=1e-12)
    np.testing.assert_allclose(
        metric.physical_gradient(jnp.asarray([[[1.0, 0.0]]])),
        [[[0.5, 0.0, 0.5]]],
        atol=1e-12,
    )


def _rational_pyramid_tets() -> tuple[
    CellMesh, CellGeometrySpec, CellMesh, CellGeometrySpec, NDArray[np.float64]
]:
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id

    element = coordinate_lagrange_element("pyramid", 1)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    controls = reference.copy()
    controls[2, 0] += 0.25
    source_mesh = CellMesh(
        controls,
        (
            CellBlock(
                "source",
                "pyramid",
                np.arange(5)[None],
                global_ids=np.asarray([17], dtype=np.int64),
            ),
        ),
    )
    source_geometry = CellGeometrySpec(
        {"source": element}, {"source": np.arange(5)[None]}, controls
    )
    rows = ((0, 1, 2, 4), (0, 2, 3, 4))
    corners = reference[np.asarray(rows, dtype=np.int64)]
    elements: dict[str, RestrictedCellGeometryElement] = {}
    for index in range(corners.shape[0]):
        values = np.array(corners[index], dtype=np.float64, copy=False)
        if values.ndim != 2:
            raise ValueError(
                "Rational child fixture requires a matrix of reference corners."
            )
        elements[f"child{index}"] = RestrictedCellGeometryElement(
            element, "tetrahedron", (values[1:] - values[0]).T, values[0]
        )
    blocks = tuple(
        CellBlock(
            name,
            "tetrahedron",
            np.asarray(row, dtype=np.int32)[None],
            global_ids=np.asarray([20 + index], dtype=np.int64),
        )
        for index, (name, row) in enumerate(zip(elements, rows, strict=True))
    )
    mesh = CellMesh(controls, blocks)
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        source_mesh.topology_id,
        {name: np.asarray([17], dtype=np.int64) for name in elements},
        {name: np.arange(5, dtype=np.int64)[None] for name in elements},
    )
    geometry = CellGeometrySpec(
        elements,
        {name: np.arange(5)[None] for name in elements},
        controls,
        restriction_source=record,
    )
    return source_mesh, source_geometry, mesh, geometry, corners


def test_rational_pyramid_tet_measures_integrate_the_expression_not_the_carrier() -> None:
    _, _, mesh, geometry, _ = _rational_pyramid_tets()
    values, errors, exact = _certified_cell_measures(mesh, geometry)
    # The root collapsed map has det J = 1 + v/4. The two reference
    # triangles v<=u and u<=v have v moments 1/6 and 1/3; height contributes 1/3.
    expected = np.asarray([13 / 72, 7 / 36], dtype=np.float64)
    assert np.all(np.abs(values - expected) <= errors)
    assert exact
    carrier_measures = np.asarray([1 / 6, 5 / 24], dtype=np.float64)
    assert np.all(np.abs(values - carrier_measures) > 1e-3)


def test_rational_pyramid_dg_content_and_component_adjoint_use_actual_children() -> None:
    source_mesh, source_geometry, mesh, geometry, corners = _rational_pyramid_tets()
    source, target = _space(source_mesh, source_geometry, 0), _space(mesh, geometry, 0)
    prepared = _prepare_mapped_nested_dg_transfer(
        source,
        target,
        source_geometry,
        geometry,
        field_name="u",
        parent_cells=np.asarray([0, 0], dtype=np.int64),
        parent_reference_vertices=corners,
    )
    coefficients = np.asarray([[2.7, -0.8]], dtype=np.float64)
    transferred = np.asarray(prepared.transfer.apply(coefficients))
    content = np.asarray([13 / 72, 7 / 36]) @ transferred
    np.testing.assert_allclose(content, (3 / 8) * coefficients[0], atol=2e-14)
    dual = np.asarray([[0.3, -0.2], [-0.7, 0.6]], dtype=np.float64)
    np.testing.assert_allclose(
        np.vdot(transferred, dual),
        np.vdot(coefficients, prepared.transfer.pullback(dual)),
        atol=2e-14,
    )
    assert prepared.evidence.passed


def test_rational_measure_one_under_work_budget_refuses_atomically() -> None:
    from phydrax.discretization._coordinate_enclosure import CoordinateEnclosureBudget

    _, _, mesh, geometry, _ = _rational_pyramid_tets()
    before = np.asarray(geometry.coordinates).copy()
    ledger = CoordinateEnclosureBudget(100_000_000, 100_000_000)
    with ledger.activate():
        _certified_cell_measures(mesh, geometry)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        _certified_cell_measures(mesh, geometry, maximum_work=ledger.work_units - 1)
    assert refusal.value.reason == "resource_limit"
    assert refusal.value.measured > refusal.value.limit == ledger.work_units - 1
    np.testing.assert_array_equal(geometry.coordinates, before)


def test_rational_wrong_carrier_refuses_without_changing_the_source() -> None:
    _, _, mesh, geometry, _ = _rational_pyramid_tets()
    before = np.asarray(geometry.coordinates).copy()
    moved = np.asarray(mesh.coordinates).copy()
    moved[2, 0] = np.nextafter(moved[2, 0], np.inf)
    with pytest.raises(ValueError, match="correctly rounded"):
        plc_mapped_carrier(
            mesh.with_coordinates(moved, numeric_version="wrong-carrier"), geometry
        )
    np.testing.assert_array_equal(geometry.coordinates, before)

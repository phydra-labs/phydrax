#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization.fem import (
    form_element,
    prepare_finite_element_field_reconstruction,
    prepare_finite_element_point_interpolation,
)
from phydrax.discretization.fem._generic import _evaluate_paired_field_basis


jax.config.update("jax_enable_x64", True)
D = phx.discretization


def _curved(
    kind: str, element: Any, *, components: tuple[int, ...] = (), restricted: bool = False
) -> Any:
    source = coordinate_lagrange_element(kind, 2)
    dimension = source.topological_dimension

    def physical(reference: Any) -> np.ndarray:
        points = np.asarray(reference).copy()
        points[:, -1] += 0.08 * points[:, 0] * (1.0 - points[:, 0])
        return points

    matrix = 0.5 * np.eye(dimension) if restricted else np.eye(dimension)
    offset = np.full(dimension, 0.05) if restricted else np.zeros(dimension)
    coordinate = (
        RestrictedCellGeometryElement(source, kind, matrix, offset)
        if restricted
        else source
    )
    corners = np.asarray(coordinate_lagrange_element(kind, 1).reference_nodes)
    local_corners = physical(corners @ matrix.T + offset)
    # Positive local chart, deliberately noncanonical global tetrahedron order:
    # Nedelec degree one uses dense, nonmonomial face transformations here.
    order = np.asarray((1, 0, 3, 2)) if kind == "tetrahedron" else np.arange(len(corners))
    vertices = np.empty_like(local_corners)
    vertices[order] = local_corners
    mesh = D.CellMesh(vertices, (D.CellBlock("cells", kind, order[None]),))
    controls = physical(source.reference_nodes)
    discretization = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec("u", element, component_shape=components),
        coordinate_spec=D.CellGeometrySpec(
            {"cells": coordinate},
            {"cells": np.arange(len(controls), dtype=np.int32)[None]},
            controls,
        ),
    ).prepare()
    return discretization, lambda reference: physical(
        np.asarray(reference) @ matrix.T + offset
    )


def _owner_values(discretization: Any, reference: Any, coefficients: Any) -> np.ndarray:
    geometry = discretization.evaluate_block_geometry(
        "u",
        0,
        discretization.default_runtime.coordinates,
        jnp.asarray(reference),
        jnp.ones((len(reference),)),
    )
    dof_map = discretization.dof_maps[0]
    dofs = np.asarray(dof_map.cell_dofs[0])[0]
    basis = np.asarray(geometry.basis_values)
    if basis.ndim == 2:
        basis = np.broadcast_to(basis, (1, *basis.shape))
    local = np.einsum(
        "ai,ik->ak",
        np.asarray(dof_map.cell_transforms[0])[0],
        np.asarray(coefficients)[dofs].reshape((len(dofs), -1)),
    )
    return np.einsum(
        "pav,ak->pvk", basis[0].reshape((len(reference), len(dofs), -1)), local
    ).reshape(
        (
            len(reference),
            *discretization.elements[0][0].value_shape,
            *np.asarray(coefficients).shape[1:],
        )
    )


@pytest.mark.parametrize("kind", ("triangle", "tetrahedron", "prism", "hexahedron"))
@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("conformity", ("H1", "L2"))
def test_curved_scalar_polynomial_components_and_transpose(
    kind: str, degree: int, conformity: str
) -> None:
    element = (
        D.lagrange_element(kind, degree)
        if conformity == "H1"
        else D.discontinuous_element(kind, degree)
    )
    discretization, physical = _curved(kind, element, components=(2,))
    dimension = element.topological_dimension
    reference = np.asarray(((0.13, 0.17, 0.19), (0.21, 0.16, 0.12)))[:, :dimension]

    def polynomial(points: Any) -> np.ndarray:
        points = np.asarray(points)
        scalar = 1.0 + 2.0 * points[:, 0] - 0.7 * points[:, -1]
        if degree == 2:
            scalar += points[:, 0] ** 2 + points[:, 0] * points[:, -1]
        return np.stack((scalar, -0.5 * scalar + 0.3), axis=-1)

    coefficients = np.zeros(discretization.field_spaces[0].vector_space.shape)
    coefficients[np.asarray(discretization.dof_maps[0].cell_dofs[0])[0]] = polynomial(
        element.reference_nodes
    )
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    query = reconstruction.prepare_query(physical(reference))
    np.testing.assert_allclose(
        query.apply(coefficients), polynomial(reference), atol=2e-10
    )
    np.testing.assert_allclose(
        query.apply(coefficients),
        _owner_values(discretization, reference, coefficients),
        atol=2e-10,
    )
    dual = jnp.asarray(((0.7, -0.4), (1.3, 0.2)))
    np.testing.assert_allclose(
        jnp.vdot(query.apply(coefficients), dual),
        jnp.vdot(coefficients, query.transpose(dual)),
        atol=2e-12,
    )
    _, pullback = jax.vjp(query.apply, jnp.asarray(coefficients))
    np.testing.assert_allclose(query.transpose(dual), pullback(dual)[0], atol=2e-12)
    cells = np.zeros(len(reference), dtype=np.int32)
    fixed = prepare_finite_element_point_interpolation(
        discretization, "u", "cells", cells, reference
    )
    np.testing.assert_allclose(
        fixed.interpolate(coefficients), polynomial(reference), atol=2e-12
    )
    np.testing.assert_allclose(
        fixed.transpose_scatter(dual), query.transpose(dual), atol=2e-10
    )
    # x_d = xi_d + 0.08 xi_0 (1-xi_0): use the known exact inverse chain,
    # not the query's own derivative weights, as the physical-gradient oracle.
    inverse = np.broadcast_to(
        np.eye(dimension), (len(reference), dimension, dimension)
    ).copy()
    inverse[:, -1, 0] = -0.08 * (1.0 - 2.0 * reference[:, 0])
    reference_gradient = np.zeros_like(reference)
    reference_gradient[:, 0] = 2.0
    reference_gradient[:, -1] -= 0.7
    if degree == 2:
        reference_gradient[:, 0] += 2.0 * reference[:, 0] + reference[:, -1]
        reference_gradient[:, -1] += reference[:, 0]
    physical_gradient = np.einsum("pr,pra->pa", reference_gradient, inverse)
    for axis in range(dimension):
        derivative = tuple(int(index == axis) for index in range(dimension))
        expected = np.stack(
            (physical_gradient[:, axis], -0.5 * physical_gradient[:, axis]), axis=-1
        )
        derivative_query = reconstruction.prepare_query(
            physical(reference), derivative=derivative
        )
        fixed_derivative = prepare_finite_element_point_interpolation(
            discretization,
            "u",
            "cells",
            cells,
            reference,
            derivative_axis=axis,
        )
        np.testing.assert_allclose(
            derivative_query.apply(coefficients), expected, atol=3e-10
        )
        np.testing.assert_allclose(
            fixed_derivative.interpolate(coefficients), expected, atol=2e-11
        )
        np.testing.assert_allclose(
            jnp.vdot(derivative_query.apply(coefficients), dual),
            jnp.vdot(coefficients, derivative_query.transpose(dual)),
            atol=2e-10,
        )


@pytest.mark.parametrize(
    "element",
    (
        form_element("tetrahedron", 1, 1, proxy="circulation"),
        form_element("tetrahedron", 1, 2, proxy="circulation"),
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"),
    ),
)
@pytest.mark.parametrize("restricted", (False, True))
def test_curved_piola_and_restricted_child_values_derivatives_and_adjoint(
    element: Any, restricted: bool
) -> None:
    discretization, physical = _curved("tetrahedron", element, restricted=restricted)
    # The final point is in the curved bulge beyond the affine corner hull.
    reference = np.asarray(((0.13, 0.17, 0.19), (0.21, 0.16, 0.12), (0.3, 0.3, 0.398)))
    cells = np.zeros(len(reference), dtype=np.int32)
    dof_map = discretization.dof_maps[0]
    coefficients = jnp.asarray(
        np.random.default_rng(31).normal(size=dof_map.global_dof_count)
    )
    fixed = prepare_finite_element_point_interpolation(
        discretization, "u", "cells", cells, reference
    )
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    points = physical(reference)
    query = reconstruction.prepare_query(points)
    expected = _owner_values(discretization, reference, coefficients)
    np.testing.assert_allclose(fixed.interpolate(coefficients), expected, atol=2e-11)
    np.testing.assert_allclose(query.apply(coefficients), expected, atol=2e-9)
    dual = jnp.asarray(((0.7, -0.4, 0.3), (1.3, 0.2, -0.6), (-0.2, 0.8, 0.5)))
    np.testing.assert_allclose(
        jnp.vdot(query.apply(coefficients), dual),
        jnp.vdot(coefficients, query.transpose(dual)),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        fixed.transpose_scatter(dual), query.transpose(dual), atol=2e-9
    )
    _, pullback = jax.vjp(query.apply, coefficients)
    np.testing.assert_allclose(query.transpose(dual), pullback(dual)[0], atol=2e-12)
    local = coefficients[dof_map.cell_dofs[0][0]]
    for axis in range(3):
        derivative = tuple(int(index == axis) for index in range(3))
        basis, valid = _evaluate_paired_field_basis(
            discretization,
            "u",
            0,
            cells,
            reference,
            discretization.default_runtime.coordinates,
            derivative_axis=axis,
        )
        assert bool(jnp.all(valid))
        expected_derivative = jnp.einsum("pnv,n->pv", basis, local)
        derivative_query = reconstruction.prepare_query(points, derivative=derivative)
        np.testing.assert_allclose(
            derivative_query.apply(coefficients), expected_derivative, atol=3e-8
        )
        np.testing.assert_allclose(
            jnp.vdot(derivative_query.apply(coefficients), dual),
            jnp.vdot(coefficients, derivative_query.transpose(dual)),
            atol=2e-10,
        )


@pytest.mark.parametrize(
    "element",
    (
        form_element("triangle", 1, 1, proxy="circulation"),
        form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
    ),
)
def test_compatible_shared_facet_requires_side_and_masked_query_retains_status(
    element: Any,
) -> None:
    mesh = D.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
    )
    discretization = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", element)
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(discretization, "u")
    coefficients = jnp.asarray(
        np.random.default_rng(11).normal(size=discretization.dof_maps[0].global_dof_count)
    )
    points = jnp.asarray(((0.3, 0.3), (1.5, 0.5), (0.7, 0.2)))
    masked = reconstruction.prepare_query(points, coverage="masked")
    assert masked.evidence.status.tolist() == [
        int(D.FieldQueryStatus.SIDE_REQUIRED),
        int(D.FieldQueryStatus.OUTSIDE_SUPPORT),
        int(D.FieldQueryStatus.VALID),
    ]
    np.testing.assert_array_equal(masked.admitted, (2,))
    np.testing.assert_allclose(
        masked.apply(coefficients),
        reconstruction.evaluate(coefficients, points[2:]).values,
        atol=2e-12,
    )
    dual = jnp.asarray(((-0.1, 0.6),))
    np.testing.assert_allclose(
        jnp.vdot(masked.apply(coefficients), dual),
        jnp.vdot(coefficients, masked.transpose(dual)),
        atol=2e-12,
    )
    facet = points[:1]
    owner = reconstruction.prepare_query(facet, side="owner", cell_ids=np.asarray((0,)))
    neighbor = reconstruction.prepare_query(
        facet, side="neighbor", cell_ids=np.asarray((1,))
    )
    average = reconstruction.prepare_query(facet, side="average")
    np.testing.assert_allclose(
        average.apply(coefficients),
        0.5 * (owner.apply(coefficients) + neighbor.apply(coefficients)),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        average.transpose(dual[:1]),
        0.5 * (owner.transpose(dual[:1]) + neighbor.transpose(dual[:1])),
        atol=2e-12,
    )
    # A regular exterior facet belongs to one cell and needs no arbitrary side.
    boundary = reconstruction.prepare_query(jnp.asarray(((0.5, 0.0),)))
    assert bool(jnp.all(boundary.evidence.valid))
    with pytest.raises(ValueError):
        reconstruction.prepare_query(points[:2], coverage="masked")


def test_mixed_block_view_keeps_global_coefficients_and_other_block_identity() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 2.0),
        )
    )
    mesh = D.CellMesh(
        vertices,
        (
            D.CellBlock(
                "prisms",
                "prism",
                np.asarray(((0, 1, 2, 3, 4, 5),)),
                global_ids=np.asarray((101,)),
            ),
            D.CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray(((6, 3, 5, 4),)),
                global_ids=np.asarray((303,)),
            ),
        ),
    )
    field = D.FiniteElementFieldSpec(
        "u",
        {
            "prisms": D.lagrange_element("prism", 1),
            "tetrahedra": D.lagrange_element("tetrahedron", 1),
        },
    )
    discretization = D.FiniteElementPlan(mesh, field).prepare()
    coefficients = jnp.asarray(
        1.0 + vertices[:, 0] - 2.0 * vertices[:, 1] + 0.3 * vertices[:, 2]
    )
    points = jnp.asarray(((0.2, 0.3, 0.4),))
    prism = prepare_finite_element_field_reconstruction(
        discretization, "u", block_name="prisms"
    )
    tetrahedron = prepare_finite_element_field_reconstruction(
        discretization, "u", block_name="tetrahedra"
    )
    query = prism.prepare_query(points)
    np.testing.assert_allclose(query.apply(coefficients), (0.72,), atol=2e-12)
    assert query.transpose(jnp.asarray((1.0,))).shape == coefficients.shape
    np.testing.assert_allclose(query.transpose(jnp.asarray((1.0,)))[6], 0.0)
    assert tetrahedron.validity(points).status.tolist() == [
        int(D.FieldQueryStatus.OUTSIDE_SUPPORT)
    ]
    interpolation = prepare_finite_element_point_interpolation(
        discretization,
        "u",
        "prisms",
        np.asarray((0,), dtype=np.int32),
        np.asarray(((0.2, 0.3, 0.4),)),
    )
    # Preparing only the prism still describes global DOFs, including the
    # tetrahedron-only vertex and the correctly averaged shared face nodes.
    np.testing.assert_allclose(
        interpolation.dof_reference_positions, vertices, atol=2e-12
    )


def test_candidate_capacity_cannot_certify_a_partial_compatible_trace() -> None:
    mesh = D.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
    )
    discretization = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "u", form_element("triangle", 1, 1, proxy="circulation")
        ),
    ).prepare()
    bounded = prepare_finite_element_field_reconstruction(
        discretization,
        "u",
        location_policy=D.SimplicialLocationPolicy(1, 16, 1),
    )
    facet = jnp.asarray(((0.3, 0.3),))
    assert bounded.validity(facet).status.tolist() == [
        int(D.FieldQueryStatus.LOCATION_FAILED)
    ]
    with pytest.raises(ValueError):
        bounded.prepare_query(facet)
    with pytest.raises(ValueError):
        bounded.prepare_query(facet, side="average")


@pytest.mark.parametrize("kind", ("tetrahedron", "hexahedron"))
@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("restricted", (False, True))
def test_actual_mapped_nodal_positions_drive_displacement(
    kind: str, degree: int, restricted: bool
) -> None:
    element = D.lagrange_element(kind, degree)
    discretization, physical = _curved(
        kind, element, components=(3,), restricted=restricted
    )
    interpolation = prepare_finite_element_point_interpolation(
        discretization,
        "u",
        "cells",
        np.asarray((0,), dtype=np.int32),
        np.asarray(((0.13, 0.17, 0.19),)),
    )
    routes = np.asarray(discretization.dof_maps[0].cell_dofs[0])[0]
    expected_nodes = physical(element.reference_nodes)
    np.testing.assert_allclose(
        interpolation.dof_reference_positions[routes],
        expected_nodes,
        atol=2e-12,
    )
    displacement = np.zeros(discretization.field_spaces[0].vector_space.shape)
    displacement[routes] = 0.1 * expected_nodes**2 + np.asarray((0.03, -0.02, 0.01))
    np.testing.assert_allclose(
        interpolation.deformed_dof_positions(displacement)[routes],
        expected_nodes + displacement[routes],
        atol=2e-12,
    )


def test_compatible_moments_are_not_displacement_nodes() -> None:
    discretization, _ = _curved(
        "tetrahedron", form_element("tetrahedron", 1, 2, proxy="circulation")
    )
    interpolation = prepare_finite_element_point_interpolation(
        discretization,
        "u",
        "cells",
        np.asarray((0,), dtype=np.int32),
        np.asarray(((0.13, 0.17, 0.19),)),
    )
    with pytest.raises(ValueError):
        interpolation.deformed_dof_positions(
            np.zeros(discretization.dof_maps[0].global_dof_count)
        )

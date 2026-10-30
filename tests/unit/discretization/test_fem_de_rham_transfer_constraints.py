from fractions import Fraction
from math import prod
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.discretization.fem._form_elements import FormBasis
from phydrax.linalg import ArraySpace


def _two_triangle_square() -> CellMesh:
    return CellMesh(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "cells",
                "triangle",
                np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
                global_ids=np.asarray((10, 11), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(4, dtype=np.int64),
    )


@pytest.mark.parametrize(
    (
        "dimension",
        "degree",
        "order",
        "family",
        "twist",
        "permutation",
        "global_vertices",
    ),
    (
        (2, 1, 2, "trimmed", "untwisted", (1, 0, 2), (41, 13, 29)),
        (3, 2, 2, "trimmed", "untwisted", (1, 0, 2, 3), (41, 13, 29, 71)),
        (2, 0, 2, "full", "twisted", (1, 0, 2), (41, 13, 29)),
        (
            2,
            1,
            2,
            "tensor-trimmed",
            "untwisted",
            (1, 0, 3, 2),
            (41, 13, 29, 71),
        ),
    ),
)
def test_exact_form_basis_banks_are_dual_and_project_consistently(
    dimension: int,
    degree: int,
    order: int,
    family: Literal["trimmed", "full", "tensor-trimmed"],
    twist: Literal["untwisted", "twisted"],
    permutation: tuple[int, ...],
    global_vertices: tuple[int, ...],
) -> None:
    basis = FormBasis(dimension, degree, order, family, twist)
    assert len(basis.exact_generators) == len(basis.exponents)
    assert len(basis.exact_generator_labels) == basis.local_dof_count
    coefficient_values = np.asarray(basis.coefficients, dtype=np.float64)
    np.testing.assert_array_equal(
        coefficient_values,
        np.asarray(basis.exact_coefficients, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(basis.functional_moments, dtype=np.float64),
        np.asarray(basis.exact_functional_moments, dtype=np.float64),
    )
    target_order = order - int(family == "full")
    target_basis = FormBasis(dimension, degree + 1, target_order, family, twist)
    exact_derivative = basis.exact_exterior_derivative_matrix(target_basis)
    np.testing.assert_allclose(
        np.asarray(exact_derivative, dtype=np.float64),
        np.asarray(basis.exterior_derivative_matrix(target_basis)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    exact_point = tuple(Fraction(1, dimension + 2) for _ in range(dimension))
    exact_values = np.asarray(
        [
            [
                sum(
                    (
                        basis.exact_coefficients[monomial][component][dof]
                        * prod(
                            exact_point[axis] ** exponent[axis]
                            for axis in range(dimension)
                        )
                        for monomial, exponent in enumerate(basis.exponents)
                    ),
                    Fraction(0),
                )
                for dof in range(basis.local_dof_count)
            ]
            for component in range(len(basis.exact_coefficients[0]))
        ],
        dtype=np.float64,
    ).T
    runtime_values, _ = basis.tabulate_components(
        np.asarray([[float(value) for value in exact_point]], dtype=np.float64)
    )
    np.testing.assert_allclose(
        np.asarray(runtime_values[0]),
        exact_values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    functional_values, _ = basis.tabulate_components(basis.functional_points)
    reconstructed = basis.interpolate(np.moveaxis(np.asarray(functional_values), 1, -1))
    np.testing.assert_allclose(
        np.asarray(reconstructed),
        np.eye(basis.local_dof_count),
        rtol=0.0,
        atol=5.0e-12,
    )
    execution_moments = np.asarray(basis.functional_moments, dtype=np.float64)
    for dof, monomials in enumerate(basis.exact_functional_moments):
        for monomial, components in enumerate(monomials):
            for component, exact_value in enumerate(components):
                execution_value = float(execution_moments[dof, monomial, component])
                assert basis.functional_moment_projection_errors[dof][monomial][
                    component
                ] == abs(Fraction(execution_value) - exact_value)
    for monomial, components in enumerate(basis.exact_coefficients):
        for component, dofs in enumerate(components):
            for dof, exact_value in enumerate(dofs):
                execution_value = float(coefficient_values[monomial, component, dof])
                assert basis.coefficient_projection_errors[monomial][component][
                    dof
                ] == abs(Fraction(execution_value) - exact_value)
    for row in range(basis.local_dof_count):
        for column in range(basis.local_dof_count):
            value = sum(
                (
                    basis.exact_functional_moments[row][monomial][component]
                    * basis.exact_coefficients[monomial][component][column]
                    for monomial in range(len(basis.exponents))
                    for component in range(len(basis.exact_coefficients[0]))
                ),
                Fraction(0),
            )
            assert value == Fraction(int(row == column))
    exact_matrix = basis.exact_permutation_matrix(permutation)
    exact_permutation = np.asarray(exact_matrix, dtype=np.float64)
    np.testing.assert_allclose(
        np.asarray(basis.permutation_matrix(permutation)),
        exact_permutation,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    for row in range(basis.local_dof_count):
        for column in range(basis.local_dof_count):
            composed = sum(
                (
                    exact_matrix[row][middle] * exact_matrix[middle][column]
                    for middle in range(basis.local_dof_count)
                ),
                Fraction(0),
            )
            assert composed == Fraction(int(row == column))
    if twist == "twisted":
        untwisted = FormBasis(dimension, degree, order, family, "untwisted")
        assert exact_matrix == tuple(
            tuple(-value for value in row)
            for row in untwisted.exact_permutation_matrix(permutation)
        )
    exact_entity_permutation = np.asarray(
        basis.exact_entity_permutation_matrix(global_vertices), dtype=np.float64
    )
    np.testing.assert_allclose(
        np.asarray(basis.entity_permutation_matrix(global_vertices)),
        exact_entity_permutation,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_relative_piola_transfer_preserves_constraints_and_commutes() -> None:
    mesh = _two_triangle_square()
    source = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    target = FiniteElementDeRhamComplex(mesh, family="trimmed", order=3)
    transfer_result = source.transfer(target, boundary="relative")
    transfer = transfer_result.complex_map

    assert (
        transfer.source.complex_id
        == source.hilbert_complex(boundary="relative").complex_id
    )
    assert (
        transfer.target.complex_id
        == target.hilbert_complex(boundary="relative").complex_id
    )
    assert transfer_result.exact_integral_count > 0
    assert transfer_result.enclosed_integral_count == 0
    assert transfer_result.commuting_evidence
    generator = np.random.default_rng(41)
    for degree in range(source.dimension + 1):
        source_space = transfer.source.space(degree)
        assert isinstance(source_space, ArraySpace)
        values = jnp.asarray(
            generator.normal(size=source_space.size), dtype=source_space.dtype
        )
        mapped = transfer.map(degree).mv(values)
        target_space = transfer.target.space(degree)
        assert isinstance(target_space, ArraySpace)
        target_values = jnp.asarray(
            generator.normal(size=target_space.size), dtype=target_space.dtype
        )
        np.testing.assert_allclose(
            target_space.inner(mapped, target_values),
            source_space.inner(
                values,
                transfer.adjoint().map(degree).mv(target_values),
            ),
            rtol=2.0e-9,
            atol=2.0e-9,
        )
        full = target._extend(degree, mapped, "relative")
        assert np.all(np.asarray(full)[np.asarray(target.boundary_masks[degree])] == 0.0)
        if degree < source.dimension:
            left = transfer.target.differential(degree).mv(mapped)
            right = transfer.map(degree + 1).mv(
                transfer.source.differential(degree).mv(values)
            )
            np.testing.assert_allclose(left, right, rtol=2.0e-10, atol=2.0e-10)


def test_relative_transfer_refuses_a_target_boundary_that_cuts_an_interior_trace() -> (
    None
):
    source_mesh = _two_triangle_square()
    source_block = source_mesh.blocks[0]
    target_vertices = np.unique(np.asarray(source_block.vertices[:1], dtype=np.int64))
    local = np.full(source_mesh.coordinates.shape[0], -1, dtype=np.int32)
    local[target_vertices] = np.arange(target_vertices.size, dtype=np.int32)
    target_mesh = CellMesh(
        source_mesh.coordinates[target_vertices],
        (
            CellBlock(
                "cells",
                "triangle",
                local[np.asarray(source_block.vertices[:1], dtype=np.int64)],
                global_ids=np.asarray(source_block.global_ids[:1], dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray(source_mesh.vertex_global_ids)[target_vertices],
    )
    source = FiniteElementDeRhamComplex(source_mesh, family="trimmed", order=2)
    target = FiniteElementDeRhamComplex(target_mesh, family="trimmed", order=2)

    with pytest.raises(ValueError, match="relative boundary"):
        source.transfer(
            target,
            boundary="relative",
            parent_cells=np.asarray((0,), dtype=np.int32),
        )


def test_fe_transfer_refuses_an_unbound_coordinate_frame_change() -> None:
    from phydrax._coordinate_measure import CoordinateMeasurePlan

    mesh = _two_triangle_square()
    source = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    target = FiniteElementDeRhamComplex(
        mesh,
        family="trimmed",
        order=1,
        coordinate_measure=CoordinateMeasurePlan("cartesian", frame="current"),
    )

    with pytest.raises(ValueError, match="same coordinate frame and measure"):
        source.transfer(target)

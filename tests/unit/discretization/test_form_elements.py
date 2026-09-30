"""Independent dimensions, duality, and polynomial differential contracts."""

from itertools import combinations
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest

from phydrax.discretization.fem._form_elements import (
    form_element,
    FormElementFamily,
)
from phydrax.exterior._algebra import map_reference_values
from phydrax.exterior._form_type import FormTwist, FormType, FormValueSpec


@pytest.mark.parametrize("dimension", range(1, 5))
@pytest.mark.parametrize("order", range(1, 4))
@pytest.mark.parametrize("family", ("trimmed", "full", "tensor-trimmed"))
def test_polynomial_form_dimensions_and_functional_duality(
    dimension: int, order: int, family: FormElementFamily
) -> None:
    cell = f"tensor:{dimension}" if family == "tensor-trimmed" else f"simplex:{dimension}"
    for degree in range(dimension + 1):
        element = form_element(
            cell, degree, order, family=family, twist="untwisted", proxy="components"
        )
        if family == "full":
            expected = comb(order + dimension, order + degree) * comb(
                order + degree, degree
            )
        elif family == "trimmed":
            expected = comb(order + dimension, order + degree) * comb(
                order + degree - 1, degree
            )
        else:
            expected = (
                comb(dimension, degree)
                * order**degree
                * (order + 1) ** (dimension - degree)
            )
        assert element.local_dof_count == expected
        basis = element.form_basis
        if basis is None:
            raise AssertionError("A form element must own its polynomial basis.")
        values, _ = basis.tabulate_components(basis.functional_points)
        functionals = np.einsum(
            "iqc,qjc->ij", np.asarray(basis.functional_weights), np.asarray(values)
        )
        np.testing.assert_allclose(functionals, np.eye(expected), atol=2e-9, rtol=2e-9)
        assert np.linalg.cond(functionals) < 1.000001


@pytest.mark.parametrize("dimension", range(1, 5))
@pytest.mark.parametrize(
    ("family", "order"),
    (("trimmed", 2), ("full", 2), ("tensor-trimmed", 2), ("tensor-trimmed", 3)),
)
def test_polynomial_differential_is_exact_and_nilpotent(
    dimension: int, family: FormElementFamily, order: int
) -> None:
    cell = f"tensor:{dimension}" if family == "tensor-trimmed" else f"simplex:{dimension}"
    previous: npt.NDArray[np.float64] | None = None
    previous_rank = 0
    for degree in range(dimension + 1):
        degree_order = order + dimension - degree if family == "full" else order
        source = form_element(
            cell,
            degree,
            degree_order,
            family=family,
            twist="untwisted",
            proxy="components",
        ).form_basis
        if source is None:
            raise AssertionError("A form element must own its polynomial basis.")
        if degree == dimension:
            assert previous_rank == source.coefficients.shape[-1]
            continue
        target_order = degree_order - 1 if family == "full" else order
        target = form_element(
            cell,
            degree + 1,
            target_order,
            family=family,
            twist="untwisted",
            proxy="components",
        ).form_basis
        if target is None:
            raise AssertionError("A form element must own its polynomial basis.")
        differential = np.asarray(source.exterior_derivative_matrix(target))
        if previous is not None:
            np.testing.assert_allclose(differential @ previous, 0.0, atol=2e-8)
        rank = np.linalg.matrix_rank(differential, tol=1e-8)
        kernel_dimension = source.coefficients.shape[-1] - rank
        assert kernel_dimension == (1 if degree == 0 else previous_rank)
        previous = differential
        previous_rank = rank


@pytest.mark.parametrize("dimension", (1, 2, 3, 4))
@pytest.mark.parametrize("twist", ("untwisted", "twisted"))
def test_top_density_reflection_uses_signed_or_absolute_volume(
    dimension: int, twist: FormTwist
) -> None:
    form_type = FormType(dimension, dimension, twist=twist)
    value_spec = FormValueSpec(form_type, proxy="density")
    jacobian = np.diag(np.asarray((-2.0,) + (3.0,) * (dimension - 1), dtype=np.float64))
    density = np.asarray((1.25, -0.75), dtype=np.float64)
    mapped = map_reference_values(density, value_spec, jacobian)
    determinant = np.linalg.det(jacobian)
    divisor = determinant if twist == "untwisted" else abs(determinant)
    np.testing.assert_allclose(mapped, density / divisor, atol=1e-14)


def test_form_element_requires_proxy_and_physical_twist_when_ambiguous() -> None:
    with pytest.raises(ValueError):
        form_element("triangle", 1, 1, twist="untwisted")
    with pytest.raises(ValueError):
        form_element("tetrahedron", 2, 1, proxy="flux")
    with pytest.raises(ValueError):
        form_element("tetrahedron", 3, 1, proxy="density")


@pytest.mark.parametrize("dimension", (2, 3, 4))
@pytest.mark.parametrize("family", ("trimmed", "full"))
@pytest.mark.parametrize("twist", ("untwisted", "twisted"))
def test_face_permutation_covariance_matches_affine_pullback(
    dimension: int, family: FormElementFamily, twist: FormTwist
) -> None:
    vertices = np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), np.eye(dimension)), axis=0
    )
    permutation = (1, 0, *range(2, dimension + 1))
    permuted = vertices[np.asarray(permutation, dtype=np.int32)]
    jacobian = (permuted[1:] - permuted[0]).T
    for degree in range(dimension + 1):
        element = form_element(
            f"simplex:{dimension}",
            degree,
            2,
            family=family,
            twist=twist,
            proxy="components",
        )
        basis = element.form_basis
        if basis is None:
            raise AssertionError("A form element must own its polynomial basis.")
        points = np.asarray(basis.functional_points)
        mapped_points = permuted[0] + points @ jacobian.T
        values, _ = basis.tabulate_components(mapped_points)
        subsets = tuple(combinations(range(dimension), degree))
        compound = np.asarray(
            [
                [np.linalg.det(jacobian.T[np.ix_(row, column)]) for column in subsets]
                for row in subsets
            ],
            dtype=np.float64,
        )
        if twist == "twisted":
            compound *= np.sign(np.linalg.det(jacobian))
        pullback = np.einsum("ab,qjb->qja", compound, np.asarray(values))
        expected = np.einsum(
            "iqc,qjc->ij", np.asarray(basis.functional_weights), pullback
        )
        action = np.asarray(basis.permutation_matrix(permutation))
        np.testing.assert_allclose(action, expected, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(action @ action, np.eye(action.shape[0]), atol=2e-9)


@pytest.mark.parametrize("dimension", (3, 4))
@pytest.mark.parametrize("degree", (0, 1, 2))
def test_cubic_tensor_interpolation_commutes_with_polynomial_differentiation(
    dimension: int, degree: int
) -> None:
    source = form_element(
        f"tensor:{dimension}",
        degree,
        3,
        family="tensor-trimmed",
        twist="untwisted",
        proxy="components",
    ).form_basis
    target = form_element(
        f"tensor:{dimension}",
        degree + 1,
        3,
        family="tensor-trimmed",
        twist="untwisted",
        proxy="components",
    ).form_basis
    if source is None or target is None:
        raise AssertionError("Form elements must own their polynomial bases.")
    blade = tuple(range(degree))
    powers = np.asarray(
        tuple(2 if axis in blade else 3 for axis in range(dimension)), dtype=np.int64
    )
    samples = np.zeros(
        (source.functional_points.shape[0], comb(dimension, degree)), dtype=np.float64
    )
    samples[:, 0] = np.prod(
        np.asarray(source.functional_points) ** powers[None, :], axis=-1
    )
    coefficients = source.interpolate(samples)
    probes = jnp.asarray(
        np.stack(
            (
                np.linspace(0.0, 1.0, dimension),
                np.linspace(0.2, 0.8, dimension),
                np.linspace(1.0, 0.0, dimension),
            )
        ),
        dtype=jnp.float64,
    )
    values, gradient = jax.jit(lambda basis, points: basis.tabulate_components(points))(
        source, probes
    )
    expected_values = np.zeros((3, comb(dimension, degree)), dtype=np.float64)
    expected_values[:, 0] = np.prod(np.asarray(probes) ** powers[None, :], axis=-1)
    np.testing.assert_allclose(
        np.einsum("qbc,b->qc", np.asarray(values), np.asarray(coefficients)),
        expected_values,
        atol=2e-10,
        rtol=2e-10,
    )
    expected_derivative = np.zeros((3, comb(dimension, degree + 1)), dtype=np.float64)
    target_blades = tuple(combinations(range(dimension), degree + 1))
    for axis in range(degree, dimension):
        lower = powers.copy()
        lower[axis] -= 1
        component = target_blades.index((*blade, axis))
        expected_derivative[:, component] = (
            (-1) ** degree
            * powers[axis]
            * np.prod(np.asarray(probes) ** lower[None, :], axis=-1)
        )
    target_values, _ = target.tabulate_components(probes)
    differentiated = source.exterior_derivative_matrix(target) @ coefficients
    np.testing.assert_allclose(
        np.einsum("qbc,b->qc", np.asarray(target_values), np.asarray(differentiated)),
        expected_derivative,
        atol=2e-10,
        rtol=2e-10,
    )
    expected_gradient = np.zeros(
        (3, comb(dimension, degree), dimension), dtype=np.float64
    )
    for axis in range(dimension):
        lower = powers.copy()
        lower[axis] -= 1
        expected_gradient[:, 0, axis] = powers[axis] * np.prod(
            np.asarray(probes) ** lower[None, :], axis=-1
        )
    np.testing.assert_allclose(
        np.einsum("qbca,b->qca", np.asarray(gradient), np.asarray(coefficients)),
        expected_gradient,
        atol=2e-10,
        rtol=2e-10,
    )

"""Closed Whitney masses and topology-exact sparse differential contracts."""

from itertools import combinations, permutations, product
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from scipy.linalg import eigvalsh
from scipy.sparse import bmat, coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem._de_rham import FiniteElementDeRhamComplex
from phydrax.discretization.fem._form_elements import FormElementFamily
from phydrax.exterior import hodge_laplacian_eigenbasis
from phydrax.exterior._algebra import form_to_vector
from phydrax.exterior._complex import ComplexBoundary
from phydrax.linalg._complexes import (
    harmonic_subspace,
    hodge_laplacian,
    HodgeLaplacianPart,
    mixed_hodge_laplacian,
)
from phydrax.linalg.eigen import EigenSolveStatus
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _generalized_eigenvalues(
    stiffness: npt.NDArray[np.float64],
    mass: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Independent SciPy generalized pencil with declared double precision."""
    stiffness_ = np.asarray(stiffness, dtype=np.float64)
    mass_ = np.asarray(mass, dtype=np.float64)
    return np.asarray(eigvalsh(stiffness_, mass_, driver="gvd"), dtype=np.float64)


def _reference_simplex(dimension: int) -> CellMesh:
    coordinates = np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), np.eye(dimension)), axis=0
    )
    return CellMesh(
        coordinates,
        (
            CellBlock(
                "simplex",
                f"simplex:{dimension}",
                np.arange(dimension + 1, dtype=np.int32)[None, :],
            ),
        ),
    )


def _whitney_mass(dimension: int, degree: int) -> npt.NDArray[np.float64]:
    gradients = np.concatenate(
        (-np.ones((1, dimension), dtype=np.float64), np.eye(dimension)), axis=0
    )
    entities = tuple(combinations(range(dimension + 1), degree + 1))
    components = tuple(combinations(range(dimension), degree))
    coefficients = np.zeros(
        (len(entities), dimension + 1, len(components)), dtype=np.float64
    )
    for entity_index, entity in enumerate(entities):
        for position, vertex in enumerate(entity):
            others = entity[:position] + entity[position + 1 :]
            for component_index, axes in enumerate(components):
                determinant = np.linalg.det(gradients[np.ix_(others, axes)])
                coefficients[entity_index, vertex, component_index] = (
                    factorial(degree) * (-1) ** position * determinant
                )
    volume = 1.0 / factorial(dimension)
    barycentric_moments = (
        volume
        * (
            np.ones((dimension + 1, dimension + 1), dtype=np.float64)
            + np.eye(dimension + 1)
        )
        / ((dimension + 1) * (dimension + 2))
    )
    return np.einsum("ivc,vw,jwc->ij", coefficients, barycentric_moments, coefficients)


@pytest.mark.parametrize("dimension", range(1, 5))
def test_lowest_order_mass_matches_closed_whitney_integrals(dimension: int) -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(dimension), family="trimmed", order=1
    )
    for degree in range(dimension + 1):
        expected = _whitney_mass(dimension, degree)
        identity = jnp.eye(expected.shape[0], dtype=jnp.float64)
        actual = np.asarray(
            jax.vmap(lambda values: complex_.hodge_star(degree, values))(identity)
        ).T
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        rhs = jnp.linspace(-0.4, 0.7, expected.shape[0], dtype=jnp.float64)
        recovered = complex_.inverse_hodge_star(degree, complex_.hodge_star(degree, rhs))
        np.testing.assert_allclose(recovered, rhs, atol=2e-10, rtol=2e-10)
    volume = 1.0 / factorial(dimension)
    np.testing.assert_allclose(
        complex_.hodge_star(dimension, jnp.asarray((volume,), dtype=jnp.float64)),
        np.asarray((1.0,), dtype=np.float64),
        atol=2e-12,
    )


@pytest.mark.parametrize("dimension", range(1, 5))
def test_lowest_order_differential_equals_signed_simplex_boundary(dimension: int) -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(dimension), family="trimmed", order=1
    )
    previous: npt.NDArray[np.float64] | None = None
    for degree in range(dimension):
        sources = tuple(combinations(range(dimension + 1), degree + 1))
        targets = tuple(combinations(range(dimension + 1), degree + 2))
        source_by_vertices = {vertices: index for index, vertices in enumerate(sources)}
        expected = np.zeros((len(targets), len(sources)), dtype=np.float64)
        for target_index, target in enumerate(targets):
            for position in range(len(target)):
                face = target[:position] + target[position + 1 :]
                expected[target_index, source_by_vertices[face]] = (-1) ** position
        identity = jnp.eye(len(sources), dtype=jnp.float64)
        actual = np.asarray(
            jax.vmap(lambda values: complex_.exterior_derivative(degree, values))(
                identity
            )
        ).T
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        if previous is not None:
            np.testing.assert_allclose(actual @ previous, 0.0, atol=1e-12)
        previous = actual


def test_hilbert_adjoint_duality_uses_metric_not_constitutive_coefficients() -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(3), family="trimmed", order=1
    )
    for degree in range(3):
        source_mass = _whitney_mass(3, degree)
        target_mass = _whitney_mass(3, degree + 1)
        source = jnp.linspace(-0.3, 0.8, source_mass.shape[0], dtype=jnp.float64)
        target = jnp.linspace(0.2, 1.1, target_mass.shape[0], dtype=jnp.float64)
        derivative = complex_.exterior_derivative(degree, source)
        adjoint = complex_.codifferential(degree + 1, target)
        left = np.asarray(derivative) @ target_mass @ np.asarray(target)
        right = np.asarray(source) @ source_mass @ np.asarray(adjoint)
        np.testing.assert_allclose(left, right, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("boundary", ("absolute", "relative"))
@pytest.mark.parametrize("degree", (0, 1))
def test_full_moment_derivative_preserves_boundary_and_metric_adjoint(
    boundary: ComplexBoundary, degree: int
) -> None:
    realization = FiniteElementDeRhamComplex(
        _simplicial_grid(2, 2), family="trimmed", order=2
    )
    source = jnp.sin(jnp.arange(realization.cell_counts[degree], dtype=jnp.float64))
    target = jnp.cos(jnp.arange(realization.cell_counts[degree + 1], dtype=jnp.float64))
    source_active = realization.active_indices(degree, boundary=boundary)
    target_active = realization.active_indices(degree + 1, boundary=boundary)
    source_projected = jnp.zeros_like(source).at[source_active].set(source[source_active])
    target_projected = jnp.zeros_like(target).at[target_active].set(target[target_active])
    derivative = realization.exterior_derivative(degree, source, boundary=boundary)
    adjoint = realization.codifferential(degree + 1, target, boundary=boundary)
    assert derivative.shape == target.shape
    assert adjoint.shape == source.shape
    np.testing.assert_allclose(
        jnp.vdot(derivative, realization.hodge_star(degree + 1, target_projected)),
        jnp.vdot(source_projected, realization.hodge_star(degree, adjoint)),
        atol=2e-10,
        rtol=2e-10,
    )
    if boundary == "relative":
        np.testing.assert_array_equal(
            derivative[realization.boundary_masks[degree + 1]], 0.0
        )
        np.testing.assert_array_equal(adjoint[realization.boundary_masks[degree]], 0.0)
    if degree == 0:
        np.testing.assert_allclose(
            realization.exterior_derivative(1, derivative, boundary=boundary),
            0.0,
            atol=2e-12,
        )


@pytest.mark.parametrize("boundary", ("absolute", "relative"))
@pytest.mark.parametrize("part", ("lower", "upper", "complete"))
def test_full_moment_laplacian_preserves_boundary_and_metric_energy(
    boundary: ComplexBoundary, part: HodgeLaplacianPart
) -> None:
    realization = FiniteElementDeRhamComplex(
        _simplicial_grid(2, 1), family="trimmed", order=2
    )
    phase = jnp.arange(realization.cell_counts[1], dtype=jnp.float64)
    source = jnp.sin(phase) + 1j * jnp.cos(phase)
    active = realization.active_indices(1, boundary=boundary)
    projected = jnp.zeros_like(source).at[active].set(source[active])
    result = realization.hodge_laplacian(1, source, boundary=boundary, part=part)
    assert result.shape == source.shape
    energy = jnp.asarray(0.0, dtype=jnp.complex128)
    if part != "lower":
        derivative = realization.exterior_derivative(1, source, boundary=boundary)
        energy = energy + jnp.vdot(derivative, realization.hodge_star(2, derivative))
    if part != "upper":
        adjoint = realization.codifferential(1, source, boundary=boundary)
        energy = energy + jnp.vdot(adjoint, realization.hodge_star(0, adjoint))
    np.testing.assert_allclose(
        jnp.vdot(projected, realization.hodge_star(1, result)),
        energy,
        atol=2e-9,
        rtol=2e-10,
    )
    if boundary == "relative":
        np.testing.assert_array_equal(result[realization.boundary_masks[1]], 0.0)


def _cube_mesh(subdivisions: int) -> CellMesh:
    axis = np.linspace(0.0, 1.0, subdivisions + 1, dtype=np.float64)
    coordinates = np.asarray(tuple(product(axis, repeat=3)), dtype=np.float64)
    width = subdivisions + 1

    def vertex(first: int, second: int, third: int) -> int:
        return (first * width + second) * width + third

    corners = (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    )
    cells = np.asarray(
        [
            [vertex(first + dx, second + dy, third + dz) for dx, dy, dz in corners]
            for first, second, third in product(range(subdivisions), repeat=3)
        ],
        dtype=np.int32,
    )
    return CellMesh(coordinates, (CellBlock("cube", "hexahedron", cells),))


def _tensor_pec_eigenvalues(subdivisions: int) -> npt.NDArray[np.float64]:
    """Independent tensor-product dispersion, including all multiplicities."""
    modes = np.arange(subdivisions, dtype=np.float64)
    phase = modes * np.pi / subdivisions
    eigenvalues = 6.0 * subdivisions**2 * (1.0 - np.cos(phase)) / (2.0 + np.cos(phase))
    spectrum: list[float] = []
    for first, second, third in product(range(subdivisions), repeat=3):
        nonzero = (first != 0) + (second != 0) + (third != 0)
        if nonzero < 2:
            continue
        value = float(eigenvalues[first] + eigenvalues[second] + eigenvalues[third])
        spectrum.extend((value,) * (nonzero - 1))
    return np.sort(np.asarray(spectrum, dtype=np.float64))


def test_pec_cube_spectrum_has_analytic_multiplicities_without_spurious_modes() -> None:
    continuum = np.pi**2 * np.asarray(
        (2.0,) * 3 + (3.0,) * 2 + (5.0,) * 6 + (6.0,) * 6,
        dtype=np.float64,
    )
    errors = []
    for subdivisions in (3, 6):
        complex_ = FiniteElementDeRhamComplex(
            _cube_mesh(subdivisions), family="tensor-trimmed", order=1
        )
        relative = complex_.hilbert_complex(boundary="relative")
        source = relative.space(1)
        target = relative.space(2)
        identity = jnp.eye(source.size, dtype=jnp.float64)
        differential = np.asarray(jax.vmap(relative.differential(1).mv)(identity)).T
        mass = np.asarray(jax.vmap(source.riesz)(identity)).T
        target_identity = jnp.eye(target.size, dtype=jnp.float64)
        target_mass = np.asarray(jax.vmap(target.riesz)(target_identity)).T
        stiffness = differential.T @ target_mass @ differential
        spectrum = _generalized_eigenvalues(stiffness, mass)
        nullity = np.count_nonzero(np.abs(spectrum) < 1e-7)
        assert nullity == relative.space(0).size == (subdivisions - 1) ** 3
        positive = spectrum[spectrum > 1e-7]
        expected = _tensor_pec_eigenvalues(subdivisions)
        np.testing.assert_allclose(positive, expected, atol=2e-8, rtol=2e-10)
        errors.append(float(np.max(np.abs(positive[:17] / continuum - 1.0))))
    assert errors[1] < 0.2
    assert errors[1] < 0.4 * errors[0]


def _simplicial_grid(dimension: int, subdivisions: int) -> CellMesh:
    coordinates = np.asarray(
        tuple(product(np.linspace(0.0, 1.0, subdivisions + 1), repeat=dimension)),
        dtype=np.float64,
    )
    width = subdivisions + 1
    cells: list[tuple[int, ...]] = []
    for lower in product(range(subdivisions), repeat=dimension):
        for order in permutations(range(dimension)):
            corner = np.asarray(lower, dtype=np.int32)
            vertices = [int(np.ravel_multi_index(tuple(corner), (width,) * dimension))]
            for axis in order:
                corner = corner.copy()
                corner[axis] += 1
                vertices.append(
                    int(np.ravel_multi_index(tuple(corner), (width,) * dimension))
                )
            jacobian = (coordinates[vertices[1:]] - coordinates[vertices[0]]).T
            if np.linalg.det(jacobian) < 0.0:
                vertices[-1], vertices[-2] = vertices[-2], vertices[-1]
            cells.append(tuple(vertices))
    return CellMesh(
        coordinates,
        (
            CellBlock(
                "simplices", f"simplex:{dimension}", np.asarray(cells, dtype=np.int32)
            ),
        ),
    )


def _integration_sites(
    mesh: CellMesh,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dimension = mesh.coordinates.shape[1]
    nodes, weights = np.polynomial.legendre.leggauss(5)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    reference: list[list[float]] = []
    reference_weights: list[float] = []
    for indices in product(range(len(nodes)), repeat=dimension):
        remaining = 1.0
        point = []
        weight = 1.0
        for axis, index in enumerate(indices):
            point.append(remaining * float(nodes[index]))
            weight *= float(
                weights[index] * (1.0 - nodes[index]) ** (dimension - axis - 1)
            )
            remaining *= 1.0 - float(nodes[index])
        reference.append(point)
        reference_weights.append(weight)
    points = np.asarray(reference, dtype=np.float64)
    quadrature_weights = np.asarray(reference_weights, dtype=np.float64)
    coordinates = np.asarray(mesh.coordinates)
    sites = []
    physical_weights = []
    for cell in np.asarray(mesh.blocks[0].vertices, dtype=np.int32):
        corners = coordinates[cell]
        jacobian = (corners[1:] - corners[0]).T
        sites.append(corners[0] + points @ jacobian.T)
        physical_weights.append(abs(np.linalg.det(jacobian)) * quadrature_weights)
    return np.concatenate(sites), np.concatenate(physical_weights)


def _manufactured_components(points: jax.Array, degree: int) -> jax.Array:
    """Absolute-boundary Hodge eigenform with eigenvalue n*pi^2."""
    dimension = points.shape[1]
    components = []
    for axes in combinations(range(dimension), degree):
        value = jnp.ones((points.shape[0],), dtype=points.dtype)
        for axis in range(dimension):
            factor = (
                jnp.sin(jnp.pi * points[:, axis])
                if axis in axes
                else jnp.cos(jnp.pi * points[:, axis])
            )
            value = value * factor
        components.append(value)
    return jnp.stack(components, axis=-1)


def _reference_metric(
    complex_: FiniteElementDeRhamComplex, degree: int
) -> csc_matrix[np.float64]:
    hodge = complex_.hodges[degree]
    rows = np.asarray(hodge.rows, dtype=np.int32)
    columns = np.asarray(hodge.columns, dtype=np.int32)
    values = np.asarray(hodge.upper_values)
    off_diagonal = rows != columns
    return coo_matrix(
        (
            np.concatenate((values, values[off_diagonal])),
            (
                np.concatenate((rows, columns[off_diagonal])),
                np.concatenate((columns, rows[off_diagonal])),
            ),
        ),
        shape=(hodge.size, hodge.size),
        dtype=np.float64,
    ).tocsc()


def _reference_differential(
    complex_: FiniteElementDeRhamComplex, degree: int
) -> csc_matrix[np.float64]:
    operator = complex_.hilbert_complex().differential(degree)
    if not isinstance(operator, SparseCoordinateOperator):
        raise AssertionError("FE differential must retain its sparse coordinate routes.")
    relation = operator.relation
    if not isinstance(relation, EdgeRelation):
        raise AssertionError("FE differential must declare canonical edge routes.")
    valid = np.asarray(relation.valid)
    return coo_matrix(
        (
            np.asarray(operator.coefficients)[valid],
            (
                np.asarray(relation.target_indices)[valid],
                np.asarray(relation.source_indices)[valid],
            ),
        ),
        shape=(operator.target.size, operator.source.size),
        dtype=np.float64,
    ).tocsc()


def _screened_mixed_error(complex_: FiniteElementDeRhamComplex, degree: int) -> float:
    mesh = complex_.mesh
    dimension = complex_.dimension
    mass = _reference_metric(complex_, degree)
    weak = mass.copy()
    if degree < dimension:
        derivative = _reference_differential(complex_, degree)
        following_mass = _reference_metric(complex_, degree + 1)
        upper = csc_matrix(derivative.T @ following_mass @ derivative, dtype=np.float64)
        weak = csc_matrix(weak + upper, dtype=np.float64)
    sites, weights = _integration_sites(mesh)
    reconstruction = complex_.reconstruction(degree)
    specification = reconstruction.value_port.form
    if specification is None:
        raise AssertionError(
            "Form reconstruction must declare its physical value specification."
        )
    exact = form_to_vector(
        _manufactured_components(jnp.asarray(sites), degree), specification
    )
    query = reconstruction.prepare_query(sites)
    quadrature_scale = jnp.asarray(weights).reshape((-1,) + (1,) * (exact.ndim - 1))
    load = query.transpose(quadrature_scale * (1.0 + dimension * np.pi**2) * exact)
    if degree > 0:
        previous_mass = _reference_metric(complex_, degree - 1)
        previous_derivative = _reference_differential(complex_, degree - 1)
        coupling = csc_matrix(mass @ previous_derivative, dtype=np.float64)
        previous_count = complex_.cell_counts[degree - 1]
        count = complex_.cell_counts[degree]
        if (
            previous_mass.shape != (previous_count, previous_count)
            or coupling.shape != (count, previous_count)
            or weak.shape != (count, count)
        ):
            raise AssertionError(
                "Mixed FE reference blocks must have their declared rank-two shapes."
            )
        # Solve the original mixed system rather than densifying its Schur
        # complement. Eliminating sigma gives M D M_prev^-1 D.T M + weak.
        blocks: list[list[csc_matrix[np.float64]]] = [
            [
                csc_matrix(-previous_mass, dtype=np.float64),
                csc_matrix(coupling.T, dtype=np.float64),
            ],
            [coupling, csc_matrix(weak, dtype=np.float64)],
        ]
        mixed = csc_matrix(bmat(blocks, format="csc"), dtype=np.float64)
        rhs = np.concatenate(
            (np.zeros(previous_mass.shape[0], dtype=np.float64), np.asarray(load))
        )
        coefficients = spsolve(mixed, rhs)[previous_mass.shape[0] :]
    else:
        coefficients = spsolve(weak, np.asarray(load))
    values = np.asarray(query.apply(coefficients))
    point_error = np.sum(
        (values - np.asarray(exact)).reshape((len(sites), -1)) ** 2, axis=1
    )
    return float(np.sqrt(np.sum(weights * point_error)))


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("order", (1, 2))
def test_mixed_hodge_laplace_manufactured_solution_converges(
    dimension: int, order: int
) -> None:
    # Independent barycentric P1/P2 Ritz assembly reproduces the h=1 errors:
    # the n*pi^2 mode is preasymptotic there. Use h=1/2 -> h=1/4 without
    # changing the manufactured PDE, quadrature, norm, or requested rate.
    coarse_complex = FiniteElementDeRhamComplex(
        _simplicial_grid(dimension, 2), family="trimmed", order=order
    )
    fine_complex = FiniteElementDeRhamComplex(
        _simplicial_grid(dimension, 4), family="trimmed", order=order
    )
    for degree in range(dimension + 1):
        coarse = _screened_mixed_error(coarse_complex, degree)
        fine = _screened_mixed_error(fine_complex, degree)
        expected_order = order + 1 if degree == 0 else order
        assert fine < 2.0 ** (-0.65 * expected_order) * coarse


@pytest.mark.parametrize("dimension", range(1, 5))
@pytest.mark.parametrize("family", ("trimmed", "full"))
def test_canonical_interpolant_commutes_with_polynomial_exterior_derivative(
    dimension: int, family: FormElementFamily
) -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(dimension), family=family, order=2
    )
    for degree in range(dimension):
        source_components = tuple(combinations(range(dimension), degree))
        target_components = tuple(combinations(range(dimension), degree + 1))
        axis_coefficients = np.arange(1, dimension + 1, dtype=np.float64)
        component_coefficients = np.arange(
            1, len(source_components) + 1, dtype=np.float64
        )
        derivative_components = np.zeros((len(target_components),), dtype=np.float64)
        component_by_axes = {axes: index for index, axes in enumerate(source_components)}
        for target_index, axes in enumerate(target_components):
            for position, axis in enumerate(axes):
                source_axes = axes[:position] + axes[position + 1 :]
                derivative_components[target_index] += (
                    (-1) ** position
                    * axis_coefficients[axis]
                    * component_coefficients[component_by_axes[source_axes]]
                )

        def polynomial(points: jax.Array) -> jax.Array:
            scalar = points @ jnp.asarray(axis_coefficients)
            return (1.0 + scalar[:, None]) * jnp.asarray(component_coefficients)[None, :]

        def derivative(points: jax.Array) -> jax.Array:
            return jnp.broadcast_to(
                jnp.asarray(derivative_components),
                (points.shape[0], len(target_components)),
            )

        interpolated = complex_.interpolant(degree, polynomial)
        expected = complex_.interpolant(degree + 1, derivative)
        np.testing.assert_allclose(
            complex_.exterior_derivative(degree, interpolated.values),
            expected.values,
            atol=2e-10,
            rtol=2e-10,
        )


def test_trace_patch_identity_rejects_equal_dimensional_cross_patch_artifacts() -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(3), family="trimmed", order=1
    )
    masks = np.eye(4, dtype=np.bool_)
    first = complex_.trace_complex_map(boundary_mask=masks[0])
    second = complex_.trace_complex_map(boundary_mask=masks[1])
    repeated = complex_.trace_complex_map(boundary_mask=masks[0].tolist())
    assert tuple(space.size for space in first.target.spaces) == (3, 3, 1)
    assert tuple(space.size for space in second.target.spaces) == (3, 3, 1)
    assert first.map_id != second.map_id
    assert first.target.complex_id != second.target.complex_id
    assert first.map_id == repeated.map_id
    assert first.target.complex_id == repeated.target.complex_id
    for degree in range(3):
        space = first.target.space(degree)
        assert not space.compatible(second.target.space(degree))
        assert space.compatible(repeated.target.space(degree))
        assert first.map(degree).operator_id != second.map(degree).operator_id
        assert first.map(degree).operator_id == repeated.map(degree).operator_id
        with pytest.raises(ValueError, match="Incompatible vector spaces"):
            hodge_laplacian(second.target, degree) @ first.map(degree)
    for degree in range(2):
        assert (
            first.target.differential(degree).operator_id
            != second.target.differential(degree).operator_id
        )
        assert (
            first.target.differential(degree).operator_id
            == repeated.target.differential(degree).operator_id
        )
    harmonic = harmonic_subspace(first.target, 0, expected_dimension=1)
    assert bool(harmonic.valid)
    with pytest.raises(ValueError, match="Harmonic basis provenance/degree"):
        mixed_hodge_laplacian(second.target, 0, harmonic=harmonic)


@pytest.mark.parametrize("order", (1, 2))
def test_single_face_traces_commute_and_preserve_metric_adjoint_duality(
    order: int,
) -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(3), family="trimmed", order=order
    )
    for mask in np.eye(4, dtype=np.bool_):
        trace = complex_.trace_complex_map(boundary_mask=mask)
        adjoint = trace.adjoint()
        for degree in range(3):
            source_space = trace.source.space(degree)
            target_space = trace.target.space(degree)
            source = jnp.sin(
                0.71 * jnp.arange(source_space.size, dtype=jnp.float64) + 0.13
            )
            target = jnp.cos(
                0.43 * jnp.arange(target_space.size, dtype=jnp.float64) - 0.21
            )
            np.testing.assert_allclose(
                target_space.inner(trace.map(degree).mv(source), target),
                source_space.inner(source, adjoint.map(degree).mv(target)),
                atol=2e-10,
                rtol=2e-10,
            )
            if degree < 2:
                identity = jnp.eye(source_space.size, dtype=jnp.float64)
                left = (trace.target.differential(degree) @ trace.map(degree)).mv_block(
                    identity
                )
                right = (
                    trace.map(degree + 1) @ trace.source.differential(degree)
                ).mv_block(identity)
                np.testing.assert_allclose(left, right, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("order", (1, 2))
@pytest.mark.parametrize("boundary", ("absolute", "relative"))
def test_contractible_simplex_harmonic_dimension_respects_boundary(
    dimension: int, order: int, boundary: ComplexBoundary
) -> None:
    complex_ = FiniteElementDeRhamComplex(
        _reference_simplex(dimension), family="trimmed", order=order
    )
    hilbert = complex_.hilbert_complex(boundary=boundary)
    for degree in range(dimension + 1):
        space = hilbert.space(degree)
        expected = int(
            (boundary == "absolute" and degree == 0)
            or (boundary == "relative" and degree == dimension)
        )
        if space.size == 0:
            assert expected == 0
            continue
        identity = jnp.eye(space.size, dtype=jnp.float64)
        mass = np.asarray(jax.vmap(space.riesz)(identity)).T
        operator = hodge_laplacian(hilbert, degree)
        laplacian = np.asarray(jax.vmap(operator.mv)(identity)).T
        weak = mass @ laplacian
        np.testing.assert_allclose(weak, weak.T, atol=2e-9)
        spectrum = _generalized_eigenvalues(weak, mass)
        assert np.count_nonzero(np.abs(spectrum) < 1e-7) == expected
        assert np.min(spectrum) >= -1e-7


@pytest.mark.parametrize("dimension", (2, 3))
@pytest.mark.parametrize("family", ("trimmed", "full"))
def test_order_transfer_reproduces_fields_and_commutes_with_d(
    dimension: int, family: FormElementFamily
) -> None:
    mesh = _reference_simplex(dimension)
    source = FiniteElementDeRhamComplex(mesh, family=family, order=1)
    target = FiniteElementDeRhamComplex(mesh, family=family, order=2)
    transfer = source.transfer(target)
    sites, _ = _integration_sites(mesh)
    for degree in range(dimension + 1):
        count = source.hilbert_complex().space(degree).size
        coefficients = jnp.sin(0.71 * jnp.arange(count, dtype=jnp.float64) + 0.13)
        transferred = transfer.map(degree).mv(coefficients)
        source_values = (
            source.reconstruction(degree).prepare_query(sites).apply(coefficients)
        )
        target_values = (
            target.reconstruction(degree).prepare_query(sites).apply(transferred)
        )
        np.testing.assert_allclose(target_values, source_values, atol=2e-9, rtol=2e-9)
        if degree < dimension:
            left = target.exterior_derivative(degree, transferred)
            right = transfer.map(degree + 1).mv(
                source.exterior_derivative(degree, coefficients)
            )
            np.testing.assert_allclose(left, right, atol=2e-9, rtol=2e-9)


def _star_refinement(
    mesh: CellMesh,
) -> tuple[CellMesh, npt.NDArray[np.int32]]:
    """An explicit, nonoverlapping centroid-star simplex subdivision."""
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    source_cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    dimension = coordinates.shape[1]
    if source_cells.ndim != 2 or source_cells.shape[1] != dimension + 1:
        raise ValueError("Star refinement requires full simplex cell connectivity.")
    centers = np.mean(coordinates[source_cells], axis=1)
    refined_coordinates = np.concatenate((coordinates, centers))
    children: list[tuple[int, ...]] = []
    parents: list[int] = []
    for parent in range(source_cells.shape[0]):
        for opposite in range(dimension + 1):
            child = [
                int(source_cells[parent, local_vertex])
                for local_vertex in range(dimension + 1)
            ]
            child[opposite] = coordinates.shape[0] + parent
            jacobian = (refined_coordinates[child[1:]] - refined_coordinates[child[0]]).T
            if np.linalg.det(jacobian) < 0.0:
                child[-1], child[-2] = child[-2], child[-1]
            children.append(tuple(child))
            parents.append(parent)
    target = CellMesh(
        refined_coordinates,
        (
            CellBlock(
                "children", f"simplex:{dimension}", np.asarray(children, dtype=np.int32)
            ),
        ),
    )
    return target, np.asarray(parents, dtype=np.int32)


@pytest.mark.parametrize("dimension", (2, 3, 4))
def test_explicit_nested_refinement_transfer_commutes_with_d(dimension: int) -> None:
    mesh = _simplicial_grid(2, 2) if dimension == 2 else _reference_simplex(dimension)
    refined, parents = _star_refinement(mesh)
    source = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    target = FiniteElementDeRhamComplex(refined, family="trimmed", order=2)
    transfer = source.transfer(target, parent_cells=parents)
    for degree in range(dimension):
        count = source.hilbert_complex().space(degree).size
        coefficients = jnp.sin(0.39 * jnp.arange(count, dtype=jnp.float64) + 0.27)
        left = target.exterior_derivative(degree, transfer.map(degree).mv(coefficients))
        right = transfer.map(degree + 1).mv(
            source.exterior_derivative(degree, coefficients)
        )
        np.testing.assert_allclose(left, right, atol=3e-9, rtol=3e-9)
    with pytest.raises(ValueError):
        source.transfer(target)


def test_whole_mesh_reconstruction_crosses_homogeneous_block_boundaries() -> None:
    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)), dtype=np.float64),
        (
            CellBlock(
                "lower",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((0,), dtype=np.int64),
            ),
            CellBlock(
                "upper",
                "triangle",
                np.asarray(((1, 3, 2),), dtype=np.int32),
                global_ids=np.asarray((1,), dtype=np.int64),
            ),
        ),
    )
    complex_ = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)

    def potential(points: jax.Array) -> jax.Array:
        return (points[:, 0] + 2.0 * points[:, 1])[:, None]

    scalar = complex_.interpolant(0, potential).values
    gradient = complex_.exterior_derivative(0, scalar)
    points = np.asarray(((0.1, 0.2), (0.8, 0.8)), dtype=np.float64)
    query = complex_.reconstruction(1).prepare_query(points)
    np.testing.assert_allclose(
        query.apply(gradient), ((1.0, 2.0), (1.0, 2.0)), atol=1e-12
    )
    dual = jnp.asarray(((0.3, -0.7), (-0.4, 0.9)), dtype=jnp.float64)
    np.testing.assert_allclose(
        jnp.vdot(query.apply(gradient), dual),
        jnp.vdot(gradient, query.transpose(dual)),
        atol=1e-12,
    )


@pytest.mark.parametrize("boundary", ("absolute", "relative"))
def test_order_two_scalar_spectrum_lifts_realization_dofs(
    boundary: ComplexBoundary,
) -> None:
    mesh = _reference_simplex(2) if boundary == "absolute" else _simplicial_grid(2, 1)
    realization = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    basis = hodge_laplacian_eigenbasis(realization, 0, boundary=boundary)
    full_count = 6 if boundary == "absolute" else 9
    active_count = full_count if boundary == "absolute" else 1
    assert basis.synthesis.shape == (full_count, active_count)
    active = np.asarray(basis.active_mask)
    expected_active = (
        np.ones(full_count, dtype=np.bool_)
        if boundary == "absolute"
        else ~np.asarray(realization.boundary_masks[0])
    )
    np.testing.assert_array_equal(active, expected_active)
    hilbert = realization.hilbert_complex(boundary=boundary)
    space = hilbert.space(0)
    identity = jnp.eye(active_count, dtype=jnp.float64)
    mass = np.asarray(jax.vmap(space.riesz)(identity)).T
    laplacian = np.asarray(jax.vmap(hodge_laplacian(hilbert, 0).mv)(identity)).T
    expected = _generalized_eigenvalues(mass @ laplacian, mass)
    np.testing.assert_allclose(basis.eigenvalues, expected, atol=2e-8, rtol=2e-8)
    synthesis = np.asarray(basis.synthesis)
    normalized_mass = mass / np.trace(mass)
    lifted_metric = np.zeros((full_count, full_count))
    lifted_metric[np.ix_(active, active)] = normalized_mass
    np.testing.assert_allclose(
        basis.analysis, synthesis.T @ lifted_metric, atol=2e-9, rtol=2e-9
    )
    np.testing.assert_allclose(
        basis.analysis @ basis.synthesis, np.eye(active_count), atol=2e-9
    )
    np.testing.assert_allclose(
        basis.synthesis @ basis.analysis, np.diag(active.astype(float)), atol=2e-9
    )
    probe = jnp.arange(full_count, dtype=jnp.float64) + 0.25
    np.testing.assert_allclose(
        basis.analysis_metric.mv(probe), lifted_metric @ probe, atol=2e-9
    )
    assert basis.eigen_solve is not None
    assert int(basis.eigen_solve.status) == int(EigenSolveStatus.SUCCESS)
    assert bool(jnp.all(basis.eigen_solve.diagnostics.converged))
    assert basis.report is not None
    assert basis.report.active_dimension == active_count
    assert basis.report.orthonormality_residual < 2e-9
    assert basis.decomposition_id == (
        f"hodge:{realization.realization_id}:0:complete:{boundary}:rank={active_count}"
    )

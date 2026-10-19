"""Independent dimensions, duality, and polynomial differential contracts."""

from collections.abc import Callable, Iterator
from fractions import Fraction
from itertools import combinations, permutations
from math import comb

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pytest
from jax import Array

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    form_element,
    FormElementFamily,
    prepare_finite_element_cell_map,
    reference_cell_topology,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import expression_evaluate
from phydrax.discretization.fem._form_basis_types import AbstractReferenceFormBasis
from phydrax.exterior._algebra import map_reference_values
from phydrax.exterior._form_type import FormProxy, FormTwist, FormType, FormValueSpec


# Empirical reference-dual regression envelopes retain the fixture specification
# from d2dfcb941 (polynomial duality, nilpotence, permutation and tensor probes).
# Hybrid probes extend those envelopes using the independent edge moments,
# smooth-field cubature, and closed-form pyramid energy defined in this module.
REFERENCE_DUAL_TOLERANCE = 2e-9
REFERENCE_POLYNOMIAL_TOLERANCE = 2e-10
REFERENCE_DIFFERENTIAL_TOLERANCE = 2e-8
REFERENCE_RANK_TOLERANCE = 1e-8
SCALAR_PULLBACK_TOLERANCE = 1e-14
EDGE_MOMENT_TOLERANCE = 3e-13
MIXED_RECONSTRUCTION_TOLERANCE = 4e-13
SMOOTH_COMMUTING_TOLERANCE = 3e-10
PYRAMID_ENERGY_TOLERANCE = 2e-13


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
        assert basis is not None
        values, _ = basis.tabulate_components(basis.functional_points)
        functionals = np.einsum(
            "iqc,qjc->ij", np.asarray(basis.functional_weights), np.asarray(values)
        )
        np.testing.assert_allclose(
            functionals,
            np.eye(expected),
            atol=REFERENCE_DUAL_TOLERANCE,
            rtol=REFERENCE_DUAL_TOLERANCE,
        )
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
        assert source is not None
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
        assert target is not None
        differential = np.asarray(source.exterior_derivative_matrix(target))
        if previous is not None:
            np.testing.assert_allclose(
                differential @ previous, 0.0, atol=REFERENCE_DIFFERENTIAL_TOLERANCE
            )
        rank = np.linalg.matrix_rank(differential, tol=REFERENCE_RANK_TOLERANCE)
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
    np.testing.assert_allclose(mapped, density / divisor, atol=SCALAR_PULLBACK_TOLERANCE)


def test_form_element_requires_proxy_and_physical_twist_when_ambiguous() -> None:
    with pytest.raises(ValueError, match="form_element.proxy"):
        form_element("triangle", 1, 1, twist="untwisted")
    with pytest.raises(ValueError, match="form_element.twist"):
        form_element("tetrahedron", 2, 1, proxy="flux")
    with pytest.raises(ValueError, match="form_element.twist"):
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
        assert basis is not None
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
        np.testing.assert_allclose(
            action,
            expected,
            atol=REFERENCE_POLYNOMIAL_TOLERANCE,
            rtol=REFERENCE_POLYNOMIAL_TOLERANCE,
        )
        np.testing.assert_allclose(
            action @ action, np.eye(action.shape[0]), atol=REFERENCE_DUAL_TOLERANCE
        )


def _second_order_tetra_circulation_moments(
    evaluate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Independent edge barycentric and face wedge functionals."""
    vertices = np.concatenate((np.zeros((1, 3), dtype=np.float64), np.eye(3)))
    gauss, weights = np.polynomial.legendre.leggauss(6)
    t, w = (gauss + 1.0) / 2.0, weights / 2.0
    rows = []
    for first, second in combinations(range(4), 2):
        tangent = vertices[second] - vertices[first]
        values = evaluate(vertices[first] + t[:, None] * tangent)
        circulation = np.einsum("qbc,c->qb", values, tangent)
        rows.extend(
            (
                np.einsum("q,q,qb->b", w, t, circulation),
                np.einsum("q,q,qb->b", w, 1.0 - t, circulation),
            )
        )
    u, v = np.meshgrid(t, t, indexing="ij")
    qu, qv = u.ravel(), ((1.0 - u) * v).ravel()
    qw = (w[:, None] * w[None, :] * (1.0 - u)).ravel()
    for first, second, third in combinations(range(4), 3):
        a, b = vertices[second] - vertices[first], vertices[third] - vertices[first]
        values = evaluate(vertices[first] + qu[:, None] * a + qv[:, None] * b)
        rows.extend(
            (
                -np.einsum("q,qbc,c->b", qw, values, b),
                np.einsum("q,qbc,c->b", qw, values, a),
            )
        )
    return np.stack(rows)


def test_second_order_tetra_nedelec_all_base_transformations_by_functionals() -> None:
    element = form_element("tetrahedron", 1, 2, proxy="circulation")
    basis = element.form_basis
    assert basis is not None
    vertices = np.concatenate((np.zeros((1, 3), dtype=np.float64), np.eye(3)))
    for order in permutations(range(4)):
        mapped = vertices[list(order)]
        jacobian = (mapped[1:] - mapped[0]).T

        def pullback(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            values, _ = basis.tabulate_components(mapped[0] + points @ jacobian.T)
            return np.asarray(values) @ jacobian

        expected = _second_order_tetra_circulation_moments(pullback)
        np.testing.assert_allclose(
            basis.permutation_matrix(order),
            expected,
            atol=EDGE_MOMENT_TOLERANCE,
            rtol=EDGE_MOMENT_TOLERANCE,
        )


def test_mixed_tetra_hex_circulation_has_one_shared_entity_moment_identity() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "tet",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray([17], dtype=np.int64),
            ),
            CellBlock(
                "hex",
                "hexahedron",
                np.asarray(((0, 1, 4, 5, 6, 7, 8, 9),), dtype=np.int32),
                global_ids=np.asarray([23], dtype=np.int64),
            ),
        ),
    )
    field = FiniteElementFieldSpec(
        "u",
        {
            "tet": form_element("tetrahedron", 1, 2, proxy="circulation"),
            "hex": form_element(
                "hexahedron", 1, 2, family="tensor-trimmed", proxy="circulation"
            ),
        },
    )
    space = FiniteElementPlan(mesh, field).prepare()

    def affine(points: Array, args: object) -> Array:
        del args
        return points + jnp.asarray((1.0, 2.0, 3.0))

    coefficients = np.asarray(space.project("u", affine))
    sites = np.asarray(
        ((0.17, 0.0, 0.0), (0.61, 0.0, 0.0), (0.89, 0.0, 0.0)), dtype=np.float64
    )
    expected = sites + np.asarray((1.0, 2.0, 3.0))
    for element, routes, transforms, inverse_frame in zip(
        space.elements[0],
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        (np.diag((1.0, -1.0, -1.0)), np.eye(3)),
        strict=True,
    ):
        local = np.asarray(transforms)[0] @ coefficients[np.asarray(routes)[0]]
        basis, _ = element.tabulate(sites)
        reconstructed = np.einsum("qnc,n->qc", np.asarray(basis), local) @ inverse_frame
        np.testing.assert_allclose(
            reconstructed,
            expected,
            atol=MIXED_RECONSTRUCTION_TOLERANCE,
            rtol=MIXED_RECONSTRUCTION_TOLERANCE,
        )


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
    assert source is not None and target is not None
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
        atol=REFERENCE_POLYNOMIAL_TOLERANCE,
        rtol=REFERENCE_POLYNOMIAL_TOLERANCE,
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
        atol=REFERENCE_POLYNOMIAL_TOLERANCE,
        rtol=REFERENCE_POLYNOMIAL_TOLERANCE,
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
        atol=REFERENCE_POLYNOMIAL_TOLERANCE,
        rtol=REFERENCE_POLYNOMIAL_TOLERANCE,
    )


@pytest.mark.parametrize("cell", ("prism", "pyramid"))
@pytest.mark.parametrize("order", (1, 2, 3))
def test_hybrid_polynomial_physics_and_exact_exterior_sequence(
    cell: str, order: int
) -> None:
    bases = tuple(
        form_element(
            cell, degree, order, proxy="components", twist="untwisted"
        ).form_basis
        for degree in range(4)
    )
    assert all(basis is not None for basis in bases)
    probes = jnp.asarray(
        ((0.0, 0.0, 0.0), (0.2, 0.15, 0.3), (0.5, 0.5, 0.5)), dtype=jnp.float64
    )
    direction = jnp.asarray((0.13, -0.07, 0.11), dtype=jnp.float64)
    previous = None
    ranks = []
    dimensions = []
    for degree, basis in enumerate(bases):
        assert basis is not None
        dimensions.append(basis.local_dof_count)
        power = order - int(degree > 0)
        samples = np.asarray(basis.functional_points)
        field = np.zeros((len(samples), comb(3, degree)), dtype=np.float64)
        field[:, 0] = (1 + samples @ np.asarray((1.0, 2.0, 3.0))) ** power
        coefficients = basis.interpolate(field)
        values, gradient = jax.jit(lambda owner, sites: owner.tabulate_components(sites))(
            basis, probes
        )
        reconstructed = np.einsum(
            "qbc,b->qc", np.asarray(values), np.asarray(coefficients)
        )
        expected = np.zeros_like(reconstructed)
        expected[:, 0] = (1 + np.asarray(probes) @ np.asarray((1.0, 2.0, 3.0))) ** power
        np.testing.assert_allclose(
            reconstructed,
            expected,
            atol=REFERENCE_DIFFERENTIAL_TOLERANCE,
            rtol=REFERENCE_DUAL_TOLERANCE,
        )
        _, tangent = jax.jvp(
            lambda sites: basis.tabulate_components(sites)[0],
            (probes,),
            (jnp.broadcast_to(direction, probes.shape),),
        )
        np.testing.assert_allclose(
            tangent,
            np.einsum("qbca,a->qbc", np.asarray(gradient), np.asarray(direction)),
            atol=REFERENCE_DIFFERENTIAL_TOLERANCE,
            rtol=REFERENCE_DUAL_TOLERANCE,
        )
        if degree == 3:
            continue
        target = bases[degree + 1]
        assert target is not None
        derivative = np.asarray(basis.exterior_derivative_matrix(target))
        if previous is not None:
            np.testing.assert_allclose(
                derivative @ previous, 0, atol=REFERENCE_DIFFERENTIAL_TOLERANCE
            )
        previous = derivative
        ranks.append(np.linalg.matrix_rank(derivative, tol=REFERENCE_RANK_TOLERANCE))
        target_values, _ = target.tabulate_components(probes)
        actual = np.einsum(
            "qbc,b->qc", np.asarray(target_values), derivative @ np.asarray(coefficients)
        )
        expected_d = np.zeros_like(actual)
        blades = tuple(combinations(range(3), degree + 1))
        if power:
            density = power * (1 + np.asarray(probes) @ np.asarray((1.0, 2.0, 3.0))) ** (
                power - 1
            )
            for axis in range(degree, 3):
                expected_d[:, blades.index((*range(degree), axis))] = (
                    (-1) ** degree * (axis + 1) * density
                )
        np.testing.assert_allclose(
            actual,
            expected_d,
            atol=REFERENCE_DIFFERENTIAL_TOLERANCE,
            rtol=REFERENCE_DUAL_TOLERANCE,
        )
    assert ranks == [dimensions[0] - 1, dimensions[1] - dimensions[0] + 1, dimensions[3]]


@pytest.mark.parametrize("cell", ("prism", "pyramid"))
@pytest.mark.parametrize("order", (1, 2, 3))
def test_hybrid_circulation_duals_have_independent_edge_moments(
    cell: str, order: int
) -> None:

    element = form_element(cell, 1, order, proxy="circulation")
    basis = element.form_basis
    assert basis is not None
    topology = reference_cell_topology(cell)
    vertices = np.asarray(topology.vertices)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    t, w = (nodes + 1) / 2, weights / 2
    for face, dofs in zip(topology.entities[1], element.entity_dofs[1], strict=True):
        first, second = face
        tangent = vertices[second] - vertices[first]
        values, _ = basis.tabulate_components(vertices[first] + t[:, None] * tangent)
        circulation = np.einsum("qbc,c->qb", np.asarray(values), tangent)
        moments = np.stack(
            tuple(
                np.einsum(
                    "q,qb->b", w * t ** (order - 1 - mode) * (1 - t) ** mode, circulation
                )
                for mode in range(order)
            )
        )
        expected = np.eye(basis.local_dof_count)[list(dofs)]
        np.testing.assert_allclose(
            moments,
            expected,
            atol=REFERENCE_DUAL_TOLERANCE,
            rtol=REFERENCE_DUAL_TOLERANCE,
        )


def _four_family_form_mesh() -> CellMesh:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.5, 0.5, 2.0),
            (0.5, -1.0, 1.25),
            (-1.0, 0.5, 0.0),
            (-1.0, 0.5, 1.0),
        ),
        dtype=np.float64,
    )
    return CellMesh(
        points,
        tuple(
            CellBlock(
                name,
                kind,
                np.asarray((vertices,), dtype=np.int32),
                global_ids=np.asarray((identity,), dtype=np.int64),
            )
            for name, kind, vertices, identity in (
                ("hex", "hexahedron", tuple(range(8)), 17),
                ("pyr", "pyramid", (4, 5, 6, 7, 8), 23),
                ("tet", "tetrahedron", (4, 5, 8, 9), 31),
                ("wedge", "prism", (0, 3, 10, 4, 7, 11), 47),
            )
        ),
    )


@pytest.mark.parametrize(("degree", "proxy"), ((1, "circulation"), (2, "flux")))
def test_four_family_shared_tri_quad_traces_reconstruct_affine_physics(
    degree: int, proxy: FormProxy
) -> None:
    mesh = _four_family_form_mesh()
    field = FiniteElementFieldSpec(
        "u",
        {
            block.name: form_element(
                block.cell_kind,
                degree,
                2,
                proxy=proxy,
                twist="untwisted",
                family="tensor-trimmed" if block.cell_kind == "hexahedron" else "trimmed",
            )
            for block in mesh.blocks
        },
    )
    space = FiniteElementPlan(mesh, field).prepare()

    def affine(points: Array, args: object) -> Array:
        del args
        return (
            points
            @ jnp.asarray(
                ((1.0, 2.0, 0.0), (0.0, 1.0, -3.0), (2.0, 0.0, 1.0)), dtype=jnp.float64
            )
            + 1
        )

    coefficients = space.project("u", affine)
    sites = jnp.asarray(((0.2, 0.15, 0.3), (0.4, 0.2, 0.6)), dtype=jnp.float64)
    for block_index, block in enumerate(mesh.blocks):
        geometry = space.evaluate_block_geometry(
            "u", block_index, mesh.coordinates, sites, jnp.ones((2,), dtype=jnp.float64)
        )
        actual = space.reconstruct("u", coefficients, block.name, sites)
        expected = affine(geometry.physical_points, None)
        np.testing.assert_allclose(
            actual, expected, atol=REFERENCE_DUAL_TOLERANCE, rtol=REFERENCE_DUAL_TOLERANCE
        )


def _independent_hybrid_cubature(kind: str) -> tuple[Array, Array]:
    nodes, weights = np.polynomial.legendre.leggauss(12)
    t, w = (nodes + 1) / 2, weights / 2
    if kind == "interval":
        return jnp.asarray(t[:, None]), jnp.asarray(w)
    dimension = 2 if kind in ("triangle", "quadrilateral") else 3
    grids = np.meshgrid(*((t,) * dimension), indexing="ij")
    weight_grids = np.meshgrid(*((w,) * dimension), indexing="ij")
    combined = np.prod(np.stack(weight_grids), axis=0)
    if kind in ("triangle", "prism"):
        points = (grids[0], (1 - grids[0]) * grids[1], *grids[2:])
        combined = combined * (1 - grids[0])
    elif kind == "pyramid":
        s = 1 - grids[2]
        points = (s * grids[0] + 0.5 * grids[2], s * grids[1] + 0.5 * grids[2], grids[2])
        combined = combined * s**2
    else:
        points = tuple(grids)
    return jnp.asarray(np.stack(points, axis=-1).reshape(-1, dimension)), jnp.asarray(
        combined.reshape(-1)
    )


def _independent_hybrid_projection(
    basis: AbstractReferenceFormBasis, evaluate: Callable[[Array], Array]
) -> Array:
    result = jnp.zeros((basis.local_dof_count,), dtype=jnp.float64)
    for dimension, entities in enumerate(basis.entity_vertices):
        for face in entities:
            if not any(label[0] == face for label in basis.dof_labels):
                continue
            if dimension == 0:
                points, weights = (
                    jnp.zeros((1, 0), dtype=jnp.float64),
                    jnp.ones((1,), dtype=jnp.float64),
                )
            else:
                points, weights = _independent_hybrid_cubature(basis.entity_kind(face))
            reference, densities = basis.functional_weights_at(face, points, weights)
            result = result + jnp.einsum("dqc,qc->d", densities, evaluate(reference))
    return result


@pytest.mark.parametrize("cell", ("prism", "pyramid"))
@pytest.mark.parametrize("degree", (0, 1, 2))
def test_hybrid_interpolation_commutes_for_smooth_fields_outside_its_space(
    cell: str, degree: int
) -> None:
    source = form_element(
        cell, degree, 2, proxy="components", twist="untwisted"
    ).form_basis
    target = form_element(
        cell, degree + 1, 2, proxy="components", twist="untwisted"
    ).form_basis
    assert source is not None and target is not None

    def field(points: Array) -> Array:
        result = jnp.zeros((points.shape[0], comb(3, degree)), dtype=jnp.float64)
        return result.at[:, 0].set((points @ jnp.asarray((1.0, 2.0, 3.0))) ** 3)

    def exterior(points: Array) -> Array:
        result = jnp.zeros((points.shape[0], comb(3, degree + 1)), dtype=jnp.float64)
        density = 3 * (points @ jnp.asarray((1.0, 2.0, 3.0))) ** 2
        blades = tuple(combinations(range(3), degree + 1))
        for axis in range(degree, 3):
            result = result.at[:, blades.index((*range(degree), axis))].set(
                (-1) ** degree * (axis + 1) * density
            )
        return result

    coefficients = _independent_hybrid_projection(source, field)
    differentiated = source.exterior_derivative_matrix(target) @ coefficients
    expected = _independent_hybrid_projection(target, exterior)
    np.testing.assert_allclose(
        differentiated,
        expected,
        atol=SMOOTH_COMMUTING_TOLERANCE,
        rtol=SMOOTH_COMMUTING_TOLERANCE,
    )


@pytest.mark.parametrize("cell", ("prism", "pyramid"))
@pytest.mark.parametrize("degree", (1, 2))
def test_complete_hybrid_cell_symmetries_preserve_physical_forms(
    cell: str, degree: int
) -> None:

    basis = form_element(
        cell, degree, 2, proxy="components", twist="untwisted"
    ).form_basis
    assert basis is not None
    vertices = np.asarray(reference_cell_topology(cell).vertices)
    probes = jnp.asarray(((0.2, 0.15, 0.25), (0.4, 0.3, 0.4)), dtype=jnp.float64)
    values, _ = basis.tabulate_components(probes)
    components = tuple(combinations(range(3), degree))
    charts = np.c_[vertices, np.ones((len(vertices),), dtype=np.float64)]
    count = 0
    for permutation in permutations(range(len(vertices))):
        affine = np.linalg.lstsq(charts, vertices[list(permutation)], rcond=None)[0]
        if np.max(np.abs(charts @ affine - vertices[list(permutation)])) > 1e-13:
            continue
        jacobian = affine[:-1].T
        compound = np.asarray(
            [
                [np.linalg.det(jacobian[np.ix_(source, target)]) for source in components]
                for target in components
            ]
        )
        mapped, _ = basis.tabulate_components(probes @ jacobian.T + affine[-1])
        expected = np.einsum("ce,qbe->qbc", compound, np.asarray(mapped))
        action = np.asarray(basis.permutation_matrix(tuple(permutation)))
        actual = np.einsum("qbc,bd->qdc", np.asarray(values), action)
        np.testing.assert_allclose(
            actual,
            expected,
            atol=REFERENCE_DIFFERENTIAL_TOLERANCE,
            rtol=REFERENCE_DUAL_TOLERANCE,
        )
        count += 1
    assert count == (12 if cell == "prism" else 8)


def test_pyramid_mass_energy_integrates_the_actual_collapsed_source() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (CellBlock("pyr", "pyramid", np.asarray(((0, 1, 2, 3, 4),), dtype=np.int32)),),
    )
    space = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", form_element("pyramid", 0, 1, proxy="scalar"))
    ).prepare()

    def coordinate(points: Array, args: object) -> Array:
        del args
        return points[..., 0]

    coefficients = space.project("u", coordinate)
    # Analytic integral over z/2 <= x,y <= 1-z/2, 0 <= z <= 1.
    energy = coefficients @ space.mass(coefficients)
    np.testing.assert_allclose(
        energy, 1 / 10, atol=PYRAMID_ENERGY_TOLERANCE, rtol=PYRAMID_ENERGY_TOLERANCE
    )


def test_hybrid_scientific_admission_refuses_changed_source() -> None:

    basis = form_element("pyramid", 1, 1, proxy="circulation").form_basis
    assert basis is not None and basis.hybrid_factors is not None
    changed = eqx.tree_at(
        lambda value: value.coefficients, basis, jnp.zeros_like(basis.coefficients)
    )
    with pytest.raises(ValueError, match="form_basis.source"):
        changed.component_expressions()


def test_hybrid_preparation_refuses_excessive_coefficient_work() -> None:
    with pytest.raises(ValueError, match="hybrid_form.coefficient_work"):
        form_element("pyramid", 1, 99, proxy="circulation")


def test_hybrid_element_refuses_incompatible_reference_cell() -> None:
    with pytest.raises(ValueError, match="form_element.family"):
        form_element("tetrahedron", 1, 2, family="prism-trimmed", proxy="circulation")


def test_exterior_derivative_refuses_replacing_simplex_by_hybrid_topology() -> None:
    with pytest.raises(ValueError, match="form_basis.derivative_target_topology"):
        source = form_element("tetrahedron", 0, 1, proxy="scalar").form_basis
        target = form_element("pyramid", 1, 1, proxy="circulation").form_basis
        assert source is not None and target is not None
        source.exterior_derivative_matrix(target)


@pytest.mark.parametrize("degree", (1, 2))
def test_pyramid_source_extraction_preserves_all_columns_under_frozen_storage_cap(
    degree: int,
    form_source_budget: CoordinateEnclosureBudget,
) -> None:

    basis = form_element(
        "pyramid", degree, 2, proxy="components", twist="untwisted"
    ).form_basis
    assert basis is not None
    budget = form_source_budget
    with budget.activate():
        source = basis.component_expressions(budget=budget)
    probes = jnp.asarray(((0.2, 0.15, 0.3), (0.4, 0.3, 0.4)), dtype=jnp.float64)
    expected, _ = basis.tabulate_components(probes)
    actual = np.asarray(
        tuple(
            tuple(
                tuple(
                    float(
                        expression_evaluate(
                            component,
                            tuple(Fraction(float(value)) for value in point),
                            budget=budget,
                        )
                    )
                    for component in field
                )
                for field in source
            )
            for point in np.asarray(probes)
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        actual, expected, atol=REFERENCE_DUAL_TOLERANCE, rtol=REFERENCE_DUAL_TOLERANCE
    )


@pytest.fixture
def form_source_budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=256_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=256_000_000,
        )
        with budget.activate():
            yield budget


@pytest.mark.parametrize("cell", ("prism", "pyramid"), ids=("product", "rational"))
@pytest.mark.parametrize(
    "degree", range(4), ids=("scalar", "circulation", "flux", "volume")
)
def test_hybrid_entity_density_sources_match_numerical_moments(
    cell: str, degree: int, form_source_budget: CoordinateEnclosureBudget
) -> None:
    basis = form_element(
        cell, degree, 2, proxy="components", twist="untwisted"
    ).form_basis
    assert basis is not None
    budget = form_source_budget
    for dimension, entities in enumerate(basis.entity_vertices):
        for face in entities:
            if not any(label[0] == face for label in basis.dof_labels):
                continue
            points = (
                np.zeros((1, 0), dtype=np.float64)
                if dimension == 0
                else np.asarray(
                    ((0.2,) * dimension, (0.3,) * dimension), dtype=np.float64
                )
            )
            quadrature = np.ones(points.shape[0], dtype=np.float64)
            _, expected = basis.functional_weights_at(face, points, quadrature)
            with budget.temporary_scope():
                densities = basis.functional_density_expressions(face, budget=budget)
                values = np.asarray(
                    [
                        [
                            [
                                float(
                                    expression_evaluate(
                                        component,
                                        tuple(Fraction(float(value)) for value in point),
                                        budget=budget,
                                    )
                                )
                                for component in field
                            ]
                            for point in points
                        ]
                        for field in densities
                    ],
                    dtype=np.float64,
                )
            np.testing.assert_allclose(
                values,
                expected,
                atol=REFERENCE_DUAL_TOLERANCE,
                rtol=REFERENCE_DUAL_TOLERANCE,
            )


@pytest.mark.parametrize(
    ("height", "expected_minimum", "expected_valid"),
    ((4.0, 4.0, True), (2.0**-30, 2.0**-60, True), (0.0, 0.0, False)),
    ids=("positive-metric", "small-positive-metric", "singular-metric"),
    # The small Jacobian remains above the native inverse's 1e-12 scaled
    # determinant threshold, while its Gram eigenvalue lies below eps.
)
def test_cell_map_metric_validity_preserves_strict_positive_threshold(
    height: float, expected_minimum: float, expected_valid: bool
) -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        coordinates,
        (CellBlock("tet", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)),),
    )
    field = FiniteElementFieldSpec("u", form_element("tetrahedron", 0, 1, proxy="scalar"))
    space = FiniteElementPlan(mesh, field).prepare()
    cell_map = prepare_finite_element_cell_map(space, 0)
    current = jnp.asarray(coordinates).at[3, 2].set(height)
    evaluation = cell_map.evaluate(
        current,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray(((0.25, 0.25, 0.25),), dtype=jnp.float64),
    )
    # The affine Jacobian is diag(2,3,height); all entries and squared lengths
    # are binary-representable values, so its diagonal spectrum needs no tolerance.
    np.testing.assert_array_equal(
        evaluation.minimum_metric_eigenvalue,
        np.asarray((expected_minimum,), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        evaluation.valid, np.asarray((expected_valid,), dtype=np.bool_)
    )
    assert evaluation.minimum_metric_eigenvalue.dtype == jnp.float64

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import permutations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax.linalg as la
from phydrax.linalg._complexes import coordinate_space


def _spd() -> la.OperatorProperties:
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={"positive_definite": "construction"},
    )


def _complex() -> tuple[la.HilbertComplex, la.AbstractLinearOperator]:
    scalar = la.ArraySpace((1,), dtype=jnp.complex128)
    edge = la.ArraySpace(
        (2,),
        dtype=jnp.complex128,
        pairing=la.DiagonalPairing(jnp.asarray([2.0, 7.0])),
    )
    gradient = la.DenseLinearOperator(
        jnp.asarray([[1.0j], [2.0]], dtype=jnp.complex128),
        source=scalar,
        target=edge,
    )
    complex = la.HilbertComplex((scalar, edge), (gradient,), complex_id="hx-duality")
    vector = la.ArraySpace((2,), dtype=jnp.complex128)
    interpolation = la.DenseLinearOperator(
        jnp.asarray([[1.0, 0.25j], [0.5, 1.0]], dtype=jnp.complex128),
        source=vector,
        target=edge,
    )
    return complex, interpolation


def _action() -> tuple[la.AbstractPreconditioner, Array]:
    complex, interpolation = _complex()
    edge = coordinate_space(complex.space(1))
    scalar = coordinate_space(complex.space(0))
    vector = coordinate_space(interpolation.source)
    builder = la.hiptmair_xu_preconditioner_builder(
        complex,
        1,
        vector_interpolation=interpolation,
        vector_builder=la.DiagonalPreconditioner(
            jnp.asarray([3.0, 5.0]),
            space=vector,
            positive_definite=True,
        ),
        potential_builder=la.DiagonalPreconditioner(
            jnp.asarray([4.0]),
            space=scalar,
            positive_definite=True,
        ),
        smoother=la.DiagonalPreconditioner(
            jnp.asarray([2.0, 6.0]),
            space=edge,
            positive_definite=True,
        ),
    )
    setup = la.DenseLinearOperator(
        jnp.asarray([[4.0, 1.0j], [-1.0j, 3.0]], dtype=jnp.complex128),
        source=edge,
        target=edge,
        properties=_spd(),
    )
    action = builder.prepare(setup, materialization=la.MaterializationPolicy())
    pi = np.asarray(interpolation._materialize())
    d = np.asarray(complex.differential(0)._materialize())
    expected = np.diag([0.5, 1.0 / 6.0]) + pi @ np.diag([1.0 / 3.0, 0.2]) @ pi.conj().T
    expected += d @ np.asarray([[0.25]]) @ d.conj().T
    return action, jnp.asarray(expected)


def test_hx_uses_coordinate_conjugate_transpose_not_metric_adjoint() -> None:
    action, expected = _action()
    residual = jnp.asarray([2.0 - 3.0j, -1.0 + 0.5j])
    np.testing.assert_allclose(action.apply(residual), expected @ residual, atol=1e-12)
    np.testing.assert_allclose(
        eqx.filter_jit(lambda prepared, value: prepared.apply(value))(action, residual),
        expected @ residual,
        atol=1e-12,
    )


def test_hx_full_smoother_certifies_a_positive_hermitian_correction() -> None:
    action, expected = _action()
    assert action.properties.certifies("positive_definite")
    np.testing.assert_allclose(expected, expected.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(np.asarray(expected)).min() > 0.0


def test_ads_recurses_into_degree_one_potential_correction() -> None:
    spaces = tuple(
        la.ArraySpace((2,), dtype=jnp.float64, space_id=f"ads-{k}") for k in range(3)
    )
    d0 = np.asarray([[1.0, -1.0], [1.0, -1.0]])
    d1 = np.asarray([[1.0, -1.0], [2.0, -2.0]])
    differentials = tuple(
        la.DenseLinearOperator(
            jnp.asarray(matrix), source=spaces[k], target=spaces[k + 1]
        )
        for k, matrix in enumerate((d0, d1))
    )
    complex = la.HilbertComplex(spaces, differentials, complex_id="ads-exact")
    interpolations = tuple(la.IdentityLinearOperator(spaces[k]) for k in (2, 1))
    builder = la.hiptmair_xu_preconditioner_builder(
        complex,
        2,
        vector_interpolation=interpolations,
        vector_builder=la.JacobiPreconditionerBuilder(),
        potential_builder=la.JacobiPreconditionerBuilder(),
        smoother=la.JacobiPreconditionerBuilder(),
    )
    space = coordinate_space(spaces[2])
    a = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    setup = la.DenseLinearOperator(
        jnp.asarray(a), source=space, target=space, properties=_spd()
    )
    action = builder.prepare(setup, materialization=la.MaterializationPolicy())
    potential_a = d1.T @ a @ d1
    inner = 2.0 * np.diag(1.0 / np.diag(potential_a))
    # The scalar correction vanishes under d1 d0, including its singular setup.
    expected = 2.0 * np.diag(1.0 / np.diag(a)) + d1 @ inner @ d1.T
    residual = jnp.asarray([1.0, -2.0])
    np.testing.assert_allclose(action.apply(residual), expected @ residual, atol=1e-12)


@pytest.mark.parametrize("degree", [0, 3])
def test_hx_refuses_non_auxiliary_degrees(degree: int) -> None:
    complex, interpolation = _complex()
    with pytest.raises(ValueError, match="degree"):
        la.hiptmair_xu_preconditioner_builder(
            complex,
            degree,
            vector_interpolation=interpolation,
            vector_builder=la.JacobiPreconditionerBuilder(),
            potential_builder=la.JacobiPreconditionerBuilder(),
            smoother=la.JacobiPreconditionerBuilder(),
        )


def test_high_order_native_auxiliary_inclusion_reproduces_an_affine_form() -> None:
    from phydrax.discretization import CellMesh
    from phydrax.discretization.fem import (
        FiniteElementDeRhamComplex,
        LowOrderAuxiliaryOperatorPlan,
    )

    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    high = FiniteElementDeRhamComplex(mesh, family="trimmed", order=2)
    low = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    plan = LowOrderAuxiliaryOperatorPlan.from_complex(high, 1)

    def form(points: Array) -> Array:
        # Constant plus rigid rotation is an exact lowest-order circulation form.
        return jnp.stack(
            (-points[..., 1], points[..., 0], jnp.ones_like(points[..., 0])), axis=-1
        )

    low_values = low.interpolant(1, form).values
    high_values = high.interpolant(1, form).values
    np.testing.assert_allclose(plan.anterpolation.mv(low_values), high_values, atol=1e-11)
    from phydrax.discretization.fem import low_order_auxiliary_preconditioner_builder

    high_hilbert = high.hilbert_complex()
    setup = la.assemble_sparse(
        la.stiffness_form(high_hilbert, 1) + la.mass_form(high_hilbert, 1)
    )
    setup = eqx.tree_at(lambda operator: operator.properties, setup, _spd())
    low_mass = la.mass_form(low.hilbert_complex(), 1)
    correction = low_order_auxiliary_preconditioner_builder(
        plan,
        la.DiagonalPreconditioner(
            la.assemble_diagonal(low_mass),
            space=low_mass.source,
            positive_definite=True,
        ),
        smoother=la.DiagonalPreconditioner(
            la.assemble_diagonal(setup),
            space=setup.source,
            positive_definite=True,
        ),
    ).prepare(setup, materialization=la.MaterializationPolicy())
    result = la.solve(
        la.LinearSystem(setup),
        setup.mv(high_values),
        policy=la.LinearSolvePolicy(
            la.ConjugateGradient(),
            preconditioning=la.PreconditioningPolicy(correction),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=100),
        ),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.value, high_values, rtol=1e-8, atol=1e-9)


def _refinement_iterations(subdivisions: int, degree: int) -> int:
    from phydrax.discretization import CellMesh
    from phydrax.discretization.fem import FiniteElementDeRhamComplex

    width = subdivisions + 1
    coordinates = np.asarray(
        [
            (x / subdivisions, y / subdivisions, z / subdivisions)
            for x in range(width)
            for y in range(width)
            for z in range(width)
        ],
        dtype=np.float64,
    )

    def vertex(point: tuple[int, int, int]) -> int:
        x, y, z = point
        return (x * width + y) * width + z

    cells: list[tuple[int, ...]] = []
    for x in range(subdivisions):
        for y in range(subdivisions):
            for z in range(subdivisions):
                for axes in permutations(range(3)):
                    point = [x, y, z]
                    corners = [vertex((point[0], point[1], point[2]))]
                    for axis in axes:
                        point[axis] += 1
                        corners.append(vertex((point[0], point[1], point[2])))
                    jacobian = (coordinates[corners[1:]] - coordinates[corners[0]]).T
                    if np.linalg.det(jacobian) < 0.0:
                        corners[-1], corners[-2] = corners[-2], corners[-1]
                    cells.append(tuple(corners))
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(coordinates),
        jnp.asarray(cells, dtype=jnp.int32),
    )
    realization = FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)
    complex = realization.hilbert_complex()
    space = coordinate_space(complex.space(degree))
    setup = la.assemble_sparse(
        la.stiffness_form(complex, degree) + la.mass_form(complex, degree)
    )
    # The mass shift proves strict positivity independently of the curl/div kernel.
    setup = eqx.tree_at(lambda operator: operator.properties, setup, _spd())
    pi = realization.vector_interpolation(degree)
    vector_space = coordinate_space(pi.source)
    scalar = la.assemble_sparse(la.stiffness_form(complex, 0) + la.mass_form(complex, 0))
    scalar = eqx.tree_at(lambda operator: operator.properties, scalar, _spd())
    scalar_matrix = la.materialize(scalar, la.MaterializationPolicy())
    vector_operator = la.DenseLinearOperator(
        jnp.kron(scalar_matrix, jnp.eye(3, dtype=jnp.float64)),
        source=vector_space,
        target=vector_space,
        properties=_spd(),
    )
    exact = la.DenseInversePreconditionerBuilder()
    vector_inverse = exact.prepare(
        vector_operator, materialization=la.MaterializationPolicy()
    )
    potential: la.AbstractPreconditioner | la.AbstractPreconditionerBuilder = (
        exact.prepare(scalar, materialization=la.MaterializationPolicy())
        if degree == 1
        else la.JacobiPreconditionerBuilder()
    )
    interpolations = pi if degree == 1 else (pi, realization.vector_interpolation(1))
    builder = la.hiptmair_xu_preconditioner_builder(
        complex,
        degree,
        vector_interpolation=interpolations,
        vector_builder=vector_inverse,
        potential_builder=potential,
        smoother=la.DiagonalPreconditioner(
            la.assemble_diagonal(setup),
            space=space,
            positive_definite=True,
        ),
    )
    inverse = builder.prepare(setup, materialization=la.MaterializationPolicy())
    rhs = jax.random.normal(
        jax.random.key(930 + degree), (space.size,), dtype=jnp.float64
    )
    result = la.solve(
        la.LinearSystem(setup),
        rhs,
        policy=la.LinearSolvePolicy(
            la.ConjugateGradient(),
            preconditioning=la.PreconditioningPolicy(inverse),
            tolerance=la.TolerancePolicy(relative=1e-8, absolute=1e-11, max_steps=100),
        ),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(setup.mv(result.value), rhs, rtol=1e-7, atol=1e-8)
    return int(np.asarray(result.diagnostics.iterations))


@pytest.mark.parametrize("degree", [1, 2], ids=["hcurl-hx", "hdiv-ads"])
def test_hx_cg_iterations_remain_bounded_across_two_mesh_refinements(degree: int) -> None:
    coarse = _refinement_iterations(1, degree)
    middle = _refinement_iterations(2, degree)
    fine = _refinement_iterations(4, degree)
    assert middle <= 60
    assert fine <= 60
    assert fine <= max(2 * middle, coarse + 15)


@pytest.mark.parametrize(
    "diagonal", [(1.0, 1.0), (1.0, 0.0)], ids=["positive", "zero-coordinate"]
)
def test_singular_psd_auxiliary_jacobi_requires_strictly_positive_diagonal(
    diagonal: tuple[float, float],
) -> None:
    space = la.ArraySpace((2,), dtype=jnp.float64)
    matrix = (
        jnp.asarray(((1.0, -1.0), (-1.0, 1.0)))
        if diagonal[1]
        else jnp.diag(jnp.asarray(diagonal))
    )
    operator = la.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        properties=la.OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    builder = la.JacobiPreconditionerBuilder()
    if diagonal[1]:
        correction = builder.prepare(operator, materialization=la.MaterializationPolicy())
        assert correction.properties.certifies("positive_definite")
        np.testing.assert_allclose(
            correction.apply(jnp.asarray([2.0, -3.0])), [2.0, -3.0]
        )
    else:
        with pytest.raises(eqx.EquinoxRuntimeError, match="strictly positive"):
            eqx.filter_jit(builder.prepare)(
                operator, materialization=la.MaterializationPolicy()
            )

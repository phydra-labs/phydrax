"""Consumer parity and fail-closed resource contracts for defect solves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.algebraic import SparsePolynomialSystem
from phydrax.applications.phase_field import (
    MappedInfiniteDefectPlan,
    PolynomialDefectPotential,
    RadialDefectPlan,
    solve_mapped_infinite_defect,
    solve_radial_defect,
)


jax.config.update("jax_enable_x64", True)


def _quadratic(field_count: int) -> PolynomialDefectPotential:
    powers = 2 * np.eye(field_count, dtype=np.int32)
    system = SparsePolynomialSystem.from_coo(
        tuple(f"phi{index}" for index in range(field_count)),
        ("potential",),
        np.zeros(field_count, dtype=np.int32),
        powers,
        jnp.full(field_count, 0.5, dtype=jnp.float64),
    )
    return PolynomialDefectPotential(system, "independent-quadratic")


@pytest.mark.parametrize("field_count", [1, 2])
@pytest.mark.parametrize("points", [9, 17, 33])
def test_sparse_radial_solve_matches_dense_stencil(points: int, field_count: int) -> None:
    gradient = np.eye(field_count) + 0.1 * np.ones((field_count, field_count))
    base = MappedInfiniteDefectPlan(
        _quadratic(field_count),
        gradient,
        np.zeros(field_count),
        np.zeros(field_count),
        collocation_count=17,
        residual_tolerance=1e-9,
    )
    plan = RadialDefectPlan(base, 3, 4.0, points)
    center = np.arange(1, field_count + 1, dtype=np.float64) * 0.2
    boundary = np.arange(1, field_count + 1, dtype=np.float64) * 0.3
    result = solve_radial_defect(plan, center, boundary)
    radius = np.linspace(0.0, 4.0, points)
    spacing = radius[1]
    count = points - 2
    matrix = np.zeros((count * field_count, count * field_count), dtype=np.float64)
    rhs = np.zeros((count, field_count), dtype=np.float64)
    for row in range(count):
        blocks = (
            (row - 1, -1.0 / spacing**2 + 1.0 / (spacing * radius[row + 1])),
            (row, 2.0 / spacing**2),
            (row + 1, -1.0 / spacing**2 - 1.0 / (spacing * radius[row + 1])),
        )
        target = slice(row * field_count, (row + 1) * field_count)
        for column, coefficient in blocks:
            if 0 <= column < count:
                matrix[target, column * field_count : (column + 1) * field_count] += (
                    coefficient * gradient
                )
            else:
                rhs[row] -= coefficient * gradient @ (center if column < 0 else boundary)
        matrix[target, target] += np.eye(field_count)
    expected = np.linalg.solve(matrix, rhs.reshape(-1)).reshape(count, field_count)
    np.testing.assert_allclose(result.fields[1:-1], expected, rtol=1e-8, atol=1e-9)
    assert bool(result.converged)
    assert float(result.residual_norm) <= base.residual_tolerance


@pytest.mark.parametrize("count", [17, 33, 65])
@pytest.mark.parametrize("field_count", [1, 2])
def test_mapped_zero_defect_retains_full_stability(count: int, field_count: int) -> None:
    gradient = np.eye(field_count) + 0.1 * np.ones((field_count, field_count))
    plan = MappedInfiniteDefectPlan(
        _quadratic(field_count),
        gradient,
        np.zeros(field_count),
        np.zeros(field_count),
        collocation_count=count,
    )
    result = solve_mapped_infinite_defect(plan)
    assert bool(result.evidence.converged)
    assert result.evidence.stability_eigenvalues.shape == ((count - 2) * field_count,)
    np.testing.assert_array_equal(result.fields, np.zeros((count, field_count)))
    np.testing.assert_array_equal(result.first_derivative, np.zeros((count, field_count)))
    nodes = np.cos(np.pi * np.arange(count) / (count - 1))[::-1]
    coefficients = np.asarray((-1.0) ** np.arange(count), dtype=np.float64)
    coefficients[[0, -1]] *= 2.0
    matrix = (coefficients[:, None] / coefficients[None, :]) / (
        nodes[:, None] - nodes[None, :] + np.eye(count)
    )
    matrix -= np.diag(matrix.sum(axis=1))
    factor = (1.0 - nodes**2) ** 2 / (1.0 + nodes**2)
    first = -factor[:, None] * matrix
    second = first @ first
    stability = -np.kron(second[1:-1, 1:-1], gradient) + np.eye((count - 2) * field_count)
    eigenvalues = np.linalg.eigvalsh(0.5 * (stability + stability.T))
    np.testing.assert_allclose(
        result.evidence.stability_eigenvalues, eigenvalues, rtol=1e-10, atol=1e-10
    )
    assert float(result.evidence.energy) == 0.0


def test_full_spectrum_budget_refuses_before_allocation() -> None:
    with pytest.raises(ValueError, match="full stability spectrum"):
        MappedInfiniteDefectPlan(
            _quadratic(2),
            np.eye(2),
            np.zeros(2),
            np.zeros(2),
            collocation_count=65,
            maximum_stability_dimension=64,
        )
    with pytest.raises(ValueError, match="full stability spectrum"):
        MappedInfiniteDefectPlan(
            _quadratic(1),
            np.eye(1),
            np.zeros(1),
            np.zeros(1),
            collocation_count=17,
            maximum_stability_bytes=100,
        )


def test_mapped_resource_controls_refuse_boolean_values() -> None:
    potential = _quadratic(1)
    for control in (
        "maximum_stability_dimension",
        "maximum_stability_bytes",
        "krylov_restart",
    ):
        with pytest.raises(TypeError, match="must be an integer"):
            MappedInfiniteDefectPlan(
                potential,
                np.eye(1),
                np.zeros(1),
                np.zeros(1),
                collocation_count=17,
                **{control: True},
            )


@pytest.mark.parametrize("count", [17, 33, 65])
def test_mapped_kink_matches_dense_newton(count: int) -> None:
    potential = PolynomialDefectPotential(
        SparsePolynomialSystem.from_coo(
            ("phi",),
            ("potential",),
            np.asarray([0, 0, 0], dtype=np.int32),
            np.asarray([[4], [2], [0]], dtype=np.int32),
            jnp.asarray([0.25, -0.5, 0.25], dtype=jnp.float64),
        ),
        "independent-phi4",
    )
    plan = MappedInfiniteDefectPlan(
        potential,
        np.eye(1),
        np.asarray([-1.0]),
        np.asarray([1.0]),
        collocation_count=count,
        residual_tolerance=1e-7,
    )
    nodes = np.cos(np.pi * np.arange(count) / (count - 1))[::-1]
    coefficients = np.asarray((-1.0) ** np.arange(count), dtype=np.float64)
    coefficients[[0, -1]] *= 2.0
    difference = nodes[:, None] - nodes[None, :]
    matrix = (coefficients[:, None] / coefficients[None, :]) / (
        difference + np.eye(count)
    )
    matrix -= np.diag(matrix.sum(axis=1))
    factor = (1.0 - nodes**2) ** 2 / (1.0 + nodes**2)
    first = -factor[:, None] * matrix
    second = first @ first
    fields = nodes.copy()

    def residual(value: np.ndarray) -> np.ndarray:
        return (-second @ value + value**3 - value)[1:-1]

    for _ in range(plan.maximum_iterations):
        value = residual(fields)
        norm = np.linalg.norm(value)
        if norm <= plan.residual_tolerance:
            break
        jacobian = -second[1:-1, 1:-1] + np.diag(3.0 * fields[1:-1] ** 2 - 1.0)
        update = np.linalg.solve(jacobian, -value)
        for backtrack in range(plan.backtracking_steps):
            trial = fields.copy()
            trial[1:-1] += 0.5**backtrack * update
            if np.linalg.norm(residual(trial)) < norm:
                fields = trial
                break
        else:
            break
    result = solve_mapped_infinite_defect(plan)
    np.testing.assert_allclose(result.fields[:, 0], fields, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        result.first_derivative[:, 0], first @ fields, rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(
        result.second_derivative[:, 0], second @ fields, rtol=1e-8, atol=1e-8
    )
    stability = -second[1:-1, 1:-1] + np.diag(3.0 * fields[1:-1] ** 2 - 1.0)
    eigenvalues = np.linalg.eigvalsh(0.5 * (stability + stability.T))
    np.testing.assert_allclose(
        result.evidence.stability_eigenvalues, eigenvalues, rtol=1e-8, atol=1e-8
    )
    assert int(result.evidence.negative_mode_count) == np.sum(
        eigenvalues < -np.sqrt(plan.residual_tolerance)
    )
    assert bool(result.evidence.converged) == (
        np.linalg.norm(residual(fields)) <= plan.residual_tolerance
    )
    warm_start = np.asarray(result.fields).copy()
    warm_start[1:-1, 0] += 1e-3 * nodes[1:-1] * (1.0 - nodes[1:-1] ** 2)
    repeated = solve_mapped_infinite_defect(plan, initial_fields=warm_start)
    assert bool(repeated.evidence.converged)
    np.testing.assert_allclose(repeated.fields, result.fields, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        repeated.first_derivative[:, 0],
        first @ np.asarray(repeated.fields[:, 0]),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        repeated.second_derivative[:, 0],
        second @ np.asarray(repeated.fields[:, 0]),
        rtol=1e-8,
        atol=1e-8,
    )

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


_BOUNDARY_CASES = (
    (("dirichlet", "dirichlet"), "dst"),
    (("neumann", "neumann"), "dct"),
    (("dirichlet", "neumann"), "dst"),
    (("neumann", "dirichlet"), "dct"),
)


def _grid(axis):
    return phx.discretization.TensorGridPlan(
        (axis,),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


@pytest.mark.parametrize(
    "axis",
    [
        phx.discretization.UniformCellAxisSpec(16),
        phx.discretization.UniformAxisSpec(17),
    ],
)
@pytest.mark.parametrize(("boundaries", "family"), _BOUNDARY_CASES)
def test_uniform_fd2_boundary_semantics_have_exact_fast_diagonalizations(
    axis,
    boundaries,
    family,
):
    diagonalization = phx.discretization.diagonalize_fd_laplacian(
        _grid(axis),
        {"x": boundaries},
    )
    values = jnp.linspace(-1.0, 2.0, np.prod(diagonalization.unknown_shape)).reshape(
        diagonalization.unknown_shape
    )

    transformed_action = diagonalization.transform.synthesize(
        diagonalization.modal_values * diagonalization.transform.analyze(values)
    )

    np.testing.assert_allclose(
        transformed_action,
        diagonalization.operator.mv(values),
        rtol=2e-12,
        atol=2e-11,
    )
    assert diagonalization.axis_reports[0].exact
    assert diagonalization.axis_reports[0].transform_family == family
    assert diagonalization.unknown_coordinates[0].shape == diagonalization.unknown_shape


def test_tensor_fd2_diagonalization_composes_mixed_entity_and_boundary_axes():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(12),
            phx.discretization.UniformAxisSpec(11),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, -1.0], [2.0, 1.0]]))
    diagonalization = phx.discretization.diagonalize_fd_laplacian(
        grid,
        {
            "x": ("dirichlet", "neumann"),
            "y": ("neumann", "dirichlet"),
        },
    )
    values = jnp.arange(np.prod(diagonalization.unknown_shape), dtype=float).reshape(
        diagonalization.unknown_shape
    )

    transformed_action = diagonalization.transform.synthesize(
        diagonalization.modal_values * diagonalization.transform.analyze(values)
    )

    assert diagonalization.unknown_shape == (12, 10)
    assert diagonalization.nullspace_dimension == 0
    np.testing.assert_allclose(
        transformed_action,
        diagonalization.operator.mv(values),
        rtol=3e-12,
        atol=3e-10,
    )


@pytest.mark.parametrize(
    "axis",
    [
        phx.discretization.UniformCellAxisSpec(32),
        phx.discretization.UniformAxisSpec(33),
    ],
)
def test_mixed_dirichlet_neumann_direct_solve_honors_nonzero_boundary_data(axis):
    diagonalization = phx.discretization.diagonalize_fd_laplacian(
        _grid(axis),
        {"x": ("dirichlet", "neumann")},
    )
    exact = diagonalization.unknown_coordinates[0]
    plan = phx.discretization.FDLaplacianSolvePlan(diagonalization)

    result = eqx.filter_jit(plan.solve)(
        jnp.zeros_like(exact),
        boundary_values={"x": (0.0, 1.0)},
    )

    assert bool(result.converged)
    np.testing.assert_allclose(result.value, exact, rtol=2e-12, atol=2e-12)
    assert float(result.residual_norm) < 2e-10


def test_neumann_direct_solve_projects_compatibility_and_enforces_minimum_norm():
    diagonalization = phx.discretization.diagonalize_fd_laplacian(
        _grid(phx.discretization.UniformAxisSpec(33)),
        {"x": ("neumann", "neumann")},
    )
    coordinates = diagonalization.unknown_coordinates[0]
    seed = jnp.sin(2.0 * jnp.pi * coordinates) + 0.4
    compatible_rhs = diagonalization.operator.mv(seed)
    plan = phx.discretization.FDLaplacianSolvePlan(
        diagonalization,
        compatibility="project_rhs",
        gauge="minimum_norm",
    )

    compatible = plan.solve(compatible_rhs)
    projected = plan.solve(compatible_rhs + 1.0)

    assert bool(compatible.converged)
    assert bool(projected.converged)
    np.testing.assert_allclose(jnp.mean(compatible.value), 0.0, atol=2e-12)
    np.testing.assert_allclose(jnp.mean(projected.value), 0.0, atol=2e-12)
    assert float(projected.removed_component_norm) > 1.0
    assert float(projected.residual_norm) < 2e-9


def test_neumann_direct_solve_has_finite_mathematical_derivatives():
    diagonalization = phx.discretization.diagonalize_fd_laplacian(
        _grid(phx.discretization.UniformCellAxisSpec(32)),
        {"x": ("neumann", "neumann")},
    )
    coordinates = diagonalization.unknown_coordinates[0]
    seed = jnp.sin(2.0 * jnp.pi * coordinates)
    right_hand_side = diagonalization.operator.mv(seed)
    plan = phx.discretization.FDLaplacianSolvePlan(
        diagonalization,
        compatibility="project_rhs",
        gauge="zero_mean",
    )

    def objective(rhs):
        value = plan.solve(rhs).value
        return 0.5 * jnp.vdot(value, value).real

    gradient = jax.jit(jax.grad(objective))(right_hand_side)
    solution = plan.solve(right_hand_side).value
    expected = plan.solve(solution).value

    assert jnp.all(jnp.isfinite(gradient))
    np.testing.assert_allclose(gradient, expected, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(jnp.mean(gradient), 0.0, atol=2e-12)


def test_fd_diagonalization_rejects_boundary_metadata_mismatch():
    periodic = _grid(phx.discretization.UniformCellAxisSpec(8, periodic=True))
    bounded = _grid(phx.discretization.UniformCellAxisSpec(8))

    with pytest.raises(ValueError, match="must agree"):
        phx.discretization.diagonalize_fd_laplacian(
            periodic,
            {"x": ("dirichlet", "dirichlet")},
        )
    with pytest.raises(ValueError, match="requires two boundaries"):
        phx.discretization.diagonalize_fd_laplacian(bounded, {})

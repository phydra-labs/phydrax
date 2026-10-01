#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    point_sbp_report,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointDiffusionOperator,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import LocalStencilPolicy


def _cloud(count: int = 11) -> tuple[Array, PreparedPointCloudDiscretization]:
    x = jnp.linspace(0.0, 1.0, count)
    boundary = (jnp.arange(count) == 0) | (jnp.arange(count) == count - 1)
    normals = jnp.zeros((count, 1)).at[0, 0].set(-1.0).at[-1, 0].set(1.0)
    cloud = PointCloudPlan(
        x[:, None],
        jnp.linspace(0.5, 1.5, count) / count,
        boundary_mask=boundary,
        boundary_normals=normals,
        boundary_quadrature_weights=boundary.astype(jnp.float64),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=5,
    ).prepare()
    return x, cloud


def test_collocated_variable_coefficient_sign_and_dirichlet_lift() -> None:
    x, cloud = _cloud()
    exact = 1 + x + x * x
    k = 1 + x
    # -((1+x)(1+2x))' = -3-4x.
    source = -3 - 4 * x
    prepared = PointCloudPoissonPlan(
        cloud, PointBoundaryPlan("dirichlet", exact), preconditioner="none"
    ).prepare(k)
    result = prepared.solve(source)
    np.testing.assert_allclose(result.values, exact, atol=2e-8)
    assert float(result.residual_norm) < 1e-8
    assert float(result.boundary_residual_norm) < 1e-9
    assert int(result.status) == 0
    assert bool(result.diagnostics.converged)
    assert bool(result.compatible)
    assert bool(result.successful)


def test_reused_default_preconditioner_with_changed_dirichlet_data() -> None:
    x, cloud = _cloud()
    prepared = PointCloudPoissonPlan(
        cloud, PointBoundaryPlan("dirichlet", x * x)
    ).prepare()
    source = jnp.full_like(x, -2.0)
    initial = prepared.solve(source)
    np.testing.assert_allclose(initial.values, x * x, atol=2e-8)
    shifted = prepared.solve(source, boundary_values=x * x + x + 2.0)
    np.testing.assert_allclose(shifted.values, x * x + x + 2.0, atol=2e-8)
    assert bool(shifted.successful)


def test_dissipative_weighted_energy_and_numeric_refresh() -> None:
    x, cloud = _cloud()
    value = jnp.stack((jnp.sin(3 * x), jnp.cos(5 * x)), axis=1)
    diffusion = PointDiffusionOperator(cloud, 1 + x, form="dissipative")
    gradients = cloud.partial_derivative(value, axis=0)
    expected = -jnp.sum(
        cloud.quadrature_weights[:, None] * (1 + x)[:, None] * gradients**2
    )
    np.testing.assert_allclose(diffusion.energy_rate(value), expected, atol=1e-10)
    u = jnp.sin(2 * x) + x
    boundary = PointBoundaryPlan("dirichlet", u)
    prepared = PointCloudPoissonPlan(
        cloud, boundary, form="dissipative", preconditioner="ilu"
    ).prepare(1 + x)
    source = -diffusion.mv(u)
    np.testing.assert_allclose(prepared.solve(source).values, u, atol=2e-8)
    refreshed = prepared.refresh(2 * (1 + x))
    np.testing.assert_allclose(refreshed.solve(2 * source).values, u, atol=2e-8)
    # Reusing old coefficients cannot solve the changed physical problem.
    assert float(jnp.max(jnp.abs(prepared.solve(2 * source).values - u))) > 1e-3


def test_robin_conormal_and_neumann_algebraic_compatibility() -> None:
    x, cloud = _cloud()
    u = 1 + x + x * x
    conormal = jnp.zeros_like(x).at[0].set(-1.0).at[-1].set(3.0)
    robin_values = conormal + u
    robin = PointCloudPoissonPlan(
        cloud,
        PointBoundaryPlan("robin", robin_values, robin_coefficient=1.0),
        preconditioner="none",
    ).prepare()
    np.testing.assert_allclose(robin.solve(jnp.full_like(x, -2.0)).values, u, atol=2e-8)
    neumann = PointCloudPoissonPlan(
        cloud,
        PointBoundaryPlan("neumann", conormal),
        preconditioner="none",
        compatibility="refuse",
    ).prepare()
    result = neumann.solve(jnp.full_like(x, -2.0))
    np.testing.assert_allclose(result.values, u - u[neumann.plan.gauge_index], atol=2e-8)
    assert float(result.gauge_residual) < 1e-10
    with pytest.raises(Exception, match="incompatible Neumann"):
        neumann.solve(jnp.full_like(x, -1.0))
    projected = PointCloudPoissonPlan(
        cloud,
        PointBoundaryPlan("neumann", conormal),
        preconditioner="none",
        compatibility="project",
    ).prepare()
    corrected = projected.solve(jnp.full_like(x, -1.0))
    assert not bool(corrected.compatible)
    np.testing.assert_allclose(corrected.source_correction[1:-1], -1.0, atol=2e-8)
    np.testing.assert_allclose(
        corrected.source_correction[jnp.asarray([0, -1])], 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        corrected.values, u - u[projected.plan.gauge_index], atol=2e-8
    )
    assert float(corrected.residual_norm) < 1e-8
    assert bool(corrected.successful)


def test_sbp_exact_identity_reports_non_sbp_and_refuses_budget() -> None:
    _, cloud = _cloud()
    report = point_sbp_report(cloud)
    assert not report.passed
    assert report.maximum_green_residual > 1e-4
    with pytest.raises(ValueError, match="maximum_coefficients"):
        point_sbp_report(cloud, maximum_coefficients=1)


def test_invalid_robin_and_nonpositive_diffusivity_are_refused() -> None:
    x, cloud = _cloud()
    with pytest.raises(ValueError, match="Zero-coefficient Robin"):
        PointCloudPoissonPlan(cloud, PointBoundaryPlan("robin", x, robin_coefficient=0.0))
    with pytest.raises(Exception, match="positive"):
        PointDiffusionOperator(cloud, 0.0)


def test_dirichlet_lift_does_not_weaken_original_equation_tolerance() -> None:
    from examples.point_cloud_poisson import run_workflow

    record = run_workflow(
        size=128,
        dimension=2,
        seed=0,
        manufactured="exponential",
        stencil=LocalStencilPolicy(approximation="gmls", polynomial_degree=2),
        neighbors=18,
    )
    residual = record["true_original_residual"]
    tolerance = record["residual_tolerance"]
    assert isinstance(residual, float)
    assert isinstance(tolerance, float)
    assert residual <= tolerance
    assert record["successful"] is True


def test_independent_neumann_refinement_and_reported_source_projection() -> None:
    from examples.point_cloud_poisson import run_workflow

    coarse = run_workflow(
        size=64, dimension=2, seed=0, manufactured="exponential", boundary_kind="neumann"
    )
    fine = run_workflow(
        size=128, dimension=2, seed=0, manufactured="exponential", boundary_kind="neumann"
    )
    coarse_error = coarse["maximum_solution_error"]
    fine_error = fine["maximum_solution_error"]
    coarse_correction = coarse["maximum_source_correction"]
    fine_correction = fine["maximum_source_correction"]
    assert isinstance(coarse_error, float) and isinstance(fine_error, float)
    assert isinstance(coarse_correction, float) and isinstance(fine_correction, float)
    assert fine_error < coarse_error
    assert 0.0 < fine_correction < coarse_correction
    assert fine["compatible"] is False
    assert fine["successful"] is True

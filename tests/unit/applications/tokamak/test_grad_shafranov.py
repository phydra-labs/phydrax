import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _manufactured_case(count=13):
    r = np.linspace(1.0, 3.0, count)
    z = np.linspace(-1.0, 1.0, count)
    rr, zz = np.meshgrid(r, z)
    center = 2.0
    psi = (rr - center) ** 2 + zz**2
    mu0 = phx.applications.tokamak.DEFAULT_VACUUM_PERMEABILITY_H_M
    current = np.zeros_like(psi)
    current[1:-1, 1:-1] = -(2.0 + 2.0 * center / rr[1:-1, 1:-1]) / (mu0 * rr[1:-1, 1:-1])
    boundary = np.zeros_like(psi)
    boundary[0] = psi[0]
    boundary[-1] = psi[-1]
    boundary[:, 0] = psi[:, 0]
    boundary[:, -1] = psi[:, -1]
    plan = phx.applications.tokamak.FixedBoundaryGradShafranovPlan(
        r,
        z,
        phx.applications.tokamak.TokamakMagneticConvention.canonical(),
        relative_tolerance=1.0e-11,
        maximum_steps=256,
    ).prepare()
    return plan, current, boundary, psi


def test_fixed_boundary_grad_shafranov_recovers_quadratic_solution():
    plan, current, boundary, expected = _manufactured_case()
    result = plan.solve(current, boundary)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.poloidal_flux_wb_per_rad, expected, rtol=2.0e-8, atol=2.0e-8
    )
    assert result.pde_residual_norm < 1.0e-7
    assert np.isfinite(result.total_plasma_current_a)


def test_grad_shafranov_solution_differentiates_through_current_source():
    plan, current, boundary, _ = _manufactured_case(9)
    current = jnp.asarray(current)
    boundary = jnp.asarray(boundary)

    def objective(scale):
        result = plan.solve(scale * current, scale * boundary)
        return jnp.sum(result.poloidal_flux_wb_per_rad)

    derivative = jax.grad(objective)(jnp.asarray(1.0))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0

import jax.numpy as jnp
import numpy as np

from phydrax.applications.compressible_flow import TransonicSmallDisturbancePlan


def test_linear_potential_is_exact_tsd_solution_and_pressure_is_constant():
    x = jnp.linspace(-1.0, 1.0, 7)
    y = jnp.linspace(-0.5, 0.5, 6)
    plan = TransonicSmallDisturbancePlan(x, y, 0.8)
    potential = 0.03 * x[:, None] + 0.02 * y[None, :]
    forcing = jnp.zeros_like(potential)
    residual = plan.residual(potential, forcing, potential)

    np.testing.assert_allclose(residual, 0.0, atol=2.0e-7)
    np.testing.assert_allclose(plan.pressure_coefficient(potential), -0.06, atol=2.0e-7)


def test_tsd_implicit_root_returns_certified_exact_state():
    x = jnp.linspace(-1.0, 1.0, 5)
    y = jnp.linspace(-0.5, 0.5, 5)
    plan = TransonicSmallDisturbancePlan(x, y, 0.75, residual_tolerance=1.0e-8)
    boundary = 0.01 * x[:, None] - 0.015 * y[None, :]
    result = plan.solve(boundary, jnp.zeros_like(boundary), boundary)

    assert bool(result.successful)
    np.testing.assert_allclose(result.potential, boundary, atol=2.0e-7)
    assert result.maximum_residual <= 1.0e-8
